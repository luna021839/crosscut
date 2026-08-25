"""Dedicated, fail-closed acquisition for the private IVE face gallery.

This module intentionally does not reuse the MEMBER-render portrait contract.
Landscape official FaceCam/close-up sources are useful identity evidence even
when they are not valid 9:16 render inputs.  Only caller-pinned video IDs that
are present in the pinned IVE STAGE catalogue are considered, and every run
performs a fresh single-video metadata observation before a cached byte can be
reused.

The output is process-local acquisition authority.  Raw frames and face
embeddings are not produced here and are never included in the acquisition
manifest.  A separate materializer adapter converts these authorities to its
own sealed gallery-source type immediately before YuNet/SFace sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import uuid4

from .acquisition import (
    AcquisitionCandidate,
    AcquisitionError,
    YTDLPClient,
    _FULL_HD_DOWNLOAD_PROFILE,
    _IVE_STAGE_PLAYLIST_ID,
    _IVE_STAGE_PLAYLIST_OWNER_CHANNEL_ID,
    _MEMBER_ALIASES,
    _broadcast_date,
    _contains_alias,
    _current_hard_title_rejection_reason,
    _has_ive_title_signal,
    _metadata_broadcast_date_with_basis,
    _music_show_program,
    _normalized_catalog_object,
    _official_channel_signal,
    _song_aliases,
)
from .ive import IVEMember
from .media import MediaInfo, MediaProbeError, probe_media


IVE_FACE_GALLERY_ACQUISITION_CONTRACT = (
    "ive_private_face_gallery_dedicated_acquisition_v1"
)
IVE_FACE_GALLERY_ACQUISITION_SCHEMA_VERSION = 1
IVE_FACE_GALLERY_SOURCE_RECEIPT_CONTRACT = (
    "ive_private_face_gallery_source_receipt_v1"
)
IVE_FACE_GALLERY_METADATA_OBSERVATION_CONTRACT = (
    "ive_face_gallery_live_single_video_metadata_observation_v1"
)
IVE_FACE_GALLERY_CATALOG_OBSERVATION_CONTRACT = (
    "ive_face_gallery_pinned_catalog_membership_observation_v1"
)
PRIVATE_LOCAL_ACQUISITION_POLICY = (
    "private_local_only_no_frames_or_embeddings_in_manifest_v1"
)
DEDICATED_FACECAM_SOURCE_AUTHORITY = (
    "official_target_member_closeup_facecam_any_aspect"
)
DEDICATED_FACECAM_ACQUISITION_AUTHORITY_BASIS = (
    "sealed_dedicated_ive_facecam_acquisition_v1"
)
FULL_HD_AVC_COMPATIBILITY_PROFILE = _FULL_HD_DOWNLOAD_PROFILE
PINNED_IVE_STAGE_PLAYLIST_ID = _IVE_STAGE_PLAYLIST_ID
PINNED_IVE_STAGE_OWNER_CHANNEL_ID = _IVE_STAGE_PLAYLIST_OWNER_CHANNEL_ID

_AUTHORITY_SEAL = object()
_VIDEO_ID = re.compile(r"[0-9A-Za-z_-]{6,32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MEMBERS = tuple(IVEMember)
_MAXIMUM_JSON_BYTES = 64 * 1024 * 1024
_MINIMUM_SOURCE_DURATION_SECONDS = 45.0
_MAXIMUM_SOURCE_DURATION_SECONDS = 15.0 * 60.0
_MINIMUM_SHORT_EDGE = 720
_MAXIMUM_LONG_EDGE = 1920
_MINIMUM_FRAME_RATE = 23.0
_MAXIMUM_FRAME_RATE = 120.0
_DURATION_ABSOLUTE_TOLERANCE_SECONDS = 2.0
_DURATION_RELATIVE_TOLERANCE = 0.02
_CLOSEUP_TITLE_MARKERS = (
    "facecam",
    "face cam",
    "face-cam",
    "페이스캠",
    "얼빡",
    "얼굴직캠",
    "close-up",
    "close up",
    "closeup",
    "클로즈업",
    "클로즈 업",
)
_FULLCAM_TITLE_MARKERS = (
    "fullcam",
    "full cam",
    "full-cam",
    "풀캠",
    "전체직캠",
    "전체 직캠",
)
_PRODUCTION_CORPUS_SONG_TOKENS = frozenset({"blackhole", "bangbang", "xoxz"})


class IVEFaceGalleryAcquisitionError(ValueError):
    """A pinned source, storage, cache, or integrity gate failed."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IVEFaceGalleryAcquisitionError(
                f"acquisition JSON repeats key {key!r}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise IVEFaceGalleryAcquisitionError(
        f"acquisition JSON contains non-finite constant {value!r}"
    )


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IVEFaceGalleryAcquisitionError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise IVEFaceGalleryAcquisitionError(f"{name} must be a positive integer")
    return value


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IVEFaceGalleryAcquisitionError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise IVEFaceGalleryAcquisitionError(
            f"{name} must be finite and positive"
        )
    return numeric


def _member(value: object) -> IVEMember:
    if isinstance(value, IVEMember):
        return value
    if not isinstance(value, str):
        raise IVEFaceGalleryAcquisitionError("member must be an IVE member")
    try:
        return IVEMember.parse(value)
    except ValueError as exc:
        raise IVEFaceGalleryAcquisitionError(
            "member must be one of the six IVE members"
        ) from exc


def _compact_token(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


@dataclass(frozen=True, slots=True)
class IVEFaceGalleryPinnedSourceSpec:
    """One explicit video ID and the exact song its live title must name."""

    video_id: str
    song: str

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or _VIDEO_ID.fullmatch(
            self.video_id
        ) is None:
            raise IVEFaceGalleryAcquisitionError("pinned video_id is invalid")
        if isinstance(self.song, str) and _compact_token(self.song) in {
            "attitude",
            "애티튜드",
        }:
            raise IVEFaceGalleryAcquisitionError(
                "ATTITUDE is strictly quarantined from face-gallery acquisition"
            )
        if (
            not isinstance(self.song, str)
            or not self.song.strip()
            or self.song != self.song.strip()
            or len(self.song) > 256
        ):
            raise IVEFaceGalleryAcquisitionError(
                "pinned source song must be canonical and non-empty"
            )


@dataclass(frozen=True, slots=True)
class IVEFaceGalleryMemberSourceSpec:
    """Explicit, song-bound pinned sources for exactly one IVE member."""

    member: IVEMember | str
    sources: tuple[IVEFaceGalleryPinnedSourceSpec, ...] | Sequence[
        IVEFaceGalleryPinnedSourceSpec
    ]

    def __post_init__(self) -> None:
        object.__setattr__(self, "member", _member(self.member))
        if isinstance(self.sources, (str, bytes)):
            raise IVEFaceGalleryAcquisitionError("sources must be a sequence")
        values = tuple(self.sources)
        if (
            len(values) < 2
            or any(
                not isinstance(item, IVEFaceGalleryPinnedSourceSpec)
                for item in values
            )
            or len({item.video_id for item in values}) != len(values)
        ):
            raise IVEFaceGalleryAcquisitionError(
                "each member needs at least two unique typed pinned sources"
            )
        object.__setattr__(self, "sources", values)

    @property
    def video_ids(self) -> tuple[str, ...]:
        return tuple(item.video_id for item in self.sources)


@dataclass(frozen=True, slots=True)
class IVEFaceGalleryAcquisitionSpec:
    """Six-member IVE acquisition request; three videos/member by default."""

    gallery_id: str
    members: tuple[IVEFaceGalleryMemberSourceSpec, ...] | Sequence[
        IVEFaceGalleryMemberSourceSpec
    ]
    target_videos_per_member: int = 3
    group: str = "IVE"

    def __post_init__(self) -> None:
        if self.group != "IVE":
            raise IVEFaceGalleryAcquisitionError("face gallery supports IVE only")
        if (
            not isinstance(self.gallery_id, str)
            or not self.gallery_id.strip()
            or self.gallery_id != self.gallery_id.strip()
            or len(self.gallery_id) > 256
        ):
            raise IVEFaceGalleryAcquisitionError(
                "gallery_id must be canonical and non-empty"
            )
        target = _positive_integer(
            self.target_videos_per_member, "target_videos_per_member"
        )
        if target < 2:
            raise IVEFaceGalleryAcquisitionError(
                "target_videos_per_member cannot be below two"
            )
        normalized = tuple(self.members)
        if (
            len(normalized) != len(_MEMBERS)
            or any(
                not isinstance(item, IVEFaceGalleryMemberSourceSpec)
                for item in normalized
            )
            or {item.member for item in normalized} != set(_MEMBERS)
        ):
            raise IVEFaceGalleryAcquisitionError(
                "spec must contain each of the six IVE members exactly once"
            )
        all_ids = tuple(source.video_id for item in normalized for source in item.sources)
        if len(all_ids) != len(set(all_ids)):
            raise IVEFaceGalleryAcquisitionError(
                "one pinned video ID cannot authorize multiple members"
            )
        if any(len(item.sources) > target for item in normalized):
            raise IVEFaceGalleryAcquisitionError(
                "member video count exceeds target_videos_per_member"
            )
        ordered = tuple(
            next(item for item in normalized if item.member is member)
            for member in _MEMBERS
        )
        object.__setattr__(self, "members", ordered)
        object.__setattr__(self, "target_videos_per_member", target)

    @property
    def request_payload(self) -> dict[str, Any]:
        return {
            "contract": IVE_FACE_GALLERY_ACQUISITION_CONTRACT,
            "group": self.group,
            "gallery_id": self.gallery_id,
            "target_videos_per_member": self.target_videos_per_member,
            "minimum_videos_per_member": 2,
            "members": [
                {
                    "member": item.member.value,
                    "sources": [
                        {"video_id": source.video_id, "song": source.song}
                        for source in item.sources
                    ],
                }
                for item in self.members
            ],
        }

    @property
    def request_sha256(self) -> str:
        return _canonical_sha256(self.request_payload)


@dataclass(frozen=True, slots=True)
class IVEFaceGalleryAcquisitionProgress:
    fraction: float
    phase: str
    member: IVEMember | None = None
    video_id: str | None = None
    song: str | None = None
    reused: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.fraction) or not 0.0 <= self.fraction <= 1.0:
            raise IVEFaceGalleryAcquisitionError(
                "progress fraction must be in [0, 1]"
            )
        scoped = (self.member is not None, self.video_id is not None, self.song is not None)
        if len(set(scoped)) != 1:
            raise IVEFaceGalleryAcquisitionError(
                "source progress must bind member, video_id, and song together"
            )


ProgressCallback = Callable[[IVEFaceGalleryAcquisitionProgress], None]


@dataclass(frozen=True, slots=True)
class _StableFile:
    path: Path
    size_bytes: int
    sha256: str
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int


def _stable_regular_file(
    path: Path, *, maximum_bytes: int | None = None
) -> _StableFile:
    unresolved = Path(path).expanduser()
    try:
        before = os.lstat(unresolved)
    except OSError as exc:
        raise IVEFaceGalleryAcquisitionError(
            f"required regular file is unavailable: {unresolved}"
        ) from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise IVEFaceGalleryAcquisitionError(
            f"required file is not a non-symlink regular file: {unresolved}"
        )
    if maximum_bytes is not None and not 1 <= before.st_size <= maximum_bytes:
        raise IVEFaceGalleryAcquisitionError("regular file is outside its size bound")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    total = 0
    try:
        descriptor = os.open(unresolved, flags)
        try:
            opened = os.fstat(descriptor)
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if maximum_bytes is not None and total > maximum_bytes:
                    raise IVEFaceGalleryAcquisitionError(
                        "regular file exceeded its read bound"
                    )
                digest.update(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except IVEFaceGalleryAcquisitionError:
        raise
    except OSError as exc:
        raise IVEFaceGalleryAcquisitionError("cannot hash regular file safely") from exc
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        identity(before) != identity(opened)
        or identity(opened) != identity(after)
        or total != after.st_size
    ):
        raise IVEFaceGalleryAcquisitionError("regular file changed while hashing")
    return _StableFile(
        path=unresolved.resolve(strict=True),
        size_bytes=after.st_size,
        sha256=digest.hexdigest(),
        device=after.st_dev,
        inode=after.st_ino,
        mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,
    )


def _read_strict_json(path: Path) -> tuple[Mapping[str, Any], _StableFile]:
    identity = _stable_regular_file(path, maximum_bytes=_MAXIMUM_JSON_BYTES)
    try:
        payload = json.loads(
            identity.path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except IVEFaceGalleryAcquisitionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IVEFaceGalleryAcquisitionError("cannot decode acquisition JSON") from exc
    after = _stable_regular_file(path, maximum_bytes=_MAXIMUM_JSON_BYTES)
    if identity != after:
        raise IVEFaceGalleryAcquisitionError(
            "acquisition JSON changed while validating"
        )
    if not isinstance(payload, Mapping) or not all(
        isinstance(key, str) for key in payload
    ):
        raise IVEFaceGalleryAcquisitionError("acquisition JSON must be an object")
    return payload, identity


def _write_immutable_content_addressed(
    directory: Path, payload: Mapping[str, Any]
) -> tuple[Path, _StableFile]:
    basis = dict(payload)
    basis["content_sha256"] = None
    content_sha256 = _canonical_sha256(basis)
    sealed = dict(payload)
    sealed["content_sha256"] = content_sha256
    encoded = _canonical_json_bytes(sealed)
    target_directory = _ensure_safe_directory(
        _ensure_safe_directory(directory) / content_sha256[:2]
    )
    target = target_directory / f"{content_sha256}.json"
    if target.exists() or target.is_symlink():
        existing, identity = _read_strict_json(target)
        if existing != sealed or identity.sha256 != hashlib.sha256(encoded).hexdigest():
            raise IVEFaceGalleryAcquisitionError(
                "content-addressed acquisition object was tampered"
            )
        return target, identity
    temporary = target_directory / f".{content_sha256}.{uuid4().hex}.tmp"
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(temporary, flags, 0o600)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, target)
        os.chmod(target, 0o400)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return target, _stable_regular_file(target, maximum_bytes=_MAXIMUM_JSON_BYTES)


def _atomic_reference(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _canonical_json_bytes(payload)
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise IVEFaceGalleryAcquisitionError(
            "active acquisition reference parent is unsafe"
        )
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _prepare_external_destination(destination: Path) -> Path:
    """Create a private job root only below a real ``/Volumes/<mount>``.

    Tests replace this one boundary with a temporary-volume fixture.  The
    production entry point has no flag that can waive the mounted-volume gate.
    """

    unresolved = Path(destination).expanduser()
    if not unresolved.is_absolute():
        raise IVEFaceGalleryAcquisitionError(
            "gallery acquisition destination must be absolute"
        )
    parts = unresolved.parts
    if len(parts) < 4 or parts[0] != os.sep or parts[1] != "Volumes":
        raise IVEFaceGalleryAcquisitionError(
            "gallery acquisition requires an external /Volumes destination"
        )
    volume_root = Path(os.sep) / "Volumes" / parts[2]
    try:
        volume_stat = os.lstat(volume_root)
    except OSError as exc:
        raise IVEFaceGalleryAcquisitionError(
            "external gallery volume is not mounted"
        ) from exc
    if (
        stat.S_ISLNK(volume_stat.st_mode)
        or not stat.S_ISDIR(volume_stat.st_mode)
        or not os.path.ismount(volume_root)
    ):
        raise IVEFaceGalleryAcquisitionError(
            "external gallery volume must be a non-symlink mount"
        )
    current = volume_root
    relative = unresolved.relative_to(volume_root)
    for component in relative.parts:
        current = current / component
        if current.exists() or current.is_symlink():
            entry = os.lstat(current)
            if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
                raise IVEFaceGalleryAcquisitionError(
                    "external gallery destination contains an unsafe component"
                )
        else:
            current.mkdir(mode=0o700)
    resolved_volume = volume_root.resolve(strict=True)
    resolved_destination = unresolved.resolve(strict=True)
    try:
        resolved_destination.relative_to(resolved_volume)
    except ValueError as exc:
        raise IVEFaceGalleryAcquisitionError(
            "external gallery destination escaped its mounted volume"
        ) from exc
    private_root = resolved_destination / ".ive-face-gallery-acquisition-v1"
    if private_root.exists() or private_root.is_symlink():
        entry = os.lstat(private_root)
        if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
            raise IVEFaceGalleryAcquisitionError(
                "private gallery acquisition root is unsafe"
            )
    else:
        private_root.mkdir(mode=0o700)
    return private_root.resolve(strict=True)


def _assert_contained_regular_file(path: Path, root: Path) -> _StableFile:
    identity = _stable_regular_file(path)
    try:
        identity.path.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise IVEFaceGalleryAcquisitionError(
            "acquisition media escaped its private root"
        ) from exc
    return identity


def _emit_progress(
    callback: ProgressCallback | None,
    *,
    fraction: float,
    phase: str,
    member: IVEMember | None = None,
    video_id: str | None = None,
    song: str | None = None,
    reused: bool = False,
) -> None:
    if callback is not None:
        callback(
            IVEFaceGalleryAcquisitionProgress(
                fraction=fraction,
                phase=phase,
                member=member,
                video_id=video_id,
                song=song,
                reused=reused,
            )
        )


@dataclass(frozen=True, slots=True)
class _LiveSourceMetadata:
    member: IVEMember
    video_id: str
    song: str
    title: str
    channel: str
    channel_id: str
    duration_seconds: float
    broadcast_date: str
    broadcast_date_basis: str
    broadcast_program: str
    webpage_url: str
    official_channel_signal: str
    observation: Mapping[str, Any]
    observation_sha256: str
    catalog_membership_sha256: str


def _live_single_video_payload(
    client: YTDLPClient, video_id: str
) -> Mapping[str, Any]:
    if not isinstance(client, YTDLPClient):
        raise IVEFaceGalleryAcquisitionError(
            "dedicated acquisition requires the verified YTDLPClient adapter"
        )
    if client.download_profile != FULL_HD_AVC_COMPATIBILITY_PROFILE:
        raise IVEFaceGalleryAcquisitionError(
            "gallery FaceCam acquisition requires full_hd_avc_compatibility_v1"
        )
    url = f"https://www.youtube.com/watch?v={video_id}"
    command = (
        *client.command,
        "--no-config",
        "--no-cookies",
        "--no-playlist",
        "--skip-download",
        "--no-warnings",
        "--dump-single-json",
        url,
    )
    try:
        completed = client._runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise IVEFaceGalleryAcquisitionError(
            f"live metadata revalidation failed for {video_id}: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or "yt-dlp returned no metadata").strip()
        raise IVEFaceGalleryAcquisitionError(
            f"live metadata revalidation failed for {video_id}: {detail}"
        )
    try:
        payload = json.loads(
            completed.stdout,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except IVEFaceGalleryAcquisitionError:
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise IVEFaceGalleryAcquisitionError(
            f"live metadata response is invalid for {video_id}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise IVEFaceGalleryAcquisitionError(
            f"live metadata response is not an object for {video_id}"
        )
    return payload


def _fold(value: str) -> str:
    return " ".join(value.casefold().replace("\u3000", " ").split())


def _recognized_title_members(title: str) -> frozenset[IVEMember]:
    return frozenset(
        member
        for member, aliases in _MEMBER_ALIASES.items()
        if _contains_alias(title, aliases)
    )


def _validate_live_source_metadata(
    *,
    payload: Mapping[str, Any],
    member: IVEMember,
    song: str,
    video_id: str,
    catalog_membership_sha256: str,
) -> _LiveSourceMetadata:
    if payload.get("id") != video_id:
        raise IVEFaceGalleryAcquisitionError(
            f"live metadata video ID mismatch for {video_id}"
        )
    title = payload.get("title")
    channel = payload.get("channel") or payload.get("uploader")
    channel_id = payload.get("channel_id") or payload.get("uploader_id")
    if not all(
        isinstance(item, str) and bool(item.strip())
        for item in (title, channel, channel_id)
    ):
        raise IVEFaceGalleryAcquisitionError(
            f"official title/channel metadata is incomplete for {video_id}"
        )
    assert isinstance(title, str)
    assert isinstance(channel, str)
    assert isinstance(channel_id, str)
    title = title.strip()
    channel = channel.strip()
    channel_id = channel_id.strip()
    folded = _fold(title)
    hard_rejection = _current_hard_title_rejection_reason(title)
    if hard_rejection is not None:
        raise IVEFaceGalleryAcquisitionError(
            f"{video_id} is rejected as {hard_rejection}"
        )
    if any(marker in folded for marker in _FULLCAM_TITLE_MARKERS):
        raise IVEFaceGalleryAcquisitionError(
            f"{video_id} is an ordinary landscape fullcam"
        )
    if not any(marker in folded for marker in _CLOSEUP_TITLE_MARKERS):
        raise IVEFaceGalleryAcquisitionError(
            f"{video_id} lacks an explicit FaceCam/얼빡/close-up title signal"
        )
    if not _has_ive_title_signal(title) or not _contains_alias(
        title, _song_aliases(song)
    ):
        raise IVEFaceGalleryAcquisitionError(
            f"{video_id} title lacks exact IVE or song authorization"
        )
    recognized_members = _recognized_title_members(title)
    if recognized_members != frozenset({member}):
        raise IVEFaceGalleryAcquisitionError(
            f"{video_id} title is missing or ambiguous for target member"
        )
    official_signal = _official_channel_signal(channel, channel_id)
    if official_signal == "none":
        raise IVEFaceGalleryAcquisitionError(
            f"{video_id} is not from a trusted official music-show channel"
        )
    program = _music_show_program(title)
    if program is None:
        raise IVEFaceGalleryAcquisitionError(
            f"{video_id} title lacks a weekly music-show program"
        )
    broadcast_date, date_basis = _metadata_broadcast_date_with_basis(payload, title)
    if broadcast_date is None or date_basis is None:
        raise IVEFaceGalleryAcquisitionError(
            f"{video_id} lacks a broadcast/upload date"
        )
    duration = _finite_positive(payload.get("duration"), "metadata duration")
    if not _MINIMUM_SOURCE_DURATION_SECONDS <= duration <= _MAXIMUM_SOURCE_DURATION_SECONDS:
        raise IVEFaceGalleryAcquisitionError(
            f"{video_id} duration is outside the music-show source bound"
        )
    if payload.get("has_drm") is True:
        raise IVEFaceGalleryAcquisitionError(f"{video_id} is DRM protected")
    availability = payload.get("availability")
    if availability not in (None, "public"):
        raise IVEFaceGalleryAcquisitionError(
            f"{video_id} is not publicly available"
        )
    live_status = payload.get("live_status")
    if live_status in {"is_live", "is_upcoming", "post_live"} or payload.get(
        "is_live"
    ) is True:
        raise IVEFaceGalleryAcquisitionError(
            f"{video_id} is a live/upcoming source"
        )
    expected_url = f"https://www.youtube.com/watch?v={video_id}"
    observation: dict[str, Any] = {
        "contract": IVE_FACE_GALLERY_METADATA_OBSERVATION_CONTRACT,
        "video_id": video_id,
        "title": title,
        "channel": channel,
        "channel_id": channel_id,
        "duration_seconds": duration,
        "broadcast_date": broadcast_date,
        "broadcast_date_basis": date_basis,
        "broadcast_program": program,
        "official_channel_signal": official_signal,
        "upload_date": payload.get("upload_date"),
        "timestamp": payload.get("timestamp"),
        "release_timestamp": payload.get("release_timestamp"),
        "availability": availability,
        "live_status": live_status,
        "has_drm": bool(payload.get("has_drm", False)),
        "webpage_url": expected_url,
        "catalog_membership_sha256": catalog_membership_sha256,
        "single_video_policy": {
            "cookies": "disabled",
            "playlist_expansion": "disabled",
            "download": "disabled_for_metadata_observation",
        },
    }
    observation_sha256 = _canonical_sha256(observation)
    return _LiveSourceMetadata(
        member=member,
        video_id=video_id,
        song=song,
        title=title,
        channel=channel,
        channel_id=channel_id,
        duration_seconds=duration,
        broadcast_date=broadcast_date,
        broadcast_date_basis=date_basis,
        broadcast_program=program,
        webpage_url=expected_url,
        official_channel_signal=official_signal,
        observation=observation,
        observation_sha256=observation_sha256,
        catalog_membership_sha256=catalog_membership_sha256,
    )


def _catalog_membership_observations(
    client: YTDLPClient,
    requested_ids: Iterable[str],
) -> tuple[dict[str, str], str]:
    try:
        raw = client.playlist_catalog(PINNED_IVE_STAGE_PLAYLIST_ID)
    except AcquisitionError as exc:
        raise IVEFaceGalleryAcquisitionError(str(exc)) from exc
    if raw.get("_crosscut_ytdlp_returncode") != 0:
        raise IVEFaceGalleryAcquisitionError(
            "pinned IVE STAGE catalogue did not revalidate completely"
        )
    try:
        normalized = _normalized_catalog_object(dict(raw))
    except AcquisitionError as exc:
        raise IVEFaceGalleryAcquisitionError(str(exc)) from exc
    entries = normalized["entries"]
    by_id = {item["id"]: item for item in entries}
    requested = tuple(requested_ids)
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise IVEFaceGalleryAcquisitionError(
            f"explicit video IDs are absent from pinned IVE STAGE catalogue: {missing!r}"
        )
    memberships = {
        video_id: _canonical_sha256(
            {
                "contract": IVE_FACE_GALLERY_CATALOG_OBSERVATION_CONTRACT,
                "playlist_id": PINNED_IVE_STAGE_PLAYLIST_ID,
                "owner_channel_id": PINNED_IVE_STAGE_OWNER_CHANNEL_ID,
                "video_id": video_id,
                "catalog_entry": by_id[video_id],
            }
        )
        for video_id in requested
    }
    aggregate = _canonical_sha256(
        {
            "contract": IVE_FACE_GALLERY_CATALOG_OBSERVATION_CONTRACT,
            "playlist_id": PINNED_IVE_STAGE_PLAYLIST_ID,
            "owner_channel_id": PINNED_IVE_STAGE_OWNER_CHANNEL_ID,
            "requested_memberships": [
                {"video_id": video_id, "sha256": memberships[video_id]}
                for video_id in sorted(memberships)
            ],
        }
    )
    return memberships, aggregate


@dataclass(frozen=True, slots=True)
class _VerifiedMedia:
    identity: _StableFile
    width: int
    height: int
    frame_rate: float
    duration_seconds: float
    codec_name: str

    def to_manifest_dict(self, root: Path) -> dict[str, Any]:
        return {
            "relative_path": self.identity.path.relative_to(root).as_posix(),
            "size_bytes": self.identity.size_bytes,
            "sha256": self.identity.sha256,
            "width": self.width,
            "height": self.height,
            "frame_rate": self.frame_rate,
            "duration_seconds": self.duration_seconds,
            "video_codec": self.codec_name,
            "ffprobe_contract": "typed_media_info_exact_dimensions_fps_duration_v1",
        }


def _verified_media(
    path: Path,
    *,
    root: Path,
    expected_duration_seconds: float,
    media_probe: Callable[[Path], MediaInfo],
) -> _VerifiedMedia:
    identity = _assert_contained_regular_file(path, root)
    try:
        info = media_probe(identity.path)
    except (MediaProbeError, OSError, ValueError) as exc:
        raise IVEFaceGalleryAcquisitionError("downloaded media probe failed") from exc
    if not isinstance(info, MediaInfo) or len(info.video_streams) != 1:
        raise IVEFaceGalleryAcquisitionError(
            "downloaded FaceCam must expose exactly one typed video stream"
        )
    stream = info.video_streams[0]
    width = _positive_integer(stream.width, "media width")
    height = _positive_integer(stream.height, "media height")
    frame_rate = _finite_positive(stream.frame_rate, "media frame rate")
    raw_duration = (
        stream.duration_seconds
        if stream.duration_seconds is not None
        else info.duration_seconds
    )
    duration = _finite_positive(raw_duration, "media duration")
    codec_name = str(stream.codec_name or "").strip().casefold()
    if codec_name not in {"h264", "avc1"}:
        raise IVEFaceGalleryAcquisitionError(
            "downloaded media is not AVC/H.264 compatible"
        )
    if (
        min(width, height) < _MINIMUM_SHORT_EDGE
        or max(width, height) > _MAXIMUM_LONG_EDGE
    ):
        raise IVEFaceGalleryAcquisitionError(
            "downloaded FaceCam is outside the 720p-to-1080p media bound"
        )
    if not _MINIMUM_FRAME_RATE <= frame_rate <= _MAXIMUM_FRAME_RATE:
        raise IVEFaceGalleryAcquisitionError(
            "downloaded FaceCam frame rate is outside its verified bound"
        )
    duration_tolerance = max(
        _DURATION_ABSOLUTE_TOLERANCE_SECONDS,
        expected_duration_seconds * _DURATION_RELATIVE_TOLERANCE,
    )
    if abs(duration - expected_duration_seconds) > duration_tolerance:
        raise IVEFaceGalleryAcquisitionError(
            "downloaded FaceCam duration does not match live metadata"
        )
    if info.size_bytes is not None and info.size_bytes != identity.size_bytes:
        raise IVEFaceGalleryAcquisitionError(
            "ffprobe size does not match current media bytes"
        )
    after_probe = _assert_contained_regular_file(path, root)
    if after_probe != identity:
        raise IVEFaceGalleryAcquisitionError(
            "downloaded media changed while ffprobe was running"
        )
    return _VerifiedMedia(
        identity=identity,
        width=width,
        height=height,
        frame_rate=frame_rate,
        duration_seconds=duration,
        codec_name=codec_name,
    )


def _validate_content_addressed_object(
    payload: Mapping[str, Any], identity: _StableFile
) -> str:
    content_sha256 = _sha256(payload.get("content_sha256"), "content SHA-256")
    basis = dict(payload)
    basis["content_sha256"] = None
    if not hmac.compare_digest(content_sha256, _canonical_sha256(basis)):
        raise IVEFaceGalleryAcquisitionError(
            "content-addressed acquisition object digest mismatch"
        )
    if identity.path.name != f"{content_sha256}.json":
        raise IVEFaceGalleryAcquisitionError(
            "content-addressed acquisition object filename mismatch"
        )
    return content_sha256


def _load_active_object(
    *,
    reference_path: Path,
    root: Path,
    expected_contract: str,
) -> tuple[Mapping[str, Any], _StableFile, str] | None:
    if not reference_path.exists() and not reference_path.is_symlink():
        return None
    reference, _ = _read_strict_json(reference_path)
    if (
        set(reference) != {"contract", "content_sha256", "relative_path"}
        or reference.get("contract")
        != "ive_face_gallery_active_content_reference_v1"
    ):
        raise IVEFaceGalleryAcquisitionError("active acquisition reference is invalid")
    content_sha256 = _sha256(
        reference.get("content_sha256"), "active content SHA-256"
    )
    relative_raw = reference.get("relative_path")
    if not isinstance(relative_raw, str):
        raise IVEFaceGalleryAcquisitionError(
            "active acquisition reference path is invalid"
        )
    relative = Path(relative_raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise IVEFaceGalleryAcquisitionError(
            "active acquisition reference escaped its job root"
        )
    target = root / relative
    payload, identity = _read_strict_json(target)
    try:
        identity.path.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise IVEFaceGalleryAcquisitionError(
            "active acquisition object escaped its job root"
        ) from exc
    if (
        payload.get("contract") != expected_contract
        or not hmac.compare_digest(
            _validate_content_addressed_object(payload, identity), content_sha256
        )
    ):
        raise IVEFaceGalleryAcquisitionError(
            "active acquisition object contract mismatch"
        )
    return payload, identity, content_sha256


def _write_active_object_reference(
    *, reference_path: Path, root: Path, object_path: Path, content_sha256: str
) -> None:
    relative_parent = reference_path.parent.relative_to(root)
    current = root
    for component in relative_parent.parts:
        current = _ensure_safe_directory(current / component)
    _atomic_reference(
        reference_path,
        {
            "contract": "ive_face_gallery_active_content_reference_v1",
            "content_sha256": content_sha256,
            "relative_path": object_path.relative_to(root).as_posix(),
        },
    )


def _receipt_payload(
    *,
    spec: IVEFaceGalleryAcquisitionSpec,
    metadata: _LiveSourceMetadata,
    media: _VerifiedMedia,
    job_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": IVE_FACE_GALLERY_ACQUISITION_SCHEMA_VERSION,
        "contract": IVE_FACE_GALLERY_SOURCE_RECEIPT_CONTRACT,
        "privacy_policy": PRIVATE_LOCAL_ACQUISITION_POLICY,
        "request_sha256": spec.request_sha256,
        "group": "IVE",
        "gallery_id": spec.gallery_id,
        "song": metadata.song,
        "member": metadata.member.value,
        "video_id": metadata.video_id,
        "source_authority": DEDICATED_FACECAM_SOURCE_AUTHORITY,
        "metadata_observation": dict(metadata.observation),
        "metadata_observation_sha256": metadata.observation_sha256,
        "catalog_membership_sha256": metadata.catalog_membership_sha256,
        "media": media.to_manifest_dict(job_root),
        "download_policy": {
            "profile": FULL_HD_AVC_COMPATIBILITY_PROFILE,
            "maximum_long_edge": _MAXIMUM_LONG_EDGE,
            "cookies": "disabled",
            "playlist_download": "disabled",
            "drm_bypass": "disabled",
            "single_video_only": True,
        },
        "content_sha256": None,
    }


def _media_path_from_entry(
    media: object, *, root: Path, video_id: str
) -> Path:
    if not isinstance(media, Mapping):
        raise IVEFaceGalleryAcquisitionError("source receipt lacks media evidence")
    relative_raw = media.get("relative_path")
    if not isinstance(relative_raw, str):
        raise IVEFaceGalleryAcquisitionError("receipt media path is invalid")
    relative = Path(relative_raw)
    if (
        relative.is_absolute()
        or len(relative.parts) != 2
        or relative.parts[0] != "videos"
        or not relative.parts[1].startswith(f"{video_id}.")
        or ".." in relative.parts
    ):
        raise IVEFaceGalleryAcquisitionError(
            "receipt media path escaped the dedicated videos directory"
        )
    return root / relative


def _validate_source_receipt(
    *,
    receipt: Mapping[str, Any],
    identity: _StableFile,
    spec: IVEFaceGalleryAcquisitionSpec,
    metadata: _LiveSourceMetadata,
    job_root: Path,
    media_probe: Callable[[Path], MediaInfo],
) -> tuple[dict[str, Any], _VerifiedMedia]:
    _validate_content_addressed_object(receipt, identity)
    if not (
        receipt.get("schema_version")
        == IVE_FACE_GALLERY_ACQUISITION_SCHEMA_VERSION
        and receipt.get("contract") == IVE_FACE_GALLERY_SOURCE_RECEIPT_CONTRACT
        and receipt.get("privacy_policy") == PRIVATE_LOCAL_ACQUISITION_POLICY
        and receipt.get("request_sha256") == spec.request_sha256
        and receipt.get("group") == "IVE"
        and receipt.get("gallery_id") == spec.gallery_id
        and receipt.get("song") == metadata.song
        and receipt.get("member") == metadata.member.value
        and receipt.get("video_id") == metadata.video_id
        and receipt.get("source_authority") == DEDICATED_FACECAM_SOURCE_AUTHORITY
        and receipt.get("metadata_observation") == metadata.observation
        and receipt.get("metadata_observation_sha256")
        == metadata.observation_sha256
        and receipt.get("catalog_membership_sha256")
        == metadata.catalog_membership_sha256
    ):
        raise IVEFaceGalleryAcquisitionError(
            "source receipt no longer matches current catalogue/metadata authority"
        )
    media_entry = receipt.get("media")
    media_path = _media_path_from_entry(
        media_entry, root=job_root, video_id=metadata.video_id
    )
    current = _verified_media(
        media_path,
        root=job_root,
        expected_duration_seconds=metadata.duration_seconds,
        media_probe=media_probe,
    )
    expected_media = current.to_manifest_dict(job_root)
    if media_entry != expected_media:
        raise IVEFaceGalleryAcquisitionError(
            "source receipt no longer matches current media SHA/ffprobe evidence"
        )
    return dict(receipt), current


def _matching_video_artifacts(videos_directory: Path, video_id: str) -> list[Path]:
    if not videos_directory.exists():
        return []
    if videos_directory.is_symlink() or not videos_directory.is_dir():
        raise IVEFaceGalleryAcquisitionError("videos cache directory is unsafe")
    return sorted(
        (
            item
            for item in videos_directory.iterdir()
            if item.name.startswith(f"{video_id}.")
        ),
        key=lambda item: item.name,
    )


def _quarantine_untrusted_video_artifacts(
    *, job_root: Path, video_id: str
) -> list[dict[str, Any]]:
    videos = job_root / "videos"
    artifacts = _matching_video_artifacts(videos, video_id)
    quarantine = _ensure_safe_directory(job_root / "quarantine")
    evidence: list[dict[str, Any]] = []
    for artifact in artifacts:
        before = os.lstat(artifact)
        kind = "symlink" if stat.S_ISLNK(before.st_mode) else "regular_or_partial"
        digest: str | None = None
        size = before.st_size
        if stat.S_ISREG(before.st_mode):
            digest = _stable_regular_file(artifact).sha256
        target = quarantine / f"{artifact.name}.invalid.{uuid4().hex}"
        os.replace(artifact, target)
        evidence.append(
            {
                "original_relative_path": artifact.relative_to(job_root).as_posix(),
                "quarantine_relative_path": target.relative_to(job_root).as_posix(),
                "file_kind": kind,
                "size_bytes": size,
                "sha256": digest,
            }
        )
    return evidence


def _persist_failure_evidence(
    *,
    job_root: Path,
    spec: IVEFaceGalleryAcquisitionSpec,
    metadata: _LiveSourceMetadata | None,
    member: IVEMember,
    video_id: str,
    error: BaseException,
    quarantined: Sequence[Mapping[str, Any]],
) -> Path:
    payload = {
        "schema_version": 1,
        "contract": "ive_face_gallery_acquisition_failure_evidence_v1",
        "request_sha256": spec.request_sha256,
        "member": member.value,
        "video_id": video_id,
        "metadata_observation_sha256": (
            metadata.observation_sha256 if metadata is not None else None
        ),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "quarantined_artifacts": [dict(item) for item in quarantined],
        "content_sha256": None,
    }
    return _write_immutable_content_addressed(
        job_root / "failures", payload
    )[0]


def _download_and_seal_source_receipt(
    *,
    client: YTDLPClient,
    spec: IVEFaceGalleryAcquisitionSpec,
    metadata: _LiveSourceMetadata,
    job_root: Path,
    media_probe: Callable[[Path], MediaInfo],
) -> tuple[dict[str, Any], _VerifiedMedia, bool]:
    reference = job_root / "receipts" / metadata.video_id / "active.json"
    active = _load_active_object(
        reference_path=reference,
        root=job_root,
        expected_contract=IVE_FACE_GALLERY_SOURCE_RECEIPT_CONTRACT,
    )
    if active is not None:
        receipt, identity, _ = active
        validated, media = _validate_source_receipt(
            receipt=receipt,
            identity=identity,
            spec=spec,
            metadata=metadata,
            job_root=job_root,
            media_probe=media_probe,
        )
        return validated, media, True

    preexisting = _quarantine_untrusted_video_artifacts(
        job_root=job_root, video_id=metadata.video_id
    )
    candidate = AcquisitionCandidate(
        video_id=metadata.video_id,
        title=metadata.title,
        channel=metadata.channel,
        channel_id=metadata.channel_id,
        duration_seconds=metadata.duration_seconds,
        webpage_url=metadata.webpage_url,
        queries=(f"pinned:{metadata.video_id}",),
        selection_tier=0,
        trust_score=100,
        broadcast_date=metadata.broadcast_date,
        broadcast_program=metadata.broadcast_program,
        broadcast_date_basis=metadata.broadcast_date_basis,
        official_channel_signal=metadata.official_channel_signal,
        discovery_sources=("pinned_ive_stage_catalog_explicit_id",),
    )
    try:
        downloaded = client.download(candidate, job_root)
        media = _verified_media(
            downloaded,
            root=job_root,
            expected_duration_seconds=metadata.duration_seconds,
            media_probe=media_probe,
        )
        payload = _receipt_payload(
            spec=spec, metadata=metadata, media=media, job_root=job_root
        )
        receipt_path, receipt_identity = _write_immutable_content_addressed(
            job_root / "receipt-objects", payload
        )
        receipt, _ = _read_strict_json(receipt_path)
        content_sha256 = _validate_content_addressed_object(
            receipt, receipt_identity
        )
        _write_active_object_reference(
            reference_path=reference,
            root=job_root,
            object_path=receipt_path,
            content_sha256=content_sha256,
        )
        validated, media = _validate_source_receipt(
            receipt=receipt,
            identity=receipt_identity,
            spec=spec,
            metadata=metadata,
            job_root=job_root,
            media_probe=media_probe,
        )
        if preexisting:
            _persist_failure_evidence(
                job_root=job_root,
                spec=spec,
                metadata=metadata,
                member=metadata.member,
                video_id=metadata.video_id,
                error=IVEFaceGalleryAcquisitionError(
                    "unauthenticated pre-existing artifacts quarantined before download"
                ),
                quarantined=preexisting,
            )
        return validated, media, False
    except Exception as exc:
        quarantined = [
            *preexisting,
            *_quarantine_untrusted_video_artifacts(
                job_root=job_root, video_id=metadata.video_id
            ),
        ]
        _persist_failure_evidence(
            job_root=job_root,
            spec=spec,
            metadata=metadata,
            member=metadata.member,
            video_id=metadata.video_id,
            error=exc,
            quarantined=quarantined,
        )
        if isinstance(exc, IVEFaceGalleryAcquisitionError):
            raise
        if isinstance(exc, AcquisitionError):
            raise IVEFaceGalleryAcquisitionError(str(exc)) from exc
        raise


def _ensure_safe_directory(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        entry = os.lstat(path)
        if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
            raise IVEFaceGalleryAcquisitionError(
                f"acquisition directory is unsafe: {path}"
            )
    else:
        path.mkdir(mode=0o700)
    return path.resolve(strict=True)


def _final_manifest_payload(
    *,
    spec: IVEFaceGalleryAcquisitionSpec,
    catalog_observation_sha256: str,
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": IVE_FACE_GALLERY_ACQUISITION_SCHEMA_VERSION,
        "contract": IVE_FACE_GALLERY_ACQUISITION_CONTRACT,
        "privacy_policy": PRIVATE_LOCAL_ACQUISITION_POLICY,
        "request": spec.request_payload,
        "request_sha256": spec.request_sha256,
        "catalog_observation": {
            "contract": IVE_FACE_GALLERY_CATALOG_OBSERVATION_CONTRACT,
            "playlist_id": PINNED_IVE_STAGE_PLAYLIST_ID,
            "owner_channel_id": PINNED_IVE_STAGE_OWNER_CHANNEL_ID,
            "sha256": catalog_observation_sha256,
        },
        "download_policy": {
            "profile": FULL_HD_AVC_COMPATIBILITY_PROFILE,
            "maximum_long_edge": _MAXIMUM_LONG_EDGE,
            "minimum_short_edge": _MINIMUM_SHORT_EDGE,
            "cookies": "disabled",
            "playlist_download": "disabled",
            "drm_bypass": "disabled",
            "single_video_only": True,
        },
        "source_authority": DEDICATED_FACECAM_SOURCE_AUTHORITY,
        "sources": [
            {
                "member": receipt["member"],
                "song": receipt["song"],
                "video_id": receipt["video_id"],
                "receipt_content_sha256": receipt["content_sha256"],
                "metadata_observation": receipt["metadata_observation"],
                "metadata_observation_sha256": receipt[
                    "metadata_observation_sha256"
                ],
                "catalog_membership_sha256": receipt[
                    "catalog_membership_sha256"
                ],
                "media": receipt["media"],
            }
            for receipt in receipts
        ],
        "raw_frame_count": 0,
        "raw_embedding_count": 0,
        "content_sha256": None,
    }


def _validate_final_manifest_shape(
    manifest: Mapping[str, Any], identity: _StableFile
) -> str:
    content_sha256 = _validate_content_addressed_object(manifest, identity)
    request = manifest.get("request")
    catalog = manifest.get("catalog_observation")
    sources = manifest.get("sources")
    if not (
        manifest.get("schema_version")
        == IVE_FACE_GALLERY_ACQUISITION_SCHEMA_VERSION
        and manifest.get("contract") == IVE_FACE_GALLERY_ACQUISITION_CONTRACT
        and manifest.get("privacy_policy") == PRIVATE_LOCAL_ACQUISITION_POLICY
        and isinstance(request, Mapping)
        and request.get("group") == "IVE"
        and isinstance(request.get("gallery_id"), str)
        and manifest.get("request_sha256") == _canonical_sha256(request)
        and isinstance(catalog, Mapping)
        and catalog.get("playlist_id") == PINNED_IVE_STAGE_PLAYLIST_ID
        and catalog.get("owner_channel_id")
        == PINNED_IVE_STAGE_OWNER_CHANNEL_ID
        and _SHA256.fullmatch(str(catalog.get("sha256", ""))) is not None
        and manifest.get("source_authority")
        == DEDICATED_FACECAM_SOURCE_AUTHORITY
        and manifest.get("raw_frame_count") == 0
        and manifest.get("raw_embedding_count") == 0
        and isinstance(sources, list)
    ):
        raise IVEFaceGalleryAcquisitionError(
            "dedicated acquisition manifest scope is invalid"
        )
    member_counts = {member: 0 for member in _MEMBERS}
    seen_ids: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            raise IVEFaceGalleryAcquisitionError(
                "dedicated acquisition source entry is invalid"
            )
        try:
            member = _member(source.get("member"))
        except IVEFaceGalleryAcquisitionError as exc:
            raise IVEFaceGalleryAcquisitionError(
                "dedicated manifest contains a non-IVE member"
            ) from exc
        video_id = source.get("video_id")
        song = source.get("song")
        if not (
            isinstance(video_id, str)
            and _VIDEO_ID.fullmatch(video_id) is not None
            and video_id not in seen_ids
            and isinstance(song, str)
            and bool(song.strip())
            and _SHA256.fullmatch(
                str(source.get("receipt_content_sha256", ""))
            )
            is not None
            and _SHA256.fullmatch(
                str(source.get("metadata_observation_sha256", ""))
            )
            is not None
            and _SHA256.fullmatch(
                str(source.get("catalog_membership_sha256", ""))
            )
            is not None
            and isinstance(source.get("metadata_observation"), Mapping)
            and isinstance(source.get("media"), Mapping)
        ):
            raise IVEFaceGalleryAcquisitionError(
                "dedicated acquisition source evidence is incomplete"
            )
        if _compact_token(song) in {"attitude", "애티튜드"}:
            raise IVEFaceGalleryAcquisitionError(
                "ATTITUDE is quarantined from dedicated acquisition manifests"
            )
        seen_ids.add(video_id)
        member_counts[member] += 1
    if set(member_counts) != set(_MEMBERS) or any(
        count < 2 for count in member_counts.values()
    ):
        raise IVEFaceGalleryAcquisitionError(
            "dedicated manifest needs at least two sources for all six members"
        )
    return content_sha256


def _authority_unsigned_payload(
    *,
    gallery_id: str,
    member: IVEMember,
    song: str,
    video_id: str,
    title: str,
    broadcast_date: str,
    broadcast_program: str,
    channel_id: str,
    metadata_observation_sha256: str,
    catalog_membership_sha256: str,
    receipt_content_sha256: str,
    manifest_path: Path,
    manifest_file_sha256: str,
    manifest_content_sha256: str,
    media_path: Path,
    media_size_bytes: int,
    media_sha256: str,
    media_width: int,
    media_height: int,
    media_frame_rate: float,
    media_duration_seconds: float,
) -> dict[str, Any]:
    return {
        "contract": DEDICATED_FACECAM_ACQUISITION_AUTHORITY_BASIS,
        "gallery_id": gallery_id,
        "group": "IVE",
        "member": member.value,
        "song": song,
        "video_id": video_id,
        "title": title,
        "broadcast_date": broadcast_date,
        "broadcast_program": broadcast_program,
        "channel_id": channel_id,
        "metadata_observation_sha256": metadata_observation_sha256,
        "catalog_membership_sha256": catalog_membership_sha256,
        "receipt_content_sha256": receipt_content_sha256,
        "manifest_path_sha256": hashlib.sha256(
            os.fspath(manifest_path).encode("utf-8")
        ).hexdigest(),
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_content_sha256": manifest_content_sha256,
        "media_path_sha256": hashlib.sha256(
            os.fspath(media_path).encode("utf-8")
        ).hexdigest(),
        "media_size_bytes": media_size_bytes,
        "media_sha256": media_sha256,
        "media_width": media_width,
        "media_height": media_height,
        "media_frame_rate": media_frame_rate,
        "media_duration_seconds": media_duration_seconds,
        "source_authority": DEDICATED_FACECAM_SOURCE_AUTHORITY,
    }


@dataclass(frozen=True, slots=True)
class DedicatedFaceCamAcquisitionAuthority:
    """Process-local proof of fresh metadata plus current media integrity."""

    gallery_id: str
    member: IVEMember
    song: str
    video_id: str
    title: str
    broadcast_date: str
    broadcast_program: str
    channel_id: str
    metadata_observation_sha256: str
    catalog_membership_sha256: str
    receipt_content_sha256: str
    manifest_path: Path
    manifest_file_sha256: str
    manifest_content_sha256: str
    media_path: Path
    media_size_bytes: int
    media_sha256: str
    media_width: int
    media_height: int
    media_frame_rate: float
    media_duration_seconds: float
    authority_content_sha256: str
    source_authority: str
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _AUTHORITY_SEAL:
            raise TypeError(
                "dedicated FaceCam authority must be issued by acquisition"
            )
        if self.member not in _MEMBERS:
            raise IVEFaceGalleryAcquisitionError("authority member is invalid")
        if _compact_token(self.song) in {"attitude", "애티튜드"}:
            raise IVEFaceGalleryAcquisitionError(
                "ATTITUDE cannot be authorized for the face gallery"
            )
        for name in (
            "metadata_observation_sha256",
            "catalog_membership_sha256",
            "receipt_content_sha256",
            "manifest_file_sha256",
            "manifest_content_sha256",
            "media_sha256",
            "authority_content_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.source_authority != DEDICATED_FACECAM_SOURCE_AUTHORITY:
            raise IVEFaceGalleryAcquisitionError(
                "dedicated FaceCam source authority mismatch"
            )
        if not isinstance(self.video_id, str) or _VIDEO_ID.fullmatch(
            self.video_id
        ) is None:
            raise IVEFaceGalleryAcquisitionError("authority video_id is invalid")
        if not isinstance(self.gallery_id, str) or not self.gallery_id.strip():
            raise IVEFaceGalleryAcquisitionError("authority gallery_id is invalid")
        _positive_integer(self.media_size_bytes, "authority media size")
        _positive_integer(self.media_width, "authority media width")
        _positive_integer(self.media_height, "authority media height")
        _finite_positive(self.media_frame_rate, "authority media frame rate")
        _finite_positive(self.media_duration_seconds, "authority media duration")
        expected = _canonical_sha256(
            _authority_unsigned_payload(
                gallery_id=self.gallery_id,
                member=self.member,
                song=self.song,
                video_id=self.video_id,
                title=self.title,
                broadcast_date=self.broadcast_date,
                broadcast_program=self.broadcast_program,
                channel_id=self.channel_id,
                metadata_observation_sha256=self.metadata_observation_sha256,
                catalog_membership_sha256=self.catalog_membership_sha256,
                receipt_content_sha256=self.receipt_content_sha256,
                manifest_path=Path(self.manifest_path),
                manifest_file_sha256=self.manifest_file_sha256,
                manifest_content_sha256=self.manifest_content_sha256,
                media_path=Path(self.media_path),
                media_size_bytes=self.media_size_bytes,
                media_sha256=self.media_sha256,
                media_width=self.media_width,
                media_height=self.media_height,
                media_frame_rate=self.media_frame_rate,
                media_duration_seconds=self.media_duration_seconds,
            )
        )
        if not hmac.compare_digest(expected, self.authority_content_sha256):
            raise IVEFaceGalleryAcquisitionError(
                "dedicated FaceCam authority content seal mismatch"
            )


def is_dedicated_facecam_acquisition_authority(value: object) -> bool:
    return bool(
        isinstance(value, DedicatedFaceCamAcquisitionAuthority)
        and value._seal is _AUTHORITY_SEAL
    )


def revalidate_dedicated_facecam_acquisition_authority(
    authority: DedicatedFaceCamAcquisitionAuthority,
    *,
    media_probe: Callable[[Path], MediaInfo] = probe_media,
) -> DedicatedFaceCamAcquisitionAuthority:
    """Rebind one process seal to its current manifest, receipt, and media."""

    if not is_dedicated_facecam_acquisition_authority(authority):
        raise IVEFaceGalleryAcquisitionError(
            "dedicated FaceCam source lacks process-local acquisition authority"
        )
    manifest, manifest_identity = _read_strict_json(Path(authority.manifest_path))
    manifest_content_sha256 = _validate_final_manifest_shape(
        manifest, manifest_identity
    )
    if not (
        hmac.compare_digest(
            manifest_identity.sha256, authority.manifest_file_sha256
        )
        and hmac.compare_digest(
            manifest_content_sha256, authority.manifest_content_sha256
        )
    ):
        raise IVEFaceGalleryAcquisitionError(
            "dedicated FaceCam manifest no longer matches its process seal"
        )
    request = manifest.get("request")
    sources = manifest.get("sources")
    assert isinstance(request, Mapping)
    assert isinstance(sources, list)
    if request.get("gallery_id") != authority.gallery_id:
        raise IVEFaceGalleryAcquisitionError(
            "dedicated FaceCam gallery scope changed"
        )
    matches = [
        item
        for item in sources
        if isinstance(item, Mapping)
        and item.get("video_id") == authority.video_id
        and item.get("member") == authority.member.value
    ]
    if len(matches) != 1:
        raise IVEFaceGalleryAcquisitionError(
            "dedicated manifest needs exactly one matching authority source"
        )
    source = matches[0]
    metadata = source.get("metadata_observation")
    media = source.get("media")
    if not isinstance(metadata, Mapping) or not isinstance(media, Mapping):
        raise IVEFaceGalleryAcquisitionError(
            "dedicated authority source evidence is invalid"
        )
    if not hmac.compare_digest(
        _canonical_sha256(metadata), authority.metadata_observation_sha256
    ):
        raise IVEFaceGalleryAcquisitionError(
            "dedicated authority metadata observation digest mismatch"
        )
    if not (
        source.get("song") == authority.song
        and metadata.get("title") == authority.title
        and metadata.get("broadcast_date") == authority.broadcast_date
        and metadata.get("broadcast_program") == authority.broadcast_program
        and metadata.get("channel_id") == authority.channel_id
        and source.get("metadata_observation_sha256")
        == authority.metadata_observation_sha256
        and source.get("catalog_membership_sha256")
        == authority.catalog_membership_sha256
        and source.get("receipt_content_sha256")
        == authority.receipt_content_sha256
    ):
        raise IVEFaceGalleryAcquisitionError(
            "dedicated authority source metadata changed"
        )
    job_root = manifest_identity.path.parents[2]
    receipt_active = _load_active_object(
        reference_path=job_root / "receipts" / authority.video_id / "active.json",
        root=job_root,
        expected_contract=IVE_FACE_GALLERY_SOURCE_RECEIPT_CONTRACT,
    )
    if receipt_active is None or not hmac.compare_digest(
        receipt_active[2], authority.receipt_content_sha256
    ):
        raise IVEFaceGalleryAcquisitionError(
            "dedicated authority source receipt is missing or changed"
        )
    media_path = _media_path_from_entry(
        media, root=job_root, video_id=authority.video_id
    )
    current = _verified_media(
        media_path,
        root=job_root,
        expected_duration_seconds=float(metadata["duration_seconds"]),
        media_probe=media_probe,
    )
    if not (
        current.identity.path == Path(authority.media_path).resolve(strict=True)
        and current.identity.size_bytes == authority.media_size_bytes
        and hmac.compare_digest(current.identity.sha256, authority.media_sha256)
        and current.width == authority.media_width
        and current.height == authority.media_height
        and math.isclose(
            current.frame_rate,
            authority.media_frame_rate,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and math.isclose(
            current.duration_seconds,
            authority.media_duration_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and media == current.to_manifest_dict(job_root)
    ):
        raise IVEFaceGalleryAcquisitionError(
            "dedicated authority no longer matches current media"
        )
    # Re-running the frozen dataclass validation reproduces its unsigned seal.
    DedicatedFaceCamAcquisitionAuthority(
        **{
            field: getattr(authority, field)
            for field in authority.__dataclass_fields__
            if field != "_seal"
        },
        _seal=_AUTHORITY_SEAL,
    )
    return authority


def _issue_authorities(
    *,
    spec: IVEFaceGalleryAcquisitionSpec,
    manifest: Mapping[str, Any],
    manifest_identity: _StableFile,
    manifest_content_sha256: str,
    job_root: Path,
) -> tuple[DedicatedFaceCamAcquisitionAuthority, ...]:
    issued: list[DedicatedFaceCamAcquisitionAuthority] = []
    raw_sources = manifest["sources"]
    assert isinstance(raw_sources, list)
    for source in raw_sources:
        assert isinstance(source, Mapping)
        metadata = source["metadata_observation"]
        media = source["media"]
        assert isinstance(metadata, Mapping)
        assert isinstance(media, Mapping)
        member = _member(source["member"])
        media_path = _media_path_from_entry(
            media, root=job_root, video_id=str(source["video_id"])
        ).resolve(strict=True)
        unsigned = _authority_unsigned_payload(
            gallery_id=spec.gallery_id,
            member=member,
            song=str(source["song"]),
            video_id=str(source["video_id"]),
            title=str(metadata["title"]),
            broadcast_date=str(metadata["broadcast_date"]),
            broadcast_program=str(metadata["broadcast_program"]),
            channel_id=str(metadata["channel_id"]),
            metadata_observation_sha256=str(
                source["metadata_observation_sha256"]
            ),
            catalog_membership_sha256=str(source["catalog_membership_sha256"]),
            receipt_content_sha256=str(source["receipt_content_sha256"]),
            manifest_path=manifest_identity.path,
            manifest_file_sha256=manifest_identity.sha256,
            manifest_content_sha256=manifest_content_sha256,
            media_path=media_path,
            media_size_bytes=int(media["size_bytes"]),
            media_sha256=str(media["sha256"]),
            media_width=int(media["width"]),
            media_height=int(media["height"]),
            media_frame_rate=float(media["frame_rate"]),
            media_duration_seconds=float(media["duration_seconds"]),
        )
        issued.append(
            DedicatedFaceCamAcquisitionAuthority(
                gallery_id=spec.gallery_id,
                member=member,
                song=str(source["song"]),
                video_id=str(source["video_id"]),
                title=str(metadata["title"]),
                broadcast_date=str(metadata["broadcast_date"]),
                broadcast_program=str(metadata["broadcast_program"]),
                channel_id=str(metadata["channel_id"]),
                metadata_observation_sha256=str(
                    source["metadata_observation_sha256"]
                ),
                catalog_membership_sha256=str(
                    source["catalog_membership_sha256"]
                ),
                receipt_content_sha256=str(source["receipt_content_sha256"]),
                manifest_path=manifest_identity.path,
                manifest_file_sha256=manifest_identity.sha256,
                manifest_content_sha256=manifest_content_sha256,
                media_path=media_path,
                media_size_bytes=int(media["size_bytes"]),
                media_sha256=str(media["sha256"]),
                media_width=int(media["width"]),
                media_height=int(media["height"]),
                media_frame_rate=float(media["frame_rate"]),
                media_duration_seconds=float(media["duration_seconds"]),
                authority_content_sha256=_canonical_sha256(unsigned),
                source_authority=DEDICATED_FACECAM_SOURCE_AUTHORITY,
                _seal=_AUTHORITY_SEAL,
            )
        )
    return tuple(issued)


@dataclass(frozen=True, slots=True)
class IVEFaceGalleryDedicatedAcquisitionResult:
    authorities: tuple[DedicatedFaceCamAcquisitionAuthority, ...]
    manifest_path: Path
    manifest_file_sha256: str
    manifest_content_sha256: str
    reused_source_count: int
    downloaded_source_count: int


@dataclass(frozen=True, slots=True)
class IVEFaceGalleryAcquisitionCoordinatorResult:
    acquisition: IVEFaceGalleryDedicatedAcquisitionResult
    materializer_authorities: tuple[object, ...]


def _validate_production_corpus_song_diversity(
    spec: IVEFaceGalleryAcquisitionSpec,
) -> None:
    """Keep the initial real corpus multi-song and ATTITUDE-free by design."""

    for member_spec in spec.members:
        songs = tuple(_compact_token(source.song) for source in member_spec.sources)
        song_set = set(songs)
        if (
            len(song_set) != len(songs)
            or not song_set.issubset(_PRODUCTION_CORPUS_SONG_TOKENS)
            or (
                len(songs) == 3
                and song_set != _PRODUCTION_CORPUS_SONG_TOKENS
            )
        ):
            raise IVEFaceGalleryAcquisitionError(
                f"{member_spec.member.value} production corpus must use distinct "
                "BLACKHOLE/BANG BANG/XOXZ sources"
            )


def acquire_ive_face_identity_gallery_sources(
    spec: IVEFaceGalleryAcquisitionSpec,
    *,
    destination: Path,
    client: YTDLPClient,
    media_probe: Callable[[Path], MediaInfo] = probe_media,
    progress: ProgressCallback | None = None,
) -> IVEFaceGalleryDedicatedAcquisitionResult:
    """Acquire and seal all explicit sources without search or playlist download."""

    if not isinstance(spec, IVEFaceGalleryAcquisitionSpec):
        raise IVEFaceGalleryAcquisitionError(
            "spec must be a typed IVE face-gallery acquisition spec"
        )
    if not isinstance(client, YTDLPClient) or (
        client.download_profile != FULL_HD_AVC_COMPATIBILITY_PROFILE
    ):
        raise IVEFaceGalleryAcquisitionError(
            "acquisition requires YTDLPClient full_hd_avc_compatibility_v1"
        )
    private_root = _prepare_external_destination(destination)
    job_root = _ensure_safe_directory(private_root / spec.request_sha256)
    _emit_progress(progress, fraction=0.0, phase="storage_verified")
    ordered_specs = tuple(
        (member_spec.member, source)
        for member_spec in spec.members
        for source in member_spec.sources
    )
    memberships, catalog_sha256 = _catalog_membership_observations(
        client, (source.video_id for _, source in ordered_specs)
    )
    _emit_progress(progress, fraction=0.05, phase="catalog_revalidated")
    live_sources: list[_LiveSourceMetadata] = []
    total = len(ordered_specs)
    for index, (member, source) in enumerate(ordered_specs, start=1):
        try:
            payload = _live_single_video_payload(client, source.video_id)
            metadata = _validate_live_source_metadata(
                payload=payload,
                member=member,
                song=source.song,
                video_id=source.video_id,
                catalog_membership_sha256=memberships[source.video_id],
            )
        except Exception as exc:
            _persist_failure_evidence(
                job_root=job_root,
                spec=spec,
                metadata=None,
                member=member,
                video_id=source.video_id,
                error=exc,
                quarantined=(),
            )
            raise
        live_sources.append(metadata)
        _emit_progress(
            progress,
            fraction=0.05 + 0.20 * index / total,
            phase="metadata_revalidated",
            member=member,
            video_id=source.video_id,
            song=source.song,
        )
    receipts: list[dict[str, Any]] = []
    reused_count = 0
    for index, metadata in enumerate(live_sources, start=1):
        receipt, _, reused = _download_and_seal_source_receipt(
            client=client,
            spec=spec,
            metadata=metadata,
            job_root=job_root,
            media_probe=media_probe,
        )
        receipts.append(receipt)
        reused_count += int(reused)
        _emit_progress(
            progress,
            fraction=0.25 + 0.65 * index / total,
            phase="source_revalidated" if reused else "source_downloaded",
            member=metadata.member,
            video_id=metadata.video_id,
            song=metadata.song,
            reused=reused,
        )
    expected_manifest = _final_manifest_payload(
        spec=spec,
        catalog_observation_sha256=catalog_sha256,
        receipts=receipts,
    )
    active = _load_active_object(
        reference_path=job_root / "active-manifest.json",
        root=job_root,
        expected_contract=IVE_FACE_GALLERY_ACQUISITION_CONTRACT,
    )
    if active is not None:
        manifest, manifest_identity, manifest_content_sha256 = active
        if manifest != {
            **expected_manifest,
            "content_sha256": _canonical_sha256(expected_manifest),
        }:
            raise IVEFaceGalleryAcquisitionError(
                "completed acquisition manifest no longer matches fresh authority"
            )
    else:
        manifest_path, manifest_identity = _write_immutable_content_addressed(
            job_root / "manifests", expected_manifest
        )
        manifest, _ = _read_strict_json(manifest_path)
        manifest_content_sha256 = _validate_final_manifest_shape(
            manifest, manifest_identity
        )
        _write_active_object_reference(
            reference_path=job_root / "active-manifest.json",
            root=job_root,
            object_path=manifest_path,
            content_sha256=manifest_content_sha256,
        )
    validated_content_sha256 = _validate_final_manifest_shape(
        manifest, manifest_identity
    )
    if not hmac.compare_digest(
        validated_content_sha256, manifest_content_sha256
    ):
        raise IVEFaceGalleryAcquisitionError(
            "completed acquisition manifest content binding mismatch"
        )
    authorities = _issue_authorities(
        spec=spec,
        manifest=manifest,
        manifest_identity=manifest_identity,
        manifest_content_sha256=manifest_content_sha256,
        job_root=job_root,
    )
    _emit_progress(progress, fraction=1.0, phase="complete")
    return IVEFaceGalleryDedicatedAcquisitionResult(
        authorities=authorities,
        manifest_path=manifest_identity.path,
        manifest_file_sha256=manifest_identity.sha256,
        manifest_content_sha256=manifest_content_sha256,
        reused_source_count=reused_count,
        downloaded_source_count=len(authorities) - reused_count,
    )


def coordinate_ive_face_identity_gallery_acquisition(
    spec: IVEFaceGalleryAcquisitionSpec,
    *,
    destination: Path,
    client: YTDLPClient,
    media_probe: Callable[[Path], MediaInfo] = probe_media,
    progress: ProgressCallback | None = None,
    legacy_materializer_authorities: Sequence[object] = (),
) -> IVEFaceGalleryAcquisitionCoordinatorResult:
    """Acquire six-member sources and return materializer-ready authorities.

    ``legacy_materializer_authorities`` may contain already sealed authorities
    issued from exact ``.invalid.3`` rejection evidence.  They are validated
    and merged by identity; this function never moves or rewrites their paths.
    """

    _validate_production_corpus_song_diversity(spec)
    acquisition = acquire_ive_face_identity_gallery_sources(
        spec,
        destination=destination,
        client=client,
        media_probe=media_probe,
        progress=progress,
    )
    from .ive_face_identity_gallery_materializer import (
        is_sealed_gallery_source_media_authority,
        revalidate_sealed_gallery_source_media_authority,
        seal_closeup_facecam_sources_from_dedicated_acquisition,
    )

    dedicated = seal_closeup_facecam_sources_from_dedicated_acquisition(
        acquisition.authorities, media_probe=media_probe
    )
    legacy = tuple(legacy_materializer_authorities)
    if any(
        not is_sealed_gallery_source_media_authority(item) for item in legacy
    ):
        raise IVEFaceGalleryAcquisitionError(
            "legacy sources must already carry materializer process seals"
        )
    legacy = tuple(
        revalidate_sealed_gallery_source_media_authority(item) for item in legacy
    )
    merged_by_video_id: dict[str, object] = {}
    ownership: dict[str, IVEMember] = {}
    for authority in dedicated:
        provenance = getattr(authority, "provenance")
        video_id = str(getattr(provenance, "video_id"))
        member = _member(getattr(provenance, "target_member"))
        if video_id in merged_by_video_id:
            raise IVEFaceGalleryAcquisitionError(
                "dedicated acquisition produced a duplicate gallery video_id"
            )
        ownership[video_id] = member
        merged_by_video_id[video_id] = authority
    for authority in sorted(
        legacy,
        key=lambda item: (
            str(getattr(getattr(item, "provenance"), "video_id")),
            str(getattr(item, "authority_content_sha256")),
        ),
    ):
        provenance = getattr(authority, "provenance")
        video_id = str(getattr(provenance, "video_id"))
        member = _member(getattr(provenance, "target_member"))
        previous = ownership.get(video_id)
        if previous is not None and previous is not member:
            raise IVEFaceGalleryAcquisitionError(
                "one gallery video cannot be owned by two members"
            )
        ownership[video_id] = member
        # A fresh dedicated receipt has current live metadata and is preferred
        # over a rejection-sealed legacy transcode of the same YouTube video.
        # ``setdefault`` also gives deterministic minimum-content-hash choice
        # when multiple valid legacy authorities refer to one legacy-only ID.
        merged_by_video_id.setdefault(video_id, authority)
    ordered = sorted(
        merged_by_video_id.values(),
        key=lambda item: (
            _MEMBERS.index(
                _member(getattr(getattr(item, "provenance"), "target_member"))
            ),
            str(getattr(getattr(item, "provenance"), "video_id")),
            str(getattr(item, "authority_content_sha256")),
        ),
    )
    return IVEFaceGalleryAcquisitionCoordinatorResult(
        acquisition=acquisition,
        materializer_authorities=tuple(ordered),
    )
