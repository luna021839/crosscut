"""Read-only IVE face-gallery materialization job.

The job deliberately sits outside the edit pipeline.  It rebinds either a
process-sealed selected portrait-fancam acquisition or an exact persistent
portrait-aspect rejection for an official close-up FaceCam to the *current*
regular-file SHA-256.  Quarantined ``.invalid.N`` FaceCam evidence is opened
in place and is never renamed, moved, deleted, or rewritten.

Production sampling is sequential and uses the existing verified YuNet/SFace
path in :mod:`ive_face_identity_gallery`.  Raw frames and working embeddings
remain process-local.  The only persistent vector-bearing output is the
private-local content-addressed gallery artifact; the public job audit is
scalar and digest-only and is not a StageEdit feature source.
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
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import cv2
import numpy as np

from .acquisition import (
    VerifiedMemberAcquisitionIdentityAuthority,
    is_verified_member_acquisition_identity_authority,
)
from .ive import IVEMember
from .ive_face_identity_gallery import (
    CLOSEUP_FACECAM_MEDIA_CONTRACT,
    FACECAM_TITLE_MEMBER_AUTHORIZATION_CONTRACT,
    PORTRAIT_FANCAM_MEDIA_CONTRACT,
    TITLE_MEMBER_AUTHORIZATION_CONTRACT,
    AuthorizedFaceEmbeddingSample,
    IVEFaceGalleryArtifact,
    IVEFaceGalleryConfig,
    IVEFaceGalleryError,
    IVEFaceGallerySourceAuthority,
    OfficialTargetMemberFancamProvenance,
    build_ive_face_identity_gallery,
    extract_authorized_face_sample,
    write_ive_face_identity_gallery_artifact,
)
from .media import MediaInfo, MediaProbeError, probe_media


IVE_FACE_GALLERY_MATERIALIZER_CONTRACT = (
    "ive_private_face_gallery_materializer_job_v1"
)
PORTRAIT_ACQUISITION_AUTHORITY_BASIS = (
    "sealed_verified_member_acquisition_identity_authority_v1"
)
FACECAM_REJECTION_AUTHORITY_BASIS = (
    "exact_persistent_portrait_aspect_rejection_current_metadata_and_media_v1"
)
PUBLIC_AUDIT_CONTRACT = "ive_face_gallery_materializer_scalar_audit_v1"

_GALLERY_SOURCE_AUTHORITY_SEAL = object()
_SHA256 = re.compile(r"[0-9a-f]{64}")
_VIDEO_ID = re.compile(r"[0-9A-Za-z_-]{6,32}")
_QUARANTINE_SUFFIX = re.compile(
    r"(?P<video>[0-9A-Za-z_-]{6,32})\.(?:mp4|mkv|mov|webm|m4v)"
    r"\.invalid(?:\.[1-9][0-9]*)?"
)
_MEMBERS = tuple(IVEMember)
_MAXIMUM_MANIFEST_BYTES = 64 * 1024 * 1024


class IVEFaceGalleryMaterializerError(ValueError):
    """A source authority or sequential materialization gate failed."""


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
            raise IVEFaceGalleryMaterializerError(
                f"acquisition manifest repeats JSON key {key!r}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise IVEFaceGalleryMaterializerError(
        f"acquisition manifest contains non-finite constant {value!r}"
    )


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IVEFaceGalleryMaterializerError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise IVEFaceGalleryMaterializerError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IVEFaceGalleryMaterializerError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise IVEFaceGalleryMaterializerError(
            f"{name} must be finite and non-negative"
        )
    return numeric


def _member(value: object) -> IVEMember:
    if isinstance(value, IVEMember):
        return value
    if not isinstance(value, str):
        raise IVEFaceGalleryMaterializerError("target_member must be an IVE member")
    try:
        return IVEMember.parse(value)
    except ValueError as exc:
        raise IVEFaceGalleryMaterializerError(
            "target_member must be one of the six IVE members"
        ) from exc


@dataclass(frozen=True, slots=True)
class _StableFileIdentity:
    path: Path
    size_bytes: int
    sha256: str
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int


def _stable_regular_file(path: Path, *, maximum_bytes: int | None = None) -> _StableFileIdentity:
    unresolved = Path(path).expanduser()
    try:
        before = os.lstat(unresolved)
    except OSError as exc:
        raise IVEFaceGalleryMaterializerError(
            f"required regular file is unavailable: {unresolved}"
        ) from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise IVEFaceGalleryMaterializerError(
            f"source must be a non-symlink regular file: {unresolved}"
        )
    if maximum_bytes is not None and not 1 <= before.st_size <= maximum_bytes:
        raise IVEFaceGalleryMaterializerError("regular file size is outside its bound")
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
                    raise IVEFaceGalleryMaterializerError(
                        "regular file exceeded its read bound"
                    )
                digest.update(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except IVEFaceGalleryMaterializerError:
        raise
    except OSError as exc:
        raise IVEFaceGalleryMaterializerError(
            f"cannot read regular file safely: {unresolved}"
        ) from exc
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
        raise IVEFaceGalleryMaterializerError("regular file changed while hashing")
    return _StableFileIdentity(
        path=unresolved.resolve(strict=True),
        size_bytes=after.st_size,
        sha256=digest.hexdigest(),
        device=after.st_dev,
        inode=after.st_ino,
        mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,
    )


def _read_strict_manifest(path: Path) -> tuple[Mapping[str, Any], _StableFileIdentity]:
    identity = _stable_regular_file(path, maximum_bytes=_MAXIMUM_MANIFEST_BYTES)
    try:
        encoded = identity.path.read_bytes()
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except IVEFaceGalleryMaterializerError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IVEFaceGalleryMaterializerError(
            "cannot decode acquisition manifest"
        ) from exc
    # Re-read identity after decoding so a writer cannot swap valid bytes
    # between the hashing and parsing phases.
    after = _stable_regular_file(path, maximum_bytes=_MAXIMUM_MANIFEST_BYTES)
    if identity != after:
        raise IVEFaceGalleryMaterializerError(
            "acquisition manifest changed while validating"
        )
    if not isinstance(payload, Mapping) or not all(
        isinstance(key, str) for key in payload
    ):
        raise IVEFaceGalleryMaterializerError(
            "acquisition manifest must be a JSON object"
        )
    return (payload, identity)


def _hashed_envelope_payload(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IVEFaceGalleryMaterializerError(f"{name} must be a hash envelope")
    if set(value) != {"algorithm", "sha256", "payload"}:
        raise IVEFaceGalleryMaterializerError(f"{name} hash envelope is invalid")
    payload = value.get("payload")
    if (
        value.get("algorithm") != "sha256"
        or not isinstance(payload, Mapping)
        or not hmac.compare_digest(
            _sha256(value.get("sha256"), f"{name} SHA-256"),
            _canonical_sha256(payload),
        )
    ):
        raise IVEFaceGalleryMaterializerError(f"{name} hash envelope mismatch")
    return payload


def _probe_video(
    path: Path,
    media_probe: Callable[[Path], MediaInfo],
) -> tuple[int, int, float]:
    try:
        info = media_probe(path)
    except (MediaProbeError, OSError, ValueError) as exc:
        raise IVEFaceGalleryMaterializerError("media probe failed") from exc
    if not isinstance(info, MediaInfo) or not info.video_streams:
        raise IVEFaceGalleryMaterializerError("source has no typed video stream")
    stream = info.video_streams[0]
    width = _positive_integer(stream.width, "video width")
    height = _positive_integer(stream.height, "video height")
    raw_duration = (
        stream.duration_seconds
        if stream.duration_seconds is not None
        else info.duration_seconds
    )
    duration = _finite_nonnegative(raw_duration, "video duration")
    if duration <= 0.0:
        raise IVEFaceGalleryMaterializerError("video duration must be positive")
    return (width, height, duration)


def _authority_unsigned_payload(
    *,
    provenance: OfficialTargetMemberFancamProvenance,
    media_path: Path,
    manifest_path: Path,
    manifest_sha256: str,
    media_sha256: str,
    media_size_bytes: int,
    duration_seconds: float,
    acquisition_authority_basis: str,
    acquisition_evidence_sha256: str,
) -> dict[str, Any]:
    return {
        "contract": IVE_FACE_GALLERY_MATERIALIZER_CONTRACT,
        "provenance_sha256": provenance.provenance_sha256,
        "media_path_sha256": hashlib.sha256(
            os.fspath(media_path).encode("utf-8")
        ).hexdigest(),
        "manifest_path_sha256": hashlib.sha256(
            os.fspath(manifest_path).encode("utf-8")
        ).hexdigest(),
        "manifest_sha256": manifest_sha256,
        "media_sha256": media_sha256,
        "media_size_bytes": media_size_bytes,
        "duration_seconds": duration_seconds,
        "acquisition_authority_basis": acquisition_authority_basis,
        "acquisition_evidence_sha256": acquisition_evidence_sha256,
    }


@dataclass(frozen=True, slots=True)
class SealedGallerySourceMediaAuthority:
    """Process-local authority issued only after exact acquisition rebinding."""

    provenance: OfficialTargetMemberFancamProvenance
    media_path: Path
    manifest_path: Path
    manifest_sha256: str
    media_sha256: str
    media_size_bytes: int
    duration_seconds: float
    acquisition_authority_basis: str
    acquisition_evidence_sha256: str
    authority_content_sha256: str
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _GALLERY_SOURCE_AUTHORITY_SEAL:
            raise TypeError("gallery source authority must be issued by this module")
        if not isinstance(self.provenance, OfficialTargetMemberFancamProvenance):
            raise IVEFaceGalleryMaterializerError("authority provenance must be typed")
        media_path = Path(self.media_path).expanduser().resolve(strict=True)
        manifest_path = Path(self.manifest_path).expanduser().resolve(strict=True)
        object.__setattr__(self, "media_path", media_path)
        object.__setattr__(self, "manifest_path", manifest_path)
        for name in (
            "manifest_sha256",
            "media_sha256",
            "acquisition_evidence_sha256",
            "authority_content_sha256",
        ):
            _sha256(getattr(self, name), name)
        _positive_integer(self.media_size_bytes, "media_size_bytes")
        duration = _finite_nonnegative(self.duration_seconds, "duration_seconds")
        if duration <= 0.0:
            raise IVEFaceGalleryMaterializerError("duration_seconds must be positive")
        object.__setattr__(self, "duration_seconds", duration)
        if self.acquisition_authority_basis not in {
            PORTRAIT_ACQUISITION_AUTHORITY_BASIS,
            FACECAM_REJECTION_AUTHORITY_BASIS,
        }:
            raise IVEFaceGalleryMaterializerError(
                "acquisition authority basis is invalid"
            )
        current_media = _stable_regular_file(media_path)
        current_manifest = _stable_regular_file(
            manifest_path, maximum_bytes=_MAXIMUM_MANIFEST_BYTES
        )
        if not (
            current_media.size_bytes == self.media_size_bytes
            and hmac.compare_digest(current_media.sha256, self.media_sha256)
            and hmac.compare_digest(current_manifest.sha256, self.manifest_sha256)
            and self.provenance.media_size_bytes == self.media_size_bytes
            and hmac.compare_digest(
                self.provenance.media_content_sha256, self.media_sha256
            )
        ):
            raise IVEFaceGalleryMaterializerError(
                "sealed gallery authority no longer matches current files"
            )
        expected = _canonical_sha256(
            _authority_unsigned_payload(
                provenance=self.provenance,
                media_path=media_path,
                manifest_path=manifest_path,
                manifest_sha256=self.manifest_sha256,
                media_sha256=self.media_sha256,
                media_size_bytes=self.media_size_bytes,
                duration_seconds=self.duration_seconds,
                acquisition_authority_basis=self.acquisition_authority_basis,
                acquisition_evidence_sha256=self.acquisition_evidence_sha256,
            )
        )
        if not hmac.compare_digest(expected, self.authority_content_sha256):
            raise IVEFaceGalleryMaterializerError(
                "gallery source authority content seal mismatch"
            )


def is_sealed_gallery_source_media_authority(value: object) -> bool:
    return bool(
        isinstance(value, SealedGallerySourceMediaAuthority)
        and value._seal is _GALLERY_SOURCE_AUTHORITY_SEAL
    )


def _issue_gallery_authority(
    *,
    provenance: OfficialTargetMemberFancamProvenance,
    media_identity: _StableFileIdentity,
    manifest_identity: _StableFileIdentity,
    duration_seconds: float,
    acquisition_authority_basis: str,
    acquisition_evidence_sha256: str,
) -> SealedGallerySourceMediaAuthority:
    unsigned = _authority_unsigned_payload(
        provenance=provenance,
        media_path=media_identity.path,
        manifest_path=manifest_identity.path,
        manifest_sha256=manifest_identity.sha256,
        media_sha256=media_identity.sha256,
        media_size_bytes=media_identity.size_bytes,
        duration_seconds=duration_seconds,
        acquisition_authority_basis=acquisition_authority_basis,
        acquisition_evidence_sha256=acquisition_evidence_sha256,
    )
    return SealedGallerySourceMediaAuthority(
        provenance=provenance,
        media_path=media_identity.path,
        manifest_path=manifest_identity.path,
        manifest_sha256=manifest_identity.sha256,
        media_sha256=media_identity.sha256,
        media_size_bytes=media_identity.size_bytes,
        duration_seconds=duration_seconds,
        acquisition_authority_basis=acquisition_authority_basis,
        acquisition_evidence_sha256=acquisition_evidence_sha256,
        authority_content_sha256=_canonical_sha256(unsigned),
        _seal=_GALLERY_SOURCE_AUTHORITY_SEAL,
    )


def seal_portrait_gallery_sources_from_acquisition(
    authority: VerifiedMemberAcquisitionIdentityAuthority,
    *,
    media_probe: Callable[[Path], MediaInfo] = probe_media,
) -> tuple[SealedGallerySourceMediaAuthority, ...]:
    """Rebind a genuine process-sealed portrait acquisition to current files."""

    if not is_verified_member_acquisition_identity_authority(authority):
        raise IVEFaceGalleryMaterializerError(
            "portrait source requires sealed member acquisition authority"
        )
    manifest, manifest_identity = _read_strict_manifest(Path(authority.manifest_path))
    if not (
        authority.group == "IVE"
        and manifest.get("group") == "IVE"
        and manifest.get("song") == authority.song
        and str(manifest.get("edit_mode", "")).lower() == "member"
        and manifest.get("target_member") == authority.target_member
        and manifest.get("member_portrait_aspect_contract_version") == 1
        and manifest.get("content_integrity_contract_version") == 1
        and manifest_identity.size_bytes == authority.manifest_size_bytes
        and hmac.compare_digest(
            manifest_identity.sha256, authority.manifest_sha256
        )
    ):
        raise IVEFaceGalleryMaterializerError(
            "sealed portrait acquisition manifest scope mismatch"
        )
    selected = manifest.get("selected")
    if not isinstance(selected, list) or len(selected) != len(authority.sources):
        raise IVEFaceGalleryMaterializerError(
            "portrait acquisition selected-source coverage mismatch"
        )
    target = _member(authority.target_member)
    issued: list[SealedGallerySourceMediaAuthority] = []
    for source_identity in authority.sources:
        if not 0 <= source_identity.source_index < len(selected):
            raise IVEFaceGalleryMaterializerError(
                "portrait acquisition source index is invalid"
            )
        raw = selected[source_identity.source_index]
        if not isinstance(raw, Mapping):
            raise IVEFaceGalleryMaterializerError(
                "portrait selected source must be an object"
            )
        integrity = raw.get("content_integrity")
        if not isinstance(integrity, Mapping):
            raise IVEFaceGalleryMaterializerError(
                "portrait selected source lacks content integrity"
            )
        source_path = Path(source_identity.source_path).expanduser()
        current = _stable_regular_file(source_path)
        try:
            local_path_matches = (
                isinstance(raw.get("local_path"), str)
                and Path(str(raw["local_path"])).expanduser().resolve(strict=True)
                == current.path
            )
        except OSError:
            local_path_matches = False
        if not (
            raw.get("source_index") == source_identity.source_index
            and raw.get("video_id") == source_identity.video_id
            and raw.get("broadcast_date") == source_identity.broadcast_date
            and raw.get("broadcast_program") == source_identity.broadcast_program
            and local_path_matches
            and integrity.get("contract_version") == 1
            and integrity.get("algorithm") == "sha256"
            and integrity.get("size_bytes") == current.size_bytes
            and integrity.get("sha256") == current.sha256
            and source_identity.source_size_bytes == current.size_bytes
            and source_identity.source_sha256 == current.sha256
        ):
            raise IVEFaceGalleryMaterializerError(
                "portrait selected source no longer matches sealed acquisition"
            )
        title = raw.get("title")
        channel_id = raw.get("channel_id")
        if not isinstance(title, str) or not isinstance(channel_id, str):
            raise IVEFaceGalleryMaterializerError(
                "portrait source lacks official title/channel metadata"
            )
        width, height, duration = _probe_video(current.path, media_probe)
        selected_metadata_binding = _canonical_sha256(
            {
                "basis": PORTRAIT_ACQUISITION_AUTHORITY_BASIS,
                "manifest_sha256": manifest_identity.sha256,
                "source_index": source_identity.source_index,
                "selected_entry_sha256": _canonical_sha256(raw),
                "source_sha256": current.sha256,
            }
        )
        try:
            provenance = OfficialTargetMemberFancamProvenance(
                song=authority.song,
                target_member=target,
                video_id=source_identity.video_id,
                title=title,
                broadcast_date=source_identity.broadcast_date,
                broadcast_program=source_identity.broadcast_program,
                channel_id=channel_id,
                metadata_authority_sha256=selected_metadata_binding,
                acquisition_manifest_sha256=manifest_identity.sha256,
                media_content_sha256=current.sha256,
                media_size_bytes=current.size_bytes,
                media_width=width,
                media_height=height,
                source_authority=(
                    IVEFaceGallerySourceAuthority.OFFICIAL_TARGET_MEMBER_PORTRAIT_FANCAM
                ),
                source_media_contract=PORTRAIT_FANCAM_MEDIA_CONTRACT,
                title_member_authorization_contract=(
                    TITLE_MEMBER_AUTHORIZATION_CONTRACT
                ),
            )
        except IVEFaceGalleryError as exc:
            raise IVEFaceGalleryMaterializerError(str(exc)) from exc
        issued.append(
            _issue_gallery_authority(
                provenance=provenance,
                media_identity=current,
                manifest_identity=manifest_identity,
                duration_seconds=duration,
                acquisition_authority_basis=PORTRAIT_ACQUISITION_AUTHORITY_BASIS,
                acquisition_evidence_sha256=selected_metadata_binding,
            )
        )
    return tuple(
        sorted(issued, key=lambda item: item.authority_content_sha256)
    )


def _quarantine_path_from_evidence(
    *, manifest_path: Path, video_id: str, artifact: object
) -> Path:
    if not isinstance(artifact, Mapping):
        raise IVEFaceGalleryMaterializerError(
            "FaceCam rejection lacks quarantine artifact"
        )
    relative_raw = artifact.get("relative_path")
    if not isinstance(relative_raw, str):
        raise IVEFaceGalleryMaterializerError(
            "FaceCam quarantine path must be relative"
        )
    relative = Path(relative_raw)
    if (
        relative.is_absolute()
        or len(relative.parts) != 2
        or relative.parts[0] != "videos"
        or _QUARANTINE_SUFFIX.fullmatch(relative.parts[1]) is None
        or _QUARANTINE_SUFFIX.fullmatch(relative.parts[1]).group("video")
        != video_id
    ):
        raise IVEFaceGalleryMaterializerError(
            "FaceCam quarantine path contract mismatch"
        )
    videos = manifest_path.parent / "videos"
    if videos.is_symlink() or not videos.is_dir():
        raise IVEFaceGalleryMaterializerError(
            "FaceCam quarantine directory is not a safe directory"
        )
    candidate = manifest_path.parent / relative
    identity = _stable_regular_file(candidate)
    if identity.path.parent != videos.resolve(strict=True):
        raise IVEFaceGalleryMaterializerError(
            "FaceCam quarantine file escaped its acquisition cache"
        )
    return identity.path


def seal_closeup_facecam_sources_from_rejection_manifest(
    acquisition_authority: VerifiedMemberAcquisitionIdentityAuthority,
    *,
    video_ids: Sequence[str],
    media_probe: Callable[[Path], MediaInfo] = probe_media,
) -> tuple[SealedGallerySourceMediaAuthority, ...]:
    """Issue read-only FaceCam authorities from exact persistent evidence.

    Only portrait-aspect rejections with an official FaceCam/얼빡 title are
    eligible.  An ordinary landscape fancam/fullcam remains rejected even if
    its bytes and rejection envelope are otherwise valid.
    """

    if not is_verified_member_acquisition_identity_authority(acquisition_authority):
        raise IVEFaceGalleryMaterializerError(
            "FaceCam source requires sealed member acquisition authority"
        )
    requested = tuple(video_ids)
    if (
        not requested
        or len(requested) != len(set(requested))
        or any(not isinstance(item, str) or _VIDEO_ID.fullmatch(item) is None for item in requested)
    ):
        raise IVEFaceGalleryMaterializerError(
            "FaceCam video_ids must be unique explicit acquisition IDs"
        )
    manifest, manifest_identity = _read_strict_manifest(
        Path(acquisition_authority.manifest_path)
    )
    if not (
        manifest_identity.size_bytes == acquisition_authority.manifest_size_bytes
        and hmac.compare_digest(
            manifest_identity.sha256, acquisition_authority.manifest_sha256
        )
    ):
        raise IVEFaceGalleryMaterializerError(
            "FaceCam acquisition manifest no longer matches sealed authority"
        )
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("content_integrity_contract_version") == 1
        and manifest.get("persistent_media_contract_rejection_contract_version") == 1
        and manifest.get("member_portrait_aspect_contract_version") == 1
        and manifest.get("group") == "IVE"
        and acquisition_authority.group == "IVE"
        and manifest.get("song") == acquisition_authority.song
        and str(manifest.get("edit_mode", "")).lower() == "member"
        and manifest.get("aspect_ratio") == "9:16"
        and isinstance(manifest.get("song"), str)
        and bool(str(manifest.get("song")).strip())
        and isinstance(manifest.get("target_member"), str)
        and manifest.get("target_member") == acquisition_authority.target_member
    ):
        raise IVEFaceGalleryMaterializerError(
            "FaceCam rejection manifest scope is not an IVE member acquisition"
        )
    target = _member(manifest["target_member"])
    entries = manifest.get("persistent_media_contract_rejections")
    if not isinstance(entries, list):
        raise IVEFaceGalleryMaterializerError(
            "FaceCam manifest lacks persistent rejection evidence"
        )
    issued: list[SealedGallerySourceMediaAuthority] = []
    for requested_id in requested:
        matches = [
            item
            for item in entries
            if isinstance(item, Mapping) and item.get("video_id") == requested_id
        ]
        if len(matches) != 1:
            raise IVEFaceGalleryMaterializerError(
                "FaceCam video ID needs exactly one persistent rejection"
            )
        evidence = matches[0]
        if not (
            evidence.get("schema_version") == 1
            and evidence.get("evidence_origin")
            in {
                "downloaded_media_contract_failure",
                "legacy_candidate_rejection_revalidated",
            }
            and evidence.get("hard_rejection_reason")
            == "automatic_member_portrait_aspect_ratio_mismatch"
        ):
            raise IVEFaceGalleryMaterializerError(
                "FaceCam source is not an exact portrait-aspect rejection"
            )
        metadata_envelope = evidence.get("candidate_media_metadata_identity")
        metadata = _hashed_envelope_payload(
            metadata_envelope, "FaceCam candidate metadata"
        )
        request = _hashed_envelope_payload(
            evidence.get("request_media_contract_identity"),
            "FaceCam acquisition request",
        )
        observations = metadata.get("current_metadata_observation_sha256")
        if not (
            metadata.get("schema_version") == 1
            and metadata.get("video_id") == requested_id
            and isinstance(metadata.get("title"), str)
            and isinstance(metadata.get("channel_id"), str)
            and isinstance(metadata.get("broadcast_date"), str)
            and isinstance(metadata.get("broadcast_program"), str)
            and metadata.get("official_channel_signal") not in {None, "none"}
            and isinstance(observations, list)
            and observations
            and all(
                isinstance(item, str) and _SHA256.fullmatch(item) is not None
                for item in observations
            )
            and request.get("schema_version") == 1
            and request.get("group") == "IVE"
            and request.get("song") == manifest.get("song")
            and request.get("source_mode") == "automatic"
            and request.get("edit_mode") == "member"
            and request.get("aspect_ratio") == "9:16"
            and request.get("target_member") == target.value
            and request.get("member_portrait_aspect_contract_version") == 1
        ):
            raise IVEFaceGalleryMaterializerError(
                "FaceCam metadata/request authority is incomplete or out of scope"
            )
        diagnostics = evidence.get("diagnostics")
        diagnostics_sha256 = evidence.get("diagnostics_sha256")
        if not (
            isinstance(diagnostics, Mapping)
            and diagnostics.get("passed") is False
            and diagnostics.get("hard_rejection_reason")
            == "automatic_member_portrait_aspect_ratio_mismatch"
            and hmac.compare_digest(
                _sha256(diagnostics_sha256, "FaceCam diagnostics SHA-256"),
                _canonical_sha256(diagnostics),
            )
        ):
            raise IVEFaceGalleryMaterializerError(
                "FaceCam rejection diagnostics do not reproduce"
            )
        media_path = _quarantine_path_from_evidence(
            manifest_path=manifest_identity.path,
            video_id=requested_id,
            artifact=evidence.get("quarantine_artifact"),
        )
        current = _stable_regular_file(media_path)
        quarantine = evidence.get("quarantine_artifact")
        assert isinstance(quarantine, Mapping)
        content = quarantine.get("content_integrity")
        if not (
            isinstance(content, Mapping)
            and content.get("contract_version") == 1
            and content.get("algorithm") == "sha256"
            and content.get("size_bytes") == current.size_bytes
            and content.get("sha256") == current.sha256
        ):
            raise IVEFaceGalleryMaterializerError(
                "FaceCam quarantine content no longer matches acquisition evidence"
            )
        width, height, duration = _probe_video(current.path, media_probe)
        if not (
            diagnostics.get("width") == width
            and diagnostics.get("height") == height
            and isinstance(diagnostics.get("actual_width_over_height"), (int, float))
            and math.isclose(
                float(diagnostics["actual_width_over_height"]),
                width / height,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise IVEFaceGalleryMaterializerError(
                "current FaceCam dimensions do not reproduce rejection evidence"
            )
        try:
            provenance = OfficialTargetMemberFancamProvenance(
                song=str(manifest["song"]),
                target_member=target,
                video_id=requested_id,
                title=str(metadata["title"]),
                broadcast_date=str(metadata["broadcast_date"]),
                broadcast_program=str(metadata["broadcast_program"]),
                channel_id=str(metadata["channel_id"]),
                metadata_authority_sha256=str(metadata_envelope["sha256"]),
                acquisition_manifest_sha256=manifest_identity.sha256,
                media_content_sha256=current.sha256,
                media_size_bytes=current.size_bytes,
                media_width=width,
                media_height=height,
                source_authority=(
                    IVEFaceGallerySourceAuthority.OFFICIAL_TARGET_MEMBER_CLOSEUP_FACECAM
                ),
                source_media_contract=CLOSEUP_FACECAM_MEDIA_CONTRACT,
                title_member_authorization_contract=(
                    FACECAM_TITLE_MEMBER_AUTHORIZATION_CONTRACT
                ),
            )
        except IVEFaceGalleryError as exc:
            raise IVEFaceGalleryMaterializerError(str(exc)) from exc
        evidence_digest = _canonical_sha256(evidence)
        issued.append(
            _issue_gallery_authority(
                provenance=provenance,
                media_identity=current,
                manifest_identity=manifest_identity,
                duration_seconds=duration,
                acquisition_authority_basis=FACECAM_REJECTION_AUTHORITY_BASIS,
                acquisition_evidence_sha256=evidence_digest,
            )
        )
    return tuple(
        sorted(issued, key=lambda item: item.authority_content_sha256)
    )


@dataclass(frozen=True, slots=True)
class IVEFaceGalleryMaterializerConfig:
    start_margin_seconds: float = 4.0
    end_margin_seconds: float = 4.0
    minimum_sample_interval_seconds: float = 6.0
    failed_attempt_retry_seconds: float = 0.50
    minimum_samples_per_source: int = 3
    maximum_samples_per_source: int = 10
    maximum_scanned_frames_per_source: int = 1_000_000

    def __post_init__(self) -> None:
        for name in (
            "start_margin_seconds",
            "end_margin_seconds",
            "minimum_sample_interval_seconds",
            "failed_attempt_retry_seconds",
        ):
            value = _finite_nonnegative(getattr(self, name), name)
            if name in {
                "minimum_sample_interval_seconds",
                "failed_attempt_retry_seconds",
            } and value <= 0.0:
                raise IVEFaceGalleryMaterializerError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in (
            "minimum_samples_per_source",
            "maximum_samples_per_source",
            "maximum_scanned_frames_per_source",
        ):
            object.__setattr__(
                self, name, _positive_integer(getattr(self, name), name)
            )
        if self.maximum_samples_per_source < self.minimum_samples_per_source:
            raise IVEFaceGalleryMaterializerError(
                "maximum samples cannot be below minimum samples"
            )


FrameIteratorFactory = Callable[[Path], Iterable[tuple[int, float, np.ndarray]]]


def sequential_video_frames(path: Path) -> Iterator[tuple[int, float, np.ndarray]]:
    """Decode frames in source order without seeking or parallel reordering."""

    capture = cv2.VideoCapture(os.fspath(path))
    if not capture.isOpened():
        capture.release()
        raise IVEFaceGalleryMaterializerError("cannot open gallery source video")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0.0:
        fps = 30.0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if not isinstance(frame, np.ndarray) or frame.size == 0:
                raise IVEFaceGalleryMaterializerError(
                    "video decoder returned an invalid frame"
                )
            raw_milliseconds = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamp = (
                raw_milliseconds / 1000.0
                if math.isfinite(raw_milliseconds) and raw_milliseconds >= 0.0
                else frame_index / fps
            )
            yield (frame_index, timestamp, frame)
            frame_index += 1
    finally:
        capture.release()


@dataclass(frozen=True, slots=True)
class GallerySourceMaterializationAudit:
    member: IVEMember
    source_authority: IVEFaceGallerySourceAuthority
    source_authority_sha256: str
    media_sha256: str
    sampled_face_count: int
    failed_face_attempt_count: int
    scanned_frame_count: int
    first_sample_seconds: float
    last_sample_seconds: float

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "member": self.member.value,
            "source_authority": self.source_authority.value,
            "source_authority_sha256": self.source_authority_sha256,
            "media_sha256": self.media_sha256,
            "sampled_face_count": self.sampled_face_count,
            "failed_face_attempt_count": self.failed_face_attempt_count,
            "scanned_frame_count": self.scanned_frame_count,
            "first_sample_seconds": self.first_sample_seconds,
            "last_sample_seconds": self.last_sample_seconds,
        }


@dataclass(frozen=True, slots=True)
class IVEFaceGalleryMaterializationResult:
    artifact: IVEFaceGalleryArtifact
    source_audits: tuple[GallerySourceMaterializationAudit, ...]
    public_audit_content_sha256: str

    def to_public_audit_dict(self) -> dict[str, Any]:
        payload = {
            "contract": PUBLIC_AUDIT_CONTRACT,
            "gallery_content_sha256": self.artifact.content_sha256,
            "source_count": len(self.source_audits),
            "sources": [item.to_public_dict() for item in self.source_audits],
            "privacy": {
                "contains_raw_frames": False,
                "contains_identity_vectors": False,
                "contains_local_media_paths": False,
                "stageedit_feature_source": False,
                "private_gallery_artifact_is_separate": True,
            },
            "public_audit_content_sha256": None,
        }
        payload["public_audit_content_sha256"] = _canonical_sha256(payload)
        return payload

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, IVEFaceGalleryArtifact):
            raise IVEFaceGalleryMaterializerError("result artifact must be typed")
        if not self.source_audits:
            raise IVEFaceGalleryMaterializerError("result needs source audits")
        expected = self.to_public_audit_dict()["public_audit_content_sha256"]
        if not hmac.compare_digest(
            _sha256(
                self.public_audit_content_sha256,
                "public_audit_content_sha256",
            ),
            expected,
        ):
            raise IVEFaceGalleryMaterializerError(
                "public audit content SHA-256 mismatch"
            )


def _assert_authority_current(authority: SealedGallerySourceMediaAuthority) -> None:
    if not is_sealed_gallery_source_media_authority(authority):
        raise IVEFaceGalleryMaterializerError(
            "materialization needs sealed gallery source authorities"
        )
    current_media = _stable_regular_file(authority.media_path)
    current_manifest = _stable_regular_file(
        authority.manifest_path, maximum_bytes=_MAXIMUM_MANIFEST_BYTES
    )
    if not (
        current_media.size_bytes == authority.media_size_bytes
        and current_media.sha256 == authority.media_sha256
        and current_manifest.sha256 == authority.manifest_sha256
    ):
        raise IVEFaceGalleryMaterializerError(
            "source authority changed after sealing"
        )


def _frame_observation_sha256(
    authority: SealedGallerySourceMediaAuthority,
    *,
    frame_index: int,
    timestamp_seconds: float,
    frame: np.ndarray,
) -> str:
    contiguous = np.ascontiguousarray(frame)
    raw_frame_sha256 = hashlib.sha256(contiguous.tobytes()).hexdigest()
    return _canonical_sha256(
        {
            "authority_content_sha256": authority.authority_content_sha256,
            "frame_index": frame_index,
            "timestamp_seconds": timestamp_seconds,
            "raw_frame_sha256": raw_frame_sha256,
            "raw_frame_persisted": False,
        }
    )


def materialize_ive_face_identity_gallery_job(
    authorities: Sequence[SealedGallerySourceMediaAuthority],
    *,
    artifact_directory: Path,
    config: IVEFaceGalleryMaterializerConfig | None = None,
    gallery_config: IVEFaceGalleryConfig | None = None,
    frame_iterator_factory: FrameIteratorFactory = sequential_video_frames,
) -> IVEFaceGalleryMaterializationResult:
    """Sequentially build one complete six-member private gallery artifact."""

    policy = config or IVEFaceGalleryMaterializerConfig()
    gallery_policy = gallery_config or IVEFaceGalleryConfig()
    if policy.minimum_samples_per_source < gallery_policy.minimum_samples_per_video:
        raise IVEFaceGalleryMaterializerError(
            "materializer sample floor cannot be below gallery video floor"
        )
    if policy.maximum_samples_per_source > gallery_policy.maximum_samples_per_video:
        raise IVEFaceGalleryMaterializerError(
            "materializer sample cap cannot exceed gallery video cap"
        )
    if policy.minimum_sample_interval_seconds < (
        gallery_policy.minimum_temporal_separation_seconds
    ):
        raise IVEFaceGalleryMaterializerError(
            "materializer temporal interval cannot weaken gallery interval"
        )
    provided = tuple(authorities)
    if not provided or not all(
        is_sealed_gallery_source_media_authority(item) for item in provided
    ):
        raise IVEFaceGalleryMaterializerError(
            "job requires typed sealed source authorities"
        )
    if len({item.authority_content_sha256 for item in provided}) != len(provided):
        raise IVEFaceGalleryMaterializerError("source authority is duplicated")
    if len({item.media_sha256 for item in provided}) != len(provided):
        raise IVEFaceGalleryMaterializerError(
            "one media file cannot seed multiple gallery sources"
        )
    ordered = tuple(
        sorted(
            provided,
            key=lambda item: (
                _MEMBERS.index(item.provenance.target_member),
                item.authority_content_sha256,
            ),
        )
    )
    for member in _MEMBERS:
        count = sum(item.provenance.target_member is member for item in ordered)
        if count < gallery_policy.minimum_videos_per_member:
            raise IVEFaceGalleryMaterializerError(
                f"{member.value} lacks independent sealed source authorities"
            )

    all_samples: list[AuthorizedFaceEmbeddingSample] = []
    audits: list[GallerySourceMaterializationAudit] = []
    for authority in ordered:
        _assert_authority_current(authority)
        if authority.duration_seconds <= (
            policy.start_margin_seconds + policy.end_margin_seconds
        ):
            raise IVEFaceGalleryMaterializerError(
                "source is too short for bounded gallery sampling"
            )
        next_attempt_seconds = policy.start_margin_seconds
        last_frame_index = -1
        last_timestamp = -1.0
        scanned = 0
        failed_attempts = 0
        source_samples: list[AuthorizedFaceEmbeddingSample] = []
        iterator = iter(frame_iterator_factory(authority.media_path))
        try:
            for raw in iterator:
                if (
                    not isinstance(raw, tuple)
                    or len(raw) != 3
                    or isinstance(raw[0], bool)
                    or not isinstance(raw[0], int)
                ):
                    raise IVEFaceGalleryMaterializerError(
                        "frame iterator must yield typed sequential triples"
                    )
                frame_index, raw_timestamp, frame = raw
                timestamp = _finite_nonnegative(raw_timestamp, "frame timestamp")
                if frame_index <= last_frame_index or timestamp < last_timestamp:
                    raise IVEFaceGalleryMaterializerError(
                        "frame iterator is not sequential and causal"
                    )
                last_frame_index = frame_index
                last_timestamp = timestamp
                scanned += 1
                if scanned > policy.maximum_scanned_frames_per_source:
                    raise IVEFaceGalleryMaterializerError(
                        "source exceeded sequential frame scan bound"
                    )
                if timestamp > authority.duration_seconds - policy.end_margin_seconds:
                    break
                if timestamp + 1e-9 < next_attempt_seconds:
                    continue
                if not isinstance(frame, np.ndarray) or frame.size == 0:
                    raise IVEFaceGalleryMaterializerError(
                        "frame iterator returned an invalid image"
                    )
                observation_sha256 = _frame_observation_sha256(
                    authority,
                    frame_index=frame_index,
                    timestamp_seconds=timestamp,
                    frame=frame,
                )
                try:
                    sample = extract_authorized_face_sample(
                        frame,
                        provenance=authority.provenance,
                        timestamp_seconds=timestamp,
                        frame_index=frame_index,
                        face_observation_sha256=observation_sha256,
                    )
                except IVEFaceGalleryError:
                    failed_attempts += 1
                    next_attempt_seconds = (
                        timestamp + policy.failed_attempt_retry_seconds
                    )
                    continue
                finally:
                    del frame
                if not (
                    isinstance(sample, AuthorizedFaceEmbeddingSample)
                    and sample.provenance == authority.provenance
                    and sample.frame_index == frame_index
                    and math.isclose(
                        sample.timestamp_seconds,
                        timestamp,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    and sample.face_observation_sha256 == observation_sha256
                ):
                    raise IVEFaceGalleryMaterializerError(
                        "face extractor returned unbound sample evidence"
                    )
                source_samples.append(sample)
                next_attempt_seconds = (
                    timestamp + policy.minimum_sample_interval_seconds
                )
                if len(source_samples) >= policy.maximum_samples_per_source:
                    break
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        if len(source_samples) < policy.minimum_samples_per_source:
            raise IVEFaceGalleryMaterializerError(
                "source did not yield enough verified single-face samples"
            )
        _assert_authority_current(authority)
        all_samples.extend(source_samples)
        audits.append(
            GallerySourceMaterializationAudit(
                member=authority.provenance.target_member,
                source_authority=authority.provenance.source_authority,
                source_authority_sha256=authority.authority_content_sha256,
                media_sha256=authority.media_sha256,
                sampled_face_count=len(source_samples),
                failed_face_attempt_count=failed_attempts,
                scanned_frame_count=scanned,
                first_sample_seconds=source_samples[0].timestamp_seconds,
                last_sample_seconds=source_samples[-1].timestamp_seconds,
            )
        )
    try:
        gallery = build_ive_face_identity_gallery(
            all_samples, config=gallery_policy
        )
        artifact = write_ive_face_identity_gallery_artifact(
            artifact_directory, gallery
        )
    except IVEFaceGalleryError as exc:
        raise IVEFaceGalleryMaterializerError(str(exc)) from exc
    # Drop per-frame vectors from the job's working collection.  The returned
    # artifact object contains only the intentionally private gallery model.
    all_samples.clear()
    unsigned_audit = {
        "contract": PUBLIC_AUDIT_CONTRACT,
        "gallery_content_sha256": artifact.content_sha256,
        "source_count": len(audits),
        "sources": [item.to_public_dict() for item in audits],
        "privacy": {
            "contains_raw_frames": False,
            "contains_identity_vectors": False,
            "contains_local_media_paths": False,
            "stageedit_feature_source": False,
            "private_gallery_artifact_is_separate": True,
        },
        "public_audit_content_sha256": None,
    }
    return IVEFaceGalleryMaterializationResult(
        artifact=artifact,
        source_audits=tuple(audits),
        public_audit_content_sha256=_canonical_sha256(unsigned_audit),
    )


__all__ = [
    "FACECAM_REJECTION_AUTHORITY_BASIS",
    "IVE_FACE_GALLERY_MATERIALIZER_CONTRACT",
    "PORTRAIT_ACQUISITION_AUTHORITY_BASIS",
    "PUBLIC_AUDIT_CONTRACT",
    "GallerySourceMaterializationAudit",
    "IVEFaceGalleryMaterializationResult",
    "IVEFaceGalleryMaterializerConfig",
    "IVEFaceGalleryMaterializerError",
    "SealedGallerySourceMediaAuthority",
    "is_sealed_gallery_source_media_authority",
    "materialize_ive_face_identity_gallery_job",
    "seal_closeup_facecam_sources_from_rejection_manifest",
    "seal_portrait_gallery_sources_from_acquisition",
    "sequential_video_frames",
]
