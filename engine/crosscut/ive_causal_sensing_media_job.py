"""Real-media, privacy-minimal IVE causal-sensing artifact materializer.

The job in this module is the production bridge between synchronized IVE
source media and the three scalar artifacts consumed by StageEdit:

* :mod:`causal_shooting_grammar`;
* :mod:`member_state_graph`; and
* :mod:`ive_performer_state`.

Every source is content-hash verified before and after a single forward decode
pass.  Only the current decision frame and its immediate following frame are
resident at once, and only one source is open at a time.  YuNet geometry and
SFace query vectors are transient.  A query is admitted through the existing
private IVE gallery's absolute-similarity and top-two-margin gates; every miss,
ambiguous result, or unavailable face remains explicit ``UNKNOWN``.

The emitted artifacts contain normalized scalars, opaque identifiers, and
content hashes.  They never contain source paths, frames, visual descriptors,
or query embeddings.  ATTITUDE is rejected before filesystem, model, gallery,
or frame-provider access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import errno
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence
import uuid

import cv2

from .causal_shooting_grammar import (
    ShootingGrammarObservation,
    ShootingGrammarProvenance,
    build_causal_shooting_grammar_artifact,
    write_causal_shooting_grammar_artifact,
)
from .face_identity import SFaceEmbedder, sface_model_asset_status
from .ive import EditMode, IVEMember
from .ive_face_identity_gallery import (
    IVEFaceGalleryArtifact,
    adapt_gallery_identity_for_member_state,
    classify_open_set_ive_identity,
    load_ive_face_identity_gallery_artifact,
)
from .ive_performer_state import (
    IVEPerformerStateProvenance,
    IVEPerformerStateSourceProvenance,
    build_ive_performer_state_artifact,
    write_ive_performer_state_artifact,
)
from .member_state_graph import (
    CausalMemberObservation,
    MemberProbabilityMass,
    MemberStateArtifactProvenance,
    MemberStateSourceProvenance,
    NormalizedBBox,
    build_member_state_graph,
    write_member_state_graph_artifact,
)
from .reference_evidence_builder import ReferenceSourceClock
from .stage_edit_feature_materializer import StageEditCausalSensingArtifacts
from .timeline import ShotType
from .visual_score import (
    FrameVisualDescriptor,
    describe_frame,
    visual_analyzer_fingerprint_sha256,
    yunet_model_asset_status,
)


IVE_CAUSAL_SENSING_MEDIA_JOB_SCHEMA_VERSION = 1
IVE_CAUSAL_SENSING_MEDIA_JOB_CONTRACT = (
    "ive_real_media_causal_sensing_triple_v1"
)
IVE_CAUSAL_SENSING_FRAME_CONTRACT = (
    "exact_master_clock_current_plus_immediate_following_native_frame_"
    "nondecreasing_reuse_v1"
)
IVE_CAUSAL_SENSING_IDENTITY_CONTRACT = (
    "verified_yunet_sface_private_gallery_open_set_unknown_on_doubt_v1"
)
IVE_CAUSAL_SENSING_MEMBER_FRAME_CONTRACT = (
    "member_graph_source_local_decision_ordinal_with_native_decode_manifest_v1"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SOURCE_ID = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._:-]{0,159}")
_ATTITUDE_TOKENS = frozenset({"attitude", "애티튜드"})
_MAXIMUM_SOURCE_COUNT = 32
_MAXIMUM_DECISION_COUNT = 100_000
# Registered StageEdit bundles admit sequences up to 2,048 tokens and as many
# as 24 synchronized candidates.  Keep that complete finite production grid
# admissible; frame residency remains one source and one adjacent pair at a
# time, while only scalar observations accumulate across the request.
_MAXIMUM_SOURCE_TIME_CELLS = 50_000
_MAXIMUM_MEMBER_OBSERVATIONS_PER_FRAME = 12
_FILE_BLOCK_BYTES = 4 * 1024 * 1024
_FLOAT_TOLERANCE = 1e-12
_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = frozenset(
    {
        errno.EINVAL,
        errno.ENOTSUP,
        getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
        getattr(errno, "ENOTTY", errno.EINVAL),
    }
)


class IVECausalSensingMediaJobError(ValueError):
    """Fail-closed error for invalid authority, media, or derived evidence."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IVECausalSensingMediaJobError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IVECausalSensingMediaJobError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise IVECausalSensingMediaJobError(f"{name} must be finite")
    return 0.0 if number == 0.0 else number


def _unit(value: object, name: str) -> float:
    number = _finite(value, name)
    if not 0.0 <= number <= 1.0:
        raise IVECausalSensingMediaJobError(f"{name} must be in [0, 1]")
    return number


def _compact_token(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _contains_attitude(value: object) -> bool:
    compact = _compact_token(value)
    return any(token in compact for token in _ATTITUDE_TOKENS)


def _source_id(value: object) -> str:
    if not isinstance(value, str):
        raise IVECausalSensingMediaJobError("source_id must be a string")
    normalized = value.strip()
    if normalized != value or _SOURCE_ID.fullmatch(normalized) is None:
        raise IVECausalSensingMediaJobError("source_id must be an opaque canonical ID")
    return normalized


def _canonical_created_at(value: object) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise IVECausalSensingMediaJobError(
            "created_at_utc must be a canonical UTC RFC3339 timestamp"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise IVECausalSensingMediaJobError(
            "created_at_utc must be a canonical UTC RFC3339 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise IVECausalSensingMediaJobError("created_at_utc must use UTC")
    if parsed.isoformat().replace("+00:00", "Z") != value:
        raise IVECausalSensingMediaJobError("created_at_utc is not canonical")
    return value


@dataclass(frozen=True, slots=True)
class IVECausalSensingSource:
    """One explicit synchronized media authority; paths never persist."""

    source_id: str
    path: Path
    media_content_sha256: str
    clock: ReferenceSourceClock
    native_cut_source_seconds: tuple[float, ...]
    native_cut_evidence_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _source_id(self.source_id))
        path = Path(self.path).expanduser()
        if not path.is_absolute():
            raise IVECausalSensingMediaJobError("source paths must be absolute")
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self,
            "media_content_sha256",
            _sha256(self.media_content_sha256, "media_content_sha256"),
        )
        if not isinstance(self.clock, ReferenceSourceClock):
            raise IVECausalSensingMediaJobError(
                "source clock must be a checked ReferenceSourceClock"
            )
        cuts = tuple(
            _finite(value, "native cut source time")
            for value in self.native_cut_source_seconds
        )
        if any(value < 0.0 for value in cuts):
            raise IVECausalSensingMediaJobError(
                "native cut source times must be non-negative"
            )
        if cuts != tuple(sorted(set(cuts))):
            raise IVECausalSensingMediaJobError(
                "native cut source times must be strict and canonical"
            )
        object.__setattr__(self, "native_cut_source_seconds", cuts)
        object.__setattr__(
            self,
            "native_cut_evidence_sha256",
            _sha256(
                self.native_cut_evidence_sha256,
                "native_cut_evidence_sha256",
            ),
        )

    @property
    def clock_sha256(self) -> str:
        return _canonical_sha256(self.clock.to_dict())

    @property
    def native_cut_manifest_sha256(self) -> str:
        return _canonical_sha256(
            {
                "evidence_sha256": self.native_cut_evidence_sha256,
                "source_seconds": list(self.native_cut_source_seconds),
            }
        )


@dataclass(frozen=True, slots=True)
class IVECausalSensingMediaRequest:
    """Complete deterministic authority for one triple-artifact build."""

    song: str
    edit_mode: EditMode | str
    sources: tuple[IVECausalSensingSource, ...]
    master_decision_times: tuple[float, ...]
    gallery_artifact_path: Path
    gallery_content_sha256: str
    artifact_directory: Path
    created_at_utc: str
    target_member: IVEMember | str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.song, str) or not self.song.strip():
            raise IVECausalSensingMediaJobError("song must be a non-empty string")
        if any(ord(character) < 32 for character in self.song):
            raise IVECausalSensingMediaJobError("song cannot contain control characters")
        song = self.song.strip()
        object.__setattr__(self, "song", song)
        try:
            mode = (
                self.edit_mode
                if isinstance(self.edit_mode, EditMode)
                else EditMode(str(self.edit_mode).strip().lower())
            )
        except ValueError as exc:
            raise IVECausalSensingMediaJobError(
                "edit_mode must be group or member"
            ) from exc
        object.__setattr__(self, "edit_mode", mode)
        target: IVEMember | None
        if self.target_member is None:
            target = None
        else:
            try:
                target = (
                    self.target_member
                    if isinstance(self.target_member, IVEMember)
                    else IVEMember.parse(str(self.target_member))
                )
            except ValueError as exc:
                raise IVECausalSensingMediaJobError(
                    "target_member must be an IVE member"
                ) from exc
        if mode is EditMode.GROUP and target is not None:
            raise IVECausalSensingMediaJobError(
                "GROUP causal sensing must not bind a target member"
            )
        if mode is EditMode.MEMBER and target is None:
            raise IVECausalSensingMediaJobError(
                "MEMBER causal sensing requires an exact target member"
            )
        object.__setattr__(self, "target_member", target)

        sources = tuple(self.sources)
        if not 1 <= len(sources) <= _MAXIMUM_SOURCE_COUNT or any(
            not isinstance(item, IVECausalSensingSource) for item in sources
        ):
            raise IVECausalSensingMediaJobError(
                "sources must contain one to 32 typed source authorities"
            )
        canonical_sources = tuple(sorted(sources, key=lambda item: item.source_id))
        if sources != canonical_sources or len(
            {item.source_id for item in sources}
        ) != len(sources):
            raise IVECausalSensingMediaJobError(
                "sources must be unique and source-ID ordered"
            )
        object.__setattr__(self, "sources", sources)

        times = tuple(
            _finite(value, "master decision time")
            for value in self.master_decision_times
        )
        if not 1 <= len(times) <= _MAXIMUM_DECISION_COUNT:
            raise IVECausalSensingMediaJobError("decision grid size is invalid")
        if any(value < 0.0 for value in times) or times != tuple(sorted(set(times))):
            raise IVECausalSensingMediaJobError(
                "master decision times must be non-negative, strict, and canonical"
            )
        object.__setattr__(self, "master_decision_times", times)
        if len(sources) * len(times) > _MAXIMUM_SOURCE_TIME_CELLS:
            raise IVECausalSensingMediaJobError(
                "source/time grid exceeds the bounded M4-memory contract"
            )
        gallery_path = Path(self.gallery_artifact_path).expanduser()
        artifact_directory = Path(self.artifact_directory).expanduser()
        if not gallery_path.is_absolute() or not artifact_directory.is_absolute():
            raise IVECausalSensingMediaJobError(
                "gallery and artifact paths must be absolute"
            )
        object.__setattr__(self, "gallery_artifact_path", gallery_path)
        object.__setattr__(self, "artifact_directory", artifact_directory)
        object.__setattr__(
            self,
            "gallery_content_sha256",
            _sha256(self.gallery_content_sha256, "gallery_content_sha256"),
        )
        object.__setattr__(
            self, "created_at_utc", _canonical_created_at(self.created_at_utc)
        )

    @property
    def decision_grid_sha256(self) -> str:
        return _canonical_sha256(list(self.master_decision_times))

    @property
    def song_scope_sha256(self) -> str:
        return _canonical_sha256({"group": "IVE", "song": self.song})


@dataclass(frozen=True, slots=True)
class CausalFrameSample:
    """Transient current/following pair for one requested source timestamp."""

    requested_source_seconds: float
    decoded_source_seconds: float
    frame_index: int
    following_frame_index: int
    frame_rate: float
    current_frame: Any = field(repr=False, compare=False)
    following_frame: Any = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        requested = _finite(
            self.requested_source_seconds, "requested source seconds"
        )
        decoded = _finite(self.decoded_source_seconds, "decoded source seconds")
        frame_rate = _finite(self.frame_rate, "frame rate")
        if requested < 0.0 or decoded < 0.0 or frame_rate <= 0.0:
            raise IVECausalSensingMediaJobError(
                "causal frame times and rate are invalid"
            )
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise IVECausalSensingMediaJobError("frame_index must be non-negative")
        if self.following_frame_index != self.frame_index + 1:
            raise IVECausalSensingMediaJobError(
                "following frame must be exactly one decoded frame later"
            )
        if self.current_frame is None or self.following_frame is None:
            raise IVECausalSensingMediaJobError(
                "current and following frames are both mandatory"
            )
        maximum_residual = 0.500001 / frame_rate
        if not math.isclose(
            decoded,
            self.frame_index / frame_rate,
            rel_tol=0.0,
            abs_tol=_FLOAT_TOLERANCE,
        ):
            raise IVECausalSensingMediaJobError(
                "decoded source time does not match frame index and rate"
            )
        if abs(requested - decoded) > maximum_residual:
            raise IVECausalSensingMediaJobError(
                "decoded decision frame exceeds half-frame clock residual"
            )
        object.__setattr__(self, "requested_source_seconds", requested)
        object.__setattr__(self, "decoded_source_seconds", decoded)
        object.__setattr__(self, "frame_rate", frame_rate)


class CausalFrameProvider(Protocol):
    """Source-sequential frame provider used by production and tests."""

    def iter_source_samples(
        self,
        source: IVECausalSensingSource,
        requested_source_seconds: Sequence[float],
    ) -> Iterator[CausalFrameSample]: ...


class OpenCVSequentialCausalFrameProvider:
    """Forward-only decoder retaining at most one current/following pair.

    Strict decision timestamps can legitimately quantize to the same native
    frame.  Such requests reuse that frame and its immediate follower without
    another decode; native frame indices therefore advance monotonically, not
    strictly.  This is still causal because no sample can see past the one
    mandatory following frame.
    """

    def iter_source_samples(
        self,
        source: IVECausalSensingSource,
        requested_source_seconds: Sequence[float],
    ) -> Iterator[CausalFrameSample]:
        capture = cv2.VideoCapture(str(source.path))
        if not capture.isOpened():
            capture.release()
            raise IVECausalSensingMediaJobError("cannot open a verified source")
        frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(frame_rate) or frame_rate <= 0.0:
            capture.release()
            raise IVECausalSensingMediaJobError("verified source frame rate is invalid")
        timestamps = tuple(
            _finite(timestamp, "requested source seconds")
            for timestamp in requested_source_seconds
        )
        if not timestamps or any(timestamp < 0.0 for timestamp in timestamps):
            capture.release()
            raise IVECausalSensingMediaJobError(
                "source frame requests must be non-empty and non-negative"
            )
        if any(
            right <= left
            for left, right in zip(timestamps, timestamps[1:], strict=False)
        ):
            capture.release()
            raise IVECausalSensingMediaJobError(
                "source frame request timestamps must be strict"
            )
        targets: list[tuple[int, float]] = []
        for timestamp in timestamps:
            # This is the engine-wide synchronized-media quantizer used by
            # reference attribution and StageEdit feature snapshots.  Python's
            # ties-to-even ``round`` is intentional; a half-up rule would bind
            # N + 0.5 labels and sensing evidence to adjacent frames.
            index = int(round(float(timestamp) * frame_rate))
            targets.append((index, float(timestamp)))
        if any(
            right[0] < left[0]
            for left, right in zip(targets, targets[1:], strict=False)
        ):
            capture.release()
            raise IVECausalSensingMediaJobError(
                "decision timestamps quantize to regressing source frames"
            )
        target_by_index: dict[int, list[float]] = {}
        for index, timestamp in targets:
            target_by_index.setdefault(index, []).append(timestamp)
        final_index = targets[-1][0] + 1
        pending: tuple[int, tuple[float, ...], Any] | None = None
        index = 0
        try:
            while index <= final_index:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                if pending is not None and index == pending[0] + 1:
                    target_index, requested_group, current = pending
                    for requested in requested_group:
                        yield CausalFrameSample(
                            requested_source_seconds=requested,
                            decoded_source_seconds=target_index / frame_rate,
                            frame_index=target_index,
                            following_frame_index=index,
                            frame_rate=frame_rate,
                            current_frame=current,
                            following_frame=frame,
                        )
                    pending = None
                if index in target_by_index:
                    if pending is not None:
                        raise IVECausalSensingMediaJobError(
                            "source frame requests overlap unexpectedly"
                        )
                    pending = (index, tuple(target_by_index[index]), frame)
                index += 1
        finally:
            capture.release()


DescriptorProvider = Callable[[Any, Any], FrameVisualDescriptor]
FaceEmbeddingProvider = Callable[[Any, FrameVisualDescriptor], Sequence[float]]


@dataclass(frozen=True, slots=True)
class IVECausalSensingRuntime:
    """Injected runtime with an immutable analyzer/model fingerprint."""

    frame_provider: CausalFrameProvider
    descriptor_provider: DescriptorProvider
    face_embedding_provider: FaceEmbeddingProvider
    analyzer_fingerprint_sha256: str

    def __post_init__(self) -> None:
        if not hasattr(self.frame_provider, "iter_source_samples"):
            raise IVECausalSensingMediaJobError(
                "runtime frame provider is invalid"
            )
        if not callable(self.descriptor_provider) or not callable(
            self.face_embedding_provider
        ):
            raise IVECausalSensingMediaJobError("runtime analyzers must be callable")
        object.__setattr__(
            self,
            "analyzer_fingerprint_sha256",
            _sha256(
                self.analyzer_fingerprint_sha256,
                "analyzer_fingerprint_sha256",
            ),
        )


def _production_descriptor(current: Any, following: Any) -> FrameVisualDescriptor:
    return describe_frame(
        current,
        following_frame=following,
        include_subject_geometry=True,
        include_formation_pose=True,
    )


def _production_runtime() -> IVECausalSensingRuntime:
    yunet = yunet_model_asset_status(verify_load=True)
    sface = sface_model_asset_status(verify_load=True)
    if not (
        bool(yunet.get("digest_valid"))
        and yunet.get("loadable") is True
        and sface.digest_valid
        and sface.loadable is True
    ):
        raise IVECausalSensingMediaJobError(
            "verified loadable YuNet and SFace assets are mandatory"
        )
    fingerprint = _canonical_sha256(
        {
            "contract": IVE_CAUSAL_SENSING_MEDIA_JOB_CONTRACT,
            "frame_contract": IVE_CAUSAL_SENSING_FRAME_CONTRACT,
            "identity_contract": IVE_CAUSAL_SENSING_IDENTITY_CONTRACT,
            "media_job_code_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "visual_analyzer_fingerprint_sha256": (
                visual_analyzer_fingerprint_sha256()
            ),
            "yunet_model_sha256": yunet.get("actual_sha256"),
            "sface_model_sha256": sface.actual_sha256,
            "opencv_version": str(cv2.__version__),
        }
    )
    # Keep one verified recognizer local to this job runtime.  The general
    # transition helper deliberately disables itself after a model miss; a
    # production sensing build must instead prove the current verified model
    # is loadable and use that exact instance for every transient query.
    embedder = SFaceEmbedder()

    def verified_face_embedding(
        current: Any,
        descriptor: FrameVisualDescriptor,
    ) -> Sequence[float]:
        bbox = descriptor.primary_face_bbox
        landmarks = descriptor.primary_face_landmarks_xy
        if (
            descriptor.face_detector_backend != "yunet"
            or bbox is None
            or landmarks is None
            or len(landmarks) != 5
        ):
            return ()
        return embedder.embedding(
            current,
            bbox_xywh=bbox,
            landmarks_xy=landmarks,
        )

    return IVECausalSensingRuntime(
        frame_provider=OpenCVSequentialCausalFrameProvider(),
        descriptor_provider=_production_descriptor,
        face_embedding_provider=verified_face_embedding,
        analyzer_fingerprint_sha256=fingerprint,
    )


def _regular_file_identity(path: Path) -> tuple[int, int, int, int]:
    if path.is_symlink() or not path.is_file():
        raise IVECausalSensingMediaJobError(
            "source media must be a non-symlink regular file"
        )
    try:
        stat_result = path.stat()
    except OSError as exc:
        raise IVECausalSensingMediaJobError("cannot stat source media") from exc
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(_FILE_BLOCK_BYTES):
                digest.update(block)
    except OSError as exc:
        raise IVECausalSensingMediaJobError("cannot hash source media") from exc
    return digest.hexdigest()


def _verify_source_media(source: IVECausalSensingSource) -> tuple[int, int, int, int]:
    before = _regular_file_identity(source.path)
    actual = _file_sha256(source.path)
    after = _regular_file_identity(source.path)
    if before != after:
        raise IVECausalSensingMediaJobError("source media changed while hashing")
    if actual != source.media_content_sha256:
        raise IVECausalSensingMediaJobError("source media content SHA-256 mismatch")
    return after


def _verify_source_unchanged(
    source: IVECausalSensingSource,
    expected_identity: tuple[int, int, int, int],
) -> None:
    if _regular_file_identity(source.path) != expected_identity:
        raise IVECausalSensingMediaJobError("source media changed during analysis")
    if _file_sha256(source.path) != source.media_content_sha256:
        raise IVECausalSensingMediaJobError("source media changed during analysis")


def _validate_descriptor(descriptor: object) -> FrameVisualDescriptor:
    if not isinstance(descriptor, FrameVisualDescriptor):
        raise IVECausalSensingMediaJobError(
            "descriptor provider must return FrameVisualDescriptor"
        )
    for name in (
        "quality_score",
        "composition_score",
        "shot_scale",
        "pose_confidence",
        "silhouette_confidence",
    ):
        _unit(getattr(descriptor, name), f"descriptor {name}")
    if descriptor.motion is not None:
        if len(descriptor.motion) != 2:
            raise IVECausalSensingMediaJobError(
                "descriptor motion must contain two values"
            )
        for value in descriptor.motion:
            number = _finite(value, "descriptor motion")
            if not -1.0 <= number <= 1.0:
                raise IVECausalSensingMediaJobError(
                    "descriptor motion must be normalized"
                )
    if isinstance(descriptor.face_count, bool) or not isinstance(
        descriptor.face_count, int
    ) or descriptor.face_count < 0:
        raise IVECausalSensingMediaJobError("descriptor face count is invalid")
    if isinstance(descriptor.detected_person_count, bool) or not isinstance(
        descriptor.detected_person_count, int
    ) or not 0 <= descriptor.detected_person_count <= 64:
        raise IVECausalSensingMediaJobError(
            "descriptor detected-person count is invalid"
        )
    return descriptor


def _shot_probabilities(
    descriptor: FrameVisualDescriptor,
) -> tuple[float, float, float] | None:
    if descriptor.shot_scale_basis == "unknown_no_subject_geometry":
        return None
    scale = _unit(descriptor.shot_scale, "descriptor shot scale")
    centers = (0.70, 0.38, 0.14)
    weights = [math.exp(-((scale - center) / 0.18) ** 2) for center in centers]
    preferred = {
        ShotType.CLOSE_UP: 0,
        ShotType.MEDIUM: 1,
        ShotType.WIDE: 2,
        ShotType.GROUP_WIDE: 2,
    }.get(descriptor.shot_type)
    if preferred is None:
        raise IVECausalSensingMediaJobError("descriptor shot type is invalid")
    weights[preferred] *= 1.75
    total = math.fsum(weights)
    probabilities = tuple(value / total for value in weights)
    return (probabilities[0], probabilities[1], probabilities[2])


def _zoom_probabilities(
    descriptor: FrameVisualDescriptor,
    previous_scale: float | None,
    *,
    cut_since_previous: bool | None,
) -> tuple[float, float, float] | None:
    if (
        previous_scale is None
        or cut_since_previous is not False
        or descriptor.shot_scale_basis == "unknown_no_subject_geometry"
    ):
        return None
    delta = descriptor.shot_scale - previous_scale
    strength = min(1.0, abs(delta) / 0.18)
    if delta > 0.0:
        return (strength, 0.0, 1.0 - strength)
    if delta < 0.0:
        return (0.0, strength, 1.0 - strength)
    return (0.0, 0.0, 1.0)


def _xywh_to_bbox(value: Sequence[float]) -> NormalizedBBox:
    if len(value) != 4:
        raise IVECausalSensingMediaJobError("normalized bbox must have four values")
    left, top, width, height = (
        _finite(item, "normalized bbox coordinate") for item in value
    )
    right = min(1.0, left + width)
    bottom = min(1.0, top + height)
    try:
        return NormalizedBBox(
            max(0.0, left),
            max(0.0, top),
            right,
            bottom,
        )
    except ValueError as exc:
        raise IVECausalSensingMediaJobError("descriptor bbox is invalid") from exc


def _face_body_bbox(face: NormalizedBBox) -> NormalizedBBox:
    center_x, center_y = face.center
    width = min(1.0, (face.right - face.left) * 2.4)
    height = min(1.0, (face.bottom - face.top) * 3.8)
    left = max(0.0, center_x - width / 2.0)
    right = min(1.0, center_x + width / 2.0)
    top = max(0.0, center_y - height * 0.28)
    bottom = min(1.0, top + height)
    if bottom - top < height:
        top = max(0.0, bottom - height)
    return NormalizedBBox(left, top, right, bottom)


def _shot_id(
    source: IVECausalSensingSource,
    source_seconds: float,
) -> str:
    segment = sum(
        cut <= source_seconds for cut in source.native_cut_source_seconds
    )
    digest = _canonical_sha256(
        {
            "source_id": source.source_id,
            "native_cut_manifest_sha256": source.native_cut_manifest_sha256,
            "segment": segment,
        }
    )
    return f"shot:{digest[:32]}"


def _cut_since_previous(
    source: IVECausalSensingSource,
    previous_source_seconds: float | None,
    source_seconds: float,
) -> bool | None:
    if previous_source_seconds is None:
        return None
    return any(
        previous_source_seconds < cut <= source_seconds
        for cut in source.native_cut_source_seconds
    )


def _identity_for_descriptor(
    *,
    runtime: IVECausalSensingRuntime,
    gallery: IVEFaceGalleryArtifact,
    current_frame: Any,
    descriptor: FrameVisualDescriptor,
) -> tuple[MemberProbabilityMass, str | None]:
    if not (
        descriptor.face_detector_backend == "yunet"
        and descriptor.primary_face_bbox is not None
        and descriptor.primary_face_landmarks_xy is not None
        and len(descriptor.primary_face_landmarks_xy) == 5
    ):
        return (MemberProbabilityMass.unknown(), None)
    try:
        query_embedding = tuple(
            float(value)
            for value in runtime.face_embedding_provider(current_frame, descriptor)
        )
    except (ArithmeticError, RuntimeError, TypeError, ValueError):
        return (MemberProbabilityMass.unknown(), None)
    if not query_embedding:
        return (MemberProbabilityMass.unknown(), None)
    try:
        result = classify_open_set_ive_identity(gallery.gallery, query_embedding)
        adapted = adapt_gallery_identity_for_member_state(result)
    finally:
        # The only query vector reference is discarded immediately after the
        # scalar/opaque adapter has been produced (or classification failed).
        del query_embedding
    return (adapted.member_probabilities, adapted.face_identity_id)


def _member_observations(
    *,
    source: IVECausalSensingSource,
    master_seconds: float,
    source_seconds: float,
    sample: CausalFrameSample,
    graph_frame_index: int,
    descriptor: FrameVisualDescriptor,
    identity_mass: MemberProbabilityMass,
    identity_token: str | None,
    starting_causal_order: int,
) -> tuple[tuple[CausalMemberObservation, ...], int]:
    face_bbox = (
        _xywh_to_bbox(descriptor.primary_face_bbox)
        if descriptor.primary_face_bbox is not None
        else None
    )
    body_candidates: list[tuple[NormalizedBBox, float]] = []
    for person in descriptor.formation_people:
        body_candidates.append(
            (
                _xywh_to_bbox(person.bbox_xywh),
                _unit(person.detection_confidence, "formation confidence"),
            )
        )
    body_candidates = body_candidates[:_MAXIMUM_MEMBER_OBSERVATIONS_PER_FRAME]
    if not body_candidates and descriptor.silhouette_bbox_xywh is not None:
        body_candidates.append(
            (
                _xywh_to_bbox(descriptor.silhouette_bbox_xywh),
                _unit(descriptor.silhouette_confidence, "silhouette confidence"),
            )
        )
    identity_index: int | None = None
    if face_bbox is not None:
        face_center = face_bbox.center
        containing = [
            index
            for index, (body, _confidence) in enumerate(body_candidates)
            if body.left <= face_center[0] <= body.right
            and body.top <= face_center[1] <= body.bottom
            and face_bbox.area <= body.area
        ]
        if containing:
            identity_index = min(containing, key=lambda index: body_candidates[index][0].area)
        else:
            face_candidate = (_face_body_bbox(face_bbox), descriptor.quality_score)
            if len(body_candidates) < _MAXIMUM_MEMBER_OBSERVATIONS_PER_FRAME:
                identity_index = len(body_candidates)
                body_candidates.append(face_candidate)
            else:
                identity_index = min(
                    range(len(body_candidates)),
                    key=lambda index: (body_candidates[index][1], -index),
                )
                body_candidates[identity_index] = face_candidate

    represented = len(body_candidates)
    detected = max(
        descriptor.detected_person_count,
        descriptor.face_count,
        len(descriptor.formation_people),
        represented,
    )
    missing_geometry_count = min(
        max(0, detected - represented),
        _MAXIMUM_MEMBER_OBSERVATIONS_PER_FRAME - represented,
    )
    if represented == 0 and missing_geometry_count == 0:
        missing_geometry_count = 1

    shot_id = _shot_id(source, source_seconds)
    shot_confidence = min(
        1.0,
        max(0.05, 0.55 * descriptor.quality_score + 0.45 * descriptor.composition_score),
    )
    observations: list[CausalMemberObservation] = []
    order = starting_causal_order
    for index, (body_bbox, geometry_confidence) in enumerate(body_candidates):
        attached_face = face_bbox if index == identity_index else None
        mass = identity_mass if index == identity_index else MemberProbabilityMass.unknown()
        token = identity_token if index == identity_index else None
        observation_digest = _canonical_sha256(
            {
                "source_id": source.source_id,
                "media_content_sha256": source.media_content_sha256,
                "master_seconds": master_seconds,
                "source_seconds": source_seconds,
                "graph_frame_index": graph_frame_index,
                "native_frame_index": sample.frame_index,
                "person_index": index,
                "body_bbox": body_bbox.to_list(),
                "face_bbox": attached_face.to_list() if attached_face else None,
            }
        )
        observations.append(
            CausalMemberObservation(
                observation_id=f"obs:{observation_digest[:40]}",
                source_id=source.source_id,
                shot_id=shot_id,
                timestamp_seconds=master_seconds,
                # Member-state graph ordering requires every distinct exact
                # timestamp to advance its frame coordinate.  A native decoder
                # frame may be reused by multiple nearby decision timestamps,
                # so this field is the source-local decision ordinal.  The
                # actual native frame/follower/rate mapping is separately
                # hash-bound in the source observation manifest.
                frame_index=graph_frame_index,
                causal_order=order,
                member_probabilities=mass,
                face_bbox=attached_face,
                body_bbox=body_bbox,
                face_identity_id=token,
                occlusion_probability=0.0,
                geometry_confidence=max(0.01, geometry_confidence),
                shot_confidence=shot_confidence,
            )
        )
        order += 1
    for missing_index in range(missing_geometry_count):
        observation_digest = _canonical_sha256(
            {
                "source_id": source.source_id,
                "media_content_sha256": source.media_content_sha256,
                "master_seconds": master_seconds,
                "source_seconds": source_seconds,
                "graph_frame_index": graph_frame_index,
                "native_frame_index": sample.frame_index,
                "missing_geometry_index": missing_index,
            }
        )
        observations.append(
            CausalMemberObservation(
                observation_id=f"obs:{observation_digest[:40]}",
                source_id=source.source_id,
                shot_id=shot_id,
                timestamp_seconds=master_seconds,
                frame_index=graph_frame_index,
                causal_order=order,
                member_probabilities=MemberProbabilityMass.unknown(),
                occlusion_probability=1.0,
                geometry_confidence=0.0,
                shot_confidence=shot_confidence,
            )
        )
        order += 1
    return (tuple(observations), order)


def _shooting_observation(
    *,
    source: IVECausalSensingSource,
    master_seconds: float,
    source_seconds: float,
    descriptor: FrameVisualDescriptor,
    previous_source_seconds: float | None,
    previous_scale: float | None,
) -> ShootingGrammarObservation:
    cut = _cut_since_previous(source, previous_source_seconds, source_seconds)
    shot = _shot_probabilities(descriptor)
    zoom = _zoom_probabilities(
        descriptor,
        previous_scale,
        cut_since_previous=cut,
    )
    motion = descriptor.motion
    attention_bbox = descriptor.primary_face_bbox or descriptor.silhouette_bbox_xywh
    headroom: float | None = None
    lead_room: float | None = None
    if attention_bbox is not None:
        left, top, width, _height = attention_bbox
        headroom = min(1.0, max(0.0, float(top)))
        center_x = float(left) + float(width) / 2.0
        lead_room = min(1.0, max(0.0, 2.0 * abs(center_x - 0.5)))
    detected = min(
        64,
        max(
            descriptor.detected_person_count,
            descriptor.face_count,
            len(descriptor.formation_people),
        ),
    )
    formation = min(detected, len(descriptor.formation_people))
    observed_geometry = descriptor.shot_scale_basis != "unknown_no_subject_geometry"
    confidence = (
        min(
            1.0,
            max(
                0.0,
                0.45 * descriptor.quality_score
                + 0.35 * descriptor.composition_score
                + 0.20 * float(observed_geometry),
            ),
        )
        if observed_geometry
        else None
    )
    return ShootingGrammarObservation(
        source_id=source.source_id,
        master_time_seconds=master_seconds,
        shot_closeup_probability=shot[0] if shot is not None else None,
        shot_medium_probability=shot[1] if shot is not None else None,
        shot_wide_probability=shot[2] if shot is not None else None,
        camera_pan_velocity=motion[0] if motion is not None else None,
        camera_tilt_velocity=motion[1] if motion is not None else None,
        camera_zoom_in_probability=zoom[0] if zoom is not None else None,
        camera_zoom_out_probability=zoom[1] if zoom is not None else None,
        camera_static_probability=zoom[2] if zoom is not None else None,
        # Face/head roll is not camera roll.  Leave the camera value missing
        # rather than silently relabeling identity geometry.
        camera_roll_degrees_normalized=None,
        subject_headroom_normalized=headroom,
        subject_lead_room_normalized=lead_room,
        detected_person_count=detected,
        formation_person_count=formation,
        internal_native_cut_since_previous=cut,
        shooting_grammar_confidence=confidence,
    )


def _observation_manifest_sha256(
    *,
    source: IVECausalSensingSource,
    request: IVECausalSensingMediaRequest,
    observations: Sequence[CausalMemberObservation],
    decoded_frame_manifest_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "contract": IVE_CAUSAL_SENSING_MEDIA_JOB_CONTRACT,
            "source_id": source.source_id,
            "source_content_sha256": source.media_content_sha256,
            "source_clock_sha256": source.clock_sha256,
            "native_cut_manifest_sha256": source.native_cut_manifest_sha256,
            "decision_grid_sha256": request.decision_grid_sha256,
            "song_scope_sha256": request.song_scope_sha256,
            "gallery_content_sha256": request.gallery_content_sha256,
            "member_frame_contract": IVE_CAUSAL_SENSING_MEMBER_FRAME_CONTRACT,
            "decoded_frame_manifest_sha256": _sha256(
                decoded_frame_manifest_sha256,
                "decoded frame manifest SHA-256",
            ),
            "observations": [item.to_dict() for item in observations],
        }
    )


def _update_manifest_digest(
    digest: Any,
    value: Mapping[str, object],
) -> None:
    """Append one unambiguous canonical record to a streaming manifest."""

    encoded = _canonical_bytes(value)
    digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
    digest.update(encoded)


def _safe_artifact_directory(parent: Path, name: str) -> Path:
    child = parent / name
    if child.is_symlink():
        raise IVECausalSensingMediaJobError(
            "artifact directory cannot be a symbolic link"
        )
    child.mkdir(exist_ok=True)
    if child.is_symlink() or not child.is_dir():
        raise IVECausalSensingMediaJobError("artifact directory is unsafe")
    return child


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError as exc:
        if exc.errno in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
            return
        raise IVECausalSensingMediaJobError(
            "cannot open artifact directory for durability"
        ) from exc
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
                raise IVECausalSensingMediaJobError(
                    "cannot flush artifact directory"
                ) from exc
    finally:
        os.close(descriptor)


def _read_regular_artifact(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise IVECausalSensingMediaJobError(
            "artifact must be a non-symlink regular file"
        )
    before = path.stat()
    try:
        encoded = path.read_bytes()
    except OSError as exc:
        raise IVECausalSensingMediaJobError("cannot read artifact") from exc
    after = path.stat()
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
    )
    if identity(before) != identity(after):
        raise IVECausalSensingMediaJobError("artifact changed while reading")
    return encoded


def _publish_encoded_artifact(
    destination: Path,
    encoded: bytes,
    *,
    content_sha256: str,
) -> Path:
    """No-clobber publish on POSIX and ExFAT under an exclusive lock."""

    digest = _sha256(content_sha256, "artifact content SHA-256")
    if destination.name != f"{digest}.json":
        raise IVECausalSensingMediaJobError(
            "artifact destination is not content addressed"
        )
    if destination.is_symlink():
        raise IVECausalSensingMediaJobError(
            "artifact destination cannot be a symbolic link"
        )
    if destination.exists():
        if hmac.compare_digest(_read_regular_artifact(destination), encoded):
            return destination
        raise IVECausalSensingMediaJobError(
            "content-addressed destination contains different bytes"
        )

    shard = destination.parent
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{digest}.",
        suffix=".tmp",
        dir=shard,
    )
    temporary = Path(temporary_name)
    lock = shard / f".{digest}.publish-lock"
    lock_descriptor: int | None = None
    lock_identity: os.stat_result | None = None
    marker = f"{os.getpid()}:{uuid.uuid4().hex}\n".encode("ascii")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o400)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            lock_descriptor = os.open(lock, flags, 0o600)
        except FileExistsError as exc:
            if destination.exists() and not destination.is_symlink() and hmac.compare_digest(
                _read_regular_artifact(destination),
                encoded,
            ):
                return destination
            raise IVECausalSensingMediaJobError(
                "artifact publication lock is already held"
            ) from exc
        lock_identity = os.fstat(lock_descriptor)
        if os.write(lock_descriptor, marker) != len(marker):
            raise IVECausalSensingMediaJobError(
                "artifact publication lock write was incomplete"
            )
        os.fsync(lock_descriptor)
        if destination.is_symlink():
            raise IVECausalSensingMediaJobError(
                "artifact destination became a symbolic link"
            )
        if destination.exists():
            if not hmac.compare_digest(
                _read_regular_artifact(destination),
                encoded,
            ):
                raise IVECausalSensingMediaJobError(
                    "concurrent artifact writer published different bytes"
                )
        else:
            os.replace(temporary, destination)
            destination.chmod(0o400)
        if not hmac.compare_digest(_read_regular_artifact(destination), encoded):
            raise IVECausalSensingMediaJobError(
                "published artifact bytes do not reproduce"
            )
        _fsync_directory(shard)
        return destination
    except IVECausalSensingMediaJobError:
        raise
    except OSError as exc:
        raise IVECausalSensingMediaJobError(
            "failed to publish immutable artifact"
        ) from exc
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        try:
            current = os.lstat(lock)
            owned = bool(
                lock_identity is not None
                and stat.S_ISREG(current.st_mode)
                and (current.st_dev, current.st_ino)
                == (lock_identity.st_dev, lock_identity.st_ino)
            )
            if not owned and stat.S_ISREG(current.st_mode):
                reader = os.open(
                    lock,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    opened = os.fstat(reader)
                    owned = bool(
                        opened.st_size == len(marker)
                        and os.read(reader, len(marker) + 1) == marker
                    )
                finally:
                    os.close(reader)
            if owned:
                lock.unlink()
        except OSError:
            pass
        try:
            temporary.unlink()
        except OSError:
            pass


def _publish_staged_artifact(
    *,
    staged_path: Path,
    staged_root: Path,
    destination_root: Path,
    content_sha256: str,
) -> Path:
    try:
        resolved_staged_root = staged_root.resolve(strict=True)
        resolved_staged_path = staged_path.resolve(strict=True)
        relative = resolved_staged_path.relative_to(resolved_staged_root)
    except (OSError, ValueError) as exc:
        raise IVECausalSensingMediaJobError(
            "staged artifact escaped its private root"
        ) from exc
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise IVECausalSensingMediaJobError("staged artifact path is unsafe")
    root = Path(destination_root)
    if root.is_symlink():
        raise IVECausalSensingMediaJobError(
            "artifact root cannot be a symbolic link"
        )
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise IVECausalSensingMediaJobError("artifact root is unsafe")
    parent = root
    for part in relative.parts[:-1]:
        parent = _safe_artifact_directory(parent, part)
    destination = parent / relative.name
    return _publish_encoded_artifact(
        destination,
        _read_regular_artifact(resolved_staged_path),
        content_sha256=content_sha256,
    )


@dataclass(frozen=True, slots=True)
class IVECausalSensingMediaJobResult:
    shooting_grammar_path: Path
    shooting_grammar_content_sha256: str
    member_state_graph_path: Path
    member_state_graph_content_sha256: str
    ive_performer_state_path: Path
    ive_performer_state_content_sha256: str
    stage_edit_artifacts: StageEditCausalSensingArtifacts
    source_count: int
    decision_count: int
    member_observation_count: int
    accepted_identity_observation_count: int


def reject_attitude_before_ive_causal_sensing_access(
    *,
    song: object,
    source_ids: Sequence[object] = (),
    source_paths: Sequence[object] = (),
) -> None:
    """Metadata-only ATTITUDE quarantine guard for upstream discovery code.

    This function performs no path resolution, file stat, gallery discovery,
    model loading, or decoder access.  Integrators can therefore call it before
    consulting any private artifact or source-media location.
    """

    if (
        _contains_attitude(song)
        or any(_contains_attitude(value) for value in source_ids)
        or any(_contains_attitude(value) for value in source_paths)
    ):
        raise IVECausalSensingMediaJobError(
            "ATTITUDE is strictly quarantined from causal sensing"
        )


def _reject_attitude_before_access(request: IVECausalSensingMediaRequest) -> None:
    reject_attitude_before_ive_causal_sensing_access(
        song=request.song,
        source_ids=tuple(source.source_id for source in request.sources),
        source_paths=tuple(source.path for source in request.sources),
    )


def materialize_ive_causal_sensing_from_media(
    request: IVECausalSensingMediaRequest,
    *,
    runtime: IVECausalSensingRuntime | None = None,
) -> IVECausalSensingMediaJobResult:
    """Verify media, derive the scalar triple, publish, and strictly reload it."""

    if not isinstance(request, IVECausalSensingMediaRequest):
        raise TypeError("request must be IVECausalSensingMediaRequest")
    _reject_attitude_before_access(request)
    gallery = load_ive_face_identity_gallery_artifact(
        request.gallery_artifact_path,
        expected_content_sha256=request.gallery_content_sha256,
    )
    active_runtime = runtime or _production_runtime()
    if not isinstance(active_runtime, IVECausalSensingRuntime):
        raise TypeError("runtime must be IVECausalSensingRuntime")
    shooting_observations: list[ShootingGrammarObservation] = []
    member_observations: list[CausalMemberObservation] = []
    member_by_source: dict[str, list[CausalMemberObservation]] = {
        source.source_id: [] for source in request.sources
    }
    decoded_frame_manifest_by_source: dict[str, str] = {}

    for source in request.sources:
        mapped_times = tuple(
            _finite(
                source.clock.master_to_source(master_seconds),
                "mapped source time",
            )
            for master_seconds in request.master_decision_times
        )
        if any(value < 0.0 for value in mapped_times) or any(
            right <= left
            for left, right in zip(mapped_times, mapped_times[1:], strict=False)
        ):
            raise IVECausalSensingMediaJobError(
                "source clock does not cover the exact decision grid monotonically"
            )
        media_identity = _verify_source_media(source)
        iterator = iter(
            active_runtime.frame_provider.iter_source_samples(source, mapped_times)
        )
        previous_source_seconds: float | None = None
        previous_scale: float | None = None
        previous_frame_index: int | None = None
        observed_frame_rate: float | None = None
        cached_descriptor: FrameVisualDescriptor | None = None
        cached_identity_mass: MemberProbabilityMass | None = None
        cached_identity_token: str | None = None
        decoded_frame_manifest = hashlib.sha256()
        _update_manifest_digest(
            decoded_frame_manifest,
            {
                "record": "header",
                "contract": IVE_CAUSAL_SENSING_FRAME_CONTRACT,
                "member_frame_contract": IVE_CAUSAL_SENSING_MEMBER_FRAME_CONTRACT,
                "source_id": source.source_id,
                "source_content_sha256": source.media_content_sha256,
                "source_clock_sha256": source.clock_sha256,
                "decision_grid_sha256": request.decision_grid_sha256,
            },
        )
        causal_order = 0
        try:
            for graph_frame_index, (
                master_seconds,
                expected_source_seconds,
            ) in enumerate(
                zip(
                    request.master_decision_times,
                    mapped_times,
                    strict=True,
                )
            ):
                try:
                    sample = next(iterator)
                except StopIteration as exc:
                    raise IVECausalSensingMediaJobError(
                        "frame provider did not cover every exact source/time"
                    ) from exc
                if not isinstance(sample, CausalFrameSample):
                    raise IVECausalSensingMediaJobError(
                        "frame provider returned an invalid sample"
                    )
                if not math.isclose(
                    sample.requested_source_seconds,
                    expected_source_seconds,
                    rel_tol=0.0,
                    abs_tol=_FLOAT_TOLERANCE,
                ):
                    raise IVECausalSensingMediaJobError(
                        "frame provider changed the exact source clock request"
                    )
                if (
                    previous_frame_index is not None
                    and sample.frame_index < previous_frame_index
                ):
                    raise IVECausalSensingMediaJobError(
                        "decoded decision frames regressed and are not causal"
                    )
                if observed_frame_rate is None:
                    observed_frame_rate = sample.frame_rate
                elif not math.isclose(
                    observed_frame_rate,
                    sample.frame_rate,
                    rel_tol=0.0,
                    abs_tol=_FLOAT_TOLERANCE,
                ):
                    raise IVECausalSensingMediaJobError(
                        "source frame rate changed within one decode pass"
                    )
                _update_manifest_digest(
                    decoded_frame_manifest,
                    {
                        "record": "sample",
                        "source_local_decision_ordinal": graph_frame_index,
                        "master_seconds": master_seconds,
                        "requested_source_seconds": expected_source_seconds,
                        "decoded_source_seconds": sample.decoded_source_seconds,
                        "native_frame_index": sample.frame_index,
                        "following_native_frame_index": sample.following_frame_index,
                        "frame_rate": sample.frame_rate,
                    },
                )
                if sample.frame_index == previous_frame_index:
                    if cached_descriptor is None or cached_identity_mass is None:
                        raise IVECausalSensingMediaJobError(
                            "reused native frame has no prior causal analysis"
                        )
                    descriptor = cached_descriptor
                    identity_mass = cached_identity_mass
                    identity_token = cached_identity_token
                else:
                    descriptor = _validate_descriptor(
                        active_runtime.descriptor_provider(
                            sample.current_frame,
                            sample.following_frame,
                        )
                    )
                    identity_mass, identity_token = _identity_for_descriptor(
                        runtime=active_runtime,
                        gallery=gallery,
                        current_frame=sample.current_frame,
                        descriptor=descriptor,
                    )
                    cached_descriptor = descriptor
                    cached_identity_mass = identity_mass
                    cached_identity_token = identity_token
                observations, causal_order = _member_observations(
                    source=source,
                    master_seconds=master_seconds,
                    source_seconds=expected_source_seconds,
                    sample=sample,
                    graph_frame_index=graph_frame_index,
                    descriptor=descriptor,
                    identity_mass=identity_mass,
                    identity_token=identity_token,
                    starting_causal_order=causal_order,
                )
                member_observations.extend(observations)
                member_by_source[source.source_id].extend(observations)
                shooting_observations.append(
                    _shooting_observation(
                        source=source,
                        master_seconds=master_seconds,
                        source_seconds=expected_source_seconds,
                        descriptor=descriptor,
                        previous_source_seconds=previous_source_seconds,
                        previous_scale=previous_scale,
                    )
                )
                previous_source_seconds = expected_source_seconds
                previous_scale = (
                    descriptor.shot_scale
                    if descriptor.shot_scale_basis != "unknown_no_subject_geometry"
                    else None
                )
                previous_frame_index = sample.frame_index
                del identity_mass, identity_token, observations, descriptor, sample
            try:
                extra = next(iterator)
            except StopIteration:
                extra = None
            if extra is not None:
                raise IVECausalSensingMediaJobError(
                    "frame provider returned evidence outside the exact decision grid"
                )
            decoded_frame_manifest_by_source[source.source_id] = (
                decoded_frame_manifest.hexdigest()
            )
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        _verify_source_unchanged(source, media_identity)
        cached_descriptor = None
        cached_identity_mass = None
        cached_identity_token = None

    expected_keys = {
        (master_seconds, source.source_id)
        for master_seconds in request.master_decision_times
        for source in request.sources
    }
    actual_shooting_keys = {
        (item.master_time_seconds, item.source_id)
        for item in shooting_observations
    }
    actual_member_keys = {
        (item.timestamp_seconds, item.source_id) for item in member_observations
    }
    if actual_shooting_keys != expected_keys or actual_member_keys != expected_keys:
        raise IVECausalSensingMediaJobError(
            "derived sensing does not exactly cover every source/time"
        )

    ordered_shooting = tuple(
        sorted(
            shooting_observations,
            key=lambda item: (item.master_time_seconds, item.source_id),
        )
    )
    member_graph = build_member_state_graph(tuple(member_observations))
    upstream: list[tuple[str, str]] = [
        ("decision_grid", request.decision_grid_sha256),
        ("gallery_artifact", request.gallery_content_sha256),
        ("song_scope", request.song_scope_sha256),
    ]
    for index, source in enumerate(request.sources):
        upstream.extend(
            (
                (f"source_{index:03d}_media", source.media_content_sha256),
                (f"source_{index:03d}_clock", source.clock_sha256),
                (
                    f"source_{index:03d}_native_cuts",
                    source.native_cut_manifest_sha256,
                ),
                (
                    f"source_{index:03d}_decoded_frames",
                    decoded_frame_manifest_by_source[source.source_id],
                ),
            )
        )
    grammar = build_causal_shooting_grammar_artifact(
        ordered_shooting,
        provenance=ShootingGrammarProvenance(
            upstream_sha256=tuple(upstream),
            analyzer_fingerprint_sha256=(
                active_runtime.analyzer_fingerprint_sha256
            ),
            contracts=(
                IVE_CAUSAL_SENSING_MEDIA_JOB_CONTRACT,
                IVE_CAUSAL_SENSING_FRAME_CONTRACT,
                IVE_CAUSAL_SENSING_MEMBER_FRAME_CONTRACT,
                "paths_frames_descriptors_and_query_embeddings_not_persisted_v1",
            ),
        ),
    )
    member_provenance = MemberStateArtifactProvenance(
        producer_id="crosscut.ive_causal_sensing_media_job",
        producer_version="1",
        created_at_utc=request.created_at_utc,
        sources=tuple(
            MemberStateSourceProvenance(
                source_id=source.source_id,
                source_content_sha256=source.media_content_sha256,
                observation_manifest_sha256=_observation_manifest_sha256(
                    source=source,
                    request=request,
                    observations=member_by_source[source.source_id],
                    decoded_frame_manifest_sha256=(
                        decoded_frame_manifest_by_source[source.source_id]
                    ),
                ),
            )
            for source in request.sources
        ),
    )
    performer = build_ive_performer_state_artifact(
        member_graph,
        grammar.records,
        provenance=IVEPerformerStateProvenance(
            analyzer_fingerprint_sha256=(
                active_runtime.analyzer_fingerprint_sha256
            ),
            sources=tuple(
                IVEPerformerStateSourceProvenance(
                    source_id=source.source_id,
                    source_content_sha256=source.media_content_sha256,
                )
                for source in request.sources
            ),
            contracts=tuple(
                sorted(
                    (
                        IVE_CAUSAL_SENSING_IDENTITY_CONTRACT,
                        IVE_CAUSAL_SENSING_MEMBER_FRAME_CONTRACT,
                        IVE_CAUSAL_SENSING_MEDIA_JOB_CONTRACT,
                    )
                )
            ),
        ),
    )
    performer_keys = {
        (item.master_time_seconds, item.source_id) for item in performer.records
    }
    if performer_keys != expected_keys:
        raise IVECausalSensingMediaJobError(
            "performer-state output lost exact source/time coverage"
        )

    # Existing scalar artifact writers use hard-link publication, which ExFAT
    # rejects.  First let those canonical writers build and self-validate the
    # exact bytes on a private local filesystem, then publish the three files
    # to the requested filesystem with a lock+rename no-clobber fallback.
    with tempfile.TemporaryDirectory(
        prefix="ive-causal-sensing-stage-"
    ) as temporary_directory:
        staged_root = Path(temporary_directory)
        staged_grammar_path = write_causal_shooting_grammar_artifact(
            grammar,
            staged_root / "shooting-grammar",
        )
        staged_member_artifact = write_member_state_graph_artifact(
            staged_root / "member-state",
            member_graph,
            member_provenance,
        )
        staged_performer_path = write_ive_performer_state_artifact(
            staged_root / "performer-state",
            performer,
        )
        output_root = request.artifact_directory
        grammar_path = _publish_staged_artifact(
            staged_path=staged_grammar_path,
            staged_root=staged_root,
            destination_root=output_root,
            content_sha256=grammar.artifact_content_sha256,
        )
        member_path = _publish_staged_artifact(
            staged_path=staged_member_artifact.path,
            staged_root=staged_root,
            destination_root=output_root,
            content_sha256=staged_member_artifact.content_sha256,
        )
        performer_path = _publish_staged_artifact(
            staged_path=staged_performer_path,
            staged_root=staged_root,
            destination_root=output_root,
            content_sha256=performer.content_sha256,
        )
    stage_edit_artifacts = StageEditCausalSensingArtifacts(
        shooting_grammar_path=grammar_path,
        shooting_grammar_content_sha256=grammar.artifact_content_sha256,
        member_state_graph_path=member_path,
        member_state_graph_content_sha256=(
            staged_member_artifact.content_sha256
        ),
        target_member=request.target_member,
        ive_performer_state_path=performer_path,
        ive_performer_state_content_sha256=performer.content_sha256,
    )
    stage_edit_artifacts.binding.require_production_scope(
        edit_mode=request.edit_mode.value,
        target_member=request.target_member,
    )
    accepted_count = sum(
        item.face_identity_id is not None for item in member_graph.observations
    )
    return IVECausalSensingMediaJobResult(
        shooting_grammar_path=grammar_path,
        shooting_grammar_content_sha256=grammar.artifact_content_sha256,
        member_state_graph_path=member_path,
        member_state_graph_content_sha256=(
            staged_member_artifact.content_sha256
        ),
        ive_performer_state_path=performer_path,
        ive_performer_state_content_sha256=performer.content_sha256,
        stage_edit_artifacts=stage_edit_artifacts,
        source_count=len(request.sources),
        decision_count=len(request.master_decision_times),
        member_observation_count=len(member_graph.observations),
        accepted_identity_observation_count=accepted_count,
    )


__all__ = [
    "IVE_CAUSAL_SENSING_FRAME_CONTRACT",
    "IVE_CAUSAL_SENSING_IDENTITY_CONTRACT",
    "IVE_CAUSAL_SENSING_MEMBER_FRAME_CONTRACT",
    "IVE_CAUSAL_SENSING_MEDIA_JOB_CONTRACT",
    "IVE_CAUSAL_SENSING_MEDIA_JOB_SCHEMA_VERSION",
    "CausalFrameProvider",
    "CausalFrameSample",
    "IVECausalSensingMediaJobError",
    "IVECausalSensingMediaJobResult",
    "IVECausalSensingMediaRequest",
    "IVECausalSensingRuntime",
    "IVECausalSensingSource",
    "OpenCVSequentialCausalFrameProvider",
    "materialize_ive_causal_sensing_from_media",
    "reject_attitude_before_ive_causal_sensing_access",
]
