"""Production coordinator for one IVE StageEdit schema-v4 song corpus.

The five human-reference supervision files are one editorial corpus, not five
independent media-analysis jobs.  This coordinator therefore derives their
union decision grid once, verifies the exact acquisition and alignment
authorities that produced the supervision, materializes one native-cut
manifest and one real causal-sensing triple for that grid, and binds the same
triple into all five feature snapshots.

The request artifact deliberately contains explicit local paths because it is
a private, machine-local launch authority.  Published causal artifacts retain
the privacy-minimal contracts of :mod:`ive_causal_sensing_media_job`; no frame,
face vector, descriptor, or query embedding is copied into this coordinator's
artifacts.

ATTITUDE quarantine is the first operation in every public filesystem entry
point.  No request, supervision, acquisition, alignment, gallery, media, or
output path is opened, resolved, stated, or globbed before that metadata-only
guard succeeds.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import errno
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Any, Callable, Mapping, Sequence
import uuid

import cv2

from .ive import EditMode, IVEMember
from .ive_causal_sensing_installation import (
    InstalledIVECausalSensingGallery,
    load_ive_causal_sensing_installation,
)
from .ive_causal_sensing_media_job import (
    IVE_CAUSAL_SENSING_AVAILABILITY_CONTRACT,
    IVE_CAUSAL_SENSING_FRAME_CONTRACT,
    IVE_CAUSAL_SENSING_MEDIA_JOB_CONTRACT,
    IVE_CAUSAL_SENSING_MEMBER_FRAME_CONTRACT,
    IVECausalSensingMediaJobResult,
    IVECausalSensingMediaRequest,
    IVECausalSensingSource,
    ive_causal_sensing_source_availability_evidence_sha256,
    materialize_ive_causal_sensing_from_media,
    reject_attitude_before_ive_causal_sensing_access,
)
from .reference_evidence_builder import ReferenceSourceClock
from .stage_edit_feature_materializer import (
    FEATURE_MATERIALIZER_CONTRACT,
    FEATURE_MATERIALIZER_SCHEMA_VERSION,
    StageEditCausalSensingArtifacts,
    StageEditFeatureSnapshot,
)
from .stage_edit_feature_snapshot_job import (
    DEFAULT_BOUNDARY_GUARD_SECONDS,
    DEFAULT_MAX_SOURCES,
    DEFAULT_NEGATIVE_SAMPLES_PER_HOLD,
    DEFAULT_VISUAL_WORKING_WIDTH,
    FEATURE_SNAPSHOT_JOB_SCHEMA_VERSION,
    StageEditFeatureSnapshotJobResult,
    _reference_scaffold,
    _source_records,
    build_stage_edit_feature_snapshots,
)
from .stage_edit_transformer import DEFAULT_STAGE_EDIT_SCHEMA
from .vision import (
    SceneChange,
    SceneDetectionConfig,
    detect_scene_changes_in_video,
)


IVE_STAGE_EDIT_TRAINING_CORPUS_REQUEST_SCHEMA_VERSION = 1
IVE_STAGE_EDIT_TRAINING_CORPUS_REQUEST_CONTRACT = (
    "ive_stage_edit_five_reference_exact_union_causal_corpus_request_v1"
)
IVE_STAGE_EDIT_NATIVE_CUT_MANIFEST_SCHEMA_VERSION = 1
IVE_STAGE_EDIT_NATIVE_CUT_MANIFEST_CONTRACT = (
    "ive_stage_edit_song_native_cut_evidence_manifest_v1"
)
IVE_STAGE_EDIT_TRAINING_CORPUS_RECEIPT_SCHEMA_VERSION = 1
IVE_STAGE_EDIT_TRAINING_CORPUS_RECEIPT_CONTRACT = (
    "ive_stage_edit_schema4_training_corpus_launch_receipt_v1"
)
IVE_STAGE_EDIT_REFERENCE_SUPERVISION_COUNT = 5
IVE_STAGE_EDIT_REFERENCE_SUPERVISION_SCHEMA_VERSION = 3
IVE_STAGE_EDIT_REFERENCE_DATASET_KIND = (
    "reference_stage_mix_pairwise_listwise_supervision"
)
IVE_STAGE_EDIT_REFERENCE_EVIDENCE_ANALYSIS = (
    "reference_hard_cut_reverse_match_evidence_v1"
)
IVE_STAGE_EDIT_REFERENCE_PURPOSE = (
    "same_song_professional_reference_learning_10source_training_pool"
)
IVE_STAGE_EDIT_EXPANDED_REFERENCE_PURPOSE = (
    "same_song_professional_reference_learning"
)
IVE_STAGE_EDIT_ALIGNMENT_CATALOG_ANALYSIS = (
    "strict_synchronized_reference_training_source_catalog_v1"
)
IVE_STAGE_EDIT_NATIVE_CUT_ANALYSIS_BASIS = (
    "ive_attributed_component_gated_hard_cut_detector_v2"
)
IVE_STAGE_EDIT_NATIVE_CUT_SAMPLE_EVERY_SOURCE_FRAMES = 2
IVE_STAGE_EDIT_MAXIMUM_SOURCE_TIME_CELLS = 50_000
IVE_STAGE_EDIT_ACQUISITION_AUTHORITY_ALIAS_CONTRACT = (
    "verified_byte_identical_acquisition_authority_alias_v1"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SOURCE_ID = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._:-]{0,159}")
_FILE_BLOCK_BYTES = 4 * 1024 * 1024
_MAXIMUM_SUPERVISION_BYTES = 64 * 1024 * 1024
_MAXIMUM_UPSTREAM_JSON_BYTES = 64 * 1024 * 1024
_MAXIMUM_REQUEST_BYTES = 16 * 1024 * 1024
_MAXIMUM_NATIVE_CUT_MANIFEST_BYTES = 64 * 1024 * 1024
_MAXIMUM_RECEIPT_BYTES = 16 * 1024 * 1024
_FLOAT_TOLERANCE = 1e-12
_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = frozenset(
    {
        errno.EINVAL,
        errno.ENOTSUP,
        errno.EOPNOTSUPP,
        errno.ENOTTY,
    }
)


class IVEStageEditTrainingCorpusError(RuntimeError):
    """The requested production corpus cannot be proved without guessing."""


@dataclass(frozen=True, slots=True)
class TrainingCorpusArtifactIdentity:
    path: Path
    content_sha256: str
    file_sha256: str

    def __post_init__(self) -> None:
        path = Path(self.path)
        if not path.is_absolute():
            raise ValueError("training-corpus artifact path must be absolute")
        for value in (self.content_sha256, self.file_sha256):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError("training-corpus artifact identity is invalid")
        object.__setattr__(self, "path", path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "content_sha256": self.content_sha256,
            "file_sha256": self.file_sha256,
        }


@dataclass(frozen=True, slots=True)
class TrainingCorpusPreflightResult:
    request: TrainingCorpusArtifactIdentity
    song_id: str
    edit_mode: str
    target_member: str | None
    reference_count: int
    source_count: int
    union_decision_time_count: int
    source_time_cell_count: int
    decision_grid_sha256: str
    total_source_media_bytes: int
    total_source_duration_seconds: float
    estimated_additional_storage_bytes: int
    required_free_storage_bytes: int
    minimum_observed_free_storage_bytes: int
    request_payload: Mapping[str, Any] = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "song_id": self.song_id,
            "edit_mode": self.edit_mode,
            "target_member": self.target_member,
            "reference_count": self.reference_count,
            "source_count": self.source_count,
            "union_decision_time_count": self.union_decision_time_count,
            "source_time_cell_count": self.source_time_cell_count,
            "decision_grid_sha256": self.decision_grid_sha256,
            "total_source_media_bytes": self.total_source_media_bytes,
            "total_source_duration_seconds": self.total_source_duration_seconds,
            "estimated_additional_storage_bytes": (
                self.estimated_additional_storage_bytes
            ),
            "required_free_storage_bytes": self.required_free_storage_bytes,
            "minimum_observed_free_storage_bytes": (
                self.minimum_observed_free_storage_bytes
            ),
            "preflight_passed": True,
        }


@dataclass(frozen=True, slots=True)
class TrainingCorpusLaunchResult:
    preflight: TrainingCorpusPreflightResult
    native_cut_manifest: TrainingCorpusArtifactIdentity
    media_job: IVECausalSensingMediaJobResult
    feature_job: StageEditFeatureSnapshotJobResult
    receipt: TrainingCorpusArtifactIdentity

    def to_dict(self) -> dict[str, Any]:
        return {
            "preflight": self.preflight.to_dict(),
            "native_cut_manifest": self.native_cut_manifest.to_dict(),
            "causal_sensing": {
                "shooting_grammar_path": str(
                    self.media_job.shooting_grammar_path
                ),
                "shooting_grammar_content_sha256": (
                    self.media_job.shooting_grammar_content_sha256
                ),
                "member_state_graph_path": str(
                    self.media_job.member_state_graph_path
                ),
                "member_state_graph_content_sha256": (
                    self.media_job.member_state_graph_content_sha256
                ),
                "ive_performer_state_path": str(
                    self.media_job.ive_performer_state_path
                ),
                "ive_performer_state_content_sha256": (
                    self.media_job.ive_performer_state_content_sha256
                ),
            },
            "feature_manifest_path": str(self.feature_job.manifest_path),
            "snapshot_count": len(self.feature_job.snapshots),
            "receipt": self.receipt.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class _DerivedInputs:
    supervisions: tuple[Mapping[str, Any], ...]
    upstream_evidence: Mapping[str, Any]
    sources: tuple[Mapping[str, Any], ...]
    master_times: tuple[float, ...]
    decision_grid_sha256: str
    source_catalog_sha256: str
    total_media_bytes: int
    total_duration_seconds: float


@dataclass(frozen=True, slots=True)
class _StableJsonObservation:
    path: Path
    encoded: bytes
    payload: Mapping[str, Any]
    file_identity: tuple[int, int, int, int, int]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IVEStageEditTrainingCorpusError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IVEStageEditTrainingCorpusError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise IVEStageEditTrainingCorpusError(f"{name} must be finite")
    return 0.0 if number == 0.0 else number


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise IVEStageEditTrainingCorpusError(f"{name} must be a positive integer")
    return value


def _source_id(value: object) -> str:
    if not isinstance(value, str):
        raise IVEStageEditTrainingCorpusError("source ID must be a string")
    normalized = value.strip()
    if normalized != value or _SOURCE_ID.fullmatch(normalized) is None:
        raise IVEStageEditTrainingCorpusError("source ID is not canonical")
    return normalized


def _normalize_scope(
    song_id: object,
    edit_mode: EditMode | str,
    target_member: IVEMember | str | None,
) -> tuple[str, EditMode, IVEMember | None]:
    if not isinstance(song_id, str) or not song_id.strip():
        raise IVEStageEditTrainingCorpusError("song_id must be a non-empty string")
    song = song_id.strip()
    if song != song_id or any(ord(character) < 32 for character in song):
        raise IVEStageEditTrainingCorpusError("song_id must be canonical text")
    try:
        mode = (
            edit_mode
            if isinstance(edit_mode, EditMode)
            else EditMode(str(edit_mode).strip().lower())
        )
    except ValueError as exc:
        raise IVEStageEditTrainingCorpusError(
            "edit_mode must be group or member"
        ) from exc
    member: IVEMember | None
    if target_member is None:
        member = None
    else:
        try:
            member = (
                target_member
                if isinstance(target_member, IVEMember)
                else IVEMember.parse(str(target_member))
            )
        except ValueError as exc:
            raise IVEStageEditTrainingCorpusError(
                "target_member must be an IVE member"
            ) from exc
    if mode is EditMode.GROUP and member is not None:
        raise IVEStageEditTrainingCorpusError(
            "GROUP training corpus cannot bind a target member"
        )
    if mode is EditMode.MEMBER and member is None:
        raise IVEStageEditTrainingCorpusError(
            "MEMBER training corpus requires a target member"
        )
    return song, mode, member


def _absolute_lexical_path(value: Path | str, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise IVEStageEditTrainingCorpusError(f"{name} must be absolute")
    return path


def _stable_regular_file_bytes(
    value: Path | str,
    *,
    maximum_bytes: int,
    name: str,
) -> tuple[Path, bytes]:
    path = _absolute_lexical_path(value, name)
    descriptor: int | None = None
    try:
        path_before = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(path_before.st_mode):
            raise IVEStageEditTrainingCorpusError(
                f"{name} must be a non-symlink regular file"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            before = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or (path_before.st_dev, path_before.st_ino)
                != (before.st_dev, before.st_ino)
            ):
                raise IVEStageEditTrainingCorpusError(
                    f"{name} changed before reading"
                )
            encoded = stream.read(maximum_bytes + 1)
            after = os.fstat(stream.fileno())
        path_after = path.lstat()
    except IVEStageEditTrainingCorpusError:
        raise
    except OSError as exc:
        raise IVEStageEditTrainingCorpusError(
            f"cannot read {name}: {path}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(encoded) > maximum_bytes:
        raise IVEStageEditTrainingCorpusError(f"{name} exceeds its size limit")
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    path_identity = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    if (
        before_identity != after_identity
        or after_identity != path_identity
        or len(encoded) != after.st_size
        or path.is_symlink()
    ):
        raise IVEStageEditTrainingCorpusError(f"{name} changed while reading")
    return path.resolve(strict=True), encoded


def _stable_json(
    value: Path | str,
    *,
    maximum_bytes: int,
    name: str,
) -> tuple[Path, bytes, Mapping[str, Any]]:
    path, encoded = _stable_regular_file_bytes(
        value,
        maximum_bytes=maximum_bytes,
        name=name,
    )
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IVEStageEditTrainingCorpusError(
            f"{name} is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise IVEStageEditTrainingCorpusError(f"{name} root must be an object")
    return path, encoded, payload


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_json_observation(
    value: Path | str,
    *,
    maximum_bytes: int,
    name: str,
) -> _StableJsonObservation:
    """Read one canonical absolute JSON path with an outer stat fence.

    ``_stable_json`` already rejects symlinks and changes during its descriptor
    read.  The outer identity is retained so a multi-path authority check can
    also prove that no participant changed between its first and replay pass.
    """

    lexical_path = _absolute_lexical_path(value, name)
    try:
        outer_before = lexical_path.lstat()
        if lexical_path.is_symlink() or not stat.S_ISREG(outer_before.st_mode):
            raise IVEStageEditTrainingCorpusError(
                f"{name} must be a non-symlink regular file"
            )
        path, encoded, payload = _stable_json(
            lexical_path,
            maximum_bytes=maximum_bytes,
            name=name,
        )
        outer_after = lexical_path.lstat()
    except IVEStageEditTrainingCorpusError:
        raise
    except OSError as exc:
        raise IVEStageEditTrainingCorpusError(
            f"cannot observe {name}: {lexical_path}: {exc}"
        ) from exc
    if (
        lexical_path != path
        or _stat_identity(outer_before) != _stat_identity(outer_after)
        or lexical_path.is_symlink()
        or not stat.S_ISREG(outer_after.st_mode)
    ):
        raise IVEStageEditTrainingCorpusError(
            f"{name} changed during stable observation"
        )
    return _StableJsonObservation(
        path=path,
        encoded=encoded,
        payload=payload,
        file_identity=_stat_identity(outer_after),
    )


def _verify_byte_identical_acquisition_authority_alias(
    acquisition_pointers: Sequence[Mapping[str, Any]],
) -> tuple[_StableJsonObservation, Mapping[str, Any]]:
    """Verify exactly two immutable paths carrying one acquisition authority."""

    if len(acquisition_pointers) < 2:
        raise IVEStageEditTrainingCorpusError(
            "acquisition authority alias requires multiple references"
        )
    by_path: dict[str, Mapping[str, Any]] = {}
    for pointer in acquisition_pointers:
        raw_path = pointer.get("path")
        if not isinstance(raw_path, str):
            raise IVEStageEditTrainingCorpusError(
                "acquisition authority alias path is invalid"
            )
        previous = by_path.setdefault(raw_path, pointer)
        if dict(previous) != dict(pointer):
            raise IVEStageEditTrainingCorpusError(
                "acquisition authority alias path has conflicting identities"
            )
    if len(by_path) != 2:
        raise IVEStageEditTrainingCorpusError(
            "acquisition authority alias requires exactly two distinct paths"
        )
    declared_hashes = {
        _sha256(pointer.get("file_sha256"), "acquisition alias file SHA-256")
        for pointer in by_path.values()
    }
    if len(declared_hashes) != 1:
        raise IVEStageEditTrainingCorpusError(
            "acquisition authority aliases declare different file SHA-256"
        )
    declared_sha256 = next(iter(declared_hashes))
    ordered_paths = tuple(sorted(by_path))

    first_pass = tuple(
        _stable_json_observation(
            path,
            maximum_bytes=_MAXIMUM_UPSTREAM_JSON_BYTES,
            name=f"acquisition authority alias {index}",
        )
        for index, path in enumerate(ordered_paths)
    )
    replay_pass = tuple(
        _stable_json_observation(
            path,
            maximum_bytes=_MAXIMUM_UPSTREAM_JSON_BYTES,
            name=f"acquisition authority alias replay {index}",
        )
        for index, path in enumerate(ordered_paths)
    )
    for initial, replay in zip(first_pass, replay_pass, strict=True):
        if (
            initial.path != replay.path
            or initial.file_identity != replay.file_identity
            or initial.encoded != replay.encoded
            or dict(initial.payload) != dict(replay.payload)
        ):
            raise IVEStageEditTrainingCorpusError(
                "acquisition authority alias changed between verification passes"
            )
    observed_hashes = {
        hashlib.sha256(item.encoded).hexdigest() for item in first_pass
    }
    canonical_payload_hashes = {
        _canonical_sha256(item.payload) for item in first_pass
    }
    if (
        observed_hashes != {declared_sha256}
        or len(canonical_payload_hashes) != 1
        or any(item.encoded != first_pass[0].encoded for item in first_pass[1:])
    ):
        raise IVEStageEditTrainingCorpusError(
            "acquisition authority aliases are not byte-identical JSON"
        )
    canonical_payload_sha256 = next(iter(canonical_payload_hashes))
    provenance = {
        "contract": IVE_STAGE_EDIT_ACQUISITION_AUTHORITY_ALIAS_CONTRACT,
        "authority_count": 2,
        "authorities": [
            {
                "path": str(item.path),
                "file_sha256": declared_sha256,
                "canonical_payload_sha256": canonical_payload_sha256,
            }
            for item in first_pass
        ],
        "verification": {
            "absolute_regular_non_symlink_paths": True,
            "stable_pre_post_stat_replay": True,
            "declared_and_observed_file_sha256_equal": True,
            "byte_identical": True,
            "canonical_payload_identical": True,
        },
    }
    return first_pass[0], provenance


def _stable_media_sha256(value: Path | str) -> tuple[Path, str, os.stat_result]:
    path = _absolute_lexical_path(value, "source media path")
    descriptor: int | None = None
    try:
        path_before = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(path_before.st_mode):
            raise IVEStageEditTrainingCorpusError(
                "source media must be a non-symlink regular file"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            before = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or (path_before.st_dev, path_before.st_ino)
                != (before.st_dev, before.st_ino)
            ):
                raise IVEStageEditTrainingCorpusError(
                    "source media changed before hashing"
                )
            while block := stream.read(_FILE_BLOCK_BYTES):
                digest.update(block)
            after = os.fstat(stream.fileno())
        path_after = path.lstat()
    except IVEStageEditTrainingCorpusError:
        raise
    except OSError as exc:
        raise IVEStageEditTrainingCorpusError(
            f"cannot hash source media: {path}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    path_identity = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    if (
        before_identity != after_identity
        or after_identity != path_identity
        or path.is_symlink()
    ):
        raise IVEStageEditTrainingCorpusError(
            "source media changed while hashing"
        )
    return path.resolve(strict=True), digest.hexdigest(), after


def _live_media_probe(value: Path | str) -> Mapping[str, Any]:
    """Read current container bounds and reject a changing or stale file."""

    path = _absolute_lexical_path(value, "source media path")
    try:
        before = path.stat()
    except OSError as exc:
        raise IVEStageEditTrainingCorpusError(
            f"cannot stat source media before probe: {path}: {exc}"
        ) from exc
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise IVEStageEditTrainingCorpusError(
            "source media must be a non-symlink regular file"
        )
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise IVEStageEditTrainingCorpusError(
                f"cannot open source media probe: {path}"
            )
        frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count_raw = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width_raw = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_raw = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    try:
        after = path.stat()
    except OSError as exc:
        raise IVEStageEditTrainingCorpusError(
            f"cannot stat source media after probe: {path}: {exc}"
        ) from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity or path.is_symlink():
        raise IVEStageEditTrainingCorpusError(
            "source media changed during live media probe"
        )
    values = (frame_rate, frame_count_raw, width_raw, height_raw)
    if any(not math.isfinite(item) or item <= 0.0 for item in values):
        raise IVEStageEditTrainingCorpusError(
            f"source media returned invalid live bounds: {path}"
        )
    frame_count = int(round(frame_count_raw))
    width = int(round(width_raw))
    height = int(round(height_raw))
    if (
        frame_count <= 1
        or width <= 0
        or height <= 0
        or not math.isclose(frame_count_raw, frame_count, abs_tol=1.0e-6)
        or not math.isclose(width_raw, width, abs_tol=1.0e-6)
        or not math.isclose(height_raw, height, abs_tol=1.0e-6)
    ):
        raise IVEStageEditTrainingCorpusError(
            f"source media returned non-integral live bounds: {path}"
        )
    return {
        "contract": "opencv_current_container_bounds_v1",
        "frame_rate": frame_rate,
        "declared_frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": frame_count / frame_rate,
        "size_bytes": after.st_size,
    }


def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
            raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _safe_output_root(value: Path | str) -> Path:
    root = _absolute_lexical_path(value, "artifact directory")
    if root.is_symlink():
        raise IVEStageEditTrainingCorpusError(
            "artifact directory cannot be a symbolic link"
        )
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise IVEStageEditTrainingCorpusError("artifact directory is unsafe")
    return root.resolve(strict=True)


def _publish_immutable_json(
    artifact_directory: Path,
    *,
    contract: str,
    unsigned_payload: Mapping[str, Any],
) -> TrainingCorpusArtifactIdentity:
    content_sha256 = _canonical_sha256(unsigned_payload)
    payload = {
        **dict(unsigned_payload),
        "artifact_content_sha256": content_sha256,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8")
    root = _safe_output_root(artifact_directory)
    contract_root = root / contract
    if contract_root.is_symlink():
        raise IVEStageEditTrainingCorpusError("artifact contract root is unsafe")
    contract_root.mkdir(exist_ok=True)
    shard = contract_root / content_sha256[:2]
    if shard.is_symlink():
        raise IVEStageEditTrainingCorpusError("artifact shard is unsafe")
    shard.mkdir(exist_ok=True)
    if contract_root.is_symlink() or shard.is_symlink():
        raise IVEStageEditTrainingCorpusError("artifact directory changed")
    destination = shard / f"{content_sha256}.json"
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink():
            raise IVEStageEditTrainingCorpusError(
                "content-addressed artifact cannot be a symlink"
            )
        _, existing = _stable_regular_file_bytes(
            destination,
            maximum_bytes=max(len(encoded), _MAXIMUM_NATIVE_CUT_MANIFEST_BYTES),
            name="content-addressed artifact",
        )
        if not hmac.compare_digest(existing, encoded):
            raise IVEStageEditTrainingCorpusError(
                "content-addressed artifact collision"
            )
    else:
        marker = uuid.uuid4().hex.encode("ascii")
        lock = destination.with_name(f".{destination.name}.lock")
        temporary = destination.with_name(
            f".{destination.name}.{uuid.uuid4().hex}.tmp"
        )
        lock_descriptor: int | None = None
        owns_lock = False
        try:
            lock_descriptor = os.open(
                lock,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            owns_lock = True
            os.write(lock_descriptor, marker)
            os.fsync(lock_descriptor)
            os.close(lock_descriptor)
            lock_descriptor = None
            if destination.exists() or destination.is_symlink():
                if destination.is_symlink():
                    raise IVEStageEditTrainingCorpusError(
                        "concurrent artifact publication is unsafe"
                    )
                _, existing = _stable_regular_file_bytes(
                    destination,
                    maximum_bytes=max(
                        len(encoded), _MAXIMUM_NATIVE_CUT_MANIFEST_BYTES
                    ),
                    name="concurrent content-addressed artifact",
                )
                if not hmac.compare_digest(existing, encoded):
                    raise IVEStageEditTrainingCorpusError(
                        "concurrent artifact publication differs"
                    )
            else:
                with temporary.open("xb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
                _fsync_directory(shard)
        except FileExistsError as exc:
            raise IVEStageEditTrainingCorpusError(
                "artifact publication is already active"
            ) from exc
        finally:
            if lock_descriptor is not None:
                os.close(lock_descriptor)
            try:
                if temporary.exists() and not temporary.is_symlink():
                    temporary.unlink()
            except OSError:
                pass
            if owns_lock:
                try:
                    if not lock.is_symlink() and lock.read_bytes() == marker:
                        lock.unlink()
                except OSError:
                    pass
    resolved, restored = _stable_regular_file_bytes(
        destination,
        maximum_bytes=max(len(encoded), _MAXIMUM_NATIVE_CUT_MANIFEST_BYTES),
        name="published content-addressed artifact",
    )
    if not hmac.compare_digest(restored, encoded):
        raise IVEStageEditTrainingCorpusError(
            "published artifact did not replay exactly"
        )
    return TrainingCorpusArtifactIdentity(
        path=resolved,
        content_sha256=content_sha256,
        file_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _scene_detector_payload(config: SceneDetectionConfig) -> dict[str, Any]:
    return {
        "analysis_basis": IVE_STAGE_EDIT_NATIVE_CUT_ANALYSIS_BASIS,
        "threshold": config.threshold,
        "minimum_frame_difference": config.minimum_frame_difference,
        "minimum_histogram_distance": config.minimum_histogram_distance,
        "min_gap_frames": config.min_gap_frames,
        "frame_difference_weight": config.frame_difference_weight,
        "histogram_weight": config.histogram_weight,
        "difference_gain": config.difference_gain,
        "analysis_size": list(config.analysis_size),
        "histogram_bins": config.histogram_bins,
        "sample_every_source_frames": (
            IVE_STAGE_EDIT_NATIVE_CUT_SAMPLE_EVERY_SOURCE_FRAMES
        ),
    }


def _requirements_payload() -> dict[str, Any]:
    return {
        "reference_count": IVE_STAGE_EDIT_REFERENCE_SUPERVISION_COUNT,
        "reference_supervision_schema_version": (
            IVE_STAGE_EDIT_REFERENCE_SUPERVISION_SCHEMA_VERSION
        ),
        "feature_snapshot_job_schema_version": (
            FEATURE_SNAPSHOT_JOB_SCHEMA_VERSION
        ),
        "feature_materializer_schema_version": (
            FEATURE_MATERIALIZER_SCHEMA_VERSION
        ),
        "stage_edit_token_schema_version": DEFAULT_STAGE_EDIT_SCHEMA.schema_version,
        "stage_edit_token_schema_fingerprint": DEFAULT_STAGE_EDIT_SCHEMA.fingerprint,
        "raw_source_dimension": DEFAULT_STAGE_EDIT_SCHEMA.raw_source_dimension,
        "encoded_source_dimension": (
            DEFAULT_STAGE_EDIT_SCHEMA.encoded_source_dimension
        ),
        "production_causal_sensing_triple_required": True,
        "same_exact_union_grid_for_all_references": True,
    }


def _sampling_payload(
    *,
    negative_samples_per_hold: int,
    boundary_guard_seconds: float,
    max_sources: int,
    working_width: int,
) -> dict[str, Any]:
    if (
        isinstance(negative_samples_per_hold, bool)
        or not isinstance(negative_samples_per_hold, int)
        or negative_samples_per_hold < 0
        or not math.isfinite(float(boundary_guard_seconds))
        or boundary_guard_seconds < 0.0
        or isinstance(max_sources, bool)
        or not isinstance(max_sources, int)
        or max_sources < 2
        or isinstance(working_width, bool)
        or not isinstance(working_width, int)
        or working_width < 320
    ):
        raise IVEStageEditTrainingCorpusError(
            "StageEdit corpus sampling settings are invalid"
        )
    return {
        "negative_samples_per_hold": negative_samples_per_hold,
        "boundary_guard_seconds": float(boundary_guard_seconds),
        "max_sources": max_sources,
        "visual_working_width": working_width,
    }


def _pointer_from_reference(
    payload: Mapping[str, Any],
    *,
    song_id: str,
    reference_id: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    evidence = payload.get("reference_evidence_provenance")
    if not isinstance(evidence, Mapping):
        raise IVEStageEditTrainingCorpusError(
            f"reference {reference_id} has no reference evidence provenance"
        )
    provenance = evidence.get("provenance")
    caller = provenance.get("caller") if isinstance(provenance, Mapping) else None
    if (
        evidence.get("schema_version") != 1
        or evidence.get("analysis") != IVE_STAGE_EDIT_REFERENCE_EVIDENCE_ANALYSIS
        or not isinstance(caller, Mapping)
        or caller.get("artist") != "IVE"
        or caller.get("song") != song_id
        or caller.get("purpose")
        not in {
            IVE_STAGE_EDIT_REFERENCE_PURPOSE,
            IVE_STAGE_EDIT_EXPANDED_REFERENCE_PURPOSE,
        }
        or caller.get("production_render_admission_modified") is True
    ):
        raise IVEStageEditTrainingCorpusError(
            f"reference {reference_id} evidence scope is unsupported"
        )
    acquisition = caller.get("training_source_manifest") or caller.get(
        "source_manifest"
    )
    alignment = caller.get("synchronized_source_catalog")
    if not isinstance(acquisition, Mapping) or (
        alignment is not None and not isinstance(alignment, Mapping)
    ):
        raise IVEStageEditTrainingCorpusError(
            f"reference {reference_id} lacks acquisition/alignment authorities"
        )
    pointer_items: list[tuple[str, Mapping[str, Any]]] = [
        ("acquisition", acquisition)
    ]
    if isinstance(alignment, Mapping):
        pointer_items.append(("alignment", alignment))
    for name, pointer in pointer_items:
        assert isinstance(pointer, Mapping)
        if set(pointer) != {"path", "file_sha256"}:
            raise IVEStageEditTrainingCorpusError(
                f"reference {reference_id} {name} pointer schema changed"
            )
        if not isinstance(pointer.get("path"), str):
            raise IVEStageEditTrainingCorpusError(
                f"reference {reference_id} {name} pointer path is invalid"
            )
        _sha256(pointer.get("file_sha256"), f"{name} pointer file SHA-256")
    return (
        dict(acquisition),
        dict(alignment) if isinstance(alignment, Mapping) else None,
    )


def _derive_inputs(
    *,
    song_id: str,
    edit_mode: EditMode,
    target_member: IVEMember | None,
    supervision_paths: Sequence[Path],
    expected_source_count: int,
    sampling: Mapping[str, Any],
) -> _DerivedInputs:
    paths = tuple(supervision_paths)
    if len(paths) != IVE_STAGE_EDIT_REFERENCE_SUPERVISION_COUNT:
        raise IVEStageEditTrainingCorpusError(
            "production corpus requires exactly five reference supervisions"
        )
    if len(set(paths)) != len(paths):
        raise IVEStageEditTrainingCorpusError(
            "reference supervision paths must be unique"
        )
    loaded: list[tuple[Path, bytes, Mapping[str, Any]]] = []
    for index, path in enumerate(paths):
        loaded.append(
            _stable_json(
                path,
                maximum_bytes=_MAXIMUM_SUPERVISION_BYTES,
                name=f"reference supervision {index}",
            )
        )
    reference_ids: list[str] = []
    catalogs: list[list[Any]] = []
    pointers: list[
        tuple[Mapping[str, Any], Mapping[str, Any] | None]
    ] = []
    for _path, _encoded, payload in loaded:
        reference_id = _source_id(payload.get("reference_id"))
        if (
            payload.get("schema_version")
            != IVE_STAGE_EDIT_REFERENCE_SUPERVISION_SCHEMA_VERSION
            or payload.get("dataset_kind") != IVE_STAGE_EDIT_REFERENCE_DATASET_KIND
        ):
            raise IVEStageEditTrainingCorpusError(
                f"reference {reference_id} is not current schema-3 supervision"
            )
        raw_catalog = payload.get("source_catalog")
        if not isinstance(raw_catalog, list):
            raise IVEStageEditTrainingCorpusError(
                f"reference {reference_id} has no source catalog"
            )
        reference_ids.append(reference_id)
        catalogs.append(raw_catalog)
        pointers.append(
            _pointer_from_reference(
                payload,
                song_id=song_id,
                reference_id=reference_id,
            )
        )
    if len(set(reference_ids)) != len(reference_ids):
        raise IVEStageEditTrainingCorpusError("reference IDs must be unique")
    catalog_hashes = tuple(_canonical_sha256(value) for value in catalogs)
    if len(set(catalog_hashes)) != 1:
        raise IVEStageEditTrainingCorpusError(
            "five references do not share one exact source catalog"
        )
    acquisition_pointers = tuple(value[0] for value in pointers)
    acquisition_alias_required = len(
        {_canonical_sha256(value) for value in acquisition_pointers}
    ) != 1
    alignment_pointers = tuple(
        value[1] for value in pointers if value[1] is not None
    )
    if not alignment_pointers or len(
        {_canonical_sha256(value) for value in alignment_pointers}
    ) != 1:
        raise IVEStageEditTrainingCorpusError(
            "reference corpus lacks one unambiguous alignment authority"
        )
    alignment_pointer = alignment_pointers[0]
    acquisition_paths = tuple(
        sorted({str(pointer["path"]) for pointer in acquisition_pointers})
    )
    reject_attitude_before_ive_causal_sensing_access(
        song=song_id,
        source_paths=(*acquisition_paths, alignment_pointer["path"]),
    )
    acquisition_alias_provenance: Mapping[str, Any] | None = None
    if acquisition_alias_required:
        acquisition_observation, acquisition_alias_provenance = (
            _verify_byte_identical_acquisition_authority_alias(
                acquisition_pointers
            )
        )
        acquisition_path = acquisition_observation.path
        acquisition_bytes = acquisition_observation.encoded
        acquisition = acquisition_observation.payload
        acquisition_pointer: Mapping[str, Any] = {
            "path": str(acquisition_path),
            "file_sha256": hashlib.sha256(acquisition_bytes).hexdigest(),
        }
    else:
        acquisition_pointer = acquisition_pointers[0]
        acquisition_path, acquisition_bytes, acquisition = _stable_json(
            acquisition_pointer["path"],
            maximum_bytes=_MAXIMUM_UPSTREAM_JSON_BYTES,
            name="training source acquisition manifest",
        )
    alignment_path, alignment_bytes, alignment = _stable_json(
        alignment_pointer["path"],
        maximum_bytes=_MAXIMUM_UPSTREAM_JSON_BYTES,
        name="synchronized source alignment catalog",
    )
    acquisition_file_sha256 = hashlib.sha256(acquisition_bytes).hexdigest()
    alignment_file_sha256 = hashlib.sha256(alignment_bytes).hexdigest()
    if not hmac.compare_digest(
        acquisition_file_sha256,
        str(acquisition_pointer["file_sha256"]),
    ) or not hmac.compare_digest(
        alignment_file_sha256,
        str(alignment_pointer["file_sha256"]),
    ):
        raise IVEStageEditTrainingCorpusError(
            "acquisition/alignment upstream file SHA-256 changed"
        )
    raw_alignment_sources = alignment.get("sources")
    if (
        alignment.get("schema_version") != 1
        or alignment.get("analysis") != IVE_STAGE_EDIT_ALIGNMENT_CATALOG_ANALYSIS
        or alignment.get("artist") != "IVE"
        or alignment.get("song") != song_id
        or alignment.get("source_count") != expected_source_count
        or not isinstance(raw_alignment_sources, list)
        or len(raw_alignment_sources) != expected_source_count
        or raw_alignment_sources != catalogs[0]
    ):
        raise IVEStageEditTrainingCorpusError(
            "alignment catalog does not exactly reproduce supervision catalog"
        )
    selected = acquisition.get("selected")
    expected_manifest_mode = edit_mode.value
    expected_manifest_target = target_member.value if target_member is not None else None
    if (
        acquisition.get("schema_version") != 1
        or acquisition.get("group") != "IVE"
        or acquisition.get("song") != song_id
        or acquisition.get("edit_mode") != expected_manifest_mode
        or acquisition.get("target_member") != expected_manifest_target
        or acquisition.get("desired_source_count") != expected_source_count
        or not isinstance(selected, list)
        or len(selected) != expected_source_count
    ):
        raise IVEStageEditTrainingCorpusError(
            "acquisition manifest scope/source count is unsupported"
        )
    acquisition_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in selected:
        if not isinstance(raw, Mapping):
            raise IVEStageEditTrainingCorpusError(
                "acquisition selected source must be an object"
            )
        source_id = _source_id(raw.get("video_id"))
        if source_id in acquisition_by_id:
            raise IVEStageEditTrainingCorpusError(
                "acquisition manifest has duplicate source IDs"
            )
        integrity = raw.get("content_integrity")
        admission = raw.get("training_admission")
        if (
            not isinstance(raw.get("download_status"), str)
            or not str(raw.get("download_status")).startswith(
                ("cached", "downloaded")
            )
            or raw.get("training_attribution_eligible") is not True
            or not isinstance(integrity, Mapping)
            or integrity.get("contract_version") != 1
            or integrity.get("algorithm") != "sha256"
            or not isinstance(admission, Mapping)
            or admission.get("contract_version") != 1
            or admission.get("purpose")
            != "reference_reverse_attribution_only_not_production_render_admission"
            or admission.get("hard_media_contract_passed") is not True
            or admission.get("production_render_admission_modified") is not False
        ):
            raise IVEStageEditTrainingCorpusError(
                f"source {source_id} lacks admitted acquisition evidence"
            )
        _sha256(integrity.get("sha256"), "acquired media SHA-256")
        _positive_int(integrity.get("size_bytes"), "acquired media size")
        acquisition_by_id[source_id] = raw
    reject_attitude_before_ive_causal_sensing_access(
        song=song_id,
        source_ids=tuple(
            str(item.get("video_id") or "") for item in catalogs[0]
        ),
        source_paths=tuple(str(item.get("path") or "") for item in catalogs[0]),
    )
    try:
        feature_sources = _source_records(catalogs[0])
    except Exception as exc:
        raise IVEStageEditTrainingCorpusError(
            f"source catalog failed current feature-job replay: {exc}"
        ) from exc
    if len(feature_sources) != expected_source_count:
        raise IVEStageEditTrainingCorpusError(
            "current source catalog count differs from request"
        )
    try:
        scaffolds = tuple(
            _reference_scaffold(
                payload,
                song_id=song_id,
                negative_samples_per_hold=int(
                    sampling["negative_samples_per_hold"]
                ),
                boundary_guard_seconds=float(sampling["boundary_guard_seconds"]),
                max_sources=int(sampling["max_sources"]),
            )
            for _path, _encoded, payload in loaded
        )
    except Exception as exc:
        raise IVEStageEditTrainingCorpusError(
            f"reference scaffold failed current adapter replay: {exc}"
        ) from exc
    master_times = tuple(sorted({time for item in scaffolds for time in item.times}))
    if not master_times or master_times != tuple(sorted(set(master_times))):
        raise IVEStageEditTrainingCorpusError(
            "reference union decision grid is empty or non-canonical"
        )
    if len(master_times) * len(feature_sources) > IVE_STAGE_EDIT_MAXIMUM_SOURCE_TIME_CELLS:
        raise IVEStageEditTrainingCorpusError(
            "reference union grid exceeds the real causal-sensing bound"
        )
    sources: list[Mapping[str, Any]] = []
    total_media_bytes = 0
    total_duration_seconds = 0.0
    decision_grid_sha256 = _canonical_sha256(list(master_times))
    for source in feature_sources:
        acquired = acquisition_by_id.get(source.source_id)
        if acquired is None:
            raise IVEStageEditTrainingCorpusError(
                f"source {source.source_id} has no acquisition authority"
            )
        integrity = acquired["content_integrity"]
        expected_size = _positive_int(
            integrity.get("size_bytes"), "acquired media size"
        )
        expected_sha = _sha256(
            integrity.get("sha256"), "acquired media SHA-256"
        )
        if (
            str(acquired.get("local_path") or "") != str(source.path)
            or source.content_sha256 != expected_sha
            or source.path.stat().st_size != expected_size
        ):
            raise IVEStageEditTrainingCorpusError(
                f"source {source.source_id} bytes differ from acquisition evidence"
            )
        raw_catalog_record = next(
            item
            for item in catalogs[0]
            if str(item.get("video_id") or item.get("source_id") or "")
            == source.source_id
        )
        raw_media_probe = raw_catalog_record.get("media_probe")
        if not isinstance(raw_media_probe, Mapping):
            raise IVEStageEditTrainingCorpusError(
                f"source {source.source_id} has no synchronized media probe"
            )
        live_media_probe = _live_media_probe(source.path)
        catalog_width = _positive_int(
            raw_media_probe.get("width"), "catalog media width"
        )
        catalog_height = _positive_int(
            raw_media_probe.get("height"), "catalog media height"
        )
        if (
            live_media_probe["size_bytes"] != expected_size
            or live_media_probe["declared_frame_count"]
            != source.declared_frame_count
            or live_media_probe["width"] != catalog_width
            or live_media_probe["height"] != catalog_height
            or not math.isclose(
                float(live_media_probe["frame_rate"]),
                source.frame_rate,
                rel_tol=2.0e-3,
                abs_tol=2.0e-3,
            )
        ):
            raise IVEStageEditTrainingCorpusError(
                f"source {source.source_id} live media bounds differ from the "
                "synchronized catalog"
            )
        acquisition_duration_raw = acquired.get("duration_seconds")
        acquisition_duration_seconds = (
            _finite(acquisition_duration_raw, "acquired duration")
            if acquisition_duration_raw is not None
            else None
        )
        live_duration_seconds = float(live_media_probe["duration_seconds"])
        if (
            acquisition_duration_seconds is not None
            and (
                acquisition_duration_seconds <= 0.0
                or abs(acquisition_duration_seconds - live_duration_seconds)
                > max(1.05, 2.0 / source.frame_rate)
            )
        ):
            raise IVEStageEditTrainingCorpusError(
                f"source {source.source_id} acquisition duration is stale"
            )
        checked_clock = ReferenceSourceClock.from_mapping(source.clock)
        if checked_clock.to_dict() != dict(source.clock):
            raise IVEStageEditTrainingCorpusError(
                f"source {source.source_id} alignment clock does not round-trip"
            )
        mapped_times = tuple(
            checked_clock.master_to_source(value) for value in master_times
        )
        if any(
            right <= left
            for left, right in zip(mapped_times, mapped_times[1:], strict=False)
        ):
            raise IVEStageEditTrainingCorpusError(
                f"source {source.source_id} clock is not monotonic on union grid"
            )
        mapped_indices = tuple(
            int(round(value * source.frame_rate)) for value in mapped_times
        )
        feature_hard_gate = (
            source.sync_confidence >= 0.32 and source.sync_margin >= 0.025
        )
        if not feature_hard_gate:
            raise IVEStageEditTrainingCorpusError(
                f"source {source.source_id} fails the feature hard-sync gate"
            )
        unavailable_master_times = tuple(
            master_seconds
            for master_seconds, source_seconds, index in zip(
                master_times, mapped_times, mapped_indices, strict=True
            )
            if source_seconds < 0.0
            or index < 0
            or index + 1 >= source.declared_frame_count
        )
        available_indices = tuple(
            index
            for index, master_seconds in enumerate(master_times)
            if master_seconds not in set(unavailable_master_times)
        )
        if not available_indices or available_indices != tuple(
            range(available_indices[0], available_indices[-1] + 1)
        ):
            raise IVEStageEditTrainingCorpusError(
                f"source {source.source_id} availability has an interior clock gap"
            )
        clock_sha256 = _canonical_sha256(checked_clock.to_dict())
        media_probe_evidence_sha256 = _canonical_sha256(
            {
                "contract": "current_media_bounds_bound_to_catalog_v1",
                "source_id": source.source_id,
                "media_content_sha256": source.content_sha256,
                "catalog_media_probe": dict(raw_media_probe),
                "live_media_probe": dict(live_media_probe),
                "acquisition_duration_seconds": acquisition_duration_seconds,
            }
        )
        availability_evidence_sha256 = (
            ive_causal_sensing_source_availability_evidence_sha256(
                source_id=source.source_id,
                media_content_sha256=source.content_sha256,
                source_clock_sha256=clock_sha256,
                decision_grid_sha256=decision_grid_sha256,
                frame_rate=source.frame_rate,
                declared_frame_count=source.declared_frame_count,
                unavailable_master_times=unavailable_master_times,
            )
        )
        sources.append(
            {
                "source_id": source.source_id,
                "path": str(source.path),
                "media_content_sha256": source.content_sha256,
                "size_bytes": expected_size,
                "frame_rate": source.frame_rate,
                "declared_frame_count": source.declared_frame_count,
                "duration_seconds": live_duration_seconds,
                "acquisition_duration_seconds": acquisition_duration_seconds,
                "live_media_probe": dict(live_media_probe),
                "media_probe_evidence_sha256": media_probe_evidence_sha256,
                "clock": checked_clock.to_dict(),
                "clock_sha256": clock_sha256,
                "feature_hard_sync_gate_passed": feature_hard_gate,
                "unavailable_master_times": list(unavailable_master_times),
                "unavailable_master_time_evidence_sha256": (
                    availability_evidence_sha256
                ),
                "acquisition_record_sha256": _canonical_sha256(dict(acquired)),
                "alignment_catalog_record_sha256": _canonical_sha256(
                    dict(raw_catalog_record)
                ),
            }
        )
        total_media_bytes += expected_size
        total_duration_seconds += live_duration_seconds
    if set(acquisition_by_id) != {item["source_id"] for item in sources}:
        raise IVEStageEditTrainingCorpusError(
            "acquisition and alignment source sets differ"
        )
    supervision_records = tuple(
        {
            "reference_id": str(payload["reference_id"]),
            "path": str(path),
            "file_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        for path, encoded, payload in sorted(
            loaded,
            key=lambda item: str(item[2]["reference_id"]),
        )
    )
    upstream_evidence = {
        "acquisition_manifest": {
            "path": str(acquisition_path),
            "file_sha256": acquisition_file_sha256,
        },
        "alignment_catalog": {
            "path": str(alignment_path),
            "file_sha256": alignment_file_sha256,
            "source_catalog_sha256": catalog_hashes[0],
        },
    }
    if acquisition_alias_provenance is not None:
        acquisition_alias_record = dict(acquisition_alias_provenance)
        acquisition_alias_record.update(
            {
                "source_catalog_sha256": catalog_hashes[0],
                "expected_source_count": expected_source_count,
                "same_ten_source_acquisition_catalog_identity": True,
                "acquisition_catalog_identity_sha256": _canonical_sha256(
                    {
                        "source_catalog_sha256": catalog_hashes[0],
                        "acquisition_manifest": acquisition,
                    }
                ),
            }
        )
        upstream_evidence["acquisition_authority_alias"] = (
            acquisition_alias_record
        )
    return _DerivedInputs(
        supervisions=supervision_records,
        upstream_evidence=upstream_evidence,
        sources=tuple(sorted(sources, key=lambda item: str(item["source_id"]))),
        master_times=master_times,
        decision_grid_sha256=decision_grid_sha256,
        source_catalog_sha256=catalog_hashes[0],
        total_media_bytes=total_media_bytes,
        total_duration_seconds=total_duration_seconds,
    )


def _load_request_after_quarantine(
    request_path: Path | str,
) -> tuple[TrainingCorpusArtifactIdentity, Mapping[str, Any]]:
    path, encoded, payload = _stable_json(
        request_path,
        maximum_bytes=_MAXIMUM_REQUEST_BYTES,
        name="training corpus request",
    )
    digest = payload.get("artifact_content_sha256")
    _sha256(digest, "training corpus request content SHA-256")
    unsigned = dict(payload)
    unsigned.pop("artifact_content_sha256", None)
    if not hmac.compare_digest(str(digest), _canonical_sha256(unsigned)):
        raise IVEStageEditTrainingCorpusError(
            "training corpus request content SHA-256 changed"
        )
    expected_keys = {
        "schema_version",
        "contract",
        "scope",
        "created_at_utc",
        "requirements",
        "sampling",
        "supervisions",
        "upstream_evidence",
        "sources",
        "decision_grid",
        "gallery_installation",
        "native_cut_detector",
        "outputs",
    }
    if set(unsigned) != expected_keys:
        raise IVEStageEditTrainingCorpusError(
            "training corpus request schema changed"
        )
    if (
        unsigned.get("schema_version")
        != IVE_STAGE_EDIT_TRAINING_CORPUS_REQUEST_SCHEMA_VERSION
        or unsigned.get("contract")
        != IVE_STAGE_EDIT_TRAINING_CORPUS_REQUEST_CONTRACT
    ):
        raise IVEStageEditTrainingCorpusError(
            "training corpus request contract is unsupported"
        )
    identity = TrainingCorpusArtifactIdentity(
        path=path,
        content_sha256=str(digest),
        file_sha256=hashlib.sha256(encoded).hexdigest(),
    )
    return identity, unsigned


def _output_paths(payload: Mapping[str, Any]) -> tuple[Path, ...]:
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "native_cut_artifact_directory",
        "causal_sensing_artifact_directory",
        "feature_snapshot_output_directory",
        "receipt_directory",
    }:
        raise IVEStageEditTrainingCorpusError(
            "training corpus output authority is invalid"
        )
    return tuple(
        _absolute_lexical_path(value, f"output {name}")
        for name, value in sorted(outputs.items())
    )


def _nearest_existing_ancestor(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            raise IVEStageEditTrainingCorpusError(
                f"no existing storage ancestor for {path}"
            )
        current = current.parent
    if current.is_symlink():
        raise IVEStageEditTrainingCorpusError(
            f"storage ancestor cannot be a symlink: {current}"
        )
    return current


def _storage_estimate(source_time_cells: int, reference_count: int) -> tuple[int, int]:
    # Causal records, schema-v4 vectors/masks, JSON publication overhead, and
    # one retry-sized reserve.  This is deliberately conservative and does not
    # count source media because it is verified in place and never copied.
    estimated = (
        512 * 1024 * 1024
        + source_time_cells * 96 * 1024
        + reference_count * 48 * 1024 * 1024
    )
    required = max(4 * 1024**3, 2 * estimated + 1024**3)
    return estimated, required


def create_ive_stage_edit_training_corpus_request(
    *,
    song_id: str,
    edit_mode: EditMode | str,
    target_member: IVEMember | str | None,
    supervision_paths: Sequence[Path],
    expected_source_count: int,
    installation_pointer_path: Path,
    request_directory: Path,
    native_cut_artifact_directory: Path,
    causal_sensing_artifact_directory: Path,
    feature_snapshot_output_directory: Path,
    receipt_directory: Path,
    created_at_utc: str,
    negative_samples_per_hold: int = DEFAULT_NEGATIVE_SAMPLES_PER_HOLD,
    boundary_guard_seconds: float = DEFAULT_BOUNDARY_GUARD_SECONDS,
    max_sources: int = DEFAULT_MAX_SOURCES,
    working_width: int = DEFAULT_VISUAL_WORKING_WIDTH,
) -> TrainingCorpusArtifactIdentity:
    """Create one immutable launch request after strict evidence replay."""

    reject_attitude_before_ive_causal_sensing_access(
        song=song_id,
        source_paths=(
            *tuple(supervision_paths),
            installation_pointer_path,
            request_directory,
            native_cut_artifact_directory,
            causal_sensing_artifact_directory,
            feature_snapshot_output_directory,
            receipt_directory,
        ),
    )
    song, mode, member = _normalize_scope(song_id, edit_mode, target_member)
    if (
        isinstance(expected_source_count, bool)
        or not isinstance(expected_source_count, int)
        or not 2 <= expected_source_count <= max_sources
    ):
        raise IVEStageEditTrainingCorpusError(
            "expected_source_count is outside the model contract"
        )
    sampling = _sampling_payload(
        negative_samples_per_hold=negative_samples_per_hold,
        boundary_guard_seconds=boundary_guard_seconds,
        max_sources=max_sources,
        working_width=working_width,
    )
    derived = _derive_inputs(
        song_id=song,
        edit_mode=mode,
        target_member=member,
        supervision_paths=tuple(
            _absolute_lexical_path(path, "reference supervision path")
            for path in supervision_paths
        ),
        expected_source_count=expected_source_count,
        sampling=sampling,
    )
    try:
        installation = load_ive_causal_sensing_installation(
            _absolute_lexical_path(
                installation_pointer_path,
                "gallery installation pointer",
            )
        )
    except Exception as exc:
        raise IVEStageEditTrainingCorpusError(
            f"private gallery installation failed strict replay: {exc}"
        ) from exc
    output_payload = {
        "native_cut_artifact_directory": str(
            _absolute_lexical_path(
                native_cut_artifact_directory, "native-cut artifact directory"
            )
        ),
        "causal_sensing_artifact_directory": str(
            _absolute_lexical_path(
                causal_sensing_artifact_directory,
                "causal-sensing artifact directory",
            )
        ),
        "feature_snapshot_output_directory": str(
            _absolute_lexical_path(
                feature_snapshot_output_directory,
                "feature snapshot output directory",
            )
        ),
        "receipt_directory": str(
            _absolute_lexical_path(receipt_directory, "receipt directory")
        ),
    }
    # IVECausalSensingMediaRequest owns the canonical timestamp parser.  Build
    # no partial request here, but reject non-canonical forms identically.
    if not isinstance(created_at_utc, str) or not created_at_utc.endswith("Z"):
        raise IVEStageEditTrainingCorpusError(
            "created_at_utc must be canonical UTC RFC3339"
        )
    plan = {
        "schema_version": IVE_STAGE_EDIT_TRAINING_CORPUS_REQUEST_SCHEMA_VERSION,
        "contract": IVE_STAGE_EDIT_TRAINING_CORPUS_REQUEST_CONTRACT,
        "scope": {
            "artist": "IVE",
            "song_id": song,
            "edit_mode": mode.value,
            "target_member": member.value if member is not None else None,
        },
        "created_at_utc": created_at_utc,
        "requirements": _requirements_payload(),
        "sampling": sampling,
        "supervisions": list(derived.supervisions),
        "upstream_evidence": dict(derived.upstream_evidence),
        "sources": list(derived.sources),
        "decision_grid": {
            "master_times": list(derived.master_times),
            "decision_grid_sha256": derived.decision_grid_sha256,
            "decision_time_count": len(derived.master_times),
            "source_time_cell_count": (
                len(derived.master_times) * len(derived.sources)
            ),
        },
        "gallery_installation": {
            "pointer_path": str(installation.pointer_path),
            "pointer_file_sha256": installation.pointer_file_sha256,
            "installation_sha256": installation.installation_sha256,
            "gallery_artifact_path": str(installation.gallery_artifact_path),
            "gallery_content_sha256": installation.gallery_content_sha256,
        },
        "native_cut_detector": _scene_detector_payload(SceneDetectionConfig()),
        "outputs": output_payload,
    }
    # Constructing the typed downstream request validates the timestamp, scope,
    # grid and source/time bound before publishing a launch authority.  Native
    # cuts are placeholders here only because launch derives and hash-binds the
    # real evidence before the production request is built.
    try:
        IVECausalSensingMediaRequest(
            song=song,
            edit_mode=mode,
            target_member=member,
            sources=tuple(
                IVECausalSensingSource(
                    source_id=str(source["source_id"]),
                    path=Path(str(source["path"])),
                    media_content_sha256=str(source["media_content_sha256"]),
                    clock=ReferenceSourceClock.from_mapping(source["clock"]),
                    native_cut_source_seconds=(),
                    native_cut_evidence_sha256="0" * 64,
                    unavailable_master_times=tuple(
                        source["unavailable_master_times"]
                    ),
                    availability_frame_rate=float(source["frame_rate"]),
                    availability_declared_frame_count=int(
                        source["declared_frame_count"]
                    ),
                    unavailable_master_time_evidence_sha256=str(
                        source[
                            "unavailable_master_time_evidence_sha256"
                        ]
                    ),
                )
                for source in derived.sources
            ),
            master_decision_times=derived.master_times,
            gallery_artifact_path=installation.gallery_artifact_path,
            gallery_content_sha256=installation.gallery_content_sha256,
            artifact_directory=Path(
                output_payload["causal_sensing_artifact_directory"]
            ),
            created_at_utc=created_at_utc,
        )
    except Exception as exc:
        raise IVEStageEditTrainingCorpusError(
            f"request does not satisfy current causal-sensing contract: {exc}"
        ) from exc
    return _publish_immutable_json(
        request_directory,
        contract=IVE_STAGE_EDIT_TRAINING_CORPUS_REQUEST_CONTRACT,
        unsigned_payload=plan,
    )


def preflight_ive_stage_edit_training_corpus_request(
    *,
    song_id: str,
    request_path: Path,
) -> TrainingCorpusPreflightResult:
    """Strictly replay a request and every current upstream byte authority."""

    reject_attitude_before_ive_causal_sensing_access(
        song=song_id,
        source_paths=(request_path,),
    )
    request_identity, request = _load_request_after_quarantine(request_path)
    scope = request.get("scope")
    if not isinstance(scope, Mapping):
        raise IVEStageEditTrainingCorpusError("request scope is invalid")
    embedded_paths: list[object] = []
    for key in ("supervisions", "sources"):
        raw_items = request.get(key)
        if isinstance(raw_items, list):
            embedded_paths.extend(
                item.get("path") for item in raw_items if isinstance(item, Mapping)
            )
    upstream = request.get("upstream_evidence")
    if isinstance(upstream, Mapping):
        embedded_paths.extend(
            value.get("path")
            for value in upstream.values()
            if isinstance(value, Mapping)
        )
        acquisition_alias = upstream.get("acquisition_authority_alias")
        if isinstance(acquisition_alias, Mapping):
            raw_authorities = acquisition_alias.get("authorities")
            if isinstance(raw_authorities, list):
                embedded_paths.extend(
                    item.get("path")
                    for item in raw_authorities
                    if isinstance(item, Mapping)
                )
    gallery = request.get("gallery_installation")
    if isinstance(gallery, Mapping):
        embedded_paths.extend(
            (gallery.get("pointer_path"), gallery.get("gallery_artifact_path"))
        )
    try:
        embedded_paths.extend(_output_paths(request))
    except IVEStageEditTrainingCorpusError:
        raise
    reject_attitude_before_ive_causal_sensing_access(
        song=scope.get("song_id"),
        source_ids=tuple(
            item.get("source_id")
            for item in request.get("sources", [])
            if isinstance(item, Mapping)
        ),
        source_paths=tuple(embedded_paths),
    )
    song, mode, member = _normalize_scope(
        scope.get("song_id"),
        scope.get("edit_mode"),
        scope.get("target_member"),
    )
    if scope.get("artist") != "IVE" or song != song_id:
        raise IVEStageEditTrainingCorpusError(
            "caller song metadata differs from request scope"
        )
    if request.get("requirements") != _requirements_payload():
        raise IVEStageEditTrainingCorpusError(
            "request was built for a different StageEdit schema/code contract"
        )
    detector = request.get("native_cut_detector")
    if detector != _scene_detector_payload(SceneDetectionConfig()):
        raise IVEStageEditTrainingCorpusError(
            "request native-cut detector contract is stale"
        )
    sampling = request.get("sampling")
    if not isinstance(sampling, Mapping):
        raise IVEStageEditTrainingCorpusError("request sampling is invalid")
    canonical_sampling = _sampling_payload(
        negative_samples_per_hold=sampling.get("negative_samples_per_hold"),
        boundary_guard_seconds=sampling.get("boundary_guard_seconds"),
        max_sources=sampling.get("max_sources"),
        working_width=sampling.get("visual_working_width"),
    )
    if dict(sampling) != canonical_sampling:
        raise IVEStageEditTrainingCorpusError("request sampling is non-canonical")
    raw_supervisions = request.get("supervisions")
    raw_sources = request.get("sources")
    if not isinstance(raw_supervisions, list) or not isinstance(raw_sources, list):
        raise IVEStageEditTrainingCorpusError(
            "request source/supervision arrays are invalid"
        )
    derived = _derive_inputs(
        song_id=song,
        edit_mode=mode,
        target_member=member,
        supervision_paths=tuple(
            _absolute_lexical_path(item.get("path"), "supervision request path")
            for item in raw_supervisions
            if isinstance(item, Mapping)
        ),
        expected_source_count=len(raw_sources),
        sampling=canonical_sampling,
    )
    if (
        list(derived.supervisions) != raw_supervisions
        or dict(derived.upstream_evidence) != request.get("upstream_evidence")
        or list(derived.sources) != raw_sources
    ):
        raise IVEStageEditTrainingCorpusError(
            "request acquisition/alignment/media evidence is no longer exact"
        )
    grid = request.get("decision_grid")
    canonical_grid = {
        "master_times": list(derived.master_times),
        "decision_grid_sha256": derived.decision_grid_sha256,
        "decision_time_count": len(derived.master_times),
        "source_time_cell_count": len(derived.master_times) * len(derived.sources),
    }
    if grid != canonical_grid:
        raise IVEStageEditTrainingCorpusError(
            "request decision union grid is no longer exact"
        )
    if not isinstance(gallery, Mapping) or set(gallery) != {
        "pointer_path",
        "pointer_file_sha256",
        "installation_sha256",
        "gallery_artifact_path",
        "gallery_content_sha256",
    }:
        raise IVEStageEditTrainingCorpusError(
            "request gallery installation binding is invalid"
        )
    try:
        installed = load_ive_causal_sensing_installation(
            Path(str(gallery["pointer_path"])),
            expected_pointer_file_sha256=str(gallery["pointer_file_sha256"]),
        )
    except Exception as exc:
        raise IVEStageEditTrainingCorpusError(
            f"request gallery installation no longer replays: {exc}"
        ) from exc
    if {
        "pointer_path": str(installed.pointer_path),
        "pointer_file_sha256": installed.pointer_file_sha256,
        "installation_sha256": installed.installation_sha256,
        "gallery_artifact_path": str(installed.gallery_artifact_path),
        "gallery_content_sha256": installed.gallery_content_sha256,
    } != dict(gallery):
        raise IVEStageEditTrainingCorpusError(
            "request gallery installation identity changed"
        )
    source_time_cells = len(derived.sources) * len(derived.master_times)
    estimated_storage, required_free = _storage_estimate(
        source_time_cells,
        len(derived.supervisions),
    )
    free_by_device: dict[int, int] = {}
    for output in _output_paths(request):
        ancestor = _nearest_existing_ancestor(output)
        device = ancestor.stat().st_dev
        free_by_device.setdefault(device, shutil.disk_usage(ancestor).free)
    minimum_free = min(free_by_device.values())
    if minimum_free < required_free:
        raise IVEStageEditTrainingCorpusError(
            "insufficient free storage for causal triple plus five schema4 snapshots: "
            f"need {required_free} bytes, observed {minimum_free}"
        )
    return TrainingCorpusPreflightResult(
        request=request_identity,
        song_id=song,
        edit_mode=mode.value,
        target_member=member.value if member is not None else None,
        reference_count=len(derived.supervisions),
        source_count=len(derived.sources),
        union_decision_time_count=len(derived.master_times),
        source_time_cell_count=source_time_cells,
        decision_grid_sha256=derived.decision_grid_sha256,
        total_source_media_bytes=derived.total_media_bytes,
        total_source_duration_seconds=derived.total_duration_seconds,
        estimated_additional_storage_bytes=estimated_storage,
        required_free_storage_bytes=required_free,
        minimum_observed_free_storage_bytes=minimum_free,
        request_payload=request,
    )


def _scene_change_payload(change: SceneChange, frame_rate: float) -> dict[str, Any]:
    if not isinstance(change, SceneChange):
        raise IVEStageEditTrainingCorpusError(
            "native-cut detector returned an unsupported record"
        )
    if change.frame_index < 0:
        raise IVEStageEditTrainingCorpusError("native-cut frame index is invalid")
    values = (
        _finite(change.score, "native-cut score"),
        _finite(change.frame_difference, "native-cut frame difference"),
        _finite(change.histogram_distance, "native-cut histogram distance"),
    )
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise IVEStageEditTrainingCorpusError(
            "native-cut detector score is outside [0, 1]"
        )
    return {
        "frame_index": change.frame_index,
        "source_seconds": change.frame_index / frame_rate,
        "score": values[0],
        "frame_difference": values[1],
        "histogram_distance": values[2],
    }


def _materialize_native_cut_manifest(
    *,
    preflight: TrainingCorpusPreflightResult,
    detector: Callable[..., tuple[SceneChange, ...]],
) -> tuple[TrainingCorpusArtifactIdentity, tuple[Mapping[str, Any], ...]]:
    request = preflight.request_payload
    detector_payload = request["native_cut_detector"]
    config = SceneDetectionConfig(
        threshold=detector_payload["threshold"],
        minimum_frame_difference=detector_payload["minimum_frame_difference"],
        minimum_histogram_distance=detector_payload[
            "minimum_histogram_distance"
        ],
        min_gap_frames=detector_payload["min_gap_frames"],
        frame_difference_weight=detector_payload["frame_difference_weight"],
        histogram_weight=detector_payload["histogram_weight"],
        difference_gain=detector_payload["difference_gain"],
        analysis_size=tuple(detector_payload["analysis_size"]),
        histogram_bins=detector_payload["histogram_bins"],
    )
    records: list[Mapping[str, Any]] = []
    for source in request["sources"]:
        path = Path(source["path"])
        changes = tuple(
            detector(
                path,
                config=config,
                sample_every=IVE_STAGE_EDIT_NATIVE_CUT_SAMPLE_EVERY_SOURCE_FRAMES,
            )
        )
        change_payloads = tuple(
            _scene_change_payload(change, float(source["frame_rate"]))
            for change in changes
        )
        frame_indices = tuple(item["frame_index"] for item in change_payloads)
        if frame_indices != tuple(sorted(set(frame_indices))):
            raise IVEStageEditTrainingCorpusError(
                f"native cuts for {source['source_id']} are not strict"
            )
        if any(index >= source["declared_frame_count"] for index in frame_indices):
            raise IVEStageEditTrainingCorpusError(
                f"native cuts for {source['source_id']} exceed declared media"
            )
        resolved, media_sha256, media_stat = _stable_media_sha256(path)
        if (
            str(resolved) != source["path"]
            or media_sha256 != source["media_content_sha256"]
            or media_stat.st_size != source["size_bytes"]
        ):
            raise IVEStageEditTrainingCorpusError(
                f"source {source['source_id']} changed during native-cut decode"
            )
        unsigned_record = {
            "source_id": source["source_id"],
            "media_content_sha256": source["media_content_sha256"],
            "clock_sha256": source["clock_sha256"],
            "decision_grid_sha256": preflight.decision_grid_sha256,
            "frame_rate": source["frame_rate"],
            "declared_frame_count": source["declared_frame_count"],
            "live_media_probe": dict(source["live_media_probe"]),
            "media_probe_evidence_sha256": source[
                "media_probe_evidence_sha256"
            ],
            "acquisition_record_sha256": source[
                "acquisition_record_sha256"
            ],
            "alignment_catalog_record_sha256": source[
                "alignment_catalog_record_sha256"
            ],
            "unavailable_master_times": list(
                source["unavailable_master_times"]
            ),
            "unavailable_master_time_evidence_sha256": source[
                "unavailable_master_time_evidence_sha256"
            ],
            "availability_reason": (
                "mapped_native_current_plus_following_frame_outside_media_edges"
            ),
            "interior_clock_gaps_allowed": False,
            "detector": detector_payload,
            "changes": list(change_payloads),
            "raw_frames_persisted": False,
        }
        records.append(
            {
                **unsigned_record,
                "native_cut_evidence_sha256": _canonical_sha256(unsigned_record),
            }
        )
    manifest_unsigned = {
        "schema_version": IVE_STAGE_EDIT_NATIVE_CUT_MANIFEST_SCHEMA_VERSION,
        "contract": IVE_STAGE_EDIT_NATIVE_CUT_MANIFEST_CONTRACT,
        "scope": dict(request["scope"]),
        "request_content_sha256": preflight.request.content_sha256,
        "decision_grid_sha256": preflight.decision_grid_sha256,
        "detector": detector_payload,
        "sources": records,
        "source_count": preflight.source_count,
        "future_native_cuts_used": False,
        "raw_frames_persisted": False,
    }
    artifact = _publish_immutable_json(
        Path(request["outputs"]["native_cut_artifact_directory"]),
        contract=IVE_STAGE_EDIT_NATIVE_CUT_MANIFEST_CONTRACT,
        unsigned_payload=manifest_unsigned,
    )
    return artifact, tuple(records)


def _verify_feature_job(
    *,
    preflight: TrainingCorpusPreflightResult,
    media_job: IVECausalSensingMediaJobResult,
    feature_job: StageEditFeatureSnapshotJobResult,
) -> Mapping[str, Any]:
    request = preflight.request_payload
    manifest_path, manifest_bytes, manifest = _stable_json(
        feature_job.manifest_path,
        maximum_bytes=_MAXIMUM_REQUEST_BYTES,
        name="schema4 feature job manifest",
    )
    if manifest_path != feature_job.manifest_path.resolve(strict=True):
        raise IVEStageEditTrainingCorpusError(
            "feature job returned a different manifest path"
        )
    unsigned = dict(manifest)
    manifest_sha256 = unsigned.pop("manifest_sha256", None)
    if (
        manifest.get("schema_version") != FEATURE_SNAPSHOT_JOB_SCHEMA_VERSION
        or not isinstance(manifest_sha256, str)
        or _canonical_sha256(unsigned) != manifest_sha256
    ):
        raise IVEStageEditTrainingCorpusError(
            "feature job manifest is not current schema4"
        )
    grid = request["decision_grid"]
    statistics = manifest.get("statistics")
    if (
        manifest.get("song_id") != preflight.song_id
        or manifest.get("source_count") != preflight.source_count
        or not isinstance(statistics, Mapping)
        or statistics.get("reference_count") != preflight.reference_count
        or statistics.get("union_decision_time_count")
        != preflight.union_decision_time_count
        or manifest.get("causal_sensing_binding")
        != media_job.stage_edit_artifacts.binding.to_dict()
    ):
        raise IVEStageEditTrainingCorpusError(
            "schema4 feature manifest changed source/grid/triple authority"
        )
    gate = manifest.get("production_causal_sensing_gate")
    expected_gate = _expected_feature_snapshot_production_gate(
        media_job=media_job,
        preflight=preflight,
    )
    if gate != expected_gate:
        raise IVEStageEditTrainingCorpusError(
            "schema4 feature manifest production causal gate differs"
        )
    union_times: set[float] = set()
    snapshot_receipts: list[Mapping[str, Any]] = []
    source_contract = {
        str(item["source_id"]): item for item in request["sources"]
    }
    if len(feature_job.snapshots) != preflight.reference_count:
        raise IVEStageEditTrainingCorpusError(
            "feature job did not produce exactly five snapshots"
        )
    for reference_id, path in sorted(feature_job.snapshots.items()):
        _resolved, _bytes, payload = _stable_json(
            path,
            # Schema4 carries the complete 396-dimensional source tokens for
            # every union-grid cell and legitimately exceeds the compact
            # request/receipt envelope.  Keep the bounded upstream-artifact
            # limit while still rejecting oversized or unstable files.
            maximum_bytes=_MAXIMUM_UPSTREAM_JSON_BYTES,
            name=f"schema4 feature snapshot {reference_id}",
        )
        try:
            snapshot = StageEditFeatureSnapshot.from_dict(payload)
        except Exception as exc:
            raise IVEStageEditTrainingCorpusError(
                f"schema4 feature snapshot {reference_id} failed strict replay: {exc}"
            ) from exc
        if (
            snapshot.schema_version != FEATURE_MATERIALIZER_SCHEMA_VERSION
            or snapshot.causal_sensing_binding.to_dict()
            != media_job.stage_edit_artifacts.binding.to_dict()
            or snapshot.require_production_causal_sensing(
                edit_mode=preflight.edit_mode,
                target_member=preflight.target_member,
            )
            != expected_gate
        ):
            raise IVEStageEditTrainingCorpusError(
                f"schema4 feature snapshot {reference_id} lost triple binding"
            )
        union_times.update(
            float(token.master_time_seconds) for token in snapshot.sequence.tokens
        )
        for token in snapshot.sequence.tokens:
            candidate_by_source = {
                candidate.source_id: candidate
                for candidate in token.source_candidates
            }
            if set(candidate_by_source) != set(source_contract):
                raise IVEStageEditTrainingCorpusError(
                    f"schema4 feature snapshot {reference_id} changed the source set"
                )
            for source_id, source in source_contract.items():
                candidate = candidate_by_source[source_id]
                expected_available = token.master_time_seconds not in set(
                    source["unavailable_master_times"]
                )
                expected_hard_gate = expected_available and bool(
                    source["feature_hard_sync_gate_passed"]
                )
                if (
                    candidate.available != expected_available
                    or candidate.hard_gate_passed != expected_hard_gate
                ):
                    raise IVEStageEditTrainingCorpusError(
                        "schema4 candidate availability/hard gate differs from "
                        f"the hash-bound source-clock mask: {reference_id}/"
                        f"{source_id}/{token.master_time_seconds:.9f}"
                    )
        snapshot_receipts.append(
            {
                "reference_id": reference_id,
                "path": str(Path(path).resolve(strict=True)),
                "snapshot_sha256": snapshot.snapshot_sha256,
                "token_count": len(snapshot.sequence.tokens),
            }
        )
    if tuple(sorted(union_times)) != tuple(grid["master_times"]):
        raise IVEStageEditTrainingCorpusError(
            "five schema4 snapshots do not reproduce the exact union grid"
        )
    return {
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_sha256": manifest_sha256,
        "snapshots": snapshot_receipts,
    }


def _expected_feature_snapshot_production_gate(
    *,
    media_job: IVECausalSensingMediaJobResult,
    preflight: TrainingCorpusPreflightResult,
) -> dict[str, Any]:
    """Reproduce the complete current snapshot gate, not its binding subset."""

    return {
        **media_job.stage_edit_artifacts.binding.require_production_scope(
            edit_mode=preflight.edit_mode,
            target_member=preflight.target_member,
        ),
        # StageEditFeatureSnapshot.require_production_causal_sensing appends
        # these three snapshot authorities to the lower-level triple gate.
        "feature_snapshot_schema_version": FEATURE_MATERIALIZER_SCHEMA_VERSION,
        "token_schema_fingerprint": DEFAULT_STAGE_EDIT_SCHEMA.fingerprint,
        "feature_materializer_contract": FEATURE_MATERIALIZER_CONTRACT,
    }


def _strictly_reload_existing_causal_media_job(
    request: IVECausalSensingMediaRequest,
    artifact_paths: tuple[Path, Path, Path],
) -> IVECausalSensingMediaJobResult:
    """Strictly resume after a completed causal triple, never from loose JSON.

    A failed feature build must not force another expensive media pass when the
    immutable triple already proves the exact source/grid/gallery authority.
    Every source-time key and every production provenance component is checked
    before the artifacts are exposed to the feature materializer.
    """

    reject_attitude_before_ive_causal_sensing_access(
        song=request.song,
        source_ids=tuple(source.source_id for source in request.sources),
        source_paths=(
            *(source.path for source in request.sources),
            *artifact_paths,
        ),
    )
    if len(artifact_paths) != 3:
        raise IVEStageEditTrainingCorpusError(
            "existing causal sensing requires exactly three artifact paths"
        )
    normalized = tuple(Path(path) for path in artifact_paths)
    digests: list[str] = []
    for name, path in zip(
        ("shooting grammar", "member-state graph", "performer-state"),
        normalized,
        strict=True,
    ):
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise IVEStageEditTrainingCorpusError(
                f"existing {name} artifact path is missing or unsafe"
            )
        digest = path.stem
        if _SHA256.fullmatch(digest) is None:
            raise IVEStageEditTrainingCorpusError(
                f"existing {name} path is not content addressed"
            )
        digests.append(digest)

    artifacts = StageEditCausalSensingArtifacts(
        shooting_grammar_path=normalized[0],
        shooting_grammar_content_sha256=digests[0],
        member_state_graph_path=normalized[1],
        member_state_graph_content_sha256=digests[1],
        target_member=request.target_member,
        ive_performer_state_path=normalized[2],
        ive_performer_state_content_sha256=digests[2],
    )
    artifacts.binding.require_production_scope(
        edit_mode=request.edit_mode.value,
        target_member=request.target_member,
    )
    source_ids = tuple(source.source_id for source in request.sources)
    artifacts.validate_source_pool(source_ids)

    grammar = artifacts._shooting_grammar
    member = artifacts._member_state
    performer = artifacts._ive_performer_state
    if grammar is None or member is None or performer is None:
        raise IVEStageEditTrainingCorpusError(
            "existing causal sensing triple did not strictly load"
        )
    expected_keys = {
        (timestamp, source.source_id)
        for timestamp in request.master_decision_times
        for source in request.sources
    }
    shooting_keys = {
        (record.master_time_seconds, record.source_id)
        for record in grammar.records
    }
    performer_keys = {
        (record.master_time_seconds, record.source_id)
        for record in performer.records
    }
    if shooting_keys != expected_keys or performer_keys != expected_keys:
        raise IVEStageEditTrainingCorpusError(
            "existing causal triple differs from the exact decision grid"
        )
    expected_available_keys = {
        (timestamp, source.source_id)
        for timestamp in request.master_decision_times
        for source in request.sources
        if timestamp not in set(source.unavailable_master_times)
    }
    member_keys = {
        (record.timestamp_seconds, record.source_id)
        for record in member.graph.observations
    }
    if member_keys != expected_available_keys:
        raise IVEStageEditTrainingCorpusError(
            "existing member-state graph differs from exact available cells"
        )

    expected_source_content = {
        source.source_id: source.media_content_sha256 for source in request.sources
    }
    member_source_content = {
        source.source_id: source.source_content_sha256
        for source in member.provenance.sources
    }
    performer_source_content = {
        source.source_id: source.source_content_sha256
        for source in performer.provenance.sources
    }
    if (
        member_source_content != expected_source_content
        or performer_source_content != expected_source_content
    ):
        raise IVEStageEditTrainingCorpusError(
            "existing causal triple source-media identities changed"
        )

    expected_upstream: dict[str, str] = {
        "decision_grid": request.decision_grid_sha256,
        "gallery_artifact": request.gallery_content_sha256,
        "song_scope": request.song_scope_sha256,
    }
    decoded_names: set[str] = set()
    for index, source in enumerate(request.sources):
        prefix = f"source_{index:03d}"
        expected_upstream.update(
            {
                f"{prefix}_media": source.media_content_sha256,
                f"{prefix}_clock": source.clock_sha256,
                f"{prefix}_native_cuts": source.native_cut_manifest_sha256,
                f"{prefix}_availability": _canonical_sha256(
                    {
                        "contract": IVE_CAUSAL_SENSING_AVAILABILITY_CONTRACT,
                        "bound": source.availability_bound,
                        "evidence_sha256": (
                            source.unavailable_master_time_evidence_sha256
                        ),
                        "unavailable_master_times": list(
                            source.unavailable_master_times
                        ),
                    }
                ),
            }
        )
        decoded_names.add(f"{prefix}_decoded_frames")
    upstream = dict(grammar.provenance.upstream_sha256)
    if any(upstream.get(name) != digest for name, digest in expected_upstream.items()):
        raise IVEStageEditTrainingCorpusError(
            "existing shooting grammar request provenance changed"
        )
    if set(upstream) != set(expected_upstream) | decoded_names or any(
        _SHA256.fullmatch(upstream.get(name, "")) is None for name in decoded_names
    ):
        raise IVEStageEditTrainingCorpusError(
            "existing shooting grammar decoded-frame provenance changed"
        )
    required_contracts = {
        IVE_CAUSAL_SENSING_MEDIA_JOB_CONTRACT,
        IVE_CAUSAL_SENSING_FRAME_CONTRACT,
        IVE_CAUSAL_SENSING_MEMBER_FRAME_CONTRACT,
        IVE_CAUSAL_SENSING_AVAILABILITY_CONTRACT,
    }
    if not required_contracts.issubset(grammar.provenance.contracts):
        raise IVEStageEditTrainingCorpusError(
            "existing shooting grammar lacks production media contracts"
        )

    accepted_count = sum(
        observation.face_identity_id is not None
        for observation in member.graph.observations
    )
    return IVECausalSensingMediaJobResult(
        shooting_grammar_path=normalized[0].resolve(strict=True),
        shooting_grammar_content_sha256=digests[0],
        member_state_graph_path=normalized[1].resolve(strict=True),
        member_state_graph_content_sha256=digests[1],
        ive_performer_state_path=normalized[2].resolve(strict=True),
        ive_performer_state_content_sha256=digests[2],
        stage_edit_artifacts=artifacts,
        source_count=len(request.sources),
        decision_count=len(request.master_decision_times),
        member_observation_count=len(member.graph.observations),
        accepted_identity_observation_count=accepted_count,
        unavailable_source_time_count=sum(
            len(source.unavailable_master_times) for source in request.sources
        ),
    )


def launch_ive_stage_edit_training_corpus_request(
    *,
    song_id: str,
    request_path: Path,
    existing_causal_sensing_paths: tuple[Path, Path, Path] | None = None,
    detector: Callable[..., tuple[SceneChange, ...]] = detect_scene_changes_in_video,
    media_materializer: Callable[..., IVECausalSensingMediaJobResult] = (
        materialize_ive_causal_sensing_from_media
    ),
    feature_builder: Callable[..., StageEditFeatureSnapshotJobResult] = (
        build_stage_edit_feature_snapshots
    ),
) -> TrainingCorpusLaunchResult:
    """Launch one real triple and five schema4 snapshots from an exact request."""

    reject_attitude_before_ive_causal_sensing_access(
        song=song_id,
        source_paths=(
            request_path,
            *(existing_causal_sensing_paths or ()),
        ),
    )
    preflight = preflight_ive_stage_edit_training_corpus_request(
        song_id=song_id,
        request_path=request_path,
    )
    request = preflight.request_payload
    native_manifest, native_records = _materialize_native_cut_manifest(
        preflight=preflight,
        detector=detector,
    )
    cut_by_source = {str(item["source_id"]): item for item in native_records}
    sources = tuple(
        IVECausalSensingSource(
            source_id=str(raw["source_id"]),
            path=Path(str(raw["path"])),
            media_content_sha256=str(raw["media_content_sha256"]),
            clock=ReferenceSourceClock.from_mapping(raw["clock"]),
            native_cut_source_seconds=tuple(
                float(change["source_seconds"])
                for change in cut_by_source[str(raw["source_id"])]["changes"]
            ),
            native_cut_evidence_sha256=str(
                cut_by_source[str(raw["source_id"])][
                    "native_cut_evidence_sha256"
                ]
            ),
            unavailable_master_times=tuple(raw["unavailable_master_times"]),
            availability_frame_rate=float(raw["frame_rate"]),
            availability_declared_frame_count=int(
                raw["declared_frame_count"]
            ),
            unavailable_master_time_evidence_sha256=str(
                raw["unavailable_master_time_evidence_sha256"]
            ),
        )
        for raw in request["sources"]
    )
    gallery = request["gallery_installation"]
    scope = request["scope"]
    media_request = IVECausalSensingMediaRequest(
        song=preflight.song_id,
        edit_mode=preflight.edit_mode,
        target_member=preflight.target_member,
        sources=sources,
        master_decision_times=tuple(request["decision_grid"]["master_times"]),
        gallery_artifact_path=Path(gallery["gallery_artifact_path"]),
        gallery_content_sha256=str(gallery["gallery_content_sha256"]),
        artifact_directory=Path(
            request["outputs"]["causal_sensing_artifact_directory"]
        ),
        created_at_utc=str(request["created_at_utc"]),
    )
    if media_request.decision_grid_sha256 != preflight.decision_grid_sha256:
        raise IVEStageEditTrainingCorpusError(
            "real causal request changed the exact union grid"
        )
    media_job = (
        _strictly_reload_existing_causal_media_job(
            media_request,
            existing_causal_sensing_paths,
        )
        if existing_causal_sensing_paths is not None
        else media_materializer(media_request)
    )
    if not isinstance(media_job, IVECausalSensingMediaJobResult):
        raise IVEStageEditTrainingCorpusError(
            "causal media materializer returned the wrong result type"
        )
    if (
        media_job.source_count != preflight.source_count
        or media_job.decision_count != preflight.union_decision_time_count
        or media_job.unavailable_source_time_count
        != sum(
            len(source["unavailable_master_times"])
            for source in request["sources"]
        )
    ):
        raise IVEStageEditTrainingCorpusError(
            "causal triple does not exactly cover the request union grid"
        )
    sampling = request["sampling"]
    feature_job = feature_builder(
        tuple(Path(item["path"]) for item in request["supervisions"]),
        song_id=preflight.song_id,
        output_directory=Path(
            request["outputs"]["feature_snapshot_output_directory"]
        ),
        expected_source_count=preflight.source_count,
        negative_samples_per_hold=sampling["negative_samples_per_hold"],
        boundary_guard_seconds=sampling["boundary_guard_seconds"],
        max_sources=sampling["max_sources"],
        working_width=sampling["visual_working_width"],
        causal_sensing=media_job.stage_edit_artifacts,
        edit_mode=preflight.edit_mode,
        target_member=preflight.target_member,
        inspection_only_without_causal_sensing=False,
    )
    if not isinstance(feature_job, StageEditFeatureSnapshotJobResult):
        raise IVEStageEditTrainingCorpusError(
            "feature builder returned the wrong result type"
        )
    feature_receipt = _verify_feature_job(
        preflight=preflight,
        media_job=media_job,
        feature_job=feature_job,
    )
    receipt_payload = {
        "schema_version": IVE_STAGE_EDIT_TRAINING_CORPUS_RECEIPT_SCHEMA_VERSION,
        "contract": IVE_STAGE_EDIT_TRAINING_CORPUS_RECEIPT_CONTRACT,
        "scope": dict(scope),
        "request": preflight.request.to_dict(),
        "decision_grid_sha256": preflight.decision_grid_sha256,
        "decision_time_count": preflight.union_decision_time_count,
        "source_count": preflight.source_count,
        "reference_count": preflight.reference_count,
        "native_cut_manifest": native_manifest.to_dict(),
        "causal_sensing": {
            "shooting_grammar_path": str(media_job.shooting_grammar_path),
            "shooting_grammar_content_sha256": (
                media_job.shooting_grammar_content_sha256
            ),
            "member_state_graph_path": str(media_job.member_state_graph_path),
            "member_state_graph_content_sha256": (
                media_job.member_state_graph_content_sha256
            ),
            "ive_performer_state_path": str(media_job.ive_performer_state_path),
            "ive_performer_state_content_sha256": (
                media_job.ive_performer_state_content_sha256
            ),
            "source_count": media_job.source_count,
            "decision_count": media_job.decision_count,
            "unavailable_source_time_count": (
                media_job.unavailable_source_time_count
            ),
        },
        "feature_job": feature_receipt,
        "same_exact_triple_bound_to_all_five_snapshots": True,
        "raw_frames_persisted_by_coordinator": False,
    }
    receipt = _publish_immutable_json(
        Path(request["outputs"]["receipt_directory"]),
        contract=IVE_STAGE_EDIT_TRAINING_CORPUS_RECEIPT_CONTRACT,
        unsigned_payload=receipt_payload,
    )
    return TrainingCorpusLaunchResult(
        preflight=preflight,
        native_cut_manifest=native_manifest,
        media_job=media_job,
        feature_job=feature_job,
        receipt=receipt,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create, preflight, or launch one exact-grid IVE StageEdit schema4 "
            "training corpus"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create-request")
    create.add_argument("--song-id", required=True)
    create.add_argument("--edit-mode", choices=("group", "member"), required=True)
    create.add_argument("--target-member")
    create.add_argument("--supervision", action="append", required=True, type=Path)
    create.add_argument("--expected-source-count", required=True, type=int)
    create.add_argument("--installation-pointer", required=True, type=Path)
    create.add_argument("--request-directory", required=True, type=Path)
    create.add_argument("--native-cut-artifact-directory", required=True, type=Path)
    create.add_argument(
        "--causal-sensing-artifact-directory", required=True, type=Path
    )
    create.add_argument(
        "--feature-snapshot-output-directory", required=True, type=Path
    )
    create.add_argument("--receipt-directory", required=True, type=Path)
    create.add_argument("--created-at-utc", required=True)
    create.add_argument(
        "--negative-samples-per-hold",
        type=int,
        default=DEFAULT_NEGATIVE_SAMPLES_PER_HOLD,
    )
    create.add_argument(
        "--boundary-guard-seconds",
        type=float,
        default=DEFAULT_BOUNDARY_GUARD_SECONDS,
    )
    create.add_argument("--max-sources", type=int, default=DEFAULT_MAX_SOURCES)
    create.add_argument(
        "--working-width", type=int, default=DEFAULT_VISUAL_WORKING_WIDTH
    )
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--song-id", required=True)
    preflight.add_argument("--request", required=True, type=Path)
    launch = subparsers.add_parser("launch")
    launch.add_argument("--song-id", required=True)
    launch.add_argument("--request", required=True, type=Path)
    launch.add_argument("--reuse-shooting-grammar", type=Path)
    launch.add_argument("--reuse-member-state-graph", type=Path)
    launch.add_argument("--reuse-ive-performer-state", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "create-request":
        artifact = create_ive_stage_edit_training_corpus_request(
            song_id=args.song_id,
            edit_mode=args.edit_mode,
            target_member=args.target_member,
            supervision_paths=args.supervision,
            expected_source_count=args.expected_source_count,
            installation_pointer_path=args.installation_pointer,
            request_directory=args.request_directory,
            native_cut_artifact_directory=args.native_cut_artifact_directory,
            causal_sensing_artifact_directory=(
                args.causal_sensing_artifact_directory
            ),
            feature_snapshot_output_directory=(
                args.feature_snapshot_output_directory
            ),
            receipt_directory=args.receipt_directory,
            created_at_utc=args.created_at_utc,
            negative_samples_per_hold=args.negative_samples_per_hold,
            boundary_guard_seconds=args.boundary_guard_seconds,
            max_sources=args.max_sources,
            working_width=args.working_width,
        )
        print(json.dumps(artifact.to_dict(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "preflight":
        result = preflight_ive_stage_edit_training_corpus_request(
            song_id=args.song_id,
            request_path=args.request,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "launch":
        reuse_values = (
            args.reuse_shooting_grammar,
            args.reuse_member_state_graph,
            args.reuse_ive_performer_state,
        )
        if any(value is not None for value in reuse_values) and not all(
            value is not None for value in reuse_values
        ):
            raise IVEStageEditTrainingCorpusError(
                "causal resume requires all three --reuse artifact paths"
            )
        result = launch_ive_stage_edit_training_corpus_request(
            song_id=args.song_id,
            request_path=args.request,
            existing_causal_sensing_paths=(
                tuple(reuse_values) if all(value is not None for value in reuse_values) else None
            ),
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
        return 0
    raise AssertionError("unreachable training corpus command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IVE_STAGE_EDIT_NATIVE_CUT_MANIFEST_CONTRACT",
    "IVE_STAGE_EDIT_NATIVE_CUT_MANIFEST_SCHEMA_VERSION",
    "IVE_STAGE_EDIT_TRAINING_CORPUS_RECEIPT_CONTRACT",
    "IVE_STAGE_EDIT_TRAINING_CORPUS_RECEIPT_SCHEMA_VERSION",
    "IVE_STAGE_EDIT_TRAINING_CORPUS_REQUEST_CONTRACT",
    "IVE_STAGE_EDIT_TRAINING_CORPUS_REQUEST_SCHEMA_VERSION",
    "IVEStageEditTrainingCorpusError",
    "TrainingCorpusArtifactIdentity",
    "TrainingCorpusLaunchResult",
    "TrainingCorpusPreflightResult",
    "build_parser",
    "create_ive_stage_edit_training_corpus_request",
    "launch_ive_stage_edit_training_corpus_request",
    "main",
    "preflight_ive_stage_edit_training_corpus_request",
]
