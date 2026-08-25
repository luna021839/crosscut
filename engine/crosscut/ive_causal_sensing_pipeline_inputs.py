"""Exact-input bridge from the automatic app to real IVE causal sensing.

This module deliberately sits before StageEdit inference snapshot creation. It
derives the decision grid once, binds request-ordered media identities, converts
the checked pipeline alignment/time-warp solution into the exact clock type
consumed by the real-media sensing job, and carries the already-detected native
scene cuts without recomputation.

ATTITUDE quarantine is the first operation in the public materializer. Gallery
installation discovery, path resolution, stat, hashing, model loading, and
decoding all happen only after that metadata-only guard succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence
import uuid

from .audio import AlignmentResult, SourceAlignment
from .draft import GroupDraftTimeline
from .ive import EditMode, IVEMember
from .ive_causal_sensing_installation import (
    InstalledIVECausalSensingGallery,
    discover_ive_causal_sensing_installation,
)
from .ive_causal_sensing_media_job import (
    IVECausalSensingMediaJobResult,
    IVECausalSensingMediaRequest,
    IVECausalSensingRuntime,
    IVECausalSensingSource,
    materialize_ive_causal_sensing_from_media,
    reject_attitude_before_ive_causal_sensing_access,
)
from .reference_evidence_builder import ClockControlPoint, ReferenceSourceClock
from .rhythm import RhythmAnalysis
from .section_director import SectionPlan
from .stage_edit_inference_snapshot import (
    StageEditInferenceSnapshotResult,
    build_stage_edit_inference_decision_times,
)
from .timewarp import TimeWarpMap


IVE_CAUSAL_SENSING_PIPELINE_INPUT_CONTRACT = (
    "automatic_pipeline_exact_grid_clock_cut_media_binding_v1"
)
IVE_CAUSAL_SENSING_NATIVE_CUT_CONTRACT = (
    "pipeline_detected_native_scene_cut_source_times_v1"
)
IVE_CAUSAL_SENSING_PIPELINE_AUDIT_SCHEMA_VERSION = 1
IVE_CAUSAL_SENSING_PIPELINE_AUDIT_CONTRACT = (
    "automatic_pipeline_real_causal_sensing_exact_binding_audit_v1"
)
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


class IVECausalSensingPipelineInputError(RuntimeError):
    """The current automatic edit inputs cannot be bound without ambiguity."""


@dataclass(frozen=True, slots=True)
class IVECausalSensingPipelineAuditArtifact:
    path: Path
    content_sha256: str
    file_sha256: str
    snapshot_sha256: str

    def __post_init__(self) -> None:
        if not Path(self.path).is_absolute():
            raise ValueError("causal-sensing pipeline audit path must be absolute")
        for value in (
            self.content_sha256,
            self.file_sha256,
            self.snapshot_sha256,
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("causal-sensing pipeline audit identity is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": IVE_CAUSAL_SENSING_PIPELINE_AUDIT_CONTRACT,
            "path": str(self.path),
            "content_sha256": self.content_sha256,
            "file_sha256": self.file_sha256,
            "snapshot_sha256": self.snapshot_sha256,
        }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


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


def _stable_small_file_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    source = Path(path)
    try:
        before = source.lstat()
        if source.is_symlink() or not stat.S_ISREG(before.st_mode):
            raise IVECausalSensingPipelineInputError(
                "causal-sensing audit input must be a non-symlink regular file"
            )
        encoded = source.read_bytes()
        after = source.lstat()
    except IVECausalSensingPipelineInputError:
        raise
    except OSError as exc:
        raise IVECausalSensingPipelineInputError(
            f"cannot read causal-sensing audit input: {source.name}: {exc}"
        ) from exc
    if len(encoded) > maximum_bytes:
        raise IVECausalSensingPipelineInputError(
            "causal-sensing audit input exceeds its size limit"
        )
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(encoded) != after.st_size
        or source.is_symlink()
    ):
        raise IVECausalSensingPipelineInputError(
            "causal-sensing audit input changed while reading"
        )
    return encoded


def _safe_audit_directory(parent: Path, name: str) -> Path:
    destination = parent / name
    if destination.is_symlink():
        raise IVECausalSensingPipelineInputError(
            "causal-sensing audit directory cannot be a symlink"
        )
    destination.mkdir(exist_ok=True)
    if destination.is_symlink() or not destination.is_dir():
        raise IVECausalSensingPipelineInputError(
            "causal-sensing audit directory is unsafe"
        )
    return destination


def _publish_pipeline_audit(
    artifact_directory: Path,
    payload: Mapping[str, Any],
) -> IVECausalSensingPipelineAuditArtifact:
    unsigned = dict(payload)
    content_sha256 = _canonical_sha256(unsigned)
    complete = {**unsigned, "artifact_content_sha256": content_sha256}
    encoded = json.dumps(
        complete,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8")
    root = Path(artifact_directory).expanduser()
    if not root.is_absolute():
        raise IVECausalSensingPipelineInputError(
            "causal-sensing audit root must be absolute"
        )
    if root.is_symlink():
        raise IVECausalSensingPipelineInputError(
            "causal-sensing audit root cannot be a symlink"
        )
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise IVECausalSensingPipelineInputError(
            "causal-sensing audit root is unsafe"
        )
    contract_root = _safe_audit_directory(
        root,
        IVE_CAUSAL_SENSING_PIPELINE_AUDIT_CONTRACT,
    )
    prefix_root = _safe_audit_directory(contract_root, content_sha256[:2])
    destination = prefix_root / f"{content_sha256}.json"
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink():
            raise IVECausalSensingPipelineInputError(
                "causal-sensing audit artifact cannot be a symlink"
            )
        existing = _stable_small_file_bytes(
            destination,
            maximum_bytes=4 * 1024 * 1024,
        )
        if existing != encoded:
            raise IVECausalSensingPipelineInputError(
                "content-addressed causal-sensing audit collision"
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
                if destination.is_symlink() or _stable_small_file_bytes(
                    destination,
                    maximum_bytes=4 * 1024 * 1024,
                ) != encoded:
                    raise IVECausalSensingPipelineInputError(
                        "concurrent causal-sensing audit publication differs"
                    )
            else:
                with temporary.open("xb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(destination)
                _fsync_directory(prefix_root)
        except FileExistsError as exc:
            raise IVECausalSensingPipelineInputError(
                "causal-sensing audit publication is already active"
            ) from exc
        finally:
            if lock_descriptor is not None:
                os.close(lock_descriptor)
            if temporary.exists() and not temporary.is_symlink():
                temporary.unlink()
            if owns_lock and lock.exists() and not lock.is_symlink():
                try:
                    current = lock.read_bytes()
                except OSError:
                    current = b""
                if current == marker:
                    lock.unlink()
    restored = _stable_small_file_bytes(
        destination,
        maximum_bytes=4 * 1024 * 1024,
    )
    if restored != encoded:
        raise IVECausalSensingPipelineInputError(
            "written causal-sensing audit failed exact replay"
        )
    snapshot_sha256 = complete.get("inference_snapshot_sha256")
    if not isinstance(snapshot_sha256, str):
        raise IVECausalSensingPipelineInputError(
            "causal-sensing audit lost its inference snapshot identity"
        )
    return IVECausalSensingPipelineAuditArtifact(
        path=destination.resolve(strict=True),
        content_sha256=content_sha256,
        file_sha256=hashlib.sha256(encoded).hexdigest(),
        snapshot_sha256=snapshot_sha256,
    )


def _stable_media_sha256(path: Path) -> str:
    source = Path(path)
    descriptor: int | None = None
    try:
        path_before = source.lstat()
        if source.is_symlink() or not stat.S_ISREG(path_before.st_mode):
            raise IVECausalSensingPipelineInputError(
                "causal-sensing source must be a non-symlink regular file"
            )
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(source, flags)
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            before = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or (path_before.st_dev, path_before.st_ino)
                != (before.st_dev, before.st_ino)
            ):
                raise IVECausalSensingPipelineInputError(
                    "causal-sensing source changed before hashing"
                )
            while block := stream.read(_FILE_BLOCK_BYTES):
                digest.update(block)
            after = os.fstat(stream.fileno())
        path_after = source.lstat()
    except IVECausalSensingPipelineInputError:
        raise
    except OSError as exc:
        raise IVECausalSensingPipelineInputError(
            f"cannot hash causal-sensing source: {source.name}: {exc}"
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
        or source.is_symlink()
    ):
        raise IVECausalSensingPipelineInputError(
            "causal-sensing source changed while hashing"
        )
    return digest.hexdigest()


def _finite_time(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IVECausalSensingPipelineInputError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise IVECausalSensingPipelineInputError(
            f"{name} must be finite and non-negative"
        )
    return 0.0 if number == 0.0 else number


def _reference_clock(
    *,
    alignment: SourceAlignment,
    timewarp: TimeWarpMap | None,
) -> ReferenceSourceClock:
    alignment_sha256 = _canonical_sha256(alignment.to_dict())
    if timewarp is None:
        return ReferenceSourceClock.scalar(
            alignment.offset_seconds,
            basis="automatic_pipeline_checked_scalar_alignment_v1",
            provenance={
                "alignment_sha256": alignment_sha256,
                "pipeline_contract": IVE_CAUSAL_SENSING_PIPELINE_INPUT_CONTRACT,
            },
        )
    if not isinstance(timewarp, TimeWarpMap):
        raise IVECausalSensingPipelineInputError(
            "source timewarp must be a checked TimeWarpMap or None"
        )
    if not math.isclose(
        timewarp.global_offset_seconds,
        alignment.offset_seconds,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise IVECausalSensingPipelineInputError(
            "causal-sensing timewarp and alignment offset disagree"
        )
    timewarp_payload = timewarp.to_dict()
    provenance = {
        "alignment_sha256": alignment_sha256,
        "timewarp_sha256": _canonical_sha256(timewarp_payload),
        "pipeline_contract": IVE_CAUSAL_SENSING_PIPELINE_INPUT_CONTRACT,
    }
    if timewarp.scalar_fallback:
        return ReferenceSourceClock.scalar(
            timewarp.global_offset_seconds,
            basis="automatic_pipeline_checked_scalar_timewarp_fallback_v1",
            provenance=provenance,
        )
    return ReferenceSourceClock(
        basis="automatic_pipeline_checked_piecewise_timewarp_v1",
        global_offset_seconds=timewarp.global_offset_seconds,
        control_points=tuple(
            ClockControlPoint(
                master_seconds=point.master_seconds,
                source_seconds=point.source_seconds,
            )
            for point in timewarp.control_points
        ),
        provenance=provenance,
    )


def _canonical_native_cuts(
    values: Sequence[float],
    *,
    source_id: str,
) -> tuple[tuple[float, ...], str]:
    cuts = tuple(
        _finite_time(value, f"native cut time for {source_id}") for value in values
    )
    if cuts != tuple(sorted(set(cuts))):
        raise IVECausalSensingPipelineInputError(
            "native scene-cut source times must be strict and canonical"
        )
    evidence_sha256 = _canonical_sha256(
        {
            "contract": IVE_CAUSAL_SENSING_NATIVE_CUT_CONTRACT,
            "source_id": source_id,
            "source_seconds": list(cuts),
        }
    )
    return cuts, evidence_sha256


@dataclass(frozen=True, slots=True)
class PreparedIVECausalSensingPipelineRequest:
    request: IVECausalSensingMediaRequest
    installation: InstalledIVECausalSensingGallery
    source_ids: tuple[str, ...]
    source_content_sha256: tuple[str, ...]
    source_clock_sha256: tuple[str, ...]
    native_cut_evidence_sha256: tuple[str, ...]
    decision_times: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, IVECausalSensingMediaRequest):
            raise TypeError("request must be IVECausalSensingMediaRequest")
        if not isinstance(self.installation, InstalledIVECausalSensingGallery):
            raise TypeError("installation must be a verified local installation")
        source_count = len(self.source_ids)
        if not (
            source_count
            == len(self.source_content_sha256)
            == len(self.source_clock_sha256)
            == len(self.native_cut_evidence_sha256)
            == len(self.request.sources)
        ):
            raise ValueError("prepared causal-sensing source bindings differ")
        if self.source_ids != tuple(item.source_id for item in self.request.sources):
            raise ValueError("prepared causal-sensing source IDs differ")
        if self.decision_times != self.request.master_decision_times:
            raise ValueError("prepared causal-sensing decision grids differ")

    def public_audit_dict(self) -> dict[str, Any]:
        """Return privacy-minimal binding metadata without any local path."""

        return {
            "contract": IVE_CAUSAL_SENSING_PIPELINE_INPUT_CONTRACT,
            "installation_sha256": self.installation.installation_sha256,
            "installation_pointer_file_sha256": (
                self.installation.pointer_file_sha256
            ),
            "gallery_content_sha256": self.installation.gallery_content_sha256,
            "source_ids": list(self.source_ids),
            "source_content_sha256": list(self.source_content_sha256),
            "source_clock_sha256": list(self.source_clock_sha256),
            "native_cut_evidence_sha256": list(
                self.native_cut_evidence_sha256
            ),
            "decision_grid_sha256": self.request.decision_grid_sha256,
            "decision_time_count": len(self.decision_times),
            "local_paths_persisted": False,
            "private_gallery_vectors_persisted": False,
        }


@dataclass(frozen=True, slots=True)
class IVECausalSensingPipelineResult:
    prepared: PreparedIVECausalSensingPipelineRequest
    media_job: IVECausalSensingMediaJobResult

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, PreparedIVECausalSensingPipelineRequest):
            raise TypeError("prepared request has the wrong type")
        if not isinstance(self.media_job, IVECausalSensingMediaJobResult):
            raise TypeError("media_job has the wrong type")
        if self.media_job.source_count != len(self.prepared.source_ids):
            raise ValueError("real causal-sensing source count changed")
        if self.media_job.decision_count != len(self.prepared.decision_times):
            raise ValueError("real causal-sensing decision count changed")

    @property
    def stage_edit_artifacts(self):
        return self.media_job.stage_edit_artifacts

    def require_exact_inference_snapshot_binding(
        self,
        snapshot: StageEditInferenceSnapshotResult,
    ) -> Mapping[str, Any]:
        """Fail unless inference replayed this exact grid and source authority."""

        if not isinstance(snapshot, StageEditInferenceSnapshotResult):
            raise TypeError("snapshot must be StageEditInferenceSnapshotResult")
        if snapshot.source_ids != self.prepared.source_ids:
            raise IVECausalSensingPipelineInputError(
                "StageEdit inference source IDs differ from real causal sensing"
            )
        if snapshot.source_content_sha256 != self.prepared.source_content_sha256:
            raise IVECausalSensingPipelineInputError(
                "StageEdit inference media SHA-256 differs from real causal sensing"
            )
        if snapshot.decision_times != self.prepared.decision_times:
            raise IVECausalSensingPipelineInputError(
                "StageEdit inference decision grid differs from real causal sensing"
            )
        expected_binding = self.stage_edit_artifacts.binding
        if snapshot.snapshot.causal_sensing_binding != expected_binding:
            raise IVECausalSensingPipelineInputError(
                "StageEdit inference causal-sensing artifact binding changed"
            )
        gate = snapshot.snapshot.require_production_causal_sensing(
            edit_mode=self.prepared.request.edit_mode,
            target_member=self.prepared.request.target_member,
        )
        return {
            **self.prepared.public_audit_dict(),
            "shooting_grammar_content_sha256": (
                self.media_job.shooting_grammar_content_sha256
            ),
            "member_state_graph_content_sha256": (
                self.media_job.member_state_graph_content_sha256
            ),
            "ive_performer_state_content_sha256": (
                self.media_job.ive_performer_state_content_sha256
            ),
            "production_causal_sensing_gate": dict(gate),
            "exact_inference_snapshot_binding_verified": True,
        }


def write_ive_causal_sensing_pipeline_audit(
    *,
    pipeline_result: IVECausalSensingPipelineResult,
    inference_snapshot: StageEditInferenceSnapshotResult,
    artifact_directory: Path,
) -> IVECausalSensingPipelineAuditArtifact:
    """Persist one immutable proof that real sensing and inference are exact."""

    if not isinstance(pipeline_result, IVECausalSensingPipelineResult):
        raise TypeError("pipeline_result must be IVECausalSensingPipelineResult")
    if not isinstance(inference_snapshot, StageEditInferenceSnapshotResult):
        raise TypeError(
            "inference_snapshot must be StageEditInferenceSnapshotResult"
        )
    exact_binding = pipeline_result.require_exact_inference_snapshot_binding(
        inference_snapshot
    )
    payload = {
        "schema_version": IVE_CAUSAL_SENSING_PIPELINE_AUDIT_SCHEMA_VERSION,
        "contract": IVE_CAUSAL_SENSING_PIPELINE_AUDIT_CONTRACT,
        "inference_snapshot_sha256": (
            inference_snapshot.snapshot.snapshot_sha256
        ),
        "inference_snapshot_file_sha256": _stable_media_sha256(
            inference_snapshot.snapshot_path
        ),
        "inference_manifest_file_sha256": _stable_media_sha256(
            inference_snapshot.manifest_path
        ),
        "exact_binding": exact_binding,
        "privacy": {
            "local_source_paths_persisted": False,
            "private_gallery_path_persisted": False,
            "private_gallery_vectors_persisted": False,
            "raw_frames_persisted": False,
            "query_embeddings_persisted": False,
        },
    }
    return _publish_pipeline_audit(artifact_directory, payload)


def prepare_ive_causal_sensing_pipeline_request(
    *,
    song: str,
    edit_mode: EditMode | str,
    target_member: IVEMember | str | None,
    sources: Sequence[Path],
    alignment: AlignmentResult,
    alignment_hard_gates: Sequence[bool],
    source_timewarps: Sequence[TimeWarpMap | None],
    source_scene_cut_times: Sequence[Sequence[float]],
    timing_draft: GroupDraftTimeline,
    rhythm: RhythmAnalysis,
    section_plan: SectionPlan,
    maximum_sequence_length: int,
    maximum_sources: int,
    installation: InstalledIVECausalSensingGallery,
    artifact_directory: Path,
    created_at_utc: str,
) -> PreparedIVECausalSensingPipelineRequest:
    """Build the typed request after quarantine and installation discovery."""

    raw_sources = tuple(Path(path).expanduser() for path in sources)
    reject_attitude_before_ive_causal_sensing_access(
        song=song,
        source_paths=raw_sources,
    )
    if not isinstance(installation, InstalledIVECausalSensingGallery):
        raise TypeError("installation must be a verified local installation")
    try:
        normalized_sources = tuple(
            source.resolve(strict=True) for source in raw_sources
        )
    except OSError as exc:
        raise IVECausalSensingPipelineInputError(
            f"causal-sensing source cannot be resolved: {exc}"
        ) from exc
    if len(normalized_sources) < 2 or len(set(normalized_sources)) != len(
        normalized_sources
    ):
        raise IVECausalSensingPipelineInputError(
            "causal sensing needs at least two unique source files"
        )
    if (
        isinstance(maximum_sources, bool)
        or not isinstance(maximum_sources, int)
        or not 2 <= len(normalized_sources) <= maximum_sources
    ):
        raise IVECausalSensingPipelineInputError(
            "causal-sensing source count exceeds the model contract"
        )
    if not isinstance(alignment, AlignmentResult):
        raise TypeError("alignment must be AlignmentResult")
    alignments = tuple(alignment.alignments)
    hard_gates = tuple(alignment_hard_gates)
    timewarps = tuple(source_timewarps)
    scene_cuts = tuple(tuple(values) for values in source_scene_cut_times)
    if not (
        len(alignments)
        == len(hard_gates)
        == len(timewarps)
        == len(scene_cuts)
        == len(normalized_sources)
    ):
        raise IVECausalSensingPipelineInputError(
            "causal-sensing source authorities are not one-to-one"
        )
    if any(not isinstance(value, bool) for value in hard_gates) or not all(
        hard_gates
    ):
        raise IVECausalSensingPipelineInputError(
            "real causal sensing requires every source alignment hard gate"
        )
    try:
        aligned_paths = tuple(
            Path(item.source).expanduser().resolve(strict=True)
            for item in alignments
        )
    except OSError as exc:
        raise IVECausalSensingPipelineInputError(
            f"causal-sensing alignment source cannot be resolved: {exc}"
        ) from exc
    if aligned_paths != normalized_sources:
        raise IVECausalSensingPipelineInputError(
            "causal-sensing alignment order differs from current media"
        )
    decision_times = build_stage_edit_inference_decision_times(
        timing_draft=timing_draft,
        rhythm=rhythm,
        section_plan=section_plan,
        maximum_sequence_length=maximum_sequence_length,
    )
    content_hashes = tuple(
        _stable_media_sha256(source) for source in normalized_sources
    )
    source_ids = tuple(
        f"source-{index:02d}-{digest[:16]}"
        for index, digest in enumerate(content_hashes)
    )
    typed_sources: list[IVECausalSensingSource] = []
    clock_hashes: list[str] = []
    cut_hashes: list[str] = []
    for source_id, path, digest, source_alignment, timewarp, raw_cuts in zip(
        source_ids,
        normalized_sources,
        content_hashes,
        alignments,
        timewarps,
        scene_cuts,
        strict=True,
    ):
        clock = _reference_clock(
            alignment=source_alignment,
            timewarp=timewarp,
        )
        for master_seconds in decision_times:
            expected = (
                master_seconds + source_alignment.offset_seconds
                if timewarp is None
                else timewarp.master_to_source(master_seconds)
            )
            actual = clock.master_to_source(master_seconds)
            if not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=_FLOAT_TOLERANCE,
            ):
                raise IVECausalSensingPipelineInputError(
                    "converted causal-sensing source clock changed the pipeline map"
                )
        cuts, cut_sha256 = _canonical_native_cuts(
            raw_cuts,
            source_id=source_id,
        )
        typed = IVECausalSensingSource(
            source_id=source_id,
            path=path,
            media_content_sha256=digest,
            clock=clock,
            native_cut_source_seconds=cuts,
            native_cut_evidence_sha256=cut_sha256,
        )
        typed_sources.append(typed)
        clock_hashes.append(typed.clock_sha256)
        cut_hashes.append(cut_sha256)
    request = IVECausalSensingMediaRequest(
        song=song,
        edit_mode=edit_mode,
        target_member=target_member,
        sources=tuple(typed_sources),
        master_decision_times=decision_times,
        gallery_artifact_path=installation.gallery_artifact_path,
        gallery_content_sha256=installation.gallery_content_sha256,
        artifact_directory=Path(artifact_directory).expanduser(),
        created_at_utc=created_at_utc,
    )
    return PreparedIVECausalSensingPipelineRequest(
        request=request,
        installation=installation,
        source_ids=source_ids,
        source_content_sha256=content_hashes,
        source_clock_sha256=tuple(clock_hashes),
        native_cut_evidence_sha256=tuple(cut_hashes),
        decision_times=decision_times,
    )


def materialize_ive_causal_sensing_for_stage_edit_pipeline(
    *,
    song: str,
    edit_mode: EditMode | str,
    target_member: IVEMember | str | None,
    sources: Sequence[Path],
    alignment: AlignmentResult,
    alignment_hard_gates: Sequence[bool],
    source_timewarps: Sequence[TimeWarpMap | None],
    source_scene_cut_times: Sequence[Sequence[float]],
    timing_draft: GroupDraftTimeline,
    rhythm: RhythmAnalysis,
    section_plan: SectionPlan,
    maximum_sequence_length: int,
    maximum_sources: int,
    artifact_directory: Path,
    created_at_utc: str,
    runtime: IVECausalSensingRuntime | None = None,
) -> IVECausalSensingPipelineResult:
    """Discover, bind, derive, and strictly return one real sensing triple."""

    raw_sources = tuple(Path(path).expanduser() for path in sources)
    # This must remain the first operation: no gallery or filesystem access may
    # precede the permanent ATTITUDE quarantine.
    reject_attitude_before_ive_causal_sensing_access(
        song=song,
        source_paths=raw_sources,
    )
    installation = discover_ive_causal_sensing_installation()
    prepared = prepare_ive_causal_sensing_pipeline_request(
        song=song,
        edit_mode=edit_mode,
        target_member=target_member,
        sources=raw_sources,
        alignment=alignment,
        alignment_hard_gates=alignment_hard_gates,
        source_timewarps=source_timewarps,
        source_scene_cut_times=source_scene_cut_times,
        timing_draft=timing_draft,
        rhythm=rhythm,
        section_plan=section_plan,
        maximum_sequence_length=maximum_sequence_length,
        maximum_sources=maximum_sources,
        installation=installation,
        artifact_directory=artifact_directory,
        created_at_utc=created_at_utc,
    )
    result = materialize_ive_causal_sensing_from_media(
        prepared.request,
        runtime=runtime,
    )
    return IVECausalSensingPipelineResult(prepared=prepared, media_job=result)


__all__ = [
    "IVE_CAUSAL_SENSING_NATIVE_CUT_CONTRACT",
    "IVE_CAUSAL_SENSING_PIPELINE_AUDIT_CONTRACT",
    "IVE_CAUSAL_SENSING_PIPELINE_AUDIT_SCHEMA_VERSION",
    "IVE_CAUSAL_SENSING_PIPELINE_INPUT_CONTRACT",
    "IVECausalSensingPipelineAuditArtifact",
    "IVECausalSensingPipelineInputError",
    "IVECausalSensingPipelineResult",
    "PreparedIVECausalSensingPipelineRequest",
    "materialize_ive_causal_sensing_for_stage_edit_pipeline",
    "prepare_ive_causal_sensing_pipeline_request",
    "write_ive_causal_sensing_pipeline_audit",
]
