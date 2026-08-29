"""Decoded ``-8..+8`` boundary evidence for joint timeline selection.

The joint dynamic program deliberately owns no media decoder.  This adapter
binds its :class:`~crosscut.joint_timeline.BoundaryMeasurementKey` to the same
descriptor extraction and professional transition gates used by visual
boundary repair.  One exact measurement produces all seventeen offset audits;
the immutable result is memoized before it is returned to the optimizer.

Decoded frames and model vectors are process-local only.  The public audit
surface contains scalar mismatch evidence and SHA-256 context identities, but
never pixels, pose/appearance embeddings, or formation keypoints.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field, fields, replace
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import cv2

from .boundary_repair import BoundaryRepairConfig
from .face_identity import cosine_similarity, sface_model_asset_status
from .joint_timeline import (
    AuthorizedBoundaryOption,
    BOUNDARY_TRANSFORM_SUBJECT_BASES,
    BoundaryMeasurementKey,
    BoundaryOffsetAudit,
    BoundaryOffsetDisposition,
    BoundaryOptionSet,
    BoundaryTransformState,
    JOINT_BOUNDARY_FRAME_OFFSETS,
    SegmentBridgeDisposition,
    SegmentBridgeDispositionRequest,
    SegmentBridgeProof,
)
from .render import (
    EDITORIAL_TRANSFORM_MAXIMUM_CENTER_SHIFT,
    EDITORIAL_TRANSFORM_MAXIMUM_ZOOM,
    EditorialTransformKeyframe,
    EditorialTransformTrack,
)
from .source_scalar_cache import (
    PersistentSourceScalarCache,
    PersistentSourceScalarCacheStorageUnavailableError,
)
from .timeline import TimelineCandidate
from .timewarp import TimeWarpMap
from .visual_boundary_repair import (
    BoundaryDescriptorCache,
    BoundaryVisualEvidence,
    _BoundaryDescriptorCacheTransaction,
    _NEAREST_TEMPORAL_POSE_RADIUS_FRAMES,
    _PRODUCTION_REVIEW_GATES,
    _actual_pose_available,
    _passes_production_transition_gates,
    _stable_temporal_pose_choice,
    _visual_evidence,
)
from .visual_feature_cache import python_symbol_dependency_sha256
from .visual_score import FrameVisualDescriptor, _VideoSampler, describe_frame


JOINT_BOUNDARY_PROVIDER_ANALYSIS_BASIS = (
    "decoded_exact_seventeen_offset_joint_boundary_provider_v2+"
    "visual_boundary_repair_professional_gates_v4+"
    "scalar_only_exact_raw_result_context_proof_v2+"
    "renderer_exact_full_segment_sface_track_v1+"
    "immutable_transform_subject_basis_v1+selected_bridge_audit_handoff_v1"
)
_SIDE_SAMPLE_RADIUS_FRAMES = 3
_BOUNDARY_TRANSFORM_HOLD_FRAMES = _SIDE_SAMPLE_RADIUS_FRAMES + 0.5
_SCENE_CUT_GUARD_FRAMES = 9
_MINIMUM_SEGMENT_FRAMES = 2
_MINIMUM_DOMINANT_FACE_TRACK_SIMILARITY = 0.55
_EPSILON = 1e-9
_BEAT_PENALTY_WEIGHT = BoundaryRepairConfig().beat_penalty_weight
_BOUNDARY_MEMO_LIMIT = 256
_BRIDGE_MEMO_LIMIT = 2_048
_SEMANTIC_BRIDGE_PATH_LIMIT = 4_096
_FACE_IDENTITY_CACHE_LIMIT = 4_096
_TRANSFORMED_DESCRIPTOR_CACHE_LIMIT = 4_096
_BOUNDARY_MATERIALIZATION_LIMIT = 64
_SEGMENT_MATERIALIZATION_LIMIT = 32
_MEASUREMENT_AUDIT_LIMIT = 8
_TRANSIENT_AUDIT_LIMIT = 32
_SCALAR_BRIDGE_CACHE_NAMESPACE = "joint-bridge-disposition-v2"
_SCALAR_BRIDGE_CACHE_CONTRACT = "joint-bridge-disposition-scalar-shard-v2"
_SCALAR_BRIDGE_MAXIMUM_ENTRIES_PER_SHARD = 8 * (
    1 + (8 - 1) * len(JOINT_BOUNDARY_FRAME_OFFSETS)
) ** 2


DescriptorProvider = Callable[[int, float], FrameVisualDescriptor | None]
FaceIdentityProvider = Callable[
    [int, float, FrameVisualDescriptor], tuple[float, ...]
]
TransformedDescriptorProvider = Callable[
    [
        int,
        float,
        FrameVisualDescriptor,
        BoundaryTransformState,
    ],
    FrameVisualDescriptor | None,
]
_ScalarBridgeEntry = tuple[bool, bool, bool, tuple[str, ...], float]


class PersistentScalarBridgeCacheWriteError(RuntimeError):
    """A scalar shard write failed while its direct root remained live."""


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be an exact non-blank identifier")
    return value


def _source_provenance(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=False)
    try:
        state = resolved.stat()
    except OSError:
        return {
            "resolved_path": str(resolved),
            "stat_available": False,
        }
    return {
        "resolved_path": str(resolved),
        "stat_available": True,
        "device": int(state.st_dev),
        "inode": int(state.st_ino),
        "size": int(state.st_size),
        "mtime_ns": int(state.st_mtime_ns),
    }


def _source_runtime_stat_identity(
    path: Path,
) -> tuple[int, int, int, int, int] | None:
    try:
        state = Path(path).resolve(strict=False).stat()
    except OSError:
        return None
    return (
        int(state.st_dev),
        int(state.st_ino),
        int(state.st_size),
        int(state.st_mtime_ns),
        int(state.st_ctime_ns),
    )


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _candidate_private_state_sha256(candidate: TimelineCandidate) -> str:
    """Hash the complete in-memory state without exposing model vectors.

    The digest binds a raw proof to the exact optimizer candidate, including
    fields whose dataclass ``repr`` is intentionally hidden.  Only the digest
    participates in the context identity and the unhashed values never enter
    an audit payload.
    """

    digest = hashlib.sha256()
    digest.update(b"joint-boundary-candidate-private-state-v1\0")
    for item in fields(TimelineCandidate):
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(repr(getattr(candidate, item.name)).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _dataclass_private_state_sha256(value: object, *, contract: str) -> str:
    """Bind proof identity to transient dataclass state without exposing it."""

    digest = hashlib.sha256()
    digest.update(contract.encode("utf-8"))
    digest.update(b"\0")
    for item in fields(value):
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(repr(getattr(value, item.name)).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _descriptor_private_state_sha256(
    descriptor: FrameVisualDescriptor,
) -> str:
    return _dataclass_private_state_sha256(
        descriptor,
        contract="joint-boundary-frame-descriptor-private-state-v1",
    )


def _evidence_private_state_sha256(
    evidence: BoundaryVisualEvidence,
) -> str:
    return _dataclass_private_state_sha256(
        evidence,
        contract="joint-boundary-visual-evidence-private-state-v1",
    )


def _identity_private_state_sha256(identity: Sequence[float]) -> str:
    digest = hashlib.sha256()
    digest.update(b"joint-boundary-face-track-private-state-v1\0")
    digest.update(repr(tuple(float(value) for value in identity)).encode("utf-8"))
    return digest.hexdigest()


def _boundary_result_evidence_identity(
    *,
    raw_evidence_identity: str,
    outgoing_transform: BoundaryTransformState,
    incoming_transform: BoundaryTransformState,
    outgoing_descriptor: FrameVisualDescriptor,
    outgoing_preceding: FrameVisualDescriptor,
    incoming_descriptor: FrameVisualDescriptor,
    incoming_following: FrameVisualDescriptor,
    evidence: BoundaryVisualEvidence,
) -> str:
    return _sha256_json(
        {
            "contract": "joint-boundary-decoded-result-evidence-proof-v1",
            "raw_evidence_identity": raw_evidence_identity,
            "outgoing_transform": _transform_payload(outgoing_transform),
            "incoming_transform": _transform_payload(incoming_transform),
            "descriptor_private_state_sha256": {
                "outgoing_exact": _descriptor_private_state_sha256(
                    outgoing_descriptor
                ),
                "outgoing_preceding": _descriptor_private_state_sha256(
                    outgoing_preceding
                ),
                "incoming_exact": _descriptor_private_state_sha256(
                    incoming_descriptor
                ),
                "incoming_following": _descriptor_private_state_sha256(
                    incoming_following
                ),
            },
            "evidence_private_state_sha256": (
                _evidence_private_state_sha256(evidence)
            ),
        }
    )


def _candidate_public_identity(candidate: TimelineCandidate) -> dict[str, object]:
    return {
        "segment_index": candidate.segment_index,
        "source_id": candidate.source_id,
        "candidate_id": candidate.candidate_id,
        "shot_id": candidate.shot_id,
        "start_seconds": candidate.start_seconds,
        "end_seconds": candidate.end_seconds,
        "private_state_sha256": _candidate_private_state_sha256(candidate),
    }


def _transform_payload(state: BoundaryTransformState) -> dict[str, object]:
    return {
        "transform_identity": state.transform_identity,
        "crop_xywh": list(state.crop_xywh) if state.crop_xywh is not None else None,
        "rotation_degrees": state.rotation_degrees,
        "track_id": state.track_id,
        "subject_basis": state.subject_basis,
    }


def _safe_evidence_payload(evidence: BoundaryVisualEvidence) -> dict[str, object]:
    """Return only scalar/count evidence suitable for manifests."""

    return {
        "total_cost": evidence.total_cost,
        "base_total_cost": evidence.base_total_cost,
        "formation_mismatch": evidence.formation_mismatch,
        "pose_mismatch": evidence.pose_mismatch,
        "pose_basis": evidence.pose_basis,
        "silhouette_mismatch": evidence.silhouette_mismatch,
        "layout_mismatch": evidence.layout_mismatch,
        "persisted_layout_mismatch": evidence.persisted_layout_mismatch,
        "focus_mismatch": evidence.focus_mismatch,
        "structure_mismatch": evidence.structure_mismatch,
        "shot_scale_delta": evidence.shot_scale_delta,
        "shot_type_mismatch": evidence.shot_type_mismatch,
        "professional_evidence_basis": (
            evidence.evidence_coverage.professional_evidence_basis
        ),
        "professional_gate_passed": (
            evidence.evidence_coverage.professional_gate_passed
        ),
        "missing_mask": list(evidence.evidence_coverage.missing_mask),
        "outgoing_side_stability": (
            evidence.outgoing_side_stability.to_dict()
            if evidence.outgoing_side_stability is not None
            else None
        ),
        "incoming_side_stability": (
            evidence.incoming_side_stability.to_dict()
            if evidence.incoming_side_stability is not None
            else None
        ),
        "temporal_pose_motion": evidence.temporal_pose_motion.to_dict(),
    }


@dataclass(frozen=True, slots=True)
class _OffsetAuditRecord:
    frame_offset: int
    disposition: BoundaryOffsetDisposition
    evidence_identity: str
    failure_codes: tuple[str, ...]
    raw_context_identity: str
    result_context_identity: str | None
    safe_evidence: Mapping[str, object] | None
    outgoing_transform: BoundaryTransformState | None
    incoming_transform: BoundaryTransformState | None
    raw_evidence_identity: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_offset": self.frame_offset,
            "disposition": self.disposition.value,
            "evidence_identity": self.evidence_identity,
            "failure_codes": list(self.failure_codes),
            "raw_context_identity": self.raw_context_identity,
            "raw_evidence_identity": self.raw_evidence_identity,
            "result_context_identity": self.result_context_identity,
            "safe_scalar_evidence": (
                dict(self.safe_evidence)
                if self.safe_evidence is not None
                else None
            ),
            "outgoing_transform": (
                _transform_payload(self.outgoing_transform)
                if self.outgoing_transform is not None
                else None
            ),
            "incoming_transform": (
                _transform_payload(self.incoming_transform)
                if self.incoming_transform is not None
                else None
            ),
            "raw_frames_persisted": False,
            "biometric_model_vectors_persisted": False,
        }


@dataclass(frozen=True, slots=True)
class _MeasurementAuditRecord:
    measurement_identity: str
    boundary_index: int
    outgoing_source_id: str
    incoming_source_id: str
    offsets: tuple[_OffsetAuditRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "measurement_identity": self.measurement_identity,
            "boundary_index": self.boundary_index,
            "outgoing_source_id": self.outgoing_source_id,
            "incoming_source_id": self.incoming_source_id,
            "offset_count": len(self.offsets),
            "offset_audits": [item.to_dict() for item in self.offsets],
            "raw_frames_persisted": False,
            "biometric_model_vectors_persisted": False,
        }


@dataclass(frozen=True, slots=True)
class _BridgeAuditRecord:
    proof_identity: str
    candidate_id: str
    entry_proof_identity: str
    exit_proof_identity: str
    duration_frames: float
    expected_sample_count: int
    sampled_frame_count: int
    endpoints_included: bool
    full_frame_centers_included: bool
    evidence_complete: bool
    crop_compatible: bool
    track_compatible: bool
    failure_codes: tuple[str, ...]
    semantic_path_proof_identity: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "proof_identity": self.proof_identity,
            "candidate_id": self.candidate_id,
            "entry_proof_identity": self.entry_proof_identity,
            "exit_proof_identity": self.exit_proof_identity,
            "duration_frames": self.duration_frames,
            "expected_sample_count": self.expected_sample_count,
            "sampled_frame_count": self.sampled_frame_count,
            "endpoints_included": self.endpoints_included,
            "full_frame_centers_included": (
                self.full_frame_centers_included
            ),
            "evidence_complete": self.evidence_complete,
            "crop_compatible": self.crop_compatible,
            "track_compatible": self.track_compatible,
            "failure_codes": list(self.failure_codes),
            "semantic_path_proof_identity": (
                self.semantic_path_proof_identity
            ),
            "raw_frames_persisted": False,
            "biometric_model_vectors_persisted": False,
        }


@dataclass(frozen=True, slots=True)
class _TransformPathEvidence:
    proof_identity: str
    duration_frames: float
    expected_sample_count: int
    sampled_frame_count: int
    endpoints_included: bool
    full_frame_centers_included: bool
    evidence_complete: bool
    crop_compatible: bool
    track_compatible: bool
    failure_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SemanticBridgeMeasurement:
    evidence: _TransformPathEvidence


@dataclass(frozen=True, slots=True)
class BoundaryMaterializationProof:
    """Trusted process-local evidence needed to materialize one option.

    This type intentionally has no ``to_dict`` method.  Descriptors can carry
    transient model vectors and must never cross the process/audit boundary.
    """

    proof_identity: str
    raw_proof_identity: str
    result_proof_identity: str
    measurement_key: BoundaryMeasurementKey = field(repr=False)
    frame_offset: int
    boundary_master_seconds: float
    side_sample_radius_frames: int
    outgoing_source_index: int
    incoming_source_index: int
    outgoing_boundary_source_seconds: float
    incoming_boundary_source_seconds: float
    outgoing_exact_sample_seconds: float
    incoming_exact_sample_seconds: float
    outgoing_preceding_sample_seconds: float
    incoming_following_sample_seconds: float
    outgoing_transform: BoundaryTransformState
    incoming_transform: BoundaryTransformState
    raw_outgoing_descriptor: FrameVisualDescriptor = field(repr=False)
    raw_outgoing_preceding_descriptor: FrameVisualDescriptor = field(repr=False)
    raw_incoming_descriptor: FrameVisualDescriptor = field(repr=False)
    raw_incoming_following_descriptor: FrameVisualDescriptor = field(repr=False)
    result_outgoing_descriptor: FrameVisualDescriptor = field(repr=False)
    result_outgoing_preceding_descriptor: FrameVisualDescriptor = field(
        repr=False
    )
    result_incoming_descriptor: FrameVisualDescriptor = field(repr=False)
    result_incoming_following_descriptor: FrameVisualDescriptor = field(
        repr=False
    )
    raw_evidence: BoundaryVisualEvidence = field(repr=False)
    result_evidence: BoundaryVisualEvidence = field(repr=False)
    outgoing_face_identity_embedding: tuple[float, ...] = field(repr=False)
    incoming_face_identity_embedding: tuple[float, ...] = field(repr=False)
    scoring_outgoing: TimelineCandidate = field(repr=False)
    scoring_incoming: TimelineCandidate = field(repr=False)
    outer_track_proof_identities: tuple[str, ...] = ()

    @property
    def outgoing_track_id(self) -> str | None:
        return self.outgoing_transform.track_id

    @property
    def incoming_track_id(self) -> str | None:
        return self.incoming_transform.track_id


@dataclass(frozen=True, slots=True)
class SegmentTransformPathSample:
    """One exact in-RAM sample from a proven segment transform path."""

    master_seconds: float
    source_seconds: float
    transform: BoundaryTransformState
    subject_basis: str
    subject_bbox_xywh: tuple[float, float, float, float] | None
    subject_bbox_coverage: float | None
    face_similarity_to_anchor: float | None
    face_similarity_to_previous: float | None
    descriptor: FrameVisualDescriptor | None = field(repr=False)
    face_identity_embedding: tuple[float, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.transform, BoundaryTransformState):
            raise TypeError("segment sample transform is invalid")
        if (
            not isinstance(self.subject_basis, str)
            or self.subject_basis not in BOUNDARY_TRANSFORM_SUBJECT_BASES
        ):
            raise ValueError("segment sample subject_basis is unsupported")
        if self.subject_basis != self.transform.subject_basis:
            raise ValueError(
                "segment sample subject_basis disagrees with its transform"
            )


@dataclass(frozen=True, slots=True)
class SegmentTransformPathMaterializationProof:
    """Process-local bridge/outer proof; deliberately has no ``to_dict``."""

    proof_identity: str
    candidate: TimelineCandidate = field(repr=False)
    source_index: int
    entry_transform: BoundaryTransformState
    exit_transform: BoundaryTransformState
    editorial_transform_track: EditorialTransformTrack | None
    start_master_seconds: float
    end_master_seconds: float
    duration_frames: float
    expected_sample_count: int
    endpoints_included: bool
    full_frame_centers_included: bool
    evidence_complete: bool
    crop_compatible: bool
    track_compatible: bool
    failure_codes: tuple[str, ...]
    samples: tuple[SegmentTransformPathSample, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class JointSelectionMaterializationProof:
    """Exact authorized boundary + middle/outer path bundle for rendering."""

    selection_identity: str
    boundary_proof_identities: tuple[str, ...]
    segment_path_proof_identities: tuple[str, ...]
    boundaries: tuple[BoundaryMaterializationProof, ...] = field(repr=False)
    segment_paths: tuple[SegmentTransformPathMaterializationProof, ...] = field(
        repr=False
    )
    selected_bridge_audits: tuple[Mapping[str, object], ...] = field(
        default=(),
        repr=False,
    )

    def __post_init__(self) -> None:
        audits = tuple(
            MappingProxyType(dict(item)) for item in self.selected_bridge_audits
        )
        expected_middle_identities = tuple(
            path.proof_identity for path in self.segment_paths[1:-1]
        )
        if len(audits) != len(expected_middle_identities):
            raise ValueError(
                "selection proof needs one scalar audit per middle segment"
            )
        if tuple(item.get("proof_identity") for item in audits) != (
            expected_middle_identities
        ):
            raise ValueError(
                "selection bridge audits do not match middle segment proofs"
            )
        object.__setattr__(self, "selected_bridge_audits", audits)


def _normalized_bbox_after_crop(
    bbox: tuple[float, float, float, float] | None,
    crop: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    if crop is None:
        return bbox
    x, y, width, height = crop
    left = max(0.0, min(1.0, (bbox[0] - x) / width))
    top = max(0.0, min(1.0, (bbox[1] - y) / height))
    right = max(
        left + 1e-6,
        min(1.0, (bbox[0] + bbox[2] - x) / width),
    )
    bottom = max(
        top + 1e-6,
        min(1.0, (bbox[1] + bbox[3] - y) / height),
    )
    return (left, top, right - left, bottom - top)


def _normalized_centroid_after_crop(
    centroid: tuple[float, float] | None,
    crop: tuple[float, float, float, float] | None,
) -> tuple[float, float] | None:
    if centroid is None:
        return None
    if crop is None:
        return centroid
    x, y, width, height = crop
    return (
        max(0.0, min(1.0, (centroid[0] - x) / width)),
        max(0.0, min(1.0, (centroid[1] - y) / height)),
    )


def _centered_crop(
    bbox: tuple[float, float, float, float],
    *,
    target_subject_extent: float,
) -> tuple[float, float, float, float]:
    extent = max(float(bbox[2]), float(bbox[3]))
    minimum_scale = 1.0 / EDITORIAL_TRANSFORM_MAXIMUM_ZOOM
    scale = max(
        minimum_scale,
        min(1.0, extent / max(target_subject_extent, 1e-6)),
    )
    raw_dx = bbox[0] + bbox[2] / 2.0 - 0.5
    raw_dy = bbox[1] + bbox[3] / 2.0 - 0.5
    axis_limit = (1.0 - scale) / 2.0
    dx = max(-axis_limit, min(axis_limit, raw_dx))
    dy = max(-axis_limit, min(axis_limit, raw_dy))
    distance = math.hypot(dx, dy)
    if distance > EDITORIAL_TRANSFORM_MAXIMUM_CENTER_SHIFT:
        ratio = EDITORIAL_TRANSFORM_MAXIMUM_CENTER_SHIFT / distance
        dx *= ratio
        dy *= ratio
    center_x = 0.5 + dx
    center_y = 0.5 + dy
    x = center_x - scale / 2.0
    y = center_y - scale / 2.0
    return (x, y, scale, scale)


def _track_id(candidate: TimelineCandidate) -> str:
    local = candidate.candidate_id or candidate.shot_id or str(
        candidate.segment_index
    )
    return f"candidate-track:{candidate.source_id}:{local}"


def _identity_transform(
    *,
    raw_context_identity: str,
    side: str,
    candidate: TimelineCandidate,
) -> BoundaryTransformState:
    identity = _sha256_json(
        {
            "contract": "joint-boundary-identity-transform-v1",
            "raw_context_identity": raw_context_identity,
            "side": side,
            "candidate_state_sha256": _candidate_private_state_sha256(candidate),
        }
    )
    return BoundaryTransformState(
        transform_identity=f"identity-transform:{identity}",
        track_id=_track_id(candidate),
        subject_basis="full_frame_identity",
    )


def _crop_resolved_transforms(
    *,
    raw_context_identity: str,
    outgoing_candidate: TimelineCandidate,
    incoming_candidate: TimelineCandidate,
    outgoing_descriptor: FrameVisualDescriptor,
    incoming_descriptor: FrameVisualDescriptor,
) -> tuple[
    BoundaryTransformState,
    BoundaryTransformState,
    str,
] | None:
    outgoing_bbox = outgoing_descriptor.primary_face_bbox
    incoming_bbox = incoming_descriptor.primary_face_bbox
    basis = "dominant_face"
    if outgoing_bbox is None or incoming_bbox is None:
        outgoing_bbox = outgoing_descriptor.silhouette_bbox_xywh
        incoming_bbox = incoming_descriptor.silhouette_bbox_xywh
        basis = "subject_silhouette"
    if outgoing_bbox is None or incoming_bbox is None:
        return None
    # Place the matched subject at a common, deliberately visible scale.  A
    # fixed displayed extent (rather than the larger raw extent) gives both
    # sides enough pan/crop authority to align an off-centre performer.
    target_extent = 0.72
    outgoing_crop = _centered_crop(
        outgoing_bbox,
        target_subject_extent=target_extent,
    )
    incoming_crop = _centered_crop(
        incoming_bbox,
        target_subject_extent=target_extent,
    )

    def state(
        side: str,
        candidate: TimelineCandidate,
        crop: tuple[float, float, float, float],
    ) -> BoundaryTransformState:
        payload = {
            "contract": "joint-boundary-crop-transform-v1",
            "raw_context_identity": raw_context_identity,
            "side": side,
            "basis": basis,
            "crop_xywh": list(crop),
            "candidate_state_sha256": _candidate_private_state_sha256(candidate),
        }
        return BoundaryTransformState(
            transform_identity=f"crop-transform:{_sha256_json(payload)}",
            crop_xywh=crop,
            track_id=_track_id(candidate),
            subject_basis=basis,
        )

    return (
        state("outgoing", outgoing_candidate, outgoing_crop),
        state("incoming", incoming_candidate, incoming_crop),
        basis,
    )


def _updated_outgoing_candidate(
    candidate: TimelineCandidate,
    descriptor: FrameVisualDescriptor,
    evidence: BoundaryVisualEvidence,
    *,
    scene_cut_near: bool,
    face_identity: tuple[float, ...],
    transform: BoundaryTransformState,
    transformed_shot_scale: float | None,
    descriptor_already_transformed: bool = False,
) -> TimelineCandidate:
    crop = None if descriptor_already_transformed else transform.crop_xywh
    return replace(
        candidate,
        exit_shot_scale=(
            descriptor.shot_scale
            if transformed_shot_scale is None
            else transformed_shot_scale
        ),
        exit_shot_type=descriptor.shot_type,
        exit_scene_cut_near=scene_cut_near,
        exit_boundary_side_stability=evidence.outgoing_side_stability,
        exit_pose_embedding=(
            descriptor.pose_embedding if _actual_pose_available(descriptor) else ()
        ),
        exit_structure_embedding=descriptor.structure_embedding,
        exit_silhouette_embedding=descriptor.silhouette_embedding,
        exit_person_layout_embedding=descriptor.person_layout_embedding,
        exit_formation_people=descriptor.formation_people,
        exit_face_identity_embedding=face_identity,
        exit_color_embedding=descriptor.color_embedding,
        exit_face_geometry_window=(),
        exit_focus_bbox_xywh=_normalized_bbox_after_crop(
            descriptor.primary_face_bbox,
            crop,
        ),
        exit_subject_bbox_xywh=_normalized_bbox_after_crop(
            descriptor.silhouette_bbox_xywh,
            crop,
        ),
        exit_subject_centroid_xy=_normalized_centroid_after_crop(
            descriptor.silhouette_centroid_xy,
            crop,
        ),
        exit_detected_person_count=descriptor.detected_person_count,
        exit_motion=descriptor.motion,
    )


def _updated_incoming_candidate(
    candidate: TimelineCandidate,
    descriptor: FrameVisualDescriptor,
    evidence: BoundaryVisualEvidence,
    *,
    scene_cut_near: bool,
    face_identity: tuple[float, ...],
    transform: BoundaryTransformState,
    transformed_shot_scale: float | None,
    descriptor_already_transformed: bool = False,
) -> TimelineCandidate:
    crop = None if descriptor_already_transformed else transform.crop_xywh
    return replace(
        candidate,
        entry_shot_scale=(
            descriptor.shot_scale
            if transformed_shot_scale is None
            else transformed_shot_scale
        ),
        entry_shot_type=descriptor.shot_type,
        entry_scene_cut_near=scene_cut_near,
        entry_boundary_side_stability=evidence.incoming_side_stability,
        entry_pose_embedding=(
            descriptor.pose_embedding if _actual_pose_available(descriptor) else ()
        ),
        entry_structure_embedding=descriptor.structure_embedding,
        entry_silhouette_embedding=descriptor.silhouette_embedding,
        entry_person_layout_embedding=descriptor.person_layout_embedding,
        entry_formation_people=descriptor.formation_people,
        entry_face_identity_embedding=face_identity,
        entry_color_embedding=descriptor.color_embedding,
        entry_face_geometry_window=(),
        entry_focus_bbox_xywh=_normalized_bbox_after_crop(
            descriptor.primary_face_bbox,
            crop,
        ),
        entry_subject_bbox_xywh=_normalized_bbox_after_crop(
            descriptor.silhouette_bbox_xywh,
            crop,
        ),
        entry_subject_centroid_xy=_normalized_centroid_after_crop(
            descriptor.silhouette_centroid_xy,
            crop,
        ),
        entry_detected_person_count=descriptor.detected_person_count,
        entry_motion=descriptor.motion,
    )


class DecodedBoundaryOptionProvider:
    """Callable decoded option provider with exact measurement memoization."""

    def __init__(
        self,
        *,
        sources: Sequence[Path],
        audio_offsets: Sequence[float],
        scene_cut_times: Sequence[Sequence[float]],
        source_timewarps: Sequence[TimeWarpMap | None] | None,
        descriptor_cache: BoundaryDescriptorCache,
        output_fps: float,
        total_segment_count: int = 2,
        beat_period_seconds: float | None = None,
        beat_intensity: float | None = None,
        analysis_basis: str = JOINT_BOUNDARY_PROVIDER_ANALYSIS_BASIS,
        source_ids: Sequence[str] | None = None,
        feature_cache_directory: Path | None = None,
        feature_derived_cache_directory: Path | None = None,
        require_persistent_feature_cache: bool = False,
        descriptor_provider: DescriptorProvider | None = None,
        face_identity_provider: FaceIdentityProvider | None = None,
        transformed_descriptor_provider: (
            TransformedDescriptorProvider | None
        ) = None,
    ) -> None:
        resolved_sources = tuple(Path(item) for item in sources)
        if len(resolved_sources) < 2:
            raise ValueError("joint boundary provider needs at least two sources")
        if not (
            len(resolved_sources)
            == len(audio_offsets)
            == len(scene_cut_times)
        ):
            raise ValueError("per-source boundary inputs must have matching lengths")
        if not isinstance(descriptor_cache, BoundaryDescriptorCache):
            raise TypeError("descriptor_cache must be a BoundaryDescriptorCache")
        fps = _finite(output_fps, "output_fps")
        if fps <= 0.0:
            raise ValueError("output_fps must be positive")
        if (
            isinstance(total_segment_count, bool)
            or not isinstance(total_segment_count, int)
            or total_segment_count < 2
        ):
            raise ValueError("total_segment_count must be an integer of at least 2")
        period = (
            _finite(beat_period_seconds, "beat_period_seconds")
            if beat_period_seconds is not None
            else None
        )
        intensity = (
            _finite(beat_intensity, "beat_intensity")
            if beat_intensity is not None
            else None
        )
        if period is not None:
            if period <= 0.0:
                raise ValueError("beat_period_seconds must be positive")
            if intensity is None:
                intensity = 1.0
            elif not 0.0 <= intensity <= 1.0:
                raise ValueError("beat_intensity must be between zero and one")
        elif intensity is not None:
            raise ValueError("beat_intensity requires beat_period_seconds")
        basis = _exact_identifier(analysis_basis, "analysis_basis")
        offsets = tuple(_finite(value, "audio offset") for value in audio_offsets)
        clocks = (
            tuple(source_timewarps)
            if source_timewarps is not None
            else (None,) * len(resolved_sources)
        )
        if len(clocks) != len(resolved_sources):
            raise ValueError("source_timewarps must match sources")
        for index, timewarp in enumerate(clocks):
            if timewarp is None:
                continue
            if not isinstance(timewarp, TimeWarpMap):
                raise TypeError("source_timewarps entries must be TimeWarpMap or None")
            if not math.isclose(
                timewarp.global_offset_seconds,
                offsets[index],
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"source_timewarps[{index}] global offset does not match audio_offsets"
                )
        ids = (
            tuple(source_ids)
            if source_ids is not None
            else tuple(str(index) for index in range(len(resolved_sources)))
        )
        if len(ids) != len(resolved_sources):
            raise ValueError("source_ids must match sources")
        ids = tuple(_exact_identifier(value, "source id") for value in ids)
        if len(set(ids)) != len(ids):
            raise ValueError("source_ids must be unique")
        cuts: list[tuple[float, ...]] = []
        for per_source in scene_cut_times:
            normalized = tuple(
                sorted(set(_finite(value, "scene cut time") for value in per_source))
            )
            if any(value < 0.0 for value in normalized):
                raise ValueError("scene cut times cannot be negative")
            cuts.append(normalized)
        if descriptor_provider is not None and not callable(descriptor_provider):
            raise TypeError("descriptor_provider must be callable or None")
        if face_identity_provider is not None and not callable(
            face_identity_provider
        ):
            raise TypeError("face_identity_provider must be callable or None")
        if transformed_descriptor_provider is not None and not callable(
            transformed_descriptor_provider
        ):
            raise TypeError(
                "transformed_descriptor_provider must be callable or None"
            )
        if not isinstance(require_persistent_feature_cache, bool):
            raise TypeError("require_persistent_feature_cache must be a boolean")
        if (
            require_persistent_feature_cache
            and feature_derived_cache_directory is None
        ):
            raise ValueError(
                "require_persistent_feature_cache needs an explicit "
                "feature_derived_cache_directory"
            )
        if descriptor_provider is not None and (
            feature_cache_directory is not None
            or feature_derived_cache_directory is not None
            or require_persistent_feature_cache
        ):
            raise ValueError(
                "persistent feature cache cannot wrap an explicit "
                "descriptor_provider"
            )

        self.sources = resolved_sources
        self.audio_offsets = offsets
        self.scene_cut_times = tuple(cuts)
        self.source_timewarps = clocks
        self.descriptor_cache = descriptor_cache
        self.output_fps = fps
        self.total_segment_count = total_segment_count
        self.beat_period_seconds = period
        self.beat_intensity = intensity
        self.analysis_basis = basis
        self.source_ids = ids
        self.feature_cache_directory = (
            Path(feature_cache_directory)
            if feature_cache_directory is not None
            else None
        )
        self.feature_derived_cache_directory = (
            Path(feature_derived_cache_directory)
            if feature_derived_cache_directory is not None
            else None
        )
        self.require_persistent_feature_cache = (
            require_persistent_feature_cache
        )
        self._source_index_by_id = {
            source_id: index for index, source_id in enumerate(ids)
        }
        self._descriptor_provider = descriptor_provider
        self._face_identity_provider = face_identity_provider
        self._transformed_descriptor_provider = transformed_descriptor_provider
        descriptor_backend_identity = (
            f"injected-process-callable:{id(descriptor_provider)}"
            if descriptor_provider is not None
            else "production-video-sampler-descriptor-v1"
        )
        face_identity_backend_identity = (
            f"injected-process-callable:{id(face_identity_provider)}"
            if face_identity_provider is not None
            else "production-video-sampler-sface-v1"
        )
        transformed_descriptor_backend_identity = (
            f"injected-process-callable:{id(transformed_descriptor_provider)}"
            if transformed_descriptor_provider is not None
            else (
                "production-decoded-renderer-transform-descriptor-v1"
                if descriptor_provider is None
                else "unavailable-with-injected-raw-descriptor-provider"
            )
        )
        self._source_provenance = tuple(
            _source_provenance(source) for source in self.sources
        )
        self._source_runtime_stat_identities = tuple(
            _source_runtime_stat_identity(source) for source in self.sources
        )
        self._cache_source_indices = tuple(
            int(
                _sha256_json(
                    {
                        "contract": "joint-boundary-ram-cache-source-v1",
                        "source_provenance": self._source_provenance[index],
                        "analysis_basis": self.analysis_basis,
                        "descriptor_backend_identity": (
                            descriptor_backend_identity
                        ),
                    }
                ),
                16,
            )
            for index, source in enumerate(self.sources)
        )
        self._samplers: dict[int, _VideoSampler] = {}
        self._face_identity_cache: OrderedDict[
            tuple[int, object], tuple[float, ...]
        ] = OrderedDict()
        self._memo: OrderedDict[
            tuple[str, str, BoundaryMeasurementKey], BoundaryOptionSet
        ] = OrderedDict()
        self._audit_by_key: OrderedDict[
            tuple[str, str, BoundaryMeasurementKey], _MeasurementAuditRecord
        ] = OrderedDict()
        self._transformed_descriptor_cache: OrderedDict[
            tuple[
                int,
                float,
                tuple[float, float, float, float] | None,
                float,
            ],
            FrameVisualDescriptor | None,
        ] = OrderedDict()
        self._bridge_memo: OrderedDict[
            tuple[
                str,
                str,
                TimelineCandidate,
                AuthorizedBoundaryOption,
                AuthorizedBoundaryOption,
                float,
            ],
            SegmentBridgeProof,
        ] = OrderedDict()
        self._semantic_bridge_path_memo: OrderedDict[
            tuple[object, ...],
            _SemanticBridgeMeasurement,
        ] = OrderedDict()
        self._bridge_audit: OrderedDict[str, _BridgeAuditRecord] = OrderedDict()
        self._outer_track_audit: OrderedDict[str, _BridgeAuditRecord] = (
            OrderedDict()
        )
        self._materialization_by_proof: OrderedDict[
            str,
            BoundaryMaterializationProof,
        ] = OrderedDict()
        self._segment_materialization_by_proof: OrderedDict[
            str,
            SegmentTransformPathMaterializationProof,
        ] = OrderedDict()
        self._pinned_boundary_proofs: set[str] = set()
        self._pinned_segment_proofs: set[str] = set()
        self._cache_eviction_counts: dict[str, int] = {
            "boundary_memo": 0,
            "measurement_audit": 0,
            "bridge_memo": 0,
            "semantic_bridge_path": 0,
            "face_identity": 0,
            "transformed_descriptor": 0,
            "boundary_materialization": 0,
            "segment_materialization": 0,
            "bridge_audit": 0,
            "outer_track_audit": 0,
            "transform_subject_basis": 0,
        }
        self._cache_peak_counts = dict(self._cache_eviction_counts)
        self._measurement_audit_digest: str | None = None
        self._measurement_audit_total_count = 0
        self._measurement_offset_disposition_counts = {
            disposition.value: 0 for disposition in BoundaryOffsetDisposition
        }
        self._measurement_failure_code_counts: dict[str, int] = {}
        self._same_source_measurement_count = 0
        self._cross_source_measurement_count = 0
        self._bridge_audit_digest: str | None = None
        self._bridge_audit_registration_count = 0
        self._bridge_failure_code_counts: dict[str, int] = {}
        self._outer_audit_digest: str | None = None
        self._outer_audit_registration_count = 0
        self._outer_failure_code_counts: dict[str, int] = {}
        self._last_selection_identity: str | None = None
        self._last_selected_boundary_proof_identities: tuple[str, ...] = ()
        self._last_selected_segment_proof_identities: tuple[str, ...] = ()
        self._last_selected_boundary_records: tuple[
            Mapping[str, object], ...
        ] = ()
        self._last_selected_segment_records: tuple[
            Mapping[str, object], ...
        ] = ()
        self._active_transaction_journal: list[
            tuple[MutableMapping[Any, Any], Any, bool, Any]
        ] | None = None
        self._active_descriptor_cache_transaction: (
            _BoundaryDescriptorCacheTransaction | None
        ) = None
        self.request_count = 0
        self.measurement_count = 0
        self.measurement_failure_count = 0
        self.memo_hit_count = 0
        self.bridge_request_count = 0
        self.bridge_measurement_count = 0
        self.bridge_measurement_failure_count = 0
        self.bridge_memo_hit_count = 0
        self.bridge_alias_registration_count = 0
        self.bridge_semantic_memo_hit_count = 0
        self.scalar_bridge_batch_call_count = 0
        self.scalar_bridge_batch_request_count = 0
        self.scalar_bridge_disposition_count = 0
        self.scalar_bridge_descriptor_snapshot_hit_count = 0
        self.scalar_bridge_descriptor_snapshot_miss_count = 0
        self.scalar_bridge_face_identity_snapshot_hit_count = 0
        self.scalar_bridge_face_identity_snapshot_miss_count = 0
        self.scalar_bridge_persistent_cache_hit_count = 0
        self.scalar_bridge_persistent_cache_miss_count = 0
        self.scalar_bridge_persistent_cache_write_count = 0
        self.scalar_bridge_persistent_cache_invalid_entry_count = 0
        self._scalar_bridge_persistent_caches: dict[
            int, PersistentSourceScalarCache
        ] = {}
        self._scalar_bridge_persistent_entries_by_segment: dict[
            int, dict[str, _ScalarBridgeEntry]
        ] = {}
        self._scalar_bridge_persistent_cache_loaded_segments: set[int] = set()
        self._scalar_bridge_persistent_analyzer_base: dict[
            str, object
        ] | None = None
        self._scalar_bridge_candidate_sha256_cache: dict[
            int, tuple[TimelineCandidate, str]
        ] = {}
        self.selection_request_count = 0
        self.selection_materialization_count = 0
        self.selection_failure_count = 0
        self.descriptor_decode_count = 0
        self.raw_descriptor_cache_hit_count = 0
        self.raw_descriptor_cache_miss_count = 0
        self.transformed_descriptor_decode_count = 0
        self.transformed_descriptor_cache_hit_count = 0
        self.face_identity_model_call_count = 0
        self.face_identity_cache_hit_count = 0
        self.boundary_materialization_rehydration_count = 0
        self.bridge_materialization_rehydration_count = 0
        self._closed = False
        self._configuration_identity = _sha256_json(
            {
                "contract": "decoded-joint-boundary-provider-configuration-v1",
                "analysis_basis": self.analysis_basis,
                "source_provenance": list(self._source_provenance),
                "source_ids": list(self.source_ids),
                "audio_offsets": list(self.audio_offsets),
                "scene_cut_times": [list(item) for item in self.scene_cut_times],
                "source_timewarp_state": [
                    (
                        _sha256_json(timewarp.to_dict())
                        if timewarp is not None
                        else None
                    )
                    for timewarp in self.source_timewarps
                ],
                "output_fps": self.output_fps,
                "total_segment_count": self.total_segment_count,
                "beat_period_seconds": self.beat_period_seconds,
                "beat_intensity": self.beat_intensity,
                "beat_penalty_weight": _BEAT_PENALTY_WEIGHT,
                "descriptor_backend_identity": descriptor_backend_identity,
                "face_identity_backend_identity": (
                    face_identity_backend_identity
                ),
                "transformed_descriptor_backend_identity": (
                    transformed_descriptor_backend_identity
                ),
                "direct_persistent_feature_cache_configured": (
                    self.feature_derived_cache_directory is not None
                ),
                "persistent_feature_cache_required": (
                    self.require_persistent_feature_cache
                ),
                "offset_domain": list(JOINT_BOUNDARY_FRAME_OFFSETS),
                "side_sample_radius_frames": _SIDE_SAMPLE_RADIUS_FRAMES,
                "scene_cut_guard_frames": _SCENE_CUT_GUARD_FRAMES,
            }
        )
        if self.require_persistent_feature_cache:
            try:
                for source_index in range(len(self.sources)):
                    self._sampler(source_index)
            except BaseException:
                for sampler in self._samplers.values():
                    sampler.close()
                self._samplers.clear()
                raise

    def __enter__(self) -> "DecodedBoundaryOptionProvider":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        for sampler in self._samplers.values():
            sampler.close()
        self._samplers.clear()
        self.descriptor_cache.discard_source_indices(
            self._cache_source_indices
        )
        self._face_identity_cache.clear()
        self._transformed_descriptor_cache.clear()
        self._memo.clear()
        self._audit_by_key.clear()
        self._bridge_memo.clear()
        self._semantic_bridge_path_memo.clear()
        self._bridge_audit.clear()
        self._outer_track_audit.clear()
        self._materialization_by_proof.clear()
        self._segment_materialization_by_proof.clear()
        self._pinned_boundary_proofs.clear()
        self._pinned_segment_proofs.clear()
        for name in self._cache_eviction_counts:
            self._cache_eviction_counts[name] = 0
            self._cache_peak_counts[name] = 0
        self._measurement_audit_digest = None
        self._measurement_audit_total_count = 0
        for name in self._measurement_offset_disposition_counts:
            self._measurement_offset_disposition_counts[name] = 0
        self._measurement_failure_code_counts.clear()
        self._same_source_measurement_count = 0
        self._cross_source_measurement_count = 0
        self._bridge_audit_digest = None
        self._bridge_audit_registration_count = 0
        self._bridge_failure_code_counts.clear()
        self._outer_audit_digest = None
        self._outer_audit_registration_count = 0
        self._outer_failure_code_counts.clear()
        self._last_selection_identity = None
        self._last_selected_boundary_proof_identities = ()
        self._last_selected_segment_proof_identities = ()
        self._last_selected_boundary_records = ()
        self._last_selected_segment_records = ()
        self._active_transaction_journal = None
        self._active_descriptor_cache_transaction = None
        self._descriptor_provider = None
        self._face_identity_provider = None
        self._transformed_descriptor_provider = None
        self.request_count = 0
        self.measurement_count = 0
        self.measurement_failure_count = 0
        self.memo_hit_count = 0
        self.bridge_request_count = 0
        self.bridge_measurement_count = 0
        self.bridge_measurement_failure_count = 0
        self.bridge_memo_hit_count = 0
        self.bridge_alias_registration_count = 0
        self.bridge_semantic_memo_hit_count = 0
        self.scalar_bridge_batch_call_count = 0
        self.scalar_bridge_batch_request_count = 0
        self.scalar_bridge_disposition_count = 0
        self.scalar_bridge_descriptor_snapshot_hit_count = 0
        self.scalar_bridge_descriptor_snapshot_miss_count = 0
        self.scalar_bridge_face_identity_snapshot_hit_count = 0
        self.scalar_bridge_face_identity_snapshot_miss_count = 0
        self.scalar_bridge_persistent_cache_hit_count = 0
        self.scalar_bridge_persistent_cache_miss_count = 0
        self.scalar_bridge_persistent_cache_write_count = 0
        self.scalar_bridge_persistent_cache_invalid_entry_count = 0
        self._scalar_bridge_persistent_caches.clear()
        self._scalar_bridge_persistent_entries_by_segment.clear()
        self._scalar_bridge_persistent_cache_loaded_segments.clear()
        self._scalar_bridge_persistent_analyzer_base = None
        self._scalar_bridge_candidate_sha256_cache.clear()
        self.selection_request_count = 0
        self.selection_materialization_count = 0
        self.selection_failure_count = 0
        self.descriptor_decode_count = 0
        self.raw_descriptor_cache_hit_count = 0
        self.raw_descriptor_cache_miss_count = 0
        self.transformed_descriptor_decode_count = 0
        self.transformed_descriptor_cache_hit_count = 0
        self.face_identity_model_call_count = 0
        self.face_identity_cache_hit_count = 0
        self.boundary_materialization_rehydration_count = 0
        self.bridge_materialization_rehydration_count = 0
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("decoded boundary option provider is closed")

    def _assert_required_persistent_cache_available(
        self,
        *source_indices: int,
    ) -> None:
        if not self.require_persistent_feature_cache:
            return
        for source_index in dict.fromkeys(source_indices):
            self._sampler(source_index).assert_required_persistent_cache_available()

    def validate_segment_count(self, segment_count: int) -> None:
        """Fail before selection when the optimizer shape is not this basis."""

        self._require_open()
        if isinstance(segment_count, bool) or not isinstance(segment_count, int):
            raise TypeError("segment_count must be an integer")
        if segment_count != self.total_segment_count:
            raise ValueError(
                "selector segment count does not match provider analysis basis"
            )

    def capacity_audit(
        self,
        maximum_candidates_per_segment: int = 8,
    ) -> dict[str, int | bool | str]:
        """Return the exact full-domain edge upper bound without decoding."""

        if (
            isinstance(maximum_candidates_per_segment, bool)
            or not isinstance(maximum_candidates_per_segment, int)
            or not 1 <= maximum_candidates_per_segment <= 8
        ):
            raise ValueError(
                "maximum_candidates_per_segment must be an integer from 1 to 8"
            )
        boundary_count = self.total_segment_count - 1
        measurement_key_upper_bound = (
            boundary_count * maximum_candidates_per_segment**2
        )
        offset_audit_upper_bound = (
            measurement_key_upper_bound * len(JOINT_BOUNDARY_FRAME_OFFSETS)
        )
        offsets_per_edge = len(JOINT_BOUNDARY_FRAME_OFFSETS)
        distinct_source_options_per_middle_candidate_side = (
            1
            + (maximum_candidates_per_segment - 1) * offsets_per_edge
        )
        middle_segment_count = max(0, self.total_segment_count - 2)
        distinct_source_bridge_request_upper_bound = (
            middle_segment_count
            * maximum_candidates_per_segment
            * distinct_source_options_per_middle_candidate_side**2
        )
        all_cross_bridge_request_upper_bound = (
            middle_segment_count
            * maximum_candidates_per_segment**3
            * offsets_per_edge**2
        )
        exact_option_state_upper_bound_per_layer = (
            maximum_candidates_per_segment
            + (
                maximum_candidates_per_segment**2
                - maximum_candidates_per_segment
            )
            * offsets_per_edge
        )
        identity_semantic_bridge_measurement_upper_bound = (
            middle_segment_count
            * maximum_candidates_per_segment
            * offsets_per_edge**2
        )
        return {
            "contract": "decoded-joint-boundary-capacity-audit-v1",
            "total_segment_count": self.total_segment_count,
            "boundary_count": boundary_count,
            "maximum_candidates_per_segment": (
                maximum_candidates_per_segment
            ),
            "measurement_key_upper_bound": measurement_key_upper_bound,
            "offsets_per_measurement": offsets_per_edge,
            "offset_audit_upper_bound": offset_audit_upper_bound,
            "middle_segment_count": middle_segment_count,
            "distinct_source_options_per_middle_candidate_side": (
                distinct_source_options_per_middle_candidate_side
            ),
            "distinct_source_bridge_request_upper_bound": (
                distinct_source_bridge_request_upper_bound
            ),
            "all_cross_bridge_request_upper_bound": (
                all_cross_bridge_request_upper_bound
            ),
            "exact_option_state_upper_bound_per_layer": (
                exact_option_state_upper_bound_per_layer
            ),
            "top_eight_node_upper_bound_per_exact_state_layer": (
                exact_option_state_upper_bound_per_layer * 8
            ),
            "identity_semantic_bridge_measurement_upper_bound": (
                identity_semantic_bridge_measurement_upper_bound
            ),
            "full_offset_domain_required": True,
            "offset_subsampling_authorized": False,
            "raw_descriptor_cache_shared_across_measurement_keys": True,
            "bridge_path_evidence_shared_by_renderer_semantics": True,
            "exact_bridge_alias_required_per_selected_proof": True,
        }

    def _transactional_set(
        self,
        mapping: MutableMapping[Any, Any],
        key: Any,
        value: Any,
    ) -> None:
        journal = self._active_transaction_journal
        if journal is not None:
            journal.append(
                (mapping, key, key in mapping, mapping.get(key))
            )
        mapping[key] = value

    def _transactional_delete(
        self,
        mapping: MutableMapping[Any, Any],
        key: Any,
    ) -> None:
        if key not in mapping:
            return
        journal = self._active_transaction_journal
        if journal is not None:
            journal.append((mapping, key, True, mapping[key]))
        del mapping[key]

    def _bounded_transactional_set(
        self,
        mapping: OrderedDict[Any, Any],
        key: Any,
        value: Any,
        *,
        maximum_entries: int,
        cache_name: str,
        pinned_keys: set[Any] | None = None,
    ) -> None:
        self._transactional_set(mapping, key, value)
        # A transactional insert of a new key is already the MRU item.  Do not
        # reorder replacements or evict until commit: both operations would
        # otherwise make callback failure change future LRU behaviour even
        # after the value journal is rolled back.
        if self._active_transaction_journal is None:
            mapping.move_to_end(key)
        self._cache_peak_counts[cache_name] = max(
            self._cache_peak_counts[cache_name],
            len(mapping),
        )
        if self._active_transaction_journal is not None:
            return
        pinned = pinned_keys or set()
        while len(mapping) > maximum_entries:
            evicted = next(
                (candidate for candidate in mapping if candidate not in pinned),
                None,
            )
            if evicted is None:
                break
            self._transactional_delete(mapping, evicted)
            self._cache_eviction_counts[cache_name] += 1

    def _enforce_lru_limit(
        self,
        mapping: OrderedDict[Any, Any],
        *,
        maximum_entries: int,
        cache_name: str,
        pinned_keys: set[Any] | None = None,
    ) -> None:
        pinned = pinned_keys or set()
        while len(mapping) > maximum_entries:
            evicted = next(
                (candidate for candidate in mapping if candidate not in pinned),
                None,
            )
            if evicted is None:
                break
            self._transactional_delete(mapping, evicted)
            self._cache_eviction_counts[cache_name] += 1

    def _enforce_transient_cache_limits(self) -> None:
        self._enforce_lru_limit(
            self._memo,
            maximum_entries=_BOUNDARY_MEMO_LIMIT,
            cache_name="boundary_memo",
        )
        self._enforce_lru_limit(
            self._audit_by_key,
            maximum_entries=_MEASUREMENT_AUDIT_LIMIT,
            cache_name="measurement_audit",
        )
        self._enforce_lru_limit(
            self._bridge_memo,
            maximum_entries=_BRIDGE_MEMO_LIMIT,
            cache_name="bridge_memo",
        )
        self._enforce_lru_limit(
            self._semantic_bridge_path_memo,
            maximum_entries=_SEMANTIC_BRIDGE_PATH_LIMIT,
            cache_name="semantic_bridge_path",
        )
        self._enforce_lru_limit(
            self._face_identity_cache,
            maximum_entries=_FACE_IDENTITY_CACHE_LIMIT,
            cache_name="face_identity",
        )
        self._enforce_lru_limit(
            self._transformed_descriptor_cache,
            maximum_entries=_TRANSFORMED_DESCRIPTOR_CACHE_LIMIT,
            cache_name="transformed_descriptor",
        )
        self._enforce_lru_limit(
            self._materialization_by_proof,
            maximum_entries=_BOUNDARY_MATERIALIZATION_LIMIT,
            cache_name="boundary_materialization",
            pinned_keys=self._pinned_boundary_proofs,
        )
        self._enforce_lru_limit(
            self._segment_materialization_by_proof,
            maximum_entries=_SEGMENT_MATERIALIZATION_LIMIT,
            cache_name="segment_materialization",
            pinned_keys=self._pinned_segment_proofs,
        )
        self._enforce_lru_limit(
            self._bridge_audit,
            maximum_entries=_TRANSIENT_AUDIT_LIMIT,
            cache_name="bridge_audit",
        )
        self._enforce_lru_limit(
            self._outer_track_audit,
            maximum_entries=_TRANSIENT_AUDIT_LIMIT,
            cache_name="outer_track_audit",
        )

    def _touch_lru(self, mapping: OrderedDict[Any, Any], key: Any) -> None:
        if self._active_transaction_journal is None and key in mapping:
            mapping.move_to_end(key)

    @staticmethod
    def _rollback_transaction(
        journal: Sequence[
            tuple[MutableMapping[Any, Any], Any, bool, Any]
        ],
    ) -> None:
        for mapping, key, existed, previous in reversed(journal):
            if existed:
                mapping[key] = previous
            else:
                mapping.pop(key, None)

    def _source_index(self, source_id: str) -> int:
        try:
            return self._source_index_by_id[source_id]
        except KeyError as exc:
            raise ValueError(
                f"timeline candidate source_id is not configured: {source_id}"
            ) from exc

    def _source_time(self, source_index: int, master_seconds: float) -> float:
        timewarp = self.source_timewarps[source_index]
        if timewarp is None:
            return master_seconds + self.audio_offsets[source_index]
        return timewarp.master_to_source(master_seconds)

    def _sampler(self, source_index: int) -> _VideoSampler:
        sampler = self._samplers.get(source_index)
        if sampler is None:
            sampler = _VideoSampler(
                self.sources[source_index],
                feature_cache_directory=self.feature_cache_directory,
                feature_derived_cache_directory=(
                    self.feature_derived_cache_directory
                ),
                require_persistent_feature_cache=(
                    self.require_persistent_feature_cache
                ),
            )
            self._samplers[source_index] = sampler
        return sampler

    def _sample_descriptor(
        self,
        source_index: int,
        seconds: float,
    ) -> FrameVisualDescriptor | None:
        if seconds < 0.0:
            return None
        self._assert_required_persistent_cache_available(source_index)
        cache_source_index = self._cache_source_indices[source_index]
        transaction = self._active_descriptor_cache_transaction
        hit, cached = self.descriptor_cache.lookup(
            cache_source_index,
            seconds,
            transaction=transaction,
        )
        if hit:
            self.raw_descriptor_cache_hit_count += 1
            return cached
        self.raw_descriptor_cache_miss_count += 1
        self.descriptor_decode_count += 1
        if self._descriptor_provider is not None:
            descriptor = self._descriptor_provider(source_index, seconds)
        else:
            descriptor = self._sampler(source_index).feature_descriptor_at(
                seconds,
                include_subject_geometry=True,
                include_formation_pose=True,
            )
        if descriptor is not None and not isinstance(
            descriptor, FrameVisualDescriptor
        ):
            raise TypeError("descriptor provider returned an invalid descriptor")
        self.descriptor_cache.store(
            cache_source_index,
            seconds,
            descriptor,
            transaction=transaction,
        )
        return descriptor

    def _sample_transformed_descriptor(
        self,
        source_index: int,
        seconds: float,
        raw_descriptor: FrameVisualDescriptor,
        transform: BoundaryTransformState,
    ) -> FrameVisualDescriptor | None:
        key = (
            self._cache_source_indices[source_index],
            round(float(seconds), 7),
            transform.crop_xywh,
            round(float(transform.rotation_degrees), 7),
        )
        if key in self._transformed_descriptor_cache:
            self.transformed_descriptor_cache_hit_count += 1
            self._touch_lru(self._transformed_descriptor_cache, key)
            return self._transformed_descriptor_cache[key]
        self.transformed_descriptor_decode_count += 1
        if self._transformed_descriptor_provider is not None:
            descriptor = self._transformed_descriptor_provider(
                source_index,
                seconds,
                raw_descriptor,
                transform,
            )
        elif self._descriptor_provider is not None:
            descriptor = None
        else:
            frame = self._sampler(source_index).frame_at(seconds)
            descriptor = None
            if frame is not None:
                height, width = frame.shape[:2]
                crop = transform.crop_xywh or (0.0, 0.0, 1.0, 1.0)
                x, y, crop_width, crop_height = crop
                if (
                    0.0 <= x < 1.0
                    and 0.0 <= y < 1.0
                    and crop_width > 0.0
                    and crop_height > 0.0
                    and x + crop_width <= 1.0 + _EPSILON
                    and y + crop_height <= 1.0 + _EPSILON
                ):
                    left = max(0, min(width - 1, int(math.floor(x * width))))
                    top = max(0, min(height - 1, int(math.floor(y * height))))
                    right = max(
                        left + 1,
                        min(width, int(math.ceil((x + crop_width) * width))),
                    )
                    bottom = max(
                        top + 1,
                        min(height, int(math.ceil((y + crop_height) * height))),
                    )
                    working = frame[top:bottom, left:right]
                    try:
                        working = cv2.resize(
                            working,
                            (width, height),
                            interpolation=cv2.INTER_LANCZOS4,
                        )
                        if abs(transform.rotation_degrees) > _EPSILON:
                            matrix = cv2.getRotationMatrix2D(
                                (width / 2.0, height / 2.0),
                                transform.rotation_degrees,
                                1.0,
                            )
                            working = cv2.warpAffine(
                                working,
                                matrix,
                                (width, height),
                                flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_REFLECT_101,
                            )
                        descriptor = describe_frame(
                            working,
                            include_subject_geometry=True,
                            include_formation_pose=True,
                        )
                    except cv2.error:
                        descriptor = None
        if descriptor is not None and not isinstance(
            descriptor,
            FrameVisualDescriptor,
        ):
            raise TypeError(
                "transformed descriptor provider returned an invalid descriptor"
            )
        self._bounded_transactional_set(
            self._transformed_descriptor_cache,
            key,
            descriptor,
            maximum_entries=_TRANSFORMED_DESCRIPTOR_CACHE_LIMIT,
            cache_name="transformed_descriptor",
        )
        return descriptor

    def _sample_face_identity(
        self,
        source_index: int,
        seconds: float,
        descriptor: FrameVisualDescriptor,
    ) -> tuple[float, ...]:
        if descriptor.primary_face_bbox is None:
            return ()
        production_backend = bool(
            self._face_identity_provider is None
            and self._descriptor_provider is None
        )
        if (
            production_backend
            and (
                descriptor.face_detector_backend != "yunet"
                or descriptor.primary_face_landmarks_xy is None
                or len(descriptor.primary_face_landmarks_xy) != 5
            )
        ):
            # The production SFace sampler would return empty evidence for
            # this descriptor after decoding the frame.  Preserve that exact
            # outcome without paying for a decode that cannot reach the model.
            return ()
        production_sampler = (
            self._sampler(source_index) if production_backend else None
        )
        key = (
            source_index,
            (
                production_sampler.frame_identity(seconds)
                if production_sampler is not None
                else round(float(seconds), 7)
            ),
        )
        cached = self._face_identity_cache.get(key)
        if cached is not None:
            self.face_identity_cache_hit_count += 1
            self._touch_lru(self._face_identity_cache, key)
            return cached
        self.face_identity_model_call_count += 1
        if self._face_identity_provider is not None:
            identity = tuple(
                float(value)
                for value in self._face_identity_provider(
                    source_index,
                    seconds,
                    descriptor,
                )
            )
        elif self._descriptor_provider is None:
            assert production_sampler is not None
            identity = tuple(
                production_sampler.sface_embedding_at(
                    seconds,
                    descriptor,
                )
            )
        else:
            identity = ()
        if any(not math.isfinite(value) for value in identity):
            raise ValueError("face identity provider returned non-finite evidence")
        self._bounded_transactional_set(
            self._face_identity_cache,
            key,
            identity,
            maximum_entries=_FACE_IDENTITY_CACHE_LIMIT,
            cache_name="face_identity",
        )
        return identity

    def _memo_key(
        self,
        key: BoundaryMeasurementKey,
    ) -> tuple[str, str, BoundaryMeasurementKey]:
        return (self.analysis_basis, self._configuration_identity, key)

    def __call__(self, key: BoundaryMeasurementKey) -> BoundaryOptionSet:
        self._require_open()
        if not isinstance(key, BoundaryMeasurementKey):
            raise TypeError("joint boundary provider key is invalid")
        self._assert_required_persistent_cache_available(
            self._source_index(key.outgoing.source_id),
            self._source_index(key.incoming.source_id),
        )
        self.request_count += 1
        memo_key = self._memo_key(key)
        cached = self._memo.get(memo_key)
        if cached is not None:
            self.memo_hit_count += 1
            self._touch_lru(self._memo, memo_key)
            return cached
        if self._active_transaction_journal is not None:
            raise RuntimeError("decoded boundary measurement cannot be re-entered")
        journal: list[
            tuple[MutableMapping[Any, Any], Any, bool, Any]
        ] = []
        counter_snapshot = (
            self.descriptor_decode_count,
            self.transformed_descriptor_decode_count,
            self.face_identity_model_call_count,
            self.raw_descriptor_cache_hit_count,
            self.raw_descriptor_cache_miss_count,
            self.transformed_descriptor_cache_hit_count,
            self.face_identity_cache_hit_count,
            dict(self._cache_eviction_counts),
            dict(self._cache_peak_counts),
            self._outer_audit_registration_count,
            self._outer_audit_digest,
            dict(self._outer_failure_code_counts),
        )
        self._active_transaction_journal = journal
        descriptor_cache_transaction = self.descriptor_cache.begin_transaction()
        self._active_descriptor_cache_transaction = descriptor_cache_transaction
        try:
            option_set, audit = self._measure(key)
            self.descriptor_cache.commit_transaction(
                descriptor_cache_transaction
            )
        except BaseException:
            self._rollback_transaction(journal)
            if descriptor_cache_transaction.active:
                self.descriptor_cache.rollback_transaction(
                    descriptor_cache_transaction
                )
            (
                self.descriptor_decode_count,
                self.transformed_descriptor_decode_count,
                self.face_identity_model_call_count,
                self.raw_descriptor_cache_hit_count,
                self.raw_descriptor_cache_miss_count,
                self.transformed_descriptor_cache_hit_count,
                self.face_identity_cache_hit_count,
                eviction_counts,
                peak_counts,
                self._outer_audit_registration_count,
                self._outer_audit_digest,
                outer_failure_code_counts,
            ) = counter_snapshot
            self._cache_eviction_counts.update(eviction_counts)
            self._cache_peak_counts.update(peak_counts)
            self._outer_failure_code_counts.clear()
            self._outer_failure_code_counts.update(
                outer_failure_code_counts
            )
            self._active_transaction_journal = None
            self._active_descriptor_cache_transaction = None
            self.measurement_failure_count += 1
            raise
        self._active_transaction_journal = None
        self._active_descriptor_cache_transaction = None
        self._bounded_transactional_set(
            self._memo,
            memo_key,
            option_set,
            maximum_entries=_BOUNDARY_MEMO_LIMIT,
            cache_name="boundary_memo",
        )
        self._commit_measurement_audit(memo_key, audit)
        self.measurement_count += 1
        self._enforce_transient_cache_limits()
        return option_set

    def _raw_context(
        self,
        key: BoundaryMeasurementKey,
    ) -> tuple[dict[str, object], str]:
        if key.boundary_index >= self.total_segment_count - 1:
            raise ValueError(
                "boundary index lies outside total_segment_count"
            )
        nominal = key.outgoing.end_seconds
        if not math.isclose(
            nominal,
            key.incoming.start_seconds,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "joint boundary candidates do not share one canonical boundary"
            )
        payload = {
            "contract": "joint-boundary-raw-context-v1",
            "analysis_basis": self.analysis_basis,
            "configuration_identity": self._configuration_identity,
            "boundary_index": key.boundary_index,
            "nominal_boundary_seconds": nominal,
            "outgoing": _candidate_public_identity(key.outgoing),
            "incoming": _candidate_public_identity(key.incoming),
        }
        return payload, _sha256_json(payload)

    def _hard_audit(
        self,
        *,
        key: BoundaryMeasurementKey,
        frame_offset: int,
        raw_context_identity: str,
        failure_codes: Sequence[str],
        safe_evidence: Mapping[str, object] | None = None,
        raw_evidence_identity: str | None = None,
        result_evidence_identity: str | None = None,
    ) -> tuple[BoundaryOffsetAudit, _OffsetAuditRecord]:
        codes = tuple(sorted(set(failure_codes)))
        identity = "joint-boundary-hard:" + _sha256_json(
            {
                "raw_context_identity": raw_context_identity,
                "frame_offset": frame_offset,
                "failure_codes": list(codes),
                "safe_evidence": safe_evidence,
                "raw_evidence_identity": raw_evidence_identity,
                "result_evidence_identity": result_evidence_identity,
            }
        )
        audit = BoundaryOffsetAudit(
            frame_offset=frame_offset,
            disposition=BoundaryOffsetDisposition.HARD_INCOMPATIBLE,
            evidence_identity=identity,
            failure_codes=codes,
        )
        record = _OffsetAuditRecord(
            frame_offset=frame_offset,
            disposition=BoundaryOffsetDisposition.HARD_INCOMPATIBLE,
            evidence_identity=identity,
            failure_codes=codes,
            raw_context_identity=raw_context_identity,
            result_context_identity=result_evidence_identity,
            safe_evidence=safe_evidence,
            outgoing_transform=None,
            incoming_transform=None,
            raw_evidence_identity=raw_evidence_identity,
        )
        return audit, record

    def _incomplete_audit(
        self,
        *,
        frame_offset: int,
        raw_context_identity: str,
        failure_codes: Sequence[str],
        safe_evidence: Mapping[str, object] | None = None,
        raw_evidence_identity: str | None = None,
        result_evidence_identity: str | None = None,
    ) -> tuple[BoundaryOffsetAudit, _OffsetAuditRecord]:
        codes = tuple(sorted(set(failure_codes)))
        identity = "joint-boundary-incomplete:" + _sha256_json(
            {
                "raw_context_identity": raw_context_identity,
                "frame_offset": frame_offset,
                "failure_codes": list(codes),
                "safe_evidence": safe_evidence,
                "raw_evidence_identity": raw_evidence_identity,
                "result_evidence_identity": result_evidence_identity,
            }
        )
        audit = BoundaryOffsetAudit(
            frame_offset=frame_offset,
            disposition=BoundaryOffsetDisposition.EVIDENCE_INCOMPLETE,
            evidence_identity=identity,
            failure_codes=codes,
        )
        record = _OffsetAuditRecord(
            frame_offset=frame_offset,
            disposition=BoundaryOffsetDisposition.EVIDENCE_INCOMPLETE,
            evidence_identity=identity,
            failure_codes=codes,
            raw_context_identity=raw_context_identity,
            result_context_identity=result_evidence_identity,
            safe_evidence=safe_evidence,
            outgoing_transform=None,
            incoming_transform=None,
            raw_evidence_identity=raw_evidence_identity,
        )
        return audit, record

    @staticmethod
    def _canonical_measurement_audit(
        record: _MeasurementAuditRecord,
    ) -> _MeasurementAuditRecord:
        for offset in record.offsets:
            if offset.disposition is BoundaryOffsetDisposition.AUTHORIZED and (
                offset.raw_evidence_identity is None
                or offset.result_context_identity is None
                or offset.outgoing_transform is None
                or offset.incoming_transform is None
            ):
                raise RuntimeError(
                    "authorized boundary audit is missing canonical proof state"
                )
            if (
                any(
                    code.startswith("outer_track_")
                    for code in offset.failure_codes
                )
                and offset.raw_evidence_identity is None
            ):
                raise RuntimeError(
                    "outer-track failure audit is missing raw evidence identity"
                )
        return record

    def _commit_measurement_audit(
        self,
        memo_key: tuple[str, str, BoundaryMeasurementKey],
        record: _MeasurementAuditRecord,
    ) -> None:
        self._measurement_audit_total_count += 1
        if record.outgoing_source_id == record.incoming_source_id:
            self._same_source_measurement_count += 1
        else:
            self._cross_source_measurement_count += 1
        for offset in record.offsets:
            disposition = offset.disposition.value
            self._measurement_offset_disposition_counts[disposition] += 1
            for code in offset.failure_codes:
                self._measurement_failure_code_counts[code] = (
                    self._measurement_failure_code_counts.get(code, 0) + 1
                )
        self._measurement_audit_digest = _sha256_json(
            {
                "contract": "joint-boundary-measurement-audit-chain-v1",
                "previous": self._measurement_audit_digest,
                "measurement_identity": record.measurement_identity,
                "offset_evidence_identities": [
                    offset.evidence_identity for offset in record.offsets
                ],
            }
        )
        self._bounded_transactional_set(
            self._audit_by_key,
            memo_key,
            record,
            maximum_entries=_MEASUREMENT_AUDIT_LIMIT,
            cache_name="measurement_audit",
        )

    def _register_outer_track_audit(
        self,
        record: _BridgeAuditRecord,
    ) -> None:
        self._bounded_transactional_set(
            self._outer_track_audit,
            record.proof_identity,
            record,
            maximum_entries=_TRANSIENT_AUDIT_LIMIT,
            cache_name="outer_track_audit",
        )
        self._outer_audit_registration_count += 1
        for code in record.failure_codes:
            self._outer_failure_code_counts[code] = (
                self._outer_failure_code_counts.get(code, 0) + 1
            )
        self._outer_audit_digest = _sha256_json(
            {
                "contract": "joint-outer-track-audit-chain-v1",
                "previous": self._outer_audit_digest,
                "proof_identity": record.proof_identity,
                "failure_codes": list(record.failure_codes),
            }
        )

    def _scene_cut_failures(
        self,
        *,
        outgoing_source: int,
        incoming_source: int,
        outgoing_candidate: TimelineCandidate,
        incoming_candidate: TimelineCandidate,
        boundary_master_seconds: float,
        outgoing_boundary_local: float,
        incoming_boundary_local: float,
    ) -> tuple[str, ...]:
        guard_master_seconds = _SCENE_CUT_GUARD_FRAMES / self.output_fps
        outgoing_guard_seconds = outgoing_boundary_local - self._source_time(
            outgoing_source,
            boundary_master_seconds - guard_master_seconds,
        )
        incoming_guard_seconds = self._source_time(
            incoming_source,
            boundary_master_seconds + guard_master_seconds,
        ) - incoming_boundary_local
        outgoing_visible_start = self._source_time(
            outgoing_source,
            outgoing_candidate.start_seconds,
        )
        incoming_visible_end = self._source_time(
            incoming_source,
            incoming_candidate.end_seconds,
        )
        outgoing_near = any(
            outgoing_visible_start + _EPSILON < cut < outgoing_boundary_local - _EPSILON
            and outgoing_boundary_local - cut
            <= outgoing_guard_seconds + _EPSILON
            for cut in self.scene_cut_times[outgoing_source]
        )
        incoming_near = any(
            incoming_boundary_local + _EPSILON < cut < incoming_visible_end - _EPSILON
            and cut - incoming_boundary_local
            <= incoming_guard_seconds + _EPSILON
            for cut in self.scene_cut_times[incoming_source]
        )
        failures: list[str] = []
        if outgoing_near:
            failures.append("outgoing_source_local_camera_cut_near_boundary")
        if incoming_near:
            failures.append("incoming_source_local_camera_cut_near_boundary")
        return tuple(failures)

    @staticmethod
    def _gate_failure_codes(
        evidence: BoundaryVisualEvidence,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        gates = _PRODUCTION_REVIEW_GATES
        hard: list[str] = []
        crop_resolvable: list[str] = []
        for code, value, maximum, resolvable in (
            (
                "scale_mismatch",
                evidence.shot_scale_delta,
                gates.maximum_scale_mismatch,
                True,
            ),
            (
                "focus_mismatch",
                evidence.focus_mismatch,
                gates.maximum_focus_mismatch,
                True,
            ),
            (
                "person_layout_mismatch",
                evidence.layout_mismatch,
                gates.maximum_layout_mismatch,
                False,
            ),
            (
                "persisted_layout_mismatch",
                evidence.persisted_layout_mismatch,
                gates.maximum_layout_mismatch,
                False,
            ),
            (
                "silhouette_mismatch",
                evidence.silhouette_mismatch,
                gates.maximum_silhouette_mismatch,
                False,
            ),
            (
                "formation_mismatch",
                evidence.formation_mismatch,
                gates.maximum_formation_mismatch,
                False,
            ),
            (
                "pose_mismatch",
                evidence.pose_mismatch,
                gates.maximum_pose_mismatch,
                False,
            ),
        ):
            if value is None or value <= maximum + _EPSILON:
                continue
            (crop_resolvable if resolvable else hard).append(code)
        for side, stability in (
            ("outgoing", evidence.outgoing_side_stability),
            ("incoming", evidence.incoming_side_stability),
        ):
            if stability is None:
                hard.append(f"{side}_boundary_side_evidence_missing")
                continue
            for metric, value, maximum in (
                (
                    "scale",
                    stability.scale_delta,
                    gates.maximum_boundary_side_scale_delta,
                ),
                (
                    "structure",
                    stability.structure_mismatch,
                    gates.maximum_boundary_side_structure_mismatch,
                ),
                (
                    "pose",
                    stability.pose_mismatch,
                    gates.maximum_boundary_side_pose_mismatch,
                ),
                (
                    "shot_type",
                    stability.shot_type_mismatch,
                    gates.maximum_boundary_side_shot_type_mismatch,
                ),
            ):
                if value is None or maximum is None or value <= maximum + _EPSILON:
                    continue
                code = f"{side}_boundary_side_{metric}_instability"
                if metric == "scale":
                    crop_resolvable.append(code)
                else:
                    hard.append(code)
        return tuple(sorted(set(hard))), tuple(sorted(set(crop_resolvable)))

    def _temporal_pose_candidates(
        self,
        *,
        source_index: int,
        exact_master_seconds: float,
        exact_source_seconds: float,
        exact_descriptor: FrameVisualDescriptor,
        direction: int,
        minimum_master_seconds: float,
        maximum_master_seconds: float,
    ) -> tuple[tuple[int, FrameVisualDescriptor], ...]:
        if _actual_pose_available(exact_descriptor):
            return ()
        values: list[tuple[int, FrameVisualDescriptor]] = []
        frame_seconds = 1.0 / self.output_fps
        cuts = self.scene_cut_times[source_index]
        for distance in range(1, _NEAREST_TEMPORAL_POSE_RADIUS_FRAMES + 1):
            offset = direction * distance
            candidate_master_seconds = (
                exact_master_seconds + offset * frame_seconds
            )
            if (
                candidate_master_seconds
                < minimum_master_seconds - _EPSILON
                or candidate_master_seconds
                > maximum_master_seconds + _EPSILON
            ):
                break
            seconds = self._source_time(
                source_index,
                candidate_master_seconds,
            )
            lower, upper = sorted((exact_source_seconds, seconds))
            if any(lower + _EPSILON < cut < upper - _EPSILON for cut in cuts):
                break
            descriptor = self._sample_descriptor(source_index, seconds)
            if descriptor is None:
                continue
            values.append((offset, descriptor))
            if _stable_temporal_pose_choice(
                exact_descriptor,
                values,
                output_fps=self.output_fps,
            ) is not None:
                break
        return tuple(values)

    @staticmethod
    def _renderer_keyframe(
        transform: BoundaryTransformState,
        time_seconds: float,
    ) -> EditorialTransformKeyframe:
        crop = transform.crop_xywh
        if crop is None:
            return EditorialTransformKeyframe(time_seconds, 1.0)
        x, y, width, height = crop
        if not math.isclose(width, height, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("renderer transform cannot represent anisotropic crop")
        return EditorialTransformKeyframe(
            time_seconds=time_seconds,
            zoom=1.0 / width,
            center_x=x + width / 2.0,
            center_y=y + height / 2.0,
            roll_degrees=transform.rotation_degrees,
        )

    @staticmethod
    def _renderer_crop(
        keyframe: EditorialTransformKeyframe,
    ) -> tuple[float, float, float, float] | None:
        if (
            abs(keyframe.zoom - 1.0) <= _EPSILON
            and abs(keyframe.center_x - 0.5) <= _EPSILON
            and abs(keyframe.center_y - 0.5) <= _EPSILON
            and abs(keyframe.roll_degrees) <= _EPSILON
        ):
            return None
        scale = 1.0 / keyframe.zoom
        return (
            keyframe.center_x - scale / 2.0,
            keyframe.center_y - scale / 2.0,
            scale,
            scale,
        )

    @staticmethod
    def _renderer_states_equal(
        left: EditorialTransformKeyframe,
        right: EditorialTransformKeyframe,
    ) -> bool:
        return bool(
            math.isclose(left.zoom, right.zoom, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(
                left.center_x,
                right.center_x,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                left.center_y,
                right.center_y,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                left.roll_degrees,
                right.roll_degrees,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )

    @staticmethod
    def _crops_equal(
        left: tuple[float, float, float, float] | None,
        right: tuple[float, float, float, float] | None,
    ) -> bool:
        if left is None or right is None:
            return left is right
        return all(
            math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)
            for a, b in zip(left, right, strict=True)
        )

    def _renderer_track(
        self,
        *,
        candidate: TimelineCandidate,
        entry_transform: BoundaryTransformState,
        exit_transform: BoundaryTransformState,
        duration_seconds: float,
    ) -> EditorialTransformTrack:
        entry = self._renderer_keyframe(entry_transform, 0.0)
        exit_ = self._renderer_keyframe(exit_transform, duration_seconds)
        if self._renderer_states_equal(entry, exit_):
            return EditorialTransformTrack(
                (entry, exit_),
                reason="joint_exact_constant_segment_transform_path",
            )

        hold_seconds = _BOUNDARY_TRANSFORM_HOLD_FRAMES / self.output_fps
        has_entry_boundary = candidate.segment_index > 0
        has_exit_boundary = (
            candidate.segment_index < self.total_segment_count - 1
        )
        entry_hold_end = hold_seconds if has_entry_boundary else 0.0
        exit_hold_start = (
            duration_seconds - hold_seconds
            if has_exit_boundary
            else duration_seconds
        )
        if entry_hold_end >= exit_hold_start - _EPSILON:
            raise ValueError(
                "segment is too short for exact boundary transform holds"
            )
        points: list[EditorialTransformKeyframe] = [entry]
        if entry_hold_end > _EPSILON:
            points.append(
                self._renderer_keyframe(
                    entry_transform,
                    entry_hold_end,
                )
            )
        if exit_hold_start < duration_seconds - _EPSILON:
            points.append(
                self._renderer_keyframe(
                    exit_transform,
                    exit_hold_start,
                )
            )
        points.append(exit_)
        return EditorialTransformTrack(
            tuple(points),
            reason="joint_exact_boundary_hold_segment_transform_path",
        )

    @staticmethod
    def _bbox_crop_coverage(
        bbox: tuple[float, float, float, float],
        crop: tuple[float, float, float, float],
    ) -> float:
        left = max(bbox[0], crop[0])
        top = max(bbox[1], crop[1])
        right = min(bbox[0] + bbox[2], crop[0] + crop[2])
        bottom = min(bbox[1] + bbox[3], crop[1] + crop[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        return intersection / max(bbox[2] * bbox[3], 1e-9)

    def _transform_path_evidence(
        self,
        *,
        candidate: TimelineCandidate,
        entry_transform: BoundaryTransformState,
        exit_transform: BoundaryTransformState,
        start_master_seconds: float,
        end_master_seconds: float,
        duration_frames: float,
        context_identity: str,
    ) -> _TransformPathEvidence:
        expected_duration_frames = (
            end_master_seconds - start_master_seconds
        ) * self.output_fps
        if not math.isclose(
            expected_duration_frames,
            duration_frames,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError("segment bridge duration disagrees with endpoint clocks")
        base_payload: dict[str, object] = {
            "contract": "joint-segment-transform-path-proof-v1",
            "analysis_basis": self.analysis_basis,
            "configuration_identity": self._configuration_identity,
            "context_identity": context_identity,
            "candidate_state_sha256": _candidate_private_state_sha256(candidate),
            "entry_transform": _transform_payload(entry_transform),
            "exit_transform": _transform_payload(exit_transform),
            "start_master_seconds": start_master_seconds,
            "end_master_seconds": end_master_seconds,
            "duration_frames": duration_frames,
        }
        source_index = self._source_index(candidate.source_id)
        expected_sample_count = 0
        expected_frame_center_count = 0
        transient_samples: list[SegmentTransformPathSample] = []
        editorial_track: EditorialTransformTrack | None = None

        def outcome(
            *,
            evidence_complete: bool,
            crop_compatible: bool,
            track_compatible: bool,
            failure_codes: Sequence[str],
            samples: Sequence[Mapping[str, object]] = (),
        ) -> _TransformPathEvidence:
            codes = tuple(sorted(set(failure_codes)))
            endpoints_included = bool(
                samples
                and math.isclose(
                    float(samples[0]["master_seconds"]),
                    start_master_seconds,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and math.isclose(
                    float(samples[-1]["master_seconds"]),
                    end_master_seconds,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )
            full_frame_centers_included = bool(
                endpoints_included
                and len(samples) == expected_sample_count
                and expected_sample_count
                == expected_frame_center_count + 2
            )
            identity = "joint-segment-path:" + _sha256_json(
                {
                    **base_payload,
                    "evidence_complete": evidence_complete,
                    "crop_compatible": crop_compatible,
                    "track_compatible": track_compatible,
                    "failure_codes": list(codes),
                    "expected_sample_count": expected_sample_count,
                    "expected_frame_center_count": (
                        expected_frame_center_count
                    ),
                    "endpoints_included": endpoints_included,
                    "full_frame_centers_included": (
                        full_frame_centers_included
                    ),
                    "editorial_transform_track": (
                        {
                            "reason": editorial_track.reason,
                            "keyframes": [
                                {
                                    "time_seconds": item.time_seconds,
                                    "zoom": item.zoom,
                                    "center_x": item.center_x,
                                    "center_y": item.center_y,
                                    "roll_degrees": item.roll_degrees,
                                }
                                for item in editorial_track.keyframes
                            ],
                        }
                        if editorial_track is not None
                        else None
                    ),
                    "sample_evidence": list(samples),
                }
            )
            result = _TransformPathEvidence(
                proof_identity=identity,
                duration_frames=duration_frames,
                expected_sample_count=expected_sample_count,
                sampled_frame_count=len(samples),
                endpoints_included=endpoints_included,
                full_frame_centers_included=(
                    full_frame_centers_included
                ),
                evidence_complete=evidence_complete,
                crop_compatible=crop_compatible,
                track_compatible=track_compatible,
                failure_codes=codes,
            )
            self._bounded_transactional_set(
                self._segment_materialization_by_proof,
                identity,
                SegmentTransformPathMaterializationProof(
                    proof_identity=identity,
                    candidate=candidate,
                    source_index=source_index,
                    entry_transform=entry_transform,
                    exit_transform=exit_transform,
                    editorial_transform_track=editorial_track,
                    start_master_seconds=start_master_seconds,
                    end_master_seconds=end_master_seconds,
                    duration_frames=duration_frames,
                    expected_sample_count=expected_sample_count,
                    endpoints_included=endpoints_included,
                    full_frame_centers_included=(
                        full_frame_centers_included
                    ),
                    evidence_complete=evidence_complete,
                    crop_compatible=crop_compatible,
                    track_compatible=track_compatible,
                    failure_codes=codes,
                    samples=tuple(transient_samples),
                ),
                maximum_entries=_SEGMENT_MATERIALIZATION_LIMIT,
                cache_name="segment_materialization",
                pinned_keys=self._pinned_segment_proofs,
            )
            return result

        if duration_frames < _MINIMUM_SEGMENT_FRAMES - _EPSILON:
            return outcome(
                evidence_complete=True,
                crop_compatible=False,
                track_compatible=True,
                failure_codes=("segment_duration_below_two_frames",),
            )

        # Rendering samples visible frames at half-frame centres.  Bind both
        # exact endpoint transforms and every such centre; boundary-aligned
        # integer-frame sampling alone can miss a one-frame crop/track escape.
        expected_frame_center_count = int(
            math.ceil(
                max(
                    0.0,
                    duration_frames - 0.5 - _EPSILON,
                )
            )
        )
        frame_centers = tuple(
            start_master_seconds + (index + 0.5) / self.output_fps
            for index in range(expected_frame_center_count)
            if start_master_seconds + (index + 0.5) / self.output_fps
            < end_master_seconds - _EPSILON
        )
        expected_frame_center_count = len(frame_centers)
        sample_times = (
            start_master_seconds,
            *frame_centers,
            end_master_seconds,
        )
        expected_sample_count = len(sample_times)

        if entry_transform.track_id is None or exit_transform.track_id is None:
            return outcome(
                evidence_complete=False,
                crop_compatible=False,
                track_compatible=False,
                failure_codes=("segment_track_identity_evidence_incomplete",),
            )
        if entry_transform.track_id != exit_transform.track_id:
            return outcome(
                evidence_complete=True,
                crop_compatible=False,
                track_compatible=False,
                failure_codes=("segment_track_identity_mismatch",),
            )
        if (
            abs(entry_transform.rotation_degrees) > _EPSILON
            or abs(exit_transform.rotation_degrees) > _EPSILON
        ):
            return outcome(
                evidence_complete=False,
                crop_compatible=False,
                track_compatible=True,
                failure_codes=("segment_rotation_path_evidence_incomplete",),
            )
        try:
            editorial_track = self._renderer_track(
                candidate=candidate,
                entry_transform=entry_transform,
                exit_transform=exit_transform,
                duration_seconds=duration_frames / self.output_fps,
            )
        except ValueError as exc:
            failure = (
                "segment_too_short_for_exact_boundary_transform_holds"
                if "too short" in str(exc)
                else "segment_renderer_transform_envelope_violation"
            )
            return outcome(
                evidence_complete=True,
                crop_compatible=False,
                track_compatible=True,
                failure_codes=(failure,),
            )

        bases = {entry_transform.subject_basis, exit_transform.subject_basis}
        if "dominant_face" in bases:
            subject_basis = "dominant_face"
        elif "subject_silhouette" in bases:
            subject_basis = "subject_silhouette"
        else:
            subject_basis = "full_frame_identity"
        track_geometry_required = bool(
            entry_transform.crop_xywh is not None
            or exit_transform.crop_xywh is not None
        )
        source_endpoint_times = (
            self._source_time(source_index, start_master_seconds),
            self._source_time(source_index, end_master_seconds),
        )
        if min(source_endpoint_times) < 0.0:
            return outcome(
                evidence_complete=True,
                crop_compatible=False,
                track_compatible=True,
                failure_codes=("segment_negative_source_local_time",),
            )
        if track_geometry_required:
            source_low = min(source_endpoint_times)
            source_high = max(source_endpoint_times)
            if any(
                source_low + _EPSILON < cut < source_high - _EPSILON
                for cut in self.scene_cut_times[source_index]
            ):
                return outcome(
                    evidence_complete=True,
                    crop_compatible=False,
                    track_compatible=False,
                    failure_codes=("segment_source_scene_cut_breaks_track",),
                )

        samples: list[dict[str, object]] = []
        anchor_face_identity: tuple[float, ...] = ()
        previous_face_identity: tuple[float, ...] = ()
        for master_seconds in sample_times:
            fraction = (
                (master_seconds - start_master_seconds)
                / max(end_master_seconds - start_master_seconds, 1e-9)
            )
            source_seconds = self._source_time(source_index, master_seconds)
            assert editorial_track is not None
            renderer_keyframe = editorial_track.evaluate(
                max(0.0, master_seconds - start_master_seconds)
            )
            sample_crop = self._renderer_crop(renderer_keyframe)
            sample_transform = BoundaryTransformState(
                transform_identity=(
                    "joint-segment-sample-transform:"
                    + _sha256_json(
                        {
                            "context_identity": context_identity,
                            "master_seconds": master_seconds,
                            "source_seconds": source_seconds,
                            "fraction": fraction,
                            "crop_xywh": (
                                list(sample_crop)
                                if sample_crop is not None
                                else None
                            ),
                            "track_id": entry_transform.track_id,
                        }
                    )
                ),
                crop_xywh=sample_crop,
                rotation_degrees=renderer_keyframe.roll_degrees,
                track_id=entry_transform.track_id,
                subject_basis=subject_basis,
            )
            if source_seconds < 0.0:
                transient_samples.append(
                    SegmentTransformPathSample(
                        master_seconds=master_seconds,
                        source_seconds=source_seconds,
                        transform=sample_transform,
                        subject_basis=subject_basis,
                        subject_bbox_xywh=None,
                        subject_bbox_coverage=None,
                        face_similarity_to_anchor=None,
                        face_similarity_to_previous=None,
                        descriptor=None,
                        face_identity_embedding=(),
                    )
                )
                return outcome(
                    evidence_complete=True,
                    crop_compatible=False,
                    track_compatible=True,
                    failure_codes=("segment_negative_source_local_time",),
                    samples=samples,
                )
            descriptor = self._sample_descriptor(source_index, source_seconds)
            if descriptor is None:
                transient_samples.append(
                    SegmentTransformPathSample(
                        master_seconds=master_seconds,
                        source_seconds=source_seconds,
                        transform=sample_transform,
                        subject_basis=subject_basis,
                        subject_bbox_xywh=None,
                        subject_bbox_coverage=None,
                        face_similarity_to_anchor=None,
                        face_similarity_to_previous=None,
                        descriptor=None,
                        face_identity_embedding=(),
                    )
                )
                return outcome(
                    evidence_complete=False,
                    crop_compatible=False,
                    track_compatible=True,
                    failure_codes=("segment_descriptor_evidence_incomplete",),
                    samples=samples,
                )
            if not track_geometry_required:
                bbox = None
            elif subject_basis == "dominant_face":
                bbox = descriptor.primary_face_bbox
            elif subject_basis == "subject_silhouette":
                bbox = descriptor.silhouette_bbox_xywh
            else:
                bbox = descriptor.silhouette_bbox_xywh
            if track_geometry_required and bbox is None:
                transient_samples.append(
                    SegmentTransformPathSample(
                        master_seconds=master_seconds,
                        source_seconds=source_seconds,
                        transform=sample_transform,
                        subject_basis=subject_basis,
                        subject_bbox_xywh=None,
                        subject_bbox_coverage=None,
                        face_similarity_to_anchor=None,
                        face_similarity_to_previous=None,
                        descriptor=descriptor,
                        face_identity_embedding=(),
                    )
                )
                return outcome(
                    evidence_complete=False,
                    crop_compatible=False,
                    track_compatible=True,
                    failure_codes=("segment_subject_track_evidence_incomplete",),
                    samples=samples,
                )
            crop = sample_crop or (0.0, 0.0, 1.0, 1.0)
            if (
                crop[0] < -_EPSILON
                or crop[1] < -_EPSILON
                or crop[2] <= 0.0
                or crop[3] <= 0.0
                or crop[0] + crop[2] > 1.0 + _EPSILON
                or crop[1] + crop[3] > 1.0 + _EPSILON
            ):
                return outcome(
                    evidence_complete=True,
                    crop_compatible=False,
                    track_compatible=True,
                    failure_codes=("segment_crop_path_out_of_frame",),
                    samples=samples,
                )
            coverage = (
                self._bbox_crop_coverage(bbox, crop)
                if bbox is not None
                else None
            )
            face_identity: tuple[float, ...] = ()
            similarity_to_anchor: float | None = None
            similarity_to_previous: float | None = None
            if subject_basis == "dominant_face":
                face_identity = self._sample_face_identity(
                    source_index,
                    source_seconds,
                    descriptor,
                )
                if face_identity:
                    if not anchor_face_identity:
                        anchor_face_identity = face_identity
                    if not previous_face_identity:
                        previous_face_identity = face_identity
                    similarity_to_anchor = cosine_similarity(
                        anchor_face_identity,
                        face_identity,
                    )
                    similarity_to_previous = cosine_similarity(
                        previous_face_identity,
                        face_identity,
                    )
            transient_samples.append(
                SegmentTransformPathSample(
                    master_seconds=master_seconds,
                    source_seconds=source_seconds,
                    transform=sample_transform,
                    subject_basis=subject_basis,
                    subject_bbox_xywh=bbox,
                    subject_bbox_coverage=coverage,
                    face_similarity_to_anchor=similarity_to_anchor,
                    face_similarity_to_previous=similarity_to_previous,
                    descriptor=descriptor,
                    face_identity_embedding=face_identity,
                )
            )
            if coverage is not None and coverage < 0.95 - _EPSILON:
                return outcome(
                    evidence_complete=True,
                    crop_compatible=False,
                    track_compatible=True,
                    failure_codes=("segment_subject_exits_crop_track",),
                    samples=samples,
                )
            if subject_basis == "dominant_face" and (
                not face_identity
                or similarity_to_anchor is None
                or similarity_to_previous is None
            ):
                return outcome(
                    evidence_complete=False,
                    crop_compatible=True,
                    track_compatible=False,
                    failure_codes=(
                        "segment_dominant_face_identity_evidence_incomplete",
                    ),
                    samples=samples,
                )
            if subject_basis == "dominant_face" and (
                similarity_to_anchor
                < _MINIMUM_DOMINANT_FACE_TRACK_SIMILARITY
                or similarity_to_previous
                < _MINIMUM_DOMINANT_FACE_TRACK_SIMILARITY
            ):
                return outcome(
                    evidence_complete=True,
                    crop_compatible=True,
                    track_compatible=False,
                    failure_codes=(
                        "segment_dominant_face_track_identity_mismatch",
                    ),
                    samples=samples,
                )
            samples.append(
                {
                    "master_seconds": master_seconds,
                    "source_seconds": source_seconds,
                    "crop_xywh": (
                        list(sample_crop)
                        if sample_crop is not None
                        else None
                    ),
                    "subject_basis": subject_basis,
                    "subject_bbox_coverage": coverage,
                    "descriptor_private_state_sha256": (
                        _descriptor_private_state_sha256(descriptor)
                    ),
                    "face_identity_private_state_sha256": (
                        _identity_private_state_sha256(face_identity)
                        if face_identity
                        else None
                    ),
                    "face_similarity_to_anchor": similarity_to_anchor,
                    "face_similarity_to_previous": similarity_to_previous,
                }
            )
            if face_identity:
                previous_face_identity = face_identity
        return outcome(
            evidence_complete=True,
            crop_compatible=True,
            track_compatible=True,
            failure_codes=(),
            samples=samples,
        )

    def _outer_track_proofs(
        self,
        *,
        key: BoundaryMeasurementKey,
        frame_offset: int,
        boundary_master_seconds: float,
        raw_context_identity: str,
        outgoing_transform: BoundaryTransformState,
        incoming_transform: BoundaryTransformState,
    ) -> tuple[_TransformPathEvidence, ...]:
        proofs: list[_TransformPathEvidence] = []
        if key.boundary_index == 0:
            entry_identity = _identity_transform(
                raw_context_identity=(raw_context_identity + ":first-outer"),
                side="first-segment-entry",
                candidate=key.outgoing,
            )
            proofs.append(
                self._transform_path_evidence(
                    candidate=key.outgoing,
                    entry_transform=entry_identity,
                    exit_transform=outgoing_transform,
                    start_master_seconds=key.outgoing.start_seconds,
                    end_master_seconds=boundary_master_seconds,
                    duration_frames=(
                        key.outgoing.duration_seconds * self.output_fps
                        + frame_offset
                    ),
                    context_identity=(raw_context_identity + ":first-outer"),
                )
            )
        if key.boundary_index == self.total_segment_count - 2:
            exit_identity = _identity_transform(
                raw_context_identity=(raw_context_identity + ":last-outer"),
                side="last-segment-exit",
                candidate=key.incoming,
            )
            proofs.append(
                self._transform_path_evidence(
                    candidate=key.incoming,
                    entry_transform=incoming_transform,
                    exit_transform=exit_identity,
                    start_master_seconds=boundary_master_seconds,
                    end_master_seconds=key.incoming.end_seconds,
                    duration_frames=(
                        key.incoming.duration_seconds * self.output_fps
                        - frame_offset
                    ),
                    context_identity=(raw_context_identity + ":last-outer"),
                )
            )
        return tuple(proofs)

    def _measure(
        self,
        key: BoundaryMeasurementKey,
    ) -> tuple[BoundaryOptionSet, _MeasurementAuditRecord]:
        _raw_payload, raw_context_identity = self._raw_context(key)
        measurement_identity = "joint-boundary-measurement:" + _sha256_json(
            {
                "raw_context_identity": raw_context_identity,
                "offset_domain": list(JOINT_BOUNDARY_FRAME_OFFSETS),
            }
        )
        outgoing_source = self._source_index(key.outgoing.source_id)
        incoming_source = self._source_index(key.incoming.source_id)

        if outgoing_source == incoming_source:
            return self._measure_same_source(
                key,
                raw_context_identity=raw_context_identity,
                measurement_identity=measurement_identity,
            )

        nominal = key.outgoing.end_seconds
        frame_seconds = 1.0 / self.output_fps
        audits: list[BoundaryOffsetAudit] = []
        records: list[_OffsetAuditRecord] = []
        for frame_offset in JOINT_BOUNDARY_FRAME_OFFSETS:
            boundary = nominal + frame_offset * frame_seconds
            if (
                boundary - key.outgoing.start_seconds
                < _MINIMUM_SEGMENT_FRAMES * frame_seconds - _EPSILON
                or key.incoming.end_seconds - boundary
                < _MINIMUM_SEGMENT_FRAMES * frame_seconds - _EPSILON
            ):
                audit, record = self._incomplete_audit(
                    frame_offset=frame_offset,
                    raw_context_identity=raw_context_identity,
                    failure_codes=(
                        "rendered_side_stability_span_evidence_incomplete",
                    ),
                )
                audits.append(audit)
                records.append(record)
                continue
            outgoing_span_frames = (
                boundary - key.outgoing.start_seconds
            ) * self.output_fps
            incoming_span_frames = (
                key.incoming.end_seconds - boundary
            ) * self.output_fps
            side_sample_radius_frames = min(
                _SIDE_SAMPLE_RADIUS_FRAMES,
                int(math.floor(outgoing_span_frames - 1.0 + _EPSILON)),
                int(math.floor(incoming_span_frames - 1.0 + _EPSILON)),
            )
            if side_sample_radius_frames < 1:
                audit, record = self._incomplete_audit(
                    frame_offset=frame_offset,
                    raw_context_identity=raw_context_identity,
                    failure_codes=(
                        "rendered_side_stability_span_evidence_incomplete",
                    ),
                )
                audits.append(audit)
                records.append(record)
                continue
            outgoing_local = self._source_time(outgoing_source, boundary)
            incoming_local = self._source_time(incoming_source, boundary)
            if min(outgoing_local, incoming_local) < 0.0:
                audit, record = self._hard_audit(
                    key=key,
                    frame_offset=frame_offset,
                    raw_context_identity=raw_context_identity,
                    failure_codes=("negative_source_local_time",),
                )
                audits.append(audit)
                records.append(record)
                continue
            scene_failures = self._scene_cut_failures(
                outgoing_source=outgoing_source,
                incoming_source=incoming_source,
                outgoing_candidate=key.outgoing,
                incoming_candidate=key.incoming,
                boundary_master_seconds=boundary,
                outgoing_boundary_local=outgoing_local,
                incoming_boundary_local=incoming_local,
            )
            if scene_failures:
                audit, record = self._hard_audit(
                    key=key,
                    frame_offset=frame_offset,
                    raw_context_identity=raw_context_identity,
                    failure_codes=scene_failures,
                )
                audits.append(audit)
                records.append(record)
                continue

            outgoing_exact_master_seconds = boundary - frame_seconds / 2.0
            incoming_exact_master_seconds = boundary + frame_seconds / 2.0
            outgoing_exact_seconds = self._source_time(
                outgoing_source,
                outgoing_exact_master_seconds,
            )
            incoming_exact_seconds = self._source_time(
                incoming_source,
                incoming_exact_master_seconds,
            )
            outgoing_preceding_seconds = self._source_time(
                outgoing_source,
                outgoing_exact_master_seconds
                - side_sample_radius_frames * frame_seconds,
            )
            incoming_following_seconds = self._source_time(
                incoming_source,
                incoming_exact_master_seconds
                + side_sample_radius_frames * frame_seconds,
            )
            if min(
                outgoing_exact_seconds,
                incoming_exact_seconds,
                outgoing_preceding_seconds,
                incoming_following_seconds,
            ) < 0.0:
                audit, record = self._hard_audit(
                    key=key,
                    frame_offset=frame_offset,
                    raw_context_identity=raw_context_identity,
                    failure_codes=("negative_source_local_time",),
                )
                audits.append(audit)
                records.append(record)
                continue
            outgoing_descriptor = self._sample_descriptor(
                outgoing_source,
                outgoing_exact_seconds,
            )
            outgoing_preceding = self._sample_descriptor(
                outgoing_source,
                outgoing_preceding_seconds,
            )
            incoming_descriptor = self._sample_descriptor(
                incoming_source,
                incoming_exact_seconds,
            )
            incoming_following = self._sample_descriptor(
                incoming_source,
                incoming_following_seconds,
            )
            missing: list[str] = []
            for name, descriptor in (
                ("outgoing_exact_descriptor", outgoing_descriptor),
                ("outgoing_preceding_descriptor", outgoing_preceding),
                ("incoming_exact_descriptor", incoming_descriptor),
                ("incoming_following_descriptor", incoming_following),
            ):
                if descriptor is None:
                    missing.append(name + "_unavailable")
            if missing:
                audit, record = self._incomplete_audit(
                    frame_offset=frame_offset,
                    raw_context_identity=raw_context_identity,
                    failure_codes=missing,
                )
                audits.append(audit)
                records.append(record)
                continue
            assert outgoing_descriptor is not None
            assert outgoing_preceding is not None
            assert incoming_descriptor is not None
            assert incoming_following is not None

            outgoing_temporal_pose_candidates = self._temporal_pose_candidates(
                source_index=outgoing_source,
                exact_master_seconds=outgoing_exact_master_seconds,
                exact_source_seconds=outgoing_exact_seconds,
                exact_descriptor=outgoing_descriptor,
                direction=-1,
                minimum_master_seconds=(
                    key.outgoing.start_seconds + frame_seconds / 2.0
                ),
                maximum_master_seconds=outgoing_exact_master_seconds,
            )
            incoming_temporal_pose_candidates = self._temporal_pose_candidates(
                source_index=incoming_source,
                exact_master_seconds=incoming_exact_master_seconds,
                exact_source_seconds=incoming_exact_seconds,
                exact_descriptor=incoming_descriptor,
                direction=1,
                minimum_master_seconds=incoming_exact_master_seconds,
                maximum_master_seconds=(
                    key.incoming.end_seconds - frame_seconds / 2.0
                ),
            )
            evidence = _visual_evidence(
                outgoing_descriptor,
                incoming_descriptor,
                outgoing_preceding=outgoing_preceding,
                incoming_following=incoming_following,
                outgoing_temporal_pose_candidates=(
                    outgoing_temporal_pose_candidates
                ),
                incoming_temporal_pose_candidates=(
                    incoming_temporal_pose_candidates
                ),
                output_fps=self.output_fps,
                side_sample_span_seconds=(
                    side_sample_radius_frames * frame_seconds
                ),
            )
            raw_evidence_identity = _sha256_json(
                {
                    "contract": "joint-boundary-decoded-raw-offset-proof-v1",
                    "raw_context_identity": raw_context_identity,
                    "frame_offset": frame_offset,
                    "side_sample_radius_frames": side_sample_radius_frames,
                    "master_clocks": {
                        "boundary": boundary,
                        "outgoing_exact": outgoing_exact_master_seconds,
                        "outgoing_preceding": (
                            outgoing_exact_master_seconds
                            - side_sample_radius_frames * frame_seconds
                        ),
                        "incoming_exact": incoming_exact_master_seconds,
                        "incoming_following": (
                            incoming_exact_master_seconds
                            + side_sample_radius_frames * frame_seconds
                        ),
                    },
                    "source_clocks": {
                        "outgoing_boundary": outgoing_local,
                        "outgoing_exact": outgoing_exact_seconds,
                        "outgoing_preceding": outgoing_preceding_seconds,
                        "incoming_boundary": incoming_local,
                        "incoming_exact": incoming_exact_seconds,
                        "incoming_following": incoming_following_seconds,
                    },
                    "descriptor_private_state_sha256": {
                        "outgoing_exact": _descriptor_private_state_sha256(
                            outgoing_descriptor
                        ),
                        "outgoing_preceding": _descriptor_private_state_sha256(
                            outgoing_preceding
                        ),
                        "incoming_exact": _descriptor_private_state_sha256(
                            incoming_descriptor
                        ),
                        "incoming_following": _descriptor_private_state_sha256(
                            incoming_following
                        ),
                    },
                    "temporal_pose_candidate_private_state_sha256": {
                        "outgoing": [
                            [offset, _descriptor_private_state_sha256(item)]
                            for offset, item in outgoing_temporal_pose_candidates
                        ],
                        "incoming": [
                            [offset, _descriptor_private_state_sha256(item)]
                            for offset, item in incoming_temporal_pose_candidates
                        ],
                    },
                    "raw_evidence_private_state_sha256": (
                        _evidence_private_state_sha256(evidence)
                    ),
                }
            )
            safe_evidence = _safe_evidence_payload(evidence)
            if not evidence.evidence_coverage.professional_gate_passed:
                audit, record = self._incomplete_audit(
                    frame_offset=frame_offset,
                    raw_context_identity=raw_context_identity,
                    failure_codes=(
                        "professional_model_evidence_incomplete",
                        *evidence.evidence_coverage.missing_mask,
                    ),
                    safe_evidence=safe_evidence,
                    raw_evidence_identity=raw_evidence_identity,
                )
                audits.append(audit)
                records.append(record)
                continue

            hard_failures, crop_failures = self._gate_failure_codes(evidence)
            if hard_failures:
                audit, record = self._hard_audit(
                    key=key,
                    frame_offset=frame_offset,
                    raw_context_identity=raw_context_identity,
                    failure_codes=hard_failures,
                    safe_evidence=safe_evidence,
                    raw_evidence_identity=raw_evidence_identity,
                )
                audits.append(audit)
                records.append(record)
                continue

            strict_pass = _passes_production_transition_gates(evidence)
            if crop_failures:
                crop_transform_result = _crop_resolved_transforms(
                    raw_context_identity=raw_context_identity,
                    outgoing_candidate=key.outgoing,
                    incoming_candidate=key.incoming,
                    outgoing_descriptor=outgoing_descriptor,
                    incoming_descriptor=incoming_descriptor,
                )
                if crop_transform_result is None:
                    audit, record = self._incomplete_audit(
                        frame_offset=frame_offset,
                        raw_context_identity=raw_context_identity,
                        failure_codes=(
                            "geometry_crop_evidence_incomplete",
                            *crop_failures,
                        ),
                        safe_evidence=safe_evidence,
                        raw_evidence_identity=raw_evidence_identity,
                    )
                    audits.append(audit)
                    records.append(record)
                    continue
                transforms = crop_transform_result[:2]
            elif strict_pass:
                transforms = (
                    _identity_transform(
                        raw_context_identity=raw_context_identity,
                        side="outgoing",
                        candidate=key.outgoing,
                    ),
                    _identity_transform(
                        raw_context_identity=raw_context_identity,
                        side="incoming",
                        candidate=key.incoming,
                    ),
                )
            else:
                audit, record = self._incomplete_audit(
                    frame_offset=frame_offset,
                    raw_context_identity=raw_context_identity,
                    failure_codes=("unclassified_professional_gate_incomplete",),
                    safe_evidence=safe_evidence,
                    raw_evidence_identity=raw_evidence_identity,
                )
                audits.append(audit)
                records.append(record)
                continue

            outgoing_transform, incoming_transform = transforms
            raw_evidence = evidence
            raw_safe_evidence = safe_evidence
            result_outgoing_descriptor = outgoing_descriptor
            result_outgoing_preceding = outgoing_preceding
            result_incoming_descriptor = incoming_descriptor
            result_incoming_following = incoming_following
            descriptor_already_transformed = False
            if crop_failures:
                transformed_values = (
                    self._sample_transformed_descriptor(
                        outgoing_source,
                        outgoing_exact_seconds,
                        outgoing_descriptor,
                        outgoing_transform,
                    ),
                    self._sample_transformed_descriptor(
                        outgoing_source,
                        outgoing_preceding_seconds,
                        outgoing_preceding,
                        outgoing_transform,
                    ),
                    self._sample_transformed_descriptor(
                        incoming_source,
                        incoming_exact_seconds,
                        incoming_descriptor,
                        incoming_transform,
                    ),
                    self._sample_transformed_descriptor(
                        incoming_source,
                        incoming_following_seconds,
                        incoming_following,
                        incoming_transform,
                    ),
                )
                if any(value is None for value in transformed_values):
                    audit, record = self._incomplete_audit(
                        frame_offset=frame_offset,
                        raw_context_identity=raw_context_identity,
                        failure_codes=(
                            "post_transform_descriptor_evidence_incomplete",
                            *crop_failures,
                        ),
                        safe_evidence={
                            "raw": raw_safe_evidence,
                            "transform_result_remeasured": False,
                        },
                        raw_evidence_identity=raw_evidence_identity,
                    )
                    audits.append(audit)
                    records.append(record)
                    continue
                (
                    result_outgoing_descriptor,
                    result_outgoing_preceding,
                    result_incoming_descriptor,
                    result_incoming_following,
                ) = transformed_values
                assert result_outgoing_descriptor is not None
                assert result_outgoing_preceding is not None
                assert result_incoming_descriptor is not None
                assert result_incoming_following is not None
                result_evidence = _visual_evidence(
                    result_outgoing_descriptor,
                    result_incoming_descriptor,
                    outgoing_preceding=result_outgoing_preceding,
                    incoming_following=result_incoming_following,
                    output_fps=self.output_fps,
                    side_sample_span_seconds=(
                        side_sample_radius_frames * frame_seconds
                    ),
                )
                post_transform_result_evidence_identity = (
                    _boundary_result_evidence_identity(
                        raw_evidence_identity=raw_evidence_identity,
                        outgoing_transform=outgoing_transform,
                        incoming_transform=incoming_transform,
                        outgoing_descriptor=result_outgoing_descriptor,
                        outgoing_preceding=result_outgoing_preceding,
                        incoming_descriptor=result_incoming_descriptor,
                        incoming_following=result_incoming_following,
                        evidence=result_evidence,
                    )
                )
                result_safe_evidence = _safe_evidence_payload(result_evidence)
                combined_safe_evidence: dict[str, object] = {
                    "raw": raw_safe_evidence,
                    "result": result_safe_evidence,
                    "transform_result_remeasured": True,
                }
                if not (
                    result_evidence.evidence_coverage.professional_gate_passed
                ):
                    audit, record = self._incomplete_audit(
                        frame_offset=frame_offset,
                        raw_context_identity=raw_context_identity,
                        failure_codes=(
                            "post_transform_professional_model_evidence_incomplete",
                            *result_evidence.evidence_coverage.missing_mask,
                        ),
                        safe_evidence=combined_safe_evidence,
                        raw_evidence_identity=raw_evidence_identity,
                        result_evidence_identity=(
                            post_transform_result_evidence_identity
                        ),
                    )
                    audits.append(audit)
                    records.append(record)
                    continue
                result_hard_failures, result_crop_failures = (
                    self._gate_failure_codes(result_evidence)
                )
                if (
                    result_hard_failures
                    or result_crop_failures
                    or not _passes_production_transition_gates(result_evidence)
                ):
                    result_failures = tuple(
                        sorted(
                            {
                                *(
                                    "post_transform_" + code
                                    for code in result_hard_failures
                                ),
                                *(
                                    "post_transform_" + code
                                    for code in result_crop_failures
                                ),
                            }
                        )
                    ) or ("post_transform_professional_gate_failure",)
                    audit, record = self._hard_audit(
                        key=key,
                        frame_offset=frame_offset,
                        raw_context_identity=raw_context_identity,
                        failure_codes=result_failures,
                        safe_evidence=combined_safe_evidence,
                        raw_evidence_identity=raw_evidence_identity,
                        result_evidence_identity=(
                            post_transform_result_evidence_identity
                        ),
                    )
                    audits.append(audit)
                    records.append(record)
                    continue
                evidence = result_evidence
                safe_evidence = combined_safe_evidence
                descriptor_already_transformed = True
            result_evidence_identity = _boundary_result_evidence_identity(
                raw_evidence_identity=raw_evidence_identity,
                outgoing_transform=outgoing_transform,
                incoming_transform=incoming_transform,
                outgoing_descriptor=result_outgoing_descriptor,
                outgoing_preceding=result_outgoing_preceding,
                incoming_descriptor=result_incoming_descriptor,
                incoming_following=result_incoming_following,
                evidence=evidence,
            )
            both_close = (
                result_outgoing_descriptor.shot_type.value == "close_up"
                and result_incoming_descriptor.shot_type.value == "close_up"
            )
            outgoing_identity = self._sample_face_identity(
                outgoing_source,
                outgoing_exact_seconds,
                outgoing_descriptor,
            )
            incoming_identity = self._sample_face_identity(
                incoming_source,
                incoming_exact_seconds,
                incoming_descriptor,
            )
            if both_close and (not outgoing_identity or not incoming_identity):
                audit, record = self._incomplete_audit(
                    frame_offset=frame_offset,
                    raw_context_identity=raw_context_identity,
                    failure_codes=("close_up_face_identity_evidence_incomplete",),
                    safe_evidence=safe_evidence,
                    raw_evidence_identity=raw_evidence_identity,
                    result_evidence_identity=result_evidence_identity,
                )
                audits.append(audit)
                records.append(record)
                continue

            scoring_outgoing = _updated_outgoing_candidate(
                key.outgoing,
                result_outgoing_descriptor,
                evidence,
                scene_cut_near=False,
                face_identity=outgoing_identity,
                transform=outgoing_transform,
                transformed_shot_scale=None,
                descriptor_already_transformed=(
                    descriptor_already_transformed
                ),
            )
            scoring_incoming = _updated_incoming_candidate(
                key.incoming,
                result_incoming_descriptor,
                evidence,
                scene_cut_near=False,
                face_identity=incoming_identity,
                transform=incoming_transform,
                transformed_shot_scale=None,
                descriptor_already_transformed=(
                    descriptor_already_transformed
                ),
            )
            outer_proofs = self._outer_track_proofs(
                key=key,
                frame_offset=frame_offset,
                boundary_master_seconds=boundary,
                raw_context_identity=raw_context_identity,
                outgoing_transform=outgoing_transform,
                incoming_transform=incoming_transform,
            )
            for outer_index, outer_proof in enumerate(outer_proofs):
                outer_record = _BridgeAuditRecord(
                    proof_identity=outer_proof.proof_identity,
                    candidate_id=(
                        key.outgoing.candidate_id
                        if key.boundary_index == 0 and outer_index == 0
                        else key.incoming.candidate_id
                    ),
                    entry_proof_identity=(
                        f"outer-entry:{raw_context_identity}:{outer_index}"
                    ),
                    exit_proof_identity=(
                        f"outer-exit:{raw_context_identity}:{outer_index}"
                    ),
                    duration_frames=outer_proof.duration_frames,
                    expected_sample_count=(
                        outer_proof.expected_sample_count
                    ),
                    sampled_frame_count=outer_proof.sampled_frame_count,
                    endpoints_included=outer_proof.endpoints_included,
                    full_frame_centers_included=(
                        outer_proof.full_frame_centers_included
                    ),
                    evidence_complete=outer_proof.evidence_complete,
                    crop_compatible=outer_proof.crop_compatible,
                    track_compatible=outer_proof.track_compatible,
                    failure_codes=outer_proof.failure_codes,
                )
                self._register_outer_track_audit(outer_record)
            outer_incomplete = tuple(
                failure
                for proof in outer_proofs
                if not proof.evidence_complete
                for failure in proof.failure_codes
            )
            outer_hard = tuple(
                failure
                for proof in outer_proofs
                if proof.evidence_complete
                and not (proof.crop_compatible and proof.track_compatible)
                for failure in proof.failure_codes
            )
            if outer_hard:
                audit, record = self._hard_audit(
                    key=key,
                    frame_offset=frame_offset,
                    raw_context_identity=raw_context_identity,
                    failure_codes=tuple(
                        "outer_track_" + code for code in outer_hard
                    ),
                    safe_evidence=safe_evidence,
                    raw_evidence_identity=raw_evidence_identity,
                    result_evidence_identity=result_evidence_identity,
                )
                audits.append(audit)
                records.append(record)
                continue
            if outer_incomplete:
                audit, record = self._incomplete_audit(
                    frame_offset=frame_offset,
                    raw_context_identity=raw_context_identity,
                    failure_codes=tuple(
                        "outer_track_" + code for code in outer_incomplete
                    ),
                    safe_evidence=safe_evidence,
                    raw_evidence_identity=raw_evidence_identity,
                    result_evidence_identity=result_evidence_identity,
                )
                audits.append(audit)
                records.append(record)
                continue
            beat_deviation_seconds = abs(frame_offset) * frame_seconds
            beat_penalty = (
                _BEAT_PENALTY_WEIGHT
                * float(self.beat_intensity)
                * beat_deviation_seconds
                / float(self.beat_period_seconds)
                if self.beat_period_seconds is not None
                and self.beat_intensity is not None
                else 0.0
            )
            result_payload = {
                "contract": "joint-boundary-result-context-v1",
                "raw_context_identity": raw_context_identity,
                "raw_evidence_identity": raw_evidence_identity,
                "result_evidence_identity": result_evidence_identity,
                "frame_offset": frame_offset,
                "boundary_seconds": boundary,
                "side_sample_radius_frames": side_sample_radius_frames,
                "outgoing_source_local_seconds": outgoing_local,
                "incoming_source_local_seconds": incoming_local,
                "outgoing_exact_sample_seconds": outgoing_exact_seconds,
                "incoming_exact_sample_seconds": incoming_exact_seconds,
                "safe_scalar_evidence": safe_evidence,
                "crop_resolved_failure_codes": list(crop_failures),
                "beat_deviation_seconds": beat_deviation_seconds,
                "beat_penalty": beat_penalty,
                "beat_penalty_weight": _BEAT_PENALTY_WEIGHT,
                "outer_track_proof_identities": [
                    proof.proof_identity for proof in outer_proofs
                ],
                "outgoing_transform": _transform_payload(outgoing_transform),
                "incoming_transform": _transform_payload(incoming_transform),
                "result_descriptor_private_state_sha256": {
                    "outgoing_exact": _descriptor_private_state_sha256(
                        result_outgoing_descriptor
                    ),
                    "outgoing_preceding": _descriptor_private_state_sha256(
                        result_outgoing_preceding
                    ),
                    "incoming_exact": _descriptor_private_state_sha256(
                        result_incoming_descriptor
                    ),
                    "incoming_following": _descriptor_private_state_sha256(
                        result_incoming_following
                    ),
                },
                "result_evidence_private_state_sha256": (
                    _evidence_private_state_sha256(evidence)
                ),
                "scoring_outgoing_private_state_sha256": (
                    _candidate_private_state_sha256(scoring_outgoing)
                ),
                "scoring_incoming_private_state_sha256": (
                    _candidate_private_state_sha256(scoring_incoming)
                ),
            }
            result_context_identity = _sha256_json(result_payload)
            proof_identity = (
                "joint-boundary-proof-v1:raw="
                + raw_evidence_identity
                + ":result="
                + result_context_identity
            )
            option = AuthorizedBoundaryOption(
                measurement_key=key,
                frame_offset=frame_offset,
                outgoing_transform=outgoing_transform,
                incoming_transform=incoming_transform,
                proof_identity=proof_identity,
                score_adjustment=-float(evidence.total_cost + beat_penalty),
                scoring_outgoing=scoring_outgoing,
                scoring_incoming=scoring_incoming,
            )
            audit = BoundaryOffsetAudit(
                frame_offset=frame_offset,
                disposition=BoundaryOffsetDisposition.AUTHORIZED,
                evidence_identity=proof_identity,
                authorized_option=option,
            )
            record = _OffsetAuditRecord(
                frame_offset=frame_offset,
                disposition=BoundaryOffsetDisposition.AUTHORIZED,
                evidence_identity=proof_identity,
                failure_codes=(),
                raw_context_identity=raw_context_identity,
                result_context_identity=result_context_identity,
                safe_evidence=safe_evidence,
                outgoing_transform=outgoing_transform,
                incoming_transform=incoming_transform,
                raw_evidence_identity=raw_evidence_identity,
            )
            audits.append(audit)
            records.append(record)
            self._bounded_transactional_set(
                self._materialization_by_proof,
                proof_identity,
                BoundaryMaterializationProof(
                    proof_identity=proof_identity,
                    raw_proof_identity=raw_evidence_identity,
                    result_proof_identity=result_context_identity,
                    measurement_key=key,
                    frame_offset=frame_offset,
                    boundary_master_seconds=boundary,
                    side_sample_radius_frames=side_sample_radius_frames,
                    outgoing_source_index=outgoing_source,
                    incoming_source_index=incoming_source,
                    outgoing_boundary_source_seconds=outgoing_local,
                    incoming_boundary_source_seconds=incoming_local,
                    outgoing_exact_sample_seconds=outgoing_exact_seconds,
                    incoming_exact_sample_seconds=incoming_exact_seconds,
                    outgoing_preceding_sample_seconds=(
                        outgoing_preceding_seconds
                    ),
                    incoming_following_sample_seconds=(
                        incoming_following_seconds
                    ),
                    outgoing_transform=outgoing_transform,
                    incoming_transform=incoming_transform,
                    raw_outgoing_descriptor=outgoing_descriptor,
                    raw_outgoing_preceding_descriptor=outgoing_preceding,
                    raw_incoming_descriptor=incoming_descriptor,
                    raw_incoming_following_descriptor=incoming_following,
                    result_outgoing_descriptor=result_outgoing_descriptor,
                    result_outgoing_preceding_descriptor=(
                        result_outgoing_preceding
                    ),
                    result_incoming_descriptor=result_incoming_descriptor,
                    result_incoming_following_descriptor=(
                        result_incoming_following
                    ),
                    raw_evidence=raw_evidence,
                    result_evidence=evidence,
                    outgoing_face_identity_embedding=outgoing_identity,
                    incoming_face_identity_embedding=incoming_identity,
                    scoring_outgoing=scoring_outgoing,
                    scoring_incoming=scoring_incoming,
                    outer_track_proof_identities=tuple(
                        proof.proof_identity for proof in outer_proofs
                    ),
                ),
                maximum_entries=_BOUNDARY_MATERIALIZATION_LIMIT,
                cache_name="boundary_materialization",
                pinned_keys=self._pinned_boundary_proofs,
            )
            if proof_identity in self._pinned_boundary_proofs:
                self._pinned_segment_proofs.update(
                    proof.proof_identity for proof in outer_proofs
                )

        option_set = BoundaryOptionSet(key, tuple(audits))
        record = self._canonical_measurement_audit(
            _MeasurementAuditRecord(
                measurement_identity=measurement_identity,
                boundary_index=key.boundary_index,
                outgoing_source_id=key.outgoing.source_id,
                incoming_source_id=key.incoming.source_id,
                offsets=tuple(records),
            )
        )
        return option_set, record

    def _measure_same_source(
        self,
        key: BoundaryMeasurementKey,
        *,
        raw_context_identity: str,
        measurement_identity: str,
    ) -> tuple[BoundaryOptionSet, _MeasurementAuditRecord]:
        audits: list[BoundaryOffsetAudit] = []
        records: list[_OffsetAuditRecord] = []
        for frame_offset in JOINT_BOUNDARY_FRAME_OFFSETS:
            if frame_offset != 0:
                audit, record = self._hard_audit(
                    key=key,
                    frame_offset=frame_offset,
                    raw_context_identity=raw_context_identity,
                    failure_codes=("same_source_timing_change_not_applicable",),
                )
            else:
                source_index = self._source_index(key.outgoing.source_id)
                boundary = key.outgoing.end_seconds
                frame_seconds = 1.0 / self.output_fps
                outgoing_span_frames = (
                    boundary - key.outgoing.start_seconds
                ) * self.output_fps
                incoming_span_frames = (
                    key.incoming.end_seconds - boundary
                ) * self.output_fps
                side_sample_radius_frames = min(
                    _SIDE_SAMPLE_RADIUS_FRAMES,
                    int(math.floor(outgoing_span_frames - 1.0 + _EPSILON)),
                    int(math.floor(incoming_span_frames - 1.0 + _EPSILON)),
                )
                if side_sample_radius_frames < 1:
                    audit, record = self._incomplete_audit(
                        frame_offset=0,
                        raw_context_identity=raw_context_identity,
                        failure_codes=(
                            "rendered_side_stability_span_evidence_incomplete",
                        ),
                    )
                    audits.append(audit)
                    records.append(record)
                    continue
                outgoing_boundary_local = self._source_time(
                    source_index,
                    boundary,
                )
                incoming_boundary_local = outgoing_boundary_local
                outgoing_exact_seconds = self._source_time(
                    source_index,
                    boundary - frame_seconds / 2.0,
                )
                incoming_exact_seconds = self._source_time(
                    source_index,
                    boundary + frame_seconds / 2.0,
                )
                outgoing_preceding_seconds = self._source_time(
                    source_index,
                    boundary
                    - (side_sample_radius_frames + 0.5) * frame_seconds,
                )
                incoming_following_seconds = self._source_time(
                    source_index,
                    boundary
                    + (side_sample_radius_frames + 0.5) * frame_seconds,
                )
                if min(
                    outgoing_boundary_local,
                    incoming_boundary_local,
                    outgoing_exact_seconds,
                    incoming_exact_seconds,
                    outgoing_preceding_seconds,
                    incoming_following_seconds,
                ) < 0.0:
                    audit, record = self._hard_audit(
                        key=key,
                        frame_offset=0,
                        raw_context_identity=raw_context_identity,
                        failure_codes=("negative_source_local_time",),
                    )
                    audits.append(audit)
                    records.append(record)
                    continue
                descriptors = (
                    self._sample_descriptor(
                        source_index,
                        outgoing_exact_seconds,
                    ),
                    self._sample_descriptor(
                        source_index,
                        outgoing_preceding_seconds,
                    ),
                    self._sample_descriptor(
                        source_index,
                        incoming_exact_seconds,
                    ),
                    self._sample_descriptor(
                        source_index,
                        incoming_following_seconds,
                    ),
                )
                if any(item is None for item in descriptors):
                    audit, record = self._incomplete_audit(
                        frame_offset=0,
                        raw_context_identity=raw_context_identity,
                        failure_codes=(
                            "same_source_boundary_descriptor_evidence_incomplete",
                        ),
                    )
                    audits.append(audit)
                    records.append(record)
                    continue
                (
                    outgoing_descriptor,
                    outgoing_preceding,
                    incoming_descriptor,
                    incoming_following,
                ) = descriptors
                assert outgoing_descriptor is not None
                assert outgoing_preceding is not None
                assert incoming_descriptor is not None
                assert incoming_following is not None
                evidence = _visual_evidence(
                    outgoing_descriptor,
                    incoming_descriptor,
                    outgoing_preceding=outgoing_preceding,
                    incoming_following=incoming_following,
                    output_fps=self.output_fps,
                    side_sample_span_seconds=(
                        side_sample_radius_frames * frame_seconds
                    ),
                )
                raw_evidence_identity = _sha256_json(
                    {
                        "contract": (
                            "joint-boundary-decoded-same-source-raw-proof-v1"
                        ),
                        "raw_context_identity": raw_context_identity,
                        "frame_offset": 0,
                        "side_sample_radius_frames": side_sample_radius_frames,
                        "master_clocks": {
                            "boundary": boundary,
                            "outgoing_exact": boundary - frame_seconds / 2.0,
                            "outgoing_preceding": boundary
                            - (side_sample_radius_frames + 0.5)
                            * frame_seconds,
                            "incoming_exact": boundary + frame_seconds / 2.0,
                            "incoming_following": boundary
                            + (side_sample_radius_frames + 0.5)
                            * frame_seconds,
                        },
                        "source_clocks": {
                            "outgoing_boundary": outgoing_boundary_local,
                            "incoming_boundary": incoming_boundary_local,
                            "outgoing_exact": outgoing_exact_seconds,
                            "outgoing_preceding": outgoing_preceding_seconds,
                            "incoming_exact": incoming_exact_seconds,
                            "incoming_following": incoming_following_seconds,
                        },
                        "descriptor_private_state_sha256": {
                            "outgoing_exact": _descriptor_private_state_sha256(
                                outgoing_descriptor
                            ),
                            "outgoing_preceding": (
                                _descriptor_private_state_sha256(
                                    outgoing_preceding
                                )
                            ),
                            "incoming_exact": _descriptor_private_state_sha256(
                                incoming_descriptor
                            ),
                            "incoming_following": (
                                _descriptor_private_state_sha256(
                                    incoming_following
                                )
                            ),
                        },
                        "raw_evidence_private_state_sha256": (
                            _evidence_private_state_sha256(evidence)
                        ),
                    }
                )
                outgoing_transform = _identity_transform(
                    raw_context_identity=raw_context_identity,
                    side="outgoing",
                    candidate=key.outgoing,
                )
                incoming_transform = _identity_transform(
                    raw_context_identity=raw_context_identity,
                    side="incoming",
                    candidate=key.incoming,
                )
                outgoing_identity = self._sample_face_identity(
                    source_index,
                    outgoing_exact_seconds,
                    outgoing_descriptor,
                )
                incoming_identity = self._sample_face_identity(
                    source_index,
                    incoming_exact_seconds,
                    incoming_descriptor,
                )
                scoring_outgoing = _updated_outgoing_candidate(
                    key.outgoing,
                    outgoing_descriptor,
                    evidence,
                    scene_cut_near=key.outgoing.exit_scene_cut_near,
                    face_identity=outgoing_identity,
                    transform=outgoing_transform,
                    transformed_shot_scale=None,
                )
                scoring_incoming = _updated_incoming_candidate(
                    key.incoming,
                    incoming_descriptor,
                    evidence,
                    scene_cut_near=key.incoming.entry_scene_cut_near,
                    face_identity=incoming_identity,
                    transform=incoming_transform,
                    transformed_shot_scale=None,
                )
                outer_proofs = self._outer_track_proofs(
                    key=key,
                    frame_offset=0,
                    boundary_master_seconds=boundary,
                    raw_context_identity=raw_context_identity,
                    outgoing_transform=outgoing_transform,
                    incoming_transform=incoming_transform,
                )
                outer_hard = tuple(
                    code
                    for proof in outer_proofs
                    if proof.evidence_complete
                    and not (
                        proof.crop_compatible and proof.track_compatible
                    )
                    for code in proof.failure_codes
                )
                outer_incomplete = tuple(
                    code
                    for proof in outer_proofs
                    if not proof.evidence_complete
                    for code in proof.failure_codes
                )
                outer_candidates: list[TimelineCandidate] = []
                if key.boundary_index == 0:
                    outer_candidates.append(key.outgoing)
                if key.boundary_index == self.total_segment_count - 2:
                    outer_candidates.append(key.incoming)
                for outer_index, (proof, outer_candidate) in enumerate(
                    zip(outer_proofs, outer_candidates, strict=True)
                ):
                    self._register_outer_track_audit(
                        _BridgeAuditRecord(
                            proof_identity=proof.proof_identity,
                            candidate_id=outer_candidate.candidate_id,
                            entry_proof_identity=(
                                "same-source-outer-entry:"
                                + raw_context_identity
                                + f":{outer_index}"
                            ),
                            exit_proof_identity=(
                                "same-source-outer-exit:"
                                + raw_context_identity
                                + f":{outer_index}"
                            ),
                            duration_frames=proof.duration_frames,
                            expected_sample_count=proof.expected_sample_count,
                            sampled_frame_count=proof.sampled_frame_count,
                            endpoints_included=proof.endpoints_included,
                            full_frame_centers_included=(
                                proof.full_frame_centers_included
                            ),
                            evidence_complete=proof.evidence_complete,
                            crop_compatible=proof.crop_compatible,
                            track_compatible=proof.track_compatible,
                            failure_codes=proof.failure_codes,
                        )
                    )
                if outer_hard:
                    audit, record = self._hard_audit(
                        key=key,
                        frame_offset=0,
                        raw_context_identity=raw_context_identity,
                        failure_codes=tuple(
                            "outer_track_" + code for code in outer_hard
                        ),
                        safe_evidence=_safe_evidence_payload(evidence),
                        raw_evidence_identity=raw_evidence_identity,
                    )
                    audits.append(audit)
                    records.append(record)
                    continue
                if outer_incomplete:
                    audit, record = self._incomplete_audit(
                        frame_offset=0,
                        raw_context_identity=raw_context_identity,
                        failure_codes=tuple(
                            "outer_track_" + code
                            for code in outer_incomplete
                        ),
                        safe_evidence=_safe_evidence_payload(evidence),
                        raw_evidence_identity=raw_evidence_identity,
                    )
                    audits.append(audit)
                    records.append(record)
                    continue
                result_context_identity = _sha256_json(
                    {
                        "contract": "joint-boundary-same-source-result-v1",
                        "raw_context_identity": raw_context_identity,
                        "raw_evidence_identity": raw_evidence_identity,
                        "frame_offset": 0,
                        "side_sample_radius_frames": side_sample_radius_frames,
                        "outgoing_transform": _transform_payload(
                            outgoing_transform
                        ),
                        "incoming_transform": _transform_payload(
                            incoming_transform
                        ),
                        "outer_track_proof_identities": [
                            proof.proof_identity for proof in outer_proofs
                        ],
                        "result_descriptor_private_state_sha256": {
                            "outgoing_exact": _descriptor_private_state_sha256(
                                outgoing_descriptor
                            ),
                            "outgoing_preceding": (
                                _descriptor_private_state_sha256(
                                    outgoing_preceding
                                )
                            ),
                            "incoming_exact": _descriptor_private_state_sha256(
                                incoming_descriptor
                            ),
                            "incoming_following": (
                                _descriptor_private_state_sha256(
                                    incoming_following
                                )
                            ),
                        },
                        "result_evidence_private_state_sha256": (
                            _evidence_private_state_sha256(evidence)
                        ),
                        "scoring_outgoing_private_state_sha256": (
                            _candidate_private_state_sha256(scoring_outgoing)
                        ),
                        "scoring_incoming_private_state_sha256": (
                            _candidate_private_state_sha256(scoring_incoming)
                        ),
                    }
                )
                proof_identity = (
                    "joint-boundary-proof-v1:raw="
                    + raw_evidence_identity
                    + ":result="
                    + result_context_identity
                )
                option = AuthorizedBoundaryOption(
                    measurement_key=key,
                    frame_offset=0,
                    outgoing_transform=outgoing_transform,
                    incoming_transform=incoming_transform,
                    proof_identity=proof_identity,
                    scoring_outgoing=scoring_outgoing,
                    scoring_incoming=scoring_incoming,
                )
                audit = BoundaryOffsetAudit(
                    frame_offset=0,
                    disposition=BoundaryOffsetDisposition.AUTHORIZED,
                    evidence_identity=proof_identity,
                    authorized_option=option,
                )
                record = _OffsetAuditRecord(
                    frame_offset=0,
                    disposition=BoundaryOffsetDisposition.AUTHORIZED,
                    evidence_identity=proof_identity,
                    failure_codes=(),
                    raw_context_identity=raw_context_identity,
                    result_context_identity=result_context_identity,
                    safe_evidence=_safe_evidence_payload(evidence),
                    outgoing_transform=outgoing_transform,
                    incoming_transform=incoming_transform,
                    raw_evidence_identity=raw_evidence_identity,
                )
                self._bounded_transactional_set(
                    self._materialization_by_proof,
                    proof_identity,
                    BoundaryMaterializationProof(
                        proof_identity=proof_identity,
                        raw_proof_identity=raw_evidence_identity,
                        result_proof_identity=result_context_identity,
                        measurement_key=key,
                        frame_offset=0,
                        boundary_master_seconds=boundary,
                        side_sample_radius_frames=side_sample_radius_frames,
                        outgoing_source_index=source_index,
                        incoming_source_index=source_index,
                        outgoing_boundary_source_seconds=(
                            outgoing_boundary_local
                        ),
                        incoming_boundary_source_seconds=(
                            incoming_boundary_local
                        ),
                        outgoing_exact_sample_seconds=(
                            outgoing_exact_seconds
                        ),
                        incoming_exact_sample_seconds=(
                            incoming_exact_seconds
                        ),
                        outgoing_preceding_sample_seconds=(
                            outgoing_preceding_seconds
                        ),
                        incoming_following_sample_seconds=(
                            incoming_following_seconds
                        ),
                        outgoing_transform=outgoing_transform,
                        incoming_transform=incoming_transform,
                        raw_outgoing_descriptor=outgoing_descriptor,
                        raw_outgoing_preceding_descriptor=(
                            outgoing_preceding
                        ),
                        raw_incoming_descriptor=incoming_descriptor,
                        raw_incoming_following_descriptor=(
                            incoming_following
                        ),
                        result_outgoing_descriptor=outgoing_descriptor,
                        result_outgoing_preceding_descriptor=(
                            outgoing_preceding
                        ),
                        result_incoming_descriptor=incoming_descriptor,
                        result_incoming_following_descriptor=(
                            incoming_following
                        ),
                        raw_evidence=evidence,
                        result_evidence=evidence,
                        outgoing_face_identity_embedding=outgoing_identity,
                        incoming_face_identity_embedding=incoming_identity,
                        scoring_outgoing=scoring_outgoing,
                        scoring_incoming=scoring_incoming,
                        outer_track_proof_identities=tuple(
                            proof.proof_identity for proof in outer_proofs
                        ),
                    ),
                    maximum_entries=_BOUNDARY_MATERIALIZATION_LIMIT,
                    cache_name="boundary_materialization",
                    pinned_keys=self._pinned_boundary_proofs,
                )
                if proof_identity in self._pinned_boundary_proofs:
                    self._pinned_segment_proofs.update(
                        proof.proof_identity for proof in outer_proofs
                    )
            audits.append(audit)
            records.append(record)
        return (
            BoundaryOptionSet(key, tuple(audits)),
            self._canonical_measurement_audit(
                _MeasurementAuditRecord(
                    measurement_identity=measurement_identity,
                    boundary_index=key.boundary_index,
                    outgoing_source_id=key.outgoing.source_id,
                    incoming_source_id=key.incoming.source_id,
                    offsets=tuple(records),
                )
            ),
        )

    def _bridge_transform_semantic_state(
        self,
        transform: BoundaryTransformState,
    ) -> tuple[object, ...]:
        return (
            transform.crop_xywh,
            float(transform.rotation_degrees),
            transform.track_id,
            transform.subject_basis,
        )

    def _canonical_bridge_transform(
        self,
        transform: BoundaryTransformState,
        *,
        side: str,
    ) -> BoundaryTransformState:
        semantic_state = self._bridge_transform_semantic_state(transform)
        canonical = BoundaryTransformState(
            transform_identity=(
                "joint-semantic-bridge-transform:"
                + _sha256_json(
                    {
                        "contract": "joint-semantic-bridge-transform-v1",
                        "side": side,
                        "renderer_semantic_state": semantic_state,
                    }
                )
            ),
            crop_xywh=transform.crop_xywh,
            rotation_degrees=transform.rotation_degrees,
            track_id=transform.track_id,
            subject_basis=transform.subject_basis,
        )
        return canonical

    def _bridge_semantic_key(
        self,
        *,
        candidate: TimelineCandidate,
        entry_transform: BoundaryTransformState,
        exit_transform: BoundaryTransformState,
        start_master_seconds: float,
        end_master_seconds: float,
        duration_frames: float,
    ) -> tuple[object, ...]:
        return (
            self.analysis_basis,
            self._configuration_identity,
            candidate,
            float(start_master_seconds),
            float(end_master_seconds),
            float(duration_frames),
            self._bridge_transform_semantic_state(entry_transform),
            self._bridge_transform_semantic_state(exit_transform),
        )

    def _scalar_bridge_request_sha256(
        self,
        request: SegmentBridgeDispositionRequest,
    ) -> str:
        """Bind a persisted scalar to every input that can change its gates."""

        candidate = request.candidate
        entry_option = request.entry_option
        exit_option = request.exit_option
        start_master_seconds = (
            candidate.start_seconds
            + entry_option.frame_offset / self.output_fps
        )
        end_master_seconds = (
            candidate.end_seconds
            + exit_option.frame_offset / self.output_fps
        )

        def renderer_state(
            transform: BoundaryTransformState,
        ) -> dict[str, object]:
            return {
                "crop_xywh": (
                    list(transform.crop_xywh)
                    if transform.crop_xywh is not None
                    else None
                ),
                "rotation_degrees": float(transform.rotation_degrees),
                "track_id": transform.track_id,
                "subject_basis": transform.subject_basis,
            }

        candidate_cache_key = id(candidate)
        cached_candidate = self._scalar_bridge_candidate_sha256_cache.get(
            candidate_cache_key
        )
        candidate_sha256 = (
            cached_candidate[1]
            if cached_candidate is not None
            and cached_candidate[0] is candidate
            else None
        )
        if candidate_sha256 is None:
            candidate_sha256 = _candidate_private_state_sha256(candidate)
            self._scalar_bridge_candidate_sha256_cache[candidate_cache_key] = (
                candidate,
                candidate_sha256,
            )
        return _sha256_json(
            {
                "contract": _SCALAR_BRIDGE_CACHE_CONTRACT,
                "provider_configuration_sha256": self._configuration_identity,
                "candidate_private_state_sha256": candidate_sha256,
                "source_index": self._source_index(candidate.source_id),
                "start_master_seconds": float(start_master_seconds),
                "end_master_seconds": float(end_master_seconds),
                "duration_frames": float(request.duration_frames),
                "entry_renderer_state": renderer_state(
                    entry_option.incoming_transform
                ),
                "exit_renderer_state": renderer_state(
                    exit_option.outgoing_transform
                ),
            }
        )

    @staticmethod
    def _decode_persisted_scalar_bridge_entry(
        value: object,
    ) -> tuple[bool, bool, bool, tuple[str, ...], float] | None:
        if not isinstance(value, list) or len(value) != 5:
            return None
        evidence_complete, crop_compatible, track_compatible, raw_codes, score = (
            value
        )
        if not all(
            type(item) is bool
            for item in (
                evidence_complete,
                crop_compatible,
                track_compatible,
            )
        ):
            return None
        # Incomplete evidence can be caused by a transient descriptor/model
        # failure.  It must be recomputed on the next run instead of becoming
        # a durable rejection in the direct scalar cache.
        if not evidence_complete:
            return None
        if not isinstance(raw_codes, list) or any(
            not isinstance(item, str) or not item or item != item.strip()
            for item in raw_codes
        ):
            return None
        codes = tuple(raw_codes)
        if codes != tuple(sorted(set(codes))):
            return None
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or float(score) > 0.0
        ):
            return None
        authorized = bool(
            evidence_complete and crop_compatible and track_compatible
        )
        if (authorized and codes) or (not authorized and not codes):
            return None
        return (
            evidence_complete,
            crop_compatible,
            track_compatible,
            codes,
            float(score),
        )

    def _scalar_bridge_analyzer_base(self) -> dict[str, object]:
        cached = self._scalar_bridge_persistent_analyzer_base
        if cached is not None:
            return dict(cached)
        source_sha256s: list[str] = []
        analyzer_sha256s: list[str] = []
        for source_index in range(len(self.sources)):
            sampler = self._sampler(source_index)
            persistent = sampler.persistent_feature_cache
            if persistent is None:  # pragma: no cover - required invariant
                raise PersistentSourceScalarCacheStorageUnavailableError(
                    "required direct source scalar cache storage is unavailable"
                )
            source_sha256s.append(str(persistent.source_sha256))
            analyzer_sha256s.append(str(persistent.analyzer.digest))
        appearance_model = sface_model_asset_status()
        cached = {
            "analysis_contract": _SCALAR_BRIDGE_CACHE_CONTRACT,
            "provider_configuration_sha256": self._configuration_identity,
            "source_content_sha256s": source_sha256s,
            "visual_analyzer_sha256s": analyzer_sha256s,
            "maximum_entries_per_segment_shard": (
                _SCALAR_BRIDGE_MAXIMUM_ENTRIES_PER_SHARD
            ),
            "implementation_sha256": {
                "provider_class_ast": python_symbol_dependency_sha256(
                    Path(__file__),
                    root_symbols=("DecodedBoundaryOptionProvider",),
                ),
                "face_vector_ast": python_symbol_dependency_sha256(
                    Path(__file__).with_name("face_identity.py"),
                    root_symbols=(
                        "extract_sface_embedding",
                        "cosine_similarity",
                    ),
                ),
                "timewarp_map_ast": python_symbol_dependency_sha256(
                    Path(__file__).with_name("timewarp.py"),
                    root_symbols=("TimeWarpMap",),
                ),
                "editorial_transform_track_ast": (
                    python_symbol_dependency_sha256(
                        Path(__file__).with_name("render.py"),
                        root_symbols=(
                            "EditorialTransformKeyframe",
                            "EditorialTransformTrack",
                        ),
                    )
                ),
            },
            "appearance_model_asset": {
                "name": appearance_model.name,
                "expected_sha256": appearance_model.expected_sha256,
                "actual_sha256": appearance_model.actual_sha256,
                "present": appearance_model.present,
                "digest_valid": appearance_model.digest_valid,
            },
        }
        self._scalar_bridge_persistent_analyzer_base = cached
        return dict(cached)

    def _decode_scalar_bridge_shard_payload(
        self,
        payload: Mapping[str, object] | None,
        *,
        segment_index: int,
    ) -> dict[str, _ScalarBridgeEntry]:
        if payload is None:
            return {}
        if (
            payload.get("contract") != _SCALAR_BRIDGE_CACHE_CONTRACT
            or payload.get("provider_configuration_sha256")
            != self._configuration_identity
            or payload.get("segment_index") != segment_index
            or not isinstance(payload.get("entries"), dict)
        ):
            self.scalar_bridge_persistent_cache_invalid_entry_count += 1
            return {}
        entries = payload["entries"]
        assert isinstance(entries, dict)
        decoded_entries: dict[str, _ScalarBridgeEntry] = {}
        for raw_key, raw_value in entries.items():
            if (
                not isinstance(raw_key, str)
                or len(raw_key) != 64
                or any(character not in "0123456789abcdef" for character in raw_key)
            ):
                self.scalar_bridge_persistent_cache_invalid_entry_count += 1
                continue
            decoded = self._decode_persisted_scalar_bridge_entry(raw_value)
            if decoded is None:
                self.scalar_bridge_persistent_cache_invalid_entry_count += 1
                continue
            decoded_entries[raw_key] = decoded
        return decoded_entries

    def _load_scalar_bridge_persistent_cache(self, segment_index: int) -> None:
        """Lazily load one bounded, path-free middle-segment scalar shard."""

        if segment_index in self._scalar_bridge_persistent_cache_loaded_segments:
            self._assert_scalar_bridge_sources_unchanged()
            self._scalar_bridge_persistent_caches[
                segment_index
            ].assert_storage_root_available()
            return
        if not (
            self.require_persistent_feature_cache
            and self.feature_derived_cache_directory is not None
        ):
            return
        if not 1 <= segment_index < self.total_segment_count - 1:
            raise ValueError("scalar bridge shard segment index is invalid")
        self._assert_scalar_bridge_sources_unchanged()
        direct_root = self.feature_derived_cache_directory
        assert direct_root is not None
        cache = PersistentSourceScalarCache(
            source_cache_directory=direct_root,
            source_path=self.sources[0],
            namespace=_SCALAR_BRIDGE_CACHE_NAMESPACE,
            analyzer={
                **self._scalar_bridge_analyzer_base(),
                "segment_index": segment_index,
            },
            derived_cache_directory=direct_root,
        )
        payload = cache.get()
        self._scalar_bridge_persistent_caches[segment_index] = cache
        self._scalar_bridge_persistent_cache_loaded_segments.add(segment_index)
        self._scalar_bridge_persistent_entries_by_segment[segment_index] = (
            self._decode_scalar_bridge_shard_payload(
                payload,
                segment_index=segment_index,
            )
        )

    def _assert_scalar_bridge_sources_unchanged(self) -> None:
        current = tuple(
            _source_runtime_stat_identity(source) for source in self.sources
        )
        if current != self._source_runtime_stat_identities:
            raise RuntimeError(
                "joint scalar bridge source changed during selection"
            )

    def _commit_scalar_bridge_persistent_cache(
        self,
        segment_index: int,
        new_entries: Mapping[str, _ScalarBridgeEntry],
    ) -> None:
        cache = self._scalar_bridge_persistent_caches.get(segment_index)
        if cache is None:
            return
        self._assert_scalar_bridge_sources_unchanged()
        cache.assert_storage_root_available()
        latest_entries = self._decode_scalar_bridge_shard_payload(
            cache.get(),
            segment_index=segment_index,
        )
        merged_entries = {
            **latest_entries,
            **self._scalar_bridge_persistent_entries_by_segment.get(
                segment_index,
                {},
            ),
            **new_entries,
        }
        bounded_entries = {
            key: merged_entries[key]
            for key in sorted(merged_entries)[
                :_SCALAR_BRIDGE_MAXIMUM_ENTRIES_PER_SHARD
            ]
        }
        entries = {
            key: [
                evidence_complete,
                crop_compatible,
                track_compatible,
                list(failure_codes),
                score_adjustment,
            ]
            for key, (
                evidence_complete,
                crop_compatible,
                track_compatible,
                failure_codes,
                score_adjustment,
            ) in sorted(bounded_entries.items())
        }
        if not cache.put(
            {
                "contract": _SCALAR_BRIDGE_CACHE_CONTRACT,
                "provider_configuration_sha256": self._configuration_identity,
                "segment_index": segment_index,
                "entries": entries,
                "media_artifacts_persisted": False,
                "biometric_model_vectors_persisted": False,
                "local_paths_persisted": False,
            }
        ):
            self._assert_scalar_bridge_sources_unchanged()
            cache.assert_storage_root_available()
            raise PersistentScalarBridgeCacheWriteError(
                "required persistent scalar bridge cache write failed"
            )
        self._scalar_bridge_persistent_entries_by_segment[segment_index] = (
            bounded_entries
        )
        self.scalar_bridge_persistent_cache_write_count += 1

    def _transform_path_scalar_disposition(
        self,
        *,
        candidate: TimelineCandidate,
        entry_transform: BoundaryTransformState,
        exit_transform: BoundaryTransformState,
        start_master_seconds: float,
        end_master_seconds: float,
        duration_frames: float,
        descriptor_resolver: Callable[
            [int, float], FrameVisualDescriptor | None
        ],
        face_identity_resolver: Callable[
            [int, float, FrameVisualDescriptor], tuple[float, ...]
        ],
    ) -> tuple[bool, bool, bool, tuple[str, ...], float]:
        """Evaluate the exact bridge gates without constructing proof state.

        This is the scalar-only counterpart of ``_transform_path_evidence``.
        Gate order, endpoint clocks, half-frame centres, crop coverage, and
        dominant-face continuity thresholds intentionally match that exact
        oracle line for line.  The resolvers are lazy so an early rejection
        never causes later frames to be decoded.  This path does not create a
        proof identity, audit record, transform-path sample, or materialization.
        """

        expected_duration_frames = (
            end_master_seconds - start_master_seconds
        ) * self.output_fps
        if not math.isclose(
            expected_duration_frames,
            duration_frames,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                "segment bridge duration disagrees with endpoint clocks"
            )
        source_index = self._source_index(candidate.source_id)

        def outcome(
            *,
            evidence_complete: bool,
            crop_compatible: bool,
            track_compatible: bool,
            failure_codes: Sequence[str],
        ) -> tuple[bool, bool, bool, tuple[str, ...], float]:
            return (
                evidence_complete,
                crop_compatible,
                track_compatible,
                tuple(sorted(set(failure_codes))),
                0.0,
            )

        if duration_frames < _MINIMUM_SEGMENT_FRAMES - _EPSILON:
            return outcome(
                evidence_complete=True,
                crop_compatible=False,
                track_compatible=True,
                failure_codes=("segment_duration_below_two_frames",),
            )

        expected_frame_center_count = int(
            math.ceil(
                max(
                    0.0,
                    duration_frames - 0.5 - _EPSILON,
                )
            )
        )
        frame_centers = tuple(
            start_master_seconds + (index + 0.5) / self.output_fps
            for index in range(expected_frame_center_count)
            if start_master_seconds + (index + 0.5) / self.output_fps
            < end_master_seconds - _EPSILON
        )
        sample_times = (
            start_master_seconds,
            *frame_centers,
            end_master_seconds,
        )

        if entry_transform.track_id is None or exit_transform.track_id is None:
            return outcome(
                evidence_complete=False,
                crop_compatible=False,
                track_compatible=False,
                failure_codes=("segment_track_identity_evidence_incomplete",),
            )
        if entry_transform.track_id != exit_transform.track_id:
            return outcome(
                evidence_complete=True,
                crop_compatible=False,
                track_compatible=False,
                failure_codes=("segment_track_identity_mismatch",),
            )
        if (
            abs(entry_transform.rotation_degrees) > _EPSILON
            or abs(exit_transform.rotation_degrees) > _EPSILON
        ):
            return outcome(
                evidence_complete=False,
                crop_compatible=False,
                track_compatible=True,
                failure_codes=("segment_rotation_path_evidence_incomplete",),
            )
        try:
            editorial_track = self._renderer_track(
                candidate=candidate,
                entry_transform=entry_transform,
                exit_transform=exit_transform,
                duration_seconds=duration_frames / self.output_fps,
            )
        except ValueError as exc:
            failure = (
                "segment_too_short_for_exact_boundary_transform_holds"
                if "too short" in str(exc)
                else "segment_renderer_transform_envelope_violation"
            )
            return outcome(
                evidence_complete=True,
                crop_compatible=False,
                track_compatible=True,
                failure_codes=(failure,),
            )

        bases = {entry_transform.subject_basis, exit_transform.subject_basis}
        if "dominant_face" in bases:
            subject_basis = "dominant_face"
        elif "subject_silhouette" in bases:
            subject_basis = "subject_silhouette"
        else:
            subject_basis = "full_frame_identity"
        track_geometry_required = bool(
            entry_transform.crop_xywh is not None
            or exit_transform.crop_xywh is not None
        )
        source_endpoint_times = (
            self._source_time(source_index, start_master_seconds),
            self._source_time(source_index, end_master_seconds),
        )
        if min(source_endpoint_times) < 0.0:
            return outcome(
                evidence_complete=True,
                crop_compatible=False,
                track_compatible=True,
                failure_codes=("segment_negative_source_local_time",),
            )
        if track_geometry_required:
            source_low = min(source_endpoint_times)
            source_high = max(source_endpoint_times)
            if any(
                source_low + _EPSILON < cut < source_high - _EPSILON
                for cut in self.scene_cut_times[source_index]
            ):
                return outcome(
                    evidence_complete=True,
                    crop_compatible=False,
                    track_compatible=False,
                    failure_codes=("segment_source_scene_cut_breaks_track",),
                )

        anchor_face_identity: tuple[float, ...] = ()
        previous_face_identity: tuple[float, ...] = ()
        for master_seconds in sample_times:
            source_seconds = self._source_time(source_index, master_seconds)
            renderer_keyframe = editorial_track.evaluate(
                max(0.0, master_seconds - start_master_seconds)
            )
            sample_crop = self._renderer_crop(renderer_keyframe)
            if source_seconds < 0.0:
                return outcome(
                    evidence_complete=True,
                    crop_compatible=False,
                    track_compatible=True,
                    failure_codes=("segment_negative_source_local_time",),
                )
            descriptor = descriptor_resolver(source_index, source_seconds)
            if descriptor is None:
                return outcome(
                    evidence_complete=False,
                    crop_compatible=False,
                    track_compatible=True,
                    failure_codes=("segment_descriptor_evidence_incomplete",),
                )
            if not track_geometry_required:
                bbox = None
            elif subject_basis == "dominant_face":
                bbox = descriptor.primary_face_bbox
            elif subject_basis == "subject_silhouette":
                bbox = descriptor.silhouette_bbox_xywh
            else:
                bbox = descriptor.silhouette_bbox_xywh
            if track_geometry_required and bbox is None:
                return outcome(
                    evidence_complete=False,
                    crop_compatible=False,
                    track_compatible=True,
                    failure_codes=("segment_subject_track_evidence_incomplete",),
                )
            crop = sample_crop or (0.0, 0.0, 1.0, 1.0)
            if (
                crop[0] < -_EPSILON
                or crop[1] < -_EPSILON
                or crop[2] <= 0.0
                or crop[3] <= 0.0
                or crop[0] + crop[2] > 1.0 + _EPSILON
                or crop[1] + crop[3] > 1.0 + _EPSILON
            ):
                return outcome(
                    evidence_complete=True,
                    crop_compatible=False,
                    track_compatible=True,
                    failure_codes=("segment_crop_path_out_of_frame",),
                )
            coverage = (
                self._bbox_crop_coverage(bbox, crop)
                if bbox is not None
                else None
            )
            if coverage is not None and coverage < 0.95 - _EPSILON:
                return outcome(
                    evidence_complete=True,
                    crop_compatible=False,
                    track_compatible=True,
                    failure_codes=("segment_subject_exits_crop_track",),
                )
            face_identity: tuple[float, ...] = ()
            similarity_to_anchor: float | None = None
            similarity_to_previous: float | None = None
            if subject_basis == "dominant_face":
                face_identity = face_identity_resolver(
                    source_index,
                    source_seconds,
                    descriptor,
                )
                if face_identity:
                    if not anchor_face_identity:
                        anchor_face_identity = face_identity
                    if not previous_face_identity:
                        previous_face_identity = face_identity
                    similarity_to_anchor = cosine_similarity(
                        anchor_face_identity,
                        face_identity,
                    )
                    similarity_to_previous = cosine_similarity(
                        previous_face_identity,
                        face_identity,
                    )
            if subject_basis == "dominant_face" and (
                not face_identity
                or similarity_to_anchor is None
                or similarity_to_previous is None
            ):
                return outcome(
                    evidence_complete=False,
                    crop_compatible=True,
                    track_compatible=False,
                    failure_codes=(
                        "segment_dominant_face_identity_evidence_incomplete",
                    ),
                )
            if subject_basis == "dominant_face" and (
                similarity_to_anchor
                < _MINIMUM_DOMINANT_FACE_TRACK_SIMILARITY
                or similarity_to_previous
                < _MINIMUM_DOMINANT_FACE_TRACK_SIMILARITY
            ):
                return outcome(
                    evidence_complete=True,
                    crop_compatible=True,
                    track_compatible=False,
                    failure_codes=(
                        "segment_dominant_face_track_identity_mismatch",
                    ),
                )
            if face_identity:
                previous_face_identity = face_identity
        return outcome(
            evidence_complete=True,
            crop_compatible=True,
            track_compatible=True,
            failure_codes=(),
        )

    def _validate_bridge_disposition_request(
        self,
        request: SegmentBridgeDispositionRequest,
    ) -> None:
        if type(request) is not SegmentBridgeDispositionRequest:
            raise TypeError("bridge disposition batch request is invalid")
        candidate = request.candidate
        entry_option = request.entry_option
        exit_option = request.exit_option
        if entry_option.measurement_key.incoming != candidate:
            raise ValueError("entry option does not enter bridge candidate")
        if exit_option.measurement_key.outgoing != candidate:
            raise ValueError("exit option does not leave bridge candidate")
        entry_boundary = entry_option.measurement_key.boundary_index
        exit_boundary = exit_option.measurement_key.boundary_index
        if exit_boundary != entry_boundary + 1:
            raise ValueError("bridge options must be adjacent boundaries")
        if (
            candidate.segment_index != entry_boundary + 1
            or exit_boundary != candidate.segment_index
            or not 0 <= entry_boundary < self.total_segment_count - 2
        ):
            raise ValueError(
                "bridge candidate and boundary identities are not in this provider"
            )
        self._source_index(candidate.source_id)
        for option in (entry_option, exit_option):
            for endpoint in (
                option.measurement_key.outgoing,
                option.measurement_key.incoming,
            ):
                self._source_index(endpoint.source_id)
        start_master_seconds = (
            candidate.start_seconds
            + entry_option.frame_offset / self.output_fps
        )
        end_master_seconds = (
            candidate.end_seconds
            + exit_option.frame_offset / self.output_fps
        )
        expected_duration = (
            end_master_seconds - start_master_seconds
        ) * self.output_fps
        if not math.isclose(
            request.duration_frames,
            expected_duration,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "bridge duration identity disagrees with exact endpoint clocks"
            )

    def bridge_disposition_batch(
        self,
        requests: tuple[SegmentBridgeDispositionRequest, ...],
    ) -> tuple[SegmentBridgeDisposition, ...]:
        """Return scalar bridge dispositions in one rollback-safe transaction."""

        self._require_open()
        if type(requests) is not tuple:
            raise TypeError("bridge disposition batch requests must be a tuple")
        if not requests:
            raise ValueError("bridge disposition batch cannot be empty")
        if self._active_transaction_journal is not None:
            raise RuntimeError("bridge disposition batch cannot be re-entered")
        for request in requests:
            self._validate_bridge_disposition_request(request)
        self._assert_required_persistent_cache_available(
            *(
                self._source_index(request.candidate.source_id)
                for request in requests
            )
        )
        for segment_index in dict.fromkeys(
            request.candidate.segment_index for request in requests
        ):
            self._load_scalar_bridge_persistent_cache(segment_index)
        journal: list[
            tuple[MutableMapping[Any, Any], Any, bool, Any]
        ] = []
        descriptor_cache_transaction = self.descriptor_cache.begin_transaction()
        counter_names = (
            "request_count",
            "measurement_count",
            "measurement_failure_count",
            "memo_hit_count",
            "bridge_request_count",
            "bridge_measurement_count",
            "bridge_measurement_failure_count",
            "bridge_memo_hit_count",
            "bridge_alias_registration_count",
            "bridge_semantic_memo_hit_count",
            "selection_request_count",
            "selection_materialization_count",
            "selection_failure_count",
            "descriptor_decode_count",
            "raw_descriptor_cache_hit_count",
            "raw_descriptor_cache_miss_count",
            "transformed_descriptor_decode_count",
            "transformed_descriptor_cache_hit_count",
            "face_identity_model_call_count",
            "face_identity_cache_hit_count",
            "boundary_materialization_rehydration_count",
            "bridge_materialization_rehydration_count",
            "scalar_bridge_batch_call_count",
            "scalar_bridge_batch_request_count",
            "scalar_bridge_disposition_count",
            "scalar_bridge_descriptor_snapshot_hit_count",
            "scalar_bridge_descriptor_snapshot_miss_count",
            "scalar_bridge_face_identity_snapshot_hit_count",
            "scalar_bridge_face_identity_snapshot_miss_count",
        )
        counter_snapshot = {
            name: getattr(self, name) for name in counter_names
        }
        audit_snapshot = (
            self._measurement_audit_digest,
            self._measurement_audit_total_count,
            dict(self._measurement_offset_disposition_counts),
            dict(self._measurement_failure_code_counts),
            self._same_source_measurement_count,
            self._cross_source_measurement_count,
            self._bridge_audit_digest,
            self._bridge_audit_registration_count,
            dict(self._bridge_failure_code_counts),
            self._outer_audit_digest,
            self._outer_audit_registration_count,
            dict(self._outer_failure_code_counts),
            dict(self._cache_eviction_counts),
            dict(self._cache_peak_counts),
        )
        sampler_indices_before = frozenset(self._samplers)
        descriptor_snapshot: dict[
            tuple[int, float],
            tuple[float, FrameVisualDescriptor | None],
        ] = {}
        face_identity_snapshot: dict[
            tuple[int, float],
            tuple[float, tuple[float, ...]],
        ] = {}
        descriptor_snapshot_hits = 0
        descriptor_snapshot_misses = 0
        face_snapshot_hits = 0
        face_snapshot_misses = 0

        def resolve_descriptor(
            source_index: int,
            seconds: float,
        ) -> FrameVisualDescriptor | None:
            nonlocal descriptor_snapshot_hits, descriptor_snapshot_misses
            self._assert_required_persistent_cache_available(source_index)
            key = (
                self._cache_source_indices[source_index],
                round(float(seconds), 7),
            )
            if key in descriptor_snapshot:
                descriptor_snapshot_hits += 1
                return descriptor_snapshot[key][1]
            descriptor_snapshot_misses += 1
            descriptor = self._sample_descriptor(source_index, seconds)
            descriptor_snapshot[key] = (float(seconds), descriptor)
            return descriptor

        def resolve_face_identity(
            source_index: int,
            seconds: float,
            descriptor: FrameVisualDescriptor,
        ) -> tuple[float, ...]:
            nonlocal face_snapshot_hits, face_snapshot_misses
            self._assert_required_persistent_cache_available(source_index)
            key = (source_index, round(float(seconds), 7))
            if key in face_identity_snapshot:
                face_snapshot_hits += 1
                return face_identity_snapshot[key][1]
            face_snapshot_misses += 1
            identity = self._sample_face_identity(
                source_index,
                seconds,
                descriptor,
            )
            face_identity_snapshot[key] = (float(seconds), identity)
            return identity

        self._active_transaction_journal = journal
        self._active_descriptor_cache_transaction = descriptor_cache_transaction
        try:
            dispositions: list[SegmentBridgeDisposition] = []
            new_persistent_entries_by_segment: dict[
                int, dict[str, _ScalarBridgeEntry]
            ] = {}
            for request in requests:
                segment_index = request.candidate.segment_index
                persistent_cache = self._scalar_bridge_persistent_caches.get(
                    segment_index
                )
                persistent_entries = (
                    self._scalar_bridge_persistent_entries_by_segment.get(
                        segment_index,
                        {},
                    )
                )
                persistent_key = (
                    self._scalar_bridge_request_sha256(request)
                    if persistent_cache is not None
                    else None
                )
                persisted = (
                    persistent_entries.get(persistent_key)
                    if persistent_key is not None
                    else None
                )
                if persisted is not None:
                    self.scalar_bridge_persistent_cache_hit_count += 1
                    (
                        evidence_complete,
                        crop_compatible,
                        track_compatible,
                        failure_codes,
                        score_adjustment,
                    ) = persisted
                    dispositions.append(
                        SegmentBridgeDisposition(
                            request=request,
                            evidence_complete=evidence_complete,
                            crop_compatible=crop_compatible,
                            track_compatible=track_compatible,
                            failure_codes=failure_codes,
                            score_adjustment=score_adjustment,
                        )
                    )
                    continue
                if persistent_cache is not None:
                    self.scalar_bridge_persistent_cache_miss_count += 1
                candidate = request.candidate
                entry_option = request.entry_option
                exit_option = request.exit_option
                start_master_seconds = (
                    candidate.start_seconds
                    + entry_option.frame_offset / self.output_fps
                )
                end_master_seconds = (
                    candidate.end_seconds
                    + exit_option.frame_offset / self.output_fps
                )
                (
                    evidence_complete,
                    crop_compatible,
                    track_compatible,
                    failure_codes,
                    score_adjustment,
                ) = self._transform_path_scalar_disposition(
                    candidate=candidate,
                    entry_transform=entry_option.incoming_transform,
                    exit_transform=exit_option.outgoing_transform,
                    start_master_seconds=start_master_seconds,
                    end_master_seconds=end_master_seconds,
                    duration_frames=request.duration_frames,
                    descriptor_resolver=resolve_descriptor,
                    face_identity_resolver=resolve_face_identity,
                )
                if persistent_key is not None and evidence_complete:
                    new_persistent_entries_by_segment.setdefault(
                        segment_index,
                        {},
                    )[persistent_key] = (
                        evidence_complete,
                        crop_compatible,
                        track_compatible,
                        failure_codes,
                        score_adjustment,
                    )
                dispositions.append(
                    SegmentBridgeDisposition(
                        request=request,
                        evidence_complete=evidence_complete,
                        crop_compatible=crop_compatible,
                        track_compatible=track_compatible,
                        failure_codes=failure_codes,
                        score_adjustment=score_adjustment,
                    )
                )
            result = tuple(dispositions)
            if len(result) != len(requests) or any(
                disposition.request is not request
                for disposition, request in zip(result, requests, strict=True)
            ):
                raise RuntimeError(
                    "bridge disposition batch lost exact request identities"
                )
            for segment_index, new_persistent_entries in sorted(
                new_persistent_entries_by_segment.items()
            ):
                self._commit_scalar_bridge_persistent_cache(
                    segment_index,
                    new_persistent_entries,
                )
            self.descriptor_cache.commit_transaction(
                descriptor_cache_transaction
            )
        except BaseException:
            self._rollback_transaction(journal)
            if descriptor_cache_transaction.active:
                self.descriptor_cache.rollback_transaction(
                    descriptor_cache_transaction
                )
            for source_index in tuple(self._samplers):
                if source_index in sampler_indices_before:
                    continue
                self._samplers.pop(source_index).close()
            for name, value in counter_snapshot.items():
                setattr(self, name, value)
            (
                self._measurement_audit_digest,
                self._measurement_audit_total_count,
                measurement_disposition_counts,
                measurement_failure_counts,
                self._same_source_measurement_count,
                self._cross_source_measurement_count,
                self._bridge_audit_digest,
                self._bridge_audit_registration_count,
                bridge_failure_counts,
                self._outer_audit_digest,
                self._outer_audit_registration_count,
                outer_failure_counts,
                cache_eviction_counts,
                cache_peak_counts,
            ) = audit_snapshot
            self._measurement_offset_disposition_counts.clear()
            self._measurement_offset_disposition_counts.update(
                measurement_disposition_counts
            )
            self._measurement_failure_code_counts.clear()
            self._measurement_failure_code_counts.update(
                measurement_failure_counts
            )
            self._bridge_failure_code_counts.clear()
            self._bridge_failure_code_counts.update(bridge_failure_counts)
            self._outer_failure_code_counts.clear()
            self._outer_failure_code_counts.update(outer_failure_counts)
            self._cache_eviction_counts.clear()
            self._cache_eviction_counts.update(cache_eviction_counts)
            self._cache_peak_counts.clear()
            self._cache_peak_counts.update(cache_peak_counts)
            self._active_transaction_journal = None
            self._active_descriptor_cache_transaction = None
            raise
        self._active_transaction_journal = None
        self._active_descriptor_cache_transaction = None
        self.scalar_bridge_batch_call_count += 1
        self.scalar_bridge_batch_request_count += len(requests)
        self.scalar_bridge_disposition_count += len(result)
        self.scalar_bridge_descriptor_snapshot_hit_count += (
            descriptor_snapshot_hits
        )
        self.scalar_bridge_descriptor_snapshot_miss_count += (
            descriptor_snapshot_misses
        )
        self.scalar_bridge_face_identity_snapshot_hit_count += face_snapshot_hits
        self.scalar_bridge_face_identity_snapshot_miss_count += (
            face_snapshot_misses
        )
        self._enforce_transient_cache_limits()
        return result

    def bridge_provider(
        self,
        candidate: TimelineCandidate,
        entry_option: AuthorizedBoundaryOption,
        exit_option: AuthorizedBoundaryOption,
        duration_frames: float,
    ) -> SegmentBridgeProof:
        self._require_open()
        if not isinstance(candidate, TimelineCandidate):
            raise TypeError("bridge candidate is invalid")
        if not isinstance(
            entry_option,
            AuthorizedBoundaryOption,
        ) or not isinstance(exit_option, AuthorizedBoundaryOption):
            raise TypeError("bridge options are invalid")
        duration = _finite(duration_frames, "bridge duration_frames")
        if entry_option.measurement_key.incoming != candidate:
            raise ValueError("entry option does not enter bridge candidate")
        if exit_option.measurement_key.outgoing != candidate:
            raise ValueError("exit option does not leave bridge candidate")
        self._assert_required_persistent_cache_available(
            self._source_index(candidate.source_id)
        )
        self.bridge_request_count += 1
        memo_key = (
            self.analysis_basis,
            self._configuration_identity,
            candidate,
            entry_option,
            exit_option,
            duration,
        )
        cached_exact = self._bridge_memo.get(memo_key)
        if cached_exact is not None:
            self.bridge_memo_hit_count += 1
            self._touch_lru(self._bridge_memo, memo_key)
            if (
                cached_exact.proof_identity in self._bridge_audit
                and (
                    cached_exact.proof_identity
                    in self._segment_materialization_by_proof
                    or cached_exact.proof_identity
                    not in self._pinned_segment_proofs
                )
            ):
                if (
                    cached_exact.proof_identity
                    in self._segment_materialization_by_proof
                ):
                    self._touch_lru(
                        self._segment_materialization_by_proof,
                        cached_exact.proof_identity,
                    )
                self._touch_lru(
                    self._bridge_audit,
                    cached_exact.proof_identity,
                )
                return cached_exact
        start_master_seconds = (
            candidate.start_seconds
            + entry_option.frame_offset / self.output_fps
        )
        end_master_seconds = (
            candidate.end_seconds
            + exit_option.frame_offset / self.output_fps
        )
        semantic_key = self._bridge_semantic_key(
            candidate=candidate,
            entry_transform=entry_option.incoming_transform,
            exit_transform=exit_option.outgoing_transform,
            start_master_seconds=start_master_seconds,
            end_master_seconds=end_master_seconds,
            duration_frames=duration,
        )
        semantic_context_identity = _sha256_json(
            {
                "contract": "joint-middle-segment-semantic-bridge-context-v1",
                "analysis_basis": self.analysis_basis,
                "configuration_identity": self._configuration_identity,
                "candidate_state_sha256": (
                    _candidate_private_state_sha256(candidate)
                ),
                "start_master_seconds": start_master_seconds,
                "end_master_seconds": end_master_seconds,
                "duration_frames": duration,
                "entry_renderer_state": list(
                    self._bridge_transform_semantic_state(
                        entry_option.incoming_transform
                    )
                ),
                "exit_renderer_state": list(
                    self._bridge_transform_semantic_state(
                        exit_option.outgoing_transform
                    )
                ),
            }
        )
        owns_transaction = self._active_transaction_journal is None
        if owns_transaction:
            journal: list[
                tuple[MutableMapping[Any, Any], Any, bool, Any]
            ] = []
            descriptor_cache_transaction = (
                self.descriptor_cache.begin_transaction()
            )
            counter_snapshot = (
                self.descriptor_decode_count,
                self.face_identity_model_call_count,
                self.raw_descriptor_cache_hit_count,
                self.raw_descriptor_cache_miss_count,
                self.face_identity_cache_hit_count,
                dict(self._cache_eviction_counts),
                dict(self._cache_peak_counts),
            )
            self._active_transaction_journal = journal
            self._active_descriptor_cache_transaction = (
                descriptor_cache_transaction
            )
        else:
            journal = self._active_transaction_journal
            descriptor_cache_transaction = (
                self._active_descriptor_cache_transaction
            )
            if descriptor_cache_transaction is None:
                raise RuntimeError(
                    "nested bridge transaction lost descriptor cache transaction"
                )
            counter_snapshot = None
        semantic_measured = False
        materialization_rehydrated = False
        try:
            semantic_entry_transform = self._canonical_bridge_transform(
                entry_option.incoming_transform,
                side="entry",
            )
            semantic_exit_transform = self._canonical_bridge_transform(
                exit_option.outgoing_transform,
                side="exit",
            )
            semantic = self._semantic_bridge_path_memo.get(semantic_key)
            semantic_materialization: (
                SegmentTransformPathMaterializationProof | None
            ) = None
            if semantic is None:
                path = self._transform_path_evidence(
                    candidate=candidate,
                    entry_transform=semantic_entry_transform,
                    exit_transform=semantic_exit_transform,
                    start_master_seconds=start_master_seconds,
                    end_master_seconds=end_master_seconds,
                    duration_frames=duration,
                    context_identity=semantic_context_identity,
                )
                semantic_materialization = (
                    self._segment_materialization_by_proof[path.proof_identity]
                )
                semantic = _SemanticBridgeMeasurement(
                    evidence=path,
                )
                self._bounded_transactional_set(
                    self._semantic_bridge_path_memo,
                    semantic_key,
                    semantic,
                    maximum_entries=_SEMANTIC_BRIDGE_PATH_LIMIT,
                    cache_name="semantic_bridge_path",
                )
                semantic_measured = True
            else:
                self.bridge_semantic_memo_hit_count += 1
                self._touch_lru(self._semantic_bridge_path_memo, semantic_key)
            path = semantic.evidence
            alias_identity = "joint-segment-bridge-alias:" + _sha256_json(
                {
                    "contract": "joint-segment-bridge-exact-alias-v1",
                    "semantic_path_proof_identity": path.proof_identity,
                    "candidate_state_sha256": (
                        _candidate_private_state_sha256(candidate)
                    ),
                    "entry_proof_identity": entry_option.proof_identity,
                    "exit_proof_identity": exit_option.proof_identity,
                    "duration_frames": duration,
                }
            )
            proof = SegmentBridgeProof(
                candidate=candidate,
                entry_option=entry_option,
                exit_option=exit_option,
                duration_frames=duration,
                proof_identity=alias_identity,
                evidence_complete=path.evidence_complete,
                crop_compatible=path.crop_compatible,
                track_compatible=path.track_compatible,
                failure_codes=path.failure_codes,
            )
            if cached_exact is not None and cached_exact != proof:
                raise RuntimeError(
                    "rehydrated bridge alias diverges from exact memo proof"
                )
            if (
                semantic_materialization is None
                and alias_identity in self._pinned_segment_proofs
            ):
                rehydrated_path = self._transform_path_evidence(
                    candidate=candidate,
                    entry_transform=semantic_entry_transform,
                    exit_transform=semantic_exit_transform,
                    start_master_seconds=start_master_seconds,
                    end_master_seconds=end_master_seconds,
                    duration_frames=duration,
                    context_identity=semantic_context_identity,
                )
                if rehydrated_path != path:
                    raise RuntimeError(
                        "selected bridge semantic evidence changed during rehydration"
                    )
                semantic_materialization = (
                    self._segment_materialization_by_proof[
                        rehydrated_path.proof_identity
                    ]
                )
                materialization_rehydrated = True
            if semantic_materialization is not None:
                alias_materialization = replace(
                    semantic_materialization,
                    proof_identity=alias_identity,
                    entry_transform=entry_option.incoming_transform,
                    exit_transform=exit_option.outgoing_transform,
                )
                self._bounded_transactional_set(
                    self._segment_materialization_by_proof,
                    alias_identity,
                    alias_materialization,
                    maximum_entries=_SEGMENT_MATERIALIZATION_LIMIT,
                    cache_name="segment_materialization",
                    pinned_keys=self._pinned_segment_proofs,
                )
                if path.proof_identity != alias_identity:
                    self._transactional_delete(
                        self._segment_materialization_by_proof,
                        path.proof_identity,
                    )
            self._bounded_transactional_set(
                self._bridge_memo,
                memo_key,
                proof,
                maximum_entries=_BRIDGE_MEMO_LIMIT,
                cache_name="bridge_memo",
            )
            self._bounded_transactional_set(
                self._bridge_audit,
                alias_identity,
                _BridgeAuditRecord(
                    proof_identity=alias_identity,
                    candidate_id=candidate.candidate_id,
                    entry_proof_identity=entry_option.proof_identity,
                    exit_proof_identity=exit_option.proof_identity,
                    duration_frames=duration,
                    expected_sample_count=path.expected_sample_count,
                    sampled_frame_count=path.sampled_frame_count,
                    endpoints_included=path.endpoints_included,
                    full_frame_centers_included=(
                        path.full_frame_centers_included
                    ),
                    evidence_complete=path.evidence_complete,
                    crop_compatible=path.crop_compatible,
                    track_compatible=path.track_compatible,
                    failure_codes=path.failure_codes,
                    semantic_path_proof_identity=path.proof_identity,
                ),
                maximum_entries=_TRANSIENT_AUDIT_LIMIT,
                cache_name="bridge_audit",
            )
            if owns_transaction:
                self.descriptor_cache.commit_transaction(
                    descriptor_cache_transaction
                )
        except BaseException:
            if owns_transaction:
                self._rollback_transaction(journal)
                if descriptor_cache_transaction.active:
                    self.descriptor_cache.rollback_transaction(
                        descriptor_cache_transaction
                    )
                assert counter_snapshot is not None
                (
                    self.descriptor_decode_count,
                    self.face_identity_model_call_count,
                    self.raw_descriptor_cache_hit_count,
                    self.raw_descriptor_cache_miss_count,
                    self.face_identity_cache_hit_count,
                    eviction_counts,
                    peak_counts,
                ) = counter_snapshot
                self._cache_eviction_counts.update(eviction_counts)
                self._cache_peak_counts.update(peak_counts)
                self._active_transaction_journal = None
                self._active_descriptor_cache_transaction = None
                self.bridge_measurement_failure_count += 1
            raise
        if owns_transaction:
            self._active_transaction_journal = None
            self._active_descriptor_cache_transaction = None
        alias_audit = self._bridge_audit[proof.proof_identity]
        self._bridge_audit_registration_count += 1
        for code in alias_audit.failure_codes:
            self._bridge_failure_code_counts[code] = (
                self._bridge_failure_code_counts.get(code, 0) + 1
            )
        self._bridge_audit_digest = _sha256_json(
            {
                "contract": "joint-bridge-audit-chain-v1",
                "previous": self._bridge_audit_digest,
                "proof_identity": proof.proof_identity,
                "semantic_path_proof_identity": (
                    alias_audit.semantic_path_proof_identity
                ),
                "failure_codes": list(alias_audit.failure_codes),
            }
        )
        if semantic_measured:
            self.bridge_measurement_count += 1
        if cached_exact is None:
            self.bridge_alias_registration_count += 1
        if materialization_rehydrated:
            self.bridge_materialization_rehydration_count += 1
        self._enforce_transient_cache_limits()
        return proof

    def bridge_audit_for(self, proof_identity: str) -> dict[str, object]:
        self._require_open()
        identity = _exact_identifier(proof_identity, "bridge proof_identity")
        try:
            record = self._bridge_audit[identity]
        except KeyError as exc:
            raise KeyError("bridge proof has not been measured") from exc
        return record.to_dict()

    def _ensure_boundary_materialization(
        self,
        option: AuthorizedBoundaryOption,
    ) -> BoundaryMaterializationProof:
        existing = self._materialization_by_proof.get(option.proof_identity)
        if existing is not None and all(
            identity in self._segment_materialization_by_proof
            for identity in existing.outer_track_proof_identities
        ):
            self._touch_lru(
                self._materialization_by_proof,
                option.proof_identity,
            )
            self._pinned_boundary_proofs.add(option.proof_identity)
            self._pinned_segment_proofs.update(
                existing.outer_track_proof_identities
            )
            return existing

        self._pinned_boundary_proofs.add(option.proof_identity)
        remeasured, _audit = self._measure(option.measurement_key)
        remeasured_audit = remeasured.audit_for(option.frame_offset)
        if remeasured_audit.authorized_option != option:
            raise ValueError(
                "selected boundary proof cannot be deterministically rehydrated"
            )
        try:
            materialization = self._materialization_by_proof[
                option.proof_identity
            ]
        except KeyError as exc:
            raise ValueError(
                "selected boundary remeasurement omitted exact materialization"
            ) from exc
        if any(
            identity not in self._segment_materialization_by_proof
            for identity in materialization.outer_track_proof_identities
        ):
            raise ValueError(
                "selected boundary remeasurement omitted an outer path"
            )
        self._pinned_segment_proofs.update(
            materialization.outer_track_proof_identities
        )
        self.boundary_materialization_rehydration_count += 1
        return materialization

    def materialization_proof(
        self,
        proof_identity: str,
    ) -> BoundaryMaterializationProof:
        self._require_open()
        identity = _exact_identifier(
            proof_identity,
            "boundary proof_identity",
        )
        try:
            materialization = self._materialization_by_proof[identity]
        except KeyError as exc:
            raise KeyError(
                "boundary proof is not pinned; pass its AuthorizedBoundaryOption "
                "through selection_materialization_proof for exact rehydration"
            ) from exc
        self._touch_lru(self._materialization_by_proof, identity)
        return materialization

    def segment_materialization_proof(
        self,
        proof_identity: str,
    ) -> SegmentTransformPathMaterializationProof:
        self._require_open()
        identity = _exact_identifier(
            proof_identity,
            "segment path proof_identity",
        )
        try:
            materialization = self._segment_materialization_by_proof[identity]
        except KeyError as exc:
            raise KeyError(
                "segment path is not pinned; pass its exact proof through "
                "selection_materialization_proof for rehydration"
            ) from exc
        self._touch_lru(self._segment_materialization_by_proof, identity)
        return materialization

    def _validate_boundary_renderer_samples(
        self,
        *,
        boundary: BoundaryMaterializationProof,
        outgoing_path: SegmentTransformPathMaterializationProof,
        incoming_path: SegmentTransformPathMaterializationProof,
    ) -> None:
        frame_seconds = 1.0 / self.output_fps

        def validate(
            *,
            path: SegmentTransformPathMaterializationProof,
            master_seconds: float,
            source_seconds: float,
            expected_transform: BoundaryTransformState,
            raw_descriptor: FrameVisualDescriptor,
            result_descriptor: FrameVisualDescriptor,
        ) -> None:
            track = path.editorial_transform_track
            if track is None:
                raise ValueError("selection segment renderer track is unavailable")
            matches = tuple(
                sample
                for sample in path.samples
                if math.isclose(
                    sample.master_seconds,
                    master_seconds,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )
            if len(matches) != 1:
                raise ValueError(
                    "selection renderer path omits an exact boundary sample"
                )
            sample = matches[0]
            evaluated = track.evaluate(
                master_seconds - path.start_master_seconds
            )
            evaluated_crop = self._renderer_crop(evaluated)
            if (
                sample.descriptor is None
                or not math.isclose(
                    sample.source_seconds,
                    source_seconds,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or sample.descriptor != raw_descriptor
                or not self._crops_equal(
                    sample.transform.crop_xywh,
                    evaluated_crop,
                )
                or not self._crops_equal(
                    evaluated_crop,
                    expected_transform.crop_xywh,
                )
                or not math.isclose(
                    evaluated.roll_degrees,
                    expected_transform.rotation_degrees,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or sample.transform.track_id != expected_transform.track_id
            ):
                raise ValueError(
                    "boundary evidence diverges from its renderer transform track"
                )
            actual_result = (
                sample.descriptor
                if evaluated_crop is None
                and abs(evaluated.roll_degrees) <= _EPSILON
                else self._sample_transformed_descriptor(
                    path.source_index,
                    sample.source_seconds,
                    sample.descriptor,
                    sample.transform,
                )
            )
            if actual_result is None or actual_result != result_descriptor:
                raise ValueError(
                    "boundary result descriptor diverges from renderer path decode"
                )

        if (
            outgoing_path.source_index != boundary.outgoing_source_index
            or incoming_path.source_index != boundary.incoming_source_index
        ):
            raise ValueError("boundary source indices diverge from segment paths")
        validate(
            path=outgoing_path,
            master_seconds=boundary.boundary_master_seconds
            - frame_seconds / 2.0,
            source_seconds=boundary.outgoing_exact_sample_seconds,
            expected_transform=boundary.outgoing_transform,
            raw_descriptor=boundary.raw_outgoing_descriptor,
            result_descriptor=boundary.result_outgoing_descriptor,
        )
        validate(
            path=outgoing_path,
            master_seconds=boundary.boundary_master_seconds
            - (boundary.side_sample_radius_frames + 0.5) * frame_seconds,
            source_seconds=boundary.outgoing_preceding_sample_seconds,
            expected_transform=boundary.outgoing_transform,
            raw_descriptor=boundary.raw_outgoing_preceding_descriptor,
            result_descriptor=(
                boundary.result_outgoing_preceding_descriptor
            ),
        )
        validate(
            path=incoming_path,
            master_seconds=boundary.boundary_master_seconds
            + frame_seconds / 2.0,
            source_seconds=boundary.incoming_exact_sample_seconds,
            expected_transform=boundary.incoming_transform,
            raw_descriptor=boundary.raw_incoming_descriptor,
            result_descriptor=boundary.result_incoming_descriptor,
        )
        validate(
            path=incoming_path,
            master_seconds=boundary.boundary_master_seconds
            + (boundary.side_sample_radius_frames + 0.5) * frame_seconds,
            source_seconds=boundary.incoming_following_sample_seconds,
            expected_transform=boundary.incoming_transform,
            raw_descriptor=boundary.raw_incoming_following_descriptor,
            result_descriptor=(
                boundary.result_incoming_following_descriptor
            ),
        )

    def selection_materialization_proof(
        self,
        boundary_options: Sequence[AuthorizedBoundaryOption],
        segment_bridge_proofs: Sequence[SegmentBridgeProof] = (),
    ) -> JointSelectionMaterializationProof:
        self._require_open()
        self.selection_request_count += 1
        if self._active_transaction_journal is not None:
            raise RuntimeError("selection materialization cannot be re-entered")
        journal: list[
            tuple[MutableMapping[Any, Any], Any, bool, Any]
        ] = []
        counter_snapshot = (
            self.descriptor_decode_count,
            self.transformed_descriptor_decode_count,
            self.face_identity_model_call_count,
            self.raw_descriptor_cache_hit_count,
            self.raw_descriptor_cache_miss_count,
            self.transformed_descriptor_cache_hit_count,
            self.face_identity_cache_hit_count,
            self.bridge_request_count,
            self.bridge_measurement_count,
            self.bridge_measurement_failure_count,
            self.bridge_memo_hit_count,
            self.bridge_alias_registration_count,
            self.bridge_semantic_memo_hit_count,
            self.boundary_materialization_rehydration_count,
            self.bridge_materialization_rehydration_count,
            self._bridge_audit_registration_count,
            self._bridge_audit_digest,
            dict(self._bridge_failure_code_counts),
            dict(self._cache_eviction_counts),
            dict(self._cache_peak_counts),
            self._outer_audit_registration_count,
            self._outer_audit_digest,
            dict(self._outer_failure_code_counts),
        )
        previous_boundary_pins = set(self._pinned_boundary_proofs)
        previous_segment_pins = set(self._pinned_segment_proofs)
        self._pinned_boundary_proofs.clear()
        self._pinned_segment_proofs.clear()
        self._active_transaction_journal = journal
        descriptor_cache_transaction = self.descriptor_cache.begin_transaction()
        self._active_descriptor_cache_transaction = descriptor_cache_transaction
        try:
            result = self._selection_materialization_proof(
                boundary_options,
                segment_bridge_proofs,
            )
            self.descriptor_cache.commit_transaction(
                descriptor_cache_transaction
            )
        except BaseException:
            self._rollback_transaction(journal)
            if descriptor_cache_transaction.active:
                self.descriptor_cache.rollback_transaction(
                    descriptor_cache_transaction
                )
            (
                self.descriptor_decode_count,
                self.transformed_descriptor_decode_count,
                self.face_identity_model_call_count,
                self.raw_descriptor_cache_hit_count,
                self.raw_descriptor_cache_miss_count,
                self.transformed_descriptor_cache_hit_count,
                self.face_identity_cache_hit_count,
                self.bridge_request_count,
                self.bridge_measurement_count,
                self.bridge_measurement_failure_count,
                self.bridge_memo_hit_count,
                self.bridge_alias_registration_count,
                self.bridge_semantic_memo_hit_count,
                self.boundary_materialization_rehydration_count,
                self.bridge_materialization_rehydration_count,
                self._bridge_audit_registration_count,
                self._bridge_audit_digest,
                bridge_failure_code_counts,
                eviction_counts,
                peak_counts,
                self._outer_audit_registration_count,
                self._outer_audit_digest,
                outer_failure_code_counts,
            ) = counter_snapshot
            self._bridge_failure_code_counts.clear()
            self._bridge_failure_code_counts.update(
                bridge_failure_code_counts
            )
            self._cache_eviction_counts.update(eviction_counts)
            self._cache_peak_counts.update(peak_counts)
            self._outer_failure_code_counts.clear()
            self._outer_failure_code_counts.update(
                outer_failure_code_counts
            )
            self._active_transaction_journal = None
            self._active_descriptor_cache_transaction = None
            self._pinned_boundary_proofs.clear()
            self._pinned_boundary_proofs.update(previous_boundary_pins)
            self._pinned_segment_proofs.clear()
            self._pinned_segment_proofs.update(previous_segment_pins)
            self._enforce_transient_cache_limits()
            self.selection_failure_count += 1
            raise
        self._active_transaction_journal = None
        self._active_descriptor_cache_transaction = None
        # Keep only the successful selection pinned.  These full decoded
        # materials are the bounded (<= total segment count) renderer handoff;
        # all non-selected DP materials are evictable.
        self._enforce_transient_cache_limits()
        self._last_selection_identity = result.selection_identity
        self._last_selected_boundary_proof_identities = (
            result.boundary_proof_identities
        )
        self._last_selected_segment_proof_identities = (
            result.segment_path_proof_identities
        )
        self._last_selected_boundary_records = tuple(
            {
                "proof_identity": item.proof_identity,
                "raw_proof_identity": item.raw_proof_identity,
                "result_proof_identity": item.result_proof_identity,
                "boundary_index": item.measurement_key.boundary_index,
                "frame_offset": item.frame_offset,
                "outgoing_transform": _transform_payload(
                    item.outgoing_transform
                ),
                "incoming_transform": _transform_payload(
                    item.incoming_transform
                ),
                "outer_track_proof_identities": list(
                    item.outer_track_proof_identities
                ),
            }
            for item in result.boundaries
        )
        self._last_selected_segment_records = tuple(
            {
                "proof_identity": item.proof_identity,
                "candidate": _candidate_public_identity(item.candidate),
                "entry_transform": _transform_payload(item.entry_transform),
                "exit_transform": _transform_payload(item.exit_transform),
                "start_master_seconds": item.start_master_seconds,
                "end_master_seconds": item.end_master_seconds,
                "duration_frames": item.duration_frames,
                "expected_sample_count": item.expected_sample_count,
                "endpoints_included": item.endpoints_included,
                "full_frame_centers_included": (
                    item.full_frame_centers_included
                ),
                "evidence_complete": item.evidence_complete,
                "crop_compatible": item.crop_compatible,
                "track_compatible": item.track_compatible,
                "failure_codes": list(item.failure_codes),
                "editorial_transform_track": (
                    {
                        "reason": item.editorial_transform_track.reason,
                        "keyframes": [
                            {
                                "time_seconds": keyframe.time_seconds,
                                "zoom": keyframe.zoom,
                                "center_x": keyframe.center_x,
                                "center_y": keyframe.center_y,
                                "roll_degrees": keyframe.roll_degrees,
                            }
                            for keyframe in (
                                item.editorial_transform_track.keyframes
                            )
                        ],
                    }
                    if item.editorial_transform_track is not None
                    else None
                ),
                "raw_frames_persisted": False,
                "biometric_model_vectors_persisted": False,
            }
            for item in result.segment_paths
        )
        self.selection_materialization_count += 1
        return result

    def _selection_materialization_proof(
        self,
        boundary_options: Sequence[AuthorizedBoundaryOption],
        segment_bridge_proofs: Sequence[SegmentBridgeProof] = (),
    ) -> JointSelectionMaterializationProof:
        if len(boundary_options) != self.total_segment_count - 1:
            raise ValueError(
                "selection needs exactly one boundary option per boundary"
            )
        if any(
            not isinstance(option, AuthorizedBoundaryOption)
            for option in boundary_options
        ):
            raise TypeError("selection contains an invalid boundary option")
        ordered_options = tuple(
            sorted(
                boundary_options,
                key=lambda option: option.measurement_key.boundary_index,
            )
        )
        if tuple(
            option.measurement_key.boundary_index for option in ordered_options
        ) != tuple(range(self.total_segment_count - 1)):
            raise ValueError("selection boundary indices are not exact and contiguous")
        boundary_materializations: list[BoundaryMaterializationProof] = []
        path_identities: list[str] = []
        for option in ordered_options:
            materialization = self._ensure_boundary_materialization(option)
            if (
                materialization.measurement_key != option.measurement_key
                or materialization.frame_offset != option.frame_offset
                or materialization.outgoing_transform
                != option.outgoing_transform
                or materialization.incoming_transform
                != option.incoming_transform
            ):
                raise ValueError(
                    "selection boundary option disagrees with decoded proof"
                )
            boundary_materializations.append(materialization)
            path_identities.extend(
                materialization.outer_track_proof_identities
            )

        expected_bridge_count = max(0, self.total_segment_count - 2)
        if len(segment_bridge_proofs) != expected_bridge_count:
            raise ValueError(
                "selection needs exactly one middle-segment bridge proof"
                " per middle segment"
            )
        if any(
            not isinstance(proof, SegmentBridgeProof)
            for proof in segment_bridge_proofs
        ):
            raise TypeError("selection contains an invalid segment bridge proof")
        ordered_bridges = tuple(
            sorted(
                segment_bridge_proofs,
                key=lambda proof: proof.candidate.segment_index,
            )
        )
        if tuple(proof.candidate.segment_index for proof in ordered_bridges) != tuple(
            range(1, self.total_segment_count - 1)
        ):
            raise ValueError("selection bridge segment indices are not exact")
        selected_bridge_audits: list[Mapping[str, object]] = []
        for proof in ordered_bridges:
            if not proof.authorized:
                raise ValueError("selection cannot materialize an unauthorized bridge")
            segment_index = proof.candidate.segment_index
            selected_entry = ordered_options[segment_index - 1]
            selected_exit = ordered_options[segment_index]
            if (
                proof.entry_option != selected_entry
                or proof.exit_option != selected_exit
            ):
                raise ValueError(
                    "selection bridge endpoints do not match selected boundaries"
                )
            # Mark the exact alias before callback re-entry.  The bridge
            # provider keeps only scalar semantic evidence during DP and
            # materializes full decoded samples only for this selected alias.
            self._pinned_segment_proofs.add(proof.proof_identity)
            rehydrated = self.bridge_provider(
                proof.candidate,
                selected_entry,
                selected_exit,
                proof.duration_frames,
            )
            if rehydrated != proof:
                raise ValueError(
                    "selection bridge cannot be deterministically rehydrated"
                )
            try:
                bridge_audit = self._bridge_audit[proof.proof_identity]
            except KeyError as exc:
                raise ValueError(
                    "selection bridge was not measured by this provider"
                ) from exc
            if (
                bridge_audit.entry_proof_identity
                != selected_entry.proof_identity
                or bridge_audit.exit_proof_identity
                != selected_exit.proof_identity
            ):
                raise ValueError(
                    "selection bridge audit endpoints do not match selection"
                )
            selected_bridge_audits.append(
                MappingProxyType(bridge_audit.to_dict())
            )
            path = self.segment_materialization_proof(proof.proof_identity)
            expected_start = (
                proof.candidate.start_seconds
                + selected_entry.frame_offset / self.output_fps
            )
            expected_end = (
                proof.candidate.end_seconds
                + selected_exit.frame_offset / self.output_fps
            )
            expected_duration = (
                expected_end - expected_start
            ) * self.output_fps
            if (
                path.candidate != proof.candidate
                or path.entry_transform
                != selected_entry.incoming_transform
                or path.exit_transform
                != selected_exit.outgoing_transform
                or not math.isclose(
                    path.start_master_seconds,
                    expected_start,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    path.end_master_seconds,
                    expected_end,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    proof.duration_frames,
                    expected_duration,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                or not math.isclose(
                    path.duration_frames,
                    expected_duration,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            ):
                raise ValueError(
                    "selection bridge materialization clocks or transforms diverge"
                )
            path_identities.append(proof.proof_identity)

        unique_path_identities = tuple(dict.fromkeys(path_identities))
        paths = tuple(
            sorted(
                (
                    self.segment_materialization_proof(identity)
                    for identity in unique_path_identities
                ),
                key=lambda item: item.candidate.segment_index,
            )
        )
        if tuple(item.candidate.segment_index for item in paths) != tuple(
            range(self.total_segment_count)
        ):
            raise ValueError(
                "selection does not carry one exact transform track per segment"
            )
        if any(
            not item.evidence_complete
            or not item.crop_compatible
            or not item.track_compatible
            or not item.endpoints_included
            or not item.full_frame_centers_included
            or item.editorial_transform_track is None
            or len(item.samples) != item.expected_sample_count
            for item in paths
        ):
            raise ValueError(
                "selection contains an incomplete or incompatible segment path"
            )
        path_by_segment = {
            item.candidate.segment_index: item for item in paths
        }
        for boundary in boundary_materializations:
            boundary_index = boundary.measurement_key.boundary_index
            self._validate_boundary_renderer_samples(
                boundary=boundary,
                outgoing_path=path_by_segment[boundary_index],
                incoming_path=path_by_segment[boundary_index + 1],
            )
        boundary_identities = tuple(
            item.proof_identity for item in boundary_materializations
        )
        ordered_path_identities = tuple(item.proof_identity for item in paths)
        selection_identity = "joint-selection-materialization:" + _sha256_json(
            {
                "contract": "joint-selection-materialization-proof-v1",
                "analysis_basis": self.analysis_basis,
                "configuration_identity": self._configuration_identity,
                "boundary_proof_identities": list(boundary_identities),
                "segment_path_proof_identities": list(
                    ordered_path_identities
                ),
            }
        )
        return JointSelectionMaterializationProof(
            selection_identity=selection_identity,
            boundary_proof_identities=boundary_identities,
            segment_path_proof_identities=ordered_path_identities,
            boundaries=tuple(boundary_materializations),
            segment_paths=paths,
            selected_bridge_audits=tuple(selected_bridge_audits),
        )

    def audit_for(self, key: BoundaryMeasurementKey) -> dict[str, object]:
        self._require_open()
        memo_key = self._memo_key(key)
        try:
            record = self._audit_by_key[memo_key]
        except KeyError as exc:
            raise KeyError("boundary measurement has not been requested") from exc
        self._touch_lru(self._audit_by_key, memo_key)
        return record.to_dict()

    def _persistent_feature_cache_audit(self) -> dict[str, object]:
        """Summarize sampler cache use without exposing media/cache paths."""

        enabled_sampler_count = 0
        source_frame_decode_count = 0
        decode_free_hit_count = 0
        status_counts: dict[str, int] = {}
        for sampler in self._samplers.values():
            diagnostics = sampler.persistent_feature_cache_diagnostics()
            enabled_sampler_count += int(bool(diagnostics.get("enabled")))
            status = str(diagnostics.get("status", "diagnostics_unavailable"))
            status_counts[status] = status_counts.get(status, 0) + 1
            source_frame_decode_count += int(
                diagnostics.get("source_frame_decode_count", 0)
            )
            decode_free_hit_count += int(
                diagnostics.get("decode_free_hit_count", 0)
            )
        required_source_count = (
            len(self.sources) if self.require_persistent_feature_cache else 0
        )
        return {
            "contract": "decoded-joint-persistent-feature-cache-audit-v1",
            "direct_cache_configured": (
                self.feature_derived_cache_directory is not None
            ),
            "required": self.require_persistent_feature_cache,
            "required_source_count": required_source_count,
            "initialized_sampler_count": len(self._samplers),
            "enabled_sampler_count": enabled_sampler_count,
            "all_required_sources_preflighted": (
                not self.require_persistent_feature_cache
                or (
                    len(self._samplers) == required_source_count
                    and enabled_sampler_count == required_source_count
                )
            ),
            "source_frame_decode_count": source_frame_decode_count,
            "decode_free_hit_count": decode_free_hit_count,
            "sampler_status_counts": dict(sorted(status_counts.items())),
            "paths_persisted": False,
        }

    def _persistent_scalar_bridge_cache_audit(self) -> dict[str, object]:
        storage_records = tuple(
            diagnostics
            for cache in self._scalar_bridge_persistent_caches.values()
            for diagnostics in (cache.diagnostics(),)
            if isinstance(diagnostics, Mapping)
        )
        storage = (
            {
                "shard_count": len(self._scalar_bridge_persistent_caches),
                "hit_count": sum(
                    int(item.get("hit_count", 0)) for item in storage_records
                ),
                "miss_count": sum(
                    int(item.get("miss_count", 0)) for item in storage_records
                ),
                "write_count": sum(
                    int(item.get("write_count", 0)) for item in storage_records
                ),
                "invalid_record_count": sum(
                    int(item.get("invalid_record_count", 0))
                    for item in storage_records
                ),
                "storage_io_failure_count": sum(
                    int(item.get("storage_io_failure_count", 0))
                    for item in storage_records
                ),
                "required": bool(storage_records)
                and all(bool(item.get("required")) for item in storage_records),
                "local_paths_exposed": False,
                "raw_media_persisted": False,
            }
            if self._scalar_bridge_persistent_caches
            else None
        )
        return {
            "contract": _SCALAR_BRIDGE_CACHE_CONTRACT,
            "configured": bool(
                self.require_persistent_feature_cache
                and self.feature_derived_cache_directory is not None
            ),
            "required": bool(
                self.require_persistent_feature_cache
                and self.feature_derived_cache_directory is not None
            ),
            "loaded": bool(
                self._scalar_bridge_persistent_cache_loaded_segments
            ),
            "loaded_shard_count": len(
                self._scalar_bridge_persistent_cache_loaded_segments
            ),
            "entry_count": sum(
                len(entries)
                for entries in (
                    self._scalar_bridge_persistent_entries_by_segment.values()
                )
            ),
            "maximum_entries_per_shard": (
                _SCALAR_BRIDGE_MAXIMUM_ENTRIES_PER_SHARD
            ),
            "request_hit_count": (
                self.scalar_bridge_persistent_cache_hit_count
            ),
            "request_miss_count": (
                self.scalar_bridge_persistent_cache_miss_count
            ),
            "atomic_write_count": (
                self.scalar_bridge_persistent_cache_write_count
            ),
            "invalid_entry_count": (
                self.scalar_bridge_persistent_cache_invalid_entry_count
            ),
            "storage": storage,
            "source_content_bound": True,
            "provider_configuration_bound": True,
            "visual_analyzer_bound": True,
            "request_private_state_bound": True,
            "middle_segment_sharded": True,
            "legacy_unsharded_records_reused": False,
            "evidence_incomplete_results_persisted": False,
            "raw_frames_persisted": False,
            "biometric_model_vectors_persisted": False,
            "local_paths_persisted": False,
        }

    def to_dict(self) -> dict[str, object]:
        records = tuple(
            sorted(
                self._audit_by_key.values(),
                key=lambda item: (
                    item.boundary_index,
                    item.outgoing_source_id,
                    item.incoming_source_id,
                    item.measurement_identity,
                ),
            )
        )
        offset_audit_count = sum(
            self._measurement_offset_disposition_counts.values()
        )
        expected_offset_audit_count = (
            self.measurement_count * len(JOINT_BOUNDARY_FRAME_OFFSETS)
        )
        retained_bridge_records = tuple(
            self._bridge_audit[key] for key in sorted(self._bridge_audit)
        )
        retained_outer_records = tuple(
            self._outer_track_audit[key]
            for key in sorted(self._outer_track_audit)
        )
        return {
            "contract": "decoded-joint-boundary-option-provider-audit-v1",
            "closed": self._closed,
            "analysis_basis": self.analysis_basis,
            "configuration_identity": self._configuration_identity,
            "offset_domain": list(JOINT_BOUNDARY_FRAME_OFFSETS),
            "request_count": self.request_count,
            "measurement_count": self.measurement_count,
            "measurement_failure_count": self.measurement_failure_count,
            "memo_hit_count": self.memo_hit_count,
            "same_source_measurement_count": (
                self._same_source_measurement_count
            ),
            "cross_source_measurement_count": (
                self._cross_source_measurement_count
            ),
            "offset_audit_count": offset_audit_count,
            "expected_offset_audit_count": expected_offset_audit_count,
            "exact_offset_domain_complete": (
                offset_audit_count == expected_offset_audit_count
                and self._measurement_audit_total_count
                == self.measurement_count
            ),
            "offset_disposition_counts": dict(
                self._measurement_offset_disposition_counts
            ),
            "measurement_failure_code_counts": dict(
                sorted(self._measurement_failure_code_counts.items())
            ),
            "bridge_request_count": self.bridge_request_count,
            "bridge_measurement_count": self.bridge_measurement_count,
            "bridge_measurement_failure_count": (
                self.bridge_measurement_failure_count
            ),
            "bridge_memo_hit_count": self.bridge_memo_hit_count,
            "bridge_alias_registration_count": (
                self.bridge_alias_registration_count
            ),
            "bridge_semantic_memo_hit_count": (
                self.bridge_semantic_memo_hit_count
            ),
            "scalar_bridge_batch": {
                "contract": "decoded-bridge-scalar-batch-audit-v1",
                "batch_call_count": self.scalar_bridge_batch_call_count,
                "request_count": self.scalar_bridge_batch_request_count,
                "disposition_count": self.scalar_bridge_disposition_count,
                "descriptor_snapshot_hit_count": (
                    self.scalar_bridge_descriptor_snapshot_hit_count
                ),
                "descriptor_snapshot_miss_count": (
                    self.scalar_bridge_descriptor_snapshot_miss_count
                ),
                "face_identity_snapshot_hit_count": (
                    self.scalar_bridge_face_identity_snapshot_hit_count
                ),
                "face_identity_snapshot_miss_count": (
                    self.scalar_bridge_face_identity_snapshot_miss_count
                ),
                "persistent_scalar_cache": (
                    self._persistent_scalar_bridge_cache_audit()
                ),
                "scalar_evidence_only": True,
                "proof_identities_created": False,
                "exact_aliases_created": False,
                "audit_records_created": False,
                "sha_chains_created": False,
                "transform_path_samples_created": False,
                "materializations_created": False,
                "raw_frames_persisted": False,
                "biometric_model_vectors_persisted": False,
            },
            "boundary_materialization_rehydration_count": (
                self.boundary_materialization_rehydration_count
            ),
            "bridge_materialization_rehydration_count": (
                self.bridge_materialization_rehydration_count
            ),
            "selection_request_count": self.selection_request_count,
            "selection_materialization_count": (
                self.selection_materialization_count
            ),
            "selection_failure_count": self.selection_failure_count,
            "descriptor_decode_count": self.descriptor_decode_count,
            "raw_descriptor_cache_hit_count": (
                self.raw_descriptor_cache_hit_count
            ),
            "raw_descriptor_cache_miss_count": (
                self.raw_descriptor_cache_miss_count
            ),
            "transformed_descriptor_decode_count": (
                self.transformed_descriptor_decode_count
            ),
            "transformed_descriptor_cache_hit_count": (
                self.transformed_descriptor_cache_hit_count
            ),
            "face_identity_model_call_count": (
                self.face_identity_model_call_count
            ),
            "face_identity_cache_hit_count": (
                self.face_identity_cache_hit_count
            ),
            "persistent_feature_cache": (
                self._persistent_feature_cache_audit()
            ),
            "descriptor_cache": self.descriptor_cache.diagnostics(),
            "measurement_audits": [item.to_dict() for item in records],
            "measurement_audit_total_count": (
                self._measurement_audit_total_count
            ),
            "measurement_audit_retained_count": len(records),
            "measurement_audits_truncated": (
                self._measurement_audit_total_count > len(records)
            ),
            "measurement_audit_chain_sha256": self._measurement_audit_digest,
            "bridge_audits": [
                item.to_dict() for item in retained_bridge_records
            ],
            "bridge_audit_registration_count": (
                self._bridge_audit_registration_count
            ),
            "bridge_audit_retained_count": len(retained_bridge_records),
            "bridge_audits_truncated": (
                self._bridge_audit_registration_count
                > len(retained_bridge_records)
            ),
            "bridge_audit_chain_sha256": self._bridge_audit_digest,
            "bridge_failure_code_counts": dict(
                sorted(self._bridge_failure_code_counts.items())
            ),
            "outer_track_audits": [
                item.to_dict() for item in retained_outer_records
            ],
            "outer_track_audit_retained_count": len(retained_outer_records),
            "outer_track_audit_registration_count": (
                self._outer_audit_registration_count
            ),
            "outer_track_audits_truncated": (
                self._outer_audit_registration_count
                > len(retained_outer_records)
            ),
            "outer_track_audit_chain_sha256": self._outer_audit_digest,
            "outer_track_failure_code_counts": dict(
                sorted(self._outer_failure_code_counts.items())
            ),
            "last_selection": {
                "selection_identity": self._last_selection_identity,
                "boundary_proof_identities": list(
                    self._last_selected_boundary_proof_identities
                ),
                "segment_path_proof_identities": list(
                    self._last_selected_segment_proof_identities
                ),
                "boundary_records": [
                    dict(item)
                    for item in self._last_selected_boundary_records
                ],
                "segment_records": [
                    dict(item)
                    for item in self._last_selected_segment_records
                ],
                "records_are_selected_only": True,
                "raw_frames_persisted": False,
                "biometric_model_vectors_persisted": False,
            },
            "materialization_proof_count": len(
                self._materialization_by_proof
            ),
            "segment_materialization_proof_count": len(
                self._segment_materialization_by_proof
            ),
            "selected_boundary_materialization_pin_count": len(
                self._pinned_boundary_proofs
            ),
            "selected_segment_materialization_pin_count": len(
                self._pinned_segment_proofs
            ),
            "boundary_memo_entry_count": len(self._memo),
            "bridge_memo_entry_count": len(self._bridge_memo),
            "semantic_bridge_path_entry_count": len(
                self._semantic_bridge_path_memo
            ),
            "face_identity_cache_entry_count": len(
                self._face_identity_cache
            ),
            "transformed_descriptor_cache_entry_count": len(
                self._transformed_descriptor_cache
            ),
            "sampler_entry_count": len(self._samplers),
            "transform_subject_basis_entry_count": 0,
            "transient_cache_limits": {
                "boundary_memo": _BOUNDARY_MEMO_LIMIT,
                "bridge_memo": _BRIDGE_MEMO_LIMIT,
                "semantic_bridge_path": _SEMANTIC_BRIDGE_PATH_LIMIT,
                "face_identity": _FACE_IDENTITY_CACHE_LIMIT,
                "transformed_descriptor": (
                    _TRANSFORMED_DESCRIPTOR_CACHE_LIMIT
                ),
                "boundary_materialization": (
                    _BOUNDARY_MATERIALIZATION_LIMIT
                ),
                "segment_materialization": (
                    _SEGMENT_MATERIALIZATION_LIMIT
                ),
                "measurement_audit": _MEASUREMENT_AUDIT_LIMIT,
                "bridge_or_outer_audit": _TRANSIENT_AUDIT_LIMIT,
                "transform_subject_basis": 0,
            },
            "transient_cache_eviction_counts": dict(
                self._cache_eviction_counts
            ),
            "transient_cache_peak_counts": dict(self._cache_peak_counts),
            "audit_payload_bounded": True,
            "selected_pin_overflow_bounded_by_total_segment_count": True,
            "materialization_proofs_persisted": False,
            "raw_frames_persisted": False,
            "biometric_model_vectors_persisted": False,
            "persistent_descriptor_storage": False,
            "capacity_audit_at_eight_candidates": self.capacity_audit(8),
        }


__all__ = [
    "BoundaryMaterializationProof",
    "DecodedBoundaryOptionProvider",
    "DescriptorProvider",
    "FaceIdentityProvider",
    "JointSelectionMaterializationProof",
    "JOINT_BOUNDARY_PROVIDER_ANALYSIS_BASIS",
    "PersistentScalarBridgeCacheWriteError",
    "SegmentTransformPathMaterializationProof",
    "SegmentTransformPathSample",
    "TransformedDescriptorProvider",
]
