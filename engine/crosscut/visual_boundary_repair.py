"""Decode a tiny frame neighborhood and refine selected group-edit cuts.

Source selection remains fixed.  For every visible cross-source boundary this
pass evaluates at most seven output-frame positions (nominal +/- three), keeps
broadcaster camera cuts outside the visible handles, and minimizes formation,
pose, framing and structure discontinuity.  The output is a new immutable
``VisualTimelinePlan`` plus manifest-ready audit evidence.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Sequence

import cv2

from .boundary_repair import (
    BoundaryCandidateContext,
    BoundaryRepairConfig,
    BoundaryRepairRequest,
    BoundaryRepairResult,
    repair_boundary,
)
from .pose import POSE_BACKEND
from .timeline import (
    BoundarySideStability,
    ShotType,
    boundary_side_stability_exceeded,
    comparable_group_formation_mismatch,
    screen_layout_mismatch,
)
from .timeline_review import TimelineReview, TimelineReviewConfig, review_timeline
from .timewarp import TimeWarpMap
from .stage_edit_temporal_pose_motion import (
    TemporalPoseMotionBoundaryAnalysis,
    TemporalPoseMotionThresholds,
    TemporalPoseWindow,
    analyze_temporal_pose_motion_boundary,
)
from .visual_score import (
    FrameVisualDescriptor,
    VisualTimelineChoice,
    VisualTimelinePlan,
    _VideoSampler,
    describe_boundary_side_stability,
)
from .visual_splice_failure_feedback import (
    build_splice_failure_feedback,
    splice_feedback_context_identity,
    splice_feedback_edge_context,
    splice_feedback_edge_context_identity,
    splice_feedback_plan_context,
    splice_feedback_plan_identity,
    splice_feedback_result_context,
    splice_feedback_result_context_identity,
)


ANALYSIS_BASIS = (
    "decoded_plus_minus_three_frame_formation_layout_and_actual_pose_boundary_"
    "repair_v4+complete_success_offset_audit_v1+"
    "temporal_pose_motion_audit_v1"
)
_PRODUCTION_REVIEW_GATES = TimelineReviewConfig()
_NEAREST_TEMPORAL_POSE_RADIUS_FRAMES = 3


def _freeze_audit_value(value: object) -> object:
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("boundary repair audit keys must be strings")
            frozen[key] = _freeze_audit_value(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_audit_value(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(
        f"boundary repair audit value is not manifest-safe: {type(value)!r}"
    )


def _thaw_audit_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _thaw_audit_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_thaw_audit_value(item) for item in value]
    return value


def _actual_pose_available(descriptor: FrameVisualDescriptor | None) -> bool:
    """Return true only for a real pose-model vector, never the HOG fallback."""

    return bool(
        descriptor is not None
        and descriptor.pose_backend == POSE_BACKEND
        and descriptor.pose_embedding
    )


def _actual_temporal_pose_available(
    descriptor: FrameVisualDescriptor | None,
) -> bool:
    """Require the exact production pose contract used by the motion analyzer."""

    return bool(
        _actual_pose_available(descriptor)
        and len(descriptor.pose_embedding) == 60
    )


def _temporal_pose_motion_threshold_values(
    thresholds: TemporalPoseMotionThresholds,
) -> dict[str, float]:
    return {
        "minimum_boundary_motion_magnitude": float(
            thresholds.minimum_boundary_motion_magnitude
        ),
        "minimum_endpoint_pose_cosine": float(
            thresholds.minimum_endpoint_pose_cosine
        ),
        "minimum_velocity_direction_cosine": float(
            thresholds.minimum_velocity_direction_cosine
        ),
        "minimum_phase_compatibility": float(
            thresholds.minimum_phase_compatibility
        ),
        "pose_norm_epsilon": float(thresholds.pose_norm_epsilon),
    }


def _sha256_json(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _temporal_pose_motion_configuration(
    thresholds: TemporalPoseMotionThresholds | None,
    preference_weight: float,
) -> dict[str, Any]:
    if thresholds is not None:
        raise ValueError(
            "temporal pose-motion calibration activation is unavailable until "
            "an immutable evidence loader/registry verifies its thresholds"
        )
    threshold_values = (
        _temporal_pose_motion_threshold_values(thresholds)
        if thresholds is not None
        else None
    )
    threshold_identity = (
        _sha256_json(
            {
                "contract": "temporal_pose_motion_thresholds_v1",
                "values": threshold_values,
            }
        )
        if threshold_values is not None
        else None
    )
    fingerprint_payload = {
        "contract": "visual_boundary_temporal_pose_motion_preference_v1",
        "threshold_identity_sha256": threshold_identity,
        "preference_weight": float(preference_weight),
    }
    return {
        "contract": fingerprint_payload["contract"],
        "calibration_status": (
            "calibration_bound"
            if thresholds is not None
            else "calibration_unbound"
        ),
        "threshold_values": threshold_values,
        "threshold_identity_sha256": threshold_identity,
        "preference_weight": float(preference_weight),
        "configuration_fingerprint_sha256": _sha256_json(
            fingerprint_payload
        ),
    }


def _rms(left: Sequence[float], right: Sequence[float]) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    return min(
        1.0,
        math.sqrt(
            sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True))
            / len(left)
        ),
    )


def _bbox_mismatch(
    left: tuple[float, float, float, float] | None,
    right: tuple[float, float, float, float] | None,
    *,
    left_centroid: tuple[float, float] | None = None,
    right_centroid: tuple[float, float] | None = None,
) -> float | None:
    """Mirror the reviewer's persisted-box mismatch exactly.

    The repaired plan persists face/silhouette boxes rather than transient
    layout embeddings.  Using the same metric here and in ``timeline_review``
    prevents a locally attractive offset from passing this gate and then
    failing the whole-timeline review after serialization.
    """

    if left is None or right is None:
        return None
    left_center = left_centroid or (
        left[0] + left[2] / 2.0,
        left[1] + left[3] / 2.0,
    )
    right_center = right_centroid or (
        right[0] + right[2] / 2.0,
        right[1] + right[3] / 2.0,
    )
    center = min(
        1.0,
        math.hypot(
            left_center[0] - right_center[0],
            left_center[1] - right_center[1],
        )
        / 0.42,
    )
    left_area = max(left[2] * left[3], 1e-6)
    right_area = max(right[2] * right[3], 1e-6)
    scale = min(1.0, abs(math.log(left_area / right_area)) / math.log(3.2))
    width = min(1.0, abs(left[2] - right[2]) / 0.55)
    height = min(1.0, abs(left[3] - right[3]) / 0.55)
    return min(
        1.0,
        0.52 * center + 0.28 * scale + 0.10 * width + 0.10 * height,
    )


def _shot_type_mismatch(left: ShotType, right: ShotType) -> float:
    order = {
        ShotType.GROUP_WIDE: 0,
        ShotType.WIDE: 1,
        ShotType.MEDIUM: 2,
        ShotType.CLOSE_UP: 3,
    }
    return abs(order[left] - order[right]) / 3.0


@dataclass(frozen=True, slots=True)
class TemporalPoseMotionEvidence:
    """Scalar-only audit of temporal choreography continuity at one cut."""

    calibration_status: str
    analysis_executed: bool
    input_valid: bool
    qualifies: bool
    classification: str
    failure_reasons: tuple[str, ...]
    feature_dimension: int | None
    outgoing_sample_count: int
    incoming_sample_count: int
    outgoing_boundary_motion_magnitude: float | None
    incoming_boundary_motion_magnitude: float | None
    endpoint_pose_cosine: float | None
    velocity_direction_cosine: float | None
    phase_compatibility: float | None
    confidence: float
    static_pose_match: bool
    dynamic_pose_phase_match: bool
    direction_evidence_available: bool
    dynamic_direction_match_proven: bool
    soft_preference_applied: float
    threshold_values: dict[str, float] | None
    threshold_identity_sha256: str | None
    preference_weight: float
    configuration_fingerprint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": "scalar_temporal_pose_motion_boundary_evidence_v1",
            "calibration_status": self.calibration_status,
            "analysis_executed": self.analysis_executed,
            "input_valid": self.input_valid,
            "qualifies": self.qualifies,
            "classification": self.classification,
            "failure_reasons": list(self.failure_reasons),
            "feature_dimension": self.feature_dimension,
            "outgoing_sample_count": self.outgoing_sample_count,
            "incoming_sample_count": self.incoming_sample_count,
            "outgoing_boundary_motion_magnitude": (
                self.outgoing_boundary_motion_magnitude
            ),
            "incoming_boundary_motion_magnitude": (
                self.incoming_boundary_motion_magnitude
            ),
            "endpoint_pose_cosine": self.endpoint_pose_cosine,
            "velocity_direction_cosine": self.velocity_direction_cosine,
            "phase_compatibility": self.phase_compatibility,
            "confidence": self.confidence,
            "static_pose_match": self.static_pose_match,
            "dynamic_pose_phase_match": self.dynamic_pose_phase_match,
            "direction_evidence_available": self.direction_evidence_available,
            "dynamic_direction_match_proven": (
                self.dynamic_direction_match_proven
            ),
            "soft_preference_applied": self.soft_preference_applied,
            "threshold_values": (
                dict(self.threshold_values)
                if self.threshold_values is not None
                else None
            ),
            "threshold_identity_sha256": self.threshold_identity_sha256,
            "preference_weight": self.preference_weight,
            "configuration_fingerprint_sha256": (
                self.configuration_fingerprint_sha256
            ),
        }


@dataclass(frozen=True, slots=True)
class BoundaryEvidenceCoverage:
    """Availability mask for the non-identifying evidence at one exact cut.

    Temporal pose availability includes an exact sample at offset zero or the
    nearest stable sample *inside* the adjacent shot.  The exact and temporal
    masks stay separate so an audit can never mistake a nearby pose for pose
    observed on the actual cut frame.
    """

    exact_pose_available: tuple[bool, bool]
    temporal_pose_available: tuple[bool, bool]
    temporal_pose_offsets_frames: tuple[int | None, int | None]
    formation_person_counts: tuple[int, int]
    formation_upper_body_counts: tuple[int, int]
    person_layout_available: tuple[bool, bool]
    silhouette_available: tuple[bool, bool]
    professional_evidence_basis: str
    professional_gate_passed: bool
    missing_mask: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_pose_available": list(self.exact_pose_available),
            "temporal_pose_available": list(self.temporal_pose_available),
            "temporal_pose_offsets_frames": list(
                self.temporal_pose_offsets_frames
            ),
            "formation_person_counts": list(self.formation_person_counts),
            "formation_upper_body_counts": list(
                self.formation_upper_body_counts
            ),
            "person_layout_available": list(self.person_layout_available),
            "silhouette_available": list(self.silhouette_available),
            "professional_evidence_basis": self.professional_evidence_basis,
            "professional_gate_passed": self.professional_gate_passed,
            "missing_mask": list(self.missing_mask),
        }


@dataclass(frozen=True, slots=True)
class BoundaryVisualEvidence:
    total_cost: float
    base_total_cost: float
    formation_mismatch: float | None
    pose_mismatch: float | None
    pose_basis: str
    exact_pose_mismatch: float | None
    nearest_temporal_pose_mismatch: float | None
    silhouette_mismatch: float | None
    layout_mismatch: float | None
    persisted_layout_mismatch: float | None
    focus_mismatch: float | None
    structure_mismatch: float | None
    shot_scale_delta: float
    shot_type_mismatch: float
    temporal_pose_motion: TemporalPoseMotionEvidence
    evidence_coverage: BoundaryEvidenceCoverage
    outgoing_side_stability: BoundarySideStability | None = None
    incoming_side_stability: BoundarySideStability | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cost": self.total_cost,
            "base_total_cost": self.base_total_cost,
            "formation_mismatch": self.formation_mismatch,
            "pose_mismatch": self.pose_mismatch,
            "pose_basis": self.pose_basis,
            "exact_pose_mismatch": self.exact_pose_mismatch,
            "nearest_temporal_pose_mismatch": (
                self.nearest_temporal_pose_mismatch
            ),
            "silhouette_mismatch": self.silhouette_mismatch,
            "layout_mismatch": self.layout_mismatch,
            "persisted_layout_mismatch": self.persisted_layout_mismatch,
            "focus_mismatch": self.focus_mismatch,
            "structure_mismatch": self.structure_mismatch,
            "shot_scale_delta": self.shot_scale_delta,
            "shot_type_mismatch": self.shot_type_mismatch,
            "temporal_pose_motion": self.temporal_pose_motion.to_dict(),
            "evidence_coverage": self.evidence_coverage.to_dict(),
            "outgoing_side_stability": (
                self.outgoing_side_stability.to_dict()
                if self.outgoing_side_stability is not None
                else None
            ),
            "incoming_side_stability": (
                self.incoming_side_stability.to_dict()
                if self.incoming_side_stability is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class VisualBoundaryRepair:
    boundary_index: int
    outgoing_source_index: int
    incoming_source_index: int
    result: BoundaryRepairResult
    selected_visual_evidence: BoundaryVisualEvidence
    candidate_evidence_audit: tuple[Mapping[str, Any], ...]
    optimizer_edge_context: Mapping[str, Any]
    optimizer_edge_context_identity: str
    observed_edge_context: Mapping[str, Any]
    observed_edge_context_identity: str
    result_edge_context: Mapping[str, Any]
    result_edge_context_identity: str
    optimizer_edge_context_match: bool

    def __post_init__(self) -> None:
        frozen_candidate_audit = tuple(
            _freeze_audit_value(item)
            for item in self.candidate_evidence_audit
        )
        if not frozen_candidate_audit or any(
            not isinstance(item, Mapping)
            for item in frozen_candidate_audit
        ):
            raise TypeError(
                "successful boundary repair needs per-offset audit mappings"
            )
        audit_offsets = tuple(
            item.get("frame_offset") for item in frozen_candidate_audit
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in audit_offsets
        ):
            raise ValueError("boundary repair audit offsets are invalid")
        if (
            len(set(audit_offsets)) != len(audit_offsets)
            or audit_offsets != tuple(sorted(audit_offsets))
            or audit_offsets
            != tuple(range(audit_offsets[0], audit_offsets[-1] + 1))
            or audit_offsets[0] != -audit_offsets[-1]
            or audit_offsets[-1] > 8
        ):
            raise ValueError(
                "boundary repair audit must cover one complete symmetric "
                "requested offset domain"
            )
        selected_audit = tuple(
            item
            for item in frozen_candidate_audit
            if item.get("frame_offset") == self.result.selected_frame_offset
        )
        if len(selected_audit) != 1:
            raise ValueError(
                "boundary repair audit lost its selected frame offset"
            )
        if selected_audit[0].get("production_transition_gate_passed") is not True:
            raise ValueError(
                "selected boundary repair offset did not pass production gates"
            )
        selected_visual_payload = selected_audit[0].get("visual_evidence")
        if (
            not isinstance(selected_visual_payload, Mapping)
            or _thaw_audit_value(selected_visual_payload)
            != self.selected_visual_evidence.to_dict()
        ):
            raise ValueError(
                "selected boundary repair evidence disagrees with its audit"
            )
        object.__setattr__(
            self,
            "candidate_evidence_audit",
            frozen_candidate_audit,
        )
        optimizer_context = _freeze_audit_value(
            self.optimizer_edge_context
        )
        observed_context = _freeze_audit_value(self.observed_edge_context)
        result_context = _freeze_audit_value(self.result_edge_context)
        if not isinstance(optimizer_context, Mapping) or not isinstance(
            observed_context,
            Mapping,
        ) or not isinstance(result_context, Mapping):
            raise TypeError("boundary repair edge contexts must be mappings")
        object.__setattr__(
            self,
            "optimizer_edge_context",
            optimizer_context,
        )
        object.__setattr__(self, "observed_edge_context", observed_context)
        object.__setattr__(self, "result_edge_context", result_context)
        if (
            splice_feedback_context_identity(optimizer_context)
            != self.optimizer_edge_context_identity
        ):
            raise ValueError("boundary repair optimizer edge identity is stale")
        if (
            splice_feedback_context_identity(observed_context)
            != self.observed_edge_context_identity
        ):
            raise ValueError("boundary repair observed edge identity is stale")
        if (
            splice_feedback_context_identity(result_context)
            != self.result_edge_context_identity
        ):
            raise ValueError("boundary repair result edge identity is stale")
        if not isinstance(self.optimizer_edge_context_match, bool):
            raise TypeError("boundary repair edge context match must be a bool")
        if self.optimizer_edge_context_match is not (
            optimizer_context == observed_context
        ):
            raise ValueError("boundary repair edge context match contradicts inputs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_index": self.boundary_index,
            "outgoing_source_index": self.outgoing_source_index,
            "incoming_source_index": self.incoming_source_index,
            "result": self.result.to_dict(),
            "selected_visual_evidence": self.selected_visual_evidence.to_dict(),
            "candidate_evidence_audit": [
                _thaw_audit_value(item)
                for item in self.candidate_evidence_audit
            ],
            "optimizer_edge_context": _thaw_audit_value(
                self.optimizer_edge_context
            ),
            "optimizer_edge_context_identity": (
                self.optimizer_edge_context_identity
            ),
            "observed_edge_context": _thaw_audit_value(
                self.observed_edge_context
            ),
            "observed_edge_context_identity": (
                self.observed_edge_context_identity
            ),
            "result_edge_context": _thaw_audit_value(
                self.result_edge_context
            ),
            "result_edge_context_identity": (
                self.result_edge_context_identity
            ),
            "optimizer_edge_context_match": (
                self.optimizer_edge_context_match
            ),
        }


@dataclass(frozen=True, slots=True)
class VisualBoundaryRepairPlan:
    timeline: VisualTimelinePlan
    attempt_input_timelines: tuple[VisualTimelinePlan, ...]
    repairs: tuple[VisualBoundaryRepair, ...]
    skipped_same_source_boundaries: int
    failures: tuple[Mapping[str, Any], ...]
    requested_boundary_indices: tuple[int, ...] | None = None
    geometry_provider_attempt_count: int = 0
    geometry_provider_failure_count: int = 0
    temporal_pose_motion_thresholds: TemporalPoseMotionThresholds | None = None
    temporal_pose_motion_preference_weight: float = 0.0
    analysis_basis: str = ANALYSIS_BASIS

    def __post_init__(self) -> None:
        if not isinstance(self.timeline, VisualTimelinePlan):
            raise TypeError("boundary repair output timeline is invalid")
        attempts = tuple(self.attempt_input_timelines)
        if not attempts or any(
            not isinstance(item, VisualTimelinePlan) for item in attempts
        ):
            raise TypeError(
                "boundary repair plan needs exact attempt input timelines"
            )
        object.__setattr__(self, "attempt_input_timelines", attempts)
        repairs = tuple(self.repairs)
        if any(not isinstance(item, VisualBoundaryRepair) for item in repairs):
            raise TypeError("boundary repair entries are invalid")
        output_boundary_count = len(self.timeline.choices) - 1
        for repair in repairs:
            if not 0 <= repair.boundary_index < output_boundary_count:
                raise ValueError("boundary repair entry is outside the timeline")
            for context_name, context in (
                ("optimizer", repair.optimizer_edge_context),
                ("observed", repair.observed_edge_context),
                ("result", repair.result_edge_context),
            ):
                if context.get("boundary_index") != repair.boundary_index:
                    raise ValueError(
                        f"boundary repair {context_name} context index drifted"
                    )
                outgoing = context.get("outgoing")
                incoming = context.get("incoming")
                if not isinstance(outgoing, Mapping) or not isinstance(
                    incoming,
                    Mapping,
                ):
                    raise TypeError("boundary repair context choices are invalid")
                if (
                    outgoing.get("source_index")
                    != repair.outgoing_source_index
                    or incoming.get("source_index")
                    != repair.incoming_source_index
                ):
                    raise ValueError(
                        f"boundary repair {context_name} source direction drifted"
                    )
        object.__setattr__(self, "repairs", repairs)
        requested = self.requested_boundary_indices
        if requested is not None:
            requested_values = tuple(requested)
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in requested_values
            ):
                raise TypeError("requested boundary indices must be integers")
            if len(set(requested_values)) != len(requested_values):
                raise ValueError("requested boundary indices cannot repeat")
            if any(
                not 0 <= value < output_boundary_count
                for value in requested_values
            ):
                raise ValueError("requested boundary index is outside timeline")
            object.__setattr__(
                self,
                "requested_boundary_indices",
                tuple(sorted(requested_values)),
            )
        for name in (
            "skipped_same_source_boundaries",
            "geometry_provider_attempt_count",
            "geometry_provider_failure_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"boundary repair {name} is invalid")
        if (
            not isinstance(self.analysis_basis, str)
            or not self.analysis_basis.strip()
            or self.analysis_basis != self.analysis_basis.strip()
        ):
            raise ValueError("boundary repair analysis_basis is invalid")
        frozen_failures = tuple(
            _freeze_audit_value(failure) for failure in self.failures
        )
        if any(not isinstance(failure, Mapping) for failure in frozen_failures):
            raise TypeError("boundary repair failure must be a mapping")
        object.__setattr__(self, "failures", frozen_failures)
        attempts_by_identity: dict[str, VisualTimelinePlan] = {}
        for attempt in attempts:
            identity = splice_feedback_plan_identity(attempt)
            previous = attempts_by_identity.get(identity)
            if previous is not None and (
                splice_feedback_plan_context(previous)
                != splice_feedback_plan_context(attempt)
            ):
                raise ValueError("boundary repair attempt identity collision")
            attempts_by_identity[identity] = attempt
        failures_by_identity: dict[str, list[Mapping[str, Any]]] = {}
        for failure in self.failures:
            identity = failure.get("timeline_path_identity")
            if not isinstance(identity, str) or identity not in attempts_by_identity:
                raise ValueError(
                    "boundary repair failure has no matching attempt input"
                )
            failures_by_identity.setdefault(identity, []).append(failure)
        for identity, failures in failures_by_identity.items():
            build_splice_failure_feedback(
                attempts_by_identity[identity],
                tuple(failures),
            )

    @property
    def changed_count(self) -> int:
        return sum(item.result.changed for item in self.repairs)

    @property
    def reviewed_cross_source_count(self) -> int:
        return len(self.repairs)

    def _coverage_summary(self) -> dict[str, Any]:
        coverage = tuple(
            item.selected_visual_evidence.evidence_coverage
            for item in self.repairs
        )
        temporal_pose_motion = tuple(
            item.selected_visual_evidence.temporal_pose_motion
            for item in self.repairs
        )

        def side_count(name: str) -> int:
            return sum(sum(getattr(item, name)) for item in coverage)

        def pair_count(name: str) -> int:
            return sum(all(getattr(item, name)) for item in coverage)

        missing_counts: dict[str, int] = {}
        for item in coverage:
            for name in item.missing_mask:
                missing_counts[name] = missing_counts.get(name, 0) + 1
        classification_counts: dict[str, int] = {}
        calibration_status_counts: dict[str, int] = {}
        for item in temporal_pose_motion:
            classification_counts[item.classification] = (
                classification_counts.get(item.classification, 0) + 1
            )
            calibration_status_counts[item.calibration_status] = (
                calibration_status_counts.get(item.calibration_status, 0) + 1
            )
        return {
            "boundary_count": len(coverage),
            "side_count": 2 * len(coverage),
            "exact_actual_pose_side_count": side_count(
                "exact_pose_available"
            ),
            "exact_actual_pose_pair_count": pair_count(
                "exact_pose_available"
            ),
            "temporal_actual_pose_side_count": side_count(
                "temporal_pose_available"
            ),
            "temporal_actual_pose_pair_count": pair_count(
                "temporal_pose_available"
            ),
            "nearest_temporal_actual_pose_pair_count": sum(
                all(item.temporal_pose_available)
                and any(
                    offset not in (None, 0)
                    for offset in item.temporal_pose_offsets_frames
                )
                for item in coverage
            ),
            "formation_side_count": sum(
                sum(count > 0 for count in item.formation_person_counts)
                for item in coverage
            ),
            "formation_pair_count": sum(
                all(count > 0 for count in item.formation_person_counts)
                for item in coverage
            ),
            "formation_upper_body_side_count": sum(
                sum(count > 0 for count in item.formation_upper_body_counts)
                for item in coverage
            ),
            "formation_upper_body_pair_count": sum(
                all(count > 0 for count in item.formation_upper_body_counts)
                for item in coverage
            ),
            "person_layout_side_count": side_count(
                "person_layout_available"
            ),
            "person_layout_pair_count": pair_count(
                "person_layout_available"
            ),
            "silhouette_side_count": side_count("silhouette_available"),
            "silhouette_pair_count": pair_count("silhouette_available"),
            "professional_gate_failure_count": sum(
                not item.professional_gate_passed for item in coverage
            ),
            "professional_gate_failure_boundary_indices": [
                repair.boundary_index
                for repair in self.repairs
                if not (
                    repair.selected_visual_evidence.evidence_coverage
                    .professional_gate_passed
                )
            ],
            "missing_mask_counts": dict(sorted(missing_counts.items())),
            "hog_used_as_pose_count": sum(
                item.selected_visual_evidence.pose_basis
                == "structure_hog_fallback"
                for item in self.repairs
            ),
            "temporal_pose_motion": {
                "calibration_status_counts": dict(
                    sorted(calibration_status_counts.items())
                ),
                "classification_counts": dict(
                    sorted(classification_counts.items())
                ),
                "analysis_executed_count": sum(
                    item.analysis_executed for item in temporal_pose_motion
                ),
                "valid_analysis_count": sum(
                    item.input_valid for item in temporal_pose_motion
                ),
                "dynamic_pose_phase_match_count": sum(
                    item.dynamic_pose_phase_match
                    for item in temporal_pose_motion
                ),
                "static_pose_match_count": sum(
                    item.static_pose_match for item in temporal_pose_motion
                ),
                "direction_evidence_available_count": sum(
                    item.direction_evidence_available
                    for item in temporal_pose_motion
                ),
                "dynamic_direction_match_proven_count": sum(
                    item.dynamic_direction_match_proven
                    for item in temporal_pose_motion
                ),
                "soft_preference_applied_count": sum(
                    item.soft_preference_applied > 0.0
                    for item in temporal_pose_motion
                ),
                "soft_preference_applied_total": sum(
                    item.soft_preference_applied
                    for item in temporal_pose_motion
                ),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        attempt_identities = tuple(
            splice_feedback_plan_identity(item)
            for item in self.attempt_input_timelines
        )
        unique_attempt_contexts: dict[str, dict[str, object]] = {}
        for identity, attempt in zip(
            attempt_identities,
            self.attempt_input_timelines,
            strict=True,
        ):
            unique_attempt_contexts.setdefault(
                identity,
                splice_feedback_plan_context(attempt),
            )
        return {
            "analysis_basis": self.analysis_basis,
            "repair_attempt_count": len(self.attempt_input_timelines),
            "repair_attempt_input_timeline_identities": list(
                attempt_identities
            ),
            "repair_attempt_input_timeline_contexts": [
                {
                    "timeline_path_identity": identity,
                    "timeline_context": context,
                }
                for identity, context in sorted(
                    unique_attempt_contexts.items()
                )
            ],
            "reviewed_cross_source_count": self.reviewed_cross_source_count,
            "changed_count": self.changed_count,
            "skipped_same_source_boundaries": self.skipped_same_source_boundaries,
            "requested_boundary_indices": (
                list(self.requested_boundary_indices)
                if self.requested_boundary_indices is not None
                else None
            ),
            "failure_count": len(self.failures),
            "failures": [
                _thaw_audit_value(failure) for failure in self.failures
            ],
            "geometry_provider_attempt_count": (
                self.geometry_provider_attempt_count
            ),
            "geometry_provider_failure_count": (
                self.geometry_provider_failure_count
            ),
            "temporal_pose_motion_configuration": (
                _temporal_pose_motion_configuration(
                    self.temporal_pose_motion_thresholds,
                    self.temporal_pose_motion_preference_weight,
                )
            ),
            "evidence_coverage": self._coverage_summary(),
            "repairs": [item.to_dict() for item in self.repairs],
        }


DescriptorProvider = Callable[[int, float], FrameVisualDescriptor | None]


class BoundaryDescriptorCache:
    """Bounded process-local descriptor reuse across candidate repairs.

    GROUP candidates share the same canonical boundaries and frequently ask
    for the same source-local frame neighborhoods.  Re-running pose, formation,
    face, and silhouette inference for every candidate made the exact-eight
    production path scale with candidate count instead of unique frames.  This
    cache stores only immutable numeric descriptors, never decoded pixels, and
    deliberately has no persistence surface.
    """

    def __init__(self, *, maximum_entries: int = 8_192) -> None:
        if (
            isinstance(maximum_entries, bool)
            or not isinstance(maximum_entries, int)
            or maximum_entries < 1
        ):
            raise ValueError("maximum_entries must be a positive integer")
        self.maximum_entries = maximum_entries
        self._values: OrderedDict[
            tuple[int, float], FrameVisualDescriptor | None
        ] = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0

    @staticmethod
    def _key(source_index: int, seconds: float) -> tuple[int, float]:
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
        ):
            raise ValueError("descriptor cache source index is invalid")
        numeric = float(seconds)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError("descriptor cache timestamp is invalid")
        return source_index, round(numeric, 7)

    def lookup(
        self,
        source_index: int,
        seconds: float,
    ) -> tuple[bool, FrameVisualDescriptor | None]:
        key = self._key(source_index, seconds)
        if key not in self._values:
            self.miss_count += 1
            return False, None
        value = self._values.pop(key)
        self._values[key] = value
        self.hit_count += 1
        return True, value

    def store(
        self,
        source_index: int,
        seconds: float,
        descriptor: FrameVisualDescriptor | None,
    ) -> None:
        key = self._key(source_index, seconds)
        self._values.pop(key, None)
        self._values[key] = descriptor
        while len(self._values) > self.maximum_entries:
            self._values.popitem(last=False)
            self.eviction_count += 1

    def diagnostics(self) -> dict[str, int | bool | str]:
        return {
            "contract": "bounded_process_local_boundary_descriptor_cache_v1",
            "maximum_entries": self.maximum_entries,
            "entry_count": len(self._values),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "eviction_count": self.eviction_count,
            "raw_frames_persisted": False,
            "persistent": False,
        }


def _stable_temporal_pose_choice(
    exact: FrameVisualDescriptor,
    candidates: Sequence[tuple[int, FrameVisualDescriptor]],
    *,
    output_fps: float,
) -> tuple[int, FrameVisualDescriptor] | None:
    """Find the nearest real pose without crossing an unstable shot edge."""

    if _actual_pose_available(exact):
        return (0, exact)
    gates = _PRODUCTION_REVIEW_GATES
    for frame_offset, descriptor in candidates:
        if not _actual_pose_available(descriptor):
            continue
        stability = describe_boundary_side_stability(
            exact,
            descriptor,
            sample_span_seconds=abs(frame_offset) / output_fps,
        )
        if boundary_side_stability_exceeded(
            stability,
            maximum_scale_delta=gates.maximum_boundary_side_scale_delta,
            maximum_structure_mismatch=(
                gates.maximum_boundary_side_structure_mismatch
            ),
            maximum_pose_mismatch=gates.maximum_boundary_side_pose_mismatch,
            maximum_shot_type_mismatch=(
                gates.maximum_boundary_side_shot_type_mismatch
            ),
        ):
            continue
        return (frame_offset, descriptor)
    return None


def _professional_evidence_basis(
    *,
    exact_pose_mismatch: float | None,
    temporal_pose_mismatch: float | None,
    formation_mismatch: float | None,
    layout_mismatch: float | None,
    persisted_layout_mismatch: float | None,
    silhouette_mismatch: float | None,
) -> str:
    """Require two complementary geometry signals, without requiring pose.

    Group stages frequently have no unambiguous dominant performer.  Missing
    dominant pose is therefore neutral, but structure alone cannot authorize a
    professional cut.  A pose comparison needs supporting screen geometry; a
    formation comparison needs layout/silhouette support; and the final
    fallback is persisted subject layout plus an independent silhouette vector.
    """

    geometry_available = any(
        value is not None
        for value in (
            formation_mismatch,
            layout_mismatch,
            persisted_layout_mismatch,
            silhouette_mismatch,
        )
    )
    if exact_pose_mismatch is not None and geometry_available:
        return "exact_actual_pose_plus_screen_geometry"
    if temporal_pose_mismatch is not None and geometry_available:
        return "nearest_interior_temporal_actual_pose_plus_screen_geometry"
    if formation_mismatch is not None and any(
        value is not None
        for value in (
            layout_mismatch,
            persisted_layout_mismatch,
            silhouette_mismatch,
        )
    ):
        return "formation_assignment_plus_screen_geometry"
    if (
        (layout_mismatch is not None or persisted_layout_mismatch is not None)
        and silhouette_mismatch is not None
    ):
        return "subject_layout_plus_silhouette"
    return "insufficient_combined_geometry_evidence"


def _temporal_pose_motion_evidence(
    *,
    outgoing_preceding: FrameVisualDescriptor | None,
    outgoing_exact: FrameVisualDescriptor,
    incoming_exact: FrameVisualDescriptor,
    incoming_following: FrameVisualDescriptor | None,
    output_fps: float,
    side_sample_span_seconds: float | None,
    thresholds: TemporalPoseMotionThresholds | None,
    preference_weight: float,
) -> TemporalPoseMotionEvidence:
    configuration = _temporal_pose_motion_configuration(
        thresholds,
        preference_weight,
    )
    if thresholds is None:
        return TemporalPoseMotionEvidence(
            calibration_status="calibration_unbound",
            analysis_executed=False,
            input_valid=False,
            qualifies=False,
            classification="calibration_unbound",
            failure_reasons=("temporal_pose_motion_thresholds_unbound",),
            feature_dimension=None,
            outgoing_sample_count=0,
            incoming_sample_count=0,
            outgoing_boundary_motion_magnitude=None,
            incoming_boundary_motion_magnitude=None,
            endpoint_pose_cosine=None,
            velocity_direction_cosine=None,
            phase_compatibility=None,
            confidence=0.0,
            static_pose_match=False,
            dynamic_pose_phase_match=False,
            direction_evidence_available=False,
            dynamic_direction_match_proven=False,
            soft_preference_applied=0.0,
            threshold_values=None,
            threshold_identity_sha256=None,
            preference_weight=float(preference_weight),
            configuration_fingerprint_sha256=str(
                configuration["configuration_fingerprint_sha256"]
            ),
        )

    span = (
        float(side_sample_span_seconds)
        if side_sample_span_seconds is not None
        and math.isfinite(side_sample_span_seconds)
        and side_sample_span_seconds > 0.0
        else 3.0 / float(output_fps)
    )
    outgoing_pair_available = (
        _actual_temporal_pose_available(outgoing_preceding)
        and _actual_temporal_pose_available(outgoing_exact)
    )
    incoming_pair_available = (
        _actual_temporal_pose_available(incoming_exact)
        and _actual_temporal_pose_available(incoming_following)
    )
    outgoing_window = (
        TemporalPoseWindow(
            times_seconds=(-span, 0.0),
            embeddings=(
                outgoing_preceding.pose_embedding,
                outgoing_exact.pose_embedding,
            ),
        )
        if outgoing_pair_available and outgoing_preceding is not None
        else None
    )
    incoming_window = (
        TemporalPoseWindow(
            times_seconds=(0.0, span),
            embeddings=(
                incoming_exact.pose_embedding,
                incoming_following.pose_embedding,
            ),
        )
        if incoming_pair_available and incoming_following is not None
        else None
    )
    analysis: TemporalPoseMotionBoundaryAnalysis = (
        analyze_temporal_pose_motion_boundary(
            outgoing=outgoing_window,
            incoming=incoming_window,
            thresholds=thresholds,
        )
    )
    missing_reasons: list[str] = []
    for name, descriptor in (
        ("outgoing_preceding", outgoing_preceding),
        ("outgoing_exact", outgoing_exact),
        ("incoming_exact", incoming_exact),
        ("incoming_following", incoming_following),
    ):
        if not _actual_temporal_pose_available(descriptor):
            missing_reasons.append(f"missing_{name}_actual_60d_pose")
    return TemporalPoseMotionEvidence(
        calibration_status="calibration_bound",
        analysis_executed=True,
        input_valid=analysis.input_valid,
        qualifies=analysis.qualifies,
        classification=analysis.classification,
        failure_reasons=tuple(
            dict.fromkeys((*analysis.failure_reasons, *missing_reasons))
        ),
        feature_dimension=analysis.feature_dimension,
        outgoing_sample_count=analysis.outgoing_sample_count,
        incoming_sample_count=analysis.incoming_sample_count,
        outgoing_boundary_motion_magnitude=(
            analysis.outgoing_boundary_motion_magnitude
        ),
        incoming_boundary_motion_magnitude=(
            analysis.incoming_boundary_motion_magnitude
        ),
        endpoint_pose_cosine=analysis.endpoint_pose_cosine,
        velocity_direction_cosine=analysis.velocity_direction_cosine,
        phase_compatibility=analysis.phase_compatibility,
        confidence=analysis.confidence,
        static_pose_match=analysis.static_pose_match,
        dynamic_pose_phase_match=analysis.dynamic_pose_phase_match,
        direction_evidence_available=(
            analysis.input_valid
            and analysis.velocity_direction_cosine is not None
        ),
        dynamic_direction_match_proven=analysis.qualifies,
        soft_preference_applied=0.0,
        threshold_values=dict(
            configuration["threshold_values"]  # type: ignore[arg-type]
        ),
        threshold_identity_sha256=str(
            configuration["threshold_identity_sha256"]
        ),
        preference_weight=float(preference_weight),
        configuration_fingerprint_sha256=str(
            configuration["configuration_fingerprint_sha256"]
        ),
    )


def _visual_evidence(
    outgoing: FrameVisualDescriptor,
    incoming: FrameVisualDescriptor,
    *,
    outgoing_preceding: FrameVisualDescriptor | None = None,
    incoming_following: FrameVisualDescriptor | None = None,
    outgoing_temporal_pose_candidates: Sequence[
        tuple[int, FrameVisualDescriptor]
    ] = (),
    incoming_temporal_pose_candidates: Sequence[
        tuple[int, FrameVisualDescriptor]
    ] = (),
    output_fps: float = 29.97,
    side_sample_span_seconds: float | None = None,
    temporal_pose_motion_thresholds: TemporalPoseMotionThresholds | None = None,
    temporal_pose_motion_preference_weight: float = 0.0,
) -> BoundaryVisualEvidence:
    formation = comparable_group_formation_mismatch(
        outgoing.formation_people,
        incoming.formation_people,
        left_shot_type=outgoing.shot_type,
        right_shot_type=incoming.shot_type,
    )
    exact_pose_available = (
        _actual_pose_available(outgoing),
        _actual_pose_available(incoming),
    )
    exact_pose = (
        _rms(outgoing.pose_embedding, incoming.pose_embedding)
        if all(exact_pose_available)
        else None
    )
    outgoing_temporal_pose = _stable_temporal_pose_choice(
        outgoing,
        outgoing_temporal_pose_candidates,
        output_fps=output_fps,
    )
    incoming_temporal_pose = _stable_temporal_pose_choice(
        incoming,
        incoming_temporal_pose_candidates,
        output_fps=output_fps,
    )
    temporal_pose_available = (
        outgoing_temporal_pose is not None,
        incoming_temporal_pose is not None,
    )
    temporal_pose_offsets = (
        outgoing_temporal_pose[0]
        if outgoing_temporal_pose is not None
        else None,
        incoming_temporal_pose[0]
        if incoming_temporal_pose is not None
        else None,
    )
    nearest_temporal_pose = (
        _rms(
            outgoing_temporal_pose[1].pose_embedding,
            incoming_temporal_pose[1].pose_embedding,
        )
        if all(temporal_pose_available)
        and any(offset not in (None, 0) for offset in temporal_pose_offsets)
        else None
    )
    if exact_pose is not None:
        pose = exact_pose
        pose_basis = "exact_frame_opencv_zoo_mediapipe_pose"
    elif nearest_temporal_pose is not None:
        pose = nearest_temporal_pose
        pose_basis = "nearest_interior_temporal_opencv_zoo_mediapipe_pose"
    else:
        # HOG remains available below as ``structure_mismatch``.  It is not a
        # pose model and must never be serialized or gated as pose evidence.
        pose = None
        pose_basis = "actual_pose_pair_unavailable"
    silhouette = _rms(
        outgoing.silhouette_embedding,
        incoming.silhouette_embedding,
    )
    persisted_layout = screen_layout_mismatch(
        (),
        (),
        left_bbox=outgoing.silhouette_bbox_xywh,
        right_bbox=incoming.silhouette_bbox_xywh,
        left_centroid=outgoing.silhouette_centroid_xy,
        right_centroid=incoming.silhouette_centroid_xy,
    )
    # Match selection/review semantics: prefer comparable boundary layout
    # embeddings, otherwise persist the exact bbox/centroid fallback. Keep the
    # bbox metric as an additional conservative gate even when embeddings are
    # present, so local retiming cannot expose a serialization-only surprise.
    layout = screen_layout_mismatch(
        outgoing.person_layout_embedding,
        incoming.person_layout_embedding,
        left_bbox=outgoing.silhouette_bbox_xywh,
        right_bbox=incoming.silhouette_bbox_xywh,
        left_centroid=outgoing.silhouette_centroid_xy,
        right_centroid=incoming.silhouette_centroid_xy,
    )
    focus = _bbox_mismatch(outgoing.primary_face_bbox, incoming.primary_face_bbox)
    structure = _rms(outgoing.structure_embedding, incoming.structure_embedding)
    scale = abs(outgoing.shot_scale - incoming.shot_scale)
    shot_type = _shot_type_mismatch(outgoing.shot_type, incoming.shot_type)
    outgoing_stability = (
        describe_boundary_side_stability(
            outgoing_preceding,
            outgoing,
            sample_span_seconds=side_sample_span_seconds,
        )
        if outgoing_preceding is not None
        and side_sample_span_seconds is not None
        else None
    )
    incoming_stability = (
        describe_boundary_side_stability(
            incoming,
            incoming_following,
            sample_span_seconds=side_sample_span_seconds,
        )
        if incoming_following is not None
        and side_sample_span_seconds is not None
        else None
    )
    temporal_pose_motion = _temporal_pose_motion_evidence(
        outgoing_preceding=outgoing_preceding,
        outgoing_exact=outgoing,
        incoming_exact=incoming,
        incoming_following=incoming_following,
        output_fps=output_fps,
        side_sample_span_seconds=side_sample_span_seconds,
        thresholds=temporal_pose_motion_thresholds,
        preference_weight=temporal_pose_motion_preference_weight,
    )

    formation_counts = (
        len(outgoing.formation_people),
        len(incoming.formation_people),
    )
    formation_upper_body_counts = (
        sum(
            bool(person.upper_body_keypoints_xyc)
            for person in outgoing.formation_people
        ),
        sum(
            bool(person.upper_body_keypoints_xyc)
            for person in incoming.formation_people
        ),
    )
    layout_available = (
        bool(outgoing.person_layout_embedding),
        bool(incoming.person_layout_embedding),
    )
    silhouette_available = (
        bool(outgoing.silhouette_embedding),
        bool(incoming.silhouette_embedding),
    )
    evidence_basis = _professional_evidence_basis(
        exact_pose_mismatch=exact_pose,
        temporal_pose_mismatch=nearest_temporal_pose,
        formation_mismatch=formation,
        layout_mismatch=layout,
        persisted_layout_mismatch=persisted_layout,
        silhouette_mismatch=silhouette,
    )
    missing: list[str] = []
    for side, present in zip(
        ("outgoing", "incoming"), exact_pose_available, strict=True
    ):
        if not present:
            missing.append(f"{side}_exact_actual_pose")
    for side, present in zip(
        ("outgoing", "incoming"), temporal_pose_available, strict=True
    ):
        if not present:
            missing.append(f"{side}_temporal_actual_pose")
    for side, count in zip(
        ("outgoing", "incoming"), formation_counts, strict=True
    ):
        if count == 0:
            missing.append(f"{side}_formation")
    for side, count in zip(
        ("outgoing", "incoming"), formation_upper_body_counts, strict=True
    ):
        if count == 0:
            missing.append(f"{side}_formation_upper_body_pose")
    for side, present in zip(
        ("outgoing", "incoming"), layout_available, strict=True
    ):
        if not present:
            missing.append(f"{side}_person_layout")
    for side, present in zip(
        ("outgoing", "incoming"), silhouette_available, strict=True
    ):
        if not present:
            missing.append(f"{side}_silhouette")
    coverage = BoundaryEvidenceCoverage(
        exact_pose_available=exact_pose_available,
        temporal_pose_available=temporal_pose_available,
        temporal_pose_offsets_frames=temporal_pose_offsets,
        formation_person_counts=formation_counts,
        formation_upper_body_counts=formation_upper_body_counts,
        person_layout_available=layout_available,
        silhouette_available=silhouette_available,
        professional_evidence_basis=evidence_basis,
        professional_gate_passed=(
            evidence_basis != "insufficient_combined_geometry_evidence"
        ),
        missing_mask=tuple(missing),
    )

    # Missing signals are neutral, never evidence of compatibility.  At least
    # structure and shot-scale are always available from ``describe_frame``.
    weighted = 1.65 * scale + 0.75 * shot_type
    weight = 2.40
    for value, factor in (
        (formation, 2.55),
        (pose, 1.35),
        (silhouette, 1.55),
        (layout, 1.85),
        (focus, 1.65),
        (structure, 0.45),
    ):
        if value is not None:
            weighted += factor * value
            weight += factor
    base_total_cost = weighted / weight
    if (
        temporal_pose_motion.dynamic_pose_phase_match
        and temporal_pose_motion_preference_weight > 0.0
    ):
        # A calibrated dynamic match contributes only a caller-weighted
        # zero-mismatch evidence term. It cannot reject a candidate, bypass
        # the professional gates, or create a quota. The resulting preference
        # is mathematically bounded between zero and the unchanged base cost.
        total_cost = weighted / (
            weight + temporal_pose_motion_preference_weight
        )
        temporal_pose_motion = replace(
            temporal_pose_motion,
            soft_preference_applied=base_total_cost - total_cost,
        )
    else:
        total_cost = base_total_cost
    return BoundaryVisualEvidence(
        total_cost=total_cost,
        base_total_cost=base_total_cost,
        formation_mismatch=formation,
        pose_mismatch=pose,
        pose_basis=pose_basis,
        exact_pose_mismatch=exact_pose,
        nearest_temporal_pose_mismatch=nearest_temporal_pose,
        silhouette_mismatch=silhouette,
        layout_mismatch=layout,
        persisted_layout_mismatch=persisted_layout,
        focus_mismatch=focus,
        structure_mismatch=structure,
        shot_scale_delta=scale,
        shot_type_mismatch=shot_type,
        temporal_pose_motion=temporal_pose_motion,
        evidence_coverage=coverage,
        outgoing_side_stability=outgoing_stability,
        incoming_side_stability=incoming_stability,
    )


def _passes_production_transition_gates(
    evidence: BoundaryVisualEvidence,
) -> bool:
    """Never let local timing repair weaken the selector's hard safety gates."""

    gates = _PRODUCTION_REVIEW_GATES
    if not evidence.evidence_coverage.professional_gate_passed:
        return False
    checks = (
        (evidence.shot_scale_delta, gates.maximum_scale_mismatch),
        (evidence.focus_mismatch, gates.maximum_focus_mismatch),
        (evidence.layout_mismatch, gates.maximum_layout_mismatch),
        (evidence.persisted_layout_mismatch, gates.maximum_layout_mismatch),
        (evidence.silhouette_mismatch, gates.maximum_silhouette_mismatch),
        (evidence.formation_mismatch, gates.maximum_formation_mismatch),
        (evidence.pose_mismatch, gates.maximum_pose_mismatch),
    )
    if not all(value is None or value <= maximum + 1e-9 for value, maximum in checks):
        return False
    for stability in (
        evidence.outgoing_side_stability,
        evidence.incoming_side_stability,
    ):
        if stability is None or boundary_side_stability_exceeded(
            stability,
            maximum_scale_delta=gates.maximum_boundary_side_scale_delta,
            maximum_structure_mismatch=(
                gates.maximum_boundary_side_structure_mismatch
            ),
            maximum_pose_mismatch=gates.maximum_boundary_side_pose_mismatch,
            maximum_shot_type_mismatch=(
                gates.maximum_boundary_side_shot_type_mismatch
            ),
        ):
            return False
    return True


def _replace_tuple_item(values: tuple[Any, Any, Any], index: int, value: Any) -> tuple:
    mutable = list(values)
    mutable[index] = value
    return tuple(mutable)


def _new_hard_failures(
    before: TimelineReview,
    after: TimelineReview,
) -> tuple[dict[str, Any], ...]:
    """Return hard failures introduced by one proposed boundary transaction.

    Source selection is fixed, so occurrence indices are stable across local
    retiming.  Comparing occurrences as well as codes catches a newly-too-long
    source run even when another run already had the same failure code.
    """

    before_by_code = {issue.code: issue for issue in before.hard_failures}
    introduced: list[dict[str, Any]] = []
    for issue in after.hard_failures:
        previous = before_by_code.get(issue.code)
        if previous is None:
            introduced.append(issue.to_dict())
            continue
        if not issue.occurrences:
            continue
        new_occurrences = tuple(
            sorted(set(issue.occurrences) - set(previous.occurrences))
        )
        if new_occurrences:
            payload = issue.to_dict()
            payload["occurrences"] = list(new_occurrences)
            introduced.append(payload)
    return tuple(introduced)


def _updated_exit(
    choice: VisualTimelineChoice,
    *,
    boundary: float,
    source_local_boundary: float,
    descriptor: FrameVisualDescriptor,
    evidence: BoundaryVisualEvidence,
    internal_scene_cut_master_seconds: tuple[float, ...],
    scene_risk: bool,
    boundary_changed: bool,
) -> VisualTimelineChoice:
    changes: dict[str, Any] = {
        "canonical_master_end": boundary,
        "source_local_end": source_local_boundary,
        "exit_shot_scale": descriptor.shot_scale,
        "exit_shot_type": descriptor.shot_type,
        "exit_scene_cut_near": scene_risk,
        "internal_scene_cut_master_seconds": internal_scene_cut_master_seconds,
        "exit_boundary_side_stability": evidence.outgoing_side_stability,
        "face_samples_xywh": _replace_tuple_item(
            choice.face_samples_xywh, 2, descriptor.primary_face_bbox
        ),
        "face_detector_backends": _replace_tuple_item(
            choice.face_detector_backends, 2, descriptor.face_detector_backend
        ),
        "face_landmark_samples_xy": _replace_tuple_item(
            choice.face_landmark_samples_xy, 2, descriptor.primary_face_landmarks_xy
        ),
        "face_head_roll_samples_radians": _replace_tuple_item(
            choice.face_head_roll_samples_radians,
            2,
            descriptor.primary_face_head_roll_radians,
        ),
        "face_eye_scale_samples": _replace_tuple_item(
            choice.face_eye_scale_samples, 2, descriptor.primary_face_eye_scale
        ),
        "pose_backends": _replace_tuple_item(
            choice.pose_backends, 2, descriptor.pose_backend
        ),
        "pose_confidences": _replace_tuple_item(
            choice.pose_confidences, 2, descriptor.pose_confidence
        ),
        "detected_person_counts": _replace_tuple_item(
            choice.detected_person_counts, 2, descriptor.detected_person_count
        ),
        "formation_person_counts": _replace_tuple_item(
            choice.formation_person_counts, 2, len(descriptor.formation_people)
        ),
        "formation_upper_body_counts": _replace_tuple_item(
            choice.formation_upper_body_counts,
            2,
            sum(bool(person.upper_body_keypoints_xyc) for person in descriptor.formation_people),
        ),
        "silhouette_samples_xywh": _replace_tuple_item(
            choice.silhouette_samples_xywh, 2, descriptor.silhouette_bbox_xywh
        ),
        "silhouette_centroid_samples_xy": _replace_tuple_item(
            choice.silhouette_centroid_samples_xy,
            2,
            descriptor.silhouette_centroid_xy,
        ),
        "silhouette_backends": _replace_tuple_item(
            choice.silhouette_backends, 2, descriptor.silhouette_backend
        ),
        "silhouette_confidences": _replace_tuple_item(
            choice.silhouette_confidences, 2, descriptor.silhouette_confidence
        ),
    }
    if boundary_changed:
        changes.update(
            dissolve_exit_face_geometry_window=(),
            dissolve_exit_face_box_window=(),
            dissolve_exit_face_identity_window=(),
            dissolve_exit_pose_embedding_window=(),
            dissolve_exit_pose_backend_window=(),
            dissolve_exit_pose_confidence_window=(),
            dissolve_exit_silhouette_embedding_window=(),
            dissolve_exit_silhouette_backend_window=(),
            dissolve_exit_silhouette_confidence_window=(),
        )
    return replace(choice, **changes)


def _updated_entry(
    choice: VisualTimelineChoice,
    *,
    boundary: float,
    source_local_boundary: float,
    descriptor: FrameVisualDescriptor,
    evidence: BoundaryVisualEvidence,
    internal_scene_cut_master_seconds: tuple[float, ...],
    scene_risk: bool,
    boundary_changed: bool,
) -> VisualTimelineChoice:
    changes: dict[str, Any] = {
        "canonical_master_start": boundary,
        "source_local_start": source_local_boundary,
        "entry_shot_scale": descriptor.shot_scale,
        "entry_shot_type": descriptor.shot_type,
        "entry_scene_cut_near": scene_risk,
        "internal_scene_cut_master_seconds": internal_scene_cut_master_seconds,
        "entry_boundary_side_stability": evidence.incoming_side_stability,
        "face_samples_xywh": _replace_tuple_item(
            choice.face_samples_xywh, 0, descriptor.primary_face_bbox
        ),
        "face_detector_backends": _replace_tuple_item(
            choice.face_detector_backends, 0, descriptor.face_detector_backend
        ),
        "face_landmark_samples_xy": _replace_tuple_item(
            choice.face_landmark_samples_xy, 0, descriptor.primary_face_landmarks_xy
        ),
        "face_head_roll_samples_radians": _replace_tuple_item(
            choice.face_head_roll_samples_radians,
            0,
            descriptor.primary_face_head_roll_radians,
        ),
        "face_eye_scale_samples": _replace_tuple_item(
            choice.face_eye_scale_samples, 0, descriptor.primary_face_eye_scale
        ),
        "pose_backends": _replace_tuple_item(
            choice.pose_backends, 0, descriptor.pose_backend
        ),
        "pose_confidences": _replace_tuple_item(
            choice.pose_confidences, 0, descriptor.pose_confidence
        ),
        "detected_person_counts": _replace_tuple_item(
            choice.detected_person_counts, 0, descriptor.detected_person_count
        ),
        "formation_person_counts": _replace_tuple_item(
            choice.formation_person_counts, 0, len(descriptor.formation_people)
        ),
        "formation_upper_body_counts": _replace_tuple_item(
            choice.formation_upper_body_counts,
            0,
            sum(bool(person.upper_body_keypoints_xyc) for person in descriptor.formation_people),
        ),
        "silhouette_samples_xywh": _replace_tuple_item(
            choice.silhouette_samples_xywh, 0, descriptor.silhouette_bbox_xywh
        ),
        "silhouette_centroid_samples_xy": _replace_tuple_item(
            choice.silhouette_centroid_samples_xy,
            0,
            descriptor.silhouette_centroid_xy,
        ),
        "silhouette_backends": _replace_tuple_item(
            choice.silhouette_backends, 0, descriptor.silhouette_backend
        ),
        "silhouette_confidences": _replace_tuple_item(
            choice.silhouette_confidences, 0, descriptor.silhouette_confidence
        ),
        "face_transition_minimax_mismatch_from_previous": None,
        "pose_transition_basis_from_previous": evidence.pose_basis,
        "pose_transition_mismatch_from_previous": evidence.pose_mismatch,
        "silhouette_transition_mismatch_from_previous": evidence.silhouette_mismatch,
        "layout_transition_mismatch_from_previous": evidence.layout_mismatch,
        "formation_transition_mismatch_from_previous": evidence.formation_mismatch,
    }
    if boundary_changed:
        changes.update(
            dissolve_entry_face_geometry_window=(),
            dissolve_entry_face_box_window=(),
            dissolve_entry_face_identity_window=(),
            dissolve_entry_pose_embedding_window=(),
            dissolve_entry_pose_backend_window=(),
            dissolve_entry_pose_confidence_window=(),
            dissolve_entry_silhouette_embedding_window=(),
            dissolve_entry_silhouette_backend_window=(),
            dissolve_entry_silhouette_confidence_window=(),
        )
    return replace(choice, **changes)


def repair_visual_timeline_boundaries(
    *,
    visual_timeline: VisualTimelinePlan,
    sources: Sequence[Path],
    audio_offsets: Sequence[float],
    scene_cut_times: Sequence[Sequence[float]],
    output_fps: float = 29.97,
    beat_period_seconds: float | None = None,
    descriptor_provider: DescriptorProvider | None = None,
    descriptor_cache: BoundaryDescriptorCache | None = None,
    feature_cache_directory: Path | None = None,
    config: BoundaryRepairConfig | None = None,
    source_timewarps: Sequence[TimeWarpMap | None] | None = None,
    boundary_indices: Sequence[int] | None = None,
) -> VisualBoundaryRepairPlan:
    """Return locally repaired timing for every or only requested boundaries.

    ``boundary_indices`` is the proxy-triggered partial re-edit contract.  It
    lets decoded-output QA revisit only the failed cuts with a wider local
    radius while preserving every decision outside those exact boundaries.
    """

    if not visual_timeline.choices:
        raise ValueError("visual timeline cannot be empty")
    if not (
        len(sources) == len(audio_offsets) == len(scene_cut_times)
    ):
        raise ValueError("boundary-repair source arrays must have matching lengths")
    if not math.isfinite(output_fps) or output_fps <= 0.0:
        raise ValueError("output_fps must be positive and finite")
    settings = config or BoundaryRepairConfig()
    # Reject unauthorized calibration before opening media, invoking a
    # descriptor provider, or discovering that every requested boundary can
    # be skipped.
    _temporal_pose_motion_configuration(
        settings.temporal_pose_motion_thresholds,
        settings.temporal_pose_motion_preference_weight,
    )
    boundary_count = len(visual_timeline.choices) - 1
    requested_boundary_indices: tuple[int, ...] | None
    if boundary_indices is None:
        requested_boundary_indices = None
        boundary_indices_to_review = tuple(range(boundary_count))
    else:
        normalized: list[int] = []
        for value in boundary_indices:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("boundary_indices must contain integers")
            if not 0 <= value < boundary_count:
                raise ValueError("boundary index is outside the visual timeline")
            normalized.append(value)
        if len(normalized) != len(set(normalized)):
            raise ValueError("boundary_indices cannot contain duplicates")
        requested_boundary_indices = tuple(sorted(normalized))
        boundary_indices_to_review = requested_boundary_indices
    if source_timewarps is None:
        clock_maps: tuple[TimeWarpMap | None, ...] = (None,) * len(sources)
    else:
        if len(source_timewarps) != len(sources):
            raise ValueError("source_timewarps must match boundary-repair sources")
        clock_maps = tuple(source_timewarps)
        for index, timewarp in enumerate(clock_maps):
            if timewarp is None:
                continue
            if not isinstance(timewarp, TimeWarpMap):
                raise TypeError(
                    "source_timewarps entries must be TimeWarpMap or None"
                )
            if not math.isclose(
                timewarp.global_offset_seconds,
                float(audio_offsets[index]),
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"source_timewarps[{index}] global offset does not match audio_offsets"
                )

    def source_time(source_index: int, master_seconds: float) -> float:
        timewarp = clock_maps[source_index]
        if timewarp is None:
            return master_seconds + float(audio_offsets[source_index])
        return timewarp.master_to_source(master_seconds)

    def master_time(source_index: int, source_seconds: float) -> float:
        timewarp = clock_maps[source_index]
        if timewarp is None:
            return source_seconds - float(audio_offsets[source_index])
        return timewarp.source_to_master(source_seconds)

    def internal_cut_times(
        source_index: int,
        start_seconds: float,
        end_seconds: float,
    ) -> tuple[float, ...]:
        return tuple(
            sorted(
                {
                    mapped
                    for source_cut in scene_cut_times[source_index]
                    for mapped in (
                        master_time(source_index, float(source_cut)),
                    )
                    if start_seconds + 1e-6 < mapped < end_seconds - 1e-6
                }
            )
        )

    samplers: dict[int, _VideoSampler] = {}
    if descriptor_provider is None:
        def provider(source_index: int, seconds: float) -> FrameVisualDescriptor | None:
            # Preserve the former boundary sampler's fail-closed behavior at
            # the start of a source.  CanonicalFrameIdentity intentionally
            # clamps negative values for general callers, but a repair handle
            # before source time zero is missing evidence, not frame zero.
            if seconds < 0.0:
                return None
            if descriptor_cache is not None:
                hit, cached = descriptor_cache.lookup(source_index, seconds)
                if hit:
                    return cached
            sampler = samplers.get(source_index)
            if sampler is None:
                sampler = _VideoSampler(
                    Path(sources[source_index]),
                    feature_cache_directory=feature_cache_directory,
                )
                samplers[source_index] = sampler
            descriptor = sampler.feature_descriptor_at(
                seconds,
                include_subject_geometry=True,
                include_formation_pose=True,
            )
            if descriptor_cache is not None:
                descriptor_cache.store(source_index, seconds, descriptor)
            return descriptor

    else:
        if feature_cache_directory is not None:
            raise ValueError(
                "feature_cache_directory cannot wrap an explicit "
                "descriptor_provider"
            )
        if descriptor_cache is not None:
            raise ValueError(
                "descriptor_cache cannot wrap an explicit descriptor_provider"
            )
        provider = descriptor_provider

    geometry_provider_attempts: set[tuple[int, float]] = set()
    geometry_provider_failures: set[tuple[int, float]] = set()

    def sample_descriptor(
        source_index: int,
        seconds: float,
    ) -> FrameVisualDescriptor | None:
        key = (source_index, round(float(seconds), 7))
        geometry_provider_attempts.add(key)
        descriptor = provider(source_index, seconds)
        if descriptor is None:
            geometry_provider_failures.add(key)
        return descriptor

    def crosses_source_camera_cut(
        source_index: int,
        first_seconds: float,
        second_seconds: float,
    ) -> bool:
        lower, upper = sorted((first_seconds, second_seconds))
        return any(
            lower + 1e-6 < float(value) < upper - 1e-6
            for value in scene_cut_times[source_index]
        )

    choices = list(visual_timeline.choices)
    optimizer_timeline_identity = splice_feedback_plan_identity(
        visual_timeline
    )

    def attempt_edge_contexts(
        boundary_index: int,
        observed_choices: Sequence[VisualTimelineChoice],
    ) -> tuple[dict[str, object], str, dict[str, object], str]:
        observed_plan = {
            "analysis_basis": visual_timeline.analysis_basis,
            "source_clock_basis": visual_timeline.source_clock_basis,
            "choices": tuple(observed_choices),
        }
        optimizer_context = splice_feedback_edge_context(
            visual_timeline,
            boundary_index,
        )
        observed_context = splice_feedback_edge_context(
            observed_plan,
            boundary_index,
        )
        return (
            optimizer_context,
            splice_feedback_edge_context_identity(
                visual_timeline,
                boundary_index,
            ),
            observed_context,
            splice_feedback_edge_context_identity(
                observed_plan,
                boundary_index,
            ),
        )

    def bind_failure_to_attempt(
        failure: dict[str, Any],
        *,
        observed_choices: Sequence[VisualTimelineChoice],
    ) -> dict[str, Any]:
        boundary_index = failure.get("boundary_index")
        if isinstance(boundary_index, bool) or not isinstance(
            boundary_index,
            int,
        ):
            raise TypeError("boundary repair failure index is invalid")
        if not 0 <= boundary_index < len(visual_timeline.choices) - 1:
            raise ValueError("boundary repair failure index is outside input")
        optimizer_outgoing = visual_timeline.choices[boundary_index]
        optimizer_incoming = visual_timeline.choices[boundary_index + 1]
        if (
            failure.get("outgoing_source_index")
            != optimizer_outgoing.source_index
            or failure.get("incoming_source_index")
            != optimizer_incoming.source_index
        ):
            raise ValueError(
                "boundary repair failure direction drifted from optimizer input"
            )
        (
            optimizer_edge_context,
            optimizer_edge_identity,
            observed_edge_context,
            observed_edge_identity,
        ) = attempt_edge_contexts(
            boundary_index,
            observed_choices,
        )
        return {
            **failure,
            "outgoing_shot_id": optimizer_outgoing.shot_id,
            "incoming_shot_id": optimizer_incoming.shot_id,
            "timeline_analysis_basis": visual_timeline.analysis_basis,
            "timeline_source_clock_basis": (
                visual_timeline.source_clock_basis
            ),
            "timeline_path_identity": optimizer_timeline_identity,
            "optimizer_edge_context": optimizer_edge_context,
            "optimizer_edge_context_identity": optimizer_edge_identity,
            "observed_edge_context": observed_edge_context,
            "observed_edge_context_identity": observed_edge_identity,
            "optimizer_edge_context_match": (
                observed_edge_context == optimizer_edge_context
            ),
        }

    current_review = review_timeline(tuple(choices))
    repairs: list[VisualBoundaryRepair] = []
    failures: list[dict[str, Any]] = []
    same_source = 0
    try:
        for boundary_index in boundary_indices_to_review:
            outgoing = choices[boundary_index]
            incoming = choices[boundary_index + 1]
            if outgoing.source_index == incoming.source_index:
                same_source += 1
                continue
            nominal = outgoing.canonical_master_end
            # The bounded local search accepts scalar offsets. Tangent it at
            # the nominal boundary; even the maximum eight-frame local-clock
            # delta is far below a decoded frame, then persist the exact
            # nonlinear mapping at the selected boundary below.
            outgoing_offset = (
                source_time(outgoing.source_index, nominal) - nominal
            )
            incoming_offset = (
                source_time(incoming.source_index, nominal) - nominal
            )
            evidence_by_offset: dict[
                int,
                tuple[BoundaryVisualEvidence, FrameVisualDescriptor, FrameVisualDescriptor],
            ] = {}

            def candidate_evidence_audit() -> tuple[dict[str, Any], ...]:
                """Serialize every requested offset, including decode gaps."""

                audit: list[dict[str, Any]] = []
                for frame_offset in range(
                    -settings.search_radius_frames,
                    settings.search_radius_frames + 1,
                ):
                    item = evidence_by_offset.get(frame_offset)
                    if item is None:
                        audit.append(
                            {
                                "frame_offset": frame_offset,
                                "pose_basis": "missing",
                                "professional_evidence_basis": (
                                    "missing_combined_geometry_descriptor"
                                ),
                                "professional_gate_passed": False,
                                "production_transition_gate_passed": False,
                                "missing_mask": [
                                    "combined_geometry_descriptor"
                                ],
                                "temporal_pose_motion": None,
                                "visual_evidence": {},
                            }
                        )
                        continue
                    evidence = item[0]
                    audit.append(
                        {
                            "frame_offset": frame_offset,
                            "pose_basis": evidence.pose_basis,
                            "professional_evidence_basis": (
                                evidence.evidence_coverage
                                .professional_evidence_basis
                            ),
                            "professional_gate_passed": (
                                evidence.evidence_coverage
                                .professional_gate_passed
                            ),
                            "production_transition_gate_passed": (
                                _passes_production_transition_gates(evidence)
                            ),
                            "missing_mask": list(
                                evidence.evidence_coverage.missing_mask
                            ),
                            "temporal_pose_motion": (
                                evidence.temporal_pose_motion.to_dict()
                            ),
                            "visual_evidence": evidence.to_dict(),
                        }
                    )
                return tuple(audit)

            def visual_cost(context: BoundaryCandidateContext) -> float | None:
                frame_seconds = 1.0 / output_fps
                side_sample_span_seconds = 3.0 * frame_seconds
                outgoing_exact_seconds = (
                    context.outgoing_source_local_seconds - frame_seconds / 2.0
                )
                incoming_exact_seconds = (
                    context.incoming_source_local_seconds + frame_seconds / 2.0
                )
                outgoing_descriptor = sample_descriptor(
                    outgoing.source_index,
                    outgoing_exact_seconds,
                )
                outgoing_preceding_descriptor = sample_descriptor(
                    outgoing.source_index,
                    outgoing_exact_seconds - side_sample_span_seconds,
                )
                incoming_descriptor = sample_descriptor(
                    incoming.source_index,
                    incoming_exact_seconds,
                )
                incoming_following_descriptor = sample_descriptor(
                    incoming.source_index,
                    incoming_exact_seconds + side_sample_span_seconds,
                )
                if (
                    outgoing_descriptor is None
                    or outgoing_preceding_descriptor is None
                    or incoming_descriptor is None
                    or incoming_following_descriptor is None
                ):
                    return None

                def temporal_pose_candidates(
                    *,
                    source_index: int,
                    exact_seconds: float,
                    exact_descriptor: FrameVisualDescriptor,
                    direction: int,
                ) -> tuple[tuple[int, FrameVisualDescriptor], ...]:
                    if _actual_pose_available(exact_descriptor):
                        return ()
                    values: list[tuple[int, FrameVisualDescriptor]] = []
                    for distance in range(
                        1, _NEAREST_TEMPORAL_POSE_RADIUS_FRAMES + 1
                    ):
                        offset = direction * distance
                        candidate_seconds = (
                            exact_seconds + offset * frame_seconds
                        )
                        if crosses_source_camera_cut(
                            source_index,
                            exact_seconds,
                            candidate_seconds,
                        ):
                            break
                        descriptor = sample_descriptor(
                            source_index,
                            candidate_seconds,
                        )
                        if descriptor is None:
                            continue
                        values.append((offset, descriptor))
                        if _stable_temporal_pose_choice(
                            exact_descriptor,
                            values,
                            output_fps=output_fps,
                        ) is not None:
                            break
                    return tuple(values)

                outgoing_temporal_pose = temporal_pose_candidates(
                    source_index=outgoing.source_index,
                    exact_seconds=outgoing_exact_seconds,
                    exact_descriptor=outgoing_descriptor,
                    direction=-1,
                )
                incoming_temporal_pose = temporal_pose_candidates(
                    source_index=incoming.source_index,
                    exact_seconds=incoming_exact_seconds,
                    exact_descriptor=incoming_descriptor,
                    direction=1,
                )
                evidence = _visual_evidence(
                    outgoing_descriptor,
                    incoming_descriptor,
                    outgoing_preceding=outgoing_preceding_descriptor,
                    incoming_following=incoming_following_descriptor,
                    outgoing_temporal_pose_candidates=(
                        outgoing_temporal_pose
                    ),
                    incoming_temporal_pose_candidates=(
                        incoming_temporal_pose
                    ),
                    output_fps=output_fps,
                    side_sample_span_seconds=side_sample_span_seconds,
                    temporal_pose_motion_thresholds=(
                        settings.temporal_pose_motion_thresholds
                    ),
                    temporal_pose_motion_preference_weight=(
                        settings.temporal_pose_motion_preference_weight
                    ),
                )
                evidence_by_offset[context.frame_offset] = (
                    evidence,
                    outgoing_descriptor,
                    incoming_descriptor,
                )
                if not _passes_production_transition_gates(evidence):
                    return None
                return evidence.total_cost

            try:
                result = repair_boundary(
                    BoundaryRepairRequest(
                        nominal_boundary_seconds=nominal,
                        outgoing_segment_start_seconds=(
                            outgoing.canonical_master_start
                        ),
                        incoming_segment_end_seconds=incoming.canonical_master_end,
                        output_fps=output_fps,
                        outgoing_source_offset_seconds=outgoing_offset,
                        incoming_source_offset_seconds=incoming_offset,
                        outgoing_scene_cut_times_seconds=tuple(
                            float(value)
                            for value in scene_cut_times[outgoing.source_index]
                        ),
                        incoming_scene_cut_times_seconds=tuple(
                            float(value)
                            for value in scene_cut_times[incoming.source_index]
                        ),
                        beat_period_seconds=beat_period_seconds,
                        beat_intensity=1.0 if beat_period_seconds is not None else None,
                    ),
                    visual_cost_callback=visual_cost,
                    config=settings,
                )
                evidence, outgoing_descriptor, incoming_descriptor = evidence_by_offset[
                    result.selected_frame_offset
                ]
            except (ValueError, KeyError, cv2.error) as exc:
                failure: dict[str, Any] = {
                    "boundary_index": boundary_index,
                    "outgoing_source_index": outgoing.source_index,
                    "incoming_source_index": incoming.source_index,
                    "reason": str(exc),
                }
                failure["candidate_evidence_audit"] = (
                    candidate_evidence_audit()
                )
                failures.append(
                    bind_failure_to_attempt(
                        failure,
                        observed_choices=tuple(choices),
                    )
                )
                continue

            boundary = result.selected_boundary_seconds
            selected_scene_risk = bool(
                result.selected.outgoing_interior_scene_cuts
                or result.selected.incoming_interior_scene_cuts
            )
            original_scene_risk = bool(
                outgoing.exit_scene_cut_near or incoming.entry_scene_cut_near
            )
            if selected_scene_risk and not original_scene_risk:
                failures.append(
                    bind_failure_to_attempt(
                        {
                        "boundary_index": boundary_index,
                        "outgoing_source_index": outgoing.source_index,
                        "incoming_source_index": incoming.source_index,
                        "reason": "selected_scene_cut_risk_exceeds_original",
                        "selected_frame_offset": result.selected_frame_offset,
                        "original_scene_cut_risk": False,
                        "selected_scene_cut_risk": True,
                        "selected_visual_evidence": evidence.to_dict(),
                        },
                        observed_choices=tuple(choices),
                    )
                )
                continue

            proposed_outgoing = _updated_exit(
                outgoing,
                boundary=boundary,
                source_local_boundary=source_time(
                    outgoing.source_index, boundary
                ),
                descriptor=outgoing_descriptor,
                evidence=evidence,
                internal_scene_cut_master_seconds=internal_cut_times(
                    outgoing.source_index,
                    outgoing.canonical_master_start,
                    boundary,
                ),
                scene_risk=bool(result.selected.outgoing_interior_scene_cuts),
                boundary_changed=result.changed,
            )
            proposed_incoming = _updated_entry(
                incoming,
                boundary=boundary,
                source_local_boundary=source_time(
                    incoming.source_index, boundary
                ),
                descriptor=incoming_descriptor,
                evidence=evidence,
                internal_scene_cut_master_seconds=internal_cut_times(
                    incoming.source_index,
                    boundary,
                    incoming.canonical_master_end,
                ),
                scene_risk=bool(result.selected.incoming_interior_scene_cuts),
                boundary_changed=result.changed,
            )
            proposed_choices = list(choices)
            proposed_choices[boundary_index] = proposed_outgoing
            proposed_choices[boundary_index + 1] = proposed_incoming
            proposed_review = review_timeline(tuple(proposed_choices))
            introduced_hard_failures = _new_hard_failures(
                current_review,
                proposed_review,
            )
            if introduced_hard_failures:
                failures.append(
                    bind_failure_to_attempt(
                        {
                        "boundary_index": boundary_index,
                        "outgoing_source_index": outgoing.source_index,
                        "incoming_source_index": incoming.source_index,
                        "reason": "repair_would_introduce_hard_timeline_failure",
                        "selected_frame_offset": result.selected_frame_offset,
                        "introduced_hard_failures": list(
                            introduced_hard_failures
                        ),
                        "selected_visual_evidence": evidence.to_dict(),
                        },
                        observed_choices=tuple(choices),
                    )
                )
                continue

            choices = proposed_choices
            current_review = proposed_review
            (
                optimizer_edge_context,
                optimizer_edge_context_identity,
                observed_edge_context,
                observed_edge_context_identity,
            ) = attempt_edge_contexts(
                boundary_index,
                # The scalar evidence was measured before applying this
                # boundary transaction.  Preserve that exact input context.
                tuple(choices[:boundary_index])
                + (outgoing, incoming)
                + tuple(choices[boundary_index + 2 :]),
            )
            result_plan = {
                "analysis_basis": visual_timeline.analysis_basis,
                "source_clock_basis": visual_timeline.source_clock_basis,
                "choices": tuple(choices),
            }
            result_edge_context = splice_feedback_result_context(
                result_plan,
                boundary_index,
            )
            result_edge_context_identity = (
                splice_feedback_result_context_identity(
                    result_plan,
                    boundary_index,
                )
            )
            repairs.append(
                VisualBoundaryRepair(
                    boundary_index=boundary_index,
                    outgoing_source_index=outgoing.source_index,
                    incoming_source_index=incoming.source_index,
                    result=result,
                    selected_visual_evidence=evidence,
                    candidate_evidence_audit=(
                        candidate_evidence_audit()
                    ),
                    optimizer_edge_context=optimizer_edge_context,
                    optimizer_edge_context_identity=(
                        optimizer_edge_context_identity
                    ),
                    observed_edge_context=observed_edge_context,
                    observed_edge_context_identity=(
                        observed_edge_context_identity
                    ),
                    result_edge_context=result_edge_context,
                    result_edge_context_identity=(
                        result_edge_context_identity
                    ),
                    optimizer_edge_context_match=(
                        optimizer_edge_context == observed_edge_context
                    ),
                )
            )
    finally:
        for sampler in samplers.values():
            sampler.close()

    repaired_timeline = VisualTimelinePlan(
        choices=tuple(choices),
        optimizer_score=visual_timeline.optimizer_score,
        analysis_basis=(
            f"{visual_timeline.analysis_basis}+local_boundary_repair_v4+"
            "complete_success_offset_audit_v1+"
            "temporal_pose_motion_audit_v1"
        ),
        selected_candidate_index=visual_timeline.selected_candidate_index,
        candidate_generation_basis=visual_timeline.candidate_generation_basis,
        candidate_review_summaries=visual_timeline.candidate_review_summaries,
        source_clock_basis=visual_timeline.source_clock_basis,
    )
    return VisualBoundaryRepairPlan(
        timeline=repaired_timeline,
        attempt_input_timelines=(visual_timeline,),
        repairs=tuple(repairs),
        skipped_same_source_boundaries=same_source,
        failures=tuple(failures),
        requested_boundary_indices=requested_boundary_indices,
        geometry_provider_attempt_count=len(geometry_provider_attempts),
        geometry_provider_failure_count=len(geometry_provider_failures),
        temporal_pose_motion_thresholds=(
            settings.temporal_pose_motion_thresholds
        ),
        temporal_pose_motion_preference_weight=(
            settings.temporal_pose_motion_preference_weight
        ),
    )


__all__ = [
    "ANALYSIS_BASIS",
    "BoundaryDescriptorCache",
    "BoundaryEvidenceCoverage",
    "BoundaryVisualEvidence",
    "TemporalPoseMotionEvidence",
    "VisualBoundaryRepair",
    "VisualBoundaryRepairPlan",
    "repair_visual_timeline_boundaries",
]
