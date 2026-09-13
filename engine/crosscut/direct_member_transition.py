"""Evidence-gated dissolves for official target-member fancam edits.

The direct-member selector already computes timestamp-bound geometry on both
sides of every real source switch.  This module consumes that *existing*
evidence; it never decodes a second, looser set of frames and never infers an
identity.  A dissolve is authored only when the endpoint pair and the first
0.2 seconds of the outgoing handle remain position/scale/body compatible.

Every failed or incomplete check deterministically degrades to the ordinary
hard cut.  The returned decision audit is scalar-only and can therefore be
embedded in the analysis report without persisting pose or silhouette vectors.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
from statistics import fmean
from typing import Any, Sequence

from .direct_member_visual import (
    DirectMemberVisualSelection, DirectMemberCinematicBridge,
    FOCAL_POSE_MOTION_BRIDGE_BASIS,
    verified_focal_pose_motion_bridge,
    verified_focal_pose_dissolve_spacing,
)
from .focal_selector_geometry import (
    FocalSelectorGeometry, FOCAL_SELECTOR_FACE_BASIS, FOCAL_SELECTOR_BODY_BASIS,
    FOCAL_SELECTOR_POSE_BASIS, common_focal_pose_vectors, focal_pose_shape_mismatch,
)
from .member_bridge_contract import (
    MINIMUM_MEMBER_BRIDGE_HANDLE_QUALITY,
    RequiredMemberBridgeAuthoringError,
    member_bridge_handle_quality_failures,
    required_member_bridge_spacing_seconds,
    MINIMUM_REQUIRED_MEMBER_BRIDGE_SPACING_SECONDS,
    MINIMUM_FOCAL_POSE_DISSOLVE_SPACING_SECONDS,
)
from .render import (
    DIRECT_MEMBER_POSE_DISSOLVE_REASON,
    MEMBER_BLUR_BRIDGE_REASON,
    MEMBER_ZOOM_BRIDGE_REASON,
    RenderSegment,
    RenderTransition,
    TransitionKind,
)
from .timeline import formation_assignment_mismatch


OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT = (
    "automatic_acquisition_official_target_member_fancam_v1"
)
DIRECT_MEMBER_POSE_DISSOLVE_CONTRACT = (
    "official_target_member_exact_endpoint_pose_matched_logistic_dissolve_v1"
)
DIRECT_MEMBER_CINEMATIC_BRIDGE_CONTRACT = (
    "official_target_member_required_cinematic_effect_bridge_v1"
)

_FACE_ATTENTION_BASIS = "unambiguous_dominant_face_bbox_geometry_only"
_BODY_ATTENTION_BASES = {
    "aggregate_human_silhouette",
    "unambiguous_dominant_person_box_geometry_only",
}
_CROSS_BASIS_CINEMATIC_BRIDGE_ATTENTION_BASIS = (
    "verified_cross_basis_pose_silhouette_effect_bridge"
)
_CINEMATIC_BRIDGE_MAXIMUM_POSE_MISMATCH = 0.30
_CINEMATIC_BRIDGE_MAXIMUM_SILHOUETTE_MISMATCH = 0.55
_CINEMATIC_BRIDGE_MAXIMUM_BASE_MISMATCH = 0.65
_POSE_DISSOLVE_BRIDGE_MAXIMUM_ENDPOINT_SHAPE_MISMATCH = 0.38
_POSE_DISSOLVE_BRIDGE_MINIMUM_DIRECTION_COSINE = 0.12
_POSE_DISSOLVE_BRIDGE_MAXIMUM_SPEED_RATIO_MISMATCH = 0.60
_POSE_DISSOLVE_BRIDGE_MAXIMUM_PHASE_MISMATCH = 0.34
_NONPRIORITY_CINEMATIC_BRIDGE_MAXIMUM_POSE_MISMATCH = 0.22
_NONPRIORITY_CINEMATIC_BRIDGE_MAXIMUM_SILHOUETTE_MISMATCH = 0.48
_NONPRIORITY_CINEMATIC_BRIDGE_MAXIMUM_BASE_MISMATCH = 0.52
_CINEMATIC_BRIDGE_MINIMUM_EVENT_SPACING_SECONDS = MINIMUM_REQUIRED_MEMBER_BRIDGE_SPACING_SECONDS


@dataclass(frozen=True, slots=True)
class DirectMemberPoseDissolveConfig:
    """Fixed conservative bounds for a short, duration-neutral dissolve."""

    minimum_duration_seconds: float = 0.18
    preferred_duration_seconds: float = 0.23
    maximum_duration_seconds: float = 0.30
    target_dissolves_per_minute: float = 5.2
    maximum_boundary_fraction: float = 0.36
    minimum_boundary_gap: int = 1
    minimum_event_spacing_seconds: float = MINIMUM_FOCAL_POSE_DISSOLVE_SPACING_SECONDS
    minimum_endpoint_evidence_component_count: int = 3
    minimum_endpoint_shape_component_count: int = 2
    minimum_motion_endpoint_component_count: int = 2
    minimum_handle_shape_component_count: int = 2
    maximum_attention_position_mismatch: float = 0.30
    maximum_attention_scale_mismatch: float = 0.29
    maximum_pose_mismatch: float = 0.38
    maximum_formation_mismatch: float = 0.28
    maximum_silhouette_mismatch: float = 0.34
    maximum_future_attention_position_mismatch: float = 0.32
    maximum_future_attention_scale_mismatch: float = 0.30
    maximum_future_pose_mismatch: float = 0.40
    maximum_future_formation_mismatch: float = 0.32
    maximum_future_silhouette_mismatch: float = 0.36
    maximum_handle_entry_attention_position_jump: float = 0.30
    maximum_handle_entry_attention_scale_jump: float = 0.29
    maximum_handle_entry_shape_jump: float = 0.42
    maximum_motion_endpoint_shape_mismatch: float = 0.38
    minimum_static_shape_strength: float = 0.76
    minimum_handle_quality_score: float = MINIMUM_MEMBER_BRIDGE_HANDLE_QUALITY
    handle_sample_horizon_seconds: float = 0.20

    def __post_init__(self) -> None:
        finite = tuple(
            float(getattr(self, name))
            for name in (
                "minimum_duration_seconds",
                "preferred_duration_seconds",
                "maximum_duration_seconds",
                "target_dissolves_per_minute",
                "maximum_boundary_fraction",
                "minimum_event_spacing_seconds",
                "maximum_attention_position_mismatch",
                "maximum_attention_scale_mismatch",
                "maximum_pose_mismatch",
                "maximum_formation_mismatch",
                "maximum_silhouette_mismatch",
                "maximum_future_attention_position_mismatch",
                "maximum_future_attention_scale_mismatch",
                "maximum_future_pose_mismatch",
                "maximum_future_formation_mismatch",
                "maximum_future_silhouette_mismatch",
                "maximum_handle_entry_attention_position_jump",
                "maximum_handle_entry_attention_scale_jump",
                "maximum_handle_entry_shape_jump",
                "maximum_motion_endpoint_shape_mismatch",
                "minimum_static_shape_strength",
                "minimum_handle_quality_score",
                "handle_sample_horizon_seconds",
            )
        )
        if any(not math.isfinite(value) or value < 0.0 for value in finite):
            raise ValueError("direct member dissolve configuration must be finite")
        if not (
            0.0
            < self.minimum_duration_seconds
            <= self.preferred_duration_seconds
            <= self.maximum_duration_seconds
            <= 0.30
        ):
            raise ValueError("direct member dissolve duration bounds are invalid")
        if self.minimum_duration_seconds < 0.18:
            raise ValueError("direct member dissolve cannot be shorter than 0.18 seconds")
        if self.target_dissolves_per_minute <= 0.0:
            raise ValueError("direct member dissolve density must be positive")
        for value in (
            self.maximum_boundary_fraction,
            self.maximum_attention_position_mismatch,
            self.maximum_attention_scale_mismatch,
            self.maximum_pose_mismatch,
            self.maximum_formation_mismatch,
            self.maximum_silhouette_mismatch,
            self.maximum_future_attention_position_mismatch,
            self.maximum_future_attention_scale_mismatch,
            self.maximum_future_pose_mismatch,
            self.maximum_future_formation_mismatch,
            self.maximum_future_silhouette_mismatch,
            self.maximum_handle_entry_attention_position_jump,
            self.maximum_handle_entry_attention_scale_jump,
            self.maximum_handle_entry_shape_jump,
            self.maximum_motion_endpoint_shape_mismatch,
            self.minimum_static_shape_strength,
            self.minimum_handle_quality_score,
            self.handle_sample_horizon_seconds,
        ):
            if value > 1.0:
                raise ValueError("direct member dissolve normalized bounds exceed one")
        for name, value in (
            ("minimum_boundary_gap", self.minimum_boundary_gap),
            (
                "minimum_endpoint_evidence_component_count",
                self.minimum_endpoint_evidence_component_count,
            ),
            (
                "minimum_endpoint_shape_component_count",
                self.minimum_endpoint_shape_component_count,
            ),
            (
                "minimum_motion_endpoint_component_count",
                self.minimum_motion_endpoint_component_count,
            ),
            (
                "minimum_handle_shape_component_count",
                self.minimum_handle_shape_component_count,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            if name.endswith("shape_component_count") and value > 3:
                raise ValueError(f"{name} cannot exceed three")
            if name == "minimum_motion_endpoint_component_count" and value > 3:
                raise ValueError(f"{name} cannot exceed three")

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_seconds": {
                "minimum": self.minimum_duration_seconds,
                "preferred": self.preferred_duration_seconds,
                "maximum": self.maximum_duration_seconds,
            },
            "target_dissolves_per_minute": self.target_dissolves_per_minute,
            "maximum_boundary_fraction": self.maximum_boundary_fraction,
            "minimum_boundary_gap": self.minimum_boundary_gap,
            "minimum_event_spacing_seconds": self.minimum_event_spacing_seconds,
            "required_bridge_spacing": {
                "generic_minimum_seconds": MINIMUM_REQUIRED_MEMBER_BRIDGE_SPACING_SECONDS,
                "verified_focal_pose_minimum_seconds": required_member_bridge_spacing_seconds(
                    True, True, pose_minimum_seconds=self.minimum_event_spacing_seconds),
                "both_events_require_source_focal_articulation": True,
                "exact_materialized_cut_times_checked": True,
            },
            "minimum_endpoint_evidence_component_count": (
                self.minimum_endpoint_evidence_component_count
            ),
            "minimum_endpoint_shape_component_count": (
                self.minimum_endpoint_shape_component_count
            ),
            "minimum_motion_endpoint_component_count": (
                self.minimum_motion_endpoint_component_count
            ),
            "minimum_handle_shape_component_count": (
                self.minimum_handle_shape_component_count
            ),
            "maximum_endpoint_mismatch": {
                "attention_position": self.maximum_attention_position_mismatch,
                "attention_scale": self.maximum_attention_scale_mismatch,
                "pose": self.maximum_pose_mismatch,
                "formation": self.maximum_formation_mismatch,
                "silhouette": self.maximum_silhouette_mismatch,
            },
            "maximum_handle_mismatch": {
                "attention_position": (
                    self.maximum_future_attention_position_mismatch
                ),
                "attention_scale": self.maximum_future_attention_scale_mismatch,
                "pose": self.maximum_future_pose_mismatch,
                "formation": self.maximum_future_formation_mismatch,
                "silhouette": self.maximum_future_silhouette_mismatch,
            },
            "maximum_same_source_handle_entry_jump": {
                "attention_position": (
                    self.maximum_handle_entry_attention_position_jump
                ),
                "attention_scale": self.maximum_handle_entry_attention_scale_jump,
                "shape": self.maximum_handle_entry_shape_jump,
            },
            "maximum_motion_endpoint_shape_mismatch": (
                self.maximum_motion_endpoint_shape_mismatch
            ),
            "minimum_static_shape_strength": self.minimum_static_shape_strength,
            "minimum_handle_quality_score": self.minimum_handle_quality_score,
            "handle_sample_horizon_seconds": self.handle_sample_horizon_seconds,
            "reference_calibration": {
                "reference_video_id": "QZD2VKiNFEA",
                "profile_path": (
                    "/Volumes/CrossCutMedia/CrossCut-Scratch/reference-transitions/"
                    "QZD2VKiNFEA-member-transition-reference-profile-v1.json"
                ),
                "profile_file_sha256": (
                    "6f9e579a0bea0900d708e98950469faa6b712c976e66d9bca45f3b0b60db167c"
                ),
                "reference_video_sha256": (
                    "2b9df25b2ea51062e6488e7cfcaa3147085407f2e74ffad6a52961b3ade499b5"
                ),
                "frame_metrics_sha256": (
                    "f8e0cb977fd7c6f59d1249ed999ee3b3a0292c6ef0a33e63cd5d61cb7a7217d9"
                ),
                "transition_geometry_sha256": (
                    "3486f55496491914edbdc2ef19409fd0f0de0478f197e5d35984d212bfffac43"
                ),
                "scope": "direct_member_only",
                "group_mode_expansion_authorized": False,
                "observed_duration_seconds": 39.52,
                "observed_source_change_count": 14,
                "observed_source_changes_per_minute": 21.2535,
                "observed_median_source_run_seconds": 2.39,
                "observed_maximum_source_run_seconds": 6.36,
                "observed_median_nearest_beat_offset_seconds": 0.1348,
                "observed_effect_duration_seconds": [0.15, 0.30],
                "observed_silhouette_height_ratio_typical": [1.0, 1.2],
                "observed_silhouette_centroid_distance_typical_maximum": 0.12,
                "policy": (
                    "calibrates_effect_shape_and_priority_only_not_global_"
                    "member_cut_cadence"
                ),
                "limitations": (
                    "manual_plus_frame_change_estimate_not_ground_truth_"
                    "editor_timeline"
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class DirectMemberPoseDissolveDecision:
    boundary_index: int
    canonical_master_seconds: float
    selected: bool
    rejection_reasons: tuple[str, ...]
    priority_score: float
    highlight_priority: bool
    both_moving: bool
    attention_basis: str | None
    endpoint_attention_position_mismatch: float | None
    endpoint_attention_scale_mismatch: float | None
    endpoint_pose_mismatch: float | None
    endpoint_formation_mismatch: float | None
    endpoint_silhouette_mismatch: float | None
    future_attention_position_max_mismatch: float | None
    future_attention_scale_max_mismatch: float | None
    future_pose_max_mismatch: float | None
    future_formation_max_mismatch: float | None
    future_silhouette_max_mismatch: float | None
    outgoing_handle_available_seconds: float | None
    required_outgoing_handle_seconds: float | None
    incoming_segment_frame_count: int
    transition_frame_count: int
    exact_endpoint_evidence_used: bool
    transition_kind: TransitionKind | None = None
    required_cinematic_bridge: bool = False
    cinematic_bridge_style: str | None = None

    @property
    def resolved_transition_kind(self) -> TransitionKind:
        if self.transition_kind is not None:
            return TransitionKind(self.transition_kind)
        return (
            TransitionKind.FACE_DISSOLVE
            if self.selected
            else TransitionKind.HARD_CUT
        )

    def to_dict(self, *, fps: float) -> dict[str, Any]:
        return {
            "boundary_index": self.boundary_index,
            "canonical_master_seconds": self.canonical_master_seconds,
            "selected": self.selected,
            "transition_kind": self.resolved_transition_kind.value,
            "required_cinematic_bridge": self.required_cinematic_bridge,
            "cinematic_bridge_style": self.cinematic_bridge_style,
            "rejection_reasons": list(self.rejection_reasons),
            "priority_score": self.priority_score,
            "highlight_priority": self.highlight_priority,
            "both_moving": self.both_moving,
            "attention_basis": self.attention_basis,
            "endpoint_mismatch": {
                "attention_position": (
                    self.endpoint_attention_position_mismatch
                ),
                "attention_scale": self.endpoint_attention_scale_mismatch,
                "pose": self.endpoint_pose_mismatch,
                "formation": self.endpoint_formation_mismatch,
                "silhouette": self.endpoint_silhouette_mismatch,
            },
            "outgoing_handle_future_max_mismatch": {
                "attention_position": (
                    self.future_attention_position_max_mismatch
                ),
                "attention_scale": self.future_attention_scale_max_mismatch,
                "pose": self.future_pose_max_mismatch,
                "formation": self.future_formation_max_mismatch,
                "silhouette": self.future_silhouette_max_mismatch,
            },
            "outgoing_handle": {
                "available_seconds": self.outgoing_handle_available_seconds,
                "required_seconds": self.required_outgoing_handle_seconds,
                "validated_against_source_duration": (
                    self.outgoing_handle_available_seconds is not None
                    and self.required_outgoing_handle_seconds is not None
                    and self.outgoing_handle_available_seconds + 1e-9
                    >= self.required_outgoing_handle_seconds
                ),
            },
            "incoming_segment_frame_count": self.incoming_segment_frame_count,
            "transition_frame_count": self.transition_frame_count,
            "duration_seconds": self.transition_frame_count / fps,
            "exact_endpoint_evidence_used": self.exact_endpoint_evidence_used,
            "raw_embeddings_persisted": False,
        }


@dataclass(frozen=True, slots=True)
class DirectMemberPoseDissolvePlan:
    transitions: tuple[RenderTransition, ...]
    decisions: tuple[DirectMemberPoseDissolveDecision, ...]
    fps: float
    timeline_duration_seconds: float
    enabled_requested: bool
    source_metadata_contract_verified: bool
    config: DirectMemberPoseDissolveConfig

    def __post_init__(self) -> None:
        if len(self.transitions) != len(self.decisions):
            raise ValueError("direct member dissolve transitions/decisions differ")
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("direct member dissolve fps is invalid")
        if (
            not math.isfinite(self.timeline_duration_seconds)
            or self.timeline_duration_seconds <= 0.0
        ):
            raise ValueError("direct member dissolve timeline duration is invalid")
        for index, (transition, decision) in enumerate(
            zip(self.transitions, self.decisions, strict=True)
        ):
            if decision.boundary_index != index:
                raise ValueError("direct member dissolve decisions are not contiguous")
            if decision.resolved_transition_kind is not transition.kind:
                raise ValueError("direct member dissolve decision disagrees with render")
            if decision.selected != transition.kind.uses_overlap:
                raise ValueError("direct member effect selection disagrees with render")
            if (
                transition.kind is TransitionKind.FACE_DISSOLVE
                and transition.reason != DIRECT_MEMBER_POSE_DISSOLVE_REASON
            ):
                raise ValueError("direct member dissolve reason is outside its contract")
            expected_bridge_reason = (
                DIRECT_MEMBER_POSE_DISSOLVE_REASON
                if decision.cinematic_bridge_style == "pose_dissolve"
                else MEMBER_ZOOM_BRIDGE_REASON
                if decision.cinematic_bridge_style == "zoom_in"
                else MEMBER_BLUR_BRIDGE_REASON
                if decision.cinematic_bridge_style == "horizontal_blur"
                else None
            )
            if decision.required_cinematic_bridge and (
                expected_bridge_reason is None
                or transition.reason != expected_bridge_reason
            ):
                raise ValueError(
                    "direct member cinematic bridge is outside its contract"
                )
            if decision.selected and not (
                self.config.minimum_duration_seconds - 1e-9
                <= transition.duration_seconds(self.fps)
                <= self.config.maximum_duration_seconds + 1e-9
            ):
                raise ValueError("direct member dissolve duration is outside its contract")
        if (self.selected_count or self.cinematic_bridge_count) and (
            not self.enabled_requested or not self.source_metadata_contract_verified
        ):
            raise ValueError("direct member dissolve lacks request or provenance authority")

    @property
    def selected_count(self) -> int:
        return sum(
            decision.resolved_transition_kind is TransitionKind.FACE_DISSOLVE
            for decision in self.decisions
        )

    @property
    def cinematic_bridge_count(self) -> int:
        return sum(
            decision.required_cinematic_bridge
            for decision in self.decisions
        )

    def to_dict(self) -> dict[str, Any]:
        rejection_counts: Counter[str] = Counter(
            reason
            for decision in self.decisions
            if not decision.selected
            for reason in decision.rejection_reasons
        )
        selected = tuple(
            item
            for item in self.decisions
            if item.resolved_transition_kind is TransitionKind.FACE_DISSOLVE
        )
        cinematic_bridges = tuple(
            item
            for item in self.decisions
            if item.required_cinematic_bridge
        )
        eligible_count = sum(
            not decision.required_cinematic_bridge
            and (
                not decision.rejection_reasons
                or decision.rejection_reasons
                == ("density_or_spacing_budget",)
            )
            for decision in self.decisions
        )
        duration_minutes = self.timeline_duration_seconds / 60.0
        eligible_density = (
            eligible_count / duration_minutes if duration_minutes > 1e-9 else None
        )
        selected_density = (
            len(selected) / duration_minutes if duration_minutes > 1e-9 else None
        )
        return {
            "contract": DIRECT_MEMBER_POSE_DISSOLVE_CONTRACT,
            "source_metadata_contract": (
                OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT
            ),
            "source_metadata_contract_verified": (
                self.source_metadata_contract_verified
            ),
            "member_pose_dissolve_requested": self.enabled_requested,
            "member_pose_dissolve_effective": bool(selected),
            "member_pose_dissolve_count": len(selected),
            "cinematic_bridge_contract": (
                DIRECT_MEMBER_CINEMATIC_BRIDGE_CONTRACT
            ),
            "cinematic_bridge_effective": bool(cinematic_bridges),
            "cinematic_bridge_count": len(cinematic_bridges),
            "cinematic_bridge_styles": dict(
                sorted(
                    Counter(
                        item.cinematic_bridge_style
                        for item in cinematic_bridges
                        if item.cinematic_bridge_style is not None
                    ).items()
                )
            ),
            "cinematic_bridge_hard_cut_authorized": False,
            "analysis_basis": (
                "precomputed_exact_boundary_and_first_0_20s_handle_"
                "non_identity_geometry_with_source_duration_proof"
            ),
            "renderer_contract": (
                "duration_neutral_outgoing_handle_normalized_logistic_"
                "face_dissolve"
            ),
            "boundary_count": len(self.decisions),
            "eligible_candidate_count": eligible_count,
            "eligible_candidates_per_minute": eligible_density,
            "selected_dissolve_count": len(selected),
            "selected_dissolves_per_minute": selected_density,
            "hard_cut_count": (
                sum(
                    item.resolved_transition_kind is TransitionKind.HARD_CUT
                    for item in self.decisions
                )
            ),
            "selected_highlight_count": sum(
                item.highlight_priority for item in selected
            ),
            "selected_both_moving_count": sum(item.both_moving for item in selected),
            "selected_face_attention_count": sum(
                item.attention_basis == _FACE_ATTENTION_BASIS for item in selected
            ),
            "selected_duration_seconds": [
                item.transition_frame_count / self.fps for item in selected
            ],
            "rejection_reason_counts": dict(sorted(rejection_counts.items())),
            "visual_gates_relaxed": False,
            "identity_inference_used": False,
            "raw_embeddings_persisted": False,
            "config": self.config.to_dict(),
            "decisions": [item.to_dict(fps=self.fps) for item in self.decisions],
        }


@dataclass(frozen=True, slots=True)
class _SafeCandidate:
    boundary_index: int
    canonical_master_seconds: float
    priority_score: float
    highlight_priority: bool
    both_moving: bool
    attention_basis: str
    endpoint_attention_position_mismatch: float
    endpoint_attention_scale_mismatch: float
    endpoint_pose_mismatch: float | None
    endpoint_formation_mismatch: float | None
    endpoint_silhouette_mismatch: float | None
    future_attention_position_max_mismatch: float
    future_attention_scale_max_mismatch: float
    future_pose_max_mismatch: float | None
    future_formation_max_mismatch: float | None
    future_silhouette_max_mismatch: float | None
    outgoing_handle_available_seconds: float
    required_outgoing_handle_seconds: float
    incoming_segment_frame_count: int
    transition_frame_count: int


@dataclass(frozen=True, slots=True)
class _RequiredCinematicBridgeCandidate:
    boundary_index: int
    canonical_master_seconds: float
    style: str
    transition_kind: TransitionKind
    transition_reason: str
    highlight_priority: bool
    both_moving: bool
    attention_basis: str
    endpoint_attention_position_mismatch: float | None
    endpoint_attention_scale_mismatch: float | None
    endpoint_pose_mismatch: float
    endpoint_formation_mismatch: float | None
    endpoint_silhouette_mismatch: float | None
    outgoing_handle_available_seconds: float
    required_outgoing_handle_seconds: float
    incoming_segment_frame_count: int
    transition_frame_count: int
    verified_focal_articulation: bool = False


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _diagnostic_cut_master_seconds(diagnostic: Any) -> float | None:
    """Return the render-materialized local-search cut, with v1 fallback."""

    selected = _finite_float(
        getattr(diagnostic, "selected_cut_master_seconds", None)
    )
    if selected is not None:
        return selected
    return _finite_float(getattr(diagnostic, "canonical_master_seconds", None))


def _embedding_distance(left: Any, right: Any) -> float | None:
    if not isinstance(left, tuple) or not isinstance(right, tuple):
        return None
    if not left or len(left) != len(right):
        return None
    try:
        values = tuple(
            float(right_value) - float(left_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(value) for value in values):
        return None
    return min(1.0, math.sqrt(sum(value * value for value in values) / len(values)))


def _attention_mismatch(left: Any, right: Any) -> tuple[float | None, float | None]:
    if left is None or right is None:
        return (None, None)
    left_center = getattr(left, "centroid_xy", None)
    right_center = getattr(right, "centroid_xy", None)
    left_box = getattr(left, "bbox_xywh", None)
    right_box = getattr(right, "bbox_xywh", None)
    if not (
        isinstance(left_center, tuple)
        and isinstance(right_center, tuple)
        and len(left_center) == len(right_center) == 2
        and isinstance(left_box, tuple)
        and isinstance(right_box, tuple)
        and len(left_box) == len(right_box) == 4
    ):
        return (None, None)
    try:
        left_center_xy = tuple(float(value) for value in left_center)
        right_center_xy = tuple(float(value) for value in right_center)
        left_area = float(left_box[2]) * float(left_box[3])
        right_area = float(right_box[2]) * float(right_box[3])
    except (TypeError, ValueError):
        return (None, None)
    if (
        any(not math.isfinite(value) for value in left_center_xy + right_center_xy)
        or left_area <= 0.0
        or right_area <= 0.0
    ):
        return (None, None)
    position = min(
        1.0,
        math.hypot(
            left_center_xy[0] - right_center_xy[0],
            left_center_xy[1] - right_center_xy[1],
        )
        / 0.42,
    )
    scale = min(1.0, abs(math.log(left_area / right_area)) / math.log(3.2))
    return (position, scale)


def _attention_basis_pair(left: Any, right: Any) -> str | None:
    if any(getattr(sample, "focal_geometry", None) is not None for sample in (left, right)):
        left_basis = _verified_attention_sample_basis(left)
        right_basis = _verified_attention_sample_basis(right)
        return left_basis if left_basis == right_basis and left_basis in {
            FOCAL_SELECTOR_FACE_BASIS, FOCAL_SELECTOR_BODY_BASIS} else None
    left_basis = getattr(getattr(left, "attention", None), "basis", None)
    right_basis = getattr(getattr(right, "attention", None), "basis", None)
    if left_basis == right_basis == _FACE_ATTENTION_BASIS:
        left_descriptor = getattr(left, "descriptor", None)
        right_descriptor = getattr(right, "descriptor", None)
        if all(
            descriptor is not None
            and str(getattr(descriptor, "face_detector_backend", "")).casefold()
            == "yunet"
            and int(getattr(descriptor, "face_count", 0)) >= 1
            for descriptor in (left_descriptor, right_descriptor)
        ):
            return _FACE_ATTENTION_BASIS
        return None
    if left_basis == right_basis and left_basis in _BODY_ATTENTION_BASES:
        return str(left_basis)
    return None


def _verified_attention_sample_basis(sample: Any) -> str | None:
    attention = getattr(sample, "attention", None)
    descriptor = getattr(sample, "descriptor", None)
    basis = getattr(attention, "basis", None)
    if attention is None or descriptor is None:
        return None
    focal = getattr(sample, "focal_geometry", None)
    if focal is not None:
        local = _finite_float(getattr(sample, "source_local_seconds", None))
        if not isinstance(focal, FocalSelectorGeometry) or local is None or abs(focal.source_seconds-local)>1e-9:
            return None
        if basis == FOCAL_SELECTOR_FACE_BASIS and focal.face_bbox_xywh is not None:
            return basis
        if basis == FOCAL_SELECTOR_BODY_BASIS and focal.person is not None and (
            focal.face_bbox_xywh is not None or focal.body_track is not None):
            return basis
        return None
    if basis == _FACE_ATTENTION_BASIS:
        if (
            str(getattr(descriptor, "face_detector_backend", "")).casefold()
            == "yunet"
            and int(getattr(descriptor, "face_count", 0)) >= 1
            and getattr(descriptor, "primary_face_bbox", None) is not None
        ):
            return _FACE_ATTENTION_BASIS
        return None
    if basis == "aggregate_human_silhouette":
        if (
            getattr(descriptor, "silhouette_bbox_xywh", None) is not None
            and float(getattr(descriptor, "silhouette_confidence", 0.0)) > 0.0
        ):
            return str(basis)
        return None
    if basis == "unambiguous_dominant_person_box_geometry_only":
        if tuple(getattr(descriptor, "formation_people", ())):
            return str(basis)
        return None
    return None


def _required_bridge_attention_basis(
    left: Any,
    right: Any,
    bridge: Any,
) -> str | None:
    """Accept unlike attention boxes only under the selector's pose contract."""

    shared = _attention_basis_pair(left, right)
    if shared is not None:
        return shared
    left_basis = _verified_attention_sample_basis(left)
    right_basis = _verified_attention_sample_basis(right)
    pose_mismatch = _finite_float(getattr(bridge, "pose_mismatch", None))
    silhouette_mismatch = _finite_float(
        getattr(bridge, "silhouette_mismatch", None)
    )
    if (
        left_basis is None
        or right_basis is None
        or left_basis == right_basis
        or pose_mismatch is None
        or pose_mismatch > _CINEMATIC_BRIDGE_MAXIMUM_POSE_MISMATCH + 1e-9
        or silhouette_mismatch is None
        or silhouette_mismatch
        > _CINEMATIC_BRIDGE_MAXIMUM_SILHOUETTE_MISMATCH + 1e-9
    ):
        return None
    return _CROSS_BASIS_CINEMATIC_BRIDGE_ATTENTION_BASIS


def _descriptor_shape_mismatches(
    left: Any,
    right: Any,
    *,
    minimum_pose_confidence: float,
    minimum_silhouette_confidence: float,
) -> tuple[float | None, float | None, float | None]:
    if left is None or right is None:
        return (None, None, None)
    pose = None
    if (
        getattr(left, "pose_backend", "structure_hog_fallback")
        != "structure_hog_fallback"
        and getattr(right, "pose_backend", "structure_hog_fallback")
        != "structure_hog_fallback"
        and float(getattr(left, "pose_confidence", 0.0))
        >= minimum_pose_confidence
        and float(getattr(right, "pose_confidence", 0.0))
        >= minimum_pose_confidence
    ):
        pose = _embedding_distance(
            getattr(left, "pose_embedding", ()),
            getattr(right, "pose_embedding", ()),
        )
    formation = formation_assignment_mismatch(
        tuple(getattr(left, "formation_people", ())),
        tuple(getattr(right, "formation_people", ())),
    )
    silhouette = None
    if (
        float(getattr(left, "silhouette_confidence", 0.0))
        >= minimum_silhouette_confidence
        and float(getattr(right, "silhouette_confidence", 0.0))
        >= minimum_silhouette_confidence
    ):
        silhouette = _embedding_distance(
            getattr(left, "silhouette_embedding", ()),
            getattr(right, "silhouette_embedding", ()),
        )
    return (pose, formation, silhouette)


def _quantized_frame_counts(
    segments: Sequence[RenderSegment],
    fps: float,
) -> tuple[int, ...]:
    counts: list[int] = []
    cumulative = 0.0
    previous = 0
    for segment in segments:
        cumulative += segment.duration
        boundary = int(round(cumulative * fps))
        count = boundary - previous
        counts.append(count)
        previous = boundary
    return tuple(counts)


def _transition_frame_count(
    *,
    fps: float,
    both_moving: bool,
    highlight: bool,
    motion_strength: float,
    config: DirectMemberPoseDissolveConfig,
) -> int:
    if both_moving:
        # Faster action gets the shorter end of the safe window to avoid a
        # double-limb ghost while still making the source morph perceptible.
        desired = config.preferred_duration_seconds - min(0.04, motion_strength * 0.04)
    elif highlight:
        desired = config.preferred_duration_seconds
    else:
        desired = min(config.maximum_duration_seconds, config.preferred_duration_seconds + 0.02)
    minimum_frames = max(3, int(math.ceil(config.minimum_duration_seconds * fps - 1e-12)))
    maximum_frames = max(
        minimum_frames,
        int(math.floor(config.maximum_duration_seconds * fps + 1e-12)),
    )
    return min(maximum_frames, max(minimum_frames, int(round(desired * fps))))


def _in_highlight(
    canonical_master_seconds: float,
    intervals: Sequence[tuple[float, float]],
) -> bool:
    return any(
        start - 1e-9 <= canonical_master_seconds < end - 1e-9
        for start, end in intervals
    )


def _span_source(span: Any, source_index: int) -> Any | None:
    sources = getattr(span, "sources", ())
    if not isinstance(sources, tuple):
        return None
    for source in sources:
        if getattr(source, "eligible_source_index", None) == source_index:
            return source
    return None


def _safe_candidate(
    *,
    boundary_index: int,
    diagnostic: Any,
    visual_selection: DirectMemberVisualSelection,
    render_segments: tuple[RenderSegment, ...],
    eligible_request_source_indices: tuple[int, ...],
    eligible_source_durations_seconds: tuple[float, ...],
    frame_counts: tuple[int, ...],
    common_master_start: float,
    fps: float,
    highlight_intervals: tuple[tuple[float, float], ...],
    config: DirectMemberPoseDissolveConfig,
) -> tuple[_SafeCandidate | None, tuple[str, ...]]:
    reasons: list[str] = []
    expected_master = common_master_start + render_segments[boundary_index + 1].master_start
    canonical_master = _diagnostic_cut_master_seconds(diagnostic)
    if (
        getattr(diagnostic, "boundary_index", None) != boundary_index
        or canonical_master is None
        or abs(canonical_master - expected_master) > 1e-6
    ):
        reasons.append("stale_or_nonexact_boundary_diagnostic")

    source_path = tuple(getattr(visual_selection, "source_indices", ()))
    if len(source_path) != len(render_segments):
        reasons.append("selected_source_path_length_mismatch")
        return (None, tuple(dict.fromkeys(reasons)))
    left_source = source_path[boundary_index]
    right_source = source_path[boundary_index + 1]
    if (
        isinstance(left_source, bool)
        or isinstance(right_source, bool)
        or not isinstance(left_source, int)
        or not isinstance(right_source, int)
        or not 0 <= left_source < len(eligible_request_source_indices)
        or not 0 <= right_source < len(eligible_request_source_indices)
    ):
        reasons.append("selected_source_domain_invalid")
        return (None, tuple(dict.fromkeys(reasons)))
    if (
        getattr(diagnostic, "outgoing_eligible_source_index", None) != left_source
        or getattr(diagnostic, "incoming_eligible_source_index", None) != right_source
        or render_segments[boundary_index].source_index
        != eligible_request_source_indices[left_source]
        or render_segments[boundary_index + 1].source_index
        != eligible_request_source_indices[right_source]
    ):
        reasons.append("boundary_source_metadata_disagrees_with_render")
    outgoing_segment = render_segments[boundary_index]
    incoming_segment = render_segments[boundary_index + 1]
    cut_applied = getattr(diagnostic, "cut_applied", True)
    if not isinstance(cut_applied, bool):
        reasons.append("cut_applied_diagnostic_invalid")
    elif not cut_applied:
        reasons.append("same_source_continuation_has_no_visible_cut")

    mismatch = getattr(diagnostic, "mismatch", None)
    warning_flags = tuple(getattr(mismatch, "warning_flags", ()))
    severe_flags = tuple(getattr(mismatch, "severe_flags", ()))
    evidence_count = getattr(mismatch, "evidence_component_count", 0)
    if severe_flags:
        reasons.append("severe_endpoint_geometry")
    if warning_flags:
        reasons.append("warning_endpoint_geometry")
    if (
        isinstance(evidence_count, bool)
        or not isinstance(evidence_count, int)
        or evidence_count < config.minimum_endpoint_evidence_component_count
    ):
        reasons.append("insufficient_endpoint_geometry_evidence")

    outgoing_sample = getattr(diagnostic, "outgoing_sample", None)
    incoming_sample = getattr(diagnostic, "incoming_sample", None)
    step = 1.0 / fps
    if canonical_master is not None:
        outgoing_master = _finite_float(
            getattr(outgoing_sample, "master_seconds", None)
        )
        incoming_master = _finite_float(
            getattr(incoming_sample, "master_seconds", None)
        )
        if (
            outgoing_master is None
            or incoming_master is None
            or abs(outgoing_master - (canonical_master - step)) > 0.26 * step
            or abs(incoming_master - canonical_master) > 0.26 * step
        ):
            reasons.append("endpoint_samples_not_output_frame_bound")
        outgoing_local = _finite_float(
            getattr(outgoing_sample, "source_local_seconds", None)
        )
        incoming_local = _finite_float(
            getattr(incoming_sample, "source_local_seconds", None)
        )
        outgoing_expected_local = (
            outgoing_segment.source_end
            - step * outgoing_segment.source_playback_rate
        )
        incoming_expected_local = incoming_segment.source_start
        outgoing_tolerance = max(
            1e-4,
            0.35 * step * outgoing_segment.source_playback_rate,
        )
        incoming_tolerance = max(
            1e-4,
            0.35 * step * incoming_segment.source_playback_rate,
        )
        if (
            outgoing_local is None
            or incoming_local is None
            or abs(outgoing_local - outgoing_expected_local)
            > outgoing_tolerance
            or abs(incoming_local - incoming_expected_local)
            > incoming_tolerance
        ):
            reasons.append("endpoint_samples_not_render_source_clock_bound")
    attention_basis = _attention_basis_pair(outgoing_sample, incoming_sample)
    if attention_basis is None:
        reasons.append("attention_basis_not_compatible_or_verified")

    # Recompute the scalar gates from the exact samples instead of trusting a
    # detached aggregate copied from an older selection.  The selector still
    # owns admission and warning/severe policy; this local recomputation binds
    # the effect itself to the endpoint descriptors that will be reported.
    endpoint_position, endpoint_scale = _attention_mismatch(
        getattr(outgoing_sample, "attention", None),
        getattr(incoming_sample, "attention", None),
    )
    minimum_pose_confidence = float(
        getattr(
            getattr(visual_selection, "config", None),
            "minimum_pose_confidence",
            0.25,
        )
    )
    minimum_silhouette_confidence = float(
        getattr(
            getattr(visual_selection, "config", None),
            "minimum_silhouette_confidence",
            0.30,
        )
    )
    endpoint_pose, endpoint_formation, endpoint_silhouette = (
        _descriptor_shape_mismatches(
            getattr(outgoing_sample, "descriptor", None),
            getattr(incoming_sample, "descriptor", None),
            minimum_pose_confidence=minimum_pose_confidence,
            minimum_silhouette_confidence=minimum_silhouette_confidence,
        )
    )
    if endpoint_position is None or endpoint_scale is None:
        reasons.append("attention_position_or_scale_evidence_missing")
    else:
        if endpoint_position > config.maximum_attention_position_mismatch + 1e-9:
            reasons.append("attention_position_mismatch_exceeds_dissolve_bound")
        if endpoint_scale > config.maximum_attention_scale_mismatch + 1e-9:
            reasons.append("attention_scale_mismatch_exceeds_dissolve_bound")
    shape_values = tuple(
        value for value in (endpoint_pose, endpoint_formation, endpoint_silhouette)
        if value is not None
    )
    if len(shape_values) < config.minimum_endpoint_shape_component_count:
        reasons.append("pose_formation_silhouette_evidence_insufficient")
    else:
        for value, maximum, reason in (
            (endpoint_pose, config.maximum_pose_mismatch, "pose_mismatch_exceeds_dissolve_bound"),
            (
                endpoint_formation,
                config.maximum_formation_mismatch,
                "formation_mismatch_exceeds_dissolve_bound",
            ),
            (
                endpoint_silhouette,
                config.maximum_silhouette_mismatch,
                "silhouette_mismatch_exceeds_dissolve_bound",
            ),
        ):
            if value is not None and value > maximum + 1e-9:
                reasons.append(reason)

    motion = getattr(mismatch, "motion_connection", None)
    both_moving = bool(getattr(motion, "both_moving", False))
    if motion is None or not bool(getattr(motion, "timestamp_bound_adjacent", False)):
        reasons.append("adjacent_motion_evidence_missing_or_nonexact")
    if canonical_master is not None and motion is not None:
        exact_motion_times = (
            (
                "outgoing_previous_master_seconds",
                canonical_master - 2.0 * step,
            ),
            (
                "outgoing_endpoint_master_seconds",
                canonical_master - step,
            ),
            (
                "incoming_start_master_seconds",
                canonical_master,
            ),
            (
                "incoming_next_master_seconds",
                canonical_master + step,
            ),
        )
        if any(
            (value := _finite_float(getattr(motion, name, None))) is None
            or abs(value - expected) > 0.26 * step
            for name, expected in exact_motion_times
        ):
            reasons.append("adjacent_motion_timestamps_not_output_frame_bound")
        outgoing_endpoint_local = _finite_float(
            getattr(motion, "outgoing_endpoint_source_local_seconds", None)
        )
        incoming_start_local = _finite_float(
            getattr(motion, "incoming_start_source_local_seconds", None)
        )
        outgoing_previous_local = _finite_float(
            getattr(motion, "outgoing_previous_source_local_seconds", None)
        )
        incoming_next_local = _finite_float(
            getattr(motion, "incoming_next_source_local_seconds", None)
        )
        outgoing_sample_local = _finite_float(
            getattr(outgoing_sample, "source_local_seconds", None)
        )
        incoming_sample_local = _finite_float(
            getattr(incoming_sample, "source_local_seconds", None)
        )
        if (
            outgoing_endpoint_local is None
            or incoming_start_local is None
            or outgoing_previous_local is None
            or incoming_next_local is None
            or outgoing_sample_local is None
            or incoming_sample_local is None
            or abs(outgoing_endpoint_local - outgoing_sample_local) > 1e-7
            or abs(incoming_start_local - incoming_sample_local) > 1e-7
            or outgoing_previous_local >= outgoing_endpoint_local - 1e-9
            or incoming_next_local <= incoming_start_local + 1e-9
        ):
            reasons.append("adjacent_motion_source_clock_binding_invalid")
    motion_endpoint_components = tuple(
        value
        for value in (endpoint_pose, endpoint_silhouette, endpoint_scale)
        if value is not None
    )
    motion_endpoint_mismatch = (
        fmean(motion_endpoint_components) if motion_endpoint_components else None
    )
    if (
        len(motion_endpoint_components)
        < config.minimum_motion_endpoint_component_count
    ):
        reasons.append("motion_endpoint_shape_evidence_insufficient")
    elif (
        motion_endpoint_mismatch is None
        or motion_endpoint_mismatch
        > config.maximum_motion_endpoint_shape_mismatch + 1e-9
    ):
        reasons.append("motion_endpoint_shape_mismatch_exceeds_dissolve_bound")
    if both_moving:
        if (
            int(getattr(motion, "outgoing_motion_component_count", 0)) < 2
            or int(getattr(motion, "incoming_motion_component_count", 0)) < 2
        ):
            reasons.append("both_moving_component_evidence_insufficient")
        speed_mismatch = _finite_float(
            getattr(motion, "motion_speed_ratio_mismatch", None)
        )
        speed_maximum = float(
            getattr(
                getattr(visual_selection, "config", None),
                "maximum_motion_speed_ratio_mismatch",
                0.60,
            )
        )
        if (
            speed_mismatch is not None
            and speed_mismatch > speed_maximum + 1e-9
        ):
            reasons.append("motion_speed_discontinuity_exceeds_dissolve_bound")
        phase_mismatch = _finite_float(
            getattr(motion, "pose_phase_continuation_mismatch", None)
        )
        phase_maximum = float(
            getattr(
                getattr(visual_selection, "config", None),
                "maximum_pose_phase_continuation_mismatch",
                0.34,
            )
        )
        if (
            phase_mismatch is not None
            and phase_mismatch > phase_maximum + 1e-9
        ):
            reasons.append("pose_phase_discontinuity_exceeds_dissolve_bound")
    elif shape_values and 1.0 - fmean(shape_values) < config.minimum_static_shape_strength:
        reasons.append("static_pose_connection_not_strong_enough")

    spans = tuple(getattr(visual_selection, "span_validity", ()))
    next_span = (
        spans[boundary_index + 1]
        if len(spans) > boundary_index + 1
        else None
    )
    outgoing_future = (
        _span_source(next_span, left_source)
        if next_span is not None
        else None
    )
    incoming_future = (
        _span_source(next_span, right_source)
        if next_span is not None
        else None
    )
    exact_outgoing_handle = tuple(
        getattr(diagnostic, "outgoing_handle_samples", ())
    )
    exact_incoming_handle = tuple(
        getattr(diagnostic, "incoming_handle_samples", ())
    )
    if len(exact_outgoing_handle) == len(exact_incoming_handle) == 2:
        # Local beat audition may move the selected cut by several frames.
        # Bind the effect handle to samples decoded from that selected cut,
        # not to the original coarse span's +1-frame/+0.20s samples.
        future_pairs = tuple(
            zip(
                exact_outgoing_handle,
                exact_incoming_handle,
                strict=True,
            )
        )
    else:
        if outgoing_future is None or incoming_future is None:
            reasons.append("outgoing_handle_future_evidence_missing")
            future_pairs = ()
        else:
            outgoing_samples = tuple(
                getattr(outgoing_future, "samples", ())
            )
            incoming_samples = tuple(
                getattr(incoming_future, "samples", ())
            )
            future_pairs = tuple(
                zip(
                    outgoing_samples[:2],
                    incoming_samples[:2],
                    strict=False,
                )
            )
            if len(future_pairs) != 2:
                reasons.append("outgoing_handle_future_evidence_incomplete")

    future_positions: list[float] = []
    future_scales: list[float] = []
    future_poses: list[float] = []
    future_formations: list[float] = []
    future_silhouettes: list[float] = []
    for pair_index, (outgoing_future_sample, incoming_future_sample) in enumerate(future_pairs):
        left_master = _finite_float(
            getattr(outgoing_future_sample, "master_seconds", None)
        )
        right_master = _finite_float(
            getattr(incoming_future_sample, "master_seconds", None)
        )
        expected_delta = (
            step if pair_index == 0 else config.handle_sample_horizon_seconds
        )
        timestamp_tolerance = (
            0.26 * step if pair_index == 0 else max(step, 0.04)
        )
        if (
            left_master is None
            or right_master is None
            or abs(left_master - right_master) > 1e-7
            or canonical_master is None
            or left_master <= canonical_master
            or abs(left_master - (canonical_master + expected_delta))
            > timestamp_tolerance
        ):
            reasons.append("outgoing_handle_future_timestamps_not_exact")
        outgoing_future_local = _finite_float(
            getattr(outgoing_future_sample, "source_local_seconds", None)
        )
        incoming_future_local = _finite_float(
            getattr(incoming_future_sample, "source_local_seconds", None)
        )
        expected_outgoing_future_local = (
            outgoing_segment.source_end
            + expected_delta * outgoing_segment.source_playback_rate
        )
        expected_incoming_future_local = (
            incoming_segment.source_start
            + expected_delta * incoming_segment.source_playback_rate
        )
        if (
            outgoing_future_local is None
            or incoming_future_local is None
            or abs(
                outgoing_future_local - expected_outgoing_future_local
            )
            > max(
                1e-4,
                0.35 * step * outgoing_segment.source_playback_rate,
            )
            or abs(incoming_future_local - expected_incoming_future_local)
            > max(
                1e-4,
                0.35 * step * incoming_segment.source_playback_rate,
            )
        ):
            reasons.append("outgoing_handle_future_not_render_source_clock_bound")
        if _attention_basis_pair(outgoing_future_sample, incoming_future_sample) != attention_basis:
            reasons.append("outgoing_handle_attention_basis_changed")
        left_descriptor = getattr(outgoing_future_sample, "descriptor", None)
        right_descriptor = getattr(incoming_future_sample, "descriptor", None)
        if left_descriptor is None or right_descriptor is None:
            reasons.append("outgoing_handle_future_descriptor_missing")
            continue
        if (
            float(getattr(left_descriptor, "quality_score", 0.0))
            < config.minimum_handle_quality_score
            or float(getattr(right_descriptor, "quality_score", 0.0))
            < config.minimum_handle_quality_score
        ):
            reasons.append("outgoing_handle_future_quality_too_low")
        position, scale = _attention_mismatch(
            getattr(outgoing_future_sample, "attention", None),
            getattr(incoming_future_sample, "attention", None),
        )
        if position is None or scale is None:
            reasons.append("outgoing_handle_future_attention_missing")
        else:
            future_positions.append(position)
            future_scales.append(scale)
        pose, formation, silhouette = _descriptor_shape_mismatches(
            left_descriptor,
            right_descriptor,
            minimum_pose_confidence=minimum_pose_confidence,
            minimum_silhouette_confidence=minimum_silhouette_confidence,
        )
        available_future_shapes = tuple(
            value for value in (pose, formation, silhouette) if value is not None
        )
        if (
            len(available_future_shapes)
            < config.minimum_handle_shape_component_count
        ):
            reasons.append("outgoing_handle_future_shape_evidence_insufficient")
        if pose is not None:
            future_poses.append(pose)
        if formation is not None:
            future_formations.append(formation)
        if silhouette is not None:
            future_silhouettes.append(silhouette)
        if pair_index == 0:
            handle_entry_position, handle_entry_scale = _attention_mismatch(
                getattr(outgoing_sample, "attention", None),
                getattr(outgoing_future_sample, "attention", None),
            )
            handle_entry_pose, handle_entry_formation, handle_entry_silhouette = (
                _descriptor_shape_mismatches(
                    getattr(outgoing_sample, "descriptor", None),
                    left_descriptor,
                    minimum_pose_confidence=minimum_pose_confidence,
                    minimum_silhouette_confidence=(
                        minimum_silhouette_confidence
                    ),
                )
            )
            handle_entry_shapes = tuple(
                value
                for value in (
                    handle_entry_pose,
                    handle_entry_formation,
                    handle_entry_silhouette,
                )
                if value is not None
            )
            if (
                _attention_basis_pair(
                    outgoing_sample,
                    outgoing_future_sample,
                )
                != attention_basis
                or handle_entry_position is None
                or handle_entry_scale is None
                or handle_entry_position
                > config.maximum_handle_entry_attention_position_jump + 1e-9
                or handle_entry_scale
                > config.maximum_handle_entry_attention_scale_jump + 1e-9
                or len(handle_entry_shapes)
                < config.minimum_handle_shape_component_count
                or fmean(handle_entry_shapes)
                > config.maximum_handle_entry_shape_jump + 1e-9
            ):
                reasons.append("outgoing_handle_entry_discontinuity")

    future_position_max = max(future_positions) if future_positions else None
    future_scale_max = max(future_scales) if future_scales else None
    future_pose_max = max(future_poses) if future_poses else None
    future_formation_max = max(future_formations) if future_formations else None
    future_silhouette_max = max(future_silhouettes) if future_silhouettes else None
    for value, maximum, reason in (
        (
            future_position_max,
            config.maximum_future_attention_position_mismatch,
            "outgoing_handle_attention_position_diverges",
        ),
        (
            future_scale_max,
            config.maximum_future_attention_scale_mismatch,
            "outgoing_handle_attention_scale_diverges",
        ),
        (future_pose_max, config.maximum_future_pose_mismatch, "outgoing_handle_pose_diverges"),
        (
            future_formation_max,
            config.maximum_future_formation_mismatch,
            "outgoing_handle_formation_diverges",
        ),
        (
            future_silhouette_max,
            config.maximum_future_silhouette_mismatch,
            "outgoing_handle_silhouette_diverges",
        ),
    ):
        if value is not None and value > maximum + 1e-9:
            reasons.append(reason)

    if canonical_master is not None and outgoing_future is not None:
        native_cuts = tuple(
            float(value)
            for value in getattr(outgoing_future, "native_scene_cut_master_seconds", ())
        )
        native_source_cuts = tuple(
            float(value)
            for value in getattr(outgoing_future, "native_scene_cut_source_seconds", ())
        )
    else:
        native_cuts = ()
        native_source_cuts = ()
    outgoing_motion = _finite_float(
        getattr(motion, "outgoing_motion_magnitude", None)
    ) or 0.0
    incoming_motion = _finite_float(
        getattr(motion, "incoming_motion_magnitude", None)
    ) or 0.0
    motion_strength = min(1.0, min(outgoing_motion, incoming_motion) / 0.08) if both_moving else 0.0
    highlight = bool(
        canonical_master is not None
        and _in_highlight(canonical_master, highlight_intervals)
    )
    transition_frames = _transition_frame_count(
        fps=fps,
        both_moving=both_moving,
        highlight=highlight,
        motion_strength=motion_strength,
        config=config,
    )
    transition_duration = transition_frames / fps
    if canonical_master is not None and any(
        canonical_master - 1e-9 < value < canonical_master + transition_duration - 1e-9
        for value in native_cuts
    ):
        reasons.append("outgoing_handle_crosses_native_source_cut")

    incoming_frame_count = frame_counts[boundary_index + 1]
    available_handle = (
        eligible_source_durations_seconds[left_source] - outgoing_segment.source_end
    )
    required_handle = transition_duration * outgoing_segment.source_playback_rate
    if any(
        outgoing_segment.source_end - 1e-7
        < value
        < outgoing_segment.source_end + required_handle - 1e-7
        for value in native_source_cuts
    ):
        reasons.append("outgoing_handle_crosses_native_source_cut")
    if available_handle + 1e-7 < required_handle:
        reasons.append("outgoing_source_handle_too_short")
    if incoming_frame_count < transition_frames:
        reasons.append("incoming_segment_shorter_than_dissolve")

    reasons = list(dict.fromkeys(reasons))
    if reasons:
        return (None, tuple(reasons))

    assert canonical_master is not None
    assert attention_basis is not None
    assert endpoint_position is not None and endpoint_scale is not None
    assert future_position_max is not None and future_scale_max is not None
    shape_quality = 1.0 - fmean(shape_values)
    attention_quality = 1.0 - fmean((endpoint_position, endpoint_scale))
    future_quality_values = tuple(
        value
        for value in (
            future_position_max,
            future_scale_max,
            future_pose_max,
            future_formation_max,
            future_silhouette_max,
        )
        if value is not None
    )
    future_quality = 1.0 - fmean(future_quality_values)
    priority = (
        0.32 * attention_quality
        + 0.28 * shape_quality
        + 0.20 * future_quality
        + (0.16 if both_moving else 0.0)
        + (0.12 if highlight else 0.0)
        + (0.05 if attention_basis == _FACE_ATTENTION_BASIS else 0.0)
    )
    return (
        _SafeCandidate(
            boundary_index=boundary_index,
            canonical_master_seconds=canonical_master,
            priority_score=priority,
            highlight_priority=highlight,
            both_moving=both_moving,
            attention_basis=attention_basis,
            endpoint_attention_position_mismatch=endpoint_position,
            endpoint_attention_scale_mismatch=endpoint_scale,
            endpoint_pose_mismatch=endpoint_pose,
            endpoint_formation_mismatch=endpoint_formation,
            endpoint_silhouette_mismatch=endpoint_silhouette,
            future_attention_position_max_mismatch=future_position_max,
            future_attention_scale_max_mismatch=future_scale_max,
            future_pose_max_mismatch=future_pose_max,
            future_formation_max_mismatch=future_formation_max,
            future_silhouette_max_mismatch=future_silhouette_max,
            outgoing_handle_available_seconds=available_handle,
            required_outgoing_handle_seconds=required_handle,
            incoming_segment_frame_count=incoming_frame_count,
            transition_frame_count=transition_frames,
        ),
        (),
    )


def _required_cinematic_bridge_candidate(
    *,
    boundary_index: int,
    diagnostic: Any,
    visual_selection: DirectMemberVisualSelection,
    render_segments: tuple[RenderSegment, ...],
    eligible_request_source_indices: tuple[int, ...],
    eligible_source_durations_seconds: tuple[float, ...],
    frame_counts: tuple[int, ...],
    common_master_start: float,
    fps: float,
    highlight_intervals: tuple[tuple[float, float], ...],
    config: DirectMemberPoseDissolveConfig,
) -> tuple[
    _RequiredCinematicBridgeCandidate | None,
    tuple[str, ...],
]:
    """Bind a selector-required effect to exact render clocks and handles."""

    reasons: list[str] = []
    mismatch = getattr(diagnostic, "mismatch", None)
    bridge = getattr(mismatch, "cinematic_bridge", None)
    if bridge is None:
        return (None, ("cinematic_bridge_contract_missing",))
    if (
        getattr(visual_selection, "usage_policy", None)
        != "heterogeneous_role_aware_cinematic_v1"
    ):
        reasons.append("cinematic_bridge_usage_authority_missing")
    if getattr(mismatch, "transform_repair", None) is not None:
        reasons.append("cinematic_bridge_conflicts_with_transform_repair")
    severe_flags = tuple(getattr(mismatch, "severe_flags", ()))
    if (
        not severe_flags
        or frozenset(severe_flags)
        != frozenset(getattr(bridge, "authorized_severe_flags", ()))
    ):
        reasons.append("cinematic_bridge_severe_authority_mismatch")

    expected_master = (
        common_master_start + render_segments[boundary_index + 1].master_start
    )
    canonical_master = _diagnostic_cut_master_seconds(diagnostic)
    highlight_priority = False
    if (
        getattr(diagnostic, "boundary_index", None) != boundary_index
        or canonical_master is None
        or abs(canonical_master - expected_master) > 1e-6
    ):
        reasons.append("stale_or_nonexact_boundary_diagnostic")
    else:
        highlight_priority = _in_highlight(
            canonical_master,
            highlight_intervals,
        )

    bridge_pose_mismatch = _finite_float(
        getattr(bridge, "pose_mismatch", None)
    )
    bridge_silhouette_mismatch = _finite_float(
        getattr(bridge, "silhouette_mismatch", None)
    )
    bridge_base_mismatch = _finite_float(
        getattr(bridge, "base_mismatch", None)
    )
    focal_bridge = bool(
        isinstance(bridge, DirectMemberCinematicBridge)
        and bridge.evidence_basis == FOCAL_POSE_MOTION_BRIDGE_BASIS
        and getattr(mismatch, "pose_basis", None) == FOCAL_SELECTOR_POSE_BASIS
    )
    if getattr(bridge, "evidence_basis", None) == FOCAL_POSE_MOTION_BRIDGE_BASIS and not focal_bridge:
        reasons.append("focal_bridge_typed_pose_authority_missing")
    if focal_bridge:
        if not verified_focal_pose_motion_bridge(getattr(mismatch, "motion_connection", None)):
            reasons.append("focal_bridge_motion_measurements_unverified")
        focal_pair = tuple(getattr(getattr(diagnostic, name, None), "focal_geometry", None)
                           for name in ("outgoing_sample", "incoming_sample"))
        measured = (focal_pose_shape_mismatch(*focal_pair)
                    if all(isinstance(value, FocalSelectorGeometry) for value in focal_pair) else None)
        if measured is None or abs(measured-bridge.pose_mismatch)>1e-9:
            reasons.append("focal_bridge_endpoint_pose_binding_invalid")
    if (
        bridge_pose_mismatch is None
        or bridge_pose_mismatch
        > _CINEMATIC_BRIDGE_MAXIMUM_POSE_MISMATCH + 1e-9
    ):
        reasons.append("cinematic_bridge_pose_mismatch_unverified")
    if (
        not focal_bridge and (bridge_silhouette_mismatch is None
        or bridge_silhouette_mismatch
        > _CINEMATIC_BRIDGE_MAXIMUM_SILHOUETTE_MISMATCH + 1e-9)
    ):
        reasons.append("cinematic_bridge_silhouette_mismatch_unverified")
    if (
        bridge_base_mismatch is None
        or bridge_base_mismatch
        > _CINEMATIC_BRIDGE_MAXIMUM_BASE_MISMATCH + 1e-9
    ):
        reasons.append("cinematic_bridge_base_mismatch_unverified")
    if canonical_master is not None and not highlight_priority:
        if (
            bridge_pose_mismatch is None
            or bridge_pose_mismatch
            > _NONPRIORITY_CINEMATIC_BRIDGE_MAXIMUM_POSE_MISMATCH + 1e-9
        ):
            reasons.append("cinematic_bridge_nonpriority_pose_mismatch")
        if (
            not focal_bridge and (bridge_silhouette_mismatch is None
            or bridge_silhouette_mismatch
            > _NONPRIORITY_CINEMATIC_BRIDGE_MAXIMUM_SILHOUETTE_MISMATCH
            + 1e-9)
        ):
            reasons.append("cinematic_bridge_nonpriority_silhouette_mismatch")
        if (
            bridge_base_mismatch is None
            or bridge_base_mismatch
            > _NONPRIORITY_CINEMATIC_BRIDGE_MAXIMUM_BASE_MISMATCH + 1e-9
        ):
            reasons.append("cinematic_bridge_nonpriority_base_mismatch")

    source_path = tuple(getattr(visual_selection, "source_indices", ()))
    if len(source_path) != len(render_segments):
        return (
            None,
            tuple(dict.fromkeys(reasons + ["selected_source_path_length_mismatch"])),
        )
    left_source = source_path[boundary_index]
    right_source = source_path[boundary_index + 1]
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < len(eligible_request_source_indices)
        for value in (left_source, right_source)
    ):
        return (
            None,
            tuple(dict.fromkeys(reasons + ["selected_source_domain_invalid"])),
        )
    outgoing_segment = render_segments[boundary_index]
    incoming_segment = render_segments[boundary_index + 1]
    if (
        left_source == right_source
        or not bool(getattr(diagnostic, "cut_applied", True))
    ):
        reasons.append("cinematic_bridge_requires_real_source_change")
    if (
        getattr(diagnostic, "outgoing_eligible_source_index", None)
        != left_source
        or getattr(diagnostic, "incoming_eligible_source_index", None)
        != right_source
        or outgoing_segment.source_index
        != eligible_request_source_indices[left_source]
        or incoming_segment.source_index
        != eligible_request_source_indices[right_source]
    ):
        reasons.append("boundary_source_metadata_disagrees_with_render")

    style = getattr(bridge, "style", None)
    if style == "pose_dissolve":
        transition_kind = TransitionKind.FACE_DISSOLVE
        transition_reason = DIRECT_MEMBER_POSE_DISSOLVE_REASON
    elif style == "zoom_in":
        transition_kind = TransitionKind.MEMBER_ZOOM_BRIDGE
        transition_reason = MEMBER_ZOOM_BRIDGE_REASON
    elif style == "horizontal_blur":
        transition_kind = TransitionKind.MEMBER_BLUR_BRIDGE
        transition_reason = MEMBER_BLUR_BRIDGE_REASON
    else:
        reasons.append("cinematic_bridge_style_invalid")
        transition_kind = TransitionKind.HARD_CUT
        transition_reason = "invalid_required_cinematic_bridge"

    step = 1.0 / fps
    outgoing_sample = getattr(diagnostic, "outgoing_sample", None)
    incoming_sample = getattr(diagnostic, "incoming_sample", None)
    attention_basis = _required_bridge_attention_basis(
        outgoing_sample,
        incoming_sample,
        bridge,
    )
    if attention_basis is None:
        reasons.append("cinematic_bridge_attention_basis_unverified")
    if canonical_master is not None:
        outgoing_master = _finite_float(
            getattr(outgoing_sample, "master_seconds", None)
        )
        incoming_master = _finite_float(
            getattr(incoming_sample, "master_seconds", None)
        )
        outgoing_local = _finite_float(
            getattr(outgoing_sample, "source_local_seconds", None)
        )
        incoming_local = _finite_float(
            getattr(incoming_sample, "source_local_seconds", None)
        )
        if (
            outgoing_master is None
            or incoming_master is None
            or abs(outgoing_master - (canonical_master - step)) > 0.26 * step
            or abs(incoming_master - canonical_master) > 0.26 * step
        ):
            reasons.append("cinematic_bridge_endpoint_samples_not_frame_bound")
        if (
            outgoing_local is None
            or incoming_local is None
            or abs(
                outgoing_local
                - (
                    outgoing_segment.source_end
                    - step * outgoing_segment.source_playback_rate
                )
            )
            > max(1e-4, 0.35 * step * outgoing_segment.source_playback_rate)
            or abs(
                incoming_local
                - (
                    incoming_segment.source_start
                )
            )
            > max(1e-4, 0.35 * step * incoming_segment.source_playback_rate)
        ):
            reasons.append("cinematic_bridge_endpoint_source_clock_invalid")

    motion = getattr(mismatch, "motion_connection", None)
    both_moving = bool(getattr(motion, "both_moving", False))
    if motion is None or not bool(
        getattr(motion, "timestamp_bound_adjacent", False)
    ):
        reasons.append("cinematic_bridge_adjacent_motion_evidence_missing")
    elif canonical_master is not None:
        exact_motion_times = (
            ("outgoing_previous_master_seconds", canonical_master - 2.0 * step),
            ("outgoing_endpoint_master_seconds", canonical_master - step),
            ("incoming_start_master_seconds", canonical_master),
            ("incoming_next_master_seconds", canonical_master + step),
        )
        if any(
            (value := _finite_float(getattr(motion, name, None))) is None
            or abs(value - expected) > 0.26 * step
            for name, expected in exact_motion_times
        ):
            reasons.append("cinematic_bridge_motion_timestamps_not_frame_bound")
        if (
            _finite_float(
                getattr(motion, "outgoing_endpoint_source_local_seconds", None)
            )
            != _finite_float(
                getattr(outgoing_sample, "source_local_seconds", None)
            )
            or _finite_float(
                getattr(motion, "incoming_start_source_local_seconds", None)
            )
            != _finite_float(
                getattr(incoming_sample, "source_local_seconds", None)
            )
        ):
            reasons.append("cinematic_bridge_motion_source_clock_invalid")
    if style == "pose_dissolve":
        direction_cosine = _finite_float(
            getattr(motion, "direction_cosine_audit_only", None)
        )
        endpoint_shape_mismatch = _finite_float(
            getattr(motion, "endpoint_pose_shape_mismatch", None)
        )
        speed_ratio_mismatch = _finite_float(
            getattr(motion, "motion_speed_ratio_mismatch", None)
        )
        phase_mismatch = _finite_float(
            getattr(motion, "pose_phase_continuation_mismatch", None)
        )
        if not both_moving:
            reasons.append("pose_dissolve_bridge_requires_both_moving")
        if (
            direction_cosine is None
            or direction_cosine
            < _POSE_DISSOLVE_BRIDGE_MINIMUM_DIRECTION_COSINE - 1e-9
        ):
            reasons.append("pose_dissolve_bridge_direction_unverified")
        if (
            endpoint_shape_mismatch is None
            or endpoint_shape_mismatch
            > _POSE_DISSOLVE_BRIDGE_MAXIMUM_ENDPOINT_SHAPE_MISMATCH + 1e-9
        ):
            reasons.append("pose_dissolve_bridge_endpoint_shape_unverified")
        if (
            speed_ratio_mismatch is not None
            and speed_ratio_mismatch
            > _POSE_DISSOLVE_BRIDGE_MAXIMUM_SPEED_RATIO_MISMATCH + 1e-9
        ):
            reasons.append("pose_dissolve_bridge_speed_discontinuity")
        if (
            phase_mismatch is not None
            and phase_mismatch
            > _POSE_DISSOLVE_BRIDGE_MAXIMUM_PHASE_MISMATCH + 1e-9
        ):
            reasons.append("pose_dissolve_bridge_phase_discontinuity")

    minimum_frames = max(
        3,
        int(math.ceil(config.minimum_duration_seconds * fps - 1e-12)),
    )
    maximum_frames = max(
        minimum_frames,
        int(math.floor(config.maximum_duration_seconds * fps + 1e-12)),
    )
    transition_frames = min(
        maximum_frames,
        max(minimum_frames, int(round(0.22 * fps))),
    )
    transition_duration = transition_frames / fps
    spans = tuple(getattr(visual_selection, "span_validity", ()))
    next_span = (
        spans[boundary_index + 1]
        if len(spans) > boundary_index + 1
        else None
    )
    outgoing_future = (
        _span_source(next_span, left_source) if next_span is not None else None
    )
    incoming_future = (
        _span_source(next_span, right_source) if next_span is not None else None
    )
    if outgoing_future is None or incoming_future is None:
        reasons.append("cinematic_bridge_future_content_evidence_missing")
    else:
        exact_outgoing_handle = tuple(
            getattr(diagnostic, "outgoing_handle_samples", ())
        )
        exact_incoming_handle = tuple(
            getattr(diagnostic, "incoming_handle_samples", ())
        )
        exact_handle_bound = bool(
            len(exact_outgoing_handle) == len(exact_incoming_handle) == 2
        )
        for label, future, segment in (
            ("outgoing", outgoing_future, outgoing_segment),
            ("incoming", incoming_future, incoming_segment),
        ):
            # The outgoing source is not rendered for the following full span;
            # only its short overlap handle is.  Whole-span invalidity can be
            # caused by a later obstruction or native cut and must not erase a
            # frame-proven 0.22 s handle.  The selected incoming source still
            # requires ordinary full-span validity.
            if (
                label == "incoming"
                and not bool(getattr(future, "valid", False))
            ):
                reasons.append(f"cinematic_bridge_{label}_future_content_invalid")
            samples = (
                exact_outgoing_handle
                if exact_handle_bound and label == "outgoing"
                else exact_incoming_handle
                if exact_handle_bound
                else tuple(getattr(future, "samples", ()))[:2]
            )
            if len(samples) != 2:
                reasons.append(f"cinematic_bridge_{label}_future_samples_incomplete")
                continue
            reasons.extend(member_bridge_handle_quality_failures(
                tuple(getattr(getattr(sample, "descriptor", None), "quality_score", None)
                      for sample in samples), label=label,
                minimum_quality=config.minimum_handle_quality_score))
            for sample in samples:
                sample_master = _finite_float(
                    getattr(sample, "master_seconds", None)
                )
                sample_local = _finite_float(
                    getattr(sample, "source_local_seconds", None)
                )
                if (
                    canonical_master is None
                    or sample_master is None
                    or sample_local is None
                    or sample_master <= canonical_master
                    or sample_master
                    > canonical_master
                    + max(transition_duration, config.handle_sample_horizon_seconds)
                    + 0.05
                ):
                    reasons.append(
                        f"cinematic_bridge_{label}_future_timestamp_invalid"
                    )
                    continue
                expected_local = (
                    segment.source_end
                    + (sample_master - canonical_master)
                    * segment.source_playback_rate
                    if label == "outgoing"
                    else segment.source_start
                    + (sample_master - canonical_master)
                    * segment.source_playback_rate
                )
                if abs(sample_local - expected_local) > max(
                    1e-4,
                    0.35 * step * segment.source_playback_rate,
                ):
                    reasons.append(
                        f"cinematic_bridge_{label}_future_source_clock_invalid"
                    )
            if canonical_master is not None and any(
                canonical_master - 1e-9
                < float(value)
                < canonical_master + transition_duration - 1e-9
                for value in getattr(
                    future,
                    "native_scene_cut_master_seconds",
                    (),
                )
            ):
                reasons.append(
                    f"cinematic_bridge_{label}_crosses_native_source_cut"
                )

    if focal_bridge:
        # The full-span screening samples intentionally carry no target pose.
        # Require the exact, target-bound overlap observations, never borrow
        # another member's aggregate silhouette/formation to fill that gap.
        handles = (tuple(getattr(diagnostic, "outgoing_handle_samples", ())),
                   tuple(getattr(diagnostic, "incoming_handle_samples", ())))
        if any(len(samples) != 2 for samples in handles):
            reasons.append("focal_bridge_exact_handle_pose_missing")
        else:
            for left, right in zip(*handles):
                geometry = tuple(getattr(sample, "focal_geometry", None) for sample in (left, right))
                pose = (focal_pose_shape_mismatch(*geometry)
                        if all(isinstance(value, FocalSelectorGeometry) for value in geometry) else None)
                if (_attention_basis_pair(left, right) != attention_basis
                    or pose is None or pose > config.maximum_future_pose_mismatch + 1e-9
                    or common_focal_pose_vectors(geometry) is None):
                    reasons.append("focal_bridge_handle_pose_unverified")

    available_handle = (
        eligible_source_durations_seconds[left_source]
        - outgoing_segment.source_end
    )
    required_handle = (
        transition_duration * outgoing_segment.source_playback_rate
    )
    if available_handle + 1e-7 < required_handle:
        reasons.append("cinematic_bridge_outgoing_source_handle_too_short")
    incoming_frame_count = frame_counts[boundary_index + 1]
    if incoming_frame_count < transition_frames:
        reasons.append("cinematic_bridge_incoming_segment_too_short")

    reasons = list(dict.fromkeys(reasons))
    if reasons:
        return (None, tuple(reasons))
    assert canonical_master is not None
    assert attention_basis is not None
    assert bridge_pose_mismatch is not None
    assert focal_bridge or bridge_silhouette_mismatch is not None
    return (
        _RequiredCinematicBridgeCandidate(
            boundary_index=boundary_index,
            canonical_master_seconds=canonical_master,
            style=str(style),
            verified_focal_articulation=verified_focal_pose_dissolve_spacing(mismatch),
            transition_kind=transition_kind,
            transition_reason=transition_reason,
            highlight_priority=highlight_priority,
            both_moving=both_moving,
            attention_basis=attention_basis,
            endpoint_attention_position_mismatch=(
                _finite_float(
                    getattr(mismatch, "attention_position_mismatch", None)
                )
            ),
            endpoint_attention_scale_mismatch=(
                _finite_float(
                    getattr(mismatch, "attention_scale_mismatch", None)
                )
            ),
            endpoint_pose_mismatch=bridge_pose_mismatch,
            endpoint_formation_mismatch=(
                _finite_float(getattr(mismatch, "formation_mismatch", None))
            ),
            endpoint_silhouette_mismatch=bridge_silhouette_mismatch,
            outgoing_handle_available_seconds=available_handle,
            required_outgoing_handle_seconds=required_handle,
            incoming_segment_frame_count=incoming_frame_count,
            transition_frame_count=transition_frames,
        ),
        (),
    )


def _choose_candidates(
    candidates: Sequence[_SafeCandidate],
    *,
    maximum_count: int,
    config: DirectMemberPoseDissolveConfig,
) -> frozenset[int]:
    ordered = tuple(sorted(candidates, key=lambda item: item.boundary_index))
    if maximum_count <= 0 or not ordered:
        return frozenset()

    @lru_cache(maxsize=None)
    def solve(position: int, remaining: int) -> tuple[int, float, tuple[int, ...]]:
        if position >= len(ordered) or remaining <= 0:
            return (0, 0.0, ())
        skipped = solve(position + 1, remaining)
        current = ordered[position]
        next_position = position + 1
        while next_position < len(ordered):
            following = ordered[next_position]
            if (
                following.boundary_index
                > current.boundary_index + config.minimum_boundary_gap
                and following.canonical_master_seconds
                - current.canonical_master_seconds
                >= config.minimum_event_spacing_seconds - 1e-9
            ):
                break
            next_position += 1
        suffix = solve(next_position, remaining - 1)
        selected = (
            suffix[0] + 1,
            suffix[1] + current.priority_score,
            (current.boundary_index,) + suffix[2],
        )
        # Cardinality is intentional: once a boundary has passed every visual
        # and handle gate, use the predeclared density budget instead of
        # collapsing back to one or two ultra-conservative effects.
        return max(
            (skipped, selected),
            key=lambda item: (item[0], item[1], tuple(-value for value in item[2])),
        )

    return frozenset(solve(0, maximum_count)[2])


def build_direct_member_pose_dissolve_plan(
    *,
    visual_selection: DirectMemberVisualSelection,
    render_segments: Sequence[RenderSegment],
    eligible_request_source_indices: Sequence[int],
    eligible_source_durations_seconds: Sequence[float] | None,
    common_master_start: float,
    source_metadata_contract: str | None,
    enabled: bool = True,
    highlight_intervals: Sequence[tuple[float, float]] = (),
    hard_cut_reason: str = "official_target_fancam_direct_full_frame",
    fps: float = 29.97,
    config: DirectMemberPoseDissolveConfig | None = None,
) -> DirectMemberPoseDissolvePlan:
    """Promote a bounded subset of exact direct-member boundaries.

    Missing production provenance, media-duration evidence, stale selection
    objects, warning-level geometry, or incomplete future-handle samples never
    raise the effect's confidence.  They produce ordinary hard cuts and an
    explicit audit reason.
    """

    renders = tuple(render_segments)
    if not renders:
        raise ValueError("direct member dissolve planner needs render segments")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("direct member dissolve planner fps must be positive")
    if not math.isfinite(common_master_start) or common_master_start < 0.0:
        raise ValueError("direct member dissolve common master start is invalid")
    if not isinstance(enabled, bool):
        raise TypeError("direct member dissolve enabled flag must be boolean")
    settings = config or DirectMemberPoseDissolveConfig()
    minimum_transition_frames = max(
        3,
        int(math.ceil(settings.minimum_duration_seconds * fps - 1e-12)),
    )
    maximum_transition_frames = int(
        math.floor(settings.maximum_duration_seconds * fps + 1e-12)
    )
    duration_quantization_valid = (
        minimum_transition_frames <= maximum_transition_frames
    )
    raw_eligible = tuple(eligible_request_source_indices)
    eligible_values_valid = bool(
        len(raw_eligible) >= 2
        and all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            for value in raw_eligible
        )
        and len(set(raw_eligible)) == len(raw_eligible)
    )
    eligible = tuple(raw_eligible) if eligible_values_valid else ()
    metadata_verified = (
        source_metadata_contract
        == OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT
    )
    try:
        raw_durations = (
            tuple(float(value) for value in eligible_source_durations_seconds)
            if eligible_source_durations_seconds is not None
            else ()
        )
    except (TypeError, ValueError):
        raw_durations = ()
    durations_valid = bool(
        len(raw_durations) == len(eligible)
        and all(math.isfinite(value) and value > 0.0 for value in raw_durations)
    )
    intervals = tuple((float(start), float(end)) for start, end in highlight_intervals)
    if any(
        not math.isfinite(start)
        or not math.isfinite(end)
        or start < 0.0
        or end <= start
        for start, end in intervals
    ):
        raise ValueError("direct member dissolve highlight intervals are invalid")
    frame_counts = _quantized_frame_counts(renders, fps)
    boundaries = tuple(getattr(visual_selection, "boundaries", ()))
    safe: list[_SafeCandidate] = []
    required_bridges: dict[int, _RequiredCinematicBridgeCandidate] = {}
    required_bridge_failures: dict[int, tuple[str, ...]] = {}
    hard_reasons: dict[int, tuple[str, ...]] = {}
    boundary_count = len(renders) - 1
    for boundary_index in range(boundary_count):
        reasons: list[str] = []
        if not enabled:
            reasons.append("effect_disabled_by_request")
        if not duration_quantization_valid:
            reasons.append("output_fps_cannot_represent_safe_dissolve_duration")
        if not metadata_verified:
            reasons.append("source_metadata_contract_unverified")
        if not durations_valid:
            reasons.append("source_duration_evidence_missing_or_invalid")
        if len(boundaries) != boundary_count:
            reasons.append("boundary_diagnostic_count_mismatch")
        if not eligible_values_valid:
            reasons.append("eligible_request_source_metadata_invalid")
        diagnostic = (
            boundaries[boundary_index]
            if boundary_index < len(boundaries)
            else None
        )
        mismatch = getattr(diagnostic, "mismatch", None)
        required_bridge = getattr(mismatch, "cinematic_bridge", None)
        if required_bridge is not None:
            if reasons:
                required_bridge_failures[boundary_index] = tuple(
                    dict.fromkeys(reasons)
                )
                continue
            bridge_candidate, bridge_reasons = (
                _required_cinematic_bridge_candidate(
                    boundary_index=boundary_index,
                    diagnostic=diagnostic,
                    visual_selection=visual_selection,
                    render_segments=renders,
                    eligible_request_source_indices=eligible,
                    eligible_source_durations_seconds=raw_durations,
                    frame_counts=frame_counts,
                    common_master_start=common_master_start,
                    fps=fps,
                    highlight_intervals=intervals,
                    config=settings,
                )
            )
            if bridge_candidate is None:
                required_bridge_failures[boundary_index] = bridge_reasons
            else:
                required_bridges[boundary_index] = bridge_candidate
            continue
        if reasons or boundary_index >= len(boundaries):
            hard_reasons[boundary_index] = tuple(dict.fromkeys(reasons))
            continue
        if getattr(mismatch, "transform_repair", None) is not None:
            hard_reasons[boundary_index] = (
                "lossless_transform_repair_requires_hard_cut",
            )
            continue
        candidate, candidate_reasons = _safe_candidate(
            boundary_index=boundary_index,
            diagnostic=boundaries[boundary_index],
            visual_selection=visual_selection,
            render_segments=renders,
            eligible_request_source_indices=eligible,
            eligible_source_durations_seconds=raw_durations,
            frame_counts=frame_counts,
            common_master_start=common_master_start,
            fps=fps,
            highlight_intervals=intervals,
            config=settings,
        )
        if candidate is None:
            hard_reasons[boundary_index] = candidate_reasons
        else:
            safe.append(candidate)

    ordered_required_bridges = tuple(
        sorted(
            required_bridges.values(),
            key=lambda candidate: (
                candidate.canonical_master_seconds,
                candidate.boundary_index,
            ),
        )
    )
    for previous, current in zip(
        ordered_required_bridges,
        ordered_required_bridges[1:],
    ):
        if (
            current.canonical_master_seconds
            - previous.canonical_master_seconds
            < required_member_bridge_spacing_seconds(
                previous.verified_focal_articulation, current.verified_focal_articulation,
                pose_minimum_seconds=settings.minimum_event_spacing_seconds) - 1e-9
        ):
            required_bridge_failures[current.boundary_index] = (
                "cinematic_bridge_required_event_spacing_too_short",
            )

    if required_bridge_failures:
        raise RequiredMemberBridgeAuthoringError(required_bridge_failures)

    if required_bridges:
        safe = [
            candidate
            for candidate in safe
            if all(
                candidate.boundary_index
                > bridge.boundary_index + settings.minimum_boundary_gap
                or bridge.boundary_index
                > candidate.boundary_index + settings.minimum_boundary_gap
                for bridge in required_bridges.values()
            )
            and all(
                abs(
                    candidate.canonical_master_seconds
                    - bridge.canonical_master_seconds
                )
                >= settings.minimum_event_spacing_seconds - 1e-9
                for bridge in required_bridges.values()
            )
        ]

    total_duration = sum(segment.duration for segment in renders)
    density_budget = max(
        1,
        int(round(total_duration * settings.target_dissolves_per_minute / 60.0)),
    )
    fraction_budget = max(
        1,
        int(math.floor(boundary_count * settings.maximum_boundary_fraction + 1e-12)),
    )
    remaining_density_budget = max(0, density_budget - len(required_bridges))
    remaining_fraction_budget = max(0, fraction_budget - len(required_bridges))
    maximum_count = min(
        len(safe),
        remaining_density_budget,
        remaining_fraction_budget,
    )
    selected_indices = _choose_candidates(
        safe,
        maximum_count=maximum_count,
        config=settings,
    )
    safe_by_index = {candidate.boundary_index: candidate for candidate in safe}

    transitions: list[RenderTransition] = []
    decisions: list[DirectMemberPoseDissolveDecision] = []
    for boundary_index in range(boundary_count):
        bridge_candidate = required_bridges.get(boundary_index)
        candidate = safe_by_index.get(boundary_index)
        selected = bool(
            bridge_candidate is not None
            or (
                candidate is not None
                and boundary_index in selected_indices
            )
        )
        if bridge_candidate is not None:
            bridge_face_values = tuple(
                value
                for value in (
                    bridge_candidate.endpoint_attention_position_mismatch,
                    bridge_candidate.endpoint_attention_scale_mismatch,
                )
                if value is not None
            )
            transition_kind = bridge_candidate.transition_kind
            transitions.append(
                RenderTransition(
                    kind=transition_kind,
                    frame_count=bridge_candidate.transition_frame_count,
                    reason=bridge_candidate.transition_reason,
                    face_mismatch=(
                        fmean(bridge_face_values)
                        if bridge_face_values
                        else None
                    ),
                    tier=("A" if bridge_candidate.highlight_priority else "B"),
                )
            )
            rejection_reasons = ()
        elif selected:
            assert candidate is not None
            transition_kind = TransitionKind.FACE_DISSOLVE
            transitions.append(
                RenderTransition(
                    kind=transition_kind,
                    frame_count=candidate.transition_frame_count,
                    reason=DIRECT_MEMBER_POSE_DISSOLVE_REASON,
                    face_mismatch=fmean(
                        (
                            candidate.endpoint_attention_position_mismatch,
                            candidate.endpoint_attention_scale_mismatch,
                        )
                    ),
                    tier=(
                        "A"
                        if candidate.both_moving and candidate.highlight_priority
                        else "B"
                    ),
                    future_pose_max_mismatch=candidate.future_pose_max_mismatch,
                    future_silhouette_max_mismatch=(
                        candidate.future_silhouette_max_mismatch
                    ),
                )
            )
            rejection_reasons = ()
        else:
            transition_kind = TransitionKind.HARD_CUT
            transitions.append(RenderTransition(reason=hard_cut_reason))
            rejection_reasons = hard_reasons.get(
                boundary_index,
                ("density_or_spacing_budget",),
            )
        evidence_candidate = (
            bridge_candidate if bridge_candidate is not None else candidate
        )
        decisions.append(
            DirectMemberPoseDissolveDecision(
                boundary_index=boundary_index,
                canonical_master_seconds=(
                    evidence_candidate.canonical_master_seconds
                    if evidence_candidate is not None
                    else common_master_start + renders[boundary_index + 1].master_start
                ),
                selected=selected,
                rejection_reasons=rejection_reasons,
                priority_score=(
                    candidate.priority_score
                    if candidate is not None
                    else 1.0
                    if bridge_candidate is not None
                    else 0.0
                ),
                highlight_priority=(
                    candidate.highlight_priority
                    if candidate is not None
                    else bridge_candidate.highlight_priority
                    if bridge_candidate is not None
                    else False
                ),
                both_moving=(
                    evidence_candidate.both_moving
                    if evidence_candidate is not None
                    else False
                ),
                attention_basis=(
                    evidence_candidate.attention_basis
                    if evidence_candidate is not None
                    else None
                ),
                endpoint_attention_position_mismatch=(
                    evidence_candidate.endpoint_attention_position_mismatch
                    if evidence_candidate is not None
                    else None
                ),
                endpoint_attention_scale_mismatch=(
                    evidence_candidate.endpoint_attention_scale_mismatch
                    if evidence_candidate is not None
                    else None
                ),
                endpoint_pose_mismatch=(
                    evidence_candidate.endpoint_pose_mismatch
                    if evidence_candidate is not None
                    else None
                ),
                endpoint_formation_mismatch=(
                    evidence_candidate.endpoint_formation_mismatch
                    if evidence_candidate is not None
                    else None
                ),
                endpoint_silhouette_mismatch=(
                    evidence_candidate.endpoint_silhouette_mismatch
                    if evidence_candidate is not None
                    else None
                ),
                future_attention_position_max_mismatch=(
                    candidate.future_attention_position_max_mismatch
                    if candidate is not None
                    else None
                ),
                future_attention_scale_max_mismatch=(
                    candidate.future_attention_scale_max_mismatch
                    if candidate is not None
                    else None
                ),
                future_pose_max_mismatch=(
                    candidate.future_pose_max_mismatch
                    if candidate is not None
                    else None
                ),
                future_formation_max_mismatch=(
                    candidate.future_formation_max_mismatch
                    if candidate is not None
                    else None
                ),
                future_silhouette_max_mismatch=(
                    candidate.future_silhouette_max_mismatch
                    if candidate is not None
                    else None
                ),
                outgoing_handle_available_seconds=(
                    evidence_candidate.outgoing_handle_available_seconds
                    if evidence_candidate is not None
                    else None
                ),
                required_outgoing_handle_seconds=(
                    evidence_candidate.required_outgoing_handle_seconds
                    if evidence_candidate is not None
                    else None
                ),
                incoming_segment_frame_count=(
                    evidence_candidate.incoming_segment_frame_count
                    if evidence_candidate is not None
                    else frame_counts[boundary_index + 1]
                ),
                transition_frame_count=(
                    evidence_candidate.transition_frame_count
                    if selected and evidence_candidate is not None
                    else 0
                ),
                exact_endpoint_evidence_used=(evidence_candidate is not None),
                transition_kind=transition_kind,
                required_cinematic_bridge=(bridge_candidate is not None),
                cinematic_bridge_style=(
                    bridge_candidate.style
                    if bridge_candidate is not None
                    else None
                ),
            )
        )
    return DirectMemberPoseDissolvePlan(
        transitions=tuple(transitions),
        decisions=tuple(decisions),
        fps=fps,
        timeline_duration_seconds=total_duration,
        enabled_requested=enabled,
        source_metadata_contract_verified=metadata_verified,
        config=settings,
    )


__all__ = [
    "DIRECT_MEMBER_CINEMATIC_BRIDGE_CONTRACT",
    "DIRECT_MEMBER_POSE_DISSOLVE_CONTRACT",
    "DIRECT_MEMBER_POSE_DISSOLVE_REASON",
    "OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT",
    "DirectMemberPoseDissolveConfig",
    "DirectMemberPoseDissolveDecision",
    "DirectMemberPoseDissolvePlan",
    "build_direct_member_pose_dissolve_plan",
]
