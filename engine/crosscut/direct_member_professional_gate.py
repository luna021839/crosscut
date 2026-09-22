"""Fail-closed professional pre-render gate for automatic member edits.

The direct-member selector already rejects individual unsafe source pairs.  A
whole-song edit can still be visually weak when those local decisions produce
too few visible cuts, too little editorial camera work, or stale transition
metadata.  This module audits the *materialized* selection, transitions, and
dense transform review together before FFmpeg is allowed to start.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .direct_member_visual import (
    DIRECT_MEMBER_PROFESSIONAL_ATTENTION_TRIGGER_TOLERANCE,
    DIRECT_MEMBER_PROFESSIONAL_MAXIMUM_MOTION_SPEED_P95,
    DIRECT_MEMBER_PROFESSIONAL_MAXIMUM_SILHOUETTE_P95,
    DirectMemberVisualSelection,
)
from .editorial_motion import EditorialMotionPlan
from .render import RenderTransition, TransitionKind


DIRECT_MEMBER_PROFESSIONAL_PRE_RENDER_CONTRACT = (
    "automatic_ive_member_professional_pre_render_gate_v1"
)


def _p95(values: Sequence[float]) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _finite_scalar(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _verified_decoded_proxy_continuity(
    payload: Mapping[str, Any] | None,
    *,
    selected_boundary_indices: Sequence[int],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]] | None:
    """Return exact-output evidence only after its complete contract passes."""

    if not isinstance(payload, Mapping) or any(
        payload.get(name) is not True
        for name in (
            "passed",
            "member_output_quality_passed",
            "direct_member_rendered_continuity_passed",
        )
    ):
        return None
    continuity = payload.get("rendered_continuity")
    if not isinstance(continuity, Mapping) or continuity.get("passed") is not True:
        return None
    if continuity.get("failure_reasons") not in ([], ()):
        return None
    metrics = continuity.get("metrics")
    boundaries = continuity.get("boundaries")
    if not isinstance(metrics, Mapping) or not isinstance(boundaries, list):
        return None
    expected = tuple(int(value) for value in selected_boundary_indices)
    observed_records: list[dict[str, Any]] = []
    observed_indices: list[int] = []
    for item in boundaries:
        if not isinstance(item, Mapping):
            return None
        boundary_index = item.get("boundary_index")
        if (
            isinstance(boundary_index, bool)
            or not isinstance(boundary_index, int)
            or boundary_index < 0
        ):
            return None
        observed_indices.append(boundary_index)
        observed_records.append(dict(item))
    if tuple(observed_indices) != expected:
        return None
    if (
        metrics.get("source_switch_boundary_count") != len(expected)
        or metrics.get("exact_incoming_first_frame_evaluated") is not True
        or metrics.get("final_render_transform_domain_evaluated") is not True
    ):
        return None
    for name in (
        "attention_position_mismatch_p95",
        "attention_scale_mismatch_p95",
        "pose_mismatch_p95",
    ):
        if _finite_scalar(metrics.get(name)) is None:
            return None
    return (dict(metrics), tuple(observed_records))


@dataclass(frozen=True, slots=True)
class DirectMemberProfessionalGateConfig:
    """Whole-song quality floors calibrated above the accepted I AM baseline."""

    minimum_visible_switches_per_minute: float = 15.0
    maximum_visible_switches_per_minute: float = 22.0
    minimum_editorial_transform_segments: int = 8
    minimum_editorial_transform_segments_per_minute: float = 3.0
    maximum_pose_mismatch_p95: float = 0.24
    maximum_silhouette_mismatch_p95: float = (
        DIRECT_MEMBER_PROFESSIONAL_MAXIMUM_SILHOUETTE_P95
    )
    maximum_effective_attention_position_mismatch_p95: float = 0.12
    maximum_effective_attention_scale_mismatch_p95: float = 0.10
    maximum_motion_speed_ratio_mismatch_p95: float = (
        DIRECT_MEMBER_PROFESSIONAL_MAXIMUM_MOTION_SPEED_P95
    )
    maximum_pose_phase_continuation_mismatch_p95: float = 0.30
    minimum_selected_effect_count_when_requested: int = 1
    minimum_selected_effects_per_minute_when_requested: float = 1.5
    maximum_native_output_portrait_duration_fraction: float = 0.28

    def __post_init__(self) -> None:
        finite = (
            self.minimum_visible_switches_per_minute,
            self.maximum_visible_switches_per_minute,
            self.minimum_editorial_transform_segments_per_minute,
            self.maximum_pose_mismatch_p95,
            self.maximum_silhouette_mismatch_p95,
            self.maximum_effective_attention_position_mismatch_p95,
            self.maximum_effective_attention_scale_mismatch_p95,
            self.maximum_motion_speed_ratio_mismatch_p95,
            self.maximum_pose_phase_continuation_mismatch_p95,
            self.minimum_selected_effects_per_minute_when_requested,
            self.maximum_native_output_portrait_duration_fraction,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in finite):
            raise ValueError("direct member professional gate values are invalid")
        if (
            self.minimum_visible_switches_per_minute
            > self.maximum_visible_switches_per_minute
        ):
            raise ValueError("direct member professional cadence bounds are reversed")
        if self.maximum_native_output_portrait_duration_fraction > 1.0:
            raise ValueError("native portrait duration fraction must be normalized")
        for name, value in (
            (
                "minimum_editorial_transform_segments",
                self.minimum_editorial_transform_segments,
            ),
            (
                "minimum_selected_effect_count_when_requested",
                self.minimum_selected_effect_count_when_requested,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_visible_switches_per_minute": (
                self.minimum_visible_switches_per_minute
            ),
            "maximum_visible_switches_per_minute": (
                self.maximum_visible_switches_per_minute
            ),
            "minimum_editorial_transform_segments": (
                self.minimum_editorial_transform_segments
            ),
            "minimum_editorial_transform_segments_per_minute": (
                self.minimum_editorial_transform_segments_per_minute
            ),
            "maximum_pose_mismatch_p95": self.maximum_pose_mismatch_p95,
            "maximum_silhouette_mismatch_p95": (
                self.maximum_silhouette_mismatch_p95
            ),
            "maximum_effective_attention_position_mismatch_p95": (
                self.maximum_effective_attention_position_mismatch_p95
            ),
            "maximum_effective_attention_scale_mismatch_p95": (
                self.maximum_effective_attention_scale_mismatch_p95
            ),
            "maximum_motion_speed_ratio_mismatch_p95": (
                self.maximum_motion_speed_ratio_mismatch_p95
            ),
            "maximum_pose_phase_continuation_mismatch_p95": (
                self.maximum_pose_phase_continuation_mismatch_p95
            ),
            "minimum_selected_effect_count_when_requested": (
                self.minimum_selected_effect_count_when_requested
            ),
            "minimum_selected_effects_per_minute_when_requested": (
                self.minimum_selected_effects_per_minute_when_requested
            ),
            "maximum_native_output_portrait_duration_fraction": (
                self.maximum_native_output_portrait_duration_fraction
            ),
        }


@dataclass(frozen=True, slots=True)
class DirectMemberProfessionalGateResult:
    passed: bool
    failure_reasons: tuple[str, ...]
    failure_boundary_indices: Mapping[str, tuple[int, ...]]
    metrics: Mapping[str, Any]
    boundary_diagnostics: tuple[Mapping[str, Any], ...]
    config: DirectMemberProfessionalGateConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": DIRECT_MEMBER_PROFESSIONAL_PRE_RENDER_CONTRACT,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "failure_boundary_indices": {
                key: list(value)
                for key, value in sorted(self.failure_boundary_indices.items())
            },
            "metrics": dict(self.metrics),
            "boundary_diagnostics": [
                dict(item) for item in self.boundary_diagnostics
            ],
            "thresholds": self.config.to_dict(),
            "authority": {
                "scope": "automatic_official_target_member_fancam_only",
                "evaluates_materialized_selection": True,
                "evaluates_dense_transform_acceptance": True,
                "evaluates_transition_materialization": True,
                "raw_frames_persisted": False,
                "identity_inference_used": False,
                "decoded_output_proxy_authority_active": bool(
                    self.metrics.get("decoded_output_proxy_authority_active")
                ),
                "failure_effect": "stop_before_render",
            },
        }


def evaluate_direct_member_professional_pre_render_gate(
    *,
    visual_selection: DirectMemberVisualSelection,
    cadence: Mapping[str, Any],
    editorial_motion_plan: EditorialMotionPlan | None,
    transitions: Sequence[RenderTransition],
    duration_seconds: float,
    dense_geometry_review_completed: bool,
    decoded_proxy_preflight: Mapping[str, Any] | None = None,
    materialized_editorial_transform_segment_count: int | None = None,
    config: DirectMemberProfessionalGateConfig | None = None,
) -> DirectMemberProfessionalGateResult:
    """Audit one automatic-member plan and fail on any weak global contract."""

    cfg = config or DirectMemberProfessionalGateConfig()
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError("direct member professional gate duration is invalid")
    boundaries = tuple(visual_selection.boundaries)
    transition_items = tuple(transitions)
    if len(transition_items) != len(boundaries):
        raise ValueError("direct member transition count does not match boundaries")

    selected = tuple(boundary for boundary in boundaries if boundary.cut_applied)
    selected_indices = tuple(boundary.boundary_index for boundary in selected)
    switch_count = len(selected)
    decoded_proxy = _verified_decoded_proxy_continuity(
        decoded_proxy_preflight,
        selected_boundary_indices=selected_indices,
    )
    decoded_proxy_metrics = decoded_proxy[0] if decoded_proxy is not None else {}
    decoded_proxy_records = decoded_proxy[1] if decoded_proxy is not None else ()
    decoded_proxy_authority_active = decoded_proxy is not None
    if materialized_editorial_transform_segment_count is not None and (
        isinstance(materialized_editorial_transform_segment_count, bool)
        or not isinstance(materialized_editorial_transform_segment_count, int)
        or materialized_editorial_transform_segment_count < 0
    ):
        raise ValueError("materialized editorial transform count is invalid")
    switches_per_minute = switch_count * 60.0 / duration_seconds
    failures: list[str] = []
    failure_indices: dict[str, tuple[int, ...]] = {}

    reported_switch_count = cadence.get("source_switch_count")
    if reported_switch_count != switch_count:
        failures.append("materialized_switch_count_disagrees_with_cadence")
    if not (
        cfg.minimum_visible_switches_per_minute - 1e-9
        <= switches_per_minute
        <= cfg.maximum_visible_switches_per_minute + 1e-9
    ):
        failures.append("visible_switch_density_outside_professional_range")

    missing_pose = tuple(
        boundary.boundary_index
        for boundary in selected
        if boundary.mismatch.dominant_pose_mismatch is None
    )
    pose_over_limit = tuple(
        boundary.boundary_index
        for boundary in selected
        if boundary.mismatch.dominant_pose_mismatch is not None
        and boundary.mismatch.dominant_pose_mismatch
        > visual_selection.config.maximum_cross_source_pose_mismatch + 1e-9
    )
    missing_silhouette = tuple(
        boundary.boundary_index
        for boundary in selected
        if boundary.mismatch.silhouette_mismatch is None
    )
    materialized_cinematic_bridge_indices = frozenset(
        boundary.boundary_index
        for boundary in selected
        if boundary.mismatch.cinematic_bridge is not None
        and transition_items[boundary.boundary_index].kind
        in {
            TransitionKind.MEMBER_ZOOM_BRIDGE,
            TransitionKind.MEMBER_BLUR_BRIDGE,
        }
    )
    silhouette_over_limit = tuple(
        boundary.boundary_index
        for boundary in selected
        if boundary.boundary_index not in materialized_cinematic_bridge_indices
        if boundary.mismatch.silhouette_mismatch is not None
        and boundary.mismatch.silhouette_mismatch
        > visual_selection.config.maximum_cross_source_silhouette_mismatch + 1e-9
    )

    missing_attention_geometry: list[int] = []
    incompatible_attention_geometry: list[int] = []
    unrepaired_attention_geometry: list[int] = []
    effective_positions: list[float] = []
    effective_scales: list[float] = []
    dense_attention_residuals = {
        int(boundary_index): (float(position), float(scale))
        for boundary_index, position, scale in (
            getattr(
                editorial_motion_plan,
                "boundary_residual_attention_mismatches",
                (),
            )
            if editorial_motion_plan is not None
            else ()
        )
    }
    for boundary in selected:
        mismatch = boundary.mismatch
        if boundary.boundary_index in materialized_cinematic_bridge_indices:
            continue
        left_attention = boundary.outgoing_sample.attention
        right_attention = boundary.incoming_sample.attention
        if left_attention is None or right_attention is None:
            missing_attention_geometry.append(boundary.boundary_index)
            continue
        repair = mismatch.transform_repair
        if left_attention.basis != right_attention.basis and repair is None:
            incompatible_attention_geometry.append(boundary.boundary_index)
            continue
        dense_residual = dense_attention_residuals.get(boundary.boundary_index)
        effective_position = (
            dense_residual[0]
            if dense_residual is not None
            else repair.residual_attention_position_mismatch
            if repair is not None
            else mismatch.attention_position_mismatch
        )
        effective_scale = (
            dense_residual[1]
            if dense_residual is not None
            else repair.residual_attention_scale_mismatch
            if repair is not None
            else mismatch.attention_scale_mismatch
        )
        if effective_position is None or effective_scale is None:
            missing_attention_geometry.append(boundary.boundary_index)
            continue
        effective_positions.append(float(effective_position))
        effective_scales.append(float(effective_scale))
        face_pair = (
            left_attention.basis
            == right_attention.basis
            == "unambiguous_dominant_face_bbox_geometry_only"
        )
        position_trigger = (
            visual_selection.config.transform_alignment_position_trigger
            if face_pair
            else visual_selection.config.transform_body_alignment_position_trigger
        )
        scale_trigger = (
            visual_selection.config.transform_alignment_scale_trigger
            if face_pair
            else visual_selection.config.transform_body_alignment_scale_trigger
        )
        if (
            (
                effective_position
                > position_trigger
                + DIRECT_MEMBER_PROFESSIONAL_ATTENTION_TRIGGER_TOLERANCE
                + 1e-9
            )
            or (
                effective_scale
                > scale_trigger
                + DIRECT_MEMBER_PROFESSIONAL_ATTENTION_TRIGGER_TOLERANCE
                + 1e-9
            )
        ):
            unrepaired_attention_geometry.append(boundary.boundary_index)

    opposed_motion: list[int] = []
    incompatible_motion_shape: list[int] = []
    incompatible_motion_speed: list[int] = []
    incompatible_pose_phase: list[int] = []
    motion_speed_mismatches: list[float] = []
    pose_phase_mismatches: list[float] = []
    both_moving_count = 0
    for boundary in selected:
        if boundary.boundary_index in materialized_cinematic_bridge_indices:
            continue
        motion = boundary.mismatch.motion_connection
        if motion is None or not motion.both_moving:
            continue
        both_moving_count += 1
        if (
            motion.direction_cosine_audit_only is None
            or motion.direction_cosine_audit_only
            < visual_selection.config.minimum_motion_direction_cosine - 1e-9
        ):
            opposed_motion.append(boundary.boundary_index)
        if (
            motion.endpoint_pose_shape_mismatch is None
            or motion.endpoint_pose_shape_mismatch
            > visual_selection.config.maximum_motion_endpoint_shape_mismatch
            + 1e-9
        ):
            incompatible_motion_shape.append(boundary.boundary_index)
        motion_speed_mismatch = getattr(
            motion,
            "motion_speed_ratio_mismatch",
            None,
        )
        if motion_speed_mismatch is not None:
            motion_speed_mismatches.append(
                float(motion_speed_mismatch)
            )
            if (
                float(motion_speed_mismatch)
                > float(
                    getattr(
                        visual_selection.config,
                        "maximum_motion_speed_ratio_mismatch",
                        0.60,
                    )
                )
                + 1e-9
            ):
                incompatible_motion_speed.append(boundary.boundary_index)
        pose_phase_mismatch = getattr(
            motion,
            "pose_phase_continuation_mismatch",
            None,
        )
        if pose_phase_mismatch is not None:
            pose_phase_mismatches.append(
                float(pose_phase_mismatch)
            )
            if (
                float(pose_phase_mismatch)
                > float(
                    getattr(
                        visual_selection.config,
                        "maximum_pose_phase_continuation_mismatch",
                        0.34,
                    )
                )
                + 1e-9
            ):
                incompatible_pose_phase.append(boundary.boundary_index)

    bridged_severe: list[int] = []
    uncontracted_severe: list[int] = []
    required_repairs: list[int] = []
    for boundary in selected:
        if boundary.mismatch.transform_repair is not None:
            required_repairs.append(boundary.boundary_index)
        if not boundary.mismatch.severe_flags:
            continue
        transition = transition_items[boundary.boundary_index]
        if (
            boundary.mismatch.cinematic_bridge is not None
            and transition.kind
            in {
                TransitionKind.MEMBER_ZOOM_BRIDGE,
                TransitionKind.MEMBER_BLUR_BRIDGE,
            }
        ):
            bridged_severe.append(boundary.boundary_index)
        else:
            uncontracted_severe.append(boundary.boundary_index)

    accepted_repairs = (
        frozenset(editorial_motion_plan.accepted_boundary_indices).union(
            getattr(
                editorial_motion_plan,
                "natural_match_boundary_indices",
                (),
            )
        )
        if editorial_motion_plan is not None
        else frozenset()
    )
    missing_dense_repairs = tuple(
        index for index in required_repairs if index not in accepted_repairs
    )
    planner_applied_motion_count = (
        editorial_motion_plan.applied_count
        if editorial_motion_plan is not None
        else 0
    )
    applied_motion_count = (
        materialized_editorial_transform_segment_count
        if materialized_editorial_transform_segment_count is not None
        else planner_applied_motion_count
    )
    applied_motion_per_minute = applied_motion_count * 60.0 / duration_seconds
    required_motion_count = max(
        cfg.minimum_editorial_transform_segments,
        math.ceil(
            cfg.minimum_editorial_transform_segments_per_minute
            * duration_seconds
            / 60.0
        ),
    )

    dissolve_payload = cadence.get("pose_matched_dissolves")
    dissolve_payload = (
        dissolve_payload if isinstance(dissolve_payload, Mapping) else {}
    )
    effect_requested = bool(cadence.get("member_pose_dissolve_requested"))
    eligible_effect_count = int(dissolve_payload.get("eligible_candidate_count", 0))
    selected_effect_count = sum(
        transition.kind is not TransitionKind.HARD_CUT
        for index, transition in enumerate(transition_items)
        if index in selected_indices
    )
    reported_effect_count = int(
        cadence.get("member_pose_dissolve_count", 0)
    ) + int(cadence.get("member_cinematic_bridge_count", 0))
    requested_effect_floor = max(
        cfg.minimum_selected_effect_count_when_requested,
        math.ceil(
            cfg.minimum_selected_effects_per_minute_when_requested
            * duration_seconds
            / 60.0
            - 1e-12
        ),
    )
    required_effect_count = (
        max(1, min(eligible_effect_count, requested_effect_floor))
        if effect_requested and switch_count
        else 0
    )

    if decoded_proxy_authority_active:
        missing_pose = ()
        pose_over_limit = ()
        missing_silhouette = ()
        silhouette_over_limit = ()
        missing_attention_geometry = []
        incompatible_attention_geometry = []
        unrepaired_attention_geometry = []
        opposed_motion = []
        incompatible_motion_shape = []
        incompatible_motion_speed = []
        incompatible_pose_phase = []
        missing_dense_repairs = ()

    index_failures = {
        "missing_actual_pose": missing_pose,
        "pose_over_hard_limit": pose_over_limit,
        "missing_silhouette": missing_silhouette,
        "silhouette_over_hard_limit": silhouette_over_limit,
        "missing_attention_geometry": tuple(missing_attention_geometry),
        "incompatible_attention_geometry": tuple(
            incompatible_attention_geometry
        ),
        "unrepaired_attention_geometry": tuple(unrepaired_attention_geometry),
        "opposed_or_missing_motion_direction": tuple(opposed_motion),
        "incompatible_or_missing_motion_shape": tuple(
            incompatible_motion_shape
        ),
        "incompatible_motion_speed": tuple(incompatible_motion_speed),
        "incompatible_pose_phase": tuple(incompatible_pose_phase),
        "uncontracted_severe_boundary": tuple(uncontracted_severe),
        "required_dense_transform_not_accepted": missing_dense_repairs,
    }
    for reason, indices in index_failures.items():
        if indices:
            failures.append(reason)
            failure_indices[reason] = indices

    source_pose_p95 = _p95(
        [
            boundary.mismatch.dominant_pose_mismatch
            for boundary in selected
            if boundary.boundary_index not in materialized_cinematic_bridge_indices
            if boundary.mismatch.dominant_pose_mismatch is not None
        ]
    )
    source_silhouette_p95 = _p95(
        [
            boundary.mismatch.silhouette_mismatch
            for boundary in selected
            if boundary.boundary_index not in materialized_cinematic_bridge_indices
            if boundary.mismatch.silhouette_mismatch is not None
        ]
    )
    source_attention_position_p95 = _p95(effective_positions)
    source_attention_scale_p95 = _p95(effective_scales)
    source_motion_speed_p95 = _p95(motion_speed_mismatches)
    source_pose_phase_p95 = _p95(pose_phase_mismatches)
    pose_p95 = (
        _finite_scalar(decoded_proxy_metrics.get("pose_mismatch_p95"))
        if decoded_proxy_authority_active
        else source_pose_p95
    )
    silhouette_p95 = source_silhouette_p95
    attention_position_p95 = (
        _finite_scalar(
            decoded_proxy_metrics.get("attention_position_mismatch_p95")
        )
        if decoded_proxy_authority_active
        else source_attention_position_p95
    )
    attention_scale_p95 = (
        _finite_scalar(
            decoded_proxy_metrics.get("attention_scale_mismatch_p95")
        )
        if decoded_proxy_authority_active
        else source_attention_scale_p95
    )
    motion_speed_p95 = (
        None if decoded_proxy_authority_active else source_motion_speed_p95
    )
    pose_phase_p95 = (
        None if decoded_proxy_authority_active else source_pose_phase_p95
    )
    if pose_p95 is None or pose_p95 > cfg.maximum_pose_mismatch_p95 + 1e-9:
        failures.append("pose_mismatch_p95_above_professional_floor")
    if (
        silhouette_p95 is None
        or silhouette_p95 > cfg.maximum_silhouette_mismatch_p95 + 1e-9
    ):
        failures.append("silhouette_mismatch_p95_above_professional_floor")
    if (
        attention_position_p95 is None
        or attention_position_p95
        > cfg.maximum_effective_attention_position_mismatch_p95 + 1e-9
    ):
        failures.append("attention_position_p95_above_professional_floor")
    if (
        attention_scale_p95 is None
        or attention_scale_p95
        > cfg.maximum_effective_attention_scale_mismatch_p95 + 1e-9
    ):
        failures.append("attention_scale_p95_above_professional_floor")
    if (
        motion_speed_p95 is not None
        and motion_speed_p95
        > cfg.maximum_motion_speed_ratio_mismatch_p95 + 1e-9
    ):
        failures.append("motion_speed_p95_above_professional_floor")
    if (
        pose_phase_p95 is not None
        and pose_phase_p95
        > cfg.maximum_pose_phase_continuation_mismatch_p95 + 1e-9
    ):
        failures.append("pose_phase_p95_above_professional_floor")
    if editorial_motion_plan is None or not dense_geometry_review_completed:
        failures.append("dense_editorial_camera_review_missing_or_incomplete")
    if applied_motion_count < required_motion_count:
        failures.append("editorial_camera_density_below_professional_floor")

    source_geometries = tuple(
        getattr(visual_selection, "source_frame_geometries", ()) or ()
    )
    source_durations = tuple(
        float(value)
        for value in (
            getattr(visual_selection, "source_durations_seconds", ()) or ()
        )
    )
    native_portrait_duration_fraction: float | None = None
    high_resolution_portrait_duration_fraction: float | None = None
    if (
        source_geometries
        and len(source_geometries) == len(source_durations)
        and sum(source_durations) > 1e-9
    ):
        high_resolution_portrait_indices = tuple(
            index
            for index, geometry in enumerate(source_geometries)
            if geometry is not None
            and geometry.source_width <= geometry.source_height
            and geometry.lossless_zoom_capacity >= 1.50
        )
        native_portrait_indices = tuple(
            index
            for index, geometry in enumerate(source_geometries)
            if geometry is not None
            and geometry.source_width <= geometry.source_height
            and geometry.lossless_zoom_capacity < 1.50
        )
        total_source_duration = sum(source_durations)
        native_portrait_duration_fraction = sum(
            source_durations[index] for index in native_portrait_indices
        ) / total_source_duration
        high_resolution_portrait_duration_fraction = sum(
            source_durations[index]
            for index in high_resolution_portrait_indices
        ) / total_source_duration
        if (
            high_resolution_portrait_indices
            and native_portrait_indices
            and native_portrait_duration_fraction
            > cfg.maximum_native_output_portrait_duration_fraction + 1e-9
        ):
            failures.append(
                "native_output_portrait_duration_fraction_above_quality_floor"
            )
    highlight = cadence.get("highlight_cut_density")
    if not isinstance(highlight, Mapping) or highlight.get("passed") is not True:
        failures.append("highlight_cut_density_gate_missing_or_failed")
    if selected_effect_count != reported_effect_count:
        failures.append("materialized_effect_count_disagrees_with_cadence")
    if selected_effect_count < required_effect_count:
        failures.append("requested_safe_transition_effect_not_materialized")

    failures = list(dict.fromkeys(failures))
    metrics = {
        "duration_seconds": duration_seconds,
        "selected_boundary_count": len(boundaries),
        "visible_switch_count": switch_count,
        "visible_switches_per_minute": switches_per_minute,
        "actual_pose_coverage_fraction": (
            (switch_count - len(missing_pose)) / switch_count
            if switch_count
            else 0.0
        ),
        "silhouette_coverage_fraction": (
            (switch_count - len(missing_silhouette)) / switch_count
            if switch_count
            else 0.0
        ),
        "attention_geometry_coverage_fraction": (
            (switch_count - len(missing_attention_geometry)) / switch_count
            if switch_count
            else 0.0
        ),
        "both_moving_boundary_count": both_moving_count,
        "pose_mismatch_p95": pose_p95,
        "silhouette_mismatch_p95": silhouette_p95,
        "effective_attention_position_mismatch_p95": attention_position_p95,
        "effective_attention_scale_mismatch_p95": attention_scale_p95,
        "motion_speed_ratio_mismatch_p95": motion_speed_p95,
        "pose_phase_continuation_mismatch_p95": pose_phase_p95,
        "source_domain_pose_mismatch_p95_diagnostic": source_pose_p95,
        "source_domain_attention_position_mismatch_p95_diagnostic": (
            source_attention_position_p95
        ),
        "source_domain_attention_scale_mismatch_p95_diagnostic": (
            source_attention_scale_p95
        ),
        "source_domain_motion_speed_ratio_mismatch_p95_diagnostic": (
            source_motion_speed_p95
        ),
        "source_domain_pose_phase_mismatch_p95_diagnostic": (
            source_pose_phase_p95
        ),
        "decoded_output_proxy_authority_active": (
            decoded_proxy_authority_active
        ),
        "decoded_output_proxy_boundary_count": len(decoded_proxy_records),
        "required_dense_transform_boundary_count": len(required_repairs),
        "accepted_dense_transform_boundary_count": sum(
            index in accepted_repairs for index in required_repairs
        ),
        "editorial_transform_segment_count": applied_motion_count,
        "planner_editorial_transform_segment_count": (
            planner_applied_motion_count
        ),
        "editorial_transform_segments_per_minute": applied_motion_per_minute,
        "required_editorial_transform_segment_count": required_motion_count,
        "effect_requested": effect_requested,
        "eligible_optional_effect_count": eligible_effect_count,
        "materialized_effect_count": selected_effect_count,
        "required_effect_count": required_effect_count,
        "native_output_portrait_duration_fraction": (
            native_portrait_duration_fraction
        ),
        "high_resolution_portrait_duration_fraction": (
            high_resolution_portrait_duration_fraction
        ),
        "cinematic_bridge_boundary_count": len(bridged_severe),
        "highlight_cut_density_passed": bool(
            isinstance(highlight, Mapping) and highlight.get("passed") is True
        ),
    }
    boundary_diagnostics = tuple(
        {
            "boundary_index": boundary.boundary_index,
            "outgoing_source_index": getattr(
                boundary,
                "outgoing_eligible_source_index",
                None,
            ),
            "incoming_source_index": getattr(
                boundary,
                "incoming_eligible_source_index",
                None,
            ),
            "outgoing_attention_basis": (
                boundary.outgoing_sample.attention.basis
                if boundary.outgoing_sample.attention is not None
                else None
            ),
            "incoming_attention_basis": (
                boundary.incoming_sample.attention.basis
                if boundary.incoming_sample.attention is not None
                else None
            ),
            "pose_mismatch": boundary.mismatch.dominant_pose_mismatch,
            "silhouette_mismatch": boundary.mismatch.silhouette_mismatch,
            "raw_attention_position_mismatch": (
                boundary.mismatch.attention_position_mismatch
            ),
            "raw_attention_scale_mismatch": (
                boundary.mismatch.attention_scale_mismatch
            ),
            "effective_attention_position_mismatch": (
                dense_attention_residuals[boundary.boundary_index][0]
                if boundary.boundary_index in dense_attention_residuals
                else boundary.mismatch.transform_repair.residual_attention_position_mismatch
                if boundary.mismatch.transform_repair is not None
                else boundary.mismatch.attention_position_mismatch
            ),
            "effective_attention_scale_mismatch": (
                dense_attention_residuals[boundary.boundary_index][1]
                if boundary.boundary_index in dense_attention_residuals
                else boundary.mismatch.transform_repair.residual_attention_scale_mismatch
                if boundary.mismatch.transform_repair is not None
                else boundary.mismatch.attention_scale_mismatch
            ),
            "coarse_transform_repair_proposed": (
                boundary.mismatch.transform_repair is not None
            ),
            "dense_transform_accepted": (
                boundary.boundary_index in accepted_repairs
            ),
            "severe_flags": list(boundary.mismatch.severe_flags),
            "motion_both_moving": bool(
                boundary.mismatch.motion_connection is not None
                and boundary.mismatch.motion_connection.both_moving
            ),
            "motion_direction_cosine": (
                boundary.mismatch.motion_connection.direction_cosine_audit_only
                if boundary.mismatch.motion_connection is not None
                else None
            ),
            "motion_speed_ratio_mismatch": (
                getattr(
                    boundary.mismatch.motion_connection,
                    "motion_speed_ratio_mismatch",
                    None,
                )
                if boundary.mismatch.motion_connection is not None
                else None
            ),
            "pose_phase_continuation_mismatch": (
                getattr(
                    boundary.mismatch.motion_connection,
                    "pose_phase_continuation_mismatch",
                    None,
                )
                if boundary.mismatch.motion_connection is not None
                else None
            ),
            "transition_kind": transition_items[
                boundary.boundary_index
            ].kind.value,
        }
        for boundary in selected
    )
    return DirectMemberProfessionalGateResult(
        passed=not failures,
        failure_reasons=tuple(failures),
        failure_boundary_indices=failure_indices,
        metrics=metrics,
        boundary_diagnostics=boundary_diagnostics,
        config=cfg,
    )


__all__ = [
    "DIRECT_MEMBER_PROFESSIONAL_PRE_RENDER_CONTRACT",
    "DirectMemberProfessionalGateConfig",
    "DirectMemberProfessionalGateResult",
    "evaluate_direct_member_professional_pre_render_gate",
]
