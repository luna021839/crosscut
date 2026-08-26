"""Non-identifying visual path selection for direct member fancams.

The direct portrait member path never identifies a face. It can authorize a
short geometry-only crop/zoom/pan repair for an already-authenticated official
target fancam, but only when source resolution and subject containment prove
that repair can be rendered without an upscaled working frame. This module
samples every eligible source on both sides of each proposed hard cut and finds
a balanced, no-repeat source path that minimizes the residual visible jumps.

Only normalized screen geometry and aggregate mismatch scores are serialized.
Pose, silhouette, layout, and structure vectors are transient and no face
appearance embedding or named identity inference is used.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import hashlib
import heapq
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Sequence

import cv2

from .direct_member_transform import (
    DirectMemberBoundaryTransformRepair,
    DirectMemberSourceFrameGeometry,
    direct_member_source_frame_geometry,
    find_lossless_boundary_transform_repair,
    map_box_to_portrait_fill,
    project_box,
)
from .pose import SILHOUETTE_BACKEND
from .timeline import formation_assignment_mismatch
from .timewarp import TimeWarpMap
from .visual_feature_cache import CanonicalFrameIdentity
from .visual_score import FrameVisualDescriptor, _VideoSampler


DescriptorProvider = Callable[
    [int, Path, float],
    FrameVisualDescriptor | None,
]


_DIRECT_CONTENT_SAMPLE_COUNT = 6
_DIRECT_MEMBER_CINEMATIC_BRIDGE_ALLOWED_SEVERE_FLAGS = frozenset(
    {"attention_position", "attention_scale", "formation"}
)
_DIRECT_MEMBER_CINEMATIC_BRIDGE_MAXIMUM_POSE_MISMATCH = 0.30
_DIRECT_MEMBER_CINEMATIC_BRIDGE_MAXIMUM_SILHOUETTE_MISMATCH = 0.55
_DIRECT_MEMBER_CINEMATIC_BRIDGE_MAXIMUM_BASE_MISMATCH = 0.65
_DIRECT_MEMBER_CINEMATIC_BRIDGE_MAXIMUM_COUNT = 4
_DIRECT_MEMBER_CINEMATIC_BRIDGE_MINIMUM_BOUNDARY_GAP = 3
_DIRECT_MEMBER_CINEMATIC_BRIDGE_COST_PENALTY = 8.0
_DIRECT_MEMBER_HETEROGENEOUS_MAXIMUM_RUN_SECONDS = 7.75


class DirectMemberVisualSelectionError(RuntimeError):
    """Fail-closed visual/quota conflict with JSON-safe diagnostics."""

    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True, slots=True)
class DirectMemberVisualConfig:
    """Thresholds and weights for one direct-member visual optimization."""

    sample_frame_step_seconds: float = 1.0 / 29.97
    early_interior_sample_seconds: float = 0.20
    beam_width: int = 12_000
    span_interior_inset_seconds: float = 0.18
    minimum_pose_confidence: float = 0.25
    minimum_silhouette_confidence: float = 0.30
    minimum_evidence_component_count: int = 2
    attention_position_weight: float = 2.20
    attention_scale_weight: float = 1.60
    dominant_pose_weight: float = 2.00
    structure_pose_weight: float = 0.65
    formation_weight: float = 2.10
    silhouette_weight: float = 1.45
    person_layout_weight: float = 1.65
    person_count_weight: float = 0.45
    attention_position_warning: float = 0.32
    attention_scale_warning: float = 0.30
    dominant_pose_warning: float = 0.24
    formation_warning: float = 0.30
    silhouette_warning: float = 0.34
    person_layout_warning: float = 0.32
    person_count_warning: float = 0.65
    total_mismatch_warning: float = 0.45
    severe_risk_margin: float = 0.20
    total_mismatch_severe: float = 0.68
    warning_penalty: float = 0.45
    severe_penalty: float = 1.25
    insufficient_evidence_penalty: float = 0.35
    missing_descriptor_cost: float = 2.75
    ping_pong_penalty: float = 0.14
    same_source_hold_penalty: float = 0.12
    maximum_same_source_hold_fraction: float = 0.12
    maximum_same_source_hold_count: int = 8
    maximum_consecutive_same_source_holds: int = 1
    maximum_same_source_run_seconds: float = 7.2
    minimum_adjacent_motion_magnitude: float = 0.020
    minimum_temporal_motion_component_count: int = 2
    motion_connection_preference_weight: float = 0.22
    priority_motion_connection_preference_weight: float = 0.18
    minimum_motion_direction_cosine: float = 0.12
    maximum_motion_speed_ratio_mismatch: float = 0.60
    maximum_pose_phase_continuation_mismatch: float = 0.34
    maximum_motion_endpoint_shape_mismatch: float = 0.30
    maximum_cross_source_pose_mismatch: float = 0.30
    maximum_cross_source_silhouette_mismatch: float = 0.42
    require_actual_pose_for_cross_source_cut: bool = True
    boundary_search_radius_frames: int = 0
    boundary_search_step_frames: int = 2
    boundary_search_offset_penalty_per_frame: float = 0.015
    duration_balance_weight: float = 4.0
    maximum_source_duration_relative_deviation: float = 0.08
    minimum_face_sample_count: int = 1
    minimum_human_sample_count: int = 2
    minimum_acceptable_quality_score: float = 0.72
    minimum_acceptable_quality_sample_count: int = 5
    portrait_fancam_quality_rescue_mean_minimum: float = 0.50
    portrait_fancam_quality_rescue_contrast_minimum: float = 0.18
    foreground_obstruction_rejection_score: float = 0.85
    terminal_composite_exit_quality_maximum: float = 0.50
    terminal_composite_pre_exit_quality_maximum: float = 0.90
    terminal_composite_first_quality_drop_minimum: float = 0.07
    terminal_composite_exit_quality_drop_minimum: float = 0.25
    terminal_composite_first_contrast_drop_minimum: float = 0.06
    terminal_composite_exit_contrast_drop_minimum: float = 0.15
    persistent_entry_overexposure_quality_maximum: float = 0.74
    persistent_entry_overexposure_gray_mean_minimum: float = 0.80
    persistent_entry_overexposure_gray_contrast_maximum: float = 0.35
    persistent_entry_overexposure_highlight_minimum: float = 0.20
    require_entry_exit_human_evidence: bool = True
    native_scene_cut_boundary_tolerance_seconds: float = 1.0 / 29.97
    portrait_fancam_scene_cut_continuity_window_seconds: float = 0.10
    portrait_fancam_scene_cut_continuity_structure_maximum: float = 0.18
    portrait_fancam_scene_cut_post_graphic_quality_maximum: float = 0.20
    portrait_fancam_scene_cut_post_graphic_gray_mean_minimum: float = 0.90
    portrait_fancam_scene_cut_post_graphic_contrast_maximum: float = 0.12
    blank_gray_std_maximum: float = 0.045
    blank_extreme_mean_minimum: float = 0.05
    blank_extreme_mean_maximum: float = 0.95
    static_structure_mismatch_maximum: float = 0.018
    static_color_mismatch_maximum: float = 0.018
    transform_repair_maximum_zoom: float = 1.50
    transform_repair_maximum_center_shift: float = 0.120
    transform_repair_maximum_boundary_center_distance: float = 0.30
    transform_repair_maximum_residual_center_distance: float = 0.080
    transform_repair_minimum_improvement_fraction: float = 0.20
    transform_repair_minimum_face_visible_fraction: float = 0.995
    transform_repair_legacy_face_only_maximum_zoom: float = 1.12
    transform_repair_legacy_face_only_maximum_center_shift: float = 0.025
    transform_repair_minimum_subject_confidence: float = 0.30
    transform_repair_minimum_subject_visible_fraction: float = 0.980
    transform_alignment_position_trigger: float = 0.08
    transform_alignment_scale_trigger: float = 0.06
    transform_body_alignment_position_trigger: float = 0.12
    transform_body_alignment_scale_trigger: float = 0.10
    maximum_adaptive_beam_width: int = 192_000
    maximum_generated_beam_state_count: int = 500_000
    feasibility_lifeline_state_limit: int = 250_000

    def __post_init__(self) -> None:
        finite = (
            self.sample_frame_step_seconds,
            self.early_interior_sample_seconds,
            self.span_interior_inset_seconds,
            self.minimum_pose_confidence,
            self.minimum_silhouette_confidence,
            self.attention_position_weight,
            self.attention_scale_weight,
            self.dominant_pose_weight,
            self.structure_pose_weight,
            self.formation_weight,
            self.silhouette_weight,
            self.person_layout_weight,
            self.person_count_weight,
            self.attention_position_warning,
            self.attention_scale_warning,
            self.dominant_pose_warning,
            self.formation_warning,
            self.silhouette_warning,
            self.person_layout_warning,
            self.person_count_warning,
            self.total_mismatch_warning,
            self.severe_risk_margin,
            self.total_mismatch_severe,
            self.warning_penalty,
            self.severe_penalty,
            self.insufficient_evidence_penalty,
            self.missing_descriptor_cost,
            self.ping_pong_penalty,
            self.same_source_hold_penalty,
            self.maximum_same_source_hold_fraction,
            self.maximum_same_source_run_seconds,
            self.minimum_adjacent_motion_magnitude,
            self.motion_connection_preference_weight,
            self.priority_motion_connection_preference_weight,
            self.minimum_motion_direction_cosine,
            self.maximum_motion_speed_ratio_mismatch,
            self.maximum_pose_phase_continuation_mismatch,
            self.maximum_motion_endpoint_shape_mismatch,
            self.maximum_cross_source_pose_mismatch,
            self.maximum_cross_source_silhouette_mismatch,
            self.boundary_search_offset_penalty_per_frame,
            self.duration_balance_weight,
            self.maximum_source_duration_relative_deviation,
            self.minimum_acceptable_quality_score,
            self.portrait_fancam_quality_rescue_mean_minimum,
            self.portrait_fancam_quality_rescue_contrast_minimum,
            self.foreground_obstruction_rejection_score,
            self.terminal_composite_exit_quality_maximum,
            self.terminal_composite_pre_exit_quality_maximum,
            self.terminal_composite_first_quality_drop_minimum,
            self.terminal_composite_exit_quality_drop_minimum,
            self.terminal_composite_first_contrast_drop_minimum,
            self.terminal_composite_exit_contrast_drop_minimum,
            self.persistent_entry_overexposure_quality_maximum,
            self.persistent_entry_overexposure_gray_mean_minimum,
            self.persistent_entry_overexposure_gray_contrast_maximum,
            self.persistent_entry_overexposure_highlight_minimum,
            self.native_scene_cut_boundary_tolerance_seconds,
            self.portrait_fancam_scene_cut_continuity_window_seconds,
            self.portrait_fancam_scene_cut_continuity_structure_maximum,
            self.portrait_fancam_scene_cut_post_graphic_quality_maximum,
            self.portrait_fancam_scene_cut_post_graphic_gray_mean_minimum,
            self.portrait_fancam_scene_cut_post_graphic_contrast_maximum,
            self.blank_gray_std_maximum,
            self.blank_extreme_mean_minimum,
            self.blank_extreme_mean_maximum,
            self.static_structure_mismatch_maximum,
            self.static_color_mismatch_maximum,
            self.transform_repair_maximum_zoom,
            self.transform_repair_maximum_center_shift,
            self.transform_repair_maximum_boundary_center_distance,
            self.transform_repair_maximum_residual_center_distance,
            self.transform_repair_minimum_improvement_fraction,
            self.transform_repair_minimum_face_visible_fraction,
            self.transform_repair_legacy_face_only_maximum_zoom,
            self.transform_repair_legacy_face_only_maximum_center_shift,
            self.transform_repair_minimum_subject_confidence,
            self.transform_repair_minimum_subject_visible_fraction,
            self.transform_alignment_position_trigger,
            self.transform_alignment_scale_trigger,
            self.transform_body_alignment_position_trigger,
            self.transform_body_alignment_scale_trigger,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in finite):
            raise ValueError("direct member visual configuration must be finite and non-negative")
        if self.sample_frame_step_seconds <= 0.0:
            raise ValueError("sample_frame_step_seconds must be positive")
        if self.early_interior_sample_seconds <= 0.0:
            raise ValueError("early_interior_sample_seconds must be positive")
        if self.span_interior_inset_seconds <= 0.0:
            raise ValueError("span_interior_inset_seconds must be positive")
        if self.maximum_same_source_hold_fraction > 1.0:
            raise ValueError(
                "maximum_same_source_hold_fraction must be at most one"
            )
        if self.maximum_same_source_run_seconds <= 0.0:
            raise ValueError("maximum_same_source_run_seconds must be positive")
        if not 1.02 <= self.transform_repair_maximum_zoom <= 1.50:
            raise ValueError("direct member transform repair zoom is invalid")
        if not 0.0 <= self.transform_repair_maximum_center_shift <= 0.120:
            raise ValueError("direct member transform repair pan is invalid")
        if not (
            0.0
            <= self.transform_repair_legacy_face_only_maximum_center_shift
            <= self.transform_repair_maximum_center_shift
        ):
            raise ValueError("direct member legacy transform pan is invalid")
        if not (
            1.0
            < self.transform_repair_legacy_face_only_maximum_zoom
            <= self.transform_repair_maximum_zoom
        ):
            raise ValueError("direct member legacy transform zoom is invalid")
        for name, value in (
            ("maximum_same_source_hold_count", self.maximum_same_source_hold_count),
            (
                "maximum_consecutive_same_source_holds",
                self.maximum_consecutive_same_source_holds,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            isinstance(self.beam_width, bool)
            or not isinstance(self.beam_width, int)
            or self.beam_width < 100
        ):
            raise ValueError("beam_width must be an integer of at least 100")
        for name, value, minimum in (
            ("maximum_adaptive_beam_width", self.maximum_adaptive_beam_width, 100),
            (
                "maximum_generated_beam_state_count",
                self.maximum_generated_beam_state_count,
                1_000,
            ),
            (
                "feasibility_lifeline_state_limit",
                self.feasibility_lifeline_state_limit,
                1,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if (
            isinstance(self.minimum_evidence_component_count, bool)
            or not isinstance(self.minimum_evidence_component_count, int)
            or self.minimum_evidence_component_count < 1
        ):
            raise ValueError(
                "minimum_evidence_component_count must be a positive integer"
            )
        if (
            isinstance(self.minimum_temporal_motion_component_count, bool)
            or not isinstance(self.minimum_temporal_motion_component_count, int)
            or not 1 <= self.minimum_temporal_motion_component_count <= 3
        ):
            raise ValueError(
                "minimum_temporal_motion_component_count must be between one and three"
            )
        if (
            isinstance(self.minimum_face_sample_count, bool)
            or not isinstance(self.minimum_face_sample_count, int)
            or not 0
            <= self.minimum_face_sample_count
            <= _DIRECT_CONTENT_SAMPLE_COUNT
        ):
            raise ValueError("minimum_face_sample_count must be between zero and six")
        if (
            isinstance(self.minimum_human_sample_count, bool)
            or not isinstance(self.minimum_human_sample_count, int)
            or not 1
            <= self.minimum_human_sample_count
            <= _DIRECT_CONTENT_SAMPLE_COUNT
        ):
            raise ValueError("minimum_human_sample_count must be between one and six")
        if (
            isinstance(self.minimum_acceptable_quality_sample_count, bool)
            or not isinstance(self.minimum_acceptable_quality_sample_count, int)
            or not 1
            <= self.minimum_acceptable_quality_sample_count
            <= _DIRECT_CONTENT_SAMPLE_COUNT
        ):
            raise ValueError(
                "minimum_acceptable_quality_sample_count must be between one and six"
            )
        if not isinstance(self.require_entry_exit_human_evidence, bool):
            raise TypeError("require_entry_exit_human_evidence must be boolean")
        if not isinstance(self.require_actual_pose_for_cross_source_cut, bool):
            raise TypeError(
                "require_actual_pose_for_cross_source_cut must be boolean"
            )
        if (
            isinstance(self.boundary_search_radius_frames, bool)
            or not isinstance(self.boundary_search_radius_frames, int)
            or not 0 <= self.boundary_search_radius_frames <= 12
        ):
            raise ValueError(
                "boundary_search_radius_frames must be between zero and twelve"
            )
        if (
            isinstance(self.boundary_search_step_frames, bool)
            or not isinstance(self.boundary_search_step_frames, int)
            or self.boundary_search_step_frames <= 0
            or self.boundary_search_step_frames > 6
        ):
            raise ValueError(
                "boundary_search_step_frames must be between one and six"
            )
        if (
            self.boundary_search_radius_frames > 0
            and self.boundary_search_step_frames
            > self.boundary_search_radius_frames
        ):
            raise ValueError(
                "boundary search step cannot exceed its enabled radius"
            )
        if self.foreground_obstruction_rejection_score <= 0.0:
            raise ValueError(
                "foreground_obstruction_rejection_score must be positive"
            )
        if not (
            0.0
            <= self.blank_extreme_mean_minimum
            < self.blank_extreme_mean_maximum
            <= 1.0
        ):
            raise ValueError("blank extreme mean thresholds are invalid")
        for value in (
            self.minimum_pose_confidence,
            self.minimum_silhouette_confidence,
            self.attention_position_warning,
            self.attention_scale_warning,
            self.dominant_pose_warning,
            self.formation_warning,
            self.silhouette_warning,
            self.person_layout_warning,
            self.person_count_warning,
            self.total_mismatch_warning,
            self.severe_risk_margin,
            self.total_mismatch_severe,
            self.maximum_source_duration_relative_deviation,
            self.minimum_acceptable_quality_score,
            self.portrait_fancam_quality_rescue_mean_minimum,
            self.portrait_fancam_quality_rescue_contrast_minimum,
            self.foreground_obstruction_rejection_score,
            self.minimum_adjacent_motion_magnitude,
            self.motion_connection_preference_weight,
            self.priority_motion_connection_preference_weight,
            self.minimum_motion_direction_cosine,
            self.maximum_motion_speed_ratio_mismatch,
            self.maximum_pose_phase_continuation_mismatch,
            self.maximum_motion_endpoint_shape_mismatch,
            self.maximum_cross_source_pose_mismatch,
            self.maximum_cross_source_silhouette_mismatch,
            self.terminal_composite_exit_quality_maximum,
            self.terminal_composite_pre_exit_quality_maximum,
            self.terminal_composite_first_quality_drop_minimum,
            self.terminal_composite_exit_quality_drop_minimum,
            self.terminal_composite_first_contrast_drop_minimum,
            self.terminal_composite_exit_contrast_drop_minimum,
            self.persistent_entry_overexposure_quality_maximum,
            self.persistent_entry_overexposure_gray_mean_minimum,
            self.persistent_entry_overexposure_gray_contrast_maximum,
            self.persistent_entry_overexposure_highlight_minimum,
            self.portrait_fancam_scene_cut_continuity_structure_maximum,
            self.portrait_fancam_scene_cut_post_graphic_quality_maximum,
            self.portrait_fancam_scene_cut_post_graphic_gray_mean_minimum,
            self.portrait_fancam_scene_cut_post_graphic_contrast_maximum,
            self.transform_repair_maximum_boundary_center_distance,
            self.transform_repair_maximum_residual_center_distance,
            self.transform_repair_minimum_improvement_fraction,
            self.transform_repair_minimum_face_visible_fraction,
            self.transform_repair_minimum_subject_confidence,
            self.transform_repair_minimum_subject_visible_fraction,
            self.transform_alignment_position_trigger,
            self.transform_alignment_scale_trigger,
            self.transform_body_alignment_position_trigger,
            self.transform_body_alignment_scale_trigger,
        ):
            if value > 1.0:
                raise ValueError("direct member visual thresholds must be at most 1")

    def warning_thresholds(self) -> dict[str, float]:
        return {
            "attention_position": self.attention_position_warning,
            "attention_scale": self.attention_scale_warning,
            "dominant_pose": self.dominant_pose_warning,
            "formation": self.formation_warning,
            "silhouette": self.silhouette_warning,
            "person_layout": self.person_layout_warning,
            "person_count": self.person_count_warning,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_frame_step_seconds": self.sample_frame_step_seconds,
            "early_interior_sample_seconds": self.early_interior_sample_seconds,
            "sample_policy": (
                "outgoing_minus_two_to_minus_one_frames_and_"
                "incoming_exact_entry_to_plus_one_frame"
            ),
            "span_content_sample_policy": (
                "cut_plus_one_output_frame_cut_plus_0_20_seconds_quarter_midpoint_"
                "three_quarter_exit_interior"
            ),
            "span_interior_inset_seconds": self.span_interior_inset_seconds,
            "beam_width": self.beam_width,
            "maximum_adaptive_beam_width": self.maximum_adaptive_beam_width,
            "maximum_generated_beam_state_count": (
                self.maximum_generated_beam_state_count
            ),
            "feasibility_lifeline_state_limit": (
                self.feasibility_lifeline_state_limit
            ),
            "minimum_pose_confidence": self.minimum_pose_confidence,
            "minimum_silhouette_confidence": (
                self.minimum_silhouette_confidence
            ),
            "minimum_evidence_component_count": (
                self.minimum_evidence_component_count
            ),
            "weights": {
                "attention_position": self.attention_position_weight,
                "attention_scale": self.attention_scale_weight,
                "dominant_pose": self.dominant_pose_weight,
                "structure_pose_fallback": self.structure_pose_weight,
                "formation": self.formation_weight,
                "silhouette": self.silhouette_weight,
                "person_layout": self.person_layout_weight,
                "person_count": self.person_count_weight,
            },
            "warning_thresholds": self.warning_thresholds(),
            "total_mismatch_warning": self.total_mismatch_warning,
            "severe_risk_margin": self.severe_risk_margin,
            "total_mismatch_severe": self.total_mismatch_severe,
            "warning_penalty": self.warning_penalty,
            "severe_penalty": self.severe_penalty,
            "severe_edge_effect": "ineligible_fail_closed_if_quota_path_exhausted",
            "lossless_transform_repair": {
                "scope": (
                    "pipeline_verified_official_target_member_fancam_only"
                ),
                "maximum_zoom": self.transform_repair_maximum_zoom,
                "maximum_center_shift": (
                    self.transform_repair_maximum_center_shift
                ),
                "maximum_boundary_center_distance": (
                    self.transform_repair_maximum_boundary_center_distance
                ),
                "maximum_residual_center_distance": (
                    self.transform_repair_maximum_residual_center_distance
                ),
                "minimum_mismatch_improvement_fraction": (
                    self.transform_repair_minimum_improvement_fraction
                ),
                "minimum_face_visible_fraction": (
                    self.transform_repair_minimum_face_visible_fraction
                ),
                "legacy_face_only_maximum_zoom": (
                    self.transform_repair_legacy_face_only_maximum_zoom
                ),
                "legacy_face_only_maximum_center_shift": (
                    self.transform_repair_legacy_face_only_maximum_center_shift
                ),
                "minimum_widened_subject_confidence": (
                    self.transform_repair_minimum_subject_confidence
                ),
                "minimum_widened_subject_visible_fraction": (
                    self.transform_repair_minimum_subject_visible_fraction
                ),
                "required_position_alignment_trigger": (
                    self.transform_alignment_position_trigger
                ),
                "required_scale_alignment_trigger": (
                    self.transform_alignment_scale_trigger
                ),
                "required_body_position_alignment_trigger": (
                    self.transform_body_alignment_position_trigger
                ),
                "required_body_scale_alignment_trigger": (
                    self.transform_body_alignment_scale_trigger
                ),
                "resolution_policy": (
                    "post_portrait_fill_even_source_pixels_must_cover_"
                    "maximum_zoom_working_frame"
                ),
                "dense_every_output_frame_revalidation_required": True,
                "failed_dense_repair_effect": "fail_closed_before_render",
            },
            "insufficient_evidence_penalty": (
                self.insufficient_evidence_penalty
            ),
            "missing_descriptor_cost": self.missing_descriptor_cost,
            "ping_pong_penalty": self.ping_pong_penalty,
            "same_source_hold": {
                "contract": (
                    "bounded_no_cut_continuation_after_strict_cross_source_"
                    "path_failure_v1"
                ),
                "penalty": self.same_source_hold_penalty,
                "maximum_fraction": self.maximum_same_source_hold_fraction,
                "maximum_count": self.maximum_same_source_hold_count,
                "maximum_consecutive": (
                    self.maximum_consecutive_same_source_holds
                ),
                "maximum_same_source_run_seconds": (
                    self.maximum_same_source_run_seconds
                ),
                "severe_cross_source_edges_remain_ineligible": True,
                "all_balanced_source_quotas_remain_required": True,
            },
            "adjacent_motion_connection": {
                "contract": (
                    "cut_adjacent_timestamp_bound_pose_silhouette_subject_motion_v2"
                ),
                "sample_policy": (
                    "outgoing_previous_to_endpoint_and_incoming_start_to_next_"
                    "at_one_output_frame_spacing"
                ),
                "minimum_motion_magnitude": (
                    self.minimum_adjacent_motion_magnitude
                ),
                "minimum_temporal_component_count": (
                    self.minimum_temporal_motion_component_count
                ),
                "soft_preference_weight": (
                    self.motion_connection_preference_weight
                ),
                "priority_boundary_additional_preference_weight": (
                    self.priority_motion_connection_preference_weight
                ),
                "minimum_direction_cosine": (
                    self.minimum_motion_direction_cosine
                ),
                "maximum_motion_speed_ratio_mismatch": (
                    self.maximum_motion_speed_ratio_mismatch
                ),
                "maximum_pose_phase_continuation_mismatch": (
                    self.maximum_pose_phase_continuation_mismatch
                ),
                "maximum_endpoint_shape_mismatch": (
                    self.maximum_motion_endpoint_shape_mismatch
                ),
                "maximum_cross_source_pose_mismatch": (
                    self.maximum_cross_source_pose_mismatch
                ),
                "maximum_cross_source_silhouette_mismatch": (
                    self.maximum_cross_source_silhouette_mismatch
                ),
                "require_actual_pose_for_cross_source_cut": (
                    self.require_actual_pose_for_cross_source_cut
                ),
                "direction_cosine_policy": (
                    "hard_gate_for_both_moving_cross_source_cuts"
                ),
                "motion_speed_and_pose_phase_policy": (
                    "rank_and_hard_gate_when_temporal_components_are_observed"
                ),
                "screen_geometry_domain": (
                    "portrait_fill_plus_authorized_boundary_transform_output"
                ),
                "sparse_or_nonadjacent_evidence_allowed": False,
                "raw_vectors_persisted": False,
            },
            "boundary_local_search": {
                "contract": (
                    "near_beat_frame_offset_pose_motion_semimarkov_v1"
                ),
                "radius_frames": self.boundary_search_radius_frames,
                "step_frames": self.boundary_search_step_frames,
                "offset_penalty_per_frame": (
                    self.boundary_search_offset_penalty_per_frame
                ),
                "duration_balance_jointly_constrained": True,
                "source_path_selected_from_best_admissible_edge_state": True,
                "selected_offsets_materialized_in_render_timeline": True,
            },
            "duration_balance": {
                "target_policy": (
                    "total_duration_times_source_quota_divided_by_span_count"
                ),
                "normalized_squared_load_weight": self.duration_balance_weight,
                "maximum_source_relative_deviation": (
                    self.maximum_source_duration_relative_deviation
                ),
                "threshold_effect": "terminal_fail_closed",
            },
            "content_validity": {
                "sample_count": _DIRECT_CONTENT_SAMPLE_COUNT,
                "sample_policy": (
                    "cut_plus_one_output_frame_cut_plus_0_20_seconds_"
                    "quarter_midpoint_three_quarter_exit_minus_0_18_seconds"
                ),
                "minimum_face_sample_count": self.minimum_face_sample_count,
                "minimum_face_sample_backend": "verified_yunet",
                "minimum_human_sample_count": self.minimum_human_sample_count,
                "minimum_acceptable_quality_score": (
                    self.minimum_acceptable_quality_score
                ),
                "minimum_acceptable_quality_sample_count": (
                    self.minimum_acceptable_quality_sample_count
                ),
                "continuous_official_portrait_fancam_quality_rescue": {
                    "scope": (
                        "exact_pipeline_verified_official_target_member_"
                        "portrait_fancam_contract_only"
                    ),
                    "minimum_mean_quality": (
                        self.portrait_fancam_quality_rescue_mean_minimum
                    ),
                    "minimum_every_sample_gray_contrast": (
                        self.portrait_fancam_quality_rescue_contrast_minimum
                    ),
                    "required_human_samples": _DIRECT_CONTENT_SAMPLE_COUNT,
                    "required_verified_yunet_face_samples": 1,
                    "requires_no_obstruction_blank_composite_or_"
                    "persistent_overexposure": True,
                    "generic_quality_threshold_lowered": False,
                },
                "foreground_obstruction_rejection_score": (
                    self.foreground_obstruction_rejection_score
                ),
                "foreground_obstruction_policy": (
                    "hard_reject_any_of_six_temporal_samples_with_"
                    "dark_edge_connected_camera_equipment_over_subject_"
                    "or_performance_area"
                ),
                "terminal_broadcast_composite_policy": {
                    "exit_quality_maximum": (
                        self.terminal_composite_exit_quality_maximum
                    ),
                    "pre_exit_quality_maximum": (
                        self.terminal_composite_pre_exit_quality_maximum
                    ),
                    "first_quality_drop_minimum": (
                        self.terminal_composite_first_quality_drop_minimum
                    ),
                    "exit_quality_drop_minimum": (
                        self.terminal_composite_exit_quality_drop_minimum
                    ),
                    "first_gray_contrast_drop_minimum": (
                        self.terminal_composite_first_contrast_drop_minimum
                    ),
                    "exit_gray_contrast_drop_minimum": (
                        self.terminal_composite_exit_contrast_drop_minimum
                    ),
                    "effect": "hard_reject",
                },
                "persistent_entry_overexposure_policy": {
                    "early_interior_seconds": self.early_interior_sample_seconds,
                    "quality_maximum": (
                        self.persistent_entry_overexposure_quality_maximum
                    ),
                    "gray_mean_minimum": (
                        self.persistent_entry_overexposure_gray_mean_minimum
                    ),
                    "gray_contrast_maximum": (
                        self.persistent_entry_overexposure_gray_contrast_maximum
                    ),
                    "highlight_clipping_minimum": (
                        self.persistent_entry_overexposure_highlight_minimum
                    ),
                    "required_consecutive_samples": 2,
                    "effect": "hard_reject",
                },
                "require_entry_exit_human_evidence": (
                    self.require_entry_exit_human_evidence
                ),
                "native_scene_cut_policy": {
                    "default": (
                        "hard_reject_source_native_cut_strictly_inside_"
                        "coarsened_span_after_one_output_frame_"
                        "boundary_quantization_tolerance"
                    ),
                    "continuous_official_portrait_fancam_exception": (
                        "suppress_only_when_four_flank_frames_preserve_"
                        "human_performance_and_low_structure_delta_with_"
                        "no_obstruction_or_persistent_post_cut_graphic"
                    ),
                    "continuity_window_seconds": (
                        self.portrait_fancam_scene_cut_continuity_window_seconds
                    ),
                    "continuity_structure_mismatch_maximum": (
                        self.portrait_fancam_scene_cut_continuity_structure_maximum
                    ),
                    "persistent_post_graphic_quality_maximum": (
                        self.portrait_fancam_scene_cut_post_graphic_quality_maximum
                    ),
                    "persistent_post_graphic_gray_mean_minimum": (
                        self.portrait_fancam_scene_cut_post_graphic_gray_mean_minimum
                    ),
                    "persistent_post_graphic_contrast_maximum": (
                        self.portrait_fancam_scene_cut_post_graphic_contrast_maximum
                    ),
                    "identity_tracking_used": False,
                    "uncertain_evidence_effect": "retain_hard_reject",
                },
                "native_scene_cut_boundary_tolerance_seconds": (
                    self.native_scene_cut_boundary_tolerance_seconds
                ),
                "blank_gray_std_maximum": self.blank_gray_std_maximum,
                "blank_extreme_mean_range": [
                    self.blank_extreme_mean_minimum,
                    self.blank_extreme_mean_maximum,
                ],
                "static_structure_mismatch_maximum": (
                    self.static_structure_mismatch_maximum
                ),
                "static_color_mismatch_maximum": (
                    self.static_color_mismatch_maximum
                ),
                "invalid_candidate_effect": "hard_reject",
                "quota_conflict_effect": "fail_closed",
            },
        }


@dataclass(frozen=True, slots=True)
class _AttentionGeometry:
    bbox_xywh: tuple[float, float, float, float]
    centroid_xy: tuple[float, float]
    basis: str


@dataclass(frozen=True, slots=True)
class _BoundarySample:
    master_seconds: float
    source_local_seconds: float
    descriptor: FrameVisualDescriptor | None
    attention: _AttentionGeometry | None

    def to_dict(self) -> dict[str, Any]:
        descriptor = self.descriptor
        return {
            "canonical_master_seconds": self.master_seconds,
            "source_local_seconds": self.source_local_seconds,
            "descriptor_available": descriptor is not None,
            "attention_basis": (
                self.attention.basis if self.attention is not None else None
            ),
            "attention_bbox_xywh": (
                list(self.attention.bbox_xywh)
                if self.attention is not None
                else None
            ),
            "attention_centroid_xy": (
                list(self.attention.centroid_xy)
                if self.attention is not None
                else None
            ),
            "pose_backend": descriptor.pose_backend if descriptor else "none",
            "face_count": descriptor.face_count if descriptor else 0,
            "face_detector_backend": (
                descriptor.face_detector_backend if descriptor else "none"
            ),
            "pose_confidence": descriptor.pose_confidence if descriptor else 0.0,
            "detected_person_count": (
                descriptor.detected_person_count if descriptor else 0
            ),
            "formation_person_count": (
                len(descriptor.formation_people) if descriptor else 0
            ),
            "formation_upper_body_count": (
                sum(
                    bool(person.upper_body_keypoints_xyc)
                    for person in descriptor.formation_people
                )
                if descriptor
                else 0
            ),
            "silhouette_backend": (
                descriptor.silhouette_backend if descriptor else "none"
            ),
            "silhouette_confidence": (
                descriptor.silhouette_confidence if descriptor else 0.0
            ),
            "foreground_obstruction_score": (
                descriptor.foreground_obstruction_score if descriptor else 0.0
            ),
            "highlight_clipping_ratio": (
                descriptor.highlight_clipping_ratio if descriptor else 0.0
            ),
            "raw_embeddings_persisted": False,
        }


@dataclass(frozen=True, slots=True)
class DirectBoundaryMotionEvidence:
    """Scalar-only motion evidence from the four cut-adjacent frames."""

    timestamp_bound_adjacent: bool
    outgoing_previous_master_seconds: float
    outgoing_endpoint_master_seconds: float
    incoming_start_master_seconds: float
    incoming_next_master_seconds: float
    outgoing_previous_source_local_seconds: float
    outgoing_endpoint_source_local_seconds: float
    incoming_start_source_local_seconds: float
    incoming_next_source_local_seconds: float
    outgoing_pose_motion_magnitude: float | None
    incoming_pose_motion_magnitude: float | None
    outgoing_silhouette_motion_magnitude: float | None
    incoming_silhouette_motion_magnitude: float | None
    outgoing_subject_motion_magnitude: float | None
    incoming_subject_motion_magnitude: float | None
    outgoing_motion_magnitude: float | None
    incoming_motion_magnitude: float | None
    outgoing_motion_component_count: int
    incoming_motion_component_count: int
    both_moving: bool
    endpoint_pose_shape_mismatch: float | None
    endpoint_pose_shape_compatibility: float | None
    direction_cosine_audit_only: float | None
    motion_speed_ratio_mismatch: float | None
    pose_phase_continuation_mismatch: float | None
    output_space_subject_motion_used: bool
    soft_preference_applied: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": (
                "cut_adjacent_exact_entry_output_space_pose_motion_v3"
            ),
            "timestamp_bound_adjacent": self.timestamp_bound_adjacent,
            "master_timestamps_seconds": {
                "outgoing_previous": self.outgoing_previous_master_seconds,
                "outgoing_endpoint": self.outgoing_endpoint_master_seconds,
                "incoming_start": self.incoming_start_master_seconds,
                "incoming_next": self.incoming_next_master_seconds,
            },
            "source_local_timestamps_seconds": {
                "outgoing_previous": (
                    self.outgoing_previous_source_local_seconds
                ),
                "outgoing_endpoint": (
                    self.outgoing_endpoint_source_local_seconds
                ),
                "incoming_start": self.incoming_start_source_local_seconds,
                "incoming_next": self.incoming_next_source_local_seconds,
            },
            "motion_magnitudes": {
                "outgoing_pose": self.outgoing_pose_motion_magnitude,
                "incoming_pose": self.incoming_pose_motion_magnitude,
                "outgoing_silhouette": (
                    self.outgoing_silhouette_motion_magnitude
                ),
                "incoming_silhouette": (
                    self.incoming_silhouette_motion_magnitude
                ),
                "outgoing_subject": self.outgoing_subject_motion_magnitude,
                "incoming_subject": self.incoming_subject_motion_magnitude,
                "outgoing_aggregate": self.outgoing_motion_magnitude,
                "incoming_aggregate": self.incoming_motion_magnitude,
            },
            "motion_component_counts": {
                "outgoing": self.outgoing_motion_component_count,
                "incoming": self.incoming_motion_component_count,
            },
            "both_moving": self.both_moving,
            "endpoint_pose_shape_mismatch": (
                self.endpoint_pose_shape_mismatch
            ),
            "endpoint_pose_shape_compatibility": (
                self.endpoint_pose_shape_compatibility
            ),
            "direction_cosine_audit_only": (
                self.direction_cosine_audit_only
            ),
            # Keep the v1 key for old analysis readers, but the value is no
            # longer audit-only: v2 uses it both to authorize the moving-cut
            # preference and to reject opposed cross-source motion.
            "direction_cosine": self.direction_cosine_audit_only,
            "direction_cosine_influenced_gate_or_bonus": True,
            "motion_speed_ratio_mismatch": (
                self.motion_speed_ratio_mismatch
            ),
            "pose_phase_continuation_mismatch": (
                self.pose_phase_continuation_mismatch
            ),
            "output_space_subject_motion_used": (
                self.output_space_subject_motion_used
            ),
            "motion_speed_and_pose_phase_influenced_gate_or_bonus": True,
            "soft_preference_applied": self.soft_preference_applied,
            "sparse_or_nonadjacent_evidence_used": False,
            "raw_vectors_persisted": False,
        }


@dataclass(frozen=True, slots=True)
class DirectSpanSourceValidity:
    eligible_source_index: int
    valid: bool
    rejection_reasons: tuple[str, ...]
    samples: tuple[_BoundarySample, ...]
    human_evidence_sample_count: int
    face_evidence_sample_count: int
    acceptable_quality_sample_count: int
    blank_sample_count: int
    static_visual_candidate: bool
    native_scene_cut_source_seconds: tuple[float, ...]
    native_scene_cut_master_seconds: tuple[float, ...]
    suppressed_portrait_fancam_scene_cut_source_seconds: tuple[float, ...]
    suppressed_portrait_fancam_scene_cut_master_seconds: tuple[float, ...]
    maximum_structure_mismatch: float | None
    maximum_color_mismatch: float | None
    mean_quality_score: float | None
    quality_sample_scores: tuple[float, ...]
    foreground_obstruction_sample_count: int
    foreground_obstruction_sample_scores: tuple[float, ...]
    terminal_composite_transition_candidate: bool
    gray_contrast_sample_scores: tuple[float | None, ...]
    persistent_entry_overexposure_candidate: bool
    gray_mean_sample_scores: tuple[float | None, ...]
    highlight_clipping_sample_scores: tuple[float, ...]
    portrait_fancam_lighting_quality_rescue_applied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible_source_index": self.eligible_source_index,
            "valid": self.valid,
            "rejection_reasons": list(self.rejection_reasons),
            "human_evidence_sample_count": self.human_evidence_sample_count,
            "face_evidence_sample_count": self.face_evidence_sample_count,
            "acceptable_quality_sample_count": (
                self.acceptable_quality_sample_count
            ),
            "blank_sample_count": self.blank_sample_count,
            "static_visual_candidate": self.static_visual_candidate,
            "native_scene_cut_source_seconds": list(
                self.native_scene_cut_source_seconds
            ),
            "native_scene_cut_master_seconds": list(
                self.native_scene_cut_master_seconds
            ),
            "suppressed_portrait_fancam_scene_cut_source_seconds": list(
                self.suppressed_portrait_fancam_scene_cut_source_seconds
            ),
            "suppressed_portrait_fancam_scene_cut_master_seconds": list(
                self.suppressed_portrait_fancam_scene_cut_master_seconds
            ),
            "maximum_structure_mismatch": self.maximum_structure_mismatch,
            "maximum_color_mismatch": self.maximum_color_mismatch,
            "mean_quality_score": self.mean_quality_score,
            "quality_sample_scores": list(self.quality_sample_scores),
            "foreground_obstruction_sample_count": (
                self.foreground_obstruction_sample_count
            ),
            "foreground_obstruction_sample_scores": list(
                self.foreground_obstruction_sample_scores
            ),
            "terminal_composite_transition_candidate": (
                self.terminal_composite_transition_candidate
            ),
            "gray_contrast_sample_scores": list(
                self.gray_contrast_sample_scores
            ),
            "persistent_entry_overexposure_candidate": (
                self.persistent_entry_overexposure_candidate
            ),
            "gray_mean_sample_scores": list(self.gray_mean_sample_scores),
            "highlight_clipping_sample_scores": list(
                self.highlight_clipping_sample_scores
            ),
            "portrait_fancam_lighting_quality_rescue_applied": (
                self.portrait_fancam_lighting_quality_rescue_applied
            ),
            "samples": [sample.to_dict() for sample in self.samples],
            "raw_embeddings_persisted": False,
        }


@dataclass(frozen=True, slots=True)
class DirectSpanVisualDiagnostic:
    segment_index: int
    canonical_master_start: float
    canonical_master_end: float
    sources: tuple[DirectSpanSourceValidity, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "canonical_master_start": self.canonical_master_start,
            "canonical_master_end": self.canonical_master_end,
            "sources": [source.to_dict() for source in self.sources],
            "valid_source_count": sum(source.valid for source in self.sources),
        }


@dataclass(frozen=True, slots=True)
class DirectMemberCinematicBridge:
    """A required effect contract for a landscape/portrait member boundary.

    This does not make the edge eligible for a hard cut.  It records that the
    target pose and silhouette are compatible while the remaining mismatch is
    confined to framing components that a short zoom/blur overlap is designed
    to conceal.  The transition planner must independently bind the overlap to
    exact source clocks and a content-valid outgoing handle.
    """

    style: str
    outgoing_is_landscape: bool
    incoming_is_landscape: bool
    authorized_severe_flags: tuple[str, ...]
    pose_mismatch: float
    silhouette_mismatch: float
    base_mismatch: float

    def __post_init__(self) -> None:
        if self.style not in {"zoom_in", "horizontal_blur"}:
            raise ValueError("direct member cinematic bridge style is invalid")
        if self.outgoing_is_landscape == self.incoming_is_landscape:
            raise ValueError("cinematic bridge requires one landscape source")
        flags = frozenset(self.authorized_severe_flags)
        if not flags or not flags <= (
            _DIRECT_MEMBER_CINEMATIC_BRIDGE_ALLOWED_SEVERE_FLAGS
        ):
            raise ValueError("cinematic bridge severe flags are invalid")
        for value in (
            self.pose_mismatch,
            self.silhouette_mismatch,
            self.base_mismatch,
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("cinematic bridge mismatch is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": (
                "official_member_landscape_portrait_pose_safe_effect_bridge_v1"
            ),
            "style": self.style,
            "outgoing_is_landscape": self.outgoing_is_landscape,
            "incoming_is_landscape": self.incoming_is_landscape,
            "authorized_severe_flags": list(self.authorized_severe_flags),
            "pose_mismatch": self.pose_mismatch,
            "silhouette_mismatch": self.silhouette_mismatch,
            "base_mismatch": self.base_mismatch,
            "hard_cut_authorized": False,
            "required_transition_effect": True,
            "identity_inference_used": False,
        }


@dataclass(frozen=True, slots=True)
class DirectBoundaryMismatch:
    cost: float
    base_mismatch: float
    attention_position_mismatch: float | None
    attention_scale_mismatch: float | None
    dominant_pose_mismatch: float | None
    pose_basis: str | None
    formation_mismatch: float | None
    silhouette_mismatch: float | None
    person_layout_mismatch: float | None
    person_count_mismatch: float | None
    evidence_component_count: int
    warning_flags: tuple[str, ...]
    severe_flags: tuple[str, ...]
    motion_connection: DirectBoundaryMotionEvidence | None = None
    transform_repair: DirectMemberBoundaryTransformRepair | None = None
    member_context_softened_flags: tuple[str, ...] = ()
    cinematic_bridge: DirectMemberCinematicBridge | None = None
    boundary_search_offset_frames: int = 0
    boundary_search_offset_seconds: float = 0.0
    boundary_search_offset_penalty: float = 0.0
    boundary_search_candidate_count: int = 1
    priority_motion_preference_applied: float = 0.0

    def components(self) -> dict[str, float | None]:
        return {
            "attention_position": self.attention_position_mismatch,
            "attention_scale": self.attention_scale_mismatch,
            "dominant_pose": self.dominant_pose_mismatch,
            "formation": self.formation_mismatch,
            "silhouette": self.silhouette_mismatch,
            "person_layout": self.person_layout_mismatch,
            "person_count": self.person_count_mismatch,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost": self.cost,
            "base_mismatch": self.base_mismatch,
            "components": self.components(),
            "pose_basis": self.pose_basis,
            "evidence_component_count": self.evidence_component_count,
            "warning_flags": list(self.warning_flags),
            "severe_flags": list(self.severe_flags),
            "motion_connection": (
                self.motion_connection.to_dict()
                if self.motion_connection is not None
                else None
            ),
            "transform_repair": (
                self.transform_repair.to_dict()
                if self.transform_repair is not None
                else None
            ),
            "member_context_softened_flags": list(
                self.member_context_softened_flags
            ),
            "cinematic_bridge": (
                self.cinematic_bridge.to_dict()
                if self.cinematic_bridge is not None
                else None
            ),
            "boundary_local_search": {
                "selected_offset_frames": self.boundary_search_offset_frames,
                "selected_offset_seconds": self.boundary_search_offset_seconds,
                "offset_penalty": self.boundary_search_offset_penalty,
                "candidate_count": self.boundary_search_candidate_count,
                "cost_includes_offset_penalty": True,
            },
            "priority_motion_preference_applied": (
                self.priority_motion_preference_applied
            ),
        }


@dataclass(frozen=True, slots=True)
class DirectBoundaryVisualDiagnostic:
    boundary_index: int
    canonical_master_seconds: float
    selected_cut_master_seconds: float
    outgoing_eligible_source_index: int
    incoming_eligible_source_index: int
    outgoing_sample: _BoundarySample
    incoming_sample: _BoundarySample
    mismatch: DirectBoundaryMismatch
    selected_pair_rank: int
    candidate_pair_count: int
    raw_cross_source_pair_count: int
    invalid_content_pair_count: int
    severe_pair_count: int
    minimum_pair_cost: float
    selected_pair_cost_delta: float
    ping_pong_penalty_applied: bool
    cut_applied: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_index": self.boundary_index,
            "canonical_master_seconds": self.canonical_master_seconds,
            "selected_cut_master_seconds": self.selected_cut_master_seconds,
            "selected_cut_offset_seconds": (
                self.selected_cut_master_seconds - self.canonical_master_seconds
            ),
            "outgoing_eligible_source_index": (
                self.outgoing_eligible_source_index
            ),
            "incoming_eligible_source_index": (
                self.incoming_eligible_source_index
            ),
            "outgoing_sample": self.outgoing_sample.to_dict(),
            "incoming_sample": self.incoming_sample.to_dict(),
            "mismatch": self.mismatch.to_dict(),
            "selected_pair_rank": self.selected_pair_rank,
            "candidate_pair_count": self.candidate_pair_count,
            "raw_cross_source_pair_count": self.raw_cross_source_pair_count,
            "invalid_content_pair_count": self.invalid_content_pair_count,
            "severe_pair_count": self.severe_pair_count,
            "minimum_pair_cost": self.minimum_pair_cost,
            "selected_pair_cost_delta": self.selected_pair_cost_delta,
            "ping_pong_penalty_applied": self.ping_pong_penalty_applied,
            "cut_applied": self.cut_applied,
        }


@dataclass(frozen=True, slots=True)
class _DirectBoundarySearchCandidate:
    """One near-beat cut state shared by every source-pair edge."""

    offset_frames: int
    cut_master_seconds: float
    outgoing_previous: tuple[_BoundarySample, ...]
    outgoing: tuple[_BoundarySample, ...]
    incoming: tuple[_BoundarySample, ...]
    incoming_next: tuple[_BoundarySample, ...]
    matrix: tuple[tuple[DirectBoundaryMismatch, ...], ...]


def _boundary_search_frame_offsets(
    config: DirectMemberVisualConfig,
) -> tuple[int, ...]:
    radius = config.boundary_search_radius_frames
    if radius == 0:
        return (0,)
    positive = list(range(config.boundary_search_step_frames, radius + 1, config.boundary_search_step_frames))
    if not positive or positive[-1] != radius:
        positive.append(radius)
    values = {0, *positive, *(-value for value in positive)}
    return tuple(sorted(values, key=lambda value: (abs(value), value)))


def _scored_boundary_search_mismatch(
    candidate: _DirectBoundarySearchCandidate,
    *,
    outgoing_source_index: int,
    incoming_source_index: int,
    candidate_count: int,
    priority_boundary: bool,
    config: DirectMemberVisualConfig,
) -> DirectBoundaryMismatch:
    mismatch = candidate.matrix[outgoing_source_index][incoming_source_index]
    penalty = (
        abs(candidate.offset_frames)
        * config.boundary_search_offset_penalty_per_frame
    )
    motion = mismatch.motion_connection
    priority_motion_preference = 0.0
    if (
        priority_boundary
        and outgoing_source_index != incoming_source_index
        and motion is not None
        and motion.both_moving
        and motion.soft_preference_applied > 0.0
        and config.motion_connection_preference_weight > 1e-12
    ):
        normalized_motion_quality = min(
            1.0,
            motion.soft_preference_applied
            / config.motion_connection_preference_weight,
        )
        priority_motion_preference = (
            config.priority_motion_connection_preference_weight
            * normalized_motion_quality
        )
    return replace(
        mismatch,
        cost=mismatch.cost + penalty - priority_motion_preference,
        boundary_search_offset_frames=candidate.offset_frames,
        boundary_search_offset_seconds=(
            candidate.offset_frames * config.sample_frame_step_seconds
        ),
        boundary_search_offset_penalty=penalty,
        boundary_search_candidate_count=candidate_count,
        priority_motion_preference_applied=priority_motion_preference,
    )


def _best_boundary_search_edge(
    candidates: Sequence[_DirectBoundarySearchCandidate],
    *,
    outgoing_source_index: int,
    incoming_source_index: int,
    priority_boundary: bool,
    config: DirectMemberVisualConfig,
) -> tuple[_DirectBoundarySearchCandidate, DirectBoundaryMismatch]:
    if not candidates:
        raise ValueError("boundary local search needs at least one candidate")
    candidate_count = len(candidates)
    scored = tuple(
        (
            candidate,
            _scored_boundary_search_mismatch(
                candidate,
                outgoing_source_index=outgoing_source_index,
                incoming_source_index=incoming_source_index,
                candidate_count=candidate_count,
                priority_boundary=priority_boundary,
                config=config,
            ),
        )
        for candidate in candidates
        if (
            outgoing_source_index != incoming_source_index
            or candidate.offset_frames == 0
        )
    )
    if not scored:
        raise RuntimeError("boundary local search lost its zero-offset hold state")
    return min(
        scored,
        key=lambda item: (
            bool(item[1].severe_flags),
            item[1].cost,
            abs(item[0].offset_frames),
            item[0].offset_frames,
        ),
    )


def _boundary_adjusted_span_durations(
    spans: Sequence[tuple[float, float]],
    selected_candidates: Sequence[_DirectBoundarySearchCandidate],
) -> tuple[float, ...]:
    boundaries = tuple(
        candidate.cut_master_seconds for candidate in selected_candidates
    )
    edges = (float(spans[0][0]), *boundaries, float(spans[-1][1]))
    durations = tuple(
        right - left for left, right in zip(edges, edges[1:])
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in durations):
        raise ValueError("boundary local search produced a non-positive span")
    return durations


def _joint_boundary_search_assignment(
    *,
    candidates_by_boundary: Sequence[Sequence[_DirectBoundarySearchCandidate]],
    path: Sequence[int],
    spans: Sequence[tuple[float, float]],
    source_duration_targets: Sequence[float],
    source_duration_minimums: Sequence[float],
    source_duration_maximums: Sequence[float],
    usage_policy: str,
    priority_boundary_indices: frozenset[int],
    config: DirectMemberVisualConfig,
) -> tuple[
    tuple[_DirectBoundarySearchCandidate, ...],
    tuple[DirectBoundaryMismatch, ...],
    dict[str, Any],
]:
    """Choose all near-beat cut states under joint exposure constraints.

    Source selection sees the best admissible state for every edge. Once that
    path is known, this binary semi-Markov layer chooses the concrete state at
    every boundary together: a positive offset gives time to the outgoing
    source and removes the same time from the incoming source. The duration
    bounds therefore remain hard rather than being audited only after render.
    """

    if len(candidates_by_boundary) != len(path) - 1:
        raise ValueError("boundary search candidates must match the source path")
    source_count = len(source_duration_targets)
    if not (
        len(source_duration_minimums)
        == len(source_duration_maximums)
        == source_count
    ):
        raise ValueError("boundary search duration bounds must match sources")

    choices: list[
        tuple[tuple[_DirectBoundarySearchCandidate, DirectBoundaryMismatch], ...]
    ] = []
    for boundary_index, (left, right) in enumerate(zip(path, path[1:])):
        candidate_count = len(candidates_by_boundary[boundary_index])
        scored = tuple(
            (
                candidate,
                _scored_boundary_search_mismatch(
                    candidate,
                    outgoing_source_index=left,
                    incoming_source_index=right,
                    candidate_count=candidate_count,
                    priority_boundary=(
                        boundary_index in priority_boundary_indices
                    ),
                    config=config,
                ),
            )
            for candidate in candidates_by_boundary[boundary_index]
            if (left != right or candidate.offset_frames == 0)
        )
        if left != right:
            scored = tuple(
                item
                for item in scored
                if (
                    not item[1].severe_flags
                    or (
                        usage_policy
                        == "heterogeneous_role_aware_cinematic_v1"
                        and item[1].cinematic_bridge is not None
                    )
                )
            )
        if not scored:
            raise DirectMemberVisualSelectionError(
                "selected source path has no admissible local cut state",
                {
                    "failure": "boundary_local_search_edge_exhausted",
                    "boundary_index": boundary_index,
                    "outgoing_source_index": left,
                    "incoming_source_index": right,
                    "visual_gates_relaxed": False,
                    "constraint_infeasibility_proven": True,
                },
            )
        choices.append(scored)

    if all(len(values) == 1 for values in choices):
        selected = tuple(values[0] for values in choices)
        offsets = tuple(item[0].offset_frames for item in selected)
        return (
            tuple(item[0] for item in selected),
            tuple(item[1] for item in selected),
            {
                "contract": "near_beat_frame_offset_pose_motion_semimarkov_v1",
                "enabled": config.boundary_search_radius_frames > 0,
                "strategy": "single_admissible_state_per_boundary",
                "candidate_state_count": sum(len(values) for values in choices),
                "candidate_count_by_boundary": [
                    len(values) for values in choices
                ],
                "selected_offset_frames": list(offsets),
                "selected_nonzero_offset_count": sum(
                    value != 0 for value in offsets
                ),
                "selected_absolute_offset_frames": sum(
                    abs(value) for value in offsets
                ),
                "maximum_absolute_offset_frames": max(
                    (abs(value) for value in offsets),
                    default=0,
                ),
                "solver_invoked": False,
                "duration_constraints_jointly_enforced": True,
                "visual_gates_relaxed": False,
            },
        )

    flat: list[
        tuple[
            int,
            int,
            _DirectBoundarySearchCandidate,
            DirectBoundaryMismatch,
        ]
    ] = []
    variables_by_boundary: list[list[int]] = []
    for boundary_index, values in enumerate(choices):
        indices: list[int] = []
        for choice_index, (candidate, mismatch) in enumerate(values):
            indices.append(len(flat))
            flat.append(
                (
                    boundary_index,
                    choice_index,
                    candidate,
                    mismatch,
                )
            )
        variables_by_boundary.append(indices)

    base_durations = tuple(float(end) - float(start) for start, end in spans)
    base_source_durations = tuple(
        sum(
            duration
            for duration, source_index in zip(base_durations, path, strict=True)
            if source_index == target_source
        )
        for target_source in range(source_count)
    )

    def _assignment_is_valid(
        selected: Sequence[
            tuple[_DirectBoundarySearchCandidate, DirectBoundaryMismatch]
        ],
    ) -> bool:
        durations = _boundary_adjusted_span_durations(
            spans,
            tuple(item[0] for item in selected),
        )
        minimum_span = 4.0 * config.sample_frame_step_seconds
        if any(value < minimum_span - 1e-9 for value in durations):
            return False
        actual = tuple(
            sum(
                duration
                for duration, source_index in zip(durations, path, strict=True)
                if source_index == target_source
            )
            for target_source in range(source_count)
        )
        return all(
            minimum - 1e-9 <= value <= maximum + 1e-9
            for value, minimum, maximum in zip(
                actual,
                source_duration_minimums,
                source_duration_maximums,
                strict=True,
            )
        )

    greedy = tuple(
        min(
            values,
            key=lambda item: (
                item[1].cost,
                abs(item[0].offset_frames),
                item[0].offset_frames,
            ),
        )
        for values in choices
    )
    if _assignment_is_valid(greedy):
        selected = greedy
        strategy = "independent_minimum_states_joint_constraints_already_met"
        solver: dict[str, Any] = {"invoked": False, "success": True}
    else:
        try:
            import numpy as np
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import coo_array
        except (ImportError, OSError):
            zero = tuple(
                next(
                    (
                        item
                        for item in values
                        if item[0].offset_frames == 0
                    ),
                    values[0],
                )
                for values in choices
            )
            if not _assignment_is_valid(zero):
                raise DirectMemberVisualSelectionError(
                    "boundary local search duration assignment is unavailable",
                    {
                        "failure": "boundary_local_search_solver_unavailable",
                        "visual_gates_relaxed": False,
                        "constraint_infeasibility_proven": False,
                    },
                )
            selected = zero
            strategy = "zero_offset_duration_safe_fallback_without_scipy"
            solver = {"invoked": False, "success": True, "backend": None}
        else:
            variable_count = len(flat)
            objective = np.asarray(
                [
                    mismatch.cost
                    + abs(candidate.offset_frames) * 1e-9
                    + index * 1e-12
                    for index, (
                        _boundary,
                        _choice,
                        candidate,
                        mismatch,
                    ) in enumerate(flat)
                ],
                dtype=np.float64,
            )
            rows: list[int] = []
            columns: list[int] = []
            coefficients: list[float] = []
            lower: list[float] = []
            upper: list[float] = []

            def add_constraint(
                terms: Sequence[tuple[int, float]],
                minimum: float,
                maximum: float,
            ) -> None:
                row = len(lower)
                for variable, coefficient in terms:
                    rows.append(row)
                    columns.append(variable)
                    coefficients.append(float(coefficient))
                lower.append(float(minimum))
                upper.append(float(maximum))

            for indices in variables_by_boundary:
                add_constraint(
                    tuple((index, 1.0) for index in indices),
                    1.0,
                    1.0,
                )
            for source_index in range(source_count):
                terms = []
                for variable, (
                    boundary_index,
                    _choice_index,
                    candidate,
                    _mismatch,
                ) in enumerate(flat):
                    left = path[boundary_index]
                    right = path[boundary_index + 1]
                    delta = (
                        candidate.offset_frames
                        * config.sample_frame_step_seconds
                    )
                    coefficient = delta * (
                        int(left == source_index) - int(right == source_index)
                    )
                    if abs(coefficient) > 1e-12:
                        terms.append((variable, coefficient))
                add_constraint(
                    tuple(terms),
                    source_duration_minimums[source_index]
                    - base_source_durations[source_index],
                    source_duration_maximums[source_index]
                    - base_source_durations[source_index],
                )
            minimum_span = 4.0 * config.sample_frame_step_seconds
            for span_index, base_duration in enumerate(base_durations):
                terms = []
                for variable, (
                    boundary_index,
                    _choice_index,
                    candidate,
                    _mismatch,
                ) in enumerate(flat):
                    coefficient = 0.0
                    delta = (
                        candidate.offset_frames
                        * config.sample_frame_step_seconds
                    )
                    if boundary_index == span_index:
                        coefficient += delta
                    if boundary_index + 1 == span_index:
                        coefficient -= delta
                    if abs(coefficient) > 1e-12:
                        terms.append((variable, coefficient))
                add_constraint(
                    tuple(terms),
                    minimum_span - base_duration,
                    math.inf,
                )
            matrix = coo_array(
                (
                    np.asarray(coefficients, dtype=np.float64),
                    (
                        np.asarray(rows, dtype=np.int32),
                        np.asarray(columns, dtype=np.int32),
                    ),
                ),
                shape=(len(lower), variable_count),
            ).tocsr()
            result = milp(
                objective,
                integrality=np.ones(variable_count, dtype=np.int32),
                bounds=Bounds(
                    np.zeros(variable_count, dtype=np.float64),
                    np.ones(variable_count, dtype=np.float64),
                ),
                constraints=LinearConstraint(
                    matrix,
                    np.asarray(lower, dtype=np.float64),
                    np.asarray(upper, dtype=np.float64),
                ),
                options={"disp": False, "presolve": True, "mip_rel_gap": 0.0},
            )
            solver = {
                "invoked": True,
                "backend": "scipy_optimize_milp_highs",
                "status": int(result.status),
                "message": str(result.message),
                "success": bool(result.success),
                "variable_count": variable_count,
                "constraint_count": len(lower),
            }
            if not result.success or result.x is None:
                raise DirectMemberVisualSelectionError(
                    "boundary local search cannot satisfy duration constraints",
                    {
                        "failure": "boundary_local_search_duration_infeasible",
                        "solver": solver,
                        "visual_gates_relaxed": False,
                        "constraint_infeasibility_proven": int(result.status) == 2,
                    },
                )
            selected_values: list[
                tuple[_DirectBoundarySearchCandidate, DirectBoundaryMismatch]
            ] = []
            for indices, values in zip(
                variables_by_boundary,
                choices,
                strict=True,
            ):
                selected_index = max(indices, key=lambda index: result.x[index])
                selected_values.append(values[flat[selected_index][1]])
            selected = tuple(selected_values)
            if not _assignment_is_valid(selected):
                raise RuntimeError(
                    "boundary local search solver result failed independent audit"
                )
            strategy = "scipy_highs_binary_boundary_state_semimarkov_v1"

    selected_candidates = tuple(item[0] for item in selected)
    selected_mismatches = tuple(item[1] for item in selected)
    offsets = tuple(candidate.offset_frames for candidate in selected_candidates)
    return (
        selected_candidates,
        selected_mismatches,
        {
            "contract": "near_beat_frame_offset_pose_motion_semimarkov_v1",
            "enabled": config.boundary_search_radius_frames > 0,
            "strategy": strategy,
            "candidate_state_count": len(flat),
            "candidate_count_by_boundary": [len(values) for values in choices],
            "selected_offset_frames": list(offsets),
            "selected_nonzero_offset_count": sum(value != 0 for value in offsets),
            "selected_absolute_offset_frames": sum(abs(value) for value in offsets),
            "maximum_absolute_offset_frames": max(
                (abs(value) for value in offsets),
                default=0,
            ),
            "duration_constraints_jointly_enforced": True,
            "source_duration_targets_seconds": list(source_duration_targets),
            "source_duration_minimums_seconds": list(source_duration_minimums),
            "source_duration_maximums_seconds": list(source_duration_maximums),
            "solver": solver,
            "visual_gates_relaxed": False,
        },
    )


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    ordered = sorted(float(value) for value in values if math.isfinite(value))
    if not ordered:
        return {"p50": None, "p95": None, "max": None}
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "p50": median(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


@dataclass(frozen=True, slots=True)
class DirectMemberVisualSelection:
    source_indices: tuple[int, ...]
    source_quotas: tuple[int, ...]
    source_counts: tuple[int, ...]
    source_durations_seconds: tuple[float, ...]
    source_target_durations_seconds: tuple[float, ...]
    source_duration_relative_deviations: tuple[float, ...]
    span_validity: tuple[DirectSpanVisualDiagnostic, ...]
    boundaries: tuple[DirectBoundaryVisualDiagnostic, ...]
    selected_boundary_master_seconds: tuple[float, ...]
    optimizer_cost: float
    sample_count: int
    failed_sample_count: int
    native_scene_cut_evidence_available: bool
    continuous_official_portrait_fancam_contract: bool
    config: DirectMemberVisualConfig
    optimizer_search_diagnostics: dict[str, Any] = field(default_factory=dict)
    source_frame_geometries: tuple[
        DirectMemberSourceFrameGeometry | None,
        ...,
    ] = ()
    usage_policy: str = "exact_balanced_quota_v1"
    source_count_minimums: tuple[int, ...] = ()
    source_count_maximums: tuple[int, ...] = ()
    source_duration_minimums_seconds: tuple[float, ...] = ()
    source_duration_maximums_seconds: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_indices:
            raise ValueError("direct member visual selection cannot be empty")
        if len(self.source_quotas) != len(self.source_counts):
            raise ValueError("source quota/count lengths must match")
        if not (
            len(self.source_durations_seconds)
            == len(self.source_target_durations_seconds)
            == len(self.source_duration_relative_deviations)
            == len(self.source_counts)
        ):
            raise ValueError("source duration-balance lengths must match")
        if self.usage_policy not in {
            "exact_balanced_quota_v1",
            "heterogeneous_role_aware_cinematic_v1",
        }:
            raise ValueError("direct member visual usage policy is invalid")
        heterogeneous_usage = bool(
            self.usage_policy == "heterogeneous_role_aware_cinematic_v1"
        )
        if not heterogeneous_usage and self.source_counts != self.source_quotas:
            raise ValueError("direct member visual selection must satisfy every quota")
        if heterogeneous_usage:
            count = len(self.source_counts)
            if not (
                len(self.source_count_minimums)
                == len(self.source_count_maximums)
                == len(self.source_duration_minimums_seconds)
                == len(self.source_duration_maximums_seconds)
                == count
            ):
                raise ValueError("heterogeneous usage bounds must match sources")
            if any(
                minimum <= 0
                or maximum < minimum
                or not minimum <= actual <= maximum
                for actual, minimum, maximum in zip(
                    self.source_counts,
                    self.source_count_minimums,
                    self.source_count_maximums,
                )
            ):
                raise ValueError("heterogeneous source counts missed hard bounds")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in self.source_durations_seconds
        ):
            raise ValueError("source durations must be finite and non-negative")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in self.source_target_durations_seconds
        ):
            raise ValueError("source target durations must be finite and positive")
        for actual, target, reported_deviation in zip(
            self.source_durations_seconds,
            self.source_target_durations_seconds,
            self.source_duration_relative_deviations,
        ):
            expected_deviation = abs(actual - target) / target
            if (
                not math.isfinite(reported_deviation)
                or reported_deviation < 0.0
                or not math.isclose(
                    reported_deviation,
                    expected_deviation,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError("source relative duration deviation is inconsistent")
        if heterogeneous_usage:
            if any(
                not math.isfinite(minimum)
                or not math.isfinite(maximum)
                or minimum <= 0.0
                or maximum < minimum
                or actual < minimum - 1e-9
                or actual > maximum + 1e-9
                for actual, minimum, maximum in zip(
                    self.source_durations_seconds,
                    self.source_duration_minimums_seconds,
                    self.source_duration_maximums_seconds,
                )
            ):
                raise ValueError("heterogeneous source durations missed hard bounds")
        elif (
            max(self.source_duration_relative_deviations)
            > self.config.maximum_source_duration_relative_deviation + 1e-9
        ):
            raise ValueError("direct member visual duration balance threshold failed")
        if len(self.boundaries) != len(self.source_indices) - 1:
            raise ValueError("direct member visual boundaries must match the path")
        if len(self.selected_boundary_master_seconds) != len(self.boundaries):
            raise ValueError(
                "direct member selected boundary times must match the path"
            )
        if any(
            not math.isfinite(value)
            for value in self.selected_boundary_master_seconds
        ) or any(
            left >= right
            for left, right in zip(
                self.selected_boundary_master_seconds,
                self.selected_boundary_master_seconds[1:],
            )
        ):
            raise ValueError(
                "direct member selected boundary times must be finite and increasing"
            )
        hold_flags = tuple(
            left == right
            for left, right in zip(
                self.source_indices,
                self.source_indices[1:],
            )
        )
        if any(
            boundary.cut_applied == is_hold
            for boundary, is_hold in zip(self.boundaries, hold_flags)
        ):
            raise ValueError("direct member cut/hold diagnostics differ from path")
        maximum_holds = min(
            self.config.maximum_same_source_hold_count,
            int(
                math.ceil(
                    len(hold_flags)
                    * self.config.maximum_same_source_hold_fraction
                )
            ),
        )
        if sum(hold_flags) > maximum_holds:
            raise ValueError("direct member same-source hold budget exceeded")
        consecutive_holds = 0
        for is_hold in hold_flags:
            consecutive_holds = consecutive_holds + 1 if is_hold else 0
            if (
                consecutive_holds
                > self.config.maximum_consecutive_same_source_holds
            ):
                raise ValueError("direct member consecutive hold budget exceeded")
        if len(self.span_validity) != len(self.source_indices):
            raise ValueError("direct member span validity must match the path")
        if any(
            not self.span_validity[index].sources[source_index].valid
            for index, source_index in enumerate(self.source_indices)
        ):
            raise ValueError("direct member visual path selected invalid content")
        uncontracted_severe = tuple(
            boundary
            for boundary in self.boundaries
            if boundary.mismatch.severe_flags
            and boundary.mismatch.cinematic_bridge is None
        )
        if uncontracted_severe:
            raise ValueError(
                "direct member visual path selected an uncontracted severe boundary"
            )
        cinematic_bridges = tuple(
            boundary
            for boundary in self.boundaries
            if boundary.mismatch.cinematic_bridge is not None
        )
        if cinematic_bridges and not heterogeneous_usage:
            raise ValueError(
                "cinematic bridge requires heterogeneous usage authority"
            )
        if len(cinematic_bridges) > _DIRECT_MEMBER_CINEMATIC_BRIDGE_MAXIMUM_COUNT:
            raise ValueError("direct member cinematic bridge budget exceeded")
        if self.source_frame_geometries and (
            len(self.source_frame_geometries) != len(self.source_counts)
        ):
            raise ValueError(
                "direct member source frame geometry must match source count"
            )

    def aggregate_diagnostics(self) -> dict[str, Any]:
        component_names = tuple(
            DirectBoundaryMismatch(
                0.0, 0.0, None, None, None, None, None, None, None, None, 0, (), ()
            ).components()
        )
        warning_counts: Counter[str] = Counter()
        severe_counts: Counter[str] = Counter()
        for boundary in self.boundaries:
            warning_counts.update(boundary.mismatch.warning_flags)
            severe_counts.update(boundary.mismatch.severe_flags)
        motion_evidence = tuple(
            evidence
            for boundary in self.boundaries
            if (evidence := boundary.mismatch.motion_connection) is not None
        )
        transform_repairs = tuple(
            repair
            for boundary in self.boundaries
            if (repair := boundary.mismatch.transform_repair) is not None
        )
        cinematic_bridges = tuple(
            bridge
            for boundary in self.boundaries
            if (bridge := boundary.mismatch.cinematic_bridge) is not None
        )
        softened_context: Counter[str] = Counter(
            flag
            for boundary in self.boundaries
            for flag in boundary.mismatch.member_context_softened_flags
        )
        validity_reasons: Counter[str] = Counter(
            reason
            for span in self.span_validity
            for source in span.sources
            for reason in source.rejection_reasons
        )
        return {
            "selected_boundary_count": len(self.boundaries),
            "selected_cut_count": sum(
                boundary.cut_applied for boundary in self.boundaries
            ),
            "same_source_hold_count": sum(
                not boundary.cut_applied for boundary in self.boundaries
            ),
            "cost": _distribution(
                [boundary.mismatch.cost for boundary in self.boundaries]
            ),
            "base_mismatch": _distribution(
                [
                    boundary.mismatch.base_mismatch
                    for boundary in self.boundaries
                ]
            ),
            "components": {
                name: _distribution(
                    [
                        value
                        for boundary in self.boundaries
                        if (value := boundary.mismatch.components()[name])
                        is not None
                    ]
                )
                for name in component_names
            },
            "selected_pair_rank": _distribution(
                [float(boundary.selected_pair_rank) for boundary in self.boundaries]
            ),
            "eligible_candidate_pair_count": _distribution(
                [float(boundary.candidate_pair_count) for boundary in self.boundaries]
            ),
            "invalid_content_pair_count": {
                "total": sum(
                    boundary.invalid_content_pair_count
                    for boundary in self.boundaries
                ),
                "max": max(
                    (
                        boundary.invalid_content_pair_count
                        for boundary in self.boundaries
                    ),
                    default=0,
                ),
            },
            "severe_ineligible_pair_count": {
                "total": sum(
                    boundary.severe_pair_count for boundary in self.boundaries
                ),
                "max": max(
                    (boundary.severe_pair_count for boundary in self.boundaries),
                    default=0,
                ),
            },
            "warning_boundary_count": sum(
                bool(boundary.mismatch.warning_flags)
                for boundary in self.boundaries
            ),
            "severe_boundary_count": sum(
                bool(boundary.mismatch.severe_flags)
                for boundary in self.boundaries
            ),
            "uncontracted_severe_boundary_count": sum(
                bool(boundary.mismatch.severe_flags)
                and boundary.mismatch.cinematic_bridge is None
                for boundary in self.boundaries
            ),
            "warning_flag_counts": dict(sorted(warning_counts.items())),
            "severe_flag_counts": dict(sorted(severe_counts.items())),
            "cinematic_effect_bridges": {
                "required_boundary_count": len(cinematic_bridges),
                "styles": dict(
                    sorted(Counter(item.style for item in cinematic_bridges).items())
                ),
                "maximum_authorized_count": (
                    _DIRECT_MEMBER_CINEMATIC_BRIDGE_MAXIMUM_COUNT
                ),
                "hard_cut_authorized": False,
                "required_transition_effect": True,
            },
            "adjacent_motion_connection": {
                "evaluated_boundary_count": len(motion_evidence),
                "timestamp_bound_adjacent_boundary_count": sum(
                    evidence.timestamp_bound_adjacent
                    for evidence in motion_evidence
                ),
                "both_moving_boundary_count": sum(
                    evidence.both_moving for evidence in motion_evidence
                ),
                "soft_preference_applied_boundary_count": sum(
                    evidence.soft_preference_applied > 0.0
                    for evidence in motion_evidence
                ),
                "soft_preference": _distribution(
                    [
                        evidence.soft_preference_applied
                        for evidence in motion_evidence
                    ]
                ),
                "priority_motion_preference": _distribution(
                    [
                        boundary.mismatch.priority_motion_preference_applied
                        for boundary in self.boundaries
                        if boundary.cut_applied
                    ]
                ),
                "priority_motion_preference_applied_boundary_count": sum(
                    boundary.mismatch.priority_motion_preference_applied > 0.0
                    for boundary in self.boundaries
                ),
                "direction_cosine": _distribution(
                    [
                        value
                        for evidence in motion_evidence
                        if (
                            value := evidence.direction_cosine_audit_only
                        )
                        is not None
                    ]
                ),
                "direction_cosine_audit_only": _distribution(
                    [
                        value
                        for evidence in motion_evidence
                        if (
                            value := evidence.direction_cosine_audit_only
                        )
                        is not None
                    ]
                ),
                "motion_speed_ratio_mismatch": _distribution(
                    [
                        value
                        for evidence in motion_evidence
                        if (
                            value := evidence.motion_speed_ratio_mismatch
                        )
                        is not None
                    ]
                ),
                "pose_phase_continuation_mismatch": _distribution(
                    [
                        value
                        for evidence in motion_evidence
                        if (
                            value := evidence.pose_phase_continuation_mismatch
                        )
                        is not None
                    ]
                ),
                "output_space_subject_motion_boundary_count": sum(
                    evidence.output_space_subject_motion_used
                    for evidence in motion_evidence
                ),
                "direction_influenced_selection": True,
                "speed_and_pose_phase_influenced_selection": True,
                "raw_vectors_persisted": False,
            },
            "lossless_transform_repair": {
                "required_boundary_count": len(transform_repairs),
                "required_zooms": _distribution(
                    [item.zoom for item in transform_repairs]
                ),
                "required_center_shifts": _distribution(
                    [item.center_shift for item in transform_repairs]
                ),
                "lossless_zoom_capacities": _distribution(
                    [
                        item.source_geometry.lossless_zoom_capacity
                        for item in transform_repairs
                    ]
                ),
                "repaired_severe_flag_counts": dict(
                    sorted(
                        Counter(
                            flag
                            for item in transform_repairs
                            for flag in item.repaired_severe_flags
                        ).items()
                    )
                ),
                "member_context_softened_flag_counts": dict(
                    sorted(softened_context.items())
                ),
                "dense_every_output_frame_revalidation_required": True,
                "upscaled_working_frame_authorized": False,
            },
            "sample_count": self.sample_count,
            "failed_sample_count": self.failed_sample_count,
            "descriptor_success_fraction": (
                (self.sample_count - self.failed_sample_count) / self.sample_count
                if self.sample_count
                else 0.0
            ),
            "content_validity": {
                "evaluated_candidate_count": sum(
                    len(span.sources) for span in self.span_validity
                ),
                "invalid_candidate_count": sum(
                    not source.valid
                    for span in self.span_validity
                    for source in span.sources
                ),
                "native_scene_cut_invalid_candidate_count": sum(
                    bool(source.native_scene_cut_master_seconds)
                    for span in self.span_validity
                    for source in span.sources
                ),
                "native_scene_cut_count": sum(
                    len(source.native_scene_cut_master_seconds)
                    for span in self.span_validity
                    for source in span.sources
                ),
                "suppressed_portrait_fancam_scene_cut_count": sum(
                    len(
                        source.suppressed_portrait_fancam_scene_cut_master_seconds
                    )
                    for span in self.span_validity
                    for source in span.sources
                ),
                "portrait_fancam_lighting_quality_rescue_count": sum(
                    source.portrait_fancam_lighting_quality_rescue_applied
                    for span in self.span_validity
                    for source in span.sources
                ),
                "rejection_reason_counts": dict(sorted(validity_reasons.items())),
                "minimum_valid_source_count_per_span": min(
                    sum(source.valid for source in span.sources)
                    for span in self.span_validity
                ),
                "valid_span_count_by_source": [
                    sum(span.sources[index].valid for span in self.span_validity)
                    for index in range(len(self.source_counts))
                ],
            },
            "duration_balance": {
                "passed": (
                    all(
                        minimum - 1e-9 <= actual <= maximum + 1e-9
                        for actual, minimum, maximum in zip(
                            self.source_durations_seconds,
                            self.source_duration_minimums_seconds,
                            self.source_duration_maximums_seconds,
                        )
                    )
                    if self.usage_policy
                    == "heterogeneous_role_aware_cinematic_v1"
                    else max(self.source_duration_relative_deviations)
                    <= self.config.maximum_source_duration_relative_deviation
                    + 1e-9
                ),
                "target_policy": (
                    "heterogeneous_role_soft_target_with_hard_exposure_bounds"
                    if self.usage_policy
                    == "heterogeneous_role_aware_cinematic_v1"
                    else "total_duration_times_source_quota_divided_by_span_count"
                ),
                "total_duration_seconds": sum(self.source_durations_seconds),
                "maximum_relative_deviation": max(
                    self.source_duration_relative_deviations
                ),
                "maximum_allowed_relative_deviation": (
                    None
                    if self.usage_policy
                    == "heterogeneous_role_aware_cinematic_v1"
                    else self.config.maximum_source_duration_relative_deviation
                ),
                "normalized_squared_load_cost": (
                    self.config.duration_balance_weight
                    * sum(
                        (actual / target) ** 2
                        for actual, target in zip(
                            self.source_durations_seconds,
                            self.source_target_durations_seconds,
                        )
                    )
                ),
            },
            "raw_embeddings_persisted": False,
            "face_identity_inference_used": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_basis": (
                "timewarp_boundary_native_cut_non_identity_geometry_"
                "cut_adjacent_motion_near_beat_semimarkov_adaptive_beam_"
                "exact_entry_output_space_speed_phase_feasibility_v7"
            ),
            "selection_objective": (
                "minimum_real_boundary_geometry_mismatch_subject_to_"
                "all_source_count_duration_constraints_plus_bounded_safe_"
                "same_source_no_cut_continuation_actual_pose_silhouette_"
                "same_direction_speed_phase_motion_exact_visible_entry_and_"
                "joint_near_beat_cut_state_search"
            ),
            "selected_source_indices": list(self.source_indices),
            "selected_boundary_master_seconds": list(
                self.selected_boundary_master_seconds
            ),
            "distinct_source_count": len(set(self.source_indices)),
            "source_usage_policy": self.usage_policy,
            "source_usage": [
                {
                    "eligible_source_index": source_index,
                    "quota": self.source_quotas[source_index],
                    "segment_count": self.source_counts[source_index],
                    "duration_seconds": self.source_durations_seconds[source_index],
                    "target_duration_seconds": (
                        self.source_target_durations_seconds[source_index]
                    ),
                    "duration_deviation_seconds": (
                        self.source_durations_seconds[source_index]
                        - self.source_target_durations_seconds[source_index]
                    ),
                    "duration_relative_deviation": (
                        self.source_duration_relative_deviations[source_index]
                    ),
                    "minimum_segment_count": (
                        self.source_count_minimums[source_index]
                        if self.source_count_minimums
                        else self.source_quotas[source_index]
                    ),
                    "maximum_segment_count": (
                        self.source_count_maximums[source_index]
                        if self.source_count_maximums
                        else self.source_quotas[source_index]
                    ),
                    "minimum_duration_seconds": (
                        self.source_duration_minimums_seconds[source_index]
                        if self.source_duration_minimums_seconds
                        else self.source_target_durations_seconds[source_index]
                        * (
                            1.0
                            - self.config.maximum_source_duration_relative_deviation
                        )
                    ),
                    "maximum_duration_seconds": (
                        self.source_duration_maximums_seconds[source_index]
                        if self.source_duration_maximums_seconds
                        else self.source_target_durations_seconds[source_index]
                        * (
                            1.0
                            + self.config.maximum_source_duration_relative_deviation
                        )
                    ),
                }
                for source_index in range(len(self.source_counts))
            ],
            "counts_differ_by_at_most_one": (
                max(self.source_counts) - min(self.source_counts) <= 1
            ),
            "same_source_adjacent_count": sum(
                left == right
                for left, right in zip(
                    self.source_indices,
                    self.source_indices[1:],
                )
            ),
            "same_source_adjacent_policy": (
                "bounded_no_cut_continuation_only_after_strict_cross_source_"
                "path_failure"
            ),
            "same_source_nonadjacent_reuse_count": sum(
                max(0, count - 1) for count in self.source_counts
            ),
            "rapid_ping_pong_count": sum(
                self.source_indices[index] == self.source_indices[index - 2]
                for index in range(2, len(self.source_indices))
            ),
            "rapid_ping_pong_penalty": self.config.ping_pong_penalty,
            "optimizer_cost": self.optimizer_cost,
            "optimizer_search": dict(self.optimizer_search_diagnostics),
            "native_scene_cut_evidence": {
                "available": self.native_scene_cut_evidence_available,
                "policy_applied": self.native_scene_cut_evidence_available,
                "continuous_official_portrait_fancam_contract": (
                    self.continuous_official_portrait_fancam_contract
                ),
            },
            "config": self.config.to_dict(),
            "source_frame_geometries": [
                geometry.to_dict() if geometry is not None else None
                for geometry in self.source_frame_geometries
            ],
            "aggregate": self.aggregate_diagnostics(),
            "span_content_validity": [
                span.to_dict() for span in self.span_validity
            ],
            "boundaries": [boundary.to_dict() for boundary in self.boundaries],
            "privacy": {
                "face_identity_inference_used": False,
                "face_appearance_embeddings_computed": False,
                "raw_pose_silhouette_layout_embeddings_persisted": False,
                "persisted_evidence": (
                    "normalized_attention_geometry_backend_confidence_"
                    "foreground_obstruction_highlight_clipping_gray_summary_"
                    "cut_adjacent_motion_scalars_resolution_capacity_and_"
                    "aggregate_mismatch_only"
                ),
            },
        }


class _VideoDescriptorProvider:
    """Sequential decoder with interruption-safe numeric feature reuse."""

    def __init__(
        self,
        sources: Sequence[Path],
        *,
        feature_cache_directory: Path | None = None,
    ) -> None:
        self._samplers: list[_VideoSampler | None] = []
        self._descriptor_cache: dict[
            tuple[int, CanonicalFrameIdentity], FrameVisualDescriptor | None
        ] = {}
        self._request_count = 0
        self._cache_hit_count = 0
        self._decode_count = 0
        for source in sources:
            try:
                sampler = _VideoSampler(
                    source,
                    feature_cache_directory=feature_cache_directory,
                )
            except (OSError, TypeError, ValueError, cv2.error):
                sampler = None
            self._samplers.append(sampler)

    def __call__(
        self,
        source_index: int,
        _source: Path,
        source_local_seconds: float,
    ) -> FrameVisualDescriptor | None:
        self._request_count += 1
        sampler = self._samplers[source_index]
        if sampler is None:
            return None
        key = (source_index, sampler.frame_identity(source_local_seconds))
        if key in self._descriptor_cache:
            self._cache_hit_count += 1
            return self._descriptor_cache[key]
        self._decode_count += 1
        try:
            descriptor = sampler.feature_descriptor_at(
                source_local_seconds,
                include_subject_geometry=True,
                include_formation_pose=True,
            )
        except (cv2.error, ValueError, FloatingPointError):
            descriptor = None
        self._descriptor_cache[key] = descriptor
        return descriptor

    def cache_diagnostics(self) -> dict[str, Any]:
        """Return RAM and persistent counters without source paths or media."""

        persistent = tuple(
            diagnostics
            for sampler in self._samplers
            if sampler is not None
            and (diagnostics := sampler.feature_cache_diagnostics()) is not None
        )

        return {
            "descriptor_request_count": self._request_count,
            "descriptor_cache_hit_count": self._cache_hit_count,
            "descriptor_decode_count": self._decode_count,
            "descriptor_cache_entry_count": len(self._descriptor_cache),
            "persistent_visual_feature_cache": {
                "contract": "ive_source_local_visual_features_v1",
                "enabled_source_count": len(persistent),
                "hit_count": sum(int(item["hit_count"]) for item in persistent),
                "miss_count": sum(int(item["miss_count"]) for item in persistent),
                "write_count": sum(int(item["write_count"]) for item in persistent),
                "invalid_record_count": sum(
                    int(item["invalid_record_count"]) for item in persistent
                ),
                "decode_free_hit_count": sum(
                    int(item.get("decode_free_hit_count", 0))
                    for item in persistent
                ),
                "decode_free_legacy_hit_count": sum(
                    int(item.get("decode_free_legacy_hit_count", 0))
                    for item in persistent
                ),
                "source_frame_decode_count": sum(
                    int(item.get("source_frame_decode_count", 0))
                    for item in persistent
                ),
                "locator_index_build_count": sum(
                    int(item.get("locator_index_build_count", 0))
                    for item in persistent
                ),
                "ambiguous_locator_count": sum(
                    int(item.get("ambiguous_locator_count", 0))
                    for item in persistent
                ),
                "interrupted_run_records_reusable": bool(persistent),
                "raw_frames_persisted": False,
                "face_identity_embeddings_persisted": False,
            },
        }

    def close(self) -> None:
        for sampler in self._samplers:
            if sampler is not None:
                sampler.close()


def _attention_geometry(
    descriptor: FrameVisualDescriptor | None,
) -> _AttentionGeometry | None:
    if descriptor is None:
        return None
    if descriptor.primary_face_bbox is not None:
        bbox = descriptor.primary_face_bbox
        return _AttentionGeometry(
            bbox_xywh=bbox,
            centroid_xy=(bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0),
            basis="unambiguous_dominant_face_bbox_geometry_only",
        )
    if descriptor.silhouette_bbox_xywh is not None:
        bbox = descriptor.silhouette_bbox_xywh
        centroid = descriptor.silhouette_centroid_xy or (
            bbox[0] + bbox[2] / 2.0,
            bbox[1] + bbox[3] / 2.0,
        )
        return _AttentionGeometry(
            bbox_xywh=bbox,
            centroid_xy=centroid,
            basis="aggregate_human_silhouette",
        )
    people = tuple(descriptor.formation_people)
    if not people:
        return None
    ranked = sorted(
        people,
        key=lambda person: person.bbox_xywh[2] * person.bbox_xywh[3],
        reverse=True,
    )
    largest = ranked[0]
    largest_area = largest.bbox_xywh[2] * largest.bbox_xywh[3]
    if len(ranked) > 1:
        second_area = ranked[1].bbox_xywh[2] * ranked[1].bbox_xywh[3]
        if second_area >= largest_area * 0.78:
            return None
    return _AttentionGeometry(
        bbox_xywh=largest.bbox_xywh,
        centroid_xy=largest.centroid_xy,
        basis="unambiguous_dominant_person_box_geometry_only",
    )


def _embedding_distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    squared = sum(
        (float(left_value) - float(right_value)) ** 2
        for left_value, right_value in zip(left, right, strict=True)
    )
    return min(1.0, math.sqrt(squared / len(left)))


def _sample_has_human_evidence(
    sample: _BoundarySample,
    config: DirectMemberVisualConfig,
) -> bool:
    descriptor = sample.descriptor
    if descriptor is None:
        return False
    return bool(
        sample.attention is not None
        or descriptor.face_count > 0
        or descriptor.detected_person_count > 0
        or (
            descriptor.silhouette_bbox_xywh is not None
            and descriptor.silhouette_confidence
            >= config.minimum_silhouette_confidence
        )
    )


def _sample_has_verified_yunet_face(sample: _BoundarySample) -> bool:
    descriptor = sample.descriptor
    return bool(
        descriptor is not None
        and descriptor.face_count > 0
        and descriptor.face_detector_backend.casefold() == "yunet"
    )


def _sample_is_blank_or_flat(
    sample: _BoundarySample,
    config: DirectMemberVisualConfig,
) -> bool:
    descriptor = sample.descriptor
    if descriptor is None or _sample_has_human_evidence(sample, config):
        return False
    color = descriptor.color_embedding
    if len(color) < 8:
        return False
    gray_mean = float(color[6])
    gray_std = float(color[7])
    chroma_spread = sum(float(value) for value in color[3:6])
    flat = (
        gray_std <= config.blank_gray_std_maximum
        and chroma_spread <= config.blank_gray_std_maximum * 3.0
    )
    extreme = (
        gray_mean <= config.blank_extreme_mean_minimum
        or gray_mean >= config.blank_extreme_mean_maximum
    ) and gray_std <= config.blank_gray_std_maximum * 2.0
    return flat or extreme


def _continuous_portrait_fancam_scene_cut_is_transient_artifact(
    descriptors: tuple[
        FrameVisualDescriptor | None,
        FrameVisualDescriptor | None,
        FrameVisualDescriptor | None,
        FrameVisualDescriptor | None,
        FrameVisualDescriptor | None,
    ],
    config: DirectMemberVisualConfig,
) -> bool:
    """Return true only for a corroborated continuous-performance pseudo-cut.

    The five descriptors are far-before, one-frame-before, detector frame,
    one-frame-after, and far-after.  This exception is intentionally narrower
    than scene detection: every flank must retain non-identifying human
    geometry, the persistent far-side structure must remain close, no sampled
    camera-equipment obstruction may be present, and the post side must not be
    a persistent bright broadcast graphic.  Missing or ambiguous evidence
    therefore keeps the raw source cut as a hard rejection.
    """

    before_far, before_near, _detector_frame, after_near, after_far = descriptors
    flank_descriptors = (before_far, before_near, after_near, after_far)
    if any(descriptor is None for descriptor in flank_descriptors):
        return False
    flank_samples = tuple(
        _BoundarySample(
            master_seconds=0.0,
            source_local_seconds=0.0,
            descriptor=descriptor,
            attention=_attention_geometry(descriptor),
        )
        for descriptor in flank_descriptors
    )
    if not all(
        _sample_has_human_evidence(sample, config) for sample in flank_samples
    ):
        return False
    if any(
        descriptor is not None
        and descriptor.foreground_obstruction_score
        >= config.foreground_obstruction_rejection_score
        for descriptor in descriptors
    ):
        return False
    assert before_far is not None and after_far is not None
    structure_mismatch = _embedding_distance(
        before_far.structure_embedding,
        after_far.structure_embedding,
    )
    if (
        structure_mismatch is None
        or structure_mismatch
        > config.portrait_fancam_scene_cut_continuity_structure_maximum
    ):
        return False

    def _persistent_bright_graphic(
        descriptor: FrameVisualDescriptor | None,
    ) -> bool:
        if descriptor is None or len(descriptor.color_embedding) < 8:
            return False
        return bool(
            descriptor.quality_score
            <= config.portrait_fancam_scene_cut_post_graphic_quality_maximum
            and float(descriptor.color_embedding[6])
            >= config.portrait_fancam_scene_cut_post_graphic_gray_mean_minimum
            and float(descriptor.color_embedding[7])
            <= config.portrait_fancam_scene_cut_post_graphic_contrast_maximum
        )

    if _persistent_bright_graphic(after_near) and _persistent_bright_graphic(
        after_far
    ):
        return False
    return True


def _span_source_validity(
    *,
    source_index: int,
    samples: tuple[_BoundarySample, ...],
    native_scene_cuts: tuple[tuple[float, float], ...],
    suppressed_portrait_fancam_scene_cuts: tuple[tuple[float, float], ...],
    continuous_official_portrait_fancam_contract: bool,
    config: DirectMemberVisualConfig,
) -> DirectSpanSourceValidity:
    descriptors = tuple(
        sample.descriptor for sample in samples if sample.descriptor is not None
    )
    human = tuple(_sample_has_human_evidence(sample, config) for sample in samples)
    blank = tuple(_sample_is_blank_or_flat(sample, config) for sample in samples)
    blank_count = sum(blank)
    structure_distances = tuple(
        value
        for left, right in zip(descriptors, descriptors[1:])
        if (
            value := _embedding_distance(
                left.structure_embedding,
                right.structure_embedding,
            )
        )
        is not None
    )
    color_distances = tuple(
        value
        for left, right in zip(descriptors, descriptors[1:])
        if (
            value := _embedding_distance(
                left.color_embedding,
                right.color_embedding,
            )
        )
        is not None
    )
    maximum_structure = max(structure_distances) if structure_distances else None
    maximum_color = max(color_distances) if color_distances else None
    static = bool(
        maximum_structure is not None
        and maximum_color is not None
        and maximum_structure <= config.static_structure_mismatch_maximum
        and maximum_color <= config.static_color_mismatch_maximum
    )
    reasons: list[str] = []
    if (
        len(samples) != _DIRECT_CONTENT_SAMPLE_COUNT
        or len(descriptors) != _DIRECT_CONTENT_SAMPLE_COUNT
    ):
        reasons.append("span_frame_decode_failed")
    human_count = sum(human)
    # Automatic member selection is advertised and diagnosed as fail-closed on
    # the bundled, integrity-checked YuNet model.  Haar is intentionally still
    # useful as a weak local composition hint, but it must never satisfy this
    # hard performance-evidence gate if YuNet inference fails at frame time.
    face_evidence_count = sum(
        _sample_has_verified_yunet_face(sample) for sample in samples
    )
    quality = tuple(descriptor.quality_score for descriptor in descriptors)
    acceptable_quality_count = sum(
        score >= config.minimum_acceptable_quality_score for score in quality
    )
    obstruction_scores = tuple(
        descriptor.foreground_obstruction_score for descriptor in descriptors
    )
    obstruction_count = sum(
        score >= config.foreground_obstruction_rejection_score
        for score in obstruction_scores
    )
    gray_contrast_scores = tuple(
        (
            float(descriptor.color_embedding[7])
            if len(descriptor.color_embedding) >= 8
            else None
        )
        for descriptor in descriptors
    )
    gray_mean_scores = tuple(
        (
            float(descriptor.color_embedding[6])
            if len(descriptor.color_embedding) >= 8
            else None
        )
        for descriptor in descriptors
    )
    highlight_clipping_scores = tuple(
        descriptor.highlight_clipping_ratio for descriptor in descriptors
    )
    terminal_composite = False
    if (
        len(quality) == _DIRECT_CONTENT_SAMPLE_COUNT
        and len(gray_contrast_scores) == _DIRECT_CONTENT_SAMPLE_COUNT
        and all(value is not None for value in gray_contrast_scores[-3:])
    ):
        before_quality, pre_exit_quality, exit_quality = quality[-3:]
        before_contrast, pre_exit_contrast, exit_contrast = (
            float(value) for value in gray_contrast_scores[-3:]
        )
        terminal_composite = bool(
            exit_quality <= config.terminal_composite_exit_quality_maximum
            and pre_exit_quality
            <= config.terminal_composite_pre_exit_quality_maximum
            and before_quality - pre_exit_quality
            >= config.terminal_composite_first_quality_drop_minimum
            and pre_exit_quality - exit_quality
            >= config.terminal_composite_exit_quality_drop_minimum
            and before_contrast - pre_exit_contrast
            >= config.terminal_composite_first_contrast_drop_minimum
            and pre_exit_contrast - exit_contrast
            >= config.terminal_composite_exit_contrast_drop_minimum
        )
    persistent_entry_overexposure = False
    if (
        len(quality) == _DIRECT_CONTENT_SAMPLE_COUNT
        and len(gray_mean_scores) == _DIRECT_CONTENT_SAMPLE_COUNT
        and len(gray_contrast_scores) == _DIRECT_CONTENT_SAMPLE_COUNT
        and len(highlight_clipping_scores) == _DIRECT_CONTENT_SAMPLE_COUNT
        and all(value is not None for value in gray_mean_scores[:2])
        and all(value is not None for value in gray_contrast_scores[:2])
    ):
        persistent_entry_overexposure = all(
            quality[index]
            <= config.persistent_entry_overexposure_quality_maximum
            and float(gray_mean_scores[index])
            >= config.persistent_entry_overexposure_gray_mean_minimum
            and float(gray_contrast_scores[index])
            <= config.persistent_entry_overexposure_gray_contrast_maximum
            and highlight_clipping_scores[index]
            >= config.persistent_entry_overexposure_highlight_minimum
            for index in range(2)
        )
    mean_quality = sum(quality) / len(quality) if quality else None
    # ``quality_score`` is a generic sharpness/exposure/contrast aesthetic
    # score.  It is not, by itself, proof that an intentionally dark or bright
    # music-show frame is unusable.  Only the exact verified continuous
    # portrait-fancam acquisition contract may rescue that one reason, and
    # only when all six samples independently preserve a detailed human
    # performance with a YuNet face and none of the stronger invalid-content
    # gates fire.  Generic/manual sources retain the original 0.72/5-of-6 gate.
    portrait_fancam_lighting_quality_rescue = bool(
        continuous_official_portrait_fancam_contract
        and acceptable_quality_count
        < config.minimum_acceptable_quality_sample_count
        and len(samples) == _DIRECT_CONTENT_SAMPLE_COUNT
        and len(descriptors) == _DIRECT_CONTENT_SAMPLE_COUNT
        and human_count == _DIRECT_CONTENT_SAMPLE_COUNT
        and face_evidence_count >= max(1, config.minimum_face_sample_count)
        and mean_quality is not None
        and mean_quality
        >= config.portrait_fancam_quality_rescue_mean_minimum
        and len(gray_contrast_scores) == _DIRECT_CONTENT_SAMPLE_COUNT
        and all(
            value is not None
            and float(value)
            >= config.portrait_fancam_quality_rescue_contrast_minimum
            for value in gray_contrast_scores
        )
        and obstruction_count == 0
        and blank_count == 0
        and not terminal_composite
        and not persistent_entry_overexposure
        and not native_scene_cuts
    )
    if face_evidence_count < config.minimum_face_sample_count:
        reasons.append("no_face_performance_evidence")
    if (
        acceptable_quality_count
        < config.minimum_acceptable_quality_sample_count
        and not portrait_fancam_lighting_quality_rescue
    ):
        reasons.append("persistent_low_visual_quality_or_foreground_obstruction")
    if obstruction_count:
        reasons.append("foreground_camera_equipment_or_subject_obstruction")
    if terminal_composite:
        reasons.append("terminal_broadcast_composite_or_unreported_native_transition")
    if persistent_entry_overexposure:
        reasons.append("persistent_entry_stage_light_overexposure")
    if native_scene_cuts:
        reasons.append("source_native_scene_cut_inside_rendered_span")
    if human_count < config.minimum_human_sample_count:
        reasons.append("no_person_performance_evidence")
    if config.require_entry_exit_human_evidence:
        if not human[0]:
            reasons.append("no_person_at_span_entry")
        if not human[-1]:
            reasons.append("no_person_at_span_exit")
    if blank_count:
        reasons.append("blank_or_flat_frame")
    # Pose/person/silhouette models can fire on large logo shapes.  A static
    # six-sample candidate therefore needs independent face evidence before
    # those weaker human signals can keep it eligible.  A held close-up with a
    # detected face remains valid; a static wide performance without any face
    # can fail closed, which is preferable to rendering a branded end card.
    if static and face_evidence_count == 0:
        reasons.append("static_nonface_graphic_or_end_card")
        reasons.append("static_logo_or_end_card_candidate")
        if human_count == 0:
            reasons.append("static_logo_or_end_card_without_person")
    return DirectSpanSourceValidity(
        eligible_source_index=source_index,
        valid=not reasons,
        rejection_reasons=tuple(dict.fromkeys(reasons)),
        samples=samples,
        human_evidence_sample_count=human_count,
        face_evidence_sample_count=face_evidence_count,
        acceptable_quality_sample_count=acceptable_quality_count,
        blank_sample_count=blank_count,
        static_visual_candidate=static,
        native_scene_cut_source_seconds=tuple(
            source_seconds for source_seconds, _master_seconds in native_scene_cuts
        ),
        native_scene_cut_master_seconds=tuple(
            master_seconds for _source_seconds, master_seconds in native_scene_cuts
        ),
        suppressed_portrait_fancam_scene_cut_source_seconds=tuple(
            source_seconds
            for source_seconds, _master_seconds in (
                suppressed_portrait_fancam_scene_cuts
            )
        ),
        suppressed_portrait_fancam_scene_cut_master_seconds=tuple(
            master_seconds
            for _source_seconds, master_seconds in (
                suppressed_portrait_fancam_scene_cuts
            )
        ),
        maximum_structure_mismatch=maximum_structure,
        maximum_color_mismatch=maximum_color,
        mean_quality_score=mean_quality,
        quality_sample_scores=quality,
        foreground_obstruction_sample_count=obstruction_count,
        foreground_obstruction_sample_scores=obstruction_scores,
        terminal_composite_transition_candidate=terminal_composite,
        gray_contrast_sample_scores=gray_contrast_scores,
        persistent_entry_overexposure_candidate=(
            persistent_entry_overexposure
        ),
        gray_mean_sample_scores=gray_mean_scores,
        highlight_clipping_sample_scores=highlight_clipping_scores,
        portrait_fancam_lighting_quality_rescue_applied=(
            portrait_fancam_lighting_quality_rescue
        ),
    )


def _attention_mismatch(
    left: _AttentionGeometry | None,
    right: _AttentionGeometry | None,
) -> tuple[float | None, float | None]:
    if left is None or right is None:
        return (None, None)
    position = min(
        1.0,
        math.hypot(
            left.centroid_xy[0] - right.centroid_xy[0],
            left.centroid_xy[1] - right.centroid_xy[1],
        )
        / 0.42,
    )
    left_area = max(1e-6, left.bbox_xywh[2] * left.bbox_xywh[3])
    right_area = max(1e-6, right.bbox_xywh[2] * right.bbox_xywh[3])
    scale = min(1.0, abs(math.log(left_area / right_area)) / math.log(3.2))
    return (position, scale)


def _portrait_fill_attention(
    attention: _AttentionGeometry | None,
    geometry: DirectMemberSourceFrameGeometry | None,
) -> _AttentionGeometry | None:
    if attention is None or geometry is None:
        return attention
    mapped = map_box_to_portrait_fill(attention.bbox_xywh, geometry)
    return _AttentionGeometry(
        bbox_xywh=mapped,
        centroid_xy=(
            mapped[0] + mapped[2] / 2.0,
            mapped[1] + mapped[3] / 2.0,
        ),
        basis=attention.basis,
    )


def _sample_in_portrait_fill(
    sample: _BoundarySample,
    geometry: DirectMemberSourceFrameGeometry | None,
) -> _BoundarySample:
    """Map one transient sample into the pixels the 9:16 renderer exposes."""

    return replace(
        sample,
        attention=_portrait_fill_attention(sample.attention, geometry),
    )


def _sample_with_boundary_transform(
    sample: _BoundarySample,
    repair: DirectMemberBoundaryTransformRepair | None,
) -> _BoundarySample:
    """Project attention through a selector-authorized boundary camera state."""

    if repair is None or sample.attention is None:
        return sample
    mapped = project_box(
        sample.attention.bbox_xywh,
        zoom=repair.zoom,
        center_x=repair.center_x,
        center_y=repair.center_y,
    )
    return replace(
        sample,
        attention=_AttentionGeometry(
            bbox_xywh=mapped,
            centroid_xy=(
                mapped[0] + mapped[2] / 2.0,
                mapped[1] + mapped[3] / 2.0,
            ),
            basis=sample.attention.basis,
        ),
    )


def _verified_direct_member_subject_geometry(
    descriptor: FrameVisualDescriptor,
    config: DirectMemberVisualConfig,
) -> tuple[tuple[float, float, float, float] | None, str | None]:
    """Return non-identifying subject containment geometry for a widened move."""

    if (
        descriptor.silhouette_bbox_xywh is not None
        and descriptor.silhouette_confidence
        >= config.transform_repair_minimum_subject_confidence
        and descriptor.silhouette_backend == SILHOUETTE_BACKEND
    ):
        return (
            descriptor.silhouette_bbox_xywh,
            "pphumanseg_full_subject_silhouette",
        )
    face = descriptor.primary_face_bbox
    if face is None or descriptor.face_count != 1:
        return (None, None)
    face_center = (face[0] + face[2] / 2.0, face[1] + face[3] / 2.0)
    associated = tuple(
        person
        for person in descriptor.formation_people
        if person.detection_confidence >= 0.50
        and person.bbox_xywh[0] - 1e-6
        <= face_center[0]
        <= person.bbox_xywh[0] + person.bbox_xywh[2] + 1e-6
        and person.bbox_xywh[1] - 1e-6
        <= face_center[1]
        <= person.bbox_xywh[1] + person.bbox_xywh[3] + 1e-6
    )
    if len(associated) != 1:
        return (None, None)
    return (
        associated[0].bbox_xywh,
        "face_associated_formation_person_box",
    )


def _person_count_mismatch(left: int, right: int) -> float | None:
    if left <= 0 or right <= 0:
        return None
    return min(
        1.0,
        abs(math.log((left + 1.0) / (right + 1.0))) / math.log(4.0),
    )


def _actual_pose_vector(
    descriptor: FrameVisualDescriptor | None,
    config: DirectMemberVisualConfig,
) -> tuple[float, ...] | None:
    if (
        descriptor is None
        or descriptor.pose_backend == "structure_hog_fallback"
        or descriptor.pose_confidence < config.minimum_pose_confidence
        or not descriptor.pose_embedding
    ):
        return None
    return tuple(float(value) for value in descriptor.pose_embedding)


def _vector_delta(
    previous: Sequence[float] | None,
    current: Sequence[float] | None,
) -> tuple[float, ...] | None:
    if (
        previous is None
        or current is None
        or not previous
        or len(previous) != len(current)
    ):
        return None
    return tuple(
        float(right) - float(left)
        for left, right in zip(previous, current, strict=True)
    )


def _delta_magnitude(values: Sequence[float] | None) -> float | None:
    if not values:
        return None
    return min(
        1.0,
        math.sqrt(sum(float(value) ** 2 for value in values) / len(values)),
    )


def _direction_cosine(
    left: Sequence[float],
    right: Sequence[float],
) -> float | None:
    if not left or len(left) != len(right):
        return None
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return None
    return max(
        -1.0,
        min(
            1.0,
            sum(
                float(left_value) * float(right_value)
                for left_value, right_value in zip(left, right, strict=True)
            )
            / (left_norm * right_norm),
        ),
    )


def _motion_speed_ratio_mismatch(
    outgoing: Mapping[str, tuple[float, tuple[float, ...]]],
    incoming: Mapping[str, tuple[float, tuple[float, ...]]],
) -> float | None:
    """Compare like-for-like cut-adjacent motion speeds without raw vectors."""

    ratios = tuple(
        min(
            1.0,
            abs(math.log(outgoing[name][0] / incoming[name][0]))
            / math.log(4.0),
        )
        for name in ("pose", "silhouette", "subject")
        if name in outgoing
        and name in incoming
        and outgoing[name][0] > 1e-9
        and incoming[name][0] > 1e-9
    )
    return float(median(ratios)) if ratios else None


def _pose_phase_continuation_mismatch(
    outgoing_previous: _BoundarySample,
    outgoing_endpoint: _BoundarySample,
    incoming_start: _BoundarySample,
    incoming_next: _BoundarySample,
    config: DirectMemberVisualConfig,
) -> float | None:
    """Measure whether the next pose phase continues across the visible cut.

    Endpoint similarity alone can join two matching poses that are moving into
    different phases.  The forward and backward one-frame predictions below
    bind the cut to the exact four rendered-adjacent poses while persisting only
    their aggregate residual.
    """

    outgoing_previous_pose = _actual_pose_vector(
        outgoing_previous.descriptor,
        config,
    )
    outgoing_endpoint_pose = _actual_pose_vector(
        outgoing_endpoint.descriptor,
        config,
    )
    incoming_start_pose = _actual_pose_vector(incoming_start.descriptor, config)
    incoming_next_pose = _actual_pose_vector(incoming_next.descriptor, config)
    poses = (
        outgoing_previous_pose,
        outgoing_endpoint_pose,
        incoming_start_pose,
        incoming_next_pose,
    )
    if any(value is None for value in poses):
        return None
    assert all(value is not None for value in poses)
    lengths = {len(value) for value in poses if value is not None}
    if len(lengths) != 1:
        return None
    outgoing_delta = _vector_delta(
        outgoing_previous_pose,
        outgoing_endpoint_pose,
    )
    incoming_delta = _vector_delta(incoming_start_pose, incoming_next_pose)
    if outgoing_delta is None or incoming_delta is None:
        return None
    predicted_next = tuple(
        value + delta
        for value, delta in zip(
            outgoing_endpoint_pose,
            outgoing_delta,
            strict=True,
        )
    )
    predicted_previous = tuple(
        value - delta
        for value, delta in zip(
            incoming_start_pose,
            incoming_delta,
            strict=True,
        )
    )
    forward = _embedding_distance(predicted_next, incoming_next_pose)
    backward = _embedding_distance(predicted_previous, outgoing_previous_pose)
    if forward is None or backward is None:
        return None
    return (forward + backward) / 2.0


def _subject_motion_vector(
    previous: _AttentionGeometry | None,
    current: _AttentionGeometry | None,
) -> tuple[float, ...] | None:
    if previous is None or current is None:
        return None
    previous_area = previous.bbox_xywh[2] * previous.bbox_xywh[3]
    current_area = current.bbox_xywh[2] * current.bbox_xywh[3]
    if previous_area <= 0.0 or current_area <= 0.0:
        return None
    return (
        current.centroid_xy[0] - previous.centroid_xy[0],
        current.centroid_xy[1] - previous.centroid_xy[1],
        math.log(current_area / previous_area) / math.log(4.0),
    )


def _motion_components(
    previous: _BoundarySample,
    current: _BoundarySample,
    config: DirectMemberVisualConfig,
) -> dict[str, tuple[float, tuple[float, ...]]]:
    previous_descriptor = previous.descriptor
    current_descriptor = current.descriptor
    if previous_descriptor is None or current_descriptor is None:
        return {}
    pose_delta = _vector_delta(
        _actual_pose_vector(previous_descriptor, config),
        _actual_pose_vector(current_descriptor, config),
    )
    silhouette_delta = _vector_delta(
        (
            tuple(previous_descriptor.silhouette_embedding)
            if previous_descriptor.silhouette_confidence
            >= config.minimum_silhouette_confidence
            else None
        ),
        (
            tuple(current_descriptor.silhouette_embedding)
            if current_descriptor.silhouette_confidence
            >= config.minimum_silhouette_confidence
            else None
        ),
    )
    subject_delta = _subject_motion_vector(previous.attention, current.attention)
    raw = {
        "pose": pose_delta,
        "silhouette": silhouette_delta,
        "subject": subject_delta,
    }
    result: dict[str, tuple[float, tuple[float, ...]]] = {}
    for name, vector in raw.items():
        magnitude = _delta_magnitude(vector)
        if magnitude is not None and vector is not None:
            result[name] = (magnitude, vector)
    return result


def _cut_adjacent_motion_evidence(
    *,
    outgoing_previous: _BoundarySample,
    outgoing_endpoint: _BoundarySample,
    incoming_start: _BoundarySample,
    incoming_next: _BoundarySample,
    config: DirectMemberVisualConfig,
    output_space_subject_motion_used: bool = False,
) -> DirectBoundaryMotionEvidence:
    step = config.sample_frame_step_seconds
    outgoing_master_delta = (
        outgoing_endpoint.master_seconds - outgoing_previous.master_seconds
    )
    incoming_master_delta = (
        incoming_next.master_seconds - incoming_start.master_seconds
    )
    outgoing_local_delta = (
        outgoing_endpoint.source_local_seconds
        - outgoing_previous.source_local_seconds
    )
    incoming_local_delta = (
        incoming_next.source_local_seconds
        - incoming_start.source_local_seconds
    )
    adjacent = bool(
        all(
            0.75 * step <= value <= 1.25 * step
            for value in (outgoing_master_delta, incoming_master_delta)
        )
        and outgoing_local_delta > 0.0
        and incoming_local_delta > 0.0
        and all(
            sample.descriptor is not None
            for sample in (
                outgoing_previous,
                outgoing_endpoint,
                incoming_start,
                incoming_next,
            )
        )
    )
    outgoing_components = (
        _motion_components(outgoing_previous, outgoing_endpoint, config)
        if adjacent
        else {}
    )
    incoming_components = (
        _motion_components(incoming_start, incoming_next, config)
        if adjacent
        else {}
    )
    outgoing_motion = (
        sum(value[0] for value in outgoing_components.values())
        / len(outgoing_components)
        if outgoing_components
        else None
    )
    incoming_motion = (
        sum(value[0] for value in incoming_components.values())
        / len(incoming_components)
        if incoming_components
        else None
    )
    left = outgoing_endpoint.descriptor
    right = incoming_start.descriptor
    endpoint_components: list[float] = []
    if left is not None and right is not None:
        left_pose = _actual_pose_vector(left, config)
        right_pose = _actual_pose_vector(right, config)
        pose_mismatch = _embedding_distance(left_pose or (), right_pose or ())
        if pose_mismatch is not None:
            endpoint_components.append(pose_mismatch)
        if (
            left.silhouette_confidence >= config.minimum_silhouette_confidence
            and right.silhouette_confidence
            >= config.minimum_silhouette_confidence
        ):
            silhouette_mismatch = _embedding_distance(
                left.silhouette_embedding,
                right.silhouette_embedding,
            )
            if silhouette_mismatch is not None:
                endpoint_components.append(silhouette_mismatch)
        _position_mismatch, subject_scale_mismatch = _attention_mismatch(
            outgoing_endpoint.attention,
            incoming_start.attention,
        )
        if subject_scale_mismatch is not None:
            endpoint_components.append(subject_scale_mismatch)
    endpoint_mismatch = (
        sum(endpoint_components) / len(endpoint_components)
        if endpoint_components
        else None
    )
    endpoint_compatibility = (
        max(0.0, 1.0 - endpoint_mismatch)
        if endpoint_mismatch is not None
        else None
    )
    both_moving = bool(
        adjacent
        and outgoing_motion is not None
        and incoming_motion is not None
        and outgoing_motion >= config.minimum_adjacent_motion_magnitude
        and incoming_motion >= config.minimum_adjacent_motion_magnitude
        and len(outgoing_components)
        >= config.minimum_temporal_motion_component_count
        and len(incoming_components)
        >= config.minimum_temporal_motion_component_count
    )
    common_directions = [
        cosine
        for name in ("pose", "silhouette", "subject")
        if name in outgoing_components and name in incoming_components
        and (
            cosine := _direction_cosine(
                outgoing_components[name][1],
                incoming_components[name][1],
            )
        )
        is not None
    ]
    direction_cosine = (
        sum(common_directions) / len(common_directions)
        if common_directions
        else None
    )
    speed_ratio_mismatch = _motion_speed_ratio_mismatch(
        outgoing_components,
        incoming_components,
    )
    pose_phase_mismatch = _pose_phase_continuation_mismatch(
        outgoing_previous,
        outgoing_endpoint,
        incoming_start,
        incoming_next,
        config,
    )
    motion_strength = (
        min(
            1.0,
            min(float(outgoing_motion), float(incoming_motion))
            / (config.minimum_adjacent_motion_magnitude * 4.0),
        )
        if both_moving
        else 0.0
    )
    direction_authorized = bool(
        direction_cosine is not None
        and direction_cosine + 1e-9
        >= config.minimum_motion_direction_cosine
    )
    endpoint_authorized = bool(
        endpoint_mismatch is not None
        and endpoint_mismatch
        <= config.maximum_motion_endpoint_shape_mismatch + 1e-9
    )
    speed_authorized = bool(
        speed_ratio_mismatch is None
        or speed_ratio_mismatch
        <= config.maximum_motion_speed_ratio_mismatch + 1e-9
    )
    phase_authorized = bool(
        pose_phase_mismatch is None
        or pose_phase_mismatch
        <= config.maximum_pose_phase_continuation_mismatch + 1e-9
    )
    speed_compatibility = (
        max(0.0, 1.0 - speed_ratio_mismatch)
        if speed_ratio_mismatch is not None
        else 0.75
    )
    phase_compatibility = (
        max(0.0, 1.0 - pose_phase_mismatch)
        if pose_phase_mismatch is not None
        else 0.75
    )
    soft_preference = (
        config.motion_connection_preference_weight
        * motion_strength
        * float(endpoint_compatibility)
        * speed_compatibility
        * phase_compatibility
        if (
            both_moving
            and endpoint_compatibility is not None
            and direction_authorized
            and endpoint_authorized
            and speed_authorized
            and phase_authorized
        )
        else 0.0
    )
    return DirectBoundaryMotionEvidence(
        timestamp_bound_adjacent=adjacent,
        outgoing_previous_master_seconds=outgoing_previous.master_seconds,
        outgoing_endpoint_master_seconds=outgoing_endpoint.master_seconds,
        incoming_start_master_seconds=incoming_start.master_seconds,
        incoming_next_master_seconds=incoming_next.master_seconds,
        outgoing_previous_source_local_seconds=(
            outgoing_previous.source_local_seconds
        ),
        outgoing_endpoint_source_local_seconds=(
            outgoing_endpoint.source_local_seconds
        ),
        incoming_start_source_local_seconds=incoming_start.source_local_seconds,
        incoming_next_source_local_seconds=incoming_next.source_local_seconds,
        outgoing_pose_motion_magnitude=(
            outgoing_components.get("pose", (None, ()))[0]
        ),
        incoming_pose_motion_magnitude=(
            incoming_components.get("pose", (None, ()))[0]
        ),
        outgoing_silhouette_motion_magnitude=(
            outgoing_components.get("silhouette", (None, ()))[0]
        ),
        incoming_silhouette_motion_magnitude=(
            incoming_components.get("silhouette", (None, ()))[0]
        ),
        outgoing_subject_motion_magnitude=(
            outgoing_components.get("subject", (None, ()))[0]
        ),
        incoming_subject_motion_magnitude=(
            incoming_components.get("subject", (None, ()))[0]
        ),
        outgoing_motion_magnitude=outgoing_motion,
        incoming_motion_magnitude=incoming_motion,
        outgoing_motion_component_count=len(outgoing_components),
        incoming_motion_component_count=len(incoming_components),
        both_moving=both_moving,
        endpoint_pose_shape_mismatch=endpoint_mismatch,
        endpoint_pose_shape_compatibility=endpoint_compatibility,
        direction_cosine_audit_only=direction_cosine,
        motion_speed_ratio_mismatch=speed_ratio_mismatch,
        pose_phase_continuation_mismatch=pose_phase_mismatch,
        output_space_subject_motion_used=output_space_subject_motion_used,
        soft_preference_applied=soft_preference,
    )


def _boundary_mismatch(
    outgoing: _BoundarySample,
    incoming: _BoundarySample,
    config: DirectMemberVisualConfig,
    *,
    outgoing_previous: _BoundarySample | None = None,
    incoming_next: _BoundarySample | None = None,
    outgoing_source_index: int | None = None,
    incoming_source_index: int | None = None,
    outgoing_source_geometry: DirectMemberSourceFrameGeometry | None = None,
    incoming_source_geometry: DirectMemberSourceFrameGeometry | None = None,
    continuous_official_portrait_fancam_contract: bool = False,
) -> DirectBoundaryMismatch:
    outgoing_output = _sample_in_portrait_fill(
        outgoing,
        outgoing_source_geometry,
    )
    incoming_output = _sample_in_portrait_fill(
        incoming,
        incoming_source_geometry,
    )
    outgoing_previous_output = (
        _sample_in_portrait_fill(
            outgoing_previous,
            outgoing_source_geometry,
        )
        if outgoing_previous is not None
        else None
    )
    incoming_next_output = (
        _sample_in_portrait_fill(
            incoming_next,
            incoming_source_geometry,
        )
        if incoming_next is not None
        else None
    )
    motion_connection = (
        _cut_adjacent_motion_evidence(
            outgoing_previous=outgoing_previous_output,
            outgoing_endpoint=outgoing_output,
            incoming_start=incoming_output,
            incoming_next=incoming_next_output,
            config=config,
            output_space_subject_motion_used=bool(
                outgoing_source_geometry is not None
                and incoming_source_geometry is not None
            ),
        )
        if (
            outgoing_previous_output is not None
            and incoming_next_output is not None
        )
        else None
    )
    left = outgoing.descriptor
    right = incoming.descriptor
    if left is None or right is None:
        return DirectBoundaryMismatch(
            cost=config.missing_descriptor_cost,
            base_mismatch=1.0,
            attention_position_mismatch=None,
            attention_scale_mismatch=None,
            dominant_pose_mismatch=None,
            pose_basis=None,
            formation_mismatch=None,
            silhouette_mismatch=None,
            person_layout_mismatch=None,
            person_count_mismatch=None,
            evidence_component_count=0,
            warning_flags=("missing_visual_descriptor",),
            severe_flags=("missing_visual_descriptor",),
            motion_connection=motion_connection,
        )

    outgoing_attention = outgoing_output.attention
    incoming_attention = incoming_output.attention
    attention_position, attention_scale = _attention_mismatch(
        outgoing_attention,
        incoming_attention,
    )
    actual_pose = (
        left.pose_backend != "structure_hog_fallback"
        and right.pose_backend != "structure_hog_fallback"
        and left.pose_confidence >= config.minimum_pose_confidence
        and right.pose_confidence >= config.minimum_pose_confidence
        and bool(left.pose_embedding)
        and len(left.pose_embedding) == len(right.pose_embedding)
    )
    if actual_pose:
        pose_basis = "opencv_zoo_mediapipe_dominant_pose"
        pose = _embedding_distance(left.pose_embedding, right.pose_embedding)
        pose_weight = config.dominant_pose_weight
    else:
        pose_basis = "structure_hog_pose_fallback"
        pose = _embedding_distance(
            left.structure_embedding,
            right.structure_embedding,
        )
        pose_weight = config.structure_pose_weight
    formation = formation_assignment_mismatch(
        left.formation_people,
        right.formation_people,
    )
    silhouette = (
        _embedding_distance(
            left.silhouette_embedding,
            right.silhouette_embedding,
        )
        if left.silhouette_confidence >= config.minimum_silhouette_confidence
        and right.silhouette_confidence >= config.minimum_silhouette_confidence
        else None
    )
    layout = _embedding_distance(
        left.person_layout_embedding,
        right.person_layout_embedding,
    )
    person_count = _person_count_mismatch(
        left.detected_person_count,
        right.detected_person_count,
    )
    weighted_components = (
        (attention_position, config.attention_position_weight),
        (attention_scale, config.attention_scale_weight),
        (pose, pose_weight),
        (formation, config.formation_weight),
        (silhouette, config.silhouette_weight),
        (layout, config.person_layout_weight),
        (person_count, config.person_count_weight),
    )
    available = tuple(
        (float(value), weight)
        for value, weight in weighted_components
        if value is not None
    )
    raw_base = (
        sum(value * weight for value, weight in available)
        / sum(weight for _value, weight in available)
        if available
        else 1.0
    )
    components = {
        "attention_position": attention_position,
        "attention_scale": attention_scale,
        "dominant_pose": pose,
        "formation": formation,
        "silhouette": silhouette,
        "person_layout": layout,
        "person_count": person_count,
    }
    warning_flags = [
        name
        for name, threshold in config.warning_thresholds().items()
        if components[name] is not None and float(components[name]) > threshold
    ]
    raw_severe_flags = [
        name
        for name, threshold in config.warning_thresholds().items()
        if components[name] is not None
        and float(components[name])
        > min(1.0, threshold + config.severe_risk_margin)
    ]
    official_face_pair = bool(
        continuous_official_portrait_fancam_contract
        and outgoing_source_geometry is not None
        and incoming_source_geometry is not None
        and outgoing_source_index is not None
        and incoming_source_index is not None
        and outgoing_attention is not None
        and incoming_attention is not None
        and outgoing_attention.basis
        == incoming_attention.basis
        == "unambiguous_dominant_face_bbox_geometry_only"
        and left.face_count == right.face_count == 1
        and left.face_detector_backend.casefold()
        == right.face_detector_backend.casefold()
        == "yunet"
    )
    official_silhouette_pair = bool(
        continuous_official_portrait_fancam_contract
        and outgoing_source_geometry is not None
        and incoming_source_geometry is not None
        and outgoing_source_index is not None
        and incoming_source_index is not None
        and outgoing_attention is not None
        and incoming_attention is not None
        and outgoing_attention.basis
        == incoming_attention.basis
        == "aggregate_human_silhouette"
        and left.silhouette_backend
        == right.silhouette_backend
        == SILHOUETTE_BACKEND
        and left.silhouette_confidence
        >= config.minimum_silhouette_confidence
        and right.silhouette_confidence
        >= config.minimum_silhouette_confidence
    )
    official_dominant_person_pair = bool(
        continuous_official_portrait_fancam_contract
        and outgoing_source_geometry is not None
        and incoming_source_geometry is not None
        and outgoing_source_index is not None
        and incoming_source_index is not None
        and outgoing_attention is not None
        and incoming_attention is not None
        and outgoing_attention.basis
        == incoming_attention.basis
        == "unambiguous_dominant_person_box_geometry_only"
        and bool(left.formation_people)
        and bool(right.formation_people)
    )
    official_body_pair = bool(
        official_silhouette_pair or official_dominant_person_pair
    )
    official_alignment_pair = bool(official_face_pair or official_body_pair)
    alignment_geometry_basis = (
        "unambiguous_dominant_face_bbox_geometry_only"
        if official_face_pair
        else outgoing_attention.basis
        if official_body_pair and outgoing_attention is not None
        else None
    )
    member_context_softened_flags = tuple(
        name
        for name in raw_severe_flags
        if official_alignment_pair
        and name in {"formation", "person_layout", "person_count"}
    )
    cross_source_edge = bool(
        outgoing_source_index is not None
        and incoming_source_index is not None
        and outgoing_source_index != incoming_source_index
    )
    transform_repair: DirectMemberBoundaryTransformRepair | None = None
    repairable_attention_flags = tuple(
        name
        for name in raw_severe_flags
        if name in {"attention_position", "attention_scale"}
    )
    alignment_transform_required = bool(
        cross_source_edge
        and official_alignment_pair
        and (
            attention_position is not None
            and attention_position
            > (
                config.transform_alignment_position_trigger
                if official_face_pair
                else config.transform_body_alignment_position_trigger
            )
            + 1e-9
            or attention_scale is not None
            and attention_scale
            > (
                config.transform_alignment_scale_trigger
                if official_face_pair
                else config.transform_body_alignment_scale_trigger
            )
            + 1e-9
        )
    )
    if official_alignment_pair and (
        repairable_attention_flags or alignment_transform_required
    ):
        assert outgoing_attention is not None
        assert incoming_attention is not None
        assert outgoing_source_geometry is not None
        assert incoming_source_geometry is not None
        assert outgoing_source_index is not None
        assert incoming_source_index is not None
        if official_face_pair:
            outgoing_subject, outgoing_subject_basis = (
                _verified_direct_member_subject_geometry(left, config)
            )
            incoming_subject, incoming_subject_basis = (
                _verified_direct_member_subject_geometry(right, config)
            )
            outgoing_subject_box = (
                map_box_to_portrait_fill(
                    outgoing_subject,
                    outgoing_source_geometry,
                )
                if outgoing_subject is not None
                else None
            )
            incoming_subject_box = (
                map_box_to_portrait_fill(
                    incoming_subject,
                    incoming_source_geometry,
                )
                if incoming_subject is not None
                else None
            )
        else:
            outgoing_subject_box = outgoing_attention.bbox_xywh
            incoming_subject_box = incoming_attention.bbox_xywh
            outgoing_subject_basis = alignment_geometry_basis
            incoming_subject_basis = alignment_geometry_basis
        transform_repair = find_lossless_boundary_transform_repair(
            outgoing_face_box=outgoing_attention.bbox_xywh,
            incoming_face_box=incoming_attention.bbox_xywh,
            outgoing_source_index=outgoing_source_index,
            incoming_source_index=incoming_source_index,
            outgoing_source_geometry=outgoing_source_geometry,
            incoming_source_geometry=incoming_source_geometry,
            outgoing_subject_box=outgoing_subject_box,
            incoming_subject_box=incoming_subject_box,
            outgoing_subject_basis=outgoing_subject_basis,
            incoming_subject_basis=incoming_subject_basis,
            maximum_zoom=config.transform_repair_maximum_zoom,
            maximum_center_shift=(
                config.transform_repair_maximum_center_shift
            ),
            maximum_boundary_center_distance=(
                config.transform_repair_maximum_boundary_center_distance
            ),
            maximum_residual_center_distance=(
                config.transform_repair_maximum_residual_center_distance
            ),
            maximum_residual_attention_scale_mismatch=(
                config.attention_scale_warning
            ),
            minimum_improvement_fraction=(
                config.transform_repair_minimum_improvement_fraction
            ),
            minimum_face_visible_fraction=(
                config.transform_repair_minimum_face_visible_fraction
            ),
            legacy_face_only_maximum_zoom=(
                config.transform_repair_legacy_face_only_maximum_zoom
            ),
            legacy_face_only_maximum_center_shift=(
                config.transform_repair_legacy_face_only_maximum_center_shift
            ),
            minimum_subject_visible_fraction=(
                config.transform_repair_minimum_subject_visible_fraction
            ),
            alignment_geometry_basis=str(alignment_geometry_basis),
        )
        if transform_repair is not None:
            transform_repair = transform_repair.with_repaired_severe_flags(
                repairable_attention_flags
            )

    # Motion must be judged in the same portrait-fill/camera domain that will
    # reach the viewer.  Raw source coordinates can reverse or exaggerate
    # apparent travel after an aspect-fill crop, while an authorized pan/zoom
    # changes the exact boundary trajectory again.  Recompute the four-frame
    # scalar evidence after that transform instead of carrying the pre-render
    # source-coordinate result into selection.
    if (
        outgoing_previous_output is not None
        and incoming_next_output is not None
    ):
        motion_outgoing_previous = outgoing_previous_output
        motion_outgoing = outgoing_output
        motion_incoming = incoming_output
        motion_incoming_next = incoming_next_output
        if transform_repair is not None:
            if transform_repair.transformed_side == "outgoing":
                motion_outgoing_previous = _sample_with_boundary_transform(
                    motion_outgoing_previous,
                    transform_repair,
                )
                motion_outgoing = _sample_with_boundary_transform(
                    motion_outgoing,
                    transform_repair,
                )
            else:
                motion_incoming = _sample_with_boundary_transform(
                    motion_incoming,
                    transform_repair,
                )
                motion_incoming_next = _sample_with_boundary_transform(
                    motion_incoming_next,
                    transform_repair,
                )
        motion_connection = _cut_adjacent_motion_evidence(
            outgoing_previous=motion_outgoing_previous,
            outgoing_endpoint=motion_outgoing,
            incoming_start=motion_incoming,
            incoming_next=motion_incoming_next,
            config=config,
            output_space_subject_motion_used=bool(
                outgoing_source_geometry is not None
                and incoming_source_geometry is not None
            ),
        )

    effective_attention_position = (
        transform_repair.residual_attention_position_mismatch
        if transform_repair is not None
        else attention_position
    )
    effective_attention_scale = (
        transform_repair.residual_attention_scale_mismatch
        if transform_repair is not None
        else attention_scale
    )
    effective_weighted_components = (
        (effective_attention_position, config.attention_position_weight),
        (effective_attention_scale, config.attention_scale_weight),
        (pose, pose_weight),
        (formation, config.formation_weight),
        (silhouette, config.silhouette_weight),
        (layout, config.person_layout_weight),
        (person_count, config.person_count_weight),
    )
    effective_available = tuple(
        (float(value), weight)
        for value, weight in effective_weighted_components
        if value is not None
    )
    base = (
        sum(value * weight for value, weight in effective_available)
        / sum(weight for _value, weight in effective_available)
        if effective_available
        else raw_base
    )
    if base > config.total_mismatch_warning:
        warning_flags.append("total_mismatch")
    if len(available) < config.minimum_evidence_component_count:
        warning_flags.append("insufficient_geometry_evidence")
    if transform_repair is not None:
        warning_flags.append("lossless_transform_repair_required")

    severe_flags = [
        name
        for name in raw_severe_flags
        if name not in member_context_softened_flags
        and not (
            transform_repair is not None
            and name in transform_repair.repaired_severe_flags
        )
    ]
    if cross_source_edge:
        if alignment_transform_required and transform_repair is None:
            severe_flags.append(
                "cross_source_face_alignment_unrepaired"
                if official_face_pair
                else "cross_source_body_alignment_unrepaired"
            )
        if config.require_actual_pose_for_cross_source_cut and not actual_pose:
            severe_flags.append("cross_source_actual_pose_missing")
        elif (
            pose is not None
            and pose > config.maximum_cross_source_pose_mismatch + 1e-9
        ):
            severe_flags.append("cross_source_pose_mismatch")
        if silhouette is None:
            severe_flags.append("cross_source_silhouette_missing")
        elif (
            silhouette
            > config.maximum_cross_source_silhouette_mismatch + 1e-9
        ):
            severe_flags.append("cross_source_silhouette_mismatch")
        if motion_connection is not None and motion_connection.both_moving:
            if motion_connection.direction_cosine_audit_only is None:
                severe_flags.append("motion_direction_evidence_missing")
            elif (
                motion_connection.direction_cosine_audit_only
                < config.minimum_motion_direction_cosine - 1e-9
            ):
                severe_flags.append("motion_direction_opposes_cut")
            if motion_connection.endpoint_pose_shape_mismatch is None:
                severe_flags.append("motion_endpoint_shape_evidence_missing")
            elif (
                motion_connection.endpoint_pose_shape_mismatch
                > config.maximum_motion_endpoint_shape_mismatch + 1e-9
            ):
                severe_flags.append("motion_endpoint_shape_mismatch")
            if (
                motion_connection.motion_speed_ratio_mismatch is not None
                and motion_connection.motion_speed_ratio_mismatch
                > config.maximum_motion_speed_ratio_mismatch + 1e-9
            ):
                severe_flags.append("motion_speed_discontinuity")
            if (
                motion_connection.pose_phase_continuation_mismatch is not None
                and motion_connection.pose_phase_continuation_mismatch
                > config.maximum_pose_phase_continuation_mismatch + 1e-9
            ):
                severe_flags.append("pose_phase_discontinuity")
    warning_flags.extend(severe_flags)
    if official_alignment_pair:
        hard_weighted_components = (
            (effective_attention_position, config.attention_position_weight),
            (effective_attention_scale, config.attention_scale_weight),
            (pose, pose_weight),
            (silhouette, config.silhouette_weight),
        )
        hard_available = tuple(
            (float(value), weight)
            for value, weight in hard_weighted_components
            if value is not None
        )
        hard_base = (
            sum(value * weight for value, weight in hard_available)
            / sum(weight for _value, weight in hard_available)
            if hard_available
            else 1.0
        )
    else:
        hard_base = raw_base
    if hard_base > config.total_mismatch_severe:
        severe_flags.append("total_mismatch")
    if not available:
        severe_flags.append("no_geometry_evidence")
    severe_flags = list(dict.fromkeys(severe_flags))
    cinematic_bridge: DirectMemberCinematicBridge | None = None
    if (
        severe_flags
        and official_face_pair
        and actual_pose
        and pose is not None
        and pose
        <= _DIRECT_MEMBER_CINEMATIC_BRIDGE_MAXIMUM_POSE_MISMATCH + 1e-9
        and silhouette is not None
        and silhouette
        <= _DIRECT_MEMBER_CINEMATIC_BRIDGE_MAXIMUM_SILHOUETTE_MISMATCH
        + 1e-9
        and base <= _DIRECT_MEMBER_CINEMATIC_BRIDGE_MAXIMUM_BASE_MISMATCH + 1e-9
        and frozenset(severe_flags)
        <= _DIRECT_MEMBER_CINEMATIC_BRIDGE_ALLOWED_SEVERE_FLAGS
        and bool(
            frozenset(severe_flags)
            & {"attention_position", "attention_scale"}
        )
        and outgoing_source_geometry is not None
        and incoming_source_geometry is not None
        and (
            outgoing_source_geometry.source_width
            > outgoing_source_geometry.source_height
        )
        != (
            incoming_source_geometry.source_width
            > incoming_source_geometry.source_height
        )
        and motion_connection is not None
        and motion_connection.timestamp_bound_adjacent
    ):
        outgoing_is_landscape = bool(
            outgoing_source_geometry.source_width
            > outgoing_source_geometry.source_height
        )
        cinematic_bridge = DirectMemberCinematicBridge(
            style=("horizontal_blur" if outgoing_is_landscape else "zoom_in"),
            outgoing_is_landscape=outgoing_is_landscape,
            incoming_is_landscape=not outgoing_is_landscape,
            authorized_severe_flags=tuple(severe_flags),
            pose_mismatch=float(pose),
            silhouette_mismatch=float(silhouette),
            base_mismatch=float(base),
        )
        warning_flags.append("cinematic_effect_bridge_required")
    warning_flags = list(dict.fromkeys(warning_flags))
    cost = base
    cost += config.warning_penalty * len(warning_flags)
    cost += config.severe_penalty * len(severe_flags)
    if len(available) < config.minimum_evidence_component_count:
        missing_count = config.minimum_evidence_component_count - len(available)
        cost += config.insufficient_evidence_penalty * missing_count
    if motion_connection is not None:
        cost -= motion_connection.soft_preference_applied
    if transform_repair is not None:
        cost += 0.35 * (
            (transform_repair.zoom - 1.0)
            / max(1e-9, config.transform_repair_maximum_zoom - 1.0)
            + transform_repair.center_shift
            / max(1e-9, config.transform_repair_maximum_center_shift)
        )
    return DirectBoundaryMismatch(
        cost=cost,
        base_mismatch=base,
        attention_position_mismatch=attention_position,
        attention_scale_mismatch=attention_scale,
        dominant_pose_mismatch=pose,
        pose_basis=pose_basis if pose is not None else None,
        formation_mismatch=formation,
        silhouette_mismatch=silhouette,
        person_layout_mismatch=layout,
        person_count_mismatch=person_count,
        evidence_component_count=len(available),
        warning_flags=tuple(warning_flags),
        severe_flags=tuple(severe_flags),
        motion_connection=motion_connection,
        transform_repair=transform_repair,
        member_context_softened_flags=member_context_softened_flags,
        cinematic_bridge=cinematic_bridge,
    )


def _source_local_time(
    master_seconds: float,
    *,
    source_index: int,
    audio_offsets: Sequence[float],
    source_timewarps: Sequence[TimeWarpMap | None],
) -> float:
    timewarp = source_timewarps[source_index]
    if timewarp is None:
        return master_seconds + float(audio_offsets[source_index])
    return timewarp.master_to_source(master_seconds)


def _source_master_time(
    source_seconds: float,
    *,
    source_index: int,
    audio_offsets: Sequence[float],
    source_timewarps: Sequence[TimeWarpMap | None],
) -> float:
    timewarp = source_timewarps[source_index]
    if timewarp is None:
        return source_seconds - float(audio_offsets[source_index])
    return timewarp.source_to_master(source_seconds)


@dataclass(frozen=True, slots=True)
class _BeamNode:
    cost: float
    path: tuple[int, ...]
    duration_totals: tuple[float, ...]


_BeamStateKey = tuple[
    tuple[int, ...],
    int,
    int,
    tuple[float, ...],
]


@dataclass(frozen=True, slots=True)
class _BeamAttemptResult:
    beam_width: int
    states: dict[_BeamStateKey, _BeamNode]
    pruned_state_count: int
    pruned_segment_indices: tuple[int, ...]
    peak_generated_state_count: int
    failed_at_segment_index: int | None
    generated_state_capacity_exhausted: bool

    def to_dict(self) -> dict[str, Any]:
        if self.generated_state_capacity_exhausted:
            outcome = "generated_state_capacity_exhausted"
        elif self.failed_at_segment_index is not None:
            outcome = "frontier_exhausted"
        else:
            outcome = "terminal_frontier_reached"
        return {
            "beam_width": self.beam_width,
            "outcome": outcome,
            "failed_at_segment_index": self.failed_at_segment_index,
            "peak_generated_state_count": self.peak_generated_state_count,
            "pruned_state_count": self.pruned_state_count,
            "pruned_segment_indices": list(self.pruned_segment_indices),
            "terminal_state_count": len(self.states),
            "generated_state_capacity_exhausted": (
                self.generated_state_capacity_exhausted
            ),
        }


@dataclass(frozen=True, slots=True)
class _FeasibilityLifelineResult:
    node: _BeamNode | None
    explored_state_count: int
    memoized_failed_state_count: int
    state_capacity_exhausted: bool
    root_candidate_diagnostics: tuple[dict[str, Any], ...]
    admissible_pruning_diagnostics: dict[str, Any]
    strategy: str = "exact_backward_memoized_hard_feasibility"
    solver_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "path_found": self.node is not None,
            "explored_state_count": self.explored_state_count,
            "memoized_failed_state_count": self.memoized_failed_state_count,
            "state_capacity_exhausted": self.state_capacity_exhausted,
            "duration_state_quantization_applied": False,
            "admissible_pruning": dict(
                self.admissible_pruning_diagnostics
            ),
            "root_candidate_diagnostics": [
                dict(item) for item in self.root_candidate_diagnostics
            ],
            "solver": dict(self.solver_diagnostics),
        }


def _adaptive_beam_widths(initial: int, maximum: int) -> tuple[int, ...]:
    effective_maximum = max(initial, maximum)
    widths: list[int] = []
    width = initial
    while True:
        widths.append(width)
        if width >= effective_maximum:
            break
        width = min(effective_maximum, width * 2)
    return tuple(widths)


def _path_optimizer_cost(
    path: Sequence[int],
    *,
    matrices: Sequence[Sequence[Sequence[DirectBoundaryMismatch]]],
    durations: Sequence[float],
    duration_targets: Sequence[float],
    config: DirectMemberVisualConfig,
) -> tuple[float, tuple[float, ...]]:
    duration_totals = [0.0] * len(duration_targets)
    cost = 0.0
    for position, source_index in enumerate(path):
        if position:
            cost += matrices[position - 1][path[position - 1]][source_index].cost
        if position >= 2 and source_index == path[position - 2]:
            cost += config.ping_pong_penalty
        previous_duration = duration_totals[source_index]
        updated_duration = previous_duration + float(durations[position])
        duration_totals[source_index] = updated_duration
        target = duration_targets[source_index]
        cost += config.duration_balance_weight * (
            (updated_duration / target) ** 2
            - (previous_duration / target) ** 2
        )
    return (cost, tuple(duration_totals))


def _run_bounded_beam_attempt(
    matrices: Sequence[Sequence[Sequence[DirectBoundaryMismatch]]],
    *,
    source_quotas: Sequence[int],
    source_order: Sequence[int],
    candidate_validity: Sequence[Sequence[bool]],
    durations: Sequence[float],
    duration_targets: Sequence[float],
    future_valid_counts: Sequence[Sequence[int]],
    beam_width: int,
    generated_state_limit: int,
    config: DirectMemberVisualConfig,
) -> _BeamAttemptResult:
    span_count = len(candidate_validity)
    source_count = len(source_quotas)
    states: dict[_BeamStateKey, _BeamNode] = {}
    for source_index in source_order:
        if not candidate_validity[0][source_index]:
            continue
        counts = [0] * source_count
        counts[source_index] = 1
        duration_totals = [0.0] * source_count
        duration_totals[source_index] = durations[0]
        normalized_load = durations[0] / duration_targets[source_index]
        cost = config.duration_balance_weight * normalized_load**2
        key = (
            tuple(counts),
            source_index,
            -1,
            _duration_signature(duration_totals),
        )
        states[key] = _BeamNode(cost, (source_index,), tuple(duration_totals))

    pruned_state_count = 0
    pruned_segments: list[int] = []
    peak_generated_state_count = len(states)
    for position in range(1, span_count):
        matrix = matrices[position - 1]
        next_states: dict[_BeamStateKey, _BeamNode] = {}
        remaining_after = span_count - position - 1
        capacity_exhausted = False
        for (
            counts,
            previous,
            previous_previous,
            _duration_state,
        ), node in states.items():
            for source_index in source_order:
                if source_index == previous:
                    continue
                if not candidate_validity[position][source_index]:
                    continue
                if matrix[previous][source_index].severe_flags:
                    continue
                if counts[source_index] >= source_quotas[source_index]:
                    continue
                updated = list(counts)
                updated[source_index] += 1
                deficits = tuple(
                    source_quotas[index] - updated[index]
                    for index in range(source_count)
                )
                feasible = True
                for index, deficit in enumerate(deficits):
                    if deficit > future_valid_counts[position + 1][index]:
                        feasible = False
                        break
                    maximum_slots = (
                        remaining_after // 2
                        if index == source_index
                        else (remaining_after + 1) // 2
                    )
                    if deficit > maximum_slots:
                        feasible = False
                        break
                if not feasible:
                    continue
                ping_pong = source_index == previous_previous
                cost = node.cost + matrix[previous][source_index].cost
                if ping_pong:
                    cost += config.ping_pong_penalty
                duration_totals = list(node.duration_totals)
                previous_duration = duration_totals[source_index]
                updated_duration = previous_duration + durations[position]
                duration_totals[source_index] = updated_duration
                target_duration = duration_targets[source_index]
                cost += config.duration_balance_weight * (
                    (updated_duration / target_duration) ** 2
                    - (previous_duration / target_duration) ** 2
                )
                path = node.path + (source_index,)
                key = (
                    tuple(updated),
                    source_index,
                    previous,
                    _duration_signature(duration_totals),
                )
                existing = next_states.get(key)
                if existing is None and len(next_states) >= generated_state_limit:
                    capacity_exhausted = True
                    break
                if existing is None or (cost, path) < (existing.cost, existing.path):
                    next_states[key] = _BeamNode(
                        cost,
                        path,
                        tuple(duration_totals),
                    )
            if capacity_exhausted:
                break
        peak_generated_state_count = max(
            peak_generated_state_count,
            len(next_states),
        )
        if capacity_exhausted:
            return _BeamAttemptResult(
                beam_width=beam_width,
                states=states,
                pruned_state_count=pruned_state_count,
                pruned_segment_indices=tuple(pruned_segments),
                peak_generated_state_count=peak_generated_state_count,
                failed_at_segment_index=position,
                generated_state_capacity_exhausted=True,
            )
        if not next_states:
            return _BeamAttemptResult(
                beam_width=beam_width,
                states={},
                pruned_state_count=pruned_state_count,
                pruned_segment_indices=tuple(pruned_segments),
                peak_generated_state_count=peak_generated_state_count,
                failed_at_segment_index=position,
                generated_state_capacity_exhausted=False,
            )
        if len(next_states) > beam_width:
            pruned_state_count += len(next_states) - beam_width
            pruned_segments.append(position)
            states = dict(
                heapq.nsmallest(
                    beam_width,
                    next_states.items(),
                    key=lambda item: (item[1].cost, item[1].path),
                )
            )
        else:
            states = next_states
    return _BeamAttemptResult(
        beam_width=beam_width,
        states=states,
        pruned_state_count=pruned_state_count,
        pruned_segment_indices=tuple(pruned_segments),
        peak_generated_state_count=peak_generated_state_count,
        failed_at_segment_index=None,
        generated_state_capacity_exhausted=False,
    )


_DIRECT_MEMBER_MILP_MINIMUM_SPAN_COUNT = 40
_DIRECT_MEMBER_MILP_MINIMUM_SOURCE_COUNT = 6
_DIRECT_MEMBER_MILP_TIME_LIMIT_SECONDS = 45.0


def _prefer_direct_member_milp(
    candidate_validity: Sequence[Sequence[bool]],
) -> bool:
    return bool(
        len(candidate_validity) >= _DIRECT_MEMBER_MILP_MINIMUM_SPAN_COUNT
        and candidate_validity
        and len(candidate_validity[0])
        >= _DIRECT_MEMBER_MILP_MINIMUM_SOURCE_COUNT
    )


def _milp_feasibility_lifeline(
    matrices: Sequence[Sequence[Sequence[DirectBoundaryMismatch]]],
    *,
    source_quotas: Sequence[int],
    source_order: Sequence[int],
    candidate_validity: Sequence[Sequence[bool]],
    durations: Sequence[float],
    duration_targets: Sequence[float],
    state_limit: int,
    config: DirectMemberVisualConfig,
    severe_transition_budget_mask: (
        Sequence[Sequence[Sequence[bool]]] | None
    ) = None,
    minimum_severe_transition_count: int | None = None,
    maximum_severe_transition_count: int | None = None,
) -> _FeasibilityLifelineResult | None:
    """Solve the same hard path contract as a sparse binary MILP.

    The model contains one source-choice variable per span, one chosen-edge
    variable per admissible cross-source boundary, and soft-objective helpers
    for ping-pong and duration centering. Quotas, content validity, exact
    source adjacency, severe-edge eligibility, and duration bands remain hard
    constraints. A reconstructed path is independently audited before use.
    """

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_array
    except (ImportError, OSError):
        return None

    span_count = len(candidate_validity)
    source_count = len(source_quotas)
    if not span_count or source_count < 2:
        return None
    if severe_transition_budget_mask is not None and (
        maximum_severe_transition_count is None
    ):
        raise ValueError("MILP severe mask requires a maximum count")

    x_count = span_count * source_count
    edge_specs: list[tuple[int, int, int, int, float]] = []
    edges_by_boundary: list[list[int]] = [
        [] for _ in range(max(0, span_count - 1))
    ]
    for boundary_index in range(span_count - 1):
        for left_source in range(source_count):
            if not candidate_validity[boundary_index][left_source]:
                continue
            for right_source in range(source_count):
                if (
                    right_source == left_source
                    or not candidate_validity[boundary_index + 1][right_source]
                ):
                    continue
                mismatch = matrices[boundary_index][left_source][right_source]
                if severe_transition_budget_mask is None:
                    if mismatch.severe_flags:
                        continue
                    severe_cost = 0
                else:
                    severe_cost = int(
                        bool(
                            severe_transition_budget_mask[boundary_index][
                                left_source
                            ][right_source]
                        )
                    )
                    if mismatch.severe_flags and severe_cost == 0:
                        continue
                edge_index = len(edge_specs)
                edge_specs.append(
                    (
                        boundary_index,
                        left_source,
                        right_source,
                        severe_cost,
                        float(mismatch.cost),
                    )
                )
                edges_by_boundary[boundary_index].append(edge_index)
    if any(not values for values in edges_by_boundary):
        return _FeasibilityLifelineResult(
            node=None,
            explored_state_count=0,
            memoized_failed_state_count=0,
            state_capacity_exhausted=False,
            root_candidate_diagnostics=(
                {
                    "search_outcome": "milp_boundary_edge_domain_empty",
                    "empty_boundary_indices": [
                        index
                        for index, values in enumerate(edges_by_boundary)
                        if not values
                    ],
                },
            ),
            admissible_pruning_diagnostics={
                "contract": "sparse_binary_path_milp_exact_hard_constraints_v1",
            },
            strategy="scipy_highs_binary_path_milp_v1",
            solver_diagnostics={
                "invoked": False,
                "infeasibility_proven_by_empty_edge_domain": True,
            },
        )

    edge_base = x_count
    ping_specs = tuple(
        (position, source_index)
        for position in range(2, span_count)
        for source_index in range(source_count)
    )
    ping_base = edge_base + len(edge_specs)
    duration_slack_base = ping_base + len(ping_specs)
    variable_count = duration_slack_base + 2 * source_count
    objective = np.zeros(variable_count, dtype=np.float64)
    lower_bounds = np.zeros(variable_count, dtype=np.float64)
    upper_bounds = np.ones(variable_count, dtype=np.float64)
    integrality = np.ones(variable_count, dtype=np.int32)
    order_rank = {
        source_index: rank for rank, source_index in enumerate(source_order)
    }
    for position in range(span_count):
        for source_index in range(source_count):
            variable = position * source_count + source_index
            if (
                not candidate_validity[position][source_index]
                or source_quotas[source_index] <= 0
            ):
                upper_bounds[variable] = 0.0
            objective[variable] = order_rank[source_index] * 1e-9
    for edge_index, (
        _boundary,
        _left,
        right,
        _severe,
        cost,
    ) in enumerate(edge_specs):
        objective[edge_base + edge_index] = cost + order_rank[right] * 1e-8
    for ping_index in range(len(ping_specs)):
        objective[ping_base + ping_index] = config.ping_pong_penalty
    for source_index, target in enumerate(duration_targets):
        positive = duration_slack_base + 2 * source_index
        negative = positive + 1
        integrality[positive] = 0
        integrality[negative] = 0
        upper_bounds[positive] = max(target, sum(durations))
        upper_bounds[negative] = max(target, sum(durations))
        weight = config.duration_balance_weight / max(1e-9, target)
        objective[positive] = weight
        objective[negative] = weight

    rows: list[int] = []
    columns: list[int] = []
    coefficients: list[float] = []
    constraint_lower: list[float] = []
    constraint_upper: list[float] = []

    def add_constraint(
        terms: Sequence[tuple[int, float]],
        lower: float,
        upper: float,
    ) -> None:
        row = len(constraint_lower)
        for variable, coefficient in terms:
            rows.append(row)
            columns.append(variable)
            coefficients.append(float(coefficient))
        constraint_lower.append(float(lower))
        constraint_upper.append(float(upper))

    for position in range(span_count):
        add_constraint(
            tuple(
                (position * source_count + source_index, 1.0)
                for source_index in range(source_count)
            ),
            1.0,
            1.0,
        )
    total_duration = float(sum(durations))
    for source_index in range(source_count):
        source_variables = tuple(
            (position * source_count + source_index, 1.0)
            for position in range(span_count)
        )
        add_constraint(
            source_variables,
            float(source_quotas[source_index]),
            float(source_quotas[source_index]),
        )
        duration_terms = tuple(
            (
                position * source_count + source_index,
                float(durations[position]),
            )
            for position in range(span_count)
        )
        target = float(duration_targets[source_index])
        tolerance = config.maximum_source_duration_relative_deviation
        add_constraint(
            duration_terms,
            target * (1.0 - tolerance),
            target * (1.0 + tolerance),
        )
        positive = duration_slack_base + 2 * source_index
        negative = positive + 1
        add_constraint(
            duration_terms + ((positive, -1.0), (negative, 1.0)),
            target,
            target,
        )
    for boundary_index, edge_indices in enumerate(edges_by_boundary):
        add_constraint(
            tuple((edge_base + edge_index, 1.0) for edge_index in edge_indices),
            1.0,
            1.0,
        )
        for edge_index in edge_indices:
            _boundary, left, right, _severe, _cost = edge_specs[edge_index]
            edge_variable = edge_base + edge_index
            add_constraint(
                (
                    (edge_variable, 1.0),
                    (boundary_index * source_count + left, -1.0),
                ),
                -math.inf,
                0.0,
            )
            add_constraint(
                (
                    (edge_variable, 1.0),
                    ((boundary_index + 1) * source_count + right, -1.0),
                ),
                -math.inf,
                0.0,
            )
    for ping_index, (position, source_index) in enumerate(ping_specs):
        ping_variable = ping_base + ping_index
        current = position * source_count + source_index
        previous_previous = (position - 2) * source_count + source_index
        add_constraint(
            ((ping_variable, 1.0), (current, -1.0)),
            -math.inf,
            0.0,
        )
        add_constraint(
            ((ping_variable, 1.0), (previous_previous, -1.0)),
            -math.inf,
            0.0,
        )
        add_constraint(
            (
                (ping_variable, 1.0),
                (current, -1.0),
                (previous_previous, -1.0),
            ),
            -1.0,
            math.inf,
        )
    if severe_transition_budget_mask is not None:
        severe_terms = tuple(
            (edge_base + edge_index, 1.0)
            for edge_index, spec in enumerate(edge_specs)
            if spec[3]
        )
        add_constraint(
            severe_terms,
            float(minimum_severe_transition_count or 0),
            float(maximum_severe_transition_count),
        )

    matrix = coo_array(
        (
            np.asarray(coefficients, dtype=np.float64),
            (
                np.asarray(rows, dtype=np.int32),
                np.asarray(columns, dtype=np.int32),
            ),
        ),
        shape=(len(constraint_lower), variable_count),
    ).tocsr()
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(
            matrix,
            np.asarray(constraint_lower, dtype=np.float64),
            np.asarray(constraint_upper, dtype=np.float64),
        ),
        options={
            "disp": False,
            "presolve": True,
            "time_limit": _DIRECT_MEMBER_MILP_TIME_LIMIT_SECONDS,
            "node_limit": int(max(1, state_limit)),
            "mip_rel_gap": 0.0,
        },
    )
    status = int(result.status)
    solver_diagnostics = {
        "invoked": True,
        "backend": "scipy_optimize_milp_highs",
        "status": status,
        "message": str(result.message),
        "success": bool(result.success),
        "variable_count": variable_count,
        "binary_variable_count": duration_slack_base,
        "continuous_duration_slack_variable_count": 2 * source_count,
        "constraint_count": len(constraint_lower),
        "edge_variable_count": len(edge_specs),
        "ping_pong_variable_count": len(ping_specs),
        "mip_node_count": int(getattr(result, "mip_node_count", 0) or 0),
        "mip_gap": (
            float(result.mip_gap)
            if getattr(result, "mip_gap", None) is not None
            else None
        ),
        "objective_value": (
            float(result.fun) if getattr(result, "fun", None) is not None else None
        ),
        "time_limit_seconds": _DIRECT_MEMBER_MILP_TIME_LIMIT_SECONDS,
        "duration_state_quantization_applied": False,
        "postsolve_path_audit_required": True,
    }
    if status == 2:
        return _FeasibilityLifelineResult(
            node=None,
            explored_state_count=solver_diagnostics["mip_node_count"],
            memoized_failed_state_count=0,
            state_capacity_exhausted=False,
            root_candidate_diagnostics=(
                {
                    "search_outcome": "milp_infeasibility_proven",
                    "final_segment_index": span_count - 1,
                    "solver_status": status,
                },
            ),
            admissible_pruning_diagnostics={
                "contract": "sparse_binary_path_milp_exact_hard_constraints_v1",
                "constraint_infeasibility_proven": True,
            },
            strategy="scipy_highs_binary_path_milp_v1",
            solver_diagnostics=solver_diagnostics,
        )
    solution = getattr(result, "x", None)
    if solution is None:
        return _FeasibilityLifelineResult(
            node=None,
            explored_state_count=solver_diagnostics["mip_node_count"],
            memoized_failed_state_count=0,
            state_capacity_exhausted=True,
            root_candidate_diagnostics=(
                {
                    "search_outcome": "milp_limit_without_witness",
                    "solver_status": status,
                },
            ),
            admissible_pruning_diagnostics={
                "contract": "sparse_binary_path_milp_exact_hard_constraints_v1",
                "constraint_infeasibility_proven": False,
            },
            strategy="scipy_highs_binary_path_milp_v1",
            solver_diagnostics=solver_diagnostics,
        )
    path = tuple(
        int(
            np.argmax(
                solution[
                    position * source_count : (position + 1) * source_count
                ]
            )
        )
        for position in range(span_count)
    )
    counts = tuple(path.count(source_index) for source_index in range(source_count))
    duration_totals = tuple(
        sum(
            float(durations[position])
            for position, selected_source in enumerate(path)
            if selected_source == source_index
        )
        for source_index in range(source_count)
    )
    severe_count = 0
    audit_failures: list[str] = []
    if counts != tuple(source_quotas):
        audit_failures.append("source_quota_mismatch")
    if any(
        not candidate_validity[position][source_index]
        for position, source_index in enumerate(path)
    ):
        audit_failures.append("content_invalid_choice")
    if any(left == right for left, right in zip(path, path[1:])):
        audit_failures.append("adjacent_source_repeat")
    allowed_edges = tuple(
        {
            (edge_specs[edge_index][1], edge_specs[edge_index][2])
            for edge_index in edge_indices
        }
        for edge_indices in edges_by_boundary
    )
    if any(
        (path[boundary], path[boundary + 1]) not in allowed_edges[boundary]
        for boundary in range(span_count - 1)
    ):
        audit_failures.append("unmodeled_boundary_edge")
    if severe_transition_budget_mask is not None:
        severe_count = sum(
            bool(
                severe_transition_budget_mask[boundary][path[boundary]][
                    path[boundary + 1]
                ]
            )
            for boundary in range(span_count - 1)
        )
        if severe_count > int(maximum_severe_transition_count):
            audit_failures.append("maximum_severe_transition_count_exceeded")
        if (
            minimum_severe_transition_count is not None
            and severe_count < minimum_severe_transition_count
        ):
            audit_failures.append("minimum_severe_transition_count_missed")
    if any(
        abs(actual - target) / target
        > config.maximum_source_duration_relative_deviation + 1e-7
        for actual, target in zip(duration_totals, duration_targets)
    ):
        audit_failures.append("duration_balance_threshold_missed")
    solver_diagnostics["postsolve_path_audit_passed"] = not audit_failures
    solver_diagnostics["postsolve_path_audit_failures"] = audit_failures
    solver_diagnostics["selected_severe_transition_count"] = severe_count
    if audit_failures:
        return _FeasibilityLifelineResult(
            node=None,
            explored_state_count=solver_diagnostics["mip_node_count"],
            memoized_failed_state_count=0,
            state_capacity_exhausted=True,
            root_candidate_diagnostics=(
                {
                    "search_outcome": "milp_witness_failed_independent_audit",
                    "audit_failures": audit_failures,
                },
            ),
            admissible_pruning_diagnostics={
                "contract": "sparse_binary_path_milp_exact_hard_constraints_v1",
                "constraint_infeasibility_proven": False,
            },
            strategy="scipy_highs_binary_path_milp_v1",
            solver_diagnostics=solver_diagnostics,
        )
    cost, verified_totals = _path_optimizer_cost(
        path,
        matrices=matrices,
        durations=durations,
        duration_targets=duration_targets,
        config=config,
    )
    return _FeasibilityLifelineResult(
        node=_BeamNode(cost, path, verified_totals),
        explored_state_count=solver_diagnostics["mip_node_count"],
        memoized_failed_state_count=0,
        state_capacity_exhausted=False,
        root_candidate_diagnostics=(
            {
                "search_outcome": "milp_witness_found_and_audited",
                "final_segment_index": span_count - 1,
                "selected_severe_transition_count": severe_count,
            },
        ),
        admissible_pruning_diagnostics={
            "contract": "sparse_binary_path_milp_exact_hard_constraints_v1",
            "constraint_infeasibility_proven": False,
        },
        strategy="scipy_highs_binary_path_milp_v1",
        solver_diagnostics=solver_diagnostics,
    )


def _exact_feasibility_lifeline(
    matrices: Sequence[Sequence[Sequence[DirectBoundaryMismatch]]],
    *,
    source_quotas: Sequence[int],
    source_order: Sequence[int],
    candidate_validity: Sequence[Sequence[bool]],
    durations: Sequence[float],
    duration_targets: Sequence[float],
    state_limit: int,
    config: DirectMemberVisualConfig,
    severe_transition_budget_mask: (
        Sequence[Sequence[Sequence[bool]]] | None
    ) = None,
    minimum_severe_transition_count: int | None = None,
    maximum_severe_transition_count: int | None = None,
) -> _FeasibilityLifelineResult:
    """Find one exact hard-feasible path without lossy beam pruning.

    Search runs backward so a restrictive final boundary constrains its
    predecessor immediately.  The memo key retains exact duration totals;
    unlike the quality beam, it applies no rounded duration signature.  A
    capacity hit is reported as search exhaustion and never mislabeled as
    constraint infeasibility.  When both severe bounds are equal, the search
    requires exactly that many original severe edges; iterative-deepening
    callers may do this only after every lower count has been proved hard
    infeasible.
    """

    if (severe_transition_budget_mask is None) != (
        maximum_severe_transition_count is None
    ):
        raise ValueError(
            "severe transition budget mask and maximum must be supplied together"
        )
    if maximum_severe_transition_count is not None and (
        isinstance(maximum_severe_transition_count, bool)
        or not isinstance(maximum_severe_transition_count, int)
        or maximum_severe_transition_count < 0
    ):
        raise ValueError("maximum severe transition count must be non-negative")
    if minimum_severe_transition_count is not None and (
        severe_transition_budget_mask is None
        or isinstance(minimum_severe_transition_count, bool)
        or not isinstance(minimum_severe_transition_count, int)
        or minimum_severe_transition_count < 0
        or minimum_severe_transition_count
        > int(maximum_severe_transition_count)
    ):
        raise ValueError(
            "minimum severe transition count requires a valid bounded budget"
        )

    span_count = len(candidate_validity)
    source_count = len(source_quotas)
    severe_budget_mask = (
        None
        if severe_transition_budget_mask is None
        else tuple(
            tuple(tuple(bool(value) for value in row) for row in matrix)
            for matrix in severe_transition_budget_mask
        )
    )
    if severe_budget_mask is not None and (
        len(severe_budget_mask) != len(matrices)
        or any(len(matrix) != source_count for matrix in severe_budget_mask)
        or any(
            len(row) != source_count
            for matrix in severe_budget_mask
            for row in matrix
        )
    ):
        raise ValueError("severe transition budget mask shape is invalid")
    if _prefer_direct_member_milp(candidate_validity):
        milp_result = _milp_feasibility_lifeline(
            matrices,
            source_quotas=source_quotas,
            source_order=source_order,
            candidate_validity=candidate_validity,
            durations=durations,
            duration_targets=duration_targets,
            state_limit=state_limit,
            config=config,
            severe_transition_budget_mask=severe_budget_mask,
            minimum_severe_transition_count=minimum_severe_transition_count,
            maximum_severe_transition_count=maximum_severe_transition_count,
        )
        if milp_result is not None:
            return milp_result
    prefix_valid_counts: list[tuple[int, ...]] = [(0,) * source_count]
    for position in range(span_count):
        previous = prefix_valid_counts[-1]
        prefix_valid_counts.append(
            tuple(
                previous[source_index]
                + int(bool(candidate_validity[position][source_index]))
                for source_index in range(source_count)
            )
        )

    def _merge_duration_bound(
        values: dict[int, tuple[float, float]],
        count: int,
        minimum_value: float,
        maximum_value: float,
    ) -> None:
        previous = values.get(count)
        if previous is None:
            values[count] = (minimum_value, maximum_value)
        else:
            values[count] = (
                min(previous[0], minimum_value),
                max(previous[1], maximum_value),
            )

    # For each source and prefix, compute the exact minimum/maximum duration
    # obtainable by choosing N content-valid, non-adjacent positions.  The
    # second map additionally requires the final prefix position to remain
    # unselected, which is mandatory when the already-fixed suffix starts with
    # the same source.  These are single-source relaxations of the complete
    # assignment problem: every real completion is represented, while other
    # sources and transition edges are deliberately ignored.  Their bounds are
    # therefore admissible necessary conditions and use the original floats.
    source_prefix_duration_bounds: list[
        list[
            tuple[
                dict[int, tuple[float, float]],
                dict[int, tuple[float, float]],
            ]
        ]
    ] = []
    for source_index in range(source_count):
        unselected: dict[int, tuple[float, float]] = {0: (0.0, 0.0)}
        selected: dict[int, tuple[float, float]] = {}
        levels = [(dict(unselected), dict(unselected))]
        for position, duration in enumerate(durations):
            next_unselected: dict[int, tuple[float, float]] = {}
            for count, (minimum_value, maximum_value) in (
                list(unselected.items()) + list(selected.items())
            ):
                _merge_duration_bound(
                    next_unselected,
                    count,
                    minimum_value,
                    maximum_value,
                )
            next_selected: dict[int, tuple[float, float]] = {}
            if candidate_validity[position][source_index]:
                for count, (minimum_value, maximum_value) in unselected.items():
                    _merge_duration_bound(
                        next_selected,
                        count + 1,
                        minimum_value + float(duration),
                        maximum_value + float(duration),
                    )
            unselected = next_unselected
            selected = next_selected
            unrestricted: dict[int, tuple[float, float]] = {}
            for count, (minimum_value, maximum_value) in (
                list(unselected.items()) + list(selected.items())
            ):
                _merge_duration_bound(
                    unrestricted,
                    count,
                    minimum_value,
                    maximum_value,
                )
            levels.append((unrestricted, dict(unselected)))
        source_prefix_duration_bounds.append(levels)
    lower_duration = tuple(
        target * (1.0 - config.maximum_source_duration_relative_deviation)
        for target in duration_targets
    )
    upper_duration = tuple(
        target * (1.0 + config.maximum_source_duration_relative_deviation)
        for target in duration_targets
    )
    order_rank = {source_index: rank for rank, source_index in enumerate(source_order)}
    failed_states: set[
        tuple[int, tuple[int, ...], int, tuple[float, ...], int]
    ] = set()
    explored_state_count = 0
    capacity_exhausted = False
    prefix_prune_reason_counts: Counter[str] = Counter()

    # This DP is an exact relaxation of the remaining prefix: it preserves
    # candidate validity, quotas, no-repeat adjacency, and the original severe
    # edge budget, while intentionally omitting duration constraints.  All
    # exact-float duration states sharing one discrete prefix state can reuse
    # its answer.  ``False`` is therefore an admissible prune.  If this helper
    # reaches its own finite state cap, it returns ``None`` and the full search
    # continues unchanged; capacity can never be mistaken for infeasibility.
    structural_prefix_memo: dict[
        tuple[int, tuple[int, ...], int, int, int], bool
    ] = {}
    structural_prefix_state_limit = state_limit
    structural_prefix_state_count = 0
    structural_prefix_memo_hit_count = 0
    structural_prefix_query_count = 0
    structural_prefix_infeasible_pruned_state_count = 0
    structural_prefix_state_capacity_exhausted = False

    def _structural_edge_severe_cost(
        boundary_index: int,
        outgoing_source: int,
        incoming_source: int,
    ) -> int | None:
        if severe_budget_mask is None:
            return (
                None
                if matrices[boundary_index][outgoing_source][
                    incoming_source
                ].severe_flags
                else 0
            )
        return int(
            severe_budget_mask[boundary_index][outgoing_source][
                incoming_source
            ]
        )

    # Precompute two uncapped, finite relaxations of every backward prefix
    # state.  Both retain content validity, no-repeat adjacency, edge
    # direction, and the original severe-edge cost exactly while omitting
    # duration and (for the subset envelope) individual source quotas.
    #
    # ``layered_severe_reachability[position][current]`` is a bitset of every
    # exact severe count attainable by a valid prefix of ``position`` slots
    # ending immediately before ``current``.  The subset table additionally
    # records, for every source subset and exact severe count, the maximum
    # number of prefix slots that can belong to that subset.  A real completion
    # must appear in both relaxations, so a missing count or insufficient
    # subset capacity is an admissible proof of infeasibility.
    layered_requested_maximum_severe_count = (
        0
        if maximum_severe_transition_count is None
        else int(maximum_severe_transition_count)
    )
    layered_severe_edge_count_upper_bound = max(0, span_count - 1)
    layered_maximum_severe_count = min(
        layered_requested_maximum_severe_count,
        layered_severe_edge_count_upper_bound,
    )
    layered_severe_bit_limit = (
        1 << (layered_maximum_severe_count + 1)
    ) - 1
    layered_severe_reachability: list[tuple[int, ...]] = [
        (1,) * source_count
    ]
    layered_subset_source_count_limit = 12
    layered_subset_projected_cell_limit = 1_000_000
    layered_subset_projected_work_limit = 10_000_000
    layered_subset_projected_subset_count = (
        1 << source_count
        if source_count <= layered_subset_source_count_limit
        else None
    )
    layered_subset_projected_cell_count = (
        span_count
        * source_count
        * (layered_maximum_severe_count + 1)
        * layered_subset_projected_subset_count
        if layered_subset_projected_subset_count is not None
        else None
    )
    layered_subset_projected_work_count = (
        max(0, span_count - 1)
        * source_count
        * max(0, source_count - 1)
        * (layered_maximum_severe_count + 1)
        * layered_subset_projected_subset_count
        if layered_subset_projected_subset_count is not None
        else None
    )
    if source_count > layered_subset_source_count_limit:
        layered_subset_disabled_reason = "source_count_limit_exceeded"
    elif (
        int(layered_subset_projected_cell_count)
        > layered_subset_projected_cell_limit
    ):
        layered_subset_disabled_reason = "projected_cell_limit_exceeded"
    elif (
        int(layered_subset_projected_work_count)
        > layered_subset_projected_work_limit
    ):
        layered_subset_disabled_reason = "projected_work_limit_exceeded"
    else:
        layered_subset_disabled_reason = None
    layered_subset_quota_enabled = layered_subset_disabled_reason is None
    layered_subset_count = (
        int(layered_subset_projected_subset_count)
        if layered_subset_quota_enabled
        else 0
    )
    layered_subset_membership: tuple[tuple[int, ...], ...] = (
        tuple(
            tuple(
                int(bool(subset_mask & (1 << source_index)))
                for subset_mask in range(layered_subset_count)
            )
            for source_index in range(source_count)
        )
        if layered_subset_quota_enabled
        else ()
    )
    layered_subset_quota_maxima: list[
        tuple[tuple[tuple[int, ...], ...], ...]
    ] = []
    if layered_subset_quota_enabled:
        unreachable_subset_row = (-1,) * layered_subset_count
        zero_subset_row = (0,) * layered_subset_count
        layered_subset_quota_maxima.append(
            tuple(
                tuple(
                    zero_subset_row
                    if severe_count == 0
                    else unreachable_subset_row
                    for severe_count in range(
                        layered_maximum_severe_count + 1
                    )
                )
                for _current_source in range(source_count)
            )
        )

    # A search state never has more than ``span_count - 1`` unassigned prefix
    # positions because its current source already owns the following span.
    for position in range(1, span_count):
        preceding_position = position - 1
        previous_reachability = layered_severe_reachability[-1]
        next_reachability: list[int] = []
        previous_subset_level = (
            layered_subset_quota_maxima[-1]
            if layered_subset_quota_enabled
            else ()
        )
        next_subset_level: list[tuple[tuple[int, ...], ...]] = []
        for current_source in range(source_count):
            exact_reachability_bits = 0
            subset_rows = (
                [
                    [-1] * layered_subset_count
                    for _severe_count in range(
                        layered_maximum_severe_count + 1
                    )
                ]
                if layered_subset_quota_enabled
                else []
            )
            for preceding_source in range(source_count):
                if (
                    preceding_source == current_source
                    or not candidate_validity[preceding_position][
                        preceding_source
                    ]
                ):
                    continue
                edge_cost = _structural_edge_severe_cost(
                    preceding_position,
                    preceding_source,
                    current_source,
                )
                if edge_cost is None:
                    continue
                exact_reachability_bits |= (
                    previous_reachability[preceding_source] << edge_cost
                ) & layered_severe_bit_limit
                if not layered_subset_quota_enabled:
                    continue
                membership = layered_subset_membership[preceding_source]
                preceding_rows = previous_subset_level[preceding_source]
                for preceding_severe_count in range(
                    layered_maximum_severe_count - edge_cost + 1
                ):
                    preceding_row = preceding_rows[
                        preceding_severe_count
                    ]
                    if preceding_row[0] < 0:
                        continue
                    target_row = subset_rows[
                        preceding_severe_count + edge_cost
                    ]
                    for subset_mask in range(layered_subset_count):
                        candidate_count = (
                            preceding_row[subset_mask]
                            + membership[subset_mask]
                        )
                        if candidate_count > target_row[subset_mask]:
                            target_row[subset_mask] = candidate_count
            next_reachability.append(exact_reachability_bits)
            if layered_subset_quota_enabled:
                next_subset_level.append(
                    tuple(tuple(row) for row in subset_rows)
                )
        layered_severe_reachability.append(tuple(next_reachability))
        if layered_subset_quota_enabled:
            layered_subset_quota_maxima.append(tuple(next_subset_level))

    layered_severe_reachability_query_count = 0
    layered_severe_reachability_pruned_state_count = 0
    layered_subset_quota_query_count = 0
    layered_subset_quota_pruned_state_count = 0
    layered_subset_quota_pruned_mask_counts: Counter[int] = Counter()

    def _structural_prefix_feasible(
        position: int,
        deficits: tuple[int, ...],
        current_source: int,
        remaining_minimum_severe_count: int,
        remaining_maximum_severe_count: int,
    ) -> bool | None:
        nonlocal structural_prefix_state_count
        nonlocal structural_prefix_memo_hit_count
        nonlocal structural_prefix_state_capacity_exhausted
        key = (
            position,
            deficits,
            current_source,
            remaining_minimum_severe_count,
            remaining_maximum_severe_count,
        )
        cached = structural_prefix_memo.get(key)
        if cached is not None:
            structural_prefix_memo_hit_count += 1
            return cached
        if structural_prefix_state_capacity_exhausted:
            return None
        if structural_prefix_state_count >= structural_prefix_state_limit:
            structural_prefix_state_capacity_exhausted = True
            return None
        structural_prefix_state_count += 1
        if (
            remaining_maximum_severe_count < 0
            or remaining_minimum_severe_count
            > remaining_maximum_severe_count
            or any(value < 0 for value in deficits)
            or sum(deficits) != position
        ):
            structural_prefix_memo[key] = False
            return False
        if position == 0:
            feasible = remaining_minimum_severe_count <= 0
            structural_prefix_memo[key] = feasible
            return feasible
        for source_index, deficit in enumerate(deficits):
            source_bounds = source_prefix_duration_bounds[source_index][
                position
            ][int(source_index == current_source)]
            if deficit not in source_bounds:
                structural_prefix_memo[key] = False
                return False

        preceding_position = position - 1
        unknown_child = False
        for source_index in source_order:
            if (
                source_index == current_source
                or deficits[source_index] <= 0
                or not candidate_validity[preceding_position][source_index]
            ):
                continue
            edge_cost = _structural_edge_severe_cost(
                preceding_position,
                source_index,
                current_source,
            )
            if (
                edge_cost is None
                or edge_cost > remaining_maximum_severe_count
            ):
                continue
            next_deficits = list(deficits)
            next_deficits[source_index] -= 1
            child = _structural_prefix_feasible(
                preceding_position,
                tuple(next_deficits),
                source_index,
                max(0, remaining_minimum_severe_count - edge_cost),
                remaining_maximum_severe_count - edge_cost,
            )
            if child is True:
                structural_prefix_memo[key] = True
                return True
            if child is None:
                unknown_child = True
        if unknown_child:
            return None
        structural_prefix_memo[key] = False
        return False

    def _prefix_prune_reasons(
        position: int,
        counts: Sequence[int],
        current_source: int,
        duration_totals: Sequence[float],
    ) -> tuple[dict[str, Any], ...]:
        reasons: list[dict[str, Any]] = []
        quota_deficit_sum = sum(
            source_quotas[index] - counts[index]
            for index in range(source_count)
        )
        if quota_deficit_sum != position:
            reasons.append(
                {
                    "reason": "quota_deficit_sum_mismatch",
                    "required_prefix_assignment_count": position,
                    "quota_deficit_sum": quota_deficit_sum,
                }
            )
        for source_index in range(source_count):
            deficit = source_quotas[source_index] - counts[source_index]
            if deficit < 0:
                reasons.append(
                    {
                        "reason": "source_quota_exceeded",
                        "eligible_source_index": source_index,
                        "quota": int(source_quotas[source_index]),
                        "assigned_suffix_count": int(counts[source_index]),
                        "quota_deficit": int(deficit),
                    }
                )
                continue
            prefix_valid_slot_count = prefix_valid_counts[position][source_index]
            if deficit > prefix_valid_slot_count:
                reasons.append(
                    {
                        "reason": "prefix_valid_slot_shortfall",
                        "eligible_source_index": source_index,
                        "quota_deficit": int(deficit),
                        "prefix_valid_slot_count": int(prefix_valid_slot_count),
                        "shortfall": int(deficit - prefix_valid_slot_count),
                        "prefix_position_count": position,
                    }
                )
            maximum_slots = (
                position // 2
                if source_index == current_source
                else (position + 1) // 2
            )
            if deficit > maximum_slots:
                reasons.append(
                    {
                        "reason": "prefix_adjacency_slot_shortfall",
                        "eligible_source_index": source_index,
                        "quota_deficit": int(deficit),
                        "maximum_nonadjacent_prefix_slots": int(maximum_slots),
                        "shortfall": int(deficit - maximum_slots),
                        "prefix_position_count": position,
                        "current_suffix_source_index": current_source,
                    }
                )
            # If the suffix starts with this source, position ``position - 1``
            # cannot use it.  Otherwise either end-state is allowed.  Absence
            # of the exact deficit proves a content-valid/non-adjacent slot
            # shortfall for this source alone.
            source_bounds = source_prefix_duration_bounds[source_index][
                position
            ][int(source_index == current_source)]
            duration_bound = source_bounds.get(deficit)
            if duration_bound is None:
                # Preserve the established, more specific diagnostics when a
                # plain validity or plain adjacency bound already proves the
                # shortfall.  This combined reason is emitted only when the
                # interaction of those two constraints is the new proof.
                if (
                    deficit <= prefix_valid_slot_count
                    and deficit <= maximum_slots
                ):
                    reasons.append(
                        {
                            "reason": (
                                "prefix_valid_nonadjacent_slot_shortfall"
                            ),
                            "eligible_source_index": source_index,
                            "quota_deficit": int(deficit),
                            "maximum_valid_nonadjacent_prefix_slots": max(
                                source_bounds,
                                default=0,
                            ),
                            "prefix_position_count": position,
                            "current_suffix_source_index": current_source,
                        }
                    )
                continue
            minimum_addition, maximum_addition = duration_bound
            projected_minimum = duration_totals[source_index] + minimum_addition
            projected_maximum = duration_totals[source_index] + maximum_addition
            if projected_minimum > upper_duration[source_index] + 1e-9:
                reasons.append(
                    {
                        "reason": "duration_upper_bound_exceeded",
                        "eligible_source_index": source_index,
                        "quota_deficit": int(deficit),
                        "assigned_suffix_duration_seconds": (
                            duration_totals[source_index]
                        ),
                        "minimum_prefix_addition_seconds": minimum_addition,
                        "projected_minimum_duration_seconds": projected_minimum,
                        "allowed_upper_duration_seconds": upper_duration[
                            source_index
                        ],
                    }
                )
            if projected_maximum < lower_duration[source_index] - 1e-9:
                reasons.append(
                    {
                        "reason": "duration_lower_bound_unreachable",
                        "eligible_source_index": source_index,
                        "quota_deficit": int(deficit),
                        "assigned_suffix_duration_seconds": (
                            duration_totals[source_index]
                        ),
                        "maximum_prefix_addition_seconds": maximum_addition,
                        "projected_maximum_duration_seconds": projected_maximum,
                        "required_lower_duration_seconds": lower_duration[
                            source_index
                        ],
                    }
                )
        return tuple(reasons)

    def _search(
        position: int,
        counts: tuple[int, ...],
        current_source: int,
        duration_totals: tuple[float, ...],
        suffix_path: tuple[int, ...],
        severe_transition_count: int,
    ) -> _BeamNode | None:
        nonlocal explored_state_count, capacity_exhausted
        nonlocal layered_severe_reachability_query_count
        nonlocal layered_severe_reachability_pruned_state_count
        nonlocal layered_subset_quota_query_count
        nonlocal layered_subset_quota_pruned_state_count
        nonlocal structural_prefix_query_count
        nonlocal structural_prefix_infeasible_pruned_state_count
        if capacity_exhausted:
            return None
        key = (
            position,
            counts,
            current_source,
            duration_totals,
            severe_transition_count,
        )
        if key in failed_states:
            return None
        if explored_state_count >= state_limit:
            capacity_exhausted = True
            return None
        explored_state_count += 1
        prefix_prune_reasons = _prefix_prune_reasons(
            position,
            counts,
            current_source,
            duration_totals,
        )
        if prefix_prune_reasons:
            prefix_prune_reason_counts.update(
                str(item["reason"]) for item in prefix_prune_reasons
            )
            failed_states.add(key)
            return None
        if position == 0:
            if (
                minimum_severe_transition_count is not None
                and severe_transition_count
                < minimum_severe_transition_count
            ):
                failed_states.add(key)
                return None
            cost, verified_totals = _path_optimizer_cost(
                suffix_path,
                matrices=matrices,
                durations=durations,
                duration_targets=duration_targets,
                config=config,
            )
            return _BeamNode(cost, suffix_path, verified_totals)

        remaining_maximum_severe_count = (
            0
            if severe_budget_mask is None
            else min(
                position,
                layered_maximum_severe_count - severe_transition_count,
            )
        )
        remaining_minimum_severe_count = (
            0
            if minimum_severe_transition_count is None
            else max(
                0,
                minimum_severe_transition_count
                - severe_transition_count,
            )
        )
        quota_deficits = tuple(
            int(source_quotas[index]) - counts[index]
            for index in range(source_count)
        )
        layered_severe_reachability_query_count += 1
        reachable_severe_bits = layered_severe_reachability[position][
            current_source
        ]
        reachable_remaining_severe_counts = tuple(
            severe_count
            for severe_count in range(
                max(0, remaining_minimum_severe_count),
                min(
                    layered_maximum_severe_count,
                    remaining_maximum_severe_count,
                )
                + 1,
            )
            if reachable_severe_bits & (1 << severe_count)
        )
        if not reachable_remaining_severe_counts:
            layered_severe_reachability_pruned_state_count += 1
            failed_states.add(key)
            return None

        if layered_subset_quota_enabled:
            layered_subset_quota_query_count += 1
            subset_rows = layered_subset_quota_maxima[position][
                current_source
            ]
            failing_subset_mask: int | None = None
            for subset_mask in range(1, layered_subset_count):
                required_subset_count = sum(
                    quota_deficits[source_index]
                    for source_index in range(source_count)
                    if subset_mask & (1 << source_index)
                )
                maximum_subset_count = max(
                    subset_rows[severe_count][subset_mask]
                    for severe_count in reachable_remaining_severe_counts
                )
                if required_subset_count > maximum_subset_count:
                    failing_subset_mask = subset_mask
                    break
            if failing_subset_mask is not None:
                layered_subset_quota_pruned_state_count += 1
                layered_subset_quota_pruned_mask_counts[
                    failing_subset_mask
                ] += 1
                failed_states.add(key)
                return None

        structural_prefix_query_count += 1
        structural_feasible = _structural_prefix_feasible(
            position,
            quota_deficits,
            current_source,
            remaining_minimum_severe_count,
            remaining_maximum_severe_count,
        )
        if structural_feasible is False:
            structural_prefix_infeasible_pruned_state_count += 1
            failed_states.add(key)
            return None

        preceding_position = position - 1
        candidates = [
            source_index
            for source_index in source_order
            if source_index != current_source
            and candidate_validity[preceding_position][source_index]
            and counts[source_index] < source_quotas[source_index]
            and (
                (
                    not matrices[preceding_position][source_index][
                        current_source
                    ].severe_flags
                )
                if severe_budget_mask is None
                else severe_transition_count
                + int(
                    severe_budget_mask[preceding_position][source_index][
                        current_source
                    ]
                )
                <= int(maximum_severe_transition_count)
            )
        ]
        candidates.sort(
            key=lambda source_index: (
                (
                    int(
                        severe_budget_mask[preceding_position][source_index][
                            current_source
                        ]
                    )
                    if severe_budget_mask is not None
                    else 0
                ),
                matrices[preceding_position][source_index][current_source].cost,
                order_rank[source_index],
            )
        )
        for source_index in candidates:
            updated_counts = list(counts)
            updated_counts[source_index] += 1
            updated_durations = list(duration_totals)
            updated_durations[source_index] += durations[preceding_position]
            updated_severe_transition_count = severe_transition_count + (
                int(
                    severe_budget_mask[preceding_position][source_index][
                        current_source
                    ]
                )
                if severe_budget_mask is not None
                else 0
            )
            result = _search(
                preceding_position,
                tuple(updated_counts),
                source_index,
                tuple(updated_durations),
                (source_index,) + suffix_path,
                updated_severe_transition_count,
            )
            if result is not None:
                return result
            if capacity_exhausted:
                return None
        failed_states.add(key)
        return None

    node: _BeamNode | None = None
    final_position = span_count - 1
    final_candidates = [
        source_index
        for source_index in source_order
        if candidate_validity[final_position][source_index]
        and source_quotas[source_index] > 0
    ]
    root_candidate_diagnostics: list[dict[str, Any]] = []
    for source_index in final_candidates:
        counts = [0] * source_count
        counts[source_index] = 1
        duration_totals = [0.0] * source_count
        duration_totals[source_index] = durations[final_position]
        root_prune_reasons = _prefix_prune_reasons(
            final_position,
            counts,
            source_index,
            duration_totals,
        )
        predecessor_diagnostics: list[dict[str, Any]] = []
        if final_position > 0:
            for predecessor_source_index in source_order:
                rejection_reasons: list[str] = []
                if predecessor_source_index == source_index:
                    rejection_reasons.append("adjacent_source_repeat")
                if not candidate_validity[final_position - 1][
                    predecessor_source_index
                ]:
                    rejection_reasons.append("preceding_content_invalid")
                if (
                    counts[predecessor_source_index]
                    >= source_quotas[predecessor_source_index]
                ):
                    rejection_reasons.append("source_quota_already_full")
                severe_flags = list(
                    matrices[final_position - 1][predecessor_source_index][
                        source_index
                    ].severe_flags
                )
                severe_budget_cost = (
                    int(
                        severe_budget_mask[final_position - 1][
                            predecessor_source_index
                        ][source_index]
                    )
                    if severe_budget_mask is not None
                    else 0
                )
                if severe_flags and severe_budget_mask is None:
                    rejection_reasons.append("severe_boundary_mismatch")
                if (
                    severe_budget_mask is not None
                    and severe_budget_cost
                    > int(maximum_severe_transition_count)
                ):
                    rejection_reasons.append(
                        "severe_transition_budget_exceeded"
                    )
                predecessor_diagnostics.append(
                    {
                        "eligible_source_index": predecessor_source_index,
                        "eligible": not rejection_reasons,
                        "rejection_reasons": rejection_reasons,
                        "severe_flags": severe_flags,
                        "severe_transition_budget_mode": (
                            severe_budget_mask is not None
                        ),
                        "severe_budget_cost": severe_budget_cost,
                        "maximum_severe_transition_count": (
                            maximum_severe_transition_count
                        ),
                    }
                )
        eligible_predecessors = [
            item["eligible_source_index"]
            for item in predecessor_diagnostics
            if item["eligible"]
        ]
        node = _search(
            final_position,
            tuple(counts),
            source_index,
            tuple(duration_totals),
            (source_index,),
            0,
        )
        if node is not None:
            search_outcome = "witness_found"
        elif capacity_exhausted:
            search_outcome = "state_capacity_exhausted"
        elif root_prune_reasons:
            search_outcome = "root_prefix_pruned"
        elif not eligible_predecessors:
            search_outcome = "root_predecessor_exhausted"
        else:
            search_outcome = "subtree_infeasible"
        root_candidate_diagnostics.append(
            {
                "final_candidate_source_index": source_index,
                "final_segment_index": final_position,
                "prefix_position_count": final_position,
                "root_prefix_feasible": not root_prune_reasons,
                "search_outcome": search_outcome,
                "prune_reasons": list(root_prune_reasons),
                "eligible_predecessor_source_indices": eligible_predecessors,
                "predecessor_diagnostics": predecessor_diagnostics,
            }
        )
        if node is not None or capacity_exhausted:
            break
    return _FeasibilityLifelineResult(
        node=node,
        explored_state_count=explored_state_count,
        memoized_failed_state_count=len(failed_states),
        state_capacity_exhausted=capacity_exhausted,
        root_candidate_diagnostics=tuple(root_candidate_diagnostics),
        admissible_pruning_diagnostics={
            "contract": (
                "exact_single_source_duration_bounds_plus_"
                "layered_severe_subset_envelope_plus_"
                "duration_independent_structural_prefix_dp_v2"
            ),
            "single_source_duration_bound_contract": (
                "candidate_valid_nonadjacent_exact_float_min_max_v1"
            ),
            "duration_state_quantization_applied": False,
            "prefix_prune_reason_counts": dict(
                sorted(prefix_prune_reason_counts.items())
            ),
            "layered_severe_reachability_contract": (
                "candidate_valid_no_repeat_directed_original_severe_"
                "exact_count_bitset_v1"
            ),
            "layered_severe_reachability_query_count": (
                layered_severe_reachability_query_count
            ),
            "layered_severe_reachability_pruned_state_count": (
                layered_severe_reachability_pruned_state_count
            ),
            "layered_severe_reachability_precomputed_cell_count": (
                len(layered_severe_reachability) * source_count
            ),
            "layered_requested_maximum_severe_transition_count": (
                maximum_severe_transition_count
            ),
            "layered_effective_maximum_severe_transition_count": (
                layered_maximum_severe_count
            ),
            "layered_severe_edge_count_upper_bound": (
                layered_severe_edge_count_upper_bound
            ),
            "layered_severe_count_clamp_applied": bool(
                layered_requested_maximum_severe_count
                > layered_maximum_severe_count
            ),
            "layered_subset_quota_contract": (
                "candidate_valid_no_repeat_directed_original_severe_"
                "exact_count_all_source_subsets_max_envelope_v1"
            ),
            "layered_subset_quota_enabled": layered_subset_quota_enabled,
            "layered_subset_quota_disabled_reason": (
                layered_subset_disabled_reason
            ),
            "layered_subset_quota_source_count_limit": (
                layered_subset_source_count_limit
            ),
            "layered_subset_quota_projected_cell_limit": (
                layered_subset_projected_cell_limit
            ),
            "layered_subset_quota_projected_work_limit": (
                layered_subset_projected_work_limit
            ),
            "layered_subset_quota_projected_cell_count": (
                layered_subset_projected_cell_count
            ),
            "layered_subset_quota_projected_work_count": (
                layered_subset_projected_work_count
            ),
            "layered_subset_quota_subset_count": layered_subset_count,
            "layered_subset_quota_query_count": (
                layered_subset_quota_query_count
            ),
            "layered_subset_quota_pruned_state_count": (
                layered_subset_quota_pruned_state_count
            ),
            "layered_subset_quota_pruned_mask_counts": {
                str(mask): count
                for mask, count in sorted(
                    layered_subset_quota_pruned_mask_counts.items()
                )
            },
            "layered_subset_quota_precomputed_cell_count": (
                len(layered_subset_quota_maxima)
                * source_count
                * (layered_maximum_severe_count + 1)
                * layered_subset_count
            ),
            "layered_relaxation_state_capacity_applied": False,
            "layered_subset_quota_fallback_contract": (
                "disabled_envelope_preserves_layered_reachability_"
                "structural_prefix_and_full_exact_search_v1"
            ),
            "structural_prefix_contract": (
                "candidate_valid_quota_no_repeat_original_severe_budget_"
                "duration_relaxation_v1"
            ),
            "structural_prefix_query_count": (
                structural_prefix_query_count
            ),
            "structural_prefix_state_count": structural_prefix_state_count,
            "structural_prefix_memoized_state_count": len(
                structural_prefix_memo
            ),
            "structural_prefix_memo_hit_count": (
                structural_prefix_memo_hit_count
            ),
            "structural_prefix_infeasible_pruned_state_count": (
                structural_prefix_infeasible_pruned_state_count
            ),
            "structural_prefix_state_limit": structural_prefix_state_limit,
            "structural_prefix_state_capacity_exhausted": (
                structural_prefix_state_capacity_exhausted
            ),
            "severe_transition_count_requirement": (
                "hard_ineligible"
                if severe_budget_mask is None
                else "exact"
                if minimum_severe_transition_count
                == maximum_severe_transition_count
                else "bounded_range"
            ),
            "minimum_severe_transition_count": (
                minimum_severe_transition_count
            ),
            "maximum_severe_transition_count": (
                maximum_severe_transition_count
            ),
            "structural_capacity_effect": (
                "unknown_fallback_to_full_exact_search"
            ),
            "hard_thresholds_changed": False,
        },
    )


def _canonical_diagnostic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _exact_hard_constraint_payload(
    matrices: Sequence[Sequence[Sequence[DirectBoundaryMismatch]]],
    *,
    source_quotas: Sequence[int],
    source_order: Sequence[int],
    candidate_validity: Sequence[Sequence[bool]],
    durations: Sequence[float],
    duration_targets: Sequence[float],
    maximum_source_duration_relative_deviation: float,
) -> dict[str, Any]:
    """Serialize only the exact constraints that decide hard feasibility."""

    return {
        "contract": "direct_member_exact_hard_constraint_input_v1",
        "span_count": len(candidate_validity),
        "source_count": len(source_quotas),
        "source_quotas": [int(value) for value in source_quotas],
        "source_order": [int(value) for value in source_order],
        "candidate_validity": [
            [bool(value) for value in row] for row in candidate_validity
        ],
        # Float hex strings bind the exact binary values used by the search;
        # decimal formatting must not silently merge nearby duration states.
        "duration_seconds_float_hex": [
            float(value).hex() for value in durations
        ],
        "duration_target_seconds_float_hex": [
            float(value).hex() for value in duration_targets
        ],
        "maximum_source_duration_relative_deviation_float_hex": (
            float(maximum_source_duration_relative_deviation).hex()
        ),
        "adjacent_severe_flags": [
            [
                [
                    [str(flag) for flag in mismatch.severe_flags]
                    for mismatch in outgoing_row
                ]
                for outgoing_row in matrix
            ]
            for matrix in matrices
        ],
        "matrix_boundary_count": len(matrices),
        "no_adjacent_source_repeat": True,
        "duration_state_quantization_applied": False,
    }


def _severe_transition_counterfactual_diagnostic(
    matrices: Sequence[Sequence[Sequence[DirectBoundaryMismatch]]],
    *,
    source_quotas: Sequence[int],
    source_order: Sequence[int],
    candidate_validity: Sequence[Sequence[bool]],
    durations: Sequence[float],
    duration_targets: Sequence[float],
    state_limit: int,
    config: DirectMemberVisualConfig,
    original_lifeline: _FeasibilityLifelineResult,
) -> dict[str, Any]:
    """Find the exact minimum severe-edge count needed by a valid path.

    This is a non-authoritative counterfactual.  It clones the mismatch
    matrices and clears only ``severe_flags`` on the clone, while a separate
    immutable mask charges one unit for every transition whose original edge
    carried any severe flag.  Exact iterative deepening starts at one because
    the caller's original hard failure is the zero-budget proof.  Candidate
    validity, exact durations, balanced quotas, source order, and the
    no-adjacent-repeat rule are passed unchanged to the same exact feasibility
    search.  A minimum-one witness may authorize only a partition proposal; it
    is never a selectable path and can never authorize a render.
    """

    if original_lifeline.node is not None:
        raise ValueError("counterfactual requires an original hard failure")
    if original_lifeline.state_capacity_exhausted:
        raise ValueError("counterfactual requires proven original infeasibility")

    original_constraint_payload = _exact_hard_constraint_payload(
        matrices,
        source_quotas=source_quotas,
        source_order=source_order,
        candidate_validity=candidate_validity,
        durations=durations,
        duration_targets=duration_targets,
        maximum_source_duration_relative_deviation=(
            config.maximum_source_duration_relative_deviation
        ),
    )
    original_constraint_sha256 = _canonical_diagnostic_sha256(
        original_constraint_payload
    )
    original_failure_certificate_sha256 = _canonical_diagnostic_sha256(
        {
            "contract": "direct_member_exact_hard_infeasibility_certificate_v1",
            "hard_constraint_input_sha256": original_constraint_sha256,
            "exact_lifeline": original_lifeline.to_dict(),
            "constraint_infeasibility_proven": True,
            "duration_state_quantization_applied": False,
        }
    )

    severe_disabled_matrices = tuple(
        tuple(
            tuple(
                replace(mismatch, severe_flags=())
                for mismatch in outgoing_row
            )
            for outgoing_row in matrix
        )
        for matrix in matrices
    )
    severe_disabled_constraint_payload = _exact_hard_constraint_payload(
        severe_disabled_matrices,
        source_quotas=source_quotas,
        source_order=source_order,
        candidate_validity=candidate_validity,
        durations=durations,
        duration_targets=duration_targets,
        maximum_source_duration_relative_deviation=(
            config.maximum_source_duration_relative_deviation
        ),
    )
    severe_disabled_constraint_sha256 = _canonical_diagnostic_sha256(
        severe_disabled_constraint_payload
    )
    severe_transition_budget_mask = tuple(
        tuple(
            tuple(bool(mismatch.severe_flags) for mismatch in outgoing_row)
            for outgoing_row in matrix
        )
        for matrix in matrices
    )
    severe_transition_budget_mask_sha256 = _canonical_diagnostic_sha256(
        {
            "contract": "direct_member_original_severe_edge_budget_mask_v1",
            "edge_cost_policy": "one_per_adjacent_transition_with_any_severe_flag",
            "mask": severe_transition_budget_mask,
        }
    )
    maximum_diagnostic_budget = 3
    budget_attempts: list[dict[str, Any]] = []
    minimum_counterfactual: _FeasibilityLifelineResult | None = None
    minimum_severe_transition_count: int | None = None
    all_lower_budgets_proven_hard = True
    any_budget_capacity_exhausted = False
    for severe_budget in range(1, maximum_diagnostic_budget + 1):
        counterfactual = _exact_feasibility_lifeline(
            severe_disabled_matrices,
            source_quotas=source_quotas,
            source_order=source_order,
            candidate_validity=candidate_validity,
            durations=durations,
            duration_targets=duration_targets,
            state_limit=state_limit,
            config=config,
            severe_transition_budget_mask=severe_transition_budget_mask,
            minimum_severe_transition_count=severe_budget,
            maximum_severe_transition_count=severe_budget,
        )
        attempt_payload = {
            "severe_transition_budget": severe_budget,
            "path_found": counterfactual.node is not None,
            "explored_state_count": counterfactual.explored_state_count,
            "memoized_failed_state_count": (
                counterfactual.memoized_failed_state_count
            ),
            "state_capacity_exhausted": (
                counterfactual.state_capacity_exhausted
            ),
            "constraint_infeasibility_proven": bool(
                counterfactual.node is None
                and not counterfactual.state_capacity_exhausted
            ),
            "duration_state_quantization_applied": False,
            "severe_transition_count_requirement": "exactly_budget",
            "admissible_pruning": dict(
                counterfactual.admissible_pruning_diagnostics
            ),
        }
        attempt_payload["certificate_sha256"] = _canonical_diagnostic_sha256(
            {
                "contract": (
                    "direct_member_severe_budget_exact_attempt_certificate_v1"
                ),
                "original_failure_certificate_sha256": (
                    original_failure_certificate_sha256
                ),
                "severe_disabled_hard_constraint_sha256": (
                    severe_disabled_constraint_sha256
                ),
                "severe_transition_budget_mask_sha256": (
                    severe_transition_budget_mask_sha256
                ),
                **attempt_payload,
                "root_candidate_diagnostics": (
                    counterfactual.to_dict()["root_candidate_diagnostics"]
                ),
            }
        )
        budget_attempts.append(attempt_payload)
        if counterfactual.state_capacity_exhausted:
            any_budget_capacity_exhausted = True
            all_lower_budgets_proven_hard = False
            break
        if counterfactual.node is not None:
            minimum_counterfactual = counterfactual
            minimum_severe_transition_count = severe_budget
            break
        if not attempt_payload["constraint_infeasibility_proven"]:
            all_lower_budgets_proven_hard = False
            break

    blocking_transitions: list[dict[str, Any]] = []
    witness_sha256: str | None = None
    integrity_check_passed = False
    maximum_duration_relative_deviation: float | None = None
    optimality_proven = bool(
        minimum_counterfactual is not None
        and minimum_severe_transition_count is not None
        and all_lower_budgets_proven_hard
        and not any_budget_capacity_exhausted
    )
    if optimality_proven:
        assert minimum_counterfactual is not None
        assert minimum_counterfactual.node is not None
        assert minimum_severe_transition_count is not None
        path = minimum_counterfactual.node.path
        counts = tuple(path.count(index) for index in range(len(source_quotas)))
        deviations = tuple(
            abs(actual - target) / target
            for actual, target in zip(
                minimum_counterfactual.node.duration_totals,
                duration_targets,
            )
        )
        maximum_duration_relative_deviation = max(deviations)
        for boundary_index, (outgoing, incoming) in enumerate(
            zip(path, path[1:])
        ):
            original_flags = matrices[boundary_index][outgoing][incoming].severe_flags
            if original_flags:
                blocking_transitions.append(
                    {
                        "boundary_index": boundary_index,
                        "outgoing_eligible_source_index": outgoing,
                        "incoming_eligible_source_index": incoming,
                        "original_severe_flags": list(original_flags),
                    }
                )
        integrity_check_passed = bool(
            len(path) == len(candidate_validity)
            and counts == tuple(int(value) for value in source_quotas)
            and all(
                candidate_validity[position][source_index]
                for position, source_index in enumerate(path)
            )
            and all(left != right for left, right in zip(path, path[1:]))
            and maximum_duration_relative_deviation
            <= config.maximum_source_duration_relative_deviation + 1e-9
            and len(blocking_transitions) == minimum_severe_transition_count
        )
        if integrity_check_passed:
            witness_sha256 = _canonical_diagnostic_sha256(
                {
                    "contract": (
                        "direct_member_minimum_severe_exact_witness_v1"
                    ),
                    "severe_disabled_hard_constraint_sha256": (
                        severe_disabled_constraint_sha256
                    ),
                    "severe_transition_budget_mask_sha256": (
                        severe_transition_budget_mask_sha256
                    ),
                    "minimum_severe_transition_count": (
                        minimum_severe_transition_count
                    ),
                    "path": list(path),
                    "duration_totals_float_hex": [
                        float(value).hex()
                        for value in minimum_counterfactual.node.duration_totals
                    ],
                    "blocking_transitions": blocking_transitions,
                }
            )

    witness_found = bool(optimality_proven and integrity_check_passed)
    explored_state_count = sum(
        int(item["explored_state_count"]) for item in budget_attempts
    )
    memoized_failed_state_count = sum(
        int(item["memoized_failed_state_count"]) for item in budget_attempts
    )
    bounded_budget_domain_infeasibility_proven = bool(
        minimum_counterfactual is None
        and len(budget_attempts) == maximum_diagnostic_budget
        and all(
            item["constraint_infeasibility_proven"] is True
            for item in budget_attempts
        )
    )
    zero_budget_hard_proof = {
        "severe_transition_budget": 0,
        "path_found": False,
        "explored_state_count": original_lifeline.explored_state_count,
        "memoized_failed_state_count": (
            original_lifeline.memoized_failed_state_count
        ),
        "state_capacity_exhausted": False,
        "constraint_infeasibility_proven": True,
        "duration_state_quantization_applied": False,
        "certificate_sha256": original_failure_certificate_sha256,
    }
    lower_budget_hard_proofs = [zero_budget_hard_proof]
    lower_budget_hard_proofs.extend(
        dict(item)
        for item in budget_attempts
        if item["constraint_infeasibility_proven"] is True
        and (
            minimum_severe_transition_count is None
            or int(item["severe_transition_budget"])
            < minimum_severe_transition_count
        )
    )
    return {
        "contract": "diagnostic_only_severe_transition_counterfactual_v1",
        "optimization_contract": (
            "exact_iterative_deepening_minimum_severe_transition_count_"
            "then_deterministic_backward_source_order_v1"
        ),
        "severe_transition_cost_policy": (
            "one_per_adjacent_transition_with_any_original_severe_flag"
        ),
        "tie_break_contract": (
            "final_source_order_then_reverse_predecessor_"
            "severe_cost_mismatch_cost_source_order_v1"
        ),
        "invoked": True,
        "original_hard_constraint_sha256": original_constraint_sha256,
        "original_failure_certificate_sha256": (
            original_failure_certificate_sha256
        ),
        "severe_disabled_hard_constraint_sha256": (
            severe_disabled_constraint_sha256
        ),
        "severe_transition_budget_mask_sha256": (
            severe_transition_budget_mask_sha256
        ),
        "severe_disabled_witness_sha256": witness_sha256,
        "minimum_witness_path_sha256": witness_sha256,
        "minimum_severe_transition_count": (
            minimum_severe_transition_count if witness_found else None
        ),
        "minimum_severe_transition_count_lower_bound": (
            maximum_diagnostic_budget + 1
            if bounded_budget_domain_infeasibility_proven
            else None
        ),
        "maximum_diagnostic_severe_transition_budget": (
            maximum_diagnostic_budget
        ),
        "zero_budget_hard_proof": zero_budget_hard_proof,
        "lower_budget_hard_proofs": lower_budget_hard_proofs,
        "budget_attempts": budget_attempts,
        "optimality_proven": bool(witness_found),
        "production_single_split_proposal_authorized": bool(
            witness_found and minimum_severe_transition_count == 1
        ),
        "witness_found": witness_found,
        "integrity_check_passed": integrity_check_passed,
        "blocking_boundary_indices": sorted(
            {item["boundary_index"] for item in blocking_transitions}
        ) if witness_found else [],
        "blocking_transitions": blocking_transitions if witness_found else [],
        "maximum_duration_relative_deviation": (
            maximum_duration_relative_deviation if witness_found else None
        ),
        "state_limit_per_budget": state_limit,
        "explored_state_count": explored_state_count,
        "memoized_failed_state_count": memoized_failed_state_count,
        "state_capacity_exhausted": any_budget_capacity_exhausted,
        "constraint_infeasibility_proven": (
            bounded_budget_domain_infeasibility_proven
        ),
        "duration_state_quantization_applied": False,
        "candidate_validity_semantics_changed": False,
        "source_quota_semantics_changed": False,
        "exact_duration_semantics_changed": False,
        "source_order_semantics_changed": False,
        "no_adjacent_repeat_semantics_changed": False,
        "severe_transition_flags_disabled": True,
        "original_matrices_mutated": False,
        "selection_eligible": False,
        "render_authorized": False,
        "visual_gates_relaxed": False,
    }


def _source_duration_balance(
    source_indices: Sequence[int],
    span_durations: Sequence[float],
    source_quotas: Sequence[int],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Return actual, quota-adjusted target, and relative source exposures."""

    if len(source_indices) != len(span_durations):
        raise ValueError("duration balance path and span lengths must match")
    span_count = len(span_durations)
    source_count = len(source_quotas)
    if span_count == 0 or source_count == 0:
        raise ValueError("duration balance needs spans and sources")
    durations = tuple(float(value) for value in span_durations)
    if any(not math.isfinite(value) or value <= 0.0 for value in durations):
        raise ValueError("duration balance spans must be finite and positive")
    if sum(int(value) for value in source_quotas) != span_count:
        raise ValueError("duration balance quotas must cover every span")
    totals = [0.0] * source_count
    for source_index, duration in zip(source_indices, durations):
        if not 0 <= int(source_index) < source_count:
            raise ValueError("duration balance path references an unknown source")
        totals[int(source_index)] += duration
    total_duration = sum(durations)
    targets = tuple(
        total_duration * int(quota) / span_count for quota in source_quotas
    )
    if any(target <= 0.0 for target in targets):
        raise ValueError("duration balance targets must be positive")
    actual = tuple(totals)
    deviations = tuple(
        abs(value - target) / target
        for value, target in zip(actual, targets)
    )
    return (actual, targets, deviations)


def _duration_signature(values: Sequence[float]) -> tuple[float, ...]:
    """Coalesce near-equal duration states inside the lossy quality beam only."""

    return tuple(round(float(value), 6) for value in values)


def _balanced_quotas(
    span_count: int,
    source_count: int,
    preferred_source_order: Sequence[int],
) -> tuple[int, ...]:
    if span_count < source_count:
        raise ValueError("direct visual path needs at least one span per source")
    order = tuple(int(index) for index in preferred_source_order)
    if len(order) != source_count or set(order) != set(range(source_count)):
        raise ValueError("preferred_source_order must be a source permutation")
    base, extra = divmod(span_count, source_count)
    quotas = [base] * source_count
    for source_index in order[:extra]:
        quotas[source_index] += 1
    return tuple(quotas)


def _milp_bounded_same_source_hold_visual_path(
    matrices: Sequence[Sequence[Sequence[DirectBoundaryMismatch]]],
    *,
    source_quotas: Sequence[int],
    source_order: Sequence[int],
    candidate_validity: Sequence[Sequence[bool]],
    durations: Sequence[float],
    duration_targets: Sequence[float],
    priority_boundaries: frozenset[int],
    maximum_priority_cut_hold_count: int,
    maximum_hold_count: int,
    config: DirectMemberVisualConfig,
) -> tuple[tuple[int, ...], float, dict[str, Any]] | None:
    """Solve the bounded no-cut fallback without an exponential beam.

    Same-source edges represent a rendered hold, not a visually relaxed cut.
    Cross-source edges retain the original fail-closed severe gate.  The
    reconstructed path is audited independently before it is returned.
    """

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_array
    except (ImportError, OSError):
        return None

    span_count = len(candidate_validity)
    source_count = len(source_quotas)
    boundary_count = span_count - 1
    x_count = span_count * source_count
    edge_specs: list[tuple[int, int, int, bool, float]] = []
    edges_by_boundary: list[list[int]] = [[] for _ in range(boundary_count)]
    hold_edges_by_boundary: list[list[int]] = [
        [] for _ in range(boundary_count)
    ]
    for boundary_index in range(boundary_count):
        for left_source in range(source_count):
            if not candidate_validity[boundary_index][left_source]:
                continue
            for right_source in range(source_count):
                if not candidate_validity[boundary_index + 1][right_source]:
                    continue
                is_hold = left_source == right_source
                mismatch = matrices[boundary_index][left_source][right_source]
                if not is_hold and mismatch.severe_flags:
                    continue
                edge_index = len(edge_specs)
                edge_specs.append(
                    (
                        boundary_index,
                        left_source,
                        right_source,
                        is_hold,
                        (
                            float(config.same_source_hold_penalty)
                            if is_hold
                            else float(mismatch.cost)
                        ),
                    )
                )
                edges_by_boundary[boundary_index].append(edge_index)
                if is_hold:
                    hold_edges_by_boundary[boundary_index].append(edge_index)
    if any(not values for values in edges_by_boundary):
        return None

    edge_base = x_count
    ping_specs = tuple(
        (position, source_index)
        for position in range(2, span_count)
        for source_index in range(source_count)
    )
    ping_base = edge_base + len(edge_specs)
    duration_slack_base = ping_base + len(ping_specs)
    variable_count = duration_slack_base + 2 * source_count
    objective = np.zeros(variable_count, dtype=np.float64)
    lower_bounds = np.zeros(variable_count, dtype=np.float64)
    upper_bounds = np.ones(variable_count, dtype=np.float64)
    integrality = np.ones(variable_count, dtype=np.int32)
    order_rank = {
        source_index: rank for rank, source_index in enumerate(source_order)
    }
    for position in range(span_count):
        for source_index in range(source_count):
            variable = position * source_count + source_index
            if (
                not candidate_validity[position][source_index]
                or source_quotas[source_index] <= 0
            ):
                upper_bounds[variable] = 0.0
            objective[variable] = order_rank[source_index] * 1e-9
    for edge_index, (_boundary, _left, right, _is_hold, cost) in enumerate(
        edge_specs
    ):
        objective[edge_base + edge_index] = cost + order_rank[right] * 1e-8
    for ping_index in range(len(ping_specs)):
        objective[ping_base + ping_index] = config.ping_pong_penalty
    for source_index, target in enumerate(duration_targets):
        positive = duration_slack_base + 2 * source_index
        negative = positive + 1
        integrality[positive] = 0
        integrality[negative] = 0
        upper_bounds[positive] = max(float(target), sum(durations))
        upper_bounds[negative] = max(float(target), sum(durations))
        weight = config.duration_balance_weight / max(1e-9, float(target))
        objective[positive] = weight
        objective[negative] = weight

    rows: list[int] = []
    columns: list[int] = []
    coefficients: list[float] = []
    constraint_lower: list[float] = []
    constraint_upper: list[float] = []

    def add_constraint(
        terms: Sequence[tuple[int, float]],
        lower: float,
        upper: float,
    ) -> None:
        row = len(constraint_lower)
        for variable, coefficient in terms:
            rows.append(row)
            columns.append(variable)
            coefficients.append(float(coefficient))
        constraint_lower.append(float(lower))
        constraint_upper.append(float(upper))

    for position in range(span_count):
        add_constraint(
            tuple(
                (position * source_count + source_index, 1.0)
                for source_index in range(source_count)
            ),
            1.0,
            1.0,
        )
    for source_index in range(source_count):
        source_variables = tuple(
            (position * source_count + source_index, 1.0)
            for position in range(span_count)
        )
        add_constraint(
            source_variables,
            float(source_quotas[source_index]),
            float(source_quotas[source_index]),
        )
        duration_terms = tuple(
            (
                position * source_count + source_index,
                float(durations[position]),
            )
            for position in range(span_count)
        )
        target = float(duration_targets[source_index])
        tolerance = config.maximum_source_duration_relative_deviation
        add_constraint(
            duration_terms,
            target * (1.0 - tolerance),
            target * (1.0 + tolerance),
        )
        positive = duration_slack_base + 2 * source_index
        negative = positive + 1
        add_constraint(
            duration_terms + ((positive, -1.0), (negative, 1.0)),
            target,
            target,
        )
    for boundary_index, edge_indices in enumerate(edges_by_boundary):
        add_constraint(
            tuple((edge_base + edge_index, 1.0) for edge_index in edge_indices),
            1.0,
            1.0,
        )
        for edge_index in edge_indices:
            _boundary, left, right, _is_hold, _cost = edge_specs[edge_index]
            edge_variable = edge_base + edge_index
            add_constraint(
                (
                    (edge_variable, 1.0),
                    (boundary_index * source_count + left, -1.0),
                ),
                -math.inf,
                0.0,
            )
            add_constraint(
                (
                    (edge_variable, 1.0),
                    ((boundary_index + 1) * source_count + right, -1.0),
                ),
                -math.inf,
                0.0,
            )
    for ping_index, (position, source_index) in enumerate(ping_specs):
        ping_variable = ping_base + ping_index
        current = position * source_count + source_index
        middle = (position - 1) * source_count + source_index
        previous_previous = (position - 2) * source_count + source_index
        add_constraint(
            ((ping_variable, 1.0), (current, -1.0)),
            -math.inf,
            0.0,
        )
        add_constraint(
            ((ping_variable, 1.0), (previous_previous, -1.0)),
            -math.inf,
            0.0,
        )
        add_constraint(
            ((ping_variable, 1.0), (middle, 1.0)),
            -math.inf,
            1.0,
        )
        add_constraint(
            (
                (ping_variable, 1.0),
                (current, -1.0),
                (previous_previous, -1.0),
                (middle, 1.0),
            ),
            -1.0,
            math.inf,
        )

    hold_variables = tuple(
        (edge_base + edge_index, 1.0)
        for edge_indices in hold_edges_by_boundary
        for edge_index in edge_indices
    )
    add_constraint(hold_variables, 0.0, float(maximum_hold_count))
    priority_hold_variables = tuple(
        (edge_base + edge_index, 1.0)
        for boundary_index in priority_boundaries
        for edge_index in hold_edges_by_boundary[boundary_index]
    )
    if priority_hold_variables:
        add_constraint(
            priority_hold_variables,
            0.0,
            float(maximum_priority_cut_hold_count),
        )
    consecutive_limit = config.maximum_consecutive_same_source_holds
    for start in range(
        0,
        max(0, boundary_count - consecutive_limit),
    ):
        window = tuple(
            (edge_base + edge_index, 1.0)
            for boundary_index in range(
                start,
                start + consecutive_limit + 1,
            )
            for edge_index in hold_edges_by_boundary[boundary_index]
        )
        add_constraint(window, 0.0, float(consecutive_limit))

    run_constraint_count = 0
    for start in range(span_count - 1):
        run_seconds = float(durations[start])
        for end in range(start + 1, span_count):
            run_seconds += float(durations[end])
            if run_seconds <= config.maximum_same_source_run_seconds + 1e-9:
                continue
            for source_index in range(source_count):
                add_constraint(
                    tuple(
                        (
                            position * source_count + source_index,
                            1.0,
                        )
                        for position in range(start, end + 1)
                    ),
                    0.0,
                    float(end - start),
                )
                run_constraint_count += 1
            break

    matrix = coo_array(
        (
            np.asarray(coefficients, dtype=np.float64),
            (
                np.asarray(rows, dtype=np.int32),
                np.asarray(columns, dtype=np.int32),
            ),
        ),
        shape=(len(constraint_lower), variable_count),
    ).tocsr()
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(
            matrix,
            np.asarray(constraint_lower, dtype=np.float64),
            np.asarray(constraint_upper, dtype=np.float64),
        ),
        options={
            "disp": False,
            "presolve": True,
            "time_limit": _DIRECT_MEMBER_MILP_TIME_LIMIT_SECONDS,
            "node_limit": int(max(1, config.feasibility_lifeline_state_limit)),
            "mip_rel_gap": 0.0,
        },
    )
    status = int(result.status)
    solver_diagnostics = {
        "invoked": True,
        "backend": "scipy_optimize_milp_highs",
        "status": status,
        "message": str(result.message),
        "success": bool(result.success),
        "variable_count": variable_count,
        "binary_variable_count": duration_slack_base,
        "constraint_count": len(constraint_lower),
        "edge_variable_count": len(edge_specs),
        "hold_edge_variable_count": sum(
            len(values) for values in hold_edges_by_boundary
        ),
        "ping_pong_variable_count": len(ping_specs),
        "same_source_run_constraint_count": run_constraint_count,
        "mip_node_count": int(getattr(result, "mip_node_count", 0) or 0),
        "mip_gap": (
            float(result.mip_gap)
            if getattr(result, "mip_gap", None) is not None
            else None
        ),
        "objective_value": (
            float(result.fun) if getattr(result, "fun", None) is not None else None
        ),
        "time_limit_seconds": _DIRECT_MEMBER_MILP_TIME_LIMIT_SECONDS,
        "postsolve_path_audit_required": True,
    }
    if status == 2:
        return None
    solution = getattr(result, "x", None)
    if solution is None:
        raise DirectMemberVisualSelectionError(
            "direct visual safe-hold MILP exhausted bounded solver capacity",
            {
                "failure": "search_capacity_exhausted",
                "constraint_infeasibility_proven": False,
                "visual_gates_relaxed": False,
                "search": {
                    "strategy": "scipy_highs_bounded_safe_hold_milp_v1",
                    "solver": solver_diagnostics,
                },
            },
        )

    path = tuple(
        int(
            np.argmax(
                solution[
                    position * source_count : (position + 1) * source_count
                ]
            )
        )
        for position in range(span_count)
    )
    counts = tuple(path.count(source_index) for source_index in range(source_count))
    duration_totals = tuple(
        sum(
            float(durations[position])
            for position, selected_source in enumerate(path)
            if selected_source == source_index
        )
        for source_index in range(source_count)
    )
    deviations = tuple(
        abs(actual - target) / target
        for actual, target in zip(duration_totals, duration_targets)
    )
    hold_indices = tuple(
        boundary_index
        for boundary_index in range(boundary_count)
        if path[boundary_index] == path[boundary_index + 1]
    )
    priority_hold_indices = tuple(
        boundary_index
        for boundary_index in hold_indices
        if boundary_index in priority_boundaries
    )
    audit_failures: list[str] = []
    if counts != tuple(source_quotas):
        audit_failures.append("source_quota_mismatch")
    if any(
        not candidate_validity[position][source_index]
        for position, source_index in enumerate(path)
    ):
        audit_failures.append("content_invalid_choice")
    if len(hold_indices) > maximum_hold_count:
        audit_failures.append("maximum_same_source_hold_count_exceeded")
    if len(priority_hold_indices) > maximum_priority_cut_hold_count:
        audit_failures.append("maximum_priority_cut_hold_count_exceeded")
    consecutive_holds = 0
    maximum_consecutive_holds = 0
    run_seconds = float(durations[0])
    maximum_run_seconds = run_seconds
    for position in range(1, span_count):
        if path[position] == path[position - 1]:
            consecutive_holds += 1
            run_seconds += float(durations[position])
        else:
            consecutive_holds = 0
            run_seconds = float(durations[position])
            mismatch = matrices[position - 1][path[position - 1]][path[position]]
            if mismatch.severe_flags:
                audit_failures.append("severe_cross_source_edge_selected")
        maximum_consecutive_holds = max(
            maximum_consecutive_holds,
            consecutive_holds,
        )
        maximum_run_seconds = max(maximum_run_seconds, run_seconds)
    if maximum_consecutive_holds > config.maximum_consecutive_same_source_holds:
        audit_failures.append("maximum_consecutive_same_source_holds_exceeded")
    if maximum_run_seconds > config.maximum_same_source_run_seconds + 1e-7:
        audit_failures.append("maximum_same_source_run_seconds_exceeded")
    if max(deviations) > config.maximum_source_duration_relative_deviation + 1e-7:
        audit_failures.append("duration_balance_threshold_missed")
    allowed_edges = tuple(
        {
            (edge_specs[edge_index][1], edge_specs[edge_index][2])
            for edge_index in edge_indices
        }
        for edge_indices in edges_by_boundary
    )
    if any(
        (path[boundary], path[boundary + 1]) not in allowed_edges[boundary]
        for boundary in range(boundary_count)
    ):
        audit_failures.append("unmodeled_boundary_edge")
    solver_diagnostics["postsolve_path_audit_passed"] = not audit_failures
    solver_diagnostics["postsolve_path_audit_failures"] = audit_failures
    if audit_failures:
        raise DirectMemberVisualSelectionError(
            "direct visual safe-hold MILP witness failed independent audit",
            {
                "failure": "solver_witness_audit_failed",
                "constraint_infeasibility_proven": False,
                "visual_gates_relaxed": False,
                "search": {
                    "strategy": "scipy_highs_bounded_safe_hold_milp_v1",
                    "solver": solver_diagnostics,
                },
            },
        )

    cost = 0.0
    for position, source_index in enumerate(path):
        if position:
            previous = path[position - 1]
            cost += (
                config.same_source_hold_penalty
                if previous == source_index
                else matrices[position - 1][previous][source_index].cost
            )
        if (
            position >= 2
            and source_index == path[position - 2]
            and source_index != path[position - 1]
        ):
            cost += config.ping_pong_penalty
    cost += config.duration_balance_weight * sum(
        (actual / target) ** 2
        for actual, target in zip(duration_totals, duration_targets)
    )
    return (
        path,
        cost,
        {
            "strategy": "scipy_highs_bounded_safe_hold_milp_v1",
            "selection_basis": "audited_safe_no_cut_continuation_milp",
            "strict_cross_source_path_failed_first": True,
            "same_source_hold_count": len(hold_indices),
            "same_source_hold_boundary_indices": list(hold_indices),
            "maximum_same_source_hold_count": maximum_hold_count,
            "maximum_same_source_hold_fraction": (
                config.maximum_same_source_hold_fraction
            ),
            "maximum_consecutive_same_source_holds": (
                config.maximum_consecutive_same_source_holds
            ),
            "maximum_same_source_run_seconds": (
                config.maximum_same_source_run_seconds
            ),
            "measured_maximum_same_source_run_seconds": maximum_run_seconds,
            "priority_cut_boundary_indices": sorted(priority_boundaries),
            "priority_cut_boundary_hold_count": len(priority_hold_indices),
            "maximum_priority_cut_hold_count": maximum_priority_cut_hold_count,
            "cross_source_severe_edge_count_selected": 0,
            "all_source_quotas_satisfied": True,
            "maximum_duration_relative_deviation": max(deviations),
            "visual_gates_relaxed": False,
            "attempted_beam_widths": [],
            "attempts": [],
            "beam_skipped_for_large_exact_problem": True,
            "search_capacity_exhausted": False,
            "constraint_infeasibility_proven": False,
            "solver": solver_diagnostics,
        },
    )


def _bounded_same_source_hold_visual_path(
    matrices: Sequence[Sequence[Sequence[DirectBoundaryMismatch]]],
    *,
    source_quotas: Sequence[int],
    source_order: Sequence[int],
    candidate_validity: Sequence[Sequence[bool]],
    span_durations: Sequence[float],
    priority_cut_boundary_indices: Sequence[int] = (),
    maximum_priority_cut_hold_count: int = 0,
    config: DirectMemberVisualConfig,
) -> tuple[tuple[int, ...], float, dict[str, Any]] | None:
    """Find a safe all-source path with a small number of no-cut holds.

    A same-source continuation is not a relaxed visual transition: no cut is
    rendered at that boundary, so cross-source severe edges remain completely
    ineligible.  The fallback is attempted only after the strict all-cut path
    fails, preserves the exact balanced quotas and duration gate, and caps
    both the total and consecutive holds so MEMBER edits retain normal cut
    density even with sparse fancams. Priority boundaries use a separate,
    caller-audited redundancy budget; their remaining cut density is never
    inferred by this optimizer.
    """

    span_count = len(matrices) + 1
    source_count = len(source_quotas)
    durations = tuple(float(value) for value in span_durations)
    if (
        span_count < 2
        or len(candidate_validity) != span_count
        or len(durations) != span_count
        or sum(int(value) for value in source_quotas) != span_count
    ):
        raise ValueError("same-source hold fallback shape is invalid")
    boundary_count = span_count - 1
    priority_boundaries = frozenset(
        int(value) for value in priority_cut_boundary_indices
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < boundary_count
        for value in priority_cut_boundary_indices
    ):
        raise ValueError("priority cut boundary indices are invalid")
    if (
        isinstance(maximum_priority_cut_hold_count, bool)
        or not isinstance(maximum_priority_cut_hold_count, int)
        or maximum_priority_cut_hold_count < 0
    ):
        raise ValueError("maximum priority cut hold count is invalid")
    maximum_hold_count = min(
        config.maximum_same_source_hold_count,
        max(
            1,
            int(
                math.floor(
                    boundary_count
                    * config.maximum_same_source_hold_fraction
                )
            ),
        ),
    )
    if maximum_hold_count <= 0 or config.maximum_consecutive_same_source_holds <= 0:
        return None
    total_duration = sum(durations)
    duration_targets = tuple(
        total_duration * int(quota) / span_count for quota in source_quotas
    )
    if _prefer_direct_member_milp(candidate_validity):
        return _milp_bounded_same_source_hold_visual_path(
            matrices,
            source_quotas=source_quotas,
            source_order=source_order,
            candidate_validity=candidate_validity,
            durations=durations,
            duration_targets=duration_targets,
            priority_boundaries=priority_boundaries,
            maximum_priority_cut_hold_count=maximum_priority_cut_hold_count,
            maximum_hold_count=maximum_hold_count,
            config=config,
        )
    future_valid_counts: list[tuple[int, ...]] = [
        (0,) * source_count for _ in range(span_count + 1)
    ]
    for position in range(span_count - 1, -1, -1):
        following = future_valid_counts[position + 1]
        future_valid_counts[position] = tuple(
            following[source_index]
            + int(bool(candidate_validity[position][source_index]))
            for source_index in range(source_count)
        )

    attempted_widths: list[int] = []
    attempt_diagnostics: list[dict[str, Any]] = []
    for width in _adaptive_beam_widths(
        config.beam_width,
        config.maximum_adaptive_beam_width,
    ):
        attempted_widths.append(width)
        states: dict[
            tuple[
                tuple[int, ...],
                int,
                int,
                int,
                int,
                int,
                tuple[float, ...],
            ],
            _BeamNode,
        ] = {}
        for source_index in source_order:
            if not candidate_validity[0][source_index]:
                continue
            counts = [0] * source_count
            counts[source_index] = 1
            duration_totals = [0.0] * source_count
            duration_totals[source_index] = durations[0]
            cost = config.duration_balance_weight * (
                durations[0] / duration_targets[source_index]
            ) ** 2
            states[
                (
                    tuple(counts),
                    source_index,
                    -1,
                    0,
                    0,
                    0,
                    _duration_signature(duration_totals),
                )
            ] = _BeamNode(cost, (source_index,), tuple(duration_totals))

        pruned_state_count = 0
        peak_generated_state_count = len(states)
        failed_at_segment_index: int | None = None
        for position in range(1, span_count):
            matrix = matrices[position - 1]
            next_states: dict[
                tuple[
                    tuple[int, ...],
                    int,
                    int,
                    int,
                    int,
                    int,
                    tuple[float, ...],
                ],
                _BeamNode,
            ] = {}
            for (
                counts,
                previous,
                previous_previous,
                hold_count,
                consecutive_hold_count,
                priority_hold_count,
                _duration_state,
            ), node in states.items():
                for source_index in source_order:
                    if (
                        not candidate_validity[position][source_index]
                        or counts[source_index] >= source_quotas[source_index]
                    ):
                        continue
                    is_hold = source_index == previous
                    is_priority_hold = bool(
                        is_hold and position - 1 in priority_boundaries
                    )
                    if is_hold:
                        if (
                            hold_count >= maximum_hold_count
                            or consecutive_hold_count
                            >= config.maximum_consecutive_same_source_holds
                            or (
                                is_priority_hold
                                and priority_hold_count
                                >= maximum_priority_cut_hold_count
                            )
                        ):
                            continue
                        same_source_run_seconds = durations[position]
                        prior_position = position - 1
                        while (
                            prior_position >= 0
                            and node.path[prior_position] == source_index
                        ):
                            same_source_run_seconds += durations[prior_position]
                            prior_position -= 1
                        if (
                            same_source_run_seconds
                            > config.maximum_same_source_run_seconds + 1e-9
                        ):
                            continue
                    elif matrix[previous][source_index].severe_flags:
                        continue
                    updated = list(counts)
                    updated[source_index] += 1
                    if any(
                        source_quotas[index] - updated[index]
                        > future_valid_counts[position + 1][index]
                        for index in range(source_count)
                    ):
                        continue
                    next_hold_count = hold_count + int(is_hold)
                    next_consecutive = consecutive_hold_count + 1 if is_hold else 0
                    next_priority_hold_count = (
                        priority_hold_count + int(is_priority_hold)
                    )
                    transition_cost = (
                        config.same_source_hold_penalty
                        if is_hold
                        else matrix[previous][source_index].cost
                    )
                    cost = node.cost + transition_cost
                    if (
                        not is_hold
                        and source_index == previous_previous
                    ):
                        cost += config.ping_pong_penalty
                    duration_totals = list(node.duration_totals)
                    previous_duration = duration_totals[source_index]
                    updated_duration = previous_duration + durations[position]
                    duration_totals[source_index] = updated_duration
                    target_duration = duration_targets[source_index]
                    cost += config.duration_balance_weight * (
                        (updated_duration / target_duration) ** 2
                        - (previous_duration / target_duration) ** 2
                    )
                    path = node.path + (source_index,)
                    key = (
                        tuple(updated),
                        source_index,
                        previous,
                        next_hold_count,
                        next_consecutive,
                        next_priority_hold_count,
                        _duration_signature(duration_totals),
                    )
                    existing = next_states.get(key)
                    if existing is None or (cost, path) < (
                        existing.cost,
                        existing.path,
                    ):
                        next_states[key] = _BeamNode(
                            cost,
                            path,
                            tuple(duration_totals),
                        )
            peak_generated_state_count = max(
                peak_generated_state_count,
                len(next_states),
            )
            if not next_states:
                states = {}
                failed_at_segment_index = position
                break
            if len(next_states) > width:
                pruned_state_count += len(next_states) - width
                states = dict(
                    sorted(
                        next_states.items(),
                        key=lambda item: (item[1].cost, item[1].path),
                    )[:width]
                )
            else:
                states = next_states

        eligible: list[
            tuple[int, int, _BeamNode, tuple[float, ...]]
        ] = []
        for (
            counts,
            _last,
            _previous,
            hold_count,
            _consecutive,
            priority_hold_count,
            _duration_state,
        ), node in states.items():
            if counts != tuple(source_quotas):
                continue
            deviations = tuple(
                abs(actual - target) / target
                for actual, target in zip(
                    node.duration_totals,
                    duration_targets,
                )
            )
            if (
                max(deviations)
                <= config.maximum_source_duration_relative_deviation + 1e-9
            ):
                eligible.append(
                    (hold_count, priority_hold_count, node, deviations)
                )
        attempt_diagnostics.append(
            {
                "beam_width": width,
                "terminal_state_count": len(states),
                "duration_eligible_path_count": len(eligible),
                "pruned_state_count": pruned_state_count,
                "peak_generated_state_count": peak_generated_state_count,
                "failed_at_segment_index": failed_at_segment_index,
            }
        )
        if eligible:
            hold_count, priority_hold_count, best, deviations = min(
                eligible,
                key=lambda item: (
                    item[2].cost,
                    item[0],
                    item[1],
                    item[2].path,
                ),
            )
            return (
                best.path,
                best.cost,
                {
                    "strategy": "bounded_safe_same_source_hold_fallback_v1",
                    "selection_basis": "safe_no_cut_continuation_beam",
                    "strict_cross_source_path_failed_first": True,
                    "same_source_hold_count": hold_count,
                    "maximum_same_source_hold_count": maximum_hold_count,
                    "maximum_same_source_hold_fraction": (
                        config.maximum_same_source_hold_fraction
                    ),
                    "maximum_consecutive_same_source_holds": (
                        config.maximum_consecutive_same_source_holds
                    ),
                    "maximum_same_source_run_seconds": (
                        config.maximum_same_source_run_seconds
                    ),
                    "priority_cut_boundary_indices": sorted(
                        priority_boundaries
                    ),
                    "priority_cut_boundary_hold_count": priority_hold_count,
                    "maximum_priority_cut_hold_count": (
                        maximum_priority_cut_hold_count
                    ),
                    "cross_source_severe_edge_count_selected": 0,
                    "all_source_quotas_satisfied": True,
                    "maximum_duration_relative_deviation": max(deviations),
                    "visual_gates_relaxed": False,
                    "attempted_beam_widths": attempted_widths,
                    "attempts": attempt_diagnostics,
                    "search_capacity_exhausted": False,
                    "constraint_infeasibility_proven": False,
                },
            )
        if pruned_state_count == 0:
            break
    return None


def _heterogeneous_cinematic_visual_path(
    matrices: Sequence[Sequence[Sequence[DirectBoundaryMismatch]]],
    *,
    source_order: Sequence[int],
    candidate_validity: Sequence[Sequence[bool]],
    span_durations: Sequence[float],
    source_geometries: Sequence[DirectMemberSourceFrameGeometry | None],
    priority_cut_boundary_indices: Sequence[int],
    maximum_priority_cut_hold_count: int,
    config: DirectMemberVisualConfig,
) -> tuple[tuple[int, ...], float, dict[str, Any]] | None:
    """Use heterogeneous fancams by role without authoring a bad hard cut.

    Landscape facecams receive a small, guaranteed highlight allocation while
    portrait performance fancams carry the continuous edit.  A raw severe edge
    is admitted only when ``DirectMemberCinematicBridge`` proves that the
    remaining discrepancy is framing-only and the outgoing handle is itself a
    content-valid span.  Such an edge is not a hard cut: downstream rendering
    must author and verify its required zoom/blur overlap.
    """

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_array
    except (ImportError, OSError):
        return None

    span_count = len(candidate_validity)
    source_count = len(source_geometries)
    boundary_count = span_count - 1
    durations = tuple(float(value) for value in span_durations)
    if (
        span_count < 2
        or len(matrices) != boundary_count
        or not source_geometries
        or len(candidate_validity[0]) != source_count
        or len(durations) != span_count
    ):
        raise ValueError("heterogeneous cinematic path shape is invalid")
    landscape_sources = tuple(
        index
        for index, geometry in enumerate(source_geometries)
        if geometry is not None
        and geometry.source_width > geometry.source_height
    )
    portrait_sources = tuple(
        index for index in range(source_count) if index not in landscape_sources
    )
    if not landscape_sources or not portrait_sources:
        return None
    high_resolution_portrait_sources = tuple(
        index
        for index in portrait_sources
        if source_geometries[index] is not None
        and source_geometries[index].lossless_zoom_capacity >= 1.50
    )
    native_resolution_portrait_sources = tuple(
        index
        for index in portrait_sources
        if index not in high_resolution_portrait_sources
    )
    priority_boundaries = frozenset(
        int(value) for value in priority_cut_boundary_indices
    )
    if not priority_boundaries:
        return None
    highlight_spans = frozenset(
        span_index
        for boundary_index in priority_boundaries
        for span_index in (boundary_index, boundary_index + 1)
    )

    average_count = span_count / source_count
    landscape_target = max(1, int(round(average_count * 0.25)))
    if span_count >= source_count * 2:
        landscape_target = max(2, landscape_target)
    count_targets = [landscape_target] * source_count
    remaining = span_count - landscape_target * len(landscape_sources)
    if remaining < len(portrait_sources):
        return None
    portrait_order = tuple(
        source_index
        for source_index in source_order
        if source_index in portrait_sources
    )
    # Every official source remains represented, but 4K portrait fancams carry
    # more of the full-song picture than native-output 1080p sources.  The
    # latter still receive several best-fit cuts; they are not discarded. This
    # prevents repeated 4K/1080p quality pumping while preserving source
    # diversity and the user's sparse-fancam requirement.
    portrait_weights = {
        source_index: (
            2.0 if source_index in high_resolution_portrait_sources else 1.0
        )
        for source_index in portrait_sources
    }
    portrait_extra_budget = remaining - len(portrait_sources)
    total_portrait_weight = sum(portrait_weights.values())
    raw_portrait_extras = {
        source_index: (
            portrait_extra_budget
            * portrait_weights[source_index]
            / total_portrait_weight
        )
        for source_index in portrait_sources
    }
    portrait_allocations = {
        source_index: 1 + int(math.floor(raw_portrait_extras[source_index]))
        for source_index in portrait_sources
    }
    portrait_remainder = remaining - sum(portrait_allocations.values())
    portrait_order_rank = {
        source_index: rank
        for rank, source_index in enumerate(portrait_order)
    }
    remainder_order = sorted(
        portrait_sources,
        key=lambda source_index: (
            -(
                raw_portrait_extras[source_index]
                - math.floor(raw_portrait_extras[source_index])
            ),
            portrait_order_rank[source_index],
        ),
    )
    for source_index in portrait_sources:
        count_targets[source_index] = portrait_allocations[source_index]
    for source_index in remainder_order[:portrait_remainder]:
        count_targets[source_index] += 1
    count_minimums = tuple(
        (
            1
            if source_index in landscape_sources
            else max(2, int(math.floor(count_targets[source_index] * 0.45)))
        )
        for source_index in range(source_count)
    )
    count_maximums = tuple(
        (
            max(landscape_target, int(math.ceil(average_count * 0.65)))
            if source_index in landscape_sources
            else max(2, int(math.ceil(count_targets[source_index] * 1.50)))
        )
        for source_index in range(source_count)
    )
    if sum(count_minimums) > span_count or sum(count_maximums) < span_count:
        return None
    total_duration = sum(durations)
    duration_targets = tuple(
        total_duration * target / span_count for target in count_targets
    )
    duration_minimums = tuple(
        target * (0.25 if source_index in landscape_sources else 0.45)
        for source_index, target in enumerate(duration_targets)
    )
    duration_maximums = tuple(
        target * (3.00 if source_index in landscape_sources else 1.65)
        for source_index, target in enumerate(duration_targets)
    )
    maximum_same_source_run_seconds = max(
        config.maximum_same_source_run_seconds,
        _DIRECT_MEMBER_HETEROGENEOUS_MAXIMUM_RUN_SECONDS,
    )

    x_count = span_count * source_count
    upper_x = [1.0] * x_count
    for position in range(span_count):
        for source_index in range(source_count):
            if (
                not candidate_validity[position][source_index]
                or (
                    source_index in landscape_sources
                    and position not in highlight_spans
                )
            ):
                upper_x[position * source_count + source_index] = 0.0
    if any(
        sum(
            upper_x[position * source_count + source_index] > 0.5
            for position in range(span_count)
        )
        < count_minimums[source_index]
        for source_index in range(source_count)
    ):
        return None

    edge_specs: list[tuple[int, int, int, bool, bool, float]] = []
    edges_by_boundary: list[list[int]] = [[] for _ in range(boundary_count)]
    hold_edges_by_boundary: list[list[int]] = [
        [] for _ in range(boundary_count)
    ]
    bridge_edges_by_boundary: list[list[int]] = [
        [] for _ in range(boundary_count)
    ]
    for boundary_index in range(boundary_count):
        bridge_candidates: list[tuple[float, int, int]] = []
        for left_source in range(source_count):
            if upper_x[boundary_index * source_count + left_source] <= 0.5:
                continue
            for right_source in range(source_count):
                if (
                    upper_x[(boundary_index + 1) * source_count + right_source]
                    <= 0.5
                    or left_source == right_source
                ):
                    continue
                mismatch = matrices[boundary_index][left_source][right_source]
                if (
                    mismatch.cinematic_bridge is not None
                    and boundary_index in priority_boundaries
                    and candidate_validity[boundary_index + 1][left_source]
                ):
                    bridge_candidates.append(
                        (float(mismatch.cost), left_source, right_source)
                    )
        retained_bridges: set[tuple[int, int]] = set()
        grouped: dict[tuple[int, str], list[tuple[float, int, int]]] = {}
        for item in bridge_candidates:
            _cost, left_source, right_source = item
            if left_source in landscape_sources:
                key = (left_source, "outgoing")
            else:
                key = (right_source, "incoming")
            grouped.setdefault(key, []).append(item)
        for values in grouped.values():
            for _cost, left_source, right_source in sorted(values)[:3]:
                retained_bridges.add((left_source, right_source))

        for left_source in range(source_count):
            if upper_x[boundary_index * source_count + left_source] <= 0.5:
                continue
            for right_source in range(source_count):
                if (
                    upper_x[(boundary_index + 1) * source_count + right_source]
                    <= 0.5
                ):
                    continue
                mismatch = matrices[boundary_index][left_source][right_source]
                is_hold = left_source == right_source
                is_bridge = bool(
                    not is_hold
                    and mismatch.severe_flags
                    and (left_source, right_source) in retained_bridges
                )
                if not is_hold and mismatch.severe_flags and not is_bridge:
                    continue
                edge_index = len(edge_specs)
                edge_specs.append(
                    (
                        boundary_index,
                        left_source,
                        right_source,
                        is_hold,
                        is_bridge,
                        float(mismatch.cost),
                    )
                )
                edges_by_boundary[boundary_index].append(edge_index)
                if is_hold:
                    hold_edges_by_boundary[boundary_index].append(edge_index)
                if is_bridge:
                    bridge_edges_by_boundary[boundary_index].append(edge_index)
    if any(not values for values in edges_by_boundary):
        return None

    edge_base = x_count
    ping_specs = tuple(
        (position, source_index)
        for position in range(2, span_count)
        for source_index in range(source_count)
    )
    ping_base = edge_base + len(edge_specs)
    count_slack_base = ping_base + len(ping_specs)
    duration_slack_base = count_slack_base + 2 * source_count
    variable_count = duration_slack_base + 2 * source_count
    objective = np.zeros(variable_count, dtype=np.float64)
    lower_bounds = np.zeros(variable_count, dtype=np.float64)
    upper_bounds = np.ones(variable_count, dtype=np.float64)
    upper_bounds[:x_count] = np.asarray(upper_x, dtype=np.float64)
    integrality = np.ones(variable_count, dtype=np.int32)
    order_rank = {
        source_index: rank for rank, source_index in enumerate(source_order)
    }
    for position in range(span_count):
        for source_index in range(source_count):
            objective[position * source_count + source_index] = (
                order_rank[source_index] * 1e-9
            )
    for edge_index, (
        _boundary,
        _left,
        right,
        is_hold,
        is_bridge,
        cost,
    ) in enumerate(edge_specs):
        objective[edge_base + edge_index] = (
            cost
            + (
                _DIRECT_MEMBER_CINEMATIC_BRIDGE_COST_PENALTY
                if is_bridge
                else 0.0
            )
            + (max(2.0, config.same_source_hold_penalty) if is_hold else 0.0)
            + order_rank[right] * 1e-8
        )
    for ping_index in range(len(ping_specs)):
        objective[ping_base + ping_index] = config.ping_pong_penalty
    for source_index, target in enumerate(count_targets):
        positive = count_slack_base + 2 * source_index
        negative = positive + 1
        integrality[positive] = 0
        integrality[negative] = 0
        upper_bounds[positive] = float(span_count)
        upper_bounds[negative] = float(span_count)
        weight = 3.0 / max(1.0, float(target))
        objective[positive] = weight
        objective[negative] = weight
    for source_index, target in enumerate(duration_targets):
        positive = duration_slack_base + 2 * source_index
        negative = positive + 1
        integrality[positive] = 0
        integrality[negative] = 0
        upper_bounds[positive] = total_duration
        upper_bounds[negative] = total_duration
        weight = config.duration_balance_weight / max(1e-9, target)
        objective[positive] = weight
        objective[negative] = weight

    rows: list[int] = []
    columns: list[int] = []
    coefficients: list[float] = []
    constraint_lower: list[float] = []
    constraint_upper: list[float] = []

    def add_constraint(
        terms: Sequence[tuple[int, float]],
        lower: float,
        upper: float,
    ) -> None:
        row = len(constraint_lower)
        for variable, coefficient in terms:
            rows.append(row)
            columns.append(variable)
            coefficients.append(float(coefficient))
        constraint_lower.append(float(lower))
        constraint_upper.append(float(upper))

    for position in range(span_count):
        add_constraint(
            tuple(
                (position * source_count + source_index, 1.0)
                for source_index in range(source_count)
            ),
            1.0,
            1.0,
        )
    for source_index in range(source_count):
        count_terms = tuple(
            (position * source_count + source_index, 1.0)
            for position in range(span_count)
        )
        add_constraint(
            count_terms,
            float(count_minimums[source_index]),
            float(count_maximums[source_index]),
        )
        count_positive = count_slack_base + 2 * source_index
        count_negative = count_positive + 1
        add_constraint(
            count_terms
            + ((count_positive, -1.0), (count_negative, 1.0)),
            float(count_targets[source_index]),
            float(count_targets[source_index]),
        )
        duration_terms = tuple(
            (
                position * source_count + source_index,
                durations[position],
            )
            for position in range(span_count)
        )
        add_constraint(
            duration_terms,
            duration_minimums[source_index],
            duration_maximums[source_index],
        )
        duration_positive = duration_slack_base + 2 * source_index
        duration_negative = duration_positive + 1
        add_constraint(
            duration_terms
            + ((duration_positive, -1.0), (duration_negative, 1.0)),
            duration_targets[source_index],
            duration_targets[source_index],
        )
    for boundary_index, edge_indices in enumerate(edges_by_boundary):
        add_constraint(
            tuple((edge_base + edge_index, 1.0) for edge_index in edge_indices),
            1.0,
            1.0,
        )
        for edge_index in edge_indices:
            _boundary, left, right, _hold, _bridge, _cost = edge_specs[edge_index]
            edge_variable = edge_base + edge_index
            add_constraint(
                (
                    (edge_variable, 1.0),
                    (boundary_index * source_count + left, -1.0),
                ),
                -math.inf,
                0.0,
            )
            add_constraint(
                (
                    (edge_variable, 1.0),
                    ((boundary_index + 1) * source_count + right, -1.0),
                ),
                -math.inf,
                0.0,
            )
    for ping_index, (position, source_index) in enumerate(ping_specs):
        ping_variable = ping_base + ping_index
        current = position * source_count + source_index
        middle = (position - 1) * source_count + source_index
        previous_previous = (position - 2) * source_count + source_index
        add_constraint(
            ((ping_variable, 1.0), (current, -1.0)),
            -math.inf,
            0.0,
        )
        add_constraint(
            ((ping_variable, 1.0), (previous_previous, -1.0)),
            -math.inf,
            0.0,
        )
        add_constraint(
            ((ping_variable, 1.0), (middle, 1.0)),
            -math.inf,
            1.0,
        )
        add_constraint(
            (
                (ping_variable, 1.0),
                (current, -1.0),
                (previous_previous, -1.0),
                (middle, 1.0),
            ),
            -1.0,
            math.inf,
        )

    maximum_hold_count = min(
        config.maximum_same_source_hold_count,
        max(
            1,
            int(
                math.floor(
                    boundary_count * config.maximum_same_source_hold_fraction
                )
            ),
        ),
    )
    hold_variables = tuple(
        (edge_base + edge_index, 1.0)
        for values in hold_edges_by_boundary
        for edge_index in values
    )
    add_constraint(hold_variables, 0.0, float(maximum_hold_count))
    priority_hold_variables = tuple(
        (edge_base + edge_index, 1.0)
        for boundary_index in priority_boundaries
        for edge_index in hold_edges_by_boundary[boundary_index]
    )
    if priority_hold_variables:
        add_constraint(
            priority_hold_variables,
            0.0,
            float(maximum_priority_cut_hold_count),
        )
    consecutive_limit = config.maximum_consecutive_same_source_holds
    for start in range(max(0, boundary_count - consecutive_limit)):
        add_constraint(
            tuple(
                (edge_base + edge_index, 1.0)
                for boundary_index in range(
                    start,
                    start + consecutive_limit + 1,
                )
                for edge_index in hold_edges_by_boundary[boundary_index]
            ),
            0.0,
            float(consecutive_limit),
        )
    for start in range(span_count - 1):
        run_seconds = durations[start]
        for end in range(start + 1, span_count):
            run_seconds += durations[end]
            if run_seconds <= maximum_same_source_run_seconds + 1e-9:
                continue
            for source_index in range(source_count):
                add_constraint(
                    tuple(
                        (
                            position * source_count + source_index,
                            1.0,
                        )
                        for position in range(start, end + 1)
                    ),
                    0.0,
                    float(end - start),
                )
            break
    bridge_variables = tuple(
        (edge_base + edge_index, 1.0)
        for values in bridge_edges_by_boundary
        for edge_index in values
    )
    if not bridge_variables:
        return None
    add_constraint(
        bridge_variables,
        0.0,
        float(_DIRECT_MEMBER_CINEMATIC_BRIDGE_MAXIMUM_COUNT),
    )
    gap = _DIRECT_MEMBER_CINEMATIC_BRIDGE_MINIMUM_BOUNDARY_GAP
    for start in range(max(0, boundary_count - gap + 1)):
        window = tuple(
            (edge_base + edge_index, 1.0)
            for boundary_index in range(start, start + gap)
            for edge_index in bridge_edges_by_boundary[boundary_index]
        )
        if window:
            add_constraint(window, 0.0, 1.0)

    matrix = coo_array(
        (
            np.asarray(coefficients, dtype=np.float64),
            (
                np.asarray(rows, dtype=np.int32),
                np.asarray(columns, dtype=np.int32),
            ),
        ),
        shape=(len(constraint_lower), variable_count),
    ).tocsr()
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(
            matrix,
            np.asarray(constraint_lower, dtype=np.float64),
            np.asarray(constraint_upper, dtype=np.float64),
        ),
        options={
            "disp": False,
            "presolve": True,
            "time_limit": _DIRECT_MEMBER_MILP_TIME_LIMIT_SECONDS,
            "node_limit": int(max(1, config.feasibility_lifeline_state_limit)),
            "mip_rel_gap": 0.02,
        },
    )
    status = int(result.status)
    solution = getattr(result, "x", None)
    solver_diagnostics = {
        "invoked": True,
        "backend": "scipy_optimize_milp_highs",
        "status": status,
        "message": str(result.message),
        "success": bool(result.success),
        "variable_count": variable_count,
        "constraint_count": len(constraint_lower),
        "edge_variable_count": len(edge_specs),
        "bridge_candidate_edge_count": sum(
            len(values) for values in bridge_edges_by_boundary
        ),
        "mip_node_count": int(getattr(result, "mip_node_count", 0) or 0),
        "mip_gap": (
            float(result.mip_gap)
            if getattr(result, "mip_gap", None) is not None
            else None
        ),
        "objective_value": (
            float(result.fun) if getattr(result, "fun", None) is not None else None
        ),
        "time_limit_seconds": _DIRECT_MEMBER_MILP_TIME_LIMIT_SECONDS,
        "postsolve_path_audit_required": True,
    }
    if status == 2:
        return None
    if solution is None:
        raise DirectMemberVisualSelectionError(
            "heterogeneous cinematic MILP exhausted bounded solver capacity",
            {
                "failure": "search_capacity_exhausted",
                "constraint_infeasibility_proven": False,
                "visual_gates_relaxed": False,
                "search": {
                    "strategy": (
                        "scipy_highs_heterogeneous_cinematic_path_milp_v1"
                    ),
                    "solver": solver_diagnostics,
                },
            },
        )

    path = tuple(
        int(
            np.argmax(
                solution[
                    position * source_count : (position + 1) * source_count
                ]
            )
        )
        for position in range(span_count)
    )
    counts = tuple(path.count(source_index) for source_index in range(source_count))
    duration_totals = tuple(
        sum(
            durations[position]
            for position, selected_source in enumerate(path)
            if selected_source == source_index
        )
        for source_index in range(source_count)
    )
    hold_indices = tuple(
        boundary_index
        for boundary_index, (left, right) in enumerate(zip(path, path[1:]))
        if left == right
    )
    bridge_indices = tuple(
        boundary_index
        for boundary_index, (left, right) in enumerate(zip(path, path[1:]))
        if left != right
        and matrices[boundary_index][left][right].cinematic_bridge is not None
    )
    audit_failures: list[str] = []
    if any(
        not minimum <= actual <= maximum
        for actual, minimum, maximum in zip(
            counts,
            count_minimums,
            count_maximums,
        )
    ):
        audit_failures.append("role_count_bounds_missed")
    if any(
        actual < minimum - 1e-7 or actual > maximum + 1e-7
        for actual, minimum, maximum in zip(
            duration_totals,
            duration_minimums,
            duration_maximums,
        )
    ):
        audit_failures.append("role_duration_bounds_missed")
    if any(
        not candidate_validity[position][source_index]
        for position, source_index in enumerate(path)
    ):
        audit_failures.append("content_invalid_choice")
    if any(
        source_index in landscape_sources and position not in highlight_spans
        for position, source_index in enumerate(path)
    ):
        audit_failures.append("landscape_facecam_outside_highlight")
    if len(hold_indices) > maximum_hold_count:
        audit_failures.append("maximum_same_source_hold_count_exceeded")
    measured_maximum_run_seconds = max(
        sum(
            durations[position]
            for position in range(start, end)
        )
        for start, end in (
            (start, end)
            for start in range(span_count)
            for end in range(start + 1, span_count + 1)
            if all(path[position] == path[start] for position in range(start, end))
            and (end == span_count or path[end] != path[start])
            and (start == 0 or path[start - 1] != path[start])
        )
    )
    if measured_maximum_run_seconds > maximum_same_source_run_seconds + 1e-7:
        audit_failures.append("maximum_same_source_run_seconds_exceeded")
    if (
        sum(index in priority_boundaries for index in hold_indices)
        > maximum_priority_cut_hold_count
    ):
        audit_failures.append("maximum_priority_cut_hold_count_exceeded")
    if not bridge_indices or len(bridge_indices) > (
        _DIRECT_MEMBER_CINEMATIC_BRIDGE_MAXIMUM_COUNT
    ):
        audit_failures.append("cinematic_bridge_count_invalid")
    if any(index not in priority_boundaries for index in bridge_indices):
        audit_failures.append("cinematic_bridge_outside_highlight")
    if any(
        right - left < _DIRECT_MEMBER_CINEMATIC_BRIDGE_MINIMUM_BOUNDARY_GAP
        for left, right in zip(bridge_indices, bridge_indices[1:])
    ):
        audit_failures.append("cinematic_bridge_spacing_missed")
    for boundary_index, (left, right) in enumerate(zip(path, path[1:])):
        mismatch = matrices[boundary_index][left][right]
        if left != right and mismatch.severe_flags:
            if (
                mismatch.cinematic_bridge is None
                or boundary_index not in bridge_indices
                or not candidate_validity[boundary_index + 1][left]
            ):
                audit_failures.append("uncontracted_severe_edge_selected")
                break
    solver_diagnostics["postsolve_path_audit_passed"] = not audit_failures
    solver_diagnostics["postsolve_path_audit_failures"] = audit_failures
    if audit_failures:
        raise DirectMemberVisualSelectionError(
            "heterogeneous cinematic MILP witness failed independent audit",
            {
                "failure": "solver_witness_audit_failed",
                "constraint_infeasibility_proven": False,
                "visual_gates_relaxed": False,
                "search": {
                    "strategy": (
                        "scipy_highs_heterogeneous_cinematic_path_milp_v1"
                    ),
                    "solver": solver_diagnostics,
                },
            },
        )
    deviations = tuple(
        abs(actual - target) / target
        for actual, target in zip(duration_totals, duration_targets)
    )
    return (
        path,
        float(result.fun),
        {
            "strategy": "scipy_highs_heterogeneous_cinematic_path_milp_v1",
            "selection_basis": (
                "all_source_role_aware_highlight_facecam_effect_bridge"
            ),
            "usage_policy": "heterogeneous_role_aware_cinematic_v1",
            "source_count_targets": list(count_targets),
            "source_count_minimums": list(count_minimums),
            "source_count_maximums": list(count_maximums),
            "source_duration_targets_seconds": list(duration_targets),
            "source_duration_minimums_seconds": list(duration_minimums),
            "source_duration_maximums_seconds": list(duration_maximums),
            "source_duration_relative_deviations": list(deviations),
            "landscape_source_indices": list(landscape_sources),
            "portrait_source_indices": list(portrait_sources),
            "high_resolution_portrait_source_indices": list(
                high_resolution_portrait_sources
            ),
            "native_resolution_portrait_source_indices": list(
                native_resolution_portrait_sources
            ),
            "portrait_resolution_weighting": {
                "high_resolution_lossless_zoom_capacity_minimum": 1.50,
                "high_resolution_weight": 2.0,
                "native_output_resolution_weight": 1.0,
                "all_sources_minimum_usage_preserved": True,
            },
            "landscape_highlight_only": True,
            "same_source_hold_count": len(hold_indices),
            "same_source_hold_boundary_indices": list(hold_indices),
            "maximum_same_source_run_seconds": (
                maximum_same_source_run_seconds
            ),
            "measured_maximum_same_source_run_seconds": (
                measured_maximum_run_seconds
            ),
            "maximum_run_policy": (
                "one_sixteen_beat_phrase_at_126_bpm_ceiling"
            ),
            "cinematic_bridge_boundary_indices": list(bridge_indices),
            "cinematic_bridge_count": len(bridge_indices),
            "maximum_cinematic_bridge_count": (
                _DIRECT_MEMBER_CINEMATIC_BRIDGE_MAXIMUM_COUNT
            ),
            "cinematic_bridge_hard_cut_authorized": False,
            "required_transition_effect": True,
            "all_sources_used": all(count > 0 for count in counts),
            "visual_gates_relaxed": False,
            "search_capacity_exhausted": False,
            "constraint_infeasibility_proven": False,
            "solver": solver_diagnostics,
        },
    )


def _balanced_visual_path(
    matrices: Sequence[Sequence[Sequence[DirectBoundaryMismatch]]],
    *,
    source_quotas: Sequence[int],
    source_order: Sequence[int],
    candidate_validity: Sequence[Sequence[bool]],
    span_durations: Sequence[float],
    config: DirectMemberVisualConfig,
) -> tuple[tuple[int, ...], float, dict[str, Any]]:
    span_count = len(matrices) + 1
    source_count = len(source_quotas)
    order = tuple(source_order)
    durations = tuple(float(value) for value in span_durations)
    if len(durations) != span_count:
        raise ValueError("direct visual span durations must match matrices")
    if any(not math.isfinite(value) or value <= 0.0 for value in durations):
        raise ValueError("direct visual span durations must be finite and positive")
    total_duration = sum(durations)
    duration_targets = tuple(
        total_duration * int(quota) / span_count for quota in source_quotas
    )
    future_valid_counts: list[tuple[int, ...]] = [
        (0,) * source_count for _ in range(span_count + 1)
    ]
    for position in range(span_count - 1, -1, -1):
        following = future_valid_counts[position + 1]
        future_valid_counts[position] = tuple(
            following[source_index]
            + int(bool(candidate_validity[position][source_index]))
            for source_index in range(source_count)
        )
    attempts: list[dict[str, Any]] = []
    closest_complete: tuple[
        float,
        float,
        _BeamNode,
        tuple[float, ...],
        int,
    ] | None = None
    last_attempt: _BeamAttemptResult | None = None
    if _prefer_direct_member_milp(candidate_validity):
        lifeline = _exact_feasibility_lifeline(
            matrices,
            source_quotas=source_quotas,
            source_order=order,
            candidate_validity=candidate_validity,
            durations=durations,
            duration_targets=duration_targets,
            state_limit=config.feasibility_lifeline_state_limit,
            config=config,
        )
        search_diagnostics = {
            "strategy": "scipy_highs_binary_path_milp_with_counterfactual",
            "selection_basis": (
                "exact_milp_feasibility_lifeline"
                if lifeline.node is not None
                else None
            ),
            "adaptive_widening_used": False,
            "attempted_beam_widths": [],
            "attempts": [],
            "beam_skipped_for_large_exact_problem": True,
            "exact_lifeline": {"invoked": True, **lifeline.to_dict()},
            "search_capacity_exhausted": lifeline.state_capacity_exhausted,
            "constraint_infeasibility_proven": (
                lifeline.node is None and not lifeline.state_capacity_exhausted
            ),
        }
        if lifeline.node is not None:
            return (
                lifeline.node.path,
                lifeline.node.cost,
                search_diagnostics,
            )
        if lifeline.state_capacity_exhausted:
            raise DirectMemberVisualSelectionError(
                "direct visual MILP exhausted bounded solver capacity",
                {
                    "failure": "search_capacity_exhausted",
                    "source_quotas": list(source_quotas),
                    "constraint_infeasibility_proven": False,
                    "visual_gates_relaxed": False,
                    "search": search_diagnostics,
                },
            )
        search_diagnostics["severe_transition_counterfactual"] = (
            _severe_transition_counterfactual_diagnostic(
                matrices,
                source_quotas=source_quotas,
                source_order=order,
                candidate_validity=candidate_validity,
                durations=durations,
                duration_targets=duration_targets,
                state_limit=config.feasibility_lifeline_state_limit,
                config=config,
                original_lifeline=lifeline,
            )
        )
        failure_position = span_count - 1
        failure_matrix = matrices[max(0, failure_position - 1)]
        raise DirectMemberVisualSelectionError(
            "direct visual MILP proved no hard-feasible quota path",
            {
                "failure": "validity_quota_path_exhausted",
                "failed_at_segment_index": failure_position,
                "source_quotas": list(source_quotas),
                "valid_source_indices": [
                    index
                    for index, valid in enumerate(
                        candidate_validity[failure_position]
                    )
                    if valid
                ],
                "severe_edge_pair_count": sum(
                    bool(failure_matrix[left][right].severe_flags)
                    for left in range(source_count)
                    for right in range(source_count)
                    if left != right
                ),
                "severe_edges_are_ineligible": True,
                "constraint_infeasibility_proven": True,
                "search": search_diagnostics,
            },
        )
    for width in _adaptive_beam_widths(
        config.beam_width,
        config.maximum_adaptive_beam_width,
    ):
        attempt = _run_bounded_beam_attempt(
            matrices,
            source_quotas=source_quotas,
            source_order=order,
            candidate_validity=candidate_validity,
            durations=durations,
            duration_targets=duration_targets,
            future_valid_counts=future_valid_counts,
            beam_width=width,
            generated_state_limit=max(
                width,
                config.maximum_generated_beam_state_count,
            ),
            config=config,
        )
        last_attempt = attempt
        attempt_payload = attempt.to_dict()
        complete = [
            node
            for (
                counts,
                _last,
                _previous,
                _duration_state,
            ), node in attempt.states.items()
            if counts == tuple(source_quotas)
        ]
        balanced: list[_BeamNode] = []
        complete_diagnostics: list[
            tuple[float, float, _BeamNode, tuple[float, ...]]
        ] = []
        for node in complete:
            deviations = tuple(
                abs(actual - target) / target
                for actual, target in zip(
                    node.duration_totals,
                    duration_targets,
                )
            )
            maximum_deviation = max(deviations)
            squared_deviation = sum(value**2 for value in deviations)
            complete_diagnostics.append(
                (maximum_deviation, squared_deviation, node, deviations)
            )
            if (
                maximum_deviation
                <= config.maximum_source_duration_relative_deviation + 1e-9
            ):
                balanced.append(node)
        attempt_payload["complete_quota_path_count"] = len(complete)
        attempt_payload["duration_eligible_path_count"] = len(balanced)
        attempts.append(attempt_payload)
        if complete_diagnostics:
            observed = min(
                complete_diagnostics,
                key=lambda item: (
                    item[0],
                    item[1],
                    item[2].cost,
                    item[2].path,
                ),
            )
            observed_with_count = (*observed, len(complete))
            if closest_complete is None or (
                observed_with_count[0],
                observed_with_count[1],
                observed_with_count[2].cost,
                observed_with_count[2].path,
            ) < (
                closest_complete[0],
                closest_complete[1],
                closest_complete[2].cost,
                closest_complete[2].path,
            ):
                closest_complete = observed_with_count
        if balanced:
            best = min(balanced, key=lambda node: (node.cost, node.path))
            return (
                best.path,
                best.cost,
                {
                    "strategy": "bounded_adaptive_cost_beam_with_exact_lifeline",
                    "selection_basis": "adaptive_cost_beam",
                    "adaptive_widening_used": len(attempts) > 1,
                    "attempted_beam_widths": [
                        item["beam_width"] for item in attempts
                    ],
                    "attempts": attempts,
                    "exact_lifeline": {"invoked": False},
                    "search_capacity_exhausted": False,
                    "constraint_infeasibility_proven": False,
                },
            )
        if attempt.generated_state_capacity_exhausted:
            break
        # A wider cost beam cannot add states when this attempt never pruned.
        # The exact lifeline still runs because the beam duration key is
        # deliberately rounded and must not be used as an infeasibility proof.
        if attempt.pruned_state_count == 0:
            break

    lifeline = _exact_feasibility_lifeline(
        matrices,
        source_quotas=source_quotas,
        source_order=order,
        candidate_validity=candidate_validity,
        durations=durations,
        duration_targets=duration_targets,
        state_limit=config.feasibility_lifeline_state_limit,
        config=config,
    )
    search_diagnostics = {
        "strategy": "bounded_adaptive_cost_beam_with_exact_lifeline",
        "selection_basis": (
            "exact_feasibility_lifeline" if lifeline.node is not None else None
        ),
        "adaptive_widening_used": len(attempts) > 1,
        "attempted_beam_widths": [item["beam_width"] for item in attempts],
        "attempts": attempts,
        "exact_lifeline": {"invoked": True, **lifeline.to_dict()},
        "search_capacity_exhausted": lifeline.state_capacity_exhausted,
        "constraint_infeasibility_proven": (
            lifeline.node is None and not lifeline.state_capacity_exhausted
        ),
    }
    if lifeline.node is not None:
        return (
            lifeline.node.path,
            lifeline.node.cost,
            search_diagnostics,
        )
    if lifeline.state_capacity_exhausted:
        raise DirectMemberVisualSelectionError(
            "direct visual selector exhausted bounded search capacity",
            {
                "failure": "search_capacity_exhausted",
                "source_quotas": list(source_quotas),
                "constraint_infeasibility_proven": False,
                "visual_gates_relaxed": False,
                "search": search_diagnostics,
            },
        )
    search_diagnostics["severe_transition_counterfactual"] = (
        _severe_transition_counterfactual_diagnostic(
            matrices,
            source_quotas=source_quotas,
            source_order=order,
            candidate_validity=candidate_validity,
            durations=durations,
            duration_targets=duration_targets,
            state_limit=config.feasibility_lifeline_state_limit,
            config=config,
            original_lifeline=lifeline,
        )
    )
    if closest_complete is not None:
        (
            maximum_deviation,
            _squared,
            closest,
            deviations,
            complete_count,
        ) = closest_complete
        raise DirectMemberVisualSelectionError(
            "balanced direct visual selector exceeded the source-duration threshold",
            {
                "failure": "duration_balance_threshold_unsatisfied",
                "target_policy": (
                    "total_duration_times_source_quota_divided_by_span_count"
                ),
                "maximum_allowed_relative_deviation": (
                    config.maximum_source_duration_relative_deviation
                ),
                "best_maximum_relative_deviation": maximum_deviation,
                "source_usage": [
                    {
                        "eligible_source_index": source_index,
                        "quota": int(source_quotas[source_index]),
                        "duration_seconds": closest.duration_totals[source_index],
                        "target_duration_seconds": duration_targets[source_index],
                        "duration_relative_deviation": deviations[source_index],
                    }
                    for source_index in range(source_count)
                ],
                "complete_quota_path_count_in_beam": complete_count,
                "duration_threshold_effect": "fail_closed_before_render",
                "constraint_infeasibility_proven": True,
                "search": search_diagnostics,
            },
        )
    failure_position = (
        last_attempt.failed_at_segment_index
        if last_attempt is not None
        and last_attempt.failed_at_segment_index is not None
        else span_count - 1
    )
    failure_matrix = matrices[max(0, failure_position - 1)]
    raise DirectMemberVisualSelectionError(
        "balanced direct visual selector proved no hard-feasible quota path",
        {
            "failure": "validity_quota_path_exhausted",
            "failed_at_segment_index": failure_position,
            "source_quotas": list(source_quotas),
            "valid_source_indices": [
                index
                for index, valid in enumerate(candidate_validity[failure_position])
                if valid
            ],
            "severe_edge_pair_count": sum(
                bool(failure_matrix[left][right].severe_flags)
                for left in range(source_count)
                for right in range(source_count)
                if left != right
            ),
            "severe_edges_are_ineligible": True,
            "constraint_infeasibility_proven": True,
            "search": search_diagnostics,
        },
    )


def _hard_boundary_connectivity_diagnostics(
    matrices: Sequence[Sequence[Sequence[DirectBoundaryMismatch]]],
    candidate_validity: Sequence[Sequence[bool]],
) -> dict[str, Any]:
    """Summarize fail-closed source connectivity at every fixed boundary.

    This deliberately ignores quota and duration state.  It is therefore not a
    proof that a complete balanced path exists, but a zero reachable edge is a
    sound proof that the current musical partition cannot be crossed without
    relaxing either content validity, the severe-boundary gate, or the
    no-adjacent-repeat rule.  The pipeline may move that musical boundary and
    run every hard gate again; it must never turn this diagnostic into a gate
    relaxation.
    """

    validity = tuple(tuple(bool(value) for value in row) for row in candidate_validity)
    if not validity:
        return {
            "analysis_basis": "content_valid_nonsevere_no_repeat_reachability_v1",
            "boundaries": [],
            "zero_local_nonsevere_transition_boundary_indices": [],
            "zero_reachable_transition_boundary_indices": [],
        }
    source_count = len(validity[0])
    forward: list[set[int]] = [
        {index for index, valid in enumerate(validity[0]) if valid}
    ]
    for boundary_index, matrix in enumerate(matrices):
        following = {
            right
            for right in range(source_count)
            if validity[boundary_index + 1][right]
            and any(
                left != right
                and not matrix[left][right].severe_flags
                for left in forward[-1]
            )
        }
        forward.append(following)

    backward: list[set[int]] = [set() for _ in validity]
    backward[-1] = {
        index for index, valid in enumerate(validity[-1]) if valid
    }
    for boundary_index in range(len(matrices) - 1, -1, -1):
        matrix = matrices[boundary_index]
        backward[boundary_index] = {
            left
            for left in range(source_count)
            if validity[boundary_index][left]
            and any(
                left != right
                and not matrix[left][right].severe_flags
                for right in backward[boundary_index + 1]
            )
        }

    boundaries: list[dict[str, Any]] = []
    for boundary_index, matrix in enumerate(matrices):
        content_valid_edges = [
            (left, right)
            for left in range(source_count)
            for right in range(source_count)
            if left != right
            and validity[boundary_index][left]
            and validity[boundary_index + 1][right]
        ]
        nonsevere_edges = [
            (left, right)
            for left, right in content_valid_edges
            if not matrix[left][right].severe_flags
        ]
        reachable_edges = [
            (left, right)
            for left, right in nonsevere_edges
            if left in forward[boundary_index]
            and right in backward[boundary_index + 1]
        ]
        boundaries.append(
            {
                "boundary_index": boundary_index,
                "left_valid_source_indices": [
                    index
                    for index, valid in enumerate(validity[boundary_index])
                    if valid
                ],
                "right_valid_source_indices": [
                    index
                    for index, valid in enumerate(validity[boundary_index + 1])
                    if valid
                ],
                "forward_reachable_left_source_indices": sorted(
                    forward[boundary_index]
                ),
                "forward_reachable_right_source_indices": sorted(
                    forward[boundary_index + 1]
                ),
                "backward_reachable_right_source_indices": sorted(
                    backward[boundary_index + 1]
                ),
                "content_valid_cross_source_edge_count": len(content_valid_edges),
                "nonsevere_cross_source_edge_count": len(nonsevere_edges),
                "reachable_nonsevere_transition_count": len(reachable_edges),
            }
        )
    return {
        "analysis_basis": "content_valid_nonsevere_no_repeat_reachability_v1",
        "quota_and_duration_considered": False,
        "zero_local_nonsevere_edge_is_partition_infeasibility_proof": True,
        "global_reachable_edge_counts_are_localization_diagnostics_only": True,
        "boundaries": boundaries,
        "zero_local_nonsevere_transition_boundary_indices": [
            item["boundary_index"]
            for item in boundaries
            if item["nonsevere_cross_source_edge_count"] == 0
        ],
        "zero_reachable_transition_boundary_indices": [
            item["boundary_index"]
            for item in boundaries
            if item["reachable_nonsevere_transition_count"] == 0
        ],
    }


def _hard_boundary_forward_exhaustion_boundary(
    connectivity: Mapping[str, Any],
) -> int | None:
    """Return the first exact quota-independent forward path extinction."""

    if connectivity.get("analysis_basis") != (
        "content_valid_nonsevere_no_repeat_reachability_v1"
    ):
        return None
    boundaries = connectivity.get("boundaries")
    if not isinstance(boundaries, list):
        return None
    for expected_index, item in enumerate(boundaries):
        if not isinstance(item, dict) or item.get("boundary_index") != expected_index:
            return None
        left = item.get("forward_reachable_left_source_indices")
        right = item.get("forward_reachable_right_source_indices")
        if not isinstance(left, list) or not isinstance(right, list):
            return None
        if left and not right:
            return expected_index
    return None


def select_balanced_direct_member_sources(
    *,
    sources: Sequence[Path],
    canonical_spans: Sequence[tuple[float, float]],
    audio_offsets: Sequence[float],
    source_timewarps: Sequence[TimeWarpMap | None] | None = None,
    source_scene_cut_times: Sequence[Sequence[float]] | None = None,
    source_frame_sizes: Sequence[tuple[int, int]] | None = None,
    output_frame_size: tuple[int, int] = (1080, 1920),
    continuous_official_portrait_fancam_contract: bool = False,
    cinematic_bridge_enabled: bool = False,
    preferred_source_order: Sequence[int] | None = None,
    extra_quota_source_indices: Sequence[int] | None = None,
    priority_cut_boundary_indices: Sequence[int] = (),
    maximum_priority_cut_hold_count: int = 0,
    config: DirectMemberVisualConfig | None = None,
    descriptor_provider: DescriptorProvider | None = None,
) -> DirectMemberVisualSelection:
    """Select a balanced visual path over already-fixed musical spans.

    Every eligible source is decoded immediately before and after every coarse
    boundary. Source-local sample times and source-native scene cuts are mapped
    through the accepted ``TimeWarpMap`` when present and retain scalar offsets
    otherwise. Any native cut that survives strictly inside a coarsened span is
    a hard-invalid source/span candidate even when six sampled frames miss it.
    The exact verified continuous official portrait-fancam contract may suppress
    only a raw detector event whose non-identifying flank evidence proves a
    continuous performance; generic/manual callers retain the strict policy.
    ``extra_quota_source_indices`` changes only which sources receive the
    remainder slots of an otherwise exactly balanced quota.  It never changes
    source search order, content validity, severe-edge, or duration gates.
    ``priority_cut_boundary_indices`` identifies section-critical cuts whose
    no-cut continuations are separately capped by
    ``maximum_priority_cut_hold_count``.
    ``source_frame_sizes`` authorizes geometry repair only by proving that the
    post-portrait-fill working frame remains no larger than decoded source
    pixels. Missing sizes retain the historical no-repair hard gates.
    ``cinematic_bridge_enabled`` authorizes a terminal role-aware allocation
    for heterogeneous official fancams only after strict balanced and bounded
    no-cut paths are proven infeasible.  Its framing-only severe edges remain
    forbidden as hard cuts and must be materialized as audited effects.
    """

    resolved_sources = tuple(Path(source) for source in sources)
    if not isinstance(continuous_official_portrait_fancam_contract, bool):
        raise TypeError(
            "continuous_official_portrait_fancam_contract must be boolean"
        )
    if not isinstance(cinematic_bridge_enabled, bool):
        raise TypeError("cinematic_bridge_enabled must be boolean")
    source_count = len(resolved_sources)
    if source_count < 2:
        raise ValueError("direct visual selection needs at least two sources")
    if len(output_frame_size) != 2:
        raise ValueError("direct visual output frame size needs width and height")
    output_width, output_height = output_frame_size
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        for value in (output_width, output_height)
    ):
        raise ValueError("direct visual output frame size is invalid")
    if source_frame_sizes is None:
        source_frame_geometries: tuple[
            DirectMemberSourceFrameGeometry | None,
            ...,
        ] = (None,) * source_count
    else:
        if len(source_frame_sizes) != source_count:
            raise ValueError("direct visual source frame sizes must match sources")
        source_frame_geometries = tuple(
            direct_member_source_frame_geometry(
                size,
                output_size=(output_width, output_height),
            )
            for size in source_frame_sizes
        )
    spans = tuple((float(start), float(end)) for start, end in canonical_spans)
    if len(spans) < source_count:
        raise ValueError("direct visual selection needs at least one span per source")
    for index, (start, end) in enumerate(spans):
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError("direct visual spans must be finite and positive")
        if index and not math.isclose(
            spans[index - 1][1],
            start,
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            raise ValueError("direct visual spans must be contiguous")
    priority_boundaries = tuple(
        int(value) for value in priority_cut_boundary_indices
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < len(spans) - 1
        for value in priority_cut_boundary_indices
    ):
        raise ValueError("priority cut boundary indices are invalid")
    if len(set(priority_boundaries)) != len(priority_boundaries):
        raise ValueError("priority cut boundary indices must be unique")
    if (
        isinstance(maximum_priority_cut_hold_count, bool)
        or not isinstance(maximum_priority_cut_hold_count, int)
        or maximum_priority_cut_hold_count < 0
    ):
        raise ValueError("maximum priority cut hold count is invalid")
    if len(audio_offsets) != source_count:
        raise ValueError("direct visual audio_offsets must match sources")
    offsets = tuple(float(value) for value in audio_offsets)
    if any(not math.isfinite(value) for value in offsets):
        raise ValueError("direct visual audio offsets must be finite")
    clocks = (
        (None,) * source_count
        if source_timewarps is None
        else tuple(source_timewarps)
    )
    if len(clocks) != source_count:
        raise ValueError("direct visual source_timewarps must match sources")
    for index, timewarp in enumerate(clocks):
        if timewarp is None:
            continue
        if not isinstance(timewarp, TimeWarpMap):
            raise TypeError("direct visual source_timewarps entries must be maps")
        if not math.isclose(
            timewarp.global_offset_seconds,
            offsets[index],
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("direct visual timewarp offset does not match audio offset")
    scene_cut_evidence_available = source_scene_cut_times is not None
    local_scene_cuts = (
        ((),) * source_count
        if source_scene_cut_times is None
        else tuple(tuple(float(value) for value in cuts) for cuts in source_scene_cut_times)
    )
    if len(local_scene_cuts) != source_count:
        raise ValueError("direct visual scene cut times must match sources")
    for source_index, cuts in enumerate(local_scene_cuts):
        if any(not math.isfinite(value) or value < 0.0 for value in cuts):
            raise ValueError("direct visual scene cut times must be finite and non-negative")
        if any(left >= right for left, right in zip(cuts, cuts[1:])):
            raise ValueError(
                f"direct visual scene cut times for source {source_index} "
                "must be strictly increasing"
            )
    mapped_scene_cuts = tuple(
        tuple(
            (
                source_seconds,
                _source_master_time(
                    source_seconds,
                    source_index=source_index,
                    audio_offsets=offsets,
                    source_timewarps=clocks,
                ),
            )
            for source_seconds in cuts
        )
        for source_index, cuts in enumerate(local_scene_cuts)
    )
    order = (
        tuple(range(source_count))
        if preferred_source_order is None
        else tuple(int(index) for index in preferred_source_order)
    )
    if len(order) != source_count or set(order) != set(range(source_count)):
        raise ValueError("preferred_source_order must be a source permutation")
    _base_quota, extra_quota_count = divmod(len(spans), source_count)
    if extra_quota_source_indices is None:
        quota_order = order
        explicit_extra_quota_sources = tuple(order[:extra_quota_count])
        quota_assignment_basis = "preferred_source_order_default"
    else:
        explicit_extra_quota_sources = tuple(
            int(index) for index in extra_quota_source_indices
        )
        if (
            len(explicit_extra_quota_sources) != extra_quota_count
            or len(set(explicit_extra_quota_sources))
            != len(explicit_extra_quota_sources)
            or any(
                not 0 <= index < source_count
                for index in explicit_extra_quota_sources
            )
        ):
            raise ValueError(
                "extra_quota_source_indices must contain exactly the balanced "
                "remainder source set"
            )
        explicit_set = set(explicit_extra_quota_sources)
        quota_order = explicit_extra_quota_sources + tuple(
            index for index in order if index not in explicit_set
        )
        quota_assignment_basis = "explicit_extra_quota_source_set"
    resolved_config = config or DirectMemberVisualConfig()
    quotas = _balanced_quotas(len(spans), source_count, quota_order)
    heterogeneous_cinematic_authorized = bool(
        cinematic_bridge_enabled
        and continuous_official_portrait_fancam_contract
        and all(geometry is not None for geometry in source_frame_geometries)
        and any(
            geometry is not None
            and geometry.source_width > geometry.source_height
            for geometry in source_frame_geometries
        )
        and any(
            geometry is not None
            and geometry.source_width <= geometry.source_height
            for geometry in source_frame_geometries
        )
    )
    quota_assignment_provenance = {
        "contract": "strict_balanced_extra_quota_assignment_v1",
        "basis": quota_assignment_basis,
        "span_count": len(spans),
        "source_count": source_count,
        "base_quota": _base_quota,
        "extra_quota_count": extra_quota_count,
        "extra_quota_source_indices": list(explicit_extra_quota_sources),
        "source_search_order": list(order),
        "source_search_order_changed": False,
        "counts_differ_by_at_most_one": True,
    }

    owned_provider: _VideoDescriptorProvider | None = None
    provider = descriptor_provider
    if provider is None:
        owned_provider = _VideoDescriptorProvider(resolved_sources)
        provider = owned_provider
    scene_cut_artifact_cache: dict[tuple[int, int], bool] = {}

    def _scene_cut_is_transient_portrait_fancam_artifact(
        *,
        source_index: int,
        source_seconds: float,
        master_seconds: float,
    ) -> bool:
        if not continuous_official_portrait_fancam_contract:
            return False
        key = (source_index, int(round(source_seconds * 1_000_000.0)))
        cached = scene_cut_artifact_cache.get(key)
        if cached is not None:
            return cached
        far = max(
            resolved_config.portrait_fancam_scene_cut_continuity_window_seconds,
            3.0 * resolved_config.sample_frame_step_seconds,
        )
        near = resolved_config.sample_frame_step_seconds
        probe_master_seconds = (
            master_seconds - far,
            master_seconds - near,
            master_seconds,
            master_seconds + near,
            master_seconds + far,
        )
        probe_source_seconds = tuple(
            (
                source_seconds
                if index == 2
                else _source_local_time(
                    probe_master,
                    source_index=source_index,
                    audio_offsets=offsets,
                    source_timewarps=clocks,
                )
            )
            for index, probe_master in enumerate(probe_master_seconds)
        )
        descriptors = tuple(
            provider(
                source_index,
                resolved_sources[source_index],
                max(0.0, local_seconds),
            )
            for local_seconds in probe_source_seconds
        )
        result = _continuous_portrait_fancam_scene_cut_is_transient_artifact(
            descriptors,  # type: ignore[arg-type]
            resolved_config,
        )
        scene_cut_artifact_cache[key] = result
        return result
    boundary_search_candidates: list[
        tuple[_DirectBoundarySearchCandidate, ...]
    ] = []
    matrices: list[tuple[tuple[DirectBoundaryMismatch, ...], ...]] = []
    span_diagnostics: list[DirectSpanVisualDiagnostic] = []
    sample_count = 0
    failed_sample_count = 0
    try:
        for segment_index, (span_start, span_end) in enumerate(spans):
            duration = span_end - span_start
            exit_inset = min(
                resolved_config.span_interior_inset_seconds,
                duration * 0.20,
            )
            # The first content sample is deliberately the actual cut plus one
            # output frame.  A rig that clears 0.2 seconds later is still a
            # visible bad cut and must not be hidden by the deeper interior
            # sample used historically.
            entry_inset = min(
                resolved_config.sample_frame_step_seconds,
                duration * 0.08,
            )
            # The second sample distinguishes a one-frame flash from a stage
            # light or near-lens obstruction that visibly persists after the
            # cut but clears before the quarter-span sample.
            early_interior_inset = min(
                resolved_config.early_interior_sample_seconds,
                duration * 0.18,
            )
            master_times = (
                span_start + entry_inset,
                span_start + early_interior_inset,
                span_start + duration * 0.25,
                (span_start + span_end) / 2.0,
                span_start + duration * 0.75,
                span_end - exit_inset,
            )
            source_validity: list[DirectSpanSourceValidity] = []
            for source_index, source in enumerate(resolved_sources):
                cut_tolerance = min(
                    resolved_config.native_scene_cut_boundary_tolerance_seconds,
                    duration * 0.20,
                )
                internal_native_cuts = tuple(
                    (source_seconds, master_seconds)
                    for source_seconds, master_seconds in mapped_scene_cuts[source_index]
                    if span_start + cut_tolerance
                    < master_seconds
                    < span_end - cut_tolerance
                )
                suppressed_portrait_fancam_scene_cuts = tuple(
                    cut
                    for cut in internal_native_cuts
                    if _scene_cut_is_transient_portrait_fancam_artifact(
                        source_index=source_index,
                        source_seconds=cut[0],
                        master_seconds=cut[1],
                    )
                )
                confirmed_internal_native_cuts = tuple(
                    cut
                    for cut in internal_native_cuts
                    if cut not in suppressed_portrait_fancam_scene_cuts
                )
                samples: list[_BoundarySample] = []
                for master_seconds in master_times:
                    local_seconds = _source_local_time(
                        master_seconds,
                        source_index=source_index,
                        audio_offsets=offsets,
                        source_timewarps=clocks,
                    )
                    descriptor = provider(source_index, source, local_seconds)
                    sample_count += 1
                    failed_sample_count += int(descriptor is None)
                    samples.append(
                        _BoundarySample(
                            master_seconds=master_seconds,
                            source_local_seconds=local_seconds,
                            descriptor=descriptor,
                            attention=_attention_geometry(descriptor),
                        )
                    )
                source_validity.append(
                    _span_source_validity(
                        source_index=source_index,
                        samples=tuple(samples),  # type: ignore[arg-type]
                        native_scene_cuts=confirmed_internal_native_cuts,
                        suppressed_portrait_fancam_scene_cuts=(
                            suppressed_portrait_fancam_scene_cuts
                        ),
                        continuous_official_portrait_fancam_contract=(
                            continuous_official_portrait_fancam_contract
                        ),
                        config=resolved_config,
                    )
                )
            span_diagnostics.append(
                DirectSpanVisualDiagnostic(
                    segment_index=segment_index,
                    canonical_master_start=span_start,
                    canonical_master_end=span_end,
                    sources=tuple(source_validity),
                )
            )

        frame_offsets = _boundary_search_frame_offsets(resolved_config)
        for boundary_index in range(len(spans) - 1):
            canonical_boundary = spans[boundary_index][1]
            candidates: list[_DirectBoundarySearchCandidate] = []
            for offset_frames in frame_offsets:
                boundary = (
                    canonical_boundary
                    + offset_frames
                    * resolved_config.sample_frame_step_seconds
                )
                outgoing_master = max(
                    spans[boundary_index][0],
                    boundary - resolved_config.sample_frame_step_seconds,
                )
                outgoing_previous_master = max(
                    spans[boundary_index][0],
                    boundary
                    - 2.0 * resolved_config.sample_frame_step_seconds,
                )
                # A hard cut renders the incoming segment's first frame at the
                # selected boundary itself.  Sampling boundary+1 frame here
                # used to hide the exact entry image that viewers actually
                # see, and could approve a visually disconnected cut when the
                # following frame happened to match.  Bind the audit to the
                # real outgoing-last/incoming-first frame pair.
                incoming_master = boundary
                incoming_next_master = min(
                    spans[boundary_index + 1][1],
                    boundary
                    + resolved_config.sample_frame_step_seconds,
                )
                if not (
                    outgoing_previous_master < outgoing_master < boundary
                    and math.isclose(
                        incoming_master,
                        boundary,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    and incoming_master < incoming_next_master
                ):
                    continue
                outgoing_previous: list[_BoundarySample] = []
                outgoing: list[_BoundarySample] = []
                incoming: list[_BoundarySample] = []
                incoming_next: list[_BoundarySample] = []
                for source_index, source in enumerate(resolved_sources):
                    master_times = (
                        outgoing_previous_master,
                        outgoing_master,
                        incoming_master,
                        incoming_next_master,
                    )
                    local_times = tuple(
                        _source_local_time(
                            master_seconds,
                            source_index=source_index,
                            audio_offsets=offsets,
                            source_timewarps=clocks,
                        )
                        for master_seconds in master_times
                    )
                    descriptors = tuple(
                        provider(source_index, source, local_seconds)
                        for local_seconds in local_times
                    )
                    sample_count += 4
                    failed_sample_count += sum(
                        descriptor is None for descriptor in descriptors
                    )
                    samples = tuple(
                        _BoundarySample(
                            master_seconds=master_seconds,
                            source_local_seconds=local_seconds,
                            descriptor=descriptor,
                            attention=_attention_geometry(descriptor),
                        )
                        for master_seconds, local_seconds, descriptor in zip(
                            master_times,
                            local_times,
                            descriptors,
                            strict=True,
                        )
                    )
                    outgoing_previous.append(samples[0])
                    outgoing.append(samples[1])
                    incoming.append(samples[2])
                    incoming_next.append(samples[3])
                outgoing_previous_tuple = tuple(outgoing_previous)
                outgoing_tuple = tuple(outgoing)
                incoming_tuple = tuple(incoming)
                incoming_next_tuple = tuple(incoming_next)
                candidate_matrix = tuple(
                    tuple(
                        _boundary_mismatch(
                            outgoing_tuple[left_index],
                            incoming_tuple[right_index],
                            resolved_config,
                            outgoing_previous=(
                                outgoing_previous_tuple[left_index]
                            ),
                            incoming_next=incoming_next_tuple[right_index],
                            outgoing_source_index=left_index,
                            incoming_source_index=right_index,
                            outgoing_source_geometry=(
                                source_frame_geometries[left_index]
                            ),
                            incoming_source_geometry=(
                                source_frame_geometries[right_index]
                            ),
                            continuous_official_portrait_fancam_contract=(
                                continuous_official_portrait_fancam_contract
                            ),
                        )
                        for right_index in range(source_count)
                    )
                    for left_index in range(source_count)
                )
                candidates.append(
                    _DirectBoundarySearchCandidate(
                        offset_frames=offset_frames,
                        cut_master_seconds=boundary,
                        outgoing_previous=outgoing_previous_tuple,
                        outgoing=outgoing_tuple,
                        incoming=incoming_tuple,
                        incoming_next=incoming_next_tuple,
                        matrix=candidate_matrix,
                    )
                )
            if not candidates or not any(
                candidate.offset_frames == 0 for candidate in candidates
            ):
                raise RuntimeError(
                    "boundary local search did not retain its canonical state"
                )
            candidate_tuple = tuple(candidates)
            boundary_search_candidates.append(candidate_tuple)
            matrices.append(
                tuple(
                    tuple(
                        _best_boundary_search_edge(
                            candidate_tuple,
                            outgoing_source_index=left_index,
                            incoming_source_index=right_index,
                            priority_boundary=(
                                boundary_index in priority_boundaries
                            ),
                            config=resolved_config,
                        )[1]
                        for right_index in range(source_count)
                    )
                    for left_index in range(source_count)
                )
            )
    finally:
        if owned_provider is not None:
            owned_provider.close()

    candidate_validity = tuple(
        tuple(source.valid for source in span.sources)
        for span in span_diagnostics
    )
    hard_boundary_connectivity = _hard_boundary_connectivity_diagnostics(
        matrices,
        candidate_validity,
    )
    valid_counts = tuple(
        sum(span[source_index] for span in candidate_validity)
        for source_index in range(source_count)
    )
    valid_count_by_source = [
        {
            "eligible_source_index": source_index,
            "quota": int(quotas[source_index]),
            "valid_span_count": int(valid_counts[source_index]),
            "invalid_span_count": int(len(spans) - valid_counts[source_index]),
            "valid_segment_indices": [
                segment_index
                for segment_index, validity in enumerate(candidate_validity)
                if validity[source_index]
            ],
        }
        for source_index in range(source_count)
    ]
    quota_shortfalls = tuple(
        {
            "eligible_source_index": source_index,
            "quota": quotas[source_index],
            "valid_span_count": valid_counts[source_index],
        }
        for source_index in range(source_count)
        if valid_counts[source_index] < quotas[source_index]
    )
    empty_spans = tuple(
        index for index, validity in enumerate(candidate_validity) if not any(validity)
    )
    if empty_spans or (
        quota_shortfalls and not heterogeneous_cinematic_authorized
    ):
        raise DirectMemberVisualSelectionError(
            "direct member content validity cannot satisfy balanced source quotas",
            {
                "failure": "content_validity_quota_precheck_failed",
                "quota_shortfalls": list(quota_shortfalls),
                "empty_segment_indices": list(empty_spans),
                "span_content_validity": [
                    span.to_dict() for span in span_diagnostics
                ],
                "valid_count_by_source": valid_count_by_source,
                "hard_boundary_connectivity": hard_boundary_connectivity,
                "source_quota_assignment": quota_assignment_provenance,
            },
        )
    forward_exhaustion_boundary = (
        _hard_boundary_forward_exhaustion_boundary(
            hard_boundary_connectivity
        )
        if resolved_config.maximum_same_source_hold_count == 0
        else None
    )
    if (
        forward_exhaustion_boundary is not None
        and not heterogeneous_cinematic_authorized
    ):
        failed_segment_index = forward_exhaustion_boundary + 1
        failure_matrix = matrices[forward_exhaustion_boundary]
        raise DirectMemberVisualSelectionError(
            "direct member transition graph is disconnected before optimization",
            {
                "failure": "boundary_connectivity_path_exhausted",
                "failed_at_segment_index": failed_segment_index,
                "source_quotas": list(quotas),
                "valid_source_indices": [
                    index
                    for index, valid in enumerate(
                        candidate_validity[failed_segment_index]
                    )
                    if valid
                ],
                "severe_edge_pair_count": sum(
                    bool(failure_matrix[left][right].severe_flags)
                    for left in range(source_count)
                    for right in range(source_count)
                    if left != right
                ),
                "severe_edges_are_ineligible": True,
                "constraint_infeasibility_proven": True,
                "visual_gates_relaxed": False,
                "search": {
                    "strategy": (
                        "exact_quota_independent_nonsevere_connectivity_precheck"
                    ),
                    "selection_basis": None,
                    "adaptive_widening_used": False,
                    "attempted_beam_widths": [],
                    "attempts": [],
                    "exact_lifeline": {
                        "invoked": False,
                        "skipped_by_stronger_quota_independent_proof": True,
                        "path_found": False,
                        "state_capacity_exhausted": False,
                        "duration_state_quantization_applied": False,
                    },
                    "search_capacity_exhausted": False,
                    "constraint_infeasibility_proven": True,
                },
                "span_content_validity": [
                    span.to_dict() for span in span_diagnostics
                ],
                "valid_count_by_source": valid_count_by_source,
                "hard_boundary_connectivity": hard_boundary_connectivity,
                "source_quota_assignment": quota_assignment_provenance,
            },
        )
    usage_policy = "exact_balanced_quota_v1"
    selected_source_targets = tuple(quotas)
    selected_count_minimums: tuple[int, ...] = ()
    selected_count_maximums: tuple[int, ...] = ()
    selected_duration_targets: tuple[float, ...] | None = None
    selected_duration_minimums: tuple[float, ...] = ()
    selected_duration_maximums: tuple[float, ...] = ()
    try:
        path, optimizer_cost, optimizer_search_diagnostics = (
            _balanced_visual_path(
                matrices,
                source_quotas=quotas,
                source_order=order,
                candidate_validity=candidate_validity,
                span_durations=tuple(end - start for start, end in spans),
                config=resolved_config,
            )
        )
    except DirectMemberVisualSelectionError as error:
        strict_search = error.diagnostics.get("search")
        strict_counterfactual = (
            strict_search.get("severe_transition_counterfactual")
            if isinstance(strict_search, Mapping)
            else None
        )
        strict_exact_lifeline = (
            strict_search.get("exact_lifeline")
            if isinstance(strict_search, Mapping)
            else None
        )
        hold_fallback_authorized = (
            error.diagnostics.get("failure")
            == "validity_quota_path_exhausted"
            and error.diagnostics.get("constraint_infeasibility_proven") is True
            and isinstance(strict_exact_lifeline, Mapping)
            and strict_exact_lifeline.get("strategy")
            in {
                "exact_backward_memoized_hard_feasibility",
                "scipy_highs_binary_path_milp_v1",
            }
            and strict_exact_lifeline.get("path_found") is False
            and strict_exact_lifeline.get("state_capacity_exhausted") is False
            and strict_exact_lifeline.get("duration_state_quantization_applied")
            is False
        )
        hold_fallback = (
            _bounded_same_source_hold_visual_path(
                matrices,
                source_quotas=quotas,
                source_order=order,
                candidate_validity=candidate_validity,
                span_durations=tuple(end - start for start, end in spans),
                priority_cut_boundary_indices=priority_boundaries,
                maximum_priority_cut_hold_count=(
                    maximum_priority_cut_hold_count
                ),
                config=resolved_config,
            )
            if hold_fallback_authorized
            else None
        )
        strict_failure_summary = {
            "failure": error.diagnostics.get("failure"),
            "constraint_infeasibility_proven": error.diagnostics.get(
                "constraint_infeasibility_proven"
            ),
            "minimum_severe_transition_count": (
                strict_counterfactual.get("minimum_severe_transition_count")
                if isinstance(strict_counterfactual, Mapping)
                else None
            ),
            "exact_lifeline_strategy": (
                strict_exact_lifeline.get("strategy")
                if isinstance(strict_exact_lifeline, Mapping)
                else None
            ),
            "exact_lifeline_state_capacity_exhausted": (
                strict_exact_lifeline.get("state_capacity_exhausted")
                if isinstance(strict_exact_lifeline, Mapping)
                else None
            ),
            "diagnostic_sha256": _canonical_diagnostic_sha256(
                error.diagnostics
            ),
        }
        if hold_fallback is not None:
            path, optimizer_cost, optimizer_search_diagnostics = hold_fallback
            optimizer_search_diagnostics = {
                **optimizer_search_diagnostics,
                "strict_cross_source_failure": {
                    **strict_failure_summary,
                    "authorization_basis": (
                        "proven_strict_no_repeat_infeasibility_then_bounded_"
                        "same_source_no_cut_search_v1"
                    ),
                },
            }
        else:
            heterogeneous_fallback_authorized = bool(
                hold_fallback_authorized
                and heterogeneous_cinematic_authorized
            )
            heterogeneous_fallback = (
                _heterogeneous_cinematic_visual_path(
                    matrices,
                    source_order=order,
                    candidate_validity=candidate_validity,
                    span_durations=tuple(
                        end - start for start, end in spans
                    ),
                    source_geometries=source_frame_geometries,
                    priority_cut_boundary_indices=priority_boundaries,
                    maximum_priority_cut_hold_count=(
                        maximum_priority_cut_hold_count
                    ),
                    config=resolved_config,
                )
                if heterogeneous_fallback_authorized
                else None
            )
            if heterogeneous_fallback is not None:
                (
                    path,
                    optimizer_cost,
                    optimizer_search_diagnostics,
                ) = heterogeneous_fallback
                usage_policy = "heterogeneous_role_aware_cinematic_v1"
                selected_source_targets = tuple(
                    int(value)
                    for value in optimizer_search_diagnostics[
                        "source_count_targets"
                    ]
                )
                selected_count_minimums = tuple(
                    int(value)
                    for value in optimizer_search_diagnostics[
                        "source_count_minimums"
                    ]
                )
                selected_count_maximums = tuple(
                    int(value)
                    for value in optimizer_search_diagnostics[
                        "source_count_maximums"
                    ]
                )
                selected_duration_targets = tuple(
                    float(value)
                    for value in optimizer_search_diagnostics[
                        "source_duration_targets_seconds"
                    ]
                )
                selected_duration_minimums = tuple(
                    float(value)
                    for value in optimizer_search_diagnostics[
                        "source_duration_minimums_seconds"
                    ]
                )
                selected_duration_maximums = tuple(
                    float(value)
                    for value in optimizer_search_diagnostics[
                        "source_duration_maximums_seconds"
                    ]
                )
                optimizer_search_diagnostics = {
                    **optimizer_search_diagnostics,
                    "strict_cross_source_failure": {
                        **strict_failure_summary,
                        "authorization_basis": (
                            "proven_strict_and_bounded_safe_hold_"
                            "infeasibility_then_required_effect_bridge_v1"
                        ),
                    },
                }
            else:
                # Persist the already-computed scalar evidence with every
                # optimizer failure. This is intentionally added here, where
                # the complete validity diagnostics still exist, so a failed
                # run never needs another expensive descriptor pass merely to
                # diagnose feasibility.
                diagnostics = dict(error.diagnostics)
                diagnostics.setdefault(
                    "span_content_validity",
                    [span.to_dict() for span in span_diagnostics],
                )
                diagnostics.setdefault(
                    "valid_count_by_source", valid_count_by_source
                )
                diagnostics.setdefault(
                    "hard_boundary_connectivity",
                    hard_boundary_connectivity,
                )
                diagnostics.setdefault(
                    "source_quota_assignment",
                    quota_assignment_provenance,
                )
                diagnostics.setdefault(
                    "heterogeneous_cinematic_fallback",
                    {
                        "requested": cinematic_bridge_enabled,
                        "continuous_official_contract": (
                            continuous_official_portrait_fancam_contract
                        ),
                        "complete_source_geometry": all(
                            geometry is not None
                            for geometry in source_frame_geometries
                        ),
                        "landscape_source_indices": [
                            index
                            for index, geometry in enumerate(
                                source_frame_geometries
                            )
                            if geometry is not None
                            and geometry.source_width > geometry.source_height
                        ],
                        "portrait_source_indices": [
                            index
                            for index, geometry in enumerate(
                                source_frame_geometries
                            )
                            if geometry is not None
                            and geometry.source_width <= geometry.source_height
                        ],
                        "strict_failure_authorized_safe_hold": (
                            hold_fallback_authorized
                        ),
                        "authorized": heterogeneous_fallback_authorized,
                        "attempted": heterogeneous_fallback_authorized,
                        "path_found": False,
                        "cinematic_bridge_candidate_edge_count": sum(
                            mismatch.cinematic_bridge is not None
                            for matrix in matrices
                            for row in matrix
                            for mismatch in row
                        ),
                        "priority_cinematic_bridge_candidate_edge_count": sum(
                            mismatch.cinematic_bridge is not None
                            for boundary_index, matrix in enumerate(matrices)
                            if boundary_index in priority_boundaries
                            for row in matrix
                            for mismatch in row
                        ),
                    },
                )
                diagnostics.setdefault(
                    "privacy",
                    {
                        "face_identity_inference_used": False,
                        "face_appearance_embeddings_persisted": False,
                        "raw_pose_silhouette_layout_embeddings_persisted": False,
                        "validity_evidence_is_scalar_only": True,
                    },
                )
                raise DirectMemberVisualSelectionError(
                    str(error),
                    diagnostics,
                ) from error
    optimizer_search_diagnostics = {
        **optimizer_search_diagnostics,
        "hard_boundary_connectivity": hard_boundary_connectivity,
        "source_quota_assignment": quota_assignment_provenance,
        "source_usage_policy": usage_policy,
        "heterogeneous_cinematic_authorized": (
            heterogeneous_cinematic_authorized
        ),
        "priority_cut_boundary_indices": list(priority_boundaries),
        "maximum_priority_cut_hold_count": maximum_priority_cut_hold_count,
    }
    priority_hold_indices = tuple(
        boundary_index
        for boundary_index in priority_boundaries
        if path[boundary_index] == path[boundary_index + 1]
    )
    if len(priority_hold_indices) > maximum_priority_cut_hold_count:
        raise RuntimeError(
            "direct member visual path exceeded priority cut hold limit"
        )
    base_span_durations = tuple(end - start for start, end in spans)
    if selected_duration_targets is None:
        _base_actual, joint_duration_targets, _base_deviations = (
            _source_duration_balance(
                path,
                base_span_durations,
                selected_source_targets,
            )
        )
        joint_duration_minimums = tuple(
            target
            * (1.0 - resolved_config.maximum_source_duration_relative_deviation)
            for target in joint_duration_targets
        )
        joint_duration_maximums = tuple(
            target
            * (1.0 + resolved_config.maximum_source_duration_relative_deviation)
            for target in joint_duration_targets
        )
    else:
        joint_duration_targets = selected_duration_targets
        joint_duration_minimums = selected_duration_minimums
        joint_duration_maximums = selected_duration_maximums
    (
        selected_boundary_candidates,
        selected_boundary_mismatches,
        boundary_search_diagnostics,
    ) = _joint_boundary_search_assignment(
        candidates_by_boundary=boundary_search_candidates,
        path=path,
        spans=spans,
        source_duration_targets=joint_duration_targets,
        source_duration_minimums=joint_duration_minimums,
        source_duration_maximums=joint_duration_maximums,
        usage_policy=usage_policy,
        priority_boundary_indices=frozenset(priority_boundaries),
        config=resolved_config,
    )
    selected_boundary_master_seconds = tuple(
        candidate.cut_master_seconds
        for candidate in selected_boundary_candidates
    )
    selected_span_durations = _boundary_adjusted_span_durations(
        spans,
        selected_boundary_candidates,
    )
    local_search_cost_delta = sum(
        selected_mismatch.cost
        - matrices[boundary_index][left][right].cost
        for boundary_index, (
            left,
            right,
            selected_mismatch,
        ) in enumerate(
            zip(
                path[:-1],
                path[1:],
                selected_boundary_mismatches,
                strict=True,
            )
        )
        if left != right
    )
    optimizer_cost += local_search_cost_delta
    optimizer_search_diagnostics = {
        **optimizer_search_diagnostics,
        "boundary_local_search": {
            **boundary_search_diagnostics,
            "selected_cost_delta_after_joint_duration_assignment": (
                local_search_cost_delta
            ),
        },
    }
    current_source = path[0]
    current_run_seconds = selected_span_durations[0]
    maximum_selected_run_seconds = float(
        optimizer_search_diagnostics.get(
            "maximum_same_source_run_seconds",
            resolved_config.maximum_same_source_run_seconds,
        )
    )
    for position, source_index in enumerate(path[1:], start=1):
        duration = selected_span_durations[position]
        if source_index == current_source:
            current_run_seconds += duration
        else:
            current_source = source_index
            current_run_seconds = duration
        if (
            current_run_seconds
            > maximum_selected_run_seconds + 1e-9
        ):
            raise RuntimeError(
                "direct member visual path exceeded same-source run limit"
            )
    counts = tuple(path.count(index) for index in range(source_count))
    if selected_duration_targets is None:
        (
            source_durations,
            source_duration_targets,
            source_duration_deviations,
        ) = _source_duration_balance(
            path,
            selected_span_durations,
            selected_source_targets,
        )
    else:
        source_durations = tuple(
            sum(
                selected_span_durations[position]
                for position, selected_source in enumerate(path)
                if selected_source == source_index
            )
            for source_index in range(source_count)
        )
        source_duration_targets = selected_duration_targets
        source_duration_deviations = tuple(
            abs(actual - target) / target
            for actual, target in zip(
                source_durations,
                source_duration_targets,
            )
        )
    diagnostics: list[DirectBoundaryVisualDiagnostic] = []
    raw_cross_source_pair_count = source_count * (source_count - 1)
    for boundary_index, (outgoing_index, incoming_index) in enumerate(
        zip(path, path[1:])
    ):
        cut_applied = outgoing_index != incoming_index
        selected_candidate = selected_boundary_candidates[boundary_index]
        mismatch = (
            selected_boundary_mismatches[boundary_index]
            if cut_applied
            else DirectBoundaryMismatch(
                cost=0.0,
                base_mismatch=0.0,
                attention_position_mismatch=None,
                attention_scale_mismatch=None,
                dominant_pose_mismatch=None,
                pose_basis=None,
                formation_mismatch=None,
                silhouette_mismatch=None,
                person_layout_mismatch=None,
                person_count_mismatch=None,
                evidence_component_count=0,
                warning_flags=(),
                severe_flags=(),
            )
        )
        content_valid_pairs = tuple(
            (left, right)
            for left in range(source_count)
            for right in range(source_count)
            if left != right
            and candidate_validity[boundary_index][left]
            and candidate_validity[boundary_index + 1][right]
        )
        severe_pair_count = sum(
            bool(matrices[boundary_index][left][right].severe_flags)
            for left, right in content_valid_pairs
        )
        ranked_pairs = sorted(
            (
                (matrices[boundary_index][left][right].cost, left, right)
                for left, right in content_valid_pairs
                if (
                    not matrices[boundary_index][left][right].severe_flags
                    or (
                        usage_policy
                        == "heterogeneous_role_aware_cinematic_v1"
                        and matrices[boundary_index][left][right].cinematic_bridge
                        is not None
                    )
                )
            ),
        )
        if cut_applied and not ranked_pairs:
            raise DirectMemberVisualSelectionError(
                "selected direct visual boundary has no eligible diagnostic pairs",
                {
                    "failure": "boundary_diagnostic_eligible_pair_exhausted",
                    "boundary_index": boundary_index,
                    "invalid_content_pair_count": (
                        raw_cross_source_pair_count - len(content_valid_pairs)
                    ),
                    "severe_pair_count": severe_pair_count,
                },
            )
        selected_rank = (
            next(
                rank
                for rank, (_cost, left, right) in enumerate(
                    ranked_pairs,
                    start=1,
                )
                if left == outgoing_index and right == incoming_index
            )
            if cut_applied
            else 0
        )
        minimum_cost = ranked_pairs[0][0] if ranked_pairs else 0.0
        diagnostics.append(
            DirectBoundaryVisualDiagnostic(
                boundary_index=boundary_index,
                canonical_master_seconds=spans[boundary_index][1],
                selected_cut_master_seconds=(
                    selected_candidate.cut_master_seconds
                ),
                outgoing_eligible_source_index=outgoing_index,
                incoming_eligible_source_index=incoming_index,
                outgoing_sample=selected_candidate.outgoing[outgoing_index],
                incoming_sample=selected_candidate.incoming[incoming_index],
                mismatch=mismatch,
                selected_pair_rank=selected_rank,
                candidate_pair_count=len(ranked_pairs),
                raw_cross_source_pair_count=raw_cross_source_pair_count,
                invalid_content_pair_count=(
                    raw_cross_source_pair_count - len(content_valid_pairs)
                ),
                severe_pair_count=severe_pair_count,
                minimum_pair_cost=minimum_cost,
                selected_pair_cost_delta=(
                    mismatch.cost - minimum_cost if cut_applied else 0.0
                ),
                ping_pong_penalty_applied=(
                    cut_applied
                    and boundary_index >= 1
                    and incoming_index == path[boundary_index - 1]
                ),
                cut_applied=cut_applied,
            )
        )
    return DirectMemberVisualSelection(
        source_indices=path,
        source_quotas=selected_source_targets,
        source_counts=counts,
        source_durations_seconds=source_durations,
        source_target_durations_seconds=source_duration_targets,
        source_duration_relative_deviations=source_duration_deviations,
        span_validity=tuple(span_diagnostics),
        boundaries=tuple(diagnostics),
        selected_boundary_master_seconds=selected_boundary_master_seconds,
        optimizer_cost=optimizer_cost,
        sample_count=sample_count,
        failed_sample_count=failed_sample_count,
        native_scene_cut_evidence_available=scene_cut_evidence_available,
        continuous_official_portrait_fancam_contract=(
            continuous_official_portrait_fancam_contract
        ),
        config=resolved_config,
        optimizer_search_diagnostics=optimizer_search_diagnostics,
        source_frame_geometries=source_frame_geometries,
        usage_policy=usage_policy,
        source_count_minimums=selected_count_minimums,
        source_count_maximums=selected_count_maximums,
        source_duration_minimums_seconds=selected_duration_minimums,
        source_duration_maximums_seconds=selected_duration_maximums,
    )


__all__ = [
    "DescriptorProvider",
    "DirectBoundaryMismatch",
    "DirectBoundaryMotionEvidence",
    "DirectBoundaryVisualDiagnostic",
    "DirectMemberBoundaryTransformRepair",
    "DirectMemberSourceFrameGeometry",
    "DirectMemberVisualConfig",
    "DirectMemberVisualSelection",
    "DirectMemberVisualSelectionError",
    "DirectSpanSourceValidity",
    "DirectSpanVisualDiagnostic",
    "select_balanced_direct_member_sources",
]
