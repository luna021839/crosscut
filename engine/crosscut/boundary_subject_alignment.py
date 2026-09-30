"""Exact-frame, boundary-local subject alignment for GROUP hard cuts.

The older group crop planner reasons about whole selected segments.  A single
missing detector sample can therefore demote a long same-source run, while an
endpoint-only affine estimate cannot prove that the short settle is safe on
the intervening frames.  This module deliberately owns a much smaller unit:
the last ``window_frames`` of the outgoing selected segment and the first
``window_frames`` of the incoming selected segment.

Every eligible decision is based on the decoded cut endpoints in the exact
canonical output space.  ORB/background correspondence is not required: a
single dominant face is matched by face/eye geometry and a group view by its
high-confidence silhouette.  Any authored track is then checked on every
transformed output frame after the same base ``PanCropTrack`` used by the
renderer.  Missing geometry, exposed frame edges, clipped subjects, native
camera cuts and dissolve overlaps all fail closed.  Pixels and appearance
descriptors are never serialized.
"""

from __future__ import annotations
from .optional_native_worker import isolated_optional

from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import product
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from .face_match import FaceMatchPlan, FaceMatchSegment
from .exact_native_frame_reader import ExactNativeFrameReader
from .render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    PanCropTrack,
    RenderPlan,
    RenderSegment,
    RenderTransition,
    TransitionKind,
    minimum_roll_safe_zoom,
)
from .render_boundary_clock import native_frame_clocks_for_output
from .render_frame_schedule import VerifiedRenderFrameClock, probe_render_frame_clock
from .visual_score import VisualTimelineChoice, VisualTimelinePlan


ANALYSIS_BASIS = "exact_cut_frame_shot_adaptive_subject_alignment_v5"
FrameProvider = Callable[[int, float], np.ndarray | None]


@dataclass(frozen=True, slots=True)
class BoundarySubjectObservation:
    """Non-identifying subject geometry in canonical output coordinates."""

    face_count: int = 0
    face_box_xywh: tuple[float, float, float, float] | None = None
    face_landmarks_xy: tuple[tuple[float, float], ...] | None = None
    face_head_roll_radians: float | None = None
    face_eye_scale: float | None = None
    silhouette_box_xywh: tuple[float, float, float, float] | None = None
    silhouette_confidence: float = 0.0
    detected_person_count: int = 0
    formation_boxes_xywh: tuple[
        tuple[float, float, float, float], ...
    ] = ()
    # The descriptor may prove one dominant face while small background faces
    # are also present. Count alone must not discard that existing evidence.
    face_is_unambiguous_dominant: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.face_is_unambiguous_dominant, bool):
            raise ValueError("dominant face flag must be boolean")
        if self.face_is_unambiguous_dominant and (
            self.face_count < 1 or self.face_box_xywh is None
        ):
            raise ValueError("dominant face requires an observed face box")
        if (
            isinstance(self.face_count, bool)
            or not isinstance(self.face_count, int)
            or self.face_count < 0
        ):
            raise ValueError("face_count must be a non-negative integer")
        for name in ("face_box_xywh", "silhouette_box_xywh"):
            raw = getattr(self, name)
            if raw is None:
                continue
            if len(raw) != 4:
                raise ValueError(f"{name} needs four normalized values")
            box = tuple(float(value) for value in raw)
            if not all(math.isfinite(value) for value in box):
                raise ValueError(f"{name} must be finite")
            x_value, y_value, width, height = box
            if not (
                0.0 <= x_value <= 1.0
                and 0.0 <= y_value <= 1.0
                and 0.0 < width <= 1.0
                and 0.0 < height <= 1.0
                and x_value + width <= 1.000001
                and y_value + height <= 1.000001
            ):
                raise ValueError(f"{name} must fit normalized output bounds")
            object.__setattr__(self, name, box)
        if (
            isinstance(self.detected_person_count, bool)
            or not isinstance(self.detected_person_count, int)
            or self.detected_person_count < 0
        ):
            raise ValueError("detected_person_count must be a non-negative integer")
        formation_boxes: list[tuple[float, float, float, float]] = []
        for raw in self.formation_boxes_xywh:
            if len(raw) != 4:
                raise ValueError("formation box needs four normalized values")
            box = tuple(float(value) for value in raw)
            if not all(math.isfinite(value) for value in box):
                raise ValueError("formation boxes must be finite")
            x_value, y_value, width, height = box
            if not (
                0.0 <= x_value <= 1.0
                and 0.0 <= y_value <= 1.0
                and 0.0 < width <= 1.0
                and 0.0 < height <= 1.0
                and x_value + width <= 1.000001
                and y_value + height <= 1.000001
            ):
                raise ValueError("formation boxes must fit normalized output bounds")
            formation_boxes.append(box)
        object.__setattr__(self, "formation_boxes_xywh", tuple(formation_boxes))
        if self.face_landmarks_xy is not None:
            landmarks = tuple(
                (float(point[0]), float(point[1]))
                for point in self.face_landmarks_xy
            )
            if len(landmarks) < 2 or any(
                not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in point)
                for point in landmarks
            ):
                raise ValueError("face landmarks must be normalized and finite")
            object.__setattr__(self, "face_landmarks_xy", landmarks)
        for name in ("face_head_roll_radians", "face_eye_scale"):
            value = getattr(self, name)
            if value is None:
                continue
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{name} must be finite")
            if name == "face_eye_scale" and number <= 0.0:
                raise ValueError("face_eye_scale must be positive")
            object.__setattr__(self, name, number)
        confidence = float(self.silhouette_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("silhouette confidence must be normalized")
        object.__setattr__(self, "silhouette_confidence", confidence)


ObservationProvider = Callable[[np.ndarray], BoundarySubjectObservation]


@dataclass(frozen=True, slots=True)
class SourceResolutionHeadroom:
    source_index: int
    width: int | None
    height: int | None
    native_zoom_headroom: float | None
    quality_profile: str
    quality_profile_max_zoom: float
    renderer_schema_max_zoom: float
    effective_max_zoom: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_index": self.source_index,
            "width": self.width,
            "height": self.height,
            "native_zoom_headroom": self.native_zoom_headroom,
            "quality_profile": self.quality_profile,
            "quality_profile_max_zoom": self.quality_profile_max_zoom,
            "renderer_schema_max_zoom": self.renderer_schema_max_zoom,
            "effective_max_zoom": self.effective_max_zoom,
        }


@dataclass(frozen=True, slots=True)
class BoundarySubjectAlignmentConfig:
    window_frames: int = 14
    analysis_width: int = 854
    analysis_height: int = 480
    renderer_schema_max_zoom: float = 2.0
    conservative_1080_max_zoom: float = 1.18
    native_1440_max_zoom: float = 1.33
    native_2160_max_zoom: float = 2.0
    unknown_resolution_max_zoom: float = 1.06
    maximum_center_shift: float = 0.040
    maximum_roll_degrees: float = 1.60
    minimum_silhouette_confidence: float = 0.72
    minimum_face_visible_fraction: float = 0.995
    minimum_silhouette_visible_fraction: float = 0.92
    minimum_performer_visible_fraction: float = 0.985
    maximum_face_zoom: float = 2.0
    maximum_upper_body_zoom: float = 1.16
    maximum_formation_zoom: float = 1.33
    maximum_silhouette_zoom: float = 1.08
    maximum_face_center_shift: float = 0.040
    maximum_paired_face_center_shift: float = 0.100
    maximum_upper_body_center_shift: float = 0.040
    maximum_formation_center_shift: float = 0.018
    maximum_silhouette_center_shift: float = 0.025
    maximum_transformed_layout_mismatch: float = 0.34
    maximum_layout_worsening: float = 0.02
    minimum_geometry_improvement: float = 0.015
    minimum_geometry_improvement_fraction: float = 0.10
    already_matched_scale_ratio: float = 1.015
    already_matched_center_distance: float = 0.025

    def __post_init__(self) -> None:
        if (
            isinstance(self.window_frames, bool)
            or not isinstance(self.window_frames, int)
            or not 8 <= self.window_frames <= 24
        ):
            raise ValueError("boundary alignment window must be 8...24 frames")
        if self.analysis_width < 320 or self.analysis_height < 180:
            raise ValueError("boundary alignment analysis space is too small")
        if abs(self.analysis_width / self.analysis_height - 16.0 / 9.0) > 0.01:
            raise ValueError("boundary alignment analysis space must be 16:9")
        values = tuple(
            float(getattr(self, name))
            for name in (
                "renderer_schema_max_zoom",
                "conservative_1080_max_zoom",
                "native_1440_max_zoom",
                "native_2160_max_zoom",
                "unknown_resolution_max_zoom",
                "maximum_center_shift",
                "maximum_roll_degrees",
                "minimum_silhouette_confidence",
                "minimum_face_visible_fraction",
                "minimum_silhouette_visible_fraction",
                "minimum_performer_visible_fraction",
                "maximum_face_zoom",
                "maximum_upper_body_zoom",
                "maximum_formation_zoom",
                "maximum_silhouette_zoom",
                "maximum_face_center_shift",
                "maximum_paired_face_center_shift",
                "maximum_upper_body_center_shift",
                "maximum_formation_center_shift",
                "maximum_silhouette_center_shift",
                "maximum_transformed_layout_mismatch",
                "maximum_layout_worsening",
                "minimum_geometry_improvement",
                "minimum_geometry_improvement_fraction",
                "already_matched_scale_ratio",
                "already_matched_center_distance",
            )
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("boundary alignment configuration must be finite")
        if not 1.0 < self.renderer_schema_max_zoom <= 2.0:
            raise ValueError("native-pixel alignment zoom cannot exceed 2.0")
        if not (
            1.0 < self.conservative_1080_max_zoom
            <= self.native_1440_max_zoom
            <= self.native_2160_max_zoom
            <= 2.0
        ):
            raise ValueError("resolution headroom profiles must be ordered")
        if not 1.0 < self.unknown_resolution_max_zoom <= self.conservative_1080_max_zoom:
            raise ValueError("unknown-resolution zoom must be conservative")
        if not 0.0 <= self.maximum_center_shift <= 0.040:
            raise ValueError("boundary pan exceeds the renderer schema")
        if not 0.0 <= self.maximum_paired_face_center_shift <= 0.100:
            raise ValueError("paired face pan exceeds the bilateral framing budget")
        if not 0.0 <= self.maximum_roll_degrees <= 1.60:
            raise ValueError("boundary roll exceeds the renderer schema")
        if not 0.0 <= self.minimum_silhouette_confidence <= 1.0:
            raise ValueError("silhouette confidence must be normalized")
        if not 0.8 <= self.minimum_face_visible_fraction <= 1.0:
            raise ValueError("face containment threshold is invalid")
        if not 0.8 <= self.minimum_silhouette_visible_fraction <= 1.0:
            raise ValueError("silhouette containment threshold is invalid")
        if not 0.8 <= self.minimum_performer_visible_fraction <= 1.0:
            raise ValueError("performer containment threshold is invalid")
        for value in (
            self.maximum_face_zoom,
            self.maximum_upper_body_zoom,
            self.maximum_formation_zoom,
            self.maximum_silhouette_zoom,
        ):
            if not 1.0 < value <= self.renderer_schema_max_zoom:
                raise ValueError("subject zoom cap exceeds the renderer schema")
        for value in (
            self.maximum_face_center_shift,
            self.maximum_upper_body_center_shift,
            self.maximum_formation_center_shift,
            self.maximum_silhouette_center_shift,
        ):
            if not 0.0 <= value <= self.maximum_center_shift:
                raise ValueError("subject pan cap exceeds the renderer schema")
        if not 0.0 < self.maximum_transformed_layout_mismatch < 1.0:
            raise ValueError("layout mismatch cap must be normalized")
        if not 0.0 <= self.maximum_layout_worsening <= 0.10:
            raise ValueError("layout worsening tolerance is invalid")


@dataclass(frozen=True, slots=True)
class _SubjectGeometry:
    kind: str
    alignment_box: tuple[float, float, float, float]
    scale: float
    roll_radians: float | None = None
    containment_boxes: tuple[tuple[float, float, float, float], ...] = ()
    layout_boxes: tuple[tuple[float, float, float, float], ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"face", "upper_body", "formation", "silhouette"}:
            raise ValueError("unsupported shot-adaptive alignment subject")
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("alignment subject scale must be positive")
        if not self.containment_boxes:
            object.__setattr__(self, "containment_boxes", (self.alignment_box,))
        if not self.layout_boxes:
            object.__setattr__(self, "layout_boxes", self.containment_boxes)


@dataclass(frozen=True, slots=True)
class _TransformState:
    zoom: float = 1.0
    center_x: float = 0.5
    center_y: float = 0.5
    roll_degrees: float = 0.0

    @property
    def neutral(self) -> bool:
        return (
            abs(self.zoom - 1.0) <= 1e-9
            and abs(self.center_x - 0.5) <= 1e-9
            and abs(self.center_y - 0.5) <= 1e-9
            and abs(self.roll_degrees) <= 1e-9
        )

    def keyframe(self, time_seconds: float) -> EditorialTransformKeyframe:
        return EditorialTransformKeyframe(
            time_seconds,
            self.zoom,
            self.center_x,
            self.center_y,
            self.roll_degrees,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "zoom": self.zoom,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "roll_degrees": self.roll_degrees,
        }


@dataclass(frozen=True, slots=True)
class BoundarySubjectAlignmentDecision:
    boundary_index: int
    outgoing_segment_index: int
    incoming_segment_index: int
    outgoing_source_index: int
    incoming_source_index: int
    outgoing_source_seconds: float
    incoming_source_seconds: float
    eligible: bool
    rejection_reason: str | None
    subject_kind: str | None = None
    transformed_side: str | None = None
    outgoing_state: _TransformState = _TransformState()
    incoming_state: _TransformState = _TransformState()
    baseline_geometry_mismatch: float | None = None
    transformed_geometry_mismatch: float | None = None
    exact_scale_ratio: float | None = None
    roll_evidence_available: bool = False
    roll_applied: bool = False
    partial_scale_alignment: bool = False
    validated_output_frame_count: int = 0
    optical_flow_containment_frame_count: int = 0
    outgoing_shot_policy: str | None = None
    incoming_shot_policy: str | None = None
    baseline_layout_mismatch: float | None = None
    transformed_layout_mismatch: float | None = None
    outgoing_native_frame_index: int | None = None
    incoming_native_frame_index: int | None = None
    outgoing_output_frame_index: int | None = None
    incoming_output_frame_index: int | None = None
    outgoing_validated_output_frame_count: int = 0
    incoming_validated_output_frame_count: int = 0
    replaced_one_sided_geometry_mismatch: float | None = None

    def __post_init__(self) -> None:
        if self.boundary_index < 0 or min(
            self.outgoing_segment_index,
            self.incoming_segment_index,
            self.outgoing_source_index,
            self.incoming_source_index,
        ) < 0:
            raise ValueError("boundary alignment indices must be non-negative")
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (self.outgoing_source_seconds, self.incoming_source_seconds)
        ):
            raise ValueError("boundary source clocks must be finite and non-negative")
        if self.eligible == bool(self.rejection_reason):
            raise ValueError("eligible decision and rejection reason disagree")
        if self.subject_kind is not None and self.subject_kind not in {
            "face",
            "upper_body",
            "formation",
            "silhouette",
        }:
            raise ValueError("boundary subject kind is invalid")
        if self.transformed_side is not None and self.transformed_side not in {
            "outgoing",
            "incoming",
            "both",
        }:
            raise ValueError("transformed side is invalid")
        if min(
            self.validated_output_frame_count,
            self.optical_flow_containment_frame_count,
            self.outgoing_validated_output_frame_count,
            self.incoming_validated_output_frame_count,
        ) < 0:
            raise ValueError("validated frame counts cannot be negative")
        if self.optical_flow_containment_frame_count > self.validated_output_frame_count:
            raise ValueError("tracked containment frames must be a validated subset")
        previous = self.replaced_one_sided_geometry_mismatch
        if previous is not None and (type(previous) not in (int, float) or not math.isfinite(previous)
                or not self.eligible or self.transformed_side != 'both'
                or self.transformed_geometry_mismatch is None
                or not 0 <= self.transformed_geometry_mismatch < previous <= 1):
            raise ValueError('paired replacement requires a strictly improved measured incumbent')

    @property
    def uses_pan(self) -> bool:
        return any(math.hypot(state.center_x - 0.5, state.center_y - 0.5) > 0.003
                   for state in (self.outgoing_state, self.incoming_state))

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_index": self.boundary_index,
            "outgoing_segment_index": self.outgoing_segment_index,
            "incoming_segment_index": self.incoming_segment_index,
            "outgoing_source_index": self.outgoing_source_index,
            "incoming_source_index": self.incoming_source_index,
            "outgoing_source_seconds": self.outgoing_source_seconds,
            "incoming_source_seconds": self.incoming_source_seconds,
            "eligible": self.eligible,
            "rejection_reason": self.rejection_reason,
            "subject_kind": self.subject_kind,
            "transformed_side": self.transformed_side,
            "outgoing_state": self.outgoing_state.to_dict(),
            "incoming_state": self.incoming_state.to_dict(),
            "baseline_geometry_mismatch": self.baseline_geometry_mismatch,
            "transformed_geometry_mismatch": self.transformed_geometry_mismatch,
            "exact_scale_ratio": self.exact_scale_ratio,
            "roll_evidence_available": self.roll_evidence_available,
            "roll_applied": self.roll_applied,
            "partial_scale_alignment": self.partial_scale_alignment,
            "validated_output_frame_count": self.validated_output_frame_count,
            "optical_flow_containment_frame_count": (
                self.optical_flow_containment_frame_count
            ),
            "outgoing_shot_policy": self.outgoing_shot_policy,
            "incoming_shot_policy": self.incoming_shot_policy,
            "baseline_layout_mismatch": self.baseline_layout_mismatch,
            "transformed_layout_mismatch": self.transformed_layout_mismatch,
            "outgoing_native_frame_index": self.outgoing_native_frame_index,
            "incoming_native_frame_index": self.incoming_native_frame_index,
            "outgoing_output_frame_index": self.outgoing_output_frame_index,
            "incoming_output_frame_index": self.incoming_output_frame_index,
            "outgoing_validated_output_frame_count": self.outgoing_validated_output_frame_count,
            "incoming_validated_output_frame_count": self.incoming_validated_output_frame_count,
            "replaced_one_sided_geometry_mismatch": self.replaced_one_sided_geometry_mismatch,
        }


@dataclass(frozen=True, slots=True)
class BoundarySubjectAlignmentSegment:
    segment_index: int
    source_index: int
    track: EditorialTransformTrack | None
    boundary_indices: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "source_index": self.source_index,
            "applied": self.track is not None,
            "boundary_indices": list(self.boundary_indices),
            "keyframes": (
                [
                    {
                        "time_seconds": keyframe.time_seconds,
                        "zoom": keyframe.zoom,
                        "center_x": keyframe.center_x,
                        "center_y": keyframe.center_y,
                        "roll_degrees": keyframe.roll_degrees,
                    }
                    for keyframe in self.track.keyframes
                ]
                if self.track is not None
                else []
            ),
        }


@dataclass(frozen=True, slots=True)
class BoundarySubjectAlignmentPlan:
    decisions: tuple[BoundarySubjectAlignmentDecision, ...]
    segments: tuple[BoundarySubjectAlignmentSegment, ...]
    source_headroom: tuple[SourceResolutionHeadroom, ...]
    window_frames: int
    analysis_basis: str = ANALYSIS_BASIS
    sample_clock_basis: str = "unverified_injected_segment_clock_v1"
    dense_track_candidate_attempt_count: int = 0

    @property
    def eligible_count(self) -> int:
        return sum(item.eligible for item in self.decisions)

    @property
    def applied_segment_count(self) -> int:
        return sum(item.track is not None for item in self.segments)

    @property
    def pan_event_count(self) -> int:
        return sum(item.eligible and item.uses_pan for item in self.decisions)

    @property
    def roll_event_count(self) -> int:
        return sum(item.eligible and item.roll_applied for item in self.decisions)

    @property
    def validated_output_frame_count(self) -> int:
        return sum(item.validated_output_frame_count for item in self.decisions)

    @property
    def optical_flow_containment_frame_count(self) -> int:
        return sum(
            item.optical_flow_containment_frame_count for item in self.decisions
        )

    def track_for(self, segment_index: int) -> EditorialTransformTrack | None:
        matches = [
            item.track for item in self.segments if item.segment_index == segment_index
        ]
        if len(matches) != 1:
            raise ValueError("boundary alignment plan must cover every segment once")
        return matches[0]

    def decision_for(self, boundary_index: int) -> BoundarySubjectAlignmentDecision | None:
        matches = [
            item for item in self.decisions if item.boundary_index == boundary_index
        ]
        if len(matches) > 1:
            raise ValueError("boundary alignment plan has duplicate decisions")
        return matches[0] if matches else None

    def to_dict(self) -> dict[str, Any]:
        rejections: dict[str, int] = {}
        subjects: dict[str, int] = {}
        eligible_subjects: dict[str, int] = {}
        for item in self.decisions:
            if item.rejection_reason is not None:
                rejections[item.rejection_reason] = rejections.get(item.rejection_reason, 0) + 1
            if item.subject_kind is not None:
                subjects[item.subject_kind] = subjects.get(item.subject_kind, 0) + 1
                if item.eligible:
                    eligible_subjects[item.subject_kind] = (
                        eligible_subjects.get(item.subject_kind, 0) + 1
                    )
        hard_cut_decision_count = sum(
            item.rejection_reason != "unreviewed_dissolve_overlap"
            for item in self.decisions
        )
        return {
            "analysis_basis": self.analysis_basis,
            "sample_clock_basis": self.sample_clock_basis,
            "policy": {
                "scope": "cross_source_hard_cut_last_and_first_local_windows_only",
                "window_frames_each_side": self.window_frames,
                "orb_required": False,
                "native_internal_cuts_transformed": False,
                "native_cut_scope": "active_boundary_window_plus_two_frame_tracking_guard",
                "dissolve_overlap_transformed": False,
                "frame_validation": "every_transformed_output_frame_after_base_crop",
                "layout_validation": "cross_source_at_cut_then_adjacent_source_local_motion_v1",
                "shot_policy": (
                    "close_face_medium_upper_body_wide_formation_"
                    "with_silhouette_fail_closed_fallback"
                ),
                "group_transition_kind": "hard_cut_only",
                "legacy_editorial_schema_globally_relaxed": False,
                "native_zoom_policy": "source_resolution_after_base_crop_face_and_formation_alignment_v2",
                "close_face_containment": "measured_face_plus_source_clipped_head_margin",
                "paired_tight_medium_containment": "bilateral_central_landmark_face_and_source_truncated_torso_head_envelope_v1",
                "medium_wide_containment": "full_visible_performer_or_formation",
                "transform_search": "endpoint_valid_strengths_ranked_then_complete_dense_windows_v1",
                "maximum_track_candidates_per_segment": 120,
                "dense_observations_reused_across_strengths": True,
                "paired_framing_fallback": "common_face_scale_center_on_two_unused_segments_v1",
                "paired_framing_competition": "replace_only_own_isolated_one_sided_ramp_after_strict_gain_and_two_dense_windows_v1",
                "maximum_paired_framing_candidates_per_boundary": 5,
                "paired_framing_publication": "both_complete_dense_windows_or_neither",
                "paired_face_center_shift_cap": 0.10,
            },
            "dense_track_candidate_attempt_count": self.dense_track_candidate_attempt_count,
            "decision_count": len(self.decisions),
            "eligible_count": self.eligible_count,
            "applied_segment_count": self.applied_segment_count,
            "pan_event_count": self.pan_event_count,
            "roll_event_count": self.roll_event_count,
            "validated_output_frame_count": self.validated_output_frame_count,
            "optical_flow_containment_frame_count": (
                self.optical_flow_containment_frame_count
            ),
            "hard_cut_decision_count": hard_cut_decision_count,
            "applied_boundary_density": (
                self.eligible_count / hard_cut_decision_count
                if hard_cut_decision_count
                else 0.0
            ),
            "alignment_subject_counts": [
                {"subject": subject, "count": count}
                for subject, count in sorted(subjects.items())
            ],
            "eligible_alignment_subject_counts": [
                {"subject": subject, "count": count}
                for subject, count in sorted(eligible_subjects.items())
            ],
            "rejection_counts": [
                {"reason": reason, "count": count}
                for reason, count in sorted(rejections.items())
            ],
            "source_resolution_headroom": [item.to_dict() for item in self.source_headroom],
            "storage_policy": {
                "decoded_pixels_serialized": False,
                "appearance_or_identity_embeddings_serialized": False,
                "landmarks_serialized": False,
            },
            "decisions": [item.to_dict() for item in self.decisions],
            "segments": [item.to_dict() for item in self.segments],
        }


class _CaptureProvider:
    def __init__(self, sources: Sequence[Path], maximum_cache_size: int = 192,
                 maximum_cache_bytes: int = 64 * 1024 * 1024) -> None:
        if any(type(value) is not int or value <= 0
               for value in (maximum_cache_size, maximum_cache_bytes)):
            raise ValueError("alignment pixel cache limits must be positive integers")
        self._captures: list[cv2.VideoCapture | None] = []
        self._native_readers: dict[int, ExactNativeFrameReader] = {}
        self._cache: OrderedDict[tuple[int, str, float | int], np.ndarray | None] = OrderedDict()
        self._maximum_cache_size = maximum_cache_size
        self._maximum_cache_bytes = maximum_cache_bytes
        self._cache_bytes = 0
        try:
            for source in sources:
                capture = cv2.VideoCapture(str(source))
                self._captures.append(capture if capture.isOpened() else None)
        except Exception:
            self.close()
            raise

    def resolution(self, source_index: int) -> tuple[int, int] | None:
        capture = self._captures[source_index]
        if capture is None:
            return None
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return (width, height) if width > 0 and height > 0 else None

    def frame_at(self, source_index: int, seconds: float) -> np.ndarray | None:
        # This legacy/test reader must not round across a native half-cell.
        # Production alignment below uses an explicit verified native index.
        seconds = max(0.0, float(seconds))
        key = (source_index, "seconds", seconds)
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return None if value is None else value.copy()
        capture = self._captures[source_index]
        frame: np.ndarray | None = None
        if capture is not None:
            if source_index in self._native_readers:
                self._native_readers[source_index].invalidate_position()
            capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.)
            success, decoded = capture.read()
            if success and isinstance(decoded, np.ndarray) and decoded.size:
                frame = decoded
        self._remember(key, frame)
        return None if frame is None else frame.copy()

    def frame_at_native_index(self, source_index: int, index: int,
                              clock: VerifiedRenderFrameClock) -> np.ndarray | None:
        """Read exactly one verified source frame, without timestamp fallback.

        Index and presentation position must agree. A failed seek/read cannot
        lend a neighboring frame to a dense crop/containment approval.
        """
        if type(index) is not int or not 0 <= index < clock.frame_count:
            raise ValueError("alignment native frame is outside the source")
        reader = self._native_readers.get(source_index)
        if reader is not None and reader.clock != clock:
            raise ValueError("alignment source clock changed inside one capture")
        key = (source_index, "native", index)
        if key in self._cache:
            frame = self._cache.pop(key)
            self._cache[key] = frame
            return None if frame is None else frame.copy()
        capture = self._captures[source_index]
        frame = None
        if capture is not None:
            if reader is None:
                reader = ExactNativeFrameReader(capture, clock)
                self._native_readers[source_index] = reader
            frame = reader.frame_at(index)
        self._remember(key, frame)
        return None if frame is None else frame.copy()

    def _remember(self, key, frame):
        self._cache[key] = frame
        self._cache_bytes += 0 if frame is None else frame.nbytes
        while (len(self._cache) > self._maximum_cache_size
                or self._cache_bytes > self._maximum_cache_bytes):
            _, expired = self._cache.popitem(last=False)
            self._cache_bytes -= 0 if expired is None else expired.nbytes

    def close(self) -> None:
        for capture in self._captures:
            if capture is not None:
                capture.release()
        self._captures.clear()
        self._native_readers.clear()
        self._cache.clear()
        self._cache_bytes = 0


def _default_observation_provider(frame: np.ndarray) -> BoundarySubjectObservation:
    from .visual_score import describe_frame

    descriptor = describe_frame(
        frame,
        include_subject_geometry=True,
        include_formation_pose=False,
    )
    return BoundarySubjectObservation(
        face_count=descriptor.face_count,
        face_box_xywh=descriptor.primary_face_bbox,
        face_landmarks_xy=descriptor.primary_face_landmarks_xy,
        face_head_roll_radians=descriptor.primary_face_head_roll_radians,
        face_eye_scale=descriptor.primary_face_eye_scale,
        silhouette_box_xywh=descriptor.silhouette_bbox_xywh,
        silhouette_confidence=descriptor.silhouette_confidence,
        detected_person_count=descriptor.detected_person_count,
        formation_boxes_xywh=tuple(
            person.bbox_xywh for person in descriptor.formation_people
        ),
        face_is_unambiguous_dominant=descriptor.primary_face_bbox is not None,
    )


def _frame_counts(
    choices: Sequence[VisualTimelineChoice], fps: float
) -> tuple[int, ...]:
    counts: list[int] = []
    cumulative = 0.0
    previous = 0
    for choice in choices:
        cumulative += choice.duration_seconds
        boundary = int(round(cumulative * fps))
        count = boundary - previous
        if count < 2:
            raise ValueError("boundary alignment segment is shorter than two frames")
        counts.append(count)
        previous = boundary
    return tuple(counts)


def _face_matches(face_match_plan: FaceMatchPlan | None) -> dict[int, FaceMatchSegment]:
    if face_match_plan is None:
        return {}
    if not isinstance(face_match_plan, FaceMatchPlan):
        raise TypeError("face_match_plan must be a FaceMatchPlan")
    matches = {item.segment_index: item for item in face_match_plan.segments}
    if len(matches) != len(face_match_plan.segments):
        raise ValueError("face-match plan contains duplicate segment indices")
    return matches


def _fixed_crop_at(
    track: PanCropTrack,
    time_seconds: float,
) -> tuple[float, float, float, float]:
    width, height = track.fixed_crop_size
    positions = track.fixed_crop_positions()
    if time_seconds <= positions[0][0] + 1e-9:
        return (positions[0][1], positions[0][2], width, height)
    for left, right in zip(positions, positions[1:]):
        if time_seconds <= right[0] + 1e-9:
            alpha = (time_seconds - left[0]) / max(1e-9, right[0] - left[0])
            return (
                left[1] + (right[1] - left[1]) * alpha,
                left[2] + (right[2] - left[2]) * alpha,
                width,
                height,
            )
    return (positions[-1][1], positions[-1][2], width, height)


def _canonical_frame(
    frame: np.ndarray,
    *,
    width: int,
    height: int,
    base_crop: tuple[float, float, float, float] | None,
) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("boundary alignment expects a BGR frame")
    working = frame
    if base_crop is not None:
        left, top, crop_width, crop_height = base_crop
        source_height, source_width = working.shape[:2]
        # Mirror render.py's ``trunc(iw*ratio/2)*2`` dimensions and bounded
        # integer crop origin instead of letting a one-pixel round-up create
        # geometry the actual FFmpeg chain never displays.
        crop_pixel_width = max(2, int(source_width * crop_width) // 2 * 2)
        crop_pixel_height = max(2, int(source_height * crop_height) // 2 * 2)
        crop_pixel_width = min(source_width, crop_pixel_width)
        crop_pixel_height = min(source_height, crop_pixel_height)
        x0 = max(
            0,
            min(source_width - crop_pixel_width, int(left * source_width)),
        )
        y0 = max(
            0,
            min(source_height - crop_pixel_height, int(top * source_height)),
        )
        x1 = x0 + crop_pixel_width
        y1 = y0 + crop_pixel_height
        working = working[y0:y1, x0:x1]
    source_height, source_width = working.shape[:2]
    scale = max(width / source_width, height / source_height)
    scaled_width = max(width, int(math.ceil(source_width * scale)))
    scaled_height = max(height, int(math.ceil(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        working,
        (scaled_width, scaled_height),
        interpolation=interpolation,
    )
    offset_x = (scaled_width - width) // 2
    offset_y = (scaled_height - height) // 2
    return resized[offset_y : offset_y + height, offset_x : offset_x + width]


def _source_seconds(
    choice: VisualTimelineChoice,
    frame_index: int,
    *,
    fps: float,
) -> float:
    return min(
        choice.source_local_end,
        choice.source_local_start + frame_index * choice.source_playback_rate / fps,
    )


def _alignment_native_frames(choices, sources, transitions, counts, fps, clocks):
    """Use the renderer's existing coalescing/rounding, not a parallel clock.

    Geometry is applied after the source clock. The numeric plan is never a
    video job, and only pure frames with verified clocks are addressable.
    """
    origin = choices[0].canonical_master_start
    plan = RenderPlan(tuple(Path(path) for path in sources), tuple(
        RenderSegment(c.source_index, c.source_local_start, c.duration_seconds,
            c.canonical_master_start-origin, source_duration=c.source_duration_seconds)
        for c in choices), 0, "16:9", Path(sources[0]).with_suffix(".alignment-clock-unused.mp4"),
        transitions=tuple(transitions), fps=fps)
    output_starts = []
    cursor = 0
    wanted = set()
    for choice, count in zip(choices, counts, strict=True):
        output_starts.append(cursor)
        if choice.source_index in clocks:
            wanted.update(range(cursor, cursor+count))
        cursor += count
    for index, transition in enumerate(transitions):
        if transition.kind.uses_overlap:
            start = output_starts[index+1]
            wanted.difference_update(range(start, start+transition.frame_count))
    samples = native_frame_clocks_for_output(plan, clocks, sorted(wanted),
        allow_pure_frames_in_overlap_plan=True)
    return tuple(output_starts), {index:sample for index,sample in samples.items()
        if not sample.terminal_clone}


def _boundary_window_frame_indices(*, outgoing: bool, frame_count: int, window_frames: int):
    return ((max(0, frame_count-window_frames-2), frame_count-1) if outgoing else
        (0, min(frame_count-1, window_frames-1+2)))


def _native_cut_in_boundary_window(
    choice: VisualTimelineChoice,
    *,
    outgoing: bool,
    frame_count: int,
    fps: float,
    window_frames: int,
    native_frame_window: tuple[int, int] | None = None,
    native_frame_rate: float | None = None,
) -> bool:
    """Exclude cuts touching this ramp, not unrelated cuts in the same span.

    Internal cut times already use the selected span's affine master clock.
    Include the two frames used by optical recovery and one frame of timing
    tolerance. An unlocalized camera-cut failure remains unknown/unsafe.
    """
    if (choice.exit_scene_cut_near if outgoing else choice.entry_scene_cut_near):
        return True
    cuts = choice.internal_scene_cut_master_seconds
    if not cuts:
        return choice.face_match_candidate_failure == "internal_camera_cut"
    if any(
        not math.isfinite(cut)
        or not choice.canonical_master_start <= cut <= choice.canonical_master_end
        for cut in cuts
    ):
        return True
    if native_frame_window is not None:
        if (len(native_frame_window) != 2
                or any(type(value) is not int or value < 0 for value in native_frame_window)
                or native_frame_window[1] < native_frame_window[0]
                or isinstance(native_frame_rate, bool) or native_frame_rate is None
                or not math.isfinite(native_frame_rate) or native_frame_rate <= 0):
            raise ValueError("native cut guard requires a verified forward frame window")
        first_native, last_native = native_frame_window
        # Internal cut events were mapped by this selected span's affine
        # endpoints. Recover their native index and compare against the actual
        # rendered window. A camera cut AT the first visible incoming frame
        # is its shot entry, not a camera switch inside an alignment ramp.
        cut_indices = tuple(int((choice.source_local_start
            + (cut-choice.canonical_master_start)*choice.source_playback_rate)
            * native_frame_rate + .5) for cut in cuts)
        return any(first_native < cut <= last_native for cut in cut_indices)
    first, last = _boundary_window_frame_indices(outgoing=outgoing,
        frame_count=frame_count, window_frames=window_frames)
    start = choice.canonical_master_start + (first - 1) / fps
    end = choice.canonical_master_start + (last + 1) / fps
    return any(start - 1e-9 <= cut <= end + 1e-9 for cut in cuts)


def _union_box(
    boxes: Sequence[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    if not boxes:
        raise ValueError("cannot union an empty box set")
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    return (left, top, right - left, bottom - top)


def _upper_body_box(
    person_box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    left, top, width, height = person_box
    # Screen-space shoulders/hands/hips anchor.  The full person box remains a
    # separate containment envelope, so matching the upper body cannot crop
    # hands or feet out of the decoded settle window.
    inset = width * 0.04
    return (left + inset, top, width - 2.0 * inset, height * 0.66)


def _central_person_box(
    boxes: Sequence[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    ranked = sorted(
        boxes,
        key=lambda box: (
            math.hypot(_center(box)[0] - 0.5, _center(box)[1] - 0.48),
            -(box[2] * box[3]),
            box[0],
        ),
    )
    if len(ranked) > 1:
        first_distance = math.hypot(
            _center(ranked[0])[0] - 0.5,
            _center(ranked[0])[1] - 0.48,
        )
        second_distance = math.hypot(
            _center(ranked[1])[0] - 0.5,
            _center(ranked[1])[1] - 0.48,
        )
        first_area = ranked[0][2] * ranked[0][3]
        second_area = ranked[1][2] * ranked[1][3]
        if abs(first_distance - second_distance) < 0.035 and (
            min(first_area, second_area) / max(first_area, second_area) > 0.82
        ):
            return None
    return ranked[0]


def _associated_person_box(
    face_box: tuple[float, float, float, float],
    boxes: Sequence[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    face_center = _center(face_box)
    containing = tuple(
        box
        for box in boxes
        if box[0] - 0.02 <= face_center[0] <= box[0] + box[2] + 0.02
        and box[1] - 0.02 <= face_center[1] <= box[1] + box[3] + 0.02
    )
    if not containing:
        return None
    return min(containing, key=lambda box: math.hypot(
        _center(box)[0] - face_center[0], _center(box)[1] - face_center[1]
    ))


def _has_unambiguous_face(observation: BoundarySubjectObservation) -> bool:
    return bool(observation.face_box_xywh is not None and (
        observation.face_count == 1 or observation.face_is_unambiguous_dominant))


def _close_face_head_envelope(
    box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Protect measured face plus head room, not a frame-truncated torso.

    This is a framing envelope, not inferred anatomy or pose. Only the padding
    outside the original picture is clipped; no visible face pixel is removed.
    Medium and wide shots continue to use the full performer envelopes.
    """
    x, y, width, height = box
    left, top = max(0.0, x - .15 * width), max(0.0, y - .25 * height)
    right, bottom = min(1.0, x + 1.15 * width), min(1.0, y + 1.10 * height)
    return (left, top, right - left, bottom - top)


def _shot_policy(observation: BoundarySubjectObservation) -> str:
    boxes = observation.formation_boxes_xywh
    observed_people = max(observation.detected_person_count, len(boxes))
    if (
        _has_unambiguous_face(observation)
        and observation.face_box_xywh[3] >= 0.2184
    ):
        return "close_face"
    if (
        _has_unambiguous_face(observation)
        and observation.face_box_xywh[3] >= 0.0924
    ):
        # Match the measured non-wide face scale used by describe_frame and
        # decoded montage review (.42 * .22). Background/foreground people
        # do not turn a dominant medium face into a formation-wide shot.
        # This grants no identity or head-only crop; medium containment still
        # protects the associated performer throughout every transformed frame.
        return "medium_focus"
    if observed_people >= 3 or len(boxes) >= 3:
        return "wide_formation"
    if _has_unambiguous_face(observation):
        return "medium_focus"
    if boxes:
        if len(boxes) == 1:
            box = boxes[0]
            clipped = box[1] <= 0.035 or box[1] + box[3] >= 0.965
            if clipped and box[3] >= 0.68:
                return "close_upper_body"
        return "medium_focus"
    if (
        observation.silhouette_box_xywh is not None
        and observation.silhouette_confidence >= 0.72
    ):
        box = observation.silhouette_box_xywh
        if box[2] >= 0.56 or box[3] < 0.50:
            return "wide_formation"
        return "medium_silhouette"
    return "unknown"


def _paired_framing_policies(outgoing, incoming):
    """Recognize a shared tight composition, not an arbitrary medium crop.

    A frame-truncated torso is not a measured full body/feet envelope. Only
    two dominant, landmark-backed faces belonging to the central performers
    can use the head-framing envelope for this pair. Full bodies, off-center
    background faces and one-sided/ambiguous evidence keep their old policy.
    The selected policy is fixed through the complete dense motion window.
    """
    policies = (_shot_policy(outgoing), _shot_policy(incoming))
    def tight(observation):
        if (not _has_unambiguous_face(observation)
                or observation.face_landmarks_xy is None
                or len(observation.face_landmarks_xy) != 5
                or observation.face_eye_scale is None):
            return False
        face = observation.face_box_xywh
        assert face is not None
        if face[3] < .14 or not .30 <= _center(face)[0] <= .70:
            return False
        person = _associated_person_box(face, observation.formation_boxes_xywh)
        if person is None or person != _central_person_box(observation.formation_boxes_xywh):
            return False
        return (person[1] + person[3] >= .965 and face[3] / person[3] >= .20
                and _center(face)[1] <= person[1] + .55 * person[3])
    if all(policy in {'close_face', 'medium_focus'} for policy in policies) and all(
            tight(observation) for observation in (outgoing, incoming)):
        return tuple('tight_medium_face' if policy == 'medium_focus' else policy for policy in policies)
    return policies


def _subject(
    observation: BoundarySubjectObservation,
    *,
    preferred_kind: str | None,
    settings: BoundarySubjectAlignmentConfig,
    framing_policy: str | None = None,
) -> _SubjectGeometry | None:
    boxes = observation.formation_boxes_xywh
    if preferred_kind in (None, "face") and (
        _has_unambiguous_face(observation)
    ):
        face_box = observation.face_box_xywh
        assert face_box is not None
        performer = _associated_person_box(face_box, boxes)
        containment = (
            (_close_face_head_envelope(face_box),)
            if (framing_policy or _shot_policy(observation)) in {"close_face", "tight_medium_face"}
            else (performer,) if performer is not None else (face_box,)
        )
        return _SubjectGeometry(
            "face",
            face_box,
            observation.face_eye_scale or face_box[3],
            observation.face_head_roll_radians,
            containment,
            containment,
        )
    if preferred_kind in (None, "upper_body"):
        performer = _central_person_box(boxes)
        if performer is not None and len(boxes) <= 2:
            upper = _upper_body_box(performer)
            return _SubjectGeometry(
                "upper_body",
                upper,
                upper[3],
                None,
                (performer,),
                (performer,),
            )
    if preferred_kind in (None, "formation") and len(boxes) >= 2:
        aggregate = _union_box(boxes)
        return _SubjectGeometry(
            "formation",
            aggregate,
            aggregate[3],
            None,
            boxes,
            boxes,
        )
    if preferred_kind in (None, "silhouette") and (
        observation.silhouette_box_xywh is not None
        and observation.silhouette_confidence >= settings.minimum_silhouette_confidence
    ):
        return _SubjectGeometry(
            "silhouette",
            observation.silhouette_box_xywh,
            observation.silhouette_box_xywh[3],
            None,
        )
    return None


def _common_subjects(
    outgoing: BoundarySubjectObservation,
    incoming: BoundarySubjectObservation,
    *,
    settings: BoundarySubjectAlignmentConfig,
) -> tuple[_SubjectGeometry | None, _SubjectGeometry | None, str | None]:
    outgoing_policy, incoming_policy = _paired_framing_policies(outgoing, incoming)
    outgoing_wide = outgoing_policy == "wide_formation"
    incoming_wide = incoming_policy == "wide_formation"
    if outgoing_wide != incoming_wide:
        return (None, None, "shot_policy_family_mismatch")
    preferred: tuple[str, ...]
    if outgoing_wide:
        preferred = ("formation", "silhouette")
    else:
        preferred = ("face", "upper_body", "silhouette")
    for kind in preferred:
        left = _subject(outgoing, preferred_kind=kind, settings=settings, framing_policy=outgoing_policy)
        right = _subject(incoming, preferred_kind=kind, settings=settings, framing_policy=incoming_policy)
        if left is not None and right is not None:
            return (left, right, None)
    return (None, None, "exact_common_subject_geometry_unavailable")


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)


def _geometry_mismatch(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_center = _center(left)
    right_center = _center(right)
    center = min(1.0, math.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1]) / 0.35)
    height = min(1.0, abs(math.log(left[3] / right[3])) / math.log(1.8))
    width = min(1.0, abs(math.log(left[2] / right[2])) / math.log(1.8))
    return 0.58 * center + 0.24 * height + 0.18 * width


def _layout_mismatch(
    left: Sequence[tuple[float, float, float, float]],
    right: Sequence[tuple[float, float, float, float]],
) -> float:
    if not left or not right:
        return 1.0
    left_sorted = sorted(left, key=lambda box: _center(box)[0])
    right_sorted = sorted(right, key=lambda box: _center(box)[0])
    count_penalty = abs(len(left_sorted) - len(right_sorted)) / max(
        len(left_sorted), len(right_sorted)
    )
    comparable = min(len(left_sorted), len(right_sorted))
    pair_mismatch = 0.0
    for left_box, right_box in zip(
        left_sorted[:comparable], right_sorted[:comparable], strict=True
    ):
        pair_mismatch += _geometry_mismatch(left_box, right_box)
    return min(1.0, 0.45 * count_penalty + 0.55 * pair_mismatch / comparable)


def _subject_zoom_cap(
    kind: str,
    settings: BoundarySubjectAlignmentConfig,
) -> float:
    return {
        "face": settings.maximum_face_zoom,
        "upper_body": settings.maximum_upper_body_zoom,
        "formation": settings.maximum_formation_zoom,
        "silhouette": settings.maximum_silhouette_zoom,
    }[kind]


def _subject_center_shift_cap(
    kind: str,
    settings: BoundarySubjectAlignmentConfig,
) -> float:
    return {
        "face": settings.maximum_face_center_shift,
        "upper_body": settings.maximum_upper_body_center_shift,
        "formation": settings.maximum_formation_center_shift,
        "silhouette": settings.maximum_silhouette_center_shift,
    }[kind]


def _subject_visibility_threshold(
    kind: str,
    settings: BoundarySubjectAlignmentConfig,
) -> float:
    if kind == "face":
        # If a full performer envelope is available, the denser performer
        # threshold below is used for every containment box.
        return settings.minimum_face_visible_fraction
    if kind in {"upper_body", "formation"}:
        return settings.minimum_performer_visible_fraction
    return settings.minimum_silhouette_visible_fraction


def _project_point(
    point: tuple[float, float],
    state: _TransformState,
    *,
    aspect: float = 16.0 / 9.0,
) -> tuple[float, float]:
    dx = point[0] - 0.5
    dy = point[1] - 0.5
    radians = math.radians(state.roll_degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    rotated_x = 0.5 + cosine * dx - sine * dy / aspect
    rotated_y = 0.5 + sine * aspect * dx + cosine * dy
    return (
        state.zoom * (rotated_x - state.center_x) + 0.5,
        state.zoom * (rotated_y - state.center_y) + 0.5,
    )


def _project_box(
    box: tuple[float, float, float, float],
    state: _TransformState,
) -> tuple[tuple[float, float, float, float], float]:
    x_value, y_value, width, height = box
    points = tuple(
        _project_point(point, state)
        for point in (
            (x_value, y_value),
            (x_value + width, y_value),
            (x_value, y_value + height),
            (x_value + width, y_value + height),
        )
    )
    left = min(point[0] for point in points)
    right = max(point[0] for point in points)
    top = min(point[1] for point in points)
    bottom = max(point[1] for point in points)
    projected = (left, top, right - left, bottom - top)
    area = max(1e-12, projected[2] * projected[3])
    visible_width = max(0.0, min(1.0, right) - max(0.0, left))
    visible_height = max(0.0, min(1.0, bottom) - max(0.0, top))
    return (projected, min(1.0, visible_width * visible_height / area))


def _optical_flow_box(
    reference_frame: np.ndarray,
    target_frame: np.ndarray,
    reference_box: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    """Track one nearby-frame box with strict forward/backward evidence.

    This is a containment fallback, never endpoint alignment evidence.  It is
    accepted only across at most two output frames by the caller and retains no
    keypoints or appearance descriptor in the plan.
    """

    if reference_frame.shape != target_frame.shape:
        return None
    reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
    height, width = reference_gray.shape
    x_value, y_value, box_width, box_height = reference_box
    x0 = max(0, min(width - 2, int(math.floor(x_value * width))))
    y0 = max(0, min(height - 2, int(math.floor(y_value * height))))
    x1 = max(x0 + 2, min(width, int(math.ceil((x_value + box_width) * width))))
    y1 = max(y0 + 2, min(height, int(math.ceil((y_value + box_height) * height))))
    mask = np.zeros_like(reference_gray)
    mask[y0:y1, x0:x1] = 255
    points = cv2.goodFeaturesToTrack(
        reference_gray,
        maxCorners=80,
        qualityLevel=0.012,
        minDistance=4.0,
        mask=mask,
        blockSize=5,
    )
    if points is None or len(points) < 12:
        return None
    forward, forward_status, _forward_error = cv2.calcOpticalFlowPyrLK(
        reference_gray,
        target_gray,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if forward is None or forward_status is None:
        return None
    backward, backward_status, _backward_error = cv2.calcOpticalFlowPyrLK(
        target_gray,
        reference_gray,
        forward,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if backward is None or backward_status is None:
        return None
    original = points.reshape(-1, 2)
    tracked = forward.reshape(-1, 2)
    returned = backward.reshape(-1, 2)
    valid = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & (np.linalg.norm(returned - original, axis=1) <= 1.25)
    )
    if int(valid.sum()) < 10:
        return None
    cv2.setRNGSeed(1937)
    affine, inlier_mask = cv2.estimateAffinePartial2D(
        original[valid],
        tracked[valid],
        method=cv2.RANSAC,
        ransacReprojThreshold=1.8,
        maxIters=2000,
        confidence=0.995,
        refineIters=15,
    )
    if affine is None or inlier_mask is None:
        return None
    inliers = int(inlier_mask.reshape(-1).sum())
    if inliers < 8 or inliers / int(valid.sum()) < 0.65:
        return None
    linear = affine[:, :2].astype(np.float64)
    determinant = float(np.linalg.det(linear))
    scale = math.sqrt(max(0.0, float(linear[0, 0] ** 2 + linear[1, 0] ** 2)))
    if determinant <= 0.0 or not 0.85 <= scale <= 1.18:
        return None
    center = np.asarray(
        [[[(x_value + box_width / 2.0) * width, (y_value + box_height / 2.0) * height]]],
        dtype=np.float32,
    )
    projected_center = cv2.transform(center, affine)[0, 0]
    center_shift = math.hypot(
        float(projected_center[0]) / width - (x_value + box_width / 2.0),
        float(projected_center[1]) / height - (y_value + box_height / 2.0),
    )
    if center_shift > 0.08:
        return None
    corners = np.asarray(
        [[
            [x_value * width, y_value * height],
            [(x_value + box_width) * width, y_value * height],
            [x_value * width, (y_value + box_height) * height],
            [(x_value + box_width) * width, (y_value + box_height) * height],
        ]],
        dtype=np.float32,
    )
    projected = cv2.transform(corners, affine)[0]
    left, top = projected.min(axis=0)
    right, bottom = projected.max(axis=0)
    box = (
        float(left / width),
        float(top / height),
        float((right - left) / width),
        float((bottom - top) / height),
    )
    if (
        box[0] < -1e-6
        or box[1] < -1e-6
        or box[2] <= 0.0
        or box[3] <= 0.0
        or box[0] + box[2] > 1.000001
        or box[1] + box[3] > 1.000001
    ):
        return None
    return tuple(min(1.0, max(0.0, value)) for value in box)


def _clamp_center(
    desired_x: float,
    desired_y: float,
    *,
    zoom: float,
    maximum_shift: float,
) -> tuple[float, float]:
    half = 0.5 / zoom
    low = half
    high = 1.0 - half
    center_x = min(high, max(low, desired_x))
    center_y = min(high, max(low, desired_y))
    dx = center_x - 0.5
    dy = center_y - 0.5
    distance = math.hypot(dx, dy)
    if distance > maximum_shift and distance > 1e-12:
        scale = maximum_shift / distance
        center_x = 0.5 + dx * scale
        center_y = 0.5 + dy * scale
    return (center_x, center_y)


def _target_state(
    *,
    box: tuple[float, float, float, float],
    target_box: tuple[float, float, float, float],
    zoom: float,
    roll_degrees: float,
    maximum_center_shift: float,
    settings: BoundarySubjectAlignmentConfig,
) -> tuple[_TransformState | None, bool]:
    source_center = _center(box)
    target_center = _center(target_box)
    desired_x = source_center[0] - (target_center[0] - 0.5) / zoom
    desired_y = source_center[1] - (target_center[1] - 0.5) / zoom
    center_x, center_y = _clamp_center(
        desired_x,
        desired_y,
        zoom=zoom,
        maximum_shift=maximum_center_shift,
    )
    roll_applied = abs(roll_degrees) > 1e-6
    candidate = _TransformState(zoom, center_x, center_y, roll_degrees)
    try:
        candidate.keyframe(0.0)
    except ValueError:
        if not roll_applied:
            return (None, False)
        # Roll evidence is optional.  Scale/position still improve a cut when
        # the renderer cannot hide the rotated corners inside this source's
        # quality-safe zoom ceiling.
        candidate = _TransformState(zoom, center_x, center_y, 0.0)
        try:
            candidate.keyframe(0.0)
        except ValueError:
            return (None, False)
        roll_applied = False
    return (candidate, roll_applied)


def _joint_face_alignment_states(
    outgoing: _SubjectGeometry,
    incoming: _SubjectGeometry,
    *,
    maximum_zooms: tuple[float, float],
    settings: BoundarySubjectAlignmentConfig,
) -> tuple[tuple[_TransformState, _TransformState, float, float], ...]:
    """Bounded common framing, still requiring two complete dense reviews.

    A nearly equal-size face can have a large vertical displacement. Scaling
    only the smaller side gives too little crop margin to pan. Fund modest
    zoom on BOTH original rasters and split the displacement between them.
    No zoom-out, synthesized pixels, roll, identity claim or relaxed gate.
    """
    if outgoing.kind != 'face' or incoming.kind != 'face':
        return ()
    subjects = (outgoing, incoming)
    baseline = _geometry_mismatch(outgoing.alignment_box, incoming.alignment_box)
    baseline_layout = _layout_mismatch(outgoing.layout_boxes, incoming.layout_boxes)
    centers = tuple(_center(subject.alignment_box) for subject in subjects)
    candidates = {}
    for extra in (1.02, 1.05, 1.08, 1.12, 1.18, 1.25, 1.33, 1.50):
        common_scale = max(outgoing.scale, incoming.scale) * extra
        zooms = tuple(common_scale / subject.scale for subject in subjects)
        if any(zoom > maximum + 1e-9 for zoom, maximum in zip(zooms, maximum_zooms)):
            continue
        for weight in (0., .25, .5, .75, 1.):
            center = tuple(centers[0][axis]*weight + centers[1][axis]*(1.-weight)
                           for axis in (0, 1))
            target = (center[0]-.005, center[1]-.005, .01, .01)
            states = tuple(_target_state(box=subject.alignment_box, target_box=target,
                zoom=zoom, roll_degrees=0., maximum_center_shift=settings.maximum_paired_face_center_shift,
                settings=settings)[0] for subject, zoom in zip(subjects, zooms))
            if any(state is None for state in states):
                continue
            visibility = min(_project_box(box, state)[1]
                for subject, state in zip(subjects, states)
                for box in (subject.alignment_box, *subject.containment_boxes))
            if visibility + 1e-9 < settings.minimum_face_visible_fraction:
                continue
            mismatch = _geometry_mismatch(*(_project_box(subject.alignment_box, state)[0]
                for subject, state in zip(subjects, states)))
            layout = _layout_mismatch(*(tuple(_project_box(box, state)[0] for box in subject.layout_boxes)
                for subject, state in zip(subjects, states)))
            if (layout > settings.maximum_transformed_layout_mismatch
                    or layout > baseline_layout + settings.maximum_layout_worsening
                    or baseline-mismatch < settings.minimum_geometry_improvement
                    or (baseline-mismatch)/max(1e-9, baseline) < settings.minimum_geometry_improvement_fraction):
                continue
            candidates[states] = (states[0], states[1], mismatch, layout)
    # Subpixel floating-point noise is not a reason to zoom farther. Prefer
    # the smallest raster consumption among equivalent geometric matches.
    return tuple(sorted(candidates.values(), key=lambda item: (round(item[2], 6), item[0].zoom+item[1].zoom))[:5])


def _headroom(
    source_index: int,
    resolution: tuple[int, int] | None,
    *,
    output_width: int,
    output_height: int,
    base_crop_scale: float,
    settings: BoundarySubjectAlignmentConfig,
    base_crop_width_scale: float | None = None,
) -> SourceResolutionHeadroom:
    if resolution is None:
        profile = "unknown_resolution_conservative"
        profile_max = settings.unknown_resolution_max_zoom
        native: float | None = None
        width: int | None = None
        height: int | None = None
    else:
        width, height = resolution
        width_scale = base_crop_scale if base_crop_width_scale is None else base_crop_width_scale
        # Match the renderer's decimal expression and even-pixel crop exactly.
        width_pixels = math.floor(width * float(f'{width_scale:.8f}') / 2.) * 2
        height_pixels = math.floor(height * float(f'{base_crop_scale:.8f}') / 2.) * 2
        native = min(width_pixels / output_width, height_pixels / output_height)
        if height >= 2000 and native >= 1.75:
            profile = "2160p_native_crop_headroom"
            profile_max = settings.native_2160_max_zoom
        elif height >= 1300 and native >= 1.20:
            profile = "1440p_native_crop_headroom"
            profile_max = settings.native_1440_max_zoom
        else:
            profile = "1080p_or_lower_conservative"
            profile_max = settings.conservative_1080_max_zoom
    effective = min(settings.renderer_schema_max_zoom, profile_max)
    if native is not None and profile in {"2160p_native_crop_headroom", "1440p_native_crop_headroom"}:
        # A base crop may leave less than the profile's nominal pixel budget.
        # Larger face matches cannot magnify an already downsampled canvas.
        effective = min(effective, native)
    if native is not None and native < 1.0:
        # A pre-existing crop already consumes native pixels.  Keep only a very
        # small output-space correction; this is still explicitly reported as
        # resampling rather than native crop headroom.
        effective = min(effective, settings.unknown_resolution_max_zoom)
    return SourceResolutionHeadroom(
        source_index,
        width,
        height,
        native,
        profile,
        profile_max,
        settings.renderer_schema_max_zoom,
        effective,
    )


def _dedupe_points(
    points: Sequence[tuple[float, _TransformState]],
) -> tuple[tuple[float, _TransformState], ...]:
    ordered: dict[float, _TransformState] = {}
    for time_value, state in sorted(points, key=lambda item: item[0]):
        ordered[round(float(time_value), 9)] = state
    return tuple((time_value, state) for time_value, state in ordered.items())


def _track_for_segment(
    *,
    last_frame_time: float,
    ramp_frames: int,
    fps: float,
    entry: tuple[int, _TransformState] | None,
    exit: tuple[int, _TransformState] | None,
) -> EditorialTransformTrack | None:
    if entry is None and exit is None:
        return None
    ramp = (ramp_frames - 1) / fps
    points: list[tuple[float, _TransformState]] = []
    neutral = _TransformState()
    if entry is not None:
        points.append((0.0, entry[1]))
    else:
        points.append((0.0, neutral))
    if entry is not None and exit is not None:
        if last_frame_time > 2.0 * ramp + 1e-9:
            points.extend(((ramp, neutral), (last_frame_time - ramp, neutral)))
        # A short segment interpolates directly between its two verified cut
        # states.  Dense validation below reviews that exact composite curve.
    elif entry is not None and last_frame_time > ramp + 1e-9:
        points.append((ramp, neutral))
    elif exit is not None and last_frame_time > ramp + 1e-9:
        points.append((last_frame_time - ramp, neutral))
    points.append((last_frame_time, exit[1] if exit is not None else neutral))
    deduped = _dedupe_points(points)
    if all(state.neutral for _time, state in deduped):
        return None
    reason_indices = sorted(
        {value[0] for value in (entry, exit) if value is not None}
    )
    return EditorialTransformTrack(
        tuple(state.keyframe(time_value) for time_value, state in deduped),
        reason=(
            "boundary_local_exact_subject_alignment_b"
            + "_b".join(str(index) for index in reason_indices)
        ),
    )


@isolated_optional()
def build_boundary_subject_alignment_plan(
    *,
    visual_timeline: VisualTimelinePlan,
    sources: Sequence[Path],
    transitions: Sequence[RenderTransition],
    face_match_plan: FaceMatchPlan | None = None,
    fps: float = 29.97,
    output_width: int = 1920,
    output_height: int = 1080,
    frame_provider: FrameProvider | None = None,
    observation_provider: ObservationProvider | None = None,
    source_resolutions: Sequence[tuple[int, int] | None] | None = None,
    render_frame_clocks: Sequence[VerifiedRenderFrameClock | None] | None = None,
    config: BoundarySubjectAlignmentConfig | None = None,
) -> BoundarySubjectAlignmentPlan:
    """Build exact-cut local tracks and validate every transformed frame."""

    settings = config or BoundarySubjectAlignmentConfig()
    choices = visual_timeline.choices
    if not choices:
        raise ValueError("visual timeline cannot be empty")
    if len(transitions) != len(choices) - 1:
        raise ValueError("boundary alignment needs one transition per boundary")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("boundary alignment fps must be positive and finite")
    if output_width < 320 or output_height < 180:
        raise ValueError("boundary alignment output size is too small")
    if abs(output_width / output_height - 16.0 / 9.0) > 0.01:
        raise ValueError("boundary subject alignment currently requires 16:9 output")
    if any(not 0 <= item.source_index < len(sources) for item in choices):
        raise ValueError("visual timeline references an unknown source")
    if source_resolutions is not None and len(source_resolutions) != len(sources):
        raise ValueError("source resolution array must match sources")

    counts = _frame_counts(choices, fps)
    face_matches = _face_matches(face_match_plan)
    renderer_bound = frame_provider is None or render_frame_clocks is not None
    if render_frame_clocks is not None and (len(render_frame_clocks) != len(sources)
            or any(clock is not None and not isinstance(clock, VerifiedRenderFrameClock)
                   for clock in render_frame_clocks)):
        raise ValueError("alignment render clocks must match the source domain")
    used_sources = {choice.source_index for choice in choices}
    if render_frame_clocks is None:
        native_clocks = ({index:clock for index in sorted(used_sources)
            if (clock := probe_render_frame_clock(Path(sources[index]))) is not None}
            if renderer_bound else {})
    else:
        native_clocks = {index:clock for index,clock in enumerate(render_frame_clocks)
            if clock is not None}
    if renderer_bound:
        output_starts, native_samples = _alignment_native_frames(
            choices, sources, transitions, counts, fps, native_clocks)
        clock_basis = ("renderer_bound_output_ticks_v2" if used_sources <= native_clocks.keys()
            else "partially_verified_renderer_output_ticks_v2")
    else:
        output_starts, native_samples = (), {}
        clock_basis = "unverified_injected_segment_clock_v1"
    capture: _CaptureProvider | None = None
    if frame_provider is None:
        capture = _CaptureProvider(tuple(Path(item) for item in sources))
        frames = capture.frame_at
    else:
        frames = frame_provider
    observations = observation_provider or _default_observation_provider
    if source_resolutions is None:
        resolutions = tuple(
            capture.resolution(index) if capture is not None else None
            for index in range(len(sources))
        )
    else:
        resolutions = tuple(source_resolutions)

    def native_sample_at(choice_index: int, frame_index: int):
        if not renderer_bound or not 0 <= frame_index < counts[choice_index]:
            return None
        sample = native_samples.get(output_starts[choice_index]+frame_index)
        return sample if sample is not None and sample.source_index == choices[choice_index].source_index else None

    def source_seconds_at(choice_index: int, frame_index: int) -> float:
        sample = native_sample_at(choice_index, frame_index)
        # Missing/overlap clock values are audit coordinates only; they can
        # never reach frame decoding or authorize an alignment below.
        return (sample.source_request_seconds if sample is not None else
            _source_seconds(choices[choice_index], frame_index, fps=fps))

    def boundary_clock_evidence(boundary_index):
        left = native_sample_at(boundary_index, counts[boundary_index]-1)
        right = native_sample_at(boundary_index+1, 0)
        return {f"{side}_{field}":getattr(sample, field) if sample is not None else None
            for side,sample in (("outgoing",left),("incoming",right))
            for field in ("native_frame_index","output_frame_index")}

    def native_cut_window_arguments(choice_index, *, outgoing):
        if not renderer_bound:
            return {}
        first, last = _boundary_window_frame_indices(outgoing=outgoing,
            frame_count=counts[choice_index], window_frames=settings.window_frames)
        before, after = native_sample_at(choice_index, first), native_sample_at(choice_index, last)
        if before is None or after is None:
            return {}  # Unknown/overlap evidence retains the conservative guard.
        return {'native_frame_window': (before.native_frame_index,after.native_frame_index),
            'native_frame_rate':float(native_clocks[choices[choice_index].source_index].frame_rate)}

    def base_crop_for(choice_index: int, frame_index: int) -> tuple[float, float, float, float] | None:
        match = face_matches.get(choices[choice_index].segment_index)
        if match is None or not match.applied or match.pan_crop is None:
            return None
        return _fixed_crop_at(match.pan_crop, frame_index / fps)

    def maximum_zoom_at(choice_index, frame_index, subject_kind):
        crop = base_crop_for(choice_index, frame_index)
        source = choices[choice_index].source_index
        return min(_headroom(source, resolutions[source], output_width=output_width,
            output_height=output_height, base_crop_scale=crop[3] if crop else 1.,
            base_crop_width_scale=crop[2] if crop else 1., settings=settings).effective_max_zoom,
            _subject_zoom_cap(subject_kind, settings))

    # The serialized source-level value describes untouched source pixels.
    # A segment-specific base crop is accounted for again at the exact side we
    # intend to transform; one cropped close-up must not penalize every raw shot
    # selected from the same broadcast.
    headroom_items = [
        _headroom(
            source_index,
            resolutions[source_index],
            output_width=output_width,
            output_height=output_height,
            base_crop_scale=1.0,
            settings=settings,
        )
        for source_index in range(len(sources))
    ]

    canonical_frame_cache: dict[tuple[int, int], np.ndarray | None] = {}
    observation_cache: dict[tuple[int, int], BoundarySubjectObservation | None] = {}

    def canonical_frame_at(choice_index: int, frame_index: int) -> np.ndarray | None:
        key = (choice_index, frame_index)
        if key in canonical_frame_cache:
            value = canonical_frame_cache[key]
            return None if value is None else value.copy()
        choice = choices[choice_index]
        sample = native_sample_at(choice_index, frame_index)
        if renderer_bound and sample is None:
            canonical_frame_cache[key] = None
            return None
        source_seconds = source_seconds_at(choice_index, frame_index)
        decoded = (capture.frame_at_native_index(choice.source_index, sample.native_frame_index,
                native_clocks[choice.source_index])
            if capture is not None and sample is not None else frames(choice.source_index, source_seconds))
        if decoded is None:
            canonical_frame_cache[key] = None
            return None
        try:
            canonical = _canonical_frame(
                decoded,
                width=settings.analysis_width,
                height=settings.analysis_height,
                base_crop=base_crop_for(choice_index, frame_index),
            )
        except (TypeError, ValueError, cv2.error):
            canonical = None
        canonical_frame_cache[key] = canonical
        return None if canonical is None else canonical.copy()

    def observation_at(choice_index: int, frame_index: int) -> BoundarySubjectObservation | None:
        key = (choice_index, frame_index)
        if key in observation_cache:
            return observation_cache[key]
        canonical = canonical_frame_at(choice_index, frame_index)
        if canonical is None:
            observation_cache[key] = None
            return None
        try:
            result = observations(canonical)
        except (TypeError, ValueError, cv2.error):
            result = None
        observation_cache[key] = result
        return result

    def optical_subject_at(
        choice_index: int,
        frame_index: int,
        preferred_kind: str,
        framing_policy: str | None = None,
    ) -> _SubjectGeometry | None:
        if preferred_kind not in {"face", "silhouette"}:
            return None
        target = canonical_frame_at(choice_index, frame_index)
        if target is None:
            return None
        for distance in (1, 2):
            for reference_index in (frame_index - distance, frame_index + distance):
                if not 0 <= reference_index < counts[choice_index]:
                    continue
                reference_observation = observation_at(
                    choice_index,
                    reference_index,
                )
                if reference_observation is None:
                    continue
                reference_subject = _subject(
                    reference_observation,
                    preferred_kind=preferred_kind,
                    settings=settings,
                    framing_policy=framing_policy,
                )
                if reference_subject is None:
                    continue
                reference = canonical_frame_at(choice_index, reference_index)
                if reference is None:
                    continue
                tracked_box = _optical_flow_box(
                    reference,
                    target,
                    reference_subject.alignment_box,
                )
                if tracked_box is None:
                    continue
                # Preserve every envelope proved on the reference frame.  A
                # face-only optical fallback must not silently discard an
                # associated full-performer box and thereby weaken hands/feet
                # or layout containment during a one-frame detector miss.
                tracked_by_box: dict[
                    tuple[float, float, float, float],
                    tuple[float, float, float, float],
                ] = {reference_subject.alignment_box: tracked_box}

                def tracked_boxes(
                    boxes: Sequence[tuple[float, float, float, float]],
                ) -> tuple[tuple[float, float, float, float], ...] | None:
                    values: list[tuple[float, float, float, float]] = []
                    for reference_box in boxes:
                        tracked = tracked_by_box.get(reference_box)
                        if tracked is None:
                            tracked = _optical_flow_box(
                                reference,
                                target,
                                reference_box,
                            )
                            if tracked is None:
                                return None
                            tracked_by_box[reference_box] = tracked
                        values.append(tracked)
                    return tuple(values)

                tracked_containment = tracked_boxes(
                    reference_subject.containment_boxes
                )
                tracked_layout = tracked_boxes(reference_subject.layout_boxes)
                if tracked_containment is None or tracked_layout is None:
                    continue
                return _SubjectGeometry(
                    preferred_kind,
                    tracked_box,
                    tracked_box[3],
                    None,
                    tracked_containment,
                    tracked_layout,
                )
        return None

    @lru_cache(maxsize=2048)
    def dense_subject_at(
        choice_index: int, frame_index: int,
        preferred_kind: str, framing_policy: str | None,
    ) -> tuple[_SubjectGeometry | None, bool]:
        observation = observation_at(choice_index, frame_index)
        if observation is None:
            return None, False
        subject = _subject(
            observation, preferred_kind=preferred_kind, settings=settings,
            framing_policy=framing_policy,
        )
        if subject is None and preferred_kind == "face":
            # Preserve the measured head through a brief detector dropout
            # before changing semantic scale to the entire stage silhouette.
            # This uses actual adjacent pixels and the existing forward/back
            # flow checks for every face/head/performer envelope. A coarse
            # silhouette must not bypass that stronger available evidence.
            tracked = optical_subject_at(
                choice_index, frame_index, preferred_kind, framing_policy,
            )
            if tracked is not None:
                return tracked, True
            # A verified whole-performer envelope can guard containment on a
            # transient face miss. It is not treated as a newly identified face.
            subject = _subject(observation, preferred_kind="silhouette", settings=settings)
            return subject, False
        if subject is not None:
            return subject, False
        subject = optical_subject_at(choice_index, frame_index, preferred_kind, framing_policy)
        return subject, subject is not None

    decisions: list[BoundarySubjectAlignmentDecision] = []
    candidate_by_boundary: dict[int, BoundarySubjectAlignmentDecision] = {}
    candidate_options: dict[int, tuple[BoundarySubjectAlignmentDecision, ...]] = {}
    joint_options: dict[int, tuple[BoundarySubjectAlignmentDecision, ...]] = {}

    def rejected(
        boundary_index: int,
        reason: str,
        *,
        subject_kind: str | None = None,
        baseline: float | None = None,
        ratio: float | None = None,
        outgoing_policy: str | None = None,
        incoming_policy: str | None = None,
        baseline_layout: float | None = None,
        transformed_layout: float | None = None,
    ) -> BoundarySubjectAlignmentDecision:
        left = choices[boundary_index]
        right = choices[boundary_index + 1]
        return BoundarySubjectAlignmentDecision(
            boundary_index,
            left.segment_index,
            right.segment_index,
            left.source_index,
            right.source_index,
            source_seconds_at(boundary_index, counts[boundary_index] - 1),
            source_seconds_at(boundary_index+1, 0),
            False,
            reason,
            subject_kind=subject_kind,
            baseline_geometry_mismatch=baseline,
            exact_scale_ratio=ratio,
            outgoing_shot_policy=outgoing_policy,
            incoming_shot_policy=incoming_policy,
            baseline_layout_mismatch=baseline_layout,
            transformed_layout_mismatch=transformed_layout,
            **boundary_clock_evidence(boundary_index),
        )

    try:
        for boundary_index, (left, right) in enumerate(zip(choices, choices[1:])):
            if left.source_index == right.source_index:
                continue
            transition = transitions[boundary_index]
            if transition.kind is not TransitionKind.HARD_CUT:
                decisions.append(rejected(boundary_index, "unreviewed_dissolve_overlap"))
                continue
            if renderer_bound and (native_sample_at(boundary_index, counts[boundary_index]-1) is None
                    or native_sample_at(boundary_index+1, 0) is None):
                decisions.append(rejected(boundary_index, "unverified_renderer_cut_frame"))
                continue
            outgoing_cut = _native_cut_in_boundary_window(
                left, outgoing=True, frame_count=counts[boundary_index],
                fps=fps, window_frames=settings.window_frames,
                **native_cut_window_arguments(boundary_index, outgoing=True),
            )
            incoming_cut = _native_cut_in_boundary_window(
                right, outgoing=False, frame_count=counts[boundary_index + 1],
                fps=fps, window_frames=settings.window_frames,
                **native_cut_window_arguments(boundary_index+1, outgoing=False),
            )
            cut_sides = ((left, outgoing_cut), (right, incoming_cut))
            localized_verified_cuts = (
                renderer_bound and left.source_index in native_clocks
                and right.source_index in native_clocks
                and not left.exit_scene_cut_near and not right.entry_scene_cut_near
                and all(choice.internal_scene_cut_master_seconds
                        and all(math.isfinite(value)
                                and choice.canonical_master_start <= value <= choice.canonical_master_end
                                for value in choice.internal_scene_cut_master_seconds)
                        for choice, has_cut in cut_sides if has_cut)
            )
            if (outgoing_cut or incoming_cut) and (
                not localized_verified_cuts or outgoing_cut and incoming_cut
            ):
                decisions.append(rejected(boundary_index, "broadcaster_native_internal_cut"))
                continue
            if min(counts[boundary_index], counts[boundary_index + 1]) < settings.window_frames:
                decisions.append(rejected(boundary_index, "local_window_shorter_than_required_frames"))
                continue
            outgoing_observation = observation_at(boundary_index, counts[boundary_index] - 1)
            incoming_observation = observation_at(boundary_index + 1, 0)
            if outgoing_observation is None or incoming_observation is None:
                decisions.append(rejected(boundary_index, "exact_cut_frame_decode_or_geometry_failed"))
                continue
            outgoing_policy, incoming_policy = _paired_framing_policies(
                outgoing_observation, incoming_observation)
            outgoing_subject, incoming_subject, subject_failure = _common_subjects(
                outgoing_observation,
                incoming_observation,
                settings=settings,
            )
            if outgoing_subject is None or incoming_subject is None:
                decisions.append(
                    rejected(
                        boundary_index,
                        subject_failure or "exact_common_subject_geometry_unavailable",
                        outgoing_policy=outgoing_policy,
                        incoming_policy=incoming_policy,
                    )
                )
                continue
            subject_kind = outgoing_subject.kind
            outgoing_box = outgoing_subject.alignment_box
            incoming_box = incoming_subject.alignment_box
            baseline = _geometry_mismatch(outgoing_box, incoming_box)
            baseline_layout = _layout_mismatch(
                outgoing_subject.layout_boxes,
                incoming_subject.layout_boxes,
            )
            outgoing_scale = outgoing_subject.scale
            incoming_scale = incoming_subject.scale
            ratio = max(outgoing_scale, incoming_scale) / max(
                1e-9, min(outgoing_scale, incoming_scale)
            )
            center_distance = math.hypot(
                _center(outgoing_box)[0] - _center(incoming_box)[0],
                _center(outgoing_box)[1] - _center(incoming_box)[1],
            )
            if (
                ratio <= settings.already_matched_scale_ratio
                and center_distance <= settings.already_matched_center_distance
            ):
                decisions.append(
                    rejected(
                        boundary_index,
                        "exact_subject_already_matched",
                        subject_kind=subject_kind,
                        baseline=baseline,
                        ratio=ratio,
                        outgoing_policy=outgoing_policy,
                        incoming_policy=incoming_policy,
                        baseline_layout=baseline_layout,
                    )
                )
                continue
            if (subject_kind == 'face' and not outgoing_cut and not incoming_cut
                    and all(observation.face_landmarks_xy is not None
                            and len(observation.face_landmarks_xy) == 5
                            and observation.face_eye_scale is not None
                            for observation in (outgoing_observation, incoming_observation))):
                paired_states = _joint_face_alignment_states(outgoing_subject, incoming_subject,
                    maximum_zooms=(maximum_zoom_at(boundary_index, counts[boundary_index]-1, subject_kind),
                                   maximum_zoom_at(boundary_index+1, 0, subject_kind)), settings=settings)
                template = rejected(boundary_index, 'paired_framing_not_dense_validated',
                    subject_kind=subject_kind, baseline=baseline, ratio=ratio,
                    outgoing_policy=outgoing_policy, incoming_policy=incoming_policy,
                    baseline_layout=baseline_layout)
                joint_options[boundary_index] = tuple(replace(template, eligible=True,
                    rejection_reason=None, transformed_side='both', outgoing_state=left_state,
                    incoming_state=right_state, transformed_geometry_mismatch=mismatch,
                    transformed_layout_mismatch=layout)
                    for left_state, right_state, mismatch, layout in paired_states)
            if outgoing_scale < incoming_scale:
                transformed_side = "outgoing"
                source_index = left.source_index
                box = outgoing_box
                target_box = incoming_box
                raw_roll = (
                    math.degrees(
                        incoming_subject.roll_radians
                        - outgoing_subject.roll_radians
                    )
                    if outgoing_subject.roll_radians is not None
                    and incoming_subject.roll_radians is not None
                    else 0.0
                )
            else:
                transformed_side = "incoming"
                source_index = right.source_index
                box = incoming_box
                target_box = outgoing_box
                raw_roll = (
                    math.degrees(
                        outgoing_subject.roll_radians
                        - incoming_subject.roll_radians
                    )
                    if outgoing_subject.roll_radians is not None
                    and incoming_subject.roll_radians is not None
                    else 0.0
                )
            transformed_choice_index = (
                boundary_index
                if transformed_side == "outgoing"
                else boundary_index + 1
            )
            # Only this side receives a ramp. A verified camera cut on the
            # untouched side cannot interrupt that ramp; its exact endpoint
            # remains the measured target. Never animate across a native cut
            # on the side that actually moves, or excuse an unlocalized cut.
            if (outgoing_cut if transformed_side == "outgoing" else incoming_cut):
                decisions.append(rejected(boundary_index, "broadcaster_native_internal_cut",
                    subject_kind=subject_kind, baseline=baseline, ratio=ratio,
                    outgoing_policy=outgoing_policy, incoming_policy=incoming_policy,
                    baseline_layout=baseline_layout))
                continue
            transformed_frame_index = (
                counts[transformed_choice_index] - 1
                if transformed_side == "outgoing"
                else 0
            )
            maximum_zoom = maximum_zoom_at(transformed_choice_index, transformed_frame_index, subject_kind)
            desired_zoom = ratio
            if ratio <= settings.already_matched_scale_ratio:
                desired_zoom = 1.02
            zoom = min(maximum_zoom, max(1.0, desired_zoom))
            roll_available = bool(
                subject_kind == "face"
                and outgoing_subject.roll_radians is not None
                and incoming_subject.roll_radians is not None
            )
            roll = (
                max(-settings.maximum_roll_degrees, min(settings.maximum_roll_degrees, raw_roll))
                if roll_available
                else 0.0
            )
            if abs(roll) > 1e-6:
                zoom = max(zoom, minimum_roll_safe_zoom(roll))
            if zoom > maximum_zoom + 1e-9:
                roll = 0.0
                zoom = min(maximum_zoom, max(1.0, desired_zoom))
            state, roll_applied = _target_state(
                box=box,
                target_box=target_box,
                zoom=zoom,
                roll_degrees=roll,
                maximum_center_shift=_subject_center_shift_cap(
                    subject_kind, settings
                ),
                settings=settings,
            )
            target_states = [state] if state is not None else []
            if subject_kind == "formation" and zoom > 1.08 + 1e-9:
                # A larger native-pixel budget must not erase the exact old
                # low-zoom family. Recompute its pan at that zoom: scaling a
                # stronger state's pan does not reproduce the old geometry.
                conservative_state, _ = _target_state(
                    box=box, target_box=target_box, zoom=1.08,
                    roll_degrees=0.0,
                    maximum_center_shift=_subject_center_shift_cap(subject_kind, settings),
                    settings=settings,
                )
                if conservative_state is not None:
                    target_states.append(conservative_state)
            if not target_states:
                decisions.append(
                    rejected(
                        boundary_index,
                        "renderer_edge_safety_rejection",
                        subject_kind=subject_kind,
                        baseline=baseline,
                        ratio=ratio,
                        outgoing_policy=outgoing_policy,
                        incoming_policy=incoming_policy,
                        baseline_layout=baseline_layout,
                    )
                )
                continue
            # Search back toward neutral instead of throwing away a useful
            # hard-cut correction merely because the full requested match
            # clips a hand/foot or crosses the layout gate.  Every tried state
            # still passes the renderer edge check and the same dense gates;
            # this is bounded partial alignment, never an unchecked crop.
            state_candidates = tuple(dict.fromkeys(
                _TransformState(
                    1.0 + (state.zoom - 1.0) * strength,
                    0.5 + (state.center_x - 0.5) * strength,
                    0.5 + (state.center_y - 0.5) * strength,
                    state.roll_degrees * strength,
                )
                for state in target_states
                for strength in (1.0, 0.82, 0.66, 0.50, 0.36)
            ))
            selected_state: _TransformState | None = None
            selected_transformed = baseline
            selected_layout = baseline_layout
            valid_states: list[tuple[_TransformState, float, float]] = []
            saw_containment_failure = False
            saw_layout_failure = False
            for trial_state in state_candidates:
                try:
                    trial_state.keyframe(0.0)
                except ValueError:
                    continue
                if transformed_side == "outgoing":
                    projected_alignment, alignment_visibility = _project_box(
                        outgoing_box, trial_state
                    )
                    trial_transformed = _geometry_mismatch(
                        projected_alignment, incoming_box
                    )
                    trial_layout = _layout_mismatch(
                        tuple(
                            _project_box(item, trial_state)[0]
                            for item in outgoing_subject.layout_boxes
                        ),
                        incoming_subject.layout_boxes,
                    )
                    containment_boxes = outgoing_subject.containment_boxes
                else:
                    projected_alignment, alignment_visibility = _project_box(
                        incoming_box, trial_state
                    )
                    trial_transformed = _geometry_mismatch(
                        outgoing_box, projected_alignment
                    )
                    trial_layout = _layout_mismatch(
                        outgoing_subject.layout_boxes,
                        tuple(
                            _project_box(item, trial_state)[0]
                            for item in incoming_subject.layout_boxes
                        ),
                    )
                    containment_boxes = incoming_subject.containment_boxes
                containment_visibility = min(
                    _project_box(item, trial_state)[1]
                    for item in containment_boxes
                )
                minimum_visibility = _subject_visibility_threshold(
                    subject_kind, settings
                )
                if (
                    min(alignment_visibility, containment_visibility) + 1e-9
                    < minimum_visibility
                ):
                    saw_containment_failure = True
                    continue
                if (
                    trial_layout > settings.maximum_transformed_layout_mismatch
                    or trial_layout
                    > baseline_layout + settings.maximum_layout_worsening
                ):
                    saw_layout_failure = True
                    continue
                improvement = baseline - trial_transformed
                improvement_fraction = improvement / max(1e-9, baseline)
                if (
                    improvement < settings.minimum_geometry_improvement
                    or improvement_fraction
                    < settings.minimum_geometry_improvement_fraction
                ):
                    continue
                valid_states.append((trial_state, trial_transformed, trial_layout))
                if selected_state is None:
                    selected_state = trial_state
                    selected_transformed = trial_transformed
                    selected_layout = trial_layout
            if selected_state is None:
                reason = (
                    "exact_subject_would_be_clipped"
                    if saw_containment_failure and not saw_layout_failure
                    else "post_transform_layout_mismatch_gate"
                    if saw_layout_failure
                    else "geometry_improvement_too_small"
                )
                decisions.append(
                    rejected(
                        boundary_index,
                        reason,
                        subject_kind=subject_kind,
                        baseline=baseline,
                        ratio=ratio,
                        outgoing_policy=outgoing_policy,
                        incoming_policy=incoming_policy,
                        baseline_layout=baseline_layout,
                        transformed_layout=selected_layout,
                    )
                )
                continue
            state = selected_state
            transformed = selected_transformed
            transformed_layout = selected_layout
            if transformed_side == "outgoing":
                outgoing_state = state
                incoming_state = _TransformState()
            else:
                outgoing_state = _TransformState()
                incoming_state = state
            candidate = BoundarySubjectAlignmentDecision(
                boundary_index,
                left.segment_index,
                right.segment_index,
                left.source_index,
                right.source_index,
                source_seconds_at(boundary_index, counts[boundary_index] - 1),
                source_seconds_at(boundary_index+1, 0),
                True,
                None,
                subject_kind,
                transformed_side,
                outgoing_state,
                incoming_state,
                baseline,
                transformed,
                ratio,
                roll_available,
                roll_applied,
                desired_zoom > state.zoom + 1e-9,
                0,
                0,
                outgoing_policy,
                incoming_policy,
                baseline_layout,
                transformed_layout,
                **boundary_clock_evidence(boundary_index),
            )
            decisions.append(candidate)
            candidate_by_boundary[boundary_index] = candidate
            candidate_options[boundary_index] = tuple(
                replace(candidate,
                    outgoing_state=trial_state if transformed_side == "outgoing" else _TransformState(),
                    incoming_state=trial_state if transformed_side == "incoming" else _TransformState(),
                    transformed_geometry_mismatch=trial_transformed,
                    transformed_layout_mismatch=trial_layout,
                    roll_applied=abs(trial_state.roll_degrees) > 1e-6,
                    partial_scale_alignment=desired_zoom > trial_state.zoom + 1e-9)
                for trial_state, trial_transformed, trial_layout in valid_states
            )

        def validate_track(choice_index, track, selected):
            """Same dense pixel/geometry gates for every competing curve."""
            touched = tuple(selected)
            failure: str | None = None
            validated = 0
            optical_flow_validated = 0
            for frame_index in range(counts[choice_index]):
                state_keyframe = track.evaluate(frame_index / fps)
                state = _TransformState(
                    state_keyframe.zoom,
                    state_keyframe.center_x,
                    state_keyframe.center_y,
                    state_keyframe.roll_degrees,
                )
                previous_keyframe = track.evaluate(max(0, frame_index - 1) / fps)
                previous_state = _TransformState(
                    previous_keyframe.zoom, previous_keyframe.center_x,
                    previous_keyframe.center_y, previous_keyframe.roll_degrees,
                )
                if state.neutral and previous_state.neutral:
                    continue
                validated += int(not state.neutral)
                try:
                    state.keyframe(frame_index / fps)
                except ValueError:
                    failure = "per_frame_renderer_edge_safety_rejection"
                    break
                if observation_at(choice_index, frame_index) is None:
                    failure = "per_frame_decode_or_geometry_failed"
                    break
                nearest_boundary = min(
                    touched,
                    key=lambda boundary_index: (
                        frame_index if boundary_index == choice_index - 1
                        else counts[choice_index] - 1 - frame_index),
                )
                boundary = selected[nearest_boundary]
                preferred_kind = boundary.subject_kind
                framing_policy = (
                    boundary.outgoing_shot_policy
                    if choice_index == nearest_boundary
                    else boundary.incoming_shot_policy
                )
                # Keep the verified endpoint's framing contract through this
                # short window. Detector-size jitter cannot replace a head
                # envelope with a whole torso part way through one ramp.
                subject, used_flow = dense_subject_at(
                    choice_index, frame_index, preferred_kind, framing_policy,
                )
                optical_flow_validated += int(used_flow and not state.neutral)
                if subject is None:
                    failure = "per_frame_subject_geometry_unavailable"
                    break
                visibility = min(
                    _project_box(box, state)[1]
                    for box in subject.containment_boxes
                )
                threshold = _subject_visibility_threshold(subject.kind, settings)
                if visibility + 1e-9 < threshold:
                    failure = "per_frame_subject_containment_failed"
                    break
                if frame_index == 0:
                    # The incoming endpoint already passed the exact
                    # cross-camera layout/improvement checks above.
                    continue
                previous_subject, _ = dense_subject_at(
                    choice_index, frame_index - 1, preferred_kind, framing_policy,
                )
                if previous_subject is None:
                    failure = "per_frame_subject_geometry_unavailable"
                    break
                if previous_subject.kind == subject.kind:
                    # Both frames are from the same visible source. Comparing
                    # a moving dancer to the other camera's frozen endpoint
                    # wrongly penalizes genuine choreography. Instead check
                    # the actual adjacent-frame motion before/after the curve,
                    # including its join back to a neutral transform.
                    baseline_frame_layout = _layout_mismatch(
                        previous_subject.layout_boxes, subject.layout_boxes,
                    )
                    transformed_frame_layout = _layout_mismatch(
                        tuple(_project_box(box, previous_state)[0]
                              for box in previous_subject.layout_boxes),
                        tuple(_project_box(box, state)[0] for box in subject.layout_boxes),
                    )
                else:
                    # A detector dropout changes evidence kind, not the
                    # dancer's size. Compare two measured silhouettes in the
                    # SAME source/time neighborhood, not a whole-stage mask
                    # against the other camera's frozen head. Both larger
                    # envelopes must also survive their actual transforms.
                    previous_common, _ = dense_subject_at(
                        choice_index, frame_index - 1, "silhouette", None,
                    )
                    current_common, _ = dense_subject_at(
                        choice_index, frame_index, "silhouette", None,
                    )
                    if previous_common is None or current_common is None:
                        failure = "per_frame_common_geometry_unavailable"
                        break
                    common_visibility = min(
                        _project_box(box, transform)[1]
                        for common, transform in (
                            (previous_common, previous_state), (current_common, state),
                        ) for box in common.containment_boxes
                    )
                    if common_visibility + 1e-9 < settings.minimum_silhouette_visible_fraction:
                        failure = "per_frame_subject_containment_failed"
                        break
                    baseline_frame_layout = _layout_mismatch(
                        previous_common.layout_boxes, current_common.layout_boxes,
                    )
                    transformed_frame_layout = _layout_mismatch(
                        tuple(_project_box(box, previous_state)[0] for box in previous_common.layout_boxes),
                        tuple(_project_box(box, state)[0] for box in current_common.layout_boxes),
                    )
                if (
                    transformed_frame_layout
                    > settings.maximum_transformed_layout_mismatch
                    or transformed_frame_layout
                    > baseline_frame_layout + settings.maximum_layout_worsening
                ):
                    failure = "per_frame_layout_mismatch_gate"
                    break
            return failure, validated, optical_flow_validated

        # Each cut transforms exactly one side, so a boundary belongs to one
        # segment only. Try its endpoint-valid strengths against the COMPLETE
        # visible settle, including any neighboring ramp. Choosing a weaker
        # curve is not permission to skip a failed frame or relax a gate.
        segment_tracks: dict[int, EditorialTransformTrack] = {}
        segment_boundaries: dict[int, tuple[int, ...]] = {}
        dense_attempts = 0
        for choice_index, choice in enumerate(choices):
            entry_options = tuple(item for item in candidate_options.get(choice_index - 1, ())
                                  if item.transformed_side == "incoming")
            exit_options = tuple(item for item in candidate_options.get(choice_index, ())
                                 if item.transformed_side == "outgoing")
            owned = {item.boundary_index for item in (*entry_options, *exit_options)}
            if not owned:
                continue
            # At most (10+1)*(10+1)-1 = 120 combinations, including formation's
            # unchanged conservative family. The absent-side option
            # preserves a valid neighbor when another ramp cannot be made
            # safe. Prefer both cuts, then greatest proven endpoint gain.
            combinations = [pair for pair in product((*entry_options, None), (*exit_options, None))
                            if any(item is not None for item in pair)]
            combinations.sort(key=lambda pair: (
                -sum(item is not None for item in pair),
                -sum(item.baseline_geometry_mismatch - item.transformed_geometry_mismatch
                     for item in pair if item is not None),
            ))
            selected = {}
            solo_failures: dict[int, str] = {}
            for entry_decision, exit_decision in combinations:
                trial = {item.boundary_index: item for item in (entry_decision, exit_decision)
                         if item is not None}
                dense_attempts += 1
                try:
                    track = _track_for_segment(
                        last_frame_time=(counts[choice_index] - 1) / fps,
                        ramp_frames=settings.window_frames, fps=fps,
                        entry=(entry_decision.boundary_index, entry_decision.incoming_state)
                              if entry_decision is not None else None,
                        exit=(exit_decision.boundary_index, exit_decision.outgoing_state)
                             if exit_decision is not None else None,
                    )
                    if track is None:
                        raise ValueError("non-neutral candidates produced no track")
                except ValueError:
                    failure = "composite_track_renderer_safety_rejection"
                else:
                    failure, validated, optical_flow_validated = validate_track(choice_index, track, trial)
                if failure is not None:
                    if len(trial) == 1:
                        solo_failures[next(iter(trial))] = failure
                    continue
                selected = trial
                segment_tracks[choice.segment_index] = track
                segment_boundaries[choice.segment_index] = tuple(trial)
                for boundary_index, decision in trial.items():
                    candidate_by_boundary[boundary_index] = replace(decision,
                        validated_output_frame_count=validated,
                        optical_flow_containment_frame_count=optical_flow_validated)
                break
            for boundary_index in owned - set(selected):
                candidate_by_boundary[boundary_index] = replace(
                    candidate_by_boundary[boundary_index], eligible=False,
                    rejection_reason=solo_failures.get(boundary_index, "no_compatible_dense_track"),
                    validated_output_frame_count=0, optical_flow_containment_frame_count=0,
                )

        # A passing one-sided crop is an incumbent, not necessarily the best
        # camera match. Compare the bounded bilateral family when both segments
        # are unused OR occupied solely by this cut. Neighbors are immutable;
        # failed candidates retain the complete incumbent, and publication is
        # atomic only after BOTH visible windows pass the same dense gates.
        initial_decisions = {item.boundary_index: item for item in decisions}
        for boundary_index, options in sorted(joint_options.items()):
            current = candidate_by_boundary.get(boundary_index, initial_decisions[boundary_index])
            sides = (boundary_index, boundary_index+1)
            if (not options or any(choices[index].segment_index in segment_tracks
                    and set(segment_boundaries[choices[index].segment_index]) != {boundary_index}
                    for index in sides)):
                continue
            for candidate in options:
                previous = current.transformed_geometry_mismatch if current.eligible else None
                if current.eligible:
                    if previous is None or current.transformed_layout_mismatch is None:
                        continue
                    gain = previous-candidate.transformed_geometry_mismatch
                    if (gain < settings.minimum_geometry_improvement
                            or gain/max(previous, 1e-9) < settings.minimum_geometry_improvement_fraction
                            or candidate.transformed_layout_mismatch > current.transformed_layout_mismatch+1e-9):
                        continue
                tracks = []
                counts_by_side = []
                flow_count = 0
                for index in sides:
                    outgoing = index == boundary_index
                    dense_attempts += 1
                    track = _track_for_segment(last_frame_time=(counts[index]-1)/fps,
                        ramp_frames=settings.window_frames, fps=fps,
                        entry=None if outgoing else (boundary_index, candidate.incoming_state),
                        exit=(boundary_index, candidate.outgoing_state) if outgoing else None)
                    if track is None:
                        break
                    failure, validated, tracked = validate_track(index, track, {boundary_index: candidate})
                    if failure is not None:
                        break
                    tracks.append(track)
                    counts_by_side.append(validated)
                    flow_count += tracked
                if len(tracks) != 2:
                    continue
                for index, track in zip(sides, tracks):
                    segment_tracks[choices[index].segment_index] = track
                    segment_boundaries[choices[index].segment_index] = (boundary_index,)
                candidate_by_boundary[boundary_index] = replace(candidate,
                    validated_output_frame_count=sum(counts_by_side),
                    optical_flow_containment_frame_count=flow_count,
                    outgoing_validated_output_frame_count=counts_by_side[0],
                    incoming_validated_output_frame_count=counts_by_side[1],
                    replaced_one_sided_geometry_mismatch=previous)
                break

        final_decisions = tuple(
            candidate_by_boundary.get(item.boundary_index, item)
            for item in decisions
        )
        final_segments = tuple(
            BoundarySubjectAlignmentSegment(
                choice.segment_index,
                choice.source_index,
                segment_tracks.get(choice.segment_index),
                segment_boundaries.get(choice.segment_index, ()),
            )
            for choice in choices
        )
        return BoundarySubjectAlignmentPlan(
            final_decisions,
            final_segments,
            tuple(headroom_items),
            settings.window_frames,
            sample_clock_basis=clock_basis,
            dense_track_candidate_attempt_count=dense_attempts,
        )
    finally:
        if capture is not None:
            capture.close()


__all__ = [
    "ANALYSIS_BASIS",
    "BoundarySubjectAlignmentConfig",
    "BoundarySubjectAlignmentDecision",
    "BoundarySubjectAlignmentPlan",
    "BoundarySubjectAlignmentSegment",
    "BoundarySubjectObservation",
    "FrameProvider",
    "ObservationProvider",
    "SourceResolutionHeadroom",
    "build_boundary_subject_alignment_plan",
]
