"""Reference-calibrated virtual-camera planning for natural cross-edits.

The planner is deliberately separate from member tracking and face matching.
Those layers establish *what* must stay visible; this layer adds the short,
duration-neutral zoom/pan/roll ramps commonly authored around stage-mix cuts.
The same render track can therefore sit after either a raw group frame or a
confirmed-member ``PanCropTrack``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Sequence

from .boundary_affine_plan import BoundaryAffinePlan
from .boundary_subject_alignment import BoundarySubjectAlignmentPlan
from .face_match import FaceMatchPlan, FaceMatchSegment
from .editorial_transform_events import rendered_transform_event_records
from .render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    PanCropTrack,
    RenderSegment,
    RenderTransition,
    TransitionKind,
    minimum_roll_safe_zoom,
)
from .section_director import SectionPlan
from .timeline import ShotType, screen_layout_mismatch
from .visual_score import VisualTimelineChoice, VisualTimelinePlan


@dataclass(frozen=True, slots=True)
class EditorialMotionConfig:
    # High-confidence reference audit: median 0.733 s / 9.05% authored zoom;
    # lower quartile 4.69%.  Keep ordinary ramps inside that central band.
    ramp_seconds: float = 0.75
    ordinary_zoom: float = 1.065
    close_up_zoom: float = 1.048
    strong_zoom: float = 1.10
    maximum_boundary_zoom: float = 1.18
    maximum_boundary_scale_ratio: float = 1.18
    # A face/silhouette crop can already magnify the source.  Bound the product
    # of that crop and this output-space layer so the two individually-safe
    # transforms cannot turn into an excessive combined punch-in.
    maximum_combined_zoom: float = 1.35
    maximum_center_shift: float = 0.025
    maximum_center_distance_without_pan: float = 0.12
    minimum_silhouette_confidence: float = 0.72
    minimum_face_visible_fraction: float = 0.995
    minimum_silhouette_visible_fraction: float = 0.92
    maximum_transformed_layout_mismatch: float = 0.32
    target_zoom_events_per_minute: float = 8.3
    # Renderer/schema support roll, but the source-relative audit did not find
    # a repeatable editorial trigger.  Keep production planning off until an
    # evidence rule also passes post-transform boundary parity.
    target_roll_events_per_minute: float = 0.0
    minimum_zoom_event_spacing_seconds: float = 3.5
    minimum_roll_event_spacing_seconds: float = 20.0
    # A decorative pulse is allowed only after the same continuous source shot
    # has been visibly neutral for at least this long, and it must settle for
    # the same interval before either surrounding cross-source cut.  Keeping
    # the guard inside the audited ramp range prevents a pulse from reading as
    # a hidden cut-adjacent correction.
    interior_pulse_neutral_guard_seconds: float = 0.55
    minimum_roll_evidence_degrees: float = 0.8
    minimum_roll_degrees: float = 1.0
    maximum_roll_degrees: float = 1.6

    def __post_init__(self) -> None:
        numeric = tuple(
            float(getattr(self, name))
            for name in (
                "ramp_seconds",
                "ordinary_zoom",
                "close_up_zoom",
                "strong_zoom",
                "maximum_boundary_zoom",
                "maximum_boundary_scale_ratio",
                "maximum_combined_zoom",
                "maximum_center_shift",
                "maximum_center_distance_without_pan",
                "minimum_silhouette_confidence",
                "minimum_face_visible_fraction",
                "minimum_silhouette_visible_fraction",
                "maximum_transformed_layout_mismatch",
                "target_zoom_events_per_minute",
                "target_roll_events_per_minute",
                "minimum_zoom_event_spacing_seconds",
                "minimum_roll_event_spacing_seconds",
                "interior_pulse_neutral_guard_seconds",
                "minimum_roll_evidence_degrees",
                "minimum_roll_degrees",
                "maximum_roll_degrees",
            )
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("editorial motion configuration must be finite")
        if not 0.55 <= self.ramp_seconds <= 0.85:
            raise ValueError("editorial ramp must stay inside the reference window")
        if not 1.0 < self.close_up_zoom <= self.ordinary_zoom <= self.strong_zoom:
            raise ValueError("editorial zoom levels must be ordered")
        if not self.strong_zoom <= self.maximum_boundary_zoom <= 1.30:
            raise ValueError("boundary zoom must be bounded")
        if not 1.0 < self.maximum_boundary_scale_ratio <= self.maximum_boundary_zoom:
            raise ValueError("boundary scale ratio must fit the zoom bound")
        if not self.maximum_boundary_zoom <= self.maximum_combined_zoom <= 1.50:
            raise ValueError("combined editorial zoom must be conservatively bounded")
        if not 0.0 <= self.maximum_center_shift <= 0.05:
            raise ValueError("editorial center shift is too large")
        if not 0.0 <= self.maximum_center_distance_without_pan <= 0.20:
            raise ValueError("static center-distance gate is invalid")
        if not 0.0 <= self.minimum_silhouette_confidence <= 1.0:
            raise ValueError("silhouette confidence must be normalized")
        for value in (
            self.minimum_face_visible_fraction,
            self.minimum_silhouette_visible_fraction,
        ):
            if not 0.8 <= value <= 1.0:
                raise ValueError("subject visible fraction must be conservative")
        if not 0.0 < self.maximum_transformed_layout_mismatch < 1.0:
            raise ValueError("transformed layout mismatch must be normalized")
        if not 0.0 <= self.minimum_roll_degrees <= self.maximum_roll_degrees <= 1.6:
            raise ValueError("roll range must stay inside the reliable reference cap")
        if self.target_zoom_events_per_minute < 0.0:
            raise ValueError("target zoom density cannot be negative")
        if self.target_roll_events_per_minute < 0.0:
            raise ValueError("target roll density cannot be negative")
        if self.target_zoom_events_per_minute > 30.0:
            raise ValueError("target zoom density is too high")
        if self.target_roll_events_per_minute > 6.0:
            raise ValueError("target roll density is too high")
        if self.minimum_zoom_event_spacing_seconds < 0.0:
            raise ValueError("zoom event spacing cannot be negative")
        if self.minimum_roll_event_spacing_seconds < 0.0:
            raise ValueError("roll event spacing cannot be negative")
        if not 0.55 <= self.interior_pulse_neutral_guard_seconds <= 0.85:
            raise ValueError(
                "interior pulse neutral guard must stay inside the reference window"
            )
        if not 0.0 <= self.minimum_roll_evidence_degrees <= 15.0:
            raise ValueError("roll evidence threshold is invalid")


@dataclass(frozen=True, slots=True)
class EditorialMotionSegment:
    segment_index: int
    source_index: int
    track: EditorialTransformTrack | None
    reasons: tuple[str, ...] = ()

    @property
    def applied(self) -> bool:
        return self.track is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "source_index": self.source_index,
            "applied": self.applied,
            "reasons": list(self.reasons),
            "keyframes": (
                [
                    {
                        "time_seconds": item.time_seconds,
                        "zoom": item.zoom,
                        "center_x": item.center_x,
                        "center_y": item.center_y,
                        "roll_degrees": item.roll_degrees,
                    }
                    for item in self.track.keyframes
                ]
                if self.track is not None
                else []
            ),
        }


@dataclass(frozen=True, slots=True)
class EditorialMotionPlan:
    segments: tuple[EditorialMotionSegment, ...]
    boundary_match_count: int
    zoom_event_count: int
    pan_event_count: int
    roll_event_count: int
    blocked_dissolve_segment_count: int
    affine_boundary_match_count: int = 0
    interior_zoom_pulse_count: int = 0
    skipped_boundary_reasons: tuple[tuple[str, int], ...] = ()
    section_directed: bool = False
    analysis_basis: str = (
        "source_relative_reference_calibrated_boundary_and_neutral_guarded_"
        "tier_d_plus_pixel_affine_virtual_camera_v3"
    )
    boundary_subject_alignment_plan: BoundarySubjectAlignmentPlan | None = None
    accepted_boundary_indices: tuple[int, ...] = ()
    natural_match_boundary_indices: tuple[int, ...] = ()
    boundary_residual_attention_mismatches: tuple[
        tuple[int, float, float], ...
    ] = ()
    member_roll_audit: tuple = ()
    # Authored gestures and frame-visible trajectory targets have different
    # cardinalities (e.g. one pulse split across two render segments).
    authored_event_counts: tuple[tuple[str, int], ...] = ()
    # A partial proposal must not discard the exact reasons a required camera
    # was withheld. This is diagnostic evidence, never acceptance authority.
    required_boundary_failures: tuple[tuple[int, tuple[str, ...], str | None], ...] = ()
    member_containment_trial_audit: tuple = ()
    # Complete-shot containment has its own clock/geometry binding. It must
    # never inherit the older boundary-local planner's analysis header.
    shot_context_cameras: tuple = ()

    def __post_init__(self):
        failures = tuple((index, tuple(reasons), context) for index, reasons, context in self.required_boundary_failures)
        if (len({row[0] for row in failures}) != len(failures)
                or any(type(index) is not int or not 0 <= index < len(self.segments) - 1
                       or not reasons or any(not isinstance(reason, str) or not reason for reason in reasons)
                       or context is not None and not isinstance(context, str)
                       for index, reasons, context in failures)):
            raise ValueError('required camera failures must name valid boundaries and diagnostic reasons')
        object.__setattr__(self, 'required_boundary_failures', failures)
        if self.shot_context_cameras:
            from .visual_boundary_repair import _freeze_audit_value
            object.__setattr__(self, 'shot_context_cameras',
                tuple(_freeze_audit_value(row) for row in self.shot_context_cameras))
        if self.member_containment_trial_audit:
            from .visual_boundary_repair import _freeze_audit_value
            object.__setattr__(self, 'member_containment_trial_audit',
                tuple(_freeze_audit_value(row) for row in self.member_containment_trial_audit))
        if self.member_roll_audit:
            from .visual_boundary_repair import _freeze_audit_value
            object.__setattr__(self,'member_roll_audit',tuple(_freeze_audit_value(row) for row in self.member_roll_audit))

    @property
    def applied_count(self) -> int:
        return sum(item.applied for item in self.segments)

    def track_for(self, segment_index: int) -> EditorialTransformTrack | None:
        matches = [
            item.track for item in self.segments if item.segment_index == segment_index
        ]
        if len(matches) != 1:
            raise ValueError("editorial plan must contain each segment exactly once")
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        from .visual_boundary_repair import _thaw_audit_value
        confirmed_member_policy = self.analysis_basis.startswith(
            "confirmed_member_anchor_"
        )
        direct_member_policy = self.analysis_basis.startswith(
            "official_target_fancam_"
        )
        member_policy = confirmed_member_policy or direct_member_policy
        return {
            "analysis_basis": self.analysis_basis,
            "policy": {
                "calibration": (
                    "official_fancam_dense_dominant_face_output_geometry"
                    if direct_member_policy
                    else "confirmed_member_post_crop_boundary_geometry"
                    if confirmed_member_policy
                    else (
                        "exact_cut_frame_face_eye_roll_or_group_silhouette_"
                        "boundary_local_alignment"
                        if self.boundary_subject_alignment_plan is not None
                        else "high_confidence_source_relative_stage_mix_affine_events"
                    )
                ),
                "ordinary_ramp": "0.55_to_0.85_seconds_monotonic_default_0.75",
                "ordinary_zoom": (
                    "boundary_scale_match_up_to_12_percent"
                    if member_policy
                    else "4.8_to_9.0_percent"
                ),
                "roll": (
                    "observed_focal_screen_pose_with_dense_rotated_containment"
                    if direct_member_policy and self.member_roll_audit
                    else
                    "disabled_without_dense_direct_member_head_pose_evidence"
                    if direct_member_policy
                    else "disabled_without_confirmed_member_head_pose_evidence"
                    if confirmed_member_policy
                    else (
                        "exact_face_eye_head_roll_when_renderer_and_dense_frame_safe"
                        if self.boundary_subject_alignment_plan is not None
                        else (
                            "section_budget_plus_confirmed_head_roll_evidence"
                            if self.section_directed
                            else "disabled_by_default_pending_repeatable_editorial_trigger"
                        )
                    )
                ),
                "same_shot_holds": (
                    "official_fancam_full_frame_curve_without_identity_inference"
                    if direct_member_policy
                    else "confirmed_crop_curve_without_identity_reselection"
                    if confirmed_member_policy
                    else (
                        "boundary_adjacent_choice_only_without_tier_d_inheritance"
                        if self.boundary_subject_alignment_plan is not None
                        else (
                            "one_run_global_curve_sliced_without_reset_with_"
                            "neutral_guarded_interior_pulses"
                        )
                    )
                ),
                "dissolve": "disabled_until_transform_aware_overlap_review",
                "subject_safety": (
                    "dense_dominant_face_every_transformed_output_frame_99.5_percent_containment"
                    if direct_member_policy
                    else "simplified_pan_crop_plus_every_output_frame_99.5_percent_containment"
                    if confirmed_member_policy
                    else (
                        "exact_cut_plus_every_transformed_output_frame_"
                        "post_base_crop_containment"
                        if self.boundary_subject_alignment_plan is not None
                        else
                        "post_crop_boundary_plus_all_available_choice_samples_"
                        "visible_fraction_review"
                    )
                ),
                "audio_and_duration": "untouched",
            },
            "applied_segment_count": self.applied_count,
            "boundary_match_count": self.boundary_match_count,
            "affine_boundary_match_count": self.affine_boundary_match_count,
            "zoom_event_count": self.zoom_event_count,
            "pan_event_count": self.pan_event_count,
            "roll_event_count": self.roll_event_count,
            "authored_event_counts": dict(self.authored_event_counts),
            "event_count_contract": (
                "frame_visible_active_targets_v1"
                if self.authored_event_counts else "legacy_authored_gestures_v1"
            ),
            "interior_zoom_pulse_count": self.interior_zoom_pulse_count,
            "section_directed": self.section_directed,
            "blocked_dissolve_segment_count": self.blocked_dissolve_segment_count,
            "skipped_boundary_reasons": [
                {"reason": reason, "count": count}
                for reason, count in self.skipped_boundary_reasons
            ],
            "accepted_boundary_indices": list(self.accepted_boundary_indices),
            "natural_match_boundary_indices": list(
                self.natural_match_boundary_indices
            ),
            "boundary_residual_attention_mismatches": [
                {
                    "boundary_index": boundary_index,
                    "position": position,
                    "scale": scale,
                }
                for boundary_index, position, scale
                in self.boundary_residual_attention_mismatches
            ],
            "boundary_subject_alignment": (
                self.boundary_subject_alignment_plan.to_dict()
                if self.boundary_subject_alignment_plan is not None
                else None
            ),
            "segments": [item.to_dict() for item in self.segments],
            **({"required_boundary_failures": [
                {"boundary_index": index, "reasons": list(reasons), "context": context}
                for index, reasons, context in self.required_boundary_failures
            ]} if self.required_boundary_failures else {}),
            **({'member_roll_audit':_thaw_audit_value(self.member_roll_audit)} if self.member_roll_audit else {}),
            **({'member_containment_trial_audit': _thaw_audit_value(self.member_containment_trial_audit)}
               if self.member_containment_trial_audit else {}),
            **({'shot_context_cameras': _thaw_audit_value(self.shot_context_cameras)}
               if self.shot_context_cameras else {}),
        }


@dataclass(frozen=True, slots=True)
class _State:
    zoom: float = 1.0
    center_x: float = 0.5
    center_y: float = 0.5
    roll_degrees: float = 0.0


@dataclass(frozen=True, slots=True)
class _Run:
    first: int
    last: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def _neutral(state: _State, *, tolerance: float = 1e-8) -> bool:
    return (
        abs(state.zoom - 1.0) <= tolerance
        and abs(state.center_x - 0.5) <= tolerance
        and abs(state.center_y - 0.5) <= tolerance
        and abs(state.roll_degrees) <= tolerance
    )


def _interpolate(left: _State, right: _State, alpha: float) -> _State:
    return _State(
        zoom=left.zoom + (right.zoom - left.zoom) * alpha,
        center_x=left.center_x + (right.center_x - left.center_x) * alpha,
        center_y=left.center_y + (right.center_y - left.center_y) * alpha,
        roll_degrees=(
            left.roll_degrees
            + (right.roll_degrees - left.roll_degrees) * alpha
        ),
    )


def _curve_value(points: Sequence[tuple[float, _State]], time_value: float) -> _State:
    if time_value <= points[0][0] + 1e-9:
        return points[0][1]
    for left, right in zip(points, points[1:]):
        if time_value <= right[0] + 1e-9:
            span = right[0] - left[0]
            if span <= 1e-9:
                return right[1]
            return _interpolate(left[1], right[1], (time_value - left[0]) / span)
    return points[-1][1]


def _dedupe_curve(points: Sequence[tuple[float, _State]]) -> list[tuple[float, _State]]:
    output: list[tuple[float, _State]] = []
    for time_value, state in sorted(points, key=lambda item: item[0]):
        if output and abs(output[-1][0] - time_value) <= 1e-7:
            output[-1] = (time_value, state)
        else:
            output.append((time_value, state))
    return output


def _quantize_curve_to_output_frames(
    points: Sequence[tuple[float, _State]],
    *,
    origin: float,
    fps: float,
) -> list[tuple[float, _State]]:
    """Put every authored slope change on the rendered output-frame clock.

    A Tier-D split can land on either side of a fractional ramp keypoint.  If
    that happens, the single cross-boundary frame step integrates two slopes
    and is neither the outgoing nor incoming velocity, which correctly looks
    like a reset to the renderer.  Quantizing the *global* curve before it is
    sliced makes each slope change coincide with a real output sample while
    preserving the exact first/last frame states of the run.
    """

    ordered = _dedupe_curve(points)
    if len(ordered) <= 1:
        return ordered
    start_time, start_state = ordered[0]
    end_time, end_state = ordered[-1]
    quantized: list[tuple[float, _State]] = [(start_time, start_state)]
    for time_value, state in ordered[1:-1]:
        frame_index = int(round((time_value - origin) * fps))
        snapped = origin + frame_index / fps
        snapped = min(end_time, max(start_time, snapped))
        quantized.append((snapped, state))
    quantized.append((end_time, end_state))
    return _dedupe_curve(quantized)


def _visible_fraction(
    box: tuple[float, float, float, float],
    state: _State,
) -> float:
    half = 0.5 / state.zoom
    crop_left = state.center_x - half
    crop_top = state.center_y - half
    crop_right = state.center_x + half
    crop_bottom = state.center_y + half
    box_left, box_top, box_width, box_height = box
    box_right = box_left + box_width
    box_bottom = box_top + box_height
    intersection_width = max(
        0.0, min(crop_right, box_right) - max(crop_left, box_left)
    )
    intersection_height = max(
        0.0, min(crop_bottom, box_bottom) - max(crop_top, box_top)
    )
    return intersection_width * intersection_height / max(
        1e-9, box_width * box_height
    )


def _project_box(
    box: tuple[float, float, float, float],
    state: _State,
) -> tuple[float, float, float, float]:
    return (
        (box[0] - state.center_x) * state.zoom + 0.5,
        (box[1] - state.center_y) * state.zoom + 0.5,
        box[2] * state.zoom,
        box[3] * state.zoom,
    )


def _face_box(
    choice: VisualTimelineChoice,
    sample_index: int,
) -> tuple[float, float, float, float] | None:
    box = choice.face_samples_xywh[sample_index]
    if box is None:
        return None
    if choice.face_detector_backends[sample_index] == "none":
        return None
    # The upstream failure field also covers unstable scale/position, tiny
    # detections and an internal camera cut.  None is the only state that
    # proves this face sample is safe enough to drive a virtual camera.
    if choice.face_match_candidate_failure is not None:
        return None
    return box


def _silhouette_box(
    choice: VisualTimelineChoice,
    sample_index: int,
    settings: EditorialMotionConfig,
) -> tuple[float, float, float, float] | None:
    box = choice.silhouette_samples_xywh[sample_index]
    if box is None:
        return None
    if choice.silhouette_backends[sample_index] == "none":
        return None
    if (
        choice.silhouette_confidences[sample_index]
        < settings.minimum_silhouette_confidence
    ):
        return None
    return box


def _paired_subject(
    left: VisualTimelineChoice,
    right: VisualTimelineChoice,
    settings: EditorialMotionConfig,
) -> tuple[
    str,
    tuple[float, float, float, float],
    tuple[float, float, float, float],
] | None:
    left_face = _face_box(left, 2)
    right_face = _face_box(right, 0)
    if left_face is not None and right_face is not None:
        return ("face", left_face, right_face)
    left_silhouette = _silhouette_box(left, 2, settings)
    right_silhouette = _silhouette_box(right, 0, settings)
    if left_silhouette is not None and right_silhouette is not None:
        return ("silhouette", left_silhouette, right_silhouette)
    return None


def _subject_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)


def _boundary_state(
    *,
    box: tuple[float, float, float, float],
    zoom: float,
    target: tuple[float, float],
    subject: str,
    settings: EditorialMotionConfig,
) -> _State | None:
    subject_x, subject_y = _subject_center(box)
    desired_x = subject_x - (target[0] - 0.5) / zoom
    desired_y = subject_y - (target[1] - 0.5) / zoom
    half = 0.5 / zoom
    low = half
    high = 1.0 - half
    center_x = min(
        high,
        max(
            low,
            min(0.5 + settings.maximum_center_shift, max(0.5 - settings.maximum_center_shift, desired_x)),
        ),
    )
    center_y = min(
        high,
        max(
            low,
            min(0.5 + settings.maximum_center_shift, max(0.5 - settings.maximum_center_shift, desired_y)),
        ),
    )
    state = _State(zoom=zoom, center_x=center_x, center_y=center_y)
    minimum_visible = (
        settings.minimum_face_visible_fraction
        if subject == "face"
        else settings.minimum_silhouette_visible_fraction
    )
    if _visible_fraction(box, state) + 1e-9 < minimum_visible:
        centered = replace(state, center_x=0.5, center_y=0.5)
        if _visible_fraction(box, centered) + 1e-9 < minimum_visible:
            return None
        state = centered
    try:
        EditorialTransformKeyframe(
            0.0,
            state.zoom,
            state.center_x,
            state.center_y,
            state.roll_degrees,
        )
    except ValueError:
        return None
    return state


def _runs(
    choices: Sequence[VisualTimelineChoice],
    transitions: Sequence[RenderTransition],
    *,
    first_frame_times: Sequence[float],
    last_frame_times: Sequence[float],
) -> tuple[_Run, ...]:
    output: list[_Run] = []
    first = 0
    for index, transition in enumerate(transitions):
        left = choices[index]
        right = choices[index + 1]
        continuous = (
            transition.tier == "D"
            and left.source_index == right.source_index
            and (
                (left.shot_id and left.shot_id == right.shot_id)
                or (not left.shot_id and not right.shot_id)
            )
        )
        if continuous:
            continue
        output.append(
            _Run(
                first,
                index,
                first_frame_times[first],
                last_frame_times[index],
            )
        )
        first = index + 1
    output.append(
        _Run(
            first,
            len(choices) - 1,
            first_frame_times[first],
            last_frame_times[-1],
        )
    )
    return tuple(output)


def _output_frame_schedule(
    choices: Sequence[VisualTimelineChoice],
    fps: float,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
    """Mirror RenderPlan cumulative rounding on the authored output clock."""

    counts: list[int] = []
    starts: list[int] = []
    cumulative_seconds = 0.0
    previous_boundary = 0
    for choice in choices:
        starts.append(previous_boundary)
        cumulative_seconds += choice.duration_seconds
        boundary = int(round(cumulative_seconds * fps))
        count = boundary - previous_boundary
        if count < 2:
            raise ValueError("editorial motion segment is shorter than two frames")
        counts.append(count)
        previous_boundary = boundary
    origin = choices[0].canonical_master_start
    first_times = tuple(origin + start / fps for start in starts)
    last_times = tuple(
        origin + (start + count - 1) / fps
        for start, count in zip(starts, counts, strict=True)
    )
    return (tuple(counts), first_times, last_times)


def _near_phrase(time_value: float, phrase_times: Sequence[float]) -> float:
    if not phrase_times:
        return 0.0
    distance = min(abs(time_value - item) for item in phrase_times)
    return max(0.0, 1.0 - distance / 0.30)


def _face_matches(
    face_match_plan: FaceMatchPlan | None,
) -> dict[int, FaceMatchSegment]:
    if face_match_plan is None:
        return {}
    matches = {item.segment_index: item for item in face_match_plan.segments}
    if len(matches) != len(face_match_plan.segments):
        raise ValueError("face-match plan contains duplicate segment indices")
    return matches


def _fixed_crop_at(
    track: PanCropTrack,
    time_seconds: float,
) -> tuple[float, float, float, float]:
    """Evaluate the same fixed-size linear crop path used by the renderer."""

    width, height = track.fixed_crop_size
    positions = track.fixed_crop_positions()
    if time_seconds <= positions[0][0] + 1e-9:
        return (positions[0][1], positions[0][2], width, height)
    for left, right in zip(positions, positions[1:]):
        if time_seconds <= right[0] + 1e-9:
            span = right[0] - left[0]
            if span <= 1e-9:
                return (right[1], right[2], width, height)
            alpha = (time_seconds - left[0]) / span
            return (
                left[1] + (right[1] - left[1]) * alpha,
                left[2] + (right[2] - left[2]) * alpha,
                width,
                height,
            )
    return (positions[-1][1], positions[-1][2], width, height)


def _project_box_through_crop(
    box: tuple[float, float, float, float],
    crop: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    crop_x, crop_y, crop_width, crop_height = crop
    return (
        (box[0] - crop_x) / crop_width,
        (box[1] - crop_y) / crop_height,
        box[2] / crop_width,
        box[3] / crop_height,
    )


def _post_base_crop_box(
    box: tuple[float, float, float, float],
    match: FaceMatchSegment | None,
    *,
    time_seconds: float,
) -> tuple[tuple[float, float, float, float], float]:
    """Return subject geometry in editorial-layer coordinates and base zoom."""

    if match is None or not match.applied or match.pan_crop is None:
        return (box, 1.0)
    crop = _fixed_crop_at(match.pan_crop, time_seconds)
    return (_project_box_through_crop(box, crop), 1.0 / crop[3])


def _sample_centroid(
    choice: VisualTimelineChoice,
    sample_index: int,
) -> tuple[float, float] | None:
    value = choice.silhouette_centroid_samples_xy[sample_index]
    if value is not None:
        return value
    box = choice.silhouette_samples_xywh[sample_index]
    return _subject_center(box) if box is not None else None


def _run_zoom_value(
    run: _Run,
    choices: Sequence[VisualTimelineChoice],
    settings: EditorialMotionConfig,
) -> float:
    shot_types = {
        choices[index].shot_type for index in range(run.first, run.last + 1)
    }
    if shot_types == {ShotType.CLOSE_UP}:
        return settings.close_up_zoom
    if ShotType.GROUP_WIDE in shot_types:
        return settings.ordinary_zoom
    return min(settings.ordinary_zoom, 1.058)


def _is_real_tier_d_continuous_run(
    run: _Run,
    choices: Sequence[VisualTimelineChoice],
    transitions: Sequence[RenderTransition],
) -> bool:
    """Require at least one proven no-cut split inside one contiguous source shot."""

    if run.last <= run.first:
        return False
    source_index = choices[run.first].source_index
    for index in range(run.first, run.last + 1):
        if choices[index].source_index != source_index:
            return False
    for index in range(run.first, run.last):
        left = choices[index]
        right = choices[index + 1]
        if transitions[index].tier != "D":
            return False
        if abs(left.source_local_end - right.source_local_start) > 1e-4:
            return False
        if not (
            (left.shot_id and left.shot_id == right.shot_id)
            or (not left.shot_id and not right.shot_id)
        ):
            return False
    return True


def _run_has_native_cut(
    run: _Run,
    choices: Sequence[VisualTimelineChoice],
) -> bool:
    return any(
        choice.internal_scene_cut_master_seconds
        or choice.entry_scene_cut_near
        or choice.exit_scene_cut_near
        or choice.face_match_candidate_failure == "internal_camera_cut"
        for choice in choices[run.first : run.last + 1]
    )


def _choice_sample_output_times(
    choice: VisualTimelineChoice,
    *,
    output_start: float,
    output_end: float,
) -> tuple[float, float, float]:
    """Mirror the entry/middle/exit sampling positions from visual analysis."""

    inset = min(0.05, choice.duration_seconds * 0.12)
    intended = (
        output_start + inset,
        output_start + choice.duration_seconds / 2.0,
        output_start + choice.duration_seconds - inset,
    )
    def clamp(time_value: float) -> float:
        return min(output_end, max(output_start, time_value))

    return (clamp(intended[0]), clamp(intended[1]), clamp(intended[2]))


def _interior_pulse_subject_safety_failure(
    *,
    run: _Run,
    choices: Sequence[VisualTimelineChoice],
    first_frame_times: Sequence[float],
    last_frame_times: Sequence[float],
    curve: Sequence[tuple[float, _State]],
    settings: EditorialMotionConfig,
) -> str | None:
    """Review every available three-point face/silhouette sample on the curve."""

    verified_subject_samples = 0
    for choice_index in range(run.first, run.last + 1):
        choice = choices[choice_index]
        sample_times = _choice_sample_output_times(
            choice,
            output_start=first_frame_times[choice_index],
            output_end=last_frame_times[choice_index],
        )
        for sample_index, time_value in enumerate(sample_times):
            state = _curve_value(curve, time_value)
            subjects: list[
                tuple[str, tuple[float, float, float, float], float]
            ] = []
            face = _face_box(choice, sample_index)
            if face is not None:
                subjects.append(
                    ("face", face, settings.minimum_face_visible_fraction)
                )
            silhouette = _silhouette_box(choice, sample_index, settings)
            if silhouette is not None:
                subjects.append(
                    (
                        "silhouette",
                        silhouette,
                        settings.minimum_silhouette_visible_fraction,
                    )
                )
            if not subjects:
                # A missing descriptor at a neutral sample cannot be clipped by
                # this layer.  Missing evidence while the pulse is active is a
                # fail-closed rejection rather than an inferred safe crop.
                if not _neutral(state):
                    return "interior_pulse_subject_evidence_missing"
                continue
            for _subject, box, minimum_visible in subjects:
                verified_subject_samples += 1
                if (
                    _visible_fraction(box, state) + 1e-9
                    < minimum_visible
                ):
                    return "interior_pulse_subject_would_be_clipped"
    if verified_subject_samples == 0:
        return "interior_pulse_subject_evidence_missing"
    return None


def _interior_pulse_timing(
    *,
    run: _Run,
    phrase_times: Sequence[float],
    settings: EditorialMotionConfig,
    origin: float,
    fps: float,
) -> tuple[float, float] | None:
    # Round the ramp *up* to complete output frames.  That preserves the 0.55 s
    # audited minimum after quantization instead of silently shaving a partial
    # frame off either side of the pulse.
    ramp_frames = int(math.ceil(settings.ramp_seconds * fps - 1e-9))
    ramp = ramp_frames / fps
    if not 0.55 <= ramp <= 0.85:
        return None
    guard = settings.interior_pulse_neutral_guard_seconds
    earliest = run.start + guard + ramp
    latest = run.end - guard - ramp
    earliest_frame = int(math.ceil((earliest - origin) * fps - 1e-9))
    latest_frame = int(math.floor((latest - origin) * fps + 1e-9))
    if latest_frame < earliest_frame:
        return None
    midpoint_frame = int(round((earliest_frame + latest_frame) / 2.0))
    candidate_frames = {earliest_frame, latest_frame, midpoint_frame}
    candidate_frames.update(
        min(
            latest_frame,
            max(
                earliest_frame,
                int(round((float(time_value) - origin) * fps)),
            ),
        )
        for time_value in phrase_times
        if earliest - 1e-9 <= time_value <= latest + 1e-9
    )
    center_frame = max(
        candidate_frames,
        key=lambda frame_index: (
            _near_phrase(origin + frame_index / fps, phrase_times),
            -abs(frame_index - midpoint_frame),
            -frame_index,
        ),
    )
    return (origin + center_frame / fps, ramp)


def build_editorial_motion_plan(
    *,
    visual_timeline: VisualTimelinePlan,
    transitions: Sequence[RenderTransition],
    face_match_plan: FaceMatchPlan | None = None,
    boundary_affine_plan: BoundaryAffinePlan | None = None,
    boundary_subject_alignment_plan: BoundarySubjectAlignmentPlan | None = None,
    phrase_times: Sequence[float] = (),
    section_plan: SectionPlan | None = None,
    fps: float = 29.97,
    config: EditorialMotionConfig | None = None,
) -> EditorialMotionPlan:
    """Plan scale-matched cuts plus sparse reference-calibrated camera ramps."""

    choices = visual_timeline.choices
    if not choices:
        raise ValueError("visual timeline cannot be empty")
    if len(transitions) != len(choices) - 1:
        raise ValueError("editorial planner needs one transition per boundary")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("editorial planner fps must be positive and finite")
    settings = config or EditorialMotionConfig()
    affine_decisions = (
        {
            item.boundary_index: item
            for item in boundary_affine_plan.decisions
        }
        if boundary_affine_plan is not None
        else {}
    )
    if boundary_affine_plan is not None and len(affine_decisions) != len(
        boundary_affine_plan.decisions
    ):
        raise ValueError("boundary affine plan contains duplicate decisions")
    subject_alignment_decisions = (
        {
            item.boundary_index: item
            for item in boundary_subject_alignment_plan.decisions
        }
        if boundary_subject_alignment_plan is not None
        else {}
    )
    if boundary_subject_alignment_plan is not None:
        if len(subject_alignment_decisions) != len(
            boundary_subject_alignment_plan.decisions
        ):
            raise ValueError("boundary subject alignment plan has duplicate decisions")
        alignment_segments = {
            item.segment_index: item
            for item in boundary_subject_alignment_plan.segments
        }
        if len(alignment_segments) != len(boundary_subject_alignment_plan.segments):
            raise ValueError("boundary subject alignment plan has duplicate segments")
        if set(alignment_segments) != {item.segment_index for item in choices}:
            raise ValueError("boundary subject alignment plan does not cover the timeline")
        if any(
            alignment_segments[item.segment_index].source_index != item.source_index
            for item in choices
        ):
            raise ValueError("boundary subject alignment sources disagree with timeline")
    else:
        alignment_segments = {}
    if section_plan is not None:
        if (
            section_plan.start_seconds
            > choices[0].canonical_master_start + 1e-6
            or section_plan.end_seconds
            < choices[-1].canonical_master_end - 1e-6
        ):
            raise ValueError("section plan does not cover the visual timeline")

    def motion_budget(time_value: float) -> float:
        if section_plan is None:
            return 1.0
        return section_plan.directive_at(time_value).camera_motion_budget
    frame_counts, first_frame_times, last_frame_times = _output_frame_schedule(
        choices,
        fps,
    )
    run_items = _runs(
        choices,
        transitions,
        first_frame_times=first_frame_times,
        last_frame_times=last_frame_times,
    )
    run_for_choice: dict[int, _Run] = {
        index: run
        for run in run_items
        for index in range(run.first, run.last + 1)
    }
    blocked: set[int] = set()
    for index, transition in enumerate(transitions):
        if transition.kind.uses_overlap:
            blocked.update((index, index + 1))

    entry_states: list[_State | None] = [None] * len(choices)
    exit_states: list[_State | None] = [None] * len(choices)
    segment_reasons: list[set[str]] = [set() for _ in choices]
    face_matches = _face_matches(face_match_plan)
    skipped: dict[str, int] = {}
    boundary_match_count = (
        boundary_subject_alignment_plan.eligible_count
        if boundary_subject_alignment_plan is not None
        else 0
    )
    pan_event_count = (
        boundary_subject_alignment_plan.pan_event_count
        if boundary_subject_alignment_plan is not None
        else 0
    )
    event_times: list[float] = [
        first_frame_times[item.boundary_index + 1]
        for item in subject_alignment_decisions.values()
        if item.eligible
    ]
    boundary_candidates: list[
        tuple[float, float, int, _State, _State, str, bool, bool]
    ] = []

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for boundary_index, (left, right) in enumerate(zip(choices, choices[1:])):
        transition = transitions[boundary_index]
        if left.source_index == right.source_index or transition.tier == "D":
            continue
        # The v1 local layer has already decoded the exact cut endpoints and
        # reviewed every transformed frame.  Whether it accepted or rejected
        # this boundary is final: never fall back to the older sampled/ORB path
        # and silently bypass its fail-closed decision.
        if boundary_index in subject_alignment_decisions:
            decision = subject_alignment_decisions[boundary_index]
            if decision.eligible:
                reason = (
                    f"boundary_local_exact_{decision.subject_kind}_alignment"
                )
                segment_reasons[boundary_index].add(reason)
                segment_reasons[boundary_index + 1].add(reason)
            else:
                skip(
                    "boundary_subject_alignment_"
                    + str(decision.rejection_reason)
                )
            continue
        if boundary_index in blocked or boundary_index + 1 in blocked:
            skip("dissolve_adjacent")
            continue
        if (
            left.internal_scene_cut_master_seconds
            or right.internal_scene_cut_master_seconds
            or left.face_match_candidate_failure == "internal_camera_cut"
            or right.face_match_candidate_failure == "internal_camera_cut"
        ):
            skip("internal_camera_cut")
            continue
        affine_decision = affine_decisions.get(boundary_index)
        if affine_decision is not None and affine_decision.eligible:
            affine_transform = affine_decision.transform
            assert affine_transform is not None
            if run_for_choice[boundary_index + 1].duration < 0.55 - 1e-9:
                skip("affine_incoming_window_shorter_than_reference_ramp")
            else:
                affine_state = _State(
                    zoom=affine_transform.zoom,
                    center_x=affine_transform.center_x,
                    center_y=affine_transform.center_y,
                    roll_degrees=affine_transform.roll_degrees,
                )
                try:
                    # Validate the complete settle, not merely the boundary
                    # keyframe.  Rotation overscan has a slightly curved bound.
                    EditorialTransformTrack(
                        (
                            EditorialTransformKeyframe(
                                0.0,
                                affine_state.zoom,
                                affine_state.center_x,
                                affine_state.center_y,
                                affine_state.roll_degrees,
                            ),
                            EditorialTransformKeyframe(
                                settings.ramp_seconds,
                                1.0,
                                0.5,
                                0.5,
                                0.0,
                            ),
                        ),
                        reason="orb_affine_subject_matched_hard_cut",
                    )
                except ValueError:
                    skip("affine_settle_renderer_safety_rejection")
                else:
                    diagnostics = affine_decision.result.diagnostics
                    subject_review = diagnostics.get("subject_review", {})
                    improvement = float(
                        subject_review.get("geometry_improvement", 0.0)
                    )
                    inlier_ratio = float(
                        diagnostics.get("affine_inlier_ratio", 0.0)
                    )
                    phrase_score = _near_phrase(
                        first_frame_times[boundary_index + 1], phrase_times
                    )
                    evidence_quality = min(left.quality_score, right.quality_score)
                    score = (
                        1.5
                        + 6.0 * improvement
                        + 0.8 * inlier_ratio
                        + 0.30 * evidence_quality
                        + 0.60 * phrase_score
                        + 0.55
                        * motion_budget(first_frame_times[boundary_index + 1])
                    )
                    uses_pan = math.hypot(
                        affine_state.center_x - 0.5,
                        affine_state.center_y - 0.5,
                    ) > 0.003
                    boundary_candidates.append(
                        (
                            score,
                            first_frame_times[boundary_index + 1],
                            boundary_index,
                            _State(),
                            affine_state,
                            "orb_affine_subject_matched_hard_cut",
                            uses_pan,
                            True,
                        )
                    )
                    # Pixel correspondence is stronger evidence than the
                    # descriptor-box fallback below; never create two rival
                    # transforms for the same boundary.
                    continue
        exact_pair = (
            affine_decision.exact_subject_pair
            if affine_decision is not None
            else None
        )
        if exact_pair is not None:
            # These boxes were measured after the same center-crop used by the
            # full-frame renderer, on the exact outgoing/incoming cut frames.
            # They remain useful geometry evidence even when background ORB
            # correspondences cannot survive a change of broadcast camera.
            subject = exact_pair.kind
            left_box = exact_pair.outgoing_box_xywh
            right_box = exact_pair.incoming_box_xywh
            left_base_zoom = 1.0
            right_base_zoom = 1.0
        else:
            pair = _paired_subject(left, right, settings)
            if pair is None:
                skip("no_common_boundary_subject")
                continue
            subject, raw_left_box, raw_right_box = pair
            left_match = face_matches.get(left.segment_index)
            right_match = face_matches.get(right.segment_index)
            left_box, left_base_zoom = _post_base_crop_box(
                raw_left_box,
                left_match,
                time_seconds=min(
                    left.duration_seconds,
                    (frame_counts[boundary_index] - 1) / fps,
                ),
            )
            right_box, right_base_zoom = _post_base_crop_box(
                raw_right_box,
                right_match,
                time_seconds=0.0,
            )
        left_height = left_box[3]
        right_height = right_box[3]
        if min(left_height, right_height) <= 1e-6:
            skip("invalid_subject_height")
            continue
        ratio = max(left_height, right_height) / min(left_height, right_height)
        if ratio > settings.maximum_boundary_scale_ratio + 1e-9:
            skip("scale_ratio_exceeds_transform_bound")
            continue
        left_center = _subject_center(left_box)
        right_center = _subject_center(right_box)
        center_distance = math.hypot(
            left_center[0] - right_center[0],
            left_center[1] - right_center[1],
        )
        baseline_mismatch = screen_layout_mismatch(
            (),
            (),
            left_bbox=left_box,
            right_bbox=right_box,
        )
        if baseline_mismatch is None:
            skip("baseline_layout_mismatch_unavailable")
            continue
        if ratio <= 1.015 and center_distance <= 0.025:
            skip("already_scale_position_matched")
            continue
        # Change only the smaller side. This is the least-effort solution and
        # mirrors the common authored pattern: outgoing push-in when the
        # outgoing subject is smaller, incoming release when the incoming one
        # is smaller. Near-equal scales use a minimal incoming reframe only
        # when screen position still needs correction.
        if left_height + 1e-9 < right_height:
            if run_for_choice[boundary_index].duration < 0.55 - 1e-9:
                skip("outgoing_window_shorter_than_reference_ramp")
                continue
            left_zoom = max(ratio, 1.02)
            if (
                left_base_zoom * left_zoom
                > settings.maximum_combined_zoom + 1e-9
            ):
                skip("combined_zoom_exceeds_transform_bound")
                continue
            left_state = _boundary_state(
                box=left_box,
                zoom=left_zoom,
                target=right_center,
                subject=subject,
                settings=settings,
            )
            right_state = _State()
        elif right_height + 1e-9 < left_height:
            if run_for_choice[boundary_index + 1].duration < 0.55 - 1e-9:
                skip("incoming_window_shorter_than_reference_ramp")
                continue
            right_zoom = max(ratio, 1.02)
            if (
                right_base_zoom * right_zoom
                > settings.maximum_combined_zoom + 1e-9
            ):
                skip("combined_zoom_exceeds_transform_bound")
                continue
            left_state = _State()
            right_state = _boundary_state(
                box=right_box,
                zoom=right_zoom,
                target=left_center,
                subject=subject,
                settings=settings,
            )
        else:
            left_state = _State()
            if run_for_choice[boundary_index + 1].duration < 0.55 - 1e-9:
                skip("incoming_window_shorter_than_reference_ramp")
                continue
            if (
                right_base_zoom * 1.04
                > settings.maximum_combined_zoom + 1e-9
            ):
                skip("combined_zoom_exceeds_transform_bound")
                continue
            right_state = _boundary_state(
                box=right_box,
                zoom=1.04,
                target=left_center,
                subject=subject,
                settings=settings,
            )
        if left_state is None or right_state is None:
            skip("subject_would_be_clipped")
            continue
        minimum_visible = (
            settings.minimum_face_visible_fraction
            if subject == "face"
            else settings.minimum_silhouette_visible_fraction
        )
        centered_left = replace(left_state, center_x=0.5, center_y=0.5)
        centered_right = replace(right_state, center_x=0.5, center_y=0.5)
        if (
            center_distance
            <= settings.maximum_center_distance_without_pan + 1e-9
            and
            _visible_fraction(left_box, centered_left) + 1e-9 >= minimum_visible
            and _visible_fraction(right_box, centered_right) + 1e-9
            >= minimum_visible
        ):
            centered_mismatch = screen_layout_mismatch(
                (),
                (),
                left_bbox=_project_box(left_box, centered_left),
                right_bbox=_project_box(right_box, centered_right),
            )
            if (
                centered_mismatch is not None
                and centered_mismatch
                <= settings.maximum_transformed_layout_mismatch + 1e-9
                and centered_mismatch < baseline_mismatch - 1e-5
            ):
                # Scale alone already passes the same reviewer threshold. Do
                # not manufacture a pan merely to chase a perfect centroid.
                left_state = centered_left
                right_state = centered_right
        transformed_left = _project_box(left_box, left_state)
        transformed_right = _project_box(right_box, right_state)
        transformed_mismatch = screen_layout_mismatch(
            (),
            (),
            left_bbox=transformed_left,
            right_bbox=transformed_right,
        )
        if (
            transformed_mismatch is None
            or transformed_mismatch
            > settings.maximum_transformed_layout_mismatch + 1e-9
        ):
            skip("post_transform_layout_mismatch")
            continue
        if transformed_mismatch >= baseline_mismatch - 1e-5:
            skip("post_transform_no_layout_improvement")
            continue
        reason = (
            f"exact_frame_{subject}_scale_position_matched_hard_cut"
            if exact_pair is not None
            else f"{subject}_scale_position_matched_hard_cut"
        )
        uses_pan = (
            abs(left_state.center_x - 0.5)
            + abs(left_state.center_y - 0.5)
            + abs(right_state.center_x - 0.5)
            + abs(right_state.center_y - 0.5)
            > 0.012
        )
        event_time = first_frame_times[boundary_index + 1]
        improvement = baseline_mismatch - transformed_mismatch
        phrase_score = _near_phrase(event_time, phrase_times)
        evidence_quality = min(left.quality_score, right.quality_score)
        score = (
            5.0 * improvement
            + abs(math.log(ratio))
            + 0.30 * evidence_quality
            + 0.60 * phrase_score
            + 0.55 * motion_budget(event_time)
            + (0.25 if exact_pair is not None else 0.0)
        )
        boundary_candidates.append(
            (
                score,
                event_time,
                boundary_index,
                left_state,
                right_state,
                reason,
                uses_pan,
                False,
            )
        )

    total_duration = choices[-1].canonical_master_end - choices[0].canonical_master_start
    if section_plan is None:
        target_zoom_events = max(
            0,
            int(
                round(
                    total_duration
                    * settings.target_zoom_events_per_minute
                    / 60.0
                )
            ),
        )
    else:
        target_zoom_events = max(
            0,
            int(
                round(
                    sum(
                        section.duration_seconds
                        * (4.0 + 8.0 * section.camera_motion_budget)
                        / 60.0
                        for section in section_plan.sections
                    )
                )
            ),
        )
    boundary_limit = max(1, target_zoom_events) if boundary_candidates else 0
    selected_affine_boundaries: set[int] = set()
    for (
        _score,
        event_time,
        boundary_index,
        left_state,
        right_state,
        reason,
        uses_pan,
        is_affine,
    ) in sorted(boundary_candidates, reverse=True):
        if boundary_match_count >= boundary_limit:
            break
        if event_times and min(
            abs(event_time - item) for item in event_times
        ) < settings.minimum_zoom_event_spacing_seconds:
            continue
        exit_states[boundary_index] = left_state
        entry_states[boundary_index + 1] = right_state
        segment_reasons[boundary_index].add(reason)
        segment_reasons[boundary_index + 1].add(reason)
        boundary_match_count += 1
        pan_event_count += int(uses_pan)
        if is_affine:
            selected_affine_boundaries.add(boundary_index)
        event_times.append(event_time)
    local_boundary_matches = (boundary_subject_alignment_plan.eligible_count
                              if boundary_subject_alignment_plan is not None else 0)
    rejected_boundary_candidates = len(boundary_candidates) - (boundary_match_count - local_boundary_matches)
    if rejected_boundary_candidates:
        skipped["boundary_density_or_spacing_cap"] = rejected_boundary_candidates
    zoom_effects: dict[int, str] = {}
    boundary_zoom_events = boundary_match_count
    needed_zoom_events = max(0, target_zoom_events - boundary_zoom_events)
    zoom_candidates: list[tuple[float, int, str]] = []
    for run_index, run in enumerate(run_items):
        if any(
            alignment_segments.get(choices[index].segment_index) is not None
            and alignment_segments[choices[index].segment_index].track is not None
            for index in range(run.first, run.last + 1)
        ):
            continue
        if run.duration + 1e-9 < settings.ramp_seconds:
            continue
        if any(index in blocked for index in range(run.first, run.last + 1)):
            continue
        if any(
            (
                match := face_matches.get(choices[index].segment_index)
            ) is not None
            and match.applied
            for index in range(run.first, run.last + 1)
        ):
            continue
        if entry_states[run.first] is not None or exit_states[run.last] is not None:
            continue
        run_choices = choices[run.first : run.last + 1]
        if any(item.internal_scene_cut_master_seconds for item in run_choices):
            continue
        entry_scale = run_choices[0].entry_shot_scale
        exit_scale = run_choices[-1].exit_shot_scale
        if entry_scale is None or exit_scale is None:
            continue
        native_delta = float(exit_scale - entry_scale)
        direction = "push_in" if native_delta >= -0.015 else "pull_out"
        # Decorative quota motion must never invent a transform at an
        # unreviewed cross-source cut.  Boundary-local matches above own those
        # endpoints; a generic native-motion continuation is only eligible at
        # the beginning or end of the complete performance.
        if direction == "push_in" and run.last != len(choices) - 1:
            continue
        if direction == "pull_out" and run.first != 0:
            continue
        evidence_choice = (
            run_choices[-1] if direction == "push_in" else run_choices[0]
        )
        sample_index = 2 if direction == "push_in" else 0
        subject_kind = "face"
        subject_box = _face_box(evidence_choice, sample_index)
        if subject_box is None:
            subject_kind = "silhouette"
            subject_box = _silhouette_box(evidence_choice, sample_index, settings)
        if subject_box is None:
            continue
        candidate_state = _State(
            zoom=_run_zoom_value(run, choices, settings),
        )
        minimum_visible = (
            settings.minimum_face_visible_fraction
            if subject_kind == "face"
            else settings.minimum_silhouette_visible_fraction
        )
        if _visible_fraction(subject_box, candidate_state) + 1e-9 < minimum_visible:
            continue
        quality = sum(item.quality_score for item in run_choices) / len(run_choices)
        phrase_score = max(_near_phrase(run.start, phrase_times), _near_phrase(run.end, phrase_times))
        budget = motion_budget((run.start + run.end) / 2.0)
        score = (
            1.4 * phrase_score
            + 1.2 * min(0.2, abs(native_delta))
            + quality
            + 0.55 * budget
        )
        zoom_candidates.append((score, run_index, direction))
    for _score, run_index, direction in sorted(zoom_candidates, reverse=True):
        if len(zoom_effects) >= needed_zoom_events:
            break
        run = run_items[run_index]
        event_time = run.end if direction == "push_in" else run.start
        if event_times and min(abs(event_time - item) for item in event_times) < settings.minimum_zoom_event_spacing_seconds:
            continue
        zoom_effects[run_index] = direction
        event_times.append(event_time)

    # Remaining decorative quota may be filled only inside a proven Tier-D
    # same-source continuous shot.  Both run endpoints stay neutral, leaving
    # uncertain cross-source cuts exactly as selected by the boundary planner.
    interior_pulse_candidates: list[
        tuple[float, float, float, int, float, float]
    ] = []
    curve_origin = choices[0].canonical_master_start
    for run_index, run in enumerate(run_items):
        if any(
            alignment_segments.get(choices[index].segment_index) is not None
            and alignment_segments[choices[index].segment_index].track is not None
            for index in range(run.first, run.last + 1)
        ):
            skip("interior_pulse_boundary_subject_alignment_owned_run")
            continue
        if section_plan is not None and motion_budget(
            (run.start + run.end) / 2.0
        ) < 0.45:
            skip("interior_pulse_section_motion_budget_low")
            continue
        if not _is_real_tier_d_continuous_run(run, choices, transitions):
            skip("interior_pulse_not_real_tier_d_continuous_run")
            continue
        if any(index in blocked for index in range(run.first, run.last + 1)):
            skip("interior_pulse_dissolve_adjacent")
            continue
        if run_index in zoom_effects:
            skip("interior_pulse_existing_decorative_effect")
            continue
        if any(
            (
                match := face_matches.get(choices[index].segment_index)
            ) is not None
            and match.applied
            for index in range(run.first, run.last + 1)
        ):
            skip("interior_pulse_applied_face_crop")
            continue
        if any(
            entry_states[index] is not None or exit_states[index] is not None
            for index in range(run.first, run.last + 1)
        ):
            skip("interior_pulse_existing_boundary_state")
            continue
        if _run_has_native_cut(run, choices):
            skip("interior_pulse_internal_or_native_cut")
            continue
        timing = _interior_pulse_timing(
            run=run,
            phrase_times=phrase_times,
            settings=settings,
            origin=curve_origin,
            fps=fps,
        )
        if timing is None:
            skip("interior_pulse_run_too_short_for_ramps_and_guards")
            continue
        center, pulse_ramp = timing
        zoom_value = _run_zoom_value(run, choices, settings)
        pulse_curve = _quantize_curve_to_output_frames(
            (
                (run.start, _State()),
                (center - pulse_ramp, _State()),
                (center, _State(zoom=zoom_value)),
                (center + pulse_ramp, _State()),
                (run.end, _State()),
            ),
            origin=curve_origin,
            fps=fps,
        )
        safety_failure = _interior_pulse_subject_safety_failure(
            run=run,
            choices=choices,
            first_frame_times=first_frame_times,
            last_frame_times=last_frame_times,
            curve=pulse_curve,
            settings=settings,
        )
        if safety_failure is not None:
            skip(safety_failure)
            continue
        run_choices = choices[run.first : run.last + 1]
        quality = sum(item.quality_score for item in run_choices) / len(run_choices)
        phrase_score = _near_phrase(center, phrase_times)
        score = (
            1.4 * phrase_score
            + quality
            + 0.55 * motion_budget(center)
        )
        interior_pulse_candidates.append(
            (score, phrase_score, quality, run_index, center, pulse_ramp)
        )

    interior_pulse_effects: dict[int, tuple[float, float]] = {}
    for (
        _score,
        _phrase_score,
        _quality,
        run_index,
        center,
        pulse_ramp,
    ) in sorted(interior_pulse_candidates, reverse=True):
        if len(zoom_effects) + len(interior_pulse_effects) >= needed_zoom_events:
            skip("interior_pulse_density_cap")
            continue
        if event_times and min(
            abs(center - item) for item in event_times
        ) < settings.minimum_zoom_event_spacing_seconds:
            skip("interior_pulse_spacing_cap")
            continue
        interior_pulse_effects[run_index] = (center, pulse_ramp)
        event_times.append(center)

    if section_plan is None:
        target_roll_events = max(
            0,
            int(
                round(
                    total_duration
                    * settings.target_roll_events_per_minute
                    / 60.0
                )
            ),
        )
    else:
        target_roll_events = max(
            0,
            int(
                round(
                    sum(
                        section.duration_seconds
                        * section.target_roll_events_per_minute
                        / 60.0
                        for section in section_plan.sections
                    )
                )
            ),
        )
    roll_candidates: list[tuple[float, int, float]] = []
    for run_index, run in enumerate(run_items):
        if any(
            alignment_segments.get(choices[index].segment_index) is not None
            and alignment_segments[choices[index].segment_index].track is not None
            for index in range(run.first, run.last + 1)
        ):
            continue
        if run_index in interior_pulse_effects:
            # An interior pulse is deliberately zoom-only; do not combine an
            # independently scored roll settle with the same curve.
            continue
        if run.start <= choices[0].canonical_master_start + 1e-9:
            continue
        if section_plan is not None and section_plan.directive_at(
            run.start
        ).target_roll_events_per_minute <= 0.0:
            continue
        if run.duration + 1e-9 < settings.ramp_seconds:
            continue
        if any(index in blocked for index in range(run.first, run.last + 1)):
            continue
        if any(
            (
                match := face_matches.get(choices[index].segment_index)
            ) is not None
            and match.applied
            for index in range(run.first, run.last + 1)
        ):
            continue
        if (
            entry_states[run.first] is not None
            or exit_states[run.last] is not None
        ):
            continue
        run_choices = choices[run.first : run.last + 1]
        if any(
            item.internal_scene_cut_master_seconds
            or item.face_match_candidate_failure == "internal_camera_cut"
            for item in run_choices
        ):
            continue
        roll_radians = choices[run.first].face_head_roll_samples_radians[0]
        if roll_radians is None:
            continue
        evidence = math.degrees(roll_radians)
        if abs(evidence) < settings.minimum_roll_evidence_degrees:
            continue
        amplitude = math.copysign(
            min(
                settings.maximum_roll_degrees,
                max(settings.minimum_roll_degrees, abs(evidence) * 0.55),
            ),
            evidence,
        )
        score = abs(evidence) + 1.5 * _near_phrase(run.start, phrase_times)
        roll_candidates.append((score, run_index, amplitude))
    roll_effects: dict[int, float] = {}
    roll_times: list[float] = []
    for _score, run_index, amplitude in sorted(roll_candidates, reverse=True):
        if len(roll_effects) >= target_roll_events:
            break
        time_value = run_items[run_index].start
        if roll_times and min(abs(time_value - item) for item in roll_times) < settings.minimum_roll_event_spacing_seconds:
            continue
        roll_effects[run_index] = amplitude
        roll_times.append(time_value)

    result: list[EditorialMotionSegment | None] = [None] * len(choices)
    for run_index, run in enumerate(run_items):
        start_state = entry_states[run.first] or _State()
        end_state = exit_states[run.last] or _State()
        points: list[tuple[float, _State]] = [(run.start, start_state)]
        ramp = min(settings.ramp_seconds, run.duration)
        starts_transformed = not _neutral(start_state)
        ends_transformed = not _neutral(end_state)
        if starts_transformed and ends_transformed:
            if run.duration >= 2.0 * ramp - 1e-6:
                points.extend(
                    (
                        (run.start + ramp, _State()),
                        (run.end - ramp, _State()),
                    )
                )
            # When two verified boundary states share a shorter shot, a direct
            # whole-shot interpolation preserves both cuts.  Inserting two
            # neutral settles would cross and collapse into sub-100ms punches.
        elif starts_transformed and run.duration > ramp + 1e-6:
            points.append((run.start + ramp, _State()))
        elif ends_transformed and run.duration > ramp + 1e-6:
            points.append((run.end - ramp, _State()))
        points.append((run.end, end_state))

        direction = zoom_effects.get(run_index)
        if direction is not None:
            zoom_value = _run_zoom_value(run, choices, settings)
            motion_state = _State(
                zoom=zoom_value,
            )
            try:
                EditorialTransformKeyframe(
                    0.0,
                    motion_state.zoom,
                    motion_state.center_x,
                    motion_state.center_y,
                )
            except ValueError:
                motion_state = replace(motion_state, center_x=0.5, center_y=0.5)
            if direction == "push_in":
                points.extend(
                    (
                        (max(run.start, run.end - ramp), _State()),
                        (run.end, motion_state),
                    )
                )
            else:
                points.extend(
                    (
                        (run.start, motion_state),
                        (min(run.end, run.start + ramp), _State()),
                    )
                )
            for index in range(run.first, run.last + 1):
                segment_reasons[index].add(f"reference_calibrated_{direction}")

        interior_pulse = interior_pulse_effects.get(run_index)
        if interior_pulse is not None:
            center, pulse_ramp = interior_pulse
            points.extend(
                (
                    (center - pulse_ramp, _State()),
                    (
                        center,
                        _State(zoom=_run_zoom_value(run, choices, settings)),
                    ),
                    (center + pulse_ramp, _State()),
                )
            )

        roll_amplitude = roll_effects.get(run_index)
        if roll_amplitude is not None:
            # A linear settle from the maximum roll to zero crosses a slightly
            # convex overscan bound.  Reserve the worst-case 1.6-degree margin
            # so every interpolated output frame remains above it, not just
            # the endpoints.
            roll_zoom = minimum_roll_safe_zoom(roll_amplitude) + 0.018
            points.extend(
                (
                    (
                        run.start,
                        _State(
                            zoom=roll_zoom,
                            roll_degrees=roll_amplitude,
                        ),
                    ),
                    (min(run.end, run.start + ramp), _State()),
                )
            )
            for index in range(run.first, run.last + 1):
                segment_reasons[index].add("reference_calibrated_roll_settle")

        curve = _quantize_curve_to_output_frames(
            points,
            origin=curve_origin,
            fps=fps,
        )
        for choice_index in range(run.first, run.last + 1):
            choice = choices[choice_index]
            start = first_frame_times[choice_index]
            end = last_frame_times[choice_index]
            local_last_frame_time = min(
                choice.duration_seconds,
                (frame_counts[choice_index] - 1) / fps,
            )
            local_points: list[tuple[float, _State]] = [
                (0.0, _curve_value(curve, start))
            ]
            for time_value, state in curve:
                if start + 1e-7 < time_value < end - 1e-7:
                    local_time = time_value - start
                    if local_time < local_last_frame_time - 1e-7:
                        local_points.append((local_time, state))
            local_points.append(
                (local_last_frame_time, _curve_value(curve, end))
            )
            local_points = _dedupe_curve(local_points)
            alignment_segment = alignment_segments.get(choice.segment_index)
            if alignment_segment is not None and alignment_segment.track is not None:
                segment_reasons[choice_index].add(
                    "exact_frame_boundary_local_subject_alignment"
                )
                result[choice_index] = EditorialMotionSegment(
                    choice.segment_index,
                    choice.source_index,
                    alignment_segment.track,
                    tuple(sorted(segment_reasons[choice_index])),
                )
                continue
            if interior_pulse is not None and any(
                not _neutral(state) for _time, state in local_points
            ):
                segment_reasons[choice_index].add(
                    "reference_calibrated_same_source_interior_zoom_pulse"
                )
            if all(_neutral(state) for _time, state in local_points):
                result[choice_index] = EditorialMotionSegment(
                    choice.segment_index,
                    choice.source_index,
                    None,
                    tuple(sorted(segment_reasons[choice_index])),
                )
                continue
            try:
                track = EditorialTransformTrack(
                    tuple(
                        EditorialTransformKeyframe(
                            time_value,
                            state.zoom,
                            state.center_x,
                            state.center_y,
                            state.roll_degrees,
                        )
                        for time_value, state in local_points
                    ),
                    reason="+".join(sorted(segment_reasons[choice_index]))
                    or "reference_calibrated_camera_motion",
                )
            except ValueError:
                # Keep the scale intent but remove optional pan/roll if their
                # interpolation would approach a transformed frame edge.
                safe_points = [
                    (
                        time_value,
                        replace(
                            state,
                            center_x=0.5,
                            center_y=0.5,
                            roll_degrees=0.0,
                        ),
                    )
                    for time_value, state in local_points
                ]
                track = EditorialTransformTrack(
                    tuple(
                        EditorialTransformKeyframe(
                            time_value,
                            state.zoom,
                            state.center_x,
                            state.center_y,
                            state.roll_degrees,
                        )
                        for time_value, state in safe_points
                    ),
                    reason="scale_only_safe_fallback",
                )
                segment_reasons[choice_index].add("optional_pan_roll_removed")
            result[choice_index] = EditorialMotionSegment(
                choice.segment_index,
                choice.source_index,
                track,
                tuple(sorted(segment_reasons[choice_index])),
            )

    if any(item is None for item in result):
        raise ValueError("editorial planner did not cover every segment")
    actual_decorative_roll_event_count = sum(
        any(
            result[index] is not None
            and result[index].track is not None
            and any(
                abs(keyframe.roll_degrees) > 1e-9
                for keyframe in result[index].track.keyframes
            )
            for index in range(run_items[run_index].first, run_items[run_index].last + 1)
        )
        for run_index in roll_effects
    )
    actual_affine_roll_event_count = sum(
        result[boundary_index + 1] is not None
        and result[boundary_index + 1].track is not None
        and any(
            abs(keyframe.roll_degrees) > 1e-9
            for keyframe in result[boundary_index + 1].track.keyframes
        )
        for boundary_index in selected_affine_boundaries
    )
    final_segments = tuple(item for item in result if item is not None)
    render_segments = tuple(
        RenderSegment(
            source_index=choice.source_index,
            source_start=choice.source_local_start,
            master_start=choice.canonical_master_start - choices[0].canonical_master_start,
            duration=choice.duration_seconds,
            editorial_transform=motion.track,
        )
        for choice, motion in zip(choices, final_segments, strict=True)
    )
    events = rendered_transform_event_records(render_segments, fps)
    return EditorialMotionPlan(
        segments=final_segments,
        boundary_match_count=boundary_match_count,
        zoom_event_count=len(events["zoom"]),
        pan_event_count=len(events["pan"]),
        roll_event_count=len(events["roll"]),
        authored_event_counts=(
            ("zoom", boundary_zoom_events + len(zoom_effects) + len(interior_pulse_effects)),
            ("pan", pan_event_count),
            ("roll", actual_decorative_roll_event_count + actual_affine_roll_event_count
             + (boundary_subject_alignment_plan.roll_event_count
                if boundary_subject_alignment_plan is not None else 0)),
        ),
        blocked_dissolve_segment_count=len(blocked),
        affine_boundary_match_count=len(selected_affine_boundaries),
        interior_zoom_pulse_count=len(interior_pulse_effects),
        skipped_boundary_reasons=tuple(sorted(skipped.items())),
        section_directed=section_plan is not None,
        analysis_basis=(
            "source_relative_reference_calibrated_boundary_and_neutral_guarded_"
            "tier_d_plus_exact_frame_local_subject_alignment_virtual_camera_v4"
            if boundary_subject_alignment_plan is not None
            else (
                "source_relative_reference_calibrated_boundary_and_neutral_guarded_"
                "tier_d_plus_pixel_affine_virtual_camera_v3"
            )
        ),
        boundary_subject_alignment_plan=boundary_subject_alignment_plan,
    )


__all__ = [
    "EditorialMotionConfig",
    "EditorialMotionPlan",
    "EditorialMotionSegment",
    "build_editorial_motion_plan",
]
