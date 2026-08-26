"""Safe editorial-camera planning for confirmed target-member crops.

The member crop remains the authority for identity and containment.  This
module only adds a short output-space scale/position match at a real source
cut, and only after replaying the simplified renderer crop plus the proposed
camera track on every quantized output frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .editorial_motion import EditorialMotionPlan, EditorialMotionSegment
from .member_draft import (
    MemberDraftSegment,
    MemberSegmentKind,
    SegmentSubjectSample,
)
from .render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    PanCropTrack,
    RenderSegment,
    RenderTransition,
    TransitionKind,
)


@dataclass(frozen=True, slots=True)
class MemberEditorialMotionConfig:
    """Conservative member-mode subset of the shared editorial camera."""

    ramp_seconds: float = 0.75
    maximum_boundary_zoom: float = 1.12
    maximum_center_shift: float = 0.025
    maximum_boundary_center_distance: float = 0.18
    maximum_residual_center_distance: float = 0.08
    minimum_mismatch_improvement_fraction: float = 0.20
    minimum_subject_confidence: float = 0.42
    minimum_subject_visible_fraction: float = 0.995
    maximum_subject_sample_gap_seconds: float = 0.20
    target_events_per_minute: float = 8.3
    minimum_event_spacing_seconds: float = 3.5

    def __post_init__(self) -> None:
        values = (
            self.ramp_seconds,
            self.maximum_boundary_zoom,
            self.maximum_center_shift,
            self.maximum_boundary_center_distance,
            self.maximum_residual_center_distance,
            self.minimum_mismatch_improvement_fraction,
            self.minimum_subject_confidence,
            self.minimum_subject_visible_fraction,
            self.maximum_subject_sample_gap_seconds,
            self.target_events_per_minute,
            self.minimum_event_spacing_seconds,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("member editorial configuration must be finite")
        if not 0.55 <= self.ramp_seconds <= 0.85:
            raise ValueError("member editorial ramp is outside the reference window")
        if not 1.02 <= self.maximum_boundary_zoom <= 1.50:
            raise ValueError("member editorial zoom is outside the renderer bound")
        if not 0.0 <= self.maximum_center_shift <= 0.12:
            raise ValueError("member editorial pan exceeds the shared-camera bound")
        if not 0.0 <= self.maximum_boundary_center_distance <= 0.30:
            raise ValueError("member boundary center-distance gate is invalid")
        if not 0.0 <= self.maximum_residual_center_distance <= 0.15:
            raise ValueError("member residual center-distance gate is invalid")
        if not 0.0 <= self.minimum_mismatch_improvement_fraction <= 0.80:
            raise ValueError("member mismatch improvement gate is invalid")
        if not 0.0 <= self.minimum_subject_confidence <= 1.0:
            raise ValueError("member subject confidence must be normalized")
        if not 0.95 <= self.minimum_subject_visible_fraction <= 1.0:
            raise ValueError("member subject containment must be conservative")
        if not 0.0 < self.maximum_subject_sample_gap_seconds <= 0.50:
            raise ValueError("member subject sample gap is invalid")
        if not 0.0 <= self.target_events_per_minute <= 30.0:
            raise ValueError("member editorial event density is invalid")
        if self.minimum_event_spacing_seconds < 0.0:
            raise ValueError("member editorial event spacing cannot be negative")


@dataclass(frozen=True, slots=True)
class _State:
    zoom: float = 1.0
    center_x: float = 0.5
    center_y: float = 0.5


@dataclass(frozen=True, slots=True)
class _BoundaryCandidate:
    boundary_index: int
    event_time: float
    score: float
    side: str
    state: _State
    uses_pan: bool
    alternative_states: tuple[_State, ...] = ()


def _frame_schedule(
    segments: Sequence[RenderSegment], fps: float
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
    counts: list[int] = []
    last_times: list[float] = []
    boundary_times: list[float] = []
    cumulative = 0.0
    previous_boundary = 0
    for segment in segments:
        cumulative += segment.duration
        boundary = int(round(cumulative * fps))
        count = boundary - previous_boundary
        if count < 2:
            raise ValueError("member editorial segment is shorter than two frames")
        counts.append(count)
        last_times.append((count - 1) / fps)
        boundary_times.append(boundary / fps)
        previous_boundary = boundary
    return (tuple(counts), tuple(last_times), tuple(boundary_times))


def _interpolate_subject(
    samples: Sequence[SegmentSubjectSample],
    time_seconds: float,
    settings: MemberEditorialMotionConfig,
) -> tuple[float, float, float, float] | None:
    if not samples:
        return None
    if time_seconds <= samples[0].time_seconds + 1e-9:
        if (
            samples[0].time_seconds - time_seconds
            > settings.maximum_subject_sample_gap_seconds + 1e-9
            or samples[0].confidence + 1e-9
            < settings.minimum_subject_confidence
        ):
            return None
        return samples[0].bbox_xyxy
    for left, right in zip(samples, samples[1:]):
        if time_seconds <= right.time_seconds + 1e-9:
            span = right.time_seconds - left.time_seconds
            if (
                span > settings.maximum_subject_sample_gap_seconds + 1e-9
                or min(left.confidence, right.confidence) + 1e-9
                < settings.minimum_subject_confidence
            ):
                return None
            alpha = (time_seconds - left.time_seconds) / max(1e-9, span)
            return tuple(
                left_value + (right_value - left_value) * alpha
                for left_value, right_value in zip(
                    left.bbox_xyxy, right.bbox_xyxy, strict=True
                )
            )
    last = samples[-1]
    if (
        time_seconds - last.time_seconds
        > settings.maximum_subject_sample_gap_seconds + 1e-9
        or last.confidence + 1e-9 < settings.minimum_subject_confidence
    ):
        return None
    return last.bbox_xyxy


def _crop_at(
    track: PanCropTrack,
    time_seconds: float,
) -> tuple[float, float, float, float]:
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


def _post_crop_box(
    bbox_xyxy: tuple[float, float, float, float],
    crop: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    left, top, right, bottom = bbox_xyxy
    crop_x, crop_y, crop_width, crop_height = crop
    return (
        (left - crop_x) / crop_width,
        (top - crop_y) / crop_height,
        (right - left) / crop_width,
        (bottom - top) / crop_height,
    )


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)


def _project(
    box: tuple[float, float, float, float], state: _State
) -> tuple[float, float, float, float]:
    return (
        (box[0] - state.center_x) * state.zoom + 0.5,
        (box[1] - state.center_y) * state.zoom + 0.5,
        box[2] * state.zoom,
        box[3] * state.zoom,
    )


def _visible_fraction(
    box: tuple[float, float, float, float], state: _State
) -> float:
    half = 0.5 / state.zoom
    camera_left = state.center_x - half
    camera_top = state.center_y - half
    camera_right = state.center_x + half
    camera_bottom = state.center_y + half
    box_left, box_top, box_width, box_height = box
    box_right = box_left + box_width
    box_bottom = box_top + box_height
    intersection_width = max(
        0.0, min(camera_right, box_right) - max(camera_left, box_left)
    )
    intersection_height = max(
        0.0, min(camera_bottom, box_bottom) - max(camera_top, box_top)
    )
    return intersection_width * intersection_height / max(
        1e-9, box_width * box_height
    )


def _mismatch(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    if min(left[3], right[3]) <= 1e-9:
        return math.inf
    return abs(math.log(left[3] / right[3])) + math.hypot(
        _center(left)[0] - _center(right)[0],
        _center(left)[1] - _center(right)[1],
    )


def _state_for_match(
    *,
    box: tuple[float, float, float, float],
    target_box: tuple[float, float, float, float],
    zoom: float,
    settings: MemberEditorialMotionConfig,
) -> tuple[_State, ...]:
    subject_x, subject_y = _center(box)
    target_x, target_y = _center(target_box)
    half = 0.5 / zoom
    low = half
    high = 1.0 - half
    desired_x = subject_x - (target_x - 0.5) / zoom
    desired_y = subject_y - (target_y - 0.5) / zoom
    bounded_x = min(high, max(low, desired_x))
    bounded_y = min(high, max(low, desired_y))
    delta_x = bounded_x - 0.5
    delta_y = bounded_y - 0.5
    distance = math.hypot(delta_x, delta_y)
    if distance > settings.maximum_center_shift + 1e-12:
        scale = settings.maximum_center_shift / distance
        bounded_x = 0.5 + delta_x * scale
        bounded_y = 0.5 + delta_y * scale
    panned = _State(
        zoom=zoom,
        center_x=bounded_x,
        center_y=bounded_y,
    )
    centered = _State(zoom=zoom)
    candidates = (panned, centered) if panned != centered else (centered,)
    valid: list[_State] = []
    for state in candidates:
        try:
            EditorialTransformKeyframe(
                0.0,
                state.zoom,
                state.center_x,
                state.center_y,
                0.0,
            )
        except ValueError:
            continue
        valid.append(state)
    return tuple(valid)


def _to_track_state(keyframe: EditorialTransformKeyframe) -> _State:
    return _State(keyframe.zoom, keyframe.center_x, keyframe.center_y)


def _dedupe_points(
    points: Sequence[tuple[float, _State]],
) -> tuple[tuple[float, _State], ...]:
    output: list[tuple[float, _State]] = []
    for time_value, state in sorted(points, key=lambda item: item[0]):
        if output and abs(output[-1][0] - time_value) <= 1e-8:
            output[-1] = (time_value, state)
        else:
            output.append((time_value, state))
    return tuple(output)


def _track_for_states(
    *,
    entry: _State | None,
    exit: _State | None,
    last_frame_time: float,
    settings: MemberEditorialMotionConfig,
) -> EditorialTransformTrack | None:
    if entry is None and exit is None:
        return None
    neutral = _State()
    ramp = min(settings.ramp_seconds, last_frame_time)
    if entry is not None and exit is not None:
        if last_frame_time >= 2.0 * settings.ramp_seconds:
            points = (
                (0.0, entry),
                (settings.ramp_seconds, neutral),
                (last_frame_time - settings.ramp_seconds, neutral),
                (last_frame_time, exit),
            )
        else:
            points = ((0.0, entry), (last_frame_time, exit))
    elif entry is not None:
        points = ((0.0, entry), (ramp, neutral))
        if ramp < last_frame_time - 1e-8:
            points += ((last_frame_time, neutral),)
    else:
        assert exit is not None
        ramp_start = max(0.0, last_frame_time - ramp)
        points = ((0.0, neutral),)
        if ramp_start > 1e-8:
            points += ((ramp_start, neutral),)
        points += ((last_frame_time, exit),)
    keyframes = tuple(
        EditorialTransformKeyframe(
            time_seconds=time_value,
            zoom=state.zoom,
            center_x=state.center_x,
            center_y=state.center_y,
            roll_degrees=0.0,
        )
        for time_value, state in _dedupe_points(points)
    )
    return EditorialTransformTrack(
        keyframes,
        reason="confirmed_member_subject_scale_position_match",
    )


def _track_contains_subject_every_frame(
    *,
    member_segment: MemberDraftSegment,
    render_segment: RenderSegment,
    track: EditorialTransformTrack | None,
    frame_count: int,
    fps: float,
    settings: MemberEditorialMotionConfig,
) -> bool:
    if render_segment.pan_crop is None or not member_segment.subject_samples:
        return False
    effective_crop = render_segment.pan_crop.simplified()
    for frame_index in range(frame_count):
        time_seconds = frame_index / fps
        subject = _interpolate_subject(
            member_segment.subject_samples, time_seconds, settings
        )
        if subject is None:
            return False
        box = _post_crop_box(subject, _crop_at(effective_crop, time_seconds))
        if track is None:
            state = _State()
        else:
            state = _to_track_state(track.evaluate(time_seconds))
        if (
            _visible_fraction(box, state) + 1e-9
            < settings.minimum_subject_visible_fraction
        ):
            return False
    return True


def _has_internal_cut(
    segment: MemberDraftSegment,
    source_scene_cut_times: Sequence[Sequence[float]],
    *,
    fps: float,
) -> bool:
    if not source_scene_cut_times:
        return False
    if segment.source_index is None or segment.source_index >= len(source_scene_cut_times):
        raise ValueError("member scene-cut evidence does not cover the selected source")
    assert segment.source_local_start is not None
    assert segment.source_local_end is not None
    margin = 0.5 / fps
    return any(
        segment.source_local_start + margin
        < float(cut_time)
        < segment.source_local_end - margin
        for cut_time in source_scene_cut_times[segment.source_index]
    )


def build_member_editorial_motion_plan(
    *,
    member_segments: Sequence[MemberDraftSegment],
    render_segments: Sequence[RenderSegment],
    transitions: Sequence[RenderTransition],
    source_scene_cut_times: Sequence[Sequence[float]] = (),
    fps: float = 29.97,
    config: MemberEditorialMotionConfig | None = None,
) -> EditorialMotionPlan:
    """Match confirmed member size/position across cuts without losing them."""

    members = tuple(member_segments)
    renders = tuple(render_segments)
    transition_items = tuple(transitions)
    if not members or len(members) != len(renders):
        raise ValueError("member editorial inputs must contain matching segments")
    if len(transition_items) != len(members) - 1:
        raise ValueError("member editorial inputs need one transition per boundary")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("member editorial fps must be positive and finite")
    settings = config or MemberEditorialMotionConfig()
    frame_counts, last_frame_times, boundary_times = _frame_schedule(renders, fps)
    for index, (member, render) in enumerate(zip(members, renders, strict=True)):
        if member.segment_index != index:
            raise ValueError("member editorial indices must be contiguous")
        if abs(member.duration_seconds - render.duration) > 1e-6:
            raise ValueError("member and render durations must match")
        if render.editorial_transform is not None:
            raise ValueError("member editorial planner needs untransformed segments")

    blocked: set[int] = set()
    for index, transition in enumerate(transition_items):
        if transition.kind.uses_overlap:
            blocked.update((index, index + 1))

    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    safe_geometry: list[bool] = []
    internal_cuts: list[bool] = []
    boundary_boxes: list[
        tuple[
            tuple[float, float, float, float] | None,
            tuple[float, float, float, float] | None,
        ]
    ] = []
    for index, (member, render) in enumerate(zip(members, renders, strict=True)):
        safe = (
            member.kind is MemberSegmentKind.TARGET_CROP
            and bool(member.subject_samples)
            and render.pan_crop is not None
        )
        safe_geometry.append(safe)
        has_cut = _has_internal_cut(
            member, source_scene_cut_times, fps=fps
        ) if safe else False
        internal_cuts.append(has_cut)
        if not safe:
            boundary_boxes.append((None, None))
            continue
        assert render.pan_crop is not None
        effective_crop = render.pan_crop.simplified()
        first_subject = _interpolate_subject(
            member.subject_samples, 0.0, settings
        )
        last_subject = _interpolate_subject(
            member.subject_samples, last_frame_times[index], settings
        )
        boundary_boxes.append(
            (
                _post_crop_box(first_subject, _crop_at(effective_crop, 0.0))
                if first_subject is not None
                else None,
                _post_crop_box(
                    last_subject,
                    _crop_at(effective_crop, last_frame_times[index]),
                )
                if last_subject is not None
                else None,
            )
        )

    candidates: list[_BoundaryCandidate] = []
    for boundary_index, (left, right) in enumerate(zip(members, members[1:])):
        if left.source_index == right.source_index:
            continue
        if boundary_index in blocked or boundary_index + 1 in blocked:
            skip("dissolve_adjacent")
            continue
        if not safe_geometry[boundary_index] or not safe_geometry[boundary_index + 1]:
            skip("no_confirmed_subject_geometry")
            continue
        if internal_cuts[boundary_index] or internal_cuts[boundary_index + 1]:
            skip("internal_camera_cut")
            continue
        left_box = boundary_boxes[boundary_index][1]
        right_box = boundary_boxes[boundary_index + 1][0]
        if left_box is None or right_box is None:
            skip("boundary_subject_gap")
            continue
        center_distance = math.hypot(
            _center(left_box)[0] - _center(right_box)[0],
            _center(left_box)[1] - _center(right_box)[1],
        )
        if center_distance > settings.maximum_boundary_center_distance + 1e-9:
            skip("boundary_position_exceeds_transform_bound")
            continue
        ratio = max(left_box[3], right_box[3]) / max(
            1e-9, min(left_box[3], right_box[3])
        )
        if ratio > settings.maximum_boundary_zoom + 1e-9:
            skip("boundary_scale_exceeds_transform_bound")
            continue
        baseline = _mismatch(left_box, right_box)
        if ratio <= 1.015 and center_distance <= 0.025:
            skip("already_scale_position_matched")
            continue
        if left_box[3] + 1e-9 < right_box[3]:
            side = "left"
            segment_index = boundary_index
            box = left_box
            target_box = right_box
        else:
            side = "right"
            segment_index = boundary_index + 1
            box = right_box
            target_box = left_box
        if last_frame_times[segment_index] < 0.55 - 1e-9:
            skip("window_shorter_than_reference_ramp")
            continue
        zoom = min(settings.maximum_boundary_zoom, max(1.02, ratio))
        qualifying_states: list[tuple[float, _State]] = []
        for state in _state_for_match(
            box=box,
            target_box=target_box,
            zoom=zoom,
            settings=settings,
        ):
            if (
                _visible_fraction(box, state) + 1e-9
                < settings.minimum_subject_visible_fraction
            ):
                continue
            projected_left = _project(left_box, state) if side == "left" else left_box
            projected_right = _project(right_box, state) if side == "right" else right_box
            transformed = _mismatch(projected_left, projected_right)
            transformed_center_distance = math.hypot(
                _center(projected_left)[0] - _center(projected_right)[0],
                _center(projected_left)[1] - _center(projected_right)[1],
            )
            if (
                transformed_center_distance
                > settings.maximum_residual_center_distance + 1e-9
            ):
                continue
            if (
                transformed
                > baseline
                * (1.0 - settings.minimum_mismatch_improvement_fraction)
                + 1e-9
            ):
                continue
            if transformed + 1e-4 < baseline:
                qualifying_states.append((transformed, state))
        if not qualifying_states:
            skip("no_improving_contained_transform")
            continue
        qualifying_states.sort(key=lambda item: item[0])
        chosen_score, chosen = qualifying_states[0]
        event_time = boundary_times[boundary_index]
        candidates.append(
            _BoundaryCandidate(
                boundary_index=boundary_index,
                event_time=event_time,
                score=baseline - chosen_score,
                side=side,
                state=chosen,
                uses_pan=(
                    abs(chosen.center_x - 0.5) > 1e-6
                    or abs(chosen.center_y - 0.5) > 1e-6
                ),
                alternative_states=tuple(
                    state for _score, state in qualifying_states[1:]
                ),
            )
        )

    quantized_duration = sum(frame_counts) / fps
    maximum_events = int(
        math.floor(
            quantized_duration * settings.target_events_per_minute / 60.0
            + 1e-12
        )
    )
    entry_states: list[_State | None] = [None] * len(members)
    exit_states: list[_State | None] = [None] * len(members)
    accepted: list[_BoundaryCandidate] = []
    tracks: list[EditorialTransformTrack | None] = [None] * len(members)
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.event_time)):
        if len(accepted) >= maximum_events:
            skip("event_density_cap")
            continue
        if any(
            abs(candidate.event_time - item.event_time)
            < settings.minimum_event_spacing_seconds - 1e-9
            for item in accepted
        ):
            skip("event_spacing_cap")
            continue
        segment_index = (
            candidate.boundary_index
            if candidate.side == "left"
            else candidate.boundary_index + 1
        )
        previous_entry = entry_states[segment_index]
        previous_exit = exit_states[segment_index]
        accepted_state: _State | None = None
        proposed: EditorialTransformTrack | None = None
        for state in (candidate.state,) + candidate.alternative_states:
            if candidate.side == "left":
                exit_states[segment_index] = state
            else:
                entry_states[segment_index] = state
            try:
                proposed = _track_for_states(
                    entry=entry_states[segment_index],
                    exit=exit_states[segment_index],
                    last_frame_time=last_frame_times[segment_index],
                    settings=settings,
                )
            except ValueError:
                proposed = None
            if proposed is not None and _track_contains_subject_every_frame(
                member_segment=members[segment_index],
                render_segment=renders[segment_index],
                track=proposed,
                frame_count=frame_counts[segment_index],
                fps=fps,
                settings=settings,
            ):
                accepted_state = state
                break
        if accepted_state is None or proposed is None:
            entry_states[segment_index] = previous_entry
            exit_states[segment_index] = previous_exit
            skip("full_frame_subject_containment_failed")
            continue
        tracks[segment_index] = proposed
        accepted.append(
            _BoundaryCandidate(
                boundary_index=candidate.boundary_index,
                event_time=candidate.event_time,
                score=candidate.score,
                side=candidate.side,
                state=accepted_state,
                uses_pan=(
                    abs(accepted_state.center_x - 0.5) > 1e-6
                    or abs(accepted_state.center_y - 0.5) > 1e-6
                ),
            )
        )

    # Rebuild once so a segment with both an accepted entry and exit event is
    # represented by one continuous curve rather than two reset tracks.
    for index in range(len(tracks)):
        if entry_states[index] is None and exit_states[index] is None:
            continue
        track = _track_for_states(
            entry=entry_states[index],
            exit=exit_states[index],
            last_frame_time=last_frame_times[index],
            settings=settings,
        )
        if track is None or not _track_contains_subject_every_frame(
            member_segment=members[index],
            render_segment=renders[index],
            track=track,
            frame_count=frame_counts[index],
            fps=fps,
            settings=settings,
        ):
            raise ValueError("accepted member camera failed its final containment review")
        tracks[index] = track

    accepted_segments = {
        candidate.boundary_index
        if candidate.side == "left"
        else candidate.boundary_index + 1
        for candidate in accepted
    }
    plan_segments = tuple(
        EditorialMotionSegment(
            segment_index=index,
            source_index=(
                member.source_index
                if member.source_index is not None
                else render.source_index
            ),
            track=tracks[index],
            reasons=(
                ("confirmed_member_subject_scale_position_match",)
                if index in accepted_segments
                else (
                    "no_safe_confirmed_member_motion",
                )
            ),
        )
        for index, (member, render) in enumerate(zip(members, renders, strict=True))
    )
    return EditorialMotionPlan(
        segments=plan_segments,
        boundary_match_count=len(accepted),
        zoom_event_count=len(accepted),
        pan_event_count=sum(item.uses_pan for item in accepted),
        roll_event_count=0,
        blocked_dissolve_segment_count=len(blocked),
        skipped_boundary_reasons=tuple(sorted(skipped.items())),
        analysis_basis=(
            "confirmed_member_anchor_post_crop_every_output_frame_containment_v1"
        ),
    )


__all__ = [
    "MemberEditorialMotionConfig",
    "build_member_editorial_motion_plan",
]
