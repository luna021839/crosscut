"""Run-local GROUP motion overlay that preserves decoded boundary authority."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Sequence

from .editorial_motion import (
    EditorialMotionConfig,
    EditorialMotionPlan,
    EditorialMotionSegment,
)
from .editorial_transform_events import rendered_transform_event_records
from .render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    RenderSegment,
    RenderTransition,
)


_PULSE_REASON = "reference_calibrated_same_source_interior_zoom_pulse"
_TIER_D_HOLD_REASON = "joint_decoded_same_source_same_shot_hold"
_EPSILON = 1.0e-9
_TIME_EPSILON = 1.0e-7
_PULSE_SETTINGS = EditorialMotionConfig()
_ALLOWED_PULSE_ZOOMS = (
    _PULSE_SETTINGS.close_up_zoom,
    min(_PULSE_SETTINGS.ordinary_zoom, 1.058),
    _PULSE_SETTINGS.ordinary_zoom,
)


def _keyframe_values(
    item: EditorialTransformKeyframe,
) -> tuple[float, float, float, float]:
    return (item.zoom, item.center_x, item.center_y, item.roll_degrees)


def _keyframe_is_neutral(item: EditorialTransformKeyframe) -> bool:
    return all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=_EPSILON)
        for left, right in zip(
            _keyframe_values(item),
            (1.0, 0.5, 0.5, 0.0),
            strict=True,
        )
    )


def _keyframes_match(
    left: EditorialTransformKeyframe,
    right: EditorialTransformKeyframe,
) -> bool:
    return all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=_EPSILON)
        for a, b in zip(
            _keyframe_values(left),
            _keyframe_values(right),
            strict=True,
        )
    )


def _track_is_neutral(track: EditorialTransformTrack | None) -> bool:
    return track is None or bool(track.keyframes) and all(
        _keyframe_is_neutral(item) for item in track.keyframes
    )


def _pulse_candidate(item: EditorialMotionSegment) -> bool:
    return bool(
        item.track is not None
        and _PULSE_REASON in item.reasons
        and any(not _keyframe_is_neutral(keyframe) for keyframe in item.track.keyframes)
    )


def _contiguous_groups(indices: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    values = tuple(sorted(set(indices)))
    if not values:
        return ()
    groups: list[list[int]] = [[values[0]]]
    for value in values[1:]:
        if value == groups[-1][-1] + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    return tuple(tuple(group) for group in groups)


def _output_frame_schedule(
    segments: Sequence[RenderSegment],
    output_fps: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Mirror RenderPlan cumulative rounding on the rendered frame clock."""

    if (
        isinstance(output_fps, bool)
        or not isinstance(output_fps, (int, float))
        or not math.isfinite(float(output_fps))
        or output_fps <= 0.0
    ):
        raise ValueError("joint editorial overlay output fps is invalid")
    starts: list[float] = []
    local_last_frame_times: list[float] = []
    cumulative_seconds = 0.0
    previous_boundary = 0
    for segment in segments:
        starts.append(previous_boundary / float(output_fps))
        cumulative_seconds += segment.duration
        boundary = int(round(cumulative_seconds * float(output_fps)))
        frame_count = boundary - previous_boundary
        if frame_count < 2:
            raise ValueError(
                "joint editorial overlay segment is shorter than two frames"
            )
        local_last_frame_times.append(
            min(segment.duration, (frame_count - 1) / float(output_fps))
        )
        previous_boundary = boundary
    return (tuple(starts), tuple(local_last_frame_times))


def _tier_d_hold_between(
    left_index: int,
    *,
    segments: Sequence[RenderSegment],
    transitions: Sequence[RenderTransition],
) -> bool:
    transition = transitions[left_index]
    return bool(
        segments[left_index].source_index
        == segments[left_index + 1].source_index
        and transition.tier == "D"
        and transition.reason == _TIER_D_HOLD_REASON
    )


def _split_distinct_pulse_events(
    groups: Sequence[tuple[int, ...]],
    proposal_by_index: dict[int, EditorialMotionSegment],
) -> tuple[tuple[int, ...], ...]:
    """Do not merge two adjacent pulses that both settle at the boundary."""

    output: list[tuple[int, ...]] = []
    for group in groups:
        current: list[int] = []
        for index in group:
            track = proposal_by_index[index].track
            if not isinstance(track, EditorialTransformTrack):
                if current:
                    output.append(tuple(current))
                    current = []
                continue
            if current:
                previous = proposal_by_index[current[-1]].track
                assert isinstance(previous, EditorialTransformTrack)
                if _keyframe_is_neutral(
                    previous.keyframes[-1]
                ) and _keyframe_is_neutral(track.keyframes[0]):
                    output.append(tuple(current))
                    current = []
            current.append(index)
        if current:
            output.append(tuple(current))
    return tuple(output)


def _pulse_group_matches_planner_contract(
    group: tuple[int, ...],
    *,
    tracks: Sequence[EditorialTransformTrack],
    segments: Sequence[RenderSegment],
    transitions: Sequence[RenderTransition],
    frame_starts: Sequence[float],
    local_last_frame_times: Sequence[float],
    output_fps: float,
) -> bool:
    """Verify the exact centered, guarded triangular pulse emitted upstream."""

    if not group or len(group) != len(tracks):
        return False
    for index, track in zip(group, tracks, strict=True):
        if not math.isclose(
            track.keyframes[-1].time_seconds,
            local_last_frame_times[index],
            rel_tol=0.0,
            abs_tol=_TIME_EPSILON,
        ):
            return False
        if any(
            abs(keyframe.center_x - 0.5) > _EPSILON
            or abs(keyframe.center_y - 0.5) > _EPSILON
            or abs(keyframe.roll_degrees) > _EPSILON
            for keyframe in track.keyframes
        ):
            return False
    if not _keyframe_is_neutral(tracks[0].keyframes[0]) or not (
        _keyframe_is_neutral(tracks[-1].keyframes[-1])
    ):
        return False
    for left_index, right_index in zip(group[:-1], group[1:], strict=True):
        if right_index != left_index + 1 or not _tier_d_hold_between(
            left_index,
            segments=segments,
            transitions=transitions,
        ):
            return False

    points: list[tuple[float, EditorialTransformKeyframe]] = []
    for index, track in zip(group, tracks, strict=True):
        for keyframe in track.keyframes:
            global_time = frame_starts[index] + keyframe.time_seconds
            frame_position = global_time * float(output_fps)
            if not math.isclose(
                frame_position,
                round(frame_position),
                rel_tol=0.0,
                abs_tol=_TIME_EPSILON,
            ):
                return False
            if points and math.isclose(
                points[-1][0],
                global_time,
                rel_tol=0.0,
                abs_tol=_TIME_EPSILON,
            ):
                if not _keyframes_match(points[-1][1], keyframe):
                    return False
                points[-1] = (global_time, keyframe)
            else:
                points.append((global_time, keyframe))
    active = [
        index
        for index, (_time, keyframe) in enumerate(points)
        if keyframe.zoom > 1.0 + _EPSILON
    ]
    if not active or active[0] == 0 or active[-1] == len(points) - 1:
        return False
    pulse_start = points[active[0] - 1]
    pulse_end = points[active[-1] + 1]
    if not _keyframe_is_neutral(pulse_start[1]) or not _keyframe_is_neutral(
        pulse_end[1]
    ):
        return False
    maximum = max(keyframe.zoom for _time, keyframe in points)
    if not any(
        math.isclose(
            maximum,
            allowed,
            rel_tol=0.0,
            abs_tol=_EPSILON,
        )
        for allowed in _ALLOWED_PULSE_ZOOMS
    ):
        return False
    peak_indices = [
        index
        for index, (_time, keyframe) in enumerate(points)
        if math.isclose(
            keyframe.zoom,
            maximum,
            rel_tol=0.0,
            abs_tol=_EPSILON,
        )
    ]
    if len(peak_indices) != 1:
        return False
    peak_index = peak_indices[0]
    if any(
        right[1].zoom + _EPSILON < left[1].zoom
        for left, right in zip(points[:peak_index], points[1 : peak_index + 1])
    ) or any(
        right[1].zoom > left[1].zoom + _EPSILON
        for left, right in zip(points[peak_index:], points[peak_index + 1 :])
    ):
        return False
    rise_seconds = points[peak_index][0] - pulse_start[0]
    fall_seconds = pulse_end[0] - points[peak_index][0]
    if not (
        0.55 - _TIME_EPSILON <= rise_seconds <= 0.85 + _TIME_EPSILON
        and 0.55 - _TIME_EPSILON <= fall_seconds <= 0.85 + _TIME_EPSILON
    ):
        return False
    for point_index, (time_value, keyframe) in enumerate(points):
        if point_index <= peak_index:
            if time_value <= pulse_start[0]:
                expected_zoom = 1.0
            else:
                alpha = (time_value - pulse_start[0]) / rise_seconds
                expected_zoom = 1.0 + (maximum - 1.0) * alpha
        else:
            if time_value >= pulse_end[0]:
                expected_zoom = 1.0
            else:
                alpha = (time_value - points[peak_index][0]) / fall_seconds
                expected_zoom = maximum + (1.0 - maximum) * alpha
        if not math.isclose(
            keyframe.zoom,
            expected_zoom,
            rel_tol=0.0,
            abs_tol=_TIME_EPSILON,
        ):
            return False

    run_first = group[0]
    while run_first > 0 and _tier_d_hold_between(
        run_first - 1,
        segments=segments,
        transitions=transitions,
    ):
        run_first -= 1
    run_last = group[-1]
    while run_last < len(segments) - 1 and _tier_d_hold_between(
        run_last,
        segments=segments,
        transitions=transitions,
    ):
        run_last += 1
    run_start = frame_starts[run_first]
    run_end = frame_starts[run_last] + local_last_frame_times[run_last]
    guard = _PULSE_SETTINGS.interior_pulse_neutral_guard_seconds
    return bool(
        pulse_start[0] - run_start + _TIME_EPSILON >= guard
        and run_end - pulse_end[0] + _TIME_EPSILON >= guard
    )


@dataclass(frozen=True, slots=True)
class JointEditorialMotionOverlayResult:
    plan: EditorialMotionPlan
    segments: tuple[RenderSegment, ...]
    applied_pulse_count: int
    applied_segment_indices: tuple[int, ...]
    rejected_group_count: int

    @property
    def applied(self) -> bool:
        return self.applied_pulse_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": "joint_run_local_interior_motion_overlay_v1",
            "applied": self.applied,
            "applied_pulse_count": self.applied_pulse_count,
            "applied_zoom_event_count": self.applied_pulse_count,
            "applied_pan_event_count": 0,
            "applied_segment_indices": list(self.applied_segment_indices),
            "rejected_group_count": self.rejected_group_count,
            "provider_nonidentity_tracks_preserved": True,
            "cross_source_boundary_state_changed": False,
            "roll_enabled": False,
            "thresholds_lowered": False,
        }


def merge_joint_interior_motion_overlay(
    *,
    provider_plan: EditorialMotionPlan,
    provider_segments: Sequence[RenderSegment],
    transitions: Sequence[RenderTransition],
    proposal_plan: EditorialMotionPlan,
    output_fps: float,
) -> JointEditorialMotionOverlayResult:
    """Adopt safe pulse runs without globally vetoing unrelated provider tracks.

    A pulse may cross an internal segmentation boundary only when the decoded
    renderer contract identifies it as the same source and same shot Tier-D
    hold, and all local tracks are slices of one frame-clock triangular curve.
    The complete pulse run must enter and leave at identity on actual rendered
    frames, so every visible cut keeps the provider-authorized boundary state.
    """

    if not isinstance(provider_plan, EditorialMotionPlan) or not isinstance(
        proposal_plan,
        EditorialMotionPlan,
    ):
        raise TypeError("joint editorial overlay requires typed motion plans")
    segments = (
        provider_segments
        if isinstance(provider_segments, tuple)
        else tuple(provider_segments)
    )
    transition_items = tuple(transitions)
    if len(transition_items) != max(0, len(segments) - 1):
        raise ValueError("joint editorial overlay transition count changed")

    def indexed(
        plan: EditorialMotionPlan,
        name: str,
    ) -> dict[int, EditorialMotionSegment]:
        values = {item.segment_index: item for item in plan.segments}
        if len(values) != len(plan.segments) or set(values) != set(
            range(len(segments))
        ):
            raise ValueError(f"{name} plan does not cover every segment once")
        return values

    provider_by_index = indexed(provider_plan, "provider")
    proposal_by_index = indexed(proposal_plan, "proposal")
    for index, segment in enumerate(segments):
        provider_item = provider_by_index[index]
        proposal_item = proposal_by_index[index]
        if (
            provider_item.source_index != segment.source_index
            or proposal_item.source_index != segment.source_index
            or provider_item.track is not segment.editorial_transform
        ):
            raise ValueError("joint editorial overlay source/track binding changed")

    frame_starts, local_last_frame_times = _output_frame_schedule(
        segments,
        output_fps,
    )
    candidate_groups = _split_distinct_pulse_events(
        _contiguous_groups(
            tuple(
                index
                for index, item in proposal_by_index.items()
                if _pulse_candidate(item)
            )
        ),
        proposal_by_index,
    )
    accepted_groups: list[tuple[int, ...]] = []
    rejected_group_count = 0
    for group in candidate_groups:
        tracks = tuple(proposal_by_index[index].track for index in group)
        if any(track is None for track in tracks):
            rejected_group_count += 1
            continue
        typed_tracks = tuple(
            track for track in tracks if isinstance(track, EditorialTransformTrack)
        )
        if len(typed_tracks) != len(group):
            rejected_group_count += 1
            continue
        if any(
            not _track_is_neutral(provider_by_index[index].track)
            or segments[index].crop is not None
            or segments[index].pan_crop is not None
            for index in group
        ):
            rejected_group_count += 1
            continue
        if not _pulse_group_matches_planner_contract(
            group,
            tracks=typed_tracks,
            segments=segments,
            transitions=transition_items,
            frame_starts=frame_starts,
            local_last_frame_times=local_last_frame_times,
            output_fps=output_fps,
        ):
            rejected_group_count += 1
            continue
        accepted_groups.append(group)

    accepted_indices = tuple(
        index for group in accepted_groups for index in group
    )
    accepted_set = set(accepted_indices)
    if not accepted_groups:
        return JointEditorialMotionOverlayResult(
            plan=provider_plan,
            segments=segments,
            applied_pulse_count=0,
            applied_segment_indices=(),
            rejected_group_count=rejected_group_count,
        )
    merged_plan_segments = tuple(
        replace(
            proposal_by_index[index],
            reasons=tuple(
                sorted(
                    set(provider_by_index[index].reasons)
                    | set(proposal_by_index[index].reasons)
                )
            ),
        )
        if index in accepted_set
        else provider_by_index[index]
        for index in range(len(segments))
    )
    merged_render_segments = tuple(
        replace(
            segment,
            editorial_transform=proposal_by_index[index].track,
        )
        if index in accepted_set
        else segment
        for index, segment in enumerate(segments)
    )
    applied_count = len(accepted_groups)
    skipped = dict(provider_plan.skipped_boundary_reasons)
    if rejected_group_count:
        skipped["joint_interior_pulse_provider_or_boundary_conflict"] = (
            skipped.get(
                "joint_interior_pulse_provider_or_boundary_conflict",
                0,
            )
            + rejected_group_count
        )
    events = rendered_transform_event_records(merged_render_segments, output_fps)
    merged_plan = replace(
        provider_plan,
        segments=merged_plan_segments,
        zoom_event_count=len(events["zoom"]),
        pan_event_count=len(events["pan"]),
        roll_event_count=len(events["roll"]),
        interior_zoom_pulse_count=(
            provider_plan.interior_zoom_pulse_count + applied_count
        ),
        skipped_boundary_reasons=tuple(sorted(skipped.items())),
        section_directed=bool(
            provider_plan.section_directed or proposal_plan.section_directed
        ),
        analysis_basis="joint_provider_plus_run_local_interior_motion_v1",
    )
    return JointEditorialMotionOverlayResult(
        plan=merged_plan,
        segments=merged_render_segments,
        applied_pulse_count=applied_count,
        applied_segment_indices=accepted_indices,
        rejected_group_count=rejected_group_count,
    )


__all__ = [
    "JointEditorialMotionOverlayResult",
    "merge_joint_interior_motion_overlay",
]
