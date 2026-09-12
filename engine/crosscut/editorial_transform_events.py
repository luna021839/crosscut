"""One frame-visible trajectory event contract for producers and reviewers.

These are active trajectory-target events, not authored pulse counts or counts
of boundaries with a static crop. A pulse sliced across segments may contribute
more than one trajectory target; its separate authored-pulse count stays one.
"""

from __future__ import annotations

import math
from typing import Sequence

from .render import (
    EditorialTransformKeyframe,
    RenderSegment,
    quantized_segment_frame_counts,
)


_CLOCK_TOLERANCE = 1e-6
_NEUTRAL_TOLERANCE = 1e-8
_NEUTRAL = {"zoom": (1.0,), "pan": (0.5, 0.5), "roll": (0.0,)}


def _value(keyframe: EditorialTransformKeyframe, component: str) -> tuple[float, ...]:
    if component == "zoom":
        return (keyframe.zoom,)
    if component == "pan":
        return (keyframe.center_x, keyframe.center_y)
    return (keyframe.roll_degrees,)


def _different(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return any(
        not math.isclose(a, b, rel_tol=0.0, abs_tol=_NEUTRAL_TOLERANCE)
        for a, b in zip(left, right, strict=True)
    )


def _active(value: tuple[float, ...], component: str) -> bool:
    if component == "pan":
        return math.hypot(value[0] - 0.5, value[1] - 0.5) > _NEUTRAL_TOLERANCE
    return abs(value[0] - _NEUTRAL[component][0]) > _NEUTRAL_TOLERANCE


def rendered_transform_event_records(
    segments: Sequence[RenderSegment], fps: float
) -> dict[str, tuple[tuple[float, int], ...]]:
    """Measure actual output clocks without decoding or modifying any media.

    Evaluate only prospective first-visible samples, sharing those samples
    across the three components. Constant tracks need no sample evaluation.
    This preserves the reviewer's output-frame/endpoint/sub-frame semantics
    without eagerly evaluating every frame three times for every candidate.
    """
    events: dict[str, list[tuple[float, int]]] = {name: [] for name in _NEUTRAL}
    frame_counts = quantized_segment_frame_counts(segments, fps)
    global_start = 0
    for index, (segment, count) in enumerate(zip(segments, frame_counts, strict=True)):
        track = segment.editorial_transform
        if track is None:
            global_start += count
            continue
        samples: dict[int, EditorialTransformKeyframe] = {}
        for target_index, (left, target) in enumerate(
            zip(track.keyframes, track.keyframes[1:]), start=1
        ):
            interval_end = (
                track.keyframes[target_index + 1].time_seconds
                if target_index + 1 < len(track.keyframes) else math.inf
            )
            first_frame = max(
                0, int(math.floor((left.time_seconds + _CLOCK_TOLERANCE) * fps)) + 1
            )
            for component, neutral in _NEUTRAL.items():
                left_value, target_value = _value(left, component), _value(target, component)
                if not _active(target_value, component) or not _different(left_value, target_value):
                    continue
                for frame in range(first_frame, count):
                    time = frame / fps
                    if time <= left.time_seconds + _CLOCK_TOLERANCE:
                        continue
                    if time > interval_end + _CLOCK_TOLERANCE:
                        break
                    if frame not in samples:
                        samples[frame] = track.evaluate(time)
                    value = _value(samples[frame], component)
                    if _active(value, component) and _different(value, left_value):
                        events[component].append(((global_start + frame) / fps, index))
                        break
        global_start += count
    result: dict[str, tuple[tuple[float, int], ...]] = {}
    for component, records in events.items():
        ordered: list[tuple[float, int]] = []
        for record in sorted(records):
            if not ordered or not math.isclose(
                record[0], ordered[-1][0], rel_tol=0.0, abs_tol=_CLOCK_TOLERANCE
            ):
                ordered.append(record)
        result[component] = tuple(ordered)
    return result
