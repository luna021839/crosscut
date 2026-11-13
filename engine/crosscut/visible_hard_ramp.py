"""Fit a single hard-cut ramp inside observed frames without moving its cut."""
import math
from collections.abc import Callable
from dataclasses import replace
from .render import EditorialTransformKeyframe, EditorialTransformTrack


def _neutral(point: EditorialTransformKeyframe) -> bool:
    return all(
        abs(a - b) <= 1e-9
        for a, b in zip(
            (point.zoom, point.center_x, point.center_y, point.roll_degrees),
            (1.0, 0.5, 0.5, 0.0),
        )
    )


def shorten_hard_ramp_to_observations(
    track: EditorialTransformTrack,
    *,
    fps: float,
    frame_count: int,
    frame_available: Callable[[int], bool],
) -> EditorialTransformTrack | None:
    """Return one bounded alternative; the caller must densely verify it again.

    Only an entry or exit ramp with an identity plateau is supported. The
    exact cut endpoint, zoom, pan and roll are preserved. Missing footage is
    left untransformed; no geometry is invented. Inspect only the original
    camera window and its one-frame halo, never the whole source or song.
    """
    if (
        not isinstance(track, EditorialTransformTrack)
        or len(track.keyframes) != 3
        or not math.isfinite(fps)
        or fps <= 0
        or isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 2
    ):
        return None
    first, middle, last = track.keyframes
    outgoing = _neutral(first) and _neutral(middle) and not _neutral(last)
    incoming = not _neutral(first) and _neutral(middle) and _neutral(last)
    if not (outgoing or incoming):
        return None
    active = {
        i for i in range(frame_count) if not _neutral(track.evaluate(i / fps))
    }
    reviewed = sorted(active | {
        j for i in active for j in (i - 1, i + 1) if 0 <= j < frame_count
    })
    missing = [i for i in reviewed if not frame_available(i)]
    if not missing:
        return None
    time = (max(missing) + 1) / fps if outgoing else (min(missing) - 1) / fps
    duration = last.time_seconds - time if outgoing else time
    original_duration = (
        last.time_seconds - middle.time_seconds if outgoing else middle.time_seconds
    )
    endpoint = last if outgoing else first
    if (
        duration < 0.55 - 1e-9
        or duration >= original_duration - 1e-9
        or abs(endpoint.zoom - 1.0) / duration > 1.0 + 1e-9
        or math.hypot(endpoint.center_x - 0.5, endpoint.center_y - 0.5)
        / duration > 0.5 + 1e-9
        or abs(endpoint.roll_degrees) > 1e-9
    ):
        return None
    try:
        return EditorialTransformTrack(
            (first, replace(middle, time_seconds=time), last), track.reason
        )
    except ValueError:
        return None
