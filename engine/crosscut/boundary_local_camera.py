"""Renderer-exact, boundary-local crop envelopes; not cut authorization.

This optional planner uses the existing renderer types. It preserves measured
boundary holds, returns to the original camera between separated handles, and
reports where a subject-containment proof is actually required. Consumers must
still validate the decoded pixels at every active frame. In particular, an
identity middle is not permission to carry a face identity across a native cut.

It is separate from the active production provider until that provider's proof
and materialization contracts can be migrated together.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from .render import EditorialTransformKeyframe, EditorialTransformTrack


def is_identity(state: EditorialTransformKeyframe) -> bool:
    return (
        abs(state.zoom - 1.) <= 1e-9
        and abs(state.center_x - .5) <= 1e-9
        and abs(state.center_y - .5) <= 1e-9
        and abs(state.roll_degrees) <= 1e-9
    )


@dataclass(frozen=True, slots=True)
class BoundaryLocalCamera:
    track: EditorialTransformTrack
    duration_seconds: float
    entry_active_until: float | None
    exit_active_from: float | None
    handles_overlap: bool

    def active_side_at(self, seconds: float) -> str | None:
        """None means actual identity pixels, not missing tracking evidence."""
        if not math.isfinite(seconds) or not 0. <= seconds <= self.duration_seconds:
            raise ValueError('camera sample is outside the segment')
        if is_identity(self.track.evaluate(seconds)):
            return None
        if self.handles_overlap:
            return 'bridge'
        if self.entry_active_until is not None and seconds <= self.entry_active_until:
            return 'entry'
        if self.exit_active_from is not None and seconds >= self.exit_active_from:
            return 'exit'
        raise RuntimeError('non-identity renderer state has no subject proof scope')


def build_boundary_local_camera(
    *,
    entry: EditorialTransformKeyframe,
    exit: EditorialTransformKeyframe,
    duration_seconds: float,
    output_fps: float,
    hold_frames: float = 3.5,
    settle_frames: float = 12.,
) -> BoundaryLocalCamera:
    """Compose measured cut states without forcing a crop over the whole shot.

Separated handles retain an identity middle. Overlapping short-shot handles
interpolate directly between the exact holds; they never insert an arbitrarily
fast zoom-out/zoom-in. Renderer edge, pan and zoom validation remains unchanged.
    """
    values = (duration_seconds, output_fps, hold_frames, settle_frames)
    if any(isinstance(v, bool) or not isinstance(v, (float, int))
           or not math.isfinite(v) for v in values):
        raise ValueError('camera timing must be finite numbers')
    if output_fps <= 0 or duration_seconds * output_fps < 2. - 1e-9:
        raise ValueError('camera segment needs at least two output frames')
    if hold_frames < 0 or settle_frames <= 0:
        raise ValueError('camera hold must be nonnegative and settle positive')
    if not isinstance(entry, EditorialTransformKeyframe) or not isinstance(exit, EditorialTransformKeyframe):
        raise TypeError('camera endpoints must use renderer keyframes')
    entry = replace(entry, time_seconds=0.)
    exit = replace(exit, time_seconds=duration_seconds)
    entry_active, exit_active = not is_identity(entry), not is_identity(exit)
    hold, ramp = hold_frames / output_fps, settle_frames / output_fps
    entry_hold = hold if entry_active else 0.
    exit_hold = duration_seconds - hold if exit_active else duration_seconds
    same = all(abs(getattr(entry, name)-getattr(exit, name)) <= 1e-9
               for name in ('zoom', 'center_x', 'center_y', 'roll_degrees'))
    if entry_hold >= exit_hold - 1e-9 and not same:
        raise ValueError('segment is too short for exact boundary transform holds')
    entry_end = entry_hold + ramp if entry_active else None
    exit_start = exit_hold - ramp if exit_active else None
    separated = (
        (entry_end or 0.) <= (exit_start if exit_start is not None else duration_seconds) - 1e-9
    )
    points = [entry]
    if separated:
        if entry_active:
            if entry_hold > 1e-9:
                points.append(replace(entry, time_seconds=entry_hold))
            points.append(EditorialTransformKeyframe(entry_end, 1.))
        if exit_active:
            points.append(EditorialTransformKeyframe(exit_start, 1.))
            if exit_hold < duration_seconds - 1e-9:
                points.append(replace(exit, time_seconds=exit_hold))
    elif not same:
        if entry_hold > 1e-9:
            points.append(replace(entry, time_seconds=entry_hold))
        if exit_hold < duration_seconds - 1e-9:
            points.append(replace(exit, time_seconds=exit_hold))
    points.append(exit)
    track = EditorialTransformTrack(
        tuple(points),
        reason=('boundary_local_exact_holds_identity_middle_v1' if separated
                else 'boundary_local_exact_holds_short_bridge_v1'),
    )
    return BoundaryLocalCamera(track, duration_seconds, entry_end, exit_start, not separated)
