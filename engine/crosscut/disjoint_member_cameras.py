"""Compose already authored, non-overlapping nominal camera intervals exactly."""
from dataclasses import replace
import math
from .render import EditorialTransformTrack

HARD_CAMERA_REASONS=frozenset(('confirmed_member_subject_scale_position_match',
    'confirmed_member_subject_scale_position_match_dense_performer_follow'))

def _neutral(point):
    return all(abs(a-b)<=1e-9 for a,b in zip(
        (point.zoom,point.center_x,point.center_y,point.roll_degrees),(1.,.5,.5,0.)))

def compose_disjoint_member_cameras(existing,joint,*,duration):
    """Preserve both piecewise-linear tracks, with no new motion or retiming.

    Union keyframes partition both curves into linear intervals. On each
    interval one entire curve must be neutral, including both endpoints.
    Dense containment and decoded QA remain the caller's responsibility.
    """
    if (not isinstance(existing,EditorialTransformTrack) or not isinstance(joint,EditorialTransformTrack)
            or existing.reason not in HARD_CAMERA_REASONS or not math.isfinite(duration) or duration<=0
            or any(t.final_keyframe.time_seconds>duration+1e-9 for t in (existing,joint))):
        raise ValueError('disjoint composition needs an authored hard camera on the same nominal clock')
    times=sorted({0.,duration,*(p.time_seconds for p in existing.keyframes),*(p.time_seconds for p in joint.keyframes)})
    samples=[(existing.evaluate(t),joint.evaluate(t)) for t in times]
    for (a,b),(c,d) in zip(samples,samples[1:]):
        if not ((_neutral(a) and _neutral(c)) or (_neutral(b) and _neutral(d))):
            raise ValueError('nominal camera intervals overlap; neither camera may be overwritten')
    points=tuple(replace(b if _neutral(a) else a,time_seconds=t) for t,(a,b) in zip(times,samples))
    return EditorialTransformTrack(points,reason='disjoint_hard_and_joint_member_camera_pending_containment')
