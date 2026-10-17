"""Common-clock focal IMAGE motion plus the actual terminal digital view rate.

Matching face position at the cut does not match its visible movement. Native
image motion is compared on the SAME music samples, preserving choreography
reversals. Actual one-sided digital view rates are composed before comparison,
so compensating a native pan is not counted twice. No cause/identity is inferred.
"""
import numpy as np

TICKS = np.arange(-7, 7, dtype=np.int64)
TICKS.setflags(write=False)


def relative_cost(risk):
    """Evidence relative to UNKNOWN, not an additional per-cut rhythm tax.

    Raw risk stays0..1 and missing measurements remain explicitly.5. Only
    their advantage/disadvantage over that neutral is added to the existing
    objective; an entirely unobserved cue changes no old edit decision.
    """
    value=np.asarray(risk,float)
    if not np.isfinite(value).all() or np.any((value<0)|(value>1)):
        raise ValueError('A bounded focal image motion residual is required')
    return value-.5


def _project(head, camera):
    return (.5 + camera[..., :1] * (head[..., :2] - camera[..., 1:])) * [16 / 9, 1.]


def describe(observation, *, fps):
    """Only exact existing source-bound head samples; never interpolate gaps."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('A positive actual output clock is required')
    result = dict(velocity=np.zeros((13, 3)), known=np.zeros(13, bool),
                  digital_velocity=np.zeros(3), digital_known=False, height=0., fps=float(fps))
    context = observation.get('dense_subject_context')
    if context is None:
        return result
    clock, window, at = context['clock'], context['window'], context['at']
    if clock.fps != fps or not np.array_equal(clock.output_frames, TICKS):
        raise ValueError('Focal image motion requires the exact common14sample music clock')
    from .subject_action_match import _sample
    if not np.array_equal(_sample(window, clock), at):
        raise ValueError('Focal image motion source/native binding changed')
    heads = np.asarray(window.heads[at], float)
    strips = np.asarray(observation['camera_strip'], float)
    side = observation['side']
    if (heads.shape != (14, 4) or strips.shape != (7, 3) or side not in (0, 1)
            or not np.isfinite(strips).all() or np.any(strips[:, 0] < 1. - 1e-9)):
        raise ValueError('Actual14heads and seven played view transforms required')
    endpoint = 6 if side == 0 else 7
    boundary = strips[-1 if side == 0 else 0]
    known = ((heads[:, 2] > .019) & (heads[:, 3] >= .75)
             & window.associated[at] & np.asarray(context['same_shot'], bool))
    # A native association hole inside two output samples also breaks a step.
    continuous = np.asarray(context['contiguous'], bool)
    if continuous.shape != (13,):
        raise ValueError('Exact observed source association intervals required')
    values = np.c_[_project(heads, boundary), np.log(np.maximum(.001, heads[:, 2] * boundary[0]))]
    result['velocity'] = np.diff(values, axis=0) * fps
    result['known'] = known[:-1] & known[1:] & continuous
    result['height'] = float(heads[endpoint, 2] * boundary[0])
    first, last = (5, 6) if side == 0 else (7, 8)
    if not result['known'][first]:
        return result
    # This difference is exactly the digital component of the two ACTUAL
    # played frames, not an assumed stationary actor or a future camera path.
    a, b = (strips[-2], strips[-1]) if side == 0 else (strips[0], strips[1])
    actual_xy = _project(heads[last], b) - _project(heads[first], a)
    native_xy = _project(heads[last], boundary) - _project(heads[first], boundary)
    result['digital_velocity'] = np.r_[(actual_xy - native_xy) * fps, np.log(b[0] / a[0]) * fps]
    result['digital_known'] = True
    return result


def matrix(left, right):
    """Bounded time-mean mismatch of composed local image motion hypotheses.

    The measured terminal view rate is a local continuation hypothesis over
    the common source window, not an observed unplayed digital trajectory.
    Unsupported values remain explicit UNKNOWN(.5), with a separate mask.
    """
    if not left or not right or any(x['fps'] != left[0]['fps'] for x in (*left, *right)):
        raise ValueError('Two nonempty focal domains on one actual clock required')
    va = np.asarray([x['velocity'] + x['digital_velocity'] for x in left])
    vb = np.asarray([x['velocity'] + x['digital_velocity'] for x in right])
    known = (np.asarray([x['known'] for x in left])[:, None]
             & np.asarray([x['known'] for x in right])[None])
    # Use the same local -4..+3 scope as articulated source motion. Do not
    # turn distant native pans into evidence about the exposed seam.
    known &= ((TICKS[:-1] >= -4) & (TICKS[1:] <= 3))[None, None]
    known &= np.asarray([x['digital_known'] for x in left])[:, None, None]
    known &= np.asarray([x['digital_known'] for x in right])[None, :, None]
    scale = np.maximum(.02, (np.asarray([x['height'] for x in left])[:, None]
                            + np.asarray([x['height'] for x in right])[None]) / 2)
    gap = va[:, None] - vb[None]
    tau = 6. / left[0]['fps']
    xy = tau * np.linalg.norm(gap[..., :2], axis=-1) / scale[..., None]
    zoom = tau * np.abs(gap[..., 2]) / np.log(1.5)
    raw = np.maximum(xy, zoom)
    bounded = raw / (1. + raw)
    count = known.sum(-1)
    return np.where(count > 0, (bounded * known).sum(-1) / np.maximum(1, count), .5), count > 0, count
