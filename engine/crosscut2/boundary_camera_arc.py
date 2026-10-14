'''Frozen two-sided camera path for one hold [first, final).

Local frame k is output frame first + k, and L = final - first.  Rows are
(zoom, x, y) crop centres in normalised source coordinates.  Roll is not
supported.  The entry strip occupies k = 0 .. n-1 and the exit strip occupies
k = L-n .. L-1, with n = min(7, L).  Both are reproduced bit-for-bit.

* L <= 13: the strips overlap (2 frames at 12, 1 at 13, all frames at L <= 7).
  Overlapping rows must already be identical, otherwise the arc is invalid
  (None).  No hidden longer minimum hold is introduced.
* 14 <= L < 14 + 2*C: one cubic-Hermite offset from the baseline spans the
  free frames.  It matches both strips' end offsets and one-sided discrete
  slopes, so neither strip overwrites the other.
* L >= 14 + 2*C: an entry connector on frames [7, 6+C] and an exit connector
  on [L-7-C, L-8] fade the offset to and from exactly zero.  Every frame
  between them is the unchanged baseline.

Offsets are additive in (zoom, x, y).  The cubic is sampled once per integer
frame, and the renderer must replay values row by row.  Slope matching is a
property of these samples, not a C1 claim about any interpolating renderer.
Every frame is checked for zoom in [1, cap], pan radius <= .25 and crop
padding.  A violation returns None instead of clamping.  Digital motion created
here is camera-cost evidence only and must never be credited as performer
motion or content.
'''
from dataclasses import dataclass
import math

import numpy as np

__all__ = ['CameraArc', 'build', 'realizable', 'STRIP']
STRIP = 7
TOLERANCE = 1e-9  # floating rounding only; values are never clamped


@dataclass(frozen=True, eq=False)
class CameraArc:
    values: np.ndarray   # (L, 3) read-only camera for every replayed frame
    changed: np.ndarray  # (L,) read-only, True where values differ from baseline
    knots: tuple         # frames where the sampled slope changes (diagnostic only)
    kind: str            # 'overlap' | 'abutting' | 'bridge' | 'connectors'


def realizable(values, cap):
    '''True when every row is finite and inside zoom, pan and padding limits.'''
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        return False
    zoom, x, y = v.T
    if np.any(zoom < 1 - TOLERANCE) or np.any(zoom > cap + TOLERANCE):
        return False
    pad = .5 / zoom
    inside = ((x >= pad - TOLERANCE) & (x <= 1 - pad + TOLERANCE)
              & (y >= pad - TOLERANCE) & (y <= 1 - pad + TOLERANCE))
    return bool(inside.all() and np.all(np.hypot(x - .5, y - .5) <= .25 + TOLERANCE))


def _hermite(p0, m0, p1, m1, span):
    '''Samples at offsets 1 .. span-1 of a cubic Hermite on [0, span].'''
    s = np.arange(1, span)[:, None] / span
    s2, s3 = s * s, s * s * s
    return ((2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * span * m0
            + (3 * s2 - 2 * s3) * p1 + (s3 - s2) * span * m1)


def _rows(value, length, name):
    a = np.array(value, dtype=np.float64)
    if a.shape != (length, 3) or not np.isfinite(a).all():
        raise ValueError(f'{name} must be finite with shape ({length}, 3)')
    return a


def build(entry_strip, exit_strip, baseline_values, *, cap, connector_frames=12):
    '''Return a frozen CameraArc, or None when no realizable path exists.'''
    base = np.array(baseline_values, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 3 or len(base) < 1:
        raise ValueError('baseline_values must have shape (L, 3) with L >= 1')
    length = len(base)
    base = _rows(base, length, 'baseline_values')
    n = min(STRIP, length)
    entry = _rows(entry_strip, n, 'entry_strip')
    leave = _rows(exit_strip, n, 'exit_strip')
    if (isinstance(cap, bool) or not isinstance(cap, (int, float, np.integer, np.floating))
            or not math.isfinite(cap) or cap < 1):
        raise ValueError('finite zoom cap >= 1 required')
    if (isinstance(connector_frames, bool) or not isinstance(connector_frames, (int, np.integer))
            or connector_frames < 1):
        raise ValueError('positive integer connector_frames required')
    overlap = 2 * n - length
    if overlap > 0 and not np.array_equal(entry[length - n:], leave[:overlap]):
        return None
    path = base.copy()
    path[:n] = entry
    path[length - n:] = leave
    span = length - 2 * STRIP + 1
    if span <= 1:
        kind = 'overlap' if overlap > 0 else 'abutting'
    else:
        c = int(connector_frames)
        d_in = entry[-1] - base[STRIP - 1]
        m_in = (entry[-1] - entry[-2]) - (base[STRIP - 1] - base[STRIP - 2])
        d_out = leave[0] - base[length - STRIP]
        m_out = (leave[1] - leave[0]) - (base[length - STRIP + 1] - base[length - STRIP])
        zero = np.zeros(3)
        if length >= 2 * STRIP + 2 * c:
            path[STRIP:STRIP - 1 + c] += _hermite(d_in, m_in, zero, zero, c)
            path[length - STRIP - c + 1:length - STRIP] += _hermite(zero, zero, d_out, m_out, c)
            kind = 'connectors'
        else:
            path[STRIP:length - STRIP] += _hermite(d_in, m_in, d_out, m_out, span)
            kind = 'bridge'
    if not realizable(path, cap):
        return None
    changed = np.any(path != base, axis=1)
    bends = np.flatnonzero(np.any(np.diff(path, 2, axis=0) != 0, axis=1)) + 1
    knots = tuple(sorted({0, length - 1, *(int(k) for k in bends)}))
    path.setflags(write=False)
    changed.setflags(write=False)
    return CameraArc(path, changed, knots, kind)
