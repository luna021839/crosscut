"""Visible person-rectangle HEIGHT intervals under an actual viewport.

Every number here is the clipped on-screen height of an ORIGINAL-frame
detector rectangle after the actual digital camera [zoom, centre x, centre y]
is applied. It is NOT face size, identity, anatomy, full-body size, a verified
camera jump or a same-person claim. All rectangles count; nothing selects the
largest, median, centre or a performer, and detector score never ranks.

Retained limitation: one performer standing (visible .8) versus crouching
(visible .25) still produces a HIGH visible-extent gap. Missing rectangles on
either side stay UNKNOWN and are never replaced by a known face height.
"""
from __future__ import annotations

import math

import numpy as np

# Same log-scale unit as played_partial_pose.face_geometry.
LOG_STEP = math.log(1.5)
SEMANTICS = dict(visible_rectangle_extent_only=True, face_size=False, identity=False,
    anatomy=False, verified_camera_jump=False, same_person=False, posture_invariant=False,
    score_used_for_ranking=False, person_count_or_mask=False)


def _rectangles(rectangles):
    if rectangles is None:
        return np.zeros((0, 5), np.float64)
    if (not isinstance(rectangles, np.ndarray) or rectangles.dtype != np.float32
            or rectangles.ndim != 2 or rectangles.shape[1] != 5):
        raise ValueError('normalized float32 N x5 XYXY+score person rectangles required')
    if (not np.isfinite(rectangles).all() or (rectangles < 0).any() or (rectangles > 1).any()
            or np.any(rectangles[:, 2:4] <= rectangles[:, :2])):
        raise ValueError('person rectangles must be finite, inside [0,1], x1>x0 and y1>y0')
    return rectangles.astype(np.float64)


def _cameras(cameras):
    camera = np.asarray(cameras, dtype=np.float64)
    if camera.ndim < 1 or camera.shape[-1] != 3:
        raise ValueError('cameras (...,3) [zoom, centre x, centre y] required')
    if not np.isfinite(camera).all() or (camera[..., 0] <= 0).any():
        raise ValueError('finite cameras with positive zoom required')
    return camera


def height_interval(rectangles, cameras):
    """Return (log_min_height, log_max_height, measured), shape cameras.shape[:-1].

    Heights are in output-height units of the visible viewport intersection.
    Only intersections with positive width AND positive height are used.
    None, zero boxes or nothing visible -> measured False with finite dummy 0.
    """
    camera = _cameras(cameras)
    boxes = _rectangles(rectangles)
    zoom, cx, cy = camera[..., 0], camera[..., 1], camera[..., 2]
    shape = camera.shape[:-1]
    low = np.zeros(shape, np.float64)
    high = np.zeros(shape, np.float64)
    measured = np.zeros(shape, bool)
    # One box at a time: memory stays O(cameras), never cameras x persons.
    with np.errstate(over='ignore'):
        for x0, y0, x1, y1 in boxes[:, :4]:
            width = np.minimum(1., .5 + zoom * (x1 - cx)) - np.maximum(0., .5 + zoom * (x0 - cx))
            height = np.minimum(1., .5 + zoom * (y1 - cy)) - np.maximum(0., .5 + zoom * (y0 - cy))
            visible = (width > 0.) & (height > 0.)
            log_height = np.log(np.where(visible, height, 1.))
            low = np.where(visible, np.where(measured, np.minimum(low, log_height), log_height), low)
            high = np.where(visible, np.where(measured, np.maximum(high, log_height), log_height), high)
            measured = measured | visible
    return np.asarray(low), np.asarray(high), np.asarray(measured)


def _triple(interval):
    try:
        low, high, measured = interval
    except (TypeError, ValueError) as error:
        raise ValueError('interval triple (log_min, log_max, measured) required') from error
    low, high, measured = np.asarray(low), np.asarray(high), np.asarray(measured)
    if measured.dtype != np.bool_ or low.dtype.kind not in 'fiu' or high.dtype.kind not in 'fiu':
        raise ValueError('real interval limits and boolean measured flags required')
    low, high, measured = np.broadcast_arrays(low.astype(np.float64), high.astype(np.float64), measured)
    if not (np.isfinite(low).all() and np.isfinite(high).all()) or np.any(measured & (low > high)):
        raise ValueError('finite ordered interval limits required')
    return low, high, measured


def interval_gap(left, right):
    """Return (gap, pair_measured) in log(1.5) units, broadcasting both sides.

    Overlapping intervals give 0. Unknown on either side gives dummy 0 with
    pair_measured False; that 0 is NOT a match.
    """
    ll, lh, lm = _triple(left)
    rl, rh, rm = _triple(right)
    ll, lh, lm, rl, rh, rm = np.broadcast_arrays(ll, lh, lm, rl, rh, rm)
    both = lm & rm
    separation = np.maximum(0., np.maximum(rl - lh, ll - rh)) / LOG_STEP
    return np.asarray(np.where(both, separation, 0.)), np.asarray(both)


def used_face_risk(face_risk, face_measured, extent_gap, extent_measured):
    """Known face keeps its EXACT original risk; otherwise max(1, extent gap) or 1.

    No admission rule, pose change, hard gate or soft correspondence claim.
    """
    face = np.asarray(face_risk)
    face_known = np.asarray(face_measured)
    gap = np.asarray(extent_gap)
    gap_known = np.asarray(extent_measured)
    if (face.dtype.kind not in 'fiu' or gap.dtype.kind not in 'fiu'
            or face_known.dtype != np.bool_ or gap_known.dtype != np.bool_):
        raise ValueError('real risks/gaps and boolean measured flags required')
    face, face_known, gap, gap_known = np.broadcast_arrays(face, face_known, gap, gap_known)
    if not (np.isfinite(face).all() and np.isfinite(gap).all()) or (gap < 0).any():
        raise ValueError('finite face risk and nonnegative finite extent gap required')
    fallback = np.where(gap_known, np.maximum(1., gap), 1.)
    return np.asarray(np.where(face_known, face, fallback))
