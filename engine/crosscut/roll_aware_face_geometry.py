"""Bounded, measured YuNet geometry for tilted heads, without identity claims.

This is an opt-in experiment, not a replacement for the persistent visual
descriptor. Rotated detections are mapped back to the original pixels. A
missing observation remains missing; neither interpolation nor Haar boxes
are promoted to measured five-point face geometry.
"""
from __future__ import annotations

from itertools import product
import math

import cv2
import numpy as np

from .visual_score import _DetectedFace, _detect_faces


ROLL_ANGLES = (0., -30., 30., -60., 60., -90., 90.)


def rotated_face_input(frame, degrees):
    """Return a no-crop canvas and its original-to-canvas affine matrix."""
    height, width = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2., height / 2.), degrees, 1.)
    sine, cosine = abs(matrix[0, 1]), abs(matrix[0, 0])
    out_width = int(math.ceil(width * cosine + height * sine))
    out_height = int(math.ceil(height * cosine + width * sine))
    matrix[0, 2] += (out_width - width) / 2.
    matrix[1, 2] += (out_height - height) / 2.
    return cv2.warpAffine(frame, matrix, (out_width, out_height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), matrix


def restore_face_coordinates(face, matrix, *, width, height):
    """Reject padding detections; do not clamp invented landmarks to edges."""
    if (face.landmarks_xy is None or len(face.landmarks_xy) != 5
            or face.confidence is None or not .76 <= face.confidence <= 1.):
        return None
    box = np.asarray(face.bbox_xywh, dtype=float)
    points = np.asarray(face.landmarks_xy, dtype=float)
    if (box.shape != (4,) or not np.isfinite(box).all()
            or (box[2:] <= 0.).any() or points.shape != (5, 2)
            or not np.isfinite(points).all()):
        return None
    inverse = cv2.invertAffineTransform(matrix)
    landmarks = cv2.transform(points[None, :, :], inverse)[0]
    if ((landmarks < 0.).any() or (landmarks >= (width, height)).any()
            or np.linalg.norm(landmarks[0] - landmarks[1]) < 2.):
        return None
    x, y, w, h = box
    corners = np.asarray([[(x, y), (x+w, y), (x, y+h), (x+w, y+h)]])
    restored = cv2.transform(corners, inverse)[0]
    low, high = restored.min(axis=0), restored.max(axis=0)
    # Clipping the observed bounding rectangle to actual frame pixels is not
    # the same as inventing out-of-frame eye/nose/mouth coordinates.
    low, high = np.maximum(low, (0., 0.)), np.minimum(high, (width, height))
    if (high <= low).any():
        return None
    return _DetectedFace(tuple(float(v) for v in (*low, *(high-low))),
        tuple(tuple(float(v) for v in point) for point in landmarks), float(face.confidence))


def same_measured_face(first, second):
    """Cross-angle deduplication requires landmarks, not overlapping boxes."""
    if first.landmarks_xy is None or second.landmarks_xy is None:
        return False
    a, b = (np.asarray(face.landmarks_xy, dtype=float) for face in (first, second))
    if a.shape != (5, 2) or b.shape != (5, 2):
        return False
    scale = min(np.linalg.norm(a[0]-a[1]), np.linalg.norm(b[0]-b[1]))
    # Some detector rotations exchange image-left/right order. Nose is never
    # exchanged, and all five points must agree for a duplicate decision.
    for eyes, mouth in product(((0, 1), (1, 0)), ((3, 4), (4, 3))):
        if np.linalg.norm(a-b[(*eyes, 2, *mouth), :], axis=1).max() <= max(2., .35*scale):
            return True
    # A tilted YuNet face can regress different eye positions in differently
    # rotated inputs. Those are conflicting predictions of ONE head, not
    # extra people. Suppression needs a near-identical nose and face center
    # as well as substantial box overlap; adjacent people's boxes alone do
    # not qualify. The retained landmarks still need downstream pixel QA.
    ax, ay, aw, ah = first.bbox_xywh
    bx, by, bw, bh = second.bbox_xywh
    intersection = max(0., min(ax+aw, bx+bw)-max(ax, bx))*max(0., min(ay+ah, by+bh)-max(ay, by))
    union = aw*ah+bw*bh-intersection
    distance = math.hypot(ax+aw/2-bx-bw/2, ay+ah/2-by-bh/2)
    if (union > 0. and intersection/union >= .45
            and distance <= .30*min(math.hypot(aw, ah), math.hypot(bw, bh))
            and np.linalg.norm(a[2]-b[2]) <= max(2., .60*scale)):
        return True
    return False


def detect_roll_aware_faces(frame, *, angles=ROLL_ANGLES, detector=None):
    """Measure at most seven rotations; retain distinct/ambiguous people."""
    if (not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3
            or min(frame.shape[:2]) < 2):
        raise ValueError('roll-aware geometry requires a BGR frame')
    angles = tuple(angles)
    if (not 1 <= len(angles) <= len(ROLL_ANGLES) or len(set(angles)) != len(angles)
            or any(type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 90. for v in angles)):
        raise ValueError('roll-aware geometry needs a bounded unique angle set')
    detect = detector or _detect_faces
    result = []
    height, width = frame.shape[:2]
    for angle in angles:
        rotated, matrix = rotated_face_input(frame, angle)
        faces, backend = detect(rotated, cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY))
        if backend != 'yunet':
            continue
        for face in sorted(faces, key=lambda face: -(face.confidence or 0.)):
            restored = restore_face_coordinates(face, matrix, width=width, height=height)
            if restored is not None and not any(same_measured_face(restored, item) for item in result):
                # Prefer the earliest angle for the SAME observed head. This
                # preserves native zero-roll geometry when it already works.
                result.append(restored)
    return tuple(result)
