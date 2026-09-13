"""Paired face geometry proposals; never quality or identity authorization.

The exact mode preserves the existing shared-extent solver. The opt-in partial
mode holds an already-large face at the least required enlargement while the
smaller face approaches it within the same renderer envelope. It does not claim
an exact match, increase the zoom limit, or bypass decoded-frame review.
"""
from __future__ import annotations

import math
from .render import (
    EDITORIAL_TRANSFORM_MAXIMUM_CENTER_SHIFT,
    EDITORIAL_TRANSFORM_MAXIMUM_ZOOM,
)

_EPSILON = 1e-9


def matched_face_crops(
    outgoing_bbox: tuple[float, float, float, float],
    incoming_bbox: tuple[float, float, float, float],
    *,
    allow_partial_extent_match: bool = False,
) -> tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
] | None:
    """Propose a shared on-screen face scale/center inside renderer bounds.

    This is a pixel-transform proposal, never substituted for decoded pose or
    identity evidence. The caller still remeasures both transformed images.
    The old fixed extent often clamped *both* faces to maximum zoom, leaving
    their scale ratio unchanged. Solve their shared feasible extent instead.
    """

    if not isinstance(allow_partial_extent_match, bool):
        raise TypeError("allow_partial_extent_match must be boolean")
    boxes = (outgoing_bbox, incoming_bbox)
    if any(
        len(box) != 4
        or not all(math.isfinite(value) for value in box)
        or min(box[2:]) <= 0.0
        or min(box[:2]) < 0.0
        or box[0] + box[2] > 1.0 + _EPSILON
        or box[1] + box[3] > 1.0 + _EPSILON
        for box in boxes
    ):
        return None
    extents = tuple(max(box[2:]) for box in boxes)
    lower = max(extents)
    maximum = min(extents) * EDITORIAL_TRANSFORM_MAXIMUM_ZOOM
    if lower > maximum + _EPSILON and not allow_partial_extent_match:
        return None
    upper = max(lower, min(0.72, maximum))
    raw_centers = tuple(
        (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
        for box in boxes
    )
    # Start with the least enlargement. Increase only when both source-frame
    # margins cannot place those equally sized faces at one common position.
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        target = lower + (upper - lower) * fraction
        scales = tuple(extent / target for extent in extents)
        if allow_partial_extent_match:
            scales = tuple(
                max(1.0 / EDITORIAL_TRANSFORM_MAXIMUM_ZOOM, min(1.0, scale))
                for scale in scales
            )
        low = tuple(
            max(
                max(
                    (center[axis] - 1.0 + scale) / scale
                    for center, scale in zip(raw_centers, scales)
                ),
                max(
                    box[axis + 2] / (2.0 * scale)
                    for box, scale in zip(boxes, scales)
                ),
            )
            for axis in (0, 1)
        )
        high = tuple(
            min(
                min(
                    center[axis] / scale
                    for center, scale in zip(raw_centers, scales)
                ),
                min(
                    1.0 - box[axis + 2] / (2.0 * scale)
                    for box, scale in zip(boxes, scales)
                ),
            )
            for axis in (0, 1)
        )
        if any(left > right + _EPSILON for left, right in zip(low, high)):
            continue
        circles = tuple(
            (
                0.5 + (center[0] - 0.5) / scale,
                0.5 + (center[1] - 0.5) / scale,
                EDITORIAL_TRANSFORM_MAXIMUM_CENTER_SHIFT / scale,
            )
            for center, scale in zip(raw_centers, scales)
        )
        preferred = (0.5, 0.42)
        candidates = [
            tuple(max(low[axis], min(high[axis], preferred[axis]))
                  for axis in (0, 1)),
            *((x, y) for x in (low[0], high[0]) for y in (low[1], high[1])),
        ]
        for cx, cy, radius in circles:
            distance = math.hypot(preferred[0] - cx, preferred[1] - cy)
            if distance > _EPSILON:
                candidates.append((
                    cx + radius * (preferred[0] - cx) / distance,
                    cy + radius * (preferred[1] - cy) / distance,
                ))
            for x in (low[0], high[0]):
                residual = radius * radius - (x - cx) ** 2
                if residual >= -_EPSILON:
                    dy = math.sqrt(max(0.0, residual))
                    candidates.extend(((x, cy - dy), (x, cy + dy)))
            for y in (low[1], high[1]):
                residual = radius * radius - (y - cy) ** 2
                if residual >= -_EPSILON:
                    dx = math.sqrt(max(0.0, residual))
                    candidates.extend(((cx - dx, y), (cx + dx, y)))
        ax, ay, ar = circles[0]
        bx, by, br = circles[1]
        distance = math.hypot(bx - ax, by - ay)
        if (
            distance > _EPSILON
            and abs(ar - br) - _EPSILON <= distance <= ar + br + _EPSILON
        ):
            along = (ar * ar - br * br + distance * distance) / (2 * distance)
            height = math.sqrt(max(0.0, ar * ar - along * along))
            ux, uy = (bx - ax) / distance, (by - ay) / distance
            center_x, center_y = ax + along * ux, ay + along * uy
            candidates.extend((
                (center_x - height * uy, center_y + height * ux),
                (center_x + height * uy, center_y - height * ux),
            ))
        feasible = [
            point for point in candidates
            if all(low[axis] - _EPSILON <= point[axis] <= high[axis] + _EPSILON
                   for axis in (0, 1))
            and all(math.hypot(point[0] - cx, point[1] - cy) <= radius + _EPSILON
                    for cx, cy, radius in circles)
        ]
        if not feasible:
            continue
        shared = min(feasible, key=lambda point: (
            (point[0] - preferred[0]) ** 2 + (point[1] - preferred[1]) ** 2,
            point,
        ))
        return tuple(
            (
                min(1.0 - scale, max(0.0, center[0] - shared[0] * scale)),
                min(1.0 - scale, max(0.0, center[1] - shared[1] * scale)),
                scale,
                scale,
            )
            for center, scale in zip(raw_centers, scales)
        )
    return None
