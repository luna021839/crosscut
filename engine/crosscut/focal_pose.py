"""Select already-observed upper-body geometry associated with one focal face.

This is source-local geometric association, NOT member identity or proof of a
valid cut. It never substitutes a different person's pose when the focal body
is missing or ambiguous. The eight 2D formation joints remain their own feature
basis; they must not be relabelled as the 60-component whole-body pose vector.
"""

from __future__ import annotations

import math
from typing import Sequence

from .pose import FormationPersonDescriptor


def face_associated_upper_body_person(
    face_bbox_xywh: tuple[float, float, float, float] | None,
    people: Sequence[FormationPersonDescriptor],
) -> FormationPersonDescriptor | None:
    """Return the unique observed body containing the focal face near its head.

Ambiguity is checked BEFORE looking for available pose points. Otherwise a
missing observation on the true subject could silently select a background
performer whose skeleton happens to be available. No model call is made.
    """
    if face_bbox_xywh is None or len(face_bbox_xywh) != 4:
        return None
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) for value in face_bbox_xywh
    ):
        return None
    x, y, width, height = face_bbox_xywh
    if (
        x < -1e-6 or y < -1e-6 or width <= 0 or height <= 0
        or x + width > 1 + 1e-6 or y + height > 1 + 1e-6
    ):
        return None
    face_center = (x + width / 2, y + height / 2)
    associated = []
    for person in people:
        if person.detection_confidence < 0.50:
            continue
        bx, by, bw, bh = person.bbox_xywh
        overlap_width = max(0.0, min(x + width, bx + bw) - max(x, bx))
        overlap_height = max(0.0, min(y + height, by + bh) - max(y, by))
        face_coverage = overlap_width * overlap_height / (width * height)
        if (
            face_coverage >= 0.85
            and bx <= face_center[0] <= bx + bw
            and by <= face_center[1] <= by + 0.55 * bh
            and 0.04 <= height / bh <= 0.65
        ):
            associated.append(person)
    if len(associated) != 1:
        return None
    person = associated[0]
    points = person.upper_body_keypoints_xyc
    if len(points) != 8 or sum(point[2] >= 0.20 for point in points) < 4:
        return None
    return person
