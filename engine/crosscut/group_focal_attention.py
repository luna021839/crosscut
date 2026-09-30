"""Source-local focal geometry for a foreground performer in a group picture.

A similarly large face at the edge must not erase an otherwise unambiguous
central foreground face. This is deliberately separate from the persistent
visual descriptor and its global-largest-face contract. It does not identify
a member, prove cross-camera identity, or approve a transition.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2

from .focal_pose import face_associated_person
from .pose import FormationPersonDescriptor


CENTRAL_FOCAL_BASIS = 'central_landmark_face_unique_body_v1'
GLOBAL_FOCAL_BASIS = 'global_dominant_or_single'


@dataclass(frozen=True, slots=True)
class CentralFocalFace:
    bbox_xywh: tuple[float, float, float, float]
    landmarks_xy: tuple[tuple[float, float], ...]
    roll_radians: float
    eye_scale: float
    person: FormationPersonDescriptor
    detected_face_count: int


def select_central_focal_face(faces, people, *, frame_width, frame_height):
    """Select before body association; missing/ambiguous ownership stays unknown.

    Keep the existing 1.8 area margin, but compare candidates in the central
    composition corridor. A face must independently reach the established
    medium-shot scale (.42 * .22). Peripheral faces still exist; this selection
    is attention geometry, not a claim that only one person is visible.
    """
    from .visual_score import _normalized_face_metadata

    central = []
    for face in faces:
        box, landmarks, roll, eye_scale, _ = _normalized_face_metadata(
            face, frame_width=frame_width, frame_height=frame_height)
        if box is not None and .30 <= box[0] + box[2] / 2 <= .70:
            central.append((box[2] * box[3], box, landmarks, roll, eye_scale))
    if not central:
        return None
    central.sort(key=lambda item: item[0], reverse=True)
    area, box, landmarks, roll, eye_scale = central[0]
    if (len(central) > 1 and area < 1.8 * central[1][0]
            or box[3] < .0924 or landmarks is None or len(landmarks) != 5
            or roll is None or eye_scale is None):
        return None
    person = face_associated_person(box, people)
    if person is None:
        return None
    return CentralFocalFace(box, landmarks, roll, eye_scale, person, len(faces))


def observe_central_focal_face(frame, people):
    from .visual_score import _detect_faces

    faces, backend = _detect_faces(frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if backend != 'yunet':
        return None  # Haar cannot prove the required five-point geometry.
    return select_central_focal_face(faces, people,
        frame_width=frame.shape[1], frame_height=frame.shape[0])
