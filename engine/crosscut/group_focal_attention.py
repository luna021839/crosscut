"""Source-local focal geometry for a foreground performer in a group picture.

A similarly large face at the edge must not erase an otherwise unambiguous
central foreground face. This is deliberately separate from the persistent
visual descriptor and its global-largest-face contract. It does not identify
a member, prove cross-camera identity, or approve a transition.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

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


@dataclass(frozen=True, slots=True)
class GroupFocalSubject:
    """Transient source-local attention, never a persistent identity label."""
    bbox_xywh: tuple[float, float, float, float]
    landmarks_xy: tuple[tuple[float, float], ...] | None
    roll_radians: float | None
    eye_scale: float | None
    person: FormationPersonDescriptor
    detected_face_count: int
    basis: str


def resolve_group_focal_subject(frame, descriptor):
    """One face/body decision shared by motion and experimental local framing.

    An observed global primary has priority. Missing ownership must not select
    another person's more convenient body. A central supplement is allowed
    only when the global primary itself is unknown; it remains a separate basis.
    """
    face = descriptor.primary_face_bbox
    if face is not None:
        person = face_associated_person(face, descriptor.formation_people)
        if person is None:
            return None
        return GroupFocalSubject(face,
            getattr(descriptor, 'primary_face_landmarks_xy', None),
            getattr(descriptor, 'primary_face_head_roll_radians', None),
            getattr(descriptor, 'primary_face_eye_scale', None), person,
            getattr(descriptor, 'face_count', 1), GLOBAL_FOCAL_BASIS)
    focal = observe_central_focal_face(frame, descriptor.formation_people)
    if focal is None:
        return None
    return GroupFocalSubject(focal.bbox_xywh, focal.landmarks_xy,
        focal.roll_radians, focal.eye_scale, focal.person,
        focal.detected_face_count, CENTRAL_FOCAL_BASIS)


def focal_landmarks_track(reference, target, before, after):
    """Check measured five-point landmarks against adjacent-frame pixel flow.

    A face turning non-rigidly may not fit the box tracker's similarity model.
    This alternative still requires four measured points, forward/backward
    pixel agreement, and agreement with the independently detected new face.
    Similar boxes or central screen position alone cannot pass it.

    Four pyramid levels permit coarse-to-fine search in a UHD source; the
    former two-level limit missed observed motion at native 4K resolution.
    Residuals below are still checked in ORIGINAL pixels, with the same 1.25px
    forward/backward limit. Neither downsampled residuals nor larger admission
    tolerances substitute for a measured path.
    """
    if reference.shape != target.shape or before is None or after is None:
        return False
    first, second = np.asarray(before, dtype=np.float32), np.asarray(after, dtype=np.float32)
    if (first.shape != (5,2) or second.shape != (5,2)
            or not np.isfinite(first).all() or not np.isfinite(second).all()
            or (first < 0).any() or (first > 1).any()
            or (second < 0).any() or (second > 1).any()):
        return False
    size = np.asarray((reference.shape[1],reference.shape[0]),dtype=np.float32)
    first, second = first*size, second*size
    eye_span = min(np.linalg.norm(first[0]-first[1]),np.linalg.norm(second[0]-second[1]))
    if eye_span < 4.:
        return False
    a, b = cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY), cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)
    options = dict(winSize=(15,15),maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    points = first.reshape(-1,1,2)
    forward, valid_forward, _ = cv2.calcOpticalFlowPyrLK(a,b,points,None,**options)
    if forward is None or valid_forward is None:
        return False
    backward, valid_backward, _ = cv2.calcOpticalFlowPyrLK(b,a,forward,None,**options)
    if backward is None or valid_backward is None:
        return False
    valid = (valid_forward.ravel().astype(bool) & valid_backward.ravel().astype(bool)
        & (np.linalg.norm(backward.reshape(-1,2)-first,axis=1) <= 1.25)
        & (np.linalg.norm(forward.reshape(-1,2)-second,axis=1) <= max(2.,.25*eye_span)))
    return int(valid.sum()) >= 4


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
