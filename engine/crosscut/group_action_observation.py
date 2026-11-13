"""Fresh globally dominant head/arms geometry, never central-person selection.

The explicit action-camera caller owns the model and native frame clock. A
missed proxy face may be reacquired from these pixels at the existing dominance
margin; a missing/ambiguous body cannot borrow another performer's arms.
"""
from dataclasses import replace
from .boundary_subject_alignment import _observation_from_descriptor
from .focal_pose import face_associated_person
from .group_focal_attention import GroupFocalSubject, GLOBAL_FOCAL_BASIS
from .native_global_face import observe_native_global_face
from .visual_score import describe_frame


def observe_global_head_arms(frame, analyzer):
    descriptor = describe_frame(frame, include_subject_geometry=True,
        include_formation_pose=False)
    observed = _observation_from_descriptor(descriptor)
    native=observe_native_global_face(frame)
    observed=replace(observed,face_count=native.face_count,face_box_xywh=native.bbox_xywh,
        face_landmarks_xy=native.landmarks_xy,face_head_roll_radians=native.head_roll_radians,
        face_eye_scale=native.eye_scale,face_is_unambiguous_dominant=native.bbox_xywh is not None)
    if not observed.face_is_unambiguous_dominant or observed.face_box_xywh is None:
        return observed
    person = face_associated_person(observed.face_box_xywh, descriptor.formation_people)
    if person is None:
        return observed
    focal = GroupFocalSubject(observed.face_box_xywh, observed.face_landmarks_xy,
        observed.face_head_roll_radians, observed.face_eye_scale, person,
        observed.face_count, GLOBAL_FOCAL_BASIS)
    return replace(observed, selected_focal_subject=focal, action_pose=analyzer.describe(frame,
        bbox_xywh=person.bbox_xywh, face_bbox_xywh=observed.face_box_xywh))
