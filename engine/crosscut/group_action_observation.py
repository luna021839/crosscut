"""Fresh globally dominant head/arms geometry, never central-person selection.

The explicit action-camera caller owns the model and native frame clock. A
missed proxy face may be reacquired from these pixels at the existing dominance
margin; a missing/ambiguous body cannot borrow another performer's arms.
"""
from dataclasses import replace
from .boundary_subject_alignment import _observation_from_descriptor
from .focal_pose import face_associated_person
from .group_focal_attention import GroupFocalSubject, GLOBAL_FOCAL_BASIS
from .roll_aware_face_geometry import detect_roll_aware_faces
from .visual_score import describe_frame, _dominant_face, _normalized_face_metadata


def observe_global_head_arms(frame, analyzer):
    descriptor = describe_frame(frame, include_subject_geometry=True,
        include_formation_pose=False)
    observed = _observation_from_descriptor(descriptor)
    if observed.face_box_xywh is None:
        faces = detect_roll_aware_faces(frame, angles=(0.,))
        primary = _dominant_face(faces)
        if primary is None:
            faces = detect_roll_aware_faces(frame)
            primary = _dominant_face(faces)
        box, points, roll, eye, _ = _normalized_face_metadata(primary,
            frame_width=frame.shape[1], frame_height=frame.shape[0])
        if box is not None:
            observed = replace(observed, face_count=len(faces), face_box_xywh=box,
                face_landmarks_xy=points, face_head_roll_radians=roll,
                face_eye_scale=eye, face_is_unambiguous_dominant=True)
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
