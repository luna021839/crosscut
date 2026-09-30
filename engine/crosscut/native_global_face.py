"""Transient native-pixel global faces for bounded GROUP motion/camera work.

A valid small upright detection is not proof that a larger foreground face
was observed. The 480px legacy pass can miss that face and promote a background
performer. Read the 854px native upright detections before deciding dominance.
Do not rewrite persistent descriptors or use centrality/member identity.
"""
from dataclasses import dataclass
import cv2

from .roll_aware_face_geometry import detect_roll_aware_faces
from .visual_score import _dominant_face,_normalized_face_metadata,_working_frame


@dataclass(frozen=True,slots=True)
class NativeGlobalFace:
    face_count: int
    bbox_xywh: tuple | None
    landmarks_xy: tuple | None
    head_roll_radians: float | None
    eye_scale: float | None
    geometry: tuple | None


def observe_native_global_face(frame):
    """Same 1.8 area margin; real upright ambiguity cannot become a guess.

    Roll fallback is allowed only with no upright five-point observations.
    Rotated box envelopes otherwise can turn conflicting predictions of one
    head into a misleading largest area. Missing/ambiguous stays unknown.
    """
    working=cv2.resize(_working_frame(frame,maximum_width=960),(854,480))
    faces=detect_roll_aware_faces(working,angles=(0.,))
    if not faces:
        faces=detect_roll_aware_faces(working)
    box,points,roll,eye,geometry=_normalized_face_metadata(_dominant_face(faces),
        frame_width=854,frame_height=480)
    return NativeGlobalFace(len(faces),box,points,roll,eye,geometry)
