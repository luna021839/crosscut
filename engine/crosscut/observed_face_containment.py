"""Measured nearby-frame face recovery, not cross-source cut authorization.

Nonrigid facial motion can fail a single-affine optical-flow model. A recovery
requires a freshly observed, unambiguous face in BOTH real frames, local box
continuity and SFace agreement. Missing/occluded faces are never synthesized.
The caller still owns exact clocks, native-cut guards and renderer containment.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import cv2
import numpy as np

from .face_identity import cosine_similarity, extract_sface_embedding
from .visual_score import _detect_faces, _normalized_face_metadata, _working_frame

Box = tuple[float, float, float, float]


def _valid_box(box):
    return (isinstance(box,(tuple,list)) and len(box)==4
            and all(not isinstance(v,bool) and isinstance(v,(float,int)) and math.isfinite(v) for v in box)
            and min(box[:2]) >= 0 and min(box[2:]) > 0
            and box[0]+box[2] <= 1.000001 and box[1]+box[3] <= 1.000001)


@dataclass(frozen=True, slots=True)
class ObservedFaceRecovery:
    box: Box
    appearance_similarity: float


def _iou(a: Box, b: Box) -> float:
    width=max(0., min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))
    height=max(0., min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
    area=width*height
    return area/max(1e-12,a[2]*a[3]+b[2]*b[3]-area)


def _observations(frame):
    working=_working_frame(frame)
    faces,backend=_detect_faces(working,cv2.cvtColor(working,cv2.COLOR_BGR2GRAY))
    if backend != 'yunet':
        return working, ()
    observations=[]
    for face in faces:
        bbox,landmarks,*_= _normalized_face_metadata(
            face,frame_width=working.shape[1],frame_height=working.shape[0])
        if bbox is not None and landmarks is not None:
            observations.append((bbox,landmarks))
    return working,tuple(observations)


def _unique_near(observations, reference_box, minimum_iou):
    ranked=sorted(((_iou(box,reference_box), index) for index,(box,_) in enumerate(observations)),reverse=True)
    if not ranked or ranked[0][0] < minimum_iou:
        return None
    if len(ranked)>1 and ranked[1][0] >= .35 and ranked[0][0]-ranked[1][0] < .1:
        return None
    return observations[ranked[0][1]]


def recover_observed_face_step(
    reference_frame: np.ndarray, target_frame: np.ndarray,
    reference_box: Box, observed_target_boxes: Sequence[Box],
) -> ObservedFaceRecovery | None:
    """Use appearance only for an already geometrically associated local face.

No appearance-based search over other image locations is performed. The target
must also coincide with a face observed by the caller's native-resolution path.
The .55 identity, .08 displacement and .85..1.18 scale limits retain the existing
track identity/motion bounds; they do not change the cross-source identity gate.
"""
    if (not _valid_box(reference_box) or not observed_target_boxes
            or any(not _valid_box(box) for box in observed_target_boxes)):
        return None
    if any(not isinstance(frame,np.ndarray) or frame.dtype != np.uint8
           or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size
           for frame in (reference_frame,target_frame)):
        return None
    if reference_frame.shape != target_frame.shape:
        return None
    before, left_faces=_observations(reference_frame)
    after, right_faces=_observations(target_frame)
    left=_unique_near(left_faces,reference_box,.7)
    right=_unique_near(right_faces,reference_box,.55)
    if left is None or right is None:
        return None
    box,landmarks=right
    matching=tuple(b for b in observed_target_boxes if _iou(box,b) >= .9)
    if len(matching) != 1:
        return None
    target_box=matching[0]
    center_shift=math.hypot(target_box[0]+target_box[2]/2-reference_box[0]-reference_box[2]/2,
                            target_box[1]+target_box[3]/2-reference_box[1]-reference_box[3]/2)
    if center_shift > .08 or any(not .85 <= target_box[i]/reference_box[i] <= 1.18 for i in (2,3)):
        return None
    left_embedding=extract_sface_embedding(before,bbox_xywh=left[0],landmarks_xy=left[1])
    right_embedding=extract_sface_embedding(after,bbox_xywh=box,landmarks_xy=landmarks)
    similarity=cosine_similarity(left_embedding,right_embedding)
    if similarity is None or similarity < .55:
        return None
    return ObservedFaceRecovery(target_box,similarity)
