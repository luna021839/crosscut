"""Measure the source-referenced member's pose in actual decoded output pixels.

The source's boxes/joints are never copied into the output. A matched output
face must have exactly one observed body before RTMPose measures that ROI.
Ambiguous or missing target bodies stay missing, rather than borrowing the
whole-frame dominant skeleton. This is the same eight-joint basis as selection.
"""
from pathlib import Path
import math
from .body_pose_2d import BodyPose2DAnalyzer
from .focal_pose import face_associated_person
from .focal_selector_geometry import FocalSelectorGeometry
from .pose import _NanoDetFormationDetector, _formation_people_descriptor


def measure_face_owned_pose(frame,candidate,people,pose):
    """Resolve duplicate body proposals only by agreeing observed skeletons.

    A face contained by two boxes is not necessarily two different people.
    Each ROI must independently locate the referenced face and actual torso.
    Missing evidence cannot win over an unmeasured competitor. Different
    skeletons remain unresolved even if one detector confidence is higher.
    """
    hypotheses=[p for p in people if face_associated_person(candidate.bbox_xywh,(p,)) is p]
    if not hypotheses or len(hypotheses)>4:return None
    verified=[]
    for body in hypotheses:
        measured=pose.describe(frame,bbox_xywh=body.bbox_xywh,face_bbox_xywh=candidate.bbox_xywh)
        if measured.reason!='observed':return None
        if measured.face_anchor_consistent is False:continue
        normalized=measured.as_formation_person(body.detection_confidence)
        if measured.face_anchor_consistent is not True or normalized is None:return None
        verified.append((measured,normalized))
    if not verified:return None
    height,width=frame.shape[:2];face_height=candidate.bbox_xywh[3]*height
    for i,(left,_) in enumerate(verified):
        for right,_ in verified[i+1:]:
            common=[j for j in range(8) if min(left.upper_body_xyc[j][2],right.upper_body_xyc[j][2])>=.3]
            if len(common)<5:return None
            for j in common:
                a,b=left.upper_body_xyc[j],right.upper_body_xyc[j]
                gap=math.hypot((a[0]-b[0])*width,(a[1]-b[1])*height)
                limit=(.20 if j in (0,1,6,7) else .50)*face_height
                if gap>limit:return None
    # Return an actual measurement, not averaged/invented joint coordinates.
    return max(verified,key=lambda item:min(item[0].upper_body_xyc[j][2] for j in (0,1,6,7)))


class RenderedFocalGeometryProvider:
    def __init__(self,frame_provider,pose_model_path):
        self.frame_provider=frame_provider
        self.pose=BodyPose2DAnalyzer(pose_model_path)
        self.detector=_NanoDetFormationDetector(Path(__file__).parent/'models/object_detection_nanodet_2022nov.onnx')
        self._cache={}

    def __call__(self,seconds,candidate):
        key=(seconds,candidate.bbox_xywh)
        if key in self._cache:return self._cache[key]
        frame=self.frame_provider(seconds)
        person=None;screen=()
        if frame is not None:
            people=_formation_people_descriptor(self.detector.infer(frame),
                frame_width=frame.shape[1],frame_height=frame.shape[0],minimum_width=.015,
                minimum_height=.045,minimum_confidence=.35)[0]
            observed=measure_face_owned_pose(frame,candidate,people,self.pose)
            if observed is not None:
                measured,person=observed
                screen=measured.upper_body_xyc
        value=FocalSelectorGeometry(0,seconds,candidate.bbox_xywh,
            candidate.eye_scale or candidate.landmark_scale,person,None,'rtmpose_body7',screen)
        self._cache[key]=value
        return value
