"""Worker-owned actual NanoDet/RTMPose sensing for a declared formation.

Each model has its own asset identity/call count in the shared session. Body
ROIs participate in the observation cache key. No pixels or feature results
are written to persistent raw caches; no assets are acquired by this module.
The caller must declare the required cohort before camera/path evaluation.
"""
from pathlib import Path
import hashlib

from .body_pose_2d import BodyPose2DAnalyzer,RTMPOSE_BODY7_MODEL_SHA256
from .group_joint_formation import ObservedFormationPerson,DeclaredFormationView
from .pose import _NanoDetFormationDetector,_EXPECTED_SHA256,FORMATION_PERSON_DETECTOR_MODEL


def register_formation_models(session,*,person_model_path,pose_model_path):
    person_model_path,pose_model_path=Path(person_model_path),Path(pose_model_path)
    for path,digest in ((person_model_path,_EXPECTED_SHA256[FORMATION_PERSON_DETECTOR_MODEL]),
            (pose_model_path,RTMPOSE_BODY7_MODEL_SHA256)):
        session.budget.check()
        with path.open('rb') as stream:
            actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=digest:raise ValueError('joint formation model content differs from the pinned asset')
    detector=_NanoDetFormationDetector(person_model_path)
    pose=BodyPose2DAnalyzer(pose_model_path)
    session.budget.check()
    session.register_model('nanodet_people',person_model_path,detector.infer)
    def body(pixels,*,bbox_xywh):
        return pose.describe(pixels,bbox_xywh=tuple(bbox_xywh))
    session.register_model('rtmpose_body7',pose_model_path,body)


def observe_formation_people(session,state,output_tick):
    """Return every bounded detected person, including missing/occluded pose.

    Missing body evidence is not dropped into a smaller 'complete' cohort.
    No central-person, named identity or silhouette substitute is introduced.
    More than twelve detections is unresolved, not silently truncated.
    """
    rows=session.observe(state,output_tick,model='nanodet_people')
    if rows is None or len(rows)>12:return None
    width,height=state.output_size
    people=[]
    for index,row in enumerate(rows):
        left,top,right,bottom,confidence=map(float,row)
        box=(left/width,top/height,(right-left)/width,(bottom-top)/height)
        if box[2]<=0 or box[3]<=0:return None
        body=session.observe(state,output_tick,model='rtmpose_body7',arguments={'bbox_xywh':list(box)})
        points=body.upper_body_xyc if body is not None and body.reason=='observed' else ()
        people.append(ObservedFormationPerson(index,box,points,confidence))
    return tuple(people)


def declared_formation_observation(session,state,output_tick,*,required_observation_ids):
    people=observe_formation_people(session,state,output_tick)
    if people is None or not set(required_observation_ids)<={p.observation_id for p in people}:return None
    return DeclaredFormationView(people,tuple(required_observation_ids),state.output_size)
