"""Worker-owned actual NanoDet/RTMPose sensing for a declared formation.

Each model has its own asset identity/call count in the shared session. Body
ROIs participate in the observation cache key. No pixels or feature results
are written to persistent raw caches; no assets are acquired by this module.
The caller must declare the required cohort before camera/path evaluation.
"""
from pathlib import Path
from dataclasses import replace
import hashlib
import numpy as np
from scipy.optimize import linear_sum_assignment

from .body_pose_2d import BodyPose2DAnalyzer,RTMPOSE_BODY7_MODEL_SHA256
from .group_joint_formation import (ObservedFormationPerson,DeclaredFormationView,
    ObservedFormationTemporalMotion,_bones)
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
    session.register_model('nanodet_people',person_model_path,detector.infer,
        adapter_fingerprint={'contract':'nanodet_native_people_roi_v1',
            'implementation_sha256':hashlib.sha256(Path(__file__).with_name('pose.py').read_bytes()).hexdigest(),
            'observer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    def body(pixels,*,bbox_xywh,face_bbox_xywh=None):
        return pose.describe(pixels,bbox_xywh=tuple(bbox_xywh),
            face_bbox_xywh=None if face_bbox_xywh is None else tuple(face_bbox_xywh))
    session.register_model('rtmpose_body7',pose_model_path,body,
        adapter_fingerprint=BodyPose2DAnalyzer.fingerprint())


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
    if people is None:return None
    return DeclaredFormationView(people,tuple(required_observation_ids),state.output_size)


def observe_temporal_formation_people(session,state,*,outgoing):
    """Sense actual three-tick articulation paths on this SAME whole camera.

    A same-source body is associated through measured pixel optical flow and
    one-to-one box correspondence, not screen sort or centrality. Uncertain
    flow/ROI ownership or fewer than two observed bones yields NO velocity.
    The caller still declares its foreground cohort and must sense the other
    side of the cut. These two-view paths are not whole-shot/dense approval.
    """
    from .boundary_subject_alignment import _optical_flow_box
    from .cut_anchored_containment import _iou
    if type(outgoing) is not bool:raise ValueError('temporal formation side must be explicit')
    if state.end_output_frame-state.first_output_frame<4:return None
    ticks=((state.end_output_frame-4,state.end_output_frame-1) if outgoing else
        (state.first_output_frame,state.first_output_frame+3))
    pixels=tuple(session.projected_pixels(state,tick) for tick in ticks)
    people=tuple(observe_formation_people(session,state,tick) for tick in ticks)
    if any(p is None for p in pixels) or any(p is None for p in people):return None
    endpoint=people[1] if outgoing else people[0]
    if not people[0] or not people[1]:return endpoint
    flow=tuple(_optical_flow_box(pixels[0],pixels[1],person.box_xywh) for person in people[0])
    overlaps=np.asarray([[(_iou(box,p.box_xywh) if box is not None else 0.)
        for p in people[1]] for box in flow])
    rows,columns=linear_sum_assignment(1.-overlaps)
    measured={};seconds=3/state.output_fps
    native=tuple(state.native_frame_indices[tick-state.first_output_frame] for tick in ticks)
    for i,j in zip(rows,columns):
        if overlaps[i,j]<.60:continue
        competing=np.delete(overlaps[i],j)
        if len(competing) and overlaps[i,j]-float(competing.max())<.10:continue
        prior_competition=np.delete(overlaps[:,j],i)
        if len(prior_competition) and overlaps[i,j]-float(prior_competition.max())<.10:continue
        previous,current=people[0][i],people[1][j]
        a,b=_bones(previous,state.output_size),_bones(current,state.output_size)
        common=tuple(name for name in a if name in b)
        if len(common)<2:continue
        velocities=tuple((name,*map(float,(b[name]-a[name])/seconds)) for name in common)
        evidence=ObservedFormationTemporalMotion(state.key,ticks,native,state.output_fps,
            (previous.observation_id,current.observation_id),velocities,True)
        identity=current.observation_id if outgoing else previous.observation_id
        measured[identity]=evidence
    return tuple(replace(p,temporal_motion=measured.get(p.observation_id)) for p in endpoint)
