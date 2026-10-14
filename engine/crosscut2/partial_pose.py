"""Partial close-up evidence from two fixed, face-guided top-down proposals.

An overlapping detector rectangle is not a reason to skip every pose forward.
Conversely, a nose near a face does not make the model's arms belong to it.
This adapter keeps only crop-corroborated measured joints, from one unchanged
primary hypothesis. It neither fuses skeletons nor invents absent joints.
Raw model hypotheses remain the caller's cache/evidence, not overwritten here.
Crop agreement is limited evidence, NOT actor ownership or cut certification.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PartialPose:
    points: np.ndarray
    supported: np.ndarray
    reason: str


def face_rectangle(head, *, aspect=16/9):
    row=np.asarray(head,float)
    if row.shape!=(4,) or not np.isfinite(row).all() or not np.isfinite(aspect) or aspect<=0:
        raise ValueError('one finite source face and positive raster aspect required')
    x,y,h,confidence=row
    if h<=0 or confidence<.75:return None
    lo=np.maximum(0.,[x-.45*h/aspect,y-h/2])
    hi=np.minimum(1.,[x+.45*h/aspect,y+h/2])
    if np.any(hi<=lo):return None
    return np.r_[lo,hi-lo]


def face_proposals(head, *, aspect=16/9):
    """Freeze two proposals before any new pose or pair-matching error is seen."""
    if face_rectangle(head,aspect=aspect) is None:return ()
    x,y,h,_=np.asarray(head,float)
    proposals=[]
    for width_scale,downward_extent in ((2.,4.5),(2.3,4.)):
        lo=np.maximum(0.,[x-width_scale*h/aspect,y-.75*h])
        hi=np.minimum(1.,[x+width_scale*h/aspect,y+downward_extent*h])
        proposals.append(tuple(map(float,np.r_[lo,hi-lo])))
    return tuple(proposals)


def _anchored_raw(evidence,face,others,aspect):
    if evidence is None or evidence.reason!='observed':return None,'missing_raw_hypothesis'
    raw=np.asarray(evidence.raw_keypoints_xyc,float)
    if raw.shape!=(17,3) or not np.isfinite(raw).all():raise ValueError('original finite raw17 observations required')
    contains=lambda box,xy:bool(np.all(xy>=box[:2]) and np.all(xy<=box[:2]+box[2:]))
    nose=raw[0]
    if nose[2]<.3 or not contains(face,nose[:2]):return None,'nose_outside_selected_face'
    for other in others:
        box=face_rectangle(other,aspect=aspect)
        if box is not None and contains(box,nose[:2]):return None,'competing_face_anchor'
    return raw[[5,6,7,8,9,10,11,12]],'single_face_guided_raw_hypothesis'


def face_guided_hypothesis(evidence,head,other_heads,*,aspect=16/9):
    """One measured proposal, explicitly NOT a two-crop corroboration.

    Only six upper-body joints are exposed. Hip visibility is not inferred
    from an upper-body crop or a large raw SimCC peak.
    """
    face=face_rectangle(head,aspect=aspect);others=np.asarray(other_heads,float)
    if others.ndim!=2 or others.shape[1]!=4 or not np.isfinite(others).all():raise ValueError('actual competing heads required')
    raw,reason=(None,'missing_face') if face is None else _anchored_raw(evidence,face,others,aspect)
    points=np.zeros((8,3),np.float32);supported=np.zeros(8,bool)
    if raw is not None:
        points[:6]=raw[:6]
        supported[:6]=(raw[:6,2]>=.3)&(raw[:6,:2]>=0).all(1)&(raw[:6,:2]<=1).all(1)
        points[:6,2]=np.where(supported[:6],np.clip(raw[:6,2],0.,1.),0.)
    return PartialPose(points,supported,reason)


def corroborate(evidences, head, other_heads, *, aspect=16/9):
    """Keep supported points from proposal 0, not the best-matching skeleton.

    A primary nose must be IN this face's approximate rectangle and in no
    competing actual face rectangle. The legacy .8 ellipse extended outside
    that rectangle and admitted a neighboring nose in the real close-up probe.
    Agreement is measured in face-height units, in the real source aspect.
    Two crops of one network are not statistically independent sensors.
    """
    missing=lambda why:PartialPose(np.zeros((8,3),np.float32),np.zeros(8,bool),why)
    face=face_rectangle(head,aspect=aspect)
    others=np.asarray(other_heads,float)
    if others.ndim!=2 or others.shape[1]!=4 or not np.isfinite(others).all():
        raise ValueError('all actual competing source heads required')
    if face is None or len(evidences)!=2:return missing('missing_face_or_crop')
    hypotheses=[]
    for evidence in evidences:
        raw,reason=_anchored_raw(evidence,face,others,aspect)
        if raw is None:return missing(reason)
        hypotheses.append(raw)
    a,b=hypotheses
    valid=lambda p:(p[:,2]>=.3)&(p[:,:2]>=0).all(1)&(p[:,:2]<=1).all(1)
    distance=np.linalg.norm((a[:,:2]-b[:,:2])*[aspect,1.],axis=1)/float(head[2])
    supported=valid(a)&valid(b)&(distance<=.3)
    points=a.astype(np.float32).copy()
    # SimCC peaks can exceed one. Retain raw peaks in the caller's hypotheses;
    # only this legacy body8 compatibility view uses a [0,1] support value.
    points[:,2]=np.where(supported,np.minimum(1.,np.minimum(a[:,2],b[:,2])),0.)
    # Keep the primary measured coordinates, even for unsupported slots. Only
    # the support mask changes; no averaging, pose completion or interpolation.
    return PartialPose(points,supported,'partial_crop_corroboration' if supported.any() else 'no_corroborated_joint')
