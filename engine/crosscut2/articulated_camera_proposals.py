"""Observed face/arm GROUP similarity proposal, not a post-selection refit.

Face-center and visible corresponding-arm blocks have equal total weight.
Raw keypoints remain untouched. No rotation, mirroring, retiming or identity
claim is made. Ordinary played face/body/crop costs still score this view.
"""
from dataclasses import dataclass
import math
import numpy as np
from .paired_geometry import Camera


@dataclass(frozen=True)
class PreparedObservation:
    head: np.ndarray
    points: np.ndarray
    known: np.ndarray


def prepare(sample):
    """Validate one physical endpoint once, not once per possible partner."""
    side=sample['side']
    if side not in (0,1):raise ValueError('actual played side required')
    at=1 if side==0 else 0
    body=np.asarray(sample['body']);kind=np.asarray(sample['evidence'])
    ambiguous=np.asarray(sample['ownership_ambiguity']);head=np.asarray(sample['head'])
    if (body.shape!=(2,8,3) or kind.shape!=(2,8) or ambiguous.shape!=(2,8)
            or kind.dtype.kind not in 'iu' or np.any((kind<0)|(kind>3)) or ambiguous.dtype!=bool
            or head.shape!=(4,) or not np.isfinite(body).all() or not np.isfinite(head).all()):
        raise ValueError('bound actual face and two body samples required')
    if head[2]<=.019 or head[3]<.75:return None
    valid=np.isin(kind[at,:6],(1,2))&~ambiguous[at,:6]&(body[at,:6,2]>=.3)
    valid&=(body[at,:6,:2]>=0).all(-1)&(body[at,:6,:2]<=1).all(-1)
    values=[head.copy(),body[at,:6,:2].copy(),valid.copy()]
    for value in values:value.setflags(write=False)
    return PreparedObservation(*values)


def prepared_proposal(left,right,*,caps):
    """The unchanged fit, consuming immutable source-bound endpoint copies."""
    if len(caps)!=2 or any(not math.isfinite(c) or not 1<=c<=2 for c in caps):
        raise ValueError('two actual source zoom capacities required')
    if left is None or right is None:return None
    common=left.known&right.known
    arms=[chain for chain in ((0,2,4),(1,3,5)) if common[list(chain)].all()]
    if not arms:return None
    joints=sorted({j for chain in arms for j in chain})
    a,b=[np.vstack((sample.head[:2],sample.points[joints])) for sample in (left,right)]
    weights=np.r_[1.,np.full(len(joints),1./len(joints))];weights/=weights.sum()
    ca,cb=[np.sum(xy*weights[:,None],axis=0) for xy in (a,b)]
    da,db=(a-ca)*[16/9,1.],(b-cb)*[16/9,1.]
    variance=float(np.sum(weights*np.sum(da*da,axis=-1)))
    covariance=float(np.sum(weights*np.sum(da*db,axis=-1)))
    if variance<=1e-8 or covariance<=0:return None
    ratio=float(np.clip(covariance/variance,1./caps[1],caps[0]))
    za,zb=max(1.,ratio),max(1.,1./ratio)
    lows=[1.+z*(c-1.) for c,z in ((ca,za),(cb,zb))]
    highs=[z*c for c,z in ((ca,za),(cb,zb))]
    low,high=np.maximum(*lows),np.minimum(*highs)
    common_position=np.where(low<=high,np.clip((ca+cb)/2.,low,high),(low+high)/2.)
    cameras=[]
    for center,z,lo,hi in ((ca,za,lows[0],highs[0]),(cb,zb,lows[1],highs[1])):
        shown=np.clip(common_position,lo,hi);camera=center+(.5-shown)/z
        delta=camera-.5;camera=.5+delta*min(1.,.25/max(1e-12,float(np.linalg.norm(delta))))
        camera=np.clip(camera,.5/z,1.-.5/z)
        cameras.append(Camera(float(z),*map(float,camera)))
    return tuple(cameras)


def proposal(left,right,*,caps):
    """A bounded extra pair, or None without a complete corresponding arm.

    Scale fits the actual last/first native samples. Absolute scale uses the
    least crop realizing the feasible ratio; old face/native options remain.
    """
    if len(caps)!=2 or not np.isfinite(caps).all() or any(not 1<=c<=2 for c in caps):
        raise ValueError('two actual source zoom capacities required')
    first=prepare(left)
    if first is None:return None
    second=prepare(right)
    return prepared_proposal(first,second,caps=caps)
