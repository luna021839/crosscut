"""Source-local pixel corroboration of raw body parts, not anatomical truth.

The same pose model can attach a wrist to a background dancer in both crops.
Agreement between those crops is not independent evidence. Here actual native
rasters corroborate each measured endpoint in both time directions. Failed
texture/flow remains UNKNOWN; no joint is moved, filled or made stationary.
The caller must bind the selected face, physical source clock and native shot.
"""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class PartPaths:
    supported: np.ndarray
    pixel_path_observed: np.ndarray
    model_pixel_disagreement: np.ndarray
    worst_error_face_heights: np.ndarray
    shoulder_anchor_observed: bool


def _paths(frames, points):
    """At most twelve real native steps; no total-displacement fast-motion veto."""
    h,w=frames[0].shape[:2];gray=[cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in frames]
    groups=[];features=[];radius=max(6,min(20,round(min(w,h)*.015)))
    for j,(x,y,score) in enumerate(points):
        if score<.3 or not 0<=x<1 or not 0<=y<1:continue
        xx,yy=round(x*w),round(y*h);left,top=max(0,xx-radius),max(0,yy-radius)
        patch=gray[0][top:min(h,yy+radius+1),left:min(w,xx+radius+1)]
        if min(patch.shape,default=0)<3:continue
        corners=cv2.goodFeaturesToTrack(patch,maxCorners=8,qualityLevel=.012,minDistance=3.,blockSize=3)
        if corners is None or len(corners)<3:continue
        first=len(features);features.extend(corners.reshape(-1,2)+[left,top]);groups.append((j,first,len(features)))
    output=np.full((8,2),np.nan)
    if not features:return output
    original=np.asarray(features,np.float32).reshape(-1,1,2);current=original.copy();valid=np.ones(len(original),bool)
    options=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    for before,after in zip(gray,gray[1:]):
        forward,fs,_=cv2.calcOpticalFlowPyrLK(before,after,current,None,**options)
        if forward is None or fs is None:return output
        backward,bs,_=cv2.calcOpticalFlowPyrLK(after,before,forward,None,**options)
        if backward is None or bs is None:return output
        xy=forward.reshape(-1,2);error=np.linalg.norm(backward.reshape(-1,2)-current.reshape(-1,2),axis=1)
        valid&=fs.ravel().astype(bool)&bs.ravel().astype(bool)&np.isfinite(error)&(error<=1.25)
        valid&=np.isfinite(xy).all(1)&(xy>=[0,0]).all(1)&(xy<[w,h]).all(1)
        current=forward
    shifts=current.reshape(-1,2)-original.reshape(-1,2)
    for j,first,end in groups:
        indices=np.flatnonzero(valid[first:end])+first
        if len(indices)<3:continue
        delta=np.median(shifts[indices],axis=0);coherent=np.linalg.norm(shifts[indices]-delta,axis=1)<=2.
        if coherent.sum()<3 or coherent.mean()<.7:continue
        delta=np.median(shifts[indices[coherent]],axis=0)
        predicted=points[j,:2]+delta/[w,h]
        if (predicted>=0).all() and (predicted<1).all():output[j]=predicted
    return output


def observe(frames, first, final, *, head_heights, tolerance_face_heights=.1):
    """Corroborate partial endpoints; raw face binding is not upgraded here.

    A limb endpoint needs its own two-direction pixel path plus a corroborated
    shoulder anchor. Requiring every arm/hip would discard real close-ups.
    Missing texture and a measured within-source model jump are separate.
    Neither result is itself a cross-source pose mismatch or a source veto.
    """
    frames=tuple(frames);first=np.asarray(first,float);final=np.asarray(final,float)
    heights=np.asarray(head_heights,float)
    if (not 2<=len(frames)<=13 or any(not isinstance(f,np.ndarray) or f.dtype!=np.uint8
            or f.ndim!=3 or f.shape[2]!=3 or min(f.shape[:2])<3 for f in frames)
            or any(f.shape!=frames[0].shape for f in frames)):
        raise ValueError('two to thirteen chronological actual native BGR8 rasters required')
    if (first.shape!=(8,3) or final.shape!=(8,3) or not np.isfinite(first).all() or not np.isfinite(final).all()
            or heights.shape!=(2,) or not np.isfinite(heights).all() or np.any(heights<=0)
            or not 0<tolerance_face_heights<=.3):
        raise ValueError('finite source-bound raw body8 and measured face heights required')
    forward=_paths(frames,first);backward=_paths(frames[::-1],final)
    observed=np.isfinite(forward).all(1)&np.isfinite(backward).all(1)
    actual=((first[:,2]>=.3)&(final[:,2]>=.3)&(first[:,:2]>=0).all(1)&(first[:,:2]<1).all(1)
        &(final[:,:2]>=0).all(1)&(final[:,:2]<1).all(1))
    error_a=np.linalg.norm((backward-first[:,:2])*[frames[0].shape[1]/frames[0].shape[0],1.],axis=1)/heights[0]
    error_b=np.linalg.norm((forward-final[:,:2])*[frames[0].shape[1]/frames[0].shape[0],1.],axis=1)/heights[1]
    worst=np.maximum(error_a,error_b);observed&=actual
    agrees=observed&(worst<=tolerance_face_heights)
    anchor=bool(agrees[:2].any())
    supported=agrees&anchor
    return PartPaths(supported,observed,observed&~agrees,np.where(observed,worst,np.nan),anchor)
