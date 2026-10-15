"""Experimental local pixel/detection chain for one displayed endpoint face.

No names, appearance descriptors, NN, source decoding, edit admission or camera
fitting occur here. A detector/flow failure is an unmeasured association, never
permission to select the nearest face. Thresholds are engineering settings,
not calibrated ownership probabilities. Contemporary competing faces travel
with the result for the downstream raw-pose nose check.
"""
from dataclasses import dataclass
import math
from types import MappingProxyType
import cv2
import numpy as np
from .partial_pose import face_rectangle,face_guided_hypothesis,PartialPose


@dataclass(frozen=True)
class FaceChain:
    accepted: bool
    reason: str
    head: np.ndarray
    others: np.ndarray
    detection_indices: tuple
    steps: tuple
    anatomical_ownership_certified: bool = False
    named_identity_certified: bool = False


@dataclass(frozen=True)
class BoundWindowFaces:
    """Explicit source/clock/endpoint-face binding for an opt-in pose consumer."""
    source_file_identity: tuple
    native_indices: tuple
    endpoint_heads: np.ndarray
    flank_chains: dict

    def __post_init__(self):
        identity=tuple(self.source_file_identity);native=tuple(self.native_indices)
        heads=np.asarray(self.endpoint_heads,float).copy();chains=dict(self.flank_chains)
        if (len(identity)!=4 or any(type(v) is not int for v in identity)
                or len(native)!=4 or any(type(v) is not int or v<0 for v in native)
                or heads.shape!=(2,4) or not np.isfinite(heads).all()
                or not chains or any(type(k) is not int or k not in (0,3)
                    or not isinstance(v,FaceChain) for k,v in chains.items())):
            raise ValueError('explicit physical source/window/head binding required')
        for phase,chain in chains.items():
            if not chain.accepted:continue
            start,end=native[1 if phase==0 else 2],native[phase]
            direction=-1 if phase==0 else 1
            expected=list(range(start,end+direction,direction))
            actual=[s['from_native'] for s in chain.steps]+[chain.steps[-1]['to_native']] if chain.steps else []
            if (not 2<=len(expected)<=13 or actual!=expected
                    or any(s['to_native']!=expected[i+1] for i,s in enumerate(chain.steps))
                    or len(chain.detection_indices)!=len(expected)):
                raise ValueError('accepted chain must expose its exact bound native clock')
        heads.setflags(write=False)
        object.__setattr__(self,'source_file_identity',identity);object.__setattr__(self,'native_indices',native)
        object.__setattr__(self,'endpoint_heads',heads);object.__setattr__(self,'flank_chains',MappingProxyType(chains))


def _bounds(head):
    box=face_rectangle(head)
    return None if box is None else np.r_[box[:2],box[:2]+box[2:]]


def _iou(a,b):
    intersection=np.maximum(0.,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(intersection/max(1e-12,np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection))


def _predict(before,after,head):
    """One native step, reset features on its observed face, not on a body."""
    gray=[cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in (before,after)]
    h,w=gray[0].shape;bounds=_bounds(head)
    lo=np.floor(bounds[:2]*[w,h]).astype(int);hi=np.ceil(bounds[2:]*[w,h]).astype(int)
    mask=np.zeros((h,w),np.uint8);mask[max(0,lo[1]):min(h,hi[1]),max(0,lo[0]):min(w,hi[0])]=255
    points=cv2.goodFeaturesToTrack(gray[0],maxCorners=32,qualityLevel=.01,minDistance=3,mask=mask,blockSize=3)
    info=dict(initial_points=0 if points is None else len(points),flow_points=0,inliers=0)
    if points is None or len(points)<6:return None,info
    options=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    forward,fs,_=cv2.calcOpticalFlowPyrLK(gray[0],gray[1],points,None,**options)
    if forward is None or fs is None:return None,info
    backward,bs,_=cv2.calcOpticalFlowPyrLK(gray[1],gray[0],forward,None,**options)
    if backward is None or bs is None:return None,info
    xy=forward[:,0];error=np.linalg.norm(backward[:,0]-points[:,0],axis=1)
    valid=fs.ravel().astype(bool)&bs.ravel().astype(bool)&np.isfinite(error)&(error<=1.25)
    valid&=np.isfinite(xy).all(1)&(xy>=[0,0]).all(1)&(xy<[w,h]).all(1)
    info['flow_points']=int(valid.sum())
    if valid.sum()<6:return None,info
    affine,inliers=cv2.estimateAffinePartial2D(points[valid,0],xy[valid],method=cv2.RANSAC,
        ransacReprojThreshold=2.,maxIters=1000,confidence=.99,refineIters=10)
    if affine is None or inliers is None or not np.isfinite(affine).all():return None,info
    info['inliers']=int(inliers.sum());info['maximum_FB_pixel_error']=float(error[valid].max())
    if info['inliers']<4 or info['inliers']/int(valid.sum())<.6:return None,info
    x0,y0,x1,y1=bounds*[w,h,w,h]
    corners=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])@affine[:,:2].T+affine[:,2]
    predicted=np.r_[corners.min(0)/[w,h],corners.max(0)/[w,h]]
    if not np.isfinite(predicted).all() or np.any(predicted[2:]<=predicted[:2]):return None,info
    return predicted,info


def bind(frames,detections,*,ordinals,endpoint_head,endpoint_at,native_cuts=(),_flow=None):
    """Consume a complete <=13-frame source-native window, in either direction.

    ``_flow`` is a test seam for deterministic positive/adversarial geometry;
    real callers leave it unset. No endpoint-only fast-motion magnitude gate
    exists. Every individual link needs independent contemporary detections.
    """
    frames=tuple(frames);detections=tuple(np.asarray(v,float) for v in detections)
    ordinals=np.asarray(ordinals);head=np.asarray(endpoint_head,float)
    if (not 2<=len(frames)<=13 or len(detections)!=len(frames)
            or ordinals.shape!=(len(frames),) or ordinals.dtype.kind not in 'iu'
            or np.any(np.diff(ordinals)!=1) or ordinals[0]<0
            or type(endpoint_at) is not int or endpoint_at not in (0,len(frames)-1)
            or head.shape!=(4,) or not np.isfinite(head).all() or _bounds(head) is None
            or any(v.ndim!=2 or v.shape[1]!=4 or not np.isfinite(v).all() for v in detections)
            or any(not isinstance(f,np.ndarray) or f.dtype!=np.uint8 or f.ndim!=3 or f.shape[2]!=3
                or min(f.shape[:2])<8 or f.shape!=frames[0].shape for f in frames)):
        raise ValueError('one finite endpoint and complete chronological native face window required')
    order=list(range(len(frames)))
    if endpoint_at:order.reverse()
    steps=[];indices=[]
    def reject(reason):
        return FaceChain(False,reason,np.zeros(4),np.zeros((0,4)),tuple(indices),tuple(steps))
    cuts=np.asarray(native_cuts)
    if cuts.ndim!=1 or cuts.dtype.kind not in 'iuf' or not np.isfinite(cuts).all():
        raise ValueError('finite native cut ordinals required')
    if np.any((cuts>ordinals[0])&(cuts<=ordinals[-1])):return reject('native_camera_cut_inside_window')
    matches=[]
    for index,candidate in enumerate(detections[endpoint_at]):
        box=_bounds(candidate)
        if box is not None and _iou(_bounds(head),box)>=.7 and abs(math.log(candidate[2]/head[2]))<=math.log(1.15):
            matches.append(index)
    if len(matches)!=1:return reject('endpoint_redetection_missing_or_ambiguous')
    index=matches[0];indices.append(index);current=detections[endpoint_at][index]
    for first,last in zip(order,order[1:]):
        predicted,info=(_predict if _flow is None else _flow)(frames[first],frames[last],current)
        step=dict(from_native=int(ordinals[first]),to_native=int(ordinals[last]),**info);steps.append(step)
        if predicted is None:return reject('temporal_face_pixel_path_unmeasured')
        predicted=np.asarray(predicted,float)
        if predicted.shape!=(4,) or not np.isfinite(predicted).all() or np.any(predicted[2:]<=predicted[:2]):
            raise ValueError('flow prediction must be a finite positive face rectangle')
        centre=(predicted[:2]+predicted[2:])/2.;height=predicted[3]-predicted[1];plausible=[]
        for k,candidate in enumerate(detections[last]):
            if _bounds(candidate) is None:continue
            error=float(np.linalg.norm((centre-candidate[:2])*[frames[0].shape[1]/frames[0].shape[0],1.])/candidate[2])
            scale=abs(math.log(height/candidate[2]))
            if error<=.35 and scale<=math.log(1.25):plausible.append((error+scale,k,error,scale))
        step['plausible_detections']=len(plausible)
        if len(plausible)!=1:return reject('temporal_face_detection_missing_or_ambiguous')
        _,index,error,scale=plausible[0]
        step.update(selected_index=index,centre_error_face_heights=error,log_scale_error=scale)
        indices.append(index);current=detections[last][index]
    selected=current.copy();others=np.delete(detections[order[-1]],indices[-1],axis=0)
    selected.setflags(write=False);others.setflags(write=False)
    return FaceChain(True,'contemporary_face_chain_model_only',selected,others,tuple(indices),tuple(steps))


def reanchor_existing_raw(evidence,chain,*,raw_roi_xywh):
    """Opt-in contemporary nose check; NEVER move/fuse/refit model coordinates.

    The caller must bind ``evidence`` to this source/native ordinal and its
    original unchanged raw ROI. Containment only licenses trying the existing
    raw hypothesis, not a claim that an unchanged ROI is an ideal crop or that
    limbs belong to this face. Failed chains cannot reuse the stale face.
    """
    roi=np.asarray(raw_roi_xywh,float)
    if (not isinstance(chain,FaceChain) or roi.shape!=(4,) or not np.isfinite(roi).all()
            or (roi[:2]<0).any() or (roi[2:]<=0).any() or (roi[:2]+roi[2:]>1.+1e-9).any()):
        raise ValueError('qualified local chain and exact original positive raw ROI required')
    missing=lambda why:PartialPose(np.zeros((8,3),np.float32),np.zeros(8,bool),why)
    if not chain.accepted:return missing('temporal_face_chain_unmeasured:'+chain.reason)
    face=face_rectangle(chain.head)
    if face is None or (face[:2]<roi[:2]).any() or (face[:2]+face[2:]>roi[:2]+roi[2:]).any():
        return missing('contemporary_face_outside_existing_raw_roi_requires_new_observation')
    return face_guided_hypothesis(evidence,chain.head,chain.others)
