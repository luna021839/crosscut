"""Experimental subject-locked, contemporary observations; no edit decisions.

Raw face/pose observations are independent of the cross-source matcher. A
local track is a multi-cue hypothesis, NOT named identity/anatomical proof.
Ambiguous/missing samples remain missing; an occluder never replaces a seed.
The caller supplies verified sparse rasters and one SHARED batch allowance.
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from time import monotonic
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from .partial_pose import face_proposals, face_guided_hypothesis
from .temporal_face_chain import _predict, _bounds, _iou


@dataclass
class ProbeBudget:
    face_limit: int = 512
    pose_limit: int = 256
    raster_limit: int = 4096
    recognition_limit: int = 4096
    seconds: float = 240.
    buffer_limit: int = 200 * 2**20
    check: object = lambda: None
    started: float = field(default_factory=monotonic)
    used: dict = field(default_factory=lambda: dict(face=0,pose=0,raster=0,recognition=0))
    peak_array_bytes: int = 0

    def __post_init__(self):
        if (any(type(v) is not int or v<0 for v in (self.face_limit,self.pose_limit,self.raster_limit,
                self.recognition_limit,self.buffer_limit)) or not np.isfinite(self.seconds) or self.seconds<=0):
            raise ValueError('finite shared observation limits required')

    def live(self):
        self.check()
        if monotonic()-self.started >= self.seconds:
            raise TimeoutError('shared subject observation allowance exhausted; no retry/fallback')

    def reserve(self, kind, count=1):
        self.live()
        if kind not in self.used or type(count) is not int or count < 0:
            raise ValueError('explicit nonnegative observation reservation required')
        if self.used[kind]+count > getattr(self,kind+'_limit'):
            raise TimeoutError('shared subject '+kind+' allowance exhausted before observation')
        self.used[kind] += count

    def arrays(self, size):
        self.live()
        if size > self.buffer_limit:
            raise MemoryError('subject observation array workspace exceeded')
        self.peak_array_bytes=max(self.peak_array_bytes,int(size))


@dataclass(frozen=True)
class SubjectWindow:
    source: int
    binding: str
    track: str
    ordinals: np.ndarray
    heads: np.ndarray
    vectors: np.ndarray
    body: np.ndarray
    support: np.ndarray
    associated: np.ndarray
    competitors: tuple
    native_cuts: tuple = ()

    def __post_init__(self):
        n=len(self.ordinals)
        if (type(self.source) is not int or self.source<0 or not self.binding or not self.track or n<3 or np.asarray(self.ordinals).dtype.kind not in 'iu'
                or np.any(np.diff(self.ordinals)!=1) or self.ordinals[0]<0
                or self.heads.shape!=(n,4) or self.vectors.shape!=(n,128)
                or self.body.shape!=(n,8,3) or self.support.shape!=(n,8)
                or self.associated.shape!=(n,) or len(self.competitors)!=n
                or self.support.dtype!=bool or self.associated.dtype!=bool
                or not all(np.isfinite(v).all() for v in (self.heads,self.vectors,self.body))
                or np.any(self.support & ~self.associated[:,None])):
            raise ValueError('one finite source-native subject window required')
        for name in ('ordinals','heads','vectors','body','support','associated'):
            value=np.asarray(getattr(self,name)).copy();value.setflags(write=False)
            object.__setattr__(self,name,value)
        others=[]
        for raw in self.competitors:
            value=np.asarray(raw,float).copy()
            if value.ndim!=2 or value.shape[1]!=4 or not np.isfinite(value).all():
                raise ValueError('contemporary competing faces required')
            value.setflags(write=False);others.append(value)
        object.__setattr__(self,'competitors',tuple(others))


def face_observation(pixels, detector, recognizer, budget):
    """All detected heads, including fourth/later; no cross-source choice."""
    budget.reserve('face');_,found=detector.detect(pixels)
    rows=[] if found is None else sorted((v for v in found if v[3]/pixels.shape[0]>=.02),
                                        key=lambda v:-float(v[3]))
    heads=[];vectors=[]
    for row in rows:
        heads.append([(row[0]+row[2]/2)/pixels.shape[1],(row[1]+row[3]/2)/pixels.shape[0],
                      row[3]/pixels.shape[0],row[-1]])
        value=np.zeros(128,np.float32)
        if row[3]>=20:
            budget.reserve('recognition')
            value=recognizer.feature(recognizer.alignCrop(pixels,row)).ravel().astype(np.float32)
            if value.shape!=(128,) or not np.isfinite(value).all():raise ValueError('SFace family changed')
            value=value/max(1e-12,float(np.linalg.norm(value)))
        vectors.append(value)
    return np.asarray(heads,float).reshape(-1,4),np.asarray(vectors,np.float32).reshape(-1,128)


def associate(frames, detections, vectors, ordinals, *, native_cuts=(), flow=None):
    """One-to-one source-local appearance + pixel/motion hypotheses.

    A flow failure alone does not discard a rotating face: contemporary face
    appearance plus bounded temporal motion can continue its MODEL hypothesis.
    Near-ties are not resolved by face order, nearest head or best cosine.
    """
    if (len(frames)!=len(detections) or len(frames)!=len(vectors) or len(frames)!=len(ordinals)
            or len(frames)<3 or np.any(np.diff(ordinals)!=1)):
        raise ValueError('complete chronological native observations required')
    tracks=[];shot=0
    for f,(heads,descriptors) in enumerate(zip(detections,vectors)):
        if descriptors.shape!=(len(heads),128):raise ValueError('one descriptor slot per actual face')
        cut=int(ordinals[f]) in native_cuts
        if cut:shot+=1
        active=[i for i,t in enumerate(tracks) if t['shot']==shot and 1<=f-t['last']<=3]
        cost=np.full((len(active),len(heads)),10.,float)
        for row,index in enumerate(active):
            t=tracks[index];last=t['last'];head=t['heads'][last];predicted=None
            if f-last==1:
                predicted,_=(_predict if flow is None else flow)(frames[last],frames[f],head)
            center=head[:2].copy()
            prior=[k for k in range(last) if t['indices'][k]>=0]
            if prior:center+=(head[:2]-t['heads'][prior[-1]][:2])*(f-last)/(last-prior[-1])
            if predicted is not None:center=(predicted[:2]+predicted[2:])/2.
            for col,h in enumerate(heads):
                distance=np.linalg.norm((center-h[:2])*[16/9,1.])/max(.02,h[2],head[2])
                scale=abs(np.log(max(.001,h[2])/max(.001,head[2])))
                seen=min(np.linalg.norm(t['prototype']),np.linalg.norm(descriptors[col]))>.5
                appearance=float(t['prototype']@descriptors[col]) if seen else None
                # Appearance alone cannot attach an arbitrary background face;
                # motion alone cannot attach a replacement intruder after loss.
                if distance>1.5 or scale>np.log(2.) or (appearance is not None and appearance<.15):continue
                if predicted is None and appearance is None:continue
                cost[row,col]=distance+.5*scale+(.75*(1.-appearance) if seen else .6)
        assigned={};used=set()
        if cost.size:
            rr,cc=linear_sum_assignment(cost)
            for row,col in zip(rr,cc):
                if cost[row,col]>=1.8:continue
                other_row=np.delete(cost[row],col);other_col=np.delete(cost[:,col],row)
                if (other_row.size and other_row.min()-cost[row,col]<.12) or (other_col.size and other_col.min()-cost[row,col]<.12):continue
                assigned[active[row]]=int(col);used.add(int(col))
        for index,col in assigned.items():
            t=tracks[index];t['heads'][f]=heads[col];t['vectors'][f]=descriptors[col];t['indices'][f]=col;t['last']=f
            if np.linalg.norm(descriptors[col])>.5:
                p=t['prototype']+descriptors[col];t['prototype']=p/max(1e-12,np.linalg.norm(p))
        for col in range(len(heads)):
            if col in used:continue
            n=len(frames);t=dict(heads=np.zeros((n,4)),vectors=np.zeros((n,128),np.float32),
                indices=np.full(n,-1,int),last=f,shot=shot,prototype=descriptors[col].copy())
            t['heads'][f]=heads[col];t['vectors'][f]=descriptors[col];t['indices'][f]=col;tracks.append(t)
    return tracks


def build(*, source, binding, frames, detections, vectors, ordinals, seed_native, seed_head,
          pose_natives, sensor, budget, native_cuts=(), flow=None):
    """Pin seed first, then observe its contemporary ROI; never reselect focal."""
    budget.live();ordinals=np.asarray(ordinals)
    budget.arrays(sum(v.nbytes for v in frames)+sum(v.nbytes for v in detections)+sum(v.nbytes for v in vectors))
    tracks=associate(frames,detections,vectors,ordinals,native_cuts=native_cuts,flow=flow)
    where=np.flatnonzero(ordinals==seed_native)
    if len(where)!=1:raise ValueError('seed must be an actually decoded native ordinal')
    at=int(where[0]);box=_bounds(seed_head)
    matches=[t for t in tracks if t['indices'][at]>=0 and box is not None and _iou(box,_bounds(t['heads'][at]))>=.7]
    if len(matches)!=1:return None,dict(status='NO_SUPPORTED_MATCH',reason='seed_redetection_ambiguous_or_missing')
    t=matches[0];n=len(ordinals);body=np.zeros((n,8,3),np.float32);support=np.zeros((n,8),bool);others=[]
    requests=set(map(int,pose_natives))
    if not requests.issubset(set(map(int,ordinals))):raise ValueError('pose request outside decoded window')
    for f,native in enumerate(ordinals):
        k=t['indices'][f];other=np.delete(detections[f],k,axis=0) if k>=0 else detections[f]
        others.append(other)
        if k<0 or int(native) not in requests:continue
        boxes=face_proposals(t['heads'][f])
        if not boxes:continue
        budget.reserve('pose');raw=sensor.describe(frames[f],bbox_xywh=boxes[0])
        partial=face_guided_hypothesis(raw,t['heads'][f],other)
        body[f]=partial.points;support[f]=partial.supported
    key=hashlib.sha256(str((source,int(seed_native))).encode()+np.asarray(seed_head,float).tobytes()).hexdigest()
    window=SubjectWindow(source,binding,key,ordinals,t['heads'],t['vectors'],body,support,
        t['indices']>=0,tuple(others),tuple(map(int,native_cuts)))
    return window,dict(status='observed_subject_hypothesis',associated_samples=int(window.associated.sum()),
        pose_samples=int(window.support.any(-1).sum()),named_identity_certified=False,
        anatomical_ownership_certified=False,target_reselected_after_camera=False)


def save(window, path, *, provenance):
    """Independent matcher-input artifact. Old face/body/raw133 caches untouched."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    offsets=np.r_[0,np.cumsum([len(v) for v in window.competitors])]
    with path.open('xb') as stream:
        np.savez_compressed(stream,source=window.source,binding=window.binding,track=window.track,
            ordinals=window.ordinals,heads=window.heads,vectors=window.vectors,body=window.body,
            support=window.support,associated=window.associated,native_cuts=window.native_cuts,
            others=np.concatenate(window.competitors),offsets=offsets,provenance=json.dumps(provenance,sort_keys=True))


def load(path, *, binding):
    with np.load(path,allow_pickle=False) as z:
        if str(z['binding'])!=binding:raise ValueError('subject source/native/model/raster binding changed')
        offsets=z['offsets'];others=z['others']
        return SubjectWindow(int(z['source']),binding,str(z['track']),z['ordinals'],z['heads'],z['vectors'],
            z['body'],z['support'],z['associated'],tuple(others[a:b] for a,b in zip(offsets[:-1],offsets[1:])),
            tuple(map(int,z['native_cuts'])))
