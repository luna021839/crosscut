"""Ordered ragged target observations; no fixed person-count truncation.

Formation is an unordered observed set, while time and each person's typed
joints remain ordered. FEATURE retains its pre-camera target even when its
face is missing. Native/displayed coordinates and conflicts remain separate.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .sequence_context import PRIMITIVES,project as scalar_project

JOINTS=('left_shoulder','right_shoulder','left_elbow','right_elbow',
    'left_wrist','right_wrist','left_hip','right_hip')
HEAD=('native_x','native_y','native_height','screen_x','screen_y','screen_height',
    'visible_fraction','native_roi','confidence')
BODY=('native_x','native_y','screen_x','screen_y','confidence',
    'model_only','pixel_supported','contradicted','ownership_ambiguous')
NAMES=(*PRIMITIVES,'intended_target_slots','observed_face_count')
PERSON_NAMES=(*(f'head.{v}' for v in HEAD),*(f'{joint}.{v}' for joint in JOINTS for v in BODY))
SCHEMA='ordered_pre_camera_ragged_target_geometry_v1'
SCHEMA_SHA256=hashlib.sha256(json.dumps(dict(schema=SCHEMA,scene=NAMES,person=PERSON_NAMES,
    order='every_actual_output_frame_typed_joints_unordered_observed_people',
    missing='independent_scene_person_cue_masks_and_persistent_conflict',
    target='pre_camera_intent_not_crop_argmax_no_person_count_cap'),sort_keys=True).encode()).hexdigest()
ROLE=np.array([n.startswith('role_') for n in NAMES])
TARGET=np.array([not n.startswith(('additional_','measured_')) for n in NAMES])


@dataclass(frozen=True)
class OrderedEvidence:
    frames: np.ndarray
    values: np.ndarray
    observed: np.ndarray
    scopes: tuple
    conflicted: np.ndarray
    person_splits: np.ndarray
    people: np.ndarray
    people_observed: np.ndarray
    people_conflicted: np.ndarray

    def __post_init__(self):
        t=np.asarray(self.frames);x=np.asarray(self.values);m=np.asarray(self.observed);bad=np.asarray(self.conflicted)
        splits=np.asarray(self.person_splits);p=np.asarray(self.people);pm=np.asarray(self.people_observed);pb=np.asarray(self.people_conflicted)
        if (t.ndim!=1 or t.dtype.kind not in 'iu' or t.min(initial=0)<0 or np.any(np.diff(t)<=0)
                or x.shape!=(len(t),len(NAMES)) or m.shape!=x.shape or m.dtype!=bool
                or bad.shape!=x.shape or bad.dtype!=bool or np.any(m&bad) or not np.isfinite(x).all()
                or len(self.scopes)!=len(t) or splits.shape!=(len(t)+1,) or splits.dtype.kind not in 'iu'
                or splits[0]!=0 or np.any(np.diff(splits)<0) or splits[-1]!=len(p)
                or p.shape!=(len(p),len(PERSON_NAMES)) or pm.shape!=p.shape or pm.dtype!=bool
                or pb.shape!=p.shape or pb.dtype!=bool or np.any(pm&pb) or not np.isfinite(p).all()):
            raise ValueError('ordered actual ticks and ragged target-bound cues with explicit masks required')
        for name,value in (('frames',t.astype(np.int64)),('values',x),('observed',m),('conflicted',bad),
                ('person_splits',splits.astype(np.int64)),('people',p),('people_observed',pm),('people_conflicted',pb)):
            value=value.copy();value.setflags(write=False);object.__setattr__(self,name,value)
        object.__setattr__(self,'scopes',tuple(self.scopes))

    def take(self,rows,*,frames=None):
        rows=np.asarray(rows,np.int64);chunks=[self.persons(int(i)) for i in rows]
        return assemble(self.frames[rows] if frames is None else frames,self.values[rows],self.observed[rows],
            tuple(self.scopes[int(i)] for i in rows),self.conflicted[rows],chunks)

    def persons(self,row):
        a,b=self.person_splits[row:row+2]
        return self.people[a:b],self.people_observed[a:b],self.people_conflicted[a:b]


def assemble(frames,values,known,scopes,bad,chunks):
    count=np.array([len(p) for p,_,_ in chunks],np.int64)
    empty=np.zeros((0,len(PERSON_NAMES)))
    arrays=[np.concatenate([v[k] for v in chunks]) if chunks else empty.astype(float if k==0 else bool) for k in range(3)]
    return OrderedEvidence(np.asarray(frames),values,known,tuple(scopes),bad,np.r_[0,np.cumsum(count)],*arrays)


def project(scene,cameras):
    base=scalar_project(scene,cameras);view=scene.project(cameras);T,H=scene.head_observed.shape
    values=np.zeros((T,len(NAMES)));known=np.zeros(values.shape,bool)
    values[:,:len(PRIMITIVES)]=base.values;known[:,:len(PRIMITIVES)]=base.observed
    values[:,-2]=H;known[:,-2]=scene.role!=0
    values[:,-1]=scene.head_observed.sum(-1);known[:,-1]=scene.head_observed.any(-1)
    person=np.zeros((T,H,len(PERSON_NAMES)));pm=np.zeros(person.shape,bool)
    person[:,:,:3]=scene.heads[...,:3];person[:,:,3:5]=view['centers'];person[:,:,5]=view['heights']
    person[:,:,6]=view['visible_fraction'];person[:,:,7]=scene.roi_readability;person[:,:,8]=scene.heads[...,3]
    pm[:,:,:len(HEAD)]=scene.head_observed[...,None];pm[:,:,7]=scene.roi_measured
    body=person[:,:,len(HEAD):].reshape(T,H,8,len(BODY));bm=pm[:,:,len(HEAD):].reshape(T,H,8,len(BODY))
    body[:,:,:,:2]=scene.body[...,:2];body[:,:,:,2:4]=view['body_centers'];body[:,:,:,4]=scene.body[...,2]
    body[:,:,:,5]=scene.body_evidence==1;body[:,:,:,6]=scene.body_evidence==2
    body[:,:,:,7]=scene.body_evidence==3;body[:,:,:,8]=scene.body_ambiguity
    bm[:]=view['body_observed'][...,None]
    bm[:,:,:,5:]=((scene.body_evidence!=0)|scene.body_ambiguity)[...,None]
    person=person.reshape(-1,len(PERSON_NAMES));pm=pm.reshape(person.shape)
    return OrderedEvidence(scene.frames,np.where(known,values,0.),known,base.target_scopes,np.zeros(known.shape,bool),
        np.arange(T+1,dtype=np.int64)*H,np.where(pm,person,0.),pm,np.zeros(pm.shape,bool))


def _same_cues(values,masks,bads):
    present=np.logical_or.reduce(masks)
    lo=np.minimum.reduce([np.where(m,x,np.inf) for x,m in zip(values,masks)])
    hi=np.maximum.reduce([np.where(m,x,-np.inf) for x,m in zip(values,masks)])
    bad=np.logical_or.reduce(bads)|(present&(lo!=hi));known=present&~bad
    return np.where(known,lo,0.),known,bad


def merge(parts):
    if not parts:
        shape=(0,len(NAMES));return assemble(np.zeros(0,int),np.zeros(shape),np.zeros(shape,bool),(),np.zeros(shape,bool),[])
    records=sorted((int(t),p,i) for p,part in enumerate(parts) for i,t in enumerate(part.frames))
    if not records:return merge([])
    ticks=[];values=[];known=[];bad=[];scopes=[];people=[];start=0
    while start<len(records):
        stop=start+1
        while stop<len(records) and records[stop][0]==records[start][0]:stop+=1
        rows=[(parts[p],i) for _,p,i in records[start:stop]]
        x,m,b=_same_cues([p.values[i] for p,i in rows],[p.observed[i] for p,i in rows],[p.conflicted[i] for p,i in rows])
        owners={p.scopes[i] for p,i in rows if p.observed[i,TARGET].any()}
        chunks=[p.persons(i) for p,i in rows]
        if len(owners)>1 or len({len(v[0]) for v in chunks})>1:
            b[TARGET]=True;m[TARGET]=False;x[TARGET]=0.
            raw=np.concatenate([v[0] for v in chunks]);pm=np.zeros(raw.shape,bool);pb=np.ones(raw.shape,bool)
        else:raw,pm,pb=_same_cues([v[0] for v in chunks],[v[1] for v in chunks],[v[2] for v in chunks])
        if b[ROLE].any():b[ROLE]=True;m[ROLE]=False;x[ROLE]=0.
        ticks.append(records[start][0]);values.append(x);known.append(m);bad.append(b)
        scopes.append(next(iter(owners)) if len(owners)==1 else None);people.append((raw,pm,pb));start=stop
    return assemble(np.asarray(ticks,np.int64),np.asarray(values),np.asarray(known),tuple(scopes),np.asarray(bad),people)


def lookup(data,frames,native):
    T=len(frames);shape=(T,len(NAMES));values=np.zeros(shape);known=np.zeros(shape,bool);bad=np.zeros(shape,bool)
    scopes=[None]*T;empty=np.zeros((0,len(PERSON_NAMES)));chunks=[(empty,empty.astype(bool),empty.astype(bool)) for _ in frames]
    at=np.searchsorted(data.frames,native)
    if len(data.frames):
        safe=np.minimum(at,len(data.frames)-1);hit=(at<len(data.frames))&(data.frames[safe]==native)
        for row,k in zip(np.flatnonzero(hit),at[hit]):
            values[row]=data.values[k];known[row]=data.observed[k];bad[row]=data.conflicted[k]
            scopes[row]=data.scopes[k];chunks[row]=data.persons(int(k))
    return assemble(np.asarray(frames),values,known,tuple(scopes),bad,chunks)


def recamera(evidence,cameras,scalar):
    cameras=np.asarray(cameras,float);T=len(evidence.frames)
    if cameras.shape!=(T,3) or scalar.values.shape!=(T,len(PRIMITIVES)):
        raise ValueError('same exact ordered ticks and played cameras required')
    values=evidence.values.copy();known=evidence.observed.copy();bad=evidence.conflicted.copy()
    values[:,:len(PRIMITIVES)]=scalar.values;known[:,:len(PRIMITIVES)]=scalar.observed;bad[:,:len(PRIMITIVES)]=scalar.conflicted
    person=evidence.people.copy();rows=np.repeat(np.arange(T),np.diff(evidence.person_splits));cam=cameras[rows]
    person[:,3:5]=.5+cam[:,:1]*(person[:,:2]-cam[:,1:]);person[:,5]=cam[:,0]*person[:,2]
    half=person[:,5,None]*[.45*9/16,.5]
    extent=np.maximum(0.,np.minimum(1.,person[:,3:5]+half)-np.maximum(0.,person[:,3:5]-half))
    person[:,6]=np.prod(extent/np.maximum(1e-12,2*half),axis=-1)
    body=person[:,len(HEAD):].reshape(-1,8,len(BODY))
    body[:,:,2:4]=.5+cam[:,None,:1]*(body[:,:,:2]-cam[:,None,1:])
    return OrderedEvidence(evidence.frames,values,known,evidence.scopes,bad,evidence.person_splits,
        np.where(evidence.people_observed,person,0.),evidence.people_observed,evidence.people_conflicted)
