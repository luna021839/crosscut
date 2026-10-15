"""Fixed-time target/camera evidence for a new contextual preference schema.

No aesthetic weights, recovery debt, cut reward or source identifier is used.
Window endpoints are global output ticks, independent of authored/native cuts
and how an identical film was split into holds. Missing samples stay missing.
This is not the old same-event ranker or a certified background-motion model.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .scene_targets import TargetScene, FEATURE, FORMATION, UNKNOWN

PRIMITIVES = ('role_feature', 'role_formation', 'role_unknown',
    'target_face_observed_fraction', 'target_visible_fraction',
    'feature_center_x', 'feature_center_y', 'feature_height',
    'formation_center_x', 'formation_center_y', 'formation_height',
    'native_ROI_readability', 'known_head_crop_fraction',
    'body_observed_fraction', 'known_body_crop_fraction', 'body_pixel_supported_fraction',
    'additional_log_zoom', 'additional_center_x', 'additional_center_y',
    'measured_full_frame_change', 'measured_background_flow')
SUMMARIES = ('observed_mean', 'observed_minimum', 'observed_maximum',
    'observed_frame_fraction', 'latest_observed_value', 'latest_age_fraction')
FEATURE_NAMES = tuple(p+'.'+s for p in PRIMITIVES for s in SUMMARIES)
SCHEMA = 'fixed_time_target_context_v1'
SCHEMA_SHA256 = hashlib.sha256(json.dumps(dict(schema=SCHEMA,
    primitives=PRIMITIVES, summaries=SUMMARIES,
    merge='persistent_per_cue_conflict_and_pre_camera_target_scope_v1'), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Evidence:
    frames: np.ndarray
    values: np.ndarray
    observed: np.ndarray
    target_scopes: tuple | None = None
    conflicted: np.ndarray | None = None

    def __post_init__(self):
        frames=np.asarray(self.frames);values=np.asarray(self.values);observed=np.asarray(self.observed)
        shape=(len(frames),len(PRIMITIVES)) if frames.ndim==1 else None
        if (frames.ndim!=1 or frames.dtype.kind not in 'iu' or frames.min(initial=0)<0
                or np.any(np.diff(frames)<=0) or values.shape!=shape
                or observed.shape!=shape or observed.dtype!=bool
                or values.dtype.kind not in 'iuf' or not np.isfinite(values).all()):
            raise ValueError('unique chronological finite contextual evidence and explicit masks required')
        if int(frames.max(initial=0))>np.iinfo(np.int64).max:
            raise ValueError('context output frame exceeds signed clock domain')
        frames=frames.astype(np.int64)
        conflicts=np.zeros(shape,bool) if self.conflicted is None else np.asarray(self.conflicted)
        scopes=(None,)*len(frames) if self.target_scopes is None else tuple(self.target_scopes)
        if (conflicts.shape!=shape or conflicts.dtype!=bool or np.any(conflicts&observed)
                or len(scopes)!=len(frames) or any(v is not None and (not isinstance(v,str) or not v) for v in scopes)):
            raise ValueError('target association scopes and explicit conflicting evidence masks required')
        object.__setattr__(self,'target_scopes',scopes)
        for name,value in (('frames',frames),('values',values),('observed',observed),('conflicted',conflicts)):
            value=value.copy();value.setflags(write=False);object.__setattr__(self,name,value)


def project(scene, cameras):
    """One existing target window -> numeric primitives with separate masks.

    Geometry ranges in later summaries are descriptive output context, NOT a
    causal camera velocity or one actor's trajectory across unrelated scenes.
    Background flow remains unmeasured; zero is not a successful observation.
    """
    if not isinstance(scene,TargetScene):raise ValueError('bound pre-camera TargetScene required')
    cameras=np.asarray(cameras,float)
    if cameras.shape!=(len(scene.frames),3):raise ValueError('one actual played camera window required')
    view=scene.project(cameras);t=len(scene.frames)
    values=np.zeros((t,len(PRIMITIVES)));known=np.zeros(values.shape,bool)
    def put(name,value,mask=True):
        k=PRIMITIVES.index(name);values[:,k]=value;known[:,k]=mask
    for name,kind in (('role_feature',FEATURE),('role_formation',FORMATION),('role_unknown',UNKNOWN)):
        put(name,float(scene.role==kind))
    face=view['head_observed'];count=face.sum(-1);face_known=count>0
    def mean(value,mask):
        return np.sum(np.where(mask,value,0.),axis=-1)/np.maximum(1,mask.sum(-1))
    put('target_face_observed_fraction',face.mean(-1))
    put('target_visible_fraction',mean(view['visible_fraction'],face),face_known)
    for label,kind in (('feature',FEATURE),('formation',FORMATION)):
        mask=face_known&(scene.role==kind)
        put(label+'_center_x',mean(view['centers'][...,0],face),mask)
        put(label+'_center_y',mean(view['centers'][...,1],face),mask)
        put(label+'_height',mean(view['heights'],face),mask)
    roi=view['roi_measured'];quality=view['roi_readability']*np.minimum(1.,scene.heads[...,2]/.12)
    put('native_ROI_readability',mean(quality,roi),roi.any(-1))
    put('known_head_crop_fraction',mean(view['known_crop_out'],face),face_known)
    body=view['body_observed'];body_count=body.sum((1,2))
    put('body_observed_fraction',body.mean((1,2)))
    put('known_body_crop_fraction',view['body_known_crop_out'].sum((1,2))/np.maximum(1,body_count),body_count>0)
    put('body_pixel_supported_fraction',((scene.body_evidence==2)&body).sum((1,2))/np.maximum(1,body_count),body_count>0)
    put('additional_log_zoom',np.log(cameras[:,0]))
    put('additional_center_x',cameras[:,1]-.5);put('additional_center_y',cameras[:,2]-.5)
    # Association provenance is NOT part of the numeric feature vector. A
    # target from another anchor/track must not win simply because one of its
    # scalar measurements coincides with this one's value at the same frame.
    scope=json.dumps([scene.binding,scene.track,scene.role,scene.heads.shape[1]],separators=(',',':'))
    return Evidence(scene.frames,values,known,(scope,)*t)


def merge(parts):
    """Merge identical physical samples once; conflicting cues stay UNKNOWN.

    Disjoint sensor channels may complement each other, but duplicate windows
    cannot increase their weight. Do not use this to combine different output
    alternatives: each proposed movie needs its own separate evidence stream.
    """
    if not parts:return Evidence(np.zeros(0,int),np.zeros((0,len(PRIMITIVES))),np.zeros((0,len(PRIMITIVES)),bool))
    if any(not isinstance(p,Evidence) for p in parts):raise ValueError('bound context evidence required')
    frames=np.concatenate([p.frames for p in parts]);values=np.concatenate([p.values for p in parts])
    masks=np.concatenate([p.observed for p in parts]);unique=np.unique(frames)
    conflicts=np.concatenate([p.conflicted for p in parts]);scopes=sum((p.target_scopes for p in parts),())
    result=np.zeros((len(unique),len(PRIMITIVES)));known=np.zeros(result.shape,bool)
    contradicted=np.zeros(result.shape,bool);result_scopes=[]
    target_fields=np.array([not name.startswith(('additional_','measured_')) for name in PRIMITIVES])
    role_fields=np.array([name.startswith('role_') for name in PRIMITIVES])
    for i,frame in enumerate(unique):
        rows=frames==frame;seen=masks[rows];raw=values[rows]
        low=np.where(seen,raw,np.inf).min(0);high=np.where(seen,raw,-np.inf).max(0)
        bad=conflicts[rows].any(0)|(seen.any(0)&(low!=high))
        owners={scopes[k] for k in np.flatnonzero(rows) if masks[k,target_fields].any()}
        # Unknown is not agreement. Contradictions persist through nested
        # merges; otherwise re-merging a convenient third window restores a
        # previously rejected cue and makes result depend on chunking order.
        if len(owners)>1:bad|=target_fields
        if bad[role_fields].any():
            # A one-hot role is atomic; independent ROI/body sensor deficits
            # do not destroy another usable cue for the SAME bound target.
            bad|=role_fields
        valid=seen.any(0)&(low==high)&~bad
        result[i]=np.where(valid,low,0.);known[i]=valid
        contradicted[i]=bad
        result_scopes.append(next(iter(owners)) if len(owners)==1 else None)
    return Evidence(unique,result,known,tuple(result_scopes),contradicted)


def windows(evidence, *, final_frame, fps, interval_frames=6, context_seconds=4.):
    """Global fixed-time vectors; four seconds is context, not a shot rule.

    Coverage is actual observed output-frame count/context length, not guessed
    continuous exposure from nearest sparse samples. Offsets/origin type are
    not fitted features. There is no arbitrary per-cut evaluation/payout.
    """
    if (not isinstance(evidence,Evidence) or type(final_frame) is not int or final_frame<1
            or type(interval_frames) is not int or interval_frames<1
            or not np.isfinite(fps) or fps<=0 or not np.isfinite(context_seconds) or context_seconds<=0
            or evidence.frames.max(initial=-1)>=final_frame):
        raise ValueError('actual film domain and fixed finite context required')
    context=max(1,int(np.ceil(context_seconds*fps)))
    ends=np.r_[np.arange(interval_frames,final_frame,interval_frames),final_frame]
    starts=np.r_[0,ends[:-1]];vectors=[]
    for end in ends:
        first=max(0,int(end)-context);a,b=np.searchsorted(evidence.frames,[first,end])
        raw=evidence.values[a:b];mask=evidence.observed[a:b];ticks=evidence.frames[a:b]
        count=mask.sum(0);valid=count>0
        means=np.where(mask,raw,0.).sum(0)/np.maximum(1,count)
        lows=np.where(valid,np.where(mask,raw,np.inf).min(0,initial=np.inf),0.)
        highs=np.where(valid,np.where(mask,raw,-np.inf).max(0,initial=-np.inf),0.)
        latest=np.where(mask,ticks[:,None],-1).max(0,initial=-1)
        last=np.zeros(len(PRIMITIVES))
        for k in np.flatnonzero(valid):last[k]=raw[int(np.searchsorted(ticks,latest[k])),k]
        age=np.where(valid,(int(end)-1-latest)/(end-first),1.)
        vectors.append(np.stack((means,lows,highs,count/(end-first),last,age),axis=-1).ravel())
    contract=dict(schema_sha256=SCHEMA_SHA256,interval_frames=interval_frames,context_seconds=float(context_seconds))
    window_hash=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    return dict(schema=SCHEMA,schema_sha256=SCHEMA_SHA256,window_contract_sha256=window_hash,
        window_config=contract,feature_names=FEATURE_NAMES,
        output_fps=float(fps),
        interval_first=starts,interval_final=ends,interval_seconds=(ends-starts)/fps,
        context_frames=context,features=np.asarray(vectors),
        aesthetic_model_fitted=False,cut_origin_or_count_used=False,
        missing_observations_imputed=False,normal_speed_AV_performed=False)


def integrate(window_evidence, model):
    """Consume an explicit NEW schema model; never reuse same-event weights."""
    if (window_evidence.get('schema')!=SCHEMA or window_evidence.get('schema_sha256')!=SCHEMA_SHA256
            or getattr(model,'sequence_feature_schema',None)!=SCHEMA_SHA256
            or getattr(model,'sequence_window_contract',None)!=window_evidence.get('window_contract_sha256')
            or not window_evidence.get('window_contract_sha256')):
        raise ValueError('explicit fixed-time sequence preference model/schema required')
    expected=hashlib.sha256(json.dumps(window_evidence['window_config'],sort_keys=True).encode()).hexdigest()
    features=np.asarray(window_evidence['features'],float)
    durations=np.asarray(window_evidence['interval_seconds'],float)
    first,last=np.asarray(window_evidence['interval_first']),np.asarray(window_evidence['interval_final'])
    fps=window_evidence['output_fps']
    if (expected!=window_evidence['window_contract_sha256'] or tuple(window_evidence['feature_names'])!=FEATURE_NAMES
            or features.shape!=(len(durations),len(FEATURE_NAMES)) or not np.isfinite(features).all()
            or durations.ndim!=1 or not len(durations) or not np.isfinite(durations).all() or np.any(durations<=0)
            or first.shape!=durations.shape or last.shape!=durations.shape or first.dtype.kind not in 'iu'
            or last.dtype.kind not in 'iu' or first[0]!=0 or np.any(first[1:]!=last[:-1]) or np.any(last<=first)
            or not np.isfinite(fps) or fps<=0 or not np.array_equal(durations,(last-first)/fps)):
        raise ValueError('complete exact fixed-time feature/window binding required')
    score=np.asarray(model.costs(features),float)
    if score.shape!=durations.shape or not np.isfinite(score).all():
        raise ValueError('one finite contextual cost per fixed output interval required')
    return float(score@durations)
