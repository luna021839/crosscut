"""Small experimental pairwise ranker; no default or fabricated preferences.

Only supplied, provenance-bearing same-event comparisons train weights. Split
membership is physical-group based. A fitted score is NOT a calibrated human
probability or evidence of professional video quality.
"""
import hashlib,json,math
import numpy as np
from .preference_features import FEATURE_NAMES,SCHEMA


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def fit(examples,*,steps=400,learning_rate=.1,regularization=.01):
    if (not 1<=steps<=2000 or not math.isfinite(learning_rate) or not 0<learning_rate<=1
            or not math.isfinite(regularization) or not 0<=regularization<=1 or not 0<=len(examples)<=96):
        raise ValueError('one bounded explicit preference fit required')
    pairs=[];train=[];seen={};kinds=set()
    for row in examples:
        group=row.get('physical_group');split=row.get('split');kind=row.get('supervision_kind')
        if not isinstance(group,str) or not group.strip() or split not in ('train','dev','held_out'):
            raise ValueError('explicit physical-group split is required')
        if group in seen and seen[group]!=split:raise ValueError('physical group leaks across preference splits')
        seen[group]=split
        if kind not in ('human_av_preference','reference_editor_choice_weak_imitation'):
            raise ValueError('unknown or advisory-still supervision cannot become human preference')
        if kind=='human_av_preference' and row.get('self_reported_normal_av') is not True:
            raise ValueError('human AV label needs its actual self-report provenance')
        kinds.add(kind)
        values=np.asarray(row['features'],float);chosen=row['chosen']
        if (values.ndim!=2 or values.shape[1]!=len(FEATURE_NAMES) or len(values)<2
                or not np.isfinite(values).all() or type(chosen) is not int or not 0<=chosen<len(values)):
            raise ValueError('finite same-event options and a real choice required')
        if split=='train':train.append(values)
        pairs.append((row,values,chosen))
    if not train:raise ValueError('no training comparisons; do not invent labels')
    raw=np.vstack(train);center=raw.mean(0);scale=raw.std(0);scale=np.where(scale>1e-8,scale,1.)
    delta=[]
    for row,values,chosen in pairs:
        if row['split']=='train':
            delta.extend((values[chosen]-values[j])/scale for j in range(len(values)) if j!=chosen)
    delta=np.asarray(delta);weights=np.zeros(len(FEATURE_NAMES))
    for _ in range(steps):
        z=delta@weights;wrong=1./(1.+np.exp(np.clip(z,-50,50)))
        weights-=learning_rate*(-(delta*wrong[:,None]).mean(0)+regularization*weights)
    report={}
    for split in ('train','dev','held_out'):
        good=total=0
        for row,values,chosen in pairs:
            if row['split']!=split:continue
            scores=(values-center)/scale@weights
            for j in range(len(values)):
                if j!=chosen:good+=float(scores[chosen]>scores[j])+.5*float(scores[chosen]==scores[j]);total+=1
        report[split]=dict(pairwise_accuracy=None if not total else good/total,pairs=total,
            physical_groups=sorted(g for g,s in seen.items() if s==split))
    artifact=dict(contract='experimental_same_event_pairwise_ranker_v1',feature_schema=SCHEMA,
        feature_names=list(FEATURE_NAMES),center=center.tolist(),scale=scale.tolist(),weights=weights.tolist(),
        training_examples_sha256=digest(examples),supervision_kinds=sorted(kinds),report=report,
        training_config=dict(steps=steps,learning_rate=learning_rate,regularization=regularization),
        probabilistic_calibration_claimed=False,professional_quality_approved=False,normal_app_activation_approved=False)
    artifact['content_sha256']=digest(artifact)
    return artifact


class Model:
    def __init__(self,artifact):
        payload=dict(artifact);expected=payload.pop('content_sha256',None)
        if (expected!=digest(payload) or payload.get('contract')!='experimental_same_event_pairwise_ranker_v1'
                or payload.get('feature_schema')!=SCHEMA or payload.get('feature_names')!=list(FEATURE_NAMES)
                or not payload.get('supervision_kinds') or any(k not in ('human_av_preference','reference_editor_choice_weak_imitation')
                    for k in payload.get('supervision_kinds',[]))
                or payload.get('report',{}).get('train',{}).get('pairs',0)<=0):
            raise ValueError('fitted preference model/schema binding differs; legacy weights cannot be reused')
        self.artifact=artifact
        self.center=np.asarray(payload['center'],float);self.scale=np.asarray(payload['scale'],float);self.weights=np.asarray(payload['weights'],float)
        if (any(v.shape!=(len(FEATURE_NAMES),) or not np.isfinite(v).all() for v in (self.center,self.scale,self.weights))
                or np.any(self.scale<=0)):
            raise ValueError('finite fitted preference parameters required')
    def scores(self,features):
        values=np.asarray(features,float)
        if values.shape[-1:]!=(len(FEATURE_NAMES),) or not np.isfinite(values).all():raise ValueError('bound finite preference features required')
        return ((values-self.center)/self.scale)@self.weights
    def losses(self,features,*,state_sources):
        scores=self.scores(features);sources=np.asarray(state_sources)
        if scores.shape!=(len(sources),len(sources)):raise ValueError('same fixed-state H/K domain required')
        allowed=(sources[:,None]!=sources[None,:])|np.eye(len(sources),dtype=bool)
        best=float(scores[allowed].max())
        # One event-wide max, not a source-count sum/softmax. A repeated option
        # cannot create an extra reward. H and K share the same fitted scale.
        return best-np.diag(scores),best-scores
