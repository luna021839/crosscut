"""Small experimental pairwise ranker; no default or fabricated preferences.

Only supplied, provenance-bearing same-event comparisons train weights. Split
membership is physical-group based. A fitted score is NOT a calibrated human
probability or evidence of professional video quality.
"""
import hashlib,json,math
from collections import Counter
import numpy as np
from .preference_features import FEATURE_NAMES,SCHEMA


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def partial_choice(scores,compatible):
    """Marginal choice NLL; no invented ranking inside the compatible set."""
    scores=np.asarray(scores,float);compatible=np.asarray(compatible,int)
    def softmax(v):
        e=np.exp(v-v.max());return e/e.sum(),float(v.max()+np.log(e.sum()))
    all_prob,all_log=softmax(scores);yes_prob,yes_log=softmax(scores[compatible])
    gradient=all_prob.copy();gradient[compatible]-=yes_prob
    return all_log-yes_log,gradient


def _options(row):
    values=np.asarray(row['features'],float);kind=row['supervision_kind']
    if (values.ndim!=2 or values.shape[1]!=len(FEATURE_NAMES) or len(values)<2
            or not np.isfinite(values).all()):raise ValueError('finite same-event options and a real choice required')
    if kind=='human_av_preference':
        chosen=row.get('chosen')
        if type(chosen) is not int or not 0<=chosen<len(values):raise ValueError('a real strict AV choice is required')
        return values,[chosen]
    if 'chosen' in row:raise ValueError('weak source choices cannot invent a strict H/K winner')
    compatible=row.get('compatible_options');identities=row.get('option_ids')
    if (not isinstance(compatible,list) or not compatible or any(type(i) is not int or not 0<=i<len(values) for i in compatible)
            or len(set(compatible))!=len(compatible) or not isinstance(identities,list) or len(identities)!=len(values)
            or any(not isinstance(i,str) or not i for i in identities)):
        raise ValueError('weak labels require compatible physical options and explicit identities')
    seen={};rows=[];yes=[]
    for i,key in enumerate(identities):
        accepted=i in compatible
        if key in seen:
            old=seen[key]
            if not np.array_equal(rows[old],values[i]) or (old in yes)!=accepted:
                raise ValueError('duplicated physical option has inconsistent observations or compatibility')
            continue
        seen[key]=len(rows);rows.append(values[i])
        if accepted:yes.append(len(rows)-1)
    if not 0<len(yes)<len(rows):raise ValueError('weak label has no contrast outside its compatible set')
    return np.asarray(rows),yes


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
        values,compatible=_options(row)
        if split=='train':train.append(values)
        pairs.append((row,values,compatible))
    if not train:raise ValueError('no training comparisons; do not invent labels')
    counts=Counter(row['physical_group'] for row,_,_ in pairs if row['split']=='train')
    raw=np.vstack(train);rw=np.concatenate([np.full(len(values),1/(counts[row['physical_group']]*len(values)))
        for row,values,_ in pairs if row['split']=='train']);rw/=rw.sum()
    center=(raw*rw[:,None]).sum(0);scale=np.sqrt(((raw-center)**2*rw[:,None]).sum(0));scale=np.where(scale>1e-8,scale,1.)
    normalized=[(row,(values-center)/scale,compatible) for row,values,compatible in pairs if row['split']=='train']
    weights=np.zeros(len(FEATURE_NAMES))
    for _ in range(steps):
        gradient=np.zeros_like(weights)
        for row,values,compatible in normalized:
            if row['supervision_kind']=='reference_editor_choice_weak_imitation':
                _,g=partial_choice(values@weights,compatible);g=values.T@g
            else:
                chosen=compatible[0];delta=values[chosen]-np.delete(values,chosen,axis=0)
                wrong=1./(1.+np.exp(np.clip(delta@weights,-50,50)));g=-(delta*wrong[:,None]).mean(0)
            gradient+=g/(len(counts)*counts[row['physical_group']])
        weights-=learning_rate*(gradient+regularization*weights)
    report={}
    for split in ('train','dev','held_out'):
        good=total=0;weak_good=weak_total=0;weak_losses=[];events=0
        for row,values,compatible in pairs:
            if row['split']!=split:continue
            events+=1
            scores=(values-center)/scale@weights
            if row['supervision_kind']=='reference_editor_choice_weak_imitation':
                best=np.flatnonzero(scores==scores.max());weak_good+=sum(i in compatible for i in best)/len(best);weak_total+=1
                weak_losses.append(partial_choice(scores,compatible)[0]);continue
            chosen=compatible[0]
            for j in range(len(values)):
                if j!=chosen:good+=float(scores[chosen]>scores[j])+.5*float(scores[chosen]==scores[j]);total+=1
        report[split]=dict(pairwise_accuracy=None if not total else good/total,pairs=total,
            supervised_events=events,weak_compatible_top1=None if not weak_total else weak_good/weak_total,
            weak_events=weak_total,weak_marginal_nll=None if not weak_losses else float(np.mean(weak_losses)),
            physical_groups=sorted(g for g,s in seen.items() if s==split))
    artifact=dict(contract='experimental_same_event_preference_ranker_v2',feature_schema=SCHEMA,
        feature_names=list(FEATURE_NAMES),center=center.tolist(),scale=scale.tolist(),weights=weights.tolist(),
        training_examples_sha256=digest(examples),supervision_kinds=sorted(kinds),report=report,
        training_config=dict(steps=steps,learning_rate=learning_rate,regularization=regularization),
        objective='group_balanced_strict_pairwise_and_weak_marginal_compatible_choice',
        probabilistic_calibration_claimed=False,professional_quality_approved=False,normal_app_activation_approved=False)
    artifact['content_sha256']=digest(artifact)
    return artifact


class Model:
    def __init__(self,artifact):
        payload=dict(artifact);expected=payload.pop('content_sha256',None)
        if (expected!=digest(payload) or payload.get('contract')!='experimental_same_event_preference_ranker_v2'
                or payload.get('feature_schema')!=SCHEMA or payload.get('feature_names')!=list(FEATURE_NAMES)
                or not payload.get('supervision_kinds') or any(k not in ('human_av_preference','reference_editor_choice_weak_imitation')
                    for k in payload.get('supervision_kinds',[]))
                or payload.get('report',{}).get('train',{}).get('supervised_events',0)<=0):
            raise ValueError('fitted preference model/schema binding differs; legacy weights cannot be reused')
        self.artifact=artifact
        self.center=np.asarray(payload['center'],float);self.scale=np.asarray(payload['scale'],float);self.weights=np.asarray(payload['weights'],float)
        if (any(v.shape!=(len(FEATURE_NAMES),) or not np.isfinite(v).all() for v in (self.center,self.scale,self.weights))
                or np.any(self.scale<=0)):
            raise ValueError('finite fitted preference parameters required')
    def scores(self,features):
        values=np.asarray(features)
        if values.dtype.kind not in 'biuf':
            raise ValueError('bound finite numeric preference features required')
        if values.shape[-1:]!=(len(FEATURE_NAMES),) or not np.isfinite(values).all():raise ValueError('bound finite preference features required')
        # The affine normalization is scalar-linear. Normalize59coefficients,
        # not everyQxQx59 observation: the latter creates multiple80MiB copies
        # per416-state event and approaches the real selection deadline.
        # Explicit float64 accumulation keeps the existing numeric precision
        # without casting/copying the whole float32 input tensor.
        coefficient=self.weights/self.scale
        offset=-float(np.einsum('f,f->',self.center,coefficient,optimize=False,dtype=np.float64))
        return np.einsum('...f,f->...',values,coefficient,optimize=False,dtype=np.float64)+offset
    def score_bounds(self,lower,upper):
        lower,upper=np.asarray(lower,float),np.asarray(upper,float)
        if (lower.shape!=upper.shape or lower.shape[-1:]!=(len(FEATURE_NAMES),)
                or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower>upper)):
            raise ValueError('finite explicit raw-feature uncertainty intervals required')
        coefficient=self.weights/self.scale
        offset=-float(np.einsum('f,f->',self.center,coefficient,optimize=False,dtype=np.float64))
        low=offset+(np.where(coefficient>=0,lower,upper)*coefficient).sum(-1)
        high=offset+(np.where(coefficient>=0,upper,lower)*coefficient).sum(-1)
        return low,high
    def losses(self,features,*,state_sources):
        scores=self.scores(features);sources=np.asarray(state_sources)
        if scores.shape!=(len(sources),len(sources)):raise ValueError('same fixed-state H/K domain required')
        allowed=(sources[:,None]!=sources[None,:])|np.eye(len(sources),dtype=bool)
        best=float(scores[allowed].max())
        # One event-wide max, not a source-count sum/softmax. A repeated option
        # cannot create an extra reward. H and K share the same fitted scale.
        return best-np.diag(scores),best-scores


def regret_bounds(lower,upper,*,physical_ids):
    """Finite worst regret with correlated copies and self-regret exactly0."""
    lower,upper=np.asarray(lower,float),np.asarray(upper,float)
    if (lower.ndim!=1 or upper.shape!=lower.shape or len(physical_ids)!=len(lower)
            or not len(lower) or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower>upper)):
        raise ValueError('finite physical-option score bounds required')
    unique={};mapping=[];lo=[];hi=[]
    for i,key in enumerate(physical_ids):
        if key not in unique:unique[key]=len(lo);lo.append(lower[i]);hi.append(upper[i])
        j=unique[key];mapping.append(j)
        if lo[j]!=lower[i] or hi[j]!=upper[i]:raise ValueError('same physical choice cannot have independent uncertainty copies')
    lo,hi=np.asarray(lo),np.asarray(hi);different=~np.eye(len(lo),dtype=bool)
    a=np.where(different,lo[None,:]-hi[:,None],0.).max(1)
    b=np.where(different,hi[None,:]-lo[:,None],0.).max(1)
    return np.maximum(0.,a)[mapping],np.maximum(0.,b)[mapping]
