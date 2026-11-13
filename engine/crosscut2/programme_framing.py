"""Source-family-deduplicated framing evidence from ORIGINAL camera views.

Not singer labels, an imposed close-up quota, or a chosen-programme statistic.
The weighted median original focal size is a soft shot-role requirement; making
an already adequate face larger earns nothing. Original formation consensus
also asks for the actually seen people, rather than cropping every scene tight.
"""
import numpy as np
from .physical_display_observations import focal_metadata


def evidence(banks,*,source_cohort=None):
    shape=banks[0]['heads'].shape[:2];families=[b.get('consensus_family',('source',s)) for s,b in enumerate(banks)]
    active=tuple(range(len(banks))) if source_cohort is None else tuple(source_cohort)
    if (not active or any(type(s) is not int or not 0<=s<len(banks) for s in active)
            or len(set(active))!=len(active)):
        raise ValueError('distinct physical framing source indices required')
    groups=list(dict.fromkeys(families[s] for s in active));height=np.zeros((len(groups),)+shape)
    people=np.zeros_like(height);weight=np.zeros_like(height);formation=np.zeros_like(height,bool)
    for s in active:
        bank,family=banks[s],families[s]
        if bank['heads'].shape[:2]!=shape:raise ValueError('one common original programme clock required')
        heads,index,ambiguity,scene=focal_metadata(bank['heads'],np.tile([1.,.5,.5],shape+(1,)))
        quality=np.full(shape,.5)
        if 'native_readability' in bank:
            for n,side in zip(*np.nonzero(index>=0)):
                h=int(index[n,side])
                if bank['native_readability_valid'][n,side,h]:
                    contrast,edge,clipped,size=bank['native_readability'][n,side,h]
                    quality[n,side]=min(1.,size/120.)*min(1.,edge/.5)*min(1.,contrast/.075)*(1.-clipped)
        w=np.sqrt(quality)*np.maximum(0.,1.-ambiguity)
        w=np.where(index>=0,w,0.);g=groups.index(family);better=w>weight[g]
        raw=bank['heads'];seen=(raw[...,2]>.019)&(raw[...,3]>=.75)
        count=(seen&(raw[...,2]>=.5*np.maximum(.019,heads[...,2,None]))).sum(-1)
        height[g]=np.where(better,heads[...,2],height[g]);people[g]=np.where(better,count,people[g])
        formation[g]=np.where(better,scene==2,formation[g]);weight[g]=np.maximum(weight[g],w)
    def median(values):
        order=np.argsort(values,axis=0);v=np.take_along_axis(values,order,axis=0)
        w=np.take_along_axis(weight,order,axis=0);at=np.argmax(np.cumsum(w,axis=0)>=weight.sum(0)[None]/2.,axis=0)
        return np.take_along_axis(v,at[None],axis=0)[0]
    support=(weight>0).sum(0);confidence=np.where(support>=2,weight.sum(0)/max(1,len(groups)),0.)
    majority=(weight*formation).sum(0)>weight.sum(0)/2.
    result=dict(height=median(height),people=median(people),formation=majority,confidence=confidence)
    return result,dict(original_source_families=len(groups),original_source_indices=list(active),
        physical_source_domain_count=len(banks),framing_supported_nodes=int((confidence>0).sum()),
        mean_framing_confidence=float(confidence.mean()),native_views_only=True,
        duplicate_family_extra_votes=False,learned_singer_or_named_identity=False,
        already_adequate_closeup_earns_no_extra_zoom_reward=True,mandatory_wide_or_closeup_count=False)
