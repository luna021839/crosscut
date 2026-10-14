"""Three distinct input states: corroborated, contradicted, unmeasured.

An untextured patch is not a vanished arm. Retain original model coordinates
as uncertain hypotheses instead of erasing them into a zero skeleton. The
neutral .5 weight is a declared uncalibrated convention, not a probability.
Only genuinely corroborated contrary motion is a confirmed contradiction.
"""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from . import part_support_bank,temporal_parts
from .grid_perception import file_identity


def reliabilities(support,disagrees,*,scope):
    support=np.asarray(support);disagrees=np.asarray(disagrees);scope=np.asarray(scope)
    if (support.shape!=disagrees.shape or support.ndim!=3 or support.shape[1:]!=(3,8)
            or support.dtype!=bool or disagrees.dtype!=bool or scope.shape!=(len(support),) or scope.dtype!=bool
            or (support&disagrees).any()):
        raise ValueError('separate measured support/contradiction and physical window scope required')
    positive=np.stack((support[:,0],support[:,0]|support[:,1],support[:,1]|support[:,2],support[:,2]),axis=1)
    negative=np.stack((disagrees[:,0],disagrees[:,0]&disagrees[:,1],disagrees[:,1]&disagrees[:,2],disagrees[:,2]),axis=1)
    phase=np.where(positive,1.,np.where(negative,0.,.5))
    motion=np.where(support[:,[0,2]],1.,np.where(disagrees[:,[0,2]],0.,.5))
    return np.where(scope[:,None,None],phase,1.).astype(np.float32),np.where(scope[:,None,None],motion,1.).astype(np.float32)


def retain_model_hypotheses(*,rows,states,sources,cache):
    """Consume ONLY the exact raw producer's completed source snapshots."""
    implementation=part_support_bank.cache_implementation()
    physical={};found_sources=set()
    for path in sorted(Path(cache).glob('*.npz')):
        # ExFAT macOS AppleDouble ._ sidecars are not serialized observations.
        # Never turn a failed np.load into permission to enable pickle.
        if re.fullmatch(r'[a-f0-9]{64}\.npz',path.name) is None:continue
        with np.load(path,allow_pickle=False) as saved:
            binding=json.loads(str(saved['binding']))
            if binding.get('implementation')!=implementation:continue
            indices=[s for s,p in enumerate(sources) if list(file_identity(p))==binding.get('source')]
            if len(indices)!=1:continue
            s=indices[0]
            if s in found_sources:raise ValueError('ambiguous source-native part snapshots; do not pick latest')
            found_sources.add(s)
            for k,v in zip(saved['keys'],saved['disagrees']):physical[s,str(k)]=v.copy()
    result=[]
    for state,row in zip(states,rows):
        d=dict(row);support=d['part_pair_support'];scope=d['pixel_part_scope'];raw=d['raw_body_hypotheses']
        disagreement=np.zeros_like(support)
        for n in np.flatnonzero(scope):
            for pair,(first,final) in enumerate(((0,1),(1,2),(2,3))):
                a,b=map(int,d['pose_native_indices'][n,[first,final]])
                heights=d['heads'][n,[first//2,final//2],2]
                key=part_support_bank.interval_key(a,b,raw[n,first],raw[n,final],heights)
                if (state.source,key) in physical:disagreement[n,pair]=physical[state.source,key]
        d['body']=raw.copy()
        d['body_reliability'],d['motion_reliability']=reliabilities(support,disagreement,scope=scope)
        d['part_pair_model_disagreement']=disagreement
        d['uncorroborated_model_geometry_is_not_verified_anatomy']=True
        result.append(d)
    return result
