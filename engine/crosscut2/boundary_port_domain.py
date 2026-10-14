"""Finite exposed camera strips; pair proposals retain both endpoints."""
from dataclasses import dataclass
import hashlib
import math
import numpy as np
from .fixed_holds import _curve_at
from .shot_camera_paths import pair_proposals
from .played_partial_pose import features, matrix


@dataclass(frozen=True)
class BoundaryPort:
    source: int
    strip: np.ndarray
    observation: dict
    proposal_ids: tuple
    geometry_hash: str


def make(nodes, catalogues, baseline_paths, *, fps, check=lambda:None, maximum_pairs=8):
    if not 0<=maximum_pairs<=8:raise ValueError('at most8 GLOBAL paired proposals per node')
    N=len(nodes);S=len(catalogues);final=int(nodes[-1]);entries=[];exits=[];counts=[]
    dense=[[_curve_at(ticks,curve,np.arange(final)) for ticks,curve in paths] for paths in baseline_paths]
    def port(source,n,side,strip,ids):
        strip=np.asarray(strip,float).copy();strip.setflags(write=False)
        observation=catalogues[source].observation(n,side,strip)
        digest=hashlib.sha256(strip.tobytes()).hexdigest()
        return BoundaryPort(source,strip,observation,tuple(ids),digest)
    def add(items,value):
        for i,old in enumerate(items):
            if old.source==value.source and np.array_equal(old.strip,value.strip):
                items[i]=BoundaryPort(old.source,old.strip,old.observation,
                    old.proposal_ids+tuple(x for x in value.proposal_ids if x not in old.proposal_ids),old.geometry_hash)
                return
        items.append(value)
    for n,tick in enumerate(nodes):
        check();incoming=[];outgoing=[]
        for s in range(S):
            for family,curve in enumerate(dense[s]):
                before=curve[np.clip(tick+np.arange(-7,0),0,final-1)]
                after=curve[np.clip(tick+np.arange(0,7),0,final-1)]
                add(outgoing,port(s,n,0,before,(('static',family),)))
                add(incoming,port(s,n,1,after,(('static',family),)))
        proposals=[]
        if 0<n<N-1:
            native_left=[catalogues[s].observation(n,0,np.tile([1.,.5,.5],(7,1))) for s in range(S)]
            native_right=[catalogues[s].observation(n,1,np.tile([1.,.5,.5],(7,1))) for s in range(S)]
            pose=matrix(features(native_left,fps=fps),features(native_right,fps=fps),details=True)['pose_rate']
            for a in range(S):
                fa=native_left[a]['head']
                if fa[2]<.055 or fa[3]<.75:continue
                for b in range(S):
                    fb=native_right[b]['head']
                    if a==b or fb[2]<.055 or fb[3]<.75:continue
                    options=[]
                    for k,(ca,cb) in enumerate(pair_proposals(fa,fb,caps=(catalogues[a].bank['zoom_cap'],catalogues[b].bank['zoom_cap']))):
                        left=np.array([ca.zoom,ca.x,ca.y]);right=np.array([cb.zoom,cb.x,cb.y])
                        pa=.5+left[0]*(fa[:2]-left[1:]);pb=.5+right[0]*(fb[:2]-right[1:])
                        ha,hb=fa[2]*left[0],fb[2]*right[0]
                        error=max(abs(math.log(ha/hb))/math.log(1.5),np.linalg.norm((pa-pb)*[16/9,1.])/max(.005,(ha+hb)/2.))
                        score=error+pose[a,b]+.1*(math.log(left[0])+math.log(right[0]))
                        options.append((float(score),a,b,k,left,right))
                    if options:proposals.append(min(options,key=lambda x:x[:4]))
            # One representative per ordered SOURCE PAIR, not a source quota.
            proposals=sorted(proposals,key=lambda x:x[:4])[:maximum_pairs]
            for _,a,b,k,ca,cb in proposals:
                pair_id=('paired',n,a,b,k)
                add(outgoing,port(a,n,0,np.tile(ca,(7,1)),(pair_id,)))
                add(incoming,port(b,n,1,np.tile(cb,(7,1)),(pair_id,)))
        if max(len(incoming),len(outgoing))>4*S+maximum_pairs:raise AssertionError('port domain grew beyond global bound')
        entries.append(tuple(incoming));exits.append(tuple(outgoing));counts.append(len(proposals))
    return entries,exits,dense,dict(global_paired_proposal_limit_per_node=maximum_pairs,
        retained_pair_counts=counts,maximum_entry_ports=max(map(len,entries)),maximum_exit_ports=max(map(len,exits)),
        paired_endpoints_kept_together=True,roll_degrees=0.,post_DP_camera_fit=False,
        proposal_ranking_uses_reference_times_or_cut_count=False,
        proposal_rank_is_declared_engineering_not_preference_learning=True)
