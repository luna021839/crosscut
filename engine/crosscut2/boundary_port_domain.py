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


def make(nodes, catalogues, baseline_paths, *, fps, check=lambda:None, maximum_pairs=8,opportunities=None):
    if not 0<=maximum_pairs<=8:raise ValueError('at most8 GLOBAL paired proposals per node')
    N=len(nodes);S=len(catalogues);final=int(nodes[-1]);entries=[];exits=[];counts=[]
    dense=[[_curve_at(ticks,curve,np.arange(final)) for ticks,curve in paths] for paths in baseline_paths]
    def port(source,n,side,strip,ids,*,target_head_index=None):
        strip=np.asarray(strip,float).copy();strip.setflags(write=False)
        observation=catalogues[source].observation(n,side,strip,target_head_index=target_head_index)
        digest=hashlib.sha256(strip.tobytes()).hexdigest()
        return BoundaryPort(source,strip,observation,tuple(ids),digest)
    def add(items,value):
        for i,old in enumerate(items):
            if (old.source==value.source and np.array_equal(old.strip,value.strip)
                    and old.observation['head_index']==value.observation['head_index']
                    and old.proposal_ids==value.proposal_ids):
                observation=value.observation if value.observation['target_pinned_before_camera'] else old.observation
                items[i]=BoundaryPort(old.source,old.strip,observation,
                    old.proposal_ids+tuple(x for x in value.proposal_ids if x not in old.proposal_ids),old.geometry_hash)
                return
        items.append(value)
    rejected_reassignments=0;actor_pairs_considered=0;non_native_focal_pairs=0
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
            # Retrieve a physical actor pair BEFORE proposing its cameras.
            # Looking at only the native largest face loses a valid same-
            # member connection whenever the broadcasters frame differently.
            # All three face/body-bound hypotheses remain available here;
            # a native fourth focal is retained with its honestly missing body.
            def observed(side):
                values=[];sources=[];native_indices=[]
                for s,cat in enumerate(catalogues):
                    native=cat.observation(n,side,np.tile([1.,.5,.5],(7,1)))
                    native_indices.append(native['head_index'])
                    indices=set(range(min(3,len(cat.bank['heads'][n,side]))))
                    if native['head_index']>=0:indices.add(native['head_index'])
                    for h in sorted(indices):
                        head=cat.bank['heads'][n,side,h]
                        if head[2]<.055 or head[3]<.75:continue
                        values.append(cat.observation(n,side,np.tile([1.,.5,.5],(7,1)),target_head_index=h))
                        sources.append(s)
                return values,sources,native_indices
            native_left,left_sources,left_native=observed(0)
            native_right,right_sources,right_native=observed(1)
            if not native_left or not native_right:
                entries.append(tuple(incoming));exits.append(tuple(outgoing));counts.append(0)
                continue
            native_pair=matrix(features(native_left,fps=fps),features(native_right,fps=fps),details=True)
            if opportunities is not None:
                opportunities.observe(n,native_left,native_right,native_pair,left_sources,right_sources)
            pose=native_pair['pose_rate'];appearance=native_pair['appearance_rate']
            candidates={}
            for ai,left_obs in enumerate(native_left):
                a=left_sources[ai];fa=left_obs['head'];ha_index=int(left_obs['head_index'])
                for bi,right_obs in enumerate(native_right):
                    b=right_sources[bi];fb=right_obs['head'];hb_index=int(right_obs['head_index'])
                    if a==b:continue
                    actor_pairs_considered+=1
                    quality=min(left_obs['displayed_native_quality_proxy'],right_obs['displayed_native_quality_proxy'])
                    for k,(ca,cb) in enumerate(pair_proposals(fa,fb,caps=(catalogues[a].bank['zoom_cap'],catalogues[b].bank['zoom_cap']))):
                        left=np.array([ca.zoom,ca.x,ca.y]);right=np.array([cb.zoom,cb.x,cb.y])
                        pa=.5+left[0]*(fa[:2]-left[1:]);pb=.5+right[0]*(fb[:2]-right[1:])
                        ha,hb=fa[2]*left[0],fb[2]*right[0]
                        error=max(abs(math.log(ha/hb))/math.log(1.5),np.linalg.norm((pa-pb)*[16/9,1.])/max(.005,(ha+hb)/2.))
                        score=error+pose[ai,bi]+appearance[ai,bi]+1.-quality+.1*(math.log(left[0])+math.log(right[0]))
                        if opportunities is not None:
                            score-=opportunities.density(n,left_obs,right_obs,a,b,
                                native_pair['matched_action_change_density'][ai,bi])
                        candidates.setdefault((a,b),[]).append((float(score),a,b,ha_index,hb_index,k,left,right))
            for a in range(S):
                for b in range(S):
                    for option in sorted(candidates.get((a,b),()),key=lambda x:x[:6]):
                        _,_,_,ha_index,hb_index,_,left,right=option
                        left_obs=catalogues[a].observation(n,0,np.tile(left,(7,1)),target_head_index=ha_index)
                        right_obs=catalogues[b].observation(n,1,np.tile(right,(7,1)),target_head_index=hb_index)
                        if not left_obs['target_is_displayed_focal'] or not right_obs['target_is_displayed_focal']:
                            # This CAMERA option lost the proposed focal; do
                            # not quietly score another actor's pose or veto
                            # the source. Other legal camera options remain.
                            rejected_reassignments+=1
                            continue
                        proposals.append(option)
                        break
            # One representative per ordered SOURCE PAIR, not a source quota.
            proposals=sorted(proposals,key=lambda x:x[:6])[:maximum_pairs]
            for _,a,b,ha,hb,k,ca,cb in proposals:
                pair_id=('paired',n,a,b,ha,hb,k)
                non_native_focal_pairs+=int(ha!=left_native[a] or hb!=right_native[b])
                add(outgoing,port(a,n,0,np.tile(ca,(7,1)),(pair_id,),
                    target_head_index=ha))
                add(incoming,port(b,n,1,np.tile(cb,(7,1)),(pair_id,),
                    target_head_index=hb))
        if max(len(incoming),len(outgoing))>4*S+maximum_pairs:raise AssertionError('port domain grew beyond global bound')
        entries.append(tuple(incoming));exits.append(tuple(outgoing));counts.append(len(proposals))
    return entries,exits,dense,dict(global_paired_proposal_limit_per_node=maximum_pairs,
        retained_pair_counts=counts,maximum_entry_ports=max(map(len,entries)),maximum_exit_ports=max(map(len,exits)),
        paired_endpoints_kept_together=True,roll_degrees=0.,post_DP_camera_fit=False,
        paired_subject_pinned_before_camera=True,rejected_camera_focal_reassignments=rejected_reassignments,
        physical_actor_pairs_compared_before_camera=actor_pairs_considered,
        retained_pairs_using_non_native_focal=non_native_focal_pairs,
        raw_fixed_opportunity_used_before_global_pruning=opportunities is not None,
        proposal_ranking_uses_reference_times_or_cut_count=False,
        proposal_rank_is_declared_engineering_not_preference_learning=True)
