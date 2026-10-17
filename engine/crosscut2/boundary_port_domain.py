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
    intent: object = None


def make(nodes, catalogues, baseline_paths, *, fps, check=lambda:None, maximum_pairs=8,opportunities=None,roles=None,
         coalesce_static_ports=False,recycle_static_capacity=False,articulated_actions=False,source_cohort=None,
         retain_pair_cameras=False):
    if not 0<=maximum_pairs<=8:raise ValueError('at most8 GLOBAL paired proposals per node')
    coalesce_static_ports=coalesce_static_ports or recycle_static_capacity
    N=len(nodes);S=len(catalogues);final=int(nodes[-1]);entries=[];exits=[];counts=[]
    active=tuple(range(S)) if source_cohort is None else tuple(source_cohort)
    if (not active or any(type(s) is not int or not 0<=s<S for s in active)
            or len(set(active))!=len(active)):
        raise ValueError('distinct actual source cohort indices required')
    old_port_limit=4*len(active)+maximum_pairs
    # Explicit finite representation expansion, not a claim of the former
    # fixed-work budget: each primary actor pair has <=3 common-size views
    # and one original-view option. Keep all old ports, then let interval
    # costs compare these physical endpoints rather than a local proxy.
    port_limit=old_port_limit+3*maximum_pairs*bool(retain_pair_cameras)
    camera_variants=0
    extra_count=0;pair_capacities=[];natural_added=0;original_retained=0
    dense=[[_curve_at(ticks,curve,np.arange(final)) for ticks,curve in paths] for paths in baseline_paths]
    intent_cache={}
    def port(source,n,side,strip,ids,*,target_head_index=None):
        from .playback_intent import before_camera
        key=(source,n,side,target_head_index)
        # Static ports use the source's pre-camera native-role schedule. Do
        # not build unused contextual scenes in the ordinary scalar E4 path
        # or impose its stronger sample-clock contract on unrelated callers.
        if target_head_index is not None and key not in intent_cache:
            intent_cache[key]=before_camera(catalogues[source],source,n,side,target_head_index)
        intent=intent_cache.get(key)
        strip=np.asarray(strip,float).copy();strip.setflags(write=False)
        observation=catalogues[source].observation(n,side,strip,target_head_index=target_head_index)
        digest=hashlib.sha256(strip.tobytes()).hexdigest()
        return BoundaryPort(source,strip,observation,tuple(ids),digest,intent)
    static_aliases=0
    def add(items,value):
        nonlocal static_aliases
        for i,old in enumerate(items):
            same_static=(coalesce_static_ports and all(x[0]=='static' for x in old.proposal_ids+value.proposal_ids))
            if (old.source==value.source and np.array_equal(old.strip,value.strip)
                    and old.observation['head_index']==value.observation['head_index']
                    and (old.proposal_ids==value.proposal_ids or same_static)):
                observation=value.observation if value.observation['target_pinned_before_camera'] else old.observation
                intent=value.intent if value.observation['target_pinned_before_camera'] else old.intent
                items[i]=BoundaryPort(old.source,old.strip,observation,
                    old.proposal_ids+tuple(x for x in value.proposal_ids if x not in old.proposal_ids),old.geometry_hash,intent)
                static_aliases+=int(same_static and old.proposal_ids!=value.proposal_ids)
                return
        items.append(value)
    rejected_reassignments=0;actor_pairs_considered=0;non_native_focal_pairs=0
    for n,tick in enumerate(nodes):
        check();incoming=[];outgoing=[]
        for s in active:
            for family,curve in enumerate(dense[s]):
                before=curve[np.clip(tick+np.arange(-7,0),0,final-1)]
                after=curve[np.clip(tick+np.arange(0,7),0,final-1)]
                add(outgoing,port(s,n,0,before,(('static',family),)))
                add(incoming,port(s,n,1,after,(('static',family),)))
        proposals=[]
        capacity=min(old_port_limit-len(incoming),old_port_limit-len(outgoing))
        if not maximum_pairs:capacity=0
        pair_capacities.append((capacity if recycle_static_capacity else maximum_pairs)
            +3*maximum_pairs*bool(retain_pair_cameras))
        if 0<n<N-1:
            # Retrieve a physical actor pair BEFORE proposing its cameras.
            # Looking at only the native largest face loses a valid same-
            # member connection whenever the broadcasters frame differently.
            # All three face/body-bound hypotheses remain available here;
            # a native fourth focal is retained with its honestly missing body.
            def observed(side):
                values=[];sources=[];native_indices=[-1]*S
                for s,cat in enumerate(catalogues):
                    if s not in active:continue
                    native=cat.observation(n,side,np.tile([1.,.5,.5],(7,1)))
                    native_indices[s]=native['head_index']
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
            if articulated_actions:
                from .played_interval_utility import articulated_seam_evidence
                native_pair=articulated_seam_evidence(native_pair,native_left,native_right)
            if opportunities is not None:
                opportunities.observe(n,native_left,native_right,native_pair,left_sources,right_sources)
            role_strength=np.zeros(native_pair['pose_rate'].shape)
            if roles is not None:
                from .editorial_roles import apply
                role_strength=roles.at(tick,np.asarray([o['recurrent_member_hint'] for o in native_left])[:,None],
                    np.asarray([o['recurrent_member_hint'] for o in native_right])[None,:])
                native_pair=apply(native_pair,role_strength)
            pose=native_pair['pose_rate'];appearance=native_pair['appearance_rate']
            candidates={}
            for ai,left_obs in enumerate(native_left):
                a=left_sources[ai];fa=left_obs['head'];ha_index=int(left_obs['head_index'])
                for bi,right_obs in enumerate(native_right):
                    b=right_sources[bi];fb=right_obs['head'];hb_index=int(right_obs['head_index'])
                    if a==b:continue
                    actor_pairs_considered+=1
                    quality=min(left_obs['displayed_native_quality_proxy'],right_obs['displayed_native_quality_proxy'])
                    options=list(pair_proposals(fa,fb,caps=(catalogues[a].bank['zoom_cap'],catalogues[b].bank['zoom_cap'])))
                    if role_strength[ai,bi]>0:
                        # A genuine role change must not be represented ONLY
                        # by common-face-size zooms. Native framing is still
                        # subject-pinned and competes in the same global8.
                        from .paired_geometry import Camera
                        options.append((Camera(1.,.5,.5),Camera(1.,.5,.5)))
                    # A naturally matching MOVING connection must not need
                    # an artificial common-size crop to become pair-bound.
                    # Keep its real discrepancy and the same finite global
                    # domain. Ordinary static ports still cannot borrow a
                    # different outgoing proposal's action benefit.
                    if (articulated_actions and native_pair['measured_face'][ai,bi]
                            and native_pair['face_rate'][ai,bi]<=1.
                            and native_pair['matched_action_change_density'][ai,bi]>0
                            and not any(all(c.zoom==1. and c.x==.5 and c.y==.5 for c in pair)
                                        for pair in options)):
                        from .paired_geometry import Camera
                        options.append((Camera(1.,.5,.5),Camera(1.,.5,.5)))
                        natural_added+=1
                    for k,(ca,cb) in enumerate(options):
                        left=np.array([ca.zoom,ca.x,ca.y]);right=np.array([cb.zoom,cb.x,cb.y])
                        pa=.5+left[0]*(fa[:2]-left[1:]);pb=.5+right[0]*(fb[:2]-right[1:])
                        ha,hb=fa[2]*left[0],fb[2]*right[0]
                        error=max(abs(math.log(ha/hb))/math.log(1.5),np.linalg.norm((pa-pb)*[16/9,1.])/max(.005,(ha+hb)/2.))
                        if role_strength[ai,bi]>0:
                            scale=abs(math.log(ha/hb))/math.log(1.5)
                            framing=(scale/(1.+scale)+np.linalg.norm((pa-pb)*[16/9,1.])/np.hypot(16/9,1.))/2.
                            error=(1.-role_strength[ai,bi])*error+role_strength[ai,bi]*framing
                        score=error+pose[ai,bi]+appearance[ai,bi]+1.-quality+.1*(math.log(left[0])+math.log(right[0]))
                        if opportunities is not None:
                            score-=opportunities.density(n,left_obs,right_obs,a,b,
                                native_pair['matched_action_change_density'][ai,bi])
                        candidates.setdefault((a,b),[]).append((float(score),a,b,ha_index,hb_index,k,left,right))
            feasibility={}
            def feasible(option):
                _,a,b,ha_index,hb_index,k,left,right=option
                key=(a,b,ha_index,hb_index,k)
                if key not in feasibility:
                    left_obs=catalogues[a].observation(n,0,np.tile(left,(7,1)),target_head_index=ha_index)
                    right_obs=catalogues[b].observation(n,1,np.tile(right,(7,1)),target_head_index=hb_index)
                    feasibility[key]=bool(left_obs['target_is_displayed_focal'] and right_obs['target_is_displayed_focal'])
                return feasibility[key]
            for a in active:
                for b in active:
                    for option in sorted(candidates.get((a,b),()),key=lambda x:x[:6]):
                        if not feasible(option):
                            # This CAMERA option lost the proposed focal; do
                            # not quietly score another actor's pose or veto
                            # the source. Other legal camera options remain.
                            rejected_reassignments+=1
                            continue
                        proposals.append(option)
                        break
            # One representative per ordered SOURCE PAIR, not a source quota.
            proposals=sorted(proposals,key=lambda x:x[:6])[:maximum_pairs]
            primary=tuple(proposals)
            def physical(option):
                _,a,b,ha,hb,_,left,right=option
                return (a,b,ha,hb,*left,*right)
            if recycle_static_capacity:
                # Identical static aliases consume DP slots but add no movie.
                # Reuse only those freed slots for distinct, already-generated
                # camera/subject options. Keep all old source-pair winners and
                # every physical baseline. Never enlarge the port-work bound.
                retained={physical(option) for option in proposals}
                before=len(proposals)
                remaining=sorted((option for options in candidates.values() for option in options),key=lambda x:x[:6])
                for option in remaining:
                    if len(proposals)>=capacity:break
                    key=physical(option)
                    if key in retained or not feasible(option):continue
                    retained.add(key);proposals.append(option)
                extra_count+=len(proposals)-before
            if retain_pair_cameras:
                retained={physical(option) for option in proposals}
                actors={(a,b,ha,hb) for _,a,b,ha,hb,*_ in primary}
                for option in sorted((o for values in candidates.values() for o in values
                        if o[1:5] in actors),key=lambda x:x[:6]):
                    key=physical(option)
                    if key in retained or not feasible(option):continue
                    retained.add(key);proposals.append(option);camera_variants+=1
            for _,a,b,ha,hb,k,ca,cb in proposals:
                original_retained+=int(np.array_equal(ca,[1.,.5,.5]) and np.array_equal(cb,[1.,.5,.5]))
                pair_id=('paired',n,a,b,ha,hb,k)
                non_native_focal_pairs+=int(ha!=left_native[a] or hb!=right_native[b])
                add(outgoing,port(a,n,0,np.tile(ca,(7,1)),(pair_id,),
                    target_head_index=ha))
                add(incoming,port(b,n,1,np.tile(cb,(7,1)),(pair_id,),
                    target_head_index=hb))
        if max(len(incoming),len(outgoing))>port_limit:raise AssertionError('port domain grew beyond global bound')
        entries.append(tuple(incoming));exits.append(tuple(outgoing));counts.append(len(proposals))
    return entries,exits,dense,dict(global_paired_proposal_limit_per_node=max(pair_capacities,default=maximum_pairs),
        baseline_paired_proposal_limit_per_node=maximum_pairs,global_port_limit_per_side=port_limit,
        source_cohort=list(active),original_source_index_domain_preserved=True,
        identical_static_capacity_recycled=bool(recycle_static_capacity),
        additional_distinct_proposals_from_static_aliases=extra_count,
        retained_pair_counts=counts,maximum_entry_ports=max(map(len,entries)),maximum_exit_ports=max(map(len,exits)),
        paired_endpoints_kept_together=True,roll_degrees=0.,post_DP_camera_fit=False,
        paired_subject_pinned_before_camera=True,rejected_camera_focal_reassignments=rejected_reassignments,
        physical_actor_pairs_compared_before_camera=actor_pairs_considered,
        retained_pairs_using_non_native_focal=non_native_focal_pairs,
        raw_fixed_opportunity_used_before_global_pruning=opportunities is not None,
        actual_articulated_action_used_before_global_pruning=bool(articulated_actions),
        natural_action_original_view_options_added=natural_added,
        retained_paired_original_boundary_views=original_retained,
        original_feature_role_used_before_global_pruning=roles is not None,
        identical_static_port_aliases_coalesced=static_aliases,
        all_baseline_interior_camera_families_retained=True,
        primary_actor_pair_camera_variants_retained=bool(retain_pair_cameras),
        additional_primary_pair_camera_options=camera_variants,
        previous_work_bound_unchanged=not bool(retain_pair_cameras),
        primary_pair_camera_options_per_actor_pair_upper_bound=4,
        proposal_ranking_uses_reference_times_or_cut_count=False,
        proposal_rank_is_declared_engineering_not_preference_learning=True)
