"""One independently anchored actor hypothesis per original native shot.

The original's output geometry picks one seed before donor/pose costs. Native
appearance/position then follows that seed in both directions. Detector IDs
are frame-local. A missing/ambiguous association is never max-face recovery,
named identity, production tracking, or a complete motion certificate.
"""
from dataclasses import replace
import math

from .group_performer_family_owner import digest
from .group_review_actor_path import _link
from .group_review_source_equivalence import physical_review_file_identity


def _native_association_view(view,unit):
    raw=view['row']['native_observation'];d=view['descriptor']
    head=next((h for h in raw['heads'] if h['observation_id']==d.frame_local_head_id),None)
    if head is None:raise ValueError('original actor view lost its actual native occurrence')
    x,y,w,h=head['box_xywh']
    # Association uses actual native head coordinates, not arbitrary original
    # editorial crop jumps. The real row/camera returned to consumers is intact.
    descriptor=replace(d,head_center=(x+w/2,y+h/2),face_width=w,face_height=h,pose_shape=None)
    return dict(view,descriptor=descriptor,row=dict(view['row'],projection=dict(
        view['row']['projection'],camera_sha256='native-association-only:'+digest(unit))))


def freeze_original_review_actor_inventory(frames,*,native_units,original_program_identity,
        maximum_frames=2048,maximum_native_gap=12,minimum_cosine=.25,ambiguity_margin=.025,
        original_attention_scopes=None,seed_attention_margin=.025,original_roles_by_tick=None,
        focal_area_rival_ratio=1.25):
    """Freeze output-tick bindings without inspecting donor/action-match costs.

    Each physical original shot uses the observed frame nearest its inventory
    midpoint, then its most central sufficiently visible output face as seed.
    Face area breaks spatial ties; body completeness never selects that seed.
    A retained direct anchor may bridge a missing slot only within the SAME
    fixed native gap. Recursive gap expansion / re-seeding after loss is absent.
    """
    if (not isinstance(frames,dict) or not 1<=len(frames)<=maximum_frames
            or set(frames)!=set(native_units) or not isinstance(original_program_identity,str)
            or not original_program_identity or not 1<=maximum_native_gap<=12
            or not 0<=minimum_cosine<=1 or not 0<=ambiguity_margin<=1
            or type(seed_attention_margin) not in (int,float) or not math.isfinite(seed_attention_margin)
            or not 0<=seed_attention_margin<=1 or type(focal_area_rival_ratio) not in (int,float)
            or not math.isfinite(focal_area_rival_ratio) or not 1<=focal_area_rival_ratio<=4):
        raise ValueError('original actor inventory needs one finite independent physical request set')
    scopes={t:'single-independent-request' for t in frames} if original_attention_scopes is None else (
        original_attention_scopes)
    if (not isinstance(scopes,dict) or set(scopes)!=set(frames)
            or any(not isinstance(v,str) or not v for v in scopes.values())):
        raise ValueError('original attention scopes must be independently frozen exact output requests')
    roles={t:'UNSPECIFIED' for t in frames} if original_roles_by_tick is None else original_roles_by_tick
    focal_roles=('focal_head','focal_head_and_observed_arms')
    composition_roles=('required_formation','observed_body_composition')
    if (not isinstance(roles,dict) or set(roles)!=set(frames)
            or any(v not in (*focal_roles,*composition_roles,'UNSPECIFIED') for v in roles.values())):
        raise ValueError('original roles require the exact independently frozen original programme')
    groups={}
    for tick,views in sorted(frames.items()):
        if type(tick) is not int or tick<0 or not isinstance(views,(list,tuple)) or len(views)>24:
            raise ValueError('original actor frame/head inventory is unbounded')
        unit=tuple(native_units[tick])
        if len(unit)!=4:raise ValueError('original actor requires source/clock/file/native-camera unit')
        for view in views:
            raw=view['row']['native_observation']
            if (view['row']['output_tick']!=tick or raw['request_source_index']!=unit[0]
                    or raw['source_clock_identity']!=unit[1]
                    or digest(physical_review_file_identity(view))!=unit[2]):
                raise ValueError('original actor inventory borrowed another raster or native unit')
        # Repeated visits to one native camera, separated by other original
        # shots, are NOT one continuous actor request. Output attention scopes
        # come from the ORIGINAL interval inventory, never failed donor/body
        # costs. They do not pretend to be new physical native camera cuts.
        groups.setdefault((unit,scopes[tick],roles[tick]),[]).append(tick)
    selected={t:None for t in frames};groups_receipts=[]
    for (unit,attention_scope,role),ticks in groups.items():
        observed=[t for t in ticks if frames[t]]
        if not observed:
            groups_receipts.append(dict(unit=list(unit),original_attention_scope=attention_scope,
                original_role=role,unique_focal_intent_certified=False,independent_original_head_hypotheses=[],
                seed_tick=None,requests=ticks,seed_attention_ambiguous=True,
                seed_attention_rivals=[],seed_actor_correspondence_certified=False));continue
        midpoint=(ticks[0]+ticks[-1])/2
        seed_tick=min(observed,key=lambda t:(abs(t-midpoint),t))
        def attention(view):
            d=view['descriptor'];x,y=d.head_center
            return (math.hypot(x-.5,(y-.4)*9/16),-d.face_height*d.face_width,
                digest((view['vector'],d.head_center,d.face_height,d.face_width)))
        ranked_seeds=sorted(frames[seed_tick],key=attention)
        seed=ranked_seeds[0];seed_distance=attention(seed)[0]
        # Centrality is an independently chosen attention heuristic, not an
        # identity certificate. A deterministic area/hash tie-break must not
        # erase plausible other original actors or the seed's existing doubt.
        seed_area=seed['descriptor'].face_height*seed['descriptor'].face_width
        seed_rivals=[v for v in ranked_seeds[1:] if attention(v)[0]-seed_distance<=seed_attention_margin
            or role in focal_roles and v['descriptor'].face_height*v['descriptor'].face_width>=
                seed_area*focal_area_rival_ratio]
        seed_ambiguous=bool(seed.get('reacquisition_ambiguous',False) or seed_rivals)
        seed=dict(seed,reacquisition_ambiguous=seed_ambiguous,original_role=role,
            unique_focal_intent_certified=False,
            original_attention_hypothesis_is_focal_duty=role in focal_roles)
        selected[seed_tick]=dict(seed,original_actor_inventory_seed=True)
        seed_position=ticks.index(seed_tick);association_links=[]
        for positions in (range(seed_position+1,len(ticks)),range(seed_position-1,-1,-1)):
            anchor=seed;anchor_tick=seed_tick
            for position in positions:
                tick=ticks[position];ranked=[]
                for value in frames[tick]:
                    left,right=(anchor,value) if anchor_tick<tick else (value,anchor)
                    link=_link(_native_association_view(left,unit),_native_association_view(right,unit),
                        maximum_native_gap=maximum_native_gap,minimum_cosine=minimum_cosine)
                    if link is not None and link['linked']:
                        ranked.append((link['cost'],attention(value),value,link))
                ranked.sort(key=lambda p:(p[0],p[1]))
                if not ranked:continue
                best=ranked[0];margin=None if len(ranked)==1 else ranked[1][0]-best[0]
                value=dict(best[2],reacquisition_ambiguous=bool(
                    best[2].get('reacquisition_ambiguous',False) or anchor.get('reacquisition_ambiguous',False)
                    or margin is not None and margin<ambiguity_margin),original_role=role,
                    unique_focal_intent_certified=False,
                    original_attention_hypothesis_is_focal_duty=role in focal_roles)
                selected[tick]=value;association_links.append(dict(anchor_tick=anchor_tick,target_tick=tick,
                    margin=margin,link=best[3],ambiguous=value['reacquisition_ambiguous']))
                anchor=value;anchor_tick=tick
        groups_receipts.append(dict(unit=list(unit),requests=ticks,seed_tick=seed_tick,
            original_attention_scope=attention_scope,
            original_role=role,unique_focal_intent_certified=False,
            composition_role_is_single_actor_intent=False,
            independent_original_head_hypotheses=[dict(tick=t,head_id=v['descriptor'].frame_local_head_id,
                raw_parent=digest(v['row']['native_observation']),
                role_relation='FOCAL_ATTENTION_ALTERNATIVE' if role in focal_roles else
                    'COMPOSITION_CONTEXT_NOT_UNIQUE_FOCAL' if role in composition_roles else 'ROLE_UNRESOLVED',
                output_head_center=list(v['descriptor'].head_center),
                output_face_size=[v['descriptor'].face_width,v['descriptor'].face_height],
                donor_costs_used=False,quality_approved=False) for t in ticks for v in frames[t]],
            seed_head_id=seed['descriptor'].frame_local_head_id,links=association_links,
            seed_attention_ambiguous=seed_ambiguous,seed_actor_correspondence_certified=False,
            seed_attention_distance=seed_distance,seed_attention_rivals=[dict(
                head_id=v['descriptor'].frame_local_head_id,raw_parent=digest(v['row']['native_observation']),
                attention_distance=attention(v)[0],distance_margin=attention(v)[0]-seed_distance,
                output_face_area=v['descriptor'].face_height*v['descriptor'].face_width,
                independently_original_focal_area_rival=bool(role in focal_roles and
                    v['descriptor'].face_height*v['descriptor'].face_width>=seed_area*focal_area_rival_ratio)) for v in seed_rivals],
            seed_basis='independent_original_output_attention_not_donor_pose_error'))
    bindings=[]
    for tick,view in sorted(selected.items()):
        raw=None if view is None else view['row']['native_observation']
        bindings.append(dict(output_tick=tick,head_id=None if view is None else view['descriptor'].frame_local_head_id,
            raw_parent=None if raw is None else digest(raw),
            ambiguous=False if view is None else view.get('reacquisition_ambiguous',False),
            state='UNMATCHED' if view is None else 'OBSERVED_ANCHORED_ORIGINAL_ACTOR_HYPOTHESIS'))
    receipt=dict(contract='original_native_shot_anchored_actor_inventory_v1',
        original_program_identity=original_program_identity,groups=groups_receipts,bindings=bindings,
        maximum_native_gap=maximum_native_gap,minimum_cosine=minimum_cosine,
        ambiguity_margin=ambiguity_margin,seed_attention_margin=seed_attention_margin,
        focal_area_rival_ratio=focal_area_rival_ratio,focal_area_rival_ratio_is_identity_certificate=False,
        seed_attention_margin_units='aspect_corrected_normalized_output_center_distance',
        seed_attention_margin_is_actor_identity_threshold=False,
        donor_costs_used=False,production_ownership_rewritten=False,
        attention_scopes_are_physical_camera_cuts=False,
        independently_frozen_original_roles=roles,all_original_head_alternatives_retained=True,
        named_actor_certified=False,quality_approved=False)
    receipt['inventory_identity']=digest(receipt)
    return selected,receipt
