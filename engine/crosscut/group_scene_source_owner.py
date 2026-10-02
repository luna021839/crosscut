"""Normal scene/source/whole-camera sibling of the retained-family owner.

Numeric alternatives precede fixed-neighbor scene discovery. A finite source-
balanced graph chooses fixed-neighbor HYPOTHESES (not fixed-neighbor pixels)
and one/two mutable cameras jointly. No successful solo is a prerequisite.
The owner is deliberately a review-candidate producer, never semantic/quality QA.
"""
from dataclasses import replace
from collections import Counter
from itertools import product, zip_longest
from time import monotonic

from . import group_performer_family_owner as family
from . import group_source_changing_owner as source
from .group_scene_observations import SceneSensors, compare_scene_edge, preserves_known, DENSE_CONTRACT
from .group_scene_observations import VARIANTS, scene_anchor_ticks
from .group_joint_source_camera import GroupOutputRole, JointGroupWorkBudget
from .group_joint_observations import GroupJointObservationSession
from .face_match import FaceMatchConfig
from .source_derivative_diversity import (SourceDerivativeDiversityResult,
    DERIVATIVE_CONFIRMED, DERIVATIVE_DISTINCT, DERIVATIVE_UNCERTAIN)

CONTRACT = 'normal_scene_source_whole_camera_producer_v1'
# Search allocation, not a confidence/visibility/quality threshold.
MAX_WINDOWS = 3
FIXED_HYPOTHESES = 2
COMPLETE_REFRESH_RESERVE_SECONDS = 8.


def freeze_derivatives(result, domain, identities):
    """Bind an existing typed app audit, including uncertainty, to this domain.

    This is provenance binding, NOT a fresh re-estimation of derivative pairs.
    Parent replay reuses the original audit and independently reobserves all
    selected original pixels; full derivative detector replay remains separate.
    """
    if result is None:
        return None
    if not isinstance(result,SourceDerivativeDiversityResult):
        raise ValueError('scene source family requires the typed normal app derivative audit')
    payload = result.to_dict()
    if (len(result.family_ids) != len(domain.eligible_indices)
            or family.digest(result._payload(include_sha256=False)) != result.audit_sha256):
        raise ValueError('scene derivative audit hash or eligible-source cardinality differs')
    result.decoder_metadata()  # Structural validation includes canonical family labels.
    pairs = {}
    for row in result.pairs:
        a,b = row.left_source_index,row.right_source_index
        if not 0 <= a < b < len(result.family_ids) or (a,b) in pairs:
            raise ValueError('scene derivative pair domain/uniqueness invalid')
        pairs[a,b] = row.status
    return family.plain(dict(contract='scene_typed_derivative_preimage_v1',audit=payload,
        eligible_request_indices=list(domain.eligible_indices),
        source_identities=[list(v) for v in identities],
        audit_is_not_fresh_native_reclassification=True))


def relation(binding, a, b):
    if a == b: return DERIVATIVE_CONFIRMED
    if not binding: return DERIVATIVE_UNCERTAIN
    audit = binding['audit']; ids = audit['family_ids']
    if ids[a] == ids[b]: return DERIVATIVE_CONFIRMED
    key = min(a,b),max(a,b)
    matches = [p for p in audit['pairs'] if (p['left_source_index'],p['right_source_index']) == key]
    # Singleton family labels do NOT establish independence without a pair.
    return matches[0]['status'] if len(matches)==1 else DERIVATIVE_UNCERTAIN


def validate_bound_derivative_path(binding, source_indices):
    """Enforce the same frozen distinct-pair contract in scene AND legacy fallback.

    This checks the normal app's supplied evidence, not a fresh derivative
    classification. An incomplete audit/UNKNOWN cannot buy a source switch.
    """
    if not isinstance(binding,dict) or not isinstance(binding.get('audit'),dict):
        raise ValueError('source-changing derivative preimage is missing')
    audit=binding['audit'];payload=dict(audit);identity=payload.pop('audit_sha256',None)
    if family.digest(payload)!=identity:
        raise ValueError('source-changing derivative audit hash changed')
    if (audit.get('production_ready') is not True or audit.get('clock_contract_complete') is not True
            or audit.get('family_consistency_conflict_count')!=0):
        raise ValueError('source-changing derivative clock/family audit is not ready')
    count=len(audit['family_ids'])
    if not 3<=len(source_indices)<=4 or any(type(i) is not int or not 0<=i<count for i in source_indices):
        raise ValueError('source-changing derivative path uses another eligible index domain')
    if any(relation(binding,a,b)!=DERIVATIVE_DISTINCT for a,b in zip(source_indices,source_indices[1:])):
        raise ValueError('source-changing cut is duplicate/UNKNOWN, not a verified distinct-camera pair')


def round_robin(groups, limit):
    selected=[]
    for row in zip_longest(*groups):
        for item in row:
            if item is not None and all(item.key != x.key for x in selected):
                if len(selected) == limit: return tuple(selected)
                selected.append(item)
    return tuple(selected)


def time_stratified_window_frontier(prepared, limit):
    """Spend the bounded first pass across the song, not a chronological prefix.

    Window midpoint is a work-allocation coordinate, not a quality score. Each
    equally sized output-time band gets the nearest available unused window.
    The full numeric inventory remains intact; this does not claim that every
    window has been observed or change any admission threshold.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError('scene frontier requires a positive window budget')
    prepared = tuple(prepared)
    if len(prepared) <= limit:
        return tuple(enumerate(prepared))
    centers = tuple((item[0][1].first_output_frame + item[0][-2].end_output_frame) / 2
        for item in prepared)
    first = min(item[0][0].first_output_frame for item in prepared)
    end = max(item[0][-1].end_output_frame for item in prepared)
    selected = []
    for band in range(limit):
        target = first + (band + .5) * (end - first) / limit
        available = [i for i in range(len(prepared)) if i not in selected]
        index = min(available, key=lambda i: (abs(centers[i] - target), i))
        selected.append(index)
    return tuple((i, prepared[i]) for i in sorted(selected, key=lambda i: centers[i]))


def _target(row):
    if row['role'] != GroupOutputRole.FORMATION.value:
        return tuple(row['projected_face_roi_xywh'])
    boxes=[p['projected_box_xywh'] for p in row['scene_people']]
    if not boxes: raise ValueError('formation screen cohort unavailable')
    x=min(b[0] for b in boxes);y=min(b[1] for b in boxes)
    return x,y,max(b[0]+b[2] for b in boxes)-x,max(b[1]+b[3] for b in boxes)-y


def _native_target(row):
    if row['role'] != GroupOutputRole.FORMATION.value: return tuple(row['raw_face_roi_xywh'])
    boxes=[p['box_xywh'] for p in row['native_observation']['people']]
    if not boxes: raise ValueError('formation native cohort unavailable')
    x=min(b[0] for b in boxes);y=min(b[1] for b in boxes)
    return x,y,max(b[0]+b[2] for b in boxes)-x,max(b[1]+b[3] for b in boxes)-y


def propose_scene_changes(context):
    attempts=[];numeric_inventory=[];alternatives=[];session=None;sensors=None
    def rejected(error):
        return family.PerformerFamilyRejection(error,dict(contract=CONTRACT,
            attempts=attempts,numeric_source_inventory=numeric_inventory,candidate_alternatives=alternatives,
            scene_observations=None if sensors is None else sensors.archive(),
            native_work=None if session is None else session.budget.to_dict(),
            source_derivative_binding=context.source_derivative_binding,
            full_epoch_retained_family_owner_implemented=False,
            quality_approved=False))
    # The normal app supplies this audit; direct older callers keep their old
    # source owner, rather than silently treating integer indices as cameras.
    if context.source_derivative_binding is None:
        return rejected('normal scene owner has no typed source-derivative preimage')
    try:
        family._fresh(context)
        if len(context.domain.eligible_indices)>source.MAX_SOURCES:
            return rejected('scene source budget exceeded without truncating sources')
        assets=source.installed_native_assets(context)
        session=GroupJointObservationSession(domain=context.domain,budget=JointGroupWorkBudget(
            absolute_deadline=context.deadline,**source.WORK_LIMITS),maximum_raw_cache_bytes=32*1024*1024)
        sensors=SceneSensors(session,assets)
        plan=context.incumbent.render_plan;shots=source.visible_group_shots(plan,context.domain)
        # Interleave paired and solo numeric domains. The current state frontier
        # is source-balanced but NOT request-window-fair; later windows may remain
        # unobserved. Do not claim that this solves the global prefix budget.
        windows=[]
        for i in range(max(0,len(shots)-2)):
            if i+4<=len(shots): windows.append(shots[i:i+4])
            windows.append(shots[i:i+3])
        prepared=[]
        # Pure numeric inventory is complete BEFORE ANY fixed-head discovery.
        for window in windows:
            if any(s.frame_count>source.MAX_MUTABLE_FRAMES for s in window[1:-1]): continue
            record=dict(bounds=[[s.first_output_frame,s.end_output_frame] for s in window],states=[],errors=[])
            numeric_inventory.append(record)
            try:
                roles=source._shot_roles(context,window)
                baseline=source._states(plan,context.domain,window,roles)
                by_slot=[]
                for incumbent in baseline[1:-1]:
                    values=[]
                    for index in range(len(context.domain.eligible_indices)):
                        try:
                            value=source._hypothesis(context,incumbent,index)
                            values.append(value);record['states'].append(dict(slot=value.slot,source=index,state_key=value.key))
                        except ValueError as exc: record['errors'].append(dict(slot=incumbent.slot,source=index,error=str(exc)))
                    by_slot.append(tuple(values))
                prepared.append((baseline,tuple(by_slot),record))
            except ValueError as exc:
                record['errors'].append(dict(stage='retained_epoch_topology',error=str(exc)))
        # Upright observations for time-distributed windows precede rotated
        # retries. A failed intro must not consume every selected work slot.
        frontier=time_stratified_window_frontier(prepared,MAX_WINDOWS)
        scheduled=tuple((wi,item,rolls) for rolls in ((None,),VARIANTS)
            for wi,item in frontier)
        for wi,(baseline,raw_slots,numeric),rolls in scheduled:
            if monotonic()+COMPLETE_REFRESH_RESERVE_SECONDS >= context.deadline:
                attempts.append(dict(stage='reserve',reason='complete_refresh_reserve_preserved'));break
            if session.budget.counts['states']>=session.budget.limits['states']:
                attempts.append(dict(stage='state_frontier_exhausted',numeric_windows=len(prepared)));break
            start_counts=dict(session.budget.counts)
            # All available sources get their upright observations before any
            # camera choice. Rotated hypotheses remain a separately named pass.
            def scope_for(index):
                return 'exit' if index==0 else 'entry' if index==len(baseline)-1 else 'whole'
            all_native=tuple((s,scope_for(i)) for i,s in enumerate(baseline))+tuple(
                (s,'whole') for group in raw_slots for s in group)
            for state,scope in all_native:
                sensors.discover(state,rolls=rolls,endpoint_scope=scope)
            def bound(state,scope='whole'):
                return tuple(sensors.bind(state,h) for h in sensors.discover(state,rolls=rolls,endpoint_scope=scope))
            lefts=bound(baseline[0],'exit');rights=bound(baseline[-1],'entry')
            # Fixed R needs distinct ENTRY hypotheses; fixed L distinct EXIT.
            def fixed(values,entry):
                seen=set();result=[]
                for value in values:
                    h=sensors.hypothesis(value);key=(h['analysis_roll'],tuple(h['anchor_head_ids'][:2] if entry else h['anchor_head_ids'][-2:]))
                    if key not in seen:seen.add(key);result.append(value)
                return tuple(result[:FIXED_HYPOTHESES])
            lefts=fixed(lefts,False);rights=fixed(rights,True)
            if not lefts or not rights:
                attempts.append(dict(stage='scene_fixed_hypotheses',window=wi,
                    analysis_rolls=list(rolls),reason='all_fixed_endpoint_paths_unavailable'));continue
            groups=[];camera_errors=[]
            for slot,raw_values in enumerate(raw_slots,1):
                by_source=[]
                for raw in raw_values:
                    values=list(bound(raw));cameras=[]
                    if raw.eligible_source_index==baseline[slot].eligible_source_index:
                        values=list(dict((v.key,v) for v in (*bound(baseline[slot]),*values)).values())
                    for value in values:
                        for left,right in product(lefts,rights):
                            try:
                                l=_target(sensors.frame(left,left.end_output_frame-1));r=_target(sensors.frame(right,right.first_output_frame))
                                interior=tuple((a+b)/2 for a,b in zip(l,r))
                                targets=(l,r) if len(baseline)==3 else ((l,interior) if slot==1 else (interior,r))
                                first=sensors.frame(value,value.first_output_frame);last=sensors.frame(value,value.end_output_frame-1)
                                a,ta,_=source._at(value,value.first_output_frame);b,tb,_=source._at(value,value.end_output_frame-1)
                                boxes=(source._base_box(_native_target(first),a,ta),source._base_box(_native_target(last),b,tb))
                                proposal=source.propose_endpoint_focal_face_roi_camera(source_face_boxes=boxes,
                                    fixed_neighbor_face_boxes=targets,frame_count=len(value.native_frame_indices),fps=plan.fps,
                                    maximum_zoom=(FaceMatchConfig().maximum_silhouette_zoom
                                        if value.role==GroupOutputRole.FORMATION else 2.))
                                camera=source._hypothesis(context,baseline[slot],value.eligible_source_index,proposal.track)
                                cameras.append(sensors.bind(camera,sensors.hypothesis(value)))
                            except ValueError as exc:
                                if len(camera_errors)<source.MAX_RECORDED_FAILURES:camera_errors.append(str(exc))
                    # Native then camera, round-robin over SOURCE, not the first
                    # source's many head/camera variants monopolizing the cap.
                    by_source.append(tuple(dict((v.key,v) for v in (*values,*cameras)).values()))
                remaining=session.budget.limits['states']-session.budget.counts['states']-len(lefts)-len(rights)
                # At most 24 states globally. Discovery still preserves all the
                # omitted hypotheses and numeric source options in its archive.
                quota=max(0,remaining//len(raw_slots))
                groups.append(round_robin(by_source,quota))
            all_states=tuple(dict((v.key,v) for v in (*lefts,*(s for g in groups for s in g),*rights)).values())
            for state in all_states:session.budget.spend('states')
            if any(not group for group in groups):
                attempts.append(dict(stage='state_budget',window=wi,unexplored=True));break
            baseline_hypotheses=[bound(s,scope_for(i)) for i,s in enumerate(baseline)]
            edges={};known={}
            def read(a,b):
                key=a.key,b.key
                if key not in edges:
                    session.budget.spend('edges')
                    evidence=compare_scene_edge(sensors,a,b)
                    evidence['source_relation']=relation(context.source_derivative_binding,a.eligible_source_index,b.eligible_source_index)
                    # Unknown/duplicate may be observed, but cannot masquerade
                    # as a verified distinct-camera cut in the selected graph.
                    if evidence['source_relation'] != DERIVATIVE_DISTINCT:evidence['available']=False
                    edges[key]=evidence
                return edges[key]
            # A candidate cannot win by choosing a conveniently poor incumbent
            # hypothesis. Retain every observed baseline component's best value.
            for index in range(len(baseline)-1):
                values=[]
                for a,b in product(baseline_hypotheses[index],baseline_hypotheses[index+1]):
                    if len(values)>=4:break
                    values.append(read(a,b))
                metrics={}
                # Cohort geometry is observed independent of focal assignment.
                # A disputed focal hypothesis is NOT an unconditional incumbent
                # fact. In particular its minimum conditional cost is not zero
                # uncertainty, nor a licence to erase known cohort components.
                unconditional={'occupancy','attention','scale','span','screen_motion',
                    'formation_pose','formation_framing'}
                unique=True
                for old_index in (index,index+1):
                    old_state=baseline[old_index]
                    try:
                        for tick in scene_anchor_ticks(old_state.first_output_frame,
                                old_state.end_output_frame,scope_for(old_index)):
                            source._dominant(sensors.raw(old_state,tick,None),old_state.role)
                    except ValueError:unique=False
                for row in values:
                    for key,value in (row.get('metrics') or {}).items():
                        if value is not None and (unique or key in unconditional):
                            metrics[key]=min(metrics.get(key,float('inf')),value)
                known[index]=dict(metrics=metrics,observed_alternatives=values,
                    focal_assignment_unique=unique,focal_conditional_cost_not_imputed=True,
                    available=any(v.get('available') for v in values))
            paths=[]
            for path in product(lefts,*groups,rights):
                measured=[]
                for index,(a,b) in enumerate(zip(path,path[1:])):
                    row=read(a,b)
                    if not preserves_known(known[index],row):break
                    measured.append(row)
                if len(measured)!=len(path)-1:continue
                gain=any(not known[i]['available'] or any(v<known[i]['metrics'][k]-1e-6
                    for k,v in m['metrics'].items() if k in known[i]['metrics']) for i,m in enumerate(measured))
                if not gain:continue
                paths.append((sum(row['framing_cost'] for row in measured),tuple(s.key for s in path),path,measured))
            for _,_,path,measured in sorted(paths,key=lambda p:p[:2]):
                session.budget.spend('dense_paths')
                alternative=dict(states=[s.to_dict() for s in path],status='dense_pending');alternatives.append(alternative)
                try:
                    assembled,changed=source._assembler(path)(plan=plan,domain=context.domain,path=path)
                    if not any(assembled.segments[i].source_index!=plan.segments[i].source_index for i in changed):continue
                    dense=sensors.dense(path)
                    # Per-tick known incumbent parts are preserved independent
                    # of whether its old dominant face was decisive.
                    preserved=[]
                    for old,new,span in zip(baseline[1:-1],path[1:-1],dense,strict=True):
                        for row in span['rows']:
                            raw=sensors.raw(old,row['output_tick'],None)
                            required=Counter(name for p in raw.people for name,status in getattr(p,'part_states',()) if name in ('left_arm','right_arm') and status=='observed')
                            actual=Counter(a['part'] for a in row['projected_observed_arms'])
                            if any(actual[name]<count for name,count in required.items()):raise ValueError('scene source change loses an incumbent observed arm')
                            preserved.append(dict(output_tick=row['output_tick'],required_parts=dict(sorted(required.items())),
                                part_count_is_not_cross_camera_actor_correspondence=True,
                                native_observation_sha256=raw.sha256,absence_is_not_a_negative_detection_certificate=True))
                    search=dict(producer=CONTRACT,generated_numeric_states=numeric['states'],
                        numeric_inventory=numeric_inventory,prior_solo_success_required=False,
                        fixed_geometry_unchanged=True,baseline_components={str(k):v for k,v in known.items()},
                        selected_scene_hypotheses=[s.editorial_contract_sha256 for s in path],
                        source_derivative_binding=context.source_derivative_binding,
                        incumbent_known_parts=preserved,camera_errors=camera_errors,
                        candidate_search_exhaustive=False,rotated_pass_exhaustive=False,
                        source_balanced_state_allocation=True,full_epoch_retained_family_owner_implemented=False,
                        window_frontier_policy='time_stratified_midpoints_upright_before_rotated_retry_v1',
                        selected_frontier_window_indices=[i for i,_ in frontier],
                        request_wide_work_accounting=False,window_fairness_guaranteed=False,
                        reserve_check_basis="before_window_only_not_hard_phase_partition")
                    result=source.complete_source_candidate(context,path,sensors,session,assets,dense,measured,search)
                    alternative['status']='complete_unapproved'
                    return result
                except (ValueError,RuntimeError,OSError) as exc:
                    alternative.update(status='rejected',error=str(exc))
            attempts.append(dict(stage='scene_graph',window=wi,state_keys=[s.key for s in all_states],
                path_count=len(paths),edge_observations=list(edges.values()),baseline_components=known,
                window_native_start_counts=start_counts,
                window_native_end_counts=dict(session.budget.counts),camera_errors=camera_errors))
        return rejected('no complete bounded scene/source/whole-camera family candidate')
    except (ValueError,RuntimeError,OSError) as exc:
        return rejected(type(exc).__name__+': '+str(exc))
    finally:
        if session is not None:session.close()
