"""Normal scene/source/whole-camera sibling of the retained-family owner.

Numeric alternatives precede fixed-neighbor scene discovery. A finite source-
balanced graph chooses fixed-neighbor HYPOTHESES (not fixed-neighbor pixels)
and one/two mutable cameras jointly. No successful solo is a prerequisite.
The owner is deliberately a review-candidate producer, never semantic/quality QA.
"""
from dataclasses import replace
from itertools import product, zip_longest
import math
from time import monotonic

from . import group_performer_family_owner as family
from . import group_source_changing_owner as source
from .group_scene_observations import (SceneSensors, compare_scene_edge,
    coherent_baseline, preserves_known, has_editorial_gain, DENSE_CONTRACT)
from .group_scene_observations import VARIANTS, scene_anchor_ticks, MAX_HYPOTHESES_PER_SPAN
from .group_joint_source_camera import GroupOutputRole, JointGroupWorkBudget,JointGroupBudgetExhausted
from .group_joint_observations import GroupJointObservationSession
from .face_match import FaceMatchConfig
from .source_derivative_diversity import (SourceDerivativeDiversityResult,
    DERIVATIVE_CONFIRMED, DERIVATIVE_DISTINCT, DERIVATIVE_UNCERTAIN)

CONTRACT = 'normal_scene_source_whole_camera_producer_v1'
# Search allocation, not a confidence/visibility/quality threshold.
MAX_WINDOWS = 3
# Initial allocation batch, NOT a proof that unprobed windows are unavailable.
# Continuations still share this request's original native/time/graph budgets.
MAX_FIXED_WINDOW_PROBES = 12
FIXED_HYPOTHESES = 2
COMPLETE_REFRESH_RESERVE_SECONDS = 8.


def fixed_endpoint_inventory(values, *, sensors, entry):
    """Keep the entire bounded canonical discovery inventory BEFORE probing.

    This deduplicates identical endpoint hypotheses only. The full original
    branch list remains a separate duty input, never the allocated subset.
    """
    if not isinstance(values,tuple) or len(values)>MAX_HYPOTHESES_PER_SPAN or type(entry) is not bool:
        raise ValueError('fixed inventory needs the complete bounded discovered endpoint list')
    seen=set();result=[]
    for value in values:
        h=sensors.hypothesis(value)
        key=(h['analysis_roll'],tuple(h['anchor_head_ids'][:2] if entry else h['anchor_head_ids'][-2:]))
        if key not in seen:seen.add(key);result.append(value)
    return tuple(result)


def measure_fixed_endpoint_inventory(values, *, sensors, outgoing, record):
    """Allocate two USABLE endpoints, not two unexamined possibilities.

    Endpoint sensing spends the same caller-owned native budget. Preserve the
    measured prefix and continuation cursor before propagating an interrupted
    primitive; no retry or new quota is started here. This child-local record
    is not the still-missing controller-wide phase lease/history authority.
    """
    if (not isinstance(values,tuple) or len(values)>MAX_HYPOTHESES_PER_SPAN
            or type(outgoing) is not bool or not isinstance(record,dict)):
        raise ValueError('fixed endpoint measurement needs a bounded inventory and retained record')
    record.update(contract='bounded_discovered_fixed_endpoint_measurement_prefix_v1',
        inventory_state_keys=[value.key for value in values], next_cursor=0,
        completed_measurement_count=0, selected_state_keys=[], attempts=[],
        outcome='NOT_STARTED', inventory_exhausted=False, unexecuted_suffix_count=len(values),
        pending_candidate_count=len(values), interrupted_current_state_key=None,
        maximum_usable_endpoints=FIXED_HYPOTHESES, new_budget_retry_started=False,
        original_duty_branches_redefined=False, controller_owned_history=False,
        quality_approved=False)
    valid=[]
    if not values:
        record['inventory_exhausted']=True
        return ()
    for index,value in enumerate(values):
        attempt=dict(state_key=value.key,status='STARTED')
        record['attempts'].append(attempt)
        record['next_cursor']=index
        record['unexecuted_suffix_count']=len(values)-index-1
        record['outcome']='PARTIAL'
        try:
            sensors.endpoint(value,outgoing=outgoing)
        except JointGroupBudgetExhausted:
            attempt['status']='BUDGET_PARTIAL'
            record['outcome']='BUDGET_PARTIAL'
            record['interrupted_current_state_key']=value.key
            raise
        except ValueError as exc:
            attempt.update(status='OBSERVED_UNAVAILABLE',error=str(exc))
        except BaseException as exc:
            attempt.update(status='PRODUCER_FAILED',error_type=type(exc).__name__)
            record['outcome']='PRODUCER_FAILED'
            raise
        else:
            attempt['status']='USABLE'
            valid.append(value)
            record['selected_state_keys'].append(value.key)
        record.update(next_cursor=index+1,completed_measurement_count=index+1,
            unexecuted_suffix_count=len(values)-index-1,
            pending_candidate_count=len(values)-index-1)
        if len(valid)==FIXED_HYPOTHESES:break
    record.update(outcome='USABLE' if valid else 'OBSERVED_UNAVAILABLE',
        inventory_exhausted=record['next_cursor']==len(values))
    return tuple(valid)


class _EdgeFrontierFull(ValueError):
    """Stop search without discarding already measured complete candidates."""



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


def camera_balanced_source_frontier(native, fitted):
    """Keep camera fitting reachable before one source's many head hypotheses.

    The outer allocator balances source IDs. Concatenating all native head
    hypotheses before fitted cameras would still starve zoom/pan alternatives
    under that finite quota. Alternate both kinds without admitting either:
    original-pixel duties, edge measurements and dense checks remain required.
    """
    native, fitted = tuple(native), tuple(fitted)
    return round_robin((native, fitted), len(native)+len(fitted))


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


def observed_wide_repair_priority(timeline, mutable_shots):
    """Work priority from the incumbent, not a verdict or raw-donor family gate.

    Spend one scarce window on a distant existing wide before revisiting an
    already close/medium window. Missing scale is not replaced with zero; a
    wide reveal is not rejected, and every proposed replacement still needs
    the same two-sided/dense checks. No song or timestamp is special-cased.
    """
    duration=0.;weighted=0.
    for shot in mutable_shots:
        for choice in timeline.choices[shot.first_segment_index:shot.last_segment_index+1]:
            seconds=choice.canonical_master_end-choice.canonical_master_start
            if seconds<=0:continue
            duration+=seconds
            if getattr(choice.shot_type,'value',choice.shot_type) not in ('wide','group_wide'):
                continue
            values=tuple(v for v in (choice.entry_shot_scale,choice.exit_shot_scale)
                if type(v) in (int,float) and math.isfinite(v) and 0<=v<=1)
            if values:weighted+=seconds*(1-sum(values)/len(values))
    return weighted/duration if duration else 0.


def repair_window_frontier(prepared,limit):
    distributed=time_stratified_window_frontier(prepared,limit)
    if not distributed:return distributed
    ranked=[(item[2].get('observed_wide_repair_priority',0.),i,item)
        for i,item in enumerate(prepared)]
    score,index,item=max(ranked,key=lambda value:(value[0],-value[1]))
    if score<=0:return distributed
    return ((index,item),*tuple(row for row in distributed if row[0]!=index)[:limit-1])


def fixed_window_probe_order(prepared,limit):
    """Cheap fixed endpoints may fail without consuming an eight-source search.

    Keep the primary distributed/weak-wide choices, then visit distinct pending
    wide work. This is a bounded ordering, not evidence that an unobserved
    source/shot is absent, or a relaxation of any path/admission check.
    """
    if type(limit) is not int or limit<1:
        raise ValueError('scene fixed probe budget must be positive')
    primary=repair_window_frontier(prepared,min(MAX_WINDOWS,limit))
    used={i for i,_ in primary}
    remaining=sorted(((i,item) for i,item in enumerate(prepared) if i not in used),
        key=lambda row:(-row[1][2].get('observed_wide_repair_priority',0.),row[0]))
    return (*primary,*remaining)[:limit]


def fixed_window_probe_batches(prepared,*,batch_size):
    """Retain the initial order and continue UNKNOWN windows on the same work.

    New reachable windows can displace a formerly measured numeric window
    from the first batch. Failed fixed/original hypotheses consume no source
    graph, so that prefix is not an end-of-search condition. This allocator
    grants neither extra model/decode/time counters nor admission authority.
    The owning loop checks the unchanged graph/native/deadline limits before
    EVERY window and preserves observations in its existing sensor session.
    """
    if type(batch_size) is not int or batch_size<1:
        raise ValueError('scene fixed probe batch size must be a positive integer')
    prepared=tuple(prepared)
    if not prepared:return ()
    initial=fixed_window_probe_order(prepared,min(batch_size,len(prepared)))
    used={index for index,_ in initial}
    pending=sorted(((index,item) for index,item in enumerate(prepared) if index not in used),
        key=lambda row:(-row[1][2].get('observed_wide_repair_priority',0.),row[0]))
    ordered=(*initial,*pending)
    return tuple(tuple(ordered[first:first+batch_size]) for first in range(0,len(ordered),batch_size))


def _target(row):
    if row['role'] not in (GroupOutputRole.FORMATION.value, GroupOutputRole.COMPOSITION.value):
        return tuple(row['projected_face_roi_xywh'])
    boxes=[p['projected_box_xywh'] for p in row['scene_people']]
    if not boxes: raise ValueError('formation screen cohort unavailable')
    x=min(b[0] for b in boxes);y=min(b[1] for b in boxes)
    return x,y,max(b[0]+b[2] for b in boxes)-x,max(b[1]+b[3] for b in boxes)-y


def _native_target(row):
    if row['role'] not in (GroupOutputRole.FORMATION.value, GroupOutputRole.COMPOSITION.value): return tuple(row['raw_face_roi_xywh'])
    boxes=[p['box_xywh'] for p in row['native_observation']['people']]
    if not boxes: raise ValueError('formation native cohort unavailable')
    x=min(b[0] for b in boxes);y=min(b[1] for b in boxes)
    return x,y,max(b[0]+b[2] for b in boxes)-x,max(b[1]+b[3] for b in boxes)-y


def preserve_incumbent_required_parts(sensors, baseline, path, dense):
    """Check the PRESEARCH original programme, never reselect a dominant actor."""
    from .group_scene_original_duties import duty_at,FOCAL
    program=getattr(sensors,'original_duty_program',None)
    if program is None:
        raise ValueError('scene source change has no presearch original duty programme')
    preserved=[]
    for old,new,span in zip(baseline[1:-1],path[1:-1],dense,strict=True):
        for row in span['rows']:
            ref=duty_at(program,old,row['output_tick'])
            if row.get('original_duty_reference')!=ref:
                raise ValueError('scene source change loses its original required duty reference')
            if old.role.value in FOCAL:
                owner=row['role_obligations']['focal_owner_id']
                expected={(owner,name) for name in ref['required_part_names']}
                actual={(a['person_observation_id'],a['part']) for a in row['projected_observed_arms']
                    if a['required_by_role']}
                if expected!=actual or expected and owner is None:
                    raise ValueError('scene source change loses a selected-owner original required arm')
            preserved.append(dict(output_tick=row['output_tick'],original_duty_reference=ref,
                part_count_is_not_cross_camera_actor_correspondence=True,
                missing_original_measurement_is_not_a_duty_waiver=True,
                fresh_original_dominant_reselection=False))
    return preserved


def propose_scene_changes(context):
    attempts=[];numeric_inventory=[];alternatives=[];session=None;sensors=None
    active_window=None;active_graph=None
    def rejected(error, *, error_type='ValueError',outcome='MEASUREMENT_UNAVAILABLE',partial=None):
        return family.PerformerFamilyRejection(error,dict(contract=CONTRACT,
            error=error,error_type=error_type,
            outcome=outcome,unexecuted_measurements_are_unknown=True,
            fallback_work_restart_allowed=False,
            # Whole raw prefix retention is the separate F22 journal contract.
            raw_prefix_completeness_certified=False,
            interrupted_measurement=partial,
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
        request_limits=dict(source.WORK_LIMITS,
            maximum_states=MAX_WINDOWS*source.WORK_LIMITS['maximum_states'],
            maximum_edges=MAX_WINDOWS*source.WORK_LIMITS['maximum_edges'],maximum_frontiers=MAX_WINDOWS)
        session=GroupJointObservationSession(domain=context.domain,budget=JointGroupWorkBudget(
            absolute_deadline=context.deadline,**request_limits),maximum_raw_cache_bytes=32*1024*1024)
        sensors=SceneSensors(session,assets)
        plan=context.incumbent.render_plan;shots=source.visible_group_shots(plan,context.domain)
        # Preserve ORIGINAL external role epochs before a uniform-role numeric
        # path can reject a mixed shot, and before any donor/model observation.
        # Sidecars alone do not grant the old positional assembler mixed-role
        # authority; the still-unsupported global owner remains explicit.
        from .group_scene_owned_program import (freeze_owned_scene_program,own_scene_intervals,
            camera_unit_from_plan,assemble_owned_scene_union)
        owned_program=freeze_owned_scene_program(context,context.domain)
        owned_units={(unit.first_output_frame,unit.end_output_frame):unit for unit in owned_program.units}
        role_programme_refs={(unit.first_output_frame,unit.end_output_frame):
            sensors.record_original_role_program(unit.role_program) for unit in owned_program.units}
        # Interleave paired and solo numeric domains. The current state frontier
        # has separate bounded state/edge frontiers. Native/model/time work is
        # still shared, so later windows may remain unobserved; do not claim
        # full-song coverage or hard completion/parent reserve guarantees.
        windows=[]
        for i in range(max(0,len(shots)-2)):
            if i+4<=len(shots): windows.append(shots[i:i+4])
            windows.append(shots[i:i+3])
        prepared=[]
        # Pure numeric inventory is complete BEFORE ANY fixed-head discovery.
        for window in windows:
            if any(s.frame_count>source.MAX_MUTABLE_FRAMES for s in window[1:-1]): continue
            record=dict(bounds=[[s.first_output_frame,s.end_output_frame] for s in window],states=[],errors=[],
                original_role_programme_sha256=[role_programme_refs[s.first_output_frame,s.end_output_frame] for s in window],
                explicit_global_mutable_unit_ids=[owned_units[s.first_output_frame,s.end_output_frame].unit_id
                    for s in window[1:-1]],
                global_original_numeric_programme_sha256=owned_program.program_sha256,
                global_mixed_role_camera_owner_implemented=False,
                observed_wide_repair_priority=observed_wide_repair_priority(
                    context.incumbent.visual_timeline,window[1:-1]))
            numeric_inventory.append(record)
            try:
                from .group_scene_intent import declared_shot_intent
                intents = tuple(declared_shot_intent(context, shot,
                    endpoint_scope='exit' if index==0 else 'entry' if index==len(window)-1 else 'whole')
                    for index,shot in enumerate(window))
                roles = tuple(GroupOutputRole(intent['role']) for intent in intents)
                for intent in intents:
                    sensors.shot_intents[intent['first_output_frame'],intent['end_output_frame'],intent['endpoint_scope']] = intent
                baseline=source._states(plan,context.domain,window,roles,fixed_subcell_views=True)
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
        probe_batches=fixed_window_probe_batches(prepared,batch_size=MAX_FIXED_WINDOW_PROBES)
        frontier=tuple(item for batch in probe_batches for item in batch)
        scheduled=tuple((wi,item,rolls) for rolls in ((None,),VARIANTS)
            for wi,item in frontier)
        graph_windows=0
        evaluated_hypothesis_frontiers=set()
        for wi,(baseline,raw_slots,numeric),rolls in scheduled:
            active_window=wi;active_graph=None
            if monotonic()+COMPLETE_REFRESH_RESERVE_SECONDS >= context.deadline:
                attempts.append(dict(stage='reserve',reason='complete_refresh_reserve_preserved'));break
            if session.budget.counts['states']>=session.budget.limits['states']:
                attempts.append(dict(stage='state_frontier_exhausted',numeric_windows=len(prepared)));break
            if graph_windows>=MAX_WINDOWS:
                attempts.append(dict(stage='graph_window_budget',evaluated=graph_windows));break
            if session.budget.counts['edges'] >= session.budget.limits['edges']:
                attempts.append(dict(stage='edge_frontier_exhausted',unmeasured_is_unknown=True));break
            start_counts=dict(session.budget.counts)
            # A fixed endpoint with no measured path makes this whole window
            # unavailable regardless of donor. Test it before decoding/inferring
            # all eight sources. Their numeric options remain in the inventory;
            # skipped native observations are not relabelled negative evidence.
            def scope_for(index):
                return 'exit' if index==0 else 'entry' if index==len(baseline)-1 else 'whole'
            def bound(state,scope='whole'):
                return tuple(sensors.bind(state,h) for h in sensors.discover(state,rolls=rolls,endpoint_scope=scope))
            original_lefts=bound(baseline[0],'exit')
            original_rights=bound(baseline[-1],'entry')
            # Fixed R needs distinct ENTRY hypotheses; fixed L distinct EXIT.
            lefts=fixed_endpoint_inventory(original_lefts,sensors=sensors,entry=False)
            rights=fixed_endpoint_inventory(original_rights,sensors=sensors,entry=True)
            if not lefts or not rights:
                attempts.append(dict(stage='scene_fixed_hypotheses',window=wi,
                    analysis_rolls=list(rolls),reason='all_fixed_endpoint_paths_unavailable',
                    original_mutable_native_observations_skipped=True,
                    donor_native_observations_skipped=True,donor_absence_inferred=False))
                continue
            # Discovery is not enough: clipping and required arm/support paths
            # on the unchanged endpoint must fail BEFORE mutable-original work,
            # not only before donor work. Each fixed endpoint is checked under
            # its own complete ORIGINAL unit, including every conditional
            # original hypothesis (not only the allocated two search states).
            # A one-unit programme grants no mutable/donor/full-window duty.
            # Prior versions remain archived; no native/time budget is reset.
            from .group_scene_original_duties import freeze_original_duty_program
            original_plan_identity=family.digest(family.plan_payload(plan))
            fixed_programs=[]
            def preflight_original_fixed(original_values,values,outgoing):
                try:
                    program=freeze_original_duty_program(sensors,(original_values,),
                        original_plan_identity=original_plan_identity)
                    sensors.install_original_duty_program(program)
                except JointGroupBudgetExhausted:
                    raise
                except ValueError as exc:
                    attempts.append(dict(stage='original_fixed_duty_freeze',window=wi,
                        outgoing=outgoing,error=str(exc),
                        original_mutable_native_observations_skipped=True,
                        donor_native_observations_skipped=True,
                        original_ambiguity_is_not_a_waiver=True))
                    return ()
                fixed_programs.append(program)
                probe_record=dict(stage='fixed_endpoint_inventory',window=wi,outgoing=outgoing,
                    original_mutable_native_observations_skipped=True,
                    donor_native_observations_skipped=True)
                attempts.append(probe_record)  # Retained BEFORE any endpoint work.
                return measure_fixed_endpoint_inventory(values,sensors=sensors,
                    outgoing=outgoing,record=probe_record)
            lefts=preflight_original_fixed(original_lefts,lefts,True)
            if lefts:
                rights=preflight_original_fixed(original_rights,rights,False)
            else:
                rights=()
            if not lefts or not rights:
                attempts.append(dict(stage='scene_fixed_hypotheses',window=wi,
                    analysis_rolls=list(rolls),reason='allocated_fixed_endpoint_paths_unavailable',
                    original_mutable_native_observations_skipped=True,
                    donor_native_observations_skipped=True,
                    donor_absence_inferred=False));continue
            # BOTH unchanged incident endpoints survived. Freeze ALL original
            # units before ANY donor query; the preflight programme cannot
            # substitute for this complete programme or waive a mutable duty.
            baseline_hypotheses=[original_lefts,
                *(bound(s,scope_for(i)) for i,s in enumerate(baseline[1:-1],1)),
                original_rights]
            try:
                original_program=freeze_original_duty_program(sensors,baseline_hypotheses,
                    original_plan_identity=original_plan_identity)
                if any(family.digest(original_program['units'][index])!=
                        family.digest(program['units'][0])
                        for index,program in zip((0,-1),fixed_programs,strict=True)):
                    raise ValueError('original fixed duties changed between preflight and complete freeze')
                sensors.install_original_duty_program(original_program)
            except JointGroupBudgetExhausted:
                raise
            except ValueError as exc:
                attempts.append(dict(stage='original_duty_freeze',window=wi,error=str(exc),
                    donor_native_observations_skipped=True,original_ambiguity_is_not_a_waiver=True))
                continue
            fixed_signature=(wi,tuple(v.key for v in lefts),tuple(v.key for v in rights))
            if all(s.role==GroupOutputRole.COMPOSITION for s in baseline):
                if fixed_signature in evaluated_hypothesis_frontiers:
                    attempts.append(dict(stage='unchanged_composition_frontier_skipped',window=wi))
                    continue
                evaluated_hypothesis_frontiers.add(fixed_signature)
            graph_windows+=1
            session.budget.begin_frontier(label=wi,
                maximum_states=source.WORK_LIMITS['maximum_states'],
                maximum_edges=source.WORK_LIMITS['maximum_edges'])
            # All numerically available sources are observed before choosing a
            # donor/camera, but only after the shared incident endpoints exist.
            all_native=tuple((s,'whole') for group in raw_slots for s in group)
            for state,scope in all_native:
                sensors.discover(state,rolls=rolls,endpoint_scope=scope)
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
                                first,entry_envelopes=sensors.mutable_camera_inputs(value,value.first_output_frame)
                                last,exit_envelopes=sensors.mutable_camera_inputs(value,value.end_output_frame-1)
                                a,ta,_=source._at(value,value.first_output_frame);b,tb,_=source._at(value,value.end_output_frame-1)
                                boxes=(source._base_box(_native_target(first),a,ta),source._base_box(_native_target(last),b,tb))
                                envelopes=(tuple(source._base_box(box,a,ta) for box in entry_envelopes),
                                    tuple(source._base_box(box,b,tb) for box in exit_envelopes))
                                proposal=source.propose_endpoint_focal_face_roi_camera(source_face_boxes=boxes,
                                    fixed_neighbor_face_boxes=targets,frame_count=len(value.native_frame_indices),fps=plan.fps,
                                    required_endpoint_envelopes=envelopes if entry_envelopes or exit_envelopes else None,
                                    maximum_zoom=(FaceMatchConfig().maximum_silhouette_zoom
                                        if value.role==GroupOutputRole.FORMATION else 2.))
                                camera=source._hypothesis(context,baseline[slot],value.eligible_source_index,proposal.track)
                                cameras.append(sensors.bind(camera,sensors.hypothesis(value)))
                            except JointGroupBudgetExhausted:
                                raise
                            except ValueError as exc:
                                if len(camera_errors)<source.MAX_RECORDED_FAILURES:camera_errors.append(str(exc))
                    # Native then camera, round-robin over SOURCE, not the first
                    # source's many head/camera variants monopolizing the cap.
                    by_source.append(camera_balanced_source_frontier(values,cameras))
                remaining=session.budget.frontier_remaining('states')-len(lefts)-len(rights)
                # At most 24 states per graph, three bounded request frontiers.
                # Discovery still preserves all the
                # omitted hypotheses and numeric source options in its archive.
                quota=max(0,remaining//len(raw_slots))
                groups.append(round_robin(by_source,quota))
            if any(not group for group in groups):
                if remaining//len(raw_slots)<=0:
                    attempts.append(dict(stage='state_budget',window=wi,unexplored=True));break
                attempts.append(dict(stage='scene_mutable_hypotheses_unavailable',window=wi,
                    analysis_rolls=list(rolls),source_options_by_slot=[len(g) for g in raw_slots],
                    bound_state_counts_by_slot=[len(g) for g in groups],
                    numeric_inventory_retained=True))
                continue
            all_states=tuple(dict((v.key,v) for v in (*lefts,*(s for g in groups for s in g),*rights)).values())
            for state in all_states:session.budget.spend('states')
            edges={};known={}
            active_graph=dict(window=wi,window_native_start_counts=start_counts,
                edge_observations=[],baseline_components=known,camera_errors=camera_errors)
            def read(a,b):
                key=a.key,b.key
                if key not in edges:
                    if session.budget.frontier_remaining('edges')<=0:
                        raise _EdgeFrontierFull('bounded edge frontier reached')
                    session.budget.spend('edges')
                    evidence=compare_scene_edge(sensors,a,b)
                    evidence['source_relation']=relation(context.source_derivative_binding,a.eligible_source_index,b.eligible_source_index)
                    # Unknown/duplicate may be observed, but cannot masquerade
                    # as a verified distinct-camera cut in the selected graph.
                    if evidence['source_relation'] != DERIVATIVE_DISTINCT:evidence['available']=False
                    edges[key]=evidence
                    active_graph['edge_observations'].append(evidence)
                return edges[key]
            composition_window=any(v.role==GroupOutputRole.COMPOSITION for v in baseline)
            composition_reference=None
            if composition_window:
                from .group_scene_editorial_policy import coherent_path_reference,evaluate_composition_path
                composition_reference=coherent_path_reference(baseline_hypotheses,read)
                known={i:(row if row is not None else dict(available=False,metrics=None))
                    for i,row in enumerate(composition_reference['edges'])}
            else:
                # Freeze one observed incumbent comparison. Combining different
                # alternatives' component minima invents an unattainable reference.
                for index in range(len(baseline)-1):
                    values=[]
                    for a,b in product(baseline_hypotheses[index],baseline_hypotheses[index+1]):
                        if len(values)>=4:break
                        values.append(read(a,b))
                    # Cohort geometry is observed independent of focal assignment.
                    # A disputed focal hypothesis is NOT an unconditional incumbent
                    # fact. In particular its minimum conditional cost is not zero
                    # uncertainty, nor a licence to erase known cohort components.
                    unique=True
                    for old_index in (index,index+1):
                        old_state=baseline[old_index]
                        try:
                            for tick in scene_anchor_ticks(old_state.first_output_frame,
                                    old_state.end_output_frame,scope_for(old_index)):
                                source._dominant(sensors.raw(old_state,tick,None),old_state.role)
                        except JointGroupBudgetExhausted:raise
                        except ValueError:unique=False
                    known[index]=coherent_baseline(values,focal_assignment_unique=unique)
            paths=[]
            for path in product(lefts,*groups,rights):
                # An unchanged-source path cannot become this owner's result;
                # do not let it consume scarce edge/dense/family refresh work.
                if not any(old.eligible_source_index!=new.eligible_source_index
                        for old,new in zip(baseline[1:-1],path[1:-1],strict=True)):
                    continue
                measured=[];decision=None
                try:
                    for index,(a,b) in enumerate(zip(path,path[1:])):
                        row=read(a,b)
                        if composition_window:
                            if not row.get('available'): break
                        elif not preserves_known(known[index],row):break
                        measured.append(row)
                except _EdgeFrontierFull:
                    attempts.append(dict(stage='bounded_edge_frontier',window=wi,
                        retained_complete_paths=len(paths),unmeasured_edges_are_not_negative=True))
                    break
                if len(measured)!=len(path)-1:continue
                if composition_window:
                    decision=evaluate_composition_path(composition_reference,measured)
                    if not decision['admitted']:continue
                    score=decision['candidate_cost']
                else:
                    gain=any(has_editorial_gain(known[i],m) for i,m in enumerate(measured))
                    if not gain:continue
                    score=sum(row['framing_cost'] for row in measured)
                paths.append((score,tuple(s.key for s in path),path,measured,decision))
            for _,_,path,measured,decision in sorted(paths,key=lambda p:p[:2]):
                alternative=dict(states=[s.to_dict() for s in path],status='dense_pending');alternatives.append(alternative)
                try:
                    assembled,changed=source._assembler(path)(plan=plan,domain=context.domain,path=path)
                    if not any(assembled.segments[i].source_index!=plan.segments[i].source_index for i in changed):continue
                    # Transitional consumer: the existing local sensor path is
                    # still 3/4 states. Its numeric proposal must ALSO close on
                    # explicit global units before dense/completion. The new
                    # assembler can union arbitrary compatible mutable units;
                    # this bridge is NOT a global search/completion owner.
                    owned_intervals=own_scene_intervals(owned_program,
                        mutable_unit_ids=tuple(numeric['explicit_global_mutable_unit_ids']),
                        maximum_mutable_units=2,maximum_mutable_frames=2*source.MAX_MUTABLE_FRAMES)
                    owned_cameras=tuple(camera_unit_from_plan(program=owned_program,unit_id=unit_id,
                        plan=assembled,domain=context.domain) for unit_id in owned_intervals.mutable_unit_ids)
                    numeric_union=assemble_owned_scene_union(context=context,domain=context.domain,
                        allocation=owned_intervals,cameras=owned_cameras)
                    if numeric_union.plan!=assembled or numeric_union.changed_segment_indices!=changed:
                        raise ValueError('scene local proposal differs from its explicit global owned numeric union')
                    session.budget.spend('dense_paths')
                    dense=sensors.dense(path)
                    # Per-tick known incumbent parts are preserved independent
                    # of whether its old dominant face was decisive.
                    preserved=preserve_incumbent_required_parts(sensors,baseline,path,dense)
                    search=dict(producer=CONTRACT,generated_numeric_states=numeric['states'],
                        numeric_inventory=numeric_inventory,prior_solo_success_required=False,
                        fixed_geometry_unchanged=True,baseline_components={str(k):v for k,v in known.items()},
                        selected_scene_hypotheses=[s.editorial_contract_sha256 for s in path],
                        composition_reference=composition_reference,editorial_decision=decision,
                        required_semantics_and_editorial_utility_separated=composition_window,
                        source_derivative_binding=context.source_derivative_binding,
                        incumbent_known_parts=preserved,camera_errors=camera_errors,
                        original_duties_frozen_before_donor_discovery=True,
                        original_duty_program_sha256=original_program['program_sha256'],
                        explicit_global_numeric_ownership=owned_intervals.to_dict(),
                        global_numeric_union_plan_sha256=family.digest(numeric_union.plan.to_json()),
                        global_numeric_union_dependency_sha256=numeric_union.dependency_support['support_sha256'],
                        global_numeric_camera_sha256=[camera.camera_sha256 for camera in numeric_union.cameras],
                        global_search_and_completion_owner_implemented=False,
                        candidate_search_exhaustive=False,rotated_pass_exhaustive=False,
                        source_balanced_state_allocation=True,full_epoch_retained_family_owner_implemented=False,
                        within_source_frontier_policy='native_fitted_camera_interleave_v1',
                        window_frontier_policy='observed_wide_repair_then_time_stratified_upright_v2',
                        selected_frontier_window_indices=[i for i,_ in frontier],
                        fixed_endpoints_before_donor_observations=True,
                        fixed_endpoints_before_original_mutable_observations=True,
                        fixed_preflight_programme_sha256=[p['program_sha256'] for p in fixed_programs],
                        fixed_preflight_units_equal_complete_original_units=True,
                        fixed_window_probe_limit=len(frontier),fixed_probe_batch_size=MAX_FIXED_WINDOW_PROBES,
                        fixed_probe_batches_share_native_session=True,
                        fixed_probe_batch_count=len(probe_batches),
                        evaluated_graph_windows=graph_windows,
                        bounded_edge_frontier_attempts=[a for a in attempts if a.get('stage')=='bounded_edge_frontier'],
                        state_edge_frontiers_explicit=True,
                        per_graph_state_limit=source.WORK_LIMITS['maximum_states'],
                        per_graph_edge_limit=source.WORK_LIMITS['maximum_edges'],
                        state_edge_prefix_starvation_removed=True,native_caps_not_multiplied=True,
                        request_wide_work_accounting=False,window_fairness_guaranteed=False,
                        reserve_check_basis="before_window_only_not_hard_phase_partition")
                    from .group_scene_selected_replay import CONTRACT as PROGRAM_CONTRACT
                    search['scene_replay_program']=dict(contract=PROGRAM_CONTRACT,
                        selected_states=[s.to_dict() for s in path],baseline_states=[s.to_dict() for s in baseline],
                        baseline_hypothesis_keys=[[s.editorial_contract_sha256 for s in group]
                            for group in baseline_hypotheses],analysis_rolls=list(rolls),
                        original_duty_program=original_program,
                        original_role_programme_sha256=numeric['original_role_programme_sha256'],
                        composition_window=composition_window,quality_approved=False)
                    result=source.complete_source_candidate(context,path,sensors,session,assets,dense,measured,search)
                    alternative['status']='complete_unapproved'
                    return result
                except JointGroupBudgetExhausted as exc:
                    alternative.update(status='budget_unverified',error=str(exc),
                        unexecuted_measurements_are_unknown=True)
                    raise
                except (ValueError,RuntimeError,OSError) as exc:
                    alternative.update(status='rejected',error=str(exc))
            attempts.append(dict(stage='scene_graph',window=wi,state_keys=[s.key for s in all_states],
                path_count=len(paths),edge_observations=list(edges.values()),baseline_components=known,
                window_native_start_counts=start_counts,
                window_native_end_counts=dict(session.budget.counts),camera_errors=camera_errors))
        return rejected('no complete bounded scene/source/whole-camera family candidate')
    except JointGroupBudgetExhausted as exc:
        counts={} if session is None else session.budget.counts
        measured=any(counts.get(k,0) for k in ('new_decodes','model_calls','pose_comparisons',
            'region_flow_feature_updates','association_nodes','association_edges'))
        attempts.append(dict(stage='budget_interrupted',window=active_window,
            outcome='BUDGET_PARTIAL' if measured else 'BUDGET_NOT_STARTED',
            retained_graph=active_graph,native_work_at_interrupt=dict(counts),
            unexecuted_measurements_are_unknown=True,new_budget_retry_started=False))
        return rejected(type(exc).__name__+': '+str(exc),error_type=type(exc).__name__,
            outcome='BUDGET_PARTIAL' if measured else 'BUDGET_NOT_STARTED',
            partial=getattr(exc,'partial_evidence',None))
    except (ValueError,RuntimeError,OSError) as exc:
        # The legacy sibling/fixed-camera fallback may also fail. Preserve this
        # exact native/lease/asset cause inside its evidence, not only in the
        # transient rejection object that the final fallback can replace.
        return rejected(type(exc).__name__+': '+str(exc),error_type=type(exc).__name__)
    finally:
        if session is not None:session.close()
