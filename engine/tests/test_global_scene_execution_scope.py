"""Global execution ownership; explicit NONMEDIA numeric/descriptor fixtures."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from engine.tests.test_scene_owned_program import fixture
from crosscut.group_scene_owned_program import (freeze_owned_scene_program,own_scene_intervals,
    camera_unit_from_plan,assemble_owned_scene_union)
from crosscut.group_scene_execution_scope import (execution_path_from_union,execution_path_from_payload,
    mutable_states,incident_pairs)
from crosscut.render import EditorialTransformTrack,EditorialTransformKeyframe


def world(indices=(0,3,10),length=12):
    context,domain=fixture(length=length)
    for row in context.incumbent.face_match_plan.segments:row.alignment_subject='face'
    program=freeze_owned_scene_program(context,domain)
    allocation=own_scene_intervals(program,mutable_unit_ids=tuple(program.units[i].unit_id for i in indices),
        maximum_mutable_units=6,maximum_mutable_frames=540)
    plan=context.incumbent.render_plan;segments=list(plan.segments)
    track=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.1),))
    cameras=[]
    for unit_id in allocation.mutable_unit_ids:
        unit=next(u for u in program.units if u.unit_id==unit_id)
        shot=unit.role_program['original_shot']
        for i in range(shot['first_segment_index'],shot['last_segment_index']+1):
            segments[i]=replace(segments[i],editorial_transform=track)
    selected=replace(plan,segments=tuple(segments))
    for unit_id in allocation.mutable_unit_ids:
        cameras.append(camera_unit_from_plan(program=program,unit_id=unit_id,plan=selected,domain=domain))
    union=assemble_owned_scene_union(context=context,domain=domain,allocation=allocation,cameras=tuple(cameras))
    return context,domain,union,execution_path_from_union(context,domain,union)


def test_distant_start_and_final_mutables_have_explicit_scope_and_no_gap_edges():
    context,domain,union,path=world()
    assert len(path)>4
    assert path.mutable_indices[0]==0 and path.mutable_indices[-1]==len(path)-1
    assert len(mutable_states(path))==3
    assert all(a.end_output_frame==b.first_output_frame for a,b in incident_pairs(path))
    assert len(incident_pairs(path))==4
    assert path.assemble(plan=context.incumbent.render_plan,domain=domain,path=path)==(
        union.plan,union.changed_segment_indices)
    assert path.to_dict()['positional_fixed_endpoints_assumed'] is False
    assert path.to_dict()['quality_approved'] is False


def test_three_consecutive_mutables_complete_one_numeric_scope_not_middle_two():
    context,domain,union,path=world(indices=(2,3,4))
    assert len(path)==5 and len(mutable_states(path))==3
    assert len(incident_pairs(path))==4
    restored=execution_path_from_payload(original=context.incumbent.render_plan,proposed=union.plan,
        domain=domain,payload=path.to_dict(),rows=[s.to_dict() for s in path])
    assert restored==path
    restored.validate_external(context,domain)


def native_boundary_world():
    """Explicit numeric native cut, not a detector or human-media fixture."""
    from engine.tests.test_scene_source_owner import scene_derivative_fixture
    from crosscut.group_scene_source_owner import freeze_derivatives
    context,domain=fixture(native_cut=2.)
    for row in context.incumbent.face_match_plan.segments:row.alignment_subject='face'
    program=freeze_owned_scene_program(context,domain)
    unit=program.units[1]
    allocation=own_scene_intervals(program,mutable_unit_ids=(unit.unit_id,),
        maximum_mutable_units=6,maximum_mutable_frames=540)
    original=context.incumbent.render_plan;segments=list(original.segments)
    # Maintain the renderer's unchanged tier-D continuity. No gate is waived.
    segments[1]=replace(segments[1],editorial_transform=EditorialTransformTrack((
        EditorialTransformKeyframe(0.,1.1),EditorialTransformKeyframe(29/30.,1.),
        EditorialTransformKeyframe(1.,1.))))
    selected=replace(original,segments=tuple(segments))
    union=assemble_owned_scene_union(context=context,domain=domain,allocation=allocation,cameras=(
        camera_unit_from_plan(program=program,unit_id=unit.unit_id,plan=selected,domain=domain),))
    path=execution_path_from_union(context,domain,union)
    binding=freeze_derivatives(scene_derivative_fixture(),domain,
        tuple((str(p),0,0) for p in domain.eligible_sources))
    return context,domain,path,binding


def test_preserved_same_file_native_cut_is_admitted_without_distinct_source_credit():
    from crosscut.group_scene_physical_boundaries import PreparedPhysicalBoundaryPolicy
    from crosscut.source_derivative_diversity import DERIVATIVE_CONFIRMED
    from crosscut.group_scene_source_owner import validate_bound_derivative_pairs
    context,domain,path,binding=native_boundary_world()
    pairs=incident_pairs(path)
    with pytest.raises(ValueError,match='duplicate/UNKNOWN'):
        validate_bound_derivative_pairs(binding,tuple((a.eligible_source_index,b.eligible_source_index) for a,b in pairs))
    rows=PreparedPhysicalBoundaryPolicy(original=path.original_plan,observed=path.numeric_plan,
        domain=domain,derivative_binding=binding).require(pairs)
    assert rows[0]['distinct_source_pair_confirmed'] is True
    assert rows[1]['output_tick']==60
    assert rows[1]['source_relation']==DERIVATIVE_CONFIRMED
    assert rows[1]['preserved_original_native_cut'] is True
    assert rows[1]['incident_source_sampler_preserved'] is True
    assert rows[1]['distinct_source_pair_confirmed'] is False
    assert rows[1]['new_distinct_source_credit'] is False
    assert rows[1]['native_motion_observed'] is False
    assert rows[1]['quality_approved'] is False


def test_canonical_or_role_seam_cannot_become_an_actual_same_source_camera_cut():
    from crosscut.group_scene_physical_boundaries import PreparedPhysicalBoundaryPolicy
    from crosscut.group_joint_source_camera import joint_state_from_render_plan,GroupOutputRole
    context,domain=fixture()
    plan=context.incumbent.render_plan
    states=tuple(joint_state_from_render_plan(plan=plan,domain=domain,slot=1,
        first_output_frame=first,end_output_frame=end,role=GroupOutputRole.FOCAL_HEAD)
        for first,end in ((30,60),(60,90)))
    policy=PreparedPhysicalBoundaryPolicy(original=plan,observed=plan,domain=domain,derivative_binding=None)
    row=policy.classify(*states)
    assert row['observed_events']==[] and row['admitted'] is False
    with pytest.raises(ValueError,match='global physical cut'):policy.require((states,))
    measured=dict(available=True,metrics=dict(scale=.2))
    assert policy.attach(measured,*states)['metrics']==dict(scale=.2)
    assert measured['available'] is False


def test_unknown_source_switch_cannot_use_the_native_camera_boundary_exception():
    from crosscut.group_scene_physical_boundaries import PreparedPhysicalBoundaryPolicy
    from crosscut.group_scene_source_owner import relation
    from crosscut.group_performer_family_owner import digest
    from crosscut.source_derivative_diversity import DERIVATIVE_UNCERTAIN
    context,domain,path,binding=native_boundary_world()
    binding=deepcopy(binding);binding['audit']['pairs']=[]
    payload=dict(binding['audit']);payload.pop('audit_sha256')
    binding['audit']['audit_sha256']=digest(payload)
    assert relation(binding,0,1)==DERIVATIVE_UNCERTAIN
    policy=PreparedPhysicalBoundaryPolicy(original=path.original_plan,observed=path.numeric_plan,
        domain=domain,derivative_binding=binding)
    a,b=incident_pairs(path)[0]
    assert policy.classify(a,b)['admitted'] is False
    with pytest.raises(ValueError,match='global physical cut'):policy.require(incident_pairs(path))


def test_equal_rounded_native_ids_cannot_hide_a_changed_source_request_clock():
    from crosscut.group_scene_physical_boundaries import PreparedPhysicalBoundaryPolicy
    from crosscut.group_joint_source_camera import joint_state_from_render_plan,GroupOutputRole
    context,domain,path,binding=native_boundary_world()
    original=path.original_plan
    selected=replace(original,segments=tuple(replace(s,source_start=s.source_start+.001)
        if i in (1,2) else s for i,s in enumerate(original.segments)))
    states=tuple(joint_state_from_render_plan(plan=selected,domain=domain,slot=1,
        first_output_frame=first,end_output_frame=end,role=GroupOutputRole.FOCAL_HEAD)
        for first,end in ((30,60),(60,90)))
    policy=PreparedPhysicalBoundaryPolicy(original=original,observed=selected,domain=domain,derivative_binding=binding)
    row=policy.classify(*states)
    assert row['observed_events']==row['original_events']
    assert row['incident_source_sampler_preserved'] is False
    assert row['admitted'] is False


@pytest.mark.parametrize('field',['native_event','tick','source_relation','source_clock','distinct_credit','admitted',
    'source_index_bool','source_index_float','drop'])
def test_rehashed_global_dense_cannot_author_a_native_boundary_policy(field):
    from crosscut.group_scene_physical_boundaries import PreparedPhysicalBoundaryPolicy,validate_dense_physical_boundaries
    context,domain,path,binding=native_boundary_world()
    rows=PreparedPhysicalBoundaryPolicy(original=path.original_plan,observed=path.numeric_plan,
        domain=domain,derivative_binding=binding).require(incident_pairs(path))
    dense=dict(global_execution_scope=path.to_dict(),
        global_execution_original_numeric_plan=path.original_plan.to_json(),
        global_execution_selected_numeric_plan=path.numeric_plan.to_json(),source_domain=domain.to_dict(),
        source_derivative_binding=binding,physical_boundary_policies=rows,
        derivative_window_source_indices=[s.eligible_source_index for s in path],
        incident_projected_observations=[dict(physical_boundary_policy=row,source_relation=row['source_relation']) for row in rows])
    assert validate_dense_physical_boundaries(dense)==rows
    dense=deepcopy(dense);row=dense['physical_boundary_policies'][1]
    if field=='native_event':row['observed_events'][0]['native_cut_frame']+=1
    elif field=='tick':row['output_tick']+=1
    elif field=='source_relation':row['source_relation']='verified_distinct_camera'
    elif field=='source_clock':row['observed_incident_sampler_sha256']='0'*64
    elif field=='distinct_credit':row['new_distinct_source_credit']=True
    elif field=='admitted':row['admitted']=1
    elif field=='source_index_bool':dense['derivative_window_source_indices'][1]=True
    elif field=='source_index_float':dense['derivative_window_source_indices'][1]=1.
    else:dense.pop('physical_boundary_policies')
    with pytest.raises(ValueError,match='physical boundary'):validate_dense_physical_boundaries(dense)


@pytest.mark.parametrize('value',[True,1.])
def test_derivative_eligible_indices_cannot_use_bool_float_aliases(value):
    from crosscut.group_scene_physical_boundaries import PreparedPhysicalBoundaryPolicy
    context,domain,path,binding=native_boundary_world()
    binding=deepcopy(binding);binding['eligible_request_indices'][1]=value
    with pytest.raises(ValueError,match='eligible source domain'):
        PreparedPhysicalBoundaryPolicy(original=path.original_plan,observed=path.numeric_plan,
            domain=domain,derivative_binding=binding)


def test_same_file_source_seek_is_not_a_preserved_native_camera_cut():
    from crosscut.group_scene_physical_boundaries import PreparedPhysicalBoundaryPolicy
    from crosscut.group_joint_source_camera import joint_state_from_render_plan,GroupOutputRole
    from crosscut.render import RenderTransition
    context,domain,path,binding=native_boundary_world()
    original=path.original_plan;segments=list(original.segments);transitions=list(original.transitions)
    segments[2]=replace(segments[2],source_start=segments[2].source_start+.1)
    transitions[1]=RenderTransition(tier='C')
    selected=replace(original,segments=tuple(segments),transitions=tuple(transitions))
    states=tuple(joint_state_from_render_plan(plan=selected,domain=domain,slot=1,
        first_output_frame=first,end_output_frame=end,role=GroupOutputRole.FOCAL_HEAD)
        for first,end in ((30,60),(60,90)))
    row=PreparedPhysicalBoundaryPolicy(original=original,observed=selected,domain=domain,
        derivative_binding=binding).classify(*states)
    assert row['observed_events'][0]['kind']=='same_source_clock_jump'
    assert row['admitted'] is False
    assert row['new_distinct_source_credit'] is False


@pytest.mark.parametrize('change',['mutable_indices','pairs','ids','caps','roles','alignment','state','authority_type'])
def test_rehashed_scope_cannot_change_external_ownership_or_alignment(change):
    context,domain,union,path=world();raw=deepcopy(path.to_dict());rows=[s.to_dict() for s in path]
    if change=='mutable_indices':raw['mutable_state_indices']=[1]
    elif change=='pairs':raw['incident_state_pair_indices']=[[0,len(path)-1]]
    elif change=='ids':raw['scoped_unit_ids'].reverse()
    elif change=='caps':raw['allocation']['maximum_mutable_units']=7
    elif change=='roles':raw['original_owned_program']['units'][0]['original_role_program']['epochs'][0]['role']='observed_body_composition'
    elif change=='alignment':raw['source_domain']['audio_offsets'][0]+=1.
    elif change=='authority_type':raw['quality_approved']=0
    else:rows[0]['key_preimage']['slot']=0
    with pytest.raises(ValueError):execution_path_from_payload(original=context.incumbent.render_plan,
        proposed=union.plan,domain=domain,payload=raw,rows=rows)


def test_global_numeric_states_preserve_complete_preimages_across_json_transport():
    context,domain,union,path=world()
    scope,rows=json.loads(json.dumps([path.to_dict(),[s.to_dict() for s in path]]))
    restored=execution_path_from_payload(original=context.incumbent.render_plan,
        proposed=union.plan,domain=domain,payload=scope,rows=rows)
    assert restored==path


@pytest.mark.parametrize('change',['gap_pair','drop_neighbour','order','role_scope','authority_type'])
def test_global_original_duty_scope_cannot_invent_or_drop_original_units(change):
    from crosscut.group_scene_original_duties import _global_scope_inventory
    context,domain,union,path=world()
    scope=path.to_dict()
    assert len(_global_scope_inventory(scope))==len(path)
    if change=='gap_pair':scope['incident_state_pair_indices']=[[1,len(path)-2]]
    elif change=='drop_neighbour':scope['unit_intervals'].pop(1)
    elif change=='order':scope['scoped_unit_ids'].reverse()
    elif change=='role_scope':scope['endpoint_scopes'][0]='exit'
    else:scope['quality_approved']=0
    with pytest.raises(ValueError):_global_scope_inventory(scope)


def test_legacy_full_receipt_cannot_omit_edges_even_when_unit_motion_consumer_has_no_edge_slice():
    from crosscut import group_source_changing_owner as source
    dense=dict(contract=source.DENSE_CONTRACT,state_keys=['left','mutable','right'],
        every_mutable_output_tick_observed=True,face_roi_is_not_physical_head_extent=True,
        semantic_raw_v2_admission=False,cross_camera_identity_certified=False,
        body_phase_certified=False,decoded_output_quality_certified=False,quality_approved=False,
        mutable_shots=[dict(rows=[dict(whole_head_extent_certified=False,
            cross_camera_identity_certified=False,body_phase_certified=False)])],
        incident_projected_observations=[])
    with pytest.raises(ValueError,match='complete native family lost its actual incident'):
        source.validate_dense_uncertainty(dense)


def test_numeric_scope_does_not_silently_admit_mixed_original_role_epochs():
    context,domain,union,path=world(indices=(2,))
    context.incumbent.face_match_plan.segments[1].alignment_subject=None
    context.incumbent.face_match_plan.segments[2].alignment_subject=None
    with pytest.raises(ValueError,match='external original'):
        path.validate_external(context,domain)


def no_look_fixture():
    from crosscut import group_performer_family_owner as family
    context,domain=fixture()
    old=context.incumbent;old.boundary_look_match=None
    old.to_dict=lambda:dict(boundary_look_match=None,explicit_nonmedia_fixture=True)
    plan,look,audit=family.refresh_look(old,old.render_plan,(0,1),None,domain)
    receipt=dict(look_refresh=audit,incident_edges=[0,1],
        plan_sha256=family.digest(family.plan_payload(plan)),
        original_plan_sha256=family.digest(family.plan_payload(old.render_plan)),
        original_metadata_sha256=family.digest(family.metadata_payload(old)))
    return old,plan,receipt


def test_original_boundary_look_absence_preserves_plan_without_any_native_or_color_work():
    from crosscut import group_performer_family_owner as family
    old,plan,receipt=no_look_fixture()
    assert plan is old.render_plan
    assert family.retained_applied_look_edges(old)==()
    family.validate_look_refresh(plan,None,receipt,original_bundle=old)
    assert receipt['look_refresh']['matched_production_renderer_used'] is False
    assert receipt['look_refresh']['retained_family']['quality_approved'] is False


@pytest.mark.parametrize('change',['renderer_track','disabled_metadata','enabled_metadata',
    'final_mode','original_mode','claimed_encoder','fabricated_verified','dropped_binding','incident_scope'])
def test_original_absence_cannot_be_rehashed_into_disabled_stripped_or_fabricated_look(change):
    from crosscut import group_performer_family_owner as family
    from crosscut.render import BoundaryLookTrack,BoundaryLookKeyframe
    old,plan,receipt=no_look_fixture();receipt=deepcopy(receipt);look=None
    audit=receipt['look_refresh'];binding=audit['retained_family']
    if change=='renderer_track':
        track=BoundaryLookTrack((BoundaryLookKeyframe(0.,brightness=.02),))
        plan=replace(plan,segments=(replace(plan.segments[0],boundary_look_track=track),*plan.segments[1:]))
    elif change=='disabled_metadata':look=dict(enabled=False,segment_tracks=[None]*len(plan.segments))
    elif change=='enabled_metadata':look=dict(enabled=True,segment_tracks=[None]*len(plan.segments))
    elif change=='final_mode':binding['final']['mode']='ENABLED'
    elif change=='original_mode':binding['original']['mode']='ENABLED'
    elif change=='claimed_encoder':audit['matched_production_renderer_used']=True
    elif change=='fabricated_verified':audit['verified']=dict(passed=True)
    elif change=='dropped_binding':audit.pop('retained_family')
    else:audit['incident_edges_without_boundary_look']=[0]
    # A matching new plan digest cannot replace external original ownership.
    receipt['plan_sha256']=family.digest(family.plan_payload(plan))
    if 'retained_family' in audit:binding['final_plan_sha256']=receipt['plan_sha256']
    with pytest.raises(ValueError):family.validate_look_refresh(plan,look,receipt,original_bundle=old)


def test_absent_metadata_with_live_track_is_rejected_before_numeric_dependency_assembly():
    from crosscut import group_performer_family_owner as family
    from crosscut.render import BoundaryLookTrack,BoundaryLookKeyframe
    context,domain,union,path=world()
    context.incumbent.boundary_look_match=None
    original=context.incumbent.render_plan
    track=BoundaryLookTrack((BoundaryLookKeyframe(0.,brightness=.02),))
    context.incumbent.render_plan=replace(original,segments=(
        replace(original.segments[0],boundary_look_track=track),*original.segments[1:]))
    with pytest.raises(ValueError,match='metadata absent but renderer tracks remain'):
        family.retained_applied_look_edges(context.incumbent)


def test_descriptor_consumer_assembles_union_first_and_refreshes_only_disjoint_closure():
    from crosscut.group_joint_render_observations import refresh_joint_group_render_observations
    from crosscut.visual_score import VisualTimelinePlan,VisualTimelineChoice
    from crosscut.timeline import ShotType
    context,domain,union,path=world()
    plan=context.incumbent.render_plan
    timeline=VisualTimelinePlan(tuple(VisualTimelineChoice(i,domain.eligible_index(s.source_index),
        s.master_start,s.master_start+s.duration,s.source_start,s.source_end,.95,.9,.9,.2,ShotType.MEDIUM,str(i))
        for i,s in enumerate(plan.segments)),.95)
    calls=[]
    def observed(**kwargs):
        assert kwargs['plan']==union.plan
        calls.append((kwargs['first_segment_index'],kwargs['last_segment_index']))
        return kwargs['clock_timeline']
    with patch('crosscut.group_joint_render_observations.refresh_fixed_group_render_observations',side_effect=observed):
        result=refresh_joint_group_render_observations.__wrapped__(plan=plan,timeline=timeline,domain=domain,
            path=path,sync_scores=(.9,)*8,maximum_refreshed_cells=18)
    assert result.plan==union.plan
    assert result.dependency_support==union.dependency_support
    assert calls==[(0,1),(3,5),(10,11)]
    assert result.refreshed_segment_indices==(0,1,3,4,5,10,11)


@pytest.mark.parametrize('case',['keep','nineteen_descriptor_cells','forged_support'])
def test_global_numerically_rejected_union_never_resolves_assets_or_starts_native_session(case):
    from crosscut import group_scene_global_execution as execution
    from crosscut import group_performer_family_owner as family
    if case=='nineteen_descriptor_cells':
        context,domain,union,path=world(indices=(1,4,7,10,13,16),length=22)
        assert len(union.dependency_support['descriptor_refresh_cells'])==19
    else:
        context,domain,union,path=world()
        if case=='keep':
            cameras=tuple(camera_unit_from_plan(program=union.allocation.program,
                unit_id=unit_id,plan=context.incumbent.render_plan,domain=domain)
                for unit_id in union.allocation.mutable_unit_ids)
            union=assemble_owned_scene_union(context=context,domain=domain,
                allocation=union.allocation,cameras=cameras)
            assert not union.changed_segment_indices
        else:
            support=deepcopy(union.dependency_support)
            support['descriptor_refresh_cells']=[]
            union=replace(union,dependency_support_json=json.dumps(support))
    context.domain=domain
    with patch.object(family,'_fresh'), \
            patch.object(execution.source,'installed_native_assets',side_effect=AssertionError('native assets reached')) as assets, \
            patch.object(execution,'GroupJointObservationSession',side_effect=AssertionError('native session reached')) as session, \
            patch.object(execution,'SceneSensors',side_effect=AssertionError('native sensors reached')) as sensors:
        result=execution.complete_global_scene_program.__wrapped__(context,union)
    assert isinstance(result,family.PerformerFamilyRejection)
    evidence=family.plain(result.evidence)
    assert evidence['outcome']=='NUMERIC_PREFLIGHT_REJECTED'
    assert evidence['normal_global_execution_performed'] is False
    assert evidence['native_work'] is None
    assert evidence['scene_observations'] is None
    assert evidence['fallback_work_restart_allowed'] is False
    assert not evidence['quality_approved']
    assets.assert_not_called();session.assert_not_called();sensors.assert_not_called()


def conditional_fixture(*,binary_units=6,maximum_paths=32):
    from crosscut.group_scene_conditional_frontier import ConditionalSceneFrontier
    groups=tuple(tuple(NS(key=f'EXPLICIT_NONMEDIA_UNIT_{i}_OPTION_{j}') for j in range(2))
        for i in range(binary_units))
    pairs=tuple((i,i+1) for i in range(binary_units-1))
    return groups,pairs,ConditionalSceneFrontier(groups,pairs,maximum_paths=maximum_paths)


def test_six_binary_conditional_units_reach_first_units_second_option_without_lexicographic_starvation():
    groups,pairs,frontier=conditional_fixture()
    seen=[]
    while frontier.peek() is not None:
        values=frontier.begin();seen.append(tuple(s.key for s in values))
        frontier.finish('NO_EDITORIAL_GAIN' if values[0] is groups[0][0] else 'DENSE_VALID_SELECTED')
        if values[0] is groups[0][1]:break
    assert len(seen)==7  # Lexicographic prefix would need combination 33.
    assert seen[-1][0]==groups[0][1].key
    assert all(any(row[i]==groups[i][1].key for row in seen) for i in range(6))
    assert frontier.to_dict()['pending_combination_count']==57


def test_conditional_path_cap_retains_exact_numeric_cursor_and_unexecuted_inventory():
    from crosscut.group_scene_conditional_frontier import ConditionalSceneFrontier
    groups,pairs,frontier=conditional_fixture()
    while frontier.peek() is not None:
        frontier.begin();frontier.finish('NO_EDITORIAL_GAIN')
    cursor=json.loads(json.dumps(frontier.to_dict()))
    assert frontier.outcome=='SEARCH_TRUNCATED'
    assert cursor['total_combinations']==64
    assert cursor['completed_combination_count']==32
    assert cursor['pending_combination_count']==32
    assert cursor['next_rank_vector'] is not None
    assert not cursor['inventory_exhausted']
    assert not cursor['new_native_budget_granted']
    restored=ConditionalSceneFrontier.from_dict(groups,pairs,cursor)
    assert restored.to_dict()==cursor
    assert restored.peek() is None  # Numeric restore cannot enlarge the request cap.


def test_exhaustive_small_conditional_rejection_is_not_a_truncated_inventory():
    groups,pairs,frontier=conditional_fixture(binary_units=1)
    while frontier.peek() is not None:
        frontier.begin();frontier.finish('NO_EDITORIAL_GAIN')
    assert frontier.outcome=='COMPLETE_NO_GAIN'
    assert frontier.to_dict()['pending_combination_count']==0
    assert frontier.to_dict()['inventory_exhausted']


def test_conditional_budget_interruption_preserves_active_and_pending_numeric_choices():
    from crosscut.group_scene_conditional_frontier import ConditionalSceneFrontier
    groups,pairs,frontier=conditional_fixture()
    frontier.begin();frontier.finish('OBSERVED_HYPOTHESIS_REJECTED')
    values=frontier.begin()
    cursor=frontier.to_dict()
    assert cursor['completed_combination_count']==1
    assert cursor['pending_combination_count']==63
    assert cursor['active_rank_vector']==cursor['next_rank_vector']
    restored=ConditionalSceneFrontier.from_dict(groups,pairs,json.loads(json.dumps(cursor)))
    assert tuple(s.key for s in restored.peek())==tuple(s.key for s in values)
    assert restored.to_dict()==cursor


def test_empty_conditional_discovery_is_unknown_not_exhaustive_no_gain():
    from crosscut.group_scene_conditional_frontier import ConditionalSceneFrontier
    groups,pairs,unused=conditional_fixture()
    frontier=ConditionalSceneFrontier(((),*groups[1:]),pairs)
    assert frontier.peek() is None
    assert frontier.outcome=='HYPOTHESES_UNAVAILABLE'
    assert not frontier.to_dict()['inventory_exhausted']
    assert not frontier.to_dict()['quality_approved']


def test_original_reference_cannot_swallow_budget_exhaustion_as_focal_ambiguity():
    from crosscut.group_scene_global_execution import global_original_reference
    from crosscut.group_joint_source_camera import JointGroupBudgetExhausted
    context,domain,union,path=world()
    def raw(*args):raise JointGroupBudgetExhausted('EXPLICIT_ORIGINAL_REFERENCE_BUDGET_CONTROL')
    with pytest.raises(JointGroupBudgetExhausted,match='ORIGINAL_REFERENCE_BUDGET'):
        global_original_reference(NS(raw=raw),path,tuple((s,) for s in path),lambda a,b:{})


def test_conditional_pair_pruning_reuses_only_observed_identical_real_incident_pair():
    groups,pairs,frontier=conditional_fixture()
    values=frontier.begin()
    frontier.remember_infeasible_pair((0,1),values[0],values[1],'EXPLICIT_OBSERVED_UNAVAILABLE_PAIR')
    frontier.finish('OBSERVED_PAIR_REJECTED')
    second=frontier.begin()
    assert frontier.blocked_pair(second)['previous_observed_pair_reused']
    with pytest.raises(ValueError,match='frozen unit inventory'):
        frontier.remember_infeasible_pair((0,5),values[0],values[5],'NOT_AN_ACTUAL_INCIDENT_EDGE')


@pytest.mark.parametrize('change',['inventory_order','pending_count','skip_current','fake_quality'])
def test_conditional_cursor_cannot_rehash_a_different_inventory_or_unexecuted_progress(change):
    from crosscut.group_scene_conditional_frontier import ConditionalSceneFrontier
    groups,pairs,frontier=conditional_fixture()
    frontier.begin();frontier.finish('NO_EDITORIAL_GAIN');cursor=deepcopy(frontier.to_dict())
    if change=='inventory_order':cursor['inventory']['ordered_state_keys'][0].reverse()
    elif change=='pending_count':cursor['pending_combination_count']-=1
    elif change=='skip_current':cursor['next_rank_vector']=[1]*6
    else:cursor['quality_approved']=0
    with pytest.raises(ValueError,match='canonical progress'):
        ConditionalSceneFrontier.from_dict(groups,pairs,cursor)
