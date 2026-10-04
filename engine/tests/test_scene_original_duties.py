"""Original-owned duty lifecycle on explicit nonhuman native observations.

Real SceneSensors projection/cache/endpoint/dense paths are exercised below.
The model observations and head pixel links are explicit fixture substitutes;
this does not validate real performers, cross-camera identity or editing quality.
"""
from copy import deepcopy
from dataclasses import replace
import json
import pytest

from crosscut import group_scene_original_duties as original
from crosscut import group_scene_observations as scene
from crosscut.group_joint_source_camera import GroupOutputRole as Role
from crosscut.group_scene_role_obligations import scene_role_obligations
from engine.tests.test_scene_focal_pose_motion import scene_world
from crosscut.render import RenderSegment
from types import SimpleNamespace as NS


def freeze(sensors,state,h):
    # Re-author the explicit original fixture anchors after fixture mutation.
    # Production observations/hypotheses must agree; do not weaken that gate.
    h=deepcopy(h)
    h['raw_observation_sha256']=[sensors.raw(state,t,h['analysis_roll']).sha256
        for t in h['anchor_output_ticks']]
    sensors.hypotheses[scene.family.digest(h)]=h
    state=sensors.bind(state,h)
    return original.freeze_original_duty_program(sensors,((state,),),
        original_plan_identity='a'*64)


def test_known_arm_is_latched_across_original_owner_ambiguity_and_missing_pose():
    sensors,state,h,rows,_=scene_world()
    for tick in (2,3):
        p=rows[tick].people[0]
        rows[tick]=replace(rows[tick],people=(replace(p,
            unique_head_observation_id=None,
            part_states=(('left_arm','unknown'),('right_arm','unknown'))),))
    program=freeze(sensors,state,h)
    unit=program['units'][0]
    assert unit['rows'][2]['required_part_names']==['left_arm','right_arm']
    branch=unit['branches'][0]['rows'][2]
    assert branch['parts']['left_arm']['obligation']=='required'
    assert branch['parts']['left_arm']['observation']=='missing'
    assert branch['parts']['left_arm']['became_required_at_output_tick']==0
    assert program['coherent_cross_camera_actor_correspondence_certified'] is False


def test_original_missing_head_cannot_clear_known_parts_or_reacquire_background():
    sensors,state,h,rows,_=scene_world()
    rows[2]=replace(rows[2],heads=())
    program=freeze(sensors,state,h)
    branch=program['units'][0]['branches'][0]['rows']
    assert branch[2]['selected_frame_local_head_id'] is None
    assert branch[2]['head_observation']=='missing'
    assert branch[3]['selected_frame_local_head_id'] is None
    assert branch[3]['head_observation']=='unresolved'
    assert all(r['parts']['right_arm']['obligation']=='required' for r in branch)


def test_donor_only_observed_arm_is_context_and_does_not_create_a_duty():
    sensors,state,h,rows,_=scene_world()
    for tick,row in tuple(rows.items()):
        rows[tick]=replace(row,people=tuple(replace(p,
            part_states=(('left_arm','observed'),('right_arm','unknown'))) for p in row.people))
    program=freeze(sensors,state,h)
    assert program['units'][0]['rows'][0]['required_part_names']==['left_arm']
    sensors.install_original_duty_program(program)
    for tick,row in tuple(rows.items()):
        rows[tick]=replace(row,people=tuple(replace(p,
            part_states=(('left_arm','observed'),('right_arm','observed'))) for p in row.people))
    row=sensors.frame(state,0)
    assert [(a['part'],a['required_by_role']) for a in row['projected_observed_arms']]==[
        ('left_arm',True),('right_arm',False)]
    assert row['role_obligations']['required_parts']==((0,'left_arm'),)
    assert len(row['projected_observed_arms'])==2
    assert row['original_duty_reference']['program_sha256']==program['program_sha256']


def test_missing_candidate_required_arm_does_not_borrow_background_or_warm_cache():
    sensors,state,h,rows,_=scene_world()
    program=freeze(sensors,state,h)
    before=sensors.frame(state,0)
    sensors.install_original_duty_program(program)
    owner=rows[0].people[0]
    rows[0]=replace(rows[0],people=(replace(owner,
        part_states=(('left_arm','observed'),('right_arm','unknown'))),
        replace(owner,observation_id=99,unique_head_observation_id=None)))
    with pytest.raises(scene.SceneFrameRejection) as exc:sensors.frame(state,0)
    assert exc.value.row is not before
    assert 'original required arm' in str(exc.value)
    assert exc.value.row['original_duty_reference']['program_sha256']==program['program_sha256']


def test_original_required_scope_reaches_actual_endpoint_and_dense_pose_paths():
    sensors,state,h,_,_=scene_world();program=freeze(sensors,state,h)
    sensors.install_original_duty_program(program)
    endpoint=sensors.endpoint(state,outgoing=True)
    assert endpoint['original_duty_program_sha256']==program['program_sha256']
    assert endpoint['registered_pose_sequence']['required_part_names']==['left_arm','right_arm']
    dense=sensors.dense((state,state,state))[0]
    assert len(dense['rows'])==8
    assert all(r['original_duty_reference']['program_sha256']==program['program_sha256'] for r in dense['rows'])
    assert all(r['required_body_temporal_link']['required_part_names']==['left_arm','right_arm']
        for r in dense['rows'][1:])


def test_original_part_becomes_required_only_on_positive_measurement_then_stays_required():
    sensors,state,h,rows,_=scene_world()
    for tick,row in tuple(rows.items()):
        if tick==2:continue
        rows[tick]=replace(row,people=tuple(replace(p,
            part_states=(('left_arm','observed'),('right_arm','unknown'))) for p in row.people))
    program=freeze(sensors,state,h);unit=program['units'][0]
    assert unit['rows'][1]['required_part_names']==['left_arm']
    assert all(row['required_part_names']==['left_arm','right_arm'] for row in unit['rows'][2:])
    assert unit['branches'][0]['rows'][3]['parts']['right_arm']['observation']=='missing'
    assert unit['branches'][0]['rows'][3]['parts']['right_arm']['became_required_at_output_tick']==2


def test_long_original_scope_has_bounded_native_sampling_and_explicit_unrequested_ticks():
    sensors,state,h,rows,calls=scene_world();count=32
    rows.update({n:replace(rows[n%8],native_frame_index=n) for n in range(8,count)})
    state=replace(state,end_output_frame=count,native_frame_indices=tuple(range(count)),
        segments=(RenderSegment(0,0.,count/30,0.),),segment_output_frame_counts=(count,))
    h=deepcopy(h);h['span_key']=list(sensors.span_key(state))
    h['anchor_output_ticks']=[0,3,count-4,count-1];h['anchor_head_ids']=[0]*4
    intent=h['shot_intent'];intent.update(end_output_frame=count,required_output_interval=[0,count])
    for inventory in ('cells','retained_output_cells'):
        for cell in intent[inventory]:cell['end_output_frame']=count
    calls.clear();program=freeze(sensors,state,h);branch=program['units'][0]['branches'][0]
    queried=set(calls)
    assert len(queried)<=original.MAX_ORIGINAL_NATIVE_SAMPLE_TICKS
    assert queried==set(branch['original_requested_output_ticks'])
    assert not branch['every_original_native_tick_observed']
    assert not branch['unobserved_actor_bridges_certified']
    for row in branch['rows']:
        if row['output_tick'] in queried:continue
        assert row['native_observation_sha256'] is None
        assert row['head_observation']=='not_requested'
        assert row['parts']['left_arm']['observation']=='not_requested'
        assert row['parts']['left_arm']['obligation']=='required'
    inventory={r.sha256:r.to_dict() for r in sensors.raw_rows.values()}
    original.validate_original_duty_program(program,raw_inventory=inventory)


def test_incomplete_original_window_fails_before_spending_more_original_native_work():
    sensors,state,h,_,calls=scene_world();calls.clear()
    with pytest.raises(ValueError,match='explicit original hypotheses'):
        original.freeze_original_duty_program(sensors,((state,),()),original_plan_identity='a'*64)
    assert calls==[]


def test_explicit_arm_role_keeps_both_duties_when_original_measurement_is_missing():
    sensors,state,h,rows,_=scene_world()
    h=deepcopy(h);h['role']=Role.FOCAL_HEAD_ARMS.value;h['span_key'][-1]=h['role']
    h['shot_intent']['role']=h['role']
    for inventory in ('cells','retained_output_cells'):
        for cell in h['shot_intent'][inventory]:cell['alignment_subject']='upper_body'
    state=replace(state,role=Role.FOCAL_HEAD_ARMS)
    sensors.hypotheses[scene.family.digest(h)]=h
    state=sensors.bind(state,h)
    for tick,row in tuple(rows.items()):rows[tick]=replace(row,people=())
    program=freeze(sensors,state,h)
    assert all(row['required_part_names']==['left_arm','right_arm'] for row in program['units'][0]['rows'])
    assert all(r['parts']['left_arm']['observation']=='missing'
        for r in program['units'][0]['branches'][0]['rows'])


def test_program_keeps_original_hypotheses_separate_not_actor_or_part_counts():
    sensors,state,h,rows,_=scene_world()
    other=deepcopy(h);other['analysis_roll']=0.0
    sensors.hypotheses[scene.family.digest(other)]=other
    other_state=sensors.bind(state,other)
    program=original.freeze_original_duty_program(sensors,((state,other_state),),
        original_plan_identity='a'*64)
    unit=program['units'][0]
    assert len(unit['branches'])==2
    assert unit['original_hypothesis_ambiguity_retained'] is True
    assert unit['rows'][0]['required_part_names']==['left_arm','right_arm']
    assert unit['part_names_are_not_cast_counts'] is True
    assert program['coherent_original_programme_selected'] is False


def test_candidate_subspan_queries_original_lifecycle_by_output_tick_not_new_bounds():
    sensors,state,h,rows,_=scene_world()
    owner=rows[3].people[0]
    rows[3]=replace(rows[3],people=(replace(owner,
        part_states=(('left_arm','observed'),('right_arm','unknown'))),))
    program=freeze(sensors,state,h)
    sensors.install_original_duty_program(program)
    original_id=program['units'][0]['original_unit_id']
    # A one-tick query is not a newly constructed render cell or a new
    # lifecycle. It asks the ORIGINAL programme about this same output tick.
    subspan=NS(first_output_frame=3,end_output_frame=4,role=state.role)
    ref=original.duty_at(program,subspan,3)
    assert ref==original.duty_at(program,state,3)
    assert ref['original_unit_id']==original_id
    assert ref['required_part_names']==['left_arm','right_arm']
    assert program['units'][0]['branches'][0]['rows'][3]['parts']['right_arm']['observation']=='missing'
    assert ref['external_original_id']==program['original_plan_identity']
    assert ref['original_hypothesis_branches']==[
        b['original_subject_hypothesis_id'] for b in program['units'][0]['branches']]
    with pytest.raises(ValueError,match='nonincident'):
        original.duty_at(program,subspan,4)
    with pytest.raises(ValueError,match='role'):
        original.duty_at(program,NS(first_output_frame=3,end_output_frame=4,role=Role.COMPOSITION),3)


def test_subspan_dense_cache_dependency_keeps_the_complete_original_duty_unit(tmp_path):
    sensors,state,h,rows,_=scene_world();program=freeze(sensors,state,h)
    sensors.install_original_duty_program(program)
    # Dependency identity only. Real NONMEDIA regular files supply identity;
    # none is decoded or represented as a camera/model-quality observation.
    paths=tuple(tmp_path/f'NONMEDIA-cache-source-{i}.bin' for i in range(2))
    for path in paths:path.write_bytes(b'EXPLICIT_NONMEDIA_IDENTITY_ONLY')
    sensors.session.domain=replace(sensors.session.domain,request_sources=paths)
    sensors.session.closed=False;sensors.session.models={};sensors.session.model_adapters={};sensors.assets={}
    candidate=replace(state,first_output_frame=2,end_output_frame=6,
        native_frame_indices=(2,3,4,5),segments=(RenderSegment(0,2/30,4/30,2/30),),
        segment_output_frame_counts=(4,))
    selected=deepcopy(h);selected['span_key']=list(sensors.span_key(candidate))
    selected['anchor_output_ticks']=[2,3,4,5];selected['anchor_head_ids']=[0]*4
    selected['raw_observation_sha256']=[rows[t].sha256 for t in (2,3,4,5)]
    sensors.hypotheses[scene.family.digest(selected)]=selected
    candidate=sensors.bind(candidate,selected)
    # This helper binds dependency identity, not a passing dense observation.
    result=sensors._state_dense_binding(candidate)
    assert result['original_duty_units_sha256']==scene.family.digest(program['units'])
    assert result['original_duty_program_sha256']==program['program_sha256']
    assert result['state_key']==candidate.key


@pytest.mark.parametrize('mode',['drop_tick','clear_latch','replace_owner','wrong_raw','promote_identity'])
def test_rehashed_original_program_cannot_change_raw_lifecycle(mode):
    sensors,state,h,_,_=scene_world();program=freeze(sensors,state,h)
    bad=deepcopy(program)
    branch=bad['units'][0]['branches'][0]
    if mode=='drop_tick':branch['rows'].pop(2)
    elif mode=='clear_latch':branch['rows'][2]['parts']['left_arm']['obligation']='unknown'
    elif mode=='replace_owner':branch['rows'][2]['selected_frame_local_head_id']=99
    elif mode=='wrong_raw':branch['rows'][2]['native_observation_sha256']='b'*64
    else:bad['coherent_cross_camera_actor_correspondence_certified']=True
    bad['program_sha256']=scene.family.digest({k:v for k,v in bad.items() if k!='program_sha256'})
    inventory={r.sha256:r.to_dict() for r in sensors.raw_rows.values()}
    with pytest.raises(ValueError):original.validate_original_duty_program(bad,raw_inventory=inventory)


def test_original_program_references_complete_raw_inventory_without_copying_native_rows():
    sensors,state,h,_,calls=scene_world();program=freeze(sensors,state,h)
    inventory={r.sha256:r.to_dict() for r in sensors.raw_rows.values()}
    original.validate_original_duty_program(program,raw_inventory=inventory)
    assert set(calls)==set(range(8))
    # The reference name is present, but complete native model records are not duplicated.
    assert 'raw_pose_xyc' not in str(program)
    assert program==json.loads(json.dumps(program,allow_nan=False))


@pytest.mark.parametrize('parts',[['left_arm'],('left_arm','left_arm'),('nose',)])
def test_external_required_names_cannot_be_unvalidated_or_redefined(parts):
    _,_,_,rows,_=scene_world()
    with pytest.raises(ValueError):scene_role_obligations(rows[0],Role.FOCAL_HEAD,0,
        required_part_names=parts)
