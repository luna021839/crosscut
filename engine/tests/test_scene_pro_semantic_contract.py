"""Pro F23-F27 regressions with explicit nonhuman observation fixtures.

Real orchestration/pose-image consumers, not real model or editorial quality.
"""
from copy import deepcopy
from dataclasses import replace
import json
import pytest

from crosscut import group_scene_original_duties as duties
from crosscut import group_scene_focal_pose_motion as pose
from crosscut import group_joint_focal_camera as camera
from crosscut.group_joint_person_ownership import bind_person_ownership
from engine.tests.test_scene_original_duties import freeze
from engine.tests.test_scene_focal_pose_motion import scene_world
from crosscut.render import RenderSegment
from crosscut.group_joint_source_camera import JointGroupBudgetExhausted


def left_original_both_donor(*, activation=None):
    sensors,state,h,rows,calls=scene_world()
    donor_rows={n:replace(row,request_source_index=1,source_file_identity=(1,1,3,4,5))
        for n,row in rows.items()}
    for tick,row in tuple(rows.items()):
        people=[]
        for person in row.people:
            joints=list(person.raw_pose_xyc)
            if activation is None or tick!=activation:
                for index in (8,10):joints[index]=(*joints[index][:2],.1)
            people.append(replace(person,raw_pose_xyc=tuple(joints),
                upper_body_xyc=tuple(joints[n] for n in (5,6,7,8,9,10,11,12))))
        bound,owners=bind_person_ownership(tuple(people),row.heads,row.frame_size_wh)
        rows[tick]=replace(row,people=bound,heads=tuple(replace(head,
            candidate_body_observation_ids=tuple(owners[head.observation_id])) for head in row.heads))
    program=freeze(sensors,state,h)
    sensors.install_original_duty_program(program)
    def raw_native(value,native,roll):
        row=(rows if value.eligible_source_index==0 else donor_rows)[native]
        calls.append(native);sensors.raw_rows[value.eligible_source_index,native,roll]=row
        return row
    sensors.raw_native=raw_native
    sensors.raw=lambda value,tick,roll:raw_native(value,
        value.native_frame_indices[tick-value.first_output_frame],roll)
    donor=replace(state,eligible_source_index=1,
        segments=tuple(replace(s,source_index=1) for s in state.segments))
    donor_h=deepcopy(h);donor_h['span_key']=list(sensors.span_key(donor))
    donor_h['raw_observation_sha256']=[donor_rows[t].sha256 for t in donor_h['anchor_output_ticks']]
    sensors.hypotheses[pose.digest(donor_h)]=donor_h
    donor=sensors.bind(donor,donor_h)
    return sensors,donor,donor_h,donor_rows,program


def consume(value,sensors,state,h,program):
    inventory={r.sha256:r.to_dict() for r in sensors.raw_rows.values()}
    pair=value['primary_raw_observation_sha256']
    pose.validate_focal_pose_view(json.loads(json.dumps(value)),inventory[pair[0]],inventory[pair[1]],
        raw_inventory=inventory,source_domain=sensors.session.domain.to_dict(),
        hypothesis=h,state_key=state.key,original_duty_program=program)


def test_original_left_only_survives_registered_endpoint_dense_and_external_consumer():
    sensors,state,h,_,program=left_original_both_donor()
    view=sensors.endpoint(state,outgoing=True)['registered_pose_sequence']
    assert view['required_part_names']==['left_arm']
    consume(view,sensors,state,h,program)
    dense=sensors.dense((state,state,state))[0]
    for row in dense['rows'][1:]:
        view=row['required_body_temporal_link']
        assert view['required_part_names']==['left_arm']
        consume(view,sensors,state,h,program)
        assert [(a['part'],a['required_by_role']) for a in row['projected_observed_arms']]==[
            ('left_arm',True),('right_arm',False)]
    bad=deepcopy(view)
    bad['required_part_names']=['left_arm','right_arm']
    with pytest.raises(ValueError):consume(bad,sensors,state,h,program)


def test_activation_changes_tick_duties_not_temporal_observation_support():
    sensors,state,h,_,program=left_original_both_donor(activation=2)
    dense=sensors.dense((state,state,state))[0]
    crossing=dense['rows'][2]['required_body_temporal_link']
    assert crossing['required_part_names']==['left_arm','right_arm']
    assert [s['original_duty_reference']['required_part_names']
        for s in crossing['actual_output_samples']]==[['left_arm'],['left_arm','right_arm']]
    consume(crossing,sensors,state,h,program)
    bad=deepcopy(crossing)
    bad['actual_output_samples'][0]['original_duty_reference']['required_part_names'].append('right_arm')
    with pytest.raises(ValueError):consume(bad,sensors,state,h,program)


def test_original_samples_are_time_distributed_and_actual_midspan_link_updates_duty():
    count=60
    sampled=duties.original_sample_ticks(0,count,(0,3,count-4,count-1))
    assert len(sampled)==16
    assert max(b-a for a,b in zip(sampled,sampled[1:]))<=6
    middle=next(n for n in sampled if count//3<n<2*count//3)
    sensors,state,h,rows,_=scene_world()
    rows.update({n:replace(rows[n%8],native_frame_index=n) for n in range(8,count)})
    for tick,row in tuple(rows.items()):
        rows[tick]=replace(row,people=tuple(replace(p,part_states=(
            ('left_arm','observed'),('right_arm','observed' if tick==middle else 'unknown')))
            for p in row.people))
    state=replace(state,end_output_frame=count,native_frame_indices=tuple(range(count)),
        segments=(RenderSegment(0,0.,count/30,0.),),segment_output_frame_counts=(count,))
    h=deepcopy(h);h['span_key']=list(sensors.span_key(state))
    h['anchor_output_ticks']=[0,3,count-4,count-1];h['anchor_head_ids']=[0]*4
    intent=h['shot_intent'];intent.update(end_output_frame=count,required_output_interval=[0,count])
    for inventory in ('cells','retained_output_cells'):
        for cell in intent[inventory]:cell['end_output_frame']=count
    program=freeze(sensors,state,h);branch=program['units'][0]['branches'][0]
    at=branch['rows'][middle]
    assert at['selected_frame_local_head_id']==0
    assert at['parts']['right_arm']['became_required_at_output_tick']==middle
    assert len(branch['head_pixel_links'])>2
    assert not branch['unobserved_actor_bridges_certified']
    inventory={r.sha256:r.to_dict() for r in sensors.raw_rows.values()}
    duties.validate_original_duty_program(program,raw_inventory=inventory)


def test_arm_aware_endpoint_camera_can_be_wider_without_changing_limits():
    from crosscut.boundary_subject_alignment import _project_box,_TransformState
    face=(.45,.35,.1,.1);target=(.4,.3,.2,.2);envelope=(.12,.3,.76,.4)
    proposal=camera.propose_endpoint_focal_face_roi_camera(source_face_boxes=(face,face),
        fixed_neighbor_face_boxes=(target,target),frame_count=30,fps=30.,
        required_endpoint_envelopes=((envelope,),(envelope,)))
    key=proposal.track.evaluate(0.)
    assert 1.<key.zoom<=1/.76
    assert _project_box(envelope,_TransformState(key.zoom,key.center_x,key.center_y,0.))[1]>=.999
    assert not proposal.to_dict()['required_arm_visibility_certified']


def test_endpoint_fit_can_reduce_zoom_to_keep_the_existing_radial_pan_cap():
    from math import hypot
    from crosscut.boundary_subject_alignment import _project_box,_TransformState
    face=(.12,.13,.05,.05);target=(.45,.45,.1,.1);envelope=(.02,.1,.21,.4)
    proposal=camera.propose_endpoint_focal_face_roi_camera(source_face_boxes=(face,face),
        fixed_neighbor_face_boxes=(target,target),frame_count=30,fps=30.,
        required_endpoint_envelopes=((envelope,),(envelope,)))
    key=proposal.track.evaluate(0.)
    assert 1.<key.zoom<2.
    assert hypot(key.center_x-.5,key.center_y-.5)<=.25+1e-9
    assert _project_box(envelope,_TransformState(key.zoom,key.center_x,key.center_y,0.))[1]>=.999999


def test_endpoint_fit_cannot_recover_pixels_lost_by_the_immutable_base_crop():
    face=(.45,.35,.1,.1)
    with pytest.raises(ValueError,match='outside the retained base raster'):
        camera.propose_endpoint_focal_face_roi_camera(source_face_boxes=(face,face),
            fixed_neighbor_face_boxes=(face,face),frame_count=30,fps=30.,
            required_endpoint_envelopes=(((-.05,.3,.4,.4),),((.2,.3,.4,.4),)))


def test_registered_pose_consumer_requires_external_original_programme():
    sensors,state,h,_,_=left_original_both_donor()
    view=sensors.endpoint(state,outgoing=True)['registered_pose_sequence']
    inventory={r.sha256:r.to_dict() for r in sensors.raw_rows.values()}
    pair=view['primary_raw_observation_sha256']
    with pytest.raises(ValueError,match='cannot borrow duties'):
        pose.validate_focal_pose_view(view,inventory[pair[0]],inventory[pair[1]],
            raw_inventory=inventory,source_domain=sensors.session.domain.to_dict(),
            hypothesis=h,state_key=state.key)


def test_missing_original_required_arm_cannot_be_replaced_by_observed_donor_context_arm():
    from crosscut.group_scene_observations import SceneFrameRejection
    sensors,state,_,rows,_=left_original_both_donor()
    row=rows[0];person=row.people[0];joints=list(person.raw_pose_xyc)
    for index in (7,9):joints[index]=(*joints[index][:2],.1)
    person=replace(person,raw_pose_xyc=tuple(joints),
        upper_body_xyc=tuple(joints[n] for n in (5,6,7,8,9,10,11,12)))
    bound,owners=bind_person_ownership((person,),row.heads,row.frame_size_wh)
    rows[0]=replace(row,people=bound,heads=tuple(replace(head,
        candidate_body_observation_ids=tuple(owners[head.observation_id])) for head in row.heads))
    with pytest.raises(SceneFrameRejection):sensors.frame(state,0)


@pytest.mark.parametrize('stage',['first_frame','endpoint'])
def test_budget_exception_is_not_a_measured_negative_edge(stage):
    from crosscut.group_scene_observations import compare_scene_edge
    sensors,state,_,_,_=scene_world()
    def fail(*args,**kwargs):raise JointGroupBudgetExhausted('EXPLICIT_NONHUMAN_BUDGET_INJECTION')
    if stage=='first_frame':sensors.frame=fail
    else:sensors.endpoint=fail
    with pytest.raises(JointGroupBudgetExhausted) as failure:compare_scene_edge(sensors,state,state)
    if stage=='endpoint':
        assert failure.value.partial_evidence['metrics']['occupancy']==0.
        assert len(failure.value.partial_evidence['rows'])==4
        assert failure.value.partial_evidence['available'] is False
    else:assert failure.value.partial_evidence['rows']==[]


def test_normal_fallback_cannot_restart_work_or_drop_original_contract(monkeypatch):
    from types import SimpleNamespace
    from crosscut import group_performer_family_owner as family
    from crosscut import group_scene_source_owner as scene
    from crosscut import group_source_changing_owner as source
    rejected=family.PerformerFamilyRejection('EXPLICIT_SCENE_INJECTION',dict(
        contract=scene.CONTRACT,native_work={'counts':{'model_calls':5}},
        scene_observations={'original_duty_program_sha256':'a'*64}))
    monkeypatch.setattr(family,'_fresh',lambda *a:None)
    monkeypatch.setattr(scene,'propose_scene_changes',lambda *a:rejected)
    def forbidden(*args,**kwargs):raise AssertionError('new-budget/old-duty fallback called')
    monkeypatch.setattr(source,'propose_source_changes',forbidden)
    value=family.propose_performer_family.__wrapped__(SimpleNamespace(source_derivative_binding={}))
    assert isinstance(value,family.PerformerFamilyRejection)
    assert value.evidence['fallback_status']=='UNSUPPORTED_ORIGINAL_DUTY_REQUEST_LEDGER'
    assert not value.evidence['legacy_proposal_called']
    assert value.evidence['scene_search']==rejected.evidence


def test_mutable_camera_input_retains_violation_but_missing_required_owner_is_not_repaired():
    from crosscut.render import EditorialTransformTrack,EditorialTransformKeyframe
    from crosscut.group_scene_observations import SceneFrameRejection
    sensors,state,_,rows,_=left_original_both_donor()
    old=replace(state,segments=(replace(state.segments[0],editorial_transform=
        EditorialTransformTrack((EditorialTransformKeyframe(0.,2.),))),))
    # Move actual known arm support wide enough that the existing camera is
    # clipped. This is nonhuman geometry, never additional model authority.
    row=rows[0];person=row.people[0];joints=list(person.raw_pose_xyc)
    joints[9]=(.13,.5,.9)
    person=replace(person,raw_pose_xyc=tuple(joints),
        upper_body_xyc=tuple(joints[n] for n in (5,6,7,8,9,10,11,12)))
    bound,owners=bind_person_ownership((person,),row.heads,row.frame_size_wh)
    rows[0]=replace(row,people=bound,heads=tuple(replace(h,
        candidate_body_observation_ids=tuple(owners[h.observation_id])) for h in row.heads))
    with pytest.raises(SceneFrameRejection):sensors.frame(old,0)
    measured,envelopes=sensors.mutable_camera_inputs(old,0)
    assert measured['frame_admission_violations'] and len(envelopes)==2
    rows[0]=replace(rows[0],people=())
    with pytest.raises(SceneFrameRejection):sensors.mutable_camera_inputs(state,0)
