"""Explicit synthetic native records isolate Pro F03/F04; no human inference."""
from dataclasses import replace
from copy import deepcopy
from types import SimpleNamespace as NS
from time import monotonic

import numpy as np
import pytest

from crosscut import group_scene_observations as scene
from crosscut import group_source_changing_owner as source
from crosscut.group_joint_source_camera import GroupSourceCameraState,GroupOutputRole as Role,JointGroupWorkBudget
from crosscut.group_joint_native_observer import NativeGroupObservation,NativeGroupHead
from crosscut.group_joint_person_ownership import OwnedPersonObservation
from crosscut.render import RenderSegment


def fixture(role=Role.FOCAL_HEAD,*,parts=('left_arm','right_arm'),textured=False):
    head=NativeGroupHead(0,(.38,.24,.15,.15),
        ((.4175,.285),(.4925,.285),(.455,.315),(.425,.3525),(.485,.3525)),.99)
    points=((.41,.43,1.),(.49,.43,1.),(.38,.52,1.),(.52,.52,1.),
        (.36,.61,1.),(.54,.61,1.),(.41,.70,1.),(.49,.70,1.))
    person=OwnedPersonObservation(0,(.33,.24,.24,.5),points,.99,
        pose_crop_xywh=(0.,0.,1.,1.),source_pixel_to_pose_input_affine=((1.,0.,0.),(0.,1.,0.)),
        raw_pose_xyc=((.45,.32,1.),)*17,pose_reason='observed',
        candidate_head_observation_ids=(0,),unique_head_observation_id=0,body_ownership_state='observed',
        part_states=tuple((n,'observed' if n in parts else 'unknown') for n in ('left_arm','right_arm')))
    state=GroupSourceCameraState(1,0,8,0,(RenderSegment(0,0.,8/30,0.),),
        tuple(range(8)),role,'a'*64,30.,(320,180),(8,))
    sensors=object.__new__(scene.SceneSensors)
    sensors.hypotheses={};sensors.discoveries={};sensors.link_cache={};sensors.frame_cache={}
    sensors.raw_rows={};sensors.dense_cache={};sensors.shot_intents={};sensors.endpoint_cache={}
    pixels=(np.random.default_rng(134).integers(10,240,(180,320,3),dtype=np.uint8)
        if textured else np.zeros((180,320,3),np.uint8))
    sensors.session=NS(budget=JointGroupWorkBudget(absolute_deadline=monotonic()+30),
        native_pixels=lambda *a:pixels,projected_pixels=lambda *a:pixels)
    raw={t:NativeGroupObservation(0,t,(320,180),'a'*64,(1,2,3,4,5),source.sha256(memoryview(pixels)).hexdigest(),(),
        (person,),(head,),'EXPLICIT_SYNTHETIC_SENSOR_NOT_REAL_MODEL') for t in range(8)}
    sensors.raw=lambda state,tick,roll:raw[tick]
    # Explicit head association substitute cannot certify any arm pixel path.
    sensors._head_links=lambda a,b:dict(unique_links=[[0,0]])
    hypothesis=sensors.discover(state,rolls=(None,))[0]
    return sensors,replace(state,editorial_contract_sha256=source.family.digest(hypothesis))


def test_known_selected_arm_needs_dense_pixels_even_in_head_role():
    sensors,state=fixture()
    with pytest.raises(ValueError,match='joint pixel lifecycle'):
        sensors.dense((state,state,state))


def test_fixed_endpoint_static_skeleton_and_perfect_head_link_do_not_prove_arm_motion():
    sensors,state=fixture()
    first,last=(sensors.frame(state,t) for t in (0,3))
    with pytest.raises(ValueError,match='arm native pixel path'):
        scene.endpoint_focal_part_paths(sensors,state,first,last)


@pytest.mark.parametrize('role',[Role.FOCAL_HEAD,Role.FOCAL_HEAD_ARMS])
def test_observed_arm_metric_cannot_replace_missing_head_motion(monkeypatch,role):
    import crosscut.observed_head_motion as hm
    import crosscut.closeup_gesture as gesture
    sensors,state=fixture(role)
    rows=[sensors.frame(state,t) for t in (0,3,4,7)]
    monkeypatch.setattr(hm,'measure_head_motion',lambda *a,**k:None)
    monkeypatch.setattr(gesture,'measure_arm_motion',lambda *a,**k:NS(
        observed_bones=(0,1,2,3),predicted_jump_radians=(.1,)*4,
        velocity_jump_per_output_tick_radians=(.1,)*4,to_dict=lambda:{'explicit_test':True}))
    complete,_,_=scene.selected_focal_motion(rows,(state,)*4)
    assert not complete


def test_head_role_cannot_drop_known_arm_motion_when_head_metric_exists(monkeypatch):
    import crosscut.observed_head_motion as hm
    import crosscut.closeup_gesture as gesture
    sensors,state=fixture()
    rows=[sensors.frame(state,t) for t in (0,3,4,7)]
    monkeypatch.setattr(hm,'measure_head_motion',lambda *a,**k:NS(
        predicted_pose_cost=.1,velocity_jump_per_output_tick=.1,to_dict=lambda:{'explicit_test':True}))
    monkeypatch.setattr(gesture,'measure_arm_motion',lambda *a,**k:None)
    complete,_,_=scene.selected_focal_motion(rows,(state,)*4)
    assert not complete


def test_one_arm_metric_cannot_replace_two_declared_known_arms(monkeypatch):
    import crosscut.observed_head_motion as hm
    import crosscut.closeup_gesture as gesture
    sensors,state=fixture()
    rows=[sensors.frame(state,t) for t in (0,3,4,7)]
    monkeypatch.setattr(hm,'measure_head_motion',lambda *a,**k:NS(
        predicted_pose_cost=.1,velocity_jump_per_output_tick=.1,to_dict=lambda:{'explicit_test':True}))
    monkeypatch.setattr(gesture,'measure_arm_motion',lambda *a,**k:NS(
        observed_bones=(0,1),predicted_jump_radians=(.1,.1,None,None),
        velocity_jump_per_output_tick_radians=(.1,.1,None,None),to_dict=lambda:{'explicit_test':True}))
    complete,_,_=scene.selected_focal_motion(rows,(state,)*4)
    assert not complete


def test_head_role_with_unknown_arms_does_not_invent_arm_temporal_obligation():
    sensors,state=fixture(parts=())
    first,last=(sensors.frame(state,t) for t in (0,3))
    assert scene.endpoint_focal_part_paths(sensors,state,first,last) is None


def dense_known_arm_fixture():
    """Measured synthetic-raster paths plus an EXPLICIT camera-relation double.

    Only consumer structure is isolated here; neither named humans nor actual
    camera independence, source program admission or editing quality is proved.
    """
    from engine.tests.test_scene_source_owner import scene_derivative_fixture
    from crosscut.group_scene_intent import CONTRACT as INTENT_CONTRACT
    sensors,state=fixture(textured=True)
    middle=deepcopy(sensors.hypothesis(state))
    cells=[dict(segment_index=0,first_output_frame=0,end_output_frame=8,
        shot_type='close_up',alignment_subject='face',face_applied=False)]
    middle['shot_intent']=dict(contract=INTENT_CONTRACT,role=state.role.value,
        first_output_frame=0,end_output_frame=8,cells=cells,retained_output_cells=deepcopy(cells),
        endpoint_scope='whole',required_output_interval=[0,8],fixed_nonincident_pixels_unchanged=False,
        frozen_before_source_camera_search=True,incident_boundary_does_not_own_whole_shot=True,
        quality_approved=False)
    hypotheses=[]
    for source_index in (1,0,2):
        hypothesis=deepcopy(middle)
        hypothesis['span_key'][0]=source_index
        hypotheses.append(hypothesis)
    keys=[source.family.digest(h) for h in hypotheses]
    state=replace(state,editorial_contract_sha256=keys[1])
    sensors.hypotheses[keys[1]]=hypotheses[1]
    dense=dict(contract=scene.DENSE_CONTRACT,state_keys=['fixture-left',state.key,'fixture-right'],
        scene_hypotheses=dict(zip(keys,hypotheses)),hypothesis_keys=keys,
        source_derivative_binding={'audit':scene_derivative_fixture(3).to_dict()},
        mutable_shots=source.family.plain(sensors.dense((state,state,state))))
    return dense


def test_real_synthetic_joint_sequence_is_consumed_for_every_known_arm_dense_tick():
    dense=dense_known_arm_fixture()
    scene.validate_scene_dense(dense)
    for before,after in zip(dense['mutable_shots'][0]['rows'],dense['mutable_shots'][0]['rows'][1:]):
        link=after['required_body_temporal_link']
        assert link['native_steps_complete']
        assert link['native_frame_indices']==[before['native_frame_index'],after['native_frame_index']]
        assert not link['quality_approved']


@pytest.mark.parametrize('change',[
    'legacy_link','legacy_dense','missing_step','reordered_steps','unread','bad_hash','endpoint_hash',
    'source','clock','file','raster','identity_claim','phase_claim','quality_claim','raw_pair'])
def test_dense_consumer_rejects_required_body_native_sequence_tampering(change):
    dense=dense_known_arm_fixture()
    scene.validate_scene_dense(dense)
    row=dense['mutable_shots'][0]['rows'][1]
    link=row['required_body_temporal_link']
    if change=='legacy_link':link['contract']='native_required_body_joint_pixel_path_v1'
    elif change=='legacy_dense':dense['contract']='source_native_scene_role_whole_camera_refresh_v5'
    elif change=='missing_step':link['native_frame_indices'].pop()
    elif change=='reordered_steps':link['native_frame_indices'].reverse()
    elif change=='unread':link['native_steps_complete']=False
    elif change=='bad_hash':link['native_pixel_sha256'][0]='z'*64
    elif change=='endpoint_hash':link['native_pixel_sha256'][-1]='0'*64
    elif change=='source':link['request_source_index']=1
    elif change=='clock':link['source_clock_identity']='c'*64
    elif change=='file':link['source_file_identity'][-1]+=1
    elif change=='raster':link['frame_size_wh'][0]+=1
    elif change=='identity_claim':link['named_identity_certified']=True
    elif change=='phase_claim':link['body_phase_certified']=True
    elif change=='quality_claim':link['quality_approved']=True
    else:link['raw_observation_sha256'][0]='f'*64
    with pytest.raises(ValueError):scene.validate_scene_dense(dense)
