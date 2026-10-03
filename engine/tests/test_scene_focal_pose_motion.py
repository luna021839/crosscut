"""Explicit synthetic model/face-link fixtures, real pixel contradiction sensor.

No human inference, physical mask/identity or professional quality is tested.
"""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace
from time import monotonic
from pathlib import Path
import json

import cv2
import numpy as np
import pytest

from crosscut import group_scene_focal_pose_motion as pose
from crosscut.group_joint_person_ownership import bind_person_ownership
from crosscut.group_joint_source_camera import JointGroupWorkBudget
from crosscut.group_joint_temporal_pose import TemporalPoseUnavailable,supported_pose_spans
from engine.tests.test_group_joint_temporal_pose import NativeTemporalPoseTests


def world(*,lost_joint=None,duplicate=False,jitter=False,flat=False,rate=Fraction(30),cuts=((),())):
    fixture=NativeTemporalPoseTests();domain=fixture.domain((rate,rate),cuts=cuts)
    image=(np.zeros((800,1000,3),np.uint8) if flat else
        np.random.default_rng(424).integers(30,220,(800,1000,3),np.uint8))
    rows=[]
    for native in range(4):
        row=fixture.row(domain,0,native,shift=.25 if jitter and native==1 else 0.,
            lost_joint=lost_joint if native==1 else None,duplicate=duplicate and native==1)
        rows.append(replace(row,native_pixel_sha256=pose.sha256(memoryview(image)).hexdigest()))
    rows=tuple(rows)
    sensor=SimpleNamespace(session=SimpleNamespace(domain=domain,
        budget=JointGroupWorkBudget(absolute_deadline=monotonic()+30),
        native_pixels=lambda *args:image),
        _head_links=lambda a,b:dict(raw_pair=[a.sha256,b.sha256],unique_links=[[0,0]],
            measured_links=[[0,0]],cross_camera_identity_certified=False))
    return sensor,rows


def measure(sensor,rows,parts=('left_arm','right_arm')):
    return pose.measure_native_focal_sequence(sensors=sensor,rows=rows,
        selected_head_ids=((0,0),(3,0)),required_part_names=parts,
        duty_reference='EXPLICIT_FIXED_SYNTHETIC_FOCAL_REQUIRED_DUTIES_NOT_HUMAN')


def test_typed_registered_pose_measurement_never_mints_legacy_arm_pixel_verification():
    sensor,rows=world();value=measure(sensor,rows);receipt=value.to_dict()
    assert value.reason is None
    assert receipt['measurement_available']
    assert receipt['contract']==pose.CONTRACT
    assert receipt['raw_observation_sha256']==[row.sha256 for row in rows]
    assert receipt['native_frame_indices']==[0,1,2,3]
    assert receipt['same_source_arm_pixel_path_verified'] is False
    assert receipt['quality_approved'] is False
    assert 'complete_required_joint_paths' not in receipt
    assert all(supported_pose_spans(value.window,j).all_scheduled_kinematic_samples_observed
        for j in value.window.duties.required_joint_indices)


@pytest.mark.parametrize('joint',[5,6,7,8,9,10])
def test_missing_middle_required_joint_remains_unknown_not_interpolated_or_removed(joint):
    sensor,rows=world(lost_joint=joint);value=measure(sensor,rows)
    assert value.reason in ('REQUIRED_MODEL_JOINT_UNKNOWN','OWNER_UNRESOLVED')
    assert joint in value.window.duties.required_joint_indices
    assert not value.to_dict()['measurement_available']
    assert not value.to_dict()['pose_or_velocity_imputed']
    assert not value.to_dict()['required_duties_removed']


def test_unknown_context_arm_does_not_invent_a_required_arm_but_both_shoulders_stay_required():
    sensor,rows=world(lost_joint=8)
    value=measure(sensor,rows,parts=('left_arm',))
    assert value.reason is None
    assert value.window.duties.required_joint_indices==(5,6,7,9)


def test_competing_middle_owners_are_archived_not_replaced_by_best_pose_cost():
    sensor,rows=world(duplicate=True);value=measure(sensor,rows);receipt=value.to_dict()
    assert value.reason=='OWNER_UNRESOLVED'
    assert value.window.anchors[1].person_observation_id is None
    graph=receipt['association_alternatives']
    assert {n['person_observation_id'] for n in graph['nodes'] if n['layer_index']==1}=={0,1,None}
    assert graph['materialized_subject_path_count']==0
    assert not graph['candidate_admitted']


def test_model_wrist_jitter_against_stationary_real_pixels_is_a_contradiction_not_new_motion():
    sensor,rows=world(jitter=True);value=measure(sensor,rows)
    assert value.reason=='MODEL_POSE_IMAGE_DISAGREEMENT'
    assert any(c['contradiction'] for c in value.to_dict()['image_consistency'])
    assert not value.to_dict()['measurement_available']


def test_source_flash_or_flat_pixels_with_fake_pose_do_not_make_an_image_support_receipt():
    sensor,rows=world(flat=True);value=measure(sensor,rows)
    assert value.reason=='NATIVE_ROI_IMAGE_UNSUPPORTED'
    assert all(c['arm_pixel_path_verified'] is False for c in value.to_dict()['image_consistency'])


def test_native_fractional_seconds_are_not_assumed_output30fps():
    sensor,rows=world(rate=Fraction(30000,1001));value=measure(sensor,rows)
    assert value.window.anchors[3].measurement_seconds==pytest.approx(.1001)
    assert value.window.anchors[3].measurement_seconds!=.1


def test_actual_source_cut_is_not_bridged_by_three_poses():
    sensor,rows=world(cuts=((.05,),()))
    with pytest.raises(TemporalPoseUnavailable,match='broadcaster cut'):measure(sensor,rows)


def test_middle_observation_native_hash_is_checked_against_actual_pixels():
    sensor,rows=world();rows=(rows[0],replace(rows[1],native_pixel_sha256='a'*64),*rows[2:])
    with pytest.raises(ValueError,match='source pixels changed'):measure(sensor,rows)


def test_old_body8_repackaging_cannot_masquerade_as_raw17_time_series():
    sensor,rows=world();person=rows[1].people[0]
    wrong=replace(person,raw_pose_xyc=((.5,.3,.9),)*17)
    rows=(rows[0],replace(rows[1],people=(wrong,)),*rows[2:])
    with pytest.raises(ValueError,match='raw17'):measure(sensor,rows)


def test_detector_order_change_is_not_a_physical_owner_change():
    sensor,rows=world();changed=[]
    for i,row in enumerate(rows):
        person=replace(row.people[0],observation_id=i+10)
        people,owners=bind_person_ownership((person,),row.heads,row.frame_size_wh)
        changed.append(replace(row,people=people,heads=tuple(replace(h,
            candidate_body_observation_ids=tuple(owners[h.observation_id])) for h in row.heads)))
    value=measure(sensor,tuple(changed))
    assert value.reason is None
    assert [a.person_observation_id for a in value.window.anchors]==[10,11,12,13]
    assert not value.to_dict()['source_local_identity_certified']


def test_exact_serialized_model_row_restoration_does_not_import_extra_authority():
    _,rows=world();payload=__import__('json').loads(__import__('json').dumps(rows[0].to_dict()))
    assert pose.restore_native_row(payload).sha256==rows[0].sha256
    payload['motion_certified']=True
    with pytest.raises(ValueError,match='preimage'):pose.restore_native_row(payload)


@pytest.mark.parametrize('parts',[(),('left_arm','left_arm'),('other',),['left_arm']])
def test_required_part_scope_cannot_be_empty_or_redefined(parts):
    sensor,rows=world()
    with pytest.raises(ValueError):measure(sensor,rows,parts=parts)


def test_two_endpoint_poses_do_not_claim_an_actual_middle_pose_sequence():
    sensor,rows=world()
    with pytest.raises(ValueError,match='three distinct'):measure(sensor,rows[:2])


def test_changed_source_or_native_model_policy_is_not_downgraded_to_missing_measurement():
    sensor,rows=world()
    for updates in (dict(source_clock_identity='b'*64),dict(source_file_identity=(9,2,3,4,5)),
            dict(model_adapter_fingerprints=tuple((n,'a'*64) for n,h in rows[1].model_adapter_fingerprints))):
        changed=(rows[0],replace(rows[1],**updates),*rows[2:])
        with pytest.raises(ValueError,match='binding'):measure(sensor,changed)


def test_registered_oriented_face_variant_does_not_fall_back_to_legacy_arm_texture_gate():
    from crosscut.group_native_head_orientation import native_head_analysis_spec,SUPPORTED_ANALYSIS_ROLLS
    _,rows=world();row=rows[0]
    for roll in SUPPORTED_ANALYSIS_ROLLS:
        name=native_head_analysis_spec(roll)[0]
        renamed=lambda values:tuple((name if n=='yunet_head_faces' else n,sha) for n,sha in values)
        changed=replace(row,model_content_sha256=renamed(row.model_content_sha256),
            model_adapter_fingerprints=renamed(row.model_adapter_fingerprints))
        assert pose.registered_raw17_basis(changed)
    assert not pose.registered_raw17_basis(replace(row,
        model_content_sha256=tuple(('unregistered_face' if n=='yunet_head_faces' else n,sha)
            for n,sha in row.model_content_sha256),
        model_adapter_fingerprints=tuple(('unregistered_face' if n=='yunet_head_faces' else n,sha)
            for n,sha in row.model_adapter_fingerprints)))


def scene_world(*,scope='whole',rate=Fraction(30),output_fps=30.,jitter=False,flat=False):
    """Real SceneSensors orchestration; explicit nonhuman NN/face-link inputs."""
    from crosscut.group_scene_observations import SceneSensors
    from crosscut.group_scene_intent import declared_shot_intent
    from crosscut.group_joint_source_camera import GroupOutputRole,joint_state_from_render_plan
    from crosscut.render import RenderPlan,RenderSegment
    fixture=NativeTemporalPoseTests();domain=fixture.domain((rate,rate))
    plan=RenderPlan(sources=domain.request_sources,
        segments=(RenderSegment(0,0.,8/output_fps,0.),),reference_source=0,
        aspect_ratio='16:9',output=Path('EXPLICIT_NONHUMAN_NOT_RENDERED.mp4'),fps=output_fps)
    state=joint_state_from_render_plan(plan=plan,domain=domain,slot=0,first_output_frame=0,
        end_output_frame=8,role=GroupOutputRole.FOCAL_HEAD)
    image=(np.zeros((540,960,3),np.uint8) if flat else
        np.random.default_rng(527).integers(30,220,(540,960,3),np.uint8))
    rows={}
    for n in set(state.native_frame_indices):
        row=fixture.row(domain,0,n,shift=.25 if jitter and n==2 else 0.)
        people,owners=bind_person_ownership(row.people,row.heads,(960,540))
        rows[n]=replace(row,frame_size_wh=(960,540),people=people,
            heads=tuple(replace(h,candidate_body_observation_ids=tuple(owners[h.observation_id]))
                for h in row.heads),native_pixel_sha256=pose.sha256(memoryview(image)).hexdigest())
    sensors=object.__new__(SceneSensors)
    sensors.session=SimpleNamespace(domain=domain,
        budget=JointGroupWorkBudget(absolute_deadline=monotonic()+30),
        native_pixels=lambda *args:image,projected_pixels=lambda *args:image)
    for name in ('hypotheses','discoveries','link_cache','frame_cache','dense_cache','shot_intents',
            'endpoint_cache','articulation_raw_rows','articulation_endpoint_cache','raw_rows','declarations'):
        setattr(sensors,name,{})
    calls=[]
    def raw_native(state,n,roll):
        calls.append(n);sensors.raw_rows[0,n,roll]=rows[n];return rows[n]
    sensors.raw_native=raw_native
    sensors.raw=lambda state,tick,roll:raw_native(state,state.native_frame_indices[tick],roll)
    sensors._head_links=lambda a,b:dict(raw_pair=[a.sha256,b.sha256],unique_links=[[0,0]],
        measured_links=[[0,0]],cross_camera_identity_certified=False)
    context=SimpleNamespace(incumbent=SimpleNamespace(render_plan=plan,
        visual_timeline=SimpleNamespace(choices=(SimpleNamespace(shot_type='medium'),)),
        face_match_plan=SimpleNamespace(segments=(SimpleNamespace(alignment_subject=None,applied=False),))))
    shot=SimpleNamespace(first_segment_index=0,last_segment_index=0,first_output_frame=0,end_output_frame=8)
    sensors.shot_intents[0,8,scope]=declared_shot_intent(context,shot,endpoint_scope=scope)
    h=sensors.discover(state,rolls=(None,),endpoint_scope=scope)[0]
    state=sensors.bind(state,h)
    return sensors,state,h,rows,calls


def validate_view(value,sensors,state,h):
    inventory={row.sha256:row.to_dict() for row in sensors.raw_rows.values()}
    first,last=value['primary_native_frames']
    pose.validate_focal_pose_view(json.loads(json.dumps(value)),inventory[value['primary_raw_observation_sha256'][0]],
        inventory[value['primary_raw_observation_sha256'][1]],raw_inventory=inventory,
        source_domain=sensors.session.domain.to_dict(),hypothesis=h,state_key=state.key)


def test_normal_fixed_focal_endpoint_consumes_actual_middle_poses_as_new_type():
    sensors,state,h,rows,_=scene_world(scope='exit')
    endpoint=sensors.endpoint(state,outgoing=True)
    value=endpoint['registered_pose_sequence']
    assert endpoint['joint_pixel_link'] is None
    assert value['native_source_measurement']['native_frame_indices']==[4,5,6,7]
    assert value['actual_output_samples'][1]['native_frame_index']==5
    assert value['same_source_arm_pixel_path_verified'] is False
    validate_view(value,sensors,state,h)


def test_normal_whole_focal_dense_uses_new_type_every_distinct_native_step():
    sensors,state,h,_,_=scene_world()
    dense=sensors.dense((state,state,state))[0]
    for row in dense['rows'][1:]:
        value=row['required_body_temporal_link']
        assert value['contract']==pose.VIEW_CONTRACT
        assert len(value['native_source_measurement']['native_frame_indices'])==3
        assert not value['formation_pixel_motion_authority']
        validate_view(value,sensors,state,h)


@pytest.mark.parametrize('mutate',[
    lambda v:v.update(same_source_arm_pixel_path_verified=True),
    lambda v:v['native_source_measurement']['raw_observation_sha256'].pop(1),
    lambda v:v['native_source_measurement']['native_pose_window']['anchors'][1].update(measurement_seconds=.123),
    lambda v:v['actual_output_samples'][1].update(native_frame_index=0),
    lambda v:v['actual_output_samples'][1].update(output_seconds=.123),
    lambda v:v['actual_output_samples'][1]['projected_required_joint_xy'][0].__setitem__(1,.123),
    lambda v:v.update(required_part_names=['left_arm']),
    lambda v:v.update(selected_head_ids=[0,1]),
])
def test_serialized_new_pose_receipt_cannot_rehash_different_middle_duties_clock_camera_or_authority(mutate):
    sensors,state,h,_,_=scene_world(scope='exit')
    value=deepcopy(sensors.endpoint(state,outgoing=True)['registered_pose_sequence'])
    mutate(value)
    with pytest.raises(ValueError):validate_view(value,sensors,state,h)


@pytest.mark.parametrize('change',[dict(jitter=True),dict(flat=True)])
def test_normal_new_sensor_negative_never_falls_back_to_old_pixel_certificate(change):
    sensors,state,h,_,_=scene_world(**change)
    with pytest.raises(ValueError):sensors.endpoint(state,outgoing=False)
    values=[v for v in sensors.link_cache.values() if v.get('contract')==pose.VIEW_CONTRACT]
    assert len(values)==1 and values[0]['measurement_available'] is False
    assert not sensors.endpoint_cache


def test_repeated_native_output_is_not_an_extra_model_sample_or_new_motion():
    sensors,state,h,_,_=scene_world(rate=Fraction(30000,1001),output_fps=59.94)
    dense=sensors.dense((state,state,state))[0]
    for row in dense['rows'][1:]:
        if row['native_repeated_alias']:
            assert row['required_body_temporal_link'] is None
        else:
            value=row['required_body_temporal_link'];validate_view(value,sensors,state,h)
            assert len(set(value['native_source_measurement']['native_frame_indices']))==3


def test_fixed_incident_pose_scope_cannot_borrow_a_third_pose_from_other_end_of_shot():
    sensors,state,h,rows,calls=scene_world(scope='exit')
    with pytest.raises(ValueError,match='scope'):
        pose.focal_pose_view(sensors,state,rows[0],rows[1],output_ticks=(0,1),
            selected_head_ids=(0,0),required_part_names=('left_arm','right_arm'))


def test_source_native_measurement_does_not_change_when_final_replay_omits_unused_audio_alignment():
    sensors,state,h,_,_=scene_world(scope='exit')
    sensors.session.domain=replace(sensors.session.domain,audio_offsets=(7.,-2.))
    first=sensors.endpoint(state,outgoing=True)['registered_pose_sequence']
    validate_view(first,sensors,state,h)
    other,other_state,other_h,_,_=scene_world(scope='exit')
    second=other.endpoint(other_state,outgoing=True)['registered_pose_sequence']
    assert first==second
    assert first['native_source_measurement']['native_measurement_domain_excludes_audio_phase_authority']


def test_camera_views_share_native_sensor_but_reproject_actual_output_points():
    from crosscut.render import EditorialTransformTrack,EditorialTransformKeyframe
    sensors,state,h,_,_=scene_world(scope='exit')
    first=sensors.endpoint(state,outgoing=True)['registered_pose_sequence']
    before=dict(sensors.session.budget.counts)
    zoomed=replace(state,segments=(replace(state.segments[0],editorial_transform=
        EditorialTransformTrack((EditorialTransformKeyframe(0.,1.05),))),))
    second=sensors.endpoint(zoomed,outgoing=True)['registered_pose_sequence']
    assert first['native_source_measurement']==second['native_source_measurement']
    assert first['actual_output_samples']!=second['actual_output_samples']
    assert first['state_key']!=second['state_key']
    assert sensors.session.budget.counts['pose_comparisons']==before['pose_comparisons']
    assert sensors.session.budget.counts['association_nodes']==before['association_nodes']
    validate_view(second,sensors,zoomed,h)


def test_normal_dense_consumer_dispatches_new_registered_type_without_legacy_pixel_authority():
    from crosscut.group_scene_observations import validate_scene_dense,DENSE_CONTRACT
    from engine.tests.test_scene_source_owner import scene_derivative_fixture
    sensors,state,h,_,_=scene_world()
    hypotheses=[]
    for source_index in (1,0,2):
        hypothesis=deepcopy(h);hypothesis['span_key'][0]=source_index
        hypotheses.append(hypothesis)
    keys=[pose.digest(item) for item in hypotheses]
    dense=dict(contract=DENSE_CONTRACT,state_keys=['explicit-fixed-left',state.key,'explicit-fixed-right'],
        scene_hypotheses=dict(zip(keys,hypotheses)),hypothesis_keys=keys,
        source_derivative_binding={'audit':scene_derivative_fixture(3).to_dict()},
        source_domain=sensors.session.domain.to_dict(),scene_discovery=sensors.archive(),
        mutable_shots=json.loads(json.dumps(sensors.dense((state,state,state)))))
    dense['scene_discovery']=sensors.archive()
    validate_scene_dense(dense)
    # Received, internally rehashed changes must still fail the normal consumer.
    for mutate in (
            lambda d:d['mutable_shots'][0]['rows'][1]['required_body_temporal_link']
                ['actual_output_samples'][0]['projected_required_joint_xy'][0].__setitem__(1,.123),
            lambda d:d['mutable_shots'][0]['rows'][1]['required_body_temporal_link']
                .update(same_source_arm_pixel_path_verified=True)):
        changed=deepcopy(dense);mutate(changed)
        with pytest.raises(ValueError):validate_scene_dense(changed)
