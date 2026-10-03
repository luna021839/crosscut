"""Lossless normalized observation wire, not NN raw-return or human quality proof."""
from copy import deepcopy
from dataclasses import asdict,fields
import json
from time import monotonic
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from crosscut.group_joint_source_camera import JointGroupWorkBudget,JointGroupBudgetExhausted
from crosscut.group_scene_native_storage import (pack_native_observation,expand_native_observation,
    pack_native_evidence,validate_normalized_view_records,MAX_NATIVE_JSON_BYTES,NativeObservationView)
from crosscut.group_joint_primitive_journal import encoded
from crosscut import group_performer_family_owner as family
from crosscut import group_source_changing_owner as source
from crosscut.group_scene_observations import SceneSensors
from engine.tests.test_scene_region_flow import world


def capture(budget,row,producer=None):
    return budget.normalized_native_views.measure(label='native_focal_observation_view',
        input_binding=dict(request_source_index=row.request_source_index,
            native_frame_index=row.native_frame_index,source_clock_identity=row.source_clock_identity),
        producer=(lambda:row) if producer is None else producer)


def test_known_native_observation_wire_roundtrips_without_new_sensor_work():
    sensors,rows,_=world();original=rows[0].to_dict();packet=pack_native_observation(original)
    assert encoded(expand_native_observation(packet))==encoded(original)
    assert family.digest(packet)==rows[0].sha256
    nested=dict(native_observation=original,other=dict(contract='unrelated_receipt',number=3))
    stored=pack_native_evidence(nested)
    assert stored['native_observation']['contract']==packet['contract']
    assert family.digest(stored)==family.digest(nested)
    assert family.plain(stored)==json.loads(encoded(nested))
    assert packet['normalized_view_not_raw_nn_or_adapter_return'] is True
    assert packet['controller_owned'] is packet['quality_approved'] is False
    assert not any(sensors.session.budget.counts.values())


def test_read_through_mapping_keeps_only_the_packet_and_replays_original_fields():
    _,rows,_=world();original=rows[0].to_dict();packet=pack_native_observation(original)
    view=NativeObservationView(packet)
    assert set(vars(view))=={'_packet'}
    assert encoded(dict(view.items()))==encoded(original)
    assert view['native_frame_index']==original['native_frame_index']
    assert family.digest(view)==rows[0].sha256
    assert encoded(pack_native_evidence(view))==encoded(packet)
    assert view['quality_approved'] is False


def test_wire_packing_does_not_change_unrelated_in_process_tuple_contracts():
    _,rows,_=world()
    value=dict(native_observation=rows[0].to_dict(),required_parts=((0,'left_arm'),))
    stored=pack_native_evidence(value)
    assert stored['required_parts']==((0,'left_arm'),)
    assert type(stored['required_parts']) is tuple


def test_frozen_motion_stores_wire_but_public_json_keeps_original_observation_schema():
    from crosscut.editorial_motion import EditorialMotionPlan
    _,rows,_=world();original=rows[0].to_dict()
    motion=EditorialMotionPlan((),0,0,0,0,0,source_camera_refresh=dict(native_rows=[original]))
    assert motion.source_camera_refresh['native_rows'][0]['contract']=='lossless_normalized_scene_native_wire_v1'
    public=motion.to_dict()['source_camera_refresh']['native_rows'][0]
    assert encoded(public)==encoded(original)
    assert json.loads(encoded(motion.to_dict()))['source_camera_refresh']['native_rows'][0]==public


def body_support_fixture():
    from crosscut.group_joint_native_observer import NativeGroupObservation
    from crosscut.group_scene_composition import NativeCompositionObservation
    _,rows,_=world();base=rows[0]
    values={field.name:getattr(base,field.name) for field in fields(NativeGroupObservation)}
    values.update(heads=(),
        model_content_sha256=(('nanodet_people','a'*64),),
        model_adapter_fingerprints=(('nanodet_people','c'*64),),
        reason='EXPLICIT_NONHUMAN_BODY_SUPPORT_SCHEMA_FIXTURE')
    return NativeCompositionObservation(**values)


def parallel_fixture():
    from crosscut.group_joint_native_observer import NativeGroupObservation
    from crosscut.group_scene_articulation import NativeCompositionPoseObservation
    from crosscut.group_joint_person_ownership import OwnedPersonObservation
    base=body_support_fixture()
    values={field.name:getattr(base,field.name) for field in fields(NativeGroupObservation)}
    values.update(people=tuple(OwnedPersonObservation(p.observation_id,p.box_xywh,(),p.confidence,
            pose_reason='EXPLICIT_NONHUMAN_UNOBSERVED_POSE') for p in base.people),
        model_content_sha256=(('nanodet_people','a'*64),('rtmpose_body7','b'*64)),
        model_adapter_fingerprints=(('nanodet_people','c'*64),('rtmpose_body7','d'*64)),
        reason='EXPLICIT_NONHUMAN_PARALLEL_SCHEMA_FIXTURE')
    return NativeCompositionPoseObservation(**values)


def test_actual_parallel_articulation_schema_roundtrips_and_survives_archive():
    raw=parallel_fixture();packet=pack_native_observation(raw.to_dict())
    assert encoded(expand_native_observation(packet))==encoded(raw.to_dict())
    sensors=object.__new__(SceneSensors);budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.)
    sensors.session=SimpleNamespace(budget=budget);sensors.raw_rows={};sensors.hypotheses={}
    sensors.discoveries={};sensors.articulation_raw_rows={'parallel':raw};sensors.link_cache={}
    archive=family.plain(sensors.archive())
    assert encoded(archive['parallel_articulation_rows'][0])==encoded(raw.to_dict())
    assert archive['parallel_articulation_rows'][0]['heads_state']=='not_requested'
    assert archive['parallel_articulation_rows'][0]['semantic_cast_membership_certified'] is False


@pytest.mark.parametrize('change',['head','cast_authority','model_names'])
def test_parallel_wire_cannot_invent_focal_or_cast_authority(change):
    raw=parallel_fixture().to_dict()
    if change=='head':
        from crosscut.group_joint_native_observer import NativeGroupHead
        raw['heads']=[asdict(NativeGroupHead(0,(.1,.1,.1,.1),((.1,.1),)*5,.9))]
    elif change=='cast_authority':raw['semantic_cast_membership_certified']=True
    else:raw['model_content_sha256']=[('yunet_head_faces','a'*64)]
    with pytest.raises(ValueError):pack_native_observation(raw)


def test_parallel_producer_reserves_view_before_expensive_pose_work():
    from crosscut import group_scene_articulation as articulation
    from crosscut.group_joint_source_camera import GroupOutputRole
    raw=parallel_fixture();calls=[]
    budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.,maximum_pose_evidence_bytes=128)
    sensors=SimpleNamespace(session=SimpleNamespace(budget=budget,models={'rtmpose_body7':object()},
            model_adapters={'rtmpose_body7':'d'*64}),
        raw=lambda *args:body_support_fixture(),articulation_raw_rows={})
    state=SimpleNamespace(role=GroupOutputRole.COMPOSITION)
    def observe(*args):calls.append(1);return raw
    with patch.object(articulation,'observe_composition_pose',side_effect=observe),pytest.raises(JointGroupBudgetExhausted):
        articulation._raw(sensors,state,0)
    assert not calls and not sensors.articulation_raw_rows
    assert not any(budget.reserved.values())


def test_parallel_return_has_its_own_normalized_view_record_not_focal_identity():
    from crosscut import group_scene_articulation as articulation
    from crosscut.group_joint_source_camera import GroupOutputRole
    raw=parallel_fixture();calls=[];budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.)
    base=body_support_fixture()
    sensors=SimpleNamespace(session=SimpleNamespace(budget=budget,models={'rtmpose_body7':object()},
            model_adapters={'rtmpose_body7':'d'*64}),
        raw=lambda *args:base,articulation_raw_rows={})
    state=SimpleNamespace(role=GroupOutputRole.COMPOSITION)
    def observe(*args):calls.append(1);return raw
    with patch.object(articulation,'observe_composition_pose',side_effect=observe):
        assert articulation._raw(sensors,state,0) is raw
        assert articulation._raw(sensors,state,0) is raw
    record=budget.normalized_native_views.to_dict()
    assert calls==[1] and len(record['attempts'])==1
    assert record['attempts'][0]['label']=='native_composition_parallel_pose_view'
    validate_normalized_view_records(record,{raw.sha256:raw.to_dict(),base.sha256:base.to_dict()})
    assert record['intermediate_nn_raw_returns_retained'] is False


@pytest.mark.parametrize('change',['omitted','focal_body','native','clock'])
def test_parallel_record_cannot_borrow_unbound_body_support(change):
    base=body_support_fixture();raw=parallel_fixture()
    budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.)
    budget.normalized_native_views.measure(label='native_composition_parallel_pose_view',
        input_binding=dict(request_source_index=base.request_source_index,native_frame_index=base.native_frame_index,
            source_clock_identity=base.source_clock_identity,body_support_sha256=base.sha256),producer=lambda:raw)
    inventory={raw.sha256:raw.to_dict(),base.sha256:base.to_dict()}
    if change=='omitted':inventory.pop(base.sha256)
    elif change=='focal_body':inventory[base.sha256]['contract']='actual_source_native_owned_person_body_face_observation_rows_v2'
    elif change=='native':inventory[base.sha256]['native_frame_index']+=1
    else:inventory[base.sha256]['source_clock_identity']='f'*64
    with pytest.raises(ValueError):validate_normalized_view_records(budget.normalized_native_views.to_dict(),inventory)


def test_unregistered_parallel_adapter_remains_not_requested_not_a_failed_view():
    from crosscut import group_scene_articulation as articulation
    from crosscut.group_joint_source_camera import GroupOutputRole
    budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.,maximum_pose_evidence_bytes=128)
    sensors=SimpleNamespace(session=SimpleNamespace(budget=budget,models={'rtmpose_body7':object()},model_adapters={}),
        raw=lambda *args:body_support_fixture(),articulation_raw_rows={})
    with patch.object(articulation,'observe_composition_pose',side_effect=AssertionError('adapter unavailable')):
        assert articulation._raw(sensors,SimpleNamespace(role=GroupOutputRole.COMPOSITION),0) is None
    assert not budget.normalized_native_views.attempts and not any(budget.reserved.values())


@pytest.mark.parametrize('change',['payload','hash','size','authority','unknown_schema'])
def test_native_wire_corruption_is_not_a_valid_observation(change):
    _,rows,_=world();packet=pack_native_observation(rows[0].to_dict())
    if change=='payload':packet['lossless_json']['bytes_base64']='AA=='
    elif change=='hash':packet['observation_sha256']='a'*64
    elif change=='size':packet['lossless_json']['decoded_byte_count']+=1
    elif change=='authority':packet['controller_owned']=True
    else:packet['extra_opaque_authority']=True
    with pytest.raises(ValueError):family.plain(packet)


def test_original_postwork_cap_counterexample_stops_before_the_observer_now():
    budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.,maximum_pose_evidence_bytes=128)
    calls=[]
    def observe(session,**kwargs):
        calls.append(kwargs);session.budget.spend('model_calls',model='explicit_nonhuman_fixture')
        raise AssertionError('insufficient view storage must prevent this producer call')
    sensors=object.__new__(source.NativeFamilySensors);sensors.raw_rows={}
    sensors.session=SimpleNamespace(budget=budget,domain=SimpleNamespace(request_index=lambda i:11))
    state=SimpleNamespace(eligible_source_index=0,native_frame_indices=(3,))
    with patch.object(source,'observe_native_group',side_effect=observe),pytest.raises(JointGroupBudgetExhausted):
        sensors.raw_native(state,3,None)
    assert not calls and budget.counts['model_calls']==0 and not sensors.raw_rows
    assert not budget.normalized_native_views.attempts and not any(budget.reserved.values())


def test_completed_normalized_view_survives_a_later_deadline_without_claiming_nn_prefixes():
    _,rows,_=world();now=[0.]
    budget=JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:now[0])
    def observe():now[0]=10.;return rows[0]
    assert capture(budget,rows[0],observe) is rows[0]
    record=budget.normalized_native_views.to_dict()
    validate_normalized_view_records(record,{rows[0].sha256:rows[0].to_dict()})
    assert record['attempts'][0]['normalized_view_retained'] is True
    assert record['intermediate_nn_raw_returns_retained'] is False
    with pytest.raises(JointGroupBudgetExhausted):budget.spend('model_calls',model='fixture')
    assert not any(budget.reserved.values())


def test_identical_normalized_views_share_storage_not_fresh_execution_charges():
    _,rows,_=world();budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.);calls=[]
    def observe():
        budget.spend('model_calls',model='explicit_nonhuman_fixture');calls.append(1);return rows[0]
    capture(budget,rows[0],observe);first=budget.counts['pose_evidence_bytes']
    capture(budget,rows[0],observe)
    journal=budget.normalized_native_views.to_dict()
    validate_normalized_view_records(journal,{rows[0].sha256:rows[0].to_dict()})
    assert calls==[1,1] and budget.counts['model_calls']==2
    assert len(journal['attempts'])==2 and len(journal['view_wire_bytes'])==1
    assert first<budget.counts['pose_evidence_bytes']<2*first


def test_failed_native_view_records_a_terminal_without_inventing_a_complete_return():
    _,rows,_=world();budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.)
    def fail():raise RuntimeError('EXPLICIT_NONHUMAN_PRODUCER_FAILURE')
    with pytest.raises(RuntimeError):capture(budget,rows[0],fail)
    journal=budget.normalized_native_views.to_dict();validate_normalized_view_records(journal,{})
    assert journal['attempts'][0]['status']=='producer_failed' and not journal['view_wire_bytes']
    assert not any(budget.reserved.values())


def test_unexpected_oversized_view_preserves_an_explicit_incomplete_terminal():
    _,rows,_=world();budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.)
    payload=rows[0].to_dict();payload['reason']='x'*MAX_NATIVE_JSON_BYTES
    with pytest.raises(ValueError,match='declared wire bound'):
        capture(budget,rows[0],lambda:SimpleNamespace(to_dict=lambda:payload))
    journal=budget.normalized_native_views.to_dict();validate_normalized_view_records(journal,{})
    assert journal['attempts'][0]['status']=='view_contract_violated'
    assert not journal['view_wire_bytes'] and not any(budget.reserved.values())


def test_cache_omission_does_not_erase_an_already_returned_local_view_from_archive():
    _,rows,_=world();budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.)
    capture(budget,rows[0])
    sensors=object.__new__(SceneSensors)
    sensors.session=SimpleNamespace(budget=budget)
    sensors.raw_rows={};sensors.hypotheses={};sensors.discoveries={};sensors.articulation_raw_rows={};sensors.link_cache={}
    archive=sensors.archive()
    assert len(archive['native_rows'])==1
    expanded=family.plain(archive)
    inventory={family.digest(r):r for r in expanded['native_rows']}
    validate_normalized_view_records(expanded['normalized_native_view_records'],inventory)
    assert expanded['normalized_native_view_records']['pre_receipt_history_certified'] is False


@pytest.mark.parametrize('change',['omit','clock','source','bytes','authority'])
def test_received_normalized_view_records_need_exact_source_reference_and_byte_inventory(change):
    _,rows,_=world();budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.)
    capture(budget,rows[0]);journal=deepcopy(budget.normalized_native_views.to_dict())
    inventory={rows[0].sha256:rows[0].to_dict()}
    from crosscut.group_joint_primitive_journal import packed,unpacked
    if change=='omit':inventory={}
    elif change in ('source','clock'):
        binding=json.loads(unpacked(journal['attempts'][0]['input_binding'],limit=2048))
        binding['request_source_index' if change=='source' else 'source_clock_identity']=999 if change=='source' else 'b'*64
        journal['attempts'][0]['input_binding']=packed(encoded(binding))
    elif change=='bytes':journal['charged_view_bytes']+=1
    else:journal['intermediate_nn_raw_returns_retained']=True
    with pytest.raises(ValueError):validate_normalized_view_records(journal,inventory)
