"""Lossless normalized observation wire, not NN raw-return or human quality proof."""
from copy import deepcopy
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
