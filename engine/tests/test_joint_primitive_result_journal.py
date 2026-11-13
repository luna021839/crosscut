"""Local primitive-prefix accounting, not a controller history or human QA.

Real OpenCV calls use nonhuman textured rasters. No IVE model or media runs.
"""
from copy import deepcopy
from time import monotonic
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from crosscut.group_joint_source_camera import JointGroupWorkBudget,JointGroupBudgetExhausted
from engine.tests.test_scene_region_flow import world


def primitive(budget,produce,*,maximum_result_bytes=4096,work=None):
    return budget.measure_primitive(label='EXPLICIT_NONHUMAN_PRIMITIVE',
        input_binding={'source':0,'native':1,'kind':'fixture_not_network'},
        work={'pose_comparisons':1} if work is None else work,
        maximum_result_bytes=maximum_result_bytes,producer=produce)


def test_insufficient_result_reservation_executes_nothing():
    budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.,maximum_pose_evidence_bytes=128)
    calls=[]
    with pytest.raises(JointGroupBudgetExhausted):primitive(budget,lambda:calls.append(1))
    assert calls==[] and not any(budget.counts.values())


def test_return_is_committed_before_a_postwork_deadline_can_reject_it():
    now=[0.];budget=JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:now[0])
    raw=np.asarray([1.,float('nan'),float('inf'),-0.],dtype='>f4')
    def produce():now[0]=10.;return raw
    assert primitive(budget,produce) is raw
    snapshot=budget.primitive_journal.to_dict()
    from crosscut.group_joint_primitive_journal import validate_journal,restore_numeric_return
    validate_journal(snapshot)
    attempt=snapshot['attempts'][0]
    restored=restore_numeric_return(snapshot['blobs'][attempt['raw_result_sha256']])
    assert restored.dtype.str==raw.dtype.str and restored.tobytes()==raw.tobytes()
    assert attempt['status']=='returned'
    with pytest.raises(JointGroupBudgetExhausted):budget.spend('pose_comparisons')
    assert budget.counts['pose_comparisons']==1


def test_identical_raw_blob_is_stored_once_but_each_execution_and_record_is_charged():
    budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.)
    raw=np.arange(2000,dtype=np.uint8);calls=[]
    def produce():calls.append(1);return raw.copy()
    primitive(budget,produce,maximum_result_bytes=8192)
    first=budget.counts['pose_evidence_bytes']
    primitive(budget,produce,maximum_result_bytes=8192)
    snapshot=budget.primitive_journal.to_dict()
    assert len(snapshot['blobs'])==1 and len(snapshot['attempts'])==2
    assert calls==[1,1] and budget.counts['pose_comparisons']==2
    assert first<budget.counts['pose_evidence_bytes']<2*first
    assert snapshot['controller_owned'] is False


def test_fresh_execution_order_does_not_change_a_result_occurrence_reference():
    from crosscut.group_joint_primitive_journal import validate_journal
    first=JointGroupWorkBudget(absolute_deadline=monotonic()+10.)
    second=JointGroupWorkBudget(absolute_deadline=monotonic()+10.)
    calls=[]
    def produce():calls.append(1);return np.arange(8,dtype=np.float32)
    primitive(first,produce)
    primitive(second,lambda:np.asarray([99],np.int16))
    primitive(second,produce)
    a=first.primitive_journal.to_dict();b=second.primitive_journal.to_dict()
    validate_journal(a);validate_journal(b)
    assert a['attempts'][0]['sequence']!=b['attempts'][-1]['sequence']
    assert a['attempts'][0]['measurement_ref_sha256']==b['attempts'][-1]['measurement_ref_sha256']
    assert calls==[1,1] and first.counts['pose_comparisons']==1 and second.counts['pose_comparisons']==2


def test_primitive_reference_uses_inputs_frozen_before_the_callback():
    from crosscut.group_joint_primitive_journal import validate_journal,primitive_inputs
    budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.)
    binding={'native':1,'kind':'explicit_fixture'}
    def produce():binding['native']=99;return np.asarray([1.],np.float32)
    budget.measure_primitive(label='EXPLICIT_NONHUMAN_PRIMITIVE',input_binding=binding,
        work={'pose_comparisons':1},maximum_result_bytes=4096,producer=produce)
    journal=budget.primitive_journal.to_dict();validate_journal(journal)
    assert primitive_inputs(journal['attempts'][0])['native']==1


def test_backend_failure_has_a_reserved_terminal_and_no_fake_result():
    budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.)
    def produce():raise RuntimeError('EXPLICIT_BACKEND_FAILURE')
    with pytest.raises(RuntimeError):primitive(budget,produce)
    attempt=budget.primitive_journal.to_dict()['attempts'][0]
    assert attempt['status']=='backend_failed' and attempt['raw_result_sha256'] is None
    assert not any(budget.reserved.values()) and budget.counts['pose_evidence_bytes']>0


def test_reserved_capacity_cannot_be_spent_again_or_settled_by_a_foreign_ticket():
    budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.,maximum_pose_evidence_bytes=8192)
    ticket=budget.reserve_primitive({'pose_comparisons':1},7000)
    with pytest.raises(JointGroupBudgetExhausted):budget.spend('pose_evidence_bytes',2000)
    with pytest.raises(ValueError):budget.settle_primitive(object(),0)
    assert budget.reserved['pose_evidence_bytes']==7000 and budget.counts['pose_evidence_bytes']==0
    budget.start_primitive(ticket)
    budget.settle_primitive(ticket,10)
    assert not any(budget.reserved.values()) and budget.counts['pose_evidence_bytes']==10
    with pytest.raises(ValueError):budget.settle_primitive(ticket,0)


def test_unexpected_oversized_return_keeps_terminal_without_claiming_complete_raw_bits():
    from crosscut.group_joint_primitive_journal import validate_journal
    budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.)
    with pytest.raises(ValueError,match='predeclared byte bound'):
        primitive(budget,lambda:np.zeros(1000,dtype=np.float32),maximum_result_bytes=32)
    snapshot=budget.primitive_journal.to_dict();validate_journal(snapshot)
    assert snapshot['attempts'][0]['status']=='return_contract_violated'
    assert not snapshot['blobs'] and not any(budget.reserved.values())


def test_received_blob_corruption_is_rejected_without_history_authentication():
    from crosscut.group_joint_primitive_journal import validate_journal
    budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.)
    primitive(budget,lambda:np.arange(4,dtype=np.float32))
    snapshot=budget.primitive_journal.to_dict();validate_journal(snapshot)
    bad=deepcopy(snapshot);key=next(iter(bad['blobs']))
    bad['blobs'][key]['bytes_base64']='AA=='
    with pytest.raises(ValueError):validate_journal(bad)
    assert snapshot['pre_receipt_history_completeness_certified'] is False


@pytest.mark.parametrize('evidence_capacity',[128,9000])
def test_region_raw_reservation_rejects_before_native_reads_or_gftt_lk(evidence_capacity):
    from crosscut.group_scene_region_flow import measure_region_distribution
    sensors,rows,_=world();reads=[]
    sensors.session.budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.,
        maximum_pose_evidence_bytes=evidence_capacity)
    sensors.session.native_pixels=lambda *args:reads.append(args)
    with patch.object(cv2,'goodFeaturesToTrack',side_effect=AssertionError('GFTT must not run')),\
            patch.object(cv2,'calcOpticalFlowPyrLK',side_effect=AssertionError('LK must not run')),\
            pytest.raises(JointGroupBudgetExhausted):
        measure_region_distribution(sensors,rows[0],rows[3])
    assert reads==[]


def test_actual_forward_lk_prefix_survives_unexecuted_backward_after_deadline():
    from crosscut.group_scene_region_flow import measure_region_distribution
    from crosscut.group_joint_primitive_journal import restore_numeric_return
    sensors,rows,_=world();now=[0.];calls=[]
    sensors.session.budget=JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:now[0])
    real=cv2.calcOpticalFlowPyrLK
    def forward(*args,**kwargs):
        value=real(*args,**kwargs);calls.append(value);now[0]=10.;return value
    with patch.object(cv2,'calcOpticalFlowPyrLK',side_effect=forward),pytest.raises(JointGroupBudgetExhausted):
        measure_region_distribution(sensors,rows[0],rows[3])
    snapshot=sensors.session.budget.primitive_journal.to_dict()
    completed=[a for a in snapshot['attempts'] if a['label']=='opencv_region_lk_forward']
    assert len(calls)==len(completed)==1
    raw=restore_numeric_return(snapshot['blobs'][completed[0]['raw_result_sha256']])
    assert all(a.tobytes()==b.tobytes() for a,b in zip(raw,calls[0]))
    assert not any(a['label']=='opencv_region_lk_backward' for a in snapshot['attempts'])
    assert not sensors.link_cache


def test_actual_forward_lk_prefix_survives_insufficient_backward_feature_reservation():
    from crosscut.group_scene_region_flow import measure_region_distribution
    sensors,rows,_=world();calls=[];real=cv2.calcOpticalFlowPyrLK
    sensors.session.budget=JointGroupWorkBudget(absolute_deadline=monotonic()+10.,
        maximum_region_flow_feature_updates=80)
    def forward(*args,**kwargs):
        value=real(*args,**kwargs);calls.append(value);return value
    with patch.object(cv2,'calcOpticalFlowPyrLK',side_effect=forward),pytest.raises(JointGroupBudgetExhausted):
        measure_region_distribution(sensors,rows[0],rows[3])
    snapshot=sensors.session.budget.primitive_journal.to_dict()
    assert len(calls)==1
    assert sensors.session.budget.counts['region_flow_feature_updates']==len(calls[0][0])
    assert snapshot['attempts'][-1]['label']=='opencv_region_lk_forward'
    assert snapshot['attempts'][-1]['raw_result_retained'] is True
    assert not any(sensors.session.budget.reserved.values())


def test_roi_track_cache_and_wire_retain_actual_lossless_compressed_bytes():
    from crosscut.group_scene_region_flow import measure_region_distribution,validate_region_distribution
    from crosscut.group_scene_region_storage import expand_region_tracks,pack_region_tracks
    from crosscut.group_joint_primitive_journal import encoded
    sensors,rows,_=world();value=measure_region_distribution(sensors,rows[0],rows[3])
    assert 'region_tracks' not in value and 'region_track_storage' in value
    assert next(iter(sensors.link_cache.values())) is value
    expanded=expand_region_tracks(value)
    assert encoded(value)==encoded(pack_region_tracks(expanded))
    assert len(encoded(value))<len(encoded(expanded))
    validate_region_distribution(value,rows[0].to_dict(),rows[3].to_dict())
    assert value['region_track_storage']['quality_approved'] is False


@pytest.mark.parametrize('change',['payload','declared_size','duplicate_projection','authority'])
def test_received_roi_wire_corruption_is_not_an_admissible_view(change):
    from crosscut.group_scene_region_flow import measure_region_distribution
    from crosscut.group_scene_region_storage import expand_region_tracks
    sensors,rows,_=world();value=deepcopy(measure_region_distribution(sensors,rows[0],rows[3]))
    stored=value['region_track_storage']
    if change=='payload':stored['lossless_json']['bytes_base64']='AA=='
    elif change=='declared_size':stored['lossless_json']['decoded_byte_count']+=1
    elif change=='duplicate_projection':value['region_tracks']=[]
    else:stored['pre_receipt_history_certified']=True
    with pytest.raises(ValueError):expand_region_tracks(value)


@pytest.mark.parametrize('change',['seed','coordinates','forward_reference','backward_reference','roi_binding'])
def test_received_roi_projection_cannot_invent_different_raw_primitive_returns(change):
    from crosscut.group_scene_region_flow import measure_region_distribution
    from crosscut.group_scene_region_storage import (expand_region_tracks,RegionPrimitiveInventory,
        validate_region_primitive_views,pack_region_tracks)
    from crosscut.group_joint_primitive_journal import validate_journal,primitive_inputs,packed,encoded
    sensors,rows,_=world();value=measure_region_distribution(sensors,rows[0],rows[3])
    journal=sensors.session.budget.primitive_journal.to_dict();validate_journal(journal)
    validate_region_primitive_views(value,RegionPrimitiveInventory(journal))
    expanded=deepcopy(expand_region_tracks(value));track=expanded['region_tracks'][0]
    if change=='seed':track['seed_points_xy'][0][0]+=.25
    elif change=='coordinates':track['steps'][0]['attempts'][0]['tracked_target_xy'][0]+=.25
    elif change=='forward_reference':track['steps'][0]['forward_primitive_ref']=track['seed_primitive_ref']
    elif change=='backward_reference':track['steps'][0]['backward_primitive_ref']=track['steps'][0]['forward_primitive_ref']
    else:
        attempt=next(a for a in journal['attempts'] if a['measurement_ref_sha256']==track['seed_primitive_ref'])
        binding=primitive_inputs(attempt);binding['roi_observation_id']=999
        attempt['input_binding']=packed(encoded(binding))
    with pytest.raises(ValueError):
        validate_region_primitive_views(pack_region_tracks(expanded),RegionPrimitiveInventory(journal))
