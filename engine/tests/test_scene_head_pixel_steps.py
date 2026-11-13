"""Explicit nonhuman landmarks and rasters, not performer/editorial approval."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from time import monotonic
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from crosscut.group_joint_source_camera import JointGroupWorkBudget, JointGroupBudgetExhausted
from crosscut.group_joint_primitive_journal import validate_journal, primitive_inputs
from crosscut.group_scene_region_storage import RegionPrimitiveInventory
from engine.tests.test_group_joint_temporal_pose import NativeTemporalPoseTests
from engine.tests.test_scene_known_part_motion import fixture as declared_fixture


def api():
    from crosscut import group_scene_head_pixel_flow as head
    return head


def fixture(*,gap=3,duplicate=False,flat_middle=False):
    declared, state = declared_fixture(textured=True)
    domain = NativeTemporalPoseTests().domain()
    original = declared.raw(state,0,None)
    pixels = declared.session.native_pixels(0,0)
    frames = {n:pixels for n in range(gap+1)}
    if flat_middle:frames[1] = np.zeros_like(pixels)
    rows = tuple(replace(original,native_frame_index=n,source_clock_identity=
        sha256(json.dumps(domain.clocks_by_request[0].to_dict(),
            sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        native_pixel_sha256=sha256(memoryview(frames[n])).hexdigest()) for n in (0,gap))
    if duplicate:
        rows = (rows[0],replace(rows[1],heads=(rows[1].heads[0],
            replace(rows[1].heads[0],observation_id=1))))
    reads=[]
    def read(source,native):
        reads.append((source,native))
        return frames.get(native)
    sensors = SimpleNamespace(session=SimpleNamespace(domain=domain,
        budget=JointGroupWorkBudget(absolute_deadline=monotonic()+30.),native_pixels=read))
    return sensors,rows,reads,frames


def validate(value,sensors,rows):
    journal=sensors.session.budget.primitive_journal.to_dict()
    validate_journal(journal)
    return api().validate_head_pixel_link(value,RegionPrimitiveInventory(journal),
        {row.sha256:row.to_dict() for row in rows})


def force_direct_failure(monkeypatch):
    original=cv2.calcOpticalFlowPyrLK
    calls=[]
    def callback(*args,**kwargs):
        calls.append(kwargs)
        return (None,None,None) if len(calls)==1 else original(*args,**kwargs)
    monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',callback)
    return calls


def test_direct_measured_link_keeps_criteria_and_does_not_read_unneeded_middle():
    head=api();sensors,rows,reads,_=fixture()
    value=head.measure_head_pixel_links(sensors,*rows)
    assert value['unique_links']==[[0,0]]
    assert value['adjacent_recovery_requested'] is False
    assert reads==[(0,0),(0,3)]
    assert len(sensors.session.budget.primitive_journal.attempts)==2
    assert sensors.session.budget.counts['model_calls']==0
    assert validate(value,sensors,rows)==value


def test_missing_direct_link_can_use_every_actual_native_step_without_new_model(monkeypatch):
    head=api();sensors,rows,reads,_=fixture()
    calls=force_direct_failure(monkeypatch)
    value=head.measure_head_pixel_links(sensors,*rows)
    assert value['direct_measured_links']==[]
    assert value['unique_links']==[[0,0]]
    assert value['adjacent_recovery_requested'] is True
    assert {native for _,native in reads}=={0,1,2,3}
    assert [step['native_step'] for step in value['adjacent_steps']]==[[0,1],[1,2],[2,3]]
    assert all(call['maxLevel']==4 and call['winSize']==(15,15) for call in calls)
    assert sensors.session.budget.counts['model_calls']==0
    assert len(sensors.session.budget.primitive_journal.attempts)==7
    assert validate(value,sensors,rows)==value
    assert value['cross_camera_identity_certified'] is value['quality_approved'] is False


def test_middle_occlusion_cannot_be_skipped_to_matching_detector_endpoints(monkeypatch):
    head=api();sensors,rows,_,_=fixture(flat_middle=True)
    force_direct_failure(monkeypatch)
    value=head.measure_head_pixel_links(sensors,*rows)
    assert value['unique_links']==[]
    assert validate(value,sensors,rows)==value


def test_competing_endpoint_heads_stay_ambiguous_and_are_not_deleted():
    head=api();sensors,rows,_,_=fixture(duplicate=True)
    value=head.measure_head_pixel_links(sensors,*rows)
    assert value['measured_links']==[[0,0],[0,1]]
    assert value['unique_links']==[]
    assert validate(value,sensors,rows)==value


@pytest.mark.parametrize('kind',['source','clock','file','raster','endpoint_pixels','order'])
def test_foreign_changed_or_reversed_native_pair_cannot_become_a_path(kind):
    head=api();sensors,rows,reads,frames=fixture()
    a,b=rows
    if kind=='source':b=replace(b,request_source_index=1)
    elif kind=='clock':b=replace(b,source_clock_identity='b'*64)
    elif kind=='file':b=replace(b,source_file_identity=(1,2,3,4,6))
    elif kind=='raster':b=replace(b,frame_size_wh=(640,360))
    elif kind=='endpoint_pixels':frames[3]=np.zeros_like(frames[3])
    else:a,b=b,a
    with pytest.raises(ValueError):head.measure_head_pixel_links(sensors,a,b)
    assert not sensors.session.budget.primitive_journal.attempts


def test_long_gap_stays_unrequested_not_an_unbounded_recovery():
    head=api();sensors,rows,reads,_=fixture(gap=13)
    value=head.measure_head_pixel_links(sensors,*rows)
    assert value['unique_links']==[]
    assert value['reason']=='native_head_gap_unrequested'
    assert reads==[]
    assert validate(value,sensors,rows)==value


def test_real_broadcaster_cut_is_not_bridged_by_similar_heads():
    head=api();sensors,rows,reads,_=fixture()
    sensors.session.domain=replace(sensors.session.domain,scene_cut_times=((.05,),()))
    value=head.measure_head_pixel_links(sensors,*rows)
    assert value['unique_links']==[]
    assert value['reason']=='native_broadcaster_cut'
    assert not reads


@pytest.mark.parametrize('mutation',['promote','link','raw_pair','drop_ref','input','step','raster'])
def test_received_head_link_cannot_resign_a_missing_primitive_or_different_input(mutation,monkeypatch):
    head=api();sensors,rows,_,_=fixture()
    force_direct_failure(monkeypatch)
    value=deepcopy(head.measure_head_pixel_links(sensors,*rows))
    journal=deepcopy(sensors.session.budget.primitive_journal.to_dict())
    if mutation=='promote':value['quality_approved']=True
    elif mutation=='link':value['unique_links']=[[0,42]]
    elif mutation=='raw_pair':value['raw_pair'][1]='b'*64
    elif mutation=='drop_ref':value['adjacent_steps'][0]['backward_primitive_ref']=None
    elif mutation=='input':value['direct_steps'][0]['forward_primitive_ref']=value['adjacent_steps'][0]['forward_primitive_ref']
    elif mutation=='step':value['adjacent_steps'][0]['native_step']=[0,2]
    else:value['adjacent_steps'][0]['target_pixel_sha256']='b'*64
    with pytest.raises(ValueError):
        head.validate_head_pixel_link(value,RegionPrimitiveInventory(journal),
            {row.sha256:row.to_dict() for row in rows})


def test_forward_return_is_retained_if_deadline_prevents_backward_execution(monkeypatch):
    head=api();sensors,rows,_,_=fixture()
    now=[0.]
    sensors.session.budget=JointGroupWorkBudget(absolute_deadline=1.,clock=lambda:now[0])
    original=cv2.calcOpticalFlowPyrLK
    def forward(*args,**kwargs):
        result=original(*args,**kwargs)
        now[0]=2.
        return result
    monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',forward)
    with pytest.raises(JointGroupBudgetExhausted):head.measure_head_pixel_links(sensors,*rows)
    journal=sensors.session.budget.primitive_journal.to_dict()
    validate_journal(journal)
    assert len(journal['attempts'])==1
    assert journal['attempts'][0]['status']=='returned'
    assert journal['attempts'][0]['label']=='opencv_head_lk_forward'
    assert journal['blobs']
    assert all(amount==0 for amount in sensors.session.budget.reserved.values())
    binding=primitive_inputs(journal['attempts'][0])
    head.validate_head_primitive_inputs(binding,journal['attempts'][0]['label'],
        {row.sha256:row.to_dict() for row in rows})


def test_scene_sensor_caches_complete_link_and_never_refunds_primitive_executions():
    head=api();sensors,rows,reads,_=fixture()
    from crosscut.group_scene_observations import SceneSensors
    owned=object.__new__(SceneSensors)
    owned.session=sensors.session;owned.link_cache={}
    value=owned._head_links(*rows)
    before=deepcopy(sensors.session.budget.to_dict())
    assert owned._head_links(*rows)==value
    assert sensors.session.budget.to_dict()==before
    assert len(reads)==2
    assert value['contract']==head.CONTRACT


@pytest.mark.parametrize('error,expected',[(1.24,[[0,0]]),(1.26,[])])
def test_forward_backward_limit_is_still_125_original_pixels(error,expected,monkeypatch):
    head=api();sensors,rows,_,_=fixture(gap=1)
    calls=[]
    def flow(a,b,points,*args,**kwargs):
        calls.append(None)
        moved=points.copy()
        if len(calls)%2==0:moved[:,:,0]+=error
        return moved,np.ones((len(points),1),np.uint8),None
    monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',flow)
    value=head.measure_head_pixel_links(sensors,*rows)
    assert value['unique_links']==expected
    assert validate(value,sensors,rows)==value


@pytest.mark.parametrize('unsupported,expected',[(1,[[0,0]]),(2,[])])
def test_four_same_observed_landmarks_are_needed_not_three(unsupported,expected,monkeypatch):
    head=api();sensors,rows,_,_=fixture(gap=1)
    def flow(a,b,points,*args,**kwargs):
        status=np.ones((len(points),1),np.uint8);status[:unsupported]=0
        return points.copy(),status,None
    monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',flow)
    value=head.measure_head_pixel_links(sensors,*rows)
    assert value['unique_links']==expected
    assert validate(value,sensors,rows)==value


@pytest.mark.parametrize('kind',['shape','nan'])
def test_malformed_numeric_return_is_retained_but_cannot_invent_backward_work(kind,monkeypatch):
    head=api();sensors,rows,_,_=fixture(gap=1)
    def flow(a,b,points,*args,**kwargs):
        result=points[:4].copy() if kind=='shape' else np.full_like(points,np.nan)
        return result,np.ones((len(result),1),np.uint8),None
    monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',flow)
    value=head.measure_head_pixel_links(sensors,*rows)
    assert value['unique_links']==[]
    assert value['direct_steps'][0]['backward_primitive_ref'] is None
    journal=sensors.session.budget.primitive_journal.to_dict()
    assert len(journal['attempts'])==1 and journal['attempts'][0]['raw_result_retained'] is True
    assert validate(value,sensors,rows)==value


def test_same_file_claim_cannot_borrow_a_decoder_with_a_changed_file_identity():
    head=api();sensors,rows,_,_=fixture()
    sensors.session.identities={0:(1,2,3,4,99)}
    with pytest.raises(ValueError,match='current native decoder'):
        head.measure_head_pixel_links(sensors,*rows)
    assert not sensors.session.budget.primitive_journal.attempts


def test_final_independently_detected_landmarks_must_still_agree_with_pixels():
    head=api();sensors,rows,_,_=fixture(gap=1)
    last=rows[1]
    shifted=replace(last.heads[0],landmarks_xy=tuple((x+.2,y) for x,y in last.heads[0].landmarks_xy))
    rows=(rows[0],replace(last,heads=(shifted,)))
    value=head.measure_head_pixel_links(sensors,*rows)
    assert value['unique_links']==[]
    assert validate(value,sensors,rows)==value
