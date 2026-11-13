"""Nonrigid chart-region motion is not an arm/body/identity certificate."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace as NS
from time import monotonic
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from crosscut.group_joint_source_camera import JointGroupWorkBudget,JointGroupBudgetExhausted
from crosscut.group_joint_formation import ObservedFormationPerson
from crosscut.group_joint_native_observer import NativeGroupObservation
from crosscut import group_scene_composition as composition
import weakref


def world(*,flat=False,duplicate=False,empty=False):
    rng=np.random.default_rng(112)
    base=np.zeros((180,320,3),np.uint8)
    boxes=((.10,.12,.28,.75),(.60,.12,.28,.75))
    for x,y,w,h in boxes:
        if not flat:base[22:156,round(x*320):round((x+w)*320)]=rng.integers(20,255,
            (134,round((x+w)*320)-round(x*320),3),np.uint8)
    images={}
    for native in range(4):
        yy,xx=np.indices((180,320),dtype=np.float32)
        # Opposite local horizontal motions across the ROI cannot be one
        # whole-body rigid affine, but every actual point has image support.
        dx=(yy>=90).astype(np.float32)*2*native-(yy<90).astype(np.float32)*2*native
        images[native]=cv2.remap(base,xx-dx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    people=tuple(ObservedFormationPerson(i,b,(),.99) for i,b in enumerate(boxes))
    if duplicate:people=(*people,replace(people[0],observation_id=2))
    if empty:people=()
    rows=tuple(NativeGroupObservation(0,n,(320,180),'c'*64,(1,2,3,4,5),
        sha256(memoryview(image)).hexdigest(),(('nanodet_people','a'*64),),people,(),
        'EXPLICIT_NONHUMAN_RECTANGLES',(('nanodet_people','b'*64),)) for n,image in images.items())
    budget=JointGroupWorkBudget(absolute_deadline=monotonic()+30.)
    sensors=NS(link_cache={},session=NS(budget=budget,native_pixels=lambda source,n:images[n]))
    return sensors,rows,images


def test_nonrigid_native_region_measurement_reaches_structural_consumer_without_body_authority():
    from crosscut.group_scene_region_flow import measure_region_distribution,validate_region_distribution
    sensors,rows,_=world();value=measure_region_distribution(sensors,rows[0],rows[3])
    assert value['observed_support_paths'] is True
    assert len(value['assignments'])==2
    assert value['articulated_motion_certified'] is False
    assert value['anatomical_extent_certified'] is False
    assert value['rigid_body_transform_certified'] is False
    assert value['quality_approved'] is False
    validate_region_distribution(value,rows[0].to_dict(),rows[3].to_dict())
    assert sensors.session.budget.counts['region_flow_feature_updates']>0
    spent=dict(sensors.session.budget.counts)
    assert measure_region_distribution(sensors,rows[0],rows[3]) is value
    assert sensors.session.budget.counts==spent


@pytest.mark.parametrize('change',['flat','duplicate','empty','missing'])
def test_unknown_or_competing_regions_do_not_gain_supported_motion(change):
    from crosscut.group_scene_region_flow import measure_region_distribution
    sensors,rows,images=world(flat=change=='flat',duplicate=change=='duplicate',empty=change=='empty')
    if change=='missing':images[1]=None
    value=measure_region_distribution(sensors,rows[0],rows[3])
    assert value['observed_support_paths'] is False
    assert value['unmatched_body_observation_ids']


@pytest.mark.parametrize('change',['source','clock','hash','source_feature','flow','live','assignment','policy','authority'])
def test_received_region_distribution_must_replay_exact_bound_algebra(change):
    from crosscut.group_scene_region_flow import measure_region_distribution,validate_region_distribution
    from crosscut.group_scene_region_storage import expand_region_tracks
    sensors,rows,_=world();value=deepcopy(expand_region_tracks(measure_region_distribution(sensors,rows[0],rows[3])))
    assert value['observed_support_paths']
    if change=='source':value['request_source_index']=1
    elif change=='clock':value['native_frame_indices'][0]+=1
    elif change=='hash':value['native_pixel_sha256'][0]='c'*64
    elif change=='source_feature':value['region_tracks'][0]['seed_points_xy'][0][0]=0.
    elif change=='flow':
        for attempt in value['region_tracks'][0]['steps'][-1]['attempts']:
            if attempt['tracked_target_xy'] is not None:attempt['tracked_target_xy'][0]+=120.
    elif change=='live':value['region_tracks'][0]['steps'][0]['usable_feature_ids']=[]
    elif change=='assignment':value['assignments'][1]=value['assignments'][0]
    elif change=='policy':value['policy_sha256']='f'*64
    else:value['articulated_motion_certified']=True
    with pytest.raises(ValueError):validate_region_distribution(value,rows[0].to_dict(),rows[3].to_dict())


def test_region_feature_work_has_a_separate_nonrenewable_limit_before_tracking():
    from crosscut.group_scene_region_flow import measure_region_distribution
    sensors,rows,_=world()
    sensors.session.budget=JointGroupWorkBudget(absolute_deadline=monotonic()+30.,maximum_region_flow_feature_updates=1)
    with pytest.raises(JointGroupBudgetExhausted),patch.object(
            cv2,'calcOpticalFlowPyrLK',side_effect=AssertionError('budget must stop before flow')):
        measure_region_distribution(sensors,rows[0],rows[3])
    assert sensors.session.budget.counts['region_flow_feature_updates']==0


@pytest.mark.parametrize('limit',[0,-1,True,1.5,131073])
def test_region_feature_limit_requires_an_explicit_bounded_integer(limit):
    with pytest.raises(ValueError):
        JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:0.,
            maximum_region_flow_feature_updates=limit)


def test_region_feature_work_and_deadline_are_not_renewed_by_another_frontier():
    now=[0.]
    budget=JointGroupWorkBudget(absolute_deadline=10.,clock=lambda:now[0],
        maximum_frontiers=3,maximum_states=72,maximum_region_flow_feature_updates=6)
    budget.begin_frontier(label='first',maximum_states=24,maximum_edges=128)
    budget.spend('region_flow_feature_updates',4)
    budget.begin_frontier(label='second',maximum_states=24,maximum_edges=128)
    assert budget.counts['region_flow_feature_updates']==4
    budget.spend('region_flow_feature_updates',2)
    with pytest.raises(JointGroupBudgetExhausted):budget.spend('region_flow_feature_updates')
    assert budget.counts['region_flow_feature_updates']==6
    now[0]=10.
    with pytest.raises(JointGroupBudgetExhausted):budget.spend('region_flow_feature_updates',0)
    assert budget.deadline==10.


def test_typed_dispatch_is_nonrigid_and_keeps_only_one_decoded_raster_alive():
    sensors,rows,images=world();references=[];reads=[];peak=[0]
    before,after=(composition.NativeCompositionObservation(row.request_source_index,
        row.native_frame_index,row.frame_size_wh,row.source_clock_identity,row.source_file_identity,
        row.native_pixel_sha256,row.model_content_sha256,row.people,(),row.reason,
        row.model_adapter_fingerprints) for row in (rows[0],rows[3]))
    def pixels(source,native):
        sensors.session.budget.spend('new_decodes')
        reads.append((source,native));image=images[native].copy()
        references.append(weakref.ref(image))
        peak[0]=max(peak[0],sum(ref() is not None for ref in references))
        return image
    sensors.session.native_pixels=pixels
    with patch.object(composition,'legacy_body_support_links',
            side_effect=AssertionError('typed native support cannot fall back to rigid success')):
        value=composition.body_support_links(sensors,before,after)
    assert value['observed_support_paths'] is True
    composition.validate_support_link(value,before.to_dict(),after.to_dict())
    assert reads==[(0,native) for native in range(4)]
    assert peak[0]<=1 and not any(ref() is not None for ref in references)


def test_flat_typed_regions_stay_unknown_without_a_success_fallback():
    sensors,rows,_=world(flat=True)
    before,after=(composition.NativeCompositionObservation(row.request_source_index,
        row.native_frame_index,row.frame_size_wh,row.source_clock_identity,row.source_file_identity,
        row.native_pixel_sha256,row.model_content_sha256,row.people,(),row.reason,
        row.model_adapter_fingerprints) for row in (rows[0],rows[3]))
    with patch.object(composition,'legacy_body_support_links',
            side_effect=AssertionError('unknown typed motion must not gain rigid fallback')):
        value=composition.body_support_links(sensors,before,after)
    assert value['observed_support_paths'] is False
    assert value['unmatched_body_observation_ids']==[[0,1],[0,1]]
