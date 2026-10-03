"""Real chart-image consistency; synthetic raw17/owners, not human QA."""
from copy import deepcopy
from dataclasses import replace

import cv2
import numpy as np
import pytest

from crosscut import group_scene_focal_pose_motion as pose
from crosscut.group_joint_person_ownership import bind_person_ownership
from engine.tests import test_scene_focal_pose_motion as fixtures


def texture_away_world(*,impulse=False):
    sensors,state,h,rows,_=fixtures.scene_world(jitter=impulse)
    image=np.zeros((540,960,3),np.uint8)
    image[115:155,250:290]=np.random.default_rng(41).integers(20,255,(40,40,3),np.uint8)
    for n,row in tuple(rows.items()):
        rows[n]=replace(row,native_pixel_sha256=pose.sha256(memoryview(image)).hexdigest())
    sensors.session.native_pixels=lambda *args:image
    sensors.session.projected_pixels=lambda *args:image
    h=deepcopy(h);h['raw_observation_sha256']=[rows[n].sha256 for n in h['anchor_output_ticks']]
    sensors.hypotheses[pose.digest(h)]=h;state=sensors.bind(state,h)
    return sensors,state,h,rows


@pytest.mark.parametrize('impulse',[False,True])
def test_texture_elsewhere_cannot_promote_unknown_required_arm_motion(impulse):
    sensors,state,_,rows=texture_away_world(impulse=impulse)
    value=pose.focal_pose_view(sensors,state,rows[0],rows[3],output_ticks=(0,3),
        selected_head_ids=(0,0),required_part_names=('left_arm','right_arm'))
    assert value['measurement_available'] is False
    assert value['required_motion_available'] is False
    assert value['model_coordinate_measurement_available'] is True
    assert value['reason']=='REQUIRED_PART_IMAGE_MOTION_UNCONSTRAINED'
    assert len(value['actual_output_samples'])==4, 'retain known projected geometry despite unknown motion'
    checks=value['native_source_measurement']['image_consistency']
    assert all(check['image_supported'] for check in checks), 'whole-body contrast is present in this counterexample'
    assert all(p['lk_point_status']=='unknown' for check in checks for p in check['attempts'])
    assert not pose.required_motion_available(value)


def test_unknown_motion_is_not_consumed_by_normal_endpoint_or_dense():
    sensors,state,_,_=texture_away_world(impulse=True)
    for operation in (lambda:sensors.endpoint(state,outgoing=False),
            lambda:sensors.dense((state,state,state))):
        with pytest.raises(ValueError,match='unconstrained|UNCONSTRAINED'):
            operation()
    assert not sensors.endpoint_cache
    assert all(not v['measurement_available'] for v in sensors.link_cache.values()
        if v.get('contract')==pose.VIEW_CONTRACT)


def test_model_coordinates_survive_flat_arm_motion_unknown_without_zero_or_interpolation():
    sensors,rows=fixtures.world(flat=True)
    native=fixtures.measure(sensors,rows).to_dict()
    assert native['model_coordinate_measurement_available'] is True
    assert native['required_motion_available'] is False
    assert native['reason']=='NATIVE_ROI_IMAGE_UNSUPPORTED'
    assert native['pose_or_velocity_imputed'] is False
    assert native['required_duties_removed'] is False


def translated_chart_world():
    sensors,base=fixtures.world();images={};rows=[]
    original=sensors.session.native_pixels(0,0)
    for native,row in enumerate(base):
        dx=12*native;shift=dx/original.shape[1]
        image=cv2.warpAffine(original,np.float32([[1,0,dx],[0,1,0]]),
            (original.shape[1],original.shape[0]),flags=cv2.INTER_LINEAR)
        images[native]=image
        p=row.people[0];raw=tuple((x+shift,y,c) for x,y,c in p.raw_pose_xyc)
        p=replace(p,box_xywh=(p.box_xywh[0]+shift,*p.box_xywh[1:]),raw_pose_xyc=raw,
            upper_body_xyc=tuple(raw[n] for n in (5,6,7,8,9,10,11,12)))
        head=row.heads[0]
        head=replace(head,box_xywh=(head.box_xywh[0]+shift,*head.box_xywh[1:]),
            landmarks_xy=tuple((x+shift,y) for x,y in head.landmarks_xy))
        people,owners=bind_person_ownership((p,),(head,),row.frame_size_wh)
        rows.append(replace(row,people=people,heads=(replace(head,
            candidate_body_observation_ids=tuple(owners[0])),),
            native_pixel_sha256=pose.sha256(memoryview(image)).hexdigest()))
    sensors.session.native_pixels=lambda source,native:images[native]
    return sensors,tuple(rows)


def test_image_constrained_fast_native_motion_is_not_rejected_by_an_arbitrary_speed_cap():
    sensors,rows=translated_chart_world();value=fixtures.measure(sensors,rows).to_dict()
    assert value['model_coordinate_measurement_available'] is True
    assert value['required_motion_available'] is True
    assert value['measurement_available'] is True
    assert all(check['required_part_motion_constrained'] for check in value['image_consistency'])
    assert value['same_source_arm_pixel_path_verified'] is False


def test_received_unknown_part_constraint_cannot_be_promoted_by_available_flags():
    sensors,state,h,rows=texture_away_world(impulse=True)
    value=pose.focal_pose_view(sensors,state,rows[0],rows[3],output_ticks=(0,3),
        selected_head_ids=(0,0),required_part_names=('left_arm','right_arm'))
    changed=deepcopy(value)
    for target in (changed,changed['native_source_measurement']):
        target.update(measurement_available=True,required_motion_available=True,reason=None)
    with pytest.raises(ValueError):fixtures.validate_view(changed,sensors,state,h)


def test_bone_features_are_spatially_sampled_not_only_the_strongest_cluster():
    image=np.zeros((180,420),np.uint8)
    image[75:106,90:145]=np.random.default_rng(52).integers(20,255,(31,55),np.uint8)
    for x in (180,240,300):
        cv2.rectangle(image,(x,84),(x+7,91),50,-1)
        cv2.rectangle(image,(x+5,98),(x+12,105),50,-1)
    before=np.asarray(((100.,90.),(320.,90.)))
    yy,xx=np.indices(image.shape);mask=np.zeros_like(image)
    mask[(xx>=90)&(xx<=330)&(yy>=75)&(yy<=105)]=255
    points=pose._spatial_bone_features(image,mask,before,budget=None)
    assert points is not None and 3<=len(points)<=pose.POLICY['part_roi_maximum_features']
    alpha,_,_=pose._bone_projection(points[:,0],before)
    assert alpha.max()-alpha.min()>=.8
    bins=np.minimum(3,np.maximum(0,np.floor(alpha*4).astype(int)))
    assert set(bins)=={0,1,2,3}
    assert all((bins==n).sum()<=4 for n in range(4))


def test_empty_bone_bins_stay_unknown_and_detector_work_is_charged_before_call():
    from unittest.mock import patch
    image=np.zeros((180,420),np.uint8);mask=np.zeros_like(image)
    mask[75:106,90:330]=255
    calls=[]
    class Budget:
        def spend(self,kind,count=1):calls.append((kind,count))
    with patch.object(cv2,'goodFeaturesToTrack',wraps=cv2.goodFeaturesToTrack) as detect:
        result=pose._spatial_bone_features(image,mask,np.asarray(((100.,90.),(320.,90.))),budget=Budget())
    assert result is None
    assert detect.call_count==4
    assert calls==[('pose_comparisons',1)]*4


@pytest.mark.parametrize('change',['feature','target','coverage','part','required_flag'])
def test_rehashed_part_image_constraints_are_rederived_by_the_consumer(change):
    sensors,state,h,_,_=fixtures.scene_world(scope='exit')
    value=deepcopy(sensors.endpoint(state,outgoing=True)['registered_pose_sequence'])
    check=value['native_source_measurement']['image_consistency'][0]
    part=check['required_part_image_constraints'][0]
    if change=='feature':part['attempts'][0]['source_feature_xy'][0]=0.
    elif change=='target':
        # The regional sensor deliberately keeps isolated outliers instead of
        # treating one background feature as a complete arm contradiction.
        # Change the entire observed support to reproduce a false motion fit.
        for attempt in part['attempts']:
            if attempt['tracked_target_xy'] is not None:attempt['tracked_target_xy'][0]+=100.
    elif change=='coverage':part['longitudinal_coverage']=0.
    elif change=='part':part['raw_joint_indices']=[0,1]
    else:check['required_part_motion_constrained']=False
    with pytest.raises(ValueError):fixtures.validate_view(value,sensors,state,h)
