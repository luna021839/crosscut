"""Geometry and actual affine-raster tests, not trained/real-stage quality proof."""
from copy import deepcopy

import cv2
import numpy as np
import pytest

from crosscut.group_review_observed_programme import coupled_focal_cameras,focal_extent
from crosscut.group_review_roll_focal_camera import (
    coupled_roll_focal_cameras,covered_roll_focal_camera,focal_eye_roll_degrees,
    observed_independent_head_reference,project_roll_focal_observation,
    steady_native_camera_sequence,
)
from crosscut.render_geometry import reproject_editorial_frame


def row(x=.5,size=.05):
    return dict(face=dict(box_xywh=[x-size,.3,2*size,3*size],landmarks_xy=[
        [x-size,.34],[x+size,.34],[x,.36],[x-size*.7,.39],[x+size*.7,.39]]),body=None)


def test_zero_roll_keeps_existing_camera_contract_exactly():
    a,b=row(.4,.045),row(.6,.06)
    assert coupled_roll_focal_cameras(a,b,rolls=(0.,0.))==coupled_focal_cameras(a,b,maximum_zoom=2.4)


@pytest.mark.parametrize('rolls',[(6.,-6.),(-3.,6.),(0.,6.)])
def test_two_rotated_native_cameras_share_actual_center_extent_and_edges(rolls):
    a,b=row(.38,.05),row(.61,.06)
    cameras=coupled_roll_focal_cameras(a,b,rolls=rolls)
    assert cameras is not None
    projected=[project_roll_focal_observation(r,c)[0] for r,c in zip((a,b),cameras)]
    ac,sa=focal_extent(projected[0]);bc,sb=focal_extent(projected[1])
    assert np.linalg.norm(ac-bc)<1e-10
    assert abs(sa-sb)<1e-10
    assert all(covered_roll_focal_camera(c) for c in cameras)


def test_projected_eye_angle_uses_real_pixel_aspect_and_clockwise_renderer_sign():
    value=row();camera=[1.4,.5,.5,6.]
    projected=project_roll_focal_observation(value,camera)[0]
    assert focal_eye_roll_degrees(projected)==pytest.approx(6.)
    assert value['face']['landmarks_xy'][0][1]==.34


def test_unavailable_body_is_not_invented_by_rotation():
    value=row();_,points=project_roll_focal_observation(value,[1.4,.5,.5,6.])
    assert points=={}


def test_feasible_raster_has_no_black_or_reflected_edge_fill():
    cameras=coupled_roll_focal_cameras(row(.38,.05),row(.61,.06),rolls=(6.,-6.))
    actual=np.full((1080,1920,3),117,np.uint8)
    for z,x,y,roll in cameras:
        pixels=reproject_editorial_frame(actual,size=(1920,1080),zoom=z,
            center_x=x,center_y=y,roll_degrees=roll,maximum_zoom=1.)
        assert pixels.min()==pixels.max()==117


def test_declines_no_shared_extent_and_rejects_uncovered_tilt():
    assert coupled_roll_focal_cameras(row(.5,.02),row(.5,.12),rolls=(6.,-6.)) is None
    assert not covered_roll_focal_camera([1.,.5,.5,6.])
    assert not covered_roll_focal_camera([1.2,.01,.5,6.])


@pytest.mark.parametrize('kwargs',[
    {'rolls':(7.,0.)},{'rolls':(float('nan'),0.)},{'rolls':(True,0.)},
    {'rolls':(6.,-6.),'maximum_zoom':3.},{'rolls':(6.,-6.),'maximum_roll_degrees':7.},
])
def test_capacity_and_numbers_are_not_silently_relaxed(kwargs):
    with pytest.raises(ValueError,match='declared'):
        coupled_roll_focal_cameras(row(),row(),**kwargs)


@pytest.mark.parametrize('rolls',[(0.,0.),(6.,-6.)])
def test_authored_shot_extent_is_not_reduced_to_improve_rotated_joint_cost(rolls):
    a,b=row(.4,.05),row(.58,.055)
    original=deepcopy((a,b))
    baseline=coupled_roll_focal_cameras(a,b,rolls=rolls)
    extent=focal_extent(project_roll_focal_observation(a,baseline[0])[0])[1]*1.15
    cameras=coupled_roll_focal_cameras(a,b,rolls=rolls,minimum_extent=extent)
    assert cameras is not None
    for observation,camera in zip((a,b),cameras):
        assert focal_extent(project_roll_focal_observation(observation,camera)[0])[1]>=extent-1e-9
        assert covered_roll_focal_camera(camera)
    assert (a,b)==original


@pytest.mark.parametrize('rolls',[(0.,0.),(6.,-6.)])
def test_impossible_authored_extent_is_unknown_not_clamped_or_relaxed(rolls):
    assert coupled_roll_focal_cameras(row(),row(),rolls=rolls,minimum_extent=.9) is None


@pytest.mark.parametrize('extent',[-1.,float('nan'),float('inf'),True])
def test_invalid_authored_extent_is_not_a_renderable_candidate(extent):
    with pytest.raises(ValueError,match='authored'):
        coupled_roll_focal_cameras(row(),row(),rolls=(6.,-6.),minimum_extent=extent)


def independently_observed_row():
    value=row()
    head=[[.50,.39,.8],[.53,.35,.9],[.47,.38,.9],[.57,.34,.7],[.44,.41,.7]]
    value['body']=dict(reason='observed',face_anchor_consistent=True,
        raw_keypoints_xyc=head+[[.5,.5,.7]]*12)
    return value


def test_real_head_basis_remains_distinct_from_detector_mouth_basis():
    value=independently_observed_row()
    original=deepcopy(value)
    reference=observed_independent_head_reference(value)
    assert reference['actual_head_reference_coco_ids']==[2,1,0,4,3]
    assert reference['face']['landmarks_xy']==[value['body']['raw_keypoints_xyc'][i][:2] for i in [2,1,0,4,3]]
    assert reference['original_face_detector_observation']==original['face']
    assert 'not_yunet_mouth5' in reference['face']['landmark_basis']
    assert reference['head_points_recomputed'] is False
    assert focal_eye_roll_degrees(reference['face'])<0
    assert value==original


@pytest.mark.parametrize('missing',[None,'no_raw','weak_eye','wrong_owner','outside','crossed_eyes'])
def test_unobserved_independent_head_is_not_filled_or_called_a_match(missing):
    value=independently_observed_row()
    if missing is None:value['body']=None
    elif missing=='no_raw':value['body']['raw_keypoints_xyc']=[]
    elif missing=='weak_eye':value['body']['raw_keypoints_xyc'][1][2]=.1
    elif missing=='wrong_owner':value['body']['face_anchor_consistent']=False
    elif missing=='outside':value['body']['raw_keypoints_xyc'][4][0]=-1.
    elif missing=='crossed_eyes':value['body']['raw_keypoints_xyc'][1][0]=.4
    assert observed_independent_head_reference(value) is None


def test_nonfinite_raw_head_and_invalid_confidence_floor_do_not_enter_geometry():
    value=independently_observed_row()
    value['body']['raw_keypoints_xyc'][0][0]=float('nan')
    with pytest.raises(ValueError,match='raw17'):
        observed_independent_head_reference(value)
    with pytest.raises(ValueError,match='confidence'):
        observed_independent_head_reference(row(),minimum_confidence=True)


def test_steady_control_removes_hunting_and_keeps_matched_incident_cameras():
    original=[[1.5,.46,.5,0.]]+[[1.5,.5+(-1)**i*.04,.5,0.] for i in range(27)]+[[1.7,.54,.5,3.]]
    before=deepcopy(original)
    result=steady_native_camera_sequence(original,native_scene_ids=[(0,2)]*len(original))
    assert result is not None and original==before
    assert result[0]==original[0] and result[-1]==original[-1]
    assert np.diff(np.asarray(result)[:,1]).min()>=0
    assert np.diff(np.asarray(result)[:,3]).min()>=0
    assert all(covered_roll_focal_camera(c) for c in result)


def test_steady_control_keeps_native_camera_cut_flanks_without_bridging_them():
    cameras=[[1.6,.46,.5,0.],[1.6,.51,.5,0.],[1.6,.47,.5,0.],
        [1.6,.56,.5,0.],[1.6,.50,.5,0.],[1.6,.55,.5,0.]]
    result=steady_native_camera_sequence(cameras,native_scene_ids=[(1,2)]*3+[(1,3)]*3)
    assert result is not None
    for i in (0,2,3,5):assert result[i]==cameras[i]
    assert result[1][1]==pytest.approx(.465)
    assert result[4][1]==pytest.approx(.555)


def test_steady_control_preserves_single_and_two_tick_native_spans_exactly():
    cameras=[[1.4,.5,.5,0.],[1.5,.51,.5,0.],[1.6,.52,.5,0.]]
    assert steady_native_camera_sequence(cameras,native_scene_ids=[(0,1),(0,2),(0,2)])==cameras


@pytest.mark.parametrize('cameras,scenes',[
    ([[1.5,.5,.5,0.]],[]),([[float('nan'),.5,.5,0.]],[(0,1)]),
    ([[3.,.5,.5,0.]],[(0,1)]),([[1.,.01,.5,0.]],[(0,1)]),
])
def test_steady_control_does_not_relax_capacity_or_fabricate_native_scene_ownership(cameras,scenes):
    with pytest.raises(ValueError):
        steady_native_camera_sequence(cameras,native_scene_ids=scenes)


def test_steady_slow_path_retains_broad_movement_without_framewise_hunting():
    ticks=np.linspace(0.,1.,49)
    cameras=[[1.6,.5,.5+.06*np.sin(np.pi*t)+(.012*(-1)**i if 0<i<48 else 0.),0.]
        for i,t in enumerate(ticks)]
    result=steady_native_camera_sequence(cameras,native_scene_ids=[(0,1)]*49,retain_slow_path=True)
    assert result is not None
    assert result[0]==cameras[0] and result[-1]==cameras[-1]
    before=np.asarray(cameras)[:,2];after=np.asarray(result)[:,2]
    assert after.max()>.55
    assert np.abs(np.diff(after)).max()<np.abs(np.diff(before)).max()/4
    assert all(covered_roll_focal_camera(c) for c in result)


def test_steady_slow_path_never_filters_across_native_camera_cuts():
    cameras=[[1.6,.45,.5,0.],[1.6,.48,.5,0.],[1.6,.46,.5,0.],
        [1.6,.55,.5,0.],[1.6,.58,.5,0.],[1.6,.56,.5,0.]]
    result=steady_native_camera_sequence(cameras,native_scene_ids=[(0,1)]*3+[(0,2)]*3,
        retain_slow_path=True)
    assert result is not None
    for i in (0,2,3,5):assert result[i]==cameras[i]


def test_steady_path_policy_is_not_an_implicit_truthy_quality_flag():
    with pytest.raises(ValueError,match='explicit boolean'):
        steady_native_camera_sequence([[1.5,.5,.5,0.]],native_scene_ids=[(0,1)],retain_slow_path=1)
