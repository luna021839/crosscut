"""Numeric construction only: no NN, real-pixel or editorial-quality claim."""
from copy import deepcopy

import numpy as np
import pytest

from crosscut.group_review_observed_programme import (
    ObservedProgrammeBudget, ObservedProgrammeUnresolved, coupled_focal_cameras, focal_extent,
    project_focal_observation, select_observed_focal_programme,
)


def observation(source, tick, *, size=.05, x=.5):
    face = dict(box_xywh=[x-size, .3, size*2, size*2.5], landmarks_xy=[
        [x-size, .34], [x+size, .34], [x, .36], [x-size*.7, .39], [x+size*.7, .39]])
    return dict(source_index=source, native_ordinal=tick, native_pixel_sha256='a'*64,
        face=face, body=None, sface_vector=[1., 0.])


def select(**kwargs):
    params = dict(first=10, end=80, layer_sources=((0,), (1,), (0,)),
        cut_centers=(35, 58), roles=('role',)*3, cut_radius=2, phases=(0,),
        lookup=lambda s,t,p,r:observation(s,t+p,size=.045 if s==0 else .055),
        native_at=lambda s,t:t, source_native_cuts={0:(), 1:()},
        pose_model_identity='b'*64, fps=30.)
    return select_observed_focal_programme(**(params | kwargs))


def test_face_extent_and_position_are_matched_on_both_actual_cameras():
    a, b = observation(0, 30, size=.05, x=.4), observation(1, 31, size=.08, x=.6)
    cameras = coupled_focal_cameras(a,b)
    assert cameras is not None
    pa, pb = [project_focal_observation(row,camera)[0] for row,camera in zip((a,b),cameras)]
    ca, sa = focal_extent(pa)
    cb, sb = focal_extent(pb)
    assert np.linalg.norm(ca-cb) < 1e-12
    assert abs(sa-sb) < 1e-12


def test_full_chosen_native_camera_grid_survives_without_postselection_refit():
    result = select()
    assert set(result['cameras']) == set(range(10,80))
    assert result['new_model_calls'] == result['completion_retries'] == 0
    assert result['camera_refit_after_selection'] is False
    assert result['full_hold_action_certified'] is False
    for c in result['cameras'].values():
        z,x,y,roll = c['camera']
        assert 1 <= z <= 2 and roll == 0
        assert x-.5/z >= -1e-9 and x+.5/z <= 1+1e-9
        assert y-.5/z >= -1e-9 and y+.5/z <= 1+1e-9


def test_retained_exterior_cameras_are_frozen_before_constructing_the_full_grid():
    cameras = {'first':[1.4,.42,.45,0.], 'last':[1.2,.52,.48,0.]}
    result = select(boundary_cameras=cameras)
    assert result['cameras'][10]['camera'] == cameras['first']
    assert result['cameras'][79]['camera'] == cameras['last']
    assert result['camera_refit_after_selection'] is False
    assert result['entry_return_pose_certified'] is False
    cameras['first'][0] = 2.
    assert result['retained_boundary_cameras']['first'][0] == 1.4
    assert result['clips'][0]['start_camera'][0] == 1.4


@pytest.mark.parametrize('cameras',[
    {'middle':[1.,.5,.5,0.]}, {'first':[1.,.1,.5,0.]},
    {'last':[2.1,.5,.5,0.]}, {'first':[1.,.5,.5,float('nan')]},
    {'last':[1.,.5,.5,True]},
])
def test_retained_boundary_cameras_cannot_hide_an_invalid_crop_or_new_zoom_capacity(cameras):
    with pytest.raises(ValueError, match='retained boundary'):
        select(boundary_cameras=cameras)


def test_missing_wrists_and_body_keep_fixed_upper_cost_not_a_match():
    result = select()
    for seam in result['seams']:
        proof = seam['actual_fixed_inventory_incident']
        assert proof['fixed_requested_joint_indices'] == list(range(6))
        assert proof['missing_terms_are_measured_zero'] is False
        assert all(r['state']=='UNOBSERVED' for r in proof['joint_requests'])
        assert proof['exploration_score'] >= .6-1e-12
        assert proof['full_action_certified'] is False


def test_native_broadcast_cut_cannot_be_hidden_inside_a_chosen_clip():
    with pytest.raises(ValueError, match='no known legal path'):
        select(source_native_cuts={0:(20,),1:()})


def test_source_or_native_parent_tamper_is_not_a_nearby_fallback():
    def corrupt(s,t,p,r):
        value = observation(s,t+p)
        value['native_ordinal'] += 1
        return value
    with pytest.raises(ValueError, match='native parent'):
        select(lookup=corrupt)


def test_original_budget_is_not_renewed_after_unknown_or_failed_search():
    with pytest.raises(TimeoutError, match='no retry'):
        select(budget=ObservedProgrammeBudget(maximum_incidents=1))


def test_unknown_focal_actor_does_not_become_a_good_geometry_score():
    with pytest.raises(ObservedProgrammeUnresolved, match='no known legal path') as caught:
        select(lookup=lambda *args:None)
    assert caught.value.report['observed_geometry_rejected'] is False
    assert caught.value.report['rendered_movie'] is False
    assert caught.value.report['new_model_calls'] == 0


def test_unrealizable_size_ratio_and_bad_camera_numbers_stay_declined():
    assert coupled_focal_cameras(observation(0,10,size=.02), observation(1,11,size=.12)) is None
    with pytest.raises(ValueError, match='landmarks'):
        row = deepcopy(observation(0,10))
        row['face']['landmarks_xy'][0][0] = float('nan')
        coupled_focal_cameras(row,observation(1,11))


@pytest.mark.parametrize('budget',[
    {'maximum_seconds':float('inf')}, {'beam_width':True}, {'maximum_zoom':3.},
    {'minimum_shot_ticks':1}, {'maximum_incidents':20001},
])
def test_invalid_resource_domain_is_not_implicitly_relaxed(budget):
    with pytest.raises(ValueError, match='budget'):
        ObservedProgrammeBudget(**budget)
