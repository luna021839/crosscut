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


def test_authored_closeup_floor_does_not_buy_lower_action_cost_by_shrinking_faces():
    a,b=observation(0,30,size=.035),observation(1,31,size=.045)
    old=coupled_focal_cameras(a,b)
    baseline=focal_extent(project_focal_observation(a,old[0])[0])[1]
    floor=baseline*1.2
    cameras=coupled_focal_cameras(a,b,minimum_extent=floor)
    assert cameras is not None
    sizes=[focal_extent(project_focal_observation(r,c)[0])[1] for r,c in zip((a,b),cameras)]
    assert min(sizes)>=floor-1e-12
    assert abs(sizes[0]-sizes[1])<1e-12


def test_impossible_authored_size_is_not_clamped_to_a_wider_shot():
    a,b=observation(0,30,size=.035),observation(1,31,size=.045)
    assert coupled_focal_cameras(a,b,minimum_extent=1.) is None
    with pytest.raises(ObservedProgrammeUnresolved):
        select(focal_extent_floors={35:1.})


def test_one_layer_floor_is_frozen_for_every_nearby_cut_and_phase():
    floor=.05
    authored={35:floor,58:floor}
    result=select(focal_extent_floors=authored)
    authored[35]=10.
    assert result['authored_focal_extent_floors']=={35:floor,58:floor}
    assert result['focal_floors_applied_before_cost'] is True
    assert result['focal_floor_is_boundary_measurement_not_unobserved_hold_guarantee'] is True
    for seam in result['seams']:
        t=seam['output_tick']
        for row,tick in ((seam['left'],t-1),(seam['right'],t)):
            actual=focal_extent(project_focal_observation(row,result['cameras'][tick]['camera'])[0])[1]
            assert actual>=floor-1e-12
        assert seam['authored_focal_extent_floor']==floor
        assert seam['actual_fixed_inventory_incident']['full_action_certified'] is False


@pytest.mark.parametrize('floors',[
    {36:.05},{True:.05},{35:True},{35:float('nan')},{35:float('inf')},{35:-.01},[(35,.05)],
])
def test_bad_or_undeclared_focal_floor_cannot_change_the_original_selection_domain(floors):
    with pytest.raises(ValueError,match='focal floors'):
        select(focal_extent_floors=floors)


@pytest.mark.parametrize('floor',[True,float('nan'),float('inf'),-.01])
def test_bad_pair_focal_floor_cannot_be_silently_ignored(floor):
    with pytest.raises(ValueError,match='focal extent floor'):
        coupled_focal_cameras(observation(0,30),observation(1,31),minimum_extent=floor)


def test_actual_cached_pose_without_identity_vector_is_not_rejected_or_invented():
    def without_identity(s, t, p, role):
        row = observation(s, t + p, size=.045 if s == 0 else .055)
        del row['sface_vector']
        return row
    original = select()
    result = select(lookup=without_identity)
    assert result['score'] == pytest.approx(original['score'] + .24 * len(result['seams']))
    assert result['new_model_calls'] == result['completion_retries'] == 0
    for seam in result['seams']:
        proof = seam['actual_fixed_inventory_incident']
        assert proof['identity_observation_state'] == 'UNOBSERVED'
        assert proof['identity_descriptor_similarity'] is None
        assert proof['missing_identity_not_imputed_as_match'] is True
        assert proof['identity_cost_interval'] == [0., .24]
        assert seam['named_identity_certified'] is False


def test_vocal_handoff_does_not_require_a_fake_descriptor_to_construct_real_cameras():
    def without_identity(s, t, p, role):
        row = observation(s, t + p)
        row['sface_vector'] = None
        return row
    result = select(lookup=without_identity, roles=('a', 'b', 'a'))
    for seam in result['seams']:
        proof = seam['actual_fixed_inventory_incident']
        assert proof['singing_part_handoff_not_same_member_pose_match'] is True
        assert proof['identity_observation_state'] == 'UNOBSERVED'
        assert proof['identity_cost_applied_to_role_handoff'] is False


def test_missing_one_side_does_not_beat_a_measured_worst_case_identity_pair():
    def missing(s, t, p, role):
        row = observation(s, t + p)
        if s == 1:
            del row['sface_vector']
        return row
    def opposed(s, t, p, role):
        row = observation(s, t + p)
        row['sface_vector'] = [1. if s == 0 else -1., 0.]
        return row
    assert select(lookup=missing)['score'] == pytest.approx(select(lookup=opposed)['score'])


@pytest.mark.parametrize('vector', [[], [float('nan'), 0.], [float('inf'), 0.],
                                   [0., 0.], [2., 0.], [True, False], [[1., 0.]]])
def test_invalid_identity_vector_is_not_a_missing_identity_shortcut(vector):
    def malformed(s, t, p, role):
        row = observation(s, t + p)
        row['sface_vector'] = vector
        return row
    with pytest.raises(ValueError, match='identity descriptor'):
        select(lookup=malformed)


def test_identity_vectors_with_different_dimensions_are_not_compared():
    def incompatible(s, t, p, role):
        row = observation(s, t + p)
        row['sface_vector'] = [1., 0.] if s == 0 else [1., 0., 0.]
        return row
    with pytest.raises(ValueError, match='comparison dimension'):
        select(lookup=incompatible)
