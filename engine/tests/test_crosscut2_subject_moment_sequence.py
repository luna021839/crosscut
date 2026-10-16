import numpy as np
import pytest
from crosscut2.subject_moment_sequence import choose,hold_curve,curve_digest,camera_motion,observed_view_cost
from crosscut2.subject_moment_sequence import convex_connectors
from crosscut2.subject_moment_sequence import camera_path_cost


def moment(frame,cost=0.,*,observed=True):
    return dict(frame=frame,configuration_cost=cost,connection_cost=0.,observed=observed)


def test_joint_choice_does_not_create_too_short_shot_from_independent_best_moments():
    levels=[[moment(0)],[moment(20,.1),moment(24)],[moment(30),moment(34,.1)],[moment(54)]]
    selected,edges,receipt=choose(levels,lambda i,a,b:dict(camera_cost=0.,frames=(a['frame'],b['frame'])))
    chosen=[v[k]['frame'] for v,k in zip(levels,selected)]
    assert chosen==[0,20,34,54]
    assert min(np.diff(chosen))>=12 and len(edges)==3
    assert not receipt['authored_cut_count_changed']


def test_same_music_moment_keeps_distinct_actual_camera_choices_before_whole_hold_cost():
    original=dict(moment(20,.1),left_strip=np.tile([1.,.5,.5],(7,1)),right_strip=np.tile([1.,.5,.5],(7,1)))
    crop=dict(moment(20),left_strip=np.tile([1.8,.5,.5],(7,1)),right_strip=np.tile([1.8,.5,.5],(7,1)))
    levels=[[moment(0)],[original,crop],[moment(50)]]
    selected,_,_=choose(levels,lambda i,a,b:dict(camera_cost=1. if b.get('left_strip',np.ones((1,1)))[0,0]>1.5 else 0.))
    assert selected==[0,0,0]  # smaller seam error cannot force the tight whole hold
    with pytest.raises(ValueError,match='distinct'):
        choose([[moment(0)],[original,dict(original)],[moment(50)]],lambda *a:dict(camera_cost=0.))


def test_unknown_is_a_sole_unchanged_anchor_not_perfect_competitor():
    with pytest.raises(ValueError,match='UNKNOWN'):
        choose([[moment(0)],[moment(20,1.),moment(21,observed=False)],[moment(40)]],lambda *a:None)
    selected,_,_=choose([[moment(0)],[moment(20,observed=False)],[moment(40)]],
                        lambda *a:dict(camera_cost=0.))
    assert selected==[0,0,0]


def test_infeasible_camera_arc_is_not_clamped_or_replaced_after_the_decision():
    selected,_,_=choose([[moment(0)],[moment(20),moment(22,.2)],[moment(44)]],
        lambda i,a,b:None if b['frame']==20 else dict(camera_cost=0.))
    assert selected==[0,1,0]
    with pytest.raises(ValueError,match='no realizable'):
        choose([[moment(0)],[moment(20)]],lambda *a:None)


def test_exact_whole_hold_camera_replay_does_not_modify_the_baseline():
    ticks=np.array([0,79]);base=np.array([[1.2,.5,.5],[1.2,.5,.5]])
    entry=np.tile([1.25,.52,.5],(7,1));exit=np.tile([1.3,.53,.51],(7,1))
    curve=hold_curve(0,80,(ticks,base),entry,exit,cap=1.35)
    assert curve is not None and not curve.values.flags.writeable
    np.testing.assert_array_equal(curve.values[:7],entry)
    np.testing.assert_array_equal(curve.values[-7:],exit)
    replay=hold_curve(0,80,(ticks,base),entry,exit,cap=1.35)
    assert curve_digest(curve.values)==curve_digest(replay.values)
    np.testing.assert_array_equal(base,np.array([[1.2,.5,.5],[1.2,.5,.5]]))


def test_production_sampler_tuple_not_native_cut_is_a_digital_camera_pan():
    from crosscut.render_frame_schedule import sampler_frame_schedule
    native=sampler_frame_schedule(source_start=99/30.,source_playback_rate=1.,native_fps=30.,
        output_fps=30.,frame_count=3,source_frame_count=500).source_frame_indices
    assert isinstance(native,tuple)
    values=np.array([[1.,.5,.5],[2.,.5,.5],[1.5,.5,.5]])
    assert camera_motion(values,native,[100],fps=30.)==pytest.approx(30.*np.log(1.5/2.)**2)
    assert camera_motion(values,native,[],fps=30.)>camera_motion(values,native,[100],fps=30.)


def test_equal_matched_endpoints_need_not_rewind_two_one_two_at38frames():
    entry=np.tile([2.,.5,.5],(7,1));interior=([0,37],[[1.,.5,.5],[1.,.5,.5]])
    old=hold_curve(0,38,interior,entry,entry,cap=2.)
    new=hold_curve(0,38,interior,entry,entry,cap=2.,family='whole_viewport')
    assert old.values[:,0].min()<1.1
    np.testing.assert_allclose(new.values,np.tile([2.,.5,.5],(38,1)),atol=1e-15)
    assert camera_motion(new.values,np.arange(38),[],fps=30.)<1e-25
    assert camera_motion(old.values,np.arange(38),[],fps=30.)>.1


def test_inverse_viewport_bridge_preserves_strips_padding_and_input_arrays():
    from crosscut2.boundary_camera_arc import realizable
    entry=np.tile([1.1,.5,.5],(7,1));exit=np.tile([2.,.7,.5],(7,1))
    old_entry=entry.copy();old_exit=exit.copy()
    curve=hold_curve(0,80,([0,79],[[1.,.5,.5]]*2),entry,exit,cap=2.,family='whole_viewport')
    assert realizable(curve.values,2.) and not curve.values.flags.writeable
    np.testing.assert_array_equal(curve.values[:7],entry);np.testing.assert_array_equal(curve.values[-7:],exit)
    assert np.diff(curve.values[:,0]).min()>=-1e-14
    np.testing.assert_array_equal(entry,old_entry);np.testing.assert_array_equal(exit,old_exit)
    assert curve_digest(curve.values)==curve_digest(hold_curve(0,80,([0,79],[[1.,.5,.5]]*2),
        entry,exit,cap=2.,family='whole_viewport').values)


def test_nonzero_endpoint_velocity_cannot_be_clamped_into_a_valid_viewport():
    entry=np.tile([2.,.5,.5],(7,1));entry[-1,1]=.75
    exit=np.tile([2.,.75,.5],(7,1))
    assert hold_curve(0,80,([0,79],[[1.,.5,.5]]*2),entry,exit,cap=2.,family='whole_viewport') is None


def test_actual_played_crop_cost_penalizes_target_loss_and_does_not_double_count_windows():
    from dataclasses import replace
    from test_crosscut2_subject_cut_moments import window
    w=window(0);head=w.heads.copy();head[:,:2]=.5;head[:,2]=.1
    w=replace(w,heads=head)
    native=np.arange(86,115);good=np.tile([2.,.5,.5],(29,1));bad=np.tile([2.,.75,.5],(29,1))
    a=observed_view_cost(good,native,[w],fps=30.);b=observed_view_cost(bad,native,[w],fps=30.)
    assert b['observed_crop_cost']>a['observed_crop_cost']
    assert a['observed_output_frames']==29 and not a['full_hold_subject_visibility_certified']
    assert a==observed_view_cost(good,native,[w,w],fps=30.)
    unknown=observed_view_cost(good,native,[],fps=30.)
    assert unknown['observed_output_frames']==0 and unknown['unobserved_output_frames']==29


def test_camera_rounding_tolerance_does_not_create_negative_content_cost():
    from test_crosscut2_subject_cut_moments import window
    w=window(0);camera=np.tile([1.-2e-16,.5,.5],(29,1));saved=camera.copy()
    value=observed_view_cost(camera,np.arange(86,115),[w],fps=30.)
    assert value['observed_crop_cost']>=0 and value['digital_zoom_cost']>=0
    np.testing.assert_array_equal(camera,saved)


def test_all_local_families_pay_whole_hold_view_cost_before_camera_selection():
    """Pro F253: the actual90-frame legal-family winner changes with the cost.

    Synthetic observed body/face support; no real-film benefit is asserted.
    Offering whole-shot camera families is not needed to pay this cost.
    """
    from types import SimpleNamespace
    length=90;native=np.arange(length)
    body=np.array([[.4,.4,.9],[.6,.4,.9],[.35,.5,.9],[.65,.5,.9],
                   [.15,.55,.9],[.85,.55,.9],[.43,.7,.9],[.57,.7,.9]])
    window=SimpleNamespace(ordinals=native,associated=np.ones(length,bool),
        heads=np.tile([.5,.3,.2,.99],(length,1)),body=np.tile(body,(length,1,1)),
        support=np.ones((length,8),bool),competitors=tuple(np.zeros((0,4)) for _ in native))
    base=np.array([[1.3407834406393584,.5039068465781759,.5028197199065984],
        [1.0856397414190302,.504322873714037,.5028058293945431],
        [1.4431110024195715,.5198280180493401,.49381993969956994],
        [1.0893981294376767,.5030689863798059,.4824388753363989]])
    endpoints=np.array([[1.6373467719602766,.5586491108123623,.48374104238864907],
                        [1.252752488334927,.5462584245729089,.5296537252702411]])
    costs={};curves={}
    for family in ('connectors','convex_connectors'):
        curve=hold_curve(0,length,([0,30,60,89],base),np.tile(endpoints[0],(7,1)),
            np.tile(endpoints[1],(7,1)),cap=1.8,family=family)
        assert curve is not None
        curves[family]=curve
        costs[family]=camera_path_cost(curve.values,native,[],[window],fps=30.)
        assert costs[family]['camera_cost']==pytest.approx(sum(costs[family][k]
            for k in ('digital_motion_cost','observed_crop_cost','digital_zoom_cost')))
        assert costs[family]['observed_output_frames']==length
        assert costs[family]['unobserved_output_frames']==0
        assert not costs[family]['full_hold_subject_visibility_certified']
    assert min(costs,key=lambda k:costs[k]['digital_motion_cost'])=='connectors'
    assert min(costs,key=lambda k:costs[k]['camera_cost'])=='convex_connectors'
    # An unrelated offer cannot alter either existing path's actual score.
    before={k:dict(v) for k,v in costs.items()}
    extra=hold_curve(0,length,([0,30,60,89],base),np.tile(endpoints[0],(7,1)),
        np.tile(endpoints[1],(7,1)),cap=1.8,family='whole_viewport')
    assert extra is not None
    camera_path_cost(extra.values,native,[],[window],fps=30.)
    for family,curve in curves.items():
        assert camera_path_cost(curve.values,native,[],[window],fps=30.)==before[family]


def test_unobserved_path_cost_keeps_unknown_coverage_and_known_digital_zoom():
    values=np.tile([1.35,.5,.5],(90,1))
    value=camera_path_cost(values,np.arange(90),[],[],fps=30.)
    assert value['observed_output_frames']==0 and value['unobserved_output_frames']==90
    assert value['observed_crop_cost']==0 and value['digital_zoom_cost']>0
    assert value['camera_cost']==value['digital_zoom_cost']
    assert not value['full_hold_subject_visibility_certified']


def test_tiny_pose_improvement_cannot_outweigh_arbitrarily_bad_camera_cost():
    levels=[[moment(0)],[moment(20),moment(21,1e-12)],[moment(50)]]
    def hold(i,a,b):return dict(camera_cost=2_000_000. if b['frame']==20 else 0.)
    selected,_,receipt=choose(levels,hold)
    assert selected==[0,1,0]
    assert not receipt['lexicographic_infinite_pose_priority']
    assert receipt['objective']==pytest.approx(1e-12)


def test_open_family_can_leave_breathing_room_without_a_twelve_frame_snapback():
    entry=np.tile([1.35,.56,.5],(7,1));exit=entry.copy();base=([0,89],[[1.,.5,.5]]*2)
    curve=hold_curve(0,90,base,entry,exit,cap=1.35,family='whole_open')
    assert curve is not None and curve.kind=='whole_open'
    assert curve.values[18,0]>1.2  # old family has already snapped to1x here
    assert curve.values[44,0]==1.
    np.testing.assert_array_equal(curve.values[:7],entry);np.testing.assert_array_equal(curve.values[-7:],exit)
    assert camera_motion(curve.values,np.arange(90),[],fps=30.)<camera_motion(
        hold_curve(0,90,base,entry,exit,cap=1.35).values,np.arange(90),[],fps=30.)
def test_convex_local_connectors_handle_legal_endpoints_that_additive_offsets_cannot_join():
    from crosscut2.boundary_camera_arc import build,realizable
    base=np.tile([1.,.5,.5],(80,1));base[:7,0]=1.2
    entry=exit=np.tile([1.,.5,.5],(7,1))
    assert build(entry,exit,base,cap=2.) is None
    value=convex_connectors(base,entry,exit,cap=2.)
    assert value is not None and realizable(value.values,2.)
    np.testing.assert_array_equal(value.values[:7],entry)
    np.testing.assert_array_equal(value.values[-7:],exit)
    np.testing.assert_array_equal(value.values[18:-18],base[18:-18])


@pytest.mark.parametrize('length',[14,15,20,37,38,39,80])
def test_convex_local_connectors_preserve_every_boundary_sample_and_long_baseline_middle(length):
    from crosscut2.boundary_camera_arc import realizable
    base=np.tile([1.35,.52,.5],(length,1))
    entry=np.tile([1.,.5,.5],(7,1));exit=np.tile([2.,.6,.55],(7,1))
    value=convex_connectors(base,entry,exit,cap=2.)
    assert value is not None and realizable(value.values,2.)
    np.testing.assert_array_equal(value.values[:7],entry)
    np.testing.assert_array_equal(value.values[-7:],exit)
    assert not value.values.flags.writeable
    if length>=38:np.testing.assert_array_equal(value.values[18:-18],base[18:-18])
    replay=hold_curve(0,length,(np.arange(length),base),entry,exit,cap=2.,family='convex_connectors')
    np.testing.assert_array_equal(replay.values,value.values)


def test_convex_family_does_not_invent_a_moving_strip_extension_or_clamp_illegal_geometry():
    base=np.tile([1.,.5,.5],(80,1));entry=np.tile([1.5,.5,.5],(7,1));exit=entry.copy()
    entry[-1,1]=.51
    assert convex_connectors(base,entry,exit,cap=2.) is None
    entry[:]=[1.,.4,.5]
    assert convex_connectors(base,entry,exit,cap=2.) is None
