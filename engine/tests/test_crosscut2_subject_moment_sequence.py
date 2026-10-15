import numpy as np
import pytest
from crosscut2.subject_moment_sequence import choose,hold_curve,curve_digest,camera_motion,observed_view_cost


def moment(frame,cost=0.,*,observed=True):
    return dict(frame=frame,configuration_cost=cost,connection_cost=0.,observed=observed)


def test_joint_choice_does_not_create_too_short_shot_from_independent_best_moments():
    levels=[[moment(0)],[moment(20,.1),moment(24)],[moment(30),moment(34,.1)],[moment(54)]]
    selected,edges,receipt=choose(levels,lambda i,a,b:dict(camera_cost=0.,frames=(a['frame'],b['frame'])))
    chosen=[v[k]['frame'] for v,k in zip(levels,selected)]
    assert chosen==[0,20,34,54]
    assert min(np.diff(chosen))>=12 and len(edges)==3
    assert not receipt['authored_cut_count_changed']


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
