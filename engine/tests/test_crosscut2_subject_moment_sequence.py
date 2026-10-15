import numpy as np
import pytest
from crosscut2.subject_moment_sequence import choose,hold_curve,curve_digest,camera_motion


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
