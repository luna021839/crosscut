import numpy as np
import pytest
from fractions import Fraction

from crosscut.render import AspectRatio, RenderPlan, RenderSegment
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from crosscut2.action_geometry import compare_screen_motion
from crosscut2.programme_critic import boundary_union, coverage, programme_schedules, rendered_source_frame,bound_source_cuts


def test_equal_counts_with_different_boundaries_do_not_prove_coverage():
    result = coverage([910, 3075], [], [910, 1111])
    assert result['authored_count'] == result['detected_count'] == 2
    assert result['missed_authored_by_detector'] == [3075]
    assert result['unexplained_detections'] == [1111]
    assert not result['detector_authored_set_exact']
    assert [r['frame'] for r in boundary_union([910, 3075], [], [910, 1111], frame_count=4357)] == [910, 1111, 3075]


def test_exact_overlap_is_one_event_but_three_frame_separation_is_preserved():
    events = boundary_union([40], [40, 100, 103], [40, 103], frame_count=200)
    assert len(events) == 3
    assert events[0]['origins'] == ['authored', 'detected', 'retained_native']
    assert [r['frame'] for r in events] == [40, 100, 103]


@pytest.mark.parametrize('tick', [0, 200, -1, 1.5, True])
def test_invalid_boundary_tick_is_not_silently_rounded(tick):
    with pytest.raises(ValueError, match='integer output tick'):
        boundary_union([tick], [], [], frame_count=200)


def test_retained_cut_uses_the_exact_renderer_schedule(tmp_path):
    plan = RenderPlan((tmp_path / 'a.mp4', tmp_path / 'b.mp4'),
        (RenderSegment(0, 0., 2., 0.), RenderSegment(1, 3., 2., 2.)), 0,
        AspectRatio.LANDSCAPE, tmp_path / 'out.mp4', fps=10.)
    clock = VerifiedRenderFrameClock(Fraction(30), 200, '0' * 64)
    schedules, authored, native = programme_schedules(plan, [clock, clock], [[15], [93, 129]])
    assert authored == (20,)
    assert native == (5, 21, 33)
    assert rendered_source_frame(schedules, 20)['native_frame'] == 90
    assert rendered_source_frame(schedules, 21)['native_frame'] == 93


def test_unverified_source_clock_is_not_mislabelled_exact(tmp_path):
    plan = RenderPlan((tmp_path / 'a.mp4',), (RenderSegment(0, 0., 2., 0.),), 0,
        AspectRatio.LANDSCAPE, tmp_path / 'out.mp4', fps=10.)
    with pytest.raises(ValueError, match='verified native clock'):
        programme_schedules(plan, [None], [[]])


def test_camera_pan_changes_screen_velocity_even_with_identical_source_joints():
    source = np.array([[.3, .3, .9], [.5, .3, .9], [.2, .4, .9], [.6, .4, .9],
        [.15, .45, .9], [.65, .45, .9], [.3, .7, .9], [.5, .7, .9]])
    left = np.stack([source.copy(), source.copy()])
    right = np.stack([source.copy(), source.copy()])
    aligned = compare_screen_motion(left, right, seconds=.2, frame_seconds=1/30)
    left[0, :, 0] -= .03; right[1, :, 0] -= .03
    opposed = compare_screen_motion(left, right, seconds=.2, frame_seconds=1/30)
    assert aligned['position_loss'] == opposed['position_loss'] == 0.
    assert aligned['velocity_loss'] == 0.
    assert opposed['velocity_loss'] == pytest.approx(1.)
    assert opposed['cost'] > aligned['cost']


def test_retained_transnet_cut_is_actually_in_critic_schedule_not_only_editor_metadata(tmp_path):
    identity=(7,8,9,10);base=[dict(fps=30.,frame_count=100,frames=[50])]
    union=[dict(contract='bound_transnet_adaptive_first_incoming_union_v1',source=0,
        source_file_identity=list(identity),fps=30.,frame_count=100,frames=[20,50])]
    actual=bound_source_cuts(base,union,[identity])
    assert actual[0]['frames']==[20,50] and base[0]['frames']==[50]
    plan=RenderPlan((tmp_path/'a.mp4',),(RenderSegment(0,0.,3.,0.),),0,
        AspectRatio.LANDSCAPE,tmp_path/'out.mp4',fps=30.)
    clock=VerifiedRenderFrameClock(Fraction(30),100,'0'*64)
    _,_,native=programme_schedules(plan,[clock],[actual[0]['frames']])
    assert native==(20,50)
    union[0]['source_file_identity']=[7,8,9,11]
    with pytest.raises(ValueError,match='bound to current'):bound_source_cuts(base,union,[identity])
