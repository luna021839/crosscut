"""The final cut objective must not erase measured action differences."""
import numpy as np
import pytest

from crosscut2.fixed_holds import State, make_states, port_cost
from crosscut2.action_geometry import synchronous_motion_windows, synchronous_velocity


def fixture(wrist_shift):
    heads = np.zeros((4, 2, 8, 4), np.float32)
    heads[:, :, 0] = [.5, .3, .16, .95]
    vectors = np.zeros((4, 2, 3, 128), np.float32)
    vectors[:, :, 0, 0] = 1.
    bank = dict(heads=heads, vectors=vectors,
                native_indices=np.arange(8).reshape(4, 2), zoom_cap=1.35)
    pose = np.array([[.4, .4, .9], [.6, .4, .9], [.3, .5, .9], [.7, .5, .9],
                     [.2, .6, .9], [.8, .6, .9], [.43, .75, .9], [.57, .75, .9]])
    left = np.tile(pose, (4, 4, 3, 1, 1))
    right = left.copy()
    right[:, :, :, 4:6, 1] -= wrist_shift
    states, rows, _ = make_states(np.array([0, 10, 20, 30]), [bank, bank],
                                  [left, right], fps=10)
    return (rows[states.index(State(0, 0, 'native'))],
            rows[states.index(State(1, 0, 'native'))])


@pytest.mark.parametrize('wrist_shift', [.08, .12, .20])
def test_equal_face_match_cannot_tie_identical_and_different_observed_arm_pose(wrist_shift):
    left, right = fixture(wrist_shift)
    exact = port_cost(1, left, left, fps=10)
    different = port_cost(1, left, right, fps=10)
    assert exact['geometry_loss'] == different['geometry_loss'] == 0.
    assert exact['appearance_cosine'] == different['appearance_cosine'] == 1.
    assert exact['conditional_pose_loss'] == 0.
    assert different['conditional_pose_loss'] > 0.
    assert different['soft_cost'] > exact['soft_cost']


def test_one_opposed_limb_or_window_is_not_hidden_by_the_reported_mean():
    left = np.zeros((2, 8, 2)); valid = np.zeros((2, 8), bool)
    left[:, 2:6, 0] = 1.; valid[:, 2:6] = True
    right = left.copy(); right[0, 4, 0] = -1.
    rows = synchronous_motion_windows(left, valid, right, valid)
    mean, moving = synchronous_velocity(left, valid, right, valid)
    assert mean == pytest.approx(.125) and moving == 8
    assert rows[0]['maximum_direction_disagreement'] == 1.
    assert rows[1]['maximum_direction_disagreement'] == 0.
    from crosscut2.fixed_holds import _action_refinement
    from crosscut2.action_geometry import ActionMatch
    screen = dict(cost=0., velocity_loss=0., common_joints=8)
    action = ActionMatch(0., 0., 8, 4, 0.)
    exact = _action_refinement(action, (0., 8), screen, direction_disagreement=0.)
    opposed = _action_refinement(action, (max(r['loss'] for r in rows), moving), screen,
                                 direction_disagreement=1.)
    assert exact < 0. <= opposed


def test_synchronized_choreographic_reversal_is_not_an_opposed_source():
    left = np.zeros((2, 8, 2)); valid = np.zeros((2, 8), bool)
    left[0, 2:6, 0] = 1.; left[1, 2:6, 0] = -1.; valid[:, 2:6] = True
    rows = synchronous_motion_windows(left, valid, left, valid)
    assert all(r['loss'] == r['maximum_direction_disagreement'] == 0. for r in rows)


def test_unobserved_temporal_window_is_not_fabricated_stationary_agreement():
    left = np.ones((2, 8, 2)); valid = np.ones((2, 8), bool); valid[1] = False
    rows = synchronous_motion_windows(left, valid, left, valid)
    assert rows[0]['loss'] == pytest.approx(0.) and rows[0]['moving_joints'] == 8
    assert rows[1]['loss'] is rows[1]['maximum_direction_disagreement'] is None
