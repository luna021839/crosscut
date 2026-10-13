import numpy as np
import pytest

from crosscut2.action_geometry import action_crop_cost, compare_action, framing_cost
from crosscut2.paired_geometry import Camera


def pose():
    return np.asarray([[.4, .4, .9], [.6, .4, .9], [.3, .5, .9], [.7, .5, .9],
        [.2, .6, .9], [.8, .6, .9], [.43, .75, .9], [.57, .75, .9]])


def test_identical_stationary_action_is_observed_not_a_missing_velocity():
    window = np.stack((pose(), pose()))
    match = compare_action(window, window, seconds=.2)
    assert match.phase_loss == pytest.approx(0)
    assert match.velocity_loss == pytest.approx(0)
    assert match.common_joints == 8


def test_same_endpoint_pose_but_opposite_hand_motion_has_a_real_cost():
    endpoint = pose()
    prior = endpoint.copy(); prior[4:6, 1] -= .08
    forward = endpoint.copy(); forward[4:6, 1] += .08
    good = compare_action(np.stack((prior, endpoint)), np.stack((endpoint, forward)), seconds=.2)
    bad = compare_action(np.stack((prior, endpoint)), np.stack((endpoint, prior)), seconds=.2)
    assert good.phase_loss == bad.phase_loss == pytest.approx(0)
    assert good.cost < bad.cost
    assert bad.velocity_loss > .9


def test_same_face_geometry_does_not_hide_a_different_arm_pose():
    a, b = pose(), pose(); b[4:6, 1] = .2
    match = compare_action(np.stack((a, a)), np.stack((b, b)), seconds=.2)
    assert match.phase_loss == pytest.approx(.5) and match.cost == pytest.approx(1.5)


def test_missing_pose_or_missing_temporal_flank_is_not_stationary():
    window = np.stack((pose(), pose()))
    missing = compare_action(np.zeros_like(window), window, seconds=.2)
    assert missing.phase_loss is None and missing.velocity_loss is None and np.isfinite(missing.cost)
    partial = window.copy(); partial[0] = 0
    match = compare_action(partial, window, seconds=.2)
    assert match.phase_loss == pytest.approx(0) and match.velocity_loss is None and match.cost > 0


def test_pose_phase_is_translation_and_scale_invariant_but_not_mirrored():
    a, b = pose(), pose(); b[:, :2] = .7 * a[:, :2] + .1
    match = compare_action(np.stack((a, a)), np.stack((b, b)), seconds=.2)
    assert match.phase_loss < 1e-12


def test_added_zoom_clips_visible_wrists_but_native_missing_wrists_are_not_invented():
    support = pose()
    costs = action_crop_cost([1., 2.], [[.5, .5], [.5, .5]], support)
    assert costs[0] == 0 and costs[1] > 0
    support[4:6, 2] = 0
    assert action_crop_cost([2.], [[.5, .5]], support)[0] < costs[1]


def test_tighter_crop_has_no_automatic_reward_and_native_closeup_is_not_zoomed_out():
    assert framing_cost(Camera(), Camera(), .3, .3) == 0
    assert framing_cost(Camera(1.35), Camera(1.35), .2, .2) > .5


def test_invalid_motion_clock_is_rejected():
    window = np.stack((pose(), pose()))
    with pytest.raises(ValueError): compare_action(window, window, seconds=0)


def test_one_complete_arm_can_compare_phase_without_the_other_shoulder():
    a = pose(); a[[1, 3, 5, 6, 7], 2] = 0
    b = a.copy(); b[4, :2] = [.3, .2]
    good = compare_action(np.stack((a, a)), np.stack((a, a)), seconds=.2)
    bad = compare_action(np.stack((a, a)), np.stack((b, b)), seconds=.2)
    assert good.measurement_basis == 'directed_complete_arm_segments'
    assert good.common_joints == 3 and good.phase_loss == good.velocity_loss == 0
    assert bad.phase_loss > .3 and bad.cost > good.cost


def test_partial_arm_rotation_and_motion_direction_are_not_normalized_away():
    endpoint = pose(); endpoint[[1, 3, 5, 6, 7], 2] = 0
    prior = endpoint.copy(); prior[4, 0] -= .06
    later = endpoint.copy(); later[4, 0] += .06
    good = compare_action(np.stack((prior, endpoint)), np.stack((endpoint, later)), seconds=.2)
    bad = compare_action(np.stack((prior, endpoint)), np.stack((endpoint, prior)), seconds=.2)
    assert good.cost < bad.cost and bad.velocity_loss > .9
    translated = endpoint.copy(); translated[:, :2] = .7 * endpoint[:, :2] + .1
    assert compare_action(np.stack((endpoint, endpoint)), np.stack((translated, translated)), seconds=.2).cost < 1e-12


def test_a_missing_wrist_or_zero_length_bone_is_not_a_complete_arm():
    a = pose(); a[[1, 3, 5, 6, 7], 2] = 0
    b = a.copy(); b[4, 2] = 0
    assert compare_action(np.stack((a, a)), np.stack((b, b)), seconds=.2).phase_loss is None
    b = a.copy(); b[4, :2] = b[2, :2]
    assert compare_action(np.stack((a, a)), np.stack((b, b)), seconds=.2).phase_loss is None
