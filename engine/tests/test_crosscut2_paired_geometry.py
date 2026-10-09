import numpy as np
import pytest

from crosscut2.paired_geometry import Camera, joint_cameras, match_heads


def heads(rows):
    result = np.zeros((8, 4))
    if rows:
        result[:len(rows)] = rows
    return result


def test_joint_crops_match_size_and_position_in_realized_coordinates():
    a, b, loss = joint_cameras([.40, .36], [.48, .39], .13, .11)
    assert loss < 1e-8
    assert a.zoom * .13 == pytest.approx(b.zoom * .11)
    assert a.project([.40, .36]) == pytest.approx(b.project([.48, .39]))
    for camera in (a, b):
        half = .5 / camera.zoom
        assert half <= camera.x <= 1 - half
        assert half <= camera.y <= 1 - half


def test_infeasible_pair_retains_actual_error_without_padding():
    a, b, loss = joint_cameras([.1, .3], [.9, .8], .12, .035)
    assert loss > 5
    assert max(a.zoom, b.zoom) <= 1.35
    assert a.project([.1, .3])[0] < b.project([.9, .8])[0]


def test_occluding_largest_face_does_not_force_wrong_focal_match():
    a = heads([[.6, .5, .22, .95], [.42, .38, .15, .94]])
    b = heads([[.44, .38, .15, .94]])
    av, bv = np.zeros((3, 128)), np.zeros((3, 128))
    av[0, 0] = 1; av[1, 1] = 1; bv[0, 1] = 1
    match = match_heads(a, b, av, bv)
    assert match.role == 'focal'
    assert match.appearance_cosine == pytest.approx(1.)
    assert match.geometry_loss < .01


def test_formation_is_independent_of_missing_member_appearance():
    a = heads([[.3, .4, .045, .95], [.5, .4, .045, .95], [.7, .4, .045, .95]])
    b = heads([[.34, .44, .04, .95], [.52, .44, .04, .95], [.7, .44, .04, .95]])
    zero = np.zeros((3, 128))
    match = match_heads(a, b, zero, zero)
    assert match.role == 'formation'
    assert match.cost < 2
    assert match.appearance_cosine is None


def test_missing_heads_have_cost_not_veto_or_perfect_identity():
    match = match_heads(heads([]), heads([]), np.zeros((3, 128)), np.zeros((3, 128)))
    assert match.role == 'unobserved' and match.cost > 0
    assert match.left == Camera()


def test_diagonal_crop_respects_existing_renderer_radial_pan_cap():
    a, b, loss = joint_cameras([.01, .01], [.4, .4], .04, .07, caps=(2., 2.))
    for c in (a, b):
        assert np.linalg.norm([c.x - .5, c.y - .5]) <= .25 + 1e-12
    assert loss > 0


def test_background_identity_match_does_not_override_salient_subjects():
    a = heads([[.5, .4, .15, .95], [.55, .45, .07, .94]])
    b = heads([[.5, .4, .15, .95], [.55, .45, .07, .94]])
    av, bv = np.zeros((3, 128)), np.zeros((3, 128))
    av[0, :2] = [1., 0.]; bv[0, :2] = [.5, np.sqrt(.75)]
    av[1, 2] = bv[1, 2] = 1.
    match = match_heads(a, b, av, bv)
    assert match.left_head_index == match.right_head_index == 0


def test_soft_occluder_salience_preserves_the_persistent_focal_head():
    a = heads([[.6, .5, .22, .95], [.42, .38, .15, .94]])
    b = heads([[.44, .38, .15, .94]])
    av, bv = np.zeros((3, 128)), np.zeros((3, 128))
    av[0, 0] = 1; av[1, 1] = 1; bv[0, 1] = 1
    match = match_heads(a, b, av, bv, left_salience=[.3, .7, 0])
    assert match.left_head_index == 1 and match.right_head_index == 0


def test_background_ensemble_fit_does_not_hide_a_foreground_subject_change():
    a = heads([[.5, .4, .16, .95], [.3, .5, .055, .94], [.7, .5, .055, .94]])
    b = heads([[.5, .4, .12, .95], [.3, .5, .06, .94], [.7, .5, .06, .94]])
    av, bv = np.zeros((3, 128)), np.zeros((3, 128))
    av[0, 0] = bv[0, 0] = 1.
    matched = match_heads(a, b, av, bv)
    assert matched.role == 'focal'
    assert matched.left_head_index == matched.right_head_index == 0
