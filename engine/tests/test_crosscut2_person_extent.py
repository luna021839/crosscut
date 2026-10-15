import math

import numpy as np
import pytest

from crosscut2 import person_extent
from crosscut2.person_extent import LOG_STEP, height_interval, interval_gap, used_face_risk

IDENTITY = np.array([1., .5, .5])


def boxes(*rows):
    return np.array(rows, np.float32).reshape(-1, 5)


def column(heights, top=.05, x0=.1, x1=.3, score=.9):
    return boxes(*[(x0, top, x1, top + h, score) for h in heights])


def test_known_face_risk_is_bitwise_original():
    rng = np.random.default_rng(7)
    face = rng.uniform(0, 5, size=(4, 6))
    gap = rng.uniform(0, 9, size=(4, 6))
    extent_known = rng.random((4, 6)) < .5
    used = used_face_risk(face, np.ones((4, 6), bool), gap, extent_known)
    assert used.dtype == np.float64
    assert np.array_equal(used.view(np.uint64), face.view(np.uint64))
    mixed = rng.random((4, 6)) < .5
    used = used_face_risk(face, mixed, gap, extent_known)
    assert np.array_equal(used[mixed].view(np.uint64), face[mixed].view(np.uint64))
    expected = np.where(extent_known, np.maximum(1., gap), 1.)
    assert np.array_equal(used[~mixed], expected[~mixed])


def test_missing_or_zero_box_extent_keeps_original_unknown_one():
    absent = height_interval(None, IDENTITY)
    empty = height_interval(np.zeros((0, 5), np.float32), IDENTITY)
    present = height_interval(column([.8]), IDENTITY)
    assert not absent[2] and not empty[2] and present[2]
    assert absent[0] == 0. and absent[1] == 0.
    for a, b in ((absent, present), (present, absent), (empty, present), (empty, empty)):
        gap, known = interval_gap(a, b)
        assert not known and gap == 0.
        assert used_face_risk(1., False, gap, known) == 1.
        # Known face stays the face; it never fills the missing extent.
        assert used_face_risk(.37, True, gap, known) == .37


def test_reference_interval_separation():
    left = height_interval(column([.1758, .2638]), IDENTITY)
    right = height_interval(column([.8219, .8359]), IDENTITY)
    assert math.exp(left[0]) == pytest.approx(.1758, abs=1e-6)
    assert math.exp(left[1]) == pytest.approx(.2638, abs=1e-6)
    gap, known = interval_gap(left, right)
    assert known
    assert gap == pytest.approx(math.log(.8219 / .2638) / math.log(1.5), abs=1e-5)
    assert gap == pytest.approx(2.802926, abs=1e-3)
    assert interval_gap(right, left)[0] == gap
    assert used_face_risk(1., False, gap, known) == gap


def test_order_and_duplicates_are_invariant_and_score_is_ignored():
    rows = column([.1758, .2638, .5])
    cameras = np.array([[1., .5, .5], [2., .3, .6], [1.5, .7, .2]])
    base = height_interval(rows, cameras)
    rescored = rows.copy()
    rescored[:, 4] = [.36, 1., .5]
    for variant in (rows[[2, 0, 1]], rows[[0, 0, 1, 2, 2, 1]], rescored):
        for x, y in zip(base, height_interval(variant, cameras)):
            assert np.array_equal(x, y)


def test_overlapping_intervals_give_zero_gap():
    left = height_interval(column([.2, .2, .8]), IDENTITY)
    right = height_interval(column([.2, .8, .8]), IDENTITY)
    gap, known = interval_gap(left, right)
    assert known and gap == 0.
    assert used_face_risk(1., False, gap, known) == 1.


def test_adding_a_box_cannot_increase_separation():
    rng = np.random.default_rng(11)
    for _ in range(200):
        a = rng.uniform(.02, .9, size=rng.integers(1, 4))
        b = rng.uniform(.02, .9, size=rng.integers(1, 4))
        left = height_interval(column(a), IDENTITY)
        right = height_interval(column(b), IDENTITY)
        grown = height_interval(column([*a, rng.uniform(.02, .9)]), IDENTITY)
        assert grown[0] <= left[0] and grown[1] >= left[1]
        assert interval_gap(grown, right)[0] <= interval_gap(left, right)[0]
        assert interval_gap(right, grown)[0] <= interval_gap(right, left)[0]


def test_actual_viewport_zoom_pan_partial_clip_and_fully_out():
    box = boxes((.4, .4, .6, .6, .9))
    cameras = np.array([[1., .5, .5], [2., .5, .5], [2., .5, .3], [4., .1, .1], [2., .5, .8]])
    low, high, measured = height_interval(box, cameras)
    assert measured.tolist() == [True, True, True, False, True]
    expected = np.array([.2, .4, .3, 1., .1])
    np.testing.assert_allclose(np.exp(low[measured]), expected[measured], atol=1e-6)
    assert np.array_equal(low, high)
    assert low[3] == 0. and high[3] == 0.


def test_zero_width_or_zero_height_edge_contact_is_not_visible():
    camera = np.array([2., .5, .5])
    assert not height_interval(boxes((.75, .4, 1., .6, .9)), camera)[2]
    assert not height_interval(boxes((.4, 0., .6, .25, .9)), camera)[2]


def test_batch_native_cameras_match_single_calls():
    rows = column([.1758, .5], x0=.2, x1=.7)
    cameras = np.array([[[1., .5, .5], [3., .2, .9], [8., .9, .1]],
                        [[1.2, .4, .4], [2., .5, .05], [1., .5, .5]]])
    low, high, measured = height_interval(rows, cameras)
    assert low.shape == high.shape == measured.shape == (2, 3)
    for index in np.ndindex(2, 3):
        single = height_interval(rows, cameras[index])
        assert single[0].shape == ()
        np.testing.assert_allclose(low[index], single[0], rtol=1e-12, atol=0)
        np.testing.assert_allclose(high[index], single[1], rtol=1e-12, atol=0)
        assert measured[index] == single[2]


def test_finite_handling_and_validation():
    for zoom in (1e6, 1e-6):
        out = height_interval(boxes((.4, .4, .6, .6, .9)), np.array([zoom, .5, .5]))
        assert out[2] and np.isfinite(out[0]) and np.isfinite(out[1])
    bad_rectangles = [
        boxes((.4, .4, .6, .6, .9)).astype(np.float64),
        np.zeros((5,), np.float32),
        np.zeros((1, 4), np.float32),
        boxes((.4, .4, .6, np.nan, .9)),
        boxes((.4, .4, .6, 1.2, .9)),
        boxes((-.1, .4, .6, .6, .9)),
        boxes((.6, .4, .4, .6, .9)),
        boxes((.4, .6, .6, .6, .9)),
        boxes((.4, .4, .6, .6, 1.5)),
    ]
    for rectangles in bad_rectangles:
        with pytest.raises(ValueError):
            height_interval(rectangles, IDENTITY)
    for camera in ([0., .5, .5], [-1., .5, .5], [np.inf, .5, .5], [1., np.nan, .5], [1., .5]):
        with pytest.raises(ValueError):
            height_interval(None, np.array(camera))
    known = (np.array(0.), np.array(0.), np.array(True))
    for interval in ((np.array(np.nan), np.array(0.), np.array(True)),
                     (np.array(0.), np.array(-1.), np.array(True)),
                     (np.array(0.), np.array(0.), np.array(1)),
                     (np.array(0.), np.array(0.))):
        with pytest.raises(ValueError):
            interval_gap(interval, known)
    for args in ((np.nan, True, 0., True), (1., True, -1., True), (1., 1, 0., True), (1., True, np.inf, True)):
        with pytest.raises(ValueError):
            used_face_risk(*args)


def test_standing_vs_crouching_is_high_visible_extent_gap_not_identity_or_camera_claim():
    standing = height_interval(column([.8]), IDENTITY)
    crouching = height_interval(column([.25], x0=.6, x1=.8), IDENTITY)
    gap, known = interval_gap(standing, crouching)
    assert known
    assert gap == pytest.approx(math.log(.8 / .25) / LOG_STEP, abs=1e-5)
    # Retained limitation: posture change alone reads as a large extent gap.
    assert gap > 2.5
    assert used_face_risk(1., False, gap, known) == gap
    assert isinstance(standing, tuple) and len(standing) == 3
    assert all(isinstance(x, np.ndarray) for x in standing)
    semantics = person_extent.SEMANTICS
    assert semantics['visible_rectangle_extent_only'] is True
    for claim in ('face_size', 'identity', 'anatomy', 'verified_camera_jump', 'same_person',
                  'posture_invariant', 'score_used_for_ranking', 'person_count_or_mask'):
        assert semantics[claim] is False
