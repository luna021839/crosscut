import numpy as np
import pytest

from crosscut2.stage_mix import camera_track, semi_markov
from crosscut2.paired_geometry import Camera


def test_dp_selects_entire_programme_not_retained_local_patches():
    nodes = np.arange(0, 121, 10)
    unary = np.zeros((2, 12))
    unary[0, 6:] = 8; unary[1, :6] = 8
    seams = np.ones((13, 2, 2))
    runs, score = semi_markov(nodes, fps=10, unary=unary, seam_cost=seams)
    assert runs[0][0] == 0 and runs[-1][1] == 12
    assert runs[0][2] == 0 and runs[-1][2] == 1
    assert all(a[1] == b[0] for a, b in zip(runs, runs[1:]))
    assert np.isfinite(score)


def test_missing_geometry_remains_finite_and_can_cover_sparse_sources():
    runs, _ = semi_markov(np.arange(0, 61, 10), fps=10,
        unary=np.full((2, 6), 1.2), seam_cost=np.full((7, 2, 2), 4.))
    assert runs[0][0] == 0 and runs[-1][1] == 6


def test_whole_hold_camera_does_not_bounce_back_to_neutral():
    a, b = Camera(1.2, .48, .49), Camera(1.3, .52, .51)
    track = camera_track(a, b, 3.)
    mid = track.evaluate(1.5)
    assert mid.zoom == pytest.approx(1.25)
    assert mid.center_x == pytest.approx(.5)
    assert track.evaluate(0).zoom == a.zoom
    assert track.evaluate(3).zoom == b.zoom


def test_one_source_has_full_coverage_without_artificial_editorial_cuts():
    runs, score = semi_markov(np.arange(0, 201, 10), fps=10,
        unary=np.ones((1, 20)), seam_cost=np.zeros((21, 1, 1)))
    assert runs == ((0, 20, 0),)
    assert score == pytest.approx(20.)
