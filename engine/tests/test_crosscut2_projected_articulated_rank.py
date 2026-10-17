from dataclasses import replace

import numpy as np
import pytest

from crosscut2.dense_subject_catalogue import DenseSubjects
from crosscut2.played_interval_utility import articulated_seam_evidence, projected_articulated_pose
from crosscut2.played_partial_pose import features, matrix
from test_crosscut2_dense_subject_catalogue import prepared


def observations(*, missing=False):
    a, _, _ = prepared(0, bump=.08)
    b, window, clock = prepared(1, bump=.08, missing=missing)
    # The same observed choreography filmed half as large and off centre.
    body = window.body.copy()
    body[..., :2] = .5*(body[..., :2]-.5)+[.6, .45]
    heads = window.heads.copy()
    heads[:, :2] = .5*(heads[:, :2]-.5)+[.6, .45]
    heads[:, 2] *= .5
    b.bank['heads'][:, :, 0] = heads[0]
    b.subjects = DenseSubjects(source=1, binding='1', indices=clock,
        windows=[replace(window, heads=heads, body=body)], bank=b.bank, fps=30.)
    return (a.observation(1, 0, np.tile([1., .5, .5], (7, 1)), target_head_index=0),
            b.observation(1, 1, np.tile([1., .5, .5], (7, 1)), target_head_index=0))


def detail(left, right, **kw):
    raw = matrix(features(left, fps=30.), features(right, fps=30.), details=True)
    return articulated_seam_evidence(raw, left, right, **kw)


def test_proposed_body_rank_matches_actual_crops_instead_of_native_screen_placement():
    a, b = observations()
    native = detail([a], [b], static_camera_terms=True)
    cameras_a = np.array([[1., .5, .5], [1., .5, .5], [1.3, .5, .5]])
    cameras_b = np.array([[1., .5, .5], [2., .6, .45], [1.6, .58, .47]])
    actual = projected_articulated_pose(native, 0, 0, cameras_a, cameras_b)
    for i, (ca, cb) in enumerate(zip(cameras_a, cameras_b)):
        left, right = dict(a, camera_strip=np.tile(ca, (7, 1))), dict(b, camera_strip=np.tile(cb, (7, 1)))
        expected = detail([left], [right])['pose_rate'][0, 0]
        assert actual[i] == pytest.approx(expected, abs=1e-12)
    assert actual[1] < 1e-6 and actual[0] > .05
    assert native['pose_rate'][0, 0] == actual[0]
    # Proposal scoring cannot alter or certify the native-source evidence.
    assert not native['anatomical_ownership_certified']


def test_unknown_arm_keeps_its_existing_rate_without_inventing_a_match():
    a, b = observations(missing=True)
    native = detail([a], [b], static_camera_terms=True)
    assert not native['articulated_arm_direction_measured'][0, 0]
    expected = native['pose_rate'][0, 0]
    np.testing.assert_array_equal(projected_articulated_pose(native, 0, 0,
        [[1., .5, .5], [1.2, .5, .5]], [[2., .6, .45], [1., .5, .5]]), [expected, expected])


def test_real_proposal_consumer_uses_projected_rate_before_discarding_camera_options(monkeypatch):
    from crosscut2 import played_interval_utility as utility
    from crosscut2.boundary_port_domain import make
    a, _, _ = prepared(0, bump=.08)
    b, _, _ = prepared(1, bump=.08)
    paths = [[(np.array([0, 200]), np.tile([1., .5, .5], (2, 1))) for _ in range(4)] for _ in range(2)]
    calls = []
    def projected(detail, ai, bi, ca, cb):
        calls.append((ca, cb))
        # Test the actual ranking consumer separately from the exact-math
        # property above: a deliberately worse first camera must lose.
        return np.array([100. if i != 1 else 0. for i in range(len(ca))])
    monkeypatch.setattr(utility, 'projected_articulated_pose', projected)
    entry, _, _, receipt = make(a.nodes, [a, b], paths, fps=30., articulated_actions=True)
    assert calls and receipt['articulated_proposal_rank_uses_actual_static_cameras']
    assert any(any(k[0] == 'paired' and k[2:6] == (0, 1, 0, 0) and k[6] == 1
                   for k in port.proposal_ids) for port in entry[1])


def test_supported_full_feature_handoff_keeps_its_existing_flow_interpretation(monkeypatch):
    from crosscut2 import played_interval_utility as utility
    from crosscut2.boundary_port_domain import make
    a, _, _ = prepared(0, bump=.08)
    b, _, _ = prepared(1, bump=.08)
    paths = [[(np.array([0, 200]), np.tile([1., .5, .5], (2, 1))) for _ in range(4)] for _ in range(2)]
    class Roles:
        def at(self, tick, left, right):
            return np.ones(np.broadcast_shapes(left.shape, right.shape))
    def domain(value):
        monkeypatch.setattr(utility, 'projected_articulated_pose',
            lambda detail, ai, bi, ca, cb: np.full(len(ca), value))
        entries, exits, _, _ = make(a.nodes, [a, b], paths, fps=30., articulated_actions=True, roles=Roles())
        return [[(p.source, p.geometry_hash, p.proposal_ids) for p in ports] for ports in (*entries, *exits)]
    assert domain(0.) == domain(100.)


@pytest.mark.parametrize('a,b', [([[0., .5, .5]], [[1., .5, .5]]),
    ([[1., .5, .5]], [[1., .5, .5], [1., .5, .5]]),
    ([[np.nan, .5, .5]], [[1., .5, .5]])])
def test_only_explicit_finite_static_camera_pairs_are_accepted(a, b):
    left, right = observations()
    native = detail([left], [right], static_camera_terms=True)
    with pytest.raises(ValueError, match='paired static cameras'):
        projected_articulated_pose(native, 0, 0, a, b)
