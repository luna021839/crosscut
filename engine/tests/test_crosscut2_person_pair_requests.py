import numpy as np
import pytest

from crosscut2.person_extent import height_interval, interval_gap, used_face_risk
from crosscut2.person_pair_requests import reserve_complete_pairs

KEYS = {'new_frames', 'accepted_pair_indices', 'deferred_pair_indices', 'new_frame_count',
        'already_cached_pair_count', 'pixel_inference_or_bank_completeness_certified'}


def test_unknown_a_known_face_b_still_reserves_b_person_key():
    # A known face on B is not a person-rectangle observation of B.
    result = reserve_complete_pairs([((0, 40), (1, 52))], available=[])
    assert result['new_frames'] == [(0, 40), (1, 52)]
    assert result['accepted_pair_indices'] == [0] and result['deferred_pair_indices'] == []
    assert result['already_cached_pair_count'] == 0


def test_cropped_out_known_face_needs_current_port_endpoint_without_neighbour_substitution():
    available = {(1, 51), (1, 53), (0, 40)}
    result = reserve_complete_pairs([((0, 40), (1, 52))], available)
    assert result['new_frames'] == [(1, 52)]
    assert result['new_frame_count'] == 1
    assert result['already_cached_pair_count'] == 0


def test_shared_frames_count_once():
    pairs = [((0, 5), (1, 7)), ((0, 5), (2, 9)), ((1, 7), (2, 9))]
    result = reserve_complete_pairs(pairs, (), maximum_new_frames=3)
    assert result['accepted_pair_indices'] == [0, 1, 2]
    assert result['new_frames'] == [(0, 5), (1, 7), (2, 9)]
    assert result['new_frame_count'] == 3


def test_native_skipped_pair_uses_exposed_ordinals():
    result = reserve_complete_pairs([((3, 97), (3, 100))], ())
    assert result['new_frames'] == [(3, 97), (3, 100)]
    assert (3, 99) not in result['new_frames']


def test_last_budget_cannot_take_one_of_two_missing_and_later_pairs_still_fit():
    pairs = [((0, 1), (1, 1)), ((0, 2), (1, 2)), ((0, 1), (2, 5)), ((1, 1), (0, 1)), ((0, 2), (1, 1))]
    result = reserve_complete_pairs(pairs, (), maximum_new_frames=3)
    assert result['accepted_pair_indices'] == [0, 2, 3]
    assert result['deferred_pair_indices'] == [1, 4]
    assert result['new_frames'] == [(0, 1), (1, 1), (2, 5)]
    assert (0, 2) not in result['new_frames'] and (1, 2) not in result['new_frames']


def test_later_one_miss_pair_fits_after_expensive_pair_is_deferred():
    pairs = [((0, 1), (1, 1)), ((0, 1), (5, 5))]
    result = reserve_complete_pairs(pairs, {(5, 5)}, maximum_new_frames=1)
    assert result['deferred_pair_indices'] == [0]
    assert result['accepted_pair_indices'] == [1]
    assert result['new_frames'] == [(0, 1)]


def test_empty_cached_attempt_is_coverage_not_known_extent():
    bank = {(0, 10): np.array([[.1, .1, .3, .9, .8]], np.float32),
            (1, 20): np.zeros((0, 5), np.float32)}
    result = reserve_complete_pairs([((0, 10), (1, 20))], bank.keys())
    assert result['accepted_pair_indices'] == [0]
    assert result['new_frames'] == [] and result['already_cached_pair_count'] == 1
    camera = np.array([1., .5, .5])
    gap, known = interval_gap(height_interval(bank[(0, 10)], camera),
                              height_interval(bank[(1, 20)], camera))
    assert not known and gap == 0.
    assert used_face_risk(1., False, gap, known) == 1.
    assert used_face_risk(.3, True, gap, known) == .3


def test_global_cap_and_pair_closure_invariants():
    rng = np.random.default_rng(5)
    domain = [(s, f) for s in range(3) for f in range(6)]
    for _ in range(300):
        count = int(rng.integers(0, 13))
        pairs = [(domain[int(rng.integers(len(domain)))], domain[int(rng.integers(len(domain)))])
                 for _ in range(count)]
        available = {k for k in domain if rng.random() < .3}
        cap = int(rng.integers(0, 7))
        result = reserve_complete_pairs(pairs, available, maximum_new_frames=cap)
        new = result['new_frames']
        assert set(result) == KEYS
        assert result['new_frame_count'] == len(new) <= cap
        assert new == sorted(set(new)) and not set(new) & available
        accepted, deferred = result['accepted_pair_indices'], result['deferred_pair_indices']
        assert sorted(accepted + deferred) == list(range(count))
        assert accepted == sorted(accepted) and deferred == sorted(deferred)
        planned = available | set(new)
        for i in accepted:
            assert set(pairs[i]) <= planned
        for i in deferred:
            assert not set(pairs[i]) <= planned
        used = {k for i in accepted for k in pairs[i]}
        assert set(new) <= used
        assert result['already_cached_pair_count'] == sum(set(p) <= available for p in pairs)


def test_no_source_quota_zero_budget_and_duplicates():
    pairs = [((0, f), (0, f + 3)) for f in range(10)]
    result = reserve_complete_pairs(pairs, (), maximum_new_frames=20)
    assert result['accepted_pair_indices'] == list(range(10))
    result = reserve_complete_pairs([((0, 1), (1, 1)), ((2, 2), (3, 3))], {(0, 1), (1, 1)},
                                    maximum_new_frames=0)
    assert result['accepted_pair_indices'] == [0] and result['deferred_pair_indices'] == [1]
    assert result['new_frames'] == []
    result = reserve_complete_pairs([((0, 1), (1, 1))] * 3, (), maximum_new_frames=2)
    assert result['accepted_pair_indices'] == [0, 1, 2] and result['new_frame_count'] == 2


def test_deterministic_and_integer_normalization():
    pairs = [((np.int64(1), 4), (0, np.int64(2))), ((2, 2), (0, 0))]
    cached = [(0, 0), (1, 9)]
    first = reserve_complete_pairs(pairs, cached, maximum_new_frames=3)
    for variant in (list(reversed(cached)), set(cached), [list(k) for k in cached]):
        assert reserve_complete_pairs(pairs, variant, maximum_new_frames=3) == first
    assert all(type(x) is int for key in first['new_frames'] for x in key)
    assert first['pixel_inference_or_bank_completeness_certified'] is False


def test_validation():
    good = ((0, 1), (1, 1))
    bad_pairs = [[((True, 1), (1, 1))], [((0, -1), (1, 1))], [((0, 1.0), (1, 1))],
                 [((0, 1),)], [((0, 1), (1, 1), (2, 2))], [((0, 1, 2), (1, 1))],
                 ['ab'], [((0, 1), 'xy')], 5]
    for pairs in bad_pairs:
        with pytest.raises(ValueError):
            reserve_complete_pairs(pairs, ())
    for available in ([(0, False)], [(0, -2)], [5], 7):
        with pytest.raises(ValueError):
            reserve_complete_pairs([good], available)
    for cap in (True, -1, 2.0, None):
        with pytest.raises(ValueError):
            reserve_complete_pairs([good], (), maximum_new_frames=cap)
