"""Training sampling must not let one densely labelled edit dominate."""
from collections import Counter
import unittest

from crosscut.stage_edit_production_training import _BalancedCutWindow, _HierarchicalCutWindowSampler
from crosscut.stage_edit_transformer import StageEditSequence, StageEditTargetSequence


def windows(song, ref, count, label=1):
    return tuple(_BalancedCutWindow(StageEditSequence(ref, ()),
                                   StageEditTargetSequence(ref, ()), label, ref, i, song)
                 for i in range(count))


class CutReferenceBalanceTests(unittest.TestCase):
    def test_large_reference_does_not_dominate_song_or_peer_reference(self):
        rows = (*windows('song-a', 'dense', 70), *windows('song-a', 'sparse', 3),
                *windows('song-b', 'other', 2), *windows('song-c', 'third', 1))
        sampler = _HierarchicalCutWindowSampler(rows, seed=3)
        chosen = [sampler.sample() for _ in range(600)]
        self.assertEqual(Counter({'song-a':200, 'song-b':200, 'song-c':200}), Counter(w.song_id for w in chosen))
        self.assertEqual(Counter(dense=100, sparse=100, other=200, third=200), Counter(w.reference_id for w in chosen))
        report = sampler.diagnostics()
        self.assertEqual(70, report['per_reference_available_windows']['dense'])
        self.assertEqual(100, report['per_reference_sampled_counts']['dense'])
        self.assertEqual(600, report['sample_count'])

    def test_seed_replay_and_input_order_invariance(self):
        rows = (*windows('a', 'one', 12), *windows('b', 'two', 8))
        a = _HierarchicalCutWindowSampler(rows, seed=123)
        b = _HierarchicalCutWindowSampler(rows[::-1], seed=123)
        self.assertEqual([a.sample() for _ in range(50)], [b.sample() for _ in range(50)])

    def test_negative_only_reference_is_not_invented_as_positive(self):
        positive = _HierarchicalCutWindowSampler(windows('a', 'positive', 1), seed=3)
        negative = _HierarchicalCutWindowSampler(windows('b', 'negative-only', 4, label=0), seed=3)
        for _ in range(10):
            self.assertEqual(1, positive.sample().label)
            self.assertEqual(0, negative.sample().label)
        self.assertNotIn('negative-only', positive.diagnostics()['per_reference_sampled_counts'])

    def test_rejects_empty_mixed_or_duplicated_anchor_input(self):
        for rows in ((), (*windows('a','ref',1), *windows('a','ref',1)),
                     (*windows('a','pos',1), *windows('a','neg',1,label=0))):
            with self.assertRaises(ValueError):
                _HierarchicalCutWindowSampler(rows, seed=1)


if __name__ == '__main__':
    unittest.main()
