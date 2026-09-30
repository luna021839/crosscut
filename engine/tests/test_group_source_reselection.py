from dataclasses import replace
import unittest
from crosscut.group_source_reselection import source_reselection_windows
from engine.tests.test_boundary_subject_alignment import _choice


def choices(sources):
    return tuple(_choice(i, source, start=float(i), end=float(i+1)) for i, source in enumerate(sources))


class GroupSourceReselectionTests(unittest.TestCase):
    def test_replacement_that_joins_next_source_sees_whole_source_hold(self):
        rows = choices((0, 0, 1, 1, 2, 2, 2, 2, 3))
        self.assertEqual(source_reselection_windows(rows, (2, 3)), ((0, 7),))
        self.assertGreater(rows[7].canonical_master_end-rows[2].canonical_master_start, 5.)

    def test_overlapping_contexts_are_solved_together_not_overwritten(self):
        self.assertEqual(source_reselection_windows(choices((0, 1, 2, 2, 3, 4)), (1, 4)), ((0, 5),))

    def test_disjoint_windows_remain_bounded_and_edges_are_clamped(self):
        self.assertEqual(source_reselection_windows(choices(tuple(range(12))), (0, 5, 11)),
            ((0, 1), (4, 6), (10, 11)))

    def test_no_mutable_cells_does_not_reanalyze_entire_song(self):
        self.assertEqual(source_reselection_windows(choices((0, 1)), ()), ())

    def test_invalid_and_noncontiguous_inputs_are_rejected(self):
        rows = choices((0, 1, 2))
        for indices in ((True,), (3,), (1, 1), (2, 1), (1,'2'), ([],)):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                source_reselection_windows(rows, indices)
        with self.assertRaises(ValueError):
            source_reselection_windows((rows[0], replace(rows[1], canonical_master_start=1.1), rows[2]), (1,))


if __name__ == '__main__':
    unittest.main()
