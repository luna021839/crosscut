from dataclasses import replace
import unittest
from crosscut.reference_editorial_agreement import (
    EditorialReference, classify_reference_neighborhood, editorial_agreement_report,
    reference_editorial_evidence,
)


class EditorialAgreementTests(unittest.TestCase):
    def setUp(self):
        self.ref = EditorialReference('a', 'song', 'style', (2.,), (4.,), (2., 4., 6.), ((0., 8.),))

    def test_unknown_boundary_and_missing_coverage_are_not_hold_votes(self):
        self.assertEqual('unknown', classify_reference_neighborhood(self.ref, 6., .1))
        self.assertEqual('unknown', classify_reference_neighborhood(self.ref, 9., .1))
        self.assertEqual('unknown', classify_reference_neighborhood(self.ref, .05, .1))
        self.assertEqual('known_hold', classify_reference_neighborhood(self.ref, 3., .1))

    def test_editorial_and_native_cuts_are_distinct(self):
        self.assertEqual('editorial_cut', classify_reference_neighborhood(self.ref, 2.05, .1))
        self.assertEqual('native_cut_only', classify_reference_neighborhood(self.ref, 4.05, .1))
        mixed = replace(self.ref, boundaries=(4., 4.05))
        self.assertEqual('unknown', classify_reference_neighborhood(mixed, 4., .1))

    def test_only_other_same_song_references_vote_and_unknown_is_excluded(self):
        hold = replace(self.ref, reference_id='b', positives=(), boundaries=())
        unknown = replace(hold, reference_id='c', known_intervals=())
        other_song = replace(self.ref, reference_id='d', song_id='other')
        report = editorial_agreement_report((self.ref, hold, unknown, other_song), tolerance_seconds=.1)
        self.assertEqual(dict(editorial_cut=0, native_cut_only=0, known_hold=1, unknown=1), report['peer_vote_counts'])
        self.assertEqual(0., report['observed_peer_cut_agreement'])
        self.assertFalse(report['labels_changed'])

    def test_requires_explicit_unmasked_boundary_and_known_dense_shot(self):
        payload = dict(reference_id='a', style_id='s', detected_boundary_records=[
            dict(cut_master_time=2., loss_masked=True, target_masks=dict(cut=True),
                 is_editorial_source_switch_positive=True, cut_target=1.)],
            shot_labels=[dict(selected_source_id='source', edit_labels=dict(
                dense_attribution=dict(ready=False), cut=dict(cut_master_time=0., end_master_time=5.)))])
        result = reference_editorial_evidence(payload, song_id='song')
        self.assertEqual((), result.positives)
        self.assertEqual((), result.known_intervals)
        self.assertEqual('unknown', classify_reference_neighborhood(result, 2., .1))


if __name__ == '__main__':
    unittest.main()
