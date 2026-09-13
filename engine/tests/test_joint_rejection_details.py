"""Failure causes survive nested reports without changing graph authority."""
from dataclasses import replace
import unittest

from crosscut.joint_timeline import select_joint_timeline
from crosscut.timeline import NoCompleteTimelinePath
from engine.tests.test_joint_timeline import candidate, isolated_score_config, option_set, valid_bridge


class JointRejectionDetailsTests(unittest.TestCase):
    def test_exact_timeline_gate_is_reported_separately(self):
        rows = ((replace(candidate(0, 'a', .9), shot_scale=.1),),
                (replace(candidate(1, 'b', .9), shot_scale=.9),))
        with self.assertRaises(NoCompleteTimelinePath) as caught:
            select_joint_timeline(rows, option_provider=lambda k: option_set(k, authorized={0: 0.}),
                                  config=replace(isolated_score_config(), maximum_cross_source_scale_delta=.2))
        error = caught.exception
        self.assertEqual(error.rejection_counts, {'boundary_option_timeline_hard_gate': 1})
        self.assertEqual(error.reason_code, 'joint_timeline_hard_incompatible')
        self.assertEqual(error.rejection_details, {'timeline/scale_delta': 1})
        wrapped = NoCompleteTimelinePath('outer', reason_code='outer', profile_failures=(('balanced', error),))
        self.assertEqual(wrapped.to_dict()['profile_failures'][0]['no_complete_path']['rejection_details'],
                         {'timeline/scale_delta': 1})

    def test_bridge_cause_does_not_become_hard_rejection(self):
        rows = tuple((candidate(i, str(i), .9),) for i in range(3))
        def incomplete(*args):
            return replace(valid_bridge(*args), evidence_complete=False, crop_compatible=False,
                           track_compatible=False, failure_codes=('segment_cut_containment_pixel_flow_failed',))
        with self.assertRaises(NoCompleteTimelinePath) as caught:
            select_joint_timeline(rows, option_provider=lambda k: option_set(k, authorized={0: 0.}),
                                  bridge_provider=incomplete, config=isolated_score_config())
        error = caught.exception
        self.assertEqual(error.rejection_counts, {'bridge_evidence_incomplete': 1})
        self.assertEqual(error.reason_code, 'joint_timeline_evidence_incomplete')
        self.assertEqual(error.rejection_details,
                         {'bridge_incomplete/segment_cut_containment_pixel_flow_failed': 1})

    def test_empty_details_preserve_legacy_payload(self):
        self.assertNotIn('rejection_details', NoCompleteTimelinePath('empty', reason_code='empty').to_dict())


if __name__ == '__main__':
    unittest.main()
