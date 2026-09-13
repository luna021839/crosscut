import unittest

from crosscut.joint_timeline import select_joint_timeline, select_joint_timeline_candidates
from crosscut.timeline import NoCompleteTimelinePath
from engine.tests.test_joint_timeline import (
    BatchedBridgeProvider, candidate, isolated_score_config, option_set, valid_bridge,
)


class JointSourceRunDurationTests(unittest.TestCase):
    def test_shifted_internal_flash_is_rejected_before_bridge_batch(self):
        rows = tuple((candidate(i, str(i), .9),) for i in range(3))
        def options(key):
            return option_set(key, authorized={5: 0.} if key.boundary_index == 0 else {-5: 0., 0: -.5})
        bridge = BatchedBridgeProvider()
        result = select_joint_timeline(rows, option_provider=options,
            bridge_provider=bridge.bridge_provider,
            bridge_disposition_batch_provider=bridge.bridge_disposition_batch,
            config=isolated_score_config(), fps=30., minimum_internal_source_run_seconds=.75)
        self.assertEqual([o.frame_offset for o in result.boundary_options], [5, 0])
        self.assertGreater(dict(result.rejection_counts)['internal_source_run_too_short'], 0)
        self.assertTrue(bridge.batch_requests)
        for batch in bridge.batch_requests:
            self.assertTrue(all(request.exit_option.frame_offset != -5 for request in batch))

    def test_no_policy_preserves_legacy_choice(self):
        rows = tuple((candidate(i, str(i), .9),) for i in range(3))
        result = select_joint_timeline(rows,
            option_provider=lambda key: option_set(key, authorized={5: 0.} if key.boundary_index == 0 else {-5: 0., 0: -.5}),
            bridge_provider=valid_bridge, config=isolated_score_config(), fps=30.)
        self.assertEqual([o.frame_offset for o in result.boundary_options], [5, -5])

    def test_source_run_entry_survives_same_source_state_merging(self):
        clocks = ((0., 1.), (1., 1.4), (1.4, 1.8), (1.8, 2.8))
        rows = tuple((candidate(i, source, .9, start_seconds=a, end_seconds=b),)
                     for i, (source, (a, b)) in enumerate(zip(('a', 'b', 'b', 'c'), clocks)))
        def options(key):
            return option_set(key, authorized={5: 0., -5: -.5} if key.boundary_index == 0 else {0: 0.})
        result = select_joint_timeline(rows, option_provider=options, bridge_provider=valid_bridge,
            config=isolated_score_config(), fps=30., minimum_internal_source_run_seconds=.75)
        # Both histories have offset 0 at the middle boundary. The more valuable
        # late entry cannot evict the earlier entry which alone leaves a valid run.
        self.assertEqual([o.frame_offset for o in result.boundary_options], [-5, 0, 0])

    def test_trimmed_first_and_last_runs_remain_exempt(self):
        clocks = ((0., .2), (.2, 1.2), (1.2, 1.4))
        rows = tuple((candidate(i, str(i), .9, start_seconds=a, end_seconds=b),)
                     for i, (a, b) in enumerate(clocks))
        results = select_joint_timeline_candidates(rows,
            option_provider=lambda key: option_set(key, authorized={0: 0.}),
            bridge_provider=valid_bridge, config=isolated_score_config(), fps=30.,
            minimum_internal_source_run_seconds=.75)
        self.assertTrue(results)

    def test_all_short_internal_options_fail_as_cadence_not_pixel_evidence(self):
        rows = tuple((candidate(i, str(i), .9),) for i in range(3))
        with self.assertRaises(NoCompleteTimelinePath) as caught:
            select_joint_timeline(rows,
                option_provider=lambda key: option_set(key, authorized={5: 0.} if key.boundary_index == 0 else {-5: 0.}),
                bridge_provider=lambda *args: self.fail('a known flash must not request bridge proof'),
                config=isolated_score_config(), fps=30., minimum_internal_source_run_seconds=.75)
        self.assertEqual(caught.exception.rejection_counts, {'internal_source_run_too_short': 1})

    def test_invalid_policy_is_rejected(self):
        rows = ((candidate(0, 'a', .9),), (candidate(1, 'b', .9),))
        for value in (True, 0., -1., float('nan'), float('inf'), '0.75'):
            with self.assertRaises(ValueError):
                select_joint_timeline(rows,
                    option_provider=lambda *args: self.fail('invalid policy must not decode'),
                    config=isolated_score_config(), minimum_internal_source_run_seconds=value)


if __name__ == '__main__':
    unittest.main()
