from dataclasses import replace
import unittest
from unittest.mock import patch

from crosscut.joint_highlight_opportunity import redistribute_highlight_switch_budget
from crosscut.joint_timeline import select_joint_timeline
from crosscut.timeline import NoCompleteTimelinePath, ShotType
from engine.tests.test_joint_timeline import candidate, isolated_score_config, option_set, valid_bridge


class JointHighlightOpportunityTests(unittest.TestCase):
    def _run(self, *, bonus=0.0, missing=False, rejected=False, offset_close=False):
        def close(c):
            return replace(c, shot_type=ShotType.CLOSE_UP,
                           entry_shot_type=ShotType.CLOSE_UP,
                           exit_shot_type=ShotType.CLOSE_UP)
        a = close(candidate(0, 'A', .9))
        hold = close(candidate(1, 'A', .8))
        wide = replace(candidate(1, 'B', 1.0), entry_shot_type=ShotType.WIDE,
                       exit_shot_type=ShotType.WIDE, shot_type=ShotType.WIDE)
        c = close(candidate(1, 'C', .4))
        if missing:
            a = replace(a, exit_shot_type=None)
        lattice = ((a,), (hold, wide, c))
        by_id = {x.candidate_id: x for row in lattice for x in row}

        def provider(key):
            left, right = by_id[key.outgoing.candidate_id], by_id[key.incoming.candidate_id]
            if right.source_id == 'C' and rejected:
                return option_set(key)
            if right.source_id == 'C' and offset_close:
                return option_set(key, authorized={0: 0.0, 1: 0.0},
                                  scoring_candidates_by_offset={
                                      0: (left, replace(right, entry_shot_type=ShotType.WIDE)),
                                      1: (left, right)})
            return option_set(key, authorized={0: 0.0},
                              scoring_candidates_by_offset={0: (left, right)})

        return select_joint_timeline(lattice, option_provider=provider,
                                     bridge_provider=valid_bridge, fps=30.,
                                     config=isolated_score_config(),
                                     nonwide_source_switch_bonus_by_segment={1: bonus})

    def test_actual_dp_rewards_close_pair_not_wide_or_same_source(self):
        self.assertEqual(self._run().candidates[-1].source_id, 'B')
        result = self._run(bonus=1.)
        self.assertEqual(result.candidates[-1].source_id, 'C')
        self.assertAlmostEqual(result.total_score, 2.3)

    def test_missing_exact_class_cannot_use_midpoint_close_class(self):
        self.assertEqual(self._run(bonus=10., missing=True).candidates[-1].source_id, 'B')

    def test_large_bonus_never_revives_hard_incompatible_pair(self):
        self.assertEqual(self._run(bonus=1000., rejected=True).candidates[-1].source_id, 'B')

    def test_bonus_follows_selected_exact_offset_not_other_option(self):
        result = self._run(bonus=1., offset_close=True)
        self.assertEqual(result.candidates[-1].source_id, 'C')
        self.assertEqual(result.boundary_options[0].frame_offset, 1)

    def test_invalid_bonus_rejected(self):
        for value in (True, -1., float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self._run(bonus=value)

    def test_budget_moves_to_all_short_neighbors_without_multiplying(self):
        lattice = tuple((candidate(i, 'A', 1.), candidate(i, 'B', 1.)) for i in range(6))
        ordinary, conditional, audit = redistribute_highlight_switch_budget(
            lattice, existing_bonuses={1: 12., 3: 12.},
            section_targets=({'section_index': 0},), section_index_at=lambda t: 0,
            maximum_nominal_shot_seconds=1.5)
        self.assertEqual(ordinary, {i: 0. for i in range(1, 6)})
        self.assertEqual(set(conditional), set(range(1, 6)))
        self.assertAlmostEqual(sum(conditional.values()), 24.)
        self.assertFalse(audit['qualified_burst_proof'])

    def test_long_or_cross_section_spans_do_not_gain_short_pair_bonus(self):
        lattice = ((candidate(0, 'A', 1., start_seconds=0., end_seconds=2.),),
                   (candidate(1, 'B', 1., start_seconds=2., end_seconds=3.),),
                   (candidate(2, 'C', 1., start_seconds=3., end_seconds=4.),))
        ordinary, conditional, _ = redistribute_highlight_switch_budget(
            lattice, existing_bonuses={1: 12., 2: 5.},
            section_targets=({'section_index': 0},),
            section_index_at=lambda t: 0 if t < 3 else 1,
            maximum_nominal_shot_seconds=1.5)
        self.assertEqual(ordinary, {1: 12., 2: 5.})
        self.assertEqual(conditional, {})

    def test_actual_adapter_keeps_balanced_and_forwards_conditional_profiles(self):
        from engine.tests import test_joint_visual_candidate_adapter as fixtures
        from crosscut.joint_visual_candidate_adapter import select_joint_visual_candidates

        candidates = fixtures._candidate_set(role_conditioned=True)
        pools = fixtures._profile_pools(candidates)
        calls = []

        def select(*args, **kwargs):
            calls.append(kwargs)
            return pools[len(calls) - 1]

        with patch('crosscut.joint_visual_candidate_adapter.select_joint_timeline_candidates',
                   side_effect=select):
            # The injected old pools do not satisfy current role review. Keep
            # that failure; this test checks production controls forwarding,
            # not a counterfeit successful final candidate cohort.
            with self.assertRaises(NoCompleteTimelinePath):
                select_joint_visual_candidates(candidates, provider=fixtures.StubDecodedProvider(), fps=10.)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]['nonwide_source_switch_bonus_by_segment'], {})
        self.assertTrue(calls[0]['source_switch_bonus_by_segment'])
        for call in calls[1:]:
            self.assertEqual(set(call['nonwide_source_switch_bonus_by_segment']), {1, 2})
            self.assertTrue(all(v > 0 for v in call['nonwide_source_switch_bonus_by_segment'].values()))
            self.assertEqual(call['source_switch_bonus_by_segment'], {1: 0., 2: 0.})


if __name__ == '__main__':
    unittest.main()
