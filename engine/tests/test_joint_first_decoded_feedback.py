import unittest
from unittest.mock import patch

from crosscut.joint_visual_candidate_adapter import (
    JointDecodedMontageFeedback, JointDecodedMontageSegmentObservation,
    _blend_single_variant_role_adjustments, _single_variant_feedback_keys,
)
from crosscut.timeline import ShotType
from engine.tests.test_joint_visual_candidate_adapter import (
    _candidate_set, _profile_pools, StubDecodedProvider,
    select_joint_visual_candidates,  # Only the tiny fixture's global duration floors are relaxed.
)


class FirstDecodedFeedbackTests(unittest.TestCase):
    def feedback(self, target, variants=('a',), classes=('wide', 'medium', 'medium')):
        return JointDecodedMontageFeedback(tuple(
            JointDecodedMontageSegmentObservation.from_candidate(
                target, render_variant_signature=variant * 64,
                sampled_proxy_shot_classes=classes,
            ) for variant in variants
        ))

    def test_first_render_changes_all_profile_rankings_without_changing_graph(self):
        candidates = _candidate_set(role_conditioned=True)
        workspace = candidates.optimizer_workspace
        lattice = workspace.candidates_by_segment
        target = lattice[0][0]
        pools = _profile_pools(candidates)
        calls = []

        def controls(shadow, **kwargs):
            # The production selector calls existing role controls on both the
            # stable and provisional scoring lattices. Graph/proofs stay original.
            penalty = -8.0 if shadow[0][0].entry_shot_type is ShotType.WIDE else 0.0
            return {(0, '0'): penalty, (0, '1'): -2.0}, {1: 3.0}, {}

        def selector(actual_lattice, **kwargs):
            self.assertIs(actual_lattice, lattice)
            calls.append(kwargs)
            return pools[len(calls) - 1]

        with patch('crosscut.joint_visual_candidate_adapter._role_conditioned_visual_controls',
                   side_effect=controls), patch(
            'crosscut.joint_visual_candidate_adapter.select_joint_timeline_candidates',
            side_effect=selector,
        ):
            selection = select_joint_visual_candidates(
                candidates, provider=StubDecodedProvider(), fps=10.0,
                decoded_montage_feedback=self.feedback(target),
            )
        self.assertEqual(len(calls), 3)
        for call in calls:
            self.assertEqual(call['emission_adjustments'], {(0, '0'): -4.0, (0, '1'): -2.0})
            self.assertEqual(call['source_switch_bonus_by_segment'], {1: 3.0})
        self.assertIs(lattice[0][0], target)
        self.assertIsNone(target.entry_shot_type)
        for audit in selection.role_control_audits:
            evidence = audit['single_variant_decoded_feedback']
            self.assertEqual(evidence['adjusted_candidate_count'], 1)
            self.assertEqual(evidence['candidate_exclusions'], 0)
            self.assertFalse(evidence['shot_labels_overridden'])

    def test_already_repeated_evidence_is_not_penalized_twice(self):
        lattice = _candidate_set(role_conditioned=True).optimizer_workspace.candidates_by_segment
        self.assertFalse(_single_variant_feedback_keys(
            lattice, self.feedback(lattice[0][0], variants=('a', 'b'))))

    def test_nonadverse_observation_has_no_penalty(self):
        lattice = _candidate_set(role_conditioned=True).optimizer_workspace.candidates_by_segment
        self.assertFalse(_single_variant_feedback_keys(
            lattice, self.feedback(lattice[0][0], classes=('close', 'medium', 'close'))))

    def test_penalty_never_rewards_or_changes_unobserved_sources(self):
        base = {(0, '0'): -8., (0, '1'): -2.}
        result, audit = _blend_single_variant_role_adjustments(
            base, {(0, '0'): -4., (0, '1'): -100.}, {(0, '0')})
        self.assertEqual(result, base)
        self.assertEqual(audit['adjusted_candidate_count'], 0)

    def test_positive_or_nonfinite_penalties_are_rejected(self):
        for invalid in (1., float('nan'), float('-inf')):
            with self.assertRaises(ValueError):
                _blend_single_variant_role_adjustments({}, {(0, '0'): invalid}, {(0, '0')})


if __name__ == '__main__':
    unittest.main()
