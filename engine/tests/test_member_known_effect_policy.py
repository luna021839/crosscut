from dataclasses import replace
import unittest
from unittest.mock import patch

import crosscut.direct_member_visual as visual
from crosscut.member_observation_search import ObservationState, ProposalEdge
from engine.tests.test_member_bridge_spacing import edge as verified_edge


class KnownMemberEffectPolicyTests(unittest.TestCase):
    def fixture(self):
        graph = [[[ProposalEdge(
            ObservationState.CONTENT_INELIGIBLE if left == right else ObservationState.UNOBSERVED, 1.)
            for right in range(3)] for left in range(3)] for _ in range(5)]
        graph[0][0][1] = ProposalEdge(ObservationState.ELIGIBLE, .10, policy_evidence=verified_edge())
        graph = tuple(tuple(tuple(row) for row in matrix) for matrix in graph)
        context = dict(
            represented_source_count=3, source_count=3, quotas=(2, 2, 2), order=(0, 1, 2),
            candidate_validity=((True,) * 3,) * 6,
            spans=tuple((3. * i, 3. * (i + 1)) for i in range(6)),
            resolved_config=visual.DirectMemberVisualConfig(maximum_same_source_hold_count=0),
            normalized_forbidden_paths=(), priority_boundaries=(0,),
            forbidden_hold_boundaries=(), maximum_priority_cut_hold_count=0,
            source_frame_geometries=(visual.direct_member_source_frame_geometry((2160, 3840)),) * 3,
            render_infeasible_edges=(), ranked_source_indices=(0, 1, 2),
            represented_source_indices=(0, 1, 2), quota_assignment_provenance={},
            heterogeneous_cinematic_authorized=True, span_diagnostics=(),
            valid_count_by_source=(6, 6, 6), hard_boundary_connectivity={},
            cinematic_bridge_enabled=True, continuous_official_portrait_fancam_contract=True,
        )
        return graph, context

    def test_real_effect_solver_keeps_verified_progress_before_unknown_hard_search(self):
        graph, context = self.fixture()
        with patch.object(visual, '_balanced_visual_path',
                          side_effect=AssertionError('unknown hard path must not discard a known effect')):
            result = visual._decide_direct_member_path_policy(graph, **context)
        self.assertEqual(result.path[:2], (0, 1))
        self.assertEqual(set(result.path), {0, 1, 2})
        self.assertIn(result.usage_policy, visual._DIRECT_MEMBER_CINEMATIC_USAGE_POLICIES)
        audit = result.optimizer_search_diagnostics['verified_effect_proposal']
        self.assertTrue(audit['unknown_edges_still_require_observation'])
        self.assertFalse(audit['hard_cut_severe_flags_cleared'])
        self.assertFalse(audit['render_authorized'])
        self.assertEqual(graph[0][0][1].severe_flags, ('attention_scale',))
        self.assertTrue(result.optimizer_search_diagnostics['solver']['postsolve_path_audit_passed'])
        self.assertEqual(result.optimizer_search_diagnostics['cinematic_bridge_boundary_indices'], [0])

    def test_unknown_rejected_legacy_and_stale_effects_cannot_trigger_priority(self):
        good = verified_edge()
        for candidate in (
            ProposalEdge(ObservationState.UNOBSERVED, .1, policy_evidence=good),
            ProposalEdge(ObservationState.REJECTED, .1, policy_evidence=good),
            ProposalEdge(ObservationState.ELIGIBLE, .1, policy_evidence=replace(good, motion_connection=None)),
            ProposalEdge(ObservationState.ELIGIBLE, .1, policy_evidence=verified_edge(focal=False)),
            ProposalEdge(ObservationState.ELIGIBLE, .1, policy_evidence=replace(
                good, severe_flags=(*good.severe_flags, 'dense_render_camera_infeasible'))),
            good,
        ):
            self.assertEqual(visual._known_focal_effect_proposal_edges((((candidate,),),), (0,)), ())
        # Use a genuine cross-source index; diagonal entries never authorize it.
        projected = ProposalEdge(ObservationState.ELIGIBLE, .1, policy_evidence=good)
        graph = (((projected, projected), (projected, projected)),)
        self.assertEqual(visual._known_focal_effect_proposal_edges(graph, (0,)), ((0, 0, 1), (0, 1, 0)))
        for candidate in (
            replace(projected, state=ObservationState.UNOBSERVED),
            replace(projected, policy_evidence=replace(good, motion_connection=None)),
            replace(projected, policy_evidence=verified_edge(focal=False)),
            replace(projected, policy_evidence=replace(good, severe_flags=('dense_render_camera_infeasible',))),
        ):
            self.assertEqual(visual._known_focal_effect_proposal_edges(
                (((candidate, candidate), (candidate, candidate)),), (0,)), ())

    def test_toggle_and_missing_source_authority_keep_previous_hard_policy(self):
        graph, base = self.fixture()
        for field in ('cinematic_bridge_enabled', 'heterogeneous_cinematic_authorized',
                      'continuous_official_portrait_fancam_contract'):
            with (
                self.subTest(field=field),
                patch.object(visual, '_heterogeneous_cinematic_visual_path') as effects,
                patch.object(visual, '_balanced_visual_path', return_value=((0, 1, 2, 0, 1, 2), .5, {})),
            ):
                result = visual._decide_direct_member_path_policy(graph, **{**base, field: False})
            effects.assert_not_called()
            self.assertEqual(result.usage_policy, 'exact_balanced_quota_v1')

    def test_failed_effect_proposal_falls_back_without_duplicate_effect_solves(self):
        graph, context = self.fixture()
        effect_failure = visual.DirectMemberVisualSelectionError('bounded effect search exhausted',
            {'failure': 'cinematic_search_capacity', 'constraint_infeasibility_proven': False})
        strict_failure = visual.DirectMemberVisualSelectionError('strict domain exhausted', {
            'failure': 'validity_quota_path_exhausted', 'constraint_infeasibility_proven': True,
            'search': {'exact_lifeline': {'strategy': 'scipy_highs_binary_path_milp_v1',
                'path_found': False, 'state_capacity_exhausted': False,
                'duration_state_quantization_applied': False}},
        })
        with (
            patch.object(visual, '_heterogeneous_cinematic_visual_path', side_effect=effect_failure) as effects,
            patch.object(visual, '_balanced_visual_path', side_effect=strict_failure),
            patch.object(visual, '_bounded_same_source_hold_visual_path', return_value=None),
        ):
            with self.assertRaises(visual.DirectMemberVisualSelectionError) as caught:
                visual._decide_direct_member_path_policy(graph, **context)
        effects.assert_called_once()
        self.assertFalse(caught.exception.diagnostics['verified_effect_proposal']['path_found'])
        self.assertFalse(caught.exception.diagnostics['constraint_infeasibility_proven'])


if __name__ == '__main__':
    unittest.main()
