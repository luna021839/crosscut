from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import crosscut.direct_member_visual as visual
from crosscut.member_observation_search import MemberObservationSearchIncomplete, ProposalEdge, ObservationState
from engine.tests.test_direct_member_visual import _descriptor, _optimizer_mismatch
from engine.tests.test_source_first_focal_geometry import geometry
from engine.tests.test_member_content_features import without_joints


class Provider:
    def __init__(self):
        self.focal_queries = []
        self.prepared = []
    def __call__(self, source, path, seconds):
        return _descriptor(.2+.02*source)
    def content_at(self, source, path, seconds):
        return without_joints(self(source, path, seconds))
    focal_proposal_at = content_at
    def focal_geometry_at(self, source, seconds, raw):
        self.focal_queries.append((source, seconds))
        return replace(geometry(source), source_seconds=seconds)
    def prepare_focal_observations(self, queries):
        self.prepared.extend(queries)


class DemandMemberSelectionTests(unittest.TestCase):
    def select(self, *, demand=True, config=None, **kwargs):
        provider = Provider()
        args = dict(sources=tuple(Path(f'{i}.mp4') for i in range(3)),
            canonical_spans=tuple((float(3*i), float(3*i+3)) for i in range(6)),
            audio_offsets=(0.,)*3, descriptor_provider=provider,
            source_frame_sizes=((2160, 3840),)*3,
            continuous_official_portrait_fancam_contract=True,
            demand_driven_observations=demand,
            config=config or visual.DirectMemberVisualConfig(beam_width=1000,
                boundary_search_radius_frames=2, sample_frame_step_seconds=1/30))
        args.update(kwargs)
        return visual.select_balanced_direct_member_sources(**args), provider

    def test_public_selector_excludes_only_the_observed_missing_endpoint_not_whole_source(self):
        from crosscut.focal_selector_geometry import FocalSelectorGeometry
        class MissingEndpoint(Provider):
            def focal_geometry_at(self, source, seconds, raw):
                result = super().focal_geometry_at(source, seconds, raw)
                if source == 0 and 2.5 < seconds < 3.5:
                    return FocalSelectorGeometry(source, seconds, None)
                return result
        provider = MissingEndpoint()
        with patch.object(visual, 'MemberObservationSearch', wraps=visual.MemberObservationSearch) as coordinator:
            result, _ = self.select(descriptor_provider=provider,
                config=visual.DirectMemberVisualConfig(beam_width=1000, boundary_search_radius_frames=2,
                                                       maximum_same_source_hold_count=0))
        self.assertEqual(result.source_counts, (2,2,2))
        self.assertNotIn(0, result.source_indices[:2])
        self.assertIn(0, result.source_indices[2:])
        audit = result.optimizer_search_diagnostics['demand_driven_observations']
        self.assertGreater(audit['endpoint_excluded_transition_count'], 0)
        self.assertGreater(result.boundaries[0].endpoint_excluded_pair_count, 0)
        self.assertEqual(coordinator.call_args.kwargs['maximum_solver_calls'],
                         coordinator.call_args.kwargs['maximum_observations']+1)
        self.assertTrue(all(not boundary.mismatch.severe_flags for boundary in result.boundaries))

    def test_public_selector_returns_only_exact_observed_full_policy_samples(self):
        result, provider = self.select()
        full, exhaustive = self.select(demand=False)
        self.assertEqual(result.source_counts, (2, 2, 2))
        self.assertEqual(len(result.boundaries), len(full.boundaries))
        self.assertLess(len(provider.focal_queries), len(exhaustive.focal_queries))
        self.assertEqual(provider.prepared, provider.focal_queries)
        self.assertTrue(all(not boundary.mismatch.severe_flags for boundary in result.boundaries))
        self.assertTrue(all(boundary.unobserved_pair_count > 0 for boundary in result.boundaries))
        self.assertTrue(all(boundary.severe_pair_count == 0 for boundary in result.boundaries))
        for boundary in result.boundaries:
            for sample, source in ((boundary.outgoing_sample, boundary.outgoing_eligible_source_index),
                                   (boundary.incoming_sample, boundary.incoming_eligible_source_index)):
                self.assertEqual(sample.focal_geometry.source_index, source)
                self.assertEqual(sample.source_local_seconds, sample.focal_geometry.source_seconds)
        audit = result.optimizer_search_diagnostics['demand_driven_observations']
        self.assertTrue(audit['policy_resolved_after_final_observation'])
        self.assertFalse(audit['global_visual_optimality_claimed'])
        self.assertGreater(audit['unobserved_transition_count'], 0)
        self.assertGreater(audit['reused_observation_count'], 0)
        self.assertEqual(audit['observed_transition_count'],
                         audit['exact_pair_request_count']+audit['reused_observation_count'])
        json.dumps(result.to_dict(), allow_nan=False)

    def force_first_boundary(self, *, bridge):
        actual = visual._boundary_mismatch
        def mismatch(outgoing, incoming, config, **kwargs):
            result = actual(outgoing, incoming, config, **kwargs)
            if 2.5 < outgoing.master_seconds < 3.5 and kwargs['outgoing_source_index'] != kwargs['incoming_source_index']:
                result = replace(result, cost=.2, severe_flags=('attention_scale',),
                    cinematic_bridge=(visual.DirectMemberCinematicBridge(
                        'pose_dissolve', False, False, ('attention_scale',), .02, .05, .02)
                        if bridge else None))
            return result
        return patch.object(visual, '_boundary_mismatch', side_effect=mismatch)

    def test_actual_policy_retains_bounded_hold_instead_of_discarding_its_sources(self):
        config = visual.DirectMemberVisualConfig(beam_width=1000,
            maximum_same_source_hold_count=1, maximum_same_source_hold_fraction=.3)
        with self.force_first_boundary(bridge=False):
            result, _ = self.select(config=config)
        self.assertEqual(result.source_counts, (2, 2, 2))
        self.assertFalse(result.boundaries[0].cut_applied)
        self.assertTrue(all(boundary.cut_applied for boundary in result.boundaries[1:]))
        self.assertTrue(result.optimizer_search_diagnostics['demand_driven_observations']
            ['policy_resolved_after_final_observation'])

    def test_actual_cinematic_policy_and_joint_offset_solver_keep_required_dissolve(self):
        config = visual.DirectMemberVisualConfig(beam_width=1000,
            maximum_same_source_hold_count=0, boundary_search_radius_frames=2)
        with self.force_first_boundary(bridge=True):
            result, _ = self.select(config=config, cinematic_bridge_enabled=True,
                                    priority_cut_boundary_indices=(0,))
        self.assertIn(result.usage_policy, visual._DIRECT_MEMBER_CINEMATIC_USAGE_POLICIES)
        self.assertIsNotNone(result.boundaries[0].mismatch.cinematic_bridge)
        self.assertEqual(result.boundaries[0].mismatch.cinematic_bridge.style, 'pose_dissolve')
        self.assertEqual(set(result.source_indices), {0, 1, 2})
        json.dumps(result.to_dict(), allow_nan=False)

    def test_effect_off_does_not_treat_required_dissolve_as_a_hard_cut(self):
        with self.force_first_boundary(bridge=True):
            with self.assertRaises(MemberObservationSearchIncomplete) as caught:
                self.select(config=visual.DirectMemberVisualConfig(beam_width=1000,
                    maximum_same_source_hold_count=0), cinematic_bridge_enabled=False)
        self.assertFalse(caught.exception.diagnostics['constraint_infeasibility_proven'])
        self.assertFalse(caught.exception.diagnostics['retry_as_another_partition'])

    def test_projection_preserves_real_bridge_but_counterfactual_is_separate(self):
        bridge = visual.DirectMemberCinematicBridge('pose_dissolve', False, False, ('attention_scale',), .02, .05, .02)
        real = visual.DirectBoundaryMismatch(.2, .1, .1, .1, .1, 'pose', .1, .1, .1, .1,
            3, ('attention_scale',), ('attention_scale',), cinematic_bridge=bridge)
        projected = ProposalEdge(ObservationState.ELIGIBLE, .1, policy_evidence=real)
        self.assertIs(projected.cinematic_bridge, bridge)
        self.assertEqual(projected.severe_flags, real.severe_flags)
        self.assertEqual(replace(projected, severe_flags=()).severe_flags, ())
        self.assertEqual(real.severe_flags, ('attention_scale',))

    def test_proposal_solver_does_not_change_measured_quality_or_lexicographic_objectives(self):
        provisional = (((ProposalEdge(ObservationState.UNOBSERVED, 1.),),),)
        measured = (((_optimizer_mismatch(.2),),),)
        self.assertEqual(visual._member_milp_solve_limits(provisional),
                         {'time_limit': 4., 'mip_rel_gap': .10})
        self.assertEqual(visual._member_milp_solve_limits(provisional, lexicographic=True),
                         {'time_limit': 4., 'mip_rel_gap': 0.})
        self.assertEqual(visual._member_milp_solve_limits(measured),
                         {'time_limit': 45., 'mip_rel_gap': 0.})

    def test_proposal_solver_timeout_is_not_infeasibility_and_bad_witness_is_not_accepted(self):
        graph = tuple(tuple(tuple(ProposalEdge(ObservationState.UNOBSERVED, 1.)
            for _ in range(3)) for _ in range(3)) for _ in range(5))
        def solve():
            return visual._milp_feasibility_lifeline(graph, source_quotas=(2,2,2),
                source_order=(0,1,2), candidate_validity=((True,)*3,)*6,
                durations=(1.,)*6, duration_targets=(2.,)*3, state_limit=1000,
                config=visual.DirectMemberVisualConfig())
        with patch('scipy.optimize.milp', return_value=SimpleNamespace(
                status=1, success=False, message='limit', x=None)) as milp:
            result = solve()
        self.assertIsNone(result.node)
        self.assertTrue(result.state_capacity_exhausted)
        self.assertEqual(milp.call_args.kwargs['options']['mip_rel_gap'], .10)
        def invalid_witness(objective, **kwargs):
            import numpy as np
            answer = np.zeros(len(objective))
            answer[range(0,18,3)] = 1.
            return SimpleNamespace(status=0, success=True, message='fixture', x=answer)
        with patch('scipy.optimize.milp', side_effect=invalid_witness):
            result = solve()
        self.assertIsNone(result.node)
        self.assertTrue(result.state_capacity_exhausted)
        self.assertIn('source_quota_mismatch', result.solver_diagnostics['postsolve_path_audit_failures'])

    def test_joint_clock_failure_does_not_become_a_whole_domain_partition_retry(self):
        error = visual.DirectMemberVisualSelectionError('joint fixture',
            {'failure': 'boundary_local_search_duration_infeasible', 'constraint_infeasibility_proven': True})
        with patch.object(visual, '_joint_boundary_search_assignment', side_effect=error) as joint:
            with self.assertRaises(MemberObservationSearchIncomplete) as caught:
                self.select()
        joint.assert_called_once()
        self.assertEqual(caught.exception.diagnostics['reason'], 'joint_cut_time_assignment_incomplete')
        self.assertFalse(caught.exception.diagnostics['constraint_infeasibility_proven'])
        self.assertGreater(caught.exception.diagnostics['unobserved_transition_count'], 0)

    def test_coarse_flank_ranking_cannot_become_cut_evidence_or_remove_a_demand(self):
        with patch.object(visual, '_coarse_member_proposal_prior', return_value=1.):
            result, provider = self.select()
        self.assertTrue(provider.focal_queries)
        self.assertTrue(all(boundary.outgoing_sample.focal_geometry is not None
                            and boundary.incoming_sample.focal_geometry is not None
                            for boundary in result.boundaries))

    def test_demand_mode_rejects_missing_focal_authority_before_opening_sources(self):
        with patch.object(visual, '_VideoDescriptorProvider', side_effect=AssertionError('must not open')):
            with self.assertRaises(ValueError):
                self.select(descriptor_provider=None)

    def test_real_partition_entry_keeps_user_effect_switch_on_first_demand_candidate(self):
        from crosscut.pipeline import _select_direct_member_partition_profile
        from engine.tests.test_pipeline import _rotating_timing_draft
        class Captured(BaseException): pass
        timing = _rotating_timing_draft()
        count = len(timing.audio_offsets)
        for effect in (False, True):
            captured = []
            def capture(**kwargs):
                captured.append(kwargs)
                raise Captured()
            with patch('crosscut.pipeline.select_balanced_direct_member_sources', side_effect=capture):
                with self.assertRaises(Captured):
                    _select_direct_member_partition_profile(timing_draft=timing,
                        eligible_indices=tuple(range(count)),
                        eligible_sources=tuple(Path(f'{index}.mp4') for index in range(count)),
                        source_timewarps=None, scene_cut_times=((),)*count,
                        source_cycle=tuple(range(count)),
                        eligible_source_frame_sizes=((2160,3840),)*count,
                        continuous_official_portrait_fancam_contract=True,
                        cinematic_bridge_enabled=effect, descriptor_provider=Provider())
            self.assertEqual(len(captured), 1)
            self.assertTrue(captured[0]['demand_driven_observations'])
            self.assertIs(captured[0]['cinematic_bridge_enabled'], effect)


if __name__ == '__main__': unittest.main()
