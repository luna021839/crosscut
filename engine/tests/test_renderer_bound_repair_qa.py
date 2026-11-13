"""Producer -> final QA clock parity, including fractional song trim origins."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import unittest

from crosscut.boundary_repair import BoundaryRepairConfig, BoundaryRepairRequest, repair_boundary
from crosscut.post_render_qa import PostRenderQAConfig, _IssueCollector, _beat_phrase_metrics
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from crosscut.visual_boundary_repair import repair_visual_timeline_boundaries
from crosscut.visual_score import VisualTimelinePlan
from tests.test_visual_boundary_repair import _choice, _descriptor


class RendererBoundRepairQATests(unittest.TestCase):
    def wide_fixture(self):
        timeline = VisualTimelinePlan((_choice(0, 0), _choice(1, 1)), 1.)
        repaired = repair_visual_timeline_boundaries(
            visual_timeline=timeline, sources=(Path('a'), Path('b')),
            audio_offsets=(0., 0.), scene_cut_times=((), ()), output_fps=30.,
            descriptor_provider=lambda source, seconds: _descriptor(
                .85 if source == 1 and seconds < 1.16 else .5),
            config=BoundaryRepairConfig(search_radius_frames=8, interior_scene_cut_guard_frames=9),
            render_frame_clocks=(VerifiedRenderFrameClock(Fraction(30), 300, 'a'*64),)*2)
        self.assertEqual(5, repaired.repairs[0].result.selected_frame_offset)
        selected = repaired.repairs[0].result.selected_boundary_seconds
        return {'rhythm_analysis': {'beat_grid':[{'time_seconds': selected, 'strength': 1.}]},
            'timeline_review': {'boundaries':[{'boundary_index': 0, 'source_switch': True,
                'at_seconds': selected}]}, 'visual_boundary_repairs': repaired.to_dict()}, \
            {'fps': 30., 'master_audio_start': 0.}, {}

    def test_actual_wide_visual_producer_passes_final_radius_gate(self):
        inputs = self.wide_fixture()
        collector = _IssueCollector()
        metrics = _beat_phrase_metrics(*inputs, PostRenderQAConfig(), collector)
        self.assertFalse([issue.code for issue in collector.issues
                          if issue.code.startswith('boundary_repair_')])
        record = metrics['repair_radius_checks'][0]
        self.assertEqual(8, record['maximum_allowed_frames'])
        self.assertEqual(8, record['complete_renderer_bound_audited_radius'])
        self.assertFalse(record['partial_reedit_boundary'])

    def test_wide_domain_cannot_be_claimed_with_incomplete_stale_or_legacy_evidence(self):
        fixture = self.wide_fixture()
        mutations = {
            'missing': lambda r: r.pop('candidate_evidence_audit'),
            'gap': lambda r: r['candidate_evidence_audit'].pop(0),
            'duplicate': lambda r: r['candidate_evidence_audit'].append(r['candidate_evidence_audit'][-1]),
            'bool_offset': lambda r: r['candidate_evidence_audit'][8].update(frame_offset=False),
            'legacy_clock': lambda r: r['candidate_evidence_audit'][13].update(sample_clock_basis='unverified_legacy_midpoint_v1'),
            'selected_failed': lambda r: r['candidate_evidence_audit'][13].update(production_transition_gate_passed=False),
            'stale_visual': lambda r: r.update(selected_visual_evidence={'total_cost': 0.}),
            'stale_selected': lambda r: r['result']['candidates'].clear(),
            'stale_context': lambda r: r['result_edge_context'].update(boundary_index=1),
            'stale_identity': lambda r: r.update(observed_edge_context_identity='0'*64),
            'source_changed': lambda r: r.update(incoming_source_index=9),
            'radius_flag_only': lambda r: (r.pop('candidate_evidence_audit'), r.update(search_radius_frames=8)),
        }
        for name, mutation in mutations.items():
            inputs = deepcopy(fixture)
            mutation(inputs[0]['visual_boundary_repairs']['repairs'][0])
            with self.subTest(mutation=name):
                self.assertIn('boundary_repair_exceeds_allowed_radius', self.issues(*inputs))

    def fixture(self, *, offset=0, legacy=False):
        origin = .051
        result = repair_boundary(BoundaryRepairRequest(
            nominal_boundary_seconds=1.063, outgoing_segment_start_seconds=origin,
            incoming_segment_end_seconds=2.5, output_fps=30.,
            output_timeline_origin_seconds=None if legacy else origin),
            visual_costs_by_frame_offset={offset: .1})
        selected = result.selected_boundary_seconds
        analysis = {'rhythm_analysis': {'beat_grid':[{'time_seconds':selected, 'strength':1.}]},
            'timeline_review': {'boundaries':[{'boundary_index':0, 'source_switch':True,
                'at_seconds':selected}]},
            'visual_boundary_repairs': {'repairs':[{'boundary_index':0,'result':result.to_dict()}]}}
        return analysis, {'fps':30.,'master_audio_start':origin}, {}

    def issues(self, analysis, plan, selection):
        collector = _IssueCollector()
        _beat_phrase_metrics(analysis, plan, selection, PostRenderQAConfig(), collector)
        return {issue.code for issue in collector.issues}

    def test_actual_producer_snapped_zero_positive_and_negative_offsets_pass_qa(self):
        for legacy in (False, True):
            for offset in (-3, 0, 3):
                with self.subTest(legacy=legacy, offset=offset):
                    inputs = self.fixture(offset=offset, legacy=legacy)
                    self.assertNotIn('boundary_repair_payload_inconsistent', self.issues(*inputs))
                    self.assertNotIn('boundary_repair_time_mismatch', self.issues(*inputs))

    def test_initial_v2_artifacts_are_recomputed_from_render_plan_not_fabricated_origin(self):
        analysis, plan, selection = self.fixture()
        result = analysis['visual_boundary_repairs']['repairs'][0]['result']
        result.pop('output_timeline_origin_seconds')
        result.pop('search_center_boundary_seconds')
        self.assertNotIn('boundary_repair_payload_inconsistent', self.issues(analysis,plan,selection))
        plan['master_audio_start'] += .01
        self.assertIn('boundary_repair_payload_inconsistent', self.issues(analysis,plan,selection))

    def test_changed_time_grid_origin_nested_offset_and_out_of_radius_still_fail(self):
        for mutation in ('origin','center','nested','policy','selected','missing_origin','too_far'):
            analysis, plan, selection = self.fixture(offset=3 if mutation == 'too_far' else 0)
            result = analysis['visual_boundary_repairs']['repairs'][0]['result']
            if mutation == 'origin': result['output_timeline_origin_seconds'] += .001
            elif mutation == 'center': result['search_center_boundary_seconds'] += .001
            elif mutation == 'nested': result['selected']['frame_offset'] = 1
            elif mutation == 'policy': result['policy'] = 'invented_unknown_grid'
            elif mutation == 'selected':
                result['selected_boundary_seconds'] += .001
                result['selected']['boundary_seconds'] += .001
                analysis['timeline_review']['boundaries'][0]['at_seconds'] += .001
            elif mutation == 'missing_origin': plan.pop('master_audio_start')
            elif mutation == 'too_far':
                result['selected_frame_offset'] = result['selected']['frame_offset'] = 4
            with self.subTest(mutation=mutation):
                codes = self.issues(analysis,plan,selection)
                self.assertIn('boundary_repair_payload_inconsistent', codes)
                if mutation == 'too_far': self.assertIn('boundary_repair_exceeds_allowed_radius',codes)


if __name__ == '__main__': unittest.main()
