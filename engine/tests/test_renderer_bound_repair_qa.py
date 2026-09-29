"""Producer -> final QA clock parity, including fractional song trim origins."""
from copy import deepcopy
import unittest

from crosscut.boundary_repair import BoundaryRepairRequest, repair_boundary
from crosscut.post_render_qa import PostRenderQAConfig, _IssueCollector, _beat_phrase_metrics


class RendererBoundRepairQATests(unittest.TestCase):
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
