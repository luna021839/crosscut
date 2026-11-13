from pathlib import Path
import ast
import math
import unittest

from crosscut.member_camera_search_feedback import MemberCameraSearchFeedback
from crosscut.direct_member_visual import DirectMemberVisualConfig, select_balanced_direct_member_sources
from engine.tests.test_direct_member_visual import _descriptor


class MemberCameraSearchFeedbackTests(unittest.TestCase):
    def test_cost_does_not_spread_to_other_frames_or_pairs(self):
        feedback = MemberCameraSearchFeedback(((3., 0, 1, 2.75),))
        self.assertEqual(feedback.penalty(3.+1e-10, 0, 1), 2.75)
        for cut, left, right in ((3.+1/29.97, 0, 1), (3.-1/29.97, 0, 1), (3., 1, 0)):
            self.assertEqual(feedback.penalty(cut, left, right), 0.)

    def test_only_failed_window_gains_previously_omitted_frames(self):
        feedback = MemberCameraSearchFeedback(((3.+2/29.97, 0, 1, 2.75),))
        old = (0, -2, 2, -4, 4, -6, 6)
        self.assertEqual(feedback.frame_offsets(canonical_cut=6., ordinary_offsets=old, frame_step=1/29.97), old)
        dense = feedback.frame_offsets(canonical_cut=3., ordinary_offsets=old, frame_step=1/29.97)
        self.assertEqual(set(dense), set(range(-6, 7)))
        self.assertEqual(len(dense), 13)
        self.assertEqual(dense[0], 0)

    def test_reject_malformed_feedback(self):
        for row in ((math.nan, 0, 1, 2.75), (3., True, 1, 2.75),
                    (3., 0, 0, 2.75), (3., 0, 1, -1), (3., 0, 1, math.inf)):
            with self.assertRaises(ValueError): MemberCameraSearchFeedback((row,))
        with self.assertRaises(ValueError): MemberCameraSearchFeedback(((3., 0, 1, 2.),)*2)

    def test_actual_selector_chooses_interleaved_frame_without_source_exclusion(self):
        args = dict(sources=(Path('a.mp4'), Path('b.mp4')),
            canonical_spans=((0., 3.), (3., 6.), (6., 9.), (9., 12.)),
            audio_offsets=(0., 0.), descriptor_provider=lambda *args: _descriptor(.3),
            config=DirectMemberVisualConfig(boundary_search_radius_frames=6,
                boundary_search_step_frames=2))
        baseline = select_balanced_direct_member_sources(**args)
        revised = select_balanced_direct_member_sources(**args,
            joint_camera_failed_proposals=((3., 0, 1, 2.75), (3., 1, 0, 2.75)))
        self.assertEqual(baseline.selected_boundary_master_seconds[0], 3.)
        self.assertEqual(abs(revised.boundaries[0].mismatch.boundary_search_offset_frames), 1)
        self.assertEqual(revised.boundaries[0].mismatch.boundary_search_candidate_count, 13)
        self.assertEqual(revised.selected_boundary_master_seconds[1:], baseline.selected_boundary_master_seconds[1:])
        self.assertEqual(set(revised.source_indices), {0, 1})
        self.assertFalse(any(b.mismatch.severe_flags for b in revised.boundaries))
        audit = revised.optimizer_search_diagnostics['joint_camera_candidate_feedback']
        self.assertEqual(audit['refined_boundary_indices'], [0])
        self.assertFalse(audit['search_window_expanded'])
        self.assertEqual(revised.optimizer_search_diagnostics['decoded_proxy_path_feedback']['penalized_edge_count'], 0)

    def test_pipeline_routes_camera_feedback_separately_at_every_layer(self):
        # Wiring evidence, in addition to the actual selector execution above.
        tree = ast.parse((Path(__file__).parents[1]/'crosscut/pipeline.py').read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id in (
                     '_official_target_fancam_direct_full_frame_segments',
                     '_select_direct_member_partition_profile',
                     'select_balanced_direct_member_sources')]
        for name in ('_official_target_fancam_direct_full_frame_segments',
                     '_select_direct_member_partition_profile', 'select_balanced_direct_member_sources'):
            self.assertTrue(any(call.func.id == name and any(
                kw.arg == 'joint_camera_failed_proposals' for kw in call.keywords) for call in calls), name)


if __name__ == '__main__': unittest.main()
