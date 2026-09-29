from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.pipeline import _review_first_wide_boundary_repair
from crosscut.visual_boundary_repair import VisualBoundaryRepairPlan
from crosscut.visual_score import VisualTimelinePlan
from tests.test_visual_boundary_repair import _choice


def review(issues=(), *, score=80., switches=5):
    return SimpleNamespace(passed=not issues, weighted_score=score, source_switch_count=switches,
        hard_failures=tuple(SimpleNamespace(code=code, occurrences=indices) for code,indices in issues))


class ReviewFirstWideBoundaryRepairTests(unittest.TestCase):
    def setUp(self):
        self.timeline = VisualTimelinePlan(tuple(_choice(i, i % 2) for i in range(6)), 1.)
        self.earlier = VisualBoundaryRepairPlan(self.timeline, (self.timeline,), (), 0, ())
        self.arguments = dict(sources=(Path('a'), Path('b')), audio_offsets=(0.,0.),
            scene_cut_times=((),()), source_timewarps=(None,None), beat_period_seconds=.5)

    def test_one_targeted_wide_call_retains_lineage_and_preserves_policy(self):
        choices = list(self.timeline.choices)
        choices[1] = replace(choices[1], canonical_master_end=2.15, source_local_end=2.15)
        choices[2] = replace(choices[2], canonical_master_start=2.15, source_local_start=2.15)
        later = replace(self.earlier, timeline=replace(self.timeline, choices=tuple(choices)))
        with patch('crosscut.pipeline.review_timeline', side_effect=(
                review((('pose_mismatch_exceeded',(1,)),)), review(score=90.))), \
             patch('crosscut.pipeline.repair_visual_timeline_boundaries', return_value=later) as repair:
            result = _review_first_wide_boundary_repair(self.earlier, **self.arguments)
        repair.assert_called_once()
        self.assertEqual((1,), repair.call_args.kwargs['boundary_indices'])
        self.assertEqual(8, repair.call_args.kwargs['config'].search_radius_frames)
        self.assertEqual(9, repair.call_args.kwargs['config'].interior_scene_cut_guard_frames)
        self.assertEqual(.5, repair.call_args.kwargs['beat_period_seconds'])
        self.assertEqual(result.timeline, later.timeline)
        self.assertEqual(len(result.attempt_input_timelines), 2)
        self.assertIn('single_targeted_eight_frame', result.analysis_basis)

    def test_no_decode_for_passing_or_nonlocal_failures(self):
        for before in (review(), review((('insufficient_source_diversity',()),))):
            with self.subTest(before=before), patch('crosscut.pipeline.review_timeline', return_value=before), \
                 patch('crosscut.pipeline.repair_visual_timeline_boundaries') as repair:
                self.assertIsNone(_review_first_wide_boundary_repair(self.earlier, **self.arguments))
                repair.assert_not_called()

    def test_at_most_four_failed_boundaries_in_one_call(self):
        before = review((('pose_mismatch_exceeded',(4,3,2,1,0)),))
        with patch('crosscut.pipeline.review_timeline', side_effect=(before, before)), \
             patch('crosscut.pipeline.repair_visual_timeline_boundaries', return_value=self.earlier) as repair:
            self.assertIsNone(_review_first_wide_boundary_repair(self.earlier, **self.arguments))
        self.assertEqual((0,1,2,3), repair.call_args.kwargs['boundary_indices'])
        repair.assert_called_once()

    def test_no_promotion_if_quality_regresses_or_cuts_are_lost(self):
        before = review((('pose_mismatch_exceeded',(1,)),))
        for after in (before, review(score=79.), review(switches=4),
                review((('focus_mismatch_exceeded',(1,)),),score=90.),
                review((('pose_mismatch_exceeded',(2,)),),score=90.)):
            with self.subTest(after=after), patch('crosscut.pipeline.review_timeline', side_effect=(before,after)), \
                 patch('crosscut.pipeline.repair_visual_timeline_boundaries', return_value=self.earlier):
                self.assertIsNone(_review_first_wide_boundary_repair(self.earlier, **self.arguments))

    def test_source_change_unrelated_cut_or_duration_change_is_rejected(self):
        for index, changes in ((1,dict(source_index=0)), (0,dict(canonical_master_end=1.1)),
                (5,dict(canonical_master_end=6.1))):
            choices = list(self.timeline.choices)
            choices[index] = replace(choices[index], **changes)
            later = replace(self.earlier, timeline=replace(self.timeline,choices=tuple(choices)))
            with self.subTest(changes=changes), \
                 patch('crosscut.pipeline.review_timeline', return_value=review((('pose_mismatch_exceeded',(1,)),))), \
                 patch('crosscut.pipeline.repair_visual_timeline_boundaries', return_value=later):
                with self.assertRaisesRegex(ValueError,'changed the source path or unrelated boundary'):
                    _review_first_wide_boundary_repair(self.earlier, **self.arguments)

    def test_decoder_errors_escape_without_retry(self):
        with patch('crosscut.pipeline.review_timeline', return_value=review((('pose_mismatch_exceeded',(1,)),))), \
             patch('crosscut.pipeline.repair_visual_timeline_boundaries', side_effect=OSError('SSD disconnected')) as repair:
            with self.assertRaisesRegex(OSError,'SSD disconnected'):
                _review_first_wide_boundary_repair(self.earlier, **self.arguments)
        repair.assert_called_once()
