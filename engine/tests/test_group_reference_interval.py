from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import unittest
from unittest.mock import patch

from crosscut.group_interval_materialization import split_group_render_intervals
from crosscut.group_reference_interval import compare_group_reference_interval
from crosscut.group_source_domain import GroupSourceDomain
from crosscut.render import (BoundaryLookKeyframe, BoundaryLookTrack, EditorialTransformKeyframe,
    EditorialTransformTrack, RenderPlan, RenderSegment, SourceLookAdjustment)
from crosscut.render_frame_schedule import VerifiedRenderFrameClock


class GroupReferenceIntervalTests(unittest.TestCase):
    def fixture(self):
        sources = (Path('reference-a.mp4'), Path('reference-b.mp4'))
        clock = VerifiedRenderFrameClock(Fraction(60), 8000, 'a'*64)
        camera = EditorialTransformTrack((EditorialTransformKeyframe(0., 1.18, .48, .49),
            EditorialTransformKeyframe(.4, 1.3, .52, .5), EditorialTransformKeyframe(1., 1.1, .5, .5)),
            sampling_canvas_zoom=1.4)
        look = BoundaryLookTrack((BoundaryLookKeyframe(0., .02, 1.02, 1.03),
            BoundaryLookKeyframe(.4, -.01, .99, .99), BoundaryLookKeyframe(1.)))
        plan = RenderPlan(sources, (RenderSegment(0, .0143, 1., 0.),
            RenderSegment(1, .02, 1., 1., editorial_transform=camera,
                look_adjustment=SourceLookAdjustment(brightness=.02), boundary_look_track=look),
            RenderSegment(0, 2.0143, 1., 2.)), 0, '16:9', Path('unused-reference-comparison.mp4'), fps=30.)
        domain = GroupSourceDomain(sources, (0, 1), (0., 0.), (None, None), ((), ()), (clock, clock))
        return plan, domain

    def compare(self, plan, candidate, domain, intervals=((30, 60),), **kwargs):
        return compare_group_reference_interval(reference=plan, candidate=candidate,
            reference_source_clocks=domain.clocks_by_request,
            candidate_source_clocks=domain.clocks_by_request, output_intervals=intervals, **kwargs)

    def test_identical_parameters_ignore_output_name_and_do_not_read_files_or_approve_quality(self):
        plan, domain = self.fixture()
        with patch.object(Path, 'open', side_effect=AssertionError('no media reads allowed')):
            result = self.compare(plan, replace(plan, output=Path('another-output.mp4')), domain)
        self.assertTrue(result.planned_parameters_preserved)
        self.assertEqual(result.compared_output_frame_count, 30)
        self.assertFalse(result.to_dict()['encoded_pixel_identity_verified'])
        self.assertFalse(result.to_dict()['quality_approved'])
        self.assertFalse(result.to_dict()['automatic_selection_gate'])

    def test_epoch_splits_preserve_reference_ticks_camera_canvas_and_boundary_look(self):
        plan, domain = self.fixture()
        split = split_group_render_intervals(plan=plan, domain=domain, split_output_frames=(45,))
        self.assertNotEqual(len(split.plan.segments), len(plan.segments))
        result = self.compare(plan, split.plan, domain)
        self.assertTrue(result.planned_parameters_preserved)
        self.assertEqual(result.changes, ())

    def test_camera_and_grade_changes_inside_reference_are_reported_but_not_automatically_rejected(self):
        plan, domain = self.fixture()
        middle = replace(plan.segments[1], editorial_transform=None,
            look_adjustment=SourceLookAdjustment(brightness=-.01))
        candidate = replace(plan, segments=(plan.segments[0], middle, plan.segments[2]))
        result = self.compare(plan, candidate, domain)
        self.assertEqual(len(result.changes), 30)
        self.assertTrue(all(c.framing_parameters_changed and c.look_parameters_changed for c in result.changes))
        self.assertTrue(all(not c.source_path_changed and not c.native_sample_changed for c in result.changes))
        self.assertFalse(result.planned_parameters_preserved)
        outside = self.compare(plan, candidate, domain, ((0, 30), (60, 90)))
        self.assertTrue(outside.planned_parameters_preserved)

    def test_source_path_clock_and_native_frame_changes_cannot_hide_behind_same_indices(self):
        plan, domain = self.fixture()
        changed = replace(plan, sources=(plan.sources[0], Path('different-b.mp4')))
        result = self.compare(plan, changed, domain)
        self.assertTrue(all(c.source_path_changed for c in result.changes))
        shifted = replace(plan.segments[1], source_start=.05)
        result = self.compare(plan, replace(plan, segments=(plan.segments[0], shifted, plan.segments[2])), domain)
        self.assertTrue(all(c.native_sample_changed for c in result.changes))
        new_clocks = dict(domain.clocks_by_request)
        new_clocks[1] = replace(new_clocks[1], packet_grid_sha256='b'*64)
        result = compare_group_reference_interval(reference=plan, candidate=plan,
            reference_source_clocks=domain.clocks_by_request, candidate_source_clocks=new_clocks,
            output_intervals=((30, 60),))
        self.assertTrue(all(c.source_clock_evidence_changed for c in result.changes))

    def test_audio_and_sampling_canvas_changes_are_not_dismissed_as_reason_string_changes(self):
        plan, domain = self.fixture()
        audio = self.compare(plan, replace(plan, master_audio_start=.5), domain)
        self.assertFalse(audio.audio_reference_parameters_equal)
        self.assertFalse(audio.planned_parameters_preserved)
        self.assertEqual(audio.changes, ())
        camera = replace(plan.segments[1].editorial_transform, sampling_canvas_zoom=1.5)
        candidate = replace(plan, segments=(plan.segments[0],
            replace(plan.segments[1], editorial_transform=camera), plan.segments[2]))
        result = self.compare(plan, candidate, domain)
        self.assertTrue(all(c.framing_parameters_changed for c in result.changes))

    def test_invalid_grid_ranges_clocks_and_frame_budget_fail_before_comparison(self):
        plan, domain = self.fixture()
        for intervals in ((), ((True, 15),), ((30, 60), (45, 75)), ((90, 91),)):
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                self.compare(plan, plan, domain, intervals)
        with self.assertRaisesRegex(ValueError, 'frame budget'):
            self.compare(plan, plan, domain, maximum_output_frames=29)
        with patch('crosscut.group_reference_interval.native_frame_clocks_for_output') as sampler:
            with self.assertRaisesRegex(ValueError, 'complete-plan sampler'):
                self.compare(plan, plan, domain, maximum_complete_plan_frames=89)
            sampler.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'output grid'):
            self.compare(plan, replace(plan, fps=29.97), domain)
        with self.assertRaisesRegex(ValueError, 'verified native clock'):
            compare_group_reference_interval(reference=plan, candidate=plan,
                reference_source_clocks={}, candidate_source_clocks=domain.clocks_by_request,
                output_intervals=((30, 60),))
