from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import unittest

from crosscut.group_source_span_proposal import build_group_source_span_proposal
from crosscut.group_cut_camera_proposal import assert_preserved_output_boundaries
from crosscut.render import (RenderPlan, RenderSegment, RenderTransition,
    PanCropTrack, PanCropKeyframe, NormalizedCrop, EditorialTransformTrack,
    EditorialTransformKeyframe)
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from crosscut.visual_score import VisualTimelinePlan
from engine.tests.test_boundary_subject_alignment import _choice


class SourceSpanProposalTests(unittest.TestCase):
    def fixture(self):
        plan=RenderPlan(sources=tuple(Path(f'{i}.mp4') for i in range(6)),
            segments=tuple(RenderSegment(i,2.*i,2.,2.*i) for i in range(5)),
            reference_source=0,aspect_ratio='16:9',output=Path('unused.mp4'),fps=30.,
            transitions=(RenderTransition(tier='C'),)*4)
        return dict(plan=plan,timeline=VisualTimelinePlan(tuple(
            _choice(i,i,start=2.*i,end=2.*(i+1)) for i in range(5)),1.),
            segment_index=2,source_index=5,entry_offset_frames=-2,exit_offset_frames=-4,
            audio_offsets=(0.,)*6,source_timewarps=(None,)*6,scene_cut_times=((),)*6,
            source_clocks={i:VerifiedRenderFrameClock(Fraction(30),1000,'a'*64) for i in range(6)})

    def test_source_and_both_cut_clocks_change_atomically_outer_windows_preserved(self):
        args=self.fixture()
        result=build_group_source_span_proposal(**args)
        self.assertEqual(result.plan.output_frame_count,args['plan'].output_frame_count)
        self.assertEqual(result.preserved_neighbor_boundaries,(0,3))
        self.assertEqual(result.plan.segments[2].source_index,5)
        self.assertAlmostEqual(result.plan.segments[2].master_start,4.-2/30.)
        self.assertAlmostEqual(result.plan.segments[3].master_start,6.-4/30.)
        for index in (0,4):self.assertEqual(result.plan.segments[index],args['plan'].segments[index])
        for field in ('timeline_descriptors_refreshed','camera_geometry_validated','production_selected','quality_approved'):
            self.assertFalse(result.to_dict()[field])

    def test_replacement_does_not_inherit_another_source_camera(self):
        args=self.fixture()
        middle=replace(args['plan'].segments[2],pan_crop=PanCropTrack((
            PanCropKeyframe(0.,NormalizedCrop(.1,.1,.8,.8)),)),editorial_transform=EditorialTransformTrack((
                EditorialTransformKeyframe(0.,1.1),)))
        args['plan']=replace(args['plan'],segments=(*args['plan'].segments[:2],middle,*args['plan'].segments[3:]))
        result=build_group_source_span_proposal(**args)
        self.assertIsNone(result.plan.segments[2].pan_crop)
        self.assertIsNone(result.plan.segments[2].editorial_transform)

    def test_native_cut_just_before_old_exit_is_avoided_by_coupled_exit_shift(self):
        args=self.fixture()
        args['scene_cut_times']=((),(),(),(),(),(5.95,))
        with self.assertRaisesRegex(ValueError,'broadcaster'):
            build_group_source_span_proposal(**dict(args,exit_offset_frames=0))
        result=build_group_source_span_proposal(**args)
        self.assertEqual(result.clock_timeline.choices[2].internal_scene_cut_master_seconds,())
        self.assertFalse(result.clock_timeline.choices[2].exit_scene_cut_near,
            'the excluded future broadcaster cut is not a visible outgoing jump')

    def test_invalid_domains_stale_clocks_and_unbounded_offsets_rejected(self):
        args=self.fixture()
        for change in (dict(source_index=2),dict(source_index=1),dict(source_index=True),
                       dict(entry_offset_frames=9),dict(exit_offset_frames=.5),
                       dict(audio_offsets=(0.,)),dict(source_clocks={}),dict(segment_index=0),
                       dict(timeline=replace(args['timeline'],choices=tuple(
                           replace(choice,source_index=0) for choice in args['timeline'].choices)))):
            with self.subTest(change=change),self.assertRaises(ValueError):
                build_group_source_span_proposal(**dict(args,**change))

    def test_short_replacement_source_does_not_pass_by_cloning_the_last_frame(self):
        args=self.fixture()
        args['source_clocks'][5]=VerifiedRenderFrameClock(Fraction(30),170,'b'*64)
        with self.assertRaisesRegex(ValueError,'terminal'):
            build_group_source_span_proposal(**args)

    def test_outer_window_proof_cannot_change_output_domain(self):
        args=self.fixture()
        for modified in (replace(args['plan'],fps=60.),replace(args['plan'],reference_source=1),
                         replace(args['plan'],sources=tuple(reversed(args['plan'].sources)))):
            with self.assertRaisesRegex(ValueError,'domain'):
                assert_preserved_output_boundaries(args['plan'],modified,args['source_clocks'],(0,3))


if __name__=='__main__':unittest.main()
