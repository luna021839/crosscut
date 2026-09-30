from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import unittest
from unittest.mock import patch

from crosscut.group_source_span_proposal import (build_group_source_span_proposal,
    refresh_group_source_span_observations)
from crosscut.group_cut_camera_proposal import (assert_preserved_output_boundaries,
    UNREVIEWED_GROUP_SOURCE_TRANSITION_REASON)
from crosscut.render import (RenderPlan, RenderSegment, RenderTransition,
    PanCropTrack, PanCropKeyframe, NormalizedCrop, EditorialTransformTrack,
    EditorialTransformKeyframe)
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from crosscut.render_boundary_clock import native_frame_clocks_for_output
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

    def test_same_source_hold_becomes_unreviewed_cross_source_cut_without_stale_scores(self):
        args=self.fixture();plan=args['plan']
        plan=replace(plan,segments=(*plan.segments[:1],replace(plan.segments[1],source_index=2),
            *plan.segments[2:]),transitions=(plan.transitions[0],RenderTransition(tier='D'),
            RenderTransition(tier='A',face_mismatch=.1,appearance_min_cosine_similarity=.99),plan.transitions[3]))
        args.update(plan=plan,timeline=replace(args['timeline'],choices=tuple(
            replace(c,source_index=s.source_index) for c,s in zip(args['timeline'].choices,plan.segments))))
        result=build_group_source_span_proposal(**args)
        neutral=RenderTransition(tier='C',reason=UNREVIEWED_GROUP_SOURCE_TRANSITION_REASON)
        self.assertEqual(result.plan.transitions,(plan.transitions[0],neutral,neutral,plan.transitions[3]))
        self.assertEqual(result.to_dict()['reset_transition_boundaries'],[1,2])
        self.assertFalse(result.to_dict()['transition_evidence_refreshed'])
        self.assertFalse(result.to_dict()['production_selected'])

    def test_multi_cell_replacement_does_not_inherit_old_internal_hold_authority(self):
        args=self.run_fixture();plan=args['plan']
        args['plan']=replace(plan,transitions=(*plan.transitions[:2],RenderTransition(tier='D'),
            *plan.transitions[3:]))
        result=build_group_source_span_proposal(**args)
        for boundary in (1,2,3):
            self.assertEqual(result.plan.transitions[boundary],RenderTransition(tier='C',
                reason=UNREVIEWED_GROUP_SOURCE_TRANSITION_REASON))
        for boundary in (0,4):self.assertEqual(result.plan.transitions[boundary],args['plan'].transitions[boundary])

    def test_native_cut_just_before_old_exit_is_avoided_by_coupled_exit_shift(self):
        args=self.fixture()
        args['scene_cut_times']=((),(),(),(),(),(5.95,))
        with self.assertRaisesRegex(ValueError,'broadcaster'):
            build_group_source_span_proposal(**dict(args,exit_offset_frames=0))
        result=build_group_source_span_proposal(**args)
        self.assertEqual(result.clock_timeline.choices[2].internal_scene_cut_master_seconds,())
        self.assertFalse(result.clock_timeline.choices[2].exit_scene_cut_near,
            'the excluded future broadcaster cut is not a visible outgoing jump')

    def test_larger_editorial_timing_window_is_explicit_and_keeps_outer_proof(self):
        args=self.fixture()
        with self.assertRaises(ValueError):
            build_group_source_span_proposal(**dict(args,entry_offset_frames=12))
        result=build_group_source_span_proposal(**dict(args,entry_offset_frames=12,
            exit_offset_frames=-12,maximum_shift_frames=18))
        self.assertEqual(result.plan.output_frame_count,args['plan'].output_frame_count)
        self.assertEqual(result.preserved_neighbor_boundaries,(0,3))
        self.assertEqual(result.to_dict()['maximum_shift_frames'],18)
        self.assertFalse(result.to_dict()['production_selected'])
        # Neither a larger search window nor a one-second cap overrides the
        # visible shot floor or the complete neighboring fourteen-frame proof.
        with self.assertRaises(ValueError):
            build_group_source_span_proposal(**dict(args,entry_offset_frames=26,
                exit_offset_frames=-26,maximum_shift_frames=30))
        for radius in (True,0,31,18.0):
            with self.assertRaises(ValueError):
                build_group_source_span_proposal(**dict(args,maximum_shift_frames=radius))
        with self.assertRaisesRegex(ValueError,'at most one second'):
            build_group_source_span_proposal(**dict(args,plan=replace(args['plan'],fps=29.97),
                maximum_shift_frames=30))

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

    def run_fixture(self):
        args=self.fixture();plan=args['plan']
        plan=replace(plan,segments=(*plan.segments[:2],RenderSegment(2,4.,.9,4.),
            RenderSegment(2,4.9,1.1,4.9),*plan.segments[3:]),
            transitions=(RenderTransition(tier='C'),)*5)
        args.update(plan=plan,last_segment_index=3,timeline=VisualTimelinePlan(tuple(
            _choice(i,s.source_index,start=s.master_start,end=s.master_start+s.duration)
            for i,s in enumerate(plan.segments)),1.))
        return args

    def test_complete_multi_cell_run_changes_both_cells_not_artificial_hold(self):
        args=self.run_fixture();result=build_group_source_span_proposal(**args)
        self.assertEqual(result.end_segment_index,3)
        self.assertEqual(result.to_dict()['contract'],'coupled_group_source_run_clock_proposal_v2')
        self.assertEqual([s.source_index for s in result.plan.segments],[0,1,5,5,3,4])
        self.assertEqual(result.preserved_neighbor_boundaries,(0,4))
        self.assertEqual(result.plan.output_frame_count,args['plan'].output_frame_count)
        self.assertEqual(result.plan.segments[3].master_start,4.9)
        self.assertIs(result.plan.segments[0],args['plan'].segments[0])
        self.assertIs(result.plan.segments[5],args['plan'].segments[5])
        for index in (2,3):
            self.assertIsNone(result.plan.segments[index].pan_crop)
            self.assertIsNone(result.plan.segments[index].editorial_transform)
        self.assertFalse(result.to_dict()['timeline_descriptors_refreshed'])
        self.assertFalse(result.to_dict()['production_selected'])

    def test_broadcaster_cut_at_artificial_cell_seam_cannot_disappear(self):
        args=self.run_fixture()
        args['scene_cut_times']=((),(),(),(),(),(4.9,))
        with self.assertRaisesRegex(ValueError,'broadcaster'):
            build_group_source_span_proposal(**args)

    def test_fragment_discontinuous_clock_and_unbounded_run_rejected(self):
        args=self.run_fixture()
        for last in (True,1,2,5,3.0):
            with self.subTest(last=last),self.assertRaises(ValueError):
                build_group_source_span_proposal(**dict(args,last_segment_index=last))
        broken=replace(args['plan'],segments=(*args['plan'].segments[:3],
            replace(args['plan'].segments[3],source_start=5.),*args['plan'].segments[4:]))
        timeline=VisualTimelinePlan(tuple(_choice(i,s.source_index,start=s.master_start,end=s.master_start+s.duration)
            for i,s in enumerate(broken.segments)),1.)
        timeline=replace(timeline,choices=tuple(replace(c,source_local_start=s.source_start,
            source_local_end=s.source_end) for c,s in zip(timeline.choices,broken.segments)))
        with self.assertRaisesRegex(ValueError,'render-source run'):
            build_group_source_span_proposal(**dict(args,plan=broken,timeline=timeline))

    def test_source_run_longer_than_three_hundred_output_frames_is_rejected(self):
        args=self.run_fixture();plan=args['plan']
        segments=(*plan.segments[:2],RenderSegment(2,4.,5.4,4.),RenderSegment(2,9.4,5.6,9.4),
            RenderSegment(3,15.,2.,15.),RenderSegment(4,17.,2.,17.))
        plan=replace(plan,segments=segments)
        timeline=VisualTimelinePlan(tuple(_choice(i,s.source_index,start=s.master_start,
            end=s.master_start+s.duration) for i,s in enumerate(segments)),1.)
        with self.assertRaisesRegex(ValueError,'bounded render-source run'):
            build_group_source_span_proposal(**dict(args,plan=plan,timeline=timeline))

    def test_full_outside_native_clock_and_same_source_outer_seam_preserved(self):
        args=self.run_fixture();plan=args['plan']
        # The fixed right neighbor and its next cell share one native clock.
        plan=replace(plan,segments=(*plan.segments[:5],RenderSegment(3,8.,2.,8.)))
        args.update(plan=plan,timeline=VisualTimelinePlan(tuple(
            _choice(i,s.source_index,start=s.master_start,end=s.master_start+s.duration)
            for i,s in enumerate(plan.segments)),1.))
        result=build_group_source_span_proposal(**args)
        self.assertEqual(result.preserved_neighbor_boundaries,(0,4))
        wanted=(*range(60),*range(240,300))
        before=native_frame_clocks_for_output(plan,args['source_clocks'],wanted)
        after=native_frame_clocks_for_output(result.plan,args['source_clocks'],wanted)
        self.assertEqual([(v.source_index,v.native_frame_index,v.terminal_clone) for v in before.values()],
            [(v.source_index,v.native_frame_index,v.terminal_clone) for v in after.values()])

    def test_expanded_interval_proof_does_not_let_other_segments_change(self):
        args=self.run_fixture();result=build_group_source_span_proposal(**args)
        altered=replace(result.plan,segments=(replace(result.plan.segments[0],source_start=.1),
            *result.plan.segments[1:]))
        with self.assertRaisesRegex(ValueError,'outside its complete interval'):
            assert_preserved_output_boundaries(args['plan'],altered,args['source_clocks'],(0,4),
                allow_same_source_seams=True,complete_interval_bounds=(1,4),reset_transition_boundaries=(1,2,3))
        for bounds in ((1,True),(4,1),(1,),[1,4],(0,6)):
            with self.subTest(bounds=bounds),self.assertRaises(ValueError):
                assert_preserved_output_boundaries(args['plan'],result.plan,args['source_clocks'],(0,4),
                    allow_same_source_seams=True,complete_interval_bounds=bounds,reset_transition_boundaries=(1,2,3))
        with self.assertRaisesRegex(ValueError,'outside-clock proof'):
            assert_preserved_output_boundaries(args['plan'],result.plan,args['source_clocks'],(0,4),
                complete_interval_bounds=(1,4),reset_transition_boundaries=(1,2,3))

    def test_transition_reset_cannot_hide_outside_changes_or_fake_new_cut_evidence(self):
        args=self.run_fixture();result=build_group_source_span_proposal(**args)
        for boundary in (0,1,4):
            transitions=list(result.plan.transitions)
            transitions[boundary]=RenderTransition(tier='C',reason='forged',face_mismatch=.01)
            altered=replace(result.plan,transitions=tuple(transitions))
            with self.subTest(boundary=boundary),self.assertRaisesRegex(ValueError,'outside transition or retained stale'):
                assert_preserved_output_boundaries(args['plan'],altered,args['source_clocks'],(0,4),
                    allow_same_source_seams=True,complete_interval_bounds=(1,4),reset_transition_boundaries=(1,2,3))
        with self.assertRaisesRegex(ValueError,'exact complete interior interval'):
            assert_preserved_output_boundaries(args['plan'],result.plan,args['source_clocks'],(0,4),
                allow_same_source_seams=True,complete_interval_bounds=(1,4),reset_transition_boundaries=(0,1,2,3))

    def test_fresh_source_window_cannot_borrow_old_source_face_cut_or_approval_metadata(self):
        from engine.tests.test_sparse_source_lattice import Sampler
        args=self.run_fixture()
        args['timeline']=replace(args['timeline'],candidate_review_summaries=({'passed':True},),choices=tuple(
            replace(c,face_match_candidate_failure='internal_camera_cut',quality_score=.99,
                pose_transition_mismatch_from_previous=.77) for c in args['timeline'].choices))
        numeric=build_group_source_span_proposal(**args)
        Sampler.instances=[];Sampler.unreadable_source=None
        with patch('crosscut.visual_score._VideoSampler',Sampler):
            fresh=refresh_group_source_span_observations(proposal=numeric,audio_offsets=args['audio_offsets'],
                source_timewarps=args['source_timewarps'],sync_scores=(1.,)*6,scene_cut_times=args['scene_cut_times'])
        self.assertEqual([len(s.times) for s in Sampler.instances],[0,5,0,5,0,10])
        self.assertTrue(all(s.closed for s in Sampler.instances))
        self.assertIn('not_a_selected_timeline',fresh.analysis_basis)
        self.assertEqual(fresh.candidate_review_summaries,())
        for index in range(1,5):
            self.assertNotEqual(fresh.choices[index].face_match_candidate_failure,'internal_camera_cut')
            self.assertAlmostEqual(fresh.choices[index].quality_score,.8)
            self.assertEqual(fresh.choices[index].source_index,numeric.plan.segments[index].source_index)
            self.assertAlmostEqual(fresh.choices[index].source_local_start,numeric.plan.segments[index].source_start)
        self.assertEqual(fresh.choices[1].pose_transition_mismatch_from_previous,.77,
            'refresh cannot erase an unchanged outside-edge failure')
        for index in (0,5):self.assertIs(fresh.choices[index],numeric.clock_timeline.choices[index])

    def test_observation_refresh_is_not_a_fallback_for_unreadable_replacement(self):
        from engine.tests.test_sparse_source_lattice import Sampler
        args=self.fixture();numeric=build_group_source_span_proposal(**args)
        Sampler.instances=[];Sampler.unreadable_source=5
        try:
            with patch('crosscut.visual_score._VideoSampler',Sampler),self.assertRaisesRegex(ValueError,'no source frame is readable'):
                refresh_group_source_span_observations(proposal=numeric,audio_offsets=args['audio_offsets'],
                    source_timewarps=args['source_timewarps'],sync_scores=(1.,)*6,scene_cut_times=args['scene_cut_times'])
        finally:Sampler.unreadable_source=None


if __name__=='__main__':unittest.main()
