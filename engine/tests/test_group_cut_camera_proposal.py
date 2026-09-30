from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.boundary_subject_alignment import _fixed_crop_at
from crosscut.group_cut_camera_proposal import (
    build_group_cut_camera_proposal,retime_base_pan,splice_boundary_camera,vars_for_state,
)
from crosscut.render import (
    EditorialTransformKeyframe as Key,EditorialTransformTrack as Track,NormalizedCrop,
    PanCropKeyframe,PanCropTrack,RenderPlan,RenderSegment,RenderTransition,
)
from crosscut.render_frame_schedule import VerifiedRenderFrameClock
from crosscut.visual_score import VisualTimelinePlan
from engine.tests.test_boundary_subject_alignment import _choice


class GroupCutCameraProposalTests(unittest.TestCase):
    def test_retimed_pan_does_not_shrink_when_widest_sample_was_trimmed(self):
        track=PanCropTrack((PanCropKeyframe(0.,NormalizedCrop(.2,.2,.6,.6)),
            PanCropKeyframe(1.,NormalizedCrop(0.,0.,1.,1.))))
        cropped=retime_base_pan(track,relative_shift=0.,duration=.5)
        self.assertEqual(cropped.fixed_crop_size,(1.,1.))
        for time in (0.,.1,.5):
            self.assertEqual(_fixed_crop_at(cropped,time),_fixed_crop_at(track,time))

    def test_pan_retime_preserves_absolute_curve_and_holds_only_new_handle(self):
        track=PanCropTrack((PanCropKeyframe(0.,NormalizedCrop(.1,.2,.7,.7)),
            PanCropKeyframe(1.,NormalizedCrop(.3,0.,.7,.7))))
        actual=retime_base_pan(track,relative_shift=-.2,duration=1.2)
        for time in (0.,.1,.2,.6,1.2):
            for a,b in zip(_fixed_crop_at(actual,time),_fixed_crop_at(track,max(0.,time-.2))):
                self.assertAlmostEqual(a,b)

    def test_new_head_joins_neutrally_and_keeps_old_tail(self):
        old=Track((Key(0.,1.3),Key(.4,1.),Key(1.,1.),Key(1.4,1.1)))
        new=Track((Key(0.,1.2),Key(.4,1.)))
        actual=splice_boundary_camera(old,new,relative_shift=-.2,frame_count=49,fps=30.,outgoing=False)
        self.assertEqual(actual.evaluate(0.).zoom,1.2)
        for frame in range(35,49):
            for a,b in zip(vars_for_state(actual.evaluate(frame/30.)),vars_for_state(old.evaluate(frame/30.-.2))):
                self.assertAlmostEqual(a,b)

    def test_no_neutral_seam_and_colliding_opposite_ramp_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'nonneutral'):
            splice_boundary_camera(Track((Key(0.,1.2),)),Track((Key(0.,1.3),)),
                relative_shift=0.,frame_count=30,fps=30.,outgoing=False)
        old=Track((Key(0.,1.2),Key(.4,1.)))
        new=Track((Key(0.,1.),Key(.2,1.),Key(.5,1.3)))
        with self.assertRaises(ValueError):
            splice_boundary_camera(old,new,relative_shift=0.,frame_count=16,fps=30.,outgoing=True)

    def fixture(self):
        timeline=VisualTimelinePlan(tuple(_choice(i,i,start=2.*i,end=2.*(i+1)) for i in range(4)),1.)
        plan=RenderPlan(sources=tuple(Path(f'{i}.mp4') for i in range(4)),
            segments=tuple(RenderSegment(i,2.*i,2.,2.*i) for i in range(4)),
            reference_source=0,aspect_ratio='16:9',output=Path('unused.mp4'),fps=30.,
            transitions=(RenderTransition(tier='C'),)*3)
        pair=(replace(plan.segments[1],duration=1.8),
              replace(plan.segments[2],source_start=3.8,master_start=3.8,duration=2.2))
        return dict(plan=plan,timeline=timeline,boundary_index=1,shifted_segments=pair,
            source_clocks={i:VerifiedRenderFrameClock(Fraction(30),3000,'a'*64) for i in range(4)},
            scene_cut_times=((),)*4)

    def alignment(self,**kwargs):
        self.boundary=kwargs['boundary_indices']
        self.aligned_timeline=kwargs['visual_timeline']
        return SimpleNamespace(decision_for=lambda _:SimpleNamespace(eligible=True),track_for=lambda _:None)

    def test_complete_proposal_preserves_both_neighbors_but_does_not_claim_fresh_descriptors(self):
        args=self.fixture()
        with patch('crosscut.group_cut_camera_proposal.build_boundary_subject_alignment_plan',side_effect=self.alignment):
            result=build_group_cut_camera_proposal(**args)
        self.assertEqual(result.preserved_neighbor_boundaries,(0,2))
        self.assertEqual(self.boundary,(1,))
        self.assertEqual(result.plan.output_frame_count,args['plan'].output_frame_count)
        self.assertEqual(result.clock_timeline.choices[2].canonical_master_start,3.8)
        self.assertIs(result.plan.segments[0],args['plan'].segments[0])
        self.assertIs(result.plan.segments[3],args['plan'].segments[3])

    def test_newly_exposed_native_cut_is_refreshed_before_geometry(self):
        args=self.fixture()
        args['scene_cut_times']=((),(),(3.9,),())
        with patch('crosscut.group_cut_camera_proposal.build_boundary_subject_alignment_plan',side_effect=self.alignment):
            build_group_cut_camera_proposal(**args)
        self.assertEqual(self.aligned_timeline.choices[2].internal_scene_cut_master_seconds,(3.9,))

    def test_source_clock_change_at_the_neighbor_is_not_hidden_by_geometry_success(self):
        args=self.fixture()
        left,right=args['shifted_segments']
        args['shifted_segments']=(replace(left,source_start=left.source_start+.1),right)
        with patch('crosscut.group_cut_camera_proposal.build_boundary_subject_alignment_plan',side_effect=self.alignment), self.assertRaisesRegex(ValueError,'native frames'):
            build_group_cut_camera_proposal(**args)

    def test_invalid_pair_or_overlapping_neighbor_is_not_a_partial_commit(self):
        args=self.fixture()
        for change in ({'boundary_index':True},{'source_index_map':(0,0,2,3)},
                       {'shifted_segments':(args['shifted_segments'][1],args['shifted_segments'][0])}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                build_group_cut_camera_proposal(**{**args,**change})
        with patch('crosscut.group_cut_camera_proposal.build_boundary_subject_alignment_plan',side_effect=self.alignment), self.assertRaisesRegex(ValueError,'native frames'):
            left,right=args['shifted_segments']
            build_group_cut_camera_proposal(**{**args,'shifted_segments':(left,replace(right,source_start=right.source_start+.1))})

    def test_search_bound_and_nonfinite_clocks_are_rejected_before_decoding(self):
        args=self.fixture()
        for seconds in (.5,.215):
            left,right=args['plan'].segments[1:3]
            pair=(replace(left,duration=2.-seconds),replace(right,source_start=4.-seconds,
                master_start=4.-seconds,duration=2.+seconds))
            with self.subTest(seconds=seconds),self.assertRaisesRegex(ValueError,'eight whole'):
                build_group_cut_camera_proposal(**{**args,'shifted_segments':pair})
        with self.assertRaises(ValueError):
            build_group_cut_camera_proposal(**{**args,'scene_cut_times':((),(),(float('nan'),),())})
        with self.assertRaises(ValueError):
            retime_base_pan(None,relative_shift=float('nan'),duration=1.)
        with self.assertRaises(ValueError):
            splice_boundary_camera(None,None,relative_shift=0.,frame_count=True,fps=30.,outgoing=True)

    def test_preexisting_plan_timeline_disagreement_is_not_repaired_implicitly(self):
        args=self.fixture()
        choices=list(args['timeline'].choices)
        choices[0]=replace(choices[0],source_local_start=.1)
        with self.assertRaisesRegex(ValueError,'clocks disagree'):
            build_group_cut_camera_proposal(**{**args,'timeline':replace(args['timeline'],choices=tuple(choices))})


if __name__=='__main__':
    unittest.main()
