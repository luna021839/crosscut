from dataclasses import replace
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from crosscut.boundary_subject_alignment import (BoundarySubjectAlignmentDecision,
    BoundarySubjectAlignmentPlan,BoundarySubjectAlignmentSegment,GLOBAL_ACTION_BASIS,
    _TransformState,_track_for_segment,_project_box)
from crosscut.shot_context_refinement import attach_shot_context_proposal
from crosscut.post_render_qa import _bound_local_alignment_tracks
import engine.tests.test_shot_context_camera as camera_tests
from engine.tests.test_shot_context_binding import motion_for
from crosscut.camera_replacement_preservation import validate_camera_replacement_preservation
from crosscut.render import EditorialTransformTrack,EditorialTransformKeyframe


class CameraReplacementPreservationTests(unittest.TestCase):
    def fixture(self):
        fixture=camera_tests.ShotContextProposalTests();args=fixture.fixture();plan=args['plan']
        state=_TransformState(1.1,.5,.5,0.)
        track=_track_for_segment(last_frame_time=29/30.,ramp_frames=14,fps=30.,entry=(0,state),exit=None)
        plan=replace(plan,segments=(plan.segments[0],replace(plan.segments[1],editorial_transform=track),
            plan.segments[2]))
        active=sum(abs(track.evaluate(n/30.).zoom-1.)>1e-8 for n in range(30))
        alignment=BoundarySubjectAlignmentPlan((BoundarySubjectAlignmentDecision(0,0,1,0,1,
            29/30.,1.,True,None,subject_kind='face',transformed_side='incoming',incoming_state=state,
            validated_output_frame_count=active),),
            tuple(BoundarySubjectAlignmentSegment(i,i,track if i==1 else None,(0,) if i==1 else ())
                for i in range(3)),(),14,analysis_basis=GLOBAL_ACTION_BASIS,global_action_framing=True)
        motion=replace(motion_for(plan),boundary_subject_alignment_plan=alignment)
        self.assertEqual(_bound_local_alignment_tracks(motion.to_dict(),json.loads(json.dumps(plan.to_json()))),({1},{0}))
        proposal=fixture.run_proposal(dict(args,plan=plan))
        return plan,motion,proposal

    def test_actual_attach_cannot_trade_existing_head_arm_camera_for_head_only_receipt(self):
        plan,motion,proposal=self.fixture()
        original=plan.to_json()
        # Both head-only framings improve; existing arms do not fit its zoom.
        arm_envelope=(.1,.3,.8,.55)
        old=plan.segments[1].editorial_transform.evaluate(0.)
        new=proposal.plan.segments[1].editorial_transform.evaluate(0.)
        self.assertGreaterEqual(_project_box(arm_envelope,_TransformState(old.zoom,
            old.center_x,old.center_y,old.roll_degrees))[1],.995)
        self.assertLess(_project_box(arm_envelope,_TransformState(new.zoom,
            new.center_x,new.center_y,new.roll_degrees))[1],.995)
        with self.assertRaisesRegex(ValueError,'prior camera.*obligation'):
            attach_shot_context_proposal(plan,motion,proposal)
        self.assertEqual(motion.shot_context_cameras,())
        self.assertEqual(plan.to_json(),original)

    def test_known_head_motion_window_is_preserved_not_replaced_by_face_scale_score(self):
        plan,motion,_=self.fixture();old=plan.segments[1].editorial_transform
        # Same zoom but one changed pan sample in the old incoming motion window.
        keys=tuple(replace(old.evaluate(n/30.),time_seconds=n/30.,
            center_x=.502 if n==2 else old.evaluate(n/30.).center_x) for n in range(30))
        after=replace(plan,segments=(plan.segments[0],replace(plan.segments[1],
            editorial_transform=EditorialTransformTrack(keys,sampling_canvas_zoom=old.effective_sampling_canvas_zoom)),
            plan.segments[2]))
        with self.assertRaisesRegex(ValueError,'prior camera.*motion obligation'):
            validate_camera_replacement_preservation(plan,after,motion)

    def test_owned_normal_context_stage_retains_whole_baseline_and_rejection_reason(self):
        from crosscut.pipeline import _review_first_group_bundle,_review_first_group_shot_context_bundle
        from crosscut.section_director import SectionPlan,SectionDirective,SectionRole
        from crosscut.visual_score import VisualTimelinePlan
        from crosscut.timeline import ShotType
        from engine.tests.test_boundary_subject_alignment import _choice
        plan,motion,proposal=self.fixture()
        timeline=VisualTimelinePlan(tuple(replace(_choice(i,i,start=float(i),end=float(i+1)),
            shot_type=ShotType.CLOSE_UP,entry_shot_type=ShotType.CLOSE_UP,exit_shot_type=ShotType.CLOSE_UP,
            shot_scale=.8,entry_shot_scale=.8,exit_shot_scale=.8) for i in range(3)),1.)
        section=SectionPlan((SectionDirective(0,SectionRole.BUILD,0.,3.,.5,.5,1,4,15.,.4,.1,False,False),))
        request=SimpleNamespace(sources=plan.sources,reference_source=0,aspect_ratio=plan.aspect_ratio)
        bundle=_review_first_group_bundle(timeline=timeline,request=request,eligible_indices=(0,1,2),
            offsets=(0.,)*3,timewarps=(None,)*3,common_start=0.,section_plan=section,
            phrase_times=(0.,1.,2.,3.),output=plan.output,reporter=SimpleNamespace(log=lambda *a,**k:None),warnings=[])
        bundle=replace(bundle,render_plan=plan,editorial_motion_plan=motion,
            group_transition_plan=replace(bundle.group_transition_plan,transitions=plan.transitions))
        with patch('crosscut.shot_context_refinement.propose_shot_context_camera',return_value=proposal):
            result=_review_first_group_shot_context_bundle(bundle,scene_cut_times=((),)*3)
        self.assertEqual(result.render_plan,bundle.render_plan)
        self.assertEqual(result.visual_timeline,bundle.visual_timeline)
        self.assertEqual(result.stage_edit_exact_shot_clock,bundle.stage_edit_exact_shot_clock)
        self.assertEqual(result.editorial_motion_plan,bundle.editorial_motion_plan)
        self.assertEqual(result.candidate_lineage[-1]['changed_segments'],[])
        self.assertIn('prior camera subject or motion obligation',result.candidate_lineage[-1]['attempts'][0]['reason'])

    def test_real_unchanged_windows_are_allowed_without_synthetic_arm_head_scores(self):
        plan,motion,_=self.fixture()
        receipt=validate_camera_replacement_preservation(plan,plan,motion)
        self.assertEqual(receipt['preserved_obligations'],[])
        self.assertFalse(receipt['new_motion_observations'])
        neutral=motion_for(plan)
        self.assertEqual(validate_camera_replacement_preservation(plan,self.fixture()[2].plan,neutral)
            ['preserved_obligations'],[])

    def test_camera_can_change_outside_previously_required_windows(self):
        plan,motion,_=self.fixture();old=plan.segments[1].editorial_transform
        keys=tuple(replace(old.evaluate(n/30.),time_seconds=n/30.,zoom=1.04 if n==20 else old.evaluate(n/30.).zoom)
            for n in range(30))
        after=replace(plan,segments=(plan.segments[0],replace(plan.segments[1],
            editorial_transform=EditorialTransformTrack(keys,sampling_canvas_zoom=old.effective_sampling_canvas_zoom)),
            plan.segments[2]))
        receipt=validate_camera_replacement_preservation(plan,after,motion)
        self.assertEqual(receipt['preserved_obligations'][0]['policies'][0]['protected_subject'],'head_and_observed_arms')
        self.assertNotIn(20,receipt['preserved_obligations'][0]['preserved_local_output_frames'])
        self.assertFalse(receipt['missing_observations_imputed'])

    def test_peak_sampling_canvas_clock_or_subject_kind_change_cannot_hide_under_same_indices(self):
        plan,motion,_=self.fixture();old=plan.segments[1].editorial_transform
        for after in (replace(plan,segments=(plan.segments[0],replace(plan.segments[1],
                editorial_transform=replace(old,sampling_canvas_zoom=1.2)),plan.segments[2])),
                replace(plan,master_audio_start=.1),
                replace(plan,segments=(plan.segments[0],replace(plan.segments[1],source_start=1.01),plan.segments[2]))):
            with self.subTest(after=after),self.assertRaisesRegex(ValueError,'prior camera'):
                validate_camera_replacement_preservation(plan,after,motion)


if __name__=='__main__':unittest.main()
