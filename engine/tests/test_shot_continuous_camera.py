from dataclasses import replace
from pathlib import Path
import unittest
import math

from crosscut.boundary_subject_alignment import (
    BoundarySubjectAlignmentConfig,SHOT_CAMERA_EXPERIMENT_BASIS,ANALYSIS_BASIS,
    _TransformState,_track_for_segment,build_boundary_subject_alignment_plan,
    _paired_framing_policies,
    _follow_face_camera_path,_project_box,_close_face_head_envelope,
)
from crosscut.render import RenderTransition,EditorialTransformTrack,EditorialTransformKeyframe
from crosscut.visual_score import VisualTimelinePlan
from engine.tests.test_boundary_subject_alignment import _choice,_frame,_observations
from engine.tests.test_action_framing import observation as action_observation


class ShotContinuousCameraTests(unittest.TestCase):
    def test_curve_preserves_both_endpoints_without_artificial_neutral_middle(self):
        incoming=_TransformState(1.4,.49,.51,0.)
        outgoing=_TransformState(1.25,.52,.48,0.)
        for end in (outgoing,None):
            track=_track_for_segment(last_frame_time=2.,ramp_frames=14,fps=30.,
                entry=(0,incoming),exit=(1,end) if end else None,settle_policy='shot_continuous')
            self.assertEqual(len(track.keyframes),2)
            self.assertEqual(track.evaluate(0.).zoom,1.4)
            self.assertEqual(track.evaluate(2.).zoom,end.zoom if end else 1.)
            self.assertGreater(track.evaluate(1.).zoom,1.19)
        old=_track_for_segment(last_frame_time=2.,ramp_frames=14,fps=30.,
            entry=(0,incoming),exit=(1,outgoing))
        self.assertEqual(old.evaluate(1.).zoom,1.,'ordinary behavior is not silently changed')

    def fixture(self):
        return dict(visual_timeline=VisualTimelinePlan((
            _choice(0,0,start=0.,end=2.),_choice(1,1,start=2.,end=4.)),1.),
            sources=(Path('a.mp4'),Path('b.mp4')),transitions=(RenderTransition(tier='C'),),
            fps=30.,frame_provider=lambda i,t:_frame(i),
            observation_provider=_observations(source0_box=(.38,.25,.24,.36),source1_box=(.42,.31,.16,.24)),
            source_resolutions=((3840,2160),)*2)

    def test_every_active_frame_is_validated_not_only_fourteen_frame_window(self):
        args=self.fixture()
        regular=build_boundary_subject_alignment_plan(**args)
        complete=build_boundary_subject_alignment_plan(**args,
            config=BoundarySubjectAlignmentConfig(settle_policy='shot_continuous'))
        self.assertTrue(complete.decisions[0].eligible)
        self.assertEqual(complete.analysis_basis,SHOT_CAMERA_EXPERIMENT_BASIS)
        self.assertEqual(complete.settle_policy,'shot_continuous')
        self.assertEqual(complete.to_dict()['policy']['native_cut_scope'],'complete_transformed_segment')
        with self.assertRaisesRegex(ValueError,'production authority'):
            replace(complete,analysis_basis=ANALYSIS_BASIS)
        self.assertGreater(complete.validated_output_frame_count,regular.validated_output_frame_count)
        self.assertEqual(complete.decisions[0].validated_output_frame_count,59)
        provider=args['observation_provider']
        def late_clipping(frame):
            observed=provider(frame)
            return replace(observed,face_box_xywh=(.42,0.,.16,.24)) if float(frame.mean())>105 else observed
        args.update(frame_provider=lambda i,t:_frame(i,110 if i==1 and 2.8<t<3.3 else None),
            observation_provider=late_clipping)
        self.assertTrue(build_boundary_subject_alignment_plan(**args).decisions[0].eligible)
        bad=build_boundary_subject_alignment_plan(**args,
            config=BoundarySubjectAlignmentConfig(settle_policy='shot_continuous'))
        self.assertFalse(bad.decisions[0].eligible,'late-frame clipping cannot evade full-shot proof')

    def test_internal_camera_cut_outside_old_ramp_is_now_inside_transformed_scope(self):
        args=self.fixture()
        left,right=args['visual_timeline'].choices
        args['visual_timeline']=replace(args['visual_timeline'],choices=(left,
            replace(right,internal_scene_cut_master_seconds=(3.,))))
        self.assertTrue(build_boundary_subject_alignment_plan(**args).decisions[0].eligible)
        whole=build_boundary_subject_alignment_plan(**args,
            config=BoundarySubjectAlignmentConfig(settle_policy='shot_continuous'))
        self.assertFalse(whole.decisions[0].eligible)
        self.assertEqual(whole.decisions[0].rejection_reason,'broadcaster_native_internal_cut')

    def test_whole_shot_does_not_inherit_unreviewed_local_authority(self):
        args=self.fixture()
        config=BoundarySubjectAlignmentConfig(settle_policy='shot_continuous')
        for extra in (dict(local_focal_attention=True),dict(action_framing_model_path=Path('pose.onnx'),local_focal_attention=True),
                dict(preserved_editorial_tracks=(None,None),preserved_track_time_shifts=(0.,0.))):
            with self.assertRaisesRegex(ValueError,'shot-continuous'):
                build_boundary_subject_alignment_plan(**args,config=config,**extra)
        with self.assertRaisesRegex(ValueError,'settle policy'):
            BoundarySubjectAlignmentConfig(settle_policy='anything')

    def test_intentional_closeup_is_explicit_not_a_false_full_body_or_hand_certificate(self):
        left,right=(replace(action_observation(small=small),selected_focal_subject=None,action_pose=None)
            for small in (False,True))
        self.assertEqual(_paired_framing_policies(left,right),('close_face','medium_focus'))
        self.assertEqual(_paired_framing_policies(left,right,reframe_closeups=True),
            ('close_face','matched_close_face'))
        for absent in (replace(right,face_landmarks_xy=None),
                replace(right,formation_boxes_xywh=(),detected_person_count=0),
                replace(right,face_count=2,face_is_unambiguous_dominant=False)):
            self.assertNotIn('matched_close_face',_paired_framing_policies(left,absent,reframe_closeups=True))
        # A wide/full-formation shot is not silently reclassified as a face crop.
        wide=replace(right,face_box_xywh=(.45,.3,.04,.06),detected_person_count=5,
            formation_boxes_xywh=((.1,.2,.2,.7),(.4,.2,.2,.7),(.7,.2,.2,.7)))
        self.assertNotIn('matched_close_face',_paired_framing_policies(left,wide,reframe_closeups=True))
        args=self.fixture()
        args['observation_provider']=lambda frame:left if frame.mean()<75 else right
        report=build_boundary_subject_alignment_plan(**args,
            config=BoundarySubjectAlignmentConfig(settle_policy='shot_continuous',reframe_closeups=True))
        self.assertTrue(report.decisions[0].eligible)
        intent=report.to_dict()['shot_continuous_experiment']
        self.assertTrue(intent['closeup_body_crop_intentional'])
        self.assertFalse(intent['hand_retention_or_gesture_continuity_certified'])
        self.assertFalse(intent['production_binding_supported'])
        with self.assertRaisesRegex(ValueError,'close-up reframing'):
            BoundarySubjectAlignmentConfig(reframe_closeups=True)
        with self.assertRaisesRegex(ValueError,'close-up reframing'):
            BoundarySubjectAlignmentConfig(settle_policy='shot_continuous',reframe_closeups='yes')

    def test_nonlinear_face_path_retains_both_cut_states_and_all_head_envelopes(self):
        track=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.4),
            EditorialTransformKeyframe(1.,1.4)))
        boxes=tuple((.44,.30-.20*math.sin(math.pi*i/30),.12,.16) for i in range(31))
        result=_follow_face_camera_path(track,boxes,fps=30.,maximum_center_shift=.10)
        self.assertEqual(result.keyframes[0],track.evaluate(0.))
        self.assertEqual(result.keyframes[-1],track.evaluate(1.))
        self.assertEqual(len(result.keyframes),31)
        self.assertLess(result.evaluate(.5).center_y,.42)
        for index,box in enumerate(boxes):
            key=result.evaluate(index/30.)
            state=_TransformState(key.zoom,key.center_x,key.center_y,key.roll_degrees)
            self.assertGreaterEqual(_project_box(_close_face_head_envelope(box),state)[1],.995)
        # No interpolation through missing detections is called measured.
        with self.assertRaisesRegex(ValueError,'complete measured'):
            _follow_face_camera_path(track,(*boxes[:15],None,*boxes[16:]),fps=30.,maximum_center_shift=.10)
        with self.assertRaisesRegex(ValueError,'face-path'):
            BoundarySubjectAlignmentConfig(follow_face_path=True)

    def test_rolled_nonlinear_curve_keeps_renderer_edges_and_endpoint_states(self):
        track=EditorialTransformTrack((EditorialTransformKeyframe(0.,1.3,.48,.49,1.6),
            EditorialTransformKeyframe(1.,1.14,.49,.5,.32)))
        boxes=tuple((.44,.32-.24*math.sin(math.pi*i/30),.12,.16) for i in range(31))
        result=_follow_face_camera_path(track,boxes,fps=30.,maximum_center_shift=.10)
        self.assertEqual(result.keyframes[0],track.evaluate(0.))
        self.assertEqual(result.keyframes[-1],track.evaluate(1.))
        for index,key in enumerate(result.keyframes):
            # Construction independently checks roll-aware boundaries at every
            # key and intervening curve samples, not just axis-aligned crops.
            self.assertIsInstance(EditorialTransformKeyframe(key.time_seconds,key.zoom,
                key.center_x,key.center_y,key.roll_degrees),EditorialTransformKeyframe)
            self.assertAlmostEqual(key.zoom,track.evaluate(index/30.).zoom)
            self.assertAlmostEqual(key.roll_degrees,track.evaluate(index/30.).roll_degrees)
        for invalid in ((-.1,.2,.2,.2),(.9,.2,.2,.2),(.2,.9,.2,.2),(.2,.2,float('nan'),.2)):
            with self.assertRaisesRegex(ValueError,'complete measured'):
                _follow_face_camera_path(track,(*boxes[:15],invalid,*boxes[16:]),fps=30.,maximum_center_shift=.10)


if __name__=='__main__':unittest.main()
