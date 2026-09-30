from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from crosscut.body_pose_2d import BodyPose2DEvidence
from crosscut.boundary_subject_alignment import (
    ACTION_EXPERIMENT_BASIS, BoundarySubjectObservation, BoundarySubjectAlignmentConfig,
    _action_envelope, _measured_action_points, _paired_framing_policies,
    _subject, _track_action_points, build_boundary_subject_alignment_plan,
)
from crosscut.group_focal_attention import GroupFocalSubject, GLOBAL_FOCAL_BASIS
from crosscut.pose import FormationPersonDescriptor
from crosscut.render import RenderTransition
from crosscut.visual_score import VisualTimelinePlan
from engine.tests.test_boundary_subject_alignment import _choice, _frame


def observation(*, small=True):
    face=(.45,.28,.10,.16) if small else (.41,.22,.18,.28)
    x,y,w,h=face
    points=((x+.3*w,y+.3*h),(x+.7*w,y+.3*h),(x+.5*w,y+.5*h),
        (x+.3*w,y+.8*h),(x+.7*w,y+.8*h))
    box=(.26,.10,.48,.90)
    person=FormationPersonDescriptor(box,(.5,.55),.9)
    focal=GroupFocalSubject(face,points,0.,h*.42,person,1,GLOBAL_FOCAL_BASIS)
    pose=BodyPose2DEvidence('observed',box,(854,480),
        ((.45,.48,.9),(.55,.48,.9),(.40,.55,.9),(.60,.55,.9),
         (.42,.37,.9),(.58,.37,.9),(.45,.75,.9),(.55,.75,.9)),True)
    return BoundarySubjectObservation(face_count=1,face_box_xywh=face,
        face_landmarks_xy=points,face_head_roll_radians=0.,face_eye_scale=h*.42,
        detected_person_count=1,formation_boxes_xywh=(box,),
        face_is_unambiguous_dominant=True,selected_focal_subject=focal,action_pose=pose)


class ActionFramingTests(unittest.TestCase):
    def fixture(self):
        return dict(visual_timeline=VisualTimelinePlan((
            _choice(0,0,start=0.,end=2.),_choice(1,1,start=2.,end=4.)),1.),
            sources=(Path('a.mp4'),Path('b.mp4')),transitions=(RenderTransition(tier='C'),),
            fps=30.,source_resolutions=((3840,2160),)*2,
            frame_provider=lambda i,t:_frame(i),
            observation_provider=lambda f:observation(small=float(f.mean())>=75),
            local_focal_attention=True,action_framing_model_path=Path('unit-injected-pose.onnx'))

    def test_close_medium_intent_preserves_head_and_arms_not_feet(self):
        left,right=observation(small=False),observation()
        self.assertEqual(_paired_framing_policies(left,right),('close_face','action_medium_face'))
        subject=_subject(right,preferred_kind='face',settings=BoundarySubjectAlignmentConfig(),
            framing_policy='action_medium_face')
        envelope=subject.containment_boxes[0]
        self.assertLess(envelope[1]+envelope[3],.75)
        for x,y in _measured_action_points(right):
            self.assertTrue(envelope[0]<x<envelope[0]+envelope[2])
            self.assertTrue(envelope[1]<y<envelope[1]+envelope[3])
        self.assertNotEqual(subject.layout_boxes,subject.containment_boxes)
        # No global reinterpretation of two medium or wide shots.
        self.assertEqual(_paired_framing_policies(right,right),('medium_focus','medium_focus'))

    def test_one_unknown_or_offscreen_joint_or_face_anchor_cannot_grant_action_crop(self):
        good=observation()
        for confidence,x,anchor in ((.299,.42,True),(.9,-.001,True),(.9,1.,True),(.9,.42,False)):
            points=list(good.action_pose.upper_body_xyc)
            points[4]=(x,.37,confidence)
            broken=replace(good,action_pose=replace(good.action_pose,
                upper_body_xyc=tuple(points),face_anchor_consistent=anchor))
            self.assertIsNone(_measured_action_points(broken))
            self.assertIsNone(_subject(broken,preferred_kind='face',
                settings=BoundarySubjectAlignmentConfig(),framing_policy='action_medium_face'))
        with self.assertRaisesRegex(ValueError,'face-owned'):
            replace(good,action_pose=replace(good.action_pose,bbox_xywh=(.1,.1,.2,.2)))

    def test_real_builder_keeps_experiment_separate_and_checks_dense_window(self):
        args=self.fixture()
        result=build_boundary_subject_alignment_plan(**args)
        self.assertTrue(result.decisions[0].eligible,result.decisions[0])
        self.assertEqual(result.analysis_basis,ACTION_EXPERIMENT_BASIS)
        self.assertGreater(result.validated_output_frame_count,0)
        self.assertFalse(result.to_dict()['action_framing_experiment']['production_binding_supported'])
        with self.assertRaisesRegex(ValueError,'local-focal'):
            build_boundary_subject_alignment_plan(**dict(args,local_focal_attention=False))
        ordinary=build_boundary_subject_alignment_plan(**dict(args,action_framing_model_path=None))
        self.assertFalse(ordinary.decisions[0].eligible,'injected arms must not become ordinary framing proof')
        good=observation()
        broken=replace(good,action_pose=replace(good.action_pose,face_anchor_consistent=False))
        with patch('crosscut.boundary_subject_alignment._optical_flow_box',return_value=None):
            failed=build_boundary_subject_alignment_plan(**dict(args,
                frame_provider=lambda i,t:_frame(i,110 if i==1 and 2.05<t<2.35 else None),
                observation_provider=lambda f:broken if float(f.mean())>105 else
                    observation(small=float(f.mean())>=75)))
        self.assertFalse(failed.decisions[0].eligible,'an endpoint alone cannot authorize the whole ramp')

    def test_hand_margin_is_isotropic_and_tracking_requires_all_six_joints(self):
        frame=np.random.default_rng(3).integers(0,255,(240,426,3),dtype=np.uint8)
        shifted=cv2.warpAffine(frame,np.float32([[1,0,2],[0,1,1]]),(426,240))
        points=((.4,.3),(.6,.3),(.35,.4),(.65,.4),(.4,.5),(.6,.5))
        tracked=_track_action_points(frame,shifted,points)
        self.assertIsNotNone(tracked)
        for p,q in zip(points,tracked):
            self.assertAlmostEqual(q[0]-p[0],2/426,delta=.002)
            self.assertAlmostEqual(q[1]-p[1],1/240,delta=.002)
        self.assertIsNone(_track_action_points(frame,np.zeros_like(frame),points))
        self.assertIsNone(_track_action_points(frame,shifted,points[:5]))
        box=_action_envelope((.45,.1,.10,.16),points)
        self.assertAlmostEqual(box[0],.35-.4*.16/(16/9))
        self.assertAlmostEqual(box[1]+box[3],.5+.4*.16)

    def test_dense_arm_recovery_can_use_measured_face_landmarks_without_rigid_face_box(self):
        args=self.fixture()
        good=observation()
        bad=replace(good,action_pose=replace(good.action_pose,face_anchor_consistent=False))
        args.update(frame_provider=lambda i,t:_frame(i,110 if i==1 and 2.025<t<2.05 else None),
            observation_provider=lambda f:bad if float(f.mean())>105 else
                observation(small=float(f.mean())>=75))
        # Exactly one detector miss; the six independent joint tracks and the
        # independently detected head remain available. A torso affine alone
        # is never substituted for articulated arms.
        with patch('crosscut.boundary_subject_alignment._optical_flow_box',return_value=None), \
                patch('crosscut.boundary_subject_alignment.focal_landmarks_track',return_value=True), \
                patch('crosscut.boundary_subject_alignment._track_action_points',
                    return_value=_measured_action_points(good)) as tracks:
            recovered=build_boundary_subject_alignment_plan(**args)
        self.assertTrue(recovered.decisions[0].eligible)
        self.assertGreater(recovered.optical_flow_containment_frame_count,0)
        self.assertGreater(tracks.call_count,0)
        with patch('crosscut.boundary_subject_alignment._optical_flow_box',return_value=None), \
                patch('crosscut.boundary_subject_alignment.focal_landmarks_track',return_value=False), \
                patch('crosscut.boundary_subject_alignment._track_action_points',return_value=_measured_action_points(good)):
            rejected=build_boundary_subject_alignment_plan(**args)
        self.assertFalse(rejected.decisions[0].eligible,'joint motion must not override lost head ownership')


if __name__=='__main__':
    unittest.main()
