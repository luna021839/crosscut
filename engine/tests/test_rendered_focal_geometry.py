from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np
from crosscut.direct_member_rendered_continuity import evaluate_direct_member_rendered_continuity
from crosscut.direct_member_target_face import FocalFaceCandidate
from crosscut.focal_selector_geometry import FocalSelectorGeometry
from crosscut.rendered_focal_geometry import RenderedFocalGeometryProvider
from crosscut.rendered_focal_geometry import measure_face_owned_pose
from crosscut.body_pose_2d import BodyPose2DEvidence
from engine.tests.test_focal_selector_geometry import BOX,person,descriptor


class Provider:
    def __call__(self,t):return descriptor(0.)
    def focal_face_candidates(self,t):return (FocalFaceCandidate(BOX,(1.,0.)),)


PLAN={'fps':10.,'output_frame_count':8,'segments':[{'source_index':0,'duration':.4},
    {'source_index':1,'duration':.4}],'transitions':[{'kind':'hard_cut','frame_count':0}]}


class RenderedFocalGeometryTests(unittest.TestCase):
    def evidence(self,shift=0.,reason='observed'):
        points=((.4,.3),(.6,.3),(.3,.4),(.7,.4),(.3,.5),(.7,.5),(.4,.6),(.6,.6))
        return BodyPose2DEvidence(reason,person().bbox_xywh,(100,100),
            tuple((x+shift,y,.9) for x,y in points),True)

    def test_duplicate_rois_agreeing_on_target_torso_are_one_pose_not_two_people(self):
        pose=Mock();pose.describe.side_effect=(self.evidence(),self.evidence(.005))
        result=measure_face_owned_pose(np.zeros((100,100,3),np.uint8),
            FocalFaceCandidate(BOX,(1.,0.)),(person(),person()),pose)
        self.assertIsNotNone(result)
        self.assertIn(result[0].upper_body_xyc,(self.evidence().upper_body_xyc,self.evidence(.005).upper_body_xyc))

    def test_competing_skeletons_cannot_be_resolved_by_higher_detector_confidence(self):
        pose=Mock();pose.describe.side_effect=(self.evidence(),self.evidence(.15))
        self.assertIsNone(measure_face_owned_pose(np.zeros((100,100,3),np.uint8),
            FocalFaceCandidate(BOX,(1.,0.)),(person(),person()),pose))

    def test_unmeasured_competitor_does_not_let_available_skeleton_win(self):
        pose=Mock();pose.describe.side_effect=(self.evidence(),self.evidence(reason='pose_model_unavailable'))
        self.assertIsNone(measure_face_owned_pose(np.zeros((100,100,3),np.uint8),
            FocalFaceCandidate(BOX,(1.,0.)),(person(),person()),pose))

    def test_missing_target_pose_cannot_pass_with_matching_background_pose(self):
        result=evaluate_direct_member_rendered_continuity(Path('synthetic'),PLAN,
            descriptor_provider=Provider(),focal_geometry_provider=lambda t,c:
                FocalSelectorGeometry(0,t,c.bbox_xywh,pose_backend='rtmpose_body7'))
        self.assertIn('rendered_actual_pose_coverage_below_floor',result.failure_reasons)
        self.assertIsNone(result.boundaries[0]['pose_mismatch'])
        self.assertEqual(result.metrics['pose_measurement_basis'],'source_referenced_output_rtmpose_body7')

    def test_actual_target_pose_is_measured_in_its_own_basis(self):
        result=evaluate_direct_member_rendered_continuity(Path('synthetic'),PLAN,
            descriptor_provider=Provider(),focal_geometry_provider=lambda t,c:
                FocalSelectorGeometry(0,t,c.bbox_xywh,person=person(.4 if t>=.4 else 0.),pose_backend='rtmpose_body7'))
        self.assertGreater(result.boundaries[0]['pose_mismatch'],0.)

    def test_two_bodies_containing_face_never_choose_available_skeleton(self):
        p=RenderedFocalGeometryProvider.__new__(RenderedFocalGeometryProvider)
        p._cache={};p.frame_provider=lambda t:np.zeros((100,100,3),np.uint8)
        p.detector=Mock();p.pose=Mock()
        with patch('crosscut.rendered_focal_geometry._formation_people_descriptor',return_value=((person(),person(.2)),None)):
            measured=p(1.,FocalFaceCandidate(BOX,(1.,0.)))
        self.assertIsNone(measured.person)
        self.assertIsNotNone(p.pose.describe.call_args)


if __name__=='__main__':unittest.main()
