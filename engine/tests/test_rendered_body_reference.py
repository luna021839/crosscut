from dataclasses import replace
import unittest
from unittest.mock import Mock
import numpy as np

from crosscut.body_pose_2d import BodyPose2DEvidence
from crosscut.rendered_body_reference import build_source_body_reference, measure_referenced_output_body
from engine.tests.test_source_first_focal_geometry import geometry


class RenderedBodyReferenceTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((100, 100, 3), np.uint8)
        self.target = (1., 0.) + (0.,)*510
        self.other = (0., 1.) + (0.,)*510
        self.encoder = Mock()
        self.encoder.fingerprint.return_value = {'model': 'test-fixed-512'}
        self.encoder.describe.return_value = (self.target,)
        self.people = (replace(geometry().person, bbox_xywh=(.2,.1,.5,.8)), geometry().person)
        points = ((.4,.3),(.6,.3),(.3,.4),(.7,.4),(.3,.5),(.7,.5),(.4,.6),(.6,.6))
        self.reference = build_source_body_reference(self.frame,
            replace(geometry(body_only=True), pose_screen_xyc=tuple((x,y,.9) for x,y in points)),
            self.encoder)
        self.evidence = BodyPose2DEvidence('observed', self.people[0].bbox_xywh,
            (100,100), tuple((x,y,.9) for x,y in points))
        self.pose = Mock()
        self.pose.describe.return_value = self.evidence

    def test_hidden_source_face_can_match_an_independently_observed_output_body(self):
        self.encoder.describe.return_value = (self.target, self.other)
        result = measure_referenced_output_body(self.frame, self.reference, self.people,
                                               self.encoder, self.pose)
        self.assertEqual(result.reason, 'observed')
        self.assertEqual(result.person.bbox_xywh, self.people[0].bbox_xywh)
        self.assertNotEqual(result.person.bbox_xywh, self.reference.source_body_bbox_xywh)
        self.assertIs(result.pose, self.evidence)
        self.assertNotIn('appearance=', repr(self.reference))
        self.assertFalse(result.audit()['face_geometry_synthesized'])
        self.assertFalse(result.audit()['production_activation_authorized'])

    def test_similar_costumes_remain_ambiguous(self):
        self.encoder.describe.return_value = (self.target, self.target)
        result = measure_referenced_output_body(self.frame, self.reference, self.people,
                                               self.encoder, self.pose)
        self.assertEqual(result.reason, 'output_target_body_ambiguous')
        self.pose.describe.assert_not_called()

    def test_missing_target_pose_never_uses_available_neighbor_pose(self):
        self.encoder.describe.return_value = (self.target, self.other)
        self.pose.describe.return_value = replace(self.evidence, reason='model_unavailable')
        result = measure_referenced_output_body(self.frame, self.reference, self.people,
                                               self.encoder, self.pose)
        self.assertEqual(result.reason, 'output_target_pose_unobserved')
        self.assertIsNone(result.pose)
        self.pose.describe.assert_called_once()

    def test_encoder_change_cannot_reuse_source_vector(self):
        self.encoder.fingerprint.return_value = {'model': 'different'}
        with self.assertRaisesRegex(ValueError, 'encoders differ'):
            measure_referenced_output_body(self.frame, self.reference, self.people,
                                            self.encoder, self.pose)

    def test_missing_output_body_cannot_copy_source_pose(self):
        result = measure_referenced_output_body(self.frame, self.reference, (), self.encoder, self.pose)
        self.assertIsNone(result.person)
        self.assertIsNone(result.pose)
        self.pose.describe.assert_not_called()

    def test_body_appearance_alone_cannot_authorize_a_neighbors_skeleton(self):
        self.encoder.describe.return_value = (self.target, self.other)
        wrong = tuple((1.-x,y,c) for x,y,c in self.evidence.upper_body_xyc)
        self.pose.describe.return_value = replace(self.evidence, upper_body_xyc=wrong)
        result = measure_referenced_output_body(self.frame, self.reference, self.people,
                                               self.encoder, self.pose)
        self.assertEqual(result.reason, 'output_pose_source_ownership_unresolved')
        self.assertIsNone(result.pose)


if __name__ == '__main__':
    unittest.main()
