from dataclasses import replace
import json
import math
import unittest

from crosscut.body_pose_2d import BodyPose2DEvidence
from crosscut.visible_arm_motion import (
    VisibleArmObservation, compare_arm_motion, visible_arm_observation,
)


def measured(dy=0., *, missing=(), transform=(1., 0., 0.)):
    points = ((.4, .3), (.6, .3), (.35, .4+dy), (.65, .4+dy),
              (.3, .5+dy), (.7, .5+dy))
    scale, tx, ty = transform
    values = tuple((x*scale+tx, y*scale+ty, .1 if i in missing else .8)
                   for i, (x, y) in enumerate(points))
    # Hips are deliberately outside the image and unreliable.
    values += ((.4, 1.2, .05), (.6, 1.2, .05))
    return BodyPose2DEvidence('observed', (0., 0., 1., 1.), (1000, 1000), values, True)


def observation(dy=0., **kwargs):
    return visible_arm_observation(measured(dy, **kwargs), minimum_joint_confidence=.3)


def window(times, direction=1., **kwargs):
    return [(t, observation(direction*t*.02, **kwargs)) for t in times]


class VisibleArmMotionTests(unittest.TestCase):
    def test_closeup_with_hidden_hips_has_a_separate_visible_arm_basis(self):
        evidence = measured()
        self.assertIsNone(evidence.as_formation_person(.8))
        actual = visible_arm_observation(evidence, minimum_joint_confidence=.3)
        self.assertIsNotNone(actual)
        self.assertEqual(len(actual.coordinates), 6)

    def test_hidden_shoulder_or_unverified_face_is_not_invented(self):
        self.assertIsNone(observation(missing=(0,)))
        for anchor in (False, None):
            self.assertIsNone(visible_arm_observation(replace(measured(), face_anchor_consistent=anchor),
                                                    minimum_joint_confidence=.3))

    def test_same_pose_with_opposite_motion_is_distinguished(self):
        left = window([-3., -2., -1., 0.])
        forward = compare_arm_motion(left, window([0., 1., 2., 3.]))
        reverse = compare_arm_motion(left, window([0., 1., 2., 3.], -1.))
        self.assertAlmostEqual(forward['endpoint_rms'], 0.)
        self.assertAlmostEqual(reverse['endpoint_rms'], 0.)
        self.assertAlmostEqual(forward['velocity_cosine'], 1.)
        self.assertAlmostEqual(reverse['velocity_cosine'], -1.)

    def test_confidence_changes_are_not_joint_movement(self):
        first = measured()
        second = replace(first, upper_body_xyc=tuple((x,y,.5) for x,y,c in first.upper_body_xyc))
        a = visible_arm_observation(first, minimum_joint_confidence=.3)
        b = visible_arm_observation(second, minimum_joint_confidence=.3)
        result = compare_arm_motion([(0.,a),(1.,b)], [(0.,b),(1.,a)])
        self.assertEqual(result['outgoing_speed'], 0.)
        self.assertEqual(result['incoming_speed'], 0.)
        self.assertIsNone(result['velocity_cosine'])

    def test_camera_translation_and_uniform_zoom_do_not_create_arm_motion(self):
        a = observation()
        b = observation(transform=(.7,.1,.1))
        result = compare_arm_motion([(0.,a),(1.,b)], [(0.,b),(1.,a)])
        self.assertLess(result['outgoing_speed'], 1e-12)
        self.assertLess(result['endpoint_rms'], 1e-12)

    def test_unequal_sample_rates_use_seconds_not_frame_differences(self):
        result = compare_arm_motion(window([0.,.1,.2,.3]), window([0.,.15,.3]))
        self.assertAlmostEqual(result['velocity_cosine'], 1.)
        self.assertAlmostEqual(result['outgoing_speed'], result['incoming_speed'])

    def test_occluded_hand_is_excluded_from_the_entire_comparison(self):
        result = compare_arm_motion(window([0.,1.], missing=(4,)), window([0.,1.]))
        self.assertTrue(result['available'])
        self.assertEqual(result['common_arm_joint_count'], 3)
        unavailable = compare_arm_motion(window([0.,1.], missing=(2,3,4)), window([0.,1.]))
        self.assertFalse(unavailable['available'])
        self.assertEqual(unavailable['reason'], 'insufficient_common_arm_joints')

    def test_missing_temporal_sample_is_not_interpolated(self):
        result = compare_arm_motion([(0., observation()), (1., None)], window([0.,1.]))
        self.assertFalse(result['available'])
        self.assertEqual(result['reason'], 'missing_observation')

    def test_invalid_observations_and_clocks_are_rejected(self):
        good = observation()
        for times in ((1.,1.), (2.,1.), (0.,math.nan), (False,1.)):
            with self.assertRaises(ValueError):
                compare_arm_motion([(t,good) for t in times], window([0.,1.]))
        invalid = VisibleArmObservation(((math.nan,0.),)*6, (True,)*6)
        with self.assertRaises(ValueError):
            compare_arm_motion([(0.,good),(1.,invalid)], window([0.,1.]))
        for confidence in (True,0.,1.1,math.inf):
            with self.assertRaises(ValueError):
                visible_arm_observation(measured(), minimum_joint_confidence=confidence)

    def test_public_evidence_contains_no_pose_vectors_or_quality_approval(self):
        result = compare_arm_motion(window([0.,1.]), window([0.,1.]))
        json.dumps(result, allow_nan=False)
        self.assertFalse(result['quality_calibrated'])
        self.assertFalse(result['final_media_authority'])
        self.assertFalse(any(isinstance(value,(list,tuple,dict)) for value in result.values()))
        self.assertNotIn('passes',result)


if __name__ == '__main__':
    unittest.main()
