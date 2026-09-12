from dataclasses import replace
import unittest
from unittest.mock import Mock

import numpy as np

from crosscut.focal_pose import face_associated_person
from crosscut.focal_roi_pose import FocalRoiPoseAnalyzer
from crosscut.pose import FormationPersonDescriptor, _PoseObservation
from crosscut.timeline import _upper_body_shape_mismatch


FACE = (.45, .15, .1, .1)
BODY = FormationPersonDescriptor((.25, .1, .5, .8), (.5, .5), .9)
IMAGE = np.zeros((200, 200, 3), np.uint8)


def observation(*, nose=(60, 28), wrist_y=85, hips_x=60, confidence=.95):
    # ROI is [40, 12, 160, 188]. All coordinates here are ROI-local.
    screen = np.zeros((33, 5))
    screen[:, 3:] = .9
    screen[0, :2] = nose
    for index, xy in {11: (40, 55), 12: (80, 55), 13: (30, 70), 14: (90, 70),
                      15: (20, wrist_y), 16: (100, wrist_y),
                      23: (hips_x - 15, 105), 24: (hips_x + 15, 105)}.items():
        screen[index, :2] = xy
    return _PoseObservation(screen, np.zeros((33, 3)), confidence)


class FocalRoiPoseTests(unittest.TestCase):
    def analyzer(self, observations):
        analyzer = FocalRoiPoseAnalyzer()
        analyzer._geometry._load_pose = Mock()
        analyzer._geometry._pose_detector = Mock()
        analyzer._geometry._pose_detector.infer.return_value = np.zeros((len(observations), 13))
        analyzer._geometry._pose_estimator = Mock()
        analyzer._geometry._pose_estimator.infer.side_effect = observations
        return analyzer

    def describe(self, analyzer, **kwargs):
        return analyzer.describe(IMAGE, face_bbox_xywh=kwargs.get('face', FACE),
                                 people=kwargs.get('people', (BODY,)))

    def test_body_association_does_not_require_preexisting_pose(self):
        self.assertIs(face_associated_person(FACE, (BODY,)), BODY)

    def test_recovers_missing_focal_pose_with_real_roi_coordinate_mapping(self):
        analyzer = self.analyzer([observation()])
        result = self.describe(analyzer)
        self.assertEqual(result.reason, 'observed')
        self.assertEqual(result.roi_xyxy, (40, 12, 160, 188))
        self.assertEqual(len(result.person.upper_body_keypoints_xyc), 8)
        self.assertEqual(result.person.bbox_xywh, BODY.bbox_xywh)
        np.testing.assert_allclose(result.screen_upper_body_xyc[0][:2], (.4, .335))
        self.assertEqual(analyzer._geometry._pose_detector.infer.call_args.args[0].shape, (176, 120, 3))
        self.assertFalse(IMAGE.any())

    def test_different_gesture_does_not_become_the_same_pose(self):
        left = self.describe(self.analyzer([observation()]))
        right = self.describe(self.analyzer([observation(wrist_y=15)]))
        self.assertGreater(_upper_body_shape_mismatch(left.person, right.person), .3)

    def test_ambiguous_body_never_runs_model(self):
        analyzer = self.analyzer([observation()])
        other = replace(BODY, bbox_xywh=(.23, .09, .55, .82))
        result = self.describe(analyzer, people=(BODY, other))
        self.assertEqual(result.reason, 'no_unique_face_body')
        analyzer._geometry._load_pose.assert_not_called()

    def test_wrong_background_nose_is_not_accepted(self):
        result = self.describe(self.analyzer([observation(nose=(10, 100))]))
        self.assertEqual(result.reason, 'no_face_associated_pose')

    def test_ambiguous_observations_do_not_select_more_confident_one(self):
        result = self.describe(self.analyzer([observation(), observation(confidence=.99)]))
        self.assertEqual(result.reason, 'ambiguous_pose')
        self.assertIsNone(result.person)

    def test_weak_pose_does_not_remove_face_association_ambiguity(self):
        bad = observation()
        bad.screen_landmarks[23, 3:] = 0
        result = self.describe(self.analyzer([bad, observation()]))
        self.assertEqual(result.reason, 'ambiguous_pose')

    def test_neighbor_torso_outside_roi_is_not_clamped_into_body(self):
        result = self.describe(self.analyzer([observation(hips_x=150)]))
        self.assertEqual(result.reason, 'unreliable_upper_body')
        self.assertIsNone(result.person)

    def test_model_unavailable_has_explicit_missing_evidence(self):
        analyzer = self.analyzer([])
        analyzer._geometry._pose_estimator = None
        result = self.describe(analyzer)
        self.assertEqual(result.reason, 'pose_model_unavailable')

    def test_bad_or_invisible_observation_has_no_pose(self):
        bad_nan = observation()
        bad_nan.screen_landmarks[11, 0] = np.nan
        invisible = observation()
        invisible.screen_landmarks[0, 3:] = 0
        bad_shape = replace(observation(), screen_landmarks=np.zeros((32, 5)))
        for bad in (bad_nan, invisible, bad_shape, None):
            with self.subTest(bad=type(bad).__name__):
                self.assertIsNone(self.describe(self.analyzer([bad])).person)

    def test_no_previous_frame_fallback_or_temporal_state(self):
        analyzer = self.analyzer([observation(), None, observation()])
        analyzer._geometry._pose_detector.infer.return_value = np.zeros((1, 13))
        first = self.describe(analyzer)
        self.assertIsNone(self.describe(analyzer).person)
        third = self.describe(analyzer)
        self.assertEqual(first, third)

    def test_invalid_images_fail_before_loading_models(self):
        analyzer = self.analyzer([])
        for image in (np.zeros((0, 0, 3), np.uint8), np.zeros((20, 20)),
                      np.zeros((20, 20, 3), np.float32)):
            with self.assertRaises(ValueError):
                analyzer.describe(image, face_bbox_xywh=FACE, people=(BODY,))
        analyzer._geometry._load_pose.assert_not_called()


if __name__ == '__main__':
    unittest.main()
