import math
import unittest

import cv2
import numpy as np

from crosscut.roll_aware_face_geometry import (detect_roll_aware_faces,
    restore_face_coordinates, rotated_face_input, same_measured_face)
from crosscut.visual_score import _DetectedFace, _dominant_face, _normalized_face_metadata


def face(x=70., y=35., confidence=.95):
    return _DetectedFace((x, y, 60., 70.),
        ((x+15., y+20.), (x+45., y+20.), (x+30., y+40.),
         (x+20., y+53.), (x+40., y+53.)), confidence)


class RollAwareFaceGeometryTests(unittest.TestCase):
    def test_inverse_landmarks_return_to_native_coordinates_for_every_angle(self):
        frame = np.zeros((180, 320, 3), np.uint8)
        original = face()
        for angle in (-90., -60., -30., 0., 30., 60., 90.):
            rotated, matrix = rotated_face_input(frame, angle)
            points = cv2.transform(np.asarray([original.landmarks_xy]), matrix)[0]
            # Use the detected rectangle in the rotated coordinate system;
            # mapped landmarks, not that AABB, define eye scale and roll.
            low, high = points.min(axis=0)-5., points.max(axis=0)+5.
            detected = _DetectedFace(tuple((*low, *(high-low))), tuple(map(tuple, points)), .94)
            restored = restore_face_coordinates(detected, matrix, width=320, height=180)
            np.testing.assert_allclose(restored.landmarks_xy, original.landmarks_xy, atol=1e-8)
            _, _, roll, eye_scale, _ = _normalized_face_metadata(restored, frame_width=320, frame_height=180)
            self.assertAlmostEqual(roll, 0.)
            self.assertAlmostEqual(eye_scale, 30./180.)

    def test_roll_is_original_roll_not_rotation_of_detection_canvas(self):
        frame = np.zeros((180, 320, 3), np.uint8)
        original = face()
        turned = cv2.getRotationMatrix2D((100., 70.), -60., 1.)
        points = cv2.transform(np.asarray([original.landmarks_xy]), turned)[0]
        _, canvas = rotated_face_input(frame, 60.)
        detections = cv2.transform(points[None, :, :], canvas)[0]
        low, high = detections.min(axis=0)-5., detections.max(axis=0)+5.
        restored = restore_face_coordinates(_DetectedFace(tuple((*low, *(high-low))),
            tuple(map(tuple, detections)), .95), canvas, width=320, height=180)
        _, _, roll, eye_scale, _ = _normalized_face_metadata(restored, frame_width=320, frame_height=180)
        self.assertAlmostEqual(roll, math.pi/3.)
        self.assertAlmostEqual(eye_scale, 30./180.)

    def test_padding_and_missing_five_point_geometry_never_become_face_evidence(self):
        identity = np.asarray(((1., 0., 0.), (0., 1., 0.)))
        for invalid in (face(-80.), _DetectedFace((1., 2., 10., 20.)), face(confidence=.5)):
            self.assertIsNone(restore_face_coordinates(invalid, identity, width=320, height=180))
        frame = np.zeros((180, 320, 3), np.uint8)
        self.assertEqual(detect_roll_aware_faces(frame, detector=lambda *_: ((face(),), 'haar')), ())

    def test_duplicates_do_not_hide_a_distinct_equally_large_person(self):
        a, b = face(), face(100.)
        self.assertFalse(same_measured_face(a, b))
        self.assertTrue(same_measured_face(a, face(70.5)))
        points = a.landmarks_xy
        exchanged = _DetectedFace(a.bbox_xywh, (points[1], points[0], points[2], points[4], points[3]), .93)
        self.assertTrue(same_measured_face(a, exchanged))
        frame = np.zeros((180, 320, 3), np.uint8)
        observed = detect_roll_aware_faces(frame, angles=(0.,), detector=lambda *_: ((a, exchanged, b), 'yunet'))
        self.assertEqual(len(observed), 2)
        self.assertIsNone(_dominant_face(observed), 'deduplication is not permission to select an ambiguous person')

    def test_rotation_budget_is_finite_and_unique(self):
        frame = np.zeros((180, 320, 3), np.uint8)
        for angles in ((), (0.,)*2, (180.,), (float('nan'),), tuple(range(8))):
            with self.assertRaises(ValueError):
                detect_roll_aware_faces(frame, angles=angles)


if __name__ == '__main__':
    unittest.main()
