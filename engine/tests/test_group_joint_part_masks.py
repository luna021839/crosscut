from dataclasses import replace
import unittest

import numpy as np

from crosscut.group_joint_part_masks import NativeSemanticParts, observed_head_components, prepare_schp_native_roi


class NativePartMaskTests(unittest.TestCase):
    def fixture(self):
        labels = np.zeros((473, 473), np.uint8)
        labels[20:80, 20:80] = 2
        labels[80:130, 30:70] = 13
        return NativeSemanticParts((1000, 1000), (0., 0., 1., 1.), (0, 0, 1000, 1000),
            labels, np.full((473, 473), .9, np.float32))

    def test_real_head_labels_extend_beyond_face_without_claiming_complete_extent(self):
        parts = self.fixture()
        values = observed_head_components(parts, face_boxes_xywh=((.06, .17, .09, .11),))
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0]['face_anchor_indices'], (0,))
        self.assertLess(values[0]['predicted_native_box_xywh'][1], .17)
        self.assertFalse(values[0]['complete_extent_verified'])
        self.assertFalse(parts.to_dict()['whole_head_extent_certified'])
        self.assertFalse(parts.to_dict()['probability_is_calibrated_confidence'])
        with self.assertRaises(ValueError):
            parts.label_map[0, 0] = 13

    def test_touching_faces_and_fragmented_masks_keep_all_hypotheses(self):
        parts = self.fixture()
        values = observed_head_components(parts, face_boxes_xywh=((.06, .17, .09, .11), (.06, .18, .09, .1)))
        self.assertTrue(values[0]['multiple_face_anchors'])
        self.assertEqual(values[0]['face_anchor_indices'], (0, 1))
        labels = parts.label_map.copy()
        labels[20:130, 49:52] = 0
        fragmented = replace(parts, label_map=labels)
        values = observed_head_components(fragmented, face_boxes_xywh=((.06, .17, .09, .11),))
        self.assertEqual(len(values), 2, 'no convenient largest component selection')

    def test_roi_affine_uses_real_integer_bounds_and_publisher_rgb_normalization(self):
        pixels = np.zeros((20, 40, 3), np.uint8)
        pixels[:, :, 2] = 255
        values, bounds = prepare_schp_native_roi(pixels, (.2, .1, .5, .6))
        self.assertEqual(bounds, (8, 2, 20, 12))
        self.assertEqual(values.shape, (1, 3, 473, 473))
        self.assertAlmostEqual(float(values[0, 0, 0, 0]), (1.-.406)/.225, places=5)
        self.assertAlmostEqual(float(values[0, 2, 0, 0]), -.485/.229, places=5)
        with self.assertRaises(ValueError):
            prepare_schp_native_roi(pixels, (-.1, .1, .5, .6))

    def test_foreground_clothes_are_not_fabricated_head_and_missing_face_stays_missing(self):
        parts = self.fixture()
        labels = np.full((473, 473), 5, np.uint8)
        self.assertEqual(observed_head_components(replace(parts, label_map=labels),
            face_boxes_xywh=((.06, .17, .09, .11),)), ())
        self.assertEqual(observed_head_components(parts, face_boxes_xywh=()), ())
        with self.assertRaises(ValueError):
            replace(parts, winning_probability=np.full((473, 473), np.nan, np.float32))
        with self.assertRaises(ValueError):
            replace(parts, label_map=np.full((473, 473), 20, np.uint8))
        with self.assertRaisesRegex(ValueError, 'differs from its requested native ROI'):
            replace(parts, requested_roi_xywh=(.1, .1, .8, .8))
