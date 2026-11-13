from __future__ import annotations

import math
import random
import unittest

from crosscut.joint_boundary_provider import _matched_face_crops
from crosscut.paired_face_crop_geometry import matched_face_crops
from crosscut.render import (
    EDITORIAL_TRANSFORM_MAXIMUM_CENTER_SHIFT,
    EDITORIAL_TRANSFORM_MAXIMUM_ZOOM,
)


class PairedFaceCropGeometryTests(unittest.TestCase):
    def assertGeometry(self, boxes, crops):
        centers = []
        sizes = []
        for box, crop in zip(boxes, crops, strict=True):
            x, y, width, height = crop
            self.assertGreaterEqual(x, -1e-9)
            self.assertGreaterEqual(y, -1e-9)
            self.assertLessEqual(x + width, 1.0 + 1e-9)
            self.assertLessEqual(y + height, 1.0 + 1e-9)
            self.assertAlmostEqual(width, height)
            self.assertGreaterEqual(width, 1.0 / EDITORIAL_TRANSFORM_MAXIMUM_ZOOM)
            self.assertLessEqual(width, 1.0)
            self.assertLessEqual(math.hypot(x + width / 2 - .5, y + height / 2 - .5),
                                 EDITORIAL_TRANSFORM_MAXIMUM_CENTER_SHIFT + 1e-9)
            projected = ((box[0] - x) / width, (box[1] - y) / height,
                         box[2] / width, box[3] / height)
            self.assertGreaterEqual(min(projected[:2]), -1e-9)
            self.assertLessEqual(projected[0] + projected[2], 1.0 + 1e-9)
            self.assertLessEqual(projected[1] + projected[3], 1.0 + 1e-9)
            centers.append((projected[0] + projected[2] / 2,
                            projected[1] + projected[3] / 2))
            sizes.append(max(projected[2:]))
        self.assertAlmostEqual(centers[0][0], centers[1][0])
        self.assertAlmostEqual(centers[0][1], centers[1][1])
        raw_sizes = [max(box[2:]) for box in boxes]
        self.assertLessEqual(max(sizes) / min(sizes), max(raw_sizes) / min(raw_sizes) + 1e-9)

    def test_default_exact_mode_matches_existing_solver_on_seeded_boxes(self):
        rng = random.Random(9214)
        for _ in range(1000):
            boxes = []
            for _side in range(2):
                width, height = rng.uniform(.035, .42), rng.uniform(.07, .70)
                boxes.append((rng.uniform(0., 1. - width), rng.uniform(0., 1. - height), width, height))
            self.assertEqual(matched_face_crops(*boxes), _matched_face_crops(*boxes))

    def test_observed_oversized_pair_is_balanced_without_enlarging_large_face(self):
        boxes = ((.3133149783, .3379135132, .2100918611, .5263528329),
                 (.4109475454, .2181653623, .0984589179, .2360827411))
        self.assertIsNone(matched_face_crops(*boxes))
        crops = matched_face_crops(*boxes, allow_partial_extent_match=True)
        self.assertIsNotNone(crops)
        self.assertEqual(crops[0], (0., 0., 1., 1.))
        self.assertEqual(crops[1][2:], (.5, .5))
        self.assertGeometry(boxes, crops)
        reverse = matched_face_crops(*reversed(boxes), allow_partial_extent_match=True)
        for first, second in zip(crops, reversed(reverse), strict=True):
            for left, right in zip(first, second, strict=True):
                self.assertAlmostEqual(left, right)

    def test_partial_proposals_preserve_frame_face_and_motion_envelopes(self):
        rng = random.Random(1492)
        accepted = 0
        for _ in range(1000):
            boxes = []
            for _side in range(2):
                width, height = rng.uniform(.04, .40), rng.uniform(.08, .72)
                boxes.append((rng.uniform(0., 1. - width), rng.uniform(0., 1. - height), width, height))
            crops = matched_face_crops(*boxes, allow_partial_extent_match=True)
            if crops is not None:
                accepted += 1
                self.assertGeometry(boxes, crops)
        self.assertGreater(accepted, 100)

    def test_unreachable_opposite_corners_are_not_fabricated(self):
        boxes = ((.0, .0, .1, .1), (.9, .9, .1, .1))
        self.assertIsNone(matched_face_crops(*boxes, allow_partial_extent_match=True))

    def test_invalid_boxes_never_produce_a_crop(self):
        valid = (.4, .3, .2, .3)
        for invalid in ((-.1, .2, .2, .3), (.9, .2, .2, .3), (.4, .3, 0., .3),
                        (.4, .3, float('nan'), .3), (.4, .3, .2, float('inf'))):
            for partial in (False, True):
                self.assertIsNone(matched_face_crops(invalid, valid, allow_partial_extent_match=partial))

    def test_partial_mode_is_explicit_boolean(self):
        for value in (None, 1, 'true'):
            with self.assertRaises(TypeError):
                matched_face_crops((.4, .3, .2, .3), (.4, .3, .2, .3),
                                   allow_partial_extent_match=value)


if __name__ == '__main__':
    unittest.main()
