from dataclasses import replace
import unittest

import cv2
import numpy as np

from crosscut.group_joint_native_joint_flow import native_joint_pixel_flows, native_joint_assignment_cost
from engine.tests.test_group_joint_formation import person


class NativeJointFlowTests(unittest.TestCase):
    def fixture(self):
        subject = person(4, .5)
        width, height = 1000, 800
        previous = np.zeros((height, width, 3), np.uint8)
        current = previous.copy()
        random = np.random.default_rng(143)
        moved = []
        for index, (x, y, confidence) in enumerate(subject.upper_body_xyc):
            xx, yy = int(round(x*width)), int(round(y*height))
            tile = random.integers(10, 240, (17, 17, 3), dtype=np.uint8)
            dx = 4 if index in (0, 1, 6, 7) else 8 if index in (2, 4) else -7
            previous[yy-8:yy+9, xx-8:xx+9] = tile
            current[yy-8:yy+9, xx+dx-8:xx+dx+9] = tile
            moved.append((x+dx/width, y, confidence))
        return previous, current, subject, replace(subject, observation_id=9, upper_body_xyc=tuple(moved))

    def test_real_lk_keeps_articulated_joint_paths_without_one_rigid_body_transform(self):
        previous, current, subject, moved = self.fixture()
        flow = native_joint_pixel_flows(previous, current, (subject,))[0]
        self.assertIsNotNone(flow)
        rows = {row[0]: row for row in flow.joints}
        self.assertGreaterEqual(len(rows), 6)
        for index, row in rows.items():
            self.assertAlmostEqual(row[3], moved.upper_body_xyc[index][0], delta=.003)
            self.assertLessEqual(row[5], 1.25)
            self.assertGreaterEqual(row[6], 3)
        self.assertLess(native_joint_assignment_cost(flow, moved, (1000, 800)), .02)
        wrong = replace(moved, upper_body_xyc=tuple((x+.15, y, c) for x, y, c in moved.upper_body_xyc))
        self.assertIsNone(native_joint_assignment_cost(flow, wrong, (1000, 800)))
        self.assertFalse(flow.to_dict()['quality_approved'])

    def test_flat_patch_missing_and_offscreen_joint_remain_unknown_not_zero(self):
        previous, current, subject, _ = self.fixture()
        flat = np.zeros_like(previous)
        self.assertEqual(native_joint_pixel_flows(flat, flat, (subject,)), (None,))
        self.assertEqual(native_joint_pixel_flows(previous, current,
            (replace(subject, upper_body_xyc=()),)), (None,))
        missing = replace(subject, upper_body_xyc=tuple((-1., y, c) for x, y, c in subject.upper_body_xyc))
        self.assertEqual(native_joint_pixel_flows(previous, current, (missing,)), (None,))

    def test_camera_raster_change_and_torso_occlusion_cannot_certify_a_path(self):
        previous, current, subject, moved = self.fixture()
        self.assertEqual(native_joint_pixel_flows(previous, current[:400], (subject,)), (None,))
        flow = native_joint_pixel_flows(previous, current, (subject,))[0]
        occluded = replace(moved, upper_body_xyc=tuple((x, y, .1 if i in (0, 1, 6, 7) else c)
            for i, (x, y, c) in enumerate(moved.upper_body_xyc)))
        self.assertIsNone(native_joint_assignment_cost(flow, occluded, (1000, 800)))
