"""Articulated target motion is not a camera-pan or face-size measurement."""
from dataclasses import replace
import unittest

from crosscut.direct_member_visual import (
    DirectMemberVisualConfig, _boundary_mismatch, _cut_adjacent_motion_evidence,
    verified_focal_pose_motion_bridge,
)
from crosscut.direct_member_transform import direct_member_source_frame_geometry
from crosscut.focal_selector_geometry import focal_pose_shape_mismatch
from engine.tests.test_focal_selector_geometry import sample


CONFIG = DirectMemberVisualConfig()


def window(deltas=(0., .06, .12, .18)):
    result = []
    for index, delta in enumerate(deltas):
        value = sample(source=int(index >= 2),
            seconds=1. + index * CONFIG.sample_frame_step_seconds)
        body = value.focal_geometry.person
        # Only elbows/wrists articulate; the normalizing torso is fixed.
        joints = tuple((x + (delta if 2 <= j <= 5 else 0.), y, c)
            for j, (x, y, c) in enumerate(body.upper_body_keypoints_xyc))
        result.append(replace(value, focal_geometry=replace(value.focal_geometry,
            person=replace(body, upper_body_keypoints_xyc=joints))))
    return tuple(result)


def moving_camera(values):
    return tuple(replace(value, attention=replace(value.attention,
        bbox_xywh=(.2 + .15 * i, .1, .1, .1),
        centroid_xy=(.25 + .15 * i, .15))) for i, value in enumerate(values))


def evidence(values):
    return _cut_adjacent_motion_evidence(outgoing_previous=values[0],
        outgoing_endpoint=values[1], incoming_start=values[2],
        incoming_next=values[3], config=CONFIG)


class FocalArticulationTests(unittest.TestCase):
    def test_moving_arms_are_not_diluted_by_stationary_face(self):
        result = evidence(window())
        self.assertAlmostEqual(result.outgoing_pose_motion_magnitude, .03)
        self.assertEqual(result.outgoing_subject_motion_magnitude, 0.)
        self.assertTrue(result.both_moving)
        self.assertTrue(verified_focal_pose_motion_bridge(result))
        self.assertEqual(result.motion_basis, 'focal_common_joint_articulation_v1')
        self.assertTrue(result.to_dict()['subject_motion_separate_from_articulation'])

    def test_camera_pan_cannot_turn_static_pose_into_moving_choreography(self):
        result = evidence(moving_camera(window((0.,) * 4)))
        self.assertGreater(result.outgoing_subject_motion_magnitude, .02)
        self.assertEqual(result.outgoing_pose_motion_magnitude, 0.)
        self.assertFalse(result.both_moving)
        self.assertFalse(verified_focal_pose_motion_bridge(result))
        self.assertEqual(result.soft_preference_applied, 0.)

    def test_matching_camera_pan_cannot_hide_opposed_arms(self):
        result = evidence(moving_camera(window((0., .06, .06, 0.))))
        self.assertTrue(result.both_moving)
        self.assertAlmostEqual(result.subject_direction_cosine, 1.)
        self.assertAlmostEqual(result.direction_cosine_audit_only, -1.)
        self.assertFalse(verified_focal_pose_motion_bridge(result))
        self.assertEqual(result.soft_preference_applied, 0.)

    def test_matching_camera_speed_cannot_dilute_articulation_speed_jump(self):
        result = evidence(moving_camera(window((0., .06, .12, .60))))
        self.assertAlmostEqual(result.subject_motion_speed_ratio_mismatch, 0.)
        self.assertAlmostEqual(result.motion_speed_ratio_mismatch, 1.)
        self.assertFalse(verified_focal_pose_motion_bridge(result))

    def test_face_size_difference_does_not_change_target_pose_shape(self):
        values = list(window())
        for index in (2, 3):
            value = values[index]
            values[index] = replace(value, attention=replace(value.attention,
                bbox_xywh=(.35, .05, .2, .2), centroid_xy=(.45, .15),
                scale_area_proxy=value.attention.scale_area_proxy * 4))
        result = evidence(values)
        self.assertAlmostEqual(result.endpoint_pose_shape_mismatch,
            focal_pose_shape_mismatch(values[1].focal_geometry, values[2].focal_geometry))
        self.assertAlmostEqual(result.endpoint_pose_shape_mismatch,
            evidence(window()).endpoint_pose_shape_mismatch)
        native = direct_member_source_frame_geometry((1080, 1920))
        mismatch = _boundary_mismatch(values[1], values[2], CONFIG,
            outgoing_previous=values[0], incoming_next=values[3],
            outgoing_source_index=0, incoming_source_index=1,
            outgoing_source_geometry=native, incoming_source_geometry=native,
            continuous_official_portrait_fancam_contract=True)
        # Shape compatibility is NOT permission to emit an unaligned hard cut.
        self.assertIn('attention_scale', mismatch.severe_flags)
        self.assertIn('cross_source_face_alignment_unrepaired', mismatch.severe_flags)
        self.assertIsNone(mismatch.transform_repair)

    def test_bad_pose_stays_bad_despite_identical_framing(self):
        result = evidence(window((0., .06, .8, .86)))
        self.assertGreater(result.endpoint_pose_shape_mismatch,
            CONFIG.maximum_motion_endpoint_shape_mismatch)
        self.assertFalse(verified_focal_pose_motion_bridge(result))

    def test_missing_frame_cannot_borrow_camera_or_dominant_pose(self):
        values = list(moving_camera(window()))
        focal = values[0].focal_geometry
        values[0] = replace(values[0], focal_geometry=replace(focal,
            person=replace(focal.person, upper_body_keypoints_xyc=())))
        result = evidence(values)
        self.assertFalse(result.both_moving)
        self.assertIsNone(result.outgoing_motion_magnitude)
        self.assertIsNone(result.incoming_motion_magnitude)
        self.assertFalse(verified_focal_pose_motion_bridge(result))

    def test_disjoint_window_joints_cannot_authorize_pose_motion(self):
        values = list(window())
        for i in (0, 3):
            focal = values[i].focal_geometry
            joints = tuple((x, y, 1. if (j < 4) == (i == 0) else 0.)
                for j, (x, y, _) in enumerate(focal.person.upper_body_keypoints_xyc))
            values[i] = replace(values[i], focal_geometry=replace(focal,
                person=replace(focal.person, upper_body_keypoints_xyc=joints)))
        self.assertFalse(evidence(values).both_moving)

    def test_nonadjacent_window_does_not_authorize_motion(self):
        values = list(window())
        values[0] = replace(values[0], master_seconds=.5)
        self.assertFalse(evidence(values).both_moving)

    def test_nonfocal_descriptor_route_keeps_aggregate_policy(self):
        values = tuple(replace(v, focal_geometry=None) for v in moving_camera(window()))
        result = evidence(values)
        self.assertEqual(result.motion_basis, 'aggregate_pose_silhouette_subject_v1')
        self.assertFalse(result.to_dict()['subject_motion_separate_from_articulation'])
        self.assertAlmostEqual(result.outgoing_motion_magnitude,
            (result.outgoing_pose_motion_magnitude + result.outgoing_silhouette_motion_magnitude
             + result.outgoing_subject_motion_magnitude) / 3)


if __name__ == '__main__':
    unittest.main()
