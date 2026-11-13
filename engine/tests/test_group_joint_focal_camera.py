import unittest

from crosscut.group_joint_focal_camera import (propose_focal_face_roi_camera,
    propose_endpoint_focal_face_roi_camera, retain_entry_camera_proposal, propose_partial_face_roi_camera)
from crosscut.render import EditorialTransformKeyframe, EditorialTransformTrack


class BothNeighborFocalCameraTests(unittest.TestCase):
    def test_entry_and_exit_use_different_targets_not_one_exit_scale(self):
        boxes = tuple((.42+.001*n, .25, .10, .16) for n in range(30))
        targets = ((.42, .25, .10, .16), (.405, .15, .18, .288))
        old = EditorialTransformTrack((EditorialTransformKeyframe(0., 1.25),
            EditorialTransformKeyframe(29/30., 1.25)))
        value = propose_focal_face_roi_camera(source_face_boxes=boxes,
            fixed_neighbor_face_boxes=targets, fps=30., incumbent_track=old)
        self.assertAlmostEqual(value.track.evaluate(0.).zoom, 1.)
        self.assertAlmostEqual(value.track.evaluate(29/30.).zoom, 1.8)
        self.assertTrue(all(new['face_height_log_residual'] < before['face_height_log_residual']
            for before, new in zip(value.endpoint_before, value.endpoint_after)))
        self.assertAlmostEqual(old.evaluate(0.).zoom, 1.25, msg='incumbent is not mutated')
        self.assertFalse(value.to_dict()['whole_head_extent_certified'])
        self.assertFalse(value.to_dict()['gesture_or_phase_continuity_certified'])
        self.assertFalse(value.to_dict()['production_admitted'])

    def test_every_nonlinear_native_face_roi_is_retained_and_unknown_is_not_filled(self):
        boxes = tuple((.42+.035*(n % 3), .25, .10, .16) for n in range(30))
        value = propose_focal_face_roi_camera(source_face_boxes=boxes,
            fixed_neighbor_face_boxes=((.42, .25, .10, .16), (.4, .2, .16, .256)), fps=30.)
        self.assertEqual(len(value.track.keyframes), len(boxes))
        self.assertGreaterEqual(value.minimum_observed_face_roi_visible_fraction, .995)
        for missing in (None, (-.1, .25, .10, .16), (.1, .1, float('nan'), .16)):
            with self.assertRaises(ValueError):
                propose_focal_face_roi_camera(source_face_boxes=(*boxes[:10], missing, *boxes[11:]),
                    fixed_neighbor_face_boxes=(boxes[0], boxes[-1]), fps=30.)

    def test_native_zoom_ceiling_is_an_explicit_residual_not_exact_match_or_approval(self):
        boxes = ((.45, .3, .1, .1),)*30
        target = (.3, .1, .3, .3)
        value = propose_focal_face_roi_camera(source_face_boxes=boxes,
            fixed_neighbor_face_boxes=(target, target), fps=30., maximum_zoom=1.5)
        self.assertAlmostEqual(value.track.evaluate(0.).zoom, 1.5)
        self.assertGreater(value.endpoint_after[0]['face_height_log_residual'], .6)
        self.assertFalse(value.to_dict()['quality_approved'])

    def test_endpoint_only_camera_interpolates_parameters_never_missing_face_evidence(self):
        value = propose_endpoint_focal_face_roi_camera(source_face_boxes=((.42, .25, .10, .16),)*2,
            fixed_neighbor_face_boxes=((.42, .25, .10, .16), (.405, .15, .18, .288)),
            frame_count=70, fps=30.)
        self.assertAlmostEqual(value.track.keyframes[-1].time_seconds, 69/30.)
        self.assertEqual(value.to_dict()['observed_geometry_sample_count'], 2)
        self.assertFalse(value.to_dict()['every_output_frame_face_roi_observed'])
        self.assertFalse(value.to_dict()['whole_head_extent_certified'])
        self.assertFalse(value.to_dict()['production_admitted'])
        with self.assertRaises(ValueError):
            propose_endpoint_focal_face_roi_camera(source_face_boxes=((.42, .25, .10, .16), None),
                fixed_neighbor_face_boxes=((.42, .25, .10, .16),)*2, frame_count=70, fps=30.)

    def test_native_safe_vertical_fit_is_not_blocked_by_unrelated_tenth_frame_style_cap(self):
        source = (.5102965355, .1259260248, .0860337973, .1920077006)
        target = (.3616104762, .2172166047, .1579130014, .3323868363)
        constrained = propose_endpoint_focal_face_roi_camera(source_face_boxes=(source, source),
            fixed_neighbor_face_boxes=(target, target), frame_count=70, fps=30., maximum_center_shift=.10)
        native_safe = propose_endpoint_focal_face_roi_camera(source_face_boxes=(source, source),
            fixed_neighbor_face_boxes=(target, target), frame_count=70, fps=30.)
        self.assertGreater(constrained.endpoint_after[1]['face_center_distance'], .2)
        self.assertLess(native_safe.endpoint_after[1]['face_center_distance'], 1e-8)
        self.assertLess(native_safe.endpoint_after[1]['face_height_log_residual'], 1e-8)
        key = native_safe.track.evaluate(69/30.)
        self.assertGreaterEqual(key.center_y-.5/key.zoom, 0.)
        self.assertFalse(native_safe.to_dict()['whole_head_extent_certified'])
        for invalid_fps in (0., float('nan'), True, None):
            with self.assertRaises(ValueError):
                propose_endpoint_focal_face_roi_camera(source_face_boxes=(source, source),
                    fixed_neighbor_face_boxes=(target, target), frame_count=70, fps=invalid_fps)

    def test_entry_camera_parameters_stay_fixed_not_just_first_key_or_a_pixel_identity_claim(self):
        old = EditorialTransformTrack((EditorialTransformKeyframe(0., 1.26),
            EditorialTransformKeyframe(69/30., 1.26)))
        value = propose_endpoint_focal_face_roi_camera(source_face_boxes=((.42, .25, .10, .16),)*2,
            fixed_neighbor_face_boxes=((.42, .25, .10, .16), (.405, .15, .18, .288)),
            frame_count=70, fps=30., incumbent_track=old)
        retained = retain_entry_camera_proposal(value, incumbent_track=old, fps=30.)
        for n in range(4):
            self.assertEqual(retained.track.evaluate(n/30.), old.evaluate(n/30.))
        self.assertEqual(retained.track.evaluate(69/30.), value.track.evaluate(69/30.))
        self.assertEqual(retained.endpoint_after[0], value.endpoint_before[0])
        self.assertEqual(retained.to_dict()['preserved_entry_camera_frame_count'], 4)
        self.assertTrue(retained.to_dict()['camera_parameter_retention_is_not_encoded_pixel_identity'])
        self.assertFalse(retained.to_dict()['production_admitted'])

    def test_dense_following_can_use_the_explicit_native_envelope_without_losing_cut_endpoints(self):
        source = (.5102965355, .1259260248, .0860337973, .1920077006)
        target = (.3616104762, .2172166047, .1579130014, .3323868363)
        boxes = tuple((source[0]+.002*(n % 3), *source[1:]) for n in range(70))
        endpoint = propose_endpoint_focal_face_roi_camera(source_face_boxes=(boxes[0], boxes[-1]),
            fixed_neighbor_face_boxes=(target, target), frame_count=70, fps=30., maximum_center_shift=.25)
        dense = propose_focal_face_roi_camera(source_face_boxes=boxes,
            fixed_neighbor_face_boxes=(target, target), fps=30., maximum_center_shift=.25)
        for n in (0, 69):
            self.assertEqual(endpoint.track.evaluate(n/30.), dense.track.evaluate(n/30.))
        self.assertEqual(len(dense.track.keyframes), 70)
        self.assertGreaterEqual(dense.minimum_observed_face_roi_visible_fraction, .995)
        self.assertGreater(abs(dense.track.keyframes[-1].center_y-.5), .1)
        self.assertFalse(dense.to_dict()['required_arm_visibility_certified'])
        self.assertFalse(dense.to_dict()['production_admitted'])
        with self.assertRaises(ValueError):
            propose_focal_face_roi_camera(source_face_boxes=boxes,
                fixed_neighbor_face_boxes=(target, target), fps=30., maximum_center_shift=.251)

    def test_partial_camera_authors_only_camera_gaps_and_cannot_claim_dense_observations(self):
        boxes = tuple(None if 12 <= i < 16 else (.42+.001*i, .25, .10, .16) for i in range(30))
        targets = ((.42, .25, .10, .16), (.405, .15, .18, .288))
        value = propose_partial_face_roi_camera(source_face_boxes=boxes,
            fixed_neighbor_face_boxes=targets, fps=30.)
        self.assertEqual(value.observed_geometry_sample_count, 26)
        self.assertEqual(value.unobserved_geometry_frame_indices, (12, 13, 14, 15))
        self.assertEqual(len(value.track.keyframes), 30)
        self.assertFalse(value.to_dict()['every_output_frame_face_roi_observed'])
        self.assertTrue(value.to_dict()['interpolated_camera_parameters_are_not_subject_observations'])
        self.assertFalse(value.to_dict()['production_admitted'])
        self.assertFalse(value.to_dict()['required_arm_visibility_certified'])
        self.assertEqual(boxes[12:16], (None,)*4)
        for invalid in ((None, *boxes[1:]), (*boxes[:-1], None),
                (*boxes[:12], None, None, None, None, None, *boxes[17:])):
            with self.assertRaises(ValueError):
                propose_partial_face_roi_camera(source_face_boxes=invalid,
                    fixed_neighbor_face_boxes=targets, fps=30.)
        with self.assertRaises(ValueError):
            propose_focal_face_roi_camera(source_face_boxes=boxes,
                fixed_neighbor_face_boxes=targets, fps=30.)
