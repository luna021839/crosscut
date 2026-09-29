"""Selector transform search must cover feasible scale/position compromises."""
from dataclasses import replace
import unittest

from crosscut.direct_member_transform import (
    direct_member_source_frame_geometry,
    find_lossless_boundary_transform_repair,
    find_resolution_bound_boundary_transform_repair,
)


def arguments():
    return dict(
        outgoing_face_box=(.30, .30, .10, .12),
        incoming_face_box=(.438, .30, .112, .1344),
        outgoing_source_index=0,
        incoming_source_index=1,
        outgoing_source_geometry=direct_member_source_frame_geometry((2160, 3840)),
        incoming_source_geometry=direct_member_source_frame_geometry((1080, 1920)),
        outgoing_subject_box=(.25, .28, .30, .52),
        incoming_subject_box=(.25, .28, .30, .52),
        outgoing_subject_basis='uniquely_face_associated_focal_body',
        incoming_subject_basis='uniquely_face_associated_focal_body',
        maximum_zoom=1.5,
        maximum_center_shift=.16,
        maximum_boundary_center_distance=.30,
        maximum_residual_center_distance=.08,
        maximum_residual_attention_scale_mismatch=.30,
        minimum_improvement_fraction=.20,
        minimum_face_visible_fraction=.995,
        legacy_face_only_maximum_zoom=1.12,
        legacy_face_only_maximum_center_shift=.025,
        minimum_subject_visible_fraction=.90,
    )


class LosslessZoomFrontierTests(unittest.TestCase):
    def test_position_can_need_more_zoom_than_exact_face_height_match(self):
        repair = find_lossless_boundary_transform_repair(**arguments())
        self.assertIsNotNone(repair)
        self.assertGreater(repair.zoom, 1.12)
        self.assertLessEqual(repair.zoom, 1.5)
        self.assertEqual(repair.transformed_side, 'outgoing')
        self.assertLessEqual(repair.residual_attention_position_mismatch * .42, .08 + 1e-9)
        self.assertLessEqual(repair.residual_attention_scale_mismatch, .30 + 1e-9)
        self.assertGreaterEqual(repair.improvement_fraction, .20)
        self.assertGreaterEqual(repair.face_visible_fraction, .995)
        self.assertGreaterEqual(repair.subject_visible_fraction, .90)

    def test_previously_valid_exact_height_match_is_retained(self):
        values = arguments()
        values['incoming_face_box'] = (.32, .30, .112, .1344)
        repair = find_lossless_boundary_transform_repair(**values)
        self.assertIsNotNone(repair)
        self.assertAlmostEqual(repair.zoom, 1.12)

    def test_observed_portrait_pair_has_a_lossless_position_scale_compromise(self):
        # Scalar geometry from an actual rejected 1080p -> 4K member cut.
        # Source identities, images and face embeddings are not required.
        values = arguments()
        values.update(
            outgoing_face_box=(.2770329158, .2637490718, .1198012273, .0825377041),
            incoming_face_box=(.4010886192, .3999350961, .1007222255, .0758010963),
            outgoing_subject_box=(.1233948980, .2013550933, .8711909734, .7007469226),
            incoming_subject_box=(.3162218894, .3681653391, .6680760204, .5435098379),
            outgoing_source_geometry=direct_member_source_frame_geometry((1080, 1920)),
            incoming_source_geometry=direct_member_source_frame_geometry((2160, 3840)),
            minimum_subject_visible_fraction=.72,
            alignment_geometry_basis='verified_official_fancam_focal_face_correspondence',
        )
        repair = find_lossless_boundary_transform_repair(**values)
        self.assertIsNotNone(repair)
        self.assertEqual(repair.transformed_side, 'incoming')
        self.assertGreaterEqual(repair.subject_visible_fraction, .99)
        self.assertLessEqual(repair.residual_attention_position_mismatch * .42, .08)
        self.assertLessEqual(repair.residual_attention_scale_mismatch, .30)
        self.assertGreaterEqual(repair.improvement_fraction, .40)
        self.assertLessEqual(repair.zoom, repair.source_geometry.lossless_zoom_capacity)

    def test_missing_subject_does_not_authorize_wider_camera(self):
        values = arguments()
        values.update(outgoing_subject_box=None, outgoing_subject_basis=None)
        self.assertIsNone(find_lossless_boundary_transform_repair(**values))

    def test_native_hd_still_cannot_claim_lossless_enlargement(self):
        values = arguments()
        values['outgoing_source_geometry'] = direct_member_source_frame_geometry((1080, 1920))
        self.assertIsNone(find_lossless_boundary_transform_repair(**values))

    def test_subject_clipping_still_rejects_the_alternative(self):
        values = arguments()
        values['outgoing_subject_box'] = (0., 0., 1., 1.)
        self.assertIsNone(find_lossless_boundary_transform_repair(**values))

    def test_configured_zoom_bound_is_not_raised(self):
        values = arguments()
        values['maximum_zoom'] = 1.19
        self.assertIsNone(find_lossless_boundary_transform_repair(**values))

    def test_initial_center_gate_is_not_bypassed(self):
        values = arguments()
        values['incoming_face_box'] = (.70, .30, .112, .1344)
        self.assertIsNone(find_lossless_boundary_transform_repair(**values))


class ResolutionBoundRepairTests(unittest.TestCase):
    def values(self):
        values = arguments()
        values.update(outgoing_source_geometry=direct_member_source_frame_geometry((1080, 1920)),
                      incoming_face_box=(.34, .30, .112, .1344),
                      alignment_geometry_basis='verified_official_fancam_focal_face_correspondence')
        return values

    def test_hd_focal_resample_is_explicit_not_reported_as_lossless(self):
        values = self.values()
        self.assertIsNone(find_lossless_boundary_transform_repair(**values))
        result = find_resolution_bound_boundary_transform_repair(**values, maximum_upscale_ratio=1.15)
        self.assertAlmostEqual(result.upscale_ratio, 1.12)
        self.assertLessEqual(result.upscale_ratio, 1.15)
        self.assertEqual(result.source_geometry.lossless_zoom_capacity, 1.0)
        payload = result.to_dict()
        self.assertEqual(payload['contract'], 'official_member_bounded_resample_subject_transform_v1')
        self.assertFalse(payload['lossless'])
        self.assertFalse(payload['rendered_quality_approved'])
        self.assertTrue(payload['dense_every_output_frame_revalidation_required'])
        with self.assertRaises(ValueError):
            replace(result, maximum_upscale_ratio=1.0)

    def test_lossless_result_remains_identical_and_preferred(self):
        values = arguments()
        self.assertEqual(find_resolution_bound_boundary_transform_repair(**values, maximum_upscale_ratio=1.15),
                         find_lossless_boundary_transform_repair(**values))

    def test_explicit_strict_setting_and_unverified_identity_do_not_allow_upscale(self):
        values = self.values()
        self.assertIsNone(find_resolution_bound_boundary_transform_repair(**values, maximum_upscale_ratio=1.0))
        values['alignment_geometry_basis'] = 'unambiguous_dominant_face_bbox_geometry_only'
        self.assertIsNone(find_resolution_bound_boundary_transform_repair(**values, maximum_upscale_ratio=1.15))

    def test_existing_output_resolution_is_not_lied_about(self):
        values = self.values()
        values.update(outgoing_source_geometry=direct_member_source_frame_geometry((720, 1280)),
                      incoming_source_geometry=direct_member_source_frame_geometry((720, 1280)))
        self.assertIsNone(find_resolution_bound_boundary_transform_repair(**values, maximum_upscale_ratio=1.15))

    def test_upscale_budget_is_finite_and_capped_by_dense_policy(self):
        for value in (True, .9, 1.1501, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                find_resolution_bound_boundary_transform_repair(**self.values(), maximum_upscale_ratio=value)


if __name__ == '__main__':
    unittest.main()
