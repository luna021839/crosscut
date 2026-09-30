from dataclasses import replace
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.direct_member_bilateral_transform import (
    DirectMemberBilateralTransformRepair, find_bilateral_boundary_transform_repair,
)
from crosscut.direct_member_transform import (
    direct_member_source_frame_geometry, find_resolution_bound_boundary_transform_repair,
)
from crosscut.direct_member_visual import (
    DirectMemberVisualConfig, _boundary_mismatch, _boundary_sample_with_focal_geometry,
    _cut_adjacent_motion_evidence,
)
from crosscut.direct_member_editorial_motion import (
    DirectMemberEditorialMotionConfig, DirectMemberFaceSample,
    build_direct_member_editorial_motion_plan,
)
from crosscut.focal_selector_geometry import FocalSelectorGeometry, FOCAL_SELECTOR_FACE_BASIS
from crosscut.render import RenderTransition
from engine.tests.test_direct_member_transform import arguments
from engine.tests import test_direct_member_editorial_motion as dense_fixtures
from engine.tests.test_focal_selector_geometry import descriptor, person


def values():
    result = arguments()
    result.update(outgoing_face_box=(.30, .30, .10, .12), incoming_face_box=(.54, .30, .10, .12),
                  outgoing_subject_box=(.25, .25, .30, .50), incoming_subject_box=(.45, .25, .30, .50),
                  incoming_source_geometry=direct_member_source_frame_geometry((2160, 3840)),
                  minimum_subject_visible_fraction=.72,
                  alignment_geometry_basis=FOCAL_SELECTOR_FACE_BASIS, maximum_upscale_ratio=1.15)
    return result


def bilateral(**overrides):
    args = values()
    args.update(maximum_residual_attention_position_mismatch=.085,
                maximum_residual_attention_scale_mismatch=.065, **overrides)
    return find_bilateral_boundary_transform_repair(**args)


class BilateralMemberTransformTests(unittest.TestCase):
    def test_both_cameras_resolve_a_pair_that_neither_one_sided_camera_can(self):
        self.assertIsNone(find_resolution_bound_boundary_transform_repair(**values()))
        result = bilateral()
        self.assertIsInstance(result, DirectMemberBilateralTransformRepair)
        self.assertLess(result.residual_attention_position_mismatch, .01)
        self.assertLess(result.residual_attention_scale_mismatch, .01)
        self.assertGreater(result.improvement_fraction, .99)
        self.assertEqual(result.upscale_ratio, 1.)
        for camera in result.cameras:
            self.assertGreater(camera.zoom, 1.)
            self.assertGreaterEqual(camera.subject_visible_fraction, .72)
        payload = result.to_dict()
        self.assertTrue(payload['paired_atomic_publication_required'])
        self.assertTrue(payload['dense_every_output_frame_revalidation_required'])
        self.assertFalse(payload['rendered_quality_approved'])
        self.assertEqual(set(payload['cameras']), {'incoming', 'outgoing'})
        self.assertEqual(pickle.loads(pickle.dumps(result)), result)

    def test_hd_pair_uses_explicit_bounded_resample_not_lossless_claim(self):
        hd = direct_member_source_frame_geometry((1080, 1920))
        args = dict(outgoing_face_box=(.37, .3, .1, .12), incoming_face_box=(.53, .3, .1, .12),
                    outgoing_source_geometry=hd, incoming_source_geometry=hd)
        self.assertIsNone(bilateral(**args, maximum_upscale_ratio=1.))
        result = bilateral(**args)
        self.assertIsNotNone(result)
        self.assertGreater(result.upscale_ratio, 1.)
        self.assertLessEqual(result.upscale_ratio, 1.15 + 1e-9)
        self.assertFalse(result.to_dict()['lossless'])

    def test_missing_or_clipped_subject_on_either_side_cannot_fund_wider_crop(self):
        for side in ('outgoing', 'incoming'):
            with self.subTest(side=side):
                self.assertIsNone(bilateral(**{side + '_subject_box': None}))
                self.assertIsNone(bilateral(**{side + '_subject_box': (0., 0., 1., 1.)},
                                            minimum_subject_visible_fraction=.98))

    def test_unverified_identity_source_capacity_and_original_gates_remain(self):
        for overrides in (
            dict(alignment_geometry_basis='unambiguous_dominant_face_bbox_geometry_only'),
            dict(incoming_source_index=0), dict(maximum_boundary_center_distance=.10),
            dict(maximum_center_shift=.01), dict(maximum_zoom=1.05),
            dict(incoming_source_geometry=direct_member_source_frame_geometry((720, 1280))),
            dict(incoming_face_box=(.54, .3, .20, .12)),
        ):
            with self.subTest(overrides=overrides):
                self.assertIsNone(bilateral(**overrides))

    def test_camera_receipt_rejects_wrong_pixels_or_fake_motion_repair(self):
        result = bilateral()
        for overrides in (dict(source_index=True), dict(zoom=float('nan')), dict(zoom=3.),
                          dict(center_x=0.), dict(maximum_upscale_ratio=1.16)):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                replace(result.incoming, **overrides)
        with self.assertRaises(ValueError):
            result.with_repaired_severe_flags(('cross_source_pose_mismatch',))
        with self.assertRaises(ValueError):
            replace(result, incoming=replace(result.incoming, source_index=0))

    def _sample(self, source, seconds, *, pose_delta=0.):
        args = values()
        box = args['outgoing_face_box' if source == 0 else 'incoming_face_box']
        body = args['outgoing_subject_box' if source == 0 else 'incoming_subject_box']
        observed = FocalSelectorGeometry(source, seconds, box, .035,
                    replace(person(pose_delta), bbox_xywh=body, centroid_xy=(body[0]+body[2]/2, body[1]+body[3]/2)))
        return _boundary_sample_with_focal_geometry(master_seconds=seconds, source_local_seconds=seconds,
            source_index=source, descriptor=descriptor(),
            provider=SimpleNamespace(focal_geometry_at=lambda *args: observed))

    def test_selector_projects_all_four_motion_frames_through_their_own_camera(self):
        config = DirectMemberVisualConfig()
        dt = config.sample_frame_step_seconds
        samples = (self._sample(0, 1.-dt), self._sample(0, 1.),
                   self._sample(1, 1.+dt), self._sample(1, 1.+2*dt))
        geometry = values()['outgoing_source_geometry']
        with patch('crosscut.direct_member_visual._cut_adjacent_motion_evidence', wraps=_cut_adjacent_motion_evidence) as motion:
            result = _boundary_mismatch(samples[1], samples[2], config,
                outgoing_previous=samples[0], incoming_next=samples[3],
                outgoing_source_index=0, incoming_source_index=1,
                outgoing_source_geometry=geometry, incoming_source_geometry=geometry,
                continuous_official_portrait_fancam_contract=True)
        repair = result.transform_repair
        self.assertIsInstance(repair, DirectMemberBilateralTransformRepair)
        applied = motion.call_args.kwargs
        for name, index, camera in (('outgoing_previous', 0, repair.outgoing),
                ('outgoing_endpoint', 1, repair.outgoing), ('incoming_start', 2, repair.incoming),
                ('incoming_next', 3, repair.incoming)):
            self.assertEqual(applied[name].attention.bbox_xywh, camera.project(samples[index].attention.bbox_xywh))
        self.assertNotIn('cross_source_face_alignment_unrepaired', result.severe_flags)

    def test_camera_cannot_waive_pose_mismatch(self):
        geometry = values()['outgoing_source_geometry']
        result = _boundary_mismatch(self._sample(0, 1.), self._sample(1, 1., pose_delta=.5),
            DirectMemberVisualConfig(), outgoing_source_index=0, incoming_source_index=1,
            outgoing_source_geometry=geometry, incoming_source_geometry=geometry,
            continuous_official_portrait_fancam_contract=True)
        self.assertIsInstance(result.transform_repair, DirectMemberBilateralTransformRepair)
        self.assertIn('cross_source_pose_mismatch', result.severe_flags)


class BilateralDensePublicationTests(unittest.TestCase):
    def setUp(self):
        self.repair = bilateral()
        self.boxes = (self.repair.outgoing_box, self.repair.incoming_box)
        fixtures = dense_fixtures.DirectMemberEditorialMotionTests
        self.selection = fixtures._selection((0, 1), self.boxes, boundary_overrides={0: dict(
            outgoing_sample=fixtures._audited_sample(self.boxes[0], basis=FOCAL_SELECTOR_FACE_BASIS),
            incoming_sample=fixtures._audited_sample(self.boxes[1], basis=FOCAL_SELECTOR_FACE_BASIS),
            mismatch=SimpleNamespace(transform_repair=self.repair))})
        self.selection.source_frame_geometries = tuple(c.source_geometry for c in self.repair.cameras)

    def sample(self, source, seconds):
        body = values()['outgoing_subject_box' if source == 0 else 'incoming_subject_box']
        return DirectMemberFaceSample(self.boxes[source], 1, 'yunet', body, .95,
                    'face_associated_formation_person_box', FOCAL_SELECTOR_FACE_BASIS)

    def build(self, provider=None, durations=(1.5, 1.5)):
        return build_direct_member_editorial_motion_plan(visual_selection=self.selection,
            render_segments=dense_fixtures.DirectMemberEditorialMotionTests._renders((0, 1), durations), transitions=(RenderTransition(),),
            common_master_start=0., geometry_provider=provider or self.sample, fps=30.,
            config=DirectMemberEditorialMotionConfig(allow_partial_required_boundary_matches=True))

    def test_both_curves_are_published_after_both_complete_windows_pass(self):
        calls = []
        def provider(source, seconds):
            calls.append((source, seconds))
            return self.sample(source, seconds)
        plan = self.build(provider)
        self.assertEqual(plan.accepted_boundary_indices, (0,))
        self.assertEqual(plan.boundary_match_count, 1)
        self.assertIsNotNone(plan.track_for(0))
        self.assertIsNotNone(plan.track_for(1))
        self.assertAlmostEqual(plan.track_for(0).final_keyframe.zoom, self.repair.outgoing.zoom)
        self.assertAlmostEqual(plan.track_for(1).keyframes[0].zoom, self.repair.incoming.zoom)
        self.assertGreater(len({t for s, t in calls if s == 0}), 20)
        self.assertGreater(len({t for s, t in calls if s == 1}), 20)

    def test_failed_incoming_interior_revokes_both_cameras_atomically(self):
        def provider(source, seconds):
            if source == 1 and 1.55 < seconds < 2.25:
                return None
            return self.sample(source, seconds)
        plan = self.build(provider)
        self.assertEqual(plan.accepted_boundary_indices, ())
        self.assertIsNone(plan.track_for(0))
        self.assertIsNone(plan.track_for(1))
        self.assertEqual(plan.required_boundary_failures[0][0], 0)
        self.assertTrue(plan.required_boundary_failures[0][1])
        self.assertTrue(plan.to_dict()['required_boundary_failures'][0]['reasons'])

    def test_both_source_bindings_and_native_capacities_are_checked(self):
        for repair in (replace(self.repair, incoming=replace(self.repair.incoming, source_index=2)),
                       replace(self.repair, incoming=replace(self.repair.incoming,
                           source_geometry=direct_member_source_frame_geometry((4320, 7680))))):
            self.selection.boundaries[0].mismatch.transform_repair = repair
            with self.assertRaises(ValueError):
                self.build()

    def test_incoming_missing_tail_can_end_at_neutral_without_changing_cut(self):
        def provider(source, seconds):
            if source == 1 and 2.19 < seconds < 2.4:
                return None
            return self.sample(source, seconds)
        plan = self.build(provider)
        self.assertEqual(plan.accepted_boundary_indices, (0,))
        track = plan.track_for(1)
        self.assertIsNotNone(plan.track_for(0))
        self.assertAlmostEqual(track.keyframes[0].zoom, self.repair.incoming.zoom)
        self.assertEqual(track.evaluate(.7).zoom, 1.)
        self.assertGreaterEqual(track.keyframes[1].time_seconds, .55)
        self.assertLess(track.keyframes[1].time_seconds, .75)

    def test_short_ramp_does_not_publish_only_one_of_the_cameras(self):
        plan = self.build(durations=(.4, 1.5))
        self.assertEqual(plan.accepted_boundary_indices, ())
        self.assertIsNone(plan.track_for(0))
        self.assertIsNone(plan.track_for(1))


if __name__ == '__main__':
    unittest.main()
