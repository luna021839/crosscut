from dataclasses import replace
import json
from types import SimpleNamespace
import unittest

from crosscut.direct_member_target_face import FocalFaceCandidate
from crosscut.direct_member_visual import (
    DirectMemberVisualConfig, _boundary_mismatch, _boundary_sample_with_focal_geometry,
    _cut_adjacent_motion_evidence,
)
from crosscut.direct_member_transform import direct_member_source_frame_geometry
from crosscut.focal_selector_geometry import (
    FOCAL_SELECTOR_FACE_BASIS, FOCAL_SELECTOR_POSE_BASIS,
    FocalSelectorGeometry, common_focal_pose_vectors, focal_pose_shape_mismatch,
    focal_selector_geometry,
)
from crosscut.pose import FormationPersonDescriptor
from crosscut.timeline import ShotType
from crosscut.visual_score import FrameVisualDescriptor


BOX = (.4, .1, .1, .1)
BODY = (.3, .08, .4, .8)


def person(delta=0., confidences=(1.,)*8):
    points = tuple((-.2 + .06*j + delta, -.3 + .05*j, confidences[j]) for j in range(8))
    return FormationPersonDescriptor(BODY, (.5, .48), .95, points)


def descriptor(raw_pose=0.):
    return FrameVisualDescriptor(quality_score=1., composition_score=1., shot_scale=.6,
        shot_type=ShotType.CLOSE_UP, face_count=3, face_detector_backend='yunet',
        primary_face_bbox=(.75, .08, .2, .2),
        structure_embedding=(raw_pose,)*6, pose_embedding=(raw_pose,)*60,
        pose_backend='unrelated_dominant_pose', pose_confidence=1.,
        silhouette_embedding=(raw_pose,)*6, silhouette_confidence=1.,
        formation_people=(person(),), detected_person_count=3)


def observation(source=0, seconds=1., delta=0., missing=False):
    return FocalSelectorGeometry(source, seconds, None if missing else BOX,
        None if missing else .035, None if missing else person(delta))


def sample(source=0, seconds=1., delta=0., missing=False, raw_pose=0.):
    focal = observation(source, seconds, delta, missing)
    return _boundary_sample_with_focal_geometry(master_seconds=seconds,
        source_local_seconds=seconds, source_index=source, descriptor=descriptor(raw_pose),
        provider=SimpleNamespace(focal_geometry_at=lambda *args:focal))


class FocalSelectorGeometryTests(unittest.TestCase):
    def test_target_geometry_does_not_mutate_raw_descriptor(self):
        raw = descriptor()
        out = _boundary_sample_with_focal_geometry(master_seconds=1.,source_local_seconds=1.,
            source_index=0,descriptor=raw,provider=SimpleNamespace(focal_geometry_at=lambda *a:observation()))
        self.assertEqual(out.attention.bbox_xywh, BOX)
        self.assertEqual(out.attention.basis, FOCAL_SELECTOR_FACE_BASIS)
        self.assertEqual(raw.primary_face_bbox, (.75,.08,.2,.2))
        self.assertEqual(raw.face_count, 3)
        self.assertIs(out.descriptor,raw)

    def test_missing_focal_cannot_fall_back_to_foreground_neighbor(self):
        out = sample(missing=True)
        self.assertIsNone(out.attention)
        mismatch = _boundary_mismatch(out,sample(source=1),DirectMemberVisualConfig(),
            outgoing_source_index=0,incoming_source_index=1)
        self.assertIn('cross_source_focal_subject_unresolved',mismatch.severe_flags)
        self.assertIsNone(mismatch.cinematic_bridge)
        self.assertIsNone(mismatch.dominant_pose_mismatch)

    def test_unresolved_focal_does_not_forbid_a_same_source_hold(self):
        mismatch = _boundary_mismatch(sample(missing=True),sample(missing=True),DirectMemberVisualConfig(),
            outgoing_source_index=0,incoming_source_index=0)
        self.assertNotIn('cross_source_focal_subject_unresolved',mismatch.severe_flags)

    def test_legacy_provider_preserves_legacy_geometry(self):
        raw=descriptor()
        out=_boundary_sample_with_focal_geometry(master_seconds=1.,source_local_seconds=1.,
            source_index=0,descriptor=raw,provider=lambda *a:raw)
        self.assertIsNone(out.focal_geometry)
        self.assertEqual(out.attention.bbox_xywh,raw.primary_face_bbox)

    def test_resolver_must_report_typed_missing_state(self):
        with self.assertRaises(TypeError):
            _boundary_sample_with_focal_geometry(master_seconds=1.,source_local_seconds=1.,
                source_index=0,descriptor=descriptor(),provider=SimpleNamespace(focal_geometry_at=lambda *a:None))

    def test_wrong_source_or_timestamp_is_rejected(self):
        for focal in (observation(source=1),observation(seconds=1.1)):
            with self.subTest(focal=focal),self.assertRaises(ValueError):
                _boundary_sample_with_focal_geometry(master_seconds=1.,source_local_seconds=1.,
                    source_index=0,descriptor=descriptor(),provider=SimpleNamespace(focal_geometry_at=lambda *a:focal))

    def test_ambiguous_overlapping_body_is_not_associated(self):
        face=FocalFaceCandidate(BOX,(1.,0.))
        result=focal_selector_geometry(source_index=0,source_seconds=1.,candidate=face,
            people=(person(),replace(person(),bbox_xywh=(.25,.08,.5,.8))))
        self.assertIsNone(result.person)
        self.assertEqual(result.face_bbox_xywh,BOX)

    def test_body_without_own_face_is_invalid(self):
        with self.assertRaises(ValueError):
            FocalSelectorGeometry(0,1.,None,person=person())
        with self.assertRaises(ValueError):
            FocalSelectorGeometry(0,1.,(.01,.01,.05,.05),person=person())

    def test_scalar_provenance_does_not_serialize_pose_or_appearance_vectors(self):
        payload=observation().to_dict()
        self.assertNotIn('upper_body_keypoints_xyc',json.dumps(payload))
        self.assertFalse(payload['appearance_embeddings_persisted'])
        self.assertTrue(payload['proposal_geometry_not_final_render_authority'])

    def test_focal_pose_differs_despite_matching_background_poses(self):
        match=_boundary_mismatch(sample(),sample(source=1,delta=.5),DirectMemberVisualConfig(),
            outgoing_source_index=0,incoming_source_index=1)
        self.assertEqual(match.pose_basis,FOCAL_SELECTOR_POSE_BASIS)
        self.assertGreater(match.dominant_pose_mismatch,.5)
        self.assertIn('cross_source_pose_mismatch',match.severe_flags)
        self.assertIsNone(match.silhouette_mismatch)

    def test_unrelated_raw_pose_cannot_override_matching_focal_pose(self):
        match=_boundary_mismatch(sample(raw_pose=-1),sample(source=1,raw_pose=1),DirectMemberVisualConfig(),
            outgoing_source_index=0,incoming_source_index=1)
        self.assertEqual(match.dominant_pose_mismatch,0.)
        self.assertNotIn('cross_source_pose_mismatch',match.severe_flags)

    def test_missing_focal_joints_cannot_borrow_dominant_pose(self):
        left=sample()
        missing=replace(left.focal_geometry,person=replace(person(),upper_body_keypoints_xyc=()))
        match=_boundary_mismatch(replace(left,focal_geometry=missing),sample(source=1),DirectMemberVisualConfig(),
            outgoing_source_index=0,incoming_source_index=1)
        self.assertIsNone(match.dominant_pose_mismatch)
        self.assertIn('cross_source_actual_pose_missing',match.severe_flags)

    def test_window_uses_one_common_joint_mask(self):
        observations=(replace(observation(),person=person(confidences=(1,1,1,1,1,0,0,0))),
                      replace(observation(),person=person(confidences=(0,1,1,1,1,1,0,0))))
        vectors=common_focal_pose_vectors(observations)
        self.assertEqual(len(vectors[0]),8)
        self.assertEqual(vectors[0],vectors[1])

    def test_disjoint_masks_are_not_equal_dimension_pose_evidence(self):
        observations=(replace(observation(),person=person(confidences=(1,1,1,1,0,0,0,0))),
                      replace(observation(),person=person(confidences=(0,0,0,0,1,1,1,1))))
        self.assertIsNone(common_focal_pose_vectors(observations))
        self.assertIsNone(focal_pose_shape_mismatch(*observations))

    def test_motion_follows_focal_joints_not_other_performers(self):
        config=DirectMemberVisualConfig()
        step=config.sample_frame_step_seconds
        result=_cut_adjacent_motion_evidence(outgoing_previous=sample(seconds=1-step,delta=-.02,raw_pose=1),
            outgoing_endpoint=sample(seconds=1.,raw_pose=-1),
            incoming_start=sample(source=1,seconds=1+step,delta=.02,raw_pose=1),
            incoming_next=sample(source=1,seconds=1+2*step,delta=.04,raw_pose=-1),config=config)
        self.assertGreater(result.direction_cosine_audit_only,.999)
        self.assertIsNone(result.outgoing_silhouette_motion_magnitude)
        self.assertLess(result.pose_phase_continuation_mismatch,1e-8)

    def test_focal_face_basis_can_authorize_actual_lossless_repair(self):
        geometry=direct_member_source_frame_geometry((2160,3840),output_size=(1080,1920))
        left=sample()
        focal=replace(observation(source=1),face_bbox_xywh=(.3825,.0825,.135,.135),face_scale=.04725)
        right=_boundary_sample_with_focal_geometry(master_seconds=1.,source_local_seconds=1.,source_index=1,
            descriptor=descriptor(),provider=SimpleNamespace(focal_geometry_at=lambda *a:focal))
        result=_boundary_mismatch(left,right,DirectMemberVisualConfig(),outgoing_source_index=0,incoming_source_index=1,
            outgoing_source_geometry=geometry,incoming_source_geometry=geometry,continuous_official_portrait_fancam_contract=True)
        self.assertIsNotNone(result.transform_repair)
        self.assertEqual(result.transform_repair.alignment_geometry_basis,FOCAL_SELECTOR_FACE_BASIS)

    def test_hd_focal_alignment_uses_the_existing_dense_resample_policy(self):
        geometry = direct_member_source_frame_geometry((1080, 1920))
        left = sample()
        focal = replace(observation(source=1), face_bbox_xywh=(.394, .094, .112, .112), face_scale=.0392)
        right = _boundary_sample_with_focal_geometry(master_seconds=1., source_local_seconds=1., source_index=1,
            descriptor=descriptor(), provider=SimpleNamespace(focal_geometry_at=lambda *a: focal))
        def mismatch(config, official=True):
            return _boundary_mismatch(left, right, config, outgoing_source_index=0, incoming_source_index=1,
                outgoing_source_geometry=geometry, incoming_source_geometry=geometry,
                continuous_official_portrait_fancam_contract=official)
        old = mismatch(DirectMemberVisualConfig(transform_repair_maximum_upscale_ratio=1.0))
        actual = mismatch(DirectMemberVisualConfig())
        self.assertIsNone(old.transform_repair)
        self.assertIn('cross_source_face_alignment_unrepaired', old.severe_flags)
        self.assertIsNotNone(actual.transform_repair)
        self.assertLessEqual(actual.transform_repair.upscale_ratio, 1.15)
        self.assertNotIn('cross_source_face_alignment_unrepaired', actual.severe_flags)
        self.assertIn('bounded_resample_transform_repair_required', actual.warning_flags)
        self.assertNotIn('lossless_transform_repair_required', actual.warning_flags)
        self.assertIsNone(mismatch(DirectMemberVisualConfig(), official=False).transform_repair)

    def test_selector_and_dense_camera_share_explicit_resample_limit(self):
        from crosscut.pipeline import _direct_member_editorial_motion_config_from_visual_config
        for ratio in (1.0, 1.08, 1.15):
            settings = DirectMemberVisualConfig(transform_repair_maximum_upscale_ratio=ratio)
            dense = _direct_member_editorial_motion_config_from_visual_config(settings)
            self.assertEqual(dense.maximum_boundary_upscale_ratio, ratio)
            self.assertEqual(settings.to_dict()['lossless_transform_repair']['maximum_upscale_ratio'], ratio)
        for ratio in (True, .99, 1.15001, float('nan')):
            with self.assertRaises(ValueError):
                DirectMemberVisualConfig(transform_repair_maximum_upscale_ratio=ratio)


if __name__ == '__main__':
    unittest.main()
