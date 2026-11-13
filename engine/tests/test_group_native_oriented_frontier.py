"""Variant-to-native-row/descriptor/frontier wiring, not real media quality."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import hashlib
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from crosscut.group_joint_native_observer import observe_native_group
from crosscut.group_native_head_orientation import (
    SUPPORTED_ANALYSIS_ROLLS, inverse_mapped_head, native_head_analysis_spec)
from crosscut.group_joint_focal_frontier import (
    observe_native_face_appearances, observe_synchronized_focal_frontier)
from crosscut.group_joint_source_camera import JointGroupWorkBudget
from crosscut.native_face_descriptor import ARCFACE_MODEL_NAME, ARCFACE_R50_SHA256
from tests import test_group_joint_temporal_pose as fixtures


class NativeOrientedFrontierTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(prefix='crosscut-oriented-row-fixture-')
        self.addCleanup(self.directory.cleanup)
        sources = tuple(Path(self.directory.name) / f'synthetic-{n}.mp4' for n in range(2))
        for source in sources:
            source.write_bytes(b'synthetic-only: decoder is explicitly mocked')
        self.domain = replace(fixtures.NativeTemporalPoseTests().domain(), request_sources=sources)
        self.pixels = np.zeros((500, 1000, 3), np.uint8)
        self.pixels[230:240, 450:460] = (30, 60, 90)

    def oriented(self, angle=30.):
        return inverse_mapped_head(
            np.asarray((390., 140., 150., 180., 425., 190., 475., 190.,
                450., 230., 425., 270., 475., 270., .92)),
            working_size_wh=(1000, 500),
            analysis_affine=cv2.getRotationMatrix2D((499.5, 249.5), angle, 1.),
            analysis_roll_degrees=angle, observation_id=0)

    def owner(self):
        names = ('nanodet_people', 'rtmpose_body7', 'yunet_head_faces',
            *(native_head_analysis_spec(a)[0] for a in SUPPORTED_ANALYSIS_ROLLS))
        budget = JointGroupWorkBudget(absolute_deadline=10., clock=lambda: 0.)
        callbacks = {'nanodet_people': ((100., 80., 900., 480., .95),),
            'rtmpose_body7': None,
            'yunet_head_faces': (replace(self.oriented().head, observation_id=0),),
            **{native_head_analysis_spec(a)[0]: (self.oriented(a),) for a in SUPPORTED_ANALYSIS_ROLLS}}
        owner = SimpleNamespace(domain=self.domain, budget=budget,
            models={**{name: (None, None, 'e'*64) for name in names},
                ARCFACE_MODEL_NAME: (None, None, ARCFACE_R50_SHA256)},
            model_adapters={name: hashlib.sha256(name.encode()).hexdigest()
                for name in (*names, ARCFACE_MODEL_NAME)},
            callbacks=callbacks, _assert_model_assets=Mock(),
            native_pixels=Mock(return_value=self.pixels), retain_native_pixels=Mock())
        def observe(source, native, *, model, arguments=None):
            budget.spend('model_calls', model=model)
            if model == ARCFACE_MODEL_NAME:
                return (1.,) + (0.,)*511
            return callbacks[model]
        owner.observe_native = Mock(side_effect=observe)
        return owner

    def test_one_variant_reaches_raw_heads_and_descriptor_on_unmodified_native_pixels(self):
        before = hashlib.sha256(memoryview(self.pixels)).hexdigest()
        hashes = set()
        for angle in SUPPORTED_ANALYSIS_ROLLS:
            owner = self.owner()
            name, base = native_head_analysis_spec(angle)
            row = observe_native_group(owner, request_source_index=1,
                native_frame_index=3, head_analysis_roll=angle)
            self.assertEqual(tuple(n for n, _ in row.model_content_sha256),
                ('nanodet_people', 'rtmpose_body7', name))
            self.assertEqual(row.heads[0].observation_id, base)
            self.assertEqual(row.heads[0].landmarks_xy, self.oriented(angle).head.landmarks_xy)
            self.assertEqual(row.native_pixel_sha256, before)
            values = observe_native_face_appearances(owner, row, descriptor_model=ARCFACE_MODEL_NAME)
            self.assertEqual(len(values), 1)
            self.assertEqual(values[0].face_observation_id, base)
            self.assertEqual(values[0].descriptor_space.dimension, 512)
            self.assertEqual(owner.budget.model_calls,
                {name: 1, 'nanodet_people': 1, 'rtmpose_body7': 1, ARCFACE_MODEL_NAME: 1})
            self.assertEqual(owner.observe_native.call_args.kwargs['arguments']['landmarks_xy'],
                [list(p) for p in row.heads[0].landmarks_xy])
            hashes.add(row.sha256)
            self.assertFalse(row.to_dict()['quality_approved'])
            self.assertFalse(row.to_dict()['visibility_certified'])
        self.assertEqual(len(hashes), 4)
        self.assertEqual(hashlib.sha256(memoryview(self.pixels)).hexdigest(), before)

    def test_default_unchanged_and_variant_ids_do_not_collide(self):
        owner = self.owner()
        old = observe_native_group(owner, request_source_index=0, native_frame_index=0)
        explicit = observe_native_group(owner, request_source_index=0, native_frame_index=0,
            head_analysis_roll=None)
        self.assertEqual(old.to_dict(), explicit.to_dict())
        self.assertEqual(old.heads[0].observation_id, 0)
        ids = [set(range(native_head_analysis_spec(a)[1], native_head_analysis_spec(a)[1]+24))
            for a in (None, *SUPPORTED_ANALYSIS_ROLLS)]
        self.assertEqual(len(set.union(*ids)), 5*24)

    def test_missing_or_invalid_variant_rejects_before_native_decode(self):
        for angle in (True, 0., 90., float('nan'), '30'):
            owner = self.owner()
            with self.subTest(angle=angle), self.assertRaises(ValueError):
                observe_native_group(owner, request_source_index=0, native_frame_index=0,
                    head_analysis_roll=angle)
            owner.native_pixels.assert_not_called()
        owner = self.owner()
        owner.models.pop(native_head_analysis_spec(30.)[0])
        with self.assertRaisesRegex(ValueError, 'predeclared registered'):
            observe_native_group(owner, request_source_index=0, native_frame_index=0,
                head_analysis_roll=30.)
        owner.native_pixels.assert_not_called()

    def test_wrong_variant_content_is_not_relabelled_and_missing_does_not_fallback(self):
        name, _ = native_head_analysis_spec(30.)
        for value in ((self.oriented(-30.),), (self.oriented(30.),)*25,
                (replace(self.oriented(30.), head=replace(self.oriented(30.).head, observation_id=24)),)):
            owner = self.owner(); owner.callbacks[name] = value
            with self.assertRaisesRegex(ValueError, 'frozen detector variant'):
                observe_native_group(owner, request_source_index=0, native_frame_index=0,
                    head_analysis_roll=30.)
            self.assertEqual(owner.budget.model_calls, {name: 1})
        owner = self.owner(); owner.callbacks[name] = None
        row = observe_native_group(owner, request_source_index=0, native_frame_index=0,
            head_analysis_roll=30.)
        self.assertEqual(row.heads, ())
        self.assertEqual(row.reason, 'head_model_unavailable_or_detection_overflow')
        self.assertNotIn('yunet_head_faces', owner.budget.model_calls)

    def test_ambiguous_bodies_stay_alternatives_after_inverse_mapping(self):
        owner = self.owner()
        owner.callbacks['nanodet_people'] *= 2
        person = fixtures.NativeTemporalPoseTests().person()
        raw = list(person.raw_pose_xyc)
        raw[0] = (*self.oriented(30.).head.landmarks_xy[2], .9)
        owner.callbacks['rtmpose_body7'] = SimpleNamespace(reason='observed',
            upper_body_xyc=person.upper_body_xyc, pose_crop_xywh=person.pose_crop_xywh,
            source_pixel_to_pose_input_affine=person.source_pixel_to_pose_input_affine,
            raw_keypoints_xyc=tuple(raw), face_anchor_consistent=True)
        row = observe_native_group(owner, request_source_index=0, native_frame_index=0,
            head_analysis_roll=30.)
        self.assertEqual(row.heads[0].candidate_body_observation_ids, (0, 1))
        self.assertTrue(all(p.upper_body_ownership_state == 'ambiguous' for p in row.people))
        values = observe_native_face_appearances(owner, row, descriptor_model=ARCFACE_MODEL_NAME)
        self.assertEqual(values[0].candidate_body_observation_ids, (0, 1))
        self.assertIsNone(values[0].person_observation_id)

    def test_frontier_freezes_one_variant_for_all_three_anchors_and_retains_original_reference(self):
        owner = self.owner()
        # Test the real observer/descriptor/frontier call chain. Only native
        # decoder/model outputs and optical flow are synthetic here.
        owner._assert_source_identity = lambda s: row_identity(owner.domain.request_sources[s])
        with patch('crosscut.group_joint_focal_frontier.focal_landmarks_track', return_value=True):
            result = observe_synchronized_focal_frontier(owner, reference_source=0,
                canonical_times=(0., .1, .2), face_roi_only=True,
                descriptor_model=ARCFACE_MODEL_NAME, head_analysis_rolls=((1, 30.),))
        self.assertEqual(result['status'], 'measured_shortlist')
        self.assertEqual(result['head_analysis_policy']['request_source_analysis_roll_degrees'], ((1, 30.),))
        source_rows = [r for r in result['raw_observations'] if r['request_source_index'] == 1]
        self.assertEqual(len(source_rows), 3)
        self.assertTrue(all(r['heads'][0]['observation_id'] == 96 for r in source_rows))
        self.assertEqual(owner.budget.model_calls['yunet_head_faces'], 3)
        self.assertEqual(owner.budget.model_calls[native_head_analysis_spec(30.)[0]], 3)
        self.assertFalse(result['head_analysis_policy']['variants_combined_as_people'])
        self.assertFalse(result['head_analysis_policy']['render_rotation_changed'])
        self.assertFalse(result['cross_source_identity_certified'])
        self.assertFalse(result['quality_approved'])

    def test_frontier_rejects_nonfrozen_unknown_duplicate_or_unregistered_policy_before_decode(self):
        for policy in ([(1, 30.)], ((1, 30.), (1, -30.)), ((1, True),),
                ((2, 30.),), ((True, 30.),), ((1, 17.),)):
            owner = self.owner()
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                observe_synchronized_focal_frontier(owner, reference_source=0,
                    canonical_times=(0., .1, .2), head_analysis_rolls=policy)
            owner.native_pixels.assert_not_called()
        owner = self.owner(); owner.models.pop(native_head_analysis_spec(30.)[0])
        with self.assertRaisesRegex(ValueError, 'not registered'):
            observe_synchronized_focal_frontier(owner, reference_source=0,
                canonical_times=(0., .1, .2), head_analysis_rolls=((1, 30.),))
        owner.native_pixels.assert_not_called()


def row_identity(path):
    from crosscut.group_joint_observations import _file_identity
    return _file_identity(path)


if __name__ == '__main__':
    unittest.main()
