from dataclasses import replace
import math
import random
import unittest

from crosscut.joint_boundary_provider import (
    DecodedBoundaryOptionProvider,
    _crop_resolved_transforms,
    _matched_face_crops,
    _normalized_bbox_after_crop,
)
from engine.tests.test_joint_boundary_provider import (
    _candidate, _descriptor, _key, _provider,
)


class JointPairCropTests(unittest.TestCase):
    def resolve(self, outgoing_box, incoming_box):
        return _crop_resolved_transforms(
            raw_context_identity='pair-crop-test',
            outgoing_candidate=_candidate(0, '0'),
            incoming_candidate=_candidate(1, '1'),
            outgoing_descriptor=_descriptor(0.5, face_bbox=outgoing_box),
            incoming_descriptor=_descriptor(0.5, face_bbox=incoming_box),
        )

    def test_pair_sized_crop_matches_faces_that_fixed_target_clamps_equally(self):
        outgoing_box = (0.34, 0.225, 0.12, 0.15)
        incoming_box = (0.562, 0.24, 0.176, 0.22)
        result = self.resolve(outgoing_box, incoming_box)
        self.assertIsNotNone(result)
        outgoing, incoming, basis = result
        self.assertEqual(basis, 'dominant_face')
        boxes = [
            _normalized_bbox_after_crop(outgoing_box, outgoing.crop_xywh),
            _normalized_bbox_after_crop(incoming_box, incoming.crop_xywh),
        ]
        self.assertAlmostEqual(boxes[0][3], boxes[1][3], places=10)
        for axis in (0, 1):
            centers = [box[axis] + box[axis + 2] / 2.0 for box in boxes]
            self.assertAlmostEqual(centers[0], centers[1], places=10)
        for transform in (outgoing, incoming):
            x, y, width, height = transform.crop_xywh
            self.assertGreaterEqual(width, 0.5 - 1e-12)
            self.assertEqual(width, height)
            self.assertGreaterEqual(x, 0.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(x + width, 1.0 + 1e-12)
            self.assertLessEqual(y + height, 1.0 + 1e-12)
            self.assertLessEqual(math.hypot(x + width / 2 - 0.5, y + height / 2 - 0.5), 0.25 + 1e-12)

    def test_already_equal_centered_faces_do_not_need_maximum_zoom(self):
        result = self.resolve((0.44, 0.32, 0.12, 0.15), (0.44, 0.32, 0.12, 0.15))
        for transform in result[:2]:
            self.assertIsNone(transform.crop_xywh)

    def test_unmatchable_ratio_retains_bounded_candidate_for_decoded_gate(self):
        result = self.resolve((0.46, 0.36, 0.08, 0.10), (0.30, 0.20, 0.40, 0.50))
        for transform in result[:2]:
            self.assertGreaterEqual(transform.crop_xywh[2], 0.5)
        # This solver proposes pixels; it must not fake matched face sizes.
        first = _normalized_bbox_after_crop((0.46, 0.36, 0.08, 0.10), result[0].crop_xywh)
        second = _normalized_bbox_after_crop((0.30, 0.20, 0.40, 0.50), result[1].crop_xywh)
        self.assertGreater(abs(first[3] - second[3]), 0.1)

    def test_silhouette_basis_keeps_existing_non_face_policy(self):
        outgoing = _descriptor(0.4)
        incoming = _descriptor(0.6)
        result = _crop_resolved_transforms(
            raw_context_identity='silhouette-test',
            outgoing_candidate=_candidate(0, '0'),
            incoming_candidate=_candidate(1, '1'),
            outgoing_descriptor=outgoing,
            incoming_descriptor=incoming,
        )
        self.assertEqual(result[2], 'subject_silhouette')
        self.assertAlmostEqual(result[0].crop_xywh[2], 0.70 / 0.72)

    def test_random_pairs_never_claim_a_match_outside_geometry_bounds(self):
        rng = random.Random(20260912)
        accepted = 0
        for _ in range(1000):
            boxes = []
            for _side in range(2):
                width = rng.uniform(0.05, 0.25)
                height = rng.uniform(0.10, 0.40)
                boxes.append((rng.uniform(0.0, 1.0-width), rng.uniform(0.0, 1.0-height), width, height))
            crops = _matched_face_crops(*boxes)
            if crops is None:
                continue
            accepted += 1
            for transform in self.resolve(*boxes)[:2]:
                keyframe = DecodedBoundaryOptionProvider._renderer_keyframe(transform, 0.0)
                self.assertEqual(
                    DecodedBoundaryOptionProvider._renderer_crop(keyframe),
                    transform.crop_xywh,
                )
            extents = []
            centers = []
            for box, crop in zip(boxes, crops):
                x, y, scale, _ = crop
                self.assertGreaterEqual(scale, 0.5 - 1e-9)
                self.assertLessEqual(scale, 1.0)
                self.assertLessEqual(math.hypot(x + scale/2 - 0.5, y + scale/2 - 0.5), 0.25 + 1e-9)
                self.assertGreaterEqual(box[0], x - 1e-9)
                self.assertGreaterEqual(box[1], y - 1e-9)
                self.assertLessEqual(box[0] + box[2], x + scale + 1e-9)
                self.assertLessEqual(box[1] + box[3], y + scale + 1e-9)
                extents.append(max(box[2:]) / scale)
                centers.append(((box[0] + box[2]/2 - x)/scale, (box[1] + box[3]/2 - y)/scale))
            self.assertAlmostEqual(extents[0], extents[1], places=8)
            self.assertLessEqual(math.dist(*centers), 1e-8)
        self.assertGreater(accepted, 100)

    def test_identity_side_matches_real_renderer_proof_representation(self):
        outgoing = _descriptor(
            0.5, shot_scale=0.76, face_bbox=(0.38, 0.28, 0.24, 0.32),
        )
        incoming = _descriptor(
            0.5, shot_scale=0.40, face_bbox=(0.435, 0.24, 0.13, 0.17),
        )

        def transformed(_source, _seconds, raw, transform):
            if transform.crop_xywh is None:
                return raw
            box = _normalized_bbox_after_crop(raw.primary_face_bbox, transform.crop_xywh)
            return replace(raw, primary_face_bbox=box, shot_scale=box[3]/0.42)

        provider = _provider(
            lambda source, _seconds: outgoing if source == 0 else incoming,
            transformed_descriptor_provider=transformed,
            face_identity_provider=lambda *_args: (1.0, 0.0),
        )
        self.addCleanup(provider.close)
        option = provider(_key()).audit_for(0).authorized_option
        self.assertIsNotNone(option)
        self.assertIsNone(option.outgoing_transform.crop_xywh)
        self.assertIsNotNone(option.incoming_transform.crop_xywh)
        proof = provider.selection_materialization_proof((option,))
        self.assertEqual(len(proof.segment_paths), 2)


if __name__ == '__main__':
    unittest.main()
