"""A foreground close-up stays close with small background faces present."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from crosscut.pose import SubjectGeometryDescriptor
from crosscut.rendered_montage_grammar_review import (
    RenderedMontageGrammarConfig,
    _proxy_geometry_shot_class,
)
from crosscut.rendered_proxy_review import _default_geometry_provider
from crosscut.timeline import ShotType
from crosscut.visual_score import _DetectedFace, describe_frame


class GroupDominantCloseFramingTests(unittest.TestCase):
    def _observe(self, faces):
        frame = np.full((180, 320, 3), 118, dtype=np.uint8)
        with patch(
            "crosscut.visual_score.describe_subject_geometry",
            return_value=SubjectGeometryDescriptor(),
        ), patch(
            "crosscut.visual_score._detect_faces",
            return_value=(tuple(faces), "test"),
        ):
            descriptor = describe_frame(frame)
            output = _default_geometry_provider(frame)
        return descriptor, output, _proxy_geometry_shot_class(
            output.to_dict(), RenderedMontageGrammarConfig()
        )

    def test_dominant_close_face_survives_four_or_more_total_faces(self):
        foreground = _DetectedFace((120.0, 20.0, 64.0, 72.0))
        for background_count in (3, 5, 8):
            with self.subTest(background_count=background_count):
                backgrounds = [
                    _DetectedFace((float(10 + index * 25), 60.0, 12.0, 20.0))
                    for index in range(background_count)
                ]
                descriptor, output, classification = self._observe(
                    [foreground, *backgrounds]
                )
                self.assertEqual(descriptor.shot_type, ShotType.CLOSE_UP)
                self.assertEqual(descriptor.shot_scale_basis, "dominant_face_height")
                self.assertEqual(output.face_count, background_count + 1)
                self.assertEqual(output.face_bbox_xywh, descriptor.primary_face_bbox)
                self.assertIsNotNone(output.face_bbox_xywh)
                self.assertEqual(classification, "close")

    def test_four_equal_large_faces_do_not_invent_dominance(self):
        descriptor, output, classification = self._observe([
            _DetectedFace((float(index * 70), 20.0, 64.0, 72.0))
            for index in range(4)
        ])
        self.assertEqual(descriptor.shot_type, ShotType.CLOSE_UP)
        self.assertIsNone(output.face_bbox_xywh)
        self.assertEqual(classification, "close")

    def test_existing_area_dominance_gate_is_not_lowered(self):
        # Below the existing 1.8x area ratio is still ambiguous, even though
        # the largest face alone is above the close-up height threshold.
        for ratio, identity_present in ((1.79, False), (1.81, True)):
            with self.subTest(ratio=ratio):
                foreground = _DetectedFace((120.0, 20.0, 64.0, 72.0))
                runner = _DetectedFace((20.0, 20.0, 64.0, 72.0 / ratio))
                descriptor, output, classification = self._observe([
                    foreground, runner,
                    _DetectedFace((250.0, 50.0, 12.0, 20.0)),
                    _DetectedFace((280.0, 50.0, 12.0, 20.0)),
                ])
                self.assertEqual(classification, "close")
                self.assertEqual(output.face_bbox_xywh is not None, identity_present)
                self.assertEqual(descriptor.shot_type, ShotType.CLOSE_UP)

    def test_dominance_does_not_promote_small_face_to_close(self):
        descriptor, _, classification = self._observe([
            _DetectedFace((140.0, 50.0, 18.0, 24.0)),
            *[_DetectedFace((float(20 + i * 30), 80.0, 5.0, 8.0)) for i in range(3)],
        ])
        self.assertNotEqual(descriptor.shot_type, ShotType.CLOSE_UP)
        self.assertNotEqual(classification, "close")

    def test_measured_large_scale_describes_framing_without_assigning_identity(self):
        self.assertEqual(_proxy_geometry_shot_class({
            "face_count": 4,
            "face_bbox_xywh": None,
            "face_backend": "yunet",
            "shot_scale_basis": "dominant_face_height",
            "framing_face_scale": .95,
        }, RenderedMontageGrammarConfig()), "close")

    def test_background_detection_count_cannot_create_scale_or_shot_jumps(self):
        from crosscut.visual_score import describe_boundary_side_stability
        foreground = _DetectedFace((120., 20., 50., 54.))
        # Similar-size neighboring faces deliberately keep identity ambiguous.
        neighbors = [_DetectedFace((float(i * 65), 22., 48., 52.)) for i in range(3)]
        observations = [self._observe([foreground, *neighbors[:count]])[0] for count in (2,3,2)]
        self.assertEqual([d.shot_type for d in observations], [ShotType.CLOSE_UP]*3)
        self.assertTrue(all(d.primary_face_bbox is None for d in observations))
        self.assertEqual([d.shot_scale for d in observations], [54./180./.42]*3)
        for left, right in zip(observations,observations[1:]):
            stability = describe_boundary_side_stability(left,right,sample_span_seconds=.1)
            self.assertEqual(stability.scale_delta, 0.)
            self.assertEqual(stability.shot_type_mismatch, 0.)
