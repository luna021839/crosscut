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
        self.assertEqual(descriptor.shot_type, ShotType.GROUP_WIDE)
        self.assertIsNone(output.face_bbox_xywh)
        self.assertEqual(classification, "wide")

    def test_existing_area_dominance_gate_is_not_lowered(self):
        # Below the existing 1.8x area ratio is still ambiguous, even though
        # the largest face alone is above the close-up height threshold.
        for ratio, expected in ((1.79, "wide"), (1.81, "close")):
            with self.subTest(ratio=ratio):
                foreground = _DetectedFace((120.0, 20.0, 64.0, 72.0))
                runner = _DetectedFace((20.0, 20.0, 64.0, 72.0 / ratio))
                descriptor, output, classification = self._observe([
                    foreground, runner,
                    _DetectedFace((250.0, 50.0, 12.0, 20.0)),
                    _DetectedFace((280.0, 50.0, 12.0, 20.0)),
                ])
                self.assertEqual(classification, expected)
                self.assertEqual(output.face_bbox_xywh is not None, expected == "close")
                self.assertEqual(
                    descriptor.shot_type,
                    ShotType.CLOSE_UP if expected == "close" else ShotType.GROUP_WIDE,
                )

    def test_dominance_does_not_promote_small_face_to_close(self):
        descriptor, _, classification = self._observe([
            _DetectedFace((140.0, 50.0, 18.0, 24.0)),
            *[_DetectedFace((float(20 + i * 30), 80.0, 5.0, 8.0)) for i in range(3)],
        ])
        self.assertNotEqual(descriptor.shot_type, ShotType.CLOSE_UP)
        self.assertNotEqual(classification, "close")

    def test_large_scale_without_unique_bbox_cannot_override_four_faces(self):
        self.assertEqual(_proxy_geometry_shot_class({
            "face_count": 4,
            "face_bbox_xywh": None,
            "face_backend": "yunet",
            "shot_scale_basis": "dominant_face_height",
            "framing_face_scale": .95,
        }, RenderedMontageGrammarConfig()), "wide")
