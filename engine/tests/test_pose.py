from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from crosscut.pose import (
    HUMAN_SEGMENTER_MODEL,
    SILHOUETTE_BACKEND,
    SubjectGeometryAnalyzer,
    _PoseObservation,
    _pose_detector_anchors,
    _pose_embedding,
    _silhouette_descriptor,
    model_asset_status,
)


class PoseModelTests(unittest.TestCase):
    def test_generated_detector_anchors_match_model_contract(self) -> None:
        anchors = _pose_detector_anchors()
        self.assertEqual(anchors.shape, (2254, 2))
        np.testing.assert_array_equal(anchors[0], (0.5 / 28, 0.5 / 28))
        np.testing.assert_array_equal(anchors[1568], (0.5 / 14, 0.5 / 14))
        np.testing.assert_array_equal(anchors[1960], (0.5 / 7, 0.5 / 7))
        np.testing.assert_array_equal(anchors[-1], (6.5 / 7, 6.5 / 7))

    def test_pose_embedding_is_translation_and_scale_invariant(self) -> None:
        landmarks = np.zeros((33, 5), dtype=np.float64)
        landmarks[:, 3:] = 0.95
        coordinates = {
            0: (100, 25, -3),
            11: (80, 70, 0),
            12: (120, 70, 0),
            13: (66, 105, 2),
            14: (134, 105, 2),
            15: (58, 140, 4),
            16: (142, 140, 4),
            23: (87, 135, 0),
            24: (113, 135, 0),
            25: (84, 190, 2),
            26: (116, 190, 2),
            27: (82, 245, 5),
            28: (118, 245, 5),
            31: (72, 252, 8),
            32: (128, 252, 8),
        }
        for index, coordinate in coordinates.items():
            landmarks[index, :3] = coordinate
        world = np.zeros((33, 3), dtype=np.float64)
        original, original_confidence = _pose_embedding(
            _PoseObservation(landmarks, world, 0.9)
        )
        transformed = landmarks.copy()
        transformed[:, :3] *= 1.7
        transformed[:, 0] += 61
        transformed[:, 1] -= 19
        shifted, shifted_confidence = _pose_embedding(
            _PoseObservation(transformed, world, 0.9)
        )
        self.assertEqual(len(original), 60)
        np.testing.assert_allclose(original, shifted, atol=1e-12)
        self.assertAlmostEqual(original_confidence, shifted_confidence)

    def test_silhouette_keeps_multiple_people_and_reports_union_geometry(self) -> None:
        probability = np.zeros((192, 192), dtype=np.float32)
        probability[30:165, 25:70] = 0.94
        probability[50:180, 120:170] = 0.91
        embedding, bbox, centroid, area, confidence = _silhouette_descriptor(
            probability
        )
        self.assertEqual(len(embedding), 121)
        self.assertIsNotNone(bbox)
        self.assertIsNotNone(centroid)
        assert bbox is not None
        self.assertLess(bbox[0], 0.14)
        self.assertGreater(bbox[0] + bbox[2], 0.88)
        self.assertGreater(area, 0.30)
        self.assertGreater(confidence, 0.90)

    def test_missing_assets_produce_empty_deterministic_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            analyzer = SubjectGeometryAnalyzer(Path(directory))
            descriptor = analyzer.describe(np.zeros((120, 200, 3), dtype=np.uint8))
        self.assertEqual(descriptor.pose_embedding, ())
        self.assertEqual(descriptor.pose_backend, "none")
        self.assertEqual(descriptor.silhouette_embedding, ())
        self.assertEqual(descriptor.silhouette_backend, "none")

    def test_bundled_assets_have_exact_digest_and_load_in_opencv(self) -> None:
        statuses = model_asset_status(verify_load=True)
        self.assertEqual(len(statuses), 3)
        self.assertTrue(all(status.present for status in statuses))
        self.assertTrue(all(status.digest_valid for status in statuses))
        self.assertTrue(all(status.loadable for status in statuses))

    def test_bundled_segmenter_executes_without_optional_python_packages(self) -> None:
        descriptor = SubjectGeometryAnalyzer().describe(
            np.zeros((120, 200, 3), dtype=np.uint8)
        )
        self.assertEqual(descriptor.silhouette_backend, SILHOUETTE_BACKEND)
        self.assertEqual(descriptor.pose_embedding, ())

    def test_corrupt_asset_is_reported_before_opencv_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / HUMAN_SEGMENTER_MODEL
            path.write_bytes(b"not an onnx model")
            status = next(
                item
                for item in model_asset_status(model_directory=Path(directory))
                if item.name == HUMAN_SEGMENTER_MODEL
            )
        self.assertTrue(status.present)
        self.assertFalse(status.digest_valid)
        self.assertIsNone(status.loadable)


if __name__ == "__main__":
    unittest.main()
