from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from crosscut.pose import (
    FORMATION_PERSON_DETECTOR_MODEL,
    FormationPersonDescriptor,
    HUMAN_SEGMENTER_MODEL,
    SILHOUETTE_BACKEND,
    SubjectGeometryAnalyzer,
    _PoseObservation,
    _formation_people_descriptor,
    _person_layout_embedding,
    _pose_detector_anchors,
    _pose_embedding,
    _silhouette_descriptor,
    _upper_body_shape,
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

        original_upper = _upper_body_shape(
            _PoseObservation(landmarks, world, 0.9)
        )
        shifted_upper = _upper_body_shape(
            _PoseObservation(transformed, world, 0.9)
        )
        self.assertEqual(len(original_upper), 8)
        np.testing.assert_allclose(original_upper, shifted_upper, atol=1e-12)

    def test_formation_descriptor_keeps_six_people_left_to_right(self) -> None:
        persons = np.asarray(
            [
                (x, 20, x + 30, 190, 0.80 + index * 0.02)
                for index, x in enumerate((250, 10, 210, 50, 170, 90, 130))
            ],
            dtype=np.float64,
        )

        formation, observations = _formation_people_descriptor(
            persons,
            frame_width=300,
            frame_height=200,
        )

        self.assertEqual(len(formation), 6)
        self.assertEqual(observations, {})
        self.assertTrue(
            all(
                left.centroid_xy[0] <= right.centroid_xy[0]
                for left, right in zip(formation, formation[1:])
            )
        )
        self.assertTrue(
            all(isinstance(person, FormationPersonDescriptor) for person in formation)
        )
        self.assertTrue(all(not person.upper_body_keypoints_xyc for person in formation))

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

    def test_person_layout_embedding_distinguishes_left_right_formation(self) -> None:
        tall_left = np.asarray(
            (
                (20, 20, 70, 190, 0.95),
                (230, 70, 270, 190, 0.92),
            ),
            dtype=np.float64,
        )
        tall_right = np.asarray(
            (
                (20, 70, 60, 190, 0.92),
                (220, 20, 270, 190, 0.95),
            ),
            dtype=np.float64,
        )

        left_embedding = _person_layout_embedding(
            tall_left, frame_width=300, frame_height=200
        )
        right_embedding = _person_layout_embedding(
            tall_right, frame_width=300, frame_height=200
        )

        self.assertEqual(len(left_embedding), 134)
        self.assertEqual(len(right_embedding), 134)
        self.assertGreater(left_embedding[99], left_embedding[104])
        self.assertLess(right_embedding[99], right_embedding[104])
        self.assertFalse(np.allclose(left_embedding, right_embedding))

    def test_person_layout_embedding_distinguishes_person_count(self) -> None:
        two_people = np.asarray(
            (
                (20, 25, 70, 190, 0.95),
                (230, 30, 280, 190, 0.93),
            ),
            dtype=np.float64,
        )
        three_people = np.vstack(
            (two_people, np.asarray((125, 45, 175, 190, 0.91), dtype=np.float64))
        )

        two_embedding = _person_layout_embedding(
            two_people, frame_width=300, frame_height=200
        )
        three_embedding = _person_layout_embedding(
            three_people, frame_width=300, frame_height=200
        )

        np.testing.assert_allclose(two_embedding[-8:], (2.0 / 6.0,) * 8)
        np.testing.assert_allclose(three_embedding[-8:], (3.0 / 6.0,) * 8)
        self.assertFalse(np.allclose(two_embedding, three_embedding))

    def test_missing_assets_produce_empty_deterministic_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            analyzer = SubjectGeometryAnalyzer(Path(directory))
            descriptor = analyzer.describe(np.zeros((120, 200, 3), dtype=np.uint8))
        self.assertEqual(descriptor.pose_embedding, ())
        self.assertEqual(descriptor.pose_backend, "none")
        self.assertEqual(descriptor.silhouette_embedding, ())
        self.assertEqual(descriptor.silhouette_backend, "none")

    def test_formation_detector_drives_six_person_layout_without_identity(self) -> None:
        class _FormationDetector:
            @staticmethod
            def infer(_image: np.ndarray) -> np.ndarray:
                return np.asarray(
                    [
                        (20 + index * 28, 18, 44 + index * 28, 190, 0.91)
                        for index in range(6)
                    ],
                    dtype=np.float64,
                )

        class _PoseDetector:
            @staticmethod
            def infer(_image: np.ndarray) -> np.ndarray:
                return np.empty((0, 13), dtype=np.float64)

        class _PoseEstimator:
            @staticmethod
            def infer(_image: np.ndarray, _person: np.ndarray) -> None:
                return None

        with tempfile.TemporaryDirectory() as directory:
            analyzer = SubjectGeometryAnalyzer(Path(directory))
            analyzer._pose_load_attempted = True
            analyzer._formation_load_attempted = True
            analyzer._segmenter_load_attempted = True
            analyzer._pose_detector = _PoseDetector()
            analyzer._pose_estimator = _PoseEstimator()
            analyzer._formation_detector = _FormationDetector()
            descriptor = analyzer.describe(
                np.zeros((200, 220, 3), dtype=np.uint8)
            )

        self.assertEqual(descriptor.detected_person_count, 6)
        self.assertEqual(len(descriptor.formation_people), 6)
        self.assertEqual(len(descriptor.person_layout_embedding), 134)
        np.testing.assert_allclose(
            descriptor.person_layout_embedding[-8:],
            (1.0,) * 8,
        )
        self.assertTrue(
            all(
                left.centroid_xy[0] < right.centroid_xy[0]
                for left, right in zip(
                    descriptor.formation_people,
                    descriptor.formation_people[1:],
                )
            )
        )

    def test_bundled_assets_have_exact_digest_and_load_in_opencv(self) -> None:
        statuses = model_asset_status(verify_load=True)
        self.assertEqual(len(statuses), 4)
        self.assertTrue(all(status.present for status in statuses))
        self.assertTrue(all(status.digest_valid for status in statuses))
        self.assertTrue(all(status.loadable for status in statuses))
        self.assertIn(
            FORMATION_PERSON_DETECTOR_MODEL,
            {status.name for status in statuses},
        )

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
