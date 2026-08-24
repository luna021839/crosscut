from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from crosscut.boundary_affine_match import (
    BoundaryAffineMatchConfig,
    BoundaryAffineTransformEvidence,
    match_boundary_affine,
)


def _rich_frame(width: int = 640, height: int = 360) -> np.ndarray:
    random = np.random.default_rng(1234)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _index in range(160):
        x = int(random.integers(20, width - 20))
        y = int(random.integers(20, height - 20))
        radius = int(random.integers(3, 10))
        color = tuple(int(value) for value in random.integers(30, 255, 3))
        cv2.circle(frame, (x, y), radius, color, -1)
    for index in range(20):
        cv2.putText(
            frame,
            str(index),
            (15 + (index % 5) * 120, 30 + (index // 5) * 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


def _known_pair(
    *,
    scale: float = 1.09,
    roll_degrees: float = 1.0,
    target_center: tuple[float, float] = (0.492, 0.506),
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]:
    outgoing = _rich_frame()
    height, width = outgoing.shape[:2]
    radians = math.radians(roll_degrees)
    linear = np.asarray(
        [
            [scale * math.cos(radians), -scale * math.sin(radians)],
            [scale * math.sin(radians), scale * math.cos(radians)],
        ],
        dtype=np.float64,
    )
    input_center = np.asarray([width / 2.0, height / 2.0])
    target = np.asarray(
        [target_center[0] * width, target_center[1] * height]
    )
    translation = target - linear @ input_center
    incoming_to_outgoing = np.column_stack((linear, translation))
    outgoing_to_incoming = cv2.invertAffineTransform(incoming_to_outgoing)
    incoming = cv2.warpAffine(
        outgoing,
        outgoing_to_incoming,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    incoming_box = (0.35, 0.25, 0.18, 0.30)
    corners = np.asarray(
        [
            [
                [incoming_box[0] * width, incoming_box[1] * height],
                [
                    (incoming_box[0] + incoming_box[2]) * width,
                    incoming_box[1] * height,
                ],
                [
                    incoming_box[0] * width,
                    (incoming_box[1] + incoming_box[3]) * height,
                ],
                [
                    (incoming_box[0] + incoming_box[2]) * width,
                    (incoming_box[1] + incoming_box[3]) * height,
                ],
            ]
        ],
        dtype=np.float32,
    )
    transformed = cv2.transform(corners, incoming_to_outgoing)[0]
    left, top = transformed.min(axis=0)
    right, bottom = transformed.max(axis=0)
    outgoing_box = (
        float(left / width),
        float(top / height),
        float((right - left) / width),
        float((bottom - top) / height),
    )
    return outgoing, incoming, outgoing_box, incoming_box


class BoundaryAffineMatchTests(unittest.TestCase):
    def test_recovers_known_bounded_transform_and_is_deterministic(self) -> None:
        outgoing, incoming, outgoing_box, incoming_box = _known_pair()

        first = match_boundary_affine(
            outgoing,
            incoming,
            outgoing_subject_box=outgoing_box,
            incoming_subject_box=incoming_box,
        )
        second = match_boundary_affine(
            outgoing,
            incoming,
            outgoing_subject_box=outgoing_box,
            incoming_subject_box=incoming_box,
        )

        self.assertTrue(first.eligible)
        self.assertIsNone(first.rejection_reason)
        self.assertIsNotNone(first.transform)
        transform = first.transform
        assert transform is not None
        self.assertAlmostEqual(transform.zoom, 1.09, delta=0.012)
        self.assertAlmostEqual(transform.roll_degrees, 1.0, delta=0.15)
        self.assertAlmostEqual(transform.center_x, 0.5073, delta=0.006)
        self.assertAlmostEqual(transform.center_y, 0.4945, delta=0.006)
        diagnostics = first.diagnostics
        self.assertGreaterEqual(diagnostics["affine_inlier_count"], 13)
        self.assertGreaterEqual(diagnostics["affine_inlier_ratio"], 0.52)
        self.assertLess(
            diagnostics["median_reprojection_error_pixels"],
            2.4,
        )
        self.assertLess(diagnostics["perspective_distortion"], 0.075)
        subject = diagnostics["subject_review"]
        self.assertGreater(subject["transformed_visible_fraction"], 0.97)
        self.assertGreater(subject["geometry_improvement_fraction"], 0.16)
        self.assertEqual(first.to_dict(), second.to_dict())
        serialized = first.to_dict()
        self.assertFalse(serialized["storage_policy"]["pixels_serialized"])
        self.assertFalse(serialized["storage_policy"]["descriptors_serialized"])
        self.assertNotIn("matched_points", serialized["diagnostics"])

    def test_low_feature_frames_are_rejected_without_a_transform(self) -> None:
        blank = np.full((360, 640, 3), 96, dtype=np.uint8)
        result = match_boundary_affine(
            blank,
            blank.copy(),
            outgoing_subject_box=(0.4, 0.2, 0.2, 0.4),
            incoming_subject_box=(0.4, 0.2, 0.2, 0.4),
        )

        self.assertFalse(result.eligible)
        self.assertEqual(result.rejection_reason, "descriptors_unavailable")
        self.assertIsNone(result.transform)

    def test_strong_visual_fit_without_subject_geometry_is_not_eligible(self) -> None:
        outgoing, incoming, _outgoing_box, _incoming_box = _known_pair()
        result = match_boundary_affine(outgoing, incoming)

        self.assertFalse(result.eligible)
        self.assertEqual(result.rejection_reason, "subject_geometry_unavailable")
        self.assertIsNone(result.transform)
        self.assertGreater(result.diagnostics["affine_inlier_count"], 20)

    def test_translation_beyond_two_point_five_percent_is_rejected(self) -> None:
        outgoing, incoming, outgoing_box, incoming_box = _known_pair(
            scale=1.05,
            roll_degrees=0.0,
            target_center=(0.455, 0.5),
        )
        result = match_boundary_affine(
            outgoing,
            incoming,
            outgoing_subject_box=outgoing_box,
            incoming_subject_box=incoming_box,
        )

        self.assertFalse(result.eligible)
        self.assertEqual(
            result.rejection_reason,
            "translation_exceeds_center_shift_bound",
        )
        self.assertGreater(result.diagnostics["center_shift"], 0.025)

    def test_rotation_and_scale_beyond_renderer_bounds_are_rejected(self) -> None:
        outgoing, incoming, outgoing_box, incoming_box = _known_pair(
            scale=1.10,
            roll_degrees=2.2,
        )
        rotation = match_boundary_affine(
            outgoing,
            incoming,
            outgoing_subject_box=outgoing_box,
            incoming_subject_box=incoming_box,
        )
        self.assertFalse(rotation.eligible)
        self.assertEqual(
            rotation.rejection_reason,
            "rotation_exceeds_roll_bound",
        )

        outgoing, incoming, outgoing_box, incoming_box = _known_pair(
            scale=1.22,
            roll_degrees=0.0,
        )
        scale = match_boundary_affine(
            outgoing,
            incoming,
            outgoing_subject_box=outgoing_box,
            incoming_subject_box=incoming_box,
            config=BoundaryAffineMatchConfig(maximum_determinant=1.60),
        )
        self.assertFalse(scale.eligible)
        self.assertEqual(scale.rejection_reason, "scale_exceeds_zoom_bound")

    def test_transform_must_improve_supplied_subject_geometry(self) -> None:
        outgoing, incoming, _outgoing_box, incoming_box = _known_pair()
        result = match_boundary_affine(
            outgoing,
            incoming,
            # Claim the subject was already screen-matched.  Background ORB
            # evidence is not allowed to disturb that safer geometry.
            outgoing_subject_box=incoming_box,
            incoming_subject_box=incoming_box,
        )

        self.assertFalse(result.eligible)
        self.assertIn(
            result.rejection_reason,
            {
                "subject_geometry_already_matched",
                "subject_geometry_improvement_too_small",
                "subject_geometry_improvement_fraction_too_small",
            },
        )
        review = result.diagnostics["subject_review"]
        self.assertLessEqual(
            review["geometry_improvement"],
            BoundaryAffineMatchConfig().minimum_geometry_improvement,
        )

    def test_transform_schema_reuses_renderer_safety_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "zoom"):
            BoundaryAffineTransformEvidence(1.19, 0.5, 0.5, 0.0)
        with self.assertRaisesRegex(ValueError, "roll"):
            BoundaryAffineTransformEvidence(1.18, 0.5, 0.5, 1.7)

    def test_configuration_rejects_bounds_stronger_than_renderer(self) -> None:
        with self.assertRaisesRegex(ValueError, "renderer safety"):
            BoundaryAffineMatchConfig(maximum_center_shift=0.05)
        with self.assertRaisesRegex(ValueError, "renderer safety"):
            BoundaryAffineMatchConfig(maximum_roll_degrees=1.7)


if __name__ == "__main__":
    unittest.main()
