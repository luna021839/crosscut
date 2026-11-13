from __future__ import annotations

import math
import unittest

from crosscut.stage_edit_temporal_pose_motion import (
    DYNAMIC_POSE_PHASE_MATCH_CLASSIFICATION,
    DYNAMIC_POSE_PHASE_MISMATCH_CLASSIFICATION,
    INVALID_CLASSIFICATION,
    STATIC_POSE_MATCH_CLASSIFICATION,
    TemporalPoseMotionThresholds,
    TemporalPoseWindow,
    analyze_temporal_pose_motion_boundary,
)


class StageEditTemporalPoseMotionTests(unittest.TestCase):
    @staticmethod
    def _thresholds() -> TemporalPoseMotionThresholds:
        return TemporalPoseMotionThresholds(
            minimum_boundary_motion_magnitude=0.35,
            minimum_endpoint_pose_cosine=0.90,
            minimum_velocity_direction_cosine=0.80,
            minimum_phase_compatibility=0.90,
            pose_norm_epsilon=1e-9,
        )

    @staticmethod
    def _pose(angle_radians: float, *, scale: float = 1.0) -> tuple[float, ...]:
        return (
            scale * math.cos(angle_radians),
            scale * math.sin(angle_radians),
            0.0,
        )

    def _same_phase_windows(
        self,
        *,
        outgoing_scale: float = 1.0,
        incoming_scale: float = 1.0,
    ) -> tuple[TemporalPoseWindow, TemporalPoseWindow]:
        return (
            TemporalPoseWindow(
                times_seconds=(-0.20, -0.10, 0.0),
                embeddings=(
                    self._pose(0.00, scale=outgoing_scale),
                    self._pose(0.05, scale=outgoing_scale),
                    self._pose(0.10, scale=outgoing_scale),
                ),
            ),
            TemporalPoseWindow(
                times_seconds=(0.0, 0.10, 0.20),
                embeddings=(
                    self._pose(0.10, scale=incoming_scale),
                    self._pose(0.15, scale=incoming_scale),
                    self._pose(0.20, scale=incoming_scale),
                ),
            ),
        )

    def test_same_moving_pose_phase_passes(self) -> None:
        outgoing, incoming = self._same_phase_windows()

        analysis = analyze_temporal_pose_motion_boundary(
            outgoing=outgoing,
            incoming=incoming,
            thresholds=self._thresholds(),
        )

        self.assertTrue(analysis.input_valid)
        self.assertTrue(analysis.qualifies)
        self.assertTrue(analysis.dynamic_pose_phase_match)
        self.assertFalse(analysis.static_pose_match)
        self.assertEqual(
            analysis.classification,
            DYNAMIC_POSE_PHASE_MATCH_CLASSIFICATION,
        )
        self.assertGreater(analysis.endpoint_pose_cosine, 0.999)
        self.assertGreater(analysis.velocity_direction_cosine, 0.99)
        self.assertGreater(analysis.phase_compatibility, 0.99)
        self.assertGreater(analysis.confidence, 0.99)

    def test_identical_static_pose_is_not_dynamic_phase_match(self) -> None:
        static = (self._pose(0.10),) * 3

        analysis = analyze_temporal_pose_motion_boundary(
            outgoing=TemporalPoseWindow((-0.2, -0.1, 0.0), static),
            incoming=TemporalPoseWindow((0.0, 0.1, 0.2), static),
            thresholds=self._thresholds(),
        )

        self.assertTrue(analysis.input_valid)
        self.assertFalse(analysis.qualifies)
        self.assertTrue(analysis.static_pose_match)
        self.assertFalse(analysis.dynamic_pose_phase_match)
        self.assertEqual(analysis.classification, STATIC_POSE_MATCH_CLASSIFICATION)
        self.assertAlmostEqual(analysis.endpoint_pose_cosine, 1.0)
        self.assertEqual(analysis.confidence, 0.0)
        self.assertIn(
            "boundary_velocity_direction_undefined",
            analysis.failure_reasons,
        )

    def test_opposite_motion_direction_fails(self) -> None:
        outgoing, _ = self._same_phase_windows()
        incoming = TemporalPoseWindow(
            times_seconds=(0.0, 0.10, 0.20),
            embeddings=(
                self._pose(0.10),
                self._pose(0.05),
                self._pose(0.00),
            ),
        )

        analysis = analyze_temporal_pose_motion_boundary(
            outgoing=outgoing,
            incoming=incoming,
            thresholds=self._thresholds(),
        )

        self.assertFalse(analysis.qualifies)
        self.assertFalse(analysis.static_pose_match)
        self.assertEqual(
            analysis.classification,
            DYNAMIC_POSE_PHASE_MISMATCH_CLASSIFICATION,
        )
        self.assertLess(analysis.velocity_direction_cosine, -0.99)
        self.assertIn(
            "velocity_direction_below_threshold",
            analysis.failure_reasons,
        )

    def test_high_motion_with_mismatched_endpoint_pose_fails(self) -> None:
        outgoing, _ = self._same_phase_windows()
        incoming = TemporalPoseWindow(
            times_seconds=(0.0, 0.10, 0.20),
            embeddings=(
                self._pose(math.pi / 2.0),
                self._pose(math.pi / 2.0 + 0.05),
                self._pose(math.pi / 2.0 + 0.10),
            ),
        )

        analysis = analyze_temporal_pose_motion_boundary(
            outgoing=outgoing,
            incoming=incoming,
            thresholds=self._thresholds(),
        )

        self.assertGreater(
            analysis.outgoing_boundary_motion_magnitude,
            self._thresholds().minimum_boundary_motion_magnitude,
        )
        self.assertGreater(
            analysis.incoming_boundary_motion_magnitude,
            self._thresholds().minimum_boundary_motion_magnitude,
        )
        self.assertFalse(analysis.qualifies)
        self.assertLess(analysis.endpoint_pose_cosine, 0.20)
        self.assertIn(
            "endpoint_pose_cosine_below_threshold",
            analysis.failure_reasons,
        )

    def test_missing_nonfinite_and_shape_mismatch_fail_closed(self) -> None:
        valid_outgoing, valid_incoming = self._same_phase_windows()
        cases = (
            (
                None,
                valid_incoming,
                "missing_outgoing_window",
            ),
            (
                TemporalPoseWindow(
                    (-0.2, -0.1, 0.0),
                    (
                        self._pose(0.0),
                        (math.nan, 0.0, 0.0),
                        self._pose(0.1),
                    ),
                ),
                valid_incoming,
                "nonfinite_outgoing_embedding",
            ),
            (
                valid_outgoing,
                TemporalPoseWindow(
                    (0.0, 0.1, 0.2),
                    ((1.0, 0.0), (0.9, 0.1), (0.8, 0.2)),
                ),
                "cross_boundary_embedding_dimension_mismatch",
            ),
        )

        for outgoing, incoming, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                analysis = analyze_temporal_pose_motion_boundary(
                    outgoing=outgoing,
                    incoming=incoming,
                    thresholds=self._thresholds(),
                )
                self.assertFalse(analysis.input_valid)
                self.assertFalse(analysis.qualifies)
                self.assertFalse(analysis.dynamic_pose_phase_match)
                self.assertEqual(analysis.classification, INVALID_CLASSIFICATION)
                self.assertEqual(analysis.confidence, 0.0)
                self.assertIn(expected_reason, analysis.failure_reasons)

    def test_positive_embedding_scale_does_not_change_analysis(self) -> None:
        outgoing, incoming = self._same_phase_windows()
        scaled_outgoing, scaled_incoming = self._same_phase_windows(
            outgoing_scale=37.0,
            incoming_scale=0.125,
        )

        original = analyze_temporal_pose_motion_boundary(
            outgoing=outgoing,
            incoming=incoming,
            thresholds=self._thresholds(),
        )
        scaled = analyze_temporal_pose_motion_boundary(
            outgoing=scaled_outgoing,
            incoming=scaled_incoming,
            thresholds=self._thresholds(),
        )

        self.assertEqual(original.qualifies, scaled.qualifies)
        self.assertAlmostEqual(
            original.outgoing_boundary_motion_magnitude,
            scaled.outgoing_boundary_motion_magnitude,
            places=12,
        )
        self.assertAlmostEqual(
            original.incoming_boundary_motion_magnitude,
            scaled.incoming_boundary_motion_magnitude,
            places=12,
        )
        self.assertAlmostEqual(
            original.endpoint_pose_cosine,
            scaled.endpoint_pose_cosine,
            places=12,
        )
        self.assertAlmostEqual(
            original.velocity_direction_cosine,
            scaled.velocity_direction_cosine,
            places=12,
        )
        self.assertAlmostEqual(original.confidence, scaled.confidence, places=12)

    def test_public_boundary_analysis_redacts_raw_velocity_vectors(self) -> None:
        def pose(first: float, second: float) -> tuple[float, ...]:
            return (first, second, *(0.0 for _ in range(58)))

        analysis = analyze_temporal_pose_motion_boundary(
            outgoing=TemporalPoseWindow(
                (-0.1, 0.0),
                (pose(1.0, 0.0), pose(0.995, 0.1)),
            ),
            incoming=TemporalPoseWindow(
                (0.0, 0.1),
                (pose(0.995, 0.1), pose(0.98, 0.2)),
            ),
            thresholds=self._thresholds(),
        )
        self.assertEqual(60, len(analysis.outgoing_normalized_velocity or ()))
        self.assertEqual(60, len(analysis.incoming_normalized_velocity or ()))

        payload = analysis.to_dict()
        velocity = payload["velocity_evidence"]
        self.assertTrue(velocity["outgoing_observed"])
        self.assertTrue(velocity["incoming_observed"])
        self.assertEqual(60, velocity["feature_dimension"])
        for key in (
            "outgoing_content_sha256",
            "incoming_content_sha256",
            "pair_content_sha256",
        ):
            self.assertEqual(64, len(velocity[key]))

        def assert_redacted(value: object) -> None:
            if isinstance(value, dict):
                for key, nested in value.items():
                    self.assertNotIn("normalized_velocity", key)
                    self.assertNotIn("pose_embedding", key)
                    self.assertNotIn("embeddings", key)
                    assert_redacted(nested)
            elif isinstance(value, list):
                self.assertNotIn(len(value), {60, 120})
                for nested in value:
                    assert_redacted(nested)

        assert_redacted(payload)

    def test_thresholds_are_required_and_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "motion magnitude"):
            TemporalPoseMotionThresholds(
                minimum_boundary_motion_magnitude=0.0,
                minimum_endpoint_pose_cosine=0.90,
                minimum_velocity_direction_cosine=0.80,
                minimum_phase_compatibility=0.90,
                pose_norm_epsilon=1e-9,
            )


if __name__ == "__main__":
    unittest.main()
