from __future__ import annotations

from itertools import permutations
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from crosscut.source_look_match import (
    LookMetricVector,
    SourceLookMatchConfig,
    analyze_source_look_match,
    match_source_look_samples,
    measure_frame_look,
)
from crosscut.timewarp import (
    TimeWarpConfig,
    TimeWarpControlPoint,
    TimeWarpMap,
    TimeWarpResidualDiagnostics,
)


def _metric(
    luma: float,
    contrast: float,
    saturation: float,
    temperature: float,
) -> LookMetricVector:
    return LookMetricVector(luma, contrast, saturation, temperature)


def _timewarp(
    global_offset: float,
    *,
    rate: float = 1.0,
    scalar_fallback: bool = False,
) -> TimeWarpMap:
    return TimeWarpMap(
        global_offset_seconds=global_offset,
        master_duration_seconds=120.0,
        source_duration_seconds=max(1.0, global_offset + rate * 120.0),
        control_points=(
            ()
            if scalar_fallback
            else (
                TimeWarpControlPoint(0.0, global_offset),
                TimeWarpControlPoint(120.0, global_offset + rate * 120.0),
            )
        ),
        anchors=(),
        residuals=TimeWarpResidualDiagnostics(None, None, None, 0),
        config=TimeWarpConfig(maximum_drift_rate=0.01),
        scalar_fallback=scalar_fallback,
        fallback_reason="synthetic_scalar" if scalar_fallback else None,
    )


class _FakeCapture:
    def __init__(self, source_index: int) -> None:
        self.source_index = source_index
        self.seeks: list[float] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def set(self, property_id: int, value: float) -> bool:
        if property_id == cv2.CAP_PROP_POS_MSEC:
            self.seeks.append(value)
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        frame = np.full((8, 8, 3), self.source_index, dtype=np.uint8)
        return (True, frame)

    def release(self) -> None:
        self.released = True


class SourceLookMatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SourceLookMatchConfig(
            sample_fractions=(0.1, 0.25, 0.4, 0.55, 0.7, 0.85),
            minimum_decoded_samples=6,
        )
        self.times = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)

    def test_stable_reference_residuals_produce_small_bounded_adjustments(self) -> None:
        reference = tuple(
            _metric(0.44 + index * 0.01, 0.36, 0.40, 0.02)
            for index in range(6)
        )
        darker_cooler_oversaturated = tuple(
            _metric(item.luma - 0.04, item.contrast / 1.08, item.saturation * 1.08, -0.08)
            for item in reference
        )
        near_reference = tuple(
            _metric(item.luma + 0.002, item.contrast, item.saturation, 0.021)
            for item in reference
        )

        result = match_source_look_samples(
            (reference, darker_cooler_oversaturated, near_reference),
            canonical_sample_times_seconds=self.times,
            reference_source=0,
            config=self.config,
        )

        self.assertEqual(result.target_mode, "explicit_reference")
        self.assertTrue(result.adjustment_for_source(0).is_identity)
        repair = result.adjustment_for_source(1)
        self.assertGreater(repair.brightness, 0.0)
        self.assertGreater(repair.contrast, 1.0)
        self.assertLess(repair.saturation, 1.0)
        self.assertGreater(repair.temperature_balance, 0.0)
        self.assertLessEqual(abs(repair.brightness), 0.045)
        self.assertLessEqual(abs(repair.contrast - 1.0), 0.05)
        self.assertLessEqual(abs(repair.saturation - 1.0), 0.05)
        self.assertLessEqual(abs(repair.temperature_balance), 0.025)
        self.assertEqual(result.evidence[1].reason, "stable_bounded_static_match")
        self.assertTrue(result.adjustment_for_source(2).is_identity)

    def test_consensus_large_stable_luma_residual_uses_bounded_directional_step(
        self,
    ) -> None:
        reference = tuple(
            _metric(0.50, 0.40, 0.40, 0.0) for _ in range(6)
        )
        dark = tuple(
            _metric(0.20, 0.40, 0.40, 0.0) for _ in range(6)
        )
        bright = tuple(
            _metric(0.80, 0.40, 0.40, 0.0) for _ in range(6)
        )

        result = match_source_look_samples(
            (reference, dark, bright),
            canonical_sample_times_seconds=self.times,
            reference_source=None,
            config=self.config,
        )

        self.assertEqual(result.to_dict()["version"], 3)
        self.assertEqual(result.target_mode, "cross_source_consensus")
        self.assertEqual(
            result.adjustment_for_source(1).brightness,
            self.config.maximum_brightness,
        )
        self.assertEqual(
            result.adjustment_for_source(2).brightness,
            -self.config.maximum_brightness,
        )
        self.assertEqual(
            dict(result.evidence[1].component_status)["luma"],
            "bounded_directional_step_applied_large_stable_residual",
        )
        self.assertEqual(
            dict(result.evidence[2].component_status)["luma"],
            "bounded_directional_step_applied_large_stable_residual",
        )

    def test_explicit_reference_large_residual_keeps_plausibility_cap(self) -> None:
        reference = tuple(
            _metric(0.50, 0.40, 0.40, 0.0) for _ in range(6)
        )
        dark = tuple(
            _metric(0.20, 0.40, 0.40, 0.0) for _ in range(6)
        )
        result = match_source_look_samples(
            (reference, dark),
            canonical_sample_times_seconds=self.times,
            reference_source=0,
            config=self.config,
        )

        self.assertTrue(result.adjustment_for_source(1).is_identity)
        self.assertEqual(
            dict(result.evidence[1].component_status)["luma"],
            "required_correction_exceeds_safe_evidence_range",
        )

    def test_consensus_is_order_invariant_and_does_not_pin_visual_outlier(
        self,
    ) -> None:
        source_values = (0.75, 0.49, 0.50, 0.51)
        for order in permutations(range(len(source_values))):
            samples = tuple(
                tuple(
                    _metric(source_values[index], 0.40, 0.40, 0.0)
                    for _ in range(6)
                )
                for index in order
            )
            result = match_source_look_samples(
                samples,
                canonical_sample_times_seconds=self.times,
                reference_source=None,
                config=self.config,
            )
            self.assertEqual(result.target_mode, "cross_source_consensus")
            assert result.target_median is not None
            self.assertAlmostEqual(result.target_median.luma, 0.505)
            outlier_position = order.index(0)
            self.assertLess(
                result.adjustment_for_source(outlier_position).brightness,
                0.0,
            )

    def test_volatile_or_missing_evidence_is_identity(self) -> None:
        reference = tuple(_metric(0.50, 0.38, 0.40, 0.0) for _ in range(6))
        volatile = tuple(
            _metric(0.35 if index % 2 else 0.65, 0.38, 0.40, 0.0)
            for index in range(6)
        )
        missing = tuple(
            _metric(0.46, 0.38, 0.40, 0.0) if index < 5 else None
            for index in range(6)
        )

        result = match_source_look_samples(
            (reference, volatile, missing),
            canonical_sample_times_seconds=self.times,
            reference_source=0,
            config=self.config,
        )

        self.assertTrue(result.adjustment_for_source(1).is_identity)
        self.assertEqual(
            result.evidence[1].reason,
            "unstable_synchronized_look_residuals",
        )
        self.assertTrue(result.adjustment_for_source(2).is_identity)
        self.assertEqual(
            result.evidence[2].reason,
            "insufficient_decoded_source_evidence",
        )

    def test_each_look_component_fails_closed_without_suppressing_stable_luma(self) -> None:
        reference = tuple(
            _metric(0.50, 0.38, 0.42, 0.02) for _ in range(6)
        )
        source = tuple(
            _metric(
                0.45,
                0.38,
                0.18 if index % 2 else 0.70,
                -0.30 if index % 2 else 0.32,
            )
            for index in range(6)
        )

        result = match_source_look_samples(
            (reference, source),
            canonical_sample_times_seconds=self.times,
            reference_source=0,
            config=self.config,
        )

        evidence = result.evidence[1]
        self.assertFalse(evidence.stable)
        self.assertEqual(evidence.reason, "partial_stable_bounded_static_match")
        self.assertGreater(evidence.adjustment.brightness, 0.0)
        self.assertEqual(evidence.adjustment.contrast, 1.0)
        self.assertEqual(evidence.adjustment.saturation, 1.0)
        self.assertEqual(evidence.adjustment.temperature_balance, 0.0)
        status = dict(evidence.component_status)
        self.assertEqual(status["luma"], "bounded_adjustment_applied")
        self.assertEqual(status["saturation"], "unstable_synchronized_residual")
        self.assertEqual(status["temperature"], "unstable_synchronized_residual")
        self.assertTrue(result.to_dict()["component_wise_fail_closed"])

    def test_unavailable_reference_falls_back_to_consensus(self) -> None:
        unavailable = (None, None, None, None, None, None)
        neutral = tuple(_metric(0.50, 0.40, 0.40, 0.0) for _ in range(6))
        dark = tuple(_metric(0.46, 0.40, 0.40, 0.0) for _ in range(6))
        bright = tuple(_metric(0.54, 0.40, 0.40, 0.0) for _ in range(6))

        result = match_source_look_samples(
            (unavailable, neutral, dark, bright),
            canonical_sample_times_seconds=self.times,
            reference_source=0,
            config=self.config,
        )

        self.assertEqual(result.target_mode, "cross_source_consensus")
        self.assertIsNone(result.target_source_index)
        self.assertTrue(result.adjustment_for_source(0).is_identity)
        self.assertGreater(result.adjustment_for_source(2).brightness, 0.0)
        self.assertLess(result.adjustment_for_source(3).brightness, 0.0)

    def test_sampling_uses_one_canonical_grid_plus_each_source_offset(self) -> None:
        config = SourceLookMatchConfig(
            sample_fractions=(0.25, 0.5, 0.75),
            minimum_decoded_samples=3,
        )
        captures = [_FakeCapture(index) for index in range(3)]
        metrics = (
            _metric(0.50, 0.40, 0.40, 0.0),
            _metric(0.47, 0.40, 0.40, 0.0),
            _metric(0.53, 0.40, 0.40, 0.0),
        )

        with (
            patch(
                "crosscut.source_look_match.cv2.VideoCapture",
                side_effect=captures,
            ),
            patch(
                "crosscut.source_look_match.measure_frame_look",
                side_effect=lambda frame: metrics[int(frame[0, 0, 0])],
            ),
        ):
            result = analyze_source_look_match(
                (Path("zero.mp4"), Path("one.mp4"), Path("two.mp4")),
                source_offsets_seconds=(0.0, 0.75, -0.5),
                common_master_start=10.0,
                common_master_end=30.0,
                reference_source=0,
                config=config,
            )

        self.assertEqual(result.canonical_sample_times_seconds, (15.0, 20.0, 25.0))
        self.assertEqual(captures[0].seeks, [15000.0, 20000.0, 25000.0])
        self.assertEqual(captures[1].seeks, [15750.0, 20750.0, 25750.0])
        self.assertEqual(captures[2].seeks, [14500.0, 19500.0, 24500.0])
        self.assertTrue(all(capture.released for capture in captures))
        payload = result.to_dict()
        self.assertFalse(payload["raw_frames_persisted"])
        self.assertFalse(payload["learned_identity_model_used"])

    def test_sampling_uses_non_scalar_timewarp_and_validates_clock_inputs(self) -> None:
        config = SourceLookMatchConfig(
            sample_fractions=(0.25, 0.5, 0.75),
            minimum_decoded_samples=3,
        )
        sources = (Path("zero.mp4"), Path("one.mp4"), Path("two.mp4"))
        offsets = (0.0, 0.75, -0.5)
        nonlinear = _timewarp(0.75, rate=1.005)
        scalar = _timewarp(-0.5, scalar_fallback=True)
        captures = [_FakeCapture(index) for index in range(3)]
        metric = _metric(0.50, 0.40, 0.40, 0.0)

        with (
            patch(
                "crosscut.source_look_match.cv2.VideoCapture",
                side_effect=captures,
            ),
            patch(
                "crosscut.source_look_match.measure_frame_look",
                return_value=metric,
            ),
        ):
            analyze_source_look_match(
                sources,
                source_offsets_seconds=offsets,
                source_timewarps=(None, nonlinear, scalar),
                common_master_start=10.0,
                common_master_end=30.0,
                reference_source=0,
                config=config,
            )

        self.assertEqual(captures[0].seeks, [15000.0, 20000.0, 25000.0])
        for actual, expected in zip(
            captures[1].seeks,
            (15825.0, 20850.0, 25875.0),
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(captures[2].seeks, [14500.0, 19500.0, 24500.0])
        self.assertTrue(all(capture.released for capture in captures))

        with self.assertRaisesRegex(ValueError, "timewarps must match"):
            analyze_source_look_match(
                sources,
                source_offsets_seconds=offsets,
                source_timewarps=(None,),
                common_master_start=10.0,
                common_master_end=30.0,
                config=config,
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            analyze_source_look_match(
                sources,
                source_offsets_seconds=offsets,
                source_timewarps=(None, _timewarp(0.5), scalar),
                common_master_start=10.0,
                common_master_end=30.0,
                config=config,
            )

    def test_frame_metrics_are_robust_and_reject_unexposed_frames(self) -> None:
        ramp = np.linspace(45, 210, 320, dtype=np.float32)
        base = np.tile(ramp, (180, 1))
        frame = np.stack(
            (
                np.clip(base * 0.90, 0, 255),
                base,
                np.clip(base * 1.10, 0, 255),
            ),
            axis=2,
        ).astype(np.uint8)

        metric = measure_frame_look(frame)

        self.assertGreater(metric.luma, 0.30)
        self.assertGreater(metric.contrast, 0.25)
        self.assertGreater(metric.saturation, 0.05)
        self.assertGreater(metric.temperature_log_red_blue, 0.0)
        with self.assertRaisesRegex(ValueError, "reliable exposed pixels"):
            measure_frame_look(np.zeros((64, 64, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
