from __future__ import annotations

import json
import math
import unittest

import numpy as np

from crosscut.timewarp import TimeWarpConfig, estimate_local_timewarp


def _synthetic_reference(duration_seconds: int, sample_rate: int) -> np.ndarray:
    """Deterministic transient-rich audio with no repeating song section."""

    rng = np.random.default_rng(991)
    sample_count = duration_seconds * sample_rate
    signal = 0.004 * rng.standard_normal(sample_count)
    time_seconds = 0.2
    index = 0
    while time_seconds < duration_seconds - 0.5:
        time_seconds += 0.18 + 0.053 * ((index * 7) % 13) + 0.017 * (index % 4)
        start = int(round(time_seconds * sample_rate))
        width = 40 + (index * 19) % 100
        if start + width >= sample_count:
            break
        local_time = np.arange(width, dtype=np.float64) / sample_rate
        frequency = 60.0 + (index * 37) % 320
        burst = (
            np.hanning(width)
            * np.sin(2.0 * math.pi * frequency * local_time)
            * (0.4 + 0.08 * (index % 6))
        )
        signal[start : start + width] += burst
        index += 1
    return signal.astype(np.float32)


def _source_for_mapping(
    reference: np.ndarray,
    sample_rate: int,
    mapped_source_times: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    source_duration = float(mapped_source_times[-1]) + 1.0
    source_times = np.arange(
        int(math.ceil(source_duration * sample_rate)), dtype=np.float64
    ) / sample_rate
    source = np.interp(
        source_times,
        mapped_source_times,
        reference,
        left=0.0,
        right=0.0,
    )
    # Exercise the spectral-flux path rather than waveform identity.
    source = np.tanh(1.5 * source)
    source += np.random.default_rng(seed).normal(0.0, 0.002, source.size)
    return source.astype(np.float32)


class LocalTimeWarpTests(unittest.TestCase):
    sample_rate = 1_000

    def test_recovers_linear_clock_drift_with_bounded_monotonic_map(self) -> None:
        duration = 96
        reference = _synthetic_reference(duration, self.sample_rate)
        master_times = np.arange(reference.size, dtype=np.float64) / self.sample_rate
        global_offset = 0.65
        clock_rate = 1.0032
        true_source_times = global_offset + clock_rate * master_times
        source = _source_for_mapping(
            reference, self.sample_rate, true_source_times, seed=2
        )

        config = TimeWarpConfig()
        result = estimate_local_timewarp(
            reference,
            source,
            sample_rate=self.sample_rate,
            global_offset_seconds=global_offset,
            config=config,
        )

        self.assertFalse(result.scalar_fallback)
        self.assertGreaterEqual(len(result.accepted_anchors), 6)
        self.assertTrue(
            all(
                config.minimum_slope <= slope <= config.maximum_slope
                for slope in result.segment_slopes
            )
        )
        for master_time in (15.0, 48.0, 82.0):
            expected = global_offset + clock_rate * master_time
            self.assertAlmostEqual(
                result.master_to_source(master_time), expected, delta=0.025
            )
        self.assertIsNotNone(result.residuals.p50_seconds)
        self.assertIsNotNone(result.residuals.p95_seconds)
        self.assertIsNotNone(result.residuals.maximum_seconds)
        self.assertLess(result.residuals.maximum_seconds or 1.0, 0.025)

        payload = result.to_dict()
        self.assertEqual(json.loads(result.to_json()), payload)
        self.assertEqual(
            payload["accepted_anchor_count"], len(payload["accepted_anchors"])
        )
        self.assertEqual(
            payload["rejected_anchor_count"], len(payload["rejected_anchors"])
        )

    def test_recovers_piecewise_drift_better_than_scalar_offset(self) -> None:
        duration = 108
        reference = _synthetic_reference(duration, self.sample_rate)
        master_times = np.arange(reference.size, dtype=np.float64) / self.sample_rate
        global_offset = 0.45
        pivot = 54.0
        true_source_times = np.where(
            master_times <= pivot,
            global_offset + 1.003 * master_times,
            global_offset
            + 1.003 * pivot
            + 0.9975 * (master_times - pivot),
        )
        source = _source_for_mapping(
            reference, self.sample_rate, true_source_times, seed=4
        )

        result = estimate_local_timewarp(
            reference,
            source,
            sample_rate=self.sample_rate,
            global_offset_seconds=global_offset,
        )

        self.assertFalse(result.scalar_fallback)
        self.assertGreaterEqual(len(result.accepted_anchors), 7)
        local_errors: list[float] = []
        scalar_errors: list[float] = []
        for master_time in (15.0, 45.0, 60.0, 90.0, 103.0):
            expected = float(np.interp(master_time, master_times, true_source_times))
            local_errors.append(abs(result.master_to_source(master_time) - expected))
            scalar_errors.append(abs(master_time + global_offset - expected))
        self.assertLess(max(local_errors), 0.025)
        self.assertLess(sum(local_errors), 0.10 * sum(scalar_errors))
        self.assertGreater(max(result.segment_slopes), 1.001)
        self.assertLess(min(result.segment_slopes), 0.999)

    def test_drift_outside_hard_slope_bound_uses_scalar_fallback(self) -> None:
        duration = 96
        reference = _synthetic_reference(duration, self.sample_rate)
        master_times = np.arange(reference.size, dtype=np.float64) / self.sample_rate
        global_offset = 0.65
        true_source_times = global_offset + 1.0032 * master_times
        source = _source_for_mapping(
            reference, self.sample_rate, true_source_times, seed=5
        )

        result = estimate_local_timewarp(
            reference,
            source,
            sample_rate=self.sample_rate,
            global_offset_seconds=global_offset,
            config=TimeWarpConfig(maximum_drift_rate=0.001),
        )

        self.assertTrue(result.scalar_fallback)
        self.assertEqual(
            result.fallback_reason, "insufficient_slope_bounded_anchors"
        )
        self.assertEqual(len(result.accepted_anchors), 0)
        self.assertAlmostEqual(result.master_to_source(80.0), 80.65)

    def test_unrelated_audio_rejects_low_confidence_anchors_and_falls_back(self) -> None:
        duration = 72
        sample_count = duration * self.sample_rate
        first = np.random.default_rng(10).normal(0.0, 1.0, sample_count)
        second = np.random.default_rng(20).normal(0.0, 1.0, sample_count)
        reference = np.convolve(first, np.hanning(31), mode="same").astype(
            np.float32
        )
        source = np.convolve(second, np.hanning(37), mode="same").astype(np.float32)

        result = estimate_local_timewarp(
            reference,
            source,
            sample_rate=self.sample_rate,
            global_offset_seconds=-0.4,
        )

        self.assertTrue(result.scalar_fallback)
        self.assertEqual(result.fallback_reason, "insufficient_confident_anchors")
        self.assertEqual(len(result.accepted_anchors), 0)
        self.assertGreater(len(result.rejected_anchors), 0)
        self.assertTrue(
            all(
                anchor.rejection_reason
                in {"low_correlation_score", "ambiguous_correlation_peak"}
                for anchor in result.rejected_anchors
            )
        )
        self.assertAlmostEqual(result.master_to_source(31.25), 30.85)
        self.assertIsNone(result.residuals.p95_seconds)

    def test_short_audio_uses_exact_scalar_contract(self) -> None:
        reference = _synthetic_reference(12, self.sample_rate)
        source = np.pad(reference, (725, 0))
        result = estimate_local_timewarp(
            reference,
            source,
            sample_rate=self.sample_rate,
            global_offset_seconds=0.725,
        )

        self.assertTrue(result.scalar_fallback)
        self.assertEqual(
            result.fallback_reason, "insufficient_duration_for_local_windows"
        )
        for master_time in (0.0, 4.25, 11.5):
            self.assertAlmostEqual(
                result.master_to_source(master_time), master_time + 0.725
            )

    def test_configuration_enforces_overlapping_twenty_to_thirty_second_windows(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 20 and 30"):
            TimeWarpConfig(window_seconds=18.0)
        with self.assertRaisesRegex(ValueError, "overlap"):
            TimeWarpConfig(window_seconds=24.0, hop_seconds=24.0)
        with self.assertRaisesRegex(ValueError, "maximum_drift_rate"):
            TimeWarpConfig(maximum_drift_rate=1.0)


if __name__ == "__main__":
    unittest.main()
