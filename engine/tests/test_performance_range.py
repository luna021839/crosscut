from __future__ import annotations

import json
import unittest

import numpy as np

from crosscut.performance_range import estimate_shared_performance_range
from crosscut.timewarp import (
    TimeWarpConfig,
    TimeWarpControlPoint,
    TimeWarpMap,
    TimeWarpResidualDiagnostics,
)


def _synthetic_sources(
    *,
    sample_rate: int,
    duration_seconds: float,
    source_count: int,
    shared_start_seconds: float | None,
    shared_end_seconds: float | None,
) -> dict[int, np.ndarray]:
    sample_count = int(round(duration_seconds * sample_rate))
    shared_rng = np.random.default_rng(9_271)
    shared = shared_rng.normal(0.0, 0.55, sample_count).astype(np.float32)
    # Deterministic transient accents make the shared section music-like while
    # retaining the exact same rhythmic structure across broadcaster mixes.
    pulse = np.zeros(sample_count, dtype=np.float32)
    for center in range(0, sample_count, max(1, sample_rate // 2)):
        width = min(sample_count - center, max(1, sample_rate // 40))
        pulse[center : center + width] += np.linspace(1.2, 0.0, width, dtype=np.float32)
    shared += pulse

    start = (
        0
        if shared_start_seconds is None
        else int(round(shared_start_seconds * sample_rate))
    )
    end = (
        sample_count
        if shared_end_seconds is None
        else int(round(shared_end_seconds * sample_rate))
    )
    result: dict[int, np.ndarray] = {}
    for source_index in range(source_count):
        rng = np.random.default_rng(40_000 + source_index)
        signal = rng.normal(0.0, 0.55, sample_count).astype(np.float32)
        if start < end:
            mix_noise = rng.normal(0.0, 0.025, end - start).astype(np.float32)
            gain = 0.72 + source_index * 0.035
            signal[start:end] = shared[start:end] * gain + mix_noise
        result[source_index] = signal
    return result


class SharedPerformanceRangeTests(unittest.TestCase):
    def test_shared_song_excludes_unrelated_intro_and_outro_but_keeps_ending_pose(
        self,
    ) -> None:
        sample_rate = 1_000
        sources = _synthetic_sources(
            sample_rate=sample_rate,
            duration_seconds=90.0,
            source_count=9,
            shared_start_seconds=10.0,
            shared_end_seconds=70.0,
        )

        estimate = estimate_shared_performance_range(
            sources,
            sample_rate=sample_rate,
            common_start_seconds=0.0,
            common_end_seconds=90.0,
            source_indices=tuple(range(9)),
            reference_source_index=0,
            audio_offsets=(0.0,) * 9,
            reference_scene_cut_master_seconds=(9.5, 72.0, 80.0),
        )

        self.assertTrue(estimate.evidence_strong)
        self.assertTrue(estimate.applied)
        self.assertAlmostEqual(estimate.final_start_seconds, 9.5)
        self.assertAlmostEqual(estimate.final_end_seconds, 72.0)
        self.assertEqual(
            estimate.start_boundary_basis,
            "last_reference_hard_cut_before_audio_consensus",
        )
        self.assertEqual(
            estimate.end_boundary_basis,
            "first_reference_hard_cut_after_audio_consensus",
        )
        self.assertGreaterEqual(estimate.audio_consensus_start_seconds or 0.0, 9.5)
        self.assertLessEqual(estimate.audio_consensus_start_seconds or 99.0, 11.5)
        self.assertGreaterEqual(estimate.audio_consensus_end_seconds or 0.0, 68.5)
        self.assertLessEqual(estimate.audio_consensus_end_seconds or 99.0, 70.5)
        payload = estimate.to_dict()
        json.dumps(payload, allow_nan=False)
        self.assertFalse(payload["privacy"]["raw_pcm_persisted"])
        self.assertFalse(payload["privacy"]["spectral_flux_vectors_persisted"])
        self.assertEqual(
            payload["selected_end_scene_cut_seconds"],
            72.0,
        )

    def test_weak_unrelated_audio_retains_original_range(self) -> None:
        sample_rate = 1_000
        sources = _synthetic_sources(
            sample_rate=sample_rate,
            duration_seconds=90.0,
            source_count=9,
            shared_start_seconds=45.0,
            shared_end_seconds=45.0,
        )

        estimate = estimate_shared_performance_range(
            sources,
            sample_rate=sample_rate,
            common_start_seconds=0.0,
            common_end_seconds=90.0,
            source_indices=tuple(range(9)),
            reference_source_index=0,
            audio_offsets=(0.0,) * 9,
            reference_scene_cut_master_seconds=(10.0, 70.0),
        )

        self.assertFalse(estimate.evidence_strong)
        self.assertFalse(estimate.applied)
        self.assertEqual(estimate.final_start_seconds, 0.0)
        self.assertEqual(estimate.final_end_seconds, 90.0)
        self.assertEqual(
            estimate.reason,
            "insufficient_dense_shared_audio_cluster",
        )

    def test_song_filling_common_range_retains_i_am_length(self) -> None:
        sample_rate = 1_000
        sources = _synthetic_sources(
            sample_rate=sample_rate,
            duration_seconds=200.0,
            source_count=9,
            shared_start_seconds=None,
            shared_end_seconds=None,
        )
        common_start = 14.062494
        common_end = 197.044535

        estimate = estimate_shared_performance_range(
            sources,
            sample_rate=sample_rate,
            common_start_seconds=common_start,
            common_end_seconds=common_end,
            source_indices=tuple(range(9)),
            reference_source_index=0,
            audio_offsets=(0.0,) * 9,
            reference_scene_cut_master_seconds=(),
        )

        self.assertTrue(estimate.evidence_strong)
        self.assertFalse(estimate.applied)
        self.assertAlmostEqual(estimate.final_start_seconds, common_start)
        self.assertAlmostEqual(estimate.final_end_seconds, common_end)
        self.assertAlmostEqual(estimate.final_duration_seconds, 182.982041)
        self.assertEqual(
            estimate.end_boundary_basis,
            "original_end_within_ending_pose_allowance",
        )

    def test_missing_peer_pcm_is_weak_and_never_trims(self) -> None:
        sample_rate = 1_000
        sources = _synthetic_sources(
            sample_rate=sample_rate,
            duration_seconds=60.0,
            source_count=3,
            shared_start_seconds=None,
            shared_end_seconds=None,
        )

        estimate = estimate_shared_performance_range(
            sources,
            sample_rate=sample_rate,
            common_start_seconds=0.0,
            common_end_seconds=60.0,
            source_indices=tuple(range(9)),
            reference_source_index=0,
            audio_offsets=(0.0,) * 9,
        )

        self.assertFalse(estimate.evidence_strong)
        self.assertFalse(estimate.applied)
        self.assertEqual(estimate.reason, "insufficient_decoded_peer_sources")
        self.assertEqual(estimate.final_end_seconds, 60.0)

    def test_checked_timewarp_maps_every_window_to_the_source_clock(self) -> None:
        sample_rate = 1_000
        duration_seconds = 70.0
        sample_count = int(round(duration_seconds * sample_rate))
        reference = _synthetic_sources(
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            source_count=1,
            shared_start_seconds=None,
            shared_end_seconds=None,
        )[0]
        reference_times = np.arange(sample_count, dtype=np.float64) / sample_rate
        sources: dict[int, np.ndarray] = {0: reference}
        timewarps: dict[int, TimeWarpMap] = {}
        offsets: dict[int, float] = {0: 0.0}
        for source_index in range(1, 9):
            offset = 0.18 + source_index * 0.025
            rate = 1.002 + source_index * 0.0002
            source_duration = offset + rate * duration_seconds + 0.5
            source_times = (
                np.arange(
                    int(round(source_duration * sample_rate)),
                    dtype=np.float64,
                )
                / sample_rate
            )
            master_times = (source_times - offset) / rate
            sources[source_index] = np.interp(
                master_times,
                reference_times,
                reference,
                left=0.0,
                right=0.0,
            ).astype(np.float32)
            offsets[source_index] = offset
            timewarps[source_index] = TimeWarpMap(
                global_offset_seconds=offset,
                master_duration_seconds=duration_seconds,
                source_duration_seconds=source_duration,
                control_points=(
                    TimeWarpControlPoint(0.0, offset),
                    TimeWarpControlPoint(
                        duration_seconds,
                        offset + rate * duration_seconds,
                    ),
                ),
                anchors=(),
                residuals=TimeWarpResidualDiagnostics(None, None, None, 0),
                config=TimeWarpConfig(maximum_drift_rate=0.01),
                scalar_fallback=False,
                fallback_reason=None,
            )

        estimate = estimate_shared_performance_range(
            sources,
            sample_rate=sample_rate,
            common_start_seconds=0.0,
            common_end_seconds=duration_seconds,
            source_indices=tuple(range(9)),
            reference_source_index=0,
            audio_offsets=offsets,
            source_timewarps=timewarps,
        )
        scalar_only = estimate_shared_performance_range(
            sources,
            sample_rate=sample_rate,
            common_start_seconds=0.0,
            common_end_seconds=duration_seconds,
            source_indices=tuple(range(9)),
            reference_source_index=0,
            audio_offsets=offsets,
        )

        self.assertTrue(estimate.evidence_strong)
        self.assertFalse(estimate.applied)
        self.assertEqual(estimate.final_start_seconds, 0.0)
        self.assertEqual(estimate.final_end_seconds, duration_seconds)
        self.assertGreater(estimate.selected_cluster_window_count, 120)
        self.assertFalse(scalar_only.evidence_strong)
        self.assertEqual(
            scalar_only.reason,
            "insufficient_dense_shared_audio_cluster",
        )


if __name__ == "__main__":
    unittest.main()
