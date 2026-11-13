from __future__ import annotations

import json
import math
import unittest

import numpy as np

from crosscut.rhythm import (
    RhythmConfig,
    analyze_rhythm,
    spectral_flux_onset_envelope,
)


def _synthetic_music(
    *,
    bpm: float,
    phase_seconds: float,
    duration_seconds: float = 24.0,
    sample_rate: int = 8_000,
) -> np.ndarray:
    """Deterministic tones plus short beat transients and four-beat accents."""

    sample_count = int(round(duration_seconds * sample_rate))
    time = np.arange(sample_count, dtype=np.float64) / sample_rate
    signal = (
        0.08 * np.sin(2.0 * math.pi * 196.0 * time)
        + 0.05 * np.sin(2.0 * math.pi * 293.66 * time + 0.4)
        + 0.025 * np.sin(2.0 * math.pi * 440.0 * time) * (1.0 + 0.3 * np.sin(2.0 * math.pi * 0.37 * time))
    )
    rng = np.random.default_rng(1406)
    period = 60.0 / bpm
    beat_index = 0
    beat_time = phase_seconds
    burst_length = max(8, int(round(0.018 * sample_rate)))
    decay = np.exp(-np.arange(burst_length) / (0.0045 * sample_rate))
    while beat_time < duration_seconds:
        start = int(round(beat_time * sample_rate))
        end = min(sample_count, start + burst_length)
        amplitude = 0.95 if beat_index % 4 == 0 else 0.58
        noise = rng.standard_normal(burst_length) * decay
        signal[start:end] += amplitude * noise[: end - start]
        beat_index += 1
        beat_time = phase_seconds + beat_index * period
    return signal.astype(np.float32)


class SpectralFluxTests(unittest.TestCase):
    def test_onset_envelope_is_normalized_and_peaks_near_clicks(self) -> None:
        sample_rate = 8_000
        signal = _synthetic_music(bpm=120.0, phase_seconds=0.37, sample_rate=sample_rate)
        envelope = spectral_flux_onset_envelope(signal, sample_rate)

        self.assertEqual(len(envelope.values), 1 + (signal.size - 1) // envelope.hop_length)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in envelope.values))
        times = np.asarray(envelope.times_seconds)
        values = np.asarray(envelope.values)
        near_first_click = np.abs(times - 0.37) <= 0.05
        self.assertGreater(float(np.max(values[near_first_click])), 0.55)


class RhythmAnalysisTests(unittest.TestCase):
    def test_recovers_click_tempo_and_absolute_beat_phase(self) -> None:
        # 137 BPM does not line up with an integer onset-envelope lag.  It
        # exercises fractional period refinement as well as phase placement.
        bpm = 137.0
        phase = 0.37
        sample_rate = 8_000
        signal = _synthetic_music(
            bpm=bpm,
            phase_seconds=phase,
            sample_rate=sample_rate,
        )

        result = analyze_rhythm(signal, sample_rate)

        self.assertAlmostEqual(result.bpm, bpm, delta=0.8)
        self.assertGreater(result.tempo_confidence, 0.45)
        self.assertGreater(len(result.beats), 40)
        self.assertAlmostEqual(result.beats[0].time_seconds, phase, delta=0.035)
        for beat_index in (0, 7, 23, 39):
            expected = phase + beat_index * 60.0 / bpm
            self.assertAlmostEqual(
                result.beats[beat_index].time_seconds, expected, delta=0.045
            )

    def test_recovers_non_default_tempo_and_returns_4_8_beat_phrases(self) -> None:
        signal = _synthetic_music(bpm=96.0, phase_seconds=0.21)
        result = analyze_rhythm(signal, 8_000)

        self.assertAlmostEqual(result.bpm, 96.0, delta=0.9)
        self.assertEqual(
            {candidate.beat_count for candidate in result.phrase_candidates},
            {4, 8},
        )
        for candidate in result.phrase_candidates:
            self.assertEqual(
                candidate.end_beat_index - candidate.start_beat_index,
                candidate.beat_count,
            )
            self.assertAlmostEqual(
                candidate.duration_seconds,
                candidate.beat_count * 60.0 / result.bpm,
                delta=1e-9,
            )

    def test_result_is_json_friendly_and_rejects_silence(self) -> None:
        result = analyze_rhythm(_synthetic_music(bpm=128.0, phase_seconds=0.30), 8_000)
        payload = result.to_dict()

        self.assertEqual(json.loads(result.to_json()), payload)
        self.assertEqual(payload["bpm"], result.bpm)
        self.assertEqual(payload["beat_times_seconds"], list(result.beat_times_seconds))
        self.assertIn("values", payload["onset_envelope"])
        with self.assertRaisesRegex(ValueError, "insufficient rhythmic onset energy"):
            analyze_rhythm(np.zeros(8_000 * 6), 8_000)

    def test_configuration_validates_bpm_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_bpm"):
            RhythmConfig(min_bpm=180.0, max_bpm=70.0)


if __name__ == "__main__":
    unittest.main()
