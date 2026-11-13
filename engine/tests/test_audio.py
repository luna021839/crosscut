from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
import wave

import numpy as np

from crosscut.audio import (
    align_signal_set,
    extract_mono_pcm,
    normalized_cross_correlation_offset,
    spectral_flux_alignment_offset,
)
from crosscut.media import probe_media


def _distinct_signal(length: int, sample_rate: int) -> np.ndarray:
    """Deterministic, non-periodic synthetic music-like signal."""

    rng = np.random.default_rng(731)
    time = np.arange(length, dtype=np.float64) / sample_rate
    chirp = np.sin(2.0 * math.pi * (2.1 * time + 0.19 * time * time))
    pulses = (rng.random(length) > 0.965).astype(np.float64)
    pulse_tail = np.convolve(pulses, np.exp(-np.arange(25) / 5.0), mode="same")
    return chirp + 0.55 * pulse_tail + 0.04 * rng.standard_normal(length)


class NormalizedCorrelationTests(unittest.TestCase):
    def test_spectral_flux_alignment_survives_polarity_and_mix_change(self) -> None:
        sample_rate = 4_000
        rng = np.random.default_rng(20260823)
        reference = np.zeros(sample_rate * 12, dtype=np.float32)
        for index, position in enumerate((700, 2100, 3900, 6400, 9100, 12700, 17100, 22300, 29100, 35600)):
            width = 35 + index * 3
            reference[position : position + width] = np.hanning(width) * (0.5 + index / 20)
        reference += rng.normal(0, 0.003, reference.size).astype(np.float32)
        delay = int(1.35 * sample_rate)
        source = np.concatenate(
            (
                np.zeros(delay, dtype=np.float32),
                -0.55 * reference,
            )
        )
        source = np.tanh(source * 2.4).astype(np.float32)
        estimate = spectral_flux_alignment_offset(
            reference,
            source,
            sample_rate=sample_rate,
            max_offset_seconds=3.0,
            min_overlap_seconds=5.0,
        )
        self.assertAlmostEqual(estimate.offset_seconds, 1.35, delta=0.04)
        self.assertGreater(estimate.score, 0.65)
        self.assertEqual(estimate.backend, "spectral_flux_ncc")

    sample_rate = 200

    def test_finds_positive_delay_despite_gain_and_dc_offset(self) -> None:
        reference = _distinct_signal(4_000, self.sample_rate)
        delay = 137
        source = np.zeros_like(reference)
        source[delay:] = 2.7 * reference[:-delay] + 3.25

        result = normalized_cross_correlation_offset(
            reference,
            source,
            sample_rate=self.sample_rate,
            max_offset_seconds=1.0,
            min_overlap_seconds=8.0,
        )

        self.assertEqual(result.offset_samples, delay)
        self.assertAlmostEqual(result.offset_seconds, delay / self.sample_rate)
        self.assertGreater(result.score, 0.999)
        self.assertGreater(result.score_margin, 0.2)
        self.assertGreater(result.overlap_ratio, 0.9)

    def test_finds_negative_offset(self) -> None:
        reference = _distinct_signal(3_000, self.sample_rate)
        advance = 83
        source = np.zeros_like(reference)
        source[:-advance] = reference[advance:]

        result = normalized_cross_correlation_offset(
            reference,
            source,
            sample_rate=self.sample_rate,
            max_offset_seconds=1.0,
            min_overlap_seconds=6.0,
        )

        self.assertEqual(result.offset_samples, -advance)
        self.assertGreater(result.score, 0.999)

    def test_alignment_result_is_json_friendly_and_defaults_to_ive(self) -> None:
        reference = _distinct_signal(2_400, self.sample_rate)
        delay = 51
        delayed = np.zeros_like(reference)
        delayed[delay:] = reference[:-delay]

        result = align_signal_set(
            {"reference.mp4": reference, "music-bank.mp4": delayed},
            reference="reference.mp4",
            sample_rate=self.sample_rate,
            max_offset_seconds=0.5,
            min_overlap_seconds=5.0,
        )
        payload = result.to_dict()
        encoded = result.to_json()

        self.assertEqual(payload["group"], "IVE")
        self.assertEqual(payload["reference"], "reference.mp4")
        self.assertEqual(payload["alignments"][1]["offset_samples"], delay)
        self.assertEqual(json.loads(encoded), payload)

    def test_rejects_constant_signals(self) -> None:
        with self.assertRaisesRegex(ValueError, "no varying overlap"):
            normalized_cross_correlation_offset(
                np.ones(100), np.ones(100), sample_rate=100
            )

    def test_periodic_alias_reports_small_peak_margin(self) -> None:
        period = 200
        repeats = 20
        time = np.arange(period, dtype=np.float64) / self.sample_rate
        motif = np.sin(2 * math.pi * 7.0 * time) + 0.2 * np.sin(
            2 * math.pi * 13.0 * time
        )
        periodic = np.tile(motif, repeats)

        result = normalized_cross_correlation_offset(
            periodic,
            periodic,
            sample_rate=self.sample_rate,
            max_offset_seconds=4.0,
            min_overlap_seconds=5.0,
        )

        self.assertLess(result.score_margin, 0.01)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class FFmpegIntegrationTests(unittest.TestCase):
    def _write_stereo_wav(self, path: Path, sample_rate: int = 8_000) -> None:
        frames = bytearray()
        for index in range(sample_rate // 10):
            sample = int(12_000 * math.sin(2.0 * math.pi * 440.0 * index / sample_rate))
            frames.extend(struct.pack("<hh", sample, sample))
        with wave.open(str(path), "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(frames)

    def test_extracts_resampled_mono_float_pcm_and_probes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            wav_path = Path(temporary_directory) / "synthetic.wav"
            self._write_stereo_wav(wav_path)

            pcm = extract_mono_pcm(wav_path, sample_rate=4_000)
            metadata = probe_media(wav_path)

        self.assertEqual(pcm.dtype, np.float32)
        self.assertEqual(pcm.ndim, 1)
        self.assertLessEqual(abs(pcm.size - 400), 1)
        self.assertGreater(float(np.max(np.abs(pcm))), 0.25)
        self.assertEqual(len(metadata.audio_streams), 1)
        self.assertEqual(metadata.audio_streams[0].channels, 2)
        self.assertEqual(metadata.audio_streams[0].sample_rate, 8_000)
        self.assertEqual(json.loads(metadata.to_json())["audio_streams"][0]["channels"], 2)


if __name__ == "__main__":
    unittest.main()
