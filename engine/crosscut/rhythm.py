"""Model-free rhythm analysis for synchronized cross-cut editing.

The module deliberately depends only on NumPy.  It extracts a spectral-flux
onset envelope from mono PCM, estimates a global tempo in a configurable BPM
range, places a beat grid, and proposes regular four/eight-beat edit phrases.

Times in the public result are source-timeline seconds.  Every result dataclass
has a ``to_dict`` method whose output contains only JSON-native values; the
top-level result also provides ``to_json`` for orchestration and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class RhythmConfig:
    """Stable, sample-rate-independent rhythm-analysis settings.

    Window and hop sizes are expressed in seconds so the same configuration can
    be used for the engine's 8 kHz analysis PCM and for full-rate WAV input.
    The defaults give roughly 86 onset-envelope samples per second.
    """

    frame_duration_seconds: float = 0.0464
    hop_duration_seconds: float = 0.0116
    onset_local_average_seconds: float = 0.20
    onset_smoothing_seconds: float = 0.023
    log_compression: float = 10.0
    min_bpm: float = 70.0
    max_bpm: float = 180.0
    tempo_step_bpm: float = 0.25
    preferred_bpm: float = 120.0
    tempo_prior_weight: float = 0.02
    beat_tolerance_seconds: float = 0.060
    minimum_duration_seconds: float = 3.0
    phrase_beats: tuple[int, ...] = (4, 8)
    stft_batch_frames: int = 512

    def __post_init__(self) -> None:
        positive = (
            "frame_duration_seconds",
            "hop_duration_seconds",
            "onset_local_average_seconds",
            "onset_smoothing_seconds",
            "log_compression",
            "min_bpm",
            "max_bpm",
            "tempo_step_bpm",
            "preferred_bpm",
            "beat_tolerance_seconds",
            "minimum_duration_seconds",
        )
        if any(not math.isfinite(getattr(self, name)) for name in positive):
            raise ValueError("rhythm configuration values must be finite")
        if any(getattr(self, name) <= 0 for name in positive):
            raise ValueError("rhythm configuration values must be positive")
        if self.min_bpm >= self.max_bpm:
            raise ValueError("min_bpm must be below max_bpm")
        if not self.min_bpm <= self.preferred_bpm <= self.max_bpm:
            raise ValueError("preferred_bpm must be inside the tempo range")
        if not 0.0 <= self.tempo_prior_weight <= 1.0:
            raise ValueError("tempo_prior_weight must be between 0 and 1")
        if self.stft_batch_frames < 1:
            raise ValueError("stft_batch_frames must be at least 1")
        if not self.phrase_beats or any(
            isinstance(value, bool) or int(value) != value or int(value) < 2
            for value in self.phrase_beats
        ):
            raise ValueError("phrase_beats must contain integers of at least 2")
        object.__setattr__(
            self, "phrase_beats", tuple(dict.fromkeys(int(v) for v in self.phrase_beats))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_duration_seconds": self.frame_duration_seconds,
            "hop_duration_seconds": self.hop_duration_seconds,
            "onset_local_average_seconds": self.onset_local_average_seconds,
            "onset_smoothing_seconds": self.onset_smoothing_seconds,
            "log_compression": self.log_compression,
            "min_bpm": self.min_bpm,
            "max_bpm": self.max_bpm,
            "tempo_step_bpm": self.tempo_step_bpm,
            "preferred_bpm": self.preferred_bpm,
            "tempo_prior_weight": self.tempo_prior_weight,
            "beat_tolerance_seconds": self.beat_tolerance_seconds,
            "minimum_duration_seconds": self.minimum_duration_seconds,
            "phrase_beats": list(self.phrase_beats),
            "stft_batch_frames": self.stft_batch_frames,
        }


@dataclass(frozen=True, slots=True)
class OnsetEnvelope:
    """Spectral-flux samples and their PCM timing metadata."""

    values: tuple[float, ...]
    sample_rate: int
    frame_length: int
    hop_length: int

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    @property
    def times_seconds(self) -> tuple[float, ...]:
        return tuple(index * self.hop_length / self.sample_rate for index in range(len(self.values)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "frame_length": self.frame_length,
            "hop_length": self.hop_length,
            "frame_rate": self.frame_rate,
            "times_seconds": list(self.times_seconds),
            "values": list(self.values),
        }


@dataclass(frozen=True, slots=True)
class TempoEstimate:
    bpm: float
    period_seconds: float
    phase_seconds: float
    confidence: float

    def to_dict(self) -> dict[str, float]:
        return {
            "bpm": self.bpm,
            "period_seconds": self.period_seconds,
            "phase_seconds": self.phase_seconds,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class Beat:
    index: int
    time_seconds: float
    strength: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "index": self.index,
            "time_seconds": self.time_seconds,
            "strength": self.strength,
        }


@dataclass(frozen=True, slots=True)
class PhraseCandidate:
    """A regular edit span from one beat boundary to another.

    ``end_beat_index`` is the exclusive boundary, so a four-beat candidate
    beginning at beat 8 ends on the grid point at beat 12.
    """

    beat_count: int
    start_beat_index: int
    end_beat_index: int
    start_seconds: float
    end_seconds: float
    confidence: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    def to_dict(self) -> dict[str, int | float]:
        return {
            "beat_count": self.beat_count,
            "start_beat_index": self.start_beat_index,
            "end_beat_index": self.end_beat_index,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_seconds": self.duration_seconds,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class RhythmAnalysis:
    sample_rate: int
    duration_seconds: float
    onset_envelope: OnsetEnvelope
    tempo: TempoEstimate
    beats: tuple[Beat, ...]
    phrase_candidates: tuple[PhraseCandidate, ...]

    @property
    def bpm(self) -> float:
        return self.tempo.bpm

    @property
    def tempo_confidence(self) -> float:
        return self.tempo.confidence

    @property
    def beat_period_seconds(self) -> float:
        return self.tempo.period_seconds

    @property
    def beat_times_seconds(self) -> tuple[float, ...]:
        return tuple(beat.time_seconds for beat in self.beats)

    @property
    def beat_grid(self) -> tuple[Beat, ...]:
        """Alias used by timeline-building code."""

        return self.beats

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "bpm": self.bpm,
            "tempo_confidence": self.tempo_confidence,
            "beat_period_seconds": self.beat_period_seconds,
            "tempo": self.tempo.to_dict(),
            "onset_envelope": self.onset_envelope.to_dict(),
            "beat_times_seconds": list(self.beat_times_seconds),
            "beat_grid": [beat.to_dict() for beat in self.beats],
            "phrase_candidates": [item.to_dict() for item in self.phrase_candidates],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def _validated_pcm(pcm: ArrayLike, sample_rate: int) -> NDArray[np.float64]:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise TypeError("sample_rate must be an integer")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    signal = np.asarray(pcm, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("pcm must be a one-dimensional mono signal")
    if signal.size < 2:
        raise ValueError("pcm must contain at least two samples")
    if not np.all(np.isfinite(signal)):
        raise ValueError("pcm contains NaN or infinite samples")
    return np.ascontiguousarray(signal)


def _odd_width(seconds: float, frame_rate: float) -> int:
    width = max(1, int(round(seconds * frame_rate)))
    return width if width % 2 else width + 1


def _moving_average(values: NDArray[np.float64], width: int) -> NDArray[np.float64]:
    if width <= 1:
        return values.copy()
    radius = width // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(padded, dtype=np.float64))
    )
    return (cumulative[width:] - cumulative[:-width]) / width


def spectral_flux_onset_envelope(
    pcm: ArrayLike,
    sample_rate: int,
    *,
    config: RhythmConfig | None = None,
) -> OnsetEnvelope:
    """Extract a locally-normalized, half-wave spectral-flux envelope.

    STFT frames are centered on their reported timestamps.  Processing is
    batched to avoid materializing a song-length frame matrix.
    """

    settings = config or RhythmConfig()
    signal = _validated_pcm(pcm, sample_rate)
    frame_length = max(32, int(round(settings.frame_duration_seconds * sample_rate)))
    if frame_length % 2:
        frame_length += 1
    hop_length = max(1, int(round(settings.hop_duration_seconds * sample_rate)))
    frame_count = 1 + (signal.size - 1) // hop_length

    signal = signal - float(np.mean(signal))
    rms = float(np.sqrt(np.mean(signal * signal)))
    if rms > np.finfo(np.float64).eps:
        signal = signal / rms

    half_window = frame_length // 2
    final_center = (frame_count - 1) * hop_length
    right_padding = max(half_window, final_center + half_window - signal.size + 1)
    padded = np.pad(signal, (half_window, right_padding), mode="constant")
    window = np.hanning(frame_length).astype(np.float64)
    window_scale = max(float(np.sum(window)), np.finfo(np.float64).eps)
    bin_count = frame_length // 2 + 1
    flux = np.zeros(frame_count, dtype=np.float64)
    previous_spectrum = np.zeros(bin_count, dtype=np.float64)
    offsets = np.arange(frame_length, dtype=np.int64)

    for first in range(0, frame_count, settings.stft_batch_frames):
        last = min(frame_count, first + settings.stft_batch_frames)
        starts = np.arange(first, last, dtype=np.int64) * hop_length
        frames = padded[starts[:, None] + offsets[None, :]] * window[None, :]
        magnitudes = np.abs(np.fft.rfft(frames, axis=1)) / window_scale
        spectra = np.log1p(settings.log_compression * magnitudes)
        previous = np.vstack((previous_spectrum, spectra[:-1]))
        positive_change = np.maximum(0.0, spectra - previous)
        flux[first:last] = np.sqrt(np.mean(positive_change * positive_change, axis=1))
        previous_spectrum = spectra[-1]

    # The first frame has no real predecessor and otherwise acts like a false
    # onset whenever a recording begins with non-silent audio.
    flux[0] = 0.0
    frame_rate = sample_rate / hop_length
    baseline_width = _odd_width(settings.onset_local_average_seconds, frame_rate)
    flux = np.maximum(0.0, flux - _moving_average(flux, baseline_width))
    smooth_width = _odd_width(settings.onset_smoothing_seconds, frame_rate)
    flux = _moving_average(flux, smooth_width)

    positive = flux[flux > np.finfo(np.float64).eps]
    if positive.size:
        scale = float(np.quantile(positive, 0.95))
        if scale <= np.finfo(np.float64).eps:
            scale = float(np.max(positive))
        flux = np.clip(flux / scale, 0.0, 1.0)
    else:
        flux.fill(0.0)

    return OnsetEnvelope(
        values=tuple(float(value) for value in flux),
        sample_rate=int(sample_rate),
        frame_length=frame_length,
        hop_length=hop_length,
    )


def _normalized_autocorrelation(
    values: NDArray[np.float64], max_lag: int
) -> NDArray[np.float64]:
    centered = values - float(np.mean(values))
    fft_size = 1 << (2 * centered.size - 1).bit_length()
    spectrum = np.fft.rfft(centered, fft_size)
    dots = np.fft.irfft(spectrum * np.conjugate(spectrum), fft_size)[: max_lag + 1]
    squares = centered * centered
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(squares, dtype=np.float64))
    )
    lags = np.arange(max_lag + 1, dtype=np.int64)
    left_energy = cumulative[centered.size - lags]
    right_energy = cumulative[centered.size] - cumulative[lags]
    denominator = np.sqrt(np.maximum(0.0, left_energy * right_energy))
    result = np.zeros(max_lag + 1, dtype=np.float64)
    usable = denominator > np.finfo(np.float64).eps
    result[usable] = dots[usable] / denominator[usable]
    return np.clip(result, -1.0, 1.0)


def _interpolate(values: NDArray[np.float64], position: float) -> float:
    if position < 0.0 or position > values.size - 1:
        return 0.0
    lower = int(math.floor(position))
    upper = min(values.size - 1, lower + 1)
    fraction = position - lower
    return float(values[lower] * (1.0 - fraction) + values[upper] * fraction)


def _circular_window_sum(values: NDArray[np.float64], radius: int) -> NDArray[np.float64]:
    if radius <= 0:
        return values.copy()
    # A tolerance can never profitably cover more than half a beat phase.
    radius = min(radius, max(1, (values.size - 1) // 2))
    extended = np.concatenate((values[-radius:], values, values[:radius]))
    width = 2 * radius + 1
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(extended, dtype=np.float64))
    )
    return cumulative[width:] - cumulative[:-width]


def _phase_metrics(
    envelope: NDArray[np.float64],
    period_frames: float,
    tolerance_frames: float,
) -> tuple[float, float, float]:
    """Return phase (frames), mean beat strength, and envelope coverage."""

    phase_bins = max(8, int(math.ceil(period_frames * 4.0)))
    positions = np.arange(envelope.size, dtype=np.float64)
    bin_indices = np.floor(
        np.mod(positions, period_frames) * phase_bins / period_frames
    ).astype(np.int64)
    bin_indices = np.minimum(bin_indices, phase_bins - 1)
    folded = np.bincount(bin_indices, weights=envelope, minlength=phase_bins)
    tolerance_bins = max(1, int(round(tolerance_frames * phase_bins / period_frames)))
    phase_profile = _circular_window_sum(folded, tolerance_bins)
    best_bin = int(np.argmax(phase_profile))
    phase_frames = (best_bin + 0.5) * period_frames / phase_bins

    # Pull the phase to the weighted center of nearby onset samples.  This
    # refines the quarter-frame phase histogram without expensive resampling.
    residuals = np.mod(
        positions - phase_frames + period_frames / 2.0, period_frames
    ) - period_frames / 2.0
    nearby = np.abs(residuals) <= tolerance_frames
    nearby_weight = float(np.sum(envelope[nearby]))
    if nearby_weight > np.finfo(np.float64).eps:
        phase_frames = float(
            np.mod(
                phase_frames
                + np.sum(residuals[nearby] * envelope[nearby]) / nearby_weight,
                period_frames,
            )
        )

    first_index = int(math.ceil(-phase_frames / period_frames))
    last_index = int(math.floor((envelope.size - 1 - phase_frames) / period_frames))
    predicted = phase_frames + np.arange(first_index, last_index + 1) * period_frames
    radius = max(1, int(math.ceil(tolerance_frames)))
    strengths: list[float] = []
    for position in predicted:
        center = int(round(float(position)))
        start = max(0, center - radius)
        end = min(envelope.size, center + radius + 1)
        strengths.append(float(np.max(envelope[start:end])) if start < end else 0.0)
    mean_strength = float(np.mean(strengths)) if strengths else 0.0

    residuals = np.mod(
        positions - phase_frames + period_frames / 2.0, period_frames
    ) - period_frames / 2.0
    captured = float(np.sum(envelope[np.abs(residuals) <= tolerance_frames]))
    total = float(np.sum(envelope))
    coverage = captured / total if total > np.finfo(np.float64).eps else 0.0
    return phase_frames, min(1.0, mean_strength), min(1.0, coverage)


def _refine_tempo_and_phase(
    envelope: NDArray[np.float64],
    *,
    frame_rate: float,
    bpm: float,
    phase_frames: float,
    settings: RhythmConfig,
) -> tuple[float, float]:
    """Fit one line through onset peaks near the coarse beat comb.

    Autocorrelation lags are quantized to onset frames.  Across a full song,
    fitting observed peak time against beat index gives a much more accurate
    fractional period and prevents that small error from accumulating near the
    end of the beat grid.
    """

    initial_period = frame_rate * 60.0 / bpm
    search_radius = max(
        2,
        int(
            math.ceil(
                max(settings.beat_tolerance_seconds * frame_rate, initial_period * 0.18)
            )
        ),
    )
    first_index = int(math.ceil(-phase_frames / initial_period))
    last_index = int(
        math.floor((envelope.size - 1 - phase_frames) / initial_period)
    )
    beat_indices: list[int] = []
    observed_frames: list[float] = []
    weights: list[float] = []
    for beat_index in range(first_index, last_index + 1):
        predicted = phase_frames + beat_index * initial_period
        center = int(round(predicted))
        start = max(0, center - search_radius)
        end = min(envelope.size, center + search_radius + 1)
        if start >= end:
            continue
        peak = start + int(np.argmax(envelope[start:end]))
        peak_strength = float(envelope[peak])
        if peak_strength < 0.10:
            continue
        # Spectral flux is deliberately smoothed and can have a two/three-frame
        # plateau.  A local weighted centroid is a more stable timestamp than
        # the first maximum returned by argmax.
        centroid_start = max(start, peak - 2)
        centroid_end = min(end, peak + 3)
        local = envelope[centroid_start:centroid_end]
        local_sum = float(np.sum(local))
        observed = (
            float(
                np.dot(
                    local,
                    np.arange(centroid_start, centroid_end, dtype=np.float64),
                )
                / local_sum
            )
            if local_sum > np.finfo(np.float64).eps
            else float(peak)
        )
        beat_indices.append(beat_index)
        observed_frames.append(observed)
        weights.append(peak_strength)

    if len(beat_indices) < 4 or max(beat_indices) - min(beat_indices) < 3:
        return bpm, phase_frames
    indices = np.asarray(beat_indices, dtype=np.float64)
    observations = np.asarray(observed_frames, dtype=np.float64)
    fit_weights = np.asarray(weights, dtype=np.float64)

    def weighted_fit(mask: NDArray[np.bool_]) -> tuple[float, float]:
        selected_indices = indices[mask]
        selected_observations = observations[mask]
        selected_weights = fit_weights[mask]
        index_mean = float(np.average(selected_indices, weights=selected_weights))
        observation_mean = float(
            np.average(selected_observations, weights=selected_weights)
        )
        centered_indices = selected_indices - index_mean
        denominator = float(
            np.sum(selected_weights * centered_indices * centered_indices)
        )
        if denominator <= np.finfo(np.float64).eps:
            return initial_period, phase_frames
        period = float(
            np.sum(
                selected_weights
                * centered_indices
                * (selected_observations - observation_mean)
            )
            / denominator
        )
        intercept = observation_mean - period * index_mean
        return period, intercept

    mask = np.ones(indices.size, dtype=bool)
    fitted_period, fitted_phase = weighted_fit(mask)
    residuals = observations - (fitted_phase + fitted_period * indices)
    median_residual = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median_residual)))
    residual_limit = max(2.0, initial_period * 0.045, mad * 3.5)
    robust_mask = np.abs(residuals - median_residual) <= residual_limit
    if np.count_nonzero(robust_mask) >= 4:
        fitted_period, fitted_phase = weighted_fit(robust_mask)

    refined_bpm = 60.0 * frame_rate / fitted_period if fitted_period > 0 else bpm
    if (
        not settings.min_bpm <= refined_bpm <= settings.max_bpm
        or abs(refined_bpm - bpm) / bpm > 0.025
    ):
        return bpm, phase_frames
    return refined_bpm, float(np.mod(fitted_phase, fitted_period))


def estimate_tempo(
    onset_envelope: OnsetEnvelope,
    *,
    config: RhythmConfig | None = None,
) -> TempoEstimate:
    """Estimate tempo and beat phase from an :class:`OnsetEnvelope`."""

    settings = config or RhythmConfig()
    envelope = np.asarray(onset_envelope.values, dtype=np.float64)
    duration = envelope.size / onset_envelope.frame_rate
    if duration < settings.minimum_duration_seconds:
        raise ValueError(
            f"pcm duration must be at least {settings.minimum_duration_seconds:g} seconds"
        )
    significant = int(np.count_nonzero(envelope >= 0.10))
    if significant < 3 or float(np.sum(envelope)) <= np.finfo(np.float64).eps:
        raise ValueError("pcm has insufficient rhythmic onset energy")

    frame_rate = onset_envelope.frame_rate
    largest_period = frame_rate * 60.0 / settings.min_bpm
    max_lag = min(envelope.size - 2, int(math.ceil(largest_period * 2.0)) + 2)
    if max_lag < 2:
        raise ValueError("onset envelope is too short for tempo estimation")
    autocorrelation = _normalized_autocorrelation(envelope, max_lag)
    tolerance_frames = max(1.0, settings.beat_tolerance_seconds * frame_rate)

    bpm_values = np.arange(
        settings.min_bpm,
        settings.max_bpm + settings.tempo_step_bpm * 0.5,
        settings.tempo_step_bpm,
        dtype=np.float64,
    )
    scores = np.empty(bpm_values.size, dtype=np.float64)
    phases = np.empty(bpm_values.size, dtype=np.float64)
    for index, bpm in enumerate(bpm_values):
        period = frame_rate * 60.0 / float(bpm)
        base_periodicity = max(0.0, _interpolate(autocorrelation, period))
        double_periodicity = max(0.0, _interpolate(autocorrelation, period * 2.0))
        periodicity = 0.82 * base_periodicity + 0.18 * double_periodicity
        phase, mean_strength, coverage = _phase_metrics(
            envelope, period, tolerance_frames
        )
        # Coverage prevents a strong every-other-beat accent from being
        # mistaken for half tempo near the top of the allowed BPM range.
        phase_quality = 0.65 * mean_strength + 0.35 * coverage
        prior = math.exp(
            -0.5 * (math.log2(float(bpm) / settings.preferred_bpm) / 0.55) ** 2
        )
        data_weight = 1.0 - settings.tempo_prior_weight
        scores[index] = data_weight * (
            0.54 * periodicity + 0.46 * phase_quality
        ) + settings.tempo_prior_weight * prior
        phases[index] = phase

    best_index = int(np.argmax(scores))
    best_bpm = float(bpm_values[best_index])
    # A parabolic interpolation reduces quantization without making tempo
    # search or output dependent on an FFT-bin boundary.
    if 0 < best_index < scores.size - 1:
        left, middle, right = scores[best_index - 1 : best_index + 2]
        denominator = left - 2.0 * middle + right
        if denominator < -1e-12:
            fractional_step = float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))
            best_bpm += fractional_step * settings.tempo_step_bpm

    best_period_frames = frame_rate * 60.0 / best_bpm
    best_phase_frames, _, _ = _phase_metrics(
        envelope, best_period_frames, tolerance_frames
    )
    best_bpm, best_phase_frames = _refine_tempo_and_phase(
        envelope,
        frame_rate=frame_rate,
        bpm=best_bpm,
        phase_frames=best_phase_frames,
        settings=settings,
    )
    separated = np.abs(bpm_values - best_bpm) >= max(2.0, best_bpm * 0.025)
    second_score = float(np.max(scores[separated])) if np.any(separated) else 0.0
    best_score = float(scores[best_index])
    margin = max(0.0, (best_score - second_score) / max(best_score, 1e-12))
    confidence = float(np.clip(0.72 * best_score + 0.28 * min(1.0, margin * 4.0), 0.0, 1.0))
    return TempoEstimate(
        bpm=best_bpm,
        period_seconds=60.0 / best_bpm,
        phase_seconds=best_phase_frames / frame_rate,
        confidence=confidence,
    )


def build_beat_grid(
    onset_envelope: OnsetEnvelope,
    tempo: TempoEstimate,
    *,
    duration_seconds: float,
    config: RhythmConfig | None = None,
) -> tuple[Beat, ...]:
    """Build a phase-aligned beat grid and attach local onset strength."""

    settings = config or RhythmConfig()
    if duration_seconds <= 0 or not math.isfinite(duration_seconds):
        raise ValueError("duration_seconds must be positive and finite")
    envelope = np.asarray(onset_envelope.values, dtype=np.float64)
    frame_rate = onset_envelope.frame_rate
    radius = max(1, int(math.ceil(settings.beat_tolerance_seconds * frame_rate)))
    period = tempo.period_seconds
    phase = tempo.phase_seconds % period
    count = max(0, int(math.floor((duration_seconds - phase) / period)) + 1)
    beats: list[Beat] = []
    for index in range(count):
        timestamp = phase + index * period
        center = int(round(timestamp * frame_rate))
        start = max(0, center - radius)
        end = min(envelope.size, center + radius + 1)
        strength = float(np.max(envelope[start:end])) if start < end else 0.0
        beats.append(Beat(index=index, time_seconds=timestamp, strength=strength))
    return tuple(beats)


def find_phrase_candidates(
    beats: Sequence[Beat],
    *,
    phrase_beats: Sequence[int] = (4, 8),
) -> tuple[PhraseCandidate, ...]:
    """Infer metrical anchors and return complete regular phrase spans.

    No downbeat classifier is assumed.  For each requested phrase length, the
    phase whose boundary beats carry the strongest average onsets is retained.
    These are candidates—not a claim that musical section boundaries are known.
    """

    lengths = tuple(dict.fromkeys(int(value) for value in phrase_beats))
    if not lengths or any(value < 2 for value in lengths):
        raise ValueError("phrase_beats must contain integers of at least 2")
    if any(beat.index != index for index, beat in enumerate(beats)):
        raise ValueError("beats must have contiguous zero-based indices")
    if any(
        beats[index].time_seconds <= beats[index - 1].time_seconds
        for index in range(1, len(beats))
    ):
        raise ValueError("beats must be in strictly increasing time order")

    candidates: list[PhraseCandidate] = []
    strengths = np.asarray([beat.strength for beat in beats], dtype=np.float64)
    for beat_count in lengths:
        if len(beats) <= beat_count:
            continue
        phase_scores = np.asarray(
            [
                float(np.mean(strengths[phase::beat_count]))
                if phase < strengths.size
                else 0.0
                for phase in range(beat_count)
            ],
            dtype=np.float64,
        )
        anchor = int(np.argmax(phase_scores))
        ordered = np.sort(phase_scores)
        runner_up = float(ordered[-2]) if ordered.size > 1 else 0.0
        metrical_margin = max(
            0.0,
            (float(phase_scores[anchor]) - runner_up)
            / max(float(phase_scores[anchor]), 1e-12),
        )
        for start in range(anchor, len(beats) - beat_count, beat_count):
            end = start + beat_count
            boundary_strength = min(1.0, float(beats[start].strength))
            confidence = float(
                np.clip(0.70 * boundary_strength + 0.30 * metrical_margin, 0.0, 1.0)
            )
            candidates.append(
                PhraseCandidate(
                    beat_count=beat_count,
                    start_beat_index=start,
                    end_beat_index=end,
                    start_seconds=beats[start].time_seconds,
                    end_seconds=beats[end].time_seconds,
                    confidence=confidence,
                )
            )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (item.start_seconds, item.beat_count, item.end_seconds),
        )
    )


def analyze_rhythm(
    pcm: ArrayLike,
    sample_rate: int,
    *,
    config: RhythmConfig | None = None,
) -> RhythmAnalysis:
    """Run onset, tempo, beat-grid, and phrase analysis on mono PCM."""

    settings = config or RhythmConfig()
    signal = _validated_pcm(pcm, sample_rate)
    onset_envelope = spectral_flux_onset_envelope(
        signal, sample_rate, config=settings
    )
    tempo = estimate_tempo(onset_envelope, config=settings)
    duration_seconds = signal.size / sample_rate
    beats = build_beat_grid(
        onset_envelope,
        tempo,
        duration_seconds=duration_seconds,
        config=settings,
    )
    phrases = find_phrase_candidates(beats, phrase_beats=settings.phrase_beats)
    return RhythmAnalysis(
        sample_rate=int(sample_rate),
        duration_seconds=duration_seconds,
        onset_envelope=onset_envelope,
        tempo=tempo,
        beats=beats,
        phrase_candidates=phrases,
    )


# Concise alias for callers that describe their input explicitly.
analyze_pcm_rhythm = analyze_rhythm
