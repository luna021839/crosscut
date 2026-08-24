"""Conservative local audio alignment for broadcaster clock drift.

The coarse synchronizer in :mod:`crosscut.audio` intentionally returns one
constant source offset.  That is a good failure-safe, but independent broadcast
encoders can accumulate a few frames of drift over a full song.  This module
measures that residual drift in overlapping spectral-flux windows and exposes a
monotonic piecewise-linear master-to-source clock map.

The result remains separate from decoding and rendering policy: callers can
consume a checked map while the explicit scalar fallback preserves the legacy
offset contract whenever local evidence is insufficient.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
import json
import math
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .rhythm import spectral_flux_onset_envelope


_EPSILON = np.finfo(np.float64).eps


@dataclass(frozen=True, slots=True)
class TimeWarpConfig:
    """Policy for local drift estimation.

    Twenty-four second windows overlap by half by default.  Long windows make
    repeated beats much less likely to win by accident, while a 12 second hop
    still observes gradual clock drift several times per song.
    """

    window_seconds: float = 24.0
    hop_seconds: float = 12.0
    search_radius_seconds: float = 1.5
    peak_exclusion_seconds: float = 0.18
    minimum_score: float = 0.32
    minimum_score_margin: float = 0.025
    minimum_anchor_count: int = 3
    minimum_anchor_fraction: float = 0.50
    maximum_drift_rate: float = 0.006
    maximum_trend_residual_seconds: float = 0.30

    def __post_init__(self) -> None:
        finite_values = (
            self.window_seconds,
            self.hop_seconds,
            self.search_radius_seconds,
            self.peak_exclusion_seconds,
            self.minimum_score,
            self.minimum_score_margin,
            self.minimum_anchor_fraction,
            self.maximum_drift_rate,
            self.maximum_trend_residual_seconds,
        )
        if any(not math.isfinite(value) for value in finite_values):
            raise ValueError("time-warp configuration values must be finite")
        if not 20.0 <= self.window_seconds <= 30.0:
            raise ValueError("window_seconds must be between 20 and 30 seconds")
        if not 0.0 < self.hop_seconds < self.window_seconds:
            raise ValueError("hop_seconds must be positive and overlap each window")
        if self.search_radius_seconds <= 0.0:
            raise ValueError("search_radius_seconds must be positive")
        if self.peak_exclusion_seconds < 0.0:
            raise ValueError("peak_exclusion_seconds cannot be negative")
        if not -1.0 <= self.minimum_score <= 1.0:
            raise ValueError("minimum_score must be between -1 and 1")
        if self.minimum_score_margin < 0.0:
            raise ValueError("minimum_score_margin cannot be negative")
        if (
            isinstance(self.minimum_anchor_count, bool)
            or not isinstance(self.minimum_anchor_count, (int, np.integer))
            or self.minimum_anchor_count < 2
        ):
            raise ValueError("minimum_anchor_count must be an integer of at least 2")
        if not 0.0 <= self.minimum_anchor_fraction <= 1.0:
            raise ValueError("minimum_anchor_fraction must be between 0 and 1")
        if not 0.0 < self.maximum_drift_rate < 1.0:
            raise ValueError("maximum_drift_rate must be between 0 and 1")
        if self.maximum_trend_residual_seconds <= 0.0:
            raise ValueError("maximum_trend_residual_seconds must be positive")

    @property
    def minimum_slope(self) -> float:
        return 1.0 - self.maximum_drift_rate

    @property
    def maximum_slope(self) -> float:
        return 1.0 + self.maximum_drift_rate

    def to_dict(self) -> dict[str, int | float]:
        return {
            "window_seconds": self.window_seconds,
            "hop_seconds": self.hop_seconds,
            "search_radius_seconds": self.search_radius_seconds,
            "peak_exclusion_seconds": self.peak_exclusion_seconds,
            "minimum_score": self.minimum_score,
            "minimum_score_margin": self.minimum_score_margin,
            "minimum_anchor_count": int(self.minimum_anchor_count),
            "minimum_anchor_fraction": self.minimum_anchor_fraction,
            "maximum_drift_rate": self.maximum_drift_rate,
            "minimum_slope": self.minimum_slope,
            "maximum_slope": self.maximum_slope,
            "maximum_trend_residual_seconds": self.maximum_trend_residual_seconds,
        }


@dataclass(frozen=True, slots=True)
class TimeWarpAnchor:
    """One local offset observation centered in an analysis window."""

    window_index: int
    window_start_seconds: float
    window_end_seconds: float
    master_seconds: float
    source_seconds: float
    offset_seconds: float
    score: float | None
    second_best_score: float | None
    score_margin: float | None
    accepted: bool
    rejection_reason: str | None = None
    trend_residual_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_index": self.window_index,
            "window_start_seconds": self.window_start_seconds,
            "window_end_seconds": self.window_end_seconds,
            "master_seconds": self.master_seconds,
            "source_seconds": self.source_seconds,
            "offset_seconds": self.offset_seconds,
            "score": self.score,
            "second_best_score": self.second_best_score,
            "score_margin": self.score_margin,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "trend_residual_seconds": self.trend_residual_seconds,
        }


@dataclass(frozen=True, slots=True)
class TimeWarpControlPoint:
    master_seconds: float
    source_seconds: float

    @property
    def offset_seconds(self) -> float:
        return self.source_seconds - self.master_seconds

    def to_dict(self) -> dict[str, float]:
        return {
            "master_seconds": self.master_seconds,
            "source_seconds": self.source_seconds,
            "offset_seconds": self.offset_seconds,
        }


@dataclass(frozen=True, slots=True)
class TimeWarpResidualDiagnostics:
    """Absolute residuals against the anchors' robust affine trend.

    The final map deliberately passes through every accepted point.  Reporting
    interpolation residuals would therefore always say zero.  Residuals here
    instead measure how coherent local observations are around their dominant
    clock trend, exposing jitter and local non-linearity honestly.
    """

    p50_seconds: float | None
    p95_seconds: float | None
    maximum_seconds: float | None
    sample_count: int
    definition: str = "absolute_anchor_residual_to_robust_affine_trend"

    def to_dict(self) -> dict[str, int | float | str | None]:
        return {
            "p50_seconds": self.p50_seconds,
            "p95_seconds": self.p95_seconds,
            "maximum_seconds": self.maximum_seconds,
            "sample_count": self.sample_count,
            "definition": self.definition,
        }


@dataclass(frozen=True, slots=True)
class TimeWarpMap:
    """A checked piecewise-linear source clock or an explicit scalar fallback."""

    global_offset_seconds: float
    master_duration_seconds: float
    source_duration_seconds: float
    control_points: tuple[TimeWarpControlPoint, ...]
    anchors: tuple[TimeWarpAnchor, ...]
    residuals: TimeWarpResidualDiagnostics
    config: TimeWarpConfig
    scalar_fallback: bool
    fallback_reason: str | None
    backend: str = "spectral_flux_local_ncc"

    @property
    def minimum_slope(self) -> float:
        return self.config.minimum_slope

    @property
    def maximum_slope(self) -> float:
        return self.config.maximum_slope

    def __post_init__(self) -> None:
        numeric_values = (
            self.global_offset_seconds,
            self.master_duration_seconds,
            self.source_duration_seconds,
            self.minimum_slope,
            self.maximum_slope,
        )
        if any(not math.isfinite(value) for value in numeric_values):
            raise ValueError("time-warp map values must be finite")
        if self.master_duration_seconds <= 0.0 or self.source_duration_seconds <= 0.0:
            raise ValueError("time-warp durations must be positive")
        if not 0.0 < self.minimum_slope <= self.maximum_slope:
            raise ValueError("time-warp slope bounds are invalid")
        if self.scalar_fallback:
            if self.fallback_reason is None:
                raise ValueError("scalar fallback must record a reason")
            return
        if self.fallback_reason is not None:
            raise ValueError("non-fallback time warp cannot record a fallback reason")
        if len(self.control_points) < 2:
            raise ValueError("piecewise time warp needs at least two control points")
        for left, right in zip(self.control_points, self.control_points[1:]):
            master_delta = right.master_seconds - left.master_seconds
            source_delta = right.source_seconds - left.source_seconds
            if master_delta <= 0.0 or source_delta <= 0.0:
                raise ValueError("time-warp control points must be strictly monotonic")
            slope = source_delta / master_delta
            if slope < self.minimum_slope - 1e-9 or slope > self.maximum_slope + 1e-9:
                raise ValueError("time-warp control-point slope violates hard bounds")

    @property
    def accepted_anchors(self) -> tuple[TimeWarpAnchor, ...]:
        return tuple(anchor for anchor in self.anchors if anchor.accepted)

    @property
    def rejected_anchors(self) -> tuple[TimeWarpAnchor, ...]:
        return tuple(anchor for anchor in self.anchors if not anchor.accepted)

    @property
    def segment_slopes(self) -> tuple[float, ...]:
        if self.scalar_fallback or len(self.control_points) < 2:
            return (1.0,)
        return tuple(
            (right.source_seconds - left.source_seconds)
            / (right.master_seconds - left.master_seconds)
            for left, right in zip(self.control_points, self.control_points[1:])
        )

    def master_to_source(self, master_seconds: float) -> float:
        """Map a finite master timestamp to the source timeline.

        Endpoint segments are linearly extrapolated.  Their already-validated
        slopes preserve monotonicity outside the measured anchor range.
        """

        master_time = float(master_seconds)
        if not math.isfinite(master_time):
            raise ValueError("master_seconds must be finite")
        if self.scalar_fallback or len(self.control_points) < 2:
            return master_time + self.global_offset_seconds

        masters = tuple(point.master_seconds for point in self.control_points)
        position = bisect_right(masters, master_time)
        if position == 0:
            left, right = self.control_points[0:2]
        elif position >= len(self.control_points):
            left, right = self.control_points[-2:]
        else:
            left, right = self.control_points[position - 1 : position + 1]
        fraction = (master_time - left.master_seconds) / (
            right.master_seconds - left.master_seconds
        )
        return left.source_seconds + fraction * (
            right.source_seconds - left.source_seconds
        )

    def source_to_master(self, source_seconds: float) -> float:
        """Invert the checked monotonic map for a finite source timestamp.

        Member tracking observations are recorded on the source clock.  Their
        canonical positions therefore need the inverse map, especially when a
        manual anchor or crop sample is compared with the master timeline.
        Endpoint segments are extrapolated symmetrically with
        :meth:`master_to_source`.
        """

        source_time = float(source_seconds)
        if not math.isfinite(source_time):
            raise ValueError("source_seconds must be finite")
        if self.scalar_fallback or len(self.control_points) < 2:
            return source_time - self.global_offset_seconds

        sources = tuple(point.source_seconds for point in self.control_points)
        position = bisect_right(sources, source_time)
        if position == 0:
            left, right = self.control_points[0:2]
        elif position >= len(self.control_points):
            left, right = self.control_points[-2:]
        else:
            left, right = self.control_points[position - 1 : position + 1]
        fraction = (source_time - left.source_seconds) / (
            right.source_seconds - left.source_seconds
        )
        return left.master_seconds + fraction * (
            right.master_seconds - left.master_seconds
        )

    def master_span_to_source(
        self, master_start_seconds: float, master_end_seconds: float
    ) -> tuple[float, float]:
        """Map both endpoints of one positive canonical span.

        The checked map is strictly monotonic, so the returned source span is
        positive even when local clock rate differs slightly from one.
        """

        master_start = float(master_start_seconds)
        master_end = float(master_end_seconds)
        if not math.isfinite(master_start) or not math.isfinite(master_end):
            raise ValueError("master span endpoints must be finite")
        if master_end <= master_start:
            raise ValueError("master span must have positive duration")
        source_start = self.master_to_source(master_start)
        source_end = self.master_to_source(master_end)
        if source_end <= source_start:
            raise ValueError("mapped source span must have positive duration")
        return source_start, source_end

    def to_dict(self) -> dict[str, Any]:
        slopes = self.segment_slopes
        accepted = self.accepted_anchors
        rejected = self.rejected_anchors
        return {
            "backend": self.backend,
            "global_offset_seconds": self.global_offset_seconds,
            "master_duration_seconds": self.master_duration_seconds,
            "source_duration_seconds": self.source_duration_seconds,
            "scalar_fallback": self.scalar_fallback,
            "fallback_reason": self.fallback_reason,
            "minimum_slope": self.minimum_slope,
            "maximum_slope": self.maximum_slope,
            "observed_minimum_slope": min(slopes),
            "observed_maximum_slope": max(slopes),
            "mean_slope": float(np.mean(slopes)),
            "estimated_drift_parts_per_million": float((np.mean(slopes) - 1.0) * 1e6),
            "configuration": self.config.to_dict(),
            "control_points": [point.to_dict() for point in self.control_points],
            "accepted_anchor_count": len(accepted),
            "rejected_anchor_count": len(rejected),
            "accepted_anchors": [anchor.to_dict() for anchor in accepted],
            "rejected_anchors": [anchor.to_dict() for anchor in rejected],
            "residuals": self.residuals.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def _validated_pcm(
    pcm: ArrayLike, *, name: str, sample_rate: int
) -> NDArray[np.float64]:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise TypeError("sample_rate must be an integer")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    values = np.asarray(pcm, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional mono signal")
    if values.size < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or infinite samples")
    return np.ascontiguousarray(values)


def _window_starts(
    total_frames: int, window_frames: int, hop_frames: int
) -> tuple[int, ...]:
    if total_frames < window_frames:
        return ()
    starts = list(range(0, total_frames - window_frames + 1, hop_frames))
    final_start = total_frames - window_frames
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def _parabolic_peak(scores: NDArray[np.float64], index: int) -> float:
    if index <= 0 or index >= scores.size - 1:
        return 0.0
    left = float(scores[index - 1])
    center = float(scores[index])
    right = float(scores[index + 1])
    if not all(math.isfinite(value) for value in (left, center, right)):
        return 0.0
    denominator = left - 2.0 * center + right
    if abs(denominator) <= _EPSILON or denominator >= 0.0:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))


def _measure_window(
    reference: NDArray[np.float64],
    source: NDArray[np.float64],
    *,
    window_index: int,
    reference_start: int,
    window_frames: int,
    envelope_rate: float,
    global_offset_seconds: float,
    config: TimeWarpConfig,
) -> TimeWarpAnchor:
    master_center_frame = reference_start + 0.5 * (window_frames - 1)
    master_seconds = master_center_frame / envelope_rate
    window_start_seconds = reference_start / envelope_rate
    window_end_seconds = (reference_start + window_frames - 1) / envelope_rate
    fallback_source_seconds = master_seconds + global_offset_seconds

    minimum_offset = int(
        math.ceil((global_offset_seconds - config.search_radius_seconds) * envelope_rate)
    )
    maximum_offset = int(
        math.floor((global_offset_seconds + config.search_radius_seconds) * envelope_rate)
    )
    offsets = np.arange(minimum_offset, maximum_offset + 1, dtype=np.int64)
    source_starts = reference_start + offsets
    valid = (source_starts >= 0) & (source_starts + window_frames <= source.size)
    offsets = offsets[valid]
    source_starts = source_starts[valid]
    if source_starts.size == 0:
        return TimeWarpAnchor(
            window_index=window_index,
            window_start_seconds=window_start_seconds,
            window_end_seconds=window_end_seconds,
            master_seconds=master_seconds,
            source_seconds=fallback_source_seconds,
            offset_seconds=global_offset_seconds,
            score=None,
            second_best_score=None,
            score_margin=None,
            accepted=False,
            rejection_reason="outside_source_overlap",
        )

    reference_window = reference[reference_start : reference_start + window_frames]
    reference_centered = reference_window - float(np.mean(reference_window))
    reference_energy = float(np.linalg.norm(reference_centered))
    if reference_energy <= _EPSILON:
        return TimeWarpAnchor(
            window_index=window_index,
            window_start_seconds=window_start_seconds,
            window_end_seconds=window_end_seconds,
            master_seconds=master_seconds,
            source_seconds=fallback_source_seconds,
            offset_seconds=global_offset_seconds,
            score=None,
            second_best_score=None,
            score_margin=None,
            accepted=False,
            rejection_reason="insufficient_reference_flux_activity",
        )

    source_windows = np.lib.stride_tricks.sliding_window_view(source, window_frames)[
        source_starts
    ]
    source_centered = source_windows - np.mean(source_windows, axis=1, keepdims=True)
    source_energies = np.linalg.norm(source_centered, axis=1)
    usable = source_energies > _EPSILON
    scores = np.full(source_starts.size, -np.inf, dtype=np.float64)
    if np.any(usable):
        # ``sum`` is deliberately used instead of BLAS ``matmul`` here.  The
        # candidate matrix is a fancy-indexed view of sliding windows; some
        # Accelerate builds emit spurious overflow warnings for that layout.
        dots = np.sum(
            source_centered[usable] * reference_centered[None, :], axis=1
        )
        scores[usable] = dots / (source_energies[usable] * reference_energy)
        scores[usable] = np.clip(scores[usable], -1.0, 1.0)
    if not np.any(np.isfinite(scores)):
        return TimeWarpAnchor(
            window_index=window_index,
            window_start_seconds=window_start_seconds,
            window_end_seconds=window_end_seconds,
            master_seconds=master_seconds,
            source_seconds=fallback_source_seconds,
            offset_seconds=global_offset_seconds,
            score=None,
            second_best_score=None,
            score_margin=None,
            accepted=False,
            rejection_reason="insufficient_source_flux_activity",
        )

    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])
    peak_exclusion_frames = max(
        1, int(round(config.peak_exclusion_seconds * envelope_rate))
    )
    competing = usable & (
        np.abs(source_starts - source_starts[best_index]) > peak_exclusion_frames
    )
    if np.any(competing):
        second_best_score: float | None = float(np.max(scores[competing]))
        score_margin = best_score - second_best_score
    else:
        second_best_score = None
        score_margin = best_score + 1.0

    fractional_peak = _parabolic_peak(scores, best_index)
    measured_offset_seconds = (
        float(offsets[best_index]) + fractional_peak
    ) / envelope_rate
    source_seconds = master_seconds + measured_offset_seconds
    if best_score < config.minimum_score:
        accepted = False
        reason = "low_correlation_score"
    elif score_margin < config.minimum_score_margin:
        accepted = False
        reason = "ambiguous_correlation_peak"
    else:
        accepted = True
        reason = None
    return TimeWarpAnchor(
        window_index=window_index,
        window_start_seconds=window_start_seconds,
        window_end_seconds=window_end_seconds,
        master_seconds=master_seconds,
        source_seconds=source_seconds,
        offset_seconds=measured_offset_seconds,
        score=best_score,
        second_best_score=second_best_score,
        score_margin=score_margin,
        accepted=accepted,
        rejection_reason=reason,
    )


def _robust_affine_trend(
    anchors: Sequence[TimeWarpAnchor],
    *,
    minimum_slope: float,
    maximum_slope: float,
) -> tuple[float, float] | None:
    if len(anchors) < 2:
        return None
    slopes: list[float] = []
    for index, left in enumerate(anchors[:-1]):
        for right in anchors[index + 1 :]:
            delta_master = right.master_seconds - left.master_seconds
            if delta_master <= 0.0:
                continue
            slope = (right.source_seconds - left.source_seconds) / delta_master
            if minimum_slope <= slope <= maximum_slope:
                slopes.append(slope)
    if not slopes:
        return None
    slope = float(np.median(np.asarray(slopes, dtype=np.float64)))
    intercepts = np.asarray(
        [anchor.source_seconds - slope * anchor.master_seconds for anchor in anchors],
        dtype=np.float64,
    )
    return slope, float(np.median(intercepts))


def _compatible_path(
    anchors: Sequence[TimeWarpAnchor],
    *,
    minimum_slope: float,
    maximum_slope: float,
) -> tuple[int, ...]:
    """Return the highest-confidence longest slope-bounded anchor path."""

    if not anchors:
        return ()
    count = [1] * len(anchors)
    confidence = [
        float(anchor.score or 0.0) + min(float(anchor.score_margin or 0.0), 0.25)
        for anchor in anchors
    ]
    previous = [-1] * len(anchors)
    for right_index, right in enumerate(anchors):
        for left_index in range(right_index):
            left = anchors[left_index]
            master_delta = right.master_seconds - left.master_seconds
            if master_delta <= 0.0:
                continue
            slope = (right.source_seconds - left.source_seconds) / master_delta
            if not minimum_slope <= slope <= maximum_slope:
                continue
            candidate_count = count[left_index] + 1
            candidate_confidence = confidence[left_index] + float(right.score or 0.0) + min(
                float(right.score_margin or 0.0), 0.25
            )
            if (candidate_count, candidate_confidence) > (
                count[right_index],
                confidence[right_index],
            ):
                count[right_index] = candidate_count
                confidence[right_index] = candidate_confidence
                previous[right_index] = left_index
    end = max(range(len(anchors)), key=lambda index: (count[index], confidence[index]))
    path: list[int] = []
    while end >= 0:
        path.append(end)
        end = previous[end]
    return tuple(reversed(path))


def _residual_diagnostics(
    anchors: Sequence[TimeWarpAnchor],
    trend: tuple[float, float] | None,
) -> TimeWarpResidualDiagnostics:
    if not anchors or trend is None:
        return TimeWarpResidualDiagnostics(None, None, None, 0)
    slope, intercept = trend
    residuals = np.asarray(
        [
            abs(anchor.source_seconds - (slope * anchor.master_seconds + intercept))
            for anchor in anchors
        ],
        dtype=np.float64,
    )
    return TimeWarpResidualDiagnostics(
        p50_seconds=float(np.percentile(residuals, 50.0)),
        p95_seconds=float(np.percentile(residuals, 95.0)),
        maximum_seconds=float(np.max(residuals)),
        sample_count=int(residuals.size),
    )


def _scalar_fallback(
    *,
    global_offset_seconds: float,
    master_duration_seconds: float,
    source_duration_seconds: float,
    anchors: Sequence[TimeWarpAnchor],
    config: TimeWarpConfig,
    reason: str,
) -> TimeWarpMap:
    rejected = tuple(
        anchor
        if not anchor.accepted
        else replace(
            anchor,
            accepted=False,
            rejection_reason="insufficient_anchor_consensus",
            trend_residual_seconds=None,
        )
        for anchor in anchors
    )
    return TimeWarpMap(
        global_offset_seconds=global_offset_seconds,
        master_duration_seconds=master_duration_seconds,
        source_duration_seconds=source_duration_seconds,
        control_points=(),
        anchors=rejected,
        residuals=TimeWarpResidualDiagnostics(None, None, None, 0),
        config=config,
        scalar_fallback=True,
        fallback_reason=reason,
    )


def estimate_local_timewarp(
    reference_pcm: ArrayLike,
    source_pcm: ArrayLike,
    *,
    sample_rate: int,
    global_offset_seconds: float,
    config: TimeWarpConfig | None = None,
) -> TimeWarpMap:
    """Estimate a conservative local master-to-source clock mapping.

    The supplied global offset follows the engine contract: master time ``t``
    corresponds approximately to source time ``t + global_offset_seconds``.
    Local searches never leave ``search_radius_seconds`` around that estimate.
    If confidence, monotonicity, or drift bounds fail, ``master_to_source`` uses
    that exact scalar contract instead.
    """

    if not math.isfinite(global_offset_seconds):
        raise ValueError("global_offset_seconds must be finite")
    settings = config or TimeWarpConfig()
    reference_signal = _validated_pcm(
        reference_pcm, name="reference_pcm", sample_rate=sample_rate
    )
    source_signal = _validated_pcm(
        source_pcm, name="source_pcm", sample_rate=sample_rate
    )
    master_duration_seconds = reference_signal.size / sample_rate
    source_duration_seconds = source_signal.size / sample_rate

    reference_envelope = spectral_flux_onset_envelope(reference_signal, sample_rate)
    source_envelope = spectral_flux_onset_envelope(source_signal, sample_rate)
    envelope_rate = reference_envelope.frame_rate
    if not math.isclose(
        envelope_rate,
        source_envelope.frame_rate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("reference and source onset-envelope rates differ")
    reference_values = np.asarray(reference_envelope.values, dtype=np.float64)
    source_values = np.asarray(source_envelope.values, dtype=np.float64)
    window_frames = max(2, int(round(settings.window_seconds * envelope_rate)))
    hop_frames = max(1, int(round(settings.hop_seconds * envelope_rate)))
    starts = _window_starts(reference_values.size, window_frames, hop_frames)
    if not starts:
        return _scalar_fallback(
            global_offset_seconds=global_offset_seconds,
            master_duration_seconds=master_duration_seconds,
            source_duration_seconds=source_duration_seconds,
            anchors=(),
            config=settings,
            reason="insufficient_duration_for_local_windows",
        )

    anchors = tuple(
        _measure_window(
            reference_values,
            source_values,
            window_index=index,
            reference_start=start,
            window_frames=window_frames,
            envelope_rate=envelope_rate,
            global_offset_seconds=global_offset_seconds,
            config=settings,
        )
        for index, start in enumerate(starts)
    )
    confident = tuple(anchor for anchor in anchors if anchor.accepted)
    required_count = max(
        int(settings.minimum_anchor_count),
        int(math.ceil(settings.minimum_anchor_fraction * len(anchors))),
    )
    if len(confident) < required_count:
        return _scalar_fallback(
            global_offset_seconds=global_offset_seconds,
            master_duration_seconds=master_duration_seconds,
            source_duration_seconds=source_duration_seconds,
            anchors=anchors,
            config=settings,
            reason="insufficient_confident_anchors",
        )

    trend = _robust_affine_trend(
        confident,
        minimum_slope=settings.minimum_slope,
        maximum_slope=settings.maximum_slope,
    )
    if trend is not None and len(confident) >= 3:
        trend_slope, trend_intercept = trend
        filtered: list[TimeWarpAnchor] = []
        for anchor in anchors:
            if not anchor.accepted:
                filtered.append(anchor)
                continue
            residual = abs(
                anchor.source_seconds
                - (trend_slope * anchor.master_seconds + trend_intercept)
            )
            if residual > settings.maximum_trend_residual_seconds:
                filtered.append(
                    replace(
                        anchor,
                        accepted=False,
                        rejection_reason="trend_residual_outlier",
                        trend_residual_seconds=residual,
                    )
                )
            else:
                filtered.append(replace(anchor, trend_residual_seconds=residual))
        anchors = tuple(filtered)

    eligible = tuple(anchor for anchor in anchors if anchor.accepted)
    path_indices = _compatible_path(
        eligible,
        minimum_slope=settings.minimum_slope,
        maximum_slope=settings.maximum_slope,
    )
    selected_eligible = {index for index in path_indices}
    selected_windows = {eligible[index].window_index for index in selected_eligible}
    anchors = tuple(
        anchor
        if not anchor.accepted or anchor.window_index in selected_windows
        else replace(
            anchor,
            accepted=False,
            rejection_reason="slope_or_monotonicity_constraint",
            trend_residual_seconds=None,
        )
        for anchor in anchors
    )
    accepted = tuple(anchor for anchor in anchors if anchor.accepted)
    if len(accepted) < required_count:
        return _scalar_fallback(
            global_offset_seconds=global_offset_seconds,
            master_duration_seconds=master_duration_seconds,
            source_duration_seconds=source_duration_seconds,
            anchors=anchors,
            config=settings,
            reason="insufficient_slope_bounded_anchors",
        )

    final_trend = _robust_affine_trend(
        accepted,
        minimum_slope=settings.minimum_slope,
        maximum_slope=settings.maximum_slope,
    )
    if final_trend is not None:
        slope, intercept = final_trend
        anchors = tuple(
            replace(
                anchor,
                trend_residual_seconds=abs(
                    anchor.source_seconds
                    - (slope * anchor.master_seconds + intercept)
                ),
            )
            if anchor.accepted
            else anchor
            for anchor in anchors
        )
        accepted = tuple(anchor for anchor in anchors if anchor.accepted)

    control_points = tuple(
        TimeWarpControlPoint(anchor.master_seconds, anchor.source_seconds)
        for anchor in accepted
    )
    return TimeWarpMap(
        global_offset_seconds=global_offset_seconds,
        master_duration_seconds=master_duration_seconds,
        source_duration_seconds=source_duration_seconds,
        control_points=control_points,
        anchors=anchors,
        residuals=_residual_diagnostics(accepted, final_trend),
        config=settings,
        scalar_fallback=False,
        fallback_reason=None,
    )


# Concise alias for future orchestration code.
estimate_timewarp = estimate_local_timewarp
