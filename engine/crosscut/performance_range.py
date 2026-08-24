"""Conservative local shared-performance range estimation.

Music-show uploads often contain unrelated presenter intros, applause, ending
poses, and post-performance close-ups around the same song.  Coarse alignment
correctly synchronizes the song but its geometric media intersection cannot
distinguish those unmatched tails.  This module compares already-decoded local
spectral-flux envelopes on the accepted master clock, then uses reference-video
hard cuts only to retain a bounded ending/start pose around the audio consensus.

No official duration, network metadata, identity inference, or raw PCM is
persisted.  Weak evidence always keeps the caller's original common range.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .rhythm import spectral_flux_onset_envelope
from .timewarp import TimeWarpMap


@dataclass(frozen=True, slots=True)
class SharedPerformanceRangeConfig:
    """Deterministic evidence and boundary-refinement policy."""

    window_seconds: float = 2.0
    hop_seconds: float = 0.5
    minimum_correlation: float = 0.30
    minimum_passing_source_fraction: float = 0.60
    minimum_comparable_source_fraction: float = 0.60
    minimum_comparable_source_count: int = 4
    maximum_internal_gap_seconds: float = 2.0
    minimum_cluster_density: float = 0.65
    minimum_shared_duration_seconds: float = 45.0
    minimum_shared_range_fraction: float = 0.55
    minimum_trim_seconds: float = 2.0
    start_edge_retention_seconds: float = 2.0
    start_pose_guard_seconds: float = 1.5
    scene_cut_minimum_tail_seconds: float = 1.0
    scene_cut_maximum_tail_seconds: float = 6.0

    def __post_init__(self) -> None:
        finite_positive = (
            self.window_seconds,
            self.hop_seconds,
            self.maximum_internal_gap_seconds,
            self.minimum_shared_duration_seconds,
            self.minimum_trim_seconds,
            self.start_edge_retention_seconds,
            self.start_pose_guard_seconds,
            self.scene_cut_minimum_tail_seconds,
            self.scene_cut_maximum_tail_seconds,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in finite_positive):
            raise ValueError("shared performance range durations must be finite and positive")
        if self.hop_seconds > self.window_seconds:
            raise ValueError("shared performance hop cannot exceed its window")
        if self.maximum_internal_gap_seconds < self.hop_seconds:
            raise ValueError("shared performance internal gap must cover at least one hop")
        for name, value in (
            ("minimum_passing_source_fraction", self.minimum_passing_source_fraction),
            ("minimum_comparable_source_fraction", self.minimum_comparable_source_fraction),
            ("minimum_cluster_density", self.minimum_cluster_density),
            ("minimum_shared_range_fraction", self.minimum_shared_range_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if not -1.0 <= self.minimum_correlation <= 1.0:
            raise ValueError("minimum_correlation must be between -1 and 1")
        if (
            isinstance(self.minimum_comparable_source_count, bool)
            or not isinstance(self.minimum_comparable_source_count, int)
            or self.minimum_comparable_source_count < 1
        ):
            raise ValueError("minimum comparable source count must be a positive integer")
        if self.scene_cut_minimum_tail_seconds >= self.scene_cut_maximum_tail_seconds:
            raise ValueError("scene-cut tail bounds must be strictly increasing")

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_seconds": self.window_seconds,
            "hop_seconds": self.hop_seconds,
            "minimum_correlation": self.minimum_correlation,
            "minimum_passing_source_fraction": self.minimum_passing_source_fraction,
            "minimum_comparable_source_fraction": self.minimum_comparable_source_fraction,
            "minimum_comparable_source_count": self.minimum_comparable_source_count,
            "maximum_internal_gap_seconds": self.maximum_internal_gap_seconds,
            "minimum_cluster_density": self.minimum_cluster_density,
            "minimum_shared_duration_seconds": self.minimum_shared_duration_seconds,
            "minimum_shared_range_fraction": self.minimum_shared_range_fraction,
            "minimum_trim_seconds": self.minimum_trim_seconds,
            "start_edge_retention_seconds": self.start_edge_retention_seconds,
            "start_pose_guard_seconds": self.start_pose_guard_seconds,
            "scene_cut_minimum_tail_seconds": self.scene_cut_minimum_tail_seconds,
            "scene_cut_maximum_tail_seconds": self.scene_cut_maximum_tail_seconds,
        }


@dataclass(frozen=True, slots=True)
class SharedPerformanceWindow:
    start_seconds: float
    end_seconds: float
    median_correlation: float | None
    passing_source_fraction: float
    comparable_source_count: int
    required_comparable_source_count: int
    shared: bool
    source_correlations: tuple[tuple[int, float | None], ...]

    @property
    def center_seconds(self) -> float:
        return (self.start_seconds + self.end_seconds) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "center_seconds": self.center_seconds,
            "median_correlation": self.median_correlation,
            "passing_source_fraction": self.passing_source_fraction,
            "comparable_source_count": self.comparable_source_count,
            "required_comparable_source_count": self.required_comparable_source_count,
            "shared": self.shared,
            "source_correlations": [
                {"source_index": source_index, "correlation": correlation}
                for source_index, correlation in self.source_correlations
            ],
        }


@dataclass(frozen=True, slots=True)
class SharedPerformanceRangeEstimate:
    original_start_seconds: float
    original_end_seconds: float
    audio_consensus_start_seconds: float | None
    audio_consensus_end_seconds: float | None
    final_start_seconds: float
    final_end_seconds: float
    applied: bool
    evidence_strong: bool
    reason: str
    start_boundary_basis: str
    end_boundary_basis: str
    reference_scene_cut_master_seconds: tuple[float, ...]
    selected_start_scene_cut_seconds: float | None
    selected_end_scene_cut_seconds: float | None
    expected_peer_source_count: int
    available_peer_source_count: int
    selected_cluster_window_count: int
    selected_cluster_density: float | None
    selected_cluster_duration_seconds: float | None
    windows: tuple[SharedPerformanceWindow, ...]
    config: SharedPerformanceRangeConfig

    @property
    def original_duration_seconds(self) -> float:
        return self.original_end_seconds - self.original_start_seconds

    @property
    def final_duration_seconds(self) -> float:
        return self.final_end_seconds - self.final_start_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_basis": (
                "local_pcm_spectral_flux_reference_peer_consensus_"
                "with_reference_hard_cut_pose_tail_v1"
            ),
            "scope": "automatic_group_only",
            "applied": self.applied,
            "evidence_strong": self.evidence_strong,
            "reason": self.reason,
            "original_range": {
                "start_seconds": self.original_start_seconds,
                "end_seconds": self.original_end_seconds,
                "duration_seconds": self.original_duration_seconds,
            },
            "audio_consensus_range": (
                {
                    "start_seconds": self.audio_consensus_start_seconds,
                    "end_seconds": self.audio_consensus_end_seconds,
                    "duration_seconds": (
                        self.audio_consensus_end_seconds
                        - self.audio_consensus_start_seconds
                    ),
                }
                if self.audio_consensus_start_seconds is not None
                and self.audio_consensus_end_seconds is not None
                else None
            ),
            "final_range": {
                "start_seconds": self.final_start_seconds,
                "end_seconds": self.final_end_seconds,
                "duration_seconds": self.final_duration_seconds,
                "trimmed_start_seconds": (
                    self.final_start_seconds - self.original_start_seconds
                ),
                "trimmed_end_seconds": (
                    self.original_end_seconds - self.final_end_seconds
                ),
            },
            "start_boundary_basis": self.start_boundary_basis,
            "end_boundary_basis": self.end_boundary_basis,
            "reference_scene_cut_master_seconds": list(
                self.reference_scene_cut_master_seconds
            ),
            "selected_start_scene_cut_seconds": self.selected_start_scene_cut_seconds,
            "selected_end_scene_cut_seconds": self.selected_end_scene_cut_seconds,
            "expected_peer_source_count": self.expected_peer_source_count,
            "available_peer_source_count": self.available_peer_source_count,
            "selected_cluster_window_count": self.selected_cluster_window_count,
            "selected_cluster_density": self.selected_cluster_density,
            "selected_cluster_duration_seconds": (
                self.selected_cluster_duration_seconds
            ),
            "window_count": len(self.windows),
            "shared_window_count": sum(window.shared for window in self.windows),
            "windows": [window.to_dict() for window in self.windows],
            "config": self.config.to_dict(),
            "privacy": {
                "raw_pcm_persisted": False,
                "spectral_flux_vectors_persisted": False,
                "persisted_evidence": "window_level_correlation_scalars_only",
            },
        }


def _unchanged_estimate(
    *,
    common_start: float,
    common_end: float,
    reason: str,
    expected_peer_count: int,
    available_peer_count: int,
    config: SharedPerformanceRangeConfig,
    windows: Sequence[SharedPerformanceWindow] = (),
) -> SharedPerformanceRangeEstimate:
    return SharedPerformanceRangeEstimate(
        original_start_seconds=common_start,
        original_end_seconds=common_end,
        audio_consensus_start_seconds=None,
        audio_consensus_end_seconds=None,
        final_start_seconds=common_start,
        final_end_seconds=common_end,
        applied=False,
        evidence_strong=False,
        reason=reason,
        start_boundary_basis="original_common_range_weak_evidence",
        end_boundary_basis="original_common_range_weak_evidence",
        reference_scene_cut_master_seconds=(),
        selected_start_scene_cut_seconds=None,
        selected_end_scene_cut_seconds=None,
        expected_peer_source_count=expected_peer_count,
        available_peer_source_count=available_peer_count,
        selected_cluster_window_count=0,
        selected_cluster_density=None,
        selected_cluster_duration_seconds=None,
        windows=tuple(windows),
        config=config,
    )


def _master_to_source_seconds(
    master_seconds: NDArray[np.float64],
    *,
    source_index: int,
    audio_offsets: Mapping[int, float],
    source_timewarps: Mapping[int, TimeWarpMap | None],
) -> NDArray[np.float64]:
    timewarp = source_timewarps.get(source_index)
    if timewarp is None or timewarp.scalar_fallback:
        return master_seconds + float(audio_offsets[source_index])
    return np.asarray(
        [timewarp.master_to_source(float(value)) for value in master_seconds],
        dtype=np.float64,
    )


def _window_correlation(
    reference: NDArray[np.float64],
    source: NDArray[np.float64],
) -> float | None:
    usable = np.isfinite(reference) & np.isfinite(source)
    if int(np.count_nonzero(usable)) < max(12, reference.size // 2):
        return None
    left = reference[usable]
    right = source[usable]
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _cluster_supported_windows(
    windows: Sequence[SharedPerformanceWindow],
    *,
    maximum_gap_seconds: float,
) -> tuple[tuple[int, ...], ...]:
    supported = [index for index, window in enumerate(windows) if window.shared]
    if not supported:
        return ()
    clusters: list[list[int]] = [[supported[0]]]
    for index in supported[1:]:
        previous = clusters[-1][-1]
        if (
            windows[index].start_seconds - windows[previous].start_seconds
            <= maximum_gap_seconds + 1e-9
        ):
            clusters[-1].append(index)
        else:
            clusters.append([index])
    return tuple(tuple(cluster) for cluster in clusters)


def estimate_shared_performance_range(
    pcm_by_source: Mapping[int, ArrayLike],
    *,
    sample_rate: int,
    common_start_seconds: float,
    common_end_seconds: float,
    source_indices: Sequence[int],
    reference_source_index: int,
    audio_offsets: Mapping[int, float] | Sequence[float],
    source_timewarps: Mapping[int, TimeWarpMap | None]
    | Sequence[TimeWarpMap | None]
    | None = None,
    reference_scene_cut_master_seconds: Sequence[float] = (),
    config: SharedPerformanceRangeConfig | None = None,
) -> SharedPerformanceRangeEstimate:
    """Estimate and refine the shared performance inside a common media range.

    Every peer is compared against the reference on a fixed master-time grid.
    The longest dense consensus cluster supplies an audio boundary.  Reference
    hard cuts then retain a bounded initial/final pose without allowing a long
    post-performance close-up tail back into the result.
    """

    settings = config or SharedPerformanceRangeConfig()
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("shared performance sample_rate must be a positive integer")
    common_start = float(common_start_seconds)
    common_end = float(common_end_seconds)
    if (
        not math.isfinite(common_start)
        or not math.isfinite(common_end)
        or common_end <= common_start
    ):
        raise ValueError("shared performance common range must be finite and positive")
    indices = tuple(dict.fromkeys(int(index) for index in source_indices))
    if len(indices) < 2 or reference_source_index not in indices:
        raise ValueError("shared performance needs a reference and at least one peer")
    offsets = (
        {index: float(audio_offsets[index]) for index in indices}
        if not isinstance(audio_offsets, Mapping)
        else {index: float(audio_offsets[index]) for index in indices}
    )
    if any(not math.isfinite(value) for value in offsets.values()):
        raise ValueError("shared performance audio offsets must be finite")
    clocks = (
        {index: None for index in indices}
        if source_timewarps is None
        else {
            index: source_timewarps.get(index)
            if isinstance(source_timewarps, Mapping)
            else source_timewarps[index]
            for index in indices
        }
    )

    expected_peer_count = len(indices) - 1
    available_indices = tuple(index for index in indices if index in pcm_by_source)
    available_peer_count = sum(index != reference_source_index for index in available_indices)
    required_available_peer_count = max(
        settings.minimum_comparable_source_count,
        int(math.ceil(expected_peer_count * settings.minimum_comparable_source_fraction)),
    )
    if (
        reference_source_index not in pcm_by_source
        or available_peer_count < required_available_peer_count
    ):
        return _unchanged_estimate(
            common_start=common_start,
            common_end=common_end,
            reason="insufficient_decoded_peer_sources",
            expected_peer_count=expected_peer_count,
            available_peer_count=available_peer_count,
            config=settings,
        )

    envelopes: dict[int, tuple[NDArray[np.float64], float]] = {}
    for index in available_indices:
        try:
            envelope = spectral_flux_onset_envelope(pcm_by_source[index], sample_rate)
        except (TypeError, ValueError, FloatingPointError):
            continue
        envelopes[index] = (
            np.asarray(envelope.values, dtype=np.float64),
            float(envelope.frame_rate),
        )
    available_peer_count = sum(
        index != reference_source_index for index in envelopes
    )
    if (
        reference_source_index not in envelopes
        or available_peer_count < required_available_peer_count
    ):
        return _unchanged_estimate(
            common_start=common_start,
            common_end=common_end,
            reason="insufficient_spectral_flux_peer_sources",
            expected_peer_count=expected_peer_count,
            available_peer_count=available_peer_count,
            config=settings,
        )

    reference_values, reference_rate = envelopes[reference_source_index]
    first_window_start = (
        math.ceil((common_start - 1e-9) / settings.hop_seconds)
        * settings.hop_seconds
    )
    last_window_start = common_end - settings.window_seconds
    if last_window_start < first_window_start:
        return _unchanged_estimate(
            common_start=common_start,
            common_end=common_end,
            reason="common_range_shorter_than_consensus_window",
            expected_peer_count=expected_peer_count,
            available_peer_count=available_peer_count,
            config=settings,
        )
    window_count = int(
        math.floor((last_window_start - first_window_start) / settings.hop_seconds)
    ) + 1
    sample_count = max(24, int(round(settings.window_seconds * reference_rate)))
    relative_master_times = np.arange(sample_count, dtype=np.float64) / reference_rate
    windows: list[SharedPerformanceWindow] = []
    required_comparable = max(
        settings.minimum_comparable_source_count,
        int(math.ceil(expected_peer_count * settings.minimum_comparable_source_fraction)),
    )

    for window_index in range(window_count):
        start = first_window_start + window_index * settings.hop_seconds
        master_times = start + relative_master_times
        reference_local = _master_to_source_seconds(
            master_times,
            source_index=reference_source_index,
            audio_offsets=offsets,
            source_timewarps=clocks,
        )
        reference_window = np.interp(
            reference_local * reference_rate,
            np.arange(reference_values.size, dtype=np.float64),
            reference_values,
            left=np.nan,
            right=np.nan,
        )
        correlations: list[tuple[int, float | None]] = []
        comparable: list[float] = []
        for source_index in indices:
            if source_index == reference_source_index or source_index not in envelopes:
                continue
            source_values, source_rate = envelopes[source_index]
            local_times = _master_to_source_seconds(
                master_times,
                source_index=source_index,
                audio_offsets=offsets,
                source_timewarps=clocks,
            )
            source_window = np.interp(
                local_times * source_rate,
                np.arange(source_values.size, dtype=np.float64),
                source_values,
                left=np.nan,
                right=np.nan,
            )
            correlation = _window_correlation(reference_window, source_window)
            correlations.append((source_index, correlation))
            if correlation is not None:
                comparable.append(correlation)
        comparable_count = len(comparable)
        passing_fraction = (
            sum(value >= settings.minimum_correlation for value in comparable)
            / comparable_count
            if comparable_count
            else 0.0
        )
        median_correlation = (
            float(np.median(np.asarray(comparable, dtype=np.float64)))
            if comparable
            else None
        )
        shared = bool(
            comparable_count >= required_comparable
            and median_correlation is not None
            and median_correlation >= settings.minimum_correlation
            and passing_fraction >= settings.minimum_passing_source_fraction
        )
        windows.append(
            SharedPerformanceWindow(
                start_seconds=start,
                end_seconds=start + settings.window_seconds,
                median_correlation=median_correlation,
                passing_source_fraction=passing_fraction,
                comparable_source_count=comparable_count,
                required_comparable_source_count=required_comparable,
                shared=shared,
                source_correlations=tuple(correlations),
            )
        )

    clusters = _cluster_supported_windows(
        windows,
        maximum_gap_seconds=settings.maximum_internal_gap_seconds,
    )
    cluster_diagnostics: list[
        tuple[float, int, float, float, tuple[int, ...]]
    ] = []
    for cluster in clusters:
        first_index = cluster[0]
        last_index = cluster[-1]
        span_window_count = last_index - first_index + 1
        density = len(cluster) / span_window_count
        duration = (
            windows[last_index].end_seconds - windows[first_index].start_seconds
        )
        range_fraction = duration / (common_end - common_start)
        if (
            density >= settings.minimum_cluster_density
            and duration >= settings.minimum_shared_duration_seconds
            and range_fraction >= settings.minimum_shared_range_fraction
        ):
            cluster_diagnostics.append(
                (duration, len(cluster), density, -windows[first_index].start_seconds, cluster)
            )
    if not cluster_diagnostics:
        return _unchanged_estimate(
            common_start=common_start,
            common_end=common_end,
            reason="insufficient_dense_shared_audio_cluster",
            expected_peer_count=expected_peer_count,
            available_peer_count=available_peer_count,
            config=settings,
            windows=windows,
        )

    _duration, _count, cluster_density, _negative_start, selected_cluster = max(
        cluster_diagnostics,
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    first_shared = windows[selected_cluster[0]]
    last_shared = windows[selected_cluster[-1]]
    audio_start = first_shared.center_seconds
    audio_end = last_shared.center_seconds
    selected_cluster_duration = last_shared.end_seconds - first_shared.start_seconds

    cuts = tuple(
        sorted(
            {
                float(value)
                for value in reference_scene_cut_master_seconds
                if math.isfinite(float(value)) and common_start < float(value) < common_end
            }
        )
    )
    selected_start_cut: float | None = None
    selected_end_cut: float | None = None

    start_gap = audio_start - common_start
    if start_gap <= settings.start_edge_retention_seconds + 1e-9:
        final_start = common_start
        start_basis = "original_start_within_pose_guard"
    else:
        preceding = [
            cut
            for cut in cuts
            if audio_start - settings.scene_cut_maximum_tail_seconds
            <= cut
            < audio_start
        ]
        if preceding:
            selected_start_cut = preceding[-1]
            final_start = selected_start_cut
            start_basis = "last_reference_hard_cut_before_audio_consensus"
        else:
            final_start = max(
                common_start,
                audio_start - settings.start_pose_guard_seconds,
            )
            start_basis = "audio_consensus_with_bounded_start_pose_guard"

    end_gap = common_end - audio_end
    if end_gap <= settings.scene_cut_maximum_tail_seconds + 1e-9:
        final_end = common_end
        end_basis = "original_end_within_ending_pose_allowance"
    else:
        following = [
            cut
            for cut in cuts
            if audio_end + settings.scene_cut_minimum_tail_seconds
            <= cut
            <= audio_end + settings.scene_cut_maximum_tail_seconds
        ]
        if following:
            selected_end_cut = following[0]
            final_end = selected_end_cut
            end_basis = "first_reference_hard_cut_after_audio_consensus"
        else:
            final_end = min(
                common_end,
                audio_end + settings.scene_cut_maximum_tail_seconds,
            )
            end_basis = "audio_consensus_with_bounded_ending_pose_tail"

    if (
        final_start > common_start + 1e-9
        and final_start - common_start < settings.minimum_trim_seconds
    ):
        final_start = common_start
        start_basis = "original_start_below_minimum_trim"
        selected_start_cut = None
    if (
        final_end < common_end - 1e-9
        and common_end - final_end < settings.minimum_trim_seconds
    ):
        final_end = common_end
        end_basis = "original_end_below_minimum_trim"
        selected_end_cut = None
    if final_end - final_start < settings.minimum_shared_duration_seconds:
        return _unchanged_estimate(
            common_start=common_start,
            common_end=common_end,
            reason="refined_range_too_short",
            expected_peer_count=expected_peer_count,
            available_peer_count=available_peer_count,
            config=settings,
            windows=windows,
        )

    applied = (
        final_start > common_start + 1e-7
        or final_end < common_end - 1e-7
    )
    return SharedPerformanceRangeEstimate(
        original_start_seconds=common_start,
        original_end_seconds=common_end,
        audio_consensus_start_seconds=audio_start,
        audio_consensus_end_seconds=audio_end,
        final_start_seconds=final_start,
        final_end_seconds=final_end,
        applied=applied,
        evidence_strong=True,
        reason=("trim_applied" if applied else "strong_consensus_original_range_retained"),
        start_boundary_basis=start_basis,
        end_boundary_basis=end_basis,
        reference_scene_cut_master_seconds=cuts,
        selected_start_scene_cut_seconds=selected_start_cut,
        selected_end_scene_cut_seconds=selected_end_cut,
        expected_peer_source_count=expected_peer_count,
        available_peer_source_count=available_peer_count,
        selected_cluster_window_count=len(selected_cluster),
        selected_cluster_density=cluster_density,
        selected_cluster_duration_seconds=selected_cluster_duration,
        windows=tuple(windows),
        config=settings,
    )


__all__ = [
    "SharedPerformanceRangeConfig",
    "SharedPerformanceRangeEstimate",
    "SharedPerformanceWindow",
    "estimate_shared_performance_range",
]
