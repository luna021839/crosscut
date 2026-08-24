"""Relative source-quality triage for automatic GROUP cross-edits.

The hard acquisition contract answers whether a file is usable at all.  This
module answers a narrower question: when the default automatic request has
nine or ten usable broadcast stages, which eight form the most consistent
editing pool?  It deliberately remains relative and fail-open.  An eight-file
pool is never reduced, and incomplete visual evidence never invents a quality
rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .media import MediaInfo


GROUP_RELATIVE_QUALITY_GATE_VERSION = 1
GROUP_RELATIVE_QUALITY_TARGET = 8
GROUP_RELATIVE_QUALITY_SAMPLE_FRACTIONS = (
    0.075,
    0.19642857142857142,
    0.31785714285714284,
    0.4392857142857143,
    0.5607142857142857,
    0.6821428571428572,
    0.8035714285714286,
    0.925,
)
_WORKING_WIDTH = 480


def _fold(value: str) -> str:
    return " ".join(value.casefold().replace("\u3000", " ").split())


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p10": _percentile(values, 10.0),
        "median": _percentile(values, 50.0),
        "p90": _percentile(values, 90.0),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def _working_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if width == _WORKING_WIDTH:
        return frame
    scale = _WORKING_WIDTH / width
    return cv2.resize(
        frame,
        (_WORKING_WIDTH, max(2, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )


def _frame_metrics(frame: np.ndarray) -> dict[str, float]:
    working = _working_frame(frame)
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    sharpness = 1.0 - math.exp(-laplacian_variance / 180.0)
    mean_luma = float(gray.mean())
    contrast = float(gray.std())
    clipped = float(np.mean((gray < 8) | (gray > 247)))
    exposure = math.exp(-((mean_luma - 118.0) / 105.0) ** 2) * (
        1.0 - min(0.8, clipped)
    )
    contrast_score = min(1.0, contrast / 58.0)
    quality = min(
        1.0,
        max(
            0.0,
            0.45 * sharpness + 0.32 * exposure + 0.23 * contrast_score,
        ),
    )
    return {
        "quality_0_100": 100.0 * quality,
        "sharpness_0_100": 100.0 * sharpness,
        "exposure_0_100": 100.0 * exposure,
        "total_clip_ratio": clipped,
    }


def _camera_variant(title: str) -> str:
    folded = _fold(title)
    if any(marker in folded for marker in ("one take", "one-shot", "oneshot", "원테이크")):
        return "one_take"
    if any(marker in folded for marker in ("followcam", "follow cam", "팔로우캠")):
        return "follow_cam"
    if any(marker in folded for marker in ("fullcam", "full cam", "풀캠")):
        return "full_cam"
    return "standard_programme_feed"


def _broadcast_family(channel: str, title: str) -> str:
    folded = _fold(f"{channel} {title}")
    families = (
        ("kbs", ("kbs", "music bank", "musicbank", "뮤직뱅크")),
        ("mbc", ("mbc", "music core", "musiccore", "음악중심")),
        ("sbs", ("sbs", "inkigayo", "인기가요")),
        ("mnet", ("mnet", "m countdown", "mcountdown", "엠카운트다운")),
        ("show_champion", ("show champion", "쇼챔피언")),
        ("the_show", ("the show", "더쇼")),
    )
    for family, markers in families:
        if any(marker in folded for marker in markers):
            return family
    return "unknown"


def _group_structure(title: str) -> tuple[float, tuple[str, ...]]:
    """Return a soft GROUP suitability signal, never a hard title rejection."""

    folded = _fold(title)
    if any(
        marker in folded
        for marker in (
            "mpd직캠",
            "facecam",
            "face cam",
            "입덕직캠",
            "페이스캠",
        )
    ):
        return (0.05, ("individual_or_closeup_fancam_structure",))
    if (
        ("fancam" in folded or "직캠" in folded)
        and not any(
            marker in folded
            for marker in ("fullcam", "full cam", "풀캠", "one take", "원테이크")
        )
    ):
        return (0.28, ("generic_fancam_structure",))
    if any(marker in folded for marker in ("fullcam", "full cam", "풀캠")):
        return (0.88, ("group_fullcam_structure",))
    if any(marker in folded for marker in ("one take", "one-shot", "oneshot", "원테이크")):
        return (0.94, ("group_one_take_structure",))
    return (1.0, ("standard_music_show_programme_feed",))


def _metadata_score(info: MediaInfo) -> float:
    if not info.video_streams:
        return 0.0
    stream = info.video_streams[0]
    width = stream.width or 0
    height = stream.height or 0
    short_edge = min(width, height)
    resolution = min(1.0, short_edge / 1080.0)
    frame_rate = min(1.0, (stream.frame_rate or 0.0) / 29.97)
    bit_rate = stream.bit_rate or info.bit_rate or 0
    bitrate = min(1.0, bit_rate / 4_000_000.0)
    return 100.0 * (0.42 * resolution + 0.18 * frame_rate + 0.40 * bitrate)


@dataclass(frozen=True, slots=True)
class GroupSourceQualityAssessment:
    rankable: bool
    analysis_error: str | None
    sample_count: int
    requested_sample_fractions: tuple[float, ...]
    visual_quality_index_0_100: float | None
    metadata_score_0_100: float
    structure_score_0_100: float
    ranking_score_0_100: float | None
    distributions: dict[str, dict[str, float]]
    structure_signals: tuple[str, ...]
    duplicate_feed_key: str | None

    @classmethod
    def unavailable(
        cls,
        reason: str,
        *,
        metadata_score: float = 0.0,
        structure_score: float = 100.0,
        structure_signals: Sequence[str] = (),
        duplicate_feed_key: str | None = None,
    ) -> "GroupSourceQualityAssessment":
        return cls(
            rankable=False,
            analysis_error=reason,
            sample_count=0,
            requested_sample_fractions=GROUP_RELATIVE_QUALITY_SAMPLE_FRACTIONS,
            visual_quality_index_0_100=None,
            metadata_score_0_100=metadata_score,
            structure_score_0_100=structure_score,
            ranking_score_0_100=None,
            distributions={},
            structure_signals=tuple(structure_signals),
            duplicate_feed_key=duplicate_feed_key,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_version": GROUP_RELATIVE_QUALITY_GATE_VERSION,
            "rankable": self.rankable,
            "analysis_error": self.analysis_error,
            "sample_count": self.sample_count,
            "requested_sample_fractions": list(self.requested_sample_fractions),
            "working_width": _WORKING_WIDTH,
            "visual_quality_index_0_100": self.visual_quality_index_0_100,
            "metadata_score_0_100": self.metadata_score_0_100,
            "structure_score_0_100": self.structure_score_0_100,
            "ranking_score_0_100": self.ranking_score_0_100,
            "distributions": self.distributions,
            "structure_signals": list(self.structure_signals),
            "duplicate_feed_key": self.duplicate_feed_key,
            "raw_frames_persisted": False,
        }


def analyze_group_source_quality(
    path: Path,
    info: MediaInfo,
    *,
    title: str,
    channel: str,
    broadcast_date: str | None,
) -> GroupSourceQualityAssessment:
    """Sample eight source-relative frames and compute audit-calibrated metrics."""

    structure, structure_signals = _group_structure(title)
    metadata = _metadata_score(info)
    family = _broadcast_family(channel, title)
    duplicate_key = (
        f"{family}:{broadcast_date}:{_camera_variant(title)}"
        if family != "unknown" and broadcast_date is not None
        else None
    )
    duration = info.duration_seconds
    if duration is None or not math.isfinite(duration) or duration <= 0.0:
        return GroupSourceQualityAssessment.unavailable(
            "duration_unavailable_for_relative_quality_sampling",
            metadata_score=metadata,
            structure_score=100.0 * structure,
            structure_signals=structure_signals,
            duplicate_feed_key=duplicate_key,
        )
    try:
        if path.is_file() and path.stat().st_size <= 0:
            return GroupSourceQualityAssessment.unavailable(
                "empty_file_relative_quality_sampling_unavailable",
                metadata_score=metadata,
                structure_score=100.0 * structure,
                structure_signals=structure_signals,
                duplicate_feed_key=duplicate_key,
            )
    except OSError as exc:
        return GroupSourceQualityAssessment.unavailable(
            f"file_stat_failed:{type(exc).__name__}:{exc}",
            metadata_score=metadata,
            structure_score=100.0 * structure,
            structure_signals=structure_signals,
            duplicate_feed_key=duplicate_key,
        )
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return GroupSourceQualityAssessment.unavailable(
            "opencv_video_open_failed",
            metadata_score=metadata,
            structure_score=100.0 * structure,
            structure_signals=structure_signals,
            duplicate_feed_key=duplicate_key,
        )
    samples: list[dict[str, float]] = []
    try:
        for fraction in GROUP_RELATIVE_QUALITY_SAMPLE_FRACTIONS:
            seconds = fraction * duration
            capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
            success, frame = capture.read()
            if not success or frame is None or frame.size == 0:
                return GroupSourceQualityAssessment.unavailable(
                    f"frame_decode_failed_at_fraction_{fraction:.6f}",
                    metadata_score=metadata,
                    structure_score=100.0 * structure,
                    structure_signals=structure_signals,
                    duplicate_feed_key=duplicate_key,
                )
            samples.append(_frame_metrics(frame))
    finally:
        capture.release()

    distributions = {
        name: _distribution([sample[name] for sample in samples])
        for name in samples[0]
    }
    quality = distributions["quality_0_100"]
    sharpness = distributions["sharpness_0_100"]
    exposure = distributions["exposure_0_100"]
    clipping = distributions["total_clip_ratio"]
    consistency = max(0.0, 100.0 - quality["std"] * 2.5)
    clipping_score = 100.0 * max(
        0.0,
        1.0 - min(1.0, clipping["p90"] / 0.20),
    )
    visual_index = (
        0.30 * quality["median"]
        + 0.25 * quality["p10"]
        + 0.18 * sharpness["median"]
        + 0.12 * sharpness["p10"]
        + 0.08 * exposure["median"]
        + 0.04 * consistency
        + 0.03 * clipping_score
    )
    # Perceptual evidence dominates.  Metadata breaks close ties, while the
    # structural penalty ensures an MPD/individual fancam does not displace a
    # normal broadcast programme feed merely because it is artificially sharp.
    ranking_score = min(
        100.0,
        max(
            0.0,
            0.94 * visual_index
            + 0.06 * metadata
            - 12.0 * (1.0 - structure),
        ),
    )
    return GroupSourceQualityAssessment(
        rankable=True,
        analysis_error=None,
        sample_count=len(samples),
        requested_sample_fractions=GROUP_RELATIVE_QUALITY_SAMPLE_FRACTIONS,
        visual_quality_index_0_100=visual_index,
        metadata_score_0_100=metadata,
        structure_score_0_100=100.0 * structure,
        ranking_score_0_100=ranking_score,
        distributions=distributions,
        structure_signals=structure_signals,
        duplicate_feed_key=duplicate_key,
    )


@dataclass(frozen=True, slots=True)
class RankedGroupSource:
    acquisition_index: int
    video_id: str
    assessment: GroupSourceQualityAssessment


@dataclass(frozen=True, slots=True)
class GroupSourceQualityDecision:
    selected_acquisition_indices: tuple[int, ...]
    relative_ranks: dict[int, int]
    exclusion_reasons: dict[int, tuple[str, ...]]
    gate: dict[str, Any]


def select_group_quality_pool(
    sources: Sequence[RankedGroupSource],
    *,
    target_count: int = GROUP_RELATIVE_QUALITY_TARGET,
) -> GroupSourceQualityDecision:
    """Select a diverse top-eight pool only when nine or ten are rankable."""

    if target_count < 1:
        raise ValueError("target_count must be positive")
    input_count = len(sources)
    rankable = tuple(source for source in sources if source.assessment.rankable)
    base_gate: dict[str, Any] = {
        "gate_version": GROUP_RELATIVE_QUALITY_GATE_VERSION,
        "scope": "automatic_group_default_ten_only",
        "target_eligible_source_count": target_count,
        "input_source_count": input_count,
        "rankable_source_count": len(rankable),
        "sample_count_per_rankable_source": len(
            GROUP_RELATIVE_QUALITY_SAMPLE_FRACTIONS
        ),
        "quality_formula": (
            "audit_calibrated_quality_p10_median_sharpness_exposure_"
            "consistency_clipping_plus_metadata_and_group_structure"
        ),
        "duplicate_policy": (
            "prefer_one_per_broadcaster_date_camera_variant_then_fill_to_eight"
        ),
        "minimum_eight_policy": "never_reduce_an_eight_source_pool",
    }
    if input_count <= target_count:
        selected = tuple(source.acquisition_index for source in sources)
        return GroupSourceQualityDecision(
            selected_acquisition_indices=selected,
            relative_ranks={},
            exclusion_reasons={},
            gate={
                **base_gate,
                "status": "not_applied_pool_already_at_minimum_eight",
                "applied": False,
                "selected_source_count": len(selected),
            },
        )
    if len(rankable) not in (9, 10):
        selected = tuple(source.acquisition_index for source in sources)
        return GroupSourceQualityDecision(
            selected_acquisition_indices=selected,
            relative_ranks={},
            exclusion_reasons={},
            gate={
                **base_gate,
                "status": "not_applied_insufficient_rankable_sources",
                "applied": False,
                "selected_source_count": len(selected),
            },
        )

    ordered = sorted(
        rankable,
        key=lambda source: (
            source.assessment.structure_score_0_100 < 50.0,
            -(source.assessment.ranking_score_0_100 or 0.0),
            source.video_id,
        ),
    )
    ranks = {
        source.acquisition_index: rank
        for rank, source in enumerate(
            sorted(
                rankable,
                key=lambda source: (
                    -(source.assessment.ranking_score_0_100 or 0.0),
                    source.video_id,
                ),
            ),
            start=1,
        )
    }
    selected: list[RankedGroupSource] = []
    used_duplicate_keys: set[str] = set()
    deferred_duplicates: list[RankedGroupSource] = []
    for source in ordered:
        duplicate_key = source.assessment.duplicate_feed_key
        if duplicate_key is not None and duplicate_key in used_duplicate_keys:
            deferred_duplicates.append(source)
            continue
        selected.append(source)
        if duplicate_key is not None:
            used_duplicate_keys.add(duplicate_key)
        if len(selected) == target_count:
            break
    if len(selected) < target_count:
        already_selected = {source.acquisition_index for source in selected}
        for source in (*deferred_duplicates, *ordered):
            if source.acquisition_index in already_selected:
                continue
            selected.append(source)
            already_selected.add(source.acquisition_index)
            if len(selected) == target_count:
                break

    selected_indices = tuple(source.acquisition_index for source in selected)
    selected_set = set(selected_indices)
    selected_duplicate_keys = {
        source.assessment.duplicate_feed_key
        for source in selected
        if source.assessment.duplicate_feed_key is not None
    }
    exclusions: dict[int, tuple[str, ...]] = {}
    for source in sources:
        if source.acquisition_index in selected_set:
            continue
        reasons = ["relative_quality_below_top_eight"]
        if source.assessment.structure_score_0_100 < 50.0:
            reasons.append("group_structure_deprioritized")
        duplicate_key = source.assessment.duplicate_feed_key
        if duplicate_key is not None and duplicate_key in selected_duplicate_keys:
            reasons.append("duplicate_broadcast_feed_deprioritized")
        if not source.assessment.rankable:
            reasons.append("relative_quality_analysis_unavailable")
        exclusions[source.acquisition_index] = tuple(reasons)
    return GroupSourceQualityDecision(
        selected_acquisition_indices=selected_indices,
        relative_ranks=ranks,
        exclusion_reasons=exclusions,
        gate={
            **base_gate,
            "status": "applied_top_eight_selected",
            "applied": True,
            "selected_source_count": len(selected_indices),
            "selected_acquisition_indices": list(selected_indices),
            "excluded_acquisition_indices": sorted(exclusions),
        },
    )


__all__ = [
    "GROUP_RELATIVE_QUALITY_GATE_VERSION",
    "GROUP_RELATIVE_QUALITY_SAMPLE_FRACTIONS",
    "GROUP_RELATIVE_QUALITY_TARGET",
    "GroupSourceQualityAssessment",
    "GroupSourceQualityDecision",
    "RankedGroupSource",
    "analyze_group_source_quality",
    "select_group_quality_pool",
]
