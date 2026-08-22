"""Deterministic metadata-only group cross-edit draft builder.

This module deliberately does **not** claim to judge the pixels in a shot.  It
uses synchronized master time, rhythm boundaries, broadcaster scene-cut
timestamps and audio-sync confidence to make a useful first draft.  A later
visual scorer can replace or re-rank its choices without changing the timing
contract defined here.

Timing contract
---------------
``canonical_master_*`` values are absolute seconds on the reference-song
timeline.  Scene-cut inputs, by contrast, are source-local seconds.  For a
source whose audio alignment offset is ``offset``::

    source_local_time = canonical_master_time + offset

``output_start_seconds`` is relative to ``common_master_start`` and therefore
always starts at zero.  Keeping these three clocks named separately prevents a
positive or negative alignment offset from silently shifting the final edit.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math
from statistics import median
from typing import Any, Sequence


_EPSILON = 1e-8
# The renderer currently emits a fixed 29.97 fps timeline and deliberately
# rejects one-frame flashes.  Do not expose beat candidates close enough to a
# range edge to create such a fragment; the neighbouring musical shot simply
# carries through those final pixels instead.
_MINIMUM_RENDERABLE_SEGMENT_SECONDS = 2.0 / 29.97


def _finite(value: float, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _strictly_increasing(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(_finite(value, name) for value in values)
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return result


@dataclass(frozen=True, slots=True)
class DraftSegment:
    """One immutable source choice with explicitly separated time domains."""

    segment_index: int
    source_index: int
    canonical_master_start: float
    canonical_master_end: float
    output_start_seconds: float
    source_local_start: float
    source_local_end: float
    audio_offset_seconds: float
    sync_score: float
    estimated_beat_count: float
    start_boundary_reasons: tuple[str, ...]
    end_boundary_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.segment_index, bool)
            or not isinstance(self.segment_index, int)
            or self.segment_index < 0
        ):
            raise ValueError("segment_index must be a non-negative integer")
        if (
            isinstance(self.source_index, bool)
            or not isinstance(self.source_index, int)
            or self.source_index < 0
        ):
            raise ValueError("source_index must be a non-negative integer")
        times = (
            self.canonical_master_start,
            self.canonical_master_end,
            self.output_start_seconds,
            self.source_local_start,
            self.source_local_end,
            self.audio_offset_seconds,
            self.estimated_beat_count,
        )
        if any(not math.isfinite(value) for value in times):
            raise ValueError("draft segment values must be finite")
        if self.canonical_master_end <= self.canonical_master_start:
            raise ValueError("draft segment must have positive duration")
        if self.output_start_seconds < -_EPSILON:
            raise ValueError("output_start_seconds cannot be negative")
        if self.source_local_start < -_EPSILON or self.source_local_end < -_EPSILON:
            raise ValueError("source-local times cannot be negative")
        if not 0.0 <= self.sync_score <= 1.0:
            raise ValueError("sync_score must be between zero and one")
        if self.estimated_beat_count <= 0.0:
            raise ValueError("estimated_beat_count must be positive")
        if not math.isclose(
            self.source_local_start,
            self.canonical_master_start + self.audio_offset_seconds,
            abs_tol=1e-7,
        ):
            raise ValueError("source_local_start violates master + offset contract")
        if not math.isclose(
            self.source_local_end,
            self.canonical_master_end + self.audio_offset_seconds,
            abs_tol=1e-7,
        ):
            raise ValueError("source_local_end violates master + offset contract")

    @property
    def duration_seconds(self) -> float:
        return self.canonical_master_end - self.canonical_master_start

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "source_index": self.source_index,
            "canonical_master_start": self.canonical_master_start,
            "canonical_master_end": self.canonical_master_end,
            "output_start_seconds": self.output_start_seconds,
            "source_local_start": self.source_local_start,
            "source_local_end": self.source_local_end,
            "audio_offset_seconds": self.audio_offset_seconds,
            "duration_seconds": self.duration_seconds,
            "sync_score": self.sync_score,
            "estimated_beat_count": self.estimated_beat_count,
            "start_boundary_reasons": list(self.start_boundary_reasons),
            "end_boundary_reasons": list(self.end_boundary_reasons),
        }


@dataclass(frozen=True, slots=True)
class GroupDraftTimeline:
    """Contiguous metadata-only draft over the common synchronized range."""

    source_count: int
    common_master_start: float
    common_master_end: float
    beat_period_seconds: float
    audio_offsets: tuple[float, ...]
    sync_scores: tuple[float, ...]
    segments: tuple[DraftSegment, ...]
    selection_basis: str = "rhythm_scene_metadata_baseline"
    visual_quality_scored: bool = False
    baseline_limitations: tuple[str, ...] = (
        "No pixel-level shot quality, pose, composition, or member score is used.",
        "Every supplied source is treated as usable; failed or unsuitable sources must be filtered upstream.",
    )

    def __post_init__(self) -> None:
        if self.source_count < 2:
            raise ValueError("group draft needs at least two sources")
        if len(self.audio_offsets) != self.source_count:
            raise ValueError("audio_offsets length must match source_count")
        if len(self.sync_scores) != self.source_count:
            raise ValueError("sync_scores length must match source_count")
        if not self.segments:
            raise ValueError("group draft must contain at least one segment")
        if self.visual_quality_scored:
            raise ValueError("this baseline cannot claim visual-quality scoring")

        expected_master = self.common_master_start
        expected_output = 0.0
        for index, segment in enumerate(self.segments):
            if segment.segment_index != index:
                raise ValueError("segment indices must be contiguous")
            if segment.source_index >= self.source_count:
                raise ValueError("segment references an unknown source")
            if not math.isclose(
                segment.canonical_master_start, expected_master, abs_tol=1e-7
            ):
                raise ValueError("segments must contiguously cover canonical master time")
            if not math.isclose(
                segment.output_start_seconds, expected_output, abs_tol=1e-7
            ):
                raise ValueError("segments must contiguously cover output time")
            expected_master = segment.canonical_master_end
            expected_output += segment.duration_seconds
        if not math.isclose(expected_master, self.common_master_end, abs_tol=1e-7):
            raise ValueError("segments must end at common_master_end")

    @property
    def duration_seconds(self) -> float:
        return self.common_master_end - self.common_master_start

    @property
    def used_source_indices(self) -> tuple[int, ...]:
        return tuple(sorted({segment.source_index for segment in self.segments}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_count": self.source_count,
            "common_master_start": self.common_master_start,
            "common_master_end": self.common_master_end,
            "duration_seconds": self.duration_seconds,
            "beat_period_seconds": self.beat_period_seconds,
            "audio_offsets": list(self.audio_offsets),
            "sync_scores": list(self.sync_scores),
            "selection_basis": self.selection_basis,
            "visual_quality_scored": self.visual_quality_scored,
            "baseline_limitations": list(self.baseline_limitations),
            "segments": [segment.to_dict() for segment in self.segments],
        }


@dataclass(frozen=True, slots=True)
class _Boundary:
    time: float
    is_beat: bool
    is_strong_beat: bool
    is_phrase: bool
    scene_sources: tuple[int, ...]
    is_synthetic: bool = False

    @property
    def reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.is_phrase:
            reasons.append("phrase")
        if len(self.scene_sources) > 1:
            reasons.append("multi_source_scene_cut")
        elif self.scene_sources:
            reasons.append("scene_cut")
        if self.is_strong_beat:
            reasons.append("strong_beat")
        elif self.is_beat:
            reasons.append("beat")
        if self.is_synthetic:
            reasons.append("synthetic_coverage_boundary")
        return tuple(reasons) or ("range_edge",)

    @property
    def edit_value(self) -> float:
        return (
            (2.5 if self.is_phrase else 0.0)
            + (1.15 if self.is_strong_beat else 0.30 if self.is_beat else 0.0)
            + min(2.1, 0.72 * len(self.scene_sources))
        )


@dataclass(frozen=True, slots=True)
class _Path:
    score: float
    node_indices: tuple[int, ...]


def _near_index(values: tuple[float, ...], target: float) -> tuple[int, float]:
    position = bisect_left(values, target)
    candidates = []
    if position < len(values):
        candidates.append(position)
    if position:
        candidates.append(position - 1)
    index = min(candidates, key=lambda item: (abs(values[item] - target), item))
    return index, abs(values[index] - target)


def _make_boundaries(
    *,
    common_start: float,
    common_end: float,
    beats: tuple[float, ...],
    phrases: tuple[float, ...],
    canonical_scene_cuts: tuple[tuple[float, ...], ...],
    period: float,
) -> tuple[_Boundary, ...]:
    tolerance = min(0.12, period * 0.22)
    node_times = [common_start]
    node_times.extend(
        beat
        for beat in beats
        if common_start + _MINIMUM_RENDERABLE_SEGMENT_SECONDS <= beat
        <= common_end - _MINIMUM_RENDERABLE_SEGMENT_SECONDS
    )
    node_times.append(common_end)

    # Rhythm analysis normally covers the whole common range.  If it does not,
    # add explicit fallback boundaries so no draft hold can exceed four beats.
    covered: list[tuple[float, bool]] = [(node_times[0], False)]
    for right in node_times[1:]:
        left = covered[-1][0]
        gap = right - left
        parts = max(1, int(math.ceil(gap / (4.0 * period + _EPSILON))))
        for part in range(1, parts):
            covered.append((left + gap * part / parts, True))
        covered.append((right, False))

    boundaries: list[_Boundary] = []
    for time, synthetic in covered:
        beat_index, beat_distance = _near_index(beats, time)
        is_beat = beat_distance <= tolerance
        is_phrase = bool(phrases) and _near_index(phrases, time)[1] <= tolerance
        scene_sources = tuple(
            source_index
            for source_index, cuts in enumerate(canonical_scene_cuts)
            if cuts and _near_index(cuts, time)[1] <= tolerance
        )
        boundaries.append(
            _Boundary(
                time=time,
                is_beat=is_beat,
                is_strong_beat=is_beat and beat_index % 4 == 0,
                is_phrase=is_phrase,
                scene_sources=scene_sources,
                is_synthetic=synthetic,
            )
        )
    return tuple(boundaries)


def _edge_score(
    left: _Boundary,
    right: _Boundary,
    *,
    period: float,
    is_final: bool,
) -> float | None:
    beats = (right.time - left.time) / period
    if beats > 4.25 + _EPSILON:
        return None
    # The common range need not begin/end exactly on a beat.  Only an edge
    # fragment may be shorter than one beat; interior flash cuts are rejected.
    if beats < 0.75 - _EPSILON and left.reasons != ("range_edge",) and not is_final:
        return None
    nearest = min(4, max(1, int(round(beats))))
    length_value = {1: -1.65, 2: 2.35, 3: 1.05, 4: 1.55}[nearest]
    irregularity_penalty = 2.2 * abs(beats - nearest)
    boundary_value = 0.0 if is_final else right.edit_value
    return length_value + boundary_value - irregularity_penalty


def _better_path(candidate: _Path, current: _Path | None) -> bool:
    if current is None:
        return True
    if not math.isclose(candidate.score, current.score, abs_tol=1e-10):
        return candidate.score > current.score
    # Equal-score drafts use fewer cuts, then the lexicographically earliest
    # boundary sequence.  Both tie-breaks are deterministic across runs.
    candidate_cuts = len(candidate.node_indices)
    current_cuts = len(current.node_indices)
    if candidate_cuts != current_cuts:
        return candidate_cuts < current_cuts
    return candidate.node_indices < current.node_indices


def _choose_boundary_path(
    boundaries: tuple[_Boundary, ...], period: float
) -> tuple[int, ...]:
    paths: list[_Path | None] = [None] * len(boundaries)
    paths[0] = _Path(0.0, (0,))
    final_index = len(boundaries) - 1
    for right_index in range(1, len(boundaries)):
        for left_index in range(right_index):
            previous = paths[left_index]
            if previous is None:
                continue
            score = _edge_score(
                boundaries[left_index],
                boundaries[right_index],
                period=period,
                is_final=right_index == final_index,
            )
            if score is None:
                continue
            candidate = _Path(
                previous.score + score,
                previous.node_indices + (right_index,),
            )
            if _better_path(candidate, paths[right_index]):
                paths[right_index] = candidate
    result = paths[-1]
    if result is None:
        raise ValueError("rhythm grid cannot cover the common range in one-to-four-beat cuts")
    return result.node_indices


def _ensure_source_coverage(
    path: tuple[int, ...],
    boundaries: tuple[_Boundary, ...],
    period: float,
    source_count: int,
) -> tuple[int, ...]:
    selected = list(path)
    while len(selected) - 1 < source_count:
        available: list[tuple[float, float, float, int, int]] = []
        selected_set = set(selected)
        for position, (left_index, right_index) in enumerate(zip(selected, selected[1:])):
            original = _edge_score(
                boundaries[left_index],
                boundaries[right_index],
                period=period,
                is_final=right_index == len(boundaries) - 1,
            )
            assert original is not None
            for insert_index in range(left_index + 1, right_index):
                if insert_index in selected_set:
                    continue
                first = _edge_score(
                    boundaries[left_index],
                    boundaries[insert_index],
                    period=period,
                    is_final=False,
                )
                second = _edge_score(
                    boundaries[insert_index],
                    boundaries[right_index],
                    period=period,
                    is_final=right_index == len(boundaries) - 1,
                )
                if first is None or second is None:
                    continue
                gain = first + second - original
                # Prefer musically useful splits, then a balanced split.  The
                # final negative indices make ties deterministic.
                balance = -abs(
                    (boundaries[insert_index].time - boundaries[left_index].time)
                    - (boundaries[right_index].time - boundaries[insert_index].time)
                )
                available.append(
                    (gain, boundaries[insert_index].edit_value, balance, -position, -insert_index)
                )
        if not available:
            raise ValueError(
                "common range is too short to show every source without sub-beat flash cuts"
            )
        best = max(available)
        insert_index = -best[-1]
        selected.append(insert_index)
        selected.sort()
    return tuple(selected)


def _source_for_segment(
    *,
    segment_index: int,
    left: _Boundary,
    right: _Boundary,
    source_order: tuple[int, ...],
    remaining_in_cycle: set[int],
    previous_source: int | None,
    sync_scores: tuple[float, ...],
) -> int:
    candidates = [source for source in source_order if source in remaining_in_cycle]
    without_repeat = [source for source in candidates if source != previous_source]
    if without_repeat:
        candidates = without_repeat

    expected = source_order[segment_index % len(source_order)]
    order_rank = {source: rank for rank, source in enumerate(source_order)}

    def value(source: int) -> tuple[float, int, int]:
        boundary_alignment = (
            (1.2 if segment_index and source in left.scene_sources else 0.0)
            + (0.45 if source in right.scene_sources else 0.0)
        )
        score = 0.85 * sync_scores[source] + boundary_alignment
        if source == expected:
            score += 0.30
        return (score, -order_rank[source], -source)

    return max(candidates, key=value)


def build_group_draft(
    *,
    source_count: int,
    common_master_start: float,
    common_master_end: float,
    audio_offsets: Sequence[float],
    sync_scores: Sequence[float],
    beat_times: Sequence[float],
    phrase_times: Sequence[float] = (),
    scene_cut_times: Sequence[Sequence[float]],
) -> GroupDraftTimeline:
    """Build a deterministic, contiguous group cross-edit first draft.

    Args:
        source_count: Number of synchronized broadcast sources (at least two).
        common_master_start/common_master_end: Absolute reference-song range.
        audio_offsets: Per-source alignment offset.  Reference master ``t`` maps
            to source-local ``t + offset``.
        sync_scores: Per-source audio alignment confidence in ``0...1``.
        beat_times: Strictly increasing canonical master beat timestamps.
        phrase_times: Canonical master phrase boundary timestamps.
        scene_cut_times: Per-source **source-local** camera-cut timestamps.

    The returned draft uses one-to-four-beat holds except for a fractional
    fragment at a non-beat common-range edge.  Every source appears before a
    second rotation begins, and adjacent segments never intentionally repeat a
    source.  This is a metadata-only baseline, not a claim of visual quality.
    """

    if isinstance(source_count, bool) or not isinstance(source_count, int):
        raise TypeError("source_count must be an integer")
    if source_count < 2:
        raise ValueError("group draft needs at least two sources")
    common_start = _finite(common_master_start, "common_master_start")
    common_end = _finite(common_master_end, "common_master_end")
    if common_start < 0.0 or common_end <= common_start:
        raise ValueError("common master range must be non-negative and have positive duration")

    offsets = tuple(_finite(value, "audio_offsets") for value in audio_offsets)
    scores = tuple(_finite(value, "sync_scores") for value in sync_scores)
    if len(offsets) != source_count or len(scores) != source_count:
        raise ValueError("per-source inputs must match source_count")
    if any(not 0.0 <= score <= 1.0 for score in scores):
        raise ValueError("sync_scores must be between zero and one")
    if any(common_start + offset < -_EPSILON for offset in offsets):
        raise ValueError("common range begins before a source's local timeline")

    beats = _strictly_increasing(beat_times, "beat_times")
    if len(beats) < 2:
        raise ValueError("at least two beat_times are required")
    period = median(right - left for left, right in zip(beats, beats[1:]))
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("beat_times do not define a positive period")
    phrases = _strictly_increasing(phrase_times, "phrase_times") if phrase_times else ()

    if len(scene_cut_times) != source_count:
        raise ValueError("scene_cut_times length must match source_count")
    local_scene_cuts = tuple(
        _strictly_increasing(cuts, f"scene_cut_times[{index}]") if cuts else ()
        for index, cuts in enumerate(scene_cut_times)
    )
    if any(cut < 0.0 for cuts in local_scene_cuts for cut in cuts):
        raise ValueError("scene cut timestamps cannot be negative")
    canonical_scene_cuts = tuple(
        tuple(cut - offsets[source_index] for cut in cuts)
        for source_index, cuts in enumerate(local_scene_cuts)
    )

    boundaries = _make_boundaries(
        common_start=common_start,
        common_end=common_end,
        beats=beats,
        phrases=phrases,
        canonical_scene_cuts=canonical_scene_cuts,
        period=period,
    )
    path = _choose_boundary_path(boundaries, period)
    path = _ensure_source_coverage(path, boundaries, period, source_count)

    source_order = tuple(sorted(range(source_count), key=lambda item: (-scores[item], item)))
    remaining_in_cycle = set(source_order)
    previous_source: int | None = None
    segments: list[DraftSegment] = []
    for segment_index, (left_index, right_index) in enumerate(zip(path, path[1:])):
        if not remaining_in_cycle:
            remaining_in_cycle = set(source_order)
        left = boundaries[left_index]
        right = boundaries[right_index]
        source = _source_for_segment(
            segment_index=segment_index,
            left=left,
            right=right,
            source_order=source_order,
            remaining_in_cycle=remaining_in_cycle,
            previous_source=previous_source,
            sync_scores=scores,
        )
        remaining_in_cycle.remove(source)
        offset = offsets[source]
        segments.append(
            DraftSegment(
                segment_index=segment_index,
                source_index=source,
                canonical_master_start=left.time,
                canonical_master_end=right.time,
                output_start_seconds=left.time - common_start,
                source_local_start=left.time + offset,
                source_local_end=right.time + offset,
                audio_offset_seconds=offset,
                sync_score=scores[source],
                estimated_beat_count=(right.time - left.time) / period,
                start_boundary_reasons=("range_start",) if segment_index == 0 else left.reasons,
                end_boundary_reasons=("range_end",) if right_index == len(boundaries) - 1 else right.reasons,
            )
        )
        previous_source = source

    return GroupDraftTimeline(
        source_count=source_count,
        common_master_start=common_start,
        common_master_end=common_end,
        beat_period_seconds=period,
        audio_offsets=offsets,
        sync_scores=scores,
        segments=tuple(segments),
    )
