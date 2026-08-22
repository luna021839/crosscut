"""Safety-first IVE member cross-fancam draft selection.

The group draft decides *when* a cut may happen.  This adapter decides whether
one of the synchronized sources can safely supply a crop of the requested IVE
member for each canonical span.  It intentionally accepts only verified
``TRACKED`` observations.  Holding/lost boxes and low-confidence identity
observations never become crops; when no source passes the interval gate, the
result explicitly requests a caller-designated group-wide source or marks the
span unavailable.

Three clocks are kept separate throughout::

    source_local_time = canonical_master_time + audio_offset
    output_time = canonical_master_time - common_master_start
    crop_sample_time = source_local_time - source_local_span_start

All crop rectangles use normalized source coordinates ``(x, y, width,
height)``.  In particular, a 9:16 pixel crop from a 16:9 source has normalized
width/height ``(9/16) / (16/9) == 81/256``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from statistics import median, pstdev
from typing import Any, Mapping, Sequence

from .ive import AspectRatio, IVEMember
from .tracking import MemberTrackObservation, MemberTrackingResult, TrackingState
from .vision import TrackObservation, VirtualCameraConfig, plan_virtual_camera


_EPSILON = 1e-8
_EXACT_SEQUENCE_STATE_BUDGET = 100_000
_BALANCED_PATH_BEAM_WIDTH = 4_096
TrackingInput = (
    MemberTrackingResult
    | Sequence[MemberTrackObservation | TrackObservation]
    | None
)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _probability(value: object, name: str) -> float:
    numeric = _finite(value, name)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return numeric


@dataclass(frozen=True, slots=True)
class CanonicalSpan:
    """One edit span in absolute reference-song time."""

    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        start = _finite(self.start_seconds, "canonical span start")
        end = _finite(self.end_seconds, "canonical span end")
        if start < 0:
            raise ValueError("canonical span start cannot be negative")
        if end <= start:
            raise ValueError("canonical span must have positive duration")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True, slots=True)
class MemberDraftConfig:
    """Confidence, coverage, and source-variety gates for member drafts."""

    minimum_confidence: float = 0.72
    minimum_interval_coverage: float = 0.88
    maximum_untracked_gap_seconds: float = 0.16
    maximum_quality_score_drop: float = 0.04
    repeat_source_penalty: float = 0.12
    accumulated_use_penalty: float = 0.015
    continuity_weight: float = 0.04
    exact_diversity_source_limit: int = 12
    confidence_weight: float = 0.55
    coverage_weight: float = 0.25
    stability_weight: float = 0.20

    def __post_init__(self) -> None:
        for name in (
            "minimum_confidence",
            "minimum_interval_coverage",
            "maximum_quality_score_drop",
            "confidence_weight",
            "coverage_weight",
            "stability_weight",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        for name in (
            "maximum_untracked_gap_seconds",
            "repeat_source_penalty",
            "accumulated_use_penalty",
            "continuity_weight",
        ):
            numeric = _finite(getattr(self, name), name)
            if name == "maximum_untracked_gap_seconds" and numeric <= 0:
                raise ValueError(f"{name} must be positive")
            if name != "maximum_untracked_gap_seconds" and numeric < 0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, numeric)
        if (
            isinstance(self.exact_diversity_source_limit, bool)
            or not isinstance(self.exact_diversity_source_limit, int)
            or self.exact_diversity_source_limit < 1
        ):
            raise ValueError("exact_diversity_source_limit must be a positive integer")
        if (
            self.confidence_weight
            + self.coverage_weight
            + self.stability_weight
            <= 0
        ):
            raise ValueError("at least one selection score weight must be positive")


@dataclass(frozen=True, slots=True)
class SegmentCropSample:
    """A renderer-ready, segment-relative normalized crop sample."""

    time_seconds: float
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        time = _finite(self.time_seconds, "crop sample time")
        values = tuple(
            _finite(value, "crop sample coordinate")
            for value in (self.x, self.y, self.width, self.height)
        )
        x, y, width, height = values
        if time < 0:
            raise ValueError("crop sample time cannot be negative")
        if not (
            0.0 <= x <= 1.0
            and 0.0 <= y <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
            and x + width <= 1.0 + 1e-7
            and y + height <= 1.0 + 1e-7
        ):
            raise ValueError("crop sample must fit inside normalized source bounds")
        object.__setattr__(self, "time_seconds", time)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)

    def to_dict(self) -> dict[str, float]:
        return {
            "time_seconds": self.time_seconds,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


class MemberSegmentKind(StrEnum):
    TARGET_CROP = "target_crop"
    GROUP_WIDE_FALLBACK = "group_wide_fallback"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class MemberDraftSegment:
    """One safe source decision for a canonical edit span."""

    segment_index: int
    canonical_master_start: float
    canonical_master_end: float
    output_start_seconds: float
    kind: MemberSegmentKind
    source_index: int | None
    source_local_start: float | None
    source_local_end: float | None
    audio_offset_seconds: float | None
    crop_samples: tuple[SegmentCropSample, ...]
    interval_coverage: float
    mean_confidence: float
    stability: float
    selection_score: float | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.segment_index, bool)
            or not isinstance(self.segment_index, int)
            or self.segment_index < 0
        ):
            raise ValueError("segment_index must be a non-negative integer")
        start = _finite(self.canonical_master_start, "canonical master start")
        end = _finite(self.canonical_master_end, "canonical master end")
        output = _finite(self.output_start_seconds, "output start")
        if end <= start or output < -_EPSILON:
            raise ValueError("member draft segment has invalid timing")
        if not isinstance(self.kind, MemberSegmentKind):
            object.__setattr__(self, "kind", MemberSegmentKind(self.kind))
        for name in ("interval_coverage", "mean_confidence", "stability"):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        if self.selection_score is not None:
            object.__setattr__(
                self,
                "selection_score",
                _finite(self.selection_score, "selection_score"),
            )

        if self.kind is MemberSegmentKind.UNAVAILABLE:
            if any(
                value is not None
                for value in (
                    self.source_index,
                    self.source_local_start,
                    self.source_local_end,
                    self.audio_offset_seconds,
                )
            ) or self.crop_samples:
                raise ValueError("unavailable segments cannot claim a source or crop")
            return

        if (
            isinstance(self.source_index, bool)
            or not isinstance(self.source_index, int)
            or self.source_index < 0
        ):
            raise ValueError("available member segments need a source_index")
        if any(
            value is None
            for value in (
                self.source_local_start,
                self.source_local_end,
                self.audio_offset_seconds,
            )
        ):
            raise ValueError("available member segments need source-local timing")
        local_start = _finite(self.source_local_start, "source local start")
        local_end = _finite(self.source_local_end, "source local end")
        offset = _finite(self.audio_offset_seconds, "audio offset")
        if local_start < -_EPSILON or local_end <= local_start:
            raise ValueError("source-local timing is invalid")
        if not math.isclose(local_start, start + offset, abs_tol=1e-7):
            raise ValueError("source_local_start violates master + offset contract")
        if not math.isclose(local_end, end + offset, abs_tol=1e-7):
            raise ValueError("source_local_end violates master + offset contract")
        if self.kind is MemberSegmentKind.TARGET_CROP:
            if not self.crop_samples:
                raise ValueError("target crop segments need crop samples")
            if abs(self.crop_samples[0].time_seconds) > 1e-7:
                raise ValueError("crop samples must start at segment-relative zero")
            previous = -1.0
            for sample in self.crop_samples:
                if sample.time_seconds <= previous:
                    raise ValueError("crop sample times must be strictly increasing")
                if sample.time_seconds > end - start + 1e-7:
                    raise ValueError("crop sample lies outside its segment")
                previous = sample.time_seconds
        elif self.crop_samples:
            raise ValueError("group-wide fallback must not claim a member crop")

    @property
    def duration_seconds(self) -> float:
        return self.canonical_master_end - self.canonical_master_start

    def to_pan_crop_track(self) -> object | None:
        """Adapt target samples to the FFmpeg renderer's panning crop type.

        The import stays local so member selection does not create a module
        dependency for callers that only need the JSON draft.
        """

        if self.kind is not MemberSegmentKind.TARGET_CROP:
            return None
        from .render import NormalizedCrop, PanCropKeyframe, PanCropTrack

        return PanCropTrack(
            tuple(
                PanCropKeyframe(
                    time_seconds=sample.time_seconds,
                    crop=NormalizedCrop(
                        x=sample.x,
                        y=sample.y,
                        width=sample.width,
                        height=sample.height,
                    ),
                )
                for sample in self.crop_samples
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "canonical_master_start": self.canonical_master_start,
            "canonical_master_end": self.canonical_master_end,
            "output_start_seconds": self.output_start_seconds,
            "duration_seconds": self.duration_seconds,
            "kind": self.kind.value,
            "source_index": self.source_index,
            "source_local_start": self.source_local_start,
            "source_local_end": self.source_local_end,
            "audio_offset_seconds": self.audio_offset_seconds,
            "interval_coverage": self.interval_coverage,
            "mean_confidence": self.mean_confidence,
            "stability": self.stability,
            "selection_score": self.selection_score,
            "crop_samples": [sample.to_dict() for sample in self.crop_samples],
        }


@dataclass(frozen=True, slots=True)
class MemberDraftTimeline:
    target_member: IVEMember
    output_aspect: AspectRatio
    source_aspect_ratio: float
    common_master_start: float
    common_master_end: float
    audio_offsets: tuple[float, ...]
    group_wide_fallback_source: int | None
    segments: tuple[MemberDraftSegment, ...]
    selection_diagnostics: "MemberDraftSelectionDiagnostics"

    def __post_init__(self) -> None:
        if not isinstance(self.target_member, IVEMember):
            object.__setattr__(self, "target_member", IVEMember.parse(self.target_member))
        if not isinstance(self.output_aspect, AspectRatio):
            object.__setattr__(self, "output_aspect", AspectRatio(self.output_aspect))
        source_aspect = _finite(self.source_aspect_ratio, "source aspect ratio")
        if source_aspect <= 0:
            raise ValueError("source aspect ratio must be positive")
        object.__setattr__(self, "source_aspect_ratio", source_aspect)
        common_start = _finite(self.common_master_start, "common master start")
        common_end = _finite(self.common_master_end, "common master end")
        if common_start < 0 or common_end <= common_start:
            raise ValueError("member draft has an invalid common master range")
        object.__setattr__(self, "common_master_start", common_start)
        object.__setattr__(self, "common_master_end", common_end)
        offsets = tuple(
            _finite(value, f"audio_offsets[{index}]")
            for index, value in enumerate(self.audio_offsets)
        )
        if not offsets:
            raise ValueError("member draft needs at least one audio offset")
        object.__setattr__(self, "audio_offsets", offsets)
        if self.group_wide_fallback_source is not None and (
            isinstance(self.group_wide_fallback_source, bool)
            or not isinstance(self.group_wide_fallback_source, int)
            or not 0 <= self.group_wide_fallback_source < len(offsets)
        ):
            raise ValueError("group-wide fallback source is out of range")
        if not self.segments:
            raise ValueError("member draft needs at least one segment")
        expected_master = self.common_master_start
        expected_output = 0.0
        for index, segment in enumerate(self.segments):
            if segment.segment_index != index:
                raise ValueError("member draft segment indices must be contiguous")
            if not math.isclose(
                segment.canonical_master_start, expected_master, abs_tol=1e-7
            ):
                raise ValueError("member draft must be contiguous in master time")
            if not math.isclose(
                segment.output_start_seconds, expected_output, abs_tol=1e-7
            ):
                raise ValueError("member draft must be contiguous in output time")
            expected_master = segment.canonical_master_end
            expected_output += segment.duration_seconds
            if segment.source_index is not None and segment.source_index >= len(
                self.audio_offsets
            ):
                raise ValueError("segment references an unknown source")
            for sample in segment.crop_samples:
                if not math.isclose(
                    sample.width / sample.height,
                    self.normalized_crop_aspect,
                    rel_tol=0.0,
                    abs_tol=2e-6,
                ):
                    raise ValueError("crop sample violates normalized output aspect")
        if not math.isclose(expected_master, self.common_master_end, abs_tol=1e-7):
            raise ValueError("member draft must end at common_master_end")

    @property
    def duration_seconds(self) -> float:
        return self.common_master_end - self.common_master_start

    @property
    def normalized_crop_aspect(self) -> float:
        pixel_aspect = 16.0 / 9.0
        if self.output_aspect is AspectRatio.PORTRAIT:
            pixel_aspect = 9.0 / 16.0
        return pixel_aspect / self.source_aspect_ratio

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_member": self.target_member.value,
            "output_aspect": self.output_aspect.value,
            "source_aspect_ratio": self.source_aspect_ratio,
            "normalized_crop_aspect": self.normalized_crop_aspect,
            "common_master_start": self.common_master_start,
            "common_master_end": self.common_master_end,
            "duration_seconds": self.duration_seconds,
            "audio_offsets": list(self.audio_offsets),
            "group_wide_fallback_source": self.group_wide_fallback_source,
            "selection_diagnostics": self.selection_diagnostics.to_dict(),
            "segments": [segment.to_dict() for segment in self.segments],
        }


@dataclass(frozen=True, slots=True)
class MemberDraftSelectionDiagnostics:
    """Explain which sources were safe, quality-eligible, and selected."""

    policy: str
    optimizer: str
    quality_score_drop_limit: float
    maximum_distinct_quality_source_count: int
    safe_candidate_counts_by_segment: tuple[int, ...]
    quality_candidate_counts_by_segment: tuple[int, ...]
    safe_candidate_counts_by_source: tuple[int, ...]
    quality_candidate_counts_by_source: tuple[int, ...]
    selected_segment_counts_by_source: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        safe_sources = tuple(
            index
            for index, count in enumerate(self.safe_candidate_counts_by_source)
            if count > 0
        )
        quality_sources = tuple(
            index
            for index, count in enumerate(self.quality_candidate_counts_by_source)
            if count > 0
        )
        selected_sources = tuple(
            index
            for index, count in enumerate(self.selected_segment_counts_by_source)
            if count > 0
        )
        sources: list[dict[str, Any]] = []
        for source_index, (safe_count, quality_count, selected_count) in enumerate(
            zip(
                self.safe_candidate_counts_by_source,
                self.quality_candidate_counts_by_source,
                self.selected_segment_counts_by_source,
                strict=True,
            )
        ):
            if safe_count == 0:
                outcome = "no_safe_candidate"
            elif quality_count == 0:
                outcome = "quality_guard_excluded"
            elif selected_count == 0:
                outcome = "quality_eligible_not_selected"
            else:
                outcome = "selected"
            sources.append(
                {
                    "source_index": source_index,
                    "safe_candidate_count": safe_count,
                    "quality_candidate_count": quality_count,
                    "selected_segment_count": selected_count,
                    "outcome": outcome,
                }
            )
        return {
            "policy": self.policy,
            "optimizer": self.optimizer,
            "quality_score_drop_limit": self.quality_score_drop_limit,
            "maximum_distinct_quality_source_count": (
                self.maximum_distinct_quality_source_count
            ),
            "selected_distinct_source_count": len(selected_sources),
            "safe_source_indices": list(safe_sources),
            "quality_eligible_source_indices": list(quality_sources),
            "selected_source_indices": list(selected_sources),
            "no_safe_candidate_source_indices": [
                index
                for index in range(len(self.safe_candidate_counts_by_source))
                if index not in safe_sources
            ],
            "quality_guard_excluded_source_indices": [
                index
                for index in safe_sources
                if index not in quality_sources
            ],
            "unused_quality_source_indices": [
                index
                for index in quality_sources
                if index not in selected_sources
            ],
            "safe_candidate_counts_by_segment": list(
                self.safe_candidate_counts_by_segment
            ),
            "quality_candidate_counts_by_segment": list(
                self.quality_candidate_counts_by_segment
            ),
            "quality_guard_rejected_candidate_count": sum(
                self.safe_candidate_counts_by_segment
            )
            - sum(self.quality_candidate_counts_by_segment),
            "sources": sources,
        }


@dataclass(frozen=True, slots=True)
class _PreparedTrack:
    source_index: int
    fps: float
    observations: tuple[TrackObservation, ...]
    timestamps_by_frame: Mapping[int, float]


@dataclass(frozen=True, slots=True)
class _Candidate:
    source_index: int
    local_start: float
    local_end: float
    coverage: float
    mean_confidence: float
    stability: float
    base_score: float
    crop_samples: tuple[SegmentCropSample, ...]
    entry_framing: tuple[float, float, float]
    exit_framing: tuple[float, float, float]


def _coerce_span(value: object) -> CanonicalSpan:
    if isinstance(value, CanonicalSpan):
        return value
    if hasattr(value, "canonical_master_start") and hasattr(
        value, "canonical_master_end"
    ):
        return CanonicalSpan(
            getattr(value, "canonical_master_start"),
            getattr(value, "canonical_master_end"),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError("canonical span tuples must contain start and end")
        return CanonicalSpan(value[0], value[1])
    raise TypeError("canonical spans must be CanonicalSpan, pairs, or draft segments")


def _infer_fps(observations: Sequence[TrackObservation]) -> float:
    rates: list[float] = []
    ordered = sorted(observations, key=lambda item: item.frame_index)
    for left, right in zip(ordered, ordered[1:]):
        if left.timestamp_seconds is None or right.timestamp_seconds is None:
            continue
        frame_delta = right.frame_index - left.frame_index
        time_delta = right.timestamp_seconds - left.timestamp_seconds
        if frame_delta > 0 and time_delta > 0:
            rates.append(frame_delta / time_delta)
    if not rates:
        raise ValueError("raw observations need source_fps or timed frame pairs")
    fps = median(rates)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("could not infer a positive source fps")
    return fps


def _source_fps_value(
    source_fps: Sequence[float | None] | Mapping[int, float] | None,
    source_index: int,
) -> float | None:
    if source_fps is None:
        return None
    if isinstance(source_fps, Mapping):
        value = source_fps.get(source_index)
    else:
        value = source_fps[source_index] if source_index < len(source_fps) else None
    if value is None:
        return None
    fps = _finite(value, f"source_fps[{source_index}]")
    if fps <= 0:
        raise ValueError("source fps must be positive")
    return fps


def _prepare_track(
    source_index: int,
    value: TrackingInput,
    *,
    target: IVEMember,
    source_fps: Sequence[float | None] | Mapping[int, float] | None,
) -> _PreparedTrack | None:
    if value is None:
        return None
    if isinstance(value, MemberTrackingResult):
        fps = _finite(value.fps, f"tracking fps for source {source_index}")
        if fps <= 0:
            raise ValueError("tracking fps must be positive")
        converted = value.to_virtual_camera_observations(include_holding=True)
        observations = tuple(
            item for item in converted if isinstance(item, TrackObservation)
        )
    else:
        raw_items = tuple(value)
        if any(
            not isinstance(item, (MemberTrackObservation, TrackObservation))
            for item in raw_items
        ):
            raise TypeError("raw tracking inputs must contain tracking observations")
        converted_items: list[TrackObservation] = []
        for item in raw_items:
            if isinstance(item, TrackObservation):
                converted_items.append(item)
                continue
            if item.bbox_xyxy is None:
                continue
            converted_items.append(
                TrackObservation(
                    frame_index=item.frame_index,
                    timestamp_seconds=item.timestamp_seconds,
                    bbox_xyxy=item.bbox_xyxy,
                    detection_confidence=item.confidence,
                    member=item.member,
                    member_confidence=item.confidence,
                    source_id=item.source_id,
                    track_id=f"anchored:{item.source_id}",
                    identity_id=f"ive:{item.member.name.casefold()}",
                    occluded=item.state is not TrackingState.TRACKED,
                )
            )
        explicit_fps = _source_fps_value(source_fps, source_index)
        observations = tuple(converted_items)
        if not observations:
            # Empty tracking is a normal safety outcome, not malformed input.
            # FPS is immaterial because this source can never become a crop.
            fps = explicit_fps if explicit_fps is not None else 30.0
        else:
            fps = (
                explicit_fps
                if explicit_fps is not None
                else _infer_fps(observations)
            )

    # A result for another member is valid input but cannot be a target crop.
    # Keep it so the confidence gate can reject it without special casing.
    if isinstance(value, MemberTrackingResult) and value.member is not target:
        observations = tuple(observations)

    timestamps: dict[int, float] = {}
    normalized: list[TrackObservation] = []
    for item in observations:
        timestamp = item.timestamp_seconds
        if timestamp is None:
            timestamp = item.frame_index / fps
        timestamp = _finite(timestamp, "observation timestamp")
        if timestamp < 0:
            raise ValueError("observation timestamps cannot be negative")
        previous = timestamps.get(item.frame_index)
        if previous is not None and not math.isclose(previous, timestamp, abs_tol=1e-7):
            raise ValueError("one frame index cannot have conflicting timestamps")
        timestamps[item.frame_index] = timestamp
        if item.timestamp_seconds is None:
            item = TrackObservation(
                frame_index=item.frame_index,
                timestamp_seconds=timestamp,
                bbox_xyxy=item.bbox_xyxy,
                detection_confidence=item.detection_confidence,
                member=item.member,
                member_confidence=item.member_confidence,
                source_id=item.source_id,
                track_id=item.track_id,
                identity_id=item.identity_id,
                occluded=item.occluded,
            )
        normalized.append(item)
    return _PreparedTrack(
        source_index=source_index,
        fps=fps,
        observations=tuple(normalized),
        timestamps_by_frame=timestamps,
    )


def _covered_interval_metrics(
    timestamps: Sequence[float],
    *,
    start: float,
    end: float,
    fps: float,
) -> tuple[float, float]:
    """Return union coverage and largest uncovered gap in ``[start, end]``."""

    radius = 0.51 / fps
    intervals = sorted(
        (max(start, time - radius), min(end, time + radius))
        for time in timestamps
        if start - radius <= time <= end + radius
    )
    intervals = [(left, right) for left, right in intervals if right > left]
    if not intervals:
        return (0.0, end - start)
    cursor = start
    covered = 0.0
    largest_gap = 0.0
    for left, right in intervals:
        if left > cursor:
            largest_gap = max(largest_gap, left - cursor)
        effective_left = max(left, cursor)
        if right > effective_left:
            covered += right - effective_left
            cursor = max(cursor, right)
    largest_gap = max(largest_gap, end - cursor)
    return (min(1.0, covered / (end - start)), largest_gap)


def _stability(observations: Sequence[TrackObservation]) -> float:
    if len(observations) < 2:
        return 1.0
    ordered = sorted(
        observations,
        key=lambda item: (item.timestamp_seconds or 0.0, item.frame_index),
    )
    steps: list[float] = []
    for left, right in zip(ordered, ordered[1:]):
        left_time = left.timestamp_seconds
        right_time = right.timestamp_seconds
        if left_time is None or right_time is None or right_time <= left_time:
            continue
        center_jump = math.hypot(
            right.center[0] - left.center[0], right.center[1] - left.center[1]
        )
        left_area = left.size[0] * left.size[1]
        right_area = right.size[0] * right.size[1]
        scale_jump = abs(math.log(max(right_area, 1e-9) / max(left_area, 1e-9)))
        # Normalize to a nominal 30fps step so sparse detectors are not
        # punished merely because their samples are farther apart in time.
        nominal_frames = max((right_time - left_time) * 30.0, 1.0)
        steps.append((center_jump * 4.0 + scale_jump * 0.35) / nominal_frames)
    motion_stability = math.exp(-median(steps)) if steps else 1.0
    confidences = [item.tracking_confidence for item in ordered]
    confidence_stability = 1.0 - min(1.0, pstdev(confidences) * 3.0)
    return max(0.0, min(1.0, 0.75 * motion_stability + 0.25 * confidence_stability))


def _candidate_for_span(
    track: _PreparedTrack,
    *,
    offset: float,
    span: CanonicalSpan,
    target: IVEMember,
    output_aspect: AspectRatio,
    source_aspect_ratio: float,
    config: MemberDraftConfig,
) -> _Candidate | None:
    local_start = span.start_seconds + offset
    local_end = span.end_seconds + offset
    if local_start < -_EPSILON:
        return None
    radius = 0.51 / track.fps
    in_range = tuple(
        item
        for item in track.observations
        if item.timestamp_seconds is not None
        and local_start - radius <= item.timestamp_seconds <= local_end + radius
    )
    safe = tuple(
        item
        for item in in_range
        if not item.occluded
        and item.member is target
        and item.detection_confidence >= config.minimum_confidence
        and item.member_confidence >= config.minimum_confidence
    )
    coverage, largest_gap = _covered_interval_metrics(
        [float(item.timestamp_seconds) for item in safe if item.timestamp_seconds is not None],
        start=local_start,
        end=local_end,
        fps=track.fps,
    )
    if (
        coverage + _EPSILON < config.minimum_interval_coverage
        or largest_gap > config.maximum_untracked_gap_seconds + _EPSILON
    ):
        return None

    pixel_aspect = 9.0 / 16.0 if output_aspect is AspectRatio.PORTRAIT else 16.0 / 9.0
    camera_config = VirtualCameraConfig(
        output_aspect_ratio=pixel_aspect,
        source_aspect_ratio=source_aspect_ratio,
        min_detection_confidence=config.minimum_confidence,
        min_member_confidence=config.minimum_confidence,
    )
    frame_indices = tuple(
        sorted(
            {
                item.frame_index
                for item in in_range
                if item.timestamp_seconds is not None
                and local_start - radius <= item.timestamp_seconds <= local_end + radius
            }
        )
    )
    if not frame_indices:
        return None
    camera_frames = plan_virtual_camera(
        track.observations,
        target_member=target,
        config=camera_config,
        frame_indices=frame_indices,
    )
    samples: list[SegmentCropSample] = []
    camera_safe_timestamps: list[float] = []
    entry_framing: tuple[float, float, float] | None = None
    exit_framing: tuple[float, float, float] | None = None
    for frame in camera_frames:
        if not frame.safe_for_target_crop or frame.crop is None:
            continue
        assert frame.observation is not None
        timestamp = track.timestamps_by_frame.get(frame.frame_index)
        if timestamp is None:
            continue
        camera_safe_timestamps.append(timestamp)
        subject_center_x, subject_center_y = frame.observation.center
        subject_height = frame.observation.size[1]
        framing = (
            (subject_center_x - frame.crop.left) / frame.crop.width,
            (subject_center_y - frame.crop.top) / frame.crop.height,
            subject_height / frame.crop.height,
        )
        if entry_framing is None:
            entry_framing = framing
        exit_framing = framing
        relative = min(max(timestamp - local_start, 0.0), span.duration_seconds)
        sample = SegmentCropSample(
            time_seconds=relative,
            x=frame.crop.left,
            y=frame.crop.top,
            width=frame.crop.width,
            height=frame.crop.height,
        )
        if samples and math.isclose(
            samples[-1].time_seconds, sample.time_seconds, abs_tol=1e-9
        ):
            samples[-1] = sample
        elif not samples or sample.time_seconds > samples[-1].time_seconds:
            samples.append(sample)
    if not samples or entry_framing is None or exit_framing is None:
        return None
    camera_coverage, camera_largest_gap = _covered_interval_metrics(
        camera_safe_timestamps,
        start=local_start,
        end=local_end,
        fps=track.fps,
    )
    if (
        camera_coverage + _EPSILON < config.minimum_interval_coverage
        or camera_largest_gap > config.maximum_untracked_gap_seconds + _EPSILON
    ):
        # A verified member box is insufficient if the speed-limited virtual
        # camera cannot keep that box inside its crop.  Falling back is safer
        # than rendering a smooth but target-free portrait window.
        return None
    coverage = min(coverage, camera_coverage)
    # Render pan tracks require an explicit state at segment-relative zero.
    if samples[0].time_seconds > _EPSILON:
        first = samples[0]
        samples.insert(
            0,
            SegmentCropSample(0.0, first.x, first.y, first.width, first.height),
        )
    elif samples[0].time_seconds != 0.0:
        first = samples[0]
        samples[0] = SegmentCropSample(
            0.0, first.x, first.y, first.width, first.height
        )

    mean_confidence = sum(item.tracking_confidence for item in safe) / len(safe)
    stability = _stability(safe)
    weight_sum = (
        config.confidence_weight + config.coverage_weight + config.stability_weight
    )
    base_score = (
        config.confidence_weight * mean_confidence
        + config.coverage_weight * coverage
        + config.stability_weight * stability
    ) / weight_sum
    return _Candidate(
        source_index=track.source_index,
        local_start=max(0.0, local_start),
        local_end=local_end,
        coverage=coverage,
        mean_confidence=mean_confidence,
        stability=stability,
        base_score=base_score,
        crop_samples=tuple(samples),
        entry_framing=entry_framing,
        exit_framing=exit_framing,
    )


@dataclass(frozen=True, slots=True)
class _PathValue:
    utility: float
    base_score_sum: float
    continuity_mismatch_sum: float
    adjacent_repeat_count: int
    accumulated_use_penalty_sum: float
    short_recurrence_cost_sum: float
    usage_counts: tuple[int, ...]
    last_positions: tuple[int, ...]
    sequence: tuple[int, ...]
    last_candidate: _Candidate | None


def _transition_mismatch(left: _Candidate | None, right: _Candidate) -> float:
    """Compare target framing in output-crop coordinates across a cut."""

    if left is None:
        return 0.0
    left_x, left_y, left_height = left.exit_framing
    right_x, right_y, right_height = right.entry_framing
    position = math.hypot(right_x - left_x, right_y - left_y)
    scale = abs(
        math.log(max(right_height, 1e-6) / max(left_height, 1e-6))
    )
    return min(3.0, position + 0.35 * scale)


def _extend_path(
    value: _PathValue,
    candidate: _Candidate,
    settings: MemberDraftConfig,
) -> _PathValue:
    mismatch = _transition_mismatch(value.last_candidate, candidate)
    repeated = (
        value.last_candidate is not None
        and value.last_candidate.source_index == candidate.source_index
    )
    prior_use_count = value.usage_counts[candidate.source_index]
    accumulated_penalty = settings.accumulated_use_penalty * prior_use_count
    previous_position = value.last_positions[candidate.source_index]
    short_recurrence_cost = (
        0.0
        if previous_position < 0
        else 1.0 / (len(value.sequence) - previous_position)
    )
    usage_counts = list(value.usage_counts)
    usage_counts[candidate.source_index] += 1
    last_positions = list(value.last_positions)
    last_positions[candidate.source_index] = len(value.sequence)
    return _PathValue(
        utility=(
            value.utility
            + candidate.base_score
            - settings.continuity_weight * mismatch
            - (settings.repeat_source_penalty if repeated else 0.0)
            - accumulated_penalty
        ),
        base_score_sum=value.base_score_sum + candidate.base_score,
        continuity_mismatch_sum=value.continuity_mismatch_sum + mismatch,
        adjacent_repeat_count=value.adjacent_repeat_count + int(repeated),
        accumulated_use_penalty_sum=(
            value.accumulated_use_penalty_sum + accumulated_penalty
        ),
        short_recurrence_cost_sum=(
            value.short_recurrence_cost_sum + short_recurrence_cost
        ),
        usage_counts=tuple(usage_counts),
        last_positions=tuple(last_positions),
        sequence=value.sequence + (candidate.source_index,),
        last_candidate=candidate,
    )


def _append_unavailable(value: _PathValue) -> _PathValue:
    return _PathValue(
        utility=value.utility,
        base_score_sum=value.base_score_sum,
        continuity_mismatch_sum=value.continuity_mismatch_sum,
        adjacent_repeat_count=value.adjacent_repeat_count,
        accumulated_use_penalty_sum=value.accumulated_use_penalty_sum,
        short_recurrence_cost_sum=value.short_recurrence_cost_sum,
        usage_counts=value.usage_counts,
        last_positions=value.last_positions,
        sequence=value.sequence + (-1,),
        last_candidate=None,
    )


def _better_path(candidate: _PathValue, incumbent: _PathValue | None) -> bool:
    if incumbent is None:
        return True
    numeric_preferences = (
        (candidate.utility, incumbent.utility, True),
        (candidate.base_score_sum, incumbent.base_score_sum, True),
        (
            candidate.continuity_mismatch_sum,
            incumbent.continuity_mismatch_sum,
            False,
        ),
        (
            float(candidate.adjacent_repeat_count),
            float(incumbent.adjacent_repeat_count),
            False,
        ),
        (
            candidate.accumulated_use_penalty_sum,
            incumbent.accumulated_use_penalty_sum,
            False,
        ),
        (
            candidate.short_recurrence_cost_sum,
            incumbent.short_recurrence_cost_sum,
            False,
        ),
    )
    for left, right, larger_is_better in numeric_preferences:
        if math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
            continue
        return left > right if larger_is_better else left < right
    return candidate.sequence < incumbent.sequence


def _maximum_source_segment_matching(
    rows: Sequence[Sequence[_Candidate]],
) -> dict[int, int]:
    """Return a deterministic maximum-cardinality source/segment matching."""

    edges_by_source: dict[int, list[tuple[int, float]]] = {}
    for segment_index, row in enumerate(rows):
        for candidate in row:
            edges_by_source.setdefault(candidate.source_index, []).append(
                (segment_index, candidate.base_score)
            )
    for edges in edges_by_source.values():
        edges.sort(key=lambda item: (-item[1], item[0]))

    segment_to_source: dict[int, int] = {}

    def augment(source_index: int, visited_segments: set[int]) -> bool:
        for segment_index, _ in edges_by_source[source_index]:
            if segment_index in visited_segments:
                continue
            visited_segments.add(segment_index)
            incumbent = segment_to_source.get(segment_index)
            if incumbent is None or augment(incumbent, visited_segments):
                segment_to_source[segment_index] = source_index
                return True
        return False

    for source_index in sorted(
        edges_by_source,
        key=lambda index: (len(edges_by_source[index]), index),
    ):
        augment(source_index, set())
    return segment_to_source


def _exact_diversity_path(
    rows: Sequence[Sequence[_Candidate]],
    *,
    quality_source_indices: Sequence[int],
    settings: MemberDraftConfig,
) -> tuple[_Candidate | None, ...]:
    source_bits = {
        source_index: bit_index
        for bit_index, source_index in enumerate(quality_source_indices)
    }
    source_count = max(quality_source_indices, default=-1) + 1
    initial = _PathValue(
        0.0,
        0.0,
        0.0,
        0,
        0.0,
        0.0,
        (0,) * source_count,
        (-1,) * source_count,
        (),
        None,
    )
    states: dict[
        tuple[int, int, tuple[int, ...], tuple[int, ...]],
        _PathValue,
    ] = {
        (0, -1, initial.usage_counts, initial.last_positions): initial
    }
    for row in rows:
        next_states: dict[
            tuple[int, int, tuple[int, ...], tuple[int, ...]],
            _PathValue,
        ] = {}
        if not row:
            for (mask, _, _, _), value in states.items():
                extended = _append_unavailable(value)
                key = (
                    mask,
                    -1,
                    extended.usage_counts,
                    extended.last_positions,
                )
                if _better_path(extended, next_states.get(key)):
                    next_states[key] = extended
        else:
            for (mask, _, _, _), value in states.items():
                for candidate in row:
                    bit = 1 << source_bits[candidate.source_index]
                    extended = _extend_path(value, candidate, settings)
                    key = (
                        mask | bit,
                        candidate.source_index,
                        extended.usage_counts,
                        extended.last_positions,
                    )
                    if _better_path(extended, next_states.get(key)):
                        next_states[key] = extended
        states = next_states

    best_mask = 0
    best: _PathValue | None = None
    for (mask, _, _, _), value in states.items():
        if mask.bit_count() > best_mask.bit_count() or (
            mask.bit_count() == best_mask.bit_count()
            and _better_path(value, best)
        ):
            best_mask = mask
            best = value
    assert best is not None
    row_maps = [
        {candidate.source_index: candidate for candidate in row}
        for row in rows
    ]
    return tuple(
        None if source_index < 0 else row_maps[index][source_index]
        for index, source_index in enumerate(best.sequence)
    )


def _beam_sort_key(value: _PathValue) -> tuple[object, ...]:
    return (
        -round(value.utility, 12),
        -round(value.base_score_sum, 12),
        round(value.continuity_mismatch_sum, 12),
        value.adjacent_repeat_count,
        round(value.accumulated_use_penalty_sum, 12),
        round(value.short_recurrence_cost_sum, 12),
        value.sequence,
    )


def _matching_anchored_balanced_beam_path(
    rows: Sequence[Sequence[_Candidate]],
    *,
    matching: Mapping[int, int],
    settings: MemberDraftConfig,
) -> tuple[_Candidate | None, ...]:
    """Bounded deterministic beam preserving maximum matching diversity."""

    source_count = max(
        (
            candidate.source_index
            for row in rows
            for candidate in row
        ),
        default=-1,
    ) + 1
    initial = _PathValue(
        0.0,
        0.0,
        0.0,
        0,
        0.0,
        0.0,
        (0,) * source_count,
        (-1,) * source_count,
        (),
        None,
    )
    states: dict[
        tuple[int, tuple[int, ...], tuple[int, ...]],
        _PathValue,
    ] = {
        (-1, initial.usage_counts, initial.last_positions): initial
    }
    for segment_index, row in enumerate(rows):
        anchor_source = matching.get(segment_index)
        choices = tuple(
            candidate
            for candidate in row
            if anchor_source is None or candidate.source_index == anchor_source
        )
        next_states: dict[
            tuple[int, tuple[int, ...], tuple[int, ...]],
            _PathValue,
        ] = {}
        if not choices:
            for value in states.values():
                extended = _append_unavailable(value)
                key = (
                    -1,
                    extended.usage_counts,
                    extended.last_positions,
                )
                if _better_path(extended, next_states.get(key)):
                    next_states[key] = extended
        else:
            for value in states.values():
                for candidate in choices:
                    extended = _extend_path(value, candidate, settings)
                    key = (
                        candidate.source_index,
                        extended.usage_counts,
                        extended.last_positions,
                    )
                    if _better_path(extended, next_states.get(key)):
                        next_states[key] = extended
        if len(next_states) > _BALANCED_PATH_BEAM_WIDTH:
            retained = sorted(next_states.values(), key=_beam_sort_key)[
                :_BALANCED_PATH_BEAM_WIDTH
            ]
            states = {
                (
                    (
                        value.last_candidate.source_index
                        if value.last_candidate is not None
                        else -1
                    ),
                    value.usage_counts,
                    value.last_positions,
                ): value
                for value in retained
            }
        else:
            states = next_states

    best: _PathValue | None = None
    for value in states.values():
        if _better_path(value, best):
            best = value
    assert best is not None
    row_maps = [
        {candidate.source_index: candidate for candidate in row}
        for row in rows
    ]
    return tuple(
        None if source_index < 0 else row_maps[index][source_index]
        for index, source_index in enumerate(best.sequence)
    )
def _select_global_candidates(
    safe_rows: Sequence[Sequence[_Candidate]],
    *,
    source_count: int,
    settings: MemberDraftConfig,
) -> tuple[
    tuple[_Candidate | None, ...],
    MemberDraftSelectionDiagnostics,
]:
    quality_rows: list[tuple[_Candidate, ...]] = []
    for row in safe_rows:
        if not row:
            quality_rows.append(())
            continue
        best_score = max(candidate.base_score for candidate in row)
        quality_rows.append(
            tuple(
                candidate
                for candidate in row
                if best_score - candidate.base_score
                <= settings.maximum_quality_score_drop + _EPSILON
            )
        )

    safe_counts_by_source = [0] * source_count
    quality_counts_by_source = [0] * source_count
    for row in safe_rows:
        for candidate in row:
            safe_counts_by_source[candidate.source_index] += 1
    for row in quality_rows:
        for candidate in row:
            quality_counts_by_source[candidate.source_index] += 1
    quality_sources = tuple(
        index for index, count in enumerate(quality_counts_by_source) if count > 0
    )
    matching = _maximum_source_segment_matching(quality_rows)
    projected_exact_states = 1
    for row in quality_rows:
        projected_exact_states *= max(1, len(row))
        if projected_exact_states > _EXACT_SEQUENCE_STATE_BUDGET:
            break
    if (
        len(quality_sources) <= settings.exact_diversity_source_limit
        and projected_exact_states <= _EXACT_SEQUENCE_STATE_BUDGET
    ):
        selected = _exact_diversity_path(
            quality_rows,
            quality_source_indices=quality_sources,
            settings=settings,
        )
        optimizer = "exact_distinct_mask_dp"
    else:
        selected = _matching_anchored_balanced_beam_path(
            quality_rows,
            matching=matching,
            settings=settings,
        )
        optimizer = "maximum_matching_anchored_balanced_beam"

    selected_counts = [0] * source_count
    for candidate in selected:
        if candidate is not None:
            selected_counts[candidate.source_index] += 1
    diagnostics = MemberDraftSelectionDiagnostics(
        policy=(
            "safe_gate_then_local_best_quality_guard_then_"
            "maximum_distinct_balanced_global_path_v2"
        ),
        optimizer=optimizer,
        quality_score_drop_limit=settings.maximum_quality_score_drop,
        maximum_distinct_quality_source_count=len(set(matching.values())),
        safe_candidate_counts_by_segment=tuple(len(row) for row in safe_rows),
        quality_candidate_counts_by_segment=tuple(
            len(row) for row in quality_rows
        ),
        safe_candidate_counts_by_source=tuple(safe_counts_by_source),
        quality_candidate_counts_by_source=tuple(quality_counts_by_source),
        selected_segment_counts_by_source=tuple(selected_counts),
    )
    if (
        sum(count > 0 for count in selected_counts)
        != diagnostics.maximum_distinct_quality_source_count
    ):
        raise RuntimeError("member diversity optimizer missed maximum matching")
    return (selected, diagnostics)


def build_member_draft(
    *,
    canonical_spans: Sequence[object],
    audio_offsets: Sequence[float],
    tracking_by_source: Sequence[TrackingInput] | Mapping[int, TrackingInput],
    target_member: IVEMember | str,
    output_aspect: AspectRatio | str,
    group_wide_fallback_source: int | None = None,
    source_aspect_ratio: float = 16.0 / 9.0,
    source_fps: Sequence[float | None] | Mapping[int, float] | None = None,
    config: MemberDraftConfig | None = None,
) -> MemberDraftTimeline:
    """Build a deterministic, confidence-gated target-member edit draft.

    ``tracking_by_source`` and ``audio_offsets`` share the same source indices.
    A raw observation sequence may contain either anchored
    :class:`MemberTrackObservation` values or identity-resolved
    :class:`~crosscut.vision.TrackObservation` values.  Raw observations need
    timestamps from which FPS can be inferred, or an explicit ``source_fps``.
    """

    spans = tuple(_coerce_span(value) for value in canonical_spans)
    if not spans:
        raise ValueError("canonical_spans cannot be empty")
    for left, right in zip(spans, spans[1:]):
        if not math.isclose(left.end_seconds, right.start_seconds, abs_tol=1e-7):
            raise ValueError("canonical_spans must be ordered and contiguous")
    offsets = tuple(
        _finite(value, f"audio_offsets[{index}]")
        for index, value in enumerate(audio_offsets)
    )
    if not offsets:
        raise ValueError("audio_offsets cannot be empty")
    target = IVEMember.parse(target_member)
    aspect = (
        output_aspect
        if isinstance(output_aspect, AspectRatio)
        else AspectRatio(output_aspect)
    )
    source_aspect = _finite(source_aspect_ratio, "source_aspect_ratio")
    if source_aspect <= 0:
        raise ValueError("source_aspect_ratio must be positive")
    settings = config or MemberDraftConfig()

    if group_wide_fallback_source is not None:
        if (
            isinstance(group_wide_fallback_source, bool)
            or not isinstance(group_wide_fallback_source, int)
            or not 0 <= group_wide_fallback_source < len(offsets)
        ):
            raise ValueError("group_wide_fallback_source is out of range")

    if isinstance(tracking_by_source, Mapping):
        unknown = set(tracking_by_source) - set(range(len(offsets)))
        if unknown:
            raise ValueError("tracking_by_source contains an unknown source index")
        raw_tracks = [tracking_by_source.get(index) for index in range(len(offsets))]
    else:
        if len(tracking_by_source) != len(offsets):
            raise ValueError("tracking_by_source length must match audio_offsets")
        raw_tracks = list(tracking_by_source)

    prepared = tuple(
        track
        for index, value in enumerate(raw_tracks)
        if (
            track := _prepare_track(
                index,
                value,
                target=target,
                source_fps=source_fps,
            )
        )
        is not None
    )

    safe_rows = tuple(
        tuple(
            candidate
            for track in prepared
            if (
                candidate := _candidate_for_span(
                    track,
                    offset=offsets[track.source_index],
                    span=span,
                    target=target,
                    output_aspect=aspect,
                    source_aspect_ratio=source_aspect,
                    config=settings,
                )
            )
            is not None
        )
        for span in spans
    )
    selected_candidates, selection_diagnostics = _select_global_candidates(
        safe_rows,
        source_count=len(offsets),
        settings=settings,
    )

    use_counts = [0] * len(offsets)
    previous_candidate: _Candidate | None = None
    segments: list[MemberDraftSegment] = []
    common_start = spans[0].start_seconds
    for segment_index, (span, chosen) in enumerate(
        zip(spans, selected_candidates, strict=True)
    ):
        if chosen is not None:
            source_index = chosen.source_index
            continuity_mismatch = _transition_mismatch(
                previous_candidate, chosen
            )
            repeated = (
                previous_candidate is not None
                and previous_candidate.source_index == source_index
            )
            effective_score = (
                chosen.base_score
                - settings.continuity_weight * continuity_mismatch
                - (settings.repeat_source_penalty if repeated else 0.0)
                - settings.accumulated_use_penalty * use_counts[source_index]
            )
            segment = MemberDraftSegment(
                segment_index=segment_index,
                canonical_master_start=span.start_seconds,
                canonical_master_end=span.end_seconds,
                output_start_seconds=span.start_seconds - common_start,
                kind=MemberSegmentKind.TARGET_CROP,
                source_index=source_index,
                source_local_start=chosen.local_start,
                source_local_end=chosen.local_end,
                audio_offset_seconds=offsets[source_index],
                crop_samples=chosen.crop_samples,
                interval_coverage=chosen.coverage,
                mean_confidence=chosen.mean_confidence,
                stability=chosen.stability,
                selection_score=effective_score,
            )
            use_counts[source_index] += 1
            previous_candidate = chosen
        elif (
            group_wide_fallback_source is not None
            and span.start_seconds + offsets[group_wide_fallback_source] >= -_EPSILON
        ):
            source_index = group_wide_fallback_source
            segment = MemberDraftSegment(
                segment_index=segment_index,
                canonical_master_start=span.start_seconds,
                canonical_master_end=span.end_seconds,
                output_start_seconds=span.start_seconds - common_start,
                kind=MemberSegmentKind.GROUP_WIDE_FALLBACK,
                source_index=source_index,
                source_local_start=max(0.0, span.start_seconds + offsets[source_index]),
                source_local_end=span.end_seconds + offsets[source_index],
                audio_offset_seconds=offsets[source_index],
                crop_samples=(),
                interval_coverage=0.0,
                mean_confidence=0.0,
                stability=0.0,
                selection_score=None,
            )
            previous_candidate = None
        else:
            segment = MemberDraftSegment(
                segment_index=segment_index,
                canonical_master_start=span.start_seconds,
                canonical_master_end=span.end_seconds,
                output_start_seconds=span.start_seconds - common_start,
                kind=MemberSegmentKind.UNAVAILABLE,
                source_index=None,
                source_local_start=None,
                source_local_end=None,
                audio_offset_seconds=None,
                crop_samples=(),
                interval_coverage=0.0,
                mean_confidence=0.0,
                stability=0.0,
                selection_score=None,
            )
            previous_candidate = None
        segments.append(segment)

    return MemberDraftTimeline(
        target_member=target,
        output_aspect=aspect,
        source_aspect_ratio=source_aspect,
        common_master_start=spans[0].start_seconds,
        common_master_end=spans[-1].end_seconds,
        audio_offsets=offsets,
        group_wide_fallback_source=group_wide_fallback_source,
        segments=tuple(segments),
        selection_diagnostics=selection_diagnostics,
    )


__all__ = [
    "CanonicalSpan",
    "MemberDraftConfig",
    "MemberDraftSegment",
    "MemberDraftSelectionDiagnostics",
    "MemberDraftTimeline",
    "MemberSegmentKind",
    "SegmentCropSample",
    "build_member_draft",
]
