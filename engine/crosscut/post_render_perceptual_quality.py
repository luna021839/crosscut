"""Scalar-only perceptual audit of a finished production render.

The proxy reviewer scores a deliberately small review render.  This module
instead decodes the final file from beginning to end and, where the render plan
provides an unambiguous identity view, compares selected output frames with the
exact source-local frame requested by that plan.  Raw frames and per-frame
descriptors remain transient and are never included in the returned payload.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np


SCHEMA_VERSION = 2
ANALYSIS_BASIS = (
    "full_final_decode_temporal_pumping_exact_source_retention_and_"
    "structure_binding_v2"
)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _mapping(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _sequence(value: Any) -> list[Any] | None:
    return list(value) if isinstance(value, list) else None


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = fraction * (len(clean) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    alpha = position - low
    return clean[low] * (1.0 - alpha) + clean[high] * alpha


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    clean = tuple(float(value) for value in values if math.isfinite(float(value)))
    return {
        "count": len(clean),
        "minimum": min(clean) if clean else None,
        "p10": _quantile(clean, 0.10),
        "median": _quantile(clean, 0.50),
        "p90": _quantile(clean, 0.90),
        "p99": _quantile(clean, 0.99),
        "maximum": max(clean) if clean else None,
    }


@dataclass(frozen=True, slots=True)
class PerceptualQualityConfig:
    analysis_width: int = 256
    maximum_structure_delta_for_temporal_comparison: float = 0.18
    minimum_same_run_comparable_pair_count: int = 24
    minimum_per_source_output_frame_count: int = 12
    retention_samples_per_second: float = 1.0
    maximum_retention_samples_per_source: int = 24
    minimum_retention_samples_per_source: int = 3
    retention_segment_guard_frames: int = 3
    minimum_reference_detail: float = 0.004
    temporal_sharpness_log_step: float = 0.45
    temporal_luma_step: float = 0.065
    temporal_chroma_step: float = 0.055
    temporal_high_step_fraction_conditional: float = 0.03
    temporal_high_step_fraction_fail: float = 0.08
    temporal_reversal_fraction_conditional: float = 0.015
    temporal_reversal_fraction_fail: float = 0.05
    source_sharpness_log_span_conditional: float = 0.55
    source_sharpness_log_span_fail: float = 0.90
    source_luma_median_span_conditional: float = 0.14
    source_luma_median_span_fail: float = 0.24
    source_chroma_median_span_conditional: float = 0.12
    source_chroma_median_span_fail: float = 0.20
    rapid_run_maximum_seconds: float = 3.0
    minimum_rapid_boundary_count: int = 4
    rapid_high_shock_fraction_conditional: float = 0.25
    rapid_high_shock_fraction_fail: float = 0.50
    retention_ratio_p10_conditional: float = 0.72
    retention_ratio_p10_fail: float = 0.55
    retention_ratio_median_conditional: float = 0.85
    retention_ratio_median_fail: float = 0.70
    exact_structure_similarity_p10_conditional: float = 0.50
    exact_structure_similarity_p10_fail: float = 0.30
    exact_structure_similarity_median_conditional: float = 0.55
    exact_structure_similarity_median_fail: float = 0.40

    def __post_init__(self) -> None:
        if self.analysis_width < 64 or self.analysis_width > 1024:
            raise ValueError("analysis_width must be in [64, 1024]")
        integer_fields = (
            "minimum_same_run_comparable_pair_count",
            "minimum_per_source_output_frame_count",
            "maximum_retention_samples_per_source",
            "minimum_retention_samples_per_source",
            "retention_segment_guard_frames",
            "minimum_rapid_boundary_count",
        )
        if any(getattr(self, name) < 0 for name in integer_fields):
            raise ValueError("perceptual quality counts must be non-negative")
        if (
            self.maximum_retention_samples_per_source
            < self.minimum_retention_samples_per_source
        ):
            raise ValueError("retention sample limits are inverted")
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, float) and (
                not math.isfinite(value) or value < 0.0
            ):
                raise ValueError("perceptual quality bounds must be finite")
        ordered_pairs = (
            (
                self.temporal_high_step_fraction_conditional,
                self.temporal_high_step_fraction_fail,
            ),
            (
                self.temporal_reversal_fraction_conditional,
                self.temporal_reversal_fraction_fail,
            ),
            (
                self.source_sharpness_log_span_conditional,
                self.source_sharpness_log_span_fail,
            ),
            (
                self.source_luma_median_span_conditional,
                self.source_luma_median_span_fail,
            ),
            (
                self.source_chroma_median_span_conditional,
                self.source_chroma_median_span_fail,
            ),
            (
                self.rapid_high_shock_fraction_conditional,
                self.rapid_high_shock_fraction_fail,
            ),
        )
        if any(left > right for left, right in ordered_pairs):
            raise ValueError("conditional bounds cannot exceed fail bounds")
        if not (
            self.retention_ratio_p10_fail
            <= self.retention_ratio_p10_conditional
            and self.retention_ratio_median_fail
            <= self.retention_ratio_median_conditional
        ):
            raise ValueError("retention fail bounds cannot exceed warning bounds")
        if not (
            0.0
            <= self.exact_structure_similarity_p10_fail
            <= self.exact_structure_similarity_p10_conditional
            <= 1.0
            and 0.0
            <= self.exact_structure_similarity_median_fail
            <= self.exact_structure_similarity_median_conditional
            <= 1.0
        ):
            raise ValueError(
                "exact structure similarity bounds are invalid"
            )

    def to_dict(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class _Span:
    segment_index: int
    source_index: int
    start_frame: int
    end_frame: int
    run_index: int
    source_start: float
    source_rate: float
    duration: float
    incoming_transition_frames: int
    crop: Mapping[str, Any] | None
    pan_crop: Mapping[str, Any] | None
    editorial_transform: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class _FrameScalar:
    detail: float
    luma: float
    chroma_a: float
    chroma_b: float
    contrast: float


class _Issues:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self._keys: set[str] = set()

    def add(
        self,
        code: str,
        severity: str,
        message: str,
        *,
        path: str,
        value: Any = None,
        threshold: Any = None,
    ) -> None:
        if code in self._keys:
            return
        self._keys.add(code)
        self.items.append(
            {
                "code": code,
                "severity": severity,
                "message": message,
                "path": path,
                "value": value,
                "threshold": threshold,
            }
        )

    @property
    def verdict(self) -> str:
        if any(item["severity"] == "fail" for item in self.items):
            return "FAIL"
        if self.items:
            return "CONDITIONAL"
        return "PASS"


def _identity_adjustment(value: Any) -> bool:
    payload = _mapping(value)
    if payload is None:
        return value is None
    expected = {
        "brightness": 0.0,
        "contrast": 1.0,
        "saturation": 1.0,
        "temperature_balance": 0.0,
        "tint_balance": 0.0,
    }
    return all(
        (numeric := _finite(payload.get(name, target))) is not None
        and math.isclose(numeric, target, rel_tol=0.0, abs_tol=1e-9)
        for name, target in expected.items()
    )


def _identity_boundary_track(value: Any) -> bool:
    payload = _mapping(value)
    if payload is None:
        return value is None
    keyframes = _sequence(payload.get("keyframes"))
    if not keyframes:
        return False
    return all(_identity_adjustment(raw) for raw in keyframes)


def _parse_plan(
    plan: Mapping[str, Any],
) -> tuple[float, int, int, int, tuple[Path, ...], tuple[_Span, ...]]:
    fps = _finite(plan.get("fps"))
    width = _integer(plan.get("width"))
    height = _integer(plan.get("height"))
    output_frames = _integer(plan.get("output_frame_count"))
    sources_raw = _sequence(plan.get("sources"))
    segments_raw = _sequence(plan.get("segments"))
    transitions = _sequence(plan.get("transitions"))
    if not (
        fps is not None
        and fps > 0.0
        and width is not None
        and width > 0
        and height is not None
        and height > 0
        and output_frames is not None
        and output_frames > 0
        and sources_raw
        and segments_raw
        and transitions is not None
        and len(transitions) == len(segments_raw) - 1
        and all(isinstance(value, str) and value for value in sources_raw)
    ):
        raise ValueError("render plan lacks a complete final-frame contract")
    sources = tuple(Path(value) for value in sources_raw)
    spans: list[_Span] = []
    cumulative_time = 0.0
    previous_boundary = 0
    previous_source: int | None = None
    previous_source_end: float | None = None
    run_index = -1
    for segment_index, raw in enumerate(segments_raw):
        segment = _mapping(raw)
        if segment is None:
            raise ValueError("render plan segment is not an object")
        source_index = _integer(segment.get("source_index"))
        duration = _finite(segment.get("duration"))
        source_start = _finite(segment.get("source_local_start"))
        if source_start is None:
            source_start = _finite(segment.get("source_start"))
        source_end = _finite(segment.get("source_local_end"))
        source_rate = _finite(segment.get("source_playback_rate"))
        if (
            source_rate is None
            and duration is not None
            and source_end is not None
            and source_start is not None
        ):
            source_rate = (source_end - source_start) / duration
        if not (
            source_index is not None
            and 0 <= source_index < len(sources)
            and duration is not None
            and duration > 0.0
            and source_start is not None
            and source_start >= 0.0
            and source_rate is not None
            and source_rate > 0.0
        ):
            raise ValueError("render plan segment timing is incomplete")
        if source_end is None:
            source_end = source_start + duration * source_rate
        cumulative_time += duration
        boundary = int(round(cumulative_time * fps))
        if boundary - previous_boundary < 2:
            raise ValueError("render plan contains an undersized segment")
        contiguous_clock = (
            source_index == previous_source
            and previous_source_end is not None
            and abs(source_start - previous_source_end) <= 1e-4
        )
        if not contiguous_clock:
            run_index += 1
        crop = _mapping(segment.get("crop"))
        pan_crop = _mapping(segment.get("pan_crop"))
        editorial_transform = _mapping(segment.get("editorial_transform"))
        if segment.get("crop") is not None and crop is None:
            raise ValueError("render plan crop is malformed")
        if segment.get("pan_crop") is not None and pan_crop is None:
            raise ValueError("render plan pan crop is malformed")
        if (
            segment.get("editorial_transform") is not None
            and editorial_transform is None
        ):
            raise ValueError("render plan editorial transform is malformed")
        incoming_transition_frames = 0
        if segment_index:
            incoming_transition = _mapping(transitions[segment_index - 1])
            if incoming_transition is None:
                raise ValueError("render plan transition is not an object")
            if incoming_transition.get("kind") == "face_dissolve":
                incoming_transition_frames = (
                    _integer(incoming_transition.get("frame_count")) or 0
                )
                if incoming_transition_frames <= 0:
                    raise ValueError("face dissolve needs a positive frame count")
        spans.append(
            _Span(
                segment_index=segment_index,
                source_index=source_index,
                start_frame=previous_boundary,
                end_frame=boundary,
                run_index=run_index,
                source_start=source_start,
                source_rate=source_rate,
                duration=duration,
                incoming_transition_frames=incoming_transition_frames,
                crop=crop,
                pan_crop=pan_crop,
                editorial_transform=editorial_transform,
            )
        )
        previous_boundary = boundary
        previous_source = source_index
        previous_source_end = source_end
    if previous_boundary != output_frames:
        raise ValueError("segment frame counts do not match output_frame_count")
    return (fps, width, height, output_frames, sources, tuple(spans))


def _analysis_size(width: int, height: int, analysis_width: int) -> tuple[int, int]:
    scaled_height = max(2, int(round(analysis_width * height / width)))
    return (analysis_width, scaled_height)


def _frame_scalar(
    frame: np.ndarray,
    size: tuple[int, int],
) -> tuple[_FrameScalar, np.ndarray]:
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    detail = float(np.mean(np.abs(laplacian)) / 255.0)
    luma = float(np.mean(gray) / 255.0)
    p05, p95 = np.percentile(gray, (5.0, 95.0))
    contrast = float((p95 - p05) / 255.0)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma_a = float((np.mean(lab[..., 1]) - 128.0) / 127.0)
    chroma_b = float((np.mean(lab[..., 2]) - 128.0) / 127.0)
    compact = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
    compact = (compact - float(np.mean(compact))) / max(
        8.0, float(np.std(compact))
    )
    compact = np.clip(compact, -3.0, 3.0)
    return (
        _FrameScalar(detail, luma, chroma_a, chroma_b, contrast),
        compact,
    )


def _structure_delta(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left - right)) / 6.0)


def _structure_similarity(
    left: np.ndarray,
    right: np.ndarray,
) -> float | None:
    denominator = float(
        np.sqrt(np.sum(left * left) * np.sum(right * right))
    )
    if not math.isfinite(denominator) or denominator <= 1e-9:
        return None
    value = float(np.sum(left * right) / denominator)
    return min(1.0, max(-1.0, value)) if math.isfinite(value) else None


def _geometry_structure(frame: np.ndarray) -> np.ndarray:
    """Return a color-grade-resistant spatial edge descriptor."""

    compact = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
    lightness = cv2.cvtColor(compact, cv2.COLOR_BGR2LAB)[..., 0].astype(
        np.float32
    )
    horizontal = cv2.Sobel(lightness, cv2.CV_32F, 1, 0, ksize=3)
    vertical = cv2.Sobel(lightness, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(horizontal * horizontal + vertical * vertical)
    magnitude = np.log1p(magnitude)
    magnitude = (magnitude - float(np.mean(magnitude))) / max(
        0.10,
        float(np.std(magnitude)),
    )
    return np.clip(magnitude, -3.0, 3.0)


def _sharpness_log_delta(left: _FrameScalar, right: _FrameScalar) -> float:
    epsilon = 1e-4
    return math.log((right.detail + epsilon) / (left.detail + epsilon))


def _color_delta(left: _FrameScalar, right: _FrameScalar) -> float:
    return math.hypot(
        right.chroma_a - left.chroma_a,
        right.chroma_b - left.chroma_b,
    )


def _fit_with_pad(
    frame: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    target_width, target_height = size
    height, width = frame.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x = (target_width - resized_width) // 2
    y = (target_height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def _normalized_crop_values(value: Mapping[str, Any]) -> tuple[float, float, float, float]:
    x = _finite(value.get("x"))
    y = _finite(value.get("y"))
    width = _finite(value.get("width"))
    height = _finite(value.get("height"))
    if not (
        x is not None
        and y is not None
        and width is not None
        and height is not None
        and 0.0 <= x <= 1.0
        and 0.0 <= y <= 1.0
        and width > 0.0
        and height > 0.0
        and x + width <= 1.000001
        and y + height <= 1.000001
    ):
        raise ValueError("normalized crop is outside source bounds")
    return (x, y, width, height)


def _interpolate(
    samples: Sequence[tuple[float, ...]],
    time_seconds: float,
) -> tuple[float, ...]:
    if not samples:
        raise ValueError("track has no keyframes")
    if time_seconds <= samples[0][0] + 1e-9:
        return samples[0]
    for left, right in zip(samples, samples[1:]):
        if time_seconds <= right[0] + 1e-9:
            span = right[0] - left[0]
            if span <= 0.0:
                raise ValueError("track times are not strictly increasing")
            alpha = (time_seconds - left[0]) / span
            return (
                time_seconds,
                *(
                    left[index] + (right[index] - left[index]) * alpha
                    for index in range(1, len(left))
                ),
            )
    return samples[-1]


def _pan_crop_values(
    payload: Mapping[str, Any],
    time_seconds: float,
) -> tuple[float, float, float, float]:
    keyframes = _sequence(payload.get("keyframes"))
    if not keyframes:
        raise ValueError("pan crop has no keyframes")
    parsed: list[tuple[float, float, float, float, float]] = []
    ratios: list[float] = []
    maximum_height = 0.0
    for raw in keyframes:
        item = _mapping(raw)
        crop = _mapping((item or {}).get("crop"))
        time_value = _finite((item or {}).get("time_seconds"))
        if crop is None or time_value is None or time_value < 0.0:
            raise ValueError("pan crop keyframe is malformed")
        x, y, width, height = _normalized_crop_values(crop)
        parsed.append((time_value, x, y, width, height))
        ratios.append(width / height)
        maximum_height = max(maximum_height, height)
    if any(
        right[0] <= left[0]
        for left, right in zip(parsed, parsed[1:])
    ):
        raise ValueError("pan crop times are not strictly increasing")
    fixed_width = min(1.0, ratios[0] * maximum_height)
    fixed_height = min(1.0, maximum_height)
    positions = []
    for time_value, x, y, width, height in parsed:
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        left = min(max(center_x - fixed_width / 2.0, 0.0), 1.0 - fixed_width)
        top = min(max(center_y - fixed_height / 2.0, 0.0), 1.0 - fixed_height)
        positions.append((time_value, left, top))
    position = _interpolate(positions, time_seconds)
    return (position[1], position[2], fixed_width, fixed_height)


def _crop_frame(
    frame: np.ndarray,
    values: tuple[float, float, float, float],
) -> np.ndarray:
    x, y, width, height = values
    frame_height, frame_width = frame.shape[:2]
    left = max(0, min(frame_width - 1, int(round(x * frame_width))))
    top = max(0, min(frame_height - 1, int(round(y * frame_height))))
    right = max(left + 1, min(frame_width, int(round((x + width) * frame_width))))
    bottom = max(top + 1, min(frame_height, int(round((y + height) * frame_height))))
    return frame[top:bottom, left:right]


def _fit_fill_native(frame: np.ndarray, target_aspect: float) -> np.ndarray:
    height, width = frame.shape[:2]
    source_aspect = width / height
    if source_aspect > target_aspect:
        crop_width = max(1, int(round(height * target_aspect)))
        left = (width - crop_width) // 2
        return frame[:, left : left + crop_width]
    crop_height = max(1, int(round(width / target_aspect)))
    top = (height - crop_height) // 2
    return frame[top : top + crop_height, :]


def _editorial_transform_values(
    payload: Mapping[str, Any],
    time_seconds: float,
) -> tuple[float, float, float, float]:
    keyframes = _sequence(payload.get("keyframes"))
    if not keyframes:
        raise ValueError("editorial transform has no keyframes")
    parsed: list[tuple[float, float, float, float, float]] = []
    for raw in keyframes:
        item = _mapping(raw)
        if item is None:
            raise ValueError("editorial transform keyframe is malformed")
        values = tuple(
            _finite(item.get(name))
            for name in (
                "time_seconds",
                "zoom",
                "center_x",
                "center_y",
                "roll_degrees",
            )
        )
        if any(value is None for value in values):
            raise ValueError("editorial transform keyframe is incomplete")
        parsed.append(tuple(float(value) for value in values))
    if any(
        right[0] <= left[0]
        for left, right in zip(parsed, parsed[1:])
    ):
        raise ValueError("editorial transform times are not strictly increasing")
    selected = _interpolate(parsed, time_seconds)
    zoom, center_x, center_y, roll = selected[1:]
    if not (
        zoom >= 1.0
        and 0.0 <= center_x <= 1.0
        and 0.0 <= center_y <= 1.0
        and math.isfinite(roll)
    ):
        raise ValueError("editorial transform is outside renderer bounds")
    return (zoom, center_x, center_y, roll)


def _reference_view_for_span(
    frame: np.ndarray,
    span: _Span,
    *,
    relative_time: float,
    size: tuple[int, int],
) -> np.ndarray:
    working = frame
    if span.crop is not None:
        working = _crop_frame(working, _normalized_crop_values(span.crop))
    elif span.pan_crop is not None:
        working = _crop_frame(
            working,
            _pan_crop_values(span.pan_crop, relative_time),
        )
    if span.editorial_transform is None:
        return _fit_with_pad(working, size)

    working = _fit_fill_native(working, size[0] / size[1])
    zoom, center_x, center_y, roll = _editorial_transform_values(
        span.editorial_transform,
        relative_time,
    )
    if abs(roll) > 1e-9:
        height, width = working.shape[:2]
        rotation = cv2.getRotationMatrix2D(
            (width / 2.0, height / 2.0),
            roll,
            1.0,
        )
        working = cv2.warpAffine(
            working,
            rotation,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
    height, width = working.shape[:2]
    crop_width = max(1, int(round(width / zoom)))
    crop_height = max(1, int(round(height / zoom)))
    left = int(round(width * center_x - crop_width / 2.0))
    top = int(round(height * center_y - crop_height / 2.0))
    left = min(max(0, left), width - crop_width)
    top = min(max(0, top), height - crop_height)
    working = working[top : top + crop_height, left : left + crop_width]
    return cv2.resize(working, size, interpolation=cv2.INTER_AREA)


def _contrast_normalized_detail(value: _FrameScalar) -> float:
    return value.detail / max(0.08, value.contrast)


def _evenly_limited(values: Sequence[int], maximum: int) -> list[int]:
    unique = sorted(set(values))
    if len(unique) <= maximum:
        return unique
    if maximum <= 1:
        return [unique[len(unique) // 2]]
    return [
        unique[round(index * (len(unique) - 1) / (maximum - 1))]
        for index in range(maximum)
    ]


def _retention_candidates(
    spans: Sequence[_Span],
    *,
    fps: float,
    config: PerceptualQualityConfig,
) -> dict[int, list[int]]:
    stride = max(1, int(round(fps / config.retention_samples_per_second)))
    mandatory_by_source: defaultdict[int, list[int]] = defaultdict(list)
    additional_by_source: defaultdict[int, list[int]] = defaultdict(list)
    for span in spans:
        first = (
            span.start_frame
            + span.incoming_transition_frames
            + config.retention_segment_guard_frames
        )
        last = span.end_frame - config.retention_segment_guard_frames - 1
        if last < first:
            continue
        mandatory_by_source[span.source_index].append((first + last) // 2)
        aligned = ((first + stride - 1) // stride) * stride
        additional_by_source[span.source_index].extend(
            range(aligned, last + 1, stride)
        )
    candidates: dict[int, list[int]] = {}
    for source_index in sorted(
        set(mandatory_by_source) | set(additional_by_source)
    ):
        mandatory = sorted(set(mandatory_by_source[source_index]))
        mandatory_set = set(mandatory)
        additional = [
            frame_index
            for frame_index in sorted(
                set(additional_by_source[source_index])
            )
            if frame_index not in mandatory_set
        ]
        candidates[source_index] = sorted(
            mandatory
            + _evenly_limited(
                additional,
                config.maximum_retention_samples_per_source,
            )
        )
    return candidates


def _rendered_source_frame_index(
    span: _Span,
    *,
    relative_time: float,
    source_fps: float,
) -> int:
    """Mirror the renderer's ``trim -> setpts -> fps`` source-frame clock.

    FFmpeg ``trim=start=...`` retains the first frame whose presentation time is
    at or after the six-decimal start serialized by :mod:`crosscut.render`.
    ``setpts=PTS-STARTPTS`` then makes that retained frame the segment-local
    origin.  Rounding the absolute source time loses this reset and is one frame
    early whenever a non-frame-aligned start lies below the half-frame mark.
    """

    serialized_start = float(f"{span.source_start:.6f}")
    first_source_frame = max(
        0,
        int(math.ceil(serialized_start * source_fps - 1e-9)),
    )
    relative_source_frame = int(
        round(relative_time * span.source_rate * source_fps)
    )
    return max(0, first_source_frame + relative_source_frame)


def _span_for_frame(spans: Sequence[_Span], frame_index: int) -> _Span:
    # Sequential output decoding normally makes this O(1) in the caller; the
    # bounded binary search here also keeps exact-source sampling independent.
    low = 0
    high = len(spans)
    while low < high:
        middle = (low + high) // 2
        if frame_index < spans[middle].end_frame:
            high = middle
        else:
            low = middle + 1
    if low >= len(spans) or frame_index < spans[low].start_frame:
        raise IndexError("output frame is outside render-plan spans")
    return spans[low]


def _add_upper_gate(
    issues: _Issues,
    *,
    code: str,
    message: str,
    path: str,
    value: float,
    conditional: float,
    fail: float,
) -> None:
    if value > fail:
        issues.add(
            code,
            "fail",
            message,
            path=path,
            value=value,
            threshold=fail,
        )
    elif value > conditional:
        issues.add(
            code,
            "conditional",
            message,
            path=path,
            value=value,
            threshold=conditional,
        )


def _add_lower_gate(
    issues: _Issues,
    *,
    code: str,
    message: str,
    path: str,
    value: float,
    conditional: float,
    fail: float,
) -> None:
    if value < fail:
        issues.add(
            code,
            "fail",
            message,
            path=path,
            value=value,
            threshold=fail,
        )
    elif value < conditional:
        issues.add(
            code,
            "conditional",
            message,
            path=path,
            value=value,
            threshold=conditional,
        )


CaptureFactory = Callable[[str], Any]


def evaluate_post_render_perceptual_quality(
    video_path: Path,
    plan: Mapping[str, Any],
    *,
    config: PerceptualQualityConfig | None = None,
    capture_factory: CaptureFactory = cv2.VideoCapture,
) -> dict[str, Any]:
    """Return fail-closed scalar evidence for one final video and render plan."""

    settings = config or PerceptualQualityConfig()
    issues = _Issues()
    metrics: dict[str, Any] = {}
    try:
        fps, width, height, expected_frames, sources, spans = _parse_plan(plan)
    except ValueError as exc:
        issues.add(
            "perceptual_render_plan_invalid",
            "fail",
            "Perceptual QA could not bind frames to exact render-plan timing.",
            path="perceptual_quality.plan",
            value=str(exc),
        )
        return _payload(settings, metrics, issues)

    size = _analysis_size(width, height, settings.analysis_width)
    source_by_frame = np.empty(expected_frames, dtype=np.int32)
    run_by_frame = np.empty(expected_frames, dtype=np.int32)
    for span in spans:
        source_by_frame[span.start_frame : span.end_frame] = span.source_index
        run_by_frame[span.start_frame : span.end_frame] = span.run_index
    planned_retention_candidates = _retention_candidates(
        spans,
        fps=fps,
        config=settings,
    )
    retention_output_indices = {
        frame_index
        for frame_indices in planned_retention_candidates.values()
        for frame_index in frame_indices
    }
    from .render_reference_clock import load_render_reference_clock
    try:
        reference_clock = load_render_reference_clock(video_path, plan, retention_output_indices)
    except (OSError, ValueError, TypeError) as exc:
        issues.add("perceptual_render_source_clock_invalid", "fail",
            "Recorded render clock could not be rebound to the exact current plan and media.",
            path="perceptual_quality.source_clock", value=str(exc))
        return _payload(settings, metrics, issues)
    metrics["source_reference_clock"] = {
        "basis": "render_sidecar_reverified_native_clock" if reference_clock is not None else "legacy_no_render_clock_sidecar",
        "verified_reference_count": len(reference_clock.frames) if reference_clock is not None else 0,
        "timestamp_fallback_sources": list(reference_clock.timestamp_fallback_sources) if reference_clock is not None else [],
        "quality_approval": False,
    }
    retention_output_structures: dict[int, np.ndarray] = {}

    capture = capture_factory(str(Path(video_path)))
    if not capture.isOpened():
        issues.add(
            "perceptual_final_video_unavailable",
            "fail",
            "The final video could not be decoded for perceptual QA.",
            path="perceptual_quality.video",
            value=str(Path(video_path)),
        )
        return _payload(settings, metrics, issues)
    declared_width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
    declared_height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    declared_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_scalars: list[_FrameScalar] = []
    per_source_scalars: defaultdict[int, list[_FrameScalar]] = defaultdict(list)
    per_run_scalars: defaultdict[int, list[_FrameScalar]] = defaultdict(list)
    all_steps: defaultdict[str, list[float]] = defaultdict(list)
    same_run_steps: defaultdict[str, list[float]] = defaultdict(list)
    same_run_comparable = 0
    same_run_total = 0
    high_steps = {"sharpness": 0, "luma": 0, "chroma": 0}
    reversals = {"sharpness": 0, "luma": 0, "chroma": 0}
    previous_scalar: _FrameScalar | None = None
    previous_structure: np.ndarray | None = None
    previous_run: int | None = None
    previous_signed: dict[str, float] | None = None
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break
            scalar, structure = _frame_scalar(frame, size)
            frame_scalars.append(scalar)
            if frame_index in retention_output_indices:
                retention_output_structures[frame_index] = (
                    _geometry_structure(frame)
                )
            if frame_index < expected_frames:
                source_index = int(source_by_frame[frame_index])
                run_index = int(run_by_frame[frame_index])
                per_source_scalars[source_index].append(scalar)
                per_run_scalars[run_index].append(scalar)
            else:
                run_index = -1
            if previous_scalar is not None:
                signed = {
                    "sharpness": _sharpness_log_delta(previous_scalar, scalar),
                    "luma": scalar.luma - previous_scalar.luma,
                    "chroma": _color_delta(previous_scalar, scalar),
                }
                all_steps["sharpness"].append(abs(signed["sharpness"]))
                all_steps["luma"].append(abs(signed["luma"]))
                all_steps["chroma"].append(signed["chroma"])
                if frame_index < expected_frames and run_index == previous_run:
                    same_run_total += 1
                    assert previous_structure is not None
                    structure_change = _structure_delta(previous_structure, structure)
                    if (
                        structure_change
                        <= settings.maximum_structure_delta_for_temporal_comparison
                    ):
                        same_run_comparable += 1
                        same_run_steps["structure"].append(structure_change)
                        for axis in ("sharpness", "luma", "chroma"):
                            same_run_steps[axis].append(abs(signed[axis]))
                        thresholds = {
                            "sharpness": settings.temporal_sharpness_log_step,
                            "luma": settings.temporal_luma_step,
                            "chroma": settings.temporal_chroma_step,
                        }
                        for axis, threshold in thresholds.items():
                            if abs(signed[axis]) > threshold:
                                high_steps[axis] += 1
                        # Chroma is a magnitude, so direction reversal needs
                        # its signed a/b vector and is intentionally omitted.
                        if previous_signed is not None:
                            for axis in ("sharpness", "luma"):
                                if (
                                    abs(signed[axis]) > thresholds[axis]
                                    and abs(previous_signed[axis]) > thresholds[axis]
                                    and signed[axis] * previous_signed[axis] < 0.0
                                ):
                                    reversals[axis] += 1
                        previous_signed = signed
                    else:
                        previous_signed = None
                else:
                    previous_signed = None
            previous_scalar = scalar
            previous_structure = structure
            previous_run = run_index
            frame_index += 1
    finally:
        capture.release()

    dimensions_match = (declared_width, declared_height) == (width, height)
    fps_matches = math.isfinite(declared_fps) and abs(declared_fps - fps) <= 0.05
    frame_count_matches = frame_index == expected_frames
    metrics["video"] = {
        "path": str(Path(video_path).resolve()),
        "declared_width": declared_width,
        "declared_height": declared_height,
        "declared_fps": declared_fps if math.isfinite(declared_fps) else None,
        "expected_frame_count": expected_frames,
        "decoded_frame_count": frame_index,
        "analysis_width": size[0],
        "analysis_height": size[1],
        "dimensions_match_plan": dimensions_match,
        "fps_matches_plan": fps_matches,
        "full_frame_coverage": frame_count_matches,
    }
    for code, passed, value, threshold in (
        (
            "perceptual_dimensions_mismatch",
            dimensions_match,
            [declared_width, declared_height],
            [width, height],
        ),
        ("perceptual_fps_mismatch", fps_matches, declared_fps, fps),
        (
            "perceptual_frame_count_mismatch",
            frame_count_matches,
            frame_index,
            expected_frames,
        ),
    ):
        if not passed:
            issues.add(
                code,
                "fail",
                "Final-video perceptual evidence does not cover the authored media contract.",
                path="perceptual_quality.video",
                value=value,
                threshold=threshold,
            )
    if not frame_scalars:
        issues.add(
            "perceptual_no_decoded_frames",
            "fail",
            "No final-video frame was available for perceptual analysis.",
            path="perceptual_quality.video.decoded_frame_count",
            value=0,
            threshold=expected_frames,
        )
        return _payload(settings, metrics, issues)

    comparable_fraction = (
        same_run_comparable / same_run_total if same_run_total else None
    )
    high_step_fractions = {
        axis: count / same_run_comparable if same_run_comparable else None
        for axis, count in high_steps.items()
    }
    reversal_denominator = max(0, same_run_comparable - 1)
    reversal_fractions = {
        axis: count / reversal_denominator if reversal_denominator else None
        for axis, count in reversals.items()
    }
    metrics["temporal_pumping"] = {
        "full_timeline_pair_count": max(0, len(frame_scalars) - 1),
        "full_timeline_absolute_step": {
            axis: _distribution(all_steps[axis])
            for axis in ("sharpness", "luma", "chroma")
        },
        "same_source_run_pair_count": same_run_total,
        "same_source_run_comparable_pair_count": same_run_comparable,
        "same_source_run_comparable_fraction": comparable_fraction,
        "same_source_run_absolute_step": {
            axis: _distribution(same_run_steps[axis])
            for axis in ("sharpness", "luma", "chroma", "structure")
        },
        "high_step_count": high_steps,
        "high_step_fraction": high_step_fractions,
        "alternating_reversal_count": reversals,
        "alternating_reversal_fraction": reversal_fractions,
    }
    if same_run_comparable < settings.minimum_same_run_comparable_pair_count:
        issues.add(
            "perceptual_same_run_temporal_evidence_insufficient",
            "fail" if same_run_comparable == 0 else "conditional",
            "Too few structure-comparable same-source pairs were available to rule out pumping.",
            path="perceptual_quality.temporal_pumping.same_source_run_comparable_pair_count",
            value=same_run_comparable,
            threshold=settings.minimum_same_run_comparable_pair_count,
        )
    else:
        for axis, fraction in high_step_fractions.items():
            assert fraction is not None
            _add_upper_gate(
                issues,
                code=f"perceptual_{axis}_high_step_pumping",
                message=f"Repeated same-source {axis} changes indicate temporal pumping.",
                path=f"perceptual_quality.temporal_pumping.high_step_fraction.{axis}",
                value=fraction,
                conditional=settings.temporal_high_step_fraction_conditional,
                fail=settings.temporal_high_step_fraction_fail,
            )
        for axis in ("sharpness", "luma"):
            fraction = reversal_fractions[axis]
            if fraction is None:
                continue
            _add_upper_gate(
                issues,
                code=f"perceptual_{axis}_alternating_pumping",
                message=f"Alternating same-source {axis} reversals indicate flicker or pumping.",
                path=f"perceptual_quality.temporal_pumping.alternating_reversal_fraction.{axis}",
                value=fraction,
                conditional=settings.temporal_reversal_fraction_conditional,
                fail=settings.temporal_reversal_fraction_fail,
            )

    used_sources = sorted({span.source_index for span in spans})
    per_source: list[dict[str, Any]] = []
    valid_source_medians: list[tuple[int, _FrameScalar]] = []
    for source_index in used_sources:
        values = per_source_scalars[source_index]
        if values:
            item = _FrameScalar(
                detail=median(item.detail for item in values),
                luma=median(item.luma for item in values),
                chroma_a=median(item.chroma_a for item in values),
                chroma_b=median(item.chroma_b for item in values),
                contrast=median(item.contrast for item in values),
            )
            if len(values) >= settings.minimum_per_source_output_frame_count:
                valid_source_medians.append((source_index, item))
            per_source.append(
                {
                    "source_index": source_index,
                    "decoded_output_frame_count": len(values),
                    "median_detail": item.detail,
                    "median_luma": item.luma,
                    "median_chroma_a": item.chroma_a,
                    "median_chroma_b": item.chroma_b,
                    "median_contrast": item.contrast,
                }
            )
        else:
            per_source.append(
                {"source_index": source_index, "decoded_output_frame_count": 0}
            )
    source_detail_span = (
        max(math.log(item.detail + 1e-4) for _, item in valid_source_medians)
        - min(math.log(item.detail + 1e-4) for _, item in valid_source_medians)
        if len(valid_source_medians) >= 2
        else None
    )
    source_luma_span = (
        max(item.luma for _, item in valid_source_medians)
        - min(item.luma for _, item in valid_source_medians)
        if len(valid_source_medians) >= 2
        else None
    )
    source_chroma_span = (
        max(
            math.hypot(left.chroma_a - right.chroma_a, left.chroma_b - right.chroma_b)
            for left_index, (_, left) in enumerate(valid_source_medians)
            for _, right in valid_source_medians[left_index + 1 :]
        )
        if len(valid_source_medians) >= 2
        else None
    )
    metrics["source_median_consistency"] = {
        "used_source_count": len(used_sources),
        "adequately_observed_source_count": len(valid_source_medians),
        "minimum_output_frames_per_source": settings.minimum_per_source_output_frame_count,
        "per_source": per_source,
        "sharpness_log_median_span": source_detail_span,
        "luma_median_span": source_luma_span,
        "chroma_median_span": source_chroma_span,
    }
    if len(valid_source_medians) != len(used_sources):
        issues.add(
            "perceptual_source_median_evidence_incomplete",
            "conditional",
            "Not every used source has enough final-output frames for a stable source median.",
            path="perceptual_quality.source_median_consistency",
            value=len(valid_source_medians),
            threshold=len(used_sources),
        )
    for name, value, conditional, fail in (
        (
            "sharpness",
            source_detail_span,
            settings.source_sharpness_log_span_conditional,
            settings.source_sharpness_log_span_fail,
        ),
        (
            "luma",
            source_luma_span,
            settings.source_luma_median_span_conditional,
            settings.source_luma_median_span_fail,
        ),
        (
            "chroma",
            source_chroma_span,
            settings.source_chroma_median_span_conditional,
            settings.source_chroma_median_span_fail,
        ),
    ):
        if value is not None:
            metric_name = (
                "sharpness_log_median_span"
                if name == "sharpness"
                else f"{name}_median_span"
            )
            _add_upper_gate(
                issues,
                code=f"perceptual_source_{name}_median_span",
                message=(
                    f"Used-source {name} medians vary enough to create visible "
                    "quality oscillation."
                ),
                path=f"perceptual_quality.source_median_consistency.{metric_name}",
                value=value,
                conditional=conditional,
                fail=fail,
            )

    run_meta: list[dict[str, Any]] = []
    spans_by_run: defaultdict[int, list[_Span]] = defaultdict(list)
    for span in spans:
        spans_by_run[span.run_index].append(span)
    for run_index in sorted(spans_by_run):
        run_spans = spans_by_run[run_index]
        values = per_run_scalars[run_index]
        if not values:
            continue
        run_meta.append(
            {
                "run_index": run_index,
                "source_index": run_spans[0].source_index,
                "duration_seconds": sum(item.duration for item in run_spans),
                "detail": median(item.detail for item in values),
                "luma": median(item.luma for item in values),
                "chroma_a": median(item.chroma_a for item in values),
                "chroma_b": median(item.chroma_b for item in values),
            }
        )
    rapid_shocks: list[float] = []
    rapid_boundary_evidence: list[dict[str, Any]] = []
    rapid_high_count = 0
    for left, right in zip(run_meta, run_meta[1:]):
        if (
            left["source_index"] == right["source_index"]
            or max(left["duration_seconds"], right["duration_seconds"])
            > settings.rapid_run_maximum_seconds
        ):
            continue
        detail_step = abs(
            math.log((right["detail"] + 1e-4) / (left["detail"] + 1e-4))
        )
        luma_step = abs(right["luma"] - left["luma"])
        chroma_step = math.hypot(
            right["chroma_a"] - left["chroma_a"],
            right["chroma_b"] - left["chroma_b"],
        )
        normalized = max(
            detail_step / settings.temporal_sharpness_log_step,
            luma_step / settings.temporal_luma_step,
            chroma_step / settings.temporal_chroma_step,
        )
        normalized_axes = {
            "sharpness": detail_step / settings.temporal_sharpness_log_step,
            "luma": luma_step / settings.temporal_luma_step,
            "chroma": chroma_step / settings.temporal_chroma_step,
        }
        dominant_axis = max(normalized_axes, key=normalized_axes.__getitem__)
        rapid_shocks.append(normalized)
        rapid_high_count += normalized > 1.0
        rapid_boundary_evidence.append(
            {
                "outgoing_run_index": left["run_index"],
                "incoming_run_index": right["run_index"],
                "outgoing_source_index": left["source_index"],
                "incoming_source_index": right["source_index"],
                "outgoing_duration_seconds": left["duration_seconds"],
                "incoming_duration_seconds": right["duration_seconds"],
                "sharpness_log_step": detail_step,
                "luma_step": luma_step,
                "chroma_step": chroma_step,
                "normalized_axes": normalized_axes,
                "normalized_shock": normalized,
                "dominant_axis": dominant_axis,
                "above_threshold": normalized > 1.0,
            }
        )
    aba_count = 0
    strong_aba_count = 0
    for first, middle, last in zip(run_meta, run_meta[1:], run_meta[2:]):
        if (
            first["source_index"] != last["source_index"]
            or first["source_index"] == middle["source_index"]
            or sum(item["duration_seconds"] for item in (first, middle, last))
            > settings.rapid_run_maximum_seconds * 3.0
        ):
            continue
        aba_count += 1
        first_step = abs(
            math.log((middle["detail"] + 1e-4) / (first["detail"] + 1e-4))
        )
        second_step = abs(
            math.log((last["detail"] + 1e-4) / (middle["detail"] + 1e-4))
        )
        strong_aba_count += (
            first_step > settings.temporal_sharpness_log_step
            and second_step > settings.temporal_sharpness_log_step
        )
    rapid_fraction = (
        rapid_high_count / len(rapid_shocks) if rapid_shocks else None
    )
    metrics["rapid_quality_alternation"] = {
        "source_run_count": len(run_meta),
        "per_run": run_meta,
        "rapid_source_switch_boundary_count": len(rapid_shocks),
        "high_quality_shock_count": rapid_high_count,
        "high_quality_shock_fraction": rapid_fraction,
        "normalized_shock": _distribution(rapid_shocks),
        "rapid_aba_count": aba_count,
        "strong_sharpness_aba_count": strong_aba_count,
        "per_boundary": rapid_boundary_evidence,
    }
    if (
        rapid_fraction is not None
        and len(rapid_shocks) >= settings.minimum_rapid_boundary_count
    ):
        _add_upper_gate(
            issues,
            code="perceptual_rapid_quality_alternation",
            message=(
                "Rapid source switching repeatedly alternates strongly "
                "mismatched visible quality."
            ),
            path="perceptual_quality.rapid_quality_alternation.high_quality_shock_fraction",
            value=rapid_fraction,
            conditional=settings.rapid_high_shock_fraction_conditional,
            fail=settings.rapid_high_shock_fraction_fail,
        )

    candidates = {
        source_index: [
            frame_index
            for frame_index in frame_indices
            if frame_index < len(frame_scalars)
        ]
        for source_index, frame_indices in planned_retention_candidates.items()
    }
    requested_by_segment = {span.segment_index: 0 for span in spans}
    for source_index, frame_indices in candidates.items():
        for frame_index in frame_indices:
            span = _span_for_frame(spans, frame_index)
            if span.source_index != source_index:
                raise ValueError(
                    "retention candidate escaped its render-plan source"
                )
            requested_by_segment[span.segment_index] += 1
    retention_ratios: list[float] = []
    exact_structure_similarities: list[float] = []
    per_source_retention: list[dict[str, Any]] = []
    decoded_by_source: dict[int, int] = {}
    low_detail_by_source: dict[int, int] = {}
    geometry_failure_by_source: dict[int, int] = {}
    decoded_by_segment: defaultdict[int, int] = defaultdict(int)
    low_detail_by_segment: defaultdict[int, int] = defaultdict(int)
    geometry_failure_by_segment: defaultdict[int, int] = defaultdict(int)
    retention_ratios_by_segment: defaultdict[int, list[float]] = defaultdict(
        list
    )
    structure_similarities_by_segment: defaultdict[
        int, list[float]
    ] = defaultdict(list)
    for source_index in used_sources:
        indices = candidates.get(source_index, [])
        source_capture = capture_factory(str(sources[source_index]))
        ratios: list[float] = []
        structure_similarities: list[float] = []
        low_detail = 0
        geometry_failures = 0
        decoded = 0
        source_fps = (
            float(source_capture.get(cv2.CAP_PROP_FPS))
            if source_capture.isOpened()
            else 0.0
        )
        if source_capture.isOpened() and math.isfinite(source_fps) and source_fps > 0.0:
            try:
                for output_index in indices:
                    span = _span_for_frame(spans, output_index)
                    relative_time = (output_index - span.start_frame) / fps
                    native_reference = reference_clock.frames.get(output_index) if reference_clock is not None else None
                    if native_reference is not None:
                        if (native_reference.source_index != source_index
                                or not math.isclose(source_fps, reference_clock.source_fps[source_index], rel_tol=1e-6)):
                            raise ValueError("decoded source metadata differs from the recorded renderer clock")
                        source_frame_index = native_reference.native_frame_index
                    else:
                        source_frame_index = _rendered_source_frame_index(
                            span, relative_time=relative_time, source_fps=source_fps)
                    source_capture.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        source_frame_index,
                    )
                    success, source_frame = source_capture.read()
                    if not success or source_frame is None:
                        continue
                    decoded += 1
                    decoded_by_segment[span.segment_index] += 1
                    try:
                        reference_view = _reference_view_for_span(
                            source_frame,
                            span,
                            relative_time=relative_time,
                            size=size,
                        )
                    except ValueError:
                        geometry_failures += 1
                        geometry_failure_by_segment[span.segment_index] += 1
                        continue
                    reference_scalar, _reference_structure = _frame_scalar(
                        reference_view,
                        size,
                    )
                    reference_geometry_structure = _geometry_structure(
                        reference_view
                    )
                    if reference_scalar.detail < settings.minimum_reference_detail:
                        low_detail += 1
                        low_detail_by_segment[span.segment_index] += 1
                        continue
                    output_scalar = frame_scalars[output_index]
                    ratio = _contrast_normalized_detail(
                        output_scalar
                    ) / _contrast_normalized_detail(reference_scalar)
                    if math.isfinite(ratio):
                        ratios.append(ratio)
                        retention_ratios.append(ratio)
                        retention_ratios_by_segment[span.segment_index].append(
                            ratio
                        )
                    output_structure = retention_output_structures.get(
                        output_index
                    )
                    similarity = (
                        _structure_similarity(
                            output_structure,
                            reference_geometry_structure,
                        )
                        if output_structure is not None
                        else None
                    )
                    if similarity is not None:
                        structure_similarities.append(similarity)
                        exact_structure_similarities.append(similarity)
                        structure_similarities_by_segment[
                            span.segment_index
                        ].append(similarity)
            finally:
                source_capture.release()
        else:
            source_capture.release()
            issues.add(
                f"perceptual_source_video_unavailable_{source_index}",
                "fail",
                "A used source video could not provide exact source-time evidence.",
                path=(
                    "perceptual_quality.exact_source_retention."
                    f"per_source[source_index={source_index}]"
                ),
                value=source_index,
            )
        decoded_by_source[source_index] = decoded
        low_detail_by_source[source_index] = low_detail
        geometry_failure_by_source[source_index] = geometry_failures
        per_source_retention.append(
            {
                "source_index": source_index,
                "requested_sample_count": len(indices),
                "decoded_source_sample_count": decoded,
                "low_detail_excluded_count": low_detail,
                "geometry_reprojection_failure_count": geometry_failures,
                "comparable_sample_count": len(ratios),
                "structure_comparable_sample_count": len(
                    structure_similarities
                ),
                "detail_retention_ratio": _distribution(ratios),
                "exact_structure_similarity": _distribution(
                    structure_similarities
                ),
                "source_fps": (
                    source_fps
                    if math.isfinite(source_fps) and source_fps > 0.0
                    else None
                ),
            }
        )
    per_segment_retention: list[dict[str, Any]] = []
    for span in spans:
        segment_index = span.segment_index
        requested = requested_by_segment[segment_index]
        decoded = decoded_by_segment[segment_index]
        segment_ratios = retention_ratios_by_segment[segment_index]
        segment_structures = structure_similarities_by_segment[segment_index]
        structure_distribution = _distribution(segment_structures)
        structure_covered = bool(segment_structures)
        structure_maximum = structure_distribution["maximum"]
        structure_binding_passed = bool(
            structure_covered
            and structure_maximum is not None
            and structure_maximum
            >= settings.exact_structure_similarity_p10_fail
        )
        if requested == 0:
            incomparability_reason = (
                "no_non_overlap_interior_frame_after_transition_and_guard"
            )
        elif decoded == 0:
            incomparability_reason = "source_frame_decode_unavailable"
        elif geometry_failure_by_segment[segment_index]:
            incomparability_reason = "geometry_reprojection_failed"
        elif not segment_structures:
            incomparability_reason = (
                "reference_detail_or_structure_below_comparability_floor"
            )
        else:
            incomparability_reason = None
        per_segment_retention.append(
            {
                "segment_index": segment_index,
                "source_index": span.source_index,
                "start_frame": span.start_frame,
                "end_frame": span.end_frame,
                "incoming_transition_frames": (
                    span.incoming_transition_frames
                ),
                "requested_sample_count": requested,
                "decoded_source_sample_count": decoded,
                "low_detail_excluded_count": low_detail_by_segment[
                    segment_index
                ],
                "geometry_reprojection_failure_count": (
                    geometry_failure_by_segment[segment_index]
                ),
                "comparable_sample_count": len(segment_ratios),
                "structure_comparable_sample_count": len(
                    segment_structures
                ),
                "detail_retention_ratio": _distribution(segment_ratios),
                "exact_structure_similarity": structure_distribution,
                "structure_coverage_passed": structure_covered,
                "structure_binding_passed": structure_binding_passed,
                "incomparability_reason": incomparability_reason,
            }
        )
    retention_distribution = _distribution(retention_ratios)
    structure_similarity_distribution = _distribution(
        exact_structure_similarities
    )
    covered_sources = sum(
        item["comparable_sample_count"] >= settings.minimum_retention_samples_per_source
        for item in per_source_retention
    )
    structure_covered_sources = sum(
        item["structure_comparable_sample_count"]
        >= settings.minimum_retention_samples_per_source
        for item in per_source_retention
    )
    structure_covered_segments = sum(
        item["structure_coverage_passed"]
        for item in per_segment_retention
    )
    structure_bound_segments = sum(
        item["structure_binding_passed"]
        for item in per_segment_retention
    )
    uncovered_segment_indices = [
        item["segment_index"]
        for item in per_segment_retention
        if not item["structure_coverage_passed"]
    ]
    failed_binding_segment_indices = [
        item["segment_index"]
        for item in per_segment_retention
        if item["structure_coverage_passed"]
        and not item["structure_binding_passed"]
    ]
    metrics["exact_source_retention"] = {
        "mapping_basis": (
            "render_sidecar_reverified_native_sampling_with_explicit_timestamp_fallback"
            if reference_clock is not None else
            "ffmpeg_trim_first_frame_at_or_after_six_decimal_source_start_"
            "plus_segment_relative_time_times_source_playback_rate"
        ),
        "comparison_domain": (
            "render_plan_crop_pan_and_editorial_geometry_reprojected_"
            "contrast_normalized_detail_and_photometric_normalized_"
            "structure_on_non_overlap_interior_frames"
        ),
        "segment_sampling_policy": (
            "one_mandatory_safe_interior_midpoint_per_render_segment_"
            "then_source_capped_additional_samples"
        ),
        "maximum_additional_samples_per_source": (
            settings.maximum_retention_samples_per_source
        ),
        "used_source_count": len(used_sources),
        "adequately_covered_source_count": covered_sources,
        "minimum_samples_per_used_source": settings.minimum_retention_samples_per_source,
        "requested_sample_count": sum(len(values) for values in candidates.values()),
        "decoded_source_sample_count": sum(decoded_by_source.values()),
        "comparable_sample_count": len(retention_ratios),
        "structure_comparable_sample_count": len(
            exact_structure_similarities
        ),
        "structure_adequately_covered_source_count": (
            structure_covered_sources
        ),
        "render_segment_count": len(spans),
        "structure_covered_segment_count": structure_covered_segments,
        "structure_bound_segment_count": structure_bound_segments,
        "structure_segment_coverage_complete": (
            structure_covered_segments == len(spans)
        ),
        "uncovered_segment_indices": uncovered_segment_indices,
        "failed_binding_segment_indices": failed_binding_segment_indices,
        "low_detail_excluded_count": sum(low_detail_by_source.values()),
        "geometry_reprojection_failure_count": sum(
            geometry_failure_by_source.values()
        ),
        "detail_retention_ratio": retention_distribution,
        "exact_structure_similarity": structure_similarity_distribution,
        "per_source": per_source_retention,
        "per_segment": per_segment_retention,
    }
    if sum(geometry_failure_by_source.values()):
        issues.add(
            "perceptual_render_geometry_reprojection_failed",
            "fail",
            (
                "A planned crop or Dynamic Crop v2 transform could not be "
                "reprojected for source retention."
            ),
            path=(
                "perceptual_quality.exact_source_retention."
                "geometry_reprojection_failure_count"
            ),
            value=sum(geometry_failure_by_source.values()),
            threshold=0,
        )
    if not retention_ratios:
        issues.add(
            "perceptual_exact_source_retention_unavailable",
            "fail",
            "No exact source-time comparison survived the conservative comparability contract.",
            path="perceptual_quality.exact_source_retention.comparable_sample_count",
            value=0,
            threshold=max(1, settings.minimum_retention_samples_per_source),
        )
    elif covered_sources != len(used_sources):
        issues.add(
            "perceptual_exact_source_retention_coverage_incomplete",
            "conditional",
            "Not every used source has enough exact source-time retention samples.",
            path="perceptual_quality.exact_source_retention.adequately_covered_source_count",
            value=covered_sources,
            threshold=len(used_sources),
        )
    if not exact_structure_similarities:
        issues.add(
            "perceptual_exact_structure_binding_unavailable",
            "fail",
            "No final-frame sample could be structurally bound to its exact rendered source view.",
            path=(
                "perceptual_quality.exact_source_retention."
                "structure_comparable_sample_count"
            ),
            value=0,
            threshold=max(1, settings.minimum_retention_samples_per_source),
        )
    elif structure_covered_sources != len(used_sources):
        issues.add(
            "perceptual_exact_structure_binding_coverage_incomplete",
            "fail",
            "Every used source must have enough exact final-frame structure checks.",
            path=(
                "perceptual_quality.exact_source_retention."
                "structure_adequately_covered_source_count"
            ),
            value=structure_covered_sources,
            threshold=len(used_sources),
        )
    if structure_covered_segments != len(spans):
        issues.add(
            "perceptual_exact_structure_segment_coverage_incomplete",
            "fail",
            (
                "Every render segment must provide at least one exact "
                "non-overlap structure comparison."
            ),
            path=(
                "perceptual_quality.exact_source_retention."
                "structure_covered_segment_count"
            ),
            value={
                "covered_segment_count": structure_covered_segments,
                "render_segment_count": len(spans),
                "uncovered_segment_indices": uncovered_segment_indices,
            },
            threshold={
                "covered_segment_count": len(spans),
                "uncovered_segment_indices": [],
            },
        )
    if failed_binding_segment_indices:
        issues.add(
            "perceptual_exact_structure_segment_binding_failed",
            "fail",
            (
                "At least one covered render segment has no frame that "
                "matches its exact RenderPlan geometry."
            ),
            path=(
                "perceptual_quality.exact_source_retention."
                "failed_binding_segment_indices"
            ),
            value=failed_binding_segment_indices,
            threshold={
                "maximum_similarity_at_least": (
                    settings.exact_structure_similarity_p10_fail
                ),
                "failed_binding_segment_indices": [],
            },
        )
    retention_p10 = retention_distribution["p10"]
    retention_median = retention_distribution["median"]
    if retention_p10 is not None:
        _add_lower_gate(
            issues,
            code="perceptual_detail_retention_p10",
            message=(
                "The weakest exact source-time samples lose too much source "
                "detail in the final render."
            ),
            path="perceptual_quality.exact_source_retention.detail_retention_ratio.p10",
            value=float(retention_p10),
            conditional=settings.retention_ratio_p10_conditional,
            fail=settings.retention_ratio_p10_fail,
        )
    if retention_median is not None:
        _add_lower_gate(
            issues,
            code="perceptual_detail_retention_median",
            message=(
                "Typical exact source-time samples lose too much source detail "
                "in the final render."
            ),
            path="perceptual_quality.exact_source_retention.detail_retention_ratio.median",
            value=float(retention_median),
            conditional=settings.retention_ratio_median_conditional,
            fail=settings.retention_ratio_median_fail,
        )
    structure_p10 = structure_similarity_distribution["p10"]
    structure_median = structure_similarity_distribution["median"]
    if structure_p10 is not None:
        _add_lower_gate(
            issues,
            code="perceptual_exact_structure_similarity_p10",
            message=(
                "The weakest final-frame samples do not match the exact "
                "RenderPlan crop and transform geometry."
            ),
            path=(
                "perceptual_quality.exact_source_retention."
                "exact_structure_similarity.p10"
            ),
            value=float(structure_p10),
            conditional=(
                settings.exact_structure_similarity_p10_conditional
            ),
            fail=settings.exact_structure_similarity_p10_fail,
        )
    if structure_median is not None:
        _add_lower_gate(
            issues,
            code="perceptual_exact_structure_similarity_median",
            message=(
                "Typical final-frame samples do not match the exact "
                "RenderPlan crop and transform geometry."
            ),
            path=(
                "perceptual_quality.exact_source_retention."
                "exact_structure_similarity.median"
            ),
            value=float(structure_median),
            conditional=(
                settings.exact_structure_similarity_median_conditional
            ),
            fail=settings.exact_structure_similarity_median_fail,
        )
    return _payload(settings, metrics, issues)


def _payload(
    config: PerceptualQualityConfig,
    metrics: Mapping[str, Any],
    issues: _Issues,
) -> dict[str, Any]:
    verdict = issues.verdict
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_BASIS,
        "verdict": verdict,
        "passed": verdict == "PASS",
        "issues": list(issues.items),
        "policy": config.to_dict(),
        "metrics": dict(metrics),
        "serialization_policy": {
            "raw_frames_serialized": False,
            "per_frame_descriptors_serialized": False,
            "sample_timestamps_serialized": False,
            "scalar_aggregate_provenance_only": True,
        },
    }


__all__ = [
    "ANALYSIS_BASIS",
    "PerceptualQualityConfig",
    "SCHEMA_VERSION",
    "evaluate_post_render_perceptual_quality",
]
