"""Conservative virtual-camera crops for basic subject-matched cross-edit.

This module never identifies a person. A dominant face is preferred; when a
face is absent or ambiguous, a stable high-confidence aggregate human
silhouette may provide a smaller group-formation crop. Unsafe measurements
produce an explicit raw-frame fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from .render import NormalizedCrop, PanCropKeyframe, PanCropTrack
from .visual_score import VisualTimelinePlan


@dataclass(frozen=True, slots=True)
class FaceMatchConfig:
    maximum_zoom: float = 1.35
    target_screen_x: float = 0.5
    target_screen_y: float = 0.42
    target_percentile: float = 75.0
    target_scale_padding: float = 1.04
    maximum_target_face_height: float = 0.46
    maximum_position_error: float = 0.105
    minimum_subject_crop_margin: float = 0.018
    maximum_source_aspect_deviation: float = 0.04
    target_face_height_override: float | None = None
    maximum_silhouette_zoom: float = 1.18
    silhouette_target_percentile: float = 65.0
    maximum_target_silhouette_height: float = 0.90
    minimum_silhouette_confidence: float = 0.82
    minimum_silhouette_height: float = 0.16
    maximum_silhouette_center_span: float = 0.15
    maximum_silhouette_scale_ratio: float = 1.42
    silhouette_target_screen_y: float = 0.50

    def __post_init__(self) -> None:
        numeric = (
            self.maximum_zoom,
            self.target_screen_x,
            self.target_screen_y,
            self.target_percentile,
            self.target_scale_padding,
            self.maximum_target_face_height,
            self.maximum_position_error,
            self.minimum_subject_crop_margin,
            self.maximum_source_aspect_deviation,
            self.maximum_silhouette_zoom,
            self.silhouette_target_percentile,
            self.maximum_target_silhouette_height,
            self.minimum_silhouette_confidence,
            self.minimum_silhouette_height,
            self.maximum_silhouette_center_span,
            self.maximum_silhouette_scale_ratio,
            self.silhouette_target_screen_y,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("face-match configuration must be finite")
        if not 1.0 < self.maximum_zoom <= 1.6:
            raise ValueError("maximum_zoom must be conservative (1.0, 1.6]")
        if not 0.0 < self.target_screen_x < 1.0:
            raise ValueError("target_screen_x must be inside the frame")
        if not 0.0 < self.target_screen_y < 1.0:
            raise ValueError("target_screen_y must be inside the frame")
        if not 0.0 <= self.target_percentile <= 100.0:
            raise ValueError("target_percentile must be between 0 and 100")
        if self.target_scale_padding < 1.0:
            raise ValueError("target_scale_padding must be at least one")
        if not 0.0 < self.maximum_target_face_height < 1.0:
            raise ValueError("maximum target face height must be normalized")
        if not 0.0 <= self.maximum_position_error <= 0.25:
            raise ValueError("maximum position error must be in [0, 0.25]")
        if not 0.0 <= self.minimum_subject_crop_margin <= 0.1:
            raise ValueError("subject crop margin must be normalized")
        if self.maximum_source_aspect_deviation < 0.0:
            raise ValueError("source aspect deviation cannot be negative")
        if not 1.0 < self.maximum_silhouette_zoom <= 1.35:
            raise ValueError("silhouette zoom must be conservative (1.0, 1.35]")
        if not 0.0 <= self.silhouette_target_percentile <= 100.0:
            raise ValueError("silhouette target percentile must be in [0, 100]")
        if not 0.0 < self.maximum_target_silhouette_height <= 1.0:
            raise ValueError("maximum silhouette height must be normalized")
        if not 0.0 <= self.minimum_silhouette_confidence <= 1.0:
            raise ValueError("minimum silhouette confidence must be normalized")
        if not 0.0 < self.minimum_silhouette_height < 1.0:
            raise ValueError("minimum silhouette height must be normalized")
        if not 0.0 <= self.maximum_silhouette_center_span <= 0.3:
            raise ValueError("silhouette center span must be in [0, 0.3]")
        if self.maximum_silhouette_scale_ratio < 1.0:
            raise ValueError("silhouette scale ratio must be at least one")
        if not 0.0 < self.silhouette_target_screen_y < 1.0:
            raise ValueError("silhouette target y must be inside the frame")
        if self.target_face_height_override is not None and not (
            math.isfinite(self.target_face_height_override)
            and 0.0 < self.target_face_height_override < 1.0
        ):
            raise ValueError("target face height override must be normalized")


@dataclass(frozen=True, slots=True)
class FaceMatchSegment:
    segment_index: int
    source_index: int
    applied: bool
    fallback_reason: str | None
    observed_face_height: float | None
    target_face_height: float | None
    required_zoom: float | None
    applied_zoom: float | None
    pan_crop: PanCropTrack | None
    alignment_subject: str | None = None
    observed_silhouette_height: float | None = None
    target_silhouette_height: float | None = None

    @property
    def transition_tier(self) -> str:
        return (
            "moderate_confidence_aligned_hard_cut"
            if self.applied
            else "raw_hard_cut_fallback"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "source_index": self.source_index,
            "mode": (
                f"{self.alignment_subject}_match_pan_crop"
                if self.applied and self.alignment_subject is not None
                else "raw_full_frame"
            ),
            "transition_tier": self.transition_tier,
            "safety_reason": (
                f"bounded_{self.alignment_subject}_scale_translation"
                if self.applied and self.alignment_subject is not None
                else self.fallback_reason
            ),
            "alignment_subject": self.alignment_subject,
            "applied": self.applied,
            "fallback_reason": self.fallback_reason,
            "observed_face_height": self.observed_face_height,
            "target_face_height": self.target_face_height,
            "observed_silhouette_height": self.observed_silhouette_height,
            "target_silhouette_height": self.target_silhouette_height,
            "required_zoom": self.required_zoom,
            "applied_zoom": self.applied_zoom,
            "crop_keyframes": (
                [
                    {
                        "time_seconds": keyframe.time_seconds,
                        "crop_xywh": [
                            keyframe.crop.x,
                            keyframe.crop.y,
                            keyframe.crop.width,
                            keyframe.crop.height,
                        ],
                    }
                    for keyframe in self.pan_crop.keyframes
                ]
                if self.pan_crop is not None
                else []
            ),
        }


@dataclass(frozen=True, slots=True)
class FaceMatchPlan:
    segments: tuple[FaceMatchSegment, ...]
    target_face_height: float | None
    maximum_zoom: float
    target_screen_position: tuple[float, float]
    target_silhouette_height: float | None = None
    target_selection_policy: str = "deterministic_max_safe_coverage_then_scale_zoom"
    target_candidate_count: int = 0
    eligible_face_segment_count: int = 0
    silhouette_target_selection_policy: str = (
        "deterministic_max_safe_coverage_then_scale_zoom"
    )
    silhouette_target_candidate_count: int = 0
    eligible_silhouette_segment_count: int = 0
    analysis_basis: str = "dominant_face_or_silhouette_safe_virtual_camera_v3"

    @property
    def applied_count(self) -> int:
        return sum(item.applied for item in self.segments)

    @property
    def fallback_count(self) -> int:
        return len(self.segments) - self.applied_count

    @property
    def silhouette_applied_count(self) -> int:
        return sum(item.alignment_subject == "silhouette" for item in self.segments)

    @property
    def aligned_segment_count(self) -> int:
        return sum(
            item.transition_tier == "moderate_confidence_aligned_hard_cut"
            for item in self.segments
        )

    @property
    def raw_segment_count(self) -> int:
        return sum(
            item.transition_tier == "raw_hard_cut_fallback"
            for item in self.segments
        )

    def pan_crop_for(self, segment_index: int) -> PanCropTrack | None:
        matches = [
            item.pan_crop
            for item in self.segments
            if item.segment_index == segment_index
        ]
        if len(matches) != 1:
            raise ValueError("face-match plan must contain each segment exactly once")
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_basis": self.analysis_basis,
            "policy": "face_then_stable_silhouette_aligned_hard_cut_else_raw_full_frame",
            "target_face_height": self.target_face_height,
            "target_selection_policy": self.target_selection_policy,
            "target_candidate_count": self.target_candidate_count,
            "eligible_face_segment_count": self.eligible_face_segment_count,
            "target_silhouette_height": self.target_silhouette_height,
            "silhouette_target_selection_policy": (
                self.silhouette_target_selection_policy
            ),
            "silhouette_target_candidate_count": (
                self.silhouette_target_candidate_count
            ),
            "eligible_silhouette_segment_count": (
                self.eligible_silhouette_segment_count
            ),
            "maximum_zoom": self.maximum_zoom,
            "target_screen_position": list(self.target_screen_position),
            "applied_count": self.applied_count,
            "fallback_count": self.fallback_count,
            "silhouette_applied_count": self.silhouette_applied_count,
            "count_basis": "selected_segments",
            "aligned_segment_count": self.aligned_segment_count,
            "raw_segment_count": self.raw_segment_count,
            "blend_eligibility_note": (
                "aligned hard cut is a moderate-confidence tier; blends require "
                "separate stricter multi-frame handle validation"
            ),
            "segments": [item.to_dict() for item in self.segments],
        }


def _fallback(
    *,
    segment_index: int,
    source_index: int,
    reason: str,
    target: float | None,
    observed: float | None = None,
    required_zoom: float | None = None,
    alignment_subject: str | None = None,
    observed_silhouette_height: float | None = None,
    target_silhouette_height: float | None = None,
) -> FaceMatchSegment:
    return FaceMatchSegment(
        segment_index=segment_index,
        source_index=source_index,
        applied=False,
        fallback_reason=reason,
        observed_face_height=observed,
        target_face_height=target,
        required_zoom=required_zoom,
        applied_zoom=None,
        pan_crop=None,
        alignment_subject=alignment_subject,
        observed_silhouette_height=observed_silhouette_height,
        target_silhouette_height=target_silhouette_height,
    )


def _safe_silhouette_geometry(
    choice: Any,
    settings: FaceMatchConfig,
) -> tuple[
    tuple[tuple[float, float, float, float], ...],
    tuple[tuple[float, float], ...],
] | None:
    """Return a stable three-sample aggregate-human track or no crop signal."""

    if choice.face_match_candidate_failure == "internal_camera_cut":
        return None
    if any(sample is None for sample in choice.silhouette_samples_xywh):
        return None
    if any(center is None for center in choice.silhouette_centroid_samples_xy):
        return None
    if any(
        backend != "opencv_zoo_pphumanseg_2023mar"
        for backend in choice.silhouette_backends
    ):
        return None
    if min(choice.silhouette_confidences) < settings.minimum_silhouette_confidence:
        return None
    samples = tuple(
        sample for sample in choice.silhouette_samples_xywh if sample is not None
    )
    centers = tuple(
        center
        for center in choice.silhouette_centroid_samples_xy
        if center is not None
    )
    heights = tuple(sample[3] for sample in samples)
    if min(heights) < settings.minimum_silhouette_height:
        return None
    if max(heights) / min(heights) > settings.maximum_silhouette_scale_ratio:
        return None
    maximum_center_span = max(
        math.hypot(left[0] - right[0], left[1] - right[1])
        for left in centers
        for right in centers
    )
    if maximum_center_span > settings.maximum_silhouette_center_span:
        return None
    return samples, centers


def _positioned_crop_for_center(
    *,
    center: tuple[float, float],
    crop_width: float,
    crop_height: float,
    target_screen_x: float,
    target_screen_y: float,
    maximum_position_error: float,
) -> tuple[float, float, float, float] | None:
    """Place the virtual camera while respecting the requested screen position."""

    if any(
        not math.isfinite(float(value))
        for value in (*center, crop_width, crop_height)
    ):
        return None
    center_x, center_y = center
    left = min(
        max(center_x - target_screen_x * crop_width, 0.0),
        1.0 - crop_width,
    )
    top = min(
        max(center_y - target_screen_y * crop_height, 0.0),
        1.0 - crop_height,
    )
    actual_x = (center_x - left) / crop_width
    actual_y = (center_y - top) / crop_height
    if math.hypot(
        actual_x - target_screen_x,
        actual_y - target_screen_y,
    ) > maximum_position_error:
        return None
    return (left, top, crop_width, crop_height)


def _contained_crop_for_sample(
    *,
    sample: tuple[float, float, float, float],
    center: tuple[float, float],
    crop_width: float,
    crop_height: float,
    target_screen_x: float,
    target_screen_y: float,
    maximum_position_error: float,
    minimum_subject_crop_margin: float,
) -> tuple[float, float, float, float] | None:
    """Place a crop only when the complete subject bbox remains safely visible."""

    box_x, box_y, box_width, box_height = sample
    if (
        any(not math.isfinite(float(value)) for value in sample)
        or box_width <= 0.0
        or box_height <= 0.0
    ):
        return None
    crop = _positioned_crop_for_center(
        center=center,
        crop_width=crop_width,
        crop_height=crop_height,
        target_screen_x=target_screen_x,
        target_screen_y=target_screen_y,
        maximum_position_error=maximum_position_error,
    )
    if crop is None:
        return None
    left, top, crop_width, crop_height = crop
    inset_x = crop_width * minimum_subject_crop_margin
    inset_y = crop_height * minimum_subject_crop_margin
    epsilon = 1e-9
    if not (
        box_x >= left + inset_x - epsilon
        and box_y >= top + inset_y - epsilon
        and box_x + box_width <= left + crop_width - inset_x + epsilon
        and box_y + box_height <= top + crop_height - inset_y + epsilon
    ):
        return None
    return crop


def _alignment_metrics(
    *,
    samples: Sequence[tuple[float, float, float, float]],
    centers: Sequence[tuple[float, float]],
    source_aspect: float,
    target: float,
    target_screen_y: float,
    maximum_zoom: float,
    settings: FaceMatchConfig,
) -> tuple[float, float] | None:
    """Preview one fixed-scale crop using the same gates as the renderer path."""

    if len(samples) != 3 or len(centers) != 3:
        return None
    if abs(source_aspect - 16.0 / 9.0) > settings.maximum_source_aspect_deviation:
        return None
    heights = tuple(float(sample[3]) for sample in samples)
    if any(not math.isfinite(value) or value <= 0.0 for value in heights):
        return None
    observed = float(np.median(heights))
    required_zoom = target / observed
    if required_zoom < 0.94 - 1e-9:
        return None
    normalized_crop_aspect = (16.0 / 9.0) / source_aspect
    applied_zoom = max(1.025, required_zoom, normalized_crop_aspect)
    if applied_zoom > maximum_zoom + 1e-9:
        return None
    crop_height = 1.0 / applied_zoom
    crop_width = crop_height * normalized_crop_aspect
    if crop_width > 1.0 + 1e-8 or crop_height > 1.0 + 1e-8:
        return None
    for sample, center in zip(samples, centers, strict=True):
        if _contained_crop_for_sample(
            sample=sample,
            center=center,
            crop_width=crop_width,
            crop_height=crop_height,
            target_screen_x=settings.target_screen_x,
            target_screen_y=target_screen_y,
            maximum_position_error=settings.maximum_position_error,
            minimum_subject_crop_margin=settings.minimum_subject_crop_margin,
        ) is None:
            return None
    return (applied_zoom, observed * applied_zoom)


def _face_alignment_metrics(
    choice: Any,
    *,
    source_aspect: float,
    target: float,
    settings: FaceMatchConfig,
) -> tuple[float, float] | None:
    """Preview whether one dominant-face crop is safe at a proposed target."""

    if choice.face_match_candidate_failure is not None:
        return None
    if any(sample is None for sample in choice.face_samples_xywh):
        return None
    samples = tuple(
        sample for sample in choice.face_samples_xywh if sample is not None
    )
    centers = tuple(
        (sample[0] + sample[2] / 2.0, sample[1] + sample[3] / 2.0)
        for sample in samples
    )
    return _alignment_metrics(
        samples=samples,
        centers=centers,
        source_aspect=source_aspect,
        target=target,
        target_screen_y=settings.target_screen_y,
        maximum_zoom=settings.maximum_zoom,
        settings=settings,
    )


def _silhouette_alignment_metrics(
    choice: Any,
    *,
    source_aspect: float,
    target: float,
    settings: FaceMatchConfig,
) -> tuple[float, float] | None:
    geometry = _safe_silhouette_geometry(choice, settings)
    if geometry is None:
        return None
    samples, centers = geometry
    return _alignment_metrics(
        samples=samples,
        centers=centers,
        source_aspect=source_aspect,
        target=target,
        target_screen_y=settings.silhouette_target_screen_y,
        maximum_zoom=min(settings.maximum_zoom, settings.maximum_silhouette_zoom),
        settings=settings,
    )


def _optimized_face_target(
    *,
    visual_timeline: VisualTimelinePlan,
    source_aspect_ratios: Sequence[float],
    settings: FaceMatchConfig,
) -> tuple[float | None, int, int]:
    """Maximize safe aligned-cut coverage, then consistency, then low zoom."""

    eligible_heights: list[float] = []
    for choice in visual_timeline.choices:
        source_aspect = float(source_aspect_ratios[choice.source_index])
        samples = choice.face_samples_xywh
        if (
            choice.face_match_candidate_failure is not None
            or any(sample is None for sample in samples)
            or abs(source_aspect - 16.0 / 9.0)
            > settings.maximum_source_aspect_deviation
        ):
            continue
        heights = tuple(float(sample[3]) for sample in samples if sample is not None)
        if heights and all(math.isfinite(value) and value > 0.0 for value in heights):
            eligible_heights.append(float(np.median(heights)))
    if not eligible_heights:
        return (None, 0, 0)

    lower = min(0.94 * height for height in eligible_heights)
    upper = min(
        settings.maximum_target_face_height,
        max(settings.maximum_zoom * height for height in eligible_heights),
    )
    if upper < lower:
        return (None, 0, len(eligible_heights))
    legacy_seed = min(
        settings.maximum_target_face_height,
        float(np.percentile(eligible_heights, settings.target_percentile))
        * settings.target_scale_padding,
    )
    breakpoints = {
        factor * height
        for height in eligible_heights
        for factor in (0.94, 1.0, 1.025, settings.maximum_zoom)
        if lower <= factor * height <= upper
    }
    if lower <= legacy_seed <= upper:
        breakpoints.add(legacy_seed)
    # The fixed grid captures edge-clamp feasibility thresholds, which cannot be
    # inferred from face-height breakpoints alone. It is tiny relative to decode.
    grid = np.linspace(lower, upper, 161) if upper > lower else np.array([lower])
    candidates = tuple(
        sorted(
            {
                float(value)
                for value in (*breakpoints, *grid)
                if math.isfinite(float(value)) and 0.0 < value <= upper + 1e-9
            }
        )
    )
    best_target: float | None = None
    best_objective: tuple[float, ...] | None = None
    for candidate_target in candidates:
        metrics = tuple(
            metric
            for choice in visual_timeline.choices
            if (
                metric := _face_alignment_metrics(
                    choice,
                    source_aspect=float(
                        source_aspect_ratios[choice.source_index]
                    ),
                    target=candidate_target,
                    settings=settings,
                )
            )
            is not None
        )
        if not metrics:
            continue
        zooms = tuple(metric[0] for metric in metrics)
        result_heights = tuple(metric[1] for metric in metrics)
        geometric_mean = math.exp(
            sum(math.log(value) for value in result_heights) / len(result_heights)
        )
        scale_spread = math.sqrt(
            sum(math.log(value / geometric_mean) ** 2 for value in result_heights)
            / len(result_heights)
        )
        objective = (
            -float(len(metrics)),
            scale_spread,
            sum(zooms) / len(zooms),
            max(zooms),
            candidate_target,
        )
        if best_objective is None or objective < best_objective:
            best_objective = objective
            best_target = candidate_target
    return (best_target, len(candidates), len(eligible_heights))


def _optimized_silhouette_target(
    *,
    visual_timeline: VisualTimelinePlan,
    source_aspect_ratios: Sequence[float],
    settings: FaceMatchConfig,
) -> tuple[float | None, int, int]:
    """Optimize the conservative group-formation fallback with identical policy."""

    eligible_heights: list[float] = []
    for choice in visual_timeline.choices:
        source_aspect = float(source_aspect_ratios[choice.source_index])
        geometry = _safe_silhouette_geometry(choice, settings)
        if (
            geometry is None
            or abs(source_aspect - 16.0 / 9.0)
            > settings.maximum_source_aspect_deviation
        ):
            continue
        samples, _centers = geometry
        eligible_heights.append(float(np.median([sample[3] for sample in samples])))
    if not eligible_heights:
        return (None, 0, 0)

    maximum_zoom = min(settings.maximum_zoom, settings.maximum_silhouette_zoom)
    lower = min(0.94 * height for height in eligible_heights)
    upper = min(
        settings.maximum_target_silhouette_height,
        max(maximum_zoom * height for height in eligible_heights),
    )
    if upper < lower:
        return (None, 0, len(eligible_heights))
    legacy_seed = min(
        settings.maximum_target_silhouette_height,
        float(np.percentile(eligible_heights, settings.silhouette_target_percentile))
        * 1.01,
    )
    breakpoints = {
        factor * height
        for height in eligible_heights
        for factor in (0.94, 1.0, 1.025, maximum_zoom)
        if lower <= factor * height <= upper
    }
    if lower <= legacy_seed <= upper:
        breakpoints.add(legacy_seed)
    grid = np.linspace(lower, upper, 161) if upper > lower else np.array([lower])
    candidates = tuple(
        sorted(
            {
                float(value)
                for value in (*breakpoints, *grid)
                if math.isfinite(float(value)) and 0.0 < value <= upper + 1e-9
            }
        )
    )
    best_target: float | None = None
    best_objective: tuple[float, ...] | None = None
    for candidate_target in candidates:
        metrics = tuple(
            metric
            for choice in visual_timeline.choices
            if (
                metric := _silhouette_alignment_metrics(
                    choice,
                    source_aspect=float(
                        source_aspect_ratios[choice.source_index]
                    ),
                    target=candidate_target,
                    settings=settings,
                )
            )
            is not None
        )
        if not metrics:
            continue
        zooms = tuple(metric[0] for metric in metrics)
        result_heights = tuple(metric[1] for metric in metrics)
        geometric_mean = math.exp(
            sum(math.log(value) for value in result_heights) / len(result_heights)
        )
        scale_spread = math.sqrt(
            sum(math.log(value / geometric_mean) ** 2 for value in result_heights)
            / len(result_heights)
        )
        objective = (
            -float(len(metrics)),
            scale_spread,
            sum(zooms) / len(zooms),
            max(zooms),
            candidate_target,
        )
        if best_objective is None or objective < best_objective:
            best_objective = objective
            best_target = candidate_target
    return (best_target, len(candidates), len(eligible_heights))


def _alignment_attempt(
    *,
    choice: Any,
    alignment_subject: str,
    samples: Sequence[tuple[float, float, float, float]],
    centers: Sequence[tuple[float, float]],
    alignment_target: float,
    face_target: float | None,
    silhouette_target: float | None,
    source_aspect: float,
    target_screen_y: float,
    maximum_zoom: float,
    settings: FaceMatchConfig,
) -> FaceMatchSegment:
    """Build one face/silhouette crop or an explicit raw-frame attempt result."""

    observed = float(np.median([sample[3] for sample in samples]))

    def failed(reason: str, required_zoom: float | None = None) -> FaceMatchSegment:
        return _fallback(
            segment_index=choice.segment_index,
            source_index=choice.source_index,
            reason=reason,
            target=face_target,
            observed=observed if alignment_subject == "face" else None,
            required_zoom=required_zoom,
            alignment_subject=alignment_subject,
            observed_silhouette_height=(
                observed if alignment_subject == "silhouette" else None
            ),
            target_silhouette_height=(
                silhouette_target if alignment_subject == "silhouette" else None
            ),
        )

    if abs(source_aspect - 16.0 / 9.0) > settings.maximum_source_aspect_deviation:
        return failed("unsupported_source_aspect_ratio")
    required_zoom = alignment_target / observed
    if required_zoom < 0.94 - 1e-9:
        return failed(
            f"{alignment_subject}_larger_than_match_target",
            required_zoom,
        )
    normalized_crop_aspect = (16.0 / 9.0) / source_aspect
    applied_zoom = max(1.025, required_zoom, normalized_crop_aspect)
    if applied_zoom > maximum_zoom + 1e-9:
        return failed("required_zoom_exceeds_limit", required_zoom)
    crop_height = 1.0 / applied_zoom
    crop_width = crop_height * normalized_crop_aspect
    if crop_width > 1.0 + 1e-8 or crop_height > 1.0 + 1e-8:
        return failed("aspect_preserving_crop_does_not_fit", required_zoom)

    keyframe_times = (0.0, choice.duration_seconds / 2.0, choice.duration_seconds)
    keyframes: list[PanCropKeyframe] = []
    for time_seconds, sample, center in zip(
        keyframe_times,
        samples,
        centers,
        strict=True,
    ):
        crop = _contained_crop_for_sample(
            sample=sample,
            center=center,
            crop_width=crop_width,
            crop_height=crop_height,
            target_screen_x=settings.target_screen_x,
            target_screen_y=target_screen_y,
            maximum_position_error=settings.maximum_position_error,
            minimum_subject_crop_margin=settings.minimum_subject_crop_margin,
        )
        if crop is None:
            position_only = _positioned_crop_for_center(
                center=center,
                crop_width=crop_width,
                crop_height=crop_height,
                target_screen_x=settings.target_screen_x,
                target_screen_y=target_screen_y,
                maximum_position_error=settings.maximum_position_error,
            )
            return failed(
                (
                    "frame_edge_prevents_safe_alignment"
                    if position_only is None
                    else f"{alignment_subject}_bbox_near_crop_edge"
                ),
                required_zoom,
            )
        left, top, _width, _height = crop
        keyframes.append(
            PanCropKeyframe(
                time_seconds,
                NormalizedCrop(left, top, crop_width, crop_height),
            )
        )

    return FaceMatchSegment(
        segment_index=choice.segment_index,
        source_index=choice.source_index,
        applied=True,
        fallback_reason=None,
        observed_face_height=observed if alignment_subject == "face" else None,
        target_face_height=(
            alignment_target if alignment_subject == "face" else None
        ),
        required_zoom=required_zoom,
        applied_zoom=applied_zoom,
        pan_crop=PanCropTrack(tuple(keyframes)),
        alignment_subject=alignment_subject,
        observed_silhouette_height=(
            observed if alignment_subject == "silhouette" else None
        ),
        target_silhouette_height=(
            alignment_target if alignment_subject == "silhouette" else None
        ),
    )


def build_face_match_plan(
    *,
    visual_timeline: VisualTimelinePlan,
    source_aspect_ratios: Sequence[float],
    config: FaceMatchConfig | None = None,
) -> FaceMatchPlan:
    """Create fixed-zoom, smoothly panning 16:9 crops for selected shots."""

    if not visual_timeline.choices:
        raise ValueError("visual timeline cannot be empty")
    if not source_aspect_ratios:
        raise ValueError("source aspect ratios cannot be empty")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in source_aspect_ratios
    ):
        raise ValueError("source aspect ratios must be positive and finite")
    if any(
        choice.source_index < 0 or choice.source_index >= len(source_aspect_ratios)
        for choice in visual_timeline.choices
    ):
        raise ValueError("visual choice references an unknown source aspect ratio")

    settings = config or FaceMatchConfig()
    if settings.target_face_height_override is not None:
        target = settings.target_face_height_override
        target_candidate_count = 1
        eligible_face_segment_count = sum(
            _face_alignment_metrics(
                choice,
                source_aspect=float(source_aspect_ratios[choice.source_index]),
                target=target,
                settings=settings,
            )
            is not None
            for choice in visual_timeline.choices
        )
        target_selection_policy = "explicit_override"
    else:
        (
            target,
            target_candidate_count,
            eligible_face_segment_count,
        ) = _optimized_face_target(
            visual_timeline=visual_timeline,
            source_aspect_ratios=source_aspect_ratios,
            settings=settings,
        )
        target_selection_policy = (
            "deterministic_max_safe_coverage_then_scale_zoom"
        )

    (
        target_silhouette,
        silhouette_target_candidate_count,
        eligible_silhouette_segment_count,
    ) = _optimized_silhouette_target(
        visual_timeline=visual_timeline,
        source_aspect_ratios=source_aspect_ratios,
        settings=settings,
    )

    decisions: list[FaceMatchSegment] = []
    for choice in visual_timeline.choices:
        failure = choice.face_match_candidate_failure
        source_aspect = float(source_aspect_ratios[choice.source_index])
        face_geometry_is_safe = (
            failure is None
            and all(sample is not None for sample in choice.face_samples_xywh)
            and target is not None
        )
        silhouette_geometry = _safe_silhouette_geometry(choice, settings)
        face_attempt: FaceMatchSegment | None = None
        if face_geometry_is_safe:
            face_samples = tuple(
                sample
                for sample in choice.face_samples_xywh
                if sample is not None
            )
            face_centers = tuple(
                (sample[0] + sample[2] / 2.0, sample[1] + sample[3] / 2.0)
                for sample in face_samples
            )
            face_attempt = _alignment_attempt(
                choice=choice,
                alignment_subject="face",
                samples=face_samples,
                centers=face_centers,
                alignment_target=target,
                face_target=target,
                silhouette_target=target_silhouette,
                source_aspect=source_aspect,
                target_screen_y=settings.target_screen_y,
                maximum_zoom=settings.maximum_zoom,
                settings=settings,
            )
            if face_attempt.applied:
                decisions.append(face_attempt)
                continue

        silhouette_attempt: FaceMatchSegment | None = None
        if silhouette_geometry is not None and target_silhouette is not None:
            silhouette_samples, silhouette_centers = silhouette_geometry
            silhouette_attempt = _alignment_attempt(
                choice=choice,
                alignment_subject="silhouette",
                samples=silhouette_samples,
                centers=silhouette_centers,
                alignment_target=target_silhouette,
                face_target=target,
                silhouette_target=target_silhouette,
                source_aspect=source_aspect,
                target_screen_y=settings.silhouette_target_screen_y,
                maximum_zoom=min(
                    settings.maximum_zoom,
                    settings.maximum_silhouette_zoom,
                ),
                settings=settings,
            )
            if silhouette_attempt.applied:
                decisions.append(silhouette_attempt)
                continue

        if face_attempt is not None:
            decisions.append(face_attempt)
        elif silhouette_attempt is not None:
            decisions.append(silhouette_attempt)
        else:
            reason = failure or "face_metadata_unavailable"
            if target is None and failure is None:
                reason = "no_safe_face_match_target"
            decisions.append(
                _fallback(
                    segment_index=choice.segment_index,
                    source_index=choice.source_index,
                    reason=reason,
                    target=target,
                )
            )

    return FaceMatchPlan(
        segments=tuple(decisions),
        target_face_height=target,
        maximum_zoom=settings.maximum_zoom,
        target_screen_position=(
            settings.target_screen_x,
            settings.target_screen_y,
        ),
        target_silhouette_height=target_silhouette,
        target_selection_policy=target_selection_policy,
        target_candidate_count=target_candidate_count,
        eligible_face_segment_count=eligible_face_segment_count,
        silhouette_target_candidate_count=silhouette_target_candidate_count,
        eligible_silhouette_segment_count=eligible_silhouette_segment_count,
    )


__all__ = [
    "FaceMatchConfig",
    "FaceMatchPlan",
    "FaceMatchSegment",
    "build_face_match_plan",
]
