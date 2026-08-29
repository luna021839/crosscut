"""Conservative virtual-camera crops for basic subject-matched cross-edit.

This module never identifies a person. A dominant face is preferred; when a
face is absent or ambiguous, a stable high-confidence aggregate human
silhouette may provide a smaller group-formation crop. Unsafe measurements
produce an explicit raw-frame fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Sequence

import numpy as np

from .render import (
    NormalizedCrop,
    PanCropKeyframe,
    PanCropTrack,
    RenderSegment,
    _tier_d_base_crop_is_frame_continuous,
)
from .section_director import SectionPlan, SectionRole
from .timeline import ShotType
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
    maximum_partial_face_center_span: float = 0.16
    maximum_partial_face_scale_ratio: float = 1.45
    minimum_nonwide_rescue_face_height: float = 0.096096

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
            self.maximum_partial_face_center_span,
            self.maximum_partial_face_scale_ratio,
            self.minimum_nonwide_rescue_face_height,
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
        if not 0.0 < self.maximum_partial_face_center_span <= 0.28:
            raise ValueError("partial face center span must be in (0, 0.28]")
        if not 1.0 <= self.maximum_partial_face_scale_ratio <= 1.8:
            raise ValueError("partial face scale ratio must be in [1.0, 1.8]")
        if not 0.09 <= self.minimum_nonwide_rescue_face_height <= 0.14:
            raise ValueError("nonwide rescue face height is outside its safe range")
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
    continuous_run_stitched: bool = False
    alignment_purpose: str | None = None

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
            "continuous_run_stitched": self.continuous_run_stitched,
            "alignment_purpose": self.alignment_purpose,
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
    analysis_basis: str = (
        "role_conditioned_nonwide_rescue_and_run_stitched_virtual_camera_v8"
    )

    @property
    def applied_count(self) -> int:
        return sum(item.applied for item in self.segments)

    @property
    def fallback_count(self) -> int:
        return len(self.segments) - self.applied_count

    @property
    def silhouette_applied_count(self) -> int:
        return sum(
            item.applied and item.alignment_subject == "silhouette"
            for item in self.segments
        )

    @property
    def aligned_segment_count(self) -> int:
        return sum(
            item.transition_tier == "moderate_confidence_aligned_hard_cut"
            for item in self.segments
        )

    @property
    def continuous_run_stitched_segment_count(self) -> int:
        return sum(item.continuous_run_stitched for item in self.segments)

    @property
    def nonwide_rescue_applied_count(self) -> int:
        return sum(
            item.applied
            and item.alignment_purpose
            == "chorus_single_subject_nonwide_rescue"
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
            "policy": (
                "face_then_stable_silhouette_static_or_pan_crop; "
                "chorus_single_subject_nonwide_rescue; "
                "formation_wides_preserved; one_sided_cross_source_edges "
                "remain_tier_c_hard_cuts"
            ),
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
            "continuous_run_stitched_segment_count": (
                self.continuous_run_stitched_segment_count
            ),
            "nonwide_rescue_applied_count": (
                self.nonwide_rescue_applied_count
            ),
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


@dataclass(frozen=True, slots=True)
class _SampledGeometry:
    """Two or three trustworthy samples with their segment-relative times."""

    samples: tuple[tuple[float, float, float, float], ...]
    centers: tuple[tuple[float, float], ...]
    time_fractions: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (
            len(self.samples)
            == len(self.centers)
            == len(self.time_fractions)
        ):
            raise ValueError("sampled geometry lengths must match")
        if len(self.samples) not in (2, 3):
            raise ValueError("sampled geometry needs two or three observations")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.time_fractions
        ):
            raise ValueError("sampled geometry times must be normalized")
        if any(
            left >= right
            for left, right in zip(
                self.time_fractions,
                self.time_fractions[1:],
            )
        ):
            raise ValueError("sampled geometry times must be strictly increasing")
        if (
            abs(self.time_fractions[0]) > 1e-9
            or abs(self.time_fractions[-1] - 1.0) > 1e-9
        ):
            raise ValueError("sampled geometry must prove both segment boundaries")


def _valid_normalized_box(
    sample: tuple[float, float, float, float],
) -> bool:
    if len(sample) != 4 or any(
        not math.isfinite(float(value)) for value in sample
    ):
        return False
    x, y, width, height = (float(value) for value in sample)
    return (
        x >= 0.0
        and y >= 0.0
        and width > 0.0
        and height > 0.0
        and x + width <= 1.000001
        and y + height <= 1.000001
    )


def _maximum_center_span(centers: Sequence[tuple[float, float]]) -> float:
    return max(
        math.hypot(left[0] - right[0], left[1] - right[1])
        for left in centers
        for right in centers
    )


def _safe_face_geometry(
    choice: Any,
    settings: FaceMatchConfig,
) -> _SampledGeometry | None:
    """Recover only a tightly stable two-of-three dominant-face observation.

    The full three-sample decision remains authoritative.  Partial recovery is
    limited to the two failure modes produced by one missing/ambiguous detector
    observation, and it never crosses an internal source-camera cut.
    """

    failure = choice.face_match_candidate_failure
    if failure == "internal_camera_cut":
        return None
    raw_samples = tuple(choice.face_samples_xywh)
    available = tuple(
        (index, sample)
        for index, sample in enumerate(raw_samples)
        if sample is not None and _valid_normalized_box(sample)
    )
    if len(available) == 3:
        if failure is not None:
            return None
    elif len(available) == 2:
        if failure not in ("face_not_detected", "ambiguous_multiple_faces"):
            return None
        # A crop may interpolate across a missing midpoint, but it must never
        # invent its entry or exit framing: those are the visible edit edges.
        if tuple(index for index, _sample in available) != (0, 2):
            return None
    else:
        return None

    samples = tuple(sample for _index, sample in available)
    heights = tuple(float(sample[3]) for sample in samples)
    if min(heights) < 0.035:
        return None
    centers = tuple(
        (sample[0] + sample[2] / 2.0, sample[1] + sample[3] / 2.0)
        for sample in samples
    )
    if len(samples) == 2 and (
        _maximum_center_span(centers)
        > settings.maximum_partial_face_center_span
        or max(heights) / min(heights)
        > settings.maximum_partial_face_scale_ratio
    ):
        return None
    return _SampledGeometry(
        samples=samples,
        centers=centers,
        time_fractions=tuple(index / 2.0 for index, _sample in available),
    )


def _safe_silhouette_geometry(
    choice: Any,
    settings: FaceMatchConfig,
) -> _SampledGeometry | None:
    """Return a stable two/three-sample aggregate-human track or no signal."""

    if choice.face_match_candidate_failure == "internal_camera_cut":
        return None
    available = tuple(
        (index, sample, center)
        for index, (sample, center, backend, confidence) in enumerate(
            zip(
                choice.silhouette_samples_xywh,
                choice.silhouette_centroid_samples_xy,
                choice.silhouette_backends,
                choice.silhouette_confidences,
                strict=True,
            )
        )
        if (
            sample is not None
            and center is not None
            and _valid_normalized_box(sample)
            and all(math.isfinite(float(value)) for value in center)
            and backend == "opencv_zoo_pphumanseg_2023mar"
            and math.isfinite(float(confidence))
            and float(confidence) >= settings.minimum_silhouette_confidence
        )
    )
    if len(available) not in (2, 3):
        return None
    if (
        len(available) == 2
        and tuple(index for index, _sample, _center in available) != (0, 2)
    ):
        return None
    samples = tuple(sample for _index, sample, _center in available)
    centers = tuple(center for _index, _sample, center in available)
    heights = tuple(sample[3] for sample in samples)
    if min(heights) < settings.minimum_silhouette_height:
        return None
    if max(heights) / min(heights) > settings.maximum_silhouette_scale_ratio:
        return None
    if _maximum_center_span(centers) > settings.maximum_silhouette_center_span:
        return None
    return _SampledGeometry(
        samples=samples,
        centers=centers,
        time_fractions=tuple(index / 2.0 for index, _sample, _center in available),
    )


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

    if len(samples) not in (2, 3) or len(samples) != len(centers):
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

    geometry = _safe_face_geometry(choice, settings)
    if geometry is None:
        return None
    return _alignment_metrics(
        samples=geometry.samples,
        centers=geometry.centers,
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
    return _alignment_metrics(
        samples=geometry.samples,
        centers=geometry.centers,
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
        geometry = _safe_face_geometry(choice, settings)
        if (
            geometry is None
            or abs(source_aspect - 16.0 / 9.0)
            > settings.maximum_source_aspect_deviation
        ):
            continue
        heights = tuple(float(sample[3]) for sample in geometry.samples)
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
        eligible_heights.append(
            float(np.median([sample[3] for sample in geometry.samples]))
        )
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
    sample_time_fractions: Sequence[float] | None = None,
    alignment_purpose: str | None = None,
) -> FaceMatchSegment:
    """Build one face/silhouette crop or an explicit raw-frame attempt result."""

    if len(samples) not in (2, 3) or len(samples) != len(centers):
        raise ValueError("alignment attempt needs two or three matched samples")
    if sample_time_fractions is None:
        if len(samples) != 3:
            raise ValueError("partial alignment samples need explicit times")
        time_fractions = (0.0, 0.5, 1.0)
    else:
        time_fractions = tuple(float(value) for value in sample_time_fractions)
    if (
        len(time_fractions) != len(samples)
        or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in time_fractions
        )
        or any(
            left >= right
            for left, right in zip(time_fractions, time_fractions[1:])
        )
        or abs(time_fractions[0]) > 1e-9
        or abs(time_fractions[-1] - 1.0) > 1e-9
    ):
        raise ValueError(
            "alignment sample times must be ordered and prove both boundaries"
        )
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

    keyframes: list[PanCropKeyframe] = []
    for time_fraction, sample, center in zip(
        time_fractions,
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
                time_fraction * choice.duration_seconds,
                NormalizedCrop(left, top, crop_width, crop_height),
            )
        )

    # Identical translations are a true static crop, not three redundant pan
    # points. This also makes the static/pan distinction explicit in diagnostics.
    first_crop = keyframes[0].crop
    if all(
        math.hypot(
            keyframe.crop.x - first_crop.x,
            keyframe.crop.y - first_crop.y,
        )
        <= 1e-7
        for keyframe in keyframes[1:]
    ):
        keyframes = [PanCropKeyframe(0.0, first_crop)]

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
        alignment_purpose=alignment_purpose,
    )


def _fixed_crop_at(
    track: PanCropTrack,
    time_seconds: float,
) -> NormalizedCrop:
    """Evaluate the renderer's fixed-size linear pan-crop path."""

    width, height = track.fixed_crop_size
    positions = track.fixed_crop_positions()
    if time_seconds <= positions[0][0] + 1e-9:
        return NormalizedCrop(positions[0][1], positions[0][2], width, height)
    for left, right in zip(positions, positions[1:]):
        if time_seconds <= right[0] + 1e-9:
            span = right[0] - left[0]
            alpha = 1.0 if span <= 1e-9 else (time_seconds - left[0]) / span
            return NormalizedCrop(
                left[1] + (right[1] - left[1]) * alpha,
                left[2] + (right[2] - left[2]) * alpha,
                width,
                height,
            )
    return NormalizedCrop(positions[-1][1], positions[-1][2], width, height)


def _covering_crop(
    rectangles: Sequence[NormalizedCrop],
    *,
    width: float,
    height: float,
) -> NormalizedCrop | None:
    """Place one crop that contains every already-proven safe rectangle."""

    if not rectangles:
        return None
    union_left = min(item.x for item in rectangles)
    union_top = min(item.y for item in rectangles)
    union_right = max(item.x + item.width for item in rectangles)
    union_bottom = max(item.y + item.height for item in rectangles)
    if (
        union_right - union_left > width + 1e-9
        or union_bottom - union_top > height + 1e-9
    ):
        return None
    minimum_left = max(0.0, union_right - width)
    maximum_left = min(union_left, 1.0 - width)
    minimum_top = max(0.0, union_bottom - height)
    maximum_top = min(union_top, 1.0 - height)
    if minimum_left > maximum_left + 1e-9 or minimum_top > maximum_top + 1e-9:
        return None
    desired_left = (union_left + union_right - width) / 2.0
    desired_top = (union_top + union_bottom - height) / 2.0
    return NormalizedCrop(
        min(max(desired_left, minimum_left), maximum_left),
        min(max(desired_top, minimum_top), maximum_top),
        width,
        height,
    )


def _stitch_continuous_applied_run(
    decisions: Sequence[FaceMatchSegment],
    choices: Sequence[Any],
    frame_counts: Sequence[int],
    *,
    fps: float,
) -> tuple[FaceMatchSegment, ...] | None:
    """Join an all-safe same-shot run without a hidden crop reset.

    Each original crop has already passed subject containment. The stitched
    run uses one fixed crop size large enough to contain both independently
    proven endpoint rectangles at every invisible split, then gives adjacent
    segments the exact same boundary rectangle. This preserves the safety
    evidence while avoiding the former all-or-nothing demotion.
    """

    if (
        not decisions
        or len(decisions) != len(choices)
        or len(decisions) != len(frame_counts)
        or any(not item.applied or item.pan_crop is None for item in decisions)
        or len({item.alignment_subject for item in decisions}) != 1
    ):
        return None
    typed_tracks = tuple(
        item.pan_crop for item in decisions if item.pan_crop is not None
    )
    if len(typed_tracks) != len(decisions):  # pragma: no cover - precondition
        return None
    ratios = tuple(
        track.fixed_crop_size[0] / track.fixed_crop_size[1]
        for track in typed_tracks
    )
    if max(ratios) - min(ratios) > 0.002:
        return None
    ratio = sum(ratios) / len(ratios)

    constraints: list[dict[float, list[NormalizedCrop]]] = []
    for choice, track in zip(choices, typed_tracks, strict=True):
        duration = float(choice.duration_seconds)
        times = {0.0, duration}
        times.update(
            min(duration, max(0.0, float(item.time_seconds)))
            for item in track.keyframes
        )
        constraints.append(
            {
                time_value: [_fixed_crop_at(track, time_value)]
                for time_value in sorted(times)
            }
        )
    for position in range(len(decisions) - 1):
        left_duration = float(choices[position].duration_seconds)
        left_crop = _fixed_crop_at(typed_tracks[position], left_duration)
        right_crop = _fixed_crop_at(typed_tracks[position + 1], 0.0)
        shared = [left_crop, right_crop]
        constraints[position][left_duration] = shared
        constraints[position + 1][0.0] = shared

    required_height = 0.0
    for by_time in constraints:
        for rectangles in by_time.values():
            union_width = max(
                item.x + item.width for item in rectangles
            ) - min(item.x for item in rectangles)
            union_height = max(
                item.y + item.height for item in rectangles
            ) - min(item.y for item in rectangles)
            required_height = max(
                required_height,
                union_height,
                union_width / ratio,
            )
    maximum_height = min(1.0, 1.0 / ratio)
    if required_height > maximum_height + 1e-9:
        return None
    common_height = min(maximum_height, required_height)
    common_width = ratio * common_height
    applied_zoom = 1.0 / common_height
    if applied_zoom < 1.025 - 1e-9:
        return None

    stitched: list[FaceMatchSegment] = []
    for decision, by_time in zip(decisions, constraints, strict=True):
        keyframes: list[PanCropKeyframe] = []
        for time_value, rectangles in sorted(by_time.items()):
            crop = _covering_crop(
                rectangles,
                width=common_width,
                height=common_height,
            )
            if crop is None:
                return None
            keyframes.append(PanCropKeyframe(time_value, crop))
        first_crop = keyframes[0].crop
        if all(
            max(
                abs(item.crop.x - first_crop.x),
                abs(item.crop.y - first_crop.y),
            )
            <= 1e-7
            for item in keyframes[1:]
        ):
            keyframes = [PanCropKeyframe(0.0, first_crop)]
        stitched.append(
            replace(
                decision,
                applied_zoom=applied_zoom,
                pan_crop=PanCropTrack(tuple(keyframes)),
                continuous_run_stitched=True,
            )
        )
    def render_safe(items: Sequence[FaceMatchSegment]) -> bool:
        for position, (left, right) in enumerate(zip(items, items[1:])):
            assert left.pan_crop is not None and right.pan_crop is not None
            left_choice = choices[position]
            right_choice = choices[position + 1]
            left_segment = RenderSegment(
                source_index=int(left_choice.source_index),
                source_start=0.0,
                duration=float(left_choice.duration_seconds),
                master_start=0.0,
                pan_crop=left.pan_crop,
            )
            right_segment = RenderSegment(
                source_index=int(right_choice.source_index),
                source_start=0.0,
                duration=float(right_choice.duration_seconds),
                master_start=float(left_choice.duration_seconds),
                pan_crop=right.pan_crop,
            )
            if not _tier_d_base_crop_is_frame_continuous(
                left_segment,
                right_segment,
                left_frame_count=int(frame_counts[position]),
                right_frame_count=int(frame_counts[position + 1]),
                fps=fps,
            ):
                return False
        return True

    for position, (left, right) in enumerate(zip(stitched, stitched[1:])):
        assert left.pan_crop is not None and right.pan_crop is not None
        left_crop = _fixed_crop_at(
            left.pan_crop,
            float(choices[position].duration_seconds),
        )
        right_crop = _fixed_crop_at(right.pan_crop, 0.0)
        if max(
            abs(left_value - right_value)
            for left_value, right_value in zip(
                (left_crop.x, left_crop.y, left_crop.width, left_crop.height),
                (right_crop.x, right_crop.y, right_crop.width, right_crop.height),
                strict=True,
            )
        ) > 1e-8:
            return None
    if render_safe(stitched):
        return tuple(stitched)

    # A sparse tracker can imply a fast pan during the fractional final-frame
    # interval even though both mathematical endpoints meet. Prefer one static
    # crop that contains every already-proven safe rectangle in the same
    # source-local shot. This keeps the scale match without introducing a
    # hidden one-frame whip. If the union would erase meaningful zoom, the
    # caller demotes the whole invisible run to raw frames instead.
    all_rectangles = tuple(
        rectangle
        for by_time in constraints
        for rectangles in by_time.values()
        for rectangle in rectangles
    )
    union_width = max(
        item.x + item.width for item in all_rectangles
    ) - min(item.x for item in all_rectangles)
    union_height = max(
        item.y + item.height for item in all_rectangles
    ) - min(item.y for item in all_rectangles)
    static_height = max(union_height, union_width / ratio)
    if static_height > maximum_height + 1e-9:
        return None
    static_width = ratio * static_height
    static_zoom = 1.0 / static_height
    if static_zoom < 1.025 - 1e-9:
        return None
    static_crop = _covering_crop(
        all_rectangles,
        width=static_width,
        height=static_height,
    )
    if static_crop is None:
        return None
    static_stitched = tuple(
        replace(
            decision,
            applied_zoom=static_zoom,
            pan_crop=PanCropTrack((PanCropKeyframe(0.0, static_crop),)),
            continuous_run_stitched=True,
        )
        for decision in decisions
    )
    if not render_safe(static_stitched):  # pragma: no cover - exact static proof
        return None
    return static_stitched


def build_face_match_plan(
    *,
    visual_timeline: VisualTimelinePlan,
    source_aspect_ratios: Sequence[float],
    config: FaceMatchConfig | None = None,
    section_plan: SectionPlan | None = None,
    fps: float = 29.97,
) -> FaceMatchPlan:
    """Create fixed-zoom, smoothly panning 16:9 crops for selected shots.

    When a section plan is supplied, a chorus-like single-subject wide shot
    may use a smaller secondary target that merely crosses the decoded
    non-wide face threshold. Genuine multi-person formation wides remain raw.
    """

    if not visual_timeline.choices:
        raise ValueError("visual timeline cannot be empty")
    if not source_aspect_ratios:
        raise ValueError("source aspect ratios cannot be empty")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("face-match fps must be positive and finite")
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
    if section_plan is not None and (
        section_plan.start_seconds
        > visual_timeline.choices[0].canonical_master_start + 1e-6
        or section_plan.end_seconds
        < visual_timeline.choices[-1].canonical_master_end - 1e-6
    ):
        raise ValueError("section plan does not cover the visual timeline")

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
        face_geometry = _safe_face_geometry(choice, settings)
        silhouette_geometry = _safe_silhouette_geometry(choice, settings)
        face_attempt: FaceMatchSegment | None = None
        if face_geometry is not None and target is not None:
            face_attempt = _alignment_attempt(
                choice=choice,
                alignment_subject="face",
                samples=face_geometry.samples,
                centers=face_geometry.centers,
                alignment_target=target,
                face_target=target,
                silhouette_target=target_silhouette,
                source_aspect=source_aspect,
                target_screen_y=settings.target_screen_y,
                maximum_zoom=settings.maximum_zoom,
                settings=settings,
                sample_time_fractions=face_geometry.time_fractions,
                alignment_purpose="global_face_scale_match",
            )
            if face_attempt.applied:
                decisions.append(face_attempt)
                continue

        nonwide_rescue: FaceMatchSegment | None = None
        shot_types = (
            choice.entry_shot_type or choice.shot_type,
            choice.shot_type,
            choice.exit_shot_type or choice.shot_type,
        )
        directive = (
            section_plan.directive_at(choice.canonical_master_start)
            if section_plan is not None
            else None
        )
        if (
            face_geometry is not None
            and directive is not None
            and directive.role is SectionRole.CHORUS_LIKE
            and sum(
                value in {ShotType.GROUP_WIDE, ShotType.WIDE}
                for value in shot_types
            )
            >= 2
            and max(
                (*choice.detected_person_counts, *choice.formation_person_counts),
                default=0,
            )
            < 3
        ):
            nonwide_rescue = _alignment_attempt(
                choice=choice,
                alignment_subject="face",
                samples=face_geometry.samples,
                centers=face_geometry.centers,
                alignment_target=settings.minimum_nonwide_rescue_face_height,
                face_target=settings.minimum_nonwide_rescue_face_height,
                silhouette_target=target_silhouette,
                source_aspect=source_aspect,
                target_screen_y=settings.target_screen_y,
                maximum_zoom=settings.maximum_zoom,
                settings=settings,
                sample_time_fractions=face_geometry.time_fractions,
                alignment_purpose="chorus_single_subject_nonwide_rescue",
            )
            if nonwide_rescue.applied:
                decisions.append(nonwide_rescue)
                continue

        silhouette_attempt: FaceMatchSegment | None = None
        if silhouette_geometry is not None and target_silhouette is not None:
            silhouette_attempt = _alignment_attempt(
                choice=choice,
                alignment_subject="silhouette",
                samples=silhouette_geometry.samples,
                centers=silhouette_geometry.centers,
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
                sample_time_fractions=silhouette_geometry.time_fractions,
                alignment_purpose="global_silhouette_scale_match",
            )
            if silhouette_attempt.applied:
                decisions.append(silhouette_attempt)
                continue

        if nonwide_rescue is not None:
            decisions.append(nonwide_rescue)
        elif face_attempt is not None:
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

    # Crop compatibility is a boundary property, not a timeline-wide state.
    # Cross-source one-sided crop/raw edges remain ordinary Tier-C hard cuts in
    # ``group_transition``; erasing their safe crop side would turn one detector
    # miss into an all-raw fixed-point cascade. A mixed face/silhouette pair must
    # not be advertised as aligned, though, so retain the stronger face crop and
    # locally demote the silhouette side. Only a same-source/same-shot boundary
    # is invisible and therefore requires exact local crop continuity.
    choices = visual_timeline.choices
    frame_counts: list[int] = []
    cumulative_duration = 0.0
    previous_boundary = 0
    for choice in choices:
        cumulative_duration += float(choice.duration_seconds)
        boundary = int(round(cumulative_duration * fps))
        frame_count = boundary - previous_boundary
        if frame_count < 2:
            raise ValueError("face-match segment is shorter than two output frames")
        frame_counts.append(frame_count)
        previous_boundary = boundary
    mixed_subject_demotions: set[int] = set()
    for position, (left_choice, right_choice) in enumerate(
        zip(choices, choices[1:])
    ):
        if left_choice.source_index == right_choice.source_index:
            continue
        left = decisions[position]
        right = decisions[position + 1]
        if (
            left.applied
            and right.applied
            and left.alignment_subject != right.alignment_subject
        ):
            if left.alignment_subject == "silhouette":
                mixed_subject_demotions.add(position)
            if right.alignment_subject == "silhouette":
                mixed_subject_demotions.add(position + 1)
    for position in mixed_subject_demotions:
        item = decisions[position]
        decisions[position] = replace(
            item,
            applied=False,
            fallback_reason="cross_source_alignment_subject_mismatch",
            applied_zoom=None,
            pan_crop=None,
        )

    def same_continuous_shot(left_choice: Any, right_choice: Any) -> bool:
        if left_choice.source_index != right_choice.source_index:
            return False
        return (
            not left_choice.shot_id
            and not right_choice.shot_id
        ) or (
            bool(left_choice.shot_id)
            and left_choice.shot_id == right_choice.shot_id
        )

    def crop_endpoint(
        item: FaceMatchSegment,
        *,
        final: bool,
    ) -> tuple[float, float, float, float] | None:
        if not item.applied or item.pan_crop is None:
            return None
        width, height = item.pan_crop.fixed_crop_size
        positions = item.pan_crop.fixed_crop_positions()
        _time, x, y = positions[-1] if final else positions[0]
        return (x, y, width, height)

    def invisible_edge_is_compatible(
        left: FaceMatchSegment,
        right: FaceMatchSegment,
    ) -> bool:
        if not left.applied and not right.applied:
            return True
        if (
            not left.applied
            or not right.applied
            or left.alignment_subject != right.alignment_subject
        ):
            return False
        left_crop = crop_endpoint(left, final=True)
        right_crop = crop_endpoint(right, final=False)
        if left_crop is None or right_crop is None:
            return False
        # Tier D is one uninterrupted source-local shot. Even a sub-percent
        # independent endpoint change is stitched below instead of being left
        # for frame quantization to expose as a one-frame punch or pan reset.
        return max(
            abs(left_value - right_value)
            for left_value, right_value in zip(left_crop, right_crop, strict=True)
        ) <= 1e-8

    run_start = 0
    while run_start < len(choices):
        run_end = run_start
        while run_end + 1 < len(choices) and same_continuous_shot(
            choices[run_end],
            choices[run_end + 1],
        ):
            run_end += 1
        if run_end > run_start and any(
            not invisible_edge_is_compatible(
                decisions[position],
                decisions[position + 1],
            )
            for position in range(run_start, run_end)
        ):
            stitched = _stitch_continuous_applied_run(
                decisions[run_start : run_end + 1],
                choices[run_start : run_end + 1],
                frame_counts[run_start : run_end + 1],
                fps=fps,
            )
            if stitched is not None:
                decisions[run_start : run_end + 1] = stitched
            else:
                for position in range(run_start, run_end + 1):
                    item = decisions[position]
                    if item.applied:
                        decisions[position] = replace(
                            item,
                            applied=False,
                            fallback_reason=(
                                "continuous_shot_crop_pair_incompatible"
                            ),
                            applied_zoom=None,
                            pan_crop=None,
                        )
        run_start = run_end + 1

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
