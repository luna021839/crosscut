"""Render and pixel-review a low-resolution copy of a finished edit plan.

Timeline scores cannot catch decoder, scaler, crop, encoder, or source-quality
failures that only exist in the actual output.  This module derives a fixed
480p ``RenderPlan``, reuses the production renderer, then sequentially decodes
that MP4.  It retains only scalar quality/continuity measurements and
normalized subject geometry; frames and appearance embeddings are never
serialized or written as review artifacts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from .ive import AspectRatio
from .render import (
    RenderPlan,
    TransitionKind,
    render,
    segment_frame_counts,
)

if TYPE_CHECKING:
    from .professional_reference_gate import ProfessionalReferenceProfile


ANALYSIS_BASIS = "decoded_output_proxy_domain_aware_quality_review_v3"
PERCEPTUAL_SOURCE_ADMISSION_ANALYSIS_BASIS = (
    "decoded_480p_proxy_source_median_and_rapid_run_quality_admission_v1"
)
HUMAN_REFERENCE_BOUNDARY_ANALYSIS_BASIS = (
    "human_reference_hard_cut_boundary_scalar_evidence_v2"
)
PROXY_LANDSCAPE_SIZE = (854, 480)
PROXY_PORTRAIT_SIZE = (270, 480)
# Hard cuts compare the last two outgoing frames with the first two incoming
# frames. Dissolves use a duration-aware schedule built from their frame count.
BOUNDARY_FRAME_OFFSETS = (-2, -1, 0, 1)

INTRODUCED_CROSS_SOURCE_DOMAIN = "introduced_cross_source"
TIER_D_CONTINUITY_DOMAIN = "tier_d_continuity"
NATIVE_SOURCE_CUT_DOMAIN = "native_source_cut"
BOUNDARY_REVIEW_DOMAINS = (
    INTRODUCED_CROSS_SOURCE_DOMAIN,
    TIER_D_CONTINUITY_DOMAIN,
    NATIVE_SOURCE_CUT_DOMAIN,
)

# These are evidence-reliability and technical-integrity bounds, not relaxed
# editorial thresholds.  A direction is meaningful only when coherent flow
# dominates local flow noise.  Tier-D/native checks use temporal neighbors to
# detect a concat freeze or a dropped/reset frame without treating an intended
# shot change as a failed cross-source edit.
MOTION_MINIMUM_COHERENT_PIXEL_RATIO = 0.60
MOTION_MINIMUM_SIGNAL_TO_NOISE = 1.0
TEMPORAL_DUPLICATE_MINIMUM_CONTEXT_DELTA = 0.010
TEMPORAL_DUPLICATE_MAXIMUM_CROSS_DELTA = 0.0015
TEMPORAL_DUPLICATE_MAXIMUM_CONTEXT_RATIO = 0.12
TIER_D_RESET_MINIMUM_CROSS_DELTA = 0.12
TIER_D_RESET_MINIMUM_CONTEXT_RATIO = 3.0
BOUNDARY_BLACK_MAXIMUM_MEAN_LUMA = 0.012
BOUNDARY_BLACK_MAXIMUM_CONTRAST = 0.012


def _unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _finite(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    alpha = position - low
    return ordered[low] * (1.0 - alpha) + ordered[high] * alpha


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    clean = tuple(float(value) for value in values if math.isfinite(float(value)))
    return {
        "count": len(clean),
        "minimum": min(clean) if clean else None,
        "p10": _quantile(clean, 0.10),
        "median": median(clean) if clean else None,
        "p90": _quantile(clean, 0.90),
        "maximum": max(clean) if clean else None,
    }


@dataclass(frozen=True, slots=True)
class ProxyReviewConfig:
    minimum_overall_quality_p10: float = 0.36
    maximum_source_quality_median_span: float = 0.22
    minimum_boundary_score_p10: float = 58.0
    minimum_weighted_score: float = 60.0
    maximum_boundary_issue_fraction: float = 0.35
    maximum_sharpness_discontinuity: float = 0.48
    maximum_exposure_discontinuity: float = 0.24
    maximum_color_shock: float = 0.48
    maximum_structure_shock: float = 0.62
    maximum_face_scale_discontinuity: float = 0.40
    maximum_face_position_discontinuity: float = 0.25
    maximum_silhouette_scale_discontinuity: float = 0.42
    maximum_silhouette_position_discontinuity: float = 0.27
    maximum_motion_direction_discontinuity: float = 0.78
    # These measurements intentionally mirror the conditional (PASS-boundary)
    # thresholds in the final perceptual-quality gate.  Candidate proxies must
    # therefore demonstrate a plausible route to final PASS instead of merely
    # relying on the final render to reject source-to-source pumping.
    perceptual_analysis_width: int = 256
    minimum_source_look_output_frames: int = 12
    maximum_source_sharpness_log_median_span: float = 0.55
    maximum_source_luma_median_span: float = 0.14
    maximum_source_chroma_median_span: float = 0.12
    rapid_run_maximum_seconds: float = 3.0
    minimum_rapid_source_switch_boundary_count: int = 4
    rapid_sharpness_log_step: float = 0.45
    rapid_luma_step: float = 0.065
    rapid_chroma_step: float = 0.055
    maximum_rapid_high_quality_shock_fraction: float = 0.25
    # A completed dissolve spans multiple canonical frames, so its endpoint
    # comparison includes a small amount of natural performance motion. Permit
    # at most 20% extra endpoint severity; the hard upper bound is 25%.
    dissolve_endpoint_tolerance_multiplier: float = 1.20

    def __post_init__(self) -> None:
        unit_fields = (
            "minimum_overall_quality_p10",
            "maximum_source_quality_median_span",
            "maximum_boundary_issue_fraction",
            "maximum_sharpness_discontinuity",
            "maximum_exposure_discontinuity",
            "maximum_color_shock",
            "maximum_structure_shock",
            "maximum_face_scale_discontinuity",
            "maximum_face_position_discontinuity",
            "maximum_silhouette_scale_discontinuity",
            "maximum_silhouette_position_discontinuity",
            "maximum_motion_direction_discontinuity",
            "maximum_source_luma_median_span",
            "maximum_source_chroma_median_span",
            "maximum_rapid_high_quality_shock_fraction",
        )
        for name in unit_fields:
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be normalized")
        for name in ("minimum_boundary_score_p10", "minimum_weighted_score"):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"{name} must be in [0, 100]")
        multiplier = _finite(
            self.dissolve_endpoint_tolerance_multiplier,
            "dissolve_endpoint_tolerance_multiplier",
        )
        if not 1.0 <= multiplier <= 1.25:
            raise ValueError(
                "dissolve endpoint tolerance multiplier must be in [1, 1.25]"
            )
        if (
            isinstance(self.perceptual_analysis_width, bool)
            or not isinstance(self.perceptual_analysis_width, int)
            or self.perceptual_analysis_width != 256
        ):
            raise ValueError("perceptual_analysis_width must remain fixed at 256")
        for name in (
            "minimum_source_look_output_frames",
            "minimum_rapid_source_switch_boundary_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "maximum_source_sharpness_log_median_span",
            "maximum_source_luma_median_span",
            "maximum_source_chroma_median_span",
            "rapid_run_maximum_seconds",
            "rapid_sharpness_log_step",
            "rapid_luma_step",
            "rapid_chroma_step",
            "maximum_rapid_high_quality_shock_fraction",
        ):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "analysis_basis": ANALYSIS_BASIS,
            "boundary_domain_policy": (
                "introduced_cross_source_editorial_gate_plus_"
                "tier_d_temporal_integrity_plus_native_cut_passthrough"
            ),
            "visible_transition_denominator": INTRODUCED_CROSS_SOURCE_DOMAIN,
            "one_sided_geometry_policy": "not_comparable_with_subject_fallback",
            "unavailable_motion_policy": "not_comparable_never_maximum_mismatch",
            "dissolve_endpoint_policy": (
                "pre_transition_two_frames_vs_completed_post_transition_"
                "two_frames_with_bounded_threshold_and_severity_relaxation"
            ),
            "perceptual_source_admission_analysis_basis": (
                PERCEPTUAL_SOURCE_ADMISSION_ANALYSIS_BASIS
            ),
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
        }


@dataclass(frozen=True, slots=True)
class OutputSubjectGeometry:
    """Non-identifying geometry extracted from one decoded output frame."""

    face_count: int = 0
    face_bbox_xywh: tuple[float, float, float, float] | None = None
    face_backend: str = "none"
    silhouette_bbox_xywh: tuple[float, float, float, float] | None = None
    silhouette_centroid_xy: tuple[float, float] | None = None
    silhouette_confidence: float = 0.0
    silhouette_backend: str = "none"

    def __post_init__(self) -> None:
        if (
            isinstance(self.face_count, bool)
            or not isinstance(self.face_count, int)
            or self.face_count < 0
        ):
            raise ValueError("face_count must be a non-negative integer")
        for name in ("face_bbox_xywh", "silhouette_bbox_xywh"):
            box = getattr(self, name)
            if box is None:
                continue
            if len(box) != 4:
                raise ValueError(f"{name} needs four normalized values")
            x, y, width, height = tuple(_finite(value, name) for value in box)
            if not (
                0.0 <= x <= 1.0
                and 0.0 <= y <= 1.0
                and 0.0 < width <= 1.0
                and 0.0 < height <= 1.0
                and x + width <= 1.000001
                and y + height <= 1.000001
            ):
                raise ValueError(f"{name} must fit normalized output bounds")
            object.__setattr__(self, name, (x, y, width, height))
        if self.silhouette_centroid_xy is not None:
            centroid = tuple(
                _finite(value, "silhouette centroid")
                for value in self.silhouette_centroid_xy
            )
            if len(centroid) != 2 or any(
                not 0.0 <= value <= 1.0 for value in centroid
            ):
                raise ValueError("silhouette centroid must be normalized")
            object.__setattr__(self, "silhouette_centroid_xy", centroid)
        confidence = _finite(self.silhouette_confidence, "silhouette confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("silhouette confidence must be normalized")
        object.__setattr__(self, "silhouette_confidence", confidence)
        object.__setattr__(self, "face_backend", self.face_backend.strip() or "none")
        object.__setattr__(
            self,
            "silhouette_backend",
            self.silhouette_backend.strip() or "none",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_count": self.face_count,
            "face_bbox_xywh": (
                list(self.face_bbox_xywh)
                if self.face_bbox_xywh is not None
                else None
            ),
            "face_backend": self.face_backend,
            "silhouette_bbox_xywh": (
                list(self.silhouette_bbox_xywh)
                if self.silhouette_bbox_xywh is not None
                else None
            ),
            "silhouette_centroid_xy": (
                list(self.silhouette_centroid_xy)
                if self.silhouette_centroid_xy is not None
                else None
            ),
            "silhouette_confidence": self.silhouette_confidence,
            "silhouette_backend": self.silhouette_backend,
        }


GeometryProvider = Callable[[np.ndarray], OutputSubjectGeometry]


def _default_geometry_provider(frame: np.ndarray) -> OutputSubjectGeometry:
    # Import lazily: the bundled YuNet/person/silhouette models are loaded only
    # for selected boundary frames, never for every proxy frame.
    from .visual_score import describe_frame

    descriptor = describe_frame(
        frame,
        include_subject_geometry=True,
        include_formation_pose=False,
    )
    return OutputSubjectGeometry(
        face_count=descriptor.face_count,
        face_bbox_xywh=descriptor.primary_face_bbox,
        face_backend=descriptor.face_detector_backend,
        silhouette_bbox_xywh=descriptor.silhouette_bbox_xywh,
        silhouette_centroid_xy=descriptor.silhouette_centroid_xy,
        silhouette_confidence=descriptor.silhouette_confidence,
        silhouette_backend=descriptor.silhouette_backend,
    )


@dataclass(frozen=True, slots=True)
class RenderedProxyReview:
    payload: dict[str, Any]

    @property
    def passed(self) -> bool:
        return bool(self.payload["passed"])

    @property
    def weighted_score(self) -> float:
        return float(self.payload["weighted_score"])

    def write(self, path: Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination.resolve()


@dataclass(frozen=True, slots=True)
class _BoundaryReviewWindow:
    boundary_index: int
    cut_frame_index: int
    outgoing_source_index: int
    incoming_source_index: int
    transition_kind: TransitionKind
    transition_frame_count: int
    transition_reason: str
    transition_tier: str | None
    review_domain: str
    source_clock_contiguous: bool
    sample_offsets: tuple[int, int, int, int]

    @property
    def frame_indices(self) -> tuple[int, int, int, int]:
        return tuple(
            self.cut_frame_index + offset for offset in self.sample_offsets
        )

    @property
    def due_frame_index(self) -> int:
        return self.frame_indices[-1]

    def duration_seconds(self, fps: float) -> float:
        return self.transition_frame_count / fps


def _boundary_review_domain(
    *,
    outgoing_source_index: int,
    incoming_source_index: int,
    transition_reason: str,
    transition_tier: str | None,
) -> str:
    """Classify what a serialized segment boundary actually represents."""

    if outgoing_source_index != incoming_source_index:
        # A cross-source boundary remains an editorial decision even when one
        # selected source segment contains a broadcaster camera cut.
        return INTRODUCED_CROSS_SOURCE_DOMAIN
    if (
        transition_tier == "D"
        or transition_reason == "same_source_hold_no_visible_switch"
    ):
        return TIER_D_CONTINUITY_DOMAIN
    return NATIVE_SOURCE_CUT_DOMAIN


def proxy_dimensions(aspect_ratio: AspectRatio) -> tuple[int, int]:
    ratio = AspectRatio(aspect_ratio)
    return (
        PROXY_PORTRAIT_SIZE
        if ratio is AspectRatio.PORTRAIT
        else PROXY_LANDSCAPE_SIZE
    )


def derive_proxy_render_plan(plan: RenderPlan, output: Path) -> RenderPlan:
    """Clone the complete edit grammar at a deterministic 480-pixel height."""

    width, height = proxy_dimensions(plan.aspect_ratio)
    return RenderPlan(
        sources=plan.sources,
        segments=plan.segments,
        reference_source=plan.reference_source,
        aspect_ratio=plan.aspect_ratio,
        output=Path(output),
        master_audio_start=plan.master_audio_start,
        width=width,
        height=height,
        fps=plan.fps,
        transitions=plan.transitions,
    )


def _frame_scalars(frame: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    sharpness = 1.0 - math.exp(-laplacian_variance / 180.0)
    mean = float(gray.mean())
    contrast = float(gray.std())
    clipped = float(np.mean((gray < 8) | (gray > 247)))
    exposure = math.exp(-((mean - 118.0) / 105.0) ** 2) * (
        1.0 - min(0.8, clipped)
    )
    contrast_score = min(1.0, contrast / 58.0)
    quality = _unit(
        0.45 * sharpness + 0.32 * exposure + 0.23 * contrast_score
    )
    return {
        "quality": quality,
        "sharpness": sharpness,
        "laplacian_variance": laplacian_variance,
        "mean_luma": mean / 255.0,
        "exposure": exposure,
        "contrast": contrast / 255.0,
        "clipped_pixel_ratio": clipped,
    }


def _perceptual_frame_scalars(
    frame: np.ndarray,
    *,
    perceptual_analysis_width: int,
) -> dict[str, float]:
    """Return the final-gate-aligned 256-pixel scalar view of one frame."""

    perceptual_width = perceptual_analysis_width
    perceptual_height = max(
        2,
        int(round(perceptual_width * frame.shape[0] / frame.shape[1])),
    )
    perceptual = cv2.resize(
        frame,
        (perceptual_width, perceptual_height),
        interpolation=cv2.INTER_AREA,
    )
    perceptual_gray = cv2.cvtColor(perceptual, cv2.COLOR_BGR2GRAY).astype(
        np.float32
    )
    perceptual_laplacian = cv2.Laplacian(
        perceptual_gray,
        cv2.CV_32F,
        ksize=3,
    )
    perceptual_p05, perceptual_p95 = np.percentile(
        perceptual_gray,
        (5.0, 95.0),
    )
    perceptual_lab = cv2.cvtColor(
        perceptual,
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)
    return {
        "perceptual_detail": float(
            np.mean(np.abs(perceptual_laplacian)) / 255.0
        ),
        "perceptual_luma": float(np.mean(perceptual_gray) / 255.0),
        "perceptual_chroma_a": float(
            (np.mean(perceptual_lab[..., 1]) - 128.0) / 127.0
        ),
        "perceptual_chroma_b": float(
            (np.mean(perceptual_lab[..., 2]) - 128.0) / 127.0
        ),
        "perceptual_contrast": float(
            (perceptual_p95 - perceptual_p05) / 255.0
        ),
    }


def _perceptual_median(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not values:
        raise ValueError("perceptual median needs scalar evidence")
    return {
        name: float(median(float(item[name]) for item in values))
        for name in (
            "perceptual_detail",
            "perceptual_luma",
            "perceptual_chroma_a",
            "perceptual_chroma_b",
            "perceptual_contrast",
        )
    }


def _source_pair_metric(
    left: Mapping[str, float],
    right: Mapping[str, float],
    metric: str,
) -> float:
    if metric == "sharpness_log":
        return abs(
            math.log(
                (float(right["perceptual_detail"]) + 1e-4)
                / (float(left["perceptual_detail"]) + 1e-4)
            )
        )
    if metric == "luma":
        return abs(
            float(right["perceptual_luma"])
            - float(left["perceptual_luma"])
        )
    if metric == "chroma":
        return math.hypot(
            float(right["perceptual_chroma_a"])
            - float(left["perceptual_chroma_a"]),
            float(right["perceptual_chroma_b"])
            - float(left["perceptual_chroma_b"]),
        )
    raise ValueError(f"unknown perceptual source-pair metric: {metric}")


def _proxy_perceptual_source_admission(
    *,
    source_metrics: Mapping[int, Sequence[Mapping[str, float]]],
    run_metrics: Mapping[int, Sequence[Mapping[str, float]]],
    run_source_indices: Sequence[int],
    run_durations_seconds: Sequence[float],
    run_incoming_boundary_indices: Sequence[int | None],
    config: ProxyReviewConfig,
) -> dict[str, Any]:
    """Build scalar-only source consistency and rapid-run admission evidence."""

    if not (
        len(run_source_indices)
        == len(run_durations_seconds)
        == len(run_incoming_boundary_indices)
    ):
        raise ValueError("perceptual run metadata arrays must match")
    used_source_indices = tuple(sorted(source_metrics))
    per_source: list[dict[str, Any]] = []
    valid_source_medians: list[tuple[int, dict[str, float]]] = []
    insufficient_source_indices: list[int] = []
    for source_index in used_source_indices:
        values = tuple(source_metrics[source_index])
        scalar_median = _perceptual_median(values) if values else None
        if len(values) >= config.minimum_source_look_output_frames:
            if scalar_median is None:
                raise ValueError("adequately observed source lacks a scalar median")
            valid_source_medians.append((source_index, scalar_median))
        else:
            insufficient_source_indices.append(source_index)
        per_source.append(
            {
                "source_index": source_index,
                "decoded_output_frame_count": len(values),
                "median_detail": (
                    scalar_median["perceptual_detail"]
                    if scalar_median is not None
                    else None
                ),
                "median_luma": (
                    scalar_median["perceptual_luma"]
                    if scalar_median is not None
                    else None
                ),
                "median_chroma_a": (
                    scalar_median["perceptual_chroma_a"]
                    if scalar_median is not None
                    else None
                ),
                "median_chroma_b": (
                    scalar_median["perceptual_chroma_b"]
                    if scalar_median is not None
                    else None
                ),
                "median_contrast": (
                    scalar_median["perceptual_contrast"]
                    if scalar_median is not None
                    else None
                ),
            }
        )

    metric_limits = {
        "sharpness_log": config.maximum_source_sharpness_log_median_span,
        "luma": config.maximum_source_luma_median_span,
        "chroma": config.maximum_source_chroma_median_span,
    }
    source_spans: dict[str, float | None] = {}
    worst_source_pairs: dict[str, dict[str, Any] | None] = {}
    for metric in metric_limits:
        pairs = tuple(
            (
                _source_pair_metric(left, right, metric),
                left_index,
                right_index,
            )
            for position, (left_index, left) in enumerate(valid_source_medians)
            for right_index, right in valid_source_medians[position + 1 :]
        )
        if pairs:
            value, left_index, right_index = max(
                pairs,
                key=lambda item: (item[0], -item[1], -item[2]),
            )
            source_spans[metric] = value
            worst_source_pairs[metric] = {
                "source_indices": [left_index, right_index],
                "value": value,
                "maximum_allowed": metric_limits[metric],
            }
        elif len(valid_source_medians) == 1:
            source_spans[metric] = 0.0
            only_index = valid_source_medians[0][0]
            worst_source_pairs[metric] = {
                "source_indices": [only_index, only_index],
                "value": 0.0,
                "maximum_allowed": metric_limits[metric],
            }
        else:
            source_spans[metric] = None
            worst_source_pairs[metric] = None

    run_medians: list[dict[str, Any]] = []
    for run_index, (
        source_index,
        duration_seconds,
        incoming_boundary_index,
    ) in enumerate(
        zip(
            run_source_indices,
            run_durations_seconds,
            run_incoming_boundary_indices,
            strict=True,
        )
    ):
        values = tuple(run_metrics.get(run_index, ()))
        scalar_median = _perceptual_median(values) if values else None
        run_medians.append(
            {
                "run_index": run_index,
                "source_index": source_index,
                "duration_seconds": duration_seconds,
                "decoded_output_frame_count": len(values),
                "incoming_boundary_index": incoming_boundary_index,
                "scalars": scalar_median,
            }
        )

    expected_rapid_boundary_count = 0
    rapid_boundaries: list[dict[str, Any]] = []
    missing_rapid_boundary_indices: list[int] = []
    all_source_switch_boundaries: list[dict[str, Any]] = []
    for left, right in zip(run_medians, run_medians[1:]):
        boundary_index = right["incoming_boundary_index"]
        if not isinstance(boundary_index, int):
            raise ValueError("non-initial source run lacks a boundary index")
        if left["source_index"] == right["source_index"]:
            continue
        source_pair = [left["source_index"], right["source_index"]]
        all_source_switch_boundaries.append(
            {
                "boundary_index": boundary_index,
                "source_indices": source_pair,
            }
        )
        if max(left["duration_seconds"], right["duration_seconds"]) > (
            config.rapid_run_maximum_seconds
        ):
            continue
        expected_rapid_boundary_count += 1
        left_scalars = left["scalars"]
        right_scalars = right["scalars"]
        if not isinstance(left_scalars, Mapping) or not isinstance(
            right_scalars,
            Mapping,
        ):
            missing_rapid_boundary_indices.append(boundary_index)
            continue
        detail_step = _source_pair_metric(
            left_scalars,
            right_scalars,
            "sharpness_log",
        )
        luma_step = _source_pair_metric(left_scalars, right_scalars, "luma")
        chroma_step = _source_pair_metric(left_scalars, right_scalars, "chroma")
        normalized_shock = max(
            detail_step / config.rapid_sharpness_log_step,
            luma_step / config.rapid_luma_step,
            chroma_step / config.rapid_chroma_step,
        )
        rapid_boundaries.append(
            {
                "boundary_index": boundary_index,
                "outgoing_run_index": left["run_index"],
                "incoming_run_index": right["run_index"],
                "outgoing_source_index": left["source_index"],
                "incoming_source_index": right["source_index"],
                "outgoing_duration_seconds": left["duration_seconds"],
                "incoming_duration_seconds": right["duration_seconds"],
                "sharpness_log_step": detail_step,
                "luma_step": luma_step,
                "chroma_step": chroma_step,
                "normalized_quality_shock": normalized_shock,
                "high_quality_shock": normalized_shock > 1.0,
            }
        )
    high_quality_shock_boundary_indices = [
        int(item["boundary_index"])
        for item in rapid_boundaries
        if item["high_quality_shock"] is True
    ]
    rapid_high_fraction = (
        len(high_quality_shock_boundary_indices) / len(rapid_boundaries)
        if rapid_boundaries
        else None
    )

    reason_codes: list[str] = []
    if insufficient_source_indices:
        reason_codes.append("perceptual_source_scalar_coverage_incomplete")
    for metric, value in source_spans.items():
        if value is not None and value > metric_limits[metric]:
            reason_codes.append(f"perceptual_source_{metric}_median_span")
    rapid_gate_applicable = (
        expected_rapid_boundary_count
        >= config.minimum_rapid_source_switch_boundary_count
    )
    if rapid_gate_applicable and missing_rapid_boundary_indices:
        reason_codes.append("perceptual_rapid_quality_evidence_incomplete")
    if (
        rapid_gate_applicable
        and rapid_high_fraction is not None
        and rapid_high_fraction
        > config.maximum_rapid_high_quality_shock_fraction
    ):
        reason_codes.append("perceptual_rapid_quality_alternation")

    pair_target_boundaries: dict[str, list[int]] = {}
    for metric, pair in worst_source_pairs.items():
        if pair is None:
            pair_target_boundaries[metric] = []
            continue
        pair_set = set(int(value) for value in pair["source_indices"])
        pair_target_boundaries[metric] = [
            int(item["boundary_index"])
            for item in all_source_switch_boundaries
            if set(int(value) for value in item["source_indices"]) == pair_set
        ]
    failing_source_indices = set(insufficient_source_indices)
    for metric, pair in worst_source_pairs.items():
        if (
            pair is not None
            and source_spans[metric] is not None
            and float(source_spans[metric]) > metric_limits[metric]
        ):
            failing_source_indices.update(
                int(value) for value in pair["source_indices"]
            )
    source_related_boundary_indices = [
        int(item["boundary_index"])
        for item in all_source_switch_boundaries
        if failing_source_indices.intersection(
            int(value) for value in item["source_indices"]
        )
    ]
    repair_target_boundary_indices = sorted(
        {
            *high_quality_shock_boundary_indices,
            *missing_rapid_boundary_indices,
            *source_related_boundary_indices,
            *(
                boundary_index
                for metric, boundary_indices in pair_target_boundaries.items()
                if source_spans[metric] is not None
                and float(source_spans[metric]) > metric_limits[metric]
                for boundary_index in boundary_indices
            ),
        }
    )

    normalized_excesses = [
        max(0.0, float(value) / metric_limits[metric] - 1.0)
        for metric, value in source_spans.items()
        if value is not None
    ]
    if rapid_gate_applicable and rapid_high_fraction is not None:
        normalized_excesses.append(
            max(
                0.0,
                rapid_high_fraction
                / config.maximum_rapid_high_quality_shock_fraction
                - 1.0,
            )
        )
    selection_penalty_points = min(
        20.0,
        8.0 * max(normalized_excesses, default=0.0)
        + (20.0 if insufficient_source_indices else 0.0)
        + (20.0 if rapid_gate_applicable and missing_rapid_boundary_indices else 0.0),
    )
    return {
        "analysis_basis": PERCEPTUAL_SOURCE_ADMISSION_ANALYSIS_BASIS,
        "analysis_width": config.perceptual_analysis_width,
        "threshold_alignment": (
            "final_perceptual_quality_conditional_thresholds_proxy_preflight"
        ),
        "index_domains": {
            "source_index": "render_plan_sources_array_index",
            "boundary_index": "render_plan_transition_index",
        },
        "raw_frames_persisted": False,
        "original_pixels_serialized": False,
        "source_median_consistency": {
            "used_source_count": len(used_source_indices),
            "adequately_observed_source_count": len(valid_source_medians),
            "minimum_output_frames_per_source": (
                config.minimum_source_look_output_frames
            ),
            "insufficient_source_indices": insufficient_source_indices,
            "per_source": per_source,
            "sharpness_log_median_span": source_spans["sharpness_log"],
            "luma_median_span": source_spans["luma"],
            "chroma_median_span": source_spans["chroma"],
            "worst_source_pairs": worst_source_pairs,
            "worst_source_pair_boundary_indices": pair_target_boundaries,
        },
        "rapid_quality_alternation": {
            "source_run_count": len(run_medians),
            "expected_rapid_source_switch_boundary_count": (
                expected_rapid_boundary_count
            ),
            "observed_rapid_source_switch_boundary_count": len(rapid_boundaries),
            "minimum_required_boundary_count": (
                config.minimum_rapid_source_switch_boundary_count
            ),
            "gate_applicable": rapid_gate_applicable,
            "missing_boundary_indices": missing_rapid_boundary_indices,
            "high_quality_shock_count": len(high_quality_shock_boundary_indices),
            "high_quality_shock_fraction": rapid_high_fraction,
            "maximum_allowed_high_quality_shock_fraction": (
                config.maximum_rapid_high_quality_shock_fraction
            ),
            "high_quality_shock_boundary_indices": (
                high_quality_shock_boundary_indices
            ),
            "normalized_quality_shock": _distribution(
                [
                    float(item["normalized_quality_shock"])
                    for item in rapid_boundaries
                ]
            ),
            "boundaries": rapid_boundaries,
        },
        "admission": {
            "passed": not reason_codes,
            "reason_codes": reason_codes,
            "repair_target_boundary_indices": repair_target_boundary_indices,
            "repair_target_policy": (
                "high_or_missing_rapid_boundaries_plus_all_switches_touching_"
                "insufficient_or_failing_worst_pair_sources"
            ),
            "selection_penalty_points": selection_penalty_points,
        },
    }


def _structure_shock(left: np.ndarray, right: np.ndarray) -> float:
    def edges(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA)
        return cv2.Canny(small, 45, 135).astype(np.float32) / 255.0

    first = edges(left).reshape(-1)
    second = edges(right).reshape(-1)
    first_centered = first - float(first.mean())
    second_centered = second - float(second.mean())
    denominator = float(
        np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    )
    if denominator <= 1e-8:
        return 0.0 if np.allclose(first, second, atol=1e-6) else 1.0
    correlation = float(np.dot(first_centered, second_centered) / denominator)
    return _unit((1.0 - correlation) / 2.0)


def _color_shock(left: np.ndarray, right: np.ndarray) -> float:
    histograms: list[np.ndarray] = []
    lab_means: list[np.ndarray] = []
    for frame in (left, right):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
        histograms.append(histogram)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_means.append(lab.reshape(-1, 3).mean(axis=0))
    histogram_distance = float(
        cv2.compareHist(histograms[0], histograms[1], cv2.HISTCMP_BHATTACHARYYA)
    )
    mean_distance = float(np.linalg.norm(lab_means[0] - lab_means[1])) / 180.0
    return _unit(0.72 * histogram_distance + 0.28 * mean_distance)


def _sharpness_discontinuity(left: float, right: float) -> float:
    return _unit(abs(math.log((left + 0.04) / (right + 0.04))) / math.log(5.0))


def _box_discontinuity(
    left: tuple[float, float, float, float] | None,
    right: tuple[float, float, float, float] | None,
) -> tuple[float | None, float | None]:
    if left is None and right is None:
        return (None, None)
    if left is None or right is None:
        # Missing geometry is absence of comparable evidence, not proof of a
        # maximum scale *and* position jump.  Group shots commonly have no
        # single primary face; silhouette/formation evidence remains available
        # as a fallback when it was measured on both sides.
        return (None, None)
    left_center = (left[0] + left[2] / 2.0, left[1] + left[3] / 2.0)
    right_center = (right[0] + right[2] / 2.0, right[1] + right[3] / 2.0)
    position = _unit(
        math.hypot(
            left_center[0] - right_center[0],
            left_center[1] - right_center[1],
        )
        / 0.5
    )
    scale = _unit(abs(math.log(left[3] / right[3])) / math.log(3.0))
    return (scale, position)


def _dominant_motion(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    def small_gray(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        width = 160
        height = max(2, int(round(gray.shape[0] * width / gray.shape[1])))
        return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

    left = small_gray(first)
    right = small_gray(second)
    flow = cv2.calcOpticalFlowFarneback(
        left,
        right,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    magnitude = np.linalg.norm(flow, axis=2)
    threshold = max(0.25, float(np.percentile(magnitude, 70.0)))
    mask = magnitude >= threshold
    active_pixel_count = int(mask.sum())
    active_pixel_ratio = active_pixel_count / max(1, int(mask.size))
    coherent_pixel_ratio = 0.0
    signal_to_noise_ratio = 0.0
    if active_pixel_count < 16:
        dx = dy = 0.0
    else:
        active_dx = flow[..., 0][mask].astype(np.float64)
        active_dy = flow[..., 1][mask].astype(np.float64)
        median_dx = float(np.median(active_dx))
        median_dy = float(np.median(active_dy))
        vector_magnitude_pixels = math.hypot(median_dx, median_dy)
        residuals = np.hypot(active_dx - median_dx, active_dy - median_dy)
        residual_median = float(np.median(residuals))
        signal_to_noise_ratio = vector_magnitude_pixels / max(
            1e-6,
            residual_median,
        )
        if vector_magnitude_pixels > 1e-8:
            active_magnitudes = np.hypot(active_dx, active_dy)
            reliable = active_magnitudes > 1e-6
            if int(reliable.sum()) >= 16:
                cosine = (
                    active_dx[reliable] * median_dx
                    + active_dy[reliable] * median_dy
                ) / (active_magnitudes[reliable] * vector_magnitude_pixels)
                coherent_pixel_ratio = float(np.mean(cosine >= 0.5))
        dx = median_dx / left.shape[1]
        dy = median_dy / left.shape[0]
    normalized_magnitude = math.hypot(dx, dy)
    direction_available = (
        normalized_magnitude >= 0.001
        and coherent_pixel_ratio >= MOTION_MINIMUM_COHERENT_PIXEL_RATIO
        and signal_to_noise_ratio >= MOTION_MINIMUM_SIGNAL_TO_NOISE
    )
    direction = (
        [dx / normalized_magnitude, dy / normalized_magnitude]
        if direction_available
        else None
    )
    return {
        "vector_xy_per_frame": [dx, dy],
        "magnitude_per_frame": normalized_magnitude,
        "direction_xy": direction,
        "direction_available": direction_available,
        "active_pixel_count": active_pixel_count,
        "active_pixel_ratio": active_pixel_ratio,
        "coherent_pixel_ratio": coherent_pixel_ratio,
        "signal_to_noise_ratio": signal_to_noise_ratio,
        "reliability_policy": {
            "minimum_normalized_magnitude": 0.001,
            "minimum_coherent_pixel_ratio": (
                MOTION_MINIMUM_COHERENT_PIXEL_RATIO
            ),
            "minimum_signal_to_noise_ratio": MOTION_MINIMUM_SIGNAL_TO_NOISE,
        },
    }


def _motion_discontinuity(
    outgoing: Mapping[str, Any],
    incoming: Mapping[str, Any],
) -> float | None:
    if outgoing.get("direction_available") is False or incoming.get(
        "direction_available"
    ) is False:
        return None
    left = outgoing["direction_xy"]
    right = incoming["direction_xy"]
    if left is None and right is None:
        return None
    if left is None or right is None:
        return None
    cosine = float(left[0]) * float(right[0]) + float(left[1]) * float(right[1])
    return _unit((1.0 - cosine) / 2.0)


def _temporal_frame_delta(left: np.ndarray, right: np.ndarray) -> float:
    """Return a compact full-frame delta for concat-integrity checks."""

    def compact(frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32)

    first = compact(left)
    second = compact(right)
    # OpenCV's 8-bit Lab representation keeps every channel in [0, 255].
    return _unit(float(np.mean(np.abs(first - second))) / 255.0)


def _boundary_temporal_integrity(
    *,
    selected_frames: Mapping[int, np.ndarray],
    sample_offsets: tuple[int, int, int, int],
    frame_metrics: Mapping[int, Mapping[str, float]],
    minimum_quality: float,
) -> dict[str, Any]:
    previous_offset, outgoing_offset, incoming_offset, next_offset = sample_offsets
    before_delta = _temporal_frame_delta(
        selected_frames[previous_offset],
        selected_frames[outgoing_offset],
    )
    cross_delta = _temporal_frame_delta(
        selected_frames[outgoing_offset],
        selected_frames[incoming_offset],
    )
    after_delta = _temporal_frame_delta(
        selected_frames[incoming_offset],
        selected_frames[next_offset],
    )
    context_delta = max(before_delta, after_delta)
    context_ratio = cross_delta / max(context_delta, 1e-9)
    duplicate_or_freeze = (
        context_delta >= TEMPORAL_DUPLICATE_MINIMUM_CONTEXT_DELTA
        and cross_delta <= TEMPORAL_DUPLICATE_MAXIMUM_CROSS_DELTA
        and context_ratio <= TEMPORAL_DUPLICATE_MAXIMUM_CONTEXT_RATIO
    )
    drop_or_reset = (
        cross_delta >= TIER_D_RESET_MINIMUM_CROSS_DELTA
        and context_ratio >= TIER_D_RESET_MINIMUM_CONTEXT_RATIO
    )
    quality_values = [
        float(frame_metrics[offset]["quality"])
        for offset in sample_offsets
    ]
    quality_p10 = _quantile(quality_values, 0.10)
    black_offsets = [
        offset
        for offset in sample_offsets
        if float(frame_metrics[offset]["mean_luma"])
        <= BOUNDARY_BLACK_MAXIMUM_MEAN_LUMA
        and float(frame_metrics[offset]["contrast"])
        <= BOUNDARY_BLACK_MAXIMUM_CONTRAST
    ]
    return {
        "frame_delta": {
            "before": before_delta,
            "cross_boundary": cross_delta,
            "after": after_delta,
            "maximum_neighbor": context_delta,
            "cross_to_maximum_neighbor_ratio": context_ratio,
        },
        "duplicate_or_freeze_detected": duplicate_or_freeze,
        "drop_or_reset_detected": drop_or_reset,
        "boundary_quality_p10": quality_p10,
        "minimum_boundary_quality": minimum_quality,
        "boundary_quality_passed": (
            quality_p10 is not None and quality_p10 >= minimum_quality
        ),
        "black_frame_offsets": black_offsets,
        "thresholds": {
            "duplicate_minimum_context_delta": (
                TEMPORAL_DUPLICATE_MINIMUM_CONTEXT_DELTA
            ),
            "duplicate_maximum_cross_delta": (
                TEMPORAL_DUPLICATE_MAXIMUM_CROSS_DELTA
            ),
            "duplicate_maximum_context_ratio": (
                TEMPORAL_DUPLICATE_MAXIMUM_CONTEXT_RATIO
            ),
            "tier_d_reset_minimum_cross_delta": (
                TIER_D_RESET_MINIMUM_CROSS_DELTA
            ),
            "tier_d_reset_minimum_context_ratio": (
                TIER_D_RESET_MINIMUM_CONTEXT_RATIO
            ),
            "black_maximum_mean_luma": BOUNDARY_BLACK_MAXIMUM_MEAN_LUMA,
            "black_maximum_contrast": BOUNDARY_BLACK_MAXIMUM_CONTRAST,
        },
    }


_BOUNDARY_WEIGHTS = {
    "sharpness_discontinuity": 0.12,
    "exposure_discontinuity": 0.08,
    "color_shock": 0.16,
    "structure_shock": 0.20,
    "face_scale_discontinuity": 0.10,
    "face_position_discontinuity": 0.10,
    "silhouette_scale_discontinuity": 0.08,
    "silhouette_position_discontinuity": 0.08,
    "motion_direction_discontinuity": 0.08,
}


def _boundary_score(
    discontinuities: Mapping[str, float | None],
    *,
    effective_severity_divisor: float = 1.0,
) -> float:
    active = tuple(
        (name, weight, discontinuities.get(name))
        for name, weight in _BOUNDARY_WEIGHTS.items()
        if discontinuities.get(name) is not None
    )
    if not active:
        return 100.0
    weight_sum = sum(weight for _name, weight, _value in active)
    severity = sum(
        weight * _unit(float(value) / effective_severity_divisor)
        for _name, weight, value in active
        if value is not None
    ) / weight_sum
    return 100.0 * (1.0 - severity)


def _issue_reasons(
    discontinuities: Mapping[str, float | None],
    score: float,
    config: ProxyReviewConfig,
    threshold_multiplier: float = 1.0,
) -> list[str]:
    thresholds = {
        "sharpness_discontinuity": config.maximum_sharpness_discontinuity,
        "exposure_discontinuity": config.maximum_exposure_discontinuity,
        "color_shock": config.maximum_color_shock,
        "structure_shock": config.maximum_structure_shock,
        "face_scale_discontinuity": config.maximum_face_scale_discontinuity,
        "face_position_discontinuity": config.maximum_face_position_discontinuity,
        "silhouette_scale_discontinuity": (
            config.maximum_silhouette_scale_discontinuity
        ),
        "silhouette_position_discontinuity": (
            config.maximum_silhouette_position_discontinuity
        ),
        "motion_direction_discontinuity": (
            config.maximum_motion_direction_discontinuity
        ),
    }
    reasons = [
        name
        for name, threshold in thresholds.items()
        if discontinuities.get(name) is not None
        and float(discontinuities[name]) > threshold * threshold_multiplier
    ]
    if score < config.minimum_boundary_score_p10:
        reasons.append("weighted_boundary_score")
    return reasons


def _safe_geometry(
    provider: GeometryProvider,
    frame: np.ndarray,
) -> tuple[OutputSubjectGeometry, bool]:
    try:
        geometry = provider(frame)
        if not isinstance(geometry, OutputSubjectGeometry):
            raise TypeError("geometry provider returned an invalid value")
        return (geometry, False)
    except (cv2.error, TypeError, ValueError, FloatingPointError):
        return (OutputSubjectGeometry(), True)


def _review_boundary(
    *,
    window: _BoundaryReviewWindow,
    fps: float,
    frames: Mapping[int, np.ndarray],
    scalars: Mapping[int, Mapping[str, float]],
    geometry_provider: GeometryProvider,
    config: ProxyReviewConfig,
) -> tuple[dict[str, Any], int]:
    boundary_index = window.boundary_index
    cut_frame_index = window.cut_frame_index
    outgoing_source_index = window.outgoing_source_index
    incoming_source_index = window.incoming_source_index
    sample_offsets = window.sample_offsets
    selected_frames = {
        offset: frames[cut_frame_index + offset]
        for offset in sample_offsets
    }
    geometries: dict[int, OutputSubjectGeometry] = {}
    geometry_failures = 0
    for offset, frame in selected_frames.items():
        geometry, failed = _safe_geometry(geometry_provider, frame)
        geometries[offset] = geometry
        geometry_failures += int(failed)

    outgoing_previous_offset, outgoing_offset, incoming_offset, incoming_next_offset = (
        sample_offsets
    )
    left_metrics = scalars[cut_frame_index + outgoing_offset]
    right_metrics = scalars[cut_frame_index + incoming_offset]
    face_scale, face_position = _box_discontinuity(
        geometries[outgoing_offset].face_bbox_xywh,
        geometries[incoming_offset].face_bbox_xywh,
    )
    silhouette_scale, silhouette_position = _box_discontinuity(
        geometries[outgoing_offset].silhouette_bbox_xywh,
        geometries[incoming_offset].silhouette_bbox_xywh,
    )
    outgoing_motion = _dominant_motion(
        selected_frames[outgoing_previous_offset],
        selected_frames[outgoing_offset],
    )
    incoming_motion = _dominant_motion(
        selected_frames[incoming_offset],
        selected_frames[incoming_next_offset],
    )
    discontinuities: dict[str, float | None] = {
        "sharpness_discontinuity": _sharpness_discontinuity(
            left_metrics["sharpness"], right_metrics["sharpness"]
        ),
        "exposure_discontinuity": abs(
            left_metrics["exposure"] - right_metrics["exposure"]
        ),
        "color_shock": _color_shock(
            selected_frames[outgoing_offset],
            selected_frames[incoming_offset],
        ),
        "structure_shock": _structure_shock(
            selected_frames[outgoing_offset],
            selected_frames[incoming_offset],
        ),
        "face_scale_discontinuity": face_scale,
        "face_position_discontinuity": face_position,
        "silhouette_scale_discontinuity": silhouette_scale,
        "silhouette_position_discontinuity": silhouette_position,
        "motion_direction_discontinuity": _motion_discontinuity(
            outgoing_motion, incoming_motion
        ),
    }
    endpoint_multiplier = (
        config.dissolve_endpoint_tolerance_multiplier
        if window.transition_kind is TransitionKind.FACE_DISSOLVE
        else 1.0
    )
    editorial_score = _boundary_score(
        discontinuities,
        effective_severity_divisor=endpoint_multiplier,
    )
    editorial_reasons = _issue_reasons(
        discontinuities,
        editorial_score,
        config,
        threshold_multiplier=endpoint_multiplier,
    )
    temporal_integrity = _boundary_temporal_integrity(
        selected_frames=selected_frames,
        sample_offsets=sample_offsets,
        frame_metrics={
            offset: scalars[cut_frame_index + offset]
            for offset in sample_offsets
        },
        minimum_quality=config.minimum_overall_quality_p10,
    )
    technical_integrity_reasons: list[str] = []
    if window.review_domain == INTRODUCED_CROSS_SOURCE_DOMAIN:
        score = editorial_score
        # Background/lighting discontinuity is judged relative to professional
        # references, but a literal black or duplicated/frozen frame remains a
        # zero-tolerance technical failure at every boundary domain.
        if temporal_integrity["duplicate_or_freeze_detected"]:
            technical_integrity_reasons.append(
                "introduced_cross_source_duplicate_or_freeze"
            )
        if temporal_integrity["black_frame_offsets"]:
            technical_integrity_reasons.append(
                "introduced_cross_source_black_frame"
            )
        reasons = [*editorial_reasons, *technical_integrity_reasons]
    else:
        reasons = []
        if not window.source_clock_contiguous:
            technical_integrity_reasons.append(
                (
                    "tier_d_source_clock_discontinuity"
                    if window.review_domain == TIER_D_CONTINUITY_DOMAIN
                    else "native_source_cut_source_clock_discontinuity"
                )
            )
        if temporal_integrity["duplicate_or_freeze_detected"]:
            technical_integrity_reasons.append(
                (
                    "tier_d_cross_boundary_duplicate_or_freeze"
                    if window.review_domain == TIER_D_CONTINUITY_DOMAIN
                    else "native_source_cut_cross_boundary_duplicate_or_freeze"
                )
            )
        if (
            window.review_domain == TIER_D_CONTINUITY_DOMAIN
            and temporal_integrity["drop_or_reset_detected"]
        ):
            technical_integrity_reasons.append(
                "tier_d_cross_boundary_drop_or_reset"
            )
        if not temporal_integrity["boundary_quality_passed"]:
            technical_integrity_reasons.append(
                (
                    "tier_d_boundary_frame_quality"
                    if window.review_domain == TIER_D_CONTINUITY_DOMAIN
                    else "native_source_cut_boundary_frame_quality"
                )
            )
        if temporal_integrity["black_frame_offsets"]:
            technical_integrity_reasons.append(
                (
                    "tier_d_black_frame"
                    if window.review_domain == TIER_D_CONTINUITY_DOMAIN
                    else "native_source_cut_black_frame"
                )
            )
        reasons = list(technical_integrity_reasons)
        score = 100.0 if not reasons else 0.0
    frame_indices = window.frame_indices
    return (
        {
            "boundary_index": boundary_index,
            "cut_frame_index": cut_frame_index,
            "outgoing_source_index": outgoing_source_index,
            "incoming_source_index": incoming_source_index,
            "review_domain": window.review_domain,
            "transition_reason": window.transition_reason,
            "transition_tier": window.transition_tier,
            "source_clock_contiguous": window.source_clock_contiguous,
            "transition_kind": window.transition_kind.value,
            "transition_frame_count": window.transition_frame_count,
            "transition_duration_seconds": window.duration_seconds(fps),
            "sample_offsets": list(sample_offsets),
            "frame_indices": list(frame_indices),
            "endpoint_review_policy": {
                "comparison": (
                    "two_frames_before_transition_vs_two_frames_after_"
                    "completed_transition"
                ),
                "threshold_multiplier": endpoint_multiplier,
                "effective_severity_divisor": endpoint_multiplier,
                "maximum_allowed_multiplier": 1.25,
            },
            "frame_metrics": [
                {
                    "offset": offset,
                    "frame_index": cut_frame_index + offset,
                    **deepcopy(scalars[cut_frame_index + offset]),
                    "subject_geometry": geometries[offset].to_dict(),
                }
                for offset in sample_offsets
            ],
            "motion": {
                "outgoing_pre_transition": outgoing_motion,
                "incoming_post_transition": incoming_motion,
                **(
                    {
                        "outgoing_minus2_to_minus1": outgoing_motion,
                        "incoming_plus0_to_plus1": incoming_motion,
                    }
                    if window.transition_kind is TransitionKind.HARD_CUT
                    else {}
                ),
            },
            "geometry_comparability": {
                "face": (
                    geometries[outgoing_offset].face_bbox_xywh is not None
                    and geometries[incoming_offset].face_bbox_xywh is not None
                ),
                "silhouette": (
                    geometries[outgoing_offset].silhouette_bbox_xywh is not None
                    and geometries[incoming_offset].silhouette_bbox_xywh is not None
                ),
                "motion_direction": (
                    outgoing_motion["direction_xy"] is not None
                    and incoming_motion["direction_xy"] is not None
                ),
                "unavailable_policy": "not_comparable_never_maximum_mismatch",
            },
            "discontinuities": discontinuities,
            "editorial_continuity_applicable": (
                window.review_domain == INTRODUCED_CROSS_SOURCE_DOMAIN
            ),
            "diagnostic_editorial_weighted_score": editorial_score,
            "diagnostic_editorial_issue_reasons": editorial_reasons,
            "temporal_integrity": temporal_integrity,
            "technical_integrity_issue_reasons": technical_integrity_reasons,
            "weighted_score": score,
            "passed": not reasons,
            "issue_reasons": reasons,
        },
        geometry_failures,
    )


def review_hard_cut_frame_window(
    frames: Mapping[int, np.ndarray],
    *,
    cut_frame_index: int,
    config: ProxyReviewConfig | None = None,
    geometry_provider: GeometryProvider | None = None,
) -> dict[str, Any]:
    """Review one decoded hard cut using the production boundary metrics.

    ``cut_frame_index`` is the first frame of the incoming shot.  Callers must
    supply the exact ``-2, -1, 0, +1`` frame window.  The returned object is
    compact scalar/geometry/motion evidence only; no frame pixels, descriptors,
    keypoints, or appearance embeddings are retained.  Absolute editorial
    warnings are calibration evidence for a human edit, while black/frozen or
    undecodable boundary frames remain a separate fail-closed technical axis.
    """

    if isinstance(cut_frame_index, bool) or not isinstance(cut_frame_index, int):
        raise TypeError("cut_frame_index must be an integer")
    if cut_frame_index < 2:
        raise ValueError("hard-cut review requires two outgoing frames")
    settings = config or ProxyReviewConfig()
    if not isinstance(settings, ProxyReviewConfig):
        raise TypeError("config must be a ProxyReviewConfig")
    provider = geometry_provider or _default_geometry_provider
    required_indices = tuple(
        cut_frame_index + offset for offset in BOUNDARY_FRAME_OFFSETS
    )
    actual_indices = set(frames)
    if actual_indices != set(required_indices):
        raise ValueError(
            "hard-cut frame window must contain exactly "
            f"{required_indices}; found {tuple(sorted(actual_indices))}"
        )

    selected: dict[int, np.ndarray] = {}
    for offset, frame_index in zip(
        BOUNDARY_FRAME_OFFSETS,
        required_indices,
        strict=True,
    ):
        frame = frames[frame_index]
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError(f"boundary frame {frame_index} is empty")
        selected[offset] = frame

    scalars = {
        offset: _frame_scalars(frame) for offset, frame in selected.items()
    }
    geometries: dict[int, OutputSubjectGeometry] = {}
    geometry_failure_count = 0
    for offset, frame in selected.items():
        geometry, failed = _safe_geometry(provider, frame)
        geometries[offset] = geometry
        geometry_failure_count += int(failed)

    outgoing_previous_offset, outgoing_offset, incoming_offset, incoming_next_offset = (
        BOUNDARY_FRAME_OFFSETS
    )
    left_metrics = scalars[outgoing_offset]
    right_metrics = scalars[incoming_offset]
    face_scale, face_position = _box_discontinuity(
        geometries[outgoing_offset].face_bbox_xywh,
        geometries[incoming_offset].face_bbox_xywh,
    )
    silhouette_scale, silhouette_position = _box_discontinuity(
        geometries[outgoing_offset].silhouette_bbox_xywh,
        geometries[incoming_offset].silhouette_bbox_xywh,
    )
    outgoing_motion = _dominant_motion(
        selected[outgoing_previous_offset],
        selected[outgoing_offset],
    )
    incoming_motion = _dominant_motion(
        selected[incoming_offset],
        selected[incoming_next_offset],
    )
    discontinuities: dict[str, float | None] = {
        "sharpness_discontinuity": _sharpness_discontinuity(
            left_metrics["sharpness"],
            right_metrics["sharpness"],
        ),
        "exposure_discontinuity": abs(
            left_metrics["exposure"] - right_metrics["exposure"]
        ),
        "color_shock": _color_shock(
            selected[outgoing_offset],
            selected[incoming_offset],
        ),
        "structure_shock": _structure_shock(
            selected[outgoing_offset],
            selected[incoming_offset],
        ),
        "face_scale_discontinuity": face_scale,
        "face_position_discontinuity": face_position,
        "silhouette_scale_discontinuity": silhouette_scale,
        "silhouette_position_discontinuity": silhouette_position,
        "motion_direction_discontinuity": _motion_discontinuity(
            outgoing_motion,
            incoming_motion,
        ),
    }
    score = _boundary_score(discontinuities)
    issue_reasons = _issue_reasons(discontinuities, score, settings)
    temporal_integrity = _boundary_temporal_integrity(
        selected_frames=selected,
        sample_offsets=BOUNDARY_FRAME_OFFSETS,
        frame_metrics=scalars,
        minimum_quality=settings.minimum_overall_quality_p10,
    )
    technical_reasons: list[str] = []
    if temporal_integrity["duplicate_or_freeze_detected"]:
        technical_reasons.append("human_reference_duplicate_or_freeze")
    # Human-reference cuts are editorial-domain boundaries.  An intentional
    # white flash or deep-lighting cue can score below the generic per-frame
    # quality floor and remains useful calibration evidence.  This mirrors the
    # production introduced-cross-source domain: only literal black/freeze or
    # failed analysis is technical here; quality remains serialized diagnostic.
    if temporal_integrity["black_frame_offsets"]:
        technical_reasons.append("human_reference_black_frame")
    if geometry_failure_count:
        technical_reasons.append("human_reference_geometry_provider_failure")

    return {
        "analysis_basis": HUMAN_REFERENCE_BOUNDARY_ANALYSIS_BASIS,
        "cut_frame_index": cut_frame_index,
        "sample_offsets": list(BOUNDARY_FRAME_OFFSETS),
        "frame_indices": list(required_indices),
        "frame_metrics": [
            {
                "offset": offset,
                "frame_index": cut_frame_index + offset,
                **deepcopy(scalars[offset]),
                "subject_geometry": geometries[offset].to_dict(),
            }
            for offset in BOUNDARY_FRAME_OFFSETS
        ],
        "motion": {
            "outgoing_minus2_to_minus1": outgoing_motion,
            "incoming_plus0_to_plus1": incoming_motion,
        },
        "geometry_comparability": {
            "face": (
                geometries[outgoing_offset].face_bbox_xywh is not None
                and geometries[incoming_offset].face_bbox_xywh is not None
            ),
            "silhouette": (
                geometries[outgoing_offset].silhouette_bbox_xywh is not None
                and geometries[incoming_offset].silhouette_bbox_xywh is not None
            ),
            "motion_direction": (
                outgoing_motion["direction_xy"] is not None
                and incoming_motion["direction_xy"] is not None
            ),
            "unavailable_policy": "not_comparable_never_maximum_mismatch",
        },
        "discontinuities": discontinuities,
        "weighted_score": score,
        "issue_reasons": issue_reasons,
        "temporal_integrity": temporal_integrity,
        "technical_integrity_issue_reasons": technical_reasons,
        "technical_integrity_passed": not technical_reasons,
        "geometry_provider_failure_count": geometry_failure_count,
        "serialization_policy": {
            "raw_frames_serialized": False,
            "descriptors_or_keypoints_serialized": False,
            "appearance_embeddings_serialized": False,
            "compact_scalar_geometry_motion_evidence_only": True,
        },
    }


def _insufficient_transition_handle_review(
    window: _BoundaryReviewWindow,
    *,
    fps: float,
    config: ProxyReviewConfig,
) -> dict[str, Any]:
    endpoint_multiplier = (
        config.dissolve_endpoint_tolerance_multiplier
        if window.transition_kind is TransitionKind.FACE_DISSOLVE
        else 1.0
    )
    return {
        "boundary_index": window.boundary_index,
        "cut_frame_index": window.cut_frame_index,
        "outgoing_source_index": window.outgoing_source_index,
        "incoming_source_index": window.incoming_source_index,
        "review_domain": window.review_domain,
        "transition_reason": window.transition_reason,
        "transition_tier": window.transition_tier,
        "source_clock_contiguous": window.source_clock_contiguous,
        "transition_kind": window.transition_kind.value,
        "transition_frame_count": window.transition_frame_count,
        "transition_duration_seconds": window.duration_seconds(fps),
        "sample_offsets": list(window.sample_offsets),
        "frame_indices": list(window.frame_indices),
        "endpoint_review_policy": {
            "comparison": (
                "two_frames_before_transition_vs_two_frames_after_"
                "completed_transition"
            ),
            "threshold_multiplier": endpoint_multiplier,
            "effective_severity_divisor": endpoint_multiplier,
            "maximum_allowed_multiplier": 1.25,
        },
        "frame_metrics": [],
        "motion": {
            "outgoing_pre_transition": None,
            "incoming_post_transition": None,
        },
        "geometry_comparability": {
            "face": False,
            "silhouette": False,
            "motion_direction": False,
            "unavailable_policy": "not_comparable_never_maximum_mismatch",
        },
        "discontinuities": {},
        "editorial_continuity_applicable": (
            window.review_domain == INTRODUCED_CROSS_SOURCE_DOMAIN
        ),
        "diagnostic_editorial_weighted_score": 0.0,
        "diagnostic_editorial_issue_reasons": [
            "insufficient_post_transition_review_handle"
        ],
        "temporal_integrity": None,
        "technical_integrity_issue_reasons": [
            "insufficient_post_transition_review_handle"
        ],
        "weighted_score": 0.0,
        "passed": False,
        "issue_reasons": ["insufficient_post_transition_review_handle"],
    }


def _boundary_domain_summary(
    boundary_reviews: Sequence[Mapping[str, Any]],
    *,
    domain: str,
    config: ProxyReviewConfig,
) -> dict[str, Any]:
    selected = [
        item for item in boundary_reviews if item.get("review_domain") == domain
    ]
    scores = [float(item["weighted_score"]) for item in selected]
    issue_indices = [
        int(item["boundary_index"])
        for item in selected
        if not bool(item["passed"])
    ]
    score_p10 = _quantile(scores, 0.10)
    if score_p10 is None:
        score_p10 = 100.0
    issue_fraction = len(issue_indices) / len(selected) if selected else 0.0
    if domain == INTRODUCED_CROSS_SOURCE_DOMAIN:
        passed = (
            score_p10 >= config.minimum_boundary_score_p10
            and issue_fraction <= config.maximum_boundary_issue_fraction
        )
        pass_policy = (
            "editorial_boundary_p10_and_issue_fraction_gate"
        )
    else:
        passed = not issue_indices
        pass_policy = "zero_technical_integrity_issues"
    return {
        "domain": domain,
        "count_basis": domain,
        "boundary_count": len(selected),
        "issue_count": len(issue_indices),
        "issue_fraction": issue_fraction,
        "issue_indices": issue_indices,
        "score": {
            "distribution": _distribution(scores),
            "p10": score_p10,
        },
        "pass_policy": pass_policy,
        "passed": passed,
    }


def review_rendered_proxy(
    plan: RenderPlan,
    *,
    config: ProxyReviewConfig | None = None,
    geometry_provider: GeometryProvider | None = None,
    render_codec: str | None = None,
    professional_reference_profile: (
        "ProfessionalReferenceProfile | Path | None"
    ) = None,
) -> RenderedProxyReview:
    """Decode one already-rendered proxy and return scalar-only QA evidence."""

    settings = config or ProxyReviewConfig()
    expected_width, expected_height = proxy_dimensions(plan.aspect_ratio)
    if (plan.width, plan.height) != (expected_width, expected_height):
        raise ValueError("proxy review requires the fixed 480p RenderPlan size")
    capture = cv2.VideoCapture(str(plan.output))
    if not capture.isOpened():
        raise ValueError(f"could not open rendered proxy: {plan.output}")

    declared_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    declared_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    declared_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if (declared_width, declared_height) != (plan.width, plan.height):
        capture.release()
        raise ValueError("decoded proxy dimensions do not match its RenderPlan")
    if not math.isfinite(declared_fps) or abs(declared_fps - plan.fps) > 0.05:
        capture.release()
        raise ValueError("decoded proxy frame rate does not match its RenderPlan")

    counts = segment_frame_counts(plan)
    segment_run_indices: list[int] = []
    run_source_indices: list[int] = []
    run_durations_seconds: list[float] = []
    run_incoming_boundary_indices: list[int | None] = []
    for segment_index_value, (segment, frame_count) in enumerate(
        zip(plan.segments, counts, strict=True)
    ):
        previous_segment = (
            plan.segments[segment_index_value - 1]
            if segment_index_value > 0
            else None
        )
        source_clock_contiguous = (
            previous_segment is not None
            and previous_segment.source_index == segment.source_index
            and abs(segment.source_start - previous_segment.source_end) <= 1e-4
        )
        if not run_source_indices or not source_clock_contiguous:
            run_source_indices.append(segment.source_index)
            run_durations_seconds.append(0.0)
            run_incoming_boundary_indices.append(
                None if segment_index_value == 0 else segment_index_value - 1
            )
        run_durations_seconds[-1] += segment.duration
        segment_run_indices.append(len(run_source_indices) - 1)
    cumulative = 0
    boundary_windows: list[_BoundaryReviewWindow] = []
    reviewable_windows: list[_BoundaryReviewWindow] = []
    boundary_reviews: list[dict[str, Any]] = []
    for index, transition in enumerate(plan.transitions):
        cumulative += counts[index]
        outgoing_segment = plan.segments[index]
        incoming_segment = plan.segments[index + 1]
        review_domain = _boundary_review_domain(
            outgoing_source_index=outgoing_segment.source_index,
            incoming_source_index=incoming_segment.source_index,
            transition_reason=transition.reason,
            transition_tier=transition.tier,
        )
        sample_offsets = (
            BOUNDARY_FRAME_OFFSETS
            if transition.kind is TransitionKind.HARD_CUT
            else (
                -2,
                -1,
                transition.frame_count,
                transition.frame_count + 1,
            )
        )
        window = _BoundaryReviewWindow(
            boundary_index=index,
            cut_frame_index=cumulative,
            outgoing_source_index=outgoing_segment.source_index,
            incoming_source_index=incoming_segment.source_index,
            transition_kind=transition.kind,
            transition_frame_count=transition.frame_count,
            transition_reason=transition.reason,
            transition_tier=transition.tier,
            review_domain=review_domain,
            source_clock_contiguous=(
                outgoing_segment.source_index == incoming_segment.source_index
                and abs(
                    outgoing_segment.source_end - incoming_segment.source_start
                )
                <= 1e-4
            ),
            sample_offsets=sample_offsets,
        )
        boundary_windows.append(window)
        if (
            transition.kind is TransitionKind.FACE_DISSOLVE
            and counts[index + 1] < transition.frame_count + 2
        ):
            boundary_reviews.append(
                _insufficient_transition_handle_review(
                    window,
                    fps=plan.fps,
                    config=settings,
                )
            )
        else:
            reviewable_windows.append(window)

    uses: Counter[int] = Counter()
    due_at: defaultdict[int, list[_BoundaryReviewWindow]] = defaultdict(list)
    for window in reviewable_windows:
        due_at[window.due_frame_index].append(window)
        for frame_index_value in window.frame_indices:
            uses[frame_index_value] += 1

    provider = geometry_provider or _default_geometry_provider
    cached_frames: dict[int, np.ndarray] = {}
    cached_scalars: dict[int, dict[str, float]] = {}
    all_quality: list[float] = []
    source_quality: defaultdict[int, list[float]] = defaultdict(list)
    source_perceptual_metrics: defaultdict[
        int,
        list[dict[str, float]],
    ] = defaultdict(list)
    run_perceptual_metrics: defaultdict[
        int,
        list[dict[str, float]],
    ] = defaultdict(list)
    geometry_failure_count = 0
    frame_index = 0
    segment_index = 0
    segment_end = counts[0]
    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break
            while frame_index >= segment_end and segment_index + 1 < len(counts):
                segment_index += 1
                segment_end += counts[segment_index]
            metrics = _frame_scalars(frame)
            perceptual_metrics = _perceptual_frame_scalars(
                frame,
                perceptual_analysis_width=settings.perceptual_analysis_width,
            )
            quality = metrics["quality"]
            all_quality.append(quality)
            source_index = plan.segments[segment_index].source_index
            source_quality[source_index].append(quality)
            source_perceptual_metrics[source_index].append(
                perceptual_metrics
            )
            run_perceptual_metrics[
                segment_run_indices[segment_index]
            ].append(perceptual_metrics)
            if frame_index in uses:
                cached_frames[frame_index] = frame.copy()
                cached_scalars[frame_index] = metrics
            for window in due_at.get(frame_index, ()):
                try:
                    review, failures = _review_boundary(
                        window=window,
                        fps=plan.fps,
                        frames=cached_frames,
                        scalars=cached_scalars,
                        geometry_provider=provider,
                        config=settings,
                    )
                except KeyError:
                    review = _insufficient_transition_handle_review(
                        window,
                        fps=plan.fps,
                        config=settings,
                    )
                    failures = 0
                boundary_reviews.append(review)
                geometry_failure_count += failures
                for used_index in window.frame_indices:
                    uses[used_index] -= 1
                    if uses[used_index] == 0:
                        cached_frames.pop(used_index, None)
                        cached_scalars.pop(used_index, None)
            frame_index += 1
    finally:
        capture.release()

    if frame_index != plan.output_frame_count:
        raise ValueError(
            "decoded proxy frame count does not match its RenderPlan: "
            f"expected {plan.output_frame_count}, decoded {frame_index}"
        )
    boundary_reviews.sort(key=lambda item: int(item["boundary_index"]))
    if len(boundary_reviews) != len(boundary_windows):
        raise ValueError("proxy ended before every transition window was reviewed")
    hard_cut_count = sum(
        window.transition_kind is TransitionKind.HARD_CUT
        for window in boundary_windows
    )
    dissolve_count = sum(
        window.transition_kind is TransitionKind.FACE_DISSOLVE
        for window in boundary_windows
    )

    overall_quality = _distribution(all_quality)
    per_source: list[dict[str, Any]] = []
    source_medians: list[float] = []
    source_p10_values: list[float] = []
    for source_index in sorted(source_quality):
        distribution = _distribution(source_quality[source_index])
        if distribution["median"] is not None:
            source_medians.append(float(distribution["median"]))
        if distribution["p10"] is not None:
            source_p10_values.append(float(distribution["p10"]))
        per_source.append(
            {
                "source_index": source_index,
                "quality": distribution,
            }
        )
    median_span = (
        max(source_medians) - min(source_medians)
        if len(source_medians) >= 2
        else 0.0
    )
    p10_span = (
        max(source_p10_values) - min(source_p10_values)
        if len(source_p10_values) >= 2
        else 0.0
    )
    used_source_indices = tuple(
        sorted({segment.source_index for segment in plan.segments})
    )
    perceptual_source_admission = _proxy_perceptual_source_admission(
        source_metrics={
            source_index: tuple(source_perceptual_metrics.get(source_index, ()))
            for source_index in used_source_indices
        },
        run_metrics={
            run_index: tuple(run_perceptual_metrics.get(run_index, ()))
            for run_index in range(len(run_source_indices))
        },
        run_source_indices=run_source_indices,
        run_durations_seconds=run_durations_seconds,
        run_incoming_boundary_indices=run_incoming_boundary_indices,
        config=settings,
    )
    boundary_scores = [float(item["weighted_score"]) for item in boundary_reviews]
    boundary_score_p10 = _quantile(boundary_scores, 0.10)
    if boundary_score_p10 is None:
        boundary_score_p10 = 100.0
    issue_indices = [
        int(item["boundary_index"])
        for item in boundary_reviews
        if not bool(item["passed"])
    ]
    issue_fraction = len(issue_indices) / len(boundary_reviews) if boundary_reviews else 0.0
    domain_summaries = {
        domain: _boundary_domain_summary(
            boundary_reviews,
            domain=domain,
            config=settings,
        )
        for domain in BOUNDARY_REVIEW_DOMAINS
    }
    visible_summary = domain_summaries[INTRODUCED_CROSS_SOURCE_DOMAIN]
    visible_score = visible_summary["score"]
    visible_boundary_score_p10 = float(visible_score["p10"])
    visible_issue_fraction = float(visible_summary["issue_fraction"])
    visible_issue_indices = list(visible_summary["issue_indices"])
    quality_p10 = float(overall_quality["p10"] or 0.0)
    consistency = 1.0 - _unit(
        median_span / max(1e-9, settings.maximum_source_quality_median_span)
    )
    weighted_score_before_perceptual_source_admission = 100.0 * (
        0.65 * (visible_boundary_score_p10 / 100.0)
        + 0.20 * quality_p10
        + 0.15 * consistency
    )
    perceptual_selection_penalty = float(
        perceptual_source_admission["admission"]["selection_penalty_points"]
    )
    weighted_score = max(
        0.0,
        weighted_score_before_perceptual_source_admission
        - perceptual_selection_penalty,
    )
    professional_gate_payload: dict[str, Any] | None = None
    introduced_technical_reasons = sorted(
        {
            str(reason)
            for boundary in boundary_reviews
            for reason in boundary.get("technical_integrity_issue_reasons", [])
            if str(reason).strip()
        }
    )
    if professional_reference_profile is not None:
        from .professional_reference_gate import (
            ProfessionalReferenceProfile,
            evaluate_professional_reference_gate,
        )

        checked_profile = (
            professional_reference_profile
            if isinstance(
                professional_reference_profile,
                ProfessionalReferenceProfile,
            )
            else ProfessionalReferenceProfile.from_file(
                Path(professional_reference_profile)
            )
        )
        visible_boundaries = tuple(
            boundary
            for boundary in boundary_reviews
            if boundary.get("review_domain")
            == INTRODUCED_CROSS_SOURCE_DOMAIN
        )
        professional_gate_payload = evaluate_professional_reference_gate(
            visible_boundaries,
            checked_profile,
            technical_issue_reasons=introduced_technical_reasons,
        ).payload
    editorial_gate_passed = (
        bool(professional_gate_payload["passed"])
        if professional_gate_payload is not None
        else bool(visible_summary["passed"])
    )
    weighted_score_gate_passed = (
        True
        if professional_gate_payload is not None
        else weighted_score >= settings.minimum_weighted_score
    )
    passed = (
        quality_p10 >= settings.minimum_overall_quality_p10
        and median_span <= settings.maximum_source_quality_median_span
        and bool(perceptual_source_admission["admission"]["passed"])
        and editorial_gate_passed
        and bool(domain_summaries[TIER_D_CONTINUITY_DOMAIN]["passed"])
        and bool(domain_summaries[NATIVE_SOURCE_CUT_DOMAIN]["passed"])
        and not introduced_technical_reasons
        and weighted_score_gate_passed
    )

    payload = {
        "schema_version": 3,
        "analysis": ANALYSIS_BASIS,
        "proxy": {
            "path": str(plan.output.resolve()),
            "width": plan.width,
            "height": plan.height,
            "fps": plan.fps,
            "decoded_frame_count": frame_index,
            "duration_seconds": plan.render_duration,
            "render_codec": render_codec,
        },
        "review_policy": settings.to_dict(),
        "storage_policy": {
            "frame_images_persisted": False,
            "original_pixels_serialized": False,
            "appearance_or_identity_embeddings_serialized": False,
            "serialized_evidence": "derived_scalars_and_normalized_geometry_only",
        },
        "overall_quality": overall_quality,
        "source_quality": {
            "per_source": per_source,
            "median_span": median_span,
            "p10_span": p10_span,
            "maximum_allowed_median_span": (
                settings.maximum_source_quality_median_span
            ),
        },
        "perceptual_source_admission": perceptual_source_admission,
        "transition_count": len(boundary_windows),
        "reviewed_boundary_count": len(boundary_reviews),
        "hard_cut_count": hard_cut_count,
        "dissolve_count": dissolve_count,
        "introduced_cross_source_count": domain_summaries[
            INTRODUCED_CROSS_SOURCE_DOMAIN
        ]["boundary_count"],
        "tier_d_continuity_count": domain_summaries[
            TIER_D_CONTINUITY_DOMAIN
        ]["boundary_count"],
        "native_source_cut_count": domain_summaries[
            NATIVE_SOURCE_CUT_DOMAIN
        ]["boundary_count"],
        # Compatibility view for schema-v2 readers.  It still summarizes all
        # serialized boundary pass flags; selection and repair use the visible
        # cross-source view below.
        "boundary_score": {
            "distribution": _distribution(boundary_scores),
            "p10": boundary_score_p10,
            "issue_count": len(issue_indices),
            "issue_fraction": issue_fraction,
            "count_basis": "all_serialized_boundaries_legacy_compatibility",
            "selection_gate": False,
        },
        "visible_boundary_score": {
            "distribution": deepcopy(visible_score["distribution"]),
            "p10": visible_boundary_score_p10,
            "issue_count": int(visible_summary["issue_count"]),
            "issue_fraction": visible_issue_fraction,
            "count_basis": INTRODUCED_CROSS_SOURCE_DOMAIN,
            "selection_gate": professional_gate_payload is None,
        },
        "boundary_domains": deepcopy(domain_summaries),
        "boundary_issue_indices": issue_indices,
        "all_boundary_issue_indices": issue_indices,
        "introduced_cross_source_boundary_issue_indices": (
            visible_issue_indices
        ),
        "partial_reedit_boundary_issue_indices": visible_issue_indices,
        "boundaries": boundary_reviews,
        "geometry_provider_failure_count": geometry_failure_count,
        "introduced_cross_source_technical_issue_reasons": (
            introduced_technical_reasons
        ),
        "professional_reference_gate": professional_gate_payload,
        "weighted_score_before_perceptual_source_admission": (
            weighted_score_before_perceptual_source_admission
        ),
        "perceptual_source_admission_penalty_points": (
            perceptual_selection_penalty
        ),
        "weighted_score": weighted_score,
        "passed": passed,
    }
    return RenderedProxyReview(payload)


def render_and_review_proxy(
    plan: RenderPlan,
    proxy_output: Path,
    *,
    config: ProxyReviewConfig | None = None,
    geometry_provider: GeometryProvider | None = None,
    ffmpeg: str = "ffmpeg",
    video_codec: str = "libx264",
    professional_reference_profile: (
        "ProfessionalReferenceProfile | Path | None"
    ) = None,
) -> tuple[RenderPlan, RenderedProxyReview]:
    """Render a 480p derivative with production FFmpeg logic, then review it."""

    proxy_plan = derive_proxy_render_plan(plan, proxy_output)
    codec = render(proxy_plan, ffmpeg=ffmpeg, video_codec=video_codec)
    review = review_rendered_proxy(
        proxy_plan,
        config=config,
        geometry_provider=geometry_provider,
        render_codec=codec,
        professional_reference_profile=professional_reference_profile,
    )
    return (proxy_plan, review)


__all__ = [
    "ANALYSIS_BASIS",
    "BOUNDARY_REVIEW_DOMAINS",
    "BOUNDARY_FRAME_OFFSETS",
    "HUMAN_REFERENCE_BOUNDARY_ANALYSIS_BASIS",
    "INTRODUCED_CROSS_SOURCE_DOMAIN",
    "NATIVE_SOURCE_CUT_DOMAIN",
    "OutputSubjectGeometry",
    "PERCEPTUAL_SOURCE_ADMISSION_ANALYSIS_BASIS",
    "PROXY_LANDSCAPE_SIZE",
    "PROXY_PORTRAIT_SIZE",
    "ProxyReviewConfig",
    "RenderedProxyReview",
    "TIER_D_CONTINUITY_DOMAIN",
    "derive_proxy_render_plan",
    "proxy_dimensions",
    "render_and_review_proxy",
    "review_hard_cut_frame_window",
    "review_rendered_proxy",
]
