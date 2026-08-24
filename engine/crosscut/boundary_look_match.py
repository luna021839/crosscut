"""Boundary-local look matching for professional 16:9 GROUP hard cuts.

Source-wide look normalization cannot account for camera-specific exposure or
color response that changes with one lighting cue.  This module measures a
short decoded window on both sides of each *actual* cross-source hard cut after
accounting for the existing static source look.  Stable, plausible residuals
become small endpoint corrections that ramp smoothly back to identity.

The proxy reviewer remains the authority: this pass never changes its limits,
never marks a boundary as passing, and simply leaves unsafe or incomplete
evidence untouched.  Only scalar measurements and audit data survive; decoded
frames are discarded immediately and never written to disk.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .ive import AspectRatio
from .render import (
    BoundaryLookKeyframe,
    BoundaryLookTrack,
    RenderPlan,
    SourceLookAdjustment,
    TransitionKind,
    segment_frame_counts,
)
from .source_look_match import (
    LookMetricVector,
    LookResidualVector,
    measure_frame_look,
)


ANALYSIS_BASIS = (
    "decoded_synchronized_paired_boundary_residual_post_global_look_match_v2"
)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _mad(values: Sequence[float]) -> float:
    center = _median(values)
    return _median(tuple(abs(value - center) for value in values))


def _metric_values(metric: LookMetricVector) -> tuple[float, float, float, float]:
    return (
        metric.luma,
        metric.contrast,
        metric.saturation,
        metric.temperature_log_red_blue,
    )


@dataclass(frozen=True, slots=True)
class BoundaryLookMatchConfig:
    sample_frame_count: int = 7
    minimum_decoded_samples: int = 5
    ramp_frame_count: int = 7
    endpoint_hold_frame_count: int = 2
    maximum_luma_mad: float = 0.030
    maximum_contrast_mad: float = 0.055
    maximum_saturation_mad: float = 0.075
    maximum_temperature_mad: float = 0.105
    maximum_luma_difference: float = 0.125
    maximum_contrast_log_difference: float = 0.260
    maximum_saturation_log_difference: float = 0.360
    maximum_unrepairable_temperature_difference: float = 0.180
    maximum_partial_temperature_difference: float = 0.420
    maximum_brightness: float = 0.040
    maximum_contrast_delta: float = 0.045
    maximum_saturation_delta: float = 0.055
    # BANG production-chain sweeps found that 0.020 improves both retained
    # color-shock boundaries beyond 0.015 while remaining inside every
    # sharpness/exposure/structure/clipping hard cap.  Values above this stay
    # forbidden even though the renderer's generic static-look type permits
    # them: boundary ramps need the tighter, empirically admitted envelope.
    maximum_temperature_balance: float = 0.020
    correction_fraction: float = 0.84
    minimum_proxy_distance: float = 0.090
    minimum_relative_predicted_reduction: float = 0.16
    brightness_deadband: float = 0.004
    contrast_deadband: float = 0.008
    saturation_deadband: float = 0.012
    temperature_deadband: float = 0.0025

    def __post_init__(self) -> None:
        for name in (
            "sample_frame_count",
            "minimum_decoded_samples",
            "ramp_frame_count",
            "endpoint_hold_frame_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_decoded_samples > self.sample_frame_count:
            raise ValueError("minimum decoded samples exceed the boundary window")
        if self.endpoint_hold_frame_count > self.ramp_frame_count:
            raise ValueError("endpoint hold cannot exceed the look ramp")
        positive = (
            self.maximum_luma_mad,
            self.maximum_contrast_mad,
            self.maximum_saturation_mad,
            self.maximum_temperature_mad,
            self.maximum_luma_difference,
            self.maximum_contrast_log_difference,
            self.maximum_saturation_log_difference,
            self.maximum_unrepairable_temperature_difference,
            self.maximum_partial_temperature_difference,
            self.maximum_brightness,
            self.maximum_contrast_delta,
            self.maximum_saturation_delta,
            self.maximum_temperature_balance,
            self.minimum_proxy_distance,
            self.minimum_relative_predicted_reduction,
            self.brightness_deadband,
            self.contrast_deadband,
            self.saturation_deadband,
            self.temperature_deadband,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("boundary look limits must be positive and finite")
        if not 0.0 < self.correction_fraction <= 1.0:
            raise ValueError("correction fraction must be inside (0, 1]")
        if not 0.0 < self.minimum_relative_predicted_reduction < 1.0:
            raise ValueError("predicted reduction threshold must be normalized")
        if self.maximum_brightness > 0.05:
            raise ValueError("boundary brightness exceeds renderer bounds")
        if self.maximum_contrast_delta > 0.06:
            raise ValueError("boundary contrast exceeds renderer bounds")
        if self.maximum_saturation_delta > 0.06:
            raise ValueError("boundary saturation exceeds renderer bounds")
        if self.maximum_temperature_balance > 0.02:
            raise ValueError("boundary temperature exceeds the chroma-affine bound")
        if (
            self.maximum_partial_temperature_difference
            <= self.maximum_unrepairable_temperature_difference
        ):
            raise ValueError(
                "partial temperature limit must exceed the full-match limit"
            )
        if self.maximum_partial_temperature_difference > 0.50:
            raise ValueError("partial temperature repair bound is too permissive")


@dataclass(frozen=True, slots=True)
class BoundaryLookRenderAdmissionConfig:
    """Fail-closed bounds evaluated on pixels from the production renderer.

    Analysis-domain measurements deliberately propose only small corrections,
    but crop, scale, gamma-domain EQ, chroma subsampling, and encode can change
    their actual effect.  A correction therefore survives only when a matched
    uncorrected/corrected proxy pair proves that every protected component is
    bounded and the combined appearance distance genuinely falls.
    """

    proxy_width: int = 854
    proxy_height: int = 480
    maximum_sharpness_regression: float = 0.012
    maximum_exposure_regression: float = 0.006
    maximum_color_regression: float = 0.010
    maximum_structure_regression: float = 0.008
    maximum_motion_direction_regression: float = 0.040
    maximum_luma_discontinuity_regression: float = 0.012
    maximum_clipped_pixel_ratio_regression: float = 0.010
    minimum_weighted_improvement: float = 0.005

    def __post_init__(self) -> None:
        if (
            isinstance(self.proxy_width, bool)
            or not isinstance(self.proxy_width, int)
            or isinstance(self.proxy_height, bool)
            or not isinstance(self.proxy_height, int)
            or self.proxy_width <= 0
            or self.proxy_height <= 0
            or self.proxy_width % 2
            or self.proxy_height % 2
        ):
            raise ValueError("render-admission proxy dimensions must be even")
        if abs(self.proxy_width / self.proxy_height - 16.0 / 9.0) > 0.002:
            raise ValueError("render-admission proxy must be 16:9")
        for name in (
            "maximum_sharpness_regression",
            "maximum_exposure_regression",
            "maximum_color_regression",
            "maximum_structure_regression",
            "maximum_motion_direction_regression",
            "maximum_luma_discontinuity_regression",
            "maximum_clipped_pixel_ratio_regression",
            "minimum_weighted_improvement",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value <= 0.05:
                raise ValueError(f"{name} must be finite and in (0, 0.05]")

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "analysis_basis": "matched_production_renderer_proxy_pair_v1",
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
            },
        }


@dataclass(frozen=True, slots=True)
class BoundaryLookWindow:
    median: LookMetricVector
    mad: LookMetricVector
    robust_half_range: LookMetricVector
    decoded_sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "decoded_sample_count": self.decoded_sample_count,
            "median": self.median.to_dict(),
            "mad": self.mad.to_dict(),
            "robust_half_range_p10_p90": self.robust_half_range.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BoundaryLookResidualWindow:
    """Same-canonical-time incoming-minus-outgoing camera residuals."""

    median: LookResidualVector
    mad: LookResidualVector
    robust_half_range: LookResidualVector
    paired_sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "paired_sample_count": self.paired_sample_count,
            "median": self.median.to_dict(),
            "mad": self.mad.to_dict(),
            "robust_half_range_p10_p90": self.robust_half_range.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BoundaryLookRenderVerification:
    """Compact evidence from an uncorrected/corrected rendered proxy pair."""

    passed: bool
    reason: str
    sharpness_before: float | None = None
    sharpness_after: float | None = None
    exposure_before: float | None = None
    exposure_after: float | None = None
    color_before: float | None = None
    color_after: float | None = None
    luma_discontinuity_before: float | None = None
    luma_discontinuity_after: float | None = None
    clipped_pixel_ratio_before: float | None = None
    clipped_pixel_ratio_after: float | None = None
    weighted_cost_before: float | None = None
    weighted_cost_after: float | None = None
    technical_integrity_issue_reasons: tuple[str, ...] = ()
    diagnostic_discontinuities_before: tuple[
        tuple[str, float | None], ...
    ] = ()
    diagnostic_discontinuities_after: tuple[
        tuple[str, float | None], ...
    ] = ()
    issue_reasons_before: tuple[str, ...] = ()
    issue_reasons_after: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("render verification reason cannot be blank")
        numeric = tuple(
            getattr(self, name)
            for name in (
                "sharpness_before",
                "sharpness_after",
                "exposure_before",
                "exposure_after",
                "color_before",
                "color_after",
                "luma_discontinuity_before",
                "luma_discontinuity_after",
                "clipped_pixel_ratio_before",
                "clipped_pixel_ratio_after",
                "weighted_cost_before",
                "weighted_cost_after",
            )
        )
        if any(value is not None and not math.isfinite(value) for value in numeric):
            raise ValueError("render verification values must be finite")
        if self.passed and any(value is None for value in numeric):
            raise ValueError("passing render verification needs complete evidence")
        if self.passed and self.technical_integrity_issue_reasons:
            raise ValueError("passing render verification cannot have technical issues")

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_basis": "matched_production_renderer_proxy_pair_v1",
            "passed": self.passed,
            "reason": self.reason,
            "discontinuities": {
                "sharpness": {
                    "before": self.sharpness_before,
                    "after": self.sharpness_after,
                },
                "exposure": {
                    "before": self.exposure_before,
                    "after": self.exposure_after,
                },
                "color": {
                    "before": self.color_before,
                    "after": self.color_after,
                },
                "mean_luma": {
                    "before": self.luma_discontinuity_before,
                    "after": self.luma_discontinuity_after,
                },
                "clipped_pixel_ratio": {
                    "before": self.clipped_pixel_ratio_before,
                    "after": self.clipped_pixel_ratio_after,
                },
            },
            "weighted_cost": {
                "before": self.weighted_cost_before,
                "after": self.weighted_cost_after,
            },
            "technical_integrity_issue_reasons": list(
                self.technical_integrity_issue_reasons
            ),
            "all_diagnostic_discontinuities": {
                "before": dict(self.diagnostic_discontinuities_before),
                "after": dict(self.diagnostic_discontinuities_after),
            },
            "qa_issue_reasons": {
                "before": list(self.issue_reasons_before),
                "after": list(self.issue_reasons_after),
            },
        }


@dataclass(frozen=True, slots=True)
class BoundaryLookDecision:
    boundary_index: int
    outgoing_source_index: int
    incoming_source_index: int
    eligible: bool
    applied: bool
    reason: str
    outgoing_window: BoundaryLookWindow | None = None
    incoming_window: BoundaryLookWindow | None = None
    synchronized_residual_window: BoundaryLookResidualWindow | None = None
    outgoing_adjustment: SourceLookAdjustment = SourceLookAdjustment()
    incoming_adjustment: SourceLookAdjustment = SourceLookAdjustment()
    predicted_distance_before: float | None = None
    predicted_distance_after: float | None = None
    render_verification: BoundaryLookRenderVerification | None = None

    def to_dict(self) -> dict[str, Any]:
        def adjustment_payload(value: SourceLookAdjustment) -> dict[str, Any]:
            return {
                "brightness": value.brightness,
                "contrast": value.contrast,
                "saturation": value.saturation,
                "temperature_balance": value.temperature_balance,
                "identity": value.is_identity,
            }

        return {
            "boundary_index": self.boundary_index,
            "outgoing_source_index": self.outgoing_source_index,
            "incoming_source_index": self.incoming_source_index,
            "eligible": self.eligible,
            "applied": self.applied,
            "reason": self.reason,
            "outgoing_window": (
                self.outgoing_window.to_dict()
                if self.outgoing_window is not None
                else None
            ),
            "incoming_window": (
                self.incoming_window.to_dict()
                if self.incoming_window is not None
                else None
            ),
            "synchronized_residual_window": (
                self.synchronized_residual_window.to_dict()
                if self.synchronized_residual_window is not None
                else None
            ),
            "outgoing_adjustment": adjustment_payload(
                self.outgoing_adjustment
            ),
            "incoming_adjustment": adjustment_payload(
                self.incoming_adjustment
            ),
            "predicted_distance_before": self.predicted_distance_before,
            "predicted_distance_after": self.predicted_distance_after,
            "render_verification": (
                self.render_verification.to_dict()
                if self.render_verification is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class BoundaryLookMatchResult:
    enabled: bool
    blocked_reason: str | None
    decisions: tuple[BoundaryLookDecision, ...]
    segment_tracks: tuple[BoundaryLookTrack | None, ...]
    decoded_frame_count: int
    cache_hit_count: int
    analysis_config: BoundaryLookMatchConfig = BoundaryLookMatchConfig()
    render_admission_config: BoundaryLookRenderAdmissionConfig | None = None
    render_admission_encode_settings: tuple[tuple[str, str], ...] = ()
    render_admission_encode_settings_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.enabled and self.blocked_reason is not None:
            raise ValueError("enabled boundary look result cannot be blocked")
        if not self.enabled and self.blocked_reason is None:
            raise ValueError("disabled boundary look result needs a reason")

    @property
    def eligible_count(self) -> int:
        return sum(item.eligible for item in self.decisions)

    @property
    def applied_count(self) -> int:
        return sum(item.applied for item in self.decisions)

    @property
    def skipped_count(self) -> int:
        return len(self.decisions) - self.applied_count

    @property
    def render_verified_count(self) -> int:
        return sum(
            item.render_verification is not None
            and item.render_verification.passed
            for item in self.decisions
        )

    @property
    def render_rejected_count(self) -> int:
        return sum(
            item.render_verification is not None
            and not item.render_verification.passed
            for item in self.decisions
        )

    def apply(self, plan: RenderPlan) -> RenderPlan:
        if len(self.segment_tracks) != len(plan.segments):
            raise ValueError("boundary look tracks do not match render segments")
        if not self.enabled:
            return plan
        return replace(
            plan,
            segments=tuple(
                replace(segment, boundary_look_track=track)
                for segment, track in zip(
                    plan.segments,
                    self.segment_tracks,
                    strict=True,
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        before_issue_reasons: Counter[str] = Counter()
        final_issue_reasons: Counter[str] = Counter()
        before_issue_count = 0
        final_issue_count = 0
        for decision in self.decisions:
            verification = decision.render_verification
            if verification is None:
                continue
            before = verification.issue_reasons_before
            final = (
                verification.issue_reasons_after
                if verification.passed
                else verification.issue_reasons_before
            )
            before_issue_count += int(bool(before))
            final_issue_count += int(bool(final))
            before_issue_reasons.update(before)
            final_issue_reasons.update(final)
        return {
            "analysis_basis": ANALYSIS_BASIS,
            "enabled": self.enabled,
            "blocked_reason": self.blocked_reason,
            "eligible_boundary_count": self.eligible_count,
            "applied_boundary_count": self.applied_count,
            "skipped_boundary_count": self.skipped_count,
            "decoded_frame_count": self.decoded_frame_count,
            "in_memory_scalar_cache_hit_count": self.cache_hit_count,
            "decisions": [item.to_dict() for item in self.decisions],
            "segment_tracks": [
                None
                if track is None
                else {
                    "reason": track.reason,
                    "keyframes": [
                        {
                            "time_seconds": keyframe.time_seconds,
                            "brightness": keyframe.brightness,
                            "contrast": keyframe.contrast,
                            "saturation": keyframe.saturation,
                            "temperature_balance": (
                                keyframe.temperature_balance
                            ),
                        }
                        for keyframe in track.keyframes
                    ],
                }
                for track in self.segment_tracks
            ],
            "static_source_look_accounted_for_before_residual": True,
            "proxy_thresholds_modified": False,
            "raw_frames_persisted": False,
            "persistent_feature_cache_used": False,
            "temperature_ramps_used": any(
                track is not None
                and any(
                    abs(keyframe.temperature_balance) > 1e-12
                    for keyframe in track.keyframes
                )
                for track in self.segment_tracks
            ),
            "same_canonical_time_pairing_used": True,
            "volatile_shared_stage_lighting_cancels_in_residual_domain": True,
            "temperature_residual_policy": (
                "bounded_luma_preserving_yuv_chroma_affine_ramp_plus_"
                "matched_render_domain_admission"
            ),
            "neutral_temperature_frames_sample_exact_yuv": True,
            "temperature_residual_never_modified": not any(
                track is not None
                and any(
                    abs(keyframe.temperature_balance) > 1e-12
                    for keyframe in track.keyframes
                )
                for track in self.segment_tracks
            ),
            "render_domain_admission": {
                "enabled": self.render_admission_config is not None,
                "config": (
                    self.render_admission_config.to_dict()
                    if self.render_admission_config is not None
                    else None
                ),
                "verified_applied_count": self.render_verified_count,
                "rejected_proposal_count": self.render_rejected_count,
                "matched_base_and_adjusted_geometry_codec": True,
                "each_component_has_hard_regression_cap": True,
                "technical_integrity_must_have_zero_issues": True,
                "verified_technical_integrity_issue_count": sum(
                    len(item.render_verification.technical_integrity_issue_reasons)
                    for item in self.decisions
                    if item.render_verification is not None
                ),
                "eligible_final_noop_boundary_count": (
                    self.eligible_count - self.applied_count
                ),
                "verified_applied_boundary_indices": [
                    item.boundary_index
                    for item in self.decisions
                    if item.render_verification is not None
                    and item.render_verification.passed
                ],
                "final_noop_boundary_indices": [
                    item.boundary_index
                    for item in self.decisions
                    if item.eligible and not item.applied
                ],
                "final_noop_review_windows_parameter_identical_to_base": True,
                "neighbor_window_identity_contract": (
                    "two_frame_endpoint_hold_keeps_unselected_minus2_minus1_"
                    "plus0_plus1_review_windows_at_identity"
                ),
                "qa_issue_boundary_count_before": before_issue_count,
                "qa_issue_boundary_count_final": final_issue_count,
                "qa_issue_boundary_count_delta": (
                    final_issue_count - before_issue_count
                ),
                "qa_issue_reason_count_before": dict(before_issue_reasons),
                "qa_issue_reason_count_final": dict(final_issue_reasons),
                "qa_issue_reason_count_delta": {
                    reason: final_issue_reasons[reason]
                    - before_issue_reasons[reason]
                    for reason in sorted(
                        set(before_issue_reasons) | set(final_issue_reasons)
                    )
                    if final_issue_reasons[reason]
                    != before_issue_reasons[reason]
                },
                "encode_settings": dict(
                    self.render_admission_encode_settings
                ),
                "encode_settings_sha256": (
                    self.render_admission_encode_settings_sha256
                ),
                "temporary_proxy_video_retained": False,
                "raw_frames_serialized": False,
            },
        }


@dataclass(slots=True)
class BoundaryLookMeasurementCache:
    """Ephemeral scalar-only cache shared by the eight candidate plans."""

    _metrics: dict[tuple[str, int], LookMetricVector | None] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    decoded_frame_count: int = 0
    cache_hit_count: int = 0

    @staticmethod
    def _key(path: Path, time_seconds: float) -> tuple[str, int]:
        return (
            str(Path(path).resolve()),
            int(round(float(time_seconds) * 1_000_000.0)),
        )

    def measure_many(
        self,
        path: Path,
        times_seconds: Iterable[float],
    ) -> dict[int, LookMetricVector | None]:
        requested = tuple(
            sorted(
                {
                    int(round(float(value) * 1_000_000.0))
                    for value in times_seconds
                    if math.isfinite(float(value)) and float(value) >= 0.0
                }
            )
        )
        resolved = str(Path(path).resolve())
        missing = tuple(
            microseconds
            for microseconds in requested
            if (resolved, microseconds) not in self._metrics
        )
        self.cache_hit_count += len(requested) - len(missing)
        if missing:
            capture: Any | None = None
            try:
                source = Path(path)
                if not source.is_file() or source.stat().st_size < 1024:
                    raise ValueError("boundary look source is missing or empty")
                capture = cv2.VideoCapture(str(source))
                if not capture.isOpened():
                    raise ValueError("could not open boundary look source")
                for microseconds in missing:
                    metric: LookMetricVector | None = None
                    try:
                        capture.set(
                            cv2.CAP_PROP_POS_MSEC,
                            microseconds / 1000.0,
                        )
                        success, frame = capture.read()
                        if (
                            success
                            and isinstance(frame, np.ndarray)
                            and frame.size > 0
                        ):
                            height, width = frame.shape[:2]
                            if (
                                height > 0
                                and abs(width / height - 16.0 / 9.0) <= 0.03
                            ):
                                metric = measure_frame_look(frame)
                                self.decoded_frame_count += 1
                    except (
                        cv2.error,
                        FloatingPointError,
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ):
                        metric = None
                    self._metrics[(resolved, microseconds)] = metric
            except (
                cv2.error,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                for microseconds in missing:
                    self._metrics[(resolved, microseconds)] = None
            finally:
                if capture is not None:
                    try:
                        capture.release()
                    except (cv2.error, RuntimeError):
                        pass
        return {
            microseconds: self._metrics[(resolved, microseconds)]
            for microseconds in requested
        }


def _after_static_source_look(
    metric: LookMetricVector,
    adjustment: SourceLookAdjustment | None,
) -> LookMetricVector:
    """Approximate the static FFmpeg layer in metric space, exactly once."""

    value = SourceLookAdjustment() if adjustment is None else adjustment
    luma = _clip(
        (metric.luma - 0.5) * value.contrast
        + 0.5
        + value.brightness,
        0.0,
        1.0,
    )
    return LookMetricVector(
        luma=luma,
        contrast=_clip(metric.contrast * value.contrast, 0.0, 1.0),
        saturation=_clip(metric.saturation * value.saturation, 0.0, 1.0),
        # ``colorbalance`` applies opposing red/blue shifts.  The factor is a
        # conservative first-order approximation used only as a safety gate;
        # boundary-local rendering never adds a second temperature filter.
        temperature_log_red_blue=_clip(
            metric.temperature_log_red_blue
            + 2.0 * value.temperature_balance,
            -2.0,
            2.0,
        ),
    )


def _aggregate_window(
    metrics: Sequence[LookMetricVector],
) -> BoundaryLookWindow:
    values = tuple(_metric_values(metric) for metric in metrics)
    median_values = tuple(
        _median(tuple(value[index] for value in values)) for index in range(4)
    )
    mad_values = tuple(
        _mad(tuple(value[index] for value in values)) for index in range(4)
    )
    half_range_values = tuple(
        float(
            (
                np.percentile(
                    np.asarray(
                        tuple(value[index] for value in values),
                        dtype=np.float64,
                    ),
                    90.0,
                )
                - np.percentile(
                    np.asarray(
                        tuple(value[index] for value in values),
                        dtype=np.float64,
                    ),
                    10.0,
                )
            )
            / 2.0
        )
        for index in range(4)
    )
    return BoundaryLookWindow(
        median=LookMetricVector(*median_values),
        mad=LookMetricVector(*mad_values),
        robust_half_range=LookMetricVector(*half_range_values),
        decoded_sample_count=len(metrics),
    )


def _paired_residual(
    left: LookMetricVector,
    right: LookMetricVector,
) -> tuple[float, float, float, float]:
    return (
        right.luma - left.luma,
        math.log(max(0.03, right.contrast) / max(0.03, left.contrast)),
        math.log(max(0.04, right.saturation) / max(0.04, left.saturation)),
        right.temperature_log_red_blue - left.temperature_log_red_blue,
    )


def _residual_vector(
    values: Sequence[tuple[float, float, float, float]],
    reducer: Any,
) -> LookResidualVector:
    return LookResidualVector(
        luma_delta=reducer(tuple(value[0] for value in values)),
        contrast_log_ratio=reducer(tuple(value[1] for value in values)),
        saturation_log_ratio=reducer(tuple(value[2] for value in values)),
        temperature_log_red_blue_delta=reducer(
            tuple(value[3] for value in values)
        ),
    )


def _aggregate_residual_window(
    values: Sequence[tuple[float, float, float, float]],
) -> BoundaryLookResidualWindow:
    def half_range(samples: Sequence[float]) -> float:
        array = np.asarray(samples, dtype=np.float64)
        return float((np.percentile(array, 90.0) - np.percentile(array, 10.0)) / 2.0)

    return BoundaryLookResidualWindow(
        median=_residual_vector(values, _median),
        mad=_residual_vector(values, _mad),
        robust_half_range=_residual_vector(values, half_range),
        paired_sample_count=len(values),
    )


def _residual_window_is_stable(
    window: BoundaryLookResidualWindow,
    config: BoundaryLookMatchConfig,
) -> bool:
    return (
        max(
            abs(window.mad.luma_delta),
            abs(window.robust_half_range.luma_delta),
        )
        <= config.maximum_luma_mad
        and max(
            abs(window.mad.contrast_log_ratio),
            abs(window.robust_half_range.contrast_log_ratio),
        )
        <= config.maximum_contrast_mad
        and max(
            abs(window.mad.saturation_log_ratio),
            abs(window.robust_half_range.saturation_log_ratio),
        )
        <= config.maximum_saturation_mad
        and max(
            abs(window.mad.temperature_log_red_blue_delta),
            abs(window.robust_half_range.temperature_log_red_blue_delta),
        )
        <= config.maximum_temperature_mad
    )


def _apply_local_metric(
    metric: LookMetricVector,
    adjustment: SourceLookAdjustment,
) -> LookMetricVector:
    return LookMetricVector(
        luma=_clip(metric.luma + adjustment.brightness, 0.0, 1.0),
        contrast=_clip(metric.contrast * adjustment.contrast, 0.0, 1.0),
        saturation=_clip(metric.saturation * adjustment.saturation, 0.0, 1.0),
        temperature_log_red_blue=metric.temperature_log_red_blue,
    )


def _proxy_distance(left: LookMetricVector, right: LookMetricVector) -> float:
    """Scalar prediction used only to decide whether a ramp is worthwhile."""

    return _proxy_distance_from_residual(_paired_residual(left, right))


def _proxy_distance_from_residual(
    residual: tuple[float, float, float, float],
) -> float:
    luma = abs(residual[0]) / 0.18
    contrast = abs(residual[1]) / 0.30
    saturation = abs(residual[2]) / 0.36
    temperature = abs(residual[3]) / 0.28
    return _clip(
        0.48 * luma
        + 0.19 * contrast
        + 0.25 * saturation
        + 0.08 * temperature,
        0.0,
        1.0,
    )


def _deadband(value: float, identity: float, tolerance: float) -> float:
    return identity if abs(value - identity) < tolerance else value


def _derive_endpoint_adjustments(
    left: LookMetricVector,
    right: LookMetricVector,
    config: BoundaryLookMatchConfig,
) -> tuple[SourceLookAdjustment, SourceLookAdjustment]:
    return _derive_endpoint_adjustments_from_residual(
        _paired_residual(left, right),
        config,
    )


def _derive_endpoint_adjustments_from_residual(
    residual: tuple[float, float, float, float],
    config: BoundaryLookMatchConfig,
) -> tuple[SourceLookAdjustment, SourceLookAdjustment]:
    half = 0.5 * config.correction_fraction
    luma_delta, contrast_delta, saturation_delta, temperature_delta = residual
    outgoing_brightness = _deadband(
        _clip(
            half * luma_delta,
            -config.maximum_brightness,
            config.maximum_brightness,
        ),
        0.0,
        config.brightness_deadband,
    )
    incoming_brightness = _deadband(
        _clip(
            -half * luma_delta,
            -config.maximum_brightness,
            config.maximum_brightness,
        ),
        0.0,
        config.brightness_deadband,
    )
    outgoing_contrast = _deadband(
        _clip(
            math.exp(half * contrast_delta),
            1.0 - config.maximum_contrast_delta,
            1.0 + config.maximum_contrast_delta,
        ),
        1.0,
        config.contrast_deadband,
    )
    incoming_contrast = _deadband(
        _clip(
            math.exp(-half * contrast_delta),
            1.0 - config.maximum_contrast_delta,
            1.0 + config.maximum_contrast_delta,
        ),
        1.0,
        config.contrast_deadband,
    )
    outgoing_saturation = _deadband(
        _clip(
            math.exp(half * saturation_delta),
            1.0 - config.maximum_saturation_delta,
            1.0 + config.maximum_saturation_delta,
        ),
        1.0,
        config.saturation_deadband,
    )
    incoming_saturation = _deadband(
        _clip(
            math.exp(-half * saturation_delta),
            1.0 - config.maximum_saturation_delta,
            1.0 + config.maximum_saturation_delta,
        ),
        1.0,
        config.saturation_deadband,
    )
    # ``measure_frame_look`` expresses temperature as log(red/blue).  The
    # renderer's chroma-affine ramp moves red and blue contributions in
    # opposite directions.  The exact log(red/blue) effect depends on scene
    # luminance, so this analytic proxy deliberately credits only twice the
    # numeric balance.  Splitting the requested repair across both cameras
    # introduces the 0.25 factor below.  Pixel-domain admission remains
    # authoritative and rejects any scene where the approximation regresses.
    temperature_half = 0.25 * config.correction_fraction
    outgoing_temperature = _deadband(
        _clip(
            temperature_half * temperature_delta,
            -config.maximum_temperature_balance,
            config.maximum_temperature_balance,
        ),
        0.0,
        config.temperature_deadband,
    )
    incoming_temperature = _deadband(
        _clip(
            -temperature_half * temperature_delta,
            -config.maximum_temperature_balance,
            config.maximum_temperature_balance,
        ),
        0.0,
        config.temperature_deadband,
    )
    return (
        SourceLookAdjustment(
            brightness=outgoing_brightness,
            contrast=outgoing_contrast,
            saturation=outgoing_saturation,
            temperature_balance=outgoing_temperature,
        ),
        SourceLookAdjustment(
            brightness=incoming_brightness,
            contrast=incoming_contrast,
            saturation=incoming_saturation,
            temperature_balance=incoming_temperature,
        ),
    )


def _residual_after_adjustments(
    residual: tuple[float, float, float, float],
    outgoing: SourceLookAdjustment,
    incoming: SourceLookAdjustment,
) -> tuple[float, float, float, float]:
    return (
        residual[0] + incoming.brightness - outgoing.brightness,
        residual[1] + math.log(incoming.contrast / outgoing.contrast),
        residual[2] + math.log(incoming.saturation / outgoing.saturation),
        residual[3]
        + 2.0 * incoming.temperature_balance
        - 2.0 * outgoing.temperature_balance,
    )


def _keyframe(
    time_seconds: float,
    adjustment: SourceLookAdjustment,
) -> BoundaryLookKeyframe:
    return BoundaryLookKeyframe(
        time_seconds=time_seconds,
        brightness=adjustment.brightness,
        contrast=adjustment.contrast,
        saturation=adjustment.saturation,
        temperature_balance=adjustment.temperature_balance,
    )


def _track_for_segment(
    *,
    frame_count: int,
    duration: float,
    fps: float,
    incoming: SourceLookAdjustment,
    outgoing: SourceLookAdjustment,
    config: BoundaryLookMatchConfig,
) -> BoundaryLookTrack | None:
    if incoming.is_identity and outgoing.is_identity:
        return None
    if frame_count < 2:
        return None
    end = min(duration, (frame_count - 1) / fps)
    frame = 1.0 / fps
    ramp = config.ramp_frame_count * frame
    hold = max(0, config.endpoint_hold_frame_count - 1) * frame
    identity = SourceLookAdjustment()
    keyframes: list[BoundaryLookKeyframe] = [_keyframe(0.0, incoming)]

    # Long shots briefly repair each endpoint and retain their native lighting
    # through the interior.  Short shots keep the exact first/last two-frame
    # endpoint holds and compress only the interpolation between them.  Thus a
    # correction at one boundary cannot leak into the four-frame QA window of
    # the unrelated boundary on the other side of the shot.
    if end > 2.0 * ramp + 2.0 * hold + 1e-9:
        if hold > 1e-9:
            keyframes.append(_keyframe(hold, incoming))
        keyframes.append(_keyframe(ramp, identity))
        keyframes.append(_keyframe(end - ramp, identity))
        if hold > 1e-9:
            keyframes.append(_keyframe(end - hold, outgoing))
        keyframes.append(_keyframe(end, outgoing))
    else:
        if hold > 1e-9:
            keyframes.append(_keyframe(min(hold, end), incoming))
        outgoing_hold_start = max(hold, end - hold)
        if outgoing_hold_start > hold + 1e-9:
            keyframes.append(_keyframe(outgoing_hold_start, outgoing))
        keyframes.append(_keyframe(end, outgoing))

    # Consecutive duplicate timestamps can arise only through very low fps or
    # caller-supplied configuration; collapse them deterministically.
    deduplicated: list[BoundaryLookKeyframe] = []
    for item in keyframes:
        if deduplicated and item.time_seconds <= deduplicated[-1].time_seconds + 1e-12:
            deduplicated[-1] = item
        else:
            deduplicated.append(item)
    if not deduplicated or all(item.is_identity for item in deduplicated):
        return None
    return BoundaryLookTrack(tuple(deduplicated))


def _tracks_from_decisions(
    plan: RenderPlan,
    decisions: Sequence[BoundaryLookDecision],
    config: BoundaryLookMatchConfig,
) -> tuple[BoundaryLookTrack | None, ...]:
    endpoint_adjustments: dict[int, dict[str, SourceLookAdjustment]] = {}
    for decision in decisions:
        if not decision.applied:
            continue
        endpoint_adjustments.setdefault(decision.boundary_index, {})[
            "outgoing"
        ] = decision.outgoing_adjustment
        endpoint_adjustments.setdefault(decision.boundary_index + 1, {})[
            "incoming"
        ] = decision.incoming_adjustment
    counts = segment_frame_counts(plan)
    return tuple(
        _track_for_segment(
            frame_count=counts[index],
            duration=plan.segments[index].duration,
            fps=plan.fps,
            incoming=endpoint_adjustments.get(index, {}).get(
                "incoming", SourceLookAdjustment()
            ),
            outgoing=endpoint_adjustments.get(index, {}).get(
                "outgoing", SourceLookAdjustment()
            ),
            config=config,
        )
        for index in range(len(plan.segments))
    )


def _boundary_pair_render_plan(
    plan: RenderPlan,
    decision: BoundaryLookDecision,
    *,
    output: Path,
    adjusted: bool,
    config: BoundaryLookMatchConfig,
    admission: BoundaryLookRenderAdmissionConfig,
) -> RenderPlan:
    index = decision.boundary_index
    left = plan.segments[index]
    right = plan.segments[index + 1]
    counts = segment_frame_counts(plan)
    left_track = (
        _track_for_segment(
            frame_count=counts[index],
            duration=left.duration,
            fps=plan.fps,
            incoming=SourceLookAdjustment(),
            outgoing=decision.outgoing_adjustment,
            config=config,
        )
        if adjusted
        else None
    )
    right_track = (
        _track_for_segment(
            frame_count=counts[index + 1],
            duration=right.duration,
            fps=plan.fps,
            incoming=decision.incoming_adjustment,
            outgoing=SourceLookAdjustment(),
            config=config,
        )
        if adjusted
        else None
    )
    left = replace(left, master_start=0.0, boundary_look_track=left_track)
    right = replace(
        right,
        master_start=left.duration,
        boundary_look_track=right_track,
    )
    return RenderPlan(
        sources=plan.sources,
        segments=(left, right),
        reference_source=plan.reference_source,
        aspect_ratio=plan.aspect_ratio,
        output=output,
        master_audio_start=plan.master_audio_start
        + plan.segments[index].master_start,
        width=admission.proxy_width,
        height=admission.proxy_height,
        fps=plan.fps,
        transitions=(plan.transitions[index],),
    )


def _decode_hard_cut_review(
    plan: RenderPlan,
    cut_frame_index: int,
) -> Mapping[str, Any]:
    from .rendered_proxy_review import (
        OutputSubjectGeometry,
        review_hard_cut_frame_window,
    )

    required = set(cut_frame_index + offset for offset in (-2, -1, 0, 1))
    if min(required) < 0:
        raise ValueError("render admission boundary has no outgoing handle")
    capture = cv2.VideoCapture(str(plan.output))
    if not capture.isOpened():
        raise ValueError("could not decode render-admission proxy")
    frames: dict[int, np.ndarray] = {}
    frame_index = 0
    try:
        while frame_index <= max(required):
            success, frame = capture.read()
            if not success or frame is None:
                break
            if frame_index in required:
                frames[frame_index] = frame.copy()
            frame_index += 1
    finally:
        capture.release()
    if set(frames) != required:
        raise ValueError("render-admission proxy lacks a complete cut window")
    return review_hard_cut_frame_window(
        frames,
        cut_frame_index=cut_frame_index,
        geometry_provider=lambda _frame: OutputSubjectGeometry(),
    )


def _render_and_review_boundary(
    plan: RenderPlan,
    cut_frame_index: int,
    *,
    ffmpeg: str,
    video_codec: str,
) -> Mapping[str, Any]:
    from .render import render

    render(plan, ffmpeg=ffmpeg, video_codec=video_codec)
    return _decode_hard_cut_review(plan, cut_frame_index)


def _review_scalar(
    review: Mapping[str, Any],
    *,
    offset: int,
    name: str,
) -> float:
    for item in review.get("frame_metrics", ()):  # type: ignore[union-attr]
        if int(item.get("offset", 999)) == offset:
            value = float(item[name])
            if math.isfinite(value):
                return value
    raise ValueError(f"render review lacks finite {name} at offset {offset}")


def _verification_from_reviews(
    before_review: Mapping[str, Any],
    after_review: Mapping[str, Any],
    config: BoundaryLookRenderAdmissionConfig,
) -> BoundaryLookRenderVerification:
    protected = (
        "sharpness_discontinuity",
        "exposure_discontinuity",
        "color_shock",
    )
    before_discontinuities = before_review.get("discontinuities")
    after_discontinuities = after_review.get("discontinuities")
    if not isinstance(before_discontinuities, Mapping) or not isinstance(
        after_discontinuities, Mapping
    ):
        raise ValueError("render review lacks discontinuity evidence")
    before_values = tuple(float(before_discontinuities[name]) for name in protected)
    after_values = tuple(float(after_discontinuities[name]) for name in protected)
    if any(not math.isfinite(value) for value in before_values + after_values):
        raise ValueError("render review discontinuity is not finite")

    before_luma = abs(
        _review_scalar(before_review, offset=-1, name="mean_luma")
        - _review_scalar(before_review, offset=0, name="mean_luma")
    )
    after_luma = abs(
        _review_scalar(after_review, offset=-1, name="mean_luma")
        - _review_scalar(after_review, offset=0, name="mean_luma")
    )
    before_clipped = max(
        _review_scalar(before_review, offset=offset, name="clipped_pixel_ratio")
        for offset in (-2, -1, 0, 1)
    )
    after_clipped = max(
        _review_scalar(after_review, offset=offset, name="clipped_pixel_ratio")
        for offset in (-2, -1, 0, 1)
    )
    before_cost = (
        0.20 * before_values[0] / 0.48
        + 0.35 * before_values[1] / 0.24
        + 0.45 * before_values[2] / 0.48
    )
    after_cost = (
        0.20 * after_values[0] / 0.48
        + 0.35 * after_values[1] / 0.24
        + 0.45 * after_values[2] / 0.48
    )
    technical = tuple(
        sorted(
            {
                f"before:{reason}"
                for reason in before_review.get(
                    "technical_integrity_issue_reasons", ()
                )
            }
            | {
                f"after:{reason}"
                for reason in after_review.get(
                    "technical_integrity_issue_reasons", ()
                )
            }
        )
    )
    regression_reasons: list[str] = []
    for name, before, after, maximum in (
        (
            "sharpness",
            before_values[0],
            after_values[0],
            config.maximum_sharpness_regression,
        ),
        (
            "exposure",
            before_values[1],
            after_values[1],
            config.maximum_exposure_regression,
        ),
        (
            "color",
            before_values[2],
            after_values[2],
            config.maximum_color_regression,
        ),
        (
            "mean_luma",
            before_luma,
            after_luma,
            config.maximum_luma_discontinuity_regression,
        ),
        (
            "clipped_pixels",
            before_clipped,
            after_clipped,
            config.maximum_clipped_pixel_ratio_regression,
        ),
    ):
        if after > before + maximum + 1e-12:
            regression_reasons.append(name)
    for name, maximum in (
        ("structure_shock", config.maximum_structure_regression),
        (
            "motion_direction_discontinuity",
            config.maximum_motion_direction_regression,
        ),
    ):
        before_raw = before_discontinuities.get(name)
        after_raw = after_discontinuities.get(name)
        if before_raw is None or after_raw is None:
            continue
        before_numeric = float(before_raw)
        after_numeric = float(after_raw)
        if not math.isfinite(before_numeric) or not math.isfinite(after_numeric):
            raise ValueError("render structural diagnostic is not finite")
        if after_numeric > before_numeric + maximum + 1e-12:
            regression_reasons.append(name)
    if technical:
        passed = False
        reason = "technical_integrity_issue"
    elif regression_reasons:
        passed = False
        reason = "hard_component_regression:" + ",".join(regression_reasons)
    elif before_cost - after_cost < config.minimum_weighted_improvement:
        passed = False
        reason = "weighted_improvement_below_minimum"
    else:
        passed = True
        reason = "matched_render_proxy_improved_without_component_regression"

    def diagnostic_items(
        values: Mapping[str, Any],
    ) -> tuple[tuple[str, float | None], ...]:
        serialized: list[tuple[str, float | None]] = []
        for name in sorted(values):
            raw_value = values[name]
            if raw_value is None:
                serialized.append((str(name), None))
                continue
            numeric = float(raw_value)
            if not math.isfinite(numeric):
                raise ValueError("render diagnostic discontinuity is not finite")
            serialized.append((str(name), numeric))
        return tuple(serialized)

    return BoundaryLookRenderVerification(
        passed=passed,
        reason=reason,
        sharpness_before=before_values[0],
        sharpness_after=after_values[0],
        exposure_before=before_values[1],
        exposure_after=after_values[1],
        color_before=before_values[2],
        color_after=after_values[2],
        luma_discontinuity_before=before_luma,
        luma_discontinuity_after=after_luma,
        clipped_pixel_ratio_before=before_clipped,
        clipped_pixel_ratio_after=after_clipped,
        weighted_cost_before=before_cost,
        weighted_cost_after=after_cost,
        technical_integrity_issue_reasons=technical,
        diagnostic_discontinuities_before=diagnostic_items(
            before_discontinuities
        ),
        diagnostic_discontinuities_after=diagnostic_items(
            after_discontinuities
        ),
        issue_reasons_before=tuple(
            str(reason) for reason in before_review.get("issue_reasons", ())
        ),
        issue_reasons_after=tuple(
            str(reason) for reason in after_review.get("issue_reasons", ())
        ),
    )


BoundaryRenderReviewFunction = Callable[
    [RenderPlan, int], Mapping[str, Any]
]


def _render_admission_encode_settings(
    plan: RenderPlan,
    config: BoundaryLookRenderAdmissionConfig,
    video_codec: str,
) -> tuple[tuple[tuple[str, str], ...], str]:
    values = {
        "codec": video_codec,
        "crf": "16",
        "fps": f"{plan.fps:.9f}",
        "height": str(config.proxy_height),
        "look_filter_contract": "eq_v1+geq_yuv_chroma_affine_temperature_v1",
        "pixel_format": "yuv420p",
        "preset": "slow",
        "profile": "high",
        "renderer": "crosscut.render.render",
        "width": str(config.proxy_width),
    }
    serialized = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return tuple(sorted(values.items())), hashlib.sha256(serialized).hexdigest()


def verify_boundary_look_match_render_domain(
    plan: RenderPlan,
    result: BoundaryLookMatchResult,
    *,
    config: BoundaryLookRenderAdmissionConfig | None = None,
    ffmpeg: str = "ffmpeg",
    video_codec: str = "libx264",
    review_function: BoundaryRenderReviewFunction | None = None,
) -> BoundaryLookMatchResult:
    """Retain a proposed correction only after matched proxy render proof."""

    if len(result.segment_tracks) != len(plan.segments):
        raise ValueError("boundary look result does not match render plan")
    settings = BoundaryLookRenderAdmissionConfig() if config is None else config
    if video_codec != "libx264":
        raise ValueError("render admission requires deterministic libx264")
    encode_settings, encode_settings_sha256 = _render_admission_encode_settings(
        plan,
        settings,
        video_codec,
    )
    if not result.enabled:
        return replace(
            result,
            render_admission_config=settings,
            render_admission_encode_settings=encode_settings,
            render_admission_encode_settings_sha256=encode_settings_sha256,
        )
    output_parent = plan.output.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    reviewer = review_function
    decisions: list[BoundaryLookDecision] = []
    with tempfile.TemporaryDirectory(
        prefix=".boundary-look-admission-", dir=output_parent
    ) as temporary_directory:
        root = Path(temporary_directory)
        for decision in result.decisions:
            if not decision.applied:
                decisions.append(decision)
                continue
            try:
                base_plan = _boundary_pair_render_plan(
                    plan,
                    decision,
                    output=root / f"boundary-{decision.boundary_index}-base.mp4",
                    adjusted=False,
                    config=result.analysis_config,
                    admission=settings,
                )
                adjusted_plan = _boundary_pair_render_plan(
                    plan,
                    decision,
                    output=root
                    / f"boundary-{decision.boundary_index}-adjusted.mp4",
                    adjusted=True,
                    config=result.analysis_config,
                    admission=settings,
                )
                cut_frame_index = segment_frame_counts(base_plan)[0]
                if reviewer is None:
                    before_review = _render_and_review_boundary(
                        base_plan,
                        cut_frame_index,
                        ffmpeg=ffmpeg,
                        video_codec=video_codec,
                    )
                    after_review = _render_and_review_boundary(
                        adjusted_plan,
                        cut_frame_index,
                        ffmpeg=ffmpeg,
                        video_codec=video_codec,
                    )
                else:
                    before_review = reviewer(base_plan, cut_frame_index)
                    after_review = reviewer(adjusted_plan, cut_frame_index)
                verification = _verification_from_reviews(
                    before_review,
                    after_review,
                    settings,
                )
            except (
                cv2.error,
                KeyError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                verification = BoundaryLookRenderVerification(
                    passed=False,
                    reason=f"render_verification_error:{type(exc).__name__}",
                )
            if verification.passed:
                decisions.append(
                    replace(decision, render_verification=verification)
                )
            else:
                decisions.append(
                    replace(
                        decision,
                        applied=False,
                        reason="render_domain_rejected:" + verification.reason,
                        outgoing_adjustment=SourceLookAdjustment(),
                        incoming_adjustment=SourceLookAdjustment(),
                        render_verification=verification,
                    )
                )
    checked_decisions = tuple(decisions)
    return replace(
        result,
        decisions=checked_decisions,
        segment_tracks=_tracks_from_decisions(
            plan,
            checked_decisions,
            result.analysis_config,
        ),
        render_admission_config=settings,
        render_admission_encode_settings=encode_settings,
        render_admission_encode_settings_sha256=encode_settings_sha256,
    )


def analyze_boundary_look_match(
    plan: RenderPlan,
    *,
    group_16x9_safe: bool,
    config: BoundaryLookMatchConfig | None = None,
    measurement_cache: BoundaryLookMeasurementCache | None = None,
) -> BoundaryLookMatchResult:
    """Measure and attach residual look ramps without rendering any files."""

    settings = BoundaryLookMatchConfig() if config is None else config
    cache = (
        BoundaryLookMeasurementCache()
        if measurement_cache is None
        else measurement_cache
    )
    if not isinstance(group_16x9_safe, bool):
        raise TypeError("group_16x9_safe must be a boolean")
    safe = (
        group_16x9_safe
        and plan.aspect_ratio is AspectRatio.LANDSCAPE
        and abs(plan.width / plan.height - 16.0 / 9.0) <= 0.002
    )
    if not safe:
        return BoundaryLookMatchResult(
            enabled=False,
            blocked_reason="output_or_source_aspect_unsafe",
            decisions=(),
            segment_tracks=(None,) * len(plan.segments),
            decoded_frame_count=0,
            cache_hit_count=0,
            analysis_config=settings,
        )

    counts = segment_frame_counts(plan)
    raw_specs: list[tuple[tuple[float, ...], tuple[float, ...]] | None] = []
    times_by_source: dict[int, list[float]] = {}
    first_offset = -(settings.sample_frame_count // 2)
    paired_offsets = tuple(
        range(first_offset, first_offset + settings.sample_frame_count)
    )
    for index, transition in enumerate(plan.transitions):
        left = plan.segments[index]
        right = plan.segments[index + 1]
        if (
            transition.kind is not TransitionKind.HARD_CUT
            or left.source_index == right.source_index
        ):
            raw_specs.append(None)
            continue
        # Decode *both* cameras on the same canonical grid around the cut.  The
        # prior one-sided implementation compared seven outgoing frames before
        # the cut with seven incoming frames after it, so a legitimate lighting
        # cue could masquerade as camera-response pumping.  Local linear source
        # clocks are explicitly serialized on each RenderSegment and are safe to
        # extrapolate by this bounded three-frame handle.
        left_times = tuple(
            left.source_start
            + (left.duration + offset / plan.fps) * left.source_playback_rate
            for offset in paired_offsets
        )
        right_times = tuple(
            right.source_start
            + (offset / plan.fps) * right.source_playback_rate
            for offset in paired_offsets
        )
        raw_specs.append((left_times, right_times))
        times_by_source.setdefault(left.source_index, []).extend(left_times)
        times_by_source.setdefault(right.source_index, []).extend(right_times)

    decoded_before = cache.decoded_frame_count
    hits_before = cache.cache_hit_count
    measured: dict[int, dict[int, LookMetricVector | None]] = {}
    for source_index in sorted(times_by_source):
        measured[source_index] = cache.measure_many(
            plan.sources[source_index],
            times_by_source[source_index],
        )

    decisions: list[BoundaryLookDecision] = []
    endpoint_adjustments: dict[
        int, dict[str, SourceLookAdjustment]
    ] = {}
    for boundary_index, raw_spec in enumerate(raw_specs):
        left = plan.segments[boundary_index]
        right = plan.segments[boundary_index + 1]
        if raw_spec is None:
            decisions.append(
                BoundaryLookDecision(
                    boundary_index=boundary_index,
                    outgoing_source_index=left.source_index,
                    incoming_source_index=right.source_index,
                    eligible=False,
                    applied=False,
                    reason="not_cross_source_hard_cut",
                )
            )
            continue
        left_times, right_times = raw_spec
        paired_metrics = tuple(
            (
                _after_static_source_look(left_metric, left.look_adjustment),
                _after_static_source_look(right_metric, right.look_adjustment),
            )
            for left_time, right_time in zip(left_times, right_times, strict=True)
            for left_metric, right_metric in (
                (
                    measured[left.source_index].get(
                        int(round(left_time * 1_000_000.0))
                    ),
                    measured[right.source_index].get(
                        int(round(right_time * 1_000_000.0))
                    ),
                ),
            )
            if left_metric is not None and right_metric is not None
        )
        if len(paired_metrics) < settings.minimum_decoded_samples:
            decisions.append(
                BoundaryLookDecision(
                    boundary_index=boundary_index,
                    outgoing_source_index=left.source_index,
                    incoming_source_index=right.source_index,
                    eligible=True,
                    applied=False,
                    reason="insufficient_synchronized_boundary_pairs",
                )
            )
            continue
        left_metrics = tuple(item[0] for item in paired_metrics)
        right_metrics = tuple(item[1] for item in paired_metrics)
        left_window = _aggregate_window(left_metrics)
        right_window = _aggregate_window(right_metrics)
        residual_values = tuple(
            _paired_residual(left_metric, right_metric)
            for left_metric, right_metric in paired_metrics
        )
        residual_window = _aggregate_residual_window(residual_values)
        if not _residual_window_is_stable(residual_window, settings):
            decisions.append(
                BoundaryLookDecision(
                    boundary_index=boundary_index,
                    outgoing_source_index=left.source_index,
                    incoming_source_index=right.source_index,
                    eligible=True,
                    applied=False,
                    reason="volatile_synchronized_camera_residual",
                    outgoing_window=left_window,
                    incoming_window=right_window,
                    synchronized_residual_window=residual_window,
                )
            )
            continue
        residual = (
            residual_window.median.luma_delta,
            residual_window.median.contrast_log_ratio,
            residual_window.median.saturation_log_ratio,
            residual_window.median.temperature_log_red_blue_delta,
        )
        luma_difference = abs(residual[0])
        contrast_difference = abs(residual[1])
        saturation_difference = abs(residual[2])
        temperature_difference = abs(residual[3])
        if (
            luma_difference > settings.maximum_luma_difference
            or contrast_difference > settings.maximum_contrast_log_difference
            or saturation_difference > settings.maximum_saturation_log_difference
        ):
            decisions.append(
                BoundaryLookDecision(
                    boundary_index=boundary_index,
                    outgoing_source_index=left.source_index,
                    incoming_source_index=right.source_index,
                    eligible=True,
                    applied=False,
                    reason="required_residual_exceeds_safe_range",
                    outgoing_window=left_window,
                    incoming_window=right_window,
                    synchronized_residual_window=residual_window,
                )
            )
            continue
        if (
            temperature_difference
            > settings.maximum_partial_temperature_difference
        ):
            decisions.append(
                BoundaryLookDecision(
                    boundary_index=boundary_index,
                    outgoing_source_index=left.source_index,
                    incoming_source_index=right.source_index,
                    eligible=True,
                    applied=False,
                    reason="temperature_residual_exceeds_partial_repair_bound",
                    outgoing_window=left_window,
                    incoming_window=right_window,
                    synchronized_residual_window=residual_window,
                )
            )
            continue
        before = _proxy_distance_from_residual(residual)
        outgoing, incoming = _derive_endpoint_adjustments_from_residual(
            residual,
            settings,
        )
        after = _proxy_distance_from_residual(
            _residual_after_adjustments(residual, outgoing, incoming)
        )
        reduction = (before - after) / max(before, 1e-9)
        applies = (
            before >= settings.minimum_proxy_distance
            and reduction >= settings.minimum_relative_predicted_reduction
            and not (outgoing.is_identity and incoming.is_identity)
        )
        if applies:
            reason = (
                "stable_bounded_partial_residual_with_yuv_chroma_affine_"
                "temperature"
                if temperature_difference
                > settings.maximum_unrepairable_temperature_difference
                else "stable_bounded_synchronized_boundary_residual_with_"
                "yuv_chroma_affine_temperature"
                if (
                    abs(outgoing.temperature_balance) > 1e-12
                    or abs(incoming.temperature_balance) > 1e-12
                )
                else "stable_bounded_synchronized_boundary_residual"
            )
        else:
            reason = "predicted_improvement_below_safe_minimum"
        decisions.append(
            BoundaryLookDecision(
                boundary_index=boundary_index,
                outgoing_source_index=left.source_index,
                incoming_source_index=right.source_index,
                eligible=True,
                applied=applies,
                reason=reason,
                outgoing_window=left_window,
                incoming_window=right_window,
                synchronized_residual_window=residual_window,
                outgoing_adjustment=outgoing if applies else SourceLookAdjustment(),
                incoming_adjustment=incoming if applies else SourceLookAdjustment(),
                predicted_distance_before=before,
                predicted_distance_after=after,
            )
        )
        if applies:
            endpoint_adjustments.setdefault(boundary_index, {})[
                "outgoing"
            ] = outgoing
            endpoint_adjustments.setdefault(boundary_index + 1, {})[
                "incoming"
            ] = incoming

    tracks = tuple(
        _track_for_segment(
            frame_count=counts[index],
            duration=plan.segments[index].duration,
            fps=plan.fps,
            incoming=endpoint_adjustments.get(index, {}).get(
                "incoming", SourceLookAdjustment()
            ),
            outgoing=endpoint_adjustments.get(index, {}).get(
                "outgoing", SourceLookAdjustment()
            ),
            config=settings,
        )
        for index in range(len(plan.segments))
    )
    return BoundaryLookMatchResult(
        enabled=True,
        blocked_reason=None,
        decisions=tuple(decisions),
        segment_tracks=tracks,
        decoded_frame_count=cache.decoded_frame_count - decoded_before,
        cache_hit_count=cache.cache_hit_count - hits_before,
        analysis_config=settings,
    )


__all__ = [
    "ANALYSIS_BASIS",
    "BoundaryLookDecision",
    "BoundaryLookMatchConfig",
    "BoundaryLookMatchResult",
    "BoundaryLookMeasurementCache",
    "BoundaryLookRenderAdmissionConfig",
    "BoundaryLookRenderVerification",
    "BoundaryLookWindow",
    "analyze_boundary_look_match",
    "verify_boundary_look_match_render_domain",
]
