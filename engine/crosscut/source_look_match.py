"""Conservative static look matching for synchronized stage sources.

This module measures the same canonical performance instants in every source.
It only applies a correction when a source has enough decoded samples and its
difference from the chosen reference (or the cross-source consensus) remains
stable across those instants.  Volatile stage lighting, incomplete decoding,
or implausibly large requested corrections therefore produce an identity
adjustment instead of speculative grading.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .render import SourceLookAdjustment
from .timewarp import TimeWarpMap


SOURCE_LOOK_MATCH_VERSION = 2
SOURCE_LOOK_SAMPLE_FRACTIONS = (
    0.08,
    0.185,
    0.29,
    0.395,
    0.50,
    0.605,
    0.71,
    0.815,
    0.92,
)
_WORKING_WIDTH = 320


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _mad(values: Sequence[float]) -> float:
    center = _median(values)
    return _median([abs(value - center) for value in values])


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _deadband(value: float, identity: float, tolerance: float) -> float:
    return identity if abs(value - identity) < tolerance else value


def _directional_residual_is_reliable(
    values: Sequence[float],
    *,
    residual_median: float,
    residual_mad: float,
    deadband: float,
) -> bool:
    """Allow one bounded step for a large but consistently directed error."""

    if not values or abs(residual_median) < 2.0 * residual_mad + deadband:
        return False
    direction = 1.0 if residual_median > 0.0 else -1.0
    agreeing = sum(value * direction > deadband for value in values)
    return agreeing / len(values) >= 0.80 - 1e-12


@dataclass(frozen=True, slots=True)
class LookMetricVector:
    """Robust per-frame or aggregate color statistics in normalized units."""

    luma: float
    contrast: float
    saturation: float
    temperature_log_red_blue: float

    def __post_init__(self) -> None:
        values = (
            self.luma,
            self.contrast,
            self.saturation,
            self.temperature_log_red_blue,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("look metrics must be finite")
        if not 0.0 <= self.luma <= 1.0:
            raise ValueError("look luma must be normalized")
        if not 0.0 <= self.contrast <= 1.0:
            raise ValueError("look contrast must be normalized")
        if not 0.0 <= self.saturation <= 1.0:
            raise ValueError("look saturation must be normalized")
        if abs(self.temperature_log_red_blue) > 2.0:
            raise ValueError("look temperature statistic is implausible")

    def to_dict(self) -> dict[str, float]:
        return {
            "luma_0_1": self.luma,
            "robust_contrast_0_1": self.contrast,
            "saturation_0_1": self.saturation,
            "temperature_log_red_blue": self.temperature_log_red_blue,
        }


@dataclass(frozen=True, slots=True)
class LookResidualVector:
    """Target-minus-source evidence (log ratios for contrast/saturation)."""

    luma_delta: float
    contrast_log_ratio: float
    saturation_log_ratio: float
    temperature_log_red_blue_delta: float

    def __post_init__(self) -> None:
        values = (
            self.luma_delta,
            self.contrast_log_ratio,
            self.saturation_log_ratio,
            self.temperature_log_red_blue_delta,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("look residuals must be finite")
        if any(abs(value) > 5.0 for value in values):
            raise ValueError("look residual is implausible")

    def to_dict(self) -> dict[str, float]:
        return {
            "luma_delta": self.luma_delta,
            "contrast_log_ratio": self.contrast_log_ratio,
            "saturation_log_ratio": self.saturation_log_ratio,
            "temperature_log_red_blue_delta": (
                self.temperature_log_red_blue_delta
            ),
        }


@dataclass(frozen=True, slots=True)
class SourceLookMatchConfig:
    sample_fractions: tuple[float, ...] = SOURCE_LOOK_SAMPLE_FRACTIONS
    minimum_decoded_samples: int = 6
    minimum_sources_for_consensus: int = 3
    maximum_luma_residual_mad: float = 0.060
    maximum_contrast_log_residual_mad: float = 0.120
    maximum_saturation_log_residual_mad: float = 0.140
    maximum_temperature_residual_mad: float = 0.120
    maximum_brightness: float = 0.045
    maximum_contrast_delta: float = 0.050
    maximum_saturation_delta: float = 0.050
    maximum_temperature_balance: float = 0.025

    def __post_init__(self) -> None:
        if len(self.sample_fractions) < 3:
            raise ValueError("source look matching needs at least three sample times")
        if any(
            not math.isfinite(value) or not 0.0 < value < 1.0
            for value in self.sample_fractions
        ):
            raise ValueError("source look sample fractions must be inside (0, 1)")
        if any(
            right <= left
            for left, right in zip(self.sample_fractions, self.sample_fractions[1:])
        ):
            raise ValueError("source look sample fractions must be strictly increasing")
        if (
            isinstance(self.minimum_decoded_samples, bool)
            or not isinstance(self.minimum_decoded_samples, int)
            or not 3 <= self.minimum_decoded_samples <= len(self.sample_fractions)
        ):
            raise ValueError("minimum decoded samples is outside the sample schedule")
        if (
            isinstance(self.minimum_sources_for_consensus, bool)
            or not isinstance(self.minimum_sources_for_consensus, int)
            or self.minimum_sources_for_consensus < 3
        ):
            raise ValueError("consensus look matching needs at least three sources")
        positive_values = (
            self.maximum_luma_residual_mad,
            self.maximum_contrast_log_residual_mad,
            self.maximum_saturation_log_residual_mad,
            self.maximum_temperature_residual_mad,
            self.maximum_brightness,
            self.maximum_contrast_delta,
            self.maximum_saturation_delta,
            self.maximum_temperature_balance,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive_values):
            raise ValueError("source look matching bounds must be positive and finite")
        if self.maximum_brightness > 0.05:
            raise ValueError("source look brightness cannot exceed renderer bounds")
        if self.maximum_contrast_delta > 0.06:
            raise ValueError("source look contrast cannot exceed renderer bounds")
        if self.maximum_saturation_delta > 0.06:
            raise ValueError("source look saturation cannot exceed renderer bounds")
        if self.maximum_temperature_balance > 0.03:
            raise ValueError("source look temperature cannot exceed renderer bounds")


@dataclass(frozen=True, slots=True)
class SourceLookEvidence:
    source_index: int
    decoded_sample_count: int
    comparison_sample_count: int
    observed_median: LookMetricVector | None
    residual_median: LookResidualVector | None
    residual_mad: LookResidualVector | None
    stable: bool
    reason: str
    adjustment: SourceLookAdjustment
    component_status: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_index": self.source_index,
            "decoded_sample_count": self.decoded_sample_count,
            "comparison_sample_count": self.comparison_sample_count,
            "observed_median": (
                None if self.observed_median is None else self.observed_median.to_dict()
            ),
            "residual_median": (
                None if self.residual_median is None else self.residual_median.to_dict()
            ),
            "residual_mad": (
                None if self.residual_mad is None else self.residual_mad.to_dict()
            ),
            "stable": self.stable,
            "reason": self.reason,
            "adjustment": {
                "brightness": self.adjustment.brightness,
                "contrast": self.adjustment.contrast,
                "saturation": self.adjustment.saturation,
                "temperature_balance": self.adjustment.temperature_balance,
                "identity": self.adjustment.is_identity,
            },
            "component_status": dict(self.component_status),
        }


@dataclass(frozen=True, slots=True)
class SourceLookMatchResult:
    target_mode: str
    target_source_index: int | None
    target_median: LookMetricVector | None
    canonical_sample_times_seconds: tuple[float, ...]
    evidence: tuple[SourceLookEvidence, ...]

    def __post_init__(self) -> None:
        if self.target_mode not in {
            "explicit_reference",
            "cross_source_consensus",
            "unavailable_identity",
        }:
            raise ValueError("unknown source look target mode")
        if len(self.evidence) < 1:
            raise ValueError("source look result needs source evidence")
        if tuple(item.source_index for item in self.evidence) != tuple(
            range(len(self.evidence))
        ):
            raise ValueError("source look evidence must retain source-array order")

    @property
    def adjustments(self) -> tuple[SourceLookAdjustment, ...]:
        return tuple(item.adjustment for item in self.evidence)

    def adjustment_for_source(self, source_index: int) -> SourceLookAdjustment:
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or not 0 <= source_index < len(self.evidence)
        ):
            raise IndexError("source look index is outside the evidence array")
        return self.evidence[source_index].adjustment

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SOURCE_LOOK_MATCH_VERSION,
            "target_mode": self.target_mode,
            "target_source_index": self.target_source_index,
            "target_median": (
                None if self.target_median is None else self.target_median.to_dict()
            ),
            "canonical_sample_times_seconds": list(
                self.canonical_sample_times_seconds
            ),
            "evidence": [item.to_dict() for item in self.evidence],
            "raw_frames_persisted": False,
            "learned_identity_model_used": False,
            "component_wise_fail_closed": True,
        }


def measure_frame_look(frame: np.ndarray) -> LookMetricVector:
    """Measure robust global look statistics without retaining the frame."""

    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("source look frame is empty")
    if frame.ndim == 2:
        working = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        working = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        working = frame
    else:
        raise ValueError("source look frame has an unsupported shape")
    height, width = working.shape[:2]
    if width > _WORKING_WIDTH:
        scale = _WORKING_WIDTH / width
        working = cv2.resize(
            working,
            (_WORKING_WIDTH, max(2, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    rgb = working[..., ::-1].astype(np.float32) / 255.0
    luma = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )
    usable = (luma >= 0.04) & (luma <= 0.96)
    minimum_pixels = max(128, int(round(luma.size * 0.20)))
    if int(np.count_nonzero(usable)) < minimum_pixels:
        raise ValueError("source look frame lacks reliable exposed pixels")

    usable_luma = luma[usable].astype(np.float64)
    lower, upper = np.percentile(usable_luma, (10.0, 90.0))
    robust_luma = float(np.median(usable_luma))
    robust_contrast = float(upper - lower)

    hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
    saturation_plane = hsv[..., 1].astype(np.float32) / 255.0
    usable_saturation = saturation_plane[usable].astype(np.float64)
    robust_saturation = float(np.median(usable_saturation))

    # Estimate temperature only from the lowest-chroma third of usable pixels.
    # This is much less sensitive to LED walls and costumes than global RGB.
    neutral_threshold = float(np.percentile(usable_saturation, 35.0))
    neutral = usable & (saturation_plane <= neutral_threshold + 1e-9)
    if int(np.count_nonzero(neutral)) < 128:
        raise ValueError("source look frame lacks neutral temperature evidence")
    red = rgb[..., 0][neutral].astype(np.float64)
    blue = rgb[..., 2][neutral].astype(np.float64)
    temperature = float(np.median(np.log((red + 0.04) / (blue + 0.04))))
    return LookMetricVector(
        luma=robust_luma,
        contrast=_clip(robust_contrast, 0.0, 1.0),
        saturation=robust_saturation,
        temperature_log_red_blue=_clip(temperature, -2.0, 2.0),
    )


def _aggregate(samples: Sequence[LookMetricVector]) -> LookMetricVector:
    return LookMetricVector(
        luma=_median([sample.luma for sample in samples]),
        contrast=_median([sample.contrast for sample in samples]),
        saturation=_median([sample.saturation for sample in samples]),
        temperature_log_red_blue=_median(
            [sample.temperature_log_red_blue for sample in samples]
        ),
    )


def _residual(
    target: LookMetricVector,
    source: LookMetricVector,
) -> LookResidualVector:
    return LookResidualVector(
        luma_delta=target.luma - source.luma,
        contrast_log_ratio=math.log(
            max(0.03, target.contrast) / max(0.03, source.contrast)
        ),
        saturation_log_ratio=math.log(
            max(0.04, target.saturation) / max(0.04, source.saturation)
        ),
        temperature_log_red_blue_delta=(
            target.temperature_log_red_blue - source.temperature_log_red_blue
        ),
    )


def _residual_values(item: LookResidualVector) -> tuple[float, float, float, float]:
    return (
        item.luma_delta,
        item.contrast_log_ratio,
        item.saturation_log_ratio,
        item.temperature_log_red_blue_delta,
    )


def _residual_vector(
    values: Sequence[tuple[float, float, float, float]],
) -> LookResidualVector:
    return LookResidualVector(
        luma_delta=_median([item[0] for item in values]),
        contrast_log_ratio=_median([item[1] for item in values]),
        saturation_log_ratio=_median([item[2] for item in values]),
        temperature_log_red_blue_delta=_median([item[3] for item in values]),
    )


def _residual_mad_vector(
    values: Sequence[tuple[float, float, float, float]],
) -> LookResidualVector:
    return LookResidualVector(
        luma_delta=_mad([item[0] for item in values]),
        contrast_log_ratio=_mad([item[1] for item in values]),
        saturation_log_ratio=_mad([item[2] for item in values]),
        temperature_log_red_blue_delta=_mad([item[3] for item in values]),
    )


def _identity_evidence(
    source_index: int,
    samples: Sequence[LookMetricVector | None],
    reason: str,
) -> SourceLookEvidence:
    decoded = tuple(sample for sample in samples if sample is not None)
    return SourceLookEvidence(
        source_index=source_index,
        decoded_sample_count=len(decoded),
        comparison_sample_count=0,
        observed_median=_aggregate(decoded) if decoded else None,
        residual_median=None,
        residual_mad=None,
        stable=False,
        reason=reason,
        adjustment=SourceLookAdjustment(),
        component_status=tuple(
            (name, reason)
            for name in ("luma", "contrast", "saturation", "temperature")
        ),
    )


def match_source_look_samples(
    samples_by_source: Sequence[Sequence[LookMetricVector | None]],
    *,
    canonical_sample_times_seconds: Sequence[float],
    reference_source: int | None = None,
    config: SourceLookMatchConfig | None = None,
) -> SourceLookMatchResult:
    """Derive per-source adjustments from already synchronized measurements."""

    settings = SourceLookMatchConfig() if config is None else config
    if not samples_by_source:
        raise ValueError("source look matching needs at least one source")
    sample_count = len(settings.sample_fractions)
    if len(canonical_sample_times_seconds) != sample_count:
        raise ValueError("canonical sample times do not match the configured schedule")
    if any(len(samples) != sample_count for samples in samples_by_source):
        raise ValueError("every source must retain the full synchronized sample grid")
    if reference_source is not None and (
        isinstance(reference_source, bool)
        or not isinstance(reference_source, int)
        or not 0 <= reference_source < len(samples_by_source)
    ):
        raise ValueError("reference source is outside the source array")

    target_samples: list[LookMetricVector | None] = [None] * sample_count
    target_mode = "cross_source_consensus"
    target_source_index: int | None = None
    if reference_source is not None and sum(
        sample is not None for sample in samples_by_source[reference_source]
    ) >= settings.minimum_decoded_samples:
        target_mode = "explicit_reference"
        target_source_index = reference_source
        target_samples = list(samples_by_source[reference_source])
    else:
        for sample_index in range(sample_count):
            synchronized = tuple(
                source_samples[sample_index]
                for source_samples in samples_by_source
                if source_samples[sample_index] is not None
            )
            if len(synchronized) >= settings.minimum_sources_for_consensus:
                target_samples[sample_index] = _aggregate(synchronized)  # type: ignore[arg-type]

    valid_targets = tuple(sample for sample in target_samples if sample is not None)
    if len(valid_targets) < settings.minimum_decoded_samples:
        reason = "insufficient_synchronized_target_evidence"
        return SourceLookMatchResult(
            target_mode="unavailable_identity",
            target_source_index=None,
            target_median=None,
            canonical_sample_times_seconds=tuple(canonical_sample_times_seconds),
            evidence=tuple(
                _identity_evidence(index, samples, reason)
                for index, samples in enumerate(samples_by_source)
            ),
        )

    evidence: list[SourceLookEvidence] = []
    for source_index, source_samples in enumerate(samples_by_source):
        decoded = tuple(sample for sample in source_samples if sample is not None)
        if len(decoded) < settings.minimum_decoded_samples:
            evidence.append(
                _identity_evidence(
                    source_index,
                    source_samples,
                    "insufficient_decoded_source_evidence",
                )
            )
            continue
        residuals = tuple(
            _residual_values(_residual(target, source))
            for source, target in zip(source_samples, target_samples, strict=True)
            if source is not None and target is not None
        )
        if len(residuals) < settings.minimum_decoded_samples:
            evidence.append(
                _identity_evidence(
                    source_index,
                    source_samples,
                    "insufficient_synchronized_comparisons",
                )
            )
            continue
        residual_median = tuple(
            _median([item[index] for item in residuals]) for index in range(4)
        )
        residual_mad = tuple(
            _mad([item[index] for item in residuals]) for index in range(4)
        )
        component_names = ("luma", "contrast", "saturation", "temperature")
        stable_limits = (
            settings.maximum_luma_residual_mad,
            settings.maximum_contrast_log_residual_mad,
            settings.maximum_saturation_log_residual_mad,
            settings.maximum_temperature_residual_mad,
        )
        plausible_limits = (
            0.10,
            math.log(1.18),
            math.log(1.25),
            0.35,
        )
        component_deadbands = (0.005, 0.010, 0.015, 0.004)
        component_stable = tuple(
            mad <= stable_limit
            for mad, stable_limit in zip(
                residual_mad,
                stable_limits,
                strict=True,
            )
        )
        component_large_directional = tuple(
            stable
            and abs(median) > plausible_limit
            and _directional_residual_is_reliable(
                tuple(item[index] for item in residuals),
                residual_median=median,
                residual_mad=mad,
                deadband=deadband,
            )
            for index, (
                median,
                mad,
                stable,
                plausible_limit,
                deadband,
            ) in enumerate(
                zip(
                    residual_median,
                    residual_mad,
                    component_stable,
                    plausible_limits,
                    component_deadbands,
                    strict=True,
                )
            )
        )
        component_safe = tuple(
            stable
            and (
                abs(median) <= plausible_limit
                or large_directional
            )
            for median, stable, plausible_limit, large_directional in zip(
                residual_median,
                component_stable,
                plausible_limits,
                component_large_directional,
                strict=True,
            )
        )
        stable = all(component_safe)
        if target_mode == "explicit_reference" and source_index == reference_source:
            reason = "explicit_reference_identity"
            adjustment = SourceLookAdjustment()
            component_status = tuple(
                (name, "explicit_reference_identity") for name in component_names
            )
        else:
            requested = (
                _deadband(
                _clip(
                    0.65 * residual_median[0],
                    -settings.maximum_brightness,
                    settings.maximum_brightness,
                ),
                0.0,
                0.005,
                ),
                _deadband(
                _clip(
                    math.exp(0.60 * residual_median[1]),
                    1.0 - settings.maximum_contrast_delta,
                    1.0 + settings.maximum_contrast_delta,
                ),
                1.0,
                0.010,
                ),
                _deadband(
                _clip(
                    math.exp(0.55 * residual_median[2]),
                    1.0 - settings.maximum_saturation_delta,
                    1.0 + settings.maximum_saturation_delta,
                ),
                1.0,
                0.015,
                ),
                _deadband(
                _clip(
                    0.16 * residual_median[3],
                    -settings.maximum_temperature_balance,
                    settings.maximum_temperature_balance,
                ),
                0.0,
                0.004,
                ),
            )
            identities = (0.0, 1.0, 1.0, 0.0)
            applied_values = tuple(
                value if safe else identity
                for value, safe, identity in zip(
                    requested,
                    component_safe,
                    identities,
                    strict=True,
                )
            )
            adjustment = SourceLookAdjustment(
                brightness=applied_values[0],
                contrast=applied_values[1],
                saturation=applied_values[2],
                temperature_balance=applied_values[3],
            )
            component_status = tuple(
                (
                    name,
                    (
                        "unstable_synchronized_residual"
                        if mad > stable_limit
                        else (
                            "bounded_directional_step_applied_"
                            "large_stable_residual"
                        )
                        if large_directional
                        and abs(value - identity) > 1e-12
                        else "required_correction_exceeds_safe_evidence_range"
                        if abs(median) > plausible_limit
                        else "identity_within_deadband"
                        if abs(value - identity) <= 1e-12
                        else "bounded_adjustment_applied"
                    ),
                )
                for (
                    name,
                    median,
                    mad,
                    stable_limit,
                    plausible_limit,
                    large_directional,
                    value,
                    identity,
                ) in zip(
                    component_names,
                    residual_median,
                    residual_mad,
                    stable_limits,
                    plausible_limits,
                    component_large_directional,
                    applied_values,
                    identities,
                    strict=True,
                )
            )
            safe_count = sum(component_safe)
            if stable:
                reason = (
                    "stable_identity_within_deadbands"
                    if adjustment.is_identity
                    else "stable_bounded_static_match"
                )
            elif safe_count == 0:
                reason = (
                    "required_correction_exceeds_safe_evidence_range"
                    if all(
                        mad <= stable_limit
                        for mad, stable_limit in zip(
                            residual_mad, stable_limits, strict=True
                        )
                    )
                    else "unstable_synchronized_look_residuals"
                )
            else:
                reason = (
                    "unstable_synchronized_look_residuals"
                    if adjustment.is_identity
                    and any(
                        mad > stable_limit
                        for mad, stable_limit in zip(
                            residual_mad, stable_limits, strict=True
                        )
                    )
                    else "required_correction_exceeds_safe_evidence_range"
                    if adjustment.is_identity
                    else "partial_stable_bounded_static_match"
                )
        evidence.append(
            SourceLookEvidence(
                source_index=source_index,
                decoded_sample_count=len(decoded),
                comparison_sample_count=len(residuals),
                observed_median=_aggregate(decoded),
                residual_median=_residual_vector(residuals),
                residual_mad=_residual_mad_vector(residuals),
                stable=stable,
                reason=reason,
                adjustment=adjustment,
                component_status=component_status,
            )
        )
    return SourceLookMatchResult(
        target_mode=target_mode,
        target_source_index=target_source_index,
        target_median=_aggregate(valid_targets),
        canonical_sample_times_seconds=tuple(canonical_sample_times_seconds),
        evidence=tuple(evidence),
    )


def analyze_source_look_match(
    sources: Sequence[Path],
    *,
    source_offsets_seconds: Sequence[float],
    common_master_start: float,
    common_master_end: float,
    reference_source: int | None = None,
    config: SourceLookMatchConfig | None = None,
    source_timewarps: Sequence[TimeWarpMap | None] | None = None,
) -> SourceLookMatchResult:
    """Decode synchronized in-performance frames and derive safe corrections.

    ``source_offsets_seconds`` follows the engine alignment invariant
    ``source_time = canonical_master_time + offset``.  A checked non-scalar
    ``source_timewarps`` entry supersedes that affine lookup at each canonical
    sample; ``None`` and explicit scalar fallbacks retain the legacy equation.
    No frames, histograms, descriptors, or learned identity embeddings are
    persisted.
    """

    settings = SourceLookMatchConfig() if config is None else config
    if not sources:
        raise ValueError("source look matching needs at least one source")
    if len(sources) != len(source_offsets_seconds):
        raise ValueError("source look offsets must match the source array")
    if source_timewarps is None:
        clock_maps: tuple[TimeWarpMap | None, ...] = (None,) * len(sources)
    else:
        if len(source_timewarps) != len(sources):
            raise ValueError("source look timewarps must match the source array")
        clock_maps = tuple(source_timewarps)
        for index, timewarp in enumerate(clock_maps):
            if timewarp is None:
                continue
            if not isinstance(timewarp, TimeWarpMap):
                raise TypeError("source look timewarps must contain TimeWarpMap or None")
            if not math.isclose(
                timewarp.global_offset_seconds,
                float(source_offsets_seconds[index]),
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"source look timewarp {index} does not match its scalar offset"
                )
    values = (common_master_start, common_master_end, *source_offsets_seconds)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("source look sampling times must be finite")
    if common_master_start < 0.0 or common_master_end <= common_master_start:
        raise ValueError("source look common master range is invalid")
    canonical_times = tuple(
        common_master_start
        + fraction * (common_master_end - common_master_start)
        for fraction in settings.sample_fractions
    )
    sampled: list[tuple[LookMetricVector | None, ...]] = []
    for path, offset, timewarp in zip(
        sources,
        source_offsets_seconds,
        clock_maps,
        strict=True,
    ):
        source_samples: list[LookMetricVector | None] = []
        try:
            capture = cv2.VideoCapture(str(Path(path)))
            opened = capture.isOpened()
        except (cv2.error, OSError, RuntimeError, TypeError, ValueError):
            sampled.append(tuple(None for _ in canonical_times))
            continue
        if not opened:
            sampled.append(tuple(None for _ in canonical_times))
            try:
                capture.release()
            except (cv2.error, RuntimeError):
                pass
            continue
        try:
            for canonical_time in canonical_times:
                source_time = (
                    timewarp.master_to_source(canonical_time)
                    if timewarp is not None and not timewarp.scalar_fallback
                    else canonical_time + offset
                )
                if source_time < 0.0:
                    source_samples.append(None)
                    continue
                try:
                    capture.set(cv2.CAP_PROP_POS_MSEC, source_time * 1000.0)
                    success, frame = capture.read()
                except (cv2.error, OSError, RuntimeError, TypeError, ValueError):
                    source_samples.append(None)
                    continue
                if (
                    not success
                    or frame is None
                    or not isinstance(frame, np.ndarray)
                    or frame.size == 0
                ):
                    source_samples.append(None)
                    continue
                try:
                    source_samples.append(measure_frame_look(frame))
                except (cv2.error, TypeError, ValueError, FloatingPointError):
                    source_samples.append(None)
        finally:
            try:
                capture.release()
            except (cv2.error, RuntimeError):
                pass
        sampled.append(tuple(source_samples))
    return match_source_look_samples(
        sampled,
        canonical_sample_times_seconds=canonical_times,
        reference_source=reference_source,
        config=settings,
    )
