"""Cache-backed static look calibration for direct-member crosscuts.

The direct-member visual selector already owns six exact, source-clock-bound
descriptors for every selected musical span.  Their color embeddings contain
mean Lab color, gray mean, and gray contrast, so the final edit can normalize
camera-grade residuals without rendering another full-song proxy or decoding
the sources again.  Only scalar aggregates and bounded static adjustments are
retained; raw frames and descriptor embeddings are never serialized here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from statistics import median
from typing import Any, Sequence

import numpy as np

from .direct_member_visual import DirectMemberVisualSelection
from .render import RenderSegment, SourceLookAdjustment


ANALYSIS_BASIS = (
    "selected_span_cached_lab_gray_graph_regularized_member_look_v1"
)


@dataclass(frozen=True, slots=True)
class DirectMemberLookConfig:
    minimum_descriptor_samples_per_segment: int = 3
    rapid_run_maximum_seconds: float = 3.0
    graph_regularizations: tuple[float, ...] = (24.0, 80.0, 240.0, 800.0)
    maximum_brightness: float = 0.18
    minimum_contrast: float = 0.60
    maximum_contrast: float = 1.40
    minimum_source_contrast: float = 0.80
    maximum_source_contrast: float = 1.20
    minimum_saturation: float = 0.80
    maximum_saturation: float = 1.20
    maximum_temperature_balance: float = 0.18
    maximum_tint_balance: float = 0.10
    maximum_source_luma_shift: float = 0.18
    maximum_source_chroma_shift: float = 0.14
    target_rapid_high_shock_fraction: float = 0.20
    maximum_rapid_high_shock_fraction: float = 0.25
    sharpness_log_step: float = 0.45
    luma_step: float = 0.065
    chroma_step: float = 0.055
    source_sharpness_log_span: float = 0.55
    source_luma_span: float = 0.14
    source_chroma_span: float = 0.12

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_descriptor_samples_per_segment, bool)
            or not isinstance(self.minimum_descriptor_samples_per_segment, int)
            or self.minimum_descriptor_samples_per_segment < 3
        ):
            raise ValueError("member look needs at least three cached samples")
        if not self.graph_regularizations or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.graph_regularizations
        ):
            raise ValueError("member look graph regularizations are invalid")
        finite = tuple(
            float(getattr(self, name))
            for name in self.__dataclass_fields__
            if name not in {
                "minimum_descriptor_samples_per_segment",
                "graph_regularizations",
            }
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in finite):
            raise ValueError("member look bounds must be positive and finite")
        if not (
            0.0
            < self.target_rapid_high_shock_fraction
            <= self.maximum_rapid_high_shock_fraction
            < 1.0
        ):
            raise ValueError("member look rapid-shock fractions are invalid")
        if not (
            self.minimum_contrast
            <= self.minimum_source_contrast
            <= self.maximum_source_contrast
            <= self.maximum_contrast
        ):
            raise ValueError("member look source contrast bounds are invalid")
        if not 0.80 <= self.minimum_saturation <= self.maximum_saturation <= 1.20:
            raise ValueError("member look saturation bounds are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in (
                (field, getattr(self, field))
                for field in self.__dataclass_fields__
            )
        }


@dataclass(frozen=True, slots=True)
class DirectMemberLookObservation:
    segment_index: int
    source_index: int
    duration_seconds: float
    luma: float
    chroma_a: float
    chroma_b: float
    detail_proxy: float
    sample_count: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.segment_index, self.source_index)
        ):
            raise ValueError("member look observation indices are invalid")
        if (
            not math.isfinite(self.duration_seconds)
            or self.duration_seconds <= 0.0
        ):
            raise ValueError("member look observation duration is invalid")
        if any(
            not math.isfinite(value)
            for value in (
                self.luma,
                self.chroma_a,
                self.chroma_b,
                self.detail_proxy,
            )
        ):
            raise ValueError("member look observation scalars must be finite")
        if not 0.0 <= self.luma <= 1.0 or self.detail_proxy <= 0.0:
            raise ValueError("member look luma/detail evidence is invalid")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 1
        ):
            raise ValueError("member look sample count is invalid")

    def to_dict(self) -> dict[str, int | float]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class _LookRun:
    run_index: int
    source_index: int
    segment_indices: tuple[int, ...]
    duration_seconds: float
    luma: float
    chroma_a: float
    chroma_b: float
    detail_proxy: float


@dataclass(frozen=True, slots=True)
class DirectMemberLookPlan:
    observations: tuple[DirectMemberLookObservation, ...]
    segment_adjustments: tuple[SourceLookAdjustment, ...]
    selected_graph_regularization: float
    rapid_boundary_count: int
    predicted_high_shock_count: int
    predicted_high_shock_fraction: float | None
    predicted_source_spans: tuple[tuple[str, float | None], ...]
    diagnostics: tuple[dict[str, Any], ...]
    config: DirectMemberLookConfig

    def __post_init__(self) -> None:
        if len(self.observations) != len(self.segment_adjustments):
            raise ValueError("member look plan segment arrays do not match")
        if not self.observations:
            raise ValueError("member look plan cannot be empty")
        if not math.isfinite(self.selected_graph_regularization):
            raise ValueError("member look selected regularization is invalid")

    def apply(
        self,
        segments: Sequence[RenderSegment],
    ) -> tuple[RenderSegment, ...]:
        if len(segments) != len(self.segment_adjustments):
            raise ValueError("member look plan does not match render segments")
        return tuple(
            replace(segment, look_adjustment=adjustment)
            for segment, adjustment in zip(
                segments,
                self.segment_adjustments,
                strict=True,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_basis": ANALYSIS_BASIS,
            "selected_graph_regularization": self.selected_graph_regularization,
            "rapid_boundary_count": self.rapid_boundary_count,
            "predicted_high_shock_count": self.predicted_high_shock_count,
            "predicted_high_shock_fraction": self.predicted_high_shock_fraction,
            "predicted_source_spans": dict(self.predicted_source_spans),
            "segment_adjustments": [
                {
                    "segment_index": observation.segment_index,
                    "source_index": observation.source_index,
                    "brightness": adjustment.brightness,
                    "contrast": adjustment.contrast,
                    "saturation": adjustment.saturation,
                    "temperature_balance": adjustment.temperature_balance,
                    "tint_balance": adjustment.tint_balance,
                    "identity": adjustment.is_identity,
                }
                for observation, adjustment in zip(
                    self.observations,
                    self.segment_adjustments,
                    strict=True,
                )
            ],
            "diagnostics": list(self.diagnostics),
            "config": self.config.to_dict(),
            "calibration": {
                "color_axes": "ffmpeg_native_yuv_luma_exact_temperature_tint_geq_v1",
                "lab_response_basis": (
                    "eight_real_ive_member_sources_exact_ffmpeg_central_difference_v2"
                ),
                "cached_descriptor_fields": (
                    "lab_mean_gray_mean_gray_contrast"
                ),
                "full_song_proxy_rendered": False,
                "additional_source_frames_decoded": False,
            },
            "privacy": {
                "raw_frames_persisted": False,
                "raw_descriptor_embeddings_persisted": False,
                "scalar_aggregate_only": True,
            },
        }


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("member look weights must be positive")
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / total


def _runs(
    observations: Sequence[DirectMemberLookObservation],
) -> tuple[_LookRun, ...]:
    grouped: list[list[DirectMemberLookObservation]] = []
    for observation in observations:
        if not grouped or grouped[-1][-1].source_index != observation.source_index:
            grouped.append([observation])
        else:
            grouped[-1].append(observation)
    result: list[_LookRun] = []
    for run_index, items in enumerate(grouped):
        weights = [item.duration_seconds for item in items]
        result.append(
            _LookRun(
                run_index=run_index,
                source_index=items[0].source_index,
                segment_indices=tuple(item.segment_index for item in items),
                duration_seconds=sum(weights),
                luma=_weighted_mean([item.luma for item in items], weights),
                chroma_a=_weighted_mean([item.chroma_a for item in items], weights),
                chroma_b=_weighted_mean([item.chroma_b for item in items], weights),
                detail_proxy=_weighted_mean(
                    [item.detail_proxy for item in items], weights
                ),
            )
        )
    return tuple(result)


def _source_span_metrics(
    runs: Sequence[_LookRun],
    values: Sequence[tuple[float, float, float, float]],
) -> dict[str, float | None]:
    sources = sorted({run.source_index for run in runs})
    medians: list[tuple[float, float, float, float]] = []
    for source_index in sources:
        selected = [
            values[index]
            for index, run in enumerate(runs)
            if run.source_index == source_index
        ]
        medians.append(
            tuple(float(median(item[axis] for item in selected)) for axis in range(4))
        )
    if len(medians) < 2:
        return {"sharpness_log": None, "luma": None, "chroma": None}
    return {
        "sharpness_log": max(item[0] for item in medians)
        - min(item[0] for item in medians),
        "luma": max(item[1] for item in medians)
        - min(item[1] for item in medians),
        "chroma": max(
            math.hypot(left[2] - right[2], left[3] - right[3])
            for index, left in enumerate(medians)
            for right in medians[index + 1 :]
        ),
    }


def solve_direct_member_look_adjustments(
    observations: Sequence[DirectMemberLookObservation],
    *,
    config: DirectMemberLookConfig | None = None,
) -> DirectMemberLookPlan:
    settings = config or DirectMemberLookConfig()
    items = tuple(observations)
    if len(items) < 2:
        raise ValueError("member look calibration needs at least two segments")
    if tuple(item.segment_index for item in items) != tuple(range(len(items))):
        raise ValueError("member look observations must retain segment order")
    runs = _runs(items)
    rapid_edges = tuple(
        (left.run_index, right.run_index)
        for left, right in zip(runs, runs[1:])
        if left.source_index != right.source_index
        and max(left.duration_seconds, right.duration_seconds)
        <= settings.rapid_run_maximum_seconds + 1e-9
    )
    count = len(runs)
    laplacian = np.zeros((count, count), dtype=np.float64)
    for left, right in rapid_edges:
        laplacian[left, left] += 1.0
        laplacian[right, right] += 1.0
        laplacian[left, right] -= 1.0
        laplacian[right, left] -= 1.0

    raw_log_detail = np.asarray(
        [math.log(run.detail_proxy + 1e-4) for run in runs], dtype=np.float64
    )
    raw_luma = np.asarray([run.luma for run in runs], dtype=np.float64)
    raw_a = np.asarray([run.chroma_a for run in runs], dtype=np.float64)
    raw_b = np.asarray([run.chroma_b for run in runs], dtype=np.float64)
    source_indices = sorted({run.source_index for run in runs})

    source_medians: dict[int, tuple[float, float, float, float]] = {}
    for source_index in source_indices:
        indices = [
            index for index, run in enumerate(runs) if run.source_index == source_index
        ]
        source_medians[source_index] = (
            float(median(raw_log_detail[index] for index in indices)),
            float(median(raw_luma[index] for index in indices)),
            float(median(raw_a[index] for index in indices)),
            float(median(raw_b[index] for index in indices)),
        )
    global_target = tuple(
        float(median(value[axis] for value in source_medians.values()))
        for axis in range(4)
    )
    source_shifted = np.empty((count, 4), dtype=np.float64)
    minimum_source_log_contrast = math.log(settings.minimum_source_contrast)
    maximum_source_log_contrast = math.log(settings.maximum_source_contrast)
    for index, run in enumerate(runs):
        source_value = source_medians[run.source_index]
        source_shifted[index, 0] = raw_log_detail[index] + _clip(
            global_target[0] - source_value[0],
            minimum_source_log_contrast,
            maximum_source_log_contrast,
        )
        source_shifted[index, 1] = raw_luma[index] + _clip(
            global_target[1] - source_value[1],
            -settings.maximum_source_luma_shift,
            settings.maximum_source_luma_shift,
        )
        source_shifted[index, 2] = raw_a[index] + _clip(
            global_target[2] - source_value[2],
            -settings.maximum_source_chroma_shift,
            settings.maximum_source_chroma_shift,
        )
        source_shifted[index, 3] = raw_b[index] + _clip(
            global_target[3] - source_value[3],
            -settings.maximum_source_chroma_shift,
            settings.maximum_source_chroma_shift,
        )

    # Median response of the renderer's native-YUV, luma-exact chroma axes on
    # eight real IVE member sources. Rows are normalized Lab a/b; columns are
    # temperature/tint.  The luma vector captures only decode/color-matrix
    # round-trip response and lets static EQ brightness compensate it.  These
    # values were measured from the exact FFmpeg filter, not inferred from the
    # retired colorbalance preserve-lightness path that visibly posterized
    # saturated stage backgrounds.
    chroma_response = np.asarray(
        ((-0.036559999, -0.421039015), (0.392331742, 0.249684229)),
        dtype=np.float64,
    )
    luma_response = np.asarray((0.017573540, -0.104232355), dtype=np.float64)

    candidates: list[
        tuple[
            float,
            tuple[SourceLookAdjustment, ...],
            tuple[tuple[float, float, float, float], ...],
            tuple[dict[str, Any], ...],
            float | None,
            dict[str, float | None],
        ]
    ] = []
    source_compliant_candidates: list[
        tuple[
            float,
            tuple[SourceLookAdjustment, ...],
            tuple[tuple[float, float, float, float], ...],
            tuple[dict[str, Any], ...],
            float | None,
            dict[str, float | None],
        ]
    ] = []
    target_candidate: tuple[
        float,
        tuple[SourceLookAdjustment, ...],
        tuple[tuple[float, float, float, float], ...],
        tuple[dict[str, Any], ...],
        float | None,
        dict[str, float | None],
    ] | None = None
    for regularization in settings.graph_regularizations:
        matrix = np.eye(count, dtype=np.float64) + regularization * laplacian
        targets = np.column_stack(
            [np.linalg.solve(matrix, source_shifted[:, axis]) for axis in range(4)]
        )
        run_adjustments: list[SourceLookAdjustment] = []
        corrected: list[tuple[float, float, float, float]] = []
        for index, run in enumerate(runs):
            contrast = _clip(
                math.exp(float(targets[index, 0] - raw_log_detail[index])),
                settings.minimum_contrast,
                settings.maximum_contrast,
            )
            raw_chroma = np.asarray(
                (raw_a[index], raw_b[index]),
                dtype=np.float64,
            )
            target_chroma = np.asarray(
                (targets[index, 2], targets[index, 3]),
                dtype=np.float64,
            )
            raw_chroma_magnitude = float(np.linalg.norm(raw_chroma))
            saturation = _clip(
                (
                    float(np.linalg.norm(target_chroma))
                    / raw_chroma_magnitude
                    if raw_chroma_magnitude > 1e-6
                    else 1.0
                ),
                settings.minimum_saturation,
                settings.maximum_saturation,
            )
            temperature, tint = np.linalg.solve(
                chroma_response,
                target_chroma - raw_chroma * saturation,
            )
            temperature = _clip(
                temperature,
                -settings.maximum_temperature_balance,
                settings.maximum_temperature_balance,
            )
            tint = _clip(
                tint,
                -settings.maximum_tint_balance,
                settings.maximum_tint_balance,
            )
            predicted_without_brightness = (
                (raw_luma[index] - 0.5) * contrast
                + 0.5
                + luma_response[0] * temperature
                + luma_response[1] * tint
            )
            brightness = _clip(
                targets[index, 1] - predicted_without_brightness,
                -settings.maximum_brightness,
                settings.maximum_brightness,
            )
            adjustment = SourceLookAdjustment(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                temperature_balance=temperature,
                tint_balance=tint,
            )
            run_adjustments.append(adjustment)
            corrected.append(
                (
                    raw_log_detail[index] + math.log(contrast),
                    predicted_without_brightness + brightness,
                    raw_a[index] * saturation
                    + chroma_response[0, 0] * temperature
                    + chroma_response[0, 1] * tint,
                    raw_b[index] * saturation
                    + chroma_response[1, 0] * temperature
                    + chroma_response[1, 1] * tint,
                )
            )

        diagnostics: list[dict[str, Any]] = []
        high_count = 0
        for left, right in rapid_edges:
            detail_step = abs(corrected[right][0] - corrected[left][0])
            luma_step = abs(corrected[right][1] - corrected[left][1])
            chroma_step = math.hypot(
                corrected[right][2] - corrected[left][2],
                corrected[right][3] - corrected[left][3],
            )
            axes = {
                "sharpness": float(
                    detail_step / settings.sharpness_log_step
                ),
                "luma": float(luma_step / settings.luma_step),
                "chroma": float(chroma_step / settings.chroma_step),
            }
            normalized = float(max(axes.values()))
            above_threshold = bool(normalized > 1.0)
            high_count += int(above_threshold)
            diagnostics.append(
                {
                    "outgoing_run_index": left,
                    "incoming_run_index": right,
                    "outgoing_source_index": runs[left].source_index,
                    "incoming_source_index": runs[right].source_index,
                    "normalized_axes": axes,
                    "normalized_shock": normalized,
                    "above_threshold": above_threshold,
                }
            )
        high_fraction = high_count / len(rapid_edges) if rapid_edges else None
        spans = _source_span_metrics(runs, corrected)
        candidate = (
            regularization,
            tuple(run_adjustments),
            tuple(corrected),
            tuple(diagnostics),
            high_fraction,
            spans,
        )
        candidates.append(candidate)
        rapid_ok = bool(
            high_fraction is None
            or len(rapid_edges) < 4
            or high_fraction <= settings.target_rapid_high_shock_fraction + 1e-12
        )
        source_ok = bool(
            (spans["sharpness_log"] or 0.0)
            <= settings.source_sharpness_log_span + 1e-12
            and (spans["luma"] or 0.0) <= settings.source_luma_span + 1e-12
            and (spans["chroma"] or 0.0) <= settings.source_chroma_span + 1e-12
        )
        if source_ok:
            source_compliant_candidates.append(candidate)
        if rapid_ok and source_ok:
            target_candidate = candidate
            break

    candidate_pool = source_compliant_candidates or candidates
    selected = target_candidate or min(
        candidate_pool,
        key=lambda item: (
            0.0 if item[4] is None else item[4],
            (item[5]["sharpness_log"] or 0.0)
            + (item[5]["luma"] or 0.0)
            + (item[5]["chroma"] or 0.0),
            item[0],
        ),
    )
    regularization, run_adjustments, _corrected, diagnostics, fraction, spans = selected
    rapid_policy_failed = bool(
        len(rapid_edges) >= 4
        and fraction is not None
        and fraction > settings.maximum_rapid_high_shock_fraction + 1e-12
    )
    source_policy_failed = bool(
        (spans["sharpness_log"] or 0.0)
        > settings.source_sharpness_log_span + 1e-12
        or (spans["luma"] or 0.0) > settings.source_luma_span + 1e-12
        or (spans["chroma"] or 0.0) > settings.source_chroma_span + 1e-12
    )
    if rapid_policy_failed or source_policy_failed:
        failed_policies = ",".join(
            name
            for name, failed in (
                ("rapid-shock", rapid_policy_failed),
                ("source-span", source_policy_failed),
            )
            if failed
        )
        raise ValueError(
            "cached member look calibration cannot satisfy "
            f"{failed_policies} policy"
        )
    by_segment: list[SourceLookAdjustment | None] = [None] * len(items)
    for run, adjustment in zip(runs, run_adjustments, strict=True):
        for segment_index in run.segment_indices:
            by_segment[segment_index] = adjustment
    if any(value is None for value in by_segment):
        raise ValueError("member look calibration missed a render segment")
    return DirectMemberLookPlan(
        observations=items,
        segment_adjustments=tuple(value for value in by_segment if value is not None),
        selected_graph_regularization=regularization,
        rapid_boundary_count=len(rapid_edges),
        predicted_high_shock_count=sum(
            bool(item["above_threshold"]) for item in diagnostics
        ),
        predicted_high_shock_fraction=fraction,
        predicted_source_spans=tuple(sorted(spans.items())),
        diagnostics=diagnostics,
        config=settings,
    )


def build_direct_member_look_plan(
    *,
    visual_selection: DirectMemberVisualSelection,
    render_segments: Sequence[RenderSegment],
    config: DirectMemberLookConfig | None = None,
) -> DirectMemberLookPlan:
    settings = config or DirectMemberLookConfig()
    segments = tuple(render_segments)
    if not (
        len(visual_selection.source_indices)
        == len(visual_selection.span_validity)
        == len(segments)
    ):
        raise ValueError("member look selection and render segments must match")
    observations: list[DirectMemberLookObservation] = []
    for segment_index, (eligible_source_index, span, segment) in enumerate(
        zip(
            visual_selection.source_indices,
            visual_selection.span_validity,
            segments,
            strict=True,
        )
    ):
        source = next(
            (
                item
                for item in span.sources
                if item.eligible_source_index == eligible_source_index
            ),
            None,
        )
        if source is None:
            raise ValueError("selected member source lacks span evidence")
        embeddings = tuple(
            sample.descriptor.color_embedding
            for sample in source.samples
            if sample.descriptor is not None
            and len(sample.descriptor.color_embedding) >= 8
        )
        if len(embeddings) < settings.minimum_descriptor_samples_per_segment:
            raise ValueError("selected member span lacks cached look evidence")
        observations.append(
            DirectMemberLookObservation(
                segment_index=segment_index,
                source_index=int(eligible_source_index),
                duration_seconds=segment.duration,
                luma=float(median(item[6] for item in embeddings)),
                chroma_a=float(
                    median((item[1] * 255.0 - 128.0) / 127.0 for item in embeddings)
                ),
                chroma_b=float(
                    median((item[2] * 255.0 - 128.0) / 127.0 for item in embeddings)
                ),
                detail_proxy=max(
                    1e-4,
                    float(median(item[7] for item in embeddings)),
                ),
                sample_count=len(embeddings),
            )
        )
    return solve_direct_member_look_adjustments(observations, config=settings)


__all__ = [
    "ANALYSIS_BASIS",
    "DirectMemberLookConfig",
    "DirectMemberLookObservation",
    "DirectMemberLookPlan",
    "build_direct_member_look_plan",
    "solve_direct_member_look_adjustments",
]
