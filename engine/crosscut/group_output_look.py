"""Bounded decoded-output look refinement for automatic GROUP edits.

The static source matcher sees synchronized full-frame inputs, while the
professional proxy gate measures the actually selected, cropped, transformed,
and encoded output.  This module closes that domain gap without changing the
cut path: it derives one small source-local residual from scalar-only decoded
proxy evidence and composes it onto the existing render plan.  A caller must
render the proposal and accept it only when :func:`decoded_look_objective`
strictly improves without regressing the non-look gates.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np

from .render import RenderPlan, SourceLookAdjustment, segment_frame_counts
from .rendered_proxy_review import (
    PERCEPTUAL_SOURCE_ADMISSION_ANALYSIS_BASIS,
    SEGMENT_INTERIOR_GEOMETRY_ANALYSIS_BASIS,
    SEGMENT_INTERIOR_GEOMETRY_CONTRACT,
    SEGMENT_INTERIOR_GEOMETRY_SCHEMA_VERSION,
    _canonical_sha256,
    _render_timeline_binding_payload,
    verified_file_content_sha256,
)


ANALYSIS_BASIS = "decoded_output_source_median_bounded_group_look_v1"
LINEAGE_CONTRACT = "group_decoded_output_look_refinement_lineage_v1"
_STABLE_PROXY_IDENTITY_FIELDS = (
    "width",
    "height",
    "fps",
    "decoded_frame_count",
    "duration_seconds",
    "render_codec",
    "content_sha256",
)


@dataclass(frozen=True, slots=True)
class GroupDecodedLookConfig:
    """Small per-round steps inside the renderer's calibrated envelope."""

    minimum_decoded_output_frames_per_source: int = 12
    maximum_refinement_iterations: int = 2
    maximum_brightness_step: float = 0.045
    maximum_contrast_step: float = 0.050
    maximum_saturation_step: float = 0.150
    maximum_temperature_step: float = 0.090
    maximum_tint_step: float = 0.050
    maximum_detail_step: float = 0.25
    maximum_cumulative_brightness: float = 0.18
    minimum_cumulative_contrast: float = 0.60
    maximum_cumulative_contrast: float = 1.40
    minimum_cumulative_saturation: float = 0.80
    maximum_cumulative_saturation: float = 1.20
    maximum_cumulative_temperature: float = 0.18
    maximum_cumulative_tint: float = 0.10
    maximum_cumulative_detail: float = 0.90

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_decoded_output_frames_per_source, bool)
            or not isinstance(self.minimum_decoded_output_frames_per_source, int)
            or self.minimum_decoded_output_frames_per_source < 12
        ):
            raise ValueError("GROUP decoded look needs at least 12 output frames")
        if (
            isinstance(self.maximum_refinement_iterations, bool)
            or not isinstance(self.maximum_refinement_iterations, int)
            or not 1 <= self.maximum_refinement_iterations <= 2
        ):
            raise ValueError("GROUP decoded look is limited to two iterations")
        positive = tuple(
            float(getattr(self, name))
            for name in (
                "maximum_brightness_step",
                "maximum_contrast_step",
                "maximum_saturation_step",
                "maximum_temperature_step",
                "maximum_tint_step",
                "maximum_detail_step",
                "maximum_cumulative_brightness",
                "minimum_cumulative_contrast",
                "maximum_cumulative_contrast",
                "minimum_cumulative_saturation",
                "maximum_cumulative_saturation",
                "maximum_cumulative_temperature",
                "maximum_cumulative_tint",
                "maximum_cumulative_detail",
            )
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("GROUP decoded look bounds must be finite and positive")
        if self.maximum_brightness_step > 0.045:
            raise ValueError("GROUP decoded brightness step exceeds static matcher")
        if self.maximum_contrast_step > 0.050:
            raise ValueError("GROUP decoded contrast step exceeds static matcher")
        if self.maximum_saturation_step > 0.150:
            raise ValueError("GROUP decoded saturation step exceeds bounded residual")
        if self.maximum_temperature_step > 0.090:
            raise ValueError("GROUP decoded temperature step exceeds bounded residual")
        if self.maximum_tint_step > 0.050:
            raise ValueError("GROUP decoded tint step exceeds bounded residual")
        if self.maximum_detail_step > 0.25:
            raise ValueError("GROUP decoded detail step exceeds bounded residual")
        if not (
            0.60 <= self.minimum_cumulative_contrast
            <= self.maximum_cumulative_contrast <= 1.40
        ):
            raise ValueError("GROUP decoded cumulative contrast bounds are invalid")
        if not (
            0.80 <= self.minimum_cumulative_saturation
            <= self.maximum_cumulative_saturation <= 1.20
        ):
            raise ValueError("GROUP decoded cumulative saturation bounds are invalid")
        if self.maximum_cumulative_brightness > 0.18:
            raise ValueError("GROUP decoded cumulative brightness exceeds renderer")
        if self.maximum_cumulative_temperature > 0.18:
            raise ValueError("GROUP decoded cumulative temperature exceeds renderer")
        if self.maximum_cumulative_tint > 0.10:
            raise ValueError("GROUP decoded cumulative tint exceeds renderer")
        if self.maximum_cumulative_detail > 0.90:
            raise ValueError("GROUP decoded cumulative detail exceeds safe policy")


@dataclass(frozen=True, slots=True)
class DecodedLookObjective:
    """Lexicographic objective used for strict accept-or-rollback."""

    admission_passed: bool
    normalized_excess: float
    selection_penalty_points: float

    def __post_init__(self) -> None:
        if not isinstance(self.admission_passed, bool):
            raise TypeError("decoded look admission flag must be boolean")
        if (
            not math.isfinite(self.normalized_excess)
            or self.normalized_excess < 0.0
            or not math.isfinite(self.selection_penalty_points)
            or not 0.0 <= self.selection_penalty_points <= 20.0
        ):
            raise ValueError("decoded look objective scalars are invalid")

    def rank(self) -> tuple[int, float, float]:
        return (
            int(self.admission_passed),
            -self.normalized_excess,
            -self.selection_penalty_points,
        )

    def strictly_improves(self, previous: "DecodedLookObjective") -> bool:
        if not isinstance(previous, DecodedLookObjective):
            raise TypeError("decoded look comparison needs a prior objective")
        return self.rank() > previous.rank()


def _sha256(value: Any, field: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class GroupDecodedLookLineage:
    """Cryptographic chain binding one refinement to its exact predecessor."""

    iteration: int
    root_render_plan_binding_sha256: str
    input_render_plan_binding_sha256: str
    output_render_plan_binding_sha256: str
    decoded_proxy_content_sha256: str
    decoded_proxy_evidence_sha256: str
    decoded_proxy_review_binding_sha256: str
    config_sha256: str
    previous_lineage_sha256: str | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.iteration, bool)
            or not isinstance(self.iteration, int)
            or not 1 <= self.iteration <= 2
        ):
            raise ValueError("GROUP decoded look lineage iteration is invalid")
        for name in (
            "root_render_plan_binding_sha256",
            "input_render_plan_binding_sha256",
            "output_render_plan_binding_sha256",
            "decoded_proxy_content_sha256",
            "decoded_proxy_evidence_sha256",
            "decoded_proxy_review_binding_sha256",
            "config_sha256",
        ):
            _sha256(getattr(self, name), f"GROUP decoded look lineage {name}")
        if self.iteration == 1:
            if self.previous_lineage_sha256 is not None:
                raise ValueError("first GROUP decoded look lineage has a predecessor")
            if (
                self.root_render_plan_binding_sha256
                != self.input_render_plan_binding_sha256
            ):
                raise ValueError(
                    "first GROUP decoded look lineage root is inconsistent"
                )
        else:
            _sha256(
                self.previous_lineage_sha256,
                "GROUP decoded look previous lineage",
            )

    @property
    def lineage_sha256(self) -> str:
        return _canonical_sha256(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract": LINEAGE_CONTRACT,
            "iteration": self.iteration,
            "root_render_plan_binding_sha256": (
                self.root_render_plan_binding_sha256
            ),
            "input_render_plan_binding_sha256": (
                self.input_render_plan_binding_sha256
            ),
            "output_render_plan_binding_sha256": (
                self.output_render_plan_binding_sha256
            ),
            "decoded_proxy_content_sha256": self.decoded_proxy_content_sha256,
            "decoded_proxy_evidence_sha256": self.decoded_proxy_evidence_sha256,
            "decoded_proxy_review_binding_sha256": (
                self.decoded_proxy_review_binding_sha256
            ),
            "config_sha256": self.config_sha256,
            "previous_lineage_sha256": self.previous_lineage_sha256,
        }
        if include_digest:
            payload["lineage_sha256"] = self.lineage_sha256
        return payload


@dataclass(frozen=True, slots=True)
class GroupDecodedLookRefinement:
    render_plan: RenderPlan
    iteration: int
    applied: bool
    source_residual_adjustments: tuple[
        tuple[int, SourceLookAdjustment], ...
    ]
    diagnostics: tuple[dict[str, Any], ...]
    lineage: GroupDecodedLookLineage

    def __post_init__(self) -> None:
        if not isinstance(self.render_plan, RenderPlan):
            raise TypeError("GROUP decoded look refinement RenderPlan is invalid")
        if not isinstance(self.lineage, GroupDecodedLookLineage):
            raise TypeError("GROUP decoded look refinement lineage is invalid")
        if self.iteration != self.lineage.iteration:
            raise ValueError(
                "GROUP decoded look refinement iteration disagrees with lineage"
            )
        if not isinstance(self.applied, bool):
            raise TypeError("GROUP decoded look applied flag must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_basis": ANALYSIS_BASIS,
            "iteration": self.iteration,
            "applied": self.applied,
            "source_residual_adjustments": [
                {
                    "source_index": source_index,
                    "brightness": adjustment.brightness,
                    "contrast": adjustment.contrast,
                    "saturation": adjustment.saturation,
                    "temperature_balance": adjustment.temperature_balance,
                    "tint_balance": adjustment.tint_balance,
                    "detail_enhancement": adjustment.detail_enhancement,
                    "identity": adjustment.is_identity,
                }
                for source_index, adjustment in self.source_residual_adjustments
            ],
            "diagnostics": list(self.diagnostics),
            "lineage": self.lineage.to_dict(),
            "acceptance_policy": (
                "same_exact_cut_path_strict_decoded_look_objective_improvement_"
                "with_nonlook_gate_nonregression_else_rollback"
            ),
            "sharpness_policy": "positive_detail_enhancement_only_never_blur",
            "raw_frames_persisted": False,
            "original_pixels_serialized": False,
        }


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _config_sha256(config: GroupDecodedLookConfig) -> str:
    return _canonical_sha256(
        {
            item.name: getattr(config, item.name)
            for item in fields(GroupDecodedLookConfig)
        }
    )


def _render_plan_binding_sha256(render_plan: RenderPlan) -> str:
    if not isinstance(render_plan, RenderPlan):
        raise TypeError("GROUP decoded look requires a RenderPlan")
    source_content_sha256 = tuple(
        verified_file_content_sha256(path) for path in render_plan.sources
    )
    return _canonical_sha256(
        _render_timeline_binding_payload(
            render_plan,
            segment_frame_counts(render_plan),
            source_content_sha256=source_content_sha256,
        )
    )


def _bound_decoded_proxy_identity(
    rendered_proxy_review: Mapping[str, Any],
    *,
    render_plan: RenderPlan,
) -> tuple[str, str, str, str]:
    """Validate the producer's signed evidence and exact RenderPlan binding."""

    if not isinstance(rendered_proxy_review, Mapping):
        raise TypeError("GROUP decoded proxy review must be a mapping")
    raw_evidence = rendered_proxy_review.get("segment_interior_geometry")
    if not isinstance(raw_evidence, Mapping):
        raise ValueError("GROUP decoded proxy exact plan evidence is missing")
    evidence = dict(raw_evidence)
    evidence_sha256 = _sha256(
        evidence.pop("evidence_sha256", None),
        "GROUP decoded proxy evidence",
    )
    if _canonical_sha256(evidence) != evidence_sha256:
        raise ValueError("GROUP decoded proxy evidence SHA-256 mismatch")
    if (
        evidence.get("schema_version")
        != SEGMENT_INTERIOR_GEOMETRY_SCHEMA_VERSION
        or evidence.get("analysis_basis")
        != SEGMENT_INTERIOR_GEOMETRY_ANALYSIS_BASIS
        or evidence.get("contract") != SEGMENT_INTERIOR_GEOMETRY_CONTRACT
    ):
        raise ValueError("GROUP decoded proxy exact plan evidence contract changed")

    raw_proxy = rendered_proxy_review.get("proxy")
    if not isinstance(raw_proxy, Mapping) or any(
        field not in raw_proxy for field in _STABLE_PROXY_IDENTITY_FIELDS
    ):
        raise ValueError("GROUP decoded proxy identity is incomplete")
    proxy_identity = {
        field: raw_proxy[field] for field in _STABLE_PROXY_IDENTITY_FIELDS
    }
    for field in ("width", "height", "decoded_frame_count"):
        value = proxy_identity[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"GROUP decoded proxy {field} is invalid")
    for field in ("fps", "duration_seconds"):
        if _finite_number(proxy_identity[field], f"GROUP decoded proxy {field}") <= 0:
            raise ValueError(f"GROUP decoded proxy {field} must be positive")
    render_codec = proxy_identity["render_codec"]
    if render_codec is not None and (
        not isinstance(render_codec, str) or not render_codec.strip()
    ):
        raise ValueError("GROUP decoded proxy render codec is invalid")
    proxy_content_sha256 = _sha256(
        proxy_identity["content_sha256"],
        "GROUP decoded proxy content",
    )
    if evidence.get("proxy_identity") != proxy_identity:
        raise ValueError("GROUP decoded proxy evidence identity mismatch")
    if evidence.get("proxy_identity_sha256") != _canonical_sha256(proxy_identity):
        raise ValueError("GROUP decoded proxy evidence identity hash mismatch")

    render_plan_binding_sha256 = _render_plan_binding_sha256(render_plan)
    if (
        evidence.get("render_timeline_binding_sha256")
        != render_plan_binding_sha256
    ):
        raise ValueError("GROUP decoded proxy exact RenderPlan binding mismatch")
    raw_admission = rendered_proxy_review.get("perceptual_source_admission")
    if not isinstance(raw_admission, Mapping):
        raise ValueError("GROUP decoded proxy perceptual admission is missing")
    review_binding_sha256 = _canonical_sha256(
        {
            "proxy_identity": proxy_identity,
            "segment_interior_geometry_evidence_sha256": evidence_sha256,
            "perceptual_source_admission": raw_admission,
        }
    )
    return (
        render_plan_binding_sha256,
        proxy_content_sha256,
        evidence_sha256,
        review_binding_sha256,
    )


def _perceptual_admission(
    rendered_proxy_review: Mapping[str, Any],
) -> Mapping[str, Any]:
    raw = rendered_proxy_review.get("perceptual_source_admission")
    if not isinstance(raw, Mapping):
        raise ValueError("GROUP decoded proxy perceptual admission is missing")
    if raw.get("analysis_basis") != PERCEPTUAL_SOURCE_ADMISSION_ANALYSIS_BASIS:
        raise ValueError("GROUP decoded proxy admission analysis basis changed")
    domains = raw.get("index_domains")
    if not isinstance(domains, Mapping) or domains.get("source_index") != (
        "render_plan_sources_array_index"
    ):
        raise ValueError("GROUP decoded proxy admission source domain changed")
    return raw


def decoded_look_objective(
    rendered_proxy_review: Mapping[str, Any],
) -> DecodedLookObjective:
    """Extract the scalar-only objective used to accept a refined rerender."""

    admission = _perceptual_admission(rendered_proxy_review)
    raw_gate = admission.get("admission")
    source = admission.get("source_median_consistency")
    rapid = admission.get("rapid_quality_alternation")
    if not all(isinstance(value, Mapping) for value in (raw_gate, source, rapid)):
        raise ValueError("GROUP decoded proxy look evidence is malformed")
    assert isinstance(raw_gate, Mapping)
    assert isinstance(source, Mapping)
    assert isinstance(rapid, Mapping)
    passed = raw_gate.get("passed")
    if not isinstance(passed, bool):
        raise ValueError("GROUP decoded proxy look gate flag is invalid")
    penalty = _finite_number(
        raw_gate.get("selection_penalty_points"),
        "GROUP decoded look selection penalty",
    )
    normalized_excesses: list[float] = []
    worst_pairs = source.get("worst_source_pairs")
    if not isinstance(worst_pairs, Mapping):
        raise ValueError("GROUP decoded proxy worst-source evidence is missing")
    for metric in ("sharpness_log", "luma", "chroma"):
        raw_pair = worst_pairs.get(metric)
        if raw_pair is None:
            raise ValueError(
                f"GROUP decoded proxy {metric} worst-source evidence is incomplete"
            )
        if not isinstance(raw_pair, Mapping):
            raise ValueError("GROUP decoded proxy worst-source pair is malformed")
        value = _finite_number(raw_pair.get("value"), f"{metric} value")
        maximum = _finite_number(
            raw_pair.get("maximum_allowed"), f"{metric} maximum"
        )
        source_indices = raw_pair.get("source_indices")
        if (
            isinstance(source_indices, (str, bytes))
            or not isinstance(source_indices, Sequence)
            or len(source_indices) != 2
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 0
                for item in source_indices
            )
        ):
            raise ValueError("GROUP decoded proxy worst-source indices are invalid")
        if value < 0.0 or maximum <= 0.0:
            raise ValueError("GROUP decoded proxy metric scalar is invalid")
        normalized_excesses.append(max(0.0, value / maximum - 1.0))
    gate_applicable = rapid.get("gate_applicable")
    if not isinstance(gate_applicable, bool):
        raise ValueError("GROUP decoded rapid gate applicability is invalid")
    maximum = _finite_number(
        rapid.get("maximum_allowed_high_quality_shock_fraction"),
        "GROUP decoded rapid maximum",
    )
    if not 0.0 < maximum <= 1.0:
        raise ValueError("GROUP decoded rapid maximum must be in (0, 1]")
    fraction = rapid.get("high_quality_shock_fraction")
    if gate_applicable:
        if fraction is None:
            raise ValueError("GROUP decoded rapid evidence is incomplete")
        rapid_fraction = _finite_number(
            fraction, "GROUP decoded rapid fraction"
        )
        if not 0.0 <= rapid_fraction <= 1.0:
            raise ValueError("GROUP decoded rapid fraction must be normalized")
        normalized_excesses.append(
            max(0.0, rapid_fraction / maximum - 1.0)
        )
    elif fraction is not None:
        rapid_fraction = _finite_number(
            fraction, "GROUP decoded rapid fraction"
        )
        if not 0.0 <= rapid_fraction <= 1.0:
            raise ValueError("GROUP decoded rapid fraction must be normalized")
    return DecodedLookObjective(
        admission_passed=passed,
        normalized_excess=sum(normalized_excesses),
        selection_penalty_points=penalty,
    )


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))


_CHROMA_RESPONSE = np.asarray(
    ((-0.036559999, -0.421039015), (0.392331742, 0.249684229)),
    dtype=np.float64,
)
_CHROMA_CODE_RESPONSE = np.asarray(
    ((-86.0, -64.0), (74.0, -80.0)),
    dtype=np.float64,
)


def _bounded_chroma_residual(
    chroma: np.ndarray,
    target: np.ndarray,
    *,
    base: SourceLookAdjustment,
    config: GroupDecodedLookConfig,
) -> tuple[float, float, float]:
    """Fit the closest residual whose composed one-pass grade stays bounded."""

    saturation_low = max(
        1.0 - config.maximum_saturation_step,
        config.minimum_cumulative_saturation / base.saturation,
    )
    saturation_high = min(
        1.0 + config.maximum_saturation_step,
        config.maximum_cumulative_saturation / base.saturation,
    )
    if saturation_low > saturation_high + 1e-12:
        raise ValueError("GROUP decoded chroma composition has no bounded saturation")
    # The response is smooth in saturation.  A fixed dense lattice plus the
    # identity and exact interval endpoints is deterministic, cheap for the
    # small GROUP source set, and prevents the later one-pass composition from
    # discarding an apparently successful sequential correction at a cumulative
    # clamp.
    saturation_candidates = tuple(
        sorted(
            {
                saturation_low,
                saturation_high,
                _clip(1.0, saturation_low, saturation_high),
                *(
                    saturation_low
                    + (saturation_high - saturation_low) * index / 160.0
                    for index in range(161)
                ),
            }
        )
    )
    best: (
        tuple[
            tuple[float, float, float, float, float],
            tuple[float, float, float],
        ]
        | None
    ) = None
    for saturation in saturation_candidates:
        temperature_low = max(
            -config.maximum_temperature_step,
            -config.maximum_cumulative_temperature
            - base.temperature_balance * saturation,
        )
        temperature_high = min(
            config.maximum_temperature_step,
            config.maximum_cumulative_temperature
            - base.temperature_balance * saturation,
        )
        tint_low = max(
            -config.maximum_tint_step,
            -config.maximum_cumulative_tint
            - base.tint_balance * saturation,
        )
        tint_high = min(
            config.maximum_tint_step,
            config.maximum_cumulative_tint
            - base.tint_balance * saturation,
        )
        if temperature_low > temperature_high or tint_low > tint_high:
            continue
        for temperature_state in (-1, 0, 1):
            for tint_state in (-1, 0, 1):
                parameters = np.zeros(2, dtype=np.float64)
                free: list[int] = []
                for index, (state, low, high) in enumerate(
                    (
                        (temperature_state, temperature_low, temperature_high),
                        (tint_state, tint_low, tint_high),
                    )
                ):
                    if state < 0:
                        parameters[index] = low
                    elif state > 0:
                        parameters[index] = high
                    else:
                        free.append(index)
                remaining = (
                    target
                    - chroma * saturation
                    - _CHROMA_RESPONSE @ parameters
                )
                if free:
                    columns = _CHROMA_RESPONSE[:, free]
                    solution, *_ = np.linalg.lstsq(columns, remaining, rcond=None)
                    for index, value in zip(free, solution, strict=True):
                        parameters[index] = float(value)
                if not (
                    temperature_low - 1e-12
                    <= parameters[0]
                    <= temperature_high + 1e-12
                    and tint_low - 1e-12
                    <= parameters[1]
                    <= tint_high + 1e-12
                ):
                    continue
                output = (
                    chroma * saturation
                    + _CHROMA_RESPONSE @ parameters
                )
                error = float(np.linalg.norm(output - target))
                regularization = (
                    (saturation - 1.0) ** 2
                    + float(parameters[0]) ** 2
                    + float(parameters[1]) ** 2
                )
                rank = (
                    error,
                    regularization,
                    saturation,
                    float(parameters[0]),
                    float(parameters[1]),
                )
                value = (
                    saturation,
                    float(parameters[0]),
                    float(parameters[1]),
                )
                if best is None or rank < best[0]:
                    best = (rank, value)
    if best is None:  # pragma: no cover - validated bounds always include identity
        raise ValueError("GROUP decoded chroma composition has no bounded solution")
    return best[1]


def _compose(
    base: SourceLookAdjustment,
    residual: SourceLookAdjustment,
    *,
    config: GroupDecodedLookConfig,
) -> SourceLookAdjustment:
    contrast = _clip(
        base.contrast * residual.contrast,
        config.minimum_cumulative_contrast,
        config.maximum_cumulative_contrast,
    )
    brightness = _clip(
        base.brightness * residual.contrast + residual.brightness,
        -config.maximum_cumulative_brightness,
        config.maximum_cumulative_brightness,
    )
    saturation = _clip(
        base.saturation * residual.saturation,
        config.minimum_cumulative_saturation,
        config.maximum_cumulative_saturation,
    )
    base_offset = _CHROMA_CODE_RESPONSE @ np.asarray(
        (base.temperature_balance, base.tint_balance), dtype=np.float64
    )
    residual_offset = _CHROMA_CODE_RESPONSE @ np.asarray(
        (residual.temperature_balance, residual.tint_balance), dtype=np.float64
    )
    combined = np.linalg.solve(
        _CHROMA_CODE_RESPONSE,
        base_offset * residual.saturation + residual_offset,
    )
    return SourceLookAdjustment(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        temperature_balance=_clip(
            float(combined[0]),
            -config.maximum_cumulative_temperature,
            config.maximum_cumulative_temperature,
        ),
        tint_balance=_clip(
            float(combined[1]),
            -config.maximum_cumulative_tint,
            config.maximum_cumulative_tint,
        ),
        detail_enhancement=_clip(
            base.detail_enhancement + residual.detail_enhancement,
            0.0,
            config.maximum_cumulative_detail,
        ),
    )


def _source_rows(
    rendered_proxy_review: Mapping[str, Any],
    *,
    render_plan: RenderPlan,
    config: GroupDecodedLookConfig,
) -> tuple[dict[str, float | int], ...]:
    admission = _perceptual_admission(rendered_proxy_review)
    raw_source = admission.get("source_median_consistency")
    if not isinstance(raw_source, Mapping):
        raise ValueError("GROUP decoded proxy source medians are missing")
    raw_rows = raw_source.get("per_source")
    if not isinstance(raw_rows, list) or len(raw_rows) < 3:
        raise ValueError("GROUP decoded proxy needs at least three source medians")
    rows: list[dict[str, float | int]] = []
    seen: set[int] = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("GROUP decoded proxy source row is malformed")
        source_index = raw.get("source_index")
        decoded_count = raw.get("decoded_output_frame_count")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or not 0 <= source_index < len(render_plan.sources)
            or source_index in seen
            or isinstance(decoded_count, bool)
            or not isinstance(decoded_count, int)
            or decoded_count < config.minimum_decoded_output_frames_per_source
        ):
            raise ValueError("GROUP decoded proxy source binding/coverage is invalid")
        row: dict[str, float | int] = {
            "source_index": source_index,
            "decoded_output_frame_count": decoded_count,
        }
        for name in (
            "median_detail",
            "median_luma",
            "median_chroma_a",
            "median_chroma_b",
            "median_contrast",
        ):
            row[name] = _finite_number(raw.get(name), f"GROUP decoded {name}")
        if (
            float(row["median_detail"]) <= 0.0
            or not 0.0 <= float(row["median_luma"]) <= 1.0
            or not 0.0 <= float(row["median_contrast"]) <= 1.0
        ):
            raise ValueError("GROUP decoded proxy source scalar is implausible")
        seen.add(source_index)
        rows.append(row)
    used = {segment.source_index for segment in render_plan.segments}
    if seen != used:
        raise ValueError("GROUP decoded proxy source rows do not bind exact plan use")
    return tuple(rows)


def refine_group_render_plan_from_decoded_proxy(
    render_plan: RenderPlan,
    *,
    rendered_proxy_review: Mapping[str, Any],
    iteration: int,
    previous_refinement: GroupDecodedLookRefinement | None = None,
    config: GroupDecodedLookConfig | None = None,
) -> GroupDecodedLookRefinement:
    """Propose one bounded same-path output-domain look correction."""

    settings = config or GroupDecodedLookConfig()
    if (
        isinstance(iteration, bool)
        or not isinstance(iteration, int)
        or not 1 <= iteration <= settings.maximum_refinement_iterations
    ):
        raise ValueError("GROUP decoded look iteration is outside the fixed bound")
    (
        input_render_plan_binding_sha256,
        decoded_proxy_content_sha256,
        decoded_proxy_evidence_sha256,
        decoded_proxy_review_binding_sha256,
    ) = _bound_decoded_proxy_identity(
        rendered_proxy_review,
        render_plan=render_plan,
    )
    # The refinement consumes the same complete scalar contract used by the
    # accept/rollback objective.  Incomplete source or rapid evidence must not
    # be allowed to author pixels even when the per-source rows happen to exist.
    decoded_look_objective(rendered_proxy_review)
    config_sha256 = _config_sha256(settings)
    if iteration == 1:
        if previous_refinement is not None:
            raise ValueError("first GROUP decoded look iteration cannot replay lineage")
        root_render_plan_binding_sha256 = input_render_plan_binding_sha256
        previous_lineage_sha256 = None
    else:
        if not isinstance(previous_refinement, GroupDecodedLookRefinement):
            raise ValueError(
                "second GROUP decoded look iteration requires exact prior refinement"
            )
        if previous_refinement.iteration != iteration - 1:
            raise ValueError("GROUP decoded look iteration lineage is not contiguous")
        previous_output_binding_sha256 = _render_plan_binding_sha256(
            previous_refinement.render_plan
        )
        if (
            previous_refinement.lineage.output_render_plan_binding_sha256
            != previous_output_binding_sha256
            or input_render_plan_binding_sha256
            != previous_output_binding_sha256
        ):
            raise ValueError(
                "GROUP decoded look predecessor RenderPlan binding changed"
            )
        if previous_refinement.lineage.config_sha256 != config_sha256:
            raise ValueError(
                "GROUP decoded look config changed inside refinement lineage"
            )
        root_render_plan_binding_sha256 = (
            previous_refinement.lineage.root_render_plan_binding_sha256
        )
        previous_lineage_sha256 = previous_refinement.lineage.lineage_sha256
    rows = _source_rows(
        rendered_proxy_review,
        render_plan=render_plan,
        config=settings,
    )
    target_luma = float(median(float(row["median_luma"]) for row in rows))
    target_contrast = float(
        median(float(row["median_contrast"]) for row in rows)
    )
    base_by_source: dict[int, SourceLookAdjustment] = {}
    for segment in render_plan.segments:
        adjustment = segment.look_adjustment or SourceLookAdjustment()
        existing = base_by_source.setdefault(segment.source_index, adjustment)
        if existing != adjustment:
            raise ValueError(
                "GROUP decoded source medians require one exact base look per source"
            )
    unadjusted_chromas: list[np.ndarray] = []
    for row in rows:
        source_index = int(row["source_index"])
        observed = np.asarray(
            (float(row["median_chroma_a"]), float(row["median_chroma_b"])),
            dtype=np.float64,
        )
        base = base_by_source[source_index]
        base_offset = _CHROMA_RESPONSE @ np.asarray(
            (base.temperature_balance, base.tint_balance),
            dtype=np.float64,
        )
        unadjusted_chromas.append(
            (observed - base_offset) / base.saturation
        )
    # Use the mean of the recovered pre-grade colors as a stable consensus.
    # Unlike a coordinate median, this does not strand opposite XOXZ-like
    # chroma extremes on different cumulative clamps during round two.
    target_chroma = np.mean(np.stack(unadjusted_chromas), axis=0)
    target_a = float(target_chroma[0])
    target_b = float(target_chroma[1])
    target_log_detail = float(
        median(math.log(float(row["median_detail"]) + 1e-4) for row in rows)
    )
    residual_by_source: dict[int, SourceLookAdjustment] = {}
    diagnostics: list[dict[str, Any]] = []
    for row in rows:
        source_index = int(row["source_index"])
        luma = float(row["median_luma"])
        contrast = float(row["median_contrast"])
        chroma = np.asarray(
            (float(row["median_chroma_a"]), float(row["median_chroma_b"])),
            dtype=np.float64,
        )
        saturation, temperature, tint = _bounded_chroma_residual(
            chroma,
            target_chroma,
            base=base_by_source[source_index],
            config=settings,
        )
        detail_log_residual = target_log_detail - math.log(
            float(row["median_detail"]) + 1e-4
        )
        residual = SourceLookAdjustment(
            brightness=_clip(
                0.65 * (target_luma - luma),
                -settings.maximum_brightness_step,
                settings.maximum_brightness_step,
            ),
            contrast=_clip(
                (target_contrast + 1e-4) / (contrast + 1e-4),
                1.0 - settings.maximum_contrast_step,
                1.0 + settings.maximum_contrast_step,
            ),
            saturation=saturation,
            temperature_balance=_clip(
                float(temperature),
                -settings.maximum_temperature_step,
                settings.maximum_temperature_step,
            ),
            tint_balance=_clip(
                float(tint),
                -settings.maximum_tint_step,
                settings.maximum_tint_step,
            ),
            # Never soften a sharper source to make the numeric span pass.
            # Only recover detail on sources below the cross-source median.
            detail_enhancement=_clip(
                max(0.0, detail_log_residual) / 1.05,
                0.0,
                settings.maximum_detail_step,
            ),
        )
        residual_by_source[source_index] = residual
        diagnostics.append(
            {
                "source_index": source_index,
                "decoded_output_frame_count": int(
                    row["decoded_output_frame_count"]
                ),
                "observed": {
                    "luma": luma,
                    "contrast": contrast,
                    "chroma_a": float(chroma[0]),
                    "chroma_b": float(chroma[1]),
                    "detail": float(row["median_detail"]),
                },
                "target": {
                    "luma": target_luma,
                    "contrast": target_contrast,
                    "chroma_a": target_a,
                    "chroma_b": target_b,
                    "log_detail": target_log_detail,
                },
                "residual_identity": residual.is_identity,
            }
        )
    refined_segments = tuple(
        replace(
            segment,
            look_adjustment=_compose(
                segment.look_adjustment or SourceLookAdjustment(),
                residual_by_source[segment.source_index],
                config=settings,
            ),
        )
        for segment in render_plan.segments
    )
    applied = any(not value.is_identity for value in residual_by_source.values())
    refined_render_plan = replace(render_plan, segments=refined_segments)
    output_render_plan_binding_sha256 = _render_plan_binding_sha256(
        refined_render_plan
    )
    lineage = GroupDecodedLookLineage(
        iteration=iteration,
        root_render_plan_binding_sha256=root_render_plan_binding_sha256,
        input_render_plan_binding_sha256=input_render_plan_binding_sha256,
        output_render_plan_binding_sha256=output_render_plan_binding_sha256,
        decoded_proxy_content_sha256=decoded_proxy_content_sha256,
        decoded_proxy_evidence_sha256=decoded_proxy_evidence_sha256,
        decoded_proxy_review_binding_sha256=(
            decoded_proxy_review_binding_sha256
        ),
        config_sha256=config_sha256,
        previous_lineage_sha256=previous_lineage_sha256,
    )
    return GroupDecodedLookRefinement(
        render_plan=refined_render_plan,
        iteration=iteration,
        applied=applied,
        source_residual_adjustments=tuple(sorted(residual_by_source.items())),
        diagnostics=tuple(diagnostics),
        lineage=lineage,
    )


__all__ = [
    "ANALYSIS_BASIS",
    "LINEAGE_CONTRACT",
    "DecodedLookObjective",
    "GroupDecodedLookConfig",
    "GroupDecodedLookLineage",
    "GroupDecodedLookRefinement",
    "decoded_look_objective",
    "refine_group_render_plan_from_decoded_proxy",
]
