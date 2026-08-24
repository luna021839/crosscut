"""Fail-closed semi-Markov beam decoding for learned StageEdit heads.

The transformer predicts evidence at candidate master times; it does not by
itself produce a legal edit.  This module performs a causal, variable-duration
search over those predictions while keeping every renderer and editorial hard
constraint explicit.  It is deliberately not imported by the production
pipeline yet.  A caller must provide a checkpoint directory that passes the
production readiness contract before any decoding work is attempted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ive import AspectRatio
from .professional_montage_policy import (
    CLOSE_BURST_EVENT_EVIDENCE_CONTRACT,
    DURATION_PREFERRED_TARGET_QUANTILE,
    DURATION_PRODUCT_MINIMUM_QUANTILE,
    HIGHLIGHT_CLOSE_DURATION_STATISTIC_PATH,
    HIGHLIGHT_DURATION_EVIDENCE_CONTRACT,
    HIGHLIGHT_NONWIDE_DURATION_STATISTIC_PATH,
    HIGHLIGHT_WIDE_RUN_EVIDENCE_CONTRACT,
    HIGHLIGHT_WIDE_RUN_STATISTIC_PATH,
    REFERENCE_FLOOR_QUANTILE,
    WIDE_RUN_HARD_MAXIMUM_QUANTILE,
    WIDE_RUN_PREFERRED_MAXIMUM_QUANTILE,
    close_burst_policy_binding_payload,
    close_burst_policy_binding_sha256,
)
from .render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    RenderSegment,
    RenderTransition,
    TransitionKind,
    minimum_roll_safe_zoom,
)
from .section_director import SectionPlan, SectionRole
from .source_derivative_diversity import SourceDerivativeFamilyMetadata
from .stage_edit import StageBoundary, StageEdit, StageShot
from .stage_edit_dynamic_crop_semimarkov_adapter import (
    DYNAMIC_CROP_V2_SEMIMARKOV_ADAPTER_CONTRACT,
    DynamicCropHeadPredictionV2,
    DynamicCropTrackDecodeV2,
)
from .stage_edit_dynamic_crop_semimarkov_adapter_v3 import (
    DYNAMIC_CROP_V3_SEMIMARKOV_ADAPTER_CONTRACT,
    DynamicCropHeadPredictionV3,
    DynamicCropTrackDecodeV3,
)
from .stage_edit_continuity_authority import (
    CONTINUITY_AUTHORITY_CONTRACT,
    HANDCRAFTED_CONTINUITY_AUTHORITY,
    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
    STAGE_EDIT_CONTINUITY_AUTHORITIES as _CONTINUITY_AUTHORITIES,
)
from .stage_edit_source_pair_scorer import (
    STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
    StageEditSourcePairScorer,
)
from .stage_edit_temporal_pose_motion import (
    TemporalPoseMotionThresholds,
    require_scalar_stage_edit_pose_payload,
)
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    STAGE_EDIT_DECISION_HISTORY_OWNER,
    STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES,
    StageEditTemporalPoseSidecar,
    StageEditTokenSchema,
    stage_edit_readiness_report,
)
from .stage_edit_transition_authority import (
    LEARNED_TRANSITION_AUTHORITY,
    RENDERER_OWNED_ALLOWED_MODEL_LABELS,
    RENDERER_OWNED_TRANSITION_AUTHORITY,
    StageEditTransitionAuthority,
)


DECODER_SCHEMA_VERSION = 14
DECODER_CONTRACT = "stage_edit_semimarkov_beam_v11"
_TIME_EPSILON = 1e-6
_SHOT_SIZES = frozenset(
    {"unknown", "wide", "full", "medium", "closeup", "extreme_closeup"}
)
_WIDE_SHOT_SIZES = frozenset({"wide", "full"})
_CLOSEUP_SHOT_SIZES = frozenset({"closeup", "extreme_closeup"})
_NONWIDE_SHOT_SIZES = frozenset({"medium", "closeup", "extreme_closeup"})
MONTAGE_GRAMMAR_CONTRACT = (
    "ive_human_reference_wide_transition_floor_plus_"
    "exact_section_and_phrase_wide_anchor_plus_formation_evidenced_"
    "highlight_burst_reset_reference_cadence_and_close_burst_soft_target_v7"
)
HIGHLIGHT_AGGREGATE_CONTRACT = (
    "professional_montage_policy_v2_strict_highlight_duration_gate_v1"
)
HIGHLIGHT_CADENCE_EVIDENCE_CONTRACT = (
    "ive_20_human_reference_chorus_shot_boundary_rate_distribution_v1"
)
HANDCRAFTED_CONTINUITY_SCORER_CONTRACT = (
    "semimarkov_boundary_continuity_handcrafted_v1"
)
HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT = hashlib.sha256(
    HANDCRAFTED_CONTINUITY_SCORER_CONTRACT.encode("utf-8")
).hexdigest()
HIGHLIGHT_MOTION_CALIBRATION_CONTRACT = (
    "ive_human_reference_highlight_close_switch_"
    "motion_amount_plus_continuity_calibration_v3"
)
HIGHLIGHT_MOTION_CALIBRATION_QUANTILES = (0.10, 0.25, 0.50, 0.75)
_HIGHLIGHT_MOTION_ACCENT_MAXIMUM_RAW_BONUS = 0.12
TEMPORAL_POSE_MOTION_CALIBRATION_CONTRACT = (
    "source_exact_temporal_pose_motion_threshold_calibration_v1"
)
TEMPORAL_POSE_MOTION_CALIBRATION_ACTIVATION_AVAILABLE = False
SAME_TIME_POSE_SCALAR_EVIDENCE_CONTRACT = (
    "hash_bound_stage_edit_token_pose60_scalar_evidence_v1"
)
TEMPORAL_POSE_SIDECAR_SCALAR_EVIDENCE_CONTRACT = (
    "hash_bound_source_exact_temporal_pose_sidecar_scalar_evidence_v1"
)


def _redacted_content_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _public_scalar_pose_payload(
    payload: dict[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    require_scalar_stage_edit_pose_payload(payload, context=context)
    return payload


def _same_time_pose_scalar_evidence(
    embedding: Sequence[float] | None,
) -> dict[str, Any]:
    observed = embedding is not None
    return {
        "contract": SAME_TIME_POSE_SCALAR_EVIDENCE_CONTRACT,
        "observed": observed,
        "feature_dimension": len(embedding) if embedding is not None else None,
        "trusted_producer_contract": (
            SAME_TIME_POSE_SCALAR_EVIDENCE_CONTRACT if observed else None
        ),
        "content_sha256": (
            _redacted_content_sha256(
                {
                    "contract": SAME_TIME_POSE_SCALAR_EVIDENCE_CONTRACT,
                    "feature_dimension": len(embedding),
                    "coordinates": [float(value) for value in embedding],
                }
            )
            if embedding is not None
            else None
        ),
    }


def _temporal_sidecar_scalar_evidence(
    sidecar: StageEditTemporalPoseSidecar | None,
) -> dict[str, Any]:
    observed = sidecar is not None
    return {
        "contract": TEMPORAL_POSE_SIDECAR_SCALAR_EVIDENCE_CONTRACT,
        "observed": observed,
        "sample_count": len(sidecar.embeddings) if sidecar is not None else 0,
        "feature_dimension": (
            len(sidecar.embeddings[0]) if sidecar is not None else None
        ),
        "interval_seconds": (
            sidecar.interval_seconds if sidecar is not None else None
        ),
        "sampling_contract": (
            sidecar.sampling_contract if sidecar is not None else None
        ),
        "trusted_producer_contract": (
            sidecar.producer_backend if sidecar is not None else None
        ),
        "content_sha256": (
            _redacted_content_sha256(sidecar.to_dict())
            if sidecar is not None
            else None
        ),
        "enters_transformer_tensors": False,
    }


class StageEditSemiMarkovError(RuntimeError):
    """Base class for decoder failures that must not silently fall back."""


class StageEditCheckpointNotReadyError(StageEditSemiMarkovError):
    """Raised before search when learned inference is not production-ready."""


class StageEditDecodeInputError(StageEditSemiMarkovError, ValueError):
    """Raised for malformed or unsafe model-head input."""


class StageEditNoValidTimelineError(StageEditSemiMarkovError):
    """Raised when hard constraints eliminate every complete timeline."""


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise StageEditDecodeInputError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise StageEditDecodeInputError(f"{name} must be finite")
    return number


def _strict_bool_tuple(values: Sequence[bool], name: str) -> tuple[bool, ...]:
    result = tuple(values)
    if any(not isinstance(value, bool) for value in result):
        raise StageEditDecodeInputError(f"{name} must contain booleans")
    return result


def _float_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
    return tuple(_finite(value, f"{name}[{index}]") for index, value in enumerate(values))


@dataclass(frozen=True, slots=True)
class StageEditHighlightMotionCalibration:
    """Immutable profile-v3 input for motion-accent ranking.

    The current v2 montage policy does not contain a choreography-motion
    distribution.  This separate payload lets an audited source-pair corpus
    bind those observations without silently turning them into a hard quota.
    Runtime thresholds remain explicit on :class:`StageEditSemiMarkovConfig`.
    Its continuity distribution is deliberately an independent handcrafted
    motion qualifier.  It is not the primary boundary-continuity authority and
    therefore remains valid when that primary authority is learned.
    """

    analysis_basis: str
    human_reference_count: int
    source_pair_negative_samples: int
    maximum_sources: int
    confirmed_source_switch_count: int
    highlight_role_close_involved_switch_count: int
    build_close_involved_switch_count: int
    chorus_close_involved_switch_count: int
    bridge_close_involved_switch_count: int
    pose_pair_observed_switch_count: int
    continuity_observed_switch_count: int
    pose_embedding_cosine_observed_switch_count: int
    sharpness_pair_observed_switch_count: int
    pose_motion_minimum_quantiles: tuple[float, float, float, float]
    continuity_score_quantiles: tuple[float, float, float, float]
    pose_embedding_cosine_quantiles: tuple[float, float, float, float]
    sharpness_minimum_quantiles: tuple[float, float, float, float] | None = None
    continuity_authority: str = HANDCRAFTED_CONTINUITY_AUTHORITY
    continuity_scorer_contract: str = HANDCRAFTED_CONTINUITY_SCORER_CONTRACT

    def __post_init__(self) -> None:
        basis = str(self.analysis_basis).strip()
        if not basis:
            raise StageEditDecodeInputError(
                "highlight motion calibration analysis_basis cannot be blank"
            )
        object.__setattr__(self, "analysis_basis", basis)
        for name in (
            "human_reference_count",
            "maximum_sources",
            "confirmed_source_switch_count",
            "highlight_role_close_involved_switch_count",
            "pose_pair_observed_switch_count",
            "continuity_observed_switch_count",
            "pose_embedding_cosine_observed_switch_count",
            "sharpness_pair_observed_switch_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                if (
                    name == "sharpness_pair_observed_switch_count"
                    and not isinstance(value, bool)
                    and value == 0
                ):
                    continue
                raise StageEditDecodeInputError(
                    f"highlight motion calibration {name} must be a positive integer"
                )
        for name in (
            "build_close_involved_switch_count",
            "chorus_close_involved_switch_count",
            "bridge_close_involved_switch_count",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise StageEditDecodeInputError(
                    f"highlight motion calibration {name} must be a non-negative "
                    "integer"
                )
        negative_samples = self.source_pair_negative_samples
        if (
            isinstance(negative_samples, bool)
            or not isinstance(negative_samples, int)
            or negative_samples < 0
        ):
            raise StageEditDecodeInputError(
                "highlight motion calibration source_pair_negative_samples must "
                "be a non-negative integer"
            )
        if self.maximum_sources < 2:
            raise StageEditDecodeInputError(
                "highlight motion calibration maximum_sources must be at least two"
            )
        if not (
            self.highlight_role_close_involved_switch_count
            <= self.confirmed_source_switch_count
            and self.pose_pair_observed_switch_count
            <= self.highlight_role_close_involved_switch_count
            and self.continuity_observed_switch_count
            <= self.highlight_role_close_involved_switch_count
            and self.sharpness_pair_observed_switch_count
            <= self.highlight_role_close_involved_switch_count
            and self.pose_embedding_cosine_observed_switch_count
            <= self.highlight_role_close_involved_switch_count
            and (
                self.build_close_involved_switch_count
                + self.chorus_close_involved_switch_count
                + self.bridge_close_involved_switch_count
            )
            == self.highlight_role_close_involved_switch_count
        ):
            raise StageEditDecodeInputError(
                "highlight motion calibration observed counts are inconsistent"
            )

        def normalize_quantiles(
            values: Sequence[float],
            *,
            name: str,
            minimum: float,
            maximum: float,
        ) -> tuple[float, float, float, float]:
            normalized = _float_tuple(values, name)
            if len(normalized) != len(HIGHLIGHT_MOTION_CALIBRATION_QUANTILES):
                raise StageEditDecodeInputError(
                    f"{name} must contain P10, P25, P50, and P75"
                )
            if any(not minimum <= value <= maximum for value in normalized):
                raise StageEditDecodeInputError(
                    f"{name} values must lie in [{minimum}, {maximum}]"
                )
            if any(
                right + _TIME_EPSILON < left
                for left, right in zip(normalized, normalized[1:])
            ):
                raise StageEditDecodeInputError(
                    f"{name} values must be monotonic"
                )
            return normalized  # type: ignore[return-value]

        object.__setattr__(
            self,
            "pose_motion_minimum_quantiles",
            normalize_quantiles(
                self.pose_motion_minimum_quantiles,
                name="pose_motion_minimum_quantiles",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "continuity_score_quantiles",
            normalize_quantiles(
                self.continuity_score_quantiles,
                name="continuity_score_quantiles",
                minimum=-1.0,
                maximum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "pose_embedding_cosine_quantiles",
            normalize_quantiles(
                self.pose_embedding_cosine_quantiles,
                name="pose_embedding_cosine_quantiles",
                minimum=-1.0,
                maximum=1.0,
            ),
        )
        sharpness = self.sharpness_minimum_quantiles
        if self.sharpness_pair_observed_switch_count == 0:
            if sharpness is not None:
                raise StageEditDecodeInputError(
                    "sharpness quantiles require observed sharpness pairs"
                )
        elif sharpness is None:
            raise StageEditDecodeInputError(
                "observed sharpness pairs require sharpness quantiles"
            )
        else:
            object.__setattr__(
                self,
                "sharpness_minimum_quantiles",
                normalize_quantiles(
                    sharpness,
                    name="sharpness_minimum_quantiles",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        authority = str(self.continuity_authority).strip()
        scorer_contract = str(self.continuity_scorer_contract).strip()
        if authority != HANDCRAFTED_CONTINUITY_AUTHORITY:
            raise StageEditDecodeInputError(
                "the current highlight motion calibration requires handcrafted "
                "continuity authority"
            )
        if scorer_contract != HANDCRAFTED_CONTINUITY_SCORER_CONTRACT:
            raise StageEditDecodeInputError(
                "the current highlight motion calibration scorer contract must "
                "match handcrafted continuity"
            )
        object.__setattr__(self, "continuity_authority", authority)
        object.__setattr__(self, "continuity_scorer_contract", scorer_contract)

    def to_dict(self) -> dict[str, Any]:
        distributions: dict[str, Any] = {
            "pose_motion_minimum": {
                "observed_switch_count": self.pose_pair_observed_switch_count,
                "values": dict(
                    zip(
                        ("p10", "p25", "p50", "p75"),
                        self.pose_motion_minimum_quantiles,
                        strict=True,
                    )
                ),
            },
            "handcrafted_continuity_score": {
                "observed_switch_count": self.continuity_observed_switch_count,
                "values": dict(
                    zip(
                        ("p10", "p25", "p50", "p75"),
                        self.continuity_score_quantiles,
                        strict=True,
                    )
                ),
            },
            "same_time_pose_embedding_cosine_proxy": {
                "observed_switch_count": (
                    self.pose_embedding_cosine_observed_switch_count
                ),
                "values": dict(
                    zip(
                        ("p10", "p25", "p50", "p75"),
                        self.pose_embedding_cosine_quantiles,
                        strict=True,
                    )
                ),
                "runtime_pair_feature_available": True,
                "eligible_as_soft_bonus_qualifier": True,
                "used_for_threshold_activation": False,
                "default_runtime_threshold_bound": False,
                "hard_gate_derived_from_sparse_support": False,
                "direction_or_velocity_proven": False,
            },
            "sharpness_minimum": {
                "observed_switch_count": self.sharpness_pair_observed_switch_count,
                "values": (
                    dict(
                        zip(
                            ("p10", "p25", "p50", "p75"),
                            self.sharpness_minimum_quantiles,
                            strict=True,
                        )
                    )
                    if self.sharpness_minimum_quantiles is not None
                    else None
                ),
            },
        }
        payload: dict[str, Any] = {
            "contract": HIGHLIGHT_MOTION_CALIBRATION_CONTRACT,
            "schema_version": 3,
            "analysis_basis": self.analysis_basis,
            "quantile_method": "nearest_rank_empirical",
            "quantiles": list(HIGHLIGHT_MOTION_CALIBRATION_QUANTILES),
            "sampling": {
                "human_reference_count": self.human_reference_count,
                "source_pair_negative_samples": self.source_pair_negative_samples,
                "maximum_sources": self.maximum_sources,
            },
            "counts": {
                "confirmed_source_switch_count": (
                    self.confirmed_source_switch_count
                ),
                "highlight_role_close_involved_switch_count": (
                    self.highlight_role_close_involved_switch_count
                ),
                "highlight_role_close_involved_switch_count_by_role": {
                    "build": self.build_close_involved_switch_count,
                    "chorus_like": self.chorus_close_involved_switch_count,
                    "bridge": self.bridge_close_involved_switch_count,
                },
            },
            "continuity_authority": self.continuity_authority,
            "continuity_scorer_contract": self.continuity_scorer_contract,
            "continuity_qualifier": {
                "role": "independent_handcrafted_motion_eligibility_qualifier",
                "independent_from_primary_boundary_continuity": True,
                "authority": self.continuity_authority,
                "scorer_contract": self.continuity_scorer_contract,
                "scorer_fingerprint": (
                    HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
                ),
                "primary_boundary_score_reused": False,
            },
            "distributions": distributions,
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload["calibration_fingerprint_sha256"] = hashlib.sha256(
            canonical
        ).hexdigest()
        return payload


@dataclass(frozen=True, slots=True)
class StageEditTemporalPoseMotionCalibration:
    """Explicit evidence identity plus fully injected temporal thresholds."""

    analysis_basis: str
    source_evidence_sha256: str
    observed_boundary_count: int
    thresholds: TemporalPoseMotionThresholds

    def __post_init__(self) -> None:
        basis = str(self.analysis_basis).strip()
        if not basis:
            raise StageEditDecodeInputError(
                "temporal pose-motion calibration analysis_basis cannot be blank"
            )
        digest = str(self.source_evidence_sha256).strip()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise StageEditDecodeInputError(
                "temporal pose-motion source evidence must be lowercase SHA-256"
            )
        count = self.observed_boundary_count
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise StageEditDecodeInputError(
                "temporal pose-motion observed boundary count must be positive"
            )
        thresholds = self.thresholds
        if isinstance(thresholds, Mapping):
            try:
                thresholds = TemporalPoseMotionThresholds(**dict(thresholds))
            except (TypeError, ValueError) as exc:
                raise StageEditDecodeInputError(
                    "temporal pose-motion thresholds cannot be round-tripped"
                ) from exc
        if not isinstance(thresholds, TemporalPoseMotionThresholds):
            raise StageEditDecodeInputError(
                "temporal pose-motion thresholds have an invalid type"
            )
        object.__setattr__(self, "analysis_basis", basis)
        object.__setattr__(self, "source_evidence_sha256", digest)
        object.__setattr__(self, "thresholds", thresholds)

    def to_dict(self) -> dict[str, Any]:
        thresholds = {
            "minimum_boundary_motion_magnitude": (
                self.thresholds.minimum_boundary_motion_magnitude
            ),
            "minimum_endpoint_pose_cosine": (
                self.thresholds.minimum_endpoint_pose_cosine
            ),
            "minimum_velocity_direction_cosine": (
                self.thresholds.minimum_velocity_direction_cosine
            ),
            "minimum_phase_compatibility": (
                self.thresholds.minimum_phase_compatibility
            ),
            "pose_norm_epsilon": self.thresholds.pose_norm_epsilon,
        }
        payload: dict[str, Any] = {
            "contract": TEMPORAL_POSE_MOTION_CALIBRATION_CONTRACT,
            "analysis_basis": self.analysis_basis,
            "source_evidence_sha256": self.source_evidence_sha256,
            "observed_boundary_count": self.observed_boundary_count,
            "thresholds": thresholds,
            "default_thresholds_present": False,
            "hard_boundary_quota": 0,
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload["calibration_fingerprint_sha256"] = hashlib.sha256(
            canonical
        ).hexdigest()
        return payload


def _normalized_shot_size(value: str, name: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "long": "wide",
        "long_shot": "wide",
        "full_shot": "full",
        "medium_shot": "medium",
        "close_up": "closeup",
        "close": "closeup",
        "extreme_close_up": "extreme_closeup",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _SHOT_SIZES:
        raise StageEditDecodeInputError(
            f"{name} must be one of {', '.join(sorted(_SHOT_SIZES))}"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class StageEditSourceMetadata:
    """Static source-clock and editorial constraints for one source."""

    source_index: int
    source_id: str
    source_duration_seconds: float
    source_start_offset_seconds: float = 0.0
    internal_cut_times_seconds: tuple[float, ...] = ()
    maximum_occupancy_fraction: float = 0.60
    quality_log_prior: float = 0.0
    default_shot_size: str = "unknown"

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_index, bool)
            or not isinstance(self.source_index, int)
            or self.source_index < 0
        ):
            raise StageEditDecodeInputError("source_index must be non-negative")
        source_id = str(self.source_id).strip()
        if not source_id:
            raise StageEditDecodeInputError("source_id cannot be blank")
        object.__setattr__(self, "source_id", source_id)
        duration = _finite(self.source_duration_seconds, "source_duration_seconds")
        if duration <= 0.0:
            raise StageEditDecodeInputError("source_duration_seconds must be positive")
        object.__setattr__(self, "source_duration_seconds", duration)
        object.__setattr__(
            self,
            "source_start_offset_seconds",
            _finite(self.source_start_offset_seconds, "source_start_offset_seconds"),
        )
        cuts = tuple(
            sorted(
                _finite(value, f"internal_cut_times_seconds[{index}]")
                for index, value in enumerate(self.internal_cut_times_seconds)
            )
        )
        if len(set(cuts)) != len(cuts):
            raise StageEditDecodeInputError("internal source cuts must be unique")
        if any(value < 0.0 or value > duration for value in cuts):
            raise StageEditDecodeInputError("internal source cut is outside the source")
        object.__setattr__(self, "internal_cut_times_seconds", cuts)
        occupancy = _finite(
            self.maximum_occupancy_fraction, "maximum_occupancy_fraction"
        )
        if not 0.0 < occupancy <= 1.0:
            raise StageEditDecodeInputError(
                "maximum_occupancy_fraction must lie in (0, 1]"
            )
        object.__setattr__(self, "maximum_occupancy_fraction", occupancy)
        object.__setattr__(
            self,
            "quality_log_prior",
            _finite(self.quality_log_prior, "quality_log_prior"),
        )
        object.__setattr__(
            self,
            "default_shot_size",
            _normalized_shot_size(self.default_shot_size, "default_shot_size"),
        )

    def source_start_at(self, master_seconds: float) -> float:
        return float(master_seconds) + self.source_start_offset_seconds


@dataclass(frozen=True, slots=True)
class StageEditSourceContinuity:
    """Same-canonical-time geometry and look evidence for one source.

    These values are label-free source observations already present in the
    Transformer token.  Keeping them on the head frame lets each semi-Markov
    beam score the *actual* outgoing/incoming pair at a proposed cut without
    leaking human/reference edit history into the global Transformer forward.
    Missing observations remain ``None`` and receive only a small uncertainty
    penalty; they can never masquerade as a detected perfect match.
    """

    face_scale: float | None = None
    face_center_x: float | None = None
    face_center_y: float | None = None
    body_center_x: float | None = None
    body_center_y: float | None = None
    pose_motion: float | None = None
    camera_motion_x: float | None = None
    camera_motion_y: float | None = None
    camera_zoom_velocity: float | None = None
    camera_roll_normalized: float | None = None
    sharpness: float | None = None
    exposure: float | None = None
    color_temperature_normalized: float | None = None
    detected_person_count_normalized: float | None = None
    formation_person_count_normalized: float | None = None
    formation_pose_coverage: float | None = None
    # Exact same-master-time 60-D model-pose observation.  This is deliberately
    # appended rather than inserted into the Transformer schema: the values
    # already exist in every source token and are copied only for pairwise
    # runtime auditing after the global forward pass.
    pose_model_embedding: tuple[float, ...] | None = None
    # Two source-local real-model observations remain a non-Transformer
    # sidecar.  The previous head frame supplies the outgoing pre-boundary
    # window and the current frame supplies the incoming post-boundary window.
    temporal_pose_sidecar: StageEditTemporalPoseSidecar | None = None

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name in {"pose_model_embedding", "temporal_pose_sidecar"}:
                continue
            value = getattr(self, name)
            if value is None:
                continue
            object.__setattr__(self, name, _finite(value, name))
        for name in (
            "detected_person_count_normalized",
            "formation_person_count_normalized",
            "formation_pose_coverage",
        ):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise StageEditDecodeInputError(f"{name} must lie in [0, 1]")

        pose_embedding = self.pose_model_embedding
        if pose_embedding is not None:
            if isinstance(pose_embedding, (str, bytes)):
                raise StageEditDecodeInputError(
                    "pose_model_embedding must contain numeric coordinates"
                )
            try:
                normalized_pose = _float_tuple(
                    tuple(pose_embedding), "pose_model_embedding"
                )
            except TypeError as exc:
                raise StageEditDecodeInputError(
                    "pose_model_embedding must be a numeric sequence or None"
                ) from exc
            if len(normalized_pose) != len(
                STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES
            ):
                raise StageEditDecodeInputError(
                    "pose_model_embedding must contain exactly 60 values"
                )
            if any(not -1.0 <= value <= 1.0 for value in normalized_pose):
                raise StageEditDecodeInputError(
                    "pose_model_embedding values must lie in [-1, 1]"
                )
            object.__setattr__(
                self, "pose_model_embedding", normalized_pose
            )
        if self.temporal_pose_sidecar is not None and not isinstance(
            self.temporal_pose_sidecar, StageEditTemporalPoseSidecar
        ):
            raise StageEditDecodeInputError(
                "temporal_pose_sidecar must use StageEditTemporalPoseSidecar"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"pose_model_embedding", "temporal_pose_sidecar"}
        }
        payload["same_time_pose_evidence"] = _same_time_pose_scalar_evidence(
            self.pose_model_embedding
        )
        payload["temporal_pose_evidence"] = _temporal_sidecar_scalar_evidence(
            self.temporal_pose_sidecar
        )
        return _public_scalar_pose_payload(
            payload,
            context="StageEdit source continuity",
        )


@dataclass(frozen=True, slots=True)
class StageEditHeadFrame:
    """All StageEdit head outputs and hard masks at one candidate time.

    ``hard_source_gate_mask`` follows the transformer convention: ``True``
    means the source passed the hard gate.  It is combined with
    ``available_source_mask`` by the decoder and is never treated as a soft
    score.
    """

    master_time_seconds: float
    cut_logit: float
    source_logits: tuple[float, ...]
    duration_logits: tuple[float, ...]
    transition_logits: tuple[float, ...]
    crop: tuple[float, float, float, float]
    available_source_mask: tuple[bool, ...]
    hard_source_gate_mask: tuple[bool, ...]
    source_shot_sizes: tuple[str, ...] = ()
    source_continuity: tuple[StageEditSourceContinuity, ...] = ()
    candidate_id: str = ""
    dynamic_crop_v2: DynamicCropHeadPredictionV2 | None = None
    # These are copied from the exact serialized inference token.  They are
    # deliberately not reconstructed from wall-clock time in the decoder.
    section_progress: float | None = None
    phrase_progress: float | None = None
    section_role: str | None = None
    # V3 retains one source-exact trajectory per canonical source slot.  A
    # hard-gated/unavailable slot is represented by ``None``; an empty tuple
    # means this is a legacy or v2 frame.  Appending the field preserves the
    # positional constructor contract of every pre-v3 caller.
    dynamic_crop_v3_by_source: tuple[
        DynamicCropHeadPredictionV3 | None, ...
    ] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "master_time_seconds",
            _finite(self.master_time_seconds, "master_time_seconds"),
        )
        object.__setattr__(self, "cut_logit", _finite(self.cut_logit, "cut_logit"))
        object.__setattr__(
            self, "source_logits", _float_tuple(self.source_logits, "source_logits")
        )
        object.__setattr__(
            self,
            "duration_logits",
            _float_tuple(self.duration_logits, "duration_logits"),
        )
        object.__setattr__(
            self,
            "transition_logits",
            _float_tuple(self.transition_logits, "transition_logits"),
        )
        crop = _float_tuple(self.crop, "crop")
        if len(crop) != 4:
            raise StageEditDecodeInputError("crop must have zoom/x/y/roll values")
        object.__setattr__(self, "crop", crop)
        object.__setattr__(
            self,
            "available_source_mask",
            _strict_bool_tuple(
                self.available_source_mask, "available_source_mask"
            ),
        )
        object.__setattr__(
            self,
            "hard_source_gate_mask",
            _strict_bool_tuple(
                self.hard_source_gate_mask, "hard_source_gate_mask"
            ),
        )
        sizes = tuple(
            _normalized_shot_size(value, f"source_shot_sizes[{index}]")
            for index, value in enumerate(self.source_shot_sizes)
        )
        object.__setattr__(self, "source_shot_sizes", sizes)
        continuity = tuple(self.source_continuity)
        if any(not isinstance(value, StageEditSourceContinuity) for value in continuity):
            raise StageEditDecodeInputError(
                "source_continuity must contain StageEditSourceContinuity values"
            )
        object.__setattr__(self, "source_continuity", continuity)
        candidate_id = str(self.candidate_id).strip()
        if not candidate_id:
            candidate_id = f"candidate-{self.master_time_seconds:.6f}"
        object.__setattr__(self, "candidate_id", candidate_id)
        if self.dynamic_crop_v2 is not None and not isinstance(
            self.dynamic_crop_v2, DynamicCropHeadPredictionV2
        ):
            raise StageEditDecodeInputError(
                "dynamic_crop_v2 must be a DynamicCropHeadPredictionV2"
            )
        dynamic_v3 = tuple(self.dynamic_crop_v3_by_source)
        if any(
            value is not None and not isinstance(value, DynamicCropHeadPredictionV3)
            for value in dynamic_v3
        ):
            raise StageEditDecodeInputError(
                "dynamic_crop_v3_by_source must contain v3 predictions or None"
            )
        if self.dynamic_crop_v2 is not None and dynamic_v3:
            raise StageEditDecodeInputError(
                "one frame cannot contain both Dynamic Crop v2 and v3"
            )
        if dynamic_v3:
            if len(dynamic_v3) != len(self.source_logits):
                raise StageEditDecodeInputError(
                    "dynamic_crop_v3_by_source must align with canonical source slots"
                )
            if (
                len(self.available_source_mask) != len(self.source_logits)
                or len(self.hard_source_gate_mask) != len(self.source_logits)
            ):
                raise StageEditDecodeInputError(
                    "Dynamic Crop v3 source masks must align with canonical slots"
                )
            effective = self.effective_source_mask
            for source_index, (prediction, is_effective) in enumerate(
                zip(dynamic_v3, effective, strict=True)
            ):
                if (prediction is not None) != is_effective:
                    raise StageEditDecodeInputError(
                        "Dynamic Crop v3 prediction presence differs from source gates"
                    )
                if prediction is not None and prediction.source_index != source_index:
                    raise StageEditDecodeInputError(
                        "Dynamic Crop v3 prediction index differs from canonical slot"
                    )
        object.__setattr__(self, "dynamic_crop_v3_by_source", dynamic_v3)
        for name in ("section_progress", "phrase_progress"):
            value = getattr(self, name)
            if value is None:
                continue
            number = _finite(value, name)
            if not 0.0 <= number <= 1.0:
                raise StageEditDecodeInputError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, number)
        if self.section_role is not None:
            role = str(self.section_role).strip()
            allowed_roles = {item.value for item in SectionRole}
            if role not in allowed_roles:
                raise StageEditDecodeInputError("section_role is outside SectionRole")
            object.__setattr__(self, "section_role", role)

    @property
    def effective_source_mask(self) -> tuple[bool, ...]:
        return tuple(
            available and gate
            for available, gate in zip(
                self.available_source_mask,
                self.hard_source_gate_mask,
                strict=True,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return _public_scalar_pose_payload({
            "master_time_seconds": self.master_time_seconds,
            "cut_logit": self.cut_logit,
            "source_logits": list(self.source_logits),
            "duration_logits": list(self.duration_logits),
            "transition_logits": list(self.transition_logits),
            "crop": list(self.crop),
            "available_source_mask": list(self.available_source_mask),
            "hard_source_gate_mask": list(self.hard_source_gate_mask),
            "source_shot_sizes": list(self.source_shot_sizes),
            "source_continuity": [
                value.to_dict() for value in self.source_continuity
            ],
            "candidate_id": self.candidate_id,
            "dynamic_crop_v2": (
                self.dynamic_crop_v2.to_dict()
                if self.dynamic_crop_v2 is not None
                else None
            ),
            "dynamic_crop_v3_by_source": [
                value.to_dict() if value is not None else None
                for value in self.dynamic_crop_v3_by_source
            ],
            "section_progress": self.section_progress,
            "phrase_progress": self.phrase_progress,
            "section_role": self.section_role,
        }, context="StageEdit head frame")


@dataclass(frozen=True, slots=True)
class StageEditSectionConstraint:
    """Optional hard overrides for one acoustic section."""

    section_index: int
    minimum_shot_seconds: float | None = None
    maximum_shot_seconds: float | None = None
    maximum_switches_per_minute: float | None = None
    allowed_transition_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.section_index, bool)
            or not isinstance(self.section_index, int)
            or self.section_index < 0
        ):
            raise StageEditDecodeInputError("section_index must be non-negative")
        for name in ("minimum_shot_seconds", "maximum_shot_seconds"):
            value = getattr(self, name)
            if value is None:
                continue
            number = _finite(value, name)
            if number <= 0.0:
                raise StageEditDecodeInputError(f"{name} must be positive")
            object.__setattr__(self, name, number)
        if (
            self.minimum_shot_seconds is not None
            and self.maximum_shot_seconds is not None
            and self.maximum_shot_seconds < self.minimum_shot_seconds
        ):
            raise StageEditDecodeInputError("section shot duration bounds are inverted")
        if self.maximum_switches_per_minute is not None:
            rate = _finite(
                self.maximum_switches_per_minute,
                "maximum_switches_per_minute",
            )
            if rate <= 0.0:
                raise StageEditDecodeInputError(
                    "maximum_switches_per_minute must be positive"
                )
            object.__setattr__(self, "maximum_switches_per_minute", rate)
        labels = tuple(str(value).strip() for value in self.allowed_transition_labels)
        if any(not value for value in labels) or len(set(labels)) != len(labels):
            raise StageEditDecodeInputError(
                "allowed transition labels must be non-empty and unique"
            )
        object.__setattr__(self, "allowed_transition_labels", labels)


@dataclass(frozen=True, slots=True)
class StageEditSemiMarkovConfig:
    """Bounded search and editorial policy for the causal decoder."""

    beam_width: int = 16
    timeline_count: int = 8
    minimum_shot_seconds: float = 0.50
    maximum_shot_seconds: float = 3.00
    minimum_unique_sources: int = 3
    maximum_source_occupancy_fraction: float = 0.60
    enforce_aba_exclusion: bool = True
    allow_consecutive_same_source: bool = False
    recent_source_window: int = 4
    recent_shot_size_window: int = 4
    maximum_same_shot_size_run: int = 3
    maximum_global_switches_per_minute: float = 30.0
    section_cut_density_multiplier: float = 1.35
    cut_density_window_seconds: float = 30.0
    internal_cut_exclusion_margin_seconds: float = 0.0
    cut_score_weight: float = 1.0
    source_score_weight: float = 1.0
    duration_score_weight: float = 0.65
    transition_score_weight: float = 0.30
    transition_authority: str = LEARNED_TRANSITION_AUTHORITY
    crop_correction_weight: float = 0.20
    source_quality_weight: float = 0.20
    boundary_continuity_weight: float = 0.35
    cadence_score_weight: float = 0.15
    diversity_score_weight: float = 0.12
    occupancy_balance_weight: float = 0.08
    montage_grammar_score_weight: float = 0.24
    enforce_montage_grammar: bool = True
    highlight_minimum_duration_seconds: float = 5.0
    highlight_minimum_known_shot_fraction: float = 0.80
    # Exact chorus-like P10 from the immutable four-song / twenty-reference
    # IVE professional-montage profile.  The non-wide/burst/reset requirements
    # below are deliberately labelled as product editorial requirements rather
    # than being misrepresented as lower-quantile reference floors.
    highlight_minimum_closeup_fraction: float = 0.1696555074498807
    highlight_minimum_nonwide_fraction: float = 0.33771372775759073
    highlight_minimum_wide_involved_source_switch_fraction: float = (
        0.37333333333333335
    )
    highlight_preferred_wide_involved_source_switch_fraction: float = 0.80
    highlight_wide_involved_source_switch_reference_count: int = 17
    highlight_minimum_early_wide_anchor_fraction: float = 1.0 / 3.0
    highlight_minimum_phrase_entry_wide_anchor_fraction: float = (
        0.2785714285714286
    )
    # Recomputed from the immutable 20-reference IVE montage profile as
    # 60 * (chorus shot count - 1) / chorus role duration.  The median is an
    # explicit IVE product minimum (not mislabelled as a P10 reference floor),
    # while P75 is the soft target requested for a livelier highlight edit.
    highlight_product_minimum_source_switches_per_minute: float = (
        18.642084765823732
    )
    highlight_preferred_source_switches_per_minute: float = (
        23.505554096533405
    )
    highlight_cadence_reference_count: int = 20
    highlight_burst_maximum_shot_seconds: float = 1.50
    # The human non-wide multi-source burst median is 2.5 shots.  Requiring
    # two internal boundaries means a three-shot product burst, rather than
    # allowing one token two-shot exchange to make an otherwise flat chorus
    # pass.
    highlight_minimum_nonwide_burst_boundaries: int = 2
    formation_anchor_decision_window: int = 3
    # The reference profile only proves identity-free observed-person evidence;
    # it explicitly cannot establish a six-member formation threshold.  One
    # observed IVE-roster unit plus non-zero pose coverage distinguishes a
    # formation-evidenced wide from a generic/unknown wide without claiming
    # that it proves the complete group is visible.
    minimum_formation_anchor_person_count_normalized: float = 1.0 / 6.0
    # IVE-only soft product preference. Four or more independent person-box
    # observations make a wide readable as a group formation even when the
    # tiny bodies are too small for per-person upper-body pose. This remains
    # identity-free evidence and is not proof that all six members are shown.
    strong_formation_anchor_person_count_normalized: float = 4.0 / 6.0
    highlight_minimum_post_burst_formation_resets: int = 1
    # Appended for positional backwards compatibility. Exactly one boundary
    # continuity authority is active for the entire decode.
    continuity_authority: str = HANDCRAFTED_CONTINUITY_AUTHORITY
    continuity_authority_contract: str = CONTINUITY_AUTHORITY_CONTRACT
    # Optional expected values let a caller bind a preloaded production scorer.
    # The audit config always records the resolved non-optional values.
    continuity_scorer_contract: str | None = None
    continuity_scorer_fingerprint: str | None = None
    # Appended for positional backwards compatibility.  The local values
    # above are duration-weighted P10 floors applied to *every* strict
    # highlight.  These P25 requirements are pooled across all strict
    # highlights, so one unusually close section cannot rescue a flat one.
    highlight_pooled_product_minimum_closeup_fraction: float = (
        0.22343208166310066
    )
    highlight_pooled_product_minimum_nonwide_fraction: float = (
        0.48417342818793935
    )
    # Descriptive human P50 targets.  They affect only bounded final-timeline
    # ranking/evidence; neither is a hard gate and overshooting the target does
    # not keep accumulating reward.
    highlight_preferred_closeup_fraction: float = 0.3267829794497803
    highlight_preferred_nonwide_fraction: float = 0.6607923807246032
    # Human-reference P100 is a fail-closed upper limit.  P75 is a saturated
    # soft upper target: shorter wide runs receive no additional reward.
    highlight_maximum_consecutive_wide_run_seconds: float = 8.258529233180866
    highlight_preferred_maximum_consecutive_wide_run_seconds: float = (
        5.833280462440257
    )
    # Primitive-only optional identity avoids mutable config mappings and lets
    # the pipeline bind the exact ProfessionalMontagePolicy payload later.
    highlight_policy_profile_analysis_basis: str | None = None
    highlight_policy_profile_schema_version: int | None = None
    highlight_policy_profile_path: str | None = None
    highlight_policy_profile_file_sha256: str | None = None
    highlight_policy_profile_content_sha256: str | None = None
    highlight_policy_fingerprint_sha256: str | None = None
    highlight_motion_calibration: StageEditHighlightMotionCalibration | None = None
    # Profile v2 has no choreography-motion boundary distribution.  These
    # hooks therefore default inactive rather than inventing a quota.  When a
    # future profile binds them, a boundary counts only when motion amount,
    # the independently recomputed handcrafted continuity qualifier, and
    # (when observed) sharpness all pass. The primary boundary scorer may be
    # learned and is never compared with these handcrafted quantiles.
    highlight_motion_boundary_minimum_pose_motion: float | None = None
    highlight_motion_boundary_minimum_continuity_score: float | None = None
    highlight_motion_boundary_minimum_sharpness: float | None = None
    highlight_minimum_motion_closeup_switch_boundaries: int = 0
    # Optional same-time pose-phase gate.  It is inactive by default and may
    # only bind one of the immutable human-calibration quantiles.  Direction
    # and velocity are intentionally outside this scalar-cosine contract.
    highlight_motion_boundary_minimum_pose_phase_cosine: float | None = None
    # Optional source-exact temporal calibration.  No thresholds are supplied
    # by default; binding this typed evidence makes temporal phase/direction an
    # additional requirement for the existing bounded motion soft bonus.
    highlight_temporal_pose_motion_calibration: (
        StageEditTemporalPoseMotionCalibration | None
    ) = None
    # Six observed human close-only multi-source burst events provide
    # conditional event-shape P10s, while per-reference occurrence P10 and
    # median are both zero.  Production binds these as a saturated soft
    # preference only; absence can never eliminate a timeline.
    highlight_minimum_close_burst_shot_count: float | None = None
    highlight_minimum_close_burst_duration_seconds: float | None = None
    highlight_minimum_close_burst_source_switch_fraction: float | None = None
    highlight_close_burst_event_reference_count: int | None = None
    highlight_close_burst_policy_binding_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enforce_montage_grammar, bool):
            raise StageEditDecodeInputError(
                "enforce_montage_grammar must be boolean"
            )
        authority = str(self.continuity_authority).strip()
        if authority not in _CONTINUITY_AUTHORITIES:
            raise StageEditDecodeInputError(
                "continuity_authority must select handcrafted or production "
                "learned-checkpoint scoring"
            )
        object.__setattr__(self, "continuity_authority", authority)
        if self.continuity_authority_contract != CONTINUITY_AUTHORITY_CONTRACT:
            raise StageEditDecodeInputError(
                "continuity_authority_contract is unsupported"
            )
        expected_contract = self.continuity_scorer_contract
        if expected_contract is not None:
            expected_contract = str(expected_contract).strip()
            if not expected_contract:
                raise StageEditDecodeInputError(
                    "continuity_scorer_contract cannot be blank"
                )
            allowed_contract = (
                HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
                if authority == HANDCRAFTED_CONTINUITY_AUTHORITY
                else STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT
            )
            if expected_contract != allowed_contract:
                raise StageEditDecodeInputError(
                    "continuity scorer contract conflicts with its authority"
                )
            object.__setattr__(
                self, "continuity_scorer_contract", expected_contract
            )
        expected_fingerprint = self.continuity_scorer_fingerprint
        if expected_fingerprint is not None:
            if (
                not isinstance(expected_fingerprint, str)
                or len(expected_fingerprint) != 64
                or any(
                    value not in "0123456789abcdef"
                    for value in expected_fingerprint
                )
            ):
                raise StageEditDecodeInputError(
                    "continuity_scorer_fingerprint must be lowercase SHA-256"
                )
            if (
                authority == HANDCRAFTED_CONTINUITY_AUTHORITY
                and expected_fingerprint
                != HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
            ):
                raise StageEditDecodeInputError(
                    "handcrafted continuity scorer fingerprint mismatch"
                )
        for name in ("beam_width", "timeline_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 8 <= value <= 32:
                raise StageEditDecodeInputError(f"{name} must be an integer in 8...32")
        if self.timeline_count > self.beam_width:
            raise StageEditDecodeInputError("timeline_count cannot exceed beam_width")
        for name in (
            "recent_source_window",
            "recent_shot_size_window",
            "maximum_same_shot_size_run",
            "minimum_unique_sources",
            "highlight_minimum_nonwide_burst_boundaries",
            "formation_anchor_decision_window",
            "highlight_minimum_post_burst_formation_resets",
            "highlight_wide_involved_source_switch_reference_count",
            "highlight_cadence_reference_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise StageEditDecodeInputError(f"{name} must be a positive integer")
        if (
            isinstance(self.highlight_minimum_motion_closeup_switch_boundaries, bool)
            or not isinstance(
                self.highlight_minimum_motion_closeup_switch_boundaries, int
            )
            or self.highlight_minimum_motion_closeup_switch_boundaries < 0
        ):
            raise StageEditDecodeInputError(
                "highlight_minimum_motion_closeup_switch_boundaries must be a "
                "non-negative integer"
            )
        if self.enforce_aba_exclusion and self.recent_source_window < 2:
            raise StageEditDecodeInputError(
                "recent_source_window must be at least two for A-B-A exclusion"
            )
        if self.recent_shot_size_window < self.maximum_same_shot_size_run:
            raise StageEditDecodeInputError(
                "recent_shot_size_window must cover maximum_same_shot_size_run"
            )
        for name in (
            "minimum_shot_seconds",
            "maximum_shot_seconds",
            "maximum_source_occupancy_fraction",
            "maximum_global_switches_per_minute",
            "section_cut_density_multiplier",
            "cut_density_window_seconds",
            "highlight_minimum_duration_seconds",
            "highlight_burst_maximum_shot_seconds",
            "highlight_product_minimum_source_switches_per_minute",
            "highlight_preferred_source_switches_per_minute",
            "highlight_maximum_consecutive_wide_run_seconds",
            "highlight_preferred_maximum_consecutive_wide_run_seconds",
        ):
            number = _finite(getattr(self, name), name)
            if number <= 0.0:
                raise StageEditDecodeInputError(f"{name} must be positive")
            object.__setattr__(self, name, number)
        authority = StageEditTransitionAuthority(mode=self.transition_authority)
        if authority.renderer_owned and self.transition_score_weight != 0.0:
            raise StageEditDecodeInputError(
                "renderer-owned transition authority requires transition_score_weight=0"
            )
        if self.maximum_shot_seconds < self.minimum_shot_seconds:
            raise StageEditDecodeInputError("global shot duration bounds are inverted")
        if self.maximum_source_occupancy_fraction > 1.0:
            raise StageEditDecodeInputError(
                "maximum_source_occupancy_fraction cannot exceed one"
            )
        margin = _finite(
            self.internal_cut_exclusion_margin_seconds,
            "internal_cut_exclusion_margin_seconds",
        )
        if margin < 0.0:
            raise StageEditDecodeInputError(
                "internal_cut_exclusion_margin_seconds cannot be negative"
            )
        object.__setattr__(self, "internal_cut_exclusion_margin_seconds", margin)
        for name in (
            "cut_score_weight",
            "source_score_weight",
            "duration_score_weight",
            "transition_score_weight",
            "crop_correction_weight",
            "source_quality_weight",
            "boundary_continuity_weight",
            "cadence_score_weight",
            "diversity_score_weight",
            "occupancy_balance_weight",
            "montage_grammar_score_weight",
        ):
            number = _finite(getattr(self, name), name)
            if number < 0.0:
                raise StageEditDecodeInputError(f"{name} cannot be negative")
            object.__setattr__(self, name, number)
        for name in (
            "highlight_minimum_known_shot_fraction",
            "highlight_minimum_closeup_fraction",
            "highlight_minimum_nonwide_fraction",
            "highlight_minimum_wide_involved_source_switch_fraction",
            "highlight_preferred_wide_involved_source_switch_fraction",
            "highlight_minimum_early_wide_anchor_fraction",
            "highlight_minimum_phrase_entry_wide_anchor_fraction",
            "highlight_pooled_product_minimum_closeup_fraction",
            "highlight_pooled_product_minimum_nonwide_fraction",
            "highlight_preferred_closeup_fraction",
            "highlight_preferred_nonwide_fraction",
            "minimum_formation_anchor_person_count_normalized",
            "strong_formation_anchor_person_count_normalized",
        ):
            number = _finite(getattr(self, name), name)
            if not 0.0 <= number <= 1.0:
                raise StageEditDecodeInputError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, number)
        for name in (
            "highlight_motion_boundary_minimum_pose_motion",
            "highlight_motion_boundary_minimum_sharpness",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            number = _finite(value, name)
            if number < 0.0:
                raise StageEditDecodeInputError(f"{name} cannot be negative")
            object.__setattr__(self, name, number)
        continuity_minimum = self.highlight_motion_boundary_minimum_continuity_score
        if continuity_minimum is not None:
            number = _finite(
                continuity_minimum,
                "highlight_motion_boundary_minimum_continuity_score",
            )
            if not -1.0 <= number <= 0.0:
                raise StageEditDecodeInputError(
                    "highlight motion continuity minimum must lie in [-1, 0]"
                )
            object.__setattr__(
                self,
                "highlight_motion_boundary_minimum_continuity_score",
                number,
            )
        pose_phase_minimum = (
            self.highlight_motion_boundary_minimum_pose_phase_cosine
        )
        if pose_phase_minimum is not None:
            number = _finite(
                pose_phase_minimum,
                "highlight_motion_boundary_minimum_pose_phase_cosine",
            )
            if not -1.0 <= number <= 1.0:
                raise StageEditDecodeInputError(
                    "highlight motion pose-phase cosine minimum must lie in "
                    "[-1, 1]"
                )
            object.__setattr__(
                self,
                "highlight_motion_boundary_minimum_pose_phase_cosine",
                number,
            )
        calibration = self.highlight_motion_calibration
        if calibration is not None:
            if isinstance(calibration, Mapping):
                try:
                    calibration = StageEditHighlightMotionCalibration(
                        **dict(calibration)
                    )
                except (TypeError, ValueError) as exc:
                    raise StageEditDecodeInputError(
                        "highlight_motion_calibration mapping cannot be "
                        "round-tripped"
                    ) from exc
                object.__setattr__(
                    self, "highlight_motion_calibration", calibration
                )
            if not isinstance(calibration, StageEditHighlightMotionCalibration):
                raise StageEditDecodeInputError(
                    "highlight_motion_calibration has an invalid type"
                )
            pose_threshold = self.highlight_motion_boundary_minimum_pose_motion
            continuity_threshold = (
                self.highlight_motion_boundary_minimum_continuity_score
            )
            if (pose_threshold is None) != (continuity_threshold is None):
                raise StageEditDecodeInputError(
                    "profile-calibrated motion ranking requires paired pose and "
                    "continuity thresholds"
                )
            if pose_threshold is not None and continuity_threshold is not None:
                if not any(
                    math.isclose(
                        pose_threshold,
                        value,
                        abs_tol=_TIME_EPSILON,
                    )
                    for value in calibration.pose_motion_minimum_quantiles
                ):
                    raise StageEditDecodeInputError(
                        "motion pose threshold is not a calibrated quantile"
                    )
                if not any(
                    math.isclose(
                        continuity_threshold,
                        value,
                        abs_tol=_TIME_EPSILON,
                    )
                    for value in calibration.continuity_score_quantiles
                ):
                    raise StageEditDecodeInputError(
                        "motion continuity threshold is not a calibrated quantile"
                    )
                # The calibration quantiles were measured with the fixed
                # handcrafted scorer.  They qualify highlight motion
                # independently of the primary boundary-continuity authority;
                # comparing a learned primary score to these quantiles would
                # mix two incompatible score distributions.
                if not (
                    calibration.continuity_authority
                    == HANDCRAFTED_CONTINUITY_AUTHORITY
                    and calibration.continuity_scorer_contract
                    == HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
                ):
                    raise StageEditDecodeInputError(
                        "motion calibration must bind the independent "
                        "handcrafted continuity qualifier"
                    )
            pose_phase_threshold = (
                self.highlight_motion_boundary_minimum_pose_phase_cosine
            )
            if pose_phase_threshold is not None:
                if pose_threshold is None or continuity_threshold is None:
                    raise StageEditDecodeInputError(
                        "pose-phase threshold requires paired pose-motion and "
                        "continuity thresholds"
                    )
                if not any(
                    math.isclose(
                        pose_phase_threshold,
                        value,
                        abs_tol=_TIME_EPSILON,
                    )
                    for value in calibration.pose_embedding_cosine_quantiles
                ):
                    raise StageEditDecodeInputError(
                        "motion pose-phase threshold is not a calibrated quantile"
                    )
            sharpness_threshold = (
                self.highlight_motion_boundary_minimum_sharpness
            )
            if sharpness_threshold is not None:
                if calibration.sharpness_minimum_quantiles is None:
                    raise StageEditDecodeInputError(
                        "motion sharpness threshold has no calibrated observations"
                    )
                if not any(
                    math.isclose(
                        sharpness_threshold,
                        value,
                        abs_tol=_TIME_EPSILON,
                    )
                    for value in calibration.sharpness_minimum_quantiles
                ):
                    raise StageEditDecodeInputError(
                        "motion sharpness threshold is not a calibrated quantile"
                    )
        elif self.highlight_motion_boundary_minimum_pose_phase_cosine is not None:
            raise StageEditDecodeInputError(
                "pose-phase threshold requires an immutable motion calibration"
            )
        temporal_calibration = self.highlight_temporal_pose_motion_calibration
        if temporal_calibration is not None:
            if not TEMPORAL_POSE_MOTION_CALIBRATION_ACTIVATION_AVAILABLE:
                raise StageEditDecodeInputError(
                    "temporal pose-motion calibration activation is unavailable "
                    "until an immutable evidence loader/registry verifies it"
                )
            if isinstance(temporal_calibration, Mapping):
                try:
                    temporal_calibration = StageEditTemporalPoseMotionCalibration(
                        **dict(temporal_calibration)
                    )
                except (TypeError, ValueError) as exc:
                    raise StageEditDecodeInputError(
                        "temporal pose-motion calibration mapping cannot be "
                        "round-tripped"
                    ) from exc
                object.__setattr__(
                    self,
                    "highlight_temporal_pose_motion_calibration",
                    temporal_calibration,
                )
            if not isinstance(
                temporal_calibration, StageEditTemporalPoseMotionCalibration
            ):
                raise StageEditDecodeInputError(
                    "highlight temporal pose-motion calibration has invalid type"
                )
            if (
                calibration is None
                or self.highlight_motion_boundary_minimum_pose_motion is None
                or self.highlight_motion_boundary_minimum_continuity_score is None
            ):
                raise StageEditDecodeInputError(
                    "temporal pose-motion qualification requires the existing "
                    "calibrated motion and continuity soft preference"
                )
        if (
            self.highlight_minimum_motion_closeup_switch_boundaries > 0
            and (
                self.highlight_motion_boundary_minimum_pose_motion is None
                or self.highlight_motion_boundary_minimum_continuity_score is None
            )
        ):
            raise StageEditDecodeInputError(
                "motion close-up switch gate requires both pose-motion amount "
                "and continuity-compatibility thresholds"
            )
        if (
            self.strong_formation_anchor_person_count_normalized
            < self.minimum_formation_anchor_person_count_normalized
        ):
            raise StageEditDecodeInputError(
                "strong formation threshold cannot be below the weak threshold"
            )
        if (
            self.highlight_minimum_closeup_fraction
            > self.highlight_minimum_nonwide_fraction
        ):
            raise StageEditDecodeInputError(
                "highlight close-up floor cannot exceed the non-wide floor"
            )
        if (
            self.highlight_pooled_product_minimum_closeup_fraction
            + _TIME_EPSILON
            < self.highlight_minimum_closeup_fraction
            or self.highlight_pooled_product_minimum_nonwide_fraction
            + _TIME_EPSILON
            < self.highlight_minimum_nonwide_fraction
        ):
            raise StageEditDecodeInputError(
                "highlight pooled P25 product minimum cannot be below its "
                "local P10 floor"
            )
        if (
            self.highlight_preferred_closeup_fraction + _TIME_EPSILON
            < self.highlight_pooled_product_minimum_closeup_fraction
            or self.highlight_preferred_nonwide_fraction + _TIME_EPSILON
            < self.highlight_pooled_product_minimum_nonwide_fraction
        ):
            raise StageEditDecodeInputError(
                "highlight P50 preferred target cannot be below its P25 "
                "product minimum"
            )
        if (
            self.highlight_preferred_maximum_consecutive_wide_run_seconds
            > self.highlight_maximum_consecutive_wide_run_seconds
            + _TIME_EPSILON
        ):
            raise StageEditDecodeInputError(
                "highlight P75 preferred wide-run maximum cannot exceed its "
                "hard maximum"
            )
        provenance_values = (
            self.highlight_policy_profile_analysis_basis,
            self.highlight_policy_profile_schema_version,
            self.highlight_policy_profile_path,
            self.highlight_policy_profile_file_sha256,
            self.highlight_policy_profile_content_sha256,
            self.highlight_policy_fingerprint_sha256,
        )
        if any(value is not None for value in provenance_values):
            if any(value is None for value in provenance_values):
                raise StageEditDecodeInputError(
                    "highlight policy provenance identity must be complete"
                )
            assert self.highlight_policy_profile_schema_version is not None
            if (
                isinstance(self.highlight_policy_profile_schema_version, bool)
                or not isinstance(self.highlight_policy_profile_schema_version, int)
                or self.highlight_policy_profile_schema_version < 1
            ):
                raise StageEditDecodeInputError(
                    "highlight policy profile schema version must be positive"
                )
            for name in (
                "highlight_policy_profile_analysis_basis",
                "highlight_policy_profile_path",
            ):
                value = getattr(self, name)
                if not isinstance(value, str) or not value.strip():
                    raise StageEditDecodeInputError(f"{name} cannot be blank")
            if not Path(str(self.highlight_policy_profile_path)).is_absolute():
                raise StageEditDecodeInputError(
                    "highlight policy profile path must be absolute"
                )
            for name in (
                "highlight_policy_profile_file_sha256",
                "highlight_policy_profile_content_sha256",
                "highlight_policy_fingerprint_sha256",
            ):
                value = getattr(self, name)
                if not (
                    isinstance(value, str)
                    and len(value) == 64
                    and all(character in "0123456789abcdef" for character in value)
                ):
                    raise StageEditDecodeInputError(
                        f"{name} must be lowercase SHA-256"
                    )
        identity_bound = all(value is not None for value in provenance_values)
        close_burst_values = (
            self.highlight_minimum_close_burst_shot_count,
            self.highlight_minimum_close_burst_duration_seconds,
            self.highlight_minimum_close_burst_source_switch_fraction,
            self.highlight_close_burst_event_reference_count,
            self.highlight_close_burst_policy_binding_sha256,
        )
        if identity_bound and any(value is None for value in close_burst_values):
            raise StageEditDecodeInputError(
                "hash-bound highlight policy is missing its close-burst soft "
                "preference binding"
            )
        if not identity_bound and any(value is not None for value in close_burst_values):
            raise StageEditDecodeInputError(
                "close-burst soft preference requires complete hash-bound "
                "highlight policy identity"
            )
        if identity_bound:
            assert self.highlight_minimum_close_burst_shot_count is not None
            assert self.highlight_minimum_close_burst_duration_seconds is not None
            assert (
                self.highlight_minimum_close_burst_source_switch_fraction
                is not None
            )
            assert self.highlight_close_burst_event_reference_count is not None
            assert self.highlight_close_burst_policy_binding_sha256 is not None
            assert self.highlight_policy_fingerprint_sha256 is not None
            try:
                expected_binding = close_burst_policy_binding_sha256(
                    profile_identity={
                        "analysis_basis": (
                            self.highlight_policy_profile_analysis_basis
                        ),
                        "schema_version": (
                            self.highlight_policy_profile_schema_version
                        ),
                        "path": self.highlight_policy_profile_path,
                        "file_sha256": (
                            self.highlight_policy_profile_file_sha256
                        ),
                        "content_sha256": (
                            self.highlight_policy_profile_content_sha256
                        ),
                    },
                    policy_fingerprint_sha256=(
                        self.highlight_policy_fingerprint_sha256
                    ),
                    minimum_shot_count=(
                        self.highlight_minimum_close_burst_shot_count
                    ),
                    minimum_duration_seconds=(
                        self.highlight_minimum_close_burst_duration_seconds
                    ),
                    minimum_source_switch_fraction=(
                        self.highlight_minimum_close_burst_source_switch_fraction
                    ),
                    event_reference_count=(
                        self.highlight_close_burst_event_reference_count
                    ),
                )
            except ValueError as exc:
                raise StageEditDecodeInputError(
                    f"close-burst soft preference binding is malformed: {exc}"
                ) from exc
            if self.highlight_close_burst_policy_binding_sha256 != expected_binding:
                raise StageEditDecodeInputError(
                    "close-burst soft preference binding SHA-256 mismatch"
                )
        if (
            self.highlight_preferred_wide_involved_source_switch_fraction
            + _TIME_EPSILON
            < self.highlight_minimum_wide_involved_source_switch_fraction
        ):
            raise StageEditDecodeInputError(
                "highlight preferred wide-switch target cannot be below its "
                "reference floor"
            )
        if (
            self.highlight_preferred_source_switches_per_minute
            + _TIME_EPSILON
            < self.highlight_product_minimum_source_switches_per_minute
        ):
            raise StageEditDecodeInputError(
                "highlight preferred cadence cannot be below its product minimum"
            )
        if (
            self.enforce_montage_grammar
            and self.highlight_preferred_source_switches_per_minute
            > self.maximum_global_switches_per_minute + _TIME_EPSILON
        ):
            raise StageEditDecodeInputError(
                "highlight preferred cadence exceeds the global safety maximum"
            )


@dataclass(frozen=True, slots=True)
class StageEditScoreComponent:
    name: str
    raw_value: float
    weight: float
    contribution: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("score component name cannot be blank")
        for value in (self.raw_value, self.weight, self.contribution):
            if not math.isfinite(value):
                raise ValueError("score components must be finite")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StageEditBeamDecision:
    shot_index: int
    start_candidate_index: int
    end_candidate_index: int
    master_start_seconds: float
    master_end_seconds: float
    source_index: int
    source_id: str
    shot_size: str
    detected_person_count_normalized: float | None
    formation_person_count_normalized: float | None
    formation_pose_coverage: float | None
    duration_bucket: int
    transition_label: str
    crop_requested: tuple[float, float, float, float]
    crop_applied: tuple[float, float, float, float]
    crop_was_corrected: bool
    decision_confidence: float
    score_components: tuple[StageEditScoreComponent, ...]
    recent_sources_after: tuple[int, ...]
    recent_shot_sizes_after: tuple[str, ...]
    recent_shot_durations_after: tuple[float, ...]
    recent_cut_density_per_minute_after: float
    source_occupancy_after_seconds: tuple[float, ...]
    section_switch_counts_after: tuple[int, ...]
    derivative_family_id: int = 0
    derivative_family_occupancy_after_seconds: tuple[float, ...] = ()
    editorial_transform: EditorialTransformTrack | None = None
    dynamic_crop_decode: DynamicCropTrackDecodeV2 | None = None
    crop_contract: str = "legacy_static_crop_head_v1"
    section_progress: float | None = None
    phrase_progress: float | None = None
    section_role: str | None = None
    # Separate from the preserved v2 field so downstream users that type-check
    # ``dynamic_crop_decode`` continue to receive only the v2 decode contract.
    dynamic_crop_decode_v3: DynamicCropTrackDecodeV3 | None = None
    boundary_pose_motion_minimum: float | None = None
    boundary_pose_motion_delta: float | None = None
    boundary_camera_motion_delta: float | None = None
    boundary_minimum_sharpness: float | None = None
    boundary_continuity_score: float | None = None
    boundary_motion_amount_plus_continuity_evidence_complete: bool = False
    boundary_motion_amount_plus_continuity_supported: bool = False
    boundary_pose_phase_cosine: float | None = None
    boundary_pose_phase_evidence_observed: bool = False
    boundary_pose_phase_threshold_bound: bool = False
    boundary_pose_phase_match_proven: bool = False
    boundary_pose_phase_feature_dimension: int | None = None
    boundary_pose_phase_outgoing_observed: bool = False
    boundary_pose_phase_incoming_observed: bool = False
    boundary_pose_phase_trusted_producer_contract: str | None = None
    boundary_pose_phase_outgoing_content_sha256: str | None = None
    boundary_pose_phase_incoming_content_sha256: str | None = None
    boundary_pose_phase_pair_content_sha256: str | None = None
    boundary_temporal_pose_motion_thresholds_bound: bool = False
    boundary_temporal_pose_motion_input_valid: bool = False
    boundary_temporal_outgoing_motion_magnitude: float | None = None
    boundary_temporal_incoming_motion_magnitude: float | None = None
    boundary_temporal_endpoint_pose_cosine: float | None = None
    boundary_temporal_velocity_direction_cosine: float | None = None
    boundary_temporal_phase_compatibility: float | None = None
    boundary_temporal_confidence: float = 0.0
    boundary_temporal_classification: str | None = None
    boundary_temporal_static_pose_match: bool = False
    boundary_temporal_dynamic_pose_phase_match: bool = False
    boundary_temporal_failure_reasons: tuple[str, ...] = ()
    boundary_temporal_sampling_contract_match: bool = False
    boundary_temporal_outgoing_sampling_contract: str | None = None
    boundary_temporal_incoming_sampling_contract: str | None = None
    boundary_temporal_feature_dimension: int | None = None
    boundary_temporal_outgoing_sidecar_observed: bool = False
    boundary_temporal_incoming_sidecar_observed: bool = False
    boundary_temporal_outgoing_trusted_producer_contract: str | None = None
    boundary_temporal_incoming_trusted_producer_contract: str | None = None
    boundary_temporal_outgoing_content_sha256: str | None = None
    boundary_temporal_incoming_content_sha256: str | None = None
    boundary_temporal_pair_content_sha256: str | None = None
    boundary_pose_direction_or_velocity_proven: bool = False
    # Appended so positional construction of the pre-v11 decision remains
    # compatible. The primary boundary score above is preserved separately.
    boundary_handcrafted_motion_continuity_qualifier_score: float | None = None

    @property
    def duration_seconds(self) -> float:
        return self.master_end_seconds - self.master_start_seconds

    @property
    def score(self) -> float:
        return sum(item.contribution for item in self.score_components)

    def signature(self) -> tuple[Any, ...]:
        transform_signature = (
            ()
            if self.editorial_transform is None
            else tuple(
                (
                    round(item.time_seconds, 9),
                    round(item.zoom, 9),
                    round(item.center_x, 9),
                    round(item.center_y, 9),
                    round(item.roll_degrees, 9),
                )
                for item in self.editorial_transform.keyframes
            )
        )
        return (
            self.end_candidate_index,
            # Alternate files from one verified camera family must not fill
            # the retained candidate set as supposedly distinct edits.
            self.derivative_family_id,
            self.duration_bucket,
            self.transition_label,
            tuple(round(value, 9) for value in self.crop_applied),
            self.crop_contract,
            transform_signature,
        )

    def to_dict(self) -> dict[str, Any]:
        return _public_scalar_pose_payload({
            "shot_index": self.shot_index,
            "start_candidate_index": self.start_candidate_index,
            "end_candidate_index": self.end_candidate_index,
            "master_start_seconds": self.master_start_seconds,
            "master_end_seconds": self.master_end_seconds,
            "duration_seconds": self.duration_seconds,
            "source_index": self.source_index,
            "source_id": self.source_id,
            "derivative_family_id": self.derivative_family_id,
            "shot_size": self.shot_size,
            "detected_person_count_normalized": (
                self.detected_person_count_normalized
            ),
            "formation_person_count_normalized": (
                self.formation_person_count_normalized
            ),
            "formation_pose_coverage": self.formation_pose_coverage,
            "boundary_motion_amount_plus_continuity": {
                "pose_motion_minimum": self.boundary_pose_motion_minimum,
                "pose_motion_delta": self.boundary_pose_motion_delta,
                "camera_motion_delta": self.boundary_camera_motion_delta,
                "minimum_sharpness": self.boundary_minimum_sharpness,
                "continuity_score": self.boundary_continuity_score,
                "primary_boundary_continuity_score": (
                    self.boundary_continuity_score
                ),
                "handcrafted_continuity_qualifier_score": (
                    self.boundary_handcrafted_motion_continuity_qualifier_score
                ),
                "motion_continuity_qualifier_provenance": {
                    "role": (
                        "independent_handcrafted_motion_eligibility_qualifier"
                    ),
                    "authority": HANDCRAFTED_CONTINUITY_AUTHORITY,
                    "scorer_contract": (
                        HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
                    ),
                    "scorer_fingerprint": (
                        HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
                    ),
                    "primary_boundary_score_reused": False,
                },
                "pose_phase_cosine": self.boundary_pose_phase_cosine,
                "pose_phase_evidence_observed": (
                    self.boundary_pose_phase_evidence_observed
                ),
                "pose_phase_threshold_bound": (
                    self.boundary_pose_phase_threshold_bound
                ),
                "evidence_complete": (
                    self.boundary_motion_amount_plus_continuity_evidence_complete
                ),
                "accent_supported": (
                    self.boundary_motion_amount_plus_continuity_supported
                ),
                "high_motion_alone_is_sufficient": False,
                "high_pose_phase_alone_is_sufficient": False,
                "pose_phase_match_proven": (
                    self.boundary_pose_phase_match_proven
                ),
                "same_time_pose_evidence": {
                    "contract": SAME_TIME_POSE_SCALAR_EVIDENCE_CONTRACT,
                    "feature_dimension": (
                        self.boundary_pose_phase_feature_dimension
                    ),
                    "outgoing_observed": (
                        self.boundary_pose_phase_outgoing_observed
                    ),
                    "incoming_observed": (
                        self.boundary_pose_phase_incoming_observed
                    ),
                    "trusted_producer_contract": (
                        self.boundary_pose_phase_trusted_producer_contract
                    ),
                    "outgoing_content_sha256": (
                        self.boundary_pose_phase_outgoing_content_sha256
                    ),
                    "incoming_content_sha256": (
                        self.boundary_pose_phase_incoming_content_sha256
                    ),
                    "pair_content_sha256": (
                        self.boundary_pose_phase_pair_content_sha256
                    ),
                },
                "temporal_pose_motion": {
                    "thresholds_bound": (
                        self.boundary_temporal_pose_motion_thresholds_bound
                    ),
                    "input_valid": (
                        self.boundary_temporal_pose_motion_input_valid
                    ),
                    "outgoing_boundary_motion_magnitude": (
                        self.boundary_temporal_outgoing_motion_magnitude
                    ),
                    "incoming_boundary_motion_magnitude": (
                        self.boundary_temporal_incoming_motion_magnitude
                    ),
                    "endpoint_pose_cosine": (
                        self.boundary_temporal_endpoint_pose_cosine
                    ),
                    "velocity_direction_cosine": (
                        self.boundary_temporal_velocity_direction_cosine
                    ),
                    "phase_compatibility": (
                        self.boundary_temporal_phase_compatibility
                    ),
                    "confidence": self.boundary_temporal_confidence,
                    "classification": self.boundary_temporal_classification,
                    "static_pose_match": (
                        self.boundary_temporal_static_pose_match
                    ),
                    "dynamic_pose_phase_match": (
                        self.boundary_temporal_dynamic_pose_phase_match
                    ),
                    "failure_reasons": list(
                        self.boundary_temporal_failure_reasons
                    ),
                    "sampling_contract_match": (
                        self.boundary_temporal_sampling_contract_match
                    ),
                    "outgoing_sampling_contract": (
                        self.boundary_temporal_outgoing_sampling_contract
                    ),
                    "incoming_sampling_contract": (
                        self.boundary_temporal_incoming_sampling_contract
                    ),
                    "feature_dimension": (
                        self.boundary_temporal_feature_dimension
                    ),
                    "outgoing_sidecar_observed": (
                        self.boundary_temporal_outgoing_sidecar_observed
                    ),
                    "incoming_sidecar_observed": (
                        self.boundary_temporal_incoming_sidecar_observed
                    ),
                    "outgoing_trusted_producer_contract": (
                        self.boundary_temporal_outgoing_trusted_producer_contract
                    ),
                    "incoming_trusted_producer_contract": (
                        self.boundary_temporal_incoming_trusted_producer_contract
                    ),
                    "outgoing_content_sha256": (
                        self.boundary_temporal_outgoing_content_sha256
                    ),
                    "incoming_content_sha256": (
                        self.boundary_temporal_incoming_content_sha256
                    ),
                    "pair_content_sha256": (
                        self.boundary_temporal_pair_content_sha256
                    ),
                },
                "pose_direction_or_velocity_proven": (
                    self.boundary_pose_direction_or_velocity_proven
                ),
            },
            "section_progress": self.section_progress,
            "phrase_progress": self.phrase_progress,
            "section_role": self.section_role,
            "duration_bucket": self.duration_bucket,
            "transition_label": self.transition_label,
            "render_transition_kind": "hard_cut",
            "crop_requested": list(self.crop_requested),
            "crop_applied": list(self.crop_applied),
            "crop_was_corrected": self.crop_was_corrected,
            "crop_contract": self.crop_contract,
            "editorial_transform": (
                None
                if self.editorial_transform is None
                else {
                    "reason": self.editorial_transform.reason,
                    "keyframes": [
                        {
                            "time_seconds": item.time_seconds,
                            "zoom": item.zoom,
                            "center_x": item.center_x,
                            "center_y": item.center_y,
                            "roll_degrees": item.roll_degrees,
                        }
                        for item in self.editorial_transform.keyframes
                    ],
                }
            ),
            "dynamic_crop_v2": (
                None
                if self.dynamic_crop_decode is None
                else {
                    "adapter_contract": self.dynamic_crop_decode.adapter_contract,
                    "eligible": self.dynamic_crop_decode.eligible,
                    "loss_masked": self.dynamic_crop_decode.loss_masked,
                    "requested_keyframe_count": (
                        self.dynamic_crop_decode.requested_keyframe_count
                    ),
                    "applied_keyframe_count": (
                        self.dynamic_crop_decode.applied_keyframe_count
                    ),
                    "minimum_safe_zoom_projection_count": (
                        self.dynamic_crop_decode.minimum_safe_zoom_projection_count
                    ),
                    "rejection_reasons": list(
                        self.dynamic_crop_decode.rejection_reasons
                    ),
                }
            ),
            "dynamic_crop_v3": (
                None
                if self.dynamic_crop_decode_v3 is None
                else {
                    "adapter_contract": self.dynamic_crop_decode_v3.adapter_contract,
                    "source_id": self.dynamic_crop_decode_v3.source_id,
                    "source_index": self.dynamic_crop_decode_v3.source_index,
                    "eligible": self.dynamic_crop_decode_v3.eligible,
                    "loss_masked": self.dynamic_crop_decode_v3.loss_masked,
                    "predicted_keyframe_count": (
                        self.dynamic_crop_decode_v3.predicted_keyframe_count
                    ),
                    "applied_keyframe_count": (
                        self.dynamic_crop_decode_v3.applied_keyframe_count
                    ),
                    "minimum_safe_zoom_projection_count": (
                        self.dynamic_crop_decode_v3.minimum_safe_zoom_projection_count
                    ),
                    "rejection_reasons": list(
                        self.dynamic_crop_decode_v3.rejection_reasons
                    ),
                }
            ),
            "decision_confidence": self.decision_confidence,
            "score": self.score,
            "score_components": [item.to_dict() for item in self.score_components],
            "state_after": {
                "current_time_seconds": self.master_end_seconds,
                "current_source": self.source_index,
                "recent_sources": list(self.recent_sources_after),
                "recent_shot_sizes": list(self.recent_shot_sizes_after),
                "recent_shot_durations": list(self.recent_shot_durations_after),
                "recent_cut_density_per_minute": (
                    self.recent_cut_density_per_minute_after
                ),
                "source_occupancy_seconds": list(
                    self.source_occupancy_after_seconds
                ),
                "derivative_family_occupancy_seconds": list(
                    self.derivative_family_occupancy_after_seconds
                ),
                "section_switch_counts": list(self.section_switch_counts_after),
            },
        }, context="StageEdit beam decision")


@dataclass(frozen=True, slots=True)
class StageEditBeamTimeline:
    rank: int
    score: float
    score_components: Mapping[str, float]
    decisions: tuple[StageEditBeamDecision, ...]
    edit: StageEdit
    source_occupancy_seconds: tuple[float, ...]
    recent_cut_density_per_minute: float
    montage_grammar: Mapping[str, Any] = field(default_factory=dict)
    derivative_family_occupancy_seconds: tuple[float, ...] = ()
    boundary_continuity_score_provenance: Mapping[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return _public_scalar_pose_payload({
            "rank": self.rank,
            "score": self.score,
            "score_components": dict(self.score_components),
            "source_occupancy_seconds": list(self.source_occupancy_seconds),
            "derivative_family_occupancy_seconds": list(
                self.derivative_family_occupancy_seconds
            ),
            "recent_cut_density_per_minute": self.recent_cut_density_per_minute,
            "montage_grammar": dict(self.montage_grammar),
            "score_provenance": {
                "boundary_continuity": dict(
                    self.boundary_continuity_score_provenance
                )
            },
            "decisions": [item.to_dict() for item in self.decisions],
            "stage_edit": self.edit.to_dict(),
        }, context="StageEdit beam timeline")


@dataclass(frozen=True, slots=True)
class StageEditDecodeAudit:
    schema_version: int
    decoder: str
    input_sha256: str
    readiness: Mapping[str, Any]
    config: Mapping[str, Any]
    input_summary: Mapping[str, Any]
    endpoint_audit: tuple[Mapping[str, Any], ...]
    constraint_rejection_counts: Mapping[str, int]
    transition_hypothesis_count: int
    accepted_hypothesis_count: int
    beam_pruned_state_count: int
    complete_state_count: int
    duplicate_completion_count: int
    requested_timeline_count: int
    returned_timeline_count: int
    request_fulfilled: bool
    score_provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _public_scalar_pose_payload({
            "schema_version": self.schema_version,
            "decoder": self.decoder,
            "input_sha256": self.input_sha256,
            "readiness": dict(self.readiness),
            "config": dict(self.config),
            "input_summary": dict(self.input_summary),
            "endpoint_audit": [dict(item) for item in self.endpoint_audit],
            "constraint_rejection_counts": dict(self.constraint_rejection_counts),
            "transition_hypothesis_count": self.transition_hypothesis_count,
            "accepted_hypothesis_count": self.accepted_hypothesis_count,
            "beam_pruned_state_count": self.beam_pruned_state_count,
            "complete_state_count": self.complete_state_count,
            "duplicate_completion_count": self.duplicate_completion_count,
            "requested_timeline_count": self.requested_timeline_count,
            "returned_timeline_count": self.returned_timeline_count,
            "request_fulfilled": self.request_fulfilled,
            "score_provenance": dict(self.score_provenance),
        }, context="StageEdit decode audit")


@dataclass(frozen=True, slots=True)
class StageEditSemiMarkovResult:
    timelines: tuple[StageEditBeamTimeline, ...]
    audit: StageEditDecodeAudit

    @property
    def edits(self) -> tuple[StageEdit, ...]:
        return tuple(item.edit for item in self.timelines)

    def to_dict(self) -> dict[str, Any]:
        return _public_scalar_pose_payload({
            "timelines": [item.to_dict() for item in self.timelines],
            "audit": self.audit.to_dict(),
        }, context="StageEdit semi-Markov result")


@dataclass(frozen=True, slots=True)
class _ResolvedSectionConstraint:
    minimum_shot_seconds: float
    maximum_shot_seconds: float
    maximum_switches_per_minute: float
    allowed_transition_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _BeamState:
    current_index: int
    current_source: int | None
    current_section_index: int | None
    section_local_source_switch_boundaries: int
    section_local_wide_involved_source_switch_boundaries: int
    recent_sources: tuple[int, ...]
    recent_shot_sizes: tuple[str, ...]
    recent_shot_durations: tuple[float, ...]
    recent_cut_times: tuple[float, ...]
    source_occupancy_seconds: tuple[float, ...]
    section_switch_counts: tuple[int, ...]
    decisions: tuple[StageEditBeamDecision, ...]
    score: float
    component_totals: tuple[tuple[str, float], ...]

    def signature(self) -> tuple[Any, ...]:
        return tuple(item.signature() for item in self.decisions)


def _log_sigmoid(value: float) -> float:
    if value >= 0.0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def _log_softmax(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        raise StageEditDecodeInputError("softmax logits cannot be empty")
    maximum = max(values)
    total = sum(math.exp(value - maximum) for value in values)
    if not math.isfinite(total) or total <= 0.0:
        raise StageEditDecodeInputError("softmax normalization is not finite")
    normalizer = maximum + math.log(total)
    result = tuple(value - normalizer for value in values)
    if any(not math.isfinite(value) for value in result):
        raise StageEditDecodeInputError("softmax produced a non-finite score")
    return result


def _masked_log_softmax(
    values: Sequence[float],
    mask: Sequence[bool],
) -> tuple[float, ...]:
    if len(values) != len(mask):
        raise StageEditDecodeInputError("masked softmax values/mask length mismatch")
    selected_indices = tuple(index for index, allowed in enumerate(mask) if allowed)
    if not selected_indices:
        raise StageEditDecodeInputError("masked softmax has no allowed value")
    selected = _log_softmax(tuple(values[index] for index in selected_indices))
    # Rejected entries are deliberately finite because the decoder's
    # no-nonfinite contract covers internal audit state too.  They are never
    # read as candidate scores.
    result = [0.0] * len(values)
    for index, score in zip(selected_indices, selected, strict=True):
        result[index] = score
    return tuple(result)


def _component(name: str, raw: float, weight: float) -> StageEditScoreComponent:
    contribution = raw * weight
    if not math.isfinite(contribution):
        raise StageEditDecodeInputError(f"{name} score is not finite")
    return StageEditScoreComponent(name, raw, weight, contribution)


def _add_component_totals(
    current: tuple[tuple[str, float], ...],
    additions: Sequence[StageEditScoreComponent],
) -> tuple[tuple[str, float], ...]:
    totals = dict(current)
    for item in additions:
        totals[item.name] = totals.get(item.name, 0.0) + item.contribution
    if any(not math.isfinite(value) for value in totals.values()):
        raise StageEditDecodeInputError("cumulative beam score is not finite")
    return tuple(sorted(totals.items()))


def _state_sort_key(state: _BeamState) -> tuple[Any, ...]:
    return (-state.score, state.signature())


def _state_structure_signature(state: _BeamState) -> tuple[tuple[int, int], ...]:
    """The timing/source skeleton, excluding cosmetic transition variants."""

    return tuple(
        (item.end_candidate_index, item.derivative_family_id)
        for item in state.decisions
    )


def _prune_states(
    states: Sequence[_BeamState],
    width: int,
) -> tuple[list[_BeamState], int, int]:
    """Keep structural edit diversity before alternate transition labels.

    A plain top-k can spend the entire beam on the four transition labels for
    one timing/source skeleton.  That destroys the variable-duration part of
    the search and can make a later density constraint appear impossible.
    The first pass retains the best state for each structural skeleton; the
    second pass uses remaining capacity for genuine transition alternatives.
    """

    ordered = sorted(states, key=_state_sort_key)
    unique_exact: list[_BeamState] = []
    exact_signatures: set[tuple[Any, ...]] = set()
    duplicate_count = 0
    for state in ordered:
        signature = state.signature()
        if signature in exact_signatures:
            duplicate_count += 1
            continue
        exact_signatures.add(signature)
        unique_exact.append(state)

    kept: list[_BeamState] = []
    kept_exact: set[tuple[Any, ...]] = set()
    structures: set[tuple[tuple[int, int], ...]] = set()
    for state in unique_exact:
        structure = _state_structure_signature(state)
        if structure in structures:
            continue
        structures.add(structure)
        kept.append(state)
        kept_exact.add(state.signature())
        if len(kept) == width:
            break
    if len(kept) < width:
        for state in unique_exact:
            signature = state.signature()
            if signature in kept_exact:
                continue
            kept.append(state)
            kept_exact.add(signature)
            if len(kept) == width:
                break
    pruned_count = len(states) - len(kept)
    return kept, pruned_count, duplicate_count


def _crop_transform(
    raw: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float, float], bool, float]:
    """Project a regression-head value into the renderer's safety envelope."""

    raw_zoom, raw_x, raw_y, raw_roll = raw
    zoom = min(1.18, max(1.0, raw_zoom))
    center_x = min(0.54, max(0.46, raw_x))
    center_y = min(0.54, max(0.46, raw_y))
    dx = center_x - 0.5
    dy = center_y - 0.5
    radius = math.hypot(dx, dy)
    if radius > 0.038:
        scale = 0.038 / radius
        center_x = 0.5 + dx * scale
        center_y = 0.5 + dy * scale
    roll = min(1.60, max(-1.60, raw_roll))
    zoom = min(1.18, max(zoom, minimum_roll_safe_zoom(roll)))

    attempts = (
        (zoom, center_x, center_y, roll),
        (zoom, 0.5, 0.5, roll),
        (zoom, 0.5, 0.5, 0.0),
        (1.0, 0.5, 0.5, 0.0),
    )
    applied = attempts[-1]
    for candidate in attempts:
        try:
            EditorialTransformKeyframe(
                time_seconds=0.0,
                zoom=candidate[0],
                center_x=candidate[1],
                center_y=candidate[2],
                roll_degrees=candidate[3],
            )
        except ValueError:
            continue
        applied = candidate
        break

    correction = (
        abs(applied[0] - raw_zoom) / 0.18
        + abs(applied[1] - raw_x) / 0.08
        + abs(applied[2] - raw_y) / 0.08
        + abs(applied[3] - raw_roll) / 1.60
    ) / 4.0
    if not math.isfinite(correction):
        raise StageEditDecodeInputError("crop correction score is not finite")
    corrected = any(abs(left - right) > 1e-9 for left, right in zip(raw, applied))
    return applied, corrected, correction


def _editorial_track(
    values: tuple[float, float, float, float],
) -> EditorialTransformTrack | None:
    if all(
        abs(left - right) <= 1e-9
        for left, right in zip(values, (1.0, 0.5, 0.5, 0.0), strict=True)
    ):
        return None
    return EditorialTransformTrack(
        keyframes=(
            EditorialTransformKeyframe(
                time_seconds=0.0,
                zoom=values[0],
                center_x=values[1],
                center_y=values[2],
                roll_degrees=values[3],
            ),
        ),
        reason="stage_edit_semimarkov_crop_head",
    )


def _resolved_constraints(
    section_plan: SectionPlan,
    overrides: Sequence[StageEditSectionConstraint],
    schema: StageEditTokenSchema,
    config: StageEditSemiMarkovConfig,
) -> tuple[_ResolvedSectionConstraint, ...]:
    authority = StageEditTransitionAuthority(mode=config.transition_authority)
    if authority.renderer_owned and "hard_cut" not in schema.transition_labels:
        raise StageEditDecodeInputError(
            "renderer-owned transition authority requires hard_cut metadata"
        )
    by_index: dict[int, StageEditSectionConstraint] = {}
    for item in overrides:
        if item.section_index >= len(section_plan.sections):
            raise StageEditDecodeInputError("section constraint references unknown section")
        if item.section_index in by_index:
            raise StageEditDecodeInputError("section constraint indices must be unique")
        unknown = set(item.allowed_transition_labels) - set(schema.transition_labels)
        if unknown:
            raise StageEditDecodeInputError(
                f"section constraint has unknown transitions: {sorted(unknown)}"
            )
        if authority.renderer_owned and item.allowed_transition_labels not in {
            (),
            RENDERER_OWNED_ALLOWED_MODEL_LABELS,
        }:
            raise StageEditDecodeInputError(
                "renderer-owned section constraints allow only hard_cut metadata"
            )
        by_index[item.section_index] = item

    resolved: list[_ResolvedSectionConstraint] = []
    for directive in section_plan.sections:
        override = by_index.get(directive.section_index)
        minimum = (
            override.minimum_shot_seconds
            if override is not None and override.minimum_shot_seconds is not None
            else config.minimum_shot_seconds
        )
        maximum = (
            override.maximum_shot_seconds
            if override is not None and override.maximum_shot_seconds is not None
            else config.maximum_shot_seconds
        )
        if maximum < minimum:
            raise StageEditDecodeInputError(
                f"section {directive.section_index} shot bounds are inverted"
            )
        cadence_target = directive.target_source_switches_per_minute
        if directive.role is SectionRole.CHORUS_LIKE:
            cadence_target = max(
                cadence_target,
                config.highlight_preferred_source_switches_per_minute,
            )
        rate = (
            override.maximum_switches_per_minute
            if override is not None
            and override.maximum_switches_per_minute is not None
            else min(
                config.maximum_global_switches_per_minute,
                cadence_target * config.section_cut_density_multiplier,
            )
        )
        labels = (
            RENDERER_OWNED_ALLOWED_MODEL_LABELS
            if authority.renderer_owned
            else override.allowed_transition_labels
            if override is not None and override.allowed_transition_labels
            else schema.transition_labels
        )
        resolved.append(
            _ResolvedSectionConstraint(minimum, maximum, rate, tuple(labels))
        )
    return tuple(resolved)


def _section_index_at_start(section_plan: SectionPlan, value: float) -> int:
    adjusted = min(section_plan.end_seconds, value + _TIME_EPSILON)
    return section_plan.directive_at(adjusted).section_index


def _maximum_switch_count(rate: float, duration: float) -> int:
    return max(1, int(math.ceil(rate * duration / 60.0 - 1e-12)))


def _shot_size(
    frame: StageEditHeadFrame,
    source: StageEditSourceMetadata,
) -> str:
    if frame.source_shot_sizes:
        return frame.source_shot_sizes[source.source_index]
    return source.default_shot_size


def _formation_wide_evidence(
    *,
    shot_size: str,
    formation_person_count_normalized: float | None,
    formation_pose_coverage: float | None,
    config: StageEditSemiMarkovConfig,
) -> bool:
    """Return only weak, identity-free evidence for a formation-aware wide.

    This must never be described as proof that all six IVE members are visible.
    A weak count still needs pose coverage; the IVE-only strong path accepts at
    least four independent person boxes because tiny full-stage bodies often
    fall below the upper-body pose model's usable scale.
    """

    if (
        shot_size not in _WIDE_SHOT_SIZES
        or formation_person_count_normalized is None
        or formation_person_count_normalized
        < config.minimum_formation_anchor_person_count_normalized
    ):
        return False
    strong_multi_person_evidence = (
        formation_person_count_normalized
        >= config.strong_formation_anchor_person_count_normalized
    )
    pose_backed_evidence = bool(
        formation_pose_coverage is not None and formation_pose_coverage > 0.0
    )
    return strong_multi_person_evidence or pose_backed_evidence


def _strong_formation_wide_evidence(
    *,
    shot_size: str,
    formation_person_count_normalized: float | None,
    config: StageEditSemiMarkovConfig,
) -> bool:
    return bool(
        shot_size in _WIDE_SHOT_SIZES
        and formation_person_count_normalized is not None
        and formation_person_count_normalized
        >= config.strong_formation_anchor_person_count_normalized
    )


def _wide_switch_preferred_target_score_raw(
    *,
    prior_source_switch_boundaries: int,
    prior_wide_involved_source_switch_boundaries: int,
    derivative_family_source_switched: bool,
    wide_involved_switch: bool,
    preferred_fraction: float,
) -> float:
    """Return a bounded soft reward only when a switch approaches the median.

    The first wide-involved switch can establish the visual punctuation, but
    repeating wide-to-wide family switches at a running ratio of 1.0 receives
    no further target reward. A non-wide switch may receive the reward when it
    pulls an over-wide sequence back toward the profile median. Hard floors and
    the non-wide burst/reset gates remain separate.
    """

    if (
        isinstance(prior_source_switch_boundaries, bool)
        or not isinstance(prior_source_switch_boundaries, int)
        or isinstance(prior_wide_involved_source_switch_boundaries, bool)
        or not isinstance(prior_wide_involved_source_switch_boundaries, int)
        or prior_source_switch_boundaries < 0
        or prior_wide_involved_source_switch_boundaries < 0
        or prior_wide_involved_source_switch_boundaries
        > prior_source_switch_boundaries
    ):
        raise StageEditDecodeInputError("wide-switch target counts are invalid")
    target = _finite(preferred_fraction, "preferred_fraction")
    if not 0.0 <= target <= 1.0:
        raise StageEditDecodeInputError("preferred_fraction must lie in [0, 1]")
    if not derivative_family_source_switched:
        return 0.0

    before_fraction = (
        prior_wide_involved_source_switch_boundaries
        / prior_source_switch_boundaries
        if prior_source_switch_boundaries
        else 0.0
    )
    after_switches = prior_source_switch_boundaries + 1
    after_wide = (
        prior_wide_involved_source_switch_boundaries
        + int(wide_involved_switch)
    )
    after_fraction = after_wide / after_switches
    before_distance = abs(before_fraction - target)
    after_distance = abs(after_fraction - target)
    if after_distance + _TIME_EPSILON >= before_distance:
        return 0.0
    # Retain f284's +0.10 maximum while replacing its unconditional bonus with
    # a profile-target-aware alignment reward.
    return 0.10 * max(0.0, 1.0 - after_distance)


def _duration_fraction_moves_toward_target(
    *,
    prior_target_duration: float,
    prior_known_duration: float,
    added_target_duration: float,
    added_known_duration: float,
    target_fraction: float,
) -> bool:
    before = (
        prior_target_duration / prior_known_duration
        if prior_known_duration > _TIME_EPSILON
        else 0.0
    )
    after_known = prior_known_duration + added_known_duration
    after = (
        (prior_target_duration + added_target_duration) / after_known
        if after_known > _TIME_EPSILON
        else 0.0
    )
    return abs(after - target_fraction) + _TIME_EPSILON < abs(
        before - target_fraction
    )


def _calibration_quantile_label(
    value: float | None,
    calibrated_values: Sequence[float],
) -> str | None:
    if value is None:
        return None
    for label, candidate in zip(
        ("p10", "p25", "p50", "p75"),
        calibrated_values,
        strict=True,
    ):
        if math.isclose(value, candidate, abs_tol=_TIME_EPSILON):
            return label
    return None


def _profile_calibrated_motion_bonus_active(
    config: StageEditSemiMarkovConfig,
) -> bool:
    calibration = config.highlight_motion_calibration
    if calibration is None:
        return False
    pose_threshold = config.highlight_motion_boundary_minimum_pose_motion
    continuity_threshold = (
        config.highlight_motion_boundary_minimum_continuity_score
    )
    pose_phase_threshold = (
        config.highlight_motion_boundary_minimum_pose_phase_cosine
    )
    return bool(
        pose_threshold is not None
        and continuity_threshold is not None
        and calibration.continuity_authority
        == HANDCRAFTED_CONTINUITY_AUTHORITY
        and calibration.continuity_scorer_contract
        == HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
        and _calibration_quantile_label(
            pose_threshold,
            calibration.pose_motion_minimum_quantiles,
        )
        is not None
        and _calibration_quantile_label(
            continuity_threshold,
            calibration.continuity_score_quantiles,
        )
        is not None
        and (
            pose_phase_threshold is None
            or _calibration_quantile_label(
                pose_phase_threshold,
                calibration.pose_embedding_cosine_quantiles,
            )
            is not None
        )
    )


def _highlight_motion_calibration_audit(
    config: StageEditSemiMarkovConfig,
) -> dict[str, Any]:
    calibration = config.highlight_motion_calibration
    pose_threshold = config.highlight_motion_boundary_minimum_pose_motion
    continuity_threshold = (
        config.highlight_motion_boundary_minimum_continuity_score
    )
    sharpness_threshold = config.highlight_motion_boundary_minimum_sharpness
    pose_phase_threshold = (
        config.highlight_motion_boundary_minimum_pose_phase_cosine
    )
    primary_scorer_contract = config.continuity_scorer_contract
    if primary_scorer_contract is None:
        primary_scorer_contract = (
            HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
            if config.continuity_authority
            == HANDCRAFTED_CONTINUITY_AUTHORITY
            else STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT
        )
    primary_scorer_fingerprint = config.continuity_scorer_fingerprint
    if (
        primary_scorer_fingerprint is None
        and config.continuity_authority == HANDCRAFTED_CONTINUITY_AUTHORITY
    ):
        primary_scorer_fingerprint = (
            HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
        )
    return {
        "evidence_semantics": (
            "scalar_pose_motion_amount_plus_independent_handcrafted_"
            "continuity_qualifier_"
            "plus_optional_same_time_pose_phase_cosine_and_sharpness"
        ),
        "profile_v2_contains_motion_distribution": False,
        "profile_v3_calibration_input_bound": calibration is not None,
        "calibration": calibration.to_dict() if calibration is not None else None,
        "primary_boundary_continuity": {
            "authority": config.continuity_authority,
            "scorer_contract": primary_scorer_contract,
            "scorer_fingerprint": primary_scorer_fingerprint,
            "used_for_motion_calibration_threshold": False,
        },
        "motion_continuity_qualifier": {
            "role": "independent_handcrafted_motion_eligibility_qualifier",
            "independent_from_primary_boundary_continuity": True,
            "authority": HANDCRAFTED_CONTINUITY_AUTHORITY,
            "scorer_contract": HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
            "scorer_fingerprint": (
                HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
            ),
            "runtime_score_source": (
                "recomputed_from_exact_boundary_pair_observations"
            ),
            "primary_boundary_score_reused": False,
            "used_for_motion_calibration_threshold": (
                calibration is not None and continuity_threshold is not None
            ),
        },
        "selected_thresholds": {
            "pose_motion_minimum": pose_threshold,
            "pose_motion_quantile": (
                _calibration_quantile_label(
                    pose_threshold,
                    calibration.pose_motion_minimum_quantiles,
                )
                if calibration is not None
                else None
            ),
            "continuity_score_minimum": continuity_threshold,
            "continuity_score_quantile": (
                _calibration_quantile_label(
                    continuity_threshold,
                    calibration.continuity_score_quantiles,
                )
                if calibration is not None
                else None
            ),
            "handcrafted_continuity_qualifier_score_minimum": (
                continuity_threshold
            ),
            "handcrafted_continuity_qualifier_score_quantile": (
                _calibration_quantile_label(
                    continuity_threshold,
                    calibration.continuity_score_quantiles,
                )
                if calibration is not None
                else None
            ),
            "sharpness_minimum": sharpness_threshold,
            "sharpness_quantile": (
                _calibration_quantile_label(
                    sharpness_threshold,
                    calibration.sharpness_minimum_quantiles or (),
                )
                if calibration is not None
                else None
            ),
            "pose_phase_cosine_minimum": pose_phase_threshold,
            "pose_phase_cosine_quantile": (
                _calibration_quantile_label(
                    pose_phase_threshold,
                    calibration.pose_embedding_cosine_quantiles,
                )
                if calibration is not None
                else None
            ),
        },
        "soft_bonus_active": _profile_calibrated_motion_bonus_active(config),
        "maximum_raw_bonus_per_eligible_boundary": (
            _HIGHLIGHT_MOTION_ACCENT_MAXIMUM_RAW_BONUS
        ),
        "hard_minimum_supported_boundary_count": (
            config.highlight_minimum_motion_closeup_switch_boundaries
        ),
        "hard_minimum_defaults_inactive": True,
        "high_motion_alone_is_sufficient": False,
        "high_pose_phase_alone_is_sufficient": False,
        "pose_phase_threshold_bound": pose_phase_threshold is not None,
        "soft_bonus_requires_pose_phase_when_threshold_bound": True,
        "pose_embedding_cosine_runtime_feature_available": True,
        "same_time_pose_embedding_cosine_is_temporal_direction_proof": False,
        "temporal_pose_motion": {
            "calibration_bound": False,
            "calibration": None,
            "activation_available": False,
            "boundary_alignment_proven": False,
            "default_thresholds_present": False,
            "new_hard_boundary_quota": 0,
            "requires_source_exact_two_sample_real_pose60_sidecars": True,
            "uses_previous_sparse_token_as_cut_adjacent_sample": False,
            "sparse_sidecars_are_scalar_hash_audit_only": True,
            "enters_transformer_tensors": False,
            "static_pose_match_can_receive_dynamic_bonus": False,
            "opposite_direction_can_receive_dynamic_bonus": False,
            "missing_sidecar_can_receive_dynamic_bonus": False,
        },
        "pose_phase_direction_or_velocity_proven": False,
        "temporal_direction_proof_capability_bound": False,
    }


def _montage_grammar_score_raw(
    *,
    directive: Any,
    shot_size: str,
    start_seconds: float,
    duration_seconds: float,
    previous_shot_size: str | None,
    previous_duration_seconds: float | None,
    derivative_family_source_switched: bool,
    previous_formation_wide_evidenced: bool,
    formation_person_count_normalized: float | None,
    formation_pose_coverage: float | None,
    config: StageEditSemiMarkovConfig,
    prior_source_switch_boundaries: int = 0,
    prior_wide_involved_source_switch_boundaries: int = 0,
    prior_section_known_duration_seconds: float = 0.0,
    prior_section_closeup_duration_seconds: float = 0.0,
    prior_section_nonwide_duration_seconds: float = 0.0,
    boundary_motion_accent_supported: bool = False,
) -> float:
    """Bounded structural prior; learned logits still own the edit decision.

    A formation-wide is useful as a phrase anchor, but it must not occupy an
    entire high-energy passage.  After that anchor, medium/close shots and one
    short non-wide chain create the visual lift that a uniform sequence of
    wide-to-wide cuts cannot provide.
    """

    if shot_size == "unknown":
        return -0.12
    progress = min(
        1.0,
        max(
            0.0,
            (start_seconds - directive.start_seconds)
            / max(directive.duration_seconds, _TIME_EPSILON),
        ),
    )
    anchor_zone = progress <= 0.20
    raw = 0.0
    if directive.establish_with_wide_shot and anchor_zone:
        if shot_size in _WIDE_SHOT_SIZES:
            if _formation_wide_evidence(
                shot_size=shot_size,
                formation_person_count_normalized=(
                    formation_person_count_normalized
                ),
                formation_pose_coverage=formation_pose_coverage,
                config=config,
            ):
                raw += 0.48
                if _strong_formation_wide_evidence(
                    shot_size=shot_size,
                    formation_person_count_normalized=(
                        formation_person_count_normalized
                    ),
                    config=config,
                ):
                    raw += 0.10
            else:
                raw += 0.04
        elif shot_size in _CLOSEUP_SHOT_SIZES:
            raw -= 0.12

    if directive.role is SectionRole.CHORUS_LIKE:
        if (
            prior_section_known_duration_seconds <= _TIME_EPSILON
            and previous_shot_size is not None
            and previous_shot_size != "unknown"
            and previous_duration_seconds is not None
        ):
            # Preserve the helper's useful standalone semantics in tests and
            # diagnostics; the production beam supplies exact section totals.
            prior_section_known_duration_seconds = previous_duration_seconds
            prior_section_closeup_duration_seconds = (
                previous_duration_seconds
                if previous_shot_size in _CLOSEUP_SHOT_SIZES
                else 0.0
            )
            prior_section_nonwide_duration_seconds = (
                previous_duration_seconds
                if previous_shot_size in _NONWIDE_SHOT_SIZES
                else 0.0
            )
        added_known_duration = duration_seconds
        closeup_target_improves = _duration_fraction_moves_toward_target(
            prior_target_duration=prior_section_closeup_duration_seconds,
            prior_known_duration=prior_section_known_duration_seconds,
            added_target_duration=(
                duration_seconds
                if shot_size in _CLOSEUP_SHOT_SIZES
                else 0.0
            ),
            added_known_duration=added_known_duration,
            target_fraction=config.highlight_preferred_closeup_fraction,
        )
        nonwide_target_improves = _duration_fraction_moves_toward_target(
            prior_target_duration=prior_section_nonwide_duration_seconds,
            prior_known_duration=prior_section_known_duration_seconds,
            added_target_duration=(
                duration_seconds
                if shot_size in _NONWIDE_SHOT_SIZES
                else 0.0
            ),
            added_known_duration=added_known_duration,
            target_fraction=config.highlight_preferred_nonwide_fraction,
        )
        closeup_involved_source_switch = bool(
            derivative_family_source_switched
            and previous_shot_size is not None
            and (
                previous_shot_size in _CLOSEUP_SHOT_SIZES
                or shot_size in _CLOSEUP_SHOT_SIZES
            )
        )
        if (
            directive.duration_seconds
            >= config.highlight_minimum_duration_seconds
            and closeup_involved_source_switch
            and boundary_motion_accent_supported
            and closeup_target_improves
            and _profile_calibrated_motion_bonus_active(config)
        ):
            # Profile authority, measured motion amount, resolved pair
            # continuity, and optional sharpness must all agree before this
            # bounded preference can influence ranking. It never rewards a
            # close-up that moves occupancy farther past the P50 target.
            raw += _HIGHLIGHT_MOTION_ACCENT_MAXIMUM_RAW_BONUS
        wide_involved_switch = bool(
            derivative_family_source_switched
            and previous_shot_size is not None
            and (
                previous_shot_size in _WIDE_SHOT_SIZES
                or shot_size in _WIDE_SHOT_SIZES
            )
        )
        soft_target_raw = _wide_switch_preferred_target_score_raw(
            prior_source_switch_boundaries=prior_source_switch_boundaries,
            prior_wide_involved_source_switch_boundaries=(
                prior_wide_involved_source_switch_boundaries
            ),
            derivative_family_source_switched=(
                derivative_family_source_switched
            ),
            wide_involved_switch=wide_involved_switch,
            preferred_fraction=(
                config.highlight_preferred_wide_involved_source_switch_fraction
            ),
        )
        raw += soft_target_raw
        if wide_involved_switch and soft_target_raw > 0.0:
            # Formation evidence is an additional confidence adjustment only
            # while this switch moves the running ratio toward the profile
            # median. It cannot keep rewarding a wide-only ratio of 1.0.
            if previous_formation_wide_evidenced or _formation_wide_evidence(
                shot_size=shot_size,
                formation_person_count_normalized=(
                    formation_person_count_normalized
                ),
                formation_pose_coverage=formation_pose_coverage,
                config=config,
            ):
                raw += 0.06
                if _strong_formation_wide_evidence(
                    shot_size=shot_size,
                    formation_person_count_normalized=(
                        formation_person_count_normalized
                    ),
                    config=config,
                ):
                    raw += 0.04
        if not anchor_zone:
            if shot_size in _CLOSEUP_SHOT_SIZES:
                raw += 0.48 * int(closeup_target_improves)
            elif shot_size == "medium":
                raw += 0.24 * int(nonwide_target_improves)
            elif shot_size in _WIDE_SHOT_SIZES:
                raw -= 0.12
        if previous_shot_size is not None:
            if (
                previous_shot_size in _NONWIDE_SHOT_SIZES
                and shot_size in _NONWIDE_SHOT_SIZES
            ):
                target_improves = bool(
                    closeup_target_improves
                    if shot_size in _CLOSEUP_SHOT_SIZES
                    else nonwide_target_improves
                )
                raw += 0.22 * int(target_improves)
                if (
                    target_improves
                    and previous_duration_seconds is not None
                    and previous_duration_seconds
                    <= config.highlight_burst_maximum_shot_seconds
                    and duration_seconds
                    <= config.highlight_burst_maximum_shot_seconds
                    and (
                        previous_shot_size in _CLOSEUP_SHOT_SIZES
                        or shot_size in _CLOSEUP_SHOT_SIZES
                    )
                ):
                    raw += 0.30
            elif (
                previous_shot_size in _NONWIDE_SHOT_SIZES
                and shot_size in _WIDE_SHOT_SIZES
            ):
                # A wide reset after a face-scale burst exposes the formation
                # without allowing the entire highlight to collapse to wides.
                if _formation_wide_evidence(
                    shot_size=shot_size,
                    formation_person_count_normalized=(
                        formation_person_count_normalized
                    ),
                    formation_pose_coverage=formation_pose_coverage,
                    config=config,
                ):
                    raw += 0.18
                    if _strong_formation_wide_evidence(
                        shot_size=shot_size,
                        formation_person_count_normalized=(
                            formation_person_count_normalized
                        ),
                        config=config,
                    ):
                        raw += 0.08
                else:
                    raw += 0.02
    elif directive.role is SectionRole.BUILD:
        if shot_size in _CLOSEUP_SHOT_SIZES:
            raw += 0.10 + 0.24 * progress
        elif shot_size == "medium":
            raw += 0.08 + 0.12 * progress
        elif shot_size in _WIDE_SHOT_SIZES:
            raw += 0.12 * (1.0 - progress)
    elif directive.role in {SectionRole.INTRO, SectionRole.OUTRO}:
        if shot_size in _WIDE_SHOT_SIZES:
            raw += 0.18
    else:
        # Verse/bridge passages retain breathing room; close-ups are legal and
        # learned, but the structural prior does not force a burst everywhere.
        if shot_size in {"wide", "full", "medium"}:
            raw += 0.06
    return min(1.0, max(-1.0, raw))


def _section_candidate_shot_support(
    *,
    frames: Sequence[StageEditHeadFrame],
    source_metadata: Sequence[StageEditSourceMetadata],
    section_plan: SectionPlan,
    derivative_family_ids: Sequence[int],
    config: StageEditSemiMarkovConfig,
) -> tuple[dict[str, Any], ...]:
    records: list[list[dict[str, Any]]] = [
        [] for _section in section_plan.sections
    ]
    for frame_index, frame in enumerate(frames):
        section_index = _section_index_at_start(
            section_plan, frame.master_time_seconds
        )
        directive = section_plan.sections[section_index]
        next_time = (
            frames[frame_index + 1].master_time_seconds
            if frame_index + 1 < len(frames)
            else section_plan.end_seconds
        )
        support_duration = max(
            0.0,
            min(directive.end_seconds, next_time)
            - max(directive.start_seconds, frame.master_time_seconds),
        )
        known_sizes: list[tuple[int, str, bool]] = []
        for source_index, metadata in enumerate(source_metadata):
            if not frame.effective_source_mask[source_index]:
                continue
            size = _shot_size(frame, metadata)
            if size != "unknown":
                continuity = (
                    frame.source_continuity[source_index]
                    if frame.source_continuity
                    else StageEditSourceContinuity()
                )
                known_sizes.append(
                    (
                        derivative_family_ids[source_index],
                        size,
                        _formation_wide_evidence(
                            shot_size=size,
                            formation_person_count_normalized=(
                                continuity.formation_person_count_normalized
                            ),
                            formation_pose_coverage=(
                                continuity.formation_pose_coverage
                            ),
                            config=config,
                        ),
                    )
                )
        nonwide_families = {
            family
            for family, size, _formation in known_sizes
            if size in _NONWIDE_SHOT_SIZES
        }
        records[section_index].append(
            {
                "known": bool(known_sizes),
                "wide": any(
                    size in _WIDE_SHOT_SIZES
                    for _family, size, _formation in known_sizes
                ),
                "formation_wide": any(
                    formation for _family, _size, formation in known_sizes
                ),
                "nonwide": any(
                    size in _NONWIDE_SHOT_SIZES
                    for _family, size, _formation in known_sizes
                ),
                "closeup": any(
                    size in _CLOSEUP_SHOT_SIZES
                    for _family, size, _formation in known_sizes
                ),
                "multi_family_nonwide": len(nonwide_families) >= 2,
                "support_duration_seconds": support_duration,
            }
        )

    result: list[dict[str, Any]] = []
    for directive, section_records in zip(
        section_plan.sections, records, strict=True
    ):
        count = len(section_records)
        known_count = sum(bool(item["known"]) for item in section_records)
        closeup_count = sum(bool(item["closeup"]) for item in section_records)
        nonwide_count = sum(bool(item["nonwide"]) for item in section_records)
        wide_count = sum(bool(item["wide"]) for item in section_records)
        multi_count = sum(
            bool(item["multi_family_nonwide"]) for item in section_records
        )
        support_duration = sum(
            float(item["support_duration_seconds"])
            for item in section_records
        )

        def supported_duration(name: str) -> float:
            return sum(
                float(item["support_duration_seconds"])
                for item in section_records
                if bool(item[name])
            )

        known_duration = supported_duration("known")
        wide_duration = supported_duration("wide")
        nonwide_duration = supported_duration("nonwide")
        closeup_duration = supported_duration("closeup")
        multi_duration = supported_duration("multi_family_nonwide")
        early = section_records[: config.formation_anchor_decision_window]
        result.append(
            {
                "section_index": directive.section_index,
                "candidate_frame_count": count,
                "candidate_support_duration_seconds": support_duration,
                "known_shot_size_frame_count": known_count,
                "known_shot_size_frame_fraction": (
                    known_count / count if count else 0.0
                ),
                "wide_supported_frame_count": wide_count,
                "wide_supported_frame_fraction": (
                    wide_count / count if count else 0.0
                ),
                "known_shot_size_supported_duration_seconds": known_duration,
                "known_shot_size_supported_duration_fraction": (
                    known_duration / support_duration if support_duration else 0.0
                ),
                "wide_supported_duration_seconds": wide_duration,
                "wide_supported_duration_fraction": (
                    wide_duration / support_duration if support_duration else 0.0
                ),
                "nonwide_supported_frame_count": nonwide_count,
                "nonwide_supported_frame_fraction": (
                    nonwide_count / count if count else 0.0
                ),
                "nonwide_supported_duration_seconds": nonwide_duration,
                "nonwide_supported_duration_fraction": (
                    nonwide_duration / support_duration if support_duration else 0.0
                ),
                "closeup_supported_frame_count": closeup_count,
                "closeup_supported_frame_fraction": (
                    closeup_count / count if count else 0.0
                ),
                "closeup_supported_duration_seconds": closeup_duration,
                "closeup_supported_duration_fraction": (
                    closeup_duration / support_duration if support_duration else 0.0
                ),
                "multi_family_nonwide_supported_frame_count": multi_count,
                "multi_family_nonwide_supported_duration_seconds": multi_duration,
                "early_wide_supported": any(
                    bool(item["wide"]) for item in early
                ),
                "early_formation_wide_supported": any(
                    bool(item["formation_wide"]) for item in early
                ),
            }
        )
    return tuple(result)


def _exact_chorus_anchor_metrics(
    *,
    decisions: Sequence[StageEditBeamDecision],
    section_plan: SectionPlan,
) -> dict[str, Any]:
    """Mirror the immutable reference profile's exact token-reset semantics."""

    missing_section_progress = sum(
        item.section_progress is None for item in decisions
    )
    missing_phrase_progress = sum(item.phrase_progress is None for item in decisions)
    missing_section_role = sum(item.section_role is None for item in decisions)
    section_role_plan_mismatch = sum(
        item.section_role is not None
        and item.section_role
        != section_plan.directive_at(item.master_start_seconds).role.value
        for item in decisions
    )

    exact_sections: list[list[StageEditBeamDecision]] = []
    if missing_section_progress == 0 and missing_section_role == 0:
        for decision in decisions:
            if not exact_sections:
                exact_sections.append([decision])
                continue
            previous = exact_sections[-1][-1]
            assert previous.section_progress is not None
            assert decision.section_progress is not None
            if (
                decision.section_role != previous.section_role
                or decision.section_progress + _TIME_EPSILON
                < previous.section_progress
            ):
                exact_sections.append([decision])
            else:
                exact_sections[-1].append(decision)

    observed_early = 0
    early_wide = 0
    for selected in exact_sections:
        if selected[0].section_role != SectionRole.CHORUS_LIKE.value:
            continue
        # Match the profile exactly: take the first three labelled shot
        # positions, then exclude unknown shot-size evidence.  An unknown
        # decision therefore still consumes an early-anchor slot.
        early_known = tuple(
            item for item in selected[:3] if item.shot_size != "unknown"
        )
        if early_known:
            observed_early += 1
            if any(item.shot_size in _WIDE_SHOT_SIZES for item in early_known):
                early_wide += 1

    phrases: list[list[StageEditBeamDecision]] = []
    if missing_phrase_progress == 0:
        for decision in decisions:
            if not phrases:
                phrases.append([decision])
                continue
            previous = phrases[-1][-1]
            assert previous.phrase_progress is not None
            assert decision.phrase_progress is not None
            if (
                decision.phrase_progress + _TIME_EPSILON
                < previous.phrase_progress
            ):
                phrases.append([decision])
            else:
                phrases[-1].append(decision)

    observed_phrase_entries = 0
    phrase_entry_wide = 0
    for phrase in phrases:
        entry = phrase[0]
        if entry.section_role != SectionRole.CHORUS_LIKE.value:
            continue
        if entry.shot_size == "unknown":
            continue
        observed_phrase_entries += 1
        if entry.shot_size in _WIDE_SHOT_SIZES:
            phrase_entry_wide += 1

    return {
        "evidence_contract": (
            "serialized_inference_token_section_and_phrase_progress_reset_v1"
        ),
        "missing_section_progress_decision_count": missing_section_progress,
        "missing_section_role_decision_count": missing_section_role,
        "section_role_plan_mismatch_count": section_role_plan_mismatch,
        "missing_phrase_progress_decision_count": missing_phrase_progress,
        "observed_chorus_section_early_anchor_count": observed_early,
        "chorus_section_early_wide_anchor_count": early_wide,
        "chorus_section_early_wide_anchor_fraction": (
            early_wide / observed_early if observed_early else None
        ),
        "observed_chorus_phrase_entry_anchor_count": observed_phrase_entries,
        "chorus_phrase_entry_wide_anchor_count": phrase_entry_wide,
        "chorus_phrase_entry_wide_anchor_fraction": (
            phrase_entry_wide / observed_phrase_entries
            if observed_phrase_entries
            else None
        ),
    }


def _maximum_consecutive_wide_duration_seconds(
    decisions: Sequence[StageEditBeamDecision],
) -> float:
    """Measure maximal contiguous known-wide occupancy in one section."""

    maximum = 0.0
    current = 0.0
    for decision in decisions:
        if decision.shot_size in _WIDE_SHOT_SIZES:
            current += decision.duration_seconds
            maximum = max(maximum, current)
        else:
            # Unknown evidence cannot be treated as wide and cannot bridge two
            # otherwise separate known-wide runs.
            current = 0.0
    return maximum


def _close_only_multi_source_burst_evidence(
    decisions: Sequence[StageEditBeamDecision],
    config: StageEditSemiMarkovConfig,
) -> dict[str, Any]:
    """Audit maximal close-only runs and their bounded conditional-event fit."""

    active = config.highlight_close_burst_policy_binding_sha256 is not None
    if not active:
        return {
            "contract": CLOSE_BURST_EVENT_EVIDENCE_CONTRACT,
            "policy_binding_active": False,
            "policy_binding": None,
            "candidate_event_count": 0,
            "qualified_event_count": 0,
            "events": [],
            "saturated_soft_preference_score": None,
            "absence_is_hard_failure": False,
            "hard_gate": False,
        }
    assert config.highlight_minimum_close_burst_shot_count is not None
    assert config.highlight_minimum_close_burst_duration_seconds is not None
    assert (
        config.highlight_minimum_close_burst_source_switch_fraction is not None
    )
    assert config.highlight_close_burst_event_reference_count is not None
    assert config.highlight_policy_fingerprint_sha256 is not None
    policy_binding = close_burst_policy_binding_payload(
        profile_identity={
            "analysis_basis": config.highlight_policy_profile_analysis_basis,
            "schema_version": config.highlight_policy_profile_schema_version,
            "path": config.highlight_policy_profile_path,
            "file_sha256": config.highlight_policy_profile_file_sha256,
            "content_sha256": config.highlight_policy_profile_content_sha256,
        },
        policy_fingerprint_sha256=config.highlight_policy_fingerprint_sha256,
        minimum_shot_count=config.highlight_minimum_close_burst_shot_count,
        minimum_duration_seconds=(
            config.highlight_minimum_close_burst_duration_seconds
        ),
        minimum_source_switch_fraction=(
            config.highlight_minimum_close_burst_source_switch_fraction
        ),
        event_reference_count=config.highlight_close_burst_event_reference_count,
    )
    policy_binding["binding_sha256"] = (
        config.highlight_close_burst_policy_binding_sha256
    )

    maximal_close_runs: list[tuple[StageEditBeamDecision, ...]] = []
    current: list[StageEditBeamDecision] = []
    for decision in decisions:
        if decision.shot_size in _CLOSEUP_SHOT_SIZES:
            current.append(decision)
            continue
        if current:
            maximal_close_runs.append(tuple(current))
            current = []
    if current:
        maximal_close_runs.append(tuple(current))

    events: list[dict[str, Any]] = []
    for close_run in maximal_close_runs:
        switch_positions = tuple(
            position
            for position, (left, right) in enumerate(
                zip(close_run, close_run[1:]),
                start=1,
            )
            if left.derivative_family_id != right.derivative_family_id
        )
        if not switch_positions:
            continue
        shot_count = len(close_run)
        duration_seconds = sum(item.duration_seconds for item in close_run)
        source_switch_count = len(switch_positions)
        source_switch_fraction = source_switch_count / max(1, shot_count - 1)
        qualified = bool(
            shot_count + _TIME_EPSILON
            >= config.highlight_minimum_close_burst_shot_count
            and duration_seconds + _TIME_EPSILON
            >= config.highlight_minimum_close_burst_duration_seconds
            and source_switch_fraction + _TIME_EPSILON
            >= config.highlight_minimum_close_burst_source_switch_fraction
        )
        switch_decisions = tuple(close_run[position] for position in switch_positions)
        events.append(
            {
                "first_shot_index": close_run[0].shot_index,
                "last_shot_index": close_run[-1].shot_index,
                "start_seconds": close_run[0].master_start_seconds,
                "end_seconds": close_run[-1].master_end_seconds,
                "shot_count": shot_count,
                "duration_seconds": duration_seconds,
                "source_switch_count": source_switch_count,
                "source_switch_fraction": source_switch_fraction,
                "derivative_family_ids": [
                    item.derivative_family_id for item in close_run
                ],
                "motion_amount_plus_continuity_supported_switch_count": sum(
                    int(item.boundary_motion_amount_plus_continuity_supported)
                    for item in switch_decisions
                ),
                "same_time_pose_phase_matched_switch_count": sum(
                    int(item.boundary_pose_phase_match_proven)
                    for item in switch_decisions
                ),
                "temporal_direction_proven_switch_count": sum(
                    int(item.boundary_pose_direction_or_velocity_proven)
                    for item in switch_decisions
                ),
                "qualified": qualified,
            }
        )
    qualified_count = sum(int(item["qualified"]) for item in events)
    return {
        "contract": CLOSE_BURST_EVENT_EVIDENCE_CONTRACT,
        "policy_binding_active": True,
        "policy_binding": policy_binding,
        "candidate_event_count": len(events),
        "qualified_event_count": qualified_count,
        "events": events,
        "saturated_soft_preference_score": 1.0 if qualified_count else 0.0,
        "saturation": "one_or_more_qualified_events",
        "absence_is_hard_failure": False,
        "hard_gate": False,
        "medium_to_close_counts_as_close_only": False,
        "constituent_switch_motion_preference_remains_separate": True,
        "temporal_direction_threshold_bound": (
            config.highlight_temporal_pose_motion_calibration is not None
        ),
    }


def _symmetric_fraction_target_evidence(
    observed: float,
    target: float,
) -> dict[str, float | bool]:
    deviation = abs(observed - target)
    return {
        "target": target,
        "observed": observed,
        "absolute_deviation": deviation,
        "shortfall": max(0.0, target - observed),
        "excess": max(0.0, observed - target),
        # A bounded triangular score is genuinely symmetric around P50.
        # Going beyond the target therefore loses, rather than earns, reward.
        "alignment_score": max(0.0, 1.0 - deviation),
        "hard_gate": False,
    }


def _wide_run_soft_target_evidence(
    observed: float,
    preferred_maximum: float,
    hard_maximum: float,
) -> dict[str, float | bool | str]:
    excess = max(0.0, observed - preferred_maximum)
    span = max(_TIME_EPSILON, hard_maximum - preferred_maximum)
    return {
        "target": preferred_maximum,
        "observed": observed,
        "upper_target_excess_seconds": excess,
        "alignment_score": (
            1.0 if excess <= _TIME_EPSILON else max(0.0, 1.0 - excess / span)
        ),
        "preference": "at_or_below",
        "saturated_at_or_below_target": True,
        "hard_gate": False,
    }


def _highlight_policy_provenance(
    config: StageEditSemiMarkovConfig,
) -> dict[str, Any]:
    identity_bound = config.highlight_policy_fingerprint_sha256 is not None
    profile_identity = (
        {
            "analysis_basis": config.highlight_policy_profile_analysis_basis,
            "schema_version": config.highlight_policy_profile_schema_version,
            "path": config.highlight_policy_profile_path,
            "file_sha256": config.highlight_policy_profile_file_sha256,
            "content_sha256": config.highlight_policy_profile_content_sha256,
        }
        if identity_bound
        else None
    )
    close_burst_binding = None
    if identity_bound:
        assert profile_identity is not None
        assert config.highlight_policy_fingerprint_sha256 is not None
        assert config.highlight_minimum_close_burst_shot_count is not None
        assert config.highlight_minimum_close_burst_duration_seconds is not None
        assert (
            config.highlight_minimum_close_burst_source_switch_fraction
            is not None
        )
        assert config.highlight_close_burst_event_reference_count is not None
        close_burst_binding = close_burst_policy_binding_payload(
            profile_identity=profile_identity,
            policy_fingerprint_sha256=(
                config.highlight_policy_fingerprint_sha256
            ),
            minimum_shot_count=(
                config.highlight_minimum_close_burst_shot_count
            ),
            minimum_duration_seconds=(
                config.highlight_minimum_close_burst_duration_seconds
            ),
            minimum_source_switch_fraction=(
                config.highlight_minimum_close_burst_source_switch_fraction
            ),
            event_reference_count=(
                config.highlight_close_burst_event_reference_count
            ),
        )
        close_burst_binding["binding_sha256"] = (
            config.highlight_close_burst_policy_binding_sha256
        )

    def metric(
        *,
        config_field: str,
        policy_member: str,
        value: float,
        statistic_path: str,
        quantile: float,
        hard_gate: bool,
        scope: str,
        comparison: str,
    ) -> dict[str, Any]:
        return {
            "config_field": config_field,
            "professional_montage_policy_member": policy_member,
            "value": value,
            "statistic_path": statistic_path,
            "quantile": quantile,
            "hard_gate": hard_gate,
            "scope": scope,
            "comparison": comparison,
        }

    return {
        "contract": HIGHLIGHT_AGGREGATE_CONTRACT,
        "profile_identity_bound": identity_bound,
        "profile_identity": profile_identity,
        "policy_fingerprint_sha256": (
            config.highlight_policy_fingerprint_sha256
        ),
        "close_burst_soft_policy_binding": close_burst_binding,
        "duration_evidence_contract": HIGHLIGHT_DURATION_EVIDENCE_CONTRACT,
        "wide_run_evidence_contract": HIGHLIGHT_WIDE_RUN_EVIDENCE_CONTRACT,
        "metrics": {
            "local_closeup_duration_minimum": metric(
                config_field="highlight_minimum_closeup_fraction",
                policy_member="minimum_close_duration_fraction",
                value=config.highlight_minimum_closeup_fraction,
                statistic_path=HIGHLIGHT_CLOSE_DURATION_STATISTIC_PATH,
                quantile=REFERENCE_FLOOR_QUANTILE,
                hard_gate=True,
                scope="each_strict_highlight",
                comparison="greater_than_or_equal",
            ),
            "local_nonwide_duration_minimum": metric(
                config_field="highlight_minimum_nonwide_fraction",
                policy_member="minimum_nonwide_duration_fraction",
                value=config.highlight_minimum_nonwide_fraction,
                statistic_path=HIGHLIGHT_NONWIDE_DURATION_STATISTIC_PATH,
                quantile=REFERENCE_FLOOR_QUANTILE,
                hard_gate=True,
                scope="each_strict_highlight",
                comparison="greater_than_or_equal",
            ),
            "pooled_closeup_duration_product_minimum": metric(
                config_field=(
                    "highlight_pooled_product_minimum_closeup_fraction"
                ),
                policy_member=(
                    "highlight_pooled_product_minimum_close_duration_fraction"
                ),
                value=(
                    config.highlight_pooled_product_minimum_closeup_fraction
                ),
                statistic_path=HIGHLIGHT_CLOSE_DURATION_STATISTIC_PATH,
                quantile=DURATION_PRODUCT_MINIMUM_QUANTILE,
                hard_gate=True,
                scope="pooled_strict_highlights",
                comparison="greater_than_or_equal",
            ),
            "pooled_nonwide_duration_product_minimum": metric(
                config_field=(
                    "highlight_pooled_product_minimum_nonwide_fraction"
                ),
                policy_member=(
                    "highlight_pooled_product_minimum_nonwide_duration_fraction"
                ),
                value=(
                    config.highlight_pooled_product_minimum_nonwide_fraction
                ),
                statistic_path=HIGHLIGHT_NONWIDE_DURATION_STATISTIC_PATH,
                quantile=DURATION_PRODUCT_MINIMUM_QUANTILE,
                hard_gate=True,
                scope="pooled_strict_highlights",
                comparison="greater_than_or_equal",
            ),
            "pooled_closeup_duration_preferred_target": metric(
                config_field="highlight_preferred_closeup_fraction",
                policy_member="preferred_close_duration_fraction",
                value=config.highlight_preferred_closeup_fraction,
                statistic_path=HIGHLIGHT_CLOSE_DURATION_STATISTIC_PATH,
                quantile=DURATION_PREFERRED_TARGET_QUANTILE,
                hard_gate=False,
                scope="pooled_strict_highlights",
                comparison="symmetric_target_deviation",
            ),
            "pooled_nonwide_duration_preferred_target": metric(
                config_field="highlight_preferred_nonwide_fraction",
                policy_member="preferred_nonwide_duration_fraction",
                value=config.highlight_preferred_nonwide_fraction,
                statistic_path=HIGHLIGHT_NONWIDE_DURATION_STATISTIC_PATH,
                quantile=DURATION_PREFERRED_TARGET_QUANTILE,
                hard_gate=False,
                scope="pooled_strict_highlights",
                comparison="symmetric_target_deviation",
            ),
            "maximum_consecutive_wide_run": metric(
                config_field="highlight_maximum_consecutive_wide_run_seconds",
                policy_member="maximum_consecutive_wide_run_seconds",
                value=config.highlight_maximum_consecutive_wide_run_seconds,
                statistic_path=HIGHLIGHT_WIDE_RUN_STATISTIC_PATH,
                quantile=WIDE_RUN_HARD_MAXIMUM_QUANTILE,
                hard_gate=True,
                scope="each_strict_highlight",
                comparison="less_than_or_equal",
            ),
            "preferred_maximum_consecutive_wide_run": metric(
                config_field=(
                    "highlight_preferred_maximum_consecutive_wide_run_seconds"
                ),
                policy_member=(
                    "preferred_maximum_consecutive_wide_run_seconds"
                ),
                value=(
                    config.highlight_preferred_maximum_consecutive_wide_run_seconds
                ),
                statistic_path=HIGHLIGHT_WIDE_RUN_STATISTIC_PATH,
                quantile=WIDE_RUN_PREFERRED_MAXIMUM_QUANTILE,
                hard_gate=False,
                scope="maximum_across_strict_highlights",
                comparison="soft_upper_target",
            ),
        },
    }


def _evaluate_montage_grammar(
    *,
    decisions: Sequence[StageEditBeamDecision],
    section_plan: SectionPlan,
    support_by_section: Sequence[Mapping[str, Any]],
    config: StageEditSemiMarkovConfig,
) -> dict[str, Any]:
    exact_anchor_metrics = _exact_chorus_anchor_metrics(
        decisions=decisions,
        section_plan=section_plan,
    )
    by_section: list[list[StageEditBeamDecision]] = [
        [] for _section in section_plan.sections
    ]
    for decision in decisions:
        section_index = _section_index_at_start(
            section_plan, decision.master_start_seconds
        )
        by_section[section_index].append(decision)

    all_failures: list[str] = []
    section_metrics: list[dict[str, Any]] = []
    strict_highlight_records: list[dict[str, Any]] = []
    for directive, selected, support in zip(
        section_plan.sections,
        by_section,
        support_by_section,
        strict=True,
    ):
        known = [item for item in selected if item.shot_size != "unknown"]
        closeups = [
            item for item in known if item.shot_size in _CLOSEUP_SHOT_SIZES
        ]
        nonwides = [
            item for item in known if item.shot_size in _NONWIDE_SHOT_SIZES
        ]
        wides = [item for item in known if item.shot_size in _WIDE_SHOT_SIZES]
        selected_duration = sum(item.duration_seconds for item in selected)
        known_duration = sum(item.duration_seconds for item in known)
        closeup_duration = sum(item.duration_seconds for item in closeups)
        nonwide_duration = sum(item.duration_seconds for item in nonwides)
        wide_duration = sum(item.duration_seconds for item in wides)
        known_fraction = (
            known_duration / selected_duration if selected_duration else 0.0
        )
        closeup_fraction = closeup_duration / known_duration if known_duration else 0.0
        nonwide_fraction = nonwide_duration / known_duration if known_duration else 0.0
        wide_fraction = wide_duration / known_duration if known_duration else 0.0
        maximum_consecutive_wide_run_seconds = (
            _maximum_consecutive_wide_duration_seconds(selected)
        )
        close_only_burst_evidence = _close_only_multi_source_burst_evidence(
            selected,
            config,
        )
        burst_boundaries = 0
        burst_right_positions: list[int] = []
        closeup_chain_boundaries = 0
        source_switch_boundaries = 0
        wide_involved_source_switch_boundaries = 0
        closeup_involved_source_switch_boundaries = 0
        nonwide_involved_source_switch_boundaries = 0
        motion_closeup_source_switch_boundaries = 0
        motion_nonwide_source_switch_boundaries = 0
        motion_evidence_complete_closeup_boundaries = 0
        motion_evidence_complete_nonwide_boundaries = 0
        pose_phase_observed_closeup_boundaries = 0
        pose_phase_observed_nonwide_boundaries = 0
        pose_phase_matched_closeup_boundaries = 0
        pose_phase_matched_nonwide_boundaries = 0
        temporal_input_valid_closeup_boundaries = 0
        temporal_input_valid_nonwide_boundaries = 0
        temporal_direction_proven_closeup_boundaries = 0
        temporal_direction_proven_nonwide_boundaries = 0
        temporal_static_match_closeup_boundaries = 0
        temporal_static_match_nonwide_boundaries = 0
        for position, (left, right) in enumerate(zip(selected, selected[1:])):
            if left.derivative_family_id == right.derivative_family_id:
                continue
            source_switch_boundaries += 1
            closeup_involved = bool(
                left.shot_size in _CLOSEUP_SHOT_SIZES
                or right.shot_size in _CLOSEUP_SHOT_SIZES
            )
            nonwide_involved = bool(
                left.shot_size in _NONWIDE_SHOT_SIZES
                or right.shot_size in _NONWIDE_SHOT_SIZES
            )
            if closeup_involved:
                closeup_involved_source_switch_boundaries += 1
                motion_evidence_complete_closeup_boundaries += int(
                    right.boundary_motion_amount_plus_continuity_evidence_complete
                )
                motion_closeup_source_switch_boundaries += int(
                    right.boundary_motion_amount_plus_continuity_supported
                )
                pose_phase_observed_closeup_boundaries += int(
                    right.boundary_pose_phase_evidence_observed
                )
                pose_phase_matched_closeup_boundaries += int(
                    right.boundary_pose_phase_match_proven
                )
                temporal_input_valid_closeup_boundaries += int(
                    right.boundary_temporal_pose_motion_input_valid
                )
                temporal_direction_proven_closeup_boundaries += int(
                    right.boundary_pose_direction_or_velocity_proven
                )
                temporal_static_match_closeup_boundaries += int(
                    right.boundary_temporal_static_pose_match
                )
            if nonwide_involved:
                nonwide_involved_source_switch_boundaries += 1
                motion_evidence_complete_nonwide_boundaries += int(
                    right.boundary_motion_amount_plus_continuity_evidence_complete
                )
                motion_nonwide_source_switch_boundaries += int(
                    right.boundary_motion_amount_plus_continuity_supported
                )
                pose_phase_observed_nonwide_boundaries += int(
                    right.boundary_pose_phase_evidence_observed
                )
                pose_phase_matched_nonwide_boundaries += int(
                    right.boundary_pose_phase_match_proven
                )
                temporal_input_valid_nonwide_boundaries += int(
                    right.boundary_temporal_pose_motion_input_valid
                )
                temporal_direction_proven_nonwide_boundaries += int(
                    right.boundary_pose_direction_or_velocity_proven
                )
                temporal_static_match_nonwide_boundaries += int(
                    right.boundary_temporal_static_pose_match
                )
            if (
                left.shot_size in _WIDE_SHOT_SIZES
                or right.shot_size in _WIDE_SHOT_SIZES
            ):
                wide_involved_source_switch_boundaries += 1
            if (
                left.shot_size in _CLOSEUP_SHOT_SIZES
                and right.shot_size in _CLOSEUP_SHOT_SIZES
            ):
                closeup_chain_boundaries += 1
            if (
                left.shot_size in _NONWIDE_SHOT_SIZES
                and right.shot_size in _NONWIDE_SHOT_SIZES
                and (
                    left.shot_size in _CLOSEUP_SHOT_SIZES
                    or right.shot_size in _CLOSEUP_SHOT_SIZES
                )
                and left.duration_seconds
                <= config.highlight_burst_maximum_shot_seconds
                and right.duration_seconds
                <= config.highlight_burst_maximum_shot_seconds
            ):
                burst_boundaries += 1
                burst_right_positions.append(position + 1)
        wide_involved_source_switch_fraction = (
            wide_involved_source_switch_boundaries / source_switch_boundaries
            if source_switch_boundaries
            else 0.0
        )
        source_switch_rate_per_minute = (
            60.0 * source_switch_boundaries / directive.duration_seconds
        )
        source_switch_cadence_target_deviation = abs(
            source_switch_rate_per_minute
            - config.highlight_preferred_source_switches_per_minute
        )
        wide_switch_preferred_target_deviation = abs(
            wide_involved_source_switch_fraction
            - config.highlight_preferred_wide_involved_source_switch_fraction
        )
        anchor_window = selected[: config.formation_anchor_decision_window]
        generic_wide_anchor_present = any(
            item.shot_size in _WIDE_SHOT_SIZES for item in anchor_window
        )
        formation_anchor_present = any(
            _formation_wide_evidence(
                shot_size=item.shot_size,
                formation_person_count_normalized=(
                    item.formation_person_count_normalized
                ),
                formation_pose_coverage=item.formation_pose_coverage,
                config=config,
            )
            for item in anchor_window
        )
        post_burst_formation_resets: set[int] = set()
        for burst_right in burst_right_positions:
            for candidate_position in range(burst_right + 1, len(selected)):
                candidate = selected[candidate_position]
                previous = selected[candidate_position - 1]
                if (
                    _formation_wide_evidence(
                        shot_size=candidate.shot_size,
                        formation_person_count_normalized=(
                            candidate.formation_person_count_normalized
                        ),
                        formation_pose_coverage=(
                            candidate.formation_pose_coverage
                        ),
                        config=config,
                    )
                    and previous.shot_size in _NONWIDE_SHOT_SIZES
                    and previous.derivative_family_id
                    != candidate.derivative_family_id
                ):
                    post_burst_formation_resets.add(candidate_position)
                    break

        strict_highlight = (
            config.enforce_montage_grammar
            and directive.role is SectionRole.CHORUS_LIKE
            and directive.duration_seconds
            >= config.highlight_minimum_duration_seconds
        )
        failures: list[str] = []
        if strict_highlight:
            if (
                source_switch_rate_per_minute + _TIME_EPSILON
                < config.highlight_product_minimum_source_switches_per_minute
            ):
                failures.append(
                    "highlight_source_switch_rate_below_product_minimum"
                )
            if (
                known_fraction + _TIME_EPSILON
                < config.highlight_minimum_known_shot_fraction
            ):
                failures.append("highlight_shot_size_evidence_incomplete")
            if (
                float(support["closeup_supported_duration_fraction"])
                + _TIME_EPSILON
                < config.highlight_minimum_closeup_fraction
            ):
                failures.append("highlight_closeup_candidate_support_insufficient")
            elif (
                closeup_fraction + _TIME_EPSILON
                < config.highlight_minimum_closeup_fraction
            ):
                failures.append("highlight_closeup_fraction_below_floor")
            if (
                float(support["nonwide_supported_duration_fraction"])
                + _TIME_EPSILON
                < config.highlight_minimum_nonwide_fraction
            ):
                failures.append("highlight_nonwide_candidate_support_insufficient")
            elif nonwide_fraction + _TIME_EPSILON < (
                config.highlight_minimum_nonwide_fraction
            ):
                failures.append("highlight_nonwide_fraction_below_floor")
            if (
                maximum_consecutive_wide_run_seconds
                > config.highlight_maximum_consecutive_wide_run_seconds
                + _TIME_EPSILON
            ):
                failures.append(
                    "highlight_maximum_consecutive_wide_run_exceeded"
                )
            if (
                motion_closeup_source_switch_boundaries
                < config.highlight_minimum_motion_closeup_switch_boundaries
            ):
                failures.append(
                    "highlight_motion_closeup_switch_boundary_missing"
                )
            if (
                source_switch_boundaries > 0
                and wide_involved_source_switch_fraction + _TIME_EPSILON
                < config.highlight_minimum_wide_involved_source_switch_fraction
            ):
                failures.append(
                    "highlight_wide_involved_source_switch_fraction_below_"
                    "reference_floor"
                )
            if int(
                support["multi_family_nonwide_supported_frame_count"]
            ) < 2:
                failures.append(
                    "highlight_multi_source_nonwide_support_insufficient"
                )
            elif (
                int(support["multi_family_nonwide_supported_frame_count"])
                >= 2
                and burst_boundaries
                < config.highlight_minimum_nonwide_burst_boundaries
            ):
                failures.append("highlight_nonwide_burst_missing")
            if (
                burst_boundaries
                >= config.highlight_minimum_nonwide_burst_boundaries
                and len(post_burst_formation_resets)
                < config.highlight_minimum_post_burst_formation_resets
            ):
                failures.append("post_burst_formation_wide_reset_missing")
            if (
                directive.establish_with_wide_shot
                and not bool(support["early_wide_supported"])
            ):
                failures.append("formation_wide_candidate_support_insufficient")
            elif (
                directive.establish_with_wide_shot
                and not generic_wide_anchor_present
            ):
                failures.append("formation_wide_anchor_missing")
            elif (
                directive.establish_with_wide_shot
                and not bool(support["early_formation_wide_supported"])
            ):
                failures.append(
                    "formation_wide_candidate_evidence_insufficient"
                )
            elif directive.establish_with_wide_shot and not formation_anchor_present:
                failures.append("formation_wide_anchor_evidence_missing")
            strict_highlight_records.append(
                {
                    "section_index": directive.section_index,
                    "known_duration_seconds": known_duration,
                    "closeup_duration_seconds": closeup_duration,
                    "nonwide_duration_seconds": nonwide_duration,
                    "maximum_consecutive_wide_run_seconds": (
                        maximum_consecutive_wide_run_seconds
                    ),
                    "candidate_frame_count": int(
                        support["candidate_frame_count"]
                    ),
                    "closeup_supported_frame_count": int(
                        support["closeup_supported_frame_count"]
                    ),
                    "nonwide_supported_frame_count": int(
                        support["nonwide_supported_frame_count"]
                    ),
                    "candidate_support_duration_seconds": float(
                        support["candidate_support_duration_seconds"]
                    ),
                    "closeup_supported_duration_seconds": float(
                        support["closeup_supported_duration_seconds"]
                    ),
                    "nonwide_supported_duration_seconds": float(
                        support["nonwide_supported_duration_seconds"]
                    ),
                    "close_only_multi_source_burst": (
                        close_only_burst_evidence
                    ),
                "closeup_involved_source_switch_boundary_count": (
                    closeup_involved_source_switch_boundaries
                ),
                "motion_closeup_source_switch_boundary_count": (
                    motion_closeup_source_switch_boundaries
                ),
                "pose_phase_observed_closeup_source_switch_boundary_count": (
                    pose_phase_observed_closeup_boundaries
                ),
                "pose_phase_matched_closeup_source_switch_boundary_count": (
                    pose_phase_matched_closeup_boundaries
                ),
                "temporal_input_valid_closeup_source_switch_boundary_count": (
                    temporal_input_valid_closeup_boundaries
                ),
                "temporal_direction_proven_closeup_source_switch_boundary_count": (
                    temporal_direction_proven_closeup_boundaries
                ),
                "temporal_static_match_closeup_source_switch_boundary_count": (
                    temporal_static_match_closeup_boundaries
                ),
                "nonwide_involved_source_switch_boundary_count": (
                    nonwide_involved_source_switch_boundaries
                ),
                "motion_nonwide_source_switch_boundary_count": (
                    motion_nonwide_source_switch_boundaries
                ),
                "pose_phase_observed_nonwide_source_switch_boundary_count": (
                    pose_phase_observed_nonwide_boundaries
                ),
                "pose_phase_matched_nonwide_source_switch_boundary_count": (
                    pose_phase_matched_nonwide_boundaries
                ),
                "temporal_input_valid_nonwide_source_switch_boundary_count": (
                    temporal_input_valid_nonwide_boundaries
                ),
                "temporal_direction_proven_nonwide_source_switch_boundary_count": (
                    temporal_direction_proven_nonwide_boundaries
                ),
                "temporal_static_match_nonwide_source_switch_boundary_count": (
                    temporal_static_match_nonwide_boundaries
                ),
                "local_failure_codes": tuple(failures),
                }
            )
        all_failures.extend(
            f"section_{directive.section_index}:{value}" for value in failures
        )
        section_metrics.append(
            {
                "section_index": directive.section_index,
                "role": directive.role.value,
                "duration_seconds": directive.duration_seconds,
                "strict_highlight_gate_applied": strict_highlight,
                "selected_shot_count": len(selected),
                "known_shot_size_count": len(known),
                "known_shot_size_fraction": known_fraction,
                "known_duration_seconds": known_duration,
                "wide_shot_count": len(wides),
                "wide_shot_fraction": wide_fraction,
                "wide_duration_seconds": wide_duration,
                "nonwide_shot_count": len(nonwides),
                "nonwide_shot_fraction": nonwide_fraction,
                "nonwide_duration_seconds": nonwide_duration,
                "closeup_shot_count": len(closeups),
                "closeup_shot_fraction": closeup_fraction,
                "closeup_duration_seconds": closeup_duration,
                "maximum_consecutive_wide_run_seconds": (
                    maximum_consecutive_wide_run_seconds
                ),
                "maximum_consecutive_wide_run_hard_limit_seconds": (
                    config.highlight_maximum_consecutive_wide_run_seconds
                ),
                "preferred_maximum_consecutive_wide_run_seconds": (
                    config.highlight_preferred_maximum_consecutive_wide_run_seconds
                ),
                "closeup_preferred_target": _symmetric_fraction_target_evidence(
                    closeup_fraction,
                    config.highlight_preferred_closeup_fraction,
                ),
                "nonwide_preferred_target": _symmetric_fraction_target_evidence(
                    nonwide_fraction,
                    config.highlight_preferred_nonwide_fraction,
                ),
                "wide_run_preferred_target": _wide_run_soft_target_evidence(
                    maximum_consecutive_wide_run_seconds,
                    config.highlight_preferred_maximum_consecutive_wide_run_seconds,
                    config.highlight_maximum_consecutive_wide_run_seconds,
                ),
                "nonwide_burst_boundary_count": burst_boundaries,
                "closeup_chain_boundary_count": closeup_chain_boundaries,
                "close_only_multi_source_burst": close_only_burst_evidence,
                "motion_amount_plus_continuity_boundaries": {
                    "evidence_contract": (
                        "exact_source_pair_pose_motion_plus_independent_"
                        "handcrafted_continuity_qualifier_plus_optional_same_"
                        "time_pose_phase_cosine_and_sharpness_v3"
                    ),
                    "motion_continuity_qualifier": {
                        "role": (
                            "independent_handcrafted_motion_eligibility_qualifier"
                        ),
                        "independent_from_primary_boundary_continuity": True,
                        "authority": HANDCRAFTED_CONTINUITY_AUTHORITY,
                        "scorer_contract": (
                            HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
                        ),
                        "scorer_fingerprint": (
                            HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
                        ),
                        "primary_boundary_score_reused": False,
                    },
                    "profile_v2_human_distribution_available": False,
                    "profile_v3_calibration_input_bound": (
                        config.highlight_motion_calibration is not None
                    ),
                    "profile_v3_derivation_required": (
                        config.highlight_motion_calibration is None
                    ),
                    "closeup_involved_source_switch_boundary_count": (
                        closeup_involved_source_switch_boundaries
                    ),
                    "closeup_motion_evidence_complete_boundary_count": (
                        motion_evidence_complete_closeup_boundaries
                    ),
                    "closeup_motion_accent_supported_boundary_count": (
                        motion_closeup_source_switch_boundaries
                    ),
                    "closeup_pose_phase_observed_boundary_count": (
                        pose_phase_observed_closeup_boundaries
                    ),
                    "closeup_pose_phase_matched_boundary_count": (
                        pose_phase_matched_closeup_boundaries
                    ),
                    "closeup_temporal_input_valid_boundary_count": (
                        temporal_input_valid_closeup_boundaries
                    ),
                    "closeup_temporal_direction_proven_boundary_count": (
                        temporal_direction_proven_closeup_boundaries
                    ),
                    "closeup_temporal_static_pose_match_boundary_count": (
                        temporal_static_match_closeup_boundaries
                    ),
                    "closeup_static_or_unsupported_boundary_count": (
                        closeup_involved_source_switch_boundaries
                        - motion_closeup_source_switch_boundaries
                    ),
                    "nonwide_involved_source_switch_boundary_count": (
                        nonwide_involved_source_switch_boundaries
                    ),
                    "nonwide_motion_evidence_complete_boundary_count": (
                        motion_evidence_complete_nonwide_boundaries
                    ),
                    "nonwide_motion_accent_supported_boundary_count": (
                        motion_nonwide_source_switch_boundaries
                    ),
                    "nonwide_pose_phase_observed_boundary_count": (
                        pose_phase_observed_nonwide_boundaries
                    ),
                    "nonwide_pose_phase_matched_boundary_count": (
                        pose_phase_matched_nonwide_boundaries
                    ),
                    "nonwide_temporal_input_valid_boundary_count": (
                        temporal_input_valid_nonwide_boundaries
                    ),
                    "nonwide_temporal_direction_proven_boundary_count": (
                        temporal_direction_proven_nonwide_boundaries
                    ),
                    "nonwide_temporal_static_pose_match_boundary_count": (
                        temporal_static_match_nonwide_boundaries
                    ),
                    "nonwide_static_or_unsupported_boundary_count": (
                        nonwide_involved_source_switch_boundaries
                        - motion_nonwide_source_switch_boundaries
                    ),
                    "minimum_closeup_motion_boundary_count": (
                        config.highlight_minimum_motion_closeup_switch_boundaries
                    ),
                    "minimum_pose_motion": (
                        config.highlight_motion_boundary_minimum_pose_motion
                    ),
                    "minimum_handcrafted_continuity_qualifier_score": (
                        config.highlight_motion_boundary_minimum_continuity_score
                    ),
                    # Retained as a clearly labelled compatibility alias for
                    # pre-v11 report readers.  It is never the primary learned
                    # boundary score.
                    "minimum_continuity_score": (
                        config.highlight_motion_boundary_minimum_continuity_score
                    ),
                    "minimum_continuity_score_legacy_alias_semantics": (
                        "minimum_handcrafted_continuity_qualifier_score"
                    ),
                    "minimum_sharpness": (
                        config.highlight_motion_boundary_minimum_sharpness
                    ),
                    "minimum_pose_phase_cosine": (
                        config.highlight_motion_boundary_minimum_pose_phase_cosine
                    ),
                    "high_motion_alone_is_sufficient": False,
                    "high_pose_phase_alone_is_sufficient": False,
                    "pose_phase_proof_requires_observed_same_time_pair": True,
                    "pose_direction_or_velocity_proven": bool(
                        temporal_direction_proven_closeup_boundaries
                        or temporal_direction_proven_nonwide_boundaries
                    ),
                },
                "source_switch_boundary_count": source_switch_boundaries,
                "source_switch_rate_per_minute": source_switch_rate_per_minute,
                "source_switch_product_minimum_per_minute": (
                    config.highlight_product_minimum_source_switches_per_minute
                ),
                "source_switch_preferred_target_per_minute": (
                    config.highlight_preferred_source_switches_per_minute
                ),
                "source_switch_preferred_target_deviation": (
                    source_switch_cadence_target_deviation
                ),
                "source_switch_product_minimum_is_reference_floor": False,
                "wide_involved_source_switch_boundary_count": (
                    wide_involved_source_switch_boundaries
                ),
                "wide_involved_source_switch_fraction": (
                    wide_involved_source_switch_fraction
                ),
                "wide_involved_source_switch_preferred_target": (
                    config.highlight_preferred_wide_involved_source_switch_fraction
                ),
                "wide_involved_source_switch_preferred_target_deviation": (
                    wide_switch_preferred_target_deviation
                ),
                "wide_involved_source_switch_preferred_target_shortfall": max(
                    0.0,
                    config.highlight_preferred_wide_involved_source_switch_fraction
                    - wide_involved_source_switch_fraction,
                ),
                "wide_involved_source_switch_preferred_target_is_hard_gate": (
                    False
                ),
                "formation_anchor_decision_window": (
                    config.formation_anchor_decision_window
                ),
                "generic_wide_anchor_present": generic_wide_anchor_present,
                "formation_wide_anchor_present": formation_anchor_present,
                "post_burst_formation_wide_reset_count": len(
                    post_burst_formation_resets
                ),
                "candidate_support": dict(support),
                "failure_codes": failures,
                "passed": not failures,
            }
        )
    aggregate_known_duration = sum(
        float(item["known_duration_seconds"])
        for item in strict_highlight_records
    )
    aggregate_closeup_duration = sum(
        float(item["closeup_duration_seconds"])
        for item in strict_highlight_records
    )
    aggregate_nonwide_duration = sum(
        float(item["nonwide_duration_seconds"])
        for item in strict_highlight_records
    )
    aggregate_closeup_fraction = (
        aggregate_closeup_duration / aggregate_known_duration
        if aggregate_known_duration
        else 0.0
    )
    aggregate_nonwide_fraction = (
        aggregate_nonwide_duration / aggregate_known_duration
        if aggregate_known_duration
        else 0.0
    )
    aggregate_maximum_wide_run = max(
        (
            float(item["maximum_consecutive_wide_run_seconds"])
            for item in strict_highlight_records
        ),
        default=0.0,
    )
    aggregate_candidate_frames = sum(
        int(item["candidate_frame_count"])
        for item in strict_highlight_records
    )
    aggregate_closeup_supported_frames = sum(
        int(item["closeup_supported_frame_count"])
        for item in strict_highlight_records
    )
    aggregate_nonwide_supported_frames = sum(
        int(item["nonwide_supported_frame_count"])
        for item in strict_highlight_records
    )
    aggregate_candidate_support_duration = sum(
        float(item["candidate_support_duration_seconds"])
        for item in strict_highlight_records
    )
    aggregate_closeup_supported_duration = sum(
        float(item["closeup_supported_duration_seconds"])
        for item in strict_highlight_records
    )
    aggregate_nonwide_supported_duration = sum(
        float(item["nonwide_supported_duration_seconds"])
        for item in strict_highlight_records
    )
    aggregate_closeup_switch_boundaries = sum(
        int(item["closeup_involved_source_switch_boundary_count"])
        for item in strict_highlight_records
    )
    aggregate_motion_closeup_switch_boundaries = sum(
        int(item["motion_closeup_source_switch_boundary_count"])
        for item in strict_highlight_records
    )
    aggregate_nonwide_switch_boundaries = sum(
        int(item["nonwide_involved_source_switch_boundary_count"])
        for item in strict_highlight_records
    )
    aggregate_motion_nonwide_switch_boundaries = sum(
        int(item["motion_nonwide_source_switch_boundary_count"])
        for item in strict_highlight_records
    )
    aggregate_close_burst_candidate_count = sum(
        int(item["close_only_multi_source_burst"]["candidate_event_count"])
        for item in strict_highlight_records
    )
    aggregate_close_burst_qualified_count = sum(
        int(item["close_only_multi_source_burst"]["qualified_event_count"])
        for item in strict_highlight_records
    )
    close_burst_binding_active = bool(
        strict_highlight_records
        and strict_highlight_records[0]["close_only_multi_source_burst"][
            "policy_binding_active"
        ]
    )
    close_burst_soft_score = (
        1.0 if aggregate_close_burst_qualified_count else 0.0
    ) if close_burst_binding_active else None
    close_burst_occurrence_support_weight = (
        min(
            1.0,
            config.highlight_close_burst_event_reference_count
            / config.highlight_cadence_reference_count,
        )
        if close_burst_binding_active
        and config.highlight_close_burst_event_reference_count is not None
        else None
    )
    aggregate_close_burst_events = [
        {
            "section_index": int(item["section_index"]),
            **dict(event),
        }
        for item in strict_highlight_records
        for event in item["close_only_multi_source_burst"]["events"]
    ]
    aggregate_closeup_support_frame_fraction = (
        aggregate_closeup_supported_frames / aggregate_candidate_frames
        if aggregate_candidate_frames
        else 0.0
    )
    aggregate_nonwide_support_frame_fraction = (
        aggregate_nonwide_supported_frames / aggregate_candidate_frames
        if aggregate_candidate_frames
        else 0.0
    )
    aggregate_closeup_support_fraction = (
        aggregate_closeup_supported_duration
        / aggregate_candidate_support_duration
        if aggregate_candidate_support_duration
        else 0.0
    )
    aggregate_nonwide_support_fraction = (
        aggregate_nonwide_supported_duration
        / aggregate_candidate_support_duration
        if aggregate_candidate_support_duration
        else 0.0
    )
    aggregate_failures: list[str] = [
        f"section_{item['section_index']}:{failure}"
        for item in strict_highlight_records
        for failure in item["local_failure_codes"]
    ]
    if strict_highlight_records:
        if (
            aggregate_closeup_support_fraction + _TIME_EPSILON
            < config.highlight_pooled_product_minimum_closeup_fraction
        ):
            aggregate_failures.append(
                "highlight_pooled_closeup_candidate_support_insufficient"
            )
        elif (
            aggregate_closeup_fraction + _TIME_EPSILON
            < config.highlight_pooled_product_minimum_closeup_fraction
        ):
            aggregate_failures.append(
                "highlight_pooled_closeup_fraction_below_product_minimum"
            )
        if (
            aggregate_nonwide_support_fraction + _TIME_EPSILON
            < config.highlight_pooled_product_minimum_nonwide_fraction
        ):
            aggregate_failures.append(
                "highlight_pooled_nonwide_candidate_support_insufficient"
            )
        elif (
            aggregate_nonwide_fraction + _TIME_EPSILON
            < config.highlight_pooled_product_minimum_nonwide_fraction
        ):
            aggregate_failures.append(
                "highlight_pooled_nonwide_fraction_below_product_minimum"
            )
    pooled_failures = [
        item
        for item in aggregate_failures
        if not item.startswith("section_")
    ]
    all_failures.extend(pooled_failures)
    closeup_target = _symmetric_fraction_target_evidence(
        aggregate_closeup_fraction,
        config.highlight_preferred_closeup_fraction,
    )
    nonwide_target = _symmetric_fraction_target_evidence(
        aggregate_nonwide_fraction,
        config.highlight_preferred_nonwide_fraction,
    )
    wide_run_target = _wide_run_soft_target_evidence(
        aggregate_maximum_wide_run,
        config.highlight_preferred_maximum_consecutive_wide_run_seconds,
        config.highlight_maximum_consecutive_wide_run_seconds,
    )
    soft_target_score = (
        (
            float(closeup_target["alignment_score"])
            + float(nonwide_target["alignment_score"])
            + float(wide_run_target["alignment_score"])
        )
        / 3.0
        if strict_highlight_records
        else None
    )
    highlight_aggregate = {
        "contract": HIGHLIGHT_AGGREGATE_CONTRACT,
        "strict_highlight_section_indices": [
            int(item["section_index"]) for item in strict_highlight_records
        ],
        "strict_highlight_section_count": len(strict_highlight_records),
        "duration_weighting": (
            "sum_observed_known_shot_duration_across_strict_highlights"
        ),
        "observed": {
            "known_duration_seconds": aggregate_known_duration,
            "closeup_duration_seconds": aggregate_closeup_duration,
            "nonwide_duration_seconds": aggregate_nonwide_duration,
            "closeup_duration_fraction": aggregate_closeup_fraction,
            "nonwide_duration_fraction": aggregate_nonwide_fraction,
            "maximum_consecutive_wide_run_seconds": (
                aggregate_maximum_wide_run
            ),
            "closeup_involved_source_switch_boundary_count": (
                aggregate_closeup_switch_boundaries
            ),
            "motion_closeup_source_switch_boundary_count": (
                aggregate_motion_closeup_switch_boundaries
            ),
            "static_or_unsupported_closeup_source_switch_boundary_count": (
                aggregate_closeup_switch_boundaries
                - aggregate_motion_closeup_switch_boundaries
            ),
            "nonwide_involved_source_switch_boundary_count": (
                aggregate_nonwide_switch_boundaries
            ),
            "motion_nonwide_source_switch_boundary_count": (
                aggregate_motion_nonwide_switch_boundaries
            ),
            "static_or_unsupported_nonwide_source_switch_boundary_count": (
                aggregate_nonwide_switch_boundaries
                - aggregate_motion_nonwide_switch_boundaries
            ),
            "close_only_multi_source_burst_candidate_event_count": (
                aggregate_close_burst_candidate_count
            ),
            "close_only_multi_source_burst_qualified_event_count": (
                aggregate_close_burst_qualified_count
            ),
        },
        "candidate_support": {
            "candidate_frame_count": aggregate_candidate_frames,
            "closeup_supported_frame_count": (
                aggregate_closeup_supported_frames
            ),
            "nonwide_supported_frame_count": (
                aggregate_nonwide_supported_frames
            ),
            "closeup_supported_frame_fraction": (
                aggregate_closeup_support_frame_fraction
            ),
            "nonwide_supported_frame_fraction": (
                aggregate_nonwide_support_frame_fraction
            ),
            "candidate_support_duration_seconds": (
                aggregate_candidate_support_duration
            ),
            "closeup_supported_duration_seconds": (
                aggregate_closeup_supported_duration
            ),
            "nonwide_supported_duration_seconds": (
                aggregate_nonwide_supported_duration
            ),
            "closeup_supported_duration_fraction": (
                aggregate_closeup_support_fraction
            ),
            "nonwide_supported_duration_fraction": (
                aggregate_nonwide_support_fraction
            ),
            "product_minimum_closeup_fraction": (
                config.highlight_pooled_product_minimum_closeup_fraction
            ),
            "product_minimum_nonwide_fraction": (
                config.highlight_pooled_product_minimum_nonwide_fraction
            ),
            "insufficient_support_fails_before_render": True,
            "hard_gate_uses_duration_weighted_support": True,
        },
        "hard_targets": {
            "local_minimum_closeup_duration_fraction": (
                config.highlight_minimum_closeup_fraction
            ),
            "local_minimum_nonwide_duration_fraction": (
                config.highlight_minimum_nonwide_fraction
            ),
            "pooled_product_minimum_closeup_duration_fraction": (
                config.highlight_pooled_product_minimum_closeup_fraction
            ),
            "pooled_product_minimum_nonwide_duration_fraction": (
                config.highlight_pooled_product_minimum_nonwide_fraction
            ),
            "maximum_consecutive_wide_run_seconds": (
                config.highlight_maximum_consecutive_wide_run_seconds
            ),
        },
        "soft_targets": {
            "closeup_p50": closeup_target,
            "nonwide_p50": nonwide_target,
            "maximum_wide_run_p75": wide_run_target,
            "close_only_multi_source_burst_event_p10": {
                "contract": CLOSE_BURST_EVENT_EVIDENCE_CONTRACT,
                "policy_binding_active": close_burst_binding_active,
                "policy_binding": (
                    strict_highlight_records[0][
                        "close_only_multi_source_burst"
                    ]["policy_binding"]
                    if close_burst_binding_active
                    else None
                ),
                "candidate_event_count": aggregate_close_burst_candidate_count,
                "qualified_event_count": aggregate_close_burst_qualified_count,
                "events": aggregate_close_burst_events,
                "alignment_score": close_burst_soft_score,
                "event_occurrence_support_weight": (
                    close_burst_occurrence_support_weight
                ),
                "evidence_weighted_alignment_score": (
                    close_burst_soft_score * close_burst_occurrence_support_weight
                    if close_burst_soft_score is not None
                    and close_burst_occurrence_support_weight is not None
                    else None
                ),
                "observed_event_count": (
                    config.highlight_close_burst_event_reference_count
                ),
                "human_reference_count": config.highlight_cadence_reference_count,
                "saturated_at_one_or_more_qualified_events": True,
                "absence_is_hard_failure": False,
                "hard_gate": False,
                "per_reference_occurrence_p10": 0.0,
                "per_reference_occurrence_median": 0.0,
                "medium_to_close_counts_as_close_only": False,
            },
            "aggregate_alignment_score": soft_target_score,
            "score_is_hard_gate": False,
            "over_close_past_target_is_rewarded": False,
        },
        "policy_provenance": _highlight_policy_provenance(config),
        "motion_amount_plus_continuity_policy": (
            _highlight_motion_calibration_audit(config)
        ),
        "local_section_failures": {
            str(item["section_index"]): list(item["local_failure_codes"])
            for item in strict_highlight_records
            if item["local_failure_codes"]
        },
        "failure_codes": aggregate_failures,
        "passed": not aggregate_failures,
    }

    strict_chorus_present = any(
        directive.role is SectionRole.CHORUS_LIKE
        and directive.duration_seconds >= config.highlight_minimum_duration_seconds
        for directive in section_plan.sections
    )
    if config.enforce_montage_grammar and strict_chorus_present:
        if (
            int(exact_anchor_metrics["missing_section_progress_decision_count"])
            or int(exact_anchor_metrics["missing_section_role_decision_count"])
        ):
            all_failures.append("highlight_exact_section_progress_evidence_missing")
        elif int(exact_anchor_metrics["section_role_plan_mismatch_count"]):
            all_failures.append("highlight_exact_section_role_plan_mismatch")
        elif not int(
            exact_anchor_metrics["observed_chorus_section_early_anchor_count"]
        ):
            all_failures.append("highlight_early_wide_anchor_evidence_missing")
        elif float(
            exact_anchor_metrics["chorus_section_early_wide_anchor_fraction"]
        ) + _TIME_EPSILON < config.highlight_minimum_early_wide_anchor_fraction:
            all_failures.append(
                "highlight_early_wide_anchor_fraction_below_reference_floor"
            )
        if int(exact_anchor_metrics["missing_phrase_progress_decision_count"]):
            all_failures.append("highlight_exact_phrase_progress_evidence_missing")
        elif not int(
            exact_anchor_metrics["observed_chorus_phrase_entry_anchor_count"]
        ):
            all_failures.append("highlight_phrase_entry_wide_anchor_evidence_missing")
        elif float(
            exact_anchor_metrics["chorus_phrase_entry_wide_anchor_fraction"]
        ) + _TIME_EPSILON < (
            config.highlight_minimum_phrase_entry_wide_anchor_fraction
        ):
            all_failures.append(
                "highlight_phrase_entry_wide_anchor_fraction_below_reference_floor"
            )
    return {
        "schema_version": 2,
        "contract": MONTAGE_GRAMMAR_CONTRACT,
        "passed": not all_failures,
        "failure_codes": all_failures,
        "exact_anchor_metrics": exact_anchor_metrics,
        "highlight_aggregate": highlight_aggregate,
        "policy": {
            "wide_shots": (
                "formation_evidenced_anchor_and_post_burst_reset_not_full_"
                "highlight_occupancy"
            ),
            "highlight_shots": (
                "per_section_duration_p10_plus_pooled_duration_p25_and_"
                "maximum_wide_run_hard_gates_with_bounded_p50_p75_soft_"
                "targets"
            ),
            "formation_evidence_semantics": (
                "identity_free_weak_person_plus_pose_or_strong_multi_person_"
                "box_evidence_not_proof_of_complete_group_formation"
            ),
            "strong_formation_wide_product_preference": {
                "minimum_normalized_person_box_count": (
                    config.strong_formation_anchor_person_count_normalized
                ),
                "minimum_ive_roster_equivalent_count": 4,
                "requires_identity_labels": False,
                "proves_all_six_members_visible": False,
                "maximum_anchor_raw_bonus": 0.10,
                "maximum_switch_raw_bonus": 0.04,
                "maximum_post_burst_reset_raw_bonus": 0.08,
            },
            "insufficient_closeup_or_nonwide_candidates": (
                "fail_before_render_instead_of_fabricating_a_flat_highlight"
            ),
            "learned_logits_remain_primary": True,
            "chorus_source_switch_cadence": {
                "evidence_contract": HIGHLIGHT_CADENCE_EVIDENCE_CONTRACT,
                "eligible_human_reference_count": (
                    config.highlight_cadence_reference_count
                ),
                "product_minimum_per_minute": (
                    config.highlight_product_minimum_source_switches_per_minute
                ),
                "product_minimum_statistic": "reference_median",
                "product_minimum_is_reference_floor": False,
                "soft_preferred_target_per_minute": (
                    config.highlight_preferred_source_switches_per_minute
                ),
                "soft_preferred_target_statistic": "reference_p75",
                "global_safety_maximum_per_minute": (
                    config.maximum_global_switches_per_minute
                ),
            },
            "chorus_wide_involved_family_switch_prior": {
                "reference_p10_hard_floor": (
                    config.highlight_minimum_wide_involved_source_switch_fraction
                ),
                "reference_median": (
                    config.highlight_preferred_wide_involved_source_switch_fraction
                ),
                "reference_median_soft_preferred_target": (
                    config.highlight_preferred_wide_involved_source_switch_fraction
                ),
                "eligible_human_reference_count": (
                    config.highlight_wide_involved_source_switch_reference_count
                ),
                "maximum_target_alignment_raw_bonus": 0.10,
                "formation_evidenced_wide_additional_raw_bonus": 0.06,
                "strong_multi_person_wide_additional_raw_bonus": 0.04,
                "same_source_or_derivative_family_bonus": 0.0,
                "repeated_wide_only_switch_bonus_after_no_target_improvement": (
                    0.0
                ),
                "hard_gate": False,
                "applied_through_montage_grammar_weight": (
                    config.montage_grammar_score_weight
                ),
            },
            "motion_amount_plus_continuity_closeup_switch_hook": {
                **_highlight_motion_calibration_audit(config),
                "eligible_boundary": (
                    "strict_chorus_closeup_involved_actual_source_switch_"
                    "that_moves_closeup_occupancy_toward_p50"
                ),
                "score_component": "montage_grammar",
                "over_close_past_p50_is_rewarded": False,
                "pose_phase_proof_is_per_actual_family_boundary": True,
                "pose_direction_or_velocity_proven": any(
                    decision.boundary_pose_direction_or_velocity_proven
                    for decision in decisions
                ),
            },
            "close_only_multi_source_burst_soft_preference": {
                "contract": CLOSE_BURST_EVENT_EVIDENCE_CONTRACT,
                "conditional_event_thresholds_not_occurrence_quota": True,
                "per_reference_occurrence_p10": 0.0,
                "per_reference_occurrence_median": 0.0,
                "hard_gate": False,
                "saturated_at_one_or_more_qualified_events": True,
                "score_component": (
                    "highlight_close_burst_soft_preference"
                ),
                "applied_through_montage_grammar_weight": (
                    config.montage_grammar_score_weight
                ),
            },
        },
        "thresholds": {
            "highlight_minimum_duration_seconds": (
                config.highlight_minimum_duration_seconds
            ),
            "highlight_minimum_known_shot_fraction": (
                config.highlight_minimum_known_shot_fraction
            ),
            "highlight_minimum_closeup_fraction": (
                config.highlight_minimum_closeup_fraction
            ),
            "highlight_minimum_nonwide_fraction": (
                config.highlight_minimum_nonwide_fraction
            ),
            "highlight_pooled_product_minimum_closeup_fraction": (
                config.highlight_pooled_product_minimum_closeup_fraction
            ),
            "highlight_pooled_product_minimum_nonwide_fraction": (
                config.highlight_pooled_product_minimum_nonwide_fraction
            ),
            "highlight_preferred_closeup_fraction": (
                config.highlight_preferred_closeup_fraction
            ),
            "highlight_preferred_nonwide_fraction": (
                config.highlight_preferred_nonwide_fraction
            ),
            "highlight_maximum_consecutive_wide_run_seconds": (
                config.highlight_maximum_consecutive_wide_run_seconds
            ),
            "highlight_preferred_maximum_consecutive_wide_run_seconds": (
                config.highlight_preferred_maximum_consecutive_wide_run_seconds
            ),
            "highlight_minimum_wide_involved_source_switch_fraction": (
                config.highlight_minimum_wide_involved_source_switch_fraction
            ),
            "highlight_preferred_wide_involved_source_switch_fraction": (
                config.highlight_preferred_wide_involved_source_switch_fraction
            ),
            "highlight_minimum_early_wide_anchor_fraction": (
                config.highlight_minimum_early_wide_anchor_fraction
            ),
            "highlight_minimum_phrase_entry_wide_anchor_fraction": (
                config.highlight_minimum_phrase_entry_wide_anchor_fraction
            ),
            "highlight_burst_maximum_shot_seconds": (
                config.highlight_burst_maximum_shot_seconds
            ),
            "highlight_product_minimum_source_switches_per_minute": (
                config.highlight_product_minimum_source_switches_per_minute
            ),
            "highlight_preferred_source_switches_per_minute": (
                config.highlight_preferred_source_switches_per_minute
            ),
            "highlight_cadence_reference_count": (
                config.highlight_cadence_reference_count
            ),
            "highlight_minimum_nonwide_burst_boundaries": (
                config.highlight_minimum_nonwide_burst_boundaries
            ),
            "highlight_minimum_close_burst_shot_count": (
                config.highlight_minimum_close_burst_shot_count
            ),
            "highlight_minimum_close_burst_duration_seconds": (
                config.highlight_minimum_close_burst_duration_seconds
            ),
            "highlight_minimum_close_burst_source_switch_fraction": (
                config.highlight_minimum_close_burst_source_switch_fraction
            ),
            "highlight_close_burst_event_reference_count": (
                config.highlight_close_burst_event_reference_count
            ),
            "highlight_close_burst_policy_binding_sha256": (
                config.highlight_close_burst_policy_binding_sha256
            ),
            "formation_anchor_decision_window": (
                config.formation_anchor_decision_window
            ),
            "minimum_formation_anchor_person_count_normalized": (
                config.minimum_formation_anchor_person_count_normalized
            ),
            "strong_formation_anchor_person_count_normalized": (
                config.strong_formation_anchor_person_count_normalized
            ),
            "highlight_minimum_post_burst_formation_resets": (
                config.highlight_minimum_post_burst_formation_resets
            ),
            "highlight_motion_boundary_minimum_pose_motion": (
                config.highlight_motion_boundary_minimum_pose_motion
            ),
            "highlight_motion_boundary_minimum_continuity_score": (
                config.highlight_motion_boundary_minimum_continuity_score
            ),
            "highlight_motion_boundary_minimum_sharpness": (
                config.highlight_motion_boundary_minimum_sharpness
            ),
            "highlight_motion_boundary_minimum_pose_phase_cosine": (
                config.highlight_motion_boundary_minimum_pose_phase_cosine
            ),
            "highlight_minimum_motion_closeup_switch_boundaries": (
                config.highlight_minimum_motion_closeup_switch_boundaries
            ),
        },
        "sections": section_metrics,
    }


def _bounded_continuity_delta(left: float, right: float, scale: float) -> float:
    return min(1.0, abs(left - right) / scale)


def _continuity_score_provenance(
    config: StageEditSemiMarkovConfig,
    source_pair_scorer: StageEditSourcePairScorer | None,
) -> dict[str, Any]:
    """Resolve one unambiguous scorer authority without exposing weights."""

    if config.continuity_authority == HANDCRAFTED_CONTINUITY_AUTHORITY:
        if source_pair_scorer is not None:
            raise StageEditDecodeInputError(
                "handcrafted continuity authority rejects an unexpected "
                "learned source-pair scorer"
            )
        scorer_contract = HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
        scorer_fingerprint = HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
    else:
        if not isinstance(source_pair_scorer, StageEditSourcePairScorer):
            raise StageEditDecodeInputError(
                "production learned continuity authority requires a typed "
                "StageEditSourcePairScorer"
            )
        scorer_contract = source_pair_scorer.contract
        scorer_fingerprint = source_pair_scorer.fingerprint
        if scorer_contract != STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT:
            raise StageEditDecodeInputError(
                "learned source-pair scorer contract mismatch"
            )
    if (
        config.continuity_scorer_contract is not None
        and config.continuity_scorer_contract != scorer_contract
    ):
        raise StageEditDecodeInputError(
            "configured continuity scorer contract does not match the scorer"
        )
    if (
        config.continuity_scorer_fingerprint is not None
        and config.continuity_scorer_fingerprint != scorer_fingerprint
    ):
        raise StageEditDecodeInputError(
            "configured continuity scorer fingerprint does not match the scorer"
        )
    return {
        "authority_contract": CONTINUITY_AUTHORITY_CONTRACT,
        "authority": config.continuity_authority,
        "scorer_contract": scorer_contract,
        "scorer_fingerprint": scorer_fingerprint,
        "scorer_weights_embedded": False,
        "score_range": [-1.0, 0.0],
        "applies_to": "actual_derivative_family_boundaries_only",
        "opening_and_same_family_raw_score": 0.0,
    }


def _audit_config_payload(
    config: StageEditSemiMarkovConfig,
    boundary_continuity_score_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve optional scorer binding fields into a round-trippable config."""

    return {
        **asdict(config),
        "continuity_scorer_contract": (
            boundary_continuity_score_provenance["scorer_contract"]
        ),
        "continuity_scorer_fingerprint": (
            boundary_continuity_score_provenance["scorer_fingerprint"]
        ),
    }


def stage_edit_boundary_continuity_score(
    outgoing: StageEditSourceContinuity,
    incoming: StageEditSourceContinuity,
) -> float:
    """Return a neutral-or-negative same-time boundary continuity score.

    The decoder should prefer a naturally matchable pair, but absent evidence
    must never look like a perfect detected match. Consequently this score is
    a bounded penalty in ``[-1, 0]``. Comparable observations determine the
    main penalty and uncovered evidence receives a small fixed uncertainty
    penalty. Exact-frame render-domain geometry and look gates remain
    authoritative.
    """

    weighted_penalties: list[tuple[float, float]] = []

    if outgoing.face_scale is not None and incoming.face_scale is not None:
        left = max(abs(outgoing.face_scale), 1.0e-6)
        right = max(abs(incoming.face_scale), 1.0e-6)
        scale_penalty = min(1.0, abs(math.log(right / left)) / math.log(1.50))
        weighted_penalties.append((0.28, scale_penalty))
    if all(
        value is not None
        for value in (
            outgoing.face_center_x,
            outgoing.face_center_y,
            incoming.face_center_x,
            incoming.face_center_y,
        )
    ):
        face_distance = math.hypot(
            float(incoming.face_center_x) - float(outgoing.face_center_x),
            float(incoming.face_center_y) - float(outgoing.face_center_y),
        )
        weighted_penalties.append((0.28, min(1.0, face_distance / 0.22)))
    elif all(
        value is not None
        for value in (
            outgoing.body_center_x,
            outgoing.body_center_y,
            incoming.body_center_x,
            incoming.body_center_y,
        )
    ):
        body_distance = math.hypot(
            float(incoming.body_center_x) - float(outgoing.body_center_x),
            float(incoming.body_center_y) - float(outgoing.body_center_y),
        )
        weighted_penalties.append((0.18, min(1.0, body_distance / 0.28)))

    if all(
        value is not None
        for value in (
            outgoing.camera_motion_x,
            outgoing.camera_motion_y,
            incoming.camera_motion_x,
            incoming.camera_motion_y,
        )
    ):
        motion_delta = math.hypot(
            float(incoming.camera_motion_x) - float(outgoing.camera_motion_x),
            float(incoming.camera_motion_y) - float(outgoing.camera_motion_y),
        )
        weighted_penalties.append((0.12, min(1.0, motion_delta / 0.75)))
    if outgoing.pose_motion is not None and incoming.pose_motion is not None:
        weighted_penalties.append(
            (
                0.06,
                _bounded_continuity_delta(
                    outgoing.pose_motion, incoming.pose_motion, 0.40
                ),
            )
        )
    if (
        outgoing.camera_zoom_velocity is not None
        and incoming.camera_zoom_velocity is not None
    ):
        weighted_penalties.append(
            (
                0.05,
                _bounded_continuity_delta(
                    outgoing.camera_zoom_velocity,
                    incoming.camera_zoom_velocity,
                    0.35,
                ),
            )
        )
    if (
        outgoing.camera_roll_normalized is not None
        and incoming.camera_roll_normalized is not None
    ):
        weighted_penalties.append(
            (
                0.05,
                _bounded_continuity_delta(
                    outgoing.camera_roll_normalized,
                    incoming.camera_roll_normalized,
                    0.25,
                ),
            )
        )

    for weight, name, scale in (
        (0.06, "exposure", 0.25),
        (0.04, "color_temperature_normalized", 0.30),
        (0.03, "sharpness", 0.35),
    ):
        left = getattr(outgoing, name)
        right = getattr(incoming, name)
        if left is not None and right is not None:
            weighted_penalties.append(
                (weight, _bounded_continuity_delta(left, right, scale))
            )

    observed_weight = sum(weight for weight, _penalty in weighted_penalties)
    if observed_weight:
        observed_penalty = sum(
            weight * value for weight, value in weighted_penalties
        ) / observed_weight
    else:
        observed_penalty = 0.0
    # The face-based full evidence path totals 0.97. Using its weight as the
    # coverage denominator keeps body-fallback evidence slightly conservative.
    evidence_coverage = min(1.0, observed_weight / 0.97)
    penalty = (
        evidence_coverage * observed_penalty
        + (1.0 - evidence_coverage) * 0.12
    )
    return -min(1.0, max(0.0, penalty))


def _boundary_continuity_raw(
    frame: StageEditHeadFrame,
    outgoing_source_index: int | None,
    incoming_source_index: int,
    derivative_family_ids: Sequence[int],
    *,
    continuity_authority: str,
    source_pair_scorer: StageEditSourcePairScorer | None,
) -> float:
    if outgoing_source_index is None:
        return 0.0
    if (
        derivative_family_ids[outgoing_source_index]
        == derivative_family_ids[incoming_source_index]
    ):
        return 0.0
    if not frame.source_continuity:
        if continuity_authority == PRODUCTION_LEARNED_CONTINUITY_AUTHORITY:
            raise StageEditDecodeInputError(
                "learned boundary scoring is missing exact source continuity"
            )
        return 0.0
    outgoing = frame.source_continuity[outgoing_source_index]
    incoming = frame.source_continuity[incoming_source_index]
    if continuity_authority == PRODUCTION_LEARNED_CONTINUITY_AUTHORITY:
        if not isinstance(source_pair_scorer, StageEditSourcePairScorer):
            raise StageEditDecodeInputError(
                "learned boundary scoring is missing its source-pair scorer"
            )
        raw = source_pair_scorer.score(outgoing, incoming)
    elif continuity_authority == HANDCRAFTED_CONTINUITY_AUTHORITY:
        raw = stage_edit_boundary_continuity_score(outgoing, incoming)
    else:  # Config validation should make this unreachable.
        raise StageEditDecodeInputError(
            "boundary continuity authority is unsupported"
        )
    result = _finite(raw, "boundary_continuity_raw")
    if not -1.0 <= result <= 0.0:
        raise StageEditDecodeInputError(
            "boundary continuity scorer violated its bounded [-1, 0] contract"
        )
    return result


def _temporal_pose_sidecar_pair_audit(
    *,
    frame: StageEditHeadFrame,
    outgoing_source_index: int,
    incoming_source_index: int,
) -> dict[str, Any]:
    """Describe sidecars without claiming they touch the current cut.

    Snapshot tokens are sparse editorial candidates.  The previous token's
    current/next source-frame pair is therefore not a cut-minus-one/cut pair.
    Keep the content-bound scalar audit, but never construct a temporal window
    or qualify direction/phase until timestamps are explicitly bound to the
    proposed cut.
    """

    # Audit both source observations from the same sparse token.  In
    # particular, do not substitute the previous sparse token for a cut-minus-
    # one frame: its timestamp can be more than a video frame away.
    outgoing_sidecar = (
        frame.source_continuity[outgoing_source_index].temporal_pose_sidecar
        if frame.source_continuity
        and outgoing_source_index < len(frame.source_continuity)
        else None
    )
    incoming_sidecar = (
        frame.source_continuity[incoming_source_index].temporal_pose_sidecar
        if frame.source_continuity
        and incoming_source_index < len(frame.source_continuity)
        else None
    )
    sampling_contract_match = bool(
        outgoing_sidecar is not None
        and incoming_sidecar is not None
        and outgoing_sidecar.sampling_contract
        == incoming_sidecar.sampling_contract
    )
    outgoing_evidence = _temporal_sidecar_scalar_evidence(outgoing_sidecar)
    incoming_evidence = _temporal_sidecar_scalar_evidence(incoming_sidecar)
    failure_reasons = ["boundary_alignment_unproven"]
    if outgoing_sidecar is None:
        failure_reasons.append("temporal_outgoing_sidecar_missing")
    if incoming_sidecar is None:
        failure_reasons.append("temporal_incoming_sidecar_missing")
    if (
        outgoing_sidecar is not None
        and incoming_sidecar is not None
        and not sampling_contract_match
    ):
        failure_reasons.append("temporal_sampling_contract_mismatch")
    pair_content_sha256 = (
        _redacted_content_sha256(
            {
                "contract": TEMPORAL_POSE_SIDECAR_SCALAR_EVIDENCE_CONTRACT,
                "outgoing_content_sha256": outgoing_evidence[
                    "content_sha256"
                ],
                "incoming_content_sha256": incoming_evidence[
                    "content_sha256"
                ],
            }
        )
        if outgoing_sidecar is not None and incoming_sidecar is not None
        else None
    )
    return {
        "sampling_contract_match": sampling_contract_match,
        "outgoing": outgoing_evidence,
        "incoming": incoming_evidence,
        "pair_content_sha256": pair_content_sha256,
        "boundary_alignment_proven": False,
        "failure_reasons": tuple(failure_reasons),
    }


def _boundary_motion_amount_plus_continuity_evidence(
    *,
    frame: StageEditHeadFrame,
    outgoing_frame: StageEditHeadFrame | None = None,
    outgoing_source_index: int | None,
    incoming_source_index: int,
    derivative_family_ids: Sequence[int],
    continuity_score: float,
    config: StageEditSemiMarkovConfig,
) -> dict[str, Any]:
    """Audit an actual-family switch using exact source observations.

    The legacy pose-phase field remains a same-time 60-D cosine proxy only.
    Sparse temporal sidecars are content-bound for audit, but cannot prove
    direction or phase until their samples are timestamp-bound to this cut.
    ``continuity_score`` is the primary boundary scorer's value and is kept
    only for audit/backwards compatibility.  Motion eligibility always
    recomputes the handcrafted qualifier calibrated by the motion profile.
    """

    inactive = {
        "pose_motion_minimum": None,
        "pose_motion_delta": None,
        "camera_motion_delta": None,
        "minimum_sharpness": None,
        "continuity_score": None,
        "primary_boundary_continuity_score": None,
        "handcrafted_continuity_qualifier_score": None,
        "pose_phase_cosine": None,
        "pose_phase_evidence_observed": False,
        "pose_phase_threshold_bound": False,
        "evidence_complete": False,
        "accent_supported": False,
        "pose_phase_match_proven": False,
        "pose_phase_feature_dimension": None,
        "pose_phase_outgoing_observed": False,
        "pose_phase_incoming_observed": False,
        "pose_phase_trusted_producer_contract": None,
        "pose_phase_outgoing_content_sha256": None,
        "pose_phase_incoming_content_sha256": None,
        "pose_phase_pair_content_sha256": None,
        "temporal_pose_motion_thresholds_bound": False,
        "temporal_pose_motion_input_valid": False,
        "temporal_outgoing_boundary_motion_magnitude": None,
        "temporal_incoming_boundary_motion_magnitude": None,
        "temporal_endpoint_pose_cosine": None,
        "temporal_velocity_direction_cosine": None,
        "temporal_phase_compatibility": None,
        "temporal_confidence": 0.0,
        "temporal_classification": None,
        "temporal_static_pose_match": False,
        "temporal_dynamic_pose_phase_match": False,
        "temporal_failure_reasons": (),
        "temporal_sampling_contract_match": False,
        "temporal_outgoing_sampling_contract": None,
        "temporal_incoming_sampling_contract": None,
        "temporal_feature_dimension": None,
        "temporal_outgoing_sidecar_observed": False,
        "temporal_incoming_sidecar_observed": False,
        "temporal_outgoing_trusted_producer_contract": None,
        "temporal_incoming_trusted_producer_contract": None,
        "temporal_outgoing_content_sha256": None,
        "temporal_incoming_content_sha256": None,
        "temporal_pair_content_sha256": None,
        "pose_direction_or_velocity_proven": False,
    }
    if outgoing_source_index is None or not frame.source_continuity:
        return inactive
    if (
        derivative_family_ids[outgoing_source_index]
        == derivative_family_ids[incoming_source_index]
    ):
        return inactive
    outgoing = frame.source_continuity[outgoing_source_index]
    incoming = frame.source_continuity[incoming_source_index]
    primary_continuity_score = _finite(
        continuity_score,
        "primary_boundary_continuity_score",
    )
    if not -1.0 <= primary_continuity_score <= 0.0:
        raise StageEditDecodeInputError(
            "primary boundary continuity scorer violated its bounded [-1, 0] "
            "contract"
        )
    handcrafted_continuity_qualifier_score = (
        stage_edit_boundary_continuity_score(outgoing, incoming)
    )
    outgoing_pose = outgoing.pose_model_embedding
    incoming_pose = incoming.pose_model_embedding
    outgoing_pose_evidence = _same_time_pose_scalar_evidence(outgoing_pose)
    incoming_pose_evidence = _same_time_pose_scalar_evidence(incoming_pose)
    pose_pair_content_sha256 = (
        _redacted_content_sha256(
            {
                "contract": SAME_TIME_POSE_SCALAR_EVIDENCE_CONTRACT,
                "outgoing_content_sha256": outgoing_pose_evidence[
                    "content_sha256"
                ],
                "incoming_content_sha256": incoming_pose_evidence[
                    "content_sha256"
                ],
            }
        )
        if outgoing_pose is not None and incoming_pose is not None
        else None
    )
    pose_phase_cosine: float | None = None
    if outgoing_pose is not None and incoming_pose is not None:
        # Both tuples were already validated as exact 60-D bounded finite
        # observations by StageEditSourceContinuity.  A zero norm is observed
        # data but cannot define a cosine and therefore cannot prove phase.
        outgoing_norm = math.sqrt(sum(value * value for value in outgoing_pose))
        incoming_norm = math.sqrt(sum(value * value for value in incoming_pose))
        if outgoing_norm > _TIME_EPSILON and incoming_norm > _TIME_EPSILON:
            pose_phase_cosine = min(
                1.0,
                max(
                    -1.0,
                    sum(
                        left * right
                        for left, right in zip(
                            outgoing_pose, incoming_pose, strict=True
                        )
                    )
                    / (outgoing_norm * incoming_norm),
                ),
            )
    pose_values = (outgoing.pose_motion, incoming.pose_motion)
    pose_complete = all(value is not None for value in pose_values)
    pose_minimum = (
        min(float(value) for value in pose_values if value is not None)
        if pose_complete
        else None
    )
    pose_delta = (
        abs(float(pose_values[1]) - float(pose_values[0]))
        if pose_complete
        else None
    )
    camera_values = (
        outgoing.camera_motion_x,
        outgoing.camera_motion_y,
        incoming.camera_motion_x,
        incoming.camera_motion_y,
    )
    camera_delta = (
        math.hypot(
            float(incoming.camera_motion_x) - float(outgoing.camera_motion_x),
            float(incoming.camera_motion_y) - float(outgoing.camera_motion_y),
        )
        if all(value is not None for value in camera_values)
        else None
    )
    sharpness_values = (outgoing.sharpness, incoming.sharpness)
    sharpness_minimum = (
        min(float(value) for value in sharpness_values if value is not None)
        if all(value is not None for value in sharpness_values)
        else None
    )
    amount_threshold = config.highlight_motion_boundary_minimum_pose_motion
    continuity_threshold = (
        config.highlight_motion_boundary_minimum_continuity_score
    )
    sharpness_threshold = config.highlight_motion_boundary_minimum_sharpness
    pose_phase_threshold = (
        config.highlight_motion_boundary_minimum_pose_phase_cosine
    )
    policy_active = amount_threshold is not None and continuity_threshold is not None
    pose_phase_threshold_bound = pose_phase_threshold is not None
    pose_phase_match_proven = bool(
        pose_phase_threshold_bound
        and pose_phase_cosine is not None
        and pose_phase_cosine + _TIME_EPSILON
        >= float(pose_phase_threshold)
    )
    temporal_calibration = config.highlight_temporal_pose_motion_calibration
    # The previous StageEdit token is not proven adjacent to this cut. Preserve
    # the hash-bound sidecar audit while keeping all direction/phase evidence
    # neutral. A non-None calibration is rejected by config validation above.
    temporal_evidence = _temporal_pose_sidecar_pair_audit(
        frame=frame,
        outgoing_source_index=outgoing_source_index,
        incoming_source_index=incoming_source_index,
    )
    temporal_evidence_complete = False
    temporal_dynamic_match = False
    evidence_complete = bool(
        policy_active
        and pose_complete
        and (
            not pose_phase_threshold_bound
            or pose_phase_cosine is not None
        )
        and (
            sharpness_threshold is None or sharpness_minimum is not None
        )
        and (
            temporal_calibration is None or temporal_evidence_complete
        )
    )
    accent_supported = bool(
        evidence_complete
        and pose_minimum is not None
        and pose_minimum + _TIME_EPSILON >= float(amount_threshold)
        # The profile quantiles were measured with handcrafted continuity.
        # Recompute that exact qualifier even when a learned scorer owns the
        # primary boundary score. High motion alone can never pass.
        and handcrafted_continuity_qualifier_score + _TIME_EPSILON
        >= float(continuity_threshold)
        # Phase is optional only while its calibrated threshold is inactive.
        # Once bound, neither motion nor continuity may bypass the pose match.
        and (
            not pose_phase_threshold_bound
            or pose_phase_match_proven
        )
        and (
            sharpness_threshold is None
            or (
                sharpness_minimum is not None
                and sharpness_minimum + _TIME_EPSILON
                >= sharpness_threshold
            )
        )
        and (
            temporal_calibration is None or temporal_dynamic_match
        )
    )
    return {
        "pose_motion_minimum": pose_minimum,
        "pose_motion_delta": pose_delta,
        "camera_motion_delta": camera_delta,
        "minimum_sharpness": sharpness_minimum,
        # Keep the legacy key bound to the primary boundary score.  The
        # explicit qualifier key below is the only value used by the motion
        # threshold.
        "continuity_score": primary_continuity_score,
        "primary_boundary_continuity_score": primary_continuity_score,
        "handcrafted_continuity_qualifier_score": (
            handcrafted_continuity_qualifier_score
        ),
        "pose_phase_cosine": pose_phase_cosine,
        "pose_phase_evidence_observed": pose_phase_cosine is not None,
        "pose_phase_threshold_bound": pose_phase_threshold_bound,
        "evidence_complete": evidence_complete,
        "accent_supported": accent_supported,
        "pose_phase_match_proven": pose_phase_match_proven,
        "pose_phase_feature_dimension": (
            len(outgoing_pose)
            if outgoing_pose is not None and incoming_pose is not None
            else None
        ),
        "pose_phase_outgoing_observed": outgoing_pose is not None,
        "pose_phase_incoming_observed": incoming_pose is not None,
        "pose_phase_trusted_producer_contract": (
            SAME_TIME_POSE_SCALAR_EVIDENCE_CONTRACT
            if outgoing_pose is not None or incoming_pose is not None
            else None
        ),
        "pose_phase_outgoing_content_sha256": outgoing_pose_evidence[
            "content_sha256"
        ],
        "pose_phase_incoming_content_sha256": incoming_pose_evidence[
            "content_sha256"
        ],
        "pose_phase_pair_content_sha256": pose_pair_content_sha256,
        "temporal_pose_motion_thresholds_bound": False,
        "temporal_pose_motion_input_valid": temporal_evidence_complete,
        "temporal_outgoing_boundary_motion_magnitude": (
            None
        ),
        "temporal_incoming_boundary_motion_magnitude": (
            None
        ),
        "temporal_endpoint_pose_cosine": (
            None
        ),
        "temporal_velocity_direction_cosine": (
            None
        ),
        "temporal_phase_compatibility": (
            None
        ),
        "temporal_confidence": (
            0.0
        ),
        "temporal_classification": (
            "boundary_alignment_unproven"
            if temporal_evidence["outgoing"]["observed"]
            and temporal_evidence["incoming"]["observed"]
            else "temporal_pose_sidecar_missing"
        ),
        "temporal_static_pose_match": bool(
            False
        ),
        "temporal_dynamic_pose_phase_match": temporal_dynamic_match,
        "temporal_failure_reasons": (
            temporal_evidence["failure_reasons"]
        ),
        "temporal_sampling_contract_match": (
            temporal_evidence["sampling_contract_match"]
        ),
        "temporal_outgoing_sampling_contract": temporal_evidence[
            "outgoing"
        ]["sampling_contract"],
        "temporal_incoming_sampling_contract": temporal_evidence[
            "incoming"
        ]["sampling_contract"],
        "temporal_feature_dimension": (
            temporal_evidence["outgoing"]["feature_dimension"]
            if temporal_evidence["outgoing"]["observed"]
            and temporal_evidence["incoming"]["observed"]
            else None
        ),
        "temporal_outgoing_sidecar_observed": temporal_evidence[
            "outgoing"
        ]["observed"],
        "temporal_incoming_sidecar_observed": temporal_evidence[
            "incoming"
        ]["observed"],
        "temporal_outgoing_trusted_producer_contract": temporal_evidence[
            "outgoing"
        ]["trusted_producer_contract"],
        "temporal_incoming_trusted_producer_contract": temporal_evidence[
            "incoming"
        ]["trusted_producer_contract"],
        "temporal_outgoing_content_sha256": temporal_evidence[
            "outgoing"
        ]["content_sha256"],
        "temporal_incoming_content_sha256": temporal_evidence[
            "incoming"
        ]["content_sha256"],
        "temporal_pair_content_sha256": temporal_evidence[
            "pair_content_sha256"
        ],
        "pose_direction_or_velocity_proven": False,
    }


def _source_span_rejection(
    *,
    source: StageEditSourceMetadata,
    frames: Sequence[StageEditHeadFrame],
    start_index: int,
    end_index: int,
    master_end_seconds: float,
    margin: float,
) -> str | None:
    for frame in frames[start_index:end_index]:
        if not frame.effective_source_mask[source.source_index]:
            return "source_unavailable_or_hard_gated_inside_span"
    master_start = frames[start_index].master_time_seconds
    source_start = source.source_start_at(master_start)
    source_end = source.source_start_at(master_end_seconds)
    if source_start < -_TIME_EPSILON or source_end > source.source_duration_seconds + _TIME_EPSILON:
        return "source_clock_outside_available_media"
    for cut_time in source.internal_cut_times_seconds:
        if source_start + margin + _TIME_EPSILON < cut_time < source_end - margin - _TIME_EPSILON:
            return "source_internal_cut_inside_shot"
    return None


def _input_digest(
    *,
    sources: Sequence[Path],
    source_metadata: Sequence[StageEditSourceMetadata],
    frames: Sequence[StageEditHeadFrame],
    section_plan: SectionPlan,
    constraints: Sequence[StageEditSectionConstraint],
    config: Mapping[str, Any],
    schema: StageEditTokenSchema,
    source_derivative_metadata: SourceDerivativeFamilyMetadata | None,
    boundary_continuity_score_provenance: Mapping[str, Any],
) -> str:
    payload = {
        "sources": [str(Path(value)) for value in sources],
        "source_metadata": [asdict(value) for value in source_metadata],
        "frames": [value.to_dict() for value in frames],
        "section_plan": section_plan.to_dict(),
        "constraints": [asdict(value) for value in constraints],
        "config": dict(config),
        "boundary_continuity_score_provenance": dict(
            boundary_continuity_score_provenance
        ),
        "token_schema_fingerprint": schema.fingerprint,
        "source_derivative_metadata": (
            source_derivative_metadata.to_dict()
            if source_derivative_metadata is not None
            else None
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _family_occupancy(
    source_occupancy: Sequence[float],
    family_ids: Sequence[int],
) -> tuple[float, ...]:
    if len(source_occupancy) != len(family_ids):
        raise StageEditDecodeInputError(
            "source occupancy and derivative-family metadata do not align"
        )
    family_count = len(set(family_ids))
    totals = [0.0] * family_count
    for source_index, value in enumerate(source_occupancy):
        totals[family_ids[source_index]] += float(value)
    return tuple(totals)


def _family_occupancy_caps(
    source_metadata: Sequence[StageEditSourceMetadata],
    family_ids: Sequence[int],
    config: StageEditSemiMarkovConfig,
) -> tuple[float, ...]:
    family_count = len(set(family_ids))
    caps = [config.maximum_source_occupancy_fraction] * family_count
    for source_index, metadata in enumerate(source_metadata):
        family = family_ids[source_index]
        caps[family] = min(caps[family], metadata.maximum_occupancy_fraction)
    return tuple(caps)


def _timeline_output_path(output: Path, rank: int) -> Path:
    suffix = output.suffix or ".mp4"
    stem = output.stem if output.suffix else output.name
    return output.with_name(f"{stem}.stageedit-{rank:02d}{suffix}")


def _build_stage_edit(
    *,
    rank: int,
    state: _BeamState,
    sources: tuple[Path, ...],
    source_metadata: tuple[StageEditSourceMetadata, ...],
    section_plan: SectionPlan,
    aspect_ratio: AspectRatio,
    width: int,
    height: int,
    fps: float,
    reference_source: int,
    output: Path,
    checkpoint_path: str,
    source_derivative_metadata: SourceDerivativeFamilyMetadata | None,
    config: StageEditSemiMarkovConfig,
    boundary_continuity_score_provenance: Mapping[str, Any],
) -> StageEdit:
    transition_authority = StageEditTransitionAuthority(
        mode=config.transition_authority
    )
    shots: list[StageShot] = []
    for decision in state.decisions:
        metadata = source_metadata[decision.source_index]
        duration = decision.duration_seconds
        midpoint = decision.master_start_seconds + duration / 2.0
        section_index = section_plan.directive_at(midpoint).section_index
        segment = RenderSegment(
            source_index=decision.source_index,
            source_start=max(0.0, metadata.source_start_at(decision.master_start_seconds)),
            duration=duration,
            master_start=decision.master_start_seconds - section_plan.start_seconds,
            editorial_transform=decision.editorial_transform,
        )
        shots.append(
            StageShot(
                shot_index=decision.shot_index,
                section_index=section_index,
                segment=segment,
                decision_id=(
                    f"stageedit-semimarkov-r{rank:02d}-s{decision.shot_index:04d}"
                ),
                decision_confidence=decision.decision_confidence,
                section_role=decision.section_role,
                section_progress=decision.section_progress,
                phrase_progress=decision.phrase_progress,
            )
        )
    boundaries = tuple(
        StageBoundary(
            boundary_index=index,
            transition=RenderTransition(
                kind=TransitionKind.HARD_CUT,
                frame_count=0,
                reason=(
                    "stage_edit_semimarkov_requested_"
                    f"{state.decisions[index + 1].transition_label}"
                ),
                tier="C",
            ),
        )
        for index in range(len(shots) - 1)
    )
    return StageEdit(
        sources=sources,
        reference_source=reference_source,
        aspect_ratio=aspect_ratio,
        width=width,
        height=height,
        fps=fps,
        master_audio_start=section_plan.start_seconds,
        section_plan=section_plan,
        shots=tuple(shots),
        boundaries=boundaries,
        output=_timeline_output_path(output, rank),
        candidate_history=(
            {
                "decoder": DECODER_CONTRACT,
                "timeline_rank": rank,
                "checkpoint_path": checkpoint_path,
                "score": state.score,
                "decision_count": len(state.decisions),
                "source_derivative_diversity": (
                    source_derivative_metadata.to_dict()
                    if source_derivative_metadata is not None
                    else {
                        "production_ready": False,
                        "family_ids": list(range(len(sources))),
                        "status": "decoder_unit_identity_default",
                    }
                ),
                "transition_translation": (
                    "model metadata is duration-neutral hard_cut; renderer-owned "
                    "group_transition alone selects hard_cut or evidence-gated "
                    "face_dissolve"
                    if transition_authority.renderer_owned
                    else "model transition labels retained in audit; renderer IR "
                    "uses duration-neutral hard cuts"
                ),
                "transition_authority": transition_authority.to_dict(),
                "boundary_continuity_score_provenance": dict(
                    boundary_continuity_score_provenance
                ),
            },
        ),
    )


def _validate_decode_inputs(
    *,
    sources: tuple[Path, ...],
    source_metadata: tuple[StageEditSourceMetadata, ...],
    frames: tuple[StageEditHeadFrame, ...],
    section_plan: SectionPlan,
    schema: StageEditTokenSchema,
    config: StageEditSemiMarkovConfig,
    reference_source: int,
    aspect_ratio: AspectRatio,
    width: int,
    height: int,
    fps: float,
    source_derivative_metadata: SourceDerivativeFamilyMetadata | None,
) -> None:
    source_count = len(sources)
    if source_count < 2:
        raise StageEditDecodeInputError("semi-Markov decoding needs at least two sources")
    if len(source_metadata) != source_count:
        raise StageEditDecodeInputError("source metadata count does not match sources")
    if tuple(value.source_index for value in source_metadata) != tuple(range(source_count)):
        raise StageEditDecodeInputError("source metadata indices must be contiguous")
    source_ids = tuple(value.source_id for value in source_metadata)
    if len(set(source_ids)) != len(source_ids):
        raise StageEditDecodeInputError("source ids must be unique")
    if config.minimum_unique_sources > source_count:
        raise StageEditDecodeInputError("minimum_unique_sources exceeds source count")
    if source_derivative_metadata is not None:
        if not isinstance(source_derivative_metadata, SourceDerivativeFamilyMetadata):
            raise StageEditDecodeInputError(
                "source derivative metadata has an invalid type"
            )
        if source_derivative_metadata.source_count != source_count:
            raise StageEditDecodeInputError(
                "source derivative metadata count does not match sources"
            )
        if not source_derivative_metadata.production_ready:
            raise StageEditDecodeInputError(
                "source derivative metadata is uncertain or clock-incomplete"
            )
        if source_derivative_metadata.family_count < config.minimum_unique_sources:
            raise StageEditDecodeInputError(
                "minimum_unique_sources exceeds verified derivative-family count"
            )
    if (
        isinstance(reference_source, bool)
        or not isinstance(reference_source, int)
        or not 0 <= reference_source < source_count
    ):
        raise StageEditDecodeInputError("reference_source is outside the source list")
    if not frames:
        raise StageEditDecodeInputError("candidate head frames cannot be empty")
    dynamic_crop_modes = tuple(
        "v2"
        if frame.dynamic_crop_v2 is not None
        else "v3"
        if frame.dynamic_crop_v3_by_source
        else "legacy"
        for frame in frames
    )
    if len(set(dynamic_crop_modes)) != 1:
        raise StageEditDecodeInputError(
            "candidate head frames cannot mix legacy, Dynamic Crop v2, and "
            "Dynamic Crop v3 contracts"
        )
    if abs(frames[0].master_time_seconds - section_plan.start_seconds) > _TIME_EPSILON:
        raise StageEditDecodeInputError(
            "first candidate time must equal the section-plan start"
        )
    times = tuple(value.master_time_seconds for value in frames)
    if any(right <= left for left, right in zip(times, times[1:])):
        raise StageEditDecodeInputError("candidate times must be strictly increasing")
    if frames[-1].master_time_seconds >= section_plan.end_seconds - _TIME_EPSILON:
        raise StageEditDecodeInputError(
            "candidate frames must be before the terminal section-plan time"
        )
    required_boundaries = tuple(
        section.end_seconds for section in section_plan.sections[:-1]
    )
    for boundary in required_boundaries:
        if not any(abs(value - boundary) <= _TIME_EPSILON for value in times):
            raise StageEditDecodeInputError(
                f"section boundary {boundary:.6f} is missing from candidate times"
            )
    for index, frame in enumerate(frames):
        if len(frame.source_logits) != source_count:
            raise StageEditDecodeInputError(
                f"frame {index} source logit count does not match sources"
            )
        if len(frame.available_source_mask) != source_count or len(
            frame.hard_source_gate_mask
        ) != source_count:
            raise StageEditDecodeInputError(
                f"frame {index} source mask count does not match sources"
            )
        if frame.source_shot_sizes and len(frame.source_shot_sizes) != source_count:
            raise StageEditDecodeInputError(
                f"frame {index} shot-size count does not match sources"
            )
        if frame.source_continuity and len(frame.source_continuity) != source_count:
            raise StageEditDecodeInputError(
                f"frame {index} continuity count does not match sources"
            )
        if (
            config.continuity_authority
            == PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
            and len(frame.source_continuity) != source_count
        ):
            raise StageEditDecodeInputError(
                f"frame {index} learned continuity authority requires exact "
                "canonical source observations"
            )
        if frame.dynamic_crop_v3_by_source:
            if len(frame.dynamic_crop_v3_by_source) != source_count:
                raise StageEditDecodeInputError(
                    f"frame {index} Dynamic Crop v3 count does not match sources"
                )
            for source_index, prediction in enumerate(
                frame.dynamic_crop_v3_by_source
            ):
                if prediction is None:
                    continue
                if (
                    prediction.source_index != source_index
                    or prediction.source_id != source_ids[source_index]
                ):
                    raise StageEditDecodeInputError(
                        f"frame {index} Dynamic Crop v3 source identity mismatch"
                    )
        if len(frame.duration_logits) != schema.duration_bucket_count:
            raise StageEditDecodeInputError(
                f"frame {index} duration logit count does not match schema"
            )
        if len(frame.transition_logits) != len(schema.transition_labels):
            raise StageEditDecodeInputError(
                f"frame {index} transition logit count does not match schema"
            )
        if not any(frame.effective_source_mask):
            raise StageEditDecodeInputError(
                f"frame {index} has no source surviving availability and hard gates"
            )
    if not isinstance(aspect_ratio, AspectRatio):
        try:
            aspect_ratio = AspectRatio(aspect_ratio)
        except ValueError as exc:
            raise StageEditDecodeInputError("aspect_ratio is invalid") from exc
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or isinstance(height, bool)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
        or width % 2
        or height % 2
    ):
        raise StageEditDecodeInputError("output dimensions must be positive even integers")
    expected_ratio = 16.0 / 9.0 if aspect_ratio is AspectRatio.LANDSCAPE else 9.0 / 16.0
    if abs(width / height - expected_ratio) > 0.002:
        raise StageEditDecodeInputError("output dimensions do not match aspect_ratio")
    if _finite(fps, "fps") <= 0.0:
        raise StageEditDecodeInputError("fps must be positive")


def decode_stage_edit_semimarkov_beam(
    *,
    checkpoint_directory: Path,
    sources: Sequence[Path],
    source_metadata: Sequence[StageEditSourceMetadata],
    head_frames: Sequence[StageEditHeadFrame],
    section_plan: SectionPlan,
    output: Path,
    section_constraints: Sequence[StageEditSectionConstraint] = (),
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
    config: StageEditSemiMarkovConfig = StageEditSemiMarkovConfig(),
    reference_source: int = 0,
    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE,
    width: int = 1920,
    height: int = 1080,
    fps: float = 29.97,
    source_derivative_metadata: SourceDerivativeFamilyMetadata | None = None,
    source_pair_scorer: StageEditSourcePairScorer | None = None,
) -> StageEditSemiMarkovResult:
    """Decode StageEdit heads into ranked, renderer-valid StageEdit timelines.

    The readiness check intentionally occurs before input validation/search so
    this function cannot be repurposed as an implicit random-checkpoint
    decoder.  It performs no rendering, media probing, or pipeline mutation.
    """

    if checkpoint_directory is None:
        raise StageEditCheckpointNotReadyError(
            "StageEdit semi-Markov decoding requires a production-ready checkpoint"
        )
    readiness = stage_edit_readiness_report(Path(checkpoint_directory))
    if not (
        readiness.architecture_available
        and readiness.learned_inference_ready
        and readiness.status == "production_checkpoint_ready"
        and readiness.checkpoint_path
        and readiness.training_steps > 0
    ):
        raise StageEditCheckpointNotReadyError(
            "StageEdit semi-Markov decoding requires a production-ready checkpoint: "
            f"{readiness.status}: {readiness.reason}"
        )

    boundary_continuity_score_provenance = _continuity_score_provenance(
        config, source_pair_scorer
    )
    audit_config = _audit_config_payload(
        config, boundary_continuity_score_provenance
    )

    normalized_sources = tuple(Path(value) for value in sources)
    normalized_metadata = tuple(source_metadata)
    frames = tuple(head_frames)
    constraints = tuple(section_constraints)
    if not isinstance(aspect_ratio, AspectRatio):
        try:
            aspect_ratio = AspectRatio(aspect_ratio)
        except ValueError as exc:
            raise StageEditDecodeInputError("aspect_ratio is invalid") from exc
    _validate_decode_inputs(
        sources=normalized_sources,
        source_metadata=normalized_metadata,
        frames=frames,
        section_plan=section_plan,
        schema=schema,
        config=config,
        reference_source=reference_source,
        aspect_ratio=aspect_ratio,
        width=width,
        height=height,
        fps=fps,
        source_derivative_metadata=source_derivative_metadata,
    )
    resolved = _resolved_constraints(
        section_plan, constraints, schema, config
    )

    source_count = len(normalized_sources)
    derivative_family_ids = (
        source_derivative_metadata.family_ids
        if source_derivative_metadata is not None
        else tuple(range(source_count))
    )
    derivative_family_count = len(set(derivative_family_ids))
    derivative_family_caps = _family_occupancy_caps(
        normalized_metadata,
        derivative_family_ids,
        config,
    )
    montage_support_by_section = _section_candidate_shot_support(
        frames=frames,
        source_metadata=normalized_metadata,
        section_plan=section_plan,
        derivative_family_ids=derivative_family_ids,
        config=config,
    )
    section_count = len(section_plan.sections)
    master_duration = section_plan.end_seconds - section_plan.start_seconds
    terminal_index = len(frames)
    all_times = tuple(value.master_time_seconds for value in frames) + (
        section_plan.end_seconds,
    )
    source_log_probabilities = tuple(
        _masked_log_softmax(frame.source_logits, frame.effective_source_mask)
        for frame in frames
    )
    duration_log_probabilities = tuple(
        _log_softmax(frame.duration_logits) for frame in frames
    )
    transition_authority = StageEditTransitionAuthority(
        mode=config.transition_authority
    )
    if transition_authority.renderer_owned:
        transition_log_probabilities = tuple(
            tuple(
                0.0 if label == "hard_cut" else -math.inf
                for label in schema.transition_labels
            )
            for _ in frames
        )
    else:
        transition_log_probabilities = tuple(
            _masked_log_softmax(
                frame.transition_logits,
                tuple(
                    label
                    in resolved[
                        _section_index_at_start(
                            section_plan, frame.master_time_seconds
                        )
                    ].allowed_transition_labels
                    for label in schema.transition_labels
                ),
            )
            for frame in frames
        )
    # Legacy four-value crops remain available for old checkpoints.  Dynamic
    # Crop predictions are intentionally *not* projected here: normalized
    # trajectory times can only be interpreted after a concrete duration
    # hypothesis exists inside the semi-Markov expansion.  V3 additionally
    # keeps the canonical source axis until a source hypothesis is selected.
    crop_values = tuple(
        None
        if frame.dynamic_crop_v2 is not None or frame.dynamic_crop_v3_by_source
        else _crop_transform(frame.crop)
        for frame in frames
    )

    empty_state = _BeamState(
        current_index=0,
        current_source=None,
        current_section_index=None,
        section_local_source_switch_boundaries=0,
        section_local_wide_involved_source_switch_boundaries=0,
        recent_sources=(),
        recent_shot_sizes=(),
        recent_shot_durations=(),
        recent_cut_times=(),
        source_occupancy_seconds=(0.0,) * source_count,
        section_switch_counts=(0,) * section_count,
        decisions=(),
        score=0.0,
        component_totals=(),
    )
    beams: list[list[_BeamState]] = [[] for _ in range(terminal_index + 1)]
    beams[0].append(empty_state)
    rejection_counts: dict[str, int] = {}
    endpoint_records: list[dict[str, Any]] = [
        {
            "candidate_index": index,
            "master_time_seconds": all_times[index],
            "incoming_state_count": 0,
            "span_candidate_count": 0,
            "source_candidate_count": 0,
            "transition_hypothesis_count": 0,
            "accepted_hypothesis_count": 0,
            "beam_pruned_state_count": 0,
        }
        for index in range(terminal_index + 1)
    ]
    transition_hypothesis_count = 0
    accepted_hypothesis_count = 0
    pruned_count = 0

    def reject(name: str) -> None:
        rejection_counts[name] = rejection_counts.get(name, 0) + 1

    for start_index in range(terminal_index):
        states = sorted(beams[start_index], key=_state_sort_key)
        if len(states) > config.beam_width:
            states, removed, duplicates = _prune_states(
                states, config.beam_width
            )
            if duplicates:
                rejection_counts["duplicate_partial_timeline"] = (
                    rejection_counts.get("duplicate_partial_timeline", 0)
                    + duplicates
                )
            pruned_count += removed
            endpoint_records[start_index]["beam_pruned_state_count"] += removed
        beams[start_index] = states
        endpoint_records[start_index]["incoming_state_count"] = len(states)
        if not states:
            continue
        frame = frames[start_index]
        start_time = frame.master_time_seconds
        section_index = _section_index_at_start(section_plan, start_time)
        directive = section_plan.sections[section_index]
        constraint = resolved[section_index]
        section_remaining = directive.end_seconds - start_time
        maximum_duration = min(constraint.maximum_shot_seconds, section_remaining)
        minimum_duration = constraint.minimum_shot_seconds

        for state in states:
            prior_section_decisions = tuple(
                item
                for item in state.decisions
                if _section_index_at_start(
                    section_plan,
                    item.master_start_seconds,
                )
                == section_index
            )
            prior_section_known_duration = sum(
                item.duration_seconds
                for item in prior_section_decisions
                if item.shot_size != "unknown"
            )
            prior_section_closeup_duration = sum(
                item.duration_seconds
                for item in prior_section_decisions
                if item.shot_size in _CLOSEUP_SHOT_SIZES
            )
            prior_section_nonwide_duration = sum(
                item.duration_seconds
                for item in prior_section_decisions
                if item.shot_size in _NONWIDE_SHOT_SIZES
            )
            prior_source_switch_boundaries = (
                state.section_local_source_switch_boundaries
                if state.current_section_index == section_index
                else 0
            )
            prior_wide_involved_source_switch_boundaries = (
                state.section_local_wide_involved_source_switch_boundaries
                if state.current_section_index == section_index
                else 0
            )
            for end_index in range(start_index + 1, terminal_index + 1):
                endpoint_records[start_index]["span_candidate_count"] += 1
                end_time = all_times[end_index]
                duration = end_time - start_time
                if duration < minimum_duration - _TIME_EPSILON:
                    reject("shot_below_minimum_duration")
                    continue
                if duration > maximum_duration + _TIME_EPSILON:
                    reject("shot_above_maximum_or_crosses_section")
                    break

                skipped_cut_score = sum(
                    _log_sigmoid(-frames[index].cut_logit)
                    for index in range(start_index + 1, end_index)
                )
                boundary_cut_score = (
                    0.0 if not state.decisions else _log_sigmoid(frame.cut_logit)
                )
                cut_raw = boundary_cut_score + skipped_cut_score
                duration_bucket = schema.duration_bucket(duration)
                duration_raw = duration_log_probabilities[start_index][duration_bucket]
                cadence_target = directive.target_source_switches_per_minute
                if directive.role is SectionRole.CHORUS_LIKE:
                    cadence_target = max(
                        cadence_target,
                        config.highlight_preferred_source_switches_per_minute,
                    )
                ideal_duration = min(
                    maximum_duration,
                    max(
                        minimum_duration,
                        60.0 / max(1e-9, cadence_target),
                    ),
                )
                cadence_raw = -abs(duration - ideal_duration) / max(
                    ideal_duration, minimum_duration
                )

                for source_index, metadata in enumerate(normalized_metadata):
                    endpoint_records[start_index]["source_candidate_count"] += 1
                    if not frame.effective_source_mask[source_index]:
                        reject("source_unavailable_or_hard_gated_at_start")
                        continue
                    selected_continuity = (
                        frame.source_continuity[source_index]
                        if frame.source_continuity
                        else StageEditSourceContinuity()
                    )
                    span_rejection = _source_span_rejection(
                        source=metadata,
                        frames=frames,
                        start_index=start_index,
                        end_index=end_index,
                        master_end_seconds=end_time,
                        margin=config.internal_cut_exclusion_margin_seconds,
                    )
                    if span_rejection is not None:
                        reject(span_rejection)
                        continue
                    candidate_family = derivative_family_ids[source_index]
                    current_family = (
                        derivative_family_ids[state.current_source]
                        if state.current_source is not None
                        else None
                    )
                    if (
                        current_family == candidate_family
                        and not config.allow_consecutive_same_source
                    ):
                        reject("consecutive_same_derivative_family")
                        continue
                    if (
                        config.enforce_aba_exclusion
                        and len(state.recent_sources) >= 2
                        and derivative_family_ids[state.recent_sources[-2]]
                        == candidate_family
                        and derivative_family_ids[state.recent_sources[-1]]
                        != candidate_family
                    ):
                        reject("aba_derivative_family_return")
                        # Compatibility alias: the v1-v3 counter name remains
                        # readable, but its v4 semantics are family-aware.
                        reject("aba_source_return")
                        continue

                    shot_size = _shot_size(frame, metadata)
                    if shot_size != "unknown":
                        same_size_run = 0
                        for prior in reversed(state.recent_shot_sizes):
                            if prior != shot_size:
                                break
                            same_size_run += 1
                        if same_size_run >= config.maximum_same_shot_size_run:
                            reject("maximum_same_shot_size_run")
                            continue

                    prospective_occupancy = list(state.source_occupancy_seconds)
                    prospective_occupancy[source_index] += duration
                    prospective_family_occupancy = _family_occupancy(
                        prospective_occupancy,
                        derivative_family_ids,
                    )
                    if prospective_family_occupancy[candidate_family] > (
                        derivative_family_caps[candidate_family]
                        * master_duration
                        + _TIME_EPSILON
                    ):
                        reject("derivative_family_occupancy_cap")
                        reject("source_occupancy_cap")
                        continue
                    capacity = sum(
                        max(0.0, cap * master_duration - value)
                        for cap, value in zip(
                            # All derivatives of one camera share one cap.
                            derivative_family_caps,
                            prospective_family_occupancy,
                            strict=True,
                        )
                    )
                    remaining_duration = section_plan.end_seconds - end_time
                    if capacity + _TIME_EPSILON < remaining_duration:
                        reject("remaining_derivative_family_occupancy_capacity")
                        reject("remaining_source_occupancy_capacity")
                        continue

                    section_counts = list(state.section_switch_counts)
                    is_boundary = bool(state.decisions)
                    if is_boundary:
                        section_counts[section_index] += 1
                        section_limit = _maximum_switch_count(
                            constraint.maximum_switches_per_minute,
                            directive.duration_seconds,
                        )
                        if section_counts[section_index] > section_limit:
                            reject("section_cut_density")
                            continue
                        global_limit = _maximum_switch_count(
                            config.maximum_global_switches_per_minute,
                            master_duration,
                        )
                        if len(state.decisions) > global_limit:
                            reject("global_cut_density")
                            continue
                        if (
                            start_time - section_plan.start_seconds
                            >= config.cut_density_window_seconds
                        ):
                            recent_count = sum(
                                value
                                >= start_time - config.cut_density_window_seconds
                                for value in state.recent_cut_times
                            ) + 1
                            recent_limit = _maximum_switch_count(
                                config.maximum_global_switches_per_minute,
                                config.cut_density_window_seconds,
                            )
                            if recent_count > recent_limit:
                                reject("rolling_cut_density")
                                continue

                    used_families = {
                        derivative_family_ids[index]
                        for index in state.recent_sources
                    }
                    used_families.update(
                        derivative_family_ids[item.source_index]
                        for item in state.decisions
                    )
                    used_families.add(candidate_family)
                    if end_index == terminal_index:
                        if len(used_families) < config.minimum_unique_sources:
                            reject("final_derivative_family_diversity")
                            continue
                    else:
                        possible_families = set(used_families)
                        for future_frame in frames[end_index:]:
                            possible_families.update(
                                derivative_family_ids[index]
                                for index, allowed in enumerate(
                                    future_frame.effective_source_mask
                                )
                                if allowed
                            )
                        if len(possible_families) < config.minimum_unique_sources:
                            reject("unreachable_derivative_family_diversity")
                            continue

                    transition_indices = (
                        (None,)
                        if not state.decisions
                        else (
                            (schema.transition_labels.index("hard_cut"),)
                            if transition_authority.renderer_owned
                            else tuple(range(len(schema.transition_labels)))
                        )
                    )
                    for transition_index in transition_indices:
                        transition_hypothesis_count += 1
                        endpoint_records[start_index]["transition_hypothesis_count"] += 1
                        if transition_index is None:
                            transition_label = "initial"
                            transition_raw = 0.0
                        else:
                            transition_label = schema.transition_labels[transition_index]
                            if transition_label not in constraint.allowed_transition_labels:
                                reject("transition_disallowed_by_section")
                                continue
                            transition_raw = transition_log_probabilities[start_index][
                                transition_index
                            ]

                        dynamic_crop_decode: DynamicCropTrackDecodeV2 | None = None
                        dynamic_crop_decode_v3: DynamicCropTrackDecodeV3 | None = None
                        editorial_transform: EditorialTransformTrack | None
                        if frame.dynamic_crop_v2 is not None:
                            dynamic_crop_decode = frame.dynamic_crop_v2.decode(duration)
                            requested_crop = frame.dynamic_crop_v2.values[0]
                            crop_contract = (
                                DYNAMIC_CROP_V2_SEMIMARKOV_ADAPTER_CONTRACT
                            )
                            editorial_transform = dynamic_crop_decode.track
                            if dynamic_crop_decode.eligible:
                                assert editorial_transform is not None
                                first_keyframe = editorial_transform.keyframes[0]
                                applied_crop = (
                                    first_keyframe.zoom,
                                    first_keyframe.center_x,
                                    first_keyframe.center_y,
                                    first_keyframe.roll_degrees,
                                )
                                crop_corrected = bool(
                                    dynamic_crop_decode.minimum_safe_zoom_projection_count
                                )
                                crop_correction = (
                                    dynamic_crop_decode.minimum_safe_zoom_projection_count
                                    / max(
                                        1,
                                        dynamic_crop_decode.requested_keyframe_count,
                                    )
                                )
                            else:
                                # A rejected dynamic trajectory means exactly
                                # "no transform" for this cut.  Never
                                # substitute the legacy crop head, a centered
                                # zoom, or a synthesized t=0 keyframe.
                                applied_crop = (1.0, 0.5, 0.5, 0.0)
                                crop_corrected = False
                                crop_correction = 1.0
                        elif frame.dynamic_crop_v3_by_source:
                            # The source axis is resolved only after this beam
                            # hypothesis selects its canonical source.  Never
                            # borrow another source's trajectory, even when its
                            # values happen to look more renderer-friendly.
                            prediction_v3 = frame.dynamic_crop_v3_by_source[
                                source_index
                            ]
                            if prediction_v3 is None:
                                raise StageEditDecodeInputError(
                                    "selected Dynamic Crop v3 source has no prediction"
                                )
                            if (
                                prediction_v3.source_index != source_index
                                or prediction_v3.source_id != metadata.source_id
                            ):
                                raise StageEditDecodeInputError(
                                    "selected Dynamic Crop v3 prediction/source mismatch"
                                )
                            dynamic_crop_decode_v3 = prediction_v3.decode(duration)
                            requested_crop = prediction_v3.values[0]
                            crop_contract = (
                                DYNAMIC_CROP_V3_SEMIMARKOV_ADAPTER_CONTRACT
                            )
                            editorial_transform = dynamic_crop_decode_v3.track
                            if dynamic_crop_decode_v3.eligible:
                                assert editorial_transform is not None
                                first_keyframe = editorial_transform.keyframes[0]
                                applied_crop = (
                                    first_keyframe.zoom,
                                    first_keyframe.center_x,
                                    first_keyframe.center_y,
                                    first_keyframe.roll_degrees,
                                )
                                crop_corrected = bool(
                                    dynamic_crop_decode_v3.minimum_safe_zoom_projection_count
                                )
                                crop_correction = (
                                    dynamic_crop_decode_v3.minimum_safe_zoom_projection_count
                                    / max(
                                        1,
                                        dynamic_crop_decode_v3.predicted_keyframe_count,
                                    )
                                )
                            else:
                                applied_crop = (1.0, 0.5, 0.5, 0.0)
                                crop_corrected = False
                                crop_correction = 1.0
                        else:
                            legacy_crop = crop_values[start_index]
                            assert legacy_crop is not None
                            applied_crop, crop_corrected, crop_correction = legacy_crop
                            requested_crop = frame.crop
                            crop_contract = "legacy_static_crop_head_v1"
                            editorial_transform = _editorial_track(applied_crop)
                        source_raw = source_log_probabilities[start_index][source_index]
                        boundary_continuity_raw = _boundary_continuity_raw(
                            frame,
                            state.current_source,
                            source_index,
                            derivative_family_ids,
                            continuity_authority=config.continuity_authority,
                            source_pair_scorer=source_pair_scorer,
                        )
                        boundary_motion_evidence = (
                            _boundary_motion_amount_plus_continuity_evidence(
                                frame=frame,
                                outgoing_frame=(
                                    frames[start_index - 1]
                                    if state.current_source is not None
                                    and start_index > 0
                                    else None
                                ),
                                outgoing_source_index=state.current_source,
                                incoming_source_index=source_index,
                                derivative_family_ids=derivative_family_ids,
                                continuity_score=boundary_continuity_raw,
                                config=config,
                            )
                        )
                        quality_raw = metadata.quality_log_prior
                        novelty_raw = (
                            1.0
                            if candidate_family
                            not in {
                                derivative_family_ids[item.source_index]
                                for item in state.decisions
                            }
                            else -0.25
                        )
                        if shot_size != "unknown" and shot_size not in state.recent_shot_sizes:
                            novelty_raw += 0.35
                        used_occupancy = [
                            value / master_duration
                            for value in prospective_family_occupancy
                        ]
                        occupancy_mean = (
                            sum(used_occupancy) / derivative_family_count
                        )
                        occupancy_raw = -sum(
                            abs(value - occupancy_mean) for value in used_occupancy
                        ) / derivative_family_count
                        previous_in_section = (
                            state.decisions[-1]
                            if state.decisions
                            and _section_index_at_start(
                                section_plan,
                                state.decisions[-1].master_start_seconds,
                            )
                            == section_index
                            else None
                        )
                        derivative_family_source_switched = bool(
                            previous_in_section is not None
                            and previous_in_section.derivative_family_id
                            != candidate_family
                        )
                        wide_involved_source_switch = bool(
                            derivative_family_source_switched
                            and (
                                previous_in_section.shot_size
                                in _WIDE_SHOT_SIZES
                                or shot_size in _WIDE_SHOT_SIZES
                            )
                        )
                        montage_grammar_raw = _montage_grammar_score_raw(
                            directive=directive,
                            shot_size=shot_size,
                            start_seconds=start_time,
                            duration_seconds=duration,
                            previous_shot_size=(
                                previous_in_section.shot_size
                                if previous_in_section is not None
                                else None
                            ),
                            previous_duration_seconds=(
                                previous_in_section.duration_seconds
                                if previous_in_section is not None
                                else None
                            ),
                            derivative_family_source_switched=(
                                derivative_family_source_switched
                            ),
                            previous_formation_wide_evidenced=(
                                previous_in_section is not None
                                and _formation_wide_evidence(
                                    shot_size=previous_in_section.shot_size,
                                    formation_person_count_normalized=(
                                        previous_in_section
                                        .formation_person_count_normalized
                                    ),
                                    formation_pose_coverage=(
                                        previous_in_section
                                        .formation_pose_coverage
                                    ),
                                    config=config,
                                )
                            ),
                            formation_person_count_normalized=(
                                selected_continuity.formation_person_count_normalized
                            ),
                            formation_pose_coverage=(
                                selected_continuity.formation_pose_coverage
                            ),
                            config=config,
                            prior_source_switch_boundaries=(
                                prior_source_switch_boundaries
                            ),
                            prior_wide_involved_source_switch_boundaries=(
                                prior_wide_involved_source_switch_boundaries
                            ),
                            prior_section_known_duration_seconds=(
                                prior_section_known_duration
                            ),
                            prior_section_closeup_duration_seconds=(
                                prior_section_closeup_duration
                            ),
                            prior_section_nonwide_duration_seconds=(
                                prior_section_nonwide_duration
                            ),
                            boundary_motion_accent_supported=bool(
                                boundary_motion_evidence["accent_supported"]
                            ),
                        )
                        components = (
                            _component("cut", cut_raw, config.cut_score_weight),
                            _component("source", source_raw, config.source_score_weight),
                            _component(
                                "duration", duration_raw, config.duration_score_weight
                            ),
                            _component(
                                "transition",
                                transition_raw,
                                config.transition_score_weight,
                            ),
                            _component(
                                "crop_correction",
                                -crop_correction,
                                config.crop_correction_weight,
                            ),
                            _component(
                                "source_quality",
                                quality_raw,
                                config.source_quality_weight,
                            ),
                            _component(
                                "boundary_continuity",
                                boundary_continuity_raw,
                                config.boundary_continuity_weight,
                            ),
                            _component(
                                "cadence", cadence_raw, config.cadence_score_weight
                            ),
                            _component(
                                "diversity", novelty_raw, config.diversity_score_weight
                            ),
                            _component(
                                "occupancy_balance",
                                occupancy_raw,
                                config.occupancy_balance_weight,
                            ),
                            _component(
                                "montage_grammar",
                                montage_grammar_raw,
                                config.montage_grammar_score_weight,
                            ),
                        )
                        decision_score = sum(item.contribution for item in components)
                        total_score = state.score + decision_score
                        if not math.isfinite(total_score):
                            raise StageEditDecodeInputError(
                                "beam expansion produced a non-finite score"
                            )
                        probability_terms = [source_raw, duration_raw]
                        if state.decisions:
                            probability_terms.extend((boundary_cut_score, transition_raw))
                        confidence = math.exp(
                            min(0.0, sum(probability_terms) / len(probability_terms))
                        )
                        recent_sources = (
                            state.recent_sources + (source_index,)
                        )[-config.recent_source_window :]
                        recent_sizes = (
                            state.recent_shot_sizes + (shot_size,)
                        )[-config.recent_shot_size_window :]
                        recent_durations = (
                            state.recent_shot_durations + (duration,)
                        )[-config.recent_shot_size_window :]
                        recent_cuts = tuple(
                            value
                            for value in (
                                state.recent_cut_times
                                + ((start_time,) if state.decisions else ())
                            )
                            if value
                            >= end_time - config.cut_density_window_seconds
                        )
                        recent_density = (
                            len(recent_cuts)
                            * 60.0
                            / config.cut_density_window_seconds
                        )
                        decision = StageEditBeamDecision(
                            shot_index=len(state.decisions),
                            start_candidate_index=start_index,
                            end_candidate_index=end_index,
                            master_start_seconds=start_time,
                            master_end_seconds=end_time,
                            source_index=source_index,
                            source_id=metadata.source_id,
                            shot_size=shot_size,
                            detected_person_count_normalized=(
                                selected_continuity.detected_person_count_normalized
                            ),
                            formation_person_count_normalized=(
                                selected_continuity.formation_person_count_normalized
                            ),
                            formation_pose_coverage=(
                                selected_continuity.formation_pose_coverage
                            ),
                            duration_bucket=duration_bucket,
                            transition_label=transition_label,
                            crop_requested=requested_crop,
                            crop_applied=applied_crop,
                            crop_was_corrected=crop_corrected,
                            decision_confidence=confidence,
                            score_components=components,
                            recent_sources_after=recent_sources,
                            recent_shot_sizes_after=recent_sizes,
                            recent_shot_durations_after=recent_durations,
                            recent_cut_density_per_minute_after=recent_density,
                            source_occupancy_after_seconds=tuple(
                                prospective_occupancy
                            ),
                            section_switch_counts_after=tuple(section_counts),
                            derivative_family_id=candidate_family,
                            derivative_family_occupancy_after_seconds=(
                                prospective_family_occupancy
                            ),
                            editorial_transform=editorial_transform,
                            dynamic_crop_decode=dynamic_crop_decode,
                            dynamic_crop_decode_v3=dynamic_crop_decode_v3,
                            crop_contract=crop_contract,
                            section_progress=frame.section_progress,
                            phrase_progress=frame.phrase_progress,
                            section_role=frame.section_role,
                            boundary_pose_motion_minimum=(
                                boundary_motion_evidence["pose_motion_minimum"]
                            ),
                            boundary_pose_motion_delta=(
                                boundary_motion_evidence["pose_motion_delta"]
                            ),
                            boundary_camera_motion_delta=(
                                boundary_motion_evidence["camera_motion_delta"]
                            ),
                            boundary_minimum_sharpness=(
                                boundary_motion_evidence["minimum_sharpness"]
                            ),
                            boundary_continuity_score=(
                                boundary_motion_evidence["continuity_score"]
                            ),
                            boundary_handcrafted_motion_continuity_qualifier_score=(
                                boundary_motion_evidence[
                                    "handcrafted_continuity_qualifier_score"
                                ]
                            ),
                            boundary_motion_amount_plus_continuity_evidence_complete=(
                                bool(
                                    boundary_motion_evidence[
                                        "evidence_complete"
                                    ]
                                )
                            ),
                            boundary_motion_amount_plus_continuity_supported=(
                                bool(
                                    boundary_motion_evidence[
                                        "accent_supported"
                                    ]
                                )
                            ),
                            boundary_pose_phase_cosine=(
                                boundary_motion_evidence[
                                    "pose_phase_cosine"
                                ]
                            ),
                            boundary_pose_phase_evidence_observed=(
                                bool(
                                    boundary_motion_evidence[
                                        "pose_phase_evidence_observed"
                                    ]
                                )
                            ),
                            boundary_pose_phase_threshold_bound=(
                                bool(
                                    boundary_motion_evidence[
                                        "pose_phase_threshold_bound"
                                    ]
                                )
                            ),
                            boundary_pose_phase_match_proven=(
                                bool(
                                    boundary_motion_evidence[
                                        "pose_phase_match_proven"
                                    ]
                                )
                            ),
                            boundary_pose_phase_feature_dimension=(
                                boundary_motion_evidence[
                                    "pose_phase_feature_dimension"
                                ]
                            ),
                            boundary_pose_phase_outgoing_observed=bool(
                                boundary_motion_evidence[
                                    "pose_phase_outgoing_observed"
                                ]
                            ),
                            boundary_pose_phase_incoming_observed=bool(
                                boundary_motion_evidence[
                                    "pose_phase_incoming_observed"
                                ]
                            ),
                            boundary_pose_phase_trusted_producer_contract=(
                                boundary_motion_evidence[
                                    "pose_phase_trusted_producer_contract"
                                ]
                            ),
                            boundary_pose_phase_outgoing_content_sha256=(
                                boundary_motion_evidence[
                                    "pose_phase_outgoing_content_sha256"
                                ]
                            ),
                            boundary_pose_phase_incoming_content_sha256=(
                                boundary_motion_evidence[
                                    "pose_phase_incoming_content_sha256"
                                ]
                            ),
                            boundary_pose_phase_pair_content_sha256=(
                                boundary_motion_evidence[
                                    "pose_phase_pair_content_sha256"
                                ]
                            ),
                            boundary_temporal_pose_motion_thresholds_bound=(
                                bool(
                                    boundary_motion_evidence[
                                        "temporal_pose_motion_thresholds_bound"
                                    ]
                                )
                            ),
                            boundary_temporal_pose_motion_input_valid=(
                                bool(
                                    boundary_motion_evidence[
                                        "temporal_pose_motion_input_valid"
                                    ]
                                )
                            ),
                            boundary_temporal_outgoing_motion_magnitude=(
                                boundary_motion_evidence[
                                    "temporal_outgoing_boundary_motion_magnitude"
                                ]
                            ),
                            boundary_temporal_incoming_motion_magnitude=(
                                boundary_motion_evidence[
                                    "temporal_incoming_boundary_motion_magnitude"
                                ]
                            ),
                            boundary_temporal_endpoint_pose_cosine=(
                                boundary_motion_evidence[
                                    "temporal_endpoint_pose_cosine"
                                ]
                            ),
                            boundary_temporal_velocity_direction_cosine=(
                                boundary_motion_evidence[
                                    "temporal_velocity_direction_cosine"
                                ]
                            ),
                            boundary_temporal_phase_compatibility=(
                                boundary_motion_evidence[
                                    "temporal_phase_compatibility"
                                ]
                            ),
                            boundary_temporal_confidence=float(
                                boundary_motion_evidence["temporal_confidence"]
                            ),
                            boundary_temporal_classification=(
                                boundary_motion_evidence[
                                    "temporal_classification"
                                ]
                            ),
                            boundary_temporal_static_pose_match=bool(
                                boundary_motion_evidence[
                                    "temporal_static_pose_match"
                                ]
                            ),
                            boundary_temporal_dynamic_pose_phase_match=bool(
                                boundary_motion_evidence[
                                    "temporal_dynamic_pose_phase_match"
                                ]
                            ),
                            boundary_temporal_failure_reasons=tuple(
                                str(value)
                                for value in boundary_motion_evidence[
                                    "temporal_failure_reasons"
                                ]
                            ),
                            boundary_temporal_sampling_contract_match=bool(
                                boundary_motion_evidence[
                                    "temporal_sampling_contract_match"
                                ]
                            ),
                            boundary_temporal_outgoing_sampling_contract=(
                                boundary_motion_evidence[
                                    "temporal_outgoing_sampling_contract"
                                ]
                            ),
                            boundary_temporal_incoming_sampling_contract=(
                                boundary_motion_evidence[
                                    "temporal_incoming_sampling_contract"
                                ]
                            ),
                            boundary_temporal_feature_dimension=(
                                boundary_motion_evidence[
                                    "temporal_feature_dimension"
                                ]
                            ),
                            boundary_temporal_outgoing_sidecar_observed=bool(
                                boundary_motion_evidence[
                                    "temporal_outgoing_sidecar_observed"
                                ]
                            ),
                            boundary_temporal_incoming_sidecar_observed=bool(
                                boundary_motion_evidence[
                                    "temporal_incoming_sidecar_observed"
                                ]
                            ),
                            boundary_temporal_outgoing_trusted_producer_contract=(
                                boundary_motion_evidence[
                                    "temporal_outgoing_trusted_producer_contract"
                                ]
                            ),
                            boundary_temporal_incoming_trusted_producer_contract=(
                                boundary_motion_evidence[
                                    "temporal_incoming_trusted_producer_contract"
                                ]
                            ),
                            boundary_temporal_outgoing_content_sha256=(
                                boundary_motion_evidence[
                                    "temporal_outgoing_content_sha256"
                                ]
                            ),
                            boundary_temporal_incoming_content_sha256=(
                                boundary_motion_evidence[
                                    "temporal_incoming_content_sha256"
                                ]
                            ),
                            boundary_temporal_pair_content_sha256=(
                                boundary_motion_evidence[
                                    "temporal_pair_content_sha256"
                                ]
                            ),
                            boundary_pose_direction_or_velocity_proven=bool(
                                boundary_motion_evidence[
                                    "pose_direction_or_velocity_proven"
                                ]
                            ),
                        )
                        new_state = _BeamState(
                            current_index=end_index,
                            current_source=source_index,
                            current_section_index=section_index,
                            section_local_source_switch_boundaries=(
                                prior_source_switch_boundaries
                                + int(derivative_family_source_switched)
                            ),
                            section_local_wide_involved_source_switch_boundaries=(
                                prior_wide_involved_source_switch_boundaries
                                + int(wide_involved_source_switch)
                            ),
                            recent_sources=recent_sources,
                            recent_shot_sizes=recent_sizes,
                            recent_shot_durations=recent_durations,
                            recent_cut_times=recent_cuts,
                            source_occupancy_seconds=tuple(prospective_occupancy),
                            section_switch_counts=tuple(section_counts),
                            decisions=state.decisions + (decision,),
                            score=total_score,
                            component_totals=_add_component_totals(
                                state.component_totals, components
                            ),
                        )
                        beams[end_index].append(new_state)
                        accepted_hypothesis_count += 1
                        endpoint_records[start_index]["accepted_hypothesis_count"] += 1

                        if len(beams[end_index]) > config.beam_width * 2:
                            unique, removed, duplicates = _prune_states(
                                beams[end_index], config.beam_width
                            )
                            if duplicates:
                                rejection_counts["duplicate_partial_timeline"] = (
                                    rejection_counts.get(
                                        "duplicate_partial_timeline", 0
                                    )
                                    + duplicates
                                )
                            if removed > 0:
                                pruned_count += removed
                                endpoint_records[end_index][
                                    "beam_pruned_state_count"
                                ] += removed
                            beams[end_index] = unique

    raw_complete = sorted(beams[terminal_index], key=_state_sort_key)
    endpoint_records[terminal_index]["incoming_state_count"] = len(raw_complete)
    if not raw_complete:
        summary = ", ".join(
            f"{name}={count}" for name, count in sorted(rejection_counts.items())
        )
        raise StageEditNoValidTimelineError(
            "hard constraints eliminated every complete StageEdit timeline"
            + (f": {summary}" if summary else "")
        )

    complete: list[_BeamState] = []
    montage_rejected_count = 0
    for state in raw_complete:
        montage = _evaluate_montage_grammar(
            decisions=state.decisions,
            section_plan=section_plan,
            support_by_section=montage_support_by_section,
            config=config,
        )
        if not bool(montage["passed"]):
            montage_rejected_count += 1
            for failure in montage["failure_codes"]:
                reason = str(failure).partition(":")[2] or str(failure)
                reject(reason)
            continue
        soft_target_score = montage["highlight_aggregate"]["soft_targets"][
            "aggregate_alignment_score"
        ]
        if soft_target_score is not None:
            aggregate_component = _component(
                "highlight_aggregate_soft_target",
                float(soft_target_score),
                config.montage_grammar_score_weight,
            )
            state = replace(
                state,
                score=state.score + aggregate_component.contribution,
                component_totals=_add_component_totals(
                    state.component_totals,
                    (aggregate_component,),
                ),
            )
        close_burst_score = montage["highlight_aggregate"]["soft_targets"][
            "close_only_multi_source_burst_event_p10"
        ]["evidence_weighted_alignment_score"]
        if close_burst_score is not None and float(close_burst_score) > 0.0:
            close_burst_component = _component(
                "highlight_close_burst_soft_preference",
                min(1.0, max(0.0, float(close_burst_score))),
                config.montage_grammar_score_weight,
            )
            state = replace(
                state,
                score=state.score + close_burst_component.contribution,
                component_totals=_add_component_totals(
                    state.component_totals,
                    (close_burst_component,),
                ),
            )
        complete.append(state)
    if not complete:
        summary = ", ".join(
            f"{name}={count}" for name, count in sorted(rejection_counts.items())
        )
        raise StageEditNoValidTimelineError(
            "montage grammar eliminated every complete StageEdit timeline"
            + (f": {summary}" if summary else "")
        )

    complete.sort(key=_state_sort_key)
    selected_states: list[_BeamState] = []
    completion_signatures: set[tuple[Any, ...]] = set()
    duplicate_completions = 0
    for state in complete:
        signature = state.signature()
        if signature in completion_signatures:
            duplicate_completions += 1
            continue
        completion_signatures.add(signature)
        selected_states.append(state)
        if len(selected_states) == config.timeline_count:
            break

    checkpoint_path = str(readiness.checkpoint_path)
    timelines: list[StageEditBeamTimeline] = []
    for rank, state in enumerate(selected_states, start=1):
        edit = _build_stage_edit(
            rank=rank,
            state=state,
            sources=normalized_sources,
            source_metadata=normalized_metadata,
            section_plan=section_plan,
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
            fps=fps,
            reference_source=reference_source,
            output=Path(output),
            checkpoint_path=checkpoint_path,
            source_derivative_metadata=source_derivative_metadata,
            config=config,
            boundary_continuity_score_provenance=(
                boundary_continuity_score_provenance
            ),
        )
        recent_density = (
            len(state.recent_cut_times)
            * 60.0
            / config.cut_density_window_seconds
        )
        timelines.append(
            StageEditBeamTimeline(
                rank=rank,
                score=state.score,
                score_components=dict(state.component_totals),
                decisions=state.decisions,
                edit=edit,
                source_occupancy_seconds=state.source_occupancy_seconds,
                recent_cut_density_per_minute=recent_density,
                montage_grammar=_evaluate_montage_grammar(
                    decisions=state.decisions,
                    section_plan=section_plan,
                    support_by_section=montage_support_by_section,
                    config=config,
                ),
                derivative_family_occupancy_seconds=_family_occupancy(
                    state.source_occupancy_seconds,
                    derivative_family_ids,
                ),
                boundary_continuity_score_provenance=(
                    boundary_continuity_score_provenance
                ),
            )
        )

    digest = _input_digest(
        sources=normalized_sources,
        source_metadata=normalized_metadata,
        frames=frames,
        section_plan=section_plan,
        constraints=constraints,
        config=audit_config,
        schema=schema,
        source_derivative_metadata=source_derivative_metadata,
        boundary_continuity_score_provenance=(
            boundary_continuity_score_provenance
        ),
    )
    selected_dynamic_v2_decodes = tuple(
        decision.dynamic_crop_decode
        for timeline in timelines
        for decision in timeline.decisions
        if decision.dynamic_crop_decode is not None
    )
    selected_dynamic_v3_decodes = tuple(
        decision.dynamic_crop_decode_v3
        for timeline in timelines
        for decision in timeline.decisions
        if decision.dynamic_crop_decode_v3 is not None
    )
    selected_dynamic_decodes = (
        *selected_dynamic_v2_decodes,
        *selected_dynamic_v3_decodes,
    )
    dynamic_rejection_counts: dict[str, int] = {}
    for decoded in selected_dynamic_decodes:
        for reason in decoded.rejection_reasons:
            dynamic_rejection_counts[reason] = (
                dynamic_rejection_counts.get(reason, 0) + 1
            )
    audit = StageEditDecodeAudit(
        schema_version=DECODER_SCHEMA_VERSION,
        decoder=DECODER_CONTRACT,
        input_sha256=digest,
        readiness=readiness.to_dict(),
        # Keep this exact dataclass payload round-trippable by the windowed
        # redecoder. Detailed runtime scorer provenance lives in the explicit
        # audit/timeline score-provenance records below.
        config=audit_config,
        input_summary={
            "source_count": source_count,
            "transition_authority": transition_authority.to_dict(),
            "boundary_continuity_score_provenance": dict(
                boundary_continuity_score_provenance
            ),
            "decoder_schema_migration": {
                "current": DECODER_SCHEMA_VERSION,
                "supersedes": 13,
                "reason": (
                    "highlight motion eligibility now recomputes its calibrated "
                    "handcrafted continuity qualifier independently of the "
                    "primary learned-or-handcrafted boundary scorer, without "
                    "changing Transformer tokens or checkpoints"
                ),
                "motion_continuity_qualifier_independent_from_primary": True,
                "learned_primary_score_reused_for_handcrafted_quantile": False,
                "legacy_source_counter_aliases_retained": True,
                "automatic_group_requires_verified_family_metadata": True,
                "same_time_pose_cosine_relabelled_as_temporal": False,
                "current_motion_profile_direction_calibration_claimed": False,
                "temporal_calibration_bound": (
                    config.highlight_temporal_pose_motion_calibration is not None
                ),
                "pose_direction_or_velocity_claimed": any(
                    decision.boundary_pose_direction_or_velocity_proven
                    for timeline in timelines
                    for decision in timeline.decisions
                ),
            },
            "source_derivative_diversity": (
                source_derivative_metadata.to_dict()
                if source_derivative_metadata is not None
                else {
                    "production_ready": False,
                    "family_ids": list(derivative_family_ids),
                    "family_count": derivative_family_count,
                    "status": "decoder_unit_identity_default",
                }
            ),
            "diversity_semantics": "verified_derivative_family_not_file_source",
            "occupancy_semantics": "aggregate_all_sources_in_derivative_family",
            "candidate_frame_count": len(frames),
            "section_count": section_count,
            "master_start_seconds": section_plan.start_seconds,
            "master_end_seconds": section_plan.end_seconds,
            "master_duration_seconds": master_duration,
            "montage_grammar_contract": MONTAGE_GRAMMAR_CONTRACT,
            "montage_grammar_candidate_support_by_section": [
                dict(value) for value in montage_support_by_section
            ],
            "raw_complete_state_count_before_montage_grammar": len(raw_complete),
            "montage_grammar_rejected_complete_state_count": (
                montage_rejected_count
            ),
            "token_schema_fingerprint": schema.fingerprint,
            "decision_history_owner": STAGE_EDIT_DECISION_HISTORY_OWNER,
            "owned_decision_history_state": [
                "current_source",
                "recent_sources",
                "recent_shot_sizes",
                "recent_shot_durations",
                "recent_cut_times",
                "source_occupancy_seconds",
                "section_switch_counts",
            ],
            "all_frames_have_effective_source": all(
                any(frame.effective_source_mask) for frame in frames
            ),
            "crop_contract": (
                DYNAMIC_CROP_V2_SEMIMARKOV_ADAPTER_CONTRACT
                if frames[0].dynamic_crop_v2 is not None
                else DYNAMIC_CROP_V3_SEMIMARKOV_ADAPTER_CONTRACT
                if frames[0].dynamic_crop_v3_by_source
                else "legacy_static_crop_head_v1"
            ),
            "dynamic_crop_mode": (
                "v2"
                if selected_dynamic_v2_decodes
                else "v3"
                if selected_dynamic_v3_decodes
                else "legacy"
            ),
            "selected_dynamic_crop_attempt_count": len(selected_dynamic_decodes),
            "selected_dynamic_crop_v2_attempt_count": len(
                selected_dynamic_v2_decodes
            ),
            "selected_dynamic_crop_v3_attempt_count": len(
                selected_dynamic_v3_decodes
            ),
            "selected_dynamic_crop_v3_source_counts": dict(
                sorted(
                    {
                        f"{source_index}:{source_id}": sum(
                            value.source_index == source_index
                            and value.source_id == source_id
                            for value in selected_dynamic_v3_decodes
                        )
                        for source_index, source_id in enumerate(
                            value.source_id for value in normalized_metadata
                        )
                    }.items()
                )
            ),
            "selected_dynamic_crop_eligible_count": sum(
                value.eligible for value in selected_dynamic_decodes
            ),
            "selected_dynamic_crop_applied_keyframe_count": sum(
                value.applied_keyframe_count for value in selected_dynamic_decodes
            ),
            "selected_dynamic_crop_rejection_reasons": dict(
                sorted(dynamic_rejection_counts.items())
            ),
            "transition_render_policy": "metadata_preserved_rendered_as_hard_cut",
        },
        endpoint_audit=tuple(endpoint_records),
        constraint_rejection_counts=dict(sorted(rejection_counts.items())),
        transition_hypothesis_count=transition_hypothesis_count,
        accepted_hypothesis_count=accepted_hypothesis_count,
        beam_pruned_state_count=pruned_count,
        complete_state_count=len(complete),
        duplicate_completion_count=duplicate_completions,
        requested_timeline_count=config.timeline_count,
        returned_timeline_count=len(timelines),
        request_fulfilled=len(timelines) == config.timeline_count,
        score_provenance={
            "boundary_continuity": dict(
                boundary_continuity_score_provenance
            )
        },
    )
    return StageEditSemiMarkovResult(tuple(timelines), audit)


__all__ = [
    "CONTINUITY_AUTHORITY_CONTRACT",
    "DECODER_CONTRACT",
    "DECODER_SCHEMA_VERSION",
    "HANDCRAFTED_CONTINUITY_AUTHORITY",
    "HANDCRAFTED_CONTINUITY_SCORER_CONTRACT",
    "HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT",
    "HIGHLIGHT_MOTION_CALIBRATION_CONTRACT",
    "HIGHLIGHT_MOTION_CALIBRATION_QUANTILES",
    "MONTAGE_GRAMMAR_CONTRACT",
    "PRODUCTION_LEARNED_CONTINUITY_AUTHORITY",
    "StageEditBeamDecision",
    "StageEditBeamTimeline",
    "StageEditCheckpointNotReadyError",
    "StageEditDecodeAudit",
    "StageEditDecodeInputError",
    "StageEditHeadFrame",
    "StageEditHighlightMotionCalibration",
    "StageEditTemporalPoseMotionCalibration",
    "StageEditNoValidTimelineError",
    "StageEditScoreComponent",
    "StageEditSectionConstraint",
    "StageEditSemiMarkovConfig",
    "StageEditSemiMarkovError",
    "StageEditSemiMarkovResult",
    "StageEditSourceContinuity",
    "StageEditSourceMetadata",
    "TEMPORAL_POSE_MOTION_CALIBRATION_ACTIVATION_AVAILABLE",
    "TEMPORAL_POSE_MOTION_CALIBRATION_CONTRACT",
    "decode_stage_edit_semimarkov_beam",
    "stage_edit_boundary_continuity_score",
]
