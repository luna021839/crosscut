"""Structured StageEdit Transformer prototype backed by Apple MLX.

This module is intentionally offline and is not wired into the production
timeline generator yet.  It provides the first trainable vertical slice:

* a versioned, dependency-free token and target schema;
* deterministic tensorization of variable source candidate sets;
* a causal Transformer with separate editorial decision heads;
* masked listwise/pairwise training losses;
* conservative checkpoint/readiness reporting.

Importing the schema does not require MLX.  Constructing/tensorizing a model
does; install the ``stage-edit-mlx`` optional dependency on Apple silicon.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .reference_dataset import (
    BOUNDARY_EVENT_SCHEMA_VERSION,
    EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT,
    EDITORIAL_SOURCE_SWITCH_UNRESOLVED_EVENT,
    KNOWN_AFTER_UNKNOWN_EVENT,
    NATIVE_SOURCE_CAMERA_CUT_EVENT,
    REFERENCE_START_EVENT,
)
from .stage_edit_transition_authority import (
    LEARNED_TRANSITION_AUTHORITY,
    RENDERER_OWNED_TRANSITION_AUTHORITY,
    StageEditTransitionAuthority,
    transition_authority_contract,
)

try:  # Optional so dataset/schema tooling remains portable.
    import mlx.core as mx
    import mlx.nn as nn
except ImportError as exc:  # pragma: no cover - exercised on non-Apple CI.
    mx = None
    nn = None
    _MLX_IMPORT_ERROR: ImportError | None = exc
else:
    _MLX_IMPORT_ERROR = None


TOKEN_SCHEMA_VERSION = 2
CHECKPOINT_FORMAT_VERSION = 3
MASKED_SOURCE_LOGIT = -1.0e9
PRODUCTION_TRAINING_REPORT_FILENAME = "training-report.json"
PRODUCTION_TRAINING_RUN_KIND = "stage_edit_production_training"
PRODUCTION_CHECKPOINT_SCOPE = "production_learned_editor"
PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION = 7
DYNAMIC_CROP_V3_ACTIVATION_CHECKPOINTING_CONTRACT = (
    "mlx_transformer_encoder_layer_recompute_training_only_v1"
)
_DYNAMIC_CROP_V3_ACTIVATION_CHECKPOINTING_KEYS = frozenset(
    {
        "contract",
        "enabled",
        "scope",
        "implementation",
        "training_only",
        "serialized_parameter_tree_changed",
        "inference_runtime_requires_checkpointing",
        "previous_value",
    }
)
PRODUCTION_FEATURE_SNAPSHOT_GATE_CONTRACT = (
    "shared_media_feature_snapshot_with_postload_history_scrub_v2"
)
STAGE_EDIT_RECENT_HISTORY_MASK_CONTRACT = (
    "transformer_recent_edit_slice_fully_unknown_train_and_serve_v1"
)
STAGE_EDIT_DECISION_HISTORY_OWNER = "duration_aware_semimarkov_beam_state_v1"
STAGE_EDIT_SOURCE_RELATION_MODE = (
    "candidate_set_mean_residual_bounded_zscore_v1"
)
STAGE_EDIT_SOURCE_SCORING_CONTRACT = (
    "candidate_relational_shared_permutation_equivariant_scorer_v2"
)
STAGE_EDIT_CUT_LOGIT_MODE = (
    "raw_context_hard_gated_mean_max_linear_plus_zero_init_transformer_residual_v1"
)
STAGE_EDIT_CUT_SCORING_CONTRACT = (
    "raw_linear_signal_preserving_cut_path_plus_transformer_residual_v1"
)
STAGE_EDIT_CUT_IMBALANCE_CONTRACT = (
    "training_partition_only_bounded_positive_weight_v1"
)
STAGE_EDIT_VALIDATION_SELECTION_CONTRACT = (
    "training_song_reference_validation_cut_auroc_source_mrr_total_composite_v1"
)
STAGE_EDIT_CUT_THRESHOLD_SELECTION_CONTRACT = (
    "training_song_reference_validation_balanced_accuracy_threshold_v1"
)
DYNAMIC_CROP_V3_VALIDATION_SELECTION_GATE_CONTRACT = (
    "dynamic_crop_v3_renderer_exact_static_prior_improvement_gate_v1"
)
DYNAMIC_CROP_V3_TRAINING_CORPUS_CONTRACT = (
    "immutable_v2_renderer_evidence_to_simplified_source_exact_v3_targets_v1"
)
DYNAMIC_CROP_V3_PRODUCTION_ARCHITECTURE = (
    "causal_long_context_transformer_with_source_conditioned_dynamic_crop_v3"
)


def valid_dynamic_crop_v3_activation_checkpointing_evidence(
    value: Any,
) -> bool:
    """Return whether a report carries the exact typed recomputation proof."""

    return (
        isinstance(value, Mapping)
        and set(value) == _DYNAMIC_CROP_V3_ACTIVATION_CHECKPOINTING_KEYS
        and value.get("contract")
        == DYNAMIC_CROP_V3_ACTIVATION_CHECKPOINTING_CONTRACT
        and value.get("enabled") is True
        and value.get("scope") == "base_transformer_encoder_all_layers"
        and value.get("implementation")
        == "mlx.nn.TransformerEncoder.checkpoint"
        and value.get("training_only") is True
        and value.get("serialized_parameter_tree_changed") is False
        and value.get("inference_runtime_requires_checkpointing") is False
        and value.get("previous_value") is False
    )


STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES = tuple(
    f"pose_model_embedding_{index:02d}" for index in range(60)
)
STAGE_EDIT_FORMATION_LAYOUT_FEATURE_NAMES = tuple(
    f"formation_layout_embedding_{index:03d}" for index in range(134)
)
STAGE_EDIT_FORMATION_POSE_FEATURE_NAMES = tuple(
    f"formation_pose_person_{person}_joint_{joint}_{axis}"
    for person in range(6)
    for joint in range(8)
    for axis in ("x", "y", "confidence")
)
STAGE_EDIT_ACTION_FORMATION_SOURCE_FEATURE_NAMES = (
    "pose_model_confidence",
    "detected_person_count_normalized",
    "formation_person_count_normalized",
    "formation_pose_coverage",
    *STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES,
    *STAGE_EDIT_FORMATION_LAYOUT_FEATURE_NAMES,
    *STAGE_EDIT_FORMATION_POSE_FEATURE_NAMES,
)

STAGE_EDIT_TEMPORAL_POSE_VISUAL_CACHE_CURRENT_NEXT_CONTRACT = (
    "source_exact_visual_cache_current_next_real_pose60_v1"
)
STAGE_EDIT_TEMPORAL_POSE_TIMELINE_ENTRY_EXIT_CONTRACT = (
    "source_exact_timeline_entry_exit_real_pose60_v1"
)
STAGE_EDIT_TEMPORAL_POSE_PRODUCER_BACKEND = (
    "opencv_zoo_mediapipe_pose_2023mar"
)
STAGE_EDIT_TEMPORAL_POSE_SAMPLING_CONTRACTS = frozenset(
    {
        STAGE_EDIT_TEMPORAL_POSE_VISUAL_CACHE_CURRENT_NEXT_CONTRACT,
        STAGE_EDIT_TEMPORAL_POSE_TIMELINE_ENTRY_EXIT_CONTRACT,
    }
)


def _tuple_of_strings(value: Sequence[Any], field_name: str) -> tuple[str, ...]:
    output = tuple(str(item).strip() for item in value)
    if not output or any(not item for item in output):
        raise ValueError(f"{field_name} must contain non-empty names")
    if len(set(output)) != len(output):
        raise ValueError(f"{field_name} names must be unique")
    return output


@dataclass(frozen=True, slots=True)
class StageEditTokenSchema:
    """Names and categorical vocabularies that define every tensor column."""

    schema_version: int
    music_feature_names: tuple[str, ...]
    section_feature_names: tuple[str, ...]
    source_feature_names: tuple[str, ...]
    recent_edit_feature_names: tuple[str, ...]
    style_feature_names: tuple[str, ...]
    duration_bucket_edges_seconds: tuple[float, ...]
    transition_labels: tuple[str, ...]
    crop_feature_names: tuple[str, ...] = (
        "zoom",
        "center_x",
        "center_y",
        "roll_degrees",
    )

    def __post_init__(self) -> None:
        if self.schema_version != TOKEN_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported StageEdit token schema {self.schema_version}"
            )
        for field_name in (
            "music_feature_names",
            "section_feature_names",
            "source_feature_names",
            "recent_edit_feature_names",
            "style_feature_names",
            "transition_labels",
            "crop_feature_names",
        ):
            normalized = _tuple_of_strings(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, normalized)
        if len(self.crop_feature_names) != 4:
            raise ValueError("crop head requires zoom, x, y, and roll fields")
        edges = tuple(float(value) for value in self.duration_bucket_edges_seconds)
        if (
            not edges
            or any(not math.isfinite(value) or value <= 0.0 for value in edges)
            or any(right <= left for left, right in zip(edges, edges[1:]))
        ):
            raise ValueError("duration bucket edges must be finite and increasing")
        object.__setattr__(self, "duration_bucket_edges_seconds", edges)

    @property
    def raw_context_dimension(self) -> int:
        return (
            len(self.music_feature_names)
            + len(self.section_feature_names)
            + len(self.recent_edit_feature_names)
            + len(self.style_feature_names)
        )

    @property
    def encoded_context_dimension(self) -> int:
        # Each normalized scalar is paired with its explicit missing bit.
        return 2 * self.raw_context_dimension

    @property
    def raw_source_dimension(self) -> int:
        return len(self.source_feature_names)

    @property
    def encoded_source_dimension(self) -> int:
        return 2 * self.raw_source_dimension

    @property
    def duration_bucket_count(self) -> int:
        return len(self.duration_bucket_edges_seconds) + 1

    def duration_bucket(self, seconds: float) -> int:
        value = float(seconds)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("duration must be finite and non-negative")
        return sum(value >= edge for edge in self.duration_bucket_edges_seconds)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageEditTokenSchema":
        tuple_fields = {
            "music_feature_names",
            "section_feature_names",
            "source_feature_names",
            "recent_edit_feature_names",
            "style_feature_names",
            "duration_bucket_edges_seconds",
            "transition_labels",
            "crop_feature_names",
        }
        values = {
            key: tuple(value) if key in tuple_fields else value
            for key, value in payload.items()
        }
        return cls(**values)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


DEFAULT_STAGE_EDIT_SCHEMA = StageEditTokenSchema(
    schema_version=TOKEN_SCHEMA_VERSION,
    music_feature_names=(
        "beat_phase_sin",
        "beat_phase_cos",
        "bar_phase_sin",
        "bar_phase_cos",
        "downbeat_probability",
        "onset_strength",
        "rms_energy",
        "spectral_flux",
        "vocal_probability",
        "tempo_normalized",
        "phrase_progress",
        "seconds_to_next_beat_normalized",
    ),
    section_feature_names=(
        "intro_probability",
        "verse_probability",
        "prechorus_probability",
        "chorus_probability",
        "bridge_probability",
        "outro_probability",
        "section_progress",
        "section_energy_delta",
    ),
    source_feature_names=(
        "sync_confidence",
        "sync_residual_normalized",
        "shot_scale",
        "face_scale",
        "face_center_x",
        "face_center_y",
        "body_center_x",
        "body_center_y",
        "pose_motion",
        "camera_motion_x",
        "camera_motion_y",
        "camera_zoom_velocity",
        "camera_roll_degrees_normalized",
        "sharpness",
        "exposure",
        "color_temperature_normalized",
        "occlusion_probability",
        "visual_match_confidence",
        *STAGE_EDIT_ACTION_FORMATION_SOURCE_FEATURE_NAMES,
    ),
    recent_edit_feature_names=(
        "seconds_since_cut_normalized",
        "recent_cut_density",
        "same_source_run_normalized",
        "source_reuse_count_normalized",
        "previous_face_scale",
        "previous_face_center_x",
        "previous_face_center_y",
        "previous_zoom",
        "previous_roll_normalized",
        "previous_transition_strength",
    ),
    style_feature_names=(
        "target_cut_density",
        "matchcut_strength",
        "motion_transition_strength",
        "virtual_camera_strength",
        "closeup_preference",
        "source_diversity_preference",
        "color_continuity_preference",
        "experimental_transition_budget",
    ),
    duration_bucket_edges_seconds=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
    transition_labels=(
        "hard_cut",
        "zoom_match",
        "whip_pan",
        "cross_dissolve",
    ),
)


@dataclass(frozen=True, slots=True)
class StageEditTemporalPoseSidecar:
    """Source-exact two-sample pose evidence excluded from model tensors.

    The sidecar is serialized and content-hashed with feature snapshots, but
    it is not named by :class:`StageEditTokenSchema` and is never consumed by
    ``tensorize_stage_edit_batch``.  This preserves every existing checkpoint
    dimension and schema fingerprint while allowing post-forward pair audits.
    """

    embeddings: tuple[tuple[float, ...], tuple[float, ...]]
    interval_seconds: float
    sampling_contract: str
    producer_backend: str = STAGE_EDIT_TEMPORAL_POSE_PRODUCER_BACKEND

    def __post_init__(self) -> None:
        samples = tuple(tuple(value for value in sample) for sample in self.embeddings)
        if len(samples) != 2 or any(
            len(sample) != len(STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES)
            for sample in samples
        ):
            raise ValueError(
                "temporal pose sidecar must contain exactly two 60-D samples"
            )
        normalized: list[tuple[float, ...]] = []
        for sample in samples:
            if any(isinstance(value, bool) for value in sample):
                raise ValueError(
                    "temporal pose sidecar values must be real coordinates"
                )
            try:
                values = tuple(float(value) for value in sample)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "temporal pose sidecar values must be real coordinates"
                ) from exc
            if any(
                not math.isfinite(value) or not -1.0 <= value <= 1.0
                for value in values
            ):
                raise ValueError(
                    "temporal pose sidecar values must be finite in [-1, 1]"
                )
            normalized.append(values)
        if isinstance(self.interval_seconds, bool):
            raise ValueError("temporal pose sidecar interval must be positive")
        try:
            interval = float(self.interval_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "temporal pose sidecar interval must be positive"
            ) from exc
        if not math.isfinite(interval) or interval <= 0.0:
            raise ValueError("temporal pose sidecar interval must be positive")
        contract = str(self.sampling_contract).strip()
        if contract not in STAGE_EDIT_TEMPORAL_POSE_SAMPLING_CONTRACTS:
            raise ValueError("temporal pose sidecar sampling contract is unsupported")
        producer_backend = str(self.producer_backend).strip()
        if producer_backend != STAGE_EDIT_TEMPORAL_POSE_PRODUCER_BACKEND:
            raise ValueError(
                "temporal pose sidecar producer backend is not trusted"
            )
        object.__setattr__(self, "embeddings", tuple(normalized))
        object.__setattr__(self, "interval_seconds", interval)
        object.__setattr__(self, "sampling_contract", contract)
        object.__setattr__(self, "producer_backend", producer_backend)

    def to_dict(self) -> dict[str, Any]:
        return {
            "embeddings": [list(sample) for sample in self.embeddings],
            "interval_seconds": self.interval_seconds,
            "sampling_contract": self.sampling_contract,
            "producer_backend": self.producer_backend,
            "transformer_tensor_input": False,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "StageEditTemporalPoseSidecar":
        expected = {
            "embeddings",
            "interval_seconds",
            "sampling_contract",
            "producer_backend",
            "transformer_tensor_input",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("temporal pose sidecar schema does not match")
        if payload.get("transformer_tensor_input") is not False:
            raise ValueError("temporal pose sidecar cannot enter Transformer tensors")
        raw_embeddings = payload.get("embeddings")
        if not isinstance(raw_embeddings, list) or any(
            not isinstance(sample, list) for sample in raw_embeddings
        ):
            raise ValueError("temporal pose sidecar embeddings must be arrays")
        return cls(
            embeddings=tuple(tuple(sample) for sample in raw_embeddings),
            interval_seconds=payload.get("interval_seconds"),
            sampling_contract=payload.get("sampling_contract"),
            producer_backend=payload.get("producer_backend"),
        )


@dataclass(frozen=True, slots=True)
class SourceCandidateToken:
    """Features for one synchronized source at one master-timeline token."""

    source_id: str
    features: tuple[float, ...]
    missing_mask: tuple[bool, ...] = ()
    available: bool = True
    hard_gate_passed: bool = True
    temporal_pose_sidecar: StageEditTemporalPoseSidecar | None = None

    def __post_init__(self) -> None:
        if self.temporal_pose_sidecar is not None and not isinstance(
            self.temporal_pose_sidecar, StageEditTemporalPoseSidecar
        ):
            raise TypeError(
                "temporal_pose_sidecar must use StageEditTemporalPoseSidecar"
            )


@dataclass(frozen=True, slots=True)
class StageEditToken:
    """One causal decision point; feature order is defined by the schema."""

    master_time_seconds: float
    music_features: tuple[float, ...]
    section_features: tuple[float, ...]
    source_candidates: tuple[SourceCandidateToken, ...]
    recent_edit_features: tuple[float, ...]
    style_features: tuple[float, ...]
    # Concatenated music + section + recent-edit + style missing bits.
    missing_mask: tuple[bool, ...] = ()


@dataclass(frozen=True, slots=True)
class StageEditSequence:
    sequence_id: str
    tokens: tuple[StageEditToken, ...]


def stage_edit_recent_history_mask_contract(
    schema: StageEditTokenSchema,
) -> dict[str, Any]:
    """Return the schema-bound policy for decision-history model inputs.

    A single global Transformer forward cannot condition each semi-Markov beam
    on that beam's different edit history.  Production therefore gives the
    Transformer no editorial-decision history at all.  The duration-aware
    semi-Markov state owns recent sources, durations, cuts, occupancy, and
    per-section switch counts.  The numeric slice is zero-filled *and* every
    corresponding missing bit is set, identically for train and serve.
    """

    start = len(schema.music_feature_names) + len(schema.section_feature_names)
    stop = start + len(schema.recent_edit_feature_names)
    unsigned: dict[str, Any] = {
        "contract": STAGE_EDIT_RECENT_HISTORY_MASK_CONTRACT,
        "decision_history_owner": STAGE_EDIT_DECISION_HISTORY_OWNER,
        "token_schema_fingerprint": schema.fingerprint,
        "raw_context_slice": {
            "start_inclusive": start,
            "stop_exclusive": stop,
            "feature_names": list(schema.recent_edit_feature_names),
        },
        "numeric_value_policy": "zero_fill_only_behind_missing_mask",
        "missing_mask_policy": "all_recent_edit_slice_bits_true",
        "reference_or_baseline_decision_history_consumed": False,
    }
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        **unsigned,
        "contract_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def mask_stage_edit_recent_history_for_semimarkov(
    sequence: StageEditSequence,
    schema: StageEditTokenSchema,
) -> StageEditSequence:
    """Remove human, baseline, or prior model choices from model inputs.

    This is intentionally applied after reading a reference feature snapshot
    during production training and again at every production forward.  It is
    not enough to write zeros: their missing bits must also be true so a model
    can never interpret the zeros as observed cold-start statistics.
    """

    recent_count = len(schema.recent_edit_feature_names)
    recent_start = len(schema.music_feature_names) + len(
        schema.section_feature_names
    )
    recent_stop = recent_start + recent_count
    masked_tokens: list[StageEditToken] = []
    for token_index, token in enumerate(sequence.tokens):
        if len(token.recent_edit_features) != recent_count:
            raise ValueError(
                f"token {token_index} recent-edit dimension does not match schema"
            )
        if token.missing_mask:
            if len(token.missing_mask) != schema.raw_context_dimension:
                raise ValueError(
                    f"token {token_index} context missing mask does not match schema"
                )
            missing = list(token.missing_mask)
        else:
            missing = [False] * schema.raw_context_dimension
        missing[recent_start:recent_stop] = [True] * recent_count
        masked_tokens.append(
            replace(
                token,
                recent_edit_features=(0.0,) * recent_count,
                missing_mask=tuple(missing),
            )
        )
    return replace(sequence, tokens=tuple(masked_tokens))


@dataclass(frozen=True, slots=True)
class StageEditTarget:
    """Sparse target; ``None`` means the corresponding loss is masked out."""

    cut: float | None = None
    selected_source_id: str | None = None
    rejected_source_ids: tuple[str, ...] = ()
    duration_bucket: int | None = None
    transition_label: str | None = None
    # Per-coordinate None supports partial reverse-match crop evidence.
    crop: tuple[float | None, float | None, float | None, float | None] | None = None


@dataclass(frozen=True, slots=True)
class StageEditTargetSequence:
    sequence_id: str
    targets: tuple[StageEditTarget, ...]


@dataclass(frozen=True, slots=True)
class StageEditTransformerConfig:
    config_version: int
    token_schema_fingerprint: str
    context_input_dimension: int
    source_input_dimension: int
    model_dimension: int
    num_heads: int
    num_layers: int
    feed_forward_dimension: int
    max_sequence_length: int
    max_sources: int
    duration_bucket_count: int
    transition_count: int
    source_relation_mode: str = STAGE_EDIT_SOURCE_RELATION_MODE
    cut_logit_mode: str = STAGE_EDIT_CUT_LOGIT_MODE
    transition_authority: str = LEARNED_TRANSITION_AUTHORITY
    dropout: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = (
            "config_version",
            "context_input_dimension",
            "source_input_dimension",
            "model_dimension",
            "num_heads",
            "num_layers",
            "feed_forward_dimension",
            "max_sequence_length",
            "max_sources",
            "duration_bucket_count",
            "transition_count",
        )
        if any(int(getattr(self, name)) <= 0 for name in integer_fields):
            raise ValueError("StageEdit model dimensions must be positive")
        if self.config_version != 3:
            raise ValueError(f"unsupported model config {self.config_version}")
        if self.source_relation_mode != STAGE_EDIT_SOURCE_RELATION_MODE:
            raise ValueError("unsupported source relational encoder contract")
        if self.cut_logit_mode != STAGE_EDIT_CUT_LOGIT_MODE:
            raise ValueError("unsupported cut scoring contract")
        StageEditTransitionAuthority(mode=self.transition_authority)
        if self.model_dimension % self.num_heads:
            raise ValueError("model dimension must be divisible by attention heads")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if len(self.token_schema_fingerprint) != 64:
            raise ValueError("token schema fingerprint must be SHA-256")

    @classmethod
    def from_schema(
        cls,
        schema: StageEditTokenSchema,
        *,
        model_dimension: int,
        num_heads: int,
        num_layers: int,
        feed_forward_dimension: int,
        max_sequence_length: int,
        max_sources: int,
        dropout: float = 0.0,
        transition_authority: str = LEARNED_TRANSITION_AUTHORITY,
    ) -> "StageEditTransformerConfig":
        return cls(
            config_version=3,
            token_schema_fingerprint=schema.fingerprint,
            context_input_dimension=schema.encoded_context_dimension,
            source_input_dimension=schema.encoded_source_dimension,
            model_dimension=model_dimension,
            num_heads=num_heads,
            num_layers=num_layers,
            feed_forward_dimension=feed_forward_dimension,
            max_sequence_length=max_sequence_length,
            max_sources=max_sources,
            duration_bucket_count=schema.duration_bucket_count,
            transition_count=len(schema.transition_labels),
            transition_authority=transition_authority,
            dropout=dropout,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageEditTransformerConfig":
        return cls(**payload)


def smoke_stage_edit_config(
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
) -> StageEditTransformerConfig:
    return StageEditTransformerConfig.from_schema(
        schema,
        model_dimension=32,
        num_heads=4,
        num_layers=2,
        feed_forward_dimension=96,
        max_sequence_length=32,
        max_sources=8,
        dropout=0.0,
    )


def production_stage_edit_config(
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
) -> StageEditTransformerConfig:
    """Return the planned long-context configuration (roughly 100M weights)."""

    return StageEditTransformerConfig.from_schema(
        schema,
        model_dimension=768,
        num_heads=12,
        num_layers=14,
        feed_forward_dimension=3072,
        max_sequence_length=2048,
        max_sources=32,
        dropout=0.1,
    )


def stage_edit_source_scoring_contract(
    config: StageEditTransformerConfig,
) -> dict[str, Any]:
    """Describe the source head's candidate-relative, index-free contract.

    Source slots are transport containers, not semantic classes.  Every
    candidate is projected by the same weights, scored against the same
    timestamp query, and masked by its own availability/gate state.  Permuting
    candidates must therefore only permute their logits; it cannot change any
    other head or attach learned meaning to an absolute source index.
    """

    unsigned: dict[str, Any] = {
        "contract": STAGE_EDIT_SOURCE_SCORING_CONTRACT,
        "token_schema_fingerprint": config.token_schema_fingerprint,
        "source_input_dimension": config.source_input_dimension,
        "relational_source_input_dimension": 4
        * config.source_input_dimension,
        "source_relation_mode": config.source_relation_mode,
        "maximum_candidate_count": config.max_sources,
        "candidate_representation": [
            "raw_encoded_features",
            "hard_gated_candidate_set_mean",
            "candidate_minus_set_mean_residual",
            "tanh_bounded_candidate_set_zscore",
        ],
        "bounded_zscore_scale": 3.0,
        "candidate_feature_projection_shared": True,
        "timestamp_query_shared_across_candidates": True,
        "candidate_feature_bias_shared": True,
        "absolute_source_index_parameters": False,
        "source_id_embedding_consumed": False,
        "candidate_pooling": "hard_gated_masked_mean",
        "candidate_order_contract": "permutation_equivariant_logits",
        "non_source_head_contract": "candidate_permutation_invariant",
    }
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        **unsigned,
        "contract_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def stage_edit_cut_scoring_contract(
    config: StageEditTransformerConfig,
) -> dict[str, Any]:
    """Bind the signal-preserving linear cut path to model schema v3.

    The low-capacity path sees only causal raw context and label-free,
    hard-gated candidate summaries.  The Transformer contributes an additive
    residual whose affine head is initialized to exactly zero.  This lets
    balanced cut-only pretraining recover the training-only weighted-linear
    signal before the high-capacity trunk can overwrite it.
    """

    linear_input_dimension = (
        config.context_input_dimension + 2 * config.source_input_dimension
    )
    unsigned: dict[str, Any] = {
        "contract": STAGE_EDIT_CUT_SCORING_CONTRACT,
        "token_schema_fingerprint": config.token_schema_fingerprint,
        "cut_logit_mode": config.cut_logit_mode,
        "linear_input_dimension": linear_input_dimension,
        "linear_features": [
            "encoded_raw_context_with_missing_bits",
            "hard_gated_candidate_set_mean",
            "hard_gated_candidate_set_max",
        ],
        "candidate_summary_uses_source_ids": False,
        "candidate_summary_uses_reference_labels": False,
        "transformer_contribution": "additive_residual",
        "transformer_residual_affine_initialization": "exact_zero",
        "candidate_order_contract": "permutation_invariant",
    }
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        **unsigned,
        "contract_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def stage_edit_cut_imbalance_contract(
    *,
    positive_count: int,
    negative_count: int,
    maximum_positive_weight: float = 8.0,
) -> dict[str, Any]:
    """Bind a bounded positive weight to training-partition labels only."""

    for value, name in (
        (positive_count, "positive_count"),
        (negative_count, "negative_count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"cut imbalance {name} must be a positive integer")
    cap = float(maximum_positive_weight)
    if not math.isfinite(cap) or cap < 1.0:
        raise ValueError("maximum cut positive weight must be finite and >= 1")
    raw_ratio = negative_count / positive_count
    applied = min(max(raw_ratio, 1.0), cap)
    unsigned: dict[str, Any] = {
        "contract": STAGE_EDIT_CUT_IMBALANCE_CONTRACT,
        "scope": "training_partition_only_excluding_reference_validation",
        "positive_count": positive_count,
        "negative_count": negative_count,
        "raw_negative_to_positive_ratio": raw_ratio,
        "minimum_positive_weight": 1.0,
        "maximum_positive_weight": cap,
        "applied_positive_weight": applied,
        "negative_weight": 1.0,
        "formula": "clip(training_negative_count/training_positive_count,1,cap)",
        "reference_validation_labels_consumed": False,
        "heldout_labels_consumed": False,
    }
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        **unsigned,
        "contract_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def stage_edit_parameter_count(config: StageEditTransformerConfig) -> int:
    """Exact parameter count for this architecture without allocating it."""

    dim = config.model_dimension
    mlp = config.feed_forward_dimension

    def linear(input_dim: int, output_dim: int, *, bias: bool = True) -> int:
        return input_dim * output_dim + (output_dim if bias else 0)

    count = 0
    count += linear(config.context_input_dimension, dim)
    relational_source_dimension = 4 * config.source_input_dimension
    count += linear(relational_source_dimension, dim)
    count += linear(2 * dim, dim)
    count += config.max_sequence_length * dim
    # MLX TransformerEncoderLayer: bias-free Q/K/V/out, two biased MLP
    # linears, and two affine LayerNorms. TransformerEncoder adds final LN.
    per_layer = 4 * dim * dim
    per_layer += linear(dim, mlp) + linear(mlp, dim)
    per_layer += 4 * dim
    count += config.num_layers * per_layer
    count += 2 * dim
    count += linear(dim, dim)  # source query
    count += linear(dim, dim)  # source key
    count += linear(relational_source_dimension, 1)
    count += linear(
        config.context_input_dimension + 2 * config.source_input_dimension,
        1,
    )  # raw-context + hard-gated mean/max linear cut path
    count += linear(dim, 1)  # cut timing
    count += linear(dim, config.duration_bucket_count)
    count += linear(dim, config.transition_count)
    count += linear(dim, 4)  # crop zoom/x/y/roll
    return count


@dataclass(frozen=True, slots=True)
class StageEditTensorBatch:
    context_features: Any
    source_features: Any
    sequence_mask: Any
    source_candidate_mask: Any
    hard_source_gate_mask: Any
    source_ids: tuple[tuple[tuple[str | None, ...], ...], ...]
    sequence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StageEditSupervisionBatch:
    cut_targets: Any
    cut_mask: Any
    source_targets: Any
    source_mask: Any
    rejected_source_targets: Any
    pairwise_source_mask: Any
    duration_targets: Any
    duration_mask: Any
    transition_targets: Any
    transition_mask: Any
    crop_targets: Any
    crop_mask: Any


def _require_mlx() -> None:
    if mx is None or nn is None:
        detail = f": {_MLX_IMPORT_ERROR}" if _MLX_IMPORT_ERROR else ""
        raise RuntimeError(
            "StageEditTransformer requires optional dependency "
            f"ive-crosscut[stage-edit-mlx]{detail}"
        )


def mlx_runtime_version() -> str | None:
    try:
        return importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        return None


def _normalized_values(
    values: Sequence[float],
    missing_mask: Sequence[bool],
    expected: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if len(values) != expected:
        raise ValueError(f"{name} has {len(values)} values; expected {expected}")
    missing = (
        np.zeros(expected, dtype=np.float32)
        if not missing_mask
        else np.asarray(tuple(bool(value) for value in missing_mask), dtype=np.float32)
    )
    if missing.shape != (expected,):
        raise ValueError(f"{name} missing mask must have {expected} values")
    raw = np.asarray(values, dtype=np.float32)
    invalid = ~np.isfinite(raw)
    if np.any(invalid & (missing == 0.0)):
        raise ValueError(f"{name} has a non-finite unmasked value")
    return np.where(missing > 0.0, 0.0, raw), missing


def tensorize_stage_edit_batch(
    sequences: Sequence[StageEditSequence],
    schema: StageEditTokenSchema,
    config: StageEditTransformerConfig,
    *,
    device: Any = None,
) -> StageEditTensorBatch:
    """Tensorize structured features while preserving candidate identities."""

    _require_mlx()
    if config.token_schema_fingerprint != schema.fingerprint:
        raise ValueError("model config and token schema fingerprints differ")
    if config.context_input_dimension != schema.encoded_context_dimension:
        raise ValueError("model context dimension does not match token schema")
    if config.source_input_dimension != schema.encoded_source_dimension:
        raise ValueError("model source dimension does not match token schema")
    if not sequences:
        raise ValueError("at least one StageEdit sequence is required")
    batch_size = len(sequences)
    max_length = max(len(sequence.tokens) for sequence in sequences)
    if max_length == 0:
        raise ValueError("StageEdit sequences cannot be empty")
    if max_length > config.max_sequence_length:
        raise ValueError(
            f"sequence length {max_length} exceeds {config.max_sequence_length}"
        )

    context = np.zeros(
        (batch_size, max_length, config.context_input_dimension), dtype=np.float32
    )
    sources = np.zeros(
        (
            batch_size,
            max_length,
            config.max_sources,
            config.source_input_dimension,
        ),
        dtype=np.float32,
    )
    sequence_mask = np.zeros((batch_size, max_length), dtype=bool)
    candidate_mask = np.zeros(
        (batch_size, max_length, config.max_sources), dtype=bool
    )
    hard_gate_mask = np.zeros_like(candidate_mask)
    ids: list[list[tuple[str | None, ...]]] = []

    for batch_index, sequence in enumerate(sequences):
        if not sequence.sequence_id.strip():
            raise ValueError("sequence_id cannot be empty")
        previous_time = -math.inf
        sequence_ids: list[tuple[str | None, ...]] = []
        for time_index, token in enumerate(sequence.tokens):
            timestamp = float(token.master_time_seconds)
            if not math.isfinite(timestamp) or timestamp < previous_time:
                raise ValueError("master times must be finite and non-decreasing")
            previous_time = timestamp
            if len(token.source_candidates) > config.max_sources:
                raise ValueError(
                    f"token has {len(token.source_candidates)} sources; "
                    f"maximum is {config.max_sources}"
                )
            base_values = (
                tuple(token.music_features)
                + tuple(token.section_features)
                + tuple(token.recent_edit_features)
                + tuple(token.style_features)
            )
            base, missing = _normalized_values(
                base_values,
                token.missing_mask,
                schema.raw_context_dimension,
                "context features",
            )
            context[batch_index, time_index] = np.concatenate((base, missing))
            sequence_mask[batch_index, time_index] = True

            observed_ids: set[str] = set()
            slot_ids: list[str | None] = [None] * config.max_sources
            for source_index, candidate in enumerate(token.source_candidates):
                source_id = candidate.source_id.strip()
                if not source_id or source_id in observed_ids:
                    raise ValueError("candidate source ids must be non-empty and unique")
                observed_ids.add(source_id)
                source_values, source_missing = _normalized_values(
                    candidate.features,
                    candidate.missing_mask,
                    schema.raw_source_dimension,
                    f"source features for {source_id}",
                )
                sources[batch_index, time_index, source_index] = np.concatenate(
                    (source_values, source_missing)
                )
                slot_ids[source_index] = source_id
                candidate_mask[batch_index, time_index, source_index] = bool(
                    candidate.available
                )
                hard_gate_mask[batch_index, time_index, source_index] = bool(
                    candidate.available and candidate.hard_gate_passed
                )
            sequence_ids.append(tuple(slot_ids))
        sequence_ids.extend(
            [tuple([None] * config.max_sources)]
            * (max_length - len(sequence.tokens))
        )
        ids.append(sequence_ids)

    stream = device if device is not None else mx.default_device()
    with mx.stream(stream):
        return StageEditTensorBatch(
            context_features=mx.array(context),
            source_features=mx.array(sources),
            sequence_mask=mx.array(sequence_mask),
            source_candidate_mask=mx.array(candidate_mask),
            hard_source_gate_mask=mx.array(hard_gate_mask),
            source_ids=tuple(tuple(value) for value in ids),
            sequence_ids=tuple(sequence.sequence_id for sequence in sequences),
        )


def tensorize_stage_edit_supervision(
    target_sequences: Sequence[StageEditTargetSequence],
    batch: StageEditTensorBatch,
    schema: StageEditTokenSchema,
    *,
    device: Any = None,
) -> StageEditSupervisionBatch:
    """Turn sparse targets into explicit loss masks aligned with a batch."""

    _require_mlx()
    targets_by_id = {sequence.sequence_id: sequence for sequence in target_sequences}
    if len(targets_by_id) != len(target_sequences):
        raise ValueError("target sequence ids must be unique")
    batch_size, max_length, max_sources = batch.source_candidate_mask.shape
    max_rejections = max(
        1,
        max(
            (
                len(target.rejected_source_ids)
                for sequence in target_sequences
                for target in sequence.targets
            ),
            default=0,
        ),
    )

    cut_targets = np.zeros((batch_size, max_length), dtype=np.float32)
    cut_mask = np.zeros_like(cut_targets, dtype=bool)
    source_targets = np.zeros((batch_size, max_length), dtype=np.int32)
    source_mask = np.zeros_like(source_targets, dtype=bool)
    rejected_targets = np.zeros(
        (batch_size, max_length, max_rejections), dtype=np.int32
    )
    pairwise_mask = np.zeros_like(rejected_targets, dtype=bool)
    duration_targets = np.zeros((batch_size, max_length), dtype=np.int32)
    duration_mask = np.zeros_like(duration_targets, dtype=bool)
    transition_targets = np.zeros((batch_size, max_length), dtype=np.int32)
    transition_mask = np.zeros_like(transition_targets, dtype=bool)
    crop_targets = np.zeros((batch_size, max_length, 4), dtype=np.float32)
    crop_mask = np.zeros_like(crop_targets, dtype=bool)
    transition_to_index = {
        label: index for index, label in enumerate(schema.transition_labels)
    }
    candidate_mask_np = np.asarray(batch.source_candidate_mask)
    gate_mask_np = np.asarray(batch.hard_source_gate_mask)

    for batch_index, sequence_id in enumerate(batch.sequence_ids):
        target_sequence = targets_by_id.get(sequence_id)
        if target_sequence is None:
            continue
        valid_length = int(np.asarray(batch.sequence_mask[batch_index]).sum())
        if len(target_sequence.targets) != valid_length:
            raise ValueError(
                f"target count for {sequence_id} does not match token count"
            )
        for time_index, target in enumerate(target_sequence.targets):
            if target.cut is not None:
                value = float(target.cut)
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError("cut targets must be in [0, 1]")
                cut_targets[batch_index, time_index] = value
                cut_mask[batch_index, time_index] = True

            slots = batch.source_ids[batch_index][time_index]
            source_to_slot = {
                source_id: index
                for index, source_id in enumerate(slots)
                if source_id is not None
            }
            if target.selected_source_id is not None:
                selected_slot = source_to_slot.get(target.selected_source_id)
                if selected_slot is None:
                    raise ValueError("selected source is absent from candidate slots")
                if not (
                    candidate_mask_np[batch_index, time_index, selected_slot]
                    and gate_mask_np[batch_index, time_index, selected_slot]
                ):
                    raise ValueError("selected source is unavailable or hard-gated")
                source_targets[batch_index, time_index] = selected_slot
                source_mask[batch_index, time_index] = True
                for pair_index, rejected_id in enumerate(
                    target.rejected_source_ids
                ):
                    rejected_slot = source_to_slot.get(rejected_id)
                    if rejected_slot is None:
                        raise ValueError("rejected source is absent from candidate slots")
                    if rejected_slot == selected_slot:
                        raise ValueError("selected source cannot also be rejected")
                    if not gate_mask_np[batch_index, time_index, rejected_slot]:
                        # A hard-gated negative is already excluded and carries no
                        # meaningful ranking gradient.
                        continue
                    rejected_targets[batch_index, time_index, pair_index] = (
                        rejected_slot
                    )
                    pairwise_mask[batch_index, time_index, pair_index] = True
            elif target.rejected_source_ids:
                raise ValueError("pairwise negatives require a selected source")

            if target.duration_bucket is not None:
                index = int(target.duration_bucket)
                if not 0 <= index < schema.duration_bucket_count:
                    raise ValueError("duration bucket target is out of range")
                duration_targets[batch_index, time_index] = index
                duration_mask[batch_index, time_index] = True
            if target.transition_label is not None:
                if target.transition_label not in transition_to_index:
                    raise ValueError("unknown transition target")
                transition_targets[batch_index, time_index] = transition_to_index[
                    target.transition_label
                ]
                transition_mask[batch_index, time_index] = True
            if target.crop is not None:
                if len(target.crop) != 4:
                    raise ValueError("crop target must have four coordinates")
                for crop_index, value in enumerate(target.crop):
                    if value is None:
                        continue
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError("crop target values must be finite")
                    crop_targets[batch_index, time_index, crop_index] = number
                    crop_mask[batch_index, time_index, crop_index] = True

    stream = device if device is not None else mx.default_device()
    with mx.stream(stream):
        return StageEditSupervisionBatch(
            cut_targets=mx.array(cut_targets),
            cut_mask=mx.array(cut_mask),
            source_targets=mx.array(source_targets),
            source_mask=mx.array(source_mask),
            rejected_source_targets=mx.array(rejected_targets),
            pairwise_source_mask=mx.array(pairwise_mask),
            duration_targets=mx.array(duration_targets),
            duration_mask=mx.array(duration_mask),
            transition_targets=mx.array(transition_targets),
            transition_mask=mx.array(transition_mask),
            crop_targets=mx.array(crop_targets),
            crop_mask=mx.array(crop_mask),
        )


def _finite_optional(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _reference_boundary_event_type(labels: Mapping[str, Any]) -> str:
    cut_labels = labels.get("cut")
    boundary_label = (
        str(cut_labels.get("boundary_label") or "").strip()
        if isinstance(cut_labels, Mapping)
        else ""
    )
    legacy_map = {
        "reference_start": REFERENCE_START_EVENT,
        "same_source_internal_camera_cut": NATIVE_SOURCE_CAMERA_CUT_EVENT,
        "cross_source_hard_cut": EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT,
        "cross_source_transition_unresolved": (
            EDITORIAL_SOURCE_SWITCH_UNRESOLVED_EVENT
        ),
        "known_after_unknown": KNOWN_AFTER_UNKNOWN_EVENT,
    }
    expected = legacy_map.get(boundary_label)
    if expected is None:
        raise ValueError("reference supervision boundary label is unsupported")
    raw_event = labels.get("boundary_event")
    if raw_event is None:
        return expected
    if not isinstance(raw_event, Mapping):
        raise ValueError("reference supervision boundary_event must be an object")
    if raw_event.get("schema_version") != BOUNDARY_EVENT_SCHEMA_VERSION:
        raise ValueError("unsupported reference boundary_event schema")
    event_type = str(raw_event.get("event_type") or "").strip()
    if event_type != expected:
        raise ValueError("reference boundary_event conflicts with boundary label")
    is_native = event_type == NATIVE_SOURCE_CAMERA_CUT_EVENT
    is_editorial = event_type in {
        EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT,
        EDITORIAL_SOURCE_SWITCH_UNRESOLVED_EVENT,
    }
    if bool(raw_event.get("is_native_source_camera_cut")) != is_native:
        raise ValueError("reference boundary_event native flag is inconsistent")
    if bool(raw_event.get("is_editorial_source_switch")) != is_editorial:
        raise ValueError("reference boundary_event editorial flag is inconsistent")
    return event_type


def targets_from_reference_supervision(
    payload: Mapping[str, Any],
    sequence: StageEditSequence,
    schema: StageEditTokenSchema,
    *,
    shot_index_to_token_index: Mapping[int, int] | None = None,
) -> StageEditTargetSequence:
    """Adapt conservative ``reference_supervision`` JSON to sparse targets.

    The adapter never invents unresolved labels.  Source targets come only
    from listwise groups with observed negatives, transition targets require
    ``supervision_eligible``, and each crop coordinate is masked independently.
    A native camera change inside one broadcast source is an explicit
    editorial no-switch target, never a cut/source-selection positive.  Shot-
    level data otherwise contains no sampled hold negatives; callers should
    add independently verified no-cut targets later.
    """

    if payload.get("dataset_kind") != "reference_stage_mix_pairwise_listwise_supervision":
        raise ValueError("unsupported reference supervision dataset")
    mapping = dict(shot_index_to_token_index or {})
    target_values = [StageEditTarget() for _ in sequence.tokens]

    def token_index(record: Mapping[str, Any]) -> int | None:
        raw = record.get("shot_index")
        if isinstance(raw, bool) or not isinstance(raw, int):
            return None
        return mapping.get(raw, raw)

    listwise_by_token: dict[int, Mapping[str, Any]] = {}
    for record in payload.get("listwise_records", []):
        if not isinstance(record, Mapping):
            continue
        index = token_index(record)
        if index is None or not 0 <= index < len(target_values):
            continue
        listwise_by_token[index] = record

    for record in payload.get("shot_labels", []):
        if not isinstance(record, Mapping):
            continue
        index = token_index(record)
        if index is None or not 0 <= index < len(target_values):
            continue
        labels = record.get("edit_labels")
        if not isinstance(labels, Mapping):
            continue
        event_type = _reference_boundary_event_type(labels)
        is_editorial_hard_cut = (
            event_type == EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT
        )
        ranking = listwise_by_token.get(index)
        selected: str | None = None
        rejected: tuple[str, ...] = ()
        if is_editorial_hard_cut and ranking is not None:
            raw_selected = str(ranking.get("chosen_source_id") or "").strip()
            candidates = ranking.get("candidates")
            if raw_selected and isinstance(candidates, list):
                selected = raw_selected
                rejected = tuple(
                    str(candidate.get("source_id")).strip()
                    for candidate in candidates
                    if isinstance(candidate, Mapping)
                    and not bool(candidate.get("selected"))
                    and str(candidate.get("source_id") or "").strip()
                )

        cut_labels = labels.get("cut")
        cut: float | None = None
        duration_bucket: int | None = None
        if isinstance(cut_labels, Mapping):
            if is_editorial_hard_cut:
                cut = 1.0
            elif event_type == NATIVE_SOURCE_CAMERA_CUT_EVENT:
                cut = 0.0
            duration = _finite_optional(cut_labels.get("duration_seconds"))
            if (
                is_editorial_hard_cut
                and duration is not None
                and duration >= 0.0
            ):
                duration_bucket = schema.duration_bucket(duration)

        transition_label: str | None = None
        transition = labels.get("source_transition")
        if (
            is_editorial_hard_cut
            and isinstance(transition, Mapping)
            and bool(transition.get("supervision_eligible"))
        ):
            raw_transition = str(transition.get("transition") or "").strip()
            if raw_transition in schema.transition_labels:
                transition_label = raw_transition

        crop_target: tuple[float | None, float | None, float | None, float | None] | None = None
        crop = labels.get("crop")
        if isinstance(crop, Mapping) and bool(crop.get("supervision_eligible")):
            transform = crop.get("transform")
            if isinstance(transform, Mapping):
                values = (
                    _finite_optional(transform.get("zoom", transform.get("scale"))),
                    _finite_optional(transform.get("center_x", transform.get("x"))),
                    _finite_optional(transform.get("center_y", transform.get("y"))),
                    _finite_optional(
                        transform.get("roll_degrees", transform.get("roll"))
                    ),
                )
                if any(value is not None for value in values):
                    crop_target = values

        target_values[index] = StageEditTarget(
            cut=cut,
            selected_source_id=selected,
            rejected_source_ids=rejected,
            duration_bucket=duration_bucket,
            transition_label=transition_label,
            crop=crop_target,
        )
    return StageEditTargetSequence(sequence.sequence_id, tuple(target_values))


@dataclass(frozen=True, slots=True)
class StageEditModelOutput:
    cut_timing_logits: Any
    source_selection_logits: Any
    duration_bucket_logits: Any
    transition_logits: Any
    crop: Any
    effective_source_mask: Any


_ModuleBase = nn.Module if nn is not None else object


class StageEditTransformer(_ModuleBase):
    """Causal structured-feature Transformer with independent edit heads."""

    def __init__(self, config: StageEditTransformerConfig):
        _require_mlx()
        super().__init__()
        self.config = config
        dim = config.model_dimension
        relational_source_dimension = 4 * config.source_input_dimension
        self.context_projection = nn.Linear(config.context_input_dimension, dim)
        self.source_projection = nn.Linear(relational_source_dimension, dim)
        self.input_fusion = nn.Linear(2 * dim, dim)
        self.position_embedding = nn.Embedding(config.max_sequence_length, dim)
        self.input_dropout = nn.Dropout(config.dropout)
        self.encoder = nn.TransformerEncoder(
            num_layers=config.num_layers,
            dims=dim,
            num_heads=config.num_heads,
            mlp_dims=config.feed_forward_dimension,
            dropout=config.dropout,
            norm_first=True,
        )
        self.source_query = nn.Linear(dim, dim)
        self.source_key = nn.Linear(dim, dim)
        self.source_bias = nn.Linear(relational_source_dimension, 1)
        self.cut_linear_baseline = nn.Linear(
            config.context_input_dimension + 2 * config.source_input_dimension,
            1,
        )
        self.cut_timing_head = nn.Linear(dim, 1)
        # Schema v3 treats the Transformer cut contribution as a residual over
        # the deliberately low-capacity raw-feature path.  Zero initialization
        # is part of the checkpoint contract, not a best-effort training hint.
        self.cut_timing_head.weight = mx.zeros_like(self.cut_timing_head.weight)
        self.cut_timing_head.bias = mx.zeros_like(self.cut_timing_head.bias)
        self.duration_head = nn.Linear(dim, config.duration_bucket_count)
        self.transition_head = nn.Linear(dim, config.transition_count)
        self.crop_head = nn.Linear(dim, 4)

    def _relational_source_features(
        self,
        batch: StageEditTensorBatch,
        effective_mask: Any,
    ) -> Any:
        """Encode each candidate relative to its timestamp candidate set.

        Raw descriptors alone can reverse editorial meaning between songs
        (for example, a larger face can be either the intended close-up or an
        unusable crop).  Set mean, residual, and a bounded within-set z-score
        expose relative geometry without introducing an absolute source slot.
        All set statistics exclude unavailable or hard-gated candidates.
        """

        raw = batch.source_features
        weights = effective_mask.astype(raw.dtype)[..., None]
        count = mx.maximum(weights.sum(axis=2), 1.0)
        mean = (raw * weights).sum(axis=2) / count
        residual = raw - mean[..., None, :]
        variance = ((residual * residual) * weights).sum(axis=2) / count
        standard_deviation = mx.sqrt(mx.maximum(variance, 1.0e-6))
        bounded_zscore = mx.tanh(
            residual / (3.0 * standard_deviation[..., None, :])
        )
        broadcast_mean = mx.broadcast_to(mean[..., None, :], raw.shape)
        return mx.concatenate(
            (raw, broadcast_mean, residual, bounded_zscore), axis=-1
        )

    def _hidden_and_sources(
        self, batch: StageEditTensorBatch
    ) -> tuple[Any, Any, Any, Any]:
        """Return shared hidden/candidate representations for all heads."""

        length = batch.context_features.shape[1]
        if length > self.config.max_sequence_length:
            raise ValueError("batch exceeds configured context window")
        effective_mask = mx.logical_and(
            batch.source_candidate_mask, batch.hard_source_gate_mask
        )
        relational_source_features = self._relational_source_features(
            batch, effective_mask
        )
        source_embeddings = self.source_projection(relational_source_features)
        # A hard-rejected source must be invisible to every head, not merely
        # receive a masked source-selection logit.  Otherwise its features can
        # leak through pooled_sources and change cut, duration, transition,
        # crop, and even the logits of valid sources.
        candidate_weights = effective_mask.astype(
            source_embeddings.dtype
        )[..., None]
        candidate_sum = (source_embeddings * candidate_weights).sum(axis=2)
        denominator = mx.maximum(candidate_weights.sum(axis=2), 1.0)
        pooled_sources = candidate_sum / denominator
        context = self.context_projection(batch.context_features)
        hidden = self.input_fusion(mx.concatenate((context, pooled_sources), axis=-1))
        positions = mx.arange(length)
        hidden = hidden + self.position_embedding(positions)[None, :, :]
        hidden = self.input_dropout(hidden)

        causal = nn.MultiHeadAttention.create_additive_causal_mask(
            length, dtype=hidden.dtype
        )[None, None, :, :]
        key_padding = mx.where(
            batch.sequence_mask[:, None, None, :],
            mx.array(0.0, dtype=hidden.dtype),
            mx.array(MASKED_SOURCE_LOGIT, dtype=hidden.dtype),
        )
        hidden = self.encoder(hidden, causal + key_padding)
        hidden = hidden * batch.sequence_mask[..., None].astype(hidden.dtype)
        return (
            hidden,
            relational_source_features,
            source_embeddings,
            effective_mask,
        )

    def _source_logits(
        self,
        hidden: Any,
        relational_source_features: Any,
        source_embeddings: Any,
        effective_mask: Any,
    ) -> Any:
        """Score candidates with shared weights and no slot-index parameters."""

        query = self.source_query(hidden)[..., None, :]
        keys = self.source_key(source_embeddings)
        source_logits = (query * keys).sum(axis=-1) / math.sqrt(
            self.config.model_dimension
        )
        source_logits = source_logits + self.source_bias(
            relational_source_features
        ).squeeze(-1)
        return mx.where(
            effective_mask,
            source_logits,
            mx.array(MASKED_SOURCE_LOGIT, dtype=source_logits.dtype),
        )

    def _cut_candidate_summary(
        self,
        batch: StageEditTensorBatch,
        effective_mask: Any,
    ) -> Any:
        """Return permutation-invariant hard-gated mean/max descriptors."""

        raw = batch.source_features
        weights = effective_mask.astype(raw.dtype)[..., None]
        count = weights.sum(axis=2)
        mean = (raw * weights).sum(axis=2) / mx.maximum(count, 1.0)
        masked_for_max = mx.where(
            effective_mask[..., None],
            raw,
            mx.array(-1.0e9, dtype=raw.dtype),
        )
        maximum = masked_for_max.max(axis=2)
        maximum = mx.where(count > 0.0, maximum, mx.zeros_like(maximum))
        return mx.concatenate((mean, maximum), axis=-1)

    def _cut_logits(
        self,
        hidden: Any,
        batch: StageEditTensorBatch,
        effective_mask: Any,
    ) -> Any:
        """Add a zero-init Transformer residual to the raw linear cut path."""

        summary = self._cut_candidate_summary(batch, effective_mask)
        linear_inputs = mx.concatenate((batch.context_features, summary), axis=-1)
        baseline = self.cut_linear_baseline(linear_inputs).squeeze(-1)
        residual = self.cut_timing_head(hidden).squeeze(-1)
        return baseline + residual

    def __call__(self, batch: StageEditTensorBatch) -> StageEditModelOutput:
        (
            hidden,
            relational_source_features,
            source_embeddings,
            effective_mask,
        ) = self._hidden_and_sources(batch)
        source_logits = self._source_logits(
            hidden,
            relational_source_features,
            source_embeddings,
            effective_mask,
        )
        return StageEditModelOutput(
            cut_timing_logits=self._cut_logits(hidden, batch, effective_mask),
            source_selection_logits=source_logits,
            duration_bucket_logits=self.duration_head(hidden),
            transition_logits=self.transition_head(hidden),
            crop=self.crop_head(hidden),
            effective_source_mask=effective_mask,
        )


@dataclass(frozen=True, slots=True)
class StageEditLossContract:
    source_objective: str = "hybrid"
    pairwise_margin: float = 0.2
    cut_weight: float = 1.0
    cut_positive_weight: float = 1.0
    source_listwise_weight: float = 1.0
    source_pairwise_weight: float = 0.5
    duration_weight: float = 0.5
    transition_weight: float = 0.5
    crop_weight: float = 0.25
    transition_authority: str = LEARNED_TRANSITION_AUTHORITY

    def __post_init__(self) -> None:
        if self.source_objective not in {
            "listwise",
            "pairwise",
            "hybrid",
            "eligible_all_pairwise",
            "eligible_all_hybrid",
        }:
            raise ValueError(
                "source objective must be listwise, pairwise, hybrid, "
                "eligible_all_pairwise, or eligible_all_hybrid"
            )
        if self.pairwise_margin < 0.0:
            raise ValueError("pairwise margin must be non-negative")
        weights = (
            self.cut_weight,
            self.source_listwise_weight,
            self.source_pairwise_weight,
            self.duration_weight,
            self.transition_weight,
            self.crop_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("loss weights must be finite and non-negative")
        if (
            not math.isfinite(self.cut_positive_weight)
            or self.cut_positive_weight < 1.0
        ):
            raise ValueError("cut positive weight must be finite and >= 1")
        authority = StageEditTransitionAuthority(mode=self.transition_authority)
        if authority.renderer_owned and self.transition_weight != 0.0:
            raise ValueError(
                "renderer-owned transition authority requires transition_weight=0"
            )


def _masked_mean(values: Any, mask: Any) -> Any:
    weights = mask.astype(values.dtype)
    return (values * weights).sum() / mx.maximum(weights.sum(), 1.0)


def _reference_balanced_masked_mean(values: Any, mask: Any) -> Any:
    """Average source groups within each reference, then references equally."""

    weights = mask.astype(values.dtype)
    per_reference_count = weights.sum(axis=1)
    per_reference = (values * weights).sum(axis=1) / mx.maximum(
        per_reference_count, 1.0
    )
    reference_mask = per_reference_count > 0
    return _masked_mean(per_reference, reference_mask)


def stage_edit_loss(
    model: StageEditTransformer,
    batch: StageEditTensorBatch,
    supervision: StageEditSupervisionBatch,
    contract: StageEditLossContract = StageEditLossContract(),
) -> tuple[Any, dict[str, Any]]:
    """Compute the complete masked multi-head training contract."""

    _require_mlx()
    output = model(batch)
    cut_values = nn.losses.binary_cross_entropy(
        output.cut_timing_logits,
        supervision.cut_targets,
        with_logits=True,
        reduction="none",
    )
    # Editorial switch positives are substantially rarer than verified holds.
    # The weight is derived once from the training partition by the production
    # trainer and is deliberately bounded.  Validation/held-out labels never
    # enter this tensor, and the denominator remains the labeled-token count,
    # matching the conventional BCE-with-logits ``pos_weight`` contract.
    cut_values = cut_values * mx.where(
        supervision.cut_targets >= 0.5,
        mx.array(contract.cut_positive_weight, dtype=cut_values.dtype),
        mx.array(1.0, dtype=cut_values.dtype),
    )
    cut_loss = _masked_mean(cut_values, supervision.cut_mask)
    listwise_values = nn.losses.cross_entropy(
        output.source_selection_logits,
        supervision.source_targets,
        reduction="none",
    )
    listwise_loss = _reference_balanced_masked_mean(
        listwise_values, supervision.source_mask
    )

    positive = mx.take_along_axis(
        output.source_selection_logits,
        supervision.source_targets[..., None],
        axis=-1,
    ).squeeze(-1)
    negative_indices = supervision.rejected_source_targets[..., None]
    expanded_logits = mx.broadcast_to(
        output.source_selection_logits[..., None, :],
        (*negative_indices.shape[:-1], output.source_selection_logits.shape[-1]),
    )
    negative = mx.take_along_axis(
        expanded_logits, negative_indices, axis=-1
    ).squeeze(-1)
    explicit_pairwise_values = nn.softplus(
        contract.pairwise_margin - positive[..., None] + negative
    )
    explicit_pairwise_weights = supervision.pairwise_source_mask.astype(
        explicit_pairwise_values.dtype
    )
    explicit_pairwise_count = explicit_pairwise_weights.sum(axis=-1)
    explicit_pairwise_group_loss = (
        explicit_pairwise_values * explicit_pairwise_weights
    ).sum(axis=-1) / mx.maximum(explicit_pairwise_count, 1.0)
    explicit_pairwise_group_mask = mx.logical_and(
        supervision.source_mask, explicit_pairwise_count > 0
    )
    explicit_pairwise_loss = _reference_balanced_masked_mean(
        explicit_pairwise_group_loss, explicit_pairwise_group_mask
    )

    selected_one_hot = mx.arange(output.source_selection_logits.shape[-1]) == (
        supervision.source_targets[..., None]
    )
    eligible_negative_mask = mx.logical_and(
        output.effective_source_mask,
        mx.logical_and(
            supervision.source_mask[..., None],
            mx.logical_not(selected_one_hot),
        ),
    )
    eligible_pairwise_values = nn.softplus(
        contract.pairwise_margin
        - positive[..., None]
        + output.source_selection_logits
    )
    eligible_weights = eligible_negative_mask.astype(
        eligible_pairwise_values.dtype
    )
    eligible_count = eligible_weights.sum(axis=-1)
    eligible_group_loss = (
        eligible_pairwise_values * eligible_weights
    ).sum(axis=-1) / mx.maximum(eligible_count, 1.0)
    eligible_group_mask = mx.logical_and(
        supervision.source_mask, eligible_count > 0
    )
    eligible_all_pairwise_loss = _reference_balanced_masked_mean(
        eligible_group_loss, eligible_group_mask
    )
    pairwise_loss = (
        eligible_all_pairwise_loss
        if contract.source_objective
        in {"eligible_all_pairwise", "eligible_all_hybrid"}
        else explicit_pairwise_loss
    )

    duration_values = nn.losses.cross_entropy(
        output.duration_bucket_logits,
        supervision.duration_targets,
        reduction="none",
    )
    duration_loss = _masked_mean(duration_values, supervision.duration_mask)
    transition_values = nn.losses.cross_entropy(
        output.transition_logits,
        supervision.transition_targets,
        reduction="none",
    )
    transition_loss = _masked_mean(
        transition_values, supervision.transition_mask
    )
    crop_values = nn.losses.smooth_l1_loss(
        output.crop, supervision.crop_targets, reduction="none"
    )
    crop_loss = _masked_mean(crop_values, supervision.crop_mask)

    source_listwise_weight = (
        contract.source_listwise_weight
        if contract.source_objective
        in {"listwise", "hybrid", "eligible_all_hybrid"}
        else 0.0
    )
    source_pairwise_weight = (
        contract.source_pairwise_weight
        if contract.source_objective
        in {
            "pairwise",
            "hybrid",
            "eligible_all_pairwise",
            "eligible_all_hybrid",
        }
        else 0.0
    )
    total = (
        contract.cut_weight * cut_loss
        + source_listwise_weight * listwise_loss
        + source_pairwise_weight * pairwise_loss
        + contract.duration_weight * duration_loss
        + contract.transition_weight * transition_loss
        + contract.crop_weight * crop_loss
    )
    return total, {
        "total": total,
        "cut": cut_loss,
        "source_listwise": listwise_loss,
        "source_pairwise": pairwise_loss,
        "duration": duration_loss,
        "transition": transition_loss,
        "crop": crop_loss,
    }


@dataclass(frozen=True, slots=True)
class StageEditCheckpointMetadata:
    format_version: int
    checkpoint_id: str
    created_at: str
    training_steps: int
    weights_trained: bool
    training_data_id: str | None
    token_schema: StageEditTokenSchema
    model_config: StageEditTransformerConfig
    weight_filename: str = "model.safetensors"
    weight_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.format_version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("unsupported StageEdit checkpoint format")
        if self.training_steps < 0:
            raise ValueError("training_steps cannot be negative")
        if self.weights_trained and self.training_steps == 0:
            raise ValueError("trained weights require at least one training step")
        if self.model_config.token_schema_fingerprint != self.token_schema.fingerprint:
            raise ValueError("checkpoint config and token schema differ")
        if Path(self.weight_filename).name != self.weight_filename:
            raise ValueError("weight filename must not contain a path")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "checkpoint_id": self.checkpoint_id,
            "created_at": self.created_at,
            "training_steps": self.training_steps,
            "weights_trained": self.weights_trained,
            "training_data_id": self.training_data_id,
            "token_schema": self.token_schema.to_dict(),
            "model_config": self.model_config.to_dict(),
            "weight_filename": self.weight_filename,
            "weight_sha256": self.weight_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageEditCheckpointMetadata":
        return cls(
            format_version=int(payload["format_version"]),
            checkpoint_id=str(payload["checkpoint_id"]),
            created_at=str(payload["created_at"]),
            training_steps=int(payload["training_steps"]),
            weights_trained=bool(payload["weights_trained"]),
            training_data_id=(
                str(payload["training_data_id"])
                if payload.get("training_data_id") is not None
                else None
            ),
            token_schema=StageEditTokenSchema.from_dict(payload["token_schema"]),
            model_config=StageEditTransformerConfig.from_dict(
                payload["model_config"]
            ),
            weight_filename=str(payload.get("weight_filename", "model.safetensors")),
            weight_sha256=(
                str(payload["weight_sha256"])
                if payload.get("weight_sha256") is not None
                else None
            ),
        )


def new_stage_edit_checkpoint_metadata(
    checkpoint_id: str,
    schema: StageEditTokenSchema,
    config: StageEditTransformerConfig,
    *,
    training_steps: int,
    weights_trained: bool,
    training_data_id: str | None = None,
) -> StageEditCheckpointMetadata:
    return StageEditCheckpointMetadata(
        format_version=CHECKPOINT_FORMAT_VERSION,
        checkpoint_id=checkpoint_id,
        created_at=datetime.now(UTC).isoformat(),
        training_steps=training_steps,
        weights_trained=weights_trained,
        training_data_id=training_data_id,
        token_schema=schema,
        model_config=config,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_stage_edit_checkpoint(
    directory: Path,
    model: StageEditTransformer,
    metadata: StageEditCheckpointMetadata,
) -> StageEditCheckpointMetadata:
    _require_mlx()
    if model.config != metadata.model_config:
        raise ValueError("model config does not match checkpoint metadata")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    weight_path = destination / metadata.weight_filename
    model.save_weights(str(weight_path))
    mx.eval(model.parameters())
    saved_metadata = replace(metadata, weight_sha256=_sha256_file(weight_path))
    (destination / "metadata.json").write_text(
        json.dumps(saved_metadata.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return saved_metadata


def load_stage_edit_checkpoint(
    directory: Path,
    *,
    expected_schema: StageEditTokenSchema | None = None,
    require_trained: bool = True,
) -> tuple[StageEditTransformer, StageEditCheckpointMetadata]:
    _require_mlx()
    source = Path(directory)
    payload = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint metadata root must be an object")
    metadata = StageEditCheckpointMetadata.from_dict(payload)
    if (
        expected_schema is not None
        and expected_schema.fingerprint != metadata.token_schema.fingerprint
    ):
        raise ValueError("checkpoint token schema does not match expected schema")
    if require_trained and not (
        metadata.weights_trained and metadata.training_steps > 0
    ):
        raise ValueError("checkpoint does not declare trained weights")
    weight_path = source / metadata.weight_filename
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    actual_sha256 = _sha256_file(weight_path)
    if metadata.weight_sha256 is None or actual_sha256 != metadata.weight_sha256:
        raise ValueError("checkpoint weight digest is missing or invalid")
    model = StageEditTransformer(metadata.model_config)
    model.load_weights(str(weight_path), strict=True)
    mx.eval(model.parameters())
    return model, metadata


@dataclass(frozen=True, slots=True)
class StageEditReadinessReport:
    architecture_available: bool
    learned_inference_ready: bool
    status: str
    reason: str
    mlx_version: str | None
    checkpoint_path: str | None
    training_steps: int
    checkpoint_format: str | None = None
    dynamic_crop_version: int | None = None
    checkpoint_metadata_filename: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dynamic_crop_v3_selection_gate_errors(
    value: Any,
    *,
    name: str,
) -> tuple[str, ...]:
    """Validate the exact validation-only renderer gate bound to v3 selection."""

    if not isinstance(value, Mapping):
        return (f"{name} is missing",)
    errors: list[str] = []
    if value.get("contract") != DYNAMIC_CROP_V3_VALIDATION_SELECTION_GATE_CONTRACT:
        errors.append(f"{name} contract differs")
    if value.get("selectable") is not True:
        errors.append(f"{name} is not selectable")
    if value.get("reasons") != []:
        errors.append(f"{name} contains rejection reasons")
    if value.get("heldout_metrics_consumed") is not False:
        errors.append(f"{name} consumed held-out metrics")
    if value.get("static_prior_can_be_selected") is not False:
        errors.append(f"{name} permits the static prior")

    attempt_count = value.get("renderer_decode_attempt_count")
    eligible_count = value.get("renderer_decode_eligible_count")
    eligible_rate = value.get("renderer_decode_eligible_rate")
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or attempt_count < 1
    ):
        errors.append(f"{name} renderer attempts are invalid")
    if (
        isinstance(eligible_count, bool)
        or not isinstance(eligible_count, int)
        or not isinstance(attempt_count, int)
        or isinstance(attempt_count, bool)
        or eligible_count != attempt_count
        or not isinstance(eligible_rate, (int, float))
        or isinstance(eligible_rate, bool)
        or not math.isfinite(float(eligible_rate))
        or float(eligible_rate) != 1.0
    ):
        errors.append(f"{name} renderer decode is not exactly 100 percent")
    projection_count = value.get("minimum_safe_zoom_projection_count")
    if (
        isinstance(projection_count, bool)
        or not isinstance(projection_count, int)
        or projection_count != 0
    ):
        errors.append(f"{name} renderer projection count is not zero")
    mismatch_count = value.get("target_source_slot_mismatch_count")
    if (
        isinstance(mismatch_count, bool)
        or not isinstance(mismatch_count, int)
        or mismatch_count != 0
    ):
        errors.append(f"{name} source-slot mismatch count is not zero")
    improvement = value.get(
        "aggregate_normalized_improvement_over_static_prior"
    )
    score_component = value.get("score_component")
    if (
        not isinstance(improvement, (int, float))
        or isinstance(improvement, bool)
        or not math.isfinite(float(improvement))
        or float(improvement) <= 0.0
        or not isinstance(score_component, (int, float))
        or isinstance(score_component, bool)
        or not math.isfinite(float(score_component))
        or float(score_component) != float(improvement)
    ):
        errors.append(f"{name} does not prove positive static-prior improvement")
    return tuple(errors)


def _dynamic_crop_v3_gate_metric_binding_errors(
    gate: Any,
    metrics: Any,
    *,
    metadata: Any,
    name: str,
) -> tuple[str, ...]:
    """Bind summarized gate evidence to the full evaluator payload."""

    if not isinstance(gate, Mapping) or not isinstance(metrics, Mapping):
        return (f"{name} metric binding is missing",)
    errors: list[str] = []
    expected_contracts = {
        "schema_version": 3,
        "model_contract": getattr(metadata, "model_contract", None),
        "target_contract": getattr(metadata, "target_contract", None),
        "output_contract": getattr(metadata, "output_contract", None),
        "legacy_v2_checkpoint_or_output_used": False,
        "legacy_static_crop_head_used_for_trajectory": False,
    }
    for field_name, expected in expected_contracts.items():
        if metrics.get(field_name) != expected:
            errors.append(f"{name} evaluator {field_name} differs")
    trajectory = metrics.get("trajectory")
    renderer = (
        trajectory.get("renderer_decode")
        if isinstance(trajectory, Mapping)
        else None
    )
    if not isinstance(trajectory, Mapping) or not isinstance(renderer, Mapping):
        errors.append(f"{name} evaluator trajectory is missing")
        return tuple(errors)
    bindings = {
        "renderer_decode_attempt_count": renderer.get("attempt_count"),
        "renderer_decode_eligible_count": renderer.get("eligible_count"),
        "renderer_decode_eligible_rate": renderer.get("eligible_rate"),
        "minimum_safe_zoom_projection_count": renderer.get(
            "minimum_safe_zoom_projection_count"
        ),
        "target_source_slot_mismatch_count": trajectory.get(
            "target_source_slot_mismatch_count"
        ),
        "aggregate_normalized_improvement_over_static_prior": trajectory.get(
            "aggregate_normalized_improvement_over_static_prior"
        ),
    }
    for field_name, expected in bindings.items():
        if gate.get(field_name) != expected:
            errors.append(f"{name} gate {field_name} differs from evaluator metrics")
    return tuple(errors)


def _dynamic_crop_v3_composite_selection_errors(
    value: Any,
    *,
    name: str,
) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return (f"{name} composite evidence is missing",)
    base_score = value.get("base_selection_score")
    dynamic_score = value.get("dynamic_crop_v3_score_component")
    score = value.get("score")
    numeric = all(
        isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(float(item))
        for item in (base_score, dynamic_score, score)
    )
    components = value.get("components")
    component_score = (
        components.get(
            "dynamic_crop_v3_aggregate_normalized_improvement_over_static_prior"
        )
        if isinstance(components, Mapping)
        else None
    )
    if (
        not numeric
        or float(dynamic_score) <= 0.0
        or not math.isclose(
            float(score),
            float(base_score) + float(dynamic_score),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        or component_score != dynamic_score
        or "dynamic_crop_v3_aggregate_normalized_improvement_over_static_prior"
        not in str(value.get("formula", ""))
    ):
        return (f"{name} composite score does not include v3 improvement",)
    return ()


def _production_training_report_errors(
    payload: Any,
    *,
    metadata: Any,
    actual_weight_sha256: str | None,
    checkpoint_format: str,
) -> tuple[str, ...]:
    """Validate the minimum evidence contract for learned production use."""

    if not isinstance(payload, Mapping):
        return ("production training report root must be an object",)
    errors: list[str] = []
    dynamic_crop_v2_checkpoint = checkpoint_format == "dynamic_crop_v2"
    dynamic_crop_v3_checkpoint = checkpoint_format == "dynamic_crop_v3"
    if checkpoint_format not in ("legacy", "dynamic_crop_v2", "dynamic_crop_v3"):
        return ("production checkpoint format is unsupported",)
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION
    ):
        errors.append(
            "production training report must use decision-history-safe "
            f"schema_version {PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION}"
        )
    if payload.get("run_kind") != PRODUCTION_TRAINING_RUN_KIND:
        errors.append(
            f"run_kind must be {PRODUCTION_TRAINING_RUN_KIND!r}"
        )
    if payload.get("checkpoint_scope") != PRODUCTION_CHECKPOINT_SCOPE:
        errors.append(
            f"checkpoint_scope must be {PRODUCTION_CHECKPOINT_SCOPE!r}"
        )
    if payload.get("production_ready") is not True:
        errors.append("production_ready must be explicitly true")

    expected_history_contract = stage_edit_recent_history_mask_contract(
        metadata.token_schema
    )
    if payload.get("decision_history_input_contract") != expected_history_contract:
        errors.append(
            "production decision-history input contract is missing or differs "
            "from the schema-bound fully-masked train/serve policy"
        )

    metadata_config = getattr(metadata, "model_config", None)
    if metadata_config is None:
        metadata_config = getattr(metadata, "base_model_config", None)
    expected_source_contract = (
        stage_edit_source_scoring_contract(metadata_config)
        if isinstance(metadata_config, StageEditTransformerConfig)
        else None
    )
    if (
        expected_source_contract is None
        or payload.get("source_scoring_contract") != expected_source_contract
    ):
        errors.append(
            "production source scorer is missing the schema-bound "
            "candidate-relative permutation-equivariant contract"
        )
    expected_cut_contract = (
        stage_edit_cut_scoring_contract(metadata_config)
        if isinstance(metadata_config, StageEditTransformerConfig)
        else None
    )
    if (
        expected_cut_contract is None
        or payload.get("cut_scoring_contract") != expected_cut_contract
    ):
        errors.append(
            "production cut scorer is missing the schema-v3 raw linear plus "
            "zero-init Transformer residual contract"
        )
    expected_transition_authority = (
        transition_authority_contract(metadata_config.transition_authority)
        if isinstance(metadata_config, StageEditTransformerConfig)
        else None
    )
    expected_transition_authority_sha256 = (
        StageEditTransitionAuthority(
            mode=metadata_config.transition_authority
        ).fingerprint
        if isinstance(metadata_config, StageEditTransformerConfig)
        else None
    )
    if (
        expected_transition_authority is None
        or payload.get("transition_authority_contract")
        != expected_transition_authority
        or payload.get("transition_authority_contract_sha256")
        != expected_transition_authority_sha256
    ):
        errors.append(
            "production transition authority is missing or differs from model metadata"
        )
    optimization = payload.get("optimization")
    effective_loss = (
        optimization.get("effective_loss_contract")
        if isinstance(optimization, Mapping)
        else None
    )
    if not isinstance(effective_loss, Mapping) or effective_loss.get(
        "transition_authority"
    ) != getattr(metadata_config, "transition_authority", None):
        errors.append("production model and loss transition authority differ")
    elif (
        getattr(metadata_config, "transition_authority", None)
        == RENDERER_OWNED_TRANSITION_AUTHORITY
        and effective_loss.get("transition_weight") != 0.0
    ):
        errors.append("renderer-owned production transition loss must be zero")
    effective_dynamic_loss = (
        optimization.get("effective_dynamic_crop_loss_contract")
        if isinstance(optimization, Mapping)
        else None
    )
    dynamic_base_loss = (
        effective_dynamic_loss.get("base")
        if isinstance(effective_dynamic_loss, Mapping)
        else None
    )
    if (dynamic_crop_v2_checkpoint or dynamic_crop_v3_checkpoint) and (
        not isinstance(dynamic_base_loss, Mapping)
        or dynamic_base_loss.get("transition_authority")
        != getattr(metadata_config, "transition_authority", None)
        or (
            getattr(metadata_config, "transition_authority", None)
            == RENDERER_OWNED_TRANSITION_AUTHORITY
            and dynamic_base_loss.get("transition_weight") != 0.0
        )
    ):
        errors.append(
            "production Dynamic Crop base transition authority differs from model"
        )

    imbalance = payload.get("cut_imbalance_contract")
    if not isinstance(imbalance, Mapping):
        errors.append("production cut imbalance contract is missing")
    else:
        positive_count = imbalance.get("positive_count")
        negative_count = imbalance.get("negative_count")
        maximum_weight = imbalance.get("maximum_positive_weight")
        try:
            expected_imbalance = stage_edit_cut_imbalance_contract(
                positive_count=positive_count,
                negative_count=negative_count,
                maximum_positive_weight=maximum_weight,
            )
        except (TypeError, ValueError):
            expected_imbalance = None
        if expected_imbalance is None or dict(imbalance) != expected_imbalance:
            errors.append(
                "production cut imbalance evidence is invalid or was not "
                "derived solely from the training partition"
            )

    head_coverage = payload.get("structured_head_coverage_gate")
    if not isinstance(head_coverage, Mapping):
        errors.append("production structured-head coverage gate is missing")
    else:
        if head_coverage.get("contract") != (
            "duration_learned_transition_or_renderer_metadata_coverage_gate_v2"
        ):
            errors.append("production structured-head coverage contract differs")
        if head_coverage.get("passed") is not True:
            errors.append("production structured-head coverage gate did not pass")
        if head_coverage.get("heldout_used_for_model_selection") is not False:
            errors.append("structured-head coverage leaked held-out model selection")

    gate = payload.get("production_supervision_gate")
    if not isinstance(gate, Mapping):
        errors.append("production_supervision_gate is missing")
    else:
        if gate.get("passed") is not True:
            errors.append("production supervision gate did not pass")
        if gate.get("scope") != (
            "training_songs_only_including_reference_validation"
        ):
            errors.append("production supervision gate must exclude held-out songs")
        if gate.get("held_out_references_included") is not False:
            errors.append("held-out references contaminated the supervision gate")
        observed = gate.get("observed")
        if not isinstance(observed, Mapping):
            errors.append("production supervision observations are missing")
        else:
            integer_minimums = (
                ("dense_editorial_switch_count", 1),
                ("reference_count_with_dense_editorial_switches", 2),
                ("song_count_with_dense_editorial_switches", 2),
            )
            for name, minimum in integer_minimums:
                value = observed.get(name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < minimum
                ):
                    errors.append(
                        f"production supervision {name} must be at least {minimum}"
                    )
            native_positive = observed.get(
                "native_source_camera_cut_editorial_positive_count"
            )
            if (
                isinstance(native_positive, bool)
                or not isinstance(native_positive, int)
                or native_positive != 0
            ):
                errors.append(
                    "native source camera cuts must have zero editorial positives"
                )

    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        errors.append("production checkpoint evidence is missing")
    else:
        if checkpoint.get("checkpoint_id") != metadata.checkpoint_id:
            errors.append("training report checkpoint_id does not match metadata")
        if checkpoint.get("training_steps") != metadata.training_steps:
            errors.append("training report training_steps do not match metadata")
        if (
            actual_weight_sha256 is None
            or checkpoint.get("weight_sha256") != actual_weight_sha256
        ):
            errors.append("training report weight digest does not match checkpoint")
        if checkpoint.get("production_activation_authorized") is not True:
            errors.append("production activation was not explicitly authorized")
        if checkpoint.get("production_supervision_gate_passed") is not True:
            errors.append("checkpoint does not bind a passed supervision gate")
        if bool(checkpoint.get("dynamic_crop_v2_checkpoint")) != (
            dynamic_crop_v2_checkpoint
        ):
            errors.append("training report dynamic-crop checkpoint kind mismatches")
        if dynamic_crop_v3_checkpoint:
            expected_checkpoint_fields = {
                "checkpoint_format": "dynamic_crop_v3",
                "dynamic_crop_version": 3,
                "metadata_filename": "dynamic-crop-v3-metadata.json",
                "dynamic_crop_v2_checkpoint": False,
                "dynamic_crop_v3_checkpoint": True,
                "dynamic_crop_model_contract": getattr(
                    metadata, "model_contract", None
                ),
                "dynamic_crop_target_contract": getattr(
                    metadata, "target_contract", None
                ),
                "dynamic_crop_output_contract": getattr(
                    metadata, "output_contract", None
                ),
            }
            for field_name, expected in expected_checkpoint_fields.items():
                if checkpoint.get(field_name) != expected:
                    errors.append(
                        f"production checkpoint {field_name} does not declare v3"
                    )
            data_id = checkpoint.get("dynamic_crop_v3_data_id")
            if not isinstance(data_id, str) or not data_id.strip():
                errors.append(
                    "production checkpoint dynamic-crop v3 data id is missing"
                )

    model = payload.get("model")
    if not isinstance(model, Mapping):
        errors.append("production model evidence is missing")
    elif dynamic_crop_v3_checkpoint:
        expected_model_fields = {
            "architecture": DYNAMIC_CROP_V3_PRODUCTION_ARCHITECTURE,
            "dynamic_crop_version": 3,
            "dynamic_crop_v2_enabled": False,
            "dynamic_crop_v3_enabled": True,
            "dynamic_crop_model_contract": getattr(
                metadata, "model_contract", None
            ),
            "dynamic_crop_target_contract": getattr(
                metadata, "target_contract", None
            ),
            "dynamic_crop_output_contract": getattr(
                metadata, "output_contract", None
            ),
        }
        for field_name, expected in expected_model_fields.items():
            if model.get(field_name) != expected:
                errors.append(f"production model {field_name} does not declare v3")
        if not valid_dynamic_crop_v3_activation_checkpointing_evidence(
            model.get("activation_checkpointing")
        ):
            errors.append(
                "production model activation checkpointing evidence differs"
            )
    elif dynamic_crop_v2_checkpoint:
        if model.get("dynamic_crop_v2_enabled") is not True:
            errors.append("dynamic-crop v2 model activation evidence is missing")
        if model.get("architecture") != (
            "causal_long_context_transformer_with_k8_dynamic_crop_v2"
        ):
            errors.append("production model is not the K=8 dynamic-crop v2 architecture")

    data = payload.get("data")
    if dynamic_crop_v3_checkpoint:
        if not isinstance(data, Mapping):
            errors.append("production dynamic-crop v3 corpus evidence is missing")
        else:
            dynamic_v3_data = data.get("dynamic_crop_v3")
            if data.get("dynamic_crop_version") != 3:
                errors.append("production corpus does not declare dynamic-crop v3")
            if data.get("dynamic_crop_v2") is not None:
                errors.append("production v3 corpus evidence contains a v2 corpus")
            if not isinstance(dynamic_v3_data, Mapping):
                errors.append("production dynamic-crop v3 corpus manifest is missing")
            else:
                expected_data_id = (
                    checkpoint.get("dynamic_crop_v3_data_id")
                    if isinstance(checkpoint, Mapping)
                    else None
                )
                if (
                    not isinstance(dynamic_v3_data.get("data_id"), str)
                    or not dynamic_v3_data.get("data_id", "").strip()
                    or dynamic_v3_data.get("data_id") != expected_data_id
                ):
                    errors.append("production dynamic-crop v3 corpus data id differs")
                if (
                    not isinstance(dynamic_v3_data.get("source_v2_data_id"), str)
                    or not dynamic_v3_data.get("source_v2_data_id", "").strip()
                ):
                    errors.append(
                        "production dynamic-crop v3 source-v2 corpus id is missing"
                    )
                if dynamic_v3_data.get("contract") != (
                    DYNAMIC_CROP_V3_TRAINING_CORPUS_CONTRACT
                ):
                    errors.append("production dynamic-crop v3 corpus contract differs")
                if dynamic_v3_data.get("target_contract") != getattr(
                    metadata, "target_contract", None
                ):
                    errors.append("production dynamic-crop v3 target contract differs")
                manifest_sha256 = dynamic_v3_data.get("manifest_sha256")
                if (
                    not isinstance(manifest_sha256, str)
                    or len(manifest_sha256) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in manifest_sha256
                    )
                ):
                    errors.append(
                        "production dynamic-crop v3 corpus manifest digest is invalid"
                    )

    split = payload.get("split")
    if not isinstance(split, Mapping):
        errors.append("leakage-safe model-selection split evidence is missing")
    else:
        if split.get("split_level") != "song":
            errors.append("production split must retain a complete song holdout")
        if split.get("model_selection_validation_level") != (
            "whole_reference_within_training_songs"
        ):
            errors.append("model selection must use whole training-song references")
        validation_references = split.get("validation_reference_ids")
        if (
            not isinstance(validation_references, list)
            or len(validation_references) < 2
            or any(
                not isinstance(value, str) or not value.strip()
                for value in validation_references
            )
            or len(set(validation_references)) != len(validation_references)
        ):
            errors.append("model selection needs at least two unique validation references")
        if split.get("held_out_used_for_model_selection") is not False:
            errors.append("held-out song must not drive checkpoint selection")

    feature_gate = payload.get("production_feature_snapshot_gate")
    if not isinstance(feature_gate, Mapping):
        errors.append("production causal feature-snapshot evidence is missing")
    else:
        if feature_gate.get("passed") is not True or feature_gate.get(
            "contract"
        ) != PRODUCTION_FEATURE_SNAPSHOT_GATE_CONTRACT:
            errors.append("production causal feature-snapshot gate did not pass")
        hashes = feature_gate.get("feature_snapshot_sha256_by_reference")
        reference_count = feature_gate.get("reference_count")
        if (
            not isinstance(hashes, Mapping)
            or isinstance(reference_count, bool)
            or not isinstance(reference_count, int)
            or reference_count < 1
            or len(hashes) != reference_count
            or any(
                not isinstance(reference_id, str)
                or not reference_id.strip()
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for reference_id, digest in hashes.items()
            )
        ):
            errors.append("production feature-snapshot hash evidence is invalid")
        if feature_gate.get(
            "reference_match_or_current_choice_features_present"
        ) is not False:
            errors.append("production feature snapshots contain prohibited leakage")
        if feature_gate.get("missing_reference_ids") != []:
            errors.append("production feature snapshots are incomplete")
        if isinstance(split, Mapping) and isinstance(hashes, Mapping):
            partition_values = (
                split.get("training_reference_ids"),
                split.get("validation_reference_ids"),
                split.get("held_out_reference_ids"),
            )
            if not all(isinstance(values, list) for values in partition_values):
                errors.append("production reference partition evidence is incomplete")
            else:
                partition_ids = [
                    value
                    for values in partition_values
                    for value in values
                    if isinstance(value, str)
                ]
                if (
                    len(partition_ids) != sum(len(values) for values in partition_values)
                    or len(partition_ids) != len(set(partition_ids))
                    or set(partition_ids) != set(hashes)
                ):
                    errors.append(
                        "feature snapshots do not cover the exact reference partition"
                    )

    optimization = payload.get("optimization")
    if not isinstance(optimization, Mapping):
        errors.append("leakage-safe optimization evidence is missing")
    else:
        if optimization.get("model_selection") != (
            STAGE_EDIT_VALIDATION_SELECTION_CONTRACT
        ):
            errors.append("checkpoint selection evidence is not leakage-safe")
        if optimization.get("unseen_song_used_for_model_selection") is not False:
            errors.append("unseen song was not excluded from model selection")
        selection = optimization.get("best_validation_selection_evidence")
        if (
            not isinstance(selection, Mapping)
            or selection.get("contract")
            != STAGE_EDIT_VALIDATION_SELECTION_CONTRACT
            or selection.get("selectable") is not True
            or selection.get("heldout_metrics_consumed") is not False
        ):
            errors.append("validation composite selection evidence is invalid")
        if dynamic_crop_v3_checkpoint:
            selected_v3_gate = None
            if optimization.get("dynamic_crop_v3_model_selection") != (
                DYNAMIC_CROP_V3_VALIDATION_SELECTION_GATE_CONTRACT
            ):
                errors.append("dynamic-crop v3 selection contract differs")
            if optimization.get(
                "effective_dynamic_crop_loss_contract_version"
            ) != 3:
                errors.append("dynamic-crop v3 loss contract version is missing")
            versioned_dynamic_loss = optimization.get(
                "effective_dynamic_crop_v3_loss_contract"
            )
            if (
                not isinstance(versioned_dynamic_loss, Mapping)
                or versioned_dynamic_loss != effective_dynamic_loss
            ):
                errors.append("dynamic-crop v3 effective loss evidence differs")
            if isinstance(selection, Mapping):
                errors.extend(
                    _dynamic_crop_v3_composite_selection_errors(
                        selection,
                        name="dynamic-crop v3 selected validation",
                    )
                )
                errors.extend(
                    _dynamic_crop_v3_selection_gate_errors(
                        selection.get("dynamic_crop_v3_selection_gate"),
                        name="dynamic-crop v3 selected validation gate",
                    )
                )
                score_component = selection.get(
                    "dynamic_crop_v3_score_component"
                )
                selected_v3_gate = selection.get(
                    "dynamic_crop_v3_selection_gate"
                )
                gate_score = (
                    selected_v3_gate.get("score_component")
                    if isinstance(selected_v3_gate, Mapping)
                    else None
                )
                if (
                    not isinstance(score_component, (int, float))
                    or isinstance(score_component, bool)
                    or not math.isfinite(float(score_component))
                    or float(score_component) <= 0.0
                    or score_component != gate_score
                ):
                    errors.append(
                        "dynamic-crop v3 selection score does not include improvement"
                    )
            reloaded_selection = optimization.get(
                "reloaded_checkpoint_validation_selection_evidence"
            )
            if (
                not isinstance(reloaded_selection, Mapping)
                or reloaded_selection.get("contract")
                != STAGE_EDIT_VALIDATION_SELECTION_CONTRACT
                or reloaded_selection.get("selectable") is not True
                or reloaded_selection.get("heldout_metrics_consumed") is not False
            ):
                errors.append(
                    "reloaded dynamic-crop v3 validation selection evidence is invalid"
                )
            else:
                errors.extend(
                    _dynamic_crop_v3_composite_selection_errors(
                        reloaded_selection,
                        name="reloaded dynamic-crop v3 validation",
                    )
                )
                errors.extend(
                    _dynamic_crop_v3_selection_gate_errors(
                        reloaded_selection.get(
                            "dynamic_crop_v3_selection_gate"
                        ),
                        name="reloaded dynamic-crop v3 validation gate",
                    )
                )
            selected_metrics = optimization.get(
                "selected_checkpoint_validation_metrics"
            )
            selected_dynamic_metrics = (
                selected_metrics.get("dynamic_crop_v3")
                if isinstance(selected_metrics, Mapping)
                else None
            )
            if not isinstance(selected_dynamic_metrics, Mapping):
                errors.append(
                    "selected checkpoint dynamic-crop v3 metrics are missing"
                )
            else:
                errors.extend(
                    _dynamic_crop_v3_gate_metric_binding_errors(
                        selected_v3_gate,
                        selected_dynamic_metrics,
                        metadata=metadata,
                        name="selected dynamic-crop v3",
                    )
                )
                reloaded_gate = (
                    reloaded_selection.get("dynamic_crop_v3_selection_gate")
                    if isinstance(reloaded_selection, Mapping)
                    else None
                )
                errors.extend(
                    _dynamic_crop_v3_gate_metric_binding_errors(
                        reloaded_gate,
                        selected_dynamic_metrics,
                        metadata=metadata,
                        name="reloaded dynamic-crop v3",
                    )
                )

    threshold_freeze = payload.get("cut_threshold_freeze")
    if not isinstance(threshold_freeze, Mapping):
        errors.append("validation-selected cut threshold freeze is missing")
    else:
        threshold = threshold_freeze.get("selected_probability_threshold")
        valid_threshold = (
            not isinstance(threshold, bool)
            and isinstance(threshold, (int, float))
            and math.isfinite(float(threshold))
            and 0.0 <= float(threshold) <= 1.0
        )
        freeze_sha256 = threshold_freeze.get("freeze_sha256")
        unsigned_freeze = dict(threshold_freeze)
        unsigned_freeze.pop("freeze_sha256", None)
        expected_freeze_sha256 = hashlib.sha256(
            json.dumps(
                unsigned_freeze,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if (
            threshold_freeze.get("contract")
            != STAGE_EDIT_CUT_THRESHOLD_SELECTION_CONTRACT
            or threshold_freeze.get("heldout_labels_consumed") is not False
            or threshold_freeze.get("heldout_probabilities_consumed") is not False
            or threshold_freeze.get("selected_before_heldout_forward") is not True
            or not valid_threshold
            or freeze_sha256 != expected_freeze_sha256
        ):
            errors.append("validation-selected cut threshold evidence is invalid")
        if (
            isinstance(checkpoint, Mapping)
            and threshold_freeze.get("selected_checkpoint_id")
            != checkpoint.get("checkpoint_id")
        ):
            errors.append("cut threshold freeze checkpoint_id mismatches")
        if (
            actual_weight_sha256 is None
            or threshold_freeze.get("selected_checkpoint_weight_sha256")
            != actual_weight_sha256
        ):
            errors.append("cut threshold freeze weight digest mismatches")

    held_out = payload.get("held_out_evaluation")
    if not isinstance(held_out, Mapping):
        errors.append("independent held-out evaluation is missing")
    else:
        if held_out.get("passed") is not True:
            errors.append("independent held-out evaluation did not pass")
        if held_out.get("split_level") != "song":
            errors.append("held-out evaluation must split at song level")
        if held_out.get("checkpoint_id") != metadata.checkpoint_id:
            errors.append("held-out checkpoint_id does not match metadata")
        if (
            actual_weight_sha256 is None
            or held_out.get("weight_sha256") != actual_weight_sha256
        ):
            errors.append("held-out evaluation weight digest does not match checkpoint")
        training_songs = held_out.get("training_song_ids")
        held_out_songs = held_out.get("held_out_song_ids")
        valid_training_songs = (
            isinstance(training_songs, list)
            and len(training_songs) >= 2
            and all(isinstance(value, str) and value.strip() for value in training_songs)
            and len(set(training_songs)) == len(training_songs)
        )
        valid_held_out_songs = (
            isinstance(held_out_songs, list)
            and bool(held_out_songs)
            and all(isinstance(value, str) and value.strip() for value in held_out_songs)
            and len(set(held_out_songs)) == len(held_out_songs)
        )
        if not valid_training_songs:
            errors.append("held-out evaluation needs at least two training songs")
        if not valid_held_out_songs:
            errors.append("held-out evaluation needs at least one held-out song")
        if valid_training_songs and valid_held_out_songs and not set(
            training_songs
        ).isdisjoint(held_out_songs):
            errors.append("training and held-out song ids must be disjoint")
        held_out_reference_count = held_out.get("held_out_reference_count")
        if (
            isinstance(held_out_reference_count, bool)
            or not isinstance(held_out_reference_count, int)
            or held_out_reference_count < 1
        ):
            errors.append("held-out evaluation needs at least one reference")
        quality_gate = held_out.get("quality_gate")
        if not isinstance(quality_gate, Mapping) or quality_gate.get("passed") is not True:
            errors.append("held-out quality gate did not pass")
        if dynamic_crop_v2_checkpoint and not isinstance(
            held_out.get("dynamic_crop_v2_metrics"), Mapping
        ):
            errors.append("held-out dynamic-crop v2 metrics are missing")
        if dynamic_crop_v3_checkpoint:
            if held_out.get("dynamic_crop_version") != 3:
                errors.append("held-out evaluation does not declare dynamic-crop v3")
            if held_out.get("dynamic_crop_v2_metrics") is not None:
                errors.append("held-out dynamic-crop v3 evaluation contains v2 metrics")
            if not isinstance(
                held_out.get("dynamic_crop_v3_metrics"), Mapping
            ):
                errors.append("held-out dynamic-crop v3 metrics are missing")
            if held_out.get(
                "heldout_dynamic_crop_metrics_used_for_checkpoint_selection"
            ) is not False:
                errors.append(
                    "held-out dynamic-crop v3 metrics contaminated checkpoint selection"
                )
    return tuple(errors)


def stage_edit_readiness_report(
    checkpoint_directory: Path | None = None,
) -> StageEditReadinessReport:
    """Fail closed unless a checkpoint has production training evidence.

    Weight integrity and a positive step counter prove only that MLX can load
    the file.  Production readiness additionally requires a colocated,
    checkpoint-bound report that records a passed dense-supervision gate and
    an independent song-level held-out quality gate.  Offline sparse
    calibration reports intentionally never satisfy this contract.
    """

    version = mlx_runtime_version()
    if version is None:
        return StageEditReadinessReport(
            architecture_available=False,
            learned_inference_ready=False,
            status="optional_dependency_missing",
            reason="MLX is not installed; schema tooling only is available",
            mlx_version=None,
            checkpoint_path=None,
            training_steps=0,
        )
    if checkpoint_directory is None:
        return StageEditReadinessReport(
            architecture_available=True,
            learned_inference_ready=False,
            status="architecture_only_no_checkpoint",
            reason="model code exists but no trained checkpoint was supplied",
            mlx_version=version,
            checkpoint_path=None,
            training_steps=0,
        )
    source = Path(checkpoint_directory)
    legacy_metadata_path = source / "metadata.json"
    dynamic_v2_metadata_path = source / "dynamic-crop-v2-metadata.json"
    dynamic_v3_metadata_path = source / "dynamic-crop-v3-metadata.json"
    namespace_paths = {
        "legacy": (
            legacy_metadata_path,
            source / "model.safetensors",
        ),
        "dynamic_crop_v2": (
            dynamic_v2_metadata_path,
            source / "dynamic-crop-v2-model.safetensors",
        ),
        "dynamic_crop_v3": (
            dynamic_v3_metadata_path,
            source / "dynamic-crop-v3-model.safetensors",
        ),
    }
    present_namespaces = tuple(
        name
        for name, paths in namespace_paths.items()
        if any(path.exists() or path.is_symlink() for path in paths)
    )
    metadata_entries = tuple(
        (path, name)
        for name, paths in namespace_paths.items()
        for path in paths[:1]
        if path.is_file()
    )
    if len(present_namespaces) > 1 or len(metadata_entries) > 1:
        return StageEditReadinessReport(
            architecture_available=True,
            learned_inference_ready=False,
            status="checkpoint_invalid",
            reason="checkpoint directory ambiguously contains mixed model namespaces",
            mlx_version=version,
            checkpoint_path=str(source.resolve()),
            training_steps=0,
        )
    if not metadata_entries:
        if present_namespaces:
            reason = "checkpoint namespace contains weights without metadata"
            status = "checkpoint_invalid"
        else:
            reason = (
                "none of legacy, dynamic-crop v2, or dynamic-crop v3 "
                "checkpoint metadata exists"
            )
            status = "checkpoint_metadata_missing"
        return StageEditReadinessReport(
            architecture_available=True,
            learned_inference_ready=False,
            status=status,
            reason=reason,
            mlx_version=version,
            checkpoint_path=str(source.resolve()),
            training_steps=0,
        )
    metadata_path, checkpoint_format = metadata_entries[0]
    dynamic_crop_version = (
        3
        if checkpoint_format == "dynamic_crop_v3"
        else 2
        if checkpoint_format == "dynamic_crop_v2"
        else None
    )
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if checkpoint_format == "dynamic_crop_v3":
            # Local imports avoid the v3 head -> base Transformer module cycle.
            from .stage_edit_dynamic_crop_head_v3 import (
                StageEditDynamicCropCheckpointMetadataV3,
            )

            metadata = StageEditDynamicCropCheckpointMetadataV3.from_dict(payload)
        elif checkpoint_format == "dynamic_crop_v2":
            # Local import avoids a module cycle: the v2 model nests this base
            # Transformer and owns a deliberately separate checkpoint schema.
            from .stage_edit_dynamic_crop_head_v2 import (
                StageEditDynamicCropCheckpointMetadataV2,
            )

            metadata = StageEditDynamicCropCheckpointMetadataV2.from_dict(payload)
        else:
            metadata = StageEditCheckpointMetadata.from_dict(payload)
        weight_path = source / metadata.weight_filename
        valid_weights = (
            weight_path.is_file()
            and metadata.weight_sha256 is not None
            and _sha256_file(weight_path) == metadata.weight_sha256
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        return StageEditReadinessReport(
            architecture_available=True,
            learned_inference_ready=False,
            status="checkpoint_invalid",
            reason=f"checkpoint validation failed: {exc}",
            mlx_version=version,
            checkpoint_path=str(source.resolve()),
            training_steps=0,
            checkpoint_format=checkpoint_format,
            dynamic_crop_version=dynamic_crop_version,
            checkpoint_metadata_filename=metadata_path.name,
        )
    integrity_ready = bool(
        metadata.weights_trained and metadata.training_steps > 0 and valid_weights
    )
    if not integrity_ready:
        return StageEditReadinessReport(
            architecture_available=True,
            learned_inference_ready=False,
            status="untrained_checkpoint_only",
            reason=(
                "checkpoint is present but is not declared and verified as trained"
            ),
            mlx_version=version,
            checkpoint_path=str(source.resolve()),
            training_steps=metadata.training_steps,
            checkpoint_format=checkpoint_format,
            dynamic_crop_version=dynamic_crop_version,
            checkpoint_metadata_filename=metadata_path.name,
        )

    report_path = source / PRODUCTION_TRAINING_REPORT_FILENAME
    if not report_path.is_file():
        return StageEditReadinessReport(
            architecture_available=True,
            learned_inference_ready=False,
            status="production_training_report_missing",
            reason=(
                "trained weights are loadable but no production training report "
                "was supplied"
            ),
            mlx_version=version,
            checkpoint_path=str(source.resolve()),
            training_steps=metadata.training_steps,
            checkpoint_format=checkpoint_format,
            dynamic_crop_version=dynamic_crop_version,
            checkpoint_metadata_filename=metadata_path.name,
        )
    try:
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        report_errors = _production_training_report_errors(
            report_payload,
            metadata=metadata,
            actual_weight_sha256=(
                metadata.weight_sha256 if valid_weights else None
            ),
            checkpoint_format=checkpoint_format,
        )
    except (OSError, json.JSONDecodeError) as exc:
        report_errors = (f"production training report cannot be read: {exc}",)
    if report_errors:
        return StageEditReadinessReport(
            architecture_available=True,
            learned_inference_ready=False,
            status="production_training_report_rejected",
            reason="; ".join(report_errors),
            mlx_version=version,
            checkpoint_path=str(source.resolve()),
            training_steps=metadata.training_steps,
            checkpoint_format=checkpoint_format,
            dynamic_crop_version=dynamic_crop_version,
            checkpoint_metadata_filename=metadata_path.name,
        )
    return StageEditReadinessReport(
        architecture_available=True,
        learned_inference_ready=True,
        status="production_checkpoint_ready",
        reason=(
            "weight integrity, dense production supervision, and independent "
            "song-level held-out quality evidence are valid"
        ),
        mlx_version=version,
        checkpoint_path=str(source.resolve()),
        training_steps=metadata.training_steps,
        checkpoint_format=checkpoint_format,
        dynamic_crop_version=dynamic_crop_version,
        checkpoint_metadata_filename=metadata_path.name,
    )


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "DEFAULT_STAGE_EDIT_SCHEMA",
    "DYNAMIC_CROP_V3_PRODUCTION_ARCHITECTURE",
    "DYNAMIC_CROP_V3_ACTIVATION_CHECKPOINTING_CONTRACT",
    "DYNAMIC_CROP_V3_TRAINING_CORPUS_CONTRACT",
    "DYNAMIC_CROP_V3_VALIDATION_SELECTION_GATE_CONTRACT",
    "MASKED_SOURCE_LOGIT",
    "PRODUCTION_CHECKPOINT_SCOPE",
    "PRODUCTION_FEATURE_SNAPSHOT_GATE_CONTRACT",
    "PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION",
    "PRODUCTION_TRAINING_REPORT_FILENAME",
    "PRODUCTION_TRAINING_RUN_KIND",
    "STAGE_EDIT_CUT_IMBALANCE_CONTRACT",
    "STAGE_EDIT_CUT_LOGIT_MODE",
    "STAGE_EDIT_CUT_SCORING_CONTRACT",
    "STAGE_EDIT_CUT_THRESHOLD_SELECTION_CONTRACT",
    "STAGE_EDIT_ACTION_FORMATION_SOURCE_FEATURE_NAMES",
    "STAGE_EDIT_DECISION_HISTORY_OWNER",
    "STAGE_EDIT_RECENT_HISTORY_MASK_CONTRACT",
    "STAGE_EDIT_FORMATION_LAYOUT_FEATURE_NAMES",
    "STAGE_EDIT_FORMATION_POSE_FEATURE_NAMES",
    "STAGE_EDIT_POSE_EMBEDDING_FEATURE_NAMES",
    "STAGE_EDIT_TEMPORAL_POSE_PRODUCER_BACKEND",
    "STAGE_EDIT_TEMPORAL_POSE_SAMPLING_CONTRACTS",
    "STAGE_EDIT_TEMPORAL_POSE_TIMELINE_ENTRY_EXIT_CONTRACT",
    "STAGE_EDIT_TEMPORAL_POSE_VISUAL_CACHE_CURRENT_NEXT_CONTRACT",
    "STAGE_EDIT_SOURCE_RELATION_MODE",
    "STAGE_EDIT_SOURCE_SCORING_CONTRACT",
    "STAGE_EDIT_VALIDATION_SELECTION_CONTRACT",
    "LEARNED_TRANSITION_AUTHORITY",
    "RENDERER_OWNED_TRANSITION_AUTHORITY",
    "TOKEN_SCHEMA_VERSION",
    "SourceCandidateToken",
    "StageEditTemporalPoseSidecar",
    "StageEditCheckpointMetadata",
    "StageEditLossContract",
    "StageEditModelOutput",
    "StageEditReadinessReport",
    "StageEditSequence",
    "StageEditSupervisionBatch",
    "StageEditTarget",
    "StageEditTargetSequence",
    "StageEditTensorBatch",
    "StageEditToken",
    "StageEditTokenSchema",
    "StageEditTransformer",
    "StageEditTransformerConfig",
    "load_stage_edit_checkpoint",
    "mask_stage_edit_recent_history_for_semimarkov",
    "mlx_runtime_version",
    "new_stage_edit_checkpoint_metadata",
    "production_stage_edit_config",
    "save_stage_edit_checkpoint",
    "smoke_stage_edit_config",
    "stage_edit_loss",
    "stage_edit_cut_imbalance_contract",
    "stage_edit_cut_scoring_contract",
    "stage_edit_parameter_count",
    "stage_edit_recent_history_mask_contract",
    "stage_edit_readiness_report",
    "stage_edit_source_scoring_contract",
    "targets_from_reference_supervision",
    "tensorize_stage_edit_batch",
    "tensorize_stage_edit_supervision",
    "valid_dynamic_crop_v3_activation_checkpointing_evidence",
]
