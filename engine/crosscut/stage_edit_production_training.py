"""Production StageEdit training with a strict song-held-out contract.

Unlike :mod:`stage_edit_training`, this module is allowed to create a
production-scoped checkpoint.  It can do so only when all of the following
are true:

* dense editorial supervision passes the shared fail-closed gate;
* at least two complete songs train the model and at least one different song
  remains wholly held out;
* every structured head has real training and held-out labels;
* the exact saved checkpoint passes fixed, mask-aware held-out metrics; and
* the caller explicitly authorizes activation.

The report is always written beside the checkpoint.  A failed quality gate is
therefore useful evidence, but it remains impossible for
``stage_edit_readiness_report`` to mistake that run for a production editor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np

from .stage_edit_dynamic_crop_head_v2 import (
    DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME,
    DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME,
    DynamicCropV2LossContract,
    StageEditTransformerDynamicCropV2,
    evaluate_stage_edit_dynamic_crop_v2,
    load_stage_edit_dynamic_crop_checkpoint_v2,
    new_stage_edit_dynamic_crop_checkpoint_metadata_v2,
    save_stage_edit_dynamic_crop_checkpoint_v2,
    stage_edit_dynamic_crop_v2_loss,
    stage_edit_dynamic_crop_v2_parameter_count,
    tensorize_stage_edit_dynamic_crop_supervision_v2,
)
from .stage_edit_dynamic_crop_head_v3 import (
    DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
    DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
    DYNAMIC_CROP_V3_MODEL_CONTRACT,
    DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
    DYNAMIC_CROP_V3_SCHEMA_VERSION,
    DYNAMIC_CROP_V3_TARGET_CONTRACT,
    DYNAMIC_CROP_V3_MAX_KEYFRAMES,
    DynamicCropV3LossContract,
    StageEditTransformerDynamicCropV3,
    evaluate_stage_edit_dynamic_crop_v3,
    load_stage_edit_dynamic_crop_checkpoint_v3,
    new_stage_edit_dynamic_crop_checkpoint_metadata_v3,
    renderer_range_normalized_values_v3,
    renderer_safe_values_from_raw_v3,
    save_stage_edit_dynamic_crop_checkpoint_v3,
    stage_edit_dynamic_crop_v3_loss,
    stage_edit_dynamic_crop_v3_parameter_count,
    tensorize_stage_edit_dynamic_crop_supervision_v3,
)
from .stage_edit_dynamic_crop_training import (
    StageEditDynamicCropTrainingCorpusV2,
)
from .stage_edit_dynamic_crop_training_v3 import (
    StageEditDynamicCropTrainingCorpusV3,
)
from .stage_edit_feature_materializer import (
    FEATURE_MATERIALIZER_SCHEMA_VERSION,
)
from .stage_edit_reference_adapter import (
    ReferenceTrainingCorpus,
    ReferenceTrainingExample,
    VerifiedReferenceFeatureSnapshotEvidence,
)
from .stage_edit_training import (
    ProductionSupervisionGate,
    STAGE_EDIT_CUT_THRESHOLD_SELECTION_CONTRACT,
    STAGE_EDIT_VALIDATION_SELECTION_CONTRACT,
    evaluate_stage_edit_model,
    require_stage_edit_production_supervision,
    select_stage_edit_cut_threshold_from_model,
    stage_edit_source_evaluation_contract,
    stage_edit_validation_selection_score,
)
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    DYNAMIC_CROP_V3_ACTIVATION_CHECKPOINTING_CONTRACT,
    PRODUCTION_CHECKPOINT_SCOPE,
    PRODUCTION_FEATURE_SNAPSHOT_GATE_CONTRACT,
    PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION,
    PRODUCTION_TRAINING_REPORT_FILENAME,
    PRODUCTION_TRAINING_RUN_KIND,
    StageEditLossContract,
    StageEditSequence,
    StageEditTarget,
    StageEditTargetSequence,
    StageEditTokenSchema,
    StageEditTransformer,
    StageEditTransformerConfig,
    load_stage_edit_checkpoint,
    mask_stage_edit_recent_history_for_semimarkov,
    mlx_runtime_version,
    new_stage_edit_checkpoint_metadata,
    production_stage_edit_config,
    save_stage_edit_checkpoint,
    stage_edit_loss,
    stage_edit_cut_imbalance_contract,
    stage_edit_cut_scoring_contract,
    stage_edit_parameter_count,
    stage_edit_recent_history_mask_contract,
    stage_edit_readiness_report,
    stage_edit_source_scoring_contract,
    tensorize_stage_edit_batch,
    tensorize_stage_edit_supervision,
)
from .stage_edit_transition_authority import (
    LEARNED_TRANSITION_AUTHORITY,
    RENDERER_OWNED_ALLOWED_MODEL_LABELS,
    RENDERER_OWNED_LEARNED_EVALUATION_SKIP_REASON,
    RENDERER_OWNED_TRANSITION_AUTHORITY,
    StageEditTransitionAuthority,
    transition_authority_contract,
)

try:  # Optional outside Apple-silicon production hosts.
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
except ImportError as exc:  # pragma: no cover - non-Apple portability.
    mx = None
    nn = None
    optim = None
    tree_flatten = None
    _MLX_IMPORT_ERROR: ImportError | None = exc
else:
    _MLX_IMPORT_ERROR = None


STRUCTURED_HEAD_COVERAGE_CONTRACT = (
    "duration_learned_transition_or_renderer_metadata_coverage_gate_v2"
)
STAGE_EDIT_TRAINING_SONG_EXPANSION_CONTRACT = (
    "complete_song_addition_with_one_reference_validation_and_snapshot_gate_v1"
)
STAGE_EDIT_SELECTION_FREEZE_FILENAME = (
    "selection-frozen-before-heldout.json"
)
DYNAMIC_CROP_V3_VALIDATION_SELECTION_CONTRACT = (
    "dynamic_crop_v3_renderer_exact_static_prior_improvement_gate_v1"
)
DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS = (50, 100, 200, 300, 400, 500)
DYNAMIC_CROP_V3_PRIOR_CANDIDATE_SELECTION_CONTRACT = (
    "predeclared_train_only_optimizer_milestones_existing_exact_validation_"
    "highest_score_lower_step_tiebreak_v1"
)


def _require_mlx() -> None:
    if mx is None or nn is None or optim is None:
        detail = f": {_MLX_IMPORT_ERROR}" if _MLX_IMPORT_ERROR else ""
        raise RuntimeError(f"production StageEdit training requires MLX{detail}")


def _enable_dynamic_crop_v3_activation_checkpointing(
    model: Any,
) -> dict[str, Any]:
    """Enable MLX layer recomputation without changing checkpoint weights.

    MLX's TransformerEncoder owns a training-only ``checkpoint`` switch.  It
    recomputes encoder-layer intermediates during backward instead of retaining
    them, while the serialized parameter tree and inference model remain
    identical.  Dynamic Crop v3 production training always uses this path so
    the pinned 105M candidate can stay below the measured device budget.
    """

    if not isinstance(model, StageEditTransformerDynamicCropV3):
        raise TypeError(
            "activation checkpointing requires a Dynamic Crop v3 model"
        )
    base = getattr(model, "base", None)
    encoder = getattr(base, "encoder", None)
    if encoder is None or not hasattr(encoder, "checkpoint"):
        raise RuntimeError(
            "MLX TransformerEncoder checkpoint support is unavailable"
        )
    previous = getattr(encoder, "checkpoint")
    if not isinstance(previous, bool):
        raise RuntimeError(
            "MLX TransformerEncoder checkpoint switch is not boolean"
        )
    if previous is not False:
        raise RuntimeError(
            "Dynamic Crop v3 activation checkpointing must start disabled"
        )
    encoder.checkpoint = True
    if encoder.checkpoint is not True:
        raise RuntimeError("activation checkpointing did not remain enabled")
    return {
        "contract": DYNAMIC_CROP_V3_ACTIVATION_CHECKPOINTING_CONTRACT,
        "enabled": True,
        "scope": "base_transformer_encoder_all_layers",
        "implementation": "mlx.nn.TransformerEncoder.checkpoint",
        "training_only": True,
        "serialized_parameter_tree_changed": False,
        "inference_runtime_requires_checkpointing": False,
        "previous_value": False,
    }


def _finite_metric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _dynamic_crop_v3_validation_selection_evidence(
    metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Fail closed on the renderer-exact v3 validation selection contract."""

    reasons: list[str] = []
    if not isinstance(metrics, Mapping):
        metrics = {}
        reasons.append("dynamic_crop_v3_validation_metrics_missing")
    if metrics.get("schema_version") != DYNAMIC_CROP_V3_SCHEMA_VERSION:
        reasons.append("dynamic_crop_v3_validation_schema_mismatch")
    if metrics.get("model_contract") != DYNAMIC_CROP_V3_MODEL_CONTRACT:
        reasons.append("dynamic_crop_v3_validation_model_contract_mismatch")
    if metrics.get("target_contract") != DYNAMIC_CROP_V3_TARGET_CONTRACT:
        reasons.append("dynamic_crop_v3_validation_target_contract_mismatch")
    if metrics.get("output_contract") != DYNAMIC_CROP_V3_OUTPUT_CONTRACT:
        reasons.append("dynamic_crop_v3_validation_output_contract_mismatch")
    if metrics.get("legacy_v2_checkpoint_or_output_used") is not False:
        reasons.append("dynamic_crop_v3_validation_used_legacy_v2")
    if metrics.get("legacy_static_crop_head_used_for_trajectory") is not False:
        reasons.append("dynamic_crop_v3_validation_used_static_crop_head")

    trajectory = metrics.get("trajectory")
    if not isinstance(trajectory, Mapping):
        trajectory = {}
        reasons.append("dynamic_crop_v3_validation_trajectory_missing")
    renderer_decode = trajectory.get("renderer_decode")
    if not isinstance(renderer_decode, Mapping):
        renderer_decode = {}
        reasons.append("dynamic_crop_v3_validation_renderer_decode_missing")

    attempt_count = renderer_decode.get("attempt_count")
    eligible_count = renderer_decode.get("eligible_count")
    eligible_rate = _finite_metric(renderer_decode.get("eligible_rate"))
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or attempt_count < 1
    ):
        reasons.append("dynamic_crop_v3_validation_renderer_attempts_missing")
    if (
        isinstance(eligible_count, bool)
        or not isinstance(eligible_count, int)
        or not isinstance(attempt_count, int)
        or isinstance(attempt_count, bool)
        or eligible_count != attempt_count
        or eligible_rate != 1.0
    ):
        reasons.append("dynamic_crop_v3_validation_renderer_decode_not_exact")

    projection_count = renderer_decode.get(
        "minimum_safe_zoom_projection_count"
    )
    if (
        isinstance(projection_count, bool)
        or not isinstance(projection_count, int)
        or projection_count != 0
    ):
        reasons.append("dynamic_crop_v3_validation_projection_not_zero")

    source_slot_mismatch_count = trajectory.get(
        "target_source_slot_mismatch_count"
    )
    if (
        isinstance(source_slot_mismatch_count, bool)
        or not isinstance(source_slot_mismatch_count, int)
        or source_slot_mismatch_count != 0
    ):
        reasons.append("dynamic_crop_v3_validation_source_slot_mismatch")

    improvement = _finite_metric(
        trajectory.get("aggregate_normalized_improvement_over_static_prior")
    )
    if improvement is None or improvement <= 0.0:
        reasons.append("dynamic_crop_v3_validation_did_not_beat_static_prior")

    return {
        "contract": DYNAMIC_CROP_V3_VALIDATION_SELECTION_CONTRACT,
        "selectable": not reasons,
        "reasons": reasons,
        "score_component": improvement,
        "renderer_decode_attempt_count": attempt_count,
        "renderer_decode_eligible_count": eligible_count,
        "renderer_decode_eligible_rate": eligible_rate,
        "minimum_safe_zoom_projection_count": projection_count,
        "target_source_slot_mismatch_count": source_slot_mismatch_count,
        "aggregate_normalized_improvement_over_static_prior": improvement,
        "static_prior_can_be_selected": False,
        "heldout_metrics_consumed": False,
    }


def _stage_edit_validation_selection_with_dynamic_crop_v3(
    base_selection: Mapping[str, Any],
    dynamic_crop_v3_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Add v3 validation utility without changing the base selection identity."""

    dynamic_evidence = _dynamic_crop_v3_validation_selection_evidence(
        dynamic_crop_v3_metrics
    )
    base_score = _finite_metric(base_selection.get("score"))
    dynamic_score = _finite_metric(dynamic_evidence.get("score_component"))
    selectable = bool(
        base_selection.get("selectable")
        and dynamic_evidence["selectable"]
        and base_score is not None
        and dynamic_score is not None
    )
    combined_score = (
        base_score + dynamic_score if selectable else None
    )
    return {
        **dict(base_selection),
        "selectable": selectable,
        "score": combined_score,
        "base_selection_score": base_score,
        "dynamic_crop_v3_score_component": dynamic_score,
        "components": {
            **dict(base_selection.get("components", {})),
            "dynamic_crop_v3_aggregate_normalized_improvement_over_static_prior": (
                dynamic_score
            ),
        },
        "formula": (
            f"{base_selection.get('formula')} + "
            "dynamic_crop_v3_aggregate_normalized_improvement_over_static_prior"
        ),
        "dynamic_crop_v3_selection_gate": dynamic_evidence,
        "heldout_metrics_consumed": False,
    }


def _require_unambiguous_checkpoint_namespace(
    checkpoint_root: Path,
    *,
    dynamic_crop_version: int | None,
) -> None:
    """Refuse to overwrite a directory owned by the other model format."""

    if not checkpoint_root.exists():
        return
    if not checkpoint_root.is_dir():
        raise ValueError("production checkpoint path exists but is not a directory")
    legacy_files = (
        checkpoint_root / "metadata.json",
        checkpoint_root / "model.safetensors",
    )
    dynamic_v2_files = (
        checkpoint_root / DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME,
        checkpoint_root / DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME,
    )
    dynamic_v3_files = (
        checkpoint_root / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
        checkpoint_root / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
    )
    if dynamic_crop_version == 2:
        incompatible = legacy_files + dynamic_v3_files
        requested = "Dynamic Crop v2"
    elif dynamic_crop_version == 3:
        incompatible = legacy_files + dynamic_v2_files
        requested = "Dynamic Crop v3"
    elif dynamic_crop_version is None:
        incompatible = dynamic_v2_files + dynamic_v3_files
        requested = "legacy"
    else:
        raise ValueError("unsupported dynamic crop checkpoint version")
    present = tuple(path.name for path in incompatible if path.exists())
    if present:
        raise ValueError(
            f"cannot train {requested} checkpoint into a directory containing "
            f"the other model format: {', '.join(present)}"
        )


def _production_feature_snapshot_gate_for_examples(
    examples: Sequence[ReferenceTrainingExample],
    *,
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
) -> dict[str, Any]:
    """Validate exact snapshot and causal-sensing evidence before preprocessing."""

    if not examples:
        raise ValueError("production StageEdit requires feature snapshot examples")
    hashes: dict[str, str] = {}
    sensing_gates: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for example in examples:
        evidence = example.verified_feature_snapshot_evidence
        digest = ""
        try:
            if not isinstance(
                evidence,
                VerifiedReferenceFeatureSnapshotEvidence,
            ):
                raise ValueError("loader-issued feature evidence is absent")
            sensing_gate = evidence.revalidate(
                example,
                schema=schema,
            )
            digest = evidence.feature_snapshot_sha256
        except (TypeError, ValueError):
            sensing_gate = None
        if (
            sensing_gate is None
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            missing.append(example.reference_id)
            continue
        hashes[example.reference_id] = digest
        sensing_gates[example.reference_id] = sensing_gate
    if missing or len(hashes) != len(examples):
        raise ValueError(
            "production StageEdit requires a verified causal feature snapshot "
            "for every reference; missing=" + ", ".join(sorted(missing))
        )
    if len(hashes) != len(set(hashes)):
        raise ValueError("production feature snapshot reference ids are not unique")
    scope_identities = {
        (
            gate["edit_mode"],
            gate["target_member"],
            gate["member_feature_mode"],
            gate["feature_snapshot_schema_version"],
            gate["token_schema_fingerprint"],
            gate["feature_materializer_contract"],
        )
        for gate in sensing_gates.values()
    }
    if len(scope_identities) != 1:
        raise ValueError(
            "production feature snapshots mix causal-sensing edit scope or schema"
        )
    return {
        "passed": True,
        "contract": PRODUCTION_FEATURE_SNAPSHOT_GATE_CONTRACT,
        "reference_count": len(hashes),
        "feature_snapshot_sha256_by_reference": dict(sorted(hashes.items())),
        "production_causal_sensing_gate_by_reference": {
            reference_id: sensing_gates[reference_id]
            for reference_id in sorted(sensing_gates)
        },
        "production_causal_sensing_scope": {
            "edit_mode": next(iter(sensing_gates.values()))["edit_mode"],
            "target_member": next(iter(sensing_gates.values()))["target_member"],
            "member_feature_mode": next(iter(sensing_gates.values()))[
                "member_feature_mode"
            ],
            "feature_snapshot_schema_version": (
                FEATURE_MATERIALIZER_SCHEMA_VERSION
            ),
            "token_schema_fingerprint": schema.fingerprint,
        },
        "reference_match_or_current_choice_features_present": False,
        "missing_reference_ids": [],
    }


def _require_production_feature_snapshots(
    corpus: ReferenceTrainingCorpus,
    *,
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
) -> dict[str, Any]:
    """Require the exact train/serve feature materializer for every reference."""

    if not isinstance(corpus, ReferenceTrainingCorpus):
        raise TypeError("production feature gate requires ReferenceTrainingCorpus")
    return _production_feature_snapshot_gate_for_examples(
        corpus.examples,
        schema=schema,
    )


@dataclass(frozen=True, slots=True)
class ProductionQualityThresholds:
    """Fixed minimum evidence for one unseen-song checkpoint evaluation."""

    minimum_cut_labels: int = 20
    minimum_cut_positives: int = 5
    minimum_cut_negatives: int = 10
    minimum_cut_auroc: float = 0.70
    minimum_cut_balanced_accuracy: float = 0.65
    minimum_source_groups: int = 5
    minimum_source_top1_accuracy: float = 0.35
    minimum_source_mrr: float = 0.55
    minimum_duration_labels: int = 5
    minimum_duration_accuracy: float = 0.30
    minimum_duration_observed_classes: int = 2
    minimum_duration_label_entropy_bits: float = 0.50
    minimum_transition_labels: int = 5
    minimum_transition_accuracy: float = 0.60
    minimum_transition_observed_classes: int = 2
    minimum_transition_label_entropy_bits: float = 0.50
    minimum_crop_labels_per_coordinate: int = 5
    maximum_zoom_mae: float = 0.08
    maximum_center_mae: float = 0.08
    maximum_roll_mae_degrees: float = 1.00
    minimum_dynamic_crop_tracks: int = 5
    minimum_dynamic_crop_keyframes: int = 8
    minimum_dynamic_crop_motion_labels: int = 3
    maximum_dynamic_crop_zoom_mae: float = 0.08
    maximum_dynamic_crop_center_mae: float = 0.08
    maximum_dynamic_crop_roll_mae_degrees: float = 1.00
    maximum_dynamic_crop_normalized_time_mae: float = 0.08
    minimum_dynamic_crop_presence_f1: float = 0.65
    minimum_dynamic_crop_keyframe_count_accuracy: float = 0.40
    minimum_dynamic_crop_motion_accuracy: float = 0.60
    minimum_dynamic_crop_renderer_decode_rate: float = 0.80

    def __post_init__(self) -> None:
        count_fields = (
            "minimum_cut_labels",
            "minimum_cut_positives",
            "minimum_cut_negatives",
            "minimum_source_groups",
            "minimum_duration_labels",
            "minimum_duration_observed_classes",
            "minimum_transition_labels",
            "minimum_transition_observed_classes",
            "minimum_crop_labels_per_coordinate",
            "minimum_dynamic_crop_tracks",
            "minimum_dynamic_crop_keyframes",
            "minimum_dynamic_crop_motion_labels",
        )
        if any(
            isinstance(getattr(self, name), bool)
            or not isinstance(getattr(self, name), int)
            or getattr(self, name) < 1
            for name in count_fields
        ):
            raise ValueError("production quality label minimums must be positive integers")
        unit_interval_fields = (
            "minimum_cut_auroc",
            "minimum_cut_balanced_accuracy",
            "minimum_source_top1_accuracy",
            "minimum_source_mrr",
            "minimum_duration_accuracy",
            "minimum_transition_accuracy",
            "minimum_dynamic_crop_presence_f1",
            "minimum_dynamic_crop_keyframe_count_accuracy",
            "minimum_dynamic_crop_motion_accuracy",
            "minimum_dynamic_crop_renderer_decode_rate",
        )
        if any(
            not 0.0 <= float(getattr(self, name)) <= 1.0
            for name in unit_interval_fields
        ):
            raise ValueError("production quality accuracies must be normalized")
        if any(
            not math.isfinite(float(getattr(self, name)))
            or float(getattr(self, name)) <= 0.0
            for name in (
                "maximum_zoom_mae",
                "maximum_center_mae",
                "maximum_roll_mae_degrees",
                "maximum_dynamic_crop_zoom_mae",
                "maximum_dynamic_crop_center_mae",
                "maximum_dynamic_crop_roll_mae_degrees",
                "maximum_dynamic_crop_normalized_time_mae",
                "minimum_duration_label_entropy_bits",
                "minimum_transition_label_entropy_bits",
            )
        ):
            raise ValueError(
                "production error and label-entropy limits must be positive and finite"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProductionQualityGate:
    passed: bool
    reasons: tuple[str, ...]
    thresholds: ProductionQualityThresholds
    metrics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "thresholds": self.thresholds.to_dict(),
            "metrics": dict(self.metrics),
        }


def evaluate_stage_edit_production_quality(
    metrics: Mapping[str, Any],
    *,
    dynamic_crop_v2_metrics: Mapping[str, Any] | None = None,
    dynamic_crop_v3_metrics: Mapping[str, Any] | None = None,
    thresholds: ProductionQualityThresholds = ProductionQualityThresholds(),
    transition_authority: str = LEARNED_TRANSITION_AUTHORITY,
) -> ProductionQualityGate:
    """Fail closed unless every renderer-active head generalizes unseen.

    Dynamic Crop v2/v3 checkpoints always carry the legacy four-value crop head
    because they share the v1 trunk/output namespace, but v2 decoding never
    consumes that head: an ineligible trajectory means identity geometry, not
    a static-head fallback.  Therefore the legacy crop metric remains in the
    report as auxiliary evidence and is activation-critical only when no v2
    trajectory metrics were supplied.  The stricter K=8 trajectory, timing,
    presence/count, motion, and real renderer-decode gates remain mandatory.
    """

    if dynamic_crop_v2_metrics is not None and dynamic_crop_v3_metrics is not None:
        raise ValueError("quality evaluation accepts exactly one dynamic crop version")

    authority = StageEditTransitionAuthority(mode=transition_authority)
    reasons: list[str] = []
    reported_metrics = dict(metrics)

    def minimum_count(value: Any, minimum: int, reason: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            reasons.append(reason)

    def minimum_metric(value: Any, minimum: float, reason: str) -> None:
        number = _finite_metric(value)
        if number is None or number < minimum:
            reasons.append(reason)

    def maximum_metric(value: Any, maximum: float, reason: str) -> None:
        number = _finite_metric(value)
        if number is None or number > maximum:
            reasons.append(reason)

    cut = metrics.get("cut")
    source = metrics.get("source")
    duration = metrics.get("duration")
    transition = metrics.get("transition")
    crop = metrics.get("crop")
    if not isinstance(cut, Mapping):
        reasons.append("cut_metrics_missing")
    else:
        minimum_count(
            cut.get("label_count"),
            thresholds.minimum_cut_labels,
            "insufficient_heldout_cut_labels",
        )
        minimum_count(
            cut.get("positive_count"),
            thresholds.minimum_cut_positives,
            "insufficient_heldout_cut_positives",
        )
        negative_count = None
        if isinstance(cut.get("label_count"), int) and isinstance(
            cut.get("positive_count"), int
        ):
            negative_count = cut["label_count"] - cut["positive_count"]
        minimum_count(
            negative_count,
            thresholds.minimum_cut_negatives,
            "insufficient_heldout_cut_negatives",
        )
        minimum_metric(
            cut.get("auroc"),
            thresholds.minimum_cut_auroc,
            "heldout_cut_auroc_below_threshold",
        )
        minimum_metric(
            cut.get("balanced_accuracy_at_frozen_validation_threshold"),
            thresholds.minimum_cut_balanced_accuracy,
            "heldout_cut_balanced_accuracy_below_threshold",
        )
    if not isinstance(source, Mapping):
        reasons.append("source_metrics_missing")
    else:
        minimum_count(
            source.get("listwise_group_count"),
            thresholds.minimum_source_groups,
            "insufficient_heldout_source_groups",
        )
        minimum_metric(
            source.get("top1_accuracy"),
            thresholds.minimum_source_top1_accuracy,
            "heldout_source_top1_below_threshold",
        )
        minimum_metric(
            source.get("mean_reciprocal_rank"),
            thresholds.minimum_source_mrr,
            "heldout_source_mrr_below_threshold",
        )
    if not isinstance(duration, Mapping):
        reasons.append("duration_metrics_missing")
    else:
        minimum_count(
            duration.get("label_count"),
            thresholds.minimum_duration_labels,
            "insufficient_heldout_duration_labels",
        )
        minimum_metric(
            duration.get("accuracy"),
            thresholds.minimum_duration_accuracy,
            "heldout_duration_accuracy_below_threshold",
        )
        minimum_count(
            duration.get("observed_class_count"),
            thresholds.minimum_duration_observed_classes,
            "insufficient_heldout_duration_class_coverage",
        )
        minimum_metric(
            duration.get("label_entropy_bits"),
            thresholds.minimum_duration_label_entropy_bits,
            "heldout_duration_label_entropy_below_threshold",
        )
    if authority.renderer_owned:
        if not isinstance(transition, Mapping):
            reasons.append("renderer_owned_transition_metadata_missing")
            transition_count = 0
            histogram: dict[str, Any] = {}
        else:
            raw_count = transition.get("label_count")
            transition_count = (
                raw_count
                if isinstance(raw_count, int) and not isinstance(raw_count, bool)
                else 0
            )
            histogram = (
                dict(transition.get("label_histogram"))
                if isinstance(transition.get("label_histogram"), Mapping)
                else {}
            )
            minimum_count(
                transition_count,
                thresholds.minimum_transition_labels,
                "insufficient_heldout_renderer_owned_transition_metadata_labels",
            )
            if set(histogram) - {"0"} or sum(
                value
                for value in histogram.values()
                if isinstance(value, int) and not isinstance(value, bool)
            ) != transition_count:
                reasons.append("heldout_renderer_owned_transition_metadata_not_hard_cut")
        reported_metrics["transition"] = {
            "status": "skipped_non_authoritative",
            "reason": RENDERER_OWNED_LEARNED_EVALUATION_SKIP_REASON,
            "label_count": transition_count,
            "allowed_model_metadata_labels": list(
                RENDERER_OWNED_ALLOWED_MODEL_LABELS
            ),
            "observed_model_label_histogram": histogram,
            "learned_accuracy": None,
            "learned_logits_consumed_for_quality": False,
        }
    elif not isinstance(transition, Mapping):
        reasons.append("transition_metrics_missing")
    else:
        minimum_count(
            transition.get("label_count"),
            thresholds.minimum_transition_labels,
            "insufficient_heldout_transition_labels",
        )
        minimum_metric(
            transition.get("accuracy"),
            thresholds.minimum_transition_accuracy,
            "heldout_transition_accuracy_below_threshold",
        )
        minimum_count(
            transition.get("observed_class_count"),
            thresholds.minimum_transition_observed_classes,
            "insufficient_heldout_transition_class_coverage",
        )
        minimum_metric(
            transition.get("label_entropy_bits"),
            thresholds.minimum_transition_label_entropy_bits,
            "heldout_transition_label_entropy_below_threshold",
        )
    dynamic_crop_v2_active = isinstance(dynamic_crop_v2_metrics, Mapping)
    dynamic_crop_v3_active = isinstance(dynamic_crop_v3_metrics, Mapping)
    if not (dynamic_crop_v2_active or dynamic_crop_v3_active):
        if not isinstance(crop, Mapping) or not isinstance(
            crop.get("coordinates"), Mapping
        ):
            reasons.append("crop_coordinate_metrics_missing")
        else:
            coordinates = crop["coordinates"]
            maximums = {
                "zoom": thresholds.maximum_zoom_mae,
                "center_x": thresholds.maximum_center_mae,
                "center_y": thresholds.maximum_center_mae,
                "roll_degrees": thresholds.maximum_roll_mae_degrees,
            }
            for name, maximum in maximums.items():
                coordinate = coordinates.get(name)
                if not isinstance(coordinate, Mapping):
                    reasons.append(f"heldout_crop_{name}_metrics_missing")
                    continue
                minimum_count(
                    coordinate.get("label_count"),
                    thresholds.minimum_crop_labels_per_coordinate,
                    f"insufficient_heldout_crop_{name}_labels",
                )
                maximum_metric(
                    coordinate.get("mean_absolute_error"),
                    maximum,
                    f"heldout_crop_{name}_mae_above_threshold",
                )
    trajectory = None
    if dynamic_crop_v2_active:
        trajectory = dynamic_crop_v2_metrics.get("trajectory")
    elif dynamic_crop_v3_active:
        trajectory = dynamic_crop_v3_metrics.get("trajectory")
    if not isinstance(trajectory, Mapping):
        reasons.append(
            "dynamic_crop_v3_metrics_missing"
            if dynamic_crop_v3_active
            else "dynamic_crop_v2_metrics_missing"
        )
    elif dynamic_crop_v2_active:
        minimum_count(
            trajectory.get("track_label_count"),
            thresholds.minimum_dynamic_crop_tracks,
            "insufficient_heldout_dynamic_crop_tracks",
        )
        minimum_count(
            trajectory.get("keyframe_label_count"),
            thresholds.minimum_dynamic_crop_keyframes,
            "insufficient_heldout_dynamic_crop_keyframes",
        )
        coordinates = trajectory.get("coordinates")
        dynamic_maximums = {
            "zoom": thresholds.maximum_dynamic_crop_zoom_mae,
            "center_x": thresholds.maximum_dynamic_crop_center_mae,
            "center_y": thresholds.maximum_dynamic_crop_center_mae,
            "roll_degrees": thresholds.maximum_dynamic_crop_roll_mae_degrees,
        }
        if not isinstance(coordinates, Mapping):
            reasons.append("dynamic_crop_v2_coordinate_metrics_missing")
        else:
            for name, maximum in dynamic_maximums.items():
                coordinate = coordinates.get(name)
                if not isinstance(coordinate, Mapping):
                    reasons.append(
                        f"heldout_dynamic_crop_{name}_metrics_missing"
                    )
                    continue
                minimum_count(
                    coordinate.get("label_count"),
                    thresholds.minimum_dynamic_crop_keyframes,
                    f"insufficient_heldout_dynamic_crop_{name}_labels",
                )
                maximum_metric(
                    coordinate.get("mean_absolute_error"),
                    maximum,
                    f"heldout_dynamic_crop_{name}_mae_above_threshold",
                )
        maximum_metric(
            trajectory.get("normalized_time_mae"),
            thresholds.maximum_dynamic_crop_normalized_time_mae,
            "heldout_dynamic_crop_time_mae_above_threshold",
        )
        presence = trajectory.get("presence")
        if not isinstance(presence, Mapping):
            reasons.append("dynamic_crop_v2_presence_metrics_missing")
        else:
            minimum_metric(
                presence.get("f1"),
                thresholds.minimum_dynamic_crop_presence_f1,
                "heldout_dynamic_crop_presence_f1_below_threshold",
            )
            minimum_metric(
                presence.get("keyframe_count_exact_accuracy"),
                thresholds.minimum_dynamic_crop_keyframe_count_accuracy,
                "heldout_dynamic_crop_keyframe_count_below_threshold",
            )
        motion = trajectory.get("motion")
        if not isinstance(motion, Mapping):
            reasons.append("dynamic_crop_v2_motion_metrics_missing")
        else:
            minimum_count(
                motion.get("label_count"),
                thresholds.minimum_dynamic_crop_motion_labels,
                "insufficient_heldout_dynamic_crop_motion_labels",
            )
            minimum_metric(
                motion.get("accuracy"),
                thresholds.minimum_dynamic_crop_motion_accuracy,
                "heldout_dynamic_crop_motion_accuracy_below_threshold",
            )
        renderer_decode = trajectory.get("renderer_decode")
        if not isinstance(renderer_decode, Mapping):
            reasons.append("dynamic_crop_v2_renderer_decode_metrics_missing")
        else:
            minimum_count(
                renderer_decode.get("attempt_count"),
                thresholds.minimum_dynamic_crop_tracks,
                "insufficient_heldout_dynamic_crop_renderer_attempts",
            )
            minimum_metric(
                renderer_decode.get("eligible_rate"),
                thresholds.minimum_dynamic_crop_renderer_decode_rate,
                "heldout_dynamic_crop_renderer_decode_rate_below_threshold",
            )
    else:
        minimum_count(
            trajectory.get("track_label_count"),
            thresholds.minimum_dynamic_crop_tracks,
            "insufficient_heldout_dynamic_crop_tracks",
        )
        minimum_count(
            trajectory.get("keyframe_label_count"),
            thresholds.minimum_dynamic_crop_keyframes,
            "insufficient_heldout_dynamic_crop_keyframes",
        )
        coordinates = trajectory.get("coordinates")
        dynamic_maximums = {
            "zoom": thresholds.maximum_dynamic_crop_zoom_mae,
            "center_x": thresholds.maximum_dynamic_crop_center_mae,
            "center_y": thresholds.maximum_dynamic_crop_center_mae,
            "roll_degrees": thresholds.maximum_dynamic_crop_roll_mae_degrees,
        }
        if not isinstance(coordinates, Mapping):
            reasons.append("dynamic_crop_v3_coordinate_metrics_missing")
        else:
            for name, maximum in dynamic_maximums.items():
                coordinate = coordinates.get(name)
                if not isinstance(coordinate, Mapping):
                    reasons.append(
                        f"heldout_dynamic_crop_{name}_metrics_missing"
                    )
                    continue
                minimum_count(
                    coordinate.get("label_count"),
                    thresholds.minimum_dynamic_crop_keyframes,
                    f"insufficient_heldout_dynamic_crop_{name}_labels",
                )
                maximum_metric(
                    coordinate.get("mean_absolute_error"),
                    maximum,
                    f"heldout_dynamic_crop_{name}_mae_above_threshold",
                )
        maximum_metric(
            trajectory.get("teacher_forced_count_normalized_time_mae"),
            thresholds.maximum_dynamic_crop_normalized_time_mae,
            "heldout_dynamic_crop_time_mae_above_threshold",
        )
        count = trajectory.get("count")
        if not isinstance(count, Mapping):
            reasons.append("dynamic_crop_v3_count_metrics_missing")
        else:
            minimum_metric(
                count.get("exact_accuracy"),
                thresholds.minimum_dynamic_crop_keyframe_count_accuracy,
                "heldout_dynamic_crop_keyframe_count_below_threshold",
            )
        motion = trajectory.get("motion")
        if not isinstance(motion, Mapping):
            reasons.append("dynamic_crop_v3_motion_metrics_missing")
        else:
            minimum_count(
                motion.get("label_count"),
                thresholds.minimum_dynamic_crop_motion_labels,
                "insufficient_heldout_dynamic_crop_motion_labels",
            )
            minimum_metric(
                motion.get("accuracy"),
                thresholds.minimum_dynamic_crop_motion_accuracy,
                "heldout_dynamic_crop_motion_accuracy_below_threshold",
            )
        renderer_decode = trajectory.get("renderer_decode")
        if not isinstance(renderer_decode, Mapping):
            reasons.append("dynamic_crop_v3_renderer_decode_metrics_missing")
        else:
            minimum_count(
                renderer_decode.get("attempt_count"),
                thresholds.minimum_dynamic_crop_tracks,
                "insufficient_heldout_dynamic_crop_renderer_attempts",
            )
            renderer_rate = _finite_metric(renderer_decode.get("eligible_rate"))
            if renderer_rate != 1.0:
                reasons.append("heldout_dynamic_crop_renderer_decode_not_exact")
            projection_count = renderer_decode.get(
                "minimum_safe_zoom_projection_count"
            )
            if (
                isinstance(projection_count, bool)
                or not isinstance(projection_count, int)
                or projection_count != 0
            ):
                reasons.append("heldout_dynamic_crop_projection_not_zero")
        source_slot_mismatch_count = trajectory.get(
            "target_source_slot_mismatch_count"
        )
        if (
            isinstance(source_slot_mismatch_count, bool)
            or not isinstance(source_slot_mismatch_count, int)
            or source_slot_mismatch_count != 0
        ):
            reasons.append("heldout_dynamic_crop_source_slot_mismatch")
        aggregate_improvement = _finite_metric(
            trajectory.get(
                "aggregate_normalized_improvement_over_static_prior"
            )
        )
        if aggregate_improvement is None or aggregate_improvement <= 0.0:
            reasons.append("heldout_dynamic_crop_did_not_beat_static_prior")
    return ProductionQualityGate(
        passed=not reasons,
        reasons=tuple(reasons),
        thresholds=thresholds,
        metrics=reported_metrics,
    )


@dataclass(frozen=True, slots=True)
class ProductionSongSplit:
    train: tuple[ReferenceTrainingExample, ...]
    validation: tuple[ReferenceTrainingExample, ...]
    held_out: tuple[ReferenceTrainingExample, ...]
    training_song_ids: tuple[str, ...]
    held_out_song_ids: tuple[str, ...]
    required_training_song_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_level": "song",
            "training_song_ids": list(self.training_song_ids),
            "held_out_song_ids": list(self.held_out_song_ids),
            "required_training_song_ids": list(
                self.required_training_song_ids
            ),
            "training_reference_ids": [item.reference_id for item in self.train],
            "model_selection_validation_level": "whole_reference_within_training_songs",
            "validation_reference_ids": [
                item.reference_id for item in self.validation
            ],
            "held_out_reference_ids": [item.reference_id for item in self.held_out],
            "held_out_used_for_model_selection": False,
        }


def production_song_split(
    corpus: ReferenceTrainingCorpus,
    *,
    held_out_song_ids: Sequence[str],
    required_training_song_ids: Sequence[str] = (),
) -> ProductionSongSplit:
    """Keep every reference for each named held-out song out of training."""

    normalized_held_out = tuple(sorted({str(value).strip() for value in held_out_song_ids}))
    normalized_required_training = tuple(
        sorted({str(value).strip() for value in required_training_song_ids})
    )
    if not normalized_held_out or any(not value for value in normalized_held_out):
        raise ValueError("at least one non-empty held-out song id is required")
    if any(not value for value in normalized_required_training):
        raise ValueError("required training song ids cannot be blank")
    if set(normalized_required_training) & set(normalized_held_out):
        raise ValueError("a required training song cannot also be held out")
    if any(example.song_id is None for example in corpus.examples):
        raise ValueError("production training requires explicit song ids for every reference")
    available = {str(example.song_id) for example in corpus.examples}
    if not set(normalized_held_out) <= available:
        missing = sorted(set(normalized_held_out) - available)
        raise ValueError(f"held-out songs are absent from corpus: {', '.join(missing)}")
    training_song_ids = tuple(sorted(available - set(normalized_held_out)))
    if len(training_song_ids) < 2:
        raise ValueError("production training requires at least two complete training songs")
    missing_required_training = sorted(
        set(normalized_required_training) - set(training_song_ids)
    )
    if missing_required_training:
        raise ValueError(
            "required training songs are absent from the training partition: "
            + ", ".join(missing_required_training)
        )
    training_by_song: dict[str, list[ReferenceTrainingExample]] = {
        song_id: [] for song_id in training_song_ids
    }
    for example in corpus.examples:
        song_id = str(example.song_id)
        if song_id in training_by_song:
            training_by_song[song_id].append(example)
    if any(len(examples) < 2 for examples in training_by_song.values()):
        raise ValueError(
            "production model selection requires at least two complete "
            "references per training song"
        )

    # Reserve one whole reference from every training song for checkpoint
    # selection.  The unseen song must not influence early stopping or the
    # chosen epoch.  Selection is stable and independent of the training seed.
    validation_ids = {
        sorted(
            examples,
            key=lambda example: hashlib.sha256(
                f"{song_id}\0{example.reference_id}".encode("utf-8")
            ).hexdigest(),
        )[0].reference_id
        for song_id, examples in training_by_song.items()
    }
    validation = tuple(
        sorted(
            (
                example
                for examples in training_by_song.values()
                for example in examples
                if example.reference_id in validation_ids
            ),
            key=lambda example: (str(example.song_id), example.reference_id),
        )
    )
    train = tuple(
        sorted(
            (
                example
                for examples in training_by_song.values()
                for example in examples
                if example.reference_id not in validation_ids
            ),
            key=lambda example: (str(example.song_id), example.reference_id),
        )
    )
    held_out = tuple(
        sorted(
            (
                example
                for example in corpus.examples
                if str(example.song_id) in normalized_held_out
            ),
            key=lambda example: (str(example.song_id), example.reference_id),
        )
    )
    if not train or not validation or not held_out:
        raise ValueError("production song split produced an empty fold")
    all_ids = [
        item.reference_id
        for partition in (train, validation, held_out)
        for item in partition
    ]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("production split requires globally unique reference ids")
    return ProductionSongSplit(
        train=train,
        validation=validation,
        held_out=held_out,
        training_song_ids=training_song_ids,
        held_out_song_ids=normalized_held_out,
        required_training_song_ids=normalized_required_training,
    )


def stage_edit_training_song_expansion_contract(
    split: ProductionSongSplit,
) -> dict[str, Any]:
    """Audit safe admission of a newly materialized complete training song."""

    training_references_by_song: dict[str, list[str]] = {}
    validation_references_by_song: dict[str, list[str]] = {}
    for partition, destination in (
        (split.train, training_references_by_song),
        (split.validation, validation_references_by_song),
    ):
        for example in partition:
            destination.setdefault(str(example.song_id), []).append(
                example.reference_id
            )
    unsigned: dict[str, Any] = {
        "contract": STAGE_EDIT_TRAINING_SONG_EXPANSION_CONTRACT,
        "required_training_song_ids": list(split.required_training_song_ids),
        "observed_training_song_ids": list(split.training_song_ids),
        "training_references_by_song": {
            key: sorted(values)
            for key, values in sorted(training_references_by_song.items())
        },
        "reference_validation_references_by_song": {
            key: sorted(values)
            for key, values in sorted(validation_references_by_song.items())
        },
        "one_complete_reference_reserved_per_training_song": all(
            len(validation_references_by_song.get(song_id, ())) == 1
            for song_id in split.training_song_ids
        ),
        "minimum_two_references_per_training_song_enforced": True,
        "feature_snapshot_gate_required_separately": True,
        "heldout_songs_disjoint": set(split.training_song_ids).isdisjoint(
            split.held_out_song_ids
        ),
        "style_or_attitude_configuration_changed": False,
    }
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        **unsigned,
        "contract_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _categorical_target_coverage(
    examples: Sequence[ReferenceTrainingExample],
    *,
    field: str,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for example in examples:
        for target in example.targets.targets:
            value = getattr(target, field)
            if value is None:
                continue
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
    label_count = sum(counts.values())
    if label_count:
        probabilities = tuple(count / label_count for count in counts.values())
        entropy = -sum(
            probability * math.log2(probability)
            for probability in probabilities
            if probability > 0.0
        )
    else:
        entropy = None
    return {
        "label_count": label_count,
        "observed_class_count": len(counts),
        "label_entropy_bits": entropy,
        "label_histogram": dict(sorted(counts.items())),
    }


def _cut_prevalence_by_song(
    examples: Sequence[ReferenceTrainingExample],
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for example in examples:
        song_id = str(example.song_id or "").strip()
        if not song_id:
            raise ValueError("cut prevalence audit requires explicit song ids")
        grouped.setdefault(song_id, []).extend(
            float(target.cut)
            for target in example.targets.targets
            if target.cut is not None
        )
    return {
        song_id: {
            "label_count": len(values),
            "positive_count": sum(value >= 0.5 for value in values),
            "negative_count": sum(value < 0.5 for value in values),
            "positive_prevalence": (
                sum(value >= 0.5 for value in values) / len(values)
                if values
                else None
            ),
        }
        for song_id, values in sorted(grouped.items())
    }


def _structured_head_coverage_gate(
    training_examples: Sequence[ReferenceTrainingExample],
    held_out_metrics: Mapping[str, Any],
    *,
    thresholds: ProductionQualityThresholds,
    transition_authority: str = LEARNED_TRANSITION_AUTHORITY,
) -> dict[str, Any]:
    """Reject trivial duration/transition accuracies on either data side."""

    authority = StageEditTransitionAuthority(mode=transition_authority)

    training = {
        "duration": _categorical_target_coverage(
            training_examples, field="duration_bucket"
        ),
        "transition": _categorical_target_coverage(
            training_examples, field="transition_label"
        ),
    }
    held_out = {
        name: dict(held_out_metrics.get(name, {}))
        if isinstance(held_out_metrics.get(name), Mapping)
        else {}
        for name in ("duration", "transition")
    }
    reasons: list[str] = []

    def require_side(
        side_name: str,
        name: str,
        evidence: Mapping[str, Any],
        *,
        minimum_classes: int,
        minimum_entropy: float,
    ) -> None:
        observed = evidence.get("observed_class_count")
        entropy = _finite_metric(evidence.get("label_entropy_bits"))
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed < minimum_classes
        ):
            reasons.append(f"{side_name}_{name}_class_coverage_insufficient")
        if entropy is None or entropy < minimum_entropy:
            reasons.append(f"{side_name}_{name}_label_entropy_insufficient")

    for side_name, side in (("training", training), ("heldout", held_out)):
        require_side(
            side_name,
            "duration",
            side["duration"],
            minimum_classes=thresholds.minimum_duration_observed_classes,
            minimum_entropy=thresholds.minimum_duration_label_entropy_bits,
        )
        if authority.renderer_owned:
            transition = side["transition"]
            histogram = transition.get("label_histogram")
            label_count = transition.get("label_count")
            expected_histogram = {
                "hard_cut" if side_name == "training" else "0": label_count
            }
            if (
                not isinstance(label_count, int)
                or isinstance(label_count, bool)
                or label_count < 1
                or histogram != expected_histogram
            ):
                reasons.append(
                    f"{side_name}_renderer_owned_transition_metadata_not_exact_hard_cut"
                )
            side["transition"] = {
                "status": "skipped_non_authoritative",
                "reason": RENDERER_OWNED_LEARNED_EVALUATION_SKIP_REASON,
                "label_count": label_count,
                "allowed_model_metadata_labels": list(
                    RENDERER_OWNED_ALLOWED_MODEL_LABELS
                ),
                "label_histogram": (
                    {"hard_cut": label_count}
                    if isinstance(label_count, int)
                    and not isinstance(label_count, bool)
                    else histogram
                ),
                "learned_accuracy_consumed": False,
                "learned_entropy_consumed": False,
            }
        else:
            require_side(
                side_name,
                "transition",
                side["transition"],
                minimum_classes=thresholds.minimum_transition_observed_classes,
                minimum_entropy=thresholds.minimum_transition_label_entropy_bits,
            )
    return {
        "contract": STRUCTURED_HEAD_COVERAGE_CONTRACT,
        "passed": not reasons,
        "reasons": reasons,
        "thresholds": {
            "minimum_duration_observed_classes": (
                thresholds.minimum_duration_observed_classes
            ),
            "minimum_duration_label_entropy_bits": (
                thresholds.minimum_duration_label_entropy_bits
            ),
            "minimum_transition_observed_classes": (
                thresholds.minimum_transition_observed_classes
            ),
            "minimum_transition_label_entropy_bits": (
                thresholds.minimum_transition_label_entropy_bits
            ),
        },
        "transition_authority": transition_authority_contract(
            transition_authority
        ),
        "training_partition": training,
        "heldout": held_out,
        "reference_validation_labels_consumed": False,
        "heldout_used_for_model_selection": False,
    }


@dataclass(frozen=True, slots=True)
class ProductionTrainingRunResult:
    checkpoint_directory: Path
    report_path: Path
    best_epoch: int
    best_training_step: int
    quality_gate: ProductionQualityGate
    production_ready: bool
    report: Mapping[str, Any]


def _stable_epoch_order(
    examples: Sequence[ReferenceTrainingExample],
    *,
    seed: int,
    epoch: int,
) -> tuple[ReferenceTrainingExample, ...]:
    ordered = list(examples)
    random.Random((seed << 16) ^ epoch).shuffle(ordered)
    return tuple(ordered)


def _batches(
    examples: Sequence[ReferenceTrainingExample],
    batch_size: int,
) -> tuple[tuple[ReferenceTrainingExample, ...], ...]:
    return tuple(
        tuple(examples[start : start + batch_size])
        for start in range(0, len(examples), batch_size)
    )


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_selection_freeze(
    checkpoint_root: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist immutable model/threshold selection before any held-out forward."""

    path = checkpoint_root / STAGE_EDIT_SELECTION_FREEZE_FILENAME
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite selection freeze: {path}")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    path.write_bytes(encoded)
    return {
        "path": str(path),
        "filename": path.name,
        "file_sha256": hashlib.sha256(encoded).hexdigest(),
        "content_sha256": _canonical_digest(payload),
        "written_before_heldout_forward": True,
    }


@dataclass(frozen=True, slots=True)
class _BalancedCutWindow:
    sequence: StageEditSequence
    targets: StageEditTargetSequence
    label: int
    reference_id: str
    anchor_token_index: int


@dataclass(frozen=True, slots=True)
class _SourcePretrainingWindow:
    sequence: StageEditSequence
    targets: StageEditTargetSequence
    song_id: str
    reference_id: str
    group_id: str
    anchor_token_index: int


@dataclass(frozen=True, slots=True)
class _SourcePretrainingInventory:
    windows: tuple[_SourcePretrainingWindow, ...]
    opening_source_excluded_count: int
    unconfirmed_source_label_excluded_count: int
    per_reference_confirmed_group_counts: Mapping[str, int]
    reference_song_ids: Mapping[str, str]


def _stable_sampler_order(
    values: Sequence[Any],
    *,
    seed: int,
    scope: str,
) -> tuple[Any, ...]:
    ordered = sorted(values, key=str)
    digest = hashlib.sha256(f"{seed}:{scope}".encode("utf-8")).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(ordered)
    return tuple(ordered)


class _HierarchicalSourceWindowSampler:
    """Deterministic song -> reference -> source-event round robin."""

    def __init__(
        self,
        windows: Sequence[_SourcePretrainingWindow],
        *,
        seed: int,
    ) -> None:
        nested: dict[str, dict[str, list[_SourcePretrainingWindow]]] = {}
        for window in windows:
            nested.setdefault(window.song_id, {}).setdefault(
                window.reference_id, []
            ).append(window)
        if not nested:
            raise ValueError("source pretraining requires confirmed source events")
        self._song_order = _stable_sampler_order(
            tuple(nested), seed=seed, scope="source-pretrain:songs"
        )
        self._reference_orders = {
            song_id: _stable_sampler_order(
                tuple(references),
                seed=seed,
                scope=f"source-pretrain:references:{song_id}",
            )
            for song_id, references in nested.items()
        }
        self._group_orders = {
            (song_id, reference_id): _stable_sampler_order(
                tuple(groups),
                seed=seed,
                scope=(
                    "source-pretrain:groups:"
                    f"{song_id}:{reference_id}"
                ),
            )
            for song_id, references in nested.items()
            for reference_id, groups in references.items()
        }
        self._song_visit_counts = {song_id: 0 for song_id in nested}
        self._reference_visit_counts = {
            key: 0 for key in self._group_orders
        }
        self._per_song_sampled_counts = {song_id: 0 for song_id in nested}
        self._per_reference_sampled_counts = {
            reference_id: 0
            for references in nested.values()
            for reference_id in references
        }
        self._sample_count = 0

    def sample(self) -> _SourcePretrainingWindow:
        song_id = self._song_order[self._sample_count % len(self._song_order)]
        song_visit = self._song_visit_counts[song_id]
        references = self._reference_orders[song_id]
        reference_id = references[song_visit % len(references)]
        reference_key = (song_id, reference_id)
        reference_visit = self._reference_visit_counts[reference_key]
        groups = self._group_orders[reference_key]
        window = groups[reference_visit % len(groups)]

        self._sample_count += 1
        self._song_visit_counts[song_id] += 1
        self._reference_visit_counts[reference_key] += 1
        self._per_song_sampled_counts[song_id] += 1
        self._per_reference_sampled_counts[reference_id] += 1
        return window

    @property
    def per_song_sampled_counts(self) -> dict[str, int]:
        return dict(sorted(self._per_song_sampled_counts.items()))

    @property
    def per_reference_sampled_counts(self) -> dict[str, int]:
        return dict(sorted(self._per_reference_sampled_counts.items()))


def _source_pretraining_inventory(
    examples: Sequence[ReferenceTrainingExample],
    *,
    schema: StageEditTokenSchema,
    window_length: int,
) -> _SourcePretrainingInventory:
    """Keep only confirmed, non-opening editorial source switches."""

    if window_length < 2:
        raise ValueError("source pretraining window must contain >= 2 tokens")
    windows: list[_SourcePretrainingWindow] = []
    opening_source_excluded_count = 0
    unconfirmed_source_label_excluded_count = 0
    reference_song_ids: dict[str, str] = {}
    per_reference_confirmed_group_counts: dict[str, int] = {}
    prepared_sequences = prepare_stage_edit_production_input_sequences(
        examples, schema=schema
    )
    for example, prepared in zip(examples, prepared_sequences, strict=True):
        if len(prepared.tokens) != len(example.targets.targets):
            raise ValueError("source pretraining sequence/target length mismatch")
        song_id = str(example.song_id or "").strip()
        for anchor_index, target in enumerate(example.targets.targets):
            if anchor_index == 0 and target.selected_source_id is not None:
                opening_source_excluded_count += 1
                continue
            if target.selected_source_id is None:
                continue
            if target.cut != 1.0:
                unconfirmed_source_label_excluded_count += 1
                continue
            if not song_id:
                raise ValueError(
                    "confirmed source pretraining events require explicit song ids"
                )
            observed_song = reference_song_ids.setdefault(
                example.reference_id, song_id
            )
            if observed_song != song_id:
                raise ValueError("source pretraining reference spans multiple songs")
            start = max(0, anchor_index - window_length + 1)
            group_id = (
                f"{example.reference_id}:confirmed-source:{anchor_index}"
            )
            token_values = tuple(prepared.tokens[start : anchor_index + 1])
            target_values = tuple(
                StageEditTarget() for _ in token_values[:-1]
            ) + (
                StageEditTarget(
                    selected_source_id=target.selected_source_id,
                    rejected_source_ids=target.rejected_source_ids,
                ),
            )
            identifier = f"{prepared.sequence_id}:{group_id}"
            windows.append(
                _SourcePretrainingWindow(
                    sequence=StageEditSequence(identifier, token_values),
                    targets=StageEditTargetSequence(identifier, target_values),
                    song_id=song_id,
                    reference_id=example.reference_id,
                    group_id=group_id,
                    anchor_token_index=anchor_index,
                )
            )
            per_reference_confirmed_group_counts[example.reference_id] = (
                per_reference_confirmed_group_counts.get(example.reference_id, 0)
                + 1
            )
    if not windows:
        raise ValueError(
            "source pretraining requires confirmed non-opening editorial switches"
        )
    return _SourcePretrainingInventory(
        windows=tuple(windows),
        opening_source_excluded_count=opening_source_excluded_count,
        unconfirmed_source_label_excluded_count=(
            unconfirmed_source_label_excluded_count
        ),
        per_reference_confirmed_group_counts=dict(
            sorted(per_reference_confirmed_group_counts.items())
        ),
        reference_song_ids=dict(sorted(reference_song_ids.items())),
    )


@dataclass(frozen=True, slots=True)
class _DynamicCropV3PriorTrack:
    song_id: str
    reference_id: str
    sequence_id: str
    token_index: int
    values: tuple[tuple[float, float, float, float], ...]
    normalized_times: tuple[float, ...]
    motion_label: str


@dataclass(frozen=True, slots=True)
class _DynamicCropV3PriorInventory:
    tracks: tuple[_DynamicCropV3PriorTrack, ...]
    per_reference_track_counts: Mapping[str, int]
    references_without_tracks: tuple[str, ...]
    reference_song_ids: Mapping[str, str]


class _HierarchicalDynamicCropV3PriorSampler:
    """Deterministic song -> reference -> trajectory round robin."""

    def __init__(
        self,
        tracks: Sequence[_DynamicCropV3PriorTrack],
        *,
        seed: int,
    ) -> None:
        nested: dict[str, dict[str, list[_DynamicCropV3PriorTrack]]] = {}
        for track in tracks:
            nested.setdefault(track.song_id, {}).setdefault(
                track.reference_id, []
            ).append(track)
        if not nested:
            raise ValueError("Dynamic Crop v3 prior warmup needs trajectory labels")
        self._song_order = _stable_sampler_order(
            tuple(nested), seed=seed, scope="dynamic-crop-v3-prior:songs"
        )
        self._reference_orders = {
            song_id: _stable_sampler_order(
                tuple(references),
                seed=seed,
                scope=f"dynamic-crop-v3-prior:references:{song_id}",
            )
            for song_id, references in nested.items()
        }
        self._track_orders = {
            (song_id, reference_id): _stable_sampler_order(
                tuple(values),
                seed=seed,
                scope=(
                    "dynamic-crop-v3-prior:tracks:"
                    f"{song_id}:{reference_id}"
                ),
            )
            for song_id, references in nested.items()
            for reference_id, values in references.items()
        }
        self._song_visits = {song_id: 0 for song_id in nested}
        self._reference_visits = {key: 0 for key in self._track_orders}
        self._per_song_sampled_counts = {song_id: 0 for song_id in nested}
        self._per_reference_sampled_counts = {
            reference_id: 0
            for references in nested.values()
            for reference_id in references
        }
        self._sample_count = 0

    def sample(self) -> _DynamicCropV3PriorTrack:
        song_id = self._song_order[self._sample_count % len(self._song_order)]
        song_visit = self._song_visits[song_id]
        references = self._reference_orders[song_id]
        reference_id = references[song_visit % len(references)]
        key = (song_id, reference_id)
        reference_visit = self._reference_visits[key]
        tracks = self._track_orders[key]
        track = tracks[reference_visit % len(tracks)]
        self._sample_count += 1
        self._song_visits[song_id] += 1
        self._reference_visits[key] += 1
        self._per_song_sampled_counts[song_id] += 1
        self._per_reference_sampled_counts[reference_id] += 1
        return track

    @property
    def per_song_sampled_counts(self) -> dict[str, int]:
        return dict(sorted(self._per_song_sampled_counts.items()))

    @property
    def per_reference_sampled_counts(self) -> dict[str, int]:
        return dict(sorted(self._per_reference_sampled_counts.items()))


def _dynamic_crop_v3_prior_inventory(
    examples: Sequence[ReferenceTrainingExample],
    corpus: StageEditDynamicCropTrainingCorpusV3,
) -> _DynamicCropV3PriorInventory:
    """Extract only training-reference trajectories for the low-capacity prior."""

    tracks: list[_DynamicCropV3PriorTrack] = []
    per_reference_track_counts: dict[str, int] = {}
    references_without_tracks: list[str] = []
    reference_song_ids: dict[str, str] = {}
    for example in examples:
        song_id = str(example.song_id or "").strip()
        if not song_id:
            raise ValueError("Dynamic Crop v3 prior tracks require song ids")
        reference_song_ids[example.reference_id] = song_id
        sequence = corpus.sequence_for(example.sequence.sequence_id)
        reference_count = 0
        for token_index, target in enumerate(sequence.targets):
            if target is None:
                continue
            values = tuple(
                (
                    float(item.zoom),
                    float(item.center_x),
                    float(item.center_y),
                    float(item.roll_degrees),
                )
                for item in target.keyframes
            )
            if not 1 <= len(values) <= DYNAMIC_CROP_V3_MAX_KEYFRAMES:
                raise ValueError("Dynamic Crop v3 prior target count is invalid")
            duration = float(target.shot_duration_seconds)
            normalized_times = tuple(
                float(item.time_seconds) / duration
                for item in target.keyframes
            )
            tracks.append(
                _DynamicCropV3PriorTrack(
                    song_id=song_id,
                    reference_id=example.reference_id,
                    sequence_id=example.sequence.sequence_id,
                    token_index=token_index,
                    values=values,
                    normalized_times=normalized_times,
                    motion_label=str(target.motion_label),
                )
            )
            reference_count += 1
        if reference_count:
            per_reference_track_counts[example.reference_id] = reference_count
        else:
            references_without_tracks.append(example.reference_id)
    if not tracks:
        raise ValueError("Dynamic Crop v3 prior warmup has no training tracks")
    return _DynamicCropV3PriorInventory(
        tracks=tuple(tracks),
        per_reference_track_counts=dict(
            sorted(per_reference_track_counts.items())
        ),
        references_without_tracks=tuple(sorted(references_without_tracks)),
        reference_song_ids=dict(sorted(reference_song_ids.items())),
    )


_DynamicCropV3PriorModuleBase = nn.Module if nn is not None else object


class _DynamicCropV3PriorBiasModule(_DynamicCropV3PriorModuleBase):
    """A detached 8x4 copy so no production-model parameter can drift."""

    def __init__(self, value_bias: Any) -> None:
        _require_mlx()
        super().__init__()
        self.value_bias = mx.array(value_bias).reshape(
            (DYNAMIC_CROP_V3_MAX_KEYFRAMES, 4)
        )


def _dynamic_crop_v3_prior_normalized_mae(
    module: _DynamicCropV3PriorBiasModule,
    target_values: Any,
) -> Any:
    """The exact normalized coordinate MAE consumed by the >0 gate."""

    predicted = renderer_range_normalized_values_v3(
        renderer_safe_values_from_raw_v3(module.value_bias)
    )
    targets = renderer_range_normalized_values_v3(target_values)
    count = int(target_values.shape[0])
    return mx.abs(predicted[:count] - targets).mean()


def _parameter_tree_sha256(tree: Any) -> str:
    if tree_flatten is None:
        raise RuntimeError("MLX tree utilities are unavailable")
    digest = hashlib.sha256()
    for name, value in sorted(tree_flatten(tree), key=lambda item: item[0]):
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(name).encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _zero_gradient_tree(tree: Any) -> Any:
    if isinstance(tree, Mapping):
        return {key: _zero_gradient_tree(value) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_zero_gradient_tree(value) for value in tree)
    if isinstance(tree, list):
        return [_zero_gradient_tree(value) for value in tree]
    return mx.zeros_like(tree)


def _dynamic_crop_v3_prior_parameter_digests(
    model: StageEditTransformerDynamicCropV3,
) -> dict[str, str]:
    values_stop = DYNAMIC_CROP_V3_MAX_KEYFRAMES * 4
    bias = model.dynamic_crop_context_head.bias
    return {
        "base_parameter_tree_sha256": _parameter_tree_sha256(
            model.base.parameters()
        ),
        "context_weight_sha256": _parameter_tree_sha256(
            {"weight": model.dynamic_crop_context_head.weight}
        ),
        "source_head_parameter_tree_sha256": _parameter_tree_sha256(
            model.dynamic_crop_source_head.parameters()
        ),
        "context_non_value_bias_sha256": _parameter_tree_sha256(
            {"bias": bias[values_stop:]}
        ),
        "context_value_bias_sha256": _parameter_tree_sha256(
            {"bias": bias[:values_stop]}
        ),
    }


def _dynamic_crop_v3_prior_component_scale_evidence(
    inventory: _DynamicCropV3PriorInventory,
    initial_values: np.ndarray,
    contract: DynamicCropV3LossContract,
) -> dict[str, Any]:
    residuals: list[float] = []
    time_residuals: list[float] = []
    keyframe_count = 0
    for track in inventory.tracks:
        count = len(track.values)
        targets = np.asarray(track.values, dtype=np.float64)
        scales = np.asarray((0.18, 0.04, 0.04, 1.60))
        normalized_residual = np.abs(
            (initial_values[:count] - targets) / scales
        )
        residuals.extend(normalized_residual.reshape(-1).tolist())
        expected_times = (
            np.arange(count, dtype=np.float64) / count
            if count > 1
            else np.zeros(1, dtype=np.float64)
        )
        time_residuals.extend(
            np.abs(
                expected_times
                - np.asarray(track.normalized_times, dtype=np.float64)
            ).tolist()
        )
        keyframe_count += count
    value_residual = np.asarray(residuals, dtype=np.float64)
    time_residual = np.asarray(time_residuals, dtype=np.float64)

    def smooth_l1(values: np.ndarray) -> float:
        absolute = np.abs(values)
        return float(
            np.mean(
                np.where(absolute < 1.0, 0.5 * absolute * absolute, absolute - 0.5)
            )
        )

    value_smooth_l1 = smooth_l1(value_residual)
    time_smooth_l1 = smooth_l1(time_residual)
    weighted_value = contract.value_weight * value_smooth_l1
    weighted_count = contract.count_weight * math.log(
        DYNAMIC_CROP_V3_MAX_KEYFRAMES
    )
    weighted_time = contract.time_weight * time_smooth_l1
    weighted_motion = contract.motion_weight * math.log(2.0)
    return {
        "contract": "training_only_v3_initial_component_scale_evidence_v1",
        "training_track_count": len(inventory.tracks),
        "training_keyframe_count": keyframe_count,
        "training_coordinate_scalar_count": int(value_residual.size),
        "gate_aligned_initial_normalized_coordinate_mae": float(
            value_residual.mean()
        ),
        "initial_value_smooth_l1_unweighted": value_smooth_l1,
        "initial_value_smooth_l1_weighted": weighted_value,
        "initial_uniform_count_cross_entropy_weighted": weighted_count,
        "initial_uniform_time_smooth_l1_weighted": weighted_time,
        "initial_motion_binary_cross_entropy_weighted": weighted_motion,
        "count_to_value_weighted_loss_ratio": (
            weighted_count / weighted_value if weighted_value > 0.0 else None
        ),
        "base_loss_weight": contract.base_loss_weight,
        "validation_labels_consumed": False,
        "heldout_labels_consumed": False,
    }


def _cut_only_pretraining_loss_contract(
    effective_loss_contract: StageEditLossContract,
) -> StageEditLossContract:
    return replace(
        effective_loss_contract,
        cut_positive_weight=1.0,
        source_listwise_weight=0.0,
        source_pairwise_weight=0.0,
        duration_weight=0.0,
        transition_weight=0.0,
        crop_weight=0.0,
    )


def _source_only_pretraining_loss_contract(
    effective_loss_contract: StageEditLossContract,
) -> StageEditLossContract:
    return replace(
        effective_loss_contract,
        source_objective="eligible_all_hybrid",
        cut_weight=0.0,
        cut_positive_weight=1.0,
        duration_weight=0.0,
        transition_weight=0.0,
        crop_weight=0.0,
    )


def _balanced_cut_windows(
    examples: Sequence[ReferenceTrainingExample],
    *,
    schema: StageEditTokenSchema,
    window_length: int,
) -> tuple[tuple[_BalancedCutWindow, ...], tuple[_BalancedCutWindow, ...]]:
    """Build causal positive/negative cut windows from training examples only."""

    if window_length < 2:
        raise ValueError("balanced cut pretraining window must contain >= 2 tokens")
    positive: list[_BalancedCutWindow] = []
    negative: list[_BalancedCutWindow] = []
    prepared_sequences = prepare_stage_edit_production_input_sequences(
        examples, schema=schema
    )
    for example, prepared in zip(examples, prepared_sequences, strict=True):
        if len(prepared.tokens) != len(example.targets.targets):
            raise ValueError("cut pretraining sequence/target length mismatch")
        for anchor_index, target in enumerate(example.targets.targets):
            if target.cut not in (0.0, 1.0):
                continue
            start = max(0, anchor_index - window_length + 1)
            identifier = (
                f"{prepared.sequence_id}:balanced-cut:{anchor_index}:"
                f"{int(target.cut)}"
            )
            token_values = tuple(prepared.tokens[start : anchor_index + 1])
            target_values = tuple(
                StageEditTarget() for _ in token_values[:-1]
            ) + (StageEditTarget(cut=float(target.cut)),)
            window = _BalancedCutWindow(
                sequence=StageEditSequence(identifier, token_values),
                targets=StageEditTargetSequence(identifier, target_values),
                label=int(target.cut),
                reference_id=example.reference_id,
                anchor_token_index=anchor_index,
            )
            (positive if window.label else negative).append(window)
    if not positive or not negative:
        raise ValueError(
            "balanced cut pretraining requires positive and negative training labels"
        )
    return tuple(positive), tuple(negative)


def _warmup_cosine_schedule(
    *,
    peak_learning_rate: float,
    total_steps: int,
    warmup_steps: int,
    minimum_learning_rate_ratio: float,
) -> Any:
    """Return one explicit warmup/cosine schedule for an MLX optimizer."""

    if total_steps < 1:
        raise ValueError("learning-rate schedule requires at least one step")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("warmup steps must be in [0, total_steps)")
    ratio = float(minimum_learning_rate_ratio)
    if not math.isfinite(ratio) or not 0.0 <= ratio < 1.0:
        raise ValueError("minimum learning-rate ratio must be in [0, 1)")
    end = peak_learning_rate * ratio
    if warmup_steps == 0:
        return optim.cosine_decay(
            peak_learning_rate, decay_steps=total_steps, end=end
        )
    warmup = optim.linear_schedule(
        peak_learning_rate * 0.1,
        peak_learning_rate,
        warmup_steps,
    )
    decay = optim.cosine_decay(
        peak_learning_rate,
        decay_steps=max(total_steps - warmup_steps, 1),
        end=end,
    )
    return optim.join_schedules((warmup, decay), (warmup_steps,))


def _offset_learning_rate_schedule(schedule: Any, offset_steps: int) -> Any:
    if offset_steps < 0:
        raise ValueError("learning-rate schedule offset cannot be negative")

    def shifted(step: Any) -> Any:
        return schedule(step + offset_steps)

    return shifted


def prepare_stage_edit_production_input_sequences(
    examples: Sequence[ReferenceTrainingExample],
    *,
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
) -> tuple[Any, ...]:
    """Create the exact train/eval inputs used by a production checkpoint.

    Finished-edit snapshots can contain strictly-past human decisions.  They
    remain useful dataset evidence but are not an inference-available model
    input.  Production preprocessing removes that complete slice and marks it
    unknown before any tensorization.
    """

    example_tuple = tuple(examples)
    _production_feature_snapshot_gate_for_examples(
        example_tuple,
        schema=schema,
    )

    return tuple(
        mask_stage_edit_recent_history_for_semimarkov(example.sequence, schema)
        for example in example_tuple
    )


def train_production_stage_edit(
    corpus: ReferenceTrainingCorpus,
    *,
    checkpoint_directory: Path,
    held_out_song_ids: Sequence[str],
    required_training_song_ids: Sequence[str] = (),
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
    model_config: StageEditTransformerConfig | None = None,
    epochs: int = 30,
    early_stopping_patience: int = 24,
    batch_size: int = 2,
    learning_rate: float = 1.0e-4,
    weight_decay: float = 1.0e-3,
    gradient_clip_norm: float = 1.0,
    maximum_cut_positive_weight: float = 8.0,
    cut_pretrain_steps: int = 500,
    cut_pretrain_learning_rate: float = 3.0e-4,
    cut_pretrain_window_length: int = 32,
    cut_pretrain_warmup_steps: int = 50,
    source_pretrain_steps: int = 500,
    source_pretrain_learning_rate: float = 3.0e-4,
    source_pretrain_window_length: int = 32,
    source_pretrain_warmup_steps: int = 50,
    dynamic_crop_v3_prior_pretrain_steps: int = 0,
    dynamic_crop_v3_prior_pretrain_learning_rate: float = 3.0e-4,
    dynamic_crop_v3_prior_pretrain_warmup_steps: int = 50,
    finetune_warmup_steps: int = 8,
    minimum_learning_rate_ratio: float = 0.10,
    minimum_finetune_optimizer_steps: int = 120,
    minimum_checkpoint_selection_finetune_steps: int = 60,
    seed: int = 20260824,
    device: Any = None,
    loss_contract: StageEditLossContract = StageEditLossContract(),
    dynamic_crop_corpus: (
        StageEditDynamicCropTrainingCorpusV2
        | StageEditDynamicCropTrainingCorpusV3
        | None
    ) = None,
    dynamic_crop_loss_contract: DynamicCropV2LossContract = (
        DynamicCropV2LossContract()
    ),
    dynamic_crop_v3_loss_contract: DynamicCropV3LossContract = (
        DynamicCropV3LossContract()
    ),
    quality_thresholds: ProductionQualityThresholds = ProductionQualityThresholds(),
    authorize_production_activation: bool = False,
) -> ProductionTrainingRunResult:
    """Train a real long-context model and evaluate its exact saved checkpoint."""

    _require_mlx()
    if epochs < 1 or early_stopping_patience < 1 or batch_size < 1:
        raise ValueError("epochs, patience, and batch size must be positive")
    for value, name in (
        (minimum_finetune_optimizer_steps, "minimum_finetune_optimizer_steps"),
        (
            minimum_checkpoint_selection_finetune_steps,
            "minimum_checkpoint_selection_finetune_steps",
        ),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        minimum_checkpoint_selection_finetune_steps
        > minimum_finetune_optimizer_steps
    ):
        raise ValueError(
            "checkpoint selection burn-in cannot exceed the finetune step floor"
        )
    if cut_pretrain_steps not in (0,) and cut_pretrain_steps < 500:
        raise ValueError("cut pretraining must be disabled or run for >= 500 steps")
    if source_pretrain_steps not in (0,) and source_pretrain_steps < 500:
        raise ValueError(
            "source pretraining must be disabled or run for >= 500 steps"
        )
    if (
        dynamic_crop_v3_prior_pretrain_steps not in (0,)
        and dynamic_crop_v3_prior_pretrain_steps < 500
    ):
        raise ValueError(
            "Dynamic Crop v3 prior pretraining must be disabled or run for "
            ">= 500 steps"
        )
    if (
        dynamic_crop_v3_prior_pretrain_steps
        and dynamic_crop_v3_prior_pretrain_steps
        < max(DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS)
    ):
        raise ValueError(
            "Dynamic Crop v3 prior pretraining must reach every predeclared "
            "candidate milestone"
        )
    if cut_pretrain_window_length < 2:
        raise ValueError("cut pretraining window length must be >= 2")
    if source_pretrain_window_length < 2:
        raise ValueError("source pretraining window length must be >= 2")
    if cut_pretrain_steps and not 0 <= cut_pretrain_warmup_steps < cut_pretrain_steps:
        raise ValueError("cut pretraining warmup must be shorter than pretraining")
    if (
        source_pretrain_steps
        and not 0 <= source_pretrain_warmup_steps < source_pretrain_steps
    ):
        raise ValueError("source pretraining warmup must be shorter than pretraining")
    if (
        dynamic_crop_v3_prior_pretrain_steps
        and not 0
        <= dynamic_crop_v3_prior_pretrain_warmup_steps
        < dynamic_crop_v3_prior_pretrain_steps
    ):
        raise ValueError(
            "Dynamic Crop v3 prior warmup must be shorter than pretraining"
        )
    if finetune_warmup_steps < 0:
        raise ValueError("finetune warmup cannot be negative")
    for value, name, allow_zero in (
        (learning_rate, "learning_rate", False),
        (cut_pretrain_learning_rate, "cut_pretrain_learning_rate", False),
        (source_pretrain_learning_rate, "source_pretrain_learning_rate", False),
        (
            dynamic_crop_v3_prior_pretrain_learning_rate,
            "dynamic_crop_v3_prior_pretrain_learning_rate",
            False,
        ),
        (weight_decay, "weight_decay", True),
        (gradient_clip_norm, "gradient_clip_norm", False),
    ):
        if (
            not math.isfinite(value)
            or value < 0.0
            or (not allow_zero and value == 0.0)
        ):
            raise ValueError(f"{name} is invalid")

    dynamic_crop_v2_enabled = isinstance(
        dynamic_crop_corpus, StageEditDynamicCropTrainingCorpusV2
    )
    dynamic_crop_v3_enabled = isinstance(
        dynamic_crop_corpus, StageEditDynamicCropTrainingCorpusV3
    )
    if dynamic_crop_corpus is not None and not (
        dynamic_crop_v2_enabled or dynamic_crop_v3_enabled
    ):
        raise TypeError(
            "dynamic crop corpus must be an explicit v2 or v3 training corpus"
        )
    if dynamic_crop_v3_prior_pretrain_steps and not dynamic_crop_v3_enabled:
        raise ValueError(
            "Dynamic Crop v3 prior pretraining requires a v3 corpus"
        )
    dynamic_crop_version = (
        3 if dynamic_crop_v3_enabled else 2 if dynamic_crop_v2_enabled else None
    )

    split = production_song_split(
        corpus,
        held_out_song_ids=held_out_song_ids,
        required_training_song_ids=required_training_song_ids,
    )
    finetune_batches_per_epoch = len(_batches(split.train, batch_size))
    finetune_steps_planned = epochs * finetune_batches_per_epoch
    if finetune_steps_planned < minimum_finetune_optimizer_steps:
        raise ValueError(
            "planned multi-head finetuning is below the explicit optimizer-step "
            f"floor: planned={finetune_steps_planned}, "
            f"minimum={minimum_finetune_optimizer_steps}"
        )
    training_cut_positive_count = sum(
        target.cut == 1.0
        for example in split.train
        for target in example.targets.targets
        if target.cut is not None
    )
    training_cut_negative_count = sum(
        target.cut == 0.0
        for example in split.train
        for target in example.targets.targets
        if target.cut is not None
    )
    cut_imbalance_contract = stage_edit_cut_imbalance_contract(
        positive_count=training_cut_positive_count,
        negative_count=training_cut_negative_count,
        maximum_positive_weight=maximum_cut_positive_weight,
    )
    effective_loss_contract = replace(
        loss_contract,
        cut_positive_weight=float(
            cut_imbalance_contract["applied_positive_weight"]
        ),
    )
    if source_pretrain_steps and (
        effective_loss_contract.source_listwise_weight <= 0.0
        or effective_loss_contract.source_pairwise_weight <= 0.0
    ):
        raise ValueError(
            "source pretraining requires positive listwise and pairwise weights"
        )
    effective_dynamic_crop_v2_loss_contract = replace(
        dynamic_crop_loss_contract,
        base=replace(
            dynamic_crop_loss_contract.base,
            cut_positive_weight=float(
                cut_imbalance_contract["applied_positive_weight"]
            ),
        ),
    )
    effective_dynamic_crop_v3_loss_contract = replace(
        dynamic_crop_v3_loss_contract,
        base=replace(
            dynamic_crop_v3_loss_contract.base,
            cut_positive_weight=float(
                cut_imbalance_contract["applied_positive_weight"]
            ),
        ),
    )
    feature_snapshot_gate = _require_production_feature_snapshots(
        corpus,
        schema=schema,
    )
    supervision_gate_corpus = ReferenceTrainingCorpus(
        examples=split.train + split.validation,
        data_id=f"{corpus.data_id}:training-songs-only",
        source_paths=corpus.source_paths,
        statistics=corpus.statistics,
    )
    supervision_gate: ProductionSupervisionGate = (
        require_stage_edit_production_supervision(supervision_gate_corpus)
    )
    config = model_config or production_stage_edit_config(schema)
    if config.token_schema_fingerprint != schema.fingerprint:
        raise ValueError("production model configuration and token schema differ")
    selected_dynamic_base_contract = (
        effective_dynamic_crop_v3_loss_contract.base
        if dynamic_crop_v3_enabled
        else effective_dynamic_crop_v2_loss_contract.base
    )
    if (
        config.transition_authority
        != effective_loss_contract.transition_authority
        or config.transition_authority
        != selected_dynamic_base_contract.transition_authority
    ):
        raise ValueError(
            "model, base loss, and Dynamic Crop base transition authority differ"
        )
    transition_authority = StageEditTransitionAuthority(
        mode=config.transition_authority
    )
    maximum_length = max(len(example.sequence.tokens) for example in corpus.examples)
    maximum_sources = max(
        len(token.source_candidates)
        for example in corpus.examples
        for token in example.sequence.tokens
    )
    if maximum_length > config.max_sequence_length:
        raise ValueError(
            f"adapted sequence length {maximum_length} exceeds model maximum "
            f"{config.max_sequence_length}"
        )
    if maximum_sources > config.max_sources:
        raise ValueError(
            f"adapted source count {maximum_sources} exceeds model maximum "
            f"{config.max_sources}"
        )

    expected_sequence_ids = {
        example.sequence.sequence_id for example in corpus.examples
    }
    if dynamic_crop_corpus is not None:
        observed_sequence_ids = set(dynamic_crop_corpus.sequences)
        if observed_sequence_ids != expected_sequence_ids:
            missing = sorted(expected_sequence_ids - observed_sequence_ids)
            unknown = sorted(observed_sequence_ids - expected_sequence_ids)
            raise ValueError(
                "dynamic crop corpus sequence identity mismatch; "
                f"missing={missing}, unknown={unknown}"
            )

    def dynamic_sequences_for(
        examples: Sequence[ReferenceTrainingExample],
    ) -> tuple[Any, ...]:
        if dynamic_crop_corpus is None:
            return ()
        return tuple(
            dynamic_crop_corpus.sequence_for(example.sequence.sequence_id)
            for example in examples
        )

    checkpoint_root = Path(checkpoint_directory).expanduser().resolve()
    _require_unambiguous_checkpoint_namespace(
        checkpoint_root,
        dynamic_crop_version=dynamic_crop_version,
    )
    report_path = checkpoint_root / PRODUCTION_TRAINING_REPORT_FILENAME
    active_device = device if device is not None else mx.default_device()
    history: list[dict[str, Any]] = []
    cut_pretraining_history: list[dict[str, Any]] = []
    source_pretraining_history: list[dict[str, Any]] = []
    dynamic_crop_v3_prior_pretraining_history: list[dict[str, Any]] = []
    optimizer_steps = 0
    cut_pretraining_optimizer_steps = 0
    source_pretraining_optimizer_steps = 0
    dynamic_crop_v3_prior_pretraining_optimizer_steps = 0
    finetune_optimizer_steps = 0
    best_epoch = 0
    best_step = 0
    best_finetune_optimizer_step = 0
    best_checkpoint_origin: str | None = None
    best_validation_loss = math.inf
    best_validation_selection_score = -math.inf
    best_validation_selection_evidence: Mapping[str, Any] | None = None
    finetune_steps_without_improvement = 0
    early_stopping_triggered = False
    training_stop_reason = "planned_finetune_steps_completed"
    finetune_optimizer_initial_step = 0
    finetune_optimizer_state_final_step = 0
    finetune_retention_optimizer_state_final_step = 0
    finetune_joint_optimizer_initial_step = 0
    finetune_optimizer_reset_at_dynamic_crop_v3_prior_release = False
    cut_pretraining_optimizer_initial_step = 0
    cut_pretraining_optimizer_state_final_step = 0
    source_pretraining_optimizer_initial_step = 0
    source_pretraining_optimizer_state_final_step = 0
    dynamic_crop_v3_prior_optimizer_initial_step = 0
    dynamic_crop_v3_prior_optimizer_state_final_step = 0
    dynamic_crop_v3_prior_candidate_selection: dict[str, Any] | None = None
    dynamic_crop_v3_prior_candidate_snapshots: dict[int, Any] = {}
    dynamic_crop_v3_prior_before_digests: dict[str, str] | None = None
    dynamic_crop_v3_prior_values_stop = DYNAMIC_CROP_V3_MAX_KEYFRAMES * 4
    dynamic_crop_v3_prior_component_scale_evidence: dict[str, Any] | None = None
    dynamic_crop_v3_prior_parameter_invariance: dict[str, Any] | None = None
    dynamic_crop_v3_prior_sampling_evidence: dict[str, Any] | None = None
    dynamic_crop_v3_prior_retention_finetune_steps = 0
    dynamic_crop_v3_prior_retention_restore_steps = 0
    dynamic_crop_v3_prior_retention_evidence: dict[str, Any] | None = None
    retained_dynamic_crop_v3_context_weight = None
    retained_dynamic_crop_v3_context_bias = None
    retained_dynamic_crop_v3_source_weight = None
    source_pretraining_sampling_evidence: dict[str, Any] = {
        "contract": "deterministic_song_reference_group_round_robin_v1",
        "opening_source_excluded_count": 0,
        "unconfirmed_source_label_excluded_count": 0,
        "confirmed_source_event_count": 0,
        "reference_song_ids": {},
        "per_reference_confirmed_group_counts": {},
        "per_song_sampled_counts": {},
        "per_reference_sampled_counts": {},
    }
    saved_metadata = None
    clipping_supported = hasattr(optim, "clip_grad_norm")
    activation_checkpointing_evidence: Mapping[str, Any] | None = None

    with mx.stream(active_device):
        mx.random.seed(seed)
        model = (
            StageEditTransformerDynamicCropV3(config)
            if dynamic_crop_v3_enabled
            else StageEditTransformerDynamicCropV2(config)
            if dynamic_crop_v2_enabled
            else StageEditTransformer(config)
        )
        if dynamic_crop_v3_enabled:
            activation_checkpointing_evidence = (
                _enable_dynamic_crop_v3_activation_checkpointing(model)
            )

        cut_only_loss_contract = _cut_only_pretraining_loss_contract(
            effective_loss_contract
        )
        if cut_pretrain_steps:
            positive_windows, negative_windows = _balanced_cut_windows(
                split.train,
                schema=schema,
                window_length=cut_pretrain_window_length,
            )
            pretrain_schedule = _warmup_cosine_schedule(
                peak_learning_rate=cut_pretrain_learning_rate,
                total_steps=cut_pretrain_steps,
                warmup_steps=cut_pretrain_warmup_steps,
                minimum_learning_rate_ratio=minimum_learning_rate_ratio,
            )
            pretrain_optimizer = optim.AdamW(
                learning_rate=pretrain_schedule,
                weight_decay=weight_decay,
            )
            cut_pretraining_optimizer_initial_step = int(
                pretrain_optimizer.state["step"].item()
            )
            if cut_pretraining_optimizer_initial_step != 0:
                raise RuntimeError(
                    "fresh cut-only optimizer did not start at step zero"
                )

            def cut_pretrain_loss(
                trained_model: Any,
                batch: Any,
                supervision: Any,
            ) -> Any:
                return stage_edit_loss(
                    trained_model,
                    batch,
                    supervision,
                    cut_only_loss_contract,
                )[0]

            cut_pretrain_loss_and_grad = nn.value_and_grad(
                model, cut_pretrain_loss
            )
            chooser = random.Random(seed ^ 0x5A17E)
            for pretrain_step in range(1, cut_pretrain_steps + 1):
                windows = (
                    positive_windows[chooser.randrange(len(positive_windows))],
                    negative_windows[chooser.randrange(len(negative_windows))],
                )
                pretrain_batch = tensorize_stage_edit_batch(
                    tuple(window.sequence for window in windows),
                    schema,
                    config,
                    device=active_device,
                )
                pretrain_supervision = tensorize_stage_edit_supervision(
                    tuple(window.targets for window in windows),
                    pretrain_batch,
                    schema,
                    device=active_device,
                )
                model.train()
                pretrain_loss, gradients = cut_pretrain_loss_and_grad(
                    model, pretrain_batch, pretrain_supervision
                )
                if clipping_supported:
                    gradients, gradient_norm = optim.clip_grad_norm(
                        gradients, gradient_clip_norm
                    )
                else:
                    gradient_norm = mx.array(float("nan"))
                pretrain_optimizer.update(model, gradients)
                optimizer_steps += 1
                cut_pretraining_optimizer_steps += 1
                mx.eval(
                    pretrain_loss,
                    gradient_norm,
                    model.parameters(),
                    pretrain_optimizer.state,
                )
                if (
                    pretrain_step == 1
                    or pretrain_step == cut_pretrain_steps
                    or pretrain_step % 25 == 0
                ):
                    cut_pretraining_history.append(
                        {
                            "pretrain_step": pretrain_step,
                            "optimizer_step": optimizer_steps,
                            "balanced_anchor_positive_count": 1,
                            "balanced_anchor_negative_count": 1,
                            "cut_only_loss": float(pretrain_loss.item()),
                            "gradient_norm_before_clip": float(
                                gradient_norm.item()
                            ),
                        }
                    )
            cut_pretraining_optimizer_state_final_step = int(
                pretrain_optimizer.state["step"].item()
            )
            if (
                cut_pretraining_optimizer_state_final_step
                != cut_pretraining_optimizer_steps
            ):
                raise RuntimeError(
                    "cut-only optimizer state step differs from history"
                )

        source_only_loss_contract = _source_only_pretraining_loss_contract(
            effective_loss_contract
        )
        if source_pretrain_steps:
            source_inventory = _source_pretraining_inventory(
                split.train,
                schema=schema,
                window_length=source_pretrain_window_length,
            )
            source_sampler = _HierarchicalSourceWindowSampler(
                source_inventory.windows,
                seed=seed ^ 0x50A7CE,
            )
            source_pretrain_schedule = _warmup_cosine_schedule(
                peak_learning_rate=source_pretrain_learning_rate,
                total_steps=source_pretrain_steps,
                warmup_steps=source_pretrain_warmup_steps,
                minimum_learning_rate_ratio=minimum_learning_rate_ratio,
            )
            source_pretrain_optimizer = optim.AdamW(
                learning_rate=source_pretrain_schedule,
                weight_decay=weight_decay,
            )
            source_pretraining_optimizer_initial_step = int(
                source_pretrain_optimizer.state["step"].item()
            )
            if source_pretraining_optimizer_initial_step != 0:
                raise RuntimeError(
                    "fresh source-only optimizer did not start at step zero"
                )

            def source_pretrain_loss(
                trained_model: Any,
                batch: Any,
                supervision: Any,
            ) -> Any:
                return stage_edit_loss(
                    trained_model,
                    batch,
                    supervision,
                    source_only_loss_contract,
                )[0]

            source_pretrain_loss_and_grad = nn.value_and_grad(
                model, source_pretrain_loss
            )
            for source_pretrain_step in range(1, source_pretrain_steps + 1):
                window = source_sampler.sample()
                source_pretrain_batch = tensorize_stage_edit_batch(
                    (window.sequence,),
                    schema,
                    config,
                    device=active_device,
                )
                source_pretrain_supervision = tensorize_stage_edit_supervision(
                    (window.targets,),
                    source_pretrain_batch,
                    schema,
                    device=active_device,
                )
                model.train()
                source_loss, gradients = source_pretrain_loss_and_grad(
                    model,
                    source_pretrain_batch,
                    source_pretrain_supervision,
                )
                if clipping_supported:
                    gradients, gradient_norm = optim.clip_grad_norm(
                        gradients, gradient_clip_norm
                    )
                else:
                    gradient_norm = mx.array(float("nan"))
                source_pretrain_optimizer.update(model, gradients)
                optimizer_steps += 1
                source_pretraining_optimizer_steps += 1
                mx.eval(
                    source_loss,
                    gradient_norm,
                    model.parameters(),
                    source_pretrain_optimizer.state,
                )
                source_pretraining_history.append(
                    {
                        "source_pretrain_step": source_pretrain_step,
                        "optimizer_step": optimizer_steps,
                        "actual_learning_rate": float(
                            source_pretrain_optimizer.learning_rate.item()
                        ),
                        "song_id": window.song_id,
                        "reference_id": window.reference_id,
                        "group_id": window.group_id,
                        "anchor_token_index": window.anchor_token_index,
                        "source_only_loss": float(source_loss.item()),
                        "gradient_norm_before_clip": float(
                            gradient_norm.item()
                        ),
                        "checkpoint_selection_eligible": False,
                    }
                )
            source_pretraining_optimizer_state_final_step = int(
                source_pretrain_optimizer.state["step"].item()
            )
            if (
                source_pretraining_optimizer_state_final_step
                != source_pretraining_optimizer_steps
            ):
                raise RuntimeError(
                    "source-only optimizer state step differs from history"
                )
            source_pretraining_sampling_evidence = {
                "contract": (
                    "deterministic_song_reference_group_round_robin_v1"
                ),
                "opening_source_excluded_count": (
                    source_inventory.opening_source_excluded_count
                ),
                "unconfirmed_source_label_excluded_count": (
                    source_inventory.unconfirmed_source_label_excluded_count
                ),
                "confirmed_source_event_count": len(source_inventory.windows),
                "reference_song_ids": dict(
                    source_inventory.reference_song_ids
                ),
                "per_reference_confirmed_group_counts": dict(
                    source_inventory.per_reference_confirmed_group_counts
                ),
                "per_song_sampled_counts": (
                    source_sampler.per_song_sampled_counts
                ),
                "per_reference_sampled_counts": (
                    source_sampler.per_reference_sampled_counts
                ),
            }

        if dynamic_crop_v3_prior_pretrain_steps:
            assert dynamic_crop_v3_enabled
            assert isinstance(model, StageEditTransformerDynamicCropV3)
            assert isinstance(
                dynamic_crop_corpus, StageEditDynamicCropTrainingCorpusV3
            )
            prior_inventory = _dynamic_crop_v3_prior_inventory(
                split.train, dynamic_crop_corpus
            )
            prior_sampler = _HierarchicalDynamicCropV3PriorSampler(
                prior_inventory.tracks,
                seed=seed ^ 0xD1A5C0,
            )
            values_stop = dynamic_crop_v3_prior_values_stop
            before_digests = _dynamic_crop_v3_prior_parameter_digests(model)
            dynamic_crop_v3_prior_before_digests = before_digests
            prior_module = _DynamicCropV3PriorBiasModule(
                model.dynamic_crop_context_head.bias[:values_stop]
            )
            initial_values_mx = renderer_safe_values_from_raw_v3(
                prior_module.value_bias
            )
            mx.eval(initial_values_mx)
            dynamic_crop_v3_prior_component_scale_evidence = (
                _dynamic_crop_v3_prior_component_scale_evidence(
                    prior_inventory,
                    np.asarray(initial_values_mx, dtype=np.float64),
                    effective_dynamic_crop_v3_loss_contract,
                )
            )
            prior_schedule = _warmup_cosine_schedule(
                peak_learning_rate=(
                    dynamic_crop_v3_prior_pretrain_learning_rate
                ),
                total_steps=dynamic_crop_v3_prior_pretrain_steps,
                warmup_steps=dynamic_crop_v3_prior_pretrain_warmup_steps,
                minimum_learning_rate_ratio=minimum_learning_rate_ratio,
            )
            prior_optimizer = optim.AdamW(
                learning_rate=prior_schedule,
                weight_decay=0.0,
            )
            dynamic_crop_v3_prior_optimizer_initial_step = int(
                prior_optimizer.state["step"].item()
            )
            if dynamic_crop_v3_prior_optimizer_initial_step != 0:
                raise RuntimeError(
                    "fresh Dynamic Crop v3 prior optimizer did not start at zero"
                )

            def prior_loss(
                trained_prior: _DynamicCropV3PriorBiasModule,
                target_values: Any,
            ) -> Any:
                return _dynamic_crop_v3_prior_normalized_mae(
                    trained_prior, target_values
                )

            prior_loss_and_grad = nn.value_and_grad(prior_module, prior_loss)
            for prior_step in range(1, dynamic_crop_v3_prior_pretrain_steps + 1):
                track = prior_sampler.sample()
                target_values = mx.array(
                    np.asarray(track.values, dtype=np.float32)
                )
                loss, gradients = prior_loss_and_grad(
                    prior_module, target_values
                )
                if clipping_supported:
                    gradients, gradient_norm = optim.clip_grad_norm(
                        gradients, gradient_clip_norm
                    )
                else:
                    gradient_norm = mx.array(float("nan"))
                prior_optimizer.update(prior_module, gradients)
                optimizer_steps += 1
                dynamic_crop_v3_prior_pretraining_optimizer_steps += 1
                mx.eval(
                    loss,
                    gradient_norm,
                    prior_module.parameters(),
                    prior_optimizer.state,
                )
                candidate_digest = None
                if prior_step in DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS:
                    snapshot = mx.array(
                        np.array(prior_module.value_bias, copy=True)
                    )
                    mx.eval(snapshot)
                    dynamic_crop_v3_prior_candidate_snapshots[prior_step] = (
                        snapshot
                    )
                    candidate_digest = _parameter_tree_sha256(
                        {"bias": snapshot.reshape((values_stop,))}
                    )
                dynamic_crop_v3_prior_pretraining_history.append(
                    {
                        "prior_pretrain_step": prior_step,
                        "optimizer_step": optimizer_steps,
                        "actual_learning_rate": float(
                            prior_optimizer.learning_rate.item()
                        ),
                        "song_id": track.song_id,
                        "reference_id": track.reference_id,
                        "sequence_id": track.sequence_id,
                        "token_index": track.token_index,
                        "keyframe_count": len(track.values),
                        "gate_aligned_normalized_coordinate_mae": float(
                            loss.item()
                        ),
                        "gradient_norm_before_clip": float(
                            gradient_norm.item()
                        ),
                        "checkpoint_selection_eligible": False,
                        "predeclared_candidate_snapshot": (
                            prior_step in DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS
                        ),
                        "candidate_value_bias_sha256": candidate_digest,
                        "validation_labels_consumed": False,
                        "heldout_labels_consumed": False,
                    }
                )
            dynamic_crop_v3_prior_optimizer_state_final_step = int(
                prior_optimizer.state["step"].item()
            )
            if (
                dynamic_crop_v3_prior_optimizer_state_final_step
                != dynamic_crop_v3_prior_pretraining_optimizer_steps
            ):
                raise RuntimeError(
                    "Dynamic Crop v3 prior optimizer state differs from history"
                )
            if set(dynamic_crop_v3_prior_candidate_snapshots) != set(
                DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS
            ):
                raise RuntimeError(
                    "Dynamic Crop v3 prior candidate snapshots are incomplete"
                )
            dynamic_crop_v3_prior_sampling_evidence = {
                "contract": (
                    "deterministic_song_reference_trajectory_round_robin_v1"
                ),
                "partition": "training_references_only",
                "trajectory_count": len(prior_inventory.tracks),
                "per_reference_track_counts": dict(
                    prior_inventory.per_reference_track_counts
                ),
                "references_without_tracks": list(
                    prior_inventory.references_without_tracks
                ),
                "reference_song_ids": dict(
                    prior_inventory.reference_song_ids
                ),
                "per_song_sampled_counts": (
                    prior_sampler.per_song_sampled_counts
                ),
                "per_reference_sampled_counts": (
                    prior_sampler.per_reference_sampled_counts
                ),
                "validation_labels_consumed": False,
                "heldout_labels_consumed": False,
            }

        finetune_warmup = min(
            finetune_warmup_steps, max(finetune_steps_planned - 1, 0)
        )
        finetune_schedule = _warmup_cosine_schedule(
            peak_learning_rate=learning_rate,
            total_steps=finetune_steps_planned,
            warmup_steps=finetune_warmup,
            minimum_learning_rate_ratio=minimum_learning_rate_ratio,
        )
        optimizer = optim.AdamW(
            learning_rate=finetune_schedule,
            weight_decay=weight_decay,
        )
        finetune_optimizer_initial_step = int(optimizer.state["step"].item())
        if finetune_optimizer_initial_step != 0:
            raise RuntimeError(
                "fresh multi-head optimizer did not start at phase-local step zero"
            )

        if dynamic_crop_v3_enabled:
            def loss_function(
                trained_model: Any,
                batch: Any,
                supervision: Any,
                dynamic_supervision: Any,
            ) -> Any:
                return stage_edit_dynamic_crop_v3_loss(
                    trained_model,
                    batch,
                    supervision,
                    dynamic_supervision,
                    effective_dynamic_crop_v3_loss_contract,
                )[0]
        elif dynamic_crop_v2_enabled:
            def loss_function(
                trained_model: Any,
                batch: Any,
                supervision: Any,
                dynamic_supervision: Any,
            ) -> Any:
                return stage_edit_dynamic_crop_v2_loss(
                    trained_model,
                    batch,
                    supervision,
                    dynamic_supervision,
                    effective_dynamic_crop_v2_loss_contract,
                )[0]
        else:
            def loss_function(
                trained_model: Any,
                batch: Any,
                supervision: Any,
                dynamic_supervision: Any = None,
            ) -> Any:
                del dynamic_supervision
                return stage_edit_loss(
                    trained_model,
                    batch,
                    supervision,
                    effective_loss_contract,
                )[0]

        loss_and_grad = nn.value_and_grad(model, loss_function)
        validation_batch = tensorize_stage_edit_batch(
            prepare_stage_edit_production_input_sequences(
                split.validation, schema=schema
            ),
            schema,
            config,
            device=active_device,
        )
        validation_supervision = tensorize_stage_edit_supervision(
            tuple(example.targets for example in split.validation),
            validation_batch,
            schema,
            device=active_device,
        )
        validation_dynamic_supervision = (
            tensorize_stage_edit_dynamic_crop_supervision_v3(
                dynamic_sequences_for(split.validation),
                validation_batch,
                device=active_device,
            )
            if dynamic_crop_v3_enabled
            else tensorize_stage_edit_dynamic_crop_supervision_v2(
                dynamic_sequences_for(split.validation),
                validation_batch,
                device=active_device,
            )
            if dynamic_crop_v2_enabled
            else None
        )

        if dynamic_crop_v3_prior_pretrain_steps:
            assert dynamic_crop_v3_enabled
            assert isinstance(model, StageEditTransformerDynamicCropV3)
            assert dynamic_crop_v3_prior_before_digests is not None
            warmup_validation_base = evaluate_stage_edit_model(
                model,
                validation_batch,
                validation_supervision,
                loss_contract=effective_loss_contract,
            )
            retained_non_value_bias = model.dynamic_crop_context_head.bias[
                dynamic_crop_v3_prior_values_stop:
            ]
            candidate_evaluations: list[dict[str, Any]] = []
            selectable_candidates: list[
                tuple[float, int, Any, dict[str, Any], dict[str, Any]]
            ] = []
            for candidate_step in DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS:
                candidate_bias = dynamic_crop_v3_prior_candidate_snapshots[
                    candidate_step
                ]
                model.dynamic_crop_context_head.bias = mx.concatenate(
                    (
                        candidate_bias.reshape(
                            (dynamic_crop_v3_prior_values_stop,)
                        ),
                        retained_non_value_bias,
                    )
                )
                mx.eval(model.dynamic_crop_context_head.bias)
                warmup_validation_dynamic = (
                    evaluate_stage_edit_dynamic_crop_v3(
                        model,
                        validation_batch,
                        validation_supervision,
                        validation_dynamic_supervision,
                        contract=effective_dynamic_crop_v3_loss_contract,
                    )
                )
                warmup_validation_metrics = {
                    **warmup_validation_base,
                    "total_loss": warmup_validation_dynamic["total_loss"],
                    "dynamic_crop_v3": warmup_validation_dynamic,
                }
                warmup_base_selection = stage_edit_validation_selection_score(
                    warmup_validation_metrics
                )
                warmup_selection = (
                    _stage_edit_validation_selection_with_dynamic_crop_v3(
                        warmup_base_selection,
                        warmup_validation_dynamic,
                    )
                )
                warmup_selection_score = _finite_metric(
                    warmup_selection.get("score")
                )
                candidate_digest = _parameter_tree_sha256(
                    {
                        "bias": candidate_bias.reshape(
                            (dynamic_crop_v3_prior_values_stop,)
                        )
                    }
                )
                candidate_evaluations.append(
                    {
                        "candidate_step": candidate_step,
                        "value_bias_sha256": candidate_digest,
                        "training_partition_only_updates": True,
                        "validation_gradients_computed": False,
                        "heldout_forward_performed": False,
                        "selection": dict(warmup_selection),
                        "validation_metrics": warmup_validation_metrics,
                    }
                )
                if (
                    warmup_selection.get("selectable") is True
                    and warmup_selection_score is not None
                ):
                    selectable_candidates.append(
                        (
                            warmup_selection_score,
                            candidate_step,
                            candidate_bias,
                            dict(warmup_selection),
                            warmup_validation_metrics,
                        )
                    )
            selected_candidate = (
                max(
                    selectable_candidates,
                    key=lambda item: (item[0], -item[1]),
                )
                if selectable_candidates
                else None
            )
            retention_authorized = selected_candidate is not None
            if selected_candidate is not None:
                (
                    _selected_score,
                    selected_candidate_step,
                    selected_candidate_bias,
                    warmup_selection,
                    warmup_validation_metrics,
                ) = selected_candidate
            else:
                selected_candidate_step = max(
                    DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS
                )
                selected_candidate_bias = (
                    dynamic_crop_v3_prior_candidate_snapshots[
                        selected_candidate_step
                    ]
                )
                failed_record = next(
                    record
                    for record in candidate_evaluations
                    if record["candidate_step"] == selected_candidate_step
                )
                warmup_selection = dict(failed_record["selection"])
                warmup_validation_metrics = dict(
                    failed_record["validation_metrics"]
                )
            model.dynamic_crop_context_head.bias = mx.concatenate(
                (
                    selected_candidate_bias.reshape(
                        (dynamic_crop_v3_prior_values_stop,)
                    ),
                    retained_non_value_bias,
                )
            )
            mx.eval(model.dynamic_crop_context_head.bias)
            after_digests = _dynamic_crop_v3_prior_parameter_digests(model)
            invariant_keys = (
                "base_parameter_tree_sha256",
                "context_weight_sha256",
                "source_head_parameter_tree_sha256",
                "context_non_value_bias_sha256",
            )
            invariant = all(
                dynamic_crop_v3_prior_before_digests[key]
                == after_digests[key]
                for key in invariant_keys
            )
            value_bias_changed = (
                dynamic_crop_v3_prior_before_digests[
                    "context_value_bias_sha256"
                ]
                != after_digests["context_value_bias_sha256"]
            )
            if not invariant or not value_bias_changed:
                raise RuntimeError(
                    "Dynamic Crop v3 prior warmup parameter scope drifted"
                )
            selected_candidate_digest = _parameter_tree_sha256(
                {
                    "bias": selected_candidate_bias.reshape(
                        (dynamic_crop_v3_prior_values_stop,)
                    )
                }
            )
            if (
                selected_candidate_digest
                != after_digests["context_value_bias_sha256"]
            ):
                raise RuntimeError(
                    "selected Dynamic Crop v3 candidate digest drifted"
                )
            dynamic_crop_v3_prior_parameter_invariance = {
                "contract": (
                    "only_dynamic_crop_context_bias_value_rows_mutable_v1"
                ),
                "before": dynamic_crop_v3_prior_before_digests,
                "after": after_digests,
                "invariant_parameter_groups": list(invariant_keys),
                "all_declared_invariant_groups_byte_identical": invariant,
                "context_value_bias_changed": value_bias_changed,
                "context_value_bias_row_count": (
                    dynamic_crop_v3_prior_values_stop
                ),
                "base_parameter_gradients_computed": False,
                "source_head_gradients_computed": False,
                "context_weight_gradients_computed": False,
                "count_time_motion_bias_gradients_computed": False,
            }
            if retention_authorized:
                batches_per_epoch = finetune_batches_per_epoch
                dynamic_crop_v3_prior_retention_finetune_steps = (
                    math.ceil(
                        minimum_checkpoint_selection_finetune_steps
                        / batches_per_epoch
                    )
                    * batches_per_epoch
                )
            else:
                dynamic_crop_v3_prior_retention_finetune_steps = 0
            retained_dynamic_crop_v3_context_weight = mx.array(
                np.array(
                    model.dynamic_crop_context_head.weight,
                    copy=True,
                )
            )
            retained_dynamic_crop_v3_context_bias = mx.array(
                np.array(
                    model.dynamic_crop_context_head.bias,
                    copy=True,
                )
            )
            retained_dynamic_crop_v3_source_weight = mx.array(
                np.array(
                    model.dynamic_crop_source_head.weight,
                    copy=True,
                )
            )
            dynamic_crop_v3_prior_candidate_selection = {
                "contract": DYNAMIC_CROP_V3_PRIOR_CANDIDATE_SELECTION_CONTRACT,
                "predeclared_candidate_steps": list(
                    DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS
                ),
                "selection_rule": (
                    "highest_finite_selectable_existing_validation_score_"
                    "then_lower_candidate_step_on_exact_tie"
                ),
                "training_partition_only_gradients": True,
                "validation_reference_ids": [
                    example.reference_id for example in split.validation
                ],
                "validation_gradients_computed": False,
                "heldout_forward_performed": False,
                "candidate_evaluations": candidate_evaluations,
                "selected_candidate_step": selected_candidate_step,
                "selected_candidate_value_bias_sha256": (
                    selected_candidate_digest
                ),
                "selection": dict(warmup_selection),
                "validation_metrics": warmup_validation_metrics,
                "retention_authorized": retention_authorized,
                "checkpoint_written_before_joint_finetuning": False,
                "checkpoint_write_requires_existing_finetune_burn_in": True,
                "retention_finetune_steps": (
                    dynamic_crop_v3_prior_retention_finetune_steps
                ),
            }

        for epoch in range(1, epochs + 1):
            epoch_losses: list[float] = []
            epoch_gradient_norms: list[float] = []
            epoch_finetune_optimizer_steps: list[int] = []
            epoch_phase_local_optimizer_steps: list[int] = []
            epoch_optimizer_reset_before_steps: list[int] = []
            epoch_actual_learning_rates: list[float] = []
            epoch_dynamic_crop_v3_prior_retention_steps: list[int] = []
            model.train()
            ordered = _stable_epoch_order(split.train, seed=seed, epoch=epoch)
            for examples in _batches(ordered, batch_size):
                if (
                    dynamic_crop_v3_prior_retention_finetune_steps
                    and finetune_optimizer_steps
                    == dynamic_crop_v3_prior_retention_finetune_steps
                    and not finetune_optimizer_reset_at_dynamic_crop_v3_prior_release
                ):
                    finetune_retention_optimizer_state_final_step = int(
                        optimizer.state["step"].item()
                    )
                    if (
                        finetune_retention_optimizer_state_final_step
                        != dynamic_crop_v3_prior_retention_finetune_steps
                    ):
                        raise RuntimeError(
                            "prior-retention optimizer step differs at release"
                        )
                    optimizer = optim.AdamW(
                        learning_rate=_offset_learning_rate_schedule(
                            finetune_schedule,
                            dynamic_crop_v3_prior_retention_finetune_steps,
                        ),
                        weight_decay=weight_decay,
                    )
                    finetune_joint_optimizer_initial_step = int(
                        optimizer.state["step"].item()
                    )
                    if finetune_joint_optimizer_initial_step != 0:
                        raise RuntimeError(
                            "released joint optimizer did not start fresh"
                        )
                    finetune_optimizer_reset_at_dynamic_crop_v3_prior_release = (
                        True
                    )
                    epoch_optimizer_reset_before_steps.append(
                        finetune_optimizer_steps + 1
                    )
                batch = tensorize_stage_edit_batch(
                    prepare_stage_edit_production_input_sequences(
                        examples, schema=schema
                    ),
                    schema,
                    config,
                    device=active_device,
                )
                supervision = tensorize_stage_edit_supervision(
                    tuple(example.targets for example in examples),
                    batch,
                    schema,
                    device=active_device,
                )
                dynamic_supervision = (
                    tensorize_stage_edit_dynamic_crop_supervision_v3(
                        dynamic_sequences_for(examples),
                        batch,
                        device=active_device,
                    )
                    if dynamic_crop_v3_enabled
                    else tensorize_stage_edit_dynamic_crop_supervision_v2(
                        dynamic_sequences_for(examples),
                        batch,
                        device=active_device,
                    )
                    if dynamic_crop_v2_enabled
                    else None
                )
                loss, gradients = loss_and_grad(
                    model, batch, supervision, dynamic_supervision
                )
                next_finetune_step = finetune_optimizer_steps + 1
                retain_dynamic_crop_v3_prior = bool(
                    dynamic_crop_v3_prior_candidate_selection
                    and dynamic_crop_v3_prior_candidate_selection.get(
                        "retention_authorized"
                    )
                    is True
                    and next_finetune_step
                    <= dynamic_crop_v3_prior_retention_finetune_steps
                )
                if retain_dynamic_crop_v3_prior:
                    if not isinstance(gradients, Mapping):
                        raise RuntimeError(
                            "Dynamic Crop v3 gradients are not name-addressable"
                        )
                    gradients = dict(gradients)
                    for head_name in (
                        "dynamic_crop_context_head",
                        "dynamic_crop_source_head",
                    ):
                        if head_name not in gradients:
                            raise RuntimeError(
                                "Dynamic Crop v3 prior retention head is missing"
                            )
                        gradients[head_name] = _zero_gradient_tree(
                            gradients[head_name]
                        )
                if clipping_supported:
                    gradients, gradient_norm = optim.clip_grad_norm(
                        gradients, gradient_clip_norm
                    )
                else:  # Pinned production MLX supports clipping.
                    gradient_norm = mx.array(float("nan"))
                optimizer.update(model, gradients)
                if retain_dynamic_crop_v3_prior:
                    assert retained_dynamic_crop_v3_context_weight is not None
                    assert retained_dynamic_crop_v3_context_bias is not None
                    assert retained_dynamic_crop_v3_source_weight is not None
                    model.dynamic_crop_context_head.weight = (
                        retained_dynamic_crop_v3_context_weight
                    )
                    model.dynamic_crop_context_head.bias = (
                        retained_dynamic_crop_v3_context_bias
                    )
                    model.dynamic_crop_source_head.weight = (
                        retained_dynamic_crop_v3_source_weight
                    )
                    dynamic_crop_v3_prior_retention_restore_steps += 1
                    epoch_dynamic_crop_v3_prior_retention_steps.append(
                        next_finetune_step
                    )
                optimizer_steps += 1
                finetune_optimizer_steps += 1
                mx.eval(loss, gradient_norm, model.parameters(), optimizer.state)
                epoch_phase_local_optimizer_steps.append(
                    int(optimizer.state["step"].item())
                )
                epoch_losses.append(float(loss.item()))
                epoch_gradient_norms.append(float(gradient_norm.item()))
                epoch_finetune_optimizer_steps.append(finetune_optimizer_steps)
                epoch_actual_learning_rates.append(
                    float(optimizer.learning_rate.item())
                )

            if (
                dynamic_crop_v3_prior_candidate_selection
                and dynamic_crop_v3_prior_candidate_selection.get(
                    "retention_authorized"
                )
                is True
                and finetune_optimizer_steps
                == dynamic_crop_v3_prior_retention_finetune_steps
            ):
                assert isinstance(model, StageEditTransformerDynamicCropV3)
                assert dynamic_crop_v3_prior_parameter_invariance is not None
                retained_digests = _dynamic_crop_v3_prior_parameter_digests(
                    model
                )
                warmup_digests = dynamic_crop_v3_prior_parameter_invariance[
                    "after"
                ]
                retained_keys = (
                    "context_weight_sha256",
                    "source_head_parameter_tree_sha256",
                    "context_non_value_bias_sha256",
                    "context_value_bias_sha256",
                )
                heads_identical = all(
                    retained_digests[key] == warmup_digests[key]
                    for key in retained_keys
                )
                if not heads_identical:
                    raise RuntimeError(
                        "Dynamic Crop v3 warmup prior drifted before burn-in"
                    )
                dynamic_crop_v3_prior_retention_evidence = {
                    "contract": (
                        "warmup_dynamic_heads_byte_retained_until_joint_"
                        "selection_burn_in_v1"
                    ),
                    "retention_finetune_step_count": (
                        dynamic_crop_v3_prior_retention_restore_steps
                    ),
                    "required_retention_finetune_step_count": (
                        dynamic_crop_v3_prior_retention_finetune_steps
                    ),
                    "warmup_head_digests": {
                        key: warmup_digests[key] for key in retained_keys
                    },
                    "retained_head_digests": {
                        key: retained_digests[key] for key in retained_keys
                    },
                    "all_dynamic_head_parameters_byte_identical": (
                        heads_identical
                    ),
                    "dynamic_head_gradients_zeroed": True,
                    "adamw_parameter_decay_restored": True,
                    "optimizer_reset_required_before_dynamic_head_release": True,
                    "validation_gradients_computed": False,
                    "heldout_forward_performed": False,
                }

            validation_base_metrics = evaluate_stage_edit_model(
                model,
                validation_batch,
                validation_supervision,
                loss_contract=effective_loss_contract,
            )
            validation_dynamic_metrics = None
            if dynamic_crop_v3_enabled:
                validation_dynamic_metrics = evaluate_stage_edit_dynamic_crop_v3(
                    model,
                    validation_batch,
                    validation_supervision,
                    validation_dynamic_supervision,
                    contract=effective_dynamic_crop_v3_loss_contract,
                )
                validation_metrics = {
                    **validation_base_metrics,
                    "total_loss": validation_dynamic_metrics["total_loss"],
                    "dynamic_crop_v3": validation_dynamic_metrics,
                }
            elif dynamic_crop_v2_enabled:
                validation_dynamic_metrics = evaluate_stage_edit_dynamic_crop_v2(
                    model,
                    validation_batch,
                    validation_supervision,
                    validation_dynamic_supervision,
                    contract=effective_dynamic_crop_v2_loss_contract,
                )
                validation_metrics = {
                    **validation_base_metrics,
                    "total_loss": validation_dynamic_metrics["total_loss"],
                    "dynamic_crop_v2": validation_dynamic_metrics,
                }
            else:
                validation_metrics = validation_base_metrics
            validation_loss = _finite_metric(validation_metrics.get("total_loss"))
            validation_selection = stage_edit_validation_selection_score(
                validation_metrics
            )
            if dynamic_crop_v3_enabled:
                validation_selection = (
                    _stage_edit_validation_selection_with_dynamic_crop_v3(
                        validation_selection,
                        validation_dynamic_metrics,
                    )
                )
            validation_selection_score = _finite_metric(
                validation_selection.get("score")
            )
            checkpoint_selection_eligible = bool(
                finetune_optimizer_steps
                >= minimum_checkpoint_selection_finetune_steps
            )
            improved = bool(
                checkpoint_selection_eligible
                and validation_selection.get("selectable")
                and validation_selection_score is not None
                and (
                    validation_selection_score
                    > best_validation_selection_score + 1.0e-8
                    or (
                        abs(
                            validation_selection_score
                            - best_validation_selection_score
                        )
                        <= 1.0e-8
                        and validation_loss is not None
                        and validation_loss < best_validation_loss - 1.0e-8
                    )
                )
            )
            history.append(
                {
                    "epoch": epoch,
                    "optimizer_step": optimizer_steps,
                    "finetune_optimizer_step_first": (
                        epoch_finetune_optimizer_steps[0]
                    ),
                    "finetune_optimizer_step_last": (
                        epoch_finetune_optimizer_steps[-1]
                    ),
                    "finetune_optimizer_steps": epoch_finetune_optimizer_steps,
                    "finetune_phase_local_optimizer_steps": (
                        epoch_phase_local_optimizer_steps
                    ),
                    "finetune_optimizer_reset_before_steps": (
                        epoch_optimizer_reset_before_steps
                    ),
                    "finetune_actual_learning_rates": (
                        epoch_actual_learning_rates
                    ),
                    "dynamic_crop_v3_prior_retention_finetune_steps": (
                        epoch_dynamic_crop_v3_prior_retention_steps
                    ),
                    "training_loss_mean": (
                        sum(epoch_losses) / len(epoch_losses)
                    ),
                    "gradient_norm_mean_before_clip": (
                        sum(epoch_gradient_norms) / len(epoch_gradient_norms)
                    ),
                    "training_song_reference_validation_total_loss": (
                        validation_loss
                    ),
                    "training_song_reference_validation_cut_auroc": (
                        validation_selection["components"]["cut_auroc"]
                    ),
                    "training_song_reference_validation_source_mrr": (
                        validation_selection["components"]["source_mrr"]
                    ),
                    "training_song_reference_validation_selection_score": (
                        validation_selection_score
                    ),
                    "training_song_reference_validation_selection_contract": (
                        STAGE_EDIT_VALIDATION_SELECTION_CONTRACT
                    ),
                    "training_song_reference_validation_dynamic_crop_v2": (
                        validation_dynamic_metrics
                        if dynamic_crop_v2_enabled
                        else None
                    ),
                    "training_song_reference_validation_dynamic_crop_v3": (
                        validation_dynamic_metrics
                        if dynamic_crop_v3_enabled
                        else None
                    ),
                    "training_song_reference_validation_dynamic_crop_version": (
                        dynamic_crop_version
                    ),
                    "training_song_reference_validation_dynamic_crop_selection_gate": (
                        validation_selection.get(
                            "dynamic_crop_v3_selection_gate"
                        )
                        if dynamic_crop_v3_enabled
                        else None
                    ),
                    "checkpoint_selection_eligible_after_burn_in": (
                        checkpoint_selection_eligible
                    ),
                    "training_song_reference_validation_improved": improved,
                    "dynamic_crop_v3_checkpoint_candidate_origin": (
                        "training_only_bias_prior_retained_to_burn_in"
                        if improved
                        and dynamic_crop_v3_prior_candidate_selection
                        and dynamic_crop_v3_prior_candidate_selection.get(
                            "retention_authorized"
                        )
                        is True
                        and finetune_optimizer_steps
                        == dynamic_crop_v3_prior_retention_finetune_steps
                        else "joint_finetuning"
                        if improved
                        else None
                    ),
                }
            )
            if improved:
                assert validation_loss is not None
                assert validation_selection_score is not None
                best_validation_loss = float(validation_loss)
                best_validation_selection_score = float(
                    validation_selection_score
                )
                best_validation_selection_evidence = validation_selection
                best_epoch = epoch
                best_step = optimizer_steps
                best_finetune_optimizer_step = finetune_optimizer_steps
                best_checkpoint_origin = (
                    "training_only_bias_prior_retained_to_burn_in"
                    if dynamic_crop_v3_prior_candidate_selection
                    and dynamic_crop_v3_prior_candidate_selection.get(
                        "retention_authorized"
                    )
                    is True
                    and finetune_optimizer_steps
                    == dynamic_crop_v3_prior_retention_finetune_steps
                    else "joint_finetuning"
                )
                finetune_steps_without_improvement = 0
                checkpoint_id = (
                    f"production-{corpus.data_id[:16]}-seed{seed}-"
                    f"step{optimizer_steps}"
                )
                if dynamic_crop_v3_enabled:
                    metadata = new_stage_edit_dynamic_crop_checkpoint_metadata_v3(
                        checkpoint_id,
                        schema,
                        config,
                        training_steps=optimizer_steps,
                        weights_trained=True,
                        training_data_id=(
                            f"{corpus.data_id}+{dynamic_crop_corpus.data_id}"
                            if dynamic_crop_corpus is not None
                            else corpus.data_id
                        ),
                    )
                    saved_metadata = save_stage_edit_dynamic_crop_checkpoint_v3(
                        checkpoint_root,
                        model,
                        metadata,
                    )
                elif dynamic_crop_v2_enabled:
                    metadata = new_stage_edit_dynamic_crop_checkpoint_metadata_v2(
                        checkpoint_id,
                        schema,
                        config,
                        training_steps=optimizer_steps,
                        weights_trained=True,
                        training_data_id=(
                            f"{corpus.data_id}+{dynamic_crop_corpus.data_id}"
                            if dynamic_crop_corpus is not None
                            else corpus.data_id
                        ),
                    )
                    saved_metadata = save_stage_edit_dynamic_crop_checkpoint_v2(
                        checkpoint_root,
                        model,
                        metadata,
                    )
                else:
                    metadata = new_stage_edit_checkpoint_metadata(
                        checkpoint_id,
                        schema,
                        config,
                        training_steps=optimizer_steps,
                        weights_trained=True,
                        training_data_id=corpus.data_id,
                    )
                    saved_metadata = save_stage_edit_checkpoint(
                        checkpoint_root,
                        model,
                        metadata,
                    )
            else:
                eligible_epoch_steps = max(
                    0,
                    epoch_finetune_optimizer_steps[-1]
                    - max(
                        epoch_finetune_optimizer_steps[0],
                        minimum_checkpoint_selection_finetune_steps,
                    )
                    + 1,
                )
                finetune_steps_without_improvement += eligible_epoch_steps
            if (
                finetune_optimizer_steps >= minimum_finetune_optimizer_steps
                and finetune_steps_without_improvement
                >= early_stopping_patience
                and finetune_optimizer_steps < finetune_steps_planned
            ):
                early_stopping_triggered = True
                training_stop_reason = (
                    "phase_local_early_stopping_after_finetune_step_floor"
                )
                break
        finetune_optimizer_state_final_step = int(
            optimizer.state["step"].item()
        )
        expected_active_optimizer_steps = (
            finetune_optimizer_steps
            - dynamic_crop_v3_prior_retention_finetune_steps
            if finetune_optimizer_reset_at_dynamic_crop_v3_prior_release
            else finetune_optimizer_steps
        )
        if finetune_optimizer_state_final_step != expected_active_optimizer_steps:
            raise RuntimeError(
                "multi-head optimizer state step differs from phase-local history"
            )
        if dynamic_crop_v3_prior_retention_finetune_steps:
            if (
                not finetune_optimizer_reset_at_dynamic_crop_v3_prior_release
                or finetune_joint_optimizer_initial_step != 0
                or dynamic_crop_v3_prior_retention_evidence is None
            ):
                raise RuntimeError(
                    "Dynamic Crop v3 prior release did not reset joint optimizer"
                )
            dynamic_crop_v3_prior_retention_evidence = {
                **dynamic_crop_v3_prior_retention_evidence,
                "retention_optimizer_state_final_step": (
                    finetune_retention_optimizer_state_final_step
                ),
                "released_joint_optimizer_initial_step": (
                    finetune_joint_optimizer_initial_step
                ),
                "released_joint_optimizer_state_final_step": (
                    finetune_optimizer_state_final_step
                ),
                "optimizer_state_reset_before_dynamic_head_release": True,
                "released_joint_optimizer_schedule_offset_steps": (
                    dynamic_crop_v3_prior_retention_finetune_steps
                ),
            }

    if (
        saved_metadata is None
        or best_step < 1
        or best_validation_selection_evidence is None
    ):
        raise RuntimeError(
            "no finite training-song reference-validation composite improvement; "
            "no checkpoint was saved"
        )

    with mx.stream(active_device):
        if dynamic_crop_v3_enabled:
            best_model, loaded_metadata = (
                load_stage_edit_dynamic_crop_checkpoint_v3(
                    checkpoint_root,
                    expected_schema=schema,
                )
            )
        elif dynamic_crop_v2_enabled:
            best_model, loaded_metadata = (
                load_stage_edit_dynamic_crop_checkpoint_v2(
                    checkpoint_root,
                    expected_schema=schema,
                )
            )
        else:
            best_model, loaded_metadata = load_stage_edit_checkpoint(
                checkpoint_root,
                expected_schema=schema,
            )
        frozen_cut_threshold = select_stage_edit_cut_threshold_from_model(
            best_model,
            validation_batch,
            validation_supervision,
        )
        frozen_cut_threshold = {
            **frozen_cut_threshold,
            "validation_reference_ids": [
                example.reference_id for example in split.validation
            ],
            "selected_checkpoint_id": loaded_metadata.checkpoint_id,
            "selected_checkpoint_weight_sha256": loaded_metadata.weight_sha256,
            "selected_before_heldout_forward": True,
        }
        frozen_cut_threshold = {
            **frozen_cut_threshold,
            "freeze_sha256": _canonical_digest(frozen_cut_threshold),
        }
        selected_cut_threshold = float(
            frozen_cut_threshold["selected_probability_threshold"]
        )
        selected_validation_base_metrics = evaluate_stage_edit_model(
            best_model,
            validation_batch,
            validation_supervision,
            loss_contract=effective_loss_contract,
            cut_probability_threshold=selected_cut_threshold,
        )
        selected_validation_dynamic_metrics = None
        if dynamic_crop_v3_enabled:
            selected_validation_dynamic_metrics = (
                evaluate_stage_edit_dynamic_crop_v3(
                    best_model,
                    validation_batch,
                    validation_supervision,
                    validation_dynamic_supervision,
                    contract=effective_dynamic_crop_v3_loss_contract,
                )
            )
            selected_validation_metrics = {
                **selected_validation_base_metrics,
                "total_loss": selected_validation_dynamic_metrics[
                    "total_loss"
                ],
                "dynamic_crop_v3": selected_validation_dynamic_metrics,
            }
        elif dynamic_crop_v2_enabled:
            selected_validation_dynamic_metrics = (
                evaluate_stage_edit_dynamic_crop_v2(
                    best_model,
                    validation_batch,
                    validation_supervision,
                    validation_dynamic_supervision,
                    contract=effective_dynamic_crop_v2_loss_contract,
                )
            )
            selected_validation_metrics = {
                **selected_validation_base_metrics,
                "total_loss": selected_validation_dynamic_metrics[
                    "total_loss"
                ],
                "dynamic_crop_v2": selected_validation_dynamic_metrics,
            }
        else:
            selected_validation_metrics = selected_validation_base_metrics
        reloaded_validation_selection = stage_edit_validation_selection_score(
            selected_validation_metrics
        )
        if dynamic_crop_v3_enabled:
            reloaded_validation_selection = (
                _stage_edit_validation_selection_with_dynamic_crop_v3(
                    reloaded_validation_selection,
                    selected_validation_dynamic_metrics,
                )
            )
        reloaded_validation_selection_score = _finite_metric(
            reloaded_validation_selection.get("score")
        )
        if (
            reloaded_validation_selection.get("selectable") is not True
            or reloaded_validation_selection_score is None
            or abs(
                reloaded_validation_selection_score
                - best_validation_selection_score
            )
            > 1.0e-8
        ):
            raise RuntimeError(
                "reloaded checkpoint no longer satisfies its frozen validation "
                "selection evidence"
            )
        selection_freeze_evidence = _write_selection_freeze(
            checkpoint_root,
            {
                "schema_version": 1,
                "contract": (
                    "stage_edit_model_config_checkpoint_and_threshold_freeze_v1"
                ),
                "created_at": datetime.now(UTC).isoformat(),
                "model_config": config.to_dict(),
                "model_config_fingerprint_sha256": _canonical_digest(
                    config.to_dict()
                ),
                "checkpoint_id": loaded_metadata.checkpoint_id,
                "checkpoint_weight_sha256": loaded_metadata.weight_sha256,
                "dynamic_crop_version": dynamic_crop_version,
                "dynamic_crop_model_contract": getattr(
                    loaded_metadata, "model_contract", None
                ),
                "dynamic_crop_corpus_data_id": (
                    dynamic_crop_corpus.data_id
                    if dynamic_crop_corpus is not None
                    else None
                ),
                "selected_training_step": best_step,
                "selected_finetune_optimizer_step": (
                    best_finetune_optimizer_step
                ),
                "selected_checkpoint_origin": best_checkpoint_origin,
                "dynamic_crop_v3_prior_pretraining_optimizer_steps": (
                    dynamic_crop_v3_prior_pretraining_optimizer_steps
                ),
                "dynamic_crop_v3_prior_candidate_retained_to_burn_in": (
                    best_checkpoint_origin
                    == "training_only_bias_prior_retained_to_burn_in"
                ),
                "dynamic_crop_v3_prior_selected_candidate_step": (
                    dynamic_crop_v3_prior_candidate_selection.get(
                        "selected_candidate_step"
                    )
                    if dynamic_crop_v3_prior_candidate_selection
                    else None
                ),
                "dynamic_crop_v3_prior_selected_candidate_value_bias_sha256": (
                    dynamic_crop_v3_prior_candidate_selection.get(
                        "selected_candidate_value_bias_sha256"
                    )
                    if dynamic_crop_v3_prior_candidate_selection
                    else None
                ),
                "minimum_checkpoint_selection_finetune_steps": (
                    minimum_checkpoint_selection_finetune_steps
                ),
                "selected_after_checkpoint_burn_in": (
                    best_finetune_optimizer_step
                    >= minimum_checkpoint_selection_finetune_steps
                ),
                "validation_selection": dict(
                    best_validation_selection_evidence
                ),
                "reloaded_validation_selection": dict(
                    reloaded_validation_selection
                ),
                "selected_validation_metrics": selected_validation_metrics,
                "cut_threshold_freeze": frozen_cut_threshold,
                "training_song_ids": list(split.training_song_ids),
                "training_reference_ids": [
                    example.reference_id for example in split.train
                ],
                "validation_reference_ids": [
                    example.reference_id for example in split.validation
                ],
                "held_out_song_ids_reserved_for_one_forward": list(
                    split.held_out_song_ids
                ),
                "heldout_metrics_present": False,
                "heldout_used_for_model_or_threshold_selection": False,
                "style_or_attitude_configuration_mutable_after_freeze": False,
            },
        )
        held_out_batch = tensorize_stage_edit_batch(
            prepare_stage_edit_production_input_sequences(
                split.held_out, schema=schema
            ),
            schema,
            config,
            device=active_device,
        )
        held_out_supervision = tensorize_stage_edit_supervision(
            tuple(example.targets for example in split.held_out),
            held_out_batch,
            schema,
            device=active_device,
        )
        held_out_metrics = evaluate_stage_edit_model(
            best_model,
            held_out_batch,
            held_out_supervision,
            loss_contract=effective_loss_contract,
            cut_probability_threshold=selected_cut_threshold,
        )
        held_out_metrics = {
            **held_out_metrics,
            "cut": {
                **held_out_metrics["cut"],
                "balanced_accuracy_at_frozen_validation_threshold": (
                    held_out_metrics["cut"]["balanced_accuracy_at_threshold"]
                ),
                "frozen_validation_threshold": selected_cut_threshold,
                "threshold_contract": (
                    STAGE_EDIT_CUT_THRESHOLD_SELECTION_CONTRACT
                ),
            },
        }
        held_out_dynamic_v2_metrics = None
        held_out_dynamic_v3_metrics = None
        if dynamic_crop_v3_enabled:
            held_out_dynamic_supervision = (
                tensorize_stage_edit_dynamic_crop_supervision_v3(
                    dynamic_sequences_for(split.held_out),
                    held_out_batch,
                    device=active_device,
                )
            )
            held_out_dynamic_v3_metrics = evaluate_stage_edit_dynamic_crop_v3(
                best_model,
                held_out_batch,
                held_out_supervision,
                held_out_dynamic_supervision,
                contract=effective_dynamic_crop_v3_loss_contract,
            )
        elif dynamic_crop_v2_enabled:
            held_out_dynamic_supervision = (
                tensorize_stage_edit_dynamic_crop_supervision_v2(
                    dynamic_sequences_for(split.held_out),
                    held_out_batch,
                    device=active_device,
                )
            )
            held_out_dynamic_v2_metrics = evaluate_stage_edit_dynamic_crop_v2(
                best_model,
                held_out_batch,
                held_out_supervision,
                held_out_dynamic_supervision,
                contract=effective_dynamic_crop_v2_loss_contract,
            )
    quality_gate = evaluate_stage_edit_production_quality(
        held_out_metrics,
        dynamic_crop_v2_metrics=held_out_dynamic_v2_metrics,
        dynamic_crop_v3_metrics=held_out_dynamic_v3_metrics,
        thresholds=quality_thresholds,
        transition_authority=config.transition_authority,
    )
    structured_head_coverage_gate = _structured_head_coverage_gate(
        split.train,
        held_out_metrics,
        thresholds=quality_thresholds,
        transition_authority=config.transition_authority,
    )
    reported_held_out_metrics = dict(held_out_metrics)
    if transition_authority.renderer_owned:
        reported_held_out_metrics["transition"] = dict(
            quality_gate.metrics["transition"]
        )
    production_ready = bool(
        authorize_production_activation
        and quality_gate.passed
        and structured_head_coverage_gate["passed"]
    )
    checkpoint_evidence = {
        "directory": str(checkpoint_root),
        "checkpoint_format": (
            "dynamic_crop_v3"
            if dynamic_crop_v3_enabled
            else "dynamic_crop_v2"
            if dynamic_crop_v2_enabled
            else "legacy"
        ),
        "dynamic_crop_version": dynamic_crop_version,
        "metadata_filename": (
            DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
            if dynamic_crop_v3_enabled
            else DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME
            if dynamic_crop_v2_enabled
            else "metadata.json"
        ),
        "checkpoint_id": loaded_metadata.checkpoint_id,
        "training_steps": loaded_metadata.training_steps,
        "weights_trained": loaded_metadata.weights_trained,
        "weight_filename": loaded_metadata.weight_filename,
        "weight_sha256": loaded_metadata.weight_sha256,
        "production_activation_authorized": production_ready,
        "production_supervision_gate_passed": supervision_gate.passed,
        "dynamic_crop_v2_checkpoint": dynamic_crop_v2_enabled,
        "dynamic_crop_v2_data_id": (
            dynamic_crop_corpus.data_id
            if dynamic_crop_v2_enabled
            else None
        ),
        "dynamic_crop_v3_checkpoint": dynamic_crop_v3_enabled,
        "dynamic_crop_v3_data_id": (
            dynamic_crop_corpus.data_id
            if dynamic_crop_v3_enabled
            else None
        ),
        "dynamic_crop_model_contract": getattr(
            loaded_metadata, "model_contract", None
        ),
        "dynamic_crop_target_contract": getattr(
            loaded_metadata, "target_contract", None
        ),
        "dynamic_crop_output_contract": getattr(
            loaded_metadata, "output_contract", None
        ),
    }
    report: dict[str, Any] = {
        "schema_version": PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION,
        "run_kind": PRODUCTION_TRAINING_RUN_KIND,
        "checkpoint_scope": PRODUCTION_CHECKPOINT_SCOPE,
        "created_at": datetime.now(UTC).isoformat(),
        "production_ready": production_ready,
        "production_readiness_reason": (
            "unseen_song_quality_gate_passed_and_activation_authorized"
            if production_ready
            else "structured_head_coverage_gate_failed"
            if not structured_head_coverage_gate["passed"]
            else "unseen_song_quality_gate_failed"
            if not quality_gate.passed
            else "activation_not_explicitly_authorized"
        ),
        "production_supervision_gate": {
            **supervision_gate.to_dict(),
            "scope": "training_songs_only_including_reference_validation",
            "held_out_references_included": False,
        },
        "production_feature_snapshot_gate": feature_snapshot_gate,
        "decision_history_input_contract": (
            stage_edit_recent_history_mask_contract(schema)
        ),
        "source_scoring_contract": stage_edit_source_scoring_contract(config),
        "source_evaluation_contract": stage_edit_source_evaluation_contract(),
        "cut_scoring_contract": stage_edit_cut_scoring_contract(config),
        "transition_authority_contract": transition_authority.to_dict(),
        "transition_authority_contract_sha256": transition_authority.fingerprint,
        "cut_imbalance_contract": cut_imbalance_contract,
        "cut_threshold_freeze": frozen_cut_threshold,
        "selection_freeze_evidence": selection_freeze_evidence,
        "structured_head_coverage_gate": structured_head_coverage_gate,
        "data": {
            "data_id": corpus.data_id,
            "dynamic_crop_version": dynamic_crop_version,
            "source_paths": list(corpus.source_paths),
            "statistics": dict(corpus.statistics),
            "corpus_manifest_sha256": _canonical_digest(corpus.manifest()),
            "dynamic_crop_v2": (
                {
                    "data_id": dynamic_crop_corpus.data_id,
                    "statistics": dict(dynamic_crop_corpus.statistics),
                    "manifest_sha256": _canonical_digest(
                        dynamic_crop_corpus.manifest()
                    ),
                }
                if dynamic_crop_v2_enabled
                else None
            ),
            "dynamic_crop_v3": (
                {
                    "data_id": dynamic_crop_corpus.data_id,
                    "source_v2_data_id": dynamic_crop_corpus.source_v2_data_id,
                    "contract": dynamic_crop_corpus.contract,
                    "target_contract": dynamic_crop_corpus.target_contract,
                    "statistics": dict(dynamic_crop_corpus.statistics),
                    "manifest_sha256": _canonical_digest(
                        dynamic_crop_corpus.manifest()
                    ),
                }
                if dynamic_crop_v3_enabled
                else None
            ),
        },
        "split": split.to_dict(),
        "training_song_expansion_contract": (
            stage_edit_training_song_expansion_contract(split)
        ),
        "split_label_statistics": {
            "training_cut_prevalence_by_song": _cut_prevalence_by_song(
                split.train
            ),
            "reference_validation_cut_prevalence_by_song": (
                _cut_prevalence_by_song(split.validation)
            ),
            "heldout_statistics_used_for_selection": False,
        },
        "model": {
            "config": config.to_dict(),
            "parameter_count": (
                stage_edit_dynamic_crop_v3_parameter_count(config)
                if dynamic_crop_v3_enabled
                else stage_edit_dynamic_crop_v2_parameter_count(config)
                if dynamic_crop_v2_enabled
                else stage_edit_parameter_count(config)
            ),
            "mlx_version": mlx_runtime_version(),
            "device": str(active_device),
            "architecture": (
                "causal_long_context_transformer_with_source_conditioned_"
                "dynamic_crop_v3"
                if dynamic_crop_v3_enabled
                else "causal_long_context_transformer_with_k8_dynamic_crop_v2"
                if dynamic_crop_v2_enabled
                else "causal_long_context_multi_head_transformer_legacy_crop"
            ),
            "dynamic_crop_version": dynamic_crop_version,
            "dynamic_crop_v2_enabled": dynamic_crop_v2_enabled,
            "dynamic_crop_v3_enabled": dynamic_crop_v3_enabled,
            "activation_checkpointing": (
                dict(activation_checkpointing_evidence)
                if activation_checkpointing_evidence is not None
                else None
            ),
            "dynamic_crop_model_contract": getattr(
                loaded_metadata, "model_contract", None
            ),
            "dynamic_crop_target_contract": getattr(
                loaded_metadata, "target_contract", None
            ),
            "dynamic_crop_output_contract": getattr(
                loaded_metadata, "output_contract", None
            ),
        },
        "optimization": {
            "seed": seed,
            "requested_epochs": epochs,
            "completed_epochs": len(history),
            "batch_size": batch_size,
            "optimizer_steps": optimizer_steps,
            "cut_pretraining_optimizer_steps": (
                cut_pretraining_optimizer_steps
            ),
            "source_pretraining_optimizer_steps": (
                source_pretraining_optimizer_steps
            ),
            "dynamic_crop_v3_prior_pretraining_optimizer_steps": (
                dynamic_crop_v3_prior_pretraining_optimizer_steps
            ),
            "multi_head_finetuning_optimizer_steps": (
                finetune_optimizer_steps
            ),
            "source_head_update_steps": (
                source_pretraining_optimizer_steps + finetune_optimizer_steps
            ),
            "learning_rate": learning_rate,
            "learning_rate_schedule": {
                "contract": (
                    "independent_warmup_cosine_cut_source_v3_prior_and_"
                    "joint_with_retained_selection_burn_in_v4"
                    if dynamic_crop_v3_prior_pretrain_steps
                    else "independent_warmup_cosine_cut_source_and_joint_"
                    "with_step_floor_v3"
                ),
                "cut_pretraining_steps": cut_pretrain_steps,
                "cut_pretraining_peak_learning_rate": (
                    cut_pretrain_learning_rate
                ),
                "cut_pretraining_window_length": cut_pretrain_window_length,
                "cut_pretraining_warmup_steps": (
                    cut_pretrain_warmup_steps if cut_pretrain_steps else 0
                ),
                "source_pretraining_steps": source_pretrain_steps,
                "source_pretraining_peak_learning_rate": (
                    source_pretrain_learning_rate
                ),
                "source_pretraining_window_length": (
                    source_pretrain_window_length
                ),
                "source_pretraining_warmup_steps": (
                    source_pretrain_warmup_steps
                    if source_pretrain_steps
                    else 0
                ),
                "dynamic_crop_v3_prior_pretraining_steps": (
                    dynamic_crop_v3_prior_pretrain_steps
                ),
                "dynamic_crop_v3_prior_pretraining_peak_learning_rate": (
                    dynamic_crop_v3_prior_pretrain_learning_rate
                ),
                "dynamic_crop_v3_prior_pretraining_warmup_steps": (
                    dynamic_crop_v3_prior_pretrain_warmup_steps
                    if dynamic_crop_v3_prior_pretrain_steps
                    else 0
                ),
                "dynamic_crop_v3_prior_retention_finetune_steps": (
                    dynamic_crop_v3_prior_retention_finetune_steps
                ),
                "finetune_steps_planned": finetune_steps_planned,
                "finetune_warmup_steps": finetune_warmup,
                "minimum_learning_rate_ratio": minimum_learning_rate_ratio,
            },
            "cut_pretraining": {
                "enabled": bool(cut_pretrain_steps),
                "partition": "training_references_only",
                "balanced_anchor_positive_count_per_step": (
                    1 if cut_pretrain_steps else 0
                ),
                "balanced_anchor_negative_count_per_step": (
                    1 if cut_pretrain_steps else 0
                ),
                "other_head_losses_zero_weighted": True,
                "source_head_losses_zero_weighted": True,
                "checkpoint_selection_eligible": False,
                "positive_weight": 1.0,
                "minimum_step_contract": 500,
                "validation_labels_consumed": False,
                "heldout_labels_consumed": False,
            },
            "source_pretraining": {
                "enabled": bool(source_pretrain_steps),
                "partition": "training_references_only",
                "confirmed_event_contract": (
                    "target_cut_equals_one_and_selected_source_id_v1"
                ),
                "opening_source_selection_excluded": True,
                "loss_contract": asdict(source_only_loss_contract),
                "listwise_loss_enabled": bool(
                    source_only_loss_contract.source_listwise_weight
                ),
                "eligible_all_pairwise_loss_enabled": bool(
                    source_only_loss_contract.source_pairwise_weight
                ),
                "other_head_losses_zero_weighted": True,
                "checkpoint_selection_eligible": False,
                "minimum_step_contract": 500,
                "validation_labels_consumed": False,
                "heldout_labels_consumed": False,
                "sampling": source_pretraining_sampling_evidence,
            },
            "dynamic_crop_v3_prior_pretraining": (
                {
                    "enabled": True,
                    "partition": "training_references_only",
                    "objective": (
                        "renderer_range_normalized_coordinate_mae_exactly_"
                        "aligned_with_validation_gate_v1"
                    ),
                    "mutable_parameter_scope": (
                        "dynamic_crop_context_head_bias_value_rows_0_31_only"
                    ),
                    "optimizer_weight_decay": 0.0,
                    "checkpoint_selection_eligible_during_warmup": False,
                    "checkpoint_write_before_joint_finetuning": False,
                    "selection_gate_evaluated_without_gradients": True,
                    "validation_labels_consumed_for_gradients": False,
                    "heldout_labels_consumed": False,
                    "sampling": dynamic_crop_v3_prior_sampling_evidence,
                    "component_scale": (
                        dynamic_crop_v3_prior_component_scale_evidence
                    ),
                    "parameter_invariance": (
                        dynamic_crop_v3_prior_parameter_invariance
                    ),
                    "candidate_selection": (
                        dynamic_crop_v3_prior_candidate_selection
                    ),
                    "retention": dynamic_crop_v3_prior_retention_evidence,
                }
                if dynamic_crop_v3_prior_pretrain_steps
                else None
            ),
            "weight_decay": weight_decay,
            "gradient_clip_norm": gradient_clip_norm,
            "effective_loss_contract": asdict(effective_loss_contract),
            "effective_dynamic_crop_loss_contract": (
                asdict(effective_dynamic_crop_v3_loss_contract)
                if dynamic_crop_v3_enabled
                else asdict(effective_dynamic_crop_v2_loss_contract)
                if dynamic_crop_v2_enabled
                else None
            ),
            "effective_dynamic_crop_v2_loss_contract": (
                asdict(effective_dynamic_crop_v2_loss_contract)
                if dynamic_crop_v2_enabled
                else None
            ),
            "effective_dynamic_crop_v3_loss_contract": (
                asdict(effective_dynamic_crop_v3_loss_contract)
                if dynamic_crop_v3_enabled
                else None
            ),
            "effective_dynamic_crop_loss_contract_version": (
                dynamic_crop_version
            ),
            "gradient_clipping_supported": clipping_supported,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_patience_unit": "finetune_optimizer_steps",
            "minimum_finetune_optimizer_steps": (
                minimum_finetune_optimizer_steps
            ),
            "minimum_checkpoint_selection_finetune_steps": (
                minimum_checkpoint_selection_finetune_steps
            ),
            "finetune_optimizer_initial_step": (
                finetune_optimizer_initial_step
            ),
            "finetune_optimizer_state_final_step": (
                finetune_optimizer_state_final_step
            ),
            "finetune_retention_optimizer_state_final_step": (
                finetune_retention_optimizer_state_final_step
            ),
            "finetune_joint_optimizer_initial_step": (
                finetune_joint_optimizer_initial_step
            ),
            "finetune_optimizer_reset_at_dynamic_crop_v3_prior_release": (
                finetune_optimizer_reset_at_dynamic_crop_v3_prior_release
            ),
            "finetune_optimizer_state_reset_after_cut_pretraining": (
                finetune_optimizer_initial_step == 0
                and finetune_optimizer_state_final_step
                == expected_active_optimizer_steps
            ),
            "cut_pretraining_optimizer_initial_step": (
                cut_pretraining_optimizer_initial_step
            ),
            "cut_pretraining_optimizer_state_final_step": (
                cut_pretraining_optimizer_state_final_step
            ),
            "source_pretraining_optimizer_initial_step": (
                source_pretraining_optimizer_initial_step
            ),
            "source_pretraining_optimizer_state_final_step": (
                source_pretraining_optimizer_state_final_step
            ),
            "dynamic_crop_v3_prior_optimizer_initial_step": (
                dynamic_crop_v3_prior_optimizer_initial_step
            ),
            "dynamic_crop_v3_prior_optimizer_state_final_step": (
                dynamic_crop_v3_prior_optimizer_state_final_step
            ),
            "source_pretraining_optimizer_state_reset_after_cut_pretraining": (
                source_pretraining_optimizer_initial_step == 0
                and source_pretraining_optimizer_state_final_step
                == source_pretraining_optimizer_steps
            ),
            "finetune_optimizer_state_reset_after_cut_and_source_pretraining": (
                finetune_optimizer_initial_step == 0
                and finetune_optimizer_state_final_step
                == expected_active_optimizer_steps
            ),
            "dynamic_crop_v3_prior_optimizer_state_reset_after_source_pretraining": (
                dynamic_crop_v3_prior_optimizer_initial_step == 0
                and dynamic_crop_v3_prior_optimizer_state_final_step
                == dynamic_crop_v3_prior_pretraining_optimizer_steps
            ),
            "finetune_optimizer_state_reset_after_dynamic_crop_v3_prior_pretraining": (
                finetune_optimizer_initial_step == 0
                and finetune_optimizer_state_final_step
                == expected_active_optimizer_steps
                and (
                    not dynamic_crop_v3_prior_retention_finetune_steps
                    or finetune_optimizer_reset_at_dynamic_crop_v3_prior_release
                    and finetune_joint_optimizer_initial_step == 0
                )
            ),
            "checkpoint_writes_before_joint_finetuning": 0,
            "finetune_steps_without_improvement_at_stop": (
                finetune_steps_without_improvement
            ),
            "early_stopping_triggered": early_stopping_triggered,
            "training_stop_reason": training_stop_reason,
            "finetune_schedule_completion_fraction": (
                finetune_optimizer_steps / finetune_steps_planned
            ),
            "best_epoch": best_epoch,
            "best_training_step": best_step,
            "best_finetune_optimizer_step": (
                best_finetune_optimizer_step
            ),
            "best_checkpoint_origin": best_checkpoint_origin,
            "best_validation_total_loss": best_validation_loss,
            "best_validation_selection_score": (
                best_validation_selection_score
            ),
            "best_validation_selection_evidence": dict(
                best_validation_selection_evidence
            ),
            "selected_checkpoint_validation_metrics": (
                selected_validation_metrics
            ),
            "reloaded_checkpoint_validation_selection_evidence": dict(
                reloaded_validation_selection
            ),
            "model_selection": STAGE_EDIT_VALIDATION_SELECTION_CONTRACT,
            "dynamic_crop_v3_model_selection": (
                DYNAMIC_CROP_V3_VALIDATION_SELECTION_CONTRACT
                if dynamic_crop_v3_enabled
                else None
            ),
            "unseen_song_used_for_model_selection": False,
        },
        "checkpoint": checkpoint_evidence,
        "held_out_evaluation": {
            "passed": quality_gate.passed,
            "split_level": "song",
            "checkpoint_id": loaded_metadata.checkpoint_id,
            "weight_sha256": loaded_metadata.weight_sha256,
            "training_song_ids": list(split.training_song_ids),
            "held_out_song_ids": list(split.held_out_song_ids),
            "held_out_reference_count": len(split.held_out),
            "cut_threshold_frozen_before_heldout_forward": True,
            "cut_threshold_freeze_sha256": frozen_cut_threshold[
                "freeze_sha256"
            ],
            "metrics": reported_held_out_metrics,
            "dynamic_crop_version": dynamic_crop_version,
            "dynamic_crop_v2_metrics": held_out_dynamic_v2_metrics,
            "dynamic_crop_v3_metrics": held_out_dynamic_v3_metrics,
            "heldout_dynamic_crop_metrics_used_for_checkpoint_selection": False,
            "quality_gate": quality_gate.to_dict(),
        },
        "cut_pretraining_history": cut_pretraining_history,
        "source_pretraining_history": source_pretraining_history,
        "dynamic_crop_v3_prior_pretraining_history": (
            dynamic_crop_v3_prior_pretraining_history
        ),
        "history": history,
    }
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    readiness = stage_edit_readiness_report(checkpoint_root)
    if production_ready and not readiness.learned_inference_ready:
        raise RuntimeError(
            "production report was written but readiness rejected it: "
            f"{readiness.status}: {readiness.reason}"
        )
    if not production_ready and readiness.learned_inference_ready:
        raise RuntimeError("non-production training run unexpectedly activated inference")
    return ProductionTrainingRunResult(
        checkpoint_directory=checkpoint_root,
        report_path=report_path,
        best_epoch=best_epoch,
        best_training_step=best_step,
        quality_gate=quality_gate,
        production_ready=production_ready,
        report=report,
    )


__all__ = [
    "DYNAMIC_CROP_V3_PRIOR_CANDIDATE_SELECTION_CONTRACT",
    "DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS",
    "DYNAMIC_CROP_V3_VALIDATION_SELECTION_CONTRACT",
    "PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION",
    "STAGE_EDIT_SELECTION_FREEZE_FILENAME",
    "STAGE_EDIT_TRAINING_SONG_EXPANSION_CONTRACT",
    "ProductionQualityGate",
    "ProductionQualityThresholds",
    "ProductionSongSplit",
    "ProductionTrainingRunResult",
    "evaluate_stage_edit_production_quality",
    "prepare_stage_edit_production_input_sequences",
    "production_song_split",
    "stage_edit_training_song_expansion_contract",
    "train_production_stage_edit",
]
