"""Offline MLX training/evaluation for sparse StageEdit supervision.

This is a calibration vertical slice, not a production editor.  Splits are
made at song level when at least two explicit song groups exist and otherwise
fall back to whole-reference groups.  Tokens from one reference are never
split across training and validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .stage_edit_reference_adapter import (
    ReferenceTrainingCorpus,
    ReferenceTrainingExample,
)
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    STAGE_EDIT_CUT_THRESHOLD_SELECTION_CONTRACT,
    STAGE_EDIT_VALIDATION_SELECTION_CONTRACT,
    StageEditLossContract,
    StageEditTransformer,
    StageEditTransformerConfig,
    StageEditTokenSchema,
    load_stage_edit_checkpoint,
    mlx_runtime_version,
    new_stage_edit_checkpoint_metadata,
    save_stage_edit_checkpoint,
    stage_edit_loss,
    stage_edit_parameter_count,
    tensorize_stage_edit_batch,
    tensorize_stage_edit_supervision,
)

try:  # Kept optional in the project package.
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ImportError as exc:  # pragma: no cover - non-Apple portability.
    mx = None
    nn = None
    optim = None
    _MLX_IMPORT_ERROR: ImportError | None = exc
else:
    _MLX_IMPORT_ERROR = None


TRAINING_REPORT_SCHEMA_VERSION = 2
STAGE_EDIT_VALIDATION_SELECTION_WEIGHTS = {
    "cut_auroc": 0.45,
    "source_mrr": 0.35,
    "bounded_inverse_total_loss": 0.20,
}
STAGE_EDIT_SOURCE_EVALUATION_CONTRACT = (
    "conditional_head_and_end_to_end_semimarkov_source_metrics_v1"
)


def _require_mlx() -> None:
    if mx is None or nn is None or optim is None:
        detail = f": {_MLX_IMPORT_ERROR}" if _MLX_IMPORT_ERROR else ""
        raise RuntimeError(f"offline StageEdit training requires MLX{detail}")


def sparse_calibration_stage_edit_config(
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
    *,
    max_sequence_length: int = 256,
    max_sources: int = 8,
) -> StageEditTransformerConfig:
    """Small configuration suitable for only hundreds of sparse labels."""

    return StageEditTransformerConfig.from_schema(
        schema,
        model_dimension=64,
        num_heads=4,
        num_layers=2,
        feed_forward_dimension=192,
        max_sequence_length=max_sequence_length,
        max_sources=max_sources,
        dropout=0.05,
    )


@dataclass(frozen=True, slots=True)
class ReferenceSplit:
    train: tuple[ReferenceTrainingExample, ...]
    validation: tuple[ReferenceTrainingExample, ...]
    split_level: str
    fallback_reason: str | None
    training_group_ids: tuple[str, ...]
    validation_group_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_level": self.split_level,
            "fallback_reason": self.fallback_reason,
            "training_group_ids": list(self.training_group_ids),
            "validation_group_ids": list(self.validation_group_ids),
            "training_reference_ids": [item.reference_id for item in self.train],
            "validation_reference_ids": [
                item.reference_id for item in self.validation
            ],
        }


@dataclass(frozen=True, slots=True)
class ProductionSupervisionGate:
    """Fail-closed evidence gate; it does not activate learned production use."""

    passed: bool
    reasons: tuple[str, ...]
    dense_editorial_switch_count: int
    reference_count_with_dense_editorial_switches: int
    song_count_with_dense_editorial_switches: int
    native_source_camera_cut_count: int
    native_source_camera_cut_editorial_positive_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "requirements": {
                "real_cross_source_editorial_switches_required": True,
                "dense_attribution_event_schema_required": True,
                "minimum_references_with_switches": 2,
                "minimum_songs_with_switches": 2,
                "native_source_camera_cuts_must_not_be_positives": True,
            },
            "observed": {
                "dense_editorial_switch_count": (
                    self.dense_editorial_switch_count
                ),
                "reference_count_with_dense_editorial_switches": (
                    self.reference_count_with_dense_editorial_switches
                ),
                "song_count_with_dense_editorial_switches": (
                    self.song_count_with_dense_editorial_switches
                ),
                "native_source_camera_cut_count": (
                    self.native_source_camera_cut_count
                ),
                "native_source_camera_cut_editorial_positive_count": (
                    self.native_source_camera_cut_editorial_positive_count
                ),
            },
            "authorizes_production_activation": False,
        }


def validate_stage_edit_production_supervision(
    corpus: ReferenceTrainingCorpus,
) -> ProductionSupervisionGate:
    """Prove the corpus contains actual editor source-switch decisions.

    A passing result is necessary, never sufficient, for production use.  The
    sparse StageEdit trainer remains an offline calibration path and cannot
    activate the transformer in the deterministic render pipeline.
    """

    dense_examples: list[ReferenceTrainingExample] = []
    dense_switch_count = 0
    native_count = 0
    native_positive_count = 0
    non_editorial_positive_count = 0
    for example in corpus.examples:
        statistics = example.statistics
        dense_count = int(
            statistics.get("dense_editorial_source_switch_positive_count", 0)
        )
        editorial_count = int(
            statistics.get("editorial_source_switch_positive_count", 0)
        )
        cut_positive_count = int(statistics.get("cut_positive_count", 0))
        dense_switch_count += dense_count
        native_count += int(
            statistics.get("native_source_camera_cut_count", 0)
        )
        native_positive_count += int(
            statistics.get(
                "native_source_camera_cut_editorial_positive_count", 0
            )
        )
        non_editorial_positive_count += max(
            cut_positive_count - editorial_count, 0
        )
        if dense_count > 0:
            dense_examples.append(example)

    reference_ids = {example.reference_id for example in dense_examples}
    song_ids = {
        example.song_id
        for example in dense_examples
        if example.song_id is not None
    }
    reasons: list[str] = []
    if dense_switch_count < 1:
        reasons.append("no_dense_cross_source_editorial_switches")
    if len(reference_ids) < 2:
        reasons.append(
            "fewer_than_two_references_with_dense_editorial_switches"
        )
    if len(song_ids) < 2:
        reasons.append("fewer_than_two_songs_with_dense_editorial_switches")
    if any(example.song_id is None for example in dense_examples):
        reasons.append("dense_editorial_switch_reference_missing_song_id")
    if native_positive_count:
        reasons.append("native_source_camera_cut_labeled_as_editorial_positive")
    if non_editorial_positive_count:
        reasons.append("non_editorial_cut_positive_detected")
    return ProductionSupervisionGate(
        passed=not reasons,
        reasons=tuple(reasons),
        dense_editorial_switch_count=dense_switch_count,
        reference_count_with_dense_editorial_switches=len(reference_ids),
        song_count_with_dense_editorial_switches=len(song_ids),
        native_source_camera_cut_count=native_count,
        native_source_camera_cut_editorial_positive_count=(
            native_positive_count
        ),
    )


def require_stage_edit_production_supervision(
    corpus: ReferenceTrainingCorpus,
) -> ProductionSupervisionGate:
    """Raise instead of allowing an under-evidenced production checkpoint."""

    gate = validate_stage_edit_production_supervision(corpus)
    if not gate.passed:
        raise ValueError(
            "StageEdit production supervision gate failed: "
            + ", ".join(gate.reasons)
        )
    return gate


def _stable_order(value: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()
    return digest, value


def deterministic_reference_split(
    corpus: ReferenceTrainingCorpus,
    *,
    seed: int = 20260823,
    validation_fraction: float = 0.25,
) -> ReferenceSplit:
    """Keep entire song/reference groups out of the training fold."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if len(corpus.examples) < 2:
        raise ValueError("at least two references are required for validation")
    explicit_songs = [example.song_id for example in corpus.examples]
    song_level_possible = (
        all(song_id is not None for song_id in explicit_songs)
        and len(set(explicit_songs)) >= 2
    )
    if song_level_possible:
        split_level = "song"
        fallback_reason = None
        group_for = lambda example: str(example.song_id)
    else:
        split_level = "reference"
        fallback_reason = (
            "fewer_than_two_explicit_song_groups; used whole-reference split"
        )
        group_for = lambda example: example.reference_id

    grouped: dict[str, list[ReferenceTrainingExample]] = {}
    for example in corpus.examples:
        grouped.setdefault(group_for(example), []).append(example)
    if len(grouped) < 2:
        raise ValueError("at least two independent split groups are required")
    ordered_groups = sorted(grouped, key=lambda value: _stable_order(value, seed))
    validation_group_count = max(
        1,
        min(len(ordered_groups) - 1, round(len(ordered_groups) * validation_fraction)),
    )
    validation_ids = set(ordered_groups[:validation_group_count])
    validation = tuple(
        sorted(
            (
                example
                for group_id in validation_ids
                for example in grouped[group_id]
            ),
            key=lambda example: example.reference_id,
        )
    )
    train = tuple(
        sorted(
            (
                example
                for group_id, examples in grouped.items()
                if group_id not in validation_ids
                for example in examples
            ),
            key=lambda example: example.reference_id,
        )
    )
    return ReferenceSplit(
        train=train,
        validation=validation,
        split_level=split_level,
        fallback_reason=fallback_reason,
        training_group_ids=tuple(
            sorted(set(ordered_groups) - validation_ids)
        ),
        validation_group_ids=tuple(sorted(validation_ids)),
    )


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if not len(positives) or not len(negatives):
        return None
    comparisons = positives[:, None] - negatives[None, :]
    return float(
        (np.sum(comparisons > 0.0) + 0.5 * np.sum(comparisons == 0.0))
        / comparisons.size
    )


def _binary_auprc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Return non-interpolated average precision (area under PR staircase)."""

    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives < 1 or negatives < 1:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    cumulative_positive = np.cumsum(ranked == 1)
    ranks = np.arange(1, len(ranked) + 1)
    precision = cumulative_positive / ranks
    return float(np.sum(precision[ranked == 1]) / positives)


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float | None:
    positive = labels == 1
    negative = labels == 0
    if not np.any(positive) or not np.any(negative):
        return None
    true_positive_rate = float(np.mean(predictions[positive] == 1))
    true_negative_rate = float(np.mean(predictions[negative] == 0))
    return 0.5 * (true_positive_rate + true_negative_rate)


def _json_float(value: Any) -> float | None:
    number = float(np.asarray(value).item())
    return number if math.isfinite(number) else None


def _categorical_label_coverage(labels: np.ndarray) -> dict[str, Any]:
    """Return class support/entropy so trivial heads cannot look learned."""

    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not values.size:
        return {
            "observed_class_count": 0,
            "label_entropy_bits": None,
            "label_histogram": {},
        }
    unique, counts = np.unique(values, return_counts=True)
    probabilities = counts.astype(np.float64) / counts.sum()
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    return {
        "observed_class_count": int(len(unique)),
        "label_entropy_bits": entropy,
        "label_histogram": {
            str(int(label)): int(count)
            for label, count in zip(unique, counts, strict=True)
        },
    }


def stage_edit_validation_selection_score(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one checkpoint using only predeclared validation evidence.

    AUROC and MRR keep the two sparse editorial heads from being sacrificed to
    easy or degenerate auxiliary losses.  Total loss remains present through a
    bounded inverse transform, so its scale cannot dominate the ranking.
    Missing cut/source evidence fails closed instead of silently becoming a
    favorable zero-loss checkpoint.
    """

    cut = metrics.get("cut")
    source = metrics.get("source")
    total_loss = metrics.get("total_loss")
    cut_auroc = (
        cut.get("auroc") if isinstance(cut, Mapping) else None
    )
    source_mrr = (
        source.get("mean_reciprocal_rank")
        if isinstance(source, Mapping)
        else None
    )
    values: list[float] = []
    for value in (cut_auroc, source_mrr, total_loss):
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = math.nan
        values.append(number)
    cut_number, source_number, total_number = values
    selectable = (
        math.isfinite(cut_number)
        and 0.0 <= cut_number <= 1.0
        and math.isfinite(source_number)
        and 0.0 <= source_number <= 1.0
        and math.isfinite(total_number)
        and total_number >= 0.0
    )
    bounded_loss_utility = (
        1.0 / (1.0 + total_number) if selectable else None
    )
    score = (
        STAGE_EDIT_VALIDATION_SELECTION_WEIGHTS["cut_auroc"] * cut_number
        + STAGE_EDIT_VALIDATION_SELECTION_WEIGHTS["source_mrr"] * source_number
        + STAGE_EDIT_VALIDATION_SELECTION_WEIGHTS[
            "bounded_inverse_total_loss"
        ]
        * float(bounded_loss_utility)
        if selectable
        else None
    )
    return {
        "contract": STAGE_EDIT_VALIDATION_SELECTION_CONTRACT,
        "selectable": selectable,
        "score": score,
        "weights": dict(STAGE_EDIT_VALIDATION_SELECTION_WEIGHTS),
        "components": {
            "cut_auroc": cut_number if math.isfinite(cut_number) else None,
            "source_mrr": (
                source_number if math.isfinite(source_number) else None
            ),
            "total_loss": (
                total_number if math.isfinite(total_number) else None
            ),
            "bounded_inverse_total_loss": bounded_loss_utility,
        },
        "formula": (
            "0.45*cut_auroc + 0.35*source_mrr + "
            "0.20/(1+total_loss)"
        ),
        "higher_is_better": True,
        "heldout_metrics_consumed": False,
    }


def stage_edit_source_evaluation_contract() -> dict[str, Any]:
    """Separate conditional source-head quality from decoded edit quality."""

    unsigned: dict[str, Any] = {
        "contract": STAGE_EDIT_SOURCE_EVALUATION_CONTRACT,
        "head_metrics": {
            "scope": "ground_truth_editorial_switch_tokens_only",
            "metrics": ["source_top1_accuracy", "source_mean_reciprocal_rank"],
            "interpretation": (
                "candidate ranking conditional on the reference cut time; not "
                "an end-to-end editing metric"
            ),
        },
        "end_to_end_decoder_metrics": {
            "scope": "top_ranked_duration_aware_semimarkov_timeline",
            "boundary_matching": {
                "algorithm": "one_to_one_minimum_absolute_time_error",
                "tolerance_seconds": 0.12,
            },
            "metrics": [
                "boundary_precision_recall_f1",
                "matched_boundary_source_accuracy",
                "joint_boundary_and_source_precision_recall_f1",
                "time_weighted_segment_source_accuracy",
                "source_macro_recall",
                "source_occupancy_distribution_jensen_shannon_bits",
                "same_time_pair_continuity_score_mean",
                "same_time_pair_continuity_hard_reset_count",
            ],
            "decoder_score_components_required": [
                "cut_timing",
                "source_selection",
                "duration",
                "transition",
                "same_canonical_time_pair_continuity",
            ],
            "pair_continuity_evidence": [
                "face_body_scale_center",
                "motion",
                "roll",
                "exposure_color_sharpness",
            ],
            "missing_pair_evidence_policy": "neutral",
            "fixed_pair_continuity_weight": 0.35,
            "validation_only_ablation_weight": 0.0,
        },
        "selection_policy": {
            "source_head_mrr_used_for_checkpoint_selection": True,
            "end_to_end_metrics_reported_separately": True,
            "development_song_may_not_change_checkpoint_or_threshold": True,
            "heldout_gate_lowering_allowed": False,
        },
    }
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        **unsigned,
        "contract_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def select_stage_edit_cut_threshold(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Choose a cut threshold on reference validation, never on held-out data."""

    label_values = np.asarray(labels, dtype=np.int32).reshape(-1)
    probability_values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if label_values.shape != probability_values.shape or not label_values.size:
        raise ValueError("cut threshold labels/probabilities must align and be nonempty")
    if not np.all(np.isin(label_values, (0, 1))):
        raise ValueError("cut threshold labels must be binary")
    if not np.all(np.isfinite(probability_values)) or np.any(
        (probability_values < 0.0) | (probability_values > 1.0)
    ):
        raise ValueError("cut threshold probabilities must be finite in [0, 1]")
    positive_count = int(np.sum(label_values == 1))
    negative_count = int(np.sum(label_values == 0))
    if positive_count < 1 or negative_count < 1:
        raise ValueError("cut threshold validation requires both classes")

    unique = np.unique(probability_values)
    above_maximum = float(np.nextafter(unique[-1], math.inf))
    candidates = np.unique(
        np.concatenate((unique, np.asarray((0.5, above_maximum))))
    )
    ranked: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        predictions = (probability_values >= threshold).astype(np.int32)
        balanced = _balanced_accuracy(label_values, predictions)
        assert balanced is not None
        # First maximize balanced accuracy, then prefer the value closest to
        # the conventional 0.5, then the higher/more conservative threshold.
        ranked.append(
            (
                float(balanced),
                -abs(float(threshold) - 0.5),
                float(threshold),
                float(threshold),
            )
        )
    best = max(ranked)
    selected_threshold = best[3]
    unsigned: dict[str, Any] = {
        "contract": STAGE_EDIT_CUT_THRESHOLD_SELECTION_CONTRACT,
        "selection_partition": (
            "whole_reference_validation_within_training_songs"
        ),
        "objective": "maximum_balanced_accuracy",
        "tie_break": ["closest_to_0_5", "higher_threshold"],
        "selected_probability_threshold": selected_threshold,
        "validation_balanced_accuracy": best[0],
        "validation_label_count": int(label_values.size),
        "validation_positive_count": positive_count,
        "validation_negative_count": negative_count,
        "candidate_threshold_count": int(len(candidates)),
        "heldout_labels_consumed": False,
        "heldout_probabilities_consumed": False,
    }
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        **unsigned,
        "contract_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def select_stage_edit_cut_threshold_from_model(
    model: StageEditTransformer,
    batch: Any,
    supervision: Any,
) -> dict[str, Any]:
    """Freeze a validation threshold from one concrete checkpoint forward."""

    _require_mlx()
    model.eval()
    output = model(batch)
    mx.eval(output.cut_timing_logits)
    cut_mask = np.asarray(supervision.cut_mask, dtype=bool)
    labels = np.asarray(supervision.cut_targets, dtype=np.int32)[cut_mask]
    logits = np.asarray(output.cut_timing_logits, dtype=np.float64)[cut_mask]
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    return select_stage_edit_cut_threshold(labels, probabilities)


def evaluate_stage_edit_model(
    model: StageEditTransformer,
    batch: Any,
    supervision: Any,
    *,
    loss_contract: StageEditLossContract,
    cut_probability_threshold: float = 0.5,
) -> dict[str, Any]:
    """Return mask-aware validation metrics for every StageEdit head."""

    _require_mlx()
    threshold = float(cut_probability_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("cut probability threshold must be finite in [0, 1]")
    model.eval()
    output = model(batch)
    total, components = stage_edit_loss(
        model,
        batch,
        supervision,
        loss_contract,
    )
    mx.eval(
        total,
        *components.values(),
        output.cut_timing_logits,
        output.source_selection_logits,
        output.duration_bucket_logits,
        output.transition_logits,
        output.crop,
    )

    cut_mask = np.asarray(supervision.cut_mask, dtype=bool)
    cut_labels = np.asarray(supervision.cut_targets, dtype=np.int32)[cut_mask]
    cut_logits = np.asarray(output.cut_timing_logits)[cut_mask]
    cut_probabilities = 1.0 / (1.0 + np.exp(-np.clip(cut_logits, -60.0, 60.0)))
    cut_predictions = (cut_probabilities >= threshold).astype(np.int32)
    cut_predictions_at_half = (cut_probabilities >= 0.5).astype(np.int32)

    source_mask = np.asarray(supervision.source_mask, dtype=bool)
    source_labels = np.asarray(supervision.source_targets, dtype=np.int32)[source_mask]
    source_logits = np.asarray(output.source_selection_logits)[source_mask]
    source_top1: float | None = None
    source_mrr: float | None = None
    if len(source_labels):
        source_top1 = float(np.mean(np.argmax(source_logits, axis=-1) == source_labels))
        reciprocal_ranks: list[float] = []
        for logits, label in zip(source_logits, source_labels, strict=True):
            ranking = np.argsort(-logits, kind="stable")
            rank = int(np.flatnonzero(ranking == label)[0]) + 1
            reciprocal_ranks.append(1.0 / rank)
        source_mrr = float(np.mean(reciprocal_ranks))

    duration_mask = np.asarray(supervision.duration_mask, dtype=bool)
    duration_labels = np.asarray(supervision.duration_targets, dtype=np.int32)[
        duration_mask
    ]
    duration_logits = np.asarray(output.duration_bucket_logits)[duration_mask]
    duration_accuracy = (
        float(np.mean(np.argmax(duration_logits, axis=-1) == duration_labels))
        if len(duration_labels)
        else None
    )

    transition_mask = np.asarray(supervision.transition_mask, dtype=bool)
    transition_labels = np.asarray(
        supervision.transition_targets, dtype=np.int32
    )[transition_mask]
    transition_logits = np.asarray(output.transition_logits)[transition_mask]
    transition_accuracy = (
        float(
            np.mean(np.argmax(transition_logits, axis=-1) == transition_labels)
        )
        if len(transition_labels)
        else None
    )
    crop_mask = np.asarray(supervision.crop_mask, dtype=bool)
    crop_targets = np.asarray(supervision.crop_targets, dtype=np.float64)
    crop_predictions = np.asarray(output.crop, dtype=np.float64)
    crop_count = int(crop_mask.sum())
    crop_names = ("zoom", "center_x", "center_y", "roll_degrees")
    crop_coordinate_metrics: dict[str, dict[str, float | int | None]] = {}
    for coordinate_index, coordinate_name in enumerate(crop_names):
        coordinate_mask = crop_mask[..., coordinate_index]
        coordinate_count = int(coordinate_mask.sum())
        if coordinate_count:
            residual = np.abs(
                crop_predictions[..., coordinate_index][coordinate_mask]
                - crop_targets[..., coordinate_index][coordinate_mask]
            )
            mae: float | None = float(np.mean(residual))
            p95_absolute_error: float | None = float(np.quantile(residual, 0.95))
        else:
            mae = None
            p95_absolute_error = None
        crop_coordinate_metrics[coordinate_name] = {
            "label_count": coordinate_count,
            "mean_absolute_error": mae,
            "p95_absolute_error": p95_absolute_error,
        }

    return {
        "total_loss": _json_float(total),
        "loss_components": {
            name: _json_float(value) for name, value in components.items()
        },
        "cut": {
            "label_count": int(len(cut_labels)),
            "positive_count": int(np.sum(cut_labels == 1)),
            "negative_count": int(np.sum(cut_labels == 0)),
            "auroc": _binary_auroc(cut_labels, cut_probabilities),
            "auprc_average_precision": _binary_auprc(
                cut_labels, cut_probabilities
            ),
            "balanced_accuracy_at_0_5": _balanced_accuracy(
                cut_labels, cut_predictions_at_half
            ),
            "probability_threshold": threshold,
            "balanced_accuracy_at_threshold": _balanced_accuracy(
                cut_labels, cut_predictions
            ),
        },
        "source": {
            "listwise_group_count": int(len(source_labels)),
            "top1_accuracy": source_top1,
            "mean_reciprocal_rank": source_mrr,
            "listwise_loss": _json_float(components["source_listwise"]),
        },
        "duration": {
            "label_count": int(len(duration_labels)),
            "accuracy": duration_accuracy,
            **_categorical_label_coverage(duration_labels),
        },
        "transition": {
            "label_count": int(len(transition_labels)),
            "accuracy": transition_accuracy,
            **_categorical_label_coverage(transition_labels),
            "status": (
                "evaluated" if len(transition_labels) else "unavailable_no_labels"
            ),
        },
        "crop": {
            "coordinate_label_count": crop_count,
            "coordinates": crop_coordinate_metrics,
            "status": (
                "evaluated" if crop_count else "masked_untrained_no_labels"
            ),
        },
    }


@dataclass(frozen=True, slots=True)
class SparseTrainingRunResult:
    checkpoint_directory: Path
    report_path: Path
    best_epoch: int
    best_training_step: int
    best_validation_metrics: Mapping[str, Any]
    report: Mapping[str, Any]


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return destination.resolve()


def train_sparse_stage_edit(
    corpus: ReferenceTrainingCorpus,
    *,
    checkpoint_directory: Path,
    report_path: Path,
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
    model_config: StageEditTransformerConfig | None = None,
    epochs: int = 40,
    early_stopping_patience: int = 8,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 1.0e-3,
    gradient_clip_norm: float = 1.0,
    validation_fraction: float = 0.25,
    seed: int = 20260823,
    device: Any = None,
    loss_contract: StageEditLossContract = StageEditLossContract(),
) -> SparseTrainingRunResult:
    """Train and save only checkpoints that improve held-out validation loss.

    Both output paths are mandatory.  Nothing is written to a repository cache
    or an implicit home-directory location.
    """

    _require_mlx()
    if epochs <= 0 or early_stopping_patience <= 0:
        raise ValueError("epochs and early_stopping_patience must be positive")
    for value, name in (
        (learning_rate, "learning_rate"),
        (weight_decay, "weight_decay"),
        (gradient_clip_norm, "gradient_clip_norm"),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    config = model_config or sparse_calibration_stage_edit_config(schema)
    if config.token_schema_fingerprint != schema.fingerprint:
        raise ValueError("model configuration and schema differ")
    maximum_length = max(len(example.sequence.tokens) for example in corpus.examples)
    if maximum_length > config.max_sequence_length:
        raise ValueError(
            f"adapted sequence length {maximum_length} exceeds model maximum "
            f"{config.max_sequence_length}"
        )
    production_supervision_gate = validate_stage_edit_production_supervision(
        corpus
    )

    split = deterministic_reference_split(
        corpus,
        seed=seed,
        validation_fraction=validation_fraction,
    )
    training_sequences = tuple(example.sequence for example in split.train)
    training_targets = tuple(example.targets for example in split.train)
    validation_sequences = tuple(
        example.sequence for example in split.validation
    )
    validation_targets = tuple(example.targets for example in split.validation)
    active_device = device if device is not None else mx.default_device()

    with mx.stream(active_device):
        mx.random.seed(seed)
        model = StageEditTransformer(config)
        train_batch = tensorize_stage_edit_batch(
            training_sequences, schema, config, device=active_device
        )
        train_supervision = tensorize_stage_edit_supervision(
            training_targets, train_batch, schema, device=active_device
        )
        validation_batch = tensorize_stage_edit_batch(
            validation_sequences, schema, config, device=active_device
        )
        validation_supervision = tensorize_stage_edit_supervision(
            validation_targets, validation_batch, schema, device=active_device
        )
        optimizer = optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        def loss_function(trained_model: StageEditTransformer) -> Any:
            return stage_edit_loss(
                trained_model,
                train_batch,
                train_supervision,
                loss_contract,
            )[0]

        loss_and_grad = nn.value_and_grad(model, loss_function)
        best_validation_loss = math.inf
        best_epoch = 0
        best_step = 0
        best_metrics: dict[str, Any] | None = None
        improvement_count = 0
        epochs_without_improvement = 0
        history: list[dict[str, Any]] = []
        optimizer_steps = 0
        checkpoint_root = Path(checkpoint_directory)
        clipping_supported = hasattr(optim, "clip_grad_norm")

        for epoch in range(1, epochs + 1):
            model.train()
            loss, gradients = loss_and_grad(model)
            if clipping_supported:
                gradients, gradient_norm = optim.clip_grad_norm(
                    gradients, gradient_clip_norm
                )
            else:  # Current pinned MLX supports clipping; retained for clarity.
                gradient_norm = mx.array(float("nan"))
            optimizer.update(model, gradients)
            optimizer_steps += 1
            mx.eval(loss, gradient_norm, model.parameters(), optimizer.state)

            train_metrics = evaluate_stage_edit_model(
                model,
                train_batch,
                train_supervision,
                loss_contract=loss_contract,
            )
            validation_metrics = evaluate_stage_edit_model(
                model,
                validation_batch,
                validation_supervision,
                loss_contract=loss_contract,
            )
            validation_loss = float(validation_metrics["total_loss"])
            improved = math.isfinite(validation_loss) and (
                validation_loss < best_validation_loss - 1.0e-8
            )
            history.append(
                {
                    "epoch": epoch,
                    "optimizer_step": optimizer_steps,
                    "train_total_loss": train_metrics["total_loss"],
                    "validation_total_loss": validation_loss,
                    "gradient_norm_before_clip": _json_float(gradient_norm),
                    "validation_improved": improved,
                }
            )
            if improved and optimizer_steps > 0:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_step = optimizer_steps
                best_metrics = validation_metrics
                improvement_count += 1
                epochs_without_improvement = 0
                metadata = new_stage_edit_checkpoint_metadata(
                    (
                        f"sparse-calibration-{corpus.data_id}-"
                        f"seed{seed}-step{optimizer_steps}"
                    ),
                    schema,
                    config,
                    training_steps=optimizer_steps,
                    weights_trained=True,
                    training_data_id=corpus.data_id,
                )
                save_stage_edit_checkpoint(checkpoint_root, model, metadata)
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break

    if best_metrics is None or best_step <= 0:
        raise RuntimeError("no finite validation improvement; no checkpoint was saved")

    # Prove the saved digest/metadata load successfully and report metrics for
    # the checkpoint rather than the possibly later in-memory epoch.
    with mx.stream(active_device):
        best_model, best_metadata = load_stage_edit_checkpoint(
            Path(checkpoint_directory), expected_schema=schema
        )
        best_validation_metrics = evaluate_stage_edit_model(
            best_model,
            validation_batch,
            validation_supervision,
            loss_contract=loss_contract,
        )

    crop_count = int(corpus.statistics["crop_coordinate_supervision_count"])
    transition_count = int(corpus.statistics["transition_supervision_count"])
    editorial_switch_count = int(
        corpus.statistics["editorial_source_switch_positive_count"]
    )
    source_supervision_count = int(
        corpus.statistics["source_supervision_count"]
    )
    duration_supervision_count = int(
        corpus.statistics["duration_supervision_count"]
    )
    report: dict[str, Any] = {
        "schema_version": TRAINING_REPORT_SCHEMA_VERSION,
        "run_kind": "stage_edit_sparse_offline_calibration",
        "created_at": datetime.now(UTC).isoformat(),
        "production_ready": False,
        "production_readiness_reason": (
            "sparse reference-only calibration cannot activate the production "
            "editor; handcrafted deterministic selection remains authoritative"
        ),
        "production_supervision_gate": production_supervision_gate.to_dict(),
        "checkpoint_scope": "offline_sparse_calibration_only",
        "data": {
            "data_id": corpus.data_id,
            "source_paths": list(corpus.source_paths),
            "statistics": dict(corpus.statistics),
        },
        "split": split.to_dict(),
        "model": {
            "config": config.to_dict(),
            "parameter_count": stage_edit_parameter_count(config),
            "mlx_version": mlx_runtime_version(),
            "device": str(active_device),
            "production_103m_config_preserved_elsewhere": True,
        },
        "optimization": {
            "seed": seed,
            "requested_epochs": epochs,
            "completed_epochs": len(history),
            "optimizer_steps": optimizer_steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gradient_clipping_supported": clipping_supported,
            "gradient_clip_norm": gradient_clip_norm,
            "early_stopping_patience": early_stopping_patience,
            "validation_improvement_count": improvement_count,
            "best_epoch": best_epoch,
            "best_training_step": best_step,
            "model_selection": "minimum held-out total validation loss",
        },
        "head_training_status": {
            "cut": (
                "trained_sparse_editorial_switch_labels"
                if editorial_switch_count
                else "negative_only_no_editorial_switch_labels"
            ),
            "source": (
                "trained_sparse_listwise_pairwise_editorial_switches"
                if source_supervision_count
                else "masked_untrained_no_editorial_switch_labels"
            ),
            "duration": (
                "trained_sparse_editorial_switch_durations"
                if duration_supervision_count
                else "masked_untrained_no_editorial_switch_labels"
            ),
            "transition": (
                "trained_sparse" if transition_count else "masked_untrained_no_labels"
            ),
            "crop": "trained_sparse" if crop_count else "masked_untrained_no_labels",
        },
        "best_validation_metrics": best_validation_metrics,
        "checkpoint": {
            "directory": str(Path(checkpoint_directory).resolve()),
            "checkpoint_id": best_metadata.checkpoint_id,
            "training_steps": best_metadata.training_steps,
            "weights_trained": best_metadata.weights_trained,
            "weight_filename": best_metadata.weight_filename,
            "weight_sha256": best_metadata.weight_sha256,
            "saved_only_after_validation_improvement": True,
            "production_activation_authorized": False,
            "production_supervision_gate_passed": (
                production_supervision_gate.passed
            ),
        },
        "history": history,
    }
    resolved_report = _write_json(Path(report_path), report)
    return SparseTrainingRunResult(
        checkpoint_directory=Path(checkpoint_directory).resolve(),
        report_path=resolved_report,
        best_epoch=best_epoch,
        best_training_step=best_step,
        best_validation_metrics=best_validation_metrics,
        report=report,
    )


__all__ = [
    "ProductionSupervisionGate",
    "ReferenceSplit",
    "SparseTrainingRunResult",
    "TRAINING_REPORT_SCHEMA_VERSION",
    "STAGE_EDIT_CUT_THRESHOLD_SELECTION_CONTRACT",
    "STAGE_EDIT_SOURCE_EVALUATION_CONTRACT",
    "STAGE_EDIT_VALIDATION_SELECTION_CONTRACT",
    "deterministic_reference_split",
    "evaluate_stage_edit_model",
    "require_stage_edit_production_supervision",
    "select_stage_edit_cut_threshold",
    "select_stage_edit_cut_threshold_from_model",
    "sparse_calibration_stage_edit_config",
    "stage_edit_source_evaluation_contract",
    "stage_edit_validation_selection_score",
    "train_sparse_stage_edit",
    "validate_stage_edit_production_supervision",
]
