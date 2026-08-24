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
from .stage_edit_dynamic_crop_training import (
    StageEditDynamicCropTrainingCorpusV2,
)
from .stage_edit_reference_adapter import (
    ReferenceTrainingCorpus,
    ReferenceTrainingExample,
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
except ImportError as exc:  # pragma: no cover - non-Apple portability.
    mx = None
    nn = None
    optim = None
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


def _require_mlx() -> None:
    if mx is None or nn is None or optim is None:
        detail = f": {_MLX_IMPORT_ERROR}" if _MLX_IMPORT_ERROR else ""
        raise RuntimeError(f"production StageEdit training requires MLX{detail}")


def _finite_metric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _require_unambiguous_checkpoint_namespace(
    checkpoint_root: Path,
    *,
    dynamic_crop_v2_enabled: bool,
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
    dynamic_files = (
        checkpoint_root / DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME,
        checkpoint_root / DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME,
    )
    incompatible = legacy_files if dynamic_crop_v2_enabled else dynamic_files
    present = tuple(path.name for path in incompatible if path.exists())
    if present:
        requested = "Dynamic Crop v2" if dynamic_crop_v2_enabled else "legacy"
        raise ValueError(
            f"cannot train {requested} checkpoint into a directory containing "
            f"the other model format: {', '.join(present)}"
        )


def _require_production_feature_snapshots(
    corpus: ReferenceTrainingCorpus,
) -> dict[str, Any]:
    """Require the exact train/serve feature materializer for every reference."""

    hashes: dict[str, str] = {}
    missing: list[str] = []
    for example in corpus.examples:
        digest = str(example.feature_snapshot_sha256 or "").strip()
        try:
            valid_digest = len(digest) == 64 and int(digest, 16) >= 0
        except ValueError:
            valid_digest = False
        if (
            not valid_digest
            or example.statistics.get("materialized_feature_snapshot_used")
            is not True
            or example.statistics.get("shared_feature_materializer_used")
            is not True
        ):
            missing.append(example.reference_id)
            continue
        hashes[example.reference_id] = digest
    if missing or len(hashes) != len(corpus.examples):
        raise ValueError(
            "production StageEdit requires a verified causal feature snapshot "
            "for every reference; missing=" + ", ".join(sorted(missing))
        )
    if len(hashes) != len(set(hashes)):
        raise ValueError("production feature snapshot reference ids are not unique")
    return {
        "passed": True,
        "contract": PRODUCTION_FEATURE_SNAPSHOT_GATE_CONTRACT,
        "reference_count": len(hashes),
        "feature_snapshot_sha256_by_reference": dict(sorted(hashes.items())),
        "reference_match_or_current_choice_features_present": False,
        "missing_reference_ids": [],
    }


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
    thresholds: ProductionQualityThresholds = ProductionQualityThresholds(),
    transition_authority: str = LEARNED_TRANSITION_AUTHORITY,
) -> ProductionQualityGate:
    """Fail closed unless every renderer-active head generalizes unseen.

    Dynamic Crop v2 checkpoints always carry the legacy four-value crop head
    because they share the v1 trunk/output namespace, but v2 decoding never
    consumes that head: an ineligible trajectory means identity geometry, not
    a static-head fallback.  Therefore the legacy crop metric remains in the
    report as auxiliary evidence and is activation-critical only when no v2
    trajectory metrics were supplied.  The stricter K=8 trajectory, timing,
    presence, motion, and real renderer-decode gates remain mandatory for v2.
    """

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
    if not dynamic_crop_v2_active:
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
    trajectory = (
        dynamic_crop_v2_metrics.get("trajectory")
        if isinstance(dynamic_crop_v2_metrics, Mapping)
        else None
    )
    if not isinstance(trajectory, Mapping):
        reasons.append("dynamic_crop_v2_metrics_missing")
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

    return tuple(
        mask_stage_edit_recent_history_for_semimarkov(example.sequence, schema)
        for example in examples
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
    early_stopping_patience: int = 6,
    batch_size: int = 2,
    learning_rate: float = 1.0e-4,
    weight_decay: float = 1.0e-3,
    gradient_clip_norm: float = 1.0,
    maximum_cut_positive_weight: float = 8.0,
    cut_pretrain_steps: int = 500,
    cut_pretrain_learning_rate: float = 3.0e-4,
    cut_pretrain_window_length: int = 32,
    cut_pretrain_warmup_steps: int = 50,
    finetune_warmup_steps: int = 8,
    minimum_learning_rate_ratio: float = 0.10,
    seed: int = 20260824,
    device: Any = None,
    loss_contract: StageEditLossContract = StageEditLossContract(),
    dynamic_crop_corpus: StageEditDynamicCropTrainingCorpusV2 | None = None,
    dynamic_crop_loss_contract: DynamicCropV2LossContract = (
        DynamicCropV2LossContract()
    ),
    quality_thresholds: ProductionQualityThresholds = ProductionQualityThresholds(),
    authorize_production_activation: bool = False,
) -> ProductionTrainingRunResult:
    """Train a real long-context model and evaluate its exact saved checkpoint."""

    _require_mlx()
    if epochs < 1 or early_stopping_patience < 1 or batch_size < 1:
        raise ValueError("epochs, patience, and batch size must be positive")
    if cut_pretrain_steps not in (0,) and cut_pretrain_steps < 500:
        raise ValueError("cut pretraining must be disabled or run for >= 500 steps")
    if cut_pretrain_window_length < 2:
        raise ValueError("cut pretraining window length must be >= 2")
    if cut_pretrain_steps and not 0 <= cut_pretrain_warmup_steps < cut_pretrain_steps:
        raise ValueError("cut pretraining warmup must be shorter than pretraining")
    if finetune_warmup_steps < 0:
        raise ValueError("finetune warmup cannot be negative")
    for value, name, allow_zero in (
        (learning_rate, "learning_rate", False),
        (cut_pretrain_learning_rate, "cut_pretrain_learning_rate", False),
        (weight_decay, "weight_decay", True),
        (gradient_clip_norm, "gradient_clip_norm", False),
    ):
        if (
            not math.isfinite(value)
            or value < 0.0
            or (not allow_zero and value == 0.0)
        ):
            raise ValueError(f"{name} is invalid")

    split = production_song_split(
        corpus,
        held_out_song_ids=held_out_song_ids,
        required_training_song_ids=required_training_song_ids,
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
    effective_dynamic_crop_loss_contract = replace(
        dynamic_crop_loss_contract,
        base=replace(
            dynamic_crop_loss_contract.base,
            cut_positive_weight=float(
                cut_imbalance_contract["applied_positive_weight"]
            ),
        ),
    )
    feature_snapshot_gate = _require_production_feature_snapshots(corpus)
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
    if (
        config.transition_authority
        != effective_loss_contract.transition_authority
        or config.transition_authority
        != effective_dynamic_crop_loss_contract.base.transition_authority
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

    dynamic_crop_v2_enabled = dynamic_crop_corpus is not None
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
        dynamic_crop_v2_enabled=dynamic_crop_v2_enabled,
    )
    report_path = checkpoint_root / PRODUCTION_TRAINING_REPORT_FILENAME
    active_device = device if device is not None else mx.default_device()
    history: list[dict[str, Any]] = []
    cut_pretraining_history: list[dict[str, Any]] = []
    optimizer_steps = 0
    cut_pretraining_optimizer_steps = 0
    best_epoch = 0
    best_step = 0
    best_validation_loss = math.inf
    best_validation_selection_score = -math.inf
    best_validation_selection_evidence: Mapping[str, Any] | None = None
    epochs_without_improvement = 0
    saved_metadata = None
    clipping_supported = hasattr(optim, "clip_grad_norm")

    with mx.stream(active_device):
        mx.random.seed(seed)
        model = (
            StageEditTransformerDynamicCropV2(config)
            if dynamic_crop_v2_enabled
            else StageEditTransformer(config)
        )

        cut_only_loss_contract = replace(
            effective_loss_contract,
            cut_positive_weight=1.0,
            source_listwise_weight=0.0,
            source_pairwise_weight=0.0,
            duration_weight=0.0,
            transition_weight=0.0,
            crop_weight=0.0,
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

        finetune_steps_planned = epochs * len(_batches(split.train, batch_size))
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

        if dynamic_crop_v2_enabled:
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
                    effective_dynamic_crop_loss_contract,
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
            tensorize_stage_edit_dynamic_crop_supervision_v2(
                dynamic_sequences_for(split.validation),
                validation_batch,
                device=active_device,
            )
            if dynamic_crop_v2_enabled
            else None
        )

        for epoch in range(1, epochs + 1):
            epoch_losses: list[float] = []
            epoch_gradient_norms: list[float] = []
            model.train()
            ordered = _stable_epoch_order(split.train, seed=seed, epoch=epoch)
            for examples in _batches(ordered, batch_size):
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
                    tensorize_stage_edit_dynamic_crop_supervision_v2(
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
                if clipping_supported:
                    gradients, gradient_norm = optim.clip_grad_norm(
                        gradients, gradient_clip_norm
                    )
                else:  # Pinned production MLX supports clipping.
                    gradient_norm = mx.array(float("nan"))
                optimizer.update(model, gradients)
                optimizer_steps += 1
                mx.eval(loss, gradient_norm, model.parameters(), optimizer.state)
                epoch_losses.append(float(loss.item()))
                epoch_gradient_norms.append(float(gradient_norm.item()))

            validation_base_metrics = evaluate_stage_edit_model(
                model,
                validation_batch,
                validation_supervision,
                loss_contract=effective_loss_contract,
            )
            if dynamic_crop_v2_enabled:
                validation_dynamic_metrics = evaluate_stage_edit_dynamic_crop_v2(
                    model,
                    validation_batch,
                    validation_supervision,
                    validation_dynamic_supervision,
                    contract=effective_dynamic_crop_loss_contract,
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
            validation_selection_score = _finite_metric(
                validation_selection.get("score")
            )
            improved = bool(
                validation_selection.get("selectable")
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
                    "training_song_reference_validation_improved": improved,
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
                epochs_without_improvement = 0
                checkpoint_id = (
                    f"production-{corpus.data_id[:16]}-seed{seed}-"
                    f"step{optimizer_steps}"
                )
                if dynamic_crop_v2_enabled:
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
                epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break

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
        if dynamic_crop_v2_enabled:
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
        if dynamic_crop_v2_enabled:
            selected_validation_dynamic_metrics = (
                evaluate_stage_edit_dynamic_crop_v2(
                    best_model,
                    validation_batch,
                    validation_supervision,
                    validation_dynamic_supervision,
                    contract=effective_dynamic_crop_loss_contract,
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
                "validation_selection": dict(
                    best_validation_selection_evidence
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
        held_out_dynamic_metrics = None
        if dynamic_crop_v2_enabled:
            held_out_dynamic_supervision = (
                tensorize_stage_edit_dynamic_crop_supervision_v2(
                    dynamic_sequences_for(split.held_out),
                    held_out_batch,
                    device=active_device,
                )
            )
            held_out_dynamic_metrics = evaluate_stage_edit_dynamic_crop_v2(
                best_model,
                held_out_batch,
                held_out_supervision,
                held_out_dynamic_supervision,
                contract=effective_dynamic_crop_loss_contract,
            )
    quality_gate = evaluate_stage_edit_production_quality(
        held_out_metrics,
        dynamic_crop_v2_metrics=held_out_dynamic_metrics,
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
            if dynamic_crop_corpus is not None
            else None
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
                if dynamic_crop_corpus is not None
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
                stage_edit_dynamic_crop_v2_parameter_count(config)
                if dynamic_crop_v2_enabled
                else stage_edit_parameter_count(config)
            ),
            "mlx_version": mlx_runtime_version(),
            "device": str(active_device),
            "architecture": (
                "causal_long_context_transformer_with_k8_dynamic_crop_v2"
                if dynamic_crop_v2_enabled
                else "causal_long_context_multi_head_transformer_legacy_crop"
            ),
            "dynamic_crop_v2_enabled": dynamic_crop_v2_enabled,
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
            "multi_head_finetuning_optimizer_steps": (
                optimizer_steps - cut_pretraining_optimizer_steps
            ),
                "learning_rate": learning_rate,
            "learning_rate_schedule": {
                "contract": "independent_warmup_cosine_pretrain_and_finetune_v1",
                "cut_pretraining_steps": cut_pretrain_steps,
                "cut_pretraining_peak_learning_rate": (
                    cut_pretrain_learning_rate
                ),
                "cut_pretraining_window_length": cut_pretrain_window_length,
                "cut_pretraining_warmup_steps": (
                    cut_pretrain_warmup_steps if cut_pretrain_steps else 0
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
                "positive_weight": 1.0,
                "minimum_step_contract": 500,
                "validation_labels_consumed": False,
                "heldout_labels_consumed": False,
            },
            "weight_decay": weight_decay,
            "gradient_clip_norm": gradient_clip_norm,
            "effective_loss_contract": asdict(effective_loss_contract),
            "effective_dynamic_crop_loss_contract": (
                asdict(effective_dynamic_crop_loss_contract)
                if dynamic_crop_v2_enabled
                else None
            ),
            "gradient_clipping_supported": clipping_supported,
            "early_stopping_patience": early_stopping_patience,
            "best_epoch": best_epoch,
            "best_training_step": best_step,
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
            "model_selection": STAGE_EDIT_VALIDATION_SELECTION_CONTRACT,
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
            "dynamic_crop_v2_metrics": held_out_dynamic_metrics,
            "quality_gate": quality_gate.to_dict(),
        },
        "cut_pretraining_history": cut_pretraining_history,
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
