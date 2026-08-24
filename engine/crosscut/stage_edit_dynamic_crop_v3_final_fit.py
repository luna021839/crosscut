"""Inactive all-reference Dynamic Crop v3 final fitting.

This module deliberately does not perform model, schedule, checkpoint, or
threshold selection.  Those decisions must be frozen by a complete
three-song LOSO release artifact before this function is called.  The final
fit then consumes all fifteen BANG BANG, BLACKHOLE, and REBEL HEART references
for a fixed number of optimizer steps and writes a checkpoint that cannot be
mistaken for an activated production bundle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from .stage_edit_dynamic_crop_capacity_probe import (
    PROBE_TRAINING_SCHEDULE,
    SELECTION_SONG_IDS,
)
from .stage_edit_dynamic_crop_capacity_fold_launcher import (
    MAXIMUM_CUT_POSITIVE_WEIGHT,
)
from .stage_edit_dynamic_crop_head_v3 import (
    DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
    DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
    DYNAMIC_CROP_V3_LOSS_CONTRACT,
    DYNAMIC_CROP_V3_MODEL_CONTRACT,
    DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
    DYNAMIC_CROP_V3_TARGET_CONTRACT,
    DynamicCropV3LossContract,
    StageEditTransformerDynamicCropV3,
    load_stage_edit_dynamic_crop_checkpoint_v3,
    new_stage_edit_dynamic_crop_checkpoint_metadata_v3,
    save_stage_edit_dynamic_crop_checkpoint_v3,
    stage_edit_dynamic_crop_v3_loss,
    stage_edit_dynamic_crop_v3_parameter_count,
    tensorize_stage_edit_dynamic_crop_supervision_v3,
)
from .stage_edit_dynamic_crop_training_v3 import (
    StageEditDynamicCropTrainingCorpusV3,
)
from .stage_edit_reference_adapter import ReferenceTrainingCorpus
from .stage_edit_training import require_stage_edit_production_supervision
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    StageEditLossContract,
    StageEditTransformerConfig,
    mlx_runtime_version,
    stage_edit_cut_imbalance_contract,
    stage_edit_loss,
    stage_edit_readiness_report,
    tensorize_stage_edit_batch,
    tensorize_stage_edit_supervision,
    valid_dynamic_crop_v3_activation_checkpointing_evidence,
)
from . import stage_edit_production_training as production


FINAL_FIT_REPORT_SCHEMA_VERSION = 1
FINAL_FIT_REPORT_FILENAME = "dynamic-crop-v3-final-fit-report.json"
FINAL_FIT_RUN_KIND = "inactive_dynamic_crop_v3_all_reference_final_fit"
FINAL_FIT_CONTRACT = (
    "loso_frozen_config_threshold_and_schedule_all_15_reference_refit_v1"
)
EXPECTED_REFERENCE_COUNT = 15
PRODUCTION_VOLUME_ROOT = Path("/Volumes/CrossCutMedia")


class DynamicCropV3FinalFitError(ValueError):
    """The frozen final-fit or output contract is invalid."""


@dataclass(frozen=True, slots=True)
class DynamicCropV3FinalFitResult:
    checkpoint_directory: Path
    report_path: Path
    checkpoint_id: str
    checkpoint_weight_sha256: str
    optimizer_steps: int
    report: Mapping[str, Any]


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _is_commit_oid(value: object) -> bool:
    if not isinstance(value, str) or len(value) not in (40, 64):
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _validate_schedule(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DynamicCropV3FinalFitError(
            "final-fit training schedule must be an object"
        )
    schedule = dict(value)
    if (
        set(schedule) != set(PROBE_TRAINING_SCHEDULE)
        or _canonical_sha256(schedule)
        != _canonical_sha256(dict(PROBE_TRAINING_SCHEDULE))
    ):
        raise DynamicCropV3FinalFitError(
            "final fit requires the exact LOSO training schedule"
        )
    return schedule


def _validate_complete_reference_corpus(
    corpus: ReferenceTrainingCorpus,
) -> tuple[tuple[Any, ...], dict[str, list[str]]]:
    examples = tuple(
        sorted(
            corpus.examples,
            key=lambda item: (str(item.song_id), item.reference_id),
        )
    )
    if len(examples) != EXPECTED_REFERENCE_COUNT:
        raise DynamicCropV3FinalFitError(
            "final fit requires exactly fifteen references"
        )
    by_song: dict[str, list[str]] = {}
    for example in examples:
        if example.song_id is None:
            raise DynamicCropV3FinalFitError(
                "final fit requires explicit song ids"
            )
        song_id = str(example.song_id)
        by_song.setdefault(song_id, []).append(example.reference_id)
    if set(by_song) != set(SELECTION_SONG_IDS) or any(
        len(reference_ids) != 5 for reference_ids in by_song.values()
    ):
        raise DynamicCropV3FinalFitError(
            "final fit requires five references for each fixed LOSO song"
        )
    reference_ids = [example.reference_id for example in examples]
    if len(reference_ids) != len(set(reference_ids)):
        raise DynamicCropV3FinalFitError(
            "final fit requires globally unique reference ids"
        )
    return examples, {
        song: sorted(reference_ids)
        for song, reference_ids in sorted(by_song.items())
    }


def _finite_probability(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicCropV3FinalFitError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise DynamicCropV3FinalFitError(
            f"{label} must be finite and strictly between zero and one"
        )
    return number


def _write_report(path: Path, payload: Mapping[str, Any]) -> Path:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise DynamicCropV3FinalFitError(
            f"refusing to overwrite final-fit report: {path}"
        ) from exc
    return path


def train_dynamic_crop_v3_final_fit(
    corpus: ReferenceTrainingCorpus,
    *,
    dynamic_crop_corpus: StageEditDynamicCropTrainingCorpusV3,
    checkpoint_directory: Path,
    model_config: StageEditTransformerConfig,
    base_loss_contract: StageEditLossContract,
    dynamic_crop_loss_contract: DynamicCropV3LossContract,
    frozen_training_schedule: Mapping[str, Any],
    frozen_cut_probability_threshold: float,
    frozen_quality_thresholds: Mapping[str, Any],
    release_freeze_path: Path,
    release_freeze_sha256: str,
    source_commit: str,
    device: Any = None,
) -> DynamicCropV3FinalFitResult:
    """Fit the selected v3 candidate on all references without reselection."""

    production._require_mlx()
    schedule = _validate_schedule(frozen_training_schedule)
    threshold = _finite_probability(
        frozen_cut_probability_threshold,
        "frozen cut probability threshold",
    )
    if not isinstance(frozen_quality_thresholds, Mapping) or not (
        frozen_quality_thresholds
    ):
        raise DynamicCropV3FinalFitError(
            "frozen held-out quality thresholds are required"
        )
    quality_thresholds = json.loads(
        json.dumps(
            dict(frozen_quality_thresholds),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    raw_freeze_path = Path(release_freeze_path).expanduser()
    if not raw_freeze_path.is_absolute():
        raise DynamicCropV3FinalFitError(
            "release freeze path must be absolute"
        )
    freeze_path = raw_freeze_path.resolve()
    if (
        not freeze_path.is_absolute()
        or not freeze_path.is_file()
        or freeze_path.is_symlink()
    ):
        raise DynamicCropV3FinalFitError(
            "release freeze must be an absolute regular non-symlink file"
        )
    actual_freeze_sha = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    if not _is_sha256(release_freeze_sha256) or (
        actual_freeze_sha != release_freeze_sha256
    ):
        raise DynamicCropV3FinalFitError("release freeze SHA-256 mismatch")
    if not _is_commit_oid(source_commit):
        raise DynamicCropV3FinalFitError(
            "source commit must be a lowercase Git object id"
        )
    if dynamic_crop_loss_contract.base != base_loss_contract:
        raise DynamicCropV3FinalFitError(
            "base and Dynamic Crop v3 loss contracts differ"
        )
    if dynamic_crop_loss_contract.contract != DYNAMIC_CROP_V3_LOSS_CONTRACT:
        raise DynamicCropV3FinalFitError("Dynamic Crop v3 loss contract differs")
    if model_config.transition_authority != base_loss_contract.transition_authority:
        raise DynamicCropV3FinalFitError(
            "model and loss transition authority differ"
        )
    if model_config.token_schema_fingerprint != DEFAULT_STAGE_EDIT_SCHEMA.fingerprint:
        raise DynamicCropV3FinalFitError(
            "frozen model configuration and token schema differ"
        )

    examples, references_by_song = _validate_complete_reference_corpus(corpus)
    steps_per_complete_epoch = math.ceil(
        len(examples) / int(schedule["batch_size"])
    )
    if (
        int(schedule["finetune_steps_planned"]) < steps_per_complete_epoch
        or int(schedule["finetune_steps_planned"]) % steps_per_complete_epoch
    ):
        raise DynamicCropV3FinalFitError(
            "frozen finetune steps must contain complete all-reference epochs"
        )
    sequence_ids = {example.sequence.sequence_id for example in examples}
    if set(dynamic_crop_corpus.sequences) != sequence_ids:
        raise DynamicCropV3FinalFitError(
            "Dynamic Crop v3 target sequences differ from all-reference corpus"
        )
    feature_gate = production._require_production_feature_snapshots(corpus)
    supervision_gate = require_stage_edit_production_supervision(corpus)
    if not supervision_gate.passed:
        raise DynamicCropV3FinalFitError(
            "all-reference production supervision gate failed"
        )
    maximum_length = max(len(example.sequence.tokens) for example in examples)
    maximum_sources = max(
        len(token.source_candidates)
        for example in examples
        for token in example.sequence.tokens
    )
    if maximum_length > model_config.max_sequence_length:
        raise DynamicCropV3FinalFitError(
            "all-reference sequence exceeds frozen model capacity"
        )
    if maximum_sources > model_config.max_sources:
        raise DynamicCropV3FinalFitError(
            "all-reference source count exceeds frozen model capacity"
        )

    checkpoint_root = Path(checkpoint_directory).expanduser().resolve()
    production_volume = PRODUCTION_VOLUME_ROOT.resolve()
    if (
        checkpoint_root != production_volume
        and production_volume not in checkpoint_root.parents
    ):
        raise DynamicCropV3FinalFitError(
            "final-fit checkpoint must be written under /Volumes/CrossCutMedia"
        )
    if checkpoint_root.exists() or checkpoint_root.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite final-fit checkpoint: {checkpoint_root}"
        )
    positive_count = sum(
        target.cut == 1.0
        for example in examples
        for target in example.targets.targets
        if target.cut is not None
    )
    negative_count = sum(
        target.cut == 0.0
        for example in examples
        for target in example.targets.targets
        if target.cut is not None
    )
    cut_imbalance = stage_edit_cut_imbalance_contract(
        positive_count=positive_count,
        negative_count=negative_count,
        maximum_positive_weight=MAXIMUM_CUT_POSITIVE_WEIGHT,
    )
    effective_base_loss = replace(
        base_loss_contract,
        cut_positive_weight=float(cut_imbalance["applied_positive_weight"]),
    )
    effective_dynamic_loss = replace(
        dynamic_crop_loss_contract,
        base=effective_base_loss,
    )
    if (
        effective_base_loss.source_listwise_weight <= 0.0
        or effective_base_loss.source_pairwise_weight <= 0.0
    ):
        raise DynamicCropV3FinalFitError(
            "source pretraining requires both frozen source objectives"
        )

    active_device = device if device is not None else production.mx.default_device()
    optimizer_steps = 0
    cut_steps = 0
    source_steps = 0
    finetune_steps = 0
    cut_history: list[dict[str, Any]] = []
    source_history: list[dict[str, Any]] = []
    finetune_history: list[dict[str, Any]] = []
    clipping_supported = hasattr(production.optim, "clip_grad_norm")

    def dynamic_sequences_for(batch_examples: Sequence[Any]) -> tuple[Any, ...]:
        return tuple(
            dynamic_crop_corpus.sequence_for(example.sequence.sequence_id)
            for example in batch_examples
        )

    with production.mx.stream(active_device):
        production.mx.random.seed(int(schedule["seed"]))
        model = StageEditTransformerDynamicCropV3(model_config)
        activation_checkpointing = (
            production._enable_dynamic_crop_v3_activation_checkpointing(model)
        )

        cut_contract = production._cut_only_pretraining_loss_contract(
            effective_base_loss
        )
        positive_windows, negative_windows = production._balanced_cut_windows(
            examples,
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
            window_length=int(schedule["cut_pretraining_window_length"]),
        )
        cut_schedule = production._warmup_cosine_schedule(
            peak_learning_rate=float(
                schedule["cut_pretraining_peak_learning_rate"]
            ),
            total_steps=int(schedule["cut_pretraining_steps"]),
            warmup_steps=int(schedule["cut_pretraining_warmup_steps"]),
            minimum_learning_rate_ratio=float(
                schedule["minimum_learning_rate_ratio"]
            ),
        )
        cut_optimizer = production.optim.AdamW(
            learning_rate=cut_schedule,
            weight_decay=float(schedule["weight_decay"]),
        )
        if int(cut_optimizer.state["step"].item()) != 0:
            raise RuntimeError("fresh final-fit cut optimizer did not start at zero")

        def cut_loss(trained_model: Any, batch: Any, supervision: Any) -> Any:
            return stage_edit_loss(
                trained_model, batch, supervision, cut_contract
            )[0]

        cut_loss_and_grad = production.nn.value_and_grad(model, cut_loss)
        chooser = random.Random(int(schedule["seed"]) ^ 0x5A17E)
        for phase_step in range(1, int(schedule["cut_pretraining_steps"]) + 1):
            windows = (
                positive_windows[chooser.randrange(len(positive_windows))],
                negative_windows[chooser.randrange(len(negative_windows))],
            )
            batch = tensorize_stage_edit_batch(
                tuple(window.sequence for window in windows),
                DEFAULT_STAGE_EDIT_SCHEMA,
                model_config,
                device=active_device,
            )
            supervision = tensorize_stage_edit_supervision(
                tuple(window.targets for window in windows),
                batch,
                DEFAULT_STAGE_EDIT_SCHEMA,
                device=active_device,
            )
            model.train()
            loss, gradients = cut_loss_and_grad(model, batch, supervision)
            if clipping_supported:
                gradients, gradient_norm = production.optim.clip_grad_norm(
                    gradients, float(schedule["gradient_clip_norm"])
                )
            else:
                gradient_norm = production.mx.array(float("nan"))
            cut_optimizer.update(model, gradients)
            optimizer_steps += 1
            cut_steps += 1
            production.mx.eval(
                loss, gradient_norm, model.parameters(), cut_optimizer.state
            )
            if phase_step == 1 or phase_step % 25 == 0:
                cut_history.append(
                    {
                        "phase_step": phase_step,
                        "optimizer_step": optimizer_steps,
                        "loss": float(loss.item()),
                        "gradient_norm_before_clip": float(
                            gradient_norm.item()
                        ),
                    }
                )
        if int(cut_optimizer.state["step"].item()) != cut_steps:
            raise RuntimeError("final-fit cut optimizer step differs")

        source_contract = production._source_only_pretraining_loss_contract(
            effective_base_loss
        )
        inventory = production._source_pretraining_inventory(
            examples,
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
            window_length=int(schedule["source_pretraining_window_length"]),
        )
        source_sampler = production._HierarchicalSourceWindowSampler(
            inventory.windows,
            seed=int(schedule["seed"]) ^ 0x50A7CE,
        )
        source_schedule = production._warmup_cosine_schedule(
            peak_learning_rate=float(
                schedule["source_pretraining_peak_learning_rate"]
            ),
            total_steps=int(schedule["source_pretraining_steps"]),
            warmup_steps=int(schedule["source_pretraining_warmup_steps"]),
            minimum_learning_rate_ratio=float(
                schedule["minimum_learning_rate_ratio"]
            ),
        )
        source_optimizer = production.optim.AdamW(
            learning_rate=source_schedule,
            weight_decay=float(schedule["weight_decay"]),
        )
        if int(source_optimizer.state["step"].item()) != 0:
            raise RuntimeError(
                "fresh final-fit source optimizer did not start at zero"
            )

        def source_loss(trained_model: Any, batch: Any, supervision: Any) -> Any:
            return stage_edit_loss(
                trained_model, batch, supervision, source_contract
            )[0]

        source_loss_and_grad = production.nn.value_and_grad(model, source_loss)
        for phase_step in range(1, int(schedule["source_pretraining_steps"]) + 1):
            window = source_sampler.sample()
            batch = tensorize_stage_edit_batch(
                (window.sequence,),
                DEFAULT_STAGE_EDIT_SCHEMA,
                model_config,
                device=active_device,
            )
            supervision = tensorize_stage_edit_supervision(
                (window.targets,),
                batch,
                DEFAULT_STAGE_EDIT_SCHEMA,
                device=active_device,
            )
            model.train()
            loss, gradients = source_loss_and_grad(model, batch, supervision)
            if clipping_supported:
                gradients, gradient_norm = production.optim.clip_grad_norm(
                    gradients, float(schedule["gradient_clip_norm"])
                )
            else:
                gradient_norm = production.mx.array(float("nan"))
            source_optimizer.update(model, gradients)
            optimizer_steps += 1
            source_steps += 1
            production.mx.eval(
                loss,
                gradient_norm,
                model.parameters(),
                source_optimizer.state,
            )
            if phase_step == 1 or phase_step % 25 == 0:
                source_history.append(
                    {
                        "phase_step": phase_step,
                        "optimizer_step": optimizer_steps,
                        "song_id": window.song_id,
                        "reference_id": window.reference_id,
                        "group_id": window.group_id,
                        "loss": float(loss.item()),
                        "gradient_norm_before_clip": float(
                            gradient_norm.item()
                        ),
                    }
                )
        if int(source_optimizer.state["step"].item()) != source_steps:
            raise RuntimeError("final-fit source optimizer step differs")

        planned_finetune_steps = int(schedule["finetune_steps_planned"])
        warmup_steps = min(
            int(schedule["finetune_warmup_steps"]),
            planned_finetune_steps - 1,
        )
        joint_schedule = production._warmup_cosine_schedule(
            peak_learning_rate=float(schedule["learning_rate"]),
            total_steps=planned_finetune_steps,
            warmup_steps=warmup_steps,
            minimum_learning_rate_ratio=float(
                schedule["minimum_learning_rate_ratio"]
            ),
        )
        joint_optimizer = production.optim.AdamW(
            learning_rate=joint_schedule,
            weight_decay=float(schedule["weight_decay"]),
        )
        if int(joint_optimizer.state["step"].item()) != 0:
            raise RuntimeError(
                "fresh final-fit joint optimizer did not start at zero"
            )

        def joint_loss(
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
                effective_dynamic_loss,
            )[0]

        joint_loss_and_grad = production.nn.value_and_grad(model, joint_loss)
        epoch = 0
        while finetune_steps < planned_finetune_steps:
            epoch += 1
            losses: list[float] = []
            gradient_norms: list[float] = []
            ordered = production._stable_epoch_order(
                examples,
                seed=int(schedule["seed"]),
                epoch=epoch,
            )
            for batch_examples in production._batches(
                ordered, int(schedule["batch_size"])
            ):
                if finetune_steps >= planned_finetune_steps:
                    break
                batch = tensorize_stage_edit_batch(
                    production.prepare_stage_edit_production_input_sequences(
                        batch_examples, schema=DEFAULT_STAGE_EDIT_SCHEMA
                    ),
                    DEFAULT_STAGE_EDIT_SCHEMA,
                    model_config,
                    device=active_device,
                )
                supervision = tensorize_stage_edit_supervision(
                    tuple(item.targets for item in batch_examples),
                    batch,
                    DEFAULT_STAGE_EDIT_SCHEMA,
                    device=active_device,
                )
                dynamic_supervision = (
                    tensorize_stage_edit_dynamic_crop_supervision_v3(
                        dynamic_sequences_for(batch_examples),
                        batch,
                        device=active_device,
                    )
                )
                model.train()
                loss, gradients = joint_loss_and_grad(
                    model, batch, supervision, dynamic_supervision
                )
                if clipping_supported:
                    gradients, gradient_norm = production.optim.clip_grad_norm(
                        gradients, float(schedule["gradient_clip_norm"])
                    )
                else:
                    gradient_norm = production.mx.array(float("nan"))
                joint_optimizer.update(model, gradients)
                optimizer_steps += 1
                finetune_steps += 1
                production.mx.eval(
                    loss,
                    gradient_norm,
                    model.parameters(),
                    joint_optimizer.state,
                )
                losses.append(float(loss.item()))
                gradient_norms.append(float(gradient_norm.item()))
            finetune_history.append(
                {
                    "epoch": epoch,
                    "finetune_optimizer_step": finetune_steps,
                    "training_loss_mean": sum(losses) / len(losses),
                    "gradient_norm_mean_before_clip": (
                        sum(gradient_norms) / len(gradient_norms)
                    ),
                    "checkpoint_selection_performed": False,
                    "validation_labels_consumed": False,
                }
            )
        if int(joint_optimizer.state["step"].item()) != finetune_steps:
            raise RuntimeError("final-fit joint optimizer step differs")

        checkpoint_id = (
            f"v3-final-fit-{corpus.data_id[:16]}-"
            f"{release_freeze_sha256[:12]}-step{optimizer_steps}"
        )
        metadata = new_stage_edit_dynamic_crop_checkpoint_metadata_v3(
            checkpoint_id,
            DEFAULT_STAGE_EDIT_SCHEMA,
            model_config,
            training_steps=optimizer_steps,
            weights_trained=True,
            training_data_id=(
                f"{corpus.data_id}+{dynamic_crop_corpus.data_id}"
            ),
        )
        saved_metadata = save_stage_edit_dynamic_crop_checkpoint_v3(
            checkpoint_root, model, metadata
        )

    with production.mx.stream(active_device):
        reloaded_model, loaded_metadata = (
            load_stage_edit_dynamic_crop_checkpoint_v3(
                checkpoint_root,
                expected_schema=DEFAULT_STAGE_EDIT_SCHEMA,
            )
        )
        production.mx.eval(reloaded_model.parameters())
    if loaded_metadata != saved_metadata:
        raise RuntimeError("reloaded final-fit metadata differs from saved metadata")
    encoder = getattr(getattr(reloaded_model, "base", None), "encoder", None)
    if encoder is None or getattr(encoder, "checkpoint", None) is not False:
        raise RuntimeError(
            "reloaded final-fit inference model retained training checkpointing"
        )

    report: dict[str, Any] = {
        "schema_version": FINAL_FIT_REPORT_SCHEMA_VERSION,
        "run_kind": FINAL_FIT_RUN_KIND,
        "contract": FINAL_FIT_CONTRACT,
        "created_at": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "release_freeze": {
            "path": str(freeze_path),
            "file_sha256": release_freeze_sha256,
        },
        "dynamic_crop_version": 3,
        "model_contract": DYNAMIC_CROP_V3_MODEL_CONTRACT,
        "target_contract": DYNAMIC_CROP_V3_TARGET_CONTRACT,
        "output_contract": DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
        "loss_contract": DYNAMIC_CROP_V3_LOSS_CONTRACT,
        "all_reference_fit": {
            "song_ids": list(SELECTION_SONG_IDS),
            "reference_count": len(examples),
            "references_by_song": references_by_song,
            "all_references_consumed_by_optimizer": True,
            "reference_validation_partition_created": False,
            "held_out_partition_created": False,
            "final_fit_data_used_for_candidate_selection": False,
            "final_fit_data_used_for_schedule_selection": False,
            "final_fit_data_used_for_threshold_selection": False,
        },
        "frozen_decisions": {
            "candidate_config": model_config.to_dict(),
            "base_loss_contract": asdict(base_loss_contract),
            "dynamic_crop_loss_contract": asdict(dynamic_crop_loss_contract),
            "training_schedule": schedule,
            "cut_probability_threshold": threshold,
            "quality_thresholds": quality_thresholds,
            "decisions_frozen_before_final_fit": True,
        },
        "training": {
            "optimizer_steps": optimizer_steps,
            "cut_pretraining_optimizer_steps": cut_steps,
            "source_pretraining_optimizer_steps": source_steps,
            "finetune_optimizer_steps": finetune_steps,
            "checkpoint_selection_performed": False,
            "validation_metrics_computed": False,
            "heldout_metrics_computed": False,
            "effective_base_loss_contract": asdict(effective_base_loss),
            "effective_dynamic_crop_loss_contract": asdict(
                effective_dynamic_loss
            ),
            "cut_imbalance_contract": cut_imbalance,
            "activation_checkpointing": dict(activation_checkpointing),
            "feature_snapshot_gate": feature_gate,
            "supervision_gate": supervision_gate.to_dict(),
            "cut_pretraining_history": cut_history,
            "source_pretraining_history": source_history,
            "finetune_history": finetune_history,
        },
        "checkpoint": {
            "directory": str(checkpoint_root),
            "metadata_filename": DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
            "weight_filename": DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
            "checkpoint_id": loaded_metadata.checkpoint_id,
            "training_steps": loaded_metadata.training_steps,
            "weight_sha256": loaded_metadata.weight_sha256,
            "model_config": model_config.to_dict(),
            "parameter_count": stage_edit_dynamic_crop_v3_parameter_count(
                model_config
            ),
            "mlx_version": mlx_runtime_version(),
            "device": str(active_device),
            "inference_activation_checkpointing_enabled": False,
        },
        "authority": {
            "production_ready": False,
            "production_registry_activation_authorized": False,
            "production_registry_activation_requested": False,
            "learned_source_pair_authority_present": False,
            "blind_holdout_quality_claim_authorized": False,
            "xoxz_dynamic_crop_v3_blind_evidence_present": False,
            "attitude_dynamic_crop_v3_blind_evidence_present": False,
            "human_parity_claim_authorized": False,
            "experimental_unseen_ive_inference_only": True,
        },
    }
    report_path = _write_report(
        checkpoint_root / FINAL_FIT_REPORT_FILENAME, report
    )
    readiness = stage_edit_readiness_report(checkpoint_root)
    if readiness.learned_inference_ready:
        raise RuntimeError(
            "inactive final-fit checkpoint unexpectedly activated production inference"
        )
    return DynamicCropV3FinalFitResult(
        checkpoint_directory=checkpoint_root,
        report_path=report_path,
        checkpoint_id=loaded_metadata.checkpoint_id,
        checkpoint_weight_sha256=loaded_metadata.weight_sha256,
        optimizer_steps=optimizer_steps,
        report=report,
    )


_FINAL_FIT_REPORT_KEYS = {
    "schema_version",
    "run_kind",
    "contract",
    "created_at",
    "source_commit",
    "release_freeze",
    "dynamic_crop_version",
    "model_contract",
    "target_contract",
    "output_contract",
    "loss_contract",
    "all_reference_fit",
    "frozen_decisions",
    "training",
    "checkpoint",
    "authority",
}


def load_dynamic_crop_v3_final_fit_report(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise DynamicCropV3FinalFitError(
            "final-fit report must be a regular non-symlink file"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DynamicCropV3FinalFitError("final-fit report must be an object")
    if set(payload) != _FINAL_FIT_REPORT_KEYS:
        raise DynamicCropV3FinalFitError(
            "final-fit report keys are not exact"
        )
    if (
        payload.get("schema_version") != FINAL_FIT_REPORT_SCHEMA_VERSION
        or payload.get("run_kind") != FINAL_FIT_RUN_KIND
        or payload.get("contract") != FINAL_FIT_CONTRACT
        or payload.get("dynamic_crop_version") != 3
        or payload.get("model_contract") != DYNAMIC_CROP_V3_MODEL_CONTRACT
        or payload.get("target_contract") != DYNAMIC_CROP_V3_TARGET_CONTRACT
        or payload.get("output_contract") != DYNAMIC_CROP_V3_OUTPUT_CONTRACT
        or payload.get("loss_contract") != DYNAMIC_CROP_V3_LOSS_CONTRACT
    ):
        raise DynamicCropV3FinalFitError("final-fit report contract differs")
    authority = payload.get("authority")
    all_reference = payload.get("all_reference_fit")
    frozen = payload.get("frozen_decisions")
    expected_authority = {
        "production_ready": False,
        "production_registry_activation_authorized": False,
        "production_registry_activation_requested": False,
        "learned_source_pair_authority_present": False,
        "blind_holdout_quality_claim_authorized": False,
        "xoxz_dynamic_crop_v3_blind_evidence_present": False,
        "attitude_dynamic_crop_v3_blind_evidence_present": False,
        "human_parity_claim_authorized": False,
        "experimental_unseen_ive_inference_only": True,
    }
    if (
        not isinstance(authority, Mapping)
        or set(authority) != set(expected_authority)
        or any(
        authority.get(key) is not expected
        for key, expected in expected_authority.items()
        )
    ):
        raise DynamicCropV3FinalFitError("final-fit authority is not inactive")
    all_reference_keys = {
        "song_ids",
        "reference_count",
        "references_by_song",
        "all_references_consumed_by_optimizer",
        "reference_validation_partition_created",
        "held_out_partition_created",
        "final_fit_data_used_for_candidate_selection",
        "final_fit_data_used_for_schedule_selection",
        "final_fit_data_used_for_threshold_selection",
    }
    if (
        not isinstance(all_reference, Mapping)
        or set(all_reference) != all_reference_keys
        or all_reference.get("song_ids") != list(SELECTION_SONG_IDS)
        or all_reference.get("reference_count") != EXPECTED_REFERENCE_COUNT
        or all_reference.get("all_references_consumed_by_optimizer") is not True
        or all_reference.get("reference_validation_partition_created") is not False
        or all_reference.get("held_out_partition_created") is not False
        or all_reference.get("final_fit_data_used_for_candidate_selection") is not False
        or all_reference.get("final_fit_data_used_for_schedule_selection") is not False
        or all_reference.get("final_fit_data_used_for_threshold_selection") is not False
    ):
        raise DynamicCropV3FinalFitError("final-fit reference scope differs")
    references_by_song = all_reference.get("references_by_song")
    if not isinstance(references_by_song, Mapping) or set(
        references_by_song
    ) != set(SELECTION_SONG_IDS):
        raise DynamicCropV3FinalFitError(
            "final-fit references-by-song scope differs"
        )
    flattened_references: list[str] = []
    for song in SELECTION_SONG_IDS:
        references = references_by_song.get(song)
        if (
            not isinstance(references, list)
            or len(references) != 5
            or any(not isinstance(item, str) or not item for item in references)
        ):
            raise DynamicCropV3FinalFitError(
                "final-fit song does not contain five reference ids"
            )
        flattened_references.extend(references)
    if len(set(flattened_references)) != EXPECTED_REFERENCE_COUNT:
        raise DynamicCropV3FinalFitError(
            "final-fit reference ids are not globally unique"
        )
    frozen_keys = {
        "candidate_config",
        "base_loss_contract",
        "dynamic_crop_loss_contract",
        "training_schedule",
        "cut_probability_threshold",
        "quality_thresholds",
        "decisions_frozen_before_final_fit",
    }
    if (
        not isinstance(frozen, Mapping)
        or set(frozen) != frozen_keys
        or _canonical_sha256(frozen.get("training_schedule"))
        != _canonical_sha256(dict(PROBE_TRAINING_SCHEDULE))
        or not isinstance(frozen.get("quality_thresholds"), Mapping)
        or not frozen.get("quality_thresholds")
        or frozen.get("decisions_frozen_before_final_fit") is not True
    ):
        raise DynamicCropV3FinalFitError("final-fit decisions were not frozen")
    _finite_probability(
        frozen.get("cut_probability_threshold"),
        "reported frozen cut probability threshold",
    )
    freeze = payload.get("release_freeze")
    if (
        not isinstance(freeze, Mapping)
        or set(freeze) != {"path", "file_sha256"}
        or not isinstance(freeze.get("path"), str)
        or not Path(str(freeze.get("path"))).is_absolute()
        or not _is_sha256(freeze.get("file_sha256"))
        or not _is_commit_oid(payload.get("source_commit"))
    ):
        raise DynamicCropV3FinalFitError(
            "final-fit freeze or source identity differs"
        )
    training = payload.get("training")
    checkpoint = payload.get("checkpoint")
    if not isinstance(training, Mapping) or not isinstance(checkpoint, Mapping):
        raise DynamicCropV3FinalFitError(
            "final-fit training or checkpoint evidence is missing"
        )
    if not valid_dynamic_crop_v3_activation_checkpointing_evidence(
        training.get("activation_checkpointing")
    ):
        raise DynamicCropV3FinalFitError(
            "final-fit activation checkpointing evidence differs"
        )
    cut_steps = PROBE_TRAINING_SCHEDULE["cut_pretraining_steps"]
    source_steps = PROBE_TRAINING_SCHEDULE["source_pretraining_steps"]
    finetune_steps = PROBE_TRAINING_SCHEDULE["finetune_steps_planned"]
    if (
        training.get("cut_pretraining_optimizer_steps") != cut_steps
        or training.get("source_pretraining_optimizer_steps") != source_steps
        or training.get("finetune_optimizer_steps") != finetune_steps
        or training.get("optimizer_steps")
        != cut_steps + source_steps + finetune_steps
        or training.get("checkpoint_selection_performed") is not False
        or training.get("validation_metrics_computed") is not False
        or training.get("heldout_metrics_computed") is not False
        or checkpoint.get("metadata_filename")
        != DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
        or checkpoint.get("weight_filename")
        != DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
        or checkpoint.get("training_steps") != training.get("optimizer_steps")
        or checkpoint.get("inference_activation_checkpointing_enabled") is not False
    ):
        raise DynamicCropV3FinalFitError(
            "final-fit optimizer or checkpoint contract differs"
        )
    return payload


__all__ = [
    "DynamicCropV3FinalFitError",
    "DynamicCropV3FinalFitResult",
    "FINAL_FIT_CONTRACT",
    "FINAL_FIT_REPORT_FILENAME",
    "FINAL_FIT_REPORT_SCHEMA_VERSION",
    "FINAL_FIT_RUN_KIND",
    "load_dynamic_crop_v3_final_fit_report",
    "train_dynamic_crop_v3_final_fit",
]
