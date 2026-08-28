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

import numpy as np

from .stage_edit_dynamic_crop_capacity_probe import (
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
    renderer_safe_values_from_raw_v3,
    save_stage_edit_dynamic_crop_checkpoint_v3,
    stage_edit_dynamic_crop_v3_loss,
    stage_edit_dynamic_crop_v3_parameter_count,
    tensorize_stage_edit_dynamic_crop_supervision_v3,
)
from .stage_edit_dynamic_crop_v3_capacity_smoke import (
    V3_PROBE_TRAINING_SCHEDULE as PROBE_TRAINING_SCHEDULE,
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
    frozen_dynamic_crop_v3_prior_candidate_step: int,
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
    prior_candidate_step = frozen_dynamic_crop_v3_prior_candidate_step
    if (
        isinstance(prior_candidate_step, bool)
        or prior_candidate_step
        not in production.DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS
    ):
        raise DynamicCropV3FinalFitError(
            "frozen Dynamic Crop v3 prior candidate milestone differs"
        )
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
    prior_steps = 0
    finetune_steps = 0
    cut_history: list[dict[str, Any]] = []
    source_history: list[dict[str, Any]] = []
    prior_history: list[dict[str, Any]] = []
    finetune_history: list[dict[str, Any]] = []
    prior_evidence: dict[str, Any] | None = None
    retention_restore_steps = 0
    retention_optimizer_state_final_step = 0
    released_joint_optimizer_initial_step = 0
    retention_boundary_head_digests: dict[str, str] | None = None
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

        prior_inventory = production._dynamic_crop_v3_prior_inventory(
            examples, dynamic_crop_corpus
        )
        prior_inventory = production._materialize_dynamic_crop_v3_prior_features(
            model,
            examples,
            prior_inventory,
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
            config=model_config,
            device=active_device,
        )
        prior_sampler = production._HierarchicalDynamicCropV3PriorSampler(
            prior_inventory.tracks,
            seed=int(schedule["seed"]) ^ 0xD1A5C0,
        )
        values_stop = 8 * 4
        before_prior_digests = (
            production._dynamic_crop_v3_prior_parameter_digests(model)
        )
        initial_head_module = production._DynamicCropV3PriorHeadModule(
            model.dynamic_crop_context_head.weight,
            model.dynamic_crop_context_head.bias,
            model.dynamic_crop_source_head.weight,
        )
        initial_prior_snapshot = production._dynamic_crop_v3_prior_head_snapshot(
            initial_head_module
        )
        projection_rank = min(
            production.DYNAMIC_CROP_V3_PRIOR_PROJECTION_RANK,
            int(model.dynamic_crop_context_head.weight.shape[-1]),
        )
        (
            prior_context_projection,
            prior_source_projection,
        ) = production._dynamic_crop_v3_structured_prior_projections(
            int(model.dynamic_crop_context_head.weight.shape[-1]),
            seed=int(schedule["seed"]),
            requested_rank=projection_rank,
        )
        prior_projection_normalizer = (
            production._dynamic_crop_v3_structured_prior_projection_normalizer(
                prior_inventory,
                prior_context_projection,
                prior_source_projection,
            )
        )
        prior_adapter = production._DynamicCropV3StructuredPriorAdapterModule(
            projection_rank
        )
        initial_values_mx = renderer_safe_values_from_raw_v3(
            initial_prior_snapshot["context_bias"][:values_stop].reshape((8, 4))
        )
        production.mx.eval(initial_values_mx)
        component_scale = production._dynamic_crop_v3_prior_component_scale_evidence(
            prior_inventory,
            np.asarray(initial_values_mx, dtype=np.float64),
            effective_dynamic_loss,
        )
        class_balance = production._dynamic_crop_v3_prior_class_balance_evidence(
            prior_inventory
        )
        prior_schedule = production._warmup_cosine_schedule(
            peak_learning_rate=float(
                schedule[
                    "dynamic_crop_v3_prior_pretraining_peak_learning_rate"
                ]
            ),
            total_steps=int(
                schedule["dynamic_crop_v3_prior_pretraining_steps"]
            ),
            warmup_steps=int(
                schedule[
                    "dynamic_crop_v3_prior_pretraining_warmup_steps"
                ]
            ),
            minimum_learning_rate_ratio=float(
                schedule["minimum_learning_rate_ratio"]
            ),
        )
        prior_optimizer = production.optim.AdamW(
            learning_rate=prior_schedule,
            weight_decay=0.0,
        )
        if int(prior_optimizer.state["step"].item()) != 0:
            raise RuntimeError(
                "fresh final-fit v3 prior optimizer did not start at zero"
            )

        def prior_loss(
            trained_adapter: Any,
            context_embedding: Any,
            source_embedding: Any,
            target_values: Any,
            target_times: Any,
            count_target: int,
            motion_target: float,
            motion_supervised: bool,
        ) -> Any:
            raw = production._dynamic_crop_v3_structured_prior_raw(
                trained_adapter,
                initial_prior_snapshot,
                prior_context_projection,
                prior_source_projection,
                prior_projection_normalizer,
                context_embedding,
                source_embedding,
            )
            return production._dynamic_crop_v3_prior_loss_from_raw(
                raw,
                target_values,
                target_times,
                count_target,
                motion_target,
                motion_supervised,
                effective_dynamic_loss,
                class_balance["count_class_weights_1_to_8"][count_target - 1],
                class_balance["motion_class_weights_static_dynamic"][
                    int(motion_target)
                ],
            )[0]

        prior_loss_and_grad = production.nn.value_and_grad(
            prior_adapter, prior_loss
        )
        prior_candidate_snapshots: dict[int, Any] = {}
        prior_candidate_value_digests: dict[int, str] = {}
        prior_candidate_head_digests: dict[int, str] = {}
        for phase_step in range(
            1,
            int(schedule["dynamic_crop_v3_prior_pretraining_steps"]) + 1,
        ):
            track = prior_sampler.sample()
            if (
                track.context_embedding is None
                or track.source_embedding is None
            ):
                raise RuntimeError(
                    "final-fit v3 prior sampled an unmaterialized feature"
                )
            context_embedding = production.mx.array(track.context_embedding)
            source_embedding = production.mx.array(track.source_embedding)
            target_values = production.mx.array(
                np.asarray(track.values, dtype=np.float32)
            )
            target_times = production.mx.array(
                np.asarray(track.normalized_times, dtype=np.float32)
            )
            count_target = len(track.values)
            motion_target = float(track.motion_label == "dynamic")
            motion_supervised = not track.motion_label_loss_masked
            loss, gradients = prior_loss_and_grad(
                prior_adapter,
                context_embedding,
                source_embedding,
                target_values,
                target_times,
                count_target,
                motion_target,
                motion_supervised,
            )
            if clipping_supported:
                gradients, gradient_norm = production.optim.clip_grad_norm(
                    gradients, float(schedule["gradient_clip_norm"])
                )
            else:
                gradient_norm = production.mx.array(float("nan"))
            prior_optimizer.update(prior_adapter, gradients)
            updated_raw = production._dynamic_crop_v3_structured_prior_raw(
                prior_adapter,
                initial_prior_snapshot,
                prior_context_projection,
                prior_source_projection,
                prior_projection_normalizer,
                context_embedding,
                source_embedding,
            )
            _, prior_components = production._dynamic_crop_v3_prior_loss_from_raw(
                updated_raw,
                target_values,
                target_times,
                count_target,
                motion_target,
                motion_supervised,
                effective_dynamic_loss,
                class_balance["count_class_weights_1_to_8"][count_target - 1],
                class_balance["motion_class_weights_static_dynamic"][
                    int(motion_target)
                ],
            )
            optimizer_steps += 1
            prior_steps += 1
            production.mx.eval(
                loss,
                gradient_norm,
                *prior_components.values(),
                prior_adapter.parameters(),
                prior_optimizer.state,
            )
            candidate_value_digest = None
            candidate_head_digest = None
            if phase_step in production.DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS:
                snapshot = (
                    production._dynamic_crop_v3_structured_prior_head_snapshot(
                        prior_adapter,
                        initial_prior_snapshot,
                        prior_context_projection,
                        prior_source_projection,
                        prior_projection_normalizer,
                    )
                )
                prior_candidate_snapshots[phase_step] = snapshot
                candidate_value_digest = (
                    production._dynamic_crop_v3_prior_value_bias_digest(
                        snapshot
                    )
                )
                candidate_head_digest = production._parameter_tree_sha256(
                    snapshot
                )
                prior_candidate_value_digests[
                    phase_step
                ] = candidate_value_digest
                prior_candidate_head_digests[
                    phase_step
                ] = candidate_head_digest
            prior_history.append(
                {
                    "phase_step": phase_step,
                    "optimizer_step": optimizer_steps,
                    "actual_learning_rate": float(
                        prior_optimizer.learning_rate.item()
                    ),
                    "song_id": track.song_id,
                    "reference_id": track.reference_id,
                    "sequence_id": track.sequence_id,
                    "token_index": track.token_index,
                    "source_id": track.source_id,
                    "keyframe_count": len(track.values),
                    "gate_aligned_normalized_coordinate_mae": float(
                        prior_components["values_normalized"].item()
                    ),
                    "prior_total_loss": float(
                        prior_components["total"].item()
                    ),
                    "prior_count_loss": float(
                        prior_components["count"].item()
                    ),
                    "prior_time_loss": float(
                        prior_components["times"].item()
                    ),
                    "prior_motion_loss": float(
                        prior_components["motion"].item()
                    ),
                    "motion_label_supervised": motion_supervised,
                    "gradient_norm_before_clip": float(
                        gradient_norm.item()
                    ),
                    "checkpoint_selection_performed": False,
                    "predeclared_candidate_snapshot": (
                        phase_step
                        in production.DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS
                    ),
                    "candidate_value_bias_sha256": (
                        candidate_value_digest
                    ),
                    "candidate_head_parameter_sha256": (
                        candidate_head_digest
                    ),
                    "validation_labels_consumed": False,
                    "heldout_labels_consumed": False,
                }
            )
        if int(prior_optimizer.state["step"].item()) != prior_steps:
            raise RuntimeError("final-fit v3 prior optimizer step differs")
        if set(prior_candidate_snapshots) != set(
            production.DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS
        ):
            raise RuntimeError("final-fit v3 prior candidate snapshots differ")
        selected_prior_snapshot = prior_candidate_snapshots[
            prior_candidate_step
        ]
        production._assign_dynamic_crop_v3_prior_head_snapshot(
            model, selected_prior_snapshot
        )
        after_prior_digests = (
            production._dynamic_crop_v3_prior_parameter_digests(model)
        )
        invariant_keys = (
            "base_parameter_tree_sha256",
            "context_value_bias_sha256",
            "context_value_weight_sha256",
            "source_value_weight_sha256",
        )
        invariant = all(
            before_prior_digests[key] == after_prior_digests[key]
            for key in invariant_keys
        )
        value_changed = (
            before_prior_digests["context_value_bias_sha256"]
            != after_prior_digests["context_value_bias_sha256"]
        )
        non_value_bias_changed = (
            before_prior_digests["context_non_value_bias_sha256"]
            != after_prior_digests["context_non_value_bias_sha256"]
        )
        context_weight_changed = (
            before_prior_digests["context_weight_sha256"]
            != after_prior_digests["context_weight_sha256"]
        )
        source_head_changed = (
            before_prior_digests["source_head_parameter_tree_sha256"]
            != after_prior_digests["source_head_parameter_tree_sha256"]
        )
        context_value_weight_changed = (
            before_prior_digests["context_value_weight_sha256"]
            != after_prior_digests["context_value_weight_sha256"]
        )
        context_structured_weight_changed = (
            before_prior_digests["context_structured_weight_sha256"]
            != after_prior_digests["context_structured_weight_sha256"]
        )
        source_value_weight_changed = (
            before_prior_digests["source_value_weight_sha256"]
            != after_prior_digests["source_value_weight_sha256"]
        )
        source_structured_weight_changed = (
            before_prior_digests["source_structured_weight_sha256"]
            != after_prior_digests["source_structured_weight_sha256"]
        )
        dynamic_head_changed = (
            before_prior_digests["dynamic_head_parameter_tree_sha256"]
            != after_prior_digests["dynamic_head_parameter_tree_sha256"]
        )
        if not invariant or not dynamic_head_changed:
            raise RuntimeError("final-fit v3 prior parameter scope drifted")
        if (
            after_prior_digests["context_value_bias_sha256"]
            != prior_candidate_value_digests[prior_candidate_step]
            or after_prior_digests["dynamic_head_parameter_tree_sha256"]
            != prior_candidate_head_digests[prior_candidate_step]
        ):
            raise RuntimeError("final-fit frozen v3 prior digest drifted")
        prior_evidence = {
            "contract": (
                "all_reference_predeclared_balanced_standardized_structured_"
                "low_rank_adapter_milestones_frozen_loso_candidate_no_"
                "reselection_v4"
            ),
            "partition": "all_references_fixed_after_loso_selection",
            "objective": (
                "detached_low_rank_context_and_source_conditioned_count_"
                "times_motion_residual_with_frozen_values_v3"
            ),
            "feature_contract": (
                production.DYNAMIC_CROP_V3_PRIOR_FEATURE_CONTRACT
            ),
            "mutable_parameter_scope": (
                production.DYNAMIC_CROP_V3_PRIOR_PARAMETER_SCOPE_CONTRACT
            ),
            "optimizer_weight_decay": 0.0,
            "optimizer_initial_step": 0,
            "optimizer_state_final_step": prior_steps,
            "checkpoint_selection_performed": False,
            "validation_metrics_computed": False,
            "heldout_metrics_computed": False,
            "predeclared_candidate_steps": list(
                production.DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS
            ),
            "candidate_selection_contract": (
                production.DYNAMIC_CROP_V3_PRIOR_CANDIDATE_SELECTION_CONTRACT
            ),
            "candidate_snapshots": [
                {
                    "candidate_step": step,
                    "value_bias_sha256": (
                        prior_candidate_value_digests[step]
                    ),
                    "head_parameter_sha256": (
                        prior_candidate_head_digests[step]
                    ),
                    "training_only_updates": True,
                }
                for step in production.DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS
            ],
            "frozen_selected_candidate_step": prior_candidate_step,
            "frozen_selected_candidate_value_bias_sha256": (
                prior_candidate_value_digests[prior_candidate_step]
            ),
            "frozen_selected_candidate_head_parameter_sha256": (
                prior_candidate_head_digests[prior_candidate_step]
            ),
            "final_fit_data_used_for_candidate_selection": False,
            "sampling": {
                "contract": (
                    "deterministic_song_reference_trajectory_round_robin_v1"
                ),
                "feature_contract": (
                    production.DYNAMIC_CROP_V3_PRIOR_FEATURE_CONTRACT
                ),
                "adapter_contract": (
                    production.DYNAMIC_CROP_V3_PRIOR_ADAPTER_CONTRACT
                ),
                "projection_rank": projection_rank,
                "trainable_parameter_count": int(
                    sum(
                        np.asarray(value).size
                        for _, value in production.tree_flatten(
                            prior_adapter.parameters()
                        )
                    )
                ),
                "context_projection_sha256": (
                    production._parameter_tree_sha256(
                        {"projection": prior_context_projection}
                    )
                ),
                "source_projection_sha256": (
                    production._parameter_tree_sha256(
                        {"projection": prior_source_projection}
                    )
                ),
                "projection_normalization": dict(
                    prior_projection_normalizer.evidence
                ),
                "detached_feature_content_sha256": (
                    prior_inventory.detached_feature_content_sha256
                ),
                "per_song_sampled_counts": (
                    prior_sampler.per_song_sampled_counts
                ),
                "per_reference_sampled_counts": (
                    prior_sampler.per_reference_sampled_counts
                ),
                "reference_song_ids": dict(
                    prior_inventory.reference_song_ids
                ),
                "per_reference_track_counts": dict(
                    prior_inventory.per_reference_track_counts
                ),
                "references_without_tracks": list(
                    prior_inventory.references_without_tracks
                ),
            },
            "component_scale": component_scale,
            "class_balance": class_balance,
            "parameter_invariance": {
                "before": before_prior_digests,
                "after": after_prior_digests,
                "invariant_parameter_groups": list(invariant_keys),
                "all_declared_invariant_groups_byte_identical": invariant,
                "dynamic_head_parameter_tree_changed": dynamic_head_changed,
                "context_value_bias_changed": value_changed,
                "context_non_value_bias_changed": non_value_bias_changed,
                "context_weight_changed": context_weight_changed,
                "source_head_changed": source_head_changed,
                "context_value_weight_changed": (
                    context_value_weight_changed
                ),
                "context_structured_weight_changed": (
                    context_structured_weight_changed
                ),
                "source_value_weight_changed": source_value_weight_changed,
                "source_structured_weight_changed": (
                    source_structured_weight_changed
                ),
                "context_value_bias_row_count": values_stop,
                "base_parameter_gradients_computed": False,
                "source_head_gradients_computed": True,
                "context_weight_gradients_computed": True,
                "count_time_motion_bias_gradients_computed": True,
                "value_row_gradients_computed": False,
                "full_head_gradients_computed": False,
                "structured_low_rank_adapter_gradients_computed": True,
            },
        }
        retained_context_weight = production.mx.array(
            np.array(model.dynamic_crop_context_head.weight, copy=True)
        )
        retained_context_bias = production.mx.array(
            np.array(model.dynamic_crop_context_head.bias, copy=True)
        )
        retained_source_weight = production.mx.array(
            np.array(model.dynamic_crop_source_head.weight, copy=True)
        )

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
        retention_steps = int(
            schedule["dynamic_crop_v3_prior_retention_finetune_steps"]
        )
        joint_optimizer_reset = False
        while finetune_steps < planned_finetune_steps:
            epoch += 1
            losses: list[float] = []
            gradient_norms: list[float] = []
            epoch_retention_steps: list[int] = []
            epoch_phase_local_optimizer_steps: list[int] = []
            epoch_optimizer_reset_before_steps: list[int] = []
            epoch_actual_learning_rates: list[float] = []
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
                if finetune_steps == retention_steps and not joint_optimizer_reset:
                    retention_optimizer_state_final_step = int(
                        joint_optimizer.state["step"].item()
                    )
                    if retention_optimizer_state_final_step != retention_steps:
                        raise RuntimeError(
                            "final-fit prior retention optimizer step differs"
                        )
                    boundary_digests = (
                        production._dynamic_crop_v3_prior_parameter_digests(
                            model
                        )
                    )
                    retained_head_keys = (
                        "context_weight_sha256",
                        "source_head_parameter_tree_sha256",
                        "context_non_value_bias_sha256",
                        "context_value_bias_sha256",
                    )
                    retention_boundary_head_digests = {
                        key: boundary_digests[key]
                        for key in retained_head_keys
                    }
                    if any(
                        retention_boundary_head_digests[key]
                        != after_prior_digests[key]
                        for key in retained_head_keys
                    ):
                        raise RuntimeError(
                            "final-fit v3 prior drifted before joint release"
                        )
                    joint_optimizer = production.optim.AdamW(
                        learning_rate=production._offset_learning_rate_schedule(
                            joint_schedule, retention_steps
                        ),
                        weight_decay=float(schedule["weight_decay"]),
                    )
                    released_joint_optimizer_initial_step = int(
                        joint_optimizer.state["step"].item()
                    )
                    if released_joint_optimizer_initial_step != 0:
                        raise RuntimeError(
                            "final-fit released joint optimizer is not fresh"
                        )
                    joint_optimizer_reset = True
                    epoch_optimizer_reset_before_steps.append(
                        finetune_steps + 1
                    )
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
                next_finetune_step = finetune_steps + 1
                retain_prior = next_finetune_step <= retention_steps
                if retain_prior:
                    if not isinstance(gradients, Mapping):
                        raise RuntimeError(
                            "final-fit v3 gradients are not name-addressable"
                        )
                    gradients = dict(gradients)
                    for head_name in (
                        "dynamic_crop_context_head",
                        "dynamic_crop_source_head",
                    ):
                        gradients[head_name] = production._zero_gradient_tree(
                            gradients[head_name]
                        )
                if clipping_supported:
                    gradients, gradient_norm = production.optim.clip_grad_norm(
                        gradients, float(schedule["gradient_clip_norm"])
                    )
                else:
                    gradient_norm = production.mx.array(float("nan"))
                joint_optimizer.update(model, gradients)
                if retain_prior:
                    model.dynamic_crop_context_head.weight = (
                        retained_context_weight
                    )
                    model.dynamic_crop_context_head.bias = retained_context_bias
                    model.dynamic_crop_source_head.weight = retained_source_weight
                    retention_restore_steps += 1
                    epoch_retention_steps.append(next_finetune_step)
                optimizer_steps += 1
                finetune_steps += 1
                production.mx.eval(
                    loss,
                    gradient_norm,
                    model.parameters(),
                    joint_optimizer.state,
                )
                epoch_phase_local_optimizer_steps.append(
                    int(joint_optimizer.state["step"].item())
                )
                epoch_actual_learning_rates.append(
                    float(joint_optimizer.learning_rate.item())
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
                    "dynamic_crop_v3_prior_retention_finetune_steps": (
                        epoch_retention_steps
                    ),
                    "phase_local_optimizer_steps": (
                        epoch_phase_local_optimizer_steps
                    ),
                    "optimizer_reset_before_finetune_steps": (
                        epoch_optimizer_reset_before_steps
                    ),
                    "actual_learning_rates": epoch_actual_learning_rates,
                    "checkpoint_selection_performed": False,
                    "validation_labels_consumed": False,
                }
            )
        expected_active_joint_steps = finetune_steps - retention_steps
        if (
            not joint_optimizer_reset
            or released_joint_optimizer_initial_step != 0
            or int(joint_optimizer.state["step"].item())
            != expected_active_joint_steps
            or retention_restore_steps != retention_steps
            or retention_boundary_head_digests is None
        ):
            raise RuntimeError("final-fit joint optimizer step differs")
        retained_head_digests = (
            production._dynamic_crop_v3_prior_parameter_digests(model)
        )
        # The released joint phase may now change the heads; the exact retention
        # proof is the restored step count plus the fresh optimizer boundary.
        assert prior_evidence is not None
        prior_evidence = {
            **prior_evidence,
            "retention": {
                "retention_finetune_step_count": retention_restore_steps,
                "retention_optimizer_state_final_step": (
                    retention_optimizer_state_final_step
                ),
                "released_joint_optimizer_initial_step": (
                    released_joint_optimizer_initial_step
                ),
                "released_joint_optimizer_state_final_step": int(
                    joint_optimizer.state["step"].item()
                ),
                "optimizer_state_reset_before_dynamic_head_release": True,
                "released_joint_optimizer_schedule_offset_steps": (
                    retention_steps
                ),
                "warmup_head_digests": {
                    key: after_prior_digests[key]
                    for key in (
                        "context_weight_sha256",
                        "source_head_parameter_tree_sha256",
                        "context_non_value_bias_sha256",
                        "context_value_bias_sha256",
                    )
                },
                "retention_boundary_head_digests": (
                    retention_boundary_head_digests
                ),
                "all_dynamic_head_parameters_byte_identical_at_release": True,
                "post_joint_head_digests": {
                    key: retained_head_digests[key]
                    for key in (
                        "context_weight_sha256",
                        "source_head_parameter_tree_sha256",
                        "context_non_value_bias_sha256",
                        "context_value_bias_sha256",
                    )
                },
            },
        }

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
            "dynamic_crop_v3_prior_candidate_step": prior_candidate_step,
            "cut_probability_threshold": threshold,
            "quality_thresholds": quality_thresholds,
            "decisions_frozen_before_final_fit": True,
        },
        "training": {
            "optimizer_steps": optimizer_steps,
            "cut_pretraining_optimizer_steps": cut_steps,
            "source_pretraining_optimizer_steps": source_steps,
            "dynamic_crop_v3_prior_pretraining_optimizer_steps": prior_steps,
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
            "dynamic_crop_v3_prior_pretraining": prior_evidence,
            "dynamic_crop_v3_prior_pretraining_history": prior_history,
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
        "dynamic_crop_v3_prior_candidate_step",
        "cut_probability_threshold",
        "quality_thresholds",
        "decisions_frozen_before_final_fit",
    }
    if (
        not isinstance(frozen, Mapping)
        or set(frozen) != frozen_keys
        or _canonical_sha256(frozen.get("training_schedule"))
        != _canonical_sha256(dict(PROBE_TRAINING_SCHEDULE))
        or isinstance(
            frozen.get("dynamic_crop_v3_prior_candidate_step"), bool
        )
        or frozen.get("dynamic_crop_v3_prior_candidate_step")
        not in production.DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS
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
    prior_steps = PROBE_TRAINING_SCHEDULE[
        "dynamic_crop_v3_prior_pretraining_steps"
    ]
    finetune_steps = PROBE_TRAINING_SCHEDULE["finetune_steps_planned"]
    prior = training.get("dynamic_crop_v3_prior_pretraining")
    prior_history = training.get("dynamic_crop_v3_prior_pretraining_history")
    finetune_history = training.get("finetune_history")
    if not isinstance(prior, Mapping):
        raise DynamicCropV3FinalFitError(
            "final-fit v3 prior evidence is missing"
        )
    prior_contract = prior.get("contract")
    full_head_prior = prior_contract == (
        "all_reference_predeclared_detached_feature_head_milestones_"
        "frozen_loso_candidate_no_reselection_v2"
    )
    standardized_structured_adapter_prior = prior_contract == (
        "all_reference_predeclared_balanced_standardized_structured_"
        "low_rank_adapter_milestones_frozen_loso_candidate_no_"
        "reselection_v4"
    )
    structured_adapter_prior = standardized_structured_adapter_prior or prior_contract == (
        "all_reference_predeclared_structured_low_rank_adapter_"
        "milestones_frozen_loso_candidate_no_reselection_v3"
    )
    head_snapshot_prior = full_head_prior or structured_adapter_prior
    if (
        not full_head_prior
        and not structured_adapter_prior
        and prior_contract is not None
        and prior_contract
        != (
            "all_reference_predeclared_bias_milestones_frozen_loso_"
            "candidate_no_reselection_v1"
        )
    ):
        raise DynamicCropV3FinalFitError(
            "final-fit v3 prior contract differs"
        )
    sampling = prior.get("sampling")
    invariance = prior.get("parameter_invariance")
    retention = prior.get("retention")
    candidate_snapshots = prior.get("candidate_snapshots")
    frozen_prior_step = frozen.get("dynamic_crop_v3_prior_candidate_step")
    expected_reference_song_ids = {
        reference_id: song_id
        for song_id in SELECTION_SONG_IDS
        for reference_id in references_by_song[song_id]
    }
    candidate_digest_by_step: dict[int, str] = {}
    candidate_head_digest_by_step: dict[int, str] = {}
    if not isinstance(candidate_snapshots, list) or len(candidate_snapshots) != len(
        production.DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS
    ):
        raise DynamicCropV3FinalFitError(
            "final-fit v3 prior candidate snapshots differ"
        )
    for expected_step, raw_candidate in zip(
        production.DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS,
        candidate_snapshots,
        strict=True,
    ):
        if not isinstance(raw_candidate, Mapping):
            raise DynamicCropV3FinalFitError(
                "final-fit v3 prior candidate snapshot is invalid"
            )
        digest = raw_candidate.get("value_bias_sha256")
        head_digest = raw_candidate.get("head_parameter_sha256")
        if (
            raw_candidate.get("candidate_step") != expected_step
            or raw_candidate.get("training_only_updates") is not True
            or not _is_sha256(digest)
            or head_snapshot_prior
            and not _is_sha256(head_digest)
        ):
            raise DynamicCropV3FinalFitError(
                "final-fit v3 prior candidate snapshot differs"
            )
        candidate_digest_by_step[expected_step] = str(digest)
        if head_snapshot_prior:
            candidate_head_digest_by_step[expected_step] = str(head_digest)
    if (
        training.get("cut_pretraining_optimizer_steps") != cut_steps
        or training.get("source_pretraining_optimizer_steps") != source_steps
        or training.get("dynamic_crop_v3_prior_pretraining_optimizer_steps")
        != prior_steps
        or training.get("finetune_optimizer_steps") != finetune_steps
        or training.get("optimizer_steps")
        != cut_steps + source_steps + prior_steps + finetune_steps
        or not isinstance(prior_history, list)
        or len(prior_history) != prior_steps
        or not isinstance(finetune_history, list)
        or not finetune_history
        or prior.get("optimizer_weight_decay") != 0.0
        or prior.get("optimizer_initial_step") != 0
        or prior.get("optimizer_state_final_step") != prior_steps
        or prior.get("checkpoint_selection_performed") is not False
        or prior.get("validation_metrics_computed") is not False
        or prior.get("heldout_metrics_computed") is not False
        or prior.get("predeclared_candidate_steps")
        != list(production.DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS)
        or prior.get("candidate_selection_contract")
        != production.DYNAMIC_CROP_V3_PRIOR_CANDIDATE_SELECTION_CONTRACT
        or prior.get("frozen_selected_candidate_step") != frozen_prior_step
        or prior.get("frozen_selected_candidate_value_bias_sha256")
        != candidate_digest_by_step.get(frozen_prior_step)
        or head_snapshot_prior
        and prior.get("frozen_selected_candidate_head_parameter_sha256")
        != candidate_head_digest_by_step.get(frozen_prior_step)
        or prior.get("final_fit_data_used_for_candidate_selection") is not False
        or not isinstance(sampling, Mapping)
        or not isinstance(invariance, Mapping)
        or invariance.get("all_declared_invariant_groups_byte_identical")
        is not True
        or structured_adapter_prior
        and invariance.get("context_value_bias_changed") is not False
        or not structured_adapter_prior
        and invariance.get("context_value_bias_changed") is not True
        or head_snapshot_prior
        and invariance.get("dynamic_head_parameter_tree_changed") is not True
        or not isinstance(invariance.get("after"), Mapping)
        or invariance.get("after", {}).get("context_value_bias_sha256")
        != candidate_digest_by_step.get(frozen_prior_step)
        or head_snapshot_prior
        and invariance.get("after", {}).get(
            "dynamic_head_parameter_tree_sha256"
        )
        != candidate_head_digest_by_step.get(frozen_prior_step)
        or not isinstance(retention, Mapping)
        or retention.get("retention_finetune_step_count")
        != PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_retention_finetune_steps"
        ]
        or retention.get("retention_optimizer_state_final_step")
        != PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_retention_finetune_steps"
        ]
        or retention.get("released_joint_optimizer_initial_step") != 0
        or retention.get("released_joint_optimizer_state_final_step")
        != finetune_steps
        - PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_retention_finetune_steps"
        ]
        or retention.get("optimizer_state_reset_before_dynamic_head_release")
        is not True
        or retention.get("released_joint_optimizer_schedule_offset_steps")
        != PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_retention_finetune_steps"
        ]
        or retention.get(
            "all_dynamic_head_parameters_byte_identical_at_release"
        )
        is not True
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
    if standardized_structured_adapter_prior:
        normalization = sampling.get("projection_normalization")
        class_balance = prior.get("class_balance")
        projection_rank = sampling.get("projection_rank")
        structured_dimension = (
            production.DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION
            - production.DYNAMIC_CROP_V3_MAX_KEYFRAMES * 4
        )
        expected_parameter_count = (
            structured_dimension * (2 * int(projection_rank) + 1)
            if isinstance(projection_rank, int)
            and not isinstance(projection_rank, bool)
            and projection_rank > 0
            else None
        )
        normalization_hash_keys = (
            "context_mean_sha256",
            "context_scale_sha256",
            "source_mean_sha256",
            "source_scale_sha256",
        )
        normalization_scale_keys = (
            "context_scale_min",
            "context_scale_max",
            "source_scale_min",
            "source_scale_max",
        )
        count_histogram = (
            class_balance.get("count_histogram_1_to_8", [])
            if isinstance(class_balance, Mapping)
            else []
        )
        count_weights = (
            class_balance.get("count_class_weights_1_to_8", [])
            if isinstance(class_balance, Mapping)
            else []
        )
        motion_histogram = (
            class_balance.get("motion_histogram_static_dynamic", [])
            if isinstance(class_balance, Mapping)
            else []
        )
        motion_weights = (
            class_balance.get("motion_class_weights_static_dynamic", [])
            if isinstance(class_balance, Mapping)
            else []
        )

        def valid_histogram(values: object, length: int) -> bool:
            return bool(
                isinstance(values, list)
                and len(values) == length
                and all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                    for value in values
                )
            )

        def valid_weights(values: object, length: int) -> bool:
            return bool(
                isinstance(values, list)
                and len(values) == length
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) >= 0.0
                    for value in values
                )
            )

        count_histogram_valid = valid_histogram(count_histogram, 8)
        motion_histogram_valid = valid_histogram(motion_histogram, 2)
        count_total = sum(count_histogram) if count_histogram_valid else -1
        motion_total = sum(motion_histogram) if motion_histogram_valid else -1
        if (
            prior.get("mutable_parameter_scope")
            != production.DYNAMIC_CROP_V3_PRIOR_PARAMETER_SCOPE_CONTRACT
            or sampling.get("adapter_contract")
            != production.DYNAMIC_CROP_V3_PRIOR_ADAPTER_CONTRACT
            or not isinstance(projection_rank, int)
            or isinstance(projection_rank, bool)
            or not 1 <= projection_rank <= production.DYNAMIC_CROP_V3_PRIOR_PROJECTION_RANK
            or sampling.get("trainable_parameter_count")
            != expected_parameter_count
            or not _is_sha256(sampling.get("context_projection_sha256"))
            or not _is_sha256(sampling.get("source_projection_sha256"))
            or not isinstance(normalization, Mapping)
            or normalization.get("contract")
            != "training_track_projection_zscore_floor_1e_3_v1"
            or normalization.get("track_count") != count_total
            or any(
                not _is_sha256(normalization.get(key))
                for key in normalization_hash_keys
            )
            or any(
                not isinstance(normalization.get(key), (int, float))
                or isinstance(normalization.get(key), bool)
                or not math.isfinite(float(normalization[key]))
                or float(normalization[key]) <= 0.0
                for key in normalization_scale_keys
            )
            or normalization.get("validation_labels_consumed") is not False
            or normalization.get("heldout_labels_consumed") is not False
            or not isinstance(class_balance, Mapping)
            or class_balance.get("contract")
            != "training_only_inverse_frequency_expected_weight_one_v1"
            or not count_histogram_valid
            or not valid_weights(count_weights, 8)
            or not motion_histogram_valid
            or not valid_weights(motion_weights, 2)
            or count_total < 1
            or motion_total < 1
            or class_balance.get("validation_labels_consumed") is not False
            or class_balance.get("heldout_labels_consumed") is not False
            or invariance.get("context_value_weight_changed") is not False
            or invariance.get("source_value_weight_changed") is not False
            or invariance.get("context_structured_weight_changed") is not True
            or invariance.get("source_structured_weight_changed") is not True
            or invariance.get("value_row_gradients_computed") is not False
            or invariance.get(
                "structured_low_rank_adapter_gradients_computed"
            )
            is not True
        ):
            raise DynamicCropV3FinalFitError(
                "final-fit standardized structured prior evidence differs"
            )
    invariant_parameter_groups = (
        (
            "base_parameter_tree_sha256",
            "context_value_bias_sha256",
            "context_value_weight_sha256",
            "source_value_weight_sha256",
        )
        if structured_adapter_prior
        else ("base_parameter_tree_sha256",)
        if full_head_prior
        else (
            "base_parameter_tree_sha256",
            "context_weight_sha256",
            "source_head_parameter_tree_sha256",
            "context_non_value_bias_sha256",
        )
    )
    all_prior_parameter_groups = (
        (
            "base_parameter_tree_sha256",
            "context_weight_sha256",
            "source_head_parameter_tree_sha256",
            "dynamic_head_parameter_tree_sha256",
            "context_bias_sha256",
            "context_non_value_bias_sha256",
            "context_value_bias_sha256",
            "context_value_weight_sha256",
            "context_structured_weight_sha256",
            "source_value_weight_sha256",
            "source_structured_weight_sha256",
        )
        if structured_adapter_prior
        else
        (
            "base_parameter_tree_sha256",
            "context_weight_sha256",
            "source_head_parameter_tree_sha256",
            "dynamic_head_parameter_tree_sha256",
            "context_bias_sha256",
            "context_non_value_bias_sha256",
            "context_value_bias_sha256",
        )
        if full_head_prior
        else (
            *invariant_parameter_groups,
            "context_value_bias_sha256",
        )
    )
    invariance_before = invariance.get("before")
    invariance_after = invariance.get("after")
    if (
        not isinstance(invariance_before, Mapping)
        or not isinstance(invariance_after, Mapping)
        or set(invariance_before) != set(all_prior_parameter_groups)
        or set(invariance_after) != set(all_prior_parameter_groups)
        or invariance.get("invariant_parameter_groups")
        != list(invariant_parameter_groups)
        or any(
            not _is_sha256(invariance_before.get(key))
            or not _is_sha256(invariance_after.get(key))
            for key in all_prior_parameter_groups
        )
        or any(
            invariance_before.get(key) != invariance_after.get(key)
            for key in invariant_parameter_groups
        )
        or structured_adapter_prior
        and invariance_before.get("context_value_bias_sha256")
        != invariance_after.get("context_value_bias_sha256")
        or not structured_adapter_prior
        and invariance_before.get("context_value_bias_sha256")
        == invariance_after.get("context_value_bias_sha256")
        or invariance_after.get("context_value_bias_sha256")
        != candidate_digest_by_step.get(frozen_prior_step)
        or head_snapshot_prior
        and invariance_before.get("dynamic_head_parameter_tree_sha256")
        == invariance_after.get("dynamic_head_parameter_tree_sha256")
        or head_snapshot_prior
        and invariance_after.get("dynamic_head_parameter_tree_sha256")
        != candidate_head_digest_by_step.get(frozen_prior_step)
    ):
        raise DynamicCropV3FinalFitError(
            "final-fit v3 prior parameter digests differ"
        )
    retained_head_groups = (
        "context_weight_sha256",
        "source_head_parameter_tree_sha256",
        "context_non_value_bias_sha256",
        "context_value_bias_sha256",
    )
    warmup_head_digests = retention.get("warmup_head_digests")
    retention_boundary_head_digests = retention.get(
        "retention_boundary_head_digests"
    )
    post_joint_head_digests = retention.get("post_joint_head_digests")
    if (
        not isinstance(warmup_head_digests, Mapping)
        or not isinstance(retention_boundary_head_digests, Mapping)
        or not isinstance(post_joint_head_digests, Mapping)
        or set(warmup_head_digests) != set(retained_head_groups)
        or set(retention_boundary_head_digests) != set(retained_head_groups)
        or set(post_joint_head_digests) != set(retained_head_groups)
        or any(
            not _is_sha256(warmup_head_digests.get(key))
            or not _is_sha256(retention_boundary_head_digests.get(key))
            or not _is_sha256(post_joint_head_digests.get(key))
            for key in retained_head_groups
        )
        or any(
            warmup_head_digests.get(key)
            != retention_boundary_head_digests.get(key)
            or warmup_head_digests.get(key) != invariance_after.get(key)
            for key in retained_head_groups
        )
        or warmup_head_digests.get("context_value_bias_sha256")
        != candidate_digest_by_step.get(frozen_prior_step)
    ):
        raise DynamicCropV3FinalFitError(
            "final-fit v3 prior retention digests differ"
        )
    assert isinstance(prior_history, list)
    observed_song_samples: dict[str, int] = {}
    observed_reference_samples: dict[str, int] = {}
    observed_prior_learning_rates: list[float] = []
    for step, raw_record in enumerate(prior_history, start=1):
        if not isinstance(raw_record, Mapping):
            raise DynamicCropV3FinalFitError(
                "final-fit v3 prior history differs"
            )
        song_id = str(raw_record.get("song_id") or "")
        reference_id = str(raw_record.get("reference_id") or "")
        is_candidate = step in production.DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS
        if (
            raw_record.get("phase_step") != step
            or raw_record.get("optimizer_step")
            != cut_steps + source_steps + step
            or raw_record.get("checkpoint_selection_performed") is not False
            or raw_record.get("validation_labels_consumed") is not False
            or raw_record.get("heldout_labels_consumed") is not False
            or raw_record.get("predeclared_candidate_snapshot") is not is_candidate
            or (
                raw_record.get("candidate_value_bias_sha256")
                != candidate_digest_by_step[step]
                if is_candidate
                else raw_record.get("candidate_value_bias_sha256") is not None
            )
            or head_snapshot_prior
            and (
                raw_record.get("candidate_head_parameter_sha256")
                != candidate_head_digest_by_step[step]
                if is_candidate
                else raw_record.get("candidate_head_parameter_sha256")
                is not None
            )
            or not song_id
            or not reference_id
            or expected_reference_song_ids.get(reference_id) != song_id
            or isinstance(raw_record.get("actual_learning_rate"), bool)
            or not isinstance(
                raw_record.get("actual_learning_rate"), (int, float)
            )
            or not math.isfinite(
                float(raw_record.get("actual_learning_rate"))
            )
            or float(raw_record.get("actual_learning_rate")) <= 0.0
        ):
            raise DynamicCropV3FinalFitError(
                "final-fit v3 prior history differs"
            )
        observed_song_samples[song_id] = observed_song_samples.get(song_id, 0) + 1
        observed_reference_samples[reference_id] = (
            observed_reference_samples.get(reference_id, 0) + 1
        )
        observed_prior_learning_rates.append(
            float(raw_record["actual_learning_rate"])
        )
    if (
        set(observed_song_samples) != set(SELECTION_SONG_IDS)
        or set(observed_reference_samples) != set(flattened_references)
        or observed_song_samples != sampling.get("per_song_sampled_counts")
        or observed_reference_samples
        != sampling.get("per_reference_sampled_counts")
        or sampling.get("reference_song_ids")
        != expected_reference_song_ids
        or head_snapshot_prior
        and sampling.get("feature_contract")
        != production.DYNAMIC_CROP_V3_PRIOR_FEATURE_CONTRACT
        or head_snapshot_prior
        and not _is_sha256(sampling.get("detached_feature_content_sha256"))
        or max(observed_song_samples.values())
        - min(observed_song_samples.values())
        > 1
    ):
        raise DynamicCropV3FinalFitError(
            "final-fit v3 prior sampling differs from all references"
        )
    prior_peak_learning_rate = float(
        PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_pretraining_peak_learning_rate"
        ]
    )
    prior_warmup_steps = int(
        PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_pretraining_warmup_steps"
        ]
    )
    prior_minimum_rate = prior_peak_learning_rate * float(
        PROBE_TRAINING_SCHEDULE["minimum_learning_rate_ratio"]
    )
    prior_decay_steps = prior_steps - prior_warmup_steps

    def expected_prior_learning_rate(schedule_index: int) -> float:
        if schedule_index < prior_warmup_steps:
            start = prior_peak_learning_rate * 0.1
            return start + (prior_peak_learning_rate - start) * (
                schedule_index / prior_warmup_steps
            )
        local = min(schedule_index - prior_warmup_steps, prior_decay_steps)
        cosine = 0.5 * (
            1.0 + math.cos(math.pi * local / prior_decay_steps)
        )
        return prior_minimum_rate + (
            prior_peak_learning_rate - prior_minimum_rate
        ) * cosine

    if any(
        not math.isclose(
            observed,
            expected_prior_learning_rate(step - 1),
            rel_tol=1.0e-6,
            abs_tol=1.0e-12,
        )
        for step, observed in enumerate(
            observed_prior_learning_rates, start=1
        )
    ):
        raise DynamicCropV3FinalFitError(
            "final-fit v3 prior learning-rate history differs"
        )
    observed_phase_local_steps: list[int] = []
    observed_reset_before_steps: list[int] = []
    observed_retention_steps: list[int] = []
    observed_learning_rates: list[float] = []
    assert isinstance(finetune_history, list)
    for raw_epoch in finetune_history:
        if not isinstance(raw_epoch, Mapping):
            raise DynamicCropV3FinalFitError(
                "final-fit optimizer reset history differs"
            )
        if (
            raw_epoch.get("checkpoint_selection_performed") is not False
            or raw_epoch.get("validation_labels_consumed") is not False
        ):
            raise DynamicCropV3FinalFitError(
                "final-fit optimizer reset history selected on labels"
            )
        for destination, key in (
            (observed_phase_local_steps, "phase_local_optimizer_steps"),
            (
                observed_reset_before_steps,
                "optimizer_reset_before_finetune_steps",
            ),
            (
                observed_retention_steps,
                "dynamic_crop_v3_prior_retention_finetune_steps",
            ),
        ):
            values = raw_epoch.get(key)
            if not isinstance(values, list) or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in values
            ):
                raise DynamicCropV3FinalFitError(
                    "final-fit optimizer reset history differs"
                )
            destination.extend(values)
        learning_rates = raw_epoch.get("actual_learning_rates")
        if not isinstance(learning_rates, list) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in learning_rates
        ):
            raise DynamicCropV3FinalFitError(
                "final-fit learning-rate history differs"
            )
        observed_learning_rates.extend(float(value) for value in learning_rates)
    expected_retention_steps = PROBE_TRAINING_SCHEDULE[
        "dynamic_crop_v3_prior_retention_finetune_steps"
    ]
    peak_learning_rate = float(PROBE_TRAINING_SCHEDULE["learning_rate"])
    warmup_steps = int(PROBE_TRAINING_SCHEDULE["finetune_warmup_steps"])
    minimum_rate = peak_learning_rate * float(
        PROBE_TRAINING_SCHEDULE["minimum_learning_rate_ratio"]
    )
    decay_steps = finetune_steps - warmup_steps

    def expected_learning_rate(schedule_index: int) -> float:
        if schedule_index < warmup_steps:
            start = peak_learning_rate * 0.1
            return start + (peak_learning_rate - start) * (
                schedule_index / warmup_steps
            )
        local = min(schedule_index - warmup_steps, decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * local / decay_steps))
        return minimum_rate + (peak_learning_rate - minimum_rate) * cosine

    expected_learning_rates = [
        expected_learning_rate(step - 1)
        for step in range(1, finetune_steps + 1)
    ]
    if (
        observed_phase_local_steps
        != [
            *range(1, expected_retention_steps + 1),
            *range(1, finetune_steps - expected_retention_steps + 1),
        ]
        or observed_reset_before_steps != [expected_retention_steps + 1]
        or observed_retention_steps
        != list(range(1, expected_retention_steps + 1))
        or len(observed_learning_rates) != finetune_steps
        or any(
            not math.isclose(observed, expected, rel_tol=1.0e-6, abs_tol=1.0e-12)
            for observed, expected in zip(
                observed_learning_rates,
                expected_learning_rates,
                strict=True,
            )
        )
    ):
        raise DynamicCropV3FinalFitError(
            "final-fit optimizer reset history differs"
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
