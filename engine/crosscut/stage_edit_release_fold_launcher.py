"""Fail-closed IVE StageEdit development-to-blind-release launcher.

This module intentionally sits beside, rather than replaces, the historical
three-song production-fold launcher.  It implements the release protocol used
once REBEL HEART is available:

* development: BANG BANG + BLACKHOLE + REBEL HEART train, XOXZ held out;
* immutable development freeze: report, selection freeze, model config and
  selected checkpoint weights are all rebound by SHA-256; and
* final: the four development songs train, ATTITUDE is evaluated exactly once
  as the blind held-out song.  Only this final run carries activation authority.

The low-level trainer remains :mod:`stage_edit_production_train_cli`.  In
particular, checkpoint and cut-threshold selection still happen on whole
references from training songs before the held-out forward pass.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import stage_edit_production_fold_launcher as legacy
from .professional_montage_policy import (
    ProfessionalMontagePolicyError,
    load_professional_montage_policy,
)
from .stage_edit_production_train_cli import main as production_train_main
from .stage_edit_production_training import (
    STAGE_EDIT_SELECTION_FREEZE_FILENAME,
    ProductionQualityThresholds,
    _categorical_target_coverage,
    production_song_split,
)
from .stage_edit_reference_adapter import load_reference_training_corpus
from .stage_edit_dynamic_crop_head_v2 import DynamicCropV2LossContract
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    PRODUCTION_TRAINING_REPORT_FILENAME,
    PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION,
    PRODUCTION_TRAINING_RUN_KIND,
    StageEditLossContract,
    StageEditTransformerConfig,
    production_stage_edit_config,
)
from .stage_edit_transition_authority import (
    RENDERER_OWNED_ALLOWED_MODEL_LABELS,
    RENDERER_OWNED_LEARNED_EVALUATION_SKIP_REASON,
    RENDERER_OWNED_TRANSITION_AUTHORITY,
    StageEditTransitionAuthority,
)


RELEASE_PLAN_SCHEMA_VERSION = 3
FINAL_RELEASE_PLAN_SCHEMA_VERSION = 4
DEVELOPMENT_PLAN_ANALYSIS_BASIS = "ive_stageedit_release_development_plan_v3"
DEVELOPMENT_FREEZE_ANALYSIS_BASIS = "ive_stageedit_development_freeze_v3"
FINAL_PLAN_ANALYSIS_BASIS = "ive_stageedit_attitude_blind_release_plan_v4"
MODEL_TRAINING_CONTRACT_VERSION = 2
STRUCTURED_HEAD_PREFLIGHT_CONTRACT_VERSION = 2

BANG_BANG = "IVE-BANG-BANG"
BLACKHOLE = "IVE-BLACKHOLE"
REBEL_HEART = "IVE-REBEL-HEART"
XOXZ = "IVE-XOXZ"
ATTITUDE = "IVE-ATTITUDE"

DEVELOPMENT_TRAINING_SONG_IDS = (BANG_BANG, BLACKHOLE, REBEL_HEART)
DEVELOPMENT_SONG_IDS = (*DEVELOPMENT_TRAINING_SONG_IDS, XOXZ)
FINAL_TRAINING_SONG_IDS = (*DEVELOPMENT_TRAINING_SONG_IDS, XOXZ)
FINAL_SONG_IDS = (*FINAL_TRAINING_SONG_IDS, ATTITUDE)

DEFAULT_RELEASE_TRAINING_CONFIG: Mapping[str, Any] = {
    "epochs": 30,
    "patience": 6,
    "batch_size": 2,
    "learning_rate": 1.0e-4,
    "weight_decay": 1.0e-3,
    "gradient_clip_norm": 1.0,
    "maximum_cut_positive_weight": 8.0,
    "cut_pretrain_steps": 500,
    "cut_pretrain_learning_rate": 3.0e-4,
    "cut_pretrain_window_length": 32,
    "cut_pretrain_warmup_steps": 50,
    "finetune_warmup_steps": 8,
    "minimum_learning_rate_ratio": 0.10,
    "seed": 20260824,
    "device": "gpu",
}

ReleaseFoldPreflightError = legacy.ProductionFoldPreflightError


@dataclass(frozen=True, slots=True)
class ReleaseFoldPreflight:
    stage: str
    plan_path: Path
    plan_sha256: str
    held_out_song_id: str
    training_song_ids: tuple[str, ...]
    checkpoint_directory: Path
    blind_consumption_marker: Path | None
    train_cli_args: tuple[str, ...]
    reference_ids_by_song: Mapping[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "plan_path": str(self.plan_path),
            "plan_sha256": self.plan_sha256,
            "held_out_song_id": self.held_out_song_id,
            "training_song_ids": list(self.training_song_ids),
            "checkpoint_directory": str(self.checkpoint_directory),
            "blind_consumption_marker": (
                str(self.blind_consumption_marker)
                if self.blind_consumption_marker is not None
                else None
            ),
            "held_out_used_for_model_or_threshold_selection": False,
            "activation_authority_present": (
                "--authorize-production-activation" in self.train_cli_args
            ),
            "reference_ids_by_song": {
                song: list(reference_ids)
                for song, reference_ids in sorted(
                    self.reference_ids_by_song.items()
                )
            },
            "train_cli_args": list(self.train_cli_args),
        }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return legacy._is_sha256(value)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseFoldPreflightError(f"{name} must be an object")
    return value


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReleaseFoldPreflightError(f"{name} must be an array")
    return value


def _file_sha256(path: Path, name: str) -> str:
    return legacy._stable_file_sha256(Path(path).expanduser().resolve(), name)


def _pointer(path: Path, name: str) -> dict[str, str]:
    resolved = Path(path).expanduser().resolve()
    return {
        "path": str(resolved),
        "file_sha256": _file_sha256(resolved, name),
    }


def _verify_pointer(value: object, name: str) -> Path:
    pointer = _mapping(value, name)
    if set(pointer) != {"path", "file_sha256"}:
        raise ReleaseFoldPreflightError(f"{name} pointer keys are not exact")
    raw_path = pointer.get("path")
    expected = pointer.get("file_sha256")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ReleaseFoldPreflightError(f"{name} path must be absolute")
    if not _is_sha256(expected):
        raise ReleaseFoldPreflightError(f"{name} SHA-256 is invalid")
    path = Path(raw_path).resolve()
    if _file_sha256(path, name) != expected:
        raise ReleaseFoldPreflightError(f"{name} SHA-256 mismatch")
    return path


def _write_content_addressed(
    unsigned: Mapping[str, Any],
    *,
    output_directory: Path,
    prefix: str,
    hash_field: str,
) -> Path:
    digest = _canonical_sha256(unsigned)
    payload = {**unsigned, hash_field: digest}
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{prefix}-{digest[:16]}.json"
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
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise ReleaseFoldPreflightError(
                f"content-addressed path contains different bytes: {path}"
            )
    return path


def _strict_content_addressed(
    path: Path,
    *,
    expected_keys: set[str],
    schema_basis: str,
    prefix: str,
    hash_field: str,
    expected_schema_version: int = RELEASE_PLAN_SCHEMA_VERSION,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ReleaseFoldPreflightError(
            f"release evidence must be a regular non-symlink file: {resolved}"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ReleaseFoldPreflightError("release evidence keys are not exact")
    if payload.get("schema_version") != expected_schema_version:
        raise ReleaseFoldPreflightError("release evidence schema mismatch")
    if payload.get("analysis_basis") != schema_basis:
        raise ReleaseFoldPreflightError("release evidence analysis basis mismatch")
    observed = payload.get(hash_field)
    unsigned = deepcopy(payload)
    unsigned.pop(hash_field, None)
    if not _is_sha256(observed) or _canonical_sha256(unsigned) != observed:
        raise ReleaseFoldPreflightError("release evidence content hash mismatch")
    if resolved.name != f"{prefix}-{str(observed)[:16]}.json":
        raise ReleaseFoldPreflightError(
            "release evidence filename is not content-addressed"
        )
    return payload


def _validate_training_config(value: object) -> dict[str, Any]:
    config = dict(_mapping(value, "release training config"))
    if set(config) != set(DEFAULT_RELEASE_TRAINING_CONFIG):
        raise ReleaseFoldPreflightError("release training config keys are not exact")
    integer_positive = (
        "epochs",
        "patience",
        "batch_size",
        "cut_pretrain_steps",
        "cut_pretrain_window_length",
        "cut_pretrain_warmup_steps",
        "finetune_warmup_steps",
        "seed",
    )
    if any(not isinstance(config[key], int) for key in integer_positive):
        raise ReleaseFoldPreflightError("release integer training config is invalid")
    if config["epochs"] < 1 or config["patience"] < 1 or config["batch_size"] < 1:
        raise ReleaseFoldPreflightError("release epochs/patience/batch size are invalid")
    if config["cut_pretrain_steps"] < 500:
        raise ReleaseFoldPreflightError("release cut pretraining requires >=500 steps")
    if not 0 <= config["cut_pretrain_warmup_steps"] < config["cut_pretrain_steps"]:
        raise ReleaseFoldPreflightError("release cut pretrain warmup is invalid")
    if config["cut_pretrain_window_length"] < 2 or config["finetune_warmup_steps"] < 0:
        raise ReleaseFoldPreflightError("release training schedule is invalid")
    for key in (
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "maximum_cut_positive_weight",
        "cut_pretrain_learning_rate",
        "minimum_learning_rate_ratio",
    ):
        value = config[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ReleaseFoldPreflightError(f"release {key} is invalid")
        if float(value) <= 0.0 and key != "weight_decay":
            raise ReleaseFoldPreflightError(f"release {key} is invalid")
        if key == "weight_decay" and float(value) < 0.0:
            raise ReleaseFoldPreflightError(f"release {key} is invalid")
    if not 0.0 < float(config["minimum_learning_rate_ratio"]) <= 1.0:
        raise ReleaseFoldPreflightError("minimum learning-rate ratio is invalid")
    if config["device"] != "gpu":
        raise ReleaseFoldPreflightError("release training requires MLX GPU")
    return config


def _validate_model_training_contract(value: object) -> dict[str, Any]:
    """Validate the immutable capacity and base-loss contract used by the CLI."""

    contract = dict(_mapping(value, "release model/training contract"))
    expected_keys = {
        "contract_version",
        "model_config",
        "model_config_sha256",
        "base_loss_contract",
        "base_loss_contract_sha256",
        "dynamic_crop_loss_contract",
        "dynamic_crop_loss_contract_sha256",
        "source_objective",
        "transition_authority_contract",
        "transition_authority_contract_sha256",
    }
    if set(contract) != expected_keys:
        raise ReleaseFoldPreflightError(
            "release model/training contract keys are not exact"
        )
    if contract.get("contract_version") != MODEL_TRAINING_CONTRACT_VERSION:
        raise ReleaseFoldPreflightError(
            "release model/training contract version is invalid"
        )
    model_payload = dict(_mapping(contract.get("model_config"), "model config"))
    if contract.get("model_config_sha256") != _canonical_sha256(model_payload):
        raise ReleaseFoldPreflightError("release model config hash mismatch")
    try:
        model = StageEditTransformerConfig.from_dict(model_payload)
    except (TypeError, ValueError) as exc:
        raise ReleaseFoldPreflightError(
            f"release model config is invalid: {exc}"
        ) from exc
    if (
        model.token_schema_fingerprint != DEFAULT_STAGE_EDIT_SCHEMA.fingerprint
        or model.max_sources != legacy.PRODUCTION_SAMPLING_CONTRACT["max_sources"]
    ):
        raise ReleaseFoldPreflightError(
            "release model config differs from token/source capacity contract"
        )

    base_payload = dict(
        _mapping(contract.get("base_loss_contract"), "base loss contract")
    )
    if contract.get("base_loss_contract_sha256") != _canonical_sha256(base_payload):
        raise ReleaseFoldPreflightError("release base loss contract hash mismatch")
    try:
        base_loss = StageEditLossContract(**base_payload)
    except (TypeError, ValueError) as exc:
        raise ReleaseFoldPreflightError(
            f"release base loss contract is invalid: {exc}"
        ) from exc
    if contract.get("source_objective") != base_loss.source_objective:
        raise ReleaseFoldPreflightError("release source objective binding mismatch")

    dynamic_payload = dict(
        _mapping(
            contract.get("dynamic_crop_loss_contract"),
            "dynamic crop loss contract",
        )
    )
    if contract.get("dynamic_crop_loss_contract_sha256") != _canonical_sha256(
        dynamic_payload
    ):
        raise ReleaseFoldPreflightError(
            "release dynamic crop loss contract hash mismatch"
        )
    dynamic_base = dict(
        _mapping(dynamic_payload.get("base"), "dynamic crop base loss contract")
    )
    try:
        DynamicCropV2LossContract(
            **{
                **dynamic_payload,
                "base": StageEditLossContract(**dynamic_base),
            }
        )
    except (TypeError, ValueError) as exc:
        raise ReleaseFoldPreflightError(
            f"release dynamic crop loss contract is invalid: {exc}"
        ) from exc
    if dynamic_base != base_payload:
        raise ReleaseFoldPreflightError(
            "release dynamic crop and base source objectives differ"
        )
    authority = StageEditTransitionAuthority(
        mode=RENDERER_OWNED_TRANSITION_AUTHORITY
    )
    if (
        model.transition_authority != authority.mode
        or base_loss.transition_authority != authority.mode
        or base_loss.transition_weight != 0.0
        or contract.get("transition_authority_contract") != authority.to_dict()
        or contract.get("transition_authority_contract_sha256")
        != authority.fingerprint
    ):
        raise ReleaseFoldPreflightError(
            "release model/loss renderer-owned transition authority differs"
        )
    return contract


def _default_model_training_contract() -> dict[str, Any]:
    authority = StageEditTransitionAuthority(
        mode=RENDERER_OWNED_TRANSITION_AUTHORITY
    )
    model = asdict(
        replace(
            production_stage_edit_config(),
            transition_authority=authority.mode,
        )
    )
    base = StageEditLossContract(
        transition_weight=0.0,
        transition_authority=authority.mode,
    )
    base_loss = asdict(base)
    dynamic_loss = asdict(DynamicCropV2LossContract(base=base))
    return _validate_model_training_contract(
        {
            "contract_version": MODEL_TRAINING_CONTRACT_VERSION,
            "model_config": model,
            "model_config_sha256": _canonical_sha256(model),
            "base_loss_contract": base_loss,
            "base_loss_contract_sha256": _canonical_sha256(base_loss),
            "dynamic_crop_loss_contract": dynamic_loss,
            "dynamic_crop_loss_contract_sha256": _canonical_sha256(dynamic_loss),
            "source_objective": base_loss["source_objective"],
            "transition_authority_contract": authority.to_dict(),
            "transition_authority_contract_sha256": authority.fingerprint,
        }
    )


def _structured_head_preflight_contract() -> dict[str, Any]:
    thresholds = ProductionQualityThresholds()
    return {
        "contract_version": STRUCTURED_HEAD_PREFLIGHT_CONTRACT_VERSION,
        "duration": {
            "mode": "learned_multiclass",
            "minimum_observed_classes": thresholds.minimum_duration_observed_classes,
            "minimum_label_entropy_bits": thresholds.minimum_duration_label_entropy_bits,
        },
        "transition": {
            "mode": "renderer_owned_metadata_only",
            "allowed_model_labels": list(RENDERER_OWNED_ALLOWED_MODEL_LABELS),
            "required_label_set": list(RENDERER_OWNED_ALLOWED_MODEL_LABELS),
            "minimum_label_count_per_partition": 1,
            "learned_evaluation_status": "skipped_non_authoritative",
            "learned_evaluation_reason": (
                RENDERER_OWNED_LEARNED_EVALUATION_SKIP_REASON
            ),
            "authority": StageEditTransitionAuthority(
                mode=RENDERER_OWNED_TRANSITION_AUTHORITY
            ).to_dict(),
        },
    }


def _validate_structured_head_preflight_contract(value: object) -> dict[str, Any]:
    contract = dict(_mapping(value, "structured head preflight contract"))
    expected = _structured_head_preflight_contract()
    if contract != expected:
        raise ReleaseFoldPreflightError(
            "structured head preflight contract is not exact"
        )
    return contract


def _release_contract(
    *, training_song_ids: Sequence[str], held_out_song_id: str, stage: str
) -> dict[str, Any]:
    return {
        "artist": "IVE",
        "stage": stage,
        "training_song_ids": list(training_song_ids),
        "held_out_song_id": held_out_song_id,
        "expected_reference_count_per_song": legacy.EXPECTED_REFERENCE_COUNT_PER_SONG,
        "expected_source_count_per_song": legacy.EXPECTED_SOURCE_COUNT_PER_SONG,
        "sampling": dict(legacy.PRODUCTION_SAMPLING_CONTRACT),
        "recent_history_mask_contract": dict(
            legacy.PRODUCTION_RECENT_HISTORY_MASK_CONTRACT
        ),
        "model_selection_validation_level": (
            "whole_reference_within_training_songs"
        ),
        "held_out_used_for_model_or_threshold_selection": False,
        "quality_threshold_override_allowed": False,
        "synthetic_or_fake_training_data_allowed": False,
        "activation_allowed": stage == "final_blind",
        "activation_requires_final_blind_pass": True,
    }


def _song_records(
    song_ids: Sequence[str],
    *,
    materialization_manifests: Mapping[str, Path],
    feature_manifests: Mapping[str, Path],
    dynamic_crop_manifests: Mapping[str, Path],
) -> list[dict[str, Any]]:
    required = set(song_ids)
    for name, paths in (
        ("materialization", materialization_manifests),
        ("feature", feature_manifests),
        ("dynamic crop", dynamic_crop_manifests),
    ):
        if set(paths) != required:
            raise ReleaseFoldPreflightError(
                f"{name} manifests must name the exact {len(required)} release songs"
            )
    return [
        {
            "song_id": song_id,
            "materialization_manifest": _pointer(
                materialization_manifests[song_id],
                f"{song_id} materialization manifest",
            ),
            "feature_manifest": _pointer(
                feature_manifests[song_id], f"{song_id} feature manifest"
            ),
            "dynamic_crop_manifest": _pointer(
                dynamic_crop_manifests[song_id],
                f"{song_id} dynamic crop manifest",
            ),
        }
        for song_id in song_ids
    ]


def create_stage_edit_release_development_plan(
    *,
    materialization_manifests: Mapping[str, Path],
    feature_manifests: Mapping[str, Path],
    dynamic_crop_manifests: Mapping[str, Path],
    checkpoint_directory: Path,
    output_directory: Path,
    training_config: Mapping[str, Any] = DEFAULT_RELEASE_TRAINING_CONFIG,
    model_training_contract: Mapping[str, Any] | None = None,
) -> Path:
    """Bind the exact four-song development fold without activation authority."""

    checkpoint = Path(checkpoint_directory).expanduser().resolve()
    volume = legacy.PRODUCTION_VOLUME_ROOT.resolve()
    if volume != checkpoint and volume not in checkpoint.parents:
        raise ReleaseFoldPreflightError(
            f"development checkpoint must be on {volume}: {checkpoint}"
        )
    unsigned = {
        "schema_version": RELEASE_PLAN_SCHEMA_VERSION,
        "analysis_basis": DEVELOPMENT_PLAN_ANALYSIS_BASIS,
        "stage": "development",
        "release_contract": _release_contract(
            training_song_ids=DEVELOPMENT_TRAINING_SONG_IDS,
            held_out_song_id=XOXZ,
            stage="development",
        ),
        "required_volume": str(legacy.PRODUCTION_VOLUME_ROOT),
        "minimum_free_bytes": legacy.MINIMUM_FREE_BYTES,
        "training_config": _validate_training_config(training_config),
        "model_training_contract": _validate_model_training_contract(
            model_training_contract
            if model_training_contract is not None
            else _default_model_training_contract()
        ),
        "structured_head_preflight_contract": (
            _structured_head_preflight_contract()
        ),
        "checkpoint_directory": str(checkpoint),
        "songs": _song_records(
            DEVELOPMENT_SONG_IDS,
            materialization_manifests=materialization_manifests,
            feature_manifests=feature_manifests,
            dynamic_crop_manifests=dynamic_crop_manifests,
        ),
    }
    return _write_content_addressed(
        unsigned,
        output_directory=output_directory,
        prefix="stageedit-release-development-plan",
        hash_field="plan_sha256",
    )


_DEVELOPMENT_PLAN_KEYS = {
    "schema_version",
    "analysis_basis",
    "stage",
    "release_contract",
    "required_volume",
    "minimum_free_bytes",
    "training_config",
    "model_training_contract",
    "structured_head_preflight_contract",
    "checkpoint_directory",
    "songs",
    "plan_sha256",
}


def _strict_development_plan(path: Path) -> dict[str, Any]:
    payload = _strict_content_addressed(
        path,
        expected_keys=_DEVELOPMENT_PLAN_KEYS,
        schema_basis=DEVELOPMENT_PLAN_ANALYSIS_BASIS,
        prefix="stageedit-release-development-plan",
        hash_field="plan_sha256",
    )
    expected = _release_contract(
        training_song_ids=DEVELOPMENT_TRAINING_SONG_IDS,
        held_out_song_id=XOXZ,
        stage="development",
    )
    if payload.get("stage") != "development" or payload.get("release_contract") != expected:
        raise ReleaseFoldPreflightError("development release contract is not exact")
    if (
        payload.get("required_volume") != str(legacy.PRODUCTION_VOLUME_ROOT)
        or payload.get("minimum_free_bytes") != legacy.MINIMUM_FREE_BYTES
    ):
        raise ReleaseFoldPreflightError("development storage contract mismatch")
    _validate_training_config(payload.get("training_config"))
    _validate_model_training_contract(payload.get("model_training_contract"))
    _validate_structured_head_preflight_contract(
        payload.get("structured_head_preflight_contract")
    )
    _indexed_song_records(payload, DEVELOPMENT_SONG_IDS)
    return payload


def _indexed_song_records(
    payload: Mapping[str, Any], song_ids: Sequence[str]
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for raw in _array(payload.get("songs"), "release songs"):
        item = _mapping(raw, "release song")
        if set(item) != {
            "song_id",
            "materialization_manifest",
            "feature_manifest",
            "dynamic_crop_manifest",
        }:
            raise ReleaseFoldPreflightError("release song keys are not exact")
        song_id = str(item.get("song_id") or "").strip()
        if song_id in indexed:
            raise ReleaseFoldPreflightError("duplicate release song id")
        indexed[song_id] = item
    if tuple(indexed) != tuple(song_ids):
        raise ReleaseFoldPreflightError("release songs/order are not exact")
    return indexed


def _training_config_cli_args(config: Mapping[str, Any]) -> list[str]:
    option_by_key = {
        "epochs": "--epochs",
        "patience": "--patience",
        "batch_size": "--batch-size",
        "learning_rate": "--learning-rate",
        "weight_decay": "--weight-decay",
        "gradient_clip_norm": "--gradient-clip-norm",
        "maximum_cut_positive_weight": "--maximum-cut-positive-weight",
        "cut_pretrain_steps": "--cut-pretrain-steps",
        "cut_pretrain_learning_rate": "--cut-pretrain-learning-rate",
        "cut_pretrain_window_length": "--cut-pretrain-window-length",
        "cut_pretrain_warmup_steps": "--cut-pretrain-warmup-steps",
        "finetune_warmup_steps": "--finetune-warmup-steps",
        "minimum_learning_rate_ratio": "--minimum-learning-rate-ratio",
        "seed": "--seed",
        "device": "--device",
    }
    args: list[str] = []
    for key in DEFAULT_RELEASE_TRAINING_CONFIG:
        args.extend([option_by_key[key], str(config[key])])
    return args


def _model_training_cli_args(contract: object) -> list[str]:
    validated = _validate_model_training_contract(contract)
    return [
        "--model-config-json",
        _canonical_bytes(validated["model_config"]).decode("utf-8"),
        "--base-loss-contract-json",
        _canonical_bytes(validated["base_loss_contract"]).decode("utf-8"),
        "--dynamic-crop-loss-contract-json",
        _canonical_bytes(validated["dynamic_crop_loss_contract"]).decode("utf-8"),
    ]


def _manifest_cli_args(
    payload: Mapping[str, Any], song_ids: Sequence[str]
) -> tuple[
    list[str],
    dict[str, tuple[str, ...]],
    dict[str, tuple[Path, ...]],
]:
    songs = _indexed_song_records(payload, song_ids)
    args: list[str] = []
    reference_ids: dict[str, tuple[str, ...]] = {}
    supervision_paths: dict[str, tuple[Path, ...]] = {}
    for song_id in song_ids:
        item = songs[song_id]
        _, materialization = legacy._validate_materialization_manifest(
            item.get("materialization_manifest"), song_id=song_id
        )
        _, feature = legacy._validate_feature_manifest(
            item.get("feature_manifest"),
            song_id=song_id,
            materialization=materialization,
        )
        dynamic = legacy._validate_dynamic_crop_manifest(
            item.get("dynamic_crop_manifest"),
            song_id=song_id,
            materialization=materialization,
        )
        reference_ids[song_id] = materialization.reference_ids
        supervision_paths[song_id] = tuple(
            materialization.supervision_paths[reference_id]
            for reference_id in materialization.reference_ids
        )
        for reference_id in materialization.reference_ids:
            args.extend(
                [
                    "--input",
                    str(materialization.supervision_paths[reference_id]),
                    "--song-id",
                    f"{reference_id}={song_id}",
                    "--feature-snapshot",
                    f"{reference_id}={feature.snapshot_paths[reference_id]}",
                ]
            )
        args.extend(["--dynamic-crop-manifest", str(dynamic.path)])
    return args, reference_ids, supervision_paths


def _preflight_structured_head_label_coverage(
    supervision_paths_by_song: Mapping[str, Sequence[Path]],
    *,
    contract: object,
) -> dict[str, Any]:
    """Read only the frozen development labels and reject doomed heads early."""

    validated = _validate_structured_head_preflight_contract(contract)
    expected_songs = set(DEVELOPMENT_SONG_IDS)
    if set(supervision_paths_by_song) != expected_songs:
        raise ReleaseFoldPreflightError(
            "structured head preflight did not receive the exact development fold"
        )
    paths: list[Path] = []
    song_by_reference: dict[str, str] = {}
    raw_transition_labels_by_song: dict[str, list[str]] = {
        song: [] for song in DEVELOPMENT_SONG_IDS
    }

    def reject_unsupported_direct_fields(value: object) -> None:
        if isinstance(value, Mapping):
            direct = value.get("transition_label")
            if direct is not None and str(direct).strip() not in (
                "",
                *RENDERER_OWNED_ALLOWED_MODEL_LABELS,
            ):
                raise ValueError(
                    "renderer-owned supervision contains a fabricated direct "
                    f"transition label: {direct!r}"
                )
            for child in value.values():
                reject_unsupported_direct_fields(child)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for child in value:
                reject_unsupported_direct_fields(child)

    def collect_authoritative_shot_labels(
        payload: Mapping[str, Any], song_id: str
    ) -> None:
        shot_labels = payload.get("shot_labels")
        if not isinstance(shot_labels, list):
            raise ValueError("supervision shot_labels must be an array")
        for raw_shot in shot_labels:
            if not isinstance(raw_shot, Mapping):
                raise ValueError("supervision shot label must be an object")
            edit_labels = raw_shot.get("edit_labels")
            transition = (
                edit_labels.get("source_transition")
                if isinstance(edit_labels, Mapping)
                else None
            )
            if not isinstance(transition, Mapping):
                continue
            eligible = transition.get("supervision_eligible")
            if eligible is not True and eligible is not False and eligible is not None:
                raise ValueError(
                    "transition supervision_eligible must be boolean"
                )
            if eligible is not True:
                continue
            label = str(transition.get("transition") or "").strip()
            if label not in RENDERER_OWNED_ALLOWED_MODEL_LABELS:
                raise ValueError(
                    "renderer-owned supervision contains an unrecognized or "
                    f"non-hard transition label: {label!r}"
                )
            raw_transition_labels_by_song[song_id].append(label)
    for song_id in DEVELOPMENT_SONG_IDS:
        for raw_path in supervision_paths_by_song[song_id]:
            path = Path(raw_path).expanduser().resolve()
            paths.append(path)
    try:
        for song_id in DEVELOPMENT_SONG_IDS:
            for raw_path in supervision_paths_by_song[song_id]:
                path = Path(raw_path).expanduser().resolve()
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, Mapping):
                    raise ValueError(f"supervision root must be an object: {path}")
                reference_id = str(payload.get("reference_id") or "").strip()
                if not reference_id or reference_id in song_by_reference:
                    raise ValueError(
                        f"supervision reference id is missing or duplicated: {path}"
                    )
                song_by_reference[reference_id] = song_id
                reject_unsupported_direct_fields(payload)
                collect_authoritative_shot_labels(payload, song_id)
        if any(not raw_transition_labels_by_song[song] for song in DEVELOPMENT_SONG_IDS):
            raise ValueError(
                "every development song needs nonzero authoritative hard_cut labels"
            )
        # The adapter derives duration/transition targets solely from the
        # immutable supervision JSON. Feature snapshots affect input features,
        # not these categorical labels, so preflight need not load video/crops.
        corpus = load_reference_training_corpus(
            paths,
            song_id_by_reference=song_by_reference,
            negative_samples_per_hold=int(
                legacy.PRODUCTION_SAMPLING_CONTRACT["negative_samples_per_hold"]
            ),
            boundary_guard_seconds=float(
                legacy.PRODUCTION_SAMPLING_CONTRACT["boundary_guard_seconds"]
            ),
            max_sources=int(legacy.PRODUCTION_SAMPLING_CONTRACT["max_sources"]),
        )
        split = production_song_split(
            corpus,
            held_out_song_ids=(XOXZ,),
            required_training_song_ids=DEVELOPMENT_TRAINING_SONG_IDS,
        )
        adapted_counts_by_song = {
            song: sum(
                target.transition_label is not None
                for example in corpus.examples
                if example.song_id == song
                for target in example.targets.targets
            )
            for song in DEVELOPMENT_SONG_IDS
        }
        raw_counts_by_song = {
            song: len(raw_transition_labels_by_song[song])
            for song in DEVELOPMENT_SONG_IDS
        }
        if adapted_counts_by_song != raw_counts_by_song:
            raise ValueError(
                "adapted transition metadata counts differ from authoritative "
                f"shot labels: raw={raw_counts_by_song}, adapted={adapted_counts_by_song}"
            )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ReleaseFoldPreflightError(
            f"structured head label preflight could not inspect bound labels: {exc}"
        ) from exc

    evidence = {
        "contract": validated,
        "training": {
            name: _categorical_target_coverage(split.train, field=field)
            for name, field in (
                ("duration", "duration_bucket"),
                ("transition", "transition_label"),
            )
        },
        "heldout": {
            name: _categorical_target_coverage(split.held_out, field=field)
            for name, field in (
                ("duration", "duration_bucket"),
                ("transition", "transition_label"),
            )
        },
        "raw_transition_label_audit": {
            song: {
                "label_count": len(raw_transition_labels_by_song[song]),
                "label_histogram": {
                    "hard_cut": raw_transition_labels_by_song[song].count(
                        "hard_cut"
                    )
                },
                "all_recognized_labels_are_hard_cut": True,
            }
            for song in DEVELOPMENT_SONG_IDS
        },
    }
    reasons: list[str] = []
    for side in ("training", "heldout"):
        duration = evidence[side]["duration"]
        required_duration = validated["duration"]
        if duration["observed_class_count"] < required_duration[
            "minimum_observed_classes"
        ]:
            reasons.append(f"{side}_duration_class_coverage_insufficient")
        entropy = duration["label_entropy_bits"]
        if (
            entropy is None
            or entropy < required_duration["minimum_label_entropy_bits"]
        ):
            reasons.append(f"{side}_duration_label_entropy_insufficient")
        transition = evidence[side]["transition"]
        count = transition.get("label_count")
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < validated["transition"]["minimum_label_count_per_partition"]
            or transition.get("label_histogram") != {"hard_cut": count}
        ):
            reasons.append(
                f"{side}_renderer_owned_transition_metadata_not_exact_hard_cut"
            )
        evidence[side]["transition"] = {
            "status": "skipped_non_authoritative",
            "reason": RENDERER_OWNED_LEARNED_EVALUATION_SKIP_REASON,
            "label_count": count,
            "label_histogram": transition.get("label_histogram"),
            "learned_accuracy_consumed": False,
            "learned_entropy_consumed": False,
        }
    evidence["passed"] = not reasons
    evidence["reasons"] = reasons
    if reasons:
        raise ReleaseFoldPreflightError(
            "structured head label preflight failed before training: "
            + ", ".join(reasons)
        )
    return evidence


def preflight_stage_edit_release_development(
    plan_path: Path,
) -> ReleaseFoldPreflight:
    payload = _strict_development_plan(plan_path)
    checkpoint = Path(str(payload["checkpoint_directory"])).resolve()
    legacy._validate_storage(checkpoint)
    args, references, supervision_paths = _manifest_cli_args(
        payload, DEVELOPMENT_SONG_IDS
    )
    _preflight_structured_head_label_coverage(
        supervision_paths,
        contract=payload["structured_head_preflight_contract"],
    )
    for song_id in DEVELOPMENT_TRAINING_SONG_IDS:
        args.extend(["--required-training-song", song_id])
    args.extend(
        [
            "--held-out-song",
            XOXZ,
            "--checkpoint-dir",
            str(checkpoint),
            "--negative-samples-per-hold",
            str(legacy.PRODUCTION_SAMPLING_CONTRACT["negative_samples_per_hold"]),
            "--boundary-guard-seconds",
            str(legacy.PRODUCTION_SAMPLING_CONTRACT["boundary_guard_seconds"]),
            "--max-sources",
            str(legacy.PRODUCTION_SAMPLING_CONTRACT["max_sources"]),
        ]
    )
    args.extend(_training_config_cli_args(_validate_training_config(payload["training_config"])))
    args.extend(_model_training_cli_args(payload["model_training_contract"]))
    if "--authorize-production-activation" in args or "--production-registry" in args:
        raise ReleaseFoldPreflightError("development plan acquired activation authority")
    return ReleaseFoldPreflight(
        stage="development",
        plan_path=Path(plan_path).expanduser().resolve(),
        plan_sha256=str(payload["plan_sha256"]),
        held_out_song_id=XOXZ,
        training_song_ids=DEVELOPMENT_TRAINING_SONG_IDS,
        checkpoint_directory=checkpoint,
        blind_consumption_marker=None,
        train_cli_args=tuple(args),
        reference_ids_by_song=references,
    )


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ReleaseFoldPreflightError(f"{name} is missing or unsafe: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReleaseFoldPreflightError(f"{name} root must be an object")
    return payload


def _assert_development_report(
    plan: Mapping[str, Any], report: Mapping[str, Any], selection: Mapping[str, Any]
) -> tuple[Path, str, str]:
    if (
        report.get("schema_version") != PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION
        or report.get("run_kind") != PRODUCTION_TRAINING_RUN_KIND
    ):
        raise ReleaseFoldPreflightError("development training report schema is invalid")
    if report.get("production_ready") is not False:
        raise ReleaseFoldPreflightError("development report must remain inactive")
    if report.get("production_readiness_reason") != "activation_not_explicitly_authorized":
        raise ReleaseFoldPreflightError("development quality gate did not pass inactive")
    checkpoint = _mapping(report.get("checkpoint"), "development checkpoint")
    if checkpoint.get("production_activation_authorized") is not False:
        raise ReleaseFoldPreflightError("development checkpoint was activation-authorized")
    heldout = _mapping(report.get("held_out_evaluation"), "development heldout")
    gate = _mapping(heldout.get("quality_gate"), "development quality gate")
    coverage = _mapping(
        report.get("structured_head_coverage_gate"), "structured head coverage gate"
    )
    if heldout.get("passed") is not True or gate.get("passed") is not True or coverage.get("passed") is not True:
        raise ReleaseFoldPreflightError("development XOXZ/coverage gate did not pass")
    if (
        tuple(heldout.get("training_song_ids") or ())
        != tuple(sorted(DEVELOPMENT_TRAINING_SONG_IDS))
        or heldout.get("held_out_song_ids") != [XOXZ]
        or heldout.get("cut_threshold_frozen_before_heldout_forward") is not True
    ):
        raise ReleaseFoldPreflightError("development song-heldout evidence is invalid")
    split = _mapping(report.get("split"), "development split")
    if (
        tuple(split.get("training_song_ids") or ())
        != tuple(sorted(DEVELOPMENT_TRAINING_SONG_IDS))
        or split.get("held_out_song_ids") != [XOXZ]
        or tuple(split.get("required_training_song_ids") or ())
        != tuple(sorted(DEVELOPMENT_TRAINING_SONG_IDS))
        or split.get("held_out_used_for_model_selection") is not False
    ):
        raise ReleaseFoldPreflightError("development split contract is invalid")
    if _mapping(report.get("optimization"), "development optimization").get(
        "unseen_song_used_for_model_selection"
    ) is not False:
        raise ReleaseFoldPreflightError("XOXZ was used for model selection")
    if (
        selection.get("heldout_metrics_present") is not False
        or selection.get("heldout_used_for_model_or_threshold_selection") is not False
        or selection.get("style_or_attitude_configuration_mutable_after_freeze") is not False
        or selection.get("held_out_song_ids_reserved_for_one_forward") != [XOXZ]
        or tuple(selection.get("training_song_ids") or ())
        != tuple(sorted(DEVELOPMENT_TRAINING_SONG_IDS))
    ):
        raise ReleaseFoldPreflightError("development selection freeze is invalid")
    config = _mapping(report.get("model"), "development model").get("config")
    if not isinstance(config, Mapping):
        raise ReleaseFoldPreflightError("development model config is missing")
    config_fingerprint = _canonical_sha256(config)
    planned_model_training = _validate_model_training_contract(
        plan.get("model_training_contract")
    )
    if (
        report.get("transition_authority_contract")
        != planned_model_training["transition_authority_contract"]
        or report.get("transition_authority_contract_sha256")
        != planned_model_training["transition_authority_contract_sha256"]
    ):
        raise ReleaseFoldPreflightError(
            "development transition authority differs from frozen plan"
        )
    if (
        config != planned_model_training["model_config"]
        or selection.get("model_config") != config
        or selection.get("model_config_fingerprint_sha256") != config_fingerprint
    ):
        raise ReleaseFoldPreflightError("development model config freeze mismatch")
    expected_training = _validate_training_config(plan.get("training_config"))
    optimization = _mapping(report.get("optimization"), "development optimization")
    cut_imbalance = _mapping(
        report.get("cut_imbalance_contract"), "cut imbalance contract"
    )
    applied_positive_weight = cut_imbalance.get("applied_positive_weight")
    if not isinstance(applied_positive_weight, (int, float)) or isinstance(
        applied_positive_weight, bool
    ):
        raise ReleaseFoldPreflightError(
            "development applied cut-positive weight is missing"
        )
    expected_effective_loss = dict(planned_model_training["base_loss_contract"])
    expected_effective_loss["cut_positive_weight"] = float(
        applied_positive_weight
    )
    expected_dynamic_loss = deepcopy(
        planned_model_training["dynamic_crop_loss_contract"]
    )
    expected_dynamic_loss["base"]["cut_positive_weight"] = float(
        applied_positive_weight
    )
    if (
        optimization.get("effective_loss_contract") != expected_effective_loss
        or optimization.get("effective_dynamic_crop_loss_contract")
        != expected_dynamic_loss
    ):
        raise ReleaseFoldPreflightError(
            "development effective loss/source objective differs from plan"
        )
    schedule = _mapping(optimization.get("learning_rate_schedule"), "learning-rate schedule")
    observed_training = {
        "epochs": optimization.get("requested_epochs"),
        "patience": optimization.get("early_stopping_patience"),
        "batch_size": optimization.get("batch_size"),
        "learning_rate": optimization.get("learning_rate"),
        "weight_decay": optimization.get("weight_decay"),
        "gradient_clip_norm": optimization.get("gradient_clip_norm"),
        "maximum_cut_positive_weight": _mapping(
            report.get("cut_imbalance_contract"), "cut imbalance contract"
        ).get("maximum_positive_weight"),
        "cut_pretrain_steps": schedule.get("cut_pretraining_steps"),
        "cut_pretrain_learning_rate": schedule.get("cut_pretraining_peak_learning_rate"),
        "cut_pretrain_window_length": schedule.get("cut_pretraining_window_length"),
        "cut_pretrain_warmup_steps": schedule.get("cut_pretraining_warmup_steps"),
        "finetune_warmup_steps": schedule.get("finetune_warmup_steps"),
        "minimum_learning_rate_ratio": schedule.get("minimum_learning_rate_ratio"),
        "seed": optimization.get("seed"),
        "device": "gpu",
    }
    if observed_training != expected_training:
        raise ReleaseFoldPreflightError("development optimization differs from plan")
    checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
    weight_sha = checkpoint.get("weight_sha256")
    filename = checkpoint.get("weight_filename")
    if not checkpoint_id or not _is_sha256(weight_sha):
        raise ReleaseFoldPreflightError("development checkpoint identity is invalid")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ReleaseFoldPreflightError("development checkpoint weight filename is invalid")
    if (
        heldout.get("checkpoint_id") != checkpoint_id
        or heldout.get("weight_sha256") != weight_sha
        or selection.get("checkpoint_id") != checkpoint_id
        or selection.get("checkpoint_weight_sha256") != weight_sha
    ):
        raise ReleaseFoldPreflightError("development checkpoint binding differs")
    return Path(filename), checkpoint_id, str(weight_sha)


def freeze_stage_edit_release_development(
    plan_path: Path, *, output_directory: Path
) -> Path:
    """Freeze a passing inactive XOXZ development run before ATTITUDE planning."""

    resolved_plan = Path(plan_path).expanduser().resolve()
    plan = _strict_development_plan(resolved_plan)
    checkpoint = Path(str(plan["checkpoint_directory"])).resolve()
    report_path = checkpoint / PRODUCTION_TRAINING_REPORT_FILENAME
    selection_path = checkpoint / STAGE_EDIT_SELECTION_FREEZE_FILENAME
    report = _read_json_object(report_path, "development training report")
    selection = _read_json_object(selection_path, "development selection freeze")
    weight_name, checkpoint_id, expected_weight_sha = _assert_development_report(
        plan, report, selection
    )
    weight_path = checkpoint / weight_name
    if _file_sha256(weight_path, "development checkpoint weights") != expected_weight_sha:
        raise ReleaseFoldPreflightError("development checkpoint weights changed")
    report_selection = _mapping(
        report.get("selection_freeze_evidence"), "report selection-freeze evidence"
    )
    selection_sha = _file_sha256(selection_path, "development selection freeze")
    if (
        report_selection.get("path") != str(selection_path)
        or report_selection.get("file_sha256") != selection_sha
        or report_selection.get("written_before_heldout_forward") is not True
    ):
        raise ReleaseFoldPreflightError("report does not bind the selection freeze")
    config = dict(_mapping(report["model"], "development model")["config"])
    unsigned = {
        "schema_version": RELEASE_PLAN_SCHEMA_VERSION,
        "analysis_basis": DEVELOPMENT_FREEZE_ANALYSIS_BASIS,
        "stage": "development_frozen",
        "development_plan": _pointer(resolved_plan, "development plan"),
        "development_checkpoint_directory": str(checkpoint),
        "development_report": _pointer(report_path, "development report"),
        "selection_freeze": _pointer(selection_path, "development selection freeze"),
        "checkpoint_weights": _pointer(weight_path, "development checkpoint weights"),
        "checkpoint_id": checkpoint_id,
        "model_config": config,
        "model_config_fingerprint_sha256": _canonical_sha256(config),
        "model_training_contract": dict(plan["model_training_contract"]),
        "structured_head_preflight_contract": dict(
            plan["structured_head_preflight_contract"]
        ),
        "training_config": dict(plan["training_config"]),
        "development_training_song_ids": list(DEVELOPMENT_TRAINING_SONG_IDS),
        "development_held_out_song_id": XOXZ,
        "development_quality_gate_passed": True,
        "production_activation_authorized": False,
        "attitude_evidence_opened": False,
    }
    return _write_content_addressed(
        unsigned,
        output_directory=output_directory,
        prefix="stageedit-release-development-freeze",
        hash_field="freeze_sha256",
    )


_DEVELOPMENT_FREEZE_KEYS = {
    "schema_version",
    "analysis_basis",
    "stage",
    "development_plan",
    "development_checkpoint_directory",
    "development_report",
    "selection_freeze",
    "checkpoint_weights",
    "checkpoint_id",
    "model_config",
    "model_config_fingerprint_sha256",
    "model_training_contract",
    "structured_head_preflight_contract",
    "training_config",
    "development_training_song_ids",
    "development_held_out_song_id",
    "development_quality_gate_passed",
    "production_activation_authorized",
    "attitude_evidence_opened",
    "freeze_sha256",
}


def _strict_development_freeze(
    freeze_path: Path, *, expected_plan_path: Path | None = None
) -> dict[str, Any]:
    payload = _strict_content_addressed(
        freeze_path,
        expected_keys=_DEVELOPMENT_FREEZE_KEYS,
        schema_basis=DEVELOPMENT_FREEZE_ANALYSIS_BASIS,
        prefix="stageedit-release-development-freeze",
        hash_field="freeze_sha256",
    )
    if (
        payload.get("stage") != "development_frozen"
        or payload.get("development_training_song_ids")
        != list(DEVELOPMENT_TRAINING_SONG_IDS)
        or payload.get("development_held_out_song_id") != XOXZ
        or payload.get("development_quality_gate_passed") is not True
        or payload.get("production_activation_authorized") is not False
        or payload.get("attitude_evidence_opened") is not False
    ):
        raise ReleaseFoldPreflightError("development freeze policy is invalid")
    plan_path = _verify_pointer(payload.get("development_plan"), "development plan")
    if expected_plan_path is not None and plan_path != Path(expected_plan_path).resolve():
        raise ReleaseFoldPreflightError("development freeze binds a different plan")
    plan = _strict_development_plan(plan_path)
    if payload.get("training_config") != plan.get("training_config"):
        raise ReleaseFoldPreflightError("development training config freeze mismatch")
    if payload.get("model_training_contract") != plan.get(
        "model_training_contract"
    ):
        raise ReleaseFoldPreflightError("development model/loss contract freeze mismatch")
    _validate_model_training_contract(payload.get("model_training_contract"))
    if payload.get("structured_head_preflight_contract") != plan.get(
        "structured_head_preflight_contract"
    ):
        raise ReleaseFoldPreflightError(
            "development structured-head contract freeze mismatch"
        )
    _validate_structured_head_preflight_contract(
        payload.get("structured_head_preflight_contract")
    )
    checkpoint = Path(str(payload.get("development_checkpoint_directory") or "")).resolve()
    if checkpoint != Path(str(plan["checkpoint_directory"])).resolve():
        raise ReleaseFoldPreflightError("development checkpoint path freeze mismatch")
    report_path = _verify_pointer(payload.get("development_report"), "development report")
    selection_path = _verify_pointer(payload.get("selection_freeze"), "development selection freeze")
    weight_path = _verify_pointer(payload.get("checkpoint_weights"), "development checkpoint weights")
    if report_path != checkpoint / PRODUCTION_TRAINING_REPORT_FILENAME:
        raise ReleaseFoldPreflightError("development report path is not canonical")
    if selection_path != checkpoint / STAGE_EDIT_SELECTION_FREEZE_FILENAME:
        raise ReleaseFoldPreflightError("development selection-freeze path is not canonical")
    if weight_path.parent != checkpoint:
        raise ReleaseFoldPreflightError("development weights escaped checkpoint directory")
    report = _read_json_object(report_path, "development report")
    selection = _read_json_object(selection_path, "development selection freeze")
    observed_weight_name, checkpoint_id, weight_sha = _assert_development_report(
        plan, report, selection
    )
    if weight_path != checkpoint / observed_weight_name:
        raise ReleaseFoldPreflightError("development weight path differs from report")
    if payload.get("checkpoint_id") != checkpoint_id:
        raise ReleaseFoldPreflightError("development checkpoint id freeze mismatch")
    if _mapping(payload.get("checkpoint_weights"), "weights").get("file_sha256") != weight_sha:
        raise ReleaseFoldPreflightError("development checkpoint weight hash mismatch")
    model_config = payload.get("model_config")
    fingerprint = payload.get("model_config_fingerprint_sha256")
    if not isinstance(model_config, Mapping) or _canonical_sha256(model_config) != fingerprint:
        raise ReleaseFoldPreflightError("development frozen model config is invalid")
    model_training = _validate_model_training_contract(
        payload.get("model_training_contract")
    )
    if model_config != model_training["model_config"]:
        raise ReleaseFoldPreflightError(
            "frozen model config differs from bound model/training contract"
        )
    return payload


def create_stage_edit_release_final_plan(
    *,
    development_plan_path: Path,
    development_freeze_path: Path,
    materialization_manifests: Mapping[str, Path],
    feature_manifests: Mapping[str, Path],
    dynamic_crop_manifests: Mapping[str, Path],
    checkpoint_directory: Path,
    output_directory: Path,
    professional_montage_grammar_profile: Path,
    production_registry: Path | None = None,
) -> Path:
    """Create the final ATTITUDE plan only after a valid development freeze."""

    resolved_development = Path(development_plan_path).expanduser().resolve()
    development = _strict_development_plan(resolved_development)
    freeze = _strict_development_freeze(
        development_freeze_path, expected_plan_path=resolved_development
    )
    checkpoint = Path(checkpoint_directory).expanduser().resolve()
    volume = legacy.PRODUCTION_VOLUME_ROOT.resolve()
    if volume != checkpoint and volume not in checkpoint.parents:
        raise ReleaseFoldPreflightError(f"final checkpoint must be on {volume}: {checkpoint}")
    montage_profile_path = Path(
        professional_montage_grammar_profile
    ).expanduser()
    if not montage_profile_path.is_absolute():
        raise ReleaseFoldPreflightError(
            "professional montage grammar profile path must be explicit and absolute"
        )
    try:
        montage_policy = load_professional_montage_policy(montage_profile_path)
    except ProfessionalMontagePolicyError as exc:
        raise ReleaseFoldPreflightError(
            "professional montage grammar profile is missing, unsafe, or invalid: "
            f"{exc}"
        ) from exc
    montage_profile_identity = montage_policy.profile_identity
    if montage_profile_identity.get("path") != str(montage_profile_path):
        raise ReleaseFoldPreflightError(
            "professional montage grammar profile path identity changed"
        )
    records = _song_records(
        FINAL_SONG_IDS,
        materialization_manifests=materialization_manifests,
        feature_manifests=feature_manifests,
        dynamic_crop_manifests=dynamic_crop_manifests,
    )
    development_records = _indexed_song_records(development, DEVELOPMENT_SONG_IDS)
    final_records = {str(item["song_id"]): item for item in records}
    for song_id in DEVELOPMENT_SONG_IDS:
        if final_records[song_id] != development_records[song_id]:
            raise ReleaseFoldPreflightError(
                f"final plan changed frozen development evidence for {song_id}"
            )
    registry = (
        str(Path(production_registry).expanduser().resolve())
        if production_registry is not None
        else None
    )
    # The marker is keyed by the immutable development freeze, not by the
    # caller-selected final checkpoint path.  Choosing another output directory
    # therefore cannot reset the one-shot ATTITUDE budget.
    freeze_digest = str(freeze["freeze_sha256"])
    development_checkpoint = Path(
        str(freeze["development_checkpoint_directory"])
    ).resolve()
    marker = development_checkpoint.parent / (
        f"stageedit-release-{freeze_digest[:16]}.attitude-blind-consumed.json"
    )
    unsigned = {
        "schema_version": FINAL_RELEASE_PLAN_SCHEMA_VERSION,
        "analysis_basis": FINAL_PLAN_ANALYSIS_BASIS,
        "stage": "final_blind",
        "release_contract": _release_contract(
            training_song_ids=FINAL_TRAINING_SONG_IDS,
            held_out_song_id=ATTITUDE,
            stage="final_blind",
        ),
        "required_volume": str(legacy.PRODUCTION_VOLUME_ROOT),
        "minimum_free_bytes": legacy.MINIMUM_FREE_BYTES,
        "training_config": dict(freeze["training_config"]),
        "model_config_fingerprint_sha256": freeze[
            "model_config_fingerprint_sha256"
        ],
        "model_training_contract": dict(freeze["model_training_contract"]),
        "structured_head_preflight_contract": dict(
            freeze["structured_head_preflight_contract"]
        ),
        "development_plan": _pointer(resolved_development, "development plan"),
        "development_freeze": _pointer(
            Path(development_freeze_path).expanduser().resolve(),
            "development freeze",
        ),
        "checkpoint_directory": str(checkpoint),
        "blind_consumption_marker": str(marker),
        "professional_montage_grammar_profile": montage_profile_identity,
        "production_registry": registry,
        "songs": records,
    }
    return _write_content_addressed(
        unsigned,
        output_directory=output_directory,
        prefix="stageedit-release-attitude-blind-plan",
        hash_field="plan_sha256",
    )


_FINAL_PLAN_KEYS = {
    "schema_version",
    "analysis_basis",
    "stage",
    "release_contract",
    "required_volume",
    "minimum_free_bytes",
    "training_config",
    "model_config_fingerprint_sha256",
    "model_training_contract",
    "structured_head_preflight_contract",
    "development_plan",
    "development_freeze",
    "checkpoint_directory",
    "blind_consumption_marker",
    "professional_montage_grammar_profile",
    "production_registry",
    "songs",
    "plan_sha256",
}


def _strict_final_plan(path: Path) -> dict[str, Any]:
    payload = _strict_content_addressed(
        path,
        expected_keys=_FINAL_PLAN_KEYS,
        schema_basis=FINAL_PLAN_ANALYSIS_BASIS,
        prefix="stageedit-release-attitude-blind-plan",
        hash_field="plan_sha256",
        expected_schema_version=FINAL_RELEASE_PLAN_SCHEMA_VERSION,
    )
    if payload.get("stage") != "final_blind" or payload.get("release_contract") != _release_contract(
        training_song_ids=FINAL_TRAINING_SONG_IDS,
        held_out_song_id=ATTITUDE,
        stage="final_blind",
    ):
        raise ReleaseFoldPreflightError("final blind release contract is not exact")
    if (
        payload.get("required_volume") != str(legacy.PRODUCTION_VOLUME_ROOT)
        or payload.get("minimum_free_bytes") != legacy.MINIMUM_FREE_BYTES
    ):
        raise ReleaseFoldPreflightError("final storage contract mismatch")
    development_path = _verify_pointer(payload.get("development_plan"), "development plan")
    development = _strict_development_plan(development_path)
    freeze_path = _verify_pointer(payload.get("development_freeze"), "development freeze")
    freeze = _strict_development_freeze(
        freeze_path, expected_plan_path=development_path
    )
    if payload.get("training_config") != freeze.get("training_config"):
        raise ReleaseFoldPreflightError("final training config differs from development freeze")
    if payload.get("model_config_fingerprint_sha256") != freeze.get(
        "model_config_fingerprint_sha256"
    ):
        raise ReleaseFoldPreflightError("final model config differs from development freeze")
    if payload.get("model_training_contract") != freeze.get(
        "model_training_contract"
    ):
        raise ReleaseFoldPreflightError(
            "final model/loss contract differs from development freeze"
        )
    _validate_model_training_contract(payload.get("model_training_contract"))
    if payload.get("structured_head_preflight_contract") != freeze.get(
        "structured_head_preflight_contract"
    ):
        raise ReleaseFoldPreflightError(
            "final structured-head contract differs from development freeze"
        )
    _validate_structured_head_preflight_contract(
        payload.get("structured_head_preflight_contract")
    )
    final_records = _indexed_song_records(payload, FINAL_SONG_IDS)
    development_records = _indexed_song_records(development, DEVELOPMENT_SONG_IDS)
    for song_id in DEVELOPMENT_SONG_IDS:
        if final_records[song_id] != development_records[song_id]:
            raise ReleaseFoldPreflightError(
                f"final plan changed development evidence for {song_id}"
            )
    checkpoint = Path(str(payload.get("checkpoint_directory") or "")).resolve()
    marker = Path(str(payload.get("blind_consumption_marker") or "")).resolve()
    development_checkpoint = Path(
        str(freeze["development_checkpoint_directory"])
    ).resolve()
    expected_marker = development_checkpoint.parent / (
        f"stageedit-release-{str(freeze['freeze_sha256'])[:16]}."
        "attitude-blind-consumed.json"
    )
    if marker != expected_marker:
        raise ReleaseFoldPreflightError("blind consumption marker path is not canonical")
    montage_profile = _mapping(
        payload.get("professional_montage_grammar_profile"),
        "professional montage grammar profile",
    )
    if set(montage_profile) != {
        "analysis_basis",
        "schema_version",
        "path",
        "file_sha256",
        "content_sha256",
    }:
        raise ReleaseFoldPreflightError(
            "professional montage grammar profile identity schema mismatch"
        )
    raw_profile_path = montage_profile.get("path")
    if not isinstance(raw_profile_path, str) or not Path(raw_profile_path).is_absolute():
        raise ReleaseFoldPreflightError(
            "professional montage grammar profile path is invalid"
        )
    expected_profile_file_sha256 = montage_profile.get("file_sha256")
    if not _is_sha256(expected_profile_file_sha256):
        raise ReleaseFoldPreflightError(
            "professional montage grammar profile file hash is invalid"
        )
    try:
        actual_montage_policy = load_professional_montage_policy(
            Path(raw_profile_path),
            expected_file_sha256=str(expected_profile_file_sha256),
        )
    except ProfessionalMontagePolicyError as exc:
        raise ReleaseFoldPreflightError(
            "professional montage grammar profile is missing, changed, or invalid: "
            f"{exc}"
        ) from exc
    if actual_montage_policy.profile_identity != dict(montage_profile):
        raise ReleaseFoldPreflightError(
            "professional montage grammar profile identity changed"
        )
    registry = payload.get("production_registry")
    if registry is not None and (
        not isinstance(registry, str) or not Path(registry).is_absolute()
    ):
        raise ReleaseFoldPreflightError("production registry path is invalid")
    return payload


def preflight_stage_edit_release_final(plan_path: Path) -> ReleaseFoldPreflight:
    """Validate frozen evidence before reading ATTITUDE manifests for one run."""

    payload = _strict_final_plan(plan_path)
    checkpoint = Path(str(payload["checkpoint_directory"])).resolve()
    legacy._validate_storage(checkpoint)
    marker = Path(str(payload["blind_consumption_marker"])).resolve()
    if marker.exists() or marker.is_symlink():
        raise ReleaseFoldPreflightError(
            "ATTITUDE blind evaluation was already consumed for this release"
        )
    args, references, _ = _manifest_cli_args(payload, FINAL_SONG_IDS)
    for song_id in FINAL_TRAINING_SONG_IDS:
        args.extend(["--required-training-song", song_id])
    args.extend(
        [
            "--held-out-song",
            ATTITUDE,
            "--checkpoint-dir",
            str(checkpoint),
            "--negative-samples-per-hold",
            str(legacy.PRODUCTION_SAMPLING_CONTRACT["negative_samples_per_hold"]),
            "--boundary-guard-seconds",
            str(legacy.PRODUCTION_SAMPLING_CONTRACT["boundary_guard_seconds"]),
            "--max-sources",
            str(legacy.PRODUCTION_SAMPLING_CONTRACT["max_sources"]),
        ]
    )
    args.extend(_training_config_cli_args(_validate_training_config(payload["training_config"])))
    args.extend(_model_training_cli_args(payload["model_training_contract"]))
    args.extend(
        [
            "--professional-montage-grammar-profile",
            str(payload["professional_montage_grammar_profile"]["path"]),
        ]
    )
    args.append("--authorize-production-activation")
    if payload.get("production_registry") is not None:
        args.extend(["--production-registry", str(payload["production_registry"])])
    return ReleaseFoldPreflight(
        stage="final_blind",
        plan_path=Path(plan_path).expanduser().resolve(),
        plan_sha256=str(payload["plan_sha256"]),
        held_out_song_id=ATTITUDE,
        training_song_ids=FINAL_TRAINING_SONG_IDS,
        checkpoint_directory=checkpoint,
        blind_consumption_marker=marker,
        train_cli_args=tuple(args),
        reference_ids_by_song=references,
    )


def launch_stage_edit_release_final(plan_path: Path) -> int:
    """Consume the ATTITUDE blind run once, then delegate to the strict CLI."""

    preflight = preflight_stage_edit_release_final(plan_path)
    assert preflight.blind_consumption_marker is not None
    marker = preflight.blind_consumption_marker
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker_payload = {
        "schema_version": 1,
        "contract": "attitude_one_shot_blind_evaluation_consumed_v1",
        "final_plan_path": str(preflight.plan_path),
        "final_plan_sha256": preflight.plan_sha256,
        "held_out_song_id": ATTITUDE,
        "held_out_used_for_model_or_threshold_selection": False,
        "retry_allowed": False,
    }
    encoded = json.dumps(
        marker_payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    try:
        with marker.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ReleaseFoldPreflightError(
            "ATTITUDE blind evaluation was already consumed"
        ) from exc
    # The general CLI registers only when every fixed held-out gate passes.
    # A failed final blind run remains inactive and the marker remains consumed.
    return production_train_main(preflight.train_cli_args)


def _keyed_paths(
    values: Sequence[str], *, expected_song_ids: Sequence[str], option: str
) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        song_id, separator, raw_path = value.partition("=")
        song_id = song_id.strip()
        raw_path = raw_path.strip()
        if not separator or not song_id or not raw_path:
            raise ReleaseFoldPreflightError(
                f"{option} must use SONG_ID=/absolute/path.json"
            )
        if song_id in output:
            raise ReleaseFoldPreflightError(f"duplicate {option} song: {song_id}")
        output[song_id] = Path(raw_path).expanduser().resolve()
    if set(output) != set(expected_song_ids):
        raise ReleaseFoldPreflightError(
            f"{option} must name the exact release songs"
        )
    return output


def _add_manifest_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--materialization-manifest", action="append", required=True)
    parser.add_argument("--feature-manifest", action="append", required=True)
    parser.add_argument("--dynamic-crop-manifest", action="append", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen IVE StageEdit development-to-ATTITUDE release protocol"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create_dev = commands.add_parser("create-development")
    _add_manifest_options(create_dev)
    create_dev.add_argument("--checkpoint-dir", required=True, type=Path)
    create_dev.add_argument("--output-directory", required=True, type=Path)
    create_dev.add_argument(
        "--model-training-contract",
        type=Path,
        help=(
            "Optional probe-selected model/loss contract JSON. When omitted, "
            "the current release candidate is still serialized and hash-bound; "
            "a later capacity probe may supply its selected contract explicitly."
        ),
    )
    for name in ("preflight-development", "launch-development"):
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True, type=Path)
    freeze = commands.add_parser("freeze-development")
    freeze.add_argument("--plan", required=True, type=Path)
    freeze.add_argument("--output-directory", required=True, type=Path)
    create_final = commands.add_parser("create-final")
    _add_manifest_options(create_final)
    create_final.add_argument("--development-plan", required=True, type=Path)
    create_final.add_argument("--development-freeze", required=True, type=Path)
    create_final.add_argument("--checkpoint-dir", required=True, type=Path)
    create_final.add_argument("--output-directory", required=True, type=Path)
    create_final.add_argument(
        "--professional-montage-grammar-profile",
        required=True,
        type=Path,
    )
    create_final.add_argument("--production-registry", type=Path)
    for name in ("preflight-final", "launch-final"):
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "create-development":
        expected = DEVELOPMENT_SONG_IDS
        path = create_stage_edit_release_development_plan(
            materialization_manifests=_keyed_paths(
                args.materialization_manifest,
                expected_song_ids=expected,
                option="--materialization-manifest",
            ),
            feature_manifests=_keyed_paths(
                args.feature_manifest,
                expected_song_ids=expected,
                option="--feature-manifest",
            ),
            dynamic_crop_manifests=_keyed_paths(
                args.dynamic_crop_manifest,
                expected_song_ids=expected,
                option="--dynamic-crop-manifest",
            ),
            checkpoint_directory=args.checkpoint_dir,
            output_directory=args.output_directory,
            model_training_contract=(
                _read_json_object(
                    args.model_training_contract.expanduser().resolve(),
                    "probe-selected model/training contract",
                )
                if args.model_training_contract is not None
                else None
            ),
        )
        print(json.dumps({"plan_path": str(path)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "preflight-development":
        result = preflight_stage_edit_release_development(args.plan)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "launch-development":
        result = preflight_stage_edit_release_development(args.plan)
        return production_train_main(result.train_cli_args)
    if args.command == "freeze-development":
        path = freeze_stage_edit_release_development(
            args.plan, output_directory=args.output_directory
        )
        print(json.dumps({"freeze_path": str(path)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "create-final":
        expected = FINAL_SONG_IDS
        path = create_stage_edit_release_final_plan(
            development_plan_path=args.development_plan,
            development_freeze_path=args.development_freeze,
            materialization_manifests=_keyed_paths(
                args.materialization_manifest,
                expected_song_ids=expected,
                option="--materialization-manifest",
            ),
            feature_manifests=_keyed_paths(
                args.feature_manifest,
                expected_song_ids=expected,
                option="--feature-manifest",
            ),
            dynamic_crop_manifests=_keyed_paths(
                args.dynamic_crop_manifest,
                expected_song_ids=expected,
                option="--dynamic-crop-manifest",
            ),
            checkpoint_directory=args.checkpoint_dir,
            output_directory=args.output_directory,
            professional_montage_grammar_profile=(
                args.professional_montage_grammar_profile
            ),
            production_registry=args.production_registry,
        )
        print(json.dumps({"plan_path": str(path)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "preflight-final":
        result = preflight_stage_edit_release_final(args.plan)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0
    return launch_stage_edit_release_final(args.plan)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ATTITUDE",
    "DEFAULT_RELEASE_TRAINING_CONFIG",
    "DEVELOPMENT_SONG_IDS",
    "DEVELOPMENT_TRAINING_SONG_IDS",
    "FINAL_SONG_IDS",
    "FINAL_TRAINING_SONG_IDS",
    "ReleaseFoldPreflight",
    "ReleaseFoldPreflightError",
    "create_stage_edit_release_development_plan",
    "create_stage_edit_release_final_plan",
    "freeze_stage_edit_release_development",
    "launch_stage_edit_release_final",
    "preflight_stage_edit_release_development",
    "preflight_stage_edit_release_final",
]
