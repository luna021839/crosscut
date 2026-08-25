"""Fail-closed IVE StageEdit development-to-blind-release launcher.

This module intentionally sits beside, rather than replaces, the historical
three-song production-fold launcher.  It implements the release protocol used
once REBEL HEART is available:

* development: BANG BANG + BLACKHOLE + REBEL HEART train, XOXZ held out;
* immutable development freeze: report, selection freeze, model config and
  selected checkpoint weights are all rebound by SHA-256; and
* final: the four development songs train, while one new, explicitly named IVE
  song is evaluated exactly once as the blind held-out song.  Only this final
  run carries activation authority.

The low-level trainer remains :mod:`stage_edit_production_train_cli`.  In
particular, checkpoint and cut-threshold selection still happen on whole
references from training songs before the held-out forward pass.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
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
from .stage_edit_dynamic_crop_head_v3 import (
    DYNAMIC_CROP_V3_LOSS_CONTRACT,
    DYNAMIC_CROP_V3_MODEL_CONTRACT,
    DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
    DYNAMIC_CROP_V3_SCHEMA_VERSION,
    DYNAMIC_CROP_V3_TARGET_CONTRACT,
    DynamicCropHeadV3Config,
    DynamicCropV3LossContract,
)
from .stage_edit_dynamic_crop_v3_capacity_smoke import (
    V3_PROBE_TRAINING_SCHEDULE,
)
from .stage_edit_dynamic_crop_v3_final_release_launcher import (
    DynamicCropV3FinalReleaseError,
    load_dynamic_crop_v3_final_fit_plan,
    load_dynamic_crop_v3_final_fit_receipt,
    load_dynamic_crop_v3_release_freeze,
)
from .stage_edit_dynamic_crop_capacity_probe import (
    DynamicCropCapacityProbeError,
    load_dynamic_crop_capacity_selection,
)
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    PRODUCTION_TRAINING_REPORT_FILENAME,
    PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION,
    PRODUCTION_TRAINING_RUN_KIND,
    StageEditLossContract,
    StageEditTransformerConfig,
)
from .stage_edit_transition_authority import (
    RENDERER_OWNED_ALLOWED_MODEL_LABELS,
    RENDERER_OWNED_LEARNED_EVALUATION_SKIP_REASON,
    RENDERER_OWNED_TRANSITION_AUTHORITY,
    StageEditTransitionAuthority,
)
from .stage_edit_continuity_authority import (
    HANDCRAFTED_CONTINUITY_AUTHORITY,
)


RELEASE_PLAN_SCHEMA_VERSION = 4
RELEASE_PLAN_V3_SCHEMA_VERSION = 5
DEVELOPMENT_FREEZE_SCHEMA_VERSION = 5
DEVELOPMENT_FREEZE_V3_SCHEMA_VERSION = 6
FINAL_RELEASE_PLAN_SCHEMA_VERSION = 7
FINAL_RELEASE_PLAN_V3_SCHEMA_VERSION = 8
DEVELOPMENT_PLAN_ANALYSIS_BASIS = "ive_stageedit_release_development_plan_v4"
DEVELOPMENT_PLAN_V3_ANALYSIS_BASIS = (
    "ive_stageedit_dynamic_crop_v3_release_development_plan_v5"
)
DEVELOPMENT_FREEZE_ANALYSIS_BASIS = "ive_stageedit_development_freeze_v5"
DEVELOPMENT_FREEZE_V3_ANALYSIS_BASIS = (
    "ive_stageedit_dynamic_crop_v3_development_freeze_v6"
)
FINAL_PLAN_ANALYSIS_BASIS = "ive_stageedit_generic_blind_release_plan_v7"
FINAL_PLAN_V3_ANALYSIS_BASIS = (
    "ive_stageedit_dynamic_crop_v3_generic_blind_release_plan_v8"
)
MODEL_TRAINING_CONTRACT_VERSION = 2
MODEL_TRAINING_CONTRACT_V3_VERSION = 3
STRUCTURED_HEAD_PREFLIGHT_CONTRACT_VERSION = 2
CAPACITY_SELECTION_BINDING_VERSION = 1
V3_RELEASE_READINESS_BINDING_VERSION = 2
V3_RELEASE_READINESS_KIND = (
    "dynamic_crop_v3_loso_freeze_and_inactive_final_fit_v1"
)
V3_FINAL_FIT_CHECKPOINT_USAGE = (
    "readiness_only_no_optimizer_initialization_v1"
)
DEFERRED_BLIND_MANIFEST_CONTRACT = (
    "absolute_paths_only_until_blind_consumption_marker_v1"
)

BANG_BANG = "IVE-BANG-BANG"
BLACKHOLE = "IVE-BLACKHOLE"
REBEL_HEART = "IVE-REBEL-HEART"
XOXZ = "IVE-XOXZ"
ATTITUDE = "IVE-ATTITUDE"

DEVELOPMENT_TRAINING_SONG_IDS = (BANG_BANG, BLACKHOLE, REBEL_HEART)
DEVELOPMENT_SONG_IDS = (*DEVELOPMENT_TRAINING_SONG_IDS, XOXZ)
FINAL_TRAINING_SONG_IDS = (*DEVELOPMENT_TRAINING_SONG_IDS, XOXZ)
QUARANTINED_BLIND_SONG_IDS = frozenset({ATTITUDE})

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

DEFAULT_RELEASE_TRAINING_CONFIG_V3: Mapping[str, Any] = {
    **dict(DEFAULT_RELEASE_TRAINING_CONFIG),
    "epochs": V3_PROBE_TRAINING_SCHEDULE["requested_epochs"],
    "patience": V3_PROBE_TRAINING_SCHEDULE["early_stopping_patience"],
    "batch_size": V3_PROBE_TRAINING_SCHEDULE["batch_size"],
    "learning_rate": V3_PROBE_TRAINING_SCHEDULE["learning_rate"],
    "weight_decay": V3_PROBE_TRAINING_SCHEDULE["weight_decay"],
    "gradient_clip_norm": V3_PROBE_TRAINING_SCHEDULE[
        "gradient_clip_norm"
    ],
    "cut_pretrain_steps": V3_PROBE_TRAINING_SCHEDULE[
        "cut_pretraining_steps"
    ],
    "cut_pretrain_learning_rate": V3_PROBE_TRAINING_SCHEDULE[
        "cut_pretraining_peak_learning_rate"
    ],
    "cut_pretrain_window_length": V3_PROBE_TRAINING_SCHEDULE[
        "cut_pretraining_window_length"
    ],
    "cut_pretrain_warmup_steps": V3_PROBE_TRAINING_SCHEDULE[
        "cut_pretraining_warmup_steps"
    ],
    "finetune_warmup_steps": V3_PROBE_TRAINING_SCHEDULE[
        "finetune_warmup_steps"
    ],
    "minimum_learning_rate_ratio": V3_PROBE_TRAINING_SCHEDULE[
        "minimum_learning_rate_ratio"
    ],
    "seed": V3_PROBE_TRAINING_SCHEDULE["seed"],
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
    capacity_selection: Mapping[str, Any]
    blind_consumption_marker: Path | None
    held_out_manifest_inspection_deferred: bool
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
            "capacity_selection": dict(self.capacity_selection),
            "blind_consumption_marker": (
                str(self.blind_consumption_marker)
                if self.blind_consumption_marker is not None
                else None
            ),
            "held_out_manifest_inspection_deferred": (
                self.held_out_manifest_inspection_deferred
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
    common_keys = {
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
    version = contract.get("contract_version")
    v3_keys = {
        "dynamic_crop_version",
        "dynamic_crop_config",
        "dynamic_crop_config_sha256",
        "v3_training_schedule",
        "v3_training_schedule_sha256",
    }
    expected_keys = (
        common_keys | v3_keys
        if version == MODEL_TRAINING_CONTRACT_V3_VERSION
        else common_keys
    )
    if set(contract) != expected_keys:
        raise ReleaseFoldPreflightError(
            "release model/training contract keys are not exact"
        )
    if version not in {
        MODEL_TRAINING_CONTRACT_VERSION,
        MODEL_TRAINING_CONTRACT_V3_VERSION,
    }:
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
        dynamic_loss = (
            DynamicCropV3LossContract
            if version == MODEL_TRAINING_CONTRACT_V3_VERSION
            else DynamicCropV2LossContract
        )(
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
    if version == MODEL_TRAINING_CONTRACT_V3_VERSION:
        if contract.get("dynamic_crop_version") != DYNAMIC_CROP_V3_SCHEMA_VERSION:
            raise ReleaseFoldPreflightError(
                "release Dynamic Crop v3 version binding differs"
            )
        dynamic_config_payload = dict(
            _mapping(contract.get("dynamic_crop_config"), "dynamic crop v3 config")
        )
        if contract.get("dynamic_crop_config_sha256") != _canonical_sha256(
            dynamic_config_payload
        ):
            raise ReleaseFoldPreflightError(
                "release Dynamic Crop v3 config hash mismatch"
            )
        try:
            dynamic_config = DynamicCropHeadV3Config(**dynamic_config_payload)
        except (TypeError, ValueError) as exc:
            raise ReleaseFoldPreflightError(
                f"release Dynamic Crop v3 config is invalid: {exc}"
            ) from exc
        if dynamic_config != DynamicCropHeadV3Config():
            raise ReleaseFoldPreflightError(
                "release Dynamic Crop v3 config is not the renderer contract"
            )
        schedule = dict(
            _mapping(contract.get("v3_training_schedule"), "v3 training schedule")
        )
        if (
            contract.get("v3_training_schedule_sha256")
            != _canonical_sha256(schedule)
            or schedule != dict(V3_PROBE_TRAINING_SCHEDULE)
        ):
            raise ReleaseFoldPreflightError(
                "release Dynamic Crop v3 training schedule differs from freeze"
            )
        if (
            not isinstance(dynamic_loss, DynamicCropV3LossContract)
            or dynamic_loss.contract != DYNAMIC_CROP_V3_LOSS_CONTRACT
        ):
            raise ReleaseFoldPreflightError(
                "release Dynamic Crop v3 loss contract differs"
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


_CAPACITY_SELECTION_BINDING_KEYS = {
    "binding_version",
    "path",
    "file_sha256",
    "artifact_content_sha256",
    "capacity_probe_schema_version",
    "capacity_probe_analysis_basis",
    "selection_identity_sha256",
    "selected_candidate_id",
    "release_model_training_contract_origin",
    "release_model_training_contract_sha256",
}

_V3_RELEASE_READINESS_BINDING_KEYS = {
    "binding_version",
    "readiness_kind",
    "dynamic_crop_version",
    "release_freeze",
    "release_freeze_content_sha256",
    "final_fit_receipt",
    "final_fit_receipt_content_sha256",
    "final_fit_plan",
    "final_fit_plan_content_sha256",
    "training_manifest_tuples",
    "training_manifest_graph_sha256",
    "professional_montage_profile",
    "professional_montage_profile_identity",
    "candidate_name",
    "candidate_id",
    "candidate_contract_sha256",
    "v3_training_schedule_sha256",
    "final_fit_checkpoint_id",
    "final_fit_checkpoint_weight_sha256",
    "final_fit_checkpoint_metadata",
    "final_fit_checkpoint_weights",
    "release_model_training_contract_sha256",
    "final_fit_checkpoint_usage",
    "production_activation_authorized",
}


def _v3_training_manifest_tuples_from_final_fit_plan(
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for raw in _array(plan.get("songs"), "v3 final-fit songs"):
        item = _mapping(raw, "v3 final-fit song")
        song_id = str(item.get("song_id") or "")
        if song_id in indexed:
            raise ReleaseFoldPreflightError(
                "Dynamic Crop v3 final-fit manifest graph has duplicate songs"
            )
        indexed[song_id] = item
    if set(indexed) != set(DEVELOPMENT_TRAINING_SONG_IDS):
        raise ReleaseFoldPreflightError(
            "Dynamic Crop v3 final-fit manifest graph is not the exact "
            "three-song training scope"
        )
    return [
        {
            "song_id": song_id,
            "materialization_manifest": dict(
                _mapping(
                    indexed[song_id].get("materialization_manifest"),
                    f"{song_id} v3 materialization pointer",
                )
            ),
            "feature_manifest": dict(
                _mapping(
                    indexed[song_id].get("feature_manifest"),
                    f"{song_id} v3 feature pointer",
                )
            ),
            "dynamic_crop_manifest": dict(
                _mapping(
                    indexed[song_id].get("dynamic_crop_manifest"),
                    f"{song_id} v3 dynamic-crop pointer",
                )
            ),
        }
        for song_id in DEVELOPMENT_TRAINING_SONG_IDS
    ]


def _v3_training_manifest_tuples_from_release_songs(
    songs: object,
) -> list[dict[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for raw in _array(songs, "release songs"):
        item = _mapping(raw, "release song")
        song_id = str(item.get("song_id") or "")
        if song_id in indexed:
            raise ReleaseFoldPreflightError("duplicate release song id")
        indexed[song_id] = item
    if not set(DEVELOPMENT_TRAINING_SONG_IDS).issubset(indexed):
        raise ReleaseFoldPreflightError(
            "release plan lacks the exact v3 training-song manifest graph"
        )
    return [
        {
            "song_id": song_id,
            "materialization_manifest": dict(
                _mapping(
                    indexed[song_id].get("materialization_manifest"),
                    f"{song_id} release materialization pointer",
                )
            ),
            "feature_manifest": dict(
                _mapping(
                    indexed[song_id].get("feature_manifest"),
                    f"{song_id} release feature pointer",
                )
            ),
            "dynamic_crop_manifest": dict(
                _mapping(
                    indexed[song_id].get("dynamic_crop_manifest"),
                    f"{song_id} release dynamic-crop pointer",
                )
            ),
        }
        for song_id in DEVELOPMENT_TRAINING_SONG_IDS
    ]


def _v3_model_training_contract(
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = dict(
        _mapping(freeze.get("candidate_contract"), "frozen v3 candidate contract")
    )
    expected_candidate_keys = {
        "dynamic_crop_version",
        "model_contract",
        "target_contract",
        "output_contract",
        "loss_contract",
        "model_class",
        "loss_class",
        "activation_checkpointing",
        "legacy_v2_checkpoint_compatible",
        "v2_model_or_loss_accepted",
        "model_config",
        "base_loss_contract",
        "dynamic_crop_config",
        "dynamic_crop_loss_contract",
        "parameter_count",
        "semantic_model_selection_authorized",
    }
    if set(candidate) != expected_candidate_keys or (
        candidate.get("dynamic_crop_version") != DYNAMIC_CROP_V3_SCHEMA_VERSION
        or candidate.get("model_contract") != DYNAMIC_CROP_V3_MODEL_CONTRACT
        or candidate.get("target_contract") != DYNAMIC_CROP_V3_TARGET_CONTRACT
        or candidate.get("output_contract") != DYNAMIC_CROP_V3_OUTPUT_CONTRACT
        or candidate.get("loss_contract") != DYNAMIC_CROP_V3_LOSS_CONTRACT
        or candidate.get("model_class") != "StageEditTransformerDynamicCropV3"
        or candidate.get("loss_class") != "DynamicCropV3LossContract"
        or candidate.get("legacy_v2_checkpoint_compatible") is not False
        or candidate.get("v2_model_or_loss_accepted") is not False
    ):
        raise ReleaseFoldPreflightError(
            "frozen release candidate is not the exact Dynamic Crop v3 contract"
        )
    model_payload = dict(_mapping(candidate.get("model_config"), "v3 model config"))
    base_payload = dict(
        _mapping(candidate.get("base_loss_contract"), "v3 base loss contract")
    )
    dynamic_config = dict(
        _mapping(candidate.get("dynamic_crop_config"), "v3 crop config")
    )
    dynamic_loss = dict(
        _mapping(candidate.get("dynamic_crop_loss_contract"), "v3 loss contract")
    )
    schedule = dict(
        _mapping(freeze.get("training_schedule"), "v3 frozen training schedule")
    )
    try:
        base = StageEditLossContract(**base_payload)
    except (TypeError, ValueError) as exc:
        raise ReleaseFoldPreflightError(
            f"frozen v3 base loss contract is invalid: {exc}"
        ) from exc
    authority = StageEditTransitionAuthority(
        mode=RENDERER_OWNED_TRANSITION_AUTHORITY
    )
    contract = {
        "contract_version": MODEL_TRAINING_CONTRACT_V3_VERSION,
        "dynamic_crop_version": DYNAMIC_CROP_V3_SCHEMA_VERSION,
        "model_config": model_payload,
        "model_config_sha256": _canonical_sha256(model_payload),
        "base_loss_contract": base_payload,
        "base_loss_contract_sha256": _canonical_sha256(base_payload),
        "dynamic_crop_config": dynamic_config,
        "dynamic_crop_config_sha256": _canonical_sha256(dynamic_config),
        "dynamic_crop_loss_contract": dynamic_loss,
        "dynamic_crop_loss_contract_sha256": _canonical_sha256(dynamic_loss),
        "v3_training_schedule": schedule,
        "v3_training_schedule_sha256": _canonical_sha256(schedule),
        "source_objective": base.source_objective,
        "transition_authority_contract": authority.to_dict(),
        "transition_authority_contract_sha256": authority.fingerprint,
    }
    return _validate_model_training_contract(contract)


def _load_verified_v3_release_readiness(
    release_freeze_path: Path,
    final_fit_receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    freeze_raw = Path(release_freeze_path).expanduser()
    receipt_raw = Path(final_fit_receipt_path).expanduser()
    for path, label in (
        (freeze_raw, "Dynamic Crop v3 release freeze"),
        (receipt_raw, "Dynamic Crop v3 final-fit receipt"),
    ):
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ReleaseFoldPreflightError(f"{label} is missing or unsafe")
    freeze_before = _file_sha256(freeze_raw, "Dynamic Crop v3 release freeze")
    receipt_before = _file_sha256(
        receipt_raw, "Dynamic Crop v3 final-fit receipt"
    )
    try:
        freeze = load_dynamic_crop_v3_release_freeze(freeze_raw)
        receipt = load_dynamic_crop_v3_final_fit_receipt(receipt_raw)
    except (DynamicCropV3FinalReleaseError, OSError, RuntimeError, ValueError) as exc:
        raise ReleaseFoldPreflightError(
            f"Dynamic Crop v3 release readiness is invalid: {exc}"
        ) from exc
    if (
        _file_sha256(freeze_raw, "Dynamic Crop v3 release freeze") != freeze_before
        or _file_sha256(receipt_raw, "Dynamic Crop v3 final-fit receipt")
        != receipt_before
    ):
        raise ReleaseFoldPreflightError(
            "Dynamic Crop v3 release readiness changed while loading"
        )
    freeze_path = freeze_raw.resolve()
    receipt_path = receipt_raw.resolve()
    final_fit_plan_path = _verify_pointer(
        receipt.get("plan"), "Dynamic Crop v3 final-fit plan"
    )
    try:
        final_fit_plan = load_dynamic_crop_v3_final_fit_plan(
            final_fit_plan_path
        )
    except (DynamicCropV3FinalReleaseError, OSError, RuntimeError, ValueError) as exc:
        raise ReleaseFoldPreflightError(
            f"Dynamic Crop v3 final-fit manifest graph is invalid: {exc}"
        ) from exc
    if (
        receipt.get("release_freeze")
        != _pointer(freeze_path, "Dynamic Crop v3 release freeze")
        or receipt.get("candidate_id") != freeze.get("candidate_id")
        or receipt.get("candidate_name") != freeze.get("candidate_name")
        or receipt.get("dynamic_crop_version") != DYNAMIC_CROP_V3_SCHEMA_VERSION
        or receipt.get("production_activation_authorized") is not False
    ):
        raise ReleaseFoldPreflightError(
            "Dynamic Crop v3 final-fit checkpoint differs from its LOSO freeze"
        )
    contract = _v3_model_training_contract(freeze)
    manifest_tuples = _v3_training_manifest_tuples_from_final_fit_plan(
        final_fit_plan
    )
    candidate_contract = dict(
        _mapping(freeze.get("candidate_contract"), "v3 candidate contract")
    )
    binding = {
        "binding_version": V3_RELEASE_READINESS_BINDING_VERSION,
        "readiness_kind": V3_RELEASE_READINESS_KIND,
        "dynamic_crop_version": DYNAMIC_CROP_V3_SCHEMA_VERSION,
        "release_freeze": _pointer(freeze_path, "Dynamic Crop v3 release freeze"),
        "release_freeze_content_sha256": freeze["freeze_content_sha256"],
        "final_fit_receipt": _pointer(
            receipt_path, "Dynamic Crop v3 final-fit receipt"
        ),
        "final_fit_receipt_content_sha256": receipt["artifact_content_sha256"],
        "final_fit_plan": _pointer(
            final_fit_plan_path, "Dynamic Crop v3 final-fit plan"
        ),
        "final_fit_plan_content_sha256": final_fit_plan[
            "plan_content_sha256"
        ],
        "training_manifest_tuples": manifest_tuples,
        "training_manifest_graph_sha256": _canonical_sha256(
            manifest_tuples
        ),
        "professional_montage_profile": dict(
            freeze["professional_montage_profile"]
        ),
        "professional_montage_profile_identity": dict(
            freeze["professional_montage_profile_identity"]
        ),
        "candidate_name": freeze["candidate_name"],
        "candidate_id": freeze["candidate_id"],
        "candidate_contract_sha256": _canonical_sha256(candidate_contract),
        "v3_training_schedule_sha256": _canonical_sha256(
            freeze["training_schedule"]
        ),
        "final_fit_checkpoint_id": receipt["checkpoint_id"],
        "final_fit_checkpoint_weight_sha256": receipt[
            "checkpoint_weight_sha256"
        ],
        "final_fit_checkpoint_metadata": dict(receipt["checkpoint_metadata"]),
        "final_fit_checkpoint_weights": dict(receipt["checkpoint_weights"]),
        "release_model_training_contract_sha256": _canonical_sha256(contract),
        "final_fit_checkpoint_usage": V3_FINAL_FIT_CHECKPOINT_USAGE,
        "production_activation_authorized": False,
    }
    if (
        set(binding) != _V3_RELEASE_READINESS_BINDING_KEYS
        or not all(
            _is_sha256(binding[name])
            for name in (
                "release_freeze_content_sha256",
                "final_fit_receipt_content_sha256",
                "final_fit_plan_content_sha256",
                "training_manifest_graph_sha256",
                "candidate_id",
                "candidate_contract_sha256",
                "v3_training_schedule_sha256",
                "final_fit_checkpoint_weight_sha256",
                "release_model_training_contract_sha256",
            )
        )
        or binding.get("final_fit_checkpoint_usage")
        != V3_FINAL_FIT_CHECKPOINT_USAGE
    ):
        raise ReleaseFoldPreflightError(
            "Dynamic Crop v3 release readiness identity is invalid"
        )
    return binding, contract


def _load_verified_capacity_selection(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load one immutable capacity result and project its release authority.

    The capacity loader performs the expensive deterministic replay of the
    BANG BANG / BLACKHOLE / REBEL HEART evidence.  This wrapper additionally
    binds the exact file bytes and the selected candidate/origin to the release
    model contract so a caller cannot substitute a standalone JSON contract.
    """

    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise ReleaseFoldPreflightError(
            "capacity selection path must be explicit and absolute"
        )
    if raw.is_symlink():
        raise ReleaseFoldPreflightError("capacity selection must not be a symlink")
    try:
        before_sha256 = _file_sha256(raw, "capacity selection")
        selection = load_dynamic_crop_capacity_selection(raw)
        after_sha256 = _file_sha256(raw, "capacity selection")
    except (DynamicCropCapacityProbeError, OSError, ValueError) as exc:
        raise ReleaseFoldPreflightError(
            f"capacity selection is missing, unsafe, or invalid: {exc}"
        ) from exc
    if before_sha256 != after_sha256:
        raise ReleaseFoldPreflightError("capacity selection changed while loading")

    resolved = raw.resolve()
    artifact_sha256 = selection.get("artifact_content_sha256")
    selection_identity = selection.get("selection_identity_sha256")
    selected_candidate_id = selection.get("selected_candidate_id")
    if not all(
        _is_sha256(value)
        for value in (
            artifact_sha256,
            selection_identity,
            selected_candidate_id,
        )
    ):
        raise ReleaseFoldPreflightError(
            "capacity selection identity hashes are invalid"
        )
    schema_version = selection.get("schema_version")
    analysis_basis = selection.get("analysis_basis")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version < 1
        or not isinstance(analysis_basis, str)
        or not analysis_basis
    ):
        raise ReleaseFoldPreflightError(
            "capacity selection schema identity is invalid"
        )

    origin = dict(
        _mapping(
            selection.get("release_model_training_contract_origin"),
            "capacity-selected release model/training contract origin",
        )
    )
    if set(origin) != {
        "capacity_probe_schema_version",
        "selection_identity_sha256",
        "selected_candidate_id",
    }:
        raise ReleaseFoldPreflightError(
            "capacity-selected release contract origin keys are not exact"
        )
    if origin != {
        "capacity_probe_schema_version": schema_version,
        "selection_identity_sha256": selection_identity,
        "selected_candidate_id": selected_candidate_id,
    }:
        raise ReleaseFoldPreflightError(
            "capacity-selected release contract origin is inconsistent"
        )

    selected_candidates = [
        dict(candidate)
        for candidate in _array(selection.get("candidates"), "capacity candidates")
        if isinstance(candidate, Mapping)
        and candidate.get("candidate_id") == selected_candidate_id
    ]
    if len(selected_candidates) != 1:
        raise ReleaseFoldPreflightError(
            "capacity-selected candidate identity is not unique"
        )
    contract = _validate_model_training_contract(
        selection.get("release_model_training_contract")
    )
    selected_contract = _mapping(
        selected_candidates[0].get("contract"),
        "capacity-selected candidate contract",
    )
    for key in (
        "model_config",
        "model_config_sha256",
        "base_loss_contract",
        "base_loss_contract_sha256",
        "dynamic_crop_loss_contract",
        "dynamic_crop_loss_contract_sha256",
    ):
        if selected_contract.get(key) != contract.get(key):
            raise ReleaseFoldPreflightError(
                "capacity-selected release contract differs from selected candidate"
            )
    binding = {
        "binding_version": CAPACITY_SELECTION_BINDING_VERSION,
        "path": str(resolved),
        "file_sha256": after_sha256,
        "artifact_content_sha256": artifact_sha256,
        "capacity_probe_schema_version": schema_version,
        "capacity_probe_analysis_basis": analysis_basis,
        "selection_identity_sha256": selection_identity,
        "selected_candidate_id": selected_candidate_id,
        "release_model_training_contract_origin": origin,
        "release_model_training_contract_sha256": _canonical_sha256(contract),
    }
    return selection, binding, contract


def _verify_capacity_selection_binding(
    value: object,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = dict(_mapping(value, "capacity selection binding"))
    if binding.get("binding_version") == V3_RELEASE_READINESS_BINDING_VERSION:
        if (
            set(binding) != _V3_RELEASE_READINESS_BINDING_KEYS
            or binding.get("readiness_kind") != V3_RELEASE_READINESS_KIND
            or binding.get("dynamic_crop_version")
            != DYNAMIC_CROP_V3_SCHEMA_VERSION
            or binding.get("production_activation_authorized") is not False
        ):
            raise ReleaseFoldPreflightError(
                "Dynamic Crop v3 release readiness binding is not exact"
            )
        freeze_path = _verify_pointer(
            binding.get("release_freeze"), "Dynamic Crop v3 release freeze"
        )
        receipt_path = _verify_pointer(
            binding.get("final_fit_receipt"),
            "Dynamic Crop v3 final-fit receipt",
        )
        observed, contract = _load_verified_v3_release_readiness(
            freeze_path,
            receipt_path,
        )
        if observed != binding:
            raise ReleaseFoldPreflightError(
                "Dynamic Crop v3 release readiness binding changed"
            )
        return observed, contract
    if set(binding) != _CAPACITY_SELECTION_BINDING_KEYS:
        raise ReleaseFoldPreflightError(
            "capacity selection binding keys are not exact"
        )
    if binding.get("binding_version") != CAPACITY_SELECTION_BINDING_VERSION:
        raise ReleaseFoldPreflightError(
            "capacity selection binding version is invalid"
        )
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ReleaseFoldPreflightError("capacity selection binding path is invalid")
    if not _is_sha256(binding.get("file_sha256")):
        raise ReleaseFoldPreflightError(
            "capacity selection binding file hash is invalid"
        )
    selection, observed, contract = _load_verified_capacity_selection(
        Path(raw_path)
    )
    if observed != binding:
        raise ReleaseFoldPreflightError(
            "capacity selection binding changed or does not match its origin"
        )
    return selection, contract


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


def _validate_blind_song_id(value: object) -> str:
    """Validate a new IVE blind-song identity without touching any paths."""

    if not isinstance(value, str) or not value or value.strip() != value:
        raise ReleaseFoldPreflightError(
            "blind_song_id must be a non-empty canonical IVE song id"
        )
    if (
        not value.startswith("IVE-")
        or value.endswith("-")
        or "--" in value
        or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            for character in value
        )
    ):
        raise ReleaseFoldPreflightError(
            "blind_song_id must use canonical IVE-UPPERCASE-HYPHEN form"
        )
    if value in DEVELOPMENT_SONG_IDS:
        raise ReleaseFoldPreflightError(
            "blind_song_id must not reuse any of the four development songs"
        )
    if value in QUARANTINED_BLIND_SONG_IDS:
        raise ReleaseFoldPreflightError(
            "blind_song_id is quarantined and cannot carry strict-blind authority"
        )
    return value


def _final_song_ids(blind_song_id: object) -> tuple[str, ...]:
    return (*FINAL_TRAINING_SONG_IDS, _validate_blind_song_id(blind_song_id))


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


def _normalized_absolute_path_without_inspection(path: Path, name: str) -> str:
    """Normalize an absolute path lexically without stat, resolve, or content IO."""

    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        raise ReleaseFoldPreflightError(f"{name} path must be absolute")
    return os.path.normpath(str(expanded))


def _deferred_blind_song_record(
    *,
    blind_song_id: str,
    materialization_manifest: Path,
    feature_manifest: Path,
    dynamic_crop_manifest: Path,
) -> dict[str, Any]:
    blind_song_id = _validate_blind_song_id(blind_song_id)
    return {
        "song_id": blind_song_id,
        "inspection_contract": DEFERRED_BLIND_MANIFEST_CONTRACT,
        "materialization_manifest_path": (
            _normalized_absolute_path_without_inspection(
                materialization_manifest,
                f"{blind_song_id} blind materialization manifest",
            )
        ),
        "feature_manifest_path": _normalized_absolute_path_without_inspection(
            feature_manifest,
            f"{blind_song_id} blind feature manifest",
        ),
        "dynamic_crop_manifest_path": (
            _normalized_absolute_path_without_inspection(
                dynamic_crop_manifest,
                f"{blind_song_id} blind dynamic crop manifest",
            )
        ),
    }


def _strict_deferred_blind_song_record(
    value: object, *, expected_blind_song_id: str
) -> dict[str, Any]:
    expected_blind_song_id = _validate_blind_song_id(expected_blind_song_id)
    record = dict(_mapping(value, "deferred blind manifest record"))
    expected_keys = {
        "song_id",
        "inspection_contract",
        "materialization_manifest_path",
        "feature_manifest_path",
        "dynamic_crop_manifest_path",
    }
    if set(record) != expected_keys:
        raise ReleaseFoldPreflightError(
            "deferred blind manifest record keys are not exact"
        )
    if (
        record.get("song_id") != expected_blind_song_id
        or record.get("inspection_contract") != DEFERRED_BLIND_MANIFEST_CONTRACT
    ):
        raise ReleaseFoldPreflightError(
            "deferred blind manifest inspection contract is invalid"
        )
    for key in (
        "materialization_manifest_path",
        "feature_manifest_path",
        "dynamic_crop_manifest_path",
    ):
        raw = record.get(key)
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            raise ReleaseFoldPreflightError(
                "deferred blind manifest path is not absolute"
            )
        if os.path.normpath(raw) != raw:
            raise ReleaseFoldPreflightError(
                "deferred blind manifest path is not lexically normalized"
            )
    return record


def _materialize_deferred_blind_song_record(
    value: object, *, expected_blind_song_id: str
) -> dict[str, Any]:
    """Read/hash blind evidence only after a verified consumption marker."""

    blind_song_id = _validate_blind_song_id(expected_blind_song_id)
    record = _strict_deferred_blind_song_record(
        value, expected_blind_song_id=blind_song_id
    )
    return {
        "song_id": blind_song_id,
        "materialization_manifest": _pointer(
            Path(record["materialization_manifest_path"]),
            f"{blind_song_id} blind materialization manifest",
        ),
        "feature_manifest": _pointer(
            Path(record["feature_manifest_path"]),
            f"{blind_song_id} blind feature manifest",
        ),
        "dynamic_crop_manifest": _pointer(
            Path(record["dynamic_crop_manifest_path"]),
            f"{blind_song_id} blind dynamic crop manifest",
        ),
    }


def create_stage_edit_release_development_plan(
    *,
    materialization_manifests: Mapping[str, Path],
    feature_manifests: Mapping[str, Path],
    dynamic_crop_manifests: Mapping[str, Path],
    checkpoint_directory: Path,
    output_directory: Path,
    capacity_selection_path: Path | None = None,
    dynamic_crop_v3_release_freeze_path: Path | None = None,
    dynamic_crop_v3_final_fit_receipt_path: Path | None = None,
    training_config: Mapping[str, Any] = DEFAULT_RELEASE_TRAINING_CONFIG,
) -> Path:
    """Bind the exact four-song development fold without activation authority."""

    checkpoint = Path(checkpoint_directory).expanduser().resolve()
    volume = legacy.PRODUCTION_VOLUME_ROOT.resolve()
    if volume != checkpoint and volume not in checkpoint.parents:
        raise ReleaseFoldPreflightError(
            f"development checkpoint must be on {volume}: {checkpoint}"
        )
    v3_values = (
        dynamic_crop_v3_release_freeze_path,
        dynamic_crop_v3_final_fit_receipt_path,
    )
    if any(value is not None for value in v3_values):
        if capacity_selection_path is not None or not all(
            value is not None for value in v3_values
        ):
            raise ReleaseFoldPreflightError(
                "development requires either one legacy capacity selection or "
                "the complete Dynamic Crop v3 freeze/final-fit pair"
            )
        capacity_selection, model_training_contract = (
            _load_verified_v3_release_readiness(
                dynamic_crop_v3_release_freeze_path,
                dynamic_crop_v3_final_fit_receipt_path,
            )
        )
    else:
        if capacity_selection_path is None:
            raise ReleaseFoldPreflightError(
                "development release readiness evidence is required"
            )
        _, capacity_selection, model_training_contract = (
            _load_verified_capacity_selection(capacity_selection_path)
        )
    if capacity_selection.get("binding_version") == (
        V3_RELEASE_READINESS_BINDING_VERSION
    ):
        requested_training = dict(training_config)
        if requested_training == dict(DEFAULT_RELEASE_TRAINING_CONFIG):
            effective_training_config = dict(DEFAULT_RELEASE_TRAINING_CONFIG_V3)
        elif requested_training == dict(DEFAULT_RELEASE_TRAINING_CONFIG_V3):
            effective_training_config = requested_training
        else:
            raise ReleaseFoldPreflightError(
                "Dynamic Crop v3 development schedule must match the frozen "
                "capacity/final-fit schedule"
            )
    else:
        effective_training_config = dict(training_config)
    song_records = _song_records(
        DEVELOPMENT_SONG_IDS,
        materialization_manifests=materialization_manifests,
        feature_manifests=feature_manifests,
        dynamic_crop_manifests=dynamic_crop_manifests,
    )
    if capacity_selection.get("binding_version") == (
        V3_RELEASE_READINESS_BINDING_VERSION
    ):
        current_manifest_tuples = (
            _v3_training_manifest_tuples_from_release_songs(song_records)
        )
        if (
            current_manifest_tuples
            != capacity_selection.get("training_manifest_tuples")
            or _canonical_sha256(current_manifest_tuples)
            != capacity_selection.get("training_manifest_graph_sha256")
        ):
            raise ReleaseFoldPreflightError(
                "release training manifests differ from the Dynamic Crop v3 "
                "LOSO/final-fit manifest graph"
            )
    is_v3_release = capacity_selection.get("binding_version") == (
        V3_RELEASE_READINESS_BINDING_VERSION
    )
    unsigned = {
        "schema_version": (
            RELEASE_PLAN_V3_SCHEMA_VERSION
            if is_v3_release
            else RELEASE_PLAN_SCHEMA_VERSION
        ),
        "analysis_basis": (
            DEVELOPMENT_PLAN_V3_ANALYSIS_BASIS
            if is_v3_release
            else DEVELOPMENT_PLAN_ANALYSIS_BASIS
        ),
        "stage": "development",
        "release_contract": _release_contract(
            training_song_ids=DEVELOPMENT_TRAINING_SONG_IDS,
            held_out_song_id=XOXZ,
            stage="development",
        ),
        "required_volume": str(legacy.PRODUCTION_VOLUME_ROOT),
        "minimum_free_bytes": legacy.MINIMUM_FREE_BYTES,
        "training_config": _validate_training_config(effective_training_config),
        "capacity_selection": capacity_selection,
        "model_training_contract": model_training_contract,
        "structured_head_preflight_contract": (
            _structured_head_preflight_contract()
        ),
        "checkpoint_directory": str(checkpoint),
        "songs": song_records,
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
    "capacity_selection",
    "model_training_contract",
    "structured_head_preflight_contract",
    "checkpoint_directory",
    "songs",
    "plan_sha256",
}


def _strict_development_plan(path: Path) -> dict[str, Any]:
    raw = _read_json_object(
        Path(path).expanduser().resolve(), "development release plan"
    )
    schema_version = raw.get("schema_version")
    if schema_version == RELEASE_PLAN_V3_SCHEMA_VERSION:
        expected_basis = DEVELOPMENT_PLAN_V3_ANALYSIS_BASIS
    elif schema_version == RELEASE_PLAN_SCHEMA_VERSION:
        expected_basis = DEVELOPMENT_PLAN_ANALYSIS_BASIS
    else:
        raise ReleaseFoldPreflightError(
            "development release plan schema mismatch or unsupported"
        )
    payload = _strict_content_addressed(
        path,
        expected_keys=_DEVELOPMENT_PLAN_KEYS,
        schema_basis=expected_basis,
        prefix="stageedit-release-development-plan",
        hash_field="plan_sha256",
        expected_schema_version=int(schema_version),
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
    selected_binding, selected_contract = _verify_capacity_selection_binding(
        payload.get("capacity_selection")
    )
    is_v3_binding = selected_binding.get("binding_version") == (
        V3_RELEASE_READINESS_BINDING_VERSION
    )
    if is_v3_binding != (
        schema_version == RELEASE_PLAN_V3_SCHEMA_VERSION
    ):
        raise ReleaseFoldPreflightError(
            "development plan schema does not match its v3 readiness binding"
        )
    if (
        selected_binding.get("binding_version")
        == V3_RELEASE_READINESS_BINDING_VERSION
        and payload.get("training_config")
        != dict(DEFAULT_RELEASE_TRAINING_CONFIG_V3)
    ):
        raise ReleaseFoldPreflightError(
            "Dynamic Crop v3 development training schedule is not frozen"
        )
    planned_contract = _validate_model_training_contract(
        payload.get("model_training_contract")
    )
    if planned_contract != selected_contract:
        raise ReleaseFoldPreflightError(
            "development model/training contract differs from capacity selection"
        )
    _validate_structured_head_preflight_contract(
        payload.get("structured_head_preflight_contract")
    )
    _indexed_song_records(payload, DEVELOPMENT_SONG_IDS)
    if selected_binding.get("binding_version") == (
        V3_RELEASE_READINESS_BINDING_VERSION
    ):
        current_manifest_tuples = (
            _v3_training_manifest_tuples_from_release_songs(
                payload.get("songs")
            )
        )
        if (
            current_manifest_tuples
            != selected_binding.get("training_manifest_tuples")
            or _canonical_sha256(current_manifest_tuples)
            != selected_binding.get("training_manifest_graph_sha256")
        ):
            raise ReleaseFoldPreflightError(
                "development manifest graph differs from frozen Dynamic Crop v3 "
                "readiness"
            )
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
    args = [
        "--model-config-json",
        _canonical_bytes(validated["model_config"]).decode("utf-8"),
        "--base-loss-contract-json",
        _canonical_bytes(validated["base_loss_contract"]).decode("utf-8"),
        "--dynamic-crop-loss-contract-json",
        _canonical_bytes(validated["dynamic_crop_loss_contract"]).decode("utf-8"),
    ]
    if validated["contract_version"] == MODEL_TRAINING_CONTRACT_V3_VERSION:
        schedule = validated["v3_training_schedule"]
        args.extend(
            [
                "--dynamic-crop-version",
                "3",
                "--source-pretrain-steps",
                str(schedule["source_pretraining_steps"]),
                "--source-pretrain-learning-rate",
                str(schedule["source_pretraining_peak_learning_rate"]),
                "--source-pretrain-window-length",
                str(schedule["source_pretraining_window_length"]),
                "--source-pretrain-warmup-steps",
                str(schedule["source_pretraining_warmup_steps"]),
                "--dynamic-crop-v3-prior-pretrain-steps",
                str(schedule["dynamic_crop_v3_prior_pretraining_steps"]),
                "--dynamic-crop-v3-prior-pretrain-learning-rate",
                str(
                    schedule[
                        "dynamic_crop_v3_prior_pretraining_peak_learning_rate"
                    ]
                ),
                "--dynamic-crop-v3-prior-pretrain-warmup-steps",
                str(schedule["dynamic_crop_v3_prior_pretraining_warmup_steps"]),
                "--minimum-finetune-optimizer-steps",
                str(schedule["minimum_finetune_optimizer_steps"]),
                "--minimum-checkpoint-selection-finetune-steps",
                str(schedule["minimum_checkpoint_selection_finetune_steps"]),
            ]
        )
    return args


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
        capacity_selection=dict(payload["capacity_selection"]),
        blind_consumption_marker=None,
        held_out_manifest_inspection_deferred=False,
        train_cli_args=tuple(args),
        reference_ids_by_song=references,
    )


def _require_dynamic_crop_v3_release_launch(
    payload: Mapping[str, Any],
    train_cli_args: Sequence[str],
    *,
    activation_authorized: bool,
) -> None:
    """Reject every legacy/default-head route before production training."""

    readiness, model_contract = _verify_capacity_selection_binding(
        payload.get("capacity_selection")
    )
    if (
        readiness.get("binding_version")
        != V3_RELEASE_READINESS_BINDING_VERSION
        or readiness.get("dynamic_crop_version")
        != DYNAMIC_CROP_V3_SCHEMA_VERSION
        or readiness.get("final_fit_checkpoint_usage")
        != V3_FINAL_FIT_CHECKPOINT_USAGE
        or model_contract.get("contract_version")
        != MODEL_TRAINING_CONTRACT_V3_VERSION
    ):
        raise ReleaseFoldPreflightError(
            "release launch requires the frozen Dynamic Crop v3 LOSO/final-fit "
            "readiness tuple; legacy capacity evidence cannot launch"
        )
    args = list(train_cli_args)
    if args.count("--dynamic-crop-version") != 1:
        raise ReleaseFoldPreflightError(
            "release launch must bind Dynamic Crop v3 exactly once"
        )
    version_index = args.index("--dynamic-crop-version")
    if version_index + 1 >= len(args) or args[version_index + 1] != "3":
        raise ReleaseFoldPreflightError(
            "release launch attempted a non-v3 dynamic crop head"
        )
    if any(
        option in args
        for option in (
            "--init-checkpoint",
            "--initial-checkpoint",
            "--warm-start-checkpoint",
        )
    ):
        raise ReleaseFoldPreflightError(
            "inactive v3 final-fit checkpoint is readiness-only and cannot "
            "silently initialize release optimization"
        )
    has_activation = "--authorize-production-activation" in args
    if has_activation != activation_authorized:
        raise ReleaseFoldPreflightError(
            "release activation authority differs from the launch stage"
        )
    if activation_authorized:
        if args.count("--continuity-authority") != 1:
            raise ReleaseFoldPreflightError(
                "authorized v3 release requires one continuity authority"
            )
        continuity_index = args.index("--continuity-authority")
        if (
            continuity_index + 1 >= len(args)
            or args[continuity_index + 1] != HANDCRAFTED_CONTINUITY_AUTHORITY
        ):
            raise ReleaseFoldPreflightError(
                "current v3 release contract requires handcrafted continuity"
            )
    elif "--continuity-authority" in args:
        raise ReleaseFoldPreflightError(
            "inactive development launch acquired continuity activation authority"
        )


def launch_stage_edit_release_development(plan_path: Path) -> int:
    """Launch only the current frozen v3 development fold."""

    preflight = preflight_stage_edit_release_development(plan_path)
    payload = _strict_development_plan(preflight.plan_path)
    _require_dynamic_crop_v3_release_launch(
        payload,
        preflight.train_cli_args,
        activation_authorized=False,
    )
    return production_train_main(preflight.train_cli_args)


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
    """Freeze a passing inactive XOXZ development run before blind planning."""

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
    is_v3_release = plan.get("schema_version") == RELEASE_PLAN_V3_SCHEMA_VERSION
    unsigned = {
        "schema_version": (
            DEVELOPMENT_FREEZE_V3_SCHEMA_VERSION
            if is_v3_release
            else DEVELOPMENT_FREEZE_SCHEMA_VERSION
        ),
        "analysis_basis": (
            DEVELOPMENT_FREEZE_V3_ANALYSIS_BASIS
            if is_v3_release
            else DEVELOPMENT_FREEZE_ANALYSIS_BASIS
        ),
        "stage": "development_frozen",
        "development_plan": _pointer(resolved_plan, "development plan"),
        "development_checkpoint_directory": str(checkpoint),
        "development_report": _pointer(report_path, "development report"),
        "selection_freeze": _pointer(selection_path, "development selection freeze"),
        "checkpoint_weights": _pointer(weight_path, "development checkpoint weights"),
        "checkpoint_id": checkpoint_id,
        "model_config": config,
        "model_config_fingerprint_sha256": _canonical_sha256(config),
        "capacity_selection": dict(plan["capacity_selection"]),
        "model_training_contract": dict(plan["model_training_contract"]),
        "structured_head_preflight_contract": dict(
            plan["structured_head_preflight_contract"]
        ),
        "training_config": dict(plan["training_config"]),
        "development_training_song_ids": list(DEVELOPMENT_TRAINING_SONG_IDS),
        "development_held_out_song_id": XOXZ,
        "development_quality_gate_passed": True,
        "production_activation_authorized": False,
        "blind_evidence_opened": False,
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
    "capacity_selection",
    "model_training_contract",
    "structured_head_preflight_contract",
    "training_config",
    "development_training_song_ids",
    "development_held_out_song_id",
    "development_quality_gate_passed",
    "production_activation_authorized",
    "blind_evidence_opened",
    "freeze_sha256",
}


def _strict_development_freeze(
    freeze_path: Path, *, expected_plan_path: Path | None = None
) -> dict[str, Any]:
    resolved_freeze = Path(freeze_path).expanduser().resolve()
    if not resolved_freeze.is_file() or resolved_freeze.is_symlink():
        raise ReleaseFoldPreflightError(
            "release evidence must be a regular non-symlink file: "
            f"{resolved_freeze}"
        )
    raw = _read_json_object(resolved_freeze, "development release freeze")
    schema_version = raw.get("schema_version")
    if schema_version == DEVELOPMENT_FREEZE_V3_SCHEMA_VERSION:
        expected_basis = DEVELOPMENT_FREEZE_V3_ANALYSIS_BASIS
    elif schema_version == DEVELOPMENT_FREEZE_SCHEMA_VERSION:
        expected_basis = DEVELOPMENT_FREEZE_ANALYSIS_BASIS
    else:
        raise ReleaseFoldPreflightError(
            "development freeze schema is unsupported"
        )
    payload = _strict_content_addressed(
        freeze_path,
        expected_keys=_DEVELOPMENT_FREEZE_KEYS,
        schema_basis=expected_basis,
        prefix="stageedit-release-development-freeze",
        hash_field="freeze_sha256",
        expected_schema_version=int(schema_version),
    )
    if (
        payload.get("stage") != "development_frozen"
        or payload.get("development_training_song_ids")
        != list(DEVELOPMENT_TRAINING_SONG_IDS)
        or payload.get("development_held_out_song_id") != XOXZ
        or payload.get("development_quality_gate_passed") is not True
        or payload.get("production_activation_authorized") is not False
        or payload.get("blind_evidence_opened") is not False
    ):
        raise ReleaseFoldPreflightError("development freeze policy is invalid")
    plan_path = _verify_pointer(payload.get("development_plan"), "development plan")
    if expected_plan_path is not None and plan_path != Path(expected_plan_path).resolve():
        raise ReleaseFoldPreflightError("development freeze binds a different plan")
    plan = _strict_development_plan(plan_path)
    if (
        schema_version == DEVELOPMENT_FREEZE_V3_SCHEMA_VERSION
    ) != (plan.get("schema_version") == RELEASE_PLAN_V3_SCHEMA_VERSION):
        raise ReleaseFoldPreflightError(
            "development freeze schema differs from its development plan"
        )
    if payload.get("training_config") != plan.get("training_config"):
        raise ReleaseFoldPreflightError("development training config freeze mismatch")
    if payload.get("capacity_selection") != plan.get("capacity_selection"):
        raise ReleaseFoldPreflightError(
            "development capacity selection freeze mismatch"
        )
    _, selected_contract = _verify_capacity_selection_binding(
        payload.get("capacity_selection")
    )
    if payload.get("model_training_contract") != plan.get(
        "model_training_contract"
    ):
        raise ReleaseFoldPreflightError("development model/loss contract freeze mismatch")
    if payload.get("model_training_contract") != selected_contract:
        raise ReleaseFoldPreflightError(
            "development frozen model/loss contract differs from capacity selection"
        )
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
    blind_song_id: str,
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
    """Freeze one new blind song after a valid development freeze."""

    # This identity check is deliberately the first operation. In particular,
    # a quarantined/development song cannot cause even a stat of caller-supplied
    # final manifests (or a read of any other release artifact).
    blind_song_id = _validate_blind_song_id(blind_song_id)
    final_song_ids = _final_song_ids(blind_song_id)
    required_songs = set(final_song_ids)
    for name, manifests in (
        ("materialization", materialization_manifests),
        ("feature", feature_manifests),
        ("dynamic crop", dynamic_crop_manifests),
    ):
        if set(manifests) != required_songs or len(manifests) != 5:
            raise ReleaseFoldPreflightError(
                f"{name} manifests must name exactly the four development songs "
                "plus the frozen blind_song_id"
            )

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
    readiness = dict(
        _mapping(freeze.get("capacity_selection"), "release readiness")
    )
    is_v3_release = readiness.get("binding_version") == (
        V3_RELEASE_READINESS_BINDING_VERSION
    )
    if is_v3_release:
        frozen_profile_pointer = dict(
            _mapping(
                readiness.get("professional_montage_profile"),
                "frozen Dynamic Crop v3 montage profile",
            )
        )
        if (
            montage_profile_identity
            != readiness.get("professional_montage_profile_identity")
            or frozen_profile_pointer
            != {
                "path": montage_profile_identity.get("path"),
                "file_sha256": montage_profile_identity.get("file_sha256"),
            }
        ):
            raise ReleaseFoldPreflightError(
                "final montage profile differs from the Dynamic Crop v3 LOSO "
                "freeze"
            )
    records = _song_records(
        DEVELOPMENT_SONG_IDS,
        materialization_manifests={
            song: materialization_manifests[song]
            for song in DEVELOPMENT_SONG_IDS
        },
        feature_manifests={
            song: feature_manifests[song] for song in DEVELOPMENT_SONG_IDS
        },
        dynamic_crop_manifests={
            song: dynamic_crop_manifests[song]
            for song in DEVELOPMENT_SONG_IDS
        },
    )
    deferred_blind = _deferred_blind_song_record(
        blind_song_id=blind_song_id,
        materialization_manifest=materialization_manifests[blind_song_id],
        feature_manifest=feature_manifests[blind_song_id],
        dynamic_crop_manifest=dynamic_crop_manifests[blind_song_id],
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
    # therefore cannot reset the one-shot blind-evaluation budget. The marker
    # is also song-agnostic, so creating a second plan with another song cannot
    # mint a second attempt from the same immutable development freeze.
    freeze_digest = str(freeze["freeze_sha256"])
    development_checkpoint = Path(
        str(freeze["development_checkpoint_directory"])
    ).resolve()
    marker = development_checkpoint.parent / (
        f"stageedit-release-{freeze_digest[:16]}.blind-consumed.json"
    )
    unsigned = {
        "schema_version": (
            FINAL_RELEASE_PLAN_V3_SCHEMA_VERSION
            if is_v3_release
            else FINAL_RELEASE_PLAN_SCHEMA_VERSION
        ),
        "analysis_basis": (
            FINAL_PLAN_V3_ANALYSIS_BASIS
            if is_v3_release
            else FINAL_PLAN_ANALYSIS_BASIS
        ),
        "stage": "final_blind",
        "blind_song_id": blind_song_id,
        "release_contract": _release_contract(
            training_song_ids=FINAL_TRAINING_SONG_IDS,
            held_out_song_id=blind_song_id,
            stage="final_blind",
        ),
        "required_volume": str(legacy.PRODUCTION_VOLUME_ROOT),
        "minimum_free_bytes": legacy.MINIMUM_FREE_BYTES,
        "training_config": dict(freeze["training_config"]),
        "model_config_fingerprint_sha256": freeze[
            "model_config_fingerprint_sha256"
        ],
        "capacity_selection": dict(freeze["capacity_selection"]),
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
        "deferred_blind_held_out_song": deferred_blind,
    }
    return _write_content_addressed(
        unsigned,
        output_directory=output_directory,
        prefix="stageedit-release-blind-plan",
        hash_field="plan_sha256",
    )


_FINAL_PLAN_KEYS = {
    "schema_version",
    "analysis_basis",
    "stage",
    "blind_song_id",
    "release_contract",
    "required_volume",
    "minimum_free_bytes",
    "training_config",
    "model_config_fingerprint_sha256",
    "capacity_selection",
    "model_training_contract",
    "structured_head_preflight_contract",
    "development_plan",
    "development_freeze",
    "checkpoint_directory",
    "blind_consumption_marker",
    "professional_montage_grammar_profile",
    "production_registry",
    "songs",
    "deferred_blind_held_out_song",
    "plan_sha256",
}


def _strict_final_plan(path: Path) -> dict[str, Any]:
    raw = _read_json_object(
        Path(path).expanduser().resolve(), "final blind release plan"
    )
    schema_version = raw.get("schema_version")
    if schema_version == FINAL_RELEASE_PLAN_V3_SCHEMA_VERSION:
        expected_basis = FINAL_PLAN_V3_ANALYSIS_BASIS
    elif schema_version == FINAL_RELEASE_PLAN_SCHEMA_VERSION:
        expected_basis = FINAL_PLAN_ANALYSIS_BASIS
    else:
        raise ReleaseFoldPreflightError(
            "final blind release plan schema is unsupported"
        )
    payload = _strict_content_addressed(
        path,
        expected_keys=_FINAL_PLAN_KEYS,
        schema_basis=expected_basis,
        prefix="stageedit-release-blind-plan",
        hash_field="plan_sha256",
        expected_schema_version=int(schema_version),
    )
    # Validate the frozen blind identity before following any pointer from the
    # plan. A forged plan cannot repurpose development/quarantined evidence.
    blind_song_id = _validate_blind_song_id(payload.get("blind_song_id"))
    if payload.get("stage") != "final_blind" or payload.get("release_contract") != _release_contract(
        training_song_ids=FINAL_TRAINING_SONG_IDS,
        held_out_song_id=blind_song_id,
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
    if payload.get("capacity_selection") != freeze.get("capacity_selection"):
        raise ReleaseFoldPreflightError(
            "final capacity selection differs from development freeze"
        )
    selected_binding, selected_contract = _verify_capacity_selection_binding(
        payload.get("capacity_selection")
    )
    is_v3_binding = selected_binding.get("binding_version") == (
        V3_RELEASE_READINESS_BINDING_VERSION
    )
    if is_v3_binding != (
        schema_version == FINAL_RELEASE_PLAN_V3_SCHEMA_VERSION
    ):
        raise ReleaseFoldPreflightError(
            "final release schema does not match its v3 readiness binding"
        )
    if is_v3_binding and (
        payload.get("professional_montage_grammar_profile")
        != selected_binding.get("professional_montage_profile_identity")
    ):
        raise ReleaseFoldPreflightError(
            "final montage profile differs from frozen Dynamic Crop v3 readiness"
        )
    if payload.get("model_training_contract") != freeze.get(
        "model_training_contract"
    ):
        raise ReleaseFoldPreflightError(
            "final model/loss contract differs from development freeze"
        )
    if payload.get("model_training_contract") != selected_contract:
        raise ReleaseFoldPreflightError(
            "final model/loss contract differs from capacity selection"
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
    final_records = _indexed_song_records(payload, DEVELOPMENT_SONG_IDS)
    _strict_deferred_blind_song_record(
        payload.get("deferred_blind_held_out_song"),
        expected_blind_song_id=blind_song_id,
    )
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
        "blind-consumed.json"
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
    """Validate final authority without reading or exposing blind inputs."""

    payload = _strict_final_plan(plan_path)
    blind_song_id = _validate_blind_song_id(payload["blind_song_id"])
    checkpoint = Path(str(payload["checkpoint_directory"])).resolve()
    legacy._validate_storage(checkpoint)
    marker = Path(str(payload["blind_consumption_marker"])).resolve()
    if marker.exists() or marker.is_symlink():
        raise ReleaseFoldPreflightError(
            "blind evaluation was already consumed for this release"
        )
    # BANG BANG / BLACKHOLE / REBEL HEART / XOXZ are already inside the
    # development evidence boundary. The frozen blind song remains path-only.
    _, references, _ = _manifest_cli_args(payload, DEVELOPMENT_SONG_IDS)
    return ReleaseFoldPreflight(
        stage="final_blind",
        plan_path=Path(plan_path).expanduser().resolve(),
        plan_sha256=str(payload["plan_sha256"]),
        held_out_song_id=blind_song_id,
        training_song_ids=FINAL_TRAINING_SONG_IDS,
        checkpoint_directory=checkpoint,
        capacity_selection=dict(payload["capacity_selection"]),
        blind_consumption_marker=marker,
        held_out_manifest_inspection_deferred=True,
        train_cli_args=(),
        reference_ids_by_song=references,
    )


def _blind_consumption_marker_payload(
    preflight: ReleaseFoldPreflight,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "contract": "stageedit_one_shot_blind_evaluation_consumed_v2",
        "final_plan_path": str(preflight.plan_path),
        "final_plan_sha256": preflight.plan_sha256,
        "held_out_song_id": _validate_blind_song_id(
            preflight.held_out_song_id
        ),
        "held_out_used_for_model_or_threshold_selection": False,
        "retry_allowed": False,
    }


def _fsync_parent_directory(path: Path) -> None:
    """Make the newly created O_EXCL marker durable before blind evidence IO."""

    descriptor = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _preflight_consumed_stage_edit_release_final(
    plan_path: Path, *, consumption_marker: Path
) -> ReleaseFoldPreflight:
    """Expand blind evidence only after the one-shot marker is durable."""

    payload = _strict_final_plan(plan_path)
    blind_song_id = _validate_blind_song_id(payload["blind_song_id"])
    checkpoint = Path(str(payload["checkpoint_directory"])).resolve()
    marker = Path(str(payload["blind_consumption_marker"])).resolve()
    if marker != Path(consumption_marker).expanduser().resolve():
        raise ReleaseFoldPreflightError(
            "consumed final preflight received a different marker"
        )
    marker_evidence = _read_json_object(marker, "blind consumption marker")
    marker_preflight = ReleaseFoldPreflight(
        stage="final_blind",
        plan_path=Path(plan_path).expanduser().resolve(),
        plan_sha256=str(payload["plan_sha256"]),
        held_out_song_id=blind_song_id,
        training_song_ids=FINAL_TRAINING_SONG_IDS,
        checkpoint_directory=checkpoint,
        capacity_selection=dict(payload["capacity_selection"]),
        blind_consumption_marker=marker,
        held_out_manifest_inspection_deferred=True,
        train_cli_args=(),
        reference_ids_by_song={},
    )
    if marker_evidence != _blind_consumption_marker_payload(marker_preflight):
        raise ReleaseFoldPreflightError(
            "blind consumption marker is invalid or mismatched"
        )

    # No blind pointer, hash, manifest, or supervision content is touched
    # above this line. Materialization below is authorized by the durable marker.
    expanded_payload = dict(payload)
    expanded_payload["songs"] = [
        *_array(payload.get("songs"), "frozen development songs"),
        _materialize_deferred_blind_song_record(
            payload.get("deferred_blind_held_out_song"),
            expected_blind_song_id=blind_song_id,
        ),
    ]
    args, references, _ = _manifest_cli_args(
        expanded_payload, _final_song_ids(blind_song_id)
    )
    for song_id in FINAL_TRAINING_SONG_IDS:
        args.extend(["--required-training-song", song_id])
    args.extend(
        [
            "--held-out-song",
            blind_song_id,
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
            "--continuity-authority",
            HANDCRAFTED_CONTINUITY_AUTHORITY,
        ]
    )
    args.append("--authorize-production-activation")
    if payload.get("production_registry") is not None:
        args.extend(["--production-registry", str(payload["production_registry"])])
    return ReleaseFoldPreflight(
        stage="final_blind",
        plan_path=Path(plan_path).expanduser().resolve(),
        plan_sha256=str(payload["plan_sha256"]),
        held_out_song_id=blind_song_id,
        training_song_ids=FINAL_TRAINING_SONG_IDS,
        checkpoint_directory=checkpoint,
        capacity_selection=dict(payload["capacity_selection"]),
        blind_consumption_marker=marker,
        held_out_manifest_inspection_deferred=False,
        train_cli_args=tuple(args),
        reference_ids_by_song=references,
    )


def launch_stage_edit_release_final(plan_path: Path) -> int:
    """Consume the generic blind run once, then delegate to the strict CLI."""

    preflight = preflight_stage_edit_release_final(plan_path)
    frozen_plan = _strict_final_plan(preflight.plan_path)
    readiness, model_contract = _verify_capacity_selection_binding(
        frozen_plan.get("capacity_selection")
    )
    if (
        frozen_plan.get("schema_version")
        != FINAL_RELEASE_PLAN_V3_SCHEMA_VERSION
        or readiness.get("binding_version")
        != V3_RELEASE_READINESS_BINDING_VERSION
        or model_contract.get("contract_version")
        != MODEL_TRAINING_CONTRACT_V3_VERSION
    ):
        raise ReleaseFoldPreflightError(
            "blind marker cannot be consumed by a legacy/non-v3 release plan"
        )
    assert preflight.blind_consumption_marker is not None
    marker = preflight.blind_consumption_marker
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker_payload = _blind_consumption_marker_payload(preflight)
    encoded = json.dumps(
        marker_payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    try:
        descriptor = os.open(
            marker,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_parent_directory(marker)
    except FileExistsError as exc:
        raise ReleaseFoldPreflightError(
            "blind evaluation was already consumed"
        ) from exc
    # The general CLI registers only when every fixed held-out gate passes.
    # A failed final blind run remains inactive and the marker remains consumed.
    consumed = _preflight_consumed_stage_edit_release_final(
        preflight.plan_path,
        consumption_marker=marker,
    )
    _require_dynamic_crop_v3_release_launch(
        frozen_plan,
        consumed.train_cli_args,
        activation_authorized=True,
    )
    return production_train_main(consumed.train_cli_args)


def _keyed_paths(
    values: Sequence[str],
    *,
    expected_song_ids: Sequence[str],
    option: str,
    content_inspection_deferred_song_ids: Sequence[str] = (),
) -> dict[str, Path]:
    output: dict[str, Path] = {}
    deferred = set(content_inspection_deferred_song_ids)
    if not deferred.issubset(set(expected_song_ids)):
        raise ReleaseFoldPreflightError(
            f"{option} deferred songs are outside the expected fold"
        )
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
        candidate = Path(raw_path).expanduser()
        if song_id in deferred:
            if not candidate.is_absolute():
                raise ReleaseFoldPreflightError(
                    f"deferred {option} path must be absolute"
                )
            output[song_id] = Path(os.path.normpath(str(candidate)))
        else:
            output[song_id] = candidate.resolve()
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
        description="Run the frozen IVE StageEdit development-to-blind release protocol"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create_dev = commands.add_parser("create-development")
    _add_manifest_options(create_dev)
    create_dev.add_argument("--checkpoint-dir", required=True, type=Path)
    create_dev.add_argument("--output-directory", required=True, type=Path)
    create_dev.add_argument(
        "--capacity-selection",
        type=Path,
        help=(
            "Legacy read/replay compatibility only. Current CLI creation "
            "requires the v3 freeze and final-fit receipt and rejects this."
        ),
    )
    create_dev.add_argument(
        "--dynamic-crop-v3-release-freeze",
        required=True,
        type=Path,
        help="Verified content-addressed three-fold Dynamic Crop v3 LOSO freeze.",
    )
    create_dev.add_argument(
        "--dynamic-crop-v3-final-fit-receipt",
        required=True,
        type=Path,
        help="Verified inactive all-reference Dynamic Crop v3 final-fit receipt.",
    )
    for name in ("preflight-development", "launch-development"):
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True, type=Path)
    freeze = commands.add_parser("freeze-development")
    freeze.add_argument("--plan", required=True, type=Path)
    freeze.add_argument("--output-directory", required=True, type=Path)
    create_final = commands.add_parser("create-final")
    _add_manifest_options(create_final)
    create_final.add_argument(
        "--blind-song-id",
        required=True,
        help=(
            "New canonical IVE song id frozen as the one-shot final holdout; "
            "development songs and quarantined ATTITUDE are rejected."
        ),
    )
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
            capacity_selection_path=args.capacity_selection,
            dynamic_crop_v3_release_freeze_path=(
                args.dynamic_crop_v3_release_freeze
            ),
            dynamic_crop_v3_final_fit_receipt_path=(
                args.dynamic_crop_v3_final_fit_receipt
            ),
        )
        print(json.dumps({"plan_path": str(path)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "preflight-development":
        result = preflight_stage_edit_release_development(args.plan)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "launch-development":
        return launch_stage_edit_release_development(args.plan)
    if args.command == "freeze-development":
        path = freeze_stage_edit_release_development(
            args.plan, output_directory=args.output_directory
        )
        print(json.dumps({"freeze_path": str(path)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "create-final":
        # Identity rejection happens before _keyed_paths can resolve/stat any
        # caller-supplied final manifest path.
        blind_song_id = _validate_blind_song_id(args.blind_song_id)
        expected = _final_song_ids(blind_song_id)
        path = create_stage_edit_release_final_plan(
            blind_song_id=blind_song_id,
            development_plan_path=args.development_plan,
            development_freeze_path=args.development_freeze,
            materialization_manifests=_keyed_paths(
                args.materialization_manifest,
                expected_song_ids=expected,
                option="--materialization-manifest",
                content_inspection_deferred_song_ids=(blind_song_id,),
            ),
            feature_manifests=_keyed_paths(
                args.feature_manifest,
                expected_song_ids=expected,
                option="--feature-manifest",
                content_inspection_deferred_song_ids=(blind_song_id,),
            ),
            dynamic_crop_manifests=_keyed_paths(
                args.dynamic_crop_manifest,
                expected_song_ids=expected,
                option="--dynamic-crop-manifest",
                content_inspection_deferred_song_ids=(blind_song_id,),
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
    "FINAL_TRAINING_SONG_IDS",
    "ReleaseFoldPreflight",
    "ReleaseFoldPreflightError",
    "create_stage_edit_release_development_plan",
    "create_stage_edit_release_final_plan",
    "freeze_stage_edit_release_development",
    "launch_stage_edit_release_final",
    "launch_stage_edit_release_development",
    "preflight_stage_edit_release_development",
    "preflight_stage_edit_release_final",
]
