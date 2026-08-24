"""Fresh-process, immutable-evidence MLX capacity smoke for Dynamic Crop v3.

This module deliberately has a narrower authority than the Dynamic Crop v2
capacity selector.  It proves that one pinned v3 model can execute a real
forward pass, backward pass, global-gradient clip, and AdamW update at the
release tensor shape ``[2, 2048, 32, 720]``.  It never computes semantic
quality metrics and its artifacts can never authorize model selection.

The input lanes are tiled only from the immutable BANG BANG, BLACKHOLE, and
REBEL HEART feature, supervision, and crop artifacts already verified by the
v2 smoke loader.  V2 trajectory evidence is simplified into v3 targets; no v2
model, v2 loss, or v2 checkpoint is accepted as a v3 runtime result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .stage_edit_dynamic_crop_capacity_probe import (
    PROBE_TRAINING_SCHEDULE,
    SELECTION_SONG_IDS,
    SMOKE_CACHE_LIMIT_BYTES,
    DeviceMemoryBudget,
    _addressed_payload,
    _atomic_exclusive_write,
    default_dynamic_crop_v2_probe_candidates,
)
from .stage_edit_dynamic_crop_head_v3 import (
    DYNAMIC_CROP_V3_LOSS_CONTRACT,
    DYNAMIC_CROP_V3_MAX_KEYFRAMES,
    DYNAMIC_CROP_V3_MODEL_CONTRACT,
    DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
    DYNAMIC_CROP_V3_SCHEMA_VERSION,
    DYNAMIC_CROP_V3_TARGET_CONTRACT,
    DynamicCropHeadV3Config,
    DynamicCropV3LossContract,
    StageEditDynamicCropSupervisionBatchV3,
    StageEditDynamicCropTargetSequenceV3,
    StageEditTransformerDynamicCropV3,
    stage_edit_dynamic_crop_v3_loss,
    stage_edit_dynamic_crop_v3_parameter_count,
)
from .stage_edit_dynamic_crop_mlx_smoke_cli import (
    FIXED_BATCH_SIZE,
    FIXED_MAX_SOURCES,
    FIXED_SEQUENCE_LENGTH,
    FIXED_SOURCE_DIMENSION,
    _all_gradients_finite,
    _build_isolated_worker_environment,
    _file_sha256,
    _hardware_provenance,
    _lane_assignments,
    _live_budget,
    _load_bbr_feature_dataset,
    _load_bbr_target_bindings,
    _pack_fixed_lane_training_arrays,
    _parse_manifest_map,
    _pointer,
    _runtime_provenance,
    _safe_mlx_memory,
    _sha256,
    _sync_memory_stage,
    _tensor_digest,
    _to_mlx_base_supervision,
    _to_mlx_batch,
    _tree_evidence,
    _worker_failure_kind,
)
from .stage_edit_reference_adapter import ReferenceTrainingExample
from .stage_edit_transformer import (
    DYNAMIC_CROP_V3_ACTIVATION_CHECKPOINTING_CONTRACT,
    StageEditLossContract,
    StageEditSupervisionBatch,
    StageEditTensorBatch,
    valid_dynamic_crop_v3_activation_checkpointing_evidence,
    StageEditTransformerConfig,
)


try:  # Optional on non-Apple unit-test hosts.
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
except ImportError as exc:  # pragma: no cover - production host has MLX.
    mx = None
    nn = None
    optim = None
    tree_flatten = None
    _MLX_IMPORT_ERROR: ImportError | None = exc
else:
    _MLX_IMPORT_ERROR = None


V3_CAPACITY_SMOKE_SCHEMA_VERSION = 2
V3_CAPACITY_SMOKE_ANALYSIS_BASIS = (
    "real_mlx_dynamic_crop_v3_rematerialized_release_shape_capacity_smoke_v2"
)
V3_CAPACITY_SMOKE_ARTIFACT_PREFIX = "dynamic-crop-v3-mlx-capacity-smoke-"
V3_CAPACITY_SMOKE_LOG_PREFIX = "dynamic-crop-v3-mlx-capacity-log-"
V3_CAPACITY_SMOKE_RUNNER_CONTRACT = (
    "dynamic_crop_v3_rematerialized_forward_backward_clip_adamw_fresh_process_v2"
)
V3_CAPACITY_SMOKE_PACKING_CONTRACT = (
    "immutable_bbr15_source_exact_v3_fixed_lane_2x2048x32x720_v1"
)
V3_CAPACITY_SMOKE_PROCESS_CONTRACT = (
    "one_supervisor_one_fresh_worker_one_pinned_rematerialized_v3_candidate_v2"
)
V3_CAPACITY_SMOKE_TARGET_CONTRACT = (
    "actual_bbr_v2_renderer_evidence_simplified_to_v3_capacity_only_v1"
)
V3_CAPACITY_SMOKE_EVENT_PREFIX = "STAGEEDIT_V3_MLX_SMOKE_EVENT="
V3_CAPACITY_SMOKE_NONCE_ENV = "STAGEEDIT_V3_MLX_SMOKE_RUN_NONCE"
V3_CAPACITY_SMOKE_ARTIFACT_FAMILY = "dynamic_crop_v3_capacity_smoke"
V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE: Mapping[str, Any] = (
    MappingProxyType(
        {
            "contract": DYNAMIC_CROP_V3_ACTIVATION_CHECKPOINTING_CONTRACT,
            "enabled": True,
            "scope": "base_transformer_encoder_all_layers",
            "implementation": "mlx.nn.TransformerEncoder.checkpoint",
            "training_only": True,
            "serialized_parameter_tree_changed": False,
            "inference_runtime_requires_checkpointing": False,
            "previous_value": False,
        }
    )
)
V3_CAPACITY_SMOKE_SOURCE_RELATIVE_PATHS: Mapping[str, str] = MappingProxyType(
    {
        "v3_capacity_runner": (
            "engine/crosscut/stage_edit_dynamic_crop_v3_capacity_smoke.py"
        ),
        "v3_dynamic_crop_head": (
            "engine/crosscut/stage_edit_dynamic_crop_head_v3.py"
        ),
        "stage_edit_transformer": (
            "engine/crosscut/stage_edit_transformer.py"
        ),
        "immutable_bbr_loader": (
            "engine/crosscut/stage_edit_dynamic_crop_mlx_smoke_cli.py"
        ),
    }
)
_PINNED_CANDIDATE_NAMES = (
    "compact-balanced",
    "intermediate-balanced",
    "production-balanced",
    "production-trajectory-emphasis",
)


class DynamicCropV3CapacitySmokeError(ValueError):
    pass


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_local(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _exact_activation_checkpointing_evidence(value: object) -> bool:
    return valid_dynamic_crop_v3_activation_checkpointing_evidence(value)


def _validate_capacity_smoke_code_provenance(
    value: object,
    *,
    require_clean_release: bool = False,
) -> dict[str, Any]:
    """Validate the complete clean-commit source receipt without trusting flags."""

    if not isinstance(value, Mapping):
        raise DynamicCropV3CapacitySmokeError(
            "v3 capacity smoke code provenance must be an object"
        )
    payload = dict(value)
    expected_keys = {
        "runner_contract",
        "expected_git_commit",
        "observed_git_commit",
        "tracked_diff_policy",
        "repository_status_clean",
        "repository_status_sha256",
        "source_files_match_observed_commit",
        "source_files",
        "release_trust_authorized",
    }
    if set(payload) != expected_keys:
        raise DynamicCropV3CapacitySmokeError(
            "v3 capacity smoke code provenance keys are not exact"
        )
    expected_commit = payload.get("expected_git_commit")
    observed_commit = payload.get("observed_git_commit")
    tracked_diff_policy = payload.get("tracked_diff_policy")
    repository_clean = payload.get("repository_status_clean")
    status_sha256 = payload.get("repository_status_sha256")
    if (
        payload.get("runner_contract") != V3_CAPACITY_SMOKE_RUNNER_CONTRACT
        or not isinstance(expected_commit, str)
        or len(expected_commit) != 40
        or any(character not in "0123456789abcdef" for character in expected_commit)
        or not isinstance(observed_commit, str)
        or len(observed_commit) != 40
        or any(character not in "0123456789abcdef" for character in observed_commit)
        or tracked_diff_policy not in {"require_clean", "record_development"}
        or not isinstance(repository_clean, bool)
        or not _is_sha256(status_sha256)
        or (repository_clean and status_sha256 != hashlib.sha256(b"").hexdigest())
        or not isinstance(payload.get("source_files_match_observed_commit"), bool)
        or not isinstance(payload.get("release_trust_authorized"), bool)
    ):
        raise DynamicCropV3CapacitySmokeError(
            "v3 capacity smoke code provenance declarations are invalid"
        )
    source_files = payload.get("source_files")
    if (
        not isinstance(source_files, Mapping)
        or set(source_files) != set(V3_CAPACITY_SMOKE_SOURCE_RELATIVE_PATHS)
    ):
        raise DynamicCropV3CapacitySmokeError(
            "v3 capacity smoke source provenance set is incomplete"
        )
    all_match = True
    for role, expected_relative in V3_CAPACITY_SMOKE_SOURCE_RELATIVE_PATHS.items():
        record = source_files.get(role)
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "file_sha256",
            "repo_relative_path",
            "observed_commit_blob_sha256",
            "matches_observed_commit",
        }:
            raise DynamicCropV3CapacitySmokeError(
                f"v3 capacity smoke source record is invalid: {role}"
            )
        raw_path = record.get("path")
        file_sha256 = record.get("file_sha256")
        blob_sha256 = record.get("observed_commit_blob_sha256")
        matches_observed = record.get("matches_observed_commit")
        if (
            not isinstance(raw_path, str)
            or not Path(raw_path).is_absolute()
            or record.get("repo_relative_path") != expected_relative
            or not _is_sha256(file_sha256)
            or (blob_sha256 is not None and not _is_sha256(blob_sha256))
            or not isinstance(matches_observed, bool)
            or matches_observed != (blob_sha256 == file_sha256)
            or tuple(Path(raw_path).parts[-len(Path(expected_relative).parts) :])
            != Path(expected_relative).parts
        ):
            raise DynamicCropV3CapacitySmokeError(
                f"v3 capacity smoke source record differs: {role}"
            )
        all_match = all_match and matches_observed
    release_trust = bool(
        tracked_diff_policy == "require_clean"
        and observed_commit == expected_commit
        and repository_clean
        and all_match
    )
    if (
        payload.get("source_files_match_observed_commit") is not all_match
        or payload.get("release_trust_authorized") is not release_trust
        or (require_clean_release and not release_trust)
    ):
        raise DynamicCropV3CapacitySmokeError(
            "v3 capacity smoke code provenance is not clean release evidence"
        )
    return payload


def _enable_v3_capacity_activation_checkpointing(
    model: StageEditTransformerDynamicCropV3,
) -> dict[str, Any]:
    """Enable the exact production v3 MLX recomputation path."""

    if not isinstance(model, StageEditTransformerDynamicCropV3):
        raise TypeError(
            "v3 capacity activation checkpointing requires the v3 model"
        )
    encoder = getattr(getattr(model, "base", None), "encoder", None)
    if encoder is None or not hasattr(encoder, "checkpoint"):
        raise RuntimeError(
            "MLX TransformerEncoder checkpoint support is unavailable"
        )
    previous = getattr(encoder, "checkpoint")
    if previous is not False:
        raise RuntimeError(
            "v3 capacity smoke requires checkpoint previous_value=false"
        )
    encoder.checkpoint = True
    if encoder.checkpoint is not True:
        raise RuntimeError("v3 activation checkpointing did not remain enabled")
    return dict(V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE)


@dataclass(frozen=True, slots=True)
class DynamicCropV3CapacityCandidate:
    """A pinned v3 runtime contract, never a semantic selection result."""

    candidate_name: str
    model_config: StageEditTransformerConfig
    base_loss_contract: StageEditLossContract
    dynamic_crop_config: DynamicCropHeadV3Config
    dynamic_crop_loss_contract: DynamicCropV3LossContract

    def __post_init__(self) -> None:
        if self.candidate_name not in _PINNED_CANDIDATE_NAMES:
            raise DynamicCropV3CapacitySmokeError("unknown v3 capacity candidate")
        if self.dynamic_crop_config != DynamicCropHeadV3Config():
            raise DynamicCropV3CapacitySmokeError(
                "v3 capacity smoke requires the exact release crop config"
            )
        if self.dynamic_crop_loss_contract.base != self.base_loss_contract:
            raise DynamicCropV3CapacitySmokeError(
                "v3 and base loss contracts differ"
            )
        if self.dynamic_crop_loss_contract.contract != DYNAMIC_CROP_V3_LOSS_CONTRACT:
            raise DynamicCropV3CapacitySmokeError("v3 loss contract differs")
        if self.model_config.max_sequence_length != FIXED_SEQUENCE_LENGTH:
            raise DynamicCropV3CapacitySmokeError(
                "v3 capacity candidate does not use release sequence length"
            )
        if self.model_config.max_sources != FIXED_MAX_SOURCES:
            raise DynamicCropV3CapacitySmokeError(
                "v3 capacity candidate does not use release source count"
            )
        if self.model_config.source_input_dimension != FIXED_SOURCE_DIMENSION:
            raise DynamicCropV3CapacitySmokeError(
                "v3 capacity candidate source dimension differs"
            )

    @property
    def parameter_count(self) -> int:
        return stage_edit_dynamic_crop_v3_parameter_count(
            self.model_config, self.dynamic_crop_config
        )

    def contract_dict(self) -> dict[str, Any]:
        return {
            "dynamic_crop_version": DYNAMIC_CROP_V3_SCHEMA_VERSION,
            "model_contract": DYNAMIC_CROP_V3_MODEL_CONTRACT,
            "target_contract": DYNAMIC_CROP_V3_TARGET_CONTRACT,
            "output_contract": DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
            "loss_contract": DYNAMIC_CROP_V3_LOSS_CONTRACT,
            "model_class": "StageEditTransformerDynamicCropV3",
            "loss_class": "DynamicCropV3LossContract",
            "activation_checkpointing": dict(
                V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE
            ),
            "legacy_v2_checkpoint_compatible": False,
            "v2_model_or_loss_accepted": False,
            "model_config": self.model_config.to_dict(),
            "base_loss_contract": asdict(self.base_loss_contract),
            "dynamic_crop_config": self.dynamic_crop_config.to_dict(),
            "dynamic_crop_loss_contract": asdict(
                self.dynamic_crop_loss_contract
            ),
            "parameter_count": self.parameter_count,
            "semantic_model_selection_authorized": False,
        }

    @property
    def candidate_id(self) -> str:
        return _sha256_local(self.contract_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "candidate_id": self.candidate_id,
            "contract": self.contract_dict(),
        }


def default_dynamic_crop_v3_capacity_candidates(
) -> tuple[DynamicCropV3CapacityCandidate, ...]:
    """Promote only the pinned base capacities; never reuse a v2 crop head."""

    result: list[DynamicCropV3CapacityCandidate] = []
    for legacy in default_dynamic_crop_v2_probe_candidates():
        loss_overrides: dict[str, float] = {}
        if legacy.candidate_name == "production-trajectory-emphasis":
            loss_overrides = {
                "value_weight": 1.15,
                "time_weight": 0.65,
                "count_weight": 0.40,
                "motion_weight": 0.25,
            }
        result.append(
            DynamicCropV3CapacityCandidate(
                candidate_name=legacy.candidate_name,
                model_config=legacy.model_config,
                base_loss_contract=legacy.base_loss_contract,
                dynamic_crop_config=DynamicCropHeadV3Config(),
                dynamic_crop_loss_contract=DynamicCropV3LossContract(
                    base=legacy.base_loss_contract, **loss_overrides
                ),
            )
        )
    if tuple(item.candidate_name for item in result) != _PINNED_CANDIDATE_NAMES:
        raise DynamicCropV3CapacitySmokeError(
            "pinned v3 candidate order differs from the v2 capacity base"
        )
    return tuple(result)


def _candidate_by_name(name: str) -> DynamicCropV3CapacityCandidate:
    matches = tuple(
        item
        for item in default_dynamic_crop_v3_capacity_candidates()
        if item.candidate_name == name
    )
    if len(matches) != 1:
        raise DynamicCropV3CapacitySmokeError("unknown v3 candidate name")
    return matches[0]


def _legacy_candidate_by_name(name: str) -> Any:
    matches = tuple(
        item
        for item in default_dynamic_crop_v2_probe_candidates()
        if item.candidate_name == name
    )
    if len(matches) != 1:
        raise DynamicCropV3CapacitySmokeError("v2 base capacity is not pinned")
    return matches[0]


def _pack_fixed_lane_training_arrays_v3(
    *,
    examples_by_reference: Mapping[str, ReferenceTrainingExample],
    dynamic_by_reference: Mapping[str, Any],
    lane_assignments: Mapping[str, Sequence[str]],
    candidate: DynamicCropV3CapacityCandidate,
    sequence_length: int = FIXED_SEQUENCE_LENGTH,
    max_sources: int = FIXED_MAX_SOURCES,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """Tile exact BBR tokens and place every v3 target in its source slot."""

    legacy = _legacy_candidate_by_name(candidate.candidate_name)
    inputs, base, _unused_v2_dynamic, legacy_provenance = (
        _pack_fixed_lane_training_arrays(
            examples_by_reference=examples_by_reference,
            dynamic_by_reference=dynamic_by_reference,
            lane_assignments=lane_assignments,
            candidate=legacy,
            sequence_length=sequence_length,
            max_sources=max_sources,
        )
    )
    promoted = {
        reference: StageEditDynamicCropTargetSequenceV3.from_v2(sequence)
        for reference, sequence in dynamic_by_reference.items()
    }
    cycles: dict[str, list[tuple[Any, Any]]] = {}
    for lane in ("A", "B"):
        records: list[tuple[Any, Any]] = []
        for reference in lane_assignments[lane]:
            example = examples_by_reference[reference]
            sequence = promoted[reference]
            if len(example.sequence.tokens) != len(sequence.targets):
                raise DynamicCropV3CapacitySmokeError(
                    "v3 feature and target lengths differ"
                )
            records.extend(zip(example.sequence.tokens, sequence.targets, strict=True))
        if not records or len(records) > sequence_length:
            raise DynamicCropV3CapacitySmokeError("v3 fixed lane cycle is invalid")
        cycles[lane] = records

    keyframes = DYNAMIC_CROP_V3_MAX_KEYFRAMES
    shape = (FIXED_BATCH_SIZE, sequence_length, max_sources)
    values = np.zeros((*shape, keyframes, 4), dtype=np.float32)
    times = np.zeros((*shape, keyframes), dtype=np.float32)
    keyframe_mask = np.zeros((*shape, keyframes), dtype=bool)
    counts = np.zeros(shape, dtype=np.int32)
    count_mask = np.zeros(shape, dtype=bool)
    motion_targets = np.zeros(shape, dtype=np.float32)
    motion_mask = np.zeros(shape, dtype=bool)
    durations = np.zeros(shape, dtype=np.float32)
    track_mask = np.zeros(shape, dtype=bool)
    target_source_indices = np.full(
        (FIXED_BATCH_SIZE, sequence_length), -1, dtype=np.int32
    )

    for lane_index, lane in enumerate(("A", "B")):
        cycle = cycles[lane]
        for time_index in range(sequence_length):
            token, target = cycle[time_index % len(cycle)]
            if target is None:
                continue
            matches = tuple(
                index
                for index, source in enumerate(token.source_candidates)
                if source.source_id == target.source_id
            )
            if len(matches) != 1:
                raise DynamicCropV3CapacitySmokeError(
                    "v3 target source has no unique candidate slot"
                )
            source_index = matches[0]
            if not inputs["source_candidate_mask"][
                lane_index, time_index, source_index
            ]:
                raise DynamicCropV3CapacitySmokeError(
                    "v3 target source is unavailable"
                )
            if not inputs["hard_source_gate_mask"][
                lane_index, time_index, source_index
            ]:
                raise DynamicCropV3CapacitySmokeError(
                    "v3 target source failed its hard gate"
                )
            count = len(target.keyframes)
            if not 1 <= count <= keyframes:
                raise DynamicCropV3CapacitySmokeError(
                    "v3 target requires one through eight keyframes"
                )
            duration = float(target.shot_duration_seconds)
            if not math.isfinite(duration) or duration <= 0.0:
                raise DynamicCropV3CapacitySmokeError(
                    "v3 target duration is invalid"
                )
            target_source_indices[lane_index, time_index] = source_index
            track_mask[lane_index, time_index, source_index] = True
            count_mask[lane_index, time_index, source_index] = True
            counts[lane_index, time_index, source_index] = count
            durations[lane_index, time_index, source_index] = duration
            if not target.motion_label_loss_masked:
                motion_mask[lane_index, time_index, source_index] = True
                motion_targets[lane_index, time_index, source_index] = float(
                    target.motion_label == "dynamic"
                )
            for keyframe_index, keyframe in enumerate(target.keyframes):
                normalized_time = float(keyframe.time_seconds) / duration
                if not 0.0 <= normalized_time <= 1.0:
                    raise DynamicCropV3CapacitySmokeError(
                        "v3 target time is outside the shot"
                    )
                values[
                    lane_index, time_index, source_index, keyframe_index
                ] = (
                    keyframe.zoom,
                    keyframe.center_x,
                    keyframe.center_y,
                    keyframe.roll_degrees,
                )
                times[
                    lane_index, time_index, source_index, keyframe_index
                ] = normalized_time
                keyframe_mask[
                    lane_index, time_index, source_index, keyframe_index
                ] = True

    dynamic = {
        "values": values,
        "normalized_times": times,
        "keyframe_mask": keyframe_mask,
        "count_targets": counts,
        "count_mask": count_mask,
        "motion_targets": motion_targets,
        "motion_mask": motion_mask,
        "shot_duration_seconds": durations,
        "track_mask": track_mask,
        "target_source_indices": target_source_indices,
    }
    supervision_digest = _tensor_digest(
        tuple((f"base.{name}", value) for name, value in base.items())
        + tuple((f"dynamic_v3.{name}", value) for name, value in dynamic.items())
    )
    provenance = {
        **{
            key: value
            for key, value in legacy_provenance.items()
            if key
            not in {
                "packing_contract",
                "dynamic_values_shape",
                "dynamic_times_shape",
                "dynamic_presence_shape",
                "dynamic_label_mask_counts",
                "parameter_count",
                "supervision_tensor_sha256",
            }
        },
        "packing_contract": V3_CAPACITY_SMOKE_PACKING_CONTRACT,
        "dynamic_crop_version": DYNAMIC_CROP_V3_SCHEMA_VERSION,
        "model_contract": DYNAMIC_CROP_V3_MODEL_CONTRACT,
        "target_contract": DYNAMIC_CROP_V3_TARGET_CONTRACT,
        "loss_contract": DYNAMIC_CROP_V3_LOSS_CONTRACT,
        "output_contract": DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
        "source_tensor_shape": list(inputs["source_features"].shape),
        "dynamic_values_shape": list(values.shape),
        "dynamic_times_shape": list(times.shape),
        "dynamic_count_shape": list(counts.shape),
        "dynamic_target_source_indices_shape": list(
            target_source_indices.shape
        ),
        "dynamic_label_mask_counts": {
            "tracks": int(track_mask.sum()),
            "keyframes": int(keyframe_mask.sum()),
            "count": int(count_mask.sum()),
            "motion": int(motion_mask.sum()),
        },
        "source_slot_mismatch_count": 0,
        "parameter_count": candidate.parameter_count,
        "supervision_tensor_sha256": supervision_digest,
        "immutable_v2_trajectory_evidence_promoted_to_v3": True,
        "legacy_v2_model_or_loss_used": False,
        "semantic_metrics_authorized": False,
    }
    if provenance["source_tensor_shape"] != [2, 2048, 32, 720] and (
        sequence_length == FIXED_SEQUENCE_LENGTH and max_sources == FIXED_MAX_SOURCES
    ):
        raise DynamicCropV3CapacitySmokeError(
            "v3 release source tensor shape differs"
        )
    return inputs, base, dynamic, provenance


def _to_mlx_dynamic_supervision_v3(
    arrays: Mapping[str, np.ndarray],
) -> StageEditDynamicCropSupervisionBatchV3:
    if mx is None:
        raise RuntimeError("MLX is unavailable")
    return StageEditDynamicCropSupervisionBatchV3(
        **{name: mx.array(value) for name, value in arrays.items()},
        sequence_ids=("mlx-v3-smoke-lane-A", "mlx-v3-smoke-lane-B"),
    )


def _git_bytes(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ("git", *args),
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise DynamicCropV3CapacitySmokeError(
            f"git {' '.join(args)} failed with {result.returncode}"
        )
    return result.stdout


def _code_provenance(
    *, expected_git_commit: str, tracked_diff_policy: str
) -> tuple[dict[str, Any], Path]:
    runner = Path(__file__).resolve()
    repo = Path(
        _git_bytes(runner.parent, "rev-parse", "--show-toplevel")
        .decode("utf-8")
        .strip()
    ).resolve()
    observed = (
        _git_bytes(repo, "rev-parse", "HEAD").decode("ascii").strip().lower()
    )
    expected = str(expected_git_commit).strip().lower()
    if len(expected) != 40 or any(c not in "0123456789abcdef" for c in expected):
        raise DynamicCropV3CapacitySmokeError(
            "expected git commit must be a full SHA-1"
        )
    if tracked_diff_policy not in {"require_clean", "record_development"}:
        raise DynamicCropV3CapacitySmokeError("tracked diff policy is invalid")
    status = _git_bytes(repo, "status", "--porcelain=v1", "--untracked-files=all")
    source_paths = {
        role: repo / relative
        for role, relative in V3_CAPACITY_SMOKE_SOURCE_RELATIVE_PATHS.items()
    }
    if source_paths["v3_capacity_runner"] != runner:
        raise DynamicCropV3CapacitySmokeError(
            "v3 capacity runner path differs from its committed provenance path"
        )
    files: dict[str, Any] = {}
    all_match = True
    for role, path in source_paths.items():
        relative = str(path.relative_to(repo))
        blob = subprocess.run(
            ("git", "show", f"{observed}:{relative}"),
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        blob_hash = hashlib.sha256(blob.stdout).hexdigest() if not blob.returncode else None
        file_hash = _file_sha256(path)
        matched = blob_hash == file_hash
        all_match = all_match and matched
        files[role] = {
            "path": str(path),
            "file_sha256": file_hash,
            "repo_relative_path": relative,
            "observed_commit_blob_sha256": blob_hash,
            "matches_observed_commit": matched,
        }
    clean = not status
    release_trust = (
        tracked_diff_policy == "require_clean"
        and observed == expected
        and clean
        and all_match
    )
    return (
        {
            "runner_contract": V3_CAPACITY_SMOKE_RUNNER_CONTRACT,
            "expected_git_commit": expected,
            "observed_git_commit": observed,
            "tracked_diff_policy": tracked_diff_policy,
            "repository_status_clean": clean,
            "repository_status_sha256": hashlib.sha256(status).hexdigest(),
            "source_files_match_observed_commit": all_match,
            "source_files": files,
            "release_trust_authorized": release_trust,
        },
        repo,
    )


def _emit_event(value: Mapping[str, Any]) -> None:
    print(
        V3_CAPACITY_SMOKE_EVENT_PREFIX
        + _canonical_bytes(dict(value)).decode("utf-8"),
        flush=True,
    )


def _parse_events(stdout: str) -> tuple[dict[str, Any], ...]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.startswith(V3_CAPACITY_SMOKE_EVENT_PREFIX):
            continue
        try:
            value = json.loads(line[len(V3_CAPACITY_SMOKE_EVENT_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise DynamicCropV3CapacitySmokeError(
                "v3 worker emitted malformed JSON"
            ) from exc
        if not isinstance(value, dict):
            raise DynamicCropV3CapacitySmokeError(
                "v3 worker event must be an object"
            )
        events.append(value)
    return tuple(events)


def _require_mlx() -> None:
    if mx is None or nn is None or optim is None or tree_flatten is None:
        detail = f": {_MLX_IMPORT_ERROR}" if _MLX_IMPORT_ERROR else ""
        raise RuntimeError(f"Dynamic Crop v3 capacity smoke requires MLX{detail}")


def _worker_main(args: argparse.Namespace) -> int:
    run_nonce = os.environ.pop(V3_CAPACITY_SMOKE_NONCE_ENV, "")
    nonce_valid = _is_sha256(run_nonce)
    nonce_sha256 = (
        hashlib.sha256(run_nonce.encode("ascii")).hexdigest()
        if nonce_valid
        else None
    )
    context = {
        "worker_pid": os.getpid(),
        "suite_id": args.suite_id,
        "run_nonce_sha256": nonce_sha256,
    }
    try:
        if not nonce_valid or not _is_sha256(args.suite_id):
            raise DynamicCropV3CapacitySmokeError(
                "worker requires a fresh supervisor nonce and suite id"
            )
        _require_mlx()
        assert mx is not None and nn is not None and optim is not None
        candidate = _candidate_by_name(args.candidate_name)
        if candidate.candidate_id != args.expected_candidate_id:
            raise DynamicCropV3CapacitySmokeError("worker candidate differs")
        feature_map = _parse_manifest_map(
            args.feature_manifest, argument_name="feature manifest"
        )
        materialization_map = _parse_manifest_map(
            args.materialization_manifest,
            argument_name="materialization manifest",
        )
        crop_map = _parse_manifest_map(
            args.dynamic_crop_manifest, argument_name="dynamic crop manifest"
        )
        feature_provenance, snapshots, references = _load_bbr_feature_dataset(
            feature_map
        )
        target_provenance, examples, dynamic_targets = _load_bbr_target_bindings(
            materialization_manifest_paths_by_song=materialization_map,
            dynamic_crop_manifest_paths_by_song=crop_map,
            snapshots_by_reference=snapshots,
            references_by_song=references,
            build_training_corpora=True,
        )
        dataset = {**feature_provenance, **target_provenance}
        code, repo = _code_provenance(
            expected_git_commit=args.expected_git_commit,
            tracked_diff_policy=args.tracked_diff_policy,
        )
        budget = _live_budget()
        runtime = _runtime_provenance(budget, repo)
        hardware = _hardware_provenance(budget)
        observed_hashes = {
            "dataset": _sha256(dataset),
            "code": _sha256(code),
            "runtime": _sha256(runtime),
            "hardware": _sha256(hardware),
        }
        expected_hashes = {
            "dataset": args.expected_dataset_sha256,
            "code": args.expected_code_sha256,
            "runtime": args.expected_runtime_sha256,
            "hardware": args.expected_hardware_sha256,
        }
        if observed_hashes != expected_hashes:
            raise DynamicCropV3CapacitySmokeError(
                "worker provenance handshake differs"
            )
        arrays, base_arrays, dynamic_arrays, tensor = (
            _pack_fixed_lane_training_arrays_v3(
                examples_by_reference=examples,
                dynamic_by_reference=dynamic_targets,
                lane_assignments=_lane_assignments(references),
                candidate=candidate,
            )
        )
        if tensor["source_tensor_shape"] != [2, 2048, 32, 720]:
            raise DynamicCropV3CapacitySmokeError(
                "worker did not materialize the release tensor shape"
            )

        previous_memory_limit = int(mx.set_memory_limit(budget.allowed_training_bytes))
        previous_cache_limit = int(mx.set_cache_limit(SMOKE_CACHE_LIMIT_BYTES))
        mx.clear_cache()
        mx.synchronize(mx.gpu)
        pre_reset_active = int(mx.get_active_memory())
        if pre_reset_active != 0:
            raise RuntimeError(
                "fresh v3 worker has active MLX memory before peak reset"
            )
        mx.reset_peak_memory()
        mx.random.seed(int(PROBE_TRAINING_SCHEDULE["seed"]))
        started_at = time.perf_counter()
        batch = _to_mlx_batch(arrays)
        batch = replace(
            batch,
            sequence_ids=("mlx-v3-smoke-lane-A", "mlx-v3-smoke-lane-B"),
        )
        base_supervision = _to_mlx_base_supervision(base_arrays)
        dynamic_supervision = _to_mlx_dynamic_supervision_v3(dynamic_arrays)
        mx.eval(
            batch.context_features,
            batch.source_features,
            batch.sequence_mask,
            batch.source_candidate_mask,
            batch.hard_source_gate_mask,
            *(
                getattr(base_supervision, name)
                for name in (
                    "cut_targets",
                    "cut_mask",
                    "source_targets",
                    "source_mask",
                    "rejected_source_targets",
                    "pairwise_source_mask",
                    "duration_targets",
                    "duration_mask",
                    "transition_targets",
                    "transition_mask",
                    "crop_targets",
                    "crop_mask",
                )
            ),
            *(
                getattr(dynamic_supervision, name)
                for name in (
                    "values",
                    "normalized_times",
                    "keyframe_mask",
                    "count_targets",
                    "count_mask",
                    "motion_targets",
                    "motion_mask",
                    "shot_duration_seconds",
                    "track_mask",
                    "target_source_indices",
                )
            ),
        )
        stages = [_sync_memory_stage("input_materialized", started_at)]
        model = StageEditTransformerDynamicCropV3(
            candidate.model_config, candidate.dynamic_crop_config
        )
        loss_contract = DynamicCropV3LossContract(
            **{
                **asdict(candidate.dynamic_crop_loss_contract),
                "base": candidate.base_loss_contract,
            }
        )
        model.train()
        mx.eval(model.parameters())
        parameter_tree_before_checkpointing = _tree_evidence(model.parameters())
        activation_checkpointing = _enable_v3_capacity_activation_checkpointing(
            model
        )
        parameter_tree = _tree_evidence(model.parameters())
        if (
            parameter_tree_before_checkpointing != parameter_tree
            or parameter_tree["element_count"] != candidate.parameter_count
        ):
            raise RuntimeError(
                "v3 activation checkpointing changed the parameter tree"
            )
        stages.append(_sync_memory_stage("model_allocated", started_at))
        optimizer = optim.AdamW(
            learning_rate=float(PROBE_TRAINING_SCHEDULE["learning_rate"]),
            weight_decay=float(PROBE_TRAINING_SCHEDULE["weight_decay"]),
        )

        def loss_function(
            trained_model: StageEditTransformerDynamicCropV3,
            smoke_batch: StageEditTensorBatch,
            smoke_base: StageEditSupervisionBatch,
            smoke_dynamic: StageEditDynamicCropSupervisionBatchV3,
        ) -> Any:
            return stage_edit_dynamic_crop_v3_loss(
                trained_model,
                smoke_batch,
                smoke_base,
                smoke_dynamic,
                loss_contract,
            )[0]

        loss_and_grad = nn.value_and_grad(model, loss_function)
        loss, gradients = loss_and_grad(
            model, batch, base_supervision, dynamic_supervision
        )
        mx.eval(loss, gradients)
        stages.append(
            _sync_memory_stage("forward_backward_completed", started_at)
        )
        loss_value = float(loss.item())
        gradient_tree = _tree_evidence(gradients)
        clipped, gradient_norm = optim.clip_grad_norm(
            gradients, float(PROBE_TRAINING_SCHEDULE["gradient_clip_norm"])
        )
        mx.eval(clipped, gradient_norm)
        stages.append(_sync_memory_stage("gradient_clip_completed", started_at))
        gradient_norm_value = float(gradient_norm.item())
        optimizer.update(model, clipped)
        mx.eval(model.parameters(), optimizer.state)
        stages.append(_sync_memory_stage("optimizer_step_completed", started_at))
        post_update_tree = _tree_evidence(model.parameters())
        optimizer_tree = _tree_evidence(optimizer.state)
        finite = (
            math.isfinite(loss_value)
            and math.isfinite(gradient_norm_value)
            and _all_gradients_finite(gradients)
            and post_update_tree["nonfinite_count"] == 0
            and optimizer_tree["nonfinite_count"] == 0
        )
        optimizer_state_materialized = bool(tree_flatten(optimizer.state))
        peak = int(mx.get_peak_memory())
        execution_passed = finite and optimizer_state_materialized
        _emit_event(
            {
                **context,
                "event": "worker_result",
                "dynamic_crop_version": 3,
                "model_class": "StageEditTransformerDynamicCropV3",
                "loss_class": "DynamicCropV3LossContract",
                "candidate_id": candidate.candidate_id,
                "provenance_hashes": observed_hashes,
                "tensor_provenance": tensor,
                "parameter_count": candidate.parameter_count,
                "activation_checkpointing": activation_checkpointing,
                "parameter_tree_before_checkpointing": (
                    parameter_tree_before_checkpointing
                ),
                "parameter_tree": parameter_tree,
                "gradient_tree": gradient_tree,
                "post_update_parameter_tree": post_update_tree,
                "optimizer_state_tree": optimizer_tree,
                "loss_value_for_execution_only": loss_value,
                "gradient_global_norm_before_clip": gradient_norm_value,
                "gradients_and_state_finite": finite,
                "optimizer_state_materialized": optimizer_state_materialized,
                "memory_stages": stages,
                "measured_peak_bytes": peak,
                "allowed_training_bytes": budget.allowed_training_bytes,
                "within_release_memory_budget": peak <= budget.allowed_training_bytes,
                "previous_memory_limit_bytes": previous_memory_limit,
                "previous_cache_limit_bytes": previous_cache_limit,
                "applied_cache_limit_bytes": SMOKE_CACHE_LIMIT_BYTES,
                "runtime_execution_passed": execution_passed,
                "semantic_metrics_authorized": False,
                "model_selection_authorized": False,
                "smoke_target_contract": V3_CAPACITY_SMOKE_TARGET_CONTRACT,
            }
        )
        return 0 if execution_passed else 71
    except BaseException as exc:
        _emit_event(
            {
                **context,
                "event": "worker_failure",
                "dynamic_crop_version": 3,
                "model_class": "StageEditTransformerDynamicCropV3",
                "loss_class": "DynamicCropV3LossContract",
                "failure_kind": _worker_failure_kind(exc),
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:4000],
                "measured_peak_bytes": _safe_mlx_memory("get_peak_memory"),
                "semantic_metrics_authorized": False,
                "model_selection_authorized": False,
            }
        )
        return 70


def _content_addressed_payload(
    payload: Mapping[str, Any], *, hash_field: str = "artifact_content_sha256"
) -> dict[str, Any]:
    if hash_field in payload and payload[hash_field] is not None:
        raise DynamicCropV3CapacitySmokeError(
            "caller cannot pre-author a content hash"
        )
    unsigned = {**dict(payload), hash_field: None}
    digest = _sha256_local(unsigned)
    return {**unsigned, hash_field: digest}


def _write_log(
    destination: Path,
    *,
    command: Sequence[str],
    returncode: int,
    stdout: str,
    stderr: str,
    supervisor_pid: int,
    worker_pid: int,
    suite_id: str,
    run_nonce_sha256: str,
    started_ns: int,
    ended_ns: int,
) -> Path:
    payload = _content_addressed_payload(
        {
            "schema_version": 1,
            "analysis_basis": "dynamic_crop_v3_fresh_worker_process_log_v1",
            "suite_id": suite_id,
            "supervisor_pid": supervisor_pid,
            "worker_pid": worker_pid,
            "fresh_process": supervisor_pid != worker_pid,
            "run_nonce_sha256": run_nonce_sha256,
            "started_at_unix_ns": started_ns,
            "ended_at_unix_ns": ended_ns,
            "returncode": int(returncode),
            "command": list(command),
            "stdout": stdout,
            "stderr": stderr,
        }
    )
    digest = payload["artifact_content_sha256"]
    return _atomic_exclusive_write(
        destination / f"{V3_CAPACITY_SMOKE_LOG_PREFIX}{digest}.json", payload
    )


def _normalize_execution(
    *,
    event: Mapping[str, Any] | None,
    returncode: int,
    candidate: DynamicCropV3CapacityCandidate,
    budget: DeviceMemoryBudget,
    supervisor_pid: int,
    worker_pid: int,
    suite_id: str,
    run_nonce_sha256: str,
    expected_hashes: Mapping[str, str],
) -> dict[str, Any]:
    result = dict(event or {})

    def exact_parameter_tree(value: object) -> bool:
        return (
            isinstance(value, Mapping)
            and value.get("element_count") == candidate.parameter_count
            and isinstance(value.get("leaf_count"), int)
            and value.get("leaf_count", 0) > 0
            and value.get("nonfinite_count") == 0
            and _is_sha256(value.get("key_digest_sha256"))
        )

    parameter_trees_ok = all(
        exact_parameter_tree(result.get(name))
        for name in (
            "parameter_tree_before_checkpointing",
            "parameter_tree",
            "gradient_tree",
            "post_update_parameter_tree",
        )
    ) and result.get("parameter_tree_before_checkpointing") == result.get(
        "parameter_tree"
    )
    activation_checkpointing_ok = _exact_activation_checkpointing_evidence(
        result.get("activation_checkpointing")
    )
    optimizer_tree = result.get("optimizer_state_tree")
    optimizer_tree_ok = (
        isinstance(optimizer_tree, Mapping)
        and isinstance(optimizer_tree.get("element_count"), int)
        and optimizer_tree.get("element_count", 0) > candidate.parameter_count
        and isinstance(optimizer_tree.get("leaf_count"), int)
        and optimizer_tree.get("leaf_count", 0) > 0
        and optimizer_tree.get("nonfinite_count") == 0
        and _is_sha256(optimizer_tree.get("key_digest_sha256"))
    )
    scalar_evidence_ok = all(
        isinstance(result.get(name), (int, float))
        and not isinstance(result.get(name), bool)
        and math.isfinite(float(result[name]))
        for name in (
            "loss_value_for_execution_only",
            "gradient_global_norm_before_clip",
        )
    )
    event_ok = (
        result.get("event") == "worker_result"
        and result.get("dynamic_crop_version") == 3
        and result.get("model_class") == "StageEditTransformerDynamicCropV3"
        and result.get("loss_class") == "DynamicCropV3LossContract"
        and result.get("candidate_id") == candidate.candidate_id
        and result.get("suite_id") == suite_id
        and result.get("run_nonce_sha256") == run_nonce_sha256
        and result.get("worker_pid") == worker_pid
        and result.get("provenance_hashes") == dict(expected_hashes)
        and result.get("parameter_count") == candidate.parameter_count
        and activation_checkpointing_ok
        and result.get("semantic_metrics_authorized") is False
        and result.get("model_selection_authorized") is False
    )
    tensor = result.get("tensor_provenance")
    tensor_ok = (
        isinstance(tensor, Mapping)
        and tensor.get("source_tensor_shape") == [2, 2048, 32, 720]
        and tensor.get("dynamic_crop_version") == 3
        and tensor.get("model_contract") == DYNAMIC_CROP_V3_MODEL_CONTRACT
        and tensor.get("loss_contract") == DYNAMIC_CROP_V3_LOSS_CONTRACT
        and tensor.get("legacy_v2_model_or_loss_used") is False
        and tensor.get("real_input_materialized") is True
    )
    stages = result.get("memory_stages")
    expected_stages = (
        "input_materialized",
        "model_allocated",
        "forward_backward_completed",
        "gradient_clip_completed",
        "optimizer_step_completed",
    )
    stages_ok = (
        isinstance(stages, list)
        and tuple(
            stage.get("stage") if isinstance(stage, Mapping) else None
            for stage in stages
        )
        == expected_stages
        and all(
            isinstance(stage, Mapping)
            and stage.get("synchronized") is True
            and isinstance(stage.get("peak_memory_bytes"), int)
            for stage in stages
        )
    )
    peak = result.get("measured_peak_bytes")
    peak_valid = isinstance(peak, int) and not isinstance(peak, bool) and peak > 0
    stage_peak_consistent = (
        stages_ok
        and peak_valid
        and max(int(stage["peak_memory_bytes"]) for stage in stages) == peak
        and all(
            int(left["peak_memory_bytes"]) <= int(right["peak_memory_bytes"])
            for left, right in zip(stages, stages[1:])
        )
    )
    execution_passed = (
        returncode == 0
        and supervisor_pid != worker_pid
        and event_ok
        and tensor_ok
        and stages_ok
        and stage_peak_consistent
        and peak_valid
        and parameter_trees_ok
        and activation_checkpointing_ok
        and optimizer_tree_ok
        and scalar_evidence_ok
        and result.get("runtime_execution_passed") is True
        and result.get("gradients_and_state_finite") is True
        and result.get("optimizer_state_materialized") is True
    )
    within_budget = (
        execution_passed
        and peak_valid
        and peak <= budget.allowed_training_bytes
        and result.get("within_release_memory_budget") is True
    )
    return {
        "process_contract": V3_CAPACITY_SMOKE_PROCESS_CONTRACT,
        "supervisor_pid": supervisor_pid,
        "worker_pid": worker_pid,
        "fresh_process": supervisor_pid != worker_pid,
        "returncode": int(returncode),
        "runtime_execution_passed": execution_passed,
        "within_release_memory_budget": within_budget,
        "capacity_smoke_passed": execution_passed and within_budget,
        "failure_kind": (
            None
            if execution_passed
            else result.get("failure_kind", "invalid_worker_receipt")
        ),
        "measured_peak_bytes": peak if peak_valid else None,
        "allowed_training_bytes": budget.allowed_training_bytes,
        "parameter_count": candidate.parameter_count,
        "model_class": "StageEditTransformerDynamicCropV3",
        "loss_class": "DynamicCropV3LossContract",
        "activation_checkpointing": (
            dict(V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE)
            if activation_checkpointing_ok
            else None
        ),
        "tensor_provenance": dict(tensor) if isinstance(tensor, Mapping) else None,
        "memory_stages": stages if isinstance(stages, list) else [],
        "worker_receipt": result,
        "semantic_metrics_authorized": False,
        "model_selection_authorized": False,
    }


def _external_destination(raw: Path) -> Path:
    root = Path("/Volumes/CrossCutMedia").resolve()
    destination = Path(raw).expanduser().resolve()
    if not root.is_dir() or not os.path.ismount(root):
        raise DynamicCropV3CapacitySmokeError("/Volumes/CrossCutMedia is not mounted")
    if destination != root and root not in destination.parents:
        raise DynamicCropV3CapacitySmokeError(
            "v3 capacity smoke artifacts must stay on /Volumes/CrossCutMedia"
        )
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise DynamicCropV3CapacitySmokeError(
            "v3 capacity destination cannot be a symlink"
        )
    return destination


def _supervisor_main(args: argparse.Namespace) -> int:
    destination = _external_destination(args.destination)
    candidate = _candidate_by_name(args.candidate_name)
    feature_map = _parse_manifest_map(
        args.feature_manifest, argument_name="feature manifest"
    )
    materialization_map = _parse_manifest_map(
        args.materialization_manifest, argument_name="materialization manifest"
    )
    crop_map = _parse_manifest_map(
        args.dynamic_crop_manifest, argument_name="dynamic crop manifest"
    )
    feature_provenance, snapshots, references = _load_bbr_feature_dataset(
        feature_map
    )
    target_provenance, _, _ = _load_bbr_target_bindings(
        materialization_manifest_paths_by_song=materialization_map,
        dynamic_crop_manifest_paths_by_song=crop_map,
        snapshots_by_reference=snapshots,
        references_by_song=references,
        build_training_corpora=False,
    )
    dataset = {**feature_provenance, **target_provenance}
    budget = _live_budget()
    code, repo = _code_provenance(
        expected_git_commit=args.expected_git_commit,
        tracked_diff_policy=args.tracked_diff_policy,
    )
    runtime = _runtime_provenance(budget, repo)
    hardware = _hardware_provenance(budget)
    hashes = {
        "dataset": _sha256(dataset),
        "code": _sha256(code),
        "runtime": _sha256(runtime),
        "hardware": _sha256(hardware),
    }
    suite_id = secrets.token_hex(32)
    run_nonce = secrets.token_hex(32)
    run_nonce_sha256 = hashlib.sha256(run_nonce.encode("ascii")).hexdigest()
    command = [
        sys.executable,
        "-P",
        "-S",
        "-m",
        "crosscut.stage_edit_dynamic_crop_v3_capacity_smoke",
        "--worker",
        "--candidate-name",
        candidate.candidate_name,
        "--expected-candidate-id",
        candidate.candidate_id,
        "--suite-id",
        suite_id,
        "--expected-dataset-sha256",
        hashes["dataset"],
        "--expected-code-sha256",
        hashes["code"],
        "--expected-runtime-sha256",
        hashes["runtime"],
        "--expected-hardware-sha256",
        hashes["hardware"],
        "--expected-git-commit",
        args.expected_git_commit,
        "--tracked-diff-policy",
        args.tracked_diff_policy,
    ]
    for song in SELECTION_SONG_IDS:
        command.extend(
            ("--feature-manifest", f"{song}={feature_map[song].resolve()}")
        )
        command.extend(
            (
                "--materialization-manifest",
                f"{song}={materialization_map[song].resolve()}",
            )
        )
        command.extend(
            ("--dynamic-crop-manifest", f"{song}={crop_map[song].resolve()}")
        )
    environment = _build_isolated_worker_environment(
        repo=repo, runtime_provenance=runtime, run_nonce=run_nonce
    )
    environment[V3_CAPACITY_SMOKE_NONCE_ENV] = run_nonce
    supervisor_pid = os.getpid()
    started_ns = time.time_ns()
    process = subprocess.Popen(
        command,
        cwd=repo,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate()
    ended_ns = time.time_ns()
    events = _parse_events(stdout)
    event = events[-1] if len(events) == 1 else None
    execution = _normalize_execution(
        event=event,
        returncode=int(process.returncode),
        candidate=candidate,
        budget=budget,
        supervisor_pid=supervisor_pid,
        worker_pid=int(process.pid),
        suite_id=suite_id,
        run_nonce_sha256=run_nonce_sha256,
        expected_hashes=hashes,
    )
    log_path = _write_log(
        destination,
        command=command,
        returncode=int(process.returncode),
        stdout=stdout,
        stderr=stderr,
        supervisor_pid=supervisor_pid,
        worker_pid=int(process.pid),
        suite_id=suite_id,
        run_nonce_sha256=run_nonce_sha256,
        started_ns=started_ns,
        ended_ns=ended_ns,
    )
    payload = _content_addressed_payload(
        {
            "schema_version": V3_CAPACITY_SMOKE_SCHEMA_VERSION,
            "analysis_basis": V3_CAPACITY_SMOKE_ANALYSIS_BASIS,
            "artifact_family": V3_CAPACITY_SMOKE_ARTIFACT_FAMILY,
            "dynamic_crop_version": 3,
            "activation_checkpointing": dict(
                V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE
            ),
            "candidate": candidate.to_dict(),
            "dataset_provenance": dataset,
            "code_provenance": code,
            "runtime_provenance": runtime,
            "hardware_provenance": hardware,
            "device_memory_budget": budget.to_dict(),
            "execution": execution,
            "runner_log": _pointer(log_path, "v3 MLX capacity runner log"),
            "supervisor_attestation": {
                "process_contract": V3_CAPACITY_SMOKE_PROCESS_CONTRACT,
                "suite_id": suite_id,
                "supervisor_pid": supervisor_pid,
                "worker_pid": int(process.pid),
                "fresh_process": supervisor_pid != int(process.pid),
                "run_nonce_sha256": run_nonce_sha256,
                "worker_event_count": len(events),
                "worker_event_sha256": _sha256(events),
            },
            "capacity_runtime_evidence_only": True,
            "semantic_metrics_authorized": False,
            "model_selection_authorized": False,
        }
    )
    digest = payload["artifact_content_sha256"]
    evidence_path = _atomic_exclusive_write(
        destination / f"{V3_CAPACITY_SMOKE_ARTIFACT_PREFIX}{digest}.json",
        payload,
    )
    load_dynamic_crop_v3_capacity_smoke(evidence_path)
    print(
        json.dumps(
            {
                "evidence_path": str(evidence_path),
                "runner_log_path": str(log_path),
                "candidate_name": candidate.candidate_name,
                "parameter_count": candidate.parameter_count,
                "measured_peak_bytes": execution["measured_peak_bytes"],
                "allowed_training_bytes": execution["allowed_training_bytes"],
                "runtime_execution_passed": execution[
                    "runtime_execution_passed"
                ],
                "within_release_memory_budget": execution[
                    "within_release_memory_budget"
                ],
                "capacity_smoke_passed": execution["capacity_smoke_passed"],
                "semantic_metrics_authorized": False,
                "model_selection_authorized": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if execution["capacity_smoke_passed"] else 2


def _validate_candidate_payload(value: object) -> DynamicCropV3CapacityCandidate:
    if not isinstance(value, Mapping):
        raise DynamicCropV3CapacitySmokeError("candidate must be an object")
    payload = dict(value)
    if set(payload) != {"candidate_name", "candidate_id", "contract"}:
        raise DynamicCropV3CapacitySmokeError("candidate keys are not exact")
    candidate = _candidate_by_name(str(payload.get("candidate_name") or ""))
    if payload != candidate.to_dict():
        raise DynamicCropV3CapacitySmokeError(
            "candidate is mixed, stale, or not a pinned v3 contract"
        )
    contract = payload["contract"]
    if (
        not isinstance(contract, Mapping)
        or contract.get("dynamic_crop_version") != 3
        or contract.get("model_contract") != DYNAMIC_CROP_V3_MODEL_CONTRACT
        or contract.get("loss_contract") != DYNAMIC_CROP_V3_LOSS_CONTRACT
        or contract.get("model_class") != "StageEditTransformerDynamicCropV3"
        or contract.get("loss_class") != "DynamicCropV3LossContract"
        or not _exact_activation_checkpointing_evidence(
            contract.get("activation_checkpointing")
        )
        or contract.get("legacy_v2_checkpoint_compatible") is not False
        or contract.get("v2_model_or_loss_accepted") is not False
    ):
        raise DynamicCropV3CapacitySmokeError(
            "candidate does not carry an exact v3-only contract"
        )
    return candidate


def _validate_smoke_log_execution_binding(
    root: Mapping[str, Any],
    *,
    candidate: DynamicCropV3CapacityCandidate,
    code: Mapping[str, Any],
) -> None:
    """Replay the recorded child event and bind every log/receipt identity."""

    log_pointer = root.get("runner_log")
    if not isinstance(log_pointer, Mapping) or set(log_pointer) != {
        "path",
        "file_sha256",
    }:
        raise DynamicCropV3CapacitySmokeError("v3 runner log pointer is invalid")
    log_path = Path(str(log_pointer.get("path") or ""))
    if (
        not log_path.is_absolute()
        or not _is_sha256(log_pointer.get("file_sha256"))
        or _file_sha256(log_path) != log_pointer.get("file_sha256")
    ):
        raise DynamicCropV3CapacitySmokeError("v3 runner log pointer differs")
    try:
        log = _addressed_payload(
            log_path,
            prefix=V3_CAPACITY_SMOKE_LOG_PREFIX,
            label="Dynamic Crop v3 capacity runner log",
        )
    except (OSError, ValueError) as exc:
        raise DynamicCropV3CapacitySmokeError(str(exc)) from exc
    expected_log_keys = {
        "schema_version",
        "analysis_basis",
        "suite_id",
        "supervisor_pid",
        "worker_pid",
        "fresh_process",
        "run_nonce_sha256",
        "started_at_unix_ns",
        "ended_at_unix_ns",
        "returncode",
        "command",
        "stdout",
        "stderr",
        "artifact_content_sha256",
    }
    command = log.get("command")
    supervisor_pid = log.get("supervisor_pid")
    worker_pid = log.get("worker_pid")
    returncode = log.get("returncode")
    started_ns = log.get("started_at_unix_ns")
    ended_ns = log.get("ended_at_unix_ns")
    if (
        set(log) != expected_log_keys
        or log.get("schema_version") != 1
        or log.get("analysis_basis")
        != "dynamic_crop_v3_fresh_worker_process_log_v1"
        or not isinstance(supervisor_pid, int)
        or isinstance(supervisor_pid, bool)
        or supervisor_pid < 1
        or not isinstance(worker_pid, int)
        or isinstance(worker_pid, bool)
        or worker_pid < 1
        or worker_pid == supervisor_pid
        or log.get("fresh_process") is not True
        or not _is_sha256(log.get("suite_id"))
        or not _is_sha256(log.get("run_nonce_sha256"))
        or not isinstance(started_ns, int)
        or isinstance(started_ns, bool)
        or not isinstance(ended_ns, int)
        or isinstance(ended_ns, bool)
        or ended_ns < started_ns
        or not isinstance(returncode, int)
        or isinstance(returncode, bool)
        or not isinstance(command, list)
        or not all(isinstance(argument, str) for argument in command)
        or not isinstance(log.get("stdout"), str)
        or not isinstance(log.get("stderr"), str)
    ):
        raise DynamicCropV3CapacitySmokeError(
            "v3 runner log process receipt is invalid"
        )
    assert isinstance(command, list)
    if command[1:6] != [
        "-P",
        "-S",
        "-m",
        "crosscut.stage_edit_dynamic_crop_v3_capacity_smoke",
        "--worker",
    ]:
        raise DynamicCropV3CapacitySmokeError(
            "v3 runner log command does not invoke the isolated worker"
        )

    def command_value(option: str) -> str:
        indices = [index for index, value in enumerate(command) if value == option]
        if len(indices) != 1 or indices[0] + 1 >= len(command):
            raise DynamicCropV3CapacitySmokeError(
                f"v3 runner log command option differs: {option}"
            )
        return command[indices[0] + 1]

    provenance_values: dict[str, Mapping[str, Any]] = {}
    for name, key in (
        ("dataset", "dataset_provenance"),
        ("code", "code_provenance"),
        ("runtime", "runtime_provenance"),
        ("hardware", "hardware_provenance"),
    ):
        value = root.get(key)
        if not isinstance(value, Mapping):
            raise DynamicCropV3CapacitySmokeError(
                f"v3 capacity smoke {name} provenance is missing"
            )
        provenance_values[name] = value
    expected_hashes = {
        name: _sha256(value) for name, value in provenance_values.items()
    }
    if (
        command_value("--candidate-name") != candidate.candidate_name
        or command_value("--expected-candidate-id") != candidate.candidate_id
        or command_value("--suite-id") != log.get("suite_id")
        or command_value("--expected-dataset-sha256")
        != expected_hashes["dataset"]
        or command_value("--expected-code-sha256") != expected_hashes["code"]
        or command_value("--expected-runtime-sha256")
        != expected_hashes["runtime"]
        or command_value("--expected-hardware-sha256")
        != expected_hashes["hardware"]
        or command_value("--expected-git-commit")
        != code.get("expected_git_commit")
        or command_value("--tracked-diff-policy")
        != code.get("tracked_diff_policy")
        or command.count("--feature-manifest") != len(SELECTION_SONG_IDS)
        or command.count("--materialization-manifest") != len(SELECTION_SONG_IDS)
        or command.count("--dynamic-crop-manifest") != len(SELECTION_SONG_IDS)
    ):
        raise DynamicCropV3CapacitySmokeError(
            "v3 runner log command/provenance binding differs"
        )
    events = _parse_events(str(log["stdout"]))
    attestation = root.get("supervisor_attestation")
    expected_attestation_keys = {
        "process_contract",
        "suite_id",
        "supervisor_pid",
        "worker_pid",
        "fresh_process",
        "run_nonce_sha256",
        "worker_event_count",
        "worker_event_sha256",
    }
    if (
        not isinstance(attestation, Mapping)
        or set(attestation) != expected_attestation_keys
        or attestation.get("process_contract")
        != V3_CAPACITY_SMOKE_PROCESS_CONTRACT
        or attestation.get("suite_id") != log.get("suite_id")
        or attestation.get("supervisor_pid") != supervisor_pid
        or attestation.get("worker_pid") != worker_pid
        or attestation.get("fresh_process") is not True
        or attestation.get("run_nonce_sha256") != log.get("run_nonce_sha256")
        or attestation.get("worker_event_count") != 1
        or len(events) != 1
        or attestation.get("worker_event_sha256") != _sha256(events)
    ):
        raise DynamicCropV3CapacitySmokeError(
            "v3 fresh-process log/event attestation differs"
        )
    budget_payload = root.get("device_memory_budget")
    if not isinstance(budget_payload, Mapping):
        raise DynamicCropV3CapacitySmokeError("v3 device memory budget is missing")
    try:
        budget = DeviceMemoryBudget(**dict(budget_payload))
    except (TypeError, ValueError) as exc:
        raise DynamicCropV3CapacitySmokeError(
            f"v3 device memory budget is invalid: {exc}"
        ) from exc
    if budget.to_dict() != dict(budget_payload):
        raise DynamicCropV3CapacitySmokeError("v3 device memory budget differs")
    normalized = _normalize_execution(
        event=events[0],
        returncode=returncode,
        candidate=candidate,
        budget=budget,
        supervisor_pid=supervisor_pid,
        worker_pid=worker_pid,
        suite_id=str(log["suite_id"]),
        run_nonce_sha256=str(log["run_nonce_sha256"]),
        expected_hashes=expected_hashes,
    )
    try:
        normalized_bytes = _canonical_bytes(normalized)
        observed_bytes = _canonical_bytes(root.get("execution"))
    except (TypeError, ValueError) as exc:
        raise DynamicCropV3CapacitySmokeError(
            f"v3 execution receipt cannot be canonicalized: {exc}"
        ) from exc
    if observed_bytes != normalized_bytes:
        raise DynamicCropV3CapacitySmokeError(
            "v3 execution receipt differs from the bound worker log event"
        )


def load_dynamic_crop_v3_capacity_smoke(path: Path) -> dict[str, Any]:
    """Load a content-addressed v3 smoke and reject mixed v2 declarations."""

    try:
        root = _addressed_payload(
            Path(path),
            prefix=V3_CAPACITY_SMOKE_ARTIFACT_PREFIX,
            label="Dynamic Crop v3 capacity smoke",
        )
    except (OSError, ValueError) as exc:
        raise DynamicCropV3CapacitySmokeError(str(exc)) from exc
    required = {
        "schema_version",
        "analysis_basis",
        "artifact_family",
        "dynamic_crop_version",
        "activation_checkpointing",
        "candidate",
        "dataset_provenance",
        "code_provenance",
        "runtime_provenance",
        "hardware_provenance",
        "device_memory_budget",
        "execution",
        "runner_log",
        "supervisor_attestation",
        "capacity_runtime_evidence_only",
        "semantic_metrics_authorized",
        "model_selection_authorized",
        "artifact_content_sha256",
    }
    if set(root) != required:
        raise DynamicCropV3CapacitySmokeError(
            "v3 capacity smoke keys are not exact"
        )
    if (
        root.get("schema_version") != V3_CAPACITY_SMOKE_SCHEMA_VERSION
        or root.get("analysis_basis") != V3_CAPACITY_SMOKE_ANALYSIS_BASIS
        or root.get("artifact_family") != V3_CAPACITY_SMOKE_ARTIFACT_FAMILY
        or root.get("dynamic_crop_version") != 3
        or not _exact_activation_checkpointing_evidence(
            root.get("activation_checkpointing")
        )
        or root.get("capacity_runtime_evidence_only") is not True
        or root.get("semantic_metrics_authorized") is not False
        or root.get("model_selection_authorized") is not False
    ):
        raise DynamicCropV3CapacitySmokeError(
            "v3 capacity smoke schema or authority differs"
        )
    candidate = _validate_candidate_payload(root.get("candidate"))
    code = _validate_capacity_smoke_code_provenance(
        root.get("code_provenance")
    )
    execution = root.get("execution")
    if not isinstance(execution, Mapping):
        raise DynamicCropV3CapacitySmokeError("v3 execution must be an object")
    tensor = execution.get("tensor_provenance")
    if execution.get("runtime_execution_passed") is True:
        if (
            execution.get("model_class") != "StageEditTransformerDynamicCropV3"
            or execution.get("loss_class") != "DynamicCropV3LossContract"
            or execution.get("parameter_count") != candidate.parameter_count
            or not _exact_activation_checkpointing_evidence(
                execution.get("activation_checkpointing")
            )
            or not isinstance(tensor, Mapping)
            or tensor.get("source_tensor_shape") != [2, 2048, 32, 720]
            or tensor.get("dynamic_crop_version") != 3
            or tensor.get("legacy_v2_model_or_loss_used") is not False
        ):
            raise DynamicCropV3CapacitySmokeError(
                "v3 execution contains a mixed or incomplete runtime receipt"
            )
    if (
        not _exact_activation_checkpointing_evidence(
            execution.get("activation_checkpointing")
        )
        or execution.get("semantic_metrics_authorized") is not False
        or execution.get("model_selection_authorized") is not False
    ):
        raise DynamicCropV3CapacitySmokeError(
            "capacity smoke cannot authorize semantic selection"
        )
    _validate_smoke_log_execution_binding(root, candidate=candidate, code=code)
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one pinned Dynamic Crop v3 capacity candidate in a fresh MLX "
            "process at [2,2048,32,720]."
        )
    )
    parser.add_argument(
        "--feature-manifest",
        action="append",
        required=True,
        metavar="SONG_ID=/ABSOLUTE/PATH.json",
    )
    parser.add_argument(
        "--materialization-manifest",
        action="append",
        required=True,
        metavar="SONG_ID=/ABSOLUTE/PATH.json",
    )
    parser.add_argument(
        "--dynamic-crop-manifest",
        action="append",
        required=True,
        metavar="SONG_ID=/ABSOLUTE/PATH.json",
    )
    parser.add_argument(
        "--candidate-name", choices=_PINNED_CANDIDATE_NAMES, required=True
    )
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--expected-git-commit")
    parser.add_argument(
        "--tracked-diff-policy",
        choices=("require_clean", "record_development"),
        default="require_clean",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-candidate-id", help=argparse.SUPPRESS)
    parser.add_argument("--suite-id", help=argparse.SUPPRESS)
    parser.add_argument("--expected-dataset-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--expected-code-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--expected-runtime-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--expected-hardware-sha256", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.worker:
            hidden = (
                args.expected_candidate_id,
                args.suite_id,
                args.expected_dataset_sha256,
                args.expected_code_sha256,
                args.expected_runtime_sha256,
                args.expected_hardware_sha256,
                args.expected_git_commit,
            )
            if any(not isinstance(value, str) or not value for value in hidden):
                raise DynamicCropV3CapacitySmokeError(
                    "worker requires the complete supervisor handshake"
                )
            return _worker_main(args)
        if args.destination is None or not args.expected_git_commit:
            raise DynamicCropV3CapacitySmokeError(
                "supervisor requires destination and expected git commit"
            )
        return _supervisor_main(args)
    except DynamicCropV3CapacitySmokeError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DynamicCropV3CapacityCandidate",
    "DynamicCropV3CapacitySmokeError",
    "V3_CAPACITY_SMOKE_ANALYSIS_BASIS",
    "V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE",
    "V3_CAPACITY_SMOKE_ARTIFACT_PREFIX",
    "V3_CAPACITY_SMOKE_SCHEMA_VERSION",
    "default_dynamic_crop_v3_capacity_candidates",
    "load_dynamic_crop_v3_capacity_smoke",
    "main",
]
