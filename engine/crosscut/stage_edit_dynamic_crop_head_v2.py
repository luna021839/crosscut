"""Versioned StageEdit Dynamic Crop Head v2.

This module is deliberately additive.  The original StageEdit checkpoint,
``StageEditTarget.crop`` four-vector, model output, and loss remain unchanged.
The v2 model owns a separate parameter namespace and an explicit K=8 padded
trajectory contract:

* renderer values ``[B, T, K, 4]``;
* normalized monotonic times ``[B, T, K]``;
* keyframe presence targets/masks ``[B, T, K]``;
* per-keyframe value/time masks; and
* a separately masked static/dynamic label.

Reverse-match diagnostics are not inputs.  Only validated target keyframes
from :mod:`stage_edit_dynamic_crop_adapter` enter tensorization.  Nothing in
this module activates production inference; callers must explicitly opt into
the v2 class and its distinct checkpoint fingerprint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .render import EditorialTransformKeyframe, EditorialTransformTrack
from .stage_edit_dynamic_crop_adapter import StageEditDynamicCropTarget
from .stage_edit_transformer import (
    MASKED_SOURCE_LOGIT,
    StageEditLossContract,
    StageEditModelOutput,
    StageEditSupervisionBatch,
    StageEditTensorBatch,
    StageEditTokenSchema,
    StageEditTransformer,
    StageEditTransformerConfig,
    stage_edit_loss,
    stage_edit_parameter_count,
)

try:  # Optional outside Apple-silicon production hosts.
    import mlx.core as mx
    import mlx.nn as nn
except ImportError as exc:  # pragma: no cover - non-Apple portability.
    mx = None
    nn = None
    _MLX_IMPORT_ERROR: ImportError | None = exc
else:
    _MLX_IMPORT_ERROR = None


DYNAMIC_CROP_V2_SCHEMA_VERSION = 2
DYNAMIC_CROP_V2_MAX_KEYFRAMES = 8
DYNAMIC_CROP_V2_TARGET_CONTRACT = (
    "stage_edit_dynamic_crop_k8_padded_trajectory_target_v2"
)
DYNAMIC_CROP_V2_MODEL_CONTRACT = (
    "stage_edit_causal_transformer_plus_dynamic_crop_k8_head_v2"
)
DYNAMIC_CROP_V2_OUTPUT_CONTRACT = (
    "values_times_presence_motion_output_v2"
)
DYNAMIC_CROP_V2_TIME_PARAMETERIZATION = (
    "observed_zero_start_plus_seven_positive_deltas_v2"
)
DYNAMIC_CROP_V2_LOSS_CONTRACT = (
    "masked_values_times_presence_motion_loss_v2"
)
DYNAMIC_CROP_V2_CHECKPOINT_FORMAT_VERSION = 3
DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME = "dynamic-crop-v2-metadata.json"
DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME = "dynamic-crop-v2-model.safetensors"


def _require_mlx() -> None:
    if mx is None or nn is None:
        detail = f": {_MLX_IMPORT_ERROR}" if _MLX_IMPORT_ERROR else ""
        raise RuntimeError(f"Dynamic Crop Head v2 requires MLX{detail}")


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class DynamicCropHeadV2Config:
    """Separate model/checkpoint identity for the K=8 trajectory head."""

    schema_version: int = DYNAMIC_CROP_V2_SCHEMA_VERSION
    max_keyframes: int = DYNAMIC_CROP_V2_MAX_KEYFRAMES
    presence_threshold: float = 0.5
    target_contract: str = DYNAMIC_CROP_V2_TARGET_CONTRACT
    output_contract: str = DYNAMIC_CROP_V2_OUTPUT_CONTRACT

    def __post_init__(self) -> None:
        if self.schema_version != DYNAMIC_CROP_V2_SCHEMA_VERSION:
            raise ValueError("unsupported Dynamic Crop Head schema version")
        if self.max_keyframes != DYNAMIC_CROP_V2_MAX_KEYFRAMES:
            raise ValueError("Dynamic Crop Head v2 requires exactly K=8")
        if not 0.0 < _finite(
            self.presence_threshold, "presence threshold"
        ) < 1.0:
            raise ValueError("presence threshold must lie in (0, 1)")
        if self.target_contract != DYNAMIC_CROP_V2_TARGET_CONTRACT:
            raise ValueError("Dynamic Crop Head target contract is invalid")
        if self.output_contract != DYNAMIC_CROP_V2_OUTPUT_CONTRACT:
            raise ValueError("Dynamic Crop Head output contract is invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class DynamicCropTrajectoryTargetV2:
    """One renderer-safe shot trajectory, before K=8 padding."""

    shot_index: int
    source_id: str
    shot_duration_seconds: float
    keyframes: tuple[EditorialTransformKeyframe, ...]
    motion_label: str
    motion_label_loss_masked: bool
    artifact_content_sha256: str
    build_fingerprint_sha256: str
    target_contract: str = DYNAMIC_CROP_V2_TARGET_CONTRACT

    def __post_init__(self) -> None:
        if isinstance(self.shot_index, bool) or not isinstance(self.shot_index, int):
            raise ValueError("dynamic crop shot index must be an integer")
        if self.shot_index < 0:
            raise ValueError("dynamic crop shot index cannot be negative")
        source_id = str(self.source_id).strip()
        if not source_id:
            raise ValueError("dynamic crop source id cannot be blank")
        duration = _finite(self.shot_duration_seconds, "shot duration")
        if duration <= 0.0:
            raise ValueError("dynamic crop shot duration must be positive")
        keyframes = tuple(self.keyframes)
        if not 1 <= len(keyframes) <= DYNAMIC_CROP_V2_MAX_KEYFRAMES:
            raise ValueError("dynamic crop trajectory requires 1..8 keyframes")
        if any(not isinstance(item, EditorialTransformKeyframe) for item in keyframes):
            raise TypeError("dynamic crop keyframes must use renderer coordinates")
        previous = -math.inf
        for item in keyframes:
            if item.time_seconds <= previous or item.time_seconds > duration:
                raise ValueError("dynamic crop keyframe time is outside its shot")
            previous = item.time_seconds
        for left, right in zip(keyframes, keyframes[1:]):
            # Rebase only for validation; serialized/learned times remain exact.
            EditorialTransformTrack(
                (
                    EditorialTransformKeyframe(
                        0.0,
                        left.zoom,
                        left.center_x,
                        left.center_y,
                        left.roll_degrees,
                    ),
                    EditorialTransformKeyframe(
                        right.time_seconds - left.time_seconds,
                        right.zoom,
                        right.center_x,
                        right.center_y,
                        right.roll_degrees,
                    ),
                ),
                reason="dynamic_crop_v2_target_interpolation_validation",
            )
        if self.motion_label not in {"static", "dynamic"}:
            raise ValueError("dynamic crop motion label is invalid")
        if not isinstance(self.motion_label_loss_masked, bool):
            raise TypeError("motion label mask must be a boolean")
        for value, name in (
            (self.artifact_content_sha256, "artifact content hash"),
            (self.build_fingerprint_sha256, "build fingerprint"),
        ):
            if not isinstance(value, str) or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if self.target_contract != DYNAMIC_CROP_V2_TARGET_CONTRACT:
            raise ValueError("dynamic crop trajectory target contract is invalid")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "shot_duration_seconds", duration)
        object.__setattr__(self, "keyframes", keyframes)

    @classmethod
    def from_adapter_target(
        cls,
        value: StageEditDynamicCropTarget,
    ) -> "DynamicCropTrajectoryTargetV2":
        if not isinstance(value, StageEditDynamicCropTarget):
            raise TypeError("value must be StageEditDynamicCropTarget")
        return cls(
            shot_index=value.shot_index,
            source_id=value.source_id,
            shot_duration_seconds=value.shot_duration_seconds,
            keyframes=tuple(
                EditorialTransformKeyframe(
                    item.time_within_shot_seconds,
                    item.zoom,
                    item.center_x,
                    item.center_y,
                    item.roll_degrees,
                )
                for item in value.keyframes
            ),
            motion_label=value.motion_label,
            motion_label_loss_masked=value.motion_label_loss_masked,
            artifact_content_sha256=value.artifact_content_sha256,
            build_fingerprint_sha256=value.build_fingerprint_sha256,
        )


@dataclass(frozen=True, slots=True)
class StageEditDynamicCropTargetSequenceV2:
    """Trajectory targets aligned one-for-one with StageEdit tokens."""

    sequence_id: str
    targets: tuple[DynamicCropTrajectoryTargetV2 | None, ...]
    target_contract: str = DYNAMIC_CROP_V2_TARGET_CONTRACT

    def __post_init__(self) -> None:
        sequence_id = str(self.sequence_id).strip()
        if not sequence_id:
            raise ValueError("dynamic crop target sequence id cannot be blank")
        targets = tuple(self.targets)
        if not targets:
            raise ValueError("dynamic crop target sequence cannot be empty")
        if any(
            item is not None and not isinstance(item, DynamicCropTrajectoryTargetV2)
            for item in targets
        ):
            raise TypeError("dynamic crop target sequence contains an invalid target")
        if self.target_contract != DYNAMIC_CROP_V2_TARGET_CONTRACT:
            raise ValueError("dynamic crop target sequence contract is invalid")
        object.__setattr__(self, "sequence_id", sequence_id)
        object.__setattr__(self, "targets", targets)


@dataclass(frozen=True, slots=True)
class StageEditDynamicCropSupervisionBatchV2:
    values: Any
    normalized_times: Any
    keyframe_mask: Any
    presence_targets: Any
    presence_mask: Any
    motion_targets: Any
    motion_mask: Any
    shot_duration_seconds: Any
    track_mask: Any
    keyframe_count_targets: Any
    sequence_ids: tuple[str, ...]
    target_contract: str = DYNAMIC_CROP_V2_TARGET_CONTRACT


def tensorize_stage_edit_dynamic_crop_supervision_v2(
    target_sequences: Sequence[StageEditDynamicCropTargetSequenceV2],
    batch: StageEditTensorBatch,
    *,
    config: DynamicCropHeadV2Config = DynamicCropHeadV2Config(),
    device: Any = None,
) -> StageEditDynamicCropSupervisionBatchV2:
    """Pad validated variable trajectories to K=8 with explicit masks."""

    _require_mlx()
    by_id = {item.sequence_id: item for item in target_sequences}
    if len(by_id) != len(target_sequences):
        raise ValueError("dynamic crop target sequence ids must be unique")
    batch_size, max_length = batch.sequence_mask.shape
    keyframes = config.max_keyframes
    values = np.zeros((batch_size, max_length, keyframes, 4), dtype=np.float32)
    times = np.zeros((batch_size, max_length, keyframes), dtype=np.float32)
    keyframe_mask = np.zeros((batch_size, max_length, keyframes), dtype=bool)
    presence_targets = np.zeros_like(keyframe_mask, dtype=np.float32)
    presence_mask = np.zeros_like(keyframe_mask, dtype=bool)
    motion_targets = np.zeros((batch_size, max_length), dtype=np.float32)
    motion_mask = np.zeros((batch_size, max_length), dtype=bool)
    durations = np.zeros((batch_size, max_length), dtype=np.float32)
    track_mask = np.zeros((batch_size, max_length), dtype=bool)
    counts = np.zeros((batch_size, max_length), dtype=np.int32)

    for batch_index, sequence_id in enumerate(batch.sequence_ids):
        sequence = by_id.get(sequence_id)
        if sequence is None:
            continue
        valid_length = int(np.asarray(batch.sequence_mask[batch_index]).sum())
        if len(sequence.targets) != valid_length:
            raise ValueError(
                f"dynamic crop target count for {sequence_id} differs from tokens"
            )
        for time_index, target in enumerate(sequence.targets):
            if target is None:
                continue
            count = len(target.keyframes)
            if count > keyframes:
                # No truncation or simplification is allowed by this contract.
                raise ValueError("dynamic crop target exceeds K=8")
            duration = target.shot_duration_seconds
            track_mask[batch_index, time_index] = True
            presence_mask[batch_index, time_index, :] = True
            presence_targets[batch_index, time_index, :count] = 1.0
            keyframe_mask[batch_index, time_index, :count] = True
            durations[batch_index, time_index] = duration
            counts[batch_index, time_index] = count
            if not target.motion_label_loss_masked:
                motion_mask[batch_index, time_index] = True
                motion_targets[batch_index, time_index] = float(
                    target.motion_label == "dynamic"
                )
            for keyframe_index, item in enumerate(target.keyframes):
                normalized_time = item.time_seconds / duration
                if not 0.0 <= normalized_time <= 1.0:
                    raise ValueError("dynamic crop normalized time is outside [0, 1]")
                # Preserve the canonical time exactly.  crop-v2 never clamps
                # an out-of-domain target into the training manifold.
                times[batch_index, time_index, keyframe_index] = normalized_time
                values[batch_index, time_index, keyframe_index] = (
                    item.zoom,
                    item.center_x,
                    item.center_y,
                    item.roll_degrees,
                )

    stream = device if device is not None else mx.default_device()
    with mx.stream(stream):
        return StageEditDynamicCropSupervisionBatchV2(
            values=mx.array(values),
            normalized_times=mx.array(times),
            keyframe_mask=mx.array(keyframe_mask),
            presence_targets=mx.array(presence_targets),
            presence_mask=mx.array(presence_mask),
            motion_targets=mx.array(motion_targets),
            motion_mask=mx.array(motion_mask),
            shot_duration_seconds=mx.array(durations),
            track_mask=mx.array(track_mask),
            keyframe_count_targets=mx.array(counts),
            sequence_ids=tuple(batch.sequence_ids),
        )


@dataclass(frozen=True, slots=True)
class StageEditDynamicCropModelOutputV2:
    """All v1 outputs plus the separately versioned trajectory outputs."""

    cut_timing_logits: Any
    source_selection_logits: Any
    duration_bucket_logits: Any
    transition_logits: Any
    crop: Any
    effective_source_mask: Any
    dynamic_crop_values: Any
    dynamic_crop_normalized_times: Any
    dynamic_crop_presence_logits: Any
    dynamic_crop_motion_logits: Any
    output_contract: str = DYNAMIC_CROP_V2_OUTPUT_CONTRACT


_ModuleBase = nn.Module if nn is not None else object


class StageEditTransformerDynamicCropV2(_ModuleBase):
    """A distinct checkpoint namespace containing the v1 trunk plus K=8 head."""

    def __init__(
        self,
        base_config: StageEditTransformerConfig,
        dynamic_crop_config: DynamicCropHeadV2Config = DynamicCropHeadV2Config(),
    ) -> None:
        _require_mlx()
        super().__init__()
        self.base_config = base_config
        self.dynamic_crop_config = dynamic_crop_config
        # Nested parameter names deliberately prevent a v2 checkpoint from
        # being mistaken for a legacy flat StageEdit checkpoint.
        self.base = StageEditTransformer(base_config)
        dimension = base_config.model_dimension
        keyframes = dynamic_crop_config.max_keyframes
        self.dynamic_crop_values_head = nn.Linear(dimension, keyframes * 4)
        # Every renderer-eligible training target begins with *observed* source
        # geometry at shot time zero.  Encode that invariant in the model
        # manifold instead of asking inference to repair a nonzero first time.
        # The remaining K-1 samples retain learned strictly-positive deltas.
        self.dynamic_crop_time_head = nn.Linear(dimension, keyframes - 1)
        self.dynamic_crop_presence_head = nn.Linear(dimension, keyframes)
        self.dynamic_crop_motion_head = nn.Linear(dimension, 1)

    @property
    def checkpoint_contract(self) -> dict[str, Any]:
        return {
            "schema_version": DYNAMIC_CROP_V2_SCHEMA_VERSION,
            "model_contract": DYNAMIC_CROP_V2_MODEL_CONTRACT,
            "base_model_config": self.base_config.to_dict(),
            "dynamic_crop_config": self.dynamic_crop_config.to_dict(),
            "dynamic_crop_config_fingerprint": (
                self.dynamic_crop_config.fingerprint
            ),
            "time_parameterization": DYNAMIC_CROP_V2_TIME_PARAMETERIZATION,
            "legacy_checkpoint_parameter_layout_compatible": False,
            "production_activation_authorized": False,
        }

    def _hidden_and_sources(
        self, batch: StageEditTensorBatch
    ) -> tuple[Any, Any, Any, Any]:
        """Use the versioned relational source encoder in the nested trunk."""

        return self.base._hidden_and_sources(batch)

    def __call__(
        self, batch: StageEditTensorBatch
    ) -> StageEditDynamicCropModelOutputV2:
        (
            hidden,
            relational_source_features,
            source_embeddings,
            effective_mask,
        ) = self._hidden_and_sources(batch)
        source_logits = self.base._source_logits(
            hidden,
            relational_source_features,
            source_embeddings,
            effective_mask,
        )
        keyframes = self.dynamic_crop_config.max_keyframes
        crop_values = self.dynamic_crop_values_head(hidden).reshape(
            (*hidden.shape[:-1], keyframes, 4)
        )
        raw_deltas = self.dynamic_crop_time_head(hidden)
        first_time = mx.zeros((*raw_deltas.shape[:-1], 1), dtype=raw_deltas.dtype)
        positive_deltas = nn.softplus(raw_deltas)
        cumulative = mx.concatenate(
            (
                first_time,
                mx.cumsum(positive_deltas, axis=-1),
            ),
            axis=-1,
        )
        normalized_times = cumulative / mx.maximum(cumulative[..., -1:], 1.0)
        return StageEditDynamicCropModelOutputV2(
            cut_timing_logits=self.base._cut_logits(
                hidden, batch, effective_mask
            ),
            source_selection_logits=source_logits,
            duration_bucket_logits=self.base.duration_head(hidden),
            transition_logits=self.base.transition_head(hidden),
            crop=self.base.crop_head(hidden),
            effective_source_mask=effective_mask,
            dynamic_crop_values=crop_values,
            dynamic_crop_normalized_times=normalized_times,
            dynamic_crop_presence_logits=self.dynamic_crop_presence_head(hidden),
            dynamic_crop_motion_logits=self.dynamic_crop_motion_head(hidden).squeeze(
                -1
            ),
        )


def stage_edit_dynamic_crop_v2_parameter_count(
    base_config: StageEditTransformerConfig,
    dynamic_crop_config: DynamicCropHeadV2Config = DynamicCropHeadV2Config(),
) -> int:
    """Return the exact nested base-plus-trajectory parameter count."""

    dimension = base_config.model_dimension
    keyframes = dynamic_crop_config.max_keyframes
    # Four affine heads: K*4 values, K-1 time deltas, K presence, one motion.
    dynamic_output_dimension = keyframes * 4 + (keyframes - 1) + keyframes + 1
    return stage_edit_parameter_count(base_config) + (
        dimension + 1
    ) * dynamic_output_dimension


@dataclass(frozen=True, slots=True)
class StageEditDynamicCropCheckpointMetadataV2:
    """Metadata that cannot be confused with legacy ``metadata.json``."""

    format_version: int
    checkpoint_id: str
    created_at: str
    training_steps: int
    weights_trained: bool
    training_data_id: str | None
    token_schema: StageEditTokenSchema
    base_model_config: StageEditTransformerConfig
    dynamic_crop_config: DynamicCropHeadV2Config
    weight_filename: str = DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME
    weight_sha256: str | None = None
    model_contract: str = DYNAMIC_CROP_V2_MODEL_CONTRACT
    target_contract: str = DYNAMIC_CROP_V2_TARGET_CONTRACT
    output_contract: str = DYNAMIC_CROP_V2_OUTPUT_CONTRACT
    loss_contract: str = DYNAMIC_CROP_V2_LOSS_CONTRACT
    time_parameterization: str = DYNAMIC_CROP_V2_TIME_PARAMETERIZATION
    production_activation_authorized: bool = False

    def __post_init__(self) -> None:
        if self.format_version != DYNAMIC_CROP_V2_CHECKPOINT_FORMAT_VERSION:
            raise ValueError("unsupported Dynamic Crop Head v2 checkpoint format")
        checkpoint_id = str(self.checkpoint_id).strip()
        if not checkpoint_id:
            raise ValueError("dynamic crop checkpoint id cannot be blank")
        if isinstance(self.training_steps, bool) or self.training_steps < 0:
            raise ValueError("dynamic crop training_steps cannot be negative")
        if self.weights_trained and self.training_steps == 0:
            raise ValueError("trained dynamic crop weights need at least one step")
        if (
            self.base_model_config.token_schema_fingerprint
            != self.token_schema.fingerprint
        ):
            raise ValueError("dynamic crop checkpoint schema and base config differ")
        if Path(self.weight_filename).name != self.weight_filename:
            raise ValueError("dynamic crop weight filename must not contain a path")
        if self.weight_filename == "model.safetensors":
            raise ValueError(
                "dynamic crop checkpoint cannot use the v1 weight filename"
            )
        if self.model_contract != DYNAMIC_CROP_V2_MODEL_CONTRACT:
            raise ValueError("dynamic crop checkpoint model contract is invalid")
        if self.target_contract != DYNAMIC_CROP_V2_TARGET_CONTRACT:
            raise ValueError("dynamic crop checkpoint target contract is invalid")
        if self.output_contract != DYNAMIC_CROP_V2_OUTPUT_CONTRACT:
            raise ValueError("dynamic crop checkpoint output contract is invalid")
        if self.loss_contract != DYNAMIC_CROP_V2_LOSS_CONTRACT:
            raise ValueError("dynamic crop checkpoint loss contract is invalid")
        if self.time_parameterization != DYNAMIC_CROP_V2_TIME_PARAMETERIZATION:
            raise ValueError("dynamic crop checkpoint time parameterization is invalid")
        if self.production_activation_authorized is not False:
            raise ValueError("v2 checkpoint metadata cannot authorize production")
        object.__setattr__(self, "checkpoint_id", checkpoint_id)

    @property
    def model_config(self) -> StageEditTransformerConfig:
        """Compatibility name for consumers of the shared base heads."""

        return self.base_model_config

    @property
    def contract_fingerprint_sha256(self) -> str:
        return _sha256_json(
            {
                "format_version": self.format_version,
                "model_contract": self.model_contract,
                "target_contract": self.target_contract,
                "output_contract": self.output_contract,
                "loss_contract": self.loss_contract,
                "token_schema_fingerprint": self.token_schema.fingerprint,
                "base_model_config": self.base_model_config.to_dict(),
                "dynamic_crop_config": self.dynamic_crop_config.to_dict(),
                "time_parameterization": self.time_parameterization,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "checkpoint_id": self.checkpoint_id,
            "created_at": self.created_at,
            "training_steps": self.training_steps,
            "weights_trained": self.weights_trained,
            "training_data_id": self.training_data_id,
            "token_schema": self.token_schema.to_dict(),
            "base_model_config": self.base_model_config.to_dict(),
            "dynamic_crop_config": self.dynamic_crop_config.to_dict(),
            "weight_filename": self.weight_filename,
            "weight_sha256": self.weight_sha256,
            "model_contract": self.model_contract,
            "target_contract": self.target_contract,
            "output_contract": self.output_contract,
            "loss_contract": self.loss_contract,
            "time_parameterization": self.time_parameterization,
            "contract_fingerprint_sha256": self.contract_fingerprint_sha256,
            "production_activation_authorized": False,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "StageEditDynamicCropCheckpointMetadataV2":
        value = cls(
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
            base_model_config=StageEditTransformerConfig.from_dict(
                payload["base_model_config"]
            ),
            dynamic_crop_config=DynamicCropHeadV2Config(
                **payload["dynamic_crop_config"]
            ),
            weight_filename=str(
                payload.get(
                    "weight_filename", DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME
                )
            ),
            weight_sha256=(
                str(payload["weight_sha256"])
                if payload.get("weight_sha256") is not None
                else None
            ),
            model_contract=str(payload.get("model_contract", "")),
            target_contract=str(payload.get("target_contract", "")),
            output_contract=str(payload.get("output_contract", "")),
            loss_contract=str(payload.get("loss_contract", "")),
            time_parameterization=str(payload.get("time_parameterization", "")),
            production_activation_authorized=bool(
                payload.get("production_activation_authorized", False)
            ),
        )
        expected = payload.get("contract_fingerprint_sha256")
        if expected != value.contract_fingerprint_sha256:
            raise ValueError("dynamic crop checkpoint contract fingerprint mismatch")
        return value


def new_stage_edit_dynamic_crop_checkpoint_metadata_v2(
    checkpoint_id: str,
    schema: StageEditTokenSchema,
    base_config: StageEditTransformerConfig,
    *,
    training_steps: int,
    weights_trained: bool,
    training_data_id: str | None = None,
    dynamic_crop_config: DynamicCropHeadV2Config = DynamicCropHeadV2Config(),
) -> StageEditDynamicCropCheckpointMetadataV2:
    return StageEditDynamicCropCheckpointMetadataV2(
        format_version=DYNAMIC_CROP_V2_CHECKPOINT_FORMAT_VERSION,
        checkpoint_id=checkpoint_id,
        created_at=datetime.now(UTC).isoformat(),
        training_steps=training_steps,
        weights_trained=weights_trained,
        training_data_id=training_data_id,
        token_schema=schema,
        base_model_config=base_config,
        dynamic_crop_config=dynamic_crop_config,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_stage_edit_dynamic_crop_checkpoint_v2(
    directory: Path,
    model: StageEditTransformerDynamicCropV2,
    metadata: StageEditDynamicCropCheckpointMetadataV2,
) -> StageEditDynamicCropCheckpointMetadataV2:
    """Save under v2-only filenames; legacy checkpoint files are untouched."""

    _require_mlx()
    if model.base_config != metadata.base_model_config:
        raise ValueError("dynamic crop model base config differs from metadata")
    if model.dynamic_crop_config != metadata.dynamic_crop_config:
        raise ValueError("dynamic crop head config differs from metadata")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    weight_path = destination / metadata.weight_filename
    model.save_weights(str(weight_path))
    mx.eval(model.parameters())
    saved = replace(metadata, weight_sha256=_sha256_file(weight_path))
    metadata_path = destination / DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME
    metadata_path.write_text(
        json.dumps(saved.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return saved


def load_stage_edit_dynamic_crop_checkpoint_v2(
    directory: Path,
    *,
    expected_schema: StageEditTokenSchema | None = None,
    require_trained: bool = True,
) -> tuple[
    StageEditTransformerDynamicCropV2, StageEditDynamicCropCheckpointMetadataV2
]:
    _require_mlx()
    source = Path(directory)
    metadata_path = source / DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("dynamic crop checkpoint metadata root must be an object")
    metadata = StageEditDynamicCropCheckpointMetadataV2.from_dict(payload)
    if (
        expected_schema is not None
        and expected_schema.fingerprint != metadata.token_schema.fingerprint
    ):
        raise ValueError("dynamic crop checkpoint schema differs from expected schema")
    if require_trained and not (
        metadata.weights_trained and metadata.training_steps > 0
    ):
        raise ValueError("dynamic crop checkpoint does not declare trained weights")
    weight_path = source / metadata.weight_filename
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    if (
        metadata.weight_sha256 is None
        or _sha256_file(weight_path) != metadata.weight_sha256
    ):
        raise ValueError("dynamic crop checkpoint weight digest is missing or invalid")
    model = StageEditTransformerDynamicCropV2(
        metadata.base_model_config, metadata.dynamic_crop_config
    )
    model.load_weights(str(weight_path), strict=True)
    mx.eval(model.parameters())
    return model, metadata


@dataclass(frozen=True, slots=True)
class DynamicCropV2LossContract:
    base: StageEditLossContract = StageEditLossContract()
    base_loss_weight: float = 1.0
    value_weight: float = 1.0
    time_weight: float = 0.50
    presence_weight: float = 0.35
    motion_weight: float = 0.15
    contract: str = DYNAMIC_CROP_V2_LOSS_CONTRACT

    def __post_init__(self) -> None:
        if not isinstance(self.base, StageEditLossContract):
            raise TypeError("base loss contract must be StageEditLossContract")
        for name in (
            "base_loss_weight",
            "value_weight",
            "time_weight",
            "presence_weight",
            "motion_weight",
        ):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if self.contract != DYNAMIC_CROP_V2_LOSS_CONTRACT:
            raise ValueError("dynamic crop v2 loss contract is invalid")


class _OutputProxy:
    """Let the unchanged v1 loss consume the already-computed v2 output."""

    def __init__(self, output: StageEditDynamicCropModelOutputV2) -> None:
        self.output = output

    def __call__(self, _: StageEditTensorBatch) -> StageEditModelOutput:
        return self.output  # type: ignore[return-value]


def _masked_mean(values: Any, mask: Any) -> Any:
    weights = mask.astype(values.dtype)
    return (values * weights).sum() / mx.maximum(weights.sum(), 1.0)


def stage_edit_dynamic_crop_v2_loss(
    model: StageEditTransformerDynamicCropV2,
    batch: StageEditTensorBatch,
    base_supervision: StageEditSupervisionBatch,
    dynamic_supervision: StageEditDynamicCropSupervisionBatchV2,
    contract: DynamicCropV2LossContract = DynamicCropV2LossContract(),
) -> tuple[Any, Mapping[str, Any]]:
    """Joint v1 + trajectory loss with no static-head coercion."""

    _require_mlx()
    output = model(batch)
    base_total, base_components = stage_edit_loss(
        _OutputProxy(output),
        batch,
        base_supervision,
        contract.base,
    )
    coordinate_mask = mx.broadcast_to(
        dynamic_supervision.keyframe_mask[..., None],
        output.dynamic_crop_values.shape,
    )
    value_values = nn.losses.smooth_l1_loss(
        output.dynamic_crop_values,
        dynamic_supervision.values,
        reduction="none",
    )
    value_loss = _masked_mean(value_values, coordinate_mask)
    time_values = nn.losses.smooth_l1_loss(
        output.dynamic_crop_normalized_times,
        dynamic_supervision.normalized_times,
        reduction="none",
    )
    time_loss = _masked_mean(time_values, dynamic_supervision.keyframe_mask)
    presence_values = nn.losses.binary_cross_entropy(
        output.dynamic_crop_presence_logits,
        dynamic_supervision.presence_targets,
        with_logits=True,
        reduction="none",
    )
    presence_loss = _masked_mean(
        presence_values, dynamic_supervision.presence_mask
    )
    motion_values = nn.losses.binary_cross_entropy(
        output.dynamic_crop_motion_logits,
        dynamic_supervision.motion_targets,
        with_logits=True,
        reduction="none",
    )
    motion_loss = _masked_mean(motion_values, dynamic_supervision.motion_mask)
    dynamic_total = (
        contract.value_weight * value_loss
        + contract.time_weight * time_loss
        + contract.presence_weight * presence_loss
        + contract.motion_weight * motion_loss
    )
    total = contract.base_loss_weight * base_total + dynamic_total
    components: dict[str, Any] = {
        **{f"base_{name}": value for name, value in base_components.items()},
        "dynamic_crop_values": value_loss,
        "dynamic_crop_times": time_loss,
        "dynamic_crop_presence": presence_loss,
        "dynamic_crop_motion": motion_loss,
        "dynamic_crop_total": dynamic_total,
        "total": total,
    }
    return total, components


def _safe_mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size else None


def evaluate_stage_edit_dynamic_crop_v2(
    model: StageEditTransformerDynamicCropV2,
    batch: StageEditTensorBatch,
    base_supervision: StageEditSupervisionBatch,
    dynamic_supervision: StageEditDynamicCropSupervisionBatchV2,
    *,
    contract: DynamicCropV2LossContract = DynamicCropV2LossContract(),
) -> dict[str, Any]:
    """Mask-aware trajectory metrics; old four-coordinate MAE is insufficient."""

    _require_mlx()
    model.eval()
    output = model(batch)
    total, components = stage_edit_dynamic_crop_v2_loss(
        model,
        batch,
        base_supervision,
        dynamic_supervision,
        contract,
    )
    mx.eval(
        total,
        *components.values(),
        output.dynamic_crop_values,
        output.dynamic_crop_normalized_times,
        output.dynamic_crop_presence_logits,
        output.dynamic_crop_motion_logits,
    )
    values = np.asarray(output.dynamic_crop_values, dtype=np.float64)
    targets = np.asarray(dynamic_supervision.values, dtype=np.float64)
    keyframe_mask = np.asarray(dynamic_supervision.keyframe_mask, dtype=bool)
    coordinate_names = ("zoom", "center_x", "center_y", "roll_degrees")
    coordinates: dict[str, Any] = {}
    for index, name in enumerate(coordinate_names):
        residual = np.abs(
            values[..., index][keyframe_mask]
            - targets[..., index][keyframe_mask]
        )
        coordinates[name] = {
            "label_count": int(residual.size),
            "mean_absolute_error": _safe_mean(residual),
            "p95_absolute_error": (
                float(np.quantile(residual, 0.95)) if residual.size else None
            ),
        }
    predicted_times = np.asarray(
        output.dynamic_crop_normalized_times, dtype=np.float64
    )
    target_times = np.asarray(
        dynamic_supervision.normalized_times, dtype=np.float64
    )
    time_residual = np.abs(predicted_times[keyframe_mask] - target_times[keyframe_mask])
    presence_mask = np.asarray(dynamic_supervision.presence_mask, dtype=bool)
    presence_targets = np.asarray(
        dynamic_supervision.presence_targets, dtype=np.int32
    )
    presence_logits = np.asarray(
        output.dynamic_crop_presence_logits, dtype=np.float64
    )
    presence_predictions = (
        1.0 / (1.0 + np.exp(-np.clip(presence_logits, -60.0, 60.0)))
        >= model.dynamic_crop_config.presence_threshold
    ).astype(np.int32)
    selected_targets = presence_targets[presence_mask]
    selected_predictions = presence_predictions[presence_mask]
    true_positive = int(
        np.sum((selected_targets == 1) & (selected_predictions == 1))
    )
    false_positive = int(
        np.sum((selected_targets == 0) & (selected_predictions == 1))
    )
    false_negative = int(
        np.sum((selected_targets == 1) & (selected_predictions == 0))
    )
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    track_mask = np.asarray(dynamic_supervision.track_mask, dtype=bool)
    target_counts = np.asarray(dynamic_supervision.keyframe_count_targets)
    predicted_counts = presence_predictions.sum(axis=-1)
    count_exact = (
        float(np.mean(predicted_counts[track_mask] == target_counts[track_mask]))
        if np.any(track_mask)
        else None
    )
    motion_mask = np.asarray(dynamic_supervision.motion_mask, dtype=bool)
    motion_targets = np.asarray(dynamic_supervision.motion_targets, dtype=np.int32)
    motion_logits = np.asarray(output.dynamic_crop_motion_logits, dtype=np.float64)
    motion_predictions = (motion_logits >= 0.0).astype(np.int32)
    motion_accuracy = (
        float(np.mean(motion_predictions[motion_mask] == motion_targets[motion_mask]))
        if np.any(motion_mask)
        else None
    )
    # A small coordinate MAE is not sufficient: the exact renderer may still
    # reject a track because it lacks a zero-time keyframe, leaves the safe
    # pan/roll domain, or becomes unsafe between keyframes.  Exercise the same
    # fail-closed adapter used by production decoding for every labeled track.
    from .stage_edit_dynamic_crop_semimarkov_adapter import (  # local: avoids import cycle
        decode_dynamic_crop_prediction_v2,
    )

    durations = np.asarray(
        dynamic_supervision.shot_duration_seconds, dtype=np.float64
    )
    renderer_attempts = 0
    renderer_eligible = 0
    renderer_projection_count = 0
    renderer_rejections: dict[str, int] = {}
    for batch_index, token_index in np.argwhere(track_mask):
        renderer_attempts += 1
        decoded = decode_dynamic_crop_prediction_v2(
            normalized_times=predicted_times[batch_index, token_index].tolist(),
            values=values[batch_index, token_index].tolist(),
            presence_logits=presence_logits[batch_index, token_index].tolist(),
            shot_duration_seconds=float(durations[batch_index, token_index]),
            config=model.dynamic_crop_config,
        )
        if decoded.eligible:
            renderer_eligible += 1
            renderer_projection_count += decoded.minimum_safe_zoom_projection_count
        else:
            for reason in decoded.rejection_reasons:
                renderer_rejections[reason] = renderer_rejections.get(reason, 0) + 1
    return {
        "schema_version": DYNAMIC_CROP_V2_SCHEMA_VERSION,
        "target_contract": DYNAMIC_CROP_V2_TARGET_CONTRACT,
        "model_contract": DYNAMIC_CROP_V2_MODEL_CONTRACT,
        "loss_contract": DYNAMIC_CROP_V2_LOSS_CONTRACT,
        "total_loss": float(np.asarray(total).item()),
        "loss_components": {
            name: float(np.asarray(value).item())
            for name, value in components.items()
        },
        "trajectory": {
            "track_label_count": int(np.sum(track_mask)),
            "keyframe_label_count": int(np.sum(keyframe_mask)),
            "coordinates": coordinates,
            "normalized_time_mae": _safe_mean(time_residual),
            "presence": {
                "label_count": int(np.sum(presence_mask)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "keyframe_count_exact_accuracy": count_exact,
            },
            "motion": {
                "label_count": int(np.sum(motion_mask)),
                "accuracy": motion_accuracy,
            },
            "renderer_decode": {
                "attempt_count": renderer_attempts,
                "eligible_count": renderer_eligible,
                "eligible_rate": (
                    renderer_eligible / renderer_attempts
                    if renderer_attempts
                    else None
                ),
                "minimum_safe_zoom_projection_count": (
                    renderer_projection_count
                ),
                "rejection_reasons": dict(sorted(renderer_rejections.items())),
            },
        },
        "legacy_static_crop_head_used_for_trajectory": False,
        "production_activation_authorized": False,
    }


__all__ = [
    "DYNAMIC_CROP_V2_CHECKPOINT_FORMAT_VERSION",
    "DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME",
    "DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME",
    "DYNAMIC_CROP_V2_LOSS_CONTRACT",
    "DYNAMIC_CROP_V2_MAX_KEYFRAMES",
    "DYNAMIC_CROP_V2_MODEL_CONTRACT",
    "DYNAMIC_CROP_V2_OUTPUT_CONTRACT",
    "DYNAMIC_CROP_V2_SCHEMA_VERSION",
    "DYNAMIC_CROP_V2_TARGET_CONTRACT",
    "DYNAMIC_CROP_V2_TIME_PARAMETERIZATION",
    "DynamicCropHeadV2Config",
    "DynamicCropTrajectoryTargetV2",
    "DynamicCropV2LossContract",
    "StageEditDynamicCropModelOutputV2",
    "StageEditDynamicCropCheckpointMetadataV2",
    "StageEditDynamicCropSupervisionBatchV2",
    "StageEditDynamicCropTargetSequenceV2",
    "StageEditTransformerDynamicCropV2",
    "evaluate_stage_edit_dynamic_crop_v2",
    "load_stage_edit_dynamic_crop_checkpoint_v2",
    "new_stage_edit_dynamic_crop_checkpoint_metadata_v2",
    "save_stage_edit_dynamic_crop_checkpoint_v2",
    "stage_edit_dynamic_crop_v2_parameter_count",
    "stage_edit_dynamic_crop_v2_loss",
    "tensorize_stage_edit_dynamic_crop_supervision_v2",
]
