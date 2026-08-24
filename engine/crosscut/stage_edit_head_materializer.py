"""Production Transformer forward pass to semi-Markov head frames.

This is the strict, auditable bridge between trained Dynamic Crop checkpoints
and :mod:`stage_edit_semimarkov_beam`.  Source logits are remapped from each
token's padded model slots into one caller-supplied canonical source order.
V2 trajectories remain in normalized K=8 form; source-conditioned v3 keeps
that normalized trajectory for every canonical source until the beam chooses
both a concrete source and duration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .stage_edit_dynamic_crop_head_v2 import (
    DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME,
    StageEditDynamicCropCheckpointMetadataV2,
    StageEditDynamicCropModelOutputV2,
    load_stage_edit_dynamic_crop_checkpoint_v2,
)
from .stage_edit_dynamic_crop_semimarkov_adapter import (
    dynamic_crop_prediction_from_model_token_v2,
)
from .stage_edit_dynamic_crop_head_v3 import (
    StageEditDynamicCropCheckpointMetadataV3,
    StageEditDynamicCropModelOutputV3,
)
from .stage_edit_dynamic_crop_semimarkov_adapter_v3 import (
    dynamic_crop_predictions_from_model_token_v3,
)
from .stage_edit_semimarkov_beam import (
    StageEditHeadFrame,
    StageEditSourceContinuity,
)
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    MASKED_SOURCE_LOGIT,
    StageEditReadinessReport,
    StageEditSequence,
    StageEditTensorBatch,
    StageEditTokenSchema,
    mask_stage_edit_recent_history_for_semimarkov,
    stage_edit_recent_history_mask_contract,
    stage_edit_readiness_report,
    tensorize_stage_edit_batch,
)


STAGE_EDIT_HEAD_MATERIALIZER_SCHEMA_VERSION = 4
STAGE_EDIT_HEAD_MATERIALIZER_CONTRACT = (
    "production_dynamic_crop_v2_v3_transformer_to_semimarkov_heads_with_"
    "source_exact_crop_pair_continuity_and_exact_music_structure_v4"
)


class StageEditHeadMaterializerError(RuntimeError):
    """Base error for a requested production model forward."""


class StageEditHeadMaterializerNotReadyError(StageEditHeadMaterializerError):
    """The supplied checkpoint is not an unambiguous production v2 bundle."""


class StageEditHeadMaterializerInputError(StageEditHeadMaterializerError, ValueError):
    """The sequence or model output violates the materializer contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_tuple(values: Any, *, name: str, expected: int) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (expected,):
        raise StageEditHeadMaterializerInputError(
            f"{name} shape {array.shape} does not match ({expected},)"
        )
    result = tuple(float(value) for value in array.tolist())
    if any(not math.isfinite(value) for value in result):
        raise StageEditHeadMaterializerInputError(f"{name} contains non-finite values")
    return result


def _exact_progress_feature(
    *,
    values: Sequence[float],
    missing_mask: Sequence[bool],
    names: Sequence[str],
    feature_name: str,
    token_index: int,
    missing_mask_offset: int,
) -> float | None:
    """Read one exact serialized token coordinate without time synthesis."""

    try:
        index = tuple(names).index(feature_name)
    except ValueError as exc:
        raise StageEditHeadMaterializerInputError(
            f"schema lacks required {feature_name} feature"
        ) from exc
    if len(values) != len(names):
        raise StageEditHeadMaterializerInputError(
            f"token {token_index} {feature_name} feature width changed"
        )
    missing_index = missing_mask_offset + index
    if missing_mask and len(missing_mask) < missing_index + 1:
        raise StageEditHeadMaterializerInputError(
            f"token {token_index} {feature_name} missing-mask width changed"
        )
    if missing_mask and bool(missing_mask[missing_index]):
        return None
    value = float(values[index])
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise StageEditHeadMaterializerInputError(
            f"token {token_index} {feature_name} must lie in [0, 1]"
        )
    return value


def _exact_section_role(
    *,
    values: Sequence[float],
    missing_mask: Sequence[bool],
    schema: StageEditTokenSchema,
    token_index: int,
) -> str | None:
    role_features = (
        ("intro_probability", "intro"),
        ("verse_probability", "verse_like"),
        ("prechorus_probability", "build"),
        ("chorus_probability", "chorus_like"),
        ("bridge_probability", "bridge_like"),
        ("outro_probability", "outro"),
    )
    names = tuple(schema.section_feature_names)
    offset = len(schema.music_feature_names)
    positives: list[str] = []
    for feature_name, role in role_features:
        try:
            index = names.index(feature_name)
        except ValueError as exc:
            raise StageEditHeadMaterializerInputError(
                f"schema lacks required {feature_name} feature"
            ) from exc
        if missing_mask and len(missing_mask) <= offset + index:
            raise StageEditHeadMaterializerInputError(
                f"token {token_index} section-role missing-mask width changed"
            )
        if missing_mask and bool(missing_mask[offset + index]):
            return None
        value = float(values[index])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise StageEditHeadMaterializerInputError(
                f"token {token_index} {feature_name} must lie in [0, 1]"
            )
        if value > 0.0:
            positives.append(role)
    return positives[0] if len(positives) == 1 else None


_CONTINUITY_FEATURE_FIELDS = {
    "face_scale": "face_scale",
    "face_center_x": "face_center_x",
    "face_center_y": "face_center_y",
    "body_center_x": "body_center_x",
    "body_center_y": "body_center_y",
    "pose_motion": "pose_motion",
    "camera_motion_x": "camera_motion_x",
    "camera_motion_y": "camera_motion_y",
    "camera_zoom_velocity": "camera_zoom_velocity",
    "camera_roll_degrees_normalized": "camera_roll_normalized",
    "sharpness": "sharpness",
    "exposure": "exposure",
    "color_temperature_normalized": "color_temperature_normalized",
    "detected_person_count_normalized": "detected_person_count_normalized",
    "formation_person_count_normalized": "formation_person_count_normalized",
    "formation_pose_coverage": "formation_pose_coverage",
}

_SHOT_SCALE_FEATURE_NAME = "shot_scale"
_GROUP_WIDE_MAX_EXCLUSIVE = 0.22
_MEDIUM_MAX_EXCLUSIVE = 0.52


def _source_feature_values(
    candidate: Any,
    schema: StageEditTokenSchema,
) -> dict[str, float | None]:
    if len(candidate.features) != len(schema.source_feature_names):
        raise StageEditHeadMaterializerInputError(
            "source candidate feature count differs from checkpoint schema"
        )
    if candidate.missing_mask and len(candidate.missing_mask) != len(
        candidate.features
    ):
        raise StageEditHeadMaterializerInputError(
            "source candidate missing mask differs from feature count"
        )
    missing = (
        tuple(candidate.missing_mask)
        if candidate.missing_mask
        else (False,) * len(candidate.features)
    )
    return {
        name: None if missing[index] else float(candidate.features[index])
        for index, name in enumerate(schema.source_feature_names)
    }


def _shot_size_from_candidate(
    candidate: Any,
    schema: StageEditTokenSchema,
) -> str:
    """Recover the inference-safe shot family already fed to the model.

    Production callers do not have to duplicate a parallel shot-size map.  A
    missing ``shot_scale`` remains unknown; it is never silently treated as a
    wide or a close-up.
    """

    value = _source_feature_values(candidate, schema).get(
        _SHOT_SCALE_FEATURE_NAME
    )
    if value is None:
        return "unknown"
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise StageEditHeadMaterializerInputError(
            "source candidate shot_scale must lie in [0, 1]"
        )
    if value < _GROUP_WIDE_MAX_EXCLUSIVE:
        return "wide"
    if value < _MEDIUM_MAX_EXCLUSIVE:
        return "medium"
    return "closeup"


def _source_continuity_from_candidate(
    candidate: Any,
    schema: StageEditTokenSchema,
) -> StageEditSourceContinuity:
    by_name = _source_feature_values(candidate, schema)
    return StageEditSourceContinuity(
        **{
            destination: by_name.get(source)
            for source, destination in _CONTINUITY_FEATURE_FIELDS.items()
        }
    )


@dataclass(frozen=True, slots=True)
class StageEditHeadMaterializationResult:
    frames: tuple[StageEditHeadFrame, ...]
    sequence_id: str
    canonical_source_ids: tuple[str, ...]
    checkpoint_directory: str
    checkpoint_id: str
    training_steps: int
    metadata_sha256: str
    weights_sha256: str
    supplied_sequence_sha256: str
    input_sequence_sha256: str
    output_frames_sha256: str
    token_schema_fingerprint: str
    dynamic_crop_config_fingerprint: str
    readiness: StageEditReadinessReport
    decision_history_input_contract: Mapping[str, Any]
    contract: str = STAGE_EDIT_HEAD_MATERIALIZER_CONTRACT
    schema_version: int = STAGE_EDIT_HEAD_MATERIALIZER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("materialized StageEdit heads cannot be empty")
        if len(self.frames) != len(
            {round(item.master_time_seconds, 9) for item in self.frames}
        ):
            raise ValueError("materialized StageEdit head times must be unique")
        if not self.canonical_source_ids or len(self.canonical_source_ids) != len(
            set(self.canonical_source_ids)
        ):
            raise ValueError("canonical source ids must be non-empty and unique")
        for digest in (
            self.metadata_sha256,
            self.weights_sha256,
            self.supplied_sequence_sha256,
            self.input_sequence_sha256,
            self.output_frames_sha256,
            self.token_schema_fingerprint,
            self.dynamic_crop_config_fingerprint,
        ):
            if len(digest) != 64:
                raise ValueError("StageEdit materialization identity must be SHA-256")
        if not (
            self.readiness.architecture_available
            and self.readiness.learned_inference_ready
            and self.readiness.status == "production_checkpoint_ready"
        ):
            raise ValueError("StageEdit materialization requires production readiness")
        if (
            not isinstance(self.decision_history_input_contract, Mapping)
            or len(str(self.decision_history_input_contract.get("contract_sha256", "")))
            != 64
        ):
            raise ValueError("StageEdit decision-history contract is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "sequence_id": self.sequence_id,
            "canonical_source_ids": list(self.canonical_source_ids),
            "frame_count": len(self.frames),
            "checkpoint_directory": self.checkpoint_directory,
            "checkpoint_id": self.checkpoint_id,
            "training_steps": self.training_steps,
            "metadata_sha256": self.metadata_sha256,
            "weights_sha256": self.weights_sha256,
            "supplied_sequence_sha256": self.supplied_sequence_sha256,
            "input_sequence_sha256": self.input_sequence_sha256,
            "output_frames_sha256": self.output_frames_sha256,
            "token_schema_fingerprint": self.token_schema_fingerprint,
            "dynamic_crop_config_fingerprint": (
                self.dynamic_crop_config_fingerprint
            ),
            "decision_history_input_contract": dict(
                self.decision_history_input_contract
            ),
            "readiness": self.readiness.to_dict(),
            "frames": [item.to_dict() for item in self.frames],
        }


def _sequence_payload(sequence: StageEditSequence) -> dict[str, Any]:
    return {
        "sequence_id": sequence.sequence_id,
        "tokens": [
            {
                "master_time_seconds": token.master_time_seconds,
                "music_features": list(token.music_features),
                "section_features": list(token.section_features),
                "source_candidates": [
                    {
                        "source_id": candidate.source_id,
                        "features": list(candidate.features),
                        "missing_mask": list(candidate.missing_mask),
                        "available": candidate.available,
                        "hard_gate_passed": candidate.hard_gate_passed,
                    }
                    for candidate in token.source_candidates
                ],
                "recent_edit_features": list(token.recent_edit_features),
                "style_features": list(token.style_features),
                "missing_mask": list(token.missing_mask),
            }
            for token in sequence.tokens
        ],
    }


def _strict_token_source_mapping(
    *,
    token: Any,
    batch: StageEditTensorBatch,
    batch_index: int,
    token_index: int,
    canonical_source_ids: tuple[str, ...],
    candidate_mask: np.ndarray,
    hard_gate_mask: np.ndarray,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Prove supplied-token, padded-slot, and canonical identities agree.

    Dict comprehensions alone silently collapse duplicate ids.  That is unsafe
    for source-conditioned crop because it could attach one camera's trajectory
    to another camera.  This helper validates identities before constructing
    either lookup and also proves the tensorized masks still describe the
    supplied token.
    """

    candidate_ids = tuple(
        str(candidate.source_id).strip() for candidate in token.source_candidates
    )
    if any(not value for value in candidate_ids) or len(candidate_ids) != len(
        set(candidate_ids)
    ):
        raise StageEditHeadMaterializerInputError(
            f"token {token_index} candidate source ids must be non-empty and unique"
        )
    candidate_by_id = dict(zip(candidate_ids, token.source_candidates, strict=True))

    try:
        padded_ids = tuple(batch.source_ids[batch_index][token_index])
    except IndexError as exc:
        raise StageEditHeadMaterializerInputError(
            f"token {token_index} padded source ids are missing"
        ) from exc
    observed_padded_ids = tuple(
        str(source_id).strip() for source_id in padded_ids if source_id is not None
    )
    if any(not value for value in observed_padded_ids) or len(
        observed_padded_ids
    ) != len(set(observed_padded_ids)):
        raise StageEditHeadMaterializerInputError(
            f"token {token_index} padded source ids must be non-empty and unique"
        )
    if set(observed_padded_ids) != set(candidate_ids):
        raise StageEditHeadMaterializerInputError(
            f"token {token_index} padded source ids differ from supplied candidates"
        )
    unknown = set(observed_padded_ids) - set(canonical_source_ids)
    if unknown:
        raise StageEditHeadMaterializerInputError(
            "token contains source ids outside canonical order: "
            + ", ".join(sorted(unknown))
        )
    slot_by_id = {
        str(source_id).strip(): slot
        for slot, source_id in enumerate(padded_ids)
        if source_id is not None
    }
    for source_id, slot in slot_by_id.items():
        candidate = candidate_by_id[source_id]
        expected_available = bool(candidate.available)
        expected_hard_gate = bool(
            candidate.available and candidate.hard_gate_passed
        )
        if (
            bool(candidate_mask[batch_index, token_index, slot])
            != expected_available
            or bool(hard_gate_mask[batch_index, token_index, slot])
            != expected_hard_gate
        ):
            raise StageEditHeadMaterializerInputError(
                f"token {token_index} source {source_id} masks differ from supplied token"
            )
    return candidate_by_id, slot_by_id


def stage_edit_head_frames_from_dynamic_crop_output_v2(
    *,
    sequence: StageEditSequence,
    batch: StageEditTensorBatch,
    output: StageEditDynamicCropModelOutputV2,
    canonical_source_ids: Sequence[str],
    schema: StageEditTokenSchema,
    metadata: StageEditDynamicCropCheckpointMetadataV2,
    source_shot_sizes_by_token: Sequence[Mapping[str, str]] = (),
    batch_index: int = 0,
) -> tuple[StageEditHeadFrame, ...]:
    """Pure output conversion with exact padded-slot to source-id remapping."""

    source_ids = tuple(str(value).strip() for value in canonical_source_ids)
    if not source_ids or any(not value for value in source_ids):
        raise StageEditHeadMaterializerInputError(
            "canonical source ids must be non-empty"
        )
    if len(source_ids) != len(set(source_ids)):
        raise StageEditHeadMaterializerInputError(
            "canonical source ids must be unique"
        )
    if len(source_ids) > metadata.base_model_config.max_sources:
        raise StageEditHeadMaterializerInputError(
            "canonical source count exceeds checkpoint maximum"
        )
    if batch_index < 0 or batch_index >= len(batch.sequence_ids):
        raise StageEditHeadMaterializerInputError("batch index is outside model batch")
    if batch.sequence_ids[batch_index] != sequence.sequence_id:
        raise StageEditHeadMaterializerInputError(
            "batch sequence identity differs from requested sequence"
        )
    if source_shot_sizes_by_token and len(source_shot_sizes_by_token) != len(
        sequence.tokens
    ):
        raise StageEditHeadMaterializerInputError(
            "source shot-size mappings must align with every token"
        )

    cut_logits = np.asarray(output.cut_timing_logits)
    source_logits = np.asarray(output.source_selection_logits)
    duration_logits = np.asarray(output.duration_bucket_logits)
    transition_logits = np.asarray(output.transition_logits)
    legacy_crop = np.asarray(output.crop)
    output_effective_mask = np.asarray(output.effective_source_mask, dtype=bool)
    candidate_mask = np.asarray(batch.source_candidate_mask, dtype=bool)
    hard_gate_mask = np.asarray(batch.hard_source_gate_mask, dtype=bool)
    expected_effective_mask = np.logical_and(candidate_mask, hard_gate_mask)
    if output_effective_mask.shape != expected_effective_mask.shape or not np.array_equal(
        output_effective_mask, expected_effective_mask
    ):
        raise StageEditHeadMaterializerInputError(
            "model effective-source mask differs from tensorized hard gates"
        )

    token_count = len(sequence.tokens)
    expected_prefix = (len(batch.sequence_ids), batch.sequence_mask.shape[1])
    for name, array, tail in (
        ("cut logits", cut_logits, ()),
        ("source logits", source_logits, (metadata.base_model_config.max_sources,)),
        (
            "duration logits",
            duration_logits,
            (schema.duration_bucket_count,),
        ),
        ("transition logits", transition_logits, (len(schema.transition_labels),)),
        ("legacy crop", legacy_crop, (4,)),
    ):
        if array.shape != (*expected_prefix, *tail):
            raise StageEditHeadMaterializerInputError(
                f"{name} shape {array.shape} violates checkpoint contract"
            )

    frames: list[StageEditHeadFrame] = []
    for token_index, token in enumerate(sequence.tokens):
        candidate_by_id, slot_by_id = _strict_token_source_mapping(
            token=token,
            batch=batch,
            batch_index=batch_index,
            token_index=token_index,
            canonical_source_ids=source_ids,
            candidate_mask=candidate_mask,
            hard_gate_mask=hard_gate_mask,
        )
        remapped_logits: list[float] = []
        available: list[bool] = []
        hard_gates: list[bool] = []
        continuity: list[StageEditSourceContinuity] = []
        for source_id in source_ids:
            slot = slot_by_id.get(source_id)
            if slot is None:
                remapped_logits.append(float(MASKED_SOURCE_LOGIT))
                available.append(False)
                hard_gates.append(False)
                continuity.append(StageEditSourceContinuity())
                continue
            candidate = candidate_by_id.get(source_id)
            if candidate is None:
                raise StageEditHeadMaterializerInputError(
                    "tensorized source slot is absent from the supplied token"
                )
            remapped_logits.append(float(source_logits[batch_index, token_index, slot]))
            available.append(bool(candidate_mask[batch_index, token_index, slot]))
            hard_gates.append(bool(hard_gate_mask[batch_index, token_index, slot]))
            continuity.append(_source_continuity_from_candidate(candidate, schema))
        if not any(left and right for left, right in zip(available, hard_gates, strict=True)):
            raise StageEditHeadMaterializerInputError(
                f"token {token_index} has no source surviving hard gates"
            )
        supplied_shot_size_map = (
            source_shot_sizes_by_token[token_index]
            if source_shot_sizes_by_token
            else {}
        )
        prediction = dynamic_crop_prediction_from_model_token_v2(
            output,
            batch_index=batch_index,
            token_index=token_index,
            config=metadata.dynamic_crop_config,
        )
        frames.append(
            StageEditHeadFrame(
                master_time_seconds=float(token.master_time_seconds),
                cut_logit=float(cut_logits[batch_index, token_index]),
                source_logits=tuple(remapped_logits),
                duration_logits=_finite_tuple(
                    duration_logits[batch_index, token_index],
                    name=f"duration logits token {token_index}",
                    expected=schema.duration_bucket_count,
                ),
                transition_logits=_finite_tuple(
                    transition_logits[batch_index, token_index],
                    name=f"transition logits token {token_index}",
                    expected=len(schema.transition_labels),
                ),
                crop=_finite_tuple(
                    legacy_crop[batch_index, token_index],
                    name=f"legacy crop token {token_index}",
                    expected=4,
                ),
                available_source_mask=tuple(available),
                hard_source_gate_mask=tuple(hard_gates),
                source_shot_sizes=tuple(
                    str(
                        supplied_shot_size_map[source_id]
                        if source_id in supplied_shot_size_map
                        else (
                            _shot_size_from_candidate(
                                candidate_by_id[source_id], schema
                            )
                            if source_id in candidate_by_id
                            else "unknown"
                        )
                    )
                    for source_id in source_ids
                ),
                source_continuity=tuple(continuity),
                candidate_id=(
                    f"{sequence.sequence_id}:token-{token_index:04d}:"
                    f"{float(token.master_time_seconds):.6f}"
                ),
                dynamic_crop_v2=prediction,
                section_progress=_exact_progress_feature(
                    values=token.section_features,
                    missing_mask=token.missing_mask,
                    names=schema.section_feature_names,
                    feature_name="section_progress",
                    token_index=token_index,
                    missing_mask_offset=len(schema.music_feature_names),
                ),
                phrase_progress=_exact_progress_feature(
                    values=token.music_features,
                    missing_mask=token.missing_mask,
                    names=schema.music_feature_names,
                    feature_name="phrase_progress",
                    token_index=token_index,
                    missing_mask_offset=0,
                ),
                section_role=_exact_section_role(
                    values=token.section_features,
                    missing_mask=token.missing_mask,
                    schema=schema,
                    token_index=token_index,
                ),
            )
        )
    if len(frames) != token_count:
        raise AssertionError("internal StageEdit head materialization count mismatch")
    return tuple(frames)


def stage_edit_head_frames_from_dynamic_crop_output_v3(
    *,
    sequence: StageEditSequence,
    batch: StageEditTensorBatch,
    output: StageEditDynamicCropModelOutputV3,
    canonical_source_ids: Sequence[str],
    schema: StageEditTokenSchema,
    metadata: StageEditDynamicCropCheckpointMetadataV3,
    source_shot_sizes_by_token: Sequence[Mapping[str, str]] = (),
    batch_index: int = 0,
) -> tuple[StageEditHeadFrame, ...]:
    """Convert source-conditioned v3 output without losing its source axis.

    Model slots are local to each token and may be permuted.  Returned v3
    prediction tuples instead use the caller's canonical source order, with
    each prediction's ``source_index`` rewritten to that same canonical slot.
    Unavailable or hard-gated canonical slots remain explicit ``None`` values.
    """

    if not isinstance(output, StageEditDynamicCropModelOutputV3):
        raise StageEditHeadMaterializerInputError(
            "Dynamic Crop v3 materialization requires a v3 model output"
        )
    if not isinstance(metadata, StageEditDynamicCropCheckpointMetadataV3):
        raise StageEditHeadMaterializerInputError(
            "Dynamic Crop v3 materialization requires v3 checkpoint metadata"
        )
    source_ids = tuple(str(value).strip() for value in canonical_source_ids)
    if not source_ids or any(not value for value in source_ids):
        raise StageEditHeadMaterializerInputError(
            "canonical source ids must be non-empty"
        )
    if len(source_ids) != len(set(source_ids)):
        raise StageEditHeadMaterializerInputError(
            "canonical source ids must be unique"
        )
    if len(source_ids) > metadata.base_model_config.max_sources:
        raise StageEditHeadMaterializerInputError(
            "canonical source count exceeds checkpoint maximum"
        )
    if batch_index < 0 or batch_index >= len(batch.sequence_ids):
        raise StageEditHeadMaterializerInputError("batch index is outside model batch")
    if batch.sequence_ids[batch_index] != sequence.sequence_id:
        raise StageEditHeadMaterializerInputError(
            "batch sequence identity differs from requested sequence"
        )
    if source_shot_sizes_by_token and len(source_shot_sizes_by_token) != len(
        sequence.tokens
    ):
        raise StageEditHeadMaterializerInputError(
            "source shot-size mappings must align with every token"
        )

    cut_logits = np.asarray(output.cut_timing_logits)
    source_logits = np.asarray(output.source_selection_logits)
    duration_logits = np.asarray(output.duration_bucket_logits)
    transition_logits = np.asarray(output.transition_logits)
    legacy_crop = np.asarray(output.crop)
    output_effective_mask = np.asarray(output.effective_source_mask, dtype=bool)
    candidate_mask = np.asarray(batch.source_candidate_mask, dtype=bool)
    hard_gate_mask = np.asarray(batch.hard_source_gate_mask, dtype=bool)
    expected_effective_mask = np.logical_and(candidate_mask, hard_gate_mask)
    if output_effective_mask.shape != expected_effective_mask.shape or not np.array_equal(
        output_effective_mask, expected_effective_mask
    ):
        raise StageEditHeadMaterializerInputError(
            "model effective-source mask differs from tensorized hard gates"
        )

    token_count = len(sequence.tokens)
    expected_prefix = (len(batch.sequence_ids), batch.sequence_mask.shape[1])
    for name, array, tail in (
        ("cut logits", cut_logits, ()),
        ("source logits", source_logits, (metadata.base_model_config.max_sources,)),
        (
            "duration logits",
            duration_logits,
            (schema.duration_bucket_count,),
        ),
        ("transition logits", transition_logits, (len(schema.transition_labels),)),
        ("legacy crop", legacy_crop, (4,)),
    ):
        if array.shape != (*expected_prefix, *tail):
            raise StageEditHeadMaterializerInputError(
                f"{name} shape {array.shape} violates checkpoint contract"
            )

    frames: list[StageEditHeadFrame] = []
    for token_index, token in enumerate(sequence.tokens):
        candidate_by_id, slot_by_id = _strict_token_source_mapping(
            token=token,
            batch=batch,
            batch_index=batch_index,
            token_index=token_index,
            canonical_source_ids=source_ids,
            candidate_mask=candidate_mask,
            hard_gate_mask=hard_gate_mask,
        )
        remapped_logits: list[float] = []
        available: list[bool] = []
        hard_gates: list[bool] = []
        continuity: list[StageEditSourceContinuity] = []
        for source_id in source_ids:
            slot = slot_by_id.get(source_id)
            if slot is None:
                remapped_logits.append(float(MASKED_SOURCE_LOGIT))
                available.append(False)
                hard_gates.append(False)
                continuity.append(StageEditSourceContinuity())
                continue
            candidate = candidate_by_id[source_id]
            remapped_logits.append(float(source_logits[batch_index, token_index, slot]))
            available.append(bool(candidate_mask[batch_index, token_index, slot]))
            hard_gates.append(bool(hard_gate_mask[batch_index, token_index, slot]))
            continuity.append(_source_continuity_from_candidate(candidate, schema))
        if not any(
            left and right for left, right in zip(available, hard_gates, strict=True)
        ):
            raise StageEditHeadMaterializerInputError(
                f"token {token_index} has no source surviving hard gates"
            )

        try:
            padded_predictions = dynamic_crop_predictions_from_model_token_v3(
                output,
                batch,
                batch_index=batch_index,
                token_index=token_index,
                config=metadata.dynamic_crop_config,
            )
        except (IndexError, TypeError, ValueError, OverflowError) as exc:
            raise StageEditHeadMaterializerInputError(
                f"token {token_index} Dynamic Crop v3 output is invalid: {exc}"
            ) from exc
        if len(padded_predictions) != metadata.base_model_config.max_sources:
            raise StageEditHeadMaterializerInputError(
                f"token {token_index} Dynamic Crop v3 slot count changed"
            )
        canonical_predictions = []
        for canonical_index, source_id in enumerate(source_ids):
            padded_slot = slot_by_id.get(source_id)
            if padded_slot is None:
                canonical_predictions.append(None)
                continue
            prediction = padded_predictions[padded_slot]
            if prediction is not None:
                if (
                    prediction.source_id != source_id
                    or prediction.source_index != padded_slot
                ):
                    raise StageEditHeadMaterializerInputError(
                        f"token {token_index} Dynamic Crop v3 source-slot mismatch"
                    )
                prediction = replace(prediction, source_index=canonical_index)
            canonical_predictions.append(prediction)

        supplied_shot_size_map = (
            source_shot_sizes_by_token[token_index]
            if source_shot_sizes_by_token
            else {}
        )
        frames.append(
            StageEditHeadFrame(
                master_time_seconds=float(token.master_time_seconds),
                cut_logit=float(cut_logits[batch_index, token_index]),
                source_logits=tuple(remapped_logits),
                duration_logits=_finite_tuple(
                    duration_logits[batch_index, token_index],
                    name=f"duration logits token {token_index}",
                    expected=schema.duration_bucket_count,
                ),
                transition_logits=_finite_tuple(
                    transition_logits[batch_index, token_index],
                    name=f"transition logits token {token_index}",
                    expected=len(schema.transition_labels),
                ),
                crop=_finite_tuple(
                    legacy_crop[batch_index, token_index],
                    name=f"legacy crop token {token_index}",
                    expected=4,
                ),
                available_source_mask=tuple(available),
                hard_source_gate_mask=tuple(hard_gates),
                source_shot_sizes=tuple(
                    str(
                        supplied_shot_size_map[source_id]
                        if source_id in supplied_shot_size_map
                        else (
                            _shot_size_from_candidate(
                                candidate_by_id[source_id], schema
                            )
                            if source_id in candidate_by_id
                            else "unknown"
                        )
                    )
                    for source_id in source_ids
                ),
                source_continuity=tuple(continuity),
                candidate_id=(
                    f"{sequence.sequence_id}:token-{token_index:04d}:"
                    f"{float(token.master_time_seconds):.6f}"
                ),
                section_progress=_exact_progress_feature(
                    values=token.section_features,
                    missing_mask=token.missing_mask,
                    names=schema.section_feature_names,
                    feature_name="section_progress",
                    token_index=token_index,
                    missing_mask_offset=len(schema.music_feature_names),
                ),
                phrase_progress=_exact_progress_feature(
                    values=token.music_features,
                    missing_mask=token.missing_mask,
                    names=schema.music_feature_names,
                    feature_name="phrase_progress",
                    token_index=token_index,
                    missing_mask_offset=0,
                ),
                section_role=_exact_section_role(
                    values=token.section_features,
                    missing_mask=token.missing_mask,
                    schema=schema,
                    token_index=token_index,
                ),
                dynamic_crop_v3_by_source=tuple(canonical_predictions),
            )
        )
    if len(frames) != token_count:
        raise AssertionError("internal StageEdit v3 head materialization count mismatch")
    return tuple(frames)


def materialize_stage_edit_head_frames_v2(
    *,
    checkpoint_directory: Path,
    sequence: StageEditSequence,
    canonical_source_ids: Sequence[str],
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
    source_shot_sizes_by_token: Sequence[Mapping[str, str]] = (),
) -> StageEditHeadMaterializationResult:
    """Run a verified production Dynamic Crop v2 checkpoint exactly once."""

    checkpoint = Path(checkpoint_directory).expanduser().resolve()
    dynamic_metadata_path = checkpoint / DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME
    legacy_metadata_path = checkpoint / "metadata.json"
    if legacy_metadata_path.exists() or not dynamic_metadata_path.is_file():
        raise StageEditHeadMaterializerNotReadyError(
            "head materialization requires one unambiguous Dynamic Crop v2 checkpoint"
        )
    readiness = stage_edit_readiness_report(checkpoint)
    if not (
        readiness.architecture_available
        and readiness.learned_inference_ready
        and readiness.status == "production_checkpoint_ready"
        and readiness.checkpoint_path == str(checkpoint)
        and readiness.training_steps > 0
    ):
        raise StageEditHeadMaterializerNotReadyError(
            "Dynamic Crop v2 checkpoint rejected: "
            f"{readiness.status}: {readiness.reason}"
        )
    try:
        metadata_payload = json.loads(dynamic_metadata_path.read_text(encoding="utf-8"))
        metadata = StageEditDynamicCropCheckpointMetadataV2.from_dict(metadata_payload)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StageEditHeadMaterializerNotReadyError(
            f"Dynamic Crop v2 metadata is invalid: {exc}"
        ) from exc
    if metadata.token_schema.fingerprint != schema.fingerprint:
        raise StageEditHeadMaterializerInputError(
            "requested token schema differs from production checkpoint"
        )
    weight_path = checkpoint / metadata.weight_filename
    before = {
        "metadata": _sha256_file(dynamic_metadata_path),
        "weights": _sha256_file(weight_path),
    }
    supplied_sequence_sha256 = _canonical_sha256(_sequence_payload(sequence))
    model_sequence = mask_stage_edit_recent_history_for_semimarkov(
        sequence, schema
    )
    history_contract = stage_edit_recent_history_mask_contract(schema)
    try:
        model, loaded_metadata = load_stage_edit_dynamic_crop_checkpoint_v2(
            checkpoint,
            expected_schema=schema,
            require_trained=True,
        )
        if loaded_metadata != metadata:
            raise ValueError("loaded metadata identity changed")
        if hasattr(model, "eval"):
            model.eval()
        batch = tensorize_stage_edit_batch(
            (model_sequence,), schema, metadata.base_model_config
        )
        output = model(batch)
        import mlx.core as mx

        mx.eval(
            output.cut_timing_logits,
            output.source_selection_logits,
            output.duration_bucket_logits,
            output.transition_logits,
            output.crop,
            output.effective_source_mask,
            output.dynamic_crop_values,
            output.dynamic_crop_normalized_times,
            output.dynamic_crop_presence_logits,
            output.dynamic_crop_motion_logits,
        )
        frames = stage_edit_head_frames_from_dynamic_crop_output_v2(
            sequence=model_sequence,
            batch=batch,
            output=output,
            canonical_source_ids=canonical_source_ids,
            schema=schema,
            metadata=metadata,
            source_shot_sizes_by_token=source_shot_sizes_by_token,
        )
    except StageEditHeadMaterializerError:
        raise
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise StageEditHeadMaterializerError(
            f"production Dynamic Crop v2 forward failed: {exc}"
        ) from exc
    after = {
        "metadata": _sha256_file(dynamic_metadata_path),
        "weights": _sha256_file(weight_path),
    }
    if before != after or metadata.weight_sha256 != after["weights"]:
        raise StageEditHeadMaterializerNotReadyError(
            "Dynamic Crop v2 checkpoint changed during inference"
        )
    sequence_sha256 = _canonical_sha256(_sequence_payload(model_sequence))
    frames_sha256 = _canonical_sha256([item.to_dict() for item in frames])
    return StageEditHeadMaterializationResult(
        frames=frames,
        sequence_id=sequence.sequence_id,
        canonical_source_ids=tuple(str(value).strip() for value in canonical_source_ids),
        checkpoint_directory=str(checkpoint),
        checkpoint_id=metadata.checkpoint_id,
        training_steps=metadata.training_steps,
        metadata_sha256=after["metadata"],
        weights_sha256=after["weights"],
        supplied_sequence_sha256=supplied_sequence_sha256,
        input_sequence_sha256=sequence_sha256,
        output_frames_sha256=frames_sha256,
        token_schema_fingerprint=schema.fingerprint,
        dynamic_crop_config_fingerprint=metadata.dynamic_crop_config.fingerprint,
        readiness=readiness,
        decision_history_input_contract=history_contract,
    )


__all__ = [
    "STAGE_EDIT_HEAD_MATERIALIZER_CONTRACT",
    "STAGE_EDIT_HEAD_MATERIALIZER_SCHEMA_VERSION",
    "StageEditHeadMaterializationResult",
    "StageEditHeadMaterializerError",
    "StageEditHeadMaterializerInputError",
    "StageEditHeadMaterializerNotReadyError",
    "materialize_stage_edit_head_frames_v2",
    "stage_edit_head_frames_from_dynamic_crop_output_v2",
    "stage_edit_head_frames_from_dynamic_crop_output_v3",
]
