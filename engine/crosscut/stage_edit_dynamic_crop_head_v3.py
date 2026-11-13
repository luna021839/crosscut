"""Source-conditioned, renderer-safe StageEdit Dynamic Crop Head v3.

Version 3 is intentionally additive and checkpoint-incompatible with v2.  It
fixes four properties that cannot be repaired by additional v2 training:

* every candidate source receives its own permutation-equivariant trajectory;
* renderer coordinates are produced on a bounded, renderer-safe manifold;
* a categorical 1..8 keyframe count replaces independent presence bits; and
* each count owns an independent simplex of time gaps, so unused tail slots
  cannot move the selected prefix.

Existing verified :class:`DynamicCropTrajectoryTargetV2` objects remain valid
target evidence.  Their source identity is resolved to one exact, hard-gated
candidate slot during v3 tensorization.  V2 model weights and metadata are
never reinterpreted as v3.
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

from .stage_edit_dynamic_crop_head_v2 import (
    DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME,
    StageEditDynamicCropTargetSequenceV2,
)
from .stage_edit_dynamic_crop_target_v3 import (
    SimplifiedDynamicCropTrajectoryV3,
    simplify_dynamic_crop_trajectory_v3,
)
from .stage_edit_transformer import (
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


DYNAMIC_CROP_V3_SCHEMA_VERSION = 3
DYNAMIC_CROP_V3_MAX_KEYFRAMES = 8
DYNAMIC_CROP_V3_TIME_LOGIT_COUNT = sum(
    range(2, DYNAMIC_CROP_V3_MAX_KEYFRAMES + 1)
)
DYNAMIC_CROP_V3_TIME_SIMPLEX_EPSILON = 1.0e-4
DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION = (
    DYNAMIC_CROP_V3_MAX_KEYFRAMES * 4
    + DYNAMIC_CROP_V3_MAX_KEYFRAMES
    + DYNAMIC_CROP_V3_TIME_LOGIT_COUNT
    + 1
)
DYNAMIC_CROP_V3_TARGET_CONTRACT = (
    "source_slot_aligned_observed_only_simplified_renderer_trajectory_target_v3"
)
DYNAMIC_CROP_V3_MODEL_CONTRACT = (
    "stage_edit_causal_transformer_source_conditioned_dynamic_crop_head_v3"
)
DYNAMIC_CROP_V3_OUTPUT_CONTRACT = (
    "source_values_count_conditional_times_motion_output_v3"
)
DYNAMIC_CROP_V3_SOURCE_CONDITIONING = (
    "shared_context_plus_shared_candidate_affine_permutation_equivariant_v3"
)
DYNAMIC_CROP_V3_INTERACTION_SOURCE_CONDITIONING = (
    "shared_centered_context_times_candidate_zero_init_interaction_v3"
)
DYNAMIC_CROP_V3_TIME_PARAMETERIZATION = (
    "count_specific_epsilon_interior_gap_simplex_with_explicit_tail_v3"
)
DYNAMIC_CROP_V3_VALUE_PARAMETERIZATION = (
    "renderer_safe_radial_pan_roll_exact_corner_and_convex_linf_interpolation_zoom_floor_v3"
)
DYNAMIC_CROP_V3_LOSS_CONTRACT = (
    "renderer_range_normalized_values_count_times_motion_and_played_curve_loss_v3"
)
DYNAMIC_CROP_V3_CHECKPOINT_FORMAT_VERSION = 2
DYNAMIC_CROP_V3_AFFINE_HEAD_REPRESENTATION = "fixed_train_centered_affine_full_rank_v1"
DYNAMIC_CROP_V3_INTERACTION_CHECKPOINT_FORMAT_VERSION = 3
DYNAMIC_CROP_V3_INTERACTION_HEAD_REPRESENTATION = "fixed_train_centered_affine_plus_context_source_product_v1"
DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME = "dynamic-crop-v3-metadata.json"
DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME = "dynamic-crop-v3-model.safetensors"

_MINIMUM_ZOOM = 1.0
_MAXIMUM_ZOOM = 1.18
_MAXIMUM_CENTER_SHIFT = 0.040
_MAXIMUM_ROLL_DEGREES = 1.60
_MANIFOLD_CENTER_SHIFT = _MAXIMUM_CENTER_SHIFT - 1.0e-4
_MANIFOLD_ROLL_DEGREES = _MAXIMUM_ROLL_DEGREES - 1.0e-3
_EDGE_MARGIN = 0.003
_PRIOR_ZOOM = 1.02
_CONVEX_CENTER_EPSILON = 1.0e-12
_CONVEX_PAN_ZOOM_SLOPE = 2.201
_CONVEX_ROLL_ZOOM_SLOPE = 0.05745
_PRIOR_ZOOM_FLOOR = (
    _MINIMUM_ZOOM
    + _CONVEX_PAN_ZOOM_SLOPE * math.sqrt(_CONVEX_CENTER_EPSILON)
)
_PRIOR_ZOOM_LOGIT = math.log(
    ((_PRIOR_ZOOM - _PRIOR_ZOOM_FLOOR) / (_MAXIMUM_ZOOM - _PRIOR_ZOOM_FLOOR))
    / (
        1.0
        - (_PRIOR_ZOOM - _PRIOR_ZOOM_FLOOR)
        / (_MAXIMUM_ZOOM - _PRIOR_ZOOM_FLOOR)
    )
)


def _require_mlx() -> None:
    if mx is None or nn is None:
        detail = f": {_MLX_IMPORT_ERROR}" if _MLX_IMPORT_ERROR else ""
        raise RuntimeError(f"Dynamic Crop Head v3 requires MLX{detail}")


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


def _time_block_slices() -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    start = 0
    for count in range(2, DYNAMIC_CROP_V3_MAX_KEYFRAMES + 1):
        result[count] = (start, start + count)
        start += count
    if start != DYNAMIC_CROP_V3_TIME_LOGIT_COUNT:
        raise AssertionError("Dynamic Crop v3 time block layout changed")
    return result


DYNAMIC_CROP_V3_TIME_BLOCK_SLICES = _time_block_slices()


@dataclass(frozen=True, slots=True)
class DynamicCropHeadV3Config:
    schema_version: int = DYNAMIC_CROP_V3_SCHEMA_VERSION
    max_keyframes: int = DYNAMIC_CROP_V3_MAX_KEYFRAMES
    minimum_zoom: float = _MINIMUM_ZOOM
    maximum_zoom: float = _MAXIMUM_ZOOM
    maximum_center_shift: float = _MAXIMUM_CENTER_SHIFT
    maximum_roll_degrees: float = _MAXIMUM_ROLL_DEGREES
    prior_zoom: float = _PRIOR_ZOOM
    target_contract: str = DYNAMIC_CROP_V3_TARGET_CONTRACT
    output_contract: str = DYNAMIC_CROP_V3_OUTPUT_CONTRACT
    source_conditioning: str = DYNAMIC_CROP_V3_SOURCE_CONDITIONING
    time_parameterization: str = DYNAMIC_CROP_V3_TIME_PARAMETERIZATION
    value_parameterization: str = DYNAMIC_CROP_V3_VALUE_PARAMETERIZATION

    def __post_init__(self) -> None:
        if self.schema_version != DYNAMIC_CROP_V3_SCHEMA_VERSION:
            raise ValueError("unsupported Dynamic Crop Head v3 schema version")
        if self.max_keyframes != DYNAMIC_CROP_V3_MAX_KEYFRAMES:
            raise ValueError("Dynamic Crop Head v3 requires exactly K=8")
        expected = (
            (self.minimum_zoom, _MINIMUM_ZOOM),
            (self.maximum_zoom, _MAXIMUM_ZOOM),
            (self.maximum_center_shift, _MAXIMUM_CENTER_SHIFT),
            (self.maximum_roll_degrees, _MAXIMUM_ROLL_DEGREES),
            (self.prior_zoom, _PRIOR_ZOOM),
        )
        if any(not math.isclose(_finite(value, "renderer bound"), required) for value, required in expected):
            raise ValueError("Dynamic Crop Head v3 renderer bounds are fixed")
        contracts = (
            (self.target_contract, DYNAMIC_CROP_V3_TARGET_CONTRACT),
            (self.output_contract, DYNAMIC_CROP_V3_OUTPUT_CONTRACT),
            (self.time_parameterization, DYNAMIC_CROP_V3_TIME_PARAMETERIZATION),
            (self.value_parameterization, DYNAMIC_CROP_V3_VALUE_PARAMETERIZATION),
        )
        if any(value != required for value, required in contracts):
            raise ValueError("Dynamic Crop Head v3 contract is invalid")
        if self.source_conditioning not in (
            DYNAMIC_CROP_V3_SOURCE_CONDITIONING,
            DYNAMIC_CROP_V3_INTERACTION_SOURCE_CONDITIONING,
        ):
            raise ValueError("Dynamic Crop Head v3 source conditioning is invalid")

    @property
    def uses_context_source_interaction(self) -> bool:
        return self.source_conditioning == DYNAMIC_CROP_V3_INTERACTION_SOURCE_CONDITIONING

    @property
    def head_representation(self) -> str:
        return (DYNAMIC_CROP_V3_INTERACTION_HEAD_REPRESENTATION
                if self.uses_context_source_interaction
                else DYNAMIC_CROP_V3_AFFINE_HEAD_REPRESENTATION)

    @property
    def checkpoint_format_version(self) -> int:
        return (DYNAMIC_CROP_V3_INTERACTION_CHECKPOINT_FORMAT_VERSION
                if self.uses_context_source_interaction
                else DYNAMIC_CROP_V3_CHECKPOINT_FORMAT_VERSION)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class StageEditDynamicCropTargetSequenceV3:
    """V3 sequence of renderer-verified curve-complexity targets."""

    sequence_id: str
    targets: tuple[SimplifiedDynamicCropTrajectoryV3 | None, ...]
    target_contract: str = DYNAMIC_CROP_V3_TARGET_CONTRACT

    def __post_init__(self) -> None:
        sequence_id = str(self.sequence_id).strip()
        if not sequence_id:
            raise ValueError("dynamic crop v3 target sequence id cannot be blank")
        targets = tuple(self.targets)
        if not targets:
            raise ValueError("dynamic crop v3 target sequence cannot be empty")
        if any(
            item is not None
            and not isinstance(item, SimplifiedDynamicCropTrajectoryV3)
            for item in targets
        ):
            raise TypeError("dynamic crop v3 sequence contains an invalid target")
        if self.target_contract != DYNAMIC_CROP_V3_TARGET_CONTRACT:
            raise ValueError("dynamic crop v3 target contract is invalid")
        object.__setattr__(self, "sequence_id", sequence_id)
        object.__setattr__(self, "targets", targets)

    @classmethod
    def from_v2(
        cls, value: StageEditDynamicCropTargetSequenceV2
    ) -> "StageEditDynamicCropTargetSequenceV3":
        if not isinstance(value, StageEditDynamicCropTargetSequenceV2):
            raise TypeError("value must be a Dynamic Crop v2 target sequence")
        return cls(
            value.sequence_id,
            tuple(
                simplify_dynamic_crop_trajectory_v3(target)
                if target is not None
                else None
                for target in value.targets
            ),
        )


@dataclass(frozen=True, slots=True)
class StageEditDynamicCropSupervisionBatchV3:
    values: Any
    normalized_times: Any
    keyframe_mask: Any
    count_targets: Any
    count_mask: Any
    motion_targets: Any
    motion_mask: Any
    shot_duration_seconds: Any
    track_mask: Any
    target_source_indices: Any
    sequence_ids: tuple[str, ...]
    target_contract: str = DYNAMIC_CROP_V3_TARGET_CONTRACT
    # Supervision-only original evidence, never copied into model features.
    original_observed_keyframes: Mapping[tuple[int,int,int],tuple[Any,...]] | None = None


def _coerce_target_sequence_v3(
    value: StageEditDynamicCropTargetSequenceV3 | StageEditDynamicCropTargetSequenceV2,
) -> StageEditDynamicCropTargetSequenceV3:
    if isinstance(value, StageEditDynamicCropTargetSequenceV3):
        return value
    if isinstance(value, StageEditDynamicCropTargetSequenceV2):
        return StageEditDynamicCropTargetSequenceV3.from_v2(value)
    raise TypeError("dynamic crop v3 tensorization received an invalid sequence")


def tensorize_stage_edit_dynamic_crop_supervision_v3(
    target_sequences: Sequence[
        StageEditDynamicCropTargetSequenceV3 | StageEditDynamicCropTargetSequenceV2
    ],
    batch: StageEditTensorBatch,
    *,
    config: DynamicCropHeadV3Config = DynamicCropHeadV3Config(),
    device: Any = None,
) -> StageEditDynamicCropSupervisionBatchV3:
    """Resolve each trajectory to exactly one available, hard-gated source slot."""

    _require_mlx()
    normalized = tuple(_coerce_target_sequence_v3(item) for item in target_sequences)
    by_id = {item.sequence_id: item for item in normalized}
    if len(by_id) != len(normalized):
        raise ValueError("dynamic crop v3 target sequence ids must be unique")
    batch_size, max_length, max_sources = batch.source_candidate_mask.shape
    keyframes = config.max_keyframes
    values = np.zeros(
        (batch_size, max_length, max_sources, keyframes, 4), dtype=np.float32
    )
    times = np.zeros(
        (batch_size, max_length, max_sources, keyframes), dtype=np.float32
    )
    keyframe_mask = np.zeros_like(times, dtype=bool)
    counts = np.zeros((batch_size, max_length, max_sources), dtype=np.int32)
    count_mask = np.zeros_like(counts, dtype=bool)
    motion_targets = np.zeros_like(counts, dtype=np.float32)
    motion_mask = np.zeros_like(counts, dtype=bool)
    durations = np.zeros_like(counts, dtype=np.float32)
    track_mask = np.zeros_like(counts, dtype=bool)
    target_source_indices = np.full((batch_size, max_length), -1, dtype=np.int32)
    original_observed_keyframes={}
    candidate_mask = np.asarray(batch.source_candidate_mask, dtype=bool)
    gate_mask = np.asarray(batch.hard_source_gate_mask, dtype=bool)

    for batch_index, sequence_id in enumerate(batch.sequence_ids):
        sequence = by_id.get(sequence_id)
        if sequence is None:
            continue
        valid_length = int(np.asarray(batch.sequence_mask[batch_index]).sum())
        if len(sequence.targets) != valid_length:
            raise ValueError(
                f"dynamic crop v3 target count for {sequence_id} differs from tokens"
            )
        for time_index, target in enumerate(sequence.targets):
            if target is None:
                continue
            slots = tuple(batch.source_ids[batch_index][time_index])
            matches = tuple(
                index for index, source_id in enumerate(slots)
                if source_id == target.source_id
            )
            if len(matches) != 1:
                raise ValueError(
                    "dynamic crop v3 target source has no unique candidate slot: "
                    f"{sequence_id}:{time_index}:{target.source_id}"
                )
            source_index = matches[0]
            if target.original_observed_keyframes:
                original_observed_keyframes[batch_index,time_index,source_index]=target.original_observed_keyframes
            if not candidate_mask[batch_index, time_index, source_index]:
                raise ValueError("dynamic crop v3 target source is unavailable")
            if not gate_mask[batch_index, time_index, source_index]:
                raise ValueError("dynamic crop v3 target source failed its hard gate")
            count = len(target.keyframes)
            if not 1 <= count <= keyframes:
                raise ValueError("dynamic crop v3 target requires 1..8 keyframes")
            duration = target.shot_duration_seconds
            target_source_indices[batch_index, time_index] = source_index
            track_mask[batch_index, time_index, source_index] = True
            count_mask[batch_index, time_index, source_index] = True
            counts[batch_index, time_index, source_index] = count
            durations[batch_index, time_index, source_index] = duration
            if not target.motion_label_loss_masked:
                motion_mask[batch_index, time_index, source_index] = True
                motion_targets[batch_index, time_index, source_index] = float(
                    target.motion_label == "dynamic"
                )
            for keyframe_index, item in enumerate(target.keyframes):
                normalized_time = item.time_seconds / duration
                if not 0.0 <= normalized_time <= 1.0:
                    raise ValueError("dynamic crop v3 target time is outside [0, 1]")
                times[
                    batch_index, time_index, source_index, keyframe_index
                ] = normalized_time
                values[
                    batch_index, time_index, source_index, keyframe_index
                ] = (
                    item.zoom,
                    item.center_x,
                    item.center_y,
                    item.roll_degrees,
                )
                keyframe_mask[
                    batch_index, time_index, source_index, keyframe_index
                ] = True

    stream = device if device is not None else mx.default_device()
    with mx.stream(stream):
        return StageEditDynamicCropSupervisionBatchV3(
            values=mx.array(values),
            normalized_times=mx.array(times),
            keyframe_mask=mx.array(keyframe_mask),
            count_targets=mx.array(counts),
            count_mask=mx.array(count_mask),
            motion_targets=mx.array(motion_targets),
            motion_mask=mx.array(motion_mask),
            shot_duration_seconds=mx.array(durations),
            track_mask=mx.array(track_mask),
            target_source_indices=mx.array(target_source_indices),
            sequence_ids=tuple(batch.sequence_ids),
            original_observed_keyframes=original_observed_keyframes,
        )


def _minimum_renderer_safe_zoom_v3(
    center_x: Any, center_y: Any, roll_degrees: Any
) -> Any:
    """Vectorized exact renderer corner lower bound for zoom."""

    radians = roll_degrees * (math.pi / 180.0)
    cosine = mx.cos(radians)
    sine = mx.sin(radians)
    absolute_cosine = mx.abs(cosine)
    absolute_sine = mx.abs(sine)
    absolute_roll = mx.abs(roll_degrees)
    strength = mx.minimum(mx.array(1.0, dtype=roll_degrees.dtype), absolute_roll)
    limit = 0.5 - _EDGE_MARGIN * strength
    dx = center_x - 0.5
    dy = center_y - 0.5

    requirements: list[Any] = []
    for aspect in (16.0 / 9.0, 9.0 / 16.0):
        projected_x = cosine * dx + sine * dy / aspect
        x_extent = absolute_cosine + absolute_sine / aspect
        x_room = limit - mx.abs(projected_x)
        requirements.append(0.5 * x_extent / mx.maximum(x_room, 1.0e-6))

        projected_y = -sine * aspect * dx + cosine * dy
        y_extent = absolute_sine * aspect + absolute_cosine
        y_room = limit - mx.abs(projected_y)
        requirements.append(0.5 * y_extent / mx.maximum(y_room, 1.0e-6))

    absolute_radians = absolute_roll * (math.pi / 180.0)
    roll_base = mx.cos(absolute_radians) + (16.0 / 9.0) * mx.sin(
        absolute_radians
    )
    roll_floor = roll_base / (1.0 - 2.0 * _EDGE_MARGIN * strength)
    roll_floor = roll_floor + 0.020 * strength
    required = mx.maximum(mx.array(_MINIMUM_ZOOM, dtype=center_x.dtype), roll_floor)
    for value in requirements:
        required = mx.maximum(required, value)
    return required


def renderer_safe_values_from_raw_v3(raw_values: Any) -> Any:
    """Map unconstrained logits to exact physical renderer coordinates."""

    _require_mlx()
    if raw_values.shape[-1] != 4:
        raise ValueError("Dynamic Crop v3 raw values need four components")
    raw_zoom = raw_values[..., 0]
    raw_center = raw_values[..., 1:3]
    raw_roll = raw_values[..., 3]
    # The epsilon is inside the square root so the radial map has a finite
    # identity-like derivative at the zero-initialized prior.  A branch around
    # ``sqrt(sum(x*x)) == 0`` still back-propagates NaNs in eager autodiff even
    # when the selected branch is constant.
    norm = mx.sqrt(
        (raw_center * raw_center).sum(axis=-1, keepdims=True) + 1.0e-12
    )
    radial_ratio = mx.tanh(norm) / norm
    # Keep the float32 saturation point strictly inside renderer boundaries.
    # ``tanh(large)`` is exactly one in float32, while 0.04/1.60 themselves
    # can round a few ulps beyond the scalar renderer checks.
    center_offset = _MANIFOLD_CENTER_SHIFT * radial_ratio * raw_center
    center_x = 0.5 + center_offset[..., 0]
    center_y = 0.5 + center_offset[..., 1]
    roll = _MANIFOLD_ROLL_DEGREES * mx.tanh(raw_roll)
    zoom_floor = _minimum_renderer_safe_zoom_v3(center_x, center_y, roll)
    center_strength = (
        (center_offset * center_offset).sum(axis=-1)
        / (_MAXIMUM_CENTER_SHIFT * _MAXIMUM_CENTER_SHIFT)
    )
    geometry_strength = mx.maximum(
        center_strength,
        mx.abs(roll) / _MAXIMUM_ROLL_DEGREES,
    )
    zoom_floor = mx.minimum(
        _MAXIMUM_ZOOM - 1.0e-4,
        zoom_floor + 1.0e-4 * mx.minimum(geometry_strength, 1.0),
    )
    # Use the convex L-infinity pan envelope, not the unnecessarily larger
    # Euclidean radius: a safe diagonal teacher must remain representable.
    # With m=max(|dx|,|dy|)<=.04, r=|roll|<=1.6, b=(16/9)*pi/180,
    # each exact corner floor is <= (1+b*r)/(1-.006*min(1,r)-2*m*(1+b*r)).
    # Multiplying by that positive denominator gives a separately concave
    # polynomial in m and r on r:[0,1] and [1,1.6]. The six rectangle corners
    # certify 1+2.201*m+.05745*r as an upper bound; the roll-only extra .020
    # margin satisfies the same piecewise concave endpoint check. L-infinity
    # norm plus absolute roll is convex, preserving safety BETWEEN knots.
    # Keep the old positive origin floor, prior initialization and finite
    # zero-center gradients. Teacher values and renderer bounds do not move.
    convex_center_distance = mx.maximum(
        mx.abs(center_offset[..., 0]), mx.abs(center_offset[..., 1])
    ) + math.sqrt(_CONVEX_CENTER_EPSILON)
    convex_zoom_floor = (
        _MINIMUM_ZOOM
        + _CONVEX_PAN_ZOOM_SLOPE * convex_center_distance
        + _CONVEX_ROLL_ZOOM_SLOPE * mx.abs(roll)
    )
    zoom_floor = mx.maximum(zoom_floor, convex_zoom_floor)
    zoom = zoom_floor + (_MAXIMUM_ZOOM - zoom_floor) * mx.sigmoid(raw_zoom)
    return mx.stack((zoom, center_x, center_y, roll), axis=-1)


def renderer_range_normalized_values_v3(values: Any) -> Any:
    """Put physical renderer coordinates on comparable loss scales."""

    _require_mlx()
    scales = mx.array(
        (
            _MAXIMUM_ZOOM - _MINIMUM_ZOOM,
            _MAXIMUM_CENTER_SHIFT,
            _MAXIMUM_CENTER_SHIFT,
            _MAXIMUM_ROLL_DEGREES,
        ),
        dtype=values.dtype,
    )
    origins = mx.array(
        (_MINIMUM_ZOOM, 0.5, 0.5, 0.0), dtype=values.dtype
    )
    return (values - origins) / scales


def dynamic_crop_v3_times_by_count(time_interval_logits: Any) -> Any:
    """Return ``[..., count_class=8, K=8]`` monotonic normalized times."""

    _require_mlx()
    if time_interval_logits.shape[-1] != DYNAMIC_CROP_V3_TIME_LOGIT_COUNT:
        raise ValueError("Dynamic Crop v3 needs exactly 35 time logits")
    prefix_shape = tuple(int(value) for value in time_interval_logits.shape[:-1])
    rows: list[Any] = [
        mx.zeros((*prefix_shape, DYNAMIC_CROP_V3_MAX_KEYFRAMES), dtype=time_interval_logits.dtype)
    ]
    for count in range(2, DYNAMIC_CROP_V3_MAX_KEYFRAMES + 1):
        start, stop = DYNAMIC_CROP_V3_TIME_BLOCK_SLICES[count]
        gaps = mx.softmax(time_interval_logits[..., start:stop], axis=-1)
        gaps = (
            (1.0 - DYNAMIC_CROP_V3_TIME_SIMPLEX_EPSILON) * gaps
            + DYNAMIC_CROP_V3_TIME_SIMPLEX_EPSILON / count
        )
        observed = mx.concatenate(
            (
                mx.zeros((*prefix_shape, 1), dtype=time_interval_logits.dtype),
                mx.cumsum(gaps[..., :-1], axis=-1),
            ),
            axis=-1,
        )
        padding = mx.zeros(
            (*prefix_shape, DYNAMIC_CROP_V3_MAX_KEYFRAMES - count),
            dtype=time_interval_logits.dtype,
        )
        rows.append(mx.concatenate((observed, padding), axis=-1))
    return mx.stack(rows, axis=-2)


@dataclass(frozen=True, slots=True)
class StageEditDynamicCropModelOutputV3:
    cut_timing_logits: Any
    source_selection_logits: Any
    duration_bucket_logits: Any
    transition_logits: Any
    crop: Any
    effective_source_mask: Any
    dynamic_crop_values: Any
    dynamic_crop_count_logits: Any
    dynamic_crop_time_interval_logits: Any
    dynamic_crop_motion_logits: Any
    output_contract: str = DYNAMIC_CROP_V3_OUTPUT_CONTRACT


_ModuleBase = nn.Module if nn is not None else object
_LinearBase = nn.Linear if nn is not None else object


class _FixedCenteredLinear(_LinearBase):
    """Full-rank trainable affine head with a non-optimizer input center.

    Private MLX state is absent from parameters()/trainable_parameters(), so
    neither gradients, global unfreeze nor AdamW can alter the fixed center.
    The v2 checkpoint writer explicitly persists these two constant buffers.
    """

    def __init__(self, input_dims: int, output_dims: int, *, bias: bool = True):
        _require_mlx()
        super().__init__(input_dims, output_dims, bias=bias)
        self._input_center = mx.zeros((input_dims,), dtype=mx.float32)

    @property
    def input_center(self):
        return self._input_center

    def set_input_center(self, value):
        values = np.asarray(value, dtype=np.float32)
        if values.shape != (self.weight.shape[-1],) or not np.isfinite(values).all():
            raise ValueError("Dynamic Crop v3 fixed affine center is invalid")
        self._input_center = mx.array(np.array(values, copy=True))

    def __call__(self, x):
        return super().__call__(x - self._input_center)


class StageEditTransformerDynamicCropV3(_ModuleBase):
    """Shared-weight source-conditioned Dynamic Crop v3 model."""

    def __init__(
        self,
        base_config: StageEditTransformerConfig,
        dynamic_crop_config: DynamicCropHeadV3Config = DynamicCropHeadV3Config(),
    ) -> None:
        _require_mlx()
        super().__init__()
        self.base_config = base_config
        self.dynamic_crop_config = dynamic_crop_config
        self.base = StageEditTransformer(base_config)
        dimension = base_config.model_dimension
        outputs = DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION
        self.dynamic_crop_context_head = _FixedCenteredLinear(dimension, outputs)
        self.dynamic_crop_source_head = _FixedCenteredLinear(dimension, outputs, bias=False)
        self.dynamic_crop_context_head.weight = mx.zeros_like(
            self.dynamic_crop_context_head.weight
        )
        self.dynamic_crop_source_head.weight = mx.zeros_like(
            self.dynamic_crop_source_head.weight
        )
        prior = np.zeros(outputs, dtype=np.float32)
        for keyframe_index in range(DYNAMIC_CROP_V3_MAX_KEYFRAMES):
            prior[keyframe_index * 4] = _PRIOR_ZOOM_LOGIT
        self.dynamic_crop_context_head.bias = mx.array(prior)
        if dynamic_crop_config.uses_context_source_interaction:
            self.dynamic_crop_interaction_head = nn.Linear(dimension, outputs, bias=False)
            self.dynamic_crop_interaction_head.weight = mx.zeros_like(
                self.dynamic_crop_interaction_head.weight
            )

    def dynamic_crop_raw(self, hidden: Any, source_embeddings: Any) -> Any:
        """One native-rank expression shared by public and diagnostic paths.

        Candidate-local products allow source motion ordering to depend on
        context without changing source masks or count/time/renderer meaning.
        The opt-in head is exactly zero at initialization; legacy checkpoints
        keep their original expression, parameter tree and format.
        """
        raw = (self.dynamic_crop_context_head(hidden)[..., None, :]
               + self.dynamic_crop_source_head(source_embeddings))
        if self.dynamic_crop_config.uses_context_source_interaction:
            context = (hidden - self.dynamic_crop_context_head.input_center)[..., None, :]
            source = source_embeddings - self.dynamic_crop_source_head.input_center
            raw = raw + self.dynamic_crop_interaction_head(context * source)
        return raw

    @property
    def checkpoint_contract(self) -> dict[str, Any]:
        return {
            "schema_version": DYNAMIC_CROP_V3_SCHEMA_VERSION,
            "model_contract": DYNAMIC_CROP_V3_MODEL_CONTRACT,
            "base_model_config": self.base_config.to_dict(),
            "dynamic_crop_config": self.dynamic_crop_config.to_dict(),
            "dynamic_crop_config_fingerprint": self.dynamic_crop_config.fingerprint,
            "source_conditioning": self.dynamic_crop_config.source_conditioning,
            "time_parameterization": DYNAMIC_CROP_V3_TIME_PARAMETERIZATION,
            "value_parameterization": DYNAMIC_CROP_V3_VALUE_PARAMETERIZATION,
            "affine_head_representation": self.dynamic_crop_config.head_representation,
            "legacy_v2_checkpoint_compatible": False,
            "production_activation_authorized": False,
        }

    def __call__(self, batch: StageEditTensorBatch) -> StageEditDynamicCropModelOutputV3:
        (
            hidden,
            relational_source_features,
            source_embeddings,
            effective_mask,
        ) = self.base._hidden_and_sources(batch)
        source_logits = self.base._source_logits(
            hidden,
            relational_source_features,
            source_embeddings,
            effective_mask,
        )
        raw = self.dynamic_crop_raw(hidden, source_embeddings)
        values_stop = DYNAMIC_CROP_V3_MAX_KEYFRAMES * 4
        count_stop = values_stop + DYNAMIC_CROP_V3_MAX_KEYFRAMES
        time_stop = count_stop + DYNAMIC_CROP_V3_TIME_LOGIT_COUNT
        raw_values = raw[..., :values_stop].reshape(
            (*raw.shape[:-1], DYNAMIC_CROP_V3_MAX_KEYFRAMES, 4)
        )
        return StageEditDynamicCropModelOutputV3(
            cut_timing_logits=self.base._cut_logits(hidden, batch, effective_mask),
            source_selection_logits=source_logits,
            duration_bucket_logits=self.base.duration_head(hidden),
            transition_logits=self.base.transition_head(hidden),
            crop=self.base.crop_head(hidden),
            effective_source_mask=effective_mask,
            dynamic_crop_values=renderer_safe_values_from_raw_v3(raw_values),
            dynamic_crop_count_logits=raw[..., values_stop:count_stop],
            dynamic_crop_time_interval_logits=raw[..., count_stop:time_stop],
            dynamic_crop_motion_logits=raw[..., time_stop],
        )


def stage_edit_dynamic_crop_v3_parameter_count(
    base_config: StageEditTransformerConfig,
    dynamic_crop_config: DynamicCropHeadV3Config = DynamicCropHeadV3Config(),
) -> int:
    if dynamic_crop_config.max_keyframes != DYNAMIC_CROP_V3_MAX_KEYFRAMES:
        raise ValueError("Dynamic Crop v3 parameter count requires K=8")
    dimension = base_config.model_dimension
    outputs = DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION
    dynamic_parameters = (dimension + 1) * outputs + dimension * outputs
    if dynamic_crop_config.uses_context_source_interaction:
        dynamic_parameters += dimension * outputs
    return stage_edit_parameter_count(base_config) + dynamic_parameters


@dataclass(frozen=True, slots=True)
class DynamicCropV3LossContract:
    base: StageEditLossContract = StageEditLossContract()
    base_loss_weight: float = 1.0
    value_weight: float = 1.0
    count_weight: float = 0.35
    time_weight: float = 0.50
    motion_weight: float = 0.15
    decoded_curve_weight: float = 1.0
    contract: str = DYNAMIC_CROP_V3_LOSS_CONTRACT

    def __post_init__(self) -> None:
        if not isinstance(self.base, StageEditLossContract):
            raise TypeError("base loss contract must be StageEditLossContract")
        for name in (
            "base_loss_weight",
            "value_weight",
            "count_weight",
            "time_weight",
            "motion_weight",
            "decoded_curve_weight",
        ):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if self.contract != DYNAMIC_CROP_V3_LOSS_CONTRACT:
            raise ValueError("Dynamic Crop v3 loss contract is invalid")


class _OutputProxy:
    def __init__(self, output: StageEditDynamicCropModelOutputV3) -> None:
        self.output = output

    def __call__(self, _: StageEditTensorBatch) -> StageEditModelOutput:
        return self.output  # type: ignore[return-value]


def _masked_mean(values: Any, mask: Any) -> Any:
    weights = mask.astype(values.dtype)
    return (values * weights).sum() / mx.maximum(weights.sum(), 1.0)


def _selected_count_times(
    time_interval_logits: Any, count_targets: Any
) -> Any:
    all_times = dynamic_crop_v3_times_by_count(time_interval_logits)
    safe_targets = mx.maximum(count_targets - 1, 0)
    selector = mx.arange(DYNAMIC_CROP_V3_MAX_KEYFRAMES) == safe_targets[..., None]
    return (all_times * selector[..., None].astype(all_times.dtype)).sum(axis=-2)


def _original_observation_curve_loss(output: Any, supervision: Any) -> Any:
    """Gather only the true target source; score every original observed knot."""
    from .stage_edit_dynamic_crop_curve_loss import decoded_curve_risk

    indices = np.argwhere(np.asarray(supervision.track_mask, dtype=bool))
    if not len(indices):
        return output.dynamic_crop_values.sum() * 0.
    original = supervision.original_observed_keyframes or {}
    rows = []
    for index in indices:
        key = tuple(map(int, index))
        points = original.get(key)
        if not points:
            raise ValueError("Played-curve training requires original observed keyframes")
        duration = float(supervision.shot_duration_seconds[key].item())
        if not math.isfinite(duration) or duration <= 0.:
            raise ValueError("Original observation needs a finite positive shot duration")
        times = np.array([p.time_seconds for p in points], float) / duration
        if (not np.isfinite(times).all() or times[0] != 0.
                or np.any(np.diff(times) <= 0.) or times[-1] > 1. + 1e-6):
            raise ValueError("Original observation clock is outside the target shot")
        values = [[p.zoom, p.center_x, p.center_y, p.roll_degrees] for p in points]
        rows.append((times, values))
    width = max(len(row[0]) for row in rows)
    times = np.zeros((len(rows), width), np.float32)
    values = np.zeros((len(rows), width, 4), np.float32)
    mask = np.zeros((len(rows), width), bool)
    for i, (clock, points) in enumerate(rows):
        times[i, :len(clock)], values[i, :len(clock)], mask[i, :len(clock)] = clock, points, True
    b, t, s = (mx.array(indices[:, axis]) for axis in range(3))
    risk, _ = decoded_curve_risk(
        output.dynamic_crop_values[b, t, s],
        dynamic_crop_v3_times_by_count(output.dynamic_crop_time_interval_logits[b, t, s]),
        output.dynamic_crop_count_logits[b, t, s],
        mx.array(times), mx.array(values), mx.array(mask),
    )
    counts = mx.array(mask.sum(axis=-1), dtype=risk.dtype)
    return (risk * counts).sum() / counts.sum()


def stage_edit_dynamic_crop_v3_loss(
    model: StageEditTransformerDynamicCropV3,
    batch: StageEditTensorBatch,
    base_supervision: StageEditSupervisionBatch,
    dynamic_supervision: StageEditDynamicCropSupervisionBatchV3,
    contract: DynamicCropV3LossContract = DynamicCropV3LossContract(),
    *, output: StageEditDynamicCropModelOutputV3 | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Joint loss with source-slot masks and renderer-range normalization."""

    _require_mlx()
    output = model(batch) if output is None else output
    base_total, base_components = stage_edit_loss(
        _OutputProxy(output), batch, base_supervision, contract.base
    )
    normalized_output = renderer_range_normalized_values_v3(
        output.dynamic_crop_values
    )
    normalized_targets = renderer_range_normalized_values_v3(
        dynamic_supervision.values
    )
    value_values = nn.losses.smooth_l1_loss(
        normalized_output, normalized_targets, reduction="none"
    )
    coordinate_mask = mx.broadcast_to(
        dynamic_supervision.keyframe_mask[..., None], value_values.shape
    )
    value_loss = _masked_mean(value_values, coordinate_mask)

    safe_count_targets = mx.maximum(dynamic_supervision.count_targets - 1, 0)
    count_values = nn.losses.cross_entropy(
        output.dynamic_crop_count_logits,
        safe_count_targets,
        reduction="none",
    )
    count_loss = _masked_mean(count_values, dynamic_supervision.count_mask)

    selected_times = _selected_count_times(
        output.dynamic_crop_time_interval_logits,
        dynamic_supervision.count_targets,
    )
    time_values = nn.losses.smooth_l1_loss(
        selected_times,
        dynamic_supervision.normalized_times,
        reduction="none",
    )
    time_loss = _masked_mean(time_values, dynamic_supervision.keyframe_mask)

    motion_values = nn.losses.binary_cross_entropy(
        output.dynamic_crop_motion_logits,
        dynamic_supervision.motion_targets,
        with_logits=True,
        reduction="none",
    )
    motion_loss = _masked_mean(motion_values, dynamic_supervision.motion_mask)
    curve_loss = (_original_observation_curve_loss(output, dynamic_supervision)
                  if contract.decoded_curve_weight > 0. else mx.array(0., dtype=value_loss.dtype))
    dynamic_total = (
        contract.value_weight * value_loss
        + contract.count_weight * count_loss
        + contract.time_weight * time_loss
        + contract.motion_weight * motion_loss
        + contract.decoded_curve_weight * curve_loss
    )
    total = contract.base_loss_weight * base_total + dynamic_total
    components: dict[str, Any] = {
        **{f"base_{name}": value for name, value in base_components.items()},
        "dynamic_crop_values_normalized": value_loss,
        "dynamic_crop_count": count_loss,
        "dynamic_crop_times": time_loss,
        "dynamic_crop_motion": motion_loss,
        "dynamic_crop_decoded_curve": curve_loss,
        "dynamic_crop_total": dynamic_total,
        "total": total,
    }
    return total, components


def _safe_mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size else None


def _safe_quantile(values: np.ndarray, quantile: float) -> float | None:
    return float(np.quantile(values, quantile)) if values.size else None


def evaluate_stage_edit_dynamic_crop_v3(
    model: StageEditTransformerDynamicCropV3,
    batch: StageEditTensorBatch,
    base_supervision: StageEditSupervisionBatch,
    dynamic_supervision: StageEditDynamicCropSupervisionBatchV3,
    *,
    contract: DynamicCropV3LossContract = DynamicCropV3LossContract(),
) -> dict[str, Any]:
    """Evaluate source-exact trajectories and the production renderer path."""

    _require_mlx()
    model.eval()
    output = model(batch)
    total, components = stage_edit_dynamic_crop_v3_loss(
        model, batch, base_supervision, dynamic_supervision, contract, output=output
    )
    all_count_times = dynamic_crop_v3_times_by_count(
        output.dynamic_crop_time_interval_logits
    )
    normalized_values = renderer_range_normalized_values_v3(
        output.dynamic_crop_values
    )
    normalized_targets = renderer_range_normalized_values_v3(
        dynamic_supervision.values
    )
    mx.eval(
        total,
        *components.values(),
        output.dynamic_crop_values,
        output.dynamic_crop_count_logits,
        output.dynamic_crop_time_interval_logits,
        output.dynamic_crop_motion_logits,
        all_count_times,
        normalized_values,
        normalized_targets,
    )

    values = np.asarray(output.dynamic_crop_values, dtype=np.float64)
    targets = np.asarray(dynamic_supervision.values, dtype=np.float64)
    normalized = np.asarray(normalized_values, dtype=np.float64)
    normalized_target_values = np.asarray(normalized_targets, dtype=np.float64)
    keyframe_mask = np.asarray(dynamic_supervision.keyframe_mask, dtype=bool)
    track_mask = np.asarray(dynamic_supervision.track_mask, dtype=bool)
    count_mask = np.asarray(dynamic_supervision.count_mask, dtype=bool)
    target_counts = np.asarray(dynamic_supervision.count_targets, dtype=np.int64)
    count_logits = np.asarray(
        output.dynamic_crop_count_logits, dtype=np.float64
    )
    predicted_counts = np.argmax(count_logits, axis=-1).astype(np.int64) + 1
    source_indices = np.asarray(
        dynamic_supervision.target_source_indices, dtype=np.int64
    )

    coordinate_names = ("zoom", "center_x", "center_y", "roll_degrees")
    static_prior = np.asarray((_PRIOR_ZOOM, 0.5, 0.5, 0.0), dtype=np.float64)
    normalized_static_prior = np.asarray(
        renderer_range_normalized_values_v3(mx.array(static_prior)),
        dtype=np.float64,
    )
    coordinates: dict[str, Any] = {}
    for index, name in enumerate(coordinate_names):
        physical_residual = np.abs(
            values[..., index][keyframe_mask]
            - targets[..., index][keyframe_mask]
        )
        normalized_residual = np.abs(
            normalized[..., index][keyframe_mask]
            - normalized_target_values[..., index][keyframe_mask]
        )
        prior_residual = np.abs(
            static_prior[index] - targets[..., index][keyframe_mask]
        )
        prior_normalized_residual = np.abs(
            normalized_static_prior[index]
            - normalized_target_values[..., index][keyframe_mask]
        )
        model_mae = _safe_mean(physical_residual)
        prior_mae = _safe_mean(prior_residual)
        model_normalized_mae = _safe_mean(normalized_residual)
        prior_normalized_mae = _safe_mean(prior_normalized_residual)
        coordinates[name] = {
            "label_count": int(physical_residual.size),
            "mean_absolute_error": model_mae,
            "p95_absolute_error": _safe_quantile(physical_residual, 0.95),
            "normalized_mean_absolute_error": model_normalized_mae,
            "normalized_p95_absolute_error": _safe_quantile(
                normalized_residual, 0.95
            ),
            "static_prior_mean_absolute_error": prior_mae,
            "static_prior_normalized_mean_absolute_error": (
                prior_normalized_mae
            ),
            "mean_absolute_error_improvement_over_static_prior": (
                prior_mae - model_mae
                if prior_mae is not None and model_mae is not None
                else None
            ),
            "normalized_improvement_over_static_prior": (
                prior_normalized_mae - model_normalized_mae
                if prior_normalized_mae is not None
                and model_normalized_mae is not None
                else None
            ),
        }

    flattened_normalized_residual = np.abs(
        normalized[keyframe_mask] - normalized_target_values[keyframe_mask]
    ).reshape(-1)
    flattened_prior_normalized_residual = np.abs(
        normalized_static_prior[None, :]
        - normalized_target_values[keyframe_mask]
    ).reshape(-1)
    aggregate_model_normalized_mae = _safe_mean(flattened_normalized_residual)
    aggregate_prior_normalized_mae = _safe_mean(
        flattened_prior_normalized_residual
    )

    teacher_forced_times = np.zeros_like(
        np.asarray(dynamic_supervision.normalized_times, dtype=np.float64)
    )
    all_times = np.asarray(all_count_times, dtype=np.float64)
    for batch_index, token_index, source_index in np.argwhere(track_mask):
        count = int(target_counts[batch_index, token_index, source_index])
        teacher_forced_times[batch_index, token_index, source_index] = all_times[
            batch_index, token_index, source_index, count - 1
        ]
    target_times = np.asarray(
        dynamic_supervision.normalized_times, dtype=np.float64
    )
    time_residual = np.abs(teacher_forced_times[keyframe_mask] - target_times[keyframe_mask])

    selected_target_counts = target_counts[count_mask]
    selected_predicted_counts = predicted_counts[count_mask]
    count_accuracy = (
        float(np.mean(selected_predicted_counts == selected_target_counts))
        if selected_target_counts.size
        else None
    )
    count_confusion = np.zeros((8, 8), dtype=np.int64)
    for target_count, predicted_count in zip(
        selected_target_counts, selected_predicted_counts
    ):
        count_confusion[target_count - 1, predicted_count - 1] += 1

    motion_mask = np.asarray(dynamic_supervision.motion_mask, dtype=bool)
    motion_targets = np.asarray(
        dynamic_supervision.motion_targets, dtype=np.int64
    )
    motion_logits = np.asarray(
        output.dynamic_crop_motion_logits, dtype=np.float64
    )
    motion_predictions = (motion_logits >= 0.0).astype(np.int64)
    selected_motion_targets = motion_targets[motion_mask]
    selected_motion_predictions = motion_predictions[motion_mask]
    motion_accuracy = (
        float(np.mean(selected_motion_predictions == selected_motion_targets))
        if selected_motion_targets.size
        else None
    )
    motion_recalls: dict[str, float | None] = {}
    present_recalls: list[float] = []
    for class_index, class_name in ((0, "static"), (1, "dynamic")):
        selected = selected_motion_targets == class_index
        recall = (
            float(np.mean(selected_motion_predictions[selected] == class_index))
            if np.any(selected)
            else None
        )
        motion_recalls[class_name] = recall
        if recall is not None:
            present_recalls.append(recall)
    motion_balanced_accuracy = (
        float(np.mean(present_recalls)) if present_recalls else None
    )

    physical_deltas: list[np.ndarray] = []
    normalized_deltas: list[np.ndarray] = []
    target_physical_steps: list[np.ndarray] = []
    predicted_physical_steps: list[np.ndarray] = []
    target_normalized_steps: list[np.ndarray] = []
    predicted_normalized_steps: list[np.ndarray] = []
    target_ranges: list[np.ndarray] = []
    predicted_ranges: list[np.ndarray] = []
    target_normalized_ranges: list[np.ndarray] = []
    predicted_normalized_ranges: list[np.ndarray] = []
    for batch_index, token_index, source_index in np.argwhere(track_mask):
        count = int(target_counts[batch_index, token_index, source_index])
        predicted_track = values[batch_index, token_index, source_index, :count]
        target_track = targets[batch_index, token_index, source_index, :count]
        predicted_normalized_track = normalized[
            batch_index, token_index, source_index, :count
        ]
        target_normalized_track = normalized_target_values[
            batch_index, token_index, source_index, :count
        ]
        predicted_ranges.append(np.ptp(predicted_track, axis=0))
        target_ranges.append(np.ptp(target_track, axis=0))
        predicted_normalized_ranges.append(
            np.ptp(predicted_normalized_track, axis=0)
        )
        target_normalized_ranges.append(
            np.ptp(target_normalized_track, axis=0)
        )
        if count > 1:
            predicted_steps = np.diff(predicted_track, axis=0)
            target_steps = np.diff(target_track, axis=0)
            predicted_normalized_step = np.diff(
                predicted_normalized_track, axis=0
            )
            target_normalized_step = np.diff(target_normalized_track, axis=0)
            physical_deltas.append(np.abs(predicted_steps - target_steps))
            normalized_deltas.append(
                np.abs(predicted_normalized_step - target_normalized_step)
            )
            predicted_physical_steps.append(np.abs(predicted_steps))
            target_physical_steps.append(np.abs(target_steps))
            predicted_normalized_steps.append(np.abs(predicted_normalized_step))
            target_normalized_steps.append(np.abs(target_normalized_step))

    def stacked(values_to_stack: list[np.ndarray]) -> np.ndarray:
        return (
            np.concatenate(
                [value.reshape(-1, 4) for value in values_to_stack], axis=0
            )
            if values_to_stack
            else np.empty((0, 4), dtype=np.float64)
        )

    physical_delta_values = stacked(physical_deltas)
    normalized_delta_values = stacked(normalized_deltas)
    predicted_step_values = stacked(predicted_physical_steps)
    target_step_values = stacked(target_physical_steps)
    predicted_normalized_step_values = stacked(predicted_normalized_steps)
    target_normalized_step_values = stacked(target_normalized_steps)
    predicted_range_values = stacked(predicted_ranges)
    target_range_values = stacked(target_ranges)
    predicted_normalized_range_values = stacked(predicted_normalized_ranges)
    target_normalized_range_values = stacked(target_normalized_ranges)
    curve_metrics: dict[str, Any] = {}
    for index, name in enumerate(coordinate_names):
        curve_metrics[name] = {
            "delta_label_count": int(physical_delta_values.shape[0]),
            "delta_mean_absolute_error": _safe_mean(
                physical_delta_values[:, index]
            ),
            "normalized_delta_mean_absolute_error": _safe_mean(
                normalized_delta_values[:, index]
            ),
            "predicted_mean_absolute_step": _safe_mean(
                predicted_step_values[:, index]
            ),
            "target_mean_absolute_step": _safe_mean(target_step_values[:, index]),
            "predicted_normalized_mean_absolute_step": _safe_mean(
                predicted_normalized_step_values[:, index]
            ),
            "target_normalized_mean_absolute_step": _safe_mean(
                target_normalized_step_values[:, index]
            ),
            "predicted_mean_range": _safe_mean(predicted_range_values[:, index]),
            "target_mean_range": _safe_mean(target_range_values[:, index]),
            "mean_absolute_range_error": _safe_mean(
                np.abs(
                    predicted_range_values[:, index]
                    - target_range_values[:, index]
                )
            ),
            "predicted_normalized_mean_range": _safe_mean(
                predicted_normalized_range_values[:, index]
            ),
            "target_normalized_mean_range": _safe_mean(
                target_normalized_range_values[:, index]
            ),
        }

    from .stage_edit_dynamic_crop_semimarkov_adapter_v3 import (  # local cycle
        decode_dynamic_crop_model_source_v3,
    )
    from .stage_edit_decoded_curve_metrics import (
        decoded_observation_error, summarize_decoded_observations,
    )

    durations = np.asarray(
        dynamic_supervision.shot_duration_seconds, dtype=np.float64
    )
    renderer_attempts = 0
    renderer_eligible = 0
    renderer_projection_count = 0
    renderer_rejections: dict[str, int] = {}
    decoded_source_ids: list[str] = []
    decoded_observation_rows=[]
    decoded_source_mismatches=0
    observation_metric_rejections={}
    for batch_index, token_index, source_index in np.argwhere(track_mask):
        renderer_attempts += 1
        decoded = decode_dynamic_crop_model_source_v3(
            output,
            batch,
            batch_index=int(batch_index),
            token_index=int(token_index),
            source_index=int(source_index),
            shot_duration_seconds=float(
                durations[batch_index, token_index, source_index]
            ),
            config=model.dynamic_crop_config,
        )
        if decoded.eligible:
            expected_source_id=batch.source_ids[int(batch_index)][int(token_index)][int(source_index)]
            if decoded.source_id!=expected_source_id:
                decoded_source_mismatches+=1
                renderer_rejections['decoded_source_identity_mismatch']=renderer_rejections.get('decoded_source_identity_mismatch',0)+1
                continue
            renderer_eligible += 1
            renderer_projection_count += decoded.minimum_safe_zoom_projection_count
            assert decoded.source_id is not None
            decoded_source_ids.append(decoded.source_id)
            observed=(dynamic_supervision.original_observed_keyframes or {}).get(
                (int(batch_index),int(token_index),int(source_index)))
            if observed:
                try:
                    decoded_observation_rows.append(decoded_observation_error(decoded,observed,
                        expected_source_id=expected_source_id))
                except ValueError as error:
                    reason=str(error)
                    observation_metric_rejections[reason]=observation_metric_rejections.get(reason,0)+1
            else:
                observation_metric_rejections['original_observed_keyframes_missing']=observation_metric_rejections.get(
                    'original_observed_keyframes_missing',0)+1
        else:
            for reason in decoded.rejection_reasons:
                renderer_rejections[reason] = renderer_rejections.get(reason, 0) + 1

    source_slot_mismatch_count = 0
    for batch_index, token_index in np.argwhere(source_indices >= 0):
        source_index = int(source_indices[batch_index, token_index])
        if not track_mask[batch_index, token_index, source_index]:
            source_slot_mismatch_count += 1

    decoded_observation_metrics=summarize_decoded_observations(decoded_observation_rows,
        expected_tracks=int(np.sum(track_mask)),source_mismatches=decoded_source_mismatches)
    decoded_observation_metrics['rejections']=observation_metric_rejections

    return {
        "schema_version": DYNAMIC_CROP_V3_SCHEMA_VERSION,
        "target_contract": DYNAMIC_CROP_V3_TARGET_CONTRACT,
        "model_contract": DYNAMIC_CROP_V3_MODEL_CONTRACT,
        "output_contract": DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
        "loss_contract": DYNAMIC_CROP_V3_LOSS_CONTRACT,
        "source_conditioning": model.dynamic_crop_config.source_conditioning,
        "affine_head_representation": model.dynamic_crop_config.head_representation,
        "time_parameterization": DYNAMIC_CROP_V3_TIME_PARAMETERIZATION,
        "value_parameterization": DYNAMIC_CROP_V3_VALUE_PARAMETERIZATION,
        "total_loss": float(np.asarray(total).item()),
        "loss_components": {
            name: float(np.asarray(value).item())
            for name, value in components.items()
        },
        "trajectory": {
            "track_label_count": int(np.sum(track_mask)),
            "keyframe_label_count": int(np.sum(keyframe_mask)),
            "source_slot_supervision_shape": list(track_mask.shape),
            "target_source_slot_mismatch_count": source_slot_mismatch_count,
            "decoded_source_ids": decoded_source_ids,
            "coordinates": decoded_observation_metrics['coordinates'],
            "decoded_observed_time": decoded_observation_metrics,
            "teacher_forced_value_diagnostics": dict(coordinates=coordinates,
                aggregate_renderer_range_normalized_mae=aggregate_model_normalized_mae,
                aggregate_static_prior_renderer_range_normalized_mae=aggregate_prior_normalized_mae,
                curve_delta_and_range=curve_metrics),
            "aggregate_renderer_range_normalized_mae": (
                decoded_observation_metrics['normalized_mae']
            ),
            "aggregate_static_prior_renderer_range_normalized_mae": (
                decoded_observation_metrics['static_prior_normalized_mae']
            ),
            "aggregate_normalized_improvement_over_static_prior": (
                decoded_observation_metrics['improvement_over_static_prior']
            ),
            "teacher_forced_count_normalized_time_mae": _safe_mean(time_residual),
            "count": {
                "label_count": int(selected_target_counts.size),
                "exact_accuracy": count_accuracy,
                "target_histogram_1_to_8": [
                    int(np.sum(selected_target_counts == count))
                    for count in range(1, 9)
                ],
                "predicted_histogram_1_to_8": [
                    int(np.sum(selected_predicted_counts == count))
                    for count in range(1, 9)
                ],
                "confusion_target_rows_predicted_columns": count_confusion.tolist(),
            },
            "motion": {
                "label_count": int(selected_motion_targets.size),
                "accuracy": motion_accuracy,
                "balanced_accuracy": motion_balanced_accuracy,
                "recall": motion_recalls,
            },
            "curve_delta_and_range": decoded_observation_metrics['curve_delta_and_range'],
            "renderer_decode": {
                "attempt_count": renderer_attempts,
                "eligible_count": renderer_eligible,
                "eligible_rate": (
                    renderer_eligible / renderer_attempts
                    if renderer_attempts
                    else None
                ),
                "minimum_safe_zoom_projection_count": renderer_projection_count,
                "rejection_reasons": dict(sorted(renderer_rejections.items())),
            },
        },
        "legacy_v2_checkpoint_or_output_used": False,
        "legacy_static_crop_head_used_for_trajectory": False,
        "production_activation_authorized": False,
    }


@dataclass(frozen=True, slots=True)
class StageEditDynamicCropCheckpointMetadataV3:
    format_version: int
    checkpoint_id: str
    created_at: str
    training_steps: int
    weights_trained: bool
    training_data_id: str | None
    token_schema: StageEditTokenSchema
    base_model_config: StageEditTransformerConfig
    dynamic_crop_config: DynamicCropHeadV3Config
    weight_filename: str = DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
    weight_sha256: str | None = None
    model_contract: str = DYNAMIC_CROP_V3_MODEL_CONTRACT
    target_contract: str = DYNAMIC_CROP_V3_TARGET_CONTRACT
    output_contract: str = DYNAMIC_CROP_V3_OUTPUT_CONTRACT
    loss_contract: str = DYNAMIC_CROP_V3_LOSS_CONTRACT
    source_conditioning: str = DYNAMIC_CROP_V3_SOURCE_CONDITIONING
    time_parameterization: str = DYNAMIC_CROP_V3_TIME_PARAMETERIZATION
    value_parameterization: str = DYNAMIC_CROP_V3_VALUE_PARAMETERIZATION
    affine_head_representation: str = DYNAMIC_CROP_V3_AFFINE_HEAD_REPRESENTATION
    production_activation_authorized: bool = False

    def __post_init__(self) -> None:
        if self.format_version != self.dynamic_crop_config.checkpoint_format_version:
            raise ValueError("unsupported Dynamic Crop v3 checkpoint format")
        if not str(self.checkpoint_id).strip():
            raise ValueError("Dynamic Crop v3 checkpoint id cannot be blank")
        if isinstance(self.training_steps, bool) or self.training_steps < 0:
            raise ValueError("Dynamic Crop v3 training_steps cannot be negative")
        if self.weights_trained and self.training_steps == 0:
            raise ValueError("trained Dynamic Crop v3 weights need at least one step")
        if self.base_model_config.token_schema_fingerprint != self.token_schema.fingerprint:
            raise ValueError("Dynamic Crop v3 schema and base config differ")
        if Path(self.weight_filename).name != self.weight_filename:
            raise ValueError("Dynamic Crop v3 weight filename must not contain a path")
        required = (
            (self.model_contract, DYNAMIC_CROP_V3_MODEL_CONTRACT),
            (self.target_contract, DYNAMIC_CROP_V3_TARGET_CONTRACT),
            (self.output_contract, DYNAMIC_CROP_V3_OUTPUT_CONTRACT),
            (self.loss_contract, DYNAMIC_CROP_V3_LOSS_CONTRACT),
            (self.source_conditioning, self.dynamic_crop_config.source_conditioning),
            (self.time_parameterization, DYNAMIC_CROP_V3_TIME_PARAMETERIZATION),
            (self.value_parameterization, DYNAMIC_CROP_V3_VALUE_PARAMETERIZATION),
            (self.affine_head_representation, self.dynamic_crop_config.head_representation),
        )
        if any(value != expected for value, expected in required):
            raise ValueError("Dynamic Crop v3 checkpoint contract is invalid")
        if self.production_activation_authorized is not False:
            raise ValueError("Dynamic Crop v3 core checkpoint cannot authorize production")

    @property
    def contract_fingerprint_sha256(self) -> str:
        return _sha256_json(
            {
                "format_version": self.format_version,
                "model_contract": self.model_contract,
                "target_contract": self.target_contract,
                "output_contract": self.output_contract,
                "loss_contract": self.loss_contract,
                "source_conditioning": self.source_conditioning,
                "time_parameterization": self.time_parameterization,
                "value_parameterization": self.value_parameterization,
                "affine_head_representation": self.affine_head_representation,
                "token_schema_fingerprint": self.token_schema.fingerprint,
                "base_model_config": self.base_model_config.to_dict(),
                "dynamic_crop_config": self.dynamic_crop_config.to_dict(),
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
            "source_conditioning": self.source_conditioning,
            "time_parameterization": self.time_parameterization,
            "value_parameterization": self.value_parameterization,
            "affine_head_representation": self.affine_head_representation,
            "contract_fingerprint_sha256": self.contract_fingerprint_sha256,
            "production_activation_authorized": False,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "StageEditDynamicCropCheckpointMetadataV3":
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
            dynamic_crop_config=DynamicCropHeadV3Config(
                **payload["dynamic_crop_config"]
            ),
            weight_filename=str(payload.get("weight_filename", "")),
            weight_sha256=(
                str(payload["weight_sha256"])
                if payload.get("weight_sha256") is not None
                else None
            ),
            model_contract=str(payload.get("model_contract", "")),
            target_contract=str(payload.get("target_contract", "")),
            output_contract=str(payload.get("output_contract", "")),
            loss_contract=str(payload.get("loss_contract", "")),
            source_conditioning=str(payload.get("source_conditioning", "")),
            time_parameterization=str(payload.get("time_parameterization", "")),
            value_parameterization=str(payload.get("value_parameterization", "")),
            affine_head_representation=str(payload.get("affine_head_representation", "")),
            production_activation_authorized=bool(
                payload.get("production_activation_authorized", False)
            ),
        )
        if payload.get("contract_fingerprint_sha256") != value.contract_fingerprint_sha256:
            raise ValueError("Dynamic Crop v3 checkpoint contract fingerprint mismatch")
        return value


def new_stage_edit_dynamic_crop_checkpoint_metadata_v3(
    checkpoint_id: str,
    schema: StageEditTokenSchema,
    base_config: StageEditTransformerConfig,
    *,
    training_steps: int,
    weights_trained: bool,
    training_data_id: str | None = None,
    dynamic_crop_config: DynamicCropHeadV3Config = DynamicCropHeadV3Config(),
) -> StageEditDynamicCropCheckpointMetadataV3:
    return StageEditDynamicCropCheckpointMetadataV3(
        format_version=dynamic_crop_config.checkpoint_format_version,
        checkpoint_id=checkpoint_id,
        created_at=datetime.now(UTC).isoformat(),
        training_steps=training_steps,
        weights_trained=weights_trained,
        training_data_id=training_data_id,
        token_schema=schema,
        base_model_config=base_config,
        dynamic_crop_config=dynamic_crop_config,
        source_conditioning=dynamic_crop_config.source_conditioning,
        affine_head_representation=dynamic_crop_config.head_representation,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_stage_edit_dynamic_crop_checkpoint_v3(
    directory: Path,
    model: StageEditTransformerDynamicCropV3,
    metadata: StageEditDynamicCropCheckpointMetadataV3,
) -> StageEditDynamicCropCheckpointMetadataV3:
    _require_mlx()
    if model.base_config != metadata.base_model_config:
        raise ValueError("Dynamic Crop v3 model/base metadata mismatch")
    if model.dynamic_crop_config != metadata.dynamic_crop_config:
        raise ValueError("Dynamic Crop v3 model/head metadata mismatch")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    weight_path = destination / metadata.weight_filename
    from mlx.utils import tree_flatten

    weights = dict(tree_flatten(model.parameters()))
    for name in ("dynamic_crop_context_head", "dynamic_crop_source_head"):
        head = getattr(model, name)
        if not isinstance(head, _FixedCenteredLinear):
            raise ValueError("Dynamic Crop v3 checkpoint requires fixed-centered affine heads")
        head.set_input_center(head.input_center)
        weights[f"{name}.input_center"] = head.input_center
    mx.save_safetensors(str(weight_path), weights)
    mx.eval(model.parameters())
    saved = replace(metadata, weight_sha256=_sha256_file(weight_path))
    (destination / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME).write_text(
        json.dumps(saved.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return saved


def load_stage_edit_dynamic_crop_checkpoint_v3(
    directory: Path,
    *,
    expected_schema: StageEditTokenSchema | None = None,
    require_trained: bool = True,
) -> tuple[StageEditTransformerDynamicCropV3, StageEditDynamicCropCheckpointMetadataV3]:
    _require_mlx()
    source = Path(directory)
    metadata_path = source / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
    if not metadata_path.is_file():
        if (source / DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME).is_file():
            raise ValueError(
                "Dynamic Crop v2 checkpoint is incompatible with source-conditioned v3"
            )
        raise FileNotFoundError(metadata_path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Dynamic Crop v3 checkpoint metadata root must be an object")
    metadata = StageEditDynamicCropCheckpointMetadataV3.from_dict(payload)
    if expected_schema is not None and expected_schema.fingerprint != metadata.token_schema.fingerprint:
        raise ValueError("Dynamic Crop v3 checkpoint schema differs from expected schema")
    if require_trained and not (metadata.weights_trained and metadata.training_steps > 0):
        raise ValueError("Dynamic Crop v3 checkpoint does not declare trained weights")
    weight_path = source / metadata.weight_filename
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    if metadata.weight_sha256 is None or _sha256_file(weight_path) != metadata.weight_sha256:
        raise ValueError("Dynamic Crop v3 checkpoint weight digest is missing or invalid")
    weights = mx.load(str(weight_path))
    centers = {}
    for name in ("dynamic_crop_context_head", "dynamic_crop_source_head"):
        key = f"{name}.input_center"
        if key not in weights:
            raise ValueError("Dynamic Crop v3 checkpoint fixed affine center is missing")
        center = np.array(weights.pop(key), dtype=np.float32, copy=True)
        if (center.shape != (metadata.base_model_config.model_dimension,)
                or not np.isfinite(center).all()):
            raise ValueError("Dynamic Crop v3 checkpoint fixed affine center is invalid")
        centers[name] = center
    model = StageEditTransformerDynamicCropV3(
        metadata.base_model_config, metadata.dynamic_crop_config
    )
    model.load_weights(list(weights.items()), strict=True)
    for name, value in centers.items():
        getattr(model, name).set_input_center(value)
    mx.eval(model.parameters())
    return model, metadata


__all__ = [
    "DYNAMIC_CROP_V3_CHECKPOINT_FORMAT_VERSION",
    "DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME",
    "DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME",
    "DYNAMIC_CROP_V3_LOSS_CONTRACT",
    "DYNAMIC_CROP_V3_MAX_KEYFRAMES",
    "DYNAMIC_CROP_V3_MODEL_CONTRACT",
    "DYNAMIC_CROP_V3_OUTPUT_CONTRACT",
    "DYNAMIC_CROP_V3_RAW_OUTPUT_DIMENSION",
    "DYNAMIC_CROP_V3_SCHEMA_VERSION",
    "DYNAMIC_CROP_V3_SOURCE_CONDITIONING",
    "DYNAMIC_CROP_V3_TARGET_CONTRACT",
    "DYNAMIC_CROP_V3_TIME_BLOCK_SLICES",
    "DYNAMIC_CROP_V3_TIME_LOGIT_COUNT",
    "DYNAMIC_CROP_V3_TIME_PARAMETERIZATION",
    "DYNAMIC_CROP_V3_TIME_SIMPLEX_EPSILON",
    "DYNAMIC_CROP_V3_VALUE_PARAMETERIZATION",
    "DYNAMIC_CROP_V3_AFFINE_HEAD_REPRESENTATION",
    "DYNAMIC_CROP_V3_INTERACTION_SOURCE_CONDITIONING",
    "DYNAMIC_CROP_V3_INTERACTION_HEAD_REPRESENTATION",
    "DYNAMIC_CROP_V3_INTERACTION_CHECKPOINT_FORMAT_VERSION",
    "DynamicCropHeadV3Config",
    "DynamicCropV3LossContract",
    "StageEditDynamicCropCheckpointMetadataV3",
    "StageEditDynamicCropModelOutputV3",
    "StageEditDynamicCropSupervisionBatchV3",
    "StageEditDynamicCropTargetSequenceV3",
    "StageEditTransformerDynamicCropV3",
    "dynamic_crop_v3_times_by_count",
    "evaluate_stage_edit_dynamic_crop_v3",
    "load_stage_edit_dynamic_crop_checkpoint_v3",
    "new_stage_edit_dynamic_crop_checkpoint_metadata_v3",
    "renderer_range_normalized_values_v3",
    "renderer_safe_values_from_raw_v3",
    "save_stage_edit_dynamic_crop_checkpoint_v3",
    "stage_edit_dynamic_crop_v3_loss",
    "stage_edit_dynamic_crop_v3_parameter_count",
    "tensorize_stage_edit_dynamic_crop_supervision_v3",
]
