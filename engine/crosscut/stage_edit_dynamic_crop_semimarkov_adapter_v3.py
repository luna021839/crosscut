"""Source-exact Dynamic Crop v3 adapter for semi-Markov inference.

The v3 model predicts one trajectory for every candidate source.  This
adapter preserves that source axis until the beam asks for one exact slot,
checks the model mask against the tensorized candidate/hard-gate masks, then
decodes the categorical keyframe count and its private gap-simplex block.
Malformed shapes, masked sources, and renderer-invalid curves all fail closed.
No v2 checkpoint or v2 output contract is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .render import EditorialTransformKeyframe, EditorialTransformTrack
from .stage_edit_dynamic_crop_head_v2 import DynamicCropHeadV2Config
from .stage_edit_dynamic_crop_head_v3 import (
    DYNAMIC_CROP_V3_MAX_KEYFRAMES,
    DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
    DYNAMIC_CROP_V3_TIME_BLOCK_SLICES,
    DYNAMIC_CROP_V3_TIME_LOGIT_COUNT,
    DYNAMIC_CROP_V3_TIME_SIMPLEX_EPSILON,
    DynamicCropHeadV3Config,
    StageEditDynamicCropModelOutputV3,
)
from .stage_edit_dynamic_crop_semimarkov_adapter import (
    decode_dynamic_crop_prediction_v2,
)
from .stage_edit_transformer import StageEditTensorBatch


DYNAMIC_CROP_V3_SEMIMARKOV_ADAPTER_CONTRACT = (
    "stage_edit_dynamic_crop_source_exact_semimarkov_renderer_adapter_v3"
)
DYNAMIC_CROP_V3_PREDICTION_CONTRACT = (
    "source_slot_count_conditional_dynamic_crop_prediction_v3"
)

_DYNAMIC_CROP_V3_DECODE_KEYS = {
    "adapter_contract",
    "source_id",
    "source_index",
    "eligible",
    "loss_masked",
    "track",
    "predicted_keyframe_count",
    "applied_keyframe_count",
    "minimum_safe_zoom_projection_count",
    "rejection_reasons",
    "motion_logit",
    "shot_duration_seconds",
}
_DYNAMIC_CROP_V3_TRACK_KEYS = {"reason", "keyframes"}
_DYNAMIC_CROP_V3_KEYFRAME_KEYS = {
    "time_seconds",
    "zoom",
    "center_x",
    "center_y",
    "roll_degrees",
}


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _source_id(value: object, *, allow_none: bool = False) -> str | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError("dynamic crop v3 source id cannot be blank")
    result = str(value).strip()
    if not result:
        raise ValueError("dynamic crop v3 source id cannot be blank")
    return result


def _normalized_times_for_count(
    time_interval_logits: tuple[float, ...], count: int
) -> tuple[float, ...]:
    if not 1 <= count <= DYNAMIC_CROP_V3_MAX_KEYFRAMES:
        raise ValueError("dynamic crop v3 count must lie in 1..8")
    if len(time_interval_logits) != DYNAMIC_CROP_V3_TIME_LOGIT_COUNT:
        raise ValueError("dynamic crop v3 needs exactly 35 time logits")
    if count == 1:
        return (0.0,) * DYNAMIC_CROP_V3_MAX_KEYFRAMES

    start, stop = DYNAMIC_CROP_V3_TIME_BLOCK_SLICES[count]
    block = np.asarray(time_interval_logits[start:stop], dtype=np.float64)
    maximum = float(np.max(block))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        shifted = np.clip(block - maximum, -700.0, 0.0)
        probabilities = np.exp(shifted)
    denominator = float(probabilities.sum())
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("dynamic crop v3 time simplex is not finite")
    probabilities /= denominator
    probabilities = (
        (1.0 - DYNAMIC_CROP_V3_TIME_SIMPLEX_EPSILON) * probabilities
        + DYNAMIC_CROP_V3_TIME_SIMPLEX_EPSILON / count
    )
    observed = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(probabilities[:-1]))
    )
    if (
        observed.shape != (count,)
        or observed[0] != 0.0
        or np.any(np.diff(observed) <= 0.0)
        or observed[-1] >= 1.0
    ):
        raise ValueError("dynamic crop v3 time simplex did not produce valid times")
    return tuple(float(value) for value in observed) + (0.0,) * (
        DYNAMIC_CROP_V3_MAX_KEYFRAMES - count
    )


@dataclass(frozen=True, slots=True)
class DynamicCropTrackDecodeV3:
    """All-or-nothing decode result for one exact source candidate slot."""

    source_id: str | None
    source_index: int
    eligible: bool
    loss_masked: bool
    track: EditorialTransformTrack | None
    predicted_keyframe_count: int
    applied_keyframe_count: int
    minimum_safe_zoom_projection_count: int
    rejection_reasons: tuple[str, ...]
    # Retained independently from track presence.  The renderer-side motion
    # authority combines this exact logit with its LOSO-bound threshold and
    # sampled physical range; a framing track alone never means "motion".
    motion_logit: float | None = None
    # Exact semi-Markov duration used to scale the normalized keyframe clock.
    # Final visual repair may reparameterize that clock; renderer provenance
    # must prove the target/original duration ratio rather than loosely accept
    # a track that merely looks similar.
    shot_duration_seconds: float | None = None
    adapter_contract: str = DYNAMIC_CROP_V3_SEMIMARKOV_ADAPTER_CONTRACT

    def __post_init__(self) -> None:
        source_id = _source_id(self.source_id, allow_none=not self.eligible)
        if (
            isinstance(self.source_index, bool)
            or not isinstance(self.source_index, int)
            or self.source_index < 0
        ):
            raise ValueError("dynamic crop v3 source index must be non-negative")
        if not 0 <= self.predicted_keyframe_count <= DYNAMIC_CROP_V3_MAX_KEYFRAMES:
            raise ValueError("dynamic crop v3 predicted count is invalid")
        if self.adapter_contract != DYNAMIC_CROP_V3_SEMIMARKOV_ADAPTER_CONTRACT:
            raise ValueError("dynamic crop v3 adapter contract mismatch")
        if self.motion_logit is not None:
            object.__setattr__(
                self,
                "motion_logit",
                _finite_float(self.motion_logit, "motion logit"),
            )
        if self.shot_duration_seconds is not None:
            duration = _finite_float(
                self.shot_duration_seconds, "shot duration seconds"
            )
            if duration <= 0.0:
                raise ValueError("dynamic crop v3 shot duration must be positive")
            object.__setattr__(self, "shot_duration_seconds", duration)
        if self.eligible:
            if self.loss_masked or self.track is None or source_id is None:
                raise ValueError("eligible dynamic crop v3 decode needs a source track")
            if self.rejection_reasons:
                raise ValueError("eligible dynamic crop v3 decode cannot be rejected")
            if self.applied_keyframe_count != len(self.track.keyframes):
                raise ValueError("dynamic crop v3 applied count differs from track")
            if self.applied_keyframe_count != self.predicted_keyframe_count:
                raise ValueError("dynamic crop v3 decode changed its predicted count")
        else:
            if not self.loss_masked or self.track is not None:
                raise ValueError("rejected dynamic crop v3 decode must fail closed")
            if self.applied_keyframe_count != 0:
                raise ValueError("rejected dynamic crop v3 decode applied keyframes")
            if not self.rejection_reasons:
                raise ValueError("rejected dynamic crop v3 decode needs a reason")
        object.__setattr__(self, "source_id", source_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize every field needed to replay renderer motion authority."""

        track = self.track
        return {
            "adapter_contract": self.adapter_contract,
            "source_id": self.source_id,
            "source_index": self.source_index,
            "eligible": self.eligible,
            "loss_masked": self.loss_masked,
            "track": (
                None
                if track is None
                else {
                    "reason": track.reason,
                    "keyframes": [
                        {
                            "time_seconds": item.time_seconds,
                            "zoom": item.zoom,
                            "center_x": item.center_x,
                            "center_y": item.center_y,
                            "roll_degrees": item.roll_degrees,
                        }
                        for item in track.keyframes
                    ],
                }
            ),
            "predicted_keyframe_count": self.predicted_keyframe_count,
            "applied_keyframe_count": self.applied_keyframe_count,
            "minimum_safe_zoom_projection_count": (
                self.minimum_safe_zoom_projection_count
            ),
            "rejection_reasons": list(self.rejection_reasons),
            "motion_logit": self.motion_logit,
            "shot_duration_seconds": self.shot_duration_seconds,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DynamicCropTrackDecodeV3":
        """Strictly restore one exact v3 decode; partial legacy payloads fail."""

        if not isinstance(payload, Mapping) or set(payload) != _DYNAMIC_CROP_V3_DECODE_KEYS:
            raise ValueError("dynamic crop v3 decode keys are not exact")
        material = dict(payload)
        for name in ("eligible", "loss_masked"):
            if not isinstance(material[name], bool):
                raise ValueError(f"dynamic crop v3 decode {name} must be boolean")
        for name in (
            "source_index",
            "predicted_keyframe_count",
            "applied_keyframe_count",
            "minimum_safe_zoom_projection_count",
        ):
            if isinstance(material[name], bool) or not isinstance(material[name], int):
                raise ValueError(f"dynamic crop v3 decode {name} must be an integer")
        source_id = material["source_id"]
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(
                "strict dynamic crop v3 decode needs an exact source identity"
            )
        if not isinstance(material["adapter_contract"], str):
            raise ValueError("dynamic crop v3 decode adapter contract must be a string")
        rejection_reasons = material["rejection_reasons"]
        if not isinstance(rejection_reasons, list) or any(
            not isinstance(reason, str) or not reason.strip()
            for reason in rejection_reasons
        ):
            raise ValueError("dynamic crop v3 decode rejection reasons are invalid")
        raw_track = material["track"]
        track: EditorialTransformTrack | None
        if raw_track is None:
            track = None
        else:
            if not isinstance(raw_track, Mapping) or set(raw_track) != _DYNAMIC_CROP_V3_TRACK_KEYS:
                raise ValueError("dynamic crop v3 decode track keys are not exact")
            reason = raw_track["reason"]
            raw_keyframes = raw_track["keyframes"]
            if not isinstance(reason, str) or not isinstance(raw_keyframes, list):
                raise ValueError("dynamic crop v3 decode track is malformed")
            keyframes: list[EditorialTransformKeyframe] = []
            for index, raw_keyframe in enumerate(raw_keyframes):
                if (
                    not isinstance(raw_keyframe, Mapping)
                    or set(raw_keyframe) != _DYNAMIC_CROP_V3_KEYFRAME_KEYS
                ):
                    raise ValueError(
                        f"dynamic crop v3 keyframe {index} keys are not exact"
                    )
                keyframes.append(
                    EditorialTransformKeyframe(
                        time_seconds=_finite_float(
                            raw_keyframe["time_seconds"],
                            f"keyframe {index} time_seconds",
                        ),
                        zoom=_finite_float(
                            raw_keyframe["zoom"], f"keyframe {index} zoom"
                        ),
                        center_x=_finite_float(
                            raw_keyframe["center_x"],
                            f"keyframe {index} center_x",
                        ),
                        center_y=_finite_float(
                            raw_keyframe["center_y"],
                            f"keyframe {index} center_y",
                        ),
                        roll_degrees=_finite_float(
                            raw_keyframe["roll_degrees"],
                            f"keyframe {index} roll_degrees",
                        ),
                    )
                )
            track = EditorialTransformTrack(tuple(keyframes), reason=reason)
        motion_logit = material["motion_logit"]
        shot_duration = material["shot_duration_seconds"]
        if motion_logit is None or shot_duration is None:
            raise ValueError(
                "strict dynamic crop v3 decode needs motion logit and shot duration"
            )
        return cls(
            source_id=source_id,
            source_index=material["source_index"],
            eligible=material["eligible"],
            loss_masked=material["loss_masked"],
            track=track,
            predicted_keyframe_count=material["predicted_keyframe_count"],
            applied_keyframe_count=material["applied_keyframe_count"],
            minimum_safe_zoom_projection_count=material[
                "minimum_safe_zoom_projection_count"
            ],
            rejection_reasons=tuple(rejection_reasons),
            motion_logit=_finite_float(motion_logit, "motion logit"),
            shot_duration_seconds=_finite_float(
                shot_duration, "shot duration seconds"
            ),
            adapter_contract=material["adapter_contract"],
        )


@dataclass(frozen=True, slots=True)
class DynamicCropHeadPredictionV3:
    """One exact source slot before a semi-Markov duration is chosen."""

    source_id: str
    source_index: int
    values: tuple[tuple[float, float, float, float], ...]
    count_logits: tuple[float, ...]
    time_interval_logits: tuple[float, ...]
    motion_logit: float
    config: DynamicCropHeadV3Config = DynamicCropHeadV3Config()
    output_contract: str = DYNAMIC_CROP_V3_OUTPUT_CONTRACT
    prediction_contract: str = DYNAMIC_CROP_V3_PREDICTION_CONTRACT

    def __post_init__(self) -> None:
        source_id = _source_id(self.source_id)
        if (
            isinstance(self.source_index, bool)
            or not isinstance(self.source_index, int)
            or self.source_index < 0
        ):
            raise ValueError("dynamic crop v3 source index must be non-negative")
        if not isinstance(self.config, DynamicCropHeadV3Config):
            raise TypeError("dynamic crop v3 prediction config is invalid")
        if self.output_contract != DYNAMIC_CROP_V3_OUTPUT_CONTRACT:
            raise ValueError("dynamic crop v3 output contract mismatch")
        if self.prediction_contract != DYNAMIC_CROP_V3_PREDICTION_CONTRACT:
            raise ValueError("dynamic crop v3 prediction contract mismatch")
        values = tuple(
            tuple(
                _finite_float(component, f"transform {index} component {offset}")
                for offset, component in enumerate(item)
            )
            for index, item in enumerate(self.values)
        )
        counts = tuple(
            _finite_float(value, f"count logit {index}")
            for index, value in enumerate(self.count_logits)
        )
        times = tuple(
            _finite_float(value, f"time logit {index}")
            for index, value in enumerate(self.time_interval_logits)
        )
        if len(values) != DYNAMIC_CROP_V3_MAX_KEYFRAMES or any(
            len(item) != 4 for item in values
        ):
            raise ValueError("dynamic crop v3 prediction needs exactly K=8 transforms")
        if len(counts) != DYNAMIC_CROP_V3_MAX_KEYFRAMES:
            raise ValueError("dynamic crop v3 prediction needs 8 count logits")
        if len(times) != DYNAMIC_CROP_V3_TIME_LOGIT_COUNT:
            raise ValueError("dynamic crop v3 prediction needs 35 time logits")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "count_logits", counts)
        object.__setattr__(self, "time_interval_logits", times)
        object.__setattr__(
            self, "motion_logit", _finite_float(self.motion_logit, "motion logit")
        )

    @property
    def predicted_keyframe_count(self) -> int:
        return int(np.argmax(np.asarray(self.count_logits))) + 1

    @property
    def normalized_times(self) -> tuple[float, ...]:
        return _normalized_times_for_count(
            self.time_interval_logits, self.predicted_keyframe_count
        )

    def decode(self, shot_duration_seconds: float) -> DynamicCropTrackDecodeV3:
        count = self.predicted_keyframe_count
        legacy = decode_dynamic_crop_prediction_v2(
            normalized_times=self.normalized_times,
            values=self.values,
            presence_logits=(20.0,) * count
            + (-20.0,) * (DYNAMIC_CROP_V3_MAX_KEYFRAMES - count),
            shot_duration_seconds=shot_duration_seconds,
            config=DynamicCropHeadV2Config(),
        )
        if not legacy.eligible:
            return DynamicCropTrackDecodeV3(
                source_id=self.source_id,
                source_index=self.source_index,
                eligible=False,
                loss_masked=True,
                track=None,
                predicted_keyframe_count=count,
                applied_keyframe_count=0,
                minimum_safe_zoom_projection_count=0,
                rejection_reasons=legacy.rejection_reasons,
                motion_logit=self.motion_logit,
                shot_duration_seconds=shot_duration_seconds,
            )
        assert legacy.track is not None
        track = EditorialTransformTrack(
            legacy.track.keyframes, reason="stage_edit_dynamic_crop_head_v3"
        )
        return DynamicCropTrackDecodeV3(
            source_id=self.source_id,
            source_index=self.source_index,
            eligible=True,
            loss_masked=False,
            track=track,
            predicted_keyframe_count=count,
            applied_keyframe_count=len(track.keyframes),
            minimum_safe_zoom_projection_count=(
                legacy.minimum_safe_zoom_projection_count
            ),
            rejection_reasons=(),
            motion_logit=self.motion_logit,
            shot_duration_seconds=shot_duration_seconds,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_index": self.source_index,
            "values": [list(item) for item in self.values],
            "count_logits": list(self.count_logits),
            "time_interval_logits": list(self.time_interval_logits),
            "motion_logit": self.motion_logit,
            "predicted_keyframe_count": self.predicted_keyframe_count,
            "normalized_times": list(self.normalized_times),
            "config": self.config.to_dict(),
            "output_contract": self.output_contract,
            "prediction_contract": self.prediction_contract,
        }


def _validate_source_output_contract_v3(
    output: StageEditDynamicCropModelOutputV3,
    batch: StageEditTensorBatch,
) -> tuple[int, int, int]:
    if not isinstance(output, StageEditDynamicCropModelOutputV3):
        raise TypeError("dynamic crop v3 adapter requires a v3 model output")
    if output.output_contract != DYNAMIC_CROP_V3_OUTPUT_CONTRACT:
        raise ValueError("dynamic crop v3 model output contract mismatch")
    expected_mask = np.logical_and(
        np.asarray(batch.source_candidate_mask, dtype=bool),
        np.asarray(batch.hard_source_gate_mask, dtype=bool),
    )
    shape = expected_mask.shape
    if len(shape) != 3:
        raise ValueError("dynamic crop v3 batch source mask must be rank three")
    output_mask = np.asarray(output.effective_source_mask, dtype=bool)
    if output_mask.shape != shape or not np.array_equal(output_mask, expected_mask):
        raise ValueError("dynamic crop v3 output source mask differs from its batch")
    expected_shapes = (
        (output.dynamic_crop_values, shape + (DYNAMIC_CROP_V3_MAX_KEYFRAMES, 4)),
        (output.dynamic_crop_count_logits, shape + (DYNAMIC_CROP_V3_MAX_KEYFRAMES,)),
        (
            output.dynamic_crop_time_interval_logits,
            shape + (DYNAMIC_CROP_V3_TIME_LOGIT_COUNT,),
        ),
        (output.dynamic_crop_motion_logits, shape),
    )
    if any(tuple(value.shape) != expected for value, expected in expected_shapes):
        raise ValueError("dynamic crop v3 source output shape mismatch")
    return tuple(int(value) for value in shape)  # type: ignore[return-value]


def dynamic_crop_prediction_from_model_source_v3(
    output: StageEditDynamicCropModelOutputV3,
    batch: StageEditTensorBatch,
    *,
    batch_index: int,
    token_index: int,
    source_index: int,
    config: DynamicCropHeadV3Config = DynamicCropHeadV3Config(),
) -> DynamicCropHeadPredictionV3:
    """Capture one source slot, rejecting unavailable or mismatched slots."""

    batch_size, token_count, source_count = _validate_source_output_contract_v3(
        output, batch
    )
    for value, limit, name in (
        (batch_index, batch_size, "batch"),
        (token_index, token_count, "token"),
        (source_index, source_count, "source"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < limit:
            raise IndexError(f"dynamic crop v3 {name} index is outside its output")
    return _prediction_from_validated_model_source_v3(
        output,
        batch,
        batch_index=batch_index,
        token_index=token_index,
        source_index=source_index,
        config=config,
    )


def _prediction_from_validated_model_source_v3(
    output: StageEditDynamicCropModelOutputV3,
    batch: StageEditTensorBatch,
    *,
    batch_index: int,
    token_index: int,
    source_index: int,
    config: DynamicCropHeadV3Config,
) -> DynamicCropHeadPredictionV3:
    """Capture a slot after one caller-owned full output-contract check."""

    if not bool(np.asarray(output.effective_source_mask)[batch_index, token_index, source_index]):
        raise ValueError("dynamic crop v3 source is unavailable or hard-gated")
    source_id = batch.source_ids[batch_index][token_index][source_index]
    if source_id is None:
        raise ValueError("dynamic crop v3 available source slot has no identity")
    values = np.asarray(
        output.dynamic_crop_values[batch_index, token_index, source_index]
    ).tolist()
    count_logits = np.asarray(
        output.dynamic_crop_count_logits[batch_index, token_index, source_index]
    ).tolist()
    time_logits = np.asarray(
        output.dynamic_crop_time_interval_logits[
            batch_index, token_index, source_index
        ]
    ).tolist()
    motion_logit = np.asarray(
        output.dynamic_crop_motion_logits[batch_index, token_index, source_index]
    ).item()
    return DynamicCropHeadPredictionV3(
        source_id=source_id,
        source_index=source_index,
        values=tuple(tuple(item) for item in values),
        count_logits=tuple(count_logits),
        time_interval_logits=tuple(time_logits),
        motion_logit=motion_logit,
        config=config,
    )


def dynamic_crop_predictions_from_model_token_v3(
    output: StageEditDynamicCropModelOutputV3,
    batch: StageEditTensorBatch,
    *,
    batch_index: int,
    token_index: int,
    config: DynamicCropHeadV3Config = DynamicCropHeadV3Config(),
) -> tuple[DynamicCropHeadPredictionV3 | None, ...]:
    """Return a slot-preserving prediction tuple for all candidate sources."""

    batch_size, token_count, source_count = _validate_source_output_contract_v3(
        output, batch
    )
    if (
        isinstance(batch_index, bool)
        or not isinstance(batch_index, int)
        or isinstance(token_index, bool)
        or not isinstance(token_index, int)
        or not 0 <= batch_index < batch_size
        or not 0 <= token_index < token_count
    ):
        raise IndexError("dynamic crop v3 token is outside its output")
    effective = np.asarray(output.effective_source_mask, dtype=bool)
    return tuple(
        _prediction_from_validated_model_source_v3(
            output,
            batch,
            batch_index=batch_index,
            token_index=token_index,
            source_index=source_index,
            config=config,
        )
        if effective[batch_index, token_index, source_index]
        else None
        for source_index in range(source_count)
    )


def decode_dynamic_crop_model_source_v3(
    output: StageEditDynamicCropModelOutputV3,
    batch: StageEditTensorBatch,
    *,
    batch_index: int,
    token_index: int,
    source_index: int,
    shot_duration_seconds: float,
    config: DynamicCropHeadV3Config = DynamicCropHeadV3Config(),
) -> DynamicCropTrackDecodeV3:
    """Decode one exact source for one concrete semi-Markov duration."""

    try:
        prediction = dynamic_crop_prediction_from_model_source_v3(
            output,
            batch,
            batch_index=batch_index,
            token_index=token_index,
            source_index=source_index,
            config=config,
        )
    except (IndexError, TypeError, ValueError, OverflowError) as exc:
        source_id: str | None = None
        try:
            source_id = batch.source_ids[batch_index][token_index][source_index]
        except (IndexError, TypeError):
            pass
        return DynamicCropTrackDecodeV3(
            source_id=source_id,
            source_index=(
                max(0, source_index)
                if isinstance(source_index, int) and not isinstance(source_index, bool)
                else 0
            ),
            eligible=False,
            loss_masked=True,
            track=None,
            predicted_keyframe_count=0,
            applied_keyframe_count=0,
            minimum_safe_zoom_projection_count=0,
            rejection_reasons=(f"dynamic crop v3 model source is unavailable: {exc}",),
        )
    return prediction.decode(shot_duration_seconds)


__all__ = [
    "DYNAMIC_CROP_V3_PREDICTION_CONTRACT",
    "DYNAMIC_CROP_V3_SEMIMARKOV_ADAPTER_CONTRACT",
    "DynamicCropHeadPredictionV3",
    "DynamicCropTrackDecodeV3",
    "decode_dynamic_crop_model_source_v3",
    "dynamic_crop_prediction_from_model_source_v3",
    "dynamic_crop_predictions_from_model_token_v3",
]
