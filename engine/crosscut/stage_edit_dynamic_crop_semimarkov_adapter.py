"""Fail-closed Dynamic Crop Head v2 adapter for semi-Markov inference.

This module is the exact connection point between the versioned K=8 model
head and the variable-duration semi-Markov search.  A model token is first
captured as an immutable, JSON-serializable prediction.  The beam then decodes
that same prediction against each concrete shot-duration hypothesis.  It never
clamps pan, roll, time, or an excessive zoom;
it may only *raise* zoom to the minimum value that makes the exact predicted
camera state renderer-safe.  Any other invalid sample, an unsafe interpolated
track, or an absent first keyframe returns a loss-masked rejection with no
identity substitute.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from .render import EditorialTransformKeyframe, EditorialTransformTrack
from .stage_edit_dynamic_crop_head_v2 import (
    DYNAMIC_CROP_V2_MAX_KEYFRAMES,
    DYNAMIC_CROP_V2_OUTPUT_CONTRACT,
    DynamicCropHeadV2Config,
    DynamicCropTrajectoryTargetV2,
    StageEditDynamicCropModelOutputV2,
)


DYNAMIC_CROP_V2_SEMIMARKOV_ADAPTER_CONTRACT = (
    "stage_edit_dynamic_crop_semimarkov_renderer_adapter_v2"
)


@dataclass(frozen=True, slots=True)
class DynamicCropTrackDecodeV2:
    """One token's all-or-nothing renderer-track decode result."""

    eligible: bool
    loss_masked: bool
    track: EditorialTransformTrack | None
    requested_keyframe_count: int
    applied_keyframe_count: int
    minimum_safe_zoom_projection_count: int
    rejection_reasons: tuple[str, ...]
    adapter_contract: str = DYNAMIC_CROP_V2_SEMIMARKOV_ADAPTER_CONTRACT

    def __post_init__(self) -> None:
        if self.eligible:
            if self.loss_masked or self.track is None:
                raise ValueError("eligible dynamic crop decode needs a track")
            if self.rejection_reasons:
                raise ValueError("eligible dynamic crop decode cannot be rejected")
            if self.applied_keyframe_count != len(self.track.keyframes):
                raise ValueError("applied dynamic crop keyframe count is inconsistent")
        else:
            if not self.loss_masked or self.track is not None:
                raise ValueError("rejected dynamic crop decode must fail closed")
            if self.applied_keyframe_count != 0:
                raise ValueError("rejected dynamic crop decode cannot apply keyframes")
            if not self.rejection_reasons:
                raise ValueError("rejected dynamic crop decode needs a reason")


@dataclass(frozen=True, slots=True)
class DynamicCropHeadPredictionV2:
    """One model token's raw K=8 prediction before shot-duration decoding.

    Keeping normalized time here is essential: a semi-Markov token can be
    expanded into several duration hypotheses, so converting to seconds before
    the beam chooses a duration would silently train and render different
    contracts.
    """

    normalized_times: tuple[float, ...]
    values: tuple[tuple[float, float, float, float], ...]
    presence_logits: tuple[float, ...]
    motion_logit: float
    config: DynamicCropHeadV2Config = DynamicCropHeadV2Config()
    output_contract: str = DYNAMIC_CROP_V2_OUTPUT_CONTRACT

    def __post_init__(self) -> None:
        if self.output_contract != DYNAMIC_CROP_V2_OUTPUT_CONTRACT:
            raise ValueError("dynamic crop prediction output contract mismatch")
        if not isinstance(self.config, DynamicCropHeadV2Config):
            raise TypeError("dynamic crop prediction config is invalid")
        expected = self.config.max_keyframes
        times = tuple(
            _finite_float(value, f"normalized time {index}")
            for index, value in enumerate(self.normalized_times)
        )
        values = tuple(
            tuple(
                _finite_float(component, f"transform {index} component {offset}")
                for offset, component in enumerate(transform)
            )
            for index, transform in enumerate(self.values)
        )
        logits = tuple(
            _finite_float(value, f"presence logit {index}")
            for index, value in enumerate(self.presence_logits)
        )
        if len(times) != expected or len(values) != expected or len(logits) != expected:
            raise ValueError("dynamic crop prediction must contain exactly K=8 samples")
        if any(len(item) != 4 for item in values):
            raise ValueError("dynamic crop prediction transforms need four values")
        object.__setattr__(self, "normalized_times", times)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "presence_logits", logits)
        object.__setattr__(
            self, "motion_logit", _finite_float(self.motion_logit, "motion logit")
        )

    def decode(self, shot_duration_seconds: float) -> DynamicCropTrackDecodeV2:
        return decode_dynamic_crop_prediction_v2(
            normalized_times=self.normalized_times,
            values=self.values,
            presence_logits=self.presence_logits,
            shot_duration_seconds=shot_duration_seconds,
            config=self.config,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalized_times": list(self.normalized_times),
            "values": [list(item) for item in self.values],
            "presence_logits": list(self.presence_logits),
            "motion_logit": self.motion_logit,
            "config": self.config.to_dict(),
            "output_contract": self.output_contract,
        }


def _rejected(reason: str, *, requested: int = 0) -> DynamicCropTrackDecodeV2:
    return DynamicCropTrackDecodeV2(
        eligible=False,
        loss_masked=True,
        track=None,
        requested_keyframe_count=requested,
        applied_keyframe_count=0,
        minimum_safe_zoom_projection_count=0,
        rejection_reasons=(reason,),
    )


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_non_zoom_geometry(
    *, center_x: float, center_y: float, roll_degrees: float
) -> None:
    if not 0.0 <= center_x <= 1.0 or not 0.0 <= center_y <= 1.0:
        raise ValueError("predicted center is outside normalized coordinates")
    if math.hypot(center_x - 0.5, center_y - 0.5) > 0.040 + 1e-9:
        raise ValueError("predicted pan exceeds renderer four-percent cap")
    if abs(roll_degrees) > 1.60:
        raise ValueError("predicted roll exceeds renderer bound")


def _keyframe_is_safe(
    *,
    time_seconds: float,
    zoom: float,
    center_x: float,
    center_y: float,
    roll_degrees: float,
) -> bool:
    try:
        EditorialTransformKeyframe(
            time_seconds,
            zoom,
            center_x,
            center_y,
            roll_degrees,
        )
    except ValueError:
        return False
    return True


def _minimum_safe_zoom_projection(
    *,
    time_seconds: float,
    predicted_zoom: float,
    center_x: float,
    center_y: float,
    roll_degrees: float,
) -> tuple[EditorialTransformKeyframe, bool]:
    """Keep an exact safe zoom or raise it to the geometric minimum.

    Decreasing an excessive zoom is a clamp and is forbidden.  Pan and roll
    are prevalidated and are never altered.  A non-positive zoom is treated as
    out-of-domain rather than silently projected into the camera manifold.
    """

    _validate_non_zoom_geometry(
        center_x=center_x,
        center_y=center_y,
        roll_degrees=roll_degrees,
    )
    if predicted_zoom <= 0.0:
        raise ValueError("predicted zoom must be positive")
    if predicted_zoom > 1.18:
        raise ValueError("predicted zoom exceeds renderer maximum")
    if _keyframe_is_safe(
        time_seconds=time_seconds,
        zoom=predicted_zoom,
        center_x=center_x,
        center_y=center_y,
        roll_degrees=roll_degrees,
    ):
        return (
            EditorialTransformKeyframe(
                time_seconds,
                predicted_zoom,
                center_x,
                center_y,
                roll_degrees,
            ),
            False,
        )

    lower = max(1.0, predicted_zoom)
    upper = 1.18
    if not _keyframe_is_safe(
        time_seconds=time_seconds,
        zoom=upper,
        center_x=center_x,
        center_y=center_y,
        roll_degrees=roll_degrees,
    ):
        raise ValueError("no renderer-safe zoom exists for predicted pan and roll")
    # Monotone bisection finds the smallest safe zoom.  The final upper bound
    # is used unchanged; no pan/roll substitution or maximum-zoom clamp occurs.
    for _ in range(64):
        midpoint = (lower + upper) / 2.0
        if _keyframe_is_safe(
            time_seconds=time_seconds,
            zoom=midpoint,
            center_x=center_x,
            center_y=center_y,
            roll_degrees=roll_degrees,
        ):
            upper = midpoint
        else:
            lower = midpoint
    return (
        EditorialTransformKeyframe(
            time_seconds,
            upper,
            center_x,
            center_y,
            roll_degrees,
        ),
        True,
    )


def decode_dynamic_crop_prediction_v2(
    *,
    normalized_times: Sequence[float],
    values: Sequence[Sequence[float]],
    presence_logits: Sequence[float],
    shot_duration_seconds: float,
    config: DynamicCropHeadV2Config = DynamicCropHeadV2Config(),
) -> DynamicCropTrackDecodeV2:
    """Decode one K=8 prediction into an exact renderer track or reject it.

    Sample-domain errors are represented by ``loss_masked=True`` rather than
    exceptions so one bad token cannot crash a semi-Markov job.  Invalid
    caller/model contracts still fail closed through the same result.
    """

    try:
        duration = _finite_float(shot_duration_seconds, "shot duration")
        if duration <= 0.0:
            raise ValueError("shot duration must be positive")
        if len(normalized_times) != config.max_keyframes:
            raise ValueError("predicted times must have exactly K=8 values")
        if len(values) != config.max_keyframes:
            raise ValueError("predicted transforms must have exactly K=8 values")
        if len(presence_logits) != config.max_keyframes:
            raise ValueError("presence logits must have exactly K=8 values")
        times = tuple(
            _finite_float(value, f"normalized time {index}")
            for index, value in enumerate(normalized_times)
        )
        transforms = tuple(tuple(value) for value in values)
        if any(len(item) != 4 for item in transforms):
            raise ValueError("each predicted transform must have four values")
        logits = tuple(
            _finite_float(value, f"presence logit {index}")
            for index, value in enumerate(presence_logits)
        )
        logit_threshold = math.log(
            config.presence_threshold / (1.0 - config.presence_threshold)
        )
        selected = tuple(
            index for index, logit in enumerate(logits) if logit >= logit_threshold
        )
        if not selected:
            return _rejected("no_predicted_keyframe_present")
        if selected != tuple(range(len(selected))):
            raise ValueError(
                "predicted keyframe presence must be one contiguous prefix"
            )
        selected_times = tuple(times[index] for index in selected)
        if any(not 0.0 <= value <= 1.0 for value in selected_times):
            raise ValueError("selected normalized time is outside [0, 1]")
        if any(
            right <= left
            for left, right in zip(selected_times, selected_times[1:])
        ):
            raise ValueError("selected normalized times are not strictly increasing")
        # EditorialTransformTrack itself requires the first sample at segment
        # zero.  Preserve the predicted value; never shift/synthesize a start.
        first_seconds = selected_times[0] * duration
        if abs(first_seconds) > 1e-9:
            raise ValueError("predicted track does not start at segment time zero")

        keyframes: list[EditorialTransformKeyframe] = []
        projection_count = 0
        for index in selected:
            zoom, center_x, center_y, roll_degrees = (
                _finite_float(component, f"transform {index} component {offset}")
                for offset, component in enumerate(transforms[index])
            )
            keyframe, projected = _minimum_safe_zoom_projection(
                time_seconds=times[index] * duration,
                predicted_zoom=zoom,
                center_x=center_x,
                center_y=center_y,
                roll_degrees=roll_degrees,
            )
            projection_count += int(projected)
            keyframes.append(keyframe)
        # This constructor rechecks the exact renderer's 15 interpolation
        # samples.  An unsafe curve rejects the entire token; no partial track.
        track = EditorialTransformTrack(
            tuple(keyframes), reason="stage_edit_dynamic_crop_head_v2"
        )
    except (TypeError, ValueError, OverflowError) as exc:
        requested = 0
        try:
            requested = len(selected)  # type: ignore[possibly-undefined]
        except (NameError, TypeError):
            pass
        return _rejected(str(exc), requested=requested)

    return DynamicCropTrackDecodeV2(
        eligible=True,
        loss_masked=False,
        track=track,
        requested_keyframe_count=len(selected),
        applied_keyframe_count=len(keyframes),
        minimum_safe_zoom_projection_count=projection_count,
        rejection_reasons=(),
    )


def decode_dynamic_crop_model_token_v2(
    output: StageEditDynamicCropModelOutputV2,
    *,
    batch_index: int,
    token_index: int,
    shot_duration_seconds: float,
    config: DynamicCropHeadV2Config = DynamicCropHeadV2Config(),
) -> DynamicCropTrackDecodeV2:
    """Decode one model token for a concrete semi-Markov duration."""

    try:
        prediction = dynamic_crop_prediction_from_model_token_v2(
            output,
            batch_index=batch_index,
            token_index=token_index,
            config=config,
        )
    except (IndexError, TypeError, ValueError, OverflowError) as exc:
        return _rejected(f"dynamic crop model token is unavailable: {exc}")
    return prediction.decode(shot_duration_seconds)


def dynamic_crop_prediction_from_model_token_v2(
    output: StageEditDynamicCropModelOutputV2,
    *,
    batch_index: int,
    token_index: int,
    config: DynamicCropHeadV2Config = DynamicCropHeadV2Config(),
) -> DynamicCropHeadPredictionV2:
    """Capture a raw model token without choosing a shot duration.

    Malformed model contracts raise here because they indicate a broken
    checkpoint/inference boundary, not an ordinary renderer-domain rejection.
    The latter remains an all-or-nothing :class:`DynamicCropTrackDecodeV2`
    result once the beam supplies a duration.
    """

    if output.output_contract != DYNAMIC_CROP_V2_OUTPUT_CONTRACT:
        raise ValueError("dynamic crop model output contract mismatch")
    times = np.asarray(
        output.dynamic_crop_normalized_times[batch_index, token_index]
    ).tolist()
    values = np.asarray(
        output.dynamic_crop_values[batch_index, token_index]
    ).tolist()
    logits = np.asarray(
        output.dynamic_crop_presence_logits[batch_index, token_index]
    ).tolist()
    motion_logit = np.asarray(
        output.dynamic_crop_motion_logits[batch_index, token_index]
    ).item()
    return DynamicCropHeadPredictionV2(
        normalized_times=tuple(times),
        values=tuple(tuple(item) for item in values),
        presence_logits=tuple(logits),
        motion_logit=motion_logit,
        config=config,
    )


def decode_dynamic_crop_target_v2(
    target: DynamicCropTrajectoryTargetV2,
    *,
    config: DynamicCropHeadV2Config = DynamicCropHeadV2Config(),
) -> DynamicCropTrackDecodeV2:
    """Exercise the identical K=8 renderer adapter on a validated artifact."""

    count = len(target.keyframes)
    times = [
        item.time_seconds / target.shot_duration_seconds
        for item in target.keyframes
    ]
    values = [
        (item.zoom, item.center_x, item.center_y, item.roll_degrees)
        for item in target.keyframes
    ]
    times.extend([0.0] * (config.max_keyframes - count))
    values.extend([(0.0, 0.0, 0.0, 0.0)] * (config.max_keyframes - count))
    logits = [20.0] * count + [-20.0] * (config.max_keyframes - count)
    return decode_dynamic_crop_prediction_v2(
        normalized_times=times,
        values=values,
        presence_logits=logits,
        shot_duration_seconds=target.shot_duration_seconds,
        config=config,
    )


__all__ = [
    "DYNAMIC_CROP_V2_SEMIMARKOV_ADAPTER_CONTRACT",
    "DynamicCropHeadPredictionV2",
    "DynamicCropTrackDecodeV2",
    "decode_dynamic_crop_model_token_v2",
    "decode_dynamic_crop_prediction_v2",
    "decode_dynamic_crop_target_v2",
    "dynamic_crop_prediction_from_model_token_v2",
]
