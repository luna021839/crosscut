"""Renderer-verified trajectory simplification for Dynamic Crop v3.

Dynamic Crop v2 preserved every successful reverse-match observation.  That
made the padded keyframe count describe detector coverage rather than camera
curve complexity: even visually static shots commonly carried four to six
nearly identical samples.  V3 simplifies only existing observations.  It
never invents, clamps, or moves a keyframe, and every proposed removal is
validated by the exact editorial renderer track.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Sequence

from .render import EditorialTransformKeyframe, EditorialTransformTrack
from .dynamic_crop_source_frame_binding import DynamicCropSourceFrameBinding
from .stage_edit_dynamic_crop_head_v2 import DynamicCropTrajectoryTargetV2


DYNAMIC_CROP_V3_SIMPLIFICATION_CONTRACT = (
    "observed_only_renderer_verified_curve_complexity_simplification_v1"
)


@dataclass(frozen=True, slots=True)
class DynamicCropV3SimplificationConfig:
    """Renderer-range-normalized error limits for observed-point removal."""

    static_maximum_normalized_error: float = 0.06
    dynamic_maximum_normalized_error: float = 0.035
    zoom_range: float = 0.18
    center_radius: float = 0.04
    roll_range_degrees: float = 1.60
    contract: str = DYNAMIC_CROP_V3_SIMPLIFICATION_CONTRACT

    def __post_init__(self) -> None:
        for name in (
            "static_maximum_normalized_error",
            "dynamic_maximum_normalized_error",
            "zoom_range",
            "center_radius",
            "roll_range_degrees",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.contract != DYNAMIC_CROP_V3_SIMPLIFICATION_CONTRACT:
            raise ValueError("dynamic crop v3 simplification contract differs")

    @property
    def fingerprint(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class SimplifiedDynamicCropTrajectoryV3:
    """One v2 target expressed as an observed-only v3 curve-complexity target."""

    shot_index: int
    source_id: str
    shot_duration_seconds: float
    keyframes: tuple[EditorialTransformKeyframe, ...]
    motion_label: str
    motion_label_loss_masked: bool
    artifact_content_sha256: str
    build_fingerprint_sha256: str
    original_keyframe_count: int
    retained_original_indices: tuple[int, ...]
    maximum_normalized_reconstruction_error: float
    simplification_config_fingerprint: str
    contract: str = DYNAMIC_CROP_V3_SIMPLIFICATION_CONTRACT
    source_frame_binding: DynamicCropSourceFrameBinding | None = None

    def __post_init__(self) -> None:
        if not self.keyframes:
            raise ValueError("simplified dynamic crop trajectory cannot be empty")
        if self.source_frame_binding is not None:
            binding = self.source_frame_binding
            if not isinstance(binding, DynamicCropSourceFrameBinding):
                raise TypeError("simplified crop source frame binding is invalid")
            if binding.source_id != self.source_id or len(binding.observations) != self.original_keyframe_count:
                raise ValueError("simplified crop must retain all original source observations")
        if self.original_keyframe_count < len(self.keyframes):
            raise ValueError("simplification cannot add keyframes")
        if len(self.retained_original_indices) != len(self.keyframes):
            raise ValueError("retained indices do not align with keyframes")
        if tuple(sorted(set(self.retained_original_indices))) != (
            self.retained_original_indices
        ):
            raise ValueError("retained keyframe indices must be unique and ordered")
        if self.retained_original_indices[0] != 0:
            raise ValueError("simplification must retain the observed time-zero sample")
        if self.retained_original_indices[-1] >= self.original_keyframe_count:
            raise ValueError("retained keyframe index is outside the original track")
        if self.source_frame_binding is not None and any(
            self.source_frame_binding.observations[index].time_within_shot_seconds
            != keyframe.time_seconds
            for index, keyframe in zip(self.retained_original_indices, self.keyframes)
        ):
            raise ValueError("simplified crop keyframes differ from their source-frame observations")
        if (
            not math.isfinite(self.maximum_normalized_reconstruction_error)
            or self.maximum_normalized_reconstruction_error < 0.0
        ):
            raise ValueError("reconstruction error must be finite and non-negative")
        if self.contract != DYNAMIC_CROP_V3_SIMPLIFICATION_CONTRACT:
            raise ValueError("simplified trajectory contract differs")
        # This constructor is the same interpolation-safety authority used by
        # render planning.  A one-keyframe track is a valid static camera.
        EditorialTransformTrack(
            self.keyframes,
            reason="dynamic_crop_v3_observed_only_simplification",
        )

    @property
    def removed_original_indices(self) -> tuple[int, ...]:
        retained = set(self.retained_original_indices)
        return tuple(
            index
            for index in range(self.original_keyframe_count)
            if index not in retained
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "source_frame_binding": (
                self.source_frame_binding.to_dict()
                if self.source_frame_binding is not None else None
            ),
            "shot_index": self.shot_index,
            "source_id": self.source_id,
            "shot_duration_seconds": self.shot_duration_seconds,
            "motion_label": self.motion_label,
            "motion_label_loss_masked": self.motion_label_loss_masked,
            "artifact_content_sha256": self.artifact_content_sha256,
            "build_fingerprint_sha256": self.build_fingerprint_sha256,
            "original_keyframe_count": self.original_keyframe_count,
            "simplified_keyframe_count": len(self.keyframes),
            "retained_original_indices": list(self.retained_original_indices),
            "removed_original_indices": list(self.removed_original_indices),
            "maximum_normalized_reconstruction_error": (
                self.maximum_normalized_reconstruction_error
            ),
            "simplification_config_fingerprint": (
                self.simplification_config_fingerprint
            ),
            "keyframes": [
                {
                    "time_seconds": item.time_seconds,
                    "zoom": item.zoom,
                    "center_x": item.center_x,
                    "center_y": item.center_y,
                    "roll_degrees": item.roll_degrees,
                }
                for item in self.keyframes
            ],
        }


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _interpolate(
    left: EditorialTransformKeyframe,
    right: EditorialTransformKeyframe,
    time_seconds: float,
) -> tuple[float, float, float, float]:
    if right.time_seconds <= left.time_seconds:
        raise ValueError("dynamic crop interpolation interval is not positive")
    alpha = (time_seconds - left.time_seconds) / (
        right.time_seconds - left.time_seconds
    )
    # Do not construct an EditorialTransformKeyframe here.  An interpolation
    # candidate can be renderer-unsafe even when both observed endpoints are
    # safe; that is exactly what the separate whole-track review must catch.
    return (
        left.zoom + (right.zoom - left.zoom) * alpha,
        left.center_x + (right.center_x - left.center_x) * alpha,
        left.center_y + (right.center_y - left.center_y) * alpha,
        (
            left.roll_degrees
            + (right.roll_degrees - left.roll_degrees) * alpha
        ),
    )


def _normalized_error(
    observed: EditorialTransformKeyframe,
    reconstructed: Sequence[float],
    config: DynamicCropV3SimplificationConfig,
) -> float:
    if len(reconstructed) != 4:
        raise ValueError("reconstructed dynamic crop sample needs four values")
    zoom, center_x, center_y, roll_degrees = (
        float(value) for value in reconstructed
    )
    return max(
        abs(observed.zoom - zoom) / config.zoom_range,
        math.hypot(
            observed.center_x - center_x,
            observed.center_y - center_y,
        )
        / config.center_radius,
        abs(observed.roll_degrees - roll_degrees)
        / config.roll_range_degrees,
    )


def _reconstruction_errors(
    original: Sequence[EditorialTransformKeyframe],
    retained_indices: Sequence[int],
    config: DynamicCropV3SimplificationConfig,
) -> tuple[float, ...]:
    retained = tuple(retained_indices)
    if not retained or retained[0] != 0:
        raise ValueError("reconstruction needs the observed time-zero sample")
    errors = [0.0] * len(original)
    for original_index, observed in enumerate(original):
        if original_index in retained:
            continue
        right_position = next(
            (position for position, value in enumerate(retained) if value > original_index),
            None,
        )
        if right_position is None:
            # A static one-knot approximation holds the first observed camera
            # state; no synthetic end sample is introduced.
            reconstructed = (
                original[retained[-1]].zoom,
                original[retained[-1]].center_x,
                original[retained[-1]].center_y,
                original[retained[-1]].roll_degrees,
            )
        else:
            left_index = retained[right_position - 1]
            right_index = retained[right_position]
            reconstructed = _interpolate(
                original[left_index], original[right_index], observed.time_seconds
            )
        errors[original_index] = _normalized_error(observed, reconstructed, config)
    return tuple(errors)


def _track_is_renderer_safe(
    original: Sequence[EditorialTransformKeyframe],
    retained_indices: Sequence[int],
) -> bool:
    try:
        EditorialTransformTrack(
            tuple(original[index] for index in retained_indices),
            reason="dynamic_crop_v3_candidate_simplification_review",
        )
    except ValueError:
        return False
    return True


def simplify_dynamic_crop_trajectory_v3(
    target: DynamicCropTrajectoryTargetV2,
    *,
    config: DynamicCropV3SimplificationConfig = (
        DynamicCropV3SimplificationConfig()
    ),
) -> SimplifiedDynamicCropTrajectoryV3:
    """Remove redundant observed knots without synthesizing renderer values."""

    if not isinstance(target, DynamicCropTrajectoryTargetV2):
        raise TypeError("v3 simplification requires a v2 renderer trajectory")
    original = tuple(target.keyframes)
    retained = list(range(len(original)))
    limit = (
        config.static_maximum_normalized_error
        if target.motion_label == "static"
        else config.dynamic_maximum_normalized_error
    )

    # Static evidence may collapse to its real time-zero observation.  Dynamic
    # evidence keeps its real final observation so its direction and range
    # cannot disappear merely because intermediate samples are redundant.
    if target.motion_label == "static" and len(retained) > 1:
        static_indices = (0,)
        static_errors = _reconstruction_errors(original, static_indices, config)
        if max(static_errors, default=0.0) <= limit and _track_is_renderer_safe(
            original, static_indices
        ):
            retained = [0]

    if len(retained) > 2:
        mandatory = {0}
        if target.motion_label == "dynamic":
            mandatory.add(len(original) - 1)
        while True:
            proposals: list[tuple[float, int, list[int]]] = []
            for index in tuple(retained):
                if index in mandatory:
                    continue
                candidate = [value for value in retained if value != index]
                if not candidate or candidate[0] != 0:
                    continue
                errors = _reconstruction_errors(original, candidate, config)
                maximum = max(errors, default=0.0)
                if maximum > limit or not _track_is_renderer_safe(original, candidate):
                    continue
                proposals.append((maximum, index, candidate))
            if not proposals:
                break
            _error, _index, retained = min(
                proposals, key=lambda item: (item[0], item[1])
            )

    errors = _reconstruction_errors(original, retained, config)
    simplified = tuple(original[index] for index in retained)
    return SimplifiedDynamicCropTrajectoryV3(
        shot_index=target.shot_index,
        source_id=target.source_id,
        shot_duration_seconds=target.shot_duration_seconds,
        keyframes=simplified,
        motion_label=target.motion_label,
        motion_label_loss_masked=target.motion_label_loss_masked,
        artifact_content_sha256=target.artifact_content_sha256,
        build_fingerprint_sha256=target.build_fingerprint_sha256,
        original_keyframe_count=len(original),
        retained_original_indices=tuple(retained),
        maximum_normalized_reconstruction_error=max(errors, default=0.0),
        simplification_config_fingerprint=config.fingerprint,
        source_frame_binding=target.source_frame_binding,
    )


__all__ = [
    "DYNAMIC_CROP_V3_SIMPLIFICATION_CONTRACT",
    "DynamicCropV3SimplificationConfig",
    "SimplifiedDynamicCropTrajectoryV3",
    "simplify_dynamic_crop_trajectory_v3",
]
