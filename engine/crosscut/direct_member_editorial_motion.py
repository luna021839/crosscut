"""Dense, non-identifying editorial-camera review for official member fancams.

Automatic member sources are already authenticated as official target-member
9:16 fancams by metadata and media contracts.  This layer still never names a
face from pixels: it only uses a continuously dominant face box around a
selected hard cut, then re-decodes every transformed output frame before a
bounded scale/position match is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Sequence

import cv2

from .direct_member_visual import DirectMemberVisualSelection
from .direct_member_transform import (
    DirectMemberBoundaryTransformRepair,
    DirectMemberSourceFrameGeometry,
    direct_member_source_frame_geometry,
    map_box_to_portrait_fill,
)
from .editorial_motion import EditorialMotionPlan, EditorialMotionSegment
from .member_editorial_motion import (
    MemberEditorialMotionConfig,
    _State,
    _center,
    _frame_schedule,
    _mismatch,
    _project,
    _state_for_match,
    _track_for_states,
    _visible_fraction,
)
from .pose import SILHOUETTE_BACKEND
from .render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    RenderSegment,
    RenderTransition,
    TransitionKind,
)
from .visual_score import _VideoSampler, _working_frame


_FACE_ATTENTION_BASIS = "unambiguous_dominant_face_bbox_geometry_only"


@dataclass(frozen=True, slots=True)
class DirectMemberFaceSample:
    bbox_xywh: tuple[float, float, float, float]
    face_count: int
    detector_backend: str
    subject_bbox_xywh: tuple[float, float, float, float] | None = None
    subject_confidence: float = 0.0
    subject_geometry_basis: str = "none"

    def __post_init__(self) -> None:
        if len(self.bbox_xywh) != 4:
            raise ValueError("direct member face box needs four values")
        x, y, width, height = tuple(float(value) for value in self.bbox_xywh)
        if any(not math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError("direct member face box must be finite")
        if not (
            0.0 <= x < 1.0
            and 0.0 <= y < 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
            and x + width <= 1.0 + 1e-7
            and y + height <= 1.0 + 1e-7
        ):
            raise ValueError("direct member face box must fit normalized bounds")
        if (
            isinstance(self.face_count, bool)
            or not isinstance(self.face_count, int)
            or self.face_count not in {0, 1}
        ):
            raise ValueError(
                "direct member geometry sample needs exactly one face or "
                "verified subject-only geometry"
            )
        backend = self.detector_backend.strip().casefold()
        subject_box = self.subject_bbox_xywh
        if subject_box is not None:
            if len(subject_box) != 4:
                raise ValueError("direct member subject box needs four values")
            sx, sy, swidth, sheight = tuple(float(value) for value in subject_box)
            if any(
                not math.isfinite(value)
                for value in (sx, sy, swidth, sheight)
            ) or not (
                0.0 <= sx < 1.0
                and 0.0 <= sy < 1.0
                and 0.0 < swidth <= 1.0
                and 0.0 < sheight <= 1.0
                and sx + swidth <= 1.0 + 1e-7
                and sy + sheight <= 1.0 + 1e-7
            ):
                raise ValueError(
                    "direct member subject box must fit normalized bounds"
                )
            subject_box = (sx, sy, swidth, sheight)
        confidence = float(self.subject_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("direct member subject confidence must be normalized")
        basis = self.subject_geometry_basis.strip()
        if subject_box is None:
            if confidence != 0.0 or basis != "none":
                raise ValueError(
                    "direct member missing subject box cannot claim geometry"
                )
        elif confidence <= 0.0 or basis not in {
            "pphumanseg_full_subject_silhouette",
            "face_associated_formation_person_box",
        }:
            raise ValueError("direct member subject geometry is unverified")
        if self.face_count == 1:
            if backend != "yunet":
                raise ValueError(
                    "direct member face sample needs verified YuNet geometry"
                )
        elif (
            backend != "none"
            or subject_box is None
            or any(
                abs(value - subject_value) > 1e-7
                for value, subject_value in zip(
                    (x, y, width, height),
                    subject_box,
                    strict=True,
                )
            )
        ):
            raise ValueError(
                "subject-only geometry must use its verified body box"
            )
        object.__setattr__(self, "bbox_xywh", (x, y, width, height))
        object.__setattr__(self, "detector_backend", backend)
        object.__setattr__(self, "subject_bbox_xywh", subject_box)
        object.__setattr__(self, "subject_confidence", confidence)
        object.__setattr__(self, "subject_geometry_basis", basis)

    @property
    def alignment_geometry_basis(self) -> str:
        if self.face_count == 1:
            return "unambiguous_dominant_face_bbox_geometry_only"
        if self.subject_geometry_basis == "pphumanseg_full_subject_silhouette":
            return "aggregate_human_silhouette"
        return "unambiguous_dominant_person_box_geometry_only"


DirectMemberGeometryProvider = Callable[
    [int, float], DirectMemberFaceSample | None
]


def _enable_dense_sequential_frame_reads(sampler: object) -> None:
    """Avoid an AV1 random seek for every adjacent dense-review frame.

    This optimization is intentionally local to the direct-member reviewer.
    It falls back to the canonical timestamp seek whenever OpenCV's reported
    frame position does not prove that the capture is exactly one frame ahead.
    """

    required_attributes = (
        "cache",
        "capture",
        "frame_at",
        "frame_decode_count",
        "frame_identity",
        "frame_rate",
    )
    if any(not hasattr(sampler, name) for name in required_attributes):
        return
    original_frame_at = sampler.frame_at  # type: ignore[attr-defined]
    capture = sampler.capture  # type: ignore[attr-defined]
    cache = sampler.cache  # type: ignore[attr-defined]
    frame_rate = float(sampler.frame_rate)  # type: ignore[attr-defined]
    if not math.isfinite(frame_rate) or frame_rate <= 0.0:
        return
    previous_decoded_identity = None
    sequential_decode_count = 0
    random_seek_decode_count = 0

    def frame_at(seconds: float):
        nonlocal previous_decoded_identity
        nonlocal sequential_decode_count
        nonlocal random_seek_decode_count
        key = sampler.frame_identity(seconds)  # type: ignore[attr-defined]
        if key in cache:
            return original_frame_at(seconds)
        expected_step = 1.0 / frame_rate
        adjacent = bool(
            previous_decoded_identity is not None
            and key.frame_index == previous_decoded_identity.frame_index + 1
            and key.fps_numerator
            == previous_decoded_identity.fps_numerator
            and key.fps_denominator
            == previous_decoded_identity.fps_denominator
            and abs(
                key.canonical_seconds
                - previous_decoded_identity.canonical_seconds
                - expected_step
            )
            <= expected_step * 0.05 + 1e-9
        )
        if adjacent:
            position_before = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if (
                math.isfinite(position_before)
                and abs(position_before - key.frame_index) <= 0.51
            ):
                success, frame = capture.read()
                sampler.frame_decode_count += 1  # type: ignore[attr-defined]
                position_after = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
                if (
                    success
                    and math.isfinite(position_after)
                    and abs(position_after - (key.frame_index + 1)) <= 0.51
                ):
                    value = _working_frame(frame)
                    cache[key] = value
                    while len(cache) > 32:
                        cache.popitem(last=False)
                    previous_decoded_identity = key
                    sequential_decode_count += 1
                    setattr(
                        sampler,
                        "dense_sequential_frame_decode_count",
                        sequential_decode_count,
                    )
                    return value
        before = int(sampler.frame_decode_count)  # type: ignore[attr-defined]
        value = original_frame_at(seconds)
        if int(sampler.frame_decode_count) > before:  # type: ignore[attr-defined]
            random_seek_decode_count += 1
            setattr(
                sampler,
                "dense_random_seek_frame_decode_count",
                random_seek_decode_count,
            )
        position_after = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
        previous_decoded_identity = (
            key
            if value is not None
            and math.isfinite(position_after)
            and abs(position_after - (key.frame_index + 1)) <= 0.51
            else None
        )
        return value

    setattr(sampler, "frame_at", frame_at)
    setattr(sampler, "dense_sequential_frame_decode_count", 0)
    setattr(sampler, "dense_random_seek_frame_decode_count", 0)


class DirectMemberDenseFaceProvider:
    """Decode selected geometry on canonical source frames, never identities."""

    def __init__(
        self,
        sources: Sequence[Path],
        *,
        feature_cache_directory: Path | None = None,
    ) -> None:
        self._sources = tuple(Path(source) for source in sources)
        self._samplers: list[_VideoSampler | None] = []
        self._cache: dict[
            tuple[object, ...], DirectMemberFaceSample | None
        ] = {}
        self._request_count = 0
        self._cache_hit_count = 0
        self._decode_count = 0
        for source in self._sources:
            try:
                sampler = _VideoSampler(
                    source,
                    feature_cache_directory=feature_cache_directory,
                )
                _enable_dense_sequential_frame_reads(sampler)
            except (OSError, TypeError, ValueError, cv2.error):
                sampler = None
            self._samplers.append(sampler)

    def sample(
        self,
        source_index: int,
        source_local_seconds: float,
    ) -> DirectMemberFaceSample | None:
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or not 0 <= source_index < len(self._sources)
        ):
            raise ValueError("dense member face source index is invalid")
        if not math.isfinite(source_local_seconds):
            raise ValueError("dense member face time must be finite")
        self._request_count += 1
        sampler = self._samplers[source_index]
        if sampler is None:
            return None
        normalized_seconds = max(0.0, source_local_seconds)
        identity = sampler.frame_identity(normalized_seconds)
        key = (
            source_index,
            identity.frame_index,
            identity.fps_numerator,
            identity.fps_denominator,
        )
        if key in self._cache:
            self._cache_hit_count += 1
            return self._cache[key]
        previous_decode_count = sampler.frame_decode_count
        try:
            descriptor = sampler.feature_descriptor_at(
                normalized_seconds,
                include_subject_geometry=True,
                include_formation_pose=True,
            )
            self._decode_count += (
                sampler.frame_decode_count - previous_decode_count
            )
            if descriptor is None:
                self._cache[key] = None
                return None
            subject_box = None
            subject_confidence = 0.0
            subject_basis = "none"
            if (
                descriptor.silhouette_bbox_xywh is not None
                and descriptor.silhouette_confidence >= 0.30
                and descriptor.silhouette_backend == SILHOUETTE_BACKEND
            ):
                subject_box = descriptor.silhouette_bbox_xywh
                subject_confidence = descriptor.silhouette_confidence
                subject_basis = "pphumanseg_full_subject_silhouette"
            elif descriptor.primary_face_bbox is not None:
                face = descriptor.primary_face_bbox
                face_center = (
                    face[0] + face[2] / 2.0,
                    face[1] + face[3] / 2.0,
                )
                associated = tuple(
                    person
                    for person in descriptor.formation_people
                    if person.bbox_xywh[0] - 1e-6
                    <= face_center[0]
                    <= person.bbox_xywh[0] + person.bbox_xywh[2] + 1e-6
                    and person.bbox_xywh[1] - 1e-6
                    <= face_center[1]
                    <= person.bbox_xywh[1] + person.bbox_xywh[3] + 1e-6
                    and person.detection_confidence >= 0.50
                )
                if len(associated) == 1:
                    subject_box = associated[0].bbox_xywh
                    subject_confidence = associated[0].detection_confidence
                    subject_basis = "face_associated_formation_person_box"
            if (
                descriptor.primary_face_bbox is not None
                and descriptor.face_count == 1
                and descriptor.face_detector_backend.casefold() == "yunet"
            ):
                sample = DirectMemberFaceSample(
                    descriptor.primary_face_bbox,
                    descriptor.face_count,
                    descriptor.face_detector_backend,
                    subject_box,
                    subject_confidence,
                    subject_basis,
                )
            elif subject_box is not None:
                sample = DirectMemberFaceSample(
                    subject_box,
                    0,
                    "none",
                    subject_box,
                    subject_confidence,
                    subject_basis,
                )
            else:
                sample = None
        except (cv2.error, ValueError, FloatingPointError):
            self._decode_count += max(
                0,
                sampler.frame_decode_count - previous_decode_count,
            )
            sample = None
        self._cache[key] = sample
        return sample

    def diagnostics(self) -> dict[str, int | bool | str]:
        persistent = tuple(
            diagnostics
            for sampler in self._samplers
            if sampler is not None
            and (diagnostics := sampler.feature_cache_diagnostics()) is not None
        )
        return {
            "backend": (
                "canonical_frame_yunet_face_or_verified_full_subject_"
                "geometry_dense_selected_ramps"
            ),
            "request_count": self._request_count,
            "cache_hit_count": self._cache_hit_count,
            "decode_count": self._decode_count,
            "cache_entry_count": len(self._cache),
            "persistent_feature_cache_enabled_source_count": len(persistent),
            "persistent_feature_cache_hit_count": sum(
                int(item.get("hit_count", 0)) for item in persistent
            ),
            "persistent_feature_cache_miss_count": sum(
                int(item.get("miss_count", 0)) for item in persistent
            ),
            "persistent_feature_cache_write_count": sum(
                int(item.get("write_count", 0)) for item in persistent
            ),
            "sequential_frame_decode_count": sum(
                int(
                    getattr(
                        sampler,
                        "dense_sequential_frame_decode_count",
                        0,
                    )
                )
                for sampler in self._samplers
                if sampler is not None
            ),
            "random_seek_frame_decode_count": sum(
                int(
                    getattr(
                        sampler,
                        "dense_random_seek_frame_decode_count",
                        0,
                    )
                )
                for sampler in self._samplers
                if sampler is not None
            ),
            "pixels_persisted": False,
            "face_embeddings_computed": False,
        }

    def close(self) -> None:
        for sampler in self._samplers:
            if sampler is not None:
                sampler.close()


@dataclass(frozen=True, slots=True)
class DirectMemberEditorialMotionConfig:
    ramp_seconds: float = 0.75
    maximum_boundary_zoom: float = 1.50
    maximum_center_shift: float = 0.120
    maximum_boundary_center_distance: float = 0.30
    maximum_residual_center_distance: float = 0.08
    minimum_mismatch_improvement_fraction: float = 0.20
    minimum_face_visible_fraction: float = 0.995
    minimum_dense_face_visible_fraction: float = 0.980
    legacy_face_only_maximum_zoom: float = 1.12
    legacy_face_only_maximum_center_shift: float = 0.025
    minimum_widened_subject_confidence: float = 0.30
    minimum_widened_subject_visible_fraction: float = 0.950
    minimum_face_associated_subject_visible_fraction: float = 0.720
    maximum_dense_subject_center_step: float = 0.12
    maximum_dense_subject_height_ratio: float = 1.40
    maximum_dense_center_step: float = 0.12
    maximum_dense_height_ratio: float = 1.35
    maximum_dense_face_interpolation_gap_frames: int = 24
    maximum_dense_face_edge_extrapolation_frames: int = 24
    dense_geometry_median_radius_frames: int = 3
    maximum_endpoint_center_disagreement: float = 0.08
    maximum_endpoint_height_ratio: float = 1.25
    target_events_per_minute: float = 12.0
    minimum_event_spacing_seconds: float = 2.0
    enable_interior_zoom_pulses: bool = False
    target_rich_motion_events_per_minute: float = 4.0
    minimum_interior_pulse_duration_seconds: float = 2.2
    interior_pulse_zoom: float = 1.05

    def __post_init__(self) -> None:
        values = tuple(float(getattr(self, name)) for name in (
            "ramp_seconds",
            "maximum_boundary_zoom",
            "maximum_center_shift",
            "maximum_boundary_center_distance",
            "maximum_residual_center_distance",
            "minimum_mismatch_improvement_fraction",
            "minimum_face_visible_fraction",
            "minimum_dense_face_visible_fraction",
            "legacy_face_only_maximum_zoom",
            "legacy_face_only_maximum_center_shift",
            "minimum_widened_subject_confidence",
            "minimum_widened_subject_visible_fraction",
            "minimum_face_associated_subject_visible_fraction",
            "maximum_dense_subject_center_step",
            "maximum_dense_subject_height_ratio",
            "maximum_dense_center_step",
            "maximum_dense_height_ratio",
            "maximum_endpoint_center_disagreement",
            "maximum_endpoint_height_ratio",
            "target_events_per_minute",
            "minimum_event_spacing_seconds",
            "target_rich_motion_events_per_minute",
            "minimum_interior_pulse_duration_seconds",
            "interior_pulse_zoom",
        ))
        if any(not math.isfinite(value) for value in values):
            raise ValueError("direct member editorial configuration must be finite")
        # Reuse the confirmed-member configuration as the shared camera/schema
        # contract.  Dense-direct adds stricter continuity gates below.
        _shared_settings(self)
        if not 0.0 <= self.maximum_dense_center_step <= 0.20:
            raise ValueError("dense face center-step gate is invalid")
        if not 1.0 <= self.maximum_dense_height_ratio <= 2.0:
            raise ValueError("dense face height-ratio gate is invalid")
        for name, value, maximum in (
            (
                "maximum_dense_face_interpolation_gap_frames",
                self.maximum_dense_face_interpolation_gap_frames,
                30,
            ),
            (
                "maximum_dense_face_edge_extrapolation_frames",
                self.maximum_dense_face_edge_extrapolation_frames,
                30,
            ),
            (
                "dense_geometry_median_radius_frames",
                self.dense_geometry_median_radius_frames,
                3,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= maximum
            ):
                raise ValueError(f"{name} is invalid")
        if not 1.0 <= self.legacy_face_only_maximum_zoom <= 1.12:
            raise ValueError("legacy face-only zoom boundary is invalid")
        if not 0.0 <= self.legacy_face_only_maximum_center_shift <= 0.025:
            raise ValueError("legacy face-only pan boundary is invalid")
        if not 0.0 <= self.minimum_widened_subject_confidence <= 1.0:
            raise ValueError("widened subject confidence gate is invalid")
        if not 0.95 <= self.minimum_widened_subject_visible_fraction <= 1.0:
            raise ValueError("widened subject containment gate is invalid")
        if not (
            0.95
            <= self.minimum_dense_face_visible_fraction
            <= self.minimum_face_visible_fraction
            <= 1.0
        ):
            raise ValueError("dense face containment gate is invalid")
        if not (
            0.60
            <= self.minimum_face_associated_subject_visible_fraction
            <= self.minimum_widened_subject_visible_fraction
        ):
            raise ValueError(
                "face-associated subject containment gate is invalid"
            )
        if not 0.0 <= self.maximum_dense_subject_center_step <= 0.20:
            raise ValueError("dense subject center-step gate is invalid")
        if not 1.0 <= self.maximum_dense_subject_height_ratio <= 2.0:
            raise ValueError("dense subject height-ratio gate is invalid")
        if not 0.0 <= self.maximum_endpoint_center_disagreement <= 0.20:
            raise ValueError("dense endpoint center gate is invalid")
        if not 1.0 <= self.maximum_endpoint_height_ratio <= 2.0:
            raise ValueError("dense endpoint height-ratio gate is invalid")
        if not isinstance(self.enable_interior_zoom_pulses, bool):
            raise ValueError("interior zoom pulse switch must be boolean")
        if not 0.0 <= self.target_rich_motion_events_per_minute <= 12.0:
            raise ValueError("rich direct-member motion density is invalid")
        if not 1.5 <= self.minimum_interior_pulse_duration_seconds <= 6.0:
            raise ValueError("interior zoom pulse duration floor is invalid")
        if not 1.02 <= self.interior_pulse_zoom <= 1.12:
            raise ValueError("interior zoom pulse must remain subtle")


def _shared_settings(
    config: DirectMemberEditorialMotionConfig,
) -> MemberEditorialMotionConfig:
    return MemberEditorialMotionConfig(
        ramp_seconds=config.ramp_seconds,
        maximum_boundary_zoom=config.maximum_boundary_zoom,
        maximum_center_shift=config.maximum_center_shift,
        maximum_boundary_center_distance=config.maximum_boundary_center_distance,
        maximum_residual_center_distance=config.maximum_residual_center_distance,
        minimum_mismatch_improvement_fraction=(
            config.minimum_mismatch_improvement_fraction
        ),
        minimum_subject_visible_fraction=config.minimum_face_visible_fraction,
        target_events_per_minute=config.target_events_per_minute,
        minimum_event_spacing_seconds=config.minimum_event_spacing_seconds,
    )


@dataclass(frozen=True, slots=True)
class _BoundaryCandidate:
    boundary_index: int
    event_time: float
    score: float
    side: str
    state: _State
    alignment_geometry_basis: str
    alternative_states: tuple[_State, ...] = ()
    required: bool = False


def _attention_alignment_geometry(
    sample: object,
) -> tuple[tuple[float, float, float, float], str] | None:
    attention = getattr(sample, "attention", None)
    if attention is None:
        return None
    basis = str(getattr(attention, "basis", ""))
    descriptor = getattr(sample, "descriptor", None)
    if descriptor is None:
        return None
    if basis == _FACE_ATTENTION_BASIS:
        if getattr(descriptor, "face_count", None) != 1:
            return None
    elif basis == "aggregate_human_silhouette":
        if (
            getattr(descriptor, "silhouette_backend", None)
            != SILHOUETTE_BACKEND
            or float(getattr(descriptor, "silhouette_confidence", 0.0))
            < 0.30
        ):
            return None
    elif basis == "unambiguous_dominant_person_box_geometry_only":
        if not tuple(getattr(descriptor, "formation_people", ())):
            return None
    else:
        return None
    raw = getattr(attention, "bbox_xywh", None)
    if not isinstance(raw, tuple) or len(raw) != 4:
        return None
    box = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) for value in box):
        return None
    return (box, basis)  # type: ignore[return-value]


def _endpoint_agrees(
    dense_box: tuple[float, float, float, float],
    audited_box: tuple[float, float, float, float],
    config: DirectMemberEditorialMotionConfig,
) -> bool:
    center_distance = math.hypot(
        _center(dense_box)[0] - _center(audited_box)[0],
        _center(dense_box)[1] - _center(audited_box)[1],
    )
    height_ratio = max(dense_box[3], audited_box[3]) / max(
        1e-9, min(dense_box[3], audited_box[3])
    )
    return (
        center_distance <= config.maximum_endpoint_center_disagreement + 1e-9
        and height_ratio <= config.maximum_endpoint_height_ratio + 1e-9
    )


def _neutral(track: EditorialTransformTrack, time_seconds: float) -> bool:
    state = track.evaluate(time_seconds)
    return (
        abs(state.zoom - 1.0) <= 1e-7
        and abs(state.center_x - 0.5) <= 1e-7
        and abs(state.center_y - 0.5) <= 1e-7
        and abs(state.roll_degrees) <= 1e-7
    )


def _geometry_step_exceeds(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    *,
    maximum_center_step: float,
    maximum_height_ratio: float,
) -> bool:
    center_step = math.hypot(
        _center(left)[0] - _center(right)[0],
        _center(left)[1] - _center(right)[1],
    )
    height_ratio = max(left[3], right[3]) / max(
        1e-9, min(left[3], right[3])
    )
    return (
        center_step > maximum_center_step + 1e-9
        or height_ratio > maximum_height_ratio + 1e-9
    )


def _midpoint_box(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return tuple(
        (left_value + right_value) / 2.0
        for left_value, right_value in zip(left, right, strict=True)
    )  # type: ignore[return-value]


def _box_contains_center(
    container: tuple[float, float, float, float],
    item: tuple[float, float, float, float],
) -> bool:
    center_x, center_y = _center(item)
    return (
        container[0] - 1e-6
        <= center_x
        <= container[0] + container[2] + 1e-6
        and container[1] - 1e-6
        <= center_y
        <= container[1] + container[3] + 1e-6
    )


def _isolated_outlier_replacement(
    previous: tuple[float, float, float, float],
    current: tuple[float, float, float, float],
    following: tuple[float, float, float, float],
    *,
    maximum_center_step: float,
    maximum_height_ratio: float,
) -> tuple[float, float, float, float] | None:
    """Repair only a one-frame detector excursion with neighbor consensus."""

    if _geometry_step_exceeds(
        previous,
        following,
        maximum_center_step=maximum_center_step,
        maximum_height_ratio=maximum_height_ratio,
    ):
        return None
    if not (
        _geometry_step_exceeds(
            previous,
            current,
            maximum_center_step=maximum_center_step,
            maximum_height_ratio=maximum_height_ratio,
        )
        or _geometry_step_exceeds(
            current,
            following,
            maximum_center_step=maximum_center_step,
            maximum_height_ratio=maximum_height_ratio,
        )
    ):
        return None
    return _midpoint_box(previous, following)


def _linear_box(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    return tuple(
        left_value + (right_value - left_value) * alpha
        for left_value, right_value in zip(left, right, strict=True)
    )  # type: ignore[return-value]


def _relative_box(
    item: tuple[float, float, float, float],
    container: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    if container[2] <= 1e-9 or container[3] <= 1e-9:
        return None
    item_center = _center(item)
    return (
        (item_center[0] - container[0]) / container[2],
        (item_center[1] - container[1]) / container[3],
        item[2] / container[2],
        item[3] / container[3],
    )


def _box_from_relative(
    relative: tuple[float, float, float, float],
    container: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    width = relative[2] * container[2]
    height = relative[3] * container[3]
    center_x = container[0] + relative[0] * container[2]
    center_y = container[1] + relative[1] * container[3]
    x = center_x - width / 2.0
    y = center_y - height / 2.0
    if (
        width <= 0.0
        or height <= 0.0
        or x < -1e-6
        or y < -1e-6
        or x + width > 1.0 + 1e-6
        or y + height > 1.0 + 1e-6
    ):
        return None
    return (
        max(0.0, x),
        max(0.0, y),
        min(width, 1.0 - max(0.0, x)),
        min(height, 1.0 - max(0.0, y)),
    )


def _median_value(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _median_smooth_geometry(
    frame_indices: Sequence[int],
    boxes: Sequence[tuple[float, float, float, float]],
    *,
    radius: int,
) -> list[tuple[float, float, float, float]]:
    if radius <= 0:
        return list(boxes)
    output = list(boxes)
    for index in range(len(boxes)):
        run_start = index
        while (
            run_start > 0
            and frame_indices[run_start - 1] + 1 == frame_indices[run_start]
        ):
            run_start -= 1
        run_end = index
        while (
            run_end + 1 < len(boxes)
            and frame_indices[run_end] + 1 == frame_indices[run_end + 1]
        ):
            run_end += 1
        low = max(run_start, index - radius)
        high = min(run_end, index + radius)
        window = boxes[low : high + 1]
        output[index] = tuple(
            _median_value(tuple(box[component] for box in window))
            for component in range(4)
        )  # type: ignore[assignment]
    return output


def _stabilize_subject_relative_face_track(
    *,
    frame_indices: Sequence[int],
    face_boxes: Sequence[tuple[float, float, float, float] | None],
    subject_boxes: Sequence[tuple[float, float, float, float] | None],
    frame_count: int,
    config: DirectMemberEditorialMotionConfig,
) -> list[tuple[float, float, float, float] | None]:
    """Keep the boundary-anchored, temporally continuous face path.

    Official portrait fancams are authenticated as target-member sources, but
    YuNet can still prefer a second member for a few frames.  The real edit
    boundary is independently decoded and audited before this function runs,
    so a transformed run touching frame zero or the final frame has a trusted
    temporal anchor.  We retain only detections connected to that anchor in
    subject-relative coordinates.  A skipped excursion is subsequently
    reconstructed only within the configured gap limit; a persistent switch
    therefore still fails closed.
    """

    output = list(face_boxes)

    def connected(left: int, right: int) -> bool:
        left_box = output[left]
        right_box = output[right]
        left_subject = subject_boxes[left]
        right_subject = subject_boxes[right]
        if (
            left_box is None
            or right_box is None
            or left_subject is None
            or right_subject is None
        ):
            return False
        left_relative = _relative_box(left_box, left_subject)
        right_relative = _relative_box(right_box, right_subject)
        if left_relative is None or right_relative is None:
            return False
        frame_gap = abs(frame_indices[right] - frame_indices[left])
        if (
            frame_gap <= 0
            or frame_gap
            > config.maximum_dense_face_interpolation_gap_frames + 1
        ):
            return False
        center_distance = math.hypot(
            left_relative[0] - right_relative[0],
            left_relative[1] - right_relative[1],
        )
        maximum_center_distance = min(
            0.30,
            0.18 + 0.010 * max(0, frame_gap - 1),
        )
        height_ratio = max(left_relative[3], right_relative[3]) / max(
            1e-9,
            min(left_relative[3], right_relative[3]),
        )
        maximum_height_ratio = min(
            1.80,
            config.maximum_dense_height_ratio
            + 0.025 * max(0, frame_gap - 1),
        )
        return (
            center_distance <= maximum_center_distance + 1e-9
            and height_ratio <= maximum_height_ratio + 1e-9
        )

    run_start = 0
    while run_start < len(frame_indices):
        run_end = run_start
        while (
            run_end + 1 < len(frame_indices)
            and frame_indices[run_end] + 1 == frame_indices[run_end + 1]
        ):
            run_end += 1
        detections = [
            index
            for index in range(run_start, run_end + 1)
            if output[index] is not None and subject_boxes[index] is not None
        ]
        if len(detections) < 3:
            run_start = run_end + 1
            continue
        if frame_indices[run_start] == 0:
            ordered = detections
            anchored = True
        elif frame_indices[run_end] == frame_count - 1:
            ordered = list(reversed(detections))
            anchored = True
        else:
            ordered = detections
            anchored = False

        scores = [float("-inf")] * len(ordered)
        lengths = [0] * len(ordered)
        predecessors: list[int | None] = [None] * len(ordered)
        for position, current in enumerate(ordered):
            if not anchored or position == 0:
                scores[position] = 1.0
                lengths[position] = 1
            for previous_position in range(position):
                if not math.isfinite(scores[previous_position]):
                    continue
                previous = ordered[previous_position]
                if not connected(previous, current):
                    continue
                frame_gap = abs(
                    frame_indices[current] - frame_indices[previous]
                )
                candidate_score = (
                    scores[previous_position]
                    + 1.0
                    - 0.02 * max(0, frame_gap - 1)
                )
                candidate_length = lengths[previous_position] + 1
                if (
                    candidate_score > scores[position] + 1e-9
                    or abs(candidate_score - scores[position]) <= 1e-9
                    and candidate_length > lengths[position]
                ):
                    scores[position] = candidate_score
                    lengths[position] = candidate_length
                    predecessors[position] = previous_position

        reachable = [
            position
            for position, score in enumerate(scores)
            if math.isfinite(score)
        ]
        if not reachable:
            run_start = run_end + 1
            continue
        if anchored:
            # Coverage is primary: if the audited face cannot be connected
            # across the run, the remaining edge gap is left for the strict
            # reconstruction limit to reject.
            endpoint = max(
                reachable,
                key=lambda position: (
                    abs(
                        frame_indices[ordered[position]]
                        - frame_indices[ordered[0]]
                    ),
                    scores[position],
                    lengths[position],
                ),
            )
        else:
            endpoint = max(
                reachable,
                key=lambda position: (
                    scores[position],
                    lengths[position],
                    -frame_indices[ordered[position]],
                ),
            )
        kept: set[int] = set()
        cursor: int | None = endpoint
        while cursor is not None:
            kept.add(ordered[cursor])
            cursor = predecessors[cursor]
        for index in detections:
            if index not in kept:
                output[index] = None
        run_start = run_end + 1
    return output


def _reconstruct_dense_face_geometry(
    *,
    frame_indices: Sequence[int],
    face_boxes: Sequence[tuple[float, float, float, float] | None],
    subject_boxes: Sequence[tuple[float, float, float, float] | None],
    config: DirectMemberEditorialMotionConfig,
) -> list[tuple[float, float, float, float]] | None:
    """Bridge short detector gaps in target-subject relative coordinates."""

    output = list(face_boxes)
    run_start = 0
    while run_start < len(frame_indices):
        run_end = run_start
        while (
            run_end + 1 < len(frame_indices)
            and frame_indices[run_end] + 1 == frame_indices[run_end + 1]
        ):
            run_end += 1
        anchors = [
            index
            for index in range(run_start, run_end + 1)
            if output[index] is not None
        ]
        if not anchors:
            return None
        first_anchor = anchors[0]
        prefix = first_anchor - run_start
        if prefix > config.maximum_dense_face_edge_extrapolation_frames:
            return None
        last_anchor = anchors[-1]
        suffix = run_end - last_anchor
        if suffix > config.maximum_dense_face_edge_extrapolation_frames:
            return None

        def project_anchor(anchor: int, target: int):
            anchor_box = output[anchor]
            assert anchor_box is not None
            anchor_subject = subject_boxes[anchor]
            target_subject = subject_boxes[target]
            if anchor_subject is None or target_subject is None:
                return anchor_box
            relative = _relative_box(anchor_box, anchor_subject)
            if relative is None:
                return anchor_box
            uncertainty = min(0.12, abs(target - anchor) * 0.005)
            expanded_relative = (
                relative[0],
                relative[1],
                relative[2] + uncertainty * 2.0,
                relative[3] + uncertainty * 2.0,
            )
            # Expansion is deliberately conservative, but a face near the
            # subject/frame edge can make only the uncertainty envelope leave
            # bounds.  Preserve the unexpanded subject-relative projection in
            # that case; if even that cannot be represented, retain the last
            # verified absolute box.  The later output-crop review still
            # rejects any unsafe camera transform.
            return (
                _box_from_relative(expanded_relative, target_subject)
                or _box_from_relative(relative, target_subject)
                or anchor_box
            )

        for index in range(run_start, first_anchor):
            output[index] = project_anchor(first_anchor, index)
        for index in range(last_anchor + 1, run_end + 1):
            output[index] = project_anchor(last_anchor, index)

        for left_anchor, right_anchor in zip(anchors, anchors[1:]):
            gap = right_anchor - left_anchor - 1
            if gap <= 0:
                continue
            if gap > config.maximum_dense_face_interpolation_gap_frames:
                return None
            left_box = output[left_anchor]
            right_box = output[right_anchor]
            assert left_box is not None and right_box is not None
            left_subject = subject_boxes[left_anchor]
            right_subject = subject_boxes[right_anchor]
            left_relative = (
                _relative_box(left_box, left_subject)
                if left_subject is not None
                else None
            )
            right_relative = (
                _relative_box(right_box, right_subject)
                if right_subject is not None
                else None
            )
            for index in range(left_anchor + 1, right_anchor):
                alpha = (index - left_anchor) / (right_anchor - left_anchor)
                target_subject = subject_boxes[index]
                reconstructed = None
                if (
                    left_relative is not None
                    and right_relative is not None
                    and target_subject is not None
                ):
                    reconstructed = _box_from_relative(
                        _linear_box(left_relative, right_relative, alpha),
                        target_subject,
                    )
                if reconstructed is None:
                    reconstructed = _linear_box(left_box, right_box, alpha)
                output[index] = reconstructed
        if any(output[index] is None for index in range(run_start, run_end + 1)):
            return None
        run_start = run_end + 1
    resolved = [box for box in output if box is not None]
    if len(resolved) != len(output):
        return None
    if all(subject_box is not None for subject_box in subject_boxes):
        resolved_subjects = [
            subject_box
            for subject_box in subject_boxes
            if subject_box is not None
        ]
        relative_boxes = [
            _relative_box(box, subject)
            for box, subject in zip(
                resolved,
                resolved_subjects,
                strict=True,
            )
        ]
        if all(relative is not None for relative in relative_boxes):
            smoothed_relative = _median_smooth_geometry(
                frame_indices,
                [
                    relative
                    for relative in relative_boxes
                    if relative is not None
                ],
                radius=config.dense_geometry_median_radius_frames,
            )
            remapped = [
                _box_from_relative(relative, subject)
                for relative, subject in zip(
                    smoothed_relative,
                    resolved_subjects,
                    strict=True,
                )
            ]
            if all(box is not None for box in remapped):
                return [box for box in remapped if box is not None]
        return _median_smooth_geometry(
            frame_indices,
            resolved,
            radius=config.dense_geometry_median_radius_frames,
        )
    return resolved


def _track_is_dense_face_safe(
    *,
    segment_index: int,
    eligible_source_index: int,
    segment: RenderSegment,
    track: EditorialTransformTrack,
    frame_count: int,
    common_master_start: float,
    geometry_provider: DirectMemberGeometryProvider,
    fps: float,
    config: DirectMemberEditorialMotionConfig,
    source_frame_geometry: DirectMemberSourceFrameGeometry | None,
    alignment_geometry_basis: str | None = None,
    rejection_callback: Callable[[str], None] | None = None,
    rejection_detail_callback: Callable[[str], None] | None = None,
) -> bool:
    def reject(reason: str, detail: str | None = None) -> bool:
        if rejection_callback is not None:
            rejection_callback(reason)
        if detail is not None and rejection_detail_callback is not None:
            rejection_detail_callback(f"{reason}({detail})")
        return False

    active_frames = tuple(
        not _neutral(track, frame_index / fps)
        for frame_index in range(frame_count)
    )
    # Include one neutral-frame halo on each side of every transformed run.
    # Otherwise a detector/subject jump exactly where the ramp begins or ends
    # has no adjacent dense sample and can look continuous inside the ramp.
    review_frames: set[int] = {
        frame_index
        for frame_index, active in enumerate(active_frames)
        if active
    }
    for frame_index in tuple(review_frames):
        if frame_index > 0:
            review_frames.add(frame_index - 1)
        if frame_index + 1 < frame_count:
            review_frames.add(frame_index + 1)

    widened_subject_proof_required = bool(
        any(
            keyframe.zoom
            > config.legacy_face_only_maximum_zoom + 1e-9
            or abs(keyframe.center_x - 0.5)
            > config.legacy_face_only_maximum_center_shift + 1e-9
            or abs(keyframe.center_y - 0.5)
            > config.legacy_face_only_maximum_center_shift + 1e-9
            for keyframe in track.keyframes
        )
    )

    ordered_frames = tuple(sorted(review_frames))
    samples: list[DirectMemberFaceSample] = []
    raw_boxes: list[tuple[float, float, float, float]] = []
    face_boxes: list[tuple[float, float, float, float] | None] = []
    subject_boxes: list[tuple[float, float, float, float] | None] = []
    for frame_index in ordered_frames:
        local_time = frame_index / fps
        canonical_time = common_master_start + segment.master_start + local_time
        sample = geometry_provider(eligible_source_index, canonical_time)
        if sample is None:
            return reject("dense_geometry_missing")
        box = (
            map_box_to_portrait_fill(
                sample.bbox_xywh,
                source_frame_geometry,
            )
            if source_frame_geometry is not None
            else sample.bbox_xywh
        )
        subject_box = sample.subject_bbox_xywh
        if subject_box is not None and source_frame_geometry is not None:
            subject_box = map_box_to_portrait_fill(
                subject_box,
                source_frame_geometry,
            )
        samples.append(sample)
        raw_boxes.append(box)
        face_boxes.append(box if sample.face_count == 1 else None)
        subject_boxes.append(subject_box)

    resolved_basis = alignment_geometry_basis
    if resolved_basis is None:
        resolved_basis = (
            _FACE_ATTENTION_BASIS
            if sum(sample.face_count == 1 for sample in samples)
            >= max(2, math.ceil(len(samples) / 2.0))
            else "aggregate_human_silhouette"
        )
    if resolved_basis == _FACE_ATTENTION_BASIS:
        if all(subject_box is not None for subject_box in subject_boxes):
            subject_boxes = _median_smooth_geometry(
                ordered_frames,
                [
                    subject_box
                    for subject_box in subject_boxes
                    if subject_box is not None
                ],
                radius=config.dense_geometry_median_radius_frames,
            )
        for index, face_box in enumerate(face_boxes):
            subject_box = subject_boxes[index]
            if (
                face_box is not None
                and subject_box is not None
                and not _box_contains_center(subject_box, face_box)
            ):
                face_boxes[index] = None
        face_boxes = _stabilize_subject_relative_face_track(
            frame_indices=ordered_frames,
            face_boxes=face_boxes,
            subject_boxes=subject_boxes,
            frame_count=frame_count,
            config=config,
        )
        reconstructed = _reconstruct_dense_face_geometry(
            frame_indices=ordered_frames,
            face_boxes=face_boxes,
            subject_boxes=subject_boxes,
            config=config,
        )
        if reconstructed is None:
            return reject(
                "dense_face_track_gap_unresolved",
                "face_detections="
                f"{sum(face_box is not None for face_box in face_boxes)},"
                f"review_frames={len(ordered_frames)}",
            )
        boxes = reconstructed
    elif resolved_basis in {
        "aggregate_human_silhouette",
        "unambiguous_dominant_person_box_geometry_only",
    }:
        if any(subject_box is None for subject_box in subject_boxes):
            return reject("dense_subject_track_gap_unresolved")
        boxes = _median_smooth_geometry(
            ordered_frames,
            [
                subject_box
                for subject_box in subject_boxes
                if subject_box is not None
            ],
            radius=config.dense_geometry_median_radius_frames,
        )
    else:
        return reject("dense_alignment_geometry_basis_invalid")

    if (
        resolved_basis != _FACE_ATTENTION_BASIS
        and all(subject_box is not None for subject_box in subject_boxes)
    ):
        smoothed_subject_boxes = _median_smooth_geometry(
            ordered_frames,
            [
                subject_box
                for subject_box in subject_boxes
                if subject_box is not None
            ],
            radius=config.dense_geometry_median_radius_frames,
        )
        subject_boxes = list(smoothed_subject_boxes)

    # YuNet and silhouette boxes can make a single-frame excursion while the
    # authenticated member track is otherwise continuous. Replace only that
    # isolated excursion, and only when the same verified subject contains the
    # face on all three frames. A persistent jump or native source cut remains
    # visible to the continuity gate below.
    for index in range(1, len(ordered_frames) - 1):
        if not (
            ordered_frames[index - 1] + 1 == ordered_frames[index]
            and ordered_frames[index] + 1 == ordered_frames[index + 1]
            and all(
                samples[item].face_count == 1
                for item in (index - 1, index, index + 1)
            )
            and all(
                subject_boxes[item] is not None
                for item in (index - 1, index, index + 1)
            )
        ):
            continue
        previous_subject = subject_boxes[index - 1]
        current_subject = subject_boxes[index]
        following_subject = subject_boxes[index + 1]
        assert previous_subject is not None
        assert current_subject is not None
        assert following_subject is not None
        if not all(
            _box_contains_center(subject, boxes[item])
            for item, subject in (
                (index - 1, previous_subject),
                (index, current_subject),
                (index + 1, following_subject),
            )
        ):
            continue
        replacement = _isolated_outlier_replacement(
            boxes[index - 1],
            boxes[index],
            boxes[index + 1],
            maximum_center_step=config.maximum_dense_center_step,
            maximum_height_ratio=config.maximum_dense_height_ratio,
        )
        if replacement is not None:
            boxes[index] = replacement

    # Silhouette bounds breathe with limbs. The same one-frame consensus rule
    # is safe when the independently tracked face stays associated throughout.
    for index in range(1, len(ordered_frames) - 1):
        if not (
            ordered_frames[index - 1] + 1 == ordered_frames[index]
            and ordered_frames[index] + 1 == ordered_frames[index + 1]
        ):
            continue
        previous_subject = subject_boxes[index - 1]
        current_subject = subject_boxes[index]
        following_subject = subject_boxes[index + 1]
        if (
            previous_subject is None
            or current_subject is None
            or following_subject is None
            or not all(
                _box_contains_center(subject, boxes[item])
                for item, subject in (
                    (index - 1, previous_subject),
                    (index, current_subject),
                    (index + 1, following_subject),
                )
            )
        ):
            continue
        replacement = _isolated_outlier_replacement(
            previous_subject,
            current_subject,
            following_subject,
            maximum_center_step=config.maximum_dense_subject_center_step,
            maximum_height_ratio=config.maximum_dense_subject_height_ratio,
        )
        if replacement is not None:
            subject_boxes[index] = replacement

    previous_box: tuple[float, float, float, float] | None = None
    previous_subject_box: tuple[float, float, float, float] | None = None
    previous_frame_index: int | None = None
    for frame_index, sample, box, subject_box in zip(
        ordered_frames,
        samples,
        boxes,
        subject_boxes,
        strict=True,
    ):
        local_time = frame_index / fps
        if (
            previous_frame_index is None
            or frame_index != previous_frame_index + 1
        ):
            previous_box = None
            previous_subject_box = None
        if previous_box is not None:
            if _geometry_step_exceeds(
                previous_box,
                box,
                maximum_center_step=config.maximum_dense_center_step,
                maximum_height_ratio=config.maximum_dense_height_ratio,
            ):
                center_step = math.hypot(
                    _center(previous_box)[0] - _center(box)[0],
                    _center(previous_box)[1] - _center(box)[1],
                )
                height_ratio = max(previous_box[3], box[3]) / max(
                    1e-9,
                    min(previous_box[3], box[3]),
                )
                return reject(
                    "dense_face_geometry_jump",
                    f"frame={frame_index},center_step={center_step:.6f},"
                    f"height_ratio={height_ratio:.6f},"
                    f"basis={resolved_basis}",
                )
        state_sample = track.evaluate(local_time)
        state = _State(
            state_sample.zoom,
            state_sample.center_x,
            state_sample.center_y,
        )
        face_visible_fraction = _visible_fraction(box, state)
        if (
            face_visible_fraction + 1e-9
            < config.minimum_dense_face_visible_fraction
        ):
            return reject(
                "dense_face_crop_exceeds_visible_fraction",
                f"frame={frame_index},visible={face_visible_fraction:.6f},"
                f"zoom={state.zoom:.6f},center_x={state.center_x:.6f},"
                f"center_y={state.center_y:.6f}",
            )
        if widened_subject_proof_required:
            if (
                subject_box is None
                or sample.subject_confidence + 1e-9
                < config.minimum_widened_subject_confidence
                or sample.subject_geometry_basis
                not in {
                    "pphumanseg_full_subject_silhouette",
                    "face_associated_formation_person_box",
                }
            ):
                return reject("dense_widened_subject_evidence_missing")
            if sample.face_count == 1 and not _box_contains_center(
                subject_box,
                box,
            ):
                return reject("dense_subject_face_association_failed")
            if previous_subject_box is not None:
                if _geometry_step_exceeds(
                    previous_subject_box,
                    subject_box,
                    maximum_center_step=(
                        config.maximum_dense_subject_center_step
                    ),
                    maximum_height_ratio=(
                        config.maximum_dense_subject_height_ratio
                    ),
                ):
                    subject_center_step = math.hypot(
                        _center(previous_subject_box)[0]
                        - _center(subject_box)[0],
                        _center(previous_subject_box)[1]
                        - _center(subject_box)[1],
                    )
                    subject_height_ratio = max(
                        previous_subject_box[3],
                        subject_box[3],
                    ) / max(
                        1e-9,
                        min(previous_subject_box[3], subject_box[3]),
                    )
                    return reject(
                        "dense_subject_geometry_jump",
                        f"frame={frame_index},"
                        f"center_step={subject_center_step:.6f},"
                        f"height_ratio={subject_height_ratio:.6f}",
                    )
            minimum_subject_visible_fraction = (
                config.minimum_face_associated_subject_visible_fraction
                if sample.face_count == 1
                else config.minimum_widened_subject_visible_fraction
            )
            subject_visible_fraction = _visible_fraction(subject_box, state)
            if (
                subject_visible_fraction + 1e-9
                < minimum_subject_visible_fraction
            ):
                return reject(
                    "dense_subject_crop_exceeds_visible_fraction",
                    f"frame={frame_index},"
                    f"visible={subject_visible_fraction:.6f},"
                    f"minimum={minimum_subject_visible_fraction:.6f},"
                    f"zoom={state.zoom:.6f}",
                )
            previous_subject_box = subject_box
        previous_box = box
        previous_frame_index = frame_index
    return True


def _resolved_source_frame_geometries(
    visual_selection: DirectMemberVisualSelection,
    *,
    source_count: int,
    source_frame_sizes: Sequence[tuple[int, int]] | None,
    output_frame_size: tuple[int, int],
) -> tuple[DirectMemberSourceFrameGeometry | None, ...]:
    stored = tuple(
        getattr(visual_selection, "source_frame_geometries", ())
    )
    if stored:
        if len(stored) != source_count or any(
            item is not None
            and not isinstance(item, DirectMemberSourceFrameGeometry)
            for item in stored
        ):
            raise ValueError("direct member stored source geometry is invalid")
        return stored
    if source_frame_sizes is None:
        return (None,) * source_count
    if len(source_frame_sizes) != source_count:
        raise ValueError("direct member source frame sizes must match sources")
    return tuple(
        direct_member_source_frame_geometry(
            size,
            output_size=output_frame_size,
        )
        for size in source_frame_sizes
    )


def build_direct_member_editorial_motion_plan(
    *,
    visual_selection: DirectMemberVisualSelection,
    render_segments: Sequence[RenderSegment],
    transitions: Sequence[RenderTransition],
    common_master_start: float,
    geometry_provider: DirectMemberGeometryProvider,
    fps: float = 29.97,
    config: DirectMemberEditorialMotionConfig | None = None,
    source_frame_sizes: Sequence[tuple[int, int]] | None = None,
    output_frame_size: tuple[int, int] = (1080, 1920),
) -> EditorialMotionPlan:
    """Add dense-verified face or full-subject matches to direct 9:16 cuts."""

    renders = tuple(render_segments)
    transition_items = tuple(transitions)
    source_path = tuple(int(value) for value in visual_selection.source_indices)
    boundaries = tuple(visual_selection.boundaries)
    if not renders or len(renders) != len(source_path):
        raise ValueError("direct member motion needs one selected source per segment")
    if len(transition_items) != len(renders) - 1 or len(boundaries) != len(renders) - 1:
        raise ValueError("direct member motion needs one audited transition per boundary")
    if not math.isfinite(common_master_start) or common_master_start < 0.0:
        raise ValueError("direct member common master start is invalid")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("direct member editorial fps is invalid")
    settings = config or DirectMemberEditorialMotionConfig()
    shared = _shared_settings(settings)
    source_count = max(source_path) + 1
    source_frame_geometries = _resolved_source_frame_geometries(
        visual_selection,
        source_count=source_count,
        source_frame_sizes=source_frame_sizes,
        output_frame_size=output_frame_size,
    )
    frame_counts, last_frame_times, boundary_times = _frame_schedule(renders, fps)
    if any(segment.crop is not None or segment.pan_crop is not None for segment in renders):
        raise ValueError("direct member editorial motion requires original full frames")

    blocked: set[int] = set()
    for index, transition in enumerate(transition_items):
        if transition.kind.uses_overlap:
            blocked.update((index, index + 1))
    skipped: dict[str, int] = {}
    required_rejection_reasons: dict[int, set[str]] = {}
    required_rejection_context: dict[int, str] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    candidates: list[_BoundaryCandidate] = []
    required_boundary_indices: set[int] = set()
    for boundary_index, diagnostic in enumerate(boundaries):
        if getattr(diagnostic, "boundary_index", None) != boundary_index:
            raise ValueError("direct member boundary indices must be contiguous")
        left_source = source_path[boundary_index]
        right_source = source_path[boundary_index + 1]
        if left_source == right_source:
            continue
        mismatch = getattr(diagnostic, "mismatch", None)
        repair = getattr(mismatch, "transform_repair", None)
        if repair is not None and not isinstance(
            repair,
            DirectMemberBoundaryTransformRepair,
        ):
            raise ValueError("direct member boundary transform repair is invalid")
        required = repair is not None
        if required:
            required_boundary_indices.add(boundary_index)

        def reject_boundary(reason: str) -> None:
            skip(reason)
            if required:
                required_rejection_reasons.setdefault(
                    boundary_index,
                    set(),
                ).add(reason)

        if (
            getattr(diagnostic, "outgoing_eligible_source_index", None)
            != left_source
            or getattr(diagnostic, "incoming_eligible_source_index", None)
            != right_source
        ):
            raise ValueError("direct member boundary source evidence disagrees with path")
        audited_left_geometry = _attention_alignment_geometry(
            diagnostic.outgoing_sample
        )
        audited_right_geometry = _attention_alignment_geometry(
            diagnostic.incoming_sample
        )
        if (
            audited_left_geometry is None
            or audited_right_geometry is None
            or audited_left_geometry[1] != audited_right_geometry[1]
        ):
            reject_boundary("no_unambiguous_face_pair")
            continue
        audited_left, audited_basis = audited_left_geometry
        audited_right, _right_basis = audited_right_geometry
        left_geometry = source_frame_geometries[left_source]
        right_geometry = source_frame_geometries[right_source]
        audited_left = (
            map_box_to_portrait_fill(audited_left, left_geometry)
            if left_geometry is not None
            else audited_left
        )
        audited_right = (
            map_box_to_portrait_fill(audited_right, right_geometry)
            if right_geometry is not None
            else audited_right
        )
        left_time = (
            common_master_start
            + renders[boundary_index].master_start
            + last_frame_times[boundary_index]
        )
        right_time = common_master_start + renders[boundary_index + 1].master_start
        left_sample = geometry_provider(left_source, left_time)
        right_sample = geometry_provider(right_source, right_time)
        if left_sample is None or right_sample is None:
            reject_boundary("dense_boundary_face_missing")
            continue
        if (
            left_sample.alignment_geometry_basis != audited_basis
            or right_sample.alignment_geometry_basis != audited_basis
        ):
            reject_boundary("dense_boundary_face_disagrees_with_audit")
            continue
        left_box = (
            map_box_to_portrait_fill(left_sample.bbox_xywh, left_geometry)
            if left_geometry is not None
            else left_sample.bbox_xywh
        )
        right_box = (
            map_box_to_portrait_fill(right_sample.bbox_xywh, right_geometry)
            if right_geometry is not None
            else right_sample.bbox_xywh
        )
        if not _endpoint_agrees(left_box, audited_left, settings) or not _endpoint_agrees(
            right_box, audited_right, settings
        ):
            reject_boundary("dense_boundary_face_disagrees_with_audit")
            continue
        center_distance = math.hypot(
            _center(left_box)[0] - _center(right_box)[0],
            _center(left_box)[1] - _center(right_box)[1],
        )
        if center_distance > settings.maximum_boundary_center_distance + 1e-9:
            reject_boundary("boundary_position_exceeds_transform_bound")
            continue
        ratio = max(left_box[3], right_box[3]) / max(
            1e-9, min(left_box[3], right_box[3])
        )
        if ratio <= 1.015 and center_distance <= 0.025:
            reject_boundary("already_scale_position_matched")
            continue
        baseline = _mismatch(left_box, right_box)
        if required and repair is not None:
            if repair.alignment_geometry_basis != audited_basis:
                raise ValueError(
                    "required transform geometry basis disagrees with dense audit"
                )
            side = "left" if repair.transformed_side == "outgoing" else "right"
            affected_index = (
                boundary_index if side == "left" else boundary_index + 1
            )
            box, target_box = (
                (left_box, right_box) if side == "left" else (right_box, left_box)
            )
            expected_source = (
                left_source if side == "left" else right_source
            )
            if repair.transformed_source_index != expected_source:
                raise ValueError(
                    "direct member required transform source disagrees with path"
                )
        elif left_box[3] + 1e-9 < right_box[3]:
            side = "left"
            affected_index = boundary_index
            box, target_box = left_box, right_box
        else:
            side = "right"
            affected_index = boundary_index + 1
            box, target_box = right_box, left_box
        if affected_index in blocked:
            alternate_side = "right" if side == "left" else "left"
            alternate_index = (
                boundary_index
                if alternate_side == "left"
                else boundary_index + 1
            )
            if alternate_index in blocked:
                reject_boundary("dissolve_adjacent")
                continue
            side = alternate_side
            affected_index = alternate_index
            box, target_box = (
                (left_box, right_box)
                if side == "left"
                else (right_box, left_box)
            )
        if last_frame_times[affected_index] < 0.55 - 1e-9:
            reject_boundary("window_shorter_than_reference_ramp")
            continue
        affected_source = source_path[affected_index]
        affected_geometry = source_frame_geometries[affected_source]
        lossless_zoom_limit = (
            min(
                settings.maximum_boundary_zoom,
                affected_geometry.lossless_zoom_capacity,
            )
            if affected_geometry is not None
            else settings.maximum_boundary_zoom
        )
        if required and affected_geometry is None:
            reject_boundary("required_transform_source_resolution_missing")
            continue
        if lossless_zoom_limit <= 1.0 + 1e-9:
            reject_boundary("source_resolution_has_no_lossless_transform_headroom")
            continue
        desired_center_shift = min(
            settings.maximum_center_shift,
            math.hypot(
                _center(box)[0] - _center(target_box)[0],
                _center(box)[1] - _center(target_box)[1],
            ),
        )
        pan_headroom_zoom = 1.0 / max(
            1e-9,
            1.0 - 2.0 * desired_center_shift,
        )
        requested_zoom = max(
            1.02,
            target_box[3] / box[3],
            pan_headroom_zoom,
        )
        maximum_candidate_zoom = min(lossless_zoom_limit, requested_zoom)
        if required and repair is not None:
            maximum_candidate_zoom = min(
                lossless_zoom_limit,
                max(maximum_candidate_zoom, repair.zoom),
            )
        qualifying: list[tuple[float, _State]] = []
        required_improvement_fraction = (
            min(settings.minimum_mismatch_improvement_fraction, 0.05)
            if required
            and baseline <= 0.10 + 1e-9
            and center_distance <= 0.05 + 1e-9
            and ratio <= 1.05 + 1e-9
            else settings.minimum_mismatch_improvement_fraction
        )
        zoom_values = [maximum_candidate_zoom]
        while zoom_values[-1] > 1.02 + 1e-9:
            next_zoom = max(1.02, zoom_values[-1] - 0.02)
            if abs(next_zoom - zoom_values[-1]) <= 1e-9:
                break
            zoom_values.append(next_zoom)
        seen_states: set[tuple[float, float, float]] = set()
        for zoom in zoom_values:
            for state in _state_for_match(
                box=box,
                target_box=target_box,
                zoom=zoom,
                settings=shared,
            ):
                state_key = (
                    round(state.zoom, 9),
                    round(state.center_x, 9),
                    round(state.center_y, 9),
                )
                if state_key in seen_states:
                    continue
                seen_states.add(state_key)
                if (
                    _visible_fraction(box, state) + 1e-9
                    < settings.minimum_face_visible_fraction
                ):
                    continue
                projected_left = (
                    _project(left_box, state) if side == "left" else left_box
                )
                projected_right = (
                    _project(right_box, state) if side == "right" else right_box
                )
                transformed = _mismatch(projected_left, projected_right)
                residual_center = math.hypot(
                    _center(projected_left)[0] - _center(projected_right)[0],
                    _center(projected_left)[1] - _center(projected_right)[1],
                )
                if (
                    residual_center
                    > settings.maximum_residual_center_distance + 1e-9
                ):
                    continue
                if transformed > baseline * (
                    1.0 - required_improvement_fraction
                ) + 1e-9:
                    continue
                if transformed + 1e-4 < baseline:
                    qualifying.append((transformed, state))
        if not qualifying:
            reject_boundary("no_improving_contained_transform")
            if required:
                required_rejection_context[boundary_index] = (
                    f"baseline={baseline:.6f},center={center_distance:.6f},"
                    f"height_ratio={ratio:.6f},side={side},"
                    f"maximum_zoom={maximum_candidate_zoom:.6f}"
                )
            continue
        qualifying.sort(key=lambda item: item[0])
        best_score, best_state = qualifying[0]
        candidates.append(
            _BoundaryCandidate(
                boundary_index=boundary_index,
                event_time=boundary_times[boundary_index],
                score=baseline - best_score,
                side=side,
                state=best_state,
                alignment_geometry_basis=audited_basis,
                alternative_states=tuple(state for _score, state in qualifying[1:]),
                required=required,
            )
        )

    quantized_duration = sum(frame_counts) / fps
    maximum_events = int(math.floor(
        quantized_duration * settings.target_events_per_minute / 60.0 + 1e-12
    ))
    entry_states: list[_State | None] = [None] * len(renders)
    exit_states: list[_State | None] = [None] * len(renders)
    tracks: list[EditorialTransformTrack | None] = [None] * len(renders)
    segment_alignment_bases: list[set[str]] = [
        set() for _segment in renders
    ]
    accepted: list[_BoundaryCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            not item.required,
            -item.score,
            item.event_time,
        ),
    ):
        if not candidate.required and len(accepted) >= maximum_events:
            skip("event_density_cap")
            continue
        if not candidate.required and any(
            abs(candidate.event_time - item.event_time)
            < settings.minimum_event_spacing_seconds - 1e-9
            for item in accepted
        ):
            skip("event_spacing_cap")
            continue
        affected_index = (
            candidate.boundary_index
            if candidate.side == "left"
            else candidate.boundary_index + 1
        )
        previous_entry = entry_states[affected_index]
        previous_exit = exit_states[affected_index]
        accepted_state: _State | None = None
        accepted_track: EditorialTransformTrack | None = None
        dense_rejection_reasons: set[str] = set()
        dense_rejection_details: set[str] = set()
        candidate_alignment_bases = set(
            segment_alignment_bases[affected_index]
        )
        candidate_alignment_bases.add(candidate.alignment_geometry_basis)
        review_alignment_basis = (
            next(iter(candidate_alignment_bases))
            if len(candidate_alignment_bases) == 1
            else None
        )
        for state in (candidate.state,) + candidate.alternative_states:
            if candidate.side == "left":
                exit_states[affected_index] = state
            else:
                entry_states[affected_index] = state
            try:
                track = _track_for_states(
                    entry=entry_states[affected_index],
                    exit=exit_states[affected_index],
                    last_frame_time=last_frame_times[affected_index],
                    settings=shared,
                )
            except ValueError:
                track = None
            if track is not None and _track_is_dense_face_safe(
                segment_index=affected_index,
                eligible_source_index=source_path[affected_index],
                segment=renders[affected_index],
                track=track,
                frame_count=frame_counts[affected_index],
                common_master_start=common_master_start,
                geometry_provider=geometry_provider,
                fps=fps,
                config=settings,
                source_frame_geometry=source_frame_geometries[
                    source_path[affected_index]
                ],
                alignment_geometry_basis=review_alignment_basis,
                rejection_callback=dense_rejection_reasons.add,
                rejection_detail_callback=dense_rejection_details.add,
            ):
                accepted_state = state
                accepted_track = track
                break
        if accepted_state is None or accepted_track is None:
            entry_states[affected_index] = previous_entry
            exit_states[affected_index] = previous_exit
            skip("dense_face_containment_or_continuity_failed")
            if candidate.required:
                required_rejection_reasons.setdefault(
                    candidate.boundary_index,
                    set(),
                ).update(
                    dense_rejection_reasons
                    or {"dense_face_containment_or_continuity_failed"}
                )
                if dense_rejection_details:
                    required_rejection_context[
                        candidate.boundary_index
                    ] = ";".join(sorted(dense_rejection_details))
            continue
        tracks[affected_index] = accepted_track
        segment_alignment_bases[affected_index] = candidate_alignment_bases
        accepted.append(
            _BoundaryCandidate(
                boundary_index=candidate.boundary_index,
                event_time=candidate.event_time,
                score=candidate.score,
                side=candidate.side,
                state=accepted_state,
                alignment_geometry_basis=(
                    candidate.alignment_geometry_basis
                ),
                required=candidate.required,
            )
        )

    interior_pulse_segments: set[int] = set()
    if settings.enable_interior_zoom_pulses:
        target_rich_motion_count = min(
            maximum_events,
            math.ceil(
                quantized_duration
                * settings.target_rich_motion_events_per_minute
                / 60.0
                - 1e-12
            ),
        )
        accepted_event_times = [item.event_time for item in accepted]

        def interior_activity_score(segment_index: int) -> float:
            values: list[float] = []
            for boundary_index in (segment_index - 1, segment_index):
                if not 0 <= boundary_index < len(boundaries):
                    continue
                motion = getattr(
                    getattr(boundaries[boundary_index], "mismatch", None),
                    "motion_connection",
                    None,
                )
                if motion is None:
                    continue
                for name in (
                    "outgoing_motion_magnitude",
                    "incoming_motion_magnitude",
                ):
                    value = getattr(motion, name, None)
                    if value is not None and math.isfinite(float(value)):
                        values.append(float(value))
            return max(values, default=0.0)

        pulse_candidates = sorted(
            range(len(renders)),
            key=lambda index: (
                -interior_activity_score(index),
                -last_frame_times[index],
                renders[index].master_start,
                index,
            ),
        )
        for index in pulse_candidates:
            if len(accepted) + len(interior_pulse_segments) >= target_rich_motion_count:
                break
            if tracks[index] is not None or index in blocked:
                continue
            if (
                last_frame_times[index]
                < settings.minimum_interior_pulse_duration_seconds - 1e-9
            ):
                skip("interior_pulse_segment_too_short")
                continue
            event_time = renders[index].master_start + last_frame_times[index] / 2.0
            if any(
                abs(event_time - other_time)
                < settings.minimum_event_spacing_seconds - 1e-9
                for other_time in accepted_event_times
            ):
                skip("interior_pulse_spacing_cap")
                continue
            source_geometry = source_frame_geometries[source_path[index]]
            if source_geometry is None:
                skip("interior_pulse_source_resolution_missing")
                continue
            zoom = min(
                settings.interior_pulse_zoom,
                settings.maximum_boundary_zoom,
                source_geometry.lossless_zoom_capacity,
            )
            if zoom < 1.02 - 1e-9:
                skip("interior_pulse_no_lossless_zoom_headroom")
                continue
            track = EditorialTransformTrack(
                keyframes=(
                    EditorialTransformKeyframe(0.0, 1.0),
                    EditorialTransformKeyframe(
                        last_frame_times[index] / 2.0,
                        zoom,
                    ),
                    EditorialTransformKeyframe(
                        last_frame_times[index],
                        1.0,
                    ),
                ),
                reason=(
                    "official_fancam_dense_subject_lossless_interior_zoom_pulse"
                ),
            )
            if not _track_is_dense_face_safe(
                segment_index=index,
                eligible_source_index=source_path[index],
                segment=renders[index],
                track=track,
                frame_count=frame_counts[index],
                common_master_start=common_master_start,
                geometry_provider=geometry_provider,
                fps=fps,
                config=settings,
                source_frame_geometry=source_geometry,
            ):
                skip("interior_pulse_dense_containment_or_continuity_failed")
                continue
            tracks[index] = track
            interior_pulse_segments.add(index)
            accepted_event_times.append(event_time)

    for index in range(len(renders)):
        if entry_states[index] is None and exit_states[index] is None:
            continue
        track = _track_for_states(
            entry=entry_states[index],
            exit=exit_states[index],
            last_frame_time=last_frame_times[index],
            settings=shared,
        )
        if track is None or not _track_is_dense_face_safe(
            segment_index=index,
            eligible_source_index=source_path[index],
            segment=renders[index],
            track=track,
            frame_count=frame_counts[index],
            common_master_start=common_master_start,
            geometry_provider=geometry_provider,
            fps=fps,
            config=settings,
            source_frame_geometry=source_frame_geometries[source_path[index]],
            alignment_geometry_basis=(
                next(iter(segment_alignment_bases[index]))
                if len(segment_alignment_bases[index]) == 1
                else None
            ),
        ):
            raise ValueError("accepted direct member camera failed final dense review")
        tracks[index] = track

    accepted_segments = {
        item.boundary_index if item.side == "left" else item.boundary_index + 1
        for item in accepted
    }
    accepted_boundary_indices = tuple(
        sorted(item.boundary_index for item in accepted)
    )
    missing_required = tuple(
        sorted(required_boundary_indices.difference(accepted_boundary_indices))
    )
    if missing_required:
        rejection_counts: dict[str, int] = {}
        for boundary_index in missing_required:
            reasons = required_rejection_reasons.get(
                boundary_index,
                {"required_candidate_missing_without_reason"},
            )
            for reason in reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        rejection_summary = ",".join(
            f"{reason}={count}"
            for reason, count in sorted(rejection_counts.items())
        )
        rejection_details = "|".join(
            f"{boundary_index}:"
            + ",".join(
                sorted(
                    required_rejection_reasons.get(
                        boundary_index,
                        {"required_candidate_missing_without_reason"},
                    )
                )
            )
            + (
                f"[{required_rejection_context[boundary_index]}]"
                if boundary_index in required_rejection_context
                else ""
            )
            for boundary_index in missing_required
        )
        raise ValueError(
            "required direct member transform failed dense review at boundaries "
            + ",".join(str(index) for index in missing_required)
            + "; required rejection counts "
            + rejection_summary
            + "; required rejection details "
            + rejection_details
        )
    return EditorialMotionPlan(
        segments=tuple(
            EditorialMotionSegment(
                segment_index=index,
                source_index=source_path[index],
                track=tracks[index],
                reasons=(
                    (
                        "official_fancam_dense_face_subject_"
                        "scale_position_match",
                    )
                    if index in accepted_segments
                    else (
                        "official_fancam_dense_subject_"
                        "lossless_interior_zoom_pulse",
                    )
                    if index in interior_pulse_segments
                    else ("no_safe_dense_face_editorial_motion",)
                ),
            )
            for index in range(len(renders))
        ),
        boundary_match_count=len(accepted),
        zoom_event_count=len(accepted) + len(interior_pulse_segments),
        pan_event_count=sum(
            abs(item.state.center_x - 0.5) > 1e-6
            or abs(item.state.center_y - 0.5) > 1e-6
            for item in accepted
        ),
        roll_event_count=0,
        interior_zoom_pulse_count=len(interior_pulse_segments),
        blocked_dissolve_segment_count=len(blocked),
        skipped_boundary_reasons=tuple(sorted(skipped.items())),
        analysis_basis=(
            "official_target_fancam_face_or_verified_full_subject_dense_"
            "output_frame_subject_relative_face_track_adaptive_review_v7"
        ),
        accepted_boundary_indices=accepted_boundary_indices,
    )


__all__ = [
    "DirectMemberDenseFaceProvider",
    "DirectMemberEditorialMotionConfig",
    "DirectMemberFaceSample",
    "build_direct_member_editorial_motion_plan",
]
