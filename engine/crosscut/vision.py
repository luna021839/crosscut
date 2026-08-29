"""Lightweight visual analysis primitives for the IVE cross-cut editor.

The module intentionally does not load a detector model.  Detector/tracker output
is represented by :class:`TrackObservation`, which keeps the expensive model
choice outside the editing engine and makes the camera planner testable with
synthetic observations.

All boxes and crop windows use normalized source-frame coordinates.  Bounding
boxes are ``(left, top, right, bottom)`` while crop windows are
``(left, top, width, height)``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import cv2
import numpy as np

from .ive import IVEMember
from .source_scalar_cache import (
    PersistentSourceScalarCache,
    PersistentSourceScalarCacheStorageUnavailableError,
)

IVE_MEMBER_STABLE_IDS: dict[IVEMember, str] = {
    IVEMember.ANYUJIN: "an_yujin",
    IVEMember.GAEUL: "gaeul",
    IVEMember.REI: "rei",
    IVEMember.JANGWONYOUNG: "jang_wonyoung",
    IVEMember.LIZ: "liz",
    IVEMember.LEESEO: "leeseo",
}


def ive_member_stable_id(member: IVEMember | str) -> str:
    """Return a locale-independent ID without introducing a second enum."""

    return IVE_MEMBER_STABLE_IDS[IVEMember.parse(member)]


def _unit_interval(value: float, name: str) -> float:
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value!r}")
    return numeric


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """A detector/tracker observation for one person in one frame.

    ``track_id`` is the short-lived ID emitted by the tracker. ``identity_id``
    preserves an opaque ID from an external re-identification model. ``member``
    is only set when that ID has been resolved to one of IVE's six members.
    Keeping these fields separate prevents a temporary tracker ID from being
    mistaken for a durable member identity.
    """

    frame_index: int
    bbox_xyxy: tuple[float, float, float, float]
    detection_confidence: float
    member: IVEMember | str | None = None
    member_confidence: float = 0.0
    source_id: str = ""
    track_id: str | int | None = None
    identity_id: str | None = None
    timestamp_seconds: float | None = None
    occluded: bool = False

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain four normalized values")
        left, top, right, bottom = (float(value) for value in self.bbox_xyxy)
        if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
            raise ValueError(
                "bbox_xyxy must satisfy 0 <= left < right <= 1 and "
                "0 <= top < bottom <= 1"
            )
        object.__setattr__(self, "bbox_xyxy", (left, top, right, bottom))
        object.__setattr__(
            self,
            "detection_confidence",
            _unit_interval(self.detection_confidence, "detection_confidence"),
        )
        object.__setattr__(
            self,
            "member_confidence",
            _unit_interval(self.member_confidence, "member_confidence"),
        )
        if self.member is not None:
            object.__setattr__(self, "member", IVEMember.parse(self.member))
        if self.timestamp_seconds is not None and self.timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must be non-negative")

    @property
    def center(self) -> tuple[float, float]:
        left, top, right, bottom = self.bbox_xyxy
        return ((left + right) / 2.0, (top + bottom) / 2.0)

    @property
    def size(self) -> tuple[float, float]:
        left, top, right, bottom = self.bbox_xyxy
        return (right - left, bottom - top)

    @property
    def tracking_confidence(self) -> float:
        """Conservative joint person-detection and identity confidence."""

        return min(self.detection_confidence, self.member_confidence)

    def confidence_for(self, member: IVEMember | str) -> float:
        target = IVEMember.parse(member)
        if self.occluded or self.member is not target:
            return 0.0
        return self.tracking_confidence


@dataclass(frozen=True, slots=True)
class SceneChange:
    """A likely first frame of a new camera shot."""

    frame_index: int
    score: float
    frame_difference: float
    histogram_distance: float


@dataclass(frozen=True, slots=True)
class SceneDetectionConfig:
    """Thresholds for model-free scene-cut detection."""

    # A combined score alone let fast pans and concert lighting pass on frame
    # difference even when the tonal distribution stayed continuous.  The
    # component gates were calibrated on 74 source-attributed IVE boundaries:
    # 41/42 real camera cuts retained while movement/light false positives fell
    # from 32 to 7 at the default combined threshold.
    threshold: float = 0.50
    minimum_frame_difference: float = 0.60
    minimum_histogram_distance: float = 0.20
    min_gap_frames: int = 8
    frame_difference_weight: float = 0.65
    histogram_weight: float = 0.35
    difference_gain: float = 3.0
    analysis_size: tuple[int, int] = (160, 90)
    histogram_bins: int = 32

    def __post_init__(self) -> None:
        _unit_interval(self.threshold, "threshold")
        _unit_interval(self.minimum_frame_difference, "minimum_frame_difference")
        _unit_interval(
            self.minimum_histogram_distance,
            "minimum_histogram_distance",
        )
        if self.min_gap_frames < 1:
            raise ValueError("min_gap_frames must be at least 1")
        if self.frame_difference_weight < 0 or self.histogram_weight < 0:
            raise ValueError("scene score weights cannot be negative")
        if self.frame_difference_weight + self.histogram_weight <= 0:
            raise ValueError("at least one scene score weight must be positive")
        if self.difference_gain <= 0:
            raise ValueError("difference_gain must be positive")
        if min(self.analysis_size) <= 0:
            raise ValueError("analysis_size dimensions must be positive")
        if self.histogram_bins < 2:
            raise ValueError("histogram_bins must be at least 2")


def _scene_features(
    frame: np.ndarray, config: SceneDetectionConfig
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("each frame must be a non-empty numpy array")
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 1:
        gray = frame[:, :, 0]
    elif frame.ndim == 3 and frame.shape[2] in (3, 4):
        color_code = cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(frame, color_code)
    else:
        raise ValueError(f"unsupported frame shape: {frame.shape}")
    resized = cv2.resize(gray, config.analysis_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    histogram = cv2.calcHist(
        [resized], [0], None, [config.histogram_bins], [0, 256]
    ).astype(np.float32)
    cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
    return normalized, histogram


def detect_scene_changes(
    frames: Iterable[np.ndarray],
    *,
    config: SceneDetectionConfig | None = None,
    frame_indices: Iterable[int] | None = None,
) -> tuple[SceneChange, ...]:
    """Detect hard camera changes using frame difference and histogram distance.

    Candidates above the threshold are subjected to score-based non-maximum
    suppression.  Therefore, when several adjacent frames react to a dissolve or
    flash, the strongest frame is retained rather than whichever was seen first.
    """

    settings = config or SceneDetectionConfig()
    index_iterator: Iterator[int] = (
        iter(frame_indices) if frame_indices is not None else iter(range(2**63 - 1))
    )
    previous_gray: np.ndarray | None = None
    previous_histogram: np.ndarray | None = None
    candidates: list[SceneChange] = []

    for frame in frames:
        try:
            frame_index = int(next(index_iterator))
        except StopIteration as exc:
            raise ValueError("frame_indices is shorter than frames") from exc
        gray, histogram = _scene_features(frame, settings)
        if previous_gray is not None and previous_histogram is not None:
            raw_difference = float(np.mean(np.abs(gray - previous_gray)))
            frame_difference = min(1.0, raw_difference * settings.difference_gain)
            histogram_distance = float(
                cv2.compareHist(
                    previous_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA
                )
            )
            weight_total = (
                settings.frame_difference_weight + settings.histogram_weight
            )
            score = (
                settings.frame_difference_weight * frame_difference
                + settings.histogram_weight * histogram_distance
            ) / weight_total
            if (
                score >= settings.threshold
                and frame_difference >= settings.minimum_frame_difference
                and histogram_distance >= settings.minimum_histogram_distance
            ):
                candidates.append(
                    SceneChange(
                        frame_index=frame_index,
                        score=score,
                        frame_difference=frame_difference,
                        histogram_distance=histogram_distance,
                    )
                )
        previous_gray = gray
        previous_histogram = histogram

    # Pick strongest changes first, then restore chronological order.
    retained: list[SceneChange] = []
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.frame_index)):
        if all(
            abs(candidate.frame_index - existing.frame_index)
            >= settings.min_gap_frames
            for existing in retained
        ):
            retained.append(candidate)
    return tuple(sorted(retained, key=lambda item: item.frame_index))


def iter_video_frames(
    path: str | Path, *, sample_every: int = 1
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield decoded video frames with their original zero-based frame indices."""

    if sample_every < 1:
        raise ValueError("sample_every must be at least 1")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")
    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % sample_every == 0:
                yield frame_index, frame
            frame_index += 1
    finally:
        capture.release()


def detect_scene_changes_in_video(
    path: str | Path,
    *,
    config: SceneDetectionConfig | None = None,
    sample_every: int = 1,
) -> tuple[SceneChange, ...]:
    """Convenience wrapper that keeps reported indices in source-frame units."""

    indexed_frames = iter_video_frames(path, sample_every=sample_every)

    def frames_only() -> Iterator[np.ndarray]:
        for frame_index, frame in indexed_frames:
            indices.append(frame_index)
            yield frame

    def indices_in_lockstep() -> Iterator[int]:
        cursor = 0
        while True:
            # ``detect_scene_changes`` advances ``frames_only`` before asking for
            # its index, so one new item is available here on every iteration.
            if cursor >= len(indices):
                return
            yield indices[cursor]
            cursor += 1

    indices: list[int] = []
    # ``indices`` is populated lazily in lockstep with frame consumption.
    return detect_scene_changes(
        frames_only(), config=config, frame_indices=indices_in_lockstep()
    )


def _scene_detection_analyzer_identity(
    config: SceneDetectionConfig,
    *,
    sample_every: int,
) -> dict[str, Any]:
    return {
        "analysis_contract": "ive_component_gated_scene_changes_v2",
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "opencv_version": str(cv2.__version__),
        "sample_every": sample_every,
        "config": {
            "threshold": config.threshold,
            "minimum_frame_difference": config.minimum_frame_difference,
            "minimum_histogram_distance": (
                config.minimum_histogram_distance
            ),
            "min_gap_frames": config.min_gap_frames,
            "frame_difference_weight": config.frame_difference_weight,
            "histogram_weight": config.histogram_weight,
            "difference_gain": config.difference_gain,
            "analysis_size": list(config.analysis_size),
            "histogram_bins": config.histogram_bins,
        },
    }


def _scene_changes_from_scalar_payload(
    payload: Mapping[str, Any],
) -> tuple[SceneChange, ...]:
    if set(payload) != {"changes"}:
        raise ValueError("scene cache payload schema does not match")
    raw_changes = payload["changes"]
    if not isinstance(raw_changes, list):
        raise TypeError("scene cache changes must be a list")
    changes: list[SceneChange] = []
    for raw in raw_changes:
        if not isinstance(raw, dict) or set(raw) != {
            "frame_index",
            "score",
            "frame_difference",
            "histogram_distance",
        }:
            raise ValueError("scene cache change schema does not match")
        frame_index = raw["frame_index"]
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise ValueError("scene cache frame index is invalid")
        measurement_keys = (
            "score",
            "frame_difference",
            "histogram_distance",
        )
        if any(
            isinstance(raw[key], bool)
            or not isinstance(raw[key], (int, float))
            for key in measurement_keys
        ):
            raise ValueError("scene cache measurement type is invalid")
        measurements = tuple(float(raw[key]) for key in measurement_keys)
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in measurements
        ):
            raise ValueError("scene cache measurement is invalid")
        changes.append(
            SceneChange(
                frame_index=frame_index,
                score=measurements[0],
                frame_difference=measurements[1],
                histogram_distance=measurements[2],
            )
        )
    if any(
        left.frame_index >= right.frame_index
        for left, right in zip(changes, changes[1:])
    ):
        raise ValueError("scene cache changes are not strictly chronological")
    return tuple(changes)


def detect_scene_changes_in_video_cached(
    path: str | Path,
    *,
    config: SceneDetectionConfig | None = None,
    sample_every: int = 1,
    source_cache_directory: Path | None = None,
    derived_cache_directory: Path | None = None,
) -> tuple[tuple[SceneChange, ...], dict[str, Any]]:
    """Reuse completed per-source scene scans and recompute on any mismatch."""

    settings = config or SceneDetectionConfig()
    if source_cache_directory is None and derived_cache_directory is None:
        return (
            detect_scene_changes_in_video(
                path,
                config=settings,
                sample_every=sample_every,
            ),
            {
                "enabled": False,
                "analysis_reused": False,
                "raw_media_persisted": False,
            },
        )
    cache_source_directory = (
        source_cache_directory or derived_cache_directory
    )
    assert cache_source_directory is not None
    try:
        cache = PersistentSourceScalarCache(
            source_cache_directory=cache_source_directory,
            derived_cache_directory=derived_cache_directory,
            source_path=Path(path),
            namespace="scene-changes-v2",
            analyzer=_scene_detection_analyzer_identity(
                settings,
                sample_every=sample_every,
            ),
        )
    except PersistentSourceScalarCacheStorageUnavailableError:
        raise
    except (OSError, TypeError, ValueError):
        if derived_cache_directory is not None:
            raise
        return (
            detect_scene_changes_in_video(
                path,
                config=settings,
                sample_every=sample_every,
            ),
            {
                "enabled": False,
                "analysis_reused": False,
                "setup_failed_closed": True,
                "raw_media_persisted": False,
            },
        )
    semantic_invalid = 0
    cached = cache.get()
    if cached is not None:
        try:
            changes = _scene_changes_from_scalar_payload(cached)
        except (TypeError, ValueError):
            semantic_invalid = 1
        else:
            return (
                changes,
                {
                    **cache.diagnostics(),
                    "enabled": True,
                    "analysis_reused": True,
                    "semantic_invalid_record_count": 0,
                },
            )
    changes = detect_scene_changes_in_video(
        path,
        config=settings,
        sample_every=sample_every,
    )
    cache.put(
        {
            "changes": [
                {
                    "frame_index": change.frame_index,
                    "score": change.score,
                    "frame_difference": change.frame_difference,
                    "histogram_distance": change.histogram_distance,
                }
                for change in changes
            ]
        }
    )
    return (
        changes,
        {
            **cache.diagnostics(),
            "enabled": True,
            "analysis_reused": False,
            "semantic_invalid_record_count": semantic_invalid,
        },
    )


@dataclass(frozen=True, slots=True)
class CropWindow:
    """Normalized crop rectangle, stored as left/top/width/height."""

    left: float
    top: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.width, self.height)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("crop values must be finite")
        if not (
            0.0 <= self.left <= 1.0
            and 0.0 <= self.top <= 1.0
            and self.width > 0.0
            and self.height > 0.0
            and self.left + self.width <= 1.0 + 1e-9
            and self.top + self.height <= 1.0 + 1e-9
        ):
            raise ValueError("crop window must fit inside the normalized source frame")

    @property
    def center(self) -> tuple[float, float]:
        return (self.left + self.width / 2.0, self.top + self.height / 2.0)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        return (self.left, self.top, self.width, self.height)

    @classmethod
    def from_center(
        cls, center_x: float, center_y: float, width: float, height: float
    ) -> "CropWindow":
        width = min(1.0, max(float(width), 1e-6))
        height = min(1.0, max(float(height), 1e-6))
        left = min(max(float(center_x) - width / 2.0, 0.0), 1.0 - width)
        top = min(max(float(center_y) - height / 2.0, 0.0), 1.0 - height)
        return cls(left=left, top=top, width=width, height=height)


class CameraFallback(str, Enum):
    NONE = "none"
    HOLD_LAST = "hold_last"
    GROUP_WIDE_REQUIRED = "group_wide_required"


@dataclass(frozen=True, slots=True)
class VirtualCameraConfig:
    """Behaviour of the target-member virtual camera.

    Speeds are normalized source-frame units per source frame.  This makes the
    planner independent of resolution; a caller can scale them for a known FPS
    if desired.
    """

    output_aspect_ratio: float = 9.0 / 16.0
    source_aspect_ratio: float = 16.0 / 9.0
    min_detection_confidence: float = 0.50
    min_member_confidence: float = 0.72
    lookahead_frames: int = 8
    hold_frames: int = 5
    deadzone_x: float = 0.12
    deadzone_y: float = 0.08
    max_center_speed: float = 0.025
    max_zoom_speed: float = 0.018
    smoothing: float = 0.42
    horizontal_padding: float = 1.55
    vertical_padding: float = 1.35
    min_crop_height: float = 0.38
    max_crop_height: float = 1.0

    def __post_init__(self) -> None:
        if self.output_aspect_ratio <= 0:
            raise ValueError("output_aspect_ratio must be positive")
        if self.source_aspect_ratio <= 0:
            raise ValueError("source_aspect_ratio must be positive")
        _unit_interval(self.min_detection_confidence, "min_detection_confidence")
        _unit_interval(self.min_member_confidence, "min_member_confidence")
        if self.lookahead_frames < 0 or self.hold_frames < 0:
            raise ValueError("lookahead_frames and hold_frames cannot be negative")
        if self.deadzone_x < 0 or self.deadzone_y < 0:
            raise ValueError("deadzones cannot be negative")
        if self.max_center_speed <= 0 or self.max_zoom_speed <= 0:
            raise ValueError("camera speed limits must be positive")
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        if self.horizontal_padding < 1 or self.vertical_padding < 1:
            raise ValueError("camera padding must be at least 1")
        if not 0 < self.min_crop_height <= self.max_crop_height <= 1:
            raise ValueError("crop height bounds must satisfy 0 < min <= max <= 1")

    @property
    def normalized_crop_aspect(self) -> float:
        """Required normalized width/height for the requested pixel aspect.

        Normalized source coordinates are not square when the source is 16:9.
        A portrait crop from a 16:9 source therefore needs a normalized ratio
        of ``(9/16) / (16/9)``, rather than simply ``9/16``.
        """

        return self.output_aspect_ratio / self.source_aspect_ratio


@dataclass(frozen=True, slots=True)
class VirtualCameraFrame:
    frame_index: int
    source_id: str
    crop: CropWindow | None
    tracking_confidence: float
    accepted_observation: bool
    fallback: CameraFallback
    observation: TrackObservation | None = None

    @property
    def safe_for_target_crop(self) -> bool:
        if (
            not self.accepted_observation
            or self.crop is None
            or self.observation is None
        ):
            return False
        # A valid identity alone is not enough after a sudden camera cut: the
        # speed-limited crop may still be travelling toward the target.  Expose
        # that state as unsafe so timeline generation can use a group wide shot.
        center_x, center_y = self.observation.center
        return (
            self.crop.left <= center_x <= self.crop.left + self.crop.width
            and self.crop.top <= center_y <= self.crop.top + self.crop.height
        )


def _eligible_observation(
    observation: TrackObservation,
    target: IVEMember,
    settings: VirtualCameraConfig,
) -> bool:
    return (
        not observation.occluded
        and observation.member is target
        and observation.detection_confidence >= settings.min_detection_confidence
        and observation.member_confidence >= settings.min_member_confidence
    )


def _best_observations_by_frame(
    observations: Iterable[TrackObservation],
    target: IVEMember,
    settings: VirtualCameraConfig,
) -> tuple[dict[int, TrackObservation], str]:
    grouped: defaultdict[int, list[TrackObservation]] = defaultdict(list)
    source_ids: set[str] = set()
    for observation in observations:
        source_ids.add(observation.source_id)
        if _eligible_observation(observation, target, settings):
            grouped[observation.frame_index].append(observation)
    if len(source_ids) > 1:
        raise ValueError("plan_virtual_camera expects observations from one source")
    source_id = next(iter(source_ids), "")
    return (
        {
            frame_index: max(
                choices,
                key=lambda observation: (
                    observation.tracking_confidence,
                    observation.detection_confidence,
                    str(observation.track_id),
                ),
            )
            for frame_index, choices in grouped.items()
        },
        source_id,
    )


def _lookahead_box(
    frame_index: int,
    accepted: dict[int, TrackObservation],
    settings: VirtualCameraConfig,
) -> tuple[float, float, float, float]:
    observations: list[tuple[int, TrackObservation]] = []
    for offset in range(settings.lookahead_frames + 1):
        observation = accepted.get(frame_index + offset)
        if observation is not None:
            observations.append((offset, observation))
    if not observations:
        raise ValueError("lookahead requires at least the current accepted observation")
    weights = np.asarray(
        [
            observation.tracking_confidence / (1.0 + offset)
            for offset, observation in observations
        ],
        dtype=np.float64,
    )
    weights /= float(np.sum(weights))
    centers = np.asarray([observation.center for _, observation in observations])
    sizes = np.asarray([observation.size for _, observation in observations])
    center_x, center_y = np.sum(centers * weights[:, None], axis=0)
    width, height = np.sum(sizes * weights[:, None], axis=0)
    return (float(center_x), float(center_y), float(width), float(height))


def _desired_crop_size(
    box_width: float, box_height: float, settings: VirtualCameraConfig
) -> tuple[float, float]:
    aspect = settings.normalized_crop_aspect
    required_height = max(
        box_height * settings.vertical_padding,
        box_width * settings.horizontal_padding / aspect,
        settings.min_crop_height,
    )
    max_height_for_aspect = min(1.0, 1.0 / aspect)
    height = min(required_height, settings.max_crop_height, max_height_for_aspect)
    return (height * aspect, height)


def _deadzone_target(
    current: float, desired: float, deadzone: float
) -> float:
    difference = desired - current
    if abs(difference) <= deadzone:
        return current
    return desired - float(np.sign(difference)) * deadzone


def _limited_step(current: float, desired: float, limit: float) -> float:
    return current + float(np.clip(desired - current, -limit, limit))


def plan_virtual_camera(
    observations: Sequence[TrackObservation],
    *,
    target_member: IVEMember | str,
    config: VirtualCameraConfig | None = None,
    frame_indices: Iterable[int] | None = None,
) -> tuple[VirtualCameraFrame, ...]:
    """Plan a smooth crop for one IVE member in one source video.

    A member crop is only marked safe when both person and identity gates pass.
    Brief gaps hold the last stable crop without claiming the observation is
    safe.  Longer gaps return ``crop=None`` with ``GROUP_WIDE_REQUIRED`` so the
    timeline optimiser can cut to another source or a group-wide shot instead
    of silently following the wrong member.
    """

    settings = config or VirtualCameraConfig()
    target = IVEMember.parse(target_member)
    accepted, source_id = _best_observations_by_frame(observations, target, settings)

    if frame_indices is None:
        if not observations:
            return ()
        first_frame = min(observation.frame_index for observation in observations)
        last_frame = max(observation.frame_index for observation in observations)
        planned_indices = tuple(range(first_frame, last_frame + 1))
    else:
        planned_indices = tuple(int(index) for index in frame_indices)
        if any(index < 0 for index in planned_indices):
            raise ValueError("frame_indices must be non-negative")
        if any(right <= left for left, right in zip(planned_indices, planned_indices[1:])):
            raise ValueError("frame_indices must be strictly increasing")

    result: list[VirtualCameraFrame] = []
    previous_crop: CropWindow | None = None
    previous_frame: int | None = None
    last_accepted_frame: int | None = None

    for frame_index in planned_indices:
        observation = accepted.get(frame_index)
        if observation is None:
            gap = (
                frame_index - last_accepted_frame
                if last_accepted_frame is not None
                else settings.hold_frames + 1
            )
            if previous_crop is not None and gap <= settings.hold_frames:
                crop = previous_crop
                fallback = CameraFallback.HOLD_LAST
            else:
                crop = None
                fallback = CameraFallback.GROUP_WIDE_REQUIRED
                # Re-enter on the newly verified subject rather than dragging an
                # obsolete crop across the frame after a long wide-shot fallback.
                previous_crop = None
            result.append(
                VirtualCameraFrame(
                    frame_index=frame_index,
                    source_id=source_id,
                    crop=crop,
                    tracking_confidence=0.0,
                    accepted_observation=False,
                    fallback=fallback,
                    observation=None,
                )
            )
            previous_frame = frame_index
            continue

        center_x, center_y, box_width, box_height = _lookahead_box(
            frame_index, accepted, settings
        )
        desired_width, desired_height = _desired_crop_size(
            box_width, box_height, settings
        )

        if previous_crop is None:
            crop = CropWindow.from_center(
                center_x, center_y, desired_width, desired_height
            )
        else:
            elapsed = (
                max(1, frame_index - previous_frame)
                if previous_frame is not None
                else 1
            )
            current_x, current_y = previous_crop.center
            zone_x = settings.deadzone_x * previous_crop.width
            zone_y = settings.deadzone_y * previous_crop.height
            target_x = _deadzone_target(current_x, center_x, zone_x)
            target_y = _deadzone_target(current_y, center_y, zone_y)
            smoothed_x = current_x + settings.smoothing * (target_x - current_x)
            smoothed_y = current_y + settings.smoothing * (target_y - current_y)
            new_x = _limited_step(
                current_x, smoothed_x, settings.max_center_speed * elapsed
            )
            new_y = _limited_step(
                current_y, smoothed_y, settings.max_center_speed * elapsed
            )
            smoothed_width = previous_crop.width + settings.smoothing * (
                desired_width - previous_crop.width
            )
            smoothed_height = previous_crop.height + settings.smoothing * (
                desired_height - previous_crop.height
            )
            new_height = _limited_step(
                previous_crop.height,
                smoothed_height,
                settings.max_zoom_speed * elapsed,
            )
            # Recompute width from the configured aspect to avoid cumulative drift.
            new_width = new_height * settings.normalized_crop_aspect
            crop = CropWindow.from_center(new_x, new_y, new_width, new_height)

        result.append(
            VirtualCameraFrame(
                frame_index=frame_index,
                source_id=source_id,
                crop=crop,
                tracking_confidence=observation.tracking_confidence,
                accepted_observation=True,
                fallback=CameraFallback.NONE,
                observation=observation,
            )
        )
        previous_crop = crop
        previous_frame = frame_index
        last_accepted_frame = frame_index

    return tuple(result)


__all__ = [
    "CameraFallback",
    "CropWindow",
    "IVE_MEMBER_STABLE_IDS",
    "IVEMember",
    "SceneChange",
    "SceneDetectionConfig",
    "TrackObservation",
    "VirtualCameraConfig",
    "VirtualCameraFrame",
    "detect_scene_changes",
    "detect_scene_changes_in_video",
    "iter_video_frames",
    "ive_member_stable_id",
    "plan_virtual_camera",
]
