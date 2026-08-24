"""Confidence-gated, manually anchored IVE member tracking.

This module deliberately solves a smaller problem than automatic member
re-identification.  A user identifies the requested IVE member once in the
first frame of a source, and a short-lived OpenCV tracker follows that exact
continuous track.  The first failed update, scene cut, or confidence-gate
failure makes the track terminal: it may hold the last box for a few frames,
but it never accepts a later box as a re-identification.  That invariant is
important for member edits because a wide-shot fallback is preferable to a
confident crop of the wrong member.

Bounding boxes are normalized ``(left, top, right, bottom)`` values throughout
the public API.  Tracker backends use pixel ``(x, y, width, height)`` values to
match OpenCV's interface.  A backend can be injected, making the state machine
fully testable even on OpenCV builds without contrib trackers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import exp, hypot, isfinite, log
from typing import Callable, Iterable, Protocol, Sequence, runtime_checkable

import cv2
import numpy as np

from .ive import IVEMember


NormalizedBox = tuple[float, float, float, float]
PixelBox = tuple[float, float, float, float]


def _probability(value: float, name: str) -> float:
    numeric = float(value)
    if not isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return numeric


def _normalized_box(value: Sequence[float], name: str = "bbox_xyxy") -> NormalizedBox:
    if len(value) != 4:
        raise ValueError(f"{name} must contain four values")
    left, top, right, bottom = (float(component) for component in value)
    if not all(isfinite(component) for component in (left, top, right, bottom)):
        raise ValueError(f"{name} values must be finite")
    if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
        raise ValueError(
            f"{name} must satisfy 0 <= left < right <= 1 and "
            "0 <= top < bottom <= 1"
        )
    return (left, top, right, bottom)


@dataclass(frozen=True, slots=True)
class MemberAnchor:
    """A user-confirmed member box in one source frame."""

    frame_index: int
    bbox_xyxy: NormalizedBox
    member: IVEMember | str
    source_id: str

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise ValueError("frame_index must be an integer")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        object.__setattr__(self, "bbox_xyxy", _normalized_box(self.bbox_xyxy))
        object.__setattr__(self, "member", IVEMember.parse(self.member))
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id cannot be empty")


class TrackingState(str, Enum):
    TRACKED = "tracked"
    HOLDING = "holding"
    LOST = "lost"


class TrackingFailure(str, Enum):
    BACKEND_INITIALIZATION = "backend_initialization"
    BACKEND_UPDATE = "backend_update"
    CONFIDENCE_GATE = "confidence_gate"
    SCENE_CHANGE = "scene_change"


@dataclass(frozen=True, slots=True)
class TrackerUpdate:
    """One backend update in OpenCV pixel coordinates."""

    success: bool
    bbox_xywh: PixelBox | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise ValueError("success must be a bool")
        if self.bbox_xywh is not None:
            if len(self.bbox_xywh) != 4:
                raise ValueError("bbox_xywh must contain four values")
            box = tuple(float(component) for component in self.bbox_xywh)
            if not all(isfinite(component) for component in box):
                raise ValueError("bbox_xywh values must be finite")
            object.__setattr__(self, "bbox_xywh", box)
        if self.success and self.bbox_xywh is None:
            raise ValueError("a successful update needs bbox_xywh")
        if self.confidence is not None:
            object.__setattr__(
                self, "confidence", _probability(self.confidence, "confidence")
            )


@runtime_checkable
class TrackerBackend(Protocol):
    """Small injectable surface shared by OpenCV and synthetic trackers."""

    @property
    def name(self) -> str: ...

    def initialize(self, frame: np.ndarray, bbox_xywh: PixelBox) -> bool: ...

    def update(self, frame: np.ndarray) -> TrackerUpdate: ...


class TrackerUnavailableError(RuntimeError):
    """Raised when the installed OpenCV has no supported tracker factory."""


class OpenCVTrackerBackend:
    """Adapter for OpenCV tracker objects.

    Most OpenCV trackers expose only a success flag, not a calibrated score.
    The enclosing state machine therefore supplies the appearance and motion
    confidence gates and treats the backend confidence as unknown.
    """

    def __init__(self, tracker: object, name: str) -> None:
        self._tracker = tracker
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def initialize(self, frame: np.ndarray, bbox_xywh: PixelBox) -> bool:
        # OpenCV 4.10's Python bindings reject floating-point tuples here even
        # though update() returns floats.  Round only at this adapter boundary;
        # the public normalized anchor remains resolution independent.
        integer_box = tuple(int(round(value)) for value in bbox_xywh)
        if integer_box[2] <= 0 or integer_box[3] <= 0:
            return False
        result = self._tracker.init(frame, integer_box)
        # Some OpenCV Python bindings return None after a successful init.
        return result is None or bool(result)

    def update(self, frame: np.ndarray) -> TrackerUpdate:
        success, bbox = self._tracker.update(frame)
        if not success:
            return TrackerUpdate(success=False)
        return TrackerUpdate(
            success=True,
            bbox_xywh=tuple(float(value) for value in bbox),
            confidence=None,
        )


def _opencv_tracker_factories(name: str) -> tuple[Callable[[], object], ...]:
    factory_name = f"Tracker{name}_create"
    candidates = (
        getattr(cv2, factory_name, None),
        getattr(getattr(cv2, "legacy", None), factory_name, None),
    )
    factories: list[Callable[[], object]] = []
    seen: set[int] = set()
    for candidate in candidates:
        if callable(candidate) and id(candidate) not in seen:
            seen.add(id(candidate))
            factories.append(candidate)
    return tuple(factories)


def _tracker_smoke_frame() -> tuple[np.ndarray, PixelBox]:
    """Return a tiny deterministic texture and an interior tracking box."""

    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    for y in range(16, 80, 8):
        for x in range(16, 80, 8):
            frame[y : y + 8, x : x + 8] = (
                40 + (x * 3) % 200,
                50 + (y * 2) % 200,
                220 if (x // 8 + y // 8) % 2 else 80,
            )
    cv2.rectangle(frame, (24, 24), (64, 64), (255, 255, 255), 2)
    cv2.line(frame, (24, 24), (64, 64), (0, 255, 0), 2)
    cv2.line(frame, (64, 24), (24, 64), (255, 0, 0), 2)
    return (frame, (24.0, 24.0, 40.0, 40.0))


def _tracker_factory_passes_smoke(factory: Callable[[], object]) -> bool:
    """Verify factory creation, initialization, and one real update."""

    try:
        tracker = factory()
        frame, box = _tracker_smoke_frame()
        initialized = tracker.init(frame, tuple(int(value) for value in box))
        if initialized is not None and not bool(initialized):
            return False
        updated = tracker.update(frame.copy())
        if not isinstance(updated, tuple) or len(updated) != 2:
            return False
        success, updated_box = updated
        if not bool(success) or updated_box is None or len(updated_box) != 4:
            return False
        values = tuple(float(value) for value in updated_box)
        return all(isfinite(value) for value in values) and values[2] > 0 and values[3] > 0
    except Exception:
        # OpenCV factories can be present in the Python namespace while their
        # implementation is unavailable or broken. Such a backend must not be
        # advertised and must not prevent trying the next preferred backend.
        return False


def available_opencv_trackers() -> tuple[str, ...]:
    """Return trackers that pass create/init/update smoke in quality order."""

    return tuple(
        name
        for name in ("CSRT", "KCF", "MIL")
        if any(
            _tracker_factory_passes_smoke(factory)
            for factory in _opencv_tracker_factories(name)
        )
    )


def create_opencv_tracker_backend(
    preferred: Sequence[str] = ("CSRT", "KCF", "MIL"),
) -> OpenCVTrackerBackend:
    """Create the first available OpenCV tracker from ``preferred``."""

    for raw_name in preferred:
        name = str(raw_name).strip().upper()
        if name not in {"CSRT", "KCF", "MIL"}:
            continue
        for factory in _opencv_tracker_factories(name):
            if not _tracker_factory_passes_smoke(factory):
                continue
            try:
                return OpenCVTrackerBackend(factory(), name)
            except Exception:
                continue
    raise TrackerUnavailableError(
        "this OpenCV build has none of the supported CSRT, KCF, or MIL trackers"
    )


@dataclass(frozen=True, slots=True)
class AnchoredTrackerConfig:
    """Safety gates for a continuous, manually anchored member track."""

    minimum_confidence: float = 0.48
    minimum_appearance_similarity: float = 0.12
    appearance_weight: float = 0.55
    maximum_center_jump: float = 0.20
    maximum_scale_change: float = 2.5
    hold_frames: int = 4
    histogram_bins: int = 16

    def __post_init__(self) -> None:
        _probability(self.minimum_confidence, "minimum_confidence")
        _probability(
            self.minimum_appearance_similarity, "minimum_appearance_similarity"
        )
        _probability(self.appearance_weight, "appearance_weight")
        if not isfinite(self.maximum_center_jump) or self.maximum_center_jump <= 0:
            raise ValueError("maximum_center_jump must be positive and finite")
        if not isfinite(self.maximum_scale_change) or self.maximum_scale_change <= 1:
            raise ValueError("maximum_scale_change must be finite and greater than 1")
        if isinstance(self.hold_frames, bool) or not isinstance(self.hold_frames, int):
            raise ValueError("hold_frames must be an integer")
        if self.hold_frames < 0:
            raise ValueError("hold_frames cannot be negative")
        if (
            isinstance(self.histogram_bins, bool)
            or not isinstance(self.histogram_bins, int)
            or self.histogram_bins < 2
        ):
            raise ValueError("histogram_bins must be an integer of at least 2")


@dataclass(frozen=True, slots=True)
class MemberTrackObservation:
    """Safe state for one decoded source frame."""

    frame_index: int
    timestamp_seconds: float
    bbox_xyxy: NormalizedBox | None
    confidence: float
    state: TrackingState | str
    member: IVEMember | str
    source_id: str
    failure: TrackingFailure | str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise ValueError("frame_index must be an integer")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if not isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must be non-negative and finite")
        if self.bbox_xyxy is not None:
            object.__setattr__(self, "bbox_xyxy", _normalized_box(self.bbox_xyxy))
        object.__setattr__(self, "confidence", _probability(self.confidence, "confidence"))
        if not isinstance(self.state, TrackingState):
            object.__setattr__(self, "state", TrackingState(self.state))
        object.__setattr__(self, "member", IVEMember.parse(self.member))
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id cannot be empty")
        if self.failure is not None and not isinstance(self.failure, TrackingFailure):
            object.__setattr__(self, "failure", TrackingFailure(self.failure))
        if self.state is TrackingState.LOST:
            if self.bbox_xyxy is not None or self.confidence != 0.0:
                raise ValueError("lost observations cannot claim a box or confidence")
        elif self.bbox_xyxy is None:
            raise ValueError("tracked and holding observations need a box")


@dataclass(frozen=True, slots=True)
class DynamicCropKeyframe:
    """Subject-box keyframe consumed by a downstream virtual camera planner."""

    frame_index: int
    timestamp_seconds: float
    bbox_xyxy: NormalizedBox
    confidence: float
    state: TrackingState

    @property
    def center(self) -> tuple[float, float]:
        left, top, right, bottom = self.bbox_xyxy
        return ((left + right) / 2.0, (top + bottom) / 2.0)

    @property
    def size(self) -> tuple[float, float]:
        left, top, right, bottom = self.bbox_xyxy
        return (right - left, bottom - top)


@dataclass(frozen=True, slots=True)
class MemberTrackingResult:
    member: IVEMember
    source_id: str
    fps: float
    backend_name: str
    observations: tuple[MemberTrackObservation, ...]
    terminal_failure: TrackingFailure | None

    @property
    def is_terminally_lost(self) -> bool:
        return self.terminal_failure is not None

    @property
    def terminal_failure_time_seconds(self) -> float | None:
        """Return the first source-local time at which cropping became unsafe.

        A terminal failure is intentionally diagnostic only: callers may use it
        to choose a non-cropped rendering fallback, but it must never be treated
        as permission to re-identify a person later in the same source track.
        Backend initialization can fail immediately after the accepted anchor;
        in that case the anchor time is the conservative failure boundary.
        """

        if self.terminal_failure is None:
            return None
        for observation in self.observations:
            if observation.failure is self.terminal_failure:
                return observation.timestamp_seconds
        if self.observations:
            return self.observations[0].timestamp_seconds
        return None

    def crop_failure_for_interval(
        self,
        source_start_seconds: float,
        source_end_seconds: float,
    ) -> TrackingFailure | None:
        """Report a terminal failure affecting a half-open source interval.

        This does not claim that the interval can be cropped, nor does it
        restart the tracker.  It lets the member-render pipeline explain why an
        automatic target-member fancam segment was safely downgraded to its
        uncropped full frame.
        """

        start = float(source_start_seconds)
        end = float(source_end_seconds)
        if not isfinite(start) or not isfinite(end) or start < 0 or end <= start:
            raise ValueError("source interval must be finite, non-negative, and ordered")
        failure_time = self.terminal_failure_time_seconds
        if failure_time is None or failure_time >= end:
            return None
        return self.terminal_failure

    def to_dynamic_crop_keyframes(
        self,
        *,
        include_holding: bool = False,
        minimum_confidence: float = 0.0,
    ) -> tuple[DynamicCropKeyframe, ...]:
        """Return only boxes explicitly safe for a dynamic crop.

        Holding boxes are excluded by default: they are useful for a very brief
        visual hold, but they are not evidence that the target is still inside
        the box.  Lost frames can never become keyframes.
        """

        threshold = _probability(minimum_confidence, "minimum_confidence")
        allowed = {TrackingState.TRACKED}
        if include_holding:
            allowed.add(TrackingState.HOLDING)
        return tuple(
            DynamicCropKeyframe(
                frame_index=item.frame_index,
                timestamp_seconds=item.timestamp_seconds,
                bbox_xyxy=item.bbox_xyxy,
                confidence=item.confidence,
                state=item.state,
            )
            for item in self.observations
            if item.state in allowed
            and item.bbox_xyxy is not None
            and item.confidence >= threshold
        )

    def to_virtual_camera_observations(
        self, *, include_holding: bool = False
    ) -> tuple[object, ...]:
        """Adapt the result to :class:`crosscut.vision.TrackObservation`.

        This import is local to keep the tracker backend independent of the
        camera planner.  A manually established anchor supplies identity; once
        continuity is uncertain, the terminal state prevents re-identification.
        Holding frames, if requested, are marked occluded so they cannot pass a
        target-confidence gate accidentally.
        """

        from .vision import TrackObservation

        converted: list[TrackObservation] = []
        for item in self.observations:
            if item.state is TrackingState.LOST or item.bbox_xyxy is None:
                continue
            if item.state is TrackingState.HOLDING and not include_holding:
                continue
            converted.append(
                TrackObservation(
                    frame_index=item.frame_index,
                    timestamp_seconds=item.timestamp_seconds,
                    bbox_xyxy=item.bbox_xyxy,
                    detection_confidence=item.confidence,
                    member=self.member,
                    member_confidence=item.confidence,
                    source_id=self.source_id,
                    track_id=f"manual-anchor:{self.source_id}",
                    identity_id=f"ive:{self.member.name.casefold()}",
                    occluded=item.state is TrackingState.HOLDING,
                )
            )
        return tuple(converted)


def _validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("each frame must be a non-empty numpy array")
    if frame.ndim not in (2, 3):
        raise ValueError(f"unsupported frame shape: {frame.shape}")
    if frame.ndim == 3 and frame.shape[2] not in (1, 3, 4):
        raise ValueError(f"unsupported frame shape: {frame.shape}")


def _box_to_pixels(box: NormalizedBox, frame: np.ndarray) -> PixelBox:
    height, width = frame.shape[:2]
    left, top, right, bottom = box
    return (
        left * width,
        top * height,
        (right - left) * width,
        (bottom - top) * height,
    )


def _pixels_to_box(box: PixelBox, frame: np.ndarray) -> NormalizedBox | None:
    height, width = frame.shape[:2]
    x, y, box_width, box_height = box
    if not all(isfinite(value) for value in box):
        return None
    if box_width <= 0 or box_height <= 0 or width <= 0 or height <= 0:
        return None
    left = max(0.0, min(1.0, x / width))
    top = max(0.0, min(1.0, y / height))
    right = max(0.0, min(1.0, (x + box_width) / width))
    bottom = max(0.0, min(1.0, (y + box_height) / height))
    if left >= right or top >= bottom:
        return None
    return (left, top, right, bottom)


def _appearance_histogram(
    frame: np.ndarray, box: NormalizedBox, bins: int
) -> np.ndarray | None:
    height, width = frame.shape[:2]
    left, top, right, bottom = box
    x0 = max(0, min(width - 1, int(round(left * width))))
    y0 = max(0, min(height - 1, int(round(top * height))))
    x1 = max(x0 + 1, min(width, int(round(right * width))))
    y1 = max(y0 + 1, min(height, int(round(bottom * height))))
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    if patch.ndim == 2:
        gray = patch
    elif patch.shape[2] == 1:
        gray = patch[:, :, 0]
    else:
        code = cv2.COLOR_BGRA2GRAY if patch.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(patch, code)
    histogram = cv2.calcHist([gray], [0], None, [bins], [0, 256]).astype(np.float32)
    total = float(histogram.sum())
    if total <= 0:
        return None
    histogram /= total
    return histogram


def _appearance_similarity(
    reference: np.ndarray | None, current: np.ndarray | None
) -> float:
    if reference is None or current is None:
        return 0.0
    distance = float(cv2.compareHist(reference, current, cv2.HISTCMP_BHATTACHARYYA))
    return max(0.0, min(1.0, 1.0 - distance))


def _geometry_confidence(
    previous: NormalizedBox,
    current: NormalizedBox,
    settings: AnchoredTrackerConfig,
) -> float | None:
    p_left, p_top, p_right, p_bottom = previous
    c_left, c_top, c_right, c_bottom = current
    previous_center = ((p_left + p_right) / 2, (p_top + p_bottom) / 2)
    current_center = ((c_left + c_right) / 2, (c_top + c_bottom) / 2)
    center_distance = hypot(
        current_center[0] - previous_center[0],
        current_center[1] - previous_center[1],
    )
    if center_distance > settings.maximum_center_jump:
        return None

    previous_area = (p_right - p_left) * (p_bottom - p_top)
    current_area = (c_right - c_left) * (c_bottom - c_top)
    scale_ratio = current_area / previous_area
    if not (
        1.0 / settings.maximum_scale_change
        <= scale_ratio
        <= settings.maximum_scale_change
    ):
        return None

    center_score = exp(-0.5 * (center_distance / settings.maximum_center_jump) ** 2)
    scale_score = exp(
        -abs(log(scale_ratio)) / max(log(settings.maximum_scale_change), 1e-9)
    )
    return float((center_score + scale_score) / 2.0)


def track_anchored_member(
    frames: Iterable[tuple[int, np.ndarray]],
    *,
    anchor: MemberAnchor,
    fps: float,
    backend: TrackerBackend | None = None,
    config: AnchoredTrackerConfig | None = None,
    scene_change_frames: Iterable[int] = (),
    start_time_seconds: float = 0.0,
) -> MemberTrackingResult:
    """Track one manually anchored IVE member without automatic re-entry.

    Frames must be ordered by their source-frame indices and must include the
    anchor frame.  Frames before the anchor are ignored.  The first failed gate
    permanently stops backend updates; subsequent frames are only ``holding``
    or ``lost``.  A new shot therefore requires a new explicit user anchor.
    """

    if not isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if not isfinite(start_time_seconds) or start_time_seconds < 0:
        raise ValueError("start_time_seconds must be non-negative and finite")
    settings = config or AnchoredTrackerConfig()
    tracker = backend or create_opencv_tracker_backend()
    if not isinstance(tracker, TrackerBackend):
        raise TypeError("backend does not implement TrackerBackend")
    scene_changes = {int(frame_index) for frame_index in scene_change_frames}
    if any(frame_index < 0 for frame_index in scene_changes):
        raise ValueError("scene_change_frames cannot contain negative indices")

    observations: list[MemberTrackObservation] = []
    found_anchor = False
    previous_index = -1
    previous_box = anchor.bbox_xyxy
    previous_confidence = 1.0
    anchor_histogram: np.ndarray | None = None
    terminal_failure: TrackingFailure | None = None
    terminal_age = 0

    def timestamp(frame_index: int) -> float:
        return start_time_seconds + frame_index / fps

    def append_terminal(frame_index: int, *, allow_hold: bool) -> None:
        nonlocal terminal_age
        if allow_hold and terminal_age < settings.hold_frames:
            decay = 1.0 - (terminal_age + 1) / (settings.hold_frames + 1)
            observations.append(
                MemberTrackObservation(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp(frame_index),
                    bbox_xyxy=previous_box,
                    confidence=max(0.0, previous_confidence * decay),
                    state=TrackingState.HOLDING,
                    member=anchor.member,
                    source_id=anchor.source_id,
                    failure=terminal_failure,
                )
            )
            terminal_age += 1
            return
        observations.append(
            MemberTrackObservation(
                frame_index=frame_index,
                timestamp_seconds=timestamp(frame_index),
                bbox_xyxy=None,
                confidence=0.0,
                state=TrackingState.LOST,
                member=anchor.member,
                source_id=anchor.source_id,
                failure=terminal_failure,
            )
        )
        terminal_age += 1

    for raw_index, frame in frames:
        if isinstance(raw_index, bool) or not isinstance(raw_index, (int, np.integer)):
            raise ValueError("frame indices must be integers")
        frame_index = int(raw_index)
        if frame_index <= previous_index:
            raise ValueError("frame indices must be strictly increasing")
        previous_index = frame_index
        _validate_frame(frame)
        if frame_index < anchor.frame_index:
            continue
        if not found_anchor:
            if frame_index != anchor.frame_index:
                raise ValueError("frames do not include the requested anchor frame")
            found_anchor = True
            anchor_histogram = _appearance_histogram(
                frame, anchor.bbox_xyxy, settings.histogram_bins
            )
            initialized = tracker.initialize(frame, _box_to_pixels(anchor.bbox_xyxy, frame))
            observations.append(
                MemberTrackObservation(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp(frame_index),
                    bbox_xyxy=anchor.bbox_xyxy,
                    confidence=1.0,
                    state=TrackingState.TRACKED,
                    member=anchor.member,
                    source_id=anchor.source_id,
                )
            )
            if not initialized:
                terminal_failure = TrackingFailure.BACKEND_INITIALIZATION
            continue

        if terminal_failure is not None:
            append_terminal(
                frame_index,
                allow_hold=terminal_failure is not TrackingFailure.SCENE_CHANGE,
            )
            continue

        if frame_index in scene_changes:
            terminal_failure = TrackingFailure.SCENE_CHANGE
            append_terminal(frame_index, allow_hold=False)
            continue

        update = tracker.update(frame)
        if not isinstance(update, TrackerUpdate):
            raise TypeError("backend.update must return TrackerUpdate")
        if not update.success or update.bbox_xywh is None:
            terminal_failure = TrackingFailure.BACKEND_UPDATE
            append_terminal(frame_index, allow_hold=True)
            continue

        candidate = _pixels_to_box(update.bbox_xywh, frame)
        geometry = (
            _geometry_confidence(previous_box, candidate, settings)
            if candidate is not None
            else None
        )
        appearance = (
            _appearance_similarity(
                anchor_histogram,
                _appearance_histogram(frame, candidate, settings.histogram_bins),
            )
            if candidate is not None
            else 0.0
        )
        backend_confidence = 1.0 if update.confidence is None else update.confidence
        if geometry is None:
            confidence = 0.0
        else:
            visual_confidence = (
                settings.appearance_weight * appearance
                + (1.0 - settings.appearance_weight) * geometry
            )
            confidence = backend_confidence * visual_confidence

        if (
            candidate is None
            or geometry is None
            or appearance < settings.minimum_appearance_similarity
            or confidence < settings.minimum_confidence
        ):
            terminal_failure = TrackingFailure.CONFIDENCE_GATE
            append_terminal(frame_index, allow_hold=True)
            continue

        previous_box = candidate
        previous_confidence = confidence
        observations.append(
            MemberTrackObservation(
                frame_index=frame_index,
                timestamp_seconds=timestamp(frame_index),
                bbox_xyxy=candidate,
                confidence=confidence,
                state=TrackingState.TRACKED,
                member=anchor.member,
                source_id=anchor.source_id,
            )
        )

    if not found_anchor:
        raise ValueError("frames do not include the requested anchor frame")
    return MemberTrackingResult(
        member=IVEMember.parse(anchor.member),
        source_id=anchor.source_id,
        fps=float(fps),
        backend_name=tracker.name,
        observations=tuple(observations),
        terminal_failure=terminal_failure,
    )
