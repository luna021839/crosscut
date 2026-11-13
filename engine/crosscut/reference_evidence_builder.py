"""Build resolver-ready visual evidence from a finished stage mix.

The builder is deliberately offline and side-effect free until ``write`` is
called.  It detects reference hard cuts, samples every recovered shot midpoint
and a small frame window around each cut, then reverse-matches those reference
frames against every synchronized original.  Each original is decoded exactly
once in ascending frame order by :func:`match_aligned_target_frames`.

Only compact scalar evidence is serialized.  Reference/source frames, ORB
descriptors, keypoints, and histograms remain transient process memory.
"""

from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from .reference_match import (
    ANALYSIS_BASIS as REFERENCE_MATCH_ANALYSIS_BASIS,
    FrameFeatures,
    OfflineReferenceMatcher,
    ReferenceAcceptanceConfig,
    ReferenceMatchConfig,
    TimedFrameMatch,
    decode_target_frames_sequential,
    decide_reference_match,
    match_aligned_target_frames,
)
from .vision import (
    SceneChange,
    SceneDetectionConfig,
    detect_scene_changes_in_video,
)


SCHEMA_VERSION = 1
ANALYSIS_BASIS = "reference_hard_cut_reverse_match_evidence_v1"


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _json_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    copied = deepcopy(dict(value))
    try:
        json.dumps(copied, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON-serializable") from exc
    return copied


@dataclass(frozen=True, slots=True)
class ClockControlPoint:
    """One checked point on a master-to-source clock."""

    master_seconds: float
    source_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "master_seconds",
            _finite(self.master_seconds, "clock master seconds"),
        )
        object.__setattr__(
            self,
            "source_seconds",
            _finite(self.source_seconds, "clock source seconds"),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "master_seconds": self.master_seconds,
            "source_seconds": self.source_seconds,
            "offset_seconds": self.source_seconds - self.master_seconds,
        }


@dataclass(frozen=True, slots=True)
class ReferenceSourceClock:
    """A scalar offset or monotonic piecewise-linear source clock.

    ``master_to_source(t)`` follows the engine-wide convention: a positive
    offset means the synchronized content occurs later in the source file.
    Piecewise control points contain absolute master/source coordinates, so
    ``global_offset_seconds`` remains serialized as their scalar alignment
    baseline but is not added a second time.
    Endpoint segments are extrapolated so a caller can use the exact same
    clock for midpoint and boundary-window samples.
    """

    basis: str
    global_offset_seconds: float = 0.0
    control_points: tuple[ClockControlPoint, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        basis = str(self.basis).strip()
        if not basis:
            raise ValueError("clock basis cannot be empty")
        offset = _finite(self.global_offset_seconds, "global clock offset")
        points = tuple(self.control_points)
        if len(points) == 1:
            raise ValueError("piecewise source clock needs at least two points")
        if any(not isinstance(point, ClockControlPoint) for point in points):
            raise TypeError("clock control_points must be ClockControlPoint values")
        for left, right in zip(points, points[1:], strict=False):
            if (
                right.master_seconds <= left.master_seconds
                or right.source_seconds <= left.source_seconds
            ):
                raise ValueError("clock control points must be strictly monotonic")
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "global_offset_seconds", offset)
        object.__setattr__(self, "control_points", points)
        object.__setattr__(
            self,
            "provenance",
            _json_mapping(self.provenance, "clock provenance"),
        )

    @classmethod
    def scalar(
        cls,
        offset_seconds: float,
        *,
        basis: str = "scalar_audio_alignment_offset_v1",
        provenance: Mapping[str, Any] | None = None,
    ) -> "ReferenceSourceClock":
        return cls(
            basis=basis,
            global_offset_seconds=offset_seconds,
            provenance=provenance or {},
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "ReferenceSourceClock":
        raw_points = value.get("control_points", ())
        if not isinstance(raw_points, Sequence) or isinstance(
            raw_points,
            (str, bytes),
        ):
            raise ValueError("clock control_points must be an array")
        points: list[ClockControlPoint] = []
        for index, raw_point in enumerate(raw_points):
            if not isinstance(raw_point, Mapping):
                raise ValueError(f"clock control point {index} must be an object")
            points.append(
                ClockControlPoint(
                    master_seconds=raw_point.get("master_seconds"),
                    source_seconds=raw_point.get("source_seconds"),
                )
            )
        basis = str(
            value.get("basis")
            or value.get("analysis_basis")
            or value.get("backend")
            or "caller_supplied_master_to_source_clock_v1"
        )
        provenance = value.get("provenance", {})
        if not isinstance(provenance, Mapping):
            raise ValueError("clock provenance must be an object")
        return cls(
            basis=basis,
            global_offset_seconds=value.get(
                "global_offset_seconds",
                value.get("offset_seconds", 0.0),
            ),
            control_points=tuple(points),
            provenance=provenance,
        )

    @property
    def mapping_mode(self) -> str:
        return "piecewise_linear" if self.control_points else "scalar_offset"

    def master_to_source(self, master_seconds: float) -> float:
        master_time = _finite(master_seconds, "master seconds")
        if not self.control_points:
            return master_time + self.global_offset_seconds
        masters = tuple(point.master_seconds for point in self.control_points)
        position = bisect_right(masters, master_time)
        if position == 0:
            left, right = self.control_points[:2]
        elif position >= len(self.control_points):
            left, right = self.control_points[-2:]
        else:
            left, right = self.control_points[position - 1 : position + 1]
        fraction = (master_time - left.master_seconds) / (
            right.master_seconds - left.master_seconds
        )
        return left.source_seconds + fraction * (
            right.source_seconds - left.source_seconds
        )

    def source_to_master(self, source_seconds: float) -> float:
        """Invert the checked source clock without quantizing either axis.

        Reference stage mixes have their own intro/outro clock.  Reverse
        matching therefore needs the exact inverse mapping from a decoded
        reference-frame timestamp back to the canonical performance clock.
        Strictly monotonic control points make this inverse unambiguous and
        preserve the same endpoint extrapolation policy as
        :meth:`master_to_source`.
        """

        source_time = _finite(source_seconds, "source seconds")
        if not self.control_points:
            return source_time - self.global_offset_seconds
        sources = tuple(point.source_seconds for point in self.control_points)
        position = bisect_right(sources, source_time)
        if position == 0:
            left, right = self.control_points[:2]
        elif position >= len(self.control_points):
            left, right = self.control_points[-2:]
        else:
            left, right = self.control_points[position - 1 : position + 1]
        fraction = (source_time - left.source_seconds) / (
            right.source_seconds - left.source_seconds
        )
        return left.master_seconds + fraction * (
            right.master_seconds - left.master_seconds
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis": self.basis,
            "mapping_mode": self.mapping_mode,
            "global_offset_seconds": self.global_offset_seconds,
            "global_offset_application": (
                "diagnostic_scalar_baseline_control_points_are_absolute"
                if self.control_points
                else "added_to_master_seconds"
            ),
            "control_points": [point.to_dict() for point in self.control_points],
            "provenance": deepcopy(dict(self.provenance)),
        }


@dataclass(frozen=True, slots=True)
class ReferenceSourceCatalogEntry:
    """One synchronized original eligible for reverse matching."""

    video_id: str
    path: Path
    clock: ReferenceSourceClock
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        video_id = str(self.video_id).strip()
        if not video_id:
            raise ValueError("source video_id cannot be empty")
        if video_id == "__UNKNOWN__":
            raise ValueError("source video_id collides with the reserved UNKNOWN state")
        if not isinstance(self.clock, ReferenceSourceClock):
            raise TypeError("source clock must be a ReferenceSourceClock")
        object.__setattr__(self, "video_id", video_id)
        object.__setattr__(self, "path", Path(self.path).expanduser())
        object.__setattr__(
            self,
            "provenance",
            _json_mapping(self.provenance, "source provenance"),
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "ReferenceSourceCatalogEntry":
        raw_clock = value.get("clock")
        if isinstance(raw_clock, ReferenceSourceClock):
            clock = raw_clock
        elif isinstance(raw_clock, Mapping):
            clock = ReferenceSourceClock.from_mapping(raw_clock)
        elif raw_clock is None:
            clock = ReferenceSourceClock.scalar(
                value.get("offset_seconds", 0.0),
                basis=str(
                    value.get("clock_basis")
                    or "scalar_audio_alignment_offset_v1"
                ),
            )
        else:
            raise ValueError("source clock must be an object")
        provenance = value.get("provenance", {})
        if not isinstance(provenance, Mapping):
            raise ValueError("source provenance must be an object")
        return cls(
            video_id=str(value.get("video_id") or value.get("source_id") or ""),
            path=Path(str(value.get("path") or value.get("local_path") or "")),
            clock=clock,
            provenance=provenance,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "path": str(self.path.resolve()),
            "clock": self.clock.to_dict(),
            "provenance": deepcopy(dict(self.provenance)),
        }


NormalizedRectangle = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class WatermarkMaskPolicy:
    """Explicit normalized overlay regions removed before feature extraction."""

    policy_id: str
    rectangles: tuple[NormalizedRectangle, ...] = ()
    fill_bgr: tuple[int, int, int] = (0, 0, 0)
    rationale: str = "caller_selected"

    def __post_init__(self) -> None:
        policy_id = str(self.policy_id).strip()
        if not policy_id:
            raise ValueError("watermark mask policy_id cannot be empty")
        rectangles: list[NormalizedRectangle] = []
        for index, rectangle in enumerate(self.rectangles):
            if len(rectangle) != 4:
                raise ValueError(f"watermark rectangle {index} needs four values")
            left, top, right, bottom = (
                _finite(value, f"watermark rectangle {index}")
                for value in rectangle
            )
            if not (
                0.0 <= left < right <= 1.0
                and 0.0 <= top < bottom <= 1.0
            ):
                raise ValueError("watermark rectangles must be normalized xyxy")
            rectangles.append((left, top, right, bottom))
        if len(self.fill_bgr) != 3 or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 255
            for value in self.fill_bgr
        ):
            raise ValueError("watermark fill_bgr must contain three byte values")
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "rectangles", tuple(rectangles))
        object.__setattr__(self, "fill_bgr", tuple(self.fill_bgr))
        object.__setattr__(self, "rationale", str(self.rationale))

    @classmethod
    def explicit_none(cls) -> "WatermarkMaskPolicy":
        return cls(
            policy_id="explicit_no_watermark_mask_v1",
            rationale="caller explicitly preserves the full frame",
        )

    @classmethod
    def broadcast_safe_default(cls) -> "WatermarkMaskPolicy":
        return cls(
            policy_id="broadcast_corner_and_lower_third_mask_v1",
            rectangles=(
                (0.00, 0.00, 0.24, 0.14),
                (0.76, 0.00, 1.00, 0.18),
                (0.00, 0.78, 0.55, 1.00),
            ),
            rationale=(
                "prevent broadcaster logos and lower-thirds from becoming "
                "source-identity shortcuts"
            ),
        )

    def apply(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("watermark masking requires a BGR frame")
        if not self.rectangles:
            return frame
        masked = frame.copy()
        height, width = masked.shape[:2]
        for left, top, right, bottom in self.rectangles:
            x0 = max(0, min(width, int(math.floor(left * width))))
            y0 = max(0, min(height, int(math.floor(top * height))))
            x1 = max(0, min(width, int(math.ceil(right * width))))
            y1 = max(0, min(height, int(math.ceil(bottom * height))))
            masked[y0:y1, x0:x1] = self.fill_bgr
        return masked

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "application": "same_normalized_regions_on_reference_and_sources",
            "application_stages": [
                "reference_match_feature_extraction",
                "source_match_feature_extraction",
            ],
            "hard_cut_detection": "unmasked_full_frame_component_gated_detector",
            "mode": "zero_fill" if self.rectangles else "none",
            "normalized_xyxy_rectangles": [
                list(rectangle) for rectangle in self.rectangles
            ],
            "fill_bgr": list(self.fill_bgr),
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class ReferenceEvidenceConfig:
    """Source-observation grid, hard-cut windows and reverse-match policy."""

    scene_detection: SceneDetectionConfig = field(
        default_factory=SceneDetectionConfig
    )
    reference_match: ReferenceMatchConfig = field(
        default_factory=ReferenceMatchConfig
    )
    acceptance: ReferenceAcceptanceConfig = field(
        default_factory=ReferenceAcceptanceConfig
    )
    boundary_offsets_frames: tuple[int, ...] = (-2, -1, 0, 1, 2)
    temporal_sample_rate_hz: float = 4.0
    watermark_mask: WatermarkMaskPolicy = field(
        default_factory=WatermarkMaskPolicy.broadcast_safe_default
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.temporal_sample_rate_hz, bool)
            or not isinstance(self.temporal_sample_rate_hz, (int, float))
            or not math.isfinite(self.temporal_sample_rate_hz)
            or not 0.0 < self.temporal_sample_rate_hz <= 60.0
        ):
            raise ValueError("temporal source sampling rate must be in (0, 60]")
        if not isinstance(self.scene_detection, SceneDetectionConfig):
            raise TypeError("scene_detection must be a SceneDetectionConfig")
        if not isinstance(self.reference_match, ReferenceMatchConfig):
            raise TypeError("reference_match must be a ReferenceMatchConfig")
        if not isinstance(self.acceptance, ReferenceAcceptanceConfig):
            raise TypeError("acceptance must be a ReferenceAcceptanceConfig")
        offsets = tuple(self.boundary_offsets_frames)
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in offsets
        ):
            raise ValueError("boundary offsets must contain integers")
        if len(set(offsets)) != len(offsets):
            raise ValueError("boundary offsets cannot contain duplicates")
        if 0 not in offsets or not any(value < 0 for value in offsets) or not any(
            value > 0 for value in offsets
        ):
            raise ValueError(
                "boundary offsets must cover frames before, at, and after a cut"
            )
        if not isinstance(self.watermark_mask, WatermarkMaskPolicy):
            raise TypeError("watermark_mask must be a WatermarkMaskPolicy")
        object.__setattr__(self, "boundary_offsets_frames", tuple(sorted(offsets)))

    def to_dict(self) -> dict[str, Any]:
        scene = self.scene_detection
        return {
            "analysis_basis": ANALYSIS_BASIS,
            "hard_cut_detection": {
                "analysis_basis": "ive_component_gated_hard_cut_detector_v2",
                "combined_threshold": scene.threshold,
                "minimum_frame_difference": scene.minimum_frame_difference,
                "minimum_histogram_distance": scene.minimum_histogram_distance,
                "minimum_gap_frames": scene.min_gap_frames,
                "frame_difference_weight": scene.frame_difference_weight,
                "histogram_weight": scene.histogram_weight,
                "difference_gain": scene.difference_gain,
                "analysis_size": list(scene.analysis_size),
                "histogram_bins": scene.histogram_bins,
            },
            "reference_match": self.reference_match.to_dict(),
            "acceptance": self.acceptance.to_dict(),
            "boundary_offsets_frames": list(self.boundary_offsets_frames),
            "temporal_source_sampling": {
                "contract": "reference_frame_clock_uniform_source_observation_grid_v1",
                "sample_rate_hz": self.temporal_sample_rate_hz,
                "first_and_last_reference_frames_requested": True,
                "grid_samples_are_not_cut_targets": True,
            },
            "watermark_mask": self.watermark_mask.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class _VideoProbe:
    frame_rate: float
    frame_count: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class _SamplePoint:
    frame_index: int
    master_seconds: float
    reasons: tuple[str, ...]


class _WatermarkMaskedMatcher(OfflineReferenceMatcher):
    def __init__(
        self,
        config: ReferenceMatchConfig,
        policy: WatermarkMaskPolicy,
    ) -> None:
        super().__init__(config)
        self._watermark_policy = policy

    def extract(self, frame: NDArray[np.uint8]) -> FrameFeatures:
        return super().extract(self._watermark_policy.apply(frame))


def _probe_video(path: Path, *, role: str) -> _VideoProbe:
    source = Path(path).expanduser()
    if not source.is_file():
        raise ValueError(f"{role} video does not exist: {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"cannot decode {role} video: {source}")
    try:
        frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
        raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if not math.isfinite(frame_rate) or frame_rate <= 0.0:
        raise ValueError(f"invalid {role} frame rate: {source}")
    if not math.isfinite(raw_count) or raw_count < 1.0:
        raise ValueError(f"invalid {role} frame count: {source}")
    frame_count = int(round(raw_count))
    width = int(round(raw_width)) if math.isfinite(raw_width) else 0
    height = int(round(raw_height)) if math.isfinite(raw_height) else 0
    if width < 1 or height < 1:
        raise ValueError(f"invalid {role} dimensions: {source}")
    return _VideoProbe(frame_rate, frame_count, width, height)


def _coerce_catalog(
    values: Sequence[ReferenceSourceCatalogEntry | Mapping[str, Any]],
) -> tuple[ReferenceSourceCatalogEntry, ...]:
    output: list[ReferenceSourceCatalogEntry] = []
    for index, value in enumerate(values):
        if isinstance(value, ReferenceSourceCatalogEntry):
            entry = value
        elif isinstance(value, Mapping):
            entry = ReferenceSourceCatalogEntry.from_mapping(value)
        else:
            raise TypeError(f"source catalog item {index} must be an entry or object")
        output.append(entry)
    if not output:
        raise ValueError("source catalog cannot be empty")
    video_ids = tuple(entry.video_id for entry in output)
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("source catalog video_ids must be unique")
    resolved_paths = tuple(entry.path.resolve() for entry in output)
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("source catalog paths must be unique")
    return tuple(output)


def _plan_samples(
    changes: Sequence[SceneChange],
    *,
    frame_count: int,
    frame_rate: float,
    master_start_seconds: float,
    master_end_seconds: float | None = None,
    reference_clock: ReferenceSourceClock,
    boundary_offsets_frames: Sequence[int],
    temporal_sample_rate_hz: float = 4.0,
) -> tuple[_SamplePoint, ...]:
    if (
        isinstance(temporal_sample_rate_hz, bool)
        or not isinstance(temporal_sample_rate_hz, (int, float))
        or not math.isfinite(temporal_sample_rate_hz)
        or not 0.0 < temporal_sample_rate_hz <= 60.0
    ):
        raise ValueError("temporal source sampling rate must be in (0, 60]")
    if frame_count < 1 or not math.isfinite(frame_rate) or frame_rate <= 0.0:
        raise ValueError("source sampling requires positive frame count and fps")
    cut_indices = tuple(
        sorted(
            {
                change.frame_index
                for change in changes
                if 0 < change.frame_index < frame_count
            }
        )
    )
    reasons: dict[int, set[str]] = {}

    def add(frame_index: int, reason: str) -> None:
        if 0 <= frame_index < frame_count:
            reasons.setdefault(frame_index, set()).add(reason)

    # Detector shots can contain matched edits/dissolves which are not hard
    # scene changes. Observe the source throughout them, rather than treating
    # a single midpoint as the only identity evidence for a long interval.
    # This grid supplies observations only; it never creates a cut or label.
    effective_rate = min(float(temporal_sample_rate_hz), frame_rate)
    grid_count = int(math.floor((frame_count - 1) * effective_rate / frame_rate)) + 1
    for grid_index in range(grid_count):
        add(int(round(grid_index * frame_rate / effective_rate)), "temporal_source_grid")
    add(frame_count - 1, "temporal_source_grid")

    boundaries = (0, *cut_indices, frame_count)
    for shot_index, (start, end) in enumerate(
        zip(boundaries, boundaries[1:], strict=False)
    ):
        if end <= start:
            continue
        add((start + end - 1) // 2, f"shot_midpoint:{shot_index}")
    for cut_index in cut_indices:
        for offset in boundary_offsets_frames:
            add(
                cut_index + offset,
                f"hard_cut_boundary:{cut_index}:{offset:+d}",
            )
    planned: list[_SamplePoint] = []
    for index in sorted(reasons):
        master_seconds = (
            master_start_seconds
            + reference_clock.source_to_master(index / frame_rate)
        )
        # Reference edits can contain a short pre-roll before the canonical
        # performance clock.  Those frames have no legal synchronized source
        # target and cannot enter the attribution DP, whose master clock is
        # explicitly non-negative.
        if master_seconds < -1e-9:
            continue
        if (
            master_end_seconds is not None
            and master_seconds > master_end_seconds + 1e-9
        ):
            continue
        planned.append(
            _SamplePoint(
                frame_index=index,
                master_seconds=max(0.0, master_seconds),
                reasons=tuple(sorted(reasons[index])),
            )
        )
    return tuple(planned)


def _project_detected_reference_cuts(
    changes: Sequence[SceneChange],
    *,
    frame_rate: float,
    master_start_seconds: float,
    master_end_seconds: float,
    reference_clock: ReferenceSourceClock,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Project detector cuts onto the legal non-negative performance clock.

    A finished reference edit can contain a broadcaster intro before master
    zero.  The detector observation remains auditable, but it cannot define a
    trainable source/crop boundary because no synchronized performance frame
    exists on its old side.  Such cuts are therefore retained separately
    instead of being clamped to zero or silently discarded.
    """

    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    previous_frame = -1
    previous_seconds = -math.inf
    for change in changes:
        master_seconds = master_start_seconds + reference_clock.source_to_master(
            change.frame_index / frame_rate
        )
        if change.frame_index <= previous_frame or master_seconds <= previous_seconds:
            raise ValueError("detected reference cuts are not strictly chronological")
        serialized = {
            "frame_index": change.frame_index,
            "master_seconds": master_seconds,
            "score": change.score,
            "frame_difference": change.frame_difference,
            "histogram_distance": change.histogram_distance,
        }
        if master_seconds <= master_start_seconds:
            excluded.append(
                {
                    **serialized,
                    "exclusion_reason": "at_or_before_performance_master_start",
                }
            )
        elif master_seconds >= master_end_seconds:
            excluded.append(
                {
                    **serialized,
                    "exclusion_reason": "at_or_after_reference_master_end",
                }
            )
        else:
            retained.append(serialized)
        previous_frame = change.frame_index
        previous_seconds = master_seconds
    return (tuple(retained), tuple(excluded))


def _candidate_payload(
    entry: ReferenceSourceCatalogEntry,
    match: TimedFrameMatch | None,
    *,
    expected_source_time: float,
    requested_indices: Sequence[int],
    source_probe: _VideoProbe,
) -> dict[str, Any]:
    if match is not None:
        payload = match.to_dict()
        status = "matched"
    else:
        payload = {
            "score": 0.0,
            "orb_good_matches": 0,
            "orb_homography_inliers": 0,
            "orb_inlier_ratio": 0.0,
            "structure_ncc": None,
            "color_similarity": None,
            "homography_reference_to_source": None,
            "source_frame_index": None,
            "source_time_seconds": None,
            "time_adjustment_seconds": None,
        }
        if expected_source_time < 0.0 or expected_source_time >= (
            source_probe.frame_count / source_probe.frame_rate
        ):
            status = "outside_source_clock_range"
        else:
            status = "requested_frames_not_decoded"
    return {
        "video_id": entry.video_id,
        **payload,
        "expected_source_time_seconds": expected_source_time,
        "requested_source_frame_indices": list(requested_indices),
        "match_status": status,
    }


@dataclass(frozen=True, slots=True)
class ReferenceEvidence:
    """Serializable evidence payload with an explicit caller-owned writer."""

    payload: dict[str, Any]

    @property
    def sample_count(self) -> int:
        return len(self.payload["samples"])

    def write(self, output_path: Path) -> Path:
        """Write only to the exact destination supplied by the caller."""

        destination = Path(output_path).expanduser()
        resolved_destination = destination.resolve()
        protected_media = {
            Path(str(self.payload["reference"])).resolve(),
            *(
                Path(str(source["path"])).resolve()
                for source in self.payload.get("sources", [])
                if isinstance(source, Mapping) and source.get("path")
            ),
        }
        if resolved_destination in protected_media:
            raise ValueError("evidence output cannot overwrite input media")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                self.payload,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        return resolved_destination


def build_reference_evidence(
    reference_path: Path,
    source_catalog: Sequence[
        ReferenceSourceCatalogEntry | Mapping[str, Any]
    ],
    *,
    master_start_seconds: float = 0.0,
    master_end_seconds: float | None = None,
    reference_clock: ReferenceSourceClock | Mapping[str, Any] | None = None,
    config: ReferenceEvidenceConfig | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> ReferenceEvidence:
    """Build resolver-ready evidence without writing media or JSON files."""

    settings = config or ReferenceEvidenceConfig()
    master_start = _finite(master_start_seconds, "master start seconds")
    if master_start < 0.0:
        raise ValueError("master_start_seconds cannot be negative")
    checked_master_end: float | None = None
    if master_end_seconds is not None:
        checked_master_end = _finite(master_end_seconds, "master end seconds")
        if checked_master_end <= master_start:
            raise ValueError("master_end_seconds must exceed master_start_seconds")
    if reference_clock is None:
        checked_reference_clock = ReferenceSourceClock.scalar(
            0.0,
            basis="reference_frame_clock_identity_v1",
        )
    elif isinstance(reference_clock, ReferenceSourceClock):
        checked_reference_clock = reference_clock
    elif isinstance(reference_clock, Mapping):
        checked_reference_clock = ReferenceSourceClock.from_mapping(
            reference_clock
        )
    else:
        raise TypeError("reference_clock must be a ReferenceSourceClock or object")
    reference = Path(reference_path).expanduser()
    reference_probe = _probe_video(reference, role="reference")
    catalog = _coerce_catalog(source_catalog)
    source_probes = {
        entry.video_id: _probe_video(entry.path, role=f"source {entry.video_id}")
        for entry in catalog
    }

    changes = detect_scene_changes_in_video(
        reference,
        config=settings.scene_detection,
        sample_every=1,
    )
    samples = _plan_samples(
        changes,
        frame_count=reference_probe.frame_count,
        frame_rate=reference_probe.frame_rate,
        master_start_seconds=master_start,
        master_end_seconds=checked_master_end,
        reference_clock=checked_reference_clock,
        boundary_offsets_frames=settings.boundary_offsets_frames,
        temporal_sample_rate_hz=settings.temporal_sample_rate_hz,
    )
    if not samples:
        raise ValueError("reference sampling produced no frame targets")

    reference_batch = decode_target_frames_sequential(
        reference,
        (sample.frame_index for sample in samples),
        config=settings.reference_match,
    )
    missing_reference = tuple(
        sample.frame_index
        for sample in samples
        if sample.frame_index not in reference_batch.frames
    )
    if missing_reference:
        raise ValueError(
            "reference sample frames could not be decoded: "
            + ", ".join(str(index) for index in missing_reference)
        )

    matcher = _WatermarkMaskedMatcher(
        settings.reference_match,
        settings.watermark_mask,
    )
    reference_features = tuple(
        matcher.extract(reference_batch.frames[sample.frame_index])
        for sample in samples
    )

    candidates_by_sample: list[list[tuple[dict[str, Any], TimedFrameMatch | None]]] = [
        [] for _ in samples
    ]
    for entry in catalog:
        source_probe = source_probes[entry.video_id]
        expected_times = tuple(
            entry.clock.master_to_source(sample.master_seconds)
            for sample in samples
        )
        target_groups: list[tuple[int, ...]] = []
        for expected_time in expected_times:
            center = int(round(expected_time * source_probe.frame_rate))
            target_groups.append(
                tuple(
                    index
                    for index in (
                        center + jitter
                        for jitter in settings.reference_match.temporal_jitter_frames
                    )
                    if 0 <= index < source_probe.frame_count
                )
            )
        matches = match_aligned_target_frames(
            entry.path,
            tuple(target_groups),
            reference_features,
            expected_source_times=tuple(max(0.0, value) for value in expected_times),
            matcher=matcher,
        )
        for index, (match, expected_time, group) in enumerate(
            zip(matches, expected_times, target_groups, strict=True)
        ):
            candidates_by_sample[index].append(
                (
                    _candidate_payload(
                        entry,
                        match,
                        expected_source_time=expected_time,
                        requested_indices=group,
                        source_probe=source_probe,
                    ),
                    match,
                )
            )

    serialized_samples: list[dict[str, Any]] = []
    accepted_diagnostics = 0
    for sample, candidates in zip(samples, candidates_by_sample, strict=True):
        ranked_pairs = sorted(
            candidates,
            key=lambda item: (-float(item[0]["score"]), item[0]["video_id"]),
        )
        ranked = [pair[0] for pair in ranked_pairs]
        best_payload = ranked[0] if ranked else None
        runner_up = ranked[1] if len(ranked) > 1 else None
        best_match = ranked_pairs[0][1] if ranked_pairs else None
        runner_score = float(runner_up["score"]) if runner_up is not None else 0.0
        decision = decide_reference_match(
            best_match,
            runner_up_score=runner_score,
            config=settings.acceptance,
        )
        accepted_video_id = (
            best_payload["video_id"]
            if decision.accepted and best_payload is not None
            else None
        )
        accepted_diagnostics += int(decision.accepted)
        serialized_samples.append(
            {
                "master_seconds": sample.master_seconds,
                "reference_frame_index": sample.frame_index,
                "sample_reasons": list(sample.reasons),
                "best": deepcopy(best_payload),
                "runner_up": deepcopy(runner_up),
                "top_two": deepcopy(ranked[:2]),
                "conservative_decision": {
                    **decision.to_dict(),
                    "accepted_video_id": accepted_video_id,
                    "resolver_remains_authoritative": True,
                },
                "ranked_candidates": ranked,
            }
        )

    mapped_reference_end = master_start + checked_reference_clock.source_to_master(
        reference_probe.frame_count / reference_probe.frame_rate
    )
    # Canonical detector ownership spans the complete reference media,
    # including a human editor's outro.  ``checked_master_end`` bounds only
    # source-attribution samples; shrinking this canonical endpoint would put
    # retained detector cuts outside their own declared partition.
    master_end = mapped_reference_end
    detected_reference_cuts, excluded_detected_reference_cuts = (
        _project_detected_reference_cuts(
            changes,
            frame_rate=reference_probe.frame_rate,
            master_start_seconds=master_start,
            master_end_seconds=master_end,
            reference_clock=checked_reference_clock,
        )
    )
    caller_provenance = provenance or {}
    if not isinstance(caller_provenance, Mapping):
        raise ValueError("builder provenance must be an object")
    reference_stat = reference.stat()
    serialized_sources: list[dict[str, Any]] = []
    for entry in catalog:
        source_probe = source_probes[entry.video_id]
        source_stat = entry.path.stat()
        serialized_sources.append(
            {
                **entry.to_dict(),
                "media_probe": {
                    "frame_rate": source_probe.frame_rate,
                    "declared_frame_count": source_probe.frame_count,
                    "width": source_probe.width,
                    "height": source_probe.height,
                    "size_bytes": source_stat.st_size,
                    "mtime_ns": source_stat.st_mtime_ns,
                },
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_BASIS,
        "reference": str(reference.resolve()),
        "master_start_seconds": master_start,
        "master_end_seconds": master_end,
        "sources": serialized_sources,
        "clock_policy": {
            "master_clock": (
                "reference_clock.source_to_master("
                "reference_frame_index_divided_by_reference_fps)"
            ),
            "master_start_seconds": master_start,
            "master_end_seconds": master_end,
            "attribution_sample_master_end_seconds": checked_master_end,
            "reference_mapping": checked_reference_clock.to_dict(),
            "source_mapping": "catalog_clock.master_to_source(master_seconds)",
            "source_frame_quantization": "round(source_seconds * source_fps)",
            "temporal_jitter_frames": list(
                settings.reference_match.temporal_jitter_frames
            ),
            "decode_policy": "one_forward_only_union_decode_per_source",
        },
        "watermark_mask_policy": settings.watermark_mask.to_dict(),
        "feature_serialization_policy": {
            "raw_frames_serialized": False,
            "orb_descriptors_serialized": False,
            "keypoints_serialized": False,
            "structure_or_color_embeddings_serialized": False,
            "compact_scores_and_homography_only": True,
        },
        "builder_policy": settings.to_dict(),
        "reference_video": {
            "path": str(reference.resolve()),
            "frame_rate": reference_probe.frame_rate,
            "declared_frame_count": reference_probe.frame_count,
            "width": reference_probe.width,
            "height": reference_probe.height,
            "size_bytes": reference_stat.st_size,
            "mtime_ns": reference_stat.st_mtime_ns,
        },
        "detected_reference_cuts": [
            deepcopy(cut) for cut in detected_reference_cuts
        ],
        "excluded_detected_reference_cuts": [
            deepcopy(cut) for cut in excluded_detected_reference_cuts
        ],
        "sampling_policy": {
            "shot_midpoint_sampled": True,
            "temporal_source_sampling": settings.to_dict()["temporal_source_sampling"],
            "temporal_grid_sample_count": sum(
                "temporal_source_grid" in sample.reasons for sample in samples
            ),
            "boundary_offsets_frames": list(settings.boundary_offsets_frames),
            "duplicate_frame_targets_merged": True,
            "attribution_sample_master_end_seconds": checked_master_end,
            "canonical_detector_partition_remains_full_reference": True,
            "pre_master_zero_detector_cuts_are_audited_but_not_trainable": True,
            "sample_count": len(samples),
        },
        "samples": serialized_samples,
        # A boundary request needs an old/new pair from the selected temporal
        # path.  The pre-resolution builder does not invent that pair.  Dense
        # boundary samples remain available for a later evidence-backed pass.
        "boundary_requests": [],
        "statistics": {
            "source_count": len(catalog),
            "hard_cut_count": len(detected_reference_cuts),
            "detector_hard_cut_count": len(changes),
            "excluded_detected_hard_cut_count": len(
                excluded_detected_reference_cuts
            ),
            "sample_count": len(samples),
            "shot_midpoint_count": sum(
                any(reason.startswith("shot_midpoint:") for reason in sample.reasons)
                for sample in samples
            ),
            "boundary_sample_count": sum(
                any(
                    reason.startswith("hard_cut_boundary:")
                    for reason in sample.reasons
                )
                for sample in samples
            ),
            "diagnostic_conservative_accept_count": accepted_diagnostics,
            "resolver_selected_source_count": None,
        },
        "provenance": {
            "created_at": datetime.now(UTC).isoformat(),
            "builder_module": "crosscut.reference_evidence_builder",
            "builder_analysis_basis": ANALYSIS_BASIS,
            "reference_match_analysis_basis": REFERENCE_MATCH_ANALYSIS_BASIS,
            "caller": _json_mapping(caller_provenance, "builder provenance"),
        },
    }
    # Assert the public artifact stays JSON-only before handing it to callers.
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return ReferenceEvidence(payload)


def build_reference_evidence_file(
    reference_path: Path,
    source_catalog: Sequence[
        ReferenceSourceCatalogEntry | Mapping[str, Any]
    ],
    output_path: Path,
    *,
    master_start_seconds: float = 0.0,
    master_end_seconds: float | None = None,
    reference_clock: ReferenceSourceClock | Mapping[str, Any] | None = None,
    config: ReferenceEvidenceConfig | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Build evidence and write it only to ``output_path``."""

    return build_reference_evidence(
        reference_path,
        source_catalog,
        master_start_seconds=master_start_seconds,
        master_end_seconds=master_end_seconds,
        reference_clock=reference_clock,
        config=config,
        provenance=provenance,
    ).write(output_path)


__all__ = [
    "ANALYSIS_BASIS",
    "ClockControlPoint",
    "ReferenceEvidence",
    "ReferenceEvidenceConfig",
    "ReferenceSourceCatalogEntry",
    "ReferenceSourceClock",
    "SCHEMA_VERSION",
    "WatermarkMaskPolicy",
    "build_reference_evidence",
    "build_reference_evidence_file",
]
