"""Fail-closed cut-local synchronization proof for production StageEdit.

The song-wide audio time-warp proves that a source clock is broadly coherent;
it does not prove that the *retained* camera switches land on the same movement
phase.  This module measures only those final RenderPlan boundaries.  It binds
the result to the exact plan, used-source time-warp payloads, and source-local
boundary clocks, then compares short pose/body-motion series decoded at actual
video PTS.

Raw frames, pose vectors, and body descriptors never leave this module.  A
boundary with missing, static, or ambiguous motion remains explicitly UNKNOWN
and therefore cannot satisfy the registered automatic StageEdit release gate.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .reference_match import (
    DecodedTargetFrames,
    ReferenceMatchConfig,
    decode_target_frames_sequential,
)
from .render import RenderPlan
from .timewarp import TimeWarpMap
from .visual_score import FrameVisualDescriptor, describe_frame


SCHEMA_VERSION = 2
ANALYSIS_BASIS = "actual_pts_renderer_bound_retained_cut_pose_body_motion_phase_v2"
ACTIVATION = "automatic_registered_stage_edit_group"
PTS_BASIS = "ffprobe_best_effort_timestamp_time"
RENDERER_SELECTION_BASIS = (
    "trim_lower_bound_setpts_first_pts_fps_round_near_tpad_clone_v1"
)
_SERIALIZATION_POLICY = {
    "raw_frames_persisted": False,
    "pose_embeddings_persisted": False,
    "formation_descriptors_persisted": False,
    "body_motion_series_persisted": False,
    "sample_pixels_persisted": False,
    "motion_timing_provenance_persisted": True,
    "scalar_aggregate_and_hash_provenance_only": True,
}


@dataclass(frozen=True, slots=True)
class CutLocalSyncConfig:
    sample_radius_frames: int = 7
    maximum_lag_frames: int = 2
    minimum_motion_pair_count: int = 7
    minimum_non_silhouette_pair_count: int = 3
    minimum_motion_standard_deviation: float = 1e-5
    minimum_correlation: float = 0.45
    minimum_correlation_margin: float = 0.05
    # A renderer-owned lower-bound trim can legitimately select just under one
    # frame after an arbitrary context start.  This guard detects larger VFR
    # holes; the release target itself remains the independently measured
    # cross-source phase p95 below.
    maximum_pts_lookup_residual_frames: float = 1.10
    maximum_residual_p95_frames: float = 0.50
    working_width: int = 480
    ffprobe_timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        for name in (
            "sample_radius_frames",
            "maximum_lag_frames",
            "minimum_motion_pair_count",
            "minimum_non_silhouette_pair_count",
            "working_width",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.sample_radius_frames <= self.maximum_lag_frames + 1:
            raise ValueError("cut-local sample radius is too small for lag search")
        for name in (
            "minimum_motion_standard_deviation",
            "minimum_correlation_margin",
            "maximum_pts_lookup_residual_frames",
            "maximum_residual_p95_frames",
            "ffprobe_timeout_seconds",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not -1.0 <= self.minimum_correlation <= 1.0:
            raise ValueError("minimum_correlation must be between -1 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class CutLocalMotionSample:
    """One scalar body-motion observation kept only long enough to correlate."""

    master_seconds: float
    value: float
    basis: str
    grid_index: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.master_seconds):
            raise ValueError("motion sample master time must be finite")
        if not math.isfinite(self.value) or self.value < 0.0:
            raise ValueError("motion sample value must be finite and non-negative")
        if self.basis not in {
            "actual_pose_motion",
            "formation_upper_body_motion",
            "person_layout_motion",
            "silhouette_body_motion",
        }:
            raise ValueError("motion sample basis is unsupported")
        if self.grid_index is not None and (
            isinstance(self.grid_index, bool)
            or not isinstance(self.grid_index, int)
        ):
            raise ValueError("motion sample grid index must be an integer")


@dataclass(frozen=True, slots=True)
class _FrameObservation:
    frame_index: int
    requested_master_seconds: float
    requested_source_seconds: float
    actual_pts_seconds: float
    normalized_pts_seconds: float
    actual_master_seconds: float
    pts_lookup_residual_frames: float
    descriptor: FrameVisualDescriptor


class _SourceEvidenceError(RuntimeError):
    """Path-safe source proof failure represented by one stable code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class _BoundaryContract:
    boundary_index: int
    outgoing_segment_index: int
    incoming_segment_index: int
    outgoing_source_index: int
    incoming_source_index: int
    output_seconds: float
    output_frame_index: int
    canonical_master_seconds: float
    outgoing_source_local_seconds: float
    incoming_source_local_seconds: float
    transition: Mapping[str, Any]
    sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "boundary_index": self.boundary_index,
            "outgoing_segment_index": self.outgoing_segment_index,
            "incoming_segment_index": self.incoming_segment_index,
            "outgoing_source_index": self.outgoing_source_index,
            "incoming_source_index": self.incoming_source_index,
            "output_seconds": self.output_seconds,
            "output_frame_index": self.output_frame_index,
            "canonical_master_seconds": self.canonical_master_seconds,
            "outgoing_source_local_seconds": self.outgoing_source_local_seconds,
            "incoming_source_local_seconds": self.incoming_source_local_seconds,
            "transition": dict(self.transition),
        }


@dataclass(frozen=True, slots=True)
class _RendererWindow:
    boundary_index: int
    role: str
    source_index: int
    grid_indices: tuple[int, ...]
    requested_master_seconds: tuple[float, ...]
    source_start_seconds: float
    source_duration_seconds: float
    timeline_duration_seconds: float
    frame_count: int


@dataclass(frozen=True, slots=True)
class _SerializedTimeWarp:
    source_index: int
    minimum_slope: float
    maximum_slope: float
    control_points: tuple[tuple[float, float], ...]

    def master_to_source(self, master_seconds: float) -> float:
        master = _finite(master_seconds, "serialized master_seconds")
        points = self.control_points
        masters = tuple(item[0] for item in points)
        position = bisect_left(masters, master)
        if position == 0:
            left, right = points[0:2]
        elif position >= len(points):
            left, right = points[-2:]
        elif math.isclose(masters[position], master, rel_tol=0.0, abs_tol=1e-15):
            return points[position][1]
        else:
            left, right = points[position - 1 : position + 1]
        fraction = (master - left[0]) / (right[0] - left[0])
        return left[1] + fraction * (right[1] - left[1])

    def source_to_master(self, source_seconds: float) -> float:
        source = _finite(source_seconds, "serialized source_seconds")
        points = self.control_points
        sources = tuple(item[1] for item in points)
        position = bisect_left(sources, source)
        if position == 0:
            left, right = points[0:2]
        elif position >= len(points):
            left, right = points[-2:]
        elif math.isclose(sources[position], source, rel_tol=0.0, abs_tol=1e-15):
            return points[position][0]
        else:
            left, right = points[position - 1 : position + 1]
        fraction = (source - left[1]) / (right[1] - left[1])
        return left[0] + fraction * (right[0] - left[0])


PTSProvider = Callable[[Path, float], tuple[float, ...]]
DecodeProvider = Callable[
    [Path, Iterable[int], ReferenceMatchConfig], DecodedTargetFrames
]
DescriptorProvider = Callable[[np.ndarray], FrameVisualDescriptor]


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _as_sequence(value: Any, name: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be an array")
    return list(value)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _plan_payload(plan: RenderPlan | Mapping[str, Any]) -> dict[str, Any]:
    return plan.to_json() if isinstance(plan, RenderPlan) else _as_mapping(plan, "plan")


def _boundary_contracts(
    plan: RenderPlan | Mapping[str, Any],
) -> tuple[_BoundaryContract, ...]:
    payload = _plan_payload(plan)
    fps = _finite(payload.get("fps"), "plan.fps")
    master_audio_start = _finite(
        payload.get("master_audio_start"), "plan.master_audio_start"
    )
    if fps <= 0.0 or master_audio_start < 0.0:
        raise ValueError("plan clocks are invalid")
    segments = tuple(
        _as_mapping(item, f"plan.segments[{index}]")
        for index, item in enumerate(_as_sequence(payload.get("segments"), "plan.segments"))
    )
    transitions = tuple(
        _as_mapping(item, f"plan.transitions[{index}]")
        for index, item in enumerate(
            _as_sequence(payload.get("transitions"), "plan.transitions")
        )
    )
    if not segments or len(transitions) != len(segments) - 1:
        raise ValueError("plan boundary arrays are inconsistent")
    result: list[_BoundaryContract] = []
    for boundary_index, (outgoing, incoming, transition) in enumerate(
        zip(segments[:-1], segments[1:], transitions, strict=True)
    ):
        outgoing_source = _integer(
            outgoing.get("source_index"),
            f"plan.segments[{boundary_index}].source_index",
        )
        incoming_source = _integer(
            incoming.get("source_index"),
            f"plan.segments[{boundary_index + 1}].source_index",
        )
        if outgoing_source == incoming_source:
            continue
        output_seconds = _finite(
            outgoing.get("master_start"),
            f"plan.segments[{boundary_index}].master_start",
        ) + _finite(
            outgoing.get("duration"),
            f"plan.segments[{boundary_index}].duration",
        )
        incoming_start = _finite(
            incoming.get("master_start"),
            f"plan.segments[{boundary_index + 1}].master_start",
        )
        if not math.isclose(output_seconds, incoming_start, abs_tol=1e-6):
            raise ValueError("plan boundary is not contiguous")
        outgoing_local = _finite(
            outgoing.get("source_local_end", outgoing.get("source_end")),
            f"plan.segments[{boundary_index}].source_local_end",
        )
        incoming_local = _finite(
            incoming.get("source_local_start", incoming.get("source_start")),
            f"plan.segments[{boundary_index + 1}].source_local_start",
        )
        boundary_payload = {
            "boundary_index": boundary_index,
            "outgoing_segment_index": boundary_index,
            "incoming_segment_index": boundary_index + 1,
            "outgoing_source_index": outgoing_source,
            "incoming_source_index": incoming_source,
            "output_seconds": output_seconds,
            "output_frame_index": int(round(output_seconds * fps)),
            "canonical_master_seconds": master_audio_start + output_seconds,
            "outgoing_source_local_seconds": outgoing_local,
            "incoming_source_local_seconds": incoming_local,
            "transition": transition,
        }
        result.append(
            _BoundaryContract(
                **boundary_payload,
                sha256=_canonical_sha256(boundary_payload),
            )
        )
    return tuple(result)


def _used_source_indices(plan: RenderPlan | Mapping[str, Any]) -> tuple[int, ...]:
    payload = _plan_payload(plan)
    segments = _as_sequence(payload.get("segments"), "plan.segments")
    return tuple(
        sorted(
            {
                _integer(
                    _as_mapping(item, f"plan.segments[{index}]").get("source_index"),
                    f"plan.segments[{index}].source_index",
                )
                for index, item in enumerate(segments)
            }
        )
    )


def _bound_timewarp_payloads(
    timewarp_payloads: Sequence[Mapping[str, Any]],
    used_source_indices: Sequence[int],
) -> tuple[dict[str, Any], ...]:
    by_index: dict[int, dict[str, Any]] = {}
    duplicates: set[int] = set()
    for position, raw in enumerate(timewarp_payloads):
        item = _as_mapping(raw, f"timewarp_payloads[{position}]")
        index = _integer(item.get("source_index"), f"timewarp_payloads[{position}].source_index")
        if index in by_index:
            duplicates.add(index)
        else:
            by_index[index] = item
    if duplicates:
        raise ValueError(f"duplicate timewarp source indices: {sorted(duplicates)}")
    missing = sorted(set(used_source_indices) - set(by_index))
    if missing:
        raise ValueError(f"used sources lack timewarp payloads: {missing}")
    return tuple(by_index[index] for index in sorted(used_source_indices))


def _strict_serialized_timewarp(
    payload: Mapping[str, Any],
    *,
    expected_source_index: int,
) -> _SerializedTimeWarp:
    item = _as_mapping(payload, "timewarp")
    if _integer(item.get("source_index"), "timewarp.source_index") != expected_source_index:
        raise ValueError("timewarp_source_index_mismatch")
    if item.get("status") != "local_monotonic_timewarp":
        raise ValueError("timewarp_status_invalid")
    if item.get("scalar_fallback") is not False or item.get("fallback_reason") is not None:
        raise ValueError("timewarp_piecewise_status_invalid")
    minimum_slope = _finite(item.get("minimum_slope"), "timewarp.minimum_slope")
    maximum_slope = _finite(item.get("maximum_slope"), "timewarp.maximum_slope")
    if not 0.0 < minimum_slope <= maximum_slope:
        raise ValueError("timewarp_slope_bounds_invalid")
    raw_points = _as_sequence(item.get("control_points"), "timewarp.control_points")
    if len(raw_points) < 2:
        raise ValueError("timewarp_control_points_insufficient")
    points: list[tuple[float, float]] = []
    for index, raw_point in enumerate(raw_points):
        point = _as_mapping(raw_point, f"timewarp.control_points[{index}]")
        master = _finite(
            point.get("master_seconds"),
            f"timewarp.control_points[{index}].master_seconds",
        )
        source = _finite(
            point.get("source_seconds"),
            f"timewarp.control_points[{index}].source_seconds",
        )
        offset = _finite(
            point.get("offset_seconds"),
            f"timewarp.control_points[{index}].offset_seconds",
        )
        if not math.isclose(offset, source - master, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("timewarp_control_point_offset_mismatch")
        points.append((master, source))
    slopes: list[float] = []
    for left, right in zip(points, points[1:]):
        master_delta = right[0] - left[0]
        source_delta = right[1] - left[1]
        if master_delta <= 0.0 or source_delta <= 0.0:
            raise ValueError("timewarp_control_points_not_strictly_monotonic")
        slope = source_delta / master_delta
        if slope < minimum_slope - 1e-9 or slope > maximum_slope + 1e-9:
            raise ValueError("timewarp_control_point_slope_out_of_bounds")
        slopes.append(slope)
    observed_minimum = _finite(
        item.get("observed_minimum_slope"), "timewarp.observed_minimum_slope"
    )
    observed_maximum = _finite(
        item.get("observed_maximum_slope"), "timewarp.observed_maximum_slope"
    )
    mean_slope = _finite(item.get("mean_slope"), "timewarp.mean_slope")
    ppm = _finite(
        item.get("estimated_drift_parts_per_million"),
        "timewarp.estimated_drift_parts_per_million",
    )
    expected_mean = float(np.mean(np.asarray(slopes, dtype=np.float64)))
    if not math.isclose(observed_minimum, min(slopes), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("timewarp_observed_minimum_slope_mismatch")
    if not math.isclose(observed_maximum, max(slopes), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("timewarp_observed_maximum_slope_mismatch")
    if not math.isclose(mean_slope, expected_mean, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("timewarp_mean_slope_mismatch")
    if not math.isclose(
        ppm, (expected_mean - 1.0) * 1e6, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError("timewarp_drift_ppm_mismatch")
    return _SerializedTimeWarp(
        source_index=expected_source_index,
        minimum_slope=minimum_slope,
        maximum_slope=maximum_slope,
        control_points=tuple(points),
    )


def _runtime_bound_timewarps(
    *,
    clocks: Sequence[TimeWarpMap | None],
    bound_payloads: Sequence[Mapping[str, Any]],
    used_source_indices: Sequence[int],
) -> dict[int, TimeWarpMap]:
    if len(bound_payloads) != len(used_source_indices):
        raise ValueError("used_timewarp_payload_runtime_mismatch")
    result: dict[int, TimeWarpMap] = {}
    for source_index, raw_payload in zip(
        used_source_indices, bound_payloads, strict=True
    ):
        if source_index >= len(clocks):
            raise ValueError("used_timewarp_payload_runtime_mismatch")
        clock = clocks[source_index]
        if clock is None or clock.scalar_fallback:
            raise ValueError("used_timewarp_payload_runtime_mismatch")
        expected = {
            **clock.to_dict(),
            "source_index": source_index,
            "status": "local_monotonic_timewarp",
        }
        if dict(raw_payload) != expected:
            raise ValueError("used_timewarp_payload_runtime_mismatch")
        result[source_index] = clock
    return result


def _probe_actual_video_pts(path: Path, timeout_seconds: float) -> tuple[float, ...]:
    command = (
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "csv=p=0",
        str(path),
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _SourceEvidenceError("pts_probe_spawn_failed") from exc
    if completed.returncode != 0:
        raise _SourceEvidenceError("pts_probe_nonzero_exit")
    values: list[float] = []
    for line in completed.stdout.splitlines():
        token = line.strip().split(",", 1)[0]
        if not token:
            continue
        try:
            value = float(token)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    if len(values) < 2 or any(
        right <= left for left, right in zip(values, values[1:])
    ):
        raise _SourceEvidenceError("pts_incomplete_or_non_monotonic")
    return tuple(values)


def renderer_selected_frame_ordinals(
    normalized_pts_seconds: Sequence[float],
    *,
    source_start_seconds: float,
    source_duration_seconds: float,
    timeline_duration_seconds: float,
    output_fps: float,
    frame_count: int,
) -> tuple[int, ...]:
    """Mirror the renderer's ``trim -> setpts -> fps -> tpad`` ownership.

    ``trim=start`` owns the first presentation timestamp greater than or equal
    to the requested start.  ``setpts`` rebases that first retained timestamp,
    and FFmpeg's default ``fps=round=near`` maps input timestamps to output
    buckets, keeping the last input frame when several land in one bucket.
    Missing buckets duplicate the preceding frame; the renderer's following
    ``tpad=stop_mode=clone`` extends the last selected frame to ``frame_count``.
    Returned values are presentation-order ordinals into the ffprobe PTS list,
    which are also the ordinals required from the forward decoder.
    """

    pts = tuple(
        _finite(value, f"normalized_pts_seconds[{index}]")
        for index, value in enumerate(normalized_pts_seconds)
    )
    if len(pts) < 2 or any(right <= left for left, right in zip(pts, pts[1:])):
        raise ValueError("renderer selection PTS must be strictly increasing")
    start = _finite(source_start_seconds, "source_start_seconds")
    source_duration = _finite(source_duration_seconds, "source_duration_seconds")
    timeline_duration = _finite(
        timeline_duration_seconds, "timeline_duration_seconds"
    )
    fps = _finite(output_fps, "output_fps")
    if start < 0.0 or source_duration <= 0.0 or timeline_duration <= 0.0 or fps <= 0.0:
        raise ValueError("renderer selection clocks must be positive")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
        raise ValueError("renderer selection frame_count must be positive")

    # `_video_chain` writes trim/fps clocks at six decimals and a non-identity
    # setpts multiplier at twelve decimals.  Apply those exact serialized
    # values here so a sub-microsecond Python-only clock cannot choose a frame
    # that the production filter graph would not own.
    renderer_start = float(f"{start:.6f}")
    renderer_source_duration = float(f"{source_duration:.6f}")
    renderer_fps = float(f"{fps:.6f}")
    raw_playback_scale = timeline_duration / source_duration
    playback_scale = (
        1.0
        if math.isclose(raw_playback_scale, 1.0, rel_tol=0.0, abs_tol=1e-9)
        else float(f"{raw_playback_scale:.12f}")
    )
    first = bisect_left(pts, renderer_start)
    end = bisect_left(pts, renderer_start + renderer_source_duration)
    if first >= len(pts) or end <= first:
        raise _SourceEvidenceError("renderer_trim_has_no_frame")
    retained = tuple(range(first, end))
    first_pts = pts[first]
    buckets: dict[int, int] = {}
    for ordinal in retained:
        adjusted = (pts[ordinal] - first_pts) * playback_scale
        # FFmpeg AV_ROUND_NEAR_INF rounds a non-negative exact half away from
        # zero.  A tiny tolerance absorbs decimal ffprobe representation noise.
        bucket = int(math.floor(adjusted * renderer_fps + 0.5 + 1e-12))
        if bucket < 0:
            continue
        buckets[bucket] = ordinal
    if not buckets or min(buckets) != 0:
        raise _SourceEvidenceError("renderer_fps_has_no_zero_bucket")

    selected: list[int] = []
    current: int | None = None
    for output_index in range(frame_count):
        if output_index in buckets:
            current = buckets[output_index]
        if current is None:
            raise _SourceEvidenceError("renderer_fps_initial_bucket_missing")
        selected.append(current)
    return tuple(selected)


def _renderer_windows(
    plan: Mapping[str, Any],
    boundaries: Sequence[_BoundaryContract],
    clocks: Mapping[int, TimeWarpMap],
    settings: CutLocalSyncConfig,
) -> tuple[tuple[_RendererWindow, ...], dict[int, str]]:
    fps = _finite(plan.get("fps"), "plan.fps")
    segments = tuple(
        _as_mapping(item, f"plan.segments[{index}]")
        for index, item in enumerate(_as_sequence(plan.get("segments"), "plan.segments"))
    )
    result: list[_RendererWindow] = []
    errors: dict[int, str] = {}
    for boundary in boundaries:
        transition_kind = boundary.transition.get("kind")
        transition_frames = _integer(
            boundary.transition.get("frame_count"),
            f"boundary[{boundary.boundary_index}].transition.frame_count",
        )
        if transition_kind == "face_dissolve":
            grids = tuple(range(transition_frames))
            masters = tuple(
                boundary.canonical_master_seconds + grid / fps for grid in grids
            )
            outgoing = segments[boundary.outgoing_segment_index]
            incoming = segments[boundary.incoming_segment_index]
            outgoing_duration = _finite(
                outgoing.get(
                    "effective_source_duration",
                    outgoing.get("source_duration", outgoing.get("duration")),
                ),
                "outgoing.effective_source_duration",
            )
            outgoing_timeline = _finite(outgoing.get("duration"), "outgoing.duration")
            outgoing_rate = outgoing_duration / outgoing_timeline
            outgoing_start = _finite(
                outgoing.get("source_local_end", outgoing.get("source_end")),
                "outgoing.source_local_end",
            )
            incoming_start = _finite(
                incoming.get("source_local_start", incoming.get("source_start")),
                "incoming.source_local_start",
            )
            incoming_duration = _finite(
                incoming.get(
                    "effective_source_duration",
                    incoming.get("source_duration", incoming.get("duration")),
                ),
                "incoming.effective_source_duration",
            )
            incoming_timeline = _finite(incoming.get("duration"), "incoming.duration")
            handle_timeline = transition_frames / fps
            result.extend(
                (
                    _RendererWindow(
                        boundary_index=boundary.boundary_index,
                        role="outgoing",
                        source_index=boundary.outgoing_source_index,
                        grid_indices=grids,
                        requested_master_seconds=masters,
                        source_start_seconds=outgoing_start,
                        source_duration_seconds=handle_timeline * outgoing_rate,
                        timeline_duration_seconds=handle_timeline,
                        frame_count=transition_frames,
                    ),
                    _RendererWindow(
                        boundary_index=boundary.boundary_index,
                        role="incoming",
                        source_index=boundary.incoming_source_index,
                        grid_indices=grids,
                        requested_master_seconds=masters,
                        source_start_seconds=incoming_start,
                        source_duration_seconds=incoming_duration,
                        timeline_duration_seconds=incoming_timeline,
                        frame_count=transition_frames,
                    ),
                )
            )
            continue
        if transition_kind != "hard_cut" or transition_frames != 0:
            raise ValueError("unsupported_cut_local_transition_contract")
        grids = tuple(
            range(-settings.sample_radius_frames, settings.sample_radius_frames + 1)
        )
        master_start = (
            boundary.canonical_master_seconds - settings.sample_radius_frames / fps
        )
        timeline_duration = len(grids) / fps
        master_end = master_start + timeline_duration
        masters = tuple(master_start + position / fps for position in range(len(grids)))
        for role, source_index in (
            ("outgoing", boundary.outgoing_source_index),
            ("incoming", boundary.incoming_source_index),
        ):
            clock = clocks[source_index]
            source_start = clock.master_to_source(master_start)
            source_end = clock.master_to_source(master_end)
            if source_start < 0.0 or source_end <= source_start:
                errors[boundary.boundary_index] = (
                    "hard_cut_context_outside_source_clock"
                )
                continue
            result.append(
                _RendererWindow(
                    boundary_index=boundary.boundary_index,
                    role=role,
                    source_index=source_index,
                    grid_indices=grids,
                    requested_master_seconds=masters,
                    source_start_seconds=source_start,
                    source_duration_seconds=source_end - source_start,
                    timeline_duration_seconds=timeline_duration,
                    frame_count=len(grids),
                )
            )
    return (tuple(result), errors)


def _safe_source_error_code(exc: BaseException) -> str:
    if isinstance(exc, _SourceEvidenceError):
        return exc.code
    if isinstance(exc, cv2.error):
        return "opencv_decode_or_descriptor_failed"
    if isinstance(exc, OSError):
        return "source_io_failed"
    if isinstance(exc, TypeError):
        return "source_evidence_type_invalid"
    if isinstance(exc, ValueError):
        return "source_evidence_value_invalid"
    return "source_evidence_runtime_failed"


def _rms_delta(left: Sequence[float], right: Sequence[float]) -> float | None:
    if not left or len(left) != len(right):
        return None
    values = tuple(float(a) - float(b) for a, b in zip(left, right, strict=True))
    if not all(math.isfinite(value) for value in values):
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def _actual_pose(descriptor: FrameVisualDescriptor) -> bool:
    return bool(
        descriptor.pose_embedding
        and descriptor.pose_backend.startswith("opencv_zoo_mediapipe_pose")
    )


def _formation_motion(
    left: FrameVisualDescriptor,
    right: FrameVisualDescriptor,
) -> float | None:
    if not left.formation_people or len(left.formation_people) != len(
        right.formation_people
    ):
        return None
    squared: list[float] = []
    for outgoing, incoming in zip(
        left.formation_people, right.formation_people, strict=True
    ):
        if not outgoing.upper_body_keypoints_xyc or len(
            outgoing.upper_body_keypoints_xyc
        ) != len(incoming.upper_body_keypoints_xyc):
            return None
        for left_point, right_point in zip(
            outgoing.upper_body_keypoints_xyc,
            incoming.upper_body_keypoints_xyc,
            strict=True,
        ):
            if min(float(left_point[2]), float(right_point[2])) < 0.20:
                continue
            squared.append(
                (float(left_point[0]) - float(right_point[0])) ** 2
                + (float(left_point[1]) - float(right_point[1])) ** 2
            )
    if len(squared) < 4:
        return None
    return math.sqrt(sum(squared) / len(squared))


def _silhouette_motion(
    left: FrameVisualDescriptor,
    right: FrameVisualDescriptor,
) -> float | None:
    if (
        left.silhouette_centroid_xy is None
        or right.silhouette_centroid_xy is None
        or left.silhouette_bbox_xywh is None
        or right.silhouette_bbox_xywh is None
        or left.silhouette_backend == "none"
        or right.silhouette_backend == "none"
        or min(left.silhouette_confidence, right.silhouette_confidence) < 0.35
    ):
        return None
    center = math.hypot(
        left.silhouette_centroid_xy[0] - right.silhouette_centroid_xy[0],
        left.silhouette_centroid_xy[1] - right.silhouette_centroid_xy[1],
    )
    left_area = max(left.silhouette_bbox_xywh[2] * left.silhouette_bbox_xywh[3], 1e-6)
    right_area = max(
        right.silhouette_bbox_xywh[2] * right.silhouette_bbox_xywh[3], 1e-6
    )
    return math.hypot(center, 0.18 * abs(math.log(left_area / right_area)))


def _descriptor_motion(
    left: FrameVisualDescriptor,
    right: FrameVisualDescriptor,
) -> tuple[float, str] | None:
    if _actual_pose(left) and _actual_pose(right):
        value = _rms_delta(left.pose_embedding, right.pose_embedding)
        if value is not None:
            return (value, "actual_pose_motion")
    formation = _formation_motion(left, right)
    if formation is not None:
        return (formation, "formation_upper_body_motion")
    layout = _rms_delta(
        left.person_layout_embedding,
        right.person_layout_embedding,
    )
    if layout is not None:
        return (layout, "person_layout_motion")
    silhouette = _silhouette_motion(left, right)
    if silhouette is not None:
        return (silhouette, "silhouette_body_motion")
    return None


def _motion_series(
    observations: Mapping[int, _FrameObservation],
    *,
    maximum_pts_lookup_residual_frames: float,
) -> tuple[CutLocalMotionSample, ...]:
    result: list[CutLocalMotionSample] = []
    for offset in sorted(observations):
        following = offset + 1
        if following not in observations:
            continue
        left, right = observations[offset], observations[following]
        if left.frame_index == right.frame_index:
            continue
        if max(
            left.pts_lookup_residual_frames,
            right.pts_lookup_residual_frames,
        ) > maximum_pts_lookup_residual_frames:
            continue
        motion = _descriptor_motion(left.descriptor, right.descriptor)
        if motion is None:
            continue
        value, basis = motion
        result.append(
            CutLocalMotionSample(
                master_seconds=(left.actual_master_seconds + right.actual_master_seconds)
                / 2.0,
                value=value,
                basis=basis,
                grid_index=offset,
            )
        )
    return tuple(result)


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size != left.size:
        return None
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denominator = float(
        math.sqrt(
            float(np.dot(left_centered, left_centered))
            * float(np.dot(right_centered, right_centered))
        )
    )
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float(np.clip(np.dot(left_centered, right_centered) / denominator, -1.0, 1.0))


def estimate_motion_phase(
    outgoing: Sequence[CutLocalMotionSample],
    incoming: Sequence[CutLocalMotionSample],
    *,
    output_fps: float,
    config: CutLocalSyncConfig | None = None,
) -> dict[str, Any]:
    """Estimate one cut's phase residual, preserving ambiguity as UNKNOWN."""

    settings = config or CutLocalSyncConfig()
    fps = _finite(output_fps, "output_fps")
    if fps <= 0.0:
        raise ValueError("output_fps must be positive")
    left = tuple(outgoing)
    right = tuple(incoming)
    candidates: list[dict[str, Any]] = []
    explicit_grid = all(item.grid_index is not None for item in (*left, *right))
    if explicit_grid:
        left_by_grid = {int(item.grid_index): item for item in left}
        right_by_grid = {int(item.grid_index): item for item in right}
        if len(left_by_grid) != len(left) or len(right_by_grid) != len(right):
            return {
                "status": "unknown",
                "reason": "duplicate_motion_grid_index",
                "outgoing_motion_sample_count": len(left),
                "incoming_motion_sample_count": len(right),
            }
    for lag in range(-settings.maximum_lag_frames, settings.maximum_lag_frames + 1):
        if explicit_grid:
            pairs = [
                (left_sample, right_by_grid[grid_index + lag])
                for grid_index, left_sample in sorted(left_by_grid.items())
                if grid_index + lag in right_by_grid
            ]
        else:
            pairs = []
            for left_index, left_sample in enumerate(left):
                right_index = left_index + lag
                if 0 <= right_index < len(right):
                    pairs.append((left_sample, right[right_index]))
        left_values = np.asarray([item[0].value for item in pairs], dtype=np.float64)
        right_values = np.asarray([item[1].value for item in pairs], dtype=np.float64)
        left_std = float(np.std(left_values)) if pairs else None
        right_std = float(np.std(right_values)) if pairs else None
        correlation = (
            _pearson(left_values, right_values)
            if len(pairs) >= settings.minimum_motion_pair_count
            and left_std is not None
            and right_std is not None
            and min(left_std, right_std)
            >= settings.minimum_motion_standard_deviation
            else None
        )
        non_silhouette_pairs = sum(
            item[0].basis != "silhouette_body_motion"
            and item[1].basis != "silhouette_body_motion"
            for item in pairs
        )
        phase_frames = (
            float(
                np.median(
                    np.asarray(
                        [
                            (item[1].master_seconds - item[0].master_seconds)
                            * fps
                            for item in pairs
                        ],
                        dtype=np.float64,
                    )
                )
            )
            if pairs
            else None
        )
        basis_counts = dict(
            sorted(
                Counter(
                    f"{item[0].basis}+{item[1].basis}" for item in pairs
                ).items()
            )
        )
        candidates.append(
            {
                "lag_frames": lag,
                "pair_count": len(pairs),
                "non_silhouette_pair_count": non_silhouette_pairs,
                "correlation": correlation,
                "outgoing_motion_stddev": left_std,
                "incoming_motion_stddev": right_std,
                "actual_pts_phase_frames": phase_frames,
                "motion_basis_pair_counts": basis_counts,
                "eligible": bool(
                    correlation is not None
                    and non_silhouette_pairs
                    >= settings.minimum_non_silhouette_pair_count
                ),
            }
        )

    provenance = {
        "outgoing": [
            {
                "grid_index": item.grid_index,
                "actual_master_seconds": item.master_seconds,
                "basis": item.basis,
            }
            for item in left
        ],
        "incoming": [
            {
                "grid_index": item.grid_index,
                "actual_master_seconds": item.master_seconds,
                "basis": item.basis,
            }
            for item in right
        ],
    }
    if min(len(left), len(right)) < settings.minimum_motion_pair_count:
        return {
            "status": "unknown",
            "reason": "insufficient_motion_samples",
            "outgoing_motion_sample_count": len(left),
            "incoming_motion_sample_count": len(right),
            "motion_timing_provenance": provenance,
            "lag_candidates": candidates,
        }

    eligible = tuple(
        item
        for item in candidates
        if item["eligible"] is True
    )
    if not eligible:
        return {
            "status": "unknown",
            "reason": "insufficient_nonstatic_pose_body_motion",
            "outgoing_motion_sample_count": len(left),
            "incoming_motion_sample_count": len(right),
            "motion_timing_provenance": provenance,
            "lag_candidates": candidates,
        }
    ordered = sorted(
        eligible,
        key=lambda item: (float(item["correlation"]), -abs(int(item["lag_frames"]))),
        reverse=True,
    )
    best = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else None
    best_correlation = float(best["correlation"])
    margin = (
        best_correlation - float(runner_up["correlation"])
        if runner_up is not None
        else 2.0
    )
    if best_correlation < settings.minimum_correlation:
        reason = "motion_correlation_below_minimum"
    elif margin < settings.minimum_correlation_margin:
        reason = "ambiguous_motion_lag"
    elif abs(int(best["lag_frames"])) >= settings.maximum_lag_frames:
        reason = "motion_lag_at_search_boundary"
    else:
        reason = None
    if reason is not None:
        return {
            "status": "unknown",
            "reason": reason,
            "outgoing_motion_sample_count": len(left),
            "incoming_motion_sample_count": len(right),
            "best_lag_frames": best["lag_frames"],
            "best_correlation": best_correlation,
            "runner_up_correlation": (
                runner_up["correlation"] if runner_up is not None else None
            ),
            "correlation_margin": margin,
            "motion_timing_provenance": provenance,
            "lag_candidates": candidates,
        }

    by_lag = {int(item["lag_frames"]): item for item in eligible}
    lag = int(best["lag_frames"])
    subframe = 0.0
    if lag - 1 in by_lag and lag + 1 in by_lag:
        left_score = float(by_lag[lag - 1]["correlation"])
        center_score = best_correlation
        right_score = float(by_lag[lag + 1]["correlation"])
        denominator = left_score - 2.0 * center_score + right_score
        if denominator < -1e-12:
            subframe = float(
                np.clip(
                    0.5 * (left_score - right_score) / denominator,
                    -0.5,
                    0.5,
                )
            )
    signed_phase = float(best["actual_pts_phase_frames"]) + subframe
    residual = abs(signed_phase)
    return {
        "status": "measured",
        "reason": None,
        "outgoing_motion_sample_count": len(left),
        "incoming_motion_sample_count": len(right),
        "best_lag_frames": lag,
        "subframe_peak_offset_frames": subframe,
        "actual_pts_phase_frames": best["actual_pts_phase_frames"],
        "signed_phase_residual_frames": signed_phase,
        "residual_frames": residual,
        "best_correlation": best_correlation,
        "runner_up_correlation": (
            runner_up["correlation"] if runner_up is not None else None
        ),
        "correlation_margin": margin,
        "matched_motion_pair_count": best["pair_count"],
        "matched_non_silhouette_pair_count": best[
            "non_silhouette_pair_count"
        ],
        "motion_basis_pair_counts": best["motion_basis_pair_counts"],
        "motion_timing_provenance": provenance,
        "lag_candidates": candidates,
    }


def _decode_provider(
    path: Path,
    indices: Iterable[int],
    config: ReferenceMatchConfig,
) -> DecodedTargetFrames:
    return decode_target_frames_sequential(path, indices, config=config)


def _descriptor_provider(frame: np.ndarray) -> FrameVisualDescriptor:
    return describe_frame(frame, include_formation_pose=True)


def _quantile95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.95))


def analyze_cut_local_sync(
    plan: RenderPlan,
    *,
    source_timewarps: Sequence[TimeWarpMap | None],
    timewarp_payloads: Sequence[Mapping[str, Any]],
    config: CutLocalSyncConfig | None = None,
    pts_provider: PTSProvider = _probe_actual_video_pts,
    decode_provider: DecodeProvider = _decode_provider,
    descriptor_provider: DescriptorProvider = _descriptor_provider,
) -> dict[str, Any]:
    """Measure every final cross-source boundary on actual decoded source PTS."""

    if not isinstance(plan, RenderPlan):
        raise TypeError("cut-local sync requires a RenderPlan")
    settings = config or CutLocalSyncConfig()
    plan_payload = plan.to_json()
    boundaries = _boundary_contracts(plan_payload)
    used_sources = _used_source_indices(plan_payload)
    clocks = tuple(source_timewarps)
    if len(clocks) != len(plan.sources):
        raise ValueError("source_timewarps must match RenderPlan.sources")
    bound_timewarps = _bound_timewarp_payloads(timewarp_payloads, used_sources)
    runtime_clocks = _runtime_bound_timewarps(
        clocks=clocks,
        bound_payloads=bound_timewarps,
        used_source_indices=used_sources,
    )
    plan_sha = _canonical_sha256(plan_payload)
    timewarp_sha = _canonical_sha256(bound_timewarps)
    boundary_clock_sha = _canonical_sha256(
        [item.payload() for item in boundaries]
    )
    fps = plan.fps
    boundary_clock_errors: dict[int, str] = {}
    for boundary in boundaries:
        for source_index, expected_local in (
            (
                boundary.outgoing_source_index,
                boundary.outgoing_source_local_seconds,
            ),
            (
                boundary.incoming_source_index,
                boundary.incoming_source_local_seconds,
            ),
        ):
            clock = runtime_clocks[source_index]
            mapped = clock.master_to_source(boundary.canonical_master_seconds)
            if not math.isclose(mapped, expected_local, rel_tol=0.0, abs_tol=1e-6):
                boundary_clock_errors[boundary.boundary_index] = (
                    "plan_source_local_endpoint_timewarp_mismatch"
                )
    windows, window_errors = _renderer_windows(
        plan_payload, boundaries, runtime_clocks, settings
    )
    windows_by_source: dict[int, list[_RendererWindow]] = {
        source_index: [] for source_index in used_sources
    }
    for window in windows:
        windows_by_source[window.source_index].append(window)

    observations: dict[
        tuple[int, str], dict[int, _FrameObservation]
    ] = {}
    renderer_selection: dict[tuple[int, str], dict[str, Any]] = {}
    source_audit: list[dict[str, Any]] = []
    decode_config = ReferenceMatchConfig(working_width=settings.working_width)
    for source_index in used_sources:
        path = plan.sources[source_index]
        clock = runtime_clocks[source_index]
        try:
            raw_pts = tuple(
                _finite(value, f"source_pts[{index}]")
                for index, value in enumerate(
                    pts_provider(path, settings.ffprobe_timeout_seconds)
                )
            )
            if len(raw_pts) < 2 or any(
                right <= left for left, right in zip(raw_pts, raw_pts[1:])
            ):
                raise _SourceEvidenceError("pts_incomplete_or_non_monotonic")
            pts_origin = raw_pts[0]
            normalized_pts = tuple(value - pts_origin for value in raw_pts)
            lookup: dict[
                tuple[int, str, int], tuple[int, float, float, float]
            ] = {}
            indices: set[int] = set()
            for window in windows_by_source[source_index]:
                ordinals = renderer_selected_frame_ordinals(
                    normalized_pts,
                    source_start_seconds=window.source_start_seconds,
                    source_duration_seconds=window.source_duration_seconds,
                    timeline_duration_seconds=window.timeline_duration_seconds,
                    output_fps=fps,
                    frame_count=window.frame_count,
                )
                if len(ordinals) != len(window.grid_indices):
                    raise _SourceEvidenceError("renderer_selection_length_mismatch")
                rate = (
                    window.source_duration_seconds
                    / window.timeline_duration_seconds
                )
                selected_pts = tuple(normalized_pts[index] for index in ordinals)
                renderer_selection[(window.boundary_index, window.role)] = {
                    "basis": RENDERER_SELECTION_BASIS,
                    "source_index": source_index,
                    "grid_indices": list(window.grid_indices),
                    "frame_ordinals": list(ordinals),
                    "normalized_pts_seconds": list(selected_pts),
                    "requested_master_seconds": list(
                        window.requested_master_seconds
                    ),
                    "source_start_seconds": window.source_start_seconds,
                    "source_duration_seconds": window.source_duration_seconds,
                    "timeline_duration_seconds": window.timeline_duration_seconds,
                    "frame_count": window.frame_count,
                }
                for position, (grid, master, frame_index) in enumerate(
                    zip(
                        window.grid_indices,
                        window.requested_master_seconds,
                        ordinals,
                        strict=True,
                    )
                ):
                    indices.add(frame_index)
                    ideal_source = (
                        window.source_start_seconds + position / fps * rate
                    )
                    actual_master = clock.source_to_master(
                        normalized_pts[frame_index]
                    )
                    lookup[(window.boundary_index, window.role, grid)] = (
                        frame_index,
                        actual_master,
                        ideal_source,
                        abs(actual_master - master) * fps,
                    )
            requested_indices = tuple(sorted(indices))
            decoded = decode_provider(path, requested_indices, decode_config)
            if decoded.declared_frame_count != len(raw_pts):
                raise _SourceEvidenceError("pts_decode_ordinal_count_mismatch")
            if tuple(decoded.requested_indices) != requested_indices:
                raise _SourceEvidenceError("decoder_requested_ordinal_mismatch")
            if set(decoded.frames) != set(requested_indices):
                raise _SourceEvidenceError("decoder_frame_ordinal_set_mismatch")
            descriptors: dict[int, FrameVisualDescriptor] = {}
            for frame_index in requested_indices:
                frame = decoded.frames[frame_index]
                try:
                    descriptors[frame_index] = descriptor_provider(frame)
                except (cv2.error, RuntimeError, TypeError, ValueError) as exc:
                    raise _SourceEvidenceError(
                        "descriptor_extraction_incomplete"
                    ) from exc
            for key, (frame_index, actual_master, source_seconds, residual) in lookup.items():
                boundary_index, role, grid = key
                selection = renderer_selection[(boundary_index, role)]
                grid_position = selection["grid_indices"].index(grid)
                requested_master = selection["requested_master_seconds"][grid_position]
                observations.setdefault((boundary_index, role), {})[grid] = _FrameObservation(
                    frame_index=frame_index,
                    requested_master_seconds=requested_master,
                    requested_source_seconds=source_seconds,
                    actual_pts_seconds=raw_pts[frame_index],
                    normalized_pts_seconds=normalized_pts[frame_index],
                    actual_master_seconds=actual_master,
                    pts_lookup_residual_frames=residual,
                    descriptor=descriptors[frame_index],
                )
            residuals = [item[3] for item in lookup.values()]
            positive_steps = [
                right - left
                for left, right in zip(normalized_pts, normalized_pts[1:])
                if right > left
            ]
            source_audit.append(
                {
                    "source_index": source_index,
                    "source_name": path.name,
                    "status": "decoded",
                    "pts_basis": PTS_BASIS,
                    "renderer_selection_basis": RENDERER_SELECTION_BASIS,
                    "pts_frame_count": len(raw_pts),
                    "decoded_declared_frame_count": decoded.declared_frame_count,
                    "pts_decode_ordinal_count_match": True,
                    "pts_origin_seconds": pts_origin,
                    "pts_sha256": _canonical_sha256(raw_pts),
                    "median_pts_step_seconds": (
                        float(np.median(np.asarray(positive_steps, dtype=np.float64)))
                        if positive_steps
                        else None
                    ),
                    "requested_ordinal_count": len(requested_indices),
                    "selected_ordinal_pts_sha256": _canonical_sha256(
                        [
                            {
                                "frame_ordinal": frame_index,
                                "normalized_pts_seconds": normalized_pts[frame_index],
                            }
                            for frame_index in requested_indices
                        ]
                    ),
                    "decoded_requested_ordinal_count": len(decoded.requested_indices),
                    "decoded_frame_count": len(decoded.frames),
                    "described_frame_count": len(descriptors),
                    "descriptor_failure_count": 0,
                    "maximum_pts_lookup_residual_frames": (
                        max(residuals) if residuals else None
                    ),
                    "raw_frames_persisted": False,
                    "motion_descriptors_persisted": False,
                }
            )
        except (cv2.error, OSError, RuntimeError, TypeError, ValueError) as exc:
            for window in windows_by_source[source_index]:
                observations.pop((window.boundary_index, window.role), None)
            source_audit.append(
                {
                    "source_index": source_index,
                    "source_name": path.name,
                    "status": "unavailable",
                    "reason": _safe_source_error_code(exc),
                    "pts_basis": PTS_BASIS,
                    "renderer_selection_basis": RENDERER_SELECTION_BASIS,
                }
            )

    records: list[dict[str, Any]] = []
    unknown_reasons: Counter[str] = Counter()
    measured_residuals: list[float] = []
    for boundary in boundaries:
        base = {
            **boundary.payload(),
            "boundary_contract_sha256": boundary.sha256,
        }
        transition_kind = boundary.transition.get("kind")
        selection_contract = {
            "outgoing": renderer_selection.get(
                (boundary.boundary_index, "outgoing")
            ),
            "incoming": renderer_selection.get(
                (boundary.boundary_index, "incoming")
            ),
        }
        sampling_base = {
            "sampling_domain": (
                "face_dissolve_exact_outgoing_handle_and_incoming_overlap"
                if transition_kind == "face_dissolve"
                else "hard_cut_symmetric_canonical_context"
            ),
            "renderer_frame_selection": selection_contract,
        }
        if boundary.boundary_index in boundary_clock_errors:
            reason = boundary_clock_errors[boundary.boundary_index]
            unknown_reasons[reason] += 1
            records.append(
                {**base, **sampling_base, "status": "unknown", "reason": reason}
            )
            continue
        if boundary.boundary_index in window_errors:
            reason = window_errors[boundary.boundary_index]
            unknown_reasons[reason] += 1
            records.append(
                {**base, **sampling_base, "status": "unknown", "reason": reason}
            )
            continue
        outgoing_observations = observations.get(
            (boundary.boundary_index, "outgoing"), {}
        )
        incoming_observations = observations.get(
            (boundary.boundary_index, "incoming"), {}
        )
        transition_frames = boundary.transition.get("frame_count")
        if (
            transition_kind == "face_dissolve"
            and isinstance(transition_frames, int)
            and transition_frames < settings.minimum_motion_pair_count + 1
        ):
            reason = "insufficient_dissolve_overlap_frames"
            unknown_reasons[reason] += 1
            records.append(
                {
                    **base,
                    **sampling_base,
                    "status": "unknown",
                    "reason": reason,
                }
            )
            continue
        phase = estimate_motion_phase(
            _motion_series(
                outgoing_observations,
                maximum_pts_lookup_residual_frames=(
                    settings.maximum_pts_lookup_residual_frames
                ),
            ),
            _motion_series(
                incoming_observations,
                maximum_pts_lookup_residual_frames=(
                    settings.maximum_pts_lookup_residual_frames
                ),
            ),
            output_fps=fps,
            config=settings,
        )
        if phase.get("status") != "measured":
            reason = str(phase.get("reason") or "unclassified_motion_uncertainty")
            unknown_reasons[reason] += 1
        else:
            measured_residuals.append(float(phase["residual_frames"]))
        records.append(
            {
                **base,
                **sampling_base,
                **phase,
            }
        )

    p95 = _quantile95(measured_residuals)
    unknown_count = sum(value for value in unknown_reasons.values())
    over_threshold = sum(
        value > settings.maximum_residual_p95_frames + 1e-12
        for value in measured_residuals
    )
    passed = bool(
        boundaries
        and len(measured_residuals) == len(boundaries)
        and unknown_count == 0
        and p95 is not None
        and p95 <= settings.maximum_residual_p95_frames + 1e-12
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_BASIS,
        "activation": ACTIVATION,
        "required": True,
        "plan_sha256": plan_sha,
        "timewarp_contract_sha256": timewarp_sha,
        "boundary_clock_contract_sha256": boundary_clock_sha,
        "fps": fps,
        "used_source_indices": list(used_sources),
        "retained_cross_source_boundary_count": len(boundaries),
        "measured_boundary_count": len(measured_residuals),
        "unknown_boundary_count": unknown_count,
        "unknown_reason_counts": dict(sorted(unknown_reasons.items())),
        "threshold_exceeding_boundary_count": over_threshold,
        "residual_p95_frames": p95,
        "maximum_allowed_residual_p95_frames": (
            settings.maximum_residual_p95_frames
        ),
        "passed": passed,
        "policy": settings.to_dict(),
        "source_decode_audit": source_audit,
        "per_boundary": records,
        "serialization_policy": dict(_SERIALIZATION_POLICY),
    }
    result["proof_sha256"] = _canonical_sha256(result)
    return result


def resolve_cut_local_sync_activation(
    analysis: Mapping[str, Any],
    selection: Mapping[str, Any] | None = None,
    *,
    trusted_registered_stage_edit: bool | None = None,
) -> dict[str, Any]:
    """Resolve StageEdit activation from independent persisted markers.

    The online automatic path supplies ``trusted_registered_stage_edit=True``
    because it still owns the verified registry bundle.  Offline QA has no
    such process-local authority, so it requires the generation marker and
    render-promotion marker to agree and also requires the StageEdit-only
    candidate lineage markers.  A valid explicit request remains optional.
    """

    selected = dict(selection) if isinstance(selection, Mapping) else {}
    generation = analysis.get("stage_edit_generation")
    generation_object = dict(generation) if isinstance(generation, Mapping) else None
    generation_activation_object = (
        generation_object.get("activation")
        if generation_object is not None
        else None
    )
    generation_activation = (
        generation_activation_object.get("activation")
        if isinstance(generation_activation_object, Mapping)
        else None
    )
    promotion = selected.get("stage_edit_render_promotion")
    promotion_object = dict(promotion) if isinstance(promotion, Mapping) else None
    promotion_activation = (
        promotion_object.get("activation")
        if promotion_object is not None
        else None
    )
    candidate_pool = (
        promotion_object.get("candidate_pool")
        if promotion_object is not None
        else None
    )
    render = analysis.get("render")
    render_object = dict(render) if isinstance(render, Mapping) else {}
    candidate_origin = render_object.get("candidate_origin")
    promotion_contract = render_object.get("stage_edit_promotion_contract")
    stage_edit_present = bool(
        generation is not None
        or promotion is not None
        or candidate_origin == "explicit_stage_edit_generation"
        or promotion_contract is not None
    )
    registered_seen = bool(
        generation_activation == "automatic_registered_production_bundle"
        or promotion_activation == "automatic_registered_production_bundle"
    )
    explicit_seen = bool(
        generation_activation == "explicit_request_bundle"
        or promotion_activation == "explicit_stage_edit_generation_request_only"
    )
    required = bool(trusted_registered_stage_edit is True or registered_seen)
    mode = (
        "automatic_registered_production_bundle"
        if required
        else "explicit_request_bundle"
        if stage_edit_present
        else "non_stage_edit_path"
    )
    errors: list[str] = []
    if trusted_registered_stage_edit is not None and not isinstance(
        trusted_registered_stage_edit, bool
    ):
        errors.append("trusted_stage_edit_activation_argument_invalid")
    if required:
        if generation_activation != "automatic_registered_production_bundle":
            errors.append("stage_edit_generation_registered_activation_missing_or_invalid")
        if promotion_activation != "automatic_registered_production_bundle":
            errors.append("stage_edit_render_promotion_registered_activation_missing_or_invalid")
    elif stage_edit_present:
        if generation_activation != "explicit_request_bundle":
            errors.append("stage_edit_generation_explicit_activation_missing_or_invalid")
        if promotion_activation != "explicit_stage_edit_generation_request_only":
            errors.append("stage_edit_render_promotion_explicit_activation_missing_or_invalid")
    if required and explicit_seen or not required and registered_seen:
        errors.append("stage_edit_activation_markers_disagree")
    if stage_edit_present:
        if candidate_pool != "stage_edit_generation_only":
            errors.append("stage_edit_candidate_pool_missing_or_invalid")
        if candidate_origin != "explicit_stage_edit_generation":
            errors.append("stage_edit_candidate_origin_missing_or_invalid")
    return {
        "required": required,
        "mode": mode,
        "stage_edit_present": stage_edit_present,
        "trusted_registered_stage_edit": trusted_registered_stage_edit,
        "generation_activation": generation_activation,
        "promotion_activation": promotion_activation,
        "candidate_pool": candidate_pool,
        "candidate_origin": candidate_origin,
        "errors": errors,
        "valid": not errors,
    }


def registered_cut_local_sync_required(analysis: Mapping[str, Any]) -> bool:
    """Compatibility shim; strict QA uses :func:`resolve_cut_local_sync_activation`."""

    return bool(resolve_cut_local_sync_activation(analysis).get("required"))


def _numbers_equal(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    try:
        return math.isclose(
            _finite(left, "left"),
            _finite(right, "right"),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    except ValueError:
        return False


def _expected_selection_contract(
    *,
    plan: Mapping[str, Any],
    boundary: _BoundaryContract,
    role: str,
    clock: _SerializedTimeWarp,
    policy: CutLocalSyncConfig,
) -> dict[str, Any]:
    fps = _finite(plan.get("fps"), "plan.fps")
    segments = _as_sequence(plan.get("segments"), "plan.segments")
    outgoing = _as_mapping(
        segments[boundary.outgoing_segment_index], "outgoing segment"
    )
    incoming = _as_mapping(
        segments[boundary.incoming_segment_index], "incoming segment"
    )
    kind = boundary.transition.get("kind")
    frame_count = _integer(
        boundary.transition.get("frame_count"), "transition.frame_count"
    )
    if kind == "face_dissolve":
        grids = tuple(range(frame_count))
        requested = tuple(
            boundary.canonical_master_seconds + index / fps for index in grids
        )
        if role == "outgoing":
            segment_duration = _finite(outgoing.get("duration"), "outgoing.duration")
            source_duration = _finite(
                outgoing.get(
                    "effective_source_duration",
                    outgoing.get("source_duration", outgoing.get("duration")),
                ),
                "outgoing.effective_source_duration",
            )
            timeline_duration = frame_count / fps
            selection_source_duration = (
                timeline_duration * source_duration / segment_duration
            )
            source_start = _finite(
                outgoing.get("source_local_end", outgoing.get("source_end")),
                "outgoing.source_local_end",
            )
            source_index = boundary.outgoing_source_index
        elif role == "incoming":
            source_start = _finite(
                incoming.get("source_local_start", incoming.get("source_start")),
                "incoming.source_local_start",
            )
            selection_source_duration = _finite(
                incoming.get(
                    "effective_source_duration",
                    incoming.get("source_duration", incoming.get("duration")),
                ),
                "incoming.effective_source_duration",
            )
            timeline_duration = _finite(incoming.get("duration"), "incoming.duration")
            source_index = boundary.incoming_source_index
        else:
            raise ValueError("selection_role_invalid")
        sampling_domain = (
            "face_dissolve_exact_outgoing_handle_and_incoming_overlap"
        )
    elif kind == "hard_cut" and frame_count == 0:
        grids = tuple(
            range(-policy.sample_radius_frames, policy.sample_radius_frames + 1)
        )
        requested_start = (
            boundary.canonical_master_seconds - policy.sample_radius_frames / fps
        )
        timeline_duration = len(grids) / fps
        requested = tuple(
            requested_start + position / fps for position in range(len(grids))
        )
        source_start = clock.master_to_source(requested_start)
        selection_source_duration = (
            clock.master_to_source(requested_start + timeline_duration)
            - source_start
        )
        source_index = (
            boundary.outgoing_source_index
            if role == "outgoing"
            else boundary.incoming_source_index
            if role == "incoming"
            else -1
        )
        if source_index < 0:
            raise ValueError("selection_role_invalid")
        sampling_domain = "hard_cut_symmetric_canonical_context"
    else:
        raise ValueError("selection_transition_invalid")
    return {
        "sampling_domain": sampling_domain,
        "basis": RENDERER_SELECTION_BASIS,
        "source_index": source_index,
        "grid_indices": grids,
        "requested_master_seconds": requested,
        "source_start_seconds": source_start,
        "source_duration_seconds": selection_source_duration,
        "timeline_duration_seconds": timeline_duration,
        "frame_count": len(grids),
    }


def _validate_renderer_selection_provenance(
    raw: Any,
    *,
    boundary: _BoundaryContract,
    role: str,
    plan: Mapping[str, Any],
    clock: _SerializedTimeWarp,
    policy: CutLocalSyncConfig,
    pts_frame_count: int | None,
    errors: list[str],
) -> dict[int, tuple[int, float, float, float]]:
    prefix = f"boundary_{boundary.boundary_index}_{role}_selection"
    if not isinstance(raw, Mapping):
        errors.append(f"{prefix}_missing_or_invalid")
        return {}
    item = dict(raw)
    try:
        expected = _expected_selection_contract(
            plan=plan,
            boundary=boundary,
            role=role,
            clock=clock,
            policy=policy,
        )
    except (TypeError, ValueError):
        errors.append(f"{prefix}_expected_contract_invalid")
        return {}
    for field in ("basis", "source_index", "frame_count"):
        if item.get(field) != expected[field]:
            errors.append(f"{prefix}_{field}_mismatch")
    if item.get("grid_indices") != list(expected["grid_indices"]):
        errors.append(f"{prefix}_grid_indices_mismatch")
    requested = item.get("requested_master_seconds")
    if not isinstance(requested, list) or len(requested) != len(
        expected["requested_master_seconds"]
    ) or any(
        not _numbers_equal(left, right)
        for left, right in zip(
            requested, expected["requested_master_seconds"], strict=False
        )
    ):
        errors.append(f"{prefix}_requested_master_clock_mismatch")
    for field in (
        "source_start_seconds",
        "source_duration_seconds",
        "timeline_duration_seconds",
    ):
        if not _numbers_equal(item.get(field), expected[field]):
            errors.append(f"{prefix}_{field}_mismatch")
    raw_ordinals = item.get("frame_ordinals")
    raw_pts = item.get("normalized_pts_seconds")
    count = len(expected["grid_indices"])
    if (
        not isinstance(raw_ordinals, list)
        or not isinstance(raw_pts, list)
        or len(raw_ordinals) != count
        or len(raw_pts) != count
    ):
        errors.append(f"{prefix}_frame_clock_array_length_mismatch")
        return {}
    ordinals: list[int] = []
    pts_values: list[float] = []
    for ordinal, pts in zip(raw_ordinals, raw_pts, strict=True):
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            errors.append(f"{prefix}_frame_ordinal_invalid")
            return {}
        if pts_frame_count is not None and ordinal >= pts_frame_count:
            errors.append(f"{prefix}_frame_ordinal_out_of_range")
        try:
            pts_value = _finite(pts, "selection pts")
        except ValueError:
            errors.append(f"{prefix}_normalized_pts_invalid")
            return {}
        ordinals.append(ordinal)
        pts_values.append(pts_value)
    for left_index, right_index in zip(range(count), range(1, count)):
        left_ordinal, right_ordinal = ordinals[left_index], ordinals[right_index]
        left_pts, right_pts = pts_values[left_index], pts_values[right_index]
        if right_ordinal < left_ordinal or right_pts < left_pts - 1e-12:
            errors.append(f"{prefix}_selection_not_monotonic")
            break
        if (right_ordinal == left_ordinal) != math.isclose(
            right_pts, left_pts, rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append(f"{prefix}_ordinal_pts_relation_invalid")
            break
    result: dict[int, tuple[int, float, float, float]] = {}
    expected_requested = expected["requested_master_seconds"]
    for grid, ordinal, pts_value, requested_master in zip(
        expected["grid_indices"],
        ordinals,
        pts_values,
        expected_requested,
        strict=True,
    ):
        actual_master = clock.source_to_master(pts_value)
        residual = abs(actual_master - requested_master) * _finite(
            plan.get("fps"), "plan.fps"
        )
        result[grid] = (ordinal, pts_value, actual_master, residual)
    return result


def _parse_motion_timing_provenance(
    raw: Any,
    *,
    boundary: _BoundaryContract,
    role: str,
    selected: Mapping[int, tuple[int, float, float, float]],
    policy: CutLocalSyncConfig,
    errors: list[str],
) -> tuple[dict[int, tuple[float, str]], ...]:
    prefix = f"boundary_{boundary.boundary_index}_{role}_motion_timing"
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        errors.append(f"{prefix}_missing_or_invalid")
        return ()
    parsed: list[dict[int, tuple[float, str]]] = []
    by_grid: dict[int, tuple[float, str]] = {}
    for entry in raw:
        if not isinstance(entry, Mapping) or set(entry) != {
            "grid_index",
            "actual_master_seconds",
            "basis",
        }:
            errors.append(f"{prefix}_record_shape_invalid")
            continue
        grid = entry.get("grid_index")
        basis = entry.get("basis")
        if isinstance(grid, bool) or not isinstance(grid, int) or grid in by_grid:
            errors.append(f"{prefix}_grid_invalid_or_duplicate")
            continue
        if basis not in {
            "actual_pose_motion",
            "formation_upper_body_motion",
            "person_layout_motion",
            "silhouette_body_motion",
        }:
            errors.append(f"{prefix}_basis_invalid")
            continue
        left = selected.get(grid)
        right = selected.get(grid + 1)
        if left is None or right is None:
            errors.append(f"{prefix}_outside_renderer_grid")
            continue
        if left[0] == right[0]:
            errors.append(f"{prefix}_duplicate_renderer_frame_used")
            continue
        if max(left[3], right[3]) > policy.maximum_pts_lookup_residual_frames + 1e-12:
            errors.append(f"{prefix}_pts_residual_out_of_policy")
            continue
        expected_master = (left[2] + right[2]) / 2.0
        if not _numbers_equal(
            entry.get("actual_master_seconds"), expected_master
        ):
            errors.append(f"{prefix}_actual_pts_phase_relation_invalid")
            continue
        by_grid[grid] = (expected_master, str(basis))
    parsed.append(by_grid)
    return tuple(parsed)


def _validate_measured_record(
    record: Mapping[str, Any],
    *,
    boundary: _BoundaryContract,
    plan: Mapping[str, Any],
    clocks: Mapping[int, _SerializedTimeWarp],
    policy: CutLocalSyncConfig,
    pts_counts: Mapping[int, int],
    selected_ordinals_by_source: dict[int, set[int]],
    selected_pts_by_source: dict[int, dict[int, float]],
    errors: list[str],
) -> float | None:
    prefix = f"boundary_{boundary.boundary_index}"
    expected_domain = (
        "face_dissolve_exact_outgoing_handle_and_incoming_overlap"
        if boundary.transition.get("kind") == "face_dissolve"
        else "hard_cut_symmetric_canonical_context"
    )
    if record.get("sampling_domain") != expected_domain:
        errors.append(f"{prefix}_sampling_domain_mismatch")
    selection = record.get("renderer_frame_selection")
    selection_object = dict(selection) if isinstance(selection, Mapping) else {}
    selected_by_role: dict[str, dict[int, tuple[int, float, float, float]]] = {}
    for role, source_index in (
        ("outgoing", boundary.outgoing_source_index),
        ("incoming", boundary.incoming_source_index),
    ):
        clock = clocks.get(source_index)
        if clock is None:
            errors.append(f"{prefix}_{role}_serialized_clock_missing")
            selected_by_role[role] = {}
            continue
        selected_by_role[role] = _validate_renderer_selection_provenance(
            selection_object.get(role),
            boundary=boundary,
            role=role,
            plan=plan,
            clock=clock,
            policy=policy,
            pts_frame_count=pts_counts.get(source_index),
            errors=errors,
        )
        selected_ordinals_by_source.setdefault(source_index, set()).update(
            item[0] for item in selected_by_role[role].values()
        )
        ordinal_pts = selected_pts_by_source.setdefault(source_index, {})
        for ordinal, pts_value, _actual_master, _residual in selected_by_role[
            role
        ].values():
            existing = ordinal_pts.get(ordinal)
            if existing is not None and not _numbers_equal(existing, pts_value):
                errors.append(
                    f"{prefix}_{role}_pts_ordinal_identity_mismatch"
                )
            ordinal_pts[ordinal] = pts_value
    timing = record.get("motion_timing_provenance")
    timing_object = dict(timing) if isinstance(timing, Mapping) else {}
    parsed_timing: dict[str, dict[int, tuple[float, str]]] = {}
    for role in ("outgoing", "incoming"):
        parsed = _parse_motion_timing_provenance(
            timing_object.get(role),
            boundary=boundary,
            role=role,
            selected=selected_by_role.get(role, {}),
            policy=policy,
            errors=errors,
        )
        parsed_timing[role] = parsed[0] if parsed else {}
    left = parsed_timing["outgoing"]
    right = parsed_timing["incoming"]
    if record.get("outgoing_motion_sample_count") != len(left):
        errors.append(f"{prefix}_outgoing_motion_sample_count_mismatch")
    if record.get("incoming_motion_sample_count") != len(right):
        errors.append(f"{prefix}_incoming_motion_sample_count_mismatch")
    raw_candidates = record.get("lag_candidates")
    if not isinstance(raw_candidates, Sequence) or isinstance(
        raw_candidates, (str, bytes, bytearray)
    ):
        errors.append(f"{prefix}_lag_candidates_missing_or_invalid")
        return None
    candidates = list(raw_candidates)
    expected_lags = list(
        range(-policy.maximum_lag_frames, policy.maximum_lag_frames + 1)
    )
    if len(candidates) != len(expected_lags):
        errors.append(f"{prefix}_lag_candidate_count_mismatch")
        return None
    validated_candidates: list[dict[str, Any]] = []
    for expected_lag, raw_candidate in zip(expected_lags, candidates, strict=True):
        if not isinstance(raw_candidate, Mapping):
            errors.append(f"{prefix}_lag_candidate_not_object")
            continue
        candidate = dict(raw_candidate)
        if candidate.get("lag_frames") != expected_lag:
            errors.append(f"{prefix}_lag_candidate_order_or_range_mismatch")
        pairs = [
            (left_sample, right[grid + expected_lag])
            for grid, left_sample in sorted(left.items())
            if grid + expected_lag in right
        ]
        pair_count = len(pairs)
        non_silhouette = sum(
            left_sample[1] != "silhouette_body_motion"
            and right_sample[1] != "silhouette_body_motion"
            for left_sample, right_sample in pairs
        )
        basis_counts = dict(
            sorted(
                Counter(
                    f"{left_sample[1]}+{right_sample[1]}"
                    for left_sample, right_sample in pairs
                ).items()
            )
        )
        actual_phase = (
            float(
                np.median(
                    np.asarray(
                        [
                            (right_sample[0] - left_sample[0])
                            * _finite(plan.get("fps"), "plan.fps")
                            for left_sample, right_sample in pairs
                        ],
                        dtype=np.float64,
                    )
                )
            )
            if pairs
            else None
        )
        if candidate.get("pair_count") != pair_count:
            errors.append(f"{prefix}_lag_{expected_lag}_pair_count_mismatch")
        if candidate.get("non_silhouette_pair_count") != non_silhouette:
            errors.append(
                f"{prefix}_lag_{expected_lag}_non_silhouette_count_mismatch"
            )
        if candidate.get("motion_basis_pair_counts") != basis_counts:
            errors.append(f"{prefix}_lag_{expected_lag}_basis_counts_mismatch")
        if actual_phase is None:
            if candidate.get("actual_pts_phase_frames") is not None:
                errors.append(f"{prefix}_lag_{expected_lag}_actual_phase_invalid")
        elif not _numbers_equal(
            candidate.get("actual_pts_phase_frames"), actual_phase
        ):
            errors.append(f"{prefix}_lag_{expected_lag}_actual_phase_mismatch")
        std_values: list[float | None] = []
        for field in ("outgoing_motion_stddev", "incoming_motion_stddev"):
            value = candidate.get(field)
            if pair_count == 0:
                if value is not None:
                    errors.append(f"{prefix}_lag_{expected_lag}_{field}_invalid")
                std_values.append(None)
            else:
                try:
                    number = _finite(value, field)
                    if number < 0.0:
                        raise ValueError(field)
                    std_values.append(number)
                except ValueError:
                    errors.append(f"{prefix}_lag_{expected_lag}_{field}_invalid")
                    std_values.append(None)
        correlation = candidate.get("correlation")
        if correlation is not None:
            try:
                correlation = _finite(correlation, "correlation")
                if not -1.0 <= correlation <= 1.0:
                    raise ValueError("correlation")
                if (
                    pair_count < policy.minimum_motion_pair_count
                    or any(value is None for value in std_values)
                    or min(value for value in std_values if value is not None)
                    < policy.minimum_motion_standard_deviation
                ):
                    errors.append(
                        f"{prefix}_lag_{expected_lag}_correlation_without_basis"
                    )
            except ValueError:
                errors.append(f"{prefix}_lag_{expected_lag}_correlation_invalid")
                correlation = None
        eligible = bool(
            correlation is not None
            and non_silhouette >= policy.minimum_non_silhouette_pair_count
        )
        if candidate.get("eligible") is not eligible:
            errors.append(f"{prefix}_lag_{expected_lag}_eligibility_mismatch")
        validated_candidates.append(
            {
                **candidate,
                "lag_frames": expected_lag,
                "correlation": correlation,
                "pair_count": pair_count,
                "non_silhouette_pair_count": non_silhouette,
                "motion_basis_pair_counts": basis_counts,
                "actual_pts_phase_frames": actual_phase,
                "eligible": eligible,
            }
        )
    eligible_candidates = [
        item for item in validated_candidates if item.get("eligible") is True
    ]
    if not eligible_candidates:
        errors.append(f"{prefix}_measured_without_eligible_lag")
        return None
    ordered = sorted(
        eligible_candidates,
        key=lambda item: (
            float(item["correlation"]),
            -abs(int(item["lag_frames"])),
        ),
        reverse=True,
    )
    best = ordered[0]
    runner = ordered[1] if len(ordered) > 1 else None
    best_lag = int(best["lag_frames"])
    best_correlation = float(best["correlation"])
    runner_correlation = float(runner["correlation"]) if runner else None
    margin = best_correlation - runner_correlation if runner else 2.0
    if record.get("best_lag_frames") != best_lag:
        errors.append(f"{prefix}_best_lag_mismatch")
    if not _numbers_equal(record.get("best_correlation"), best_correlation):
        errors.append(f"{prefix}_best_correlation_mismatch")
    if runner_correlation is None:
        if record.get("runner_up_correlation") is not None:
            errors.append(f"{prefix}_runner_up_correlation_mismatch")
    elif not _numbers_equal(record.get("runner_up_correlation"), runner_correlation):
        errors.append(f"{prefix}_runner_up_correlation_mismatch")
    if not _numbers_equal(record.get("correlation_margin"), margin):
        errors.append(f"{prefix}_correlation_margin_mismatch")
    if (
        best_correlation < policy.minimum_correlation
        or margin < policy.minimum_correlation_margin
        or abs(best_lag) >= policy.maximum_lag_frames
    ):
        errors.append(f"{prefix}_measured_lag_fails_policy")
    by_lag = {
        int(item["lag_frames"]): item
        for item in eligible_candidates
    }
    subframe = 0.0
    if best_lag - 1 in by_lag and best_lag + 1 in by_lag:
        left_score = float(by_lag[best_lag - 1]["correlation"])
        right_score = float(by_lag[best_lag + 1]["correlation"])
        denominator = left_score - 2.0 * best_correlation + right_score
        if denominator < -1e-12:
            subframe = float(
                np.clip(
                    0.5 * (left_score - right_score) / denominator,
                    -0.5,
                    0.5,
                )
            )
    if not _numbers_equal(record.get("subframe_peak_offset_frames"), subframe):
        errors.append(f"{prefix}_subframe_peak_mismatch")
    actual_phase = best.get("actual_pts_phase_frames")
    if actual_phase is None:
        errors.append(f"{prefix}_best_actual_pts_phase_missing")
        return None
    if not _numbers_equal(record.get("actual_pts_phase_frames"), actual_phase):
        errors.append(f"{prefix}_actual_pts_phase_mismatch")
    signed = float(actual_phase) + subframe
    residual = abs(signed)
    if not _numbers_equal(record.get("signed_phase_residual_frames"), signed):
        errors.append(f"{prefix}_signed_phase_residual_mismatch")
    if not _numbers_equal(record.get("residual_frames"), residual):
        errors.append(f"{prefix}_residual_relation_mismatch")
    if record.get("matched_motion_pair_count") != best["pair_count"]:
        errors.append(f"{prefix}_matched_pair_count_mismatch")
    if (
        record.get("matched_non_silhouette_pair_count")
        != best["non_silhouette_pair_count"]
    ):
        errors.append(f"{prefix}_matched_non_silhouette_count_mismatch")
    if record.get("motion_basis_pair_counts") != best["motion_basis_pair_counts"]:
        errors.append(f"{prefix}_matched_basis_counts_mismatch")
    if record.get("reason") is not None:
        errors.append(f"{prefix}_measured_reason_must_be_null")
    return residual


def validate_cut_local_sync_evidence(
    evidence: Mapping[str, Any] | None,
    *,
    plan: Mapping[str, Any],
    timewarp_payloads: Sequence[Mapping[str, Any]],
    maximum_residual_p95_frames: float = 0.5,
) -> dict[str, Any]:
    """Independently verify serialized scalar evidence against final artifacts."""

    errors: list[str] = []
    if evidence is None:
        return {
            "available": False,
            "hash_bindings_valid": False,
            "errors": ["cut_local_sync_evidence_missing"],
            "unknown_reason_counts": {},
            "passed": False,
        }
    payload = dict(evidence)
    try:
        boundaries = _boundary_contracts(plan)
        used_sources = _used_source_indices(plan)
        bound_timewarps = _bound_timewarp_payloads(
            timewarp_payloads, used_sources
        )
        plan_fps = _finite(plan.get("fps"), "plan.fps")
        threshold = _finite(
            maximum_residual_p95_frames, "maximum_residual_p95_frames"
        )
        if threshold <= 0.0:
            raise ValueError("threshold invalid")
        trusted_policy = CutLocalSyncConfig(
            maximum_residual_p95_frames=threshold
        )
    except (TypeError, ValueError):
        return {
            "available": True,
            "hash_bindings_valid": False,
            "errors": ["artifact_contract_invalid"],
            "unknown_reason_counts": {},
            "passed": False,
        }
    proof_hash = payload.get("proof_sha256")
    unsigned_payload = dict(payload)
    unsigned_payload.pop("proof_sha256", None)
    proof_hash_valid = bool(
        isinstance(proof_hash, str)
        and len(proof_hash) == 64
        and proof_hash == _canonical_sha256(unsigned_payload)
    )
    if not proof_hash_valid:
        errors.append("proof_sha256_mismatch")
    expected_hashes = {
        "plan_sha256": _canonical_sha256(dict(plan)),
        "timewarp_contract_sha256": _canonical_sha256(bound_timewarps),
        "boundary_clock_contract_sha256": _canonical_sha256(
            [item.payload() for item in boundaries]
        ),
    }
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported_schema_version")
    if payload.get("analysis") != ANALYSIS_BASIS:
        errors.append("unsupported_analysis_basis")
    if payload.get("activation") != ACTIVATION or payload.get("required") is not True:
        errors.append("activation_contract_invalid")
    if not _numbers_equal(payload.get("fps"), plan_fps):
        errors.append("fps_mismatch")
    if payload.get("policy") != trusted_policy.to_dict():
        errors.append("trusted_policy_mismatch")
    if payload.get("serialization_policy") != _SERIALIZATION_POLICY:
        errors.append("serialization_policy_mismatch")
    if payload.get("used_source_indices") != list(used_sources):
        errors.append("used_source_indices_mismatch")
    if not _numbers_equal(
        payload.get("maximum_allowed_residual_p95_frames"), threshold
    ):
        errors.append("maximum_allowed_residual_p95_frames_mismatch")
    for name, expected in expected_hashes.items():
        if payload.get(name) != expected:
            errors.append(f"{name}_mismatch")

    clocks: dict[int, _SerializedTimeWarp] = {}
    for source_index, timewarp in zip(
        used_sources, bound_timewarps, strict=True
    ):
        try:
            clocks[source_index] = _strict_serialized_timewarp(
                timewarp, expected_source_index=source_index
            )
        except (TypeError, ValueError) as exc:
            code = str(exc)
            if not code.replace("_", "").isalnum():
                code = "invalid"
            errors.append(f"source_{source_index}_timewarp_{code}")
    for boundary in boundaries:
        outgoing_clock = clocks.get(boundary.outgoing_source_index)
        incoming_clock = clocks.get(boundary.incoming_source_index)
        if outgoing_clock is not None and not _numbers_equal(
            outgoing_clock.master_to_source(boundary.canonical_master_seconds),
            boundary.outgoing_source_local_seconds,
            tolerance=1e-6,
        ):
            errors.append(
                f"boundary_{boundary.boundary_index}_outgoing_timewarp_endpoint_mismatch"
            )
        if incoming_clock is not None and not _numbers_equal(
            incoming_clock.master_to_source(boundary.canonical_master_seconds),
            boundary.incoming_source_local_seconds,
            tolerance=1e-6,
        ):
            errors.append(
                f"boundary_{boundary.boundary_index}_incoming_timewarp_endpoint_mismatch"
            )

    raw_audit = payload.get("source_decode_audit")
    audits = (
        list(raw_audit)
        if isinstance(raw_audit, Sequence)
        and not isinstance(raw_audit, (str, bytes, bytearray))
        else []
    )
    audit_by_source: dict[int, Mapping[str, Any]] = {}
    pts_counts: dict[int, int] = {}
    plan_sources = _as_sequence(plan.get("sources"), "plan.sources")
    for raw_audit_item in audits:
        if not isinstance(raw_audit_item, Mapping):
            errors.append("source_decode_audit_record_not_object")
            continue
        audit = dict(raw_audit_item)
        source_index = audit.get("source_index")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index not in used_sources
        ):
            errors.append("source_decode_audit_source_index_invalid")
            continue
        if source_index in audit_by_source:
            errors.append("source_decode_audit_source_index_duplicate")
            continue
        audit_by_source[source_index] = audit
        if audit.get("source_name") != Path(str(plan_sources[source_index])).name:
            errors.append(f"source_{source_index}_audit_source_name_mismatch")
        if audit.get("pts_basis") != PTS_BASIS:
            errors.append(f"source_{source_index}_audit_pts_basis_mismatch")
        if audit.get("renderer_selection_basis") != RENDERER_SELECTION_BASIS:
            errors.append(
                f"source_{source_index}_audit_renderer_selection_basis_mismatch"
            )
        if audit.get("status") == "decoded":
            expected_decoded_keys = {
                "source_index",
                "source_name",
                "status",
                "pts_basis",
                "renderer_selection_basis",
                "pts_frame_count",
                "decoded_declared_frame_count",
                "pts_decode_ordinal_count_match",
                "pts_origin_seconds",
                "pts_sha256",
                "median_pts_step_seconds",
                "requested_ordinal_count",
                "selected_ordinal_pts_sha256",
                "decoded_requested_ordinal_count",
                "decoded_frame_count",
                "described_frame_count",
                "descriptor_failure_count",
                "maximum_pts_lookup_residual_frames",
                "raw_frames_persisted",
                "motion_descriptors_persisted",
            }
            if set(audit) != expected_decoded_keys:
                errors.append(f"source_{source_index}_audit_shape_mismatch")
            count = audit.get("pts_frame_count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 2:
                errors.append(f"source_{source_index}_audit_pts_count_invalid")
            else:
                pts_counts[source_index] = count
            if audit.get("decoded_declared_frame_count") != count:
                errors.append(f"source_{source_index}_audit_declared_count_mismatch")
            if audit.get("pts_decode_ordinal_count_match") is not True:
                errors.append(f"source_{source_index}_audit_ordinal_count_unbound")
            sha = audit.get("pts_sha256")
            if not isinstance(sha, str) or len(sha) != 64 or any(
                character not in "0123456789abcdef" for character in sha
            ):
                errors.append(f"source_{source_index}_audit_pts_sha256_invalid")
            try:
                _finite(audit.get("pts_origin_seconds"), "pts_origin_seconds")
            except ValueError:
                errors.append(f"source_{source_index}_audit_pts_origin_seconds_invalid")
            try:
                if _finite(
                    audit.get("median_pts_step_seconds"),
                    "median_pts_step_seconds",
                ) <= 0.0:
                    raise ValueError("median step")
            except ValueError:
                errors.append(
                    f"source_{source_index}_audit_median_pts_step_seconds_invalid"
                )
            requested_count = audit.get("requested_ordinal_count")
            if (
                isinstance(requested_count, bool)
                or not isinstance(requested_count, int)
                or requested_count < 0
            ):
                errors.append(f"source_{source_index}_audit_requested_count_invalid")
            selected_sha = audit.get("selected_ordinal_pts_sha256")
            if not isinstance(selected_sha, str) or len(selected_sha) != 64 or any(
                character not in "0123456789abcdef" for character in selected_sha
            ):
                errors.append(
                    f"source_{source_index}_audit_selected_ordinal_pts_sha256_invalid"
                )
            for field in (
                "decoded_requested_ordinal_count",
                "decoded_frame_count",
                "described_frame_count",
            ):
                if audit.get(field) != requested_count:
                    errors.append(f"source_{source_index}_audit_{field}_mismatch")
            if audit.get("descriptor_failure_count") != 0:
                errors.append(f"source_{source_index}_audit_descriptor_failure")
            try:
                if _finite(
                    audit.get("maximum_pts_lookup_residual_frames"),
                    "maximum_pts_lookup_residual_frames",
                ) < 0.0:
                    raise ValueError("maximum residual")
            except ValueError:
                errors.append(f"source_{source_index}_audit_maximum_residual_invalid")
            if (
                audit.get("raw_frames_persisted") is not False
                or audit.get("motion_descriptors_persisted") is not False
            ):
                errors.append(f"source_{source_index}_audit_persistence_policy_invalid")
        elif audit.get("status") == "unavailable":
            expected_unavailable_keys = {
                "source_index",
                "source_name",
                "status",
                "reason",
                "pts_basis",
                "renderer_selection_basis",
            }
            if set(audit) != expected_unavailable_keys:
                errors.append(f"source_{source_index}_audit_shape_mismatch")
            reason = audit.get("reason")
            if (
                not isinstance(reason, str)
                or not reason
                or "/" in reason
                or "\\" in reason
                or ":" in reason
                or not reason.replace("_", "").isalnum()
                or reason.lower() != reason
            ):
                errors.append(f"source_{source_index}_audit_reason_not_sanitized")
        else:
            errors.append(f"source_{source_index}_audit_status_invalid")
    if set(audit_by_source) != set(used_sources):
        errors.append("source_decode_audit_used_source_set_mismatch")

    raw_records = payload.get("per_boundary")
    records = (
        list(raw_records)
        if isinstance(raw_records, Sequence)
        and not isinstance(raw_records, (str, bytes, bytearray))
        else []
    )
    by_index: dict[int, Mapping[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            errors.append("boundary_record_not_object")
            continue
        index = raw.get("boundary_index")
        if isinstance(index, bool) or not isinstance(index, int):
            errors.append("boundary_index_invalid")
            continue
        if index in by_index:
            errors.append("duplicate_boundary_index")
        by_index[index] = raw
    expected_indices = {item.boundary_index for item in boundaries}
    if set(by_index) != expected_indices:
        errors.append("retained_boundary_set_mismatch")

    residuals: list[float] = []
    unknown_reasons: Counter[str] = Counter()
    selected_ordinals_by_source: dict[int, set[int]] = {
        source_index: set() for source_index in used_sources
    }
    selected_pts_by_source: dict[int, dict[int, float]] = {
        source_index: {} for source_index in used_sources
    }
    for boundary in boundaries:
        record = by_index.get(boundary.boundary_index)
        if record is None:
            continue
        for field, expected in boundary.payload().items():
            if record.get(field) != expected:
                errors.append(
                    f"boundary_{boundary.boundary_index}_{field}_mismatch"
                )
        if record.get("boundary_contract_sha256") != boundary.sha256:
            errors.append(
                f"boundary_{boundary.boundary_index}_contract_sha256_mismatch"
            )
        if record.get("status") != "measured":
            expected_domain = (
                "face_dissolve_exact_outgoing_handle_and_incoming_overlap"
                if boundary.transition.get("kind") == "face_dissolve"
                else "hard_cut_symmetric_canonical_context"
            )
            raw_selection = record.get("renderer_frame_selection")
            if isinstance(raw_selection, Mapping) and record.get(
                "sampling_domain"
            ) != expected_domain:
                errors.append(
                    f"boundary_{boundary.boundary_index}_sampling_domain_mismatch"
                )
            selection_object = (
                dict(raw_selection) if isinstance(raw_selection, Mapping) else {}
            )
            for role, source_index in (
                ("outgoing", boundary.outgoing_source_index),
                ("incoming", boundary.incoming_source_index),
            ):
                raw_role_selection = selection_object.get(role)
                clock = clocks.get(source_index)
                if not isinstance(raw_role_selection, Mapping) or clock is None:
                    continue
                parsed_selection = _validate_renderer_selection_provenance(
                    raw_role_selection,
                    boundary=boundary,
                    role=role,
                    plan=plan,
                    clock=clock,
                    policy=trusted_policy,
                    pts_frame_count=pts_counts.get(source_index),
                    errors=errors,
                )
                selected_ordinals_by_source[source_index].update(
                    item[0] for item in parsed_selection.values()
                )
                ordinal_pts = selected_pts_by_source[source_index]
                for ordinal, pts_value, _master, _residual in (
                    parsed_selection.values()
                ):
                    existing = ordinal_pts.get(ordinal)
                    if existing is not None and not _numbers_equal(
                        existing, pts_value
                    ):
                        errors.append(
                            f"boundary_{boundary.boundary_index}_{role}_pts_ordinal_identity_mismatch"
                        )
                    ordinal_pts[ordinal] = pts_value
            reason = record.get("reason")
            if not isinstance(reason, str) or not reason:
                errors.append(
                    f"boundary_{boundary.boundary_index}_unknown_reason_invalid"
                )
                reason = "unknown"
            unknown_reasons[str(reason)] += 1
            continue
        residual = _validate_measured_record(
            record,
            boundary=boundary,
            plan=plan,
            clocks=clocks,
            policy=trusted_policy,
            pts_counts=pts_counts,
            selected_ordinals_by_source=selected_ordinals_by_source,
            selected_pts_by_source=selected_pts_by_source,
            errors=errors,
        )
        if residual is not None:
            residuals.append(residual)

    for source_index, audit in audit_by_source.items():
        if audit.get("status") != "decoded":
            continue
        expected_requested_count = len(selected_ordinals_by_source[source_index])
        if audit.get("requested_ordinal_count") != expected_requested_count:
            errors.append(
                f"source_{source_index}_audit_requested_ordinal_union_mismatch"
            )
        ordinal_pts = sorted(selected_pts_by_source[source_index].items())
        if any(
            right[1] <= left[1]
            for left, right in zip(ordinal_pts, ordinal_pts[1:])
        ):
            errors.append(f"source_{source_index}_selected_pts_not_strict_by_ordinal")
        expected_selected_sha = _canonical_sha256(
            [
                {
                    "frame_ordinal": ordinal,
                    "normalized_pts_seconds": pts_value,
                }
                for ordinal, pts_value in ordinal_pts
            ]
        )
        if audit.get("selected_ordinal_pts_sha256") != expected_selected_sha:
            errors.append(
                f"source_{source_index}_audit_selected_ordinal_pts_sha256_mismatch"
            )

    recomputed_p95 = _quantile95(residuals)
    serialized_p95 = payload.get("residual_p95_frames")
    if recomputed_p95 is None:
        if serialized_p95 is not None:
            errors.append("residual_p95_without_measurements")
    else:
        try:
            if not math.isclose(
                recomputed_p95,
                _finite(serialized_p95, "residual_p95_frames"),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                errors.append("residual_p95_mismatch")
        except ValueError:
            errors.append("residual_p95_invalid")
    if payload.get("retained_cross_source_boundary_count") != len(boundaries):
        errors.append("boundary_count_mismatch")
    if payload.get("measured_boundary_count") != len(residuals):
        errors.append("measured_boundary_count_mismatch")
    if payload.get("unknown_boundary_count") != sum(unknown_reasons.values()):
        errors.append("unknown_boundary_count_mismatch")
    serialized_reasons = payload.get("unknown_reason_counts")
    if serialized_reasons != dict(sorted(unknown_reasons.items())):
        errors.append("unknown_reason_counts_mismatch")
    threshold_exceeding_count = sum(
        residual > threshold + 1e-12 for residual in residuals
    )
    if payload.get("threshold_exceeding_boundary_count") != threshold_exceeding_count:
        errors.append("threshold_exceeding_boundary_count_mismatch")
    within_threshold = bool(
        recomputed_p95 is not None and recomputed_p95 <= threshold + 1e-12
    )
    computed_pass = bool(
        boundaries
        and len(residuals) == len(boundaries)
        and not unknown_reasons
        and all(
            audit_by_source.get(source_index, {}).get("status") == "decoded"
            for source_index in used_sources
        )
        and within_threshold
        and not errors
    )
    if payload.get("passed") is not computed_pass:
        errors.append("serialized_passed_mismatch")
        computed_pass = False
    return {
        "available": True,
        "analysis": payload.get("analysis"),
        "proof_sha256_valid": proof_hash_valid,
        "hash_bindings_valid": proof_hash_valid and not any(
            "sha256" in error
            or "contract" in error
            or "boundary_set" in error
            or "timewarp_endpoint" in error
            for error in errors
        ),
        "used_source_indices": list(used_sources),
        "retained_cross_source_boundary_count": len(boundaries),
        "measured_boundary_count": len(residuals),
        "unknown_boundary_count": sum(unknown_reasons.values()),
        "unknown_reason_counts": dict(sorted(unknown_reasons.items())),
        "residual_p95_frames": recomputed_p95,
        "maximum_allowed_residual_p95_frames": threshold,
        "within_threshold": within_threshold,
        "errors": errors,
        "passed": computed_pass and not errors,
    }


__all__ = [
    "ACTIVATION",
    "ANALYSIS_BASIS",
    "CutLocalMotionSample",
    "CutLocalSyncConfig",
    "analyze_cut_local_sync",
    "estimate_motion_phase",
    "renderer_selected_frame_ordinals",
    "registered_cut_local_sync_required",
    "resolve_cut_local_sync_activation",
    "validate_cut_local_sync_evidence",
]
