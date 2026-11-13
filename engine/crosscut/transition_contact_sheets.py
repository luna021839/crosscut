"""Final-output contact sheets for every real source switch in a render plan.

This module is intentionally independent from the render pipeline.  It reads a
finished render-plan JSON and the MP4 named by that plan, samples only the final
rendered pixels around source-changing segment boundaries, and writes labeled
PNG pages.  Source videos, raw source frames, visual features, and embeddings
are never opened or persisted here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

# Match the main CLI's quiet decoder policy; actual read failures are converted
# into explicit exceptions below.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import numpy as np


SCHEMA_VERSION = 1
ANALYSIS_BASIS = "final_output_source_switch_contact_sheets_v1"


@dataclass(frozen=True, slots=True)
class ContactSheetConfig:
    """Layout and temporal sampling policy for final-output inspection."""

    frame_offsets: tuple[int, ...] = (-2, -1, 0, 1)
    boundaries_per_page: int = 6
    tile_width: int = 320
    tile_height: int = 180
    page_margin: int = 24
    page_header_height: int = 54
    boundary_header_height: int = 54
    tile_label_height: int = 28
    row_gap: int = 18
    column_gap: int = 8
    # OpenCV commonly reports 29.970029 for a 29.97 CFR stream.  One
    # thousandth tolerates that representation noise while still rejecting
    # distinct common cadences such as 29.97 versus 30.00.
    fps_tolerance: float = 0.001

    def __post_init__(self) -> None:
        if not self.frame_offsets:
            raise ValueError("frame_offsets cannot be empty")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.frame_offsets
        ):
            raise ValueError("frame_offsets must contain integers")
        if tuple(sorted(set(self.frame_offsets))) != self.frame_offsets:
            raise ValueError("frame_offsets must be strictly increasing and unique")
        if not any(value < 0 for value in self.frame_offsets):
            raise ValueError("frame_offsets need at least one outgoing sample")
        if 0 not in self.frame_offsets:
            raise ValueError("frame_offsets must include the incoming boundary frame (0)")
        for name in (
            "boundaries_per_page",
            "tile_width",
            "tile_height",
            "page_margin",
            "page_header_height",
            "boundary_header_height",
            "tile_label_height",
            "row_gap",
            "column_gap",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.fps_tolerance) or self.fps_tolerance < 0.0:
            raise ValueError("fps_tolerance must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    fps: float
    width: int
    height: int
    frame_count: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("decoded video FPS must be positive and finite")
        for name in ("width", "height", "frame_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"decoded video {name} must be a positive integer")


class FrameDecoder(Protocol):
    """Minimal injectable decoder used by the focused synthetic tests."""

    @property
    def metadata(self) -> VideoMetadata: ...

    def iter_frames(
        self,
        frame_indices: Sequence[int],
    ) -> Iterator[tuple[int, np.ndarray]]: ...

    def close(self) -> None: ...


def _validate_decoded_frame(
    frame: np.ndarray,
    metadata: VideoMetadata,
    frame_index: int,
) -> None:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise RuntimeError(
            f"decoded frame {frame_index} is not a uint8 NumPy image"
        )
    expected_shape = (metadata.height, metadata.width, 3)
    if frame.shape != expected_shape:
        raise RuntimeError(
            f"decoded frame {frame_index} dimensions/channels {frame.shape} "
            f"do not match final video {expected_shape}"
        )


class OpenCVFrameDecoder:
    """Sequential full decoder that never uses imprecise random seeking."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._capture = cv2.VideoCapture(str(self._path))
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"cannot open final rendered video: {self._path}")
        frame_count_value = self._capture.get(cv2.CAP_PROP_FRAME_COUNT)
        width_value = self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height_value = self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps_value = self._capture.get(cv2.CAP_PROP_FPS)
        values = (frame_count_value, width_value, height_value, fps_value)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            self.close()
            raise RuntimeError("final rendered video has incomplete decoder metadata")
        self._metadata = VideoMetadata(
            fps=float(fps_value),
            width=int(round(width_value)),
            height=int(round(height_value)),
            frame_count=int(round(frame_count_value)),
        )
        self._consumed = False

    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata

    def iter_frames(
        self,
        frame_indices: Sequence[int],
    ) -> Iterator[tuple[int, np.ndarray]]:
        if self._consumed:
            raise RuntimeError("frame decoder can only be consumed once")
        self._consumed = True
        requested = tuple(frame_indices)
        if not requested:
            return
        if tuple(sorted(set(requested))) != requested or requested[0] < 0:
            raise ValueError("frame indices must be sorted, unique, and non-negative")
        if requested[-1] >= self.metadata.frame_count:
            raise ValueError("requested frame index exceeds decoder metadata")
        target_position = 0
        for frame_index in range(self.metadata.frame_count):
            decoded, frame = self._capture.read()
            if not decoded or frame is None:
                raise RuntimeError(
                    "final rendered video became undecodable at exact frame "
                    f"{frame_index}"
                )
            _validate_decoded_frame(frame, self.metadata, frame_index)
            if (
                target_position < len(requested)
                and frame_index == requested[target_position]
            ):
                yield (frame_index, frame)
                target_position += 1
        if target_position != len(requested):
            raise RuntimeError("final rendered video did not yield every requested frame")
        decoded_extra, extra_frame = self._capture.read()
        if decoded_extra or extra_frame is not None:
            raise RuntimeError(
                "final rendered video did not produce an exact clean EOF after "
                "its declared frame count"
            )

    def close(self) -> None:
        capture = getattr(self, "_capture", None)
        if capture is not None:
            capture.release()


@dataclass(frozen=True, slots=True)
class SourceSwitchBoundary:
    switch_ordinal: int
    plan_boundary_index: int
    outgoing_segment_index: int
    incoming_segment_index: int
    outgoing_source_index: int
    incoming_source_index: int
    outgoing_source_label: str
    incoming_source_label: str
    plan_boundary_seconds: float
    boundary_frame_index: int
    output_boundary_seconds: float
    transition_kind: str
    transition_frame_count: int

    def sample_side(self, offset: int) -> str:
        if offset < 0:
            return "outgoing"
        if (
            self.transition_kind == "face_dissolve"
            and offset < self.transition_frame_count
        ):
            return "transition"
        return "incoming"

    def sheet_sample_label(self, offset: int) -> str:
        side = self.sample_side(offset)
        if side == "outgoing":
            return "OUT/PRE"
        if side == "transition":
            return f"XFADE {offset + 1}/{self.transition_frame_count}"
        return "CUT/IN" if offset == 0 else "IN/POST"

    def to_dict(self, offsets: Sequence[int]) -> dict[str, Any]:
        return {
            "switch_ordinal": self.switch_ordinal,
            "plan_boundary_index": self.plan_boundary_index,
            "outgoing_segment_index": self.outgoing_segment_index,
            "incoming_segment_index": self.incoming_segment_index,
            "outgoing_source": {
                "source_index": self.outgoing_source_index,
                "label": self.outgoing_source_label,
            },
            "incoming_source": {
                "source_index": self.incoming_source_index,
                "label": self.incoming_source_label,
            },
            "plan_boundary_seconds": self.plan_boundary_seconds,
            "boundary_frame_index": self.boundary_frame_index,
            "output_boundary_seconds": self.output_boundary_seconds,
            "transition": {
                "kind": self.transition_kind,
                "frame_count": self.transition_frame_count,
            },
            "samples": [
                {
                    "relative_frame_offset": offset,
                    "frame_index": self.boundary_frame_index + offset,
                    "side": self.sample_side(offset),
                }
                for offset in offsets
            ],
        }


@dataclass(frozen=True, slots=True)
class TransitionContactSheetResult:
    render_plan_path: Path
    video_path: Path
    output_directory: Path
    page_paths: tuple[Path, ...]
    manifest_path: Path | None
    source_switch_count: int
    sampled_frame_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "render_plan_path": str(self.render_plan_path.resolve()),
            "video_path": str(self.video_path.resolve()),
            "output_directory": str(self.output_directory.resolve()),
            "page_paths": [str(path.resolve()) for path in self.page_paths],
            "manifest_path": (
                str(self.manifest_path.resolve())
                if self.manifest_path is not None
                else None
            ),
            "source_switch_count": self.source_switch_count,
            "sampled_frame_count": self.sampled_frame_count,
        }


DecoderFactory = Callable[[Path], FrameDecoder]


def _require_input_file(path: Path, description: str) -> Path:
    candidate = Path(path)
    try:
        valid = candidate.is_file() and candidate.stat().st_size > 0
    except OSError as exc:
        raise ValueError(f"cannot inspect {description}: {candidate}") from exc
    if not valid:
        raise ValueError(f"{description} is missing or empty: {candidate}")
    return candidate.resolve()


def _load_json_object(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"render plan is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("render plan JSON root must be an object")
    return (payload, encoded)


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    try:
        metadata = path.stat()
    except OSError as exc:
        raise RuntimeError(f"input artifact became unavailable: {path}") from exc
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _require_inputs_unchanged(
    *,
    plan: Path,
    video: Path,
    plan_identity: tuple[int, int, int, int],
    video_identity: tuple[int, int, int, int],
    plan_encoded: bytes,
) -> None:
    if _file_identity(plan) != plan_identity or _file_identity(video) != video_identity:
        raise RuntimeError("an input artifact changed during contact-sheet generation")
    try:
        current_plan = plan.read_bytes()
    except OSError as exc:
        raise RuntimeError("render plan became unreadable during generation") from exc
    if current_plan != plan_encoded:
        raise RuntimeError("an input artifact changed during contact-sheet generation")


def _finite_positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return numeric


def _source_label(value: str, source_index: int) -> str:
    name = Path(value).name.strip()
    return name or f"source-{source_index}"


def _ascii_sheet_label(value: str, fallback: str, *, maximum: int = 38) -> str:
    candidate = Path(value)
    visible_stem = (
        candidate.stem.encode("ascii", errors="ignore").decode("ascii").strip()
    )
    if any(character.isalnum() for character in visible_stem):
        visible = visible_stem + candidate.suffix.encode(
            "ascii", errors="ignore"
        ).decode("ascii")
    else:
        visible = fallback
    visible = " ".join(visible.split())
    if len(visible) > maximum:
        return visible[: maximum - 3] + "..."
    return visible


def _resolve_declared_output(plan_path: Path, declared_output: str) -> Path:
    output = Path(declared_output)
    if not output.is_absolute():
        output = plan_path.parent / output
    return output.resolve()


def _parse_switch_boundaries(
    payload: Mapping[str, Any],
    *,
    render_plan_path: Path,
    video_path: Path,
) -> tuple[
    float,
    int,
    int,
    int,
    tuple[SourceSwitchBoundary, ...],
    tuple[str, ...],
]:
    fps = _finite_positive_number(payload.get("fps"), "render plan fps")
    width = payload.get("width")
    height = payload.get("height")
    for value, name in ((width, "width"), (height, "height")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"render plan {name} must be a positive integer")
    declared_output = payload.get("output")
    if not isinstance(declared_output, str) or not declared_output.strip():
        raise ValueError("render plan output must be a non-empty path string")
    if _resolve_declared_output(render_plan_path, declared_output) != video_path.resolve():
        raise ValueError("final MP4 does not match the exact output path in the render plan")

    sources_value = payload.get("sources")
    if not isinstance(sources_value, list) or not sources_value:
        raise ValueError("render plan sources must be a non-empty array")
    if any(not isinstance(value, str) or not value.strip() for value in sources_value):
        raise ValueError("render plan source paths must be non-empty strings")
    source_labels = tuple(
        _source_label(value, index) for index, value in enumerate(sources_value)
    )

    segments_value = payload.get("segments")
    if not isinstance(segments_value, list) or not segments_value:
        raise ValueError("render plan segments must be a non-empty array")
    transitions_value = payload.get("transitions")
    if (
        not isinstance(transitions_value, list)
        or len(transitions_value) != len(segments_value) - 1
    ):
        raise ValueError("render plan needs one transition per segment boundary")
    transitions = transitions_value

    parsed: list[tuple[int, float, float]] = []
    cumulative = 0.0
    previous_frame_boundary = 0
    for segment_index, segment in enumerate(segments_value):
        if not isinstance(segment, dict):
            raise ValueError(f"render plan segment {segment_index} must be an object")
        source_index = segment.get("source_index")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or not 0 <= source_index < len(source_labels)
        ):
            raise ValueError(f"render plan segment {segment_index} has an invalid source_index")
        duration = _finite_positive_number(
            segment.get("duration"),
            f"render plan segment {segment_index} duration",
        )
        master_start = segment.get("master_start")
        if isinstance(master_start, bool) or not isinstance(master_start, (int, float)):
            raise ValueError(f"render plan segment {segment_index} master_start must be numeric")
        master_start_number = float(master_start)
        if not math.isfinite(master_start_number) or master_start_number < 0.0:
            raise ValueError(f"render plan segment {segment_index} master_start is invalid")
        if abs(master_start_number - cumulative) > 1e-6:
            raise ValueError("render plan segments are not contiguous on the master timeline")
        cumulative += duration
        frame_boundary = int(round(cumulative * fps))
        if frame_boundary - previous_frame_boundary < 2:
            raise ValueError("render plan contains a segment shorter than two output frames")
        previous_frame_boundary = frame_boundary
        parsed.append((source_index, duration, cumulative))

    expected_frame_count = int(round(cumulative * fps))
    declared_frame_count = payload.get("output_frame_count")
    if (
        isinstance(declared_frame_count, bool)
        or not isinstance(declared_frame_count, int)
        or declared_frame_count != expected_frame_count
    ):
        raise ValueError("render plan output_frame_count does not match its timeline")

    parsed_transitions: list[tuple[str, int]] = []
    for boundary_index, transition in enumerate(transitions):
        if not isinstance(transition, dict):
            raise ValueError(f"render plan transition {boundary_index} must be an object")
        kind = transition.get("kind")
        frame_count = transition.get("frame_count")
        if kind not in {"hard_cut", "face_dissolve"}:
            raise ValueError(f"render plan transition {boundary_index} kind is invalid")
        if isinstance(frame_count, bool) or not isinstance(frame_count, int):
            raise ValueError(f"render plan transition {boundary_index} frame_count is invalid")
        if kind == "hard_cut" and frame_count != 0:
            raise ValueError("hard-cut transition must have zero overlap frames")
        if kind == "face_dissolve" and frame_count < 3:
            raise ValueError("face-dissolve transition needs at least three frames")
        if (
            kind == "face_dissolve"
            and parsed[boundary_index][0] == parsed[boundary_index + 1][0]
        ):
            raise ValueError("face-dissolve transition requires a real source change")
        if kind == "face_dissolve":
            incoming_start_frame = int(round(parsed[boundary_index][2] * fps))
            incoming_end_frame = int(round(parsed[boundary_index + 1][2] * fps))
            if frame_count > incoming_end_frame - incoming_start_frame:
                raise ValueError(
                    "face-dissolve transition cannot exceed the incoming segment"
                )
        parsed_transitions.append((kind, frame_count))

    boundaries: list[SourceSwitchBoundary] = []
    for boundary_index in range(len(parsed) - 1):
        outgoing_source = parsed[boundary_index][0]
        incoming_source = parsed[boundary_index + 1][0]
        if outgoing_source == incoming_source:
            continue
        kind, frame_count = parsed_transitions[boundary_index]
        plan_seconds = parsed[boundary_index][2]
        boundary_frame = int(round(plan_seconds * fps))
        boundaries.append(
            SourceSwitchBoundary(
                switch_ordinal=len(boundaries) + 1,
                plan_boundary_index=boundary_index,
                outgoing_segment_index=boundary_index,
                incoming_segment_index=boundary_index + 1,
                outgoing_source_index=outgoing_source,
                incoming_source_index=incoming_source,
                outgoing_source_label=source_labels[outgoing_source],
                incoming_source_label=source_labels[incoming_source],
                plan_boundary_seconds=plan_seconds,
                boundary_frame_index=boundary_frame,
                output_boundary_seconds=boundary_frame / fps,
                transition_kind=kind,
                transition_frame_count=frame_count,
            )
        )
    if not boundaries:
        raise ValueError("render plan contains no source-switch boundaries")
    return (
        fps,
        width,
        height,
        expected_frame_count,
        tuple(boundaries),
        source_labels,
    )


def _letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise RuntimeError("decoded frame must be a uint8 NumPy array")
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise RuntimeError("decoded frame must be a non-empty three-channel BGR image")
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, min(width, int(round(source_width * scale))))
    resized_height = max(1, min(height, int(round(source_height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.full((height, width, 3), 12, dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float,
    color: tuple[int, int, int] = (232, 232, 232),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _render_page(
    *,
    page_boundaries: Sequence[SourceSwitchBoundary],
    thumbnails: Mapping[int, np.ndarray],
    offsets: Sequence[int],
    page_number: int,
    page_count: int,
    fps: float,
    video_name: str,
    config: ContactSheetConfig,
) -> np.ndarray:
    columns = len(offsets)
    page_width = (
        2 * config.page_margin
        + columns * config.tile_width
        + (columns - 1) * config.column_gap
    )
    row_height = config.boundary_header_height + config.tile_label_height + config.tile_height
    page_height = (
        config.page_header_height
        + 2 * config.page_margin
        + len(page_boundaries) * row_height
        + max(0, len(page_boundaries) - 1) * config.row_gap
    )
    image = np.full((page_height, page_width, 3), 22, dtype=np.uint8)
    safe_video_name = _ascii_sheet_label(video_name, "final-render")
    title = (
        f"FINAL OUTPUT SOURCE SWITCHES | {safe_video_name} | "
        f"PAGE {page_number:03d}/{page_count:03d}"
    )
    _put_text(image, title, (config.page_margin, config.page_margin + 22), scale=0.62, thickness=2)
    y = config.page_margin + config.page_header_height
    for boundary in page_boundaries:
        outgoing = _ascii_sheet_label(
            boundary.outgoing_source_label,
            f"source-{boundary.outgoing_source_index}",
        )
        incoming = _ascii_sheet_label(
            boundary.incoming_source_label,
            f"source-{boundary.incoming_source_index}",
        )
        row_title = (
            f"SW {boundary.switch_ordinal:03d} | PLAN BOUNDARY "
            f"{boundary.plan_boundary_index:03d} | t={boundary.output_boundary_seconds:09.3f}s "
            f"| SRC {boundary.outgoing_source_index}:{outgoing} -> "
            f"{boundary.incoming_source_index}:{incoming}"
        )
        transition = (
            f"{boundary.transition_kind}"
            + (
                f" ({boundary.transition_frame_count}f)"
                if boundary.transition_frame_count
                else ""
            )
            + f" | boundary frame #{boundary.boundary_frame_index} | {fps:.6f} fps"
        )
        _put_text(image, row_title, (config.page_margin, y + 18), scale=0.48, thickness=1)
        _put_text(
            image,
            transition,
            (config.page_margin, y + 40),
            scale=0.44,
            color=(174, 205, 255),
        )
        tile_y = y + config.boundary_header_height
        for column, offset in enumerate(offsets):
            frame_index = boundary.boundary_frame_index + offset
            thumbnail = thumbnails.get(frame_index)
            if thumbnail is None:
                raise RuntimeError(f"missing decoded thumbnail for exact frame {frame_index}")
            x = config.page_margin + column * (config.tile_width + config.column_gap)
            side = boundary.sheet_sample_label(offset)
            label = f"{offset:+d}f  {side}  frame #{frame_index}"
            cv2.rectangle(
                image,
                (x, tile_y),
                (x + config.tile_width - 1, tile_y + config.tile_label_height - 1),
                (38, 38, 38),
                -1,
            )
            _put_text(image, label, (x + 7, tile_y + 19), scale=0.43)
            frame_y = tile_y + config.tile_label_height
            image[
                frame_y : frame_y + config.tile_height,
                x : x + config.tile_width,
            ] = thumbnail
            cv2.rectangle(
                image,
                (x, frame_y),
                (x + config.tile_width - 1, frame_y + config.tile_height - 1),
                (82, 82, 82),
                1,
            )
        y += row_height + config.row_gap
    return image


def _encode_png(image: np.ndarray) -> bytes:
    encoded, buffer = cv2.imencode(
        ".png",
        image,
        (cv2.IMWRITE_PNG_COMPRESSION, 3),
    )
    if not encoded:
        raise RuntimeError("OpenCV could not encode a contact-sheet PNG")
    return buffer.tobytes()


def _atomic_write_bytes(destination: Path, payload: bytes) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return destination.resolve()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def generate_transition_contact_sheets(
    render_plan_path: Path,
    video_path: Path,
    output_directory: Path,
    *,
    config: ContactSheetConfig | None = None,
    decoder_factory: DecoderFactory = OpenCVFrameDecoder,
    write_manifest: bool = True,
) -> TransitionContactSheetResult:
    """Generate deterministic, paginated final-output source-switch sheets.

    All requested frames are decoded and all pages are PNG-encoded before the
    first destination is replaced.  The manifest and its page hashes are
    committed last, so a consumer can verify a coherent completed run.
    """

    policy = config or ContactSheetConfig()
    plan = _require_input_file(render_plan_path, "render plan")
    video = _require_input_file(video_path, "final rendered MP4")
    output_root = Path(output_directory).resolve()
    if output_root in {plan, video}:
        raise ValueError("contact-sheet output directory cannot be an input artifact")
    if output_root.exists() and not output_root.is_dir():
        raise ValueError("contact-sheet output directory path is not a directory")
    plan_identity = _file_identity(plan)
    video_identity = _file_identity(video)
    payload, plan_encoded = _load_json_object(plan)
    (
        fps,
        expected_width,
        expected_height,
        expected_frame_count,
        boundaries,
        _source_labels,
    ) = _parse_switch_boundaries(
        payload,
        render_plan_path=plan,
        video_path=video,
    )

    requested_indices = tuple(
        sorted(
            {
                boundary.boundary_frame_index + offset
                for boundary in boundaries
                for offset in policy.frame_offsets
            }
        )
    )
    if requested_indices[0] < 0 or requested_indices[-1] >= expected_frame_count:
        raise ValueError("source-switch frame window exceeds the final output timeline")

    page_count = math.ceil(len(boundaries) / policy.boundaries_per_page)
    video_stem = video.stem
    planned_page_paths = tuple(
        output_root
        / (
            f"{video_stem}-source-switches-page-"
            f"{page_index + 1:03d}-of-{page_count:03d}.png"
        )
        for page_index in range(page_count)
    )
    planned_manifest_path = (
        output_root / f"{video_stem}-source-switches-manifest.json"
        if write_manifest
        else None
    )
    protected_inputs = {plan.resolve(), video.resolve()}
    planned_destinations = planned_page_paths + (
        (planned_manifest_path,) if planned_manifest_path is not None else ()
    )
    if len({path.resolve() for path in planned_destinations}) != len(
        planned_destinations
    ):
        raise ValueError("contact-sheet destination paths must be unique")
    if any(path.resolve() in protected_inputs for path in planned_destinations):
        raise ValueError("contact-sheet output cannot overwrite an input artifact")

    decoder = decoder_factory(video)
    try:
        metadata = decoder.metadata
        if abs(metadata.fps - fps) > policy.fps_tolerance:
            raise RuntimeError(
                f"final rendered video FPS {metadata.fps:.6f} does not match "
                f"render plan FPS {fps:.6f}"
            )
        if metadata.frame_count != expected_frame_count:
            raise RuntimeError(
                f"final rendered video frame count {metadata.frame_count} does not "
                f"match render plan frame count {expected_frame_count}"
            )
        if (metadata.width, metadata.height) != (expected_width, expected_height):
            raise RuntimeError(
                "final rendered video dimensions "
                f"{metadata.width}x{metadata.height} do not match render plan "
                f"dimensions {expected_width}x{expected_height}"
            )
        requested = set(requested_indices)
        thumbnails: dict[int, np.ndarray] = {}
        for frame_index, frame in decoder.iter_frames(requested_indices):
            if frame_index not in requested:
                raise RuntimeError(f"decoder yielded unrequested frame {frame_index}")
            if frame_index in thumbnails:
                raise RuntimeError(f"decoder yielded duplicate frame {frame_index}")
            _validate_decoded_frame(frame, metadata, frame_index)
            thumbnails[frame_index] = _letterbox(
                frame,
                policy.tile_width,
                policy.tile_height,
            )
        missing = sorted(requested - thumbnails.keys())
        if missing:
            preview = ", ".join(str(value) for value in missing[:8])
            raise RuntimeError(f"final rendered video did not decode exact frames: {preview}")
    finally:
        decoder.close()

    encoded_pages: list[tuple[Path, bytes, tuple[SourceSwitchBoundary, ...]]] = []
    for page_index in range(page_count):
        start = page_index * policy.boundaries_per_page
        page_boundaries = boundaries[start : start + policy.boundaries_per_page]
        page = _render_page(
            page_boundaries=page_boundaries,
            thumbnails=thumbnails,
            offsets=policy.frame_offsets,
            page_number=page_index + 1,
            page_count=page_count,
            fps=fps,
            video_name=video.name,
            config=policy,
        )
        encoded_pages.append(
            (
                planned_page_paths[page_index],
                _encode_png(page),
                tuple(page_boundaries),
            )
        )

    _require_inputs_unchanged(
        plan=plan,
        video=video,
        plan_identity=plan_identity,
        video_identity=video_identity,
        plan_encoded=plan_encoded,
    )
    plan_sha256 = _sha256_bytes(plan_encoded)
    page_manifest = [
        {
            "page_number": index + 1,
            "path": str(path.resolve()),
            "byte_size": len(encoded),
            "sha256": _sha256_bytes(encoded),
            "switch_ordinals": [item.switch_ordinal for item in page_boundaries],
            "plan_boundary_indices": [
                item.plan_boundary_index for item in page_boundaries
            ],
        }
        for index, (path, encoded, page_boundaries) in enumerate(encoded_pages)
    ]
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_BASIS,
        "created_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "render_plan_path": str(plan),
            "render_plan_sha256": plan_sha256,
            "video_path": str(video),
            "video_byte_size": video_identity[2],
        },
        "video": {
            "plan_fps": fps,
            "decoded_fps": metadata.fps,
            "width": metadata.width,
            "height": metadata.height,
            "frame_count": metadata.frame_count,
        },
        "sampling": {
            "frame_offsets": list(policy.frame_offsets),
            "sampled_distinct_frame_count": len(requested_indices),
            "source_switch_only": True,
            "same_source_segment_boundaries_excluded": True,
            "decode_basis": (
                "sequential_full_decode_zero_based_final_output_frames_"
                "through_verified_eof_no_random_seek"
            ),
            "raw_source_frames_persisted": False,
            "embeddings_persisted": False,
        },
        "source_switch_count": len(boundaries),
        "page_count": page_count,
        "pages": page_manifest,
        "boundaries": [
            boundary.to_dict(policy.frame_offsets) for boundary in boundaries
        ],
    }
    manifest_encoded = json.dumps(
        manifest_payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")

    page_paths = tuple(
        _atomic_write_bytes(path, encoded)
        for path, encoded, _page_boundaries in encoded_pages
    )
    _require_inputs_unchanged(
        plan=plan,
        video=video,
        plan_identity=plan_identity,
        video_identity=video_identity,
        plan_encoded=plan_encoded,
    )
    manifest_path: Path | None = None
    if planned_manifest_path is not None:
        manifest_path = _atomic_write_bytes(
            planned_manifest_path,
            manifest_encoded,
        )
    return TransitionContactSheetResult(
        render_plan_path=plan,
        video_path=video,
        output_directory=output_root,
        page_paths=page_paths,
        manifest_path=manifest_path,
        source_switch_count=len(boundaries),
        sampled_frame_count=len(requested_indices),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m crosscut.transition_contact_sheets",
        description=(
            "Generate final-output PNG contact sheets at every render-plan "
            "source switch."
        ),
    )
    parser.add_argument("--render-plan", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--boundaries-per-page", type=int, default=6)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--tile-height", type=int, default=180)
    parser.add_argument("--no-manifest", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = generate_transition_contact_sheets(
            arguments.render_plan,
            arguments.video,
            arguments.output_directory,
            config=ContactSheetConfig(
                boundaries_per_page=arguments.boundaries_per_page,
                tile_width=arguments.tile_width,
                tile_height=arguments.tile_height,
            ),
            write_manifest=not arguments.no_manifest,
        )
    except Exception as exc:  # Standalone human-inspection CLI boundary.
        print(f"transition contact-sheet generation failed: {exc}", file=os.sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
