"""FFmpeg render-plan construction for synchronized cross-edits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable

from .ive import AspectRatio


RENDER_PLAN_SOURCE_INDEX_DOMAIN = "render_plan_sources_array_index"


@dataclass(frozen=True, slots=True)
class NormalizedCrop:
    """A normalized crop rectangle in source coordinates."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("crop values must be finite")
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("crop values must be normalized to 0...1")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("crop dimensions must be positive")
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("crop rectangle exceeds source bounds")


@dataclass(frozen=True, slots=True)
class PanCropKeyframe:
    """A segment-relative crop sample used for smooth target-member pans."""

    time_seconds: float
    crop: NormalizedCrop

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_seconds) or self.time_seconds < 0:
            raise ValueError("crop keyframe time must be non-negative and finite")


@dataclass(frozen=True, slots=True)
class PanCropTrack:
    """Per-shot virtual-camera path.

    FFmpeg's crop x/y expressions are evaluated for every frame.  Zoom is held
    to the widest requested keyframe inside one short edit segment so changes
    in detector box size cannot create visible breathing; scene/beat cuts are
    the safe places where the crop size may change.
    """

    keyframes: tuple[PanCropKeyframe, ...]

    def __post_init__(self) -> None:
        if not self.keyframes:
            raise ValueError("pan crop track needs at least one keyframe")
        if abs(self.keyframes[0].time_seconds) > 1e-9:
            raise ValueError("pan crop track must start at segment time zero")
        previous = -1.0
        ratios: list[float] = []
        for keyframe in self.keyframes:
            if keyframe.time_seconds <= previous:
                raise ValueError("crop keyframe times must be strictly increasing")
            previous = keyframe.time_seconds
            ratios.append(keyframe.crop.width / keyframe.crop.height)
        if max(ratios) - min(ratios) > 0.002:
            raise ValueError("crop keyframes must use one normalized aspect ratio")

    @property
    def fixed_crop_size(self) -> tuple[float, float]:
        ratio = self.keyframes[0].crop.width / self.keyframes[0].crop.height
        height = max(keyframe.crop.height for keyframe in self.keyframes)
        width = ratio * height
        if width > 1.0 + 1e-9 or height > 1.0 + 1e-9:
            raise ValueError("pan crop track cannot fit inside the source frame")
        return (min(width, 1.0), min(height, 1.0))

    def fixed_crop_positions(self) -> tuple[tuple[float, float, float], ...]:
        width, height = self.fixed_crop_size
        positions: list[tuple[float, float, float]] = []
        for keyframe in self.keyframes:
            center_x = keyframe.crop.x + keyframe.crop.width / 2.0
            center_y = keyframe.crop.y + keyframe.crop.height / 2.0
            left = min(max(center_x - width / 2.0, 0.0), 1.0 - width)
            top = min(max(center_y - height / 2.0, 0.0), 1.0 - height)
            positions.append((keyframe.time_seconds, left, top))
        return tuple(positions)

    def simplified(
        self,
        *,
        max_keyframes: int = 80,
        tolerance: float = 0.0015,
    ) -> "PanCropTrack":
        """Reduce a dense tracker path before building an FFmpeg expression.

        FFmpeg's expression parser has a practical nesting limit.  Short edit
        segments rarely reach it, but this Ramer-Douglas-Peucker pass preserves
        bends while placing a hard cap on adversarial/dense paths.  The widest
        keyframe is mandatory so simplification cannot accidentally zoom in.
        """

        if (
            isinstance(max_keyframes, bool)
            or not isinstance(max_keyframes, int)
            or max_keyframes < 3
        ):
            raise ValueError("max_keyframes must be an integer of at least 3")
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be positive and finite")
        if len(self.keyframes) <= max_keyframes:
            return self

        positions = self.fixed_crop_positions()
        widest_index = max(
            range(len(self.keyframes)),
            key=lambda index: self.keyframes[index].crop.height,
        )

        def rdp(first: int, last: int, epsilon: float) -> set[int]:
            if last <= first + 1:
                return {first, last}
            start_time, start_x, start_y = positions[first]
            end_time, end_x, end_y = positions[last]
            span = end_time - start_time
            maximum_error = -1.0
            maximum_index = first
            for index in range(first + 1, last):
                time_value, x_value, y_value = positions[index]
                alpha = (time_value - start_time) / span
                expected_x = start_x + (end_x - start_x) * alpha
                expected_y = start_y + (end_y - start_y) * alpha
                error = math.hypot(x_value - expected_x, y_value - expected_y)
                if error > maximum_error:
                    maximum_error = error
                    maximum_index = index
            if maximum_error <= epsilon:
                return {first, last}
            return rdp(first, maximum_index, epsilon) | rdp(
                maximum_index, last, epsilon
            )

        epsilon = tolerance
        mandatory = {0, widest_index, len(self.keyframes) - 1}
        selected: set[int] = set()
        for _ in range(24):
            selected = rdp(0, len(self.keyframes) - 1, epsilon) | mandatory
            if len(selected) <= max_keyframes:
                break
            epsilon *= 1.6
        if len(selected) > max_keyframes:
            optional = sorted(selected - mandatory)
            capacity = max_keyframes - len(mandatory)
            if capacity > 0:
                chosen = {
                    optional[round(index * (len(optional) - 1) / max(1, capacity - 1))]
                    for index in range(capacity)
                }
            else:
                chosen = set()
            selected = mandatory | chosen
        return PanCropTrack(tuple(self.keyframes[index] for index in sorted(selected)))


@dataclass(frozen=True, slots=True)
class RenderSegment:
    source_index: int
    source_start: float
    duration: float
    master_start: float
    crop: NormalizedCrop | None = None
    pan_crop: PanCropTrack | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_index, bool)
            or not isinstance(self.source_index, int)
            or self.source_index < 0
        ):
            raise ValueError("source_index must be non-negative")
        if not all(
            math.isfinite(value)
            for value in (self.source_start, self.duration, self.master_start)
        ):
            raise ValueError("segment times must be finite")
        if self.source_start < 0 or self.master_start < 0:
            raise ValueError("segment times must be non-negative")
        if self.duration <= 0:
            raise ValueError("segment duration must be positive")
        if self.crop is not None and self.pan_crop is not None:
            raise ValueError("segment cannot use both static and panning crops")
        if (
            self.pan_crop is not None
            and self.pan_crop.keyframes[-1].time_seconds > self.duration + 1e-6
        ):
            raise ValueError("crop keyframes must stay inside the segment")


class TransitionKind(str, Enum):
    """Video transition applied at one nominal segment boundary."""

    HARD_CUT = "hard_cut"
    FACE_DISSOLVE = "face_dissolve"


@dataclass(frozen=True, slots=True)
class RenderTransition:
    """Explicit per-boundary transition metadata.

    ``FACE_DISSOLVE`` uses an outgoing-only source handle.  The incoming shot
    still begins at the nominal cut, so both pictures show the same canonical
    song time during the overlap and the edit's total frame count is unchanged.
    """

    kind: TransitionKind = TransitionKind.HARD_CUT
    frame_count: int = 0
    reason: str = "default_hard_cut"
    face_mismatch: float | None = None
    tier: str | None = None
    appearance_min_cosine_similarity: float | None = None
    future_pose_max_mismatch: float | None = None
    future_silhouette_max_mismatch: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TransitionKind):
            object.__setattr__(self, "kind", TransitionKind(self.kind))
        if (
            isinstance(self.frame_count, bool)
            or not isinstance(self.frame_count, int)
            or self.frame_count < 0
        ):
            raise ValueError("transition frame_count must be a non-negative integer")
        if self.kind is TransitionKind.HARD_CUT and self.frame_count != 0:
            raise ValueError("hard cuts cannot have overlap frames")
        if self.kind is TransitionKind.FACE_DISSOLVE and self.frame_count < 3:
            raise ValueError("face dissolves need at least three frames")
        if not self.reason:
            raise ValueError("transition reason cannot be blank")
        if self.tier is not None and not self.tier.strip():
            raise ValueError("transition tier cannot be blank")
        if self.face_mismatch is not None and (
            not math.isfinite(self.face_mismatch)
            or not 0.0 <= self.face_mismatch <= 1.0
        ):
            raise ValueError("face mismatch must be normalized")
        if self.appearance_min_cosine_similarity is not None and (
            not math.isfinite(self.appearance_min_cosine_similarity)
            or not -1.0 <= self.appearance_min_cosine_similarity <= 1.0
        ):
            raise ValueError("appearance cosine similarity must be normalized")
        for value in (
            self.future_pose_max_mismatch,
            self.future_silhouette_max_mismatch,
        ):
            if value is not None and (
                not math.isfinite(value) or not 0.0 <= value <= 1.0
            ):
                raise ValueError("future body mismatch must be normalized")

    def duration_seconds(self, fps: float) -> float:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")
        return self.frame_count / fps

    def to_dict(self, fps: float) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "frame_count": self.frame_count,
            "duration_seconds": self.duration_seconds(fps),
            "reason": self.reason,
            "face_mismatch": self.face_mismatch,
            "tier": self.tier,
            "appearance_min_cosine_similarity": (
                self.appearance_min_cosine_similarity
            ),
            "future_pose_max_mismatch": self.future_pose_max_mismatch,
            "future_silhouette_max_mismatch": (
                self.future_silhouette_max_mismatch
            ),
        }


@dataclass(frozen=True, slots=True)
class RenderPlan:
    sources: tuple[Path, ...]
    segments: tuple[RenderSegment, ...]
    reference_source: int
    aspect_ratio: AspectRatio
    output: Path
    master_audio_start: float = 0.0
    width: int = 1920
    height: int = 1080
    fps: float = 29.97
    transitions: tuple[RenderTransition, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.aspect_ratio, AspectRatio):
            object.__setattr__(self, "aspect_ratio", AspectRatio(self.aspect_ratio))
        if len(self.sources) < 1:
            raise ValueError("render plan needs at least one source")
        if not self.segments:
            raise ValueError("render plan needs at least one segment")
        if (
            isinstance(self.reference_source, bool)
            or not isinstance(self.reference_source, int)
            or not 0 <= self.reference_source < len(self.sources)
        ):
            raise ValueError("reference_source is outside the source list")
        if not math.isfinite(self.master_audio_start) or self.master_audio_start < 0:
            raise ValueError("master_audio_start must be non-negative")
        if (
            isinstance(self.width, bool)
            or not isinstance(self.width, int)
            or isinstance(self.height, bool)
            or not isinstance(self.height, int)
            or self.width <= 0
            or self.height <= 0
            or self.width % 2
            or self.height % 2
        ):
            raise ValueError("output dimensions must be positive even integers")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive and finite")
        expected_ratio = 16 / 9 if self.aspect_ratio is AspectRatio.LANDSCAPE else 9 / 16
        if abs(self.width / self.height - expected_ratio) > 0.002:
            raise ValueError("output dimensions do not match aspect_ratio")
        if any(segment.source_index >= len(self.sources) for segment in self.segments):
            raise ValueError("segment references an unknown source")
        resolved_sources = {source.resolve() for source in self.sources}
        if self.output.resolve() in resolved_sources:
            raise ValueError("output must not overwrite a source video")
        expected = 0.0
        for segment in self.segments:
            if abs(segment.master_start - expected) > 1e-6:
                raise ValueError("segments must be contiguous and ordered on the master timeline")
            expected += segment.duration
        if not self.transitions and len(self.segments) > 1:
            object.__setattr__(
                self,
                "transitions",
                tuple(RenderTransition() for _ in range(len(self.segments) - 1)),
            )
        if len(self.transitions) != len(self.segments) - 1:
            raise ValueError("render plan needs one transition per segment boundary")
        counts = segment_frame_counts(self)
        for index, transition in enumerate(self.transitions):
            if (
                transition.kind is TransitionKind.FACE_DISSOLVE
                and transition.frame_count > counts[index + 1]
            ):
                raise ValueError("face dissolve cannot exceed the incoming segment")
            if (
                transition.kind is TransitionKind.FACE_DISSOLVE
                and self.segments[index].source_index
                == self.segments[index + 1].source_index
            ):
                raise ValueError("face dissolve requires a real source change")

    @property
    def duration(self) -> float:
        return sum(segment.duration for segment in self.segments)

    @property
    def output_frame_count(self) -> int:
        return int(round(self.duration * self.fps))

    @property
    def render_duration(self) -> float:
        return self.output_frame_count / self.fps

    def to_json(self) -> dict[str, object]:
        return {
            "sources": [str(path) for path in self.sources],
            "segments": [asdict(segment) for segment in self.segments],
            "reference_source": self.reference_source,
            "source_index_domains": {
                "sources[]": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
                "reference_source": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
                "segments[].source_index": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
            },
            "aspect_ratio": self.aspect_ratio.value,
            "output": str(self.output),
            "master_audio_start": self.master_audio_start,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "transitions": [
                transition.to_dict(self.fps) for transition in self.transitions
            ],
            "duration": self.duration,
            "render_duration": self.render_duration,
            "output_frame_count": self.output_frame_count,
        }


def output_size(aspect_ratio: AspectRatio, long_edge: int = 1920) -> tuple[int, int]:
    if isinstance(long_edge, bool) or not isinstance(long_edge, int) or long_edge <= 0:
        raise ValueError("long_edge must be a positive integer")
    long_edge -= long_edge % 2
    short_edge = int(round(long_edge * 9 / 16))
    short_edge -= short_edge % 2
    if aspect_ratio is AspectRatio.PORTRAIT:
        return (short_edge, long_edge)
    return (long_edge, short_edge)


def aligned_render_segment(
    *,
    source_index: int,
    reference_start: float,
    common_master_start: float,
    source_offset: float,
    duration: float,
    crop: NormalizedCrop | None = None,
    pan_crop: PanCropTrack | None = None,
) -> RenderSegment:
    """Map one canonical reference span into output and source time.

    Alignment uses the invariant ``source_time = reference_time + offset``.
    The rendered output starts at zero even when the common usable reference
    range starts later, so ``master_start`` deliberately subtracts
    ``common_master_start`` while ``source_start`` does not.
    """

    values = (reference_start, common_master_start, source_offset, duration)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("alignment times must be finite")
    if reference_start < common_master_start:
        raise ValueError("reference_start precedes the common master range")
    return RenderSegment(
        source_index=source_index,
        source_start=reference_start + source_offset,
        duration=duration,
        master_start=reference_start - common_master_start,
        crop=crop,
        pan_crop=pan_crop,
    )


def _escape_filter_expression(expression: str) -> str:
    return expression.replace("\\", "\\\\").replace("'", "\\'")


def _piecewise_linear_expression(
    positions: tuple[tuple[float, float, float], ...],
    coordinate_index: int,
) -> str:
    """Return an FFmpeg expression interpolating keyframe x or y positions."""

    if coordinate_index not in (1, 2):
        raise ValueError("coordinate_index must select x or y")
    if len(positions) == 1:
        return f"{positions[0][coordinate_index]:.10f}"
    expression = f"{positions[-1][coordinate_index]:.10f}"
    for left, right in reversed(tuple(zip(positions, positions[1:]))):
        start_time = left[0]
        end_time = right[0]
        start_value = left[coordinate_index]
        end_value = right[coordinate_index]
        slope = (end_value - start_value) / (end_time - start_time)
        linear = f"({start_value:.10f}+({slope:.10f})*(t-{start_time:.10f}))"
        expression = f"if(lt(t,{end_time:.10f}),{linear},{expression})"
    return expression


def segment_frame_counts(plan: RenderPlan) -> tuple[int, ...]:
    """Quantize global cut boundaries once so per-cut rounding cannot drift."""

    counts: list[int] = []
    cumulative_time = 0.0
    previous_boundary = 0
    for segment in plan.segments:
        cumulative_time += segment.duration
        boundary = int(round(cumulative_time * plan.fps))
        count = boundary - previous_boundary
        if count < 2:
            raise ValueError(
                f"segment at {segment.master_start:.6f}s is shorter than two output frames"
            )
        counts.append(count)
        previous_boundary = boundary
    if sum(counts) != plan.output_frame_count:
        raise ValueError("quantized segment frames do not match the output duration")
    return tuple(counts)


def _video_chain(
    *,
    source_index: int,
    source_start: float,
    source_duration: float,
    crop: NormalizedCrop | None,
    pan_crop: PanCropTrack | None,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    output_label: str,
) -> str:
    chain = [
        f"[{source_index}:v]trim=start={source_start:.6f}:duration={source_duration:.6f}",
        "setpts=PTS-STARTPTS",
    ]
    if crop is not None:
        chain.append(
            "crop="
            f"w=trunc(iw*{crop.width:.8f}/2)*2:"
            f"h=trunc(ih*{crop.height:.8f}/2)*2:"
            f"x=iw*{crop.x:.8f}:y=ih*{crop.y:.8f}"
        )
    elif pan_crop is not None:
        pan_crop = pan_crop.simplified()
        width_ratio, height_ratio = pan_crop.fixed_crop_size
        positions = pan_crop.fixed_crop_positions()
        x_expression = _escape_filter_expression(
            _piecewise_linear_expression(positions, 1)
        )
        y_expression = _escape_filter_expression(
            _piecewise_linear_expression(positions, 2)
        )
        chain.append(
            "crop="
            f"w=trunc(iw*{width_ratio:.8f}/2)*2:"
            f"h=trunc(ih*{height_ratio:.8f}/2)*2:"
            f"x='iw*({x_expression})':y='ih*({y_expression})'"
        )
    chain.extend(
        [
            f"scale={width}:{height}:force_original_aspect_ratio=decrease",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
            "setsar=1",
            f"fps={fps:.6f}",
            "tpad=stop_mode=clone:stop_duration=1",
            f"trim=end_frame={frame_count}",
            f"setpts=N/({fps:.6f}*TB)",
            "settb=AVTB",
            "format=yuv420p",
        ]
    )
    return ",".join(chain) + f"[{output_label}]"


def _video_filter(
    index: int,
    segment: RenderSegment,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    extension_frames: int = 0,
) -> str:
    if extension_frames == 0:
        return _video_chain(
            source_index=segment.source_index,
            source_start=segment.source_start,
            source_duration=segment.duration,
            crop=segment.crop,
            pan_crop=segment.pan_crop,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            output_label=f"v{index}",
        )

    # Quantize the nominal shot and its dissolve handle independently.  The
    # handle therefore seeks to the exact canonical source boundary instead of
    # inheriting up to one frame of cumulative duration-rounding error.
    base = _video_chain(
        source_index=segment.source_index,
        source_start=segment.source_start,
        source_duration=segment.duration,
        crop=segment.crop,
        pan_crop=segment.pan_crop,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        output_label=f"vb{index}",
    )
    handle_crop = segment.crop
    if segment.pan_crop is not None:
        track = segment.pan_crop.simplified()
        crop_width, crop_height = track.fixed_crop_size
        _time, left, top = track.fixed_crop_positions()[-1]
        handle_crop = NormalizedCrop(left, top, crop_width, crop_height)
    handle = _video_chain(
        source_index=segment.source_index,
        source_start=segment.source_start + segment.duration,
        source_duration=extension_frames / fps,
        crop=handle_crop,
        pan_crop=None,
        width=width,
        height=height,
        fps=fps,
        frame_count=extension_frames,
        output_label=f"vh{index}",
    )
    return (
        f"{base};\n{handle};\n"
        f"[vb{index}][vh{index}]concat=n=2:v=1:a=0[v{index}]"
    )


def build_filter_graph(plan: RenderPlan) -> str:
    frame_counts = segment_frame_counts(plan)
    extension_frames = tuple(
        transition.frame_count
        if transition.kind is TransitionKind.FACE_DISSOLVE
        else 0
        for transition in plan.transitions
    ) + (0,)
    parts = [
        _video_filter(
            index,
            segment,
            plan.width,
            plan.height,
            plan.fps,
            frame_counts[index],
            extension_frames[index],
        )
        for index, segment in enumerate(plan.segments)
    ]
    # Build each maximal run of dissolve-linked shots as one xfade chain, then
    # concatenate those runs across ordinary hard cuts.  Extending only the
    # outgoing side by the overlap makes every xfade duration-neutral.
    chunk_labels: list[str] = []
    index = 0
    xfade_index = 0
    while index < len(plan.segments):
        current_label = f"v{index}"
        cumulative_frames = frame_counts[index]
        while (
            index < len(plan.transitions)
            and plan.transitions[index].kind is TransitionKind.FACE_DISSOLVE
        ):
            transition = plan.transitions[index]
            next_index = index + 1
            output_label = f"vx{xfade_index}"
            parts.append(
                f"[{current_label}][v{next_index}]"
                "xfade=transition=fade:"
                f"duration={transition.duration_seconds(plan.fps):.9f}:"
                f"offset={cumulative_frames / plan.fps:.9f},format=yuv420p"
                f"[{output_label}]"
            )
            current_label = output_label
            cumulative_frames += frame_counts[next_index]
            xfade_index += 1
            index = next_index
        chunk_labels.append(current_label)
        index += 1
    if len(chunk_labels) == 1:
        parts.append(f"[{chunk_labels[0]}]null[vcat]")
    else:
        labels = "".join(f"[{label}]" for label in chunk_labels)
        parts.append(f"{labels}concat=n={len(chunk_labels)}:v=1:a=0[vcat]")
    parts.append(
        f"[vcat]fps={plan.fps:.6f},tpad=stop_mode=clone:stop_duration=1,"
        f"trim=end_frame={plan.output_frame_count},"
        f"setpts=N/({plan.fps:.6f}*TB),settb=AVTB[vout]"
    )
    parts.append(
        f"[{plan.reference_source}:a]atrim=start={plan.master_audio_start:.6f}:"
        f"duration={plan.render_duration:.6f},"
        "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0[aout]"
    )
    return ";\n".join(parts)


def render(
    plan: RenderPlan,
    *,
    ffmpeg: str = "ffmpeg",
    video_codec: str = "h264_videotoolbox",
) -> str:
    plan.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary output on the destination volume so the final replace
    # is atomic even when the user renders to an external SSD.
    with tempfile.TemporaryDirectory(
        prefix=".ive-crosscut-", dir=plan.output.parent
    ) as temp:
        root = Path(temp)
        filter_path = root / "filter.ffscript"
        filter_path.write_text(build_filter_graph(plan), encoding="utf-8")
        temporary_output = root / f"render{plan.output.suffix or '.mp4'}"
        codecs = (video_codec, "libx264") if video_codec == "h264_videotoolbox" else (video_codec,)
        failures: list[str] = []
        for codec in codecs:
            command: list[str] = [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-y",
            ]
            for source in plan.sources:
                command.extend(("-i", str(source)))
            command.extend(
                (
                    "-filter_complex_script",
                    str(filter_path),
                    "-map",
                    "[vout]",
                    "-map",
                    "[aout]",
                    "-c:v",
                    codec,
                )
            )
            if codec == "libx264":
                # A quality-targeted software fallback avoids wasting bits on
                # static shots while retaining fine stage lighting, confetti,
                # and hair detail in high-motion sections.
                command.extend(("-preset", "slow", "-crf", "16", "-profile:v", "high"))
            else:
                # VideoToolbox has no CRF-equivalent quality mode. Give the
                # hardware encoder enough headroom for dense choreography and
                # broadcast lighting while retaining a compatible H.264 file.
                command.extend(
                    (
                        "-b:v",
                        "24M",
                        "-maxrate",
                        "36M",
                        "-bufsize",
                        "48M",
                        "-profile:v",
                        "high",
                    )
                )
            command.extend(
                (
                    "-tag:v",
                    "avc1",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "320k",
                    "-movflags",
                    "+faststart",
                    str(temporary_output),
                )
            )
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                temporary_output.replace(plan.output)
                return codec
            failures.append(f"{codec}: {completed.stderr.strip() or 'unknown FFmpeg error'}")
        raise RuntimeError("video render failed: " + " | ".join(failures))


def write_plan(plan: RenderPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")


def contiguous_segments(segments: Iterable[RenderSegment]) -> tuple[RenderSegment, ...]:
    return tuple(sorted(segments, key=lambda segment: segment.master_start))
