"""Exact native-frame schedules, without re-quantizing a fractional trim.

The sampler's selected frame indices are the authority. FFmpeg receives a
monotone integer output-tick schedule rather than independently rounding a
source trim and then resetting its first decoded PTS to zero. Only verified
zero-origin sources with a verified presentation-index grid can use this path.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Sequence


FRAME_SCHEDULE_CONTRACT = "explicit_native_frame_to_output_tick_schedule_v1"


@dataclass(frozen=True, slots=True)
class VerifiedRenderFrameClock:
    frame_rate: Fraction
    frame_count: int
    packet_grid_sha256: str
    grid_basis: str = "exact_cfr_presentation_grid"

    def __post_init__(self) -> None:
        if not isinstance(self.frame_rate, Fraction) or self.frame_rate <= 0:
            raise ValueError("verified render frame rate must be a positive rational")
        if isinstance(self.frame_count, bool) or not isinstance(self.frame_count, int) or not 2 <= self.frame_count <= 1_000_000:
            raise ValueError("verified render frame count is invalid")
        if not isinstance(self.packet_grid_sha256, str) or len(self.packet_grid_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.packet_grid_sha256):
            raise ValueError("verified render frame clock needs a grid digest")

    def to_dict(self) -> dict[str, object]:
        return {"frame_rate": str(self.frame_rate), "frame_count": self.frame_count,
                "packet_grid_sha256": self.packet_grid_sha256,
                "grid_basis": self.grid_basis, "contract": FRAME_SCHEDULE_CONTRACT}


def render_frame_clock_from_probe(payload: dict) -> VerifiedRenderFrameClock | None:
    """Prove a complete zero-origin presentation-index grid, not average fps.

    H.264 frame-index parity is tested using real B-frame media. AV1 also has
    the existing production sampler's complete-grid decoder parity coverage.
    VP9 can retain millisecond timestamp quantization after MP4 remuxing. Its
    packet PTS must map bijectively to all native indices under the sampler's
    nearest-index clock; a missing/duplicated/cross-cell frame fails the proof.
    Unproven sources retain the timestamp renderer; they are not excluded.
    """
    try:
        streams, packets = payload['streams'], payload['packets']
        if len(streams) != 1 or streams[0]['codec_name'] not in {'h264', 'av1', 'vp9'}:
            return None
        stream = streams[0]
        rate, time_base = Fraction(stream['avg_frame_rate']), Fraction(stream['time_base'])
        if rate <= 0 or time_base <= 0 or Fraction(stream['start_time']) != 0:
            return None
        count = int(stream['nb_frames'])
        if not 2 <= count <= 1_000_000 or len(packets) != count:
            return None
        stride = 1 / rate / time_base
        vp9 = stream['codec_name'] == 'vp9'
        if not vp9 and (rate != Fraction(stream['r_frame_rate']) or stride.denominator != 1):
            return None
        pts = []
        for packet in packets:
            point, duration = packet['pts'], packet['duration']
            if isinstance(point, bool) or not isinstance(point, int) or isinstance(duration, bool) or not isinstance(duration, int) or duration <= 0:
                return None
            if not vp9 and duration != stride.numerator:
                return None
            pts.append(point)
        ordered_pts = sorted(pts)
        if vp9:
            if len(set(ordered_pts)) != count or ordered_pts[0] != 0:
                return None
            for index, point in enumerate(ordered_pts):
                position = point * time_base * rate
                # A rational half-cell boundary is ambiguous under floating
                # decoder arithmetic, so it cannot establish index identity.
                if int(position + Fraction(1, 2)) != index or abs(position - index) == Fraction(1, 2):
                    return None
        elif ordered_pts != [index * stride.numerator for index in range(count)]:
            return None
        digest = hashlib.sha256(json.dumps([str(rate), str(time_base), sorted(pts)], separators=(',', ':')).encode()).hexdigest()
        return VerifiedRenderFrameClock(rate, count, digest,
            "bijective_nearest_index_presentation_grid" if vp9 else "exact_cfr_presentation_grid")
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def probe_render_frame_clock(path: Path, *, ffprobe: str = 'ffprobe') -> VerifiedRenderFrameClock | None:
    """Inspect only a requested render source, without changing its media."""
    try:
        before = path.stat()
        result = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
            '-show_streams', '-show_packets', '-show_entries',
            'stream=codec_name,avg_frame_rate,r_frame_rate,time_base,start_time,nb_frames:packet=pts,duration',
            '-of', 'json', str(path)], check=True, capture_output=True, timeout=10.)
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            return None
        return render_frame_clock_from_probe(json.loads(result.stdout))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class NativeFrameSchedule:
    """Native frame required at each output tick, repeats included."""

    source_frame_indices: tuple[int, ...]
    terminal_hold_output_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        values = tuple(self.source_frame_indices)
        if not values or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in values):
            raise ValueError("a frame schedule needs nonnegative native indices")
        if any(right < left for left, right in zip(values, values[1:])):
            raise ValueError("a frame schedule cannot reverse source playback")
        holds = tuple(self.terminal_hold_output_indices)
        if holds and (holds != tuple(range(holds[0], len(values))) or holds[0] <= 0 or any(isinstance(i, bool) or not isinstance(i, int) for i in holds)):
            raise ValueError("terminal holds must be an explicit output suffix")
        object.__setattr__(self, "source_frame_indices", values)
        object.__setattr__(self, "terminal_hold_output_indices", holds)

    def first_output_tick(self, source_frame_index: int) -> int:
        return bisect_left(self.source_frame_indices, source_frame_index)

    def ffmpeg_tick_expression(self) -> str:
        # Distinct desired native frames and their first output tick. Frames
        # skipped during downsampling share the next selected frame's tick;
        # fps then retains that last frame, rather than a skipped predecessor.
        points = []
        for tick, native in enumerate(self.source_frame_indices):
            if not points or native != points[-1][0]:
                points.append((native, tick))
        groups: list[tuple[int, str]] = []
        if len(points) == 1:
            groups.append((points[0][0], "0"))
        else:
            start = 0
            while start < len(points) - 1:
                native_step = points[start + 1][0] - points[start][0]
                tick_step = points[start + 1][1] - points[start][1]
                end = start + 1
                # Integer affine runs are exact only when one step is 1.
                # Other schedules still work through constant-tick leaves.
                if native_step == 1 or tick_step == 1:
                    while end + 1 < len(points) and (
                        points[end + 1][0] - points[end][0],
                        points[end + 1][1] - points[end][1],
                    ) == (native_step, tick_step):
                        end += 1
                    native, tick = points[start]
                    expression = (
                        f"clip(ceil((N-{native})*{tick_step}/{native_step})"
                        f"+{tick},{tick},{points[end][1]})"
                    )
                else:
                    # For a non-affine requested schedule the only correct
                    # value between two distinct selected frames is the next
                    # frame's first tick, not an invented interpolation.
                    native, tick = points[start]
                    expression = f"if(lte(N,{native}),{tick},{points[end][1]})"
                groups.append((points[end][0], expression))
                start = end

        def tree(items: Sequence[tuple[int, str]]) -> str:
            if len(items) == 1:
                return items[0][1]
            middle = len(items) // 2
            return (
                f"if(lte(N,{items[middle - 1][0]}),"
                f"{tree(items[:middle])},{tree(items[middle:])})"
            )

        return (
            # Keep the post-budget PTS increasing. A constant tail tick makes
            # fps await source EOF before downstream trim can finish, turning
            # a short excerpt into a full-video decode.
            f"if(gt(N,{points[-1][0]}),{len(self.source_frame_indices)}+N-{points[-1][0]},"
            f"{tree(groups)})"
        )

    def ffmpeg_filters(self, output_fps: float) -> tuple[str, ...]:
        if isinstance(output_fps, bool) or not math.isfinite(output_fps) or output_fps <= 0:
            raise ValueError("output frame rate must be positive")
        fps = format(output_fps, ".17g")
        return (
            # One real last-frame clone supplies a known EOF lookahead tick.
            # It cannot create a requested source frame outside the bound
            # source domain: that check belongs to the clock-binding caller.
            "tpad=stop_mode=clone:stop=1",
            f"settb=expr=1/({fps})",
            f"setpts='{self.ffmpeg_tick_expression()}'",
            f"fps={fps}:start_time=0",
            f"trim=end_frame={len(self.source_frame_indices)}",
            f"setpts=N/({fps}*TB)",
            "settb=AVTB",
        )


def sampler_frame_schedule(
    *, source_start: float, source_playback_rate: float,
    native_fps: float, output_fps: float, frame_count: int,
    source_frame_count: int,
    maximum_terminal_hold_frames: int = 0,
) -> NativeFrameSchedule:
    """Match the production OpenCV POS_MSEC request arithmetic exactly.

    This deliberately does not read target/ref-match phase information.
    Existing training snapshot ties-to-even quantization is a different
    contract at exact half frames; callers must not mislabel this as its
    replacement without checking/rebuilding those features.
    """
    for value in (source_start, source_playback_rate, native_fps, output_fps):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("source/output clock values must be finite")
    if source_start < 0 or min(source_playback_rate, native_fps, output_fps) <= 0:
        raise ValueError("source/output clock domain is invalid")
    for count in (frame_count, source_frame_count):
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("frame counts must be positive integers")
    if isinstance(maximum_terminal_hold_frames, bool) or not isinstance(maximum_terminal_hold_frames, int) or maximum_terminal_hold_frames < 0:
        raise ValueError("terminal hold allowance must be a nonnegative integer")
    indices, terminal_holds = [], []
    for tick in range(frame_count):
        seconds = source_start + (tick / output_fps) * source_playback_rate
        timestamp_ms = seconds * 1000.0
        index = int((timestamp_ms / 1000.0) * native_fps + 0.5)
        if index >= source_frame_count:
            if not indices or len(terminal_holds) >= maximum_terminal_hold_frames:
                raise ValueError("requested frame schedule exceeds source media")
            # Preserve the renderer's explicit EOF clone policy; never use
            # these as observed source pixels for a detector/teacher label.
            terminal_holds.append(tick)
            index = source_frame_count - 1
        indices.append(index)
    return NativeFrameSchedule(tuple(indices), tuple(terminal_holds))
