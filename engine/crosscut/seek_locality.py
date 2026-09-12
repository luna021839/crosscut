"""Bounded sequential decoding for sources with a verified constant frame clock.

This module is not yet wired into the production sampler. Packet timing must
prove a complete AV1 presentation grid before decoder-index reuse is allowed.
Other codecs, irregular clocks and unsupported decoder backends retain exact
timestamp seeking. No analysis timestamps or output video pixels are rounded.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class ConstantFrameClock:
    frame_rate: Fraction
    frame_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.frame_rate, Fraction) or self.frame_rate <= 0
            or isinstance(self.frame_count, bool)
            or not isinstance(self.frame_count, int)
            or not 2 <= self.frame_count <= 1_000_000
        ):
            raise ValueError('invalid constant frame clock')


def constant_frame_clock_from_probe(
    payload: Mapping[str, Any],
) -> ConstantFrameClock | None:
    """Reject incomplete, duplicated, non-zero-start or variable packet clocks.

Reported average FPS alone is not evidence of constant-rate presentation.
Every packet must occupy one consecutive point of the complete presentation
grid and its duration must match. This deliberately supports only AV1 until
other codecs have their own decoder parity coverage.
    """
    try:
        streams = payload['streams']
        packets = payload['packets']
        if not isinstance(streams, list) or len(streams) != 1:
            return None
        stream = streams[0]
        if stream['codec_name'] != 'av1':
            return None
        rate = Fraction(stream['avg_frame_rate'])
        if rate <= 0 or rate != Fraction(stream['r_frame_rate']):
            return None
        time_base = Fraction(stream['time_base'])
        if time_base <= 0 or Fraction(str(stream['start_time'])) != 0:
            return None
        raw_count = stream['nb_frames']
        if isinstance(raw_count, bool) or not isinstance(raw_count, (int, str)):
            return None
        frame_count = int(raw_count)
        if not 2 <= frame_count <= 1_000_000:
            return None
        if not isinstance(packets, list) or len(packets) != frame_count:
            return None
        stride = 1 / rate / time_base
        if stride.denominator != 1:
            return None
        pts = []
        for packet in packets:
            point = packet['pts']
            duration = packet['duration']
            if (
                isinstance(point, bool) or not isinstance(point, int)
                or isinstance(duration, bool) or not isinstance(duration, int)
                or duration != stride.numerator
            ):
                return None
            pts.append(point)
        if sorted(pts) != [index * stride.numerator for index in range(frame_count)]:
            return None
        return ConstantFrameClock(rate, frame_count)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def probe_constant_frame_clock(
    path: Path,
    *,
    ffprobe: str = 'ffprobe',
    timeout_seconds: float = 10.0,
) -> ConstantFrameClock | None:
    """Inspect packet metadata once; never decode or write source media."""
    try:
        before = path.stat()
        if not path.is_file():
            return None
        result = subprocess.run(
            [ffprobe, '-v', 'error', '-select_streams', 'v:0',
             '-show_streams', '-show_packets', '-show_entries',
             'stream=codec_name,avg_frame_rate,r_frame_rate,time_base,start_time,'
             'nb_frames:packet=pts,duration', '-of', 'json', str(path)],
            capture_output=True, check=True, timeout=timeout_seconds,
        )
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            return None
        return constant_frame_clock_from_probe(json.loads(result.stdout))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class SeekLocalityReader:
    """Reuse a bounded pixel window, or fall back to ordinary timestamp seeks.

The caller still owns the exact rational request identities and persistent
cache. These decoder-index aliases are transient, source-local pixels only.
Any observed decoder position or clock inconsistency disables reuse instead
of issuing unverified pixels. Exceptions from the frame processor propagate.
    """

    def __init__(
        self,
        capture: Any,
        *,
        clock: ConstantFrameClock | None,
        process_frame: Callable[[np.ndarray], np.ndarray],
        maximum_cached_frames: int = 32,
        maximum_forward_frames: int = 48,
    ) -> None:
        if (
            isinstance(maximum_cached_frames, bool)
            or not isinstance(maximum_cached_frames, int)
            or not 1 <= maximum_cached_frames <= 128
            or isinstance(maximum_forward_frames, bool)
            or not isinstance(maximum_forward_frames, int)
            or not 0 <= maximum_forward_frames <= 120
        ):
            raise ValueError('invalid decoder locality limits')
        self.capture = capture
        self.process_frame = process_frame
        self.clock = clock
        self.maximum_cached_frames = maximum_cached_frames
        self.maximum_forward_frames = maximum_forward_frames
        self.frames: OrderedDict[int, np.ndarray] = OrderedDict()
        self.next_index: int | None = None
        self.seek_count = 0
        self.read_count = 0
        self.pixel_cache_hit_count = 0
        self.fallback_count = 0
        self.frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
        try:
            backend = capture.getBackendName()
        except (AttributeError, cv2.error):
            backend = ''
        self.enabled = bool(
            clock is not None and backend == 'FFMPEG'
            and math.isfinite(self.frame_rate) and self.frame_rate > 0
            and math.isclose(
                self.frame_rate, float(clock.frame_rate), rel_tol=1e-12
            )
            and capture.get(cv2.CAP_PROP_FRAME_COUNT) == clock.frame_count
        )

    def _disable(self) -> None:
        self.enabled = False
        self.next_index = None
        self.frames.clear()
        self.fallback_count += 1

    def _seek_read(self, timestamp_ms: float) -> np.ndarray | None:
        self.capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        self.seek_count += 1
        success, frame = self.capture.read()
        self.read_count += 1
        return self.process_frame(frame) if success else None

    def frame_at(self, seconds: float) -> np.ndarray | None:
        if not math.isfinite(seconds):
            raise ValueError('frame timestamp must be finite')
        timestamp_ms = max(0.0, float(seconds)) * 1000.0
        if not self.enabled:
            return self._seek_read(timestamp_ms)
        assert self.clock is not None
        # Match OpenCV FFmpeg POS_MSEC arithmetic, including the millisecond
        # round trip. Do not replace the caller's exact source-time identity.
        target = int((timestamp_ms / 1000.0) * self.frame_rate + 0.5)
        if not 0 <= target < self.clock.frame_count:
            return self._seek_read(timestamp_ms)
        if target in self.frames:
            self.pixel_cache_hit_count += 1
            self.frames.move_to_end(target)
            return self.frames[target]
        if self.next_index is not None and not math.isclose(
            self.capture.get(cv2.CAP_PROP_POS_FRAMES), self.next_index,
            rel_tol=0.0, abs_tol=1e-6,
        ):
            self._disable()
            return self._seek_read(timestamp_ms)
        if (
            self.next_index is None
            or not 0 <= target - self.next_index <= self.maximum_forward_frames
        ):
            self.capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
            self.seek_count += 1
            self.next_index = target
        while self.next_index <= target:
            frame_index = self.next_index
            success, frame = self.capture.read()
            self.read_count += 1
            if (
                not success
                or not math.isclose(
                    self.capture.get(cv2.CAP_PROP_POS_FRAMES), frame_index + 1,
                    rel_tol=0.0, abs_tol=1e-6,
                )
                or not math.isclose(
                    self.capture.get(cv2.CAP_PROP_POS_MSEC),
                    frame_index / self.frame_rate * 1000.0,
                    rel_tol=0.0, abs_tol=1e-3,
                )
            ):
                self._disable()
                return self._seek_read(timestamp_ms)
            self.frames[frame_index] = self.process_frame(frame)
            self.next_index += 1
            while len(self.frames) > self.maximum_cached_frames:
                self.frames.popitem(last=False)
        return self.frames[target]
