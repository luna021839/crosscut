"""Isolated native-raster reader; production containment is not yet switched.

Keep one original frame per capture. The existing descriptor sampler and its
feature fingerprint are untouched. H264 admission requires a complete packet
presentation grid as well as the reader's per-decode position/PTS checks.
"""
from fractions import Fraction
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .seek_locality import ConstantFrameClock, SeekLocalityReader, constant_frame_clock_from_probe


def native_constant_frame_clock(payload: Mapping[str, Any]) -> ConstantFrameClock | None:
    existing = constant_frame_clock_from_probe(payload)
    if existing is not None:
        return existing
    try:
        streams, packets = payload['streams'], payload['packets']
        if not isinstance(streams, list) or len(streams) != 1 or not isinstance(packets, list):
            return None
        stream = streams[0]
        if stream['codec_name'] != 'h264':
            return None
        rate = Fraction(stream['avg_frame_rate'])
        time_base = Fraction(stream['time_base'])
        if (rate <= 0 or time_base <= 0 or rate != Fraction(stream['r_frame_rate'])
                or Fraction(str(stream['start_time'])) != 0):
            return None
        raw_count = stream['nb_frames']
        if isinstance(raw_count, bool) or not isinstance(raw_count, (int, str)):
            return None
        count = int(raw_count)
        stride = 1/rate/time_base
        if not 2 <= count <= 1_000_000 or len(packets) != count or stride.denominator != 1:
            return None
        points = []
        for packet in packets:
            pts, duration = packet['pts'], packet['duration']
            if (isinstance(pts, bool) or not isinstance(pts, int)
                    or isinstance(duration, bool) or not isinstance(duration, int)
                    or duration != stride.numerator):
                return None
            points.append(pts)
        if sorted(points) != [i*stride.numerator for i in range(count)]:
            return None
        return ConstantFrameClock(rate, count)
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None


def probe_native_constant_frame_clock(source: Path) -> ConstantFrameClock | None:
    try:
        before = source.stat()
        if not source.is_file():
            return None
        output = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_streams', '-show_packets', '-show_entries',
            'stream=codec_name,avg_frame_rate,r_frame_rate,time_base,start_time,nb_frames:packet=pts,duration',
            '-of', 'json', str(source),
        ], capture_output=True, check=True, timeout=10.)
        after = source.stat()
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) != (
                after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
            return None
        return native_constant_frame_clock(json.loads(output.stdout))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def native_locality_reader(capture: Any, source: Path) -> SeekLocalityReader:
    """Capture remains caller-owned; one 4K frame is bounded, not a frame bank."""
    return SeekLocalityReader(
        capture, clock=probe_native_constant_frame_clock(source),
        process_frame=lambda frame: frame, maximum_cached_frames=1,
    )
