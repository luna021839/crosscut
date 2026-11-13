"""Transfer reference times through the SAME physical source, not a grid snap.

This only recovers a clock coordinate. It neither transfers face/body samples
nor converts an editor's source choice into a human hold-versus-cut preference.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re

import numpy as np

from .grid_perception import file_identity


def _points(rows):
    values = tuple((float(r['master_seconds']), float(r['source_seconds'])) for r in rows)
    if (len(values) < 2 or not all(math.isfinite(x) for row in values for x in row)
            or any(b[0] <= a[0] or b[1] <= a[1] for a, b in zip(values, values[1:]))):
        raise ValueError('two finite strictly monotonic clock anchors are required')
    return values


def _bounded_interpolate(value, points, source_axis):
    coordinates = tuple(p[source_axis] for p in points)
    if not math.isfinite(value) or not coordinates[0] <= value <= coordinates[-1]:
        raise ValueError('reference clock bridge cannot extrapolate beyond measured anchors')
    return float(np.interp(value, coordinates, tuple(p[1-source_axis] for p in points)))


@dataclass(frozen=True)
class SharedContentClockBridge:
    old_points: tuple
    current_points: tuple
    content_sha256: str
    source_file_identity: tuple

    @classmethod
    def from_source_file(cls, path, *, expected_sha256, old_points, current_points, check=lambda: None):
        """Hash a real common asset once; a matching video ID is insufficient."""
        if not isinstance(expected_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
            raise ValueError('the archived physical source digest is required')
        old, current = _points(old_points), _points(current_points)
        path = Path(path)
        check()
        identity = file_identity(path)
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                check()
                hasher.update(chunk)
        check()
        if file_identity(path) != identity or hasher.hexdigest() != expected_sha256:
            raise ValueError('shared physical reference asset differs; clocks cannot be transferred')
        return cls(old, current, expected_sha256, identity)

    def transfer(self, old_master_seconds):
        physical = _bounded_interpolate(float(old_master_seconds), self.old_points, 0)
        current = _bounded_interpolate(physical, self.current_points, 1)
        return current, physical

    def output_event(self, old_master_seconds, *, master_start, fps, frame_count):
        if (not math.isfinite(master_start) or not math.isfinite(fps) or fps <= 0
                or type(frame_count) is not int or frame_count < 2):
            raise ValueError('a real output clock is required')
        current, physical = self.transfer(old_master_seconds)
        frame = (current-master_start)*fps
        ordinal = round(frame)
        if not 0 < ordinal < frame_count:
            raise ValueError('reference event is outside the current movie interior')
        return dict(old_master_seconds=float(old_master_seconds), shared_source_seconds=physical,
            current_master_seconds=current, output_frame_float=frame, output_frame=ordinal,
            quantization_frames=ordinal-frame, quantization_master_seconds=(ordinal-frame)/fps,
            exact_event_observations_required=True, human_av_preference_created=False)
