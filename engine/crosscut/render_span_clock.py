"""Event clocks for the affine source interval actually consumed by FFmpeg."""
from __future__ import annotations

import math
from typing import Sequence


def rendered_internal_cut_master_times(
    source_cut_times: Sequence[float], *, source_start: float, source_end: float,
    master_start: float, master_end: float,
) -> tuple[float, ...]:
    """Map native cuts using the selected span's two checked source endpoints.

    A piecewise alignment curve chooses those endpoints. Rendering one segment
    uses their single playback rate, not the inverse of that entire curve.
    Calling source_to_master for an interior event therefore describes a
    different clock when the segment crosses an alignment control point.
    """
    values = (source_start, source_end, master_start, master_end, *source_cut_times)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in values):
        raise ValueError("render span event clocks must be finite numbers")
    if source_end <= source_start or master_end <= master_start:
        raise ValueError("render span event clocks require positive durations")
    rate = (source_end - source_start) / (master_end - master_start)
    return tuple(master_start + (cut - source_start) / rate
                 for cut in sorted(set(source_cut_times))
                 if source_start + 1e-6 < cut < source_end - 1e-6)
