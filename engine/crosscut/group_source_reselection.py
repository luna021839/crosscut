"""Complete incumbent source-run context for bounded local GROUP reselection.

Source clocks/cameras and decoder observations remain owned by the caller.
These windows do not certify a pose connection, authorize output or prove a
continuous broadcaster camera. Source reuse/cadence is distinct from camera
cuts inside one broadcast source.
"""
import math


def source_reselection_windows(choices, mutable_segment_indices):
    rows = tuple(choices)
    indices = tuple(mutable_segment_indices)
    if (not rows or any(type(index) is not int or not 0 <= index < len(rows) for index in indices)
            or indices != tuple(sorted(set(indices)))):
        raise ValueError('local source reselection needs ordered unique indices in a complete timeline')
    for index, row in enumerate(rows):
        if (row.segment_index != index or type(row.source_index) is not int or row.source_index < 0
                or not math.isfinite(row.canonical_master_start) or not math.isfinite(row.canonical_master_end)
                or row.canonical_master_end <= row.canonical_master_start or index and not math.isclose(
                    rows[index-1].canonical_master_end, row.canonical_master_start, abs_tol=1e-6)):
            raise ValueError('local source reselection requires exact contiguous canonical spans')
    runs = []
    for index in indices:
        if runs and index == runs[-1][-1]+1:
            runs[-1].append(index)
        else:
            runs.append([index])
    windows = []
    for run in runs:
        left, right = max(0, run[0]-1), min(len(rows)-1, run[-1]+1)
        # A newly selected source can join the fixed right/left camera source.
        # One neighbor cell hides the resulting total source hold duration.
        while left > 0 and rows[left-1].source_index == rows[left].source_index:
            left -= 1
        while right+1 < len(rows) and rows[right+1].source_index == rows[right].source_index:
            right += 1
        if windows and left <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(right, windows[-1][1]))
        else:
            windows.append((left, right))
    return tuple(windows)
