"""Joint clock/geometry proposals from already measured crop candidates.

An independent per-image argmax can move backwards through a source or jump
between incompatible camera shots. This resolver tests one CONSTANT residual
source phase over the existing bound source clock. It does not shift the
reference observations, infer new geometry, drop difficult observations, or
claim to be a learned phase head. Failure proves only that this particular
constant-phase family cannot cover the supplied accepted candidate lattice.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


CONTRACT = "all_observations_joint_constant_source_phase_crop_lattice_v1"


@dataclass(frozen=True, slots=True)
class CropPhaseCandidate:
    source_frame_index: int
    warped_structure_ncc: float
    geometry: tuple[float, float, float, float]

    def __post_init__(self):
        if isinstance(self.source_frame_index, bool) or not isinstance(self.source_frame_index, int) or self.source_frame_index < 0:
            raise ValueError("crop phase candidate needs a real source frame")
        if not math.isfinite(self.warped_structure_ncc) or not -1 <= self.warped_structure_ncc <= 1:
            raise ValueError("crop phase candidate needs measured NCC")
        if len(self.geometry) != 4 or any(not math.isfinite(v) for v in self.geometry):
            raise ValueError("crop phase candidate needs measured geometry")


@dataclass(frozen=True, slots=True)
class CropPhaseObservation:
    time_seconds: float
    canonical_frame_position: float
    candidates: tuple[CropPhaseCandidate, ...]

    def __post_init__(self):
        if not math.isfinite(self.time_seconds) or self.time_seconds < 0 or not math.isfinite(self.canonical_frame_position):
            raise ValueError("crop phase observation clock is invalid")
        if any(not isinstance(c, CropPhaseCandidate) for c in self.candidates):
            raise TypeError("crop phase candidates are invalid")
        if len({c.source_frame_index for c in self.candidates}) != len(self.candidates):
            raise ValueError("one observation cannot duplicate candidate frames")


@dataclass(frozen=True, slots=True)
class JointCropPhaseProposal:
    phase_frames: float
    phase_interval: tuple[float, float]
    candidates: tuple[CropPhaseCandidate, ...]
    mean_warped_structure_ncc: float
    mean_independent_argmax_ncc: float
    feasible_interval_count: int
    contract: str = CONTRACT

    @property
    def ncc_loss_against_independent_argmax(self) -> float:
        return self.mean_independent_argmax_ncc - self.mean_warped_structure_ncc


def resolve_constant_source_phase(
    observations: Sequence[CropPhaseObservation],
) -> JointCropPhaseProposal | None:
    """Find the exact best accepted lattice path for a constant source phase.

    Candidate n at canonical real-valued position x owns the half-open phase
    interval [n-.5-x, n+.5-x). All such endpoints partition phase space into
    atoms. A midpoint of every nonempty atom exhausts every attainable path;
    evaluate actual floating-point rounding as a second check. Ties prefer
    smaller absolute residual phase, then lower phase, not a source identity.
    No extrapolation into unsearched/rejected frames is permitted.
    """
    rows = tuple(observations)
    if not rows:
        raise ValueError("joint crop phase requires observations")
    if any(b.time_seconds <= a.time_seconds or b.canonical_frame_position < a.canonical_frame_position
           for a, b in zip(rows, rows[1:])):
        raise ValueError("joint crop phase needs forward, strictly timed observations")
    if any(not row.candidates for row in rows):
        return None
    lookups = [{c.source_frame_index: c for c in row.candidates} for row in rows]
    endpoints = sorted({c.source_frame_index + side - row.canonical_frame_position
                        for row in rows for c in row.candidates for side in (-.5, .5)})
    independent = sum(max(c.warped_structure_ncc for c in row.candidates) for row in rows) / len(rows)
    feasible = []
    for lower, upper in zip(endpoints, endpoints[1:]):
        phase = lower + (upper - lower) / 2
        if not lower < phase < upper:
            continue  # No representable interior; do not promise tie parity.
        indices = tuple(math.floor(row.canonical_frame_position + phase + .5) for row in rows)
        if any(index not in lookup for index, lookup in zip(indices, lookups)):
            continue
        selected = tuple(lookup[index] for index, lookup in zip(indices, lookups))
        score = sum(c.warped_structure_ncc for c in selected) / len(rows)
        feasible.append((score, phase, lower, upper, selected))
    if not feasible:
        return None
    best = min(feasible, key=lambda item: (-item[0], abs(item[1]), item[1]))
    score, phase, lower, upper, selected = best
    return JointCropPhaseProposal(phase, (lower, upper), selected, score, independent, len(feasible))
