"""Rate-feasible source-clock witnesses for measured crop candidate lattices.

Nondecreasing frame IDs alone admit long freezes and arbitrarily fast jumps.
This resolver retains continuous feasible intervals, not only each frame's
center: repeated IDs at 60/30 fps can still represent a moving source clock.
It preserves measured geometry and all observations. This is target-side
diagnostic evidence, NOT a learned phase head, native-PTS verification, label
publication, playback-quality approval or permission to retime a final edit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .reference_crop_phase_lattice import CropPhaseCandidate, CropPhaseObservation
from .render import EditorialTransformKeyframe, EditorialTransformTrack


CONTRACT = "measured_crop_lattice_bounded_rate_continuous_clock_witness_v1"


class CropClockSearchLimitExceeded(RuntimeError):
    """Resource exhaustion is not proof that the lattice is infeasible."""


@dataclass(frozen=True, slots=True)
class CropClockBounds:
    source_frame_rate: float
    minimum_playback_rate: float
    maximum_playback_rate: float
    maximum_state_count: int = 32768
    maximum_extension_count: int = 250000

    def __post_init__(self):
        for value in (self.source_frame_rate, self.minimum_playback_rate,
                      self.maximum_playback_rate):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("crop source-clock bounds must be finite and positive")
        if self.minimum_playback_rate > self.maximum_playback_rate:
            raise ValueError("crop source-clock rate bounds are reversed")
        for value in (self.maximum_state_count, self.maximum_extension_count):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("crop source-clock work bounds must be positive integers")


@dataclass(frozen=True, slots=True)
class BoundedRateCropClockProposal:
    observations: tuple[CropPhaseObservation, ...]
    candidates: tuple[CropPhaseCandidate, ...]
    source_frame_positions: tuple[float, ...]
    bounds: CropClockBounds
    explored_extensions: int
    maximum_frontier_size: int
    contract: str = CONTRACT

    def __post_init__(self):
        if self.contract != CONTRACT or not isinstance(self.bounds, CropClockBounds):
            raise ValueError("crop source-clock proposal contract differs")
        if not self.observations or not len(self.observations) == len(self.candidates) == len(self.source_frame_positions):
            raise ValueError("crop source-clock proposal must retain every observation")
        if self.observations[0].time_seconds != 0 or any(
                b.time_seconds <= a.time_seconds for a, b in zip(self.observations, self.observations[1:])):
            raise ValueError("crop source-clock witness has an invalid observation clock")
        for count in (self.explored_extensions, self.maximum_frontier_size):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("crop source-clock work counters must be nonnegative integers")
        for row, candidate, position in zip(self.observations, self.candidates, self.source_frame_positions):
            if (candidate not in row.candidates or isinstance(position, bool)
                    or not isinstance(position, (int, float)) or not math.isfinite(position) or position < 0):
                raise ValueError("crop source-clock witness lacks its measured candidate")
            # Avoid ambiguous exact half-frame ties in either existing sampler.
            if round(position) != candidate.source_frame_index or math.floor(position + .5) != candidate.source_frame_index:
                raise ValueError("crop source-clock witness selects a different native frame")
        if any(not self.bounds.minimum_playback_rate - 1e-9 <= rate <= self.bounds.maximum_playback_rate + 1e-9
               for rate in self.playback_rates):
            raise ValueError("crop source-clock witness violates its declared rate bounds")
        EditorialTransformTrack(tuple(
            EditorialTransformKeyframe(row.time_seconds, *candidate.geometry)
            for row, candidate in zip(self.observations, self.candidates)),
            reason="measured_rate_feasible_crop_clock_geometry")

    @property
    def playback_rates(self) -> tuple[float, ...]:
        return tuple((right - left) / ((b.time_seconds - a.time_seconds) * self.bounds.source_frame_rate)
                     for a, b, left, right in zip(self.observations, self.observations[1:],
                                                  self.source_frame_positions, self.source_frame_positions[1:]))

    @property
    def mean_warped_structure_ncc(self) -> float:
        return sum(item.warped_structure_ncc for item in self.candidates) / len(self.candidates)

    def to_dict(self) -> dict:
        independent = sum(max(c.warped_structure_ncc for c in row.candidates)
                          for row in self.observations) / len(self.observations)
        return dict(contract=self.contract, source_frame_rate=self.bounds.source_frame_rate,
                    rate_bounds=[self.bounds.minimum_playback_rate, self.bounds.maximum_playback_rate],
                    source_frame_indices=[c.source_frame_index for c in self.candidates],
                    source_frame_positions=list(self.source_frame_positions),
                    observation_times=[row.time_seconds for row in self.observations],
                    residual_phase_frames=[position-row.canonical_frame_position
                                           for row, position in zip(self.observations, self.source_frame_positions)],
                    playback_rates=list(self.playback_rates),
                    measured_geometry=[list(c.geometry) for c in self.candidates],
                    mean_warped_structure_ncc=self.mean_warped_structure_ncc,
                    ncc_loss_against_independent_argmax=independent-self.mean_warped_structure_ncc,
                    explored_extensions=self.explored_extensions,
                    maximum_frontier_size=self.maximum_frontier_size,
                    clock_domain="nominal_cfr_frame_cells_not_native_pts_verified",
                    observations_removed=0, labels_modified=False,
                    retiming_authorized=False, model_input=False, quality_approved=False)


@dataclass(frozen=True, slots=True)
class _State:
    lower: float
    upper: float
    score: float
    variation: float
    absolute_phase: float
    path: tuple[CropPhaseCandidate, ...]
    previous: _State | None

    @property
    def rank(self):
        return (-self.score, self.variation, self.absolute_phase,
                tuple(c.source_frame_index for c in self.path))


def _cell(index: int) -> tuple[float, float]:
    # Interior cells guarantee parity across ties-to-even and ties-up samplers.
    margin = 32 * math.ulp(float(max(1, index)))
    lower, upper = max(0., index - .5 + margin), index + .5 - margin
    if lower >= upper:
        raise ValueError("source frame index exceeds floating-point clock precision")
    return lower, upper


def resolve_bounded_rate_source_clock(
    observations: Sequence[CropPhaseObservation], *, bounds: CropClockBounds,
) -> BoundedRateCropClockProposal | None:
    """Max-NCC measured path with a continuous clock inside explicit rate bounds.

    Interval-containing, better-ranked prefixes dominate weaker prefixes at
    the SAME terminal frame only. Keeping just the best score at that frame
    is incorrect: its feasible interval may not reach a later observation.
    Backtracking produces one feasible clock, not a least-acceleration fit.
    No constraints or observations are dropped on failure or search exhaustion.
    """
    rows = tuple(observations)
    if not rows or rows[0].time_seconds != 0 or not isinstance(bounds, CropClockBounds):
        raise ValueError("crop source clock needs bounds and a real time-zero observation")
    if any(b.time_seconds <= a.time_seconds or b.canonical_frame_position < a.canonical_frame_position
           for a, b in zip(rows, rows[1:])):
        raise ValueError("crop source clock needs forward, strictly timed observations")
    if any(not row.candidates for row in rows):
        return None
    for row in rows:
        for candidate in row.candidates:
            EditorialTransformKeyframe(row.time_seconds, *candidate.geometry)
    states = [_State(*_cell(c.source_frame_index), c.warped_structure_ncc, 0.,
                     abs(c.source_frame_index-rows[0].canonical_frame_position), (c,), None)
              for c in rows[0].candidates]
    if len(states) > bounds.maximum_state_count:
        raise CropClockSearchLimitExceeded("crop source-clock initial frontier exceeds budget")
    extensions, peak = 0, len(states)
    for previous_row, row in zip(rows, rows[1:]):
        dt = (row.time_seconds-previous_row.time_seconds) * bounds.source_frame_rate
        following = []
        for candidate in row.candidates:
            cell_lower, cell_upper = _cell(candidate.source_frame_index)
            phase = candidate.source_frame_index-row.canonical_frame_position
            feasible = []
            for state in states:
                extensions += 1
                if extensions > bounds.maximum_extension_count:
                    raise CropClockSearchLimitExceeded("crop source-clock extension budget exhausted")
                last = state.path[-1]
                lower = max(cell_lower, state.lower + dt*bounds.minimum_playback_rate)
                upper = min(cell_upper, state.upper + dt*bounds.maximum_playback_rate)
                if lower > upper:
                    continue
                try:
                    EditorialTransformTrack((EditorialTransformKeyframe(0., *last.geometry),
                        EditorialTransformKeyframe(row.time_seconds-previous_row.time_seconds, *candidate.geometry)),
                        reason="measured_source_clock_adjacent_geometry")
                except ValueError:
                    continue
                old_phase = last.source_frame_index-previous_row.canonical_frame_position
                feasible.append(_State(lower, upper, state.score+candidate.warped_structure_ncc,
                    state.variation+abs(phase-old_phase), state.absolute_phase+abs(phase),
                    state.path+(candidate,), state))
            # Ranking is additive and future feasibility only needs this cell's
            # reachable interval. Equal intervals therefore need one best prefix.
            retained = []
            for state in sorted(feasible, key=lambda item: item.rank):
                if any(other.lower <= state.lower and other.upper >= state.upper for other in retained):
                    continue
                retained.append(state)
            following.extend(retained)
            if len(following) > bounds.maximum_state_count:
                raise CropClockSearchLimitExceeded("crop source-clock interval frontier exceeds budget")
        if not following:
            return None
        states = following
        peak = max(peak, len(states))
    winner = min(states, key=lambda item: item.rank)
    positions = [0.] * len(rows)
    state = winner
    for index in reversed(range(len(rows))):
        lower, upper = state.lower, state.upper
        if index + 1 < len(rows):
            dt = (rows[index+1].time_seconds-rows[index].time_seconds)*bounds.source_frame_rate
            lower = max(lower, positions[index+1]-dt*bounds.maximum_playback_rate)
            upper = min(upper, positions[index+1]-dt*bounds.minimum_playback_rate)
        if lower > upper:
            if lower-upper > 16*math.ulp(max(1., abs(lower), abs(upper))):
                raise RuntimeError("crop clock backtracking lost a feasible interval")
            lower = upper  # Arithmetic-only overlap tolerance; final rates recheck.
        positions[index] = min(upper, max(lower, rows[index].canonical_frame_position))
        state = state.previous
    return BoundedRateCropClockProposal(rows, winner.path, tuple(positions), bounds, extensions, peak)


def phase_lattice_from_crop_payload(payload: Mapping[str, Any]) -> tuple[CropPhaseObservation, ...]:
    """Read existing accepted alternatives without changing immutable labels.

    Artifact content/provenance contracts are checked by the production loader's
    validator. This does not rehash or decode the original media. Every original
    teacher keyframe remains an observation, even if its best path fails later.
    A caller must explicitly label these results as unpublished diagnostics.
    """
    from .reference_dynamic_crop_supervision import validate_reference_dynamic_crop_supervision_payload
    from .reference_evidence_builder import ReferenceSourceClock
    from .reference_identity_crop import validate_identity_observation

    validate_reference_dynamic_crop_supervision_payload(payload, verify_input_files=False)
    if payload['supervision']['eligible'] is not True:
        raise ValueError("crop clock cannot promote a loss-masked artifact")
    source = payload['provenance']['source_media']
    clock = ReferenceSourceClock.from_mapping(source['clock'])
    frame_rate = source['frame_rate']
    samples = {}
    for sample in payload['samples']:
        key = sample['time_within_shot_seconds']
        if key in samples:
            raise ValueError("crop lattice contains duplicate observation times")
        samples[key] = sample
    rows = []
    for keyframe in payload['supervision']['renderer_keyframes']:
        sample = samples[keyframe['time_within_shot_seconds']]
        if sample['accepted'] is not True or sample['renderer_keyframe'] != keyframe:
            raise ValueError("crop lattice differs from its retained teacher observation")
        accepted = []
        for value in sample['audit']['local_visual_search']['candidates']:
            if value['accepted'] is not True:
                continue
            if value.get('rejection_reason') is not None or value.get('decode_attempted') is not True:
                raise ValueError("accepted crop candidate lacks a decoded measurement")
            diagnostics = value['diagnostics']
            projection = diagnostics.get('renderer_safe_zoom_projection', {})
            geometry = (projection['projected_target'] if projection.get('applied') is True
                        else diagnostics['estimated_renderer_transform'])
            validate_identity_observation(diagnostics, geometry,
                payload['configuration'].get('verified_identity_observation_policy'))
            candidate = CropPhaseCandidate(value['source_frame_index'],
                diagnostics['actual_warp_review']['warped_structure_ncc'],
                tuple(geometry[name] for name in ('zoom', 'center_x', 'center_y', 'roll_degrees')))
            EditorialTransformKeyframe(sample['time_within_shot_seconds'], *candidate.geometry)
            accepted.append(candidate)
        chosen = next((c for c in accepted if c.source_frame_index == keyframe['source_frame_index']), None)
        if chosen is None or chosen.geometry != tuple(keyframe[name] for name in ('zoom', 'center_x', 'center_y', 'roll_degrees')):
            raise ValueError("crop lattice lost or changed the measured selected geometry")
        rows.append(CropPhaseObservation(sample['time_within_shot_seconds'],
            clock.master_to_source(sample['master_seconds']) * frame_rate, tuple(accepted)))
    return tuple(rows)
