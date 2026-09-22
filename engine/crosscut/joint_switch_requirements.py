"""Exact section switch counts carried by the joint selector, not a QA waiver."""
from dataclasses import dataclass
import math
from time import monotonic
from typing import Mapping, Sequence

from .timeline import NoCompleteTimelinePath


MAXIMUM_SOURCE_SWITCH_CAPACITY_STATES = 50_000
MAXIMUM_SOURCE_SWITCH_EXPANSIONS = 2_000_000


@dataclass(slots=True)
class SourceSwitchSearchBudget:
    """Bound intermediate work too, not only the final Pareto frontier."""

    deadline_monotonic_seconds: float | None = None
    maximum_states: int = MAXIMUM_SOURCE_SWITCH_CAPACITY_STATES
    maximum_expansions: int = MAXIMUM_SOURCE_SWITCH_EXPANSIONS
    expansions: int = 0
    peak_states: int = 0

    def __post_init__(self):
        for limit in (self.maximum_states, self.maximum_expansions):
            if type(limit) is not int or limit < 1:
                raise ValueError('source-switch search limits must be positive integers')
        deadline = self.deadline_monotonic_seconds
        if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline)):
            raise ValueError('source-switch deadline must be finite')
        self.check()

    def check(self, *, states: int = 0, segment_index: int | None = None):
        self.expansions += 1
        self.peak_states = max(self.peak_states, states)
        reason = (
            'section_source_switch_wall_clock_budget_exhausted'
            if self.deadline_monotonic_seconds is not None and monotonic() >= self.deadline_monotonic_seconds
            else 'section_source_switch_capacity_state_budget_exhausted'
            if states > self.maximum_states
            else 'section_source_switch_work_budget_exhausted'
            if self.expansions > self.maximum_expansions
            else None
        )
        if reason:
            raise NoCompleteTimelinePath(
                'source-switch search exceeded its bounded descriptor budget',
                reason_code=reason, segment_index=segment_index,
                rejection_counts={reason: 1, 'proof_materialization_call_count': 0,
                                  'capacity_state_count': states,
                                  'descriptor_expansion_count': self.expansions},
            )


def quantized_boundary_seconds(
    *,
    origin_seconds: float,
    nominal_boundary_seconds: float,
    fps: float,
    frame_offset: int,
) -> float:
    """Map one concrete boundary offset onto the output frame clock."""

    tick = round(
        (nominal_boundary_seconds - origin_seconds) * fps + frame_offset
    )
    return origin_seconds + tick / fps


def source_switch_increment_vectors(
    requirements: Sequence["SourceSwitchRequirement"],
    *,
    origin_seconds: float,
    nominal_boundary_seconds: float,
    fps: float,
    maximum_offset_frames: int,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate requirement increments reachable by one concrete offset.

    Returning concrete vectors is important near adjacent section windows: a
    nominal boundary may reach the left window at ``-1`` and the right window
    at ``+1``, but no single cut is allowed to increment both.
    """

    if any(
        not isinstance(requirement, SourceSwitchRequirement)
        for requirement in requirements
    ):
        raise TypeError("source-switch increments require typed windows")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or float(fps) <= 0.0
    ):
        raise ValueError("source-switch fps must be positive and finite")
    if (
        isinstance(maximum_offset_frames, bool)
        or not isinstance(maximum_offset_frames, int)
        or maximum_offset_frames < 0
    ):
        raise ValueError(
            "source-switch maximum offset must be a nonnegative integer"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in (origin_seconds, nominal_boundary_seconds)
    ):
        raise ValueError("source-switch boundary clock must be finite")

    vectors = {
        tuple(
            int(
                requirement.contains(
                    quantized_boundary_seconds(
                        origin_seconds=origin_seconds,
                        nominal_boundary_seconds=nominal_boundary_seconds,
                        fps=fps,
                        frame_offset=frame_offset,
                    )
                )
            )
            for requirement in requirements
        )
        for frame_offset in range(
            -maximum_offset_frames,
            maximum_offset_frames + 1,
        )
    }
    return tuple(sorted(vectors))


def _pareto_capacity_states(states, budget):
    """Drop count vectors dominated for the same current candidate."""

    by_candidate = {}
    for candidate, counts in states:
        by_candidate.setdefault(candidate, set()).add(counts)
    retained = set()
    for candidate, count_vectors in by_candidate.items():
        frontier = []
        for counts in sorted(
            count_vectors,
            key=lambda item: (sum(item), item),
            reverse=True,
        ):
            budget.check(states=len(states))
            if any(
                _dominates(kept, counts, budget)
                for kept in frontier
            ):
                continue
            frontier.append(counts)
        retained.update((candidate, counts) for counts in frontier)
    return retained


def _dominates(left, right, budget):
    budget.check()
    return all(a >= b for a, b in zip(left, right))


@dataclass(frozen=True, slots=True)
class SourceSwitchRequirement:
    start_seconds: float
    end_seconds: float
    minimum_count: int

    def __post_init__(self):
        if (any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)
                for v in (self.start_seconds,self.end_seconds))
                or not 0 <= self.start_seconds < self.end_seconds
                or type(self.minimum_count) is not int or self.minimum_count < 1):
            raise ValueError('source-switch requirement needs a finite window and positive count')

    def contains(self, seconds):
        # A cut exactly on a section's start has no preceding run overlapping
        # that section, so the decoded montage reviewer does not count it.
        return self.start_seconds + 1e-9 < seconds < self.end_seconds - 1e-9


class SourceSwitchCounter:
    def __init__(self, requirements: Sequence[SourceSwitchRequirement], lattice,
                 *, fps: float, maximum_offset_frames: int, domain=None,
                 source_families: Mapping[str,str] | None = None,
                 budget: SourceSwitchSearchBudget | None = None):
        self.budget = budget or SourceSwitchSearchBudget()
        self._increment_vectors = {}
        self.requirements=tuple(requirements)
        if len(self.requirements)>32 or any(not isinstance(w,SourceSwitchRequirement) for w in self.requirements):
            raise ValueError('source-switch requirements must be at most 32 typed windows')
        if any(a.end_seconds>b.start_seconds for a,b in zip(self.requirements,self.requirements[1:])):
            raise ValueError('source-switch windows must be ordered and non-overlapping')
        self.lattice=tuple(tuple(row) for row in lattice)
        if not self.lattice or any(not row for row in self.lattice):
            raise ValueError('source-switch lattice must contain nonempty rows')
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isfinite(float(fps))
            or float(fps) <= 0.0
        ):
            raise ValueError('source-switch fps must be positive and finite')
        if (
            isinstance(maximum_offset_frames, bool)
            or not isinstance(maximum_offset_frames, int)
            or maximum_offset_frames < 0
        ):
            raise ValueError('source-switch maximum offset must be a nonnegative integer')
        self.fps=float(fps)
        self.origin=self.lattice[0][0].start_seconds
        self.maximum_offset_frames=maximum_offset_frames
        self.families=dict(source_families or {})
        ids={c.source_id for row in lattice for c in row}
        if self.families and (not ids.issubset(self.families) or any(not isinstance(v,str) or not v for v in self.families.values())):
            raise ValueError('source families must bind every current source exactly')
        self.initial_counts=(0,)*len(self.requirements)
        # Count potential boundary positions, not source pairs. This is only
        # an optimistic reachability bound; every actual edge still needs proof.
        allowed_positions=(None if domain is None else {k.boundary_index for k in domain
            if self.different(k.outgoing.source_id,k.incoming.source_id)})
        potential=[]
        for boundary,(left,right) in enumerate(zip(lattice,lattice[1:])):
            self.budget.check(segment_index=right[0].segment_index)
            has_pair=(boundary in allowed_positions if allowed_positions is not None else
                      any(self.different(a.source_id,b.source_id) for a in left for b in right))
            vectors = (
                source_switch_increment_vectors(
                    self.requirements,
                    origin_seconds=self.origin,
                    nominal_boundary_seconds=right[0].start_seconds,
                    fps=self.fps,
                    maximum_offset_frames=self.maximum_offset_frames,
                )
                if has_pair
                else (self.initial_counts,)
            )
            potential.append(tuple(max(vector[index] for vector in vectors)
                                   for index in range(len(self.requirements))))
        remaining=[self.initial_counts]*(len(potential)+1)
        for i in range(len(potential)-1,-1,-1):
            remaining[i]=tuple(a+b for a,b in zip(remaining[i+1],potential[i]))
        self.remaining=tuple(remaining)
        connected, blocking_segment_index = self._connected_capacity(domain)
        if not self.feasible(self.initial_counts,0) or not connected:
            raise NoCompleteTimelinePath('search domain cannot supply the required section source switches',
                reason_code='section_source_switch_capacity_insufficient',
                segment_index=blocking_segment_index,
                rejection_counts={'section_source_switch_capacity_insufficient':1,
                                  'connected_source_path_required':1,
                                  'proof_materialization_call_count':0})

    def potential_increment_vectors(self, boundary, left, right):
        if (
            isinstance(boundary, bool)
            or not isinstance(boundary, int)
            or not 0 <= boundary < len(self.lattice) - 1
            or left not in self.lattice[boundary]
            or right not in self.lattice[boundary + 1]
        ):
            raise ValueError("source-switch boundary candidates do not match lattice")
        if not self.different(left.source_id,right.source_id):
            return (self.initial_counts,)
        cache_key = (boundary, right.start_seconds)
        if cache_key not in self._increment_vectors:
            self.budget.check(segment_index=right.segment_index)
            self._increment_vectors[cache_key] = source_switch_increment_vectors(
            self.requirements,
            origin_seconds=self.origin,
            nominal_boundary_seconds=right.start_seconds,
            fps=self.fps,
            maximum_offset_frames=self.maximum_offset_frames,
            )
        return self._increment_vectors[cache_key]

    def _connected_capacity(self,domain):
        """Prove one connected source/candidate path can meet every count.

        This remains an optimistic descriptor-domain check: a potential switch
        may use any allowed frame offset.  Unlike the previous position count,
        it cannot combine mutually disconnected A-to-B edges from unrelated
        paths into a fictional capacity proof.
        """
        if not self.requirements:
            return True,None
        allowed=(None if domain is None else {
            (key.boundary_index,key.outgoing,key.incoming) for key in domain
        })
        states={(candidate,self.initial_counts) for candidate in self.lattice[0]}
        self.budget.check(states=len(states))
        potential_boundaries=[[] for _ in self.requirements]
        for boundary,(left_row,right_row) in enumerate(zip(self.lattice,self.lattice[1:])):
            next_states=set()
            for left,counts in states:
                for right in right_row:
                    self.budget.check(states=len(next_states), segment_index=right.segment_index)
                    if allowed is not None and (boundary,left,right) not in allowed:
                        continue
                    for increments in self.potential_increment_vectors(
                        boundary, left, right
                    ):
                        next_counts=tuple(min(requirement.minimum_count,count+increment)
                            for requirement,count,increment in zip(self.requirements,counts,increments))
                        if self.feasible(next_counts, boundary + 1):
                            next_states.add((right,next_counts))
            for index,requirement in enumerate(self.requirements):
                if any(any(vector[index] for vector in
                           self.potential_increment_vectors(boundary,left,right))
                       for left in left_row for right in right_row
                       if allowed is None or (boundary,left,right) in allowed):
                    potential_boundaries[index].append(boundary)
            if not next_states:
                return False,right_row[0].segment_index
            self.budget.check(states=len(next_states), segment_index=right_row[0].segment_index)
            states=_pareto_capacity_states(next_states, self.budget)
            if len(states) > MAXIMUM_SOURCE_SWITCH_CAPACITY_STATES:
                raise NoCompleteTimelinePath(
                    'source-switch capacity search exceeded its deterministic state budget',
                    reason_code='section_source_switch_capacity_state_budget_exhausted',
                    segment_index=right_row[0].segment_index,
                    rejection_counts={
                        'section_source_switch_capacity_state_budget_exhausted':1,
                        'capacity_state_count':len(states),
                        'capacity_state_limit':MAXIMUM_SOURCE_SWITCH_CAPACITY_STATES,
                        'proof_materialization_call_count':0,
                    },
                )
        if any(all(count>=requirement.minimum_count
                   for count,requirement in zip(counts,self.requirements))
               for _candidate,counts in states):
            return True,None
        best=max((counts for _candidate,counts in states),default=self.initial_counts,
                 key=lambda counts:(sum(counts),counts))
        for index,(count,requirement) in enumerate(zip(best,self.requirements)):
            if count>=requirement.minimum_count:
                continue
            boundaries=potential_boundaries[index]
            target=min(len(boundaries)-1,requirement.minimum_count-1)
            if boundaries:
                return False,self.lattice[boundaries[target]+1][0].segment_index
        return False,self.lattice[-1][0].segment_index

    def different(self,left,right):
        return self.families.get(left,left)!=self.families.get(right,right)

    def advance(self,counts,*,left,right,nominal_boundary_seconds,frame_offset):
        if (
            isinstance(frame_offset, bool)
            or not isinstance(frame_offset, int)
            or abs(frame_offset) > self.maximum_offset_frames
        ):
            raise ValueError('source-switch frame offset is outside the exact domain')
        if (
            isinstance(nominal_boundary_seconds, bool)
            or not isinstance(nominal_boundary_seconds, (int, float))
            or not math.isfinite(float(nominal_boundary_seconds))
        ):
            raise ValueError('source-switch nominal boundary must be finite')
        if not self.different(left,right):return counts
        seconds=quantized_boundary_seconds(
            origin_seconds=self.origin,
            nominal_boundary_seconds=nominal_boundary_seconds,
            fps=self.fps,
            frame_offset=frame_offset,
        )
        return tuple(min(w.minimum_count,count+int(w.contains(seconds)))
                     for w,count in zip(self.requirements,counts))

    def feasible(self,counts,next_boundary):
        return all(count+remaining>=w.minimum_count for w,count,remaining in
                   zip(self.requirements,counts,self.remaining[next_boundary]))
