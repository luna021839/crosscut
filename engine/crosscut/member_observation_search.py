"""Bounded proposal/observation coordination, separate from render authority.

The path solver owns quotas, duration, cadence and effect allocation. This
coordinator owns only which exact transitions are observed, and never turns an
unobserved transition into either visual approval or an infeasibility proof.
It returns an *observed proposal*, not DirectMemberVisualSelection. Full camera,
offset/handle, effect and decoded-output validation remain required downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Callable, Generic, Sequence, TypeVar


Evidence = TypeVar("Evidence")


@dataclass(frozen=True, slots=True)
class TransitionDemand:
    boundary_index: int
    outgoing_source_index: int
    incoming_source_index: int

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (self.boundary_index, self.outgoing_source_index, self.incoming_source_index)):
            raise ValueError("transition indices must be nonnegative integers")


class ObservationState(Enum):
    UNOBSERVED = "unobserved"
    CONTENT_INELIGIBLE = "content_ineligible"
    ELIGIBLE = "observed_eligible"
    REJECTED = "observed_rejected"


@dataclass(frozen=True, slots=True)
class TransitionObservation(Generic[Evidence]):
    demand: TransitionDemand
    eligible: bool
    cost: float
    evidence: Evidence

    def __post_init__(self) -> None:
        if type(self.eligible) is not bool:
            raise ValueError("observed eligibility must be boolean")
        if (isinstance(self.cost, bool) or not isinstance(self.cost, (int, float))
                or not math.isfinite(self.cost)):
            raise ValueError("observed cost must be finite")
        if self.evidence is None:
            raise ValueError("an observed transition needs its actual evidence")


@dataclass(frozen=True, slots=True)
class ProposalEdge:
    """Search priority, deliberately NOT a measured DirectBoundaryMismatch.

    Known eligible edges have a small monotonic cost, unobserved edges cost 1.
    That keeps verified progress useful during repair instead of abandoning
    every measured edge for a fresh optimistic zero-cost alternative. This is
    an incumbent-first priority, NOT an admissible quality bound or a claim of
    globally optimal editing. The final optimizer must use measured costs.
    """
    state: ObservationState
    cost: float

    @property
    def severe_flags(self) -> tuple[str, ...]:
        # Adapter to the existing bounded path MILP; no sample/geometry fields.
        if self.state is ObservationState.CONTENT_INELIGIBLE:
            return ("content_or_cross_source_policy_ineligible",)
        return ("observed_transition_rejected",) if self.state is ObservationState.REJECTED else ()


ProposalGraph = tuple[tuple[tuple[ProposalEdge, ...], ...], ...]


@dataclass(frozen=True, slots=True)
class ObservedPathProposal(Generic[Evidence]):
    source_indices: tuple[int, ...]
    transitions: tuple[TransitionObservation[Evidence], ...]
    solver_call_count: int
    observation_count: int
    unobserved_transition_count: int


class MemberObservationSearchIncomplete(RuntimeError):
    """Not visual infeasibility; do not restart another musical partition."""

    def __init__(self, reason: str, *, solver_calls: int, observations: int, unobserved: int):
        super().__init__(reason)
        self.diagnostics = {
            "failure": "member_observation_search_incomplete",
            "reason": reason,
            "solver_call_count": solver_calls,
            "observation_count": observations,
            "unobserved_transition_count": unobserved,
            "constraint_infeasibility_proven": False,
            "retry_as_another_partition": False,
            "render_authorized": False,
        }


class MemberObservationSearch(Generic[Evidence]):
    """Run-local demand ledger with a finite progress and work contract.

    The current proposal adapter is cross-source only. Do not enable it as a
    replacement for the production selector until its policy adapter covers
    the existing hold/cinematic and joint-offset contracts too. In particular,
    `propose` returning None cannot prove that those other policies are infeasible.
    Provider/SSD/cancellation errors escape directly, without a retry loop.
    """
    def __init__(
        self,
        candidate_validity: Sequence[Sequence[bool]],
        *,
        maximum_observations: int,
        maximum_solver_calls: int,
    ) -> None:
        self.validity = tuple(tuple(row) for row in candidate_validity)
        if len(self.validity) < 2 or len(self.validity[0]) < 2:
            raise ValueError("observation search needs two spans and sources")
        self.source_count = len(self.validity[0])
        if any(len(row) != self.source_count or any(type(value) is not bool for value in row)
               for row in self.validity):
            raise ValueError("content validity must be a rectangular boolean domain")
        for value in (maximum_observations, maximum_solver_calls):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("observation/solver budgets must be positive integers")
        self.maximum_observations = maximum_observations
        self.maximum_solver_calls = maximum_solver_calls
        self._domain = frozenset(
            TransitionDemand(boundary, left, right)
            for boundary in range(len(self.validity)-1)
            for left in range(self.source_count)
            for right in range(self.source_count)
            if left != right and self.validity[boundary][left] and self.validity[boundary+1][right]
        )
        self._observed: dict[TransitionDemand, TransitionObservation[Evidence]] = {}
        self._solver_calls = 0
        self._observation_calls = 0

    def proposal_graph(self) -> ProposalGraph:
        def edge(demand: TransitionDemand) -> ProposalEdge:
            observed = self._observed.get(demand)
            if demand not in self._domain:
                return ProposalEdge(ObservationState.CONTENT_INELIGIBLE, 1.0)
            if observed is not None and not observed.eligible:
                return ProposalEdge(ObservationState.REJECTED, 1.0)
            if observed is None:
                return ProposalEdge(ObservationState.UNOBSERVED, 1.0)
            priority = .25 * (.5 + math.atan(observed.cost) / math.pi)
            return ProposalEdge(ObservationState.ELIGIBLE, priority)
        return tuple(tuple(tuple(edge(TransitionDemand(boundary, left, right))
            for right in range(self.source_count)) for left in range(self.source_count))
            for boundary in range(len(self.validity)-1))

    def _incomplete(self, reason: str) -> MemberObservationSearchIncomplete:
        return MemberObservationSearchIncomplete(reason, solver_calls=self._solver_calls,
            observations=self._observation_calls, unobserved=len(self._domain)-len(self._observed))

    def run(
        self,
        *,
        propose: Callable[[ProposalGraph], Sequence[int] | None],
        observe: Callable[[TransitionDemand], TransitionObservation[Evidence]],
    ) -> ObservedPathProposal[Evidence]:
        while self._solver_calls < self.maximum_solver_calls:
            self._solver_calls += 1
            proposed = propose(self.proposal_graph())
            if proposed is None:
                raise self._incomplete("proposal_solver_returned_no_path")
            path = tuple(proposed)
            if (len(path) != len(self.validity)
                    or any(isinstance(source, bool) or not isinstance(source, int)
                           or not 0 <= source < self.source_count for source in path)):
                raise ValueError("proposal has invalid source indices or length")
            demands = tuple(TransitionDemand(index, left, right)
                for index, (left, right) in enumerate(zip(path, path[1:])))
            if any(demand not in self._domain for demand in demands):
                raise ValueError("proposal violates content validity or cross-source policy")
            if any(demand in self._observed and not self._observed[demand].eligible for demand in demands):
                raise ValueError("proposal repeats an observed rejected transition")
            rejected = False
            for demand in demands:
                if demand in self._observed:
                    continue
                if self._observation_calls >= self.maximum_observations:
                    raise self._incomplete("exact_observation_budget_exhausted")
                # Charge before the call so external failures do not reset work
                # accounting if a caller deliberately resumes this same ledger.
                self._observation_calls += 1
                observed = observe(demand)
                if not isinstance(observed, TransitionObservation) or observed.demand != demand:
                    raise ValueError("observer returned evidence for a different transition")
                self._observed[demand] = observed
                if not observed.eligible:
                    rejected = True
                    break  # Repair this path before observing its unused tail.
            if not rejected:
                return ObservedPathProposal(path, tuple(self._observed[demand] for demand in demands),
                    self._solver_calls, self._observation_calls, len(self._domain)-len(self._observed))
        raise self._incomplete("proposal_solver_budget_exhausted")
