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
from time import monotonic
from typing import Callable, Generic, Mapping, Sequence, TypeVar

from .member_endpoint_exclusions import MemberEndpointExclusion


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
    ENDPOINT_INELIGIBLE = "observed_endpoint_ineligible"
    ELIGIBLE = "observed_eligible"
    REJECTED = "observed_rejected"


@dataclass(frozen=True, slots=True)
class TransitionObservation(Generic[Evidence]):
    demand: TransitionDemand
    eligible: bool
    cost: float
    evidence: Evidence
    policy_evidence: object | None = None
    complete: bool = True

    def __post_init__(self) -> None:
        if type(self.eligible) is not bool:
            raise ValueError("observed eligibility must be boolean")
        if type(self.complete) is not bool or (not self.eligible and not self.complete):
            raise ValueError("a rejected pair requires complete offset observations")
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
    severe_flags: tuple[str, ...] | None = None
    policy_evidence: object | None = None

    def __post_init__(self) -> None:
        if self.severe_flags is None:
            flags = (("content_or_cross_source_policy_ineligible",)
                if self.state is ObservationState.CONTENT_INELIGIBLE else
                ("cross_source_focal_subject_unresolved",)
                if self.state is ObservationState.ENDPOINT_INELIGIBLE else
                ("observed_transition_rejected",)
                if self.state is ObservationState.REJECTED else
                tuple(getattr(self.policy_evidence, 'severe_flags', ())))
            object.__setattr__(self, 'severe_flags', flags)

    def __getattr__(self, name: str):
        # These are only policy scalars/real measured effect certificates. No
        # pose, sample or optimistic effect certificate is ever manufactured.
        # The shared solver may optimistically consider an unknown hard edge;
        # all selected demands must be observed and the policy solved AGAIN.
        if name in {'cinematic_bridge', 'motion_connection', 'transform_repair',
                    'attention_position_mismatch', 'attention_scale_mismatch',
                    'attention_geometry_basis', 'silhouette_mismatch'}:
            if self.state in {ObservationState.REJECTED, ObservationState.CONTENT_INELIGIBLE,
                              ObservationState.ENDPOINT_INELIGIBLE}:
                return None
            return getattr(self.policy_evidence, name, None)
        raise AttributeError(name)


ProposalGraph = tuple[tuple[tuple[ProposalEdge, ...], ...], ...]


@dataclass(frozen=True, slots=True)
class ObservedPathProposal(Generic[Evidence]):
    source_indices: tuple[int, ...]
    transitions: tuple[TransitionObservation[Evidence], ...]
    solver_call_count: int
    observation_count: int
    unobserved_transition_count: int
    reused_observation_count: int = 0
    endpoint_excluded_transition_count: int = 0
    refinement_count: int = 0
    reused_pair_refinement_reservation_count: int = 0
    endpoint_only_observation_count: int = 0


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

    Holds require an explicit policy adapter; they are disabled by default.
    `propose` returning None cannot prove the complete edit domain infeasible.
    Provider/SSD/cancellation errors escape directly, without a retry loop.
    """
    def __init__(
        self,
        candidate_validity: Sequence[Sequence[bool]],
        *,
        maximum_observations: int,
        maximum_solver_calls: int,
        allow_same_source_holds: bool = False,
        proposal_costs: Mapping[TransitionDemand, float] | None = None,
        progress_callback: Callable[[dict], None] | None = None,
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
        if progress_callback is not None and not callable(progress_callback):
            raise ValueError('observation progress callback must be callable')
        self._progress_callback = progress_callback
        self._started = monotonic()
        if type(allow_same_source_holds) is not bool:
            raise ValueError("same-source policy must be explicit")
        self.allow_same_source_holds = allow_same_source_holds
        self._domain = frozenset(
            TransitionDemand(boundary, left, right)
            for boundary in range(len(self.validity)-1)
            for left in range(self.source_count)
            for right in range(self.source_count)
            if (left != right or allow_same_source_holds)
            and self.validity[boundary][left] and self.validity[boundary+1][right]
        )
        self._observed: dict[TransitionDemand, TransitionObservation[Evidence]] = {}
        self._excluded: dict[TransitionDemand, MemberEndpointExclusion] = {}
        self._endpoint_certificates: set[MemberEndpointExclusion] = set()
        self._proposal_costs = dict(proposal_costs or {})
        if any(demand not in self._domain or isinstance(cost, bool)
               or not isinstance(cost, (int, float)) or not math.isfinite(cost) or not 0 <= cost <= 1
               for demand, cost in self._proposal_costs.items()):
            raise ValueError('coarse proposal priorities must be finite domain-bound values in [0, 1]')
        self._solver_calls = 0
        self._observation_calls = 0
        self._refinement_calls = 0
        self._reused_observations = 0
        self._reserved_pairs: set[TransitionDemand] = set()
        self._reused_refinement_reservations = 0
        self._endpoint_only_observations = 0

    def proposal_graph(self) -> ProposalGraph:
        def edge(demand: TransitionDemand) -> ProposalEdge:
            observed = self._observed.get(demand)
            if demand not in self._domain:
                return ProposalEdge(ObservationState.CONTENT_INELIGIBLE, 1.0)
            if demand in self._excluded:
                return ProposalEdge(ObservationState.ENDPOINT_INELIGIBLE, 1.0)
            if observed is not None and not observed.eligible:
                return ProposalEdge(ObservationState.REJECTED, 1.0)
            if observed is None:
                return ProposalEdge(ObservationState.UNOBSERVED, 1.0+self._proposal_costs.get(demand, 0.))
            priority = .25 * (.5 + math.atan(observed.cost) / math.pi)
            return ProposalEdge(ObservationState.ELIGIBLE, priority, policy_evidence=observed.policy_evidence)
        return tuple(tuple(tuple(edge(TransitionDemand(boundary, left, right))
            for right in range(self.source_count)) for left in range(self.source_count))
            for boundary in range(len(self.validity)-1))

    def _incomplete(self, reason: str) -> MemberObservationSearchIncomplete:
        error = MemberObservationSearchIncomplete(reason, solver_calls=self._solver_calls,
            observations=self._observation_calls, unobserved=self.unobserved_count)
        error.diagnostics['reused_observation_count'] = self._reused_observations
        error.diagnostics['endpoint_excluded_transition_count'] = len(self._excluded)
        error.diagnostics['refinement_count'] = self._refinement_calls
        error.diagnostics['reused_pair_refinement_reservation_count'] = self._reused_refinement_reservations
        error.diagnostics['endpoint_only_observation_count'] = self._endpoint_only_observations
        return error

    @property
    def unobserved_count(self) -> int:
        return len(self._domain)-len(self._observed)-len(self._excluded)

    @property
    def endpoint_excluded_count(self) -> int:
        return len(self._excluded)

    @property
    def endpoint_excluded_demands(self) -> frozenset[TransitionDemand]:
        return frozenset(self._excluded)

    @property
    def observations(self) -> tuple[TransitionObservation[Evidence], ...]:
        return tuple(self._observed.values())

    def unobserved_at(self, boundary_index: int) -> tuple[TransitionDemand, ...]:
        """Only legal, unmeasured pairs; no inference or eligibility defaults."""
        return tuple(sorted((demand for demand in self._domain
            if demand.boundary_index == boundary_index and demand not in self._observed
            and demand not in self._excluded),
            key=lambda demand: (demand.outgoing_source_index, demand.incoming_source_index)))

    def _emit(self, event, **payload):
        if self._progress_callback is not None:
            self._progress_callback(dict(phase='member_observation_search', event=event,
                elapsed_seconds=round(monotonic()-self._started, 3),
                solver_call_count=self._solver_calls, observation_count=self._observation_calls,
                refinement_count=self._refinement_calls,
                reused_pair_refinement_reservation_count=self._reused_refinement_reservations,
                reused_observation_count=self._reused_observations,
                unobserved_transition_count=self.unobserved_count,
                endpoint_only_observation_count=self._endpoint_only_observations,
                endpoint_excluded_transition_count=len(self._excluded), **payload))

    def run(
        self,
        *,
        propose: Callable[[ProposalGraph], Sequence[int] | None],
        observe: Callable[[TransitionDemand], TransitionObservation[Evidence] | MemberEndpointExclusion],
        refine: Callable[[TransitionObservation[Evidence]], TransitionObservation[Evidence]] | None = None,
        reuse_observed: Callable[[TransitionDemand], Sequence[TransitionObservation[Evidence]]] | None = None,
        endpoint_exclusions: Callable[[TransitionDemand], Sequence[MemberEndpointExclusion]] | None = None,
    ) -> ObservedPathProposal[Evidence]:
        def publish_related(demand):
            if reuse_observed is not None:
                related = tuple(reuse_observed(demand))
                if len(related) > self.source_count ** 2:
                    raise ValueError('cached pair publication exceeded one boundary domain')
                seen = set()
                for item in related:
                    if (not isinstance(item, TransitionObservation)
                            or item.demand.boundary_index != demand.boundary_index
                            or item.demand not in self._domain
                            or item.demand in self._observed or item.demand in self._excluded
                            or item.demand in seen):
                        raise ValueError('cached pair evidence must be new and boundary-local')
                    seen.add(item.demand)
                self._observed.update((item.demand, item) for item in related)
                self._reused_observations += len(related)
            if endpoint_exclusions is not None:
                certificates = tuple(endpoint_exclusions(demand))
                if len(certificates) > 2*self.source_count:
                    raise ValueError('endpoint certificates exceed one boundary domain')
                for certificate in certificates:
                    if (not isinstance(certificate, MemberEndpointExclusion)
                            or certificate.boundary_index != demand.boundary_index
                            or certificate.source_index >= self.source_count):
                        raise ValueError('endpoint certificates must be boundary/source-local')
                    if certificate in self._endpoint_certificates:
                        continue
                    if any(item.eligible and certificate.excludes(item.demand)
                           for item in self._observed.values()):
                        raise ValueError('endpoint certificate contradicts eligible paired evidence')
                    self._endpoint_certificates.add(certificate)
                    for other in self._domain:
                        if other not in self._observed and certificate.excludes(other):
                            self._excluded[other] = certificate

        def charge_observation(demand, *, refining=False):
            if not refining or demand not in self._reserved_pairs:
                # Cache-only positive offsets cost no new inference. If that
                # pair is selected, reserve its remaining work BEFORE calling
                # refinement; cache publication never grants free native work.
                if self._observation_calls >= self.maximum_observations:
                    raise self._incomplete('exact_observation_budget_exhausted')
                self._observation_calls += 1
                self._reserved_pairs.add(demand)
                if refining:
                    self._reused_refinement_reservations += 1
            if refining:
                self._refinement_calls += 1

        while self._solver_calls < self.maximum_solver_calls:
            self._solver_calls += 1
            self._emit('policy_proposal_started')
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
            if any(demand in self._excluded for demand in demands):
                raise ValueError('proposal repeats an exact endpoint exclusion')
            if any(demand in self._observed and not self._observed[demand].eligible for demand in demands):
                raise ValueError("proposal repeats an observed rejected transition")
            rejected = False
            observation_changed = False
            for demand in demands:
                if demand in self._observed:
                    continue
                # Charge before the call so external failures do not reset work
                # accounting if a caller deliberately resumes this same ledger.
                charge_observation(demand)
                self._emit('exact_pair_started', boundary_index=demand.boundary_index,
                    outgoing_source_index=demand.outgoing_source_index, incoming_source_index=demand.incoming_source_index)
                observed = observe(demand)
                if isinstance(observed, MemberEndpointExclusion):
                    if not observed.excludes(demand):
                        raise ValueError('endpoint observation does not exclude its requested pair')
                    publish_related(demand)
                    # Publishing cached evidence may have completed this exact
                    # rejected pair without I/O. Keep that real measurement;
                    # never double-count it as a separate excluded edge.
                    cached_pair = self._observed.get(demand)
                    if observed not in self._endpoint_certificates or not (
                            demand in self._excluded
                            or cached_pair is not None and not cached_pair.eligible):
                        raise ValueError('endpoint observation must be confirmed by read-only exact samples')
                    self._endpoint_only_observations += 1
                    self._emit('exact_pair_endpoint_excluded', boundary_index=demand.boundary_index,
                        outgoing_source_index=demand.outgoing_source_index,
                        incoming_source_index=demand.incoming_source_index,
                        excluded_source_index=observed.source_index, excluded_role=observed.role)
                    rejected = True
                    observation_changed = True
                    break
                if not isinstance(observed, TransitionObservation) or observed.demand != demand:
                    raise ValueError("observer returned evidence for a different transition")
                self._observed[demand] = observed
                publish_related(demand)
                self._emit('exact_pair_observed', boundary_index=demand.boundary_index,
                    outgoing_source_index=demand.outgoing_source_index,
                    incoming_source_index=demand.incoming_source_index,
                    eligible=observed.eligible,
                    complete_offsets=observed.complete,
                    severe_flags=list(getattr(observed.policy_evidence, 'severe_flags', ())),
                    effect=getattr(getattr(observed.policy_evidence, 'cinematic_bridge', None), 'style', None))
                observation_changed = True
                if not observed.eligible:
                    rejected = True
                    break  # Repair this path before observing its unused tail.
            # New evidence can turn an optimistic hard edge into a required
            # dissolve and change spacing/role/exposure feasibility. A path is
            # never returned under the stale policy that preceded observation.
            if not rejected and not observation_changed:
                pending = tuple(self._observed[demand] for demand in demands
                    if not self._observed[demand].complete)
                if pending:
                    if refine is None:
                        raise ValueError('partial selected observations require complete refinement')
                    for previous in pending:
                        demand = previous.demand
                        charge_observation(demand, refining=True)
                        self._emit('exact_pair_refinement_started', boundary_index=demand.boundary_index,
                            outgoing_source_index=demand.outgoing_source_index,
                            incoming_source_index=demand.incoming_source_index)
                        completed = refine(previous)
                        if (not isinstance(completed, TransitionObservation)
                                or completed.demand != demand or not completed.complete
                                or not completed.eligible):
                            raise ValueError('refinement must retain eligibility and complete the same pair')
                        self._observed[demand] = completed
                        publish_related(demand)
                        self._emit('exact_pair_refinement_completed', boundary_index=demand.boundary_index)
                    # Added offsets can change the best cut/effect/spacing and
                    # its cost. Re-solve the complete policy, never publish the
                    # stale proposal that preceded selected-pair refinement.
                    continue
                self._emit('observed_path_policy_verified')
                return ObservedPathProposal(path, tuple(self._observed[demand] for demand in demands),
                    self._solver_calls, self._observation_calls, self.unobserved_count,
                    self._reused_observations, len(self._excluded), self._refinement_calls,
                    self._reused_refinement_reservations, self._endpoint_only_observations)
        raise self._incomplete("proposal_solver_budget_exhausted")
