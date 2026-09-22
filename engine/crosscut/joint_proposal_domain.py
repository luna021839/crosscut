"""Bound expensive pixel search to complete, descriptor-ranked proposals.

This is a search domain, never an authorization or a hard-incompatibility
label. Exact offset/crop/track, decoded review and final media gates still run.
The search is not claimed to find the exhaustive graph's global optimum.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from time import monotonic
from typing import Any, Callable, Mapping, Sequence

from .joint_timeline import BoundaryMeasurementKey
from .joint_switch_requirements import (
    MAXIMUM_SOURCE_SWITCH_CAPACITY_STATES,
    SourceSwitchCounter,
    SourceSwitchRequirement,
    SourceSwitchSearchBudget,
)
from .review_first_runtime import enforce_review_first_deadline
from .timeline import NoCompleteTimelinePath, TimelineCandidate, TimelineConfig, TimelineResult, select_timeline_candidates
from .visual_score import VisualTimelineOptimizerWorkspace, _safe_cadence_profiles, _role_conditioned_visual_controls

CONTRACT = "complete_descriptor_proposal_domain_then_exact_pixel_validation_v2"
PROPOSAL_PATH_LIMITS = (3, 6, 9)
# Historical three-layer work units, now converted to a unique added-pair
# budget rather than treating every one-boundary repair as three layers.
MAXIMUM_LOCAL_PROPOSAL_REPAIRS = 6


def local_proposal_added_pair_budget(pool: JointProposalPool) -> int:
    """Retain the old six three-layer expansions' worst-case pair budget.

    Counting *new pairs*, not failed-frontier visits, lets a cheap one-boundary
    repair leave budget for later parts of a full song. This does not authorize
    unmeasured edges or grant additional renderer-proof calls.
    """
    lattice = pool.candidate_lattice
    if len(lattice) < 2:
        return 0
    cross_counts = [sum(a.source_id != b.source_id for a in left for b in right)
                    for left, right in zip(lattice,lattice[1:])]
    largest_old_repair = max(sum(cross_counts[max(0,i-1):i+2])
                             for i in range(len(cross_counts)))
    total_pairs = sum(len(a)*len(b) for a,b in zip(lattice,lattice[1:]))
    remaining = max(0, total_pairs-len(pool.domain(PROPOSAL_PATH_LIMITS[0])))
    return min(remaining, MAXIMUM_LOCAL_PROPOSAL_REPAIRS * largest_old_repair)


@dataclass(frozen=True, slots=True)
class JointProposalPool:
    paths: tuple[tuple[TimelineCandidate, ...], ...]
    cadence_profiles: tuple[str, ...]
    full_boundary_pair_count: int
    candidate_lattice: tuple[tuple[TimelineCandidate, ...], ...] = ()
    warm_start_paths: tuple[tuple[TimelineCandidate, ...], ...] = ()

    def __post_init__(self) -> None:
        if not self.paths or len(self.paths) != len(self.cadence_profiles):
            raise ValueError("proposal pool needs profile-bound complete paths")
        length = len(self.paths[0])
        if not length or any(len(path) != length for path in self.paths):
            raise ValueError("proposal paths must share a complete timeline")
        if any(not isinstance(c, TimelineCandidate) for path in self.paths for c in path):
            raise TypeError("proposal paths require measured timeline candidates")
        if len(self.warm_start_paths) > 3 or any(
            len(path) != length or any(c not in row for c, row in zip(path, self.candidate_lattice))
            for path in self.warm_start_paths
        ) or (self.warm_start_paths and len(self.candidate_lattice) != length):
            raise ValueError("warm proposals must rebind to the complete current lattice")
        if (isinstance(self.full_boundary_pair_count, bool)
                or not isinstance(self.full_boundary_pair_count, int)
                or self.full_boundary_pair_count < 0):
            raise ValueError("proposal pool pair count is invalid")

    def domain(self, maximum_paths: int) -> frozenset[BoundaryMeasurementKey]:
        if (isinstance(maximum_paths, bool) or not isinstance(maximum_paths, int)
                or maximum_paths < 0 or (maximum_paths == 0 and not self.warm_start_paths)):
            raise ValueError("proposal path limit must be a positive integer")
        proposed = frozenset(
            BoundaryMeasurementKey.from_candidates(index, outgoing, incoming)
            for path in self.warm_start_paths + self.paths[:maximum_paths]
            for index, (outgoing, incoming) in enumerate(zip(path, path[1:]))
        )
        # A complete proposal must not force a premature return to its source
        # sequence. Keeping the current source until a later matched cut is a
        # real option, still subject to exact proof and existing hold/cadence caps.
        continuations = frozenset(
            BoundaryMeasurementKey.from_candidates(index, outgoing, incoming)
            for index, (left, right) in enumerate(zip(self.candidate_lattice, self.candidate_lattice[1:]))
            for outgoing in left for incoming in right
            if outgoing.source_id == incoming.source_id
        )
        return proposed | continuations

    def scalar_audit(self) -> dict[str, Any]:
        return {
            "contract": CONTRACT,
            "canonical_spans": [[c.segment_index, c.start_seconds, c.end_seconds] for c in self.paths[0]],
            "proposal_source_paths": [list(c.source_id for c in path) for path in self.paths],
            "cadence_profiles": list(self.cadence_profiles),
            "requirement_covering_path_count": sum(
                profile == "section_switch_requirement"
                for profile in self.cadence_profiles
            ),
            "full_boundary_pair_count": self.full_boundary_pair_count,
            "source_continuations_included": bool(self.candidate_lattice),
            "warm_start_proposal_count": len(self.warm_start_paths),
            "warm_start_proofs_restored": False,
            "final_media_authority": False,
            "raw_descriptors_persisted": False,
        }


def proposal_pool_from_results(
    lattice: Sequence[Sequence[TimelineCandidate]],
    results: Sequence[TimelineResult],
    cadence_profiles: Sequence[str],
) -> JointProposalPool:
    if len(results) != len(cadence_profiles):
        raise ValueError("proposal paths and cadence profiles must align")
    segments = tuple(tuple(row) for row in lattice)
    if not segments or any(not row for row in segments):
        raise ValueError("proposal lattice must contain nonempty segments")
    paths, profiles, seen = [], [], set()
    for result, profile in zip(results, cadence_profiles, strict=True):
        path = tuple(result.candidates)
        if len(path) != len(segments) or any(c not in row for c, row in zip(path, segments)):
            raise ValueError("proposal does not cover the exact complete lattice")
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
        profiles.append(profile)
        if len(paths) == PROPOSAL_PATH_LIMITS[-1]:
            break
    if not paths:
        raise NoCompleteTimelinePath("no complete descriptor proposal", reason_code="joint_descriptor_proposal_pool_empty")
    return JointProposalPool(tuple(paths), tuple(profiles), sum(
        len(a) * len(b) for a, b in zip(segments, segments[1:])
    ), segments)


def add_local_proposal_detour(
    pool: JointProposalPool,
    domain: frozenset[BoundaryMeasurementKey],
    failure: NoCompleteTimelinePath,
) -> tuple[frozenset[BoundaryMeasurementKey], dict[str, Any]] | None:
    """Try the failed boundary first, then its adjacent reconnection edges."""
    lattice = pool.candidate_lattice
    if not lattice:
        return None
    positions = {row[0].segment_index: index for index, row in enumerate(lattice)}
    pending, blocked = [failure], set()
    while pending:
        item = pending.pop()
        pending.extend(child for _, child in item.profile_failures)
        position = positions.get(item.segment_index)
        if position is not None and position > 0:
            blocked.add(position - 1)
    for boundary in sorted(blocked):
        outgoing = {key.outgoing for key in domain if key.boundary_index == boundary}
        incoming = {key.incoming for key in domain if key.boundary_index == boundary}
        before = {key.outgoing for key in domain if key.boundary_index == boundary - 1}
        after = {key.incoming for key in domain if key.boundary_index == boundary + 1}
        additions = set()
        def add(index, left, right):
            additions.update(BoundaryMeasurementKey.from_candidates(index, a, b) for a in left for b in right)
        # Continuations already exist throughout the lattice. A switch at the
        # failed boundary can therefore continue without reopening both nearby
        # layers. Escalate only after that smaller search is also disconnected.
        add(boundary, lattice[boundary], lattice[boundary+1])
        additions.difference_update(domain)
        if additions:
            return domain | frozenset(additions), {
                "blocked_boundary_index": boundary,
                "added_boundary_pair_count": len(additions),
                "touched_boundary_indices": [boundary],
                "repair_scope": "failed_boundary_only",
                "hard_no_goods_authored": False, "final_media_authority": False,
            }
        # Replacing either side of the failed edge must also allow rejoining
        # the original complete path. No source is declared safe by this.
        add(boundary, lattice[boundary], incoming)
        add(boundary, outgoing, lattice[boundary + 1])
        if boundary > 0:
            add(boundary - 1, before, lattice[boundary])
        if boundary + 1 < len(lattice) - 1:
            add(boundary + 1, lattice[boundary + 1], after)
        additions.difference_update(domain)
        if additions:
            return domain | frozenset(additions), {
                "blocked_boundary_index": boundary,
                "added_boundary_pair_count": len(additions),
                "touched_boundary_indices": sorted({key.boundary_index for key in additions}),
                "repair_scope": "adjacent_reconnection",
                "hard_no_goods_authored": False, "final_media_authority": False,
            }
    return None


def proposal_only_config(config: TimelineConfig) -> TimelineConfig:
    """Do not reject a zoom-alignable close-up using untransformed pixels."""
    return replace(
        config,
        maximum_cross_source_scale_delta=None,
        maximum_cross_source_focus_mismatch=None,
        maximum_cross_source_layout_mismatch=None,
        maximum_cross_source_formation_mismatch=None,
        maximum_cross_source_silhouette_mismatch=None,
        minimum_cross_source_face_similarity=None,
        require_cross_source_visual_evidence=False,
        maximum_boundary_side_scale_delta=None,
        maximum_boundary_side_structure_mismatch=None,
        maximum_boundary_side_pose_mismatch=None,
        maximum_boundary_side_shot_type_mismatch=None,
        require_cross_source_boundary_side_stability=False,
        scale_transition_weight=min(config.scale_transition_weight, .35),
        focus_position_transition_weight=min(config.focus_position_transition_weight, .35),
    )


@dataclass(frozen=True, slots=True)
class _RequirementPathNode:
    candidate: TimelineCandidate
    parent: "_RequirementPathNode | None"

    def path(self) -> tuple[TimelineCandidate, ...]:
        path = []
        node = self
        while node is not None:
            path.append(node.candidate)
            node = node.parent
        return tuple(reversed(path))


def _requirement_covering_path(
    lattice: Sequence[Sequence[TimelineCandidate]],
    requirements: Sequence[SourceSwitchRequirement],
    source_families: Mapping[str, str] | None,
    *,
    fps: float,
    maximum_offset_frames: int,
    budget: SourceSwitchSearchBudget | None = None,
) -> tuple[TimelineCandidate, ...] | None:
    """Return a connected path with a concrete frame-offset solution."""

    windows = tuple(requirements)
    if not windows:
        return None
    budget = budget or SourceSwitchSearchBudget()
    counter = SourceSwitchCounter(
        windows,
        lattice,
        fps=fps,
        maximum_offset_frames=maximum_offset_frames,
        source_families=source_families,
        budget=budget,
    )

    def emission(candidate: TimelineCandidate) -> float:
        return (
            candidate.quality_score
            + candidate.sync_score
            + candidate.composition_score
            + candidate.boundary_score
        )

    initial_counts = (0,) * len(windows)
    states: dict[
        tuple[TimelineCandidate, tuple[int, ...]],
        tuple[float, _RequirementPathNode],
    ] = {
        (candidate, initial_counts): (emission(candidate), _RequirementPathNode(candidate, None))
        for candidate in lattice[0]
    }
    for boundary_index, right_row in enumerate(lattice[1:]):
        next_by_candidate: dict[
            TimelineCandidate,
            dict[
                tuple[int, ...],
                tuple[float, _RequirementPathNode],
            ],
        ] = {}
        state_count = 0
        for (left, counts), (score, path) in states.items():
            for right in right_row:
                budget.check(states=state_count, segment_index=right.segment_index)
                switched = counter.different(left.source_id, right.source_id)
                for increments in counter.potential_increment_vectors(
                    boundary_index,
                    left,
                    right,
                ):
                    next_counts = tuple(
                        min(window.minimum_count, count + increment)
                        for window, count, increment in zip(
                            windows,
                            counts,
                            increments,
                            strict=True,
                        )
                    )
                    if not counter.feasible(next_counts, boundary_index + 1):
                        continue
                    next_score = (
                        score + emission(right) - (0.02 if switched else 0.0)
                    )
                    value = (next_score, _RequirementPathNode(right, path))
                    frontier = next_by_candidate.setdefault(right, {})
                    previous = frontier.get(next_counts)
                    if previous is not None and previous[0] >= next_score:
                        continue
                    def dominates(existing_counts, existing_score):
                        budget.check(states=state_count, segment_index=right.segment_index)
                        return existing_score >= next_score and all(
                            a >= b for a, b in zip(existing_counts, next_counts, strict=True)
                        )
                    if any(
                        dominates(existing_counts, existing_score)
                        for existing_counts, (
                            existing_score,
                            _existing_path,
                        ) in frontier.items()
                    ):
                        continue
                    dominated = [
                        existing_counts
                        for existing_counts, (
                            existing_score,
                            _existing_path,
                        ) in frontier.items()
                        if next_score >= existing_score
                        and all(
                            current >= existing
                            for current, existing in zip(
                                next_counts,
                                existing_counts,
                                strict=True,
                            )
                        )
                    ]
                    for existing_counts in dominated:
                        del frontier[existing_counts]
                        state_count -= 1
                    if next_counts not in frontier:
                        state_count += 1
                    budget.check(states=state_count, segment_index=right.segment_index)
                    frontier[next_counts] = value
        next_states = {
            (candidate, counts): value
            for candidate, frontier in next_by_candidate.items()
            for counts, value in frontier.items()
        }
        if len(next_states) > MAXIMUM_SOURCE_SWITCH_CAPACITY_STATES:
            raise NoCompleteTimelinePath(
                "descriptor requirement path exceeded its deterministic state budget",
                reason_code="joint_descriptor_switch_requirement_state_budget_exhausted",
                segment_index=right_row[0].segment_index,
                rejection_counts={
                    "joint_descriptor_switch_requirement_state_budget_exhausted": 1,
                    "capacity_state_count": len(next_states),
                    "capacity_state_limit": MAXIMUM_SOURCE_SWITCH_CAPACITY_STATES,
                    "proof_materialization_call_count": 0,
                },
            )
        states = next_states
        if not states:
            break
    eligible = [
        value
        for (_candidate, counts), value in states.items()
        if all(
            count >= window.minimum_count
            for count, window in zip(counts, windows, strict=True)
        )
    ]
    if not eligible:
        raise NoCompleteTimelinePath(
            "descriptor proposal graph has no connected path satisfying "
            "section source-switch requirements",
            reason_code="joint_descriptor_switch_requirement_unreachable",
            segment_index=lattice[-1][0].segment_index,
            rejection_counts={
                "joint_descriptor_switch_requirement_unreachable": 1,
                "proof_materialization_call_count": 0,
            },
        )
    return max(eligible, key=lambda item: item[0])[1].path()


def build_joint_proposal_pool(
    workspace: VisualTimelineOptimizerWorkspace,
    *,
    source_switch_requirements: Sequence[SourceSwitchRequirement] = (),
    source_switch_families: Mapping[str, str] | None = None,
    fps: float = 29.97,
    maximum_offset_frames: int = 8,
    deadline_monotonic_seconds: float | None = None,
) -> JointProposalPool:
    budget = SourceSwitchSearchBudget(deadline_monotonic_seconds=deadline_monotonic_seconds)
    lattice = workspace.candidates_by_segment
    if workspace.prescribed_source_ids is not None:
        lattice = tuple(tuple(c for c in row if c.source_id == source_id)
                        for row, source_id in zip(lattice, workspace.prescribed_source_ids, strict=True))
        if any(len(row) != 1 for row in lattice):
            raise ValueError("prescribed proposal path is not readable")
    pools, failures = [], []
    for profile in _safe_cadence_profiles(workspace.diversity):
        budget.check()
        config = proposal_only_config(profile.config)
        kwargs = {}
        if workspace.professional_montage_policy is not None:
            adjustments, bonus, _ = _role_conditioned_visual_controls(
                lattice, section_plan=workspace.section_plan, phrase_times=workspace.phrase_times,
                policy=workspace.professional_montage_policy, cadence_profile=profile.name,
                subject_metadata_by_candidate=workspace.subject_metadata_by_candidate,
                timeline_config=config,
            )
            kwargs.update(emission_adjustments=adjustments, source_switch_bonus_by_segment=bonus)
        try:
            with enforce_review_first_deadline(deadline_monotonic_seconds, stage="descriptor proposals"):
                paths = select_timeline_candidates(
                    lattice, config=config, preferred_source_ids=workspace.preferred_source_ids,
                    maximum_results=3, diversity_penalty=.12, minimum_hamming_fraction=.02, **kwargs,
                )
        except NoCompleteTimelinePath as exc:
            failures.append((profile.name, exc))
            paths = ()
        pools.append((profile.name, paths))
    interleaved = [(name, paths[rank]) for rank in range(3) for name, paths in pools if rank < len(paths)]
    if not interleaved:
        raise NoCompleteTimelinePath("descriptor proposal cadence graph is empty",
            reason_code="joint_descriptor_proposal_pool_empty", profile_failures=failures)
    profiles, results = zip(*interleaved)
    pool = proposal_pool_from_results(
        workspace.candidates_by_segment,
        results,
        profiles,
    )
    requirement_path = _requirement_covering_path(
        lattice,
        source_switch_requirements,
        source_switch_families,
        fps=fps,
        maximum_offset_frames=maximum_offset_frames,
        budget=budget,
    )
    if requirement_path is None:
        return pool
    if requirement_path in pool.paths:
        matched_index = pool.paths.index(requirement_path)
        retained_paths = (requirement_path,) + tuple(
            path
            for index, path in enumerate(pool.paths)
            if index != matched_index
        )
        retained_profiles = ("section_switch_requirement",) + tuple(
            profile
            for index, profile in enumerate(pool.cadence_profiles)
            if index != matched_index
        )
    else:
        retained_paths = (requirement_path,) + pool.paths
        retained_profiles = (
            "section_switch_requirement",
        ) + pool.cadence_profiles
    return replace(
        pool,
        paths=retained_paths[: PROPOSAL_PATH_LIMITS[-1]],
        cadence_profiles=retained_profiles[: PROPOSAL_PATH_LIMITS[-1]],
    )


def select_with_joint_proposal_domains(
    pool: JointProposalPool,
    *,
    maximum_proof_calls: int,
    select: Callable[..., Any],
    report: Callable[[dict[str, Any]], None] | None = None,
    deadline_monotonic_seconds: float | None = None,
    clock: Callable[[], float] = monotonic,
) -> Any:
    """Proposal sizes and pair-budgeted local detours share one proof budget."""
    if isinstance(maximum_proof_calls, bool) or not isinstance(maximum_proof_calls, int) or maximum_proof_calls < 1:
        raise ValueError("proposal selection needs a positive proof budget")
    if deadline_monotonic_seconds is not None and (
        isinstance(deadline_monotonic_seconds, bool)
        or not isinstance(deadline_monotonic_seconds, (int, float))
        or not math.isfinite(float(deadline_monotonic_seconds))
    ):
        raise ValueError("proposal selection deadline must be finite")
    if not callable(clock):
        raise TypeError("proposal selection clock must be callable")

    def require_wall_clock_budget() -> None:
        if (
            deadline_monotonic_seconds is not None
            and float(clock()) >= float(deadline_monotonic_seconds)
        ):
            raise NoCompleteTimelinePath(
                "joint proposal selection exceeded its wall-clock budget",
                reason_code="joint_proposal_wall_clock_budget_exhausted",
                rejection_counts={
                    "joint_proposal_wall_clock_budget_exhausted": 1,
                    "proof_materialization_call_count": charged,
                },
            )

    charged = 0
    events: list[dict[str, Any]] = []
    seen_domains = set()
    extra_domain: frozenset[BoundaryMeasurementKey] = frozenset()
    local_repairs = []
    added_pair_budget = local_proposal_added_pair_budget(pool)
    limit_index = 0
    limits = ((0,) + PROPOSAL_PATH_LIMITS) if pool.warm_start_paths else PROPOSAL_PATH_LIMITS
    last_failure = None
    while limit_index < len(limits):
        require_wall_clock_budget()
        limit = limits[limit_index]
        domain = pool.domain(limit) | extra_domain
        if domain in seen_domains:
            limit_index += 1
            continue
        seen_domains.add(domain)
        remaining = maximum_proof_calls - charged
        if remaining <= 0:
            break
        event = {
            "contract": CONTRACT, "proposal_path_count": min(limit, len(pool.paths)) + len(pool.warm_start_paths),
            "warm_start_only": limit == 0,
            "boundary_pair_count": len(domain), "full_boundary_pair_count": pool.full_boundary_pair_count,
            "remaining_proof_calls": remaining, "final_media_authority": False,
            "raw_frame_visual_limits_used_as_final_authority": False,
            "local_repair_round_count": len(local_repairs),
            "local_added_boundary_pair_count": len(extra_domain),
            "maximum_local_added_boundary_pairs": added_pair_budget,
            "source_continuation_pair_count": sum(key.outgoing.source_id == key.incoming.source_id for key in domain),
        }
        if report is not None:
            report({**event, "phase": "proposal_domain_start"})
        try:
            # The independent CLI supervisor is the native-call backstop. This
            # alarm also interrupts blocking Python providers in direct use.
            with enforce_review_first_deadline(
                deadline_monotonic_seconds if clock is monotonic else None,
                stage="exact joint proposal selection",
            ):
                result = select(boundary_search_domain=domain, maximum_proof_materialization_calls=remaining)
        except NoCompleteTimelinePath as exc:
            if exc.reason_code == "joint_proposal_wall_clock_budget_exhausted":
                raise
            used = exc.rejection_counts.get("proof_materialization_call_count", 0)
            if isinstance(used, bool) or not isinstance(used, int) or not 0 <= used <= remaining:
                raise RuntimeError("failed proposal selection lost its proof budget") from exc
            charged += used
            last_failure = exc
            event.update(status="no_complete_passing_path", proof_calls=used, failure=exc.to_dict())
            events.append(event)
            if report is not None:
                report({**event, "phase": "proposal_domain_exhausted"})
            if limit == 0:
                # A stale hint must not spend local repair work ahead of the
                # current descriptor proposals, nor replenish proof budgets.
                limit_index += 1
                continue
            repair = (
                add_local_proposal_detour(pool, domain, exc)
                if len(extra_domain) < added_pair_budget
                else None
            )
            if repair is not None and len(extra_domain) + len(repair[0]-domain) > added_pair_budget:
                # Never take an arbitrary source-ID prefix of a repair, or
                # silently replenish the graph-work budget at the next size.
                repair = None
                if report is not None:
                    report({"contract": CONTRACT, "phase": "proposal_local_pair_budget_exhausted",
                            "used_added_boundary_pairs": len(extra_domain),
                            "maximum_local_added_boundary_pairs": added_pair_budget})
            if repair is not None:
                widened, repair_audit = repair
                extra_domain |= widened - domain
                local_repairs.append(repair_audit)
                if report is not None:
                    report({"contract": CONTRACT, "phase": "proposal_local_detour_added", **repair_audit})
            else:
                limit_index += 1
            continue
        require_wall_clock_budget()
        audit = dict(result.selection_retry_audit)
        used = audit.get("proof_materialization_call_count")
        if isinstance(used, bool) or not isinstance(used, int) or not 0 <= used <= remaining:
            raise RuntimeError("passing proposal selection lost its proof budget")
        charged += used
        events.append({**event, "status": "exact_selection_passed", "proof_calls": used})
        audit.update(
            proof_materialization_call_count=charged,
            maximum_proof_materialization_calls=maximum_proof_calls,
            proposal_search_domain={
                "contract": CONTRACT, "events": events,
                "local_repairs": local_repairs,
                "maximum_local_added_boundary_pairs": added_pair_budget,
                "local_added_boundary_pair_count": len(extra_domain),
                "historical_three_layer_repair_work_units": MAXIMUM_LOCAL_PROPOSAL_REPAIRS,
                "exhaustive_global_optimum_claimed": False,
                "quality_gates_waived": False, "hard_no_goods_authored": False,
            },
        )
        return replace(result, selection_retry_audit=audit)
    raise NoCompleteTimelinePath(
        "bounded descriptor proposals have no complete pixel-validated timeline",
        reason_code="joint_proposal_domains_exhausted",
        rejection_counts={"proposal_domain_count": len(events), "local_repair_round_count": len(local_repairs),
                          "local_added_boundary_pair_count": len(extra_domain),
                          "maximum_local_added_boundary_pairs": added_pair_budget,
                          "proof_materialization_call_count": charged},
        profile_failures=(("last_proposal_domain", last_failure),) if last_failure is not None else (),
    )
