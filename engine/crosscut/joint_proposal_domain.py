"""Bound expensive pixel search to complete, descriptor-ranked proposals.

This is a search domain, never an authorization or a hard-incompatibility
label. Exact offset/crop/track, decoded review and final media gates still run.
The search is not claimed to find the exhaustive graph's global optimum.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

from .joint_timeline import BoundaryMeasurementKey
from .timeline import NoCompleteTimelinePath, TimelineCandidate, TimelineConfig, TimelineResult, select_timeline_candidates
from .visual_score import VisualTimelineOptimizerWorkspace, _safe_cadence_profiles, _role_conditioned_visual_controls

CONTRACT = "complete_descriptor_proposal_domain_then_exact_pixel_validation_v1"
PROPOSAL_PATH_LIMITS = (3, 6, 9)
MAXIMUM_LOCAL_PROPOSAL_REPAIRS = 6


@dataclass(frozen=True, slots=True)
class JointProposalPool:
    paths: tuple[tuple[TimelineCandidate, ...], ...]
    cadence_profiles: tuple[str, ...]
    full_boundary_pair_count: int
    candidate_lattice: tuple[tuple[TimelineCandidate, ...], ...] = ()

    def __post_init__(self) -> None:
        if not self.paths or len(self.paths) != len(self.cadence_profiles):
            raise ValueError("proposal pool needs profile-bound complete paths")
        length = len(self.paths[0])
        if not length or any(len(path) != length for path in self.paths):
            raise ValueError("proposal paths must share a complete timeline")
        if any(not isinstance(c, TimelineCandidate) for path in self.paths for c in path):
            raise TypeError("proposal paths require measured timeline candidates")
        if (isinstance(self.full_boundary_pair_count, bool)
                or not isinstance(self.full_boundary_pair_count, int)
                or self.full_boundary_pair_count < 0):
            raise ValueError("proposal pool pair count is invalid")

    def domain(self, maximum_paths: int) -> frozenset[BoundaryMeasurementKey]:
        if isinstance(maximum_paths, bool) or not isinstance(maximum_paths, int) or maximum_paths < 1:
            raise ValueError("proposal path limit must be a positive integer")
        proposed = frozenset(
            BoundaryMeasurementKey.from_candidates(index, outgoing, incoming)
            for path in self.paths[:maximum_paths]
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
            "full_boundary_pair_count": self.full_boundary_pair_count,
            "source_continuations_included": bool(self.candidate_lattice),
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
    """Open only a failed frontier's adjacent substitutions, not the full graph."""
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


def build_joint_proposal_pool(workspace: VisualTimelineOptimizerWorkspace) -> JointProposalPool:
    lattice = workspace.candidates_by_segment
    if workspace.prescribed_source_ids is not None:
        lattice = tuple(tuple(c for c in row if c.source_id == source_id)
                        for row, source_id in zip(lattice, workspace.prescribed_source_ids, strict=True))
        if any(len(row) != 1 for row in lattice):
            raise ValueError("prescribed proposal path is not readable")
    pools, failures = [], []
    for profile in _safe_cadence_profiles(workspace.diversity):
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
    return proposal_pool_from_results(workspace.candidates_by_segment, results, profiles)


def select_with_joint_proposal_domains(
    pool: JointProposalPool,
    *,
    maximum_proof_calls: int,
    select: Callable[..., Any],
    report: Callable[[dict[str, Any]], None] | None = None,
) -> Any:
    """Three proposal sizes plus bounded local detours share one proof budget."""
    if isinstance(maximum_proof_calls, bool) or not isinstance(maximum_proof_calls, int) or maximum_proof_calls < 1:
        raise ValueError("proposal selection needs a positive proof budget")
    charged = 0
    events: list[dict[str, Any]] = []
    seen_domains = set()
    extra_domain: frozenset[BoundaryMeasurementKey] = frozenset()
    local_repairs = []
    limit_index = 0
    last_failure = None
    while limit_index < len(PROPOSAL_PATH_LIMITS):
        limit = PROPOSAL_PATH_LIMITS[limit_index]
        domain = pool.domain(limit) | extra_domain
        if domain in seen_domains:
            limit_index += 1
            continue
        seen_domains.add(domain)
        remaining = maximum_proof_calls - charged
        if remaining <= 0:
            break
        event = {
            "contract": CONTRACT, "proposal_path_count": min(limit, len(pool.paths)),
            "boundary_pair_count": len(domain), "full_boundary_pair_count": pool.full_boundary_pair_count,
            "remaining_proof_calls": remaining, "final_media_authority": False,
            "raw_frame_visual_limits_used_as_final_authority": False,
            "local_repair_round_count": len(local_repairs),
            "source_continuation_pair_count": sum(key.outgoing.source_id == key.incoming.source_id for key in domain),
        }
        if report is not None:
            report({**event, "phase": "proposal_domain_start"})
        try:
            result = select(boundary_search_domain=domain, maximum_proof_materialization_calls=remaining)
        except NoCompleteTimelinePath as exc:
            used = exc.rejection_counts.get("proof_materialization_call_count", 0)
            if isinstance(used, bool) or not isinstance(used, int) or not 0 <= used <= remaining:
                raise RuntimeError("failed proposal selection lost its proof budget") from exc
            charged += used
            last_failure = exc
            event.update(status="no_complete_passing_path", proof_calls=used, failure=exc.to_dict())
            events.append(event)
            if report is not None:
                report({**event, "phase": "proposal_domain_exhausted"})
            repair = (
                add_local_proposal_detour(pool, domain, exc)
                if len(local_repairs) < MAXIMUM_LOCAL_PROPOSAL_REPAIRS
                else None
            )
            if repair is not None:
                widened, repair_audit = repair
                extra_domain |= widened - domain
                local_repairs.append(repair_audit)
                if report is not None:
                    report({"contract": CONTRACT, "phase": "proposal_local_detour_added", **repair_audit})
            else:
                limit_index += 1
            continue
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
                "maximum_local_repair_rounds": MAXIMUM_LOCAL_PROPOSAL_REPAIRS,
                "exhaustive_global_optimum_claimed": False,
                "quality_gates_waived": False, "hard_no_goods_authored": False,
            },
        )
        return replace(result, selection_retry_audit=audit)
    raise NoCompleteTimelinePath(
        "bounded descriptor proposals have no complete pixel-validated timeline",
        reason_code="joint_proposal_domains_exhausted",
        rejection_counts={"proposal_domain_count": len(events), "local_repair_round_count": len(local_repairs),
                          "proof_materialization_call_count": charged},
        profile_failures=(("last_proposal_domain", last_failure),) if last_failure is not None else (),
    )
