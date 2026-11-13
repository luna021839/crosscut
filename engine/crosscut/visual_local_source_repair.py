"""Failure-driven local source-path repair for automatic GROUP edits.

The global visual optimizer intentionally retains whole-song alternatives.  A
small Hamming difference elsewhere in the song can therefore satisfy its
diversity objective while every retained path still uses the same unsafe source
edge in one local window.  This module creates new *source* alternatives after
that failure is observed.  It never changes a continuity threshold and it never
claims that copied transition metadata is authoritative: every proposal must be
re-measured by :func:`repair_visual_timeline_boundaries` and pass the complete
timeline and decoded-output gates in ``pipeline.py``.

Only complete source runs from another optimizer candidate are transplanted.
The donor does not need a post-repair review because every newly created splice
edge is re-measured downstream; only the failed base candidates need reviewed
boundary evidence.  Single-frame or arbitrary per-segment mosaics are not
generated.  That keeps the proposal space bounded and preserves the donor
camera's natural shot continuity inside the transplanted interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import Counter
import math
from typing import Any, Sequence

from .timeline_review import TimelineReview, review_timeline
from .visual_score import VisualTimelineChoice, VisualTimelinePlan


LOCAL_SOURCE_REPAIR_CONTRACT = (
    "failure_driven_complete_source_run_recombination_v1"
)

# Every allowed issue is localized to one or more exact edit boundaries.  A
# source substitution cannot repair global cadence, coverage, duration, or
# dominance failures, so those candidates are excluded before proposal work.
LOCAL_SOURCE_REPAIR_FAILURE_CODES = frozenset(
    {
        "scene_cut_risk_boundary",
        "outgoing_boundary_side_instability",
        "incoming_boundary_side_instability",
        "scale_mismatch_exceeded",
        "focus_mismatch_exceeded",
        "layout_mismatch_exceeded",
        "silhouette_mismatch_exceeded",
        "formation_mismatch_exceeded",
        "pose_mismatch_exceeded",
    }
)


def _source_shot_signature(
    plan: VisualTimelinePlan,
) -> tuple[tuple[int, str], ...]:
    return tuple((item.source_index, item.shot_id) for item in plan.choices)


def _canonical_span_signature(
    plan: VisualTimelinePlan,
) -> tuple[tuple[int, float, float], ...]:
    return tuple(
        (
            item.segment_index,
            round(float(item.canonical_master_start), 7),
            round(float(item.canonical_master_end), 7),
        )
        for item in plan.choices
    )


def _repairable_boundary_indices(review: TimelineReview) -> tuple[int, ...]:
    if review.passed or not review.hard_failures:
        return ()
    boundaries: set[int] = set()
    for issue in review.hard_failures:
        if (
            issue.code not in LOCAL_SOURCE_REPAIR_FAILURE_CODES
            or not issue.occurrences
        ):
            return ()
        boundaries.update(int(value) for value in issue.occurrences)
    if any(value < 0 or value >= review.boundary_count for value in boundaries):
        raise ValueError("local source repair review has an invalid boundary index")
    return tuple(sorted(boundaries))


@dataclass(frozen=True, slots=True)
class LocalSourceRepairSeed:
    """One failed fully-derived candidate plus its untouched optimizer plan."""

    candidate_index: int
    review_rank: int
    cadence_profile: str
    visual_plan: VisualTimelinePlan = field(repr=False)
    post_repair_review: TimelineReview = field(repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.candidate_index, bool)
            or not isinstance(self.candidate_index, int)
            or self.candidate_index < 0
        ):
            raise ValueError("local source repair candidate index is invalid")
        if (
            isinstance(self.review_rank, bool)
            or not isinstance(self.review_rank, int)
            or self.review_rank < 0
        ):
            raise ValueError("local source repair review rank is invalid")
        if not self.cadence_profile.strip():
            raise ValueError("local source repair cadence profile cannot be blank")
        if not isinstance(self.visual_plan, VisualTimelinePlan):
            raise TypeError("local source repair visual plan is invalid")
        if not isinstance(self.post_repair_review, TimelineReview):
            raise TypeError("local source repair review is invalid")
        if not self.visual_plan.choices:
            raise ValueError("local source repair visual plan cannot be empty")

    @property
    def failure_boundary_indices(self) -> tuple[int, ...]:
        return _repairable_boundary_indices(self.post_repair_review)


@dataclass(frozen=True, slots=True)
class LocalSourceRepairDonor:
    """One optimizer plan whose complete source runs may be transplanted.

    A donor intentionally carries no post-repair review.  Its existing splice
    metadata is not trusted, and the production caller must obtain fresh exact
    boundary evidence for every cross-source edge created by a proposal.
    """

    candidate_index: int
    review_rank: int
    cadence_profile: str
    visual_plan: VisualTimelinePlan = field(repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.candidate_index, bool)
            or not isinstance(self.candidate_index, int)
            or self.candidate_index < 0
        ):
            raise ValueError("local source repair donor candidate index is invalid")
        if (
            isinstance(self.review_rank, bool)
            or not isinstance(self.review_rank, int)
            or self.review_rank < 0
        ):
            raise ValueError("local source repair donor review rank is invalid")
        if not self.cadence_profile.strip():
            raise ValueError("local source repair donor cadence profile cannot be blank")
        if not isinstance(self.visual_plan, VisualTimelinePlan):
            raise TypeError("local source repair donor visual plan is invalid")
        if not self.visual_plan.choices:
            raise ValueError("local source repair donor visual plan cannot be empty")


def _donor_from_seed(seed: LocalSourceRepairSeed) -> LocalSourceRepairDonor:
    return LocalSourceRepairDonor(
        candidate_index=seed.candidate_index,
        review_rank=seed.review_rank,
        cadence_profile=seed.cadence_profile,
        visual_plan=seed.visual_plan,
    )


def rank_divergent_local_source_repair_donors(
    seeds: Sequence[LocalSourceRepairSeed],
    donors: Sequence[LocalSourceRepairDonor],
    *,
    minimum_failure_support: int = 2,
    window_radius_segments: int = 4,
) -> tuple[LocalSourceRepairDonor, ...]:
    """Rank whole donor plans by novelty at recurring failed boundaries.

    Only reviewed seed failures establish the recurring boundary set.  Donors
    remain deliberately unreviewed: this ranking is a cheap source-path
    diversity signal, not boundary-quality evidence.  A novel exact source
    local-window divergence is only tie-breaking evidence.  A donor must
    introduce a new exact source edge at *every* recurring failure boundary;
    an unchanged outgoing/incoming source pair cannot repair the observed
    geometry failure even if a nearby choice differs.  Donors whose source path
    differs only elsewhere remain for ordinary recovery and are not returned.
    Full source-and-shot paths that already exist in the seed set are excluded.

    Duplicate donor paths are represented by their deterministic earliest
    ``(review_rank, candidate_index)`` value so downstream work cannot spend
    twice on the same source path.
    """

    if (
        isinstance(minimum_failure_support, bool)
        or not isinstance(minimum_failure_support, int)
        or minimum_failure_support < 1
    ):
        raise ValueError("local source repair failure support is invalid")
    if (
        isinstance(window_radius_segments, bool)
        or not isinstance(window_radius_segments, int)
        or not 0 <= window_radius_segments <= 12
    ):
        raise ValueError("local source repair donor window radius is invalid")
    if any(not isinstance(seed, LocalSourceRepairSeed) for seed in seeds):
        raise TypeError("local source repair seed pool is invalid")
    if any(not isinstance(donor, LocalSourceRepairDonor) for donor in donors):
        raise TypeError("local source repair donor pool is invalid")

    reviewed = tuple(
        sorted(
            (seed for seed in seeds if seed.failure_boundary_indices),
            key=lambda seed: (
                seed.review_rank,
                seed.candidate_index,
                seed.cadence_profile,
                _source_shot_signature(seed.visual_plan),
            ),
        )
    )
    if not reviewed:
        return ()
    reference_spans = _canonical_span_signature(reviewed[0].visual_plan)
    reviewed = tuple(
        seed
        for seed in reviewed
        if _canonical_span_signature(seed.visual_plan) == reference_spans
    )
    failure_support = Counter(
        boundary_index
        for seed in reviewed
        for boundary_index in seed.failure_boundary_indices
    )
    recurring_boundaries = tuple(
        sorted(
            boundary_index
            for boundary_index, support in failure_support.items()
            if support >= minimum_failure_support
        )
    )
    if not recurring_boundaries:
        return ()

    seed_signatures = {
        _source_shot_signature(seed.visual_plan) for seed in reviewed
    }
    unique_donors: dict[
        tuple[tuple[int, str], ...], LocalSourceRepairDonor
    ] = {}
    for donor in sorted(
        donors,
        key=lambda value: (
            value.review_rank,
            value.candidate_index,
            value.cadence_profile,
            _source_shot_signature(value.visual_plan),
        ),
    ):
        if _canonical_span_signature(donor.visual_plan) != reference_spans:
            continue
        signature = _source_shot_signature(donor.visual_plan)
        if signature in seed_signatures:
            continue
        unique_donors.setdefault(signature, donor)

    ranked: list[
        tuple[tuple[int, int, int, int, int, int], LocalSourceRepairDonor]
    ] = []
    for donor in unique_donors.values():
        donor_choices = donor.visual_plan.choices
        novel_edge_count = 0
        nearest_edge_divergence = 0
        novel_window_count = 0
        nearest_window_divergence = 0
        pairwise_edge_divergence = 0
        pairwise_window_divergence = 0
        for boundary_index in recurring_boundaries:
            supporting_seeds = tuple(
                seed
                for seed in reviewed
                if boundary_index in seed.failure_boundary_indices
            )
            donor_edge = (
                donor_choices[boundary_index].source_index,
                donor_choices[boundary_index + 1].source_index,
            )
            window_start = max(
                0,
                boundary_index - window_radius_segments,
            )
            window_end = min(
                len(donor_choices),
                boundary_index + window_radius_segments + 2,
            )
            donor_window = tuple(
                item.source_index
                for item in donor_choices[window_start:window_end]
            )
            seed_edges = tuple(
                (
                    seed.visual_plan.choices[boundary_index].source_index,
                    seed.visual_plan.choices[boundary_index + 1].source_index,
                )
                for seed in supporting_seeds
            )
            seed_windows = tuple(
                tuple(
                    item.source_index
                    for item in seed.visual_plan.choices[
                        window_start:window_end
                    ]
                )
                for seed in supporting_seeds
            )
            edge_distances = tuple(
                sum(
                    actual != expected
                    for actual, expected in zip(
                        donor_edge,
                        seed_edge,
                        strict=True,
                    )
                )
                for seed_edge in seed_edges
            )
            window_distances = tuple(
                sum(
                    actual != expected
                    for actual, expected in zip(
                        donor_window,
                        seed_window,
                        strict=True,
                    )
                )
                for seed_window in seed_windows
            )
            edge_is_novel = donor_edge not in set(seed_edges)
            window_is_novel = donor_window not in set(seed_windows)
            novel_edge_count += int(edge_is_novel)
            nearest_edge_divergence += min(edge_distances)
            novel_window_count += int(window_is_novel)
            nearest_window_divergence += min(window_distances)
            pairwise_edge_divergence += sum(edge_distances)
            pairwise_window_divergence += sum(window_distances)
        # Promotion is intentionally narrower than ordinary candidate
        # recovery.  A donor must offer a new exact source edge at every
        # recurring geometry-failure boundary; local-window novelty alone
        # cannot change that boundary's outgoing/incoming source geometry.
        if novel_edge_count != len(recurring_boundaries):
            continue
        ranked.append(
            (
                (
                    novel_edge_count,
                    nearest_edge_divergence,
                    novel_window_count,
                    nearest_window_divergence,
                    pairwise_edge_divergence,
                    pairwise_window_divergence,
                ),
                donor,
            )
        )

    ranked.sort(
        key=lambda item: (
            tuple(-value for value in item[0]),
            item[1].review_rank,
            item[1].candidate_index,
            item[1].cadence_profile,
            _source_shot_signature(item[1].visual_plan),
        )
    )
    return tuple(donor for _, donor in ranked)


@dataclass(frozen=True, slots=True)
class LocalSourceRepairCandidateRecord:
    optimizer_score: float
    review: TimelineReview = field(repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.optimizer_score)):
            raise ValueError("local source repair optimizer score must be finite")
        if not isinstance(self.review, TimelineReview):
            raise TypeError("local source repair pre-review is invalid")


@dataclass(frozen=True, slots=True)
class LocalSourceRepairMaterializedCandidate:
    """Pipeline-compatible proposal awaiting exact decoded boundary repair."""

    candidate_index: int
    review_rank: int
    cadence_profile: str
    visual_plan: VisualTimelinePlan = field(repr=False)
    candidate_record: LocalSourceRepairCandidateRecord = field(repr=False)
    base_candidate_index: int
    donor_candidate_index: int
    replaced_segment_indices: tuple[int, ...]
    touched_boundary_indices: tuple[int, ...]
    source_substitution_count: int
    generation_contract: str = LOCAL_SOURCE_REPAIR_CONTRACT

    def __post_init__(self) -> None:
        if self.visual_plan.selected_candidate_index != self.candidate_index:
            raise ValueError("local source repair selected candidate index drifted")
        replaced = tuple(int(value) for value in self.replaced_segment_indices)
        touched = tuple(int(value) for value in self.touched_boundary_indices)
        if not replaced or replaced != tuple(sorted(set(replaced))):
            raise ValueError("local source repair replacement indices are invalid")
        if not touched or touched != tuple(sorted(set(touched))):
            raise ValueError("local source repair boundary indices are invalid")
        if self.source_substitution_count != len(replaced):
            raise ValueError("local source repair substitution count drifted")
        if any(not 0 <= value < len(self.visual_plan.choices) for value in replaced):
            raise ValueError("local source repair replacement is outside the plan")
        if any(
            not 0 <= value < len(self.visual_plan.choices) - 1
            for value in touched
        ):
            raise ValueError("local source repair boundary is outside the plan")
        if self.generation_contract != LOCAL_SOURCE_REPAIR_CONTRACT:
            raise ValueError("local source repair generation contract changed")

    def lineage_payload(self) -> dict[str, Any]:
        return {
            "stage": "local_source_window_recombination",
            "contract": self.generation_contract,
            "base_candidate_index": self.base_candidate_index,
            "donor_candidate_index": self.donor_candidate_index,
            "replaced_segment_indices": list(self.replaced_segment_indices),
            "touched_boundary_indices": list(self.touched_boundary_indices),
            "source_substitution_count": self.source_substitution_count,
            "complete_source_runs_only": True,
            "transition_thresholds_lowered": False,
            "full_post_repair_timeline_review_required": True,
            "decoded_proxy_review_required": True,
        }


def _source_runs(
    choices: Sequence[VisualTimelineChoice],
) -> tuple[tuple[int, int], ...]:
    if not choices:
        return ()
    runs: list[tuple[int, int]] = []
    first = 0
    for index in range(1, len(choices)):
        if choices[index].source_index == choices[index - 1].source_index:
            continue
        runs.append((first, index - 1))
        first = index
    runs.append((first, len(choices) - 1))
    return tuple(runs)


def _touching_boundaries(
    replaced_indices: Sequence[int],
    *,
    segment_count: int,
) -> tuple[int, ...]:
    touched = {
        boundary
        for index in replaced_indices
        for boundary in (index - 1, index)
        if 0 <= boundary < segment_count - 1
    }
    return tuple(sorted(touched))


def _choice_identity(
    choice: VisualTimelineChoice,
) -> tuple[int, str]:
    """Return the persisted identity used by source-path de-duplication."""

    return (choice.source_index, choice.shot_id)


def _run_failure_coverage(
    *,
    run: tuple[int, ...],
    base_choices: Sequence[VisualTimelineChoice],
    donor_choices: Sequence[VisualTimelineChoice],
    failure_boundaries: frozenset[int],
) -> frozenset[int]:
    """Find failed edges whose final choice identity changes for one run."""

    first = run[0]
    last = run[-1]
    covered: set[int] = set()
    for boundary_index in failure_boundaries:
        if boundary_index < first - 1 or boundary_index > last:
            continue
        outgoing = (
            donor_choices[boundary_index]
            if first <= boundary_index <= last
            else base_choices[boundary_index]
        )
        incoming = (
            donor_choices[boundary_index + 1]
            if first <= boundary_index + 1 <= last
            else base_choices[boundary_index + 1]
        )
        if (
            _choice_identity(outgoing),
            _choice_identity(incoming),
        ) != (
            _choice_identity(base_choices[boundary_index]),
            _choice_identity(base_choices[boundary_index + 1]),
        ):
            covered.add(boundary_index)
    return frozenset(covered)


def _minimal_failure_covering_run_groups(
    run_values: Sequence[tuple[tuple[int, ...], frozenset[int]]],
    *,
    required_failures: frozenset[int],
    maximum_runs: int,
    maximum_groups: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Enumerate bounded minimum-cardinality complete-run set covers.

    Runs with no failure-edge identity effect are removed by the caller.  The
    search branches on the least-supported uncovered failure and stops at the
    first run count that can cover the complete failure set.  This avoids the
    ``n choose k`` expansion that a blind cross-window combination would cause.
    """

    if not required_failures or not run_values:
        return ()
    coverage_by_run = tuple(value[1] for value in run_values)
    runs_by_failure = {
        failure: tuple(
            sorted(
                (
                    run_index
                    for run_index, coverage in enumerate(coverage_by_run)
                    if failure in coverage
                ),
                key=lambda run_index: (
                    -len(coverage_by_run[run_index]),
                    len(run_values[run_index][0]),
                    run_values[run_index][0],
                ),
            )
        )
        for failure in sorted(required_failures)
    }
    if any(not values for values in runs_by_failure.values()):
        return ()

    maximum_single_run_coverage = max(map(len, coverage_by_run))
    minimum_run_count = math.ceil(
        len(required_failures) / maximum_single_run_coverage
    )
    for run_limit in range(minimum_run_count, maximum_runs + 1):
        solutions: set[tuple[int, ...]] = set()
        visited: set[tuple[int, ...]] = set()

        def search(
            selected: tuple[int, ...],
            covered: frozenset[int],
        ) -> None:
            if len(solutions) >= maximum_groups:
                return
            if required_failures.issubset(covered):
                solutions.add(selected)
                return
            if len(selected) >= run_limit or selected in visited:
                return
            visited.add(selected)

            uncovered = required_failures.difference(covered)
            remaining_slots = run_limit - len(selected)
            available = tuple(
                run_index
                for run_index in range(len(run_values))
                if run_index not in selected
                and coverage_by_run[run_index].intersection(uncovered)
            )
            if not available:
                return
            possible_coverage: set[int] = set()
            for run_index in available:
                possible_coverage.update(coverage_by_run[run_index])
            if not uncovered.issubset(possible_coverage):
                return
            largest_new_coverages = sorted(
                (
                    len(coverage_by_run[run_index].intersection(uncovered))
                    for run_index in available
                ),
                reverse=True,
            )
            if sum(largest_new_coverages[:remaining_slots]) < len(uncovered):
                return

            target_failure = min(
                uncovered,
                key=lambda failure: (
                    sum(
                        run_index in available
                        for run_index in runs_by_failure[failure]
                    ),
                    failure,
                ),
            )
            for run_index in runs_by_failure[target_failure]:
                if run_index not in available:
                    continue
                next_selected = tuple(sorted((*selected, run_index)))
                search(
                    next_selected,
                    frozenset(
                        covered.union(coverage_by_run[run_index])
                    ),
                )

        search((), frozenset())
        if solutions:
            ordered = sorted(
                solutions,
                key=lambda selected: (
                    sum(len(run_values[index][0]) for index in selected),
                    tuple(run_values[index][0] for index in selected),
                ),
            )
            return tuple(
                tuple(run_values[index][0] for index in selected)
                for selected in ordered[:maximum_groups]
            )
    return ()


def _review_summary(
    *,
    candidate_index: int,
    review: TimelineReview,
    optimizer_score: float,
) -> dict[str, Any]:
    return {
        "candidate_index": candidate_index,
        "cadence_profile": "local_source_window_recombination",
        "selected": True,
        "optimizer_score": float(optimizer_score),
        "passed": review.passed,
        "weighted_score": review.weighted_score,
        "hard_failure_codes": [item.code for item in review.hard_failures],
        "warning_codes": [item.code for item in review.warnings],
        "source_switch_count": review.source_switch_count,
        "source_switch_rate_on_visual_boundaries": review.source_switch_rate,
        "source_switches_per_minute": review.source_switches_per_minute,
        "comparable_visual_boundary_count": (
            review.comparable_visual_boundary_count
        ),
        "unique_source_count": review.unique_source_count,
        "longest_source_run_seconds": review.longest_source_run_seconds,
        "source_run_median_seconds": review.source_run_median_seconds,
        "visual_shot_median_seconds": review.visual_shot_median_seconds,
        "ping_pong_count": len(review.ping_pongs),
        "pre_review_transition_metadata_authoritative": False,
        "exact_boundary_remeasurement_required": True,
    }


def build_local_source_repair_proposals(
    seeds: Sequence[LocalSourceRepairSeed],
    *,
    donors: Sequence[LocalSourceRepairDonor] | None = None,
    available_source_count: int,
    first_candidate_index: int,
    maximum_proposals: int = 64,
    window_radius_segments: int = 4,
    maximum_donor_runs_per_proposal: int = 4,
) -> tuple[LocalSourceRepairMaterializedCandidate, ...]:
    """Recombine complete donor runs across all observed failure windows.

    The function is metadata-only.  Its pre-review intentionally carries stale
    splice-edge metrics and is therefore advisory.  The caller must run exact
    frame boundary repair over the complete plan before counting a proposal.
    """

    if available_source_count < 2:
        raise ValueError("local source repair needs at least two sources")
    if first_candidate_index < 0:
        raise ValueError("first local source repair candidate index is invalid")
    if not 1 <= maximum_proposals <= 256:
        raise ValueError("maximum local source repair proposals is invalid")
    if not 1 <= window_radius_segments <= 12:
        raise ValueError("local source repair window radius is invalid")
    if (
        isinstance(maximum_donor_runs_per_proposal, bool)
        or not isinstance(maximum_donor_runs_per_proposal, int)
        or not 1 <= maximum_donor_runs_per_proposal <= 6
    ):
        raise ValueError("local source repair donor run bound is invalid")

    compatible = tuple(
        sorted(
            (seed for seed in seeds if seed.failure_boundary_indices),
            key=lambda seed: (
                seed.review_rank,
                seed.candidate_index,
                seed.cadence_profile,
                _source_shot_signature(seed.visual_plan),
            ),
        )
    )
    if not compatible:
        return ()
    reference_spans = _canonical_span_signature(compatible[0].visual_plan)
    compatible = tuple(
        seed
        for seed in compatible
        if _canonical_span_signature(seed.visual_plan) == reference_spans
    )
    if not compatible:
        return ()

    if donors is None:
        donor_values = tuple(_donor_from_seed(seed) for seed in compatible)
    else:
        if any(not isinstance(donor, LocalSourceRepairDonor) for donor in donors):
            raise TypeError("local source repair donor pool is invalid")
        donor_values = tuple(donors)
    donor_values = tuple(
        sorted(
            (
                donor
                for donor in donor_values
                if _canonical_span_signature(donor.visual_plan) == reference_spans
            ),
            key=lambda donor: (
                donor.review_rank,
                donor.candidate_index,
                donor.cadence_profile,
                _source_shot_signature(donor.visual_plan),
            ),
        )
    )
    if not donor_values:
        return ()

    existing_signatures = {
        _source_shot_signature(seed.visual_plan) for seed in seeds
    }
    existing_signatures.update(
        _source_shot_signature(donor.visual_plan) for donor in donor_values
    )
    known_donor_failures = {
        (seed.candidate_index, _source_shot_signature(seed.visual_plan)): (
            seed.failure_boundary_indices
        )
        for seed in compatible
    }
    failure_frequency = Counter(
        boundary_index
        for seed in compatible
        for boundary_index in seed.failure_boundary_indices
    )
    proposal_values: list[
        tuple[
            tuple[float, ...],
            LocalSourceRepairSeed,
            LocalSourceRepairDonor,
            tuple[int, ...],
            VisualTimelinePlan,
            TimelineReview,
        ]
    ] = []
    proposal_signatures: set[tuple[tuple[int, str], ...]] = set()
    for base in compatible:
        base_failures = frozenset(base.failure_boundary_indices)
        base_choices = base.visual_plan.choices
        base_failure_occurrence_count = sum(
            len(issue.occurrences)
            for issue in base.post_repair_review.hard_failures
        )
        failure_windows = tuple(
            (
                max(0, boundary_index - window_radius_segments),
                min(
                    len(base_choices) - 1,
                    boundary_index + window_radius_segments + 1,
                ),
            )
            for boundary_index in sorted(base_failures)
        )
        for donor in donor_values:
            if donor.candidate_index == base.candidate_index:
                continue
            donor_choices = donor.visual_plan.choices
            run_values: list[tuple[tuple[int, ...], frozenset[int]]] = []
            for first, last in _source_runs(donor_choices):
                if not any(
                    first <= window_end and last >= window_start
                    for window_start, window_end in failure_windows
                ):
                    continue
                run = tuple(range(first, last + 1))
                # Preserve complete natural donor runs and reject a generated
                # one-segment flash before the set-cover search.
                if len(run) < 2 or not any(
                    base_choices[index].source_index
                    != donor_choices[index].source_index
                    for index in run
                ):
                    continue
                coverage = _run_failure_coverage(
                    run=run,
                    base_choices=base_choices,
                    donor_choices=donor_choices,
                    failure_boundaries=base_failures,
                )
                if coverage:
                    run_values.append((run, coverage))
            selected_run_groups = _minimal_failure_covering_run_groups(
                tuple(run_values),
                required_failures=base_failures,
                maximum_runs=maximum_donor_runs_per_proposal,
                maximum_groups=maximum_proposals,
            )
            for selected_runs in selected_run_groups:
                replaced = tuple(
                    sorted({index for run in selected_runs for index in run})
                )
                touched = _touching_boundaries(
                    replaced,
                    segment_count=len(base_choices),
                )
                if not base_failures.issubset(touched):
                    continue
                choices = list(base_choices)
                for index in replaced:
                    choices[index] = donor_choices[index]
                # Merely touching a failed boundary is insufficient.  At
                # least one side of every exact failed edge must be a genuinely
                # different persisted source-and-shot choice.
                if any(
                    (
                        _choice_identity(choices[boundary_index]),
                        _choice_identity(choices[boundary_index + 1]),
                    )
                    == (
                        _choice_identity(base_choices[boundary_index]),
                        _choice_identity(base_choices[boundary_index + 1]),
                    )
                    for boundary_index in base_failures
                ):
                    continue
                provisional = VisualTimelinePlan(
                    choices=tuple(choices),
                    optimizer_score=base.visual_plan.optimizer_score,
                    analysis_basis=(
                        f"{base.visual_plan.analysis_basis}+"
                        "local_source_window_recombination_v1"
                    ),
                    selected_candidate_index=0,
                    candidate_generation_basis=(
                        f"{base.visual_plan.candidate_generation_basis}+"
                        f"{LOCAL_SOURCE_REPAIR_CONTRACT}"
                    ),
                    candidate_review_summaries=(),
                    source_clock_basis=base.visual_plan.source_clock_basis,
                )
                signature = _source_shot_signature(provisional)
                if (
                    signature in existing_signatures
                    or signature in proposal_signatures
                ):
                    continue
                proposal_signatures.add(signature)
                advisory_review = review_timeline(
                    provisional,
                    available_source_count=available_source_count,
                )
                nonlocal_failure_count = sum(
                    issue.code not in LOCAL_SOURCE_REPAIR_FAILURE_CODES
                    or not issue.occurrences
                    for issue in advisory_review.hard_failures
                )
                if nonlocal_failure_count > 0:
                    continue
                advisory_failure_occurrence_count = sum(
                    max(1, len(issue.occurrences))
                    for issue in advisory_review.hard_failures
                )
                donor_failure_touch_count = len(
                    set(
                        known_donor_failures.get(
                            (
                                donor.candidate_index,
                                _source_shot_signature(donor.visual_plan),
                            ),
                            (),
                        )
                    ).intersection(touched)
                )
                priority = (
                    -float(base_failure_occurrence_count),
                    float(
                        sum(
                            failure_frequency[boundary_index]
                            for boundary_index in base_failures
                        )
                    ),
                    -float(donor_failure_touch_count),
                    -float(advisory_failure_occurrence_count),
                    float(base.post_repair_review.weighted_score),
                    -float(len(selected_runs)),
                    -float(len(replaced)),
                    -float(base.review_rank),
                    -float(donor.review_rank),
                )
                proposal_values.append(
                    (
                        priority,
                        base,
                        donor,
                        replaced,
                        provisional,
                        advisory_review,
                    )
                )

    proposal_values.sort(
        key=lambda item: (
            tuple(-value for value in item[0]),
            item[1].review_rank,
            item[1].candidate_index,
            item[2].review_rank,
            item[2].candidate_index,
            item[3],
        )
    )
    proposals: list[LocalSourceRepairMaterializedCandidate] = []
    for ordinal, (
        _,
        base,
        donor,
        replaced,
        provisional,
        advisory_review,
    ) in enumerate(
        proposal_values[:maximum_proposals]
    ):
        candidate_index = first_candidate_index + ordinal
        indexed = replace(
            provisional,
            selected_candidate_index=candidate_index,
        )
        plan = replace(
            indexed,
            candidate_review_summaries=(
                _review_summary(
                    candidate_index=candidate_index,
                    review=advisory_review,
                    optimizer_score=indexed.optimizer_score,
                ),
            ),
        )
        proposals.append(
            LocalSourceRepairMaterializedCandidate(
                candidate_index=candidate_index,
                review_rank=candidate_index,
                cadence_profile="local_source_window_recombination",
                visual_plan=plan,
                candidate_record=LocalSourceRepairCandidateRecord(
                    optimizer_score=plan.optimizer_score,
                    review=advisory_review,
                ),
                base_candidate_index=base.candidate_index,
                donor_candidate_index=donor.candidate_index,
                replaced_segment_indices=replaced,
                touched_boundary_indices=_touching_boundaries(
                    replaced,
                    segment_count=len(plan.choices),
                ),
                source_substitution_count=len(replaced),
            )
        )
    return tuple(proposals)


__all__ = [
    "LOCAL_SOURCE_REPAIR_CONTRACT",
    "LOCAL_SOURCE_REPAIR_FAILURE_CODES",
    "LocalSourceRepairDonor",
    "LocalSourceRepairMaterializedCandidate",
    "LocalSourceRepairSeed",
    "build_local_source_repair_proposals",
    "rank_divergent_local_source_repair_donors",
]
