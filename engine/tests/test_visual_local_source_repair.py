from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

from crosscut.pipeline import (
    _adaptive_post_repair_candidate_boundary_indices,
    _candidate_has_material_source_or_timing_repair,
    _local_source_repair_boundary_evidence,
)
from crosscut.timeline import ShotType
from crosscut.timeline_review import review_timeline
from crosscut.visual_local_source_repair import (
    LOCAL_SOURCE_REPAIR_CONTRACT,
    LOCAL_SOURCE_REPAIR_FAILURE_CODES,
    LocalSourceRepairDonor,
    LocalSourceRepairSeed,
    build_local_source_repair_proposals,
    rank_divergent_local_source_repair_donors,
)
from crosscut.visual_score import VisualTimelineChoice, VisualTimelinePlan


_FACE = (0.40, 0.22, 0.20, 0.30)
_SHIFTED_FACE = (0.72, 0.22, 0.20, 0.30)


def _choice(index: int, source_index: int) -> VisualTimelineChoice:
    duration = 1.2244767444324
    start = index * duration
    return VisualTimelineChoice(
        segment_index=index,
        source_index=source_index,
        canonical_master_start=start,
        canonical_master_end=start + duration,
        source_local_start=start,
        source_local_end=start + duration,
        quality_score=0.92,
        composition_score=0.82,
        boundary_score=0.84,
        shot_scale=0.42,
        shot_type=ShotType.GROUP_WIDE,
        shot_id=f"{source_index}:shot-{index // 3}",
        entry_shot_scale=0.42,
        exit_shot_scale=0.42,
        entry_shot_type=ShotType.GROUP_WIDE,
        exit_shot_type=ShotType.GROUP_WIDE,
        face_samples_xywh=(_FACE, _FACE, _FACE),
        layout_transition_mismatch_from_previous=0.08,
        pose_transition_basis_from_previous="opencv_zoo_mediapipe_pose",
        pose_transition_mismatch_from_previous=0.08,
        silhouette_transition_mismatch_from_previous=0.08,
        formation_transition_mismatch_from_previous=0.08,
    )


def _plan(
    *,
    candidate_index: int,
    local_sources: tuple[int, ...],
    failure: str,
) -> VisualTimelinePlan:
    sources = [(index // 3) % 8 for index in range(120)]
    sources[93:105] = list(local_sources)
    choices = [_choice(index, source) for index, source in enumerate(sources)]
    if failure == "formation":
        choices[101] = replace(
            choices[101],
            formation_transition_mismatch_from_previous=0.4358238167575896,
        )
    elif failure == "focus":
        choices[97] = replace(
            choices[97],
            face_samples_xywh=(_SHIFTED_FACE, _SHIFTED_FACE, _SHIFTED_FACE),
        )
    else:  # pragma: no cover - fixture invariant
        raise AssertionError(failure)
    return VisualTimelinePlan(
        choices=tuple(choices),
        optimizer_score=100.0 - candidate_index,
        selected_candidate_index=candidate_index,
        candidate_review_summaries=(),
    )


def _seed(
    *,
    candidate_index: int,
    local_sources: tuple[int, ...],
    failure: str,
) -> LocalSourceRepairSeed:
    plan = _plan(
        candidate_index=candidate_index,
        local_sources=local_sources,
        failure=failure,
    )
    review = review_timeline(plan, available_source_count=8)
    return LocalSourceRepairSeed(
        candidate_index=candidate_index,
        review_rank=candidate_index,
        cadence_profile="balanced",
        visual_plan=plan,
        post_repair_review=review,
    )


def _donor(
    *,
    candidate_index: int,
    local_sources: tuple[int, ...],
    failure: str,
) -> LocalSourceRepairDonor:
    return LocalSourceRepairDonor(
        candidate_index=candidate_index,
        review_rank=candidate_index,
        cadence_profile="balanced",
        visual_plan=_plan(
            candidate_index=candidate_index,
            local_sources=local_sources,
            failure=failure,
        ),
    )


def _donor_with_source_override(
    *,
    candidate_index: int,
    local_sources: tuple[int, ...],
    override_index: int,
    override_source: int,
) -> LocalSourceRepairDonor:
    donor = _donor(
        candidate_index=candidate_index,
        local_sources=local_sources,
        failure="formation",
    )
    choices = list(donor.visual_plan.choices)
    choices[override_index] = _choice(override_index, override_source)
    return replace(
        donor,
        visual_plan=replace(donor.visual_plan, choices=tuple(choices)),
    )


def _source_path_with_runs(
    runs: tuple[tuple[int, int, int], ...],
) -> tuple[int, ...]:
    sources = [(index // 3) % 8 for index in range(120)]
    for first, last, source_index in runs:
        sources[first : last + 1] = [source_index] * (last - first + 1)
    return tuple(sources)


def _plan_with_failure_boundaries(
    *,
    candidate_index: int,
    sources: tuple[int, ...],
    failure_boundaries: tuple[int, ...] = (),
) -> VisualTimelinePlan:
    choices = [_choice(index, source) for index, source in enumerate(sources)]
    for boundary_index in failure_boundaries:
        choices[boundary_index + 1] = replace(
            choices[boundary_index + 1],
            formation_transition_mismatch_from_previous=0.4358238167575896,
        )
    return VisualTimelinePlan(
        choices=tuple(choices),
        optimizer_score=100.0 - candidate_index,
        selected_candidate_index=candidate_index,
        candidate_review_summaries=(),
    )


def _seed_with_failure_boundaries(
    *,
    candidate_index: int,
    sources: tuple[int, ...],
    failure_boundaries: tuple[int, ...],
) -> LocalSourceRepairSeed:
    plan = _plan_with_failure_boundaries(
        candidate_index=candidate_index,
        sources=sources,
        failure_boundaries=failure_boundaries,
    )
    review = review_timeline(plan, available_source_count=8)
    assert tuple(
        occurrence
        for issue in review.hard_failures
        for occurrence in issue.occurrences
    ) == failure_boundaries
    return LocalSourceRepairSeed(
        candidate_index=candidate_index,
        review_rank=candidate_index,
        cadence_profile="balanced",
        visual_plan=plan,
        post_repair_review=review,
    )


def _donor_from_source_path(
    *,
    candidate_index: int,
    sources: tuple[int, ...],
) -> LocalSourceRepairDonor:
    return LocalSourceRepairDonor(
        candidate_index=candidate_index,
        review_rank=candidate_index,
        cadence_profile="balanced",
        visual_plan=_plan_with_failure_boundaries(
            candidate_index=candidate_index,
            sources=sources,
        ),
    )


def _xoxz_seed12_sources() -> tuple[int, ...]:
    return _source_path_with_runs(
        (
            (0, 0, 0),
            (1, 4, 5),
            (5, 5, 4),
            (6, 7, 3),
            (8, 10, 1),
            (22, 23, 7),
            (24, 26, 6),
            (27, 30, 0),
            (69, 69, 0),
            (70, 73, 4),
            (74, 77, 5),
            (78, 81, 0),
            (94, 96, 7),
            (97, 100, 3),
            (101, 103, 1),
            (104, 104, 6),
        )
    )


def _xoxz_seed13_sources() -> tuple[int, ...]:
    return _source_path_with_runs(
        (
            (0, 0, 0),
            (1, 4, 5),
            (5, 5, 4),
            (6, 7, 3),
            (8, 9, 1),
            (10, 10, 4),
            (22, 22, 0),
            (23, 26, 7),
            (27, 29, 4),
            (69, 69, 0),
            (70, 73, 5),
            (74, 75, 2),
            (76, 77, 4),
            (78, 81, 0),
            (94, 96, 7),
            (97, 100, 3),
            (101, 103, 1),
            (104, 104, 6),
        )
    )


def _xoxz_candidate30_sources() -> tuple[int, ...]:
    return _source_path_with_runs(
        (
            (0, 2, 5),
            (3, 5, 0),
            (6, 7, 7),
            (8, 10, 4),
            (11, 11, 7),
            (22, 23, 7),
            (24, 27, 0),
            (28, 30, 3),
            (69, 69, 0),
            (70, 73, 2),
            (74, 77, 4),
            (78, 81, 0),
            (94, 96, 7),
            (97, 100, 5),
            (101, 103, 3),
            (104, 104, 6),
        )
    )


def _complete_replaced_donor_run_count(
    proposal,
    donor: LocalSourceRepairDonor,
) -> int:
    sources = tuple(
        choice.source_index for choice in donor.visual_plan.choices
    )
    runs: list[tuple[int, ...]] = []
    first = 0
    for index in range(1, len(sources)):
        if sources[index] == sources[index - 1]:
            continue
        runs.append(tuple(range(first, index)))
        first = index
    runs.append(tuple(range(first, len(sources))))
    replaced = set(proposal.replaced_segment_indices)
    selected = tuple(run for run in runs if set(run).issubset(replaced))
    assert replaced == {index for run in selected for index in run}
    return len(selected)


def _assert_xoxz_seed12_uses_three_runs_across_all_failure_windows() -> None:
    failures = (5, 26, 100)
    base_sources = list(_xoxz_seed12_sources())
    # Keep the donor run covering boundary 26 from merging into an unrelated
    # seven-segment source run in the materialized composite.
    base_sources[27:31] = [3, 3, 3, 3]
    base = _seed_with_failure_boundaries(
        candidate_index=12,
        sources=tuple(base_sources),
        failure_boundaries=failures,
    )
    donor = _donor_from_source_path(
        candidate_index=30,
        sources=_xoxz_candidate30_sources(),
    )

    proposals = build_local_source_repair_proposals(
        (base,),
        donors=(donor,),
        available_source_count=8,
        first_candidate_index=120,
        maximum_proposals=16,
    )

    assert proposals
    assert all(
        _complete_replaced_donor_run_count(proposal, donor) == 3
        for proposal in proposals
    )
    for proposal in proposals:
        assert set(failures).issubset(proposal.touched_boundary_indices)
        for boundary_index in failures:
            actual = tuple(
                (
                    choice.source_index,
                    choice.shot_id,
                )
                for choice in proposal.visual_plan.choices[
                    boundary_index : boundary_index + 2
                ]
            )
            expected = tuple(
                (
                    choice.source_index,
                    choice.shot_id,
                )
                for choice in base.visual_plan.choices[
                    boundary_index : boundary_index + 2
                ]
            )
            assert actual != expected
    assert (
        build_local_source_repair_proposals(
            (base,),
            donors=(donor,),
            available_source_count=8,
            first_candidate_index=140,
            maximum_donor_runs_per_proposal=2,
        )
        == ()
    )


def _assert_xoxz_seed13_uses_four_runs_across_all_failure_windows() -> None:
    failures = (5, 9, 73, 100)
    base = _seed_with_failure_boundaries(
        candidate_index=13,
        sources=_xoxz_seed13_sources(),
        failure_boundaries=failures,
    )
    donor = _donor_from_source_path(
        candidate_index=30,
        sources=_xoxz_candidate30_sources(),
    )

    proposals = build_local_source_repair_proposals(
        (base,),
        donors=(donor,),
        available_source_count=8,
        first_candidate_index=160,
        maximum_proposals=16,
    )

    assert proposals
    assert all(
        _complete_replaced_donor_run_count(proposal, donor) == 4
        for proposal in proposals
    )
    assert all(
        set(failures).issubset(proposal.touched_boundary_indices)
        for proposal in proposals
    )
    assert (
        build_local_source_repair_proposals(
            (base,),
            donors=(donor,),
            available_source_count=8,
            first_candidate_index=180,
            maximum_donor_runs_per_proposal=2,
        )
        == ()
    )


def _assert_recombines_complete_donor_run_across_xoxz_failure_window() -> None:
    dominant = _seed(
        candidate_index=15,
        local_sources=(2, 7, 7, 7, 3, 3, 3, 3, 1, 1, 1, 6),
        failure="formation",
    )
    donor = _seed(
        candidate_index=30,
        local_sources=(2, 1, 1, 1, 5, 5, 5, 5, 3, 3, 3, 6),
        failure="focus",
    )

    assert [item.code for item in dominant.post_repair_review.hard_failures] == [
        "formation_mismatch_exceeded"
    ]
    assert dominant.failure_boundary_indices == (100,)
    assert [item.code for item in donor.post_repair_review.hard_failures] == [
        "focus_mismatch_exceeded"
    ]
    assert donor.failure_boundary_indices == (96,)

    proposals = build_local_source_repair_proposals(
        (dominant, donor),
        available_source_count=8,
        first_candidate_index=32,
        maximum_proposals=32,
    )

    hybrid = next(
        item
        for item in proposals
        if item.base_candidate_index == 15
        and item.donor_candidate_index == 30
        and item.replaced_segment_indices == (97, 98, 99, 100)
    )
    assert tuple(
        item.source_index for item in hybrid.visual_plan.choices[93:105]
    ) == (2, 7, 7, 7, 5, 5, 5, 5, 1, 1, 1, 6)
    assert hybrid.touched_boundary_indices == (96, 97, 98, 99, 100)
    assert hybrid.source_substitution_count == 4
    assert hybrid.generation_contract == LOCAL_SOURCE_REPAIR_CONTRACT
    assert hybrid.lineage_payload()["transition_thresholds_lowered"] is False
    # Transition metrics at splice edges are intentionally stale here.  The
    # production caller must re-measure exact decoded frames before admission.
    assert hybrid.candidate_record.review.passed is False


def _assert_unreviewed_donor_outside_seed_pool_can_supply_xoxz_run() -> None:
    dominant = _seed(
        candidate_index=15,
        local_sources=(2, 7, 7, 7, 3, 3, 3, 3, 1, 1, 1, 6),
        failure="formation",
    )
    donor = _donor(
        candidate_index=30,
        local_sources=(2, 1, 1, 1, 5, 5, 5, 5, 3, 3, 3, 6),
        failure="focus",
    )

    assert donor.candidate_index not in {
        seed.candidate_index for seed in (dominant,)
    }
    proposals = build_local_source_repair_proposals(
        (dominant,),
        donors=(donor,),
        available_source_count=8,
        first_candidate_index=80,
        maximum_proposals=32,
    )

    hybrid = next(
        item
        for item in proposals
        if item.base_candidate_index == 15
        and item.donor_candidate_index == 30
        and item.replaced_segment_indices == (97, 98, 99, 100)
    )
    assert tuple(
        item.source_index for item in hybrid.visual_plan.choices[93:105]
    ) == (2, 7, 7, 7, 5, 5, 5, 5, 1, 1, 1, 6)
    assert hybrid.touched_boundary_indices == (96, 97, 98, 99, 100)


def _assert_ranks_unreviewed_xoxz_donor_by_recurring_local_divergence() -> None:
    base_sources = (2, 7, 7, 7, 3, 3, 3, 3, 1, 1, 1, 6)
    first = _seed(
        candidate_index=15,
        local_sources=base_sources,
        failure="formation",
    )
    second = _seed(
        candidate_index=16,
        local_sources=base_sources,
        failure="formation",
    )
    xoxz_donor = _donor(
        candidate_index=30,
        local_sources=(2, 1, 1, 1, 5, 5, 5, 5, 3, 3, 3, 6),
        failure="focus",
    )
    differs_only_elsewhere = _donor_with_source_override(
        candidate_index=20,
        local_sources=base_sources,
        override_index=10,
        override_source=6,
    )
    same_edge_but_different_local_window = _donor_with_source_override(
        candidate_index=19,
        local_sources=base_sources,
        override_index=105,
        override_source=6,
    )
    duplicate_elsewhere_path = replace(
        differs_only_elsewhere,
        candidate_index=21,
        review_rank=21,
    )
    seed_equivalent = _donor(
        candidate_index=5,
        local_sources=base_sources,
        failure="formation",
    )

    ranked = rank_divergent_local_source_repair_donors(
        (second, first),
        (
            differs_only_elsewhere,
            same_edge_but_different_local_window,
            seed_equivalent,
            duplicate_elsewhere_path,
            xoxz_donor,
        ),
        minimum_failure_support=2,
    )

    assert tuple(item.candidate_index for item in ranked) == (30,)
    assert all(isinstance(item, LocalSourceRepairDonor) for item in ranked)
    assert (
        rank_divergent_local_source_repair_donors(
            (first,),
            (xoxz_donor, differs_only_elsewhere),
            minimum_failure_support=2,
        )
        == ()
    )


def _assert_requires_novel_exact_edge_at_every_recurring_failure_boundary() -> None:
    base_sources = (2, 7, 7, 7, 3, 3, 3, 3, 1, 1, 1, 6)
    first = _seed(
        candidate_index=15,
        local_sources=base_sources,
        failure="formation",
    )
    second = _seed(
        candidate_index=16,
        local_sources=base_sources,
        failure="formation",
    )
    failure_issue = first.post_repair_review.hard_failures[0]
    reviewed = tuple(
        replace(
            seed,
            post_repair_review=replace(
                seed.post_repair_review,
                hard_failures=(
                    replace(failure_issue, occurrences=(20, 100)),
                ),
            ),
        )
        for seed in (first, second)
    )
    differs_only_at_boundary_100 = _donor(
        candidate_index=30,
        local_sources=(2, 1, 1, 1, 5, 5, 5, 5, 3, 3, 3, 6),
        failure="focus",
    )
    differs_at_both = _donor_with_source_override(
        candidate_index=31,
        local_sources=(2, 1, 1, 1, 5, 5, 5, 5, 3, 3, 3, 6),
        override_index=20,
        override_source=7,
    )

    ranked = rank_divergent_local_source_repair_donors(
        reviewed,
        (differs_only_at_boundary_100, differs_at_both),
        minimum_failure_support=2,
    )

    assert tuple(item.candidate_index for item in ranked) == (31,)


def _assert_proposals_cover_every_known_base_failure_boundary() -> None:
    base = _seed(
        candidate_index=15,
        local_sources=(2, 7, 7, 7, 3, 3, 3, 3, 1, 1, 1, 6),
        failure="formation",
    )
    donor = _donor(
        candidate_index=30,
        local_sources=(2, 1, 1, 1, 5, 5, 5, 5, 3, 3, 3, 6),
        failure="focus",
    )
    failure_issue = base.post_repair_review.hard_failures[0]
    nearby_failures = replace(
        base,
        post_repair_review=replace(
            base.post_repair_review,
            hard_failures=(
                replace(failure_issue, occurrences=(96, 100)),
            ),
        ),
    )
    nearby_proposals = build_local_source_repair_proposals(
        (nearby_failures,),
        donors=(donor,),
        available_source_count=8,
        first_candidate_index=90,
    )
    assert nearby_proposals
    assert all(
        {96, 100}.issubset(item.touched_boundary_indices)
        for item in nearby_proposals
    )

    distant_failure = replace(
        base,
        post_repair_review=replace(
            base.post_repair_review,
            hard_failures=(
                replace(failure_issue, occurrences=(20, 100)),
            ),
        ),
    )
    assert (
        build_local_source_repair_proposals(
            (distant_failure,),
            donors=(donor,),
            available_source_count=8,
            first_candidate_index=100,
        )
        == ()
    )


def _assert_preserves_every_choice_outside_transplanted_complete_runs() -> None:
    base = _seed(
        candidate_index=15,
        local_sources=(2, 7, 7, 7, 3, 3, 3, 3, 1, 1, 1, 6),
        failure="formation",
    )
    donor = _seed(
        candidate_index=30,
        local_sources=(2, 1, 1, 1, 5, 5, 5, 5, 3, 3, 3, 6),
        failure="focus",
    )
    proposal = next(
        item
        for item in build_local_source_repair_proposals(
            (base, donor),
            available_source_count=8,
            first_candidate_index=40,
        )
        if item.replaced_segment_indices == (97, 98, 99, 100)
        and item.base_candidate_index == 15
    )

    replaced = set(proposal.replaced_segment_indices)
    for index, (actual, expected) in enumerate(
        zip(proposal.visual_plan.choices, base.visual_plan.choices, strict=True)
    ):
        if index not in replaced:
            assert actual is expected
    assert all(
        proposal.visual_plan.choices[index] is donor.visual_plan.choices[index]
        for index in replaced
    )


def _assert_global_or_occurrence_free_failures_do_not_create_fake_rescues() -> None:
    first = _seed(
        candidate_index=15,
        local_sources=(2, 7, 7, 7, 3, 3, 3, 3, 1, 1, 1, 6),
        failure="formation",
    )
    second = _seed(
        candidate_index=30,
        local_sources=(2, 1, 1, 1, 5, 5, 5, 5, 3, 3, 3, 6),
        failure="focus",
    )
    global_review = replace(
        first.post_repair_review,
        hard_failures=tuple(
            replace(
                issue,
                code="source_switch_density_below_hard_minimum",
                occurrences=(),
            )
            for issue in first.post_repair_review.hard_failures
        ),
    )
    blocked = replace(first, post_repair_review=global_review)

    assert not blocked.failure_boundary_indices
    assert (
        build_local_source_repair_proposals(
            (blocked, second),
            available_source_count=8,
            first_candidate_index=50,
        )
        == ()
    )


def _assert_nonlocal_advisory_failures_are_excluded_from_proposals() -> None:
    base = _seed(
        candidate_index=15,
        local_sources=(2, 7, 7, 7, 3, 3, 3, 3, 1, 1, 1, 6),
        failure="formation",
    )
    unsafe_donor = _donor(
        candidate_index=29,
        local_sources=(2, 7, 7, 7, 1, 1, 1, 1, 3, 3, 3, 6),
        failure="focus",
    )
    repairable_donor = _donor(
        candidate_index=30,
        local_sources=(2, 1, 1, 1, 5, 5, 5, 5, 3, 3, 3, 6),
        failure="focus",
    )

    unsafe_choices = list(base.visual_plan.choices)
    unsafe_choices[97:101] = unsafe_donor.visual_plan.choices[97:101]
    unsafe_review = review_timeline(
        replace(base.visual_plan, choices=tuple(unsafe_choices)),
        available_source_count=8,
    )
    assert "source_run_hard_max_exceeded" in {
        issue.code for issue in unsafe_review.hard_failures
    }

    proposals = build_local_source_repair_proposals(
        (base,),
        donors=(unsafe_donor, repairable_donor),
        available_source_count=8,
        first_candidate_index=50,
        maximum_proposals=32,
    )

    assert proposals
    assert {proposal.donor_candidate_index for proposal in proposals} == {30}
    assert all(
        issue.code in LOCAL_SOURCE_REPAIR_FAILURE_CODES and issue.occurrences
        for proposal in proposals
        for issue in proposal.candidate_record.review.hard_failures
    )


def _assert_does_not_duplicate_existing_source_and_shot_path() -> None:
    first = _seed(
        candidate_index=15,
        local_sources=(2, 7, 7, 7, 3, 3, 3, 3, 1, 1, 1, 6),
        failure="formation",
    )
    duplicate = replace(
        first,
        candidate_index=16,
        review_rank=16,
    )
    assert (
        build_local_source_repair_proposals(
            (first, duplicate),
            available_source_count=8,
            first_candidate_index=60,
        )
        == ()
    )


def _assert_does_not_duplicate_filtered_seed_source_and_shot_path() -> None:
    base = _seed(
        candidate_index=15,
        local_sources=(2, 7, 7, 7, 3, 3, 3, 3, 1, 1, 1, 6),
        failure="formation",
    )
    donor = _donor(
        candidate_index=30,
        local_sources=(2, 1, 1, 1, 5, 5, 5, 5, 2, 2, 2, 6),
        failure="focus",
    )
    existing_choices = list(base.visual_plan.choices)
    existing_choices[97:101] = donor.visual_plan.choices[97:101]
    existing_plan = replace(
        base.visual_plan,
        choices=tuple(existing_choices),
        selected_candidate_index=31,
    )
    failure_issue = base.post_repair_review.hard_failures[0]
    nonrepairable_review = replace(
        base.post_repair_review,
        hard_failures=(
            replace(
                failure_issue,
                code="source_switch_density_below_hard_minimum",
                occurrences=(),
            ),
        ),
    )
    nonrepairable_seed = LocalSourceRepairSeed(
        candidate_index=31,
        review_rank=31,
        cadence_profile="balanced",
        visual_plan=existing_plan,
        post_repair_review=nonrepairable_review,
    )
    assert not nonrepairable_seed.failure_boundary_indices

    proposals = build_local_source_repair_proposals(
        (base, nonrepairable_seed),
        donors=(donor,),
        available_source_count=8,
        first_candidate_index=90,
    )
    existing_signature = tuple(
        (choice.source_index, choice.shot_id)
        for choice in nonrepairable_seed.visual_plan.choices
    )
    assert proposals
    assert all(
        tuple(
            (choice.source_index, choice.shot_id)
            for choice in proposal.visual_plan.choices
        )
        != existing_signature
        for proposal in proposals
    )


def _assert_validates_maximum_donor_runs_per_proposal() -> None:
    for value in (1, 6):
        assert (
            build_local_source_repair_proposals(
                (),
                available_source_count=2,
                first_candidate_index=0,
                maximum_donor_runs_per_proposal=value,
            )
            == ()
        )
    for value in (0, 7, True, 1.0):
        try:
            build_local_source_repair_proposals(
                (),
                available_source_count=2,
                first_candidate_index=0,
                maximum_donor_runs_per_proposal=value,
            )
        except ValueError:
            pass
        else:  # pragma: no cover - strict validation invariant
            raise AssertionError(value)


def _hybrid_proposal():
    base = _seed(
        candidate_index=15,
        local_sources=(2, 7, 7, 7, 3, 3, 3, 3, 1, 1, 1, 6),
        failure="formation",
    )
    donor = _seed(
        candidate_index=30,
        local_sources=(2, 1, 1, 1, 5, 5, 5, 5, 3, 3, 3, 6),
        failure="focus",
    )
    return next(
        item
        for item in build_local_source_repair_proposals(
            (base, donor),
            available_source_count=8,
            first_candidate_index=70,
        )
        if item.base_candidate_index == 15
        and item.replaced_segment_indices == (97, 98, 99, 100)
    )


def _successful_repair(boundary_index: int, outgoing: int, incoming: int):
    return SimpleNamespace(
        boundary_index=boundary_index,
        outgoing_source_index=outgoing,
        incoming_source_index=incoming,
        selected_visual_evidence=SimpleNamespace(
            evidence_coverage=SimpleNamespace(professional_gate_passed=True)
        ),
    )


def _assert_zero_frame_change_is_valid_only_with_source_substitution() -> None:
    proposal = _hybrid_proposal()
    zero_timing_change = SimpleNamespace(changed_count=0)
    assert _candidate_has_material_source_or_timing_repair(
        proposal,
        zero_timing_change,
    )
    assert not _candidate_has_material_source_or_timing_repair(
        object(),
        zero_timing_change,
    )


def _assert_every_changed_cross_source_edge_needs_fresh_evidence() -> None:
    proposal = _hybrid_proposal()
    one_edge = SimpleNamespace(
        changed_count=0,
        repairs=(_successful_repair(96, 7, 5),),
    )
    incomplete = _local_source_repair_boundary_evidence(
        proposal,
        visual_timeline=proposal.visual_plan,
        boundary_repairs=one_edge,
    )
    assert incomplete["passed"] is False
    assert incomplete["missing_boundary_indices"] == [100]

    both_edges = SimpleNamespace(
        changed_count=0,
        repairs=(
            _successful_repair(96, 7, 5),
            _successful_repair(100, 5, 1),
        ),
    )
    complete = _local_source_repair_boundary_evidence(
        proposal,
        visual_timeline=proposal.visual_plan,
        boundary_repairs=both_edges,
    )
    assert complete["passed"] is True
    assert complete["required_cross_source_boundary_indices"] == [96, 100]
    assert complete["successful_boundary_indices"] == [96, 100]


def _assert_missing_splice_forces_wide_search_after_clean_review() -> None:
    proposal = _hybrid_proposal()
    clean_advisory_review = replace(
        review_timeline(proposal.visual_plan, available_source_count=8),
        hard_failures=(),
    )
    one_edge = SimpleNamespace(
        changed_count=0,
        repairs=(_successful_repair(96, 7, 5),),
    )

    targets, local_missing = (
        _adaptive_post_repair_candidate_boundary_indices(
            proposal,
            timeline_review=clean_advisory_review,
            visual_timeline=proposal.visual_plan,
            boundary_repairs=one_edge,
        )
    )

    assert clean_advisory_review.passed
    assert targets == (100,)
    assert local_missing == (100,)


class VisualLocalSourceRepairTests(unittest.TestCase):
    def test_xoxz_seed12_uses_three_runs_across_all_failure_windows(self) -> None:
        _assert_xoxz_seed12_uses_three_runs_across_all_failure_windows()

    def test_xoxz_seed13_uses_four_runs_across_all_failure_windows(self) -> None:
        _assert_xoxz_seed13_uses_four_runs_across_all_failure_windows()

    def test_recombines_complete_donor_run_across_xoxz_failure_window(self) -> None:
        _assert_recombines_complete_donor_run_across_xoxz_failure_window()

    def test_unreviewed_donor_outside_seed_pool_can_supply_xoxz_run(self) -> None:
        _assert_unreviewed_donor_outside_seed_pool_can_supply_xoxz_run()

    def test_ranks_unreviewed_xoxz_donor_by_recurring_local_divergence(self) -> None:
        _assert_ranks_unreviewed_xoxz_donor_by_recurring_local_divergence()

    def test_proposals_cover_every_known_base_failure_boundary(self) -> None:
        _assert_proposals_cover_every_known_base_failure_boundary()

    def test_requires_novel_exact_edge_at_every_recurring_failure_boundary(self) -> None:
        _assert_requires_novel_exact_edge_at_every_recurring_failure_boundary()

    def test_preserves_every_choice_outside_transplanted_complete_runs(self) -> None:
        _assert_preserves_every_choice_outside_transplanted_complete_runs()

    def test_global_failures_do_not_create_fake_rescues(self) -> None:
        _assert_global_or_occurrence_free_failures_do_not_create_fake_rescues()

    def test_nonlocal_advisory_failures_are_excluded_from_proposals(self) -> None:
        _assert_nonlocal_advisory_failures_are_excluded_from_proposals()

    def test_does_not_duplicate_existing_source_and_shot_path(self) -> None:
        _assert_does_not_duplicate_existing_source_and_shot_path()

    def test_does_not_duplicate_filtered_seed_source_and_shot_path(self) -> None:
        _assert_does_not_duplicate_filtered_seed_source_and_shot_path()

    def test_validates_maximum_donor_runs_per_proposal(self) -> None:
        _assert_validates_maximum_donor_runs_per_proposal()

    def test_zero_frame_change_is_valid_only_with_source_substitution(self) -> None:
        _assert_zero_frame_change_is_valid_only_with_source_substitution()

    def test_every_changed_cross_source_edge_needs_fresh_evidence(self) -> None:
        _assert_every_changed_cross_source_edge_needs_fresh_evidence()

    def test_missing_splice_forces_wide_search_after_clean_review(self) -> None:
        _assert_missing_splice_forces_wide_search_after_clean_review()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
