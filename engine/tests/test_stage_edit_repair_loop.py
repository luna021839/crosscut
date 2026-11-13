from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from crosscut.ive import AspectRatio
from crosscut.render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    NormalizedCrop,
    RenderSegment,
    RenderTransition,
)
from crosscut.section_director import SectionDirective, SectionPlan, SectionRole
from crosscut.stage_edit import StageBoundary, StageEdit, StageShot
from crosscut.stage_edit_repair_loop import (
    StageEditRepairCallbackError,
    StageEditRepairConfig,
    StageEditRepairExclusions,
    StageEditRepairIssue,
    StageEditRepairProposal,
    StageEditRepairReview,
    repair_stage_edit_candidates,
)


def _plan() -> SectionPlan:
    return SectionPlan(
        (
            SectionDirective(
                section_index=0,
                role=SectionRole.CHORUS_LIKE,
                start_seconds=10.0,
                end_seconds=22.0,
                energy_score=0.8,
                confidence=0.9,
                phrase_count=2,
                preferred_source_hold_beats=2,
                target_source_switches_per_minute=20.0,
                camera_motion_budget=0.7,
                target_roll_events_per_minute=0.2,
                establish_with_wide_shot=False,
                protect_last_beat_from_extra_cut=False,
            ),
        )
    )


def _edit() -> StageEdit:
    sources = tuple(Path(f"/tmp/repair-source-{index}.mp4") for index in range(3))
    shots = tuple(
        StageShot(
            shot_index=index,
            section_index=0,
            segment=RenderSegment(
                source_index=index % 3,
                source_start=10.0 + index * 2.0,
                duration=2.0,
                master_start=index * 2.0,
            ),
            decision_id=f"decision-{index}",
            decision_confidence=0.8,
        )
        for index in range(6)
    )
    boundaries = tuple(
        StageBoundary(
            boundary_index=index,
            transition=RenderTransition(reason=f"original-boundary-{index}"),
            reviewer_status="passed",
        )
        for index in range(5)
    )
    return StageEdit(
        sources=sources,
        reference_source=0,
        aspect_ratio=AspectRatio.LANDSCAPE,
        width=1920,
        height=1080,
        fps=30.0,
        master_audio_start=10.0,
        section_plan=_plan(),
        shots=shots,
        boundaries=boundaries,
        output=Path("/tmp/repair-result.mp4"),
    )


CANONICAL = tuple(float(value) for value in range(10, 23, 2))


def _issue(index: int) -> StageEditRepairIssue:
    start = 10.0 + index * 2.0 + 0.8
    return StageEditRepairIssue(
        issue_id=f"issue-{index}",
        start_seconds=start,
        end_seconds=start + 0.2,
        causes=(f"cause-{index}",),
    )


def _boundary_patch(
    request,
    replacement_count: int,
    *,
    reason: str = "repair_hard_cut",
) -> tuple[StageBoundary, ...]:
    count = (
        replacement_count
        - 1
        + (1 if request.window.start_shot > 0 else 0)
        + (
            1
            if request.window.end_shot_exclusive < len(request.base_edit.shots)
            else 0
        )
    )
    return tuple(
        StageBoundary(
            boundary_index=index,
            transition=RenderTransition(reason=reason),
            reviewer_status="unreviewed",
        )
        for index in range(count)
    )


def _proposal_repairing_shot(
    request,
    target_index: int,
    *,
    proposal_id: str,
    rank: int = 1,
    source_index: int | None = None,
    crop: bool = False,
    editorial_transform: bool = False,
    transition_reason: str = "repair_hard_cut",
    shift_start: float = 0.0,
) -> StageEditRepairProposal:
    replacements = list(
        request.base_edit.shots[
            request.window.start_shot : request.window.end_shot_exclusive
        ]
    )
    local_index = target_index - request.window.start_shot
    target = replacements[local_index]
    segment = target.segment
    selected_source = (
        (segment.source_index + 1) % len(request.base_edit.sources)
        if source_index is None
        else source_index
    )
    transform = (
        EditorialTransformTrack(
            (
                EditorialTransformKeyframe(
                    time_seconds=0.0,
                    zoom=1.04,
                    center_x=0.5,
                    center_y=0.5,
                    roll_degrees=0.0,
                ),
            ),
            reason="synthetic repair crop",
        )
        if editorial_transform
        else None
    )
    new_segment = replace(
        segment,
        source_index=selected_source,
        master_start=segment.master_start + shift_start,
        crop=NormalizedCrop(0.1, 0.1, 0.8, 0.8) if crop else None,
        editorial_transform=transform,
    )
    replacements[local_index] = replace(
        target,
        segment=new_segment,
        decision_id=f"{target.decision_id}-repaired",
    )
    return StageEditRepairProposal(
        proposal_id=proposal_id,
        rank=rank,
        replacement_shots=tuple(replacements),
        replacement_boundaries=_boundary_patch(
            request,
            len(replacements),
            reason=transition_reason,
        ),
        audit={"synthetic": True, "target_index": target_index},
    )


def _current_result_proposal(request, *, proposal_id: str) -> StageEditRepairProposal:
    start = request.window.start_shot
    end = request.window.end_shot_exclusive
    boundary_start = max(0, start - 1)
    boundary_end = min(len(request.base_edit.boundaries), end)
    return StageEditRepairProposal(
        proposal_id=proposal_id,
        rank=1,
        replacement_shots=request.base_edit.shots[start:end],
        replacement_boundaries=request.base_edit.boundaries[
            boundary_start:boundary_end
        ],
    )


def _objective_review(
    edit: StageEdit,
    *,
    target_indices: tuple[int, ...],
    technical_when_repaired: bool = False,
    flat_professional: bool = False,
    keep_repaired_issue: bool = False,
) -> StageEditRepairReview:
    repaired = tuple(
        index
        for index in target_indices
        if "-repaired" in edit.shots[index].decision_id
    )
    issues = tuple(
        _issue(index)
        for index in target_indices
        if index not in repaired or keep_repaired_issue
    )
    count = len(repaired)
    return StageEditRepairReview(
        decoded_objective=50.0 + count,
        professional_objective=50.0 if flat_professional else 60.0 + count,
        cadence_objective=70.0 + count,
        technical_errors=("decoded_black_frame",)
        if technical_when_repaired and count
        else (),
        issue_windows=issues,
        audit={"repaired_count": count},
    )


class StageEditRepairLoopTests(unittest.TestCase):
    def test_candidate_selection_is_deterministic_when_no_repair_is_requested(self) -> None:
        first = _edit()
        better_first_shot = replace(
            first.shots[0],
            segment=replace(first.shots[0].segment, source_index=2),
            decision_id="better-initial-candidate",
        )
        second = replace(first, shots=(better_first_shot,) + first.shots[1:])

        def scorer(value):
            bonus = 5.0 if value.shots[0].segment.source_index == 2 else 0.0
            return StageEditRepairReview(
                decoded_objective=50.0 + bonus,
                professional_objective=60.0 + bonus,
                cadence_objective=70.0 + bonus,
            )

        result = repair_stage_edit_candidates(
            candidates=(first, second),
            reviewer_issues=(),
            canonical_shot_boundaries_seconds=CANONICAL,
            beam_redecoder=lambda request: self.fail("decoder must not be called"),
            review_scorer=scorer,
        )

        self.assertIs(result.original_edit, second)
        self.assertIs(result.edit, second)
        self.assertFalse(result.changed)
        self.assertEqual(1, result.audit.selected_candidate_index)
        self.assertEqual("no_reviewer_issues", result.audit.termination_reason)
        self.assertEqual((), result.audit.iterations)

    def test_default_three_second_context_snaps_to_canonical_and_preserves_outside_ir(
        self,
    ) -> None:
        edit = _edit()
        requests = []

        def decoder(request):
            requests.append(request)
            return (
                _proposal_repairing_shot(
                    request,
                    2,
                    proposal_id="repair-shot-2",
                ),
            )

        result = repair_stage_edit_candidates(
            candidates=(edit,),
            reviewer_issues=(_issue(2),),
            canonical_shot_boundaries_seconds=CANONICAL,
            beam_redecoder=decoder,
            review_scorer=lambda value: _objective_review(
                value, target_indices=(2,)
            ),
        )

        self.assertTrue(result.changed)
        self.assertEqual(1, result.audit.committed_iteration_count)
        self.assertEqual("reviewer_clean_no_remaining_issues", result.audit.termination_reason)
        request = requests[0]
        self.assertEqual(3.0, result.audit.config["context_seconds"])
        self.assertEqual(10.0, request.window.canonical_start_seconds)
        self.assertEqual(18.0, request.window.canonical_end_seconds)
        self.assertEqual((0, 4), (request.window.start_shot, request.window.end_shot_exclusive))
        self.assertIs(result.edit.shots[4], edit.shots[4])
        self.assertIs(result.edit.shots[5], edit.shots[5])
        self.assertEqual(edit.duration_seconds, result.edit.duration_seconds)
        self.assertEqual(1, len(result.edit.lineage))
        self.assertEqual(1, len(result.edit.reviewer_history))
        json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False)

    def test_synthetic_reviewer_drives_two_strictly_improving_iterations(self) -> None:
        edit = _edit()
        target_indices = (1, 4)
        requests = []

        def decoder(request):
            requests.append(request)
            target = next(
                index
                for index in target_indices
                if request.window.start_shot <= index < request.window.end_shot_exclusive
                and "-repaired" not in request.base_edit.shots[index].decision_id
            )
            return (
                _proposal_repairing_shot(
                    request,
                    target,
                    proposal_id=f"repair-{target}",
                ),
            )

        result = repair_stage_edit_candidates(
            candidates=(edit,),
            reviewer_issues=tuple(_issue(index) for index in target_indices),
            canonical_shot_boundaries_seconds=CANONICAL,
            beam_redecoder=decoder,
            review_scorer=lambda value: _objective_review(
                value, target_indices=target_indices
            ),
            config=StageEditRepairConfig(context_seconds=0.25, maximum_iterations=4),
        )

        self.assertEqual(2, result.audit.committed_iteration_count)
        self.assertEqual(2, len(result.audit.iterations))
        self.assertEqual(2, len(result.edit.lineage))
        self.assertEqual(2, len(result.edit.reviewer_history))
        self.assertTrue(
            all(
                "-repaired" in result.edit.shots[index].decision_id
                for index in target_indices
            )
        )
        self.assertEqual(
            ["repair-1", "repair-4"],
            [value.committed_proposal_id for value in result.audit.iterations],
        )
        for value in result.audit.iterations:
            proposal = value.proposal_audits[0]
            self.assertTrue(proposal["committed"])
            self.assertTrue(
                all(delta > 0.0 for delta in proposal["objective_deltas"].values())
            )
            self.assertEqual([], proposal["review"]["technical_errors"])
        self.assertEqual(2, len(requests))

    def test_window_rebeam_can_change_cut_duration_transition_crop_and_source(self) -> None:
        edit = _edit()

        def decoder(request):
            self.assertEqual((2, 3), (
                request.window.start_shot,
                request.window.end_shot_exclusive,
            ))
            transform = EditorialTransformTrack(
                (
                    EditorialTransformKeyframe(
                        time_seconds=0.0,
                        zoom=1.05,
                        center_x=0.5,
                        center_y=0.5,
                        roll_degrees=0.0,
                    ),
                ),
                reason="learned zoom match",
            )
            replacements = (
                StageShot(
                    shot_index=2,
                    section_index=0,
                    segment=RenderSegment(
                        source_index=0,
                        source_start=14.0,
                        duration=1.0,
                        master_start=4.0,
                        editorial_transform=transform,
                    ),
                    decision_id="split-left-repaired",
                    decision_confidence=0.9,
                ),
                StageShot(
                    shot_index=3,
                    section_index=0,
                    segment=RenderSegment(
                        source_index=1,
                        source_start=15.0,
                        duration=1.0,
                        master_start=5.0,
                    ),
                    decision_id="split-right-repaired",
                    decision_confidence=0.9,
                ),
            )
            return (
                StageEditRepairProposal(
                    proposal_id="joint-cut-source-duration-transition-crop",
                    rank=1,
                    replacement_shots=replacements,
                    replacement_boundaries=_boundary_patch(
                        request,
                        2,
                        reason="stage_edit_requested_zoom_match",
                    ),
                ),
            )

        def scorer(value):
            repaired = len(value.shots) == 7
            return StageEditRepairReview(
                decoded_objective=50.0 + repaired,
                professional_objective=60.0 + repaired,
                cadence_objective=70.0 + repaired,
                issue_windows=() if repaired else (_issue(2),),
            )

        result = repair_stage_edit_candidates(
            candidates=(edit,),
            reviewer_issues=(_issue(2),),
            canonical_shot_boundaries_seconds=CANONICAL,
            beam_redecoder=decoder,
            review_scorer=scorer,
            config=StageEditRepairConfig(context_seconds=0.0),
        )

        self.assertTrue(result.changed)
        self.assertEqual(7, len(result.edit.shots))
        self.assertEqual(12.0, result.edit.duration_seconds)
        self.assertEqual((1.0, 1.0), tuple(
            result.edit.shots[index].segment.duration for index in (2, 3)
        ))
        self.assertIsNotNone(result.edit.shots[2].segment.editorial_transform)
        self.assertTrue(all(
            boundary.transition.reason == "stage_edit_requested_zoom_match"
            for boundary in result.edit.boundaries[1:4]
        ))
        self.assertEqual(
            [shot.decision_id for shot in edit.shots[3:]],
            [shot.decision_id for shot in result.edit.shots[4:]],
        )
        proposal = result.audit.iterations[0].proposal_audits[0]
        self.assertTrue(proposal["committed"])

    def test_non_improvement_and_technical_error_both_roll_back(self) -> None:
        edit = _edit()

        def decoder(request):
            return (
                _proposal_repairing_shot(
                    request,
                    2,
                    proposal_id="flat-professional",
                    rank=1,
                    source_index=1,
                ),
                _proposal_repairing_shot(
                    request,
                    2,
                    proposal_id="technical-error",
                    rank=2,
                    source_index=0,
                ),
            )

        def scorer(value):
            repaired = "-repaired" in value.shots[2].decision_id
            if not repaired:
                return _objective_review(value, target_indices=(2,))
            if value.shots[2].segment.source_index == 0:
                return _objective_review(
                    value,
                    target_indices=(2,),
                    technical_when_repaired=True,
                )
            return _objective_review(
                value,
                target_indices=(2,),
                flat_professional=True,
            )

        result = repair_stage_edit_candidates(
            candidates=(edit,),
            reviewer_issues=(_issue(2),),
            canonical_shot_boundaries_seconds=CANONICAL,
            beam_redecoder=decoder,
            review_scorer=scorer,
            config=StageEditRepairConfig(context_seconds=0.25),
        )

        self.assertFalse(result.changed)
        self.assertIs(result.edit, edit)
        self.assertEqual(0, result.audit.committed_iteration_count)
        audits = {
            value["proposal_id"]: value
            for value in result.audit.iterations[0].proposal_audits
        }
        self.assertEqual(
            ["all_objectives_must_strictly_improve"],
            audits["flat-professional"]["rejection_reasons"],
        )
        self.assertEqual(
            ["technical_zero_tolerance_failed"],
            audits["technical-error"]["rejection_reasons"],
        )

    def test_tamper_and_source_crop_transition_exclusions_fail_closed(self) -> None:
        edit = _edit()

        def decoder(request):
            return (
                _proposal_repairing_shot(
                    request,
                    2,
                    proposal_id="tampered-clock",
                    rank=1,
                    shift_start=0.25,
                ),
                _proposal_repairing_shot(
                    request,
                    2,
                    proposal_id="excluded-source",
                    rank=2,
                    source_index=1,
                ),
                _proposal_repairing_shot(
                    request,
                    2,
                    proposal_id="excluded-crop",
                    rank=3,
                    crop=True,
                ),
                _proposal_repairing_shot(
                    request,
                    2,
                    proposal_id="excluded-transition",
                    rank=4,
                    transition_reason="stage_edit_requested_whip_pan",
                ),
            )

        result = repair_stage_edit_candidates(
            candidates=(edit,),
            reviewer_issues=(_issue(2),),
            canonical_shot_boundaries_seconds=CANONICAL,
            beam_redecoder=decoder,
            review_scorer=lambda value: _objective_review(
                value, target_indices=(2,)
            ),
            exclusions=StageEditRepairExclusions(
                source_indices=(1,),
                transition_labels=("whip_pan",),
                crop_modes=("static_crop",),
            ),
            config=StageEditRepairConfig(context_seconds=0.25),
        )

        self.assertFalse(result.changed)
        audits = {
            value["proposal_id"]: value["rejection_reasons"]
            for value in result.audit.iterations[0].proposal_audits
        }
        self.assertTrue(audits["tampered-clock"][0].startswith("splice_or_ir_validation_failed:"))
        self.assertEqual(["excluded_source:1"], audits["excluded-source"])
        self.assertEqual(["excluded_crop_mode:static_crop"], audits["excluded-crop"])
        self.assertEqual(
            ["excluded_transition_label:whip_pan"],
            audits["excluded-transition"],
        )

    def test_repeated_result_is_not_rescored_or_committed_twice(self) -> None:
        edit = _edit()
        calls = []

        def decoder(request):
            calls.append(request.iteration)
            if request.iteration == 1:
                return (
                    _proposal_repairing_shot(
                        request,
                        2,
                        proposal_id="first-repair",
                    ),
                )
            return (
                _current_result_proposal(
                    request,
                    proposal_id="same-pixels-again",
                ),
            )

        result = repair_stage_edit_candidates(
            candidates=(edit,),
            reviewer_issues=(_issue(2),),
            canonical_shot_boundaries_seconds=CANONICAL,
            beam_redecoder=decoder,
            review_scorer=lambda value: _objective_review(
                value,
                target_indices=(2,),
                keep_repaired_issue=True,
            ),
            config=StageEditRepairConfig(
                context_seconds=0.25,
                maximum_iterations=4,
                maximum_consecutive_no_improvement=1,
            ),
        )

        self.assertTrue(result.changed)
        self.assertEqual([1, 2], calls)
        self.assertEqual(1, result.audit.committed_iteration_count)
        self.assertEqual("no_strict_improvement", result.audit.termination_reason)
        second = result.audit.iterations[1].proposal_audits[0]
        self.assertEqual(["repeated_edit_result"], second["rejection_reasons"])
        self.assertIsNone(second["review"])
        self.assertEqual(1, len(result.edit.lineage))

    def test_callback_contract_errors_are_zero_tolerance(self) -> None:
        edit = _edit()

        with self.assertRaisesRegex(StageEditRepairCallbackError, "review scorer"):
            repair_stage_edit_candidates(
                candidates=(edit,),
                reviewer_issues=(_issue(2),),
                canonical_shot_boundaries_seconds=CANONICAL,
                beam_redecoder=lambda request: (),
                review_scorer=lambda value: {
                    "decoded_objective": 1.0,
                    "professional_objective": 1.0,
                    "cadence_objective": 1.0,
                    # technical_errors is intentionally missing.
                },
            )

        with self.assertRaisesRegex(StageEditRepairCallbackError, "beam redecoder"):
            repair_stage_edit_candidates(
                candidates=(edit,),
                reviewer_issues=(_issue(2),),
                canonical_shot_boundaries_seconds=CANONICAL,
                beam_redecoder=lambda request: (_ for _ in ()).throw(
                    RuntimeError("decoder crashed")
                ),
                review_scorer=lambda value: _objective_review(
                    value, target_indices=(2,)
                ),
            )

        def one_proposal(request):
            return (
                _proposal_repairing_shot(
                    request,
                    2,
                    proposal_id="nondeterministic-review-check",
                ),
            )

        def lineage_sensitive_scorer(value):
            if not value.lineage:
                bonus = 0.0
            elif "repair_trial" in value.lineage[-1].reason:
                bonus = 1.0
            else:
                bonus = 2.0
            return StageEditRepairReview(
                decoded_objective=50.0 + bonus,
                professional_objective=60.0 + bonus,
                cadence_objective=70.0 + bonus,
                issue_windows=(_issue(2),),
            )

        with self.assertRaisesRegex(
            StageEditRepairCallbackError,
            "identical rendered IR",
        ):
            repair_stage_edit_candidates(
                candidates=(edit,),
                reviewer_issues=(_issue(2),),
                canonical_shot_boundaries_seconds=CANONICAL,
                beam_redecoder=one_proposal,
                review_scorer=lineage_sensitive_scorer,
                config=StageEditRepairConfig(context_seconds=0.25),
            )


if __name__ == "__main__":
    unittest.main()
