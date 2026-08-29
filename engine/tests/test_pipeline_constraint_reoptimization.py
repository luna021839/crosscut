from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crosscut.lazy_splice_feasibility_graph import LazySpliceFeasibilityGraph
from crosscut.pipeline import (
    _BoundarySearchBudget,
    _BoundarySearchBudgetExhausted,
    _GlobalVisualCandidateStream,
    _assessment_has_final_boundary_geometry_evidence,
    _authorize_partial_reedit_splice_lineage,
    _boundary_repair_assessments,
    _candidate_local_splice_proxy_authorization,
    _cross_source_timeline_edge_contexts,
    _cross_source_timeline_edge_keys,
    _group_splice_analysis_basis_identity,
    _optimizer_workspace_cross_source_edge_bound,
    _optimizer_blocked_set_for_contexts,
    _record_boundary_repair_assessments,
    _regenerate_visual_candidate_set_or_no_path,
    _resume_or_reset_splice_feasibility_graph,
    _run_initial_edge_constrained_stream,
    _splice_authorization_refinement_boundary_indices,
    _splice_graph_keys_for_plan,
    _validate_constrained_precomputed_repair_origin,
    _merge_visual_boundary_repair_plans,
)
from crosscut.timeline import NoCompleteTimelinePath, ShotType
from crosscut.timeline_no_good import TimelineNoGoodSet
from crosscut.timeline_review import TimelineReviewConfig
from crosscut.visual_boundary_repair import VisualBoundaryRepairPlan
from crosscut.visual_score import VisualTimelineChoice, VisualTimelinePlan
from crosscut.visual_splice_failure_feedback import (
    splice_feedback_edge_context,
    splice_feedback_edge_context_identity,
    splice_feedback_plan_identity,
    splice_feedback_result_context,
    splice_feedback_result_context_identity,
)
from crosscut.visual_splice_feasibility import (
    DEFAULT_WIDE_FRAME_OFFSETS,
    SpliceFeasibilityStatus,
    SpliceOffsetAssessment,
    build_splice_feasibility_assessment,
)


def _choice(source: int, shot: str) -> VisualTimelineChoice:
    return VisualTimelineChoice(
        segment_index=0,
        source_index=source,
        canonical_master_start=0.0,
        canonical_master_end=1.0,
        source_local_start=0.0,
        source_local_end=1.0,
        quality_score=1.0,
        composition_score=1.0,
        boundary_score=1.0,
        shot_scale=0.5,
        shot_type=ShotType.MEDIUM,
        shot_id=shot,
    )


def _plan(
    *choices: VisualTimelineChoice,
    analysis_basis: str = "test-plan-v1",
) -> VisualTimelinePlan:
    timed = tuple(
        replace(
            choice,
            segment_index=index,
            canonical_master_start=float(index),
            canonical_master_end=float(index + 1),
            source_local_start=float(index),
            source_local_end=float(index + 1),
        )
        for index, choice in enumerate(choices)
    )
    return VisualTimelinePlan(
        choices=timed,
        optimizer_score=1.0,
        analysis_basis=analysis_basis,
    )


def _materialized(*choices: VisualTimelineChoice) -> SimpleNamespace:
    return SimpleNamespace(
        candidate_index=0,
        review_rank=0,
        cadence_profile="test",
        candidate_record=SimpleNamespace(review=SimpleNamespace(hard_failures=())),
        visual_plan=_plan(*choices),
    )


def _stability() -> dict[str, float]:
    return {
        "scale_delta": 0.01,
        "structure_mismatch": 0.01,
        "pose_mismatch": 0.01,
        "shot_type_mismatch": 0.0,
    }


def _audit(
    offset: int,
    *,
    silhouette: float = 0.1,
    focus: float = 0.1,
) -> dict[str, object]:
    return {
        "frame_offset": offset,
        "professional_gate_passed": True,
        "production_transition_gate_passed": False,
        "visual_evidence": {
            "shot_scale_delta": 0.1,
            "focus_mismatch": focus,
            "layout_mismatch": 0.1,
            "persisted_layout_mismatch": 0.1,
            "silhouette_mismatch": silhouette,
            "formation_mismatch": 0.1,
            "pose_mismatch": 0.1,
            "outgoing_side_stability": _stability(),
            "incoming_side_stability": _stability(),
        },
    }


def _missing_audit(offset: int) -> dict[str, object]:
    return {
        "frame_offset": offset,
        "professional_gate_passed": False,
        "production_transition_gate_passed": False,
        "missing_mask": ["combined_geometry_descriptor"],
        "visual_evidence": {},
    }


def _failure(
    plan: VisualTimelinePlan,
    boundary_index: int,
    audit: tuple[dict[str, object], ...],
) -> dict[str, object]:
    outgoing = plan.choices[boundary_index]
    incoming = plan.choices[boundary_index + 1]
    return {
        "boundary_index": boundary_index,
        "outgoing_source_index": outgoing.source_index,
        "incoming_source_index": incoming.source_index,
        "outgoing_shot_id": outgoing.shot_id,
        "incoming_shot_id": incoming.shot_id,
        "timeline_analysis_basis": plan.analysis_basis,
        "timeline_source_clock_basis": plan.source_clock_basis,
        "timeline_path_identity": splice_feedback_plan_identity(plan),
        "optimizer_edge_context": splice_feedback_edge_context(
            plan,
            boundary_index,
        ),
        "optimizer_edge_context_identity": (
            splice_feedback_edge_context_identity(plan, boundary_index)
        ),
        "observed_edge_context": splice_feedback_edge_context(
            plan,
            boundary_index,
        ),
        "observed_edge_context_identity": (
            splice_feedback_edge_context_identity(plan, boundary_index)
        ),
        "optimizer_edge_context_match": True,
        "reason": "no acceptable offset",
        "candidate_evidence_audit": audit,
    }


def _repair_plan(
    plan: VisualTimelinePlan,
    *,
    result_timeline: VisualTimelinePlan | None = None,
    repairs: tuple[SimpleNamespace, ...] = (),
    failures: tuple[dict[str, object], ...] = (),
    requested: tuple[int, ...] | None = None,
) -> VisualBoundaryRepairPlan:
    with patch(
        "crosscut.visual_boundary_repair.VisualBoundaryRepair",
        SimpleNamespace,
    ):
        return VisualBoundaryRepairPlan(
            timeline=result_timeline or plan,
            attempt_input_timelines=(plan,),
            repairs=repairs,  # type: ignore[arg-type]
            skipped_same_source_boundaries=0,
            failures=failures,
            requested_boundary_indices=requested,
        )


def _successful_repair(
    plan: VisualTimelinePlan,
    boundary_index: int,
    frame_offset: int,
) -> SimpleNamespace:
    edge_context = splice_feedback_edge_context(plan, boundary_index)
    result_context = splice_feedback_result_context(plan, boundary_index)
    outgoing = plan.choices[boundary_index]
    incoming = plan.choices[boundary_index + 1]
    return SimpleNamespace(
        boundary_index=boundary_index,
        outgoing_source_index=outgoing.source_index,
        incoming_source_index=incoming.source_index,
        optimizer_edge_context=edge_context,
        observed_edge_context=edge_context,
        result_edge_context=result_context,
        optimizer_edge_context_match=True,
        optimizer_edge_context_identity=(
            splice_feedback_edge_context_identity(plan, boundary_index)
        ),
        observed_edge_context_identity=(
            splice_feedback_edge_context_identity(plan, boundary_index)
        ),
        result_edge_context_identity=(
            splice_feedback_result_context_identity(plan, boundary_index)
        ),
        result=SimpleNamespace(selected_frame_offset=frame_offset),
        selected_visual_evidence=SimpleNamespace(
            evidence_coverage=SimpleNamespace(professional_gate_passed=True)
        ),
    )


def _hard_assessment():
    return build_splice_feasibility_assessment(
        expected_frame_offsets=DEFAULT_WIDE_FRAME_OFFSETS,
        offsets=tuple(
            SpliceOffsetAssessment(offset, False, ("silhouette",), True)
            for offset in DEFAULT_WIDE_FRAME_OFFSETS
        ),
    )


class PipelineConstraintReoptimizationTests(unittest.TestCase):
    def test_stream_ids_are_global_and_source_shot_paths_dedupe_across_rounds(
        self,
    ) -> None:
        stream = _GlobalVisualCandidateStream()
        raw_candidates = tuple(
            _materialized(
                _choice(0, "0:a"),
                _choice(1, f"1:{suffix}"),
            )
            for suffix in ("b", "c", "d")
        )
        for raw_candidate, original_index in zip(
            raw_candidates,
            (3, 8, 12),
            strict=True,
        ):
            raw_candidate.candidate_index = original_index
        first, second_raw, third_raw = tuple(
            stream.admit_unseen(candidate) for candidate in raw_candidates
        )
        repeated = stream.admit_unseen(
            _materialized(_choice(0, "0:a"), _choice(1, "1:b"))
        )
        regenerated = stream.admit_unseen(
            _materialized(_choice(0, "0:e"), _choice(1, "1:f"))
        )
        repeated_donor = stream.admit_unseen(
            _materialized(_choice(0, "0:a"), _choice(1, "1:c"))
        )
        donor = stream.admit_unseen(
            _materialized(_choice(1, "1:g"), _choice(0, "0:h"))
        )

        admitted = (first, second_raw, third_raw, regenerated, donor)
        self.assertTrue(all(item is not None for item in admitted))
        self.assertEqual(
            tuple(item.candidate_index for item in admitted if item is not None),
            (0, 1, 2, 3, 4),
        )
        self.assertIsNone(repeated)
        self.assertIsNone(repeated_donor)
        self.assertEqual(stream.seen_path_count, 5)

    def test_shared_blocked_edge_initial_pool_runs_expensive_derivation_once(
        self,
    ) -> None:
        graph = LazySpliceFeasibilityGraph("basis-v1")
        stream = _GlobalVisualCandidateStream()
        candidates = tuple(
            stream.admit_unseen(
                _materialized(
                    _choice(0, "0:shared"),
                    _choice(1, "1:shared"),
                    _choice(0, f"0:tail-{index}"),
                )
            )
            for index in range(32)
        )
        self.assertTrue(all(candidate is not None for candidate in candidates))
        first_edge = _cross_source_timeline_edge_keys(candidates[0].visual_plan)[0]  # type: ignore[union-attr]
        first_edge_context = _cross_source_timeline_edge_contexts(
            candidates[0].visual_plan  # type: ignore[union-attr]
        )[0][1]
        derived: list[int] = []
        rejected: list[int] = []

        def preflight(candidate):
            blocked_set = graph.blocked_set_for_contexts(
                _splice_graph_keys_for_plan(candidate.visual_plan)
            )
            blocked = tuple(
                edge
                for edge in _cross_source_timeline_edge_keys(candidate.visual_plan)
                if edge in blocked_set
            )
            return blocked, None, 0

        def derive(candidate, _repairs):
            derived.append(candidate.candidate_index)
            graph.record_assessment(
                first_edge,
                first_edge_context,
                _hard_assessment(),
            )

        audit = _run_initial_edge_constrained_stream(
            candidates,  # type: ignore[arg-type]
            preflight=preflight,
            derive=derive,
            reject=lambda candidate, _edges: rejected.append(candidate.candidate_index),
            should_stop=lambda: False,
        )

        self.assertEqual(derived, [0])
        self.assertEqual(len(rejected), 31)
        self.assertEqual(audit["derived_count"], 1)
        self.assertEqual(audit["blocked_before_expensive_derivation_count"], 31)

    def test_partial_three_frame_evidence_refines_once_to_wide_pass_or_hard(
        self,
    ) -> None:
        plan = _plan(_choice(0, "0:a"), _choice(1, "1:b"))
        narrow = _repair_plan(
            plan,
            failures=(
                _failure(
                    plan,
                    0,
                    tuple(
                        _audit(offset, silhouette=0.60)
                        for offset in range(-3, 4)
                    ),
                ),
            ),
            requested=(0,),
        )
        edge = _cross_source_timeline_edge_keys(plan)[0]
        edge_context = _cross_source_timeline_edge_contexts(plan)[0][1]
        result_context = splice_feedback_result_context_identity(plan, 0)

        pass_graph = LazySpliceFeasibilityGraph("basis-v1")
        _record_boundary_repair_assessments(pass_graph, narrow)
        self.assertTrue(pass_graph.needs_assessment(edge, edge_context))
        self.assertNotIn(
            edge,
            pass_graph.blocked_set_for_contexts(
                _splice_graph_keys_for_plan(plan)
            ),
        )
        plus_five_pass = _repair_plan(
            plan,
            repairs=(_successful_repair(plan, 0, 5),),
            requested=(0,),
        )
        _record_boundary_repair_assessments(pass_graph, plus_five_pass)
        self.assertFalse(pass_graph.needs_assessment(edge, result_context))
        self.assertIs(
            pass_graph.assessment_for(edge, result_context).status,  # type: ignore[union-attr]
            SpliceFeasibilityStatus.RAW_PASS,
        )

        hard_graph = LazySpliceFeasibilityGraph("basis-v1")
        _record_boundary_repair_assessments(hard_graph, narrow)
        wide_hard = _repair_plan(
            plan,
            failures=(
                _failure(
                    plan,
                    0,
                    tuple(
                        _audit(offset, silhouette=0.60)
                        for offset in DEFAULT_WIDE_FRAME_OFFSETS
                    ),
                ),
            ),
            requested=(0,),
        )
        _record_boundary_repair_assessments(hard_graph, wide_hard)
        self.assertFalse(hard_graph.needs_assessment(edge, edge_context))
        self.assertIn(
            edge,
            hard_graph.blocked_set_for_contexts(
                _splice_graph_keys_for_plan(plan)
            ),
        )

    def test_changed_span_positive_is_reassessed_and_transient_hard_not_projected(
        self,
    ) -> None:
        raw_plan = _plan(_choice(0, "0:a"), _choice(1, "1:b"))
        transient_plan = replace(
            raw_plan,
            choices=(
                replace(
                    raw_plan.choices[0],
                    canonical_master_end=0.9,
                    source_local_end=0.9,
                ),
                replace(
                    raw_plan.choices[1],
                    canonical_master_start=0.9,
                    source_local_start=0.9,
                ),
            ),
        )
        edge = _cross_source_timeline_edge_keys(raw_plan)[0]
        raw_context = _cross_source_timeline_edge_contexts(raw_plan)[0][1]
        transient_context = _cross_source_timeline_edge_contexts(
            transient_plan
        )[0][1]
        raw_result_context = splice_feedback_result_context_identity(
            raw_plan,
            0,
        )
        transient_result_context = splice_feedback_result_context_identity(
            transient_plan,
            0,
        )
        self.assertNotEqual(raw_context, transient_context)
        self.assertNotEqual(raw_result_context, transient_result_context)

        graph = LazySpliceFeasibilityGraph("basis-v1")
        graph.record_assessment(
            edge,
            raw_result_context,
            build_splice_feasibility_assessment(
                expected_frame_offsets=DEFAULT_WIDE_FRAME_OFFSETS,
                offsets=(SpliceOffsetAssessment(0, True, (), True),),
            ),
        )
        self.assertIsNone(
            graph.assessment_for(edge, transient_result_context)
        )

        graph.record_assessment(
            edge,
            transient_context,
            _hard_assessment(),
        )
        self.assertNotIn(
            edge,
            _optimizer_blocked_set_for_contexts(
                graph,
                _splice_graph_keys_for_plan(raw_plan),
            ),
        )
        self.assertIn(
            edge,
            graph.blocked_set_for_contexts(
                _splice_graph_keys_for_plan(transient_plan)
            ),
        )

    def test_zero_and_changed_result_contexts_authorize_only_fresh_final_edge(
        self,
    ) -> None:
        input_plan = _plan(_choice(0, "0:a"), _choice(1, "1:b"))
        changed_plan = replace(
            input_plan,
            choices=(
                replace(
                    input_plan.choices[0],
                    canonical_master_end=0.9,
                    source_local_end=0.9,
                ),
                replace(
                    input_plan.choices[1],
                    canonical_master_start=0.9,
                    source_local_start=0.9,
                ),
            ),
        )
        for frame_offset, final_plan in ((0, input_plan), (3, changed_plan)):
            graph = LazySpliceFeasibilityGraph(f"basis-{frame_offset}")
            repair_plan = _repair_plan(
                input_plan,
                result_timeline=final_plan,
                repairs=(
                    _successful_repair(final_plan, 0, frame_offset),
                ),
                requested=(0,),
            )
            _record_boundary_repair_assessments(graph, repair_plan)
            authorization = _candidate_local_splice_proxy_authorization(
                graph,
                final_timeline=final_plan,
                repair_plan=repair_plan,
            )
            self.assertTrue(authorization["proxy_eligible"])
            self.assertEqual(authorization["unauthorized_keys"], [])

            if final_plan is changed_plan:
                stale = _candidate_local_splice_proxy_authorization(
                    graph,
                    final_timeline=input_plan,
                    repair_plan=repair_plan,
                )
                self.assertFalse(stale["proxy_eligible"])

    def test_adjacent_successes_retain_final_result_context_authorization(
        self,
    ) -> None:
        input_plan = _plan(
            _choice(0, "0:a"),
            _choice(1, "1:b"),
            _choice(0, "0:c"),
        )
        final_plan = replace(
            input_plan,
            choices=(
                replace(
                    input_plan.choices[0],
                    canonical_master_end=0.9,
                    source_local_end=0.9,
                ),
                replace(
                    input_plan.choices[1],
                    canonical_master_start=0.9,
                    canonical_master_end=2.1,
                    source_local_start=0.9,
                    source_local_end=2.1,
                ),
                replace(
                    input_plan.choices[2],
                    canonical_master_start=2.1,
                    source_local_start=2.1,
                ),
            ),
        )
        repair_plan = _repair_plan(
            input_plan,
            result_timeline=final_plan,
            repairs=(
                _successful_repair(final_plan, 0, -3),
                _successful_repair(final_plan, 1, 3),
            ),
            requested=(0, 1),
        )
        graph = LazySpliceFeasibilityGraph("adjacent-successes")
        _record_boundary_repair_assessments(graph, repair_plan)

        authorization = _candidate_local_splice_proxy_authorization(
            graph,
            final_timeline=final_plan,
            repair_plan=repair_plan,
        )

        self.assertTrue(authorization["proxy_eligible"])
        self.assertEqual(authorization["unauthorized_keys"], [])

    def test_partial_reedit_reauthorizes_target_and_untouched_edge_lineage(
        self,
    ) -> None:
        selected_plan = _plan(
            _choice(0, "0:a"),
            _choice(1, "1:b"),
            _choice(0, "0:c"),
        )
        selected_repairs = _repair_plan(
            selected_plan,
            repairs=(
                _successful_repair(selected_plan, 0, 0),
                _successful_repair(selected_plan, 1, 0),
            ),
            requested=(0, 1),
        )
        partial_plan = replace(
            selected_plan,
            choices=(
                replace(
                    selected_plan.choices[0],
                    canonical_master_end=0.9,
                    source_local_end=0.9,
                ),
                replace(
                    selected_plan.choices[1],
                    canonical_master_start=0.9,
                    source_local_start=0.9,
                ),
                selected_plan.choices[2],
            ),
        )
        targeted_repairs = _repair_plan(
            selected_plan,
            result_timeline=partial_plan,
            repairs=(_successful_repair(partial_plan, 0, -3),),
            requested=(0,),
        )
        graph = LazySpliceFeasibilityGraph("partial-lineage")
        _record_boundary_repair_assessments(graph, selected_repairs)
        _record_boundary_repair_assessments(graph, targeted_repairs)

        with patch(
            "crosscut.visual_boundary_repair.VisualBoundaryRepair",
            SimpleNamespace,
        ):
            merged, authorization = (
                _authorize_partial_reedit_splice_lineage(
                    graph,
                    selected_repairs=selected_repairs,
                    targeted_repairs=targeted_repairs,
                    final_timeline=partial_plan,
                )
            )

        self.assertEqual(len(merged.attempt_input_timelines), 2)
        self.assertTrue(authorization["proxy_eligible"])
        self.assertEqual(authorization["cross_source_edge_count"], 2)

    def test_geometry_proof_requires_nominal_final_boundary_offset(self) -> None:
        plan = _plan(_choice(0, "0:a"), _choice(1, "1:b"))
        for offset, expected in ((7, False), (0, True)):
            repair_plan = _repair_plan(
                plan,
                failures=(
                    _failure(
                        plan,
                        0,
                        (_audit(offset, focus=1.0),),
                    ),
                ),
                requested=(0,),
            )
            assessment = _boundary_repair_assessments(repair_plan)[0][2]
            self.assertIs(
                assessment.status,
                SpliceFeasibilityStatus.RAW_GEOMETRY_PROOF_REQUIRED,
            )
            self.assertEqual(
                _assessment_has_final_boundary_geometry_evidence(assessment),
                expected,
            )
            graph = LazySpliceFeasibilityGraph(f"geometry-{offset}")
            _record_boundary_repair_assessments(graph, repair_plan)
            authorization = _candidate_local_splice_proxy_authorization(
                graph,
                final_timeline=plan,
                repair_plan=repair_plan,
            )
            self.assertEqual(authorization["proxy_eligible"], expected)
            self.assertEqual(
                authorization["geometry_proof_boundary_indices"],
                [0] if expected else [],
            )

    def test_incomplete_normal_evidence_requests_one_terminal_wide_refinement(
        self,
    ) -> None:
        plan = _plan(_choice(0, "0:a"), _choice(1, "1:b"))
        narrow = _repair_plan(
            plan,
            failures=(
                _failure(
                    plan,
                    0,
                    tuple(
                        _audit(offset, silhouette=0.60)
                        for offset in range(-3, 4)
                    ),
                ),
            ),
            requested=(0,),
        )
        graph = LazySpliceFeasibilityGraph("partial-to-wide")
        _record_boundary_repair_assessments(graph, narrow)
        initial = _candidate_local_splice_proxy_authorization(
            graph,
            final_timeline=plan,
            repair_plan=narrow,
        )
        self.assertFalse(initial["proxy_eligible"])
        self.assertEqual(
            _splice_authorization_refinement_boundary_indices(initial),
            (0,),
        )

        wide_pass = _repair_plan(
            plan,
            repairs=(_successful_repair(plan, 0, 0),),
            requested=(0,),
        )
        _record_boundary_repair_assessments(graph, wide_pass)
        with patch(
            "crosscut.visual_boundary_repair.VisualBoundaryRepair",
            SimpleNamespace,
        ):
            merged = _merge_visual_boundary_repair_plans(
                narrow,
                wide_pass,
                analysis_suffix="test_terminal_wide_refinement",
            )
        terminal = _candidate_local_splice_proxy_authorization(
            graph,
            final_timeline=plan,
            repair_plan=merged,
        )
        self.assertTrue(terminal["proxy_eligible"])
        self.assertEqual(
            _splice_authorization_refinement_boundary_indices(terminal),
            (),
        )

    def test_missing_descriptor_initial_or_wide_never_invokes_proxy(self) -> None:
        plan = _plan(_choice(0, "0:a"), _choice(1, "1:b"))
        cases = (
            tuple(_missing_audit(offset) for offset in range(-3, 4)),
            tuple(
                _missing_audit(offset)
                for offset in DEFAULT_WIDE_FRAME_OFFSETS
            ),
            tuple(
                _missing_audit(offset)
                if offset == 0
                else _audit(offset, silhouette=0.60)
                for offset in DEFAULT_WIDE_FRAME_OFFSETS
            ),
        )
        for case_index, audit in enumerate(cases):
            graph = LazySpliceFeasibilityGraph(f"missing-{case_index}")
            repair_plan = _repair_plan(
                plan,
                failures=(_failure(plan, 0, audit),),
                requested=(0,),
            )
            _record_boundary_repair_assessments(graph, repair_plan)
            authorization = _candidate_local_splice_proxy_authorization(
                graph,
                final_timeline=plan,
                repair_plan=repair_plan,
            )
            proxy_calls: list[str] = []
            if authorization["proxy_eligible"]:
                proxy_calls.append("decoded-proxy")

            self.assertFalse(authorization["proxy_eligible"])
            self.assertEqual(proxy_calls, [])

    def test_plus_five_wide_plan_is_passed_to_derivation_without_replacement(
        self,
    ) -> None:
        plan = _plan(_choice(0, "0:a"), _choice(1, "1:b"))
        wide_plan = _repair_plan(
            plan,
            repairs=(_successful_repair(plan, 0, 5),),
            requested=(0,),
        )
        stream = _GlobalVisualCandidateStream()
        candidate = stream.admit_unseen(_materialized(*plan.choices))
        assert candidate is not None
        received: list[VisualBoundaryRepairPlan | None] = []

        audit = _run_initial_edge_constrained_stream(
            (candidate,),
            preflight=lambda _candidate: ((), wide_plan, 1),
            derive=lambda _candidate, repairs: received.append(repairs),
            reject=lambda _candidate, _edges: self.fail("unexpected rejection"),
            should_stop=lambda: False,
        )

        self.assertEqual(received, [wide_plan])
        self.assertEqual(audit["derived_count"], 1)
        self.assertEqual(audit["preflight_evidence_change_count"], 1)

    def test_precomputed_success_from_another_candidate_is_rejected(self) -> None:
        plan_a = _plan(_choice(0, "0:a"), _choice(1, "1:b"))
        plan_b = replace(
            plan_a,
            choices=(
                replace(
                    plan_a.choices[0],
                    canonical_master_end=0.95,
                    source_local_end=0.95,
                ),
                replace(
                    plan_a.choices[1],
                    canonical_master_start=0.95,
                    source_local_start=0.95,
                ),
            ),
        )
        repair_from_a = _repair_plan(
            plan_a,
            repairs=(_successful_repair(plan_a, 0, 0),),
            requested=(0,),
        )

        with self.assertRaisesRegex(RuntimeError, "origin does not match"):
            _validate_constrained_precomputed_repair_origin(
                plan_b,
                repair_from_a,
            )

    def test_typed_no_path_is_handled_and_untyped_value_error_propagates(self) -> None:
        no_path = NoCompleteTimelinePath(
            "no complete path",
            reason_code="forward_reachability_exhausted",
            segment_index=4,
            rejection_counts={"forbidden_edge": 2},
        )
        with patch(
            "crosscut.pipeline.regenerate_visual_timeline_candidate_set",
            side_effect=no_path,
        ):
            regenerated, evidence = _regenerate_visual_candidate_set_or_no_path(
                SimpleNamespace(),  # type: ignore[arg-type]
                forbidden_boundary_edges=TimelineNoGoodSet(),
            )
        self.assertIsNone(regenerated)
        self.assertEqual(evidence, no_path.to_dict())

        with patch(
            "crosscut.pipeline.regenerate_visual_timeline_candidate_set",
            side_effect=ValueError("schema drift"),
        ):
            with self.assertRaisesRegex(ValueError, "schema drift"):
                _regenerate_visual_candidate_set_or_no_path(
                    SimpleNamespace(),  # type: ignore[arg-type]
                    forbidden_boundary_edges=TimelineNoGoodSet(),
                )

    def test_legacy_graph_is_discarded_but_malformed_current_graph_aborts(
        self,
    ) -> None:
        expected_basis = "basis-v4"
        for legacy_version in (1, 2, 3):
            reset, resumed, reason = (
                _resume_or_reset_splice_feasibility_graph(
                    {
                        "contract": (
                            "analysis_bound_exact_splice_feasibility_graph_"
                            f"v{legacy_version}"
                        ),
                        "legacy_payload": True,
                    },
                    expected_analysis_basis_identity=expected_basis,
                )
            )
            self.assertFalse(resumed)
            self.assertEqual(len(reset), 0)
            self.assertIn("known_legacy", reason or "")

        stale_basis_payload = LazySpliceFeasibilityGraph(
            "old-basis"
        ).to_dict()
        reset_basis, resumed_basis, basis_reason = (
            _resume_or_reset_splice_feasibility_graph(
                stale_basis_payload,
                expected_analysis_basis_identity=expected_basis,
            )
        )
        self.assertFalse(resumed_basis)
        self.assertEqual(reset_basis.analysis_basis_identity, expected_basis)
        self.assertIn("basis_mismatch", basis_reason or "")

        malformed_current = LazySpliceFeasibilityGraph(
            expected_basis
        ).to_dict()
        malformed_current["fingerprint"] = "0" * 64
        with self.assertRaisesRegex(
            RuntimeError,
            "canonical validation",
        ):
            _resume_or_reset_splice_feasibility_graph(
                malformed_current,
                expected_analysis_basis_identity=expected_basis,
            )

    def test_resumed_indeterminate_context_is_reopened_for_new_run(self) -> None:
        plan = _plan(_choice(0, "0:a"), _choice(1, "1:b"))
        graph = LazySpliceFeasibilityGraph("same-basis")
        repair_plan = _repair_plan(
            plan,
            failures=(
                _failure(
                    plan,
                    0,
                    tuple(
                        _missing_audit(offset)
                        for offset in DEFAULT_WIDE_FRAME_OFFSETS
                    ),
                ),
            ),
            requested=(0,),
        )
        _record_boundary_repair_assessments(graph, repair_plan)
        edge, context = _cross_source_timeline_edge_contexts(plan)[0]
        self.assertFalse(graph.needs_assessment(edge, context))

        reopened, resumed, reason = _resume_or_reset_splice_feasibility_graph(
            graph.to_dict(),
            expected_analysis_basis_identity="same-basis",
        )
        self.assertTrue(resumed)
        self.assertTrue(reopened.needs_assessment(edge, context))
        self.assertIn("dropped_run_local_indeterminate", reason or "")

    def test_workspace_or_source_identity_changes_splice_graph_basis(self) -> None:
        first = _group_splice_analysis_basis_identity(
            selector_input_identity_sha256="a" * 64,
            selector_input_identity_basis="verified-workspace-v1",
        )
        swapped_source = _group_splice_analysis_basis_identity(
            selector_input_identity_sha256="b" * 64,
            selector_input_identity_basis="verified-workspace-v1",
        )
        swapped_basis = _group_splice_analysis_basis_identity(
            selector_input_identity_sha256="a" * 64,
            selector_input_identity_basis="verified-checkpoint-v1",
        )

        self.assertNotEqual(first, swapped_source)
        self.assertNotEqual(first, swapped_basis)

    def test_threshold_or_classifier_contract_change_invalidates_graph_basis(
        self,
    ) -> None:
        default_review = TimelineReviewConfig()
        baseline = _group_splice_analysis_basis_identity(
            selector_input_identity_sha256="a" * 64,
            selector_input_identity_basis="verified-workspace-v1",
            timeline_review_config=default_review,
        )
        changed_threshold = _group_splice_analysis_basis_identity(
            selector_input_identity_sha256="a" * 64,
            selector_input_identity_basis="verified-workspace-v1",
            timeline_review_config=replace(
                default_review,
                maximum_scale_mismatch=(
                    default_review.maximum_scale_mismatch + 0.01
                ),
            ),
        )
        changed_classifier = _group_splice_analysis_basis_identity(
            selector_input_identity_sha256="a" * 64,
            selector_input_identity_basis="verified-workspace-v1",
            timeline_review_config=default_review,
            splice_feasibility_contract="test-classifier-contract-v999",
        )

        self.assertNotEqual(baseline, changed_threshold)
        self.assertNotEqual(baseline, changed_classifier)

    def test_edge_domain_resource_bound_counts_exact_cross_source_pairs(self) -> None:
        lattice = tuple(
            tuple(
                SimpleNamespace(source_id=str(source), shot_id=f"{source}:{segment}")
                for source in (0, 1)
            )
            for segment in range(3)
        )
        candidate_set = SimpleNamespace(
            optimizer_workspace=SimpleNamespace(candidates_by_segment=lattice)
        )
        self.assertEqual(
            _optimizer_workspace_cross_source_edge_bound(candidate_set),  # type: ignore[arg-type]
            4,
        )

    def test_boundary_search_budget_is_unique_context_bounded_and_fail_closed(
        self,
    ) -> None:
        baseline = _plan(_choice(0, "0:a"), _choice(1, "1:b"))
        budget = _BoundarySearchBudget(structural_edge_bound=0)
        for index in range(16):
            boundary = 0.80 + index * 0.005
            plan = replace(
                baseline,
                choices=(
                    replace(
                        baseline.choices[0],
                        canonical_master_end=boundary,
                        source_local_end=boundary,
                    ),
                    replace(
                        baseline.choices[1],
                        canonical_master_start=boundary,
                        source_local_start=boundary,
                    ),
                ),
            )
            budget.reserve(
                phase="test_wide",
                visual_timeline=plan,
                boundary_indices=(0,),
                search_radius_frames=8,
            )
            # An identical request is telemetry-only reuse, not another unique
            # decoded search token.
            budget.reserve(
                phase="test_wide_reuse",
                visual_timeline=plan,
                boundary_indices=(0,),
                search_radius_frames=8,
            )

        exhausted_boundary = 0.95
        exhausted_plan = replace(
            baseline,
            choices=(
                replace(
                    baseline.choices[0],
                    canonical_master_end=exhausted_boundary,
                    source_local_end=exhausted_boundary,
                ),
                replace(
                    baseline.choices[1],
                    canonical_master_start=exhausted_boundary,
                    source_local_start=exhausted_boundary,
                ),
            ),
        )
        with self.assertRaises(_BoundarySearchBudgetExhausted):
            budget.reserve(
                phase="test_wide_exhausted",
                visual_timeline=exhausted_plan,
                boundary_indices=(0,),
                search_radius_frames=8,
            )
        audit = budget.to_dict()
        self.assertEqual(audit["unique_boundary_search_count"], 16)
        self.assertEqual(audit["planned_offset_evaluation_count"], 272)
        self.assertTrue(audit["exhausted"])
        self.assertTrue(audit["production_use_forbidden"])

    def test_failure_assessments_are_retained_instead_of_status_collapsed(self) -> None:
        plan = _plan(_choice(0, "0:a"), _choice(1, "1:b"))
        narrow = _repair_plan(
            plan,
            failures=(
                _failure(
                    plan,
                    0,
                    tuple(
                        _audit(offset, silhouette=0.60)
                        for offset in range(-3, 4)
                    ),
                ),
            ),
        )
        assessments = _boundary_repair_assessments(narrow)
        self.assertEqual(len(assessments), 1)
        self.assertEqual(
            assessments[0][2].observed_frame_offsets,
            tuple(range(-3, 4)),
        )
        self.assertIs(
            assessments[0][2].status,
            SpliceFeasibilityStatus.INDETERMINATE,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
