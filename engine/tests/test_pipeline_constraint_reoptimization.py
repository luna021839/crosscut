from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from crosscut.lazy_splice_feasibility_graph import LazySpliceFeasibilityGraph
from crosscut.boundary_repair import BoundaryRepairConfig
from crosscut.editorial_motion import EditorialMotionPlan, EditorialMotionSegment
from crosscut.ive import AspectRatio, EditMode, SourceMode
from crosscut.joint_visual_candidate_adapter import (
    JointMaterializedVisualCandidate,
)
from crosscut.pipeline import (
    ELIGIBLE_SOURCE_INDEX_DOMAIN,
    _BoundarySearchBudget,
    _BoundarySearchBudgetExhausted,
    _BoundarySearchNoProgress,
    _GlobalVisualCandidateStream,
    _assessment_has_final_boundary_geometry_evidence,
    _authorize_partial_reedit_splice_lineage,
    _boundary_repair_assessments,
    _budgeted_repair_visual_timeline_boundaries,
    _candidate_local_splice_proxy_authorization,
    _claim_exact_splice_refinement_keys,
    _constraint_terminal_status,
    _cross_source_timeline_edge_contexts,
    _cross_source_timeline_edge_keys,
    _derive_complete_group_candidate,
    _derive_complete_joint_group_candidate,
    _enforce_automatic_group_stage_edit_requirement,
    _group_splice_analysis_basis_identity,
    _incomplete_splice_context_keys,
    _optimizer_workspace_cross_source_edge_bound,
    _optimizer_blocked_set_for_contexts,
    _record_boundary_repair_assessments,
    _joint_group_hard_cut_transition,
    _raise_joint_visual_selection_failure,
    _regenerate_visual_candidate_set_or_no_path,
    _route_visual_timeline_selection_failure,
    _resume_or_reset_splice_feasibility_graph,
    _run_initial_edge_constrained_stream,
    _source_index_mapping,
    _splice_authorization_evidence_incomplete,
    _splice_authorization_refinement_boundary_indices,
    _splice_graph_keys_for_plan,
    _validate_constrained_precomputed_repair_origin,
    _visual_feature_cache_arguments,
    _merge_visual_boundary_repair_plans,
    _attempt_proxy_triggered_partial_reedit,
)
from crosscut.render import (
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    RenderSegment,
    TransitionKind,
)
from crosscut.timeline import NoCompleteTimelinePath, ShotType
from crosscut.timeline_no_good import TimelineNoGoodSet
from crosscut.timeline_review import TimelineReviewConfig
from crosscut.visual_boundary_repair import VisualBoundaryRepairPlan
from crosscut.visual_score import (
    PersistentVisualFeatureCacheInitializationError,
    PersistentVisualFeatureCacheRuntimeError,
    VisualTimelineChoice,
    VisualTimelinePlan,
)
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
        candidate_evidence_audit=tuple(
            (
                {
                    "frame_offset": offset,
                    "professional_gate_passed": True,
                    "production_transition_gate_passed": True,
                    "visual_evidence": {},
                }
                if offset == frame_offset
                else _audit(offset, silhouette=0.60)
            )
            for offset in DEFAULT_WIDE_FRAME_OFFSETS
        ),
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
    def test_direct_visual_feature_cache_is_strict_while_legacy_fallback_remains(
        self,
    ) -> None:
        legacy_directory = Path("/legacy/source-cache")
        direct_directory = Path("/direct/derived-cache")

        legacy = _visual_feature_cache_arguments(
            SimpleNamespace(
                source_cache_directory=legacy_directory,
                visual_feature_derived_cache_directory=None,
            )
        )
        self.assertEqual(legacy["feature_cache_directory"], legacy_directory)
        self.assertIsNone(legacy["feature_derived_cache_directory"])
        self.assertIs(legacy["require_persistent_feature_cache"], False)

        direct = _visual_feature_cache_arguments(
            SimpleNamespace(
                source_cache_directory=legacy_directory,
                visual_feature_derived_cache_directory=direct_directory,
            )
        )
        self.assertEqual(direct["feature_cache_directory"], legacy_directory)
        self.assertEqual(
            direct["feature_derived_cache_directory"],
            direct_directory,
        )
        self.assertIs(direct["require_persistent_feature_cache"], True)

    def test_partial_reedit_propagates_required_cache_failures(self) -> None:
        plan = _plan(_choice(0, "0:a"), _choice(1, "1:b"))
        selected_bundle = SimpleNamespace(
            candidate_origin="test",
            candidate_lineage=(),
            visual_timeline=plan,
            boundary_repairs=object(),
            splice_graph_authorization={"proxy_eligible": True},
        )
        review_payload = {
            "passed": False,
            "partial_reedit_boundary_issue_indices": [0],
        }
        request = SimpleNamespace(
            sources=(Path("source-0.mp4"), Path("source-1.mp4")),
            source_cache_directory=Path("legacy-cache"),
            visual_feature_derived_cache_directory=Path("derived-cache"),
        )

        for error_type in (
            PersistentVisualFeatureCacheInitializationError,
            PersistentVisualFeatureCacheRuntimeError,
        ):
            with self.subTest(error_type=error_type.__name__), patch(
                "crosscut.pipeline._budgeted_repair_visual_timeline_boundaries",
                side_effect=error_type("required cache unavailable"),
            ):
                with self.assertRaises(error_type):
                    _attempt_proxy_triggered_partial_reedit(
                        selected_bundle,
                        object(),
                        review_payload,
                        request=request,
                        eligible_indices=(0, 1),
                        audio_offsets=(0.0, 0.0),
                        scene_cut_times=((), ()),
                        source_timewarps=(None, None),
                        source_aspect_ratios=(16 / 9, 16 / 9),
                        source_durations_seconds=(10.0, 10.0),
                        beat_period_seconds=None,
                        phrase_times=(),
                        section_plan=None,
                        common_master_start=0.0,
                        width=1920,
                        height=1080,
                        scratch_directory=Path("scratch"),
                        boundary_descriptor_cache=object(),
                        splice_feasibility_graph=object(),
                        boundary_search_budget=object(),
                    )

    def test_automatic_group_authority_names_joint_primary_pool(self) -> None:
        request = SimpleNamespace(
            source_mode=SourceMode.AUTOMATIC,
            edit_mode=EditMode.GROUP,
            acquire_only=False,
        )

        authority = _enforce_automatic_group_stage_edit_requirement(
            request,  # type: ignore[arg-type]
            None,
            review_rendered_proxies=True,
        )

        assert authority is not None
        self.assertEqual(
            authority["candidate_pool"],
            "joint_decoded_visual_selector",
        )
        self.assertTrue(
            authority["joint_decoded_visual_candidate_is_primary_not_fallback"]
        )
        self.assertNotIn(
            "ordinary_visual_candidate_is_primary_not_fallback",
            authority,
        )

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

    def test_success_records_full_raw_audit_and_separate_result_proof(
        self,
    ) -> None:
        plan = _plan(_choice(0, "0:a"), _choice(1, "1:b"))
        repair = _successful_repair(plan, 0, 5)
        repair_plan = _repair_plan(
            plan,
            repairs=(repair,),
            requested=(0,),
        )

        assessments = _boundary_repair_assessments(repair_plan)
        by_context = {
            context_identity: assessment
            for _edge, context_identity, assessment in assessments
        }
        raw = by_context[repair.observed_edge_context_identity]
        result = by_context[repair.result_edge_context_identity]

        self.assertEqual(
            raw.observed_frame_offsets,
            DEFAULT_WIDE_FRAME_OFFSETS,
        )
        self.assertIs(raw.status, SpliceFeasibilityStatus.RAW_PASS)
        self.assertEqual(result.observed_frame_offsets, (5,))
        self.assertIs(result.status, SpliceFeasibilityStatus.RAW_PASS)
        self.assertNotEqual(
            repair.observed_edge_context_identity,
            repair.result_edge_context_identity,
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
            repairs=(_successful_repair(plan, 0, 5),),
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

        balanced_failure = NoCompleteTimelinePath(
            "balanced exhausted",
            reason_code="forward_reachability_exhausted",
            segment_index=4,
            rejection_counts={"exact_boundary_no_good": 3},
        )
        dense_failure = NoCompleteTimelinePath(
            "dense exhausted",
            reason_code="forward_reachability_exhausted",
            segment_index=7,
            rejection_counts={"exact_boundary_no_good": 5},
        )
        profile_no_path = NoCompleteTimelinePath(
            "all profiles exhausted",
            reason_code="all_safe_cadence_profiles_exhausted",
            rejection_counts={
                "balanced:forward_reachability_exhausted": 1,
                "dense_safe:forward_reachability_exhausted": 1,
            },
            profile_failures=(
                ("balanced", balanced_failure),
                ("dense_safe", dense_failure),
            ),
        )
        with patch(
            "crosscut.pipeline.regenerate_visual_timeline_candidate_set",
            side_effect=profile_no_path,
        ):
            _regenerated, profile_evidence = (
                _regenerate_visual_candidate_set_or_no_path(
                    SimpleNamespace(),  # type: ignore[arg-type]
                    forbidden_boundary_edges=TimelineNoGoodSet(),
                )
            )
        assert profile_evidence is not None
        self.assertEqual(
            [
                item["cadence_profile"]
                for item in profile_evidence[
                    "nested_profile_diagnostics"
                ]
            ],
            ["balanced", "dense_safe"],
        )
        self.assertEqual(
            profile_evidence["nested_profile_diagnostics"][0][
                "no_complete_path"
            ],
            balanced_failure.to_dict(),
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
            with self.assertRaises(_BoundarySearchNoProgress):
                budget.reserve(
                    phase="test_wide_reuse_without_result_memo",
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

    def test_exact_repair_request_uses_result_memo_without_redecode(self) -> None:
        plan = _plan(_choice(0, "0:a"), _choice(1, "1:b"))
        repair_plan = _repair_plan(plan, requested=(0,))
        budget = _BoundarySearchBudget(structural_edge_bound=1)
        settings = BoundaryRepairConfig(
            search_radius_frames=8,
            interior_scene_cut_guard_frames=9,
        )
        arguments = {
            "budget": budget,
            "phase": "test_exact_memo",
            "visual_timeline": plan,
            "sources": (Path("/tmp/source-a.mp4"), Path("/tmp/source-b.mp4")),
            "audio_offsets": (0.0, 0.0),
            "scene_cut_times": ((), ()),
            "beat_period_seconds": 0.5,
            "source_timewarps": (None, None),
            "boundary_indices": (0,),
            "config": settings,
        }
        with patch(
            "crosscut.pipeline.repair_visual_timeline_boundaries",
            return_value=repair_plan,
        ) as repair:
            first = _budgeted_repair_visual_timeline_boundaries(**arguments)
            second = _budgeted_repair_visual_timeline_boundaries(**arguments)

        self.assertIs(first, repair_plan)
        self.assertIs(second, repair_plan)
        self.assertEqual(repair.call_count, 1)
        audit = budget.to_dict()
        self.assertEqual(audit["repair_call_count"], 1)
        self.assertEqual(audit["repair_result_memo_entry_count"], 1)
        self.assertEqual(audit["repair_result_memo_hit_count"], 1)

    def test_budgeted_boundary_repair_forwards_direct_cache_contract(self) -> None:
        plan = _plan(_choice(0, "0:a"), _choice(1, "1:b"))
        repair_plan = _repair_plan(plan, requested=(0,))
        budget = _BoundarySearchBudget(structural_edge_bound=1)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_root = root / "legacy-source-cache"
            direct_root = root / "mounted-derived-cache"
            direct_root.mkdir()

            with patch(
                "crosscut.pipeline.repair_visual_timeline_boundaries",
                return_value=repair_plan,
            ) as repair:
                result = _budgeted_repair_visual_timeline_boundaries(
                    budget=budget,
                    phase="test_direct_cache_contract",
                    visual_timeline=plan,
                    sources=(Path("/tmp/source-a.mp4"), Path("/tmp/source-b.mp4")),
                    audio_offsets=(0.0, 0.0),
                    scene_cut_times=((), ()),
                    beat_period_seconds=0.5,
                    source_timewarps=(None, None),
                    boundary_indices=(0,),
                    feature_cache_directory=source_root,
                    feature_derived_cache_directory=direct_root,
                    require_persistent_feature_cache=True,
                )

        self.assertIs(result, repair_plan)
        self.assertEqual(
            repair.call_args.kwargs["feature_cache_directory"],
            source_root,
        )
        self.assertEqual(
            repair.call_args.kwargs["feature_derived_cache_directory"],
            direct_root,
        )
        self.assertIs(
            repair.call_args.kwargs["require_persistent_feature_cache"],
            True,
        )

    def test_required_cache_root_loss_preempts_repair_result_memo(self) -> None:
        plan = _plan(_choice(0, "0:a"), _choice(1, "1:b"))
        repair_plan = _repair_plan(plan, requested=(0,))
        budget = _BoundarySearchBudget(structural_edge_bound=1)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            direct_root = root / "mounted-derived-cache"
            direct_root.mkdir()
            arguments = {
                "budget": budget,
                "phase": "test_required_cache_memo_liveness",
                "visual_timeline": plan,
                "sources": (Path("/tmp/source-a.mp4"), Path("/tmp/source-b.mp4")),
                "audio_offsets": (0.0, 0.0),
                "scene_cut_times": ((), ()),
                "beat_period_seconds": 0.5,
                "source_timewarps": (None, None),
                "boundary_indices": (0,),
                "feature_derived_cache_directory": direct_root,
                "require_persistent_feature_cache": True,
            }
            with patch(
                "crosscut.pipeline.repair_visual_timeline_boundaries",
                return_value=repair_plan,
            ) as repair:
                first = _budgeted_repair_visual_timeline_boundaries(**arguments)
                direct_root.rename(root / "detached-derived-cache")
                with self.assertRaises(
                    PersistentVisualFeatureCacheRuntimeError
                ) as raised:
                    _budgeted_repair_visual_timeline_boundaries(**arguments)

            self.assertIs(first, repair_plan)
            self.assertEqual(repair.call_count, 1)
            self.assertNotIn(str(root), str(raised.exception))

    def test_094259_boundary65_exact_raw_context_search_count_is_one(
        self,
    ) -> None:
        plan = _plan(
            *tuple(
                _choice(index % 2, f"{index % 2}:shot-{index}")
                for index in range(67)
            )
        )
        key = next(
            item
            for item in _splice_graph_keys_for_plan(plan)
            if item.edge.boundary_index == 65
        )
        attempted = set()

        self.assertEqual(
            _claim_exact_splice_refinement_keys(attempted, (key,)),
            (key,),
        )
        self.assertEqual(
            _claim_exact_splice_refinement_keys(attempted, (key,)),
            (),
        )
        self.assertEqual(attempted, {key})

    def test_terminal_status_keeps_structural_evidence_and_resource_split(
        self,
    ) -> None:
        cases = (
            (
                dict(
                    structural_hard_no_path=True,
                    resource_incomplete=False,
                    incomplete_exact_context_count=0,
                    authorized_path_count=0,
                ),
                "structural_hard_no_path",
            ),
            (
                dict(
                    structural_hard_no_path=False,
                    resource_incomplete=False,
                    incomplete_exact_context_count=1,
                    authorized_path_count=0,
                ),
                "evidence_incomplete_before_authorized_cohort",
            ),
            (
                dict(
                    structural_hard_no_path=False,
                    resource_incomplete=True,
                    incomplete_exact_context_count=0,
                    authorized_path_count=0,
                ),
                "resource_incomplete",
            ),
            (
                dict(
                    structural_hard_no_path=False,
                    resource_incomplete=False,
                    incomplete_exact_context_count=0,
                    authorized_path_count=3,
                ),
                "authorized_path_count_insufficient",
            ),
        )
        for arguments, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    _constraint_terminal_status(**arguments),
                    expected,
                )

    def test_splice_authorization_separates_hard_from_incomplete_evidence(
        self,
    ) -> None:
        complete_hard = {
            "evidence": [
                {
                    "authorization_failure": (
                        "fresh_candidate_edge_hard_incompatible"
                    ),
                    "fresh_status": "hard_incompatible",
                    "fresh_terminal": True,
                }
            ]
        }
        missing = {
            "evidence": [
                {
                    "authorization_failure": (
                        "fresh_candidate_edge_evidence_missing"
                    ),
                }
            ]
        }
        hard_result = {
            "evidence": [
                {
                    "authorization_failure": (
                        "fresh_candidate_result_proof_invalid"
                    ),
                    "fresh_result_status": "hard_incompatible",
                    "fresh_result_terminal": True,
                }
            ]
        }
        indeterminate_result = {
            "evidence": [
                {
                    "authorization_failure": (
                        "fresh_candidate_result_proof_invalid"
                    ),
                    "fresh_result_status": "indeterminate",
                    "fresh_result_terminal": False,
                }
            ]
        }

        self.assertFalse(
            _splice_authorization_evidence_incomplete(complete_hard)
        )
        self.assertFalse(
            _splice_authorization_evidence_incomplete(hard_result)
        )
        self.assertTrue(
            _splice_authorization_evidence_incomplete(missing)
        )
        self.assertTrue(
            _splice_authorization_evidence_incomplete(indeterminate_result)
        )
        self.assertTrue(_splice_authorization_evidence_incomplete({}))

    def test_094259_terminal_indeterminate_is_never_structural_dp_no_good(
        self,
    ) -> None:
        plan = _plan(_choice(0, "0:a"), _choice(1, "1:b"))
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
        graph = LazySpliceFeasibilityGraph("indeterminate-not-structural")
        _record_boundary_repair_assessments(graph, repair_plan)
        keys = _splice_graph_keys_for_plan(plan)
        edge = keys[0].edge

        self.assertIn(edge, graph.blocked_set_for_contexts(keys))
        self.assertNotIn(
            edge,
            _optimizer_blocked_set_for_contexts(graph, keys),
        )
        self.assertEqual(
            _incomplete_splice_context_keys(graph, keys),
            keys,
        )

    def test_unmeasured_context_is_evidence_incomplete(self) -> None:
        plan = _plan(_choice(0, "0:a"), _choice(1, "1:b"))
        graph = LazySpliceFeasibilityGraph("unmeasured-is-incomplete")
        keys = _splice_graph_keys_for_plan(plan)

        self.assertEqual(
            _incomplete_splice_context_keys(graph, keys),
            keys,
        )

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

    def test_joint_complete_derivation_keeps_provider_tracks_and_hard_cuts(
        self,
    ) -> None:
        tracks = tuple(
            EditorialTransformTrack(
                (
                    EditorialTransformKeyframe(
                        time_seconds=0.0,
                        zoom=1.0 + 0.05 * index,
                    ),
                    EditorialTransformKeyframe(
                        time_seconds=1.0,
                        zoom=1.0 + 0.05 * index,
                    ),
                ),
                reason=f"joint-track-{index}",
            )
            for index in range(2)
        )
        segments = tuple(
            RenderSegment(
                source_index=index,
                source_start=float(index),
                duration=1.0,
                master_start=float(index),
                editorial_transform=tracks[index],
            )
            for index in range(2)
        )
        motion = EditorialMotionPlan(
            segments=tuple(
                EditorialMotionSegment(
                    segment_index=index,
                    source_index=index,
                    track=tracks[index],
                    reasons=("joint_decoded_segment_transform",),
                )
                for index in range(2)
            ),
            boundary_match_count=1,
            zoom_event_count=1,
            pan_event_count=0,
            roll_event_count=0,
            blocked_dissolve_segment_count=0,
            analysis_basis="joint_decoded_test_motion_v1",
            accepted_boundary_indices=(0,),
        )
        proof_bundle = SimpleNamespace(
            selection_identity="joint-selection:test",
            to_dict=lambda: {
                "selection_identity": "joint-selection:test",
                "raw_frames_persisted": False,
            },
        )
        materialized = SimpleNamespace(
            common_master_start=0.0,
            visual_plan=_plan(_choice(0, "0:a"), _choice(1, "1:b")),
            editorial_motion_plan=motion,
            segments=segments,
            output_fps=29.97,
            boundary_proof_bundle=proof_bundle,
            candidate_index=3,
            review_rank=0,
            cadence_profile="joint-balanced",
            privacy_safe_signature="joint-visual-path:test",
            candidate_history_entry={"joint_path_signature": "test"},
        )
        request = SimpleNamespace(
            sources=tuple(
                Path(f"/tmp/joint-source-{index}.mp4")
                for index in range(4)
            ),
            reference_source=0,
            aspect_ratio=AspectRatio.PORTRAIT,
            source_mode=SourceMode.AUTOMATIC,
        )
        review = SimpleNamespace(
            passed=True,
            weighted_score=91.5,
            hard_failures=(),
        )
        with patch("crosscut.pipeline.review_timeline", return_value=review):
            bundle = _derive_complete_joint_group_candidate(
                materialized,  # type: ignore[arg-type]
                request=request,  # type: ignore[arg-type]
                eligible_indices=(1, 3),
                source_aspect_ratios=(16.0 / 9.0, 16.0 / 9.0),
                phrase_times=(),
                section_plan=None,
                common_master_start=0.0,
                width=1080,
                height=1920,
                output=Path("/tmp/joint-output.mp4"),
                source_look_match=None,
                boundary_look_measurement_cache=None,
            )

        self.assertEqual(bundle.candidate_origin, "joint_decoded_visual_selector")
        self.assertIsNone(bundle.boundary_repairs)
        self.assertIsNone(bundle.face_match_plan)
        self.assertIsNone(bundle.boundary_affine_plan)
        self.assertEqual(
            tuple(item.source_index for item in bundle.render_plan.segments),
            (1, 3),
        )
        self.assertEqual(
            tuple(
                item.source_index for item in bundle.editorial_motion_plan.segments
            ),
            (0, 1),
        )
        for index, segment in enumerate(bundle.render_plan.segments):
            self.assertIs(segment.editorial_transform, tracks[index])
            self.assertIs(
                bundle.editorial_motion_plan.segments[index].track,
                tracks[index],
            )
        self.assertEqual(bundle.group_transition_plan.dissolve_count, 0)
        self.assertTrue(
            all(
                item.kind is TransitionKind.HARD_CUT
                for item in bundle.group_transition_plan.transitions
            )
        )
        self.assertTrue(bundle.splice_graph_authorization["proxy_eligible"])
        self.assertEqual(
            bundle.candidate_lineage[0]["candidate_pool"],
            "joint_decoded_visual_selector",
        )

    def test_joint_motion_identity_stays_in_eligible_domain_with_request_gaps(
        self,
    ) -> None:
        track = EditorialTransformTrack(
            (
                EditorialTransformKeyframe(time_seconds=0.0, zoom=1.0),
                EditorialTransformKeyframe(time_seconds=1.0, zoom=1.0),
            ),
            reason="joint-identity",
        )
        motion = EditorialMotionPlan(
            segments=(
                EditorialMotionSegment(0, 0, track),
                EditorialMotionSegment(1, 1, track),
            ),
            boundary_match_count=1,
            zoom_event_count=0,
            pan_event_count=0,
            roll_event_count=0,
            blocked_dissolve_segment_count=0,
            analysis_basis="joint_identity_test_v1",
        )
        bundle_request = SimpleNamespace(
            sources=tuple(Path(f"/tmp/source-{index}.mp4") for index in range(4)),
            reference_source=0,
            aspect_ratio=AspectRatio.PORTRAIT,
            source_mode=SourceMode.AUTOMATIC,
        )
        materialized = SimpleNamespace(
            editorial_motion_plan=motion,
            segments=(
                RenderSegment(0, 0.0, 1.0, 0.0, editorial_transform=track),
                RenderSegment(1, 1.0, 1.0, 1.0, editorial_transform=track),
            ),
            common_master_start=0.0,
            visual_plan=_plan(_choice(0, "0:a"), _choice(1, "1:b")),
            output_fps=29.97,
            boundary_proof_bundle=SimpleNamespace(
                selection_identity="joint-selection:gapped",
                to_dict=lambda: {"selection_identity": "joint-selection:gapped"},
            ),
            candidate_index=0,
            review_rank=0,
            cadence_profile="joint-balanced",
            privacy_safe_signature="joint-visual-path:gapped",
            candidate_history_entry={},
        )
        review = SimpleNamespace(passed=True, weighted_score=1.0, hard_failures=())
        with patch("crosscut.pipeline.review_timeline", return_value=review):
            bundle = _derive_complete_joint_group_candidate(
                materialized,  # type: ignore[arg-type]
                request=bundle_request,  # type: ignore[arg-type]
                eligible_indices=(1, 3),
                source_aspect_ratios=(16.0 / 9.0, 16.0 / 9.0),
                phrase_times=(),
                section_plan=None,
                common_master_start=0.0,
                width=1080,
                height=1920,
                output=Path("/tmp/joint-gapped.mp4"),
                source_look_match=None,
                boundary_look_measurement_cache=None,
            )

        self.assertIs(bundle.editorial_motion_plan, motion)
        self.assertEqual(
            tuple(item.source_index for item in bundle.editorial_motion_plan.segments),
            (0, 1),
        )
        self.assertEqual(
            tuple(item.source_index for item in bundle.render_plan.segments),
            (1, 3),
        )

    def test_joint_transition_distinguishes_internal_shot_change_from_hold(
        self,
    ) -> None:
        same_shot = _joint_group_hard_cut_transition(
            _choice(0, "0:a"),
            _choice(0, "0:a"),
        )
        changed_shot = _joint_group_hard_cut_transition(
            _choice(0, "0:a"),
            _choice(0, "0:b"),
        )
        changed_source = _joint_group_hard_cut_transition(
            _choice(0, "0:a"),
            _choice(1, "1:b"),
        )

        self.assertEqual(
            (same_shot.tier, same_shot.reason),
            ("D", "joint_decoded_same_source_same_shot_hold"),
        )
        self.assertEqual(
            (changed_shot.tier, changed_shot.reason),
            ("C", "joint_decoded_source_internal_shot_change_hard_cut"),
        )
        self.assertEqual(
            (changed_source.tier, changed_source.reason),
            ("C", "joint_decoded_pose_motion_matched_hard_cut"),
        )

    def test_joint_value_error_is_persisted_and_fails_closed(self) -> None:
        no_path = NoCompleteTimelinePath(
            "joint path exhausted",
            reason_code="forward_reachability_exhausted",
            segment_index=4,
            rejection_counts={"exact_boundary_no_good": 2},
        )
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory)
            request = SimpleNamespace(
                sources=(Path("/tmp/a.mp4"), Path("/tmp/b.mp4")),
                output_directory=output_directory,
            )
            with self.assertRaisesRegex(RuntimeError, "failed closed"):
                _raise_joint_visual_selection_failure(
                    request=request,  # type: ignore[arg-type]
                    base_filename="joint-test",
                    eligible_indices=(0, 1),
                    exc=no_path,
                    selection_active=False,
                    materialized_candidate_count=0,
                )
            payload = json.loads(
                (
                    output_directory
                    / "joint-test-joint-visual-selection-failure.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(payload["status"], "joint_visual_selection_failed_closed")
        self.assertFalse(payload["rhythm_timeline_fallback_allowed"])
        self.assertEqual(
            payload["no_complete_timeline_path"]["reason_code"],
            "forward_reachability_exhausted",
        )

    def test_joint_failure_router_never_appends_rhythm_fallback(self) -> None:
        request = SimpleNamespace(
            sources=(Path("/tmp/a.mp4"), Path("/tmp/b.mp4")),
            output_directory=Path("/tmp"),
        )
        warnings: list[str] = []
        with patch(
            "crosscut.pipeline._raise_joint_visual_selection_failure",
            side_effect=RuntimeError("sentinel joint failure"),
        ) as fail_closed:
            with self.assertRaisesRegex(RuntimeError, "sentinel joint failure"):
                _route_visual_timeline_selection_failure(
                    request=request,  # type: ignore[arg-type]
                    base_filename="joint-test",
                    eligible_indices=(0, 1),
                    exc=ValueError("provider mismatch"),
                    joint_visual_selection_attempted=True,
                    joint_visual_selection_active=False,
                    materialized_candidate_count=0,
                    warnings=warnings,
                )

        fail_closed.assert_called_once()
        self.assertEqual(warnings, [])

    def test_legacy_visual_failure_router_retains_rhythm_fallback(self) -> None:
        request = SimpleNamespace(
            sources=(Path("/tmp/a.mp4"), Path("/tmp/b.mp4")),
            output_directory=Path("/tmp"),
        )
        warnings: list[str] = []
        with patch(
            "crosscut.pipeline._raise_joint_visual_selection_failure"
        ) as fail_closed:
            _route_visual_timeline_selection_failure(
                request=request,  # type: ignore[arg-type]
                base_filename="legacy-test",
                eligible_indices=(0, 1),
                exc=ValueError("legacy descriptor unavailable"),
                joint_visual_selection_attempted=False,
                joint_visual_selection_active=False,
                materialized_candidate_count=0,
                warnings=warnings,
            )

        fail_closed.assert_not_called()
        self.assertEqual(len(warnings), 1)
        self.assertIn("리듬 타임라인", warnings[0])

    def test_joint_nested_source_domains_cover_final_render_copies(self) -> None:
        domains = _source_index_mapping(
            (Path("/tmp/a.mp4"), Path("/tmp/b.mp4")),
            (0, 1),
        )["payload_index_domains"]
        suffixes = (
            "boundary_proof_bundle.segment_paths[].source_index",
            "editorial_motion_plan.segments[].source_index",
        )
        for suffix in suffixes:
            self.assertEqual(
                domains[
                    "render.joint_visual_selection.candidate_histories[]."
                    f"{suffix}"
                ],
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            self.assertEqual(
                domains[
                    "render.rendered_candidate_selection."
                    "joint_visual_selection.candidate_histories[]."
                    f"{suffix}"
                ],
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )

    def test_joint_candidate_cannot_enter_legacy_splice_graph(self) -> None:
        materialized = Mock(spec=JointMaterializedVisualCandidate)
        self.assertIsInstance(materialized, JointMaterializedVisualCandidate)
        request = SimpleNamespace(
            sources=(Path("/tmp/source-a.mp4"), Path("/tmp/source-b.mp4")),
        )
        with self.assertRaisesRegex(ValueError, "lazy splice graph"):
            _derive_complete_group_candidate(
                materialized,
                request=request,  # type: ignore[arg-type]
                eligible_indices=(0, 1),
                audio_offsets=(0.0, 0.0),
                scene_cut_times=((), ()),
                source_timewarps=(None, None),
                source_aspect_ratios=(16.0 / 9.0, 16.0 / 9.0),
                source_durations_seconds=(30.0, 30.0),
                beat_period_seconds=0.5,
                phrase_times=(),
                section_plan=None,
                common_master_start=0.0,
                width=1920,
                height=1080,
                output=Path("/tmp/output.mp4"),
                splice_feasibility_graph=LazySpliceFeasibilityGraph("test"),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
