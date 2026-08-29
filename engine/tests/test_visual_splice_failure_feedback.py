from __future__ import annotations

from types import SimpleNamespace
import unittest

from crosscut.visual_splice_failure_feedback import (
    build_splice_failure_feedback,
    splice_feedback_edge_context,
    splice_feedback_edge_context_identity,
    splice_feedback_plan_identity,
    splice_feedback_result_context_identity,
)
from crosscut.visual_splice_feasibility import SpliceFeasibilityStatus


def _choice(index: int, source: int, shot: str) -> SimpleNamespace:
    return SimpleNamespace(
        segment_index=index,
        source_index=source,
        shot_id=shot,
        canonical_master_start=float(index),
        canonical_master_end=float(index + 1),
        source_local_start=float(index) + 0.25 * source,
        source_local_end=float(index + 1) + 0.25 * source,
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
    passed: bool = False,
    focus: float = 0.1,
    silhouette: float = 0.1,
) -> dict[str, object]:
    return {
        "frame_offset": offset,
        "professional_gate_passed": True,
        "production_transition_gate_passed": passed,
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


def _failure(
    plan: SimpleNamespace,
    boundary: int,
    outgoing: int,
    incoming: int,
    audit: tuple[dict[str, object], ...] | None,
    *,
    observed_plan: SimpleNamespace | None = None,
) -> dict[str, object]:
    observed = observed_plan or plan
    optimizer_context = splice_feedback_edge_context(plan, boundary)
    observed_context = splice_feedback_edge_context(observed, boundary)
    payload: dict[str, object] = {
        "boundary_index": boundary,
        "outgoing_source_index": outgoing,
        "incoming_source_index": incoming,
        "reason": "no acceptable offset",
        "outgoing_shot_id": plan.choices[boundary].shot_id,
        "incoming_shot_id": plan.choices[boundary + 1].shot_id,
        "timeline_analysis_basis": plan.analysis_basis,
        "timeline_source_clock_basis": plan.source_clock_basis,
        "timeline_path_identity": splice_feedback_plan_identity(plan),
        "optimizer_edge_context": optimizer_context,
        "optimizer_edge_context_identity": (
            splice_feedback_edge_context_identity(plan, boundary)
        ),
        "observed_edge_context": observed_context,
        "observed_edge_context_identity": (
            splice_feedback_edge_context_identity(observed, boundary)
        ),
        "optimizer_edge_context_match": observed_context == optimizer_context,
    }
    if audit is not None:
        payload["candidate_evidence_audit"] = audit
    return payload


class VisualSpliceFailureFeedbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = SimpleNamespace(
            analysis_basis="synthetic-plan-v1",
            source_clock_basis="synthetic-piecewise-clock-v1",
            choices=(
                _choice(0, 4, "4:shot-2"),
                _choice(1, 3, "3:shot-0"),
                _choice(2, 7, "7:shot-1"),
            )
        )

    def test_complete_nonwaivable_failure_becomes_exact_no_good(self) -> None:
        feedback = build_splice_failure_feedback(
            self.plan,
            (
                _failure(
                    self.plan,
                    0,
                    4,
                    3,
                    tuple(
                        _audit(offset, silhouette=0.50)
                        for offset in range(-8, 9)
                    ),
                ),
            ),
        )

        self.assertEqual(len(feedback.hard_contexts), 1)
        edge = feedback.hard_contexts[0].edge
        self.assertEqual(edge.boundary_index, 0)
        self.assertEqual(edge.outgoing_shot_id, "4:shot-2")
        self.assertEqual(edge.incoming_shot_id, "3:shot-0")

    def test_focus_only_and_missing_evidence_never_become_no_good(self) -> None:
        feedback = build_splice_failure_feedback(
            self.plan,
            (
                _failure(
                    self.plan,
                    0,
                    4,
                    3,
                    tuple(_audit(offset, focus=0.55) for offset in range(-8, 9)),
                ),
                _failure(self.plan, 1, 3, 7, None),
            ),
        )

        self.assertEqual(len(feedback.hard_contexts), 0)
        self.assertEqual(
            tuple(item.status for item in feedback.edges),
            (
                SpliceFeasibilityStatus.RAW_GEOMETRY_PROOF_REQUIRED,
                SpliceFeasibilityStatus.INDETERMINATE,
            ),
        )

    def test_partial_failure_then_new_raw_pass_refines_to_pass(self) -> None:
        failures = (
            _failure(
                self.plan,
                0,
                4,
                3,
                tuple(
                    _audit(offset, silhouette=0.50)
                    for offset in range(-3, 4)
                ),
            ),
            _failure(self.plan, 0, 4, 3, (_audit(7, passed=True),)),
        )
        feedback = build_splice_failure_feedback(
            self.plan,
            failures,
        )
        reversed_feedback = build_splice_failure_feedback(
            self.plan,
            tuple(reversed(failures)),
        )

        self.assertEqual(len(feedback.hard_contexts), 0)
        self.assertIs(
            feedback.edges[0].status,
            SpliceFeasibilityStatus.RAW_PASS,
        )
        self.assertEqual(feedback, reversed_feedback)
        self.assertEqual(feedback.to_dict(), reversed_feedback.to_dict())

    def test_contradictory_complete_hard_and_raw_pass_is_rejected(self) -> None:
        failures = (
            _failure(
                self.plan,
                0,
                4,
                3,
                tuple(
                    _audit(offset, silhouette=0.50)
                    for offset in range(-8, 9)
                ),
            ),
            _failure(self.plan, 0, 4, 3, (_audit(7, passed=True),)),
        )

        with self.assertRaisesRegex(ValueError, "conflicting"):
            build_splice_failure_feedback(self.plan, failures)

    def test_short_search_cannot_create_permanent_constraint(self) -> None:
        feedback = build_splice_failure_feedback(
            self.plan,
            (
                _failure(
                    self.plan,
                    0,
                    4,
                    3,
                    tuple(
                        _audit(offset, silhouette=0.50)
                        for offset in range(-3, 4)
                    ),
                ),
            ),
        )

        self.assertEqual(len(feedback.hard_contexts), 0)
        self.assertIs(
            feedback.edges[0].status,
            SpliceFeasibilityStatus.INDETERMINATE,
        )

    def test_source_direction_or_boundary_drift_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "direction"):
            build_splice_failure_feedback(
                self.plan,
                (_failure(self.plan, 0, 3, 4, None),),
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            build_splice_failure_feedback(
                self.plan,
                (
                    {
                        "boundary_index": 2,
                        "outgoing_source_index": 7,
                        "incoming_source_index": 1,
                        "reason": "outside plan",
                        "outgoing_shot_id": "7:shot-1",
                        "incoming_shot_id": "1:shot-0",
                        "timeline_analysis_basis": self.plan.analysis_basis,
                        "timeline_path_identity": (
                            splice_feedback_plan_identity(self.plan)
                        ),
                    },
                ),
            )

    def test_serialization_is_deterministic(self) -> None:
        failures = (
            _failure(
                self.plan,
                0,
                4,
                3,
                tuple(
                    _audit(offset, silhouette=0.50)
                    for offset in range(-8, 9)
                ),
            ),
            _failure(self.plan, 1, 3, 7, None),
        )
        forward = build_splice_failure_feedback(self.plan, failures)
        reverse = build_splice_failure_feedback(
            self.plan,
            tuple(reversed(failures)),
        )

        self.assertEqual(forward, reverse)
        self.assertEqual(forward.to_dict(), reverse.to_dict())
        self.assertFalse(forward.to_dict()["thresholds_lowered"])

    def test_stale_same_source_direction_different_shot_or_plan_is_rejected(
        self,
    ) -> None:
        stale = _failure(
            self.plan,
            0,
            4,
            3,
            tuple(
                _audit(offset, silhouette=0.50)
                for offset in range(-8, 9)
            ),
        )
        different_shot_plan = SimpleNamespace(
            analysis_basis=self.plan.analysis_basis,
            source_clock_basis=self.plan.source_clock_basis,
            choices=(
                _choice(0, 4, "4:shot-99"),
                self.plan.choices[1],
                self.plan.choices[2],
            ),
        )
        with self.assertRaisesRegex(ValueError, "shot edge"):
            build_splice_failure_feedback(different_shot_plan, (stale,))

        changed_basis_plan = SimpleNamespace(
            analysis_basis="synthetic-plan-v2",
            source_clock_basis=self.plan.source_clock_basis,
            choices=self.plan.choices,
        )
        with self.assertRaisesRegex(ValueError, "analysis basis"):
            build_splice_failure_feedback(changed_basis_plan, (stale,))

    def test_same_shot_path_with_changed_clock_span_is_rejected(self) -> None:
        stale = _failure(
            self.plan,
            0,
            4,
            3,
            tuple(
                _audit(offset, silhouette=0.50)
                for offset in range(-8, 9)
            ),
        )
        changed_first = SimpleNamespace(
            **{
                **vars(self.plan.choices[0]),
                "canonical_master_end": 0.875,
                "source_local_end": 1.8125,
            }
        )
        changed_time_plan = SimpleNamespace(
            analysis_basis=self.plan.analysis_basis,
            source_clock_basis=self.plan.source_clock_basis,
            choices=(
                changed_first,
                self.plan.choices[1],
                self.plan.choices[2],
            ),
        )

        self.assertNotEqual(
            splice_feedback_plan_identity(self.plan),
            splice_feedback_plan_identity(changed_time_plan),
        )
        with self.assertRaisesRegex(ValueError, "path identity"):
            build_splice_failure_feedback(changed_time_plan, (stale,))

    def test_positive_proof_ignores_far_end_but_not_displayed_cut(self) -> None:
        base_identity = splice_feedback_result_context_identity(self.plan, 0)
        changed_far_end = SimpleNamespace(
            **{
                **vars(self.plan.choices[1]),
                "canonical_master_end": 2.25,
                "source_local_end": 3.0,
            }
        )
        far_end_plan = SimpleNamespace(
            analysis_basis="different-stage-label",
            source_clock_basis=self.plan.source_clock_basis,
            choices=(
                self.plan.choices[0],
                changed_far_end,
                self.plan.choices[2],
            ),
        )
        moved_outgoing = SimpleNamespace(
            **{
                **vars(self.plan.choices[0]),
                "canonical_master_end": 1.125,
                "source_local_end": 2.125,
            }
        )
        moved_incoming = SimpleNamespace(
            **{
                **vars(self.plan.choices[1]),
                "canonical_master_start": 1.125,
                "source_local_start": 1.875,
            }
        )
        moved_cut_plan = SimpleNamespace(
            analysis_basis=self.plan.analysis_basis,
            source_clock_basis=self.plan.source_clock_basis,
            choices=(
                moved_outgoing,
                moved_incoming,
                self.plan.choices[2],
            ),
        )

        self.assertEqual(
            splice_feedback_result_context_identity(far_end_plan, 0),
            base_identity,
        )
        self.assertNotEqual(
            splice_feedback_result_context_identity(moved_cut_plan, 0),
            base_identity,
        )

    def test_failure_after_adjacent_repair_cannot_poison_raw_edge(self) -> None:
        changed_middle = SimpleNamespace(
            **{
                **vars(self.plan.choices[1]),
                "canonical_master_start": 1.125,
                "source_local_start": 1.875,
            }
        )
        observed_plan = SimpleNamespace(
            analysis_basis=self.plan.analysis_basis,
            source_clock_basis=self.plan.source_clock_basis,
            choices=(
                self.plan.choices[0],
                changed_middle,
                self.plan.choices[2],
            ),
        )
        feedback = build_splice_failure_feedback(
            self.plan,
            (
                _failure(
                    self.plan,
                    0,
                    4,
                    3,
                    tuple(
                        _audit(offset, silhouette=0.50)
                        for offset in range(-8, 9)
                    ),
                    observed_plan=observed_plan,
                ),
            ),
        )

        self.assertEqual(len(feedback.hard_contexts), 0)
        self.assertIs(
            feedback.edges[0].status,
            SpliceFeasibilityStatus.INDETERMINATE,
        )

    def test_same_raw_edge_keeps_distinct_observed_context_feedback(self) -> None:
        changed_middle = SimpleNamespace(
            **{
                **vars(self.plan.choices[1]),
                "canonical_master_start": 1.125,
                "source_local_start": 1.875,
            }
        )
        observed_plan = SimpleNamespace(
            analysis_basis=self.plan.analysis_basis,
            source_clock_basis=self.plan.source_clock_basis,
            choices=(
                self.plan.choices[0],
                changed_middle,
                self.plan.choices[2],
            ),
        )
        audit = tuple(
            _audit(offset, silhouette=0.50)
            for offset in range(-8, 9)
        )
        feedback = build_splice_failure_feedback(
            self.plan,
            (
                _failure(self.plan, 0, 4, 3, audit),
                _failure(
                    self.plan,
                    0,
                    4,
                    3,
                    audit,
                    observed_plan=observed_plan,
                ),
            ),
        )

        self.assertEqual(len(feedback.edges), 2)
        self.assertEqual(
            {item.status for item in feedback.edges},
            {
                SpliceFeasibilityStatus.HARD_INCOMPATIBLE,
                SpliceFeasibilityStatus.INDETERMINATE,
            },
        )
        self.assertEqual(
            len({item.edge_context_identity for item in feedback.edges}),
            2,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
