from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import crosscut.pipeline as pipeline
from crosscut.candidate_stream_verdict import CandidateStreamReducer
from crosscut.group_output_look import DecodedLookObjective
from crosscut.joint_decoded_budget import (
    JointDecodedBudgetLedger,
    JointDecodedLookRefinementEligibility,
    JointDecodedRenderKind,
)
from crosscut.joint_look_breadth import (
    JointLookBreadthScheduler,
    PendingJointLookRefinement,
)


@dataclass(frozen=True)
class _Plan:
    candidate_index: int
    version: int = 0


@dataclass(frozen=True)
class _Bundle:
    render_plan: _Plan
    candidate_lineage: tuple = ()
    output_geometry_deferral: object = None


class _ProductionHarness:
    """Execute the production nested callbacks, replacing only media/gate IO.

    These are compiled directly from pipeline.py, not scheduler replicas. The
    real ledger, refinement control flow, terminal reducer, no-good authoring,
    retained continuation drain and feedback disposition execute unchanged.
    Separate gate tests exercise the measured media acceptance functions.
    """

    def __init__(self, *, passing_versions=None, invalid_on=None):
        self.passing_versions = passing_versions or {}
        self.invalid_on = invalid_on
        self.calls = []
        self.progress = []
        self.scheduler = JointLookBreadthScheduler()
        self.ledger = JointDecodedBudgetLedger()
        self.started = set()
        self.reducer = CandidateStreamReducer(required_passing_count=8)
        self.classifications = {}
        self.no_goods = set()
        self.derived = []
        self.audits = {}
        self.failures = []
        self.exhaustions = []
        self.env = dict(vars(pipeline))
        self.env.update({
            "joint_visual_selection_active": True,
            "joint_look_scheduler": self.scheduler,
            "joint_decoded_budget": self.ledger,
            "joint_decoded_look_refinement_audits": self.audits,
            "attempted_candidate_indices": self.started,
            "candidate_stream_reducer": self.reducer,
            "joint_candidate_started_at": {},
            "joint_candidate_runtime_audits": [],
            "candidate_derivation_failures": [],
            "candidate_post_repair_review_rejections": [],
            "candidate_decoded_review_failures": self.failures,
            "joint_decoded_feedback_classifications": self.classifications,
            "joint_decoded_forbidden_complete_paths": self.no_goods,
            "derived": self.derived,
            "streamed_group_candidate_reviews": [],
            "reporter": Mock(),
            "_underlying_joint_materialized_candidate": lambda x: x,
            "_joint_complete_path_signature": lambda x: x.candidate_index,
            "decoded_look_objective": self.objective,
            "_joint_decoded_render_variant_signature": lambda p: f"v{p['version']}",
            "_bound_decoded_proxy_identity": lambda p, **kw: (f"v{p['version']}",),
            "_joint_group_look_refinement_eligibility": self.eligibility,
            "refine_group_render_plan_from_decoded_proxy": self.propose,
            "_joint_require_refined_render_variant_lineage": lambda *args: None,
            "_joint_group_look_sensitive_visible_audit": lambda *args: {},
            "_joint_group_exact_decoded_path_invariant_reasons": lambda *a, **kw: (),
            "_joint_group_look_refinement_acceptance": self.accept,
            "decode_and_review_complete_group_bundle": self.decode,
            "record_joint_decoded_feedback": self.record_feedback,
            "record_joint_decoded_budget_exhaustion": self.exhaustions.append,
            "persist_candidate_stream_progress": self.persist,
            "_derive_and_gate_candidate_verdict": self.derive,
        })
        wanted = {
            "refine_joint_candidate_look_from_decoded_output",
            "finish_decoded_group_candidate",
            "derive_and_gate_candidate",
        }
        tree = ast.parse(Path(pipeline.__file__).read_text())
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                 and n.name in wanted]
        assert len(nodes) == len(wanted)
        module = ast.Module(body=nodes, type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), pipeline.__file__, "exec"), self.env)
        # Execute the real early-resume branch too. Its only nonlocal variable
        # belongs to base derivation and is not accessed by a continuation.
        resume_node = deepcopy(next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
            and n.name == "_derive_and_gate_candidate_verdict"
        ))
        resume_node.name = "production_derive_and_gate_candidate_verdict"
        resume_node.body = [n for n in resume_node.body if not isinstance(n, ast.Nonlocal)]
        module = ast.Module(body=[resume_node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), pipeline.__file__, "exec"), self.env)

    @staticmethod
    def objective(payload):
        return DecodedLookObjective(payload["passed"], payload["excess"], 0.0)

    @staticmethod
    def eligibility(payload, **kwargs):
        value = JointDecodedLookRefinementEligibility(
            True, True, kwargs["look_admission_passed"], kwargs["decoded_review_passed"],
        )
        return value, value.to_dict()

    def propose(self, plan, *, iteration, previous_refinement, **kwargs):
        if iteration == 2:
            assert previous_refinement is not None
            assert previous_refinement.render_plan == plan
        return SimpleNamespace(
            applied=True, render_plan=replace(plan, version=iteration),
            to_dict=lambda: {"iteration": iteration},
        )

    def accept(self, objective, previous, proposed, **kwargs):
        if self.invalid_on == (proposed["candidate_index"], proposed["version"]):
            raise ValueError("synthetic decoded evidence binding broke")
        return self.objective(proposed), (), True

    def decode(self, bundle, *, materialized, render_kind, look_refinement_eligibility=None):
        index, version = materialized.candidate_index, bundle.render_plan.version
        if render_kind is JointDecodedRenderKind.BASE_CANDIDATE:
            self.ledger.claim_base_render(candidate_index=index, path_identity=index)
        else:
            self.ledger.claim_look_refinement(
                candidate_index=index, path_identity=index,
                eligibility=look_refinement_eligibility,
            )
        self.calls.append((index, version))
        passed = version >= self.passing_versions.get(index, 99)
        payload = {
            "candidate_index": index, "version": version,
            "passed": passed, "excess": 1.0 - 0.25 * version,
        }
        return SimpleNamespace(review_payload=payload, passed=passed, error=None)

    def record_feedback(self, materialized, bundle, payload):
        self.classifications[materialized.candidate_index] = {
            "feedback_binding_valid": True,
            "feedback_authoring_allowed": not payload["passed"],
            "complete_path_no_good_deferred_for_render_variant": not payload["passed"],
        }
        return True

    def persist(self, **kwargs):
        self.scheduler.assert_partition(
            self.started, (v.candidate_index for v in self.reducer.verdicts),
        )
        self.progress.append({
            "scheduler": self.scheduler.to_dict(),
            "terminal": [v.candidate_index for v in self.reducer.verdicts],
            "no_goods": sorted(self.no_goods),
        })

    def derive(self, materialized, **kwargs):
        index = materialized.candidate_index
        pending = self.scheduler.get(index)
        if pending is None:
            self.started.add(index)
            bundle = _Bundle(_Plan(index))
            review = self.decode(
                bundle, materialized=materialized,
                render_kind=JointDecodedRenderKind.BASE_CANDIDATE,
            )
        else:
            return self.env["production_derive_and_gate_candidate_verdict"](materialized)
        return self.env["finish_decoded_group_candidate"](
            materialized, bundle, review, False,
        )

    def admit(self, count):
        for index in range(count):
            self.env["derive_and_gate_candidate"](
                SimpleNamespace(candidate_index=index, review_rank=index),
            )

    def drain(self, stop=lambda: False):
        pipeline._drain_joint_look_continuations(
            self.scheduler, resume=self.env["derive_and_gate_candidate"], should_stop=stop,
        )


class JointLookBreadthProductionTests(unittest.TestCase):
    def test_thirteen_paths_receive_first_look_before_any_second(self):
        run = _ProductionHarness()
        run.admit(13)
        self.assertEqual(run.scheduler.pending_indices, tuple(range(13)))
        self.assertEqual(run.reducer.verdicts, ())
        self.assertFalse(run.no_goods)
        self.assertEqual([c for c in run.calls if c[1]], [(i, 1) for i in range(13)])
        run.drain()
        self.assertEqual([c for c in run.calls if c[1]],
                         [(i, 1) for i in range(13)] + [(i, 2) for i in range(11)])
        self.assertEqual(run.ledger.base_render_count, 13)
        self.assertEqual(run.ledger.look_refinement_render_count, 24)
        self.assertEqual(run.ledger.total_render_count, 37)
        self.assertEqual(run.no_goods, set(range(11)))
        self.assertEqual(len(run.exhaustions), 2)
        self.assertEqual(len(run.reducer.verdicts), 13)
        self.assertFalse(run.scheduler.pending_indices)
        self.assertFalse(run.audits[12]["complete_path_no_good_allowed"])

    def test_eventual_second_look_recovery_preserves_bundle_and_lineage(self):
        run = _ProductionHarness(passing_versions={i: 2 for i in range(8)})
        run.admit(8)
        self.assertFalse(run.derived)
        first_bundle = run.scheduler.get(0).bundle
        self.assertEqual(first_bundle.render_plan.version, 1)
        run.drain()
        self.assertEqual(len(run.derived), 8)
        self.assertTrue(run.reducer.required_passing_cohort_complete)
        self.assertFalse(run.no_goods)
        self.assertEqual(run.ledger.base_render_count, 8)
        self.assertEqual(run.ledger.look_refinement_render_count, 16)
        for index, audit in run.audits.items():
            self.assertEqual([a["iteration"] for a in audit["iterations"]], [1, 2])
            self.assertEqual(audit["initial_render_variant_signature"], "v0")
            self.assertTrue(audit["final_passed"])
            self.assertFalse(audit["pending_second_look"])
        self.assertEqual(first_bundle.render_plan.version, 1)

    def test_all_twenty_four_first_attempts_fit_and_pending_is_not_no_good(self):
        run = _ProductionHarness()
        run.admit(24)
        self.assertEqual(run.ledger.total_render_count, 48)
        self.assertEqual([c for c in run.calls if c[1]], [(i, 1) for i in range(24)])
        run.drain()
        self.assertEqual(run.ledger.total_render_count, 48)
        self.assertEqual(len(run.reducer.verdicts), 24)
        self.assertFalse(run.no_goods)
        self.assertEqual(len(run.exhaustions), 24)

    def test_diversity_stop_callback_is_obeyed_instead_of_raw_pass_count(self):
        run = _ProductionHarness(passing_versions={i: 2 for i in range(10)})
        run.admit(10)
        # First eight share one synthetic style; candidate 8 supplies diversity.
        def diverse_cohort():
            ids = {b.render_plan.candidate_index for b in run.derived}
            return len(ids) >= 8 and 8 in ids
        run.drain(diverse_cohort)
        self.assertEqual(len(run.derived), 9)
        self.assertEqual(run.scheduler.pending_indices, (9,))
        self.assertNotIn((9, 2), run.calls)
        self.assertNotIn(9, run.no_goods)
        run.persist()

    def test_invalid_second_evidence_rolls_back_and_preserves_exact_error(self):
        run = _ProductionHarness(invalid_on=(0, 2))
        run.admit(1)
        pending_bundle = run.scheduler.get(0).bundle
        run.drain()
        audit = run.audits[0]
        self.assertIn("synthetic decoded evidence binding broke", audit["iterations"][-1]["error"])
        self.assertTrue(audit["iterations"][-1]["rollback"])
        self.assertFalse(audit["evidence_complete"])
        self.assertFalse(run.no_goods)
        self.assertEqual(run.failures[-1]["review"]["version"], pending_bundle.render_plan.version)

    def test_resume_before_admission_close_is_rejected(self):
        run = _ProductionHarness()
        run.admit(1)
        with self.assertRaisesRegex(RuntimeError, "remaining first looks"):
            run.env["derive_and_gate_candidate"](SimpleNamespace(candidate_index=0))
        self.assertEqual(run.scheduler.pending_indices, (0,))
        self.assertFalse(run.no_goods)

    def test_continuation_capacity_and_partition_are_bounded(self):
        scheduler = JointLookBreadthScheduler(maximum_pending_count=1)
        state = PendingJointLookRefinement(SimpleNamespace(candidate_index=0), None, None, None)
        scheduler.defer(0, state)
        scheduler.assert_partition({0, 1}, {1})
        with self.assertRaisesRegex(RuntimeError, "partition"):
            scheduler.assert_partition({0, 1}, {0, 1})
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            scheduler.defer(1, replace(state, materialized=SimpleNamespace(candidate_index=1)))
        scheduler.close_admission()
        with self.assertRaisesRegex(RuntimeError, "admission closes"):
            scheduler.defer(1, replace(state, materialized=SimpleNamespace(candidate_index=1)))

    def test_unexpected_resume_error_is_persisted_once_as_terminal(self):
        run = _ProductionHarness()
        run.admit(1)
        def broken_renderer(*args, **kwargs):
            raise RuntimeError("synthetic renderer original failure")
        run.env["decode_and_review_complete_group_bundle"] = broken_renderer
        with self.assertRaisesRegex(RuntimeError, "renderer original failure"):
            run.drain()
        self.assertFalse(run.scheduler.pending_indices)
        self.assertEqual(len(run.reducer.verdicts), 1)
        self.assertEqual(run.reducer.verdicts[0].status.value, "system_error")
        self.assertEqual(run.progress[-1]["terminal"], [0])
        self.assertFalse(run.no_goods)

    def test_production_drain_is_after_distinct_feedback_batch_loop(self):
        tree = ast.parse(Path(pipeline.__file__).read_text())
        drains = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == "_drain_joint_look_continuations"]
        self.assertEqual(len(drains), 1)
        drain = drains[0]
        materialize_calls = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                             and isinstance(n.func, ast.Name)
                             and n.func.id == "materialize_joint_visual_feedback_batch"]
        self.assertGreater(drain.lineno, max(materialize_calls))
        self.assertTrue(any(isinstance(n, ast.Name) and n.id == "joint_diverse_passing_cohort"
                            for n in ast.walk(drain)))

    def test_terminal_proof_refreshes_pending_and_budget_after_drain(self):
        run = _ProductionHarness()
        run.admit(1)
        before_budget = deepcopy(run.ledger.report_payload())
        historical = {
            "pending_candidate_indices": [0],
            "decoded_render_budget": before_budget,
            "exact_path_no_good_added_for_pending": False,
        }
        recorded = []
        run.env.update({
            "joint_diverse_passing_cohort": lambda: (),
            "joint_shared_render_contract_failure": None,
            "constraint_reoptimization_termination_reason": "distinct_candidate_limit_exhausted",
            "constraint_reoptimization_terminal_proof": {"evidence": historical},
            "set_joint_decoded_feedback_terminal_proof": lambda cause, **kw: recorded.append(kw["evidence"]),
        })
        tree = ast.parse(Path(pipeline.__file__).read_text())
        block = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and n.body
                     and isinstance(n.body[0], ast.Expr)
                     and isinstance(n.body[0].value, ast.Call)
                     and isinstance(n.body[0].value.func, ast.Name)
                     and n.body[0].value.func.id == "_drain_joint_look_continuations")
        exec(compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])),
                     pipeline.__file__, "exec"), run.env)
        final = recorded[-1]
        self.assertEqual(final["pending_candidate_indices_at_admission_close"], [0])
        self.assertEqual(final["decoded_render_budget_at_admission_close"], before_budget)
        self.assertEqual(final["pending_candidate_indices"], [])
        self.assertEqual(final["pending_candidate_count"], 0)
        self.assertEqual(final["decoded_render_budget"], run.ledger.report_payload())
        self.assertEqual(run.no_goods, {0})
        self.assertEqual(historical["pending_candidate_indices"], [0])
        self.assertEqual(run.env["joint_candidate_runtime_audits"][0]["elapsed_seconds_basis"],
                         "candidate_lifecycle_including_pending_look_wait")


if __name__ == "__main__":
    unittest.main()
