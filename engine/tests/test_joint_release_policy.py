from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crosscut.joint_release_policy import joint_release_policy_payload
from crosscut.pipeline import (
    _joint_selection_proof_materialization_call_count,
    _materialize_joint_feedback_batch_with_fresh_provider,
)
from crosscut.post_render_qa import evaluate_post_render_qa, PostRenderQAVerdict
from engine.tests.test_joint_visual_candidate_adapter import (
    _candidate_set, _limited_profile_pools, StubMaterializingProvider,
    select_joint_visual_candidates, materialize_joint_visual_candidates,
)
from engine.tests.test_post_render_qa import (
    _persist_fixture, _FakeRunner, _perceptual_pass_payload,
)


def _single_delivery(analysis, _plan, selection):
    policy = joint_release_policy_payload()
    selection["release_policy"] = deepcopy(policy)
    selection["candidate_pool"] = "joint_decoded_visual_selector"
    analysis["render"]["release_policy"] = deepcopy(policy)
    analysis["render"]["candidate_origin"] = "joint_decoded_visual_selector"
    for field in (
        "requested_review_count", "reviewed_candidate_count",
        "minimum_required_candidate_count", "successful_review_count",
        "passing_candidate_count", "complete_candidate_count",
    ):
        selection[field] = 1
    selection["candidates"] = selection["candidates"][:1]
    selection["complete_candidates"] = selection["complete_candidates"][:1]


class JointReleasePolicyTests(unittest.TestCase):
    def evaluate(self, extra=None):
        def mutate(analysis, plan, selection):
            _single_delivery(analysis, plan, selection)
            if extra:
                extra(analysis, plan, selection)
        with tempfile.TemporaryDirectory() as directory:
            artifacts, analysis, plan, selection = _persist_fixture(Path(directory), mutate=mutate)
            with patch("crosscut.post_render_qa.evaluate_post_render_perceptual_quality",
                       return_value=_perceptual_pass_payload()):
                result = evaluate_post_render_qa(artifacts, runner=_FakeRunner())
            return result, analysis, plan, selection

    def test_one_fully_reviewed_video_passes_without_changing_eight_sources(self):
        report, analysis, plan, _ = self.evaluate()
        self.assertEqual(report.verdict, PostRenderQAVerdict.PASS,
                         [(issue.code, issue.message) for issue in report.issues])
        self.assertEqual(len(plan["sources"]), 8)
        self.assertEqual(report.policy.minimum_distinct_sources_when_required, 8)
        self.assertEqual(report.policy.required_complete_candidate_count, 1)

    def test_zero_reviews_cannot_be_released(self):
        def remove(_analysis, _plan, selection):
            selection["candidates"] = []
            selection["complete_candidates"] = []
        report, *_ = self.evaluate(remove)
        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)

    def test_failed_selected_review_cannot_be_released(self):
        def reject(_analysis, _plan, selection):
            selection["final_selected_review"]["passed"] = False
        report, *_ = self.evaluate(reject)
        self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
        self.assertIn("selection_final_selected_review_passed", {i.code for i in report.issues})

    def test_policy_must_match_both_artifacts_and_joint_origin(self):
        for mutation in (
            lambda a, p, s: a["render"].pop("release_policy"),
            lambda a, p, s: s["release_policy"].update(minimum_passing_candidate_count=0),
            lambda a, p, s: s.update(candidate_pool="ordinary_visual_baseline"),
        ):
            with self.subTest(mutation=mutation):
                report, *_ = self.evaluate(mutation)
                self.assertEqual(report.verdict, PostRenderQAVerdict.FAIL)
                self.assertIn("candidate_release_policy_mismatch", {i.code for i in report.issues})

    def test_single_candidate_crosses_real_batch_wrapper_with_proof_accounting(self):
        candidate_set = _candidate_set()
        provider = StubMaterializingProvider(fps=10.0)
        with patch("crosscut.joint_visual_candidate_adapter.select_joint_timeline_candidates",
                   side_effect=_limited_profile_pools(candidate_set, unique_path_count=1, alternate_paths=())):
            selected = select_joint_visual_candidates(
                candidate_set, provider=provider, fps=10.0, maximum_candidates=8,
                minimum_unique_paths=1, minimum_required_candidates=1,
            )
        materialized = materialize_joint_visual_candidates(selected, provider=provider, fps=10.0)
        self.assertEqual(_joint_selection_proof_materialization_call_count(selected, available_budget=1), 1)
        bad = replace(selected, selection_retry_audit={"proof_materialization_call_count": 0})
        with self.assertRaisesRegex(RuntimeError, "invalid proof usage"):
            _joint_selection_proof_materialization_call_count(bad, available_budget=1)
        class OwnedProvider:
            closed = False
            def to_dict(self): return {"fixture": True}
            def close(self): self.closed = True
        owned = OwnedProvider()
        audits = []
        returned, items = _materialize_joint_feedback_batch_with_fresh_provider(
            batch_index=0, provider_factory=lambda: owned,
            select_candidates=lambda *args: selected,
            materialize_candidates=lambda *args: materialized,
            forbidden_boundary_selections=(), forbidden_complete_paths=(),
            maximum_proof_materialization_calls=1, provider_audit_sink=audits.append,
        )
        self.assertIs(returned, selected)
        self.assertEqual(len(items), 1)
        self.assertTrue(owned.closed)
        self.assertEqual(len(audits), 1)


if __name__ == "__main__":
    unittest.main()
