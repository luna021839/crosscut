from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from crosscut.post_render_qa import (
    PostRenderQAConfig,
    _IssueCollector,
    _validate_selection,
)
from crosscut.review_first_selection_contract import (
    REVIEW_FIRST_CANDIDATE_POOL,
    review_first_release_binding_valid,
    review_first_release_policy_payload,
)
from engine.tests.test_post_render_qa import _fixture_payloads


def _install_review_first_contract(
    analysis: dict,
    selection: dict,
) -> dict:
    policy = review_first_release_policy_payload()
    review = deepcopy(selection["final_selected_review"])
    candidate = deepcopy(selection["candidates"][0])
    candidate["selected"] = True
    candidate["review"] = deepcopy(review)
    candidate["passed"] = review["passed"]
    complete = deepcopy(selection["complete_candidates"][0])
    selection.update(
        {
            "analysis_basis": "bounded_review_first_single_full_song_v1",
            "release_policy": policy,
            "candidate_pool": REVIEW_FIRST_CANDIDATE_POOL,
            "selected_path_authority": "deterministic_full_song_visual_selector",
            "requested_review_count": 1,
            "reviewed_candidate_count": 1,
            "minimum_required_candidate_count": 1,
            "minimum_candidate_count_satisfied": True,
            "successful_review_count": 1,
            "passing_candidate_count": int(review["passed"] is True),
            "complete_candidate_count": 1,
            "exact_successful_decoded_review_count_satisfied": True,
            "exact_complete_candidate_count_satisfied": True,
            "selected_candidate_index": 0,
            "final_selected_review": review,
            "final_selected_version": "original",
            "proxy_triggered_partial_reedit": {
                "attempted": False,
                "improved": False,
                "selected_version": "original",
                "changed_boundary_indices": [],
            },
            "passed": review["passed"],
            "candidates": [candidate],
            "complete_candidates": [complete],
        }
    )
    render = analysis["render"]
    render.update(
        {
            "release_policy": policy,
            "candidate_pool": REVIEW_FIRST_CANDIDATE_POOL,
            "candidate_origin": "deterministic_full_song_visual_selector",
        }
    )
    analysis["rendered_candidate_selection"] = selection
    render["rendered_candidate_selection"] = selection
    analysis["rendered_proxy_review"] = review
    render["rendered_proxy_review"] = review
    return render


class ReviewFirstSelectionContractTests(unittest.TestCase):
    def test_real_selection_validator_accepts_genuine_singleton_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _artifacts, analysis, plan, selection = _fixture_payloads(
                Path(directory)
            )
            render = _install_review_first_contract(analysis, selection)
            collector = _IssueCollector()

            self.assertTrue(review_first_release_binding_valid(selection, render))
            metrics, final_review = _validate_selection(
                analysis,
                plan,
                selection,
                PostRenderQAConfig(required_complete_candidate_count=1),
                collector,
            )

            self.assertEqual(collector.issues, [])
            self.assertTrue(all(metrics["checks"].values()))
            self.assertEqual(final_review, selection["final_selected_review"])

    def test_failed_but_well_formed_review_is_not_a_selection_contract_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _artifacts, analysis, plan, selection = _fixture_payloads(
                Path(directory)
            )
            render = _install_review_first_contract(analysis, selection)
            failed_review = deepcopy(selection["final_selected_review"])
            failed_review["passed"] = False
            selection["final_selected_review"] = failed_review
            selection["candidates"][0]["review"] = deepcopy(failed_review)
            selection["candidates"][0]["passed"] = False
            selection["passing_candidate_count"] = 0
            selection["passed"] = False
            analysis["rendered_proxy_review"] = failed_review
            render["rendered_proxy_review"] = failed_review
            collector = _IssueCollector()

            self.assertTrue(review_first_release_binding_valid(selection, render))
            metrics, _final_review = _validate_selection(
                analysis,
                plan,
                selection,
                PostRenderQAConfig(required_complete_candidate_count=1),
                collector,
                allow_failed_review=True,
            )

            self.assertEqual(collector.issues, [])
            self.assertTrue(all(metrics["checks"].values()))

    def test_policy_token_alone_cannot_lower_candidate_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _artifacts, analysis, _plan, selection = _fixture_payloads(
                Path(directory)
            )
            render = _install_review_first_contract(analysis, selection)
            selection["complete_candidate_count"] = 8

            self.assertFalse(review_first_release_binding_valid(selection, render))


if __name__ == "__main__":
    unittest.main()
