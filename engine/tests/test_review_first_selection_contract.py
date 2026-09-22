from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import shutil
import subprocess

from crosscut.post_render_qa import (
    PostRenderQAConfig,
    _IssueCollector,
    _validate_selection,
    evaluate_post_render_qa,
)
from crosscut.review_first_selection_contract import (
    REVIEW_FIRST_CANDIDATE_POOL,
    review_first_release_binding_valid,
    review_first_release_policy_payload,
    build_review_first_selection_payload,
)
from engine.tests.test_post_render_qa import _fixture_payloads, _persist_fixture, _perceptual_pass_payload


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
    # Exercise the production producer, not just a hand-copied approximation.
    selection.update(build_review_first_selection_payload(
        selection, complete_candidate=complete,
        selected_path_authority="deterministic_full_song_visual_selector",
    ))
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
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_actual_singleton_producer_to_full_qa_with_real_decodable_media(self):
        # Continuity/appearance is fixture evidence here, NOT a real quality
        # claim. JSON loading, release validation, ffprobe and decoding are real.
        with tempfile.TemporaryDirectory() as directory:
            artifacts, analysis, plan, selection = _persist_fixture(Path(directory),
                mutate=lambda analysis, plan, selection: _install_review_first_contract(analysis, selection))
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
                f"color=blue:s={plan['width']}x{plan['height']}:r={plan['fps']}:d={plan['render_duration']}",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", str(plan["render_duration"]),
                "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(artifacts.video_path)],
                check=True, timeout=30)
            report = evaluate_post_render_qa(artifacts,
                perceptual_evaluator=lambda *args, **kwargs: _perceptual_pass_payload())
            self.assertTrue(report.passed, [(issue.code, issue.message) for issue in report.issues])

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

    def test_boolean_count_is_not_an_integer_candidate_count(self):
        with tempfile.TemporaryDirectory() as directory:
            _, analysis, _, selection = _fixture_payloads(Path(directory))
            render = _install_review_first_contract(analysis, selection)
            selection["reviewed_candidate_count"] = True
            self.assertFalse(review_first_release_binding_valid(selection, render))
            with self.assertRaises(ValueError):
                build_review_first_selection_payload(selection, complete_candidate=selection["complete_candidates"][0],
                    selected_path_authority="deterministic_full_song_visual_selector")


if __name__ == "__main__":
    unittest.main()
