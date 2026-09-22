"""Production producer-to-consumer regressions from the independent Pro audit."""

from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

from crosscut.post_render_qa import PostRenderQAConfig, _IssueCollector, _validate_selection, _proxy_quality_and_geometry_metrics
from crosscut.post_render_perceptual_quality import _Span, _reference_view_for_span
from crosscut.render import EditorialTransformKeyframe, EditorialTransformTrack, _editorial_transform_filters
from crosscut.rendered_proxy_review import ProxyReviewConfig, _boundary_domain_summary, _boundary_evidence, _distribution
from engine.tests.test_post_render_qa import _fixture_payloads, _promote_review_to_domain_aware_v3
from engine.tests.test_review_first_selection_contract import _install_review_first_contract


class ProReviewFollowupTests(unittest.TestCase):
    @staticmethod
    def _current_review_fixture(directory, *, missing_count=0):
        _, analysis, plan, selection = _fixture_payloads(Path(directory))
        review = selection["final_selected_review"]
        _promote_review_to_domain_aware_v3(review, plan)
        for boundary in review["boundaries"][:missing_count]:
            boundary.update(weighted_score=None, passed=False, evidence_coverage=_boundary_evidence({}))
        review["boundary_domains"] = {
            domain: _boundary_domain_summary(review["boundaries"], domain=domain, config=ProxyReviewConfig())
            for domain in ("introduced_cross_source", "tier_d_continuity", "native_source_cut")
        }
        scores = [b["weighted_score"] for b in review["boundaries"] if b["weighted_score"] is not None]
        distribution = _distribution(scores)
        review["boundary_score"].update(distribution=distribution, p10=distribution["p10"],
                                        issue_count=missing_count, issue_fraction=missing_count / len(review["boundaries"]))
        review["boundary_issue_indices"] = list(range(missing_count))
        review["introduced_cross_source_boundary_issue_indices"] = list(range(missing_count))
        review["passed"] = not missing_count
        selection["complete_candidates"][0]["group_transitions"] = {"transitions": deepcopy(plan["transitions"])}
        selection["complete_candidates"][0]["visual_timeline"] = {"choices": deepcopy(plan["segments"])}
        _install_review_first_contract(analysis, selection)
        return analysis, plan, selection

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required")
    def test_prescribed_roll_matches_actual_ffmpeg_pixels_for_both_signs(self):
        width, height = 960, 540
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        cv2.circle(frame, (760, 170), 12, (0, 0, 255), -1)
        def centroid(image):
            ys, xs = np.nonzero(
                (image[:, :, 2] > 180) & (image[:, :, 1] < 70) & (image[:, :, 0] < 70)
            )
            self.assertGreater(len(xs), 100)
            return np.array([xs.mean(), ys.mean()])
        for roll in (-1.5, 0.0, 1.5):
            with self.subTest(roll=roll):
                track = EditorialTransformTrack((EditorialTransformKeyframe(0., 1.12, .52, .49, roll),))
                filters = _editorial_transform_filters(track, width=width, height=height, fps=30.)
                result = subprocess.run(
                    ["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                     "-s", f"{width}x{height}", "-r", "30", "-i", "pipe:0", "-vf", ",".join(filters),
                     "-frames:v", "1", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"],
                    input=frame.tobytes(), capture_output=True, timeout=15, check=True,
                )
                rendered = np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3)
                span = _Span(
                    segment_index=0, source_index=0, start_frame=0, end_frame=1, run_index=0,
                    source_start=0., source_rate=1., duration=1., incoming_transition_frames=0,
                    crop=None, pan_crop=None, editorial_transform=json.loads(json.dumps(asdict(track))),
                )
                observed = _reference_view_for_span(frame, span, relative_time=0., size=(width, height))
                self.assertLess(np.linalg.norm(centroid(rendered) - centroid(observed)), 2.0)

    def test_current_domain_summary_flows_into_final_selection_validator(self):
        with tempfile.TemporaryDirectory() as directory:
            analysis, plan, selection = self._current_review_fixture(directory)
            review = selection["final_selected_review"]
            empty_domain = review["boundary_domains"]["native_source_cut"]
            self.assertIsNone(empty_domain["score"]["p10"])
            self.assertEqual(empty_domain["score"]["evidence_status"], "NOT_APPLICABLE")
            collector = _IssueCollector()
            metrics, _ = _validate_selection(
                analysis, plan, selection,
                PostRenderQAConfig(required_complete_candidate_count=1), collector,
            )
            self.assertEqual(collector.issues, [])
            self.assertTrue(all(metrics["checks"].values()), metrics["checks"])

    def test_unknown_scores_are_quality_misses_not_malformed_contracts(self):
        for missing_count in (1, 8):
            with self.subTest(missing_count=missing_count), tempfile.TemporaryDirectory() as directory:
                analysis, plan, selection = self._current_review_fixture(directory, missing_count=missing_count)
                collector = _IssueCollector()
                metrics, review = _validate_selection(
                    analysis, plan, selection, PostRenderQAConfig(required_complete_candidate_count=1),
                    collector, allow_failed_review=True,
                )
                self.assertEqual(collector.issues, [])
                self.assertTrue(all(metrics["checks"].values()))
                _proxy_quality_and_geometry_metrics(analysis, review, PostRenderQAConfig(), collector)
                codes = {issue.code for issue in collector.issues}
                self.assertIn("decoded_proxy_review_failed", codes)
                self.assertNotIn("proxy_boundary_score_distribution_mismatch", codes)
                self.assertIs(review["passed"], False)

    def test_missing_or_forged_observations_remain_contract_errors(self):
        for mutation in ("empty_perfect", "missing_null", "missing_status", "unknown_passed", "false_count"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                analysis, plan, selection = self._current_review_fixture(directory, missing_count=1)
                review = selection["final_selected_review"]
                empty = review["boundary_domains"]["native_source_cut"]["score"]
                if mutation == "empty_perfect": empty["p10"] = 100.
                elif mutation == "missing_null": empty.pop("p10")
                elif mutation == "missing_status": empty.pop("evidence_status")
                elif mutation == "unknown_passed": review["boundaries"][0]["passed"] = True
                else: review["boundary_domains"]["introduced_cross_source"]["score"]["distribution"]["count"] = 8
                _install_review_first_contract(analysis, selection)
                collector = _IssueCollector()
                metrics, _ = _validate_selection(
                    analysis, plan, selection, PostRenderQAConfig(required_complete_candidate_count=1),
                    collector, allow_failed_review=True,
                )
                self.assertFalse(metrics["checks"]["final_selected_decoded_review_contract_valid"])


if __name__ == "__main__":
    unittest.main()
