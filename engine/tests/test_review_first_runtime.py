from __future__ import annotations

import os
import signal
import threading
import time
import unittest

from crosscut.review_first_runtime import (
    ReviewFirstDeadlineExceeded,
    ReviewFirstRuntime,
    ReviewFirstStatus,
    classify_review_first_qa,
    enforce_review_first_deadline,
)


class ReviewFirstRuntimeTests(unittest.TestCase):
    def test_editorial_failure_is_exposed_only_as_review_ready(self) -> None:
        disposition = classify_review_first_qa(
            {
                "passed": False,
                "issues": [
                    {"code": "source_switch_density_below_hard_minimum"},
                    {"code": "decoded_proxy_review_failed"},
                ],
            },
            content_binding_passed=True,
            review_first_enabled=True,
            member=False,
        )

        self.assertEqual(disposition.status, ReviewFirstStatus.REVIEW_READY)
        self.assertEqual(disposition.artifact_role, "review_ready")
        self.assertTrue(disposition.video_output_exposed)
        self.assertFalse(disposition.delivery_completion_message_allowed)

    def test_technical_failure_remains_unexposed(self) -> None:
        disposition = classify_review_first_qa(
            {
                "passed": False,
                "issues": [{"code": "full_decode_failed"}],
            },
            content_binding_passed=True,
            review_first_enabled=True,
            member=False,
        )

        self.assertEqual(disposition.status, ReviewFirstStatus.INPUT_UNUSABLE)
        self.assertEqual(disposition.artifact_role, "rejected")
        self.assertFalse(disposition.video_output_exposed)

    def test_unknown_failure_is_technical_and_unexposed(self) -> None:
        disposition = classify_review_first_qa(
            {
                "passed": False,
                "issues": [{"code": "new_unclassified_quality_failure"}],
            },
            content_binding_passed=True,
            review_first_enabled=True,
            member=False,
        )

        self.assertEqual(disposition.status, ReviewFirstStatus.INPUT_UNUSABLE)
        self.assertIn(
            "new_unclassified_quality_failure",
            disposition.technical_issue_codes,
        )
        self.assertFalse(disposition.video_output_exposed)

    def test_omitted_media_integrity_code_is_fail_closed(self) -> None:
        disposition = classify_review_first_qa(
            {
                "passed": False,
                "issues": [
                    {"code": "final_media_exactly_one_audio_stream"},
                ],
            },
            content_binding_passed=True,
            review_first_enabled=True,
            member=False,
        )

        self.assertEqual(disposition.status, ReviewFirstStatus.INPUT_UNUSABLE)
        self.assertIn(
            "final_media_exactly_one_audio_stream",
            disposition.technical_issue_codes,
        )

    def test_passed_report_with_any_issue_is_contradictory(self) -> None:
        disposition = classify_review_first_qa(
            {
                "passed": True,
                "issues": [{"code": "decoded_proxy_review_failed"}],
            },
            content_binding_passed=True,
            review_first_enabled=True,
            member=False,
        )

        self.assertEqual(disposition.status, ReviewFirstStatus.INPUT_UNUSABLE)
        self.assertIn("qa_passed_with_issues", disposition.technical_issue_codes)

    def test_malformed_issue_is_technical_and_unexposed(self) -> None:
        disposition = classify_review_first_qa(
            {"passed": False, "issues": [{"message": "missing code"}]},
            content_binding_passed=True,
            review_first_enabled=True,
            member=False,
        )

        self.assertEqual(disposition.status, ReviewFirstStatus.INPUT_UNUSABLE)
        self.assertIn("qa_issue_0_code_invalid", disposition.technical_issue_codes)

    def test_member_quality_code_needs_valid_nested_evidence(self) -> None:
        malformed = classify_review_first_qa(
            {
                "passed": False,
                "issues": [{"code": "member_perceptual_quality_not_pass"}],
                "metrics": {"perceptual_quality": {}},
            },
            content_binding_passed=True,
            review_first_enabled=True,
            member=True,
        )
        valid = classify_review_first_qa(
            {
                "passed": False,
                "issues": [{"code": "member_perceptual_quality_not_pass"}],
                "metrics": {
                    "perceptual_quality": {
                        "schema_version": 2,
                        "analysis": "full_final_decode_temporal_pumping_exact_source_retention_and_structure_binding_v2",
                        "verdict": "FAIL",
                        "passed": False,
                        "issues": [{"code": "perceptual_detail_retention_p10"}],
                        "metrics": {"video": {"full_frame_coverage": True, "dimensions_match_plan": True,
                            "fps_matches_plan": True, "decoded_frame_count": 30, "expected_frame_count": 30}},
                    }
                },
            },
            content_binding_passed=True,
            review_first_enabled=True,
            member=True,
        )

        self.assertEqual(malformed.status, ReviewFirstStatus.INPUT_UNUSABLE)
        self.assertEqual(valid.status, ReviewFirstStatus.REVIEW_READY)

    def test_binding_failure_cannot_be_review_ready(self) -> None:
        disposition = classify_review_first_qa(
            {
                "passed": False,
                "issues": [{"code": "member_perceptual_quality_not_pass"}],
            },
            content_binding_passed=False,
            review_first_enabled=True,
            member=True,
        )

        self.assertEqual(disposition.status, ReviewFirstStatus.INPUT_UNUSABLE)
        self.assertFalse(disposition.video_output_exposed)

    def test_member_look_provenance_aliases_must_agree_and_have_measurements(self):
        for fields, expected in (
            ({"provenance_valid": True}, True),
            ({"cache_only_provenance_valid": True}, True),
            ({"provenance_valid": True, "cache_only_provenance_valid": True}, True),
            ({"provenance_valid": True, "cache_only_provenance_valid": False}, False),
            ({"provenance_valid": 1}, False),
            ({"provenance_valid": True, "predicted_evidence_valid": False}, False),
            ({}, False),
        ):
            with self.subTest(fields=fields):
                disposition = classify_review_first_qa(
                    {"passed": False,
                     "issues": [{"code": "member_look_calibration_predicted_quality_not_pass"}],
                     "metrics": {"member_look_calibration": {
                         "locations_bound": True, "segment_binding_valid": True,
                         "predicted_evidence_valid": True, "predicted_quality_valid": False,
                         **fields,
                     }}},
                    content_binding_passed=True, review_first_enabled=True, member=True,
                )
                self.assertEqual(disposition.video_output_exposed, expected)

    def test_nested_member_technical_failure_cannot_be_relabelled_editorial(self):
        for nested_code in ("perceptual_final_video_unavailable", "perceptual_render_source_clock_invalid",
                            "perceptual_exact_structure_segment_binding_failed", "new_unknown_failure"):
            with self.subTest(code=nested_code):
                disposition = classify_review_first_qa({"passed": False,
                    "issues": [{"code": "member_perceptual_quality_not_pass"}], "metrics": {
                        "perceptual_quality": {"schema_version": 2,
                            "analysis": "full_final_decode_temporal_pumping_exact_source_retention_and_structure_binding_v2",
                            "passed": False, "verdict": "FAIL", "issues": [{"code": nested_code}],
                            "metrics": {"video": {"full_frame_coverage": True, "dimensions_match_plan": True,
                                "fps_matches_plan": True, "decoded_frame_count": 30, "expected_frame_count": 30}}}}},
                    content_binding_passed=True, review_first_enabled=True, member=True)
                self.assertFalse(disposition.video_output_exposed)

    def test_named_but_empty_member_continuity_report_is_not_quality_evidence(self):
        disposition = classify_review_first_qa({"passed": False,
            "issues": [{"code": "member_rendered_continuity_not_pass"}], "metrics": {
                "rendered_continuity": {"schema_version": 1,
                    "analysis": "automatic_member_exact_rendered_boundary_continuity_v1",
                    "passed": False, "failure_reasons": [], "failure_boundary_indices": {},
                    "metrics": {}, "boundaries": []}}},
            content_binding_passed=True, review_first_enabled=True, member=True)
        self.assertFalse(disposition.video_output_exposed)

    def test_success_is_approved_delivery(self) -> None:
        disposition = classify_review_first_qa(
            {"passed": True, "issues": []},
            content_binding_passed=True,
            review_first_enabled=True,
            member=True,
        )

        self.assertEqual(disposition.status, ReviewFirstStatus.APPROVED)
        self.assertEqual(disposition.artifact_role, "delivery")
        self.assertTrue(disposition.delivery_completion_message_allowed)

    def test_runtime_uses_monotonic_budget_for_optional_refinement(self) -> None:
        now = [100.0]
        runtime = ReviewFirstRuntime(enabled=True, _clock=lambda: now[0])
        runtime.checkpoint("selection")
        now[0] += 541.0

        self.assertFalse(runtime.optional_refinement_allowed)
        with self.assertRaises(ReviewFirstDeadlineExceeded):
            runtime.require_first_render_budget("candidate_search")
        payload = runtime.to_dict(status=ReviewFirstStatus.REVIEW_READY)
        self.assertEqual(payload["status"], "REVIEW_READY")
        self.assertEqual(payload["checkpoints"][0]["stage"], "selection")

    def test_external_deadline_counts_work_before_runtime_construction(self) -> None:
        now = [100.0]
        runtime = ReviewFirstRuntime(
            enabled=True,
            absolute_deadline_monotonic_seconds=650.0,
            _clock=lambda: now[0],
        )

        self.assertEqual(runtime.elapsed_seconds, 50.0)
        self.assertEqual(runtime.remaining_total_seconds, 550.0)
        self.assertEqual(runtime.bounded_timeout(420.0, "proxy"), 420.0)
        now[0] = 500.0
        self.assertEqual(runtime.bounded_timeout(420.0, "final"), 150.0)
        now[0] = 650.0
        with self.assertRaises(ReviewFirstDeadlineExceeded):
            runtime.require_total_budget("final QA")

    def test_total_deadline_cannot_be_swallowed_by_generic_recovery(self) -> None:
        now = [701.0]
        runtime = ReviewFirstRuntime(
            enabled=True,
            absolute_deadline_monotonic_seconds=700.0,
            _clock=lambda: now[0],
        )
        swallowed = False
        try:
            runtime.require_total_budget("fixture")
        except Exception:  # pragma: no cover - this is the behavior under test.
            swallowed = True
        except ReviewFirstDeadlineExceeded:
            pass
        self.assertFalse(swallowed)

    def test_boolean_deadline_and_budget_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            ReviewFirstRuntime(enabled=True, total_budget_seconds=True)
        with self.assertRaises(TypeError):
            ReviewFirstRuntime(
                enabled=True,
                absolute_deadline_monotonic_seconds=True,
            )
        with self.assertRaises(TypeError):
            with enforce_review_first_deadline(True, stage="invalid"):
                self.fail("boolean deadline must fail before entering context")

    @unittest.skipUnless(
        os.name == "posix"
        and threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGALRM"),
        "hard synchronous deadline requires a POSIX main thread",
    )
    def test_absolute_deadline_interrupts_blocking_work_and_restores_handler(
        self,
    ) -> None:
        previous_handler = signal.getsignal(signal.SIGALRM)
        started = time.monotonic()

        with self.assertRaises(ReviewFirstDeadlineExceeded):
            with enforce_review_first_deadline(
                time.monotonic() + 0.05,
                stage="blocking fixture",
            ):
                time.sleep(0.5)

        self.assertLess(time.monotonic() - started, 0.4)
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous_handler)


if __name__ == "__main__":
    unittest.main()
