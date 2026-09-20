from __future__ import annotations

import unittest

from crosscut.review_first_runtime import (
    ReviewFirstDeadlineExceeded,
    ReviewFirstRuntime,
    ReviewFirstStatus,
    classify_review_first_qa,
)


class ReviewFirstRuntimeTests(unittest.TestCase):
    def test_editorial_failure_is_exposed_only_as_review_ready(self) -> None:
        disposition = classify_review_first_qa(
            {
                "passed": False,
                "issues": [
                    {"code": "close_burst_density_below_reference"},
                    {"code": "pose_continuity_below_reference"},
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


if __name__ == "__main__":
    unittest.main()
