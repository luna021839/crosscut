from __future__ import annotations

import unittest

from crosscut.pipeline import (
    _direct_member_feedback_source_diversity_candidates,
)


class DirectMemberFeedbackSourceDiversityTests(unittest.TestCase):
    def test_without_decoded_feedback_keeps_bounded_minimum_cohort(self) -> None:
        self.assertEqual(
            _direct_member_feedback_source_diversity_candidates(
                minimum_required_source_count=6,
                maximum_renderable_source_count=8,
                feedback_certificate_count=0,
            ),
            (6,),
        )

    def test_feedback_uses_one_joint_minimum_diversity_solve(self) -> None:
        self.assertEqual(
            _direct_member_feedback_source_diversity_candidates(
                minimum_required_source_count=6,
                maximum_renderable_source_count=8,
                feedback_certificate_count=3,
            ),
            (6,),
        )

    def test_full_source_request_does_not_duplicate_candidate(self) -> None:
        self.assertEqual(
            _direct_member_feedback_source_diversity_candidates(
                minimum_required_source_count=8,
                maximum_renderable_source_count=8,
                feedback_certificate_count=9,
            ),
            (8,),
        )

    def test_invalid_bounds_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            _direct_member_feedback_source_diversity_candidates(
                minimum_required_source_count=6,
                maximum_renderable_source_count=5,
                feedback_certificate_count=1,
            )
        with self.assertRaises(ValueError):
            _direct_member_feedback_source_diversity_candidates(
                minimum_required_source_count=6,
                maximum_renderable_source_count=8,
                feedback_certificate_count=-1,
            )


if __name__ == "__main__":
    unittest.main()
