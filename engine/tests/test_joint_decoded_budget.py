from __future__ import annotations

import json
import unittest

from crosscut.joint_decoded_budget import (
    JOINT_DECODED_DISTINCT_CANDIDATE_LIMIT,
    JOINT_DECODED_LOOK_REFINEMENT_RENDER_LIMIT,
    JOINT_DECODED_PER_PATH_REFINEMENT_LIMIT,
    JOINT_DECODED_TOTAL_RENDER_SAFETY_LIMIT,
    JointDecodedBudgetExhausted,
    JointDecodedBudgetExhaustionReason,
    JointDecodedBudgetLedger,
    JointDecodedBudgetLimits,
    JointDecodedLookRefinementEligibility,
    JointDecodedRenderKind,
)


def _eligible_look_only_failure() -> (
    JointDecodedLookRefinementEligibility
):
    return JointDecodedLookRefinementEligibility(
        evidence_complete=True,
        nonlook_gates_passed=True,
        look_admission_passed=False,
        decoded_review_passed=False,
    )


class JointDecodedBudgetTests(unittest.TestCase):
    def test_production_defaults_keep_independent_bounded_capacity(
        self,
    ) -> None:
        ledger = JointDecodedBudgetLedger()

        self.assertEqual(
            ledger.limits.distinct_candidate_limit,
            JOINT_DECODED_DISTINCT_CANDIDATE_LIMIT,
        )
        self.assertEqual(
            ledger.limits.look_refinement_render_limit,
            JOINT_DECODED_LOOK_REFINEMENT_RENDER_LIMIT,
        )
        self.assertEqual(
            ledger.limits.total_render_safety_limit,
            JOINT_DECODED_TOTAL_RENDER_SAFETY_LIMIT,
        )
        self.assertEqual(
            ledger.limits.per_path_refinement_limit,
            JOINT_DECODED_PER_PATH_REFINEMENT_LIMIT,
        )

    def test_failed_first_eight_do_not_starve_later_exact_eight(
        self,
    ) -> None:
        ledger = JointDecodedBudgetLedger()
        eligibility = _eligible_look_only_failure()

        # First cohort: eight base renders and one failed refinement each.
        for candidate_index in range(8):
            path = f"first-{candidate_index}"
            ledger.claim_base_render(
                candidate_index=candidate_index,
                path_identity=path,
            )
            ledger.claim_look_refinement(
                candidate_index=candidate_index,
                path_identity=path,
                eligibility=eligibility,
            )

        self.assertEqual(ledger.base_render_count, 8)
        self.assertEqual(ledger.look_refinement_render_count, 8)
        self.assertEqual(ledger.total_render_count, 16)
        self.assertEqual(ledger.remaining_distinct_candidate_count, 16)
        self.assertEqual(ledger.remaining_look_refinement_render_count, 16)

        # A later exact-eight cohort can still use the full two refinements
        # per path.  The first cohort's rerenders never consumed base capacity.
        for candidate_index in range(8, 16):
            path = f"second-{candidate_index}"
            ledger.claim_base_render(
                candidate_index=candidate_index,
                path_identity=path,
            )
            for _iteration in range(2):
                ledger.claim_look_refinement(
                    candidate_index=candidate_index,
                    path_identity=path,
                    eligibility=eligibility,
                )

        self.assertEqual(ledger.distinct_candidate_count, 16)
        self.assertEqual(ledger.base_render_count, 16)
        self.assertEqual(ledger.look_refinement_render_count, 24)
        self.assertEqual(ledger.total_render_count, 40)
        self.assertEqual(ledger.remaining_distinct_candidate_count, 8)
        self.assertEqual(ledger.remaining_look_refinement_render_count, 0)
        self.assertEqual(ledger.remaining_total_render_count, 8)

    def test_nonlook_ineligible_request_consumes_no_refinement_claim(
        self,
    ) -> None:
        ledger = JointDecodedBudgetLedger()
        ledger.claim_base_render(candidate_index=0, path_identity="path-0")
        ineligible = JointDecodedLookRefinementEligibility(
            evidence_complete=True,
            nonlook_gates_passed=False,
            look_admission_passed=False,
            decoded_review_passed=False,
        )

        claim = ledger.claim_look_refinement(
            candidate_index=0,
            path_identity="path-0",
            eligibility=ineligible,
        )

        self.assertIsNone(claim)
        self.assertFalse(ineligible.authorized)
        self.assertEqual(ineligible.reason_code, "nonlook_gate_failed")
        self.assertEqual(ledger.base_render_count, 1)
        self.assertEqual(ledger.look_refinement_render_count, 0)
        self.assertEqual(ledger.total_render_count, 1)
        self.assertEqual(
            ledger.report_payload()["counts"][
                "ineligible_refinement_request_count"
            ],
            1,
        )

    def test_successful_claims_are_monotonic_and_kind_specific(self) -> None:
        ledger = JointDecodedBudgetLedger()
        base = ledger.claim_base_render(
            candidate_index=4,
            path_identity=("source", "shot"),
        )
        refinement = ledger.claim_look_refinement(
            candidate_index=4,
            path_identity=("source", "shot"),
            eligibility=_eligible_look_only_failure(),
        )

        self.assertEqual(base.sequence, 1)
        self.assertIs(base.render_kind, JointDecodedRenderKind.BASE_CANDIDATE)
        self.assertEqual(base.path_refinement_count, 0)
        self.assertIsNotNone(refinement)
        assert refinement is not None
        self.assertEqual(refinement.sequence, 2)
        self.assertIs(
            refinement.render_kind,
            JointDecodedRenderKind.LOOK_REFINEMENT,
        )
        self.assertEqual(refinement.path_refinement_count, 1)

    def test_per_path_third_refinement_has_typed_exhaustion(self) -> None:
        ledger = JointDecodedBudgetLedger()
        eligibility = _eligible_look_only_failure()
        ledger.claim_base_render(candidate_index=0, path_identity="path-0")
        for _iteration in range(2):
            ledger.claim_look_refinement(
                candidate_index=0,
                path_identity="path-0",
                eligibility=eligibility,
            )

        with self.assertRaises(JointDecodedBudgetExhausted) as caught:
            ledger.claim_look_refinement(
                candidate_index=0,
                path_identity="path-0",
                eligibility=eligibility,
            )

        self.assertIs(
            caught.exception.reason,
            JointDecodedBudgetExhaustionReason.PER_PATH_REFINEMENT_LIMIT,
        )
        self.assertEqual(ledger.look_refinement_render_count, 2)

    def test_distinct_candidate_limit_has_typed_exhaustion(self) -> None:
        ledger = JointDecodedBudgetLedger()
        for candidate_index in range(24):
            ledger.claim_base_render(
                candidate_index=candidate_index,
                path_identity=f"path-{candidate_index}",
            )

        with self.assertRaises(JointDecodedBudgetExhausted) as caught:
            ledger.claim_base_render(
                candidate_index=24,
                path_identity="path-24",
            )

        self.assertIs(
            caught.exception.reason,
            JointDecodedBudgetExhaustionReason.DISTINCT_CANDIDATE_LIMIT,
        )

    def test_global_refinement_limit_has_typed_exhaustion(self) -> None:
        ledger = JointDecodedBudgetLedger()
        eligibility = _eligible_look_only_failure()
        for candidate_index in range(13):
            ledger.claim_base_render(
                candidate_index=candidate_index,
                path_identity=f"path-{candidate_index}",
            )
        for candidate_index in range(12):
            for _iteration in range(2):
                ledger.claim_look_refinement(
                    candidate_index=candidate_index,
                    path_identity=f"path-{candidate_index}",
                    eligibility=eligibility,
                )

        with self.assertRaises(JointDecodedBudgetExhausted) as caught:
            ledger.claim_look_refinement(
                candidate_index=12,
                path_identity="path-12",
                eligibility=eligibility,
            )

        self.assertIs(
            caught.exception.reason,
            JointDecodedBudgetExhaustionReason
            .LOOK_REFINEMENT_RENDER_LIMIT,
        )
        self.assertEqual(ledger.total_render_count, 37)

    def test_total_safety_limit_has_typed_exhaustion(self) -> None:
        ledger = JointDecodedBudgetLedger(
            JointDecodedBudgetLimits(
                distinct_candidate_limit=2,
                look_refinement_render_limit=2,
                total_render_safety_limit=2,
                per_path_refinement_limit=2,
            )
        )
        ledger.claim_base_render(candidate_index=0, path_identity="path-0")
        ledger.claim_base_render(candidate_index=1, path_identity="path-1")

        with self.assertRaises(JointDecodedBudgetExhausted) as caught:
            ledger.claim_look_refinement(
                candidate_index=0,
                path_identity="path-0",
                eligibility=_eligible_look_only_failure(),
            )

        self.assertIs(
            caught.exception.reason,
            JointDecodedBudgetExhaustionReason.TOTAL_RENDER_SAFETY_LIMIT,
        )
        self.assertEqual(
            caught.exception.report_payload["counts"][
                "total_render_count"
            ],
            2,
        )

    def test_refinement_requires_exact_registered_base_path(self) -> None:
        ledger = JointDecodedBudgetLedger()
        eligibility = _eligible_look_only_failure()
        with self.assertRaisesRegex(ValueError, "claimed base render"):
            ledger.claim_look_refinement(
                candidate_index=0,
                path_identity="path-0",
                eligibility=eligibility,
            )
        ledger.claim_base_render(candidate_index=0, path_identity="path-0")
        with self.assertRaisesRegex(ValueError, "does not match"):
            ledger.claim_look_refinement(
                candidate_index=0,
                path_identity="different-path",
                eligibility=eligibility,
            )

    def test_report_is_json_safe_and_proves_accounting_invariant(self) -> None:
        ledger = JointDecodedBudgetLedger()
        ledger.claim_base_render(candidate_index=0, path_identity="path-0")
        ledger.claim_look_refinement(
            candidate_index=0,
            path_identity="path-0",
            eligibility=_eligible_look_only_failure(),
        )

        report = ledger.report_payload()
        encoded = json.dumps(report, sort_keys=True)

        self.assertIn("joint_decoded_render_budget_v1", encoded)
        self.assertEqual(report["counts"]["base_render_count"], 1)
        self.assertEqual(
            report["counts"]["look_refinement_render_count"],
            1,
        )
        self.assertEqual(report["counts"]["total_render_count"], 2)
        self.assertTrue(
            report["invariants"][
                "total_equals_base_plus_refinement"
            ]
        )
        self.assertFalse(report["thresholds_lowered"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
