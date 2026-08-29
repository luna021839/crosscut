from __future__ import annotations

import unittest

from crosscut.visual_splice_feasibility import (
    DEFAULT_WIDE_FRAME_OFFSETS,
    SpliceFeasibilityAssessment,
    SpliceFeasibilityStatus,
    SpliceOffsetAssessment,
    classify_splice_evidence_audit,
)


def _stability() -> dict[str, float]:
    return {
        "scale_delta": 0.02,
        "structure_mismatch": 0.03,
        "pose_mismatch": 0.04,
        "shot_type_mismatch": 0.0,
    }


def _audit(
    frame_offset: int,
    *,
    passed: bool = False,
    professional: bool = True,
    scale: float = 0.10,
    focus: float = 0.10,
    silhouette: float = 0.10,
    formation: float = 0.10,
    include_stability: bool = True,
) -> dict[str, object]:
    visual = {
        "shot_scale_delta": scale,
        "focus_mismatch": focus,
        "layout_mismatch": 0.10,
        "persisted_layout_mismatch": 0.10,
        "silhouette_mismatch": silhouette,
        "formation_mismatch": formation,
        "pose_mismatch": 0.10,
    }
    if include_stability:
        visual["outgoing_side_stability"] = _stability()
        visual["incoming_side_stability"] = _stability()
    return {
        "frame_offset": frame_offset,
        "professional_gate_passed": professional,
        "production_transition_gate_passed": passed,
        "visual_evidence": visual,
    }


class VisualSpliceFeasibilityTests(unittest.TestCase):
    def test_any_raw_pass_wins_without_requiring_other_offsets(self) -> None:
        assessment = classify_splice_evidence_audit(
            (_audit(7, passed=True),),
        )

        self.assertIs(assessment.status, SpliceFeasibilityStatus.RAW_PASS)
        self.assertFalse(assessment.optimizer_no_good_eligible)

    def test_focus_or_scale_only_failure_requires_output_geometry_proof(self) -> None:
        assessment = classify_splice_evidence_audit(
            (_audit(0, focus=0.55),),
        )

        self.assertIs(
            assessment.status,
            SpliceFeasibilityStatus.RAW_GEOMETRY_PROOF_REQUIRED,
        )
        self.assertFalse(assessment.optimizer_no_good_eligible)

    def test_all_offsets_need_nonwaivable_failures_for_no_good(self) -> None:
        assessment = classify_splice_evidence_audit(
            tuple(_audit(offset, silhouette=0.50) for offset in range(-8, 9)),
        )

        self.assertIs(
            assessment.status,
            SpliceFeasibilityStatus.HARD_INCOMPATIBLE,
        )
        self.assertTrue(assessment.optimizer_no_good_eligible)
        self.assertTrue(
            all(item.hard_incompatible for item in assessment.offsets)
        )

    def test_short_plus_three_failure_cannot_poison_plus_eight_graph(self) -> None:
        short = classify_splice_evidence_audit(
            tuple(_audit(offset, formation=0.50) for offset in range(-3, 4)),
        )
        full_with_late_pass = classify_splice_evidence_audit(
            tuple(
                _audit(offset, passed=(offset == 7), formation=0.50)
                for offset in range(-8, 9)
            ),
        )

        self.assertIs(
            short.status,
            SpliceFeasibilityStatus.INDETERMINATE,
        )
        self.assertIs(
            full_with_late_pass.status,
            SpliceFeasibilityStatus.RAW_PASS,
        )

    def test_missing_descriptor_or_side_stability_stays_indeterminate(self) -> None:
        missing_professional = classify_splice_evidence_audit(
            tuple(
                _audit(offset, professional=False)
                for offset in range(-8, 9)
            ),
        )
        missing_stability = classify_splice_evidence_audit(
            tuple(
                _audit(offset, formation=0.50, include_stability=False)
                for offset in range(-8, 9)
            ),
        )

        self.assertIs(
            missing_professional.status,
            SpliceFeasibilityStatus.INDETERMINATE,
        )
        self.assertIs(
            missing_stability.status,
            SpliceFeasibilityStatus.INDETERMINATE,
        )
        self.assertFalse(missing_professional.optimizer_no_good_eligible)
        self.assertFalse(missing_stability.optimizer_no_good_eligible)

    def test_audit_and_serialization_are_deterministic(self) -> None:
        forward = tuple(
            _audit(offset, silhouette=0.50) for offset in range(-8, 9)
        )
        reverse = tuple(reversed(forward))
        left = classify_splice_evidence_audit(forward)
        right = classify_splice_evidence_audit(reverse)

        self.assertEqual(left, right)
        self.assertEqual(left.to_dict(), right.to_dict())
        self.assertFalse(left.to_dict()["thresholds_lowered"])

    def test_shifted_or_wrong_offset_domain_cannot_certify_hard(self) -> None:
        shifted = classify_splice_evidence_audit(
            tuple(_audit(offset, silhouette=0.50) for offset in range(100, 117))
        )
        missing_with_extra = classify_splice_evidence_audit(
            tuple(
                _audit(offset, silhouette=0.50)
                for offset in (*range(-8, 8), 99)
            )
        )

        self.assertIs(shifted.status, SpliceFeasibilityStatus.INDETERMINATE)
        self.assertIs(
            missing_with_extra.status,
            SpliceFeasibilityStatus.INDETERMINATE,
        )
        self.assertFalse(shifted.optimizer_no_good_eligible)
        self.assertFalse(missing_with_extra.optimizer_no_good_eligible)

    def test_duplicate_offsets_and_invalid_expected_domain_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            classify_splice_evidence_audit((_audit(0), _audit(0)))
        for invalid in ((), (0, 0), (True,), (1.0,), "0"):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    classify_splice_evidence_audit(
                        (_audit(0),),
                        expected_frame_offsets=invalid,  # type: ignore[arg-type]
                    )

    def test_terminal_status_cannot_be_asserted_without_matching_evidence(self) -> None:
        empty_offsets: tuple[SpliceOffsetAssessment, ...] = ()
        with self.assertRaisesRegex(ValueError, "contradicts"):
            SpliceFeasibilityAssessment(
                status=SpliceFeasibilityStatus.HARD_INCOMPATIBLE,
                expected_frame_offsets=DEFAULT_WIDE_FRAME_OFFSETS,
                offsets=empty_offsets,
                reason_codes=("all_allowed_offsets_nonwaivable_failure",),
            )
        hard_offset = SpliceOffsetAssessment(
            0,
            False,
            ("silhouette",),
            True,
        )
        with self.assertRaisesRegex(ValueError, "contradicts"):
            SpliceFeasibilityAssessment(
                status=SpliceFeasibilityStatus.RAW_PASS,
                expected_frame_offsets=(0,),
                offsets=(hard_offset,),
                reason_codes=("at_least_one_raw_offset_passed",),
            )
        with self.assertRaisesRegex(ValueError, "contradicts"):
            SpliceFeasibilityAssessment(
                status=SpliceFeasibilityStatus.RAW_GEOMETRY_PROOF_REQUIRED,
                expected_frame_offsets=(0,),
                offsets=(hard_offset,),
                reason_codes=("focus_or_scale_only_offset_available",),
            )

    def test_reason_code_cannot_be_forged_for_matching_status(self) -> None:
        passed_offset = SpliceOffsetAssessment(0, True, (), True)
        with self.assertRaisesRegex(ValueError, "reasons contradict"):
            SpliceFeasibilityAssessment(
                status=SpliceFeasibilityStatus.RAW_PASS,
                expected_frame_offsets=(0,),
                offsets=(passed_offset,),
                reason_codes=("allowed_offset_search_incomplete",),
            )

    def test_assessment_round_trip_is_strict_and_tamper_evident(self) -> None:
        assessment = classify_splice_evidence_audit(
            tuple(_audit(offset, silhouette=0.50) for offset in range(-8, 9))
        )
        payload = assessment.to_dict()

        self.assertEqual(
            SpliceFeasibilityAssessment.from_dict(payload),
            assessment,
        )
        tampered = dict(payload)
        tampered["status"] = SpliceFeasibilityStatus.RAW_PASS.value
        with self.assertRaises(ValueError):
            SpliceFeasibilityAssessment.from_dict(tampered)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
