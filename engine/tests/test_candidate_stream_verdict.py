from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import unittest

from crosscut.candidate_stream_verdict import (
    CandidateBlocker,
    CandidateBlockerKind,
    CandidateStreamAction,
    CandidateStreamReducer,
    CandidateVerdictStatus,
    TerminalCandidateVerdict,
)


def _blocker(
    kind: CandidateBlockerKind,
    code: str,
) -> CandidateBlocker:
    return CandidateBlocker(kind=kind, code=code)


def _verdict(
    candidate_index: int,
    status: CandidateVerdictStatus,
    *codes: str,
    extra_blockers: tuple[CandidateBlocker, ...] = (),
    deferred_pre_proxy_geometry: bool = False,
) -> TerminalCandidateVerdict:
    kind = {
        CandidateVerdictStatus.DERIVATION_REJECTED: (
            CandidateBlockerKind.DERIVATION
        ),
        CandidateVerdictStatus.TIMELINE_REJECTED: CandidateBlockerKind.TIMELINE,
        CandidateVerdictStatus.SPLICE_BLOCKED: CandidateBlockerKind.SPLICE,
        CandidateVerdictStatus.PROXY_EXECUTION_ERROR: (
            CandidateBlockerKind.PROXY_EXECUTION
        ),
        CandidateVerdictStatus.DECODED_REVIEW_FAILED: (
            CandidateBlockerKind.DECODED_REVIEW
        ),
        CandidateVerdictStatus.SYSTEM_ERROR: CandidateBlockerKind.SYSTEM,
    }.get(status)
    blockers = (
        extra_blockers
        if kind is None
        else tuple(_blocker(kind, code) for code in codes) + extra_blockers
    )
    return TerminalCandidateVerdict(
        candidate_index,
        status,
        blockers,
        deferred_pre_proxy_geometry,
    )


def _xoxz_like_verdicts() -> tuple[TerminalCandidateVerdict, ...]:
    """Eighteen terminal verdicts matching the real XOXZ run snapshot."""

    values = [
        _verdict(
            12,
            CandidateVerdictStatus.TIMELINE_REJECTED,
            "silhouette_and_formation_mismatch",
        ),
        _verdict(
            13,
            CandidateVerdictStatus.TIMELINE_REJECTED,
            "focus_silhouette_and_formation_mismatch",
        ),
        _verdict(30, CandidateVerdictStatus.TIMELINE_REJECTED, "focus_mismatch"),
        _verdict(
            32,
            CandidateVerdictStatus.SPLICE_BLOCKED,
            "fresh_splice_evidence_missing_94_96",
        ),
        _verdict(
            33,
            CandidateVerdictStatus.SPLICE_BLOCKED,
            "fresh_splice_evidence_missing_96",
        ),
    ]
    values.append(
        _verdict(
            34,
            CandidateVerdictStatus.SPLICE_BLOCKED,
            "fresh_splice_evidence_missing",
            extra_blockers=(
                _blocker(
                    CandidateBlockerKind.TIMELINE,
                    "incoming_boundary_side_instability",
                ),
            ),
        )
    )
    values.append(
        _verdict(
            35,
            CandidateVerdictStatus.SPLICE_BLOCKED,
            "fresh_splice_evidence_missing",
            extra_blockers=(
                _blocker(
                    CandidateBlockerKind.TIMELINE,
                    "instability_and_silhouette_mismatch",
                ),
            ),
        )
    )
    values.append(
        _verdict(
            36,
            CandidateVerdictStatus.TIMELINE_REJECTED,
            "source_run_and_focus_mismatch",
        )
    )
    values.extend(
        _verdict(
            index,
            CandidateVerdictStatus.SPLICE_BLOCKED,
            "fresh_splice_evidence_missing",
            extra_blockers=(
                _blocker(
                    CandidateBlockerKind.TIMELINE,
                    "source_run_or_visual_mismatch",
                ),
            ),
        )
        for index in range(37, 47)
    )
    return tuple(values)


class CandidateStreamVerdictTests(unittest.TestCase):
    def test_first_eight_decoded_failures_do_not_masquerade_as_a_pass(self) -> None:
        reducer = CandidateStreamReducer(required_passing_count=1)

        for candidate_index in range(1, 9):
            action = reducer.apply(
                _verdict(
                    candidate_index,
                    CandidateVerdictStatus.DECODED_REVIEW_FAILED,
                    f"decoded_failure_{candidate_index}",
                )
            )
            self.assertIs(action, CandidateStreamAction.CONTINUE)
            self.assertTrue(reducer.should_continue)

        self.assertEqual(reducer.decoded_review_completed_count, 8)
        self.assertEqual(reducer.decoded_review_passed_count, 0)
        self.assertEqual(reducer.required_passing_cohort, ())

        action = reducer.apply(
            _verdict(9, CandidateVerdictStatus.DECODED_REVIEW_PASSED)
        )

        self.assertIs(
            action,
            CandidateStreamAction.REQUIRED_PASSING_COHORT_COMPLETE,
        )
        self.assertFalse(reducer.should_continue)
        self.assertEqual(reducer.attempted_candidate_count, 9)
        self.assertEqual(reducer.decoded_review_completed_count, 9)
        self.assertEqual(reducer.decoded_review_passed_count, 1)
        self.assertEqual(
            tuple(item.candidate_index for item in reducer.required_passing_cohort),
            (9,),
        )

    def test_xoxz_like_sparse_fixture_has_disjoint_terminal_counters(self) -> None:
        reducer = CandidateStreamReducer(required_passing_count=2)
        reducer.apply_all(_xoxz_like_verdicts())

        expected_indices = (12, 13, 30, *range(32, 47))
        self.assertEqual(
            tuple(item.candidate_index for item in reducer.verdicts),
            expected_indices,
        )
        counts = reducer.terminal_verdict_counts
        self.assertEqual(
            counts,
            {
                CandidateVerdictStatus.DERIVATION_REJECTED: 0,
                CandidateVerdictStatus.TIMELINE_REJECTED: 4,
                CandidateVerdictStatus.SPLICE_BLOCKED: 14,
                CandidateVerdictStatus.PROXY_EXECUTION_ERROR: 0,
                CandidateVerdictStatus.DECODED_REVIEW_FAILED: 0,
                CandidateVerdictStatus.DECODED_REVIEW_PASSED: 0,
                CandidateVerdictStatus.SYSTEM_ERROR: 0,
            },
        )
        self.assertEqual(sum(counts.values()), 18)
        self.assertEqual(reducer.terminal_verdict_count, 18)
        self.assertEqual(reducer.decoded_review_completed_count, 0)
        self.assertEqual(reducer.decoded_review_passed_count, 0)
        self.assertEqual(reducer.required_passing_cohort, ())
        self.assertTrue(reducer.should_continue)

        by_index = {item.candidate_index: item for item in reducer.verdicts}
        for candidate_index in (32, 33):
            self.assertIs(by_index[candidate_index].timeline_review_passed, True)
            self.assertIs(by_index[candidate_index].splice_review_passed, False)
            self.assertFalse(by_index[candidate_index].decoded_review_completed)
        for candidate_index in (34, 35, *range(37, 47)):
            self.assertIs(by_index[candidate_index].timeline_review_passed, False)
            self.assertIs(by_index[candidate_index].splice_review_passed, False)
            self.assertEqual(len(by_index[candidate_index].blockers), 2)

    def test_to_dict_is_independent_of_apply_and_blocker_order(self) -> None:
        verdicts = _xoxz_like_verdicts()
        forward = CandidateStreamReducer(required_passing_count=2)
        reverse = CandidateStreamReducer(required_passing_count=2)
        forward.apply_all(verdicts)
        reverse.apply_all(
            replace(item, blockers=tuple(reversed(item.blockers)))
            for item in reversed(verdicts)
        )

        forward_payload = forward.to_dict()
        reverse_payload = reverse.to_dict()
        self.assertEqual(forward_payload, reverse_payload)
        self.assertEqual(
            json.dumps(forward_payload, separators=(",", ":")),
            json.dumps(reverse_payload, separators=(",", ":")),
        )
        self.assertEqual(
            list(forward_payload["terminal_verdict_counts"]),
            [status.value for status in CandidateVerdictStatus],
        )
        self.assertEqual(
            [item["candidate_index"] for item in forward_payload["verdicts"]],
            [12, 13, 30, *range(32, 47)],
        )
        candidate_33 = next(
            item
            for item in forward_payload["verdicts"]
            if item["candidate_index"] == 33
        )
        self.assertEqual(
            [item["code"] for item in candidate_33["blockers"]],
            ["fresh_splice_evidence_missing_96"],
        )

    def test_duplicate_terminal_verdict_raises_without_overwriting(self) -> None:
        reducer = CandidateStreamReducer()
        original = _verdict(
            7,
            CandidateVerdictStatus.DECODED_REVIEW_FAILED,
            "quality_floor",
        )
        reducer.apply(original)

        with self.assertRaisesRegex(ValueError, "already has a terminal verdict"):
            reducer.apply(
                _verdict(7, CandidateVerdictStatus.DECODED_REVIEW_PASSED)
            )

        self.assertEqual(reducer.verdicts, (original,))

    def test_system_error_requests_immediate_abort(self) -> None:
        reducer = CandidateStreamReducer(required_passing_count=3)

        action = reducer.apply(
            _verdict(
                30,
                CandidateVerdictStatus.SYSTEM_ERROR,
                "proxy_decoder_crashed",
            )
        )

        self.assertIs(action, CandidateStreamAction.ABORT_IMMEDIATELY)
        self.assertTrue(reducer.immediate_abort_requested)
        self.assertTrue(reducer.abort_requested)
        self.assertFalse(reducer.should_continue)
        self.assertEqual(reducer.remaining_required_passing_count, 3)

    def test_required_passing_count_is_configurable_and_positive(self) -> None:
        reducer = CandidateStreamReducer(required_passing_count=2)
        reducer.apply(_verdict(1, CandidateVerdictStatus.DECODED_REVIEW_PASSED))
        self.assertTrue(reducer.should_continue)
        reducer.apply(_verdict(2, CandidateVerdictStatus.DECODED_REVIEW_PASSED))
        self.assertTrue(reducer.required_passing_cohort_complete)

        for invalid in (True, 0, -1, 1.5, "2"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    CandidateStreamReducer(required_passing_count=invalid)  # type: ignore[arg-type]

    def test_verdicts_and_blockers_are_frozen_and_semantically_valid(self) -> None:
        blocker = CandidateBlocker("splice", "overlap")  # type: ignore[arg-type]
        verdict = TerminalCandidateVerdict(
            32,
            "splice_blocked",  # type: ignore[arg-type]
            (blocker,),
        )
        self.assertIs(blocker.kind, CandidateBlockerKind.SPLICE)
        self.assertIs(verdict.status, CandidateVerdictStatus.SPLICE_BLOCKED)

        deferred = _verdict(
            33,
            CandidateVerdictStatus.DECODED_REVIEW_FAILED,
            "decoded_geometry_unresolved",
            deferred_pre_proxy_geometry=True,
        )
        self.assertTrue(deferred.deferred_pre_proxy_geometry)
        self.assertTrue(deferred.to_dict()["deferred_pre_proxy_geometry"])

        with self.assertRaises(FrozenInstanceError):
            blocker.code = "changed"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            verdict.status = CandidateVerdictStatus.DECODED_REVIEW_PASSED  # type: ignore[misc]
        with self.assertRaisesRegex(ValueError, "cannot have blockers"):
            TerminalCandidateVerdict(
                1,
                CandidateVerdictStatus.DECODED_REVIEW_PASSED,
                (_blocker(CandidateBlockerKind.DECODED_REVIEW, "stale"),),
            )
        with self.assertRaisesRegex(ValueError, "requires at least one"):
            TerminalCandidateVerdict(
                2,
                CandidateVerdictStatus.DECODED_REVIEW_FAILED,
            )
        with self.assertRaisesRegex(ValueError, "requires a proxy-stage"):
            _verdict(
                3,
                CandidateVerdictStatus.TIMELINE_REJECTED,
                "raw_timeline_failure",
                deferred_pre_proxy_geometry=True,
            )


if __name__ == "__main__":
    unittest.main()
