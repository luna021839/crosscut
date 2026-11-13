"""Pure terminal-state reduction for sequential candidate attempts.

The reducer in this module deliberately has no dependency on pipeline, render,
or review payload classes.  Producers translate their result into exactly one
``TerminalCandidateVerdict`` and consumers can then make stop/continue
decisions without inferring success from a merely completed decoded review.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class CandidateVerdictStatus(str, Enum):
    """The mutually exclusive terminal outcomes of one candidate attempt."""

    DERIVATION_REJECTED = "derivation_rejected"
    TIMELINE_REJECTED = "timeline_rejected"
    SPLICE_BLOCKED = "splice_blocked"
    PROXY_EXECUTION_ERROR = "proxy_execution_error"
    DECODED_REVIEW_FAILED = "decoded_review_failed"
    DECODED_REVIEW_PASSED = "decoded_review_passed"
    SYSTEM_ERROR = "system_error"


class CandidateBlockerKind(str, Enum):
    """The stage that supplied a blocker for a non-passing verdict."""

    DERIVATION = "derivation"
    TIMELINE = "timeline"
    SPLICE = "splice"
    PROXY_EXECUTION = "proxy_execution"
    DECODED_REVIEW = "decoded_review"
    SYSTEM = "system"


class CandidateStreamAction(str, Enum):
    """Control signal after reducing all terminal verdicts seen so far."""

    CONTINUE = "continue"
    REQUIRED_PASSING_COHORT_COMPLETE = "required_passing_cohort_complete"
    ABORT_IMMEDIATELY = "abort_immediately"


_EXPECTED_BLOCKER_KIND = {
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
}


def _non_blank_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")
    return value


@dataclass(frozen=True, slots=True)
class CandidateBlocker:
    """One explicit reason that a candidate could not join the pass cohort."""

    kind: CandidateBlockerKind
    code: str
    message: str | None = None

    def __post_init__(self) -> None:
        try:
            kind = CandidateBlockerKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ValueError("kind must be a CandidateBlockerKind value") from exc
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "code", _non_blank_string(self.code, "code"))
        if self.message is not None:
            object.__setattr__(
                self,
                "message",
                _non_blank_string(self.message, "message"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class TerminalCandidateVerdict:
    """The sole terminal verdict for one attempted candidate."""

    candidate_index: int
    status: CandidateVerdictStatus
    blockers: tuple[CandidateBlocker, ...] = ()
    deferred_pre_proxy_geometry: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.candidate_index, bool)
            or not isinstance(self.candidate_index, int)
            or self.candidate_index < 0
        ):
            raise ValueError("candidate_index must be a non-negative integer")
        try:
            status = CandidateVerdictStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise ValueError("status must be a CandidateVerdictStatus value") from exc
        object.__setattr__(self, "status", status)

        if isinstance(self.blockers, (str, bytes)):
            raise TypeError("blockers must contain CandidateBlocker values")
        blockers = tuple(self.blockers)
        if any(not isinstance(item, CandidateBlocker) for item in blockers):
            raise TypeError("blockers must contain CandidateBlocker values")
        object.__setattr__(self, "blockers", blockers)

        if status is CandidateVerdictStatus.DECODED_REVIEW_PASSED:
            if blockers:
                raise ValueError("a passed decoded review cannot have blockers")
        else:
            expected_kind = _EXPECTED_BLOCKER_KIND[status]
            if not any(blocker.kind is expected_kind for blocker in blockers):
                raise ValueError(
                    f"{status.value} requires at least one "
                    f"{expected_kind.value} blocker"
                )
        if self.deferred_pre_proxy_geometry and status not in (
            CandidateVerdictStatus.PROXY_EXECUTION_ERROR,
            CandidateVerdictStatus.DECODED_REVIEW_FAILED,
            CandidateVerdictStatus.DECODED_REVIEW_PASSED,
        ):
            raise ValueError(
                "pre-proxy geometry deferral requires a proxy-stage verdict"
            )

    @property
    def timeline_review_passed(self) -> bool | None:
        """Return timeline outcome, or ``None`` when a system error obscures it."""

        if self.status is CandidateVerdictStatus.SYSTEM_ERROR:
            return None
        if any(
            blocker.kind is CandidateBlockerKind.TIMELINE
            for blocker in self.blockers
        ):
            return False
        if self.status is CandidateVerdictStatus.DERIVATION_REJECTED:
            return None
        return True

    @property
    def splice_review_passed(self) -> bool | None:
        """Return splice outcome when the candidate reached that stage."""

        if any(
            blocker.kind is CandidateBlockerKind.SPLICE
            for blocker in self.blockers
        ):
            return False
        if self.status in (
            CandidateVerdictStatus.PROXY_EXECUTION_ERROR,
            CandidateVerdictStatus.DECODED_REVIEW_FAILED,
            CandidateVerdictStatus.DECODED_REVIEW_PASSED,
        ):
            return True
        return None

    @property
    def decoded_review_completed(self) -> bool:
        return self.status in (
            CandidateVerdictStatus.DECODED_REVIEW_FAILED,
            CandidateVerdictStatus.DECODED_REVIEW_PASSED,
        )

    @property
    def decoded_review_passed(self) -> bool:
        return self.status is CandidateVerdictStatus.DECODED_REVIEW_PASSED

    def to_dict(self) -> dict[str, Any]:
        blockers = sorted(
            self.blockers,
            key=lambda item: (item.kind.value, item.code, item.message or ""),
        )
        return {
            "candidate_index": self.candidate_index,
            "terminal_verdict": self.status.value,
            "timeline_review_passed": self.timeline_review_passed,
            "splice_review_passed": self.splice_review_passed,
            "decoded_review_completed": self.decoded_review_completed,
            "decoded_review_passed": self.decoded_review_passed,
            "deferred_pre_proxy_geometry": (
                self.deferred_pre_proxy_geometry
            ),
            "blockers": [item.to_dict() for item in blockers],
        }


class CandidateStreamReducer:
    """Reduce one unique terminal verdict per attempted candidate.

    ``required_passing_count`` is intentionally a count of
    ``DECODED_REVIEW_PASSED`` verdicts, not a count of reviews that merely
    completed.  The reducer remains append-only so duplicates are visible as
    producer errors instead of silently replacing earlier evidence.
    """

    def __init__(self, *, required_passing_count: int = 1) -> None:
        if (
            isinstance(required_passing_count, bool)
            or not isinstance(required_passing_count, int)
            or required_passing_count < 1
        ):
            raise ValueError("required_passing_count must be a positive integer")
        self._required_passing_count = required_passing_count
        self._verdicts_by_candidate: dict[int, TerminalCandidateVerdict] = {}

    @property
    def required_passing_count(self) -> int:
        return self._required_passing_count

    @property
    def verdicts(self) -> tuple[TerminalCandidateVerdict, ...]:
        return tuple(
            self._verdicts_by_candidate[index]
            for index in sorted(self._verdicts_by_candidate)
        )

    @property
    def terminal_verdict_count(self) -> int:
        return len(self._verdicts_by_candidate)

    @property
    def attempted_candidate_count(self) -> int:
        return self.terminal_verdict_count

    def has_verdict(self, candidate_index: int) -> bool:
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or candidate_index < 0
        ):
            raise ValueError("candidate_index must be a non-negative integer")
        return candidate_index in self._verdicts_by_candidate

    @property
    def terminal_verdict_counts(self) -> dict[CandidateVerdictStatus, int]:
        counts = {status: 0 for status in CandidateVerdictStatus}
        for verdict in self._verdicts_by_candidate.values():
            counts[verdict.status] += 1
        if sum(counts.values()) != self.terminal_verdict_count:
            raise AssertionError("terminal verdict counters must be disjoint")
        return counts

    @property
    def decoded_review_completed_count(self) -> int:
        return sum(item.decoded_review_completed for item in self.verdicts)

    @property
    def decoded_review_passed_count(self) -> int:
        return sum(item.decoded_review_passed for item in self.verdicts)

    @property
    def required_passing_cohort(self) -> tuple[TerminalCandidateVerdict, ...]:
        passing = tuple(item for item in self.verdicts if item.decoded_review_passed)
        return passing[: self.required_passing_count]

    @property
    def required_passing_cohort_complete(self) -> bool:
        return self.decoded_review_passed_count >= self.required_passing_count

    @property
    def remaining_required_passing_count(self) -> int:
        return max(
            0,
            self.required_passing_count - self.decoded_review_passed_count,
        )

    @property
    def immediate_abort_requested(self) -> bool:
        return any(
            item.status is CandidateVerdictStatus.SYSTEM_ERROR
            for item in self._verdicts_by_candidate.values()
        )

    @property
    def abort_requested(self) -> bool:
        return self.immediate_abort_requested

    @property
    def action(self) -> CandidateStreamAction:
        if self.immediate_abort_requested:
            return CandidateStreamAction.ABORT_IMMEDIATELY
        if self.required_passing_cohort_complete:
            return CandidateStreamAction.REQUIRED_PASSING_COHORT_COMPLETE
        return CandidateStreamAction.CONTINUE

    @property
    def should_continue(self) -> bool:
        return self.action is CandidateStreamAction.CONTINUE

    def apply(self, verdict: TerminalCandidateVerdict) -> CandidateStreamAction:
        if not isinstance(verdict, TerminalCandidateVerdict):
            raise TypeError("verdict must be a TerminalCandidateVerdict")
        if verdict.candidate_index in self._verdicts_by_candidate:
            raise ValueError(
                "candidate already has a terminal verdict: "
                f"{verdict.candidate_index}"
            )
        self._verdicts_by_candidate[verdict.candidate_index] = verdict
        return self.action

    def apply_all(
        self,
        verdicts: Iterable[TerminalCandidateVerdict],
    ) -> CandidateStreamAction:
        for verdict in verdicts:
            self.apply(verdict)
        return self.action

    def to_dict(self) -> dict[str, Any]:
        counts = self.terminal_verdict_counts
        return {
            "schema_version": 1,
            "required_passing_count": self.required_passing_count,
            "attempted_candidate_count": self.attempted_candidate_count,
            "terminal_verdict_count": self.terminal_verdict_count,
            "terminal_verdict_counts": {
                status.value: counts[status] for status in CandidateVerdictStatus
            },
            "decoded_review_completed_count": (
                self.decoded_review_completed_count
            ),
            "decoded_review_passed_count": self.decoded_review_passed_count,
            "required_passing_cohort_candidate_indices": [
                item.candidate_index for item in self.required_passing_cohort
            ],
            "required_passing_cohort_complete": (
                self.required_passing_cohort_complete
            ),
            "remaining_required_passing_count": (
                self.remaining_required_passing_count
            ),
            "immediate_abort_requested": self.immediate_abort_requested,
            "stream_action": self.action.value,
            "should_continue": self.should_continue,
            "verdicts": [item.to_dict() for item in self.verdicts],
        }


__all__ = [
    "CandidateBlocker",
    "CandidateBlockerKind",
    "CandidateStreamAction",
    "CandidateStreamReducer",
    "CandidateVerdictStatus",
    "TerminalCandidateVerdict",
]
