"""Bounded render accounting for joint decoded candidate selection.

The joint selector has two different kinds of expensive work: the first
decoded proxy for a distinct candidate, and a same-path look refinement
rerender.  They intentionally have independent budgets so a useful look
repair cannot consume the remaining distinct-candidate search capacity.

This module owns accounting only.  It does not define or relax any visual,
technical, or perceptual QA threshold.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum
from typing import Any


JOINT_DECODED_DISTINCT_CANDIDATE_LIMIT = 24
JOINT_DECODED_LOOK_REFINEMENT_RENDER_LIMIT = 24
JOINT_DECODED_TOTAL_RENDER_SAFETY_LIMIT = 48
JOINT_DECODED_PER_PATH_REFINEMENT_LIMIT = 2


class JointDecodedRenderKind(str, Enum):
    """The two independently budgeted decoded proxy render kinds."""

    BASE_CANDIDATE = "base_candidate"
    LOOK_REFINEMENT = "look_refinement"


class JointDecodedBudgetExhaustionReason(str, Enum):
    """Stable terminal reasons suitable for production failure reports."""

    DISTINCT_CANDIDATE_LIMIT = "distinct_candidate_limit_exhausted"
    LOOK_REFINEMENT_RENDER_LIMIT = (
        "look_refinement_render_limit_exhausted"
    )
    TOTAL_RENDER_SAFETY_LIMIT = "total_render_safety_limit_exhausted"
    PER_PATH_REFINEMENT_LIMIT = (
        "per_path_refinement_limit_exhausted"
    )


@dataclass(frozen=True, slots=True)
class JointDecodedBudgetLimits:
    """Immutable capacity limits for one joint decoded selection session."""

    distinct_candidate_limit: int = JOINT_DECODED_DISTINCT_CANDIDATE_LIMIT
    look_refinement_render_limit: int = (
        JOINT_DECODED_LOOK_REFINEMENT_RENDER_LIMIT
    )
    total_render_safety_limit: int = (
        JOINT_DECODED_TOTAL_RENDER_SAFETY_LIMIT
    )
    per_path_refinement_limit: int = (
        JOINT_DECODED_PER_PATH_REFINEMENT_LIMIT
    )

    def __post_init__(self) -> None:
        values = {
            "distinct candidate limit": self.distinct_candidate_limit,
            "look refinement render limit": (
                self.look_refinement_render_limit
            ),
            "total render safety limit": self.total_render_safety_limit,
            "per-path refinement limit": self.per_path_refinement_limit,
        }
        for name, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.per_path_refinement_limit
            > self.look_refinement_render_limit
        ):
            raise ValueError(
                "per-path refinement limit cannot exceed the session "
                "refinement limit"
            )
        if self.total_render_safety_limit > (
            self.distinct_candidate_limit
            + self.look_refinement_render_limit
        ):
            raise ValueError(
                "total render safety limit cannot exceed the sum of its "
                "independent render budgets"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "distinct_candidate_limit": self.distinct_candidate_limit,
            "look_refinement_render_limit": (
                self.look_refinement_render_limit
            ),
            "total_render_safety_limit": self.total_render_safety_limit,
            "per_path_refinement_limit": self.per_path_refinement_limit,
        }


@dataclass(frozen=True, slots=True)
class JointDecodedLookRefinementEligibility:
    """Threshold-free policy inputs for authorizing a look-only rerender.

    The caller remains responsible for evaluating the existing QA gates.  A
    rerender is authorized only when that evidence is complete, every
    non-look gate passed, the look admission itself failed, and the decoded
    review has not already passed.
    """

    evidence_complete: bool
    nonlook_gates_passed: bool
    look_admission_passed: bool
    decoded_review_passed: bool

    def __post_init__(self) -> None:
        for name in (
            "evidence_complete",
            "nonlook_gates_passed",
            "look_admission_passed",
            "decoded_review_passed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")

    @property
    def authorized(self) -> bool:
        return bool(
            self.evidence_complete
            and self.nonlook_gates_passed
            and not self.look_admission_passed
            and not self.decoded_review_passed
        )

    @property
    def reason_code(self) -> str:
        if not self.evidence_complete:
            return "decoded_look_evidence_incomplete"
        if self.decoded_review_passed:
            return "decoded_review_already_passed"
        if not self.nonlook_gates_passed:
            return "nonlook_gate_failed"
        if self.look_admission_passed:
            return "look_admission_already_passed"
        return "eligible_look_only_failure"

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_complete": self.evidence_complete,
            "nonlook_gates_passed": self.nonlook_gates_passed,
            "look_admission_passed": self.look_admission_passed,
            "decoded_review_passed": self.decoded_review_passed,
            "authorized": self.authorized,
            "reason_code": self.reason_code,
            "thresholds_lowered": False,
        }


@dataclass(frozen=True, slots=True)
class JointDecodedRenderClaim:
    """A successful, monotonically numbered render-budget claim."""

    sequence: int
    render_kind: JointDecodedRenderKind
    candidate_index: int
    base_render_count: int
    look_refinement_render_count: int
    total_render_count: int
    path_refinement_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "render_kind": self.render_kind.value,
            "candidate_index": self.candidate_index,
            "base_render_count": self.base_render_count,
            "look_refinement_render_count": (
                self.look_refinement_render_count
            ),
            "total_render_count": self.total_render_count,
            "path_refinement_count": self.path_refinement_count,
        }


class JointDecodedBudgetExhausted(RuntimeError):
    """Typed, report-carrying budget exhaustion."""

    def __init__(
        self,
        reason: JointDecodedBudgetExhaustionReason,
        report_payload: dict[str, Any],
    ) -> None:
        if not isinstance(reason, JointDecodedBudgetExhaustionReason):
            raise TypeError("joint decoded exhaustion reason is invalid")
        if not isinstance(report_payload, dict):
            raise TypeError("joint decoded exhaustion report is invalid")
        self.reason = reason
        self.report_payload = report_payload
        super().__init__(reason.value)


class JointDecodedBudgetLedger:
    """Independent, fail-closed accounting for decoded proxy renders."""

    REPORT_CONTRACT = "joint_decoded_render_budget_v1"

    def __init__(
        self,
        limits: JointDecodedBudgetLimits | None = None,
    ) -> None:
        if limits is not None and not isinstance(
            limits,
            JointDecodedBudgetLimits,
        ):
            raise TypeError("joint decoded budget limits are invalid")
        self._limits = limits or JointDecodedBudgetLimits()
        self._candidate_paths: dict[int, Hashable] = {}
        self._path_candidates: dict[Hashable, int] = {}
        self._refinement_count_by_path: dict[Hashable, int] = {}
        self._base_render_count = 0
        self._look_refinement_render_count = 0
        self._claim_sequence = 0
        self._ineligible_refinement_request_count = 0
        self._assert_invariants()

    @property
    def limits(self) -> JointDecodedBudgetLimits:
        return self._limits

    @property
    def distinct_candidate_count(self) -> int:
        return len(self._candidate_paths)

    @property
    def base_render_count(self) -> int:
        return self._base_render_count

    @property
    def look_refinement_render_count(self) -> int:
        return self._look_refinement_render_count

    @property
    def total_render_count(self) -> int:
        return self._base_render_count + self._look_refinement_render_count

    @property
    def remaining_distinct_candidate_count(self) -> int:
        return (
            self._limits.distinct_candidate_limit
            - self.distinct_candidate_count
        )

    @property
    def remaining_look_refinement_render_count(self) -> int:
        return (
            self._limits.look_refinement_render_limit
            - self._look_refinement_render_count
        )

    @property
    def remaining_total_render_count(self) -> int:
        return (
            self._limits.total_render_safety_limit
            - self.total_render_count
        )

    def path_refinement_count(self, path_identity: Hashable) -> int:
        path = self._validate_path_identity(path_identity)
        return self._refinement_count_by_path.get(path, 0)

    def claim_base_render(
        self,
        *,
        candidate_index: int,
        path_identity: Hashable,
    ) -> JointDecodedRenderClaim:
        """Claim the single base render for one new distinct candidate."""

        candidate = self._validate_candidate_index(candidate_index)
        path = self._validate_path_identity(path_identity)
        if candidate in self._candidate_paths:
            raise ValueError("candidate already claimed a base render")
        if path in self._path_candidates:
            raise ValueError("exact path already claimed a base render")
        self._raise_if_total_exhausted()
        if (
            self.distinct_candidate_count
            >= self._limits.distinct_candidate_limit
        ):
            self._raise_exhausted(
                JointDecodedBudgetExhaustionReason.DISTINCT_CANDIDATE_LIMIT
            )
        self._candidate_paths[candidate] = path
        self._path_candidates[path] = candidate
        self._refinement_count_by_path[path] = 0
        self._base_render_count += 1
        self._claim_sequence += 1
        self._assert_invariants()
        return self._claim(
            JointDecodedRenderKind.BASE_CANDIDATE,
            candidate,
            path,
        )

    def claim_look_refinement(
        self,
        *,
        candidate_index: int,
        path_identity: Hashable,
        eligibility: JointDecodedLookRefinementEligibility,
    ) -> JointDecodedRenderClaim | None:
        """Claim an eligible same-path look rerender without using base budget.

        An ineligible request returns ``None`` and records a policy rejection;
        it never consumes a render claim.
        """

        candidate = self._validate_candidate_index(candidate_index)
        path = self._validate_path_identity(path_identity)
        if not isinstance(
            eligibility,
            JointDecodedLookRefinementEligibility,
        ):
            raise TypeError("look refinement eligibility is invalid")
        registered_path = self._candidate_paths.get(candidate)
        if registered_path is None:
            raise ValueError("look refinement requires a claimed base render")
        if registered_path != path:
            raise ValueError(
                "look refinement path does not match its base render"
            )
        if not eligibility.authorized:
            self._ineligible_refinement_request_count += 1
            self._assert_invariants()
            return None
        current_path_count = self._refinement_count_by_path[path]
        if current_path_count >= self._limits.per_path_refinement_limit:
            self._raise_exhausted(
                JointDecodedBudgetExhaustionReason.PER_PATH_REFINEMENT_LIMIT
            )
        self._raise_if_total_exhausted()
        if (
            self._look_refinement_render_count
            >= self._limits.look_refinement_render_limit
        ):
            self._raise_exhausted(
                JointDecodedBudgetExhaustionReason
                .LOOK_REFINEMENT_RENDER_LIMIT
            )
        self._refinement_count_by_path[path] = current_path_count + 1
        self._look_refinement_render_count += 1
        self._claim_sequence += 1
        self._assert_invariants()
        return self._claim(
            JointDecodedRenderKind.LOOK_REFINEMENT,
            candidate,
            path,
        )

    def report_payload(self) -> dict[str, Any]:
        """Return JSON-safe accounting without exposing path identities."""

        self._assert_invariants()
        paths_at_limit = sum(
            value >= self._limits.per_path_refinement_limit
            for value in self._refinement_count_by_path.values()
        )
        return {
            "contract": self.REPORT_CONTRACT,
            "limits": self._limits.to_dict(),
            "counts": {
                "distinct_candidate_count": self.distinct_candidate_count,
                "base_render_count": self._base_render_count,
                "look_refinement_render_count": (
                    self._look_refinement_render_count
                ),
                "total_render_count": self.total_render_count,
                "tracked_path_count": len(self._path_candidates),
                "paths_at_refinement_limit": paths_at_limit,
                "ineligible_refinement_request_count": (
                    self._ineligible_refinement_request_count
                ),
                "successful_claim_count": self._claim_sequence,
            },
            "remaining": {
                "distinct_candidate_count": (
                    self.remaining_distinct_candidate_count
                ),
                "look_refinement_render_count": (
                    self.remaining_look_refinement_render_count
                ),
                "total_render_count": self.remaining_total_render_count,
            },
            "invariants": {
                "total_equals_base_plus_refinement": True,
                "one_base_render_per_distinct_candidate": True,
                "per_path_refinement_limit_satisfied": True,
            },
            "thresholds_lowered": False,
        }

    def _claim(
        self,
        kind: JointDecodedRenderKind,
        candidate_index: int,
        path_identity: Hashable,
    ) -> JointDecodedRenderClaim:
        return JointDecodedRenderClaim(
            sequence=self._claim_sequence,
            render_kind=kind,
            candidate_index=candidate_index,
            base_render_count=self._base_render_count,
            look_refinement_render_count=(
                self._look_refinement_render_count
            ),
            total_render_count=self.total_render_count,
            path_refinement_count=(
                self._refinement_count_by_path[path_identity]
            ),
        )

    def _raise_if_total_exhausted(self) -> None:
        if self.total_render_count >= self._limits.total_render_safety_limit:
            self._raise_exhausted(
                JointDecodedBudgetExhaustionReason.TOTAL_RENDER_SAFETY_LIMIT
            )

    def _raise_exhausted(
        self,
        reason: JointDecodedBudgetExhaustionReason,
    ) -> None:
        raise JointDecodedBudgetExhausted(reason, self.report_payload())

    def _assert_invariants(self) -> None:
        if self._base_render_count != len(self._candidate_paths):
            raise RuntimeError(
                "joint decoded base-render accounting diverged"
            )
        if len(self._candidate_paths) != len(self._path_candidates):
            raise RuntimeError(
                "joint decoded candidate/path accounting diverged"
            )
        if set(self._path_candidates) != set(
            self._refinement_count_by_path
        ):
            raise RuntimeError(
                "joint decoded refinement path accounting diverged"
            )
        if self._look_refinement_render_count != sum(
            self._refinement_count_by_path.values()
        ):
            raise RuntimeError(
                "joint decoded refinement-render accounting diverged"
            )
        if self.total_render_count != (
            self._base_render_count
            + self._look_refinement_render_count
        ):
            raise RuntimeError(
                "joint decoded total-render accounting diverged"
            )
        if any(
            count < 0
            or count > self._limits.per_path_refinement_limit
            for count in self._refinement_count_by_path.values()
        ):
            raise RuntimeError(
                "joint decoded per-path refinement accounting diverged"
            )
        if self.distinct_candidate_count > (
            self._limits.distinct_candidate_limit
        ):
            raise RuntimeError("joint decoded candidate budget was exceeded")
        if self._look_refinement_render_count > (
            self._limits.look_refinement_render_limit
        ):
            raise RuntimeError(
                "joint decoded look-refinement budget was exceeded"
            )
        if self.total_render_count > self._limits.total_render_safety_limit:
            raise RuntimeError(
                "joint decoded total-render safety budget was exceeded"
            )

    @staticmethod
    def _validate_candidate_index(candidate_index: int) -> int:
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or candidate_index < 0
        ):
            raise ValueError(
                "joint decoded candidate index must be non-negative"
            )
        return candidate_index

    @staticmethod
    def _validate_path_identity(path_identity: Hashable) -> Hashable:
        if path_identity is None or isinstance(path_identity, bool):
            raise ValueError("joint decoded path identity is invalid")
        if isinstance(path_identity, str) and (
            not path_identity or path_identity != path_identity.strip()
        ):
            raise ValueError("joint decoded path identity is invalid")
        try:
            hash(path_identity)
        except TypeError as exc:
            raise TypeError(
                "joint decoded path identity must be hashable"
            ) from exc
        return path_identity
