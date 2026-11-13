"""Bounded in-process continuations for breadth-first joint look retries.

Retained bundles are not serialized or render authority. The renderer budget
ledger still authorizes every attempt; this queue only orders second attempts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class PendingJointLookRefinement:
    materialized: Any
    bundle: Any
    decoded_review: Any
    previous_refinement: Any


class JointLookBreadthScheduler:
    """Never revisit a look path while distinct-candidate admission is open."""

    def __init__(self, *, maximum_pending_count: int = 24) -> None:
        if isinstance(maximum_pending_count, bool) or not isinstance(
            maximum_pending_count, int
        ) or maximum_pending_count <= 0:
            raise ValueError("maximum pending count must be a positive integer")
        self.maximum_pending_count = maximum_pending_count
        self._pending: dict[int, PendingJointLookRefinement] = {}
        self._admission_closed = False
        self._resuming: int | None = None
        self._resumed_indices: list[int] = []

    @property
    def pending_indices(self) -> tuple[int, ...]:
        return tuple(self._pending)

    @property
    def admission_closed(self) -> bool:
        return self._admission_closed

    def get(self, candidate_index: int) -> PendingJointLookRefinement | None:
        return self._pending.get(candidate_index)

    def defer(self, candidate_index: int, state: PendingJointLookRefinement) -> None:
        if self._admission_closed:
            raise RuntimeError("cannot defer a first look after admission closes")
        if candidate_index in self._pending:
            raise RuntimeError("look path already has a pending continuation")
        if len(self._pending) >= self.maximum_pending_count:
            raise RuntimeError("look continuation capacity exhausted")
        if int(state.materialized.candidate_index) != candidate_index:
            raise ValueError("pending look candidate identity differs")
        self._pending[candidate_index] = state

    def close_admission(self) -> None:
        self._admission_closed = True

    def begin_resume(self, candidate_index: int) -> PendingJointLookRefinement:
        if not self._admission_closed:
            raise RuntimeError("second look cannot precede remaining first looks")
        if self._resuming is not None:
            raise RuntimeError("a look continuation is already running")
        if not self._pending or next(iter(self._pending)) != candidate_index:
            raise RuntimeError("look continuations must resume in admission order")
        self._resuming = candidate_index
        self._resumed_indices.append(candidate_index)
        return self._pending[candidate_index]

    def finish_resume(self, candidate_index: int) -> None:
        if self._resuming != candidate_index:
            raise RuntimeError("look completion does not match the active continuation")
        del self._pending[candidate_index]
        self._resuming = None

    def finish_terminal(self, candidate_index: int) -> None:
        """Release retained state after the production verdict is terminal."""
        if self._resuming is not None and self._resuming != candidate_index:
            raise RuntimeError("another look continuation is running")
        self._pending.pop(candidate_index, None)
        if self._resuming == candidate_index:
            self._resuming = None

    def assert_partition(
        self, started: Iterable[int], terminal: Iterable[int]
    ) -> None:
        started_set, terminal_set = set(started), set(terminal)
        pending_set = set(self._pending)
        if terminal_set & pending_set or started_set != terminal_set | pending_set:
            raise RuntimeError("started candidates must partition into terminal and pending")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": "joint_decoded_breadth_first_look_scheduler_v1",
            "distinct_candidate_admission_closed": self._admission_closed,
            "maximum_pending_count": self.maximum_pending_count,
            "pending_candidate_indices": list(self._pending),
            "resuming_candidate_index": self._resuming,
            "resumed_candidate_indices": list(self._resumed_indices),
            "pending_bundles_restartable": False,
            "pending_is_terminal_no_good": False,
            "thresholds_lowered": False,
        }
