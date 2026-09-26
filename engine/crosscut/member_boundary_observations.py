"""Exact, demand-addressable observations for a single member cut boundary.

This store has no pose/cost defaults and makes no editing decisions. An absent
source/offset is *unobserved*, not a missing target or a rejected transition.
Only the supplied exact observer can materialize it. Keep a store scoped to one
selection call/boundary; provider source/model identity checks remain in force.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Generic, Iterable, TypeVar


Sample = TypeVar("Sample")
SampleTimes = tuple[float, float, float, float, float]
Eligibility = tuple[bool, bool, bool, bool, bool]


@dataclass(frozen=True, slots=True)
class MemberBoundaryObservationPlan:
    offset_frames: int
    cut_master_seconds: float
    master_times: SampleTimes
    local_times_by_source: tuple[SampleTimes, ...]
    eligibility_by_source: tuple[Eligibility, ...]

    def __post_init__(self) -> None:
        if isinstance(self.offset_frames, bool) or not isinstance(self.offset_frames, int):
            raise ValueError("boundary offset must be an integer")
        if not self.local_times_by_source or len(self.local_times_by_source) != len(self.eligibility_by_source):
            raise ValueError("boundary source times and eligibility must match")
        for times in (self.master_times, *self.local_times_by_source):
            if len(times) != 5 or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) for value in times
            ):
                raise ValueError("boundary samples require five finite exact times")
        if any(len(values) != 5 or any(type(value) is not bool for value in values)
               for values in self.eligibility_by_source):
            raise ValueError("boundary eligibility requires five booleans")
        if (isinstance(self.cut_master_seconds, bool)
                or not isinstance(self.cut_master_seconds, (int, float))
                or not math.isfinite(self.cut_master_seconds)
                or self.master_times[2] != self.cut_master_seconds
                or any(left >= right for left, right in zip(self.master_times, self.master_times[1:]))):
            raise ValueError("boundary samples must straddle their exact cut")


class MemberBoundaryObservationStore(Generic[Sample]):
    """Observe only explicit source/offset demands; retain completed samples.

    ``observe`` receives the plan and source index. Content-invalid handles are
    interpreted by that authority, never synthesized here. ``prepare`` is only
    an optional dependency scheduler: it neither publishes nor approves samples.
    Exceptions/cancellation propagate, and an incomplete observation is not
    cached. No persistent cache or nearest-time substitution is introduced.
    """

    def __init__(
        self,
        plans: Iterable[MemberBoundaryObservationPlan],
        *,
        observe: Callable[[MemberBoundaryObservationPlan, int], tuple[Sample, ...]],
        prepare: Callable[[tuple[tuple[int, float], ...]], None] | None = None,
    ) -> None:
        self.plans = tuple(plans)
        if not self.plans or not any(plan.offset_frames == 0 for plan in self.plans):
            raise ValueError("boundary observations need their canonical plan")
        self._plans = {plan.offset_frames: plan for plan in self.plans}
        if len(self._plans) != len(self.plans):
            raise ValueError("boundary offsets must be unique")
        self.source_count = len(self.plans[0].local_times_by_source)
        if any(len(plan.local_times_by_source) != self.source_count for plan in self.plans):
            raise ValueError("boundary plans must share a source domain")
        self._observe = observe
        self._prepare = prepare
        self._samples: dict[tuple[int, int], tuple[Sample, ...]] = {}

    def _demand(self, offset_frames: int, source_index: int) -> tuple[int, int]:
        if (isinstance(offset_frames, bool) or not isinstance(offset_frames, int)
                or offset_frames not in self._plans):
            raise ValueError("boundary offset is outside the observation domain")
        if (isinstance(source_index, bool) or not isinstance(source_index, int)
                or not 0 <= source_index < self.source_count):
            raise ValueError("boundary source is outside the observation domain")
        return offset_frames, source_index

    def prepare(self, demands: Iterable[tuple[int, int]]) -> None:
        # Validate the entire demand before invoking any I/O. Do not prepare
        # sources already materialized, including observed missing-target results.
        keys = tuple(dict.fromkeys(self._demand(*demand) for demand in demands))
        if self._prepare is None:
            return
        queries = tuple(
            (source_index, local_seconds)
            for offset, source_index in keys if (offset, source_index) not in self._samples
            for local_seconds, eligible in zip(
                self._plans[offset].local_times_by_source[source_index],
                self._plans[offset].eligibility_by_source[source_index], strict=True)
            if eligible
        )
        if queries:
            self._prepare(queries)

    def samples_for(self, offset_frames: int, source_index: int) -> tuple[Sample, ...]:
        key = self._demand(offset_frames, source_index)
        if key not in self._samples:
            samples = self._observe(self._plans[offset_frames], source_index)
            if not isinstance(samples, tuple) or len(samples) != 5:
                raise RuntimeError("boundary observer did not return five exact samples")
            self._samples[key] = samples
        return self._samples[key]

    def observed_samples(self, offset_frames: int, source_index: int) -> tuple[Sample, ...]:
        """Read-only access. An unobserved demand raises, never returns None."""
        return self._samples[self._demand(offset_frames, source_index)]

    @property
    def observed_demands(self) -> frozenset[tuple[int, int]]:
        return frozenset(self._samples)
