"""Frame-local repair for an already selected cross-edit boundary.

The source path is deliberately out of scope here.  Given one nominal cut and
the two sources already selected on either side, this module searches a bounded
number of output frames earlier or later (up to eight for a proxy-triggered
partial re-edit).  It first avoids broadcaster camera cuts
that would remain visible immediately beside the edit, then compares visual
continuity cost, and finally uses beat distance as a small soft preference.

All clocks are explicit::

    source_local_time = output_master_time + source_offset_seconds

The functions are pure.  They do not decode frames, inspect global timelines,
or mutate source selection, which makes them suitable for both a descriptor
callback and precomputed per-offset costs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping


BOUNDARY_REPAIR_POLICY = (
    "fixed_source_local_frame_search_scene_safe_visual_primary_soft_beat_v1"
)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _normalized_scene_times(
    values: tuple[float, ...] | list[float],
    name: str,
) -> tuple[float, ...]:
    normalized = tuple(_finite(value, name) for value in values)
    if any(value < 0.0 for value in normalized):
        raise ValueError(f"{name} cannot contain negative times")
    return tuple(sorted(set(normalized)))


@dataclass(frozen=True, slots=True)
class BoundaryRepairConfig:
    """Local search and scoring policy.

    ``visual_cost_slack`` defines a near-equivalent visual band.  Beat cadence
    may choose within that band, but can never promote a candidate whose visual
    cost lies farther from the best safe candidate.  This keeps cadence soft
    rather than turning a beat grid into a mandatory cutoff.
    """

    search_radius_frames: int = 3
    minimum_segment_frames: int = 2
    interior_scene_cut_guard_frames: int = 4
    visual_cost_slack: float = 0.015
    beat_penalty_weight: float = 0.04

    def __post_init__(self) -> None:
        if (
            isinstance(self.search_radius_frames, bool)
            or not isinstance(self.search_radius_frames, int)
            or not 0 <= self.search_radius_frames <= 8
        ):
            raise ValueError("search_radius_frames must be an integer from 0 to 8")
        if (
            isinstance(self.minimum_segment_frames, bool)
            or not isinstance(self.minimum_segment_frames, int)
            or self.minimum_segment_frames < 2
        ):
            raise ValueError("minimum_segment_frames must be at least 2")
        if (
            isinstance(self.interior_scene_cut_guard_frames, bool)
            or not isinstance(self.interior_scene_cut_guard_frames, int)
            or self.interior_scene_cut_guard_frames < 1
        ):
            raise ValueError(
                "interior_scene_cut_guard_frames must be a positive integer"
            )
        # Global selection treats a detected cut within one decoded frame of the
        # nominal edit as that same quantized cut. A local move can expose that
        # formerly hidden frame too, so preserve the full search radius plus the
        # selector's one-frame quantization allowance.
        if self.interior_scene_cut_guard_frames < self.search_radius_frames + 1:
            raise ValueError(
                "interior_scene_cut_guard_frames must be at least "
                "search_radius_frames + 1"
            )
        slack = _finite(self.visual_cost_slack, "visual_cost_slack")
        beat_weight = _finite(self.beat_penalty_weight, "beat_penalty_weight")
        if slack < 0.0 or beat_weight < 0.0:
            raise ValueError("cost slack and beat penalty weight cannot be negative")
        object.__setattr__(self, "visual_cost_slack", slack)
        object.__setattr__(self, "beat_penalty_weight", beat_weight)


@dataclass(frozen=True, slots=True)
class BoundaryRepairRequest:
    """Timing and scene evidence for one fixed-source boundary."""

    nominal_boundary_seconds: float
    outgoing_segment_start_seconds: float
    incoming_segment_end_seconds: float
    output_fps: float
    outgoing_source_offset_seconds: float = 0.0
    incoming_source_offset_seconds: float = 0.0
    outgoing_scene_cut_times_seconds: tuple[float, ...] = ()
    incoming_scene_cut_times_seconds: tuple[float, ...] = ()
    beat_period_seconds: float | None = None
    beat_intensity: float | None = None

    def __post_init__(self) -> None:
        nominal = _finite(self.nominal_boundary_seconds, "nominal boundary")
        start = _finite(
            self.outgoing_segment_start_seconds,
            "outgoing segment start",
        )
        end = _finite(self.incoming_segment_end_seconds, "incoming segment end")
        fps = _finite(self.output_fps, "output fps")
        outgoing_offset = _finite(
            self.outgoing_source_offset_seconds,
            "outgoing source offset",
        )
        incoming_offset = _finite(
            self.incoming_source_offset_seconds,
            "incoming source offset",
        )
        if start < 0.0 or nominal <= start or end <= nominal:
            raise ValueError(
                "nominal boundary must lie strictly between its neighboring bounds"
            )
        if fps <= 0.0:
            raise ValueError("output fps must be positive")

        beat_period = self.beat_period_seconds
        beat_intensity = self.beat_intensity
        if beat_period is not None:
            beat_period = _finite(beat_period, "beat period")
            if beat_period <= 0.0:
                raise ValueError("beat period must be positive")
            if beat_intensity is None:
                beat_intensity = 1.0
            else:
                beat_intensity = _finite(beat_intensity, "beat intensity")
                if not 0.0 <= beat_intensity <= 1.0:
                    raise ValueError("beat intensity must be between zero and one")
        elif beat_intensity is not None:
            raise ValueError("beat intensity requires a beat period")

        object.__setattr__(self, "nominal_boundary_seconds", nominal)
        object.__setattr__(self, "outgoing_segment_start_seconds", start)
        object.__setattr__(self, "incoming_segment_end_seconds", end)
        object.__setattr__(self, "output_fps", fps)
        object.__setattr__(
            self,
            "outgoing_source_offset_seconds",
            outgoing_offset,
        )
        object.__setattr__(
            self,
            "incoming_source_offset_seconds",
            incoming_offset,
        )
        object.__setattr__(
            self,
            "outgoing_scene_cut_times_seconds",
            _normalized_scene_times(
                list(self.outgoing_scene_cut_times_seconds),
                "outgoing scene cut time",
            ),
        )
        object.__setattr__(
            self,
            "incoming_scene_cut_times_seconds",
            _normalized_scene_times(
                list(self.incoming_scene_cut_times_seconds),
                "incoming scene cut time",
            ),
        )
        object.__setattr__(self, "beat_period_seconds", beat_period)
        object.__setattr__(self, "beat_intensity", beat_intensity)


@dataclass(frozen=True, slots=True)
class BoundaryCandidateContext:
    """Candidate clocks supplied to an on-demand visual-cost callback."""

    frame_offset: int
    boundary_seconds: float
    outgoing_source_local_seconds: float
    incoming_source_local_seconds: float


BoundaryVisualCost = Callable[[BoundaryCandidateContext], float | None]


@dataclass(frozen=True, slots=True)
class BoundaryCandidateScore:
    frame_offset: int
    boundary_seconds: float
    outgoing_source_local_seconds: float
    incoming_source_local_seconds: float
    outgoing_interior_scene_cuts: tuple[float, ...]
    incoming_interior_scene_cuts: tuple[float, ...]
    visual_cost: float
    beat_deviation_seconds: float
    beat_deviation_periods: float
    beat_penalty: float
    combined_soft_cost: float

    @property
    def interior_scene_cut_count(self) -> int:
        return len(self.outgoing_interior_scene_cuts) + len(
            self.incoming_interior_scene_cuts
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_offset": self.frame_offset,
            "boundary_seconds": self.boundary_seconds,
            "outgoing_source_local_seconds": self.outgoing_source_local_seconds,
            "incoming_source_local_seconds": self.incoming_source_local_seconds,
            "outgoing_interior_scene_cuts": list(
                self.outgoing_interior_scene_cuts
            ),
            "incoming_interior_scene_cuts": list(
                self.incoming_interior_scene_cuts
            ),
            "interior_scene_cut_count": self.interior_scene_cut_count,
            "visual_cost": self.visual_cost,
            "beat_deviation_seconds": self.beat_deviation_seconds,
            "beat_deviation_periods": self.beat_deviation_periods,
            "beat_penalty": self.beat_penalty,
            "combined_soft_cost": self.combined_soft_cost,
        }


@dataclass(frozen=True, slots=True)
class BoundaryCandidateRejection:
    frame_offset: int
    boundary_seconds: float
    reason: str

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "frame_offset": self.frame_offset,
            "boundary_seconds": self.boundary_seconds,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class BoundaryRepairResult:
    policy: str
    nominal_boundary_seconds: float
    selected: BoundaryCandidateScore
    candidates: tuple[BoundaryCandidateScore, ...]
    rejected: tuple[BoundaryCandidateRejection, ...]
    minimum_interior_scene_cut_count: int
    best_visual_cost_at_minimum_scene_risk: float
    visual_shortlist_frame_offsets: tuple[int, ...]

    @property
    def selected_boundary_seconds(self) -> float:
        return self.selected.boundary_seconds

    @property
    def selected_frame_offset(self) -> int:
        return self.selected.frame_offset

    @property
    def changed(self) -> bool:
        return self.selected.frame_offset != 0

    @property
    def nominal_candidate(self) -> BoundaryCandidateScore | None:
        return next(
            (candidate for candidate in self.candidates if candidate.frame_offset == 0),
            None,
        )

    @property
    def camera_cut_avoidance_improved(self) -> bool:
        nominal = self.nominal_candidate
        return (
            nominal is not None
            and self.selected.interior_scene_cut_count
            < nominal.interior_scene_cut_count
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "nominal_boundary_seconds": self.nominal_boundary_seconds,
            "selected_boundary_seconds": self.selected_boundary_seconds,
            "selected_frame_offset": self.selected_frame_offset,
            "changed": self.changed,
            "camera_cut_avoidance_improved": self.camera_cut_avoidance_improved,
            "minimum_interior_scene_cut_count": (
                self.minimum_interior_scene_cut_count
            ),
            "best_visual_cost_at_minimum_scene_risk": (
                self.best_visual_cost_at_minimum_scene_risk
            ),
            "visual_shortlist_frame_offsets": list(
                self.visual_shortlist_frame_offsets
            ),
            "selected": self.selected.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "rejected": [item.to_dict() for item in self.rejected],
        }


def _precomputed_costs(
    values: Mapping[int, float],
    *,
    search_radius_frames: int,
) -> dict[int, float]:
    result: dict[int, float] = {}
    for raw_offset, raw_cost in values.items():
        if isinstance(raw_offset, bool) or not isinstance(raw_offset, int):
            raise TypeError("visual-cost offsets must be integers")
        if abs(raw_offset) > search_radius_frames:
            raise ValueError("visual-cost offset lies outside the configured search")
        cost = _finite(raw_cost, f"visual cost at offset {raw_offset}")
        if cost < 0.0:
            raise ValueError("visual costs cannot be negative")
        result[raw_offset] = cost
    return result


def _interior_scene_cuts(
    cuts: tuple[float, ...],
    *,
    visible_start: float,
    boundary: float,
    visible_end: float,
    guard_seconds: float,
    outgoing: bool,
) -> tuple[float, ...]:
    if outgoing:
        return tuple(
            cut
            for cut in cuts
            if visible_start < cut < boundary
            and boundary - cut <= guard_seconds + 1e-9
        )
    return tuple(
        cut
        for cut in cuts
        if boundary < cut < visible_end
        and cut - boundary <= guard_seconds + 1e-9
    )


def repair_boundary(
    request: BoundaryRepairRequest,
    *,
    visual_cost_callback: BoundaryVisualCost | None = None,
    visual_costs_by_frame_offset: Mapping[int, float] | None = None,
    config: BoundaryRepairConfig | None = None,
) -> BoundaryRepairResult:
    """Choose a locally repaired cut without changing either source.

    Exactly one visual-cost provider is required.  A callback receives all
    three relevant clocks; a precomputed mapping uses integer offsets in output
    frames.  Missing mapping entries and callbacks that return ``None`` simply
    make that candidate unavailable.
    """

    if not isinstance(request, BoundaryRepairRequest):
        raise TypeError("request must be a BoundaryRepairRequest")
    settings = config or BoundaryRepairConfig()
    if not isinstance(settings, BoundaryRepairConfig):
        raise TypeError("config must be a BoundaryRepairConfig")
    if (visual_cost_callback is None) == (visual_costs_by_frame_offset is None):
        raise ValueError("provide exactly one visual-cost callback or mapping")
    if visual_cost_callback is not None and not callable(visual_cost_callback):
        raise TypeError("visual_cost_callback must be callable")
    precomputed = (
        _precomputed_costs(
            visual_costs_by_frame_offset,
            search_radius_frames=settings.search_radius_frames,
        )
        if visual_costs_by_frame_offset is not None
        else None
    )

    frame_seconds = 1.0 / request.output_fps
    minimum_duration = settings.minimum_segment_frames * frame_seconds
    guard_seconds = settings.interior_scene_cut_guard_frames * frame_seconds
    outgoing_visible_start = (
        request.outgoing_segment_start_seconds
        + request.outgoing_source_offset_seconds
    )
    incoming_visible_end = (
        request.incoming_segment_end_seconds
        + request.incoming_source_offset_seconds
    )
    candidates: list[BoundaryCandidateScore] = []
    rejected: list[BoundaryCandidateRejection] = []

    for frame_offset in range(
        -settings.search_radius_frames,
        settings.search_radius_frames + 1,
    ):
        boundary = request.nominal_boundary_seconds + frame_offset * frame_seconds
        left_duration = boundary - request.outgoing_segment_start_seconds
        right_duration = request.incoming_segment_end_seconds - boundary
        if (
            left_duration + 1e-9 < minimum_duration
            or right_duration + 1e-9 < minimum_duration
        ):
            rejected.append(
                BoundaryCandidateRejection(
                    frame_offset,
                    boundary,
                    "minimum_two_frame_neighbor_violation",
                )
            )
            continue

        outgoing_local = boundary + request.outgoing_source_offset_seconds
        incoming_local = boundary + request.incoming_source_offset_seconds
        if outgoing_local < 0.0 or incoming_local < 0.0:
            rejected.append(
                BoundaryCandidateRejection(
                    frame_offset,
                    boundary,
                    "negative_source_local_time",
                )
            )
            continue

        context = BoundaryCandidateContext(
            frame_offset=frame_offset,
            boundary_seconds=boundary,
            outgoing_source_local_seconds=outgoing_local,
            incoming_source_local_seconds=incoming_local,
        )
        if precomputed is not None:
            raw_cost: float | None = precomputed.get(frame_offset)
        else:
            assert visual_cost_callback is not None
            raw_cost = visual_cost_callback(context)
        if raw_cost is None:
            rejected.append(
                BoundaryCandidateRejection(
                    frame_offset,
                    boundary,
                    "visual_cost_unavailable",
                )
            )
            continue
        visual_cost = _finite(raw_cost, f"visual cost at offset {frame_offset}")
        if visual_cost < 0.0:
            raise ValueError("visual costs cannot be negative")

        outgoing_cuts = _interior_scene_cuts(
            request.outgoing_scene_cut_times_seconds,
            visible_start=outgoing_visible_start,
            boundary=outgoing_local,
            visible_end=outgoing_local,
            guard_seconds=guard_seconds,
            outgoing=True,
        )
        incoming_cuts = _interior_scene_cuts(
            request.incoming_scene_cut_times_seconds,
            visible_start=incoming_local,
            boundary=incoming_local,
            visible_end=incoming_visible_end,
            guard_seconds=guard_seconds,
            outgoing=False,
        )
        beat_deviation = abs(boundary - request.nominal_boundary_seconds)
        if request.beat_period_seconds is None:
            beat_periods = 0.0
            beat_penalty = 0.0
        else:
            beat_periods = beat_deviation / request.beat_period_seconds
            assert request.beat_intensity is not None
            beat_penalty = (
                settings.beat_penalty_weight
                * request.beat_intensity
                * beat_periods
            )
        candidates.append(
            BoundaryCandidateScore(
                frame_offset=frame_offset,
                boundary_seconds=boundary,
                outgoing_source_local_seconds=outgoing_local,
                incoming_source_local_seconds=incoming_local,
                outgoing_interior_scene_cuts=outgoing_cuts,
                incoming_interior_scene_cuts=incoming_cuts,
                visual_cost=visual_cost,
                beat_deviation_seconds=beat_deviation,
                beat_deviation_periods=beat_periods,
                beat_penalty=beat_penalty,
                combined_soft_cost=visual_cost + beat_penalty,
            )
        )

    if not candidates:
        raise ValueError(
            "no boundary candidate preserves two-frame neighbors and visual evidence"
        )

    minimum_scene_risk = min(
        candidate.interior_scene_cut_count for candidate in candidates
    )
    scene_safe = tuple(
        candidate
        for candidate in candidates
        if candidate.interior_scene_cut_count == minimum_scene_risk
    )
    best_visual_cost = min(candidate.visual_cost for candidate in scene_safe)
    visual_shortlist = tuple(
        candidate
        for candidate in scene_safe
        if candidate.visual_cost
        <= best_visual_cost + settings.visual_cost_slack + 1e-12
    )
    selected = min(
        visual_shortlist,
        key=lambda candidate: (
            candidate.combined_soft_cost,
            candidate.beat_penalty,
            candidate.visual_cost,
            abs(candidate.frame_offset),
            candidate.frame_offset,
        ),
    )
    return BoundaryRepairResult(
        policy=BOUNDARY_REPAIR_POLICY,
        nominal_boundary_seconds=request.nominal_boundary_seconds,
        selected=selected,
        candidates=tuple(candidates),
        rejected=tuple(rejected),
        minimum_interior_scene_cut_count=minimum_scene_risk,
        best_visual_cost_at_minimum_scene_risk=best_visual_cost,
        visual_shortlist_frame_offsets=tuple(
            candidate.frame_offset for candidate in visual_shortlist
        ),
    )


__all__ = [
    "BOUNDARY_REPAIR_POLICY",
    "BoundaryCandidateContext",
    "BoundaryCandidateRejection",
    "BoundaryCandidateScore",
    "BoundaryRepairConfig",
    "BoundaryRepairRequest",
    "BoundaryRepairResult",
    "BoundaryVisualCost",
    "repair_boundary",
]
