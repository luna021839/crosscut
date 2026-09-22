"""Pre-render structural review for a selected cross-cut timeline.

The optimiser answers "which source is best at this decision span?".  This
module answers the complementary whole-edit question before an expensive
render: does the resulting sequence have the cadence, source coverage, and
boundary continuity of a plausible stage mix?

The reviewer is deliberately pure.  It opens no media and accepts the three
timeline representations used by the engine:

* a sequence of :class:`~crosscut.timeline.TimelineCandidate` objects;
* a :class:`~crosscut.timeline.TimelineResult`;
* a ``VisualTimelinePlan`` or its ``to_dict()`` JSON-shaped mapping.

Only evidence already present on a boundary is scored.  Missing pose or
silhouette data is reported as unavailable rather than silently converted into
a mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, log, sqrt
from typing import Any, Mapping, Sequence

from .pose import FormationPersonDescriptor
from .timeline import (
    BOUNDARY_SIDE_MAX_POSE_MISMATCH,
    BOUNDARY_SIDE_MAX_SCALE_DELTA,
    BOUNDARY_SIDE_MAX_SHOT_TYPE_MISMATCH,
    BOUNDARY_SIDE_MAX_STRUCTURE_MISMATCH,
    PRODUCTION_MAX_FOCUS_MISMATCH,
    PRODUCTION_MAX_FORMATION_MISMATCH,
    PRODUCTION_MAX_LAYOUT_MISMATCH,
    PRODUCTION_MAX_POSE_MISMATCH,
    PRODUCTION_MAX_SCALE_MISMATCH,
    PRODUCTION_MAX_SILHOUETTE_MISMATCH,
    BoundarySideStability,
    ShotType,
    boundary_side_stability_exceeded,
    comparable_group_formation_mismatch,
    intentional_phrase_anchor_authorized_mismatches,
    screen_layout_mismatch,
)


_MISSING = object()
JOINT_BOUNDARY_RESULT_EVIDENCE_CONTRACT = (
    "joint_decoded_post_transform_boundary_result_evidence_v1"
)
# Deliberately process-local and omitted from every serialized plan.  A JSON
# mapping can repeat the public contract label for audit, but cannot acquire
# authority to suppress fallback evidence.
JOINT_BOUNDARY_RESULT_EVIDENCE_AUTHORITY = object()


def _finite_non_negative(value: float, name: str) -> float:
    numeric = float(value)
    if not isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


def _unit_interval(value: float, name: str) -> float:
    numeric = float(value)
    if not isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return numeric


@dataclass(frozen=True, slots=True)
class TimelineReviewConfig:
    """Reference-calibrated structural and continuity review thresholds.

    The cadence defaults come from the current IVE reference analysis: a
    roughly 1.4 second visual-shot median and source changes at about 49% of
    comparable visual boundaries.  That implies multi-shot source runs are
    normal; it does *not* imply changing broadcaster at every opportunity.
    """

    reference_visual_shot_median_seconds: float = 1.4
    visual_shot_median_warning_seconds: float = 2.2
    reference_source_switch_ratio: float = 0.49
    preferred_min_source_switch_ratio: float = 0.34
    preferred_max_source_switch_ratio: float = 0.64
    reference_fan_source_switches_per_minute: float = 14.09
    reference_kbs_source_switches_per_minute: float = 15.39
    hard_min_source_switches_per_minute: float = 12.5
    preferred_min_source_switches_per_minute: float = 13.5
    preferred_max_source_switches_per_minute: float = 17.5
    preferred_min_source_run_seconds: float = 2.5
    preferred_max_source_run_seconds: float = 5.5
    reference_kbs_source_run_median_seconds: float = 3.47
    reference_fan_source_run_median_seconds: float = 3.92
    hard_max_source_run_median_seconds: float = 5.5
    # The reference source-run p90 is 7.20 seconds.  Treat it as an outlier
    # ceiling rather than allowing selected candidates to cluster just below
    # the former nine-second limit.
    reference_source_run_p90_seconds: float = 7.2
    hard_max_source_run_seconds: float = 7.2
    short_flash_seconds: float = 0.75
    ping_pong_middle_max_seconds: float = 5.5
    minimum_sources_when_available: int = 8
    dominant_source_share_warning: float = 0.28
    dominant_source_share_hard: float = 0.45
    continuity_tolerance_seconds: float = 0.050

    maximum_scale_mismatch: float = PRODUCTION_MAX_SCALE_MISMATCH
    maximum_focus_mismatch: float = PRODUCTION_MAX_FOCUS_MISMATCH
    maximum_layout_mismatch: float = PRODUCTION_MAX_LAYOUT_MISMATCH
    maximum_silhouette_mismatch: float = PRODUCTION_MAX_SILHOUETTE_MISMATCH
    maximum_formation_mismatch: float = PRODUCTION_MAX_FORMATION_MISMATCH
    maximum_pose_mismatch: float = PRODUCTION_MAX_POSE_MISMATCH
    maximum_boundary_side_scale_delta: float = BOUNDARY_SIDE_MAX_SCALE_DELTA
    maximum_boundary_side_structure_mismatch: float = (
        BOUNDARY_SIDE_MAX_STRUCTURE_MISMATCH
    )
    maximum_boundary_side_pose_mismatch: float = BOUNDARY_SIDE_MAX_POSE_MISMATCH
    maximum_boundary_side_shot_type_mismatch: float = (
        BOUNDARY_SIDE_MAX_SHOT_TYPE_MISMATCH
    )

    switch_ratio_weight: float = 12.0
    switch_density_weight: float = 14.0
    visual_shot_cadence_weight: float = 10.0
    source_run_weight: float = 15.0
    source_run_median_weight: float = 12.0
    diversity_weight: float = 24.0
    dominance_weight: float = 10.0
    ping_pong_weight: float = 8.0
    short_flash_weight: float = 12.0
    scene_cut_risk_weight: float = 18.0
    boundary_side_stability_weight: float = 18.0
    mismatch_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "scale": 7.0,
            "focus": 9.0,
            "layout": 9.0,
            "silhouette": 10.0,
            "formation": 12.0,
            "pose": 10.0,
        }
    )

    def __post_init__(self) -> None:
        positive_names = (
            "reference_visual_shot_median_seconds",
            "visual_shot_median_warning_seconds",
            "reference_fan_source_switches_per_minute",
            "reference_kbs_source_switches_per_minute",
            "hard_min_source_switches_per_minute",
            "preferred_min_source_switches_per_minute",
            "preferred_max_source_switches_per_minute",
            "preferred_min_source_run_seconds",
            "preferred_max_source_run_seconds",
            "reference_kbs_source_run_median_seconds",
            "reference_fan_source_run_median_seconds",
            "hard_max_source_run_median_seconds",
            "reference_source_run_p90_seconds",
            "hard_max_source_run_seconds",
            "short_flash_seconds",
            "ping_pong_middle_max_seconds",
        )
        for name in positive_names:
            value = _finite_non_negative(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be greater than zero")
        if not (
            self.reference_visual_shot_median_seconds
            <= self.visual_shot_median_warning_seconds
        ):
            raise ValueError("visual-shot median thresholds must be ordered")
        if not (
            self.hard_min_source_switches_per_minute
            <= self.preferred_min_source_switches_per_minute
            <= self.reference_fan_source_switches_per_minute
            <= self.preferred_max_source_switches_per_minute
            and self.preferred_min_source_switches_per_minute
            <= self.reference_kbs_source_switches_per_minute
            <= self.preferred_max_source_switches_per_minute
        ):
            raise ValueError("source-switch density thresholds must be ordered")
        if not (
            self.preferred_min_source_run_seconds
            <= self.reference_kbs_source_run_median_seconds
            <= self.reference_fan_source_run_median_seconds
            <= self.hard_max_source_run_median_seconds
            and self.preferred_max_source_run_seconds
            <= self.hard_max_source_run_median_seconds
            <= self.reference_source_run_p90_seconds
            <= self.hard_max_source_run_seconds
        ):
            raise ValueError("source-run thresholds must be ordered")
        for name in (
            "reference_source_switch_ratio",
            "preferred_min_source_switch_ratio",
            "preferred_max_source_switch_ratio",
            "dominant_source_share_warning",
            "dominant_source_share_hard",
            "maximum_scale_mismatch",
            "maximum_focus_mismatch",
            "maximum_layout_mismatch",
            "maximum_silhouette_mismatch",
            "maximum_formation_mismatch",
            "maximum_pose_mismatch",
            "maximum_boundary_side_scale_delta",
            "maximum_boundary_side_structure_mismatch",
            "maximum_boundary_side_pose_mismatch",
            "maximum_boundary_side_shot_type_mismatch",
        ):
            _unit_interval(getattr(self, name), name)
        if not (
            self.preferred_min_source_switch_ratio
            <= self.reference_source_switch_ratio
            <= self.preferred_max_source_switch_ratio
        ):
            raise ValueError("reference switch ratio must lie in the preferred range")
        if self.dominant_source_share_warning > self.dominant_source_share_hard:
            raise ValueError("dominant-source share thresholds must be ordered")
        if self.minimum_sources_when_available < 1:
            raise ValueError("minimum_sources_when_available must be at least one")
        _finite_non_negative(
            self.continuity_tolerance_seconds,
            "continuity_tolerance_seconds",
        )
        for name in (
            "switch_ratio_weight",
            "switch_density_weight",
            "visual_shot_cadence_weight",
            "source_run_weight",
            "source_run_median_weight",
            "diversity_weight",
            "dominance_weight",
            "ping_pong_weight",
            "short_flash_weight",
            "scene_cut_risk_weight",
            "boundary_side_stability_weight",
        ):
            _finite_non_negative(getattr(self, name), name)
        expected_metrics = {
            "scale",
            "focus",
            "layout",
            "silhouette",
            "formation",
            "pose",
        }
        if set(self.mismatch_weights) != expected_metrics:
            raise ValueError(
                "mismatch_weights must define scale, focus, layout, "
                "silhouette, formation, and pose"
            )
        for name, weight in self.mismatch_weights.items():
            _finite_non_negative(weight, f"mismatch_weights[{name!r}]")


@dataclass(frozen=True, slots=True)
class ReviewIssue:
    code: str
    severity: str
    message: str
    occurrences: tuple[int, ...] = ()
    value: float | int | None = None
    threshold: float | int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "occurrences": list(self.occurrences),
            "value": self.value,
            "threshold": self.threshold,
        }


@dataclass(frozen=True, slots=True)
class SourceRun:
    run_index: int
    source_id: str
    first_segment_position: int
    last_segment_position: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_index": self.run_index,
            "source_id": self.source_id,
            "first_segment_position": self.first_segment_position,
            "last_segment_position": self.last_segment_position,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True, slots=True)
class PingPongDiagnostic:
    run_index: int
    source_sequence: tuple[str, str, str]
    middle_duration_seconds: float
    within_review_window: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_index": self.run_index,
            "source_sequence": list(self.source_sequence),
            "middle_duration_seconds": self.middle_duration_seconds,
            "within_review_window": self.within_review_window,
        }


@dataclass(frozen=True, slots=True)
class BoundaryDiagnostic:
    boundary_index: int
    outgoing_segment_index: int
    incoming_segment_index: int
    at_seconds: float
    outgoing_source_id: str
    incoming_source_id: str
    source_switch: bool
    visual_boundary: bool
    scene_cut_risk: bool
    outgoing_side_stability: BoundarySideStability | None = None
    incoming_side_stability: BoundarySideStability | None = None
    outgoing_side_stability_exceeded_metrics: tuple[str, ...] = ()
    incoming_side_stability_exceeded_metrics: tuple[str, ...] = ()
    scale_mismatch: float | None = None
    focus_mismatch: float | None = None
    layout_mismatch: float | None = None
    silhouette_mismatch: float | None = None
    formation_mismatch: float | None = None
    pose_mismatch: float | None = None
    intentional_transition_anchor_basis: str | None = None
    authorized_mismatch_metrics: tuple[str, ...] = ()
    exceeded_metrics: tuple[str, ...] = ()

    @property
    def available_metrics(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (
                "scale",
                "focus",
                "layout",
                "silhouette",
                "formation",
                "pose",
            )
            if getattr(self, f"{name}_mismatch") is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_index": self.boundary_index,
            "outgoing_segment_index": self.outgoing_segment_index,
            "incoming_segment_index": self.incoming_segment_index,
            "at_seconds": self.at_seconds,
            "outgoing_source_id": self.outgoing_source_id,
            "incoming_source_id": self.incoming_source_id,
            "source_switch": self.source_switch,
            "visual_boundary": self.visual_boundary,
            "scene_cut_risk": self.scene_cut_risk,
            "boundary_side_stability": {
                "outgoing": (
                    {
                        **self.outgoing_side_stability.to_dict(),
                        "exceeded_metrics": list(
                            self.outgoing_side_stability_exceeded_metrics
                        ),
                    }
                    if self.outgoing_side_stability is not None
                    else None
                ),
                "incoming": (
                    {
                        **self.incoming_side_stability.to_dict(),
                        "exceeded_metrics": list(
                            self.incoming_side_stability_exceeded_metrics
                        ),
                    }
                    if self.incoming_side_stability is not None
                    else None
                ),
            },
            "available_metrics": list(self.available_metrics),
            "mismatches": {
                name: getattr(self, f"{name}_mismatch")
                for name in self.available_metrics
            },
            "intentional_transition_authority": {
                "anchor_basis": self.intentional_transition_anchor_basis,
                "authorized_mismatch_metrics": list(
                    self.authorized_mismatch_metrics
                ),
                "applied": bool(self.authorized_mismatch_metrics),
            },
            "exceeded_metrics": list(self.exceeded_metrics),
        }


@dataclass(frozen=True, slots=True)
class MismatchSummary:
    metric: str
    threshold: float
    available_count: int
    mean: float | None
    maximum: float | None
    exceeded_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "threshold": self.threshold,
            "available_count": self.available_count,
            "mean": self.mean,
            "maximum": self.maximum,
            "exceeded_count": self.exceeded_count,
        }


@dataclass(frozen=True, slots=True)
class TimelineReview:
    segment_count: int
    boundary_count: int
    internal_scene_cut_count: int
    comparable_visual_boundary_count: int
    visual_shot_durations_seconds: tuple[float, ...]
    visual_shot_median_seconds: float
    total_duration_seconds: float
    source_switch_count: int
    source_switch_rate: float
    source_switches_per_minute: float
    unique_source_count: int
    available_source_count: int | None
    required_unique_source_count: int | None
    source_duration_seconds: Mapping[str, float]
    per_source_share: Mapping[str, float]
    source_runs: tuple[SourceRun, ...]
    source_run_median_seconds: float
    longest_source_run_seconds: float
    ping_pongs: tuple[PingPongDiagnostic, ...]
    short_flash_run_indices: tuple[int, ...]
    scene_cut_risk_boundary_indices: tuple[int, ...]
    outgoing_boundary_side_instability_indices: tuple[int, ...]
    incoming_boundary_side_instability_indices: tuple[int, ...]
    boundaries: tuple[BoundaryDiagnostic, ...]
    mismatch_summaries: tuple[MismatchSummary, ...]
    hard_failures: tuple[ReviewIssue, ...]
    warnings: tuple[ReviewIssue, ...]
    weighted_score: float
    score_deductions: Mapping[str, float]
    reference_profile: Mapping[str, float]

    @property
    def passed(self) -> bool:
        return not self.hard_failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "weighted_score": self.weighted_score,
            "score_deductions": dict(self.score_deductions),
            "segment_count": self.segment_count,
            "boundary_count": self.boundary_count,
            "internal_scene_cut_count": self.internal_scene_cut_count,
            "comparable_visual_boundary_count": (
                self.comparable_visual_boundary_count
            ),
            "visual_shot_count": len(self.visual_shot_durations_seconds),
            "visual_shot_durations_seconds": list(
                self.visual_shot_durations_seconds
            ),
            "visual_shot_median_seconds": self.visual_shot_median_seconds,
            "total_duration_seconds": self.total_duration_seconds,
            "source_switch_count": self.source_switch_count,
            "source_switch_rate": self.source_switch_rate,
            "source_switches_per_minute": self.source_switches_per_minute,
            "unique_source_count": self.unique_source_count,
            "available_source_count": self.available_source_count,
            "required_unique_source_count": self.required_unique_source_count,
            "source_duration_seconds": dict(self.source_duration_seconds),
            "per_source_share": dict(self.per_source_share),
            "source_runs": [run.to_dict() for run in self.source_runs],
            "source_run_median_seconds": self.source_run_median_seconds,
            "longest_source_run_seconds": self.longest_source_run_seconds,
            "ping_pong_count": len(self.ping_pongs),
            "ping_pongs": [item.to_dict() for item in self.ping_pongs],
            "short_flash_count": len(self.short_flash_run_indices),
            "short_flash_run_indices": list(self.short_flash_run_indices),
            "scene_cut_risk_count": len(self.scene_cut_risk_boundary_indices),
            "scene_cut_risk_boundary_indices": list(
                self.scene_cut_risk_boundary_indices
            ),
            "boundary_side_instability_count": (
                len(self.outgoing_boundary_side_instability_indices)
                + len(self.incoming_boundary_side_instability_indices)
            ),
            "outgoing_boundary_side_instability_indices": list(
                self.outgoing_boundary_side_instability_indices
            ),
            "incoming_boundary_side_instability_indices": list(
                self.incoming_boundary_side_instability_indices
            ),
            "boundaries": [boundary.to_dict() for boundary in self.boundaries],
            "mismatch_summaries": {
                summary.metric: summary.to_dict()
                for summary in self.mismatch_summaries
            },
            "hard_failures": [issue.to_dict() for issue in self.hard_failures],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "reference_profile": dict(self.reference_profile),
        }


@dataclass(frozen=True, slots=True)
class _ReviewSegment:
    original: Any = field(repr=False)
    segment_index: int
    source_id: str
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _first(value: Any, names: Sequence[str], default: Any = _MISSING) -> Any:
    for name in names:
        found = _get(value, name, _MISSING)
        if found is not _MISSING:
            return found
    if default is _MISSING:
        raise TypeError(f"timeline item is missing all of: {', '.join(names)}")
    return default


def _unwrap_choices(timeline: Any) -> tuple[Any, ...]:
    if isinstance(timeline, Mapping):
        choices = timeline.get("choices", _MISSING)
        if choices is _MISSING:
            # A JSON list is the unambiguous representation for bare choices;
            # a lone candidate mapping is more likely a caller bug.
            raise TypeError("timeline mapping must contain a choices array")
    elif hasattr(timeline, "choices"):
        choices = getattr(timeline, "choices")
    elif isinstance(timeline, Sequence) and not isinstance(
        timeline, (str, bytes, bytearray)
    ):
        choices = timeline
    else:
        raise TypeError(
            "timeline must be a choice sequence, TimelineResult, "
            "VisualTimelinePlan, or JSON-shaped mapping"
        )
    if not isinstance(choices, Sequence):
        choices = tuple(choices)
    unwrapped: list[Any] = []
    for choice in choices:
        candidate = _get(choice, "candidate", _MISSING)
        unwrapped.append(choice if candidate is _MISSING else candidate)
    return tuple(unwrapped)


def _adapt_segments(timeline: Any) -> tuple[_ReviewSegment, ...]:
    choices = _unwrap_choices(timeline)
    adapted: list[_ReviewSegment] = []
    for position, choice in enumerate(choices):
        source = _first(choice, ("source_id", "source_index"))
        start = float(
            _first(choice, ("canonical_master_start", "start_seconds"))
        )
        end = float(_first(choice, ("canonical_master_end", "end_seconds")))
        segment_index = int(_get(choice, "segment_index", position))
        adapted.append(
            _ReviewSegment(
                original=choice,
                segment_index=segment_index,
                source_id=str(source),
                start_seconds=start,
                end_seconds=end,
            )
        )
    return tuple(adapted)


def _as_numeric_tuple(value: Any) -> tuple[float, ...]:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return ()
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return ()
    return values if values and all(isfinite(item) for item in values) else ()


def _as_bbox(value: Any) -> tuple[float, float, float, float] | None:
    values = _as_numeric_tuple(value)
    if len(values) != 4 or values[2] <= 0.0 or values[3] <= 0.0:
        return None
    return (values[0], values[1], values[2], values[3])


def _as_centroid(value: Any) -> tuple[float, float] | None:
    values = _as_numeric_tuple(value)
    if len(values) != 2:
        return None
    return (values[0], values[1])


def _boundary_value(
    item: Any,
    *,
    boundary_name: str,
    fallback_name: str | None = None,
) -> Any:
    value = _get(item, boundary_name, _MISSING)
    if value is not _MISSING and value is not None:
        return value
    return _get(item, fallback_name, None) if fallback_name else None


def _sample_at(item: Any, field_name: str, position: int) -> Any:
    samples = _get(item, field_name, ())
    if not samples:
        return None
    try:
        return samples[position]
    except (IndexError, TypeError):
        return None


def _rms_mismatch(left: Any, right: Any) -> float | None:
    left_values = _as_numeric_tuple(left)
    right_values = _as_numeric_tuple(right)
    if not left_values or len(left_values) != len(right_values):
        return None
    squared = sum(
        (left_item - right_item) ** 2
        for left_item, right_item in zip(left_values, right_values, strict=True)
    )
    return min(1.0, sqrt(squared / len(left_values)))


def _bbox_mismatch(
    left: Any,
    right: Any,
    *,
    left_centroid: Any = None,
    right_centroid: Any = None,
) -> float | None:
    left_box = _as_bbox(left)
    right_box = _as_bbox(right)
    if left_box is None or right_box is None:
        return None
    left_center = _as_centroid(left_centroid) or (
        left_box[0] + left_box[2] / 2.0,
        left_box[1] + left_box[3] / 2.0,
    )
    right_center = _as_centroid(right_centroid) or (
        right_box[0] + right_box[2] / 2.0,
        right_box[1] + right_box[3] / 2.0,
    )
    center = min(
        1.0,
        sqrt(
            (left_center[0] - right_center[0]) ** 2
            + (left_center[1] - right_center[1]) ** 2
        )
        / 0.42,
    )
    left_area = max(left_box[2] * left_box[3], 1e-6)
    right_area = max(right_box[2] * right_box[3], 1e-6)
    scale = min(1.0, abs(log(left_area / right_area)) / log(3.2))
    width = min(1.0, abs(left_box[2] - right_box[2]) / 0.55)
    height = min(1.0, abs(left_box[3] - right_box[3]) / 0.55)
    return min(1.0, 0.52 * center + 0.28 * scale + 0.10 * width + 0.10 * height)


def _formation_people(value: Any) -> tuple[FormationPersonDescriptor, ...]:
    people: list[FormationPersonDescriptor] = []
    if not value:
        return ()
    for person in value:
        if isinstance(person, FormationPersonDescriptor):
            people.append(person)
            continue
        if not isinstance(person, Mapping):
            return ()
        try:
            people.append(
                FormationPersonDescriptor(
                    bbox_xywh=tuple(float(item) for item in person["bbox_xywh"]),
                    centroid_xy=tuple(float(item) for item in person["centroid_xy"]),
                    detection_confidence=float(person.get("detection_confidence", 1.0)),
                    upper_body_keypoints_xyc=tuple(
                        tuple(float(item) for item in point)
                        for point in person.get("upper_body_keypoints_xyc", ())
                    ),
                )
            )
        except (KeyError, TypeError, ValueError):
            return ()
    return tuple(people)


def _boundary_side_stability(value: Any) -> BoundarySideStability | None:
    if isinstance(value, BoundarySideStability):
        return value
    if not isinstance(value, Mapping):
        return None
    try:
        return BoundarySideStability(
            sample_span_seconds=float(value["sample_span_seconds"]),
            scale_delta=float(value["scale_delta"]),
            structure_mismatch=(
                float(value["structure_mismatch"])
                if value.get("structure_mismatch") is not None
                else None
            ),
            pose_mismatch=(
                float(value["pose_mismatch"])
                if value.get("pose_mismatch") is not None
                else None
            ),
            shot_type_mismatch=float(value.get("shot_type_mismatch", 0.0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _precomputed_mismatch(incoming: Any, name: str) -> float | None:
    value = _get(incoming, f"{name}_transition_mismatch_from_previous", None)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, numeric)) if isfinite(numeric) else None


def _authoritative_transition_mismatch(
    incoming: Any,
    name: str,
) -> tuple[bool, float | None]:
    """Read an exact joint-boundary scalar without legacy fallback.

    ``None`` is meaningful for the provider contract: it says the exact
    post-transform metric was unavailable while the provider's professional
    evidence-coverage gate still passed.  Treating that value like a missing
    field would reintroduce midpoint/HOG fallback evidence from another clock.
    """

    contract = _get(
        incoming,
        "transition_evidence_contract_from_previous",
        None,
    )
    authority = _get(
        incoming,
        "transition_evidence_authority_from_previous",
        None,
    )
    if authority is not JOINT_BOUNDARY_RESULT_EVIDENCE_AUTHORITY:
        return False, None
    if contract != JOINT_BOUNDARY_RESULT_EVIDENCE_CONTRACT:
        raise ValueError(
            "joint boundary authority is missing its exact evidence contract"
        )
    field_name = f"{name}_transition_mismatch_from_previous"
    raw = _get(incoming, field_name, _MISSING)
    if raw is _MISSING:
        raise ValueError(
            "joint boundary evidence contract omitted authoritative "
            f"{name} scalar"
        )
    if raw is None:
        return True, None
    try:
        numeric = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"joint boundary authoritative {name} scalar is invalid"
        ) from exc
    if not isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(
            f"joint boundary authoritative {name} scalar is outside [0, 1]"
        )
    return True, numeric


def _scale_mismatch(outgoing: Any, incoming: Any) -> float | None:
    authoritative, value = _authoritative_transition_mismatch(
        incoming,
        "scale",
    )
    if authoritative:
        return value
    left = _boundary_value(
        outgoing,
        boundary_name="exit_shot_scale",
        fallback_name="shot_scale",
    )
    right = _boundary_value(
        incoming,
        boundary_name="entry_shot_scale",
        fallback_name="shot_scale",
    )
    if left is None or right is None:
        return None
    try:
        mismatch = abs(float(left) - float(right))
    except (TypeError, ValueError):
        return None
    return min(1.0, mismatch) if isfinite(mismatch) else None


def _focus_mismatch(outgoing: Any, incoming: Any) -> float | None:
    authoritative, value = _authoritative_transition_mismatch(
        incoming,
        "focus",
    )
    if authoritative:
        return value
    left = _boundary_value(
        outgoing,
        boundary_name="exit_focus_bbox_xywh",
        fallback_name="focus_bbox_xywh",
    )
    right = _boundary_value(
        incoming,
        boundary_name="entry_focus_bbox_xywh",
        fallback_name="focus_bbox_xywh",
    )
    # VisualTimelineChoice persists the three sampled face boxes rather than
    # the candidate's transient focus box.  Exit and entry samples are the
    # correct boundary equivalents.
    if left is None:
        left = _sample_at(outgoing, "face_samples_xywh", -1)
    if right is None:
        right = _sample_at(incoming, "face_samples_xywh", 0)
    return _bbox_mismatch(left, right)


def _layout_mismatch(outgoing: Any, incoming: Any) -> float | None:
    authoritative, value = _authoritative_transition_mismatch(
        incoming,
        "layout",
    )
    if authoritative:
        return value
    precomputed = _precomputed_mismatch(incoming, "layout")
    if precomputed is not None:
        return precomputed
    left_embedding = _boundary_value(
        outgoing,
        boundary_name="exit_person_layout_embedding",
        fallback_name="person_layout_embedding",
    )
    right_embedding = _boundary_value(
        incoming,
        boundary_name="entry_person_layout_embedding",
        fallback_name="person_layout_embedding",
    )
    # TimelineCandidate keeps the same three silhouette samples as transient
    # midpoint/entry/exit subject boxes, while VisualTimelineChoice persists
    # them as ``silhouette_samples_xywh``.  Consult the boundary-specific
    # candidate fields first, then the serialized representation.  Without
    # this equivalence a missing person-layout embedding could pass candidate
    # selection and only become a hard layout failure after serialization.
    left_box = _boundary_value(
        outgoing,
        boundary_name="exit_subject_bbox_xywh",
        fallback_name="subject_bbox_xywh",
    )
    right_box = _boundary_value(
        incoming,
        boundary_name="entry_subject_bbox_xywh",
        fallback_name="subject_bbox_xywh",
    )
    left_centroid = _boundary_value(
        outgoing,
        boundary_name="exit_subject_centroid_xy",
        fallback_name="subject_centroid_xy",
    )
    right_centroid = _boundary_value(
        incoming,
        boundary_name="entry_subject_centroid_xy",
        fallback_name="subject_centroid_xy",
    )
    if left_box is None:
        left_box = _sample_at(outgoing, "silhouette_samples_xywh", -1)
        left_centroid = _sample_at(
            outgoing, "silhouette_centroid_samples_xy", -1
        )
    if right_box is None:
        right_box = _sample_at(incoming, "silhouette_samples_xywh", 0)
        right_centroid = _sample_at(
            incoming, "silhouette_centroid_samples_xy", 0
        )
    return screen_layout_mismatch(
        _as_numeric_tuple(left_embedding),
        _as_numeric_tuple(right_embedding),
        left_bbox=_as_bbox(left_box),
        right_bbox=_as_bbox(right_box),
        left_centroid=_as_centroid(left_centroid),
        right_centroid=_as_centroid(right_centroid),
    )


def _silhouette_mismatch(outgoing: Any, incoming: Any) -> float | None:
    authoritative, value = _authoritative_transition_mismatch(
        incoming,
        "silhouette",
    )
    if authoritative:
        return value
    precomputed = _precomputed_mismatch(incoming, "silhouette")
    if precomputed is not None:
        return precomputed
    left = _boundary_value(
        outgoing,
        boundary_name="exit_silhouette_embedding",
        fallback_name="silhouette_embedding",
    )
    right = _boundary_value(
        incoming,
        boundary_name="entry_silhouette_embedding",
        fallback_name="silhouette_embedding",
    )
    return _rms_mismatch(left, right)


def _formation_mismatch(outgoing: Any, incoming: Any) -> float | None:
    authoritative, value = _authoritative_transition_mismatch(
        incoming,
        "formation",
    )
    if authoritative:
        return value
    left_shot_type = _boundary_value(
        outgoing,
        boundary_name="exit_shot_type",
        fallback_name="shot_type",
    )
    right_shot_type = _boundary_value(
        incoming,
        boundary_name="entry_shot_type",
        fallback_name="shot_type",
    )
    try:
        comparable_shot_types = (
            ShotType(left_shot_type) is ShotType.GROUP_WIDE
            and ShotType(right_shot_type) is ShotType.GROUP_WIDE
        )
    except (TypeError, ValueError):
        comparable_shot_types = False
    if not comparable_shot_types:
        return None
    precomputed = _precomputed_mismatch(incoming, "formation")
    if precomputed is not None:
        return precomputed
    left = _formation_people(
        _boundary_value(
            outgoing,
            boundary_name="exit_formation_people",
            fallback_name="formation_people",
        )
    )
    right = _formation_people(
        _boundary_value(
            incoming,
            boundary_name="entry_formation_people",
            fallback_name="formation_people",
        )
    )
    return comparable_group_formation_mismatch(
        left,
        right,
        left_shot_type=left_shot_type,
        right_shot_type=right_shot_type,
    )


def _pose_mismatch(outgoing: Any, incoming: Any) -> float | None:
    authoritative, value = _authoritative_transition_mismatch(
        incoming,
        "pose",
    )
    if authoritative:
        return value
    precomputed = _precomputed_mismatch(incoming, "pose")
    if precomputed is not None:
        return precomputed
    left = _boundary_value(
        outgoing,
        boundary_name="exit_pose_embedding",
        fallback_name="pose_embedding",
    )
    right = _boundary_value(
        incoming,
        boundary_name="entry_pose_embedding",
        fallback_name="pose_embedding",
    )
    mismatch = _rms_mismatch(left, right)
    if mismatch is not None:
        return mismatch
    left = _boundary_value(
        outgoing,
        boundary_name="exit_structure_embedding",
        fallback_name="structure_embedding",
    )
    right = _boundary_value(
        incoming,
        boundary_name="entry_structure_embedding",
        fallback_name="structure_embedding",
    )
    return _rms_mismatch(left, right)


def _scene_cut_risk(outgoing: Any, incoming: Any) -> bool:
    return bool(
        _get(outgoing, "exit_scene_cut_near", False)
        or _get(incoming, "entry_scene_cut_near", False)
    )


def _thresholds(config: TimelineReviewConfig) -> dict[str, float]:
    return {
        "scale": config.maximum_scale_mismatch,
        "focus": config.maximum_focus_mismatch,
        "layout": config.maximum_layout_mismatch,
        "silhouette": config.maximum_silhouette_mismatch,
        "formation": config.maximum_formation_mismatch,
        "pose": config.maximum_pose_mismatch,
    }


def _build_boundaries(
    segments: Sequence[_ReviewSegment],
    config: TimelineReviewConfig,
) -> tuple[BoundaryDiagnostic, ...]:
    thresholds = _thresholds(config)
    boundaries: list[BoundaryDiagnostic] = []
    for boundary_index, (outgoing, incoming) in enumerate(
        zip(segments, segments[1:])
    ):
        source_switch = outgoing.source_id != incoming.source_id
        outgoing_shot_id = _get(outgoing.original, "shot_id", None)
        incoming_shot_id = _get(incoming.original, "shot_id", None)
        internal_cut_explains_native_change = bool(
            not source_switch
            and outgoing_shot_id
            and incoming_shot_id
            and outgoing_shot_id != incoming_shot_id
            and (
                _internal_scene_cut_times(outgoing)
                or _internal_scene_cut_times(incoming)
            )
        )
        # If shot IDs are present, equal IDs prove an invisible continuation.
        # Generic/legacy JSON has no such proof, so conservatively treat its
        # decision boundary as a comparable visual boundary. Exact detector
        # cuts inside either adjacent same-source choice already account for a
        # changed midpoint shot ID and must not be counted again at the rhythm
        # decision boundary.
        visual_boundary = bool(
            source_switch
            or not outgoing_shot_id
            or not incoming_shot_id
            or (
                outgoing_shot_id != incoming_shot_id
                and not internal_cut_explains_native_change
            )
        )
        metrics: dict[str, float | None] = {
            "scale": None,
            "focus": None,
            "layout": None,
            "silhouette": None,
            "formation": None,
            "pose": None,
        }
        scene_risk = False
        outgoing_stability: BoundarySideStability | None = None
        incoming_stability: BoundarySideStability | None = None
        outgoing_stability_exceeded: tuple[str, ...] = ()
        incoming_stability_exceeded: tuple[str, ...] = ()
        anchor_basis: str | None = None
        authorized_mismatches: tuple[str, ...] = ()
        if source_switch:
            metrics = {
                "scale": _scale_mismatch(outgoing.original, incoming.original),
                "focus": _focus_mismatch(outgoing.original, incoming.original),
                "layout": _layout_mismatch(outgoing.original, incoming.original),
                "silhouette": _silhouette_mismatch(
                    outgoing.original, incoming.original
                ),
                "formation": _formation_mismatch(
                    outgoing.original, incoming.original
                ),
                "pose": _pose_mismatch(outgoing.original, incoming.original),
            }
            scene_risk = _scene_cut_risk(outgoing.original, incoming.original)
            outgoing_stability = _boundary_side_stability(
                _get(
                    outgoing.original,
                    "exit_boundary_side_stability",
                    None,
                )
            )
            incoming_stability = _boundary_side_stability(
                _get(
                    incoming.original,
                    "entry_boundary_side_stability",
                    None,
                )
            )
            for evidence, outgoing_side in (
                (outgoing_stability, True),
                (incoming_stability, False),
            ):
                if evidence is None:
                    continue
                exceeded_side = boundary_side_stability_exceeded(
                    evidence,
                    maximum_scale_delta=(
                        config.maximum_boundary_side_scale_delta
                    ),
                    maximum_structure_mismatch=(
                        config.maximum_boundary_side_structure_mismatch
                    ),
                    maximum_pose_mismatch=(
                        config.maximum_boundary_side_pose_mismatch
                    ),
                    maximum_shot_type_mismatch=(
                        config.maximum_boundary_side_shot_type_mismatch
                    ),
                )
                if outgoing_side:
                    outgoing_stability_exceeded = exceeded_side
                else:
                    incoming_stability_exceeded = exceeded_side
            raw_anchor_basis = _get(
                incoming.original,
                "intentional_transition_anchor_basis",
                None,
            )
            anchor_basis = (
                raw_anchor_basis if isinstance(raw_anchor_basis, str) else None
            )
            production_gate_contract = all(
                abs(left - right) <= 1e-12
                for left, right in (
                    (
                        config.maximum_scale_mismatch,
                        PRODUCTION_MAX_SCALE_MISMATCH,
                    ),
                    (
                        config.maximum_focus_mismatch,
                        PRODUCTION_MAX_FOCUS_MISMATCH,
                    ),
                    (
                        config.maximum_layout_mismatch,
                        PRODUCTION_MAX_LAYOUT_MISMATCH,
                    ),
                    (
                        config.maximum_silhouette_mismatch,
                        PRODUCTION_MAX_SILHOUETTE_MISMATCH,
                    ),
                    (
                        config.maximum_formation_mismatch,
                        PRODUCTION_MAX_FORMATION_MISMATCH,
                    ),
                    (
                        config.maximum_pose_mismatch,
                        PRODUCTION_MAX_POSE_MISMATCH,
                    ),
                )
            )
            trusted_joint_boundary_evidence = (
                _get(
                    incoming.original,
                    "transition_evidence_authority_from_previous",
                    None,
                )
                is JOINT_BOUNDARY_RESULT_EVIDENCE_AUTHORITY
                and _get(
                    incoming.original,
                    "transition_evidence_contract_from_previous",
                    None,
                )
                == JOINT_BOUNDARY_RESULT_EVIDENCE_CONTRACT
            )
            if production_gate_contract and trusted_joint_boundary_evidence:
                authorized_mismatches = (
                    intentional_phrase_anchor_authorized_mismatches(
                        anchor_basis=anchor_basis,
                        scale_mismatch=metrics["scale"],
                        focus_mismatch=metrics["focus"],
                        layout_mismatch=metrics["layout"],
                        silhouette_mismatch=metrics["silhouette"],
                        formation_mismatch=metrics["formation"],
                        pose_mismatch=metrics["pose"],
                    )
                )
        exceeded = tuple(
            name
            for name, value in metrics.items()
            if value is not None
            and value > thresholds[name]
            and name not in authorized_mismatches
        )
        boundaries.append(
            BoundaryDiagnostic(
                boundary_index=boundary_index,
                outgoing_segment_index=outgoing.segment_index,
                incoming_segment_index=incoming.segment_index,
                at_seconds=outgoing.end_seconds,
                outgoing_source_id=outgoing.source_id,
                incoming_source_id=incoming.source_id,
                source_switch=source_switch,
                visual_boundary=visual_boundary,
                scene_cut_risk=scene_risk,
                outgoing_side_stability=outgoing_stability,
                incoming_side_stability=incoming_stability,
                outgoing_side_stability_exceeded_metrics=(
                    outgoing_stability_exceeded
                ),
                incoming_side_stability_exceeded_metrics=(
                    incoming_stability_exceeded
                ),
                scale_mismatch=metrics["scale"],
                focus_mismatch=metrics["focus"],
                layout_mismatch=metrics["layout"],
                silhouette_mismatch=metrics["silhouette"],
                formation_mismatch=metrics["formation"],
                pose_mismatch=metrics["pose"],
                intentional_transition_anchor_basis=anchor_basis,
                authorized_mismatch_metrics=authorized_mismatches,
                exceeded_metrics=exceeded,
            )
        )
    return tuple(boundaries)


def _build_runs(segments: Sequence[_ReviewSegment]) -> tuple[SourceRun, ...]:
    if not segments:
        return ()
    runs: list[SourceRun] = []
    source = segments[0].source_id
    first_position = 0
    start = segments[0].start_seconds
    end = segments[0].end_seconds
    duration = max(0.0, segments[0].duration_seconds)
    for position, segment in enumerate(segments[1:], start=1):
        if segment.source_id == source:
            end = segment.end_seconds
            duration += max(0.0, segment.duration_seconds)
            continue
        runs.append(
            SourceRun(
                run_index=len(runs),
                source_id=source,
                first_segment_position=first_position,
                last_segment_position=position - 1,
                start_seconds=start,
                end_seconds=end,
                duration_seconds=duration,
            )
        )
        source = segment.source_id
        first_position = position
        start = segment.start_seconds
        end = segment.end_seconds
        duration = max(0.0, segment.duration_seconds)
    runs.append(
        SourceRun(
            run_index=len(runs),
            source_id=source,
            first_segment_position=first_position,
            last_segment_position=len(segments) - 1,
            start_seconds=start,
            end_seconds=end,
            duration_seconds=duration,
        )
    )
    return tuple(runs)


def _median_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _internal_scene_cut_times(segment: _ReviewSegment) -> tuple[float, ...]:
    raw = _get(segment.original, "internal_scene_cut_master_seconds", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    values: set[float] = set()
    for item in raw:
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if (
            isfinite(value)
            and segment.start_seconds + 1e-6
            < value
            < segment.end_seconds - 1e-6
        ):
            values.add(value)
    return tuple(sorted(values))


def _build_visual_shot_durations(
    segments: Sequence[_ReviewSegment],
    boundaries: Sequence[BoundaryDiagnostic],
) -> tuple[float, ...]:
    """Aggregate actual screen time between visible shot boundaries.

    Exact detector cuts serialized inside a choice split it at their canonical
    master times. Rhythm decision spans that retain the same ``shot_id`` are
    otherwise folded into one shot. A source switch is always a visible
    boundary, as established by :func:`_build_boundaries`.
    """

    if not segments:
        return ()
    durations: list[float] = []
    accumulated = 0.0
    for position, segment in enumerate(segments):
        cursor = segment.start_seconds
        for cut_time in _internal_scene_cut_times(segment):
            accumulated += max(0.0, cut_time - cursor)
            if accumulated > 0.0:
                durations.append(accumulated)
            accumulated = 0.0
            cursor = cut_time
        accumulated += max(0.0, segment.end_seconds - cursor)
        if position < len(boundaries) and boundaries[position].visual_boundary:
            if accumulated > 0.0:
                durations.append(accumulated)
            accumulated = 0.0
    if accumulated > 0.0:
        durations.append(accumulated)
    return tuple(durations)


def _mismatch_summaries(
    boundaries: Sequence[BoundaryDiagnostic],
    config: TimelineReviewConfig,
) -> tuple[MismatchSummary, ...]:
    summaries: list[MismatchSummary] = []
    for metric, threshold in _thresholds(config).items():
        values = tuple(
            value
            for boundary in boundaries
            if boundary.source_switch
            for value in (getattr(boundary, f"{metric}_mismatch"),)
            if value is not None
        )
        summaries.append(
            MismatchSummary(
                metric=metric,
                threshold=threshold,
                available_count=len(values),
                mean=(sum(values) / len(values) if values else None),
                maximum=(max(values) if values else None),
                exceeded_count=sum(
                    metric in boundary.exceeded_metrics
                    for boundary in boundaries
                    if boundary.source_switch
                    and getattr(boundary, f"{metric}_mismatch") is not None
                ),
            )
        )
    return tuple(summaries)


def _source_run_deviation(run: SourceRun, config: TimelineReviewConfig) -> float:
    duration = run.duration_seconds
    if duration < config.preferred_min_source_run_seconds:
        return min(
            1.0,
            (config.preferred_min_source_run_seconds - duration)
            / config.preferred_min_source_run_seconds,
        )
    if duration > config.preferred_max_source_run_seconds:
        return min(
            1.0,
            (duration - config.preferred_max_source_run_seconds)
            / (
                config.hard_max_source_run_seconds
                - config.preferred_max_source_run_seconds
            ),
        )
    return 0.0


def review_timeline(
    timeline: Any,
    *,
    available_source_count: int | None = None,
    config: TimelineReviewConfig | None = None,
) -> TimelineReview:
    """Review a selected timeline without opening or rendering its sources."""

    review_config = config or TimelineReviewConfig()
    if available_source_count is not None and available_source_count < 1:
        raise ValueError("available_source_count must be at least one or None")
    segments = _adapt_segments(timeline)
    hard_failures: list[ReviewIssue] = []
    warnings: list[ReviewIssue] = []

    if not segments:
        hard_failures.append(
            ReviewIssue(
                code="empty_timeline",
                severity="hard",
                message="The selected timeline has no segments.",
            )
        )

    invalid_duration_positions = tuple(
        position
        for position, segment in enumerate(segments)
        if (
            not isfinite(segment.start_seconds)
            or not isfinite(segment.end_seconds)
            or segment.duration_seconds <= 0.0
        )
    )
    if invalid_duration_positions:
        hard_failures.append(
            ReviewIssue(
                code="non_positive_segment_duration",
                severity="hard",
                message="Every timeline segment must have a finite positive duration.",
                occurrences=invalid_duration_positions,
            )
        )

    non_monotonic = tuple(
        position
        for position, (left, right) in enumerate(zip(segments, segments[1:]))
        if right.start_seconds + review_config.continuity_tolerance_seconds
        < left.start_seconds
    )
    if non_monotonic:
        hard_failures.append(
            ReviewIssue(
                code="non_monotonic_timeline",
                severity="hard",
                message="Canonical segment starts are not chronological.",
                occurrences=non_monotonic,
            )
        )

    gaps = tuple(
        position
        for position, (left, right) in enumerate(zip(segments, segments[1:]))
        if right.start_seconds - left.end_seconds
        > review_config.continuity_tolerance_seconds
    )
    overlaps = tuple(
        position
        for position, (left, right) in enumerate(zip(segments, segments[1:]))
        if left.end_seconds - right.start_seconds
        > review_config.continuity_tolerance_seconds
    )
    if gaps:
        hard_failures.append(
            ReviewIssue(
                code="timeline_gap",
                severity="hard",
                message="The timeline contains an uncovered canonical-time gap.",
                occurrences=gaps,
            )
        )
    if overlaps:
        hard_failures.append(
            ReviewIssue(
                code="timeline_overlap",
                severity="hard",
                message="The timeline contains overlapping canonical segments.",
                occurrences=overlaps,
            )
        )

    valid_durations = tuple(max(0.0, item.duration_seconds) for item in segments)
    total_duration = sum(valid_durations)
    source_durations: dict[str, float] = {}
    for segment, duration in zip(segments, valid_durations, strict=True):
        source_durations[segment.source_id] = (
            source_durations.get(segment.source_id, 0.0) + duration
        )
    per_source_share = {
        source: duration / total_duration if total_duration > 0.0 else 0.0
        for source, duration in source_durations.items()
    }
    unique_source_count = len(source_durations)

    required_unique_sources = (
        review_config.minimum_sources_when_available
        if available_source_count is not None
        and available_source_count >= review_config.minimum_sources_when_available
        else None
    )
    if (
        required_unique_sources is not None
        and unique_source_count < required_unique_sources
    ):
        hard_failures.append(
            ReviewIssue(
                code="insufficient_source_diversity",
                severity="hard",
                message=(
                    f"Use at least {required_unique_sources} distinct sources when "
                    f"{available_source_count} are available."
                ),
                value=unique_source_count,
                threshold=required_unique_sources,
            )
        )

    dominant_share = max(per_source_share.values(), default=0.0)
    if dominant_share > review_config.dominant_source_share_hard:
        hard_failures.append(
            ReviewIssue(
                code="dominant_source_share_hard",
                severity="hard",
                message="One source occupies too much of the complete edit.",
                value=dominant_share,
                threshold=review_config.dominant_source_share_hard,
            )
        )
    elif dominant_share > review_config.dominant_source_share_warning:
        warnings.append(
            ReviewIssue(
                code="dominant_source_share",
                severity="warning",
                message="One source is overrepresented in the complete edit.",
                value=dominant_share,
                threshold=review_config.dominant_source_share_warning,
            )
        )

    runs = _build_runs(segments)
    source_run_median = _median_or_zero(
        tuple(run.duration_seconds for run in runs)
    )
    longest_run = max((run.duration_seconds for run in runs), default=0.0)
    if (
        runs
        and source_run_median
        > review_config.hard_max_source_run_median_seconds
    ):
        hard_failures.append(
            ReviewIssue(
                code="source_run_median_hard_max_exceeded",
                severity="hard",
                message=(
                    "The median continuous source run exceeds the production "
                    "cadence ceiling."
                ),
                value=source_run_median,
                threshold=review_config.hard_max_source_run_median_seconds,
            )
        )
    over_hard_max = tuple(
        run.run_index
        for run in runs
        if run.duration_seconds > review_config.hard_max_source_run_seconds
    )
    if over_hard_max:
        hard_failures.append(
            ReviewIssue(
                code="source_run_hard_max_exceeded",
                severity="hard",
                message=(
                    "A continuous source run exceeds the reference-p90 "
                    f"{review_config.hard_max_source_run_seconds:.2f}-second cap."
                ),
                occurrences=over_hard_max,
                value=longest_run,
                threshold=review_config.hard_max_source_run_seconds,
            )
        )
    soft_long_runs = tuple(
        run.run_index
        for run in runs
        if review_config.preferred_max_source_run_seconds
        < run.duration_seconds
        <= review_config.hard_max_source_run_seconds
    )
    if soft_long_runs:
        warnings.append(
            ReviewIssue(
                code="source_run_above_preferred",
                severity="warning",
                message="Some source runs exceed the preferred 2.5–5.5 second cadence.",
                occurrences=soft_long_runs,
                threshold=review_config.preferred_max_source_run_seconds,
            )
        )
    soft_short_runs = tuple(
        run.run_index
        for run in runs[1:-1]
        if review_config.short_flash_seconds
        <= run.duration_seconds
        < review_config.preferred_min_source_run_seconds
    )
    if soft_short_runs:
        warnings.append(
            ReviewIssue(
                code="source_run_below_preferred",
                severity="warning",
                message="Some internal source runs are shorter than the preferred 2.5 seconds.",
                occurrences=soft_short_runs,
                threshold=review_config.preferred_min_source_run_seconds,
            )
        )

    # First/last runs may be short solely because the requested song range was
    # trimmed.  Only an internal source burst can be an accidental flash.
    short_flash_runs = tuple(
        run.run_index
        for run in runs[1:-1]
        if run.duration_seconds < review_config.short_flash_seconds
    )
    if short_flash_runs:
        hard_failures.append(
            ReviewIssue(
                code="short_source_flash",
                severity="hard",
                message="An internal source appears too briefly to read as an intentional cut.",
                occurrences=short_flash_runs,
                threshold=review_config.short_flash_seconds,
            )
        )

    ping_pongs = tuple(
        PingPongDiagnostic(
            run_index=middle.run_index,
            source_sequence=(left.source_id, middle.source_id, right.source_id),
            middle_duration_seconds=middle.duration_seconds,
            within_review_window=(
                middle.duration_seconds <= review_config.ping_pong_middle_max_seconds
            ),
        )
        for left, middle, right in zip(runs, runs[1:], runs[2:])
        if left.source_id == right.source_id and left.source_id != middle.source_id
    )
    risky_ping_pongs = tuple(
        item.run_index for item in ping_pongs if item.within_review_window
    )
    if risky_ping_pongs:
        warnings.append(
            ReviewIssue(
                code="a_b_a_ping_pong",
                severity="warning",
                message="A→B→A source reversals can make the edit feel indecisive.",
                occurrences=risky_ping_pongs,
                threshold=review_config.ping_pong_middle_max_seconds,
            )
        )

    boundaries = _build_boundaries(segments, review_config)
    switch_count = sum(boundary.source_switch for boundary in boundaries)
    internal_scene_cut_count = sum(
        len(_internal_scene_cut_times(segment)) for segment in segments
    )
    comparable_visual_boundaries = (
        sum(boundary.visual_boundary for boundary in boundaries)
        + internal_scene_cut_count
    )
    visual_shot_durations = _build_visual_shot_durations(segments, boundaries)
    visual_shot_median = _median_or_zero(visual_shot_durations)
    if (
        visual_shot_durations
        and visual_shot_median
        > review_config.visual_shot_median_warning_seconds
    ):
        warnings.append(
            ReviewIssue(
                code="visual_shot_median_above_preferred",
                severity="warning",
                message=(
                    "The actual visual-shot median is slower than the "
                    "reference-calibrated production cadence."
                ),
                value=visual_shot_median,
                threshold=review_config.visual_shot_median_warning_seconds,
            )
        )
    switch_rate = (
        switch_count / comparable_visual_boundaries
        if comparable_visual_boundaries
        else 0.0
    )
    switches_per_minute = (
        switch_count * 60.0 / total_duration if total_duration > 0.0 else 0.0
    )
    if (
        total_duration > 0.0
        and switches_per_minute
        < review_config.hard_min_source_switches_per_minute
    ):
        hard_failures.append(
            ReviewIssue(
                code="source_switch_density_below_hard_minimum",
                severity="hard",
                message=(
                    "Absolute source-switch density is too low to read as a "
                    "production stage mix."
                ),
                value=switches_per_minute,
                threshold=review_config.hard_min_source_switches_per_minute,
            )
        )
    elif total_duration > 0.0 and not (
        review_config.preferred_min_source_switches_per_minute
        <= switches_per_minute
        <= review_config.preferred_max_source_switches_per_minute
    ):
        warnings.append(
            ReviewIssue(
                code="source_switch_density_outside_preferred_range",
                severity="warning",
                message=(
                    "Absolute source-switch density is outside the fan/KBS "
                    "reference band."
                ),
                value=switches_per_minute,
                threshold=(
                    review_config.preferred_min_source_switches_per_minute
                    if switches_per_minute
                    < review_config.preferred_min_source_switches_per_minute
                    else review_config.preferred_max_source_switches_per_minute
                ),
            )
        )
    if comparable_visual_boundaries and not (
        review_config.preferred_min_source_switch_ratio
        <= switch_rate
        <= review_config.preferred_max_source_switch_ratio
    ):
        warnings.append(
            ReviewIssue(
                code="source_switch_rate_outside_reference_band",
                severity="warning",
                message=(
                    "The source-switch ratio is outside the reference-calibrated "
                    "band around 49% of comparable visual boundaries."
                ),
                value=switch_rate,
                threshold=review_config.reference_source_switch_ratio,
            )
        )

    scene_risk_boundaries = tuple(
        boundary.boundary_index
        for boundary in boundaries
        if boundary.source_switch and boundary.scene_cut_risk
    )
    if scene_risk_boundaries:
        hard_failures.append(
            ReviewIssue(
                code="scene_cut_risk_boundary",
                severity="hard",
                message=(
                    "A source switch crosses a broadcaster camera cut on its visible side."
                ),
                occurrences=scene_risk_boundaries,
            )
        )

    outgoing_side_instability = tuple(
        boundary.boundary_index
        for boundary in boundaries
        if boundary.source_switch
        and boundary.outgoing_side_stability_exceeded_metrics
    )
    incoming_side_instability = tuple(
        boundary.boundary_index
        for boundary in boundaries
        if boundary.source_switch
        and boundary.incoming_side_stability_exceeded_metrics
    )
    if outgoing_side_instability:
        hard_failures.append(
            ReviewIssue(
                code="outgoing_boundary_side_instability",
                severity="hard",
                message=(
                    "The outgoing source changes camera or framing inside the "
                    "final three-frame boundary window."
                ),
                occurrences=outgoing_side_instability,
            )
        )
    if incoming_side_instability:
        hard_failures.append(
            ReviewIssue(
                code="incoming_boundary_side_instability",
                severity="hard",
                message=(
                    "The incoming source changes camera or framing inside the "
                    "first three-frame boundary window."
                ),
                occurrences=incoming_side_instability,
            )
        )

    mismatch_summaries = _mismatch_summaries(boundaries, review_config)
    for summary in mismatch_summaries:
        if not summary.exceeded_count:
            continue
        occurrences = tuple(
            boundary.boundary_index
            for boundary in boundaries
            if summary.metric in boundary.exceeded_metrics
        )
        hard_failures.append(
            ReviewIssue(
                code=f"{summary.metric}_mismatch_exceeded",
                severity="hard",
                message=(
                    f"{summary.metric.capitalize()} continuity exceeds the "
                    "production transition gate."
                ),
                occurrences=occurrences,
                value=summary.maximum,
                threshold=summary.threshold,
            )
        )

    deductions: dict[str, float] = {}
    if boundaries:
        deductions["switch_ratio"] = review_config.switch_ratio_weight * min(
            1.0,
            abs(switch_rate - review_config.reference_source_switch_ratio)
            / max(review_config.reference_source_switch_ratio, 1e-9),
        )
    else:
        deductions["switch_ratio"] = review_config.switch_ratio_weight
    if total_duration <= 0.0:
        density_deviation = 1.0
    elif (
        switches_per_minute
        < review_config.preferred_min_source_switches_per_minute
    ):
        density_deviation = min(
            1.0,
            (
                review_config.preferred_min_source_switches_per_minute
                - switches_per_minute
            )
            / max(
                1e-9,
                review_config.preferred_min_source_switches_per_minute
                - review_config.hard_min_source_switches_per_minute,
            ),
        )
    elif (
        switches_per_minute
        > review_config.preferred_max_source_switches_per_minute
    ):
        density_deviation = min(
            1.0,
            (
                switches_per_minute
                - review_config.preferred_max_source_switches_per_minute
            )
            / review_config.preferred_max_source_switches_per_minute,
        )
    else:
        density_deviation = 0.0
    deductions["source_switch_density"] = (
        review_config.switch_density_weight * density_deviation
    )
    visual_shot_warning_span = max(
        1e-9,
        review_config.visual_shot_median_warning_seconds
        - review_config.reference_visual_shot_median_seconds,
    )
    deductions["visual_shot_cadence"] = (
        review_config.visual_shot_cadence_weight
        * min(
            1.0,
            max(
                0.0,
                visual_shot_median
                - review_config.visual_shot_median_warning_seconds,
            )
            / visual_shot_warning_span,
        )
        if visual_shot_durations
        else review_config.visual_shot_cadence_weight
    )
    run_deviations = tuple(
        _source_run_deviation(run, review_config) for run in runs
    )
    deductions["source_run_cadence"] = (
        review_config.source_run_weight
        * sum(run_deviations)
        / len(run_deviations)
        if run_deviations
        else review_config.source_run_weight
    )
    source_run_median_reference = max(
        review_config.reference_kbs_source_run_median_seconds,
        review_config.reference_fan_source_run_median_seconds,
    )
    source_run_median_span = max(
        1e-9,
        review_config.hard_max_source_run_median_seconds
        - source_run_median_reference,
    )
    deductions["source_run_median"] = (
        review_config.source_run_median_weight
        * min(
            1.0,
            max(0.0, source_run_median - source_run_median_reference)
            / source_run_median_span,
        )
        if runs
        else review_config.source_run_median_weight
    )
    if required_unique_sources is not None:
        diversity_deficit = max(0, required_unique_sources - unique_source_count)
        deductions["source_diversity"] = review_config.diversity_weight * (
            diversity_deficit / required_unique_sources
        )
    else:
        deductions["source_diversity"] = 0.0
    dominance_span = max(
        1e-9,
        review_config.dominant_source_share_hard
        - review_config.dominant_source_share_warning,
    )
    deductions["source_dominance"] = review_config.dominance_weight * min(
        1.0,
        max(0.0, dominant_share - review_config.dominant_source_share_warning)
        / dominance_span,
    )
    deductions["ping_pong"] = review_config.ping_pong_weight * min(
        1.0,
        len(risky_ping_pongs) / max(1, len(runs) - 2),
    )
    deductions["short_flashes"] = review_config.short_flash_weight * min(
        1.0,
        len(short_flash_runs) / max(1, len(runs) - 2),
    )
    deductions["scene_cut_risk"] = review_config.scene_cut_risk_weight * min(
        1.0,
        len(scene_risk_boundaries) / max(1, switch_count),
    )
    deductions["boundary_side_stability"] = (
        review_config.boundary_side_stability_weight
        * min(
            1.0,
            (
                len(outgoing_side_instability)
                + len(incoming_side_instability)
            )
            / max(1, 2 * switch_count),
        )
    )
    summary_by_metric = {item.metric: item for item in mismatch_summaries}
    for metric, weight in review_config.mismatch_weights.items():
        summary = summary_by_metric[metric]
        deductions[f"{metric}_mismatch"] = (
            weight * summary.exceeded_count / summary.available_count
            if summary.available_count
            else 0.0
        )
    if invalid_duration_positions or gaps or overlaps or non_monotonic:
        deductions["structural_integrity"] = 25.0
    else:
        deductions["structural_integrity"] = 0.0
    if not segments:
        deductions["empty_timeline"] = 100.0

    weighted_score = max(0.0, 100.0 - sum(deductions.values()))
    reference_profile = {
        "visual_shot_median_seconds": (
            review_config.reference_visual_shot_median_seconds
        ),
        "visual_shot_median_warning_seconds": (
            review_config.visual_shot_median_warning_seconds
        ),
        "source_switch_ratio": review_config.reference_source_switch_ratio,
        "fan_source_switches_per_minute": (
            review_config.reference_fan_source_switches_per_minute
        ),
        "kbs_source_switches_per_minute": (
            review_config.reference_kbs_source_switches_per_minute
        ),
        "hard_min_source_switches_per_minute": (
            review_config.hard_min_source_switches_per_minute
        ),
        "preferred_min_source_switches_per_minute": (
            review_config.preferred_min_source_switches_per_minute
        ),
        "preferred_max_source_switches_per_minute": (
            review_config.preferred_max_source_switches_per_minute
        ),
        "preferred_source_run_min_seconds": (
            review_config.preferred_min_source_run_seconds
        ),
        "preferred_source_run_max_seconds": (
            review_config.preferred_max_source_run_seconds
        ),
        "kbs_source_run_median_seconds": (
            review_config.reference_kbs_source_run_median_seconds
        ),
        "fan_source_run_median_seconds": (
            review_config.reference_fan_source_run_median_seconds
        ),
        "hard_source_run_median_seconds": (
            review_config.hard_max_source_run_median_seconds
        ),
        "source_run_p90_seconds": review_config.reference_source_run_p90_seconds,
        "hard_source_run_max_seconds": review_config.hard_max_source_run_seconds,
        "boundary_side_max_scale_delta": (
            review_config.maximum_boundary_side_scale_delta
        ),
        "boundary_side_max_structure_mismatch": (
            review_config.maximum_boundary_side_structure_mismatch
        ),
        "boundary_side_max_pose_mismatch": (
            review_config.maximum_boundary_side_pose_mismatch
        ),
        "boundary_side_max_shot_type_mismatch": (
            review_config.maximum_boundary_side_shot_type_mismatch
        ),
    }
    return TimelineReview(
        segment_count=len(segments),
        boundary_count=len(boundaries),
        internal_scene_cut_count=internal_scene_cut_count,
        comparable_visual_boundary_count=comparable_visual_boundaries,
        visual_shot_durations_seconds=visual_shot_durations,
        visual_shot_median_seconds=visual_shot_median,
        total_duration_seconds=total_duration,
        source_switch_count=switch_count,
        source_switch_rate=switch_rate,
        source_switches_per_minute=switches_per_minute,
        unique_source_count=unique_source_count,
        available_source_count=available_source_count,
        required_unique_source_count=required_unique_sources,
        source_duration_seconds=source_durations,
        per_source_share=per_source_share,
        source_runs=runs,
        source_run_median_seconds=source_run_median,
        longest_source_run_seconds=longest_run,
        ping_pongs=ping_pongs,
        short_flash_run_indices=short_flash_runs,
        scene_cut_risk_boundary_indices=scene_risk_boundaries,
        outgoing_boundary_side_instability_indices=(
            outgoing_side_instability
        ),
        incoming_boundary_side_instability_indices=(
            incoming_side_instability
        ),
        boundaries=boundaries,
        mismatch_summaries=mismatch_summaries,
        hard_failures=tuple(hard_failures),
        warnings=tuple(warnings),
        weighted_score=weighted_score,
        score_deductions=deductions,
        reference_profile=reference_profile,
    )


__all__ = [
    "BoundaryDiagnostic",
    "JOINT_BOUNDARY_RESULT_EVIDENCE_CONTRACT",
    "MismatchSummary",
    "PingPongDiagnostic",
    "ReviewIssue",
    "SourceRun",
    "TimelineReview",
    "TimelineReviewConfig",
    "review_timeline",
]
