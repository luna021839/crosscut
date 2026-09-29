"""Exact, fail-closed adapter for professional group-boundary grammar.

The grammar assessment deliberately consumes only output-clock event evidence.
This adapter is the authority boundary between the production render objects and
that compact evidence: every clock is either recomputed from ``RenderPlan`` or
verified against an existing authoritative source-local clock payload.  Counts
are never expanded into synthetic timestamps.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .editorial_motion import EditorialMotionPlan
from .editorial_transform_events import rendered_transform_event_records
from .professional_montage_grammar import (
    EditorialTransformEvidence,
    IntroducedBoundaryContinuityEvidence,
    MontageGrammarEvidenceError,
    MontageSectionWindow,
    ProfessionalBoundaryGrammarEvidence,
)
from .render import (
    RENDER_PLAN_SOURCE_INDEX_DOMAIN,
    RenderPlan,
    segment_frame_counts,
)
from .rendered_proxy_review import (
    ANALYSIS_BASIS as RENDERED_PROXY_REVIEW_ANALYSIS_BASIS,
    INTRODUCED_CROSS_SOURCE_DOMAIN,
    NATIVE_SOURCE_CUT_DOMAIN,
    TIER_D_CONTINUITY_DOMAIN,
    RenderedProxyReview,
)
from .section_director import SectionPlan
from .stage_edit_dynamic_crop_motion_authority_v3 import (
    DEFAULT_MAXIMUM_MOTION_APPLIED_FRACTION,
    DEFAULT_MINIMUM_MOTION_APPLIED_FRACTION,
    DynamicCropMotionAuthorityV3Error,
    DynamicCropMotionTimelineEvidenceV3,
)
from .timeline_review import TimelineReview
from .visual_score import VisualTimelinePlan


_CLOCK_ABS_TOLERANCE = 1e-6
_SCALAR_ABS_TOLERANCE = 1e-9
_TRANSFORM_NEUTRAL_TOLERANCE = 1e-8
_PROXY_DOMAINS = (
    INTRODUCED_CROSS_SOURCE_DOMAIN,
    TIER_D_CONTINUITY_DOMAIN,
    NATIVE_SOURCE_CUT_DOMAIN,
)
_GENERAL_SECTION_ROLES = {
    "intro",
    "verse_like",
    "build",
    "bridge_like",
    "outro",
    "general",
}
_REQUEST_SOURCE_INDEX_DOMAIN = "request_source_index"
_ELIGIBLE_SOURCE_INDEX_DOMAIN = "eligible_source_index"


class GroupBoundaryGrammarAdapterError(MontageGrammarEvidenceError):
    """Raised when production payloads cannot prove exact grammar evidence."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GroupBoundaryGrammarAdapterError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise GroupBoundaryGrammarAdapterError(f"{name} must be an array")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GroupBoundaryGrammarAdapterError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise GroupBoundaryGrammarAdapterError(f"{name} must be finite")
    return number


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GroupBoundaryGrammarAdapterError(f"{name} must be an integer")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise GroupBoundaryGrammarAdapterError(f"{name} must be boolean")
    return value


def _required(value: Mapping[str, Any], key: str, name: str) -> Any:
    if key not in value:
        raise GroupBoundaryGrammarAdapterError(f"{name}.{key} is required")
    return value[key]


def _close(
    observed: object,
    expected: float,
    name: str,
    *,
    tolerance: float = _CLOCK_ABS_TOLERANCE,
) -> float:
    number = _finite(observed, name)
    if not math.isclose(number, expected, rel_tol=0.0, abs_tol=tolerance):
        raise GroupBoundaryGrammarAdapterError(
            f"{name} disagrees with the render authority: "
            f"observed={number!r}, expected={expected!r}"
        )
    return number


def _reject_inspection_only(value: object, name: str) -> None:
    """Reject explicit non-production markers without interpreting gate flags."""

    if isinstance(value, Mapping):
        if value.get("inspection_only") is True:
            raise GroupBoundaryGrammarAdapterError(
                f"{name} is inspection-only evidence"
            )
        for marker in ("analysis_mode", "payload_role", "evidence_role"):
            marker_value = value.get(marker)
            if (
                isinstance(marker_value, str)
                and marker_value.strip().lower().replace("-", "_")
                == "inspection_only"
            ):
                raise GroupBoundaryGrammarAdapterError(
                    f"{name} is inspection-only evidence"
                )
        for key, child in value.items():
            _reject_inspection_only(child, f"{name}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _reject_inspection_only(child, f"{name}[{index}]")


def _payload(value: object, name: str) -> Mapping[str, Any]:
    if isinstance(value, RenderedProxyReview):
        result = _mapping(value.payload, name)
    elif isinstance(value, (TimelineReview, SectionPlan, VisualTimelinePlan, EditorialMotionPlan)):
        result = _mapping(value.to_dict(), name)
    else:
        result = _mapping(value, name)
    _reject_inspection_only(result, name)
    return result


def _strict_clock_array(value: object, name: str) -> tuple[float, ...]:
    result = tuple(
        _finite(item, f"{name}[{index}]")
        for index, item in enumerate(_sequence(value, name))
    )
    if any(left >= right for left, right in zip(result, result[1:])):
        raise GroupBoundaryGrammarAdapterError(
            f"{name} must be strictly increasing and unique"
        )
    return result


def _same_clocks(
    observed: Sequence[float],
    expected: Sequence[float],
    name: str,
) -> None:
    if len(observed) != len(expected):
        raise GroupBoundaryGrammarAdapterError(
            f"{name} count mismatch: observed={len(observed)}, "
            f"expected={len(expected)}"
        )
    for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
        _close(left, right, f"{name}[{index}]")


def _source_switch_times(plan: RenderPlan) -> tuple[float, ...]:
    return tuple(
        incoming.master_start
        for outgoing, incoming in zip(plan.segments, plan.segments[1:])
        if outgoing.source_index != incoming.source_index
    )


def _validated_source_index_mapping(
    *,
    plan: RenderPlan,
    source_index_mapping: object,
) -> tuple[int, ...]:
    mapping = _mapping(source_index_mapping, "source_index_mapping")
    _reject_inspection_only(mapping, "source_index_mapping")
    expected_domains = {
        "request_index_domain": _REQUEST_SOURCE_INDEX_DOMAIN,
        "eligible_index_domain": _ELIGIBLE_SOURCE_INDEX_DOMAIN,
        "render_plan_index_domain": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
    }
    for key, expected in expected_domains.items():
        if str(_required(mapping, key, "source_index_mapping")) != expected:
            raise GroupBoundaryGrammarAdapterError(
                f"source_index_mapping.{key} is not the production domain"
            )
    equivalences = _mapping(
        _required(mapping, "domain_equivalences", "source_index_mapping"),
        "source_index_mapping.domain_equivalences",
    )
    if equivalences.get(RENDER_PLAN_SOURCE_INDEX_DOMAIN) != _REQUEST_SOURCE_INDEX_DOMAIN:
        raise GroupBoundaryGrammarAdapterError(
            "RenderPlan and request source domains are not explicitly equivalent"
        )
    eligible_raw = _sequence(
        _required(mapping, "eligible_to_request", "source_index_mapping"),
        "source_index_mapping.eligible_to_request",
    )
    if not eligible_raw:
        raise GroupBoundaryGrammarAdapterError("eligible source mapping is empty")
    eligible_to_request = tuple(
        _integer(item, f"eligible_to_request[{index}]")
        for index, item in enumerate(eligible_raw)
    )
    if any(item < 0 or item >= len(plan.sources) for item in eligible_to_request):
        raise GroupBoundaryGrammarAdapterError(
            "eligible source mapping points outside RenderPlan.sources"
        )
    if len(set(eligible_to_request)) != len(eligible_to_request):
        raise GroupBoundaryGrammarAdapterError(
            "eligible source mapping contains a duplicate request index"
        )
    request_raw = _sequence(
        _required(mapping, "request_to_eligible", "source_index_mapping"),
        "source_index_mapping.request_to_eligible",
    )
    if len(request_raw) != len(plan.sources):
        raise GroupBoundaryGrammarAdapterError(
            "request_to_eligible length differs from RenderPlan.sources"
        )
    for request_index, raw_eligible in enumerate(request_raw):
        if raw_eligible is None:
            if request_index in eligible_to_request:
                raise GroupBoundaryGrammarAdapterError(
                    "request_to_eligible omits an eligible request source"
                )
            continue
        eligible_index = _integer(
            raw_eligible, f"request_to_eligible[{request_index}]"
        )
        if (
            eligible_index < 0
            or eligible_index >= len(eligible_to_request)
            or eligible_to_request[eligible_index] != request_index
        ):
            raise GroupBoundaryGrammarAdapterError(
                "source index mapping is not bidirectionally exact"
            )
    entries = _sequence(
        _required(mapping, "eligible_sources", "source_index_mapping"),
        "source_index_mapping.eligible_sources",
    )
    if len(entries) != len(eligible_to_request):
        raise GroupBoundaryGrammarAdapterError(
            "eligible source identity count differs from the mapping"
        )
    for eligible_index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, f"eligible_sources[{eligible_index}]")
        request_index = eligible_to_request[eligible_index]
        if _integer(
            _required(entry, "eligible_source_index", f"eligible_sources[{eligible_index}]"),
            f"eligible_sources[{eligible_index}].eligible_source_index",
        ) != eligible_index or _integer(
            _required(entry, "request_source_index", f"eligible_sources[{eligible_index}]"),
            f"eligible_sources[{eligible_index}].request_source_index",
        ) != request_index:
            raise GroupBoundaryGrammarAdapterError(
                "eligible source identity indices disagree with the mapping"
            )
        if str(_required(entry, "source", f"eligible_sources[{eligible_index}]")) != str(
            plan.sources[request_index]
        ):
            raise GroupBoundaryGrammarAdapterError(
                "eligible source identity disagrees with RenderPlan.sources"
            )
    payload_domains = _mapping(
        _required(mapping, "payload_index_domains", "source_index_mapping"),
        "source_index_mapping.payload_index_domains",
    )
    required_payload_domains = {
        "scene_analysis[].source_index": _REQUEST_SOURCE_INDEX_DOMAIN,
        "visual_timeline.choices[].source_index": _ELIGIBLE_SOURCE_INDEX_DOMAIN,
        "editorial_motion.segments[].source_index": _ELIGIBLE_SOURCE_INDEX_DOMAIN,
        "render_plan.segments[].source_index": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
        (
            "rendered_proxy_review.boundaries[].outgoing_source_index"
        ): _REQUEST_SOURCE_INDEX_DOMAIN,
        (
            "rendered_proxy_review.boundaries[].incoming_source_index"
        ): _REQUEST_SOURCE_INDEX_DOMAIN,
    }
    for path, expected in required_payload_domains.items():
        if payload_domains.get(path) != expected:
            raise GroupBoundaryGrammarAdapterError(
                f"source index domain is missing or wrong for {path}"
            )
    used_request_indices = {segment.source_index for segment in plan.segments}
    if not used_request_indices.issubset(set(eligible_to_request)):
        raise GroupBoundaryGrammarAdapterError(
            "RenderPlan uses a request source excluded from the eligible mapping"
        )
    return eligible_to_request


def _validate_selected_timeline_and_map_native_cuts(
    *,
    plan: RenderPlan,
    selected_timeline: object,
    source_scene_cut_times: object,
    eligible_to_request: Sequence[int],
) -> tuple[tuple[float, ...], tuple[Mapping[str, Any], ...]]:
    timeline = _payload(selected_timeline, "selected_timeline")
    if not isinstance(selected_timeline, VisualTimelinePlan) and timeline.get(
        "index_domain"
    ) != _ELIGIBLE_SOURCE_INDEX_DOMAIN:
        raise GroupBoundaryGrammarAdapterError(
            "serialized selected_timeline index domain is missing or wrong"
        )
    choices = _sequence(
        _required(timeline, "choices", "selected_timeline"),
        "selected_timeline.choices",
    )
    if len(choices) != len(plan.segments):
        raise GroupBoundaryGrammarAdapterError(
            "selected_timeline must contain exactly one choice per render segment"
        )

    raw_by_source = _sequence(source_scene_cut_times, "source_scene_cut_times")
    if len(raw_by_source) != len(plan.sources):
        raise GroupBoundaryGrammarAdapterError(
            "source_scene_cut_times must contain exact clocks for every source"
        )
    source_clocks = tuple(
        _strict_clock_array(items, f"source_scene_cut_times[{source_index}]")
        for source_index, items in enumerate(raw_by_source)
    )

    mapped_output_clocks: list[float] = []
    validated_choices: list[Mapping[str, Any]] = []
    for index, (segment, raw_choice) in enumerate(
        zip(plan.segments, choices, strict=True)
    ):
        choice = _mapping(raw_choice, f"selected_timeline.choices[{index}]")
        validated_choices.append(choice)
        if _integer(
            _required(choice, "segment_index", f"choice[{index}]"),
            f"choice[{index}].segment_index",
        ) != index:
            raise GroupBoundaryGrammarAdapterError(
                "selected timeline segment indices are not exact and contiguous"
            )
        eligible_source_index = _integer(
            _required(choice, "source_index", f"choice[{index}]"),
            f"choice[{index}].source_index",
        )
        if (
            eligible_source_index < 0
            or eligible_source_index >= len(eligible_to_request)
            or eligible_to_request[eligible_source_index] != segment.source_index
        ):
            raise GroupBoundaryGrammarAdapterError(
                f"choice[{index}] source disagrees with RenderPlan"
            )
        canonical_start = plan.master_audio_start + segment.master_start
        canonical_end = canonical_start + segment.duration
        for key, expected in (
            ("canonical_master_start", canonical_start),
            ("canonical_master_end", canonical_end),
            ("source_local_start", segment.source_start),
            ("source_local_end", segment.source_end),
        ):
            _close(
                _required(choice, key, f"choice[{index}]"),
                expected,
                f"choice[{index}].{key}",
            )

        expected_canonical: list[float] = []
        for source_cut in source_clocks[segment.source_index]:
            if (
                source_cut > segment.source_start + _CLOCK_ABS_TOLERANCE
                and source_cut < segment.source_end - _CLOCK_ABS_TOLERANCE
            ):
                output_cut = segment.master_start + (
                    source_cut - segment.source_start
                ) / segment.source_playback_rate
                expected_canonical.append(plan.master_audio_start + output_cut)
                mapped_output_clocks.append(output_cut)
        serialized_canonical = _strict_clock_array(
            _required(
                choice,
                "internal_scene_cut_master_seconds",
                f"choice[{index}]",
            ),
            f"choice[{index}].internal_scene_cut_master_seconds",
        )
        _same_clocks(
            serialized_canonical,
            expected_canonical,
            f"choice[{index}] internal scene-cut clocks",
        )

    ordered = tuple(sorted(mapped_output_clocks))
    if any(
        math.isclose(left, right, rel_tol=0.0, abs_tol=_CLOCK_ABS_TOLERANCE)
        for left, right in zip(ordered, ordered[1:])
    ):
        raise GroupBoundaryGrammarAdapterError(
            "native source cuts collide after mapping to the output clock"
        )
    switches = _source_switch_times(plan)
    if any(
        math.isclose(cut, switch, rel_tol=0.0, abs_tol=_CLOCK_ABS_TOLERANCE)
        for cut in ordered
        for switch in switches
    ):
        raise GroupBoundaryGrammarAdapterError(
            "native source cut collides with a source-switch boundary"
        )
    return (ordered, tuple(validated_choices))


def _validate_timeline_review(
    *,
    plan: RenderPlan,
    timeline_review: object,
    native_cut_times: Sequence[float],
    eligible_to_request: Sequence[int],
    selected_choices: Sequence[Mapping[str, Any]],
) -> None:
    review = _payload(timeline_review, "timeline_review")
    _boolean(_required(review, "passed", "timeline_review"), "timeline_review.passed")
    segment_count = len(plan.segments)
    boundary_count = max(0, segment_count - 1)
    if _integer(
        _required(review, "segment_count", "timeline_review"),
        "timeline_review.segment_count",
    ) != segment_count:
        raise GroupBoundaryGrammarAdapterError("timeline segment_count mismatch")
    if _integer(
        _required(review, "boundary_count", "timeline_review"),
        "timeline_review.boundary_count",
    ) != boundary_count:
        raise GroupBoundaryGrammarAdapterError("timeline boundary_count mismatch")
    _close(
        _required(review, "total_duration_seconds", "timeline_review"),
        plan.duration,
        "timeline_review.total_duration_seconds",
    )
    boundaries = _sequence(
        _required(review, "boundaries", "timeline_review"),
        "timeline_review.boundaries",
    )
    if len(boundaries) != boundary_count:
        raise GroupBoundaryGrammarAdapterError(
            "timeline boundary records do not cover every render boundary"
        )

    visual_boundary_count = 0
    switch_count = 0
    for index, raw_boundary in enumerate(boundaries):
        boundary = _mapping(raw_boundary, f"timeline_review.boundaries[{index}]")
        outgoing = plan.segments[index]
        incoming = plan.segments[index + 1]
        expected_switch = outgoing.source_index != incoming.source_index
        outgoing_choice = selected_choices[index]
        incoming_choice = selected_choices[index + 1]
        if _integer(
            _required(boundary, "boundary_index", f"timeline boundary[{index}]"),
            f"timeline boundary[{index}].boundary_index",
        ) != index:
            raise GroupBoundaryGrammarAdapterError(
                "timeline boundary indices are not exact and contiguous"
            )
        for key, expected in (
            ("outgoing_segment_index", index),
            ("incoming_segment_index", index + 1),
        ):
            if _integer(
                _required(boundary, key, f"timeline boundary[{index}]"),
                f"timeline boundary[{index}].{key}",
            ) != expected:
                raise GroupBoundaryGrammarAdapterError(
                    f"timeline boundary[{index}] {key} mismatch"
                )
        _close(
            _required(boundary, "at_seconds", f"timeline boundary[{index}]"),
            plan.master_audio_start + incoming.master_start,
            f"timeline boundary[{index}].at_seconds",
        )
        for key, expected_request in (
            ("outgoing_source_id", outgoing.source_index),
            ("incoming_source_id", incoming.source_index),
        ):
            raw_source_id = str(
                _required(boundary, key, f"timeline boundary[{index}]")
            )
            try:
                eligible_index = int(raw_source_id)
            except ValueError as exc:
                raise GroupBoundaryGrammarAdapterError(
                    f"timeline boundary[{index}] {key} is not an eligible index"
                ) from exc
            if (
                str(eligible_index) != raw_source_id
                or eligible_index < 0
                or eligible_index >= len(eligible_to_request)
                or eligible_to_request[eligible_index] != expected_request
            ):
                raise GroupBoundaryGrammarAdapterError(
                    f"timeline boundary[{index}] {key} mismatch"
                )
        observed_switch = _boolean(
            _required(boundary, "source_switch", f"timeline boundary[{index}]"),
            f"timeline boundary[{index}].source_switch",
        )
        if observed_switch != expected_switch:
            raise GroupBoundaryGrammarAdapterError(
                f"timeline boundary[{index}] source-switch domain mismatch"
            )
        visual_boundary = _boolean(
            _required(boundary, "visual_boundary", f"timeline boundary[{index}]"),
            f"timeline boundary[{index}].visual_boundary",
        )
        if expected_switch and not visual_boundary:
            raise GroupBoundaryGrammarAdapterError(
                "a source switch cannot be marked as an invisible boundary"
            )
        outgoing_shot_id = str(
            _required(outgoing_choice, "shot_id", f"choice[{index}]") or ""
        )
        incoming_shot_id = str(
            _required(incoming_choice, "shot_id", f"choice[{index + 1}]") or ""
        )
        internal_cut_explains_native_change = bool(
            not expected_switch
            and outgoing_shot_id
            and incoming_shot_id
            and outgoing_shot_id != incoming_shot_id
            and (
                _sequence(
                    _required(
                        outgoing_choice,
                        "internal_scene_cut_master_seconds",
                        f"choice[{index}]",
                    ),
                    f"choice[{index}].internal_scene_cut_master_seconds",
                )
                or _sequence(
                    _required(
                        incoming_choice,
                        "internal_scene_cut_master_seconds",
                        f"choice[{index + 1}]",
                    ),
                    f"choice[{index + 1}].internal_scene_cut_master_seconds",
                )
            )
        )
        expected_visual_boundary = bool(
            expected_switch
            or not outgoing_shot_id
            or not incoming_shot_id
            or (
                outgoing_shot_id != incoming_shot_id
                and not internal_cut_explains_native_change
            )
        )
        if visual_boundary != expected_visual_boundary:
            raise GroupBoundaryGrammarAdapterError(
                f"timeline boundary[{index}] visual-boundary domain mismatch"
            )
        visual_boundary_count += int(visual_boundary)
        switch_count += int(expected_switch)

    internal_count = len(native_cut_times)
    comparable = visual_boundary_count + internal_count
    expected_values = (
        ("internal_scene_cut_count", internal_count),
        ("comparable_visual_boundary_count", comparable),
        ("visual_shot_count", comparable + 1),
        ("source_switch_count", switch_count),
        ("unique_source_count", len({item.source_index for item in plan.segments})),
    )
    for key, expected in expected_values:
        if _integer(
            _required(review, key, "timeline_review"),
            f"timeline_review.{key}",
        ) != expected:
            raise GroupBoundaryGrammarAdapterError(f"timeline {key} mismatch")
    switch_rate = switch_count / comparable if comparable else 0.0
    switches_per_minute = switch_count * 60.0 / plan.duration
    _close(
        _required(review, "source_switch_rate", "timeline_review"),
        switch_rate,
        "timeline_review.source_switch_rate",
        tolerance=_SCALAR_ABS_TOLERANCE,
    )
    _close(
        _required(review, "source_switches_per_minute", "timeline_review"),
        switches_per_minute,
        "timeline_review.source_switches_per_minute",
        tolerance=_SCALAR_ABS_TOLERANCE,
    )


def _section_kind(role: str) -> str:
    normalized = role.strip().lower()
    if normalized in _GENERAL_SECTION_ROLES:
        return "general"
    if normalized in {"chorus", "chorus_like"}:
        return "chorus"
    if normalized == "highlight":
        return "highlight"
    raise GroupBoundaryGrammarAdapterError(
        f"unrecognized section role cannot be inferred: {role!r}"
    )


def _validated_sections(
    *,
    plan: RenderPlan,
    section_plan: object,
) -> tuple[MontageSectionWindow, ...]:
    payload = _payload(section_plan, "section_plan")
    sections = _sequence(
        _required(payload, "sections", "section_plan"),
        "section_plan.sections",
    )
    if not sections:
        raise GroupBoundaryGrammarAdapterError("section plan is empty")
    if _integer(
        _required(payload, "section_count", "section_plan"),
        "section_plan.section_count",
    ) != len(sections):
        raise GroupBoundaryGrammarAdapterError("section_count mismatch")
    absolute_start = plan.master_audio_start
    absolute_end = absolute_start + plan.duration
    _close(
        _required(payload, "start_seconds", "section_plan"),
        absolute_start,
        "section_plan.start_seconds",
    )
    _close(
        _required(payload, "end_seconds", "section_plan"),
        absolute_end,
        "section_plan.end_seconds",
    )

    result: list[MontageSectionWindow] = []
    previous_end = absolute_start
    for index, raw_section in enumerate(sections):
        section = _mapping(raw_section, f"section_plan.sections[{index}]")
        if _integer(
            _required(section, "section_index", f"section[{index}]"),
            f"section[{index}].section_index",
        ) != index:
            raise GroupBoundaryGrammarAdapterError(
                "section indices are not exact and contiguous"
            )
        start = _finite(
            _required(section, "start_seconds", f"section[{index}]"),
            f"section[{index}].start_seconds",
        )
        end = _finite(
            _required(section, "end_seconds", f"section[{index}]"),
            f"section[{index}].end_seconds",
        )
        if not math.isclose(
            start,
            previous_end,
            rel_tol=0.0,
            abs_tol=_CLOCK_ABS_TOLERANCE,
        ):
            raise GroupBoundaryGrammarAdapterError(
                "section plan must be a complete contiguous partition"
            )
        if end <= start or end > absolute_end + _CLOCK_ABS_TOLERANCE:
            raise GroupBoundaryGrammarAdapterError("section span is invalid")
        output_start = 0.0 if index == 0 else start - absolute_start
        output_end = plan.duration if index == len(sections) - 1 else end - absolute_start
        result.append(
            MontageSectionWindow(
                section_id=f"section-{index}",
                kind=_section_kind(
                    str(_required(section, "role", f"section[{index}]"))
                ),
                start_seconds=output_start,
                end_seconds=output_end,
            )
        )
        previous_end = end
    if not math.isclose(
        previous_end,
        absolute_end,
        rel_tol=0.0,
        abs_tol=_CLOCK_ABS_TOLERANCE,
    ):
        raise GroupBoundaryGrammarAdapterError(
            "section plan does not end at the rendered output endpoint"
        )
    return tuple(result)


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _proxy_domain(plan: RenderPlan, boundary_index: int) -> str:
    outgoing = plan.segments[boundary_index]
    incoming = plan.segments[boundary_index + 1]
    transition = plan.transitions[boundary_index]
    if outgoing.source_index != incoming.source_index:
        return INTRODUCED_CROSS_SOURCE_DOMAIN
    if transition.tier == "D" or transition.reason == "same_source_hold_no_visible_switch":
        return TIER_D_CONTINUITY_DOMAIN
    return NATIVE_SOURCE_CUT_DOMAIN


def _validate_domain_summary(
    *,
    summary: object,
    domain: str,
    records: Sequence[tuple[int, float | None, bool]],
    review_policy: object,
) -> None:
    value = _mapping(summary, f"rendered_proxy_review.boundary_domains.{domain}")
    if str(_required(value, "domain", f"proxy domain {domain}")) != domain:
        raise GroupBoundaryGrammarAdapterError(f"proxy {domain} domain label mismatch")
    if str(_required(value, "count_basis", f"proxy domain {domain}")) != domain:
        raise GroupBoundaryGrammarAdapterError(f"proxy {domain} count basis mismatch")
    issues = [index for index, score, passed in records if not passed or score is None]
    if _integer(
        _required(value, "boundary_count", f"proxy domain {domain}"),
        f"proxy domain {domain}.boundary_count",
    ) != len(records):
        raise GroupBoundaryGrammarAdapterError(f"proxy {domain} boundary_count mismatch")
    if _integer(
        _required(value, "issue_count", f"proxy domain {domain}"),
        f"proxy domain {domain}.issue_count",
    ) != len(issues):
        raise GroupBoundaryGrammarAdapterError(f"proxy {domain} issue_count mismatch")
    expected_fraction = len(issues) / len(records) if records else 0.0
    _close(
        _required(value, "issue_fraction", f"proxy domain {domain}"),
        expected_fraction,
        f"proxy domain {domain}.issue_fraction",
        tolerance=_SCALAR_ABS_TOLERANCE,
    )
    raw_issue_indices = _sequence(
        _required(value, "issue_indices", f"proxy domain {domain}"),
        f"proxy domain {domain}.issue_indices",
    )
    observed_issue_indices = [
        _integer(item, f"proxy domain {domain}.issue_indices[{index}]")
        for index, item in enumerate(raw_issue_indices)
    ]
    if observed_issue_indices != issues:
        raise GroupBoundaryGrammarAdapterError(f"proxy {domain} issue indices mismatch")
    score = _mapping(
        _required(value, "score", f"proxy domain {domain}"),
        f"proxy domain {domain}.score",
    )
    distribution = _mapping(
        _required(score, "distribution", f"proxy domain {domain}.score"),
        f"proxy domain {domain}.score.distribution",
    )
    scores = [item[1] for item in records if item[1] is not None]
    expected_distribution: dict[str, float | int | None] = {
        "count": len(scores),
        "minimum": min(scores) if scores else None,
        "p10": _percentile(scores, 0.10),
        "median": _percentile(scores, 0.50),
        "p90": _percentile(scores, 0.90),
        "maximum": max(scores) if scores else None,
    }
    if set(distribution) != set(expected_distribution):
        raise GroupBoundaryGrammarAdapterError(
            f"proxy {domain} score distribution schema mismatch"
        )
    for key, expected in expected_distribution.items():
        observed = distribution[key]
        if expected is None:
            if observed is not None:
                raise GroupBoundaryGrammarAdapterError(
                    f"proxy {domain} score distribution {key} mismatch"
                )
        elif key == "count":
            if _integer(observed, f"proxy {domain} score distribution.count") != expected:
                raise GroupBoundaryGrammarAdapterError(
                    f"proxy {domain} score distribution count mismatch"
                )
        else:
            _close(
                observed,
                float(expected),
                f"proxy {domain} score distribution.{key}",
                tolerance=_SCALAR_ABS_TOLERANCE,
            )
    expected_p10 = _percentile(scores, 0.10)
    observed_p10 = _required(score, "p10", f"proxy domain {domain}.score")
    if expected_p10 is None:
        if observed_p10 is not None:
            raise GroupBoundaryGrammarAdapterError(f"proxy {domain} unknown score must be null")
    else:
        _close(observed_p10, expected_p10, f"proxy domain {domain}.score.p10",
               tolerance=_SCALAR_ABS_TOLERANCE)
    expected_status = ("NOT_APPLICABLE" if not records else
                       "OBSERVED" if len(scores) == len(records) else "INCOMPLETE")
    if score.get("evidence_status") != expected_status:
        raise GroupBoundaryGrammarAdapterError(f"proxy {domain} evidence status mismatch")
    from .proxy_review_contract import boundary_domain_passed
    try:
        expected_passed = boundary_domain_passed(domain=domain, score_p10=expected_p10,
            issue_fraction=expected_fraction, issue_count=len(issues), review_policy=review_policy)
    except ValueError as exc:
        raise GroupBoundaryGrammarAdapterError(str(exc)) from exc
    if _boolean(_required(value, "passed", f"proxy domain {domain}"),
                f"proxy domain {domain}.passed") is not expected_passed:
        raise GroupBoundaryGrammarAdapterError(f"proxy {domain} derived passed mismatch")
    expected_policy = (
        "editorial_boundary_p10_and_issue_fraction_gate"
        if domain == INTRODUCED_CROSS_SOURCE_DOMAIN
        else "zero_technical_integrity_issues"
    )
    if str(_required(value, "pass_policy", f"proxy domain {domain}")) != expected_policy:
        raise GroupBoundaryGrammarAdapterError(
            f"proxy {domain} pass policy mismatch"
        )


def _validated_proxy_continuity(
    *,
    plan: RenderPlan,
    rendered_proxy_review: object,
) -> IntroducedBoundaryContinuityEvidence:
    payload = _payload(rendered_proxy_review, "rendered_proxy_review")
    if _integer(
        _required(payload, "schema_version", "rendered_proxy_review"),
        "rendered_proxy_review.schema_version",
    ) != 3:
        raise GroupBoundaryGrammarAdapterError("unsupported proxy review schema")
    if (
        str(_required(payload, "analysis", "rendered_proxy_review"))
        != RENDERED_PROXY_REVIEW_ANALYSIS_BASIS
    ):
        raise GroupBoundaryGrammarAdapterError("unsupported proxy review analysis basis")
    records = _sequence(
        _required(payload, "boundaries", "rendered_proxy_review"),
        "rendered_proxy_review.boundaries",
    )
    boundary_count = max(0, len(plan.segments) - 1)
    for key in ("reviewed_boundary_count", "transition_count"):
        if _integer(
            _required(payload, key, "rendered_proxy_review"),
            f"rendered_proxy_review.{key}",
        ) != boundary_count:
            raise GroupBoundaryGrammarAdapterError(f"proxy {key} mismatch")
    if len(records) != boundary_count:
        raise GroupBoundaryGrammarAdapterError(
            "proxy boundary records do not cover every render boundary"
        )

    frame_counts = segment_frame_counts(plan)
    cumulative_frames = 0
    classified: dict[str, list[tuple[int, float | None, bool]]] = {
        domain: [] for domain in _PROXY_DOMAINS
    }
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"rendered_proxy_review.boundaries[{index}]")
        outgoing = plan.segments[index]
        incoming = plan.segments[index + 1]
        transition = plan.transitions[index]
        cumulative_frames += frame_counts[index]
        expected_domain = _proxy_domain(plan, index)
        expected_values = (
            ("boundary_index", index),
            ("cut_frame_index", cumulative_frames),
            ("outgoing_source_index", outgoing.source_index),
            ("incoming_source_index", incoming.source_index),
            ("transition_frame_count", transition.frame_count),
        )
        for key, expected in expected_values:
            if _integer(
                _required(record, key, f"proxy boundary[{index}]"),
                f"proxy boundary[{index}].{key}",
            ) != expected:
                raise GroupBoundaryGrammarAdapterError(
                    f"proxy boundary[{index}] {key} mismatch"
                )
        if str(_required(record, "review_domain", f"proxy boundary[{index}]")) != expected_domain:
            raise GroupBoundaryGrammarAdapterError(
                f"proxy boundary[{index}] review domain mismatch"
            )
        if (
            str(
                _required(
                    record,
                    "transition_reason",
                    f"proxy boundary[{index}]",
                )
            )
            != transition.reason
        ):
            raise GroupBoundaryGrammarAdapterError(
                f"proxy boundary[{index}] transition reason mismatch"
            )
        if _required(record, "transition_tier", f"proxy boundary[{index}]") != transition.tier:
            raise GroupBoundaryGrammarAdapterError(
                f"proxy boundary[{index}] transition tier mismatch"
            )
        if (
            str(
                _required(
                    record,
                    "transition_kind",
                    f"proxy boundary[{index}]",
                )
            )
            != transition.kind.value
        ):
            raise GroupBoundaryGrammarAdapterError(
                f"proxy boundary[{index}] transition kind mismatch"
            )
        _close(
            _required(record, "transition_duration_seconds", f"proxy boundary[{index}]"),
            transition.frame_count / plan.fps,
            f"proxy boundary[{index}].transition_duration_seconds",
        )
        expected_contiguous = bool(
            outgoing.source_index == incoming.source_index
            and abs(outgoing.source_end - incoming.source_start) <= 1e-4
        )
        if _boolean(
            _required(record, "source_clock_contiguous", f"proxy boundary[{index}]"),
            f"proxy boundary[{index}].source_clock_contiguous",
        ) != expected_contiguous:
            raise GroupBoundaryGrammarAdapterError(
                f"proxy boundary[{index}] source clock continuity mismatch"
            )
        passed = _boolean(
            _required(record, "passed", f"proxy boundary[{index}]"),
            f"proxy boundary[{index}].passed",
        )
        raw_score = _required(record, "weighted_score", f"proxy boundary[{index}]")
        score = None if raw_score is None else _finite(raw_score, f"proxy boundary[{index}].weighted_score")
        if score is None:
            coverage = _mapping(record.get("evidence_coverage"), f"proxy boundary[{index}].evidence_coverage")
            if (passed or expected_domain != INTRODUCED_CROSS_SOURCE_DOMAIN
                or coverage.get("contract") != "boundary_observation_coverage_v1"
                or coverage.get("status") != "UNKNOWN_SUBJECT_CONTINUITY"
                or coverage.get("subject_geometry_observed") is not False
                or "quality_score" not in coverage or coverage["quality_score"] is not None):
                raise GroupBoundaryGrammarAdapterError("proxy unknown boundary score lacks missing-evidence contract")
        elif not 0.0 <= score <= 100.0:
            raise GroupBoundaryGrammarAdapterError("proxy boundary score is outside [0, 100]")
        classified[expected_domain].append((index, score, passed))

    domain_payload = _mapping(
        _required(payload, "boundary_domains", "rendered_proxy_review"),
        "rendered_proxy_review.boundary_domains",
    )
    if set(domain_payload) != set(_PROXY_DOMAINS):
        raise GroupBoundaryGrammarAdapterError(
            "proxy boundary_domains must contain exactly the production domains"
        )
    top_count_keys = {
        INTRODUCED_CROSS_SOURCE_DOMAIN: "introduced_cross_source_count",
        TIER_D_CONTINUITY_DOMAIN: "tier_d_continuity_count",
        NATIVE_SOURCE_CUT_DOMAIN: "native_source_cut_count",
    }
    for domain in _PROXY_DOMAINS:
        _validate_domain_summary(
            summary=domain_payload[domain],
            domain=domain,
            records=classified[domain],
            review_policy=payload.get("review_policy"),
        )
        if _integer(
            _required(payload, top_count_keys[domain], "rendered_proxy_review"),
            f"rendered_proxy_review.{top_count_keys[domain]}",
        ) != len(classified[domain]):
            raise GroupBoundaryGrammarAdapterError(f"proxy {domain} top-level count mismatch")

    introduced = classified[INTRODUCED_CROSS_SOURCE_DOMAIN]
    issue_indices = [index for index, _score, passed in introduced if not passed]
    top_issue_indices = [
        _integer(item, f"introduced issue index[{index}]")
        for index, item in enumerate(
            _sequence(
                _required(
                    payload,
                    "introduced_cross_source_boundary_issue_indices",
                    "rendered_proxy_review",
                ),
                "introduced_cross_source_boundary_issue_indices",
            )
        )
    ]
    if top_issue_indices != issue_indices:
        raise GroupBoundaryGrammarAdapterError(
            "proxy introduced issue-index compatibility view is inconsistent"
        )
    p10 = _percentile([score for _index, score, _passed in introduced if score is not None], 0.10)
    return IntroducedBoundaryContinuityEvidence.from_counts(
        boundary_count=len(introduced),
        issue_count=len(issue_indices),
        score_p10=p10,
    )


def _track_keyframe_payload(track: object) -> tuple[Mapping[str, Any], ...]:
    keyframes = getattr(track, "keyframes", None)
    if keyframes is None:
        raise GroupBoundaryGrammarAdapterError("render transform track has no keyframes")
    return tuple(
        {
            "time_seconds": item.time_seconds,
            "zoom": item.zoom,
            "center_x": item.center_x,
            "center_y": item.center_y,
            "roll_degrees": item.roll_degrees,
        }
        for item in keyframes
    )


def _component_active(keyframe: Mapping[str, Any], component: str) -> bool:
    if component == "zoom":
        return abs(float(keyframe["zoom"]) - 1.0) > _TRANSFORM_NEUTRAL_TOLERANCE
    if component == "pan":
        return math.hypot(
            float(keyframe["center_x"]) - 0.5,
            float(keyframe["center_y"]) - 0.5,
        ) > _TRANSFORM_NEUTRAL_TOLERANCE
    return abs(float(keyframe["roll_degrees"])) > _TRANSFORM_NEUTRAL_TOLERANCE


def _component_value(
    keyframe: Mapping[str, Any], component: str
) -> tuple[float, ...]:
    if component == "zoom":
        return (float(keyframe["zoom"]),)
    if component == "pan":
        return (
            float(keyframe["center_x"]),
            float(keyframe["center_y"]),
        )
    return (float(keyframe["roll_degrees"]),)


def _different_component_values(
    left: Sequence[float] | None,
    right: Sequence[float],
) -> bool:
    return left is None or any(
        not math.isclose(
            observed,
            expected,
            rel_tol=0.0,
            abs_tol=_TRANSFORM_NEUTRAL_TOLERANCE,
        )
        for observed, expected in zip(left, right, strict=True)
    )


def _transform_event_records(
    plan: RenderPlan,
    component: str,
) -> tuple[tuple[float, int], ...]:
    """Return frame-visible component events and their owning segments.

    A Dynamic Crop v3 checkpoint can attach a safe constant framing track to
    every selected shot.  Track presence is therefore not evidence of visible
    editorial motion.  Conversely, a component may remain active while its
    value changes (for example zoom 1.02 -> 1.08); that is a real event and must
    not be lost merely because neither keyframe is neutral.

    The renderer samples each segment at ``frame_index / fps`` after globally
    quantizing segment frame counts.  Raw keyframe clocks are therefore not
    themselves proof of a visible event: a keyframe can sit at the segment
    endpoint (or after the last emitted frame), and a neutral-to-active linear
    ramp becomes visible on the first sampled frame after its neutral anchor,
    not only when the active target keyframe is reached.  Each authored active
    target is consequently represented by the first actual output sample on
    which its incoming trajectory is visibly different.  Sub-frame pulses that
    return before any emitted sample produce no event.
    """

    return rendered_transform_event_records(plan.segments, plan.fps)[component]


def _transform_event_times(plan: RenderPlan, component: str) -> tuple[float, ...]:
    return tuple(
        seconds
        for seconds, _segment_index in _transform_event_records(plan, component)
    )


def _validated_dynamic_crop_motion_v3(
    *,
    plan: RenderPlan,
    value: object,
) -> DynamicCropMotionTimelineEvidenceV3:
    """Strictly restore and bind the v3 motion authority to the full plan."""

    try:
        if isinstance(value, DynamicCropMotionTimelineEvidenceV3):
            # Reparse even a typed instance.  This rechecks every nested
            # content hash and the frozen 8--35% motion-budget verdict at the
            # production grammar boundary instead of trusting object identity.
            evidence = DynamicCropMotionTimelineEvidenceV3.from_dict(
                value.to_dict()
            )
        elif isinstance(value, Mapping):
            evidence = DynamicCropMotionTimelineEvidenceV3.from_dict(value)
        else:
            raise DynamicCropMotionAuthorityV3Error(
                "Dynamic Crop v3 motion evidence must be typed or serialized"
            )
    except (TypeError, ValueError) as exc:
        raise GroupBoundaryGrammarAdapterError(
            f"Dynamic Crop v3 motion evidence is invalid: {exc}"
        ) from exc

    # The authority module defines this digest over the complete canonical
    # RenderPlan payload, including paths, output identity, transforms and
    # transitions.  A partial timeline/segment rebind is not accepted here.
    from .stage_edit_dynamic_crop_motion_authority_v3 import _canonical_sha256

    expected_plan_sha256 = _canonical_sha256(plan.to_json())
    if evidence.render_plan_content_sha256 != expected_plan_sha256:
        raise GroupBoundaryGrammarAdapterError(
            "Dynamic Crop v3 motion evidence belongs to another RenderPlan"
        )
    if evidence.segment_count != len(plan.segments):
        raise GroupBoundaryGrammarAdapterError(
            "Dynamic Crop v3 motion evidence segment count differs from RenderPlan"
        )
    if (
        evidence.motion_authority_consistent is not True
        or evidence.renderer_motion_without_loso_authority_count != 0
        or evidence.motion_budget_within_range is not True
        or evidence.minimum_motion_applied_fraction
        != DEFAULT_MINIMUM_MOTION_APPLIED_FRACTION
        or evidence.maximum_motion_applied_fraction
        != DEFAULT_MAXIMUM_MOTION_APPLIED_FRACTION
    ):
        raise GroupBoundaryGrammarAdapterError(
            "Dynamic Crop v3 motion authority is not production-consistent"
        )
    return evidence


def _dynamic_crop_v3_transform_evidence(
    evidence: DynamicCropMotionTimelineEvidenceV3,
) -> EditorialTransformEvidence:
    """Use only LOSO-authorized, renderer-visible v3 component clocks."""

    events: dict[str, list[float]] = {
        "zoom": [],
        "pan": [],
        "roll": [],
    }
    eventful_segments: set[int] = set()
    component_names = {
        "zoom": "zoom",
        "center": "pan",
        "roll_degrees": "roll",
    }
    for segment in evidence.segment_evidence:
        if not segment.motion_event_applied:
            continue
        eventful_segments.add(segment.segment_index)
        for clock in segment.renderer_visible_first_motion_clocks:
            events[component_names[clock.component]].append(
                clock.master_time_seconds
            )

    normalized: dict[str, tuple[float, ...]] = {}
    for component, raw_times in events.items():
        ordered = tuple(sorted(raw_times))
        if any(
            left >= right
            for left, right in zip(ordered, ordered[1:])
        ):
            raise GroupBoundaryGrammarAdapterError(
                f"Dynamic Crop v3 {component} motion clocks are not unique"
            )
        normalized[component] = ordered
    return EditorialTransformEvidence(
        segment_count=evidence.segment_count,
        applied_segment_count=len(eventful_segments),
        zoom_event_times_seconds=normalized["zoom"],
        pan_event_times_seconds=normalized["pan"],
        roll_event_times_seconds=normalized["roll"],
    )


def _validated_transforms(
    *,
    plan: RenderPlan,
    editorial_motion: object,
    eligible_to_request: Sequence[int],
    dynamic_crop_version: int | None,
    dynamic_crop_motion_v3: object,
) -> EditorialTransformEvidence:
    if dynamic_crop_version is not None and (
        isinstance(dynamic_crop_version, bool)
        or not isinstance(dynamic_crop_version, int)
        or dynamic_crop_version not in {2, 3}
    ):
        raise GroupBoundaryGrammarAdapterError(
            "dynamic_crop_version must be absent, 2, or 3"
        )
    if dynamic_crop_version == 3:
        if dynamic_crop_motion_v3 is None:
            raise GroupBoundaryGrammarAdapterError(
                "Dynamic Crop v3 requires typed motion timeline evidence"
            )
        if editorial_motion is not None:
            raise GroupBoundaryGrammarAdapterError(
                "Dynamic Crop v3 cannot mix ordinary editorial-motion authority"
            )
        return _dynamic_crop_v3_transform_evidence(
            _validated_dynamic_crop_motion_v3(
                plan=plan,
                value=dynamic_crop_motion_v3,
            )
        )
    if dynamic_crop_motion_v3 is not None:
        raise GroupBoundaryGrammarAdapterError(
            "Dynamic Crop v3 motion evidence lacks its version marker"
        )

    attached_track_count = sum(
        segment.editorial_transform is not None for segment in plan.segments
    )
    payload: Mapping[str, Any] | None
    if editorial_motion is None:
        # StageEdit Dynamic Crop v2/v3 deliberately discards the ordinary
        # EditorialMotionPlan after copying its learned tracks onto the final
        # typed RenderPlan.  The actual RenderSegment tracks are the authority;
        # accepting None without at least one such track would instead turn a
        # missing payload into a silent zero-transform assessment.
        if not attached_track_count:
            raise GroupBoundaryGrammarAdapterError(
                "editorial_motion is unavailable and RenderPlan has no learned tracks"
            )
        payload = None
    else:
        payload = _payload(editorial_motion, "editorial_motion")
        segments = _sequence(
            _required(payload, "segments", "editorial_motion"),
            "editorial_motion.segments",
        )
        if len(segments) != len(plan.segments):
            raise GroupBoundaryGrammarAdapterError(
                "editorial motion must cover every render segment exactly once"
            )
        for index, (render_segment, raw_motion_segment) in enumerate(
            zip(plan.segments, segments, strict=True)
        ):
            motion_segment = _mapping(
                raw_motion_segment, f"editorial_motion.segments[{index}]"
            )
            if _integer(
                _required(motion_segment, "segment_index", f"motion segment[{index}]"),
                f"motion segment[{index}].segment_index",
            ) != index:
                raise GroupBoundaryGrammarAdapterError(
                    f"motion segment[{index}] segment_index mismatch"
                )
            eligible_index = _integer(
                _required(motion_segment, "source_index", f"motion segment[{index}]"),
                f"motion segment[{index}].source_index",
            )
            if (
                eligible_index < 0
                or eligible_index >= len(eligible_to_request)
                or eligible_to_request[eligible_index] != render_segment.source_index
            ):
                raise GroupBoundaryGrammarAdapterError(
                    f"motion segment[{index}] source_index mismatch"
                )
            expected_applied = render_segment.editorial_transform is not None
            if _boolean(
                _required(motion_segment, "applied", f"motion segment[{index}]"),
                f"motion segment[{index}].applied",
            ) != expected_applied:
                raise GroupBoundaryGrammarAdapterError(
                    f"motion segment[{index}] applied flag mismatch"
                )
            observed_keyframes = _sequence(
                _required(motion_segment, "keyframes", f"motion segment[{index}]"),
                f"motion segment[{index}].keyframes",
            )
            expected_keyframes = (
                _track_keyframe_payload(render_segment.editorial_transform)
                if render_segment.editorial_transform is not None
                else ()
            )
            if len(observed_keyframes) != len(expected_keyframes):
                raise GroupBoundaryGrammarAdapterError(
                    f"motion segment[{index}] keyframe count mismatch"
                )
            for keyframe_index, (raw_keyframe, expected_keyframe) in enumerate(
                zip(observed_keyframes, expected_keyframes, strict=True)
            ):
                keyframe = _mapping(
                    raw_keyframe,
                    f"motion segment[{index}].keyframes[{keyframe_index}]",
                )
                for key, expected in expected_keyframe.items():
                    _close(
                        _required(keyframe, key, f"motion keyframe[{index}:{keyframe_index}]"),
                        float(expected),
                        f"motion keyframe[{index}:{keyframe_index}].{key}",
                        tolerance=_SCALAR_ABS_TOLERANCE,
                    )
        if _integer(
            _required(payload, "applied_segment_count", "editorial_motion"),
            "editorial_motion.applied_segment_count",
        ) != attached_track_count:
            raise GroupBoundaryGrammarAdapterError("applied transform count mismatch")

    event_records = rendered_transform_event_records(plan.segments, plan.fps)
    events = {
        component: tuple(seconds for seconds, _index in records)
        for component, records in event_records.items()
    }
    eventful_segment_indices = {
        segment_index
        for records in event_records.values()
        for _seconds, segment_index in records
    }
    if payload is not None:
        for component in events:
            if _integer(
                _required(payload, f"{component}_event_count", "editorial_motion"),
                f"editorial_motion.{component}_event_count",
            ) != len(events[component]):
                raise GroupBoundaryGrammarAdapterError(
                    f"{component} event count disagrees with the actual keyframe tracks"
                )
    return EditorialTransformEvidence(
        segment_count=len(plan.segments),
        applied_segment_count=len(eventful_segment_indices),
        zoom_event_times_seconds=events["zoom"],
        pan_event_times_seconds=events["pan"],
        roll_event_times_seconds=events["roll"],
    )


def validate_group_boundary_grammar_plan(
    *, render_plan: RenderPlan, source_index_mapping: Mapping[str, Any],
    selected_timeline: VisualTimelinePlan | Mapping[str, Any],
    source_scene_cut_times: Sequence[Sequence[float]],
    timeline_review: TimelineReview | Mapping[str, Any],
    section_plan: SectionPlan | Mapping[str, Any],
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[MontageSectionWindow, ...]]:
    """Preflight static source/clock/review contracts before encoding pixels.

    The returned index map, native cut clocks and section windows are validated
    plan context only. They neither replace decoded QA nor approve quality.
    """
    if not isinstance(render_plan, RenderPlan):
        raise GroupBoundaryGrammarAdapterError("render_plan must be a RenderPlan")
    if not render_plan.segments:
        raise GroupBoundaryGrammarAdapterError("render_plan has no segments")
    if _finite(render_plan.duration, "render_plan.duration") <= 0.0:
        raise GroupBoundaryGrammarAdapterError("render duration must be positive")
    eligible_to_request = _validated_source_index_mapping(
        plan=render_plan, source_index_mapping=source_index_mapping,
    )
    native_cuts, selected_choices = _validate_selected_timeline_and_map_native_cuts(
        plan=render_plan, selected_timeline=selected_timeline,
        source_scene_cut_times=source_scene_cut_times,
        eligible_to_request=eligible_to_request,
    )
    _validate_timeline_review(
        plan=render_plan, timeline_review=timeline_review,
        native_cut_times=native_cuts, eligible_to_request=eligible_to_request,
        selected_choices=selected_choices,
    )
    sections = _validated_sections(plan=render_plan, section_plan=section_plan)
    return eligible_to_request, native_cuts, sections


def adapt_group_boundary_grammar_evidence(
    *,
    performance_id: str,
    render_plan: RenderPlan,
    source_index_mapping: Mapping[str, Any],
    selected_timeline: VisualTimelinePlan | Mapping[str, Any],
    source_scene_cut_times: Sequence[Sequence[float]],
    timeline_review: TimelineReview | Mapping[str, Any],
    section_plan: SectionPlan | Mapping[str, Any],
    rendered_proxy_review: RenderedProxyReview | Mapping[str, Any],
    editorial_motion: EditorialMotionPlan | Mapping[str, Any] | None,
    dynamic_crop_version: int | None = None,
    dynamic_crop_motion_v3: DynamicCropMotionTimelineEvidenceV3
    | Mapping[str, Any]
    | None = None,
) -> ProfessionalBoundaryGrammarEvidence:
    """Convert production evidence without estimating a missing event clock.

    ``source_scene_cut_times`` is intentionally required in the full request /
    RenderPlan source-array domain in addition to the eligible-domain selected
    timeline.  ``source_index_mapping`` proves that translation.  The former is
    the source-local detector authority; the latter proves the exact canonical
    clock chosen by the production timeline.  A serialized count, either
    payload alone, or any disagreement is rejected.
    """

    eligible_to_request, native_cuts, sections = validate_group_boundary_grammar_plan(
        render_plan=render_plan,
        source_index_mapping=source_index_mapping,
        selected_timeline=selected_timeline,
        source_scene_cut_times=source_scene_cut_times,
        timeline_review=timeline_review,
        section_plan=section_plan,
    )
    continuity = _validated_proxy_continuity(
        plan=render_plan,
        rendered_proxy_review=rendered_proxy_review,
    )
    transforms = _validated_transforms(
        plan=render_plan,
        editorial_motion=editorial_motion,
        eligible_to_request=eligible_to_request,
        dynamic_crop_version=dynamic_crop_version,
        dynamic_crop_motion_v3=dynamic_crop_motion_v3,
    )
    return ProfessionalBoundaryGrammarEvidence(
        performance_id=performance_id,
        duration_seconds=render_plan.duration,
        source_switch_times_seconds=_source_switch_times(render_plan),
        native_internal_cut_times_seconds=native_cuts,
        sections=sections,
        introduced_boundary_continuity=continuity,
        editorial_transforms=transforms,
    )


__all__ = [
    "GroupBoundaryGrammarAdapterError",
    "adapt_group_boundary_grammar_evidence",
    "validate_group_boundary_grammar_plan",
]
