"""Fail-closed rendered-plan review for IVE stage-mix visual grammar.

The learned decoder owns source and timing decisions.  This module independently
checks the *selected* visual timeline and decoded proxy evidence so an edit made
almost entirely from formation wides cannot pass merely because its cuts are on
beat.  Formation wides remain desirable anchors; a long highlight must also
contain medium/close coverage and at least one short non-wide source burst.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence


RENDERED_MONTAGE_GRAMMAR_ANALYSIS_BASIS = (
    "rendered_selected_plan_exact_phrase_and_section_anchor_formation_"
    "highlight_closeup_and_wide_switch_soft_target_grammar_v3"
)
EXACT_STAGE_EDIT_SHOT_CLOCK_CONTRACT = (
    "serialized_inference_token_section_and_phrase_progress_per_selected_shot_v1"
)
_WIDE = frozenset({"wide", "full"})
_CLOSE = frozenset({"close"})
_NONWIDE = frozenset({"medium", "close"})
_SECTION_ROLES = frozenset(
    {"intro", "verse_like", "build", "chorus_like", "bridge_like", "outro"}
)
_ELIGIBLE_SOURCE_INDEX_DOMAIN = "eligible_source_index"
_REQUEST_SOURCE_INDEX_DOMAIN = "request_source_index"
_RENDER_PLAN_SOURCE_INDEX_DOMAIN = "render_plan_sources_array_index"
_RENDER_PLAN_BOUNDARY_INDEX_DOMAIN = "render_plan_transition_index"
_HARD_CUT = "hard_cut"
_FACE_DISSOLVE = "face_dissolve"
_EPSILON = 1.0e-6


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an array")
    return value


def _shot_class(value: Any) -> str:
    normalized = str(value or "unknown").strip().lower().replace("-", "_")
    aliases = {
        "group_wide": "wide",
        "long": "wide",
        "long_shot": "wide",
        "full_shot": "full",
        "full_body": "full",
        "medium_shot": "medium",
        "close_up": "close",
        "closeup": "close",
        "extreme_close_up": "close",
        "extreme_closeup": "close",
    }
    result = aliases.get(normalized, normalized)
    return result if result in {"wide", "full", "medium", "close"} else "unknown"


@dataclass(frozen=True, slots=True)
class RenderedMontageGrammarConfig:
    highlight_minimum_duration_seconds: float = 5.0
    minimum_known_shot_fraction: float = 0.80
    minimum_closeup_fraction: float = 0.15096153846153848
    minimum_nonwide_fraction: float = 0.35
    minimum_wide_involved_source_switch_fraction: float = (
        0.37333333333333335
    )
    preferred_wide_involved_source_switch_fraction: float = 0.80
    wide_involved_source_switch_reference_count: int = 17
    minimum_early_wide_anchor_fraction: float = 1.0 / 3.0
    minimum_phrase_entry_wide_anchor_fraction: float = 0.2785714285714286
    burst_maximum_run_seconds: float = 1.50
    minimum_nonwide_burst_boundaries: int = 1
    formation_anchor_run_window: int = 3
    minimum_formation_anchor_person_count: int = 1
    strong_formation_anchor_person_count: int = 4
    minimum_post_burst_formation_reset_count: int = 1
    minimum_motion_accent_count: int = 1
    minimum_closeup_proxy_face_coverage: float = 0.80
    minimum_closeup_proxy_face_height_fraction: float = 0.2184

    def __post_init__(self) -> None:
        for name in (
            "highlight_minimum_duration_seconds",
            "burst_maximum_run_seconds",
        ):
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "minimum_known_shot_fraction",
            "minimum_closeup_fraction",
            "minimum_nonwide_fraction",
            "minimum_wide_involved_source_switch_fraction",
            "preferred_wide_involved_source_switch_fraction",
            "minimum_early_wide_anchor_fraction",
            "minimum_phrase_entry_wide_anchor_fraction",
            "minimum_closeup_proxy_face_coverage",
            "minimum_closeup_proxy_face_height_fraction",
        ):
            number = _finite(getattr(self, name), name)
            if not 0.0 <= number <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in (
            "minimum_nonwide_burst_boundaries",
            "formation_anchor_run_window",
            "minimum_formation_anchor_person_count",
            "strong_formation_anchor_person_count",
            "minimum_post_burst_formation_reset_count",
            "minimum_motion_accent_count",
            "wide_involved_source_switch_reference_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.formation_anchor_run_window < 1:
            raise ValueError("formation_anchor_run_window must be positive")
        if self.minimum_formation_anchor_person_count < 1:
            raise ValueError(
                "minimum_formation_anchor_person_count must be at least one"
            )
        if (
            self.strong_formation_anchor_person_count
            < self.minimum_formation_anchor_person_count
        ):
            raise ValueError(
                "strong formation count cannot be below the weak count"
            )
        if self.wide_involved_source_switch_reference_count < 1:
            raise ValueError(
                "wide-switch preferred target needs at least one reference"
            )
        if self.minimum_closeup_fraction > self.minimum_nonwide_fraction:
            raise ValueError("close-up floor cannot exceed non-wide floor")
        if (
            self.preferred_wide_involved_source_switch_fraction + _EPSILON
            < self.minimum_wide_involved_source_switch_fraction
        ):
            raise ValueError(
                "preferred wide-switch target cannot be below its reference floor"
            )


def _count_evidence(value: Any, name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    samples = _sequence(value, name)
    if len(samples) != 3:
        raise ValueError(f"{name} must contain entry/middle/exit samples")
    result: list[int] = []
    for index, sample in enumerate(samples):
        if isinstance(sample, bool) or not isinstance(sample, int) or sample < 0:
            raise ValueError(f"{name}[{index}] must be a non-negative integer")
        result.append(sample)
    return tuple(result)


def _source_family_ids(
    analysis: Mapping[str, Any], choices: Sequence[Any]
) -> tuple[int, ...]:
    visual = _mapping(analysis.get("visual_timeline"), "visual_timeline")
    if visual.get("index_domain") != "eligible_source_index":
        raise ValueError("visual timeline source index domain is not explicit")
    diversity = _mapping(
        analysis.get("source_derivative_diversity"),
        "source_derivative_diversity",
    )
    if diversity.get("index_domain") != "eligible_source_index":
        raise ValueError("derivative family source index domain is not explicit")
    raw_ids = _sequence(
        diversity.get("family_ids"), "source_derivative_diversity.family_ids"
    )
    family_ids: list[int] = []
    for index, value in enumerate(raw_ids):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"derivative family id {index} is invalid")
        family_ids.append(value)
    source_count = diversity.get("source_count")
    if source_count != len(family_ids) or not family_ids:
        raise ValueError("derivative family source count is inconsistent")
    if (
        diversity.get("production_ready") is not True
        or diversity.get("clock_contract_complete") is not True
        or diversity.get("uncertain_pair_count") != 0
        or diversity.get("family_consistency_conflict_count") != 0
    ):
        raise ValueError("derivative family evidence is not production ready")
    for position, raw in enumerate(choices):
        choice = _mapping(raw, f"choices[{position}]")
        source_index = choice.get("source_index")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or not 0 <= source_index < len(family_ids)
        ):
            raise ValueError("visual choice source index is outside family metadata")
    return tuple(family_ids)


def _visible_runs(
    choices: Sequence[Any], family_ids: Sequence[int]
) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    previous_end: float | None = None
    for position, raw in enumerate(choices):
        choice = _mapping(raw, f"choices[{position}]")
        index = choice.get("segment_index", position)
        if isinstance(index, bool) or not isinstance(index, int) or index != position:
            raise ValueError("visual choices must use contiguous segment_index values")
        start = _finite(choice.get("canonical_master_start"), "choice start")
        end_value = choice.get("canonical_master_end")
        if end_value is None:
            end = start + _finite(choice.get("duration_seconds"), "choice duration")
        else:
            end = _finite(end_value, "choice end")
        if end <= start:
            raise ValueError("visual choice duration must be positive")
        if previous_end is not None and not math.isclose(
            previous_end, start, abs_tol=_EPSILON
        ):
            raise ValueError("visual choices must be contiguous")
        source_index = choice.get("source_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise ValueError("visual choice source_index must be an integer")
        shot_id = str(choice.get("shot_id") or "").strip() or None
        source_local_start = _finite(
            choice.get("source_local_start"), "choice source_local_start"
        )
        source_local_end = _finite(
            choice.get("source_local_end"), "choice source_local_end"
        )
        if source_local_end <= source_local_start:
            raise ValueError("visual choice source-local duration must be positive")
        detected_counts = _count_evidence(
            choice.get("detected_person_counts"),
            f"choices[{position}].detected_person_counts",
        )
        formation_counts = _count_evidence(
            choice.get("formation_person_counts"),
            f"choices[{position}].formation_person_counts",
        )
        formation_upper_counts = _count_evidence(
            choice.get("formation_upper_body_counts"),
            f"choices[{position}].formation_upper_body_counts",
        )
        normalized.append(
            {
                "first_segment_index": position,
                "last_segment_index": position,
                "source_index": source_index,
                "derivative_family_id": family_ids[source_index],
                "shot_id": shot_id,
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": end - start,
                "source_local_start": source_local_start,
                "source_local_end": source_local_end,
                "shot_class": _shot_class(choice.get("shot_type")),
                "detected_person_count_max": (
                    max(detected_counts) if detected_counts else None
                ),
                "formation_person_count_max": (
                    max(formation_counts) if formation_counts else None
                ),
                "formation_upper_body_count_max": (
                    max(formation_upper_counts)
                    if formation_upper_counts
                    else None
                ),
            }
        )
        previous_end = end

    runs: list[dict[str, Any]] = []
    for item in normalized:
        prior = runs[-1] if runs else None
        merge = bool(
            prior is not None
            and item["shot_id"] is not None
            and prior["shot_id"] == item["shot_id"]
            and prior["source_index"] == item["source_index"]
            and prior["derivative_family_id"] == item["derivative_family_id"]
            and prior["shot_class"] == item["shot_class"]
            and math.isclose(
                float(prior["end_seconds"]),
                float(item["start_seconds"]),
                abs_tol=_EPSILON,
            )
            and math.isclose(
                float(prior["source_local_end"]),
                float(item["source_local_start"]),
                abs_tol=_EPSILON,
            )
        )
        if merge:
            assert prior is not None
            prior["last_segment_index"] = item["last_segment_index"]
            prior["end_seconds"] = item["end_seconds"]
            prior["duration_seconds"] = (
                float(prior["end_seconds"]) - float(prior["start_seconds"])
            )
            prior["source_local_end"] = item["source_local_end"]
            for name in (
                "detected_person_count_max",
                "formation_person_count_max",
                "formation_upper_body_count_max",
            ):
                values = tuple(
                    value
                    for value in (prior.get(name), item.get(name))
                    if value is not None
                )
                prior[name] = max(values) if values else None
        else:
            runs.append(dict(item))
    return tuple(runs)


def _overlap(left: float, right: float, start: float, end: float) -> float:
    return max(0.0, min(right, end) - max(left, start))


def _transform_has_nonidentity(value: Any, name: str) -> bool:
    transform = _mapping(value, name)
    if transform.get("keyframes") is None:
        return False
    keyframes = _sequence(transform.get("keyframes"), f"{name}.keyframes")
    if not keyframes:
        return False
    nonidentity = False
    for index, raw in enumerate(keyframes):
        keyframe = _mapping(raw, f"{name}.keyframes[{index}]")
        zoom = _finite(keyframe.get("zoom"), "editorial zoom")
        center_x = _finite(keyframe.get("center_x"), "editorial center_x")
        center_y = _finite(keyframe.get("center_y"), "editorial center_y")
        roll = _finite(keyframe.get("roll_degrees"), "editorial roll")
        nonidentity = nonidentity or not (
            math.isclose(zoom, 1.0, abs_tol=_EPSILON)
            and math.isclose(center_x, 0.5, abs_tol=_EPSILON)
            and math.isclose(center_y, 0.5, abs_tol=_EPSILON)
            and math.isclose(roll, 0.0, abs_tol=_EPSILON)
        )
    return nonidentity


def _motion_accent_segment_indices(analysis: Mapping[str, Any]) -> set[int]:
    result: set[int] = set()
    editorial = analysis.get("editorial_motion")
    if isinstance(editorial, Mapping):
        segments = editorial.get("segments")
        if isinstance(segments, (list, tuple)):
            for raw in segments:
                if not isinstance(raw, Mapping) or raw.get("applied") is not True:
                    continue
                index = raw.get("segment_index")
                if (
                    isinstance(index, int)
                    and not isinstance(index, bool)
                    and _transform_has_nonidentity(
                        {"keyframes": raw.get("keyframes")},
                        f"editorial_motion.segments[{index}]",
                    )
                ):
                    result.add(index)
    render = analysis.get("render")
    if isinstance(render, Mapping):
        segments = render.get("segments")
        if isinstance(segments, (list, tuple)):
            for position, raw in enumerate(segments):
                if (
                    isinstance(raw, Mapping)
                    and raw.get("editorial_transform") is not None
                    and _transform_has_nonidentity(
                        raw.get("editorial_transform"),
                        f"render.segments[{position}].editorial_transform",
                    )
                ):
                    result.add(position)
    return result


def _geometry_proves_closeup(
    value: Any, config: RenderedMontageGrammarConfig
) -> bool:
    geometry = value if isinstance(value, Mapping) else {}
    bbox = geometry.get("face_bbox_xywh")
    face_count = geometry.get("face_count")
    if (
        isinstance(face_count, bool)
        or not isinstance(face_count, int)
        or face_count < 1
        or not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or not all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in bbox
        )
    ):
        return False
    x, y, width, height = tuple(float(item) for item in bbox)
    return bool(
        0.0 <= x <= 1.0
        and 0.0 <= y <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
        and x + width <= 1.000001
        and y + height <= 1.000001
        and height >= config.minimum_closeup_proxy_face_height_fraction
    )


@dataclass(frozen=True, slots=True)
class _BoundRenderBoundary:
    boundary_index: int
    cut_frame_index: int
    outgoing_source_index: int
    incoming_source_index: int
    transition_kind: str
    transition_frame_count: int
    sample_offsets: tuple[int, int, int, int]
    frame_indices: tuple[int, int, int, int]

    @property
    def incoming_endpoint_offset(self) -> int:
        return (
            0
            if self.transition_kind == _HARD_CUT
            else self.transition_frame_count
        )


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _source_index_translation(
    analysis: Mapping[str, Any],
    *,
    choices: Sequence[Any],
    plan: Mapping[str, Any],
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Translate compact eligible indices into RenderPlan-local indices.

    The production pipeline's ``RenderPlan.sources`` array is request-indexed,
    while the learned visual timeline uses a compact eligible-source domain.
    A removed source therefore creates a real numeric gap.  Refuse to infer the
    translation from coincidentally equal integers: the bidirectional mapping
    and its declared domain equivalence must be present and internally exact.
    """

    visual = _mapping(analysis.get("visual_timeline"), "visual_timeline")
    if visual.get("index_domain") != _ELIGIBLE_SOURCE_INDEX_DOMAIN:
        raise ValueError("visual timeline source index domain is not explicit")
    plan_domains = _mapping(
        plan.get("source_index_domains"), "render_plan.source_index_domains"
    )
    plan_source_domain = plan_domains.get("segments[].source_index")
    if plan_source_domain != _RENDER_PLAN_SOURCE_INDEX_DOMAIN:
        raise ValueError("render plan segment source index domain is not explicit")
    for path in ("sources[]", "reference_source"):
        if plan_domains.get(path) != _RENDER_PLAN_SOURCE_INDEX_DOMAIN:
            raise ValueError("render plan source index domains are inconsistent")

    plan_sources = _sequence(plan.get("sources"), "render_plan.sources")
    if not plan_sources:
        raise ValueError("render plan sources cannot be empty")
    mapping = _mapping(analysis.get("source_index_mapping"), "source_index_mapping")
    if mapping.get("eligible_index_domain") != _ELIGIBLE_SOURCE_INDEX_DOMAIN:
        raise ValueError("eligible source mapping domain is not explicit")
    if mapping.get("request_index_domain") != _REQUEST_SOURCE_INDEX_DOMAIN:
        raise ValueError("request source mapping domain is not explicit")
    if mapping.get("render_plan_index_domain") != plan_source_domain:
        raise ValueError("render plan mapping domain differs from the RenderPlan")
    equivalences = _mapping(
        mapping.get("domain_equivalences"), "source_index_mapping.domain_equivalences"
    )
    if equivalences.get(plan_source_domain) != _REQUEST_SOURCE_INDEX_DOMAIN:
        raise ValueError("render plan and request source domains are not equivalent")

    eligible_to_request_raw = _sequence(
        mapping.get("eligible_to_request"),
        "source_index_mapping.eligible_to_request",
    )
    request_to_eligible_raw = _sequence(
        mapping.get("request_to_eligible"),
        "source_index_mapping.request_to_eligible",
    )
    if len(request_to_eligible_raw) != len(plan_sources):
        raise ValueError(
            "request source mapping length differs from RenderPlan sources"
        )
    diversity = _mapping(
        analysis.get("source_derivative_diversity"),
        "source_derivative_diversity",
    )
    diversity_source_count = _nonnegative_integer(
        diversity.get("source_count"), "source_derivative_diversity.source_count"
    )
    if len(eligible_to_request_raw) != diversity_source_count:
        raise ValueError("eligible source mapping length differs from visual metadata")
    eligible_to_request: list[int] = []
    for eligible_index, raw_request_index in enumerate(eligible_to_request_raw):
        request_index = _nonnegative_integer(
            raw_request_index,
            f"eligible_to_request[{eligible_index}]",
        )
        if request_index >= len(plan_sources):
            raise ValueError(
                "eligible source mapping points outside RenderPlan sources"
            )
        eligible_to_request.append(request_index)
    if len(set(eligible_to_request)) != len(eligible_to_request):
        raise ValueError("eligible source mapping contains duplicate request indices")
    eligible_sources = _sequence(
        mapping.get("eligible_sources"), "source_index_mapping.eligible_sources"
    )
    if len(eligible_sources) != len(eligible_to_request):
        raise ValueError("eligible source identity count differs from source mapping")
    for eligible_index, raw_entry in enumerate(eligible_sources):
        entry = _mapping(
            raw_entry, f"source_index_mapping.eligible_sources[{eligible_index}]"
        )
        request_index = eligible_to_request[eligible_index]
        if (
            entry.get("eligible_source_index") != eligible_index
            or entry.get("request_source_index") != request_index
        ):
            raise ValueError("eligible source identity indices differ from mapping")
        mapped_source = str(entry.get("source") or "").strip()
        plan_source = str(plan_sources[request_index] or "").strip()
        if not mapped_source or mapped_source != plan_source:
            raise ValueError("eligible source identity differs from RenderPlan sources")
    for request_index, raw_eligible_index in enumerate(request_to_eligible_raw):
        if raw_eligible_index is None:
            if request_index in eligible_to_request:
                raise ValueError("request-to-eligible mapping omits an eligible source")
            continue
        eligible_index = _nonnegative_integer(
            raw_eligible_index,
            f"request_to_eligible[{request_index}]",
        )
        if (
            eligible_index >= len(eligible_to_request)
            or eligible_to_request[eligible_index] != request_index
        ):
            raise ValueError("source index mapping is not bidirectionally exact")

    payload_domains = _mapping(
        mapping.get("payload_index_domains"),
        "source_index_mapping.payload_index_domains",
    )
    required_payload_domains = {
        "visual_timeline.choices[].source_index": _ELIGIBLE_SOURCE_INDEX_DOMAIN,
        (
            "rendered_proxy_review.boundaries[].outgoing_source_index"
        ): _REQUEST_SOURCE_INDEX_DOMAIN,
        (
            "rendered_proxy_review.boundaries[].incoming_source_index"
        ): _REQUEST_SOURCE_INDEX_DOMAIN,
        "render_plan.segments[].source_index": plan_source_domain,
    }
    for path, expected_domain in required_payload_domains.items():
        if payload_domains.get(path) != expected_domain:
            raise ValueError(f"payload source index domain is missing for {path}")

    translated: list[int] = []
    for position, raw_choice in enumerate(choices):
        choice = _mapping(raw_choice, f"visual choice {position}")
        eligible_index = _nonnegative_integer(
            choice.get("source_index"),
            f"visual choice {position} source_index",
        )
        if eligible_index >= len(eligible_to_request):
            raise ValueError("visual choice source is outside eligible source mapping")
        translated.append(eligible_to_request[eligible_index])
    return (
        tuple(translated),
        {
            "visual_source_index_domain": _ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "render_plan_source_index_domain": plan_source_domain,
            "proxy_boundary_source_index_domain": _REQUEST_SOURCE_INDEX_DOMAIN,
            "eligible_to_request": list(eligible_to_request),
            "translated_choice_source_indices": list(translated),
            "mapping_verified_bidirectionally": True,
        },
    )


def _render_plan_binding(
    analysis: Mapping[str, Any],
    choices: Sequence[Any],
    render_plan: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], tuple[_BoundRenderBoundary, ...]]:
    if render_plan is None:
        raise ValueError("render_plan is required for rendered montage review")
    plan = _mapping(render_plan, "render_plan")
    segments = _sequence(plan.get("segments"), "render_plan.segments")
    transitions = _sequence(plan.get("transitions"), "render_plan.transitions")
    if len(segments) != len(choices):
        raise ValueError("render plan segment count does not match visual choices")
    if len(transitions) != max(0, len(segments) - 1):
        raise ValueError("render plan transition count does not match segments")
    translated_sources, source_binding = _source_index_translation(
        analysis,
        choices=choices,
        plan=plan,
    )
    master_audio_start = _finite(
        plan.get("master_audio_start"), "render_plan.master_audio_start"
    )
    fps = _finite(plan.get("fps"), "render_plan.fps")
    if fps <= 0.0:
        raise ValueError("render plan fps must be positive")

    cumulative_duration = 0.0
    previous_frame_boundary = 0
    frame_counts: list[int] = []
    for index, (raw_choice, raw_segment) in enumerate(
        zip(choices, segments, strict=True)
    ):
        choice = _mapping(raw_choice, f"visual choice {index}")
        segment = _mapping(raw_segment, f"render segment {index}")
        choice_start = _finite(
            choice.get("canonical_master_start"), "visual choice start"
        )
        choice_end = _finite(
            choice.get("canonical_master_end"), "visual choice end"
        )
        segment_start = _finite(segment.get("master_start"), "render segment start")
        segment_duration = _finite(segment.get("duration"), "render segment duration")
        if segment_duration <= 0.0:
            raise ValueError("render plan segment duration must be positive")
        segment_source = _nonnegative_integer(
            segment.get("source_index"), f"render segment {index} source_index"
        )
        if segment_source != translated_sources[index]:
            raise ValueError("render plan source differs from selected visual choice")
        if not math.isclose(
            segment_start,
            choice_start - master_audio_start,
            abs_tol=_EPSILON,
        ):
            raise ValueError("render plan start differs from selected visual choice")
        if not math.isclose(
            segment_duration,
            choice_end - choice_start,
            abs_tol=_EPSILON,
        ):
            raise ValueError("render plan duration differs from selected visual choice")
        cumulative_duration += segment_duration
        frame_boundary = int(round(cumulative_duration * fps))
        frame_count = frame_boundary - previous_frame_boundary
        if frame_count < 2:
            raise ValueError("render plan segment is shorter than two output frames")
        frame_counts.append(frame_count)
        previous_frame_boundary = frame_boundary

    expected_output_frame_count = sum(frame_counts)
    if (
        _nonnegative_integer(
            plan.get("output_frame_count"), "render_plan.output_frame_count"
        )
        != expected_output_frame_count
    ):
        raise ValueError("render plan output frame count is inconsistent")
    render_duration = _finite(
        plan.get("render_duration"), "render_plan.render_duration"
    )
    if not math.isclose(
        render_duration,
        expected_output_frame_count / fps,
        abs_tol=_EPSILON,
    ):
        raise ValueError("render plan render duration is inconsistent")

    bound_boundaries: list[_BoundRenderBoundary] = []
    cut_frame_index = 0
    for boundary_index, raw_transition in enumerate(transitions):
        cut_frame_index += frame_counts[boundary_index]
        transition = _mapping(
            raw_transition, f"render transition {boundary_index}"
        )
        transition_kind = str(transition.get("kind") or "")
        if transition_kind not in {_HARD_CUT, _FACE_DISSOLVE}:
            raise ValueError("render plan transition kind is unsupported")
        transition_frame_count = _nonnegative_integer(
            transition.get("frame_count"),
            f"render transition {boundary_index} frame_count",
        )
        if transition_kind == _HARD_CUT and transition_frame_count != 0:
            raise ValueError("hard-cut RenderPlan transition has overlap frames")
        if transition_kind == _FACE_DISSOLVE and transition_frame_count < 3:
            raise ValueError("dissolve RenderPlan transition is too short")
        if (
            transition_kind == _FACE_DISSOLVE
            and transition_frame_count + 2 > frame_counts[boundary_index + 1]
        ):
            raise ValueError(
                "dissolve RenderPlan transition lacks two completed endpoint frames"
            )
        sample_offsets = (
            (-2, -1, 0, 1)
            if transition_kind == _HARD_CUT
            else (
                -2,
                -1,
                transition_frame_count,
                transition_frame_count + 1,
            )
        )
        bound_boundaries.append(
            _BoundRenderBoundary(
                boundary_index=boundary_index,
                cut_frame_index=cut_frame_index,
                outgoing_source_index=translated_sources[boundary_index],
                incoming_source_index=translated_sources[boundary_index + 1],
                transition_kind=transition_kind,
                transition_frame_count=transition_frame_count,
                sample_offsets=sample_offsets,
                frame_indices=tuple(
                    cut_frame_index + offset for offset in sample_offsets
                ),
            )
        )
    return (
        {
            "required": True,
            "passed": True,
            "bound_segment_count": len(segments),
            "bound_transition_count": len(bound_boundaries),
            "fps": fps,
            "output_frame_count": expected_output_frame_count,
            "clock_mapping": (
                "canonical_master_minus_master_audio_start_to_output_master"
            ),
            "source_index_binding": source_binding,
            "boundary_index_domain": _RENDER_PLAN_BOUNDARY_INDEX_DOMAIN,
        },
        tuple(bound_boundaries),
    )


def _proxy_closeup_face_coverage(
    *,
    choices: Sequence[Any],
    rendered_proxy_review: Mapping[str, Any],
    bound_boundaries: Sequence[_BoundRenderBoundary],
    render_plan_binding: Mapping[str, Any],
    config: RenderedMontageGrammarConfig,
) -> dict[str, Any]:
    schema_version = _nonnegative_integer(
        rendered_proxy_review.get("schema_version"),
        "rendered proxy schema_version",
    )
    if schema_version < 3:
        raise ValueError(
            "rendered proxy schema does not carry strict boundary evidence"
        )
    expected_boundary_count = len(bound_boundaries)
    for field in ("transition_count", "reviewed_boundary_count"):
        if (
            _nonnegative_integer(
                rendered_proxy_review.get(field), f"rendered proxy {field}"
            )
            != expected_boundary_count
        ):
            raise ValueError(f"rendered proxy {field} differs from RenderPlan")
    hard_cut_count = sum(
        item.transition_kind == _HARD_CUT for item in bound_boundaries
    )
    dissolve_count = expected_boundary_count - hard_cut_count
    if (
        _nonnegative_integer(
            rendered_proxy_review.get("hard_cut_count"),
            "rendered proxy hard_cut_count",
        )
        != hard_cut_count
        or _nonnegative_integer(
            rendered_proxy_review.get("dissolve_count"),
            "rendered proxy dissolve_count",
        )
        != dissolve_count
    ):
        raise ValueError("rendered proxy transition-kind counts differ from RenderPlan")
    proxy_identity = _mapping(
        rendered_proxy_review.get("proxy"), "rendered_proxy_review.proxy"
    )
    if not math.isclose(
        _finite(proxy_identity.get("fps"), "rendered proxy fps"),
        _finite(render_plan_binding.get("fps"), "bound render plan fps"),
        abs_tol=_EPSILON,
    ):
        raise ValueError("rendered proxy fps differs from RenderPlan")
    if (
        _nonnegative_integer(
            proxy_identity.get("decoded_frame_count"),
            "rendered proxy decoded_frame_count",
        )
        != render_plan_binding.get("output_frame_count")
    ):
        raise ValueError("rendered proxy frame count differs from RenderPlan")

    boundaries = _sequence(
        rendered_proxy_review.get("boundaries"), "rendered proxy boundaries"
    )
    if len(boundaries) != expected_boundary_count:
        raise ValueError("rendered proxy boundary count does not match visual choices")
    boundary_records: list[Mapping[str, Any]] = []
    metric_records: list[dict[int, Mapping[str, Any]]] = []
    for expected, raw in zip(bound_boundaries, boundaries, strict=True):
        boundary_index = expected.boundary_index
        boundary = _mapping(raw, f"rendered proxy boundary {boundary_index}")
        if (
            _nonnegative_integer(
                boundary.get("boundary_index"),
                f"rendered proxy boundary {boundary_index} boundary_index",
            )
            != boundary_index
        ):
            raise ValueError("rendered proxy boundary indices must be contiguous")
        integer_fields = {
            "cut_frame_index": expected.cut_frame_index,
            "outgoing_source_index": expected.outgoing_source_index,
            "incoming_source_index": expected.incoming_source_index,
            "transition_frame_count": expected.transition_frame_count,
        }
        for field, expected_value in integer_fields.items():
            if (
                _nonnegative_integer(
                    boundary.get(field), f"boundary {boundary_index} {field}"
                )
                != expected_value
            ):
                raise ValueError(
                    f"rendered proxy boundary {field} differs from RenderPlan"
                )
        if boundary.get("transition_kind") != expected.transition_kind:
            raise ValueError(
                "rendered proxy boundary transition kind differs from RenderPlan"
            )
        raw_sample_offsets = _sequence(
            boundary.get("sample_offsets"),
            f"boundary {boundary_index} sample_offsets",
        )
        if tuple(raw_sample_offsets) != expected.sample_offsets or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_sample_offsets
        ):
            raise ValueError(
                "rendered proxy boundary sample offsets differ from RenderPlan"
            )
        raw_frame_indices = _sequence(
            boundary.get("frame_indices"),
            f"boundary {boundary_index} frame_indices",
        )
        if tuple(raw_frame_indices) != expected.frame_indices or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_frame_indices
        ):
            raise ValueError(
                "rendered proxy boundary frame indices differ from RenderPlan"
            )
        metrics = _sequence(
            boundary.get("frame_metrics"),
            f"boundary {boundary_index} frame_metrics",
        )
        if len(metrics) != len(expected.sample_offsets):
            raise ValueError("rendered proxy boundary frame metric count is incomplete")
        by_offset: dict[int, Mapping[str, Any]] = {}
        for metric_position, (
            raw_metric,
            expected_offset,
            expected_frame_index,
        ) in enumerate(
            zip(metrics, expected.sample_offsets, expected.frame_indices, strict=True)
        ):
            metric = _mapping(
                raw_metric,
                f"boundary {boundary_index} frame metric {metric_position}",
            )
            offset = metric.get("offset")
            frame_index = metric.get("frame_index")
            if (
                isinstance(offset, bool)
                or not isinstance(offset, int)
                or isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or offset != expected_offset
                or frame_index != expected_frame_index
            ):
                raise ValueError(
                    "rendered proxy frame metric is not bound to its exact sample"
                )
            by_offset[offset] = metric
        boundary_records.append(boundary)
        metric_records.append(by_offset)

    required = 0
    covered = 0
    records: list[dict[str, Any]] = []
    claimed_close_segments = 0
    for segment_index, raw_choice in enumerate(choices):
        choice = _mapping(raw_choice, f"visual choice {segment_index}")
        central_close = _shot_class(choice.get("shot_type")) == "close"
        entry_close = _shot_class(
            choice.get("entry_shot_type") or choice.get("shot_type")
        ) == "close"
        exit_close = _shot_class(
            choice.get("exit_shot_type") or choice.get("shot_type")
        ) == "close"
        if not (central_close or entry_close or exit_close):
            continue
        claimed_close_segments += 1
        endpoint_records: list[dict[str, Any]] = []
        incoming_offset = (
            bound_boundaries[segment_index - 1].incoming_endpoint_offset
            if segment_index > 0
            else None
        )
        endpoints = (
            (
                "incoming",
                segment_index - 1,
                incoming_offset,
                entry_close or central_close,
            ),
            ("outgoing", segment_index, -1, exit_close or central_close),
        )
        for side, boundary_index, offset, relevant in endpoints:
            if not relevant:
                continue
            if boundary_index < 0 or boundary_index >= len(boundary_records):
                continue
            if offset is None:
                raise ValueError("close-up endpoint offset could not be bound")
            required += 1
            metric = metric_records[boundary_index].get(offset)
            if metric is None:
                raise ValueError("close-up endpoint lacks its exact proxy sample")
            face = _geometry_proves_closeup(
                metric.get("subject_geometry"),
                config,
            )
            covered += int(face)
            endpoint_records.append(
                {
                    "side": side,
                    "boundary_index": boundary_index,
                    "offset": offset,
                    "frame_index": metric.get("frame_index"),
                    "transition_kind": bound_boundaries[
                        boundary_index
                    ].transition_kind,
                    "face_geometry_present": face,
                }
            )
        records.append(
            {"segment_index": segment_index, "endpoints": endpoint_records}
        )
    return {
        "claimed_closeup_segment_count": claimed_close_segments,
        "required_closeup_endpoint_count": required,
        "face_covered_closeup_endpoint_count": covered,
        "face_coverage_fraction": covered / required if required else None,
        "records": records,
    }


def _formation_people(item: Mapping[str, Any]) -> int | None:
    values = tuple(
        int(value)
        for value in (
            item.get("detected_person_count_max"),
            item.get("formation_person_count_max"),
        )
        if value is not None
    )
    return max(values) if values else None


def _proven_formation_wide(
    item: Mapping[str, Any], config: RenderedMontageGrammarConfig
) -> bool:
    formation_people = item.get("formation_person_count_max")
    upper_body_people = item.get("formation_upper_body_count_max")
    if item.get("shot_class") not in _WIDE or formation_people is None:
        return False
    count = int(formation_people)
    if count < config.minimum_formation_anchor_person_count:
        return False
    strong_multi_person_evidence = (
        count >= config.strong_formation_anchor_person_count
    )
    pose_backed_evidence = bool(
        upper_body_people is not None and int(upper_body_people) >= 1
    )
    return strong_multi_person_evidence or pose_backed_evidence


def _exact_stage_edit_anchor_metrics(
    root: Mapping[str, Any],
    *,
    choices: Sequence[Any],
    sections: Sequence[Any],
) -> dict[str, Any]:
    """Bind exact inference-token progress evidence to the rendered shot path."""

    payload = _mapping(
        root.get("stage_edit_exact_shot_clock"),
        "stage_edit_exact_shot_clock",
    )
    if payload.get("contract") != EXACT_STAGE_EDIT_SHOT_CLOCK_CONTRACT:
        raise ValueError("exact StageEdit shot clock contract changed")
    if payload.get("source_index_domain") != _ELIGIBLE_SOURCE_INDEX_DOMAIN:
        raise ValueError("exact StageEdit shot clock source domain changed")
    raw_shots = _sequence(payload.get("shots"), "exact StageEdit shot clocks")
    if len(raw_shots) != len(choices):
        raise ValueError("exact StageEdit shot clocks do not match visual choices")

    normalized_sections = tuple(
        _mapping(raw, f"section {index}") for index, raw in enumerate(sections)
    )

    def section_index_at(timestamp: float) -> int:
        matches = [
            index
            for index, section in enumerate(normalized_sections)
            if _finite(section.get("start_seconds"), "section start")
            - _EPSILON
            <= timestamp
            < _finite(section.get("end_seconds"), "section end") - _EPSILON
        ]
        if not matches and normalized_sections and math.isclose(
            timestamp,
            _finite(normalized_sections[-1].get("end_seconds"), "section end"),
            abs_tol=_EPSILON,
        ):
            return len(normalized_sections) - 1
        if len(matches) != 1:
            raise ValueError("exact StageEdit shot clock cannot bind to one section")
        return matches[0]

    records: list[dict[str, Any]] = []
    for position, (raw_shot, raw_choice) in enumerate(
        zip(raw_shots, choices, strict=True)
    ):
        shot = _mapping(raw_shot, f"exact StageEdit shot clock {position}")
        choice = _mapping(raw_choice, f"visual choice {position}")
        if shot.get("shot_index") != position:
            raise ValueError("exact StageEdit shot indices must be contiguous")
        source_index = choice.get("source_index")
        if shot.get("source_index") != source_index:
            raise ValueError("exact StageEdit shot source differs from visual choice")
        section_progress = _finite(
            shot.get("section_progress"), "exact section_progress"
        )
        phrase_progress = _finite(
            shot.get("phrase_progress"), "exact phrase_progress"
        )
        if not 0.0 <= section_progress <= 1.0:
            raise ValueError("exact section_progress lies outside [0, 1]")
        if not 0.0 <= phrase_progress <= 1.0:
            raise ValueError("exact phrase_progress lies outside [0, 1]")
        start = _finite(
            choice.get("canonical_master_start"), "choice master start"
        )
        section_index = section_index_at(start)
        section_role = str(shot.get("section_role") or "")
        planned_role = str(normalized_sections[section_index].get("role") or "")
        if section_role not in _SECTION_ROLES:
            raise ValueError("exact StageEdit shot clock section_role is invalid")
        if section_role != planned_role:
            raise ValueError(
                "exact StageEdit shot clock section_role differs from SectionPlan"
            )
        records.append(
            {
                "shot_index": position,
                "source_index": source_index,
                "section_index": section_index,
                "role": section_role,
                "shot_class": _shot_class(choice.get("shot_type")),
                "section_progress": section_progress,
                "phrase_progress": phrase_progress,
            }
        )

    for left, right in zip(records, records[1:]):
        if (
            left["section_index"] == right["section_index"]
            and right["section_progress"] + _EPSILON < left["section_progress"]
        ):
            raise ValueError("exact section_progress reset inside one section")

    observed_early = 0
    early_wide = 0
    exact_sections: list[list[dict[str, Any]]] = []
    for record in records:
        if not exact_sections:
            exact_sections.append([record])
        elif (
            record["role"] != exact_sections[-1][-1]["role"]
            or record["section_progress"] + _EPSILON
            < exact_sections[-1][-1]["section_progress"]
        ):
            exact_sections.append([record])
        else:
            exact_sections[-1].append(record)
    for exact_section in exact_sections:
        if exact_section[0]["role"] != "chorus_like":
            continue
        known = tuple(
            item
            for item in exact_section[:3]
            if item["shot_class"] != "unknown"
        )
        if known:
            observed_early += 1
            early_wide += int(any(item["shot_class"] in _WIDE for item in known))

    phrases: list[list[dict[str, Any]]] = []
    for record in records:
        if not phrases:
            phrases.append([record])
        elif (
            record["phrase_progress"] + _EPSILON
            < phrases[-1][-1]["phrase_progress"]
        ):
            phrases.append([record])
        else:
            phrases[-1].append(record)
    observed_phrase = 0
    phrase_wide = 0
    for phrase in phrases:
        entry = phrase[0]
        if entry["role"] != "chorus_like" or entry["shot_class"] == "unknown":
            continue
        observed_phrase += 1
        phrase_wide += int(entry["shot_class"] in _WIDE)

    return {
        "evidence_contract": EXACT_STAGE_EDIT_SHOT_CLOCK_CONTRACT,
        "shot_clock_count": len(records),
        "observed_chorus_section_early_anchor_count": observed_early,
        "chorus_section_early_wide_anchor_count": early_wide,
        "chorus_section_early_wide_anchor_fraction": (
            early_wide / observed_early if observed_early else None
        ),
        "observed_chorus_phrase_entry_anchor_count": observed_phrase,
        "chorus_phrase_entry_wide_anchor_count": phrase_wide,
        "chorus_phrase_entry_wide_anchor_fraction": (
            phrase_wide / observed_phrase if observed_phrase else None
        ),
    }


def review_rendered_montage_grammar(
    analysis: Mapping[str, Any],
    *,
    render_plan: Mapping[str, Any] | None = None,
    config: RenderedMontageGrammarConfig = RenderedMontageGrammarConfig(),
) -> dict[str, Any]:
    """Review the final selected-plan grammar and rendered close-up evidence."""

    root = _mapping(analysis, "analysis")
    section_plan = _mapping(root.get("section_plan"), "section_plan")
    sections = _sequence(section_plan.get("sections"), "section_plan.sections")
    visual = _mapping(root.get("visual_timeline"), "visual_timeline")
    choices = _sequence(visual.get("choices"), "visual_timeline.choices")
    if not choices:
        raise ValueError("rendered montage review needs visual choices")
    exact_anchor_metrics = _exact_stage_edit_anchor_metrics(
        root,
        choices=choices,
        sections=sections,
    )
    family_ids = _source_family_ids(root, choices)
    runs = _visible_runs(choices, family_ids)
    plan_binding, bound_boundaries = _render_plan_binding(
        root,
        choices,
        render_plan,
    )
    motion_segments = _motion_accent_segment_indices(root)

    timeline_start = float(runs[0]["start_seconds"])
    timeline_end = float(runs[-1]["end_seconds"])

    all_failures: list[str] = []
    section_metrics: list[dict[str, Any]] = []
    previous_end: float | None = None
    for position, raw in enumerate(sections):
        section = _mapping(raw, f"section {position}")
        if section.get("section_index") != position:
            raise ValueError("section indices must be contiguous")
        start = _finite(section.get("start_seconds"), "section start")
        end = _finite(section.get("end_seconds"), "section end")
        if end <= start:
            raise ValueError("section duration must be positive")
        if previous_end is not None and not math.isclose(
            previous_end, start, abs_tol=_EPSILON
        ):
            raise ValueError("section plan must be contiguous")
        if position == 0 and not math.isclose(start, timeline_start, abs_tol=_EPSILON):
            raise ValueError("section plan start does not match visual timeline")
        previous_end = end
        role = str(section.get("role") or "")
        if role not in _SECTION_ROLES:
            raise ValueError("section role is outside the editing-role schema")
        if not isinstance(section.get("establish_with_wide_shot"), bool):
            raise ValueError("section establish_with_wide_shot must be boolean")
        _finite(section.get("camera_motion_budget"), "camera motion budget")
        relevant_runs = [
            item
            for item in runs
            if _overlap(
                float(item["start_seconds"]),
                float(item["end_seconds"]),
                start,
                end,
            )
            > _EPSILON
        ]
        class_durations = {"wide": 0.0, "medium": 0.0, "close": 0.0, "unknown": 0.0}
        for item in relevant_runs:
            amount = _overlap(
                float(item["start_seconds"]),
                float(item["end_seconds"]),
                start,
                end,
            )
            key = "wide" if item["shot_class"] in _WIDE else str(item["shot_class"])
            class_durations[key] += amount
        covered = sum(class_durations.values())
        if not math.isclose(covered, end - start, abs_tol=_EPSILON):
            raise ValueError("visual timeline does not fully cover a section")
        known = covered - class_durations["unknown"]
        known_fraction = known / covered if covered else 0.0
        close_fraction = class_durations["close"] / known if known else 0.0
        nonwide_fraction = (
            class_durations["medium"] + class_durations["close"]
        ) / known if known else 0.0
        wide_fraction = class_durations["wide"] / known if known else 0.0
        burst_count = 0
        burst_right_positions: list[int] = []
        source_switch_count = 0
        wide_involved_source_switch_count = 0
        for run_position, (left, right) in enumerate(
            zip(relevant_runs, relevant_runs[1:])
        ):
            boundary = float(right["start_seconds"])
            if not (start - _EPSILON <= boundary < end - _EPSILON):
                continue
            if left["derivative_family_id"] == right["derivative_family_id"]:
                continue
            source_switch_count += 1
            if (
                left["shot_class"] in _WIDE
                or right["shot_class"] in _WIDE
            ):
                wide_involved_source_switch_count += 1
            if (
                left["shot_class"] in _NONWIDE
                and right["shot_class"] in _NONWIDE
                and (
                    left["shot_class"] in _CLOSE
                    or right["shot_class"] in _CLOSE
                )
                and float(left["duration_seconds"])
                <= config.burst_maximum_run_seconds
                and float(right["duration_seconds"])
                <= config.burst_maximum_run_seconds
            ):
                burst_count += 1
                burst_right_positions.append(run_position + 1)
        wide_involved_source_switch_fraction = (
            wide_involved_source_switch_count / source_switch_count
            if source_switch_count
            else 0.0
        )
        wide_switch_preferred_target_deviation = abs(
            wide_involved_source_switch_fraction
            - config.preferred_wide_involved_source_switch_fraction
        )
        anchor_runs = relevant_runs[: config.formation_anchor_run_window]
        generic_wide_anchor = any(
            item["shot_class"] in _WIDE for item in anchor_runs
        )
        formation_anchor = any(
            _proven_formation_wide(item, config) for item in anchor_runs
        )
        post_burst_formation_resets: set[int] = set()
        for burst_right in burst_right_positions:
            for candidate_position in range(burst_right + 1, len(relevant_runs)):
                candidate = relevant_runs[candidate_position]
                previous = relevant_runs[candidate_position - 1]
                if (
                    _proven_formation_wide(candidate, config)
                    and previous["shot_class"] in _NONWIDE
                    and previous["derivative_family_id"]
                    != candidate["derivative_family_id"]
                ):
                    post_burst_formation_resets.add(candidate_position)
                    break
        motion_count = sum(
            any(
                int(item["first_segment_index"]) <= segment_index
                <= int(item["last_segment_index"])
                for segment_index in motion_segments
            )
            for item in relevant_runs
        )
        strict_highlight = (
            role == "chorus_like"
            and end - start >= config.highlight_minimum_duration_seconds
        )
        failures: list[str] = []
        if strict_highlight:
            if known_fraction + _EPSILON < config.minimum_known_shot_fraction:
                failures.append("highlight_shot_type_evidence_incomplete")
            if close_fraction + _EPSILON < config.minimum_closeup_fraction:
                failures.append("highlight_closeup_fraction_below_floor")
            if nonwide_fraction + _EPSILON < config.minimum_nonwide_fraction:
                failures.append("highlight_nonwide_fraction_below_floor")
            if (
                source_switch_count > 0
                and wide_involved_source_switch_fraction + _EPSILON
                < config.minimum_wide_involved_source_switch_fraction
            ):
                failures.append(
                    "highlight_wide_involved_source_switch_fraction_below_"
                    "reference_floor"
                )
            if burst_count < config.minimum_nonwide_burst_boundaries:
                failures.append("highlight_nonwide_burst_missing")
            if (
                burst_count >= config.minimum_nonwide_burst_boundaries
                and len(post_burst_formation_resets)
                < config.minimum_post_burst_formation_reset_count
            ):
                failures.append("post_burst_formation_wide_reset_missing")
            if section.get("establish_with_wide_shot") is True:
                if not generic_wide_anchor:
                    failures.append("formation_wide_anchor_missing")
                elif not formation_anchor:
                    failures.append("formation_wide_anchor_evidence_missing")
            if (
                _finite(section.get("camera_motion_budget", 0.0), "camera motion budget")
                >= 0.50
                and motion_count < config.minimum_motion_accent_count
            ):
                failures.append("highlight_motion_accent_missing")
        all_failures.extend(f"section_{position}:{item}" for item in failures)
        section_metrics.append(
            {
                "section_index": position,
                "role": role,
                "duration_seconds": end - start,
                "strict_highlight_gate_applied": strict_highlight,
                "visible_run_count": len(relevant_runs),
                "visible_source_switch_count": source_switch_count,
                "wide_involved_source_switch_count": (
                    wide_involved_source_switch_count
                ),
                "wide_involved_source_switch_fraction": (
                    wide_involved_source_switch_fraction
                ),
                "wide_involved_source_switch_preferred_target": (
                    config.preferred_wide_involved_source_switch_fraction
                ),
                "wide_involved_source_switch_preferred_target_deviation": (
                    wide_switch_preferred_target_deviation
                ),
                "wide_involved_source_switch_preferred_target_shortfall": max(
                    0.0,
                    config.preferred_wide_involved_source_switch_fraction
                    - wide_involved_source_switch_fraction,
                ),
                "wide_involved_source_switch_preferred_target_is_hard_gate": (
                    False
                ),
                "wide_involved_source_switch_reference_count": (
                    config.wide_involved_source_switch_reference_count
                ),
                "known_duration_fraction": known_fraction,
                "wide_duration_fraction": wide_fraction,
                "nonwide_duration_fraction": nonwide_fraction,
                "closeup_duration_fraction": close_fraction,
                "nonwide_burst_boundary_count": burst_count,
                "post_burst_formation_wide_reset_count": len(
                    post_burst_formation_resets
                ),
                "generic_wide_anchor_present": generic_wide_anchor,
                "formation_wide_anchor_present": formation_anchor,
                "formation_anchor_people_max": max(
                    (
                        value
                        for item in anchor_runs
                        if (value := _formation_people(item)) is not None
                    ),
                    default=None,
                ),
                "motion_accent_run_count": motion_count,
                "failure_codes": failures,
                "passed": not failures,
            }
        )

    if not math.isclose(previous_end or 0.0, timeline_end, abs_tol=_EPSILON):
        raise ValueError("section plan end does not match visual timeline")

    strict_chorus_present = any(
        str(_mapping(raw, f"section {index}").get("role") or "")
        == "chorus_like"
        and _finite(
            _mapping(raw, f"section {index}").get("end_seconds"),
            "section end",
        )
        - _finite(
            _mapping(raw, f"section {index}").get("start_seconds"),
            "section start",
        )
        >= config.highlight_minimum_duration_seconds
        for index, raw in enumerate(sections)
    )
    if strict_chorus_present:
        early_fraction = exact_anchor_metrics[
            "chorus_section_early_wide_anchor_fraction"
        ]
        if early_fraction is None:
            all_failures.append("highlight_early_wide_anchor_evidence_missing")
        elif float(early_fraction) + _EPSILON < (
            config.minimum_early_wide_anchor_fraction
        ):
            all_failures.append(
                "highlight_early_wide_anchor_fraction_below_reference_floor"
            )
        phrase_fraction = exact_anchor_metrics[
            "chorus_phrase_entry_wide_anchor_fraction"
        ]
        if phrase_fraction is None:
            all_failures.append("highlight_phrase_entry_wide_anchor_evidence_missing")
        elif float(phrase_fraction) + _EPSILON < (
            config.minimum_phrase_entry_wide_anchor_fraction
        ):
            all_failures.append(
                "highlight_phrase_entry_wide_anchor_fraction_below_reference_floor"
            )

    proxy = _mapping(root.get("rendered_proxy_review"), "rendered_proxy_review")
    closeup_proxy = _proxy_closeup_face_coverage(
        choices=choices,
        rendered_proxy_review=proxy,
        bound_boundaries=bound_boundaries,
        render_plan_binding=plan_binding,
        config=config,
    )
    proxy_fraction = closeup_proxy["face_coverage_fraction"]
    if (
        int(closeup_proxy["claimed_closeup_segment_count"]) > 0
        and int(closeup_proxy["required_closeup_endpoint_count"]) == 0
    ):
        all_failures.append("rendered_closeup_proxy_evidence_missing")
    if (
        proxy_fraction is not None
        and float(proxy_fraction) + _EPSILON
        < config.minimum_closeup_proxy_face_coverage
    ):
        all_failures.append("rendered_closeup_face_coverage_below_floor")
    return {
        "schema_version": 1,
        "analysis_basis": RENDERED_MONTAGE_GRAMMAR_ANALYSIS_BASIS,
        "passed": not all_failures,
        "failure_codes": all_failures,
        "configuration": asdict(config),
        "visible_run_count": len(runs),
        "visible_runs": [dict(item) for item in runs],
        "render_plan_binding": plan_binding,
        "sections": section_metrics,
        "exact_anchor_metrics": exact_anchor_metrics,
        "rendered_closeup_proxy": closeup_proxy,
        "policy": {
            "formation_wide": (
                "identity_free_weak_person_plus_upper_body_or_strong_four_"
                "person_box_evidence_for_anchor_and_reset_never_full_group_"
                "proof"
            ),
            "highlight": (
                "reference_close_wide_transition_exact_section_and_phrase_"
                "anchor_floors_plus_reference_median_wide_switch_soft_target_"
                "plus_explicit_ive_product_nonwide_burst_requirement"
            ),
            "motion": "bounded_learned_or_evidence_safe_accent_not_random_quota",
            "pixel_proof": "plan_closeup_claim_requires_decoded_proxy_face_geometry",
        },
    }


__all__ = [
    "EXACT_STAGE_EDIT_SHOT_CLOCK_CONTRACT",
    "RENDERED_MONTAGE_GRAMMAR_ANALYSIS_BASIS",
    "RenderedMontageGrammarConfig",
    "review_rendered_montage_grammar",
]
