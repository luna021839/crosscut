"""IVE-only delivery floor with an immutable whole-song I AM minimum.

New StageEdit renders must first pass their role-conditioned professional
montage review and then pass the same three whole-song metrics achieved by the
initial I AM result.  Old, generation-less analyses retain the narrow legacy
fallback that evaluates those same metrics directly.  Other artists/modes and
explicitly hidden diagnostic/comparison artifacts remain out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .rendered_montage_grammar_review import (
    RENDERED_SEGMENT_INTERIOR_GEOMETRY_REVIEW_CONTRACT,
)
from .rendered_proxy_review import (
    SEGMENT_INTERIOR_GEOMETRY_ANALYSIS_BASIS,
    SEGMENT_INTERIOR_GEOMETRY_CONTRACT,
    SEGMENT_INTERIOR_GEOMETRY_SCHEMA_VERSION,
)


ANALYSIS_BASIS = "ive_automatic_group_delivery_product_quality_floor_v1"
APP_RESULT_EXPOSURE_CONTRACT = "automatic_ive_group_app_result_exposure_v1"
PRODUCT_QUALITY_FLOOR_ASSET = Path(__file__).with_name("models") / (
    "ive_product_quality_floor_v1.json"
)
PRODUCT_QUALITY_FLOOR_ASSET_SHA256 = (
    "59da0a147212ccc6330fa72997e566918c978dc88b4b554024b894dd89b5a9dc"
)
I_AM_ABSOLUTE_MINIMUM_ANALYSIS_SHA256 = (
    "43accf583a7749243ddaab2c9b8b6040e8eaab5d52016c24e31668d1efc2e16e"
)
# Backward-compatible names for callers that imported the original v1
# constants before the baseline became mandatory for modern renders too.
LEGACY_BASELINE_ASSET = PRODUCT_QUALITY_FLOOR_ASSET
LEGACY_BASELINE_ASSET_SHA256 = PRODUCT_QUALITY_FLOOR_ASSET_SHA256
LEGACY_I_AM_ANALYSIS_SHA256 = I_AM_ABSOLUTE_MINIMUM_ANALYSIS_SHA256
PRE_REVIEW_REPAIR_LINEAGE_CONTRACT = (
    "ive_selected_failed_pre_review_successful_boundary_repair_v1"
)
ROLE_CONDITIONED_HIGHLIGHT_AGGREGATE_CONTRACT = (
    "professional_montage_policy_v2_strict_highlight_duration_gate_v1"
)
_EPSILON = 1.0e-12


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class IVEProductQualityFloorError(ValueError):
    """Raised when the immutable floor or its input evidence is malformed."""


@dataclass(frozen=True, slots=True)
class IVEProductQualityFloorResult:
    required: bool
    passed: bool | None
    evaluation_mode: str
    failure_codes: tuple[str, ...]
    metrics: Mapping[str, Any]
    baseline: Mapping[str, Any] | None
    repair_lineage: Mapping[str, Any]
    analysis_basis: str = ANALYSIS_BASIS

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "analysis_basis": self.analysis_basis,
            "required": self.required,
            "passed": self.passed,
            "evaluation_mode": self.evaluation_mode,
            "failure_codes": list(self.failure_codes),
            "metrics": dict(self.metrics),
            "baseline": dict(self.baseline) if self.baseline is not None else None,
            "repair_lineage": dict(self.repair_lineage),
        }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IVEProductQualityFloorError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise IVEProductQualityFloorError(f"{name} must be an array")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IVEProductQualityFloorError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise IVEProductQualityFloorError(f"{name} must be finite")
    return number


def _load_immutable_floor() -> dict[str, Any]:
    try:
        raw = PRODUCT_QUALITY_FLOOR_ASSET.read_bytes()
    except OSError as exc:
        raise IVEProductQualityFloorError(
            f"could not read immutable IVE product floor: {exc}"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != PRODUCT_QUALITY_FLOOR_ASSET_SHA256:
        raise IVEProductQualityFloorError(
            "immutable IVE product-floor asset SHA-256 changed"
        )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                IVEProductQualityFloorError(
                    f"product floor contains non-finite JSON number {token}"
                )
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IVEProductQualityFloorError(
            f"could not decode immutable IVE product floor: {exc}"
        ) from exc
    floor = dict(_mapping(payload, "IVE product floor"))
    if set(floor) != {
        "schema_version",
        "analysis_basis",
        "scope",
        "i_am_absolute_minimum_baseline",
        "legacy_fallback_policy",
        "delivery_policy",
    }:
        raise IVEProductQualityFloorError("IVE product-floor schema changed")
    if (
        floor.get("schema_version") != 1
        or floor.get("analysis_basis") != ANALYSIS_BASIS
    ):
        raise IVEProductQualityFloorError("IVE product-floor identity changed")
    scope = _mapping(floor.get("scope"), "IVE product-floor scope")
    if dict(scope) != {
        "artist": "IVE",
        "edit_mode": "group",
        "source_mode": "automatic",
    }:
        raise IVEProductQualityFloorError("IVE product-floor scope changed")
    baseline = _mapping(
        floor.get("i_am_absolute_minimum_baseline"),
        "I AM absolute-minimum baseline",
    )
    if baseline.get("analysis_sha256") != I_AM_ABSOLUTE_MINIMUM_ANALYSIS_SHA256:
        raise IVEProductQualityFloorError("I AM baseline analysis identity changed")
    metrics = _mapping(baseline.get("metrics"), "I AM absolute-minimum metrics")
    expected_metrics = {
        "close_duration_fraction": 0.32152109124843137,
        "nonwide_duration_fraction": 0.5143116148397875,
        "maximum_consecutive_wide_run_seconds": 7.331084476681127,
    }
    if set(metrics) != set(expected_metrics) or any(
        not math.isclose(
            _finite(metrics.get(name), f"I AM baseline {name}"),
            expected,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        for name, expected in expected_metrics.items()
    ):
        raise IVEProductQualityFloorError("I AM baseline metric identity changed")
    legacy_policy = _mapping(
        floor.get("legacy_fallback_policy"),
        "legacy fallback policy",
    )
    if dict(legacy_policy) != {
        "eligible_only_when_stage_edit_generation_absent": True,
        "eligible_only_when_role_conditioned_evidence_absent": True,
        "close_duration_fraction_comparison": "greater_than_or_equal",
        "nonwide_duration_fraction_comparison": "greater_than_or_equal",
        "maximum_consecutive_wide_run_seconds_comparison": "less_than_or_equal",
        "new_generation_missing_role_conditioned_evidence_fails_closed": True,
    }:
        raise IVEProductQualityFloorError("legacy fallback policy changed")
    delivery_policy = _mapping(
        floor.get("delivery_policy"),
        "delivery policy",
    )
    if dict(delivery_policy) != {
        "evaluation_order": [
            "role_conditioned_professional_review",
            "whole_song_i_am_absolute_minimum",
        ],
        "modern_role_conditioned_review_must_pass": True,
        "modern_whole_song_i_am_absolute_floor_must_pass": True,
        "selected_failed_pre_review_requires_successful_repair_lineage": True,
        "other_artist_or_mode_behavior_unchanged": True,
    }:
        raise IVEProductQualityFloorError("IVE delivery policy changed")
    return floor


def _is_automatic_ive_group(analysis: Mapping[str, Any]) -> bool:
    request = analysis.get("request")
    if not isinstance(request, Mapping):
        return False
    return (
        str(request.get("group", "")).strip().upper() == "IVE"
        and str(request.get("edit_mode", "")).strip().lower() == "group"
        and str(request.get("source_mode", "")).strip().lower() == "automatic"
    )


def _explicit_non_delivery_artifact(
    analysis: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Recognize only a fully bound diagnostic/comparison exemption.

    Missing role metadata remains a product delivery.  This keeps archived
    BANG-like outputs from becoming exempt merely because they predate the
    StageEdit generation record.  A comparison must also bind the explicit
    comparison-render opt-in, and neither non-delivery role may be exposed as
    a completed video in the app result.
    """

    role = analysis.get("artifact_role")
    if role not in {"diagnostic", "comparison"}:
        return False, None
    exposure = analysis.get("app_result_exposure")
    if not (
        isinstance(exposure, Mapping)
        and exposure.get("contract")
        == APP_RESULT_EXPOSURE_CONTRACT
        and exposure.get("video_output_exposed") is False
        and exposure.get("delivery_completion_message_allowed") is False
    ):
        return False, None
    render = analysis.get("render")
    if role == "comparison":
        if not (
            isinstance(render, Mapping)
            and render.get("artifact_role") == "comparison"
            and render.get("allow_failed_comparison_render") is True
        ):
            return False, None
    elif render is not None:
        # A rendered diagnostic must be introduced under a future explicit
        # contract rather than silently reusing the no-render diagnostic role.
        return False, None
    return True, str(role)


def automatic_ive_group_product_floor_required(
    analysis: Mapping[str, Any],
) -> bool:
    """Return whether this analysis represents an IVE product delivery."""

    if not isinstance(analysis, Mapping):
        raise TypeError("analysis must be an object")
    if not _is_automatic_ive_group(analysis):
        return False
    exempt, _role = _explicit_non_delivery_artifact(analysis)
    return not exempt


def _shot_class(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(
        " ", "_"
    )
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


def _whole_song_i_am_metrics(analysis: Mapping[str, Any]) -> dict[str, float]:
    visual = _mapping(analysis.get("visual_timeline"), "visual_timeline")
    choices = _sequence(visual.get("choices"), "visual_timeline.choices")
    if not choices:
        raise IVEProductQualityFloorError("whole-song visual timeline is empty")
    common_range = _mapping(
        analysis.get("common_master_range"), "common_master_range"
    )
    common_start = _finite(
        common_range.get("start_seconds"), "common_master_range.start_seconds"
    )
    common_end = _finite(
        common_range.get("end_seconds"), "common_master_range.end_seconds"
    )
    common_duration = _finite(
        common_range.get("duration_seconds"),
        "common_master_range.duration_seconds",
    )
    if (
        common_end <= common_start
        or not math.isclose(
            common_end - common_start,
            common_duration,
            rel_tol=0.0,
            abs_tol=_EPSILON,
        )
    ):
        raise IVEProductQualityFloorError("common master range is inconsistent")
    durations = {name: 0.0 for name in ("wide", "full", "medium", "close")}
    current_wide_run = 0.0
    maximum_wide_run = 0.0
    previous_end: float | None = None
    for index, raw in enumerate(choices):
        choice = _mapping(raw, f"visual_timeline.choices[{index}]")
        start = _finite(
            choice.get("canonical_master_start"),
            f"visual_timeline.choices[{index}].canonical_master_start",
        )
        end = _finite(
            choice.get("canonical_master_end"),
            f"visual_timeline.choices[{index}].canonical_master_end",
        )
        serialized_duration = _finite(
            choice.get("duration_seconds"),
            f"visual_timeline.choices[{index}].duration_seconds",
        )
        if end <= start or not math.isclose(
            end - start,
            serialized_duration,
            rel_tol=0.0,
            abs_tol=_EPSILON,
        ):
            raise IVEProductQualityFloorError(
                "whole-song visual choice duration disagrees with canonical endpoints"
            )
        expected_start = common_start if previous_end is None else previous_end
        if not math.isclose(
            start,
            expected_start,
            rel_tol=0.0,
            abs_tol=_EPSILON,
        ):
            raise IVEProductQualityFloorError(
                "whole-song visual timeline must be exactly contiguous"
            )
        previous_end = end
        shot = _shot_class(choice.get("shot_type"))
        duration = end - start
        if shot == "unknown":
            raise IVEProductQualityFloorError(
                "whole-song visual timeline contains unknown shot-scale duration"
            )
        durations[shot] += duration
        if shot in {"wide", "full"}:
            current_wide_run += duration
            maximum_wide_run = max(maximum_wide_run, current_wide_run)
        else:
            current_wide_run = 0.0
    known_duration = sum(durations.values())
    if (
        known_duration <= 0.0
        or previous_end is None
        or not math.isclose(
            previous_end,
            common_end,
            rel_tol=0.0,
            abs_tol=_EPSILON,
        )
        or not math.isclose(
            known_duration,
            common_duration,
            rel_tol=0.0,
            abs_tol=_EPSILON,
        )
    ):
        raise IVEProductQualityFloorError(
            "whole-song visual timeline does not bind the complete common master range"
        )
    return {
        "known_duration_seconds": known_duration,
        "close_duration_fraction": durations["close"] / known_duration,
        "nonwide_duration_fraction": (
            durations["medium"] + durations["close"]
        )
        / known_duration,
        "maximum_consecutive_wide_run_seconds": maximum_wide_run,
    }


def _modern_rendered_geometry_i_am_metrics(
    analysis: Mapping[str, Any],
    review: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Bind the modern whole-song floor to decoded interior geometry.

    The proxy evidence is self-hashed and the montage review carries a second
    hash over its thresholded, duration-weighted interpretation.  This check
    verifies both layers and their cross-reference before accepting any
    close/non-wide duration or consecutive-wide metric.
    """

    raw_summary = review.get("rendered_segment_interior_geometry")
    summary = dict(
        _mapping(
            raw_summary,
            "rendered montage segment-interior geometry review",
        )
    )
    review_evidence_sha256 = summary.pop("review_evidence_sha256", None)
    if not (
        isinstance(review_evidence_sha256, str)
        and len(review_evidence_sha256) == 64
        and all(
            character in "0123456789abcdef"
            for character in review_evidence_sha256
        )
    ):
        raise IVEProductQualityFloorError(
            "rendered segment-interior review SHA-256 is invalid"
        )
    if _canonical_sha256(summary) != review_evidence_sha256:
        raise IVEProductQualityFloorError(
            "rendered segment-interior review SHA-256 mismatch"
        )
    if summary.get("contract") != (
        RENDERED_SEGMENT_INTERIOR_GEOMETRY_REVIEW_CONTRACT
    ):
        raise IVEProductQualityFloorError(
            "rendered segment-interior review contract changed"
        )
    if summary.get("required") is not True or summary.get("available") is not True:
        raise IVEProductQualityFloorError(
            "modern IVE delivery lacks required segment-interior geometry"
        )
    if not isinstance(summary.get("passed"), bool):
        raise IVEProductQualityFloorError(
            "rendered segment-interior review pass flag is invalid"
        )
    failure_codes = summary.get("failure_codes")
    if not isinstance(failure_codes, list) or any(
        not isinstance(value, str) or not value for value in failure_codes
    ):
        raise IVEProductQualityFloorError(
            "rendered segment-interior review failure codes are invalid"
        )
    if bool(failure_codes) == bool(summary.get("passed")):
        raise IVEProductQualityFloorError(
            "rendered segment-interior review pass state is inconsistent"
        )

    proxy_review = _mapping(
        analysis.get("rendered_proxy_review"), "rendered_proxy_review"
    )
    raw_source = proxy_review.get("segment_interior_geometry")
    source = dict(
        _mapping(raw_source, "rendered proxy segment-interior geometry")
    )
    source_evidence_sha256 = source.pop("evidence_sha256", None)
    if not (
        isinstance(source_evidence_sha256, str)
        and len(source_evidence_sha256) == 64
        and all(
            character in "0123456789abcdef"
            for character in source_evidence_sha256
        )
    ):
        raise IVEProductQualityFloorError(
            "rendered proxy segment-interior evidence SHA-256 is invalid"
        )
    if _canonical_sha256(source) != source_evidence_sha256:
        raise IVEProductQualityFloorError(
            "rendered proxy segment-interior evidence SHA-256 mismatch"
        )
    if (
        source.get("schema_version")
        != SEGMENT_INTERIOR_GEOMETRY_SCHEMA_VERSION
        or source.get("analysis_basis")
        != SEGMENT_INTERIOR_GEOMETRY_ANALYSIS_BASIS
        or source.get("contract") != SEGMENT_INTERIOR_GEOMETRY_CONTRACT
    ):
        raise IVEProductQualityFloorError(
            "rendered proxy segment-interior evidence contract changed"
        )
    if summary.get("source_evidence_sha256") != source_evidence_sha256:
        raise IVEProductQualityFloorError(
            "montage review references different segment-interior evidence"
        )
    if summary.get("source_sampling_schedule_sha256") != source.get(
        "sampling_schedule_sha256"
    ):
        raise IVEProductQualityFloorError(
            "segment-interior sampling schedule provenance mismatch"
        )
    if summary.get("source_render_timeline_binding_sha256") != source.get(
        "render_timeline_binding_sha256"
    ):
        raise IVEProductQualityFloorError(
            "segment-interior RenderPlan provenance mismatch"
        )

    observed = _mapping(
        summary.get("whole_song_observed"),
        "rendered segment-interior whole-song metrics",
    )
    total_duration = _finite(
        observed.get("total_duration_seconds"),
        "rendered segment-interior total duration",
    )
    known_duration = _finite(
        observed.get("known_duration_seconds"),
        "rendered segment-interior known duration",
    )
    known_fraction = _finite(
        observed.get("known_duration_fraction"),
        "rendered segment-interior known duration fraction",
    )
    close_fraction = _finite(
        observed.get("close_duration_fraction"),
        "rendered segment-interior close duration fraction",
    )
    nonwide_fraction = _finite(
        observed.get("nonwide_duration_fraction"),
        "rendered segment-interior nonwide duration fraction",
    )
    maximum_wide_run = _finite(
        observed.get("maximum_consecutive_wide_run_seconds"),
        "rendered segment-interior maximum wide run",
    )
    if (
        total_duration <= 0.0
        or known_duration < 0.0
        or not 0.0 <= known_fraction <= 1.0
        or not 0.0 <= close_fraction <= nonwide_fraction <= 1.0
        or not 0.0 <= maximum_wide_run <= total_duration + _EPSILON
        or not math.isclose(
            known_duration / total_duration,
            known_fraction,
            rel_tol=0.0,
            abs_tol=_EPSILON,
        )
    ):
        raise IVEProductQualityFloorError(
            "rendered segment-interior whole-song metrics are inconsistent"
        )
    failures: list[str] = []
    if summary.get("passed") is not True:
        failures.append("modern_rendered_segment_interior_geometry_failed")
    if known_fraction < 1.0 - _EPSILON:
        failures.append(
            "modern_rendered_segment_interior_geometry_duration_incomplete"
        )
    return (
        {
            "known_duration_seconds": known_duration,
            "close_duration_fraction": close_fraction,
            "nonwide_duration_fraction": nonwide_fraction,
            "maximum_consecutive_wide_run_seconds": maximum_wide_run,
            "rendered_geometry_known_duration_fraction": known_fraction,
            "rendered_geometry_source_evidence_sha256": source_evidence_sha256,
            "rendered_geometry_review_evidence_sha256": (
                review_evidence_sha256
            ),
            "metric_source": (
                "decoded_proxy_segment_interior_sample_weighted_geometry"
            ),
            "not_all_frame_measurement": True,
        },
        tuple(failures),
    )


def _i_am_thresholds(
    baseline_payload: Mapping[str, Any],
) -> dict[str, float]:
    baseline_metrics = _mapping(
        baseline_payload.get("metrics"),
        "I AM absolute-minimum metrics",
    )
    return {
        name: _finite(baseline_metrics.get(name), f"I AM baseline {name}")
        for name in (
            "close_duration_fraction",
            "nonwide_duration_fraction",
            "maximum_consecutive_wide_run_seconds",
        )
    }


def _i_am_floor_failures(
    observed: Mapping[str, float],
    thresholds: Mapping[str, float],
    *,
    modern: bool,
) -> tuple[str, ...]:
    prefix = "modern" if modern else "legacy"
    failures: list[str] = []
    if (
        observed["close_duration_fraction"] + _EPSILON
        < thresholds["close_duration_fraction"]
    ):
        failures.append(
            f"{prefix}_close_duration_fraction_below_i_am_floor"
        )
    if (
        observed["nonwide_duration_fraction"] + _EPSILON
        < thresholds["nonwide_duration_fraction"]
    ):
        failures.append(
            f"{prefix}_nonwide_duration_fraction_below_i_am_floor"
        )
    if (
        observed["maximum_consecutive_wide_run_seconds"]
        > thresholds["maximum_consecutive_wide_run_seconds"] + _EPSILON
    ):
        failures.append(
            f"{prefix}_maximum_consecutive_wide_run_exceeds_i_am_floor"
        )
    return tuple(failures)


def _selected_pre_review_and_lineage(
    analysis: Mapping[str, Any],
    selection: Mapping[str, Any] | None,
) -> tuple[
    dict[str, Any] | None,
    tuple[Mapping[str, Any], ...],
    int | None,
]:
    if selection is None:
        selection = {}
    selected_index = selection.get("selected_candidate_index")
    if isinstance(selected_index, bool) or not isinstance(selected_index, int):
        selected_index = None
    complete = selection.get("complete_candidates")
    if not isinstance(complete, list):
        complete = []
    matches = [
        item
        for item in complete
        if isinstance(item, Mapping)
        and item.get("candidate_index") == selected_index
    ]
    selected = matches[0] if len(matches) == 1 else {}
    visual = selected.get("visual_timeline") if selected else None
    summaries = (
        visual.get("candidate_review_summaries")
        if isinstance(visual, Mapping)
        else None
    )
    selected_summary: dict[str, Any] | None = None
    if isinstance(summaries, list):
        explicit = [
            item
            for item in summaries
            if isinstance(item, Mapping) and item.get("selected") is True
        ]
        if len(explicit) == 1:
            selected_summary = dict(explicit[0])
        elif isinstance(visual, Mapping):
            visual_index = visual.get("selected_candidate_index")
            indexed = [
                item
                for item in summaries
                if isinstance(item, Mapping)
                and item.get("candidate_index") == visual_index
            ]
            if len(indexed) == 1:
                selected_summary = dict(indexed[0])
    if selected_summary is None:
        final_visual = analysis.get("visual_timeline")
        final_summaries = (
            final_visual.get("candidate_review_summaries")
            if isinstance(final_visual, Mapping)
            else None
        )
        if isinstance(final_summaries, list):
            final_selected_index = final_visual.get("selected_candidate_index")
            if (
                selected_index is None
                and isinstance(final_selected_index, int)
                and not isinstance(final_selected_index, bool)
            ):
                selected_index = final_selected_index
            explicit = [
                item
                for item in final_summaries
                if isinstance(item, Mapping) and item.get("selected") is True
            ]
            if len(explicit) == 1:
                selected_summary = dict(explicit[0])
    raw_lineage = selected.get("candidate_lineage")
    lineage = (
        tuple(item for item in raw_lineage if isinstance(item, Mapping))
        if isinstance(raw_lineage, list)
        else ()
    )
    promotion = selection.get("stage_edit_render_promotion")
    promotion_lineage = (
        promotion.get("selected_candidate_lineage")
        if isinstance(promotion, Mapping)
        else None
    )
    if isinstance(promotion_lineage, list):
        lineage = tuple(
            item for item in promotion_lineage if isinstance(item, Mapping)
        )
    if selected_summary is None:
        adapter_reviews = [
            item.get("visual_review")
            for item in lineage
            if isinstance(item.get("visual_hard_gates_passed"), bool)
            and isinstance(item.get("visual_review"), Mapping)
        ]
        if len(adapter_reviews) == 1:
            selected_summary = dict(adapter_reviews[0])
    return selected_summary, lineage, selected_index


def _repair_lineage_evidence(
    analysis: Mapping[str, Any],
    selection: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    summary, lineage, selected_index = _selected_pre_review_and_lineage(
        analysis,
        selection,
    )
    if summary is None:
        return {
            "selected_pre_review_evidence_available": False,
            "selected_pre_review_failure_observed": False,
            "successful_repair_lineage_required": False,
            "successful_repair_lineage_found": None,
        }, ()
    summary_passed = summary.get("passed")
    summary_candidate_index = summary.get("candidate_index")
    failures = summary.get("hard_failure_codes")
    failure_codes = (
        tuple(
            str(value)
            for value in failures
            if isinstance(value, str) and value
        )
        if isinstance(failures, list)
        else ()
    )
    summary_identity_valid = bool(
        isinstance(summary_passed, bool)
        and isinstance(summary_candidate_index, int)
        and not isinstance(summary_candidate_index, bool)
        and selected_index is not None
        and summary_candidate_index == selected_index
        and isinstance(failures, list)
        and len(failure_codes) == len(failures)
        and len(failure_codes) == len(set(failure_codes))
        and (
            (summary_passed is True and not failure_codes)
            or (summary_passed is False and bool(failure_codes))
        )
    )
    if not summary_identity_valid:
        return {
            "selected_pre_review_evidence_available": True,
            "selected_pre_review_evidence_valid": False,
            "selected_pre_review_failure_observed": (
                summary_passed is False
            ),
            "successful_repair_lineage_required": (
                summary_passed is False
            ),
            "successful_repair_lineage_found": False,
        }, ("selected_pre_review_evidence_invalid",)
    if summary_passed is not False:
        return {
            "selected_pre_review_evidence_available": True,
            "selected_pre_review_evidence_valid": True,
            "selected_pre_review_failure_observed": False,
            "successful_repair_lineage_required": False,
            "successful_repair_lineage_found": None,
        }, ()
    valid_events = [
        item
        for item in lineage
        if item.get("contract") == PRE_REVIEW_REPAIR_LINEAGE_CONTRACT
        and item.get("stage") == "pre_review_failure_repair_validation"
        and item.get("pre_review_passed") is False
        and item.get("post_repair_timeline_review_passed") is True
        and item.get("delivery_eligible") is True
        and item.get("candidate_index") == selected_index
        and isinstance(item.get("boundary_repair_changed_count"), int)
        and not isinstance(item.get("boundary_repair_changed_count"), bool)
        and int(item["boundary_repair_changed_count"]) > 0
        and item.get("pre_review_hard_failure_codes") == list(failure_codes)
    ]
    found = len(valid_events) == 1
    evidence = {
        "selected_pre_review_evidence_available": True,
        "selected_pre_review_evidence_valid": True,
        "selected_candidate_index": selected_index,
        "selected_pre_review_failure_observed": True,
        "pre_review_hard_failure_codes": list(failure_codes),
        "successful_repair_lineage_required": True,
        "successful_repair_lineage_found": found,
        "valid_event_count": len(valid_events),
        "contract": PRE_REVIEW_REPAIR_LINEAGE_CONTRACT,
    }
    return evidence, (
        ()
        if found
        else ("selected_failed_pre_review_lacks_successful_repair_lineage",)
    )


def evaluate_ive_product_quality_floor(
    analysis: Mapping[str, Any],
    *,
    selection: Mapping[str, Any] | None = None,
) -> IVEProductQualityFloorResult:
    """Evaluate the IVE product floor without affecting any other scope."""

    if not isinstance(analysis, Mapping):
        raise TypeError("analysis must be an object")
    if selection is not None and not isinstance(selection, Mapping):
        raise TypeError("selection must be an object when supplied")
    if not _is_automatic_ive_group(analysis):
        return IVEProductQualityFloorResult(
            required=False,
            passed=None,
            evaluation_mode="not_applicable",
            failure_codes=(),
            metrics={},
            baseline=None,
            repair_lineage={
                "selected_pre_review_evidence_available": False,
                "selected_pre_review_failure_observed": False,
                "successful_repair_lineage_required": False,
                "successful_repair_lineage_found": None,
            },
        )

    explicit_non_delivery, artifact_role = _explicit_non_delivery_artifact(
        analysis
    )
    if explicit_non_delivery:
        return IVEProductQualityFloorResult(
            required=False,
            passed=None,
            evaluation_mode="explicit_non_delivery_artifact",
            failure_codes=(),
            metrics={
                "artifact_role": artifact_role,
                "app_result_video_output_exposed": False,
            },
            baseline=None,
            repair_lineage={
                "selected_pre_review_evidence_available": False,
                "selected_pre_review_failure_observed": False,
                "successful_repair_lineage_required": False,
                "successful_repair_lineage_found": None,
            },
        )

    floor = _load_immutable_floor()
    baseline_payload = _mapping(
        floor["i_am_absolute_minimum_baseline"],
        "I AM absolute-minimum baseline",
    )
    thresholds = _i_am_thresholds(baseline_payload)
    repair_evidence, repair_failures = _repair_lineage_evidence(
        analysis,
        selection,
    )
    review = analysis.get("rendered_montage_grammar_review")
    aggregate = (
        review.get("highlight_aggregate")
        if isinstance(review, Mapping)
        else None
    )
    role_conditioned = bool(
        isinstance(aggregate, Mapping)
        and aggregate.get("contract") == ROLE_CONDITIONED_HIGHLIGHT_AGGREGATE_CONTRACT
    )
    generation_present = isinstance(
        analysis.get("stage_edit_generation"),
        Mapping,
    )
    current_artifact_contract_present = bool(
        "artifact_role" in analysis or "app_result_exposure" in analysis
    )

    failures = list(repair_failures)
    if role_conditioned:
        assert isinstance(review, Mapping)
        assert isinstance(aggregate, Mapping)
        observed = aggregate.get("observed")
        modern_metrics = dict(observed) if isinstance(observed, Mapping) else {}
        if not repair_evidence.get("selected_pre_review_evidence_available"):
            failures.append("modern_selected_pre_review_evidence_missing")
        if review.get("passed") is not True:
            failures.append("modern_role_conditioned_montage_review_failed")
        if aggregate.get("passed") is not True:
            failures.append("modern_highlight_aggregate_failed")
        section_count = aggregate.get("strict_highlight_section_count")
        if (
            isinstance(section_count, bool)
            or not isinstance(section_count, int)
            or section_count < 1
        ):
            failures.append("modern_role_conditioned_highlight_evidence_empty")
        (
            whole_song_observed,
            rendered_geometry_failures,
        ) = _modern_rendered_geometry_i_am_metrics(analysis, review)
        failures.extend(rendered_geometry_failures)
        whole_song_failures = _i_am_floor_failures(
            whole_song_observed,
            thresholds,
            modern=True,
        )
        failures.extend(whole_song_failures)
        return IVEProductQualityFloorResult(
            required=True,
            passed=not failures,
            evaluation_mode="modern_role_conditioned_profile_gate",
            failure_codes=tuple(dict.fromkeys(failures)),
            metrics={
                "role_conditioned_evidence_present": True,
                "highlight_aggregate": dict(aggregate),
                "observed": modern_metrics,
                "whole_song_i_am_absolute_floor": {
                    **whole_song_observed,
                    "passed": not whole_song_failures,
                    "failure_codes": list(whole_song_failures),
                },
                "legacy_whole_song_fallback_applied": False,
                "whole_song_i_am_absolute_floor_applied": True,
            },
            baseline={
                "song": baseline_payload["song"],
                "analysis_sha256": baseline_payload["analysis_sha256"],
                "thresholds": thresholds,
                "usage": (
                    "secondary_absolute_minimum_applied_after_"
                    "role_conditioned_review"
                ),
            },
            repair_lineage=repair_evidence,
        )

    if generation_present or current_artifact_contract_present:
        failures.append(
            "modern_generation_missing_role_conditioned_highlight_evidence"
            if generation_present
            else "current_delivery_missing_role_conditioned_highlight_evidence"
        )
        return IVEProductQualityFloorResult(
            required=True,
            passed=False,
            evaluation_mode="modern_missing_role_conditioned_evidence",
            failure_codes=tuple(dict.fromkeys(failures)),
            metrics={
                "role_conditioned_evidence_present": False,
                "legacy_whole_song_fallback_applied": False,
                "whole_song_i_am_absolute_floor_applied": False,
            },
            baseline={
                "analysis_sha256": baseline_payload["analysis_sha256"],
                "usage": (
                    "provenance_only_fallback_forbidden_for_modern_generation"
                    if generation_present
                    else "provenance_only_fallback_forbidden_for_current_delivery"
                ),
            },
            repair_lineage=repair_evidence,
        )

    observed = _whole_song_i_am_metrics(analysis)
    failures.extend(
        _i_am_floor_failures(observed, thresholds, modern=False)
    )
    return IVEProductQualityFloorResult(
        required=True,
        passed=not failures,
        evaluation_mode="legacy_whole_song_i_am_positive_regression_fallback",
        failure_codes=tuple(dict.fromkeys(failures)),
        metrics={
            **observed,
            "role_conditioned_evidence_present": False,
            "legacy_whole_song_fallback_applied": True,
            "whole_song_i_am_absolute_floor_applied": True,
        },
        baseline={
            "song": baseline_payload["song"],
            "analysis_sha256": baseline_payload["analysis_sha256"],
            "thresholds": thresholds,
        },
        repair_lineage=repair_evidence,
    )


__all__ = [
    "ANALYSIS_BASIS",
    "APP_RESULT_EXPOSURE_CONTRACT",
    "I_AM_ABSOLUTE_MINIMUM_ANALYSIS_SHA256",
    "IVEProductQualityFloorError",
    "IVEProductQualityFloorResult",
    "LEGACY_BASELINE_ASSET_SHA256",
    "LEGACY_I_AM_ANALYSIS_SHA256",
    "PRE_REVIEW_REPAIR_LINEAGE_CONTRACT",
    "PRODUCT_QUALITY_FLOOR_ASSET_SHA256",
    "ROLE_CONDITIONED_HIGHLIGHT_AGGREGATE_CONTRACT",
    "automatic_ive_group_product_floor_required",
    "evaluate_ive_product_quality_floor",
]
