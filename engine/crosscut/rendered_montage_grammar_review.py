"""Fail-closed rendered-plan review for IVE stage-mix visual grammar.

The learned decoder owns source and timing decisions.  This module independently
checks the *selected* visual timeline and decoded proxy evidence so an edit made
almost entirely from formation wides cannot pass merely because its cuts are on
beat.  Formation wides remain desirable anchors; a long highlight must also
contain medium/close coverage and at least one short non-wide source burst.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .professional_montage_policy import (
    CLOSE_BURST_EVENT_EVIDENCE_CONTRACT,
    DURATION_PREFERRED_TARGET_QUANTILE,
    DURATION_PRODUCT_MINIMUM_QUANTILE,
    HIGHLIGHT_CADENCE_EVIDENCE_CONTRACT,
    HIGHLIGHT_CADENCE_PREFERRED_TARGET_QUANTILE,
    HIGHLIGHT_CADENCE_PRODUCT_MINIMUM_QUANTILE,
    HIGHLIGHT_CADENCE_STATISTIC_PATH,
    HIGHLIGHT_CLOSE_DURATION_STATISTIC_PATH,
    HIGHLIGHT_DURATION_EVIDENCE_CONTRACT,
    HIGHLIGHT_NONWIDE_DURATION_STATISTIC_PATH,
    HIGHLIGHT_WIDE_RUN_EVIDENCE_CONTRACT,
    HIGHLIGHT_WIDE_RUN_STATISTIC_PATH,
    REFERENCE_FLOOR_QUANTILE,
    ROLE_CONDITIONED_DURATION_EVIDENCE_CONTRACT,
    WIDE_RUN_HARD_MAXIMUM_QUANTILE,
    WIDE_RUN_PREFERRED_MAXIMUM_QUANTILE,
    close_burst_policy_binding_payload,
    close_burst_policy_binding_sha256,
)
from .rendered_proxy_review import (
    SEGMENT_INTERIOR_GEOMETRY_ANALYSIS_BASIS,
    SEGMENT_INTERIOR_GEOMETRY_CONTRACT,
    SEGMENT_INTERIOR_GEOMETRY_SCHEMA_VERSION,
    SEGMENT_INTERIOR_SAMPLE_FRACTIONS,
    verified_file_content_sha256,
)
from .stage_edit_dynamic_crop_motion_authority_v3 import (
    DEFAULT_MAXIMUM_MOTION_APPLIED_FRACTION,
    DEFAULT_MINIMUM_MOTION_APPLIED_FRACTION,
    DynamicCropMotionAuthorityV3Error,
    DynamicCropMotionTimelineEvidenceV3,
)


RENDERED_MONTAGE_GRAMMAR_ANALYSIS_BASIS = (
    "rendered_selected_plan_exact_phrase_and_section_anchor_formation_"
    "role_conditioned_duration_chorus_cadence_close_burst_and_"
    "interior_geometry_grammar_v7"
)
EXACT_STAGE_EDIT_SHOT_CLOCK_CONTRACT = (
    "serialized_inference_token_section_and_phrase_progress_per_selected_shot_v1"
)
HIGHLIGHT_AGGREGATE_CONTRACT = (
    "professional_montage_policy_v2_strict_highlight_duration_gate_v1"
)
RENDERED_SEGMENT_INTERIOR_GEOMETRY_REVIEW_CONTRACT = (
    "content_hash_bound_sampled_rendered_segment_geometry_duration_review_v2"
)
_WIDE = frozenset({"wide", "full"})
_CLOSE = frozenset({"close"})
_NONWIDE = frozenset({"medium", "close"})
_DURATION_GATED_ROLES = frozenset({"build", "chorus_like", "bridge_like"})
_SECTION_ROLES = frozenset(
    {"intro", "verse_like", "build", "chorus_like", "bridge_like", "outro"}
)
_ELIGIBLE_SOURCE_INDEX_DOMAIN = "eligible_source_index"
_REQUEST_SOURCE_INDEX_DOMAIN = "request_source_index"
_RENDER_PLAN_SOURCE_INDEX_DOMAIN = "render_plan_sources_array_index"
_RENDER_PLAN_BOUNDARY_INDEX_DOMAIN = "render_plan_transition_index"
_HARD_CUT = "hard_cut"
_FACE_DISSOLVE = "face_dissolve"
_STABLE_PROXY_IDENTITY_FIELDS = (
    "width",
    "height",
    "fps",
    "decoded_frame_count",
    "duration_seconds",
    "render_codec",
    "content_sha256",
)
_EPSILON = 1.0e-6


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    # Retained only so archived configuration payloads stay explicit.  The
    # release gate no longer uses duration to exempt a short chorus.
    highlight_minimum_duration_seconds: float = 5.0
    minimum_known_shot_fraction: float = 0.80
    minimum_closeup_fraction: float = 0.1696555074498807
    minimum_nonwide_fraction: float = 0.33771372775759073
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
    # The existing visual-score classifier uses face_height / 0.42 and calls
    # values below 0.22 wide: 0.42 * 0.22 = 0.0924.
    minimum_nonwide_proxy_face_height_fraction: float = 0.0924
    minimum_closeup_proxy_face_height_fraction: float = 0.2184
    pooled_product_minimum_closeup_fraction: float = 0.22343208166310066
    pooled_product_minimum_nonwide_fraction: float = 0.48417342818793935
    preferred_closeup_fraction: float = 0.3267829794497803
    preferred_nonwide_fraction: float = 0.6607923807246032
    maximum_consecutive_wide_run_seconds: float = 8.258529233180866
    preferred_maximum_consecutive_wide_run_seconds: float = (
        5.833280462440257
    )
    chorus_role_pooled_product_minimum_closeup_fraction: float = (
        0.2532731561991967
    )
    chorus_role_pooled_product_minimum_nonwide_fraction: float = (
        0.48842910464414785
    )
    chorus_role_preferred_closeup_fraction: float = 0.3409589126803301
    chorus_role_preferred_nonwide_fraction: float = 0.6631535806873134
    chorus_role_preferred_maximum_consecutive_wide_run_seconds: float = (
        5.872243112415582
    )
    build_minimum_closeup_fraction: float | None = None
    build_minimum_nonwide_fraction: float | None = None
    build_pooled_product_minimum_closeup_fraction: float = (
        0.11876455826051503
    )
    build_pooled_product_minimum_nonwide_fraction: float = (
        0.1511882107564876
    )
    build_preferred_closeup_fraction: float = 0.251412375006117
    build_preferred_nonwide_fraction: float = 0.3118626699888136
    build_maximum_consecutive_wide_run_seconds: float = 7.633375911839522
    build_preferred_maximum_consecutive_wide_run_seconds: float = (
        4.145836014419238
    )
    build_duration_reference_count: int = 20
    bridge_minimum_closeup_fraction: float | None = None
    bridge_minimum_nonwide_fraction: float | None = 0.28697576489194593
    bridge_pooled_product_minimum_closeup_fraction: float = (
        0.2725037829221335
    )
    bridge_pooled_product_minimum_nonwide_fraction: float = (
        0.4220031799500228
    )
    bridge_preferred_closeup_fraction: float = 0.4312104569122468
    bridge_preferred_nonwide_fraction: float = 0.5808194894277292
    bridge_maximum_consecutive_wide_run_seconds: float = 12.077887222555745
    bridge_preferred_maximum_consecutive_wide_run_seconds: float = (
        4.075505767656015
    )
    bridge_duration_reference_count: int = 18
    highlight_product_minimum_source_switches_per_minute: float = (
        18.642084765823732
    )
    highlight_preferred_source_switches_per_minute: float = (
        23.505554096533405
    )
    highlight_cadence_reference_count: int = 20
    policy_profile_analysis_basis: str | None = None
    policy_profile_schema_version: int | None = None
    policy_profile_path: str | None = None
    policy_profile_file_sha256: str | None = None
    policy_profile_content_sha256: str | None = None
    policy_fingerprint_sha256: str | None = None
    minimum_close_burst_shot_count: float | None = None
    minimum_close_burst_duration_seconds: float | None = None
    minimum_close_burst_source_switch_fraction: float | None = None
    close_burst_event_reference_count: int | None = None
    close_burst_policy_binding_sha256: str | None = None
    # This counts only bounded editorial transform accents at a close/nonwide
    # source switch.  It is intentionally not called choreography-motion proof.
    minimum_closeup_editorial_motion_accent_switch_boundaries: int = 0

    def __post_init__(self) -> None:
        for name in (
            "highlight_minimum_duration_seconds",
            "burst_maximum_run_seconds",
            "maximum_consecutive_wide_run_seconds",
            "preferred_maximum_consecutive_wide_run_seconds",
            "chorus_role_preferred_maximum_consecutive_wide_run_seconds",
            "build_maximum_consecutive_wide_run_seconds",
            "build_preferred_maximum_consecutive_wide_run_seconds",
            "bridge_maximum_consecutive_wide_run_seconds",
            "bridge_preferred_maximum_consecutive_wide_run_seconds",
            "highlight_product_minimum_source_switches_per_minute",
            "highlight_preferred_source_switches_per_minute",
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
            "minimum_nonwide_proxy_face_height_fraction",
            "minimum_closeup_proxy_face_height_fraction",
            "pooled_product_minimum_closeup_fraction",
            "pooled_product_minimum_nonwide_fraction",
            "preferred_closeup_fraction",
            "preferred_nonwide_fraction",
            "chorus_role_pooled_product_minimum_closeup_fraction",
            "chorus_role_pooled_product_minimum_nonwide_fraction",
            "chorus_role_preferred_closeup_fraction",
            "chorus_role_preferred_nonwide_fraction",
            "build_pooled_product_minimum_closeup_fraction",
            "build_pooled_product_minimum_nonwide_fraction",
            "build_preferred_closeup_fraction",
            "build_preferred_nonwide_fraction",
            "bridge_pooled_product_minimum_closeup_fraction",
            "bridge_pooled_product_minimum_nonwide_fraction",
            "bridge_preferred_closeup_fraction",
            "bridge_preferred_nonwide_fraction",
        ):
            number = _finite(getattr(self, name), name)
            if not 0.0 <= number <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in (
            "build_minimum_closeup_fraction",
            "build_minimum_nonwide_fraction",
            "bridge_minimum_closeup_fraction",
            "bridge_minimum_nonwide_fraction",
        ):
            value = getattr(self, name)
            if value is not None and not 0.0 < _finite(value, name) <= 1.0:
                raise ValueError(f"{name} must be None or lie in (0, 1]")
        for name in (
            "minimum_nonwide_burst_boundaries",
            "formation_anchor_run_window",
            "minimum_formation_anchor_person_count",
            "strong_formation_anchor_person_count",
            "minimum_post_burst_formation_reset_count",
            "minimum_motion_accent_count",
            "wide_involved_source_switch_reference_count",
            "minimum_closeup_editorial_motion_accent_switch_boundaries",
            "highlight_cadence_reference_count",
            "build_duration_reference_count",
            "bridge_duration_reference_count",
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
        if self.highlight_cadence_reference_count < 1:
            raise ValueError("highlight cadence needs at least one reference")
        if self.build_duration_reference_count < 1:
            raise ValueError("build duration policy needs at least one reference")
        if self.bridge_duration_reference_count < 1:
            raise ValueError("bridge duration policy needs at least one reference")
        if self.minimum_closeup_fraction > self.minimum_nonwide_fraction:
            raise ValueError("close-up floor cannot exceed non-wide floor")
        if (
            self.minimum_nonwide_proxy_face_height_fraction
            >= self.minimum_closeup_proxy_face_height_fraction
        ):
            raise ValueError(
                "non-wide proxy face threshold must remain below close-up threshold"
            )
        if (
            self.pooled_product_minimum_closeup_fraction + _EPSILON
            < self.minimum_closeup_fraction
            or self.pooled_product_minimum_nonwide_fraction + _EPSILON
            < self.minimum_nonwide_fraction
        ):
            raise ValueError(
                "pooled P25 product minimum cannot be below local P10 floor"
            )
        if (
            self.preferred_closeup_fraction + _EPSILON
            < self.pooled_product_minimum_closeup_fraction
            or self.preferred_nonwide_fraction + _EPSILON
            < self.pooled_product_minimum_nonwide_fraction
        ):
            raise ValueError(
                "P50 preferred target cannot be below P25 product minimum"
            )
        if (
            self.preferred_maximum_consecutive_wide_run_seconds
            > self.maximum_consecutive_wide_run_seconds + _EPSILON
        ):
            raise ValueError(
                "P75 preferred wide-run maximum cannot exceed hard maximum"
            )
        if (
            self.chorus_role_preferred_closeup_fraction + _EPSILON
            < self.chorus_role_pooled_product_minimum_closeup_fraction
            or self.chorus_role_preferred_nonwide_fraction + _EPSILON
            < self.chorus_role_pooled_product_minimum_nonwide_fraction
        ):
            raise ValueError("chorus role P50 cannot be below role-pooled P25")
        if (
            self.chorus_role_preferred_maximum_consecutive_wide_run_seconds
            > self.maximum_consecutive_wide_run_seconds + _EPSILON
        ):
            raise ValueError("chorus role P75 wide target cannot exceed P100")
        for role in ("build", "bridge"):
            local_close = getattr(self, f"{role}_minimum_closeup_fraction")
            local_nonwide = getattr(self, f"{role}_minimum_nonwide_fraction")
            pooled_close = getattr(
                self, f"{role}_pooled_product_minimum_closeup_fraction"
            )
            pooled_nonwide = getattr(
                self, f"{role}_pooled_product_minimum_nonwide_fraction"
            )
            preferred_close = getattr(self, f"{role}_preferred_closeup_fraction")
            preferred_nonwide = getattr(self, f"{role}_preferred_nonwide_fraction")
            if local_close is not None and pooled_close + _EPSILON < local_close:
                raise ValueError(f"{role} pooled close P25 cannot be below P10")
            if local_nonwide is not None and pooled_nonwide + _EPSILON < local_nonwide:
                raise ValueError(f"{role} pooled nonwide P25 cannot be below P10")
            if (
                preferred_close + _EPSILON < pooled_close
                or preferred_nonwide + _EPSILON < pooled_nonwide
            ):
                raise ValueError(f"{role} P50 target cannot be below P25")
            if (
                getattr(
                    self,
                    f"{role}_preferred_maximum_consecutive_wide_run_seconds",
                )
                > getattr(self, f"{role}_maximum_consecutive_wide_run_seconds")
                + _EPSILON
            ):
                raise ValueError(f"{role} P75 wide target cannot exceed P100")
        if (
            self.highlight_preferred_source_switches_per_minute + _EPSILON
            < self.highlight_product_minimum_source_switches_per_minute
        ):
            raise ValueError("highlight preferred cadence cannot be below minimum")
        provenance_values = (
            self.policy_profile_analysis_basis,
            self.policy_profile_schema_version,
            self.policy_profile_path,
            self.policy_profile_file_sha256,
            self.policy_profile_content_sha256,
            self.policy_fingerprint_sha256,
        )
        if any(value is not None for value in provenance_values):
            if any(value is None for value in provenance_values):
                raise ValueError("policy provenance identity must be complete")
            assert self.policy_profile_schema_version is not None
            if (
                isinstance(self.policy_profile_schema_version, bool)
                or not isinstance(self.policy_profile_schema_version, int)
                or self.policy_profile_schema_version < 1
            ):
                raise ValueError("policy profile schema version must be positive")
            for name in ("policy_profile_analysis_basis", "policy_profile_path"):
                value = getattr(self, name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name} cannot be blank")
            if not Path(str(self.policy_profile_path)).is_absolute():
                raise ValueError("policy profile path must be absolute")
            for name in (
                "policy_profile_file_sha256",
                "policy_profile_content_sha256",
                "policy_fingerprint_sha256",
            ):
                value = getattr(self, name)
                if not (
                    isinstance(value, str)
                    and len(value) == 64
                    and all(character in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(f"{name} must be lowercase SHA-256")
        identity_bound = all(value is not None for value in provenance_values)
        close_burst_values = (
            self.minimum_close_burst_shot_count,
            self.minimum_close_burst_duration_seconds,
            self.minimum_close_burst_source_switch_fraction,
            self.close_burst_event_reference_count,
            self.close_burst_policy_binding_sha256,
        )
        if identity_bound and any(value is None for value in close_burst_values):
            raise ValueError(
                "hash-bound montage policy is missing its close-burst soft "
                "preference binding"
            )
        if not identity_bound and any(value is not None for value in close_burst_values):
            raise ValueError(
                "close-burst soft preference requires complete hash-bound "
                "montage policy identity"
            )
        if identity_bound:
            assert self.minimum_close_burst_shot_count is not None
            assert self.minimum_close_burst_duration_seconds is not None
            assert self.minimum_close_burst_source_switch_fraction is not None
            assert self.close_burst_event_reference_count is not None
            assert self.close_burst_policy_binding_sha256 is not None
            assert self.policy_fingerprint_sha256 is not None
            expected_binding = close_burst_policy_binding_sha256(
                profile_identity={
                    "analysis_basis": self.policy_profile_analysis_basis,
                    "schema_version": self.policy_profile_schema_version,
                    "path": self.policy_profile_path,
                    "file_sha256": self.policy_profile_file_sha256,
                    "content_sha256": self.policy_profile_content_sha256,
                },
                policy_fingerprint_sha256=self.policy_fingerprint_sha256,
                minimum_shot_count=self.minimum_close_burst_shot_count,
                minimum_duration_seconds=(
                    self.minimum_close_burst_duration_seconds
                ),
                minimum_source_switch_fraction=(
                    self.minimum_close_burst_source_switch_fraction
                ),
                event_reference_count=self.close_burst_event_reference_count,
            )
            if self.close_burst_policy_binding_sha256 != expected_binding:
                raise ValueError(
                    "close-burst soft preference binding SHA-256 mismatch"
                )
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


def _close_only_multi_source_burst_evidence(
    runs: Sequence[Mapping[str, Any]],
    *,
    section_start: float,
    section_end: float,
    motion_segment_indices: set[int],
    config: RenderedMontageGrammarConfig,
) -> dict[str, Any]:
    """Recompute maximal close-only family-switch runs from rendered choices."""

    active = config.close_burst_policy_binding_sha256 is not None
    if not active:
        return {
            "contract": CLOSE_BURST_EVENT_EVIDENCE_CONTRACT,
            "policy_binding_active": False,
            "policy_binding": None,
            "candidate_event_count": 0,
            "qualified_event_count": 0,
            "events": [],
            "saturated_soft_preference_score": None,
            "absence_is_hard_failure": False,
            "hard_gate": False,
        }
    assert config.minimum_close_burst_shot_count is not None
    assert config.minimum_close_burst_duration_seconds is not None
    assert config.minimum_close_burst_source_switch_fraction is not None
    assert config.close_burst_event_reference_count is not None
    assert config.policy_fingerprint_sha256 is not None
    policy_binding = close_burst_policy_binding_payload(
        profile_identity={
            "analysis_basis": config.policy_profile_analysis_basis,
            "schema_version": config.policy_profile_schema_version,
            "path": config.policy_profile_path,
            "file_sha256": config.policy_profile_file_sha256,
            "content_sha256": config.policy_profile_content_sha256,
        },
        policy_fingerprint_sha256=config.policy_fingerprint_sha256,
        minimum_shot_count=config.minimum_close_burst_shot_count,
        minimum_duration_seconds=config.minimum_close_burst_duration_seconds,
        minimum_source_switch_fraction=(
            config.minimum_close_burst_source_switch_fraction
        ),
        event_reference_count=config.close_burst_event_reference_count,
    )
    policy_binding["binding_sha256"] = config.close_burst_policy_binding_sha256

    clipped: list[dict[str, Any]] = []
    for raw in runs:
        start = max(section_start, float(raw["start_seconds"]))
        end = min(section_end, float(raw["end_seconds"]))
        if end - start <= _EPSILON:
            continue
        clipped.append(
            {
                **dict(raw),
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": end - start,
            }
        )

    maximal_close_runs: list[tuple[dict[str, Any], ...]] = []
    current: list[dict[str, Any]] = []
    for run in clipped:
        if run["shot_class"] in _CLOSE:
            current.append(run)
            continue
        if current:
            maximal_close_runs.append(tuple(current))
            current = []
    if current:
        maximal_close_runs.append(tuple(current))

    events: list[dict[str, Any]] = []
    for close_run in maximal_close_runs:
        switch_positions = tuple(
            position
            for position, (left, right) in enumerate(
                zip(close_run, close_run[1:]),
                start=1,
            )
            if left["derivative_family_id"] != right["derivative_family_id"]
        )
        if not switch_positions:
            continue
        shot_count = len(close_run)
        duration_seconds = sum(float(item["duration_seconds"]) for item in close_run)
        source_switch_count = len(switch_positions)
        source_switch_fraction = source_switch_count / max(1, shot_count - 1)
        qualified = bool(
            shot_count + _EPSILON >= config.minimum_close_burst_shot_count
            and duration_seconds + _EPSILON
            >= config.minimum_close_burst_duration_seconds
            and source_switch_fraction + _EPSILON
            >= config.minimum_close_burst_source_switch_fraction
        )
        accented_switch_count = sum(
            int(
                int(close_run[position - 1]["last_segment_index"])
                in motion_segment_indices
                or int(close_run[position]["first_segment_index"])
                in motion_segment_indices
            )
            for position in switch_positions
        )
        events.append(
            {
                "first_segment_index": int(close_run[0]["first_segment_index"]),
                "last_segment_index": int(close_run[-1]["last_segment_index"]),
                "start_seconds": float(close_run[0]["start_seconds"]),
                "end_seconds": float(close_run[-1]["end_seconds"]),
                "shot_count": shot_count,
                "duration_seconds": duration_seconds,
                "source_switch_count": source_switch_count,
                "source_switch_fraction": source_switch_fraction,
                "derivative_family_ids": [
                    int(item["derivative_family_id"]) for item in close_run
                ],
                "editorial_motion_accent_switch_count": accented_switch_count,
                "editorial_motion_is_choreography_motion_proof": False,
                "qualified": qualified,
            }
        )
    qualified_count = sum(int(item["qualified"]) for item in events)
    return {
        "contract": CLOSE_BURST_EVENT_EVIDENCE_CONTRACT,
        "policy_binding_active": True,
        "policy_binding": policy_binding,
        "candidate_event_count": len(events),
        "qualified_event_count": qualified_count,
        "events": events,
        "saturated_soft_preference_score": 1.0 if qualified_count else 0.0,
        "saturation": "one_or_more_qualified_events",
        "absence_is_hard_failure": False,
        "hard_gate": False,
        "medium_to_close_counts_as_close_only": False,
        "constituent_switch_motion_preference_remains_separate": True,
        "temporal_direction_proven": False,
    }


def _overlap(left: float, right: float, start: float, end: float) -> float:
    return max(0.0, min(right, end) - max(left, start))


def _transform_has_within_segment_motion(value: Any, name: str) -> bool:
    """Return whether a track changes while one rendered shot is on screen.

    A constant non-identity Dynamic Crop track is safe shot framing, not an
    editorial motion accent.  In particular, different constant crops on two
    sides of a hard cut must never satisfy the highlight motion requirement.
    The release-bound Dynamic Crop v3 authority applies a stricter model-logit
    and renderer-sampled physical-range gate later; this structural review
    first ensures that track presence or a cut-time reframe cannot qualify on
    its own.
    """

    transform = _mapping(value, name)
    if transform.get("keyframes") is None:
        return False
    keyframes = _sequence(transform.get("keyframes"), f"{name}.keyframes")
    if not keyframes:
        return False
    states: list[tuple[float, float, float, float]] = []
    for index, raw in enumerate(keyframes):
        keyframe = _mapping(raw, f"{name}.keyframes[{index}]")
        zoom = _finite(keyframe.get("zoom"), "editorial zoom")
        center_x = _finite(keyframe.get("center_x"), "editorial center_x")
        center_y = _finite(keyframe.get("center_y"), "editorial center_y")
        roll = _finite(keyframe.get("roll_degrees"), "editorial roll")
        states.append((zoom, center_x, center_y, roll))
    first = states[0]
    return any(
        any(
            not math.isclose(left, right, abs_tol=_EPSILON)
            for left, right in zip(first, state, strict=True)
        )
        for state in states[1:]
    )


def _validated_dynamic_crop_v3_motion_segments(
    *,
    value: object,
    render_plan: Mapping[str, Any],
) -> set[int]:
    """Restore exact v3 evidence and bind it to the complete final plan."""

    try:
        if isinstance(value, DynamicCropMotionTimelineEvidenceV3):
            # Reparse typed inputs as well as serialized inputs so nested
            # self-hashes and the frozen 8--35% budget are checked here.
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
        raise ValueError(
            f"Dynamic Crop v3 motion evidence is invalid: {exc}"
        ) from exc

    segments = _sequence(render_plan.get("segments"), "render_plan.segments")
    if evidence.render_plan_content_sha256 != _canonical_sha256(render_plan):
        raise ValueError(
            "Dynamic Crop v3 motion evidence belongs to another RenderPlan"
        )
    if evidence.segment_count != len(segments):
        raise ValueError(
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
        raise ValueError(
            "Dynamic Crop v3 motion authority is not production-consistent"
        )
    return {
        item.segment_index
        for item in evidence.segment_evidence
        if item.motion_event_applied
    }


def _motion_accent_segment_indices(
    analysis: Mapping[str, Any],
    *,
    render_plan: Mapping[str, Any],
) -> set[int]:
    dynamic_crop_version = analysis.get("dynamic_crop_version")
    if dynamic_crop_version is not None and (
        isinstance(dynamic_crop_version, bool)
        or not isinstance(dynamic_crop_version, int)
        or dynamic_crop_version not in {2, 3}
    ):
        raise ValueError("dynamic_crop_version must be absent, 2, or 3")
    dynamic_crop_motion_v3 = analysis.get("dynamic_crop_motion_v3")
    if dynamic_crop_version == 3:
        if dynamic_crop_motion_v3 is None:
            raise ValueError(
                "Dynamic Crop v3 requires typed motion timeline evidence"
            )
        return _validated_dynamic_crop_v3_motion_segments(
            value=dynamic_crop_motion_v3,
            render_plan=render_plan,
        )
    if dynamic_crop_motion_v3 is not None:
        raise ValueError(
            "Dynamic Crop v3 motion evidence lacks its version marker"
        )

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
                    and _transform_has_within_segment_motion(
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
                    and _transform_has_within_segment_motion(
                        raw.get("editorial_transform"),
                        f"render.segments[{position}].editorial_transform",
                    )
                ):
                    result.add(position)
    return result


def _normalized_face_geometry(
    value: Any,
) -> tuple[int, tuple[float, float, float, float]] | None:
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
        return None
    x, y, width, height = tuple(float(item) for item in bbox)
    if not (
        0.0 <= x <= 1.0
        and 0.0 <= y <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
        and x + width <= 1.000001
        and y + height <= 1.000001
    ):
        return None
    return int(face_count), (x, y, width, height)


def _proxy_geometry_shot_class(
    value: Any,
    config: RenderedMontageGrammarConfig,
) -> str:
    geometry = value if isinstance(value, Mapping) else {}
    parsed = _normalized_face_geometry(value)
    framing_face = (parsed[0], parsed[1][3]) if parsed is not None else None
    if framing_face is None:
        count = geometry.get("face_count")
        scale = geometry.get("framing_face_scale")
        backend = geometry.get("face_backend")
        if (
            isinstance(count, int) and not isinstance(count, bool) and count > 0
            and isinstance(scale, (int, float)) and not isinstance(scale, bool)
            and math.isfinite(float(scale)) and 0.0 < float(scale) <= 1.0
            and isinstance(backend, str) and backend.strip() not in {"", "none"}
            and geometry.get("shot_scale_basis") == "dominant_face_height"
        ):
            # Saturated scale is a lower bound on height; it is already above
            # the existing close-up threshold. Never construct a focal bbox.
            framing_face = (count, float(scale) * 0.42)
    if framing_face is not None:
        face_count, height = framing_face
        # Physical framing and identity are separate observations. Multiple
        # large faces can be a tight composition without a unique focal bbox.
        # Counts must not overwrite a decoded size measurement as "wide".
        if height >= config.minimum_closeup_proxy_face_height_fraction:
            return "close"
        if height >= config.minimum_nonwide_proxy_face_height_fraction:
            return "medium"
    raw_formation = geometry.get("formation_boxes_xywh")
    reported_people = geometry.get("detected_person_count")
    if (
        isinstance(raw_formation, (list, tuple))
        and len(raw_formation) >= 3
        and isinstance(reported_people, int)
        and not isinstance(reported_people, bool)
        and reported_people >= 3
        and all(
            isinstance(raw, (list, tuple))
            and len(raw) == 4
            and all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                for item in raw
            )
            and 0.0 <= float(raw[0]) <= 1.0
            and 0.0 <= float(raw[1]) <= 1.0
            and 0.0 < float(raw[2]) <= 1.0
            and 0.0 < float(raw[3]) <= 1.0
            and float(raw[0]) + float(raw[2]) <= 1.000001
            and float(raw[1]) + float(raw[3]) <= 1.000001
            for raw in raw_formation
        )
    ):
        # Use the same geometry rule as source analysis, not its serialized
        # shot label. These shape-only views supply no face identity, detection
        # confidence or invented pose. Keep the close-face priority above.
        from .visual_score import _cropped_foreground_group_scale

        if _cropped_foreground_group_scale(
            tuple(SimpleNamespace(bbox_xywh=tuple(float(item) for item in raw))
                  for raw in raw_formation)
        ) is not None:
            return "medium"
        return "wide"
    if framing_face is not None:
        face_count, height = framing_face
        # Reuse visual_score's face thresholds exactly.
        if height < config.minimum_nonwide_proxy_face_height_fraction:
            return "wide"
        if height < config.minimum_closeup_proxy_face_height_fraction:
            return "medium"
        return "close"

    raw_boxes = geometry.get("formation_boxes_xywh")
    boxes: list[tuple[float, float, float, float]] = []
    if isinstance(raw_boxes, (list, tuple)):
        for raw in raw_boxes:
            if not isinstance(raw, (list, tuple)) or len(raw) != 4:
                return "unknown"
            if not all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                for item in raw
            ):
                return "unknown"
            box = tuple(float(item) for item in raw)
            if not (
                0.0 <= box[0] <= 1.0
                and 0.0 <= box[1] <= 1.0
                and 0.0 < box[2] <= 1.0
                and 0.0 < box[3] <= 1.0
                and box[0] + box[2] <= 1.000001
                and box[1] + box[3] <= 1.000001
            ):
                return "unknown"
            boxes.append(box)
    person_count = geometry.get("detected_person_count")
    if (
        isinstance(person_count, bool)
        or not isinstance(person_count, int)
        or person_count < 0
    ):
        person_count = len(boxes)
    if boxes:
        heights = sorted(box[3] for box in boxes)
        median_height = heights[len(heights) // 2]
        union_left = min(box[0] for box in boxes)
        union_top = min(box[1] for box in boxes)
        union_right = max(box[0] + box[2] for box in boxes)
        union_bottom = max(box[1] + box[3] for box in boxes)
        union_width = union_right - union_left
        boundary_clipped = union_top <= 0.035 or union_bottom >= 0.965
        if max(person_count, len(boxes)) >= 3:
            return "wide"
        if (
            len(boxes) == 1
            and boundary_clipped
            and median_height >= 0.68
            and union_width <= 0.58
        ):
            return "close"
        if median_height >= 0.43 and union_width <= 0.72:
            return "medium"
        return "wide"

    silhouette = geometry.get("silhouette_bbox_xywh")
    confidence = geometry.get("silhouette_confidence")
    if (
        isinstance(silhouette, (list, tuple))
        and len(silhouette) == 4
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in silhouette
        )
        and isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and float(confidence) >= 0.72
    ):
        _left, top, width, height = tuple(float(item) for item in silhouette)
        boundary_clipped = top <= 0.035 or top + height >= 0.965
        if boundary_clipped and height >= 0.70 and width <= 0.46:
            return "close"
        if height >= 0.54 and width <= 0.54:
            return "medium"
        return "wide"

    # A face miss without corroborating body geometry is unknown, never proof
    # of a wide shot.  This keeps decoded QA aligned with visual_score's
    # explicit ``unknown_no_subject_geometry`` basis.
    return "unknown"


def _geometry_proves_closeup(
    value: Any, config: RenderedMontageGrammarConfig
) -> bool:
    return _proxy_geometry_shot_class(value, config) == "close"


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
            "master_audio_start": master_audio_start,
            "output_frame_count": expected_output_frame_count,
            "segment_frame_counts": list(frame_counts),
            "clock_mapping": (
                "canonical_master_minus_master_audio_start_to_output_master"
            ),
            "source_index_binding": source_binding,
            "boundary_index_domain": _RENDER_PLAN_BOUNDARY_INDEX_DOMAIN,
        },
        tuple(bound_boundaries),
    )


def _automatic_ive_product_scope(analysis: Mapping[str, Any]) -> bool:
    request = analysis.get("request")
    if not isinstance(request, Mapping):
        return False
    automatic_ive = bool(
        str(request.get("group") or "").strip().upper() == "IVE"
        and str(request.get("edit_mode") or "").strip().lower() == "group"
        and str(request.get("source_mode") or "").strip().lower() == "automatic"
    )
    if not automatic_ive:
        return False
    role = analysis.get("artifact_role")
    exposure = analysis.get("app_result_exposure")
    explicitly_hidden = bool(
        role in {"diagnostic", "comparison"}
        and isinstance(exposure, Mapping)
        and exposure.get("video_output_exposed") is False
        and exposure.get("delivery_completion_message_allowed") is False
    )
    return not explicitly_hidden


def _serialized_render_timeline_binding_payload(
    plan: Mapping[str, Any],
    *,
    frame_counts: Sequence[int],
    source_content_sha256: Sequence[str],
) -> dict[str, Any]:
    sources = _sequence(plan.get("sources"), "render_plan.sources")
    segments = _sequence(plan.get("segments"), "render_plan.segments")
    transitions = _sequence(plan.get("transitions"), "render_plan.transitions")
    if len(frame_counts) != len(segments):
        raise ValueError("render plan frame-count binding is inconsistent")
    content_digests = tuple(source_content_sha256)
    if len(content_digests) != len(sources) or any(
        not (
            isinstance(value, str)
            and len(value) == 64
            and all(
                character in "0123456789abcdef" for character in value
            )
        )
        for value in content_digests
    ):
        raise ValueError("render plan source content SHA-256 binding is invalid")
    return {
        "source_content_sha256": list(content_digests),
        "master_audio_start": _finite(
            plan.get("master_audio_start"), "render_plan.master_audio_start"
        ),
        "fps": _finite(plan.get("fps"), "render_plan.fps"),
        "output_frame_count": _nonnegative_integer(
            plan.get("output_frame_count"), "render_plan.output_frame_count"
        ),
        "segments": [
            {
                "segment_index": index,
                "source_index": _nonnegative_integer(
                    _mapping(raw, f"render segment {index}").get("source_index"),
                    f"render segment {index} source_index",
                ),
                "master_start": _finite(
                    _mapping(raw, f"render segment {index}").get("master_start"),
                    f"render segment {index} master_start",
                ),
                "duration": _finite(
                    _mapping(raw, f"render segment {index}").get("duration"),
                    f"render segment {index} duration",
                ),
                "source_start": _finite(
                    _mapping(raw, f"render segment {index}").get("source_start"),
                    f"render segment {index} source_start",
                ),
                "source_duration": _mapping(
                    raw, f"render segment {index}"
                ).get("source_duration"),
                "effective_source_duration": _finite(
                    _mapping(raw, f"render segment {index}").get(
                        "effective_source_duration"
                    ),
                    f"render segment {index} effective_source_duration",
                ),
                "crop": _mapping(raw, f"render segment {index}").get("crop"),
                "pan_crop": _mapping(raw, f"render segment {index}").get(
                    "pan_crop"
                ),
                "editorial_transform": _mapping(
                    raw, f"render segment {index}"
                ).get("editorial_transform"),
                "look_adjustment": _mapping(
                    raw, f"render segment {index}"
                ).get("look_adjustment"),
                "boundary_look_track": _mapping(
                    raw, f"render segment {index}"
                ).get("boundary_look_track"),
                "output_frame_count": int(frame_counts[index]),
            }
            for index, raw in enumerate(segments)
        ],
        "transitions": [
            {
                "boundary_index": index,
                "kind": str(
                    _mapping(raw, f"render transition {index}").get("kind")
                    or ""
                ),
                "frame_count": _nonnegative_integer(
                    _mapping(raw, f"render transition {index}").get(
                        "frame_count"
                    ),
                    f"render transition {index} frame_count",
                ),
            }
            for index, raw in enumerate(transitions)
        ],
    }


def _segment_sample_offsets(frame_count: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                min(
                    frame_count - 1,
                    max(
                        0,
                        int(
                            math.floor(
                                (frame_count - 1) * fraction + 0.5
                            )
                        ),
                    ),
                )
                for fraction in SEGMENT_INTERIOR_SAMPLE_FRACTIONS
            }
        )
    )


def _expected_segment_sampling_schedule(
    *,
    plan: Mapping[str, Any],
    frame_counts: Sequence[int],
) -> tuple[dict[str, int], ...]:
    segments = _sequence(plan.get("segments"), "render_plan.segments")
    schedule: list[dict[str, int]] = []
    segment_frame_start = 0
    for segment_index, (raw_segment, frame_count) in enumerate(
        zip(segments, frame_counts, strict=True)
    ):
        segment = _mapping(raw_segment, f"render segment {segment_index}")
        source_index = _nonnegative_integer(
            segment.get("source_index"),
            f"render segment {segment_index} source_index",
        )
        offsets = _segment_sample_offsets(int(frame_count))
        represented_starts = [0]
        represented_starts.extend(
            (left + right + 1) // 2
            for left, right in zip(offsets, offsets[1:])
        )
        represented_ends = [*represented_starts[1:], int(frame_count)]
        for offset, represented_start, represented_end in zip(
            offsets,
            represented_starts,
            represented_ends,
            strict=True,
        ):
            schedule.append(
                {
                    "segment_index": segment_index,
                    "source_index": source_index,
                    "segment_frame_offset": offset,
                    "frame_index": segment_frame_start + offset,
                    "represented_frame_start": (
                        segment_frame_start + represented_start
                    ),
                    "represented_frame_end": (
                        segment_frame_start + represented_end
                    ),
                    "represented_frame_count": represented_end - represented_start,
                }
            )
        segment_frame_start += int(frame_count)
    return tuple(schedule)


def _review_segment_interior_geometry(
    *,
    choices: Sequence[Any],
    rendered_proxy_review: Mapping[str, Any],
    render_plan: Mapping[str, Any],
    render_plan_binding: Mapping[str, Any],
    config: RenderedMontageGrammarConfig,
    required: bool,
) -> tuple[dict[str, Any] | None, tuple[dict[str, Any], ...]]:
    raw_evidence = rendered_proxy_review.get("segment_interior_geometry")
    if raw_evidence is None:
        if required:
            return (
                {
                    "contract": RENDERED_SEGMENT_INTERIOR_GEOMETRY_REVIEW_CONTRACT,
                    "required": True,
                    "available": False,
                    "passed": False,
                    "failure_codes": [
                        "rendered_segment_interior_geometry_evidence_missing"
                    ],
                },
                (),
            )
        return None, ()
    evidence = dict(_mapping(raw_evidence, "segment interior geometry evidence"))
    evidence_sha256 = evidence.pop("evidence_sha256", None)
    if not (
        isinstance(evidence_sha256, str)
        and len(evidence_sha256) == 64
        and all(character in "0123456789abcdef" for character in evidence_sha256)
    ):
        raise ValueError("segment interior geometry evidence SHA-256 is invalid")
    if _canonical_sha256(evidence) != evidence_sha256:
        raise ValueError("segment interior geometry evidence SHA-256 mismatch")
    if (
        evidence.get("schema_version")
        != SEGMENT_INTERIOR_GEOMETRY_SCHEMA_VERSION
        or evidence.get("analysis_basis")
        != SEGMENT_INTERIOR_GEOMETRY_ANALYSIS_BASIS
        or evidence.get("contract") != SEGMENT_INTERIOR_GEOMETRY_CONTRACT
    ):
        raise ValueError("segment interior geometry evidence contract changed")
    if evidence.get("source_index_domain") != _RENDER_PLAN_SOURCE_INDEX_DOMAIN:
        raise ValueError("segment interior geometry source domain changed")
    if evidence.get("output_frame_index_domain") != (
        "decoded_proxy_output_frame_index"
    ):
        raise ValueError("segment interior geometry frame domain changed")
    sampling_policy = _mapping(
        evidence.get("sampling_policy"), "segment interior sampling policy"
    )
    if tuple(sampling_policy.get("sample_fractions") or ()) != (
        SEGMENT_INTERIOR_SAMPLE_FRACTIONS
    ):
        raise ValueError("segment interior sampling fractions changed")
    if sampling_policy.get("rounding") != (
        "nearest_output_frame_half_up_unique"
    ):
        raise ValueError("segment interior sampling rounding changed")

    serialized_proxy_identity = _mapping(
        rendered_proxy_review.get("proxy"), "rendered proxy identity"
    )
    proxy_identity = {
        field: serialized_proxy_identity[field]
        for field in _STABLE_PROXY_IDENTITY_FIELDS
        if field in serialized_proxy_identity
    }
    proxy_content_sha256 = proxy_identity.get("content_sha256")
    if not (
        isinstance(proxy_content_sha256, str)
        and len(proxy_content_sha256) == 64
        and all(
            character in "0123456789abcdef"
            for character in proxy_content_sha256
        )
    ):
        raise ValueError("rendered proxy content SHA-256 is invalid")
    if evidence.get("proxy_identity") != proxy_identity:
        raise ValueError("segment geometry proxy identity differs from proxy review")
    if evidence.get("proxy_identity_sha256") != _canonical_sha256(proxy_identity):
        raise ValueError("segment geometry proxy identity hash mismatch")
    frame_counts = tuple(
        _nonnegative_integer(value, "bound segment frame count")
        for value in _sequence(
            render_plan_binding.get("segment_frame_counts"),
            "bound segment frame counts",
        )
    )
    plan_sources = _sequence(render_plan.get("sources"), "render_plan.sources")
    source_content_sha256 = tuple(
        verified_file_content_sha256(Path(str(path))) for path in plan_sources
    )
    timeline_binding = _serialized_render_timeline_binding_payload(
        render_plan,
        frame_counts=frame_counts,
        source_content_sha256=source_content_sha256,
    )
    if evidence.get("render_timeline_binding_sha256") != _canonical_sha256(
        timeline_binding
    ):
        raise ValueError("segment geometry RenderPlan binding hash mismatch")
    expected_schedule = _expected_segment_sampling_schedule(
        plan=render_plan,
        frame_counts=frame_counts,
    )
    if evidence.get("sampling_schedule_sha256") != _canonical_sha256(
        expected_schedule
    ):
        raise ValueError("segment geometry sampling schedule hash mismatch")

    raw_segments = _sequence(
        evidence.get("segments"), "segment interior geometry segments"
    )
    if len(raw_segments) != len(choices) or evidence.get("segment_count") != len(
        choices
    ):
        raise ValueError("segment interior geometry segment count mismatch")
    expected_by_segment: defaultdict[int, list[Mapping[str, int]]] = defaultdict(
        list
    )
    for item in expected_schedule:
        expected_by_segment[int(item["segment_index"])].append(item)
    cells: list[dict[str, Any]] = []
    segment_records: list[dict[str, Any]] = []
    unknown_sample_count = 0
    provider_failure_sample_count = 0
    close_claim_contradiction_count = 0
    nonwide_claim_wide_contradiction_count = 0
    observed_sample_count = 0
    shot_scale_basis_counts: dict[str, int] = {}
    for segment_index, (raw_segment, raw_choice) in enumerate(
        zip(raw_segments, choices, strict=True)
    ):
        segment = _mapping(
            raw_segment, f"segment interior geometry segment {segment_index}"
        )
        choice = _mapping(raw_choice, f"visual choice {segment_index}")
        expected_samples = expected_by_segment[segment_index]
        frame_count = int(frame_counts[segment_index])
        frame_start = sum(frame_counts[:segment_index])
        render_segment = _mapping(
            _sequence(render_plan.get("segments"), "render_plan.segments")[
                segment_index
            ],
            f"render segment {segment_index}",
        )
        source_index = _nonnegative_integer(
            render_segment.get("source_index"),
            f"render segment {segment_index} source_index",
        )
        expected_segment_identity = {
            "segment_index": segment_index,
            "source_index": source_index,
            "source_content_sha256": source_content_sha256[source_index],
            "segment_frame_start": frame_start,
            "segment_frame_end": frame_start + frame_count,
            "segment_frame_count": frame_count,
        }
        for field, expected_value in expected_segment_identity.items():
            if segment.get(field) != expected_value:
                raise ValueError(
                    f"segment interior geometry {field} differs from RenderPlan"
                )
        raw_samples = _sequence(
            segment.get("samples"),
            f"segment interior geometry segment {segment_index} samples",
        )
        if len(raw_samples) != len(expected_samples):
            raise ValueError("segment interior geometry sample count mismatch")
        choice_start = _finite(
            choice.get("canonical_master_start"), "visual choice start"
        )
        choice_end = _finite(
            choice.get("canonical_master_end"), "visual choice end"
        )
        declared_class = _shot_class(choice.get("shot_type"))
        classifications: list[str] = []
        for sample_position, (raw_sample, expected) in enumerate(
            zip(raw_samples, expected_samples, strict=True)
        ):
            sample = _mapping(
                raw_sample,
                (
                    f"segment interior geometry segment {segment_index} "
                    f"sample {sample_position}"
                ),
            )
            for field in (
                "segment_frame_offset",
                "frame_index",
                "represented_frame_start",
                "represented_frame_end",
                "represented_frame_count",
            ):
                if sample.get(field) != expected[field]:
                    raise ValueError(
                        f"segment interior geometry {field} differs from schedule"
                    )
            failed = sample.get("geometry_provider_failed")
            if not isinstance(failed, bool):
                raise ValueError("segment geometry provider failure flag is invalid")
            proxy_class = (
                "unknown"
                if failed
                else _proxy_geometry_shot_class(
                    sample.get("subject_geometry"), config
                )
            )
            raw_geometry = (
                sample.get("subject_geometry")
                if isinstance(sample.get("subject_geometry"), Mapping)
                else {}
            )
            raw_basis = raw_geometry.get("shot_scale_basis")
            shot_scale_basis = (
                raw_basis.strip()
                if isinstance(raw_basis, str) and raw_basis.strip()
                else "legacy_geometry_without_basis"
            )
            shot_scale_basis_counts[shot_scale_basis] = (
                shot_scale_basis_counts.get(shot_scale_basis, 0) + 1
            )
            classifications.append(proxy_class)
            observed_sample_count += 1
            provider_failure_sample_count += int(failed)
            unknown_sample_count += int(proxy_class == "unknown")
            close_claim_contradiction_count += int(
                declared_class == "close" and proxy_class != "close"
            )
            nonwide_claim_wide_contradiction_count += int(
                declared_class in _NONWIDE and proxy_class == "wide"
            )
            local_start_frames = int(expected["represented_frame_start"]) - frame_start
            local_end_frames = int(expected["represented_frame_end"]) - frame_start
            duration = choice_end - choice_start
            cell_start = choice_start + duration * local_start_frames / frame_count
            cell_end = choice_start + duration * local_end_frames / frame_count
            cells.append(
                {
                    "segment_index": segment_index,
                    "source_index": source_index,
                    "sample_position": sample_position,
                    "sample_frame_index": int(expected["frame_index"]),
                    "start_seconds": cell_start,
                    "end_seconds": cell_end,
                    "duration_seconds": cell_end - cell_start,
                    "shot_class": proxy_class,
                    "shot_scale_basis": shot_scale_basis,
                    "declared_shot_class": declared_class,
                }
            )
        segment_records.append(
            {
                "segment_index": segment_index,
                "source_index": source_index,
                "declared_shot_class": declared_class,
                "sampled_proxy_shot_classes": classifications,
                "sample_count": len(classifications),
                "all_samples_known": all(
                    value != "unknown" for value in classifications
                ),
                "declared_nonwide_contradicted_by_wide_interior": bool(
                    declared_class in _NONWIDE and "wide" in classifications
                ),
            }
        )

    if evidence.get("sample_count") != observed_sample_count:
        raise ValueError("segment interior geometry total sample count mismatch")
    serialized_provider_failures = _nonnegative_integer(
        evidence.get("geometry_provider_failure_count"),
        "segment interior geometry provider failure count",
    )
    if serialized_provider_failures != provider_failure_sample_count:
        raise ValueError("segment geometry provider failure count mismatch")
    evidence_complete = evidence.get("complete")
    if not isinstance(evidence_complete, bool):
        raise ValueError("segment interior geometry completeness is invalid")
    expected_complete = provider_failure_sample_count == 0
    if evidence_complete != expected_complete:
        raise ValueError("segment interior geometry completeness mismatch")

    total_duration = sum(float(item["duration_seconds"]) for item in cells)
    known_duration = sum(
        float(item["duration_seconds"])
        for item in cells
        if item["shot_class"] != "unknown"
    )
    close_duration = sum(
        float(item["duration_seconds"])
        for item in cells
        if item["shot_class"] == "close"
    )
    nonwide_duration = sum(
        float(item["duration_seconds"])
        for item in cells
        if item["shot_class"] in _NONWIDE
    )
    maximum_wide_run = _maximum_consecutive_wide_run_seconds(
        cells,
        section_start=float(cells[0]["start_seconds"]),
        section_end=float(cells[-1]["end_seconds"]),
    )
    failure_codes: list[str] = []
    if not evidence_complete or provider_failure_sample_count:
        failure_codes.append("rendered_segment_interior_geometry_provider_failure")
    if unknown_sample_count:
        failure_codes.append("rendered_segment_interior_geometry_unclassifiable")
    if nonwide_claim_wide_contradiction_count:
        failure_codes.append(
            "rendered_nonwide_claim_contradicted_by_wide_interior_geometry"
        )
    summary_core: dict[str, Any] = {
        "contract": RENDERED_SEGMENT_INTERIOR_GEOMETRY_REVIEW_CONTRACT,
        "required": required,
        "available": True,
        "source_evidence_sha256": evidence_sha256,
        "source_sampling_schedule_sha256": evidence[
            "sampling_schedule_sha256"
        ],
        "source_render_timeline_binding_sha256": evidence[
            "render_timeline_binding_sha256"
        ],
        "classification_policy": {
            "basis": (
                "decoded_face_then_person_formation_then_silhouette_"
                "shot_geometry_hierarchy_v4"
            ),
            "measured_face_size_precedes_person_count": True,
            "framing_class_does_not_assign_identity": True,
            "minimum_nonwide_face_height_fraction": (
                config.minimum_nonwide_proxy_face_height_fraction
            ),
            "minimum_closeup_face_height_fraction": (
                config.minimum_closeup_proxy_face_height_fraction
            ),
            "no_face_person_geometry_policy": (
                "single_or_pair_bbox_medium_or_close_when_large_"
                "three_plus_people_formation_wide"
            ),
            "silhouette_fallback_policy": (
                "single_large_subject_nonwide_aggregate_or_distant_wide"
            ),
            "missing_face_and_body_geometry": (
                "unknown_fail_closed_when_required_not_observed_medium_or_wide"
            ),
            "wide_dominance_gate": (
                "whole_and_section_duration_plus_consecutive_run_with_"
                "formation_anchor_retained"
            ),
            "duration_semantics": (
                "nearest_sample_frame_cell_proxy_estimate_not_all_frame_claim"
            ),
        },
        "sample_count": observed_sample_count,
        "unknown_sample_count": unknown_sample_count,
        "provider_failure_sample_count": provider_failure_sample_count,
        "shot_scale_basis_counts": [
            {"basis": basis, "count": count}
            for basis, count in sorted(shot_scale_basis_counts.items())
        ],
        "close_claim_contradiction_count": close_claim_contradiction_count,
        "nonwide_claim_wide_contradiction_count": (
            nonwide_claim_wide_contradiction_count
        ),
        "segment_records": segment_records,
        "whole_song_observed": {
            "total_duration_seconds": total_duration,
            "known_duration_seconds": known_duration,
            "known_duration_fraction": (
                known_duration / total_duration if total_duration > 0.0 else 0.0
            ),
            "close_duration_seconds": close_duration,
            "nonwide_duration_seconds": nonwide_duration,
            "wide_duration_seconds": known_duration - nonwide_duration,
            "close_duration_fraction": (
                close_duration / known_duration if known_duration > 0.0 else 0.0
            ),
            "nonwide_duration_fraction": (
                nonwide_duration / known_duration if known_duration > 0.0 else 0.0
            ),
            "wide_duration_fraction": (
                (known_duration - nonwide_duration) / known_duration
                if known_duration > 0.0
                else 0.0
            ),
            "maximum_consecutive_wide_run_seconds": maximum_wide_run,
        },
        "failure_codes": failure_codes,
        "passed": not failure_codes,
    }
    return (
        {
            **summary_core,
            "review_evidence_sha256": _canonical_sha256(summary_core),
        },
        tuple(cells),
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


def _maximum_consecutive_wide_run_seconds(
    runs: Sequence[Mapping[str, Any]],
    *,
    section_start: float,
    section_end: float,
) -> float:
    maximum = 0.0
    current = 0.0
    for item in runs:
        duration = _overlap(
            float(item["start_seconds"]),
            float(item["end_seconds"]),
            section_start,
            section_end,
        )
        if duration <= _EPSILON:
            continue
        if item["shot_class"] in _WIDE:
            current += duration
            maximum = max(maximum, current)
        else:
            current = 0.0
    return maximum


def _symmetric_fraction_target_evidence(
    observed: float,
    target: float,
) -> dict[str, float | bool]:
    deviation = abs(observed - target)
    return {
        "target": target,
        "observed": observed,
        "absolute_deviation": deviation,
        "shortfall": max(0.0, target - observed),
        "excess": max(0.0, observed - target),
        "alignment_score": max(0.0, 1.0 - deviation),
        "hard_gate": False,
    }


def _wide_run_soft_target_evidence(
    observed: float,
    preferred_maximum: float,
    hard_maximum: float,
) -> dict[str, float | bool | str]:
    excess = max(0.0, observed - preferred_maximum)
    span = max(_EPSILON, hard_maximum - preferred_maximum)
    return {
        "target": preferred_maximum,
        "observed": observed,
        "upper_target_excess_seconds": excess,
        "alignment_score": (
            1.0 if excess <= _EPSILON else max(0.0, 1.0 - excess / span)
        ),
        "preference": "at_or_below",
        "saturated_at_or_below_target": True,
        "hard_gate": False,
    }


def _role_duration_thresholds(
    config: RenderedMontageGrammarConfig,
    role: str,
) -> dict[str, Any] | None:
    """Return primitive, JSON-safe thresholds for one gated editing role."""

    if role == "chorus_like":
        return {
            "role": role,
            "reference_count": config.highlight_cadence_reference_count,
            "local_minimum_closeup_fraction": config.minimum_closeup_fraction,
            "local_minimum_nonwide_fraction": config.minimum_nonwide_fraction,
            "pooled_product_minimum_closeup_fraction": (
                config.chorus_role_pooled_product_minimum_closeup_fraction
            ),
            "pooled_product_minimum_nonwide_fraction": (
                config.chorus_role_pooled_product_minimum_nonwide_fraction
            ),
            "preferred_closeup_fraction": (
                config.chorus_role_preferred_closeup_fraction
            ),
            "preferred_nonwide_fraction": (
                config.chorus_role_preferred_nonwide_fraction
            ),
            "maximum_consecutive_wide_run_seconds": (
                config.maximum_consecutive_wide_run_seconds
            ),
            "preferred_maximum_consecutive_wide_run_seconds": (
                config.chorus_role_preferred_maximum_consecutive_wide_run_seconds
            ),
        }
    if role not in {"build", "bridge_like"}:
        return None
    prefix = "build" if role == "build" else "bridge"
    return {
        "role": role,
        "reference_count": getattr(config, f"{prefix}_duration_reference_count"),
        "local_minimum_closeup_fraction": getattr(
            config, f"{prefix}_minimum_closeup_fraction"
        ),
        "local_minimum_nonwide_fraction": getattr(
            config, f"{prefix}_minimum_nonwide_fraction"
        ),
        "pooled_product_minimum_closeup_fraction": getattr(
            config, f"{prefix}_pooled_product_minimum_closeup_fraction"
        ),
        "pooled_product_minimum_nonwide_fraction": getattr(
            config, f"{prefix}_pooled_product_minimum_nonwide_fraction"
        ),
        "preferred_closeup_fraction": getattr(
            config, f"{prefix}_preferred_closeup_fraction"
        ),
        "preferred_nonwide_fraction": getattr(
            config, f"{prefix}_preferred_nonwide_fraction"
        ),
        "maximum_consecutive_wide_run_seconds": getattr(
            config, f"{prefix}_maximum_consecutive_wide_run_seconds"
        ),
        "preferred_maximum_consecutive_wide_run_seconds": getattr(
            config, f"{prefix}_preferred_maximum_consecutive_wide_run_seconds"
        ),
    }


def _role_duration_failure_code(role: str, suffix: str) -> str:
    prefix = "highlight" if role == "chorus_like" else role.removesuffix("_like")
    return f"{prefix}_{suffix}"


def _highlight_policy_provenance(
    config: RenderedMontageGrammarConfig,
) -> dict[str, Any]:
    identity_bound = config.policy_fingerprint_sha256 is not None
    profile_identity = (
        {
            "analysis_basis": config.policy_profile_analysis_basis,
            "schema_version": config.policy_profile_schema_version,
            "path": config.policy_profile_path,
            "file_sha256": config.policy_profile_file_sha256,
            "content_sha256": config.policy_profile_content_sha256,
        }
        if identity_bound
        else None
    )
    close_burst_binding = None
    if identity_bound:
        assert profile_identity is not None
        assert config.policy_fingerprint_sha256 is not None
        assert config.minimum_close_burst_shot_count is not None
        assert config.minimum_close_burst_duration_seconds is not None
        assert config.minimum_close_burst_source_switch_fraction is not None
        assert config.close_burst_event_reference_count is not None
        close_burst_binding = close_burst_policy_binding_payload(
            profile_identity=profile_identity,
            policy_fingerprint_sha256=config.policy_fingerprint_sha256,
            minimum_shot_count=config.minimum_close_burst_shot_count,
            minimum_duration_seconds=config.minimum_close_burst_duration_seconds,
            minimum_source_switch_fraction=(
                config.minimum_close_burst_source_switch_fraction
            ),
            event_reference_count=config.close_burst_event_reference_count,
        )
        close_burst_binding["binding_sha256"] = (
            config.close_burst_policy_binding_sha256
        )

    def metric(
        *,
        config_field: str,
        policy_member: str,
        value: float,
        statistic_path: str,
        quantile: float,
        hard_gate: bool,
        scope: str,
        comparison: str,
    ) -> dict[str, Any]:
        return {
            "config_field": config_field,
            "professional_montage_policy_member": policy_member,
            "value": value,
            "statistic_path": statistic_path,
            "quantile": quantile,
            "hard_gate": hard_gate,
            "scope": scope,
            "comparison": comparison,
        }

    return {
        "contract": HIGHLIGHT_AGGREGATE_CONTRACT,
        "profile_identity_bound": identity_bound,
        "profile_identity": profile_identity,
        "policy_fingerprint_sha256": config.policy_fingerprint_sha256,
        "close_burst_soft_policy_binding": close_burst_binding,
        "duration_evidence_contract": HIGHLIGHT_DURATION_EVIDENCE_CONTRACT,
        "wide_run_evidence_contract": HIGHLIGHT_WIDE_RUN_EVIDENCE_CONTRACT,
        "cadence_evidence_contract": HIGHLIGHT_CADENCE_EVIDENCE_CONTRACT,
        "role_conditioned_duration_evidence_contract": (
            ROLE_CONDITIONED_DURATION_EVIDENCE_CONTRACT
        ),
        "role_conditioned_duration_thresholds": {
            role: _role_duration_thresholds(config, role)
            for role in ("build", "chorus_like", "bridge_like")
        },
        "metrics": {
            "local_closeup_duration_minimum": metric(
                config_field="minimum_closeup_fraction",
                policy_member="minimum_close_duration_fraction",
                value=config.minimum_closeup_fraction,
                statistic_path=HIGHLIGHT_CLOSE_DURATION_STATISTIC_PATH,
                quantile=REFERENCE_FLOOR_QUANTILE,
                hard_gate=True,
                scope="each_strict_highlight",
                comparison="greater_than_or_equal",
            ),
            "local_nonwide_duration_minimum": metric(
                config_field="minimum_nonwide_fraction",
                policy_member="minimum_nonwide_duration_fraction",
                value=config.minimum_nonwide_fraction,
                statistic_path=HIGHLIGHT_NONWIDE_DURATION_STATISTIC_PATH,
                quantile=REFERENCE_FLOOR_QUANTILE,
                hard_gate=True,
                scope="each_strict_highlight",
                comparison="greater_than_or_equal",
            ),
            "pooled_closeup_duration_product_minimum": metric(
                config_field=(
                    "chorus_role_pooled_product_minimum_closeup_fraction"
                ),
                policy_member=(
                    "highlight_pooled_product_minimum_close_duration_fraction"
                ),
                value=(
                    config.chorus_role_pooled_product_minimum_closeup_fraction
                ),
                statistic_path=HIGHLIGHT_CLOSE_DURATION_STATISTIC_PATH,
                quantile=DURATION_PRODUCT_MINIMUM_QUANTILE,
                hard_gate=True,
                scope="pooled_strict_highlights",
                comparison="greater_than_or_equal",
            ),
            "pooled_nonwide_duration_product_minimum": metric(
                config_field=(
                    "chorus_role_pooled_product_minimum_nonwide_fraction"
                ),
                policy_member=(
                    "highlight_pooled_product_minimum_nonwide_duration_fraction"
                ),
                value=(
                    config.chorus_role_pooled_product_minimum_nonwide_fraction
                ),
                statistic_path=HIGHLIGHT_NONWIDE_DURATION_STATISTIC_PATH,
                quantile=DURATION_PRODUCT_MINIMUM_QUANTILE,
                hard_gate=True,
                scope="pooled_strict_highlights",
                comparison="greater_than_or_equal",
            ),
            "pooled_closeup_duration_preferred_target": metric(
                config_field="chorus_role_preferred_closeup_fraction",
                policy_member="preferred_close_duration_fraction",
                value=config.chorus_role_preferred_closeup_fraction,
                statistic_path=HIGHLIGHT_CLOSE_DURATION_STATISTIC_PATH,
                quantile=DURATION_PREFERRED_TARGET_QUANTILE,
                hard_gate=False,
                scope="pooled_strict_highlights",
                comparison="symmetric_target_deviation",
            ),
            "pooled_nonwide_duration_preferred_target": metric(
                config_field="chorus_role_preferred_nonwide_fraction",
                policy_member="preferred_nonwide_duration_fraction",
                value=config.chorus_role_preferred_nonwide_fraction,
                statistic_path=HIGHLIGHT_NONWIDE_DURATION_STATISTIC_PATH,
                quantile=DURATION_PREFERRED_TARGET_QUANTILE,
                hard_gate=False,
                scope="pooled_strict_highlights",
                comparison="symmetric_target_deviation",
            ),
            "maximum_consecutive_wide_run": metric(
                config_field="maximum_consecutive_wide_run_seconds",
                policy_member="maximum_consecutive_wide_run_seconds",
                value=config.maximum_consecutive_wide_run_seconds,
                statistic_path=HIGHLIGHT_WIDE_RUN_STATISTIC_PATH,
                quantile=WIDE_RUN_HARD_MAXIMUM_QUANTILE,
                hard_gate=True,
                scope="each_strict_highlight",
                comparison="less_than_or_equal",
            ),
            "preferred_maximum_consecutive_wide_run": metric(
                config_field=(
                    "chorus_role_preferred_maximum_consecutive_wide_run_seconds"
                ),
                policy_member=(
                    "preferred_maximum_consecutive_wide_run_seconds"
                ),
                value=(
                    config.chorus_role_preferred_maximum_consecutive_wide_run_seconds
                ),
                statistic_path=HIGHLIGHT_WIDE_RUN_STATISTIC_PATH,
                quantile=WIDE_RUN_PREFERRED_MAXIMUM_QUANTILE,
                hard_gate=False,
                scope="maximum_across_strict_highlights",
                comparison="soft_upper_target",
            ),
            "highlight_source_switch_cadence_product_minimum": metric(
                config_field=(
                    "highlight_product_minimum_source_switches_per_minute"
                ),
                policy_member=(
                    "highlight_product_minimum_source_switches_per_minute"
                ),
                value=(
                    config.highlight_product_minimum_source_switches_per_minute
                ),
                statistic_path=HIGHLIGHT_CADENCE_STATISTIC_PATH,
                quantile=HIGHLIGHT_CADENCE_PRODUCT_MINIMUM_QUANTILE,
                hard_gate=True,
                scope="each_positive_duration_chorus",
                comparison="greater_than_or_equal",
            ),
            "highlight_source_switch_cadence_preferred_target": metric(
                config_field="highlight_preferred_source_switches_per_minute",
                policy_member="highlight_preferred_source_switches_per_minute",
                value=config.highlight_preferred_source_switches_per_minute,
                statistic_path=HIGHLIGHT_CADENCE_STATISTIC_PATH,
                quantile=HIGHLIGHT_CADENCE_PREFERRED_TARGET_QUANTILE,
                hard_gate=False,
                scope="each_positive_duration_chorus",
                comparison="absolute_target_deviation",
            ),
        },
    }


def _role_duration_aggregates(
    records_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    config: RenderedMontageGrammarConfig,
) -> dict[str, dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    for role in ("build", "chorus_like", "bridge_like"):
        records = records_by_role.get(role, ())
        thresholds = _role_duration_thresholds(config, role)
        assert thresholds is not None
        known_duration = sum(
            float(item["known_duration_seconds"]) for item in records
        )
        close_duration = sum(
            float(item["closeup_duration_seconds"]) for item in records
        )
        nonwide_duration = sum(
            float(item["nonwide_duration_seconds"]) for item in records
        )
        close_fraction = (
            close_duration / known_duration if known_duration > 0.0 else 0.0
        )
        nonwide_fraction = (
            nonwide_duration / known_duration if known_duration > 0.0 else 0.0
        )
        maximum_wide_run = max(
            (
                float(item["maximum_consecutive_wide_run_seconds"])
                for item in records
            ),
            default=0.0,
        )
        failures: list[str] = []
        if records:
            pooled_close = thresholds[
                "pooled_product_minimum_closeup_fraction"
            ]
            if (
                pooled_close is not None
                and close_fraction + _EPSILON < float(pooled_close)
            ):
                failures.append(
                    _role_duration_failure_code(
                        role,
                        "pooled_closeup_fraction_below_product_minimum",
                    )
                )
            pooled_nonwide = thresholds[
                "pooled_product_minimum_nonwide_fraction"
            ]
            if (
                pooled_nonwide is not None
                and nonwide_fraction + _EPSILON < float(pooled_nonwide)
            ):
                failures.append(
                    _role_duration_failure_code(
                        role,
                        "pooled_nonwide_fraction_below_product_minimum",
                    )
                )
        close_target = _symmetric_fraction_target_evidence(
            close_fraction,
            float(thresholds["preferred_closeup_fraction"]),
        )
        nonwide_target = _symmetric_fraction_target_evidence(
            nonwide_fraction,
            float(thresholds["preferred_nonwide_fraction"]),
        )
        wide_target = _wide_run_soft_target_evidence(
            maximum_wide_run,
            float(
                thresholds[
                    "preferred_maximum_consecutive_wide_run_seconds"
                ]
            ),
            float(thresholds["maximum_consecutive_wide_run_seconds"]),
        )
        aggregates[role] = {
            "contract": ROLE_CONDITIONED_DURATION_EVIDENCE_CONTRACT,
            "positive_duration_section_indices": [
                int(item["section_index"]) for item in records
            ],
            "positive_duration_section_count": len(records),
            "duration_weighting": (
                "sum_observed_known_shot_duration_within_role"
            ),
            "reference_count": thresholds["reference_count"],
            "observed": {
                "known_duration_seconds": known_duration,
                "closeup_duration_seconds": close_duration,
                "nonwide_duration_seconds": nonwide_duration,
                "closeup_duration_fraction": close_fraction,
                "nonwide_duration_fraction": nonwide_fraction,
                "maximum_consecutive_wide_run_seconds": maximum_wide_run,
            },
            "hard_targets": {
                "local_minimum_closeup_duration_fraction": thresholds[
                    "local_minimum_closeup_fraction"
                ],
                "local_minimum_nonwide_duration_fraction": thresholds[
                    "local_minimum_nonwide_fraction"
                ],
                "pooled_product_minimum_closeup_duration_fraction": thresholds[
                    "pooled_product_minimum_closeup_fraction"
                ],
                "pooled_product_minimum_nonwide_duration_fraction": thresholds[
                    "pooled_product_minimum_nonwide_fraction"
                ],
                "maximum_consecutive_wide_run_seconds": thresholds[
                    "maximum_consecutive_wide_run_seconds"
                ],
                "zero_p10_promoted_to_positive_minimum": False,
            },
            "soft_targets": {
                "closeup_p50": close_target,
                "nonwide_p50": nonwide_target,
                "maximum_wide_run_p75": wide_target,
                "aggregate_alignment_score": (
                    float(close_target["alignment_score"])
                    + float(nonwide_target["alignment_score"])
                    + float(wide_target["alignment_score"])
                )
                / 3.0,
                "score_is_hard_gate": False,
            },
            "local_section_failures": {
                str(item["section_index"]): list(item["local_failure_codes"])
                for item in records
                if item["local_failure_codes"]
            },
            "failure_codes": failures,
            "passed": not failures
            and all(not item["local_failure_codes"] for item in records),
        }
    return aggregates


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
    bound_render_plan = _mapping(render_plan, "render_plan")
    proxy = _mapping(root.get("rendered_proxy_review"), "rendered_proxy_review")
    interior_required = bool(
        _automatic_ive_product_scope(root)
        or proxy.get("segment_interior_geometry") is not None
    )
    (
        rendered_segment_interior_geometry,
        rendered_geometry_duration_runs,
    ) = _review_segment_interior_geometry(
        choices=choices,
        rendered_proxy_review=proxy,
        render_plan=bound_render_plan,
        render_plan_binding=plan_binding,
        config=config,
        required=interior_required,
    )
    duration_runs: Sequence[Mapping[str, Any]] = (
        rendered_geometry_duration_runs
        if rendered_geometry_duration_runs
        else runs
    )
    motion_segments = _motion_accent_segment_indices(
        root,
        render_plan=bound_render_plan,
    )

    timeline_start = float(runs[0]["start_seconds"])
    timeline_end = float(runs[-1]["end_seconds"])

    all_failures: list[str] = (
        list(rendered_segment_interior_geometry["failure_codes"])
        if rendered_segment_interior_geometry is not None
        else []
    )
    section_metrics: list[dict[str, Any]] = []
    strict_highlight_records: list[dict[str, Any]] = []
    role_duration_records: dict[str, list[dict[str, Any]]] = {
        role: [] for role in _DURATION_GATED_ROLES
    }
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
        relevant_duration_runs = [
            item
            for item in duration_runs
            if _overlap(
                float(item["start_seconds"]),
                float(item["end_seconds"]),
                start,
                end,
            )
            > _EPSILON
        ]
        class_durations = {"wide": 0.0, "medium": 0.0, "close": 0.0, "unknown": 0.0}
        for item in relevant_duration_runs:
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
        maximum_consecutive_wide_run_seconds = (
            _maximum_consecutive_wide_run_seconds(
                relevant_duration_runs,
                section_start=start,
                section_end=end,
            )
        )
        close_only_burst_evidence = _close_only_multi_source_burst_evidence(
            relevant_runs,
            section_start=start,
            section_end=end,
            motion_segment_indices=motion_segments,
            config=config,
        )
        burst_count = 0
        burst_right_positions: list[int] = []
        source_switch_count = 0
        wide_involved_source_switch_count = 0
        closeup_involved_source_switch_count = 0
        nonwide_involved_source_switch_count = 0
        closeup_editorial_motion_accent_switch_count = 0
        nonwide_editorial_motion_accent_switch_count = 0
        for run_position, (left, right) in enumerate(
            zip(relevant_runs, relevant_runs[1:])
        ):
            boundary = float(right["start_seconds"])
            if not (start - _EPSILON <= boundary < end - _EPSILON):
                continue
            if left["derivative_family_id"] == right["derivative_family_id"]:
                continue
            source_switch_count += 1
            closeup_involved = bool(
                left["shot_class"] in _CLOSE
                or right["shot_class"] in _CLOSE
            )
            nonwide_involved = bool(
                left["shot_class"] in _NONWIDE
                or right["shot_class"] in _NONWIDE
            )
            editorial_motion_accent = bool(
                int(left["last_segment_index"]) in motion_segments
                or int(right["first_segment_index"]) in motion_segments
            )
            if closeup_involved:
                closeup_involved_source_switch_count += 1
                closeup_editorial_motion_accent_switch_count += int(
                    editorial_motion_accent
                )
            if nonwide_involved:
                nonwide_involved_source_switch_count += 1
                nonwide_editorial_motion_accent_switch_count += int(
                    editorial_motion_accent
                )
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
        role_duration_thresholds = _role_duration_thresholds(config, role)
        strict_role_duration_gate = role_duration_thresholds is not None
        strict_highlight = role == "chorus_like"
        achieved_source_switches_per_minute = (
            source_switch_count * 60.0 / (end - start)
            if strict_highlight
            else None
        )
        failures: list[str] = []
        duration_local_failures: list[str] = []
        if strict_role_duration_gate:
            assert role_duration_thresholds is not None
            if known_fraction + _EPSILON < config.minimum_known_shot_fraction:
                duration_local_failures.append(
                    _role_duration_failure_code(
                        role, "shot_type_evidence_incomplete"
                    )
                )
            local_close = role_duration_thresholds[
                "local_minimum_closeup_fraction"
            ]
            if (
                local_close is not None
                and close_fraction + _EPSILON < float(local_close)
            ):
                duration_local_failures.append(
                    _role_duration_failure_code(
                        role, "closeup_fraction_below_floor"
                    )
                )
            local_nonwide = role_duration_thresholds[
                "local_minimum_nonwide_fraction"
            ]
            if (
                local_nonwide is not None
                and nonwide_fraction + _EPSILON < float(local_nonwide)
            ):
                duration_local_failures.append(
                    _role_duration_failure_code(
                        role, "nonwide_fraction_below_floor"
                    )
                )
            if (
                maximum_consecutive_wide_run_seconds
                > float(
                    role_duration_thresholds[
                        "maximum_consecutive_wide_run_seconds"
                    ]
                )
                + _EPSILON
            ):
                duration_local_failures.append(
                    _role_duration_failure_code(
                        role, "maximum_consecutive_wide_run_exceeded"
                    )
                )
            failures.extend(duration_local_failures)
            role_duration_records[role].append(
                {
                    "section_index": position,
                    "known_duration_seconds": known,
                    "closeup_duration_seconds": class_durations["close"],
                    "nonwide_duration_seconds": (
                        class_durations["medium"] + class_durations["close"]
                    ),
                    "maximum_consecutive_wide_run_seconds": (
                        maximum_consecutive_wide_run_seconds
                    ),
                    "source_switch_count": source_switch_count,
                    "duration_seconds": end - start,
                    "local_failure_codes": tuple(duration_local_failures),
                }
            )
        if strict_highlight:
            assert achieved_source_switches_per_minute is not None
            if (
                achieved_source_switches_per_minute + _EPSILON
                < config.highlight_product_minimum_source_switches_per_minute
            ):
                failures.append(
                    "highlight_source_switches_per_minute_below_product_minimum"
                )
            if (
                closeup_editorial_motion_accent_switch_count
                < config.minimum_closeup_editorial_motion_accent_switch_boundaries
            ):
                failures.append(
                    "highlight_closeup_editorial_motion_accent_switch_missing"
                )
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
            strict_highlight_records.append(
                {
                    "section_index": position,
                    "known_duration_seconds": known,
                    "closeup_duration_seconds": class_durations["close"],
                    "nonwide_duration_seconds": (
                        class_durations["medium"] + class_durations["close"]
                    ),
                    "maximum_consecutive_wide_run_seconds": (
                        maximum_consecutive_wide_run_seconds
                    ),
                    "closeup_editorial_motion_accent_switch_count": (
                        closeup_editorial_motion_accent_switch_count
                    ),
                    "closeup_involved_source_switch_count": (
                        closeup_involved_source_switch_count
                    ),
                    "nonwide_editorial_motion_accent_switch_count": (
                        nonwide_editorial_motion_accent_switch_count
                    ),
                    "nonwide_involved_source_switch_count": (
                        nonwide_involved_source_switch_count
                    ),
                    "source_switch_count": source_switch_count,
                    "duration_seconds": end - start,
                    "achieved_source_switches_per_minute": (
                        achieved_source_switches_per_minute
                    ),
                    "close_only_multi_source_burst": (
                        close_only_burst_evidence
                    ),
                    "local_failure_codes": tuple(failures),
                }
            )
        all_failures.extend(f"section_{position}:{item}" for item in failures)
        section_metrics.append(
            {
                "section_index": position,
                "role": role,
                "duration_seconds": end - start,
                "strict_highlight_gate_applied": strict_highlight,
                "strict_role_duration_gate_applied": (
                    strict_role_duration_gate
                ),
                "short_section_duration_exemption_applied": False,
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
                "known_duration_seconds": known,
                "wide_duration_fraction": wide_fraction,
                "wide_duration_seconds": class_durations["wide"],
                "nonwide_duration_fraction": nonwide_fraction,
                "nonwide_duration_seconds": (
                    class_durations["medium"] + class_durations["close"]
                ),
                "closeup_duration_fraction": close_fraction,
                "closeup_duration_seconds": class_durations["close"],
                "maximum_consecutive_wide_run_seconds": (
                    maximum_consecutive_wide_run_seconds
                ),
                "maximum_consecutive_wide_run_hard_limit_seconds": (
                    role_duration_thresholds[
                        "maximum_consecutive_wide_run_seconds"
                    ]
                    if role_duration_thresholds is not None
                    else None
                ),
                "preferred_maximum_consecutive_wide_run_seconds": (
                    role_duration_thresholds[
                        "preferred_maximum_consecutive_wide_run_seconds"
                    ]
                    if role_duration_thresholds is not None
                    else None
                ),
                "closeup_preferred_target": (
                    _symmetric_fraction_target_evidence(
                        close_fraction,
                        float(
                            role_duration_thresholds[
                                "preferred_closeup_fraction"
                            ]
                        ),
                    )
                    if role_duration_thresholds is not None
                    else None
                ),
                "nonwide_preferred_target": (
                    _symmetric_fraction_target_evidence(
                        nonwide_fraction,
                        float(
                            role_duration_thresholds[
                                "preferred_nonwide_fraction"
                            ]
                        ),
                    )
                    if role_duration_thresholds is not None
                    else None
                ),
                "wide_run_preferred_target": (
                    _wide_run_soft_target_evidence(
                        maximum_consecutive_wide_run_seconds,
                        float(
                            role_duration_thresholds[
                                "preferred_maximum_consecutive_wide_run_seconds"
                            ]
                        ),
                        float(
                            role_duration_thresholds[
                                "maximum_consecutive_wide_run_seconds"
                            ]
                        ),
                    )
                    if role_duration_thresholds is not None
                    else None
                ),
                "source_switches_per_minute": (
                    achieved_source_switches_per_minute
                ),
                "source_switches_per_minute_hard_minimum": (
                    config.highlight_product_minimum_source_switches_per_minute
                    if strict_highlight
                    else None
                ),
                "source_switches_per_minute_preferred_target": (
                    config.highlight_preferred_source_switches_per_minute
                    if strict_highlight
                    else None
                ),
                "source_switches_per_minute_preferred_target_evidence": (
                    {
                        "target": (
                            config.highlight_preferred_source_switches_per_minute
                        ),
                        "observed": achieved_source_switches_per_minute,
                        "absolute_deviation": abs(
                            float(achieved_source_switches_per_minute)
                            - config.highlight_preferred_source_switches_per_minute
                        ),
                        "hard_gate": False,
                    }
                    if strict_highlight
                    else None
                ),
                "nonwide_burst_boundary_count": burst_count,
                "close_only_multi_source_burst": close_only_burst_evidence,
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
                "editorial_motion_accent_switch_boundaries": {
                    "evidence_contract": (
                        "bounded_within_segment_editorial_transform_change_v1"
                    ),
                    "is_choreography_motion_proof": False,
                    "profile_v2_human_distribution_available": False,
                    "profile_v3_derivation_required": True,
                    "closeup_involved_source_switch_boundary_count": (
                        closeup_involved_source_switch_count
                    ),
                    "closeup_accented_source_switch_boundary_count": (
                        closeup_editorial_motion_accent_switch_count
                    ),
                    "closeup_static_source_switch_boundary_count": (
                        closeup_involved_source_switch_count
                        - closeup_editorial_motion_accent_switch_count
                    ),
                    "nonwide_involved_source_switch_boundary_count": (
                        nonwide_involved_source_switch_count
                    ),
                    "nonwide_accented_source_switch_boundary_count": (
                        nonwide_editorial_motion_accent_switch_count
                    ),
                    "nonwide_static_source_switch_boundary_count": (
                        nonwide_involved_source_switch_count
                        - nonwide_editorial_motion_accent_switch_count
                    ),
                    "minimum_closeup_accented_switch_boundary_count": (
                        config.minimum_closeup_editorial_motion_accent_switch_boundaries
                    ),
                },
                "failure_codes": failures,
                "passed": not failures,
            }
        )

    if not math.isclose(previous_end or 0.0, timeline_end, abs_tol=_EPSILON):
        raise ValueError("section plan end does not match visual timeline")

    role_duration_aggregates = _role_duration_aggregates(
        role_duration_records,
        config,
    )
    for role in ("build", "bridge_like"):
        all_failures.extend(
            role_duration_aggregates[role]["failure_codes"]
        )

    aggregate_known_duration = sum(
        float(item["known_duration_seconds"])
        for item in strict_highlight_records
    )
    aggregate_closeup_duration = sum(
        float(item["closeup_duration_seconds"])
        for item in strict_highlight_records
    )
    aggregate_nonwide_duration = sum(
        float(item["nonwide_duration_seconds"])
        for item in strict_highlight_records
    )
    aggregate_closeup_fraction = (
        aggregate_closeup_duration / aggregate_known_duration
        if aggregate_known_duration
        else 0.0
    )
    aggregate_nonwide_fraction = (
        aggregate_nonwide_duration / aggregate_known_duration
        if aggregate_known_duration
        else 0.0
    )
    aggregate_maximum_wide_run = max(
        (
            float(item["maximum_consecutive_wide_run_seconds"])
            for item in strict_highlight_records
        ),
        default=0.0,
    )
    aggregate_closeup_switch_count = sum(
        int(item["closeup_involved_source_switch_count"])
        for item in strict_highlight_records
    )
    aggregate_closeup_motion_switch_count = sum(
        int(item["closeup_editorial_motion_accent_switch_count"])
        for item in strict_highlight_records
    )
    aggregate_nonwide_switch_count = sum(
        int(item["nonwide_involved_source_switch_count"])
        for item in strict_highlight_records
    )
    aggregate_nonwide_motion_switch_count = sum(
        int(item["nonwide_editorial_motion_accent_switch_count"])
        for item in strict_highlight_records
    )
    aggregate_source_switch_count = sum(
        int(item["source_switch_count"])
        for item in strict_highlight_records
    )
    aggregate_highlight_duration = sum(
        float(item["duration_seconds"])
        for item in strict_highlight_records
    )
    aggregate_source_switches_per_minute = (
        aggregate_source_switch_count * 60.0 / aggregate_highlight_duration
        if aggregate_highlight_duration > 0.0
        else 0.0
    )
    aggregate_close_burst_candidate_count = sum(
        int(item["close_only_multi_source_burst"]["candidate_event_count"])
        for item in strict_highlight_records
    )
    aggregate_close_burst_qualified_count = sum(
        int(item["close_only_multi_source_burst"]["qualified_event_count"])
        for item in strict_highlight_records
    )
    close_burst_binding_active = bool(
        strict_highlight_records
        and strict_highlight_records[0]["close_only_multi_source_burst"][
            "policy_binding_active"
        ]
    )
    close_burst_soft_score = (
        1.0 if aggregate_close_burst_qualified_count else 0.0
    ) if close_burst_binding_active else None
    close_burst_occurrence_support_weight = (
        min(
            1.0,
            config.close_burst_event_reference_count
            / config.highlight_cadence_reference_count,
        )
        if close_burst_binding_active
        and config.close_burst_event_reference_count is not None
        else None
    )
    aggregate_close_burst_events = [
        {"section_index": int(item["section_index"]), **dict(event)}
        for item in strict_highlight_records
        for event in item["close_only_multi_source_burst"]["events"]
    ]
    aggregate_failures: list[str] = [
        f"section_{item['section_index']}:{failure}"
        for item in strict_highlight_records
        for failure in item["local_failure_codes"]
    ]
    if strict_highlight_records:
        if (
            aggregate_closeup_fraction + _EPSILON
            < config.chorus_role_pooled_product_minimum_closeup_fraction
        ):
            aggregate_failures.append(
                "highlight_pooled_closeup_fraction_below_product_minimum"
            )
        if (
            aggregate_nonwide_fraction + _EPSILON
            < config.chorus_role_pooled_product_minimum_nonwide_fraction
        ):
            aggregate_failures.append(
                "highlight_pooled_nonwide_fraction_below_product_minimum"
            )
    all_failures.extend(
        item for item in aggregate_failures if not item.startswith("section_")
    )
    closeup_target = _symmetric_fraction_target_evidence(
        aggregate_closeup_fraction,
        config.chorus_role_preferred_closeup_fraction,
    )
    nonwide_target = _symmetric_fraction_target_evidence(
        aggregate_nonwide_fraction,
        config.chorus_role_preferred_nonwide_fraction,
    )
    wide_run_target = _wide_run_soft_target_evidence(
        aggregate_maximum_wide_run,
        config.chorus_role_preferred_maximum_consecutive_wide_run_seconds,
        config.maximum_consecutive_wide_run_seconds,
    )
    soft_target_score = (
        (
            float(closeup_target["alignment_score"])
            + float(nonwide_target["alignment_score"])
            + float(wide_run_target["alignment_score"])
        )
        / 3.0
        if strict_highlight_records
        else None
    )
    highlight_aggregate = {
        "contract": HIGHLIGHT_AGGREGATE_CONTRACT,
        "strict_highlight_section_indices": [
            int(item["section_index"]) for item in strict_highlight_records
        ],
        "strict_highlight_section_count": len(strict_highlight_records),
        "duration_weighting": (
            "sum_observed_known_shot_duration_across_strict_highlights"
        ),
        "observed": {
            "known_duration_seconds": aggregate_known_duration,
            "closeup_duration_seconds": aggregate_closeup_duration,
            "nonwide_duration_seconds": aggregate_nonwide_duration,
            "closeup_duration_fraction": aggregate_closeup_fraction,
            "nonwide_duration_fraction": aggregate_nonwide_fraction,
            "maximum_consecutive_wide_run_seconds": (
                aggregate_maximum_wide_run
            ),
            "closeup_involved_source_switch_boundary_count": (
                aggregate_closeup_switch_count
            ),
            "editorial_motion_accent_closeup_switch_boundary_count": (
                aggregate_closeup_motion_switch_count
            ),
            "static_closeup_switch_boundary_count": (
                aggregate_closeup_switch_count
                - aggregate_closeup_motion_switch_count
            ),
            "nonwide_involved_source_switch_boundary_count": (
                aggregate_nonwide_switch_count
            ),
            "editorial_motion_accent_nonwide_switch_boundary_count": (
                aggregate_nonwide_motion_switch_count
            ),
            "static_nonwide_switch_boundary_count": (
                aggregate_nonwide_switch_count
                - aggregate_nonwide_motion_switch_count
            ),
            "editorial_motion_is_choreography_motion_proof": False,
            "source_switch_count": aggregate_source_switch_count,
            "source_switches_per_minute": (
                aggregate_source_switches_per_minute
            ),
            "close_only_multi_source_burst_candidate_event_count": (
                aggregate_close_burst_candidate_count
            ),
            "close_only_multi_source_burst_qualified_event_count": (
                aggregate_close_burst_qualified_count
            ),
        },
        "hard_targets": {
            "local_minimum_closeup_duration_fraction": (
                config.minimum_closeup_fraction
            ),
            "local_minimum_nonwide_duration_fraction": (
                config.minimum_nonwide_fraction
            ),
            "pooled_product_minimum_closeup_duration_fraction": (
                config.chorus_role_pooled_product_minimum_closeup_fraction
            ),
            "pooled_product_minimum_nonwide_duration_fraction": (
                config.chorus_role_pooled_product_minimum_nonwide_fraction
            ),
            "maximum_consecutive_wide_run_seconds": (
                config.maximum_consecutive_wide_run_seconds
            ),
            "minimum_source_switches_per_minute": (
                config.highlight_product_minimum_source_switches_per_minute
            ),
        },
        "soft_targets": {
            "closeup_p50": closeup_target,
            "nonwide_p50": nonwide_target,
            "maximum_wide_run_p75": wide_run_target,
            "source_switches_per_minute_p75": {
                "target": config.highlight_preferred_source_switches_per_minute,
                "observed": aggregate_source_switches_per_minute,
                "absolute_deviation": abs(
                    aggregate_source_switches_per_minute
                    - config.highlight_preferred_source_switches_per_minute
                ),
                "hard_gate": False,
                "reference_count": config.highlight_cadence_reference_count,
            },
            "close_only_multi_source_burst_event_p10": {
                "contract": CLOSE_BURST_EVENT_EVIDENCE_CONTRACT,
                "policy_binding_active": close_burst_binding_active,
                "policy_binding": (
                    strict_highlight_records[0][
                        "close_only_multi_source_burst"
                    ]["policy_binding"]
                    if close_burst_binding_active
                    else None
                ),
                "candidate_event_count": aggregate_close_burst_candidate_count,
                "qualified_event_count": aggregate_close_burst_qualified_count,
                "events": aggregate_close_burst_events,
                "alignment_score": close_burst_soft_score,
                "event_occurrence_support_weight": (
                    close_burst_occurrence_support_weight
                ),
                "evidence_weighted_alignment_score": (
                    close_burst_soft_score * close_burst_occurrence_support_weight
                    if close_burst_soft_score is not None
                    and close_burst_occurrence_support_weight is not None
                    else None
                ),
                "observed_event_count": config.close_burst_event_reference_count,
                "human_reference_count": config.highlight_cadence_reference_count,
                "saturated_at_one_or_more_qualified_events": True,
                "absence_is_hard_failure": False,
                "hard_gate": False,
                "per_reference_occurrence_p10": 0.0,
                "per_reference_occurrence_median": 0.0,
                "medium_to_close_counts_as_close_only": False,
            },
            "aggregate_alignment_score": soft_target_score,
            "score_is_hard_gate": False,
            "over_close_past_target_is_rewarded": False,
        },
        "policy_provenance": _highlight_policy_provenance(config),
        "local_section_failures": {
            str(item["section_index"]): list(item["local_failure_codes"])
            for item in strict_highlight_records
            if item["local_failure_codes"]
        },
        "failure_codes": aggregate_failures,
        "passed": not aggregate_failures,
    }

    strict_chorus_present = any(
        str(_mapping(raw, f"section {index}").get("role") or "")
        == "chorus_like"
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
        "highlight_aggregate": highlight_aggregate,
        "role_conditioned_duration_aggregates": role_duration_aggregates,
        "exact_anchor_metrics": exact_anchor_metrics,
        "rendered_closeup_proxy": closeup_proxy,
        "rendered_segment_interior_geometry": (
            rendered_segment_interior_geometry
        ),
        "policy": {
            "formation_wide": (
                "identity_free_weak_person_plus_upper_body_or_strong_four_"
                "person_box_evidence_for_anchor_and_reset_never_full_group_"
                "proof"
            ),
            "highlight": (
                "every_positive_build_chorus_bridge_section_uses_role_p10_"
                "when_positive_plus_role_pooled_p25_and_p100_wide_run_hard_"
                "gates_with_p50_p75_soft_targets_no_short_chorus_bypass"
            ),
            "highlight_cadence": (
                "every_positive_chorus_uses_hash_bound_p50_source_switch_"
                "rate_hard_minimum_and_p75_soft_target"
            ),
            "motion": (
                "bounded_editorial_accent_audit_not_choreography_motion_proof_"
                "profile_v3_distribution_required"
            ),
            "pixel_proof": (
                "plan_closeup_endpoints_plus_hash_bound_segment_interior_"
                "sampled_geometry_duration_and_wide_run_proxy_estimate"
            ),
        },
    }


__all__ = [
    "EXACT_STAGE_EDIT_SHOT_CLOCK_CONTRACT",
    "RENDERED_MONTAGE_GRAMMAR_ANALYSIS_BASIS",
    "RENDERED_SEGMENT_INTERIOR_GEOMETRY_REVIEW_CONTRACT",
    "RenderedMontageGrammarConfig",
    "review_rendered_montage_grammar",
]
