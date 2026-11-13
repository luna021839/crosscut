"""Turn recovered reference edits into ranking-ready training records.

The expensive video-to-source attribution pass intentionally lives separately
from this module.  This builder accepts its conservative JSON payload, keeps
unknown shots out of supervised labels, and preserves them as explicit gaps so
later training never learns a guessed source as ground truth.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from .reference_attribution import (
    REFERENCE_ATTRIBUTION_BASIS,
    REFERENCE_CANONICAL_SHOT_SCHEMA_VERSION,
    REFERENCE_DETECTED_BOUNDARY_SCHEMA_VERSION,
    REFERENCE_DENSE_ATTRIBUTION_SCHEMA_VERSION,
    REFERENCE_EVIDENCE_ANALYSIS_BASIS,
)
from .reference_crop_supervision import (
    ANALYSIS_BASIS as REFERENCE_CROP_SUPERVISION_ANALYSIS_BASIS,
    SCHEMA_VERSION as REFERENCE_CROP_SUPERVISION_SCHEMA_VERSION,
    ReferenceCropSupervisionArtifact,
    load_reference_crop_supervision,
    validate_reference_crop_supervision_payload,
)


SCHEMA_VERSION = 4
BOUNDARY_EVENT_SCHEMA_VERSION = 1

REFERENCE_START_EVENT = "reference_start"
NATIVE_SOURCE_CAMERA_CUT_EVENT = "native_source_camera_cut"
EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT = "editorial_source_switch_hard_cut"
EDITORIAL_SOURCE_SWITCH_UNRESOLVED_EVENT = (
    "editorial_source_switch_unresolved"
)
KNOWN_AFTER_UNKNOWN_EVENT = "known_after_unknown"
DETECTED_BOUNDARY_LOSS_MASKED_EVENT = "detected_boundary_loss_masked"


CropSupervisionInput = (
    ReferenceCropSupervisionArtifact | Mapping[str, Any] | Path
)


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


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    clean = tuple(_finite(value, "distribution value") for value in values)
    return {
        "count": len(clean),
        "minimum": min(clean) if clean else None,
        "p25": _quantile(clean, 0.25),
        "median": median(clean) if clean else None,
        "p75": _quantile(clean, 0.75),
        "p90": _quantile(clean, 0.90),
        "maximum": max(clean) if clean else None,
    }


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_attribution_hash_identity(
    attribution: Mapping[str, Any],
    shots: Sequence[Mapping[str, Any]],
    detected_boundaries: Sequence[Mapping[str, Any]],
    *,
    partition_content_sha256: str,
) -> dict[str, Any]:
    source_path = attribution.get("source_path")
    if not isinstance(source_path, list) or any(
        not isinstance(state, Mapping) for state in source_path
    ):
        raise ValueError("canonical attribution hash needs a source_path array")
    return {
        "partition_content_sha256": partition_content_sha256,
        "source_path_source_ids": [
            state.get("attributed_video_id") for state in source_path
        ],
        "shot_aggregates": [
            {
                "shot_index": shot.get("shot_index"),
                "accepted": shot.get("accepted"),
                "attributed_video_id": shot.get("attributed_video_id"),
                "canonical_detected_shot": deepcopy(
                    shot.get("canonical_detected_shot")
                ),
                "dense_attribution_span": deepcopy(
                    shot.get("dense_attribution_span")
                ),
                "boundary_refinement": deepcopy(
                    shot.get("boundary_refinement")
                ),
            }
            for shot in shots
        ],
        "detected_boundary_labels": [
            {
                "boundary_index": boundary.get("boundary_index"),
                "incoming_shot_index": boundary.get("incoming_shot_index"),
                "old_source_id": boundary.get("old_source_id"),
                "new_source_id": boundary.get("new_source_id"),
                "supervision_classification": boundary.get(
                    "supervision_classification"
                ),
                "classification_reason": boundary.get(
                    "classification_reason"
                ),
                "cut_target": boundary.get("cut_target"),
                "loss_masked": boundary.get("loss_masked"),
                "dense_attribution_ready": boundary.get(
                    "dense_attribution_ready"
                ),
                "boundary_refinement": deepcopy(
                    boundary.get("boundary_refinement")
                ),
            }
            for boundary in detected_boundaries
        ],
    }


def _source_id(shot: Mapping[str, Any]) -> str | None:
    selected = shot.get("attributed_video_id")
    if selected is None or not bool(shot.get("accepted")):
        return None
    value = str(selected).strip()
    return value or None


def _dense_attribution_context(
    attribution: Mapping[str, Any],
    shot: Mapping[str, Any],
    selected_source: str | None,
) -> dict[str, Any]:
    """Validate and summarize optional dense temporal-attribution provenance."""

    raw_schema = attribution.get("dense_attribution_schema")
    raw_span = shot.get("dense_attribution_span")
    if raw_schema is None and raw_span is None:
        return {
            "schema_version": REFERENCE_DENSE_ATTRIBUTION_SCHEMA_VERSION,
            "ready": False,
            "reason": "dense_attribution_provenance_unavailable",
            "first_sample_index": shot.get("first_sample_index"),
            "last_sample_index": shot.get("last_sample_index"),
            "sample_count": None,
        }
    if not isinstance(raw_schema, Mapping) or not isinstance(raw_span, Mapping):
        raise ValueError(
            "dense attribution schema and shot span must be supplied together"
        )
    if raw_schema.get("schema_version") != REFERENCE_DENSE_ATTRIBUTION_SCHEMA_VERSION:
        raise ValueError("unsupported dense attribution schema version")
    if raw_span.get("schema_version") != REFERENCE_DENSE_ATTRIBUTION_SCHEMA_VERSION:
        raise ValueError("dense attribution span schema does not match")
    if raw_schema.get("temporal_granularity") != "synchronized_sample_path":
        raise ValueError("dense attribution temporal granularity is unsupported")
    if not bool(raw_schema.get("sample_level_source_path_available")):
        raise ValueError("dense attribution must retain its sample-level source path")
    if not bool(
        raw_schema.get(
            "shot_spans_reference_contiguous_source_path_samples"
        )
    ):
        raise ValueError("dense attribution shot-span contract is missing")
    if not bool(
        raw_schema.get("frame_dense_boundary_proof_required_for_hard_cut")
    ):
        raise ValueError("dense attribution hard-cut proof contract is missing")
    if not bool(
        raw_schema.get("native_source_camera_cut_is_not_editorial_source_switch")
    ):
        raise ValueError("dense attribution event-separation contract is missing")
    claimed_ready = bool(raw_span.get("ready", True))
    sample_count = _nonnegative_integer(
        raw_span.get("sample_count"), "dense attribution sample count"
    )
    if sample_count == 0:
        if (
            raw_span.get("first_sample_index") is not None
            or raw_span.get("last_sample_index") is not None
            or claimed_ready
        ):
            raise ValueError("empty dense attribution span has invalid readiness")
        return {
            "schema_version": REFERENCE_DENSE_ATTRIBUTION_SCHEMA_VERSION,
            "ready": False,
            "reason": str(
                raw_span.get("reason")
                or "canonical_detected_shot_has_no_attribution_sample"
            ),
            "first_sample_index": None,
            "last_sample_index": None,
            "sample_count": 0,
        }
    first = _nonnegative_integer(
        raw_span.get("first_sample_index"), "dense first sample index"
    )
    last = _nonnegative_integer(
        raw_span.get("last_sample_index"), "dense last sample index"
    )
    if last < first:
        raise ValueError("dense attribution sample span is reversed")
    if sample_count != last - first + 1:
        raise ValueError("dense attribution sample count does not match its span")
    if raw_span.get("selected_source_id") != selected_source:
        raise ValueError("dense attribution span selected source does not match shot")
    expected_state = "known" if selected_source is not None else "unknown"
    if raw_span.get("state") != expected_state:
        raise ValueError("dense attribution span state does not match shot")
    path = attribution.get("source_path")
    if not isinstance(path, list) or last >= len(path):
        raise ValueError("dense attribution span is outside source_path")
    path_sources: list[str | None] = []
    for sample_index in range(first, last + 1):
        sample = path[sample_index]
        if not isinstance(sample, Mapping):
            raise ValueError("dense attribution source_path sample is invalid")
        path_sources.append(sample.get("attributed_video_id"))
        if claimed_ready and sample.get("attributed_video_id") != selected_source:
            raise ValueError(
                "dense attribution shot source disagrees with source_path"
            )
    if bool(raw_span.get("source_path_samples_are_contiguous")) is not True:
        raise ValueError("dense attribution shot samples must be contiguous")
    if not claimed_ready and selected_source not in path_sources and selected_source is not None:
        raise ValueError(
            "canonical aggregate source is absent from its attribution span"
        )
    return {
        "schema_version": REFERENCE_DENSE_ATTRIBUTION_SCHEMA_VERSION,
        "ready": claimed_ready,
        "reason": (
            "validated_synchronized_sample_path"
            if claimed_ready
            else str(raw_span.get("reason") or "canonical_aggregate_not_dense")
        ),
        "first_sample_index": first,
        "last_sample_index": last,
        "sample_count": sample_count,
    }


def _boundary_event(
    *,
    previous_source: str | None,
    selected_source: str,
    boundary_label: str,
    dense_attribution_ready: bool,
) -> dict[str, Any]:
    if boundary_label == "reference_start":
        event_type = REFERENCE_START_EVENT
        scope = "reference"
        is_native = False
        is_editorial_switch = False
        label_basis = "first_known_reference_shot"
    elif boundary_label == "same_source_internal_camera_cut":
        event_type = NATIVE_SOURCE_CAMERA_CUT_EVENT
        scope = "native_source"
        is_native = True
        is_editorial_switch = False
        label_basis = "same_source_on_both_sides_of_visual_boundary"
    elif boundary_label == "cross_source_hard_cut":
        event_type = EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT
        scope = "editorial"
        is_native = False
        is_editorial_switch = True
        label_basis = "resolved_cross_source_frame_boundary"
    elif boundary_label == "cross_source_transition_unresolved":
        event_type = EDITORIAL_SOURCE_SWITCH_UNRESOLVED_EVENT
        scope = "editorial"
        is_native = False
        is_editorial_switch = True
        label_basis = "cross_source_state_change_without_resolved_transition"
    elif boundary_label == "known_after_unknown":
        event_type = KNOWN_AFTER_UNKNOWN_EVENT
        scope = "attribution_gap"
        is_native = False
        is_editorial_switch = False
        label_basis = "previous_source_identity_unknown"
    elif boundary_label == "detected_boundary_loss_masked":
        event_type = DETECTED_BOUNDARY_LOSS_MASKED_EVENT
        scope = "detected_boundary_gate_failure"
        is_native = False
        is_editorial_switch = False
        label_basis = "detected_boundary_unknown_or_gate_failure"
    else:  # pragma: no cover - builder owns the closed label vocabulary.
        raise ValueError(f"unsupported boundary label: {boundary_label}")
    timing_eligible = event_type == EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT
    return {
        "schema_version": BOUNDARY_EVENT_SCHEMA_VERSION,
        "event_type": event_type,
        "event_scope": scope,
        "previous_source_id": previous_source,
        "selected_source_id": selected_source,
        "is_native_source_camera_cut": is_native,
        "is_editorial_source_switch": is_editorial_switch,
        "editorial_timing_supervision_eligible": timing_eligible,
        "editorial_source_selection_supervision_eligible": timing_eligible,
        "editorial_duration_supervision_eligible": timing_eligible,
        "editorial_transition_supervision_eligible": timing_eligible,
        "dense_attribution_ready": dense_attribution_ready,
        "label_basis": label_basis,
    }


def _selected_features(shot: Mapping[str, Any]) -> dict[str, Any]:
    best = shot.get("best")
    if not isinstance(best, Mapping):
        return {}
    names = (
        "score",
        "orb_good_matches",
        "orb_homography_inliers",
        "orb_inlier_ratio",
        "structure_ncc",
        "color_similarity",
        "source_frame_index",
        "source_time_seconds",
        "time_adjustment_seconds",
        "crop_transform",
        "homography_reference_to_source",
    )
    return {name: best[name] for name in names if name in best}


def _negative_examples(shot: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    candidates = shot.get("ranked_candidates")
    if not isinstance(candidates, list):
        runner_up = shot.get("runner_up")
        candidates = [runner_up] if isinstance(runner_up, Mapping) else []
    selected = _source_id(shot)
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = str(candidate.get("video_id") or "").strip()
        if not candidate_id or candidate_id == selected:
            continue
        output.append(
            {
                "source_id": candidate_id,
                "score": candidate.get("score"),
                "homography_inliers": candidate.get("orb_homography_inliers"),
                "homography_inlier_ratio": candidate.get("orb_inlier_ratio"),
                "structure_ncc": candidate.get("structure_ncc"),
                "color_similarity": candidate.get("color_similarity"),
                "source_time_seconds": candidate.get("source_time_seconds"),
            }
        )
    return output


def _resolved_boundary_policy(
    attribution: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    schema_version = attribution.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or schema_version != 1
        or attribution.get("analysis") != REFERENCE_ATTRIBUTION_BASIS
    ):
        return None
    decision_policy = attribution.get("decision_policy")
    if not isinstance(decision_policy, Mapping):
        return None
    if decision_policy.get("analysis_basis") != REFERENCE_ATTRIBUTION_BASIS:
        return None
    boundary_policy = decision_policy.get("boundary_refinement")
    return boundary_policy if isinstance(boundary_policy, Mapping) else None


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_resolved_boundary_proof(
    normalized: dict[str, Any],
    boundary_policy: Mapping[str, Any] | None,
) -> None:
    if boundary_policy is None:
        raise ValueError(
            "resolved_hard_cut requires resolver schema and decision-policy provenance"
        )
    minimum_identity_frames = _nonnegative_integer(
        boundary_policy.get("minimum_identity_frames_per_side"),
        "minimum identity frames",
    )
    maximum_uncertain_frames = _nonnegative_integer(
        boundary_policy.get("maximum_uncertain_middle_frames"),
        "maximum uncertain frames",
    )
    if minimum_identity_frames < 1:
        raise ValueError("minimum identity frames must be positive")
    policy_radius = _finite(
        boundary_policy.get("radius_seconds"),
        "policy boundary radius",
    )
    timestamp_tolerance = _finite(
        boundary_policy.get("timestamp_step_tolerance_ratio"),
        "timestamp step tolerance",
    )
    if policy_radius < 0.0 or not 0.0 <= timestamp_tolerance <= 1.0:
        raise ValueError("invalid boundary-refinement decision policy")

    nominal_frame = _nonnegative_integer(
        normalized.get("nominal_frame_index"),
        "nominal frame index",
    )
    selected_frame = _nonnegative_integer(
        normalized.get("selected_frame_index"),
        "selected frame index",
    )
    frame_offset = normalized.get("frame_offset")
    if isinstance(frame_offset, bool) or not isinstance(frame_offset, int):
        raise ValueError("frame_offset must be an integer")
    if selected_frame - nominal_frame != frame_offset:
        raise ValueError("boundary frame_offset does not match selected frame")
    old_count = _nonnegative_integer(
        normalized.get("old_identity_frames"),
        "old identity frames",
    )
    new_count = _nonnegative_integer(
        normalized.get("new_identity_frames"),
        "new identity frames",
    )
    uncertain_count = _nonnegative_integer(
        normalized.get("uncertain_middle_frames"),
        "uncertain middle frames",
    )

    raw_identities = normalized.get("frame_identities")
    raw_indices = normalized.get("evidence_frame_indices")
    raw_times = normalized.get("evidence_master_seconds")
    if (
        not isinstance(raw_identities, list)
        or not isinstance(raw_indices, list)
        or not isinstance(raw_times, list)
        or not len(raw_identities) == len(raw_indices) == len(raw_times)
        or len(raw_identities) < 2 * minimum_identity_frames
    ):
        raise ValueError("resolved boundary proof arrays are missing or incomplete")
    identities = tuple(str(value) for value in raw_identities)
    if any(
        identity not in {"old", "new", "ambiguous", "neither"}
        for identity in identities
    ):
        raise ValueError("resolved boundary proof has an invalid frame identity")
    indices = tuple(
        _nonnegative_integer(value, "evidence frame index")
        for value in raw_indices
    )
    times = tuple(_finite(value, "evidence master time") for value in raw_times)
    if any(
        right != left + 1
        for left, right in zip(indices, indices[1:], strict=False)
    ):
        raise ValueError("resolved boundary evidence frames must be consecutive")
    time_steps = tuple(
        right - left for left, right in zip(times, times[1:], strict=False)
    )
    if any(step <= 0.0 for step in time_steps):
        raise ValueError("resolved boundary evidence times must increase")
    expected_step = median(time_steps)
    if any(
        abs(step - expected_step)
        > expected_step * timestamp_tolerance + 1e-9
        for step in time_steps
    ):
        raise ValueError("resolved boundary evidence cadence is inconsistent")
    if nominal_frame not in indices or selected_frame not in indices:
        raise ValueError("resolved boundary proof omits a boundary frame")
    nominal_position = indices.index(nominal_frame)
    selected_position = indices.index(selected_frame)
    nominal_seconds = float(normalized["nominal_boundary_seconds"])
    selected_seconds = float(normalized["selected_boundary_seconds"])
    if abs(times[nominal_position] - nominal_seconds) > expected_step + 1e-9:
        raise ValueError("nominal frame timestamp does not match boundary proof")
    if abs(times[selected_position] - selected_seconds) > 1e-9:
        raise ValueError("selected frame timestamp does not match boundary proof")

    observed_old_count = 0
    for identity in reversed(identities[:selected_position]):
        if identity != "old":
            break
        observed_old_count += 1
    observed_new_count = 0
    for identity in identities[selected_position:]:
        if identity != "new":
            break
        observed_new_count += 1
    if any(identity == "new" for identity in identities[:selected_position]) or any(
        identity == "old" for identity in identities[selected_position:]
    ):
        raise ValueError("resolved boundary proof is not a monotonic source change")
    last_old = max(
        (
            index
            for index, identity in enumerate(identities[:selected_position])
            if identity == "old"
        ),
        default=-1,
    )
    first_new = min(
        (
            index
            for index, identity in enumerate(identities[selected_position:])
            if identity == "new"
        ),
        default=len(identities) - selected_position,
    ) + selected_position
    observed_uncertain_count = sum(
        identity in {"ambiguous", "neither"}
        for identity in identities[last_old + 1 : first_new]
    )
    if (
        old_count != observed_old_count
        or new_count != observed_new_count
        or uncertain_count != observed_uncertain_count
    ):
        raise ValueError("resolved boundary proof counts do not match identities")
    if (
        old_count < minimum_identity_frames
        or new_count < minimum_identity_frames
        or uncertain_count > maximum_uncertain_frames
    ):
        raise ValueError("resolved boundary proof does not pass decision policy")
    refinement_radius = _finite(
        normalized.get("refinement_radius_seconds"),
        "boundary refinement radius",
    )
    if not math.isclose(refinement_radius, policy_radius, abs_tol=1e-9):
        raise ValueError("boundary refinement radius does not match policy")


def _boundary_refinement(
    shot: Mapping[str, Any],
    *,
    boundary_policy: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    raw = shot.get("boundary_refinement")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("boundary_refinement must be an object")
    status = str(raw.get("status") or "").strip()
    if not status:
        raise ValueError("boundary_refinement.status cannot be empty")
    normalized = dict(raw)
    normalized["status"] = status
    selected = raw.get("selected_boundary_seconds")
    if selected is not None:
        normalized["selected_boundary_seconds"] = _finite(
            selected,
            "selected boundary",
        )
    if status == "resolved_hard_cut" and selected is None:
        raise ValueError(
            "resolved_hard_cut boundary refinement needs a selected boundary"
        )
    if status == "resolved_hard_cut":
        old_source = str(normalized.get("old_source_id") or "").strip()
        new_source = str(normalized.get("new_source_id") or "").strip()
        if not old_source or not new_source or old_source == new_source:
            raise ValueError(
                "resolved_hard_cut needs distinct old_source_id/new_source_id"
            )
        nominal = _finite(
            normalized.get(
                "nominal_boundary_seconds",
                normalized["selected_boundary_seconds"],
            ),
            "nominal boundary",
        )
        radius = _finite(
            normalized.get("refinement_radius_seconds", 0.5),
            "boundary refinement radius",
        )
        if radius < 0.0:
            raise ValueError("boundary refinement radius cannot be negative")
        if (
            abs(normalized["selected_boundary_seconds"] - nominal)
            > radius + 1e-9
        ):
            raise ValueError("selected boundary is outside its refinement radius")
        normalized["old_source_id"] = old_source
        normalized["new_source_id"] = new_source
        normalized["nominal_boundary_seconds"] = nominal
        normalized["refinement_radius_seconds"] = radius
        _nonnegative_integer(
            normalized.get("to_run_index"),
            "boundary refinement to_run_index",
        )
        if "to_shot_index" in normalized:
            _nonnegative_integer(
                normalized.get("to_shot_index"),
                "boundary refinement to_shot_index",
            )
        _validate_resolved_boundary_proof(normalized, boundary_policy)
    return normalized


def _effective_shot_spans(
    shots: Sequence[Mapping[str, Any]],
    source_ids: Sequence[str | None],
    refinements: Sequence[dict[str, Any] | None],
) -> tuple[list[float], list[float]]:
    starts = [
        _finite(shot["reference_start_seconds"], "shot start")
        for shot in shots
    ]
    ends = [
        _finite(shot["reference_end_seconds"], "shot end")
        for shot in shots
    ]
    if any(
        right < left
        for left, right in zip(starts, starts[1:], strict=False)
    ):
        raise ValueError("shot starts must be chronological")

    for index in range(1, len(shots)):
        refinement = refinements[index]
        if refinement is None or refinement["status"] != "resolved_hard_cut":
            continue
        previous_source = source_ids[index - 1]
        selected_source = source_ids[index]
        if (
            previous_source is None
            or selected_source is None
            or previous_source == selected_source
        ):
            raise ValueError(
                "resolved_hard_cut must describe adjacent distinct known sources"
            )
        declared_old = str(refinement.get("old_source_id") or "").strip()
        declared_new = str(refinement.get("new_source_id") or "").strip()
        if declared_old and declared_old != previous_source:
            raise ValueError("boundary_refinement old_source_id does not match")
        if declared_new and declared_new != selected_source:
            raise ValueError("boundary_refinement new_source_id does not match")
        declared_incoming_index = refinement.get(
            "to_shot_index", refinement.get("to_run_index")
        )
        if declared_incoming_index != index:
            raise ValueError(
                "boundary_refinement incoming index does not match the shot"
            )
        boundary = float(refinement["selected_boundary_seconds"])
        nominal = float(refinement["nominal_boundary_seconds"])
        radius = float(refinement["refinement_radius_seconds"])
        if abs(nominal - starts[index]) > radius + 1e-9:
            raise ValueError(
                "nominal refinement boundary does not match the incoming shot"
            )
        if abs(boundary - starts[index]) > radius + 1e-9:
            raise ValueError(
                "selected refinement boundary does not match the incoming shot"
            )
        if not starts[index - 1] < boundary < ends[index]:
            raise ValueError(
                "selected boundary must preserve positive adjacent shot spans"
            )
        ends[index - 1] = boundary
        starts[index] = boundary

    if any(end <= start for start, end in zip(starts, ends, strict=True)):
        raise ValueError("effective reference shots must have positive duration")
    return starts, ends


def _validated_canonical_boundaries(
    attribution: Mapping[str, Any],
    shots: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...] | None:
    """Fail closed on the detector partition and its boundary labels."""

    raw_schema = attribution.get("canonical_shot_schema")
    if raw_schema is None:
        return None
    if not isinstance(raw_schema, Mapping):
        raise ValueError("canonical_shot_schema must be an object")
    required_schema = {
        "schema_version": REFERENCE_CANONICAL_SHOT_SCHEMA_VERSION,
        "partition_basis": "reference_hard_cut_detector",
        "reference_evidence_analysis": REFERENCE_EVIDENCE_ANALYSIS_BASIS,
        "source_runs_are_attribution_only_not_shot_partitions": True,
        "same_source_adjacent_detector_shots_are_never_merged": True,
        "unknown_or_failed_boundary_gate_is_loss_masked": True,
        "attribution_hash_basis": (
            "sha256_canonical_json_of_partition_path_shot_aggregates_and_boundary_labels"
        ),
    }
    for name, expected in required_schema.items():
        if raw_schema.get(name) != expected:
            raise ValueError(f"canonical shot schema {name} is unsupported")
    raw_provenance = attribution.get("reference_evidence_provenance")
    if not isinstance(raw_provenance, Mapping):
        raise ValueError("canonical shot partition lacks evidence provenance")
    if (
        raw_provenance.get("schema_version") != 1
        or raw_provenance.get("analysis") != REFERENCE_EVIDENCE_ANALYSIS_BASIS
    ):
        raise ValueError("canonical shot evidence provenance is unsupported")

    raw_cuts = attribution.get("detected_reference_cuts")
    raw_boundaries = attribution.get("detected_boundaries")
    if not isinstance(raw_cuts, list) or not isinstance(raw_boundaries, list):
        raise ValueError("canonical detected cuts and boundaries must be arrays")
    if len(shots) != len(raw_cuts) + 1 or len(raw_boundaries) != len(raw_cuts):
        raise ValueError("canonical shot and detected-cut counts do not match")
    if (
        raw_schema.get("detected_cut_count") != len(raw_cuts)
        or raw_schema.get("canonical_shot_count") != len(shots)
    ):
        raise ValueError("canonical shot schema counts do not match payload")

    raw_source_path = attribution.get("source_path")
    if not isinstance(raw_source_path, list) or any(
        not isinstance(state, Mapping) for state in raw_source_path
    ):
        raise ValueError("canonical shot partition needs a source_path array")
    path_source_ids = [
        state.get("attributed_video_id") for state in raw_source_path
    ]

    sample_assignment: list[list[int]] = []
    previous_end: float | None = None
    for shot_index, shot in enumerate(shots):
        canonical = shot.get("canonical_detected_shot")
        if not isinstance(canonical, Mapping):
            raise ValueError("canonical shot lacks detector-owned metadata")
        if (
            canonical.get("schema_version")
            != REFERENCE_CANONICAL_SHOT_SCHEMA_VERSION
            or canonical.get("detector_shot_index") != shot_index
            or not bool(
                canonical.get(
                    "source_run_merge_does_not_define_shot_partition"
                )
            )
        ):
            raise ValueError("canonical detected-shot metadata is inconsistent")
        raw_indices = canonical.get("sample_indices")
        if not isinstance(raw_indices, list):
            raise ValueError("canonical detected-shot sample indices must be an array")
        indices = [
            _nonnegative_integer(value, "canonical shot sample index")
            for value in raw_indices
        ]
        if len(indices) != len(set(indices)) or any(
            right != left + 1
            for left, right in zip(indices, indices[1:], strict=False)
        ):
            raise ValueError("canonical shot sample indices must be contiguous")
        sample_assignment.append(indices)
        if any(index >= len(path_source_ids) for index in indices):
            raise ValueError("canonical shot sample index is outside source_path")
        shot_path_sources = [path_source_ids[index] for index in indices]
        raw_midpoints = canonical.get("midpoint_sample_indices")
        if not isinstance(raw_midpoints, list):
            raise ValueError("canonical midpoint sample indices must be an array")
        midpoint_indices = [
            _nonnegative_integer(value, "canonical midpoint sample index")
            for value in raw_midpoints
        ]
        if len(midpoint_indices) > 1 or any(
            index not in indices for index in midpoint_indices
        ):
            raise ValueError("canonical midpoint sample attribution is invalid")
        observed_counts = Counter(
            source for source in shot_path_sources if source is not None
        )
        if canonical.get("known_source_sample_counts") != dict(
            sorted(observed_counts.items())
        ) or canonical.get("unknown_sample_count") != sum(
            source is None for source in shot_path_sources
        ):
            raise ValueError("canonical shot aggregate counts are inconsistent")
        known_sources = set(observed_counts)
        expected_source: str | None = None
        if midpoint_indices:
            midpoint_source = path_source_ids[midpoint_indices[0]]
            if midpoint_source is not None and known_sources == {midpoint_source}:
                expected_source = str(midpoint_source)
        elif (
            shot_path_sources
            and all(source is not None for source in shot_path_sources)
            and len(known_sources) == 1
        ):
            expected_source = str(next(iter(known_sources)))
        if _source_id(shot) != expected_source or bool(
            shot.get("accepted")
        ) != (expected_source is not None):
            raise ValueError("canonical shot selected source is inconsistent")
        dense_span = shot.get("dense_attribution_span")
        if not isinstance(dense_span, Mapping):
            raise ValueError("canonical shot lacks a dense attribution span")
        if (
            dense_span.get("sample_count") != len(indices)
            or dense_span.get("first_sample_index")
            != (indices[0] if indices else None)
            or dense_span.get("last_sample_index")
            != (indices[-1] if indices else None)
            or dense_span.get("selected_source_id") != expected_source
        ):
            raise ValueError("canonical dense attribution span is inconsistent")
        start = _finite(shot.get("reference_start_seconds"), "shot start")
        end = _finite(shot.get("reference_end_seconds"), "shot end")
        if end <= start:
            raise ValueError("canonical shot span must be positive")
        if previous_end is not None and not math.isclose(
            start, previous_end, abs_tol=1e-9
        ):
            raise ValueError("canonical detected shots must be contiguous")
        previous_end = end

    if [index for group in sample_assignment for index in group] != list(
        range(len(path_source_ids))
    ):
        raise ValueError("canonical shots must partition every source_path sample")

    normalized_cuts: list[dict[str, Any]] = []
    validated_boundaries: list[dict[str, Any]] = []
    for boundary_index, (raw_cut, raw_boundary) in enumerate(
        zip(raw_cuts, raw_boundaries, strict=True)
    ):
        if not isinstance(raw_cut, Mapping) or not isinstance(
            raw_boundary, Mapping
        ):
            raise ValueError("canonical cut boundary must be an object")
        cut = dict(raw_cut)
        cut["frame_index"] = _nonnegative_integer(
            raw_cut.get("frame_index"), "detected cut frame index"
        )
        cut["master_seconds"] = _finite(
            raw_cut.get("master_seconds"), "detected cut time"
        )
        normalized_cuts.append(cut)
        if (
            raw_boundary.get("schema_version")
            != REFERENCE_DETECTED_BOUNDARY_SCHEMA_VERSION
            or raw_boundary.get("boundary_index") != boundary_index
            or raw_boundary.get("incoming_shot_index") != boundary_index + 1
            or raw_boundary.get("detector_cut") != raw_cut
        ):
            raise ValueError("detected boundary provenance is inconsistent")
        if not math.isclose(
            float(shots[boundary_index]["reference_end_seconds"]),
            cut["master_seconds"],
            abs_tol=1e-9,
        ) or not math.isclose(
            float(shots[boundary_index + 1]["reference_start_seconds"]),
            cut["master_seconds"],
            abs_tol=1e-9,
        ):
            raise ValueError("detected boundary does not partition adjacent shots")

        classification = raw_boundary.get("supervision_classification")
        cut_target = raw_boundary.get("cut_target")
        loss_masked = raw_boundary.get("loss_masked")
        old_source = raw_boundary.get("old_source_id")
        new_source = raw_boundary.get("new_source_id")
        outgoing_aggregate = _source_id(shots[boundary_index])
        incoming_aggregate = _source_id(shots[boundary_index + 1])
        raw_offsets = raw_boundary.get("boundary_sample_indices_by_offset")
        raw_boundary_sources = raw_boundary.get("boundary_path_source_ids")
        if not isinstance(raw_offsets, Mapping) or not isinstance(
            raw_boundary_sources, list
        ):
            raise ValueError("detected boundary sample provenance is missing")
        parsed_offsets: dict[int, int] = {}
        for raw_offset, raw_sample_index in raw_offsets.items():
            try:
                offset = int(raw_offset)
            except (TypeError, ValueError) as exc:
                raise ValueError("detected boundary offset is invalid") from exc
            if str(offset) != str(raw_offset) or offset not in {-2, -1, 0, 1, 2}:
                raise ValueError("detected boundary offset is unsupported")
            sample_index = _nonnegative_integer(
                raw_sample_index, "detected boundary sample index"
            )
            if sample_index >= len(path_source_ids):
                raise ValueError("detected boundary sample is outside source_path")
            parsed_offsets[offset] = sample_index
        if len(set(parsed_offsets.values())) != len(parsed_offsets):
            raise ValueError("detected boundary samples cannot repeat")
        expected_boundary_sources = [
            path_source_ids[parsed_offsets[offset]]
            for offset in (-2, -1, 0, 1, 2)
            if offset in parsed_offsets
        ]
        if raw_boundary_sources != expected_boundary_sources:
            raise ValueError("detected boundary path sources are inconsistent")
        complete_window = set(parsed_offsets) == {-2, -1, 0, 1, 2}
        if classification == "proven_cross_source_editorial_switch_positive":
            refinement = raw_boundary.get("boundary_refinement")
            shot_refinement = shots[boundary_index + 1].get(
                "boundary_refinement"
            )
            if (
                cut_target != 1.0
                or loss_masked is not False
                or not isinstance(refinement, Mapping)
                or refinement != shot_refinement
                or refinement.get("status") != "resolved_hard_cut"
                or old_source is None
                or new_source is None
                or old_source == new_source
                or outgoing_aggregate != old_source
                or incoming_aggregate != new_source
                or not complete_window
                or raw_boundary.get("dense_attribution_ready") is not True
            ):
                raise ValueError("positive detected boundary proof is inconsistent")
        elif classification == "native_source_camera_cut_hard_negative":
            if (
                cut_target != 0.0
                or loss_masked is not False
                or old_source is None
                or old_source != new_source
                or outgoing_aggregate != old_source
                or incoming_aggregate != old_source
                or raw_boundary.get("boundary_refinement") is not None
                or not complete_window
                or set(expected_boundary_sources) != {old_source}
                or raw_boundary.get("dense_attribution_ready") is not True
            ):
                raise ValueError("native-source hard negative is inconsistent")
        elif classification == "loss_masked":
            if (
                cut_target is not None
                or loss_masked is not True
                or raw_boundary.get("dense_attribution_ready") is not False
            ):
                raise ValueError("loss-masked detected boundary exposes a target")
        else:
            raise ValueError("detected boundary classification is unsupported")
        validated_boundaries.append(deepcopy(dict(raw_boundary)))

    partition_identity = {
        "master_start_seconds": float(shots[0]["reference_start_seconds"]),
        "master_end_seconds": float(shots[-1]["reference_end_seconds"]),
        "detected_reference_cuts": normalized_cuts,
        "sample_assignment": sample_assignment,
    }
    observed_hash = str(raw_schema.get("partition_content_sha256") or "")
    if len(observed_hash) != 64 or observed_hash != _canonical_json_sha256(
        partition_identity
    ):
        raise ValueError("canonical detected-shot partition hash mismatch")
    if raw_provenance.get("partition_content_sha256") != observed_hash:
        raise ValueError("canonical partition provenance hash mismatch")
    attribution_hash = str(
        raw_schema.get("attribution_content_sha256") or ""
    )
    if len(attribution_hash) != 64 or attribution_hash != _canonical_json_sha256(
        _canonical_attribution_hash_identity(
            attribution,
            shots,
            validated_boundaries,
            partition_content_sha256=observed_hash,
        )
    ):
        raise ValueError("canonical attribution content hash mismatch")
    if raw_provenance.get("attribution_content_sha256") != attribution_hash:
        raise ValueError("canonical attribution provenance hash mismatch")
    return tuple(validated_boundaries)


def _provenance_path(value: object, name: str) -> Path:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{name} path is missing")
    return Path(normalized).expanduser().resolve()


def _crop_artifact_payload(value: CropSupervisionInput) -> dict[str, Any]:
    if isinstance(value, ReferenceCropSupervisionArtifact):
        payload = deepcopy(value.payload)
        validate_reference_crop_supervision_payload(
            payload,
            verify_input_files=True,
        )
        return payload
    if isinstance(value, Path):
        return deepcopy(
            load_reference_crop_supervision(
                value,
                verify_input_files=True,
            ).payload
        )
    if isinstance(value, Mapping):
        payload = deepcopy(dict(value))
        validate_reference_crop_supervision_payload(
            payload,
            verify_input_files=True,
        )
        return payload
    raise TypeError(
        "crop supervision inputs must be artifacts, mappings, or JSON paths"
    )


def _validated_crop_supervision_by_shot(
    attribution: Mapping[str, Any],
    shots: Sequence[Mapping[str, Any]],
    inputs: Sequence[CropSupervisionInput],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Bind self-hashed crop artifacts to this exact canonical attribution.

    Artifact-level validation re-hashes all three input files.  The additional
    checks here bind those valid files to the exact in-memory attribution,
    detector-owned shot, selected source, and reference/source catalog paths
    being converted into records.
    """

    raw_schema = attribution.get("canonical_shot_schema")
    if not isinstance(raw_schema, Mapping):
        raise ValueError(
            "crop supervision integration requires a canonical detected-shot partition"
        )
    current_attribution_sha256 = _canonical_json_sha256(attribution)
    expected_reference_path = _provenance_path(
        attribution.get("reference"),
        "canonical attribution reference",
    )
    raw_sources = attribution.get("sources")
    if not isinstance(raw_sources, list):
        raise ValueError("canonical attribution source catalog must be an array")
    sources_by_id: dict[str, Mapping[str, Any]] = {}
    for source_index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, Mapping):
            raise ValueError(f"source catalog entry {source_index} must be an object")
        source_id = str(
            raw_source.get("video_id") or raw_source.get("source_id") or ""
        ).strip()
        if not source_id:
            raise ValueError(f"source catalog entry {source_index} has no source id")
        if source_id in sources_by_id:
            raise ValueError("canonical attribution source catalog repeats a source id")
        sources_by_id[source_id] = raw_source

    by_shot: dict[int, dict[str, Any]] = {}
    manifest: list[dict[str, Any]] = []
    for input_index, value in enumerate(inputs):
        payload = _crop_artifact_payload(value)
        raw_shot = payload.get("shot")
        provenance = payload.get("provenance")
        supervision = payload.get("supervision")
        if not isinstance(raw_shot, Mapping) or not isinstance(
            provenance, Mapping
        ) or not isinstance(supervision, Mapping):
            raise ValueError("crop supervision artifact is structurally incomplete")
        shot_index = _nonnegative_integer(
            raw_shot.get("shot_index"),
            "crop supervision shot index",
        )
        if shot_index >= len(shots):
            raise ValueError("crop supervision shot is outside canonical attribution")
        if shot_index in by_shot:
            raise ValueError("a canonical shot cannot have two crop supervision artifacts")
        attributed_shot = shots[shot_index]
        selected_source = _source_id(attributed_shot)
        artifact_source = str(raw_shot.get("source_id") or "").strip()
        if selected_source is None or artifact_source != selected_source:
            raise ValueError(
                "crop supervision source does not match the accepted canonical shot"
            )
        if raw_shot.get("attributed_shot_content_sha256") != _canonical_json_sha256(
            attributed_shot
        ):
            raise ValueError("crop supervision attributed-shot content hash mismatch")
        canonical = attributed_shot.get("canonical_detected_shot")
        if not isinstance(canonical, Mapping) or raw_shot.get(
            "canonical_detected_shot"
        ) != canonical:
            raise ValueError("crop supervision canonical shot identity mismatch")
        if not math.isclose(
            _finite(raw_shot.get("master_start_seconds"), "crop shot start"),
            _finite(attributed_shot.get("reference_start_seconds"), "shot start"),
            abs_tol=1e-9,
        ) or not math.isclose(
            _finite(raw_shot.get("master_end_seconds"), "crop shot end"),
            _finite(attributed_shot.get("reference_end_seconds"), "shot end"),
            abs_tol=1e-9,
        ):
            raise ValueError("crop supervision shot clock span mismatch")

        attribution_provenance = provenance.get("attribution")
        reference_provenance = provenance.get("reference_media")
        source_provenance = provenance.get("source_media")
        if not isinstance(attribution_provenance, Mapping) or not isinstance(
            reference_provenance, Mapping
        ) or not isinstance(source_provenance, Mapping):
            raise ValueError("crop supervision input provenance is incomplete")
        if (
            attribution_provenance.get("canonical_json_sha256")
            != current_attribution_sha256
        ):
            raise ValueError(
                "crop supervision attribution canonical content hash mismatch"
            )
        if _provenance_path(
            reference_provenance.get("path"),
            "crop reference media",
        ) != expected_reference_path:
            raise ValueError("crop supervision reference media path mismatch")
        catalog_source = sources_by_id.get(selected_source)
        if catalog_source is None:
            raise ValueError("crop supervision source is absent from source catalog")
        expected_source_path = _provenance_path(
            catalog_source.get("path") or catalog_source.get("local_path"),
            "canonical attribution source",
        )
        if _provenance_path(
            source_provenance.get("path"),
            "crop source media",
        ) != expected_source_path:
            raise ValueError("crop supervision source media path mismatch")

        eligible = supervision.get("eligible") is True
        loss_mask = supervision.get("loss_mask")
        if not isinstance(loss_mask, Mapping):
            raise ValueError("crop supervision loss mask is missing")
        by_shot[shot_index] = payload
        manifest.append(
            {
                "input_index": input_index,
                "shot_index": shot_index,
                "source_id": selected_source,
                "eligible": eligible,
                "rejection_reason": supervision.get("rejection_reason"),
                "loss_mask": deepcopy(dict(loss_mask)),
                "artifact_content_sha256": provenance[
                    "artifact_content_sha256"
                ],
                "attribution_file_sha256": attribution_provenance[
                    "file_sha256"
                ],
                "reference_media_sha256": reference_provenance[
                    "media_sha256"
                ],
                "source_media_sha256": source_provenance["media_sha256"],
            }
        )
    return (by_shot, manifest)


@dataclass(frozen=True, slots=True)
class ReferenceDataset:
    """Serializable reference-edit records plus an empirical style profile."""

    payload: dict[str, Any]

    @property
    def record_count(self) -> int:
        return len(self.payload["records"])

    def write(self, path: Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination.resolve()


def build_reference_dataset(
    attribution: Mapping[str, Any],
    *,
    style_id: str,
    reference_id: str | None = None,
    crop_supervision_artifacts: Sequence[CropSupervisionInput] | None = None,
) -> ReferenceDataset:
    """Build honest positive/negative edit-policy examples.

    Only conservatively accepted attributions become positives.  A visually
    detected boundary is not automatically called a source switch: adjacent
    accepted shots from the same source are labelled as broadcaster-internal
    camera cuts, matching how real stage mixes are actually assembled.
    """

    normalized_style = style_id.strip()
    if not normalized_style:
        raise ValueError("style_id cannot be empty")
    raw_shots = attribution.get("shots")
    if not isinstance(raw_shots, list) or not raw_shots:
        raise ValueError("attribution must contain a non-empty shots array")

    shots: list[Mapping[str, Any]] = []
    for index, value in enumerate(raw_shots):
        if not isinstance(value, Mapping):
            raise ValueError(f"shot {index} must be an object")
        start = _finite(value.get("reference_start_seconds"), "shot start")
        end = _finite(value.get("reference_end_seconds"), "shot end")
        if end <= start:
            raise ValueError(f"shot {index} must have positive duration")
        shots.append(value)

    reference_path = attribution.get("reference")
    inferred_reference_id = reference_id or (
        Path(str(reference_path)).stem if reference_path else "reference"
    )

    canonical_boundaries = _validated_canonical_boundaries(attribution, shots)
    crop_supervision_enabled = crop_supervision_artifacts is not None
    if crop_supervision_enabled:
        if isinstance(crop_supervision_artifacts, (str, bytes)):
            raise TypeError("crop_supervision_artifacts must be a sequence of artifacts")
        crop_supervision_by_shot, crop_supervision_manifest = (
            _validated_crop_supervision_by_shot(
                attribution,
                shots,
                tuple(crop_supervision_artifacts or ()),
            )
        )
    else:
        crop_supervision_by_shot = {}
        crop_supervision_manifest = []
    source_ids = [_source_id(shot) for shot in shots]
    dense_contexts = [
        _dense_attribution_context(attribution, shot, selected)
        for shot, selected in zip(shots, source_ids, strict=True)
    ]
    boundary_policy = _resolved_boundary_policy(attribution)
    refinements = [
        _boundary_refinement(shot, boundary_policy=boundary_policy)
        for shot in shots
    ]
    effective_starts, effective_ends = _effective_shot_spans(
        shots,
        source_ids,
        refinements,
    )
    all_durations = [
        end - start
        for start, end in zip(effective_starts, effective_ends, strict=True)
    ]
    records: list[dict[str, Any]] = []
    unknown_shot_indices: list[int] = []
    comparable_edges = 0
    source_switches = 0
    confirmed_cross_source_hard_cuts = 0
    unresolved_cross_source_transitions = 0
    same_source_boundaries = 0
    unresolved_transition_edges: list[dict[str, Any]] = []
    run_durations: list[float] = []
    current_run_source: str | None = None
    current_run_duration = 0.0
    seen_known_record = False
    canonical_boundary_by_incoming = {
        int(boundary["incoming_shot_index"]): boundary
        for boundary in canonical_boundaries or ()
    }

    for index, (shot, selected) in enumerate(zip(shots, source_ids, strict=True)):
        nominal_start = _finite(shot["reference_start_seconds"], "shot start")
        start = effective_starts[index]
        end = effective_ends[index]
        duration = end - start
        previous = source_ids[index - 1] if index > 0 else None
        refinement = refinements[index]
        canonical_boundary = canonical_boundary_by_incoming.get(index)
        if selected is None:
            unknown_shot_indices.append(index)
            if current_run_source is not None:
                run_durations.append(current_run_duration)
            current_run_source = None
            current_run_duration = 0.0
            continue

        if current_run_source == selected:
            current_run_duration += duration
        else:
            if current_run_source is not None:
                run_durations.append(current_run_duration)
            current_run_source = selected
            current_run_duration = duration

        if not seen_known_record:
            # A leading UNKNOWN attribution span is not ground truth, but the
            # first later shot with a proven source is still the observed
            # beginning of the retained training sequence.  Mark that real
            # shot as the reference start; do not reinterpret the unknown-to-
            # known detector edge as an editorial cut.
            boundary_label = "reference_start"
            transition = None
            transition_supervision_eligible = False
            previous = None
        elif index > 0 and canonical_boundary is not None:
            classification = canonical_boundary["supervision_classification"]
            if previous is not None:
                comparable_edges += 1
            if classification == "native_source_camera_cut_hard_negative":
                same_source_boundaries += 1
                boundary_label = "same_source_internal_camera_cut"
                transition = None
                transition_supervision_eligible = False
            elif (
                classification
                == "proven_cross_source_editorial_switch_positive"
            ):
                source_switches += 1
                confirmed_cross_source_hard_cuts += 1
                boundary_label = "cross_source_hard_cut"
                transition = "hard_cut"
                transition_supervision_eligible = True
            else:
                if previous is not None and previous != selected:
                    source_switches += 1
                    unresolved_cross_source_transitions += 1
                    unresolved_transition_edges.append(
                        {
                            "from_shot_index": index - 1,
                            "to_shot_index": index,
                            "old_source_id": previous,
                            "new_source_id": selected,
                            "nominal_boundary_seconds": nominal_start,
                            "status": str(
                                canonical_boundary.get(
                                    "classification_reason"
                                )
                            ),
                        }
                    )
                boundary_label = "detected_boundary_loss_masked"
                transition = None
                transition_supervision_eligible = False
        elif previous is not None:
            comparable_edges += 1
            if previous == selected:
                same_source_boundaries += 1
                boundary_label = "same_source_internal_camera_cut"
                transition = None
                transition_supervision_eligible = False
            else:
                source_switches += 1
                if (
                    refinement is not None
                    and refinement["status"] == "resolved_hard_cut"
                ):
                    confirmed_cross_source_hard_cuts += 1
                    boundary_label = "cross_source_hard_cut"
                    transition = "hard_cut"
                    transition_supervision_eligible = True
                else:
                    unresolved_cross_source_transitions += 1
                    boundary_label = "cross_source_transition_unresolved"
                    transition = None
                    transition_supervision_eligible = False
                    unresolved_transition_edges.append(
                        {
                            "from_shot_index": index - 1,
                            "to_shot_index": index,
                            "old_source_id": previous,
                            "new_source_id": selected,
                            "nominal_boundary_seconds": nominal_start,
                            "status": (
                                refinement["status"]
                                if refinement is not None
                                else "not_refined"
                            ),
                        }
                    )
        elif index == 0:
            boundary_label = "reference_start"
            transition = None
            transition_supervision_eligible = False
        else:
            boundary_label = "known_after_unknown"
            transition = None
            transition_supervision_eligible = False

        positive = _selected_features(shot)
        crop_supervision_record: dict[str, Any] | None = None
        crop_transform = positive.get("crop_transform")
        if crop_supervision_enabled:
            # Once the strict crop channel is enabled, unverified legacy crop
            # hints can no longer leak around a rejected or missing artifact.
            positive.pop("crop_transform", None)
            crop_artifact = crop_supervision_by_shot.get(index)
            if crop_artifact is None:
                crop_transform = None
                crop_supervision_record = {
                    "status": "loss_masked_missing",
                    "supervision_eligible": False,
                    "transform": None,
                    "rejection_reason": "crop_supervision_artifact_missing",
                    "loss_mask": {
                        "crop_transform_loss_masked": True,
                        "reason": "crop_supervision_artifact_missing",
                        "unsafe_or_failed_values_clamped": False,
                    },
                    "provenance": None,
                }
            else:
                artifact_supervision = crop_artifact["supervision"]
                artifact_provenance = crop_artifact["provenance"]
                if artifact_supervision["eligible"] is True:
                    crop_transform = deepcopy(
                        artifact_supervision["renderer_target"]
                    )
                    positive["crop_transform"] = deepcopy(crop_transform)
                    status = "accepted_renderer_target"
                else:
                    crop_transform = None
                    status = "loss_masked_rejected"
                crop_supervision_record = {
                    "status": status,
                    "supervision_eligible": crop_transform is not None,
                    "transform": deepcopy(crop_transform),
                    "rejection_reason": artifact_supervision[
                        "rejection_reason"
                    ],
                    "loss_mask": deepcopy(
                        artifact_supervision["loss_mask"]
                    ),
                    "provenance": {
                        "schema_version": crop_artifact["schema_version"],
                        "analysis_basis": crop_artifact["analysis_basis"],
                        "artifact_content_sha256": artifact_provenance[
                            "artifact_content_sha256"
                        ],
                        "attribution_file_sha256": artifact_provenance[
                            "attribution"
                        ]["file_sha256"],
                        "reference_media_sha256": artifact_provenance[
                            "reference_media"
                        ]["media_sha256"],
                        "source_media_sha256": artifact_provenance[
                            "source_media"
                        ]["media_sha256"],
                    },
                }
        source_time = positive.get("source_time_seconds")
        negatives = _negative_examples(shot)
        ranking_pairs = [
            {
                "chosen_source_id": selected,
                "rejected_source_id": negative["source_id"],
                "observed_score_margin": (
                    float(positive["score"]) - float(negative["score"])
                    if positive.get("score") is not None
                    and negative.get("score") is not None
                    else None
                ),
            }
            for negative in negatives
        ]
        record = {
                "record_id": f"{normalized_style}:{inferred_reference_id}:{index}",
                "style_id": normalized_style,
                "shot_index": index,
                "nominal_cut_master_time": nominal_start,
                "cut_master_time": start,
                "end_master_time": end,
                "duration_seconds": duration,
                "selected_source": selected,
                "selected_source_time": source_time,
                "previous_known_source": previous,
                "boundary_label": boundary_label,
                "boundary_event": _boundary_event(
                    previous_source=previous,
                    selected_source=selected,
                    boundary_label=boundary_label,
                    dense_attribution_ready=bool(
                        dense_contexts[index]["ready"]
                    ),
                ),
                "dense_attribution": dense_contexts[index],
                "is_source_switch": (
                    boundary_label
                    in {
                        "cross_source_hard_cut",
                        "cross_source_transition_unresolved",
                    }
                ),
                "source_switch_hard_cut_confirmed": (
                    boundary_label == "cross_source_hard_cut"
                ),
                "transition": transition,
                "transition_supervision_eligible": (
                    transition_supervision_eligible
                ),
                "boundary_refinement": refinement,
                "crop_transform": deepcopy(crop_transform),
                "positive_features": positive,
                "negative_examples": negatives,
                "ranking_pairs": ranking_pairs,
                "attribution_confidence": {
                    "decision_reason": shot.get("decision_reason"),
                    "top_two_margin": shot.get("top_two_margin"),
                },
            }
        if crop_supervision_record is not None:
            record["crop_supervision"] = crop_supervision_record
        records.append(record)
        seen_known_record = True

    if current_run_source is not None:
        run_durations.append(current_run_duration)

    detected_boundary_records: list[dict[str, Any]] = []
    for boundary in canonical_boundaries or ():
        boundary_index = int(boundary["boundary_index"])
        incoming_index = int(boundary["incoming_shot_index"])
        classification = str(boundary["supervision_classification"])
        positive = (
            classification
            == "proven_cross_source_editorial_switch_positive"
        )
        native_negative = (
            classification == "native_source_camera_cut_hard_negative"
        )
        detector_cut = boundary["detector_cut"]
        refinement = boundary.get("boundary_refinement")
        selected_time = (
            float(refinement["selected_boundary_seconds"])
            if positive and isinstance(refinement, Mapping)
            else float(detector_cut["master_seconds"])
        )
        detected_boundary_records.append(
            {
                "record_id": (
                    f"{normalized_style}:{inferred_reference_id}:"
                    f"detected-boundary:{boundary_index}"
                ),
                "style_id": normalized_style,
                "reference_id": inferred_reference_id,
                "boundary_index": boundary_index,
                "outgoing_shot_index": incoming_index - 1,
                "incoming_shot_index": incoming_index,
                "detector_frame_index": detector_cut["frame_index"],
                "nominal_cut_master_time": detector_cut["master_seconds"],
                "cut_master_time": selected_time,
                "old_source_id": boundary.get("old_source_id"),
                "new_source_id": boundary.get("new_source_id"),
                "classification": classification,
                "classification_reason": boundary.get(
                    "classification_reason"
                ),
                "cut_target": boundary.get("cut_target"),
                "loss_masked": bool(boundary.get("loss_masked")),
                "dense_attribution_ready": bool(
                    boundary.get("dense_attribution_ready")
                ),
                "target_masks": {
                    "cut": positive or native_negative,
                    "source_selection": positive,
                    "duration": positive,
                    "transition": positive,
                    "crop": False,
                },
                "is_editorial_source_switch_positive": positive,
                "is_native_source_camera_cut_hard_negative": native_negative,
                "boundary_refinement": deepcopy(refinement),
                "boundary_path_source_ids": deepcopy(
                    boundary.get("boundary_path_source_ids", [])
                ),
                "boundary_sample_indices_by_offset": deepcopy(
                    boundary.get("boundary_sample_indices_by_offset", {})
                ),
                "provenance": {
                    "schema_version": boundary.get("schema_version"),
                    "partition_content_sha256": attribution[
                        "canonical_shot_schema"
                    ]["partition_content_sha256"],
                    "attribution_content_sha256": attribution[
                        "canonical_shot_schema"
                    ]["attribution_content_sha256"],
                    "reference_hard_cut_detector_owns_partition": True,
                    "sample_run_merge_cannot_remove_boundary": True,
                },
            }
        )

    usage = Counter(source for source in source_ids if source is not None)
    known_count = len(records)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_kind": "reference_stage_mix_source_ranking",
        "style_id": normalized_style,
        "reference_id": inferred_reference_id,
        "reference_path": reference_path,
        "attribution_analysis": attribution.get("analysis"),
        "dense_attribution_schema": attribution.get(
            "dense_attribution_schema"
        ),
        "decision_policy": attribution.get("decision_policy"),
        "source_catalog": attribution.get("sources", []),
        "label_policy": {
            "unknown_is_not_ground_truth": True,
            "unknown_shots_excluded_from_supervised_records": True,
            "same_source_visual_boundary_is_internal_camera_cut": True,
            "native_source_camera_cut_is_not_editorial_cut_positive": True,
            "native_source_camera_cut_is_not_source_selection_supervision": True,
            "editorial_cut_positive_requires_cross_source_hard_cut": True,
            "negative_examples_are_synchronized_unselected_candidates": True,
            "transition_type_is_not_invented_when_unobserved": True,
            "cross_source_state_change_is_not_automatically_hard_cut": True,
            "hard_cut_requires_resolved_boundary_refinement": True,
            "unresolved_transitions_excluded_from_transition_supervision": True,
            "detected_boundary_positive_requires_dense_attribution": True,
            "detected_boundary_unknown_or_gate_failure_is_loss_masked": True,
        },
        "style_profile": {
            "shot_duration_seconds": _distribution(all_durations),
            "known_source_run_duration_seconds": _distribution(run_durations),
            "source_switch_rate_on_comparable_edges": (
                source_switches / comparable_edges if comparable_edges else None
            ),
            "known_rate": known_count / len(shots),
            "shot_count": len(shots),
            "known_shot_count": known_count,
            "unknown_shot_count": len(unknown_shot_indices),
            "comparable_adjacent_edges": comparable_edges,
            "observed_cross_source_switches": source_switches,
            "cross_source_hard_cuts": confirmed_cross_source_hard_cuts,
            "unresolved_cross_source_transitions": (
                unresolved_cross_source_transitions
            ),
            "confirmed_hard_cut_rate_on_cross_source_switches": (
                confirmed_cross_source_hard_cuts / source_switches
                if source_switches
                else None
            ),
            "same_source_internal_camera_cuts": same_source_boundaries,
            "editorial_source_switch_supervision_events": (
                confirmed_cross_source_hard_cuts
            ),
            "dense_attribution_ready_shots": sum(
                bool(context["ready"]) for context in dense_contexts
            ),
            "unique_attributed_sources": len(usage),
            "source_usage_by_shot": dict(usage.most_common()),
        },
        "unresolved_shot_indices": unknown_shot_indices,
        "unresolved_transition_edges": unresolved_transition_edges,
        "records": records,
    }
    if crop_supervision_enabled:
        known_shot_indices = {
            index for index, source in enumerate(source_ids) if source is not None
        }
        accepted_crop_indices = sorted(
            shot_index
            for shot_index, artifact in crop_supervision_by_shot.items()
            if artifact["supervision"]["eligible"] is True
        )
        rejected_crop_indices = sorted(
            shot_index
            for shot_index, artifact in crop_supervision_by_shot.items()
            if artifact["supervision"]["eligible"] is not True
        )
        missing_crop_indices = sorted(
            known_shot_indices - crop_supervision_by_shot.keys()
        )
        payload["label_policy"].update(
            {
                "crop_target_requires_validated_multiframe_artifact": True,
                "crop_rejected_or_missing_is_loss_masked": True,
                "legacy_crop_hint_disabled_when_crop_artifacts_are_enabled": True,
                "crop_failure_values_are_never_clamped": True,
            }
        )
        payload["style_profile"].update(
            {
                "crop_supervision_eligible_shots": len(accepted_crop_indices),
                "crop_supervision_loss_masked_shots": len(
                    rejected_crop_indices
                )
                + len(missing_crop_indices),
            }
        )
        payload["crop_supervision_schema"] = {
            "schema_version": REFERENCE_CROP_SUPERVISION_SCHEMA_VERSION,
            "analysis_basis": REFERENCE_CROP_SUPERVISION_ANALYSIS_BASIS,
            "integration_basis": (
                "self_hashed_artifact_plus_exact_attribution_shot_media_provenance_v1"
            ),
            "renderer_coordinate_system": (
                "EditorialTransformKeyframe_zoom_center_x_center_y_roll_degrees"
            ),
            "strict_input_file_sha256_revalidation": True,
            "accepted_target_count": len(accepted_crop_indices),
            "rejected_artifact_count": len(rejected_crop_indices),
            "missing_artifact_count": len(missing_crop_indices),
            "accepted_shot_indices": accepted_crop_indices,
            "rejected_shot_indices": rejected_crop_indices,
            "missing_shot_indices": missing_crop_indices,
            "loss_masked_shot_indices": sorted(
                {*rejected_crop_indices, *missing_crop_indices}
            ),
            "artifact_manifest": crop_supervision_manifest,
        }
    if canonical_boundaries is not None:
        payload.update(
            {
                "canonical_shot_schema": deepcopy(
                    attribution["canonical_shot_schema"]
                ),
                "reference_evidence_provenance": deepcopy(
                    attribution["reference_evidence_provenance"]
                ),
                "detected_boundary_records": detected_boundary_records,
                "detected_boundary_statistics": {
                    "detected_boundary_count": len(detected_boundary_records),
                    "editorial_switch_positive_count": sum(
                        record["is_editorial_source_switch_positive"]
                        for record in detected_boundary_records
                    ),
                    "native_source_camera_cut_hard_negative_count": sum(
                        record["is_native_source_camera_cut_hard_negative"]
                        for record in detected_boundary_records
                    ),
                    "loss_masked_count": sum(
                        record["loss_masked"]
                        for record in detected_boundary_records
                    ),
                    "all_detector_boundaries_preserved": (
                        len(detected_boundary_records)
                        == len(attribution["detected_reference_cuts"])
                    ),
                },
            }
        )
    return ReferenceDataset(payload)


def build_reference_dataset_file(
    attribution_path: Path,
    output_path: Path,
    *,
    style_id: str,
    reference_id: str | None = None,
    crop_supervision_paths: Sequence[Path] | None = None,
) -> Path:
    source = Path(attribution_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("attribution JSON root must be an object")
    dataset = build_reference_dataset(
        payload,
        style_id=style_id,
        reference_id=reference_id,
        crop_supervision_artifacts=crop_supervision_paths,
    )
    return dataset.write(Path(output_path))


__all__ = [
    "BOUNDARY_EVENT_SCHEMA_VERSION",
    "DETECTED_BOUNDARY_LOSS_MASKED_EVENT",
    "EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT",
    "EDITORIAL_SOURCE_SWITCH_UNRESOLVED_EVENT",
    "KNOWN_AFTER_UNKNOWN_EVENT",
    "NATIVE_SOURCE_CAMERA_CUT_EVENT",
    "REFERENCE_START_EVENT",
    "CropSupervisionInput",
    "ReferenceDataset",
    "SCHEMA_VERSION",
    "build_reference_dataset",
    "build_reference_dataset_file",
]
