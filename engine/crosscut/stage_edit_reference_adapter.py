"""Leakage-safe StageEdit examples from reference-supervision JSON.

The reverse matcher needs ORB/NCC/color evidence to recover the editor's
choice.  Those measurements are *labels*, not inputs available when StageEdit
has to edit a new song.  This adapter therefore uses that evidence only to
identify the selected/rejected source and deliberately omits every match
score, observed rank, homography, and crop proxy from model features.

All inputs now pass through the same feature materializer used by inference.
Without an external snapshot it populates the inference-available catalog,
timeline progress and causal edit history.  A caller may instead supply a
schema-bound, content-hashed feature snapshot containing richer rhythm,
section and source-local visual observations produced before editorial source
choice.  Every unavailable scalar retains an explicit missing bit.  Cut
negatives are sampled strictly inside known shot holds and never near either
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .reference_attribution import (
    REFERENCE_ATTRIBUTION_BASIS,
    REFERENCE_CANONICAL_SHOT_SCHEMA_VERSION,
    REFERENCE_DENSE_ATTRIBUTION_SCHEMA_VERSION,
    REFERENCE_DETECTED_BOUNDARY_SCHEMA_VERSION,
    REFERENCE_EVIDENCE_ANALYSIS_BASIS,
)
from .reference_dataset import (
    BOUNDARY_EVENT_SCHEMA_VERSION,
    DETECTED_BOUNDARY_LOSS_MASKED_EVENT,
    EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT,
    EDITORIAL_SOURCE_SWITCH_UNRESOLVED_EVENT,
    KNOWN_AFTER_UNKNOWN_EVENT,
    NATIVE_SOURCE_CAMERA_CUT_EVENT,
    REFERENCE_START_EVENT,
)
from .stage_edit_feature_materializer import (
    CausalEditDecision,
    FEATURE_MATERIALIZER_CONTRACT,
    SourceFeatureObservation,
    StageEditFeatureSnapshot,
    materialize_stage_edit_source_candidate,
    materialize_stage_edit_token,
    subset_stage_edit_token_sources,
)
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    SourceCandidateToken,
    StageEditSequence,
    StageEditTarget,
    StageEditTargetSequence,
    StageEditToken,
    StageEditTokenSchema,
)


ADAPTER_SCHEMA_VERSION = 5
EXPECTED_DATASET_KIND = "reference_stage_mix_pairwise_listwise_supervision"
HOLD_NEGATIVE_EVENT = "known_source_hold_negative"

_BOUNDARY_EVENT_TYPES = frozenset(
    {
        REFERENCE_START_EVENT,
        DETECTED_BOUNDARY_LOSS_MASKED_EVENT,
        NATIVE_SOURCE_CAMERA_CUT_EVENT,
        EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT,
        EDITORIAL_SOURCE_SWITCH_UNRESOLVED_EVENT,
        KNOWN_AFTER_UNKNOWN_EVENT,
    }
)

# These fields may determine labels but must never become model inputs.
PROHIBITED_REFERENCE_MATCH_INPUT_FIELDS = (
    "score",
    "observed_match_rank",
    "observed_score_margin",
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


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalized_id(value: Any) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


@dataclass(frozen=True, slots=True)
class ReferenceTrainingExample:
    reference_id: str
    song_id: str | None
    style_id: str
    sequence: StageEditSequence
    targets: StageEditTargetSequence
    statistics: Mapping[str, Any]
    feature_snapshot_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ReferenceTrainingCorpus:
    examples: tuple[ReferenceTrainingExample, ...]
    data_id: str
    source_paths: tuple[str, ...]
    statistics: Mapping[str, Any]

    def manifest(self) -> dict[str, Any]:
        """Return a compact, feature-free provenance manifest."""

        return {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "dataset_kind": "stage_edit_leakage_safe_training_corpus",
            "data_id": self.data_id,
            "source_paths": list(self.source_paths),
            "input_policy": {
                "reference_match_evidence_used_only_as_labels": True,
                "candidate_order_is_canonical_not_observed_rank": True,
                "prohibited_reference_match_input_fields": list(
                    PROHIBITED_REFERENCE_MATCH_INPUT_FIELDS
                ),
                "unavailable_features_have_explicit_missing_masks": True,
                "cut_negatives_are_inside_known_shot_holds": True,
                "native_source_camera_cuts_are_editorial_cut_negatives": True,
                "native_source_camera_cuts_never_supervise_source_switches": True,
                "production_gate_requires_dense_cross_source_events": True,
                "training_and_inference_share_feature_materializer": (
                    FEATURE_MATERIALIZER_CONTRACT
                ),
                "optional_feature_snapshots_are_schema_and_content_hashed": True,
                "canonical_detector_boundaries_override_known_shot_edges": True,
                "loss_masked_detector_boundaries_remain_tokens": True,
            },
            "examples": [
                {
                    "reference_id": example.reference_id,
                    "song_id": example.song_id,
                    "style_id": example.style_id,
                    "sequence_id": example.sequence.sequence_id,
                    "token_count": len(example.sequence.tokens),
                    "feature_snapshot_sha256": (
                        example.feature_snapshot_sha256
                    ),
                    "statistics": dict(example.statistics),
                }
                for example in self.examples
            ],
            "statistics": dict(self.statistics),
        }

    def write_manifest(self, path: Path) -> Path:
        """Write only when the caller supplies an explicit destination."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.manifest(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination.resolve()


@dataclass(frozen=True, slots=True)
class _TrainingEvent:
    time_seconds: float
    shot_index: int
    event_kind: str
    candidates: tuple[SourceCandidateToken, ...]
    target: StageEditTarget
    explicit_event_schema: bool = False
    dense_attribution_ready: bool = False
    causal_source_id: str | None = None


@dataclass(frozen=True, slots=True)
class _BoundaryEventLabel:
    event_type: str
    explicit_schema: bool
    dense_attribution_ready: bool


@dataclass(frozen=True, slots=True)
class _CanonicalDetectedBoundary:
    boundary_index: int
    incoming_shot_index: int
    time_seconds: float
    nominal_time_seconds: float
    event_type: str
    old_source_id: str | None
    new_source_id: str | None
    cut_target: float | None
    dense_attribution_ready: bool
    record: Mapping[str, Any]


def _dense_payload_provenance_ready(payload: Mapping[str, Any]) -> bool:
    raw = payload.get("dense_attribution_schema")
    if raw is None:
        return False
    if not isinstance(raw, Mapping):
        raise ValueError("dense_attribution_schema must be an object")
    if payload.get("attribution_analysis") != REFERENCE_ATTRIBUTION_BASIS:
        raise ValueError("dense attribution analysis basis is unsupported")
    if raw.get("schema_version") != REFERENCE_DENSE_ATTRIBUTION_SCHEMA_VERSION:
        raise ValueError("unsupported dense attribution schema version")
    required_flags = (
        "sample_level_source_path_available",
        "shot_spans_reference_contiguous_source_path_samples",
        "frame_dense_boundary_proof_required_for_hard_cut",
        "native_source_camera_cut_is_not_editorial_source_switch",
    )
    if not all(bool(raw.get(name)) for name in required_flags):
        raise ValueError("dense attribution schema is missing a safety contract")
    return raw.get("temporal_granularity") == "synchronized_sample_path"


def _sha256_string(value: Any, name: str) -> str:
    normalized = str(value or "").strip()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return normalized


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _strict_nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _canonical_detected_boundaries(
    payload: Mapping[str, Any],
    *,
    dense_payload_provenance_ready: bool,
) -> tuple[_CanonicalDetectedBoundary, ...] | None:
    """Validate the detector-owned boundary partition copied into supervision."""

    raw_records = payload.get("detected_boundary_records")
    if raw_records is None or raw_records == []:
        return None
    if not isinstance(raw_records, list):
        raise ValueError("detected_boundary_records must be an array")
    raw_schema = payload.get("canonical_shot_schema")
    raw_provenance = payload.get("reference_evidence_provenance")
    if not isinstance(raw_schema, Mapping) or not isinstance(
        raw_provenance, Mapping
    ):
        raise ValueError("canonical boundary records lack partition provenance")
    required_schema = {
        "schema_version": REFERENCE_CANONICAL_SHOT_SCHEMA_VERSION,
        "partition_basis": "reference_hard_cut_detector",
        "reference_evidence_analysis": REFERENCE_EVIDENCE_ANALYSIS_BASIS,
        "source_runs_are_attribution_only_not_shot_partitions": True,
        "same_source_adjacent_detector_shots_are_never_merged": True,
        "unknown_or_failed_boundary_gate_is_loss_masked": True,
        "hash_basis": (
            "sha256_canonical_json_of_master_bounds_detected_cuts_and_"
            "sample_assignment"
        ),
        "attribution_hash_basis": (
            "sha256_canonical_json_of_partition_path_shot_aggregates_and_"
            "boundary_labels"
        ),
    }
    for name, expected in required_schema.items():
        if raw_schema.get(name) != expected:
            raise ValueError(f"canonical boundary schema {name} is unsupported")
    partition_sha256 = _sha256_string(
        raw_schema.get("partition_content_sha256"),
        "canonical partition identity",
    )
    attribution_sha256 = _sha256_string(
        raw_schema.get("attribution_content_sha256"),
        "canonical attribution identity",
    )
    if (
        raw_provenance.get("schema_version") != 1
        or raw_provenance.get("analysis") != REFERENCE_EVIDENCE_ANALYSIS_BASIS
        or raw_provenance.get("partition_content_sha256") != partition_sha256
        or raw_provenance.get("attribution_content_sha256")
        != attribution_sha256
    ):
        raise ValueError("canonical reference evidence provenance is inconsistent")
    if not dense_payload_provenance_ready:
        raise ValueError("canonical boundaries lack dense payload provenance")
    expected_count = _strict_nonnegative_integer(
        raw_schema.get("detected_cut_count"), "canonical detected cut count"
    )
    shot_count = _strict_nonnegative_integer(
        raw_schema.get("canonical_shot_count"), "canonical shot count"
    )
    if expected_count != len(raw_records) or shot_count != expected_count + 1:
        raise ValueError("canonical boundary counts do not match the partition")

    normalized: list[_CanonicalDetectedBoundary] = []
    previous_frame = -1
    previous_nominal = -math.inf
    positive_count = 0
    native_count = 0
    masked_count = 0
    reference_id = _normalized_id(payload.get("reference_id"))
    style_id = _normalized_id(payload.get("style_id"))
    for expected_index, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise ValueError("canonical detected boundary must be an object")
        boundary_index = _strict_nonnegative_integer(
            raw.get("boundary_index"), "canonical boundary index"
        )
        outgoing_index = _strict_nonnegative_integer(
            raw.get("outgoing_shot_index"), "canonical outgoing shot index"
        )
        incoming_index = _strict_nonnegative_integer(
            raw.get("incoming_shot_index"), "canonical incoming shot index"
        )
        frame_index = _strict_nonnegative_integer(
            raw.get("detector_frame_index"), "canonical detector frame index"
        )
        if (
            boundary_index != expected_index
            or outgoing_index != boundary_index
            or incoming_index != boundary_index + 1
            or frame_index <= previous_frame
        ):
            raise ValueError("canonical detected-boundary ordering is inconsistent")
        nominal = _finite(raw.get("nominal_cut_master_time"))
        timestamp = _finite(raw.get("cut_master_time"))
        if (
            nominal is None
            or timestamp is None
            or nominal < 0.0
            or timestamp < 0.0
            or nominal <= previous_nominal
        ):
            raise ValueError("canonical detected-boundary time is invalid")
        if (
            _normalized_id(raw.get("reference_id")) != reference_id
            or _normalized_id(raw.get("style_id")) != style_id
        ):
            raise ValueError("canonical boundary dataset identity is inconsistent")
        record_provenance = raw.get("provenance")
        if not isinstance(record_provenance, Mapping) or (
            record_provenance.get("schema_version")
            != REFERENCE_DETECTED_BOUNDARY_SCHEMA_VERSION
            or record_provenance.get("partition_content_sha256")
            != partition_sha256
            or record_provenance.get("attribution_content_sha256")
            != attribution_sha256
            or record_provenance.get(
                "reference_hard_cut_detector_owns_partition"
            )
            is not True
            or record_provenance.get("sample_run_merge_cannot_remove_boundary")
            is not True
        ):
            raise ValueError("canonical boundary provenance is inconsistent")
        masks = raw.get("target_masks")
        if not isinstance(masks, Mapping) or set(masks) != {
            "cut",
            "source_selection",
            "duration",
            "transition",
            "crop",
        }:
            raise ValueError("canonical boundary target masks are invalid")
        mask_values = {
            name: _strict_bool(masks.get(name), f"canonical {name} target mask")
            for name in masks
        }
        classification = _normalized_id(raw.get("classification"))
        old_source = _normalized_id(raw.get("old_source_id"))
        new_source = _normalized_id(raw.get("new_source_id"))
        dense_ready = _strict_bool(
            raw.get("dense_attribution_ready"),
            "canonical dense-attribution readiness",
        )
        loss_masked = _strict_bool(
            raw.get("loss_masked"), "canonical boundary loss mask"
        )
        editorial_flag = _strict_bool(
            raw.get("is_editorial_source_switch_positive"),
            "canonical editorial-positive flag",
        )
        native_flag = _strict_bool(
            raw.get("is_native_source_camera_cut_hard_negative"),
            "canonical native-negative flag",
        )
        cut_target = _finite(raw.get("cut_target"))
        path_sources = raw.get("boundary_path_source_ids")
        offsets = raw.get("boundary_sample_indices_by_offset")
        missing_window_path = (
            classification == "loss_masked"
            and _normalized_id(raw.get("classification_reason"))
            == "missing_boundary_window_samples"
            and path_sources == []
            and isinstance(offsets, Mapping)
            and not offsets
        )
        if not missing_window_path and (
            not isinstance(path_sources, list)
            or len(path_sources) != 5
            or not isinstance(offsets, Mapping)
            or set(offsets) != {"-2", "-1", "0", "1", "2"}
        ):
            raise ValueError("canonical boundary lacks a complete five-sample path")
        sample_indices = (
            ()
            if missing_window_path
            else tuple(
                _strict_nonnegative_integer(
                    offsets[str(offset)], "canonical boundary sample index"
                )
                for offset in (-2, -1, 0, 1, 2)
            )
        )
        if sample_indices and (len(set(sample_indices)) != 5 or any(
            right <= left for left, right in zip(sample_indices, sample_indices[1:])
        )):
            raise ValueError("canonical boundary sample path is not ordered")

        if classification == "proven_cross_source_editorial_switch_positive":
            expected_masks = {
                "cut": True,
                "source_selection": True,
                "duration": True,
                "transition": True,
                "crop": False,
            }
            refinement = raw.get("boundary_refinement")
            if (
                cut_target != 1.0
                or loss_masked
                or not editorial_flag
                or native_flag
                or not dense_ready
                or old_source is None
                or new_source is None
                or old_source == new_source
                or mask_values != expected_masks
                or not isinstance(refinement, Mapping)
                or refinement.get("status") != "resolved_hard_cut"
                or _normalized_id(refinement.get("old_source_id")) != old_source
                or _normalized_id(refinement.get("new_source_id")) != new_source
                or _finite(refinement.get("selected_boundary_seconds"))
                != timestamp
                or tuple(_normalized_id(value) for value in path_sources[:2])
                != (old_source, old_source)
                or tuple(_normalized_id(value) for value in path_sources[-2:])
                != (new_source, new_source)
            ):
                raise ValueError("canonical editorial-switch positive is inconsistent")
            event_type = EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT
            positive_count += 1
        elif classification == "native_source_camera_cut_hard_negative":
            expected_masks = {
                "cut": True,
                "source_selection": False,
                "duration": False,
                "transition": False,
                "crop": False,
            }
            if (
                cut_target != 0.0
                or loss_masked
                or editorial_flag
                or not native_flag
                or not dense_ready
                or old_source is None
                or old_source != new_source
                or mask_values != expected_masks
                or raw.get("boundary_refinement") is not None
                or any(_normalized_id(value) != old_source for value in path_sources)
                or not math.isclose(timestamp, nominal, abs_tol=1e-9)
            ):
                raise ValueError("canonical native-source negative is inconsistent")
            event_type = NATIVE_SOURCE_CAMERA_CUT_EVENT
            native_count += 1
        elif classification == "loss_masked":
            if (
                raw.get("cut_target") is not None
                or cut_target is not None
                or not loss_masked
                or editorial_flag
                or native_flag
                or dense_ready
                or any(mask_values.values())
            ):
                raise ValueError("canonical loss-masked boundary is inconsistent")
            event_type = DETECTED_BOUNDARY_LOSS_MASKED_EVENT
            masked_count += 1
        else:
            raise ValueError("canonical boundary classification is unsupported")
        normalized.append(
            _CanonicalDetectedBoundary(
                boundary_index=boundary_index,
                incoming_shot_index=incoming_index,
                time_seconds=timestamp,
                nominal_time_seconds=nominal,
                event_type=event_type,
                old_source_id=old_source,
                new_source_id=new_source,
                cut_target=cut_target,
                dense_attribution_ready=dense_ready,
                record=raw,
            )
        )
        previous_frame = frame_index
        previous_nominal = nominal

    statistics = payload.get("detected_boundary_statistics")
    if not isinstance(statistics, Mapping) or (
        statistics.get("detected_boundary_count") != len(normalized)
        or statistics.get("editorial_switch_positive_count") != positive_count
        or statistics.get("native_source_camera_cut_hard_negative_count")
        != native_count
        or statistics.get("loss_masked_count") != masked_count
        or statistics.get("all_detector_boundaries_preserved") is not True
    ):
        raise ValueError("canonical detected-boundary statistics are inconsistent")
    return tuple(normalized)


def _boundary_event_label(
    labels: Mapping[str, Any],
    *,
    dense_payload_provenance_ready: bool,
) -> _BoundaryEventLabel:
    raw = labels.get("boundary_event")
    cut_labels = labels.get("cut")
    boundary_label = (
        _normalized_id(cut_labels.get("boundary_label"))
        if isinstance(cut_labels, Mapping)
        else None
    )
    legacy_map = {
        "reference_start": REFERENCE_START_EVENT,
        "same_source_internal_camera_cut": NATIVE_SOURCE_CAMERA_CUT_EVENT,
        "cross_source_hard_cut": EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT,
        "cross_source_transition_unresolved": (
            EDITORIAL_SOURCE_SWITCH_UNRESOLVED_EVENT
        ),
        "known_after_unknown": KNOWN_AFTER_UNKNOWN_EVENT,
        "detected_boundary_loss_masked": DETECTED_BOUNDARY_LOSS_MASKED_EVENT,
    }
    expected = legacy_map.get(boundary_label)
    if expected is None:
        raise ValueError("boundary cut label is missing or unsupported")
    if raw is None:
        # Legacy supervision remains usable for offline calibration, but never
        # satisfies the dense production-supervision gate.
        return _BoundaryEventLabel(expected, False, False)
    if not isinstance(raw, Mapping):
        raise ValueError("boundary_event must be an object")
    if raw.get("schema_version") != BOUNDARY_EVENT_SCHEMA_VERSION:
        raise ValueError("unsupported boundary_event schema version")
    event_type = _normalized_id(raw.get("event_type"))
    if event_type not in _BOUNDARY_EVENT_TYPES:
        raise ValueError("boundary_event has an unsupported event type")
    if event_type != expected:
        raise ValueError("boundary_event conflicts with the legacy boundary label")
    is_native = event_type == NATIVE_SOURCE_CAMERA_CUT_EVENT
    is_editorial = event_type in {
        EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT,
        EDITORIAL_SOURCE_SWITCH_UNRESOLVED_EVENT,
    }
    if bool(raw.get("is_native_source_camera_cut")) != is_native:
        raise ValueError("boundary_event native-source flag is inconsistent")
    if bool(raw.get("is_editorial_source_switch")) != is_editorial:
        raise ValueError("boundary_event editorial-switch flag is inconsistent")
    transition = labels.get("source_transition")
    if not isinstance(transition, Mapping):
        raise ValueError("explicit boundary_event requires source_transition")
    if bool(transition.get("is_source_switch")) != is_editorial:
        raise ValueError("source_transition conflicts with boundary_event")
    for event_name, transition_name in (
        ("previous_source_id", "previous_source_id"),
        ("selected_source_id", "selected_source_id"),
    ):
        if raw.get(event_name) != transition.get(transition_name):
            raise ValueError(
                f"boundary_event {event_name} conflicts with source_transition"
            )
    expected_eligible = event_type == EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT
    for name in (
        "editorial_timing_supervision_eligible",
        "editorial_source_selection_supervision_eligible",
        "editorial_duration_supervision_eligible",
        "editorial_transition_supervision_eligible",
    ):
        if bool(raw.get(name)) != expected_eligible:
            raise ValueError(f"boundary_event {name} is inconsistent")
    dense_ready = bool(raw.get("dense_attribution_ready"))
    dense_record = labels.get("dense_attribution")
    if not isinstance(dense_record, Mapping):
        raise ValueError("explicit boundary_event requires dense_attribution metadata")
    if bool(dense_record.get("ready")) != dense_ready:
        raise ValueError("boundary_event dense-attribution flag is inconsistent")
    if dense_ready:
        if not dense_payload_provenance_ready:
            raise ValueError(
                "dense boundary event lacks trusted payload-level provenance"
            )
        if (
            dense_record.get("schema_version")
            != REFERENCE_DENSE_ATTRIBUTION_SCHEMA_VERSION
            or dense_record.get("reason")
            != "validated_synchronized_sample_path"
        ):
            raise ValueError("dense boundary event metadata is unsupported")
        sample_count = dense_record.get("sample_count")
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count < 1
        ):
            raise ValueError("dense boundary event sample count is invalid")
    return _BoundaryEventLabel(
        event_type=event_type,
        explicit_schema=True,
        dense_attribution_ready=dense_ready,
    )


def _catalog_by_source(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = payload.get("source_catalog")
    if not isinstance(raw, list):
        return {}
    output: dict[str, Mapping[str, Any]] = {}
    for record in raw:
        if not isinstance(record, Mapping):
            continue
        source_id = _normalized_id(record.get("video_id", record.get("source_id")))
        if source_id is not None:
            output[source_id] = record
    return output


def _source_token(
    source_id: str,
    catalog: Mapping[str, Mapping[str, Any]],
    schema: StageEditTokenSchema,
) -> SourceCandidateToken:
    record = catalog.get(source_id, {})
    alignment_score = _finite(record.get("alignment_score"))
    return materialize_stage_edit_source_candidate(
        SourceFeatureObservation(
            source_id=source_id,
            sync_confidence=alignment_score,
            available=True,
            hard_gate_passed=True,
        ),
        schema=schema,
    )


def _source_observation(
    source_id: str,
    catalog: Mapping[str, Mapping[str, Any]],
) -> SourceFeatureObservation:
    record = catalog.get(source_id, {})
    return SourceFeatureObservation(
        source_id=source_id,
        sync_confidence=_finite(record.get("alignment_score")),
        available=True,
        hard_gate_passed=True,
    )


def _ranking_by_shot(payload: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    raw = payload.get("listwise_records")
    if not isinstance(raw, list):
        return {}
    output: dict[int, Mapping[str, Any]] = {}
    for record in raw:
        if not isinstance(record, Mapping):
            continue
        index = record.get("shot_index")
        if isinstance(index, int) and not isinstance(index, bool):
            output[index] = record
    return output


def _candidate_ids(
    shot: Mapping[str, Any],
    ranking: Mapping[str, Any] | None,
    *,
    max_sources: int,
    catalog_source_ids: Sequence[str] = (),
) -> tuple[str, ...]:
    # The synchronized source catalog is available before any editorial
    # choice.  Prefer it over reverse-match ranking membership, which could
    # otherwise leak which sources happened to compare well to the reference.
    observed: set[str] = {
        source_id
        for raw_source_id in catalog_source_ids
        for source_id in (_normalized_id(raw_source_id),)
        if source_id is not None
    }
    if not observed and ranking is not None:
        candidates = ranking.get("candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    continue
                source_id = _normalized_id(candidate.get("source_id"))
                if source_id is not None:
                    observed.add(source_id)
    # Canonical ordering is essential: observed match rank would trivially leak
    # that the selected source was rank zero in the recovered reference.
    return tuple(sorted(observed)[:max_sources])


def _event_candidates(
    *,
    timestamp: float,
    shot: Mapping[str, Any],
    ranking: Mapping[str, Any] | None,
    catalog: Mapping[str, Mapping[str, Any]],
    feature_snapshot: StageEditFeatureSnapshot | None,
    max_sources: int,
    schema: StageEditTokenSchema,
) -> tuple[SourceCandidateToken, ...]:
    if feature_snapshot is not None:
        snapshot_candidates = feature_snapshot.token_at(
            timestamp
        ).source_candidates
        if len(snapshot_candidates) > max_sources:
            raise ValueError(
                "StageEdit feature snapshot source count exceeds max_sources"
            )
        if len(snapshot_candidates) < 2:
            raise ValueError("StageEdit training event needs at least two sources")
        return tuple(
            sorted(snapshot_candidates, key=lambda item: item.source_id)
        )
    else:
        candidate_ids = _candidate_ids(
            shot,
            ranking,
            max_sources=max_sources,
            catalog_source_ids=tuple(catalog),
        )
    if len(candidate_ids) < 2:
        raise ValueError("StageEdit training event needs at least two sources")
    return tuple(
        _source_token(source_id, catalog, schema) for source_id in candidate_ids
    )


def _crop_target(labels: Mapping[str, Any]) -> tuple[float | None, ...] | None:
    crop = labels.get("crop")
    if not isinstance(crop, Mapping) or not bool(crop.get("supervision_eligible")):
        return None
    transform = crop.get("transform")
    if not isinstance(transform, Mapping):
        return None
    values = (
        _finite(transform.get("zoom", transform.get("scale"))),
        _finite(transform.get("center_x", transform.get("x"))),
        _finite(transform.get("center_y", transform.get("y"))),
        _finite(transform.get("roll_degrees", transform.get("roll"))),
    )
    return values if any(value is not None for value in values) else None


def _boundary_target(
    shot: Mapping[str, Any],
    ranking: Mapping[str, Any] | None,
    schema: StageEditTokenSchema,
    event_label: _BoundaryEventLabel,
) -> StageEditTarget:
    labels = shot.get("edit_labels")
    if not isinstance(labels, Mapping):
        return StageEditTarget()
    cut_labels = labels.get("cut")
    cut: float | None = None
    duration_bucket: int | None = None
    if isinstance(cut_labels, Mapping):
        if event_label.event_type == EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT:
            cut = 1.0
        elif event_label.event_type == NATIVE_SOURCE_CAMERA_CUT_EVENT:
            # The displayed image changed inside the chosen broadcast source;
            # the reference editor did not switch source here.
            cut = 0.0
        duration = _finite(cut_labels.get("duration_seconds"))
        if (
            event_label.event_type == EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT
            and duration is not None
            and duration >= 0.0
        ):
            duration_bucket = schema.duration_bucket(duration)

    selected: str | None = None
    rejected: tuple[str, ...] = ()
    if (
        event_label.event_type == EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT
        and ranking is not None
    ):
        candidate_records = ranking.get("candidates")
        chosen = _normalized_id(ranking.get("chosen_source_id"))
        if chosen is not None and isinstance(candidate_records, list):
            observed_ids = {
                source_id
                for candidate in candidate_records
                if isinstance(candidate, Mapping)
                for source_id in (_normalized_id(candidate.get("source_id")),)
                if source_id is not None
            }
            if chosen in observed_ids:
                selected = chosen
                rejected = tuple(sorted(observed_ids - {chosen}))
                if not rejected:
                    selected = None

    transition_label: str | None = None
    transition = labels.get("source_transition")
    if (
        event_label.event_type == EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT
        and isinstance(transition, Mapping)
        and bool(transition.get("supervision_eligible"))
    ):
        observed_transition = _normalized_id(transition.get("transition"))
        if observed_transition in schema.transition_labels:
            transition_label = observed_transition

    return StageEditTarget(
        cut=cut,
        selected_source_id=selected,
        rejected_source_ids=rejected,
        duration_bucket=duration_bucket,
        transition_label=transition_label,
        crop=_crop_target(labels),
    )


def _canonical_boundary_target(
    boundary: _CanonicalDetectedBoundary,
    *,
    candidates: Sequence[SourceCandidateToken],
    duration_seconds: float | None,
    schema: StageEditTokenSchema,
    crop: tuple[float | None, ...] | None = None,
) -> StageEditTarget:
    if boundary.event_type == DETECTED_BOUNDARY_LOSS_MASKED_EVENT:
        return StageEditTarget()
    if boundary.event_type == NATIVE_SOURCE_CAMERA_CUT_EVENT:
        return StageEditTarget(cut=0.0)
    if boundary.event_type != EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT:
        raise ValueError("canonical boundary event type is unsupported")
    selected: str | None = None
    rejected: tuple[str, ...] = ()
    new_source = boundary.new_source_id
    by_id = {candidate.source_id: candidate for candidate in candidates}
    selected_candidate = by_id.get(new_source or "")
    if (
        new_source is not None
        and selected_candidate is not None
        and selected_candidate.available
        and selected_candidate.hard_gate_passed
    ):
        rejected = tuple(
            candidate.source_id
            for candidate in candidates
            if candidate.source_id != new_source
            and candidate.available
            and candidate.hard_gate_passed
        )
        if rejected:
            selected = new_source
    if duration_seconds is None or duration_seconds <= 0.0:
        raise ValueError("canonical positive duration target is unavailable")
    duration_bucket = schema.duration_bucket(duration_seconds)
    if "hard_cut" not in schema.transition_labels:
        raise ValueError("StageEdit schema cannot represent canonical hard cuts")
    transition_label = "hard_cut"
    return StageEditTarget(
        cut=1.0,
        selected_source_id=selected,
        rejected_source_ids=rejected if selected is not None else (),
        duration_bucket=duration_bucket,
        transition_label=transition_label,
        crop=crop,
    )


def _reference_start_target(
    shot: Mapping[str, Any],
    *,
    candidates: Sequence[SourceCandidateToken],
    schema: StageEditTokenSchema,
) -> StageEditTarget:
    """Supervise the initial shot choice without inventing a cut at time zero."""

    source_id = _normalized_id(shot.get("selected_source_id"))
    by_id = {candidate.source_id: candidate for candidate in candidates}
    selected_candidate = by_id.get(source_id or "")
    selected: str | None = None
    rejected: tuple[str, ...] = ()
    if (
        source_id is not None
        and selected_candidate is not None
        and selected_candidate.available
        and selected_candidate.hard_gate_passed
    ):
        rejected = tuple(
            candidate.source_id
            for candidate in candidates
            if candidate.source_id != source_id
            and candidate.available
            and candidate.hard_gate_passed
        )
        if rejected:
            selected = source_id
    labels = shot.get("edit_labels")
    cut = labels.get("cut") if isinstance(labels, Mapping) else None
    duration = _finite(cut.get("duration_seconds")) if isinstance(cut, Mapping) else None
    duration_bucket = (
        schema.duration_bucket(duration)
        if duration is not None and duration > 0.0
        else None
    )
    return StageEditTarget(
        # The beginning of a program is a source-selection decision, not a
        # visible editorial cut.  Timing/transition losses remain masked.
        cut=None,
        selected_source_id=selected,
        rejected_source_ids=rejected if selected is not None else (),
        duration_bucket=duration_bucket,
        transition_label=None,
        crop=(
            _crop_target(labels) if isinstance(labels, Mapping) else None
        ),
    )


def _negative_times(
    start: float,
    end: float,
    *,
    sample_count: int,
    boundary_guard_seconds: float,
) -> tuple[float, ...]:
    if sample_count <= 0:
        return ()
    interior_start = start + boundary_guard_seconds
    interior_end = end - boundary_guard_seconds
    if interior_end <= interior_start:
        return ()
    span = interior_end - interior_start
    return tuple(
        interior_start + span * (index + 1) / (sample_count + 1)
        for index in range(sample_count)
    )


def _context_token(
    event: _TrainingEvent,
    *,
    performance_start_seconds: float,
    performance_end_seconds: float,
    causal_history: Sequence[CausalEditDecision],
    schema: StageEditTokenSchema,
    catalog: Mapping[str, Mapping[str, Any]],
    feature_snapshot: StageEditFeatureSnapshot | None,
) -> StageEditToken:
    source_ids = tuple(item.source_id for item in event.candidates)
    if feature_snapshot is not None:
        return subset_stage_edit_token_sources(
            feature_snapshot.token_at(event.time_seconds),
            source_ids,
        )
    return materialize_stage_edit_token(
        master_time_seconds=event.time_seconds,
        source_observations=tuple(
            _source_observation(source_id, catalog) for source_id in source_ids
        ),
        schema=schema,
        causal_history=causal_history,
        performance_start_seconds=performance_start_seconds,
        performance_end_seconds=performance_end_seconds,
    )


def build_reference_training_example(
    payload: Mapping[str, Any],
    *,
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
    feature_snapshot: StageEditFeatureSnapshot | None = None,
    song_id: str | None = None,
    negative_samples_per_hold: int = 1,
    boundary_guard_seconds: float = 0.18,
    max_sources: int = 8,
) -> ReferenceTrainingExample:
    """Convert one reference-supervision payload without feature leakage."""

    if payload.get("dataset_kind") != EXPECTED_DATASET_KIND:
        raise ValueError("unsupported reference supervision dataset")
    if negative_samples_per_hold < 0:
        raise ValueError("negative_samples_per_hold cannot be negative")
    if not math.isfinite(boundary_guard_seconds) or boundary_guard_seconds < 0.0:
        raise ValueError("boundary_guard_seconds must be finite and non-negative")
    if max_sources < 2:
        raise ValueError("max_sources must allow a positive and a negative")
    if feature_snapshot is not None:
        if not isinstance(feature_snapshot, StageEditFeatureSnapshot):
            raise TypeError("feature_snapshot must use StageEditFeatureSnapshot")
        feature_snapshot.require_schema(schema)

    reference_id = _normalized_id(payload.get("reference_id"))
    style_id = _normalized_id(payload.get("style_id"))
    if reference_id is None or style_id is None:
        raise ValueError("reference_id and style_id are required")
    normalized_song = _normalized_id(song_id or payload.get("song_id"))
    dense_payload_ready = _dense_payload_provenance_ready(payload)
    canonical_boundaries = _canonical_detected_boundaries(
        payload,
        dense_payload_provenance_ready=dense_payload_ready,
    )
    raw_shots = payload.get("shot_labels")
    if not isinstance(raw_shots, list) or not raw_shots:
        raise ValueError("shot_labels must be a non-empty array")
    rankings = _ranking_by_shot(payload)
    catalog = _catalog_by_source(payload)

    events: list[_TrainingEvent] = []
    performance_start = math.inf
    performance_end = 0.0
    shot_spans: list[tuple[int, Mapping[str, Any], float, float]] = []
    shots_by_index: dict[int, Mapping[str, Any]] = {}
    for shot in raw_shots:
        if not isinstance(shot, Mapping):
            continue
        shot_index = shot.get("shot_index")
        if isinstance(shot_index, bool) or not isinstance(shot_index, int):
            continue
        labels = shot.get("edit_labels")
        cut_labels = labels.get("cut") if isinstance(labels, Mapping) else None
        if not isinstance(cut_labels, Mapping):
            continue
        start = _finite(cut_labels.get("cut_master_time"))
        end = _finite(cut_labels.get("end_master_time"))
        if start is None or end is None or start < 0.0 or end <= start:
            continue
        performance_start = min(performance_start, start)
        performance_end = max(performance_end, end)
        shot_spans.append((shot_index, shot, start, end))
        shots_by_index[shot_index] = shot

    if not shot_spans:
        raise ValueError("reference supervision has no valid timed shots")

    if canonical_boundaries is not None:
        # The hard-cut detector owns this partition.  Do not add the known-shot
        # boundary labels as a second, source-run-collapsed event stream.
        initial_shot = min(shot_spans, key=lambda item: (item[2], item[0]))[1]
        initial_index = int(initial_shot["shot_index"])
        initial_labels = initial_shot.get("edit_labels")
        if not isinstance(initial_labels, Mapping):
            raise ValueError("canonical initial shot lacks edit labels")
        initial_event = _boundary_event_label(
            initial_labels,
            dense_payload_provenance_ready=dense_payload_ready,
        )
        if initial_event.event_type != REFERENCE_START_EVENT:
            raise ValueError("canonical partition lacks an explicit reference start")
        initial_candidates = _event_candidates(
            timestamp=performance_start,
            shot=initial_shot,
            ranking=rankings.get(initial_index),
            catalog=catalog,
            feature_snapshot=feature_snapshot,
            max_sources=max_sources,
            schema=schema,
        )
        initial_target = _reference_start_target(
            initial_shot,
            candidates=initial_candidates,
            schema=schema,
        )
        events.append(
            _TrainingEvent(
                time_seconds=performance_start,
                shot_index=initial_index,
                event_kind=REFERENCE_START_EVENT,
                candidates=initial_candidates,
                target=initial_target,
                explicit_event_schema=initial_event.explicit_schema,
                dense_attribution_ready=initial_event.dense_attribution_ready,
                causal_source_id=(
                    _normalized_id(initial_shot.get("selected_source_id"))
                ),
            )
        )
        canonical_prefix_boundary_excluded_count = sum(
            boundary.incoming_shot_index <= initial_index
            for boundary in canonical_boundaries
        )
        for boundary_position, boundary in enumerate(canonical_boundaries):
            if boundary.incoming_shot_index <= initial_index:
                # The retained sequence starts at the first source-attributed
                # shot.  Detector edges wholly inside the leading UNKNOWN
                # prefix, including the final UNKNOWN-to-known edge at this
                # same timestamp, have no honest cut/source target and must
                # not create duplicate or pre-sequence tokens.
                continue
            shot = shots_by_index.get(boundary.incoming_shot_index, {})
            ranking = rankings.get(boundary.incoming_shot_index)
            candidates = _event_candidates(
                timestamp=boundary.time_seconds,
                shot=shot,
                ranking=ranking,
                catalog=catalog,
                feature_snapshot=feature_snapshot,
                max_sources=max_sources,
                schema=schema,
            )
            duration: float | None = None
            crop: tuple[float | None, ...] | None = None
            if boundary.event_type == EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT:
                labels = shot.get("edit_labels") if isinstance(shot, Mapping) else None
                cut_labels = (
                    labels.get("cut") if isinstance(labels, Mapping) else None
                )
                crop = _crop_target(labels) if isinstance(labels, Mapping) else None
                duration = (
                    _finite(cut_labels.get("duration_seconds"))
                    if isinstance(cut_labels, Mapping)
                    else None
                )
                if duration is None or duration <= 0.0:
                    next_time = (
                        canonical_boundaries[boundary_position + 1].time_seconds
                        if boundary_position + 1 < len(canonical_boundaries)
                        else performance_end
                    )
                    duration = next_time - boundary.time_seconds
            target = _canonical_boundary_target(
                boundary,
                candidates=candidates,
                duration_seconds=duration,
                schema=schema,
                crop=crop,
            )
            events.append(
                _TrainingEvent(
                    time_seconds=boundary.time_seconds,
                    shot_index=boundary.incoming_shot_index,
                    event_kind=boundary.event_type,
                    candidates=candidates,
                    target=target,
                    explicit_event_schema=True,
                    dense_attribution_ready=(
                        boundary.dense_attribution_ready
                    ),
                    causal_source_id=boundary.new_source_id,
                )
            )

        # Interior no-cut samples remain useful, but are generated only from
        # known positive-duration holds and never replace a detected boundary.
        for shot_index, shot, start, end in shot_spans:
            ranking = rankings.get(shot_index)
            for negative_time in _negative_times(
                start,
                end,
                sample_count=negative_samples_per_hold,
                boundary_guard_seconds=boundary_guard_seconds,
            ):
                candidates = _event_candidates(
                    timestamp=negative_time,
                    shot=shot,
                    ranking=ranking,
                    catalog=catalog,
                    feature_snapshot=feature_snapshot,
                    max_sources=max_sources,
                    schema=schema,
                )
                events.append(
                    _TrainingEvent(
                        time_seconds=negative_time,
                        shot_index=shot_index,
                        event_kind=HOLD_NEGATIVE_EVENT,
                        candidates=candidates,
                        target=StageEditTarget(cut=0.0),
                    )
                )
    else:
        canonical_prefix_boundary_excluded_count = 0
        # Legacy reference payloads do not contain the detector-owned complete
        # boundary partition.  Preserve their previous known-shot behavior.
        for shot_index, shot, start, end in shot_spans:
            labels = shot.get("edit_labels")
            if not isinstance(labels, Mapping):
                continue
            event_label = _boundary_event_label(
                labels,
                dense_payload_provenance_ready=dense_payload_ready,
            )
            ranking = rankings.get(shot_index)
            candidates = _event_candidates(
                timestamp=start,
                shot=shot,
                ranking=ranking,
                catalog=catalog,
                feature_snapshot=feature_snapshot,
                max_sources=max_sources,
                schema=schema,
            )
            candidate_ids = tuple(item.source_id for item in candidates)
            boundary_target = _boundary_target(
                shot, ranking, schema, event_label
            )
            candidates_by_id = {
                candidate.source_id: candidate for candidate in candidates
            }
            supervised_candidate = candidates_by_id.get(
                boundary_target.selected_source_id or ""
            )
            # Keep target identities aligned with any max-source truncation.
            if (
                boundary_target.selected_source_id not in candidate_ids
                or (
                    boundary_target.selected_source_id is not None
                    and (
                        supervised_candidate is None
                        or not supervised_candidate.available
                        or not supervised_candidate.hard_gate_passed
                    )
                )
                or any(
                    source_id not in candidate_ids
                    for source_id in boundary_target.rejected_source_ids
                )
            ):
                boundary_target = StageEditTarget(
                    cut=boundary_target.cut,
                    duration_bucket=boundary_target.duration_bucket,
                    transition_label=boundary_target.transition_label,
                    crop=boundary_target.crop,
                )
            events.append(
                _TrainingEvent(
                    time_seconds=start,
                    shot_index=shot_index,
                    event_kind=event_label.event_type,
                    candidates=candidates,
                    target=boundary_target,
                    explicit_event_schema=event_label.explicit_schema,
                    dense_attribution_ready=(
                        event_label.dense_attribution_ready
                    ),
                    causal_source_id=boundary_target.selected_source_id,
                )
            )
            for negative_time in _negative_times(
                start,
                end,
                sample_count=negative_samples_per_hold,
                boundary_guard_seconds=boundary_guard_seconds,
            ):
                negative_candidates = _event_candidates(
                    timestamp=negative_time,
                    shot=shot,
                    ranking=ranking,
                    catalog=catalog,
                    feature_snapshot=feature_snapshot,
                    max_sources=max_sources,
                    schema=schema,
                )
                if tuple(item.source_id for item in negative_candidates) != candidate_ids:
                    raise ValueError(
                        "StageEdit feature snapshot source set changes inside a hold"
                    )
                events.append(
                    _TrainingEvent(
                        time_seconds=negative_time,
                        shot_index=shot_index,
                        event_kind=HOLD_NEGATIVE_EVENT,
                        candidates=negative_candidates,
                        target=StageEditTarget(cut=0.0),
                    )
                )

    if not events:
        raise ValueError("reference supervision has no valid timed shots")
    events.sort(
        key=lambda event: (
            event.time_seconds,
            1 if event.event_kind == HOLD_NEGATIVE_EVENT else 0,
            event.shot_index,
        )
    )
    causal_history: list[CausalEditDecision] = []
    tokens: list[StageEditToken] = []
    targets: list[StageEditTarget] = []
    for event in events:
        tokens.append(
            _context_token(
                event,
                performance_start_seconds=performance_start,
                performance_end_seconds=performance_end,
                causal_history=causal_history,
                schema=schema,
                catalog=catalog,
                feature_snapshot=feature_snapshot,
            )
        )
        targets.append(event.target)
        if event.target.cut == 1.0 or event.event_kind == REFERENCE_START_EVENT:
            causal_history.append(
                CausalEditDecision(
                    cut_time_seconds=event.time_seconds,
                    source_id=(
                        event.causal_source_id
                        or event.target.selected_source_id
                    ),
                    transition_strength=1.0,
                )
            )

    sequence_id = f"stageedit-reference:{normalized_song or 'song-unknown'}:{reference_id}"
    statistics = {
        "token_count": len(tokens),
        "boundary_token_count": sum(
            event.event_kind != HOLD_NEGATIVE_EVENT for event in events
        ),
        "editorial_source_switch_positive_count": sum(
            event.event_kind == EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT
            and event.target.cut == 1.0
            for event in events
        ),
        "dense_editorial_source_switch_positive_count": sum(
            event.event_kind == EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT
            and event.target.cut == 1.0
            and event.explicit_event_schema
            and event.dense_attribution_ready
            for event in events
        ),
        "native_source_camera_cut_count": sum(
            event.event_kind == NATIVE_SOURCE_CAMERA_CUT_EVENT
            for event in events
        ),
        "native_source_camera_cut_editorial_positive_count": sum(
            event.event_kind == NATIVE_SOURCE_CAMERA_CUT_EVENT
            and event.target.cut == 1.0
            for event in events
        ),
        "explicit_boundary_event_count": sum(
            event.event_kind != HOLD_NEGATIVE_EVENT
            and event.explicit_event_schema
            for event in events
        ),
        "legacy_boundary_event_count": sum(
            event.event_kind != HOLD_NEGATIVE_EVENT
            and not event.explicit_event_schema
            for event in events
        ),
        "cut_positive_count": sum(target.cut == 1.0 for target in targets),
        "cut_negative_count": sum(target.cut == 0.0 for target in targets),
        "cut_masked_count": sum(target.cut is None for target in targets),
        "source_supervision_count": sum(
            target.selected_source_id is not None for target in targets
        ),
        "duration_supervision_count": sum(
            target.duration_bucket is not None for target in targets
        ),
        "transition_supervision_count": sum(
            target.transition_label is not None for target in targets
        ),
        "crop_coordinate_supervision_count": sum(
            sum(value is not None for value in target.crop)
            for target in targets
            if target.crop is not None
        ),
        "reference_match_features_in_model_input": 0,
        "shared_feature_materializer_used": True,
        "materialized_feature_snapshot_used": feature_snapshot is not None,
        "canonical_detected_boundary_partition_used": (
            canonical_boundaries is not None
        ),
        "canonical_detected_boundary_count": (
            len(canonical_boundaries) if canonical_boundaries is not None else 0
        ),
        "canonical_loss_masked_boundary_count": sum(
            event.event_kind == DETECTED_BOUNDARY_LOSS_MASKED_EVENT
            for event in events
        ),
        "canonical_prefix_boundary_excluded_count": (
            canonical_prefix_boundary_excluded_count
        ),
    }
    return ReferenceTrainingExample(
        reference_id=reference_id,
        song_id=normalized_song,
        style_id=style_id,
        sequence=StageEditSequence(sequence_id, tuple(tokens)),
        targets=StageEditTargetSequence(sequence_id, tuple(targets)),
        statistics=statistics,
        feature_snapshot_sha256=(
            feature_snapshot.snapshot_sha256
            if feature_snapshot is not None
            else None
        ),
    )


def load_reference_training_corpus(
    paths: Sequence[Path],
    *,
    schema: StageEditTokenSchema = DEFAULT_STAGE_EDIT_SCHEMA,
    feature_snapshot_by_reference: Mapping[
        str, StageEditFeatureSnapshot
    ] | None = None,
    song_id_by_reference: Mapping[str, str] | None = None,
    negative_samples_per_hold: int = 1,
    boundary_guard_seconds: float = 0.18,
    max_sources: int = 8,
) -> ReferenceTrainingCorpus:
    """Load caller-supplied JSON files; no output is written implicitly."""

    if not paths:
        raise ValueError("at least one reference-supervision path is required")
    song_map = dict(song_id_by_reference or {})
    snapshot_map = dict(feature_snapshot_by_reference or {})
    inputs: list[tuple[str, Path, bytes, Mapping[str, Any]]] = []
    for raw_path in paths:
        path = Path(raw_path)
        content = path.read_bytes()
        payload = json.loads(content)
        if not isinstance(payload, Mapping):
            raise ValueError(f"reference dataset root must be an object: {path}")
        reference_id = _normalized_id(payload.get("reference_id"))
        if reference_id is None:
            raise ValueError(f"reference_id is missing: {path}")
        inputs.append((reference_id, path, content, payload))
    if len({value[0] for value in inputs}) != len(inputs):
        raise ValueError("reference ids must be unique")
    unknown_snapshot_ids = set(snapshot_map) - {value[0] for value in inputs}
    if unknown_snapshot_ids:
        raise ValueError(
            "feature snapshots reference unknown datasets: "
            + ", ".join(sorted(unknown_snapshot_ids))
        )

    examples: list[ReferenceTrainingExample] = []
    source_paths: list[str] = []
    digest = hashlib.sha256()
    digest.update(f"stage-edit-reference-adapter-v{ADAPTER_SCHEMA_VERSION}".encode())
    digest.update(schema.fingerprint.encode())
    digest.update(str(negative_samples_per_hold).encode())
    digest.update(repr(boundary_guard_seconds).encode())
    digest.update(str(max_sources).encode())
    for reference_id, path, content, payload in sorted(inputs):
        example = build_reference_training_example(
            payload,
            schema=schema,
            feature_snapshot=snapshot_map.get(reference_id),
            song_id=song_map.get(reference_id),
            negative_samples_per_hold=negative_samples_per_hold,
            boundary_guard_seconds=boundary_guard_seconds,
            max_sources=max_sources,
        )
        examples.append(example)
        resolved = str(path.resolve())
        source_paths.append(resolved)
        digest.update(reference_id.encode())
        digest.update(hashlib.sha256(content).digest())
        digest.update(str(example.song_id).encode())
        digest.update(str(example.feature_snapshot_sha256).encode())
    examples.sort(key=lambda example: example.reference_id)
    total: dict[str, int] = {}
    for key in (
        "token_count",
        "boundary_token_count",
        "editorial_source_switch_positive_count",
        "dense_editorial_source_switch_positive_count",
        "native_source_camera_cut_count",
        "native_source_camera_cut_editorial_positive_count",
        "explicit_boundary_event_count",
        "legacy_boundary_event_count",
        "cut_positive_count",
        "cut_negative_count",
        "cut_masked_count",
        "source_supervision_count",
        "duration_supervision_count",
        "transition_supervision_count",
        "crop_coordinate_supervision_count",
        "reference_match_features_in_model_input",
        "shared_feature_materializer_used",
        "materialized_feature_snapshot_used",
        "canonical_detected_boundary_partition_used",
        "canonical_detected_boundary_count",
        "canonical_loss_masked_boundary_count",
    ):
        total[key] = sum(int(example.statistics[key]) for example in examples)
    total["reference_count"] = len(examples)
    total["song_count"] = len(
        {example.song_id for example in examples if example.song_id is not None}
    )
    dense_examples = tuple(
        example
        for example in examples
        if int(
            example.statistics[
                "dense_editorial_source_switch_positive_count"
            ]
        )
        > 0
    )
    total["reference_count_with_dense_editorial_switches"] = len(
        dense_examples
    )
    total["song_count_with_dense_editorial_switches"] = len(
        {
            example.song_id
            for example in dense_examples
            if example.song_id is not None
        }
    )
    return ReferenceTrainingCorpus(
        examples=tuple(examples),
        data_id=f"stageedit-ref-{digest.hexdigest()[:20]}",
        source_paths=tuple(source_paths),
        statistics=total,
    )


__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "EXPECTED_DATASET_KIND",
    "PROHIBITED_REFERENCE_MATCH_INPUT_FIELDS",
    "ReferenceTrainingCorpus",
    "ReferenceTrainingExample",
    "build_reference_training_example",
    "load_reference_training_corpus",
]
