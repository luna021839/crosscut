"""Build explicit pairwise/listwise supervision from recovered stage mixes.

``reference_attribution`` answers *which synchronized original most likely
appears in each shot*.  ``reference_dataset`` turns that conservative path
into honest edit labels.  This module is the final, deliberately offline,
adapter: it emits normalized ranking groups that a learned reranker can
consume without having to reinterpret attribution JSON.

No candidate or transition label is invented here.  UNKNOWN shots remain out
of source-selection supervision, unresolved source changes stay masked, and a
crop target is eligible only when the reverse-matching evidence contains an
explicit crop transform.  A homography is retained as geometry evidence, but
is not silently relabelled as a crop parameterization.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .reference_dataset import (
    CropSupervisionInput,
    EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT,
    NATIVE_SOURCE_CAMERA_CUT_EVENT,
    SCHEMA_VERSION as REFERENCE_DATASET_SCHEMA_VERSION,
    build_reference_dataset,
)


SCHEMA_VERSION = 3

_CANDIDATE_EVIDENCE_FIELDS = (
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


def _source_id(candidate: Mapping[str, Any]) -> str | None:
    raw = candidate.get("video_id", candidate.get("source_id"))
    if raw is None:
        return None
    normalized = str(raw).strip()
    return normalized or None


def _score(candidate: Mapping[str, Any]) -> float | None:
    raw = candidate.get("score")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _candidate_evidence(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: deepcopy(candidate[name])
        for name in _CANDIDATE_EVIDENCE_FIELDS
        if name in candidate and candidate[name] is not None
    }


def _raw_candidates(shot: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return deterministic, source-unique evidence in observed rank order."""

    ranked = shot.get("ranked_candidates")
    values: Sequence[Any]
    if isinstance(ranked, list) and ranked:
        values = ranked
    else:
        values = (shot.get("best"), shot.get("runner_up"))

    output: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, Mapping):
            continue
        candidate_id = _source_id(value)
        if candidate_id is None or candidate_id in seen:
            continue
        seen.add(candidate_id)
        output.append(value)

    # A temporal path can select a source that is not locally rank one.  Older
    # evidence files may omit that source from ``ranked_candidates`` while
    # still retaining it as ``best``; preserve the actual positive evidence.
    selected = str(shot.get("attributed_video_id") or "").strip()
    if selected and selected not in seen:
        best = shot.get("best")
        if isinstance(best, Mapping) and _source_id(best) == selected:
            output.insert(0, best)
    return tuple(output)


def _candidate_records(
    shot: Mapping[str, Any],
    selected_source: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for observed_rank, raw in enumerate(_raw_candidates(shot)):
        candidate_id = _source_id(raw)
        if candidate_id is None:
            continue
        selected = candidate_id == selected_source
        candidates.append(
            {
                "source_id": candidate_id,
                "selected": selected,
                "relevance": 1 if selected else 0,
                "observed_match_rank": observed_rank,
                "evidence": _candidate_evidence(raw),
            }
        )
    return candidates


def _edit_labels(record: Mapping[str, Any]) -> dict[str, Any]:
    positive = record.get("positive_features")
    if not isinstance(positive, Mapping):
        positive = {}
    explicit_crop = positive.get("crop_transform")
    homography = positive.get("homography_reference_to_source")
    boundary_event = record.get("boundary_event")
    if not isinstance(boundary_event, Mapping):
        raise ValueError("reference record is missing boundary_event")
    crop_label: dict[str, Any] = {
        "supervision_eligible": explicit_crop is not None,
        "transform": deepcopy(explicit_crop),
        "label_basis": (
            "explicit_reverse_match_crop_transform"
            if explicit_crop is not None
            else "unavailable"
        ),
    }
    crop_supervision = record.get("crop_supervision")
    if crop_supervision is not None:
        if not isinstance(crop_supervision, Mapping):
            raise ValueError("reference record crop_supervision must be an object")
        eligible = crop_supervision.get("supervision_eligible") is True
        transform = crop_supervision.get("transform")
        loss_mask = crop_supervision.get("loss_mask")
        if not isinstance(loss_mask, Mapping):
            raise ValueError("reference record crop supervision lacks a loss mask")
        if eligible != (isinstance(transform, Mapping) and explicit_crop is not None):
            raise ValueError("reference record crop target and eligibility disagree")
        if eligible and transform != explicit_crop:
            raise ValueError("reference record crop target copies disagree")
        if eligible and loss_mask.get("crop_transform_loss_masked") is not False:
            raise ValueError("eligible reference crop target cannot be loss-masked")
        if not eligible and (
            explicit_crop is not None
            or transform is not None
            or loss_mask.get("crop_transform_loss_masked") is not True
        ):
            raise ValueError("rejected or missing reference crop must remain loss-masked")
        crop_label = {
            "supervision_eligible": eligible,
            "transform": deepcopy(transform) if eligible else None,
            "label_basis": (
                "validated_multiframe_source_to_reference_renderer_transform"
                if eligible
                else str(crop_supervision.get("status") or "loss_masked")
            ),
            "rejection_reason": crop_supervision.get("rejection_reason"),
            "loss_mask": deepcopy(dict(loss_mask)),
            "provenance": deepcopy(crop_supervision.get("provenance")),
        }
    return {
        "cut": {
            "boundary_label": record.get("boundary_label"),
            "nominal_cut_master_time": record.get("nominal_cut_master_time"),
            "cut_master_time": record.get("cut_master_time"),
            "end_master_time": record.get("end_master_time"),
            "duration_seconds": record.get("duration_seconds"),
        },
        "source_transition": {
            "previous_source_id": record.get("previous_known_source"),
            "selected_source_id": record.get("selected_source"),
            "is_source_switch": bool(record.get("is_source_switch")),
            "transition": record.get("transition"),
            "supervision_eligible": bool(
                record.get("transition_supervision_eligible")
            ),
        },
        "boundary_event": deepcopy(boundary_event),
        "dense_attribution": deepcopy(record.get("dense_attribution")),
        "crop": crop_label,
        "geometry_evidence": {
            "available": homography is not None,
            "homography_reference_to_source": deepcopy(homography),
            "is_direct_crop_target": False,
        },
    }


@dataclass(frozen=True, slots=True)
class ReferenceSupervisionDataset:
    """Serializable ranking examples and masked edit labels."""

    payload: dict[str, Any]

    @property
    def listwise_count(self) -> int:
        return len(self.payload["listwise_records"])

    @property
    def pairwise_count(self) -> int:
        return len(self.payload["pairwise_records"])

    def write(self, path: Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination.resolve()


def build_reference_supervision(
    attribution: Mapping[str, Any],
    *,
    style_id: str,
    reference_id: str | None = None,
    crop_supervision_artifacts: Sequence[CropSupervisionInput] | None = None,
) -> ReferenceSupervisionDataset:
    """Create pairwise/listwise source ranking records plus edit labels.

    The existing reference dataset builder remains the single authority for
    UNKNOWN handling and boundary/transition eligibility.  This adapter only
    normalizes its known-shot records against the original ranked evidence.
    """

    base = build_reference_dataset(
        attribution,
        style_id=style_id,
        reference_id=reference_id,
        crop_supervision_artifacts=crop_supervision_artifacts,
    ).payload
    raw_shots = attribution.get("shots")
    if not isinstance(raw_shots, list):
        raise ValueError("attribution must contain a shots array")
    shots_by_index = {
        index: value
        for index, value in enumerate(raw_shots)
        if isinstance(value, Mapping)
    }

    shot_labels: list[dict[str, Any]] = []
    listwise_records: list[dict[str, Any]] = []
    pairwise_records: list[dict[str, Any]] = []
    missing_positive_indices: list[int] = []
    insufficient_candidate_indices: list[int] = []

    for base_record in base["records"]:
        shot_index = int(base_record["shot_index"])
        raw_shot = shots_by_index[shot_index]
        selected_source = str(base_record["selected_source"])
        labels = _edit_labels(base_record)
        group_id = str(base_record["record_id"])
        candidates = _candidate_records(raw_shot, selected_source)
        if base.get("crop_supervision_schema") is not None:
            validated_crop = labels["crop"]["transform"]
            crop_eligible = labels["crop"]["supervision_eligible"] is True
            for candidate in candidates:
                evidence = candidate["evidence"]
                evidence.pop("crop_transform", None)
                if candidate["selected"] and crop_eligible:
                    evidence["crop_transform"] = deepcopy(validated_crop)
        selected_candidates = [
            candidate for candidate in candidates if candidate["selected"]
        ]

        shot_labels.append(
            {
                "record_id": group_id,
                "style_id": base["style_id"],
                "reference_id": base["reference_id"],
                "shot_index": shot_index,
                "selected_source_id": selected_source,
                "selected_source_time": base_record.get("selected_source_time"),
                "edit_labels": labels,
                "attribution_confidence": deepcopy(
                    base_record.get("attribution_confidence")
                ),
            }
        )

        if len(selected_candidates) != 1:
            missing_positive_indices.append(shot_index)
            continue
        negatives = [
            candidate for candidate in candidates if not candidate["selected"]
        ]
        if not negatives:
            insufficient_candidate_indices.append(shot_index)
            continue

        chosen = selected_candidates[0]
        listwise_records.append(
            {
                "group_id": group_id,
                "style_id": base["style_id"],
                "reference_id": base["reference_id"],
                "shot_index": shot_index,
                "chosen_source_id": selected_source,
                "candidate_count": len(candidates),
                "candidates": candidates,
                "edit_labels": labels,
                "attribution_confidence": deepcopy(
                    base_record.get("attribution_confidence")
                ),
            }
        )

        chosen_score = _score(chosen["evidence"])
        for negative in negatives:
            rejected_score = _score(negative["evidence"])
            pairwise_records.append(
                {
                    "pair_id": f"{group_id}:{negative['source_id']}",
                    "group_id": group_id,
                    "style_id": base["style_id"],
                    "reference_id": base["reference_id"],
                    "shot_index": shot_index,
                    "chosen": deepcopy(chosen),
                    "rejected": deepcopy(negative),
                    "observed_score_margin": (
                        chosen_score - rejected_score
                        if chosen_score is not None and rejected_score is not None
                        else None
                    ),
                    "edit_labels": labels,
                }
            )

    crop_labels = sum(
        bool(record["edit_labels"]["crop"]["supervision_eligible"])
        for record in shot_labels
    )
    geometry_labels = sum(
        bool(record["edit_labels"]["geometry_evidence"]["available"])
        for record in shot_labels
    )
    transition_labels = sum(
        bool(
            record["edit_labels"]["source_transition"][
                "supervision_eligible"
            ]
        )
        for record in shot_labels
    )
    editorial_switch_labels = sum(
        record["edit_labels"]["boundary_event"]["event_type"]
        == EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT
        for record in shot_labels
    )
    native_source_cuts = sum(
        record["edit_labels"]["boundary_event"]["event_type"]
        == NATIVE_SOURCE_CAMERA_CUT_EVENT
        for record in shot_labels
    )
    dense_editorial_switch_labels = sum(
        record["edit_labels"]["boundary_event"]["event_type"]
        == EDITORIAL_SOURCE_SWITCH_HARD_CUT_EVENT
        and bool(
            record["edit_labels"]["boundary_event"][
                "dense_attribution_ready"
            ]
        )
        for record in shot_labels
    )

    output = {
            "schema_version": SCHEMA_VERSION,
            "dataset_kind": (
                "reference_stage_mix_pairwise_listwise_supervision"
            ),
            "base_reference_dataset_schema_version": (
                REFERENCE_DATASET_SCHEMA_VERSION
            ),
            "style_id": base["style_id"],
            "reference_id": base["reference_id"],
            "reference_path": base.get("reference_path"),
            "attribution_analysis": base.get("attribution_analysis"),
            "dense_attribution_schema": deepcopy(
                base.get("dense_attribution_schema")
            ),
            "canonical_shot_schema": deepcopy(
                base.get("canonical_shot_schema")
            ),
            "reference_evidence_provenance": deepcopy(
                base.get("reference_evidence_provenance")
            ),
            "source_catalog": deepcopy(base.get("source_catalog", [])),
            "label_policy": {
                "unknown_shots_excluded_from_source_supervision": True,
                "positive_source_must_have_visual_match_evidence": True,
                "listwise_group_requires_at_least_one_negative": True,
                "every_pair_compares_selected_to_observed_candidate": True,
                "unresolved_transitions_remain_masked": True,
                "native_source_camera_cuts_never_label_editorial_switches": True,
                "editorial_cut_requires_confirmed_cross_source_event": True,
                "crop_requires_explicit_crop_transform": True,
                "homography_is_geometry_evidence_not_direct_crop_target": True,
                "every_detected_reference_boundary_is_retained": bool(
                    base.get("canonical_shot_schema")
                ),
                "unknown_or_gate_failed_detected_boundaries_are_loss_masked": True,
            },
            "statistics": {
                "known_shot_label_count": len(shot_labels),
                "listwise_record_count": len(listwise_records),
                "pairwise_record_count": len(pairwise_records),
                "transition_supervision_count": transition_labels,
                "editorial_source_switch_supervision_count": (
                    editorial_switch_labels
                ),
                "dense_editorial_source_switch_supervision_count": (
                    dense_editorial_switch_labels
                ),
                "native_source_camera_cut_count": native_source_cuts,
                "crop_supervision_count": crop_labels,
                "geometry_evidence_count": geometry_labels,
                "missing_positive_shot_indices": missing_positive_indices,
                "insufficient_candidate_shot_indices": (
                    insufficient_candidate_indices
                ),
            },
            "shot_labels": shot_labels,
            "listwise_records": listwise_records,
            "pairwise_records": pairwise_records,
            "detected_boundary_records": deepcopy(
                base.get("detected_boundary_records", [])
            ),
            "detected_boundary_statistics": deepcopy(
                base.get("detected_boundary_statistics")
            ),
        }
    crop_supervision_schema = base.get("crop_supervision_schema")
    if crop_supervision_schema is not None:
        if not isinstance(crop_supervision_schema, Mapping):
            raise ValueError("base crop_supervision_schema must be an object")
        output["crop_supervision_schema"] = deepcopy(
            dict(crop_supervision_schema)
        )
        output["label_policy"].update(
            {
                "crop_target_requires_validated_multiframe_artifact": True,
                "crop_rejected_or_missing_is_loss_masked": True,
                "legacy_crop_hint_disabled_when_crop_artifacts_are_enabled": True,
            }
        )
        output["statistics"].update(
            {
                "crop_loss_masked_count": sum(
                    not bool(
                        record["edit_labels"]["crop"][
                            "supervision_eligible"
                        ]
                    )
                    for record in shot_labels
                ),
                "crop_missing_artifact_count": len(
                    crop_supervision_schema.get("missing_shot_indices", [])
                ),
                "crop_rejected_artifact_count": len(
                    crop_supervision_schema.get("rejected_shot_indices", [])
                ),
            }
        )
    return ReferenceSupervisionDataset(output)


def build_reference_supervision_file(
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
    return build_reference_supervision(
        payload,
        style_id=style_id,
        reference_id=reference_id,
        crop_supervision_artifacts=crop_supervision_paths,
    ).write(Path(output_path))


__all__ = [
    "ReferenceSupervisionDataset",
    "SCHEMA_VERSION",
    "build_reference_supervision",
    "build_reference_supervision_file",
]
