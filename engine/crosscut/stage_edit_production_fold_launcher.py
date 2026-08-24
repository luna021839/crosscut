"""Manifest-driven launcher for the two real IVE StageEdit production folds.

The production trainer deliberately accepts low-level paths so its core stays
reusable.  Supplying fifteen supervision/snapshot pairs and hundreds of
dynamic-crop artifacts by hand, however, is both error-prone and unauditable.
This module binds the three immutable per-song manifest families into one
content-addressed plan, validates their cross-manifest identities, and only
then expands the exact arguments consumed by ``stage_edit_production_train_cli``.

Preflight hashes JSON evidence, not the large provenance media.  The training
corpus builder performs that media verification once with its process-wide
per-path cache.  This preserves end-to-end integrity without reading the same
multi-gigabyte source once during manifest validation and again during corpus
assembly.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from .reference_dynamic_crop_job import (
    validate_reference_dynamic_crop_manifest,
)
from .reference_dynamic_crop_supervision import (
    load_reference_dynamic_crop_supervision,
)
from .reference_training_materialization_job import (
    ANALYSIS_BASIS as MATERIALIZATION_ANALYSIS_BASIS,
    SCHEMA_VERSION as MATERIALIZATION_SCHEMA_VERSION,
)
from .stage_edit_feature_materializer import StageEditFeatureSnapshot
from .stage_edit_feature_snapshot_job import (
    ACTION_FORMATION_EXTRACTION_CONTRACT,
    FEATURE_SNAPSHOT_JOB_ANALYSIS_BASIS,
    FEATURE_SNAPSHOT_JOB_SCHEMA_VERSION,
)
from .stage_edit_production_train_cli import main as production_train_main
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    stage_edit_recent_history_mask_contract,
)


PLAN_SCHEMA_VERSION = 1
PLAN_ANALYSIS_BASIS = "ive_three_song_stage_edit_production_fold_plan_v1"
PRODUCTION_VOLUME_ROOT = Path("/Volumes/CrossCutMedia")
MINIMUM_FREE_BYTES = 8 * 1024**3
EXPECTED_REFERENCE_COUNT_PER_SONG = 5
EXPECTED_SOURCE_COUNT_PER_SONG = 10
PRODUCTION_SONG_IDS = (
    "IVE-BANG-BANG",
    "IVE-BLACKHOLE",
    "IVE-XOXZ",
)
PRODUCTION_SAMPLING_CONTRACT: Mapping[str, int | float] = {
    "negative_samples_per_hold": 2,
    "boundary_guard_seconds": 0.18,
    "max_sources": 32,
    "visual_working_width": 480,
}
PRODUCTION_FOLDS: Mapping[str, str] = {
    "heldout-xoxz": "IVE-XOXZ",
    "heldout-blackhole": "IVE-BLACKHOLE",
}
REQUIRED_FEATURE_PROVENANCE_CONTRACTS = frozenset(
    {
        ACTION_FORMATION_EXTRACTION_CONTRACT,
        "reference_match_scores_and_current_or_future_targets_excluded",
        "synchronized_source_causal_stage_edit_feature_snapshot_v2",
        "visual_descriptors_decoded_once_per_source_in_forward_order",
    }
)
PRODUCTION_RECENT_HISTORY_MASK_CONTRACT: Mapping[str, Any] = (
    stage_edit_recent_history_mask_contract(DEFAULT_STAGE_EDIT_SCHEMA)
)


class ProductionFoldPreflightError(ValueError):
    """Raised before training when immutable production evidence is incomplete."""


@dataclass(frozen=True, slots=True)
class ManifestPointer:
    path: Path
    file_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "file_sha256": self.file_sha256}


@dataclass(frozen=True, slots=True)
class ProductionFoldPreflight:
    plan_path: Path
    plan_sha256: str
    fold_id: str
    held_out_song_id: str
    training_song_ids: tuple[str, ...]
    checkpoint_directory: Path
    free_bytes: int
    recent_history_mask_contract: Mapping[str, Any]
    reference_ids_by_song: Mapping[str, tuple[str, ...]]
    train_cli_args: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_path": str(self.plan_path),
            "plan_sha256": self.plan_sha256,
            "fold_id": self.fold_id,
            "held_out_song_id": self.held_out_song_id,
            "training_song_ids": list(self.training_song_ids),
            "checkpoint_directory": str(self.checkpoint_directory),
            "free_bytes": self.free_bytes,
            "recent_history_mask_contract": dict(
                self.recent_history_mask_contract
            ),
            "reference_ids_by_song": {
                song: list(reference_ids)
                for song, reference_ids in sorted(
                    self.reference_ids_by_song.items()
                )
            },
            "training_authorized": False,
            "production_registry_activation_requested": False,
            "train_cli_args": list(self.train_cli_args),
        }


@dataclass(frozen=True, slots=True)
class _MaterializationBinding:
    reference_ids: tuple[str, ...]
    supervision_paths: Mapping[str, Path]
    supervision_sha256: Mapping[str, str]
    attribution_paths: Mapping[str, Path]
    attribution_sha256: Mapping[str, str]
    known_shot_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class _FeatureBinding:
    snapshot_paths: Mapping[str, Path]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionFoldPreflightError(f"{name} must be an object")
    return value


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProductionFoldPreflightError(f"{name} must be an array")
    return value


def _stable_file_sha256(path: Path, name: str) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise ProductionFoldPreflightError(f"{name} path must be absolute: {path}")
    if not candidate.is_file() or candidate.is_symlink():
        raise ProductionFoldPreflightError(
            f"{name} must be a regular non-symlink file: {candidate}"
        )
    before = candidate.stat()
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = candidate.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise ProductionFoldPreflightError(f"{name} changed while hashing: {path}")
    return digest.hexdigest()


def _load_pointer(value: object, name: str) -> tuple[ManifestPointer, dict[str, Any]]:
    raw = _mapping(value, name)
    if set(raw) != {"path", "file_sha256"}:
        raise ProductionFoldPreflightError(
            f"{name} must contain exactly path and file_sha256"
        )
    path = Path(str(raw.get("path") or "")).expanduser()
    expected = raw.get("file_sha256")
    if not _is_sha256(expected):
        raise ProductionFoldPreflightError(f"{name} file hash is invalid")
    actual = _stable_file_sha256(path, name)
    if actual != expected:
        raise ProductionFoldPreflightError(f"{name} file hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProductionFoldPreflightError(f"{name} JSON root must be an object")
    return ManifestPointer(path.resolve(), str(expected)), dict(payload)


def _verify_file_record(
    value: Mapping[str, Any],
    *,
    path_key: str,
    sha256_key: str,
    name: str,
) -> tuple[Path, str]:
    path = Path(str(value.get(path_key) or "")).expanduser()
    expected = value.get(sha256_key)
    if not _is_sha256(expected):
        raise ProductionFoldPreflightError(f"{name} hash is invalid")
    if _stable_file_sha256(path, name) != expected:
        raise ProductionFoldPreflightError(f"{name} hash mismatch: {path}")
    return path.resolve(), str(expected)


def _validate_materialization_manifest(
    pointer_value: object,
    *,
    song_id: str,
) -> tuple[ManifestPointer, _MaterializationBinding]:
    pointer, payload = _load_pointer(pointer_value, f"{song_id} materialization manifest")
    if payload.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION:
        raise ProductionFoldPreflightError(
            f"{song_id} materialization schema mismatch"
        )
    if payload.get("analysis_basis") != MATERIALIZATION_ANALYSIS_BASIS:
        raise ProductionFoldPreflightError(
            f"{song_id} materialization analysis basis mismatch"
        )
    fingerprint = payload.get("input_fingerprint_sha256")
    contract = _mapping(payload.get("input_contract"), "materialization input contract")
    if not _is_sha256(fingerprint) or _canonical_sha256(contract) != fingerprint:
        raise ProductionFoldPreflightError(
            f"{song_id} materialization input fingerprint mismatch"
        )
    expected_name = f"reference-training-materialization-{str(fingerprint)[:16]}.json"
    if pointer.path.name != expected_name:
        raise ProductionFoldPreflightError(
            f"{song_id} materialization filename is not content-addressed"
        )
    if payload.get("immutable_inputs_modified") is not False:
        raise ProductionFoldPreflightError(
            f"{song_id} materialization does not preserve immutable inputs"
        )
    if (
        contract.get("expected_reference_count")
        != EXPECTED_REFERENCE_COUNT_PER_SONG
        or contract.get("expected_source_count")
        != EXPECTED_SOURCE_COUNT_PER_SONG
    ):
        raise ProductionFoldPreflightError(
            f"{song_id} materialization count contract mismatch"
        )

    attribution_paths: dict[str, Path] = {}
    attribution_hashes: dict[str, str] = {}
    known_counts: dict[str, int] = {}
    inputs = _array(contract.get("inputs"), "materialization inputs")
    if len(inputs) != EXPECTED_REFERENCE_COUNT_PER_SONG:
        raise ProductionFoldPreflightError(
            f"{song_id} needs exactly five materialization inputs"
        )
    for raw in inputs:
        item = _mapping(raw, "materialization input")
        reference_id = str(item.get("reference_id") or "").strip()
        if not reference_id or reference_id in attribution_paths:
            raise ProductionFoldPreflightError(
                f"{song_id} materialization reference ids are blank or duplicate"
            )
        if item.get("source_count") != EXPECTED_SOURCE_COUNT_PER_SONG:
            raise ProductionFoldPreflightError(
                f"{song_id}:{reference_id} source count is not ten"
            )
        attribution_path, attribution_hash = _verify_file_record(
            item,
            path_key="attribution_path",
            sha256_key="attribution_file_sha256",
            name=f"{song_id}:{reference_id} attribution",
        )
        known_count = item.get("known_shot_count")
        if isinstance(known_count, bool) or not isinstance(known_count, int) or known_count < 1:
            raise ProductionFoldPreflightError(
                f"{song_id}:{reference_id} known shot count is invalid"
            )
        crop_records = _array(item.get("crop_artifacts"), "materialization crop artifacts")
        if (
            item.get("crop_artifact_count") != known_count
            or len(crop_records) != known_count
        ):
            raise ProductionFoldPreflightError(
                f"{song_id}:{reference_id} crop-v1 coverage is incomplete"
            )
        seen_shots: set[int] = set()
        for crop in crop_records:
            crop_item = _mapping(crop, "materialization crop artifact")
            shot_index = crop_item.get("shot_index")
            if (
                isinstance(shot_index, bool)
                or not isinstance(shot_index, int)
                or shot_index in seen_shots
            ):
                raise ProductionFoldPreflightError(
                    f"{song_id}:{reference_id} crop-v1 shot ids are invalid"
                )
            seen_shots.add(shot_index)
            _verify_file_record(
                crop_item,
                path_key="path",
                sha256_key="file_sha256",
                name=f"{song_id}:{reference_id} crop-v1 shot {shot_index}",
            )
            if not _is_sha256(crop_item.get("artifact_content_sha256")):
                raise ProductionFoldPreflightError("crop-v1 content hash is invalid")
        attribution_paths[reference_id] = attribution_path
        attribution_hashes[reference_id] = attribution_hash
        known_counts[reference_id] = known_count

    supervision_paths: dict[str, Path] = {}
    supervision_hashes: dict[str, str] = {}
    outputs = _array(payload.get("outputs"), "materialization outputs")
    if len(outputs) != EXPECTED_REFERENCE_COUNT_PER_SONG:
        raise ProductionFoldPreflightError(
            f"{song_id} needs exactly five materialization outputs"
        )
    for raw in outputs:
        item = _mapping(raw, "materialization output")
        reference_id = str(item.get("reference_id") or "").strip()
        if not reference_id or reference_id in supervision_paths:
            raise ProductionFoldPreflightError(
                f"{song_id} output reference ids are blank or duplicate"
            )
        _verify_file_record(
            item,
            path_key="dataset_path",
            sha256_key="dataset_file_sha256",
            name=f"{song_id}:{reference_id} dataset",
        )
        supervision_path, supervision_hash = _verify_file_record(
            item,
            path_key="supervision_path",
            sha256_key="supervision_file_sha256",
            name=f"{song_id}:{reference_id} supervision",
        )
        supervision_paths[reference_id] = supervision_path
        supervision_hashes[reference_id] = supervision_hash

    references = tuple(sorted(attribution_paths))
    if set(references) != set(supervision_paths):
        raise ProductionFoldPreflightError(
            f"{song_id} materialization input/output reference sets differ"
        )
    statistics = _mapping(payload.get("statistics"), "materialization statistics")
    if statistics.get("reference_count") != EXPECTED_REFERENCE_COUNT_PER_SONG:
        raise ProductionFoldPreflightError(
            f"{song_id} materialization statistics reference count mismatch"
        )
    return pointer, _MaterializationBinding(
        reference_ids=references,
        supervision_paths=supervision_paths,
        supervision_sha256=supervision_hashes,
        attribution_paths=attribution_paths,
        attribution_sha256=attribution_hashes,
        known_shot_counts=known_counts,
    )


def _validate_feature_manifest(
    pointer_value: object,
    *,
    song_id: str,
    materialization: _MaterializationBinding,
) -> tuple[ManifestPointer, _FeatureBinding]:
    pointer, payload = _load_pointer(pointer_value, f"{song_id} feature manifest")
    if payload.get("schema_version") != FEATURE_SNAPSHOT_JOB_SCHEMA_VERSION:
        raise ProductionFoldPreflightError(f"{song_id} feature schema mismatch")
    if payload.get("analysis_basis") != FEATURE_SNAPSHOT_JOB_ANALYSIS_BASIS:
        raise ProductionFoldPreflightError(
            f"{song_id} feature analysis basis mismatch"
        )
    if payload.get("song_id") != song_id:
        raise ProductionFoldPreflightError(f"{song_id} feature song id mismatch")
    manifest_hash = payload.get("manifest_sha256")
    unsigned = deepcopy(payload)
    unsigned.pop("manifest_sha256", None)
    if not _is_sha256(manifest_hash) or _canonical_sha256(unsigned) != manifest_hash:
        raise ProductionFoldPreflightError(f"{song_id} feature manifest hash mismatch")
    if pointer.path.name != (
        f"stageedit-feature-snapshot-job-manifest-{str(manifest_hash)[:16]}.json"
    ):
        raise ProductionFoldPreflightError(
            f"{song_id} feature filename is not content-addressed"
        )
    sampling = _mapping(payload.get("sampling"), "feature sampling")
    if dict(sampling) != dict(PRODUCTION_SAMPLING_CONTRACT):
        raise ProductionFoldPreflightError(
            f"{song_id} feature sampling contract mismatch"
        )
    if payload.get("source_count") != EXPECTED_SOURCE_COUNT_PER_SONG:
        raise ProductionFoldPreflightError(f"{song_id} feature source count is not ten")
    input_policy = _mapping(payload.get("input_policy"), "feature input policy")
    required_policy = {
        "features_available_before_current_editorial_choice": True,
        "strictly_past_causal_edit_history_only": True,
        "reference_match_scores_present": False,
        "selected_or_rejected_source_targets_present": False,
        "raw_frames_persisted": False,
        "identity_embeddings_persisted": False,
    }
    if any(input_policy.get(key) is not expected for key, expected in required_policy.items()):
        raise ProductionFoldPreflightError(f"{song_id} feature leakage policy mismatch")
    provenance = _mapping(payload.get("feature_provenance"), "feature provenance")
    contracts = set(_array(provenance.get("contracts"), "feature contracts"))
    if not REQUIRED_FEATURE_PROVENANCE_CONTRACTS <= contracts:
        raise ProductionFoldPreflightError(
            f"{song_id} feature provenance contracts are incomplete"
        )
    source_content = _array(payload.get("source_content"), "feature source content")
    if len(source_content) != EXPECTED_SOURCE_COUNT_PER_SONG:
        raise ProductionFoldPreflightError(
            f"{song_id} feature source catalog does not contain ten sources"
        )
    source_ids: set[str] = set()
    for raw in source_content:
        item = _mapping(raw, "feature source")
        source_id = str(item.get("source_id") or "").strip()
        source_path = Path(str(item.get("path") or ""))
        if not source_id or source_id in source_ids:
            raise ProductionFoldPreflightError(
                f"{song_id} feature source ids are blank or duplicate"
            )
        if not source_path.is_absolute() or not source_path.is_file() or source_path.is_symlink():
            raise ProductionFoldPreflightError(
                f"{song_id} feature source is unavailable: {source_path}"
            )
        if not _is_sha256(item.get("content_sha256")):
            raise ProductionFoldPreflightError(
                f"{song_id}:{source_id} feature source hash is invalid"
            )
        source_ids.add(source_id)
    # ``source_catalog_sha256`` intentionally binds the raw supervision
    # catalog, while ``source_content`` is its normalized decode identity.  Do
    # not compare those two different schemas; bind the hash back to all five
    # immutable supervision files instead.
    supervision_catalog_hashes: set[str] = set()
    for reference_id in materialization.reference_ids:
        supervision_payload = json.loads(
            materialization.supervision_paths[reference_id].read_text(
                encoding="utf-8"
            )
        )
        supervision_catalog = _array(
            _mapping(
                supervision_payload,
                f"{song_id}:{reference_id} supervision",
            ).get("source_catalog"),
            "supervision source catalog",
        )
        supervision_catalog_hashes.add(_canonical_sha256(supervision_catalog))
    if supervision_catalog_hashes != {payload.get("source_catalog_sha256")}:
        raise ProductionFoldPreflightError(
            f"{song_id} feature source catalog hash mismatch"
        )

    supervision_inputs = _array(
        payload.get("supervision_inputs"), "feature supervision inputs"
    )
    seen_supervision: set[str] = set()
    for raw in supervision_inputs:
        item = _mapping(raw, "feature supervision input")
        reference_id = str(item.get("reference_id") or "").strip()
        if reference_id in seen_supervision or reference_id not in materialization.reference_ids:
            raise ProductionFoldPreflightError(
                f"{song_id} feature supervision references are invalid"
            )
        if (
            Path(str(item.get("path") or "")).resolve()
            != materialization.supervision_paths[reference_id]
            or item.get("sha256") != materialization.supervision_sha256[reference_id]
        ):
            raise ProductionFoldPreflightError(
                f"{song_id}:{reference_id} feature/supervision binding mismatch"
            )
        seen_supervision.add(reference_id)
    if seen_supervision != set(materialization.reference_ids):
        raise ProductionFoldPreflightError(
            f"{song_id} feature manifest lacks one or more supervision inputs"
        )

    snapshots: dict[str, Path] = {}
    for raw in _array(payload.get("snapshots"), "feature snapshots"):
        item = _mapping(raw, "feature snapshot")
        reference_id = str(item.get("reference_id") or "").strip()
        if reference_id in snapshots or reference_id not in materialization.reference_ids:
            raise ProductionFoldPreflightError(
                f"{song_id} feature snapshot references are invalid"
            )
        if item.get("source_count") != EXPECTED_SOURCE_COUNT_PER_SONG:
            raise ProductionFoldPreflightError(
                f"{song_id}:{reference_id} snapshot source count is not ten"
            )
        path = Path(str(item.get("path") or ""))
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise ProductionFoldPreflightError(
                f"{song_id}:{reference_id} snapshot is unavailable: {path}"
            )
        snapshot_payload = json.loads(path.read_text(encoding="utf-8"))
        snapshot = StageEditFeatureSnapshot.from_dict(
            snapshot_payload,
            schema=DEFAULT_STAGE_EDIT_SCHEMA,
        )
        if snapshot.snapshot_sha256 != item.get("snapshot_sha256"):
            raise ProductionFoldPreflightError(
                f"{song_id}:{reference_id} snapshot content hash mismatch"
            )
        sequence_id = snapshot.sequence.sequence_id
        if not sequence_id.endswith(f":{reference_id}"):
            raise ProductionFoldPreflightError(
                f"{song_id}:{reference_id} snapshot sequence id mismatch"
            )
        snapshots[reference_id] = path.resolve()
    if set(snapshots) != set(materialization.reference_ids):
        raise ProductionFoldPreflightError(
            f"{song_id} feature snapshot reference set mismatch"
        )
    statistics = _mapping(payload.get("statistics"), "feature statistics")
    if (
        statistics.get("reference_count") != EXPECTED_REFERENCE_COUNT_PER_SONG
        or statistics.get("source_count") != EXPECTED_SOURCE_COUNT_PER_SONG
    ):
        raise ProductionFoldPreflightError(f"{song_id} feature statistics mismatch")
    per_source = _mapping(
        statistics.get("per_source"), "feature per-source statistics"
    )
    if set(per_source) != source_ids:
        raise ProductionFoldPreflightError(
            f"{song_id} feature per-source statistics mismatch"
        )
    observed_totals = {
        "model_pose60_observed_time_count": 0,
        "formation_layout134_observed_time_count": 0,
        "formation_upper_body_observed_time_count": 0,
    }
    for source_id, raw_stats in per_source.items():
        source_stats = _mapping(raw_stats, f"{song_id}:{source_id} feature statistics")
        available = source_stats.get("available_time_count")
        if isinstance(available, bool) or not isinstance(available, int) or available < 0:
            raise ProductionFoldPreflightError(
                f"{song_id}:{source_id} available feature count is invalid"
            )
        for field_name in (
            "model_pose60_observed_time_count",
            "formation_layout134_observed_time_count",
            "formation_people_observed_time_count",
            "formation_upper_body_observed_time_count",
        ):
            count = source_stats.get(field_name)
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                or count > available
            ):
                raise ProductionFoldPreflightError(
                    f"{song_id}:{source_id} {field_name} is invalid"
                )
            if field_name in observed_totals:
                observed_totals[field_name] += count
        upper_people = source_stats.get(
            "formation_upper_body_observed_person_count"
        )
        if (
            isinstance(upper_people, bool)
            or not isinstance(upper_people, int)
            or upper_people < 0
            or upper_people > 6 * available
        ):
            raise ProductionFoldPreflightError(
                f"{song_id}:{source_id} formation person coverage is invalid"
            )
    if any(count <= 0 for count in observed_totals.values()):
        raise ProductionFoldPreflightError(
            f"{song_id} action/formation extraction has zero observed coverage"
        )
    return pointer, _FeatureBinding(snapshot_paths=snapshots)


def _validate_dynamic_crop_manifest(
    pointer_value: object,
    *,
    song_id: str,
    materialization: _MaterializationBinding,
) -> ManifestPointer:
    pointer, payload = _load_pointer(pointer_value, f"{song_id} dynamic crop manifest")
    # This checks the manifest's own content hash, schema, provenance policy and
    # aggregate statistics.  Per-shot JSON is checked immediately below; large
    # media hashes are intentionally deferred to the corpus builder's one cache.
    validate_reference_dynamic_crop_manifest(payload, verify_artifact_files=False)
    dynamic_hash = _mapping(
        payload.get("provenance"), "dynamic crop provenance"
    ).get("manifest_content_sha256")
    if pointer.path.name != f"reference-dynamic-crop-v3-manifest-{dynamic_hash}.json":
        raise ProductionFoldPreflightError(
            f"{song_id} dynamic crop filename is not content-addressed"
        )
    selection = _mapping(payload.get("selection"), "dynamic crop selection")
    selected_references = tuple(sorted(_array(selection.get("reference_ids"), "dynamic references")))
    if selected_references != materialization.reference_ids:
        raise ProductionFoldPreflightError(
            f"{song_id} dynamic/materialization reference sets differ"
        )
    if selection.get("only_retained_canonical_shots") is not True:
        raise ProductionFoldPreflightError(
            f"{song_id} dynamic crop selection is not canonical-only"
        )
    selected_shots = _mapping(
        selection.get("shot_indices_by_reference"), "dynamic selected shots"
    )

    dynamic_inputs: set[str] = set()
    source_catalog_hashes: set[str] = set()
    for raw in _array(payload.get("inputs"), "dynamic crop inputs"):
        item = _mapping(raw, "dynamic crop input")
        reference_id = str(item.get("reference_id") or "").strip()
        if reference_id in dynamic_inputs or reference_id not in materialization.reference_ids:
            raise ProductionFoldPreflightError(
                f"{song_id} dynamic crop input references are invalid"
            )
        attribution = _mapping(item.get("attribution"), "dynamic attribution")
        if (
            Path(str(attribution.get("path") or "")).resolve()
            != materialization.attribution_paths[reference_id]
            or attribution.get("file_sha256")
            != materialization.attribution_sha256[reference_id]
        ):
            raise ProductionFoldPreflightError(
                f"{song_id}:{reference_id} dynamic attribution binding mismatch"
            )
        evidence = _mapping(item.get("evidence"), "dynamic evidence")
        _verify_file_record(
            evidence,
            path_key="path",
            sha256_key="file_sha256",
            name=f"{song_id}:{reference_id} dynamic evidence",
        )
        reference_media = Path(str(item.get("reference_media_path") or ""))
        if (
            not reference_media.is_absolute()
            or not reference_media.is_file()
            or reference_media.is_symlink()
        ):
            raise ProductionFoldPreflightError(
                f"{song_id}:{reference_id} reference media is unavailable"
            )
        catalog_hash = item.get("source_catalog_content_sha256")
        if not _is_sha256(catalog_hash):
            raise ProductionFoldPreflightError(
                f"{song_id}:{reference_id} source catalog hash is invalid"
            )
        source_catalog_hashes.add(str(catalog_hash))
        dynamic_inputs.add(reference_id)
    if dynamic_inputs != set(materialization.reference_ids):
        raise ProductionFoldPreflightError(
            f"{song_id} dynamic crop input reference set mismatch"
        )
    if len(source_catalog_hashes) != 1:
        raise ProductionFoldPreflightError(
            f"{song_id} dynamic references do not share one ten-source catalog"
        )

    observed_shots: dict[str, set[int]] = {
        reference_id: set() for reference_id in materialization.reference_ids
    }
    for raw in _array(payload.get("artifacts"), "dynamic crop artifacts"):
        item = _mapping(raw, "dynamic crop artifact")
        reference_id = str(item.get("reference_id") or "").strip()
        shot_index = item.get("shot_index")
        if (
            reference_id not in observed_shots
            or isinstance(shot_index, bool)
            or not isinstance(shot_index, int)
            or shot_index in observed_shots[reference_id]
        ):
            raise ProductionFoldPreflightError(
                f"{song_id} dynamic crop artifact identity is invalid"
            )
        artifact = _mapping(item.get("artifact"), "dynamic crop artifact provenance")
        path, _ = _verify_file_record(
            artifact,
            path_key="path",
            sha256_key="file_sha256",
            name=f"{song_id}:{reference_id} crop-v2 shot {shot_index}",
        )
        restored = load_reference_dynamic_crop_supervision(
            path,
            verify_input_files=False,
        )
        if (
            restored.payload["provenance"]["artifact_content_sha256"]
            != artifact.get("content_sha256")
        ):
            raise ProductionFoldPreflightError(
                f"{song_id}:{reference_id} crop-v2 content hash mismatch"
            )
        observed_shots[reference_id].add(shot_index)

    for reference_id, observed in observed_shots.items():
        declared = selected_shots.get(reference_id)
        if not isinstance(declared, list) or set(declared) != observed:
            raise ProductionFoldPreflightError(
                f"{song_id}:{reference_id} dynamic selection/artifact shots differ"
            )
        if len(observed) != materialization.known_shot_counts[reference_id]:
            raise ProductionFoldPreflightError(
                f"{song_id}:{reference_id} dynamic crop coverage is incomplete"
            )
    return pointer


def _validate_storage(checkpoint_directory: Path) -> int:
    volume = PRODUCTION_VOLUME_ROOT.resolve()
    if not volume.is_dir() or not os.path.ismount(volume):
        raise ProductionFoldPreflightError(
            f"required external SSD is not mounted: {volume}"
        )
    checkpoint = Path(checkpoint_directory).expanduser().resolve()
    if volume != checkpoint and volume not in checkpoint.parents:
        raise ProductionFoldPreflightError(
            f"checkpoint directory must be on {volume}: {checkpoint}"
        )
    if checkpoint.exists() or checkpoint.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite checkpoint directory: {checkpoint}"
        )
    free = shutil.disk_usage(volume).free
    if free < MINIMUM_FREE_BYTES:
        raise ProductionFoldPreflightError(
            "external SSD free space is below the fixed production minimum: "
            f"{free} < {MINIMUM_FREE_BYTES}"
        )
    return free


def _strict_plan_payload(path: Path) -> dict[str, Any]:
    plan_path = Path(path).expanduser().resolve()
    if not plan_path.is_file() or plan_path.is_symlink():
        raise ProductionFoldPreflightError(
            f"production plan must be a regular non-symlink file: {plan_path}"
        )
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProductionFoldPreflightError("production plan root must be an object")
    result = dict(payload)
    expected_keys = {
        "schema_version",
        "analysis_basis",
        "production_contract",
        "required_volume",
        "minimum_free_bytes",
        "songs",
        "folds",
        "plan_sha256",
    }
    if set(result) != expected_keys:
        raise ProductionFoldPreflightError("production plan keys are not exact")
    if result.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ProductionFoldPreflightError("production plan schema mismatch")
    if result.get("analysis_basis") != PLAN_ANALYSIS_BASIS:
        raise ProductionFoldPreflightError("production plan analysis basis mismatch")
    observed_hash = result.get("plan_sha256")
    unsigned = deepcopy(result)
    unsigned.pop("plan_sha256", None)
    if not _is_sha256(observed_hash) or _canonical_sha256(unsigned) != observed_hash:
        raise ProductionFoldPreflightError("production plan content hash mismatch")
    if plan_path.name != f"stageedit-production-fold-plan-{str(observed_hash)[:16]}.json":
        raise ProductionFoldPreflightError("production plan filename is not content-addressed")
    return result


def preflight_stage_edit_production_fold(
    plan_path: Path,
    *,
    fold_id: str,
) -> ProductionFoldPreflight:
    """Validate one fixed IVE fold and expand its exact low-level train args."""

    resolved_plan = Path(plan_path).expanduser().resolve()
    plan = _strict_plan_payload(resolved_plan)
    contract = _mapping(plan.get("production_contract"), "production contract")
    expected_contract = {
        "artist": "IVE",
        "song_ids": list(PRODUCTION_SONG_IDS),
        "expected_reference_count_per_song": EXPECTED_REFERENCE_COUNT_PER_SONG,
        "expected_source_count_per_song": EXPECTED_SOURCE_COUNT_PER_SONG,
        "sampling": dict(PRODUCTION_SAMPLING_CONTRACT),
        "recent_history_mask_contract": dict(
            PRODUCTION_RECENT_HISTORY_MASK_CONTRACT
        ),
        "held_out_used_for_model_selection": False,
        "synthetic_or_fake_training_data_allowed": False,
        "quality_threshold_override_allowed": False,
    }
    if dict(contract) != expected_contract:
        raise ProductionFoldPreflightError("production contract is not exact")
    if (
        plan.get("required_volume") != str(PRODUCTION_VOLUME_ROOT)
        or plan.get("minimum_free_bytes") != MINIMUM_FREE_BYTES
    ):
        raise ProductionFoldPreflightError("production storage contract mismatch")

    songs_raw = _array(plan.get("songs"), "production songs")
    songs: dict[str, Mapping[str, Any]] = {}
    for raw in songs_raw:
        item = _mapping(raw, "production song")
        if set(item) != {
            "song_id",
            "materialization_manifest",
            "feature_manifest",
            "dynamic_crop_manifest",
        }:
            raise ProductionFoldPreflightError("production song keys are not exact")
        song_id = str(item.get("song_id") or "").strip()
        if song_id in songs:
            raise ProductionFoldPreflightError("duplicate production song id")
        songs[song_id] = item
    if set(songs) != set(PRODUCTION_SONG_IDS):
        raise ProductionFoldPreflightError("production plan must contain the exact three songs")

    folds_raw = _array(plan.get("folds"), "production folds")
    folds: dict[str, Mapping[str, Any]] = {}
    for raw in folds_raw:
        item = _mapping(raw, "production fold")
        if set(item) != {"fold_id", "held_out_song_id", "checkpoint_directory"}:
            raise ProductionFoldPreflightError("production fold keys are not exact")
        observed_fold = str(item.get("fold_id") or "").strip()
        if observed_fold in folds:
            raise ProductionFoldPreflightError("duplicate production fold id")
        folds[observed_fold] = item
    if set(folds) != set(PRODUCTION_FOLDS):
        raise ProductionFoldPreflightError("both fixed production folds are required")
    for expected_fold, expected_heldout in PRODUCTION_FOLDS.items():
        if folds[expected_fold].get("held_out_song_id") != expected_heldout:
            raise ProductionFoldPreflightError(
                f"{expected_fold} held-out song is not fixed"
            )
    if fold_id not in folds:
        raise ProductionFoldPreflightError(f"unknown production fold: {fold_id}")
    checkpoint = Path(str(folds[fold_id].get("checkpoint_directory") or "")).resolve()
    if len(
        {
            str(Path(str(value.get("checkpoint_directory") or "")).resolve())
            for value in folds.values()
        }
    ) != len(folds):
        raise ProductionFoldPreflightError("production fold checkpoint paths overlap")
    free_bytes = _validate_storage(checkpoint)

    materializations: dict[str, _MaterializationBinding] = {}
    features: dict[str, _FeatureBinding] = {}
    dynamic_paths: dict[str, Path] = {}
    for song_id in PRODUCTION_SONG_IDS:
        item = songs[song_id]
        _, materialization = _validate_materialization_manifest(
            item.get("materialization_manifest"),
            song_id=song_id,
        )
        _, feature = _validate_feature_manifest(
            item.get("feature_manifest"),
            song_id=song_id,
            materialization=materialization,
        )
        dynamic_pointer = _validate_dynamic_crop_manifest(
            item.get("dynamic_crop_manifest"),
            song_id=song_id,
            materialization=materialization,
        )
        materializations[song_id] = materialization
        features[song_id] = feature
        dynamic_paths[song_id] = dynamic_pointer.path

    held_out = PRODUCTION_FOLDS[fold_id]
    args: list[str] = []
    for song_id in PRODUCTION_SONG_IDS:
        materialization = materializations[song_id]
        feature = features[song_id]
        for reference_id in materialization.reference_ids:
            args.extend(
                [
                    "--input",
                    str(materialization.supervision_paths[reference_id]),
                    "--song-id",
                    f"{reference_id}={song_id}",
                    "--feature-snapshot",
                    f"{reference_id}={feature.snapshot_paths[reference_id]}",
                ]
            )
        args.extend(["--dynamic-crop-manifest", str(dynamic_paths[song_id])])
    args.extend(
        [
            "--held-out-song",
            held_out,
            "--checkpoint-dir",
            str(checkpoint),
            "--negative-samples-per-hold",
            str(PRODUCTION_SAMPLING_CONTRACT["negative_samples_per_hold"]),
            "--boundary-guard-seconds",
            str(PRODUCTION_SAMPLING_CONTRACT["boundary_guard_seconds"]),
            "--max-sources",
            str(PRODUCTION_SAMPLING_CONTRACT["max_sources"]),
        ]
    )
    return ProductionFoldPreflight(
        plan_path=resolved_plan,
        plan_sha256=str(plan["plan_sha256"]),
        fold_id=fold_id,
        held_out_song_id=held_out,
        training_song_ids=tuple(
            song for song in PRODUCTION_SONG_IDS if song != held_out
        ),
        checkpoint_directory=checkpoint,
        free_bytes=free_bytes,
        recent_history_mask_contract=dict(
            PRODUCTION_RECENT_HISTORY_MASK_CONTRACT
        ),
        reference_ids_by_song={
            song: materializations[song].reference_ids for song in PRODUCTION_SONG_IDS
        },
        train_cli_args=tuple(args),
    )


def _keyed_paths(values: Sequence[str], option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        song_id, separator, raw_path = value.partition("=")
        song_id = song_id.strip()
        if not separator or not song_id or not raw_path.strip():
            raise ProductionFoldPreflightError(
                f"{option} must use SONG_ID=/absolute/path.json"
            )
        if song_id in result:
            raise ProductionFoldPreflightError(f"duplicate {option} song: {song_id}")
        result[song_id] = Path(raw_path.strip()).expanduser().resolve()
    if set(result) != set(PRODUCTION_SONG_IDS):
        raise ProductionFoldPreflightError(
            f"{option} must name the exact three production songs"
        )
    return result


def create_stage_edit_production_plan(
    *,
    materialization_manifests: Mapping[str, Path],
    feature_manifests: Mapping[str, Path],
    dynamic_crop_manifests: Mapping[str, Path],
    checkpoint_root: Path,
    output_directory: Path,
) -> Path:
    """Write an exclusive content-addressed plan from nine manifest paths."""

    for name, values in (
        ("materialization", materialization_manifests),
        ("feature", feature_manifests),
        ("dynamic crop", dynamic_crop_manifests),
    ):
        if set(values) != set(PRODUCTION_SONG_IDS):
            raise ProductionFoldPreflightError(
                f"{name} manifests must name the exact three production songs"
            )
    root = Path(checkpoint_root).expanduser().resolve()
    volume = PRODUCTION_VOLUME_ROOT.resolve()
    if volume != root and volume not in root.parents:
        raise ProductionFoldPreflightError(
            f"checkpoint root must be on {volume}: {root}"
        )

    def pointer(path: Path, name: str) -> dict[str, str]:
        resolved = Path(path).expanduser().resolve()
        return ManifestPointer(
            resolved,
            _stable_file_sha256(resolved, name),
        ).to_dict()

    unsigned: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "analysis_basis": PLAN_ANALYSIS_BASIS,
        "production_contract": {
            "artist": "IVE",
            "song_ids": list(PRODUCTION_SONG_IDS),
            "expected_reference_count_per_song": EXPECTED_REFERENCE_COUNT_PER_SONG,
            "expected_source_count_per_song": EXPECTED_SOURCE_COUNT_PER_SONG,
            "sampling": dict(PRODUCTION_SAMPLING_CONTRACT),
            "recent_history_mask_contract": dict(
                PRODUCTION_RECENT_HISTORY_MASK_CONTRACT
            ),
            "held_out_used_for_model_selection": False,
            "synthetic_or_fake_training_data_allowed": False,
            "quality_threshold_override_allowed": False,
        },
        "required_volume": str(PRODUCTION_VOLUME_ROOT),
        "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "songs": [
            {
                "song_id": song_id,
                "materialization_manifest": pointer(
                    materialization_manifests[song_id],
                    f"{song_id} materialization manifest",
                ),
                "feature_manifest": pointer(
                    feature_manifests[song_id],
                    f"{song_id} feature manifest",
                ),
                "dynamic_crop_manifest": pointer(
                    dynamic_crop_manifests[song_id],
                    f"{song_id} dynamic crop manifest",
                ),
            }
            for song_id in PRODUCTION_SONG_IDS
        ],
        "folds": [
            {
                "fold_id": fold_id,
                "held_out_song_id": held_out,
                "checkpoint_directory": str(root / fold_id),
            }
            for fold_id, held_out in PRODUCTION_FOLDS.items()
        ],
    }
    plan_hash = _canonical_sha256(unsigned)
    payload = {**unsigned, "plan_sha256": plan_hash}
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"stageedit-production-fold-plan-{plan_hash[:16]}.json"
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != serialized:
            raise ProductionFoldPreflightError(
                f"content-addressed plan path contains different bytes: {path}"
            )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create, preflight, or launch the two fixed IVE StageEdit folds"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="bind nine immutable song manifests")
    create.add_argument("--materialization-manifest", action="append", required=True)
    create.add_argument("--feature-manifest", action="append", required=True)
    create.add_argument("--dynamic-crop-manifest", action="append", required=True)
    create.add_argument("--checkpoint-root", required=True, type=Path)
    create.add_argument("--output-directory", required=True, type=Path)
    for command in ("preflight", "launch"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--plan", required=True, type=Path)
        subparser.add_argument("--fold", required=True, choices=tuple(PRODUCTION_FOLDS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "create":
        path = create_stage_edit_production_plan(
            materialization_manifests=_keyed_paths(
                args.materialization_manifest, "--materialization-manifest"
            ),
            feature_manifests=_keyed_paths(
                args.feature_manifest, "--feature-manifest"
            ),
            dynamic_crop_manifests=_keyed_paths(
                args.dynamic_crop_manifest, "--dynamic-crop-manifest"
            ),
            checkpoint_root=args.checkpoint_root,
            output_directory=args.output_directory,
        )
        print(json.dumps({"plan_path": str(path)}, ensure_ascii=False, indent=2))
        return 0
    result = preflight_stage_edit_production_fold(args.plan, fold_id=args.fold)
    if args.command == "preflight":
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    # Activation is intentionally absent.  Both folds must be reviewed before
    # a separately authorized registry operation can select a production bundle.
    return production_train_main(result.train_cli_args)


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "EXPECTED_REFERENCE_COUNT_PER_SONG",
    "EXPECTED_SOURCE_COUNT_PER_SONG",
    "MINIMUM_FREE_BYTES",
    "PLAN_ANALYSIS_BASIS",
    "PLAN_SCHEMA_VERSION",
    "PRODUCTION_FOLDS",
    "PRODUCTION_RECENT_HISTORY_MASK_CONTRACT",
    "PRODUCTION_SAMPLING_CONTRACT",
    "PRODUCTION_SONG_IDS",
    "PRODUCTION_VOLUME_ROOT",
    "ProductionFoldPreflight",
    "ProductionFoldPreflightError",
    "create_stage_edit_production_plan",
    "preflight_stage_edit_production_fold",
]
