"""Immutable three-song Dynamic Crop v2 capacity-fold launcher.

This launcher is deliberately narrower than the release launcher.  It admits
only the three fixed capacity-selection songs, validates all nine immutable
manifest inputs, and emits exactly three whole-song folds for each selected
default capacity candidate.  It never grants production activation authority.

The plan is useful evidence, not merely a convenience wrapper: it binds a
content-addressed selection dataset, every materialization output and feature
snapshot, the exact production-trainer arguments, a clean Git HEAD, and the
Git/file hashes of the code that interprets the plan.  A launch replays all of
those checks before invoking the production trainer.  A successful inactive
quality-gated run is immediately wrapped with ``write_fold_run_evidence`` for
the capacity selector.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from . import stage_edit_production_fold_launcher as legacy
from .stage_edit_dynamic_crop_capacity_probe import (
    BANG_BANG,
    BLACKHOLE,
    REBEL_HEART,
    PROBE_TRAINING_SCHEDULE,
    SELECTION_SONG_IDS,
    DynamicCropCapacityCandidate,
    DynamicCropCapacityProbeError,
    _load_selection_dataset,
    default_dynamic_crop_v2_probe_candidates,
    write_fold_run_evidence,
)
from .stage_edit_dynamic_crop_head_v2 import (
    DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME,
    DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME,
)
from .stage_edit_production_train_cli import main as production_train_main
from .stage_edit_production_training import STAGE_EDIT_SELECTION_FREEZE_FILENAME
from .stage_edit_transformer import PRODUCTION_TRAINING_REPORT_FILENAME


CAPACITY_FOLD_PLAN_SCHEMA_VERSION = 1
CAPACITY_FOLD_PLAN_ANALYSIS_BASIS = (
    "ive_bang_blackhole_rebel_dynamic_crop_capacity_fold_plan_v1"
)
CAPACITY_FOLD_PLAN_PREFIX = "stageedit-dynamic-crop-capacity-fold-plan-"
CAPACITY_FOLD_CONTRACT = "exact_three_song_whole_song_capacity_cross_validation_v1"
TRAIN_MODULE = "crosscut.stage_edit_production_train_cli"
PRODUCTION_VOLUME_ROOT = Path("/Volumes/CrossCutMedia")
MAXIMUM_CUT_POSITIVE_WEIGHT = 8.0

_EXPECTED_SONG_IDS = (BANG_BANG, BLACKHOLE, REBEL_HEART)
if tuple(SELECTION_SONG_IDS) != _EXPECTED_SONG_IDS:  # pragma: no cover - import guard
    raise RuntimeError("capacity selection song contract drifted")

SOURCE_RELATIVE_PATHS = (
    "engine/crosscut/stage_edit_dynamic_crop_capacity_fold_launcher.py",
    "engine/crosscut/stage_edit_dynamic_crop_capacity_probe.py",
    "engine/crosscut/stage_edit_production_fold_launcher.py",
    "engine/crosscut/stage_edit_production_train_cli.py",
    "engine/crosscut/stage_edit_production_training.py",
    "engine/crosscut/stage_edit_classifier_warmup.py",
    "engine/crosscut/stage_edit_dynamic_crop_training.py",
    "engine/crosscut/stage_edit_dynamic_crop_head_v2.py",
    "engine/crosscut/stage_edit_transformer.py",
    "engine/crosscut/stage_edit_reference_adapter.py",
    "engine/crosscut/stage_edit_feature_materializer.py",
    "engine/crosscut/reference_dynamic_crop_job.py",
    "engine/crosscut/reference_dynamic_crop_supervision.py",
)


class CapacityFoldLauncherError(ValueError):
    """Raised before any capacity-fold training when evidence is unsafe."""


@dataclass(frozen=True, slots=True)
class _ValidatedManifestSet:
    selection_dataset_pointer: Mapping[str, str]
    selection_dataset_content_sha256: str
    song_records: tuple[Mapping[str, Any], ...]
    reference_ids_by_song: Mapping[str, tuple[str, ...]]
    supervision_paths_by_song: Mapping[str, Mapping[str, Path]]
    snapshot_paths_by_song: Mapping[str, Mapping[str, Path]]
    dynamic_manifest_paths_by_song: Mapping[str, Path]


@dataclass(frozen=True, slots=True)
class CapacityFoldPreflight:
    plan_path: Path
    plan_content_sha256: str
    candidate_name: str
    candidate_id: str
    held_out_song_id: str
    training_song_ids: tuple[str, ...]
    checkpoint_directory: Path
    evidence_directory: Path
    selection_dataset_manifest_path: Path
    reference_ids_by_song: Mapping[str, tuple[str, ...]]
    train_cli_args: tuple[str, ...]
    command_argv: tuple[str, ...]
    free_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_path": str(self.plan_path),
            "plan_content_sha256": self.plan_content_sha256,
            "candidate_name": self.candidate_name,
            "candidate_id": self.candidate_id,
            "held_out_song_id": self.held_out_song_id,
            "training_song_ids": list(self.training_song_ids),
            "checkpoint_directory": str(self.checkpoint_directory),
            "evidence_directory": str(self.evidence_directory),
            "selection_dataset_manifest_path": str(
                self.selection_dataset_manifest_path
            ),
            "reference_ids_by_song": {
                song: list(references)
                for song, references in sorted(self.reference_ids_by_song.items())
            },
            "free_bytes": self.free_bytes,
            "training_authorized": True,
            "production_activation_authorized": False,
            "production_registry_activation_requested": False,
            "train_cli_args": list(self.train_cli_args),
            "command_argv": list(self.command_argv),
        }


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
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CapacityFoldLauncherError(f"{label} must be an object")
    return dict(value)


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CapacityFoldLauncherError(f"{label} must be an array")
    return value


def _reject_quarantined_text(value: object, label: str) -> str:
    """Reject quarantined song spellings before any filesystem operation."""

    text = str(value).strip()
    if not text:
        raise CapacityFoldLauncherError(f"{label} cannot be blank")
    compact = "".join(character for character in text.upper() if character.isalnum())
    # Build the forbidden title spellings from the capacity module's private
    # quarantine contract.  Keeping their literals out of plans prevents an
    # accidental future path from becoming admissible evidence.
    from .stage_edit_dynamic_crop_capacity_probe import QUARANTINED_SONG_IDS

    for song_id in QUARANTINED_SONG_IDS:
        song_compact = "".join(
            character for character in str(song_id).upper() if character.isalnum()
        )
        title_compact = song_compact.removeprefix("IVE")
        if song_compact in compact or title_compact in compact:
            raise CapacityFoldLauncherError(
                f"{label} references a quarantined song"
            )
    return text


def _scan_payload_for_quarantined_text(value: object, label: str) -> None:
    if isinstance(value, str):
        # Empty optional metadata is not a song identity.  Path/identity fields
        # still use ``_reject_quarantined_text`` directly and remain non-empty.
        if value.strip():
            _reject_quarantined_text(value, label)
            candidate = Path(value).expanduser()
            if candidate.is_absolute():
                _absolute_safe_path(candidate, label)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            # Schema keys can describe generic concepts; only stored values can
            # identify a song, reference, or filesystem object.
            _scan_payload_for_quarantined_text(child, f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _scan_payload_for_quarantined_text(child, f"{label}[{index}]")


def _absolute_safe_path(value: Path | str, label: str) -> Path:
    raw_text = _reject_quarantined_text(value, f"{label} path")
    raw = Path(raw_text).expanduser()
    if not raw.is_absolute():
        raise CapacityFoldLauncherError(f"{label} path must be absolute")
    try:
        resolved = raw.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise CapacityFoldLauncherError(
            f"{label} path cannot be resolved safely: {exc}"
        ) from exc
    # A safe-looking alias must not hide a quarantined song in its canonical
    # target ancestry. Check the complete canonical spelling as well as each
    # component so a parent-directory alias cannot bypass the quarantine.
    _reject_quarantined_text(resolved, f"{label} resolved path")
    for ancestor in (resolved, *resolved.parents):
        if ancestor.name:
            _reject_quarantined_text(
                ancestor.name, f"{label} resolved path ancestry"
            )

    current = Path(raw.anchor)
    for component in raw.parts[1:]:
        current /= component
        if current.is_symlink():
            raise CapacityFoldLauncherError(
                f"{label} path must not traverse a symlink component: {current}"
            )
    return resolved


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_artifact_path_isolation(
    *,
    plan_path: Path,
    plan_label: str,
    evidence_directory: Path,
    planned_checkpoint_directories: Sequence[Path],
) -> None:
    """Keep plans/evidence disjoint from every immutable checkpoint tree."""

    if _paths_overlap(plan_path, evidence_directory):
        raise CapacityFoldLauncherError(
            f"{plan_label} and evidence directory must not overlap"
        )
    seen: list[Path] = []
    for checkpoint in planned_checkpoint_directories:
        if any(_paths_overlap(checkpoint, prior) for prior in seen):
            raise CapacityFoldLauncherError(
                "planned checkpoint directories must not overlap"
            )
        seen.append(checkpoint)
        for label, path in (
            (plan_label, plan_path),
            ("evidence directory", evidence_directory),
        ):
            if _paths_overlap(path, checkpoint):
                raise CapacityFoldLauncherError(
                    f"{label} must not overlap a planned checkpoint directory"
                )


def _regular_safe_path(value: Path | str, label: str) -> Path:
    path = _absolute_safe_path(value, label)
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise CapacityFoldLauncherError(f"{label} is unavailable: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise CapacityFoldLauncherError(f"{label} must be a regular file")
    return path


def _stable_file_sha256(path: Path | str, label: str) -> str:
    resolved = _regular_safe_path(path, label)
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = resolved.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise CapacityFoldLauncherError(f"{label} changed while hashing")
    return digest.hexdigest()


def _pointer(path: Path | str, label: str) -> dict[str, str]:
    resolved = _regular_safe_path(path, label)
    return {
        "path": str(resolved),
        "file_sha256": _stable_file_sha256(resolved, label),
    }


def _verify_pointer(value: object, label: str) -> Path:
    pointer = _mapping(value, label)
    if set(pointer) != {"path", "file_sha256"}:
        raise CapacityFoldLauncherError(f"{label} pointer keys are not exact")
    path = _regular_safe_path(str(pointer.get("path") or ""), label)
    expected = pointer.get("file_sha256")
    if not _is_sha256(expected):
        raise CapacityFoldLauncherError(f"{label} SHA-256 is invalid")
    if _stable_file_sha256(path, label) != expected:
        raise CapacityFoldLauncherError(f"{label} SHA-256 mismatch")
    return path


def _read_json_object(path: Path | str, label: str) -> dict[str, Any]:
    resolved = _regular_safe_path(path, label)
    before = resolved.stat()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CapacityFoldLauncherError(f"{label} is unreadable: {exc}") from exc
    after = resolved.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise CapacityFoldLauncherError(f"{label} changed while reading")
    root = _mapping(payload, label)
    _scan_payload_for_quarantined_text(root, label)
    return root


def _git_output(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ("git", "-C", str(repo), *args),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CapacityFoldLauncherError(
            f"Git provenance command failed: {' '.join(args)}"
        ) from exc


def _capture_source_provenance(
    repository_root: Path | None = None,
    *,
    source_relative_paths: Sequence[str] = SOURCE_RELATIVE_PATHS,
) -> dict[str, Any]:
    """Bind a completely clean repository and exact committed source blobs."""

    if repository_root is None:
        probe = Path(__file__).resolve().parents[2]
        root_text = _git_output(probe, "rev-parse", "--show-toplevel").decode(
            "utf-8"
        ).strip()
        root = _absolute_safe_path(root_text, "repository root")
    else:
        root = _absolute_safe_path(repository_root, "repository root")
    status = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise CapacityFoldLauncherError(
            "capacity-fold planning/launch requires a completely clean repository"
        )
    head = _git_output(root, "rev-parse", "HEAD").decode("ascii").strip()
    if len(head) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in head
    ):
        raise CapacityFoldLauncherError("Git HEAD identity is invalid")
    files: dict[str, dict[str, str]] = {}
    for raw_relative in source_relative_paths:
        relative = str(raw_relative).strip()
        if (
            not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in files
        ):
            raise CapacityFoldLauncherError("source provenance path is unsafe")
        _reject_quarantined_text(relative, "source provenance path")
        path = root / relative
        _git_output(root, "ls-files", "--error-unmatch", "--", relative)
        head_blob = _git_output(root, "rev-parse", f"HEAD:{relative}").decode(
            "ascii"
        ).strip()
        worktree_blob = _git_output(root, "hash-object", "--", relative).decode(
            "ascii"
        ).strip()
        if head_blob != worktree_blob:
            raise CapacityFoldLauncherError(
                f"committed/worktree source blob differs: {relative}"
            )
        files[relative] = {
            "git_blob_oid": head_blob,
            "file_sha256": _stable_file_sha256(path, f"source {relative}"),
        }
    return {
        "contract": "clean_head_and_exact_interpreter_source_blobs_v1",
        "repository_root": str(root),
        "head_commit_oid": head,
        "repository_status_clean": True,
        "repository_status_porcelain_sha256": hashlib.sha256(status).hexdigest(),
        "source_files": files,
    }


def _validate_source_provenance(value: object) -> dict[str, Any]:
    payload = _mapping(value, "source provenance")
    if set(payload) != {
        "contract",
        "repository_root",
        "head_commit_oid",
        "repository_status_clean",
        "repository_status_porcelain_sha256",
        "source_files",
    }:
        raise CapacityFoldLauncherError("source provenance keys are not exact")
    if (
        payload.get("contract")
        != "clean_head_and_exact_interpreter_source_blobs_v1"
        or payload.get("repository_status_clean") is not True
        or payload.get("repository_status_porcelain_sha256")
        != hashlib.sha256(b"").hexdigest()
    ):
        raise CapacityFoldLauncherError("source provenance contract is invalid")
    root = _absolute_safe_path(
        str(payload.get("repository_root") or ""), "repository root"
    )
    observed = _capture_source_provenance(root)
    if observed != payload:
        raise CapacityFoldLauncherError(
            "current clean HEAD/source blobs differ from immutable plan"
        )
    return payload


def _require_same_source_provenance(expected: Mapping[str, Any]) -> None:
    root = _absolute_safe_path(
        str(expected.get("repository_root") or ""), "repository root"
    )
    if _capture_source_provenance(root) != dict(expected):
        raise CapacityFoldLauncherError(
            "clean HEAD/source blobs changed during evidence validation"
        )


def _paths_under_external_volume(path: Path | str, label: str) -> Path:
    candidate = _absolute_safe_path(path, label)
    volume = PRODUCTION_VOLUME_ROOT.resolve()
    if candidate != volume and volume not in candidate.parents:
        raise CapacityFoldLauncherError(
            f"{label} must be explicitly under {volume}: {candidate}"
        )
    return candidate


def _keyed_paths(values: Sequence[str], option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        _reject_quarantined_text(raw, option)
        song_id, separator, path_text = raw.partition("=")
        song_id = song_id.strip()
        if not separator or song_id not in SELECTION_SONG_IDS or song_id in result:
            raise CapacityFoldLauncherError(
                f"{option} must name each fixed capacity song exactly once"
            )
        result[song_id] = _absolute_safe_path(path_text, option)
    if tuple(result) != tuple(SELECTION_SONG_IDS) and set(result) != set(
        SELECTION_SONG_IDS
    ):
        raise CapacityFoldLauncherError(
            f"{option} must contain exactly the three capacity songs"
        )
    return result


def _validate_exact_manifest_maps(
    materialization_manifests: Mapping[str, Path],
    feature_manifests: Mapping[str, Path],
    dynamic_crop_manifests: Mapping[str, Path],
) -> None:
    expected = set(SELECTION_SONG_IDS)
    for label, values in (
        ("materialization", materialization_manifests),
        ("feature", feature_manifests),
        ("dynamic crop", dynamic_crop_manifests),
    ):
        if set(values) != expected:
            raise CapacityFoldLauncherError(
                f"{label} manifests must name exactly the three capacity songs"
            )
        for song_id, path in values.items():
            if song_id not in expected:
                raise CapacityFoldLauncherError(f"unexpected {label} song id")
            _absolute_safe_path(path, f"{song_id} {label} manifest")


def _materialization_output_records(
    manifest_path: Path,
    *,
    song_id: str,
    materialization: legacy._MaterializationBinding,
) -> tuple[dict[str, Any], ...]:
    payload = _read_json_object(
        manifest_path, f"{song_id} materialization manifest"
    )
    records: dict[str, dict[str, Any]] = {}
    for raw in _array(payload.get("outputs"), "materialization outputs"):
        item = _mapping(raw, "materialization output")
        reference_id = str(item.get("reference_id") or "").strip()
        if reference_id not in materialization.reference_ids or reference_id in records:
            raise CapacityFoldLauncherError(
                f"{song_id} materialization output references are invalid"
            )
        dataset_path = _regular_safe_path(
            str(item.get("dataset_path") or ""),
            f"{song_id}:{reference_id} dataset",
        )
        supervision_path = _regular_safe_path(
            str(item.get("supervision_path") or ""),
            f"{song_id}:{reference_id} supervision",
        )
        dataset_sha = _stable_file_sha256(
            dataset_path, f"{song_id}:{reference_id} dataset"
        )
        supervision_sha = _stable_file_sha256(
            supervision_path, f"{song_id}:{reference_id} supervision"
        )
        if (
            dataset_sha != item.get("dataset_file_sha256")
            or supervision_sha != item.get("supervision_file_sha256")
            or supervision_path != materialization.supervision_paths[reference_id]
            or supervision_sha != materialization.supervision_sha256[reference_id]
        ):
            raise CapacityFoldLauncherError(
                f"{song_id}:{reference_id} materialization output binding mismatch"
            )
        dataset_payload = _read_json_object(
            dataset_path, f"{song_id}:{reference_id} dataset"
        )
        supervision_payload = _read_json_object(
            supervision_path, f"{song_id}:{reference_id} supervision"
        )
        if (
            dataset_payload.get("reference_id") != reference_id
            or supervision_payload.get("reference_id") != reference_id
        ):
            raise CapacityFoldLauncherError(
                f"{song_id}:{reference_id} materialized output identity mismatch"
            )
        records[reference_id] = {
            "reference_id": reference_id,
            "dataset": {"path": str(dataset_path), "file_sha256": dataset_sha},
            "supervision": {
                "path": str(supervision_path),
                "file_sha256": supervision_sha,
            },
        }
    if set(records) != set(materialization.reference_ids):
        raise CapacityFoldLauncherError(
            f"{song_id} materialization outputs are incomplete"
        )
    return tuple(records[reference] for reference in sorted(records))


def _feature_snapshot_records(
    manifest_path: Path,
    *,
    song_id: str,
    materialization: legacy._MaterializationBinding,
    feature: legacy._FeatureBinding,
) -> tuple[dict[str, Any], ...]:
    payload = _read_json_object(manifest_path, f"{song_id} feature manifest")
    if payload.get("song_id") != song_id:
        raise CapacityFoldLauncherError(f"{song_id} feature song binding mismatch")
    entries: dict[str, Mapping[str, Any]] = {}
    for raw in _array(payload.get("snapshots"), "feature snapshots"):
        item = _mapping(raw, "feature snapshot")
        reference_id = str(item.get("reference_id") or "").strip()
        if reference_id in entries:
            raise CapacityFoldLauncherError("duplicate feature snapshot reference")
        entries[reference_id] = item
    records: list[dict[str, Any]] = []
    for reference_id in materialization.reference_ids:
        item = entries.get(reference_id)
        if item is None:
            raise CapacityFoldLauncherError(
                f"{song_id}:{reference_id} feature snapshot is missing"
            )
        path = _regular_safe_path(
            str(item.get("path") or ""), f"{song_id}:{reference_id} snapshot"
        )
        if path != feature.snapshot_paths[reference_id]:
            raise CapacityFoldLauncherError(
                f"{song_id}:{reference_id} feature snapshot path differs"
            )
        snapshot_payload = _read_json_object(
            path, f"{song_id}:{reference_id} feature snapshot"
        )
        try:
            snapshot = legacy.StageEditFeatureSnapshot.from_dict(snapshot_payload)
            snapshot.require_production_causal_sensing(
                edit_mode="group",
                target_member=None,
            )
        except (TypeError, ValueError) as exc:
            raise CapacityFoldLauncherError(
                f"{song_id}:{reference_id} feature snapshot is invalid: {exc}"
            ) from exc
        expected_sequence_id = f"stageedit-features:{song_id}:{reference_id}"
        snapshot_sha = str(item.get("snapshot_sha256") or "")
        if (
            not _is_sha256(snapshot_sha)
            or snapshot.snapshot_sha256 != snapshot_sha
            or snapshot.sequence.sequence_id != expected_sequence_id
            or path.name
            != f"{reference_id}-stageedit-feature-snapshot-{snapshot_sha[:16]}.json"
        ):
            raise CapacityFoldLauncherError(
                f"{song_id}:{reference_id} feature snapshot hash/reference/song differs"
            )
        records.append(
            {
                "song_id": song_id,
                "reference_id": reference_id,
                "path": str(path),
                "file_sha256": _stable_file_sha256(
                    path, f"{song_id}:{reference_id} feature snapshot"
                ),
                "snapshot_sha256": snapshot_sha,
                "sequence_id": expected_sequence_id,
            }
        )
    if set(entries) != set(materialization.reference_ids):
        raise CapacityFoldLauncherError(
            f"{song_id} feature snapshot reference set is not exact"
        )
    return tuple(records)


def _scan_dynamic_artifact_payloads(manifest_path: Path, *, song_id: str) -> None:
    payload = _read_json_object(manifest_path, f"{song_id} dynamic crop manifest")
    for raw in _array(payload.get("artifacts"), "dynamic crop artifacts"):
        item = _mapping(raw, "dynamic crop artifact")
        artifact = _mapping(item.get("artifact"), "dynamic crop artifact pointer")
        path = _regular_safe_path(
            str(artifact.get("path") or ""), f"{song_id} dynamic crop artifact"
        )
        _read_json_object(path, f"{song_id} dynamic crop artifact")


def _validate_manifest_set(
    *,
    materialization_manifests: Mapping[str, Path],
    feature_manifests: Mapping[str, Path],
    dynamic_crop_manifests: Mapping[str, Path],
    selection_dataset_manifest_path: Path,
) -> _ValidatedManifestSet:
    _validate_exact_manifest_maps(
        materialization_manifests,
        feature_manifests,
        dynamic_crop_manifests,
    )
    dataset_path = _regular_safe_path(
        selection_dataset_manifest_path, "selection dataset manifest"
    )
    _read_json_object(dataset_path, "selection dataset manifest")
    try:
        dataset = _load_selection_dataset(dataset_path)
    except DynamicCropCapacityProbeError as exc:
        raise CapacityFoldLauncherError(
            f"selection dataset manifest is invalid: {exc}"
        ) from exc

    materializations: dict[str, legacy._MaterializationBinding] = {}
    features: dict[str, legacy._FeatureBinding] = {}
    song_records: list[Mapping[str, Any]] = []
    supervision_paths: dict[str, Mapping[str, Path]] = {}
    snapshot_paths: dict[str, Mapping[str, Path]] = {}
    dynamic_paths: dict[str, Path] = {}

    for song_id in SELECTION_SONG_IDS:
        materialization_path = _regular_safe_path(
            materialization_manifests[song_id],
            f"{song_id} materialization manifest",
        )
        feature_path = _regular_safe_path(
            feature_manifests[song_id], f"{song_id} feature manifest"
        )
        dynamic_path = _regular_safe_path(
            dynamic_crop_manifests[song_id],
            f"{song_id} dynamic crop manifest",
        )
        materialization_pointer = _pointer(
            materialization_path, f"{song_id} materialization manifest"
        )
        feature_pointer = _pointer(feature_path, f"{song_id} feature manifest")
        dynamic_pointer = _pointer(dynamic_path, f"{song_id} dynamic crop manifest")

        # Reject a quarantined spelling in any manifest before the legacy
        # validators follow its nested paths.
        _read_json_object(
            materialization_path, f"{song_id} materialization manifest"
        )
        _read_json_object(feature_path, f"{song_id} feature manifest")
        _read_json_object(dynamic_path, f"{song_id} dynamic crop manifest")
        try:
            _, materialization = legacy._validate_materialization_manifest(
                materialization_pointer, song_id=song_id
            )
            _, feature = legacy._validate_feature_manifest(
                feature_pointer,
                song_id=song_id,
                materialization=materialization,
            )
            legacy._validate_dynamic_crop_manifest(
                dynamic_pointer,
                song_id=song_id,
                materialization=materialization,
            )
        except (legacy.ProductionFoldPreflightError, OSError, ValueError) as exc:
            raise CapacityFoldLauncherError(
                f"{song_id} cross-manifest validation failed: {exc}"
            ) from exc

        expected_dataset_pointer = dict(
            dataset.materialization_manifests_by_song[song_id]
        )
        if (
            materialization_pointer != expected_dataset_pointer
            or materialization.reference_ids
            != dataset.reference_ids_by_song[song_id]
        ):
            raise CapacityFoldLauncherError(
                f"{song_id} manifests differ from the selection dataset"
            )

        output_records = _materialization_output_records(
            materialization_path,
            song_id=song_id,
            materialization=materialization,
        )
        feature_records = _feature_snapshot_records(
            feature_path,
            song_id=song_id,
            materialization=materialization,
            feature=feature,
        )
        _scan_dynamic_artifact_payloads(dynamic_path, song_id=song_id)

        materializations[song_id] = materialization
        features[song_id] = feature
        supervision_paths[song_id] = dict(materialization.supervision_paths)
        snapshot_paths[song_id] = dict(feature.snapshot_paths)
        dynamic_paths[song_id] = dynamic_path
        song_records.append(
            {
                "song_id": song_id,
                "reference_ids": list(materialization.reference_ids),
                "materialization_manifest": materialization_pointer,
                "materialization_outputs": list(output_records),
                "feature_manifest": feature_pointer,
                "feature_snapshots": list(feature_records),
                "dynamic_crop_manifest": dynamic_pointer,
            }
        )

    return _ValidatedManifestSet(
        selection_dataset_pointer=_pointer(
            dataset_path, "selection dataset manifest"
        ),
        selection_dataset_content_sha256=dataset.artifact_content_sha256,
        song_records=tuple(song_records),
        reference_ids_by_song={
            song: materializations[song].reference_ids for song in SELECTION_SONG_IDS
        },
        supervision_paths_by_song=MappingProxyType(supervision_paths),
        snapshot_paths_by_song=MappingProxyType(snapshot_paths),
        dynamic_manifest_paths_by_song=MappingProxyType(dynamic_paths),
    )


def _default_candidate_map() -> dict[str, DynamicCropCapacityCandidate]:
    candidates = default_dynamic_crop_v2_probe_candidates()
    result: dict[str, DynamicCropCapacityCandidate] = {}
    for candidate in candidates:
        if candidate.candidate_name in result or candidate.candidate_id in result:
            raise CapacityFoldLauncherError("default capacity candidates overlap")
        result[candidate.candidate_name] = candidate
        result[candidate.candidate_id] = candidate
    return result


def _resolve_candidates(
    candidate_selectors: Sequence[str] | None,
) -> tuple[DynamicCropCapacityCandidate, ...]:
    defaults = default_dynamic_crop_v2_probe_candidates()
    if not candidate_selectors:
        return defaults
    lookup = _default_candidate_map()
    selected: list[DynamicCropCapacityCandidate] = []
    seen: set[str] = set()
    for selector in candidate_selectors:
        normalized = _reject_quarantined_text(selector, "candidate selector")
        candidate = lookup.get(normalized)
        if candidate is None:
            raise CapacityFoldLauncherError(
                f"unknown default capacity candidate: {normalized}"
            )
        if candidate.candidate_id in seen:
            raise CapacityFoldLauncherError("duplicate capacity candidate")
        seen.add(candidate.candidate_id)
        selected.append(candidate)
    return tuple(selected)


def _json_argument(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _training_schedule_cli_args() -> tuple[str, ...]:
    schedule = PROBE_TRAINING_SCHEDULE
    return (
        "--epochs",
        str(schedule["requested_epochs"]),
        "--patience",
        str(schedule["early_stopping_patience"]),
        "--batch-size",
        str(schedule["batch_size"]),
        "--learning-rate",
        str(schedule["learning_rate"]),
        "--weight-decay",
        str(schedule["weight_decay"]),
        "--gradient-clip-norm",
        str(schedule["gradient_clip_norm"]),
        "--maximum-cut-positive-weight",
        str(MAXIMUM_CUT_POSITIVE_WEIGHT),
        "--cut-pretrain-steps",
        str(schedule["cut_pretraining_steps"]),
        "--cut-pretrain-learning-rate",
        str(schedule["cut_pretraining_peak_learning_rate"]),
        "--cut-pretrain-window-length",
        str(schedule["cut_pretraining_window_length"]),
        "--cut-pretrain-warmup-steps",
        str(schedule["cut_pretraining_warmup_steps"]),
        "--source-pretrain-steps",
        str(schedule["source_pretraining_steps"]),
        "--source-pretrain-learning-rate",
        str(schedule["source_pretraining_peak_learning_rate"]),
        "--source-pretrain-window-length",
        str(schedule["source_pretraining_window_length"]),
        "--source-pretrain-warmup-steps",
        str(schedule["source_pretraining_warmup_steps"]),
        "--finetune-warmup-steps",
        str(schedule["finetune_warmup_steps"]),
        "--minimum-finetune-optimizer-steps",
        str(schedule["minimum_finetune_optimizer_steps"]),
        "--minimum-checkpoint-selection-finetune-steps",
        str(schedule["minimum_checkpoint_selection_finetune_steps"]),
        "--minimum-learning-rate-ratio",
        str(schedule["minimum_learning_rate_ratio"]),
        "--seed",
        str(schedule["seed"]),
        "--device",
        "gpu",
    )


def _build_train_cli_args(
    manifests: _ValidatedManifestSet,
    *,
    candidate: DynamicCropCapacityCandidate,
    held_out_song_id: str,
    checkpoint_directory: Path,
) -> tuple[str, ...]:
    if held_out_song_id not in SELECTION_SONG_IDS:
        raise CapacityFoldLauncherError("held-out song is not a capacity song")
    training_song_ids = tuple(
        sorted(set(SELECTION_SONG_IDS) - {held_out_song_id})
    )
    args: list[str] = []
    for song_id in SELECTION_SONG_IDS:
        references = manifests.reference_ids_by_song[song_id]
        if len(references) != legacy.EXPECTED_REFERENCE_COUNT_PER_SONG:
            raise CapacityFoldLauncherError(
                f"{song_id} reference completeness changed"
            )
        for reference_id in references:
            args.extend(
                (
                    "--input",
                    str(manifests.supervision_paths_by_song[song_id][reference_id]),
                    "--song-id",
                    f"{reference_id}={song_id}",
                    "--feature-snapshot",
                    f"{reference_id}={manifests.snapshot_paths_by_song[song_id][reference_id]}",
                )
            )
        args.extend(
            (
                "--dynamic-crop-manifest",
                str(manifests.dynamic_manifest_paths_by_song[song_id]),
            )
        )
    for song_id in training_song_ids:
        args.extend(("--required-training-song", song_id))
    args.extend(
        (
            "--held-out-song",
            held_out_song_id,
            "--checkpoint-dir",
            str(checkpoint_directory),
            "--negative-samples-per-hold",
            str(legacy.PRODUCTION_SAMPLING_CONTRACT["negative_samples_per_hold"]),
            "--boundary-guard-seconds",
            str(legacy.PRODUCTION_SAMPLING_CONTRACT["boundary_guard_seconds"]),
            "--max-sources",
            str(legacy.PRODUCTION_SAMPLING_CONTRACT["max_sources"]),
        )
    )
    args.extend(_training_schedule_cli_args())
    contract = candidate.contract_dict()
    args.extend(
        (
            "--model-config-json",
            _json_argument(contract["model_config"]),
            "--base-loss-contract-json",
            _json_argument(contract["base_loss_contract"]),
            "--dynamic-crop-loss-contract-json",
            _json_argument(contract["dynamic_crop_loss_contract"]),
        )
    )
    forbidden = {
        "--authorize-production-activation",
        "--production-registry",
        "--professional-montage-grammar-profile",
    }
    if forbidden & set(args):  # pragma: no cover - construction guard
        raise CapacityFoldLauncherError("capacity fold acquired activation authority")
    if args.count("--held-out-song") != 1:
        raise CapacityFoldLauncherError("capacity fold must have one held-out song")
    return tuple(args)


def _run_id(candidate: DynamicCropCapacityCandidate, held_out_song_id: str) -> str:
    song_slug = held_out_song_id.removeprefix("IVE-").lower()
    return f"{candidate.candidate_name}-{candidate.candidate_id[:16]}-heldout-{song_slug}"


def _command_argv(
    train_cli_args: Sequence[str], *, repository_root: Path
) -> list[str]:
    engine_root = repository_root / "engine"
    return [
        "/usr/bin/env",
        f"PYTHONPATH={engine_root}",
        "python3",
        "-m",
        TRAIN_MODULE,
        *train_cli_args,
    ]


def _plan_contract() -> dict[str, Any]:
    return {
        "contract": CAPACITY_FOLD_CONTRACT,
        "artist": "IVE",
        "song_ids": list(SELECTION_SONG_IDS),
        "fold_count_per_candidate": len(SELECTION_SONG_IDS),
        "expected_reference_count_per_song": legacy.EXPECTED_REFERENCE_COUNT_PER_SONG,
        "expected_source_count_per_song": legacy.EXPECTED_SOURCE_COUNT_PER_SONG,
        "selection_dataset_required": True,
        "whole_song_heldout_only": True,
        "heldout_used_for_model_or_threshold_selection": False,
        "synthetic_or_fake_training_data_allowed": False,
        "production_activation_authorized": False,
        "quarantined_song_evidence_opened": False,
        "training_schedule": dict(PROBE_TRAINING_SCHEDULE),
        "maximum_cut_positive_weight": MAXIMUM_CUT_POSITIVE_WEIGHT,
        "sampling": dict(legacy.PRODUCTION_SAMPLING_CONTRACT),
        "recent_history_mask_contract": dict(
            legacy.PRODUCTION_RECENT_HISTORY_MASK_CONTRACT
        ),
        "candidate_policy": "one_or_all_default_dynamic_crop_v2_candidates_v1",
    }


def _atomic_exclusive_write(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = _absolute_safe_path(path.parent, "plan output directory")
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / path.name
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable plan: {final}")
    encoded = _canonical_bytes(payload) + b"\n"
    descriptor = os.open(final, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        directory_descriptor = os.open(destination, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        # The file was created by this call and has never been published as a
        # valid hashed plan when the write/fsync failed.
        try:
            final.unlink()
        except FileNotFoundError:
            pass
        raise
    return final


def create_capacity_fold_plan(
    *,
    materialization_manifests: Mapping[str, Path],
    feature_manifests: Mapping[str, Path],
    dynamic_crop_manifests: Mapping[str, Path],
    selection_dataset_manifest_path: Path,
    checkpoint_root: Path,
    evidence_directory: Path,
    output_directory: Path,
    candidate_selectors: Sequence[str] | None = None,
) -> Path:
    """Validate all inputs and write one exclusive content-addressed plan."""

    checkpoint = _paths_under_external_volume(checkpoint_root, "checkpoint root")
    evidence = _paths_under_external_volume(evidence_directory, "evidence directory")
    output = _absolute_safe_path(output_directory, "plan output directory")
    candidates = _resolve_candidates(candidate_selectors)
    source_provenance = _capture_source_provenance()
    repository_root = Path(source_provenance["repository_root"])
    if output == repository_root or repository_root in output.parents:
        raise CapacityFoldLauncherError(
            "plan output directory must be outside the bound clean repository"
        )
    planned_runs = tuple(
        (
            candidate,
            held_out_song_id,
            checkpoint / _run_id(candidate, held_out_song_id),
        )
        for candidate in candidates
        for held_out_song_id in SELECTION_SONG_IDS
    )
    _validate_artifact_path_isolation(
        plan_path=output,
        plan_label="plan output directory",
        evidence_directory=evidence,
        planned_checkpoint_directories=tuple(
            run_checkpoint for _, _, run_checkpoint in planned_runs
        ),
    )
    for _, _, run_checkpoint in planned_runs:
        if run_checkpoint.exists() or run_checkpoint.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite checkpoint directory: {run_checkpoint}"
            )
    manifests = _validate_manifest_set(
        materialization_manifests=materialization_manifests,
        feature_manifests=feature_manifests,
        dynamic_crop_manifests=dynamic_crop_manifests,
        selection_dataset_manifest_path=selection_dataset_manifest_path,
    )
    runs: list[dict[str, Any]] = []
    checkpoint_paths: set[str] = set()
    for candidate, held_out_song_id, run_checkpoint in planned_runs:
        run_id = _run_id(candidate, held_out_song_id)
        if str(run_checkpoint) in checkpoint_paths:
            raise CapacityFoldLauncherError("capacity checkpoint paths overlap")
        checkpoint_paths.add(str(run_checkpoint))
        args = _build_train_cli_args(
            manifests,
            candidate=candidate,
            held_out_song_id=held_out_song_id,
            checkpoint_directory=run_checkpoint,
        )
        training_song_ids = tuple(
            sorted(set(SELECTION_SONG_IDS) - {held_out_song_id})
        )
        runs.append(
            {
                "run_id": run_id,
                "candidate_name": candidate.candidate_name,
                "candidate_id": candidate.candidate_id,
                "held_out_song_id": held_out_song_id,
                "training_song_ids": list(training_song_ids),
                "checkpoint_directory": str(run_checkpoint),
                "evidence_directory": str(evidence),
                "train_cli_args": list(args),
                "command_argv": _command_argv(
                    args, repository_root=repository_root
                ),
                "production_activation_authorized": False,
            }
        )
    _require_same_source_provenance(source_provenance)
    unsigned = {
        "schema_version": CAPACITY_FOLD_PLAN_SCHEMA_VERSION,
        "analysis_basis": CAPACITY_FOLD_PLAN_ANALYSIS_BASIS,
        "capacity_fold_contract": _plan_contract(),
        "source_provenance": source_provenance,
        "required_volume": str(PRODUCTION_VOLUME_ROOT),
        "checkpoint_root": str(checkpoint),
        "evidence_directory": str(evidence),
        "selection_dataset_manifest": dict(manifests.selection_dataset_pointer),
        "selection_dataset_content_sha256": (
            manifests.selection_dataset_content_sha256
        ),
        "songs": [dict(record) for record in manifests.song_records],
        "candidates": [candidate.to_dict() for candidate in candidates],
        "runs": runs,
        "plan_content_sha256": None,
    }
    digest = _canonical_sha256(unsigned)
    payload = {**unsigned, "plan_content_sha256": digest}
    return _atomic_exclusive_write(
        output / f"{CAPACITY_FOLD_PLAN_PREFIX}{digest}.json", payload
    )


_PLAN_KEYS = {
    "schema_version",
    "analysis_basis",
    "capacity_fold_contract",
    "source_provenance",
    "required_volume",
    "checkpoint_root",
    "evidence_directory",
    "selection_dataset_manifest",
    "selection_dataset_content_sha256",
    "songs",
    "candidates",
    "runs",
    "plan_content_sha256",
}


def _strict_plan(path: Path) -> dict[str, Any]:
    resolved = _regular_safe_path(path, "capacity fold plan")
    payload = _read_json_object(resolved, "capacity fold plan")
    if set(payload) != _PLAN_KEYS:
        raise CapacityFoldLauncherError("capacity fold plan keys are not exact")
    if (
        payload.get("schema_version") != CAPACITY_FOLD_PLAN_SCHEMA_VERSION
        or payload.get("analysis_basis") != CAPACITY_FOLD_PLAN_ANALYSIS_BASIS
        or payload.get("capacity_fold_contract") != _plan_contract()
        or payload.get("required_volume") != str(PRODUCTION_VOLUME_ROOT)
    ):
        raise CapacityFoldLauncherError("capacity fold plan contract mismatch")
    digest = payload.get("plan_content_sha256")
    unsigned = deepcopy(payload)
    unsigned["plan_content_sha256"] = None
    if not _is_sha256(digest) or _canonical_sha256(unsigned) != digest:
        raise CapacityFoldLauncherError("capacity fold plan content hash mismatch")
    if resolved.name != f"{CAPACITY_FOLD_PLAN_PREFIX}{digest}.json":
        raise CapacityFoldLauncherError(
            "capacity fold plan filename is not content-addressed"
        )
    _validate_source_provenance(payload.get("source_provenance"))
    return payload


def _candidates_from_plan(
    value: object,
) -> tuple[DynamicCropCapacityCandidate, ...]:
    candidates = tuple(
        DynamicCropCapacityCandidate.from_dict(item)
        for item in _array(value, "capacity candidates")
    )
    if not candidates:
        raise CapacityFoldLauncherError("capacity plan has no candidates")
    defaults = {
        item.candidate_id: item for item in default_dynamic_crop_v2_probe_candidates()
    }
    seen: set[str] = set()
    for candidate in candidates:
        if (
            candidate.candidate_id not in defaults
            or candidate.to_dict() != defaults[candidate.candidate_id].to_dict()
            or candidate.candidate_id in seen
        ):
            raise CapacityFoldLauncherError(
                "capacity plan candidate is not one unique exact default"
            )
        seen.add(candidate.candidate_id)
    return candidates


def _manifest_maps_from_plan(
    songs_value: object,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    songs: dict[str, Mapping[str, Any]] = {}
    for raw in _array(songs_value, "capacity songs"):
        item = _mapping(raw, "capacity song")
        song_id = str(item.get("song_id") or "")
        if song_id in songs:
            raise CapacityFoldLauncherError("duplicate capacity song")
        songs[song_id] = item
    if set(songs) != set(SELECTION_SONG_IDS):
        raise CapacityFoldLauncherError("capacity plan songs are not exact")
    materialization: dict[str, Path] = {}
    feature: dict[str, Path] = {}
    dynamic: dict[str, Path] = {}
    for song_id in SELECTION_SONG_IDS:
        item = songs[song_id]
        materialization[song_id] = _verify_pointer(
            item.get("materialization_manifest"),
            f"{song_id} materialization manifest",
        )
        feature[song_id] = _verify_pointer(
            item.get("feature_manifest"), f"{song_id} feature manifest"
        )
        dynamic[song_id] = _verify_pointer(
            item.get("dynamic_crop_manifest"),
            f"{song_id} dynamic crop manifest",
        )
    return materialization, feature, dynamic


def _select_candidate(
    candidates: Sequence[DynamicCropCapacityCandidate], selector: str
) -> DynamicCropCapacityCandidate:
    normalized = _reject_quarantined_text(selector, "candidate selector")
    matches = tuple(
        candidate
        for candidate in candidates
        if normalized in {candidate.candidate_name, candidate.candidate_id}
    )
    if len(matches) != 1:
        raise CapacityFoldLauncherError(
            "candidate selector does not identify one planned candidate"
        )
    return matches[0]


def preflight_capacity_fold(
    plan_path: Path,
    *,
    candidate_selector: str,
    held_out_song_id: str,
) -> CapacityFoldPreflight:
    """Replay one immutable candidate/fold and expand its exact command."""

    if held_out_song_id not in SELECTION_SONG_IDS:
        raise CapacityFoldLauncherError("held-out song is not one fixed capacity song")
    resolved_plan = _regular_safe_path(plan_path, "capacity fold plan")
    plan = _strict_plan(resolved_plan)
    checkpoint_root = _paths_under_external_volume(
        str(plan.get("checkpoint_root") or ""), "checkpoint root"
    )
    evidence_directory = _paths_under_external_volume(
        str(plan.get("evidence_directory") or ""), "evidence directory"
    )
    dataset_path = _verify_pointer(
        plan.get("selection_dataset_manifest"), "selection dataset manifest"
    )
    materialization, feature, dynamic = _manifest_maps_from_plan(plan.get("songs"))
    manifests = _validate_manifest_set(
        materialization_manifests=materialization,
        feature_manifests=feature,
        dynamic_crop_manifests=dynamic,
        selection_dataset_manifest_path=dataset_path,
    )
    if (
        dict(manifests.selection_dataset_pointer)
        != plan.get("selection_dataset_manifest")
        or manifests.selection_dataset_content_sha256
        != plan.get("selection_dataset_content_sha256")
        or [dict(record) for record in manifests.song_records] != plan.get("songs")
    ):
        raise CapacityFoldLauncherError(
            "current manifest graph differs from immutable capacity plan"
        )
    candidates = _candidates_from_plan(plan.get("candidates"))
    candidate = _select_candidate(candidates, candidate_selector)
    planned_checkpoint_directories = tuple(
        checkpoint_root / _run_id(planned_candidate, planned_heldout)
        for planned_candidate in candidates
        for planned_heldout in SELECTION_SONG_IDS
    )
    _validate_artifact_path_isolation(
        plan_path=resolved_plan,
        plan_label="capacity fold plan",
        evidence_directory=evidence_directory,
        planned_checkpoint_directories=planned_checkpoint_directories,
    )
    expected_run_count = len(candidates) * len(SELECTION_SONG_IDS)
    runs = _array(plan.get("runs"), "capacity runs")
    if len(runs) != expected_run_count:
        raise CapacityFoldLauncherError("capacity plan run matrix is incomplete")
    run_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw in runs:
        run = _mapping(raw, "capacity run")
        key = (
            str(run.get("candidate_id") or ""),
            str(run.get("held_out_song_id") or ""),
        )
        if key in run_by_key:
            raise CapacityFoldLauncherError("duplicate capacity run")
        run_by_key[key] = run
    expected_keys = {
        (item.candidate_id, song)
        for item in candidates
        for song in SELECTION_SONG_IDS
    }
    if set(run_by_key) != expected_keys:
        raise CapacityFoldLauncherError("capacity run matrix identities differ")
    repository_root = Path(
        str(_mapping(plan["source_provenance"], "source provenance")["repository_root"])
    )
    expected_runs: list[dict[str, Any]] = []
    replayed_args: dict[tuple[str, str], tuple[str, ...]] = {}
    for planned_candidate in candidates:
        for planned_heldout in SELECTION_SONG_IDS:
            planned_checkpoint = checkpoint_root / _run_id(
                planned_candidate, planned_heldout
            )
            planned_args = _build_train_cli_args(
                manifests,
                candidate=planned_candidate,
                held_out_song_id=planned_heldout,
                checkpoint_directory=planned_checkpoint,
            )
            planned_training = tuple(
                sorted(set(SELECTION_SONG_IDS) - {planned_heldout})
            )
            replayed_args[(planned_candidate.candidate_id, planned_heldout)] = (
                planned_args
            )
            expected_runs.append(
                {
                    "run_id": _run_id(planned_candidate, planned_heldout),
                    "candidate_name": planned_candidate.candidate_name,
                    "candidate_id": planned_candidate.candidate_id,
                    "held_out_song_id": planned_heldout,
                    "training_song_ids": list(planned_training),
                    "checkpoint_directory": str(planned_checkpoint),
                    "evidence_directory": str(evidence_directory),
                    "train_cli_args": list(planned_args),
                    "command_argv": _command_argv(
                        planned_args, repository_root=repository_root
                    ),
                    "production_activation_authorized": False,
                }
            )
    if runs != expected_runs:
        raise CapacityFoldLauncherError("capacity run commands differ from replay")
    run = run_by_key[(candidate.candidate_id, held_out_song_id)]
    checkpoint = checkpoint_root / _run_id(candidate, held_out_song_id)
    args = replayed_args[(candidate.candidate_id, held_out_song_id)]
    training_song_ids = tuple(
        sorted(set(SELECTION_SONG_IDS) - {held_out_song_id})
    )
    _require_same_source_provenance(plan["source_provenance"])
    free_bytes = legacy._validate_storage(checkpoint)
    return CapacityFoldPreflight(
        plan_path=resolved_plan,
        plan_content_sha256=str(plan["plan_content_sha256"]),
        candidate_name=candidate.candidate_name,
        candidate_id=candidate.candidate_id,
        held_out_song_id=held_out_song_id,
        training_song_ids=training_song_ids,
        checkpoint_directory=checkpoint,
        evidence_directory=evidence_directory,
        selection_dataset_manifest_path=dataset_path,
        reference_ids_by_song=dict(manifests.reference_ids_by_song),
        train_cli_args=args,
        command_argv=tuple(run["command_argv"]),
        free_bytes=free_bytes,
    )


def record_capacity_fold_evidence(preflight: CapacityFoldPreflight) -> Path:
    """Write selector evidence for one successful inactive real fold."""

    checkpoint = preflight.checkpoint_directory
    report = checkpoint / PRODUCTION_TRAINING_REPORT_FILENAME
    freeze = checkpoint / STAGE_EDIT_SELECTION_FREEZE_FILENAME
    metadata = checkpoint / DYNAMIC_CROP_V2_CHECKPOINT_METADATA_FILENAME
    weights = checkpoint / DYNAMIC_CROP_V2_CHECKPOINT_WEIGHT_FILENAME
    for path, label in (
        (report, "production training report"),
        (freeze, "selection freeze"),
        (metadata, "Dynamic Crop v2 checkpoint metadata"),
        (weights, "Dynamic Crop v2 checkpoint weights"),
    ):
        _stable_file_sha256(path, label)
    return write_fold_run_evidence(
        preflight.evidence_directory,
        candidate_id=preflight.candidate_id,
        validation_song_id=preflight.held_out_song_id,
        dataset_manifest_path=preflight.selection_dataset_manifest_path,
        production_training_report_path=report,
        selection_freeze_path=freeze,
        checkpoint_metadata_path=metadata,
        checkpoint_weights_path=weights,
    )


def launch_capacity_fold(
    plan_path: Path,
    *,
    candidate_selector: str,
    held_out_song_id: str,
) -> tuple[int, CapacityFoldPreflight, Path | None]:
    """Launch exactly one fold; never overwrite and never activate."""

    preflight = preflight_capacity_fold(
        plan_path,
        candidate_selector=candidate_selector,
        held_out_song_id=held_out_song_id,
    )
    plan = _strict_plan(preflight.plan_path)
    _require_same_source_provenance(plan["source_provenance"])
    exit_code = production_train_main(preflight.train_cli_args)
    if exit_code != 0:
        return exit_code, preflight, None
    # Rebind the clean interpreter after training and before publishing fold
    # evidence.  Training outputs live on the external volume and must not have
    # changed the repository that authorized execution.
    plan = _strict_plan(preflight.plan_path)
    _validate_source_provenance(plan["source_provenance"])
    evidence = record_capacity_fold_evidence(preflight)
    return 0, preflight, evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create, preflight, or launch strict three-song Dynamic Crop v2 "
            "capacity folds"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--materialization-manifest", action="append", required=True)
    create.add_argument("--feature-manifest", action="append", required=True)
    create.add_argument("--dynamic-crop-manifest", action="append", required=True)
    create.add_argument("--selection-dataset", required=True, type=Path)
    create.add_argument("--checkpoint-root", required=True, type=Path)
    create.add_argument("--evidence-directory", required=True, type=Path)
    create.add_argument("--output-directory", required=True, type=Path)
    create.add_argument(
        "--candidate",
        action="append",
        help="Default candidate name or full candidate id; omit for all defaults",
    )
    for name in ("preflight", "launch"):
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True, type=Path)
        command.add_argument("--candidate", required=True)
        command.add_argument(
            "--held-out-song", required=True, choices=tuple(SELECTION_SONG_IDS)
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "create":
        plan = create_capacity_fold_plan(
            materialization_manifests=_keyed_paths(
                args.materialization_manifest, "--materialization-manifest"
            ),
            feature_manifests=_keyed_paths(
                args.feature_manifest, "--feature-manifest"
            ),
            dynamic_crop_manifests=_keyed_paths(
                args.dynamic_crop_manifest, "--dynamic-crop-manifest"
            ),
            selection_dataset_manifest_path=args.selection_dataset,
            checkpoint_root=args.checkpoint_root,
            evidence_directory=args.evidence_directory,
            output_directory=args.output_directory,
            candidate_selectors=args.candidate,
        )
        print(json.dumps({"plan_path": str(plan)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "preflight":
        result = preflight_capacity_fold(
            args.plan,
            candidate_selector=args.candidate,
            held_out_song_id=args.held_out_song,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0
    exit_code, result, evidence = launch_capacity_fold(
        args.plan,
        candidate_selector=args.candidate,
        held_out_song_id=args.held_out_song,
    )
    print(
        json.dumps(
            {
                "exit_code": exit_code,
                "preflight": result.to_dict(),
                "fold_evidence_path": str(evidence) if evidence else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return exit_code


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "CAPACITY_FOLD_PLAN_ANALYSIS_BASIS",
    "CAPACITY_FOLD_PLAN_SCHEMA_VERSION",
    "CAPACITY_FOLD_CONTRACT",
    "CapacityFoldLauncherError",
    "CapacityFoldPreflight",
    "SOURCE_RELATIVE_PATHS",
    "create_capacity_fold_plan",
    "launch_capacity_fold",
    "preflight_capacity_fold",
    "record_capacity_fold_evidence",
]
