"""Receipt-bound four-song StageEdit Dynamic Crop v3 LOSO execution.

This module deliberately keeps model selection separate from production
activation.  It binds four complete IVE schema4 corpora, materialized human
references, causal feature snapshots, and Dynamic Crop v3 supervision into
four whole-song held-out folds.  Every fold is an inactive evaluation run;
no command in this module can write the production registry.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

from . import stage_edit_production_fold_launcher as legacy
from .stage_edit_dynamic_crop_head_v3 import DynamicCropV3LossContract
from .stage_edit_production_train_cli import main as production_train_main
from .stage_edit_production_training import STAGE_EDIT_SELECTION_FREEZE_FILENAME
from .stage_edit_transformer import (
    PRODUCTION_TRAINING_REPORT_FILENAME,
    StageEditLossContract,
    production_stage_edit_config,
)


FOUR_SONG_LOSO_REQUEST_SCHEMA_VERSION = 1
FOUR_SONG_LOSO_REQUEST_CONTRACT = "ive_stage_edit_four_song_loso_request_v1"
FOUR_SONG_LOSO_PLAN_SCHEMA_VERSION = 1
FOUR_SONG_LOSO_PLAN_CONTRACT = (
    "ive_stage_edit_schema4_dynamic_crop_v3_four_song_loso_plan_v1"
)
FOUR_SONG_LOSO_PLAN_PREFIX = "stageedit-four-song-loso-plan-"
FOUR_SONG_LOSO_RECEIPT_SCHEMA_VERSION = 1
FOUR_SONG_LOSO_RECEIPT_CONTRACT = (
    "ive_stage_edit_dynamic_crop_v3_inactive_loso_fold_receipt_v1"
)
FOUR_SONG_LOSO_RECEIPT_PREFIX = "stageedit-four-song-loso-fold-"
SCHEMA4_RECEIPT_CONTRACT = "ive_stage_edit_schema4_training_corpus_launch_receipt_v1"
PRODUCTION_VOLUME_ROOT = Path("/Volumes/CrossCutMedia")
MINIMUM_FREE_BYTES = 8 * 1024**3

BANG_BANG = "IVE-BANG-BANG"
BLACKHOLE = "IVE-BLACKHOLE"
REBEL_HEART = "IVE-REBEL-HEART"
XOXZ = "IVE-XOXZ"
LOSO_SONG_IDS = (BANG_BANG, BLACKHOLE, REBEL_HEART, XOXZ)
FEATURE_SONG_ID_BY_TRAINING_SONG: Mapping[str, str] = {
    BANG_BANG: "BANG BANG",
    BLACKHOLE: "BLACKHOLE",
    REBEL_HEART: "REBEL HEART",
    XOXZ: "XOXZ",
}

SOURCE_RELATIVE_PATHS = (
    "engine/crosscut/stage_edit_four_song_loso.py",
    "engine/crosscut/stage_edit_production_fold_launcher.py",
    "engine/crosscut/stage_edit_production_train_cli.py",
    "engine/crosscut/stage_edit_production_training.py",
    "engine/crosscut/stage_edit_transformer.py",
    "engine/crosscut/stage_edit_reference_adapter.py",
    "engine/crosscut/stage_edit_feature_materializer.py",
    "engine/crosscut/stage_edit_dynamic_crop_training_v3.py",
    "engine/crosscut/stage_edit_dynamic_crop_head_v3.py",
    "engine/crosscut/reference_dynamic_crop_job.py",
    "engine/crosscut/reference_dynamic_crop_supervision.py",
)

TRAINING_SCHEDULE: Mapping[str, int | float | str] = {
    "epochs": 30,
    "patience": 6,
    "batch_size": 2,
    "learning_rate": 1.0e-4,
    "weight_decay": 1.0e-3,
    "gradient_clip_norm": 1.0,
    "maximum_cut_positive_weight": 8.0,
    "cut_pretrain_steps": 500,
    "cut_pretrain_learning_rate": 3.0e-4,
    "cut_pretrain_window_length": 32,
    "cut_pretrain_warmup_steps": 50,
    "source_pretrain_steps": 500,
    "source_pretrain_learning_rate": 3.0e-4,
    "source_pretrain_window_length": 32,
    "source_pretrain_warmup_steps": 50,
    "dynamic_crop_v3_prior_pretrain_steps": 500,
    "dynamic_crop_v3_prior_pretrain_learning_rate": 3.0e-4,
    "dynamic_crop_v3_prior_pretrain_warmup_steps": 50,
    "finetune_warmup_steps": 8,
    "minimum_finetune_optimizer_steps": 120,
    "minimum_checkpoint_selection_finetune_steps": 60,
    "minimum_learning_rate_ratio": 0.10,
    "seed": 20260824,
    "negative_samples_per_hold": 2,
    "boundary_guard_seconds": 0.18,
    "max_sources": 32,
    "device": "gpu",
}


class FourSongLOSOError(ValueError):
    """Raised before training when any immutable LOSO input is unsafe."""


@dataclass(frozen=True, slots=True)
class FourSongFoldPreflight:
    plan_path: Path
    plan_content_sha256: str
    held_out_song_id: str
    training_song_ids: tuple[str, ...]
    checkpoint_directory: Path
    receipt_directory: Path
    train_cli_args: tuple[str, ...]
    free_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_path": str(self.plan_path),
            "plan_content_sha256": self.plan_content_sha256,
            "held_out_song_id": self.held_out_song_id,
            "training_song_ids": list(self.training_song_ids),
            "checkpoint_directory": str(self.checkpoint_directory),
            "receipt_directory": str(self.receipt_directory),
            "free_bytes": self.free_bytes,
            "train_cli_args": list(self.train_cli_args),
            "whole_song_heldout_only": True,
            "heldout_used_for_model_or_threshold_selection": False,
            "production_activation_authorized": False,
            "production_registry_activation_requested": False,
        }


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FourSongLOSOError(f"{label} must be an object")
    return dict(value)


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise FourSongLOSOError(f"{label} must be an array")
    return value


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
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _regular_safe_path(value: Path | str, label: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise FourSongLOSOError(f"{label} path must be absolute")
    if candidate.is_symlink() or not candidate.is_file():
        raise FourSongLOSOError(f"{label} must be a regular non-symlink file")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise FourSongLOSOError(f"{label} is unavailable: {exc}") from exc
    if resolved != candidate:
        raise FourSongLOSOError(f"{label} path traverses a symbolic link")
    return resolved


def _stable_file_sha256(value: Path | str, label: str) -> str:
    path = _regular_safe_path(value, label)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
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
        raise FourSongLOSOError(f"{label} changed while hashing")
    return digest.hexdigest()


def _pointer(value: Path | str, label: str) -> dict[str, str]:
    path = _regular_safe_path(value, label)
    return {
        "path": str(path),
        "file_sha256": _stable_file_sha256(path, label),
    }


def _verify_pointer(value: object, label: str) -> Path:
    pointer = _mapping(value, label)
    if set(pointer) != {"path", "file_sha256"}:
        raise FourSongLOSOError(f"{label} pointer keys are not exact")
    path = _regular_safe_path(str(pointer.get("path") or ""), label)
    expected = pointer.get("file_sha256")
    if not _is_sha256(expected):
        raise FourSongLOSOError(f"{label} hash is invalid")
    if _stable_file_sha256(path, label) != expected:
        raise FourSongLOSOError(f"{label} hash mismatch")
    return path


def _read_json(path: Path | str, label: str) -> dict[str, Any]:
    resolved = _regular_safe_path(path, label)
    before = resolved.stat()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FourSongLOSOError(f"{label} is unreadable: {exc}") from exc
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
        raise FourSongLOSOError(f"{label} changed while reading")
    return _mapping(payload, label)


def _external_directory(value: Path | str, label: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise FourSongLOSOError(f"{label} must be absolute")
    resolved_parent = candidate.parent.resolve(strict=True)
    resolved = resolved_parent / candidate.name
    volume = PRODUCTION_VOLUME_ROOT.resolve(strict=True)
    if resolved != volume and volume not in resolved.parents:
        raise FourSongLOSOError(f"{label} must be under {volume}")
    if resolved.is_symlink():
        raise FourSongLOSOError(f"{label} cannot be a symbolic link")
    return resolved


def _atomic_content_addressed_write(
    directory: Path,
    *,
    prefix: str,
    unsigned_payload: Mapping[str, Any],
) -> Path:
    output = _external_directory(directory, "artifact output directory")
    output.mkdir(parents=True, exist_ok=True)
    unsigned = dict(unsigned_payload)
    unsigned["artifact_content_sha256"] = None
    digest = _canonical_sha256(unsigned)
    payload = {**unsigned, "artifact_content_sha256": digest}
    destination = output / f"{prefix}{digest}.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError:
        existing = _read_json(destination, "existing content-addressed artifact")
        if existing != payload:
            raise FourSongLOSOError(
                "content-addressed destination exists with different bytes"
            )
        return destination
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            destination.unlink()
        except OSError:
            pass
        raise
    return destination


def _git_output(repository_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ("git", "-C", str(repository_root), *args),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FourSongLOSOError(
            f"Git source provenance command failed: {' '.join(args)}"
        ) from exc


def _capture_source_provenance() -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[2]
    head = _git_output(repository_root, "rev-parse", "HEAD")
    if len(head) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in head
    ):
        raise FourSongLOSOError("Git HEAD identity is invalid")
    files: dict[str, dict[str, str]] = {}
    for relative in SOURCE_RELATIVE_PATHS:
        _git_output(repository_root, "ls-files", "--error-unmatch", "--", relative)
        committed_blob = _git_output(repository_root, "rev-parse", f"HEAD:{relative}")
        worktree_blob = _git_output(repository_root, "hash-object", "--", relative)
        if committed_blob != worktree_blob:
            raise FourSongLOSOError(
                f"LOSO source file differs from committed HEAD: {relative}"
            )
        files[relative] = {
            "git_blob_oid": committed_blob,
            "file_sha256": _stable_file_sha256(
                repository_root / relative,
                f"LOSO source {relative}",
            ),
        }
    return {
        "contract": "exact_committed_loso_source_blobs_v1",
        "repository_root": str(repository_root),
        "head_commit_oid": head,
        "source_files": files,
    }


def _verify_source_provenance(value: object) -> dict[str, Any]:
    expected = _mapping(value, "source provenance")
    if set(expected) != {
        "contract",
        "repository_root",
        "head_commit_oid",
        "source_files",
    } or expected.get("contract") != "exact_committed_loso_source_blobs_v1":
        raise FourSongLOSOError("source provenance contract is invalid")
    observed = _capture_source_provenance()
    if observed["repository_root"] != expected.get("repository_root"):
        raise FourSongLOSOError("source provenance repository differs")
    if observed["source_files"] != expected.get("source_files"):
        raise FourSongLOSOError("LOSO source blobs changed after plan creation")
    return expected


def _strict_schema4_receipt(
    path: Path,
    *,
    training_song_id: str,
    feature_song_id: str,
    feature_manifest_path: Path,
    feature_manifest_file_sha256: str,
    reference_ids: Sequence[str],
) -> dict[str, Any]:
    payload = _read_json(path, f"{training_song_id} schema4 receipt")
    scope = _mapping(payload.get("scope"), "schema4 receipt scope")
    native = _mapping(
        payload.get("native_cut_execution"), "schema4 native-cut execution"
    )
    causal = _mapping(
        payload.get("causal_sensing_execution"),
        "schema4 causal-sensing execution",
    )
    feature = _mapping(
        payload.get("feature_job_execution"), "schema4 feature execution"
    )
    feature_job = _mapping(payload.get("feature_job"), "schema4 feature job")
    if (
        payload.get("schema_version") != 1
        or payload.get("contract") != SCHEMA4_RECEIPT_CONTRACT
        or scope
        != {
            "artist": "IVE",
            "edit_mode": "group",
            "song_id": feature_song_id,
            "target_member": None,
        }
        or payload.get("source_count") != 10
        or payload.get("reference_count") != 5
        or not isinstance(payload.get("decision_time_count"), int)
        or isinstance(payload.get("decision_time_count"), bool)
        or payload["decision_time_count"] <= 0
        or native.get("contract") != "strict_native_cut_manifest_resume_v1"
        or native.get("existing_manifest_strictly_reloaded") is not True
        or native.get("detector_invoked") is not False
        or native.get("all_source_records_strictly_validated") is not True
        or native.get("persisted_manifest_strictly_replayed") is not True
        or causal.get("contract") != "strict_causal_sensing_triple_resume_v1"
        or causal.get("existing_triple_strictly_reloaded") is not True
        or causal.get("media_materializer_invoked") is not False
        or causal.get("all_artifacts_bound_to_exact_request") is not True
        or feature.get("contract")
        != "strict_schema4_feature_manifest_resume_v1"
        or feature.get("existing_manifest_strictly_reloaded") is not True
        or feature.get("feature_builder_invoked") is not False
        or feature.get("all_snapshots_strictly_replayed") is not True
        or payload.get("same_exact_triple_bound_to_all_five_snapshots") is not True
        or payload.get("raw_frames_persisted_by_coordinator") is not False
    ):
        raise FourSongLOSOError(
            f"{training_song_id} schema4 receipt is not a strict replay"
        )
    if (
        Path(str(feature_job.get("manifest_path") or ""))
        != feature_manifest_path
        or feature_job.get("manifest_file_sha256")
        != feature_manifest_file_sha256
    ):
        raise FourSongLOSOError(
            f"{training_song_id} schema4 receipt feature manifest differs"
        )
    receipt_snapshots = _array(
        feature_job.get("snapshots"), "schema4 receipt snapshots"
    )
    if tuple(
        sorted(
            str(_mapping(item, "schema4 receipt snapshot").get("reference_id") or "")
            for item in receipt_snapshots
        )
    ) != tuple(sorted(reference_ids)):
        raise FourSongLOSOError(
            f"{training_song_id} schema4 receipt reference set differs"
        )
    return payload


def _validate_song_artifacts(raw: object) -> dict[str, Any]:
    record = _mapping(raw, "LOSO song artifact record")
    expected_keys = {
        "training_song_id",
        "feature_song_id",
        "materialization_manifest",
        "feature_manifest",
        "dynamic_crop_manifest",
        "schema4_receipt",
    }
    if set(record) != expected_keys:
        raise FourSongLOSOError("LOSO song artifact record keys are not exact")
    training_song_id = str(record.get("training_song_id") or "")
    feature_song_id = str(record.get("feature_song_id") or "")
    if (
        training_song_id not in LOSO_SONG_IDS
        or feature_song_id
        != FEATURE_SONG_ID_BY_TRAINING_SONG[training_song_id]
    ):
        raise FourSongLOSOError("LOSO training/feature song identity differs")

    materialization_path = _regular_safe_path(
        str(record["materialization_manifest"]),
        f"{training_song_id} materialization manifest",
    )
    feature_path = _regular_safe_path(
        str(record["feature_manifest"]),
        f"{training_song_id} feature manifest",
    )
    dynamic_path = _regular_safe_path(
        str(record["dynamic_crop_manifest"]),
        f"{training_song_id} dynamic crop manifest",
    )
    receipt_path = _regular_safe_path(
        str(record["schema4_receipt"]),
        f"{training_song_id} schema4 receipt",
    )
    materialization_pointer = _pointer(
        materialization_path, f"{training_song_id} materialization manifest"
    )
    feature_pointer = _pointer(feature_path, f"{training_song_id} feature manifest")
    dynamic_pointer = _pointer(dynamic_path, f"{training_song_id} dynamic crop manifest")
    receipt_pointer = _pointer(receipt_path, f"{training_song_id} schema4 receipt")

    try:
        _, materialization = legacy._validate_materialization_manifest(
            materialization_pointer,
            song_id=training_song_id,
        )
        _, features = legacy._validate_feature_manifest(
            feature_pointer,
            song_id=feature_song_id,
            materialization=materialization,
        )
        legacy._validate_dynamic_crop_manifest(
            dynamic_pointer,
            song_id=training_song_id,
            materialization=materialization,
        )
    except (legacy.ProductionFoldPreflightError, OSError, ValueError) as exc:
        raise FourSongLOSOError(
            f"{training_song_id} cross-manifest validation failed: {exc}"
        ) from exc

    _strict_schema4_receipt(
        receipt_path,
        training_song_id=training_song_id,
        feature_song_id=feature_song_id,
        feature_manifest_path=feature_path,
        feature_manifest_file_sha256=feature_pointer["file_sha256"],
        reference_ids=materialization.reference_ids,
    )
    return {
        "training_song_id": training_song_id,
        "feature_song_id": feature_song_id,
        "reference_ids": list(materialization.reference_ids),
        "materialization_manifest": materialization_pointer,
        "feature_manifest": feature_pointer,
        "dynamic_crop_manifest": dynamic_pointer,
        "schema4_receipt": receipt_pointer,
        "supervision_paths_by_reference": {
            reference_id: str(materialization.supervision_paths[reference_id])
            for reference_id in materialization.reference_ids
        },
        "feature_snapshot_paths_by_reference": {
            reference_id: str(features.snapshot_paths[reference_id])
            for reference_id in materialization.reference_ids
        },
    }


def _validate_planned_song(raw: object) -> dict[str, Any]:
    """Replay every external pointer embedded in one immutable plan song."""

    record = _mapping(raw, "planned LOSO song")
    expected_keys = {
        "training_song_id",
        "feature_song_id",
        "reference_ids",
        "materialization_manifest",
        "feature_manifest",
        "dynamic_crop_manifest",
        "schema4_receipt",
        "supervision_paths_by_reference",
        "feature_snapshot_paths_by_reference",
    }
    if set(record) != expected_keys:
        raise FourSongLOSOError("planned LOSO song keys are not exact")
    training_song_id = str(record.get("training_song_id") or "")
    feature_song_id = str(record.get("feature_song_id") or "")
    if (
        training_song_id not in LOSO_SONG_IDS
        or feature_song_id
        != FEATURE_SONG_ID_BY_TRAINING_SONG[training_song_id]
    ):
        raise FourSongLOSOError("planned LOSO song identity differs")
    materialization_path = _verify_pointer(
        record["materialization_manifest"],
        f"{training_song_id} materialization manifest",
    )
    feature_path = _verify_pointer(
        record["feature_manifest"], f"{training_song_id} feature manifest"
    )
    dynamic_path = _verify_pointer(
        record["dynamic_crop_manifest"],
        f"{training_song_id} dynamic crop manifest",
    )
    receipt_path = _verify_pointer(
        record["schema4_receipt"], f"{training_song_id} schema4 receipt"
    )
    try:
        _, materialization = legacy._validate_materialization_manifest(
            record["materialization_manifest"],
            song_id=training_song_id,
        )
        _, features = legacy._validate_feature_manifest(
            record["feature_manifest"],
            song_id=feature_song_id,
            materialization=materialization,
        )
        legacy._validate_dynamic_crop_manifest(
            record["dynamic_crop_manifest"],
            song_id=training_song_id,
            materialization=materialization,
        )
    except (legacy.ProductionFoldPreflightError, OSError, ValueError) as exc:
        raise FourSongLOSOError(
            f"{training_song_id} planned cross-manifest replay failed: {exc}"
        ) from exc
    references = tuple(str(value) for value in _array(
        record["reference_ids"], "planned reference ids"
    ))
    expected_supervision = {
        reference_id: str(materialization.supervision_paths[reference_id])
        for reference_id in materialization.reference_ids
    }
    expected_snapshots = {
        reference_id: str(features.snapshot_paths[reference_id])
        for reference_id in materialization.reference_ids
    }
    if (
        references != materialization.reference_ids
        or _mapping(
            record["supervision_paths_by_reference"],
            "planned supervision paths",
        )
        != expected_supervision
        or _mapping(
            record["feature_snapshot_paths_by_reference"],
            "planned feature snapshot paths",
        )
        != expected_snapshots
    ):
        raise FourSongLOSOError(
            f"{training_song_id} planned derived paths differ from manifests"
        )
    feature_pointer = _mapping(record["feature_manifest"], "feature pointer")
    _strict_schema4_receipt(
        receipt_path,
        training_song_id=training_song_id,
        feature_song_id=feature_song_id,
        feature_manifest_path=feature_path,
        feature_manifest_file_sha256=str(feature_pointer["file_sha256"]),
        reference_ids=references,
    )
    # These locals are intentionally evaluated above even though the returned
    # record retains the immutable pointer spellings.
    del materialization_path, dynamic_path
    return record


def _json_argument(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _build_train_cli_args(
    songs: Sequence[Mapping[str, Any]],
    *,
    held_out_song_id: str,
    checkpoint_directory: Path,
    model_config: Mapping[str, Any],
    base_loss_contract: Mapping[str, Any],
    dynamic_crop_loss_contract: Mapping[str, Any],
) -> tuple[str, ...]:
    args: list[str] = []
    for song in songs:
        training_song_id = str(song["training_song_id"])
        references = tuple(str(value) for value in song["reference_ids"])
        supervision = _mapping(
            song["supervision_paths_by_reference"], "song supervision paths"
        )
        snapshots = _mapping(
            song["feature_snapshot_paths_by_reference"], "song snapshot paths"
        )
        for reference_id in references:
            args.extend(("--input", str(supervision[reference_id])))
            args.extend(("--song-id", f"{reference_id}={training_song_id}"))
            args.extend(
                ("--feature-snapshot", f"{reference_id}={snapshots[reference_id]}")
            )
    args.extend(("--held-out-song", held_out_song_id))
    for song_id in LOSO_SONG_IDS:
        if song_id != held_out_song_id:
            args.extend(("--required-training-song", song_id))
    for song in songs:
        dynamic_pointer = _mapping(
            song["dynamic_crop_manifest"], "dynamic crop manifest pointer"
        )
        args.extend(("--dynamic-crop-manifest", str(dynamic_pointer["path"])))
    args.extend(("--dynamic-crop-version", "3"))
    args.extend(("--checkpoint-dir", str(checkpoint_directory)))
    args.extend(("--edit-mode", "group"))
    option_names = {
        "epochs": "--epochs",
        "patience": "--patience",
        "batch_size": "--batch-size",
        "learning_rate": "--learning-rate",
        "weight_decay": "--weight-decay",
        "gradient_clip_norm": "--gradient-clip-norm",
        "maximum_cut_positive_weight": "--maximum-cut-positive-weight",
        "cut_pretrain_steps": "--cut-pretrain-steps",
        "cut_pretrain_learning_rate": "--cut-pretrain-learning-rate",
        "cut_pretrain_window_length": "--cut-pretrain-window-length",
        "cut_pretrain_warmup_steps": "--cut-pretrain-warmup-steps",
        "source_pretrain_steps": "--source-pretrain-steps",
        "source_pretrain_learning_rate": "--source-pretrain-learning-rate",
        "source_pretrain_window_length": "--source-pretrain-window-length",
        "source_pretrain_warmup_steps": "--source-pretrain-warmup-steps",
        "dynamic_crop_v3_prior_pretrain_steps": (
            "--dynamic-crop-v3-prior-pretrain-steps"
        ),
        "dynamic_crop_v3_prior_pretrain_learning_rate": (
            "--dynamic-crop-v3-prior-pretrain-learning-rate"
        ),
        "dynamic_crop_v3_prior_pretrain_warmup_steps": (
            "--dynamic-crop-v3-prior-pretrain-warmup-steps"
        ),
        "finetune_warmup_steps": "--finetune-warmup-steps",
        "minimum_finetune_optimizer_steps": (
            "--minimum-finetune-optimizer-steps"
        ),
        "minimum_checkpoint_selection_finetune_steps": (
            "--minimum-checkpoint-selection-finetune-steps"
        ),
        "minimum_learning_rate_ratio": "--minimum-learning-rate-ratio",
        "seed": "--seed",
        "negative_samples_per_hold": "--negative-samples-per-hold",
        "boundary_guard_seconds": "--boundary-guard-seconds",
        "max_sources": "--max-sources",
        "device": "--device",
    }
    for key, option in option_names.items():
        args.extend((option, str(TRAINING_SCHEDULE[key])))
    args.extend(("--model-config-json", _json_argument(model_config)))
    args.extend(("--base-loss-contract-json", _json_argument(base_loss_contract)))
    args.extend(
        (
            "--dynamic-crop-loss-contract-json",
            _json_argument(dynamic_crop_loss_contract),
        )
    )
    return tuple(args)


def create_four_song_loso_plan(
    request_path: Path,
    *,
    checkpoint_root: Path,
    receipt_directory: Path,
    output_directory: Path,
) -> Path:
    request_file = _regular_safe_path(request_path, "four-song LOSO request")
    request = _read_json(request_file, "four-song LOSO request")
    if set(request) != {"schema_version", "contract", "songs"} or (
        request.get("schema_version") != FOUR_SONG_LOSO_REQUEST_SCHEMA_VERSION
        or request.get("contract") != FOUR_SONG_LOSO_REQUEST_CONTRACT
    ):
        raise FourSongLOSOError("four-song LOSO request contract differs")
    songs = tuple(
        _validate_song_artifacts(raw)
        for raw in _array(request.get("songs"), "LOSO request songs")
    )
    if tuple(song["training_song_id"] for song in songs) != LOSO_SONG_IDS:
        raise FourSongLOSOError("LOSO request songs are not in the exact fixed order")
    all_references = [
        reference_id
        for song in songs
        for reference_id in song["reference_ids"]
    ]
    if len(all_references) != 20 or len(set(all_references)) != 20:
        raise FourSongLOSOError("four-song references must be globally unique")

    checkpoint = _external_directory(checkpoint_root, "LOSO checkpoint root")
    receipts = _external_directory(receipt_directory, "LOSO receipt directory")
    output = _external_directory(output_directory, "LOSO plan output directory")
    model_config = production_stage_edit_config().to_dict()
    base_loss = StageEditLossContract(source_objective="eligible_all_hybrid")
    base_loss_payload = asdict(base_loss)
    dynamic_loss_payload = asdict(DynamicCropV3LossContract(base=base_loss))
    folds: list[dict[str, Any]] = []
    for held_out_song_id in LOSO_SONG_IDS:
        slug = held_out_song_id.removeprefix("IVE-").lower()
        checkpoint_directory = checkpoint / f"heldout-{slug}"
        if checkpoint_directory.exists() or checkpoint_directory.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite LOSO checkpoint: {checkpoint_directory}"
            )
        args = _build_train_cli_args(
            songs,
            held_out_song_id=held_out_song_id,
            checkpoint_directory=checkpoint_directory,
            model_config=model_config,
            base_loss_contract=base_loss_payload,
            dynamic_crop_loss_contract=dynamic_loss_payload,
        )
        folds.append(
            {
                "held_out_song_id": held_out_song_id,
                "training_song_ids": [
                    song_id
                    for song_id in LOSO_SONG_IDS
                    if song_id != held_out_song_id
                ],
                "checkpoint_directory": str(checkpoint_directory),
                "train_cli_args": list(args),
                "whole_song_heldout_only": True,
                "heldout_used_for_model_or_threshold_selection": False,
                "production_activation_authorized": False,
                "production_registry_activation_requested": False,
            }
        )
    source_provenance = _capture_source_provenance()
    request_pointer = _pointer(request_file, "four-song LOSO request")
    unsigned = {
        "schema_version": FOUR_SONG_LOSO_PLAN_SCHEMA_VERSION,
        "contract": FOUR_SONG_LOSO_PLAN_CONTRACT,
        "request": request_pointer,
        "source_provenance": source_provenance,
        "song_ids": list(LOSO_SONG_IDS),
        "songs": list(songs),
        "model_config": model_config,
        "base_loss_contract": base_loss_payload,
        "dynamic_crop_v3_loss_contract": dynamic_loss_payload,
        "training_schedule": dict(TRAINING_SCHEDULE),
        "checkpoint_root": str(checkpoint),
        "receipt_directory": str(receipts),
        "folds": folds,
        "fold_count": 4,
        "all_fold_receipts_required_before_selection": True,
        "fold_selection_authorized": False,
        "production_activation_authorized": False,
        "production_registry_activation_requested": False,
        "artifact_content_sha256": None,
    }
    return _atomic_content_addressed_write(
        output,
        prefix=FOUR_SONG_LOSO_PLAN_PREFIX,
        unsigned_payload=unsigned,
    )


def load_four_song_loso_plan(path: Path) -> dict[str, Any]:
    resolved = _regular_safe_path(path, "four-song LOSO plan")
    payload = _read_json(resolved, "four-song LOSO plan")
    expected_keys = {
        "schema_version",
        "contract",
        "request",
        "source_provenance",
        "song_ids",
        "songs",
        "model_config",
        "base_loss_contract",
        "dynamic_crop_v3_loss_contract",
        "training_schedule",
        "checkpoint_root",
        "receipt_directory",
        "folds",
        "fold_count",
        "all_fold_receipts_required_before_selection",
        "fold_selection_authorized",
        "production_activation_authorized",
        "production_registry_activation_requested",
        "artifact_content_sha256",
    }
    if set(payload) != expected_keys:
        raise FourSongLOSOError("four-song LOSO plan keys are not exact")
    digest = payload.get("artifact_content_sha256")
    unsigned = dict(payload)
    unsigned["artifact_content_sha256"] = None
    if (
        payload.get("schema_version") != FOUR_SONG_LOSO_PLAN_SCHEMA_VERSION
        or payload.get("contract") != FOUR_SONG_LOSO_PLAN_CONTRACT
        or not _is_sha256(digest)
        or _canonical_sha256(unsigned) != digest
        or resolved.name != f"{FOUR_SONG_LOSO_PLAN_PREFIX}{digest}.json"
        or payload.get("song_ids") != list(LOSO_SONG_IDS)
        or payload.get("fold_count") != 4
        or payload.get("all_fold_receipts_required_before_selection") is not True
        or payload.get("fold_selection_authorized") is not False
        or payload.get("production_activation_authorized") is not False
        or payload.get("production_registry_activation_requested") is not False
        or payload.get("training_schedule") != dict(TRAINING_SCHEDULE)
    ):
        raise FourSongLOSOError("four-song LOSO plan contract or hash differs")
    _verify_pointer(payload["request"], "four-song LOSO request")
    _verify_source_provenance(payload["source_provenance"])
    base_loss = StageEditLossContract(source_objective="eligible_all_hybrid")
    expected_model = production_stage_edit_config().to_dict()
    expected_base_loss = asdict(base_loss)
    expected_dynamic_loss = asdict(DynamicCropV3LossContract(base=base_loss))
    if (
        payload.get("model_config") != expected_model
        or payload.get("base_loss_contract") != expected_base_loss
        or payload.get("dynamic_crop_v3_loss_contract")
        != expected_dynamic_loss
    ):
        raise FourSongLOSOError("four-song LOSO model or loss contract differs")
    songs = tuple(
        _validate_planned_song(raw)
        for raw in _array(payload.get("songs"), "LOSO plan songs")
    )
    if tuple(song["training_song_id"] for song in songs) != LOSO_SONG_IDS:
        raise FourSongLOSOError("four-song LOSO plan song order differs")
    references = [
        reference_id
        for song in songs
        for reference_id in song["reference_ids"]
    ]
    if len(references) != 20 or len(set(references)) != 20:
        raise FourSongLOSOError("four-song LOSO plan references are not exact")
    folds = _array(payload.get("folds"), "LOSO plan folds")
    if tuple(
        str(fold.get("held_out_song_id") or "")
        for fold in folds
        if isinstance(fold, Mapping)
    ) != LOSO_SONG_IDS:
        raise FourSongLOSOError("four-song LOSO fold order differs")
    checkpoint_root = _external_directory(
        str(payload.get("checkpoint_root") or ""), "LOSO checkpoint root"
    )
    _external_directory(
        str(payload.get("receipt_directory") or ""), "LOSO receipt directory"
    )
    for raw_fold in folds:
        fold = _mapping(raw_fold, "LOSO fold")
        held_out_song_id = str(fold.get("held_out_song_id") or "")
        if held_out_song_id not in LOSO_SONG_IDS:
            raise FourSongLOSOError("LOSO fold held-out identity is invalid")
        slug = held_out_song_id.removeprefix("IVE-").lower()
        checkpoint = checkpoint_root / f"heldout-{slug}"
        expected_args = _build_train_cli_args(
            songs,
            held_out_song_id=held_out_song_id,
            checkpoint_directory=checkpoint,
            model_config=expected_model,
            base_loss_contract=expected_base_loss,
            dynamic_crop_loss_contract=expected_dynamic_loss,
        )
        if (
            fold.get("checkpoint_directory") != str(checkpoint)
            or tuple(str(value) for value in _array(
                fold.get("train_cli_args"), "LOSO train args"
            ))
            != expected_args
        ):
            raise FourSongLOSOError("LOSO fold command differs from its plan")
    return payload


def preflight_four_song_loso_fold(
    plan_path: Path,
    *,
    held_out_song_id: str,
) -> FourSongFoldPreflight:
    if held_out_song_id not in LOSO_SONG_IDS:
        raise FourSongLOSOError("held-out song is not in the fixed LOSO set")
    resolved_plan = _regular_safe_path(plan_path, "four-song LOSO plan")
    plan = load_four_song_loso_plan(resolved_plan)
    fold = next(
        _mapping(raw, "LOSO fold")
        for raw in _array(plan["folds"], "LOSO folds")
        if isinstance(raw, Mapping)
        and raw.get("held_out_song_id") == held_out_song_id
    )
    expected_training = tuple(
        song_id for song_id in LOSO_SONG_IDS if song_id != held_out_song_id
    )
    if (
        tuple(fold.get("training_song_ids") or ()) != expected_training
        or fold.get("whole_song_heldout_only") is not True
        or fold.get("heldout_used_for_model_or_threshold_selection") is not False
        or fold.get("production_activation_authorized") is not False
        or fold.get("production_registry_activation_requested") is not False
    ):
        raise FourSongLOSOError("LOSO fold leakage or authority contract differs")
    checkpoint = _external_directory(
        str(fold.get("checkpoint_directory") or ""),
        "LOSO checkpoint directory",
    )
    if checkpoint.exists() or checkpoint.is_symlink():
        raise FileExistsError(f"refusing to overwrite LOSO checkpoint: {checkpoint}")
    receipt_directory = _external_directory(
        str(plan.get("receipt_directory") or ""), "LOSO receipt directory"
    )
    free_bytes = shutil.disk_usage(checkpoint.parent).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise FourSongLOSOError(
            f"LOSO fold requires at least {MINIMUM_FREE_BYTES} free bytes"
        )
    args = tuple(str(value) for value in _array(fold.get("train_cli_args"), "train args"))
    if (
        "--authorize-production-activation" in args
        or "--production-registry" in args
        or args.count("--held-out-song") != 1
        or held_out_song_id not in args
        or args.count("--dynamic-crop-version") != 1
        or "3" not in args
    ):
        raise FourSongLOSOError("LOSO train arguments gained unsafe authority")
    return FourSongFoldPreflight(
        plan_path=resolved_plan,
        plan_content_sha256=str(plan["artifact_content_sha256"]),
        held_out_song_id=held_out_song_id,
        training_song_ids=expected_training,
        checkpoint_directory=checkpoint,
        receipt_directory=receipt_directory,
        train_cli_args=args,
        free_bytes=free_bytes,
    )


def launch_four_song_loso_fold(
    plan_path: Path,
    *,
    held_out_song_id: str,
) -> tuple[FourSongFoldPreflight, Path, int]:
    preflight = preflight_four_song_loso_fold(
        plan_path,
        held_out_song_id=held_out_song_id,
    )
    exit_code = int(production_train_main(preflight.train_cli_args))
    report_path = preflight.checkpoint_directory / PRODUCTION_TRAINING_REPORT_FILENAME
    selection_path = preflight.checkpoint_directory / STAGE_EDIT_SELECTION_FREEZE_FILENAME
    report_pointer = _pointer(report_path, "LOSO production report")
    selection_pointer = _pointer(selection_path, "LOSO selection freeze")
    report = _read_json(report_path, "LOSO production report")
    heldout = _mapping(report.get("held_out_evaluation"), "LOSO held-out evaluation")
    quality = _mapping(heldout.get("quality_gate"), "LOSO held-out quality gate")
    quality_passed = quality.get("passed") is True
    if exit_code not in {0, 2} or (exit_code == 0) != quality_passed:
        raise FourSongLOSOError("LOSO process exit code and quality gate differ")
    unsigned = {
        "schema_version": FOUR_SONG_LOSO_RECEIPT_SCHEMA_VERSION,
        "contract": FOUR_SONG_LOSO_RECEIPT_CONTRACT,
        "plan": _pointer(preflight.plan_path, "four-song LOSO plan"),
        "plan_content_sha256": preflight.plan_content_sha256,
        "held_out_song_id": held_out_song_id,
        "training_song_ids": list(preflight.training_song_ids),
        "checkpoint_directory": str(preflight.checkpoint_directory),
        "training_exit_code": exit_code,
        "quality_gate_passed": quality_passed,
        "production_training_report": report_pointer,
        "selection_freeze": selection_pointer,
        "whole_song_heldout_only": True,
        "heldout_used_for_model_or_threshold_selection": False,
        "production_activation_authorized": False,
        "production_registry_activation_requested": False,
        "artifact_content_sha256": None,
    }
    receipt = _atomic_content_addressed_write(
        preflight.receipt_directory,
        prefix=FOUR_SONG_LOSO_RECEIPT_PREFIX,
        unsigned_payload=unsigned,
    )
    return preflight, receipt, exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or run receipt-bound four-song StageEdit LOSO folds"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-plan")
    create.add_argument("--request", required=True, type=Path)
    create.add_argument("--checkpoint-root", required=True, type=Path)
    create.add_argument("--receipt-directory", required=True, type=Path)
    create.add_argument("--output-directory", required=True, type=Path)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--plan", required=True, type=Path)
    preflight.add_argument(
        "--held-out-song", required=True, choices=LOSO_SONG_IDS
    )
    launch = commands.add_parser("launch-fold")
    launch.add_argument("--plan", required=True, type=Path)
    launch.add_argument("--held-out-song", required=True, choices=LOSO_SONG_IDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "create-plan":
        path = create_four_song_loso_plan(
            args.request,
            checkpoint_root=args.checkpoint_root,
            receipt_directory=args.receipt_directory,
            output_directory=args.output_directory,
        )
        print(json.dumps({"plan": str(path)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "preflight":
        preflight = preflight_four_song_loso_fold(
            args.plan,
            held_out_song_id=args.held_out_song,
        )
        print(json.dumps(preflight.to_dict(), ensure_ascii=False, indent=2))
        return 0
    preflight, receipt, exit_code = launch_four_song_loso_fold(
        args.plan,
        held_out_song_id=args.held_out_song,
    )
    print(
        json.dumps(
            {
                "preflight": preflight.to_dict(),
                "receipt": str(receipt),
                "training_exit_code": exit_code,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BANG_BANG",
    "BLACKHOLE",
    "FEATURE_SONG_ID_BY_TRAINING_SONG",
    "FOUR_SONG_LOSO_PLAN_CONTRACT",
    "FOUR_SONG_LOSO_RECEIPT_CONTRACT",
    "FOUR_SONG_LOSO_REQUEST_CONTRACT",
    "FourSongFoldPreflight",
    "FourSongLOSOError",
    "LOSO_SONG_IDS",
    "REBEL_HEART",
    "TRAINING_SCHEDULE",
    "XOXZ",
    "create_four_song_loso_plan",
    "launch_four_song_loso_fold",
    "load_four_song_loso_plan",
    "preflight_four_song_loso_fold",
]
