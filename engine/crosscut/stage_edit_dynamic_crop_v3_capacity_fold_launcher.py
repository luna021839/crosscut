"""Strict Dynamic Crop v3 three-song capacity-fold launcher and receipts.

The existing v2 plan schema is intentionally untouched.  This module emits a
separate v3-only plan after admitting candidates through clean, passing,
content-addressed v3 MLX capacity smokes.  Every candidate receives exactly
three whole-song folds over BANG BANG, BLACKHOLE, and REBEL HEART.

Fold receipts remain deliberately non-selecting: one successful fold proves
only that the planned inactive v3 training job completed.  Neither a plan nor
a receipt can activate the production registry or select a winning capacity;
an eventual selector must separately require all three held-out folds.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import stage_edit_dynamic_crop_capacity_fold_launcher as v2_launcher
from .dynamic_crop_input_verification import verify_dynamic_crop_input_files
from .reference_dynamic_crop_supervision import (
    load_reference_dynamic_crop_supervision,
)
from .stage_edit_dynamic_crop_capacity_probe import (
    SELECTION_SONG_IDS,
    _atomic_exclusive_write,
)
from .stage_edit_dynamic_crop_head_v3 import (
    DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
    DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
    DYNAMIC_CROP_V3_LOSS_CONTRACT,
    DYNAMIC_CROP_V3_MODEL_CONTRACT,
    DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
    DYNAMIC_CROP_V3_SCHEMA_VERSION,
    DYNAMIC_CROP_V3_TARGET_CONTRACT,
    DynamicCropHeadV3Config,
    StageEditDynamicCropCheckpointMetadataV3,
)
from .stage_edit_dynamic_crop_v3_capacity_smoke import (
    DynamicCropV3CapacityCandidate,
    V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE,
    V3_CAPACITY_SMOKE_ARTIFACT_PREFIX,
    V3_PROBE_TRAINING_SCHEDULE as PROBE_TRAINING_SCHEDULE,
    V3_CAPACITY_SMOKE_RUNNER_CONTRACT,
    V3_CAPACITY_SMOKE_SOURCE_RELATIVE_PATHS,
    _validate_capacity_smoke_code_provenance,
    default_dynamic_crop_v3_capacity_candidates,
    load_dynamic_crop_v3_capacity_smoke,
)
from .stage_edit_production_train_cli import main as production_train_main
from .stage_edit_production_training import (
    DYNAMIC_CROP_V3_PRIOR_CANDIDATE_SELECTION_CONTRACT,
    DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS,
    STAGE_EDIT_SELECTION_FREEZE_FILENAME,
    _stage_edit_validation_selection_with_dynamic_crop_v3,
)
from .stage_edit_training import stage_edit_validation_selection_score
from .stage_edit_transformer import (
    PRODUCTION_CHECKPOINT_SCOPE,
    PRODUCTION_TRAINING_REPORT_FILENAME,
    PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION,
    PRODUCTION_TRAINING_RUN_KIND,
    valid_dynamic_crop_v3_activation_checkpointing_evidence,
)


V3_CAPACITY_FOLD_PLAN_SCHEMA_VERSION = 1
V3_CAPACITY_FOLD_PLAN_ANALYSIS_BASIS = (
    "ive_bang_blackhole_rebel_dynamic_crop_v3_capacity_fold_plan_v1"
)
V3_CAPACITY_FOLD_PLAN_PREFIX = "stageedit-dynamic-crop-v3-capacity-fold-plan-"
V3_CAPACITY_FOLD_CONTRACT = (
    "exact_three_song_whole_song_dynamic_crop_v3_capacity_cross_validation_v1"
)
V3_CAPACITY_FOLD_EVIDENCE_SCHEMA_VERSION = 1
V3_CAPACITY_FOLD_EVIDENCE_ANALYSIS_BASIS = (
    "verified_dynamic_crop_v3_inactive_fold_execution_receipt_v1"
)
V3_CAPACITY_FOLD_EVIDENCE_PREFIX = "dynamic-crop-v3-fold-run-"
V3_CAPACITY_FOLD_EVIDENCE_CONTRACT = (
    "single_v3_fold_nonselecting_all_three_required_v1"
)
V3_CAPACITY_SMOKE_ADMISSION_CONTRACT = (
    "clean_release_trusted_content_addressed_runtime_and_capacity_pass_v1"
)
V3_CAPACITY_PREFLIGHT_INPUT_VERIFICATION_CONTRACT = (
    "all_artifact_provenance_inputs_shared_sha256_cache_v1"
)
TRAIN_MODULE = "crosscut.stage_edit_production_train_cli"
PRODUCTION_VOLUME_ROOT = Path("/Volumes/CrossCutMedia")

_EXPECTED_SONG_IDS = (
    "IVE-BANG-BANG",
    "IVE-BLACKHOLE",
    "IVE-REBEL-HEART",
)
if tuple(SELECTION_SONG_IDS) != _EXPECTED_SONG_IDS:  # pragma: no cover
    raise RuntimeError("Dynamic Crop v3 capacity song contract drifted")

V3_SOURCE_RELATIVE_PATHS = (
    "engine/crosscut/stage_edit_dynamic_crop_v3_capacity_fold_launcher.py",
    "engine/crosscut/stage_edit_dynamic_crop_v3_capacity_smoke.py",
    "engine/crosscut/stage_edit_dynamic_crop_mlx_smoke_cli.py",
    "engine/crosscut/stage_edit_dynamic_crop_capacity_fold_launcher.py",
    "engine/crosscut/stage_edit_dynamic_crop_capacity_probe.py",
    "engine/crosscut/stage_edit_production_fold_launcher.py",
    "engine/crosscut/stage_edit_production_train_cli.py",
    "engine/crosscut/stage_edit_production_training.py",
    "engine/crosscut/stage_edit_dynamic_crop_training.py",
    "engine/crosscut/stage_edit_dynamic_crop_training_v3.py",
    "engine/crosscut/stage_edit_dynamic_crop_head_v3.py",
    "engine/crosscut/stage_edit_dynamic_crop_target_v3.py",
    "engine/crosscut/stage_edit_transformer.py",
    "engine/crosscut/stage_edit_reference_adapter.py",
    "engine/crosscut/stage_edit_feature_materializer.py",
    "engine/crosscut/reference_dynamic_crop_job.py",
    "engine/crosscut/reference_dynamic_crop_supervision.py",
    "engine/crosscut/dynamic_crop_input_verification.py",
)


class DynamicCropV3CapacityFoldLauncherError(ValueError):
    pass


_ValidatedManifestSet = v2_launcher._ValidatedManifestSet


@dataclass(frozen=True, slots=True)
class _AdmittedCapacitySmoke:
    candidate: DynamicCropV3CapacityCandidate
    path: Path
    pointer: Mapping[str, str]
    payload: Mapping[str, Any]
    source_commit: str

    def plan_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "candidate_name": self.candidate.candidate_name,
            "evidence": dict(self.pointer),
            "evidence_content_sha256": self.payload[
                "artifact_content_sha256"
            ],
            "source_commit": self.source_commit,
            "admission_contract": V3_CAPACITY_SMOKE_ADMISSION_CONTRACT,
            "runtime_execution_passed": True,
            "within_release_memory_budget": True,
            "capacity_smoke_passed": True,
            "release_trust_authorized": True,
            "activation_checkpointing": dict(
                V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE
            ),
            "semantic_metrics_authorized": False,
            "model_selection_authorized": False,
        }


@dataclass(frozen=True, slots=True)
class DynamicCropV3ProvenanceVerificationEvidence:
    manifest_count: int
    artifact_count: int
    verified_path_count: int
    verified_byte_count: int
    verified_path_records_sha256: str

    def __post_init__(self) -> None:
        if self.manifest_count != len(SELECTION_SONG_IDS):
            raise ValueError("provenance evidence must cover all three manifests")
        if self.artifact_count < 1 or self.verified_path_count < 1:
            raise ValueError("provenance evidence counts must be positive")
        if self.verified_byte_count < 1:
            raise ValueError("provenance evidence byte count must be positive")
        if not _is_sha256(self.verified_path_records_sha256):
            raise ValueError("provenance evidence path-record hash is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": V3_CAPACITY_PREFLIGHT_INPUT_VERIFICATION_CONTRACT,
            "hash_algorithm": "sha256",
            "manifest_count": self.manifest_count,
            "artifact_count": self.artifact_count,
            "verified_path_count": self.verified_path_count,
            "verified_byte_count": self.verified_byte_count,
            "verified_path_records_sha256": self.verified_path_records_sha256,
            "all_artifact_provenance_inputs_verified": True,
        }


@dataclass(frozen=True, slots=True)
class DynamicCropV3CapacityFoldPreflight:
    plan_path: Path
    plan_content_sha256: str
    candidate_name: str
    candidate_id: str
    capacity_smoke_path: Path
    held_out_song_id: str
    training_song_ids: tuple[str, ...]
    checkpoint_directory: Path
    evidence_directory: Path
    selection_dataset_manifest_path: Path
    reference_ids_by_song: Mapping[str, tuple[str, ...]]
    train_cli_args: tuple[str, ...]
    command_argv: tuple[str, ...]
    free_bytes: int
    dynamic_crop_provenance_verification: (
        DynamicCropV3ProvenanceVerificationEvidence | None
    ) = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_path": str(self.plan_path),
            "plan_content_sha256": self.plan_content_sha256,
            "candidate_name": self.candidate_name,
            "candidate_id": self.candidate_id,
            "capacity_smoke_path": str(self.capacity_smoke_path),
            "held_out_song_id": self.held_out_song_id,
            "training_song_ids": list(self.training_song_ids),
            "checkpoint_directory": str(self.checkpoint_directory),
            "evidence_directory": str(self.evidence_directory),
            "selection_dataset_manifest_path": str(
                self.selection_dataset_manifest_path
            ),
            "reference_ids_by_song": {
                song: list(references)
                for song, references in sorted(
                    self.reference_ids_by_song.items()
                )
            },
            "train_cli_args": list(self.train_cli_args),
            "command_argv": list(self.command_argv),
            "free_bytes": self.free_bytes,
            "dynamic_crop_provenance_verification": (
                self.dynamic_crop_provenance_verification.to_dict()
                if self.dynamic_crop_provenance_verification is not None
                else None
            ),
            "dynamic_crop_version": 3,
            "training_authorized": (
                self.dynamic_crop_provenance_verification is not None
            ),
            "fold_selection_authorized": False,
            "capacity_selection_authorized": False,
            "production_activation_authorized": False,
            "production_registry_activation_requested": False,
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


def _exact_activation_checkpointing(value: object) -> bool:
    return valid_dynamic_crop_v3_activation_checkpointing_evidence(value)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DynamicCropV3CapacityFoldLauncherError(
            f"{label} must be an object"
        )
    return dict(value)


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise DynamicCropV3CapacityFoldLauncherError(
            f"{label} must be an array"
        )
    return value


def _validate_manifest_set(**kwargs: Any) -> _ValidatedManifestSet:
    try:
        return v2_launcher._validate_manifest_set(**kwargs)
    except (OSError, ValueError) as exc:
        raise DynamicCropV3CapacityFoldLauncherError(str(exc)) from exc


def _verify_dynamic_crop_provenance_inputs(
    manifests: _ValidatedManifestSet,
) -> DynamicCropV3ProvenanceVerificationEvidence:
    """Hash every immutable crop input once before training is authorized."""

    dynamic_manifests = dict(manifests.dynamic_manifest_paths_by_song)
    if set(dynamic_manifests) != set(SELECTION_SONG_IDS):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 provenance verification requires all three dynamic manifests"
        )

    verified_sha256_cache: dict[str, str] = {}
    verified_identity_cache: dict[str, tuple[int, int, int, int, int]] = {}
    seen_artifact_paths: set[Path] = set()
    artifact_count = 0
    for song_id in SELECTION_SONG_IDS:
        manifest_path = dynamic_manifests[song_id]
        try:
            manifest = _mapping(
                json.loads(manifest_path.read_text(encoding="utf-8")),
                f"{song_id} dynamic crop manifest",
            )
            records = _array(
                manifest.get("artifacts"), f"{song_id} dynamic crop artifacts"
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise DynamicCropV3CapacityFoldLauncherError(
                f"{song_id} dynamic crop provenance manifest is invalid: {exc}"
            ) from exc

        for index, raw_record in enumerate(records):
            reference_id = "unknown-reference"
            artifact_path: Path | None = None
            try:
                record = _mapping(
                    raw_record, f"{song_id} dynamic crop artifact {index}"
                )
                reference_id = str(record.get("reference_id") or "").strip()
                if not reference_id:
                    raise ValueError("dynamic crop artifact reference id is blank")
                artifact = _mapping(
                    record.get("artifact"),
                    f"{song_id}:{reference_id} dynamic crop artifact pointer",
                )
                artifact_path = v2_launcher._regular_safe_path(
                    str(artifact.get("path") or ""),
                    f"{song_id}:{reference_id} dynamic crop artifact",
                )
                if artifact_path in seen_artifact_paths:
                    raise ValueError(
                        f"duplicate dynamic crop artifact input: {artifact_path}"
                    )
                expected_file_sha256 = artifact.get("file_sha256")
                expected_content_sha256 = artifact.get("content_sha256")
                if not _is_sha256(expected_file_sha256) or not _is_sha256(
                    expected_content_sha256
                ):
                    raise ValueError("dynamic crop artifact hashes are invalid")
                observed_file_sha256 = v2_launcher._stable_file_sha256(
                    artifact_path,
                    f"{song_id}:{reference_id} dynamic crop artifact",
                )
                if observed_file_sha256 != expected_file_sha256:
                    raise ValueError(
                        "dynamic crop artifact file hash mismatch "
                        f"path={artifact_path} "
                        f"expected_sha256={expected_file_sha256} "
                        f"observed_sha256={observed_file_sha256}"
                    )
                restored = load_reference_dynamic_crop_supervision(
                    artifact_path,
                    verify_input_files=False,
                )
                observed_content_sha256 = restored.payload["provenance"].get(
                    "artifact_content_sha256"
                )
                if observed_content_sha256 != expected_content_sha256:
                    raise ValueError(
                        "dynamic crop artifact content hash mismatch "
                        f"path={artifact_path} "
                        f"expected_sha256={expected_content_sha256} "
                        f"observed_sha256={observed_content_sha256}"
                    )
                verify_dynamic_crop_input_files(
                    restored.payload,
                    verified_sha256_cache=verified_sha256_cache,
                    verified_identity_cache=verified_identity_cache,
                )
            except (OSError, TypeError, ValueError) as exc:
                path_label = str(artifact_path) if artifact_path else "unresolved"
                raise DynamicCropV3CapacityFoldLauncherError(
                    "v3 dynamic crop provenance verification failed "
                    f"song={song_id} reference={reference_id} "
                    f"artifact_index={index} artifact_path={path_label}: {exc}"
                ) from exc
            assert artifact_path is not None
            seen_artifact_paths.add(artifact_path)
            artifact_count += 1

    if artifact_count < 1 or not verified_sha256_cache:
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 dynamic crop provenance verification found no artifact inputs"
        )

    path_records: list[dict[str, Any]] = []
    verified_byte_count = 0
    for raw_path, observed_sha256 in sorted(verified_sha256_cache.items()):
        path = Path(raw_path)
        identity = verified_identity_cache.get(raw_path)
        if identity is None:
            raise DynamicCropV3CapacityFoldLauncherError(
                f"verified v3 dynamic crop input identity is missing: {path}"
            )
        try:
            observed = path.stat()
        except OSError as exc:
            raise DynamicCropV3CapacityFoldLauncherError(
                f"verified v3 dynamic crop input became unavailable: {path}: {exc}"
            ) from exc
        current_identity = (
            observed.st_dev,
            observed.st_ino,
            observed.st_size,
            observed.st_mtime_ns,
            observed.st_ctime_ns,
        )
        if current_identity != identity:
            raise DynamicCropV3CapacityFoldLauncherError(
                f"verified v3 dynamic crop input changed before audit completed: {path}"
            )
        device, inode, byte_count, mtime_ns, ctime_ns = identity
        verified_byte_count += byte_count
        path_records.append(
            {
                "path": raw_path,
                "device": device,
                "inode": inode,
                "byte_count": byte_count,
                "mtime_ns": mtime_ns,
                "ctime_ns": ctime_ns,
                "sha256": observed_sha256,
            }
        )
    return DynamicCropV3ProvenanceVerificationEvidence(
        manifest_count=len(dynamic_manifests),
        artifact_count=artifact_count,
        verified_path_count=len(path_records),
        verified_byte_count=verified_byte_count,
        verified_path_records_sha256=_canonical_sha256(path_records),
    )


def _capture_source_provenance(
    repository_root: Path | None = None,
) -> dict[str, Any]:
    try:
        return v2_launcher._capture_source_provenance(
            repository_root,
            source_relative_paths=V3_SOURCE_RELATIVE_PATHS,
        )
    except (OSError, ValueError) as exc:
        raise DynamicCropV3CapacityFoldLauncherError(str(exc)) from exc


def _validate_source_provenance(value: object) -> dict[str, Any]:
    payload = _mapping(value, "v3 source provenance")
    required = {
        "contract",
        "repository_root",
        "head_commit_oid",
        "repository_status_clean",
        "repository_status_porcelain_sha256",
        "source_files",
    }
    if set(payload) != required:
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 source provenance keys are not exact"
        )
    files = _mapping(payload.get("source_files"), "v3 source files")
    if set(files) != set(V3_SOURCE_RELATIVE_PATHS):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 source provenance file set is incomplete"
        )
    if (
        payload.get("contract")
        != "clean_head_and_exact_interpreter_source_blobs_v1"
        or payload.get("repository_status_clean") is not True
        or payload.get("repository_status_porcelain_sha256")
        != hashlib.sha256(b"").hexdigest()
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 source provenance is not clean"
        )
    observed = _capture_source_provenance(
        Path(str(payload.get("repository_root") or ""))
    )
    if observed != payload:
        raise DynamicCropV3CapacityFoldLauncherError(
            "current clean v3 source differs from immutable plan"
        )
    return payload


def _require_same_source_provenance(expected: Mapping[str, Any]) -> None:
    if _validate_source_provenance(expected) != dict(expected):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 source provenance changed"
        )


def _candidate_defaults_by_id() -> dict[str, DynamicCropV3CapacityCandidate]:
    candidates = default_dynamic_crop_v3_capacity_candidates()
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if len(candidates) != 4 or len(by_id) != 4:
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 capacity candidate defaults overlap"
        )
    return by_id


def _admit_capacity_smokes(
    paths: Sequence[Path],
    *,
    expected_source_provenance: Mapping[str, Any],
) -> dict[str, _AdmittedCapacitySmoke]:
    if not paths:
        raise DynamicCropV3CapacityFoldLauncherError(
            "at least one v3 capacity smoke is required"
        )
    current_source = _validate_source_provenance(expected_source_provenance)
    current_commit = str(current_source["head_commit_oid"])
    current_files = _mapping(
        current_source.get("source_files"), "current v3 source files"
    )
    defaults = _candidate_defaults_by_id()
    admitted: dict[str, _AdmittedCapacitySmoke] = {}
    for raw_path in paths:
        try:
            path = v2_launcher._regular_safe_path(
                raw_path, "Dynamic Crop v3 capacity smoke"
            )
            payload = load_dynamic_crop_v3_capacity_smoke(path)
            pointer = v2_launcher._pointer(
                path, "Dynamic Crop v3 capacity smoke"
            )
        except (OSError, ValueError) as exc:
            raise DynamicCropV3CapacityFoldLauncherError(
                f"v3 capacity smoke is invalid: {exc}"
            ) from exc
        candidate_payload = _mapping(payload.get("candidate"), "smoke candidate")
        candidate_id = str(candidate_payload.get("candidate_id") or "")
        candidate = defaults.get(candidate_id)
        if candidate is None or candidate_payload != candidate.to_dict():
            raise DynamicCropV3CapacityFoldLauncherError(
                "capacity smoke candidate is not one exact v3 default"
            )
        if candidate_id in admitted:
            raise DynamicCropV3CapacityFoldLauncherError(
                "duplicate v3 capacity smoke candidate"
            )
        execution = _mapping(payload.get("execution"), "smoke execution")
        try:
            code = _validate_capacity_smoke_code_provenance(
                payload.get("code_provenance"),
                require_clean_release=True,
            )
        except (OSError, ValueError) as exc:
            raise DynamicCropV3CapacityFoldLauncherError(
                "capacity smoke did not pass clean v3 runtime/capacity "
                f"admission: {exc}"
            ) from exc
        smoke_files = _mapping(
            code.get("source_files"), "smoke code source files"
        )
        source_files_match_current = all(
            isinstance(smoke_files.get(role), Mapping)
            and isinstance(current_files.get(relative), Mapping)
            and smoke_files[role].get("file_sha256")
            == current_files[relative].get("file_sha256")
            for role, relative in V3_CAPACITY_SMOKE_SOURCE_RELATIVE_PATHS.items()
        )
        candidate_contract = _mapping(
            candidate_payload.get("contract"), "smoke candidate contract"
        )
        if (
            payload.get("dynamic_crop_version") != 3
            or payload.get("artifact_family")
            != "dynamic_crop_v3_capacity_smoke"
            or not path.name.startswith(V3_CAPACITY_SMOKE_ARTIFACT_PREFIX)
            or payload.get("semantic_metrics_authorized") is not False
            or payload.get("model_selection_authorized") is not False
            or not _exact_activation_checkpointing(
                payload.get("activation_checkpointing")
            )
            or not _exact_activation_checkpointing(
                execution.get("activation_checkpointing")
            )
            or not _exact_activation_checkpointing(
                candidate_contract.get("activation_checkpointing")
            )
            or execution.get("runtime_execution_passed") is not True
            or execution.get("within_release_memory_budget") is not True
            or execution.get("capacity_smoke_passed") is not True
            or execution.get("semantic_metrics_authorized") is not False
            or execution.get("model_selection_authorized") is not False
            or code.get("release_trust_authorized") is not True
            or code.get("repository_status_clean") is not True
            or code.get("source_files_match_observed_commit") is not True
            or code.get("expected_git_commit")
            != code.get("observed_git_commit")
            or code.get("observed_git_commit") != current_commit
            or not source_files_match_current
        ):
            raise DynamicCropV3CapacityFoldLauncherError(
                "capacity smoke did not pass clean v3 runtime/capacity admission"
            )
        commit = str(code.get("observed_git_commit") or "")
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise DynamicCropV3CapacityFoldLauncherError(
                "capacity smoke source commit is invalid"
            )
        admitted[candidate_id] = _AdmittedCapacitySmoke(
            candidate=candidate,
            path=path,
            pointer=pointer,
            payload=payload,
            source_commit=commit,
        )
    return admitted


def _resolve_candidates(
    admitted: Mapping[str, _AdmittedCapacitySmoke],
    candidate_selectors: Sequence[str] | None,
) -> tuple[DynamicCropV3CapacityCandidate, ...]:
    if not candidate_selectors:
        return tuple(
            admitted[candidate_id].candidate
            for candidate_id in sorted(admitted)
        )
    lookup: dict[str, DynamicCropV3CapacityCandidate] = {}
    for smoke in admitted.values():
        lookup[smoke.candidate.candidate_id] = smoke.candidate
        lookup[smoke.candidate.candidate_name] = smoke.candidate
    selected: list[DynamicCropV3CapacityCandidate] = []
    seen: set[str] = set()
    for raw in candidate_selectors:
        selector = str(raw).strip()
        candidate = lookup.get(selector)
        if candidate is None:
            raise DynamicCropV3CapacityFoldLauncherError(
                "candidate lacks an admitted clean passing v3 capacity smoke"
            )
        if candidate.candidate_id in seen:
            raise DynamicCropV3CapacityFoldLauncherError(
                "duplicate v3 capacity candidate"
            )
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


def _build_train_cli_args(
    manifests: _ValidatedManifestSet,
    *,
    candidate: DynamicCropV3CapacityCandidate,
    held_out_song_id: str,
    checkpoint_directory: Path,
) -> tuple[str, ...]:
    try:
        args = list(
            v2_launcher._build_train_cli_args(
                manifests,
                candidate=candidate,  # structural contract is intentionally shared
                held_out_song_id=held_out_song_id,
                checkpoint_directory=checkpoint_directory,
            )
        )
    except (OSError, ValueError) as exc:
        raise DynamicCropV3CapacityFoldLauncherError(str(exc)) from exc
    args.extend(
        (
            "--dynamic-crop-v3-prior-pretrain-steps",
            str(
                PROBE_TRAINING_SCHEDULE[
                    "dynamic_crop_v3_prior_pretraining_steps"
                ]
            ),
            "--dynamic-crop-v3-prior-pretrain-learning-rate",
            str(
                PROBE_TRAINING_SCHEDULE[
                    "dynamic_crop_v3_prior_pretraining_peak_learning_rate"
                ]
            ),
            "--dynamic-crop-v3-prior-pretrain-warmup-steps",
            str(
                PROBE_TRAINING_SCHEDULE[
                    "dynamic_crop_v3_prior_pretraining_warmup_steps"
                ]
            ),
            "--dynamic-crop-version",
            "3",
        )
    )
    contract = candidate.contract_dict()
    if (
        args.count("--dynamic-crop-version") != 1
        or args[args.index("--dynamic-crop-version") + 1] != "3"
        or json.loads(args[args.index("--model-config-json") + 1])
        != contract["model_config"]
        or json.loads(args[args.index("--base-loss-contract-json") + 1])
        != contract["base_loss_contract"]
        or json.loads(
            args[args.index("--dynamic-crop-loss-contract-json") + 1]
        )
        != contract["dynamic_crop_loss_contract"]
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 fold training contracts differ from candidate"
        )
    required_schedule_flags = {
        "--source-pretrain-steps": "source_pretraining_steps",
        "--source-pretrain-learning-rate": "source_pretraining_peak_learning_rate",
        "--source-pretrain-window-length": "source_pretraining_window_length",
        "--source-pretrain-warmup-steps": "source_pretraining_warmup_steps",
        "--dynamic-crop-v3-prior-pretrain-steps": (
            "dynamic_crop_v3_prior_pretraining_steps"
        ),
        "--dynamic-crop-v3-prior-pretrain-learning-rate": (
            "dynamic_crop_v3_prior_pretraining_peak_learning_rate"
        ),
        "--dynamic-crop-v3-prior-pretrain-warmup-steps": (
            "dynamic_crop_v3_prior_pretraining_warmup_steps"
        ),
        "--minimum-finetune-optimizer-steps": "minimum_finetune_optimizer_steps",
        "--minimum-checkpoint-selection-finetune-steps": (
            "minimum_checkpoint_selection_finetune_steps"
        ),
    }
    for flag, schedule_key in required_schedule_flags.items():
        if args.count(flag) != 1 or args[args.index(flag) + 1] != str(
            PROBE_TRAINING_SCHEDULE[schedule_key]
        ):
            raise DynamicCropV3CapacityFoldLauncherError(
                f"v3 fold did not forward {schedule_key}"
            )
    forbidden = {
        "--authorize-production-activation",
        "--production-registry",
        "--professional-montage-grammar-profile",
        "--continuity-authority",
        "--source-pair-scorer",
        "--source-pair-production-report",
    }
    if forbidden & set(args):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 capacity fold acquired registry activation authority"
        )
    return tuple(args)


def _run_id(
    candidate: DynamicCropV3CapacityCandidate, held_out_song_id: str
) -> str:
    song_slug = held_out_song_id.removeprefix("IVE-").lower()
    return (
        f"v3-{candidate.candidate_name}-{candidate.candidate_id[:16]}-"
        f"heldout-{song_slug}"
    )


def _command_argv(
    train_cli_args: Sequence[str], *, repository_root: Path
) -> list[str]:
    return [
        "/usr/bin/env",
        f"PYTHONPATH={repository_root / 'engine'}",
        "python3",
        "-m",
        TRAIN_MODULE,
        *train_cli_args,
    ]


def _plan_contract() -> dict[str, Any]:
    return {
        "contract": V3_CAPACITY_FOLD_CONTRACT,
        "artist": "IVE",
        "song_ids": list(SELECTION_SONG_IDS),
        "fold_count_per_candidate": 3,
        "expected_reference_count_per_song": (
            v2_launcher.legacy.EXPECTED_REFERENCE_COUNT_PER_SONG
        ),
        "expected_source_count_per_song": (
            v2_launcher.legacy.EXPECTED_SOURCE_COUNT_PER_SONG
        ),
        "dynamic_crop_version": 3,
        "model_contract": DYNAMIC_CROP_V3_MODEL_CONTRACT,
        "target_contract": DYNAMIC_CROP_V3_TARGET_CONTRACT,
        "output_contract": DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
        "loss_contract": DYNAMIC_CROP_V3_LOSS_CONTRACT,
        "checkpoint_metadata_filename": (
            DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
        ),
        "checkpoint_weight_filename": DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
        "activation_checkpointing": dict(
            V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE
        ),
        "capacity_smoke_admission_contract": (
            V3_CAPACITY_SMOKE_ADMISSION_CONTRACT
        ),
        "selection_dataset_required": True,
        "whole_song_heldout_only": True,
        "heldout_used_for_model_or_threshold_selection": False,
        "synthetic_or_fake_training_data_allowed": False,
        "training_schedule": dict(PROBE_TRAINING_SCHEDULE),
        "sampling": dict(v2_launcher.legacy.PRODUCTION_SAMPLING_CONTRACT),
        "fold_receipt_contract": V3_CAPACITY_FOLD_EVIDENCE_CONTRACT,
        "all_three_fold_receipts_required_before_future_selection": True,
        "fold_selection_authorized": False,
        "capacity_selection_authorized": False,
        "production_activation_authorized": False,
        "production_registry_activation_requested": False,
        "quarantined_song_evidence_opened": False,
    }


def _paths_under_external_volume(path: Path, label: str) -> Path:
    try:
        candidate = v2_launcher._absolute_safe_path(path, label)
    except (OSError, ValueError) as exc:
        raise DynamicCropV3CapacityFoldLauncherError(str(exc)) from exc
    root = PRODUCTION_VOLUME_ROOT.resolve()
    if candidate != root and root not in candidate.parents:
        raise DynamicCropV3CapacityFoldLauncherError(
            f"{label} must be under {root}"
        )
    return candidate


def create_dynamic_crop_v3_capacity_fold_plan(
    *,
    materialization_manifests: Mapping[str, Path],
    feature_manifests: Mapping[str, Path],
    dynamic_crop_manifests: Mapping[str, Path],
    selection_dataset_manifest_path: Path,
    capacity_smoke_paths: Sequence[Path],
    checkpoint_root: Path,
    evidence_directory: Path,
    output_directory: Path,
    candidate_selectors: Sequence[str] | None = None,
) -> Path:
    """Create a separate immutable v3 plan from admitted capacity smokes."""

    source = _capture_source_provenance()
    admitted_all = _admit_capacity_smokes(
        capacity_smoke_paths,
        expected_source_provenance=source,
    )
    candidates = _resolve_candidates(admitted_all, candidate_selectors)
    admitted = {
        candidate.candidate_id: admitted_all[candidate.candidate_id]
        for candidate in candidates
    }
    source_commit = str(source["head_commit_oid"])
    if any(smoke.source_commit != source_commit for smoke in admitted.values()):
        raise DynamicCropV3CapacityFoldLauncherError(
            "capacity smoke clean commit differs from v3 plan source commit"
        )
    repository_root = Path(str(source["repository_root"]))
    checkpoint = _paths_under_external_volume(checkpoint_root, "checkpoint root")
    evidence = _paths_under_external_volume(
        evidence_directory, "evidence directory"
    )
    try:
        output = v2_launcher._absolute_safe_path(
            output_directory, "plan output directory"
        )
    except (OSError, ValueError) as exc:
        raise DynamicCropV3CapacityFoldLauncherError(str(exc)) from exc
    if output == repository_root or repository_root in output.parents:
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 plan output must stay outside the clean repository"
        )
    manifests = _validate_manifest_set(
        materialization_manifests=materialization_manifests,
        feature_manifests=feature_manifests,
        dynamic_crop_manifests=dynamic_crop_manifests,
        selection_dataset_manifest_path=selection_dataset_manifest_path,
    )
    planned_runs = tuple(
        (
            candidate,
            song,
            checkpoint / _run_id(candidate, song),
        )
        for candidate in candidates
        for song in SELECTION_SONG_IDS
    )
    try:
        v2_launcher._validate_artifact_path_isolation(
            plan_path=output,
            plan_label="v3 plan output directory",
            evidence_directory=evidence,
            planned_checkpoint_directories=tuple(
                path for _candidate, _song, path in planned_runs
            ),
        )
    except (OSError, ValueError) as exc:
        raise DynamicCropV3CapacityFoldLauncherError(str(exc)) from exc
    for _candidate, _song, path in planned_runs:
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite v3 checkpoint directory: {path}"
            )
    runs: list[dict[str, Any]] = []
    for candidate, heldout, run_checkpoint in planned_runs:
        args = _build_train_cli_args(
            manifests,
            candidate=candidate,
            held_out_song_id=heldout,
            checkpoint_directory=run_checkpoint,
        )
        runs.append(
            {
                "run_id": _run_id(candidate, heldout),
                "candidate_name": candidate.candidate_name,
                "candidate_id": candidate.candidate_id,
                "capacity_smoke_content_sha256": admitted[
                    candidate.candidate_id
                ].payload["artifact_content_sha256"],
                "held_out_song_id": heldout,
                "training_song_ids": sorted(
                    set(SELECTION_SONG_IDS) - {heldout}
                ),
                "checkpoint_directory": str(run_checkpoint),
                "evidence_directory": str(evidence),
                "train_cli_args": list(args),
                "command_argv": _command_argv(
                    args, repository_root=repository_root
                ),
                "dynamic_crop_version": 3,
                "activation_checkpointing": dict(
                    V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE
                ),
                "fold_selection_authorized": False,
                "capacity_selection_authorized": False,
                "production_activation_authorized": False,
                "production_registry_activation_requested": False,
            }
        )
    _require_same_source_provenance(source)
    unsigned = {
        "schema_version": V3_CAPACITY_FOLD_PLAN_SCHEMA_VERSION,
        "analysis_basis": V3_CAPACITY_FOLD_PLAN_ANALYSIS_BASIS,
        "capacity_fold_contract": _plan_contract(),
        "source_provenance": source,
        "required_volume": str(PRODUCTION_VOLUME_ROOT),
        "checkpoint_root": str(checkpoint),
        "evidence_directory": str(evidence),
        "selection_dataset_manifest": dict(
            manifests.selection_dataset_pointer
        ),
        "selection_dataset_content_sha256": (
            manifests.selection_dataset_content_sha256
        ),
        "songs": [dict(record) for record in manifests.song_records],
        "capacity_smokes": [
            admitted[candidate.candidate_id].plan_record()
            for candidate in candidates
        ],
        "candidates": [candidate.to_dict() for candidate in candidates],
        "runs": runs,
        "fold_selection_authorized": False,
        "capacity_selection_authorized": False,
        "production_activation_authorized": False,
        "plan_content_sha256": None,
    }
    digest = _canonical_sha256(unsigned)
    payload = {**unsigned, "plan_content_sha256": digest}
    try:
        return v2_launcher._atomic_exclusive_write(
            output / f"{V3_CAPACITY_FOLD_PLAN_PREFIX}{digest}.json", payload
        )
    except (OSError, ValueError) as exc:
        raise DynamicCropV3CapacityFoldLauncherError(str(exc)) from exc


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
    "capacity_smokes",
    "candidates",
    "runs",
    "fold_selection_authorized",
    "capacity_selection_authorized",
    "production_activation_authorized",
    "plan_content_sha256",
}


def _strict_plan(path: Path) -> dict[str, Any]:
    try:
        resolved = v2_launcher._regular_safe_path(path, "v3 capacity fold plan")
        payload = v2_launcher._read_json_object(
            resolved, "v3 capacity fold plan"
        )
    except (OSError, ValueError) as exc:
        raise DynamicCropV3CapacityFoldLauncherError(str(exc)) from exc
    if set(payload) != _PLAN_KEYS:
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 capacity fold plan keys are not exact"
        )
    if (
        payload.get("schema_version") != V3_CAPACITY_FOLD_PLAN_SCHEMA_VERSION
        or payload.get("analysis_basis")
        != V3_CAPACITY_FOLD_PLAN_ANALYSIS_BASIS
        or payload.get("capacity_fold_contract") != _plan_contract()
        or payload.get("required_volume") != str(PRODUCTION_VOLUME_ROOT)
        or payload.get("fold_selection_authorized") is not False
        or payload.get("capacity_selection_authorized") is not False
        or payload.get("production_activation_authorized") is not False
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 capacity fold plan contract differs"
        )
    digest = payload.get("plan_content_sha256")
    unsigned = deepcopy(payload)
    unsigned["plan_content_sha256"] = None
    if not _is_sha256(digest) or _canonical_sha256(unsigned) != digest:
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 capacity fold plan content hash mismatch"
        )
    if resolved.name != f"{V3_CAPACITY_FOLD_PLAN_PREFIX}{digest}.json":
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 capacity fold plan filename is not content-addressed"
        )
    _validate_source_provenance(payload.get("source_provenance"))
    return payload


def _candidates_from_plan(
    value: object,
) -> tuple[DynamicCropV3CapacityCandidate, ...]:
    defaults = _candidate_defaults_by_id()
    selected: list[DynamicCropV3CapacityCandidate] = []
    seen: set[str] = set()
    for raw in _array(value, "v3 capacity candidates"):
        payload = _mapping(raw, "v3 capacity candidate")
        candidate_id = str(payload.get("candidate_id") or "")
        candidate = defaults.get(candidate_id)
        if (
            candidate is None
            or payload != candidate.to_dict()
            or candidate_id in seen
        ):
            raise DynamicCropV3CapacityFoldLauncherError(
                "plan candidate is not one unique exact v3 default"
            )
        seen.add(candidate_id)
        selected.append(candidate)
    if not selected:
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 capacity plan has no candidates"
        )
    return tuple(selected)


def _admitted_smokes_from_plan(
    value: object,
    *,
    candidates: Sequence[DynamicCropV3CapacityCandidate],
    source_provenance: Mapping[str, Any],
) -> dict[str, _AdmittedCapacitySmoke]:
    records = _array(value, "v3 capacity smoke records")
    paths: list[Path] = []
    for raw in records:
        record = _mapping(raw, "v3 capacity smoke record")
        pointer = _mapping(record.get("evidence"), "v3 capacity smoke pointer")
        try:
            paths.append(
                v2_launcher._verify_pointer(
                    pointer, "Dynamic Crop v3 capacity smoke"
                )
            )
        except (OSError, ValueError) as exc:
            raise DynamicCropV3CapacityFoldLauncherError(str(exc)) from exc
    admitted = _admit_capacity_smokes(
        paths,
        expected_source_provenance=source_provenance,
    )
    expected_ids = {candidate.candidate_id for candidate in candidates}
    if set(admitted) != expected_ids:
        raise DynamicCropV3CapacityFoldLauncherError(
            "plan capacity smoke set differs from candidates"
        )
    replayed = [admitted[candidate.candidate_id].plan_record() for candidate in candidates]
    source_commit = str(source_provenance.get("head_commit_oid") or "")
    if records != replayed or any(
        smoke.source_commit != source_commit for smoke in admitted.values()
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "plan capacity smoke admission differs on replay"
        )
    return admitted


def _manifest_maps_from_plan(
    songs: object,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    try:
        return v2_launcher._manifest_maps_from_plan(songs)
    except (OSError, ValueError) as exc:
        raise DynamicCropV3CapacityFoldLauncherError(str(exc)) from exc


def _select_candidate(
    candidates: Sequence[DynamicCropV3CapacityCandidate], selector: str
) -> DynamicCropV3CapacityCandidate:
    matches = tuple(
        candidate
        for candidate in candidates
        if selector in {candidate.candidate_name, candidate.candidate_id}
    )
    if len(matches) != 1:
        raise DynamicCropV3CapacityFoldLauncherError(
            "candidate selector does not identify one planned v3 candidate"
        )
    return matches[0]


def preflight_dynamic_crop_v3_capacity_fold(
    plan_path: Path,
    *,
    candidate_selector: str,
    held_out_song_id: str,
) -> DynamicCropV3CapacityFoldPreflight:
    if held_out_song_id not in SELECTION_SONG_IDS:
        raise DynamicCropV3CapacityFoldLauncherError(
            "held-out song is not one fixed capacity song"
        )
    plan = _strict_plan(plan_path)
    source = _mapping(plan["source_provenance"], "source provenance")
    candidates = _candidates_from_plan(plan.get("candidates"))
    admitted = _admitted_smokes_from_plan(
        plan.get("capacity_smokes"),
        candidates=candidates,
        source_provenance=source,
    )
    candidate = _select_candidate(candidates, candidate_selector)
    checkpoint_root = _paths_under_external_volume(
        Path(str(plan.get("checkpoint_root") or "")), "checkpoint root"
    )
    evidence_directory = _paths_under_external_volume(
        Path(str(plan.get("evidence_directory") or "")), "evidence directory"
    )
    try:
        dataset_path = v2_launcher._verify_pointer(
            plan.get("selection_dataset_manifest"),
            "selection dataset manifest",
        )
    except (OSError, ValueError) as exc:
        raise DynamicCropV3CapacityFoldLauncherError(str(exc)) from exc
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
        or [dict(record) for record in manifests.song_records]
        != plan.get("songs")
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "current manifest graph differs from immutable v3 plan"
        )
    repository_root = Path(str(source["repository_root"]))
    expected_runs: list[dict[str, Any]] = []
    replayed_args: dict[tuple[str, str], tuple[str, ...]] = {}
    for planned_candidate in candidates:
        for heldout in SELECTION_SONG_IDS:
            checkpoint = checkpoint_root / _run_id(planned_candidate, heldout)
            args = _build_train_cli_args(
                manifests,
                candidate=planned_candidate,
                held_out_song_id=heldout,
                checkpoint_directory=checkpoint,
            )
            replayed_args[(planned_candidate.candidate_id, heldout)] = args
            expected_runs.append(
                {
                    "run_id": _run_id(planned_candidate, heldout),
                    "candidate_name": planned_candidate.candidate_name,
                    "candidate_id": planned_candidate.candidate_id,
                    "capacity_smoke_content_sha256": admitted[
                        planned_candidate.candidate_id
                    ].payload["artifact_content_sha256"],
                    "held_out_song_id": heldout,
                    "training_song_ids": sorted(
                        set(SELECTION_SONG_IDS) - {heldout}
                    ),
                    "checkpoint_directory": str(checkpoint),
                    "evidence_directory": str(evidence_directory),
                    "train_cli_args": list(args),
                    "command_argv": _command_argv(
                        args, repository_root=repository_root
                    ),
                    "dynamic_crop_version": 3,
                    "activation_checkpointing": dict(
                        V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE
                    ),
                    "fold_selection_authorized": False,
                    "capacity_selection_authorized": False,
                    "production_activation_authorized": False,
                    "production_registry_activation_requested": False,
                }
            )
    if plan.get("runs") != expected_runs:
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 capacity run commands differ from replay"
        )
    checkpoint = checkpoint_root / _run_id(candidate, held_out_song_id)
    if checkpoint.exists() or checkpoint.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite v3 checkpoint directory: {checkpoint}"
        )
    _require_same_source_provenance(source)
    provenance_verification = _verify_dynamic_crop_provenance_inputs(manifests)
    try:
        free_bytes = v2_launcher.legacy._validate_storage(checkpoint)
    except (OSError, ValueError) as exc:
        raise DynamicCropV3CapacityFoldLauncherError(str(exc)) from exc
    run = next(
        item
        for item in expected_runs
        if item["candidate_id"] == candidate.candidate_id
        and item["held_out_song_id"] == held_out_song_id
    )
    return DynamicCropV3CapacityFoldPreflight(
        plan_path=Path(plan_path).resolve(),
        plan_content_sha256=str(plan["plan_content_sha256"]),
        candidate_name=candidate.candidate_name,
        candidate_id=candidate.candidate_id,
        capacity_smoke_path=admitted[candidate.candidate_id].path,
        held_out_song_id=held_out_song_id,
        training_song_ids=tuple(
            sorted(set(SELECTION_SONG_IDS) - {held_out_song_id})
        ),
        checkpoint_directory=checkpoint,
        evidence_directory=evidence_directory,
        selection_dataset_manifest_path=dataset_path,
        reference_ids_by_song=dict(manifests.reference_ids_by_song),
        train_cli_args=replayed_args[(candidate.candidate_id, held_out_song_id)],
        command_argv=tuple(run["command_argv"]),
        free_bytes=free_bytes,
        dynamic_crop_provenance_verification=provenance_verification,
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return _mapping(
            json.loads(path.read_text(encoding="utf-8")), label
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DynamicCropV3CapacityFoldLauncherError(
            f"{label} is invalid: {exc}"
        ) from exc


def _file_sha256(path: Path) -> str:
    try:
        return v2_launcher._stable_file_sha256(path, str(path))
    except (OSError, ValueError) as exc:
        raise DynamicCropV3CapacityFoldLauncherError(str(exc)) from exc


def _validate_completed_v3_fold(
    preflight: DynamicCropV3CapacityFoldPreflight,
) -> tuple[dict[str, Any], dict[str, Any], StageEditDynamicCropCheckpointMetadataV3]:
    candidate = _candidate_defaults_by_id().get(preflight.candidate_id)
    if candidate is None or candidate.candidate_name != preflight.candidate_name:
        raise DynamicCropV3CapacityFoldLauncherError(
            "preflight candidate is not one exact v3 default"
        )
    checkpoint = preflight.checkpoint_directory
    report_path = checkpoint / PRODUCTION_TRAINING_REPORT_FILENAME
    freeze_path = checkpoint / STAGE_EDIT_SELECTION_FREEZE_FILENAME
    metadata_path = checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
    weights_path = checkpoint / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
    for path in (report_path, freeze_path, metadata_path, weights_path):
        _file_sha256(path)
    report = _read_json(report_path, "v3 production training report")
    freeze = _read_json(freeze_path, "v3 selection freeze")
    try:
        metadata = StageEditDynamicCropCheckpointMetadataV3.from_dict(
            _read_json(metadata_path, "v3 checkpoint metadata")
        )
    except (TypeError, ValueError) as exc:
        raise DynamicCropV3CapacityFoldLauncherError(
            f"v3 checkpoint metadata differs: {exc}"
        ) from exc
    if (
        metadata.base_model_config != candidate.model_config
        or metadata.dynamic_crop_config != DynamicCropHeadV3Config()
        or not metadata.weights_trained
        or metadata.training_steps < 1
        or metadata.weight_filename != DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
        or metadata.weight_sha256 != _file_sha256(weights_path)
        or metadata.model_contract != DYNAMIC_CROP_V3_MODEL_CONTRACT
        or metadata.target_contract != DYNAMIC_CROP_V3_TARGET_CONTRACT
        or metadata.output_contract != DYNAMIC_CROP_V3_OUTPUT_CONTRACT
        or metadata.loss_contract != DYNAMIC_CROP_V3_LOSS_CONTRACT
        or metadata.production_activation_authorized is not False
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 checkpoint files/contracts differ"
        )
    if (
        report.get("schema_version") != PRODUCTION_TRAINING_REPORT_SCHEMA_VERSION
        or report.get("run_kind") != PRODUCTION_TRAINING_RUN_KIND
        or report.get("checkpoint_scope") != PRODUCTION_CHECKPOINT_SCOPE
        or report.get("production_ready") is not False
        or report.get("production_readiness_reason")
        != "activation_not_explicitly_authorized"
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 inactive production report contract differs"
        )
    model = _mapping(report.get("model"), "v3 report model")
    if (
        model.get("config") != candidate.model_config.to_dict()
        or model.get("dynamic_crop_version") != 3
        or model.get("dynamic_crop_v3_enabled") is not True
        or model.get("dynamic_crop_v2_enabled") is not False
        or model.get("dynamic_crop_model_contract")
        != DYNAMIC_CROP_V3_MODEL_CONTRACT
        or model.get("dynamic_crop_target_contract")
        != DYNAMIC_CROP_V3_TARGET_CONTRACT
        or model.get("dynamic_crop_output_contract")
        != DYNAMIC_CROP_V3_OUTPUT_CONTRACT
        or not _exact_activation_checkpointing(
            model.get("activation_checkpointing")
        )
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 report model is mixed or stale"
        )
    split = _mapping(report.get("split"), "v3 report split")
    expected_training = list(preflight.training_song_ids)
    expected_all_training_references = sorted(
        reference
        for song in preflight.training_song_ids
        for reference in preflight.reference_ids_by_song.get(song, ())
    )
    expected_validation_references = sorted(
        min(
            preflight.reference_ids_by_song.get(song, ()),
            key=lambda reference: hashlib.sha256(
                f"{song}\0{reference}".encode("utf-8")
            ).hexdigest(),
        )
        for song in preflight.training_song_ids
    )
    expected_training_references = sorted(
        set(expected_all_training_references)
        - set(expected_validation_references)
    )
    expected_heldout_references = sorted(
        preflight.reference_ids_by_song.get(preflight.held_out_song_id, ())
    )
    observed_train_references = sorted(split.get("training_reference_ids") or ())
    observed_validation_references = sorted(
        split.get("validation_reference_ids") or ()
    )
    if (
        split.get("training_song_ids") != expected_training
        or split.get("held_out_song_ids") != [preflight.held_out_song_id]
        or split.get("held_out_used_for_model_selection") is not False
        or observed_train_references != expected_training_references
        or observed_validation_references != expected_validation_references
        or set(observed_train_references) & set(observed_validation_references)
        or sorted((*observed_train_references, *observed_validation_references))
        != expected_all_training_references
        or sorted(split.get("held_out_reference_ids") or ())
        != expected_heldout_references
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 whole-song split contract differs"
        )
    optimization = _mapping(report.get("optimization"), "v3 optimization")
    schedule = _mapping(
        optimization.get("learning_rate_schedule"), "v3 learning schedule"
    )
    if (
        optimization.get("source_pretraining_optimizer_steps")
        != PROBE_TRAINING_SCHEDULE["source_pretraining_steps"]
        or schedule.get("source_pretraining_steps")
        != PROBE_TRAINING_SCHEDULE["source_pretraining_steps"]
        or schedule.get("source_pretraining_peak_learning_rate")
        != PROBE_TRAINING_SCHEDULE["source_pretraining_peak_learning_rate"]
        or schedule.get("source_pretraining_window_length")
        != PROBE_TRAINING_SCHEDULE["source_pretraining_window_length"]
        or schedule.get("source_pretraining_warmup_steps")
        != PROBE_TRAINING_SCHEDULE["source_pretraining_warmup_steps"]
        or optimization.get(
            "dynamic_crop_v3_prior_pretraining_optimizer_steps"
        )
        != PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_pretraining_steps"
        ]
        or schedule.get("dynamic_crop_v3_prior_pretraining_steps")
        != PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_pretraining_steps"
        ]
        or schedule.get(
            "dynamic_crop_v3_prior_pretraining_peak_learning_rate"
        )
        != PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_pretraining_peak_learning_rate"
        ]
        or schedule.get(
            "dynamic_crop_v3_prior_pretraining_warmup_steps"
        )
        != PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_pretraining_warmup_steps"
        ]
        or schedule.get("dynamic_crop_v3_prior_retention_finetune_steps")
        != PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_retention_finetune_steps"
        ]
        or schedule.get("contract")
        != PROBE_TRAINING_SCHEDULE["schedule_contract"]
        or optimization.get("minimum_finetune_optimizer_steps")
        != PROBE_TRAINING_SCHEDULE["minimum_finetune_optimizer_steps"]
        or optimization.get("minimum_checkpoint_selection_finetune_steps")
        != PROBE_TRAINING_SCHEDULE[
            "minimum_checkpoint_selection_finetune_steps"
        ]
        or optimization.get("unseen_song_used_for_model_selection") is not False
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 source-pretrain/finetune schedule differs"
        )
    cut_steps = int(PROBE_TRAINING_SCHEDULE["cut_pretraining_steps"])
    source_steps = int(PROBE_TRAINING_SCHEDULE["source_pretraining_steps"])
    prior_steps = int(
        PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_pretraining_steps"
        ]
    )
    finetune_steps = int(PROBE_TRAINING_SCHEDULE["finetune_steps_planned"])
    best_finetune_step = optimization.get("best_finetune_optimizer_step")
    if (
        optimization.get("optimizer_steps")
        != cut_steps + source_steps + prior_steps + finetune_steps
        or optimization.get("dynamic_crop_v3_prior_optimizer_initial_step")
        != 0
        or optimization.get(
            "dynamic_crop_v3_prior_optimizer_state_final_step"
        )
        != prior_steps
        or optimization.get(
            "dynamic_crop_v3_prior_optimizer_state_reset_after_source_pretraining"
        )
        is not True
        or optimization.get(
            "finetune_optimizer_state_reset_after_dynamic_crop_v3_prior_pretraining"
        )
        is not True
        or optimization.get(
            "finetune_retention_optimizer_state_final_step"
        )
        != PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_retention_finetune_steps"
        ]
        or optimization.get("finetune_joint_optimizer_initial_step") != 0
        or optimization.get(
            "finetune_optimizer_reset_at_dynamic_crop_v3_prior_release"
        )
        is not True
        or optimization.get("finetune_optimizer_state_final_step")
        != finetune_steps
        - PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_retention_finetune_steps"
        ]
        or optimization.get("checkpoint_writes_before_joint_finetuning") != 0
        or isinstance(best_finetune_step, bool)
        or not isinstance(best_finetune_step, int)
        or best_finetune_step
        < PROBE_TRAINING_SCHEDULE[
            "minimum_checkpoint_selection_finetune_steps"
        ]
        or optimization.get("best_training_step")
        != cut_steps + source_steps + prior_steps + best_finetune_step
        or metadata.training_steps != optimization.get("best_training_step")
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 prior/global optimizer-step evidence differs"
        )
    prior = _mapping(
        optimization.get("dynamic_crop_v3_prior_pretraining"),
        "v3 prior pretraining",
    )
    sampling = _mapping(prior.get("sampling"), "v3 prior sampling")
    component_scale = _mapping(
        prior.get("component_scale"), "v3 prior component scale"
    )
    invariance = _mapping(
        prior.get("parameter_invariance"), "v3 prior invariance"
    )
    candidate_selection = _mapping(
        prior.get("candidate_selection"), "v3 prior candidate selection"
    )
    retention = _mapping(prior.get("retention"), "v3 prior retention")
    warmup_selection = _mapping(
        candidate_selection.get("selection"), "v3 prior validation selection"
    )
    warmup_gate = _mapping(
        warmup_selection.get("dynamic_crop_v3_selection_gate"),
        "v3 prior validation gate",
    )
    candidate_evaluations = candidate_selection.get("candidate_evaluations")
    selected_candidate_step = candidate_selection.get(
        "selected_candidate_step"
    )
    selected_candidate_digest = candidate_selection.get(
        "selected_candidate_value_bias_sha256"
    )
    candidate_validation_reference_ids = candidate_selection.get(
        "validation_reference_ids"
    )
    invariance_before = invariance.get("before")
    invariance_after = invariance.get("after")
    invariant_parameter_groups = (
        "base_parameter_tree_sha256",
        "context_weight_sha256",
        "source_head_parameter_tree_sha256",
        "context_non_value_bias_sha256",
    )
    all_prior_parameter_groups = (
        *invariant_parameter_groups,
        "context_value_bias_sha256",
    )
    if (
        prior.get("enabled") is not True
        or prior.get("partition") != "training_references_only"
        or prior.get("optimizer_weight_decay") != 0.0
        or prior.get("checkpoint_selection_eligible_during_warmup") is not False
        or prior.get("checkpoint_write_before_joint_finetuning") is not False
        or prior.get("validation_labels_consumed_for_gradients") is not False
        or prior.get("heldout_labels_consumed") is not False
        or sampling.get("partition") != "training_references_only"
        or sampling.get("validation_labels_consumed") is not False
        or sampling.get("heldout_labels_consumed") is not False
        or component_scale.get("validation_labels_consumed") is not False
        or component_scale.get("heldout_labels_consumed") is not False
        or invariance.get(
            "all_declared_invariant_groups_byte_identical"
        )
        is not True
        or invariance.get("context_value_bias_changed") is not True
        or candidate_selection.get("training_partition_only_gradients")
        is not True
        or candidate_selection.get("contract")
        != DYNAMIC_CROP_V3_PRIOR_CANDIDATE_SELECTION_CONTRACT
        or candidate_selection.get("predeclared_candidate_steps")
        != list(DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS)
        or candidate_selection.get("selection_rule")
        != (
            "highest_finite_selectable_existing_validation_score_"
            "then_lower_candidate_step_on_exact_tie"
        )
        or candidate_selection.get("validation_gradients_computed") is not False
        or candidate_selection.get("heldout_forward_performed") is not False
        or not isinstance(candidate_validation_reference_ids, list)
        or any(
            not isinstance(reference_id, str) or not reference_id
            for reference_id in candidate_validation_reference_ids
        )
        or sorted(candidate_validation_reference_ids)
        != observed_validation_references
        or len(candidate_validation_reference_ids)
        != len(set(candidate_validation_reference_ids))
        or candidate_selection.get("retention_authorized") is not True
        or not isinstance(candidate_evaluations, list)
        or len(candidate_evaluations)
        != len(DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS)
        or isinstance(selected_candidate_step, bool)
        or selected_candidate_step not in DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS
        or not _is_sha256(selected_candidate_digest)
        or warmup_selection.get("selectable") is not True
        or warmup_gate.get("selectable") is not True
        or not isinstance(warmup_gate.get("score_component"), (int, float))
        or isinstance(warmup_gate.get("score_component"), bool)
        or not math.isfinite(float(warmup_gate.get("score_component")))
        or warmup_gate.get("score_component") <= 0.0
        or retention.get("retention_finetune_step_count")
        != PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_retention_finetune_steps"
        ]
        or retention.get("required_retention_finetune_step_count")
        != PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_retention_finetune_steps"
        ]
        or retention.get("all_dynamic_head_parameters_byte_identical")
        is not True
        or retention.get("optimizer_state_reset_before_dynamic_head_release")
        is not True
        or retention.get("retention_optimizer_state_final_step")
        != PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_retention_finetune_steps"
        ]
        or retention.get("released_joint_optimizer_initial_step") != 0
        or retention.get("released_joint_optimizer_state_final_step")
        != finetune_steps
        - PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_retention_finetune_steps"
        ]
        or retention.get("released_joint_optimizer_schedule_offset_steps")
        != PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_retention_finetune_steps"
        ]
        or retention.get("validation_gradients_computed") is not False
        or retention.get("heldout_forward_performed") is not False
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 train-only prior warmup evidence differs"
        )
    if (
        not isinstance(invariance_before, Mapping)
        or not isinstance(invariance_after, Mapping)
        or set(invariance_before) != set(all_prior_parameter_groups)
        or set(invariance_after) != set(all_prior_parameter_groups)
        or invariance.get("invariant_parameter_groups")
        != list(invariant_parameter_groups)
        or any(
            not _is_sha256(invariance_before.get(key))
            or not _is_sha256(invariance_after.get(key))
            for key in all_prior_parameter_groups
        )
        or any(
            invariance_before.get(key) != invariance_after.get(key)
            for key in invariant_parameter_groups
        )
        or invariance_before.get("context_value_bias_sha256")
        == invariance_after.get("context_value_bias_sha256")
        or invariance_after.get("context_value_bias_sha256")
        != selected_candidate_digest
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 prior parameter invariance digests differ"
        )
    retained_head_groups = (
        "context_weight_sha256",
        "source_head_parameter_tree_sha256",
        "context_non_value_bias_sha256",
        "context_value_bias_sha256",
    )
    warmup_head_digests = retention.get("warmup_head_digests")
    retained_head_digests = retention.get("retained_head_digests")
    if (
        not isinstance(warmup_head_digests, Mapping)
        or not isinstance(retained_head_digests, Mapping)
        or set(warmup_head_digests) != set(retained_head_groups)
        or set(retained_head_digests) != set(retained_head_groups)
        or any(
            not _is_sha256(warmup_head_digests.get(key))
            or not _is_sha256(retained_head_digests.get(key))
            for key in retained_head_groups
        )
        or any(
            warmup_head_digests.get(key) != retained_head_digests.get(key)
            or warmup_head_digests.get(key) != invariance_after.get(key)
            for key in retained_head_groups
        )
        or retained_head_digests.get("context_value_bias_sha256")
        != selected_candidate_digest
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 prior retention byte digests differ"
        )
    assert isinstance(candidate_evaluations, list)
    selectable_candidate_records: list[tuple[float, int, dict[str, Any]]] = []
    candidate_digest_by_step: dict[int, str] = {}
    for expected_step, raw_candidate in zip(
        DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS,
        candidate_evaluations,
        strict=True,
    ):
        candidate_record = _mapping(
            raw_candidate, f"v3 prior candidate {expected_step}"
        )
        candidate_step = candidate_record.get("candidate_step")
        candidate_digest = candidate_record.get("value_bias_sha256")
        selection = _mapping(
            candidate_record.get("selection"),
            f"v3 prior candidate {expected_step} selection",
        )
        validation_metrics = _mapping(
            candidate_record.get("validation_metrics"),
            f"v3 prior candidate {expected_step} validation metrics",
        )
        dynamic_metrics = _mapping(
            validation_metrics.get("dynamic_crop_v3"),
            f"v3 prior candidate {expected_step} dynamic metrics",
        )
        recomputed_selection = (
            _stage_edit_validation_selection_with_dynamic_crop_v3(
                stage_edit_validation_selection_score(validation_metrics),
                dynamic_metrics,
            )
        )
        gate = _mapping(
            selection.get("dynamic_crop_v3_selection_gate"),
            f"v3 prior candidate {expected_step} gate",
        )
        score = selection.get("score")
        score_is_finite = bool(
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(float(score))
        )
        if (
            candidate_step != expected_step
            or not _is_sha256(candidate_digest)
            or candidate_record.get("training_partition_only_updates") is not True
            or candidate_record.get("validation_gradients_computed") is not False
            or candidate_record.get("heldout_forward_performed") is not False
            or selection != recomputed_selection
            or gate.get("heldout_metrics_consumed") is not False
        ):
            raise DynamicCropV3CapacityFoldLauncherError(
                "v3 prior candidate evidence differs"
            )
        candidate_digest_by_step[expected_step] = str(candidate_digest)
        if selection.get("selectable") is True:
            gate_component = gate.get("score_component")
            if (
                gate.get("selectable") is not True
                or not score_is_finite
                or not isinstance(gate_component, (int, float))
                or isinstance(gate_component, bool)
                or not math.isfinite(float(gate_component))
                or float(gate_component) <= 0.0
            ):
                raise DynamicCropV3CapacityFoldLauncherError(
                    "v3 selectable prior candidate gate differs"
                )
            selectable_candidate_records.append(
                (float(score), expected_step, candidate_record)
            )
    if not selectable_candidate_records:
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 prior candidate set has no selectable milestone"
        )
    expected_selected = max(
        selectable_candidate_records,
        key=lambda item: (item[0], -item[1]),
    )
    if (
        selected_candidate_step != expected_selected[1]
        or selected_candidate_digest
        != candidate_digest_by_step[expected_selected[1]]
        or warmup_selection != expected_selected[2].get("selection")
        or candidate_selection.get("validation_metrics")
        != expected_selected[2].get("validation_metrics")
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 prior candidate selection is not the deterministic maximum"
        )
    prior_history = report.get("dynamic_crop_v3_prior_pretraining_history")
    if not isinstance(prior_history, list) or len(prior_history) != prior_steps:
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 prior pretraining history is incomplete"
        )
    observed_song_samples: dict[str, int] = {}
    observed_reference_samples: dict[str, int] = {}
    observed_prior_learning_rates: list[float] = []
    for step, raw_record in enumerate(prior_history, start=1):
        record = _mapping(raw_record, f"v3 prior history step {step}")
        song_id = str(record.get("song_id") or "")
        reference_id = str(record.get("reference_id") or "")
        is_candidate_step = step in DYNAMIC_CROP_V3_PRIOR_CANDIDATE_STEPS
        if (
            record.get("prior_pretrain_step") != step
            or record.get("optimizer_step")
            != cut_steps + source_steps + step
            or record.get("checkpoint_selection_eligible") is not False
            or record.get("validation_labels_consumed") is not False
            or record.get("heldout_labels_consumed") is not False
            or record.get("predeclared_candidate_snapshot")
            is not is_candidate_step
            or (
                record.get("candidate_value_bias_sha256")
                != candidate_digest_by_step[step]
                if is_candidate_step
                else record.get("candidate_value_bias_sha256") is not None
            )
            or not song_id
            or not reference_id
            or isinstance(record.get("actual_learning_rate"), bool)
            or not isinstance(record.get("actual_learning_rate"), (int, float))
            or not math.isfinite(float(record.get("actual_learning_rate")))
            or float(record.get("actual_learning_rate")) <= 0.0
        ):
            raise DynamicCropV3CapacityFoldLauncherError(
                "v3 prior history step differs"
            )
        observed_song_samples[song_id] = (
            observed_song_samples.get(song_id, 0) + 1
        )
        observed_reference_samples[reference_id] = (
            observed_reference_samples.get(reference_id, 0) + 1
        )
        observed_prior_learning_rates.append(
            float(record["actual_learning_rate"])
        )
    expected_reference_song_ids = {
        reference_id: song_id
        for song_id in preflight.training_song_ids
        for reference_id in preflight.reference_ids_by_song.get(song_id, ())
        if reference_id in observed_train_references
    }
    if (
        observed_song_samples != sampling.get("per_song_sampled_counts")
        or observed_reference_samples
        != sampling.get("per_reference_sampled_counts")
        or set(observed_song_samples) != set(preflight.training_song_ids)
        or set(observed_reference_samples) != set(observed_train_references)
        or sampling.get("reference_song_ids")
        != expected_reference_song_ids
        or max(observed_song_samples.values())
        - min(observed_song_samples.values())
        > 1
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 prior song/reference sampling evidence differs"
        )
    prior_peak_learning_rate = float(
        PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_pretraining_peak_learning_rate"
        ]
    )
    prior_warmup_steps = int(
        PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_pretraining_warmup_steps"
        ]
    )
    prior_minimum_rate = prior_peak_learning_rate * float(
        PROBE_TRAINING_SCHEDULE["minimum_learning_rate_ratio"]
    )
    prior_decay_steps = prior_steps - prior_warmup_steps

    def expected_prior_learning_rate(schedule_index: int) -> float:
        if schedule_index < prior_warmup_steps:
            start = prior_peak_learning_rate * 0.1
            return start + (prior_peak_learning_rate - start) * (
                schedule_index / prior_warmup_steps
            )
        local = min(schedule_index - prior_warmup_steps, prior_decay_steps)
        cosine = 0.5 * (
            1.0 + math.cos(math.pi * local / prior_decay_steps)
        )
        return prior_minimum_rate + (
            prior_peak_learning_rate - prior_minimum_rate
        ) * cosine

    if any(
        not math.isclose(
            observed,
            expected_prior_learning_rate(step - 1),
            rel_tol=1.0e-6,
            abs_tol=1.0e-12,
        )
        for step, observed in enumerate(
            observed_prior_learning_rates, start=1
        )
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 prior learning-rate history differs"
        )
    finetune_history = report.get("history")
    if not isinstance(finetune_history, list) or not finetune_history:
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 finetune optimizer history is incomplete"
        )
    observed_finetune_steps: list[int] = []
    observed_phase_local_steps: list[int] = []
    observed_reset_before_steps: list[int] = []
    observed_retention_steps: list[int] = []
    observed_learning_rates: list[float] = []
    for epoch_index, raw_epoch in enumerate(finetune_history, start=1):
        epoch_record = _mapping(
            raw_epoch, f"v3 finetune history epoch {epoch_index}"
        )
        for destination, key in (
            (observed_finetune_steps, "finetune_optimizer_steps"),
            (
                observed_phase_local_steps,
                "finetune_phase_local_optimizer_steps",
            ),
            (
                observed_reset_before_steps,
                "finetune_optimizer_reset_before_steps",
            ),
            (
                observed_retention_steps,
                "dynamic_crop_v3_prior_retention_finetune_steps",
            ),
        ):
            values = epoch_record.get(key)
            if not isinstance(values, list) or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in values
            ):
                raise DynamicCropV3CapacityFoldLauncherError(
                    f"v3 finetune history {key} differs"
                )
            destination.extend(values)
        raw_learning_rates = epoch_record.get(
            "finetune_actual_learning_rates"
        )
        if not isinstance(raw_learning_rates, list) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in raw_learning_rates
        ):
            raise DynamicCropV3CapacityFoldLauncherError(
                "v3 finetune learning-rate history differs"
            )
        observed_learning_rates.extend(float(value) for value in raw_learning_rates)
    expected_retention_steps = int(
        PROBE_TRAINING_SCHEDULE[
            "dynamic_crop_v3_prior_retention_finetune_steps"
        ]
    )
    expected_phase_local_steps = [
        *range(1, expected_retention_steps + 1),
        *range(1, finetune_steps - expected_retention_steps + 1),
    ]
    peak_learning_rate = float(PROBE_TRAINING_SCHEDULE["learning_rate"])
    warmup_steps = int(PROBE_TRAINING_SCHEDULE["finetune_warmup_steps"])
    minimum_rate = peak_learning_rate * float(
        PROBE_TRAINING_SCHEDULE["minimum_learning_rate_ratio"]
    )
    decay_steps = finetune_steps - warmup_steps

    def expected_learning_rate(schedule_index: int) -> float:
        if schedule_index < warmup_steps:
            start = peak_learning_rate * 0.1
            return start + (peak_learning_rate - start) * (
                schedule_index / warmup_steps
            )
        local = min(schedule_index - warmup_steps, decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * local / decay_steps))
        return minimum_rate + (peak_learning_rate - minimum_rate) * cosine

    expected_learning_rates = [
        expected_learning_rate(step - 1)
        for step in range(1, finetune_steps + 1)
    ]
    if (
        observed_finetune_steps != list(range(1, finetune_steps + 1))
        or observed_phase_local_steps != expected_phase_local_steps
        or observed_reset_before_steps != [expected_retention_steps + 1]
        or observed_retention_steps
        != list(range(1, expected_retention_steps + 1))
        or len(observed_learning_rates) != finetune_steps
        or any(
            not math.isclose(observed, expected, rel_tol=1.0e-6, abs_tol=1.0e-12)
            for observed, expected in zip(
                observed_learning_rates,
                expected_learning_rates,
                strict=True,
            )
        )
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 finetune optimizer reset history differs"
        )
    observed_dynamic_loss = _mapping(
        optimization.get("effective_dynamic_crop_v3_loss_contract"),
        "v3 effective loss",
    )
    expected_dynamic_loss = asdict(candidate.dynamic_crop_loss_contract)
    observed_base = _mapping(
        optimization.get("effective_loss_contract"), "v3 effective base loss"
    )
    expected_base = asdict(candidate.base_loss_contract)
    applied_positive_weight = observed_base.get("cut_positive_weight")
    expected_base["cut_positive_weight"] = applied_positive_weight
    expected_dynamic_loss["base"]["cut_positive_weight"] = (
        applied_positive_weight
    )
    if (
        observed_base != expected_base
        or observed_dynamic_loss != expected_dynamic_loss
        or optimization.get("effective_dynamic_crop_loss_contract_version") != 3
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 effective loss contract differs"
        )
    heldout = _mapping(report.get("held_out_evaluation"), "v3 heldout")
    metrics = _mapping(
        heldout.get("dynamic_crop_v3_metrics"), "v3 heldout metrics"
    )
    quality = _mapping(heldout.get("quality_gate"), "v3 heldout quality")
    if (
        heldout.get("passed") is not True
        or quality.get("passed") is not True
        or heldout.get("training_song_ids") != expected_training
        or heldout.get("held_out_song_ids") != [preflight.held_out_song_id]
        or heldout.get("held_out_reference_count")
        != len(expected_heldout_references)
        or heldout.get("checkpoint_id") != metadata.checkpoint_id
        or heldout.get("weight_sha256") != metadata.weight_sha256
        or heldout.get("cut_threshold_frozen_before_heldout_forward") is not True
        or heldout.get("heldout_dynamic_crop_metrics_used_for_checkpoint_selection")
        is not False
        or heldout.get("dynamic_crop_version") != 3
        or heldout.get("dynamic_crop_v2_metrics") is not None
        or metrics.get("schema_version") != DYNAMIC_CROP_V3_SCHEMA_VERSION
        or metrics.get("model_contract") != DYNAMIC_CROP_V3_MODEL_CONTRACT
        or metrics.get("target_contract") != DYNAMIC_CROP_V3_TARGET_CONTRACT
        or metrics.get("output_contract") != DYNAMIC_CROP_V3_OUTPUT_CONTRACT
        or metrics.get("loss_contract") != DYNAMIC_CROP_V3_LOSS_CONTRACT
        or metrics.get("legacy_v2_checkpoint_or_output_used") is not False
        or metrics.get("production_activation_authorized") is not False
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 heldout quality/evaluator contract differs"
        )
    checkpoint_report = _mapping(report.get("checkpoint"), "v3 checkpoint report")
    if (
        checkpoint_report.get("metadata_filename")
        != DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
        or checkpoint_report.get("weight_filename")
        != DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
        or checkpoint_report.get("checkpoint_id") != metadata.checkpoint_id
        or checkpoint_report.get("training_steps") != metadata.training_steps
        or checkpoint_report.get("weight_sha256") != metadata.weight_sha256
        or checkpoint_report.get("production_activation_authorized") is not False
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 report/checkpoint identity differs"
        )
    if (
        freeze.get("heldout_metrics_present") is not False
        or freeze.get("heldout_used_for_model_or_threshold_selection") is not False
        or freeze.get("held_out_song_ids_reserved_for_one_forward")
        != [preflight.held_out_song_id]
        or freeze.get("checkpoint_id") != metadata.checkpoint_id
        or freeze.get("checkpoint_weight_sha256") != metadata.weight_sha256
        or freeze.get("minimum_checkpoint_selection_finetune_steps")
        != PROBE_TRAINING_SCHEDULE[
            "minimum_checkpoint_selection_finetune_steps"
        ]
        or freeze.get("selected_after_checkpoint_burn_in") is not True
        or freeze.get("selected_training_step") != metadata.training_steps
        or freeze.get("selected_finetune_optimizer_step")
        != best_finetune_step
        or freeze.get("dynamic_crop_v3_prior_selected_candidate_step")
        != selected_candidate_step
        or freeze.get(
            "dynamic_crop_v3_prior_selected_candidate_value_bias_sha256"
        )
        != selected_candidate_digest
        or sorted(freeze.get("training_reference_ids") or ())
        != observed_train_references
        or sorted(freeze.get("validation_reference_ids") or ())
        != observed_validation_references
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 pre-heldout selection freeze differs"
        )
    return report, freeze, metadata


def _revalidate_preflight_receipt_binding(
    preflight: DynamicCropV3CapacityFoldPreflight,
) -> None:
    """Bind direct receipt publication back to its clean immutable plan."""

    plan = _strict_plan(preflight.plan_path)
    if plan.get("plan_content_sha256") != preflight.plan_content_sha256:
        raise DynamicCropV3CapacityFoldLauncherError(
            "preflight plan identity changed before fold publication"
        )
    source = _mapping(plan.get("source_provenance"), "source provenance")
    candidates = _candidates_from_plan(plan.get("candidates"))
    admitted = _admitted_smokes_from_plan(
        plan.get("capacity_smokes"),
        candidates=candidates,
        source_provenance=source,
    )
    candidate = _select_candidate(candidates, preflight.candidate_id)
    smoke = admitted[candidate.candidate_id]
    runs = _array(plan.get("runs"), "v3 capacity runs")
    matches = tuple(
        run
        for raw in runs
        for run in (_mapping(raw, "v3 capacity run"),)
        if run.get("candidate_id") == preflight.candidate_id
        and run.get("held_out_song_id") == preflight.held_out_song_id
    )
    try:
        planned_dataset = v2_launcher._verify_pointer(
            plan.get("selection_dataset_manifest"),
            "selection dataset manifest",
        )
    except (OSError, ValueError) as exc:
        raise DynamicCropV3CapacityFoldLauncherError(str(exc)) from exc
    if (
        len(matches) != 1
        or candidate.candidate_name != preflight.candidate_name
        or smoke.path != preflight.capacity_smoke_path
        or planned_dataset != preflight.selection_dataset_manifest_path
        or tuple(matches[0].get("training_song_ids") or ())
        != preflight.training_song_ids
        or matches[0].get("checkpoint_directory")
        != str(preflight.checkpoint_directory)
        or matches[0].get("evidence_directory")
        != str(preflight.evidence_directory)
        or tuple(matches[0].get("train_cli_args") or ())
        != preflight.train_cli_args
        or matches[0].get("fold_selection_authorized") is not False
        or matches[0].get("capacity_selection_authorized") is not False
        or matches[0].get("production_activation_authorized") is not False
        or matches[0].get("production_registry_activation_requested") is not False
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "preflight fold/smoke binding differs from immutable v3 plan"
        )
    _require_same_source_provenance(source)


def record_dynamic_crop_v3_capacity_fold_evidence(
    preflight: DynamicCropV3CapacityFoldPreflight,
) -> Path:
    """Publish one non-selecting v3 fold receipt after strict validation."""

    _revalidate_preflight_receipt_binding(preflight)
    _report, _freeze, metadata = _validate_completed_v3_fold(preflight)
    checkpoint = preflight.checkpoint_directory
    paths = {
        "plan": preflight.plan_path,
        "capacity_smoke": preflight.capacity_smoke_path,
        "selection_dataset": preflight.selection_dataset_manifest_path,
        "production_training_report": checkpoint
        / PRODUCTION_TRAINING_REPORT_FILENAME,
        "selection_freeze": checkpoint / STAGE_EDIT_SELECTION_FREEZE_FILENAME,
        "checkpoint_metadata": checkpoint
        / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
        "checkpoint_weights": checkpoint
        / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
    }
    pointers = {
        name: v2_launcher._pointer(path, name.replace("_", " "))
        for name, path in paths.items()
    }
    unsigned = {
        "schema_version": V3_CAPACITY_FOLD_EVIDENCE_SCHEMA_VERSION,
        "analysis_basis": V3_CAPACITY_FOLD_EVIDENCE_ANALYSIS_BASIS,
        "contract": V3_CAPACITY_FOLD_EVIDENCE_CONTRACT,
        "candidate_id": preflight.candidate_id,
        "candidate_name": preflight.candidate_name,
        "validation_song_id": preflight.held_out_song_id,
        "training_song_ids": list(preflight.training_song_ids),
        "dynamic_crop_version": 3,
        "model_contract": DYNAMIC_CROP_V3_MODEL_CONTRACT,
        "target_contract": DYNAMIC_CROP_V3_TARGET_CONTRACT,
        "output_contract": DYNAMIC_CROP_V3_OUTPUT_CONTRACT,
        "loss_contract": DYNAMIC_CROP_V3_LOSS_CONTRACT,
        "activation_checkpointing": dict(
            V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE
        ),
        "checkpoint_id": metadata.checkpoint_id,
        "checkpoint_weight_sha256": metadata.weight_sha256,
        **pointers,
        "fold_selection_authorized": False,
        "capacity_selection_authorized": False,
        "production_activation_authorized": False,
        "production_registry_activation_authorized": False,
        "all_three_fold_receipts_required_before_future_selection": True,
        "artifact_content_sha256": None,
    }
    digest = _canonical_sha256(unsigned)
    payload = {**unsigned, "artifact_content_sha256": digest}
    destination = _paths_under_external_volume(
        preflight.evidence_directory, "fold evidence directory"
    )
    return _atomic_exclusive_write(
        destination / f"{V3_CAPACITY_FOLD_EVIDENCE_PREFIX}{digest}.json",
        payload,
    )


_FOLD_EVIDENCE_KEYS = {
    "schema_version",
    "analysis_basis",
    "contract",
    "candidate_id",
    "candidate_name",
    "validation_song_id",
    "training_song_ids",
    "dynamic_crop_version",
    "model_contract",
    "target_contract",
    "output_contract",
    "loss_contract",
    "activation_checkpointing",
    "checkpoint_id",
    "checkpoint_weight_sha256",
    "plan",
    "capacity_smoke",
    "selection_dataset",
    "production_training_report",
    "selection_freeze",
    "checkpoint_metadata",
    "checkpoint_weights",
    "fold_selection_authorized",
    "capacity_selection_authorized",
    "production_activation_authorized",
    "production_registry_activation_authorized",
    "all_three_fold_receipts_required_before_future_selection",
    "artifact_content_sha256",
}


def load_dynamic_crop_v3_capacity_fold_evidence(path: Path) -> dict[str, Any]:
    try:
        resolved = v2_launcher._regular_safe_path(path, "v3 fold evidence")
        payload = v2_launcher._read_json_object(resolved, "v3 fold evidence")
    except (OSError, ValueError) as exc:
        raise DynamicCropV3CapacityFoldLauncherError(str(exc)) from exc
    if set(payload) != _FOLD_EVIDENCE_KEYS:
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 fold evidence keys are not exact"
        )
    digest = payload.get("artifact_content_sha256")
    unsigned = deepcopy(payload)
    unsigned["artifact_content_sha256"] = None
    if (
        not _is_sha256(digest)
        or _canonical_sha256(unsigned) != digest
        or resolved.name != f"{V3_CAPACITY_FOLD_EVIDENCE_PREFIX}{digest}.json"
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 fold evidence is not content-addressed"
        )
    if (
        payload.get("schema_version")
        != V3_CAPACITY_FOLD_EVIDENCE_SCHEMA_VERSION
        or payload.get("analysis_basis")
        != V3_CAPACITY_FOLD_EVIDENCE_ANALYSIS_BASIS
        or payload.get("contract") != V3_CAPACITY_FOLD_EVIDENCE_CONTRACT
        or payload.get("dynamic_crop_version") != 3
        or payload.get("model_contract") != DYNAMIC_CROP_V3_MODEL_CONTRACT
        or payload.get("target_contract") != DYNAMIC_CROP_V3_TARGET_CONTRACT
        or payload.get("output_contract") != DYNAMIC_CROP_V3_OUTPUT_CONTRACT
        or payload.get("loss_contract") != DYNAMIC_CROP_V3_LOSS_CONTRACT
        or not _exact_activation_checkpointing(
            payload.get("activation_checkpointing")
        )
        or payload.get("fold_selection_authorized") is not False
        or payload.get("capacity_selection_authorized") is not False
        or payload.get("production_activation_authorized") is not False
        or payload.get("production_registry_activation_authorized") is not False
        or payload.get("all_three_fold_receipts_required_before_future_selection")
        is not True
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 fold evidence contract or authority differs"
        )
    candidate = _candidate_defaults_by_id().get(
        str(payload.get("candidate_id") or "")
    )
    if candidate is None or candidate.candidate_name != payload.get("candidate_name"):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 fold evidence candidate is not exact"
        )
    if payload.get("validation_song_id") not in SELECTION_SONG_IDS or payload.get(
        "training_song_ids"
    ) != sorted(
        set(SELECTION_SONG_IDS) - {payload.get("validation_song_id")}
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 fold evidence song split differs"
        )
    verified_paths: dict[str, Path] = {}
    for name in (
        "plan",
        "capacity_smoke",
        "selection_dataset",
        "production_training_report",
        "selection_freeze",
        "checkpoint_metadata",
        "checkpoint_weights",
    ):
        try:
            verified_paths[name] = v2_launcher._verify_pointer(
                payload.get(name), name.replace("_", " ")
            )
        except (OSError, ValueError) as exc:
            raise DynamicCropV3CapacityFoldLauncherError(str(exc)) from exc
    if (
        verified_paths["production_training_report"].name
        != PRODUCTION_TRAINING_REPORT_FILENAME
        or verified_paths["selection_freeze"].name
        != STAGE_EDIT_SELECTION_FREEZE_FILENAME
        or verified_paths["checkpoint_metadata"].name
        != DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME
        or verified_paths["checkpoint_weights"].name
        != DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 fold evidence checkpoint filenames differ"
        )
    plan = _strict_plan(verified_paths["plan"])
    source = _mapping(plan.get("source_provenance"), "source provenance")
    candidates = _candidates_from_plan(plan.get("candidates"))
    if candidate.candidate_id not in {
        planned.candidate_id for planned in candidates
    }:
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 fold receipt candidate is absent from its plan"
        )
    admitted = _admitted_smokes_from_plan(
        plan.get("capacity_smokes"),
        candidates=candidates,
        source_provenance=source,
    )
    admitted_smoke = admitted.get(candidate.candidate_id)
    if (
        admitted_smoke is None
        or admitted_smoke.path != verified_paths["capacity_smoke"]
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 fold receipt smoke differs from its planned candidate"
        )
    if (
        v2_launcher._verify_pointer(
            plan.get("selection_dataset_manifest"),
            "selection dataset manifest",
        )
        != verified_paths["selection_dataset"]
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 fold receipt selection dataset differs from its plan"
        )
    materialization, feature, dynamic = _manifest_maps_from_plan(
        plan.get("songs")
    )
    manifests = _validate_manifest_set(
        materialization_manifests=materialization,
        feature_manifests=feature,
        dynamic_crop_manifests=dynamic,
        selection_dataset_manifest_path=verified_paths["selection_dataset"],
    )
    if (
        dict(manifests.selection_dataset_pointer)
        != plan.get("selection_dataset_manifest")
        or manifests.selection_dataset_content_sha256
        != plan.get("selection_dataset_content_sha256")
        or [dict(record) for record in manifests.song_records]
        != plan.get("songs")
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 fold receipt manifest graph differs from its plan"
        )
    checkpoint_root = _paths_under_external_volume(
        Path(str(plan.get("checkpoint_root") or "")), "checkpoint root"
    )
    evidence_directory = _paths_under_external_volume(
        Path(str(plan.get("evidence_directory") or "")), "evidence directory"
    )
    expected_checkpoint = checkpoint_root / _run_id(
        candidate, str(payload["validation_song_id"])
    )
    if resolved.parent != evidence_directory:
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 fold receipt is outside its planned evidence directory"
        )
    expected_checkpoint_paths = {
        "production_training_report": expected_checkpoint
        / PRODUCTION_TRAINING_REPORT_FILENAME,
        "selection_freeze": expected_checkpoint
        / STAGE_EDIT_SELECTION_FREEZE_FILENAME,
        "checkpoint_metadata": expected_checkpoint
        / DYNAMIC_CROP_V3_CHECKPOINT_METADATA_FILENAME,
        "checkpoint_weights": expected_checkpoint
        / DYNAMIC_CROP_V3_CHECKPOINT_WEIGHT_FILENAME,
    }
    if any(
        verified_paths[name] != expected
        for name, expected in expected_checkpoint_paths.items()
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 fold receipt checkpoint paths differ from its selected plan run"
        )
    args = _build_train_cli_args(
        manifests,
        candidate=candidate,
        held_out_song_id=str(payload["validation_song_id"]),
        checkpoint_directory=expected_checkpoint,
    )
    expected_run = {
        "run_id": _run_id(candidate, str(payload["validation_song_id"])),
        "candidate_name": candidate.candidate_name,
        "candidate_id": candidate.candidate_id,
        "capacity_smoke_content_sha256": admitted_smoke.payload[
            "artifact_content_sha256"
        ],
        "held_out_song_id": payload["validation_song_id"],
        "training_song_ids": list(payload["training_song_ids"]),
        "checkpoint_directory": str(expected_checkpoint),
        "evidence_directory": str(evidence_directory),
        "train_cli_args": list(args),
        "command_argv": _command_argv(
            args,
            repository_root=Path(str(source["repository_root"])),
        ),
        "dynamic_crop_version": 3,
        "activation_checkpointing": dict(
            V3_CAPACITY_ACTIVATION_CHECKPOINTING_EVIDENCE
        ),
        "fold_selection_authorized": False,
        "capacity_selection_authorized": False,
        "production_activation_authorized": False,
        "production_registry_activation_requested": False,
    }
    selected_runs = tuple(
        _mapping(raw, "v3 capacity run")
        for raw in _array(plan.get("runs"), "v3 capacity runs")
        if isinstance(raw, Mapping)
        and raw.get("candidate_id") == candidate.candidate_id
        and raw.get("held_out_song_id") == payload["validation_song_id"]
    )
    if len(selected_runs) != 1 or selected_runs[0] != expected_run:
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 fold receipt does not bind one exact selected plan run"
        )
    preflight = DynamicCropV3CapacityFoldPreflight(
        plan_path=verified_paths["plan"],
        plan_content_sha256=str(plan["plan_content_sha256"]),
        candidate_name=candidate.candidate_name,
        candidate_id=candidate.candidate_id,
        capacity_smoke_path=admitted_smoke.path,
        held_out_song_id=str(payload["validation_song_id"]),
        training_song_ids=tuple(payload["training_song_ids"]),
        checkpoint_directory=expected_checkpoint,
        evidence_directory=evidence_directory,
        selection_dataset_manifest_path=verified_paths["selection_dataset"],
        reference_ids_by_song=dict(manifests.reference_ids_by_song),
        train_cli_args=args,
        command_argv=tuple(expected_run["command_argv"]),
        free_bytes=0,
    )
    _report, _freeze, metadata = _validate_completed_v3_fold(preflight)
    if (
        payload.get("checkpoint_id") != metadata.checkpoint_id
        or payload.get("checkpoint_weight_sha256") != metadata.weight_sha256
    ):
        raise DynamicCropV3CapacityFoldLauncherError(
            "v3 fold receipt checkpoint identity differs from metadata"
        )
    return payload


def launch_dynamic_crop_v3_capacity_fold(
    plan_path: Path,
    *,
    candidate_selector: str,
    held_out_song_id: str,
) -> tuple[int, DynamicCropV3CapacityFoldPreflight, Path | None]:
    preflight = preflight_dynamic_crop_v3_capacity_fold(
        plan_path,
        candidate_selector=candidate_selector,
        held_out_song_id=held_out_song_id,
    )
    plan = _strict_plan(preflight.plan_path)
    _require_same_source_provenance(plan["source_provenance"])
    exit_code = production_train_main(preflight.train_cli_args)
    if exit_code != 0:
        return exit_code, preflight, None
    plan = _strict_plan(preflight.plan_path)
    _require_same_source_provenance(plan["source_provenance"])
    evidence = record_dynamic_crop_v3_capacity_fold_evidence(preflight)
    return 0, preflight, evidence


def _keyed_paths(values: Sequence[str], option: str) -> dict[str, Path]:
    try:
        return v2_launcher._keyed_paths(values, option)
    except (OSError, ValueError) as exc:
        raise DynamicCropV3CapacityFoldLauncherError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or launch strict non-activating Dynamic Crop v3 capacity folds"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--materialization-manifest", action="append", required=True)
    create.add_argument("--feature-manifest", action="append", required=True)
    create.add_argument("--dynamic-crop-manifest", action="append", required=True)
    create.add_argument("--selection-dataset", required=True, type=Path)
    create.add_argument("--capacity-smoke", action="append", required=True, type=Path)
    create.add_argument("--checkpoint-root", required=True, type=Path)
    create.add_argument("--evidence-directory", required=True, type=Path)
    create.add_argument("--output-directory", required=True, type=Path)
    create.add_argument("--candidate", action="append")
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
        path = create_dynamic_crop_v3_capacity_fold_plan(
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
            capacity_smoke_paths=args.capacity_smoke,
            checkpoint_root=args.checkpoint_root,
            evidence_directory=args.evidence_directory,
            output_directory=args.output_directory,
            candidate_selectors=args.candidate,
        )
        print(json.dumps({"plan_path": str(path)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "preflight":
        result = preflight_dynamic_crop_v3_capacity_fold(
            args.plan,
            candidate_selector=args.candidate,
            held_out_song_id=args.held_out_song,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0
    exit_code, result, evidence = launch_dynamic_crop_v3_capacity_fold(
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DynamicCropV3CapacityFoldLauncherError",
    "DynamicCropV3CapacityFoldPreflight",
    "DynamicCropV3ProvenanceVerificationEvidence",
    "V3_CAPACITY_FOLD_CONTRACT",
    "V3_CAPACITY_FOLD_EVIDENCE_ANALYSIS_BASIS",
    "V3_CAPACITY_FOLD_EVIDENCE_PREFIX",
    "V3_CAPACITY_FOLD_PLAN_ANALYSIS_BASIS",
    "V3_CAPACITY_FOLD_PLAN_PREFIX",
    "V3_CAPACITY_PREFLIGHT_INPUT_VERIFICATION_CONTRACT",
    "V3_SOURCE_RELATIVE_PATHS",
    "create_dynamic_crop_v3_capacity_fold_plan",
    "launch_dynamic_crop_v3_capacity_fold",
    "load_dynamic_crop_v3_capacity_fold_evidence",
    "preflight_dynamic_crop_v3_capacity_fold",
    "record_dynamic_crop_v3_capacity_fold_evidence",
]
