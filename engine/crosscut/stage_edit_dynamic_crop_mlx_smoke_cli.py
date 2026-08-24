"""Real, provenance-bound MLX memory smoke runner for Dynamic Crop v2.

The public process is a supervisor.  It launches every selected capacity
candidate in a fresh Python process, captures a content-addressed runner log,
and publishes a schema-v2 smoke evidence artifact even when the worker is
killed or runs out of memory.  The worker consumes only the fifteen immutable
BANG BANG / BLACKHOLE / REBEL HEART feature snapshots and packs them into the
fixed release tensor shape ``[2, 2048, 32, 720]``.

This is a capacity/runtime probe, not an evaluator.  It tiles the immutable
real BBR feature and supervision cycles; the resulting loss never authorizes
semantic metrics or model selection losses.

The attestation boundary rejects API-authored evidence, mixed/omitted genuine
suite records, ambient import shadowing, dirty release code, and installed
runtime byte drift. It does not claim to resist a same-user attacker who can
monkeypatch the running supervisor or construct perfectly self-consistent JSON;
that would require an external signing trust anchor not present in this tool.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import secrets
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
from functools import lru_cache

import numpy as np

from . import stage_edit_dynamic_crop_capacity_probe as capacity_probe_module
from . import stage_edit_dynamic_crop_head_v2 as dynamic_crop_head_module
from . import stage_edit_dynamic_crop_training as dynamic_crop_training_module
from . import stage_edit_feature_materializer as feature_materializer_module
from . import stage_edit_feature_snapshot_job as feature_snapshot_job_module
from . import stage_edit_production_training as production_training_module
from . import stage_edit_reference_adapter as reference_adapter_module
from . import reference_dynamic_crop_job as dynamic_crop_job_module
from . import reference_dynamic_crop_supervision as dynamic_crop_supervision_module
from . import stage_edit_transformer as transformer_module
from .stage_edit_dynamic_crop_capacity_probe import (
    EXPECTED_REFERENCE_COUNT_PER_SONG,
    PROBE_TRAINING_SCHEDULE,
    QUARANTINED_SONG_IDS,
    SELECTION_SONG_IDS,
    SMOKE_CACHE_LIMIT_BYTES,
    SMOKE_INPUT_PACKING_CONTRACT,
    SMOKE_MEMORY_LIMIT_CONTRACT,
    SMOKE_PROCESS_ISOLATION_CONTRACT,
    SMOKE_RUNNER_CONTRACT,
    SMOKE_RUNNER_LOG_ANALYSIS_BASIS,
    SMOKE_RUNNER_LOG_PREFIX,
    SMOKE_RUNNER_LOG_SCHEMA_VERSION,
    SMOKE_SUITE_ANALYSIS_BASIS,
    SMOKE_SUITE_PREFIX,
    SMOKE_SUITE_SCHEMA_VERSION,
    SMOKE_TILING_CONTRACT,
    SMOKE_WORKER_EVENT_PREFIX,
    DeviceMemoryBudget,
    DynamicCropCapacityCandidate,
    DynamicCropCapacityProbeError,
    _atomic_exclusive_write,
    _publish_supervised_mlx_smoke_evidence,
    default_dynamic_crop_v2_probe_candidates,
)
from .stage_edit_dynamic_crop_head_v2 import (
    DYNAMIC_CROP_V2_MAX_KEYFRAMES,
    StageEditDynamicCropSupervisionBatchV2,
    StageEditTransformerDynamicCropV2,
    stage_edit_dynamic_crop_v2_loss,
    stage_edit_dynamic_crop_v2_parameter_count,
)
from .stage_edit_dynamic_crop_training import (
    build_stage_edit_dynamic_crop_training_corpus_v2,
)
from .stage_edit_feature_materializer import (
    FEATURE_MATERIALIZER_CONTRACT,
    FEATURE_MATERIALIZER_SCHEMA_VERSION,
    StageEditFeatureSnapshot,
)
from .stage_edit_feature_snapshot_job import (
    FEATURE_SNAPSHOT_JOB_ANALYSIS_BASIS,
    FEATURE_SNAPSHOT_JOB_INPUT_POLICY,
    FEATURE_SNAPSHOT_JOB_PROVENANCE_CONTRACTS,
    FEATURE_SNAPSHOT_JOB_SCHEMA_VERSION,
)
from .stage_edit_production_training import (
    prepare_stage_edit_production_input_sequences,
)
from .stage_edit_reference_adapter import (
    ReferenceTrainingExample,
    load_reference_training_corpus,
)
from .reference_dynamic_crop_job import validate_reference_dynamic_crop_manifest
from .stage_edit_transformer import (
    DEFAULT_STAGE_EDIT_SCHEMA,
    StageEditSupervisionBatch,
    StageEditTensorBatch,
)


try:  # Imported lazily enough for schema/unit tests on non-Apple hosts.
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
except ImportError as exc:  # pragma: no cover - release host is Apple silicon.
    mx = None
    nn = None
    optim = None
    tree_flatten = None
    _MLX_IMPORT_ERROR: ImportError | None = exc
else:
    _MLX_IMPORT_ERROR = None


RUNNER_LOG_SCHEMA_VERSION = SMOKE_RUNNER_LOG_SCHEMA_VERSION
RUNNER_LOG_ANALYSIS_BASIS = SMOKE_RUNNER_LOG_ANALYSIS_BASIS
RUNNER_LOG_PREFIX = SMOKE_RUNNER_LOG_PREFIX
WORKER_EVENT_PREFIX = SMOKE_WORKER_EVENT_PREFIX
EXTERNAL_VOLUME_ROOT = Path("/Volumes/CrossCutMedia")
FIXED_BATCH_SIZE = 2
FIXED_SEQUENCE_LENGTH = 2048
FIXED_MAX_SOURCES = 32
FIXED_SOURCE_DIMENSION = 720
SMOKE_TARGET_CONTRACT = (
    "actual_bbr_supervision_repeated_for_capacity_only_semantic_metrics_forbidden_v1"
)
WORKER_RUN_NONCE_ENV = "STAGEEDIT_MLX_SMOKE_RUN_NONCE"
WORKER_ENVIRONMENT_CONTRACT = (
    "safe_path_isolated_verified_import_roots_no_site_empty_pycache_nonce_v3"
)
RUNTIME_DISTRIBUTION_VERIFICATION_CONTRACT = (
    "verify_record_sha256_for_non_bytecode_files_bind_all_installed_bytes_v1"
)
_WORKER_EVENT_CONTEXT: dict[str, Any] = {}


class StageEditMlxSmokeRunnerError(ValueError):
    pass


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dynamic_crop_artifact_content_sha256(payload: Mapping[str, Any]) -> str:
    """Recompute the artifact's own canonical content identity."""

    normalized = dict(payload)
    provenance = normalized.get("provenance")
    if not isinstance(provenance, Mapping):
        raise StageEditMlxSmokeRunnerError(
            "dynamic crop artifact provenance must be an object"
        )
    normalized_provenance = dict(provenance)
    normalized_provenance["artifact_content_sha256"] = None
    normalized["provenance"] = normalized_provenance
    return _sha256(normalized)


def _reject_quarantined_text(value: object, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise StageEditMlxSmokeRunnerError(f"{label} cannot be blank")
    compact = "".join(character for character in text.upper() if character.isalnum())
    for song in QUARANTINED_SONG_IDS:
        song_compact = "".join(
            character for character in song.upper() if character.isalnum()
        )
        title_compact = song_compact.removeprefix("IVE")
        if song_compact in compact or title_compact in compact:
            raise StageEditMlxSmokeRunnerError(
                f"{label} references a quarantined song"
            )
    return text


def _regular_absolute_file(raw_path: Path, label: str) -> Path:
    # Reject quarantined text before stat/read so held-out evidence stays closed.
    _reject_quarantined_text(raw_path, label)
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise StageEditMlxSmokeRunnerError(f"{label} must be absolute")
    if path.is_symlink() or not path.is_file():
        raise StageEditMlxSmokeRunnerError(
            f"{label} must be a regular non-symlink file"
        )
    return path.resolve()


def _pointer(path: Path, label: str) -> dict[str, str]:
    resolved = _regular_absolute_file(path, label)
    return {"path": str(resolved), "file_sha256": _file_sha256(resolved)}


def _parse_manifest_map(
    values: Sequence[str], *, argument_name: str = "feature manifest"
) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for raw in values:
        song, separator, raw_path = raw.partition("=")
        song = song.strip()
        raw_path = raw_path.strip()
        _reject_quarantined_text(raw, f"{argument_name} argument")
        if not separator or song not in SELECTION_SONG_IDS or not raw_path:
            raise StageEditMlxSmokeRunnerError(
                f"--{argument_name.replace(' ', '-')} must use exact BBR "
                "SONG_ID=/absolute/path.json"
            )
        if song in output:
            raise StageEditMlxSmokeRunnerError(
                f"duplicate {argument_name} for {song}"
            )
        output[song] = Path(raw_path)
    if set(output) != set(SELECTION_SONG_IDS):
        raise StageEditMlxSmokeRunnerError(
            "exactly one BANG BANG, BLACKHOLE, and REBEL HEART "
            f"{argument_name} is required"
        )
    return output


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    resolved = _regular_absolute_file(path, label)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageEditMlxSmokeRunnerError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise StageEditMlxSmokeRunnerError(f"{label} root must be an object")
    return value


def _load_bbr_feature_dataset(
    manifest_paths_by_song: Mapping[str, Path],
) -> tuple[
    dict[str, Any],
    dict[str, StageEditFeatureSnapshot],
    dict[str, tuple[str, ...]],
]:
    """Load and bind exactly five real snapshots from each BBR song."""

    if set(manifest_paths_by_song) != set(SELECTION_SONG_IDS):
        raise StageEditMlxSmokeRunnerError("feature manifest song set is not BBR")
    manifests: dict[str, dict[str, str]] = {}
    references_by_song: dict[str, tuple[str, ...]] = {}
    snapshots_by_reference: dict[str, StageEditFeatureSnapshot] = {}
    snapshot_records: list[dict[str, Any]] = []
    globally_seen: set[str] = set()
    for song in SELECTION_SONG_IDS:
        manifest_path = _regular_absolute_file(
            manifest_paths_by_song[song], f"{song} feature manifest"
        )
        manifest = _load_json_object(manifest_path, f"{song} feature manifest")
        declared_digest = manifest.get("manifest_sha256")
        unsigned = dict(manifest)
        unsigned.pop("manifest_sha256", None)
        records = manifest.get("snapshots")
        statistics = manifest.get("statistics")
        input_policy = manifest.get("input_policy")
        feature_provenance = manifest.get("feature_provenance")
        if (
            manifest.get("schema_version") != FEATURE_SNAPSHOT_JOB_SCHEMA_VERSION
            or manifest.get("analysis_basis") != FEATURE_SNAPSHOT_JOB_ANALYSIS_BASIS
            or manifest.get("song_id") != song
            or manifest.get("source_count") != 10
            or not isinstance(declared_digest, str)
            or _sha256(unsigned) != declared_digest
            or not isinstance(records, list)
            or len(records) != EXPECTED_REFERENCE_COUNT_PER_SONG
            or not isinstance(statistics, Mapping)
            or statistics.get("reference_count")
            != EXPECTED_REFERENCE_COUNT_PER_SONG
            or not isinstance(input_policy, Mapping)
            or dict(input_policy) != dict(FEATURE_SNAPSHOT_JOB_INPUT_POLICY)
            or not isinstance(feature_provenance, Mapping)
            or feature_provenance.get("producer")
            != FEATURE_MATERIALIZER_CONTRACT
            or set(feature_provenance.get("contracts") or ())
            != set(FEATURE_SNAPSHOT_JOB_PROVENANCE_CONTRACTS)
        ):
            raise StageEditMlxSmokeRunnerError(
                f"{song} feature manifest contract differs"
            )
        references: list[str] = []
        for raw_record in records:
            if not isinstance(raw_record, Mapping):
                raise StageEditMlxSmokeRunnerError(
                    f"{song} feature snapshot record must be an object"
                )
            reference = _reject_quarantined_text(
                raw_record.get("reference_id"), "feature snapshot reference id"
            )
            if reference in globally_seen:
                raise StageEditMlxSmokeRunnerError(
                    "feature reference ids must be globally unique"
                )
            snapshot_path = _regular_absolute_file(
                Path(str(raw_record.get("path") or "")),
                f"{song}/{reference} feature snapshot",
            )
            snapshot_payload = _load_json_object(
                snapshot_path, f"{song}/{reference} feature snapshot"
            )
            try:
                snapshot = StageEditFeatureSnapshot.from_dict(
                    snapshot_payload, schema=DEFAULT_STAGE_EDIT_SCHEMA
                )
            except (TypeError, ValueError) as exc:
                raise StageEditMlxSmokeRunnerError(
                    f"{song}/{reference} feature snapshot is invalid: {exc}"
                ) from exc
            token_count = len(snapshot.sequence.tokens)
            candidate_counts = {
                len(token.source_candidates) for token in snapshot.sequence.tokens
            }
            if (
                raw_record.get("snapshot_sha256") != snapshot.snapshot_sha256
                or snapshot.schema_version != FEATURE_MATERIALIZER_SCHEMA_VERSION
                or raw_record.get("token_count") != token_count
                or raw_record.get("source_count") != 10
                or not candidate_counts
                or max(candidate_counts) > FIXED_MAX_SOURCES
            ):
                raise StageEditMlxSmokeRunnerError(
                    f"{song}/{reference} snapshot record differs from file"
                )
            globally_seen.add(reference)
            references.append(reference)
            snapshots_by_reference[reference] = snapshot
            snapshot_records.append(
                {
                    "song_id": song,
                    "reference_id": reference,
                    "token_count": token_count,
                    "snapshot_content_sha256": snapshot.snapshot_sha256,
                    "file": _pointer(snapshot_path, "feature snapshot"),
                }
            )
        references_by_song[song] = tuple(sorted(references))
        manifests[song] = _pointer(manifest_path, f"{song} feature manifest")
    if len(globally_seen) != 15:
        raise StageEditMlxSmokeRunnerError("BBR smoke dataset is not exactly 15 refs")
    snapshot_records.sort(key=lambda item: (SELECTION_SONG_IDS.index(item["song_id"]), item["reference_id"]))
    provenance = {
        "selection_song_ids": list(SELECTION_SONG_IDS),
        "quarantined_song_ids_opened": False,
        "reference_count": 15,
        "reference_ids_by_song": {
            song: list(references_by_song[song]) for song in SELECTION_SONG_IDS
        },
        "feature_manifests": manifests,
        "feature_snapshots": snapshot_records,
    }
    return provenance, snapshots_by_reference, references_by_song


def _load_bbr_target_bindings(
    *,
    materialization_manifest_paths_by_song: Mapping[str, Path],
    dynamic_crop_manifest_paths_by_song: Mapping[str, Path],
    snapshots_by_reference: Mapping[str, StageEditFeatureSnapshot],
    references_by_song: Mapping[str, Sequence[str]],
    build_training_corpora: bool,
) -> tuple[
    dict[str, Any],
    dict[str, ReferenceTrainingExample],
    dict[str, Any],
]:
    """Bind 3+3 target manifests, 15 supervisions, and all 599 crop files."""

    if set(materialization_manifest_paths_by_song) != set(
        SELECTION_SONG_IDS
    ) or set(dynamic_crop_manifest_paths_by_song) != set(SELECTION_SONG_IDS):
        raise StageEditMlxSmokeRunnerError("target manifest song set is not BBR")
    materialization_manifests: dict[str, dict[str, str]] = {}
    dynamic_crop_manifests: dict[str, dict[str, str]] = {}
    supervision_records: list[dict[str, Any]] = []
    supervision_paths: list[Path] = []
    artifact_paths_by_reference: dict[str, list[Path]] = {}
    artifact_records: list[dict[str, Any]] = []
    song_by_reference: dict[str, str] = {}
    for song in SELECTION_SONG_IDS:
        expected_references = set(references_by_song[song])
        materialization_path = _regular_absolute_file(
            materialization_manifest_paths_by_song[song],
            f"{song} materialization manifest",
        )
        materialization = _load_json_object(
            materialization_path, f"{song} materialization manifest"
        )
        input_contract = materialization.get("input_contract")
        outputs = materialization.get("outputs")
        statistics = materialization.get("statistics")
        if (
            materialization.get("schema_version") != 1
            or materialization.get("analysis_basis")
            != "immutable_reference_training_materialization_v1"
            or materialization.get("immutable_inputs_modified") is not False
            or not isinstance(input_contract, Mapping)
            or input_contract.get("expected_reference_count") != 5
            or input_contract.get("expected_source_count") != 10
            or not isinstance(outputs, list)
            or len(outputs) != 5
            or not isinstance(statistics, Mapping)
            or statistics.get("reference_count") != 5
        ):
            raise StageEditMlxSmokeRunnerError(
                f"{song} materialization manifest contract differs"
            )
        output_references: set[str] = set()
        for raw in outputs:
            if not isinstance(raw, Mapping):
                raise StageEditMlxSmokeRunnerError(
                    "materialization output must be an object"
                )
            reference = _reject_quarantined_text(
                raw.get("reference_id"), "materialization reference id"
            )
            path = _regular_absolute_file(
                Path(str(raw.get("supervision_path") or "")),
                "reference supervision",
            )
            expected_hash = raw.get("supervision_file_sha256")
            actual_hash = _file_sha256(path)
            if expected_hash != actual_hash:
                raise StageEditMlxSmokeRunnerError(
                    f"{song}/{reference} supervision hash mismatch"
                )
            output_references.add(reference)
            song_by_reference[reference] = song
            supervision_paths.append(path)
            record = {
                "song_id": song,
                "reference_id": reference,
                "file": {"path": str(path), "file_sha256": actual_hash},
            }
            supervision_records.append(record)
        if output_references != expected_references:
            raise StageEditMlxSmokeRunnerError(
                f"{song} materialization references differ from features"
            )
        materialization_manifests[song] = _pointer(
            materialization_path, f"{song} materialization manifest"
        )

        dynamic_manifest_path = _regular_absolute_file(
            dynamic_crop_manifest_paths_by_song[song],
            f"{song} dynamic crop manifest",
        )
        dynamic_manifest = _load_json_object(
            dynamic_manifest_path, f"{song} dynamic crop manifest"
        )
        try:
            validate_reference_dynamic_crop_manifest(
                dynamic_manifest, verify_artifact_files=False
            )
        except (TypeError, ValueError) as exc:
            raise StageEditMlxSmokeRunnerError(
                f"{song} dynamic crop manifest is invalid: {exc}"
            ) from exc
        selection = dynamic_manifest.get("selection")
        records = dynamic_manifest.get("artifacts")
        if (
            not isinstance(selection, Mapping)
            or set(selection.get("reference_ids") or ()) != expected_references
            or not isinstance(records, list)
        ):
            raise StageEditMlxSmokeRunnerError(
                f"{song} dynamic crop references differ"
            )
        for raw in records:
            if not isinstance(raw, Mapping):
                raise StageEditMlxSmokeRunnerError(
                    "dynamic crop artifact record must be an object"
                )
            reference = _reject_quarantined_text(
                raw.get("reference_id"), "dynamic crop reference id"
            )
            artifact = raw.get("artifact")
            if reference not in expected_references or not isinstance(
                artifact, Mapping
            ):
                raise StageEditMlxSmokeRunnerError(
                    "dynamic crop artifact identity is invalid"
                )
            path = _regular_absolute_file(
                Path(str(artifact.get("path") or "")),
                "dynamic crop artifact",
            )
            actual_hash = _file_sha256(path)
            if actual_hash != artifact.get("file_sha256"):
                raise StageEditMlxSmokeRunnerError(
                    f"{song}/{reference} dynamic crop file hash mismatch"
                )
            artifact_payload = _load_json_object(path, "dynamic crop artifact")
            actual_content_hash = _dynamic_crop_artifact_content_sha256(
                artifact_payload
            )
            recorded_content_hash = artifact.get("content_sha256")
            if (
                actual_content_hash != recorded_content_hash
                or artifact_payload.get("provenance", {}).get(
                    "artifact_content_sha256"
                )
                != actual_content_hash
            ):
                raise StageEditMlxSmokeRunnerError(
                    f"{song}/{reference} dynamic crop content hash mismatch"
                )
            record = {
                "song_id": song,
                "reference_id": reference,
                "shot_index": int(raw.get("shot_index")),
                "file": {"path": str(path), "file_sha256": actual_hash},
                "content_sha256": actual_content_hash,
            }
            if len(record["content_sha256"]) != 64:
                raise StageEditMlxSmokeRunnerError(
                    "dynamic crop content hash is invalid"
                )
            artifact_records.append(record)
            artifact_paths_by_reference.setdefault(reference, []).append(path)
        dynamic_crop_manifests[song] = _pointer(
            dynamic_manifest_path, f"{song} dynamic crop manifest"
        )
    if (
        len(supervision_records) != 15
        or len({item["reference_id"] for item in supervision_records}) != 15
        or len(artifact_records) != 599
        or set(artifact_paths_by_reference) != set(snapshots_by_reference)
    ):
        raise StageEditMlxSmokeRunnerError(
            "target corpus must contain exactly 15 supervisions and 599 crop artifacts"
        )
    sorted_supervisions = sorted(
        supervision_records,
        key=lambda item: (
            SELECTION_SONG_IDS.index(item["song_id"]),
            item["reference_id"],
        ),
    )
    sorted_artifacts = sorted(
        artifact_records,
        key=lambda item: (
            SELECTION_SONG_IDS.index(item["song_id"]),
            item["reference_id"],
            item["shot_index"],
        ),
    )
    if (
        len(
            {
                (item["song_id"], item["reference_id"], item["shot_index"])
                for item in sorted_artifacts
            }
        )
        != 599
        or len({item["file"]["path"] for item in sorted_artifacts}) != 599
    ):
        raise StageEditMlxSmokeRunnerError(
            "dynamic crop artifact identities and paths must be unique"
        )
    target_provenance: dict[str, Any] = {
        "materialization_manifests": materialization_manifests,
        "dynamic_crop_manifests": dynamic_crop_manifests,
        "reference_supervisions": sorted_supervisions,
        "reference_supervision_count": 15,
        "reference_supervision_aggregate_sha256": _sha256(sorted_supervisions),
        "dynamic_crop_artifacts": sorted_artifacts,
        "dynamic_crop_artifact_count": 599,
        "dynamic_crop_artifact_aggregate_sha256": _sha256(sorted_artifacts),
    }
    if not build_training_corpora:
        return target_provenance, {}, {}
    corpus = load_reference_training_corpus(
        tuple(supervision_paths),
        feature_snapshot_by_reference=snapshots_by_reference,
        song_id_by_reference=song_by_reference,
        negative_samples_per_hold=2,
        boundary_guard_seconds=0.18,
        max_sources=32,
    )
    prepared_sequences = {
        example.reference_id: sequence
        for example, sequence in zip(
            corpus.examples,
            prepare_stage_edit_production_input_sequences(corpus.examples),
            strict=True,
        )
    }
    examples_by_reference = {
        example.reference_id: replace(
            example, sequence=prepared_sequences[example.reference_id]
        )
        for example in corpus.examples
    }
    dynamic_corpus = build_stage_edit_dynamic_crop_training_corpus_v2(
        corpus.examples,
        artifact_paths_by_reference={
            reference: tuple(paths)
            for reference, paths in artifact_paths_by_reference.items()
        },
        verify_input_files=False,
    )
    dynamic_by_reference = {
        example.reference_id: dynamic_corpus.sequence_for(
            example.sequence.sequence_id
        )
        for example in corpus.examples
    }
    return target_provenance, examples_by_reference, dynamic_by_reference


def _lane_assignments(
    references_by_song: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    ordered = tuple(
        reference
        for song in SELECTION_SONG_IDS
        for reference in sorted(references_by_song[song])
    )
    if len(ordered) != 15 or len(set(ordered)) != 15:
        raise StageEditMlxSmokeRunnerError("fixed lane assignment requires 15 refs")
    return {
        "A": ordered[0::2],
        "B": ordered[1::2],
    }


def _tensor_digest(named_arrays: Sequence[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, value in named_arrays:
        contiguous = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(_canonical_bytes(list(contiguous.shape)))
        digest.update(b"\0")
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _encoded_feature(
    values: Sequence[float],
    missing: Sequence[bool],
    *,
    expected_raw_dimension: int,
    label: str,
) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float32)
    mask = np.asarray(missing, dtype=bool)
    if raw.shape != (expected_raw_dimension,) or mask.shape != (
        expected_raw_dimension,
    ):
        raise StageEditMlxSmokeRunnerError(f"{label} dimension differs")
    if not np.all(np.isfinite(raw)):
        raise StageEditMlxSmokeRunnerError(f"{label} contains non-finite values")
    return np.concatenate((np.where(mask, 0.0, raw), mask.astype(np.float32)))


def _pack_fixed_lane_training_arrays(
    *,
    examples_by_reference: Mapping[str, ReferenceTrainingExample],
    dynamic_by_reference: Mapping[str, Any],
    lane_assignments: Mapping[str, Sequence[str]],
    candidate: DynamicCropCapacityCandidate,
    sequence_length: int = FIXED_SEQUENCE_LENGTH,
    max_sources: int = FIXED_MAX_SOURCES,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Tile exact BBR features/targets to the full release memory shape."""

    schema = DEFAULT_STAGE_EDIT_SCHEMA
    cycles: dict[str, list[tuple[Any, Any, Any]]] = {}
    original_counts: dict[str, int] = {}
    observed_source_max = 0
    max_rejections = 1
    for lane in ("A", "B"):
        records: list[tuple[Any, Any, Any]] = []
        for reference in lane_assignments[lane]:
            example = examples_by_reference[reference]
            dynamic_sequence = dynamic_by_reference[reference]
            if not (
                len(example.sequence.tokens)
                == len(example.targets.targets)
                == len(dynamic_sequence.targets)
            ):
                raise StageEditMlxSmokeRunnerError(
                    f"{reference} feature/base/dynamic target lengths differ"
                )
            for token, target, dynamic_target in zip(
                example.sequence.tokens,
                example.targets.targets,
                dynamic_sequence.targets,
                strict=True,
            ):
                records.append((token, target, dynamic_target))
                observed_source_max = max(
                    observed_source_max, len(token.source_candidates)
                )
                max_rejections = max(
                    max_rejections, len(target.rejected_source_ids)
                )
        if not records or len(records) > sequence_length:
            raise StageEditMlxSmokeRunnerError(
                f"fixed lane {lane} original cycle is invalid"
            )
        cycles[lane] = records
        original_counts[lane] = len(records)

    context = np.zeros(
        (2, sequence_length, schema.encoded_context_dimension),
        dtype=np.float32,
    )
    sources = np.zeros(
        (2, sequence_length, max_sources, FIXED_SOURCE_DIMENSION),
        dtype=np.float32,
    )
    sequence_mask = np.ones((2, sequence_length), dtype=bool)
    candidate_mask = np.zeros((2, sequence_length, max_sources), dtype=bool)
    hard_gate_mask = np.zeros_like(candidate_mask)
    cut_targets = np.zeros((2, sequence_length), dtype=np.float32)
    cut_mask = np.zeros_like(cut_targets, dtype=bool)
    source_targets = np.zeros((2, sequence_length), dtype=np.int32)
    source_mask = np.zeros_like(source_targets, dtype=bool)
    rejected_targets = np.zeros(
        (2, sequence_length, max_rejections), dtype=np.int32
    )
    pairwise_mask = np.zeros_like(rejected_targets, dtype=bool)
    duration_targets = np.zeros((2, sequence_length), dtype=np.int32)
    duration_mask = np.zeros_like(duration_targets, dtype=bool)
    transition_targets = np.zeros((2, sequence_length), dtype=np.int32)
    transition_mask = np.zeros_like(transition_targets, dtype=bool)
    crop_targets = np.zeros((2, sequence_length, 4), dtype=np.float32)
    crop_mask = np.zeros_like(crop_targets, dtype=bool)
    keyframes = DYNAMIC_CROP_V2_MAX_KEYFRAMES
    dynamic_values = np.zeros((2, sequence_length, keyframes, 4), dtype=np.float32)
    normalized_times = np.zeros((2, sequence_length, keyframes), dtype=np.float32)
    keyframe_mask = np.zeros_like(normalized_times, dtype=bool)
    presence_targets = np.zeros_like(normalized_times, dtype=np.float32)
    presence_mask = np.zeros_like(normalized_times, dtype=bool)
    motion_targets = np.zeros((2, sequence_length), dtype=np.float32)
    motion_mask = np.zeros_like(motion_targets, dtype=bool)
    durations = np.zeros((2, sequence_length), dtype=np.float32)
    track_mask = np.zeros_like(durations, dtype=bool)
    keyframe_counts = np.zeros((2, sequence_length), dtype=np.int32)
    transition_to_index = {
        label: index for index, label in enumerate(schema.transition_labels)
    }

    for lane_index, lane in enumerate(("A", "B")):
        cycle = cycles[lane]
        for time_index in range(sequence_length):
            token, target, dynamic_target = cycle[time_index % len(cycle)]
            context_values = (
                tuple(token.music_features)
                + tuple(token.section_features)
                + tuple(token.recent_edit_features)
                + tuple(token.style_features)
            )
            context[lane_index, time_index] = _encoded_feature(
                context_values,
                token.missing_mask,
                expected_raw_dimension=schema.raw_context_dimension,
                label="smoke context feature",
            )
            source_to_slot: dict[str, int] = {}
            for source_index, source in enumerate(token.source_candidates):
                sources[lane_index, time_index, source_index] = _encoded_feature(
                    source.features,
                    source.missing_mask,
                    expected_raw_dimension=schema.raw_source_dimension,
                    label="smoke source feature",
                )
                source_to_slot[source.source_id] = source_index
                candidate_mask[lane_index, time_index, source_index] = bool(
                    source.available
                )
                hard_gate_mask[lane_index, time_index, source_index] = bool(
                    source.available and source.hard_gate_passed
                )
            if target.cut is not None:
                cut_targets[lane_index, time_index] = float(target.cut)
                cut_mask[lane_index, time_index] = True
            if target.selected_source_id is not None:
                selected = source_to_slot.get(target.selected_source_id)
                if selected is None or not hard_gate_mask[
                    lane_index, time_index, selected
                ]:
                    raise StageEditMlxSmokeRunnerError(
                        "actual selected source is absent or hard-gated"
                    )
                source_targets[lane_index, time_index] = selected
                source_mask[lane_index, time_index] = True
                for rejection_index, rejected_id in enumerate(
                    target.rejected_source_ids
                ):
                    rejected = source_to_slot.get(rejected_id)
                    if rejected is None:
                        raise StageEditMlxSmokeRunnerError(
                            "actual rejected source is absent"
                        )
                    if hard_gate_mask[lane_index, time_index, rejected]:
                        rejected_targets[
                            lane_index, time_index, rejection_index
                        ] = rejected
                        pairwise_mask[
                            lane_index, time_index, rejection_index
                        ] = True
            if target.duration_bucket is not None:
                duration_targets[lane_index, time_index] = int(
                    target.duration_bucket
                )
                duration_mask[lane_index, time_index] = True
            if target.transition_label is not None:
                transition_targets[lane_index, time_index] = transition_to_index[
                    target.transition_label
                ]
                transition_mask[lane_index, time_index] = True
            if target.crop is not None:
                for coordinate, value in enumerate(target.crop):
                    if value is not None:
                        crop_targets[lane_index, time_index, coordinate] = float(value)
                        crop_mask[lane_index, time_index, coordinate] = True
            if dynamic_target is not None:
                count = len(dynamic_target.keyframes)
                if not 1 <= count <= keyframes:
                    raise StageEditMlxSmokeRunnerError(
                        "actual dynamic crop target exceeds K=8"
                    )
                track_mask[lane_index, time_index] = True
                presence_mask[lane_index, time_index, :] = True
                presence_targets[lane_index, time_index, :count] = 1.0
                keyframe_mask[lane_index, time_index, :count] = True
                duration = float(dynamic_target.shot_duration_seconds)
                durations[lane_index, time_index] = duration
                keyframe_counts[lane_index, time_index] = count
                if not dynamic_target.motion_label_loss_masked:
                    motion_mask[lane_index, time_index] = True
                    motion_targets[lane_index, time_index] = float(
                        dynamic_target.motion_label == "dynamic"
                    )
                for keyframe_index, keyframe in enumerate(
                    dynamic_target.keyframes
                ):
                    normalized_times[
                        lane_index, time_index, keyframe_index
                    ] = float(keyframe.time_seconds) / duration
                    dynamic_values[
                        lane_index, time_index, keyframe_index
                    ] = (
                        keyframe.zoom,
                        keyframe.center_x,
                        keyframe.center_y,
                        keyframe.roll_degrees,
                    )

    input_arrays = {
        "context_features": context,
        "source_features": sources,
        "sequence_mask": sequence_mask,
        "source_candidate_mask": candidate_mask,
        "hard_source_gate_mask": hard_gate_mask,
    }
    base_arrays = {
        "cut_targets": cut_targets,
        "cut_mask": cut_mask,
        "source_targets": source_targets,
        "source_mask": source_mask,
        "rejected_source_targets": rejected_targets,
        "pairwise_source_mask": pairwise_mask,
        "duration_targets": duration_targets,
        "duration_mask": duration_mask,
        "transition_targets": transition_targets,
        "transition_mask": transition_mask,
        "crop_targets": crop_targets,
        "crop_mask": crop_mask,
    }
    dynamic_arrays = {
        "values": dynamic_values,
        "normalized_times": normalized_times,
        "keyframe_mask": keyframe_mask,
        "presence_targets": presence_targets,
        "presence_mask": presence_mask,
        "motion_targets": motion_targets,
        "motion_mask": motion_mask,
        "shot_duration_seconds": durations,
        "track_mask": track_mask,
        "keyframe_count_targets": keyframe_counts,
    }
    supervision_digest = _tensor_digest(
        tuple((f"base.{name}", value) for name, value in base_arrays.items())
        + tuple(
            (f"dynamic.{name}", value) for name, value in dynamic_arrays.items()
        )
    )
    provenance = {
        "packing_contract": SMOKE_INPUT_PACKING_CONTRACT,
        "tiling_contract": SMOKE_TILING_CONTRACT,
        "batch_size": 2,
        "sequence_length": sequence_length,
        "max_sources": max_sources,
        "actual_observed_source_max": observed_source_max,
        "context_feature_dimension": schema.encoded_context_dimension,
        "source_feature_dimension": schema.encoded_source_dimension,
        "source_tensor_shape": list(sources.shape),
        "context_tensor_shape": list(context.shape),
        "dynamic_values_shape": list(dynamic_values.shape),
        "dynamic_times_shape": list(normalized_times.shape),
        "dynamic_presence_shape": list(presence_targets.shape),
        "dtype": "float32",
        "lane_reference_ids": {
            lane: list(lane_assignments[lane]) for lane in ("A", "B")
        },
        "lane_original_cycle_token_counts": original_counts,
        "lane_repeat_counts": {
            lane: sequence_length // original_counts[lane]
            for lane in ("A", "B")
        },
        "lane_truncated_prefix_counts": {
            lane: sequence_length % original_counts[lane]
            for lane in ("A", "B")
        },
        "lane_expanded_token_counts": {"A": sequence_length, "B": sequence_length},
        "lane_padding_counts": {"A": 0, "B": 0},
        "base_label_mask_counts": {
            "cut": int(cut_mask.sum()),
            "source_listwise": int(source_mask.sum()),
            "source_pairwise": int(pairwise_mask.sum()),
            "duration": int(duration_mask.sum()),
            "transition_metadata_only": int(transition_mask.sum()),
            "crop_coordinates": int(crop_mask.sum()),
        },
        "dynamic_label_mask_counts": {
            "tracks": int(track_mask.sum()),
            "keyframes": int(keyframe_mask.sum()),
            "presence": int(presence_mask.sum()),
            "motion": int(motion_mask.sum()),
        },
        "parameter_count": stage_edit_dynamic_crop_v2_parameter_count(
            candidate.model_config, candidate.dynamic_crop_config
        ),
        "input_tensor_sha256": _tensor_digest(tuple(input_arrays.items())),
        "supervision_tensor_sha256": supervision_digest,
        "real_input_materialized": True,
    }
    return input_arrays, base_arrays, dynamic_arrays, provenance


def _tensor_provenance_template(
    snapshots_by_reference: Mapping[str, StageEditFeatureSnapshot],
    lane_assignments: Mapping[str, Sequence[str]],
    candidate: DynamicCropCapacityCandidate,
) -> dict[str, Any]:
    counts = {
        lane: sum(
            len(snapshots_by_reference[reference].sequence.tokens)
            for reference in lane_assignments[lane]
        )
        for lane in ("A", "B")
    }
    if any(count < 1 or count > FIXED_SEQUENCE_LENGTH for count in counts.values()):
        raise StageEditMlxSmokeRunnerError("fixed lane token count is invalid")
    return {
        "packing_contract": SMOKE_INPUT_PACKING_CONTRACT,
        "tiling_contract": SMOKE_TILING_CONTRACT,
        "batch_size": FIXED_BATCH_SIZE,
        "sequence_length": FIXED_SEQUENCE_LENGTH,
        "max_sources": FIXED_MAX_SOURCES,
        "actual_observed_source_max": max(
            len(token.source_candidates)
            for snapshot in snapshots_by_reference.values()
            for token in snapshot.sequence.tokens
        ),
        "context_feature_dimension": DEFAULT_STAGE_EDIT_SCHEMA.encoded_context_dimension,
        "source_feature_dimension": DEFAULT_STAGE_EDIT_SCHEMA.encoded_source_dimension,
        "source_tensor_shape": [2, 2048, 32, 720],
        "context_tensor_shape": [
            2,
            2048,
            DEFAULT_STAGE_EDIT_SCHEMA.encoded_context_dimension,
        ],
        "dynamic_values_shape": [2, 2048, 8, 4],
        "dynamic_times_shape": [2, 2048, 8],
        "dynamic_presence_shape": [2, 2048, 8],
        "dtype": "float32",
        "lane_reference_ids": {
            lane: list(lane_assignments[lane]) for lane in ("A", "B")
        },
        "lane_original_cycle_token_counts": counts,
        "lane_repeat_counts": {
            lane: FIXED_SEQUENCE_LENGTH // counts[lane] for lane in ("A", "B")
        },
        "lane_truncated_prefix_counts": {
            lane: FIXED_SEQUENCE_LENGTH % counts[lane] for lane in ("A", "B")
        },
        "lane_expanded_token_counts": {"A": 2048, "B": 2048},
        "lane_padding_counts": {"A": 0, "B": 0},
        "base_label_mask_counts": None,
        "dynamic_label_mask_counts": None,
        "parameter_count": stage_edit_dynamic_crop_v2_parameter_count(
            candidate.model_config, candidate.dynamic_crop_config
        ),
        "input_tensor_sha256": None,
        "supervision_tensor_sha256": None,
        "real_input_materialized": False,
    }


def _live_budget() -> DeviceMemoryBudget:
    if mx is None:
        detail = f": {_MLX_IMPORT_ERROR}" if _MLX_IMPORT_ERROR else ""
        raise StageEditMlxSmokeRunnerError(f"MLX is unavailable{detail}")
    device = dict(mx.device_info())
    try:
        mlx_version = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError as exc:
        raise StageEditMlxSmokeRunnerError("MLX package metadata is unavailable") from exc
    return DeviceMemoryBudget(
        device_name=str(device.get("device_name") or ""),
        physical_memory_bytes=int(device.get("memory_size") or 0),
        max_recommended_working_set_size=int(
            device.get("max_recommended_working_set_size") or 0
        ),
        max_buffer_length=int(device.get("max_buffer_length") or 0),
        architecture=str(device.get("architecture") or ""),
        mlx_version=mlx_version,
    )


def _hardware_provenance(budget: DeviceMemoryBudget) -> dict[str, Any]:
    return {
        "device_name": budget.device_name,
        "physical_memory_bytes": budget.physical_memory_bytes,
        "max_recommended_working_set_size": budget.max_recommended_working_set_size,
        "max_buffer_length": budget.max_buffer_length,
        "architecture": budget.architecture,
        "os_reserve_bytes": budget.os_reserve_bytes,
        "allowed_training_bytes": budget.allowed_training_bytes,
        "device_info_api": "mlx.core.device_info",
    }


def _compute_distribution_record_provenance(
    distribution_name: str,
) -> dict[str, Any]:
    """Bind every installed distribution file to its live bytes and RECORD.

    RECORD itself is normally unhashed. It is still included in the live-byte
    aggregate; every entry carrying a wheel hash is independently verified
    before the aggregate can be emitted.
    """

    distribution = importlib.metadata.distribution(distribution_name)
    records = []
    verified_record_hash_count = 0
    mutable_bytecode_cache_record_count = 0
    unhashed_record_count = 0
    for package_path in distribution.files or ():
        file_hash = package_path.hash
        installed_path = Path(package_path.locate()).resolve()
        if installed_path.is_symlink() or not installed_path.is_file():
            raise StageEditMlxSmokeRunnerError(
                f"{distribution_name} installed file is unavailable: {installed_path}"
            )
        actual_file_sha256 = _file_sha256(installed_path)
        actual_size = installed_path.stat().st_size
        is_bytecode_cache = (
            installed_path.suffix == ".pyc"
            or "__pycache__" in installed_path.parts
        )
        if is_bytecode_cache:
            # Python bytecode caches are mutable and this host's RECORD has
            # duplicate stale hashed/unhashed entries for the same pyc. The
            # worker uses an empty nonce-scoped PYTHONPYCACHEPREFIX, so these
            # installed cache bytes are never executable by the child. They
            # remain bound into both aggregate hashes for full disclosure.
            mutable_bytecode_cache_record_count += 1
        elif file_hash is not None:
            if package_path.size is not None and int(package_path.size) != actual_size:
                raise StageEditMlxSmokeRunnerError(
                    f"{distribution_name} installed file size differs from RECORD: {package_path}"
                )
            if str(file_hash.mode).lower() != "sha256":
                raise StageEditMlxSmokeRunnerError(
                    f"{distribution_name} RECORD uses unsupported hash mode"
                )
            recorded_digest = str(file_hash.value).rstrip("=")
            actual_record_digest = base64.urlsafe_b64encode(
                bytes.fromhex(actual_file_sha256)
            ).decode("ascii").rstrip("=")
            if recorded_digest != actual_record_digest:
                raise StageEditMlxSmokeRunnerError(
                    f"{distribution_name} installed file differs from RECORD: {package_path}"
                )
            verified_record_hash_count += 1
        else:
            unhashed_record_count += 1
        records.append(
            {
                "path": str(package_path),
                "installed_path": str(installed_path),
                "hash": (
                    None
                    if file_hash is None
                    else {
                        "mode": str(file_hash.mode),
                        "value": str(file_hash.value),
                    }
                ),
                "recorded_size": package_path.size,
                "actual_size": actual_size,
                "actual_file_sha256": actual_file_sha256,
            }
        )
    records.sort(key=lambda item: item["path"])
    return {
        "distribution_name": distribution_name,
        "version": distribution.version,
        "record_count": len(records),
        "verified_record_hash_count": verified_record_hash_count,
        "mutable_bytecode_cache_record_count": (
            mutable_bytecode_cache_record_count
        ),
        "unhashed_record_count": unhashed_record_count,
        "verification_contract": RUNTIME_DISTRIBUTION_VERIFICATION_CONTRACT,
        "record_identity_sha256": _sha256(records),
        "actual_files_aggregate_sha256": _sha256(
            [
                {
                    "path": record["path"],
                    "actual_file_sha256": record["actual_file_sha256"],
                    "actual_size": record["actual_size"],
                }
                for record in records
            ]
        ),
    }


@lru_cache(maxsize=4)
def _distribution_record_provenance(distribution_name: str) -> dict[str, Any]:
    return _compute_distribution_record_provenance(distribution_name)


def _runtime_import_roots() -> tuple[str, ...]:
    roots: list[str] = []
    for distribution_name in ("numpy", "mlx"):
        distribution = importlib.metadata.distribution(distribution_name)
        root = str(Path(distribution.locate_file("")).resolve())
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def _worker_pythonpath_entries(repo: Path) -> list[str]:
    entries = [str((repo / "engine").resolve()), *_runtime_import_roots()]
    if len(entries) != len(set(entries)):
        raise StageEditMlxSmokeRunnerError(
            "worker Python import roots are not unique"
        )
    for entry in entries:
        root = Path(entry)
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise StageEditMlxSmokeRunnerError(
                f"worker Python import root is invalid: {entry}"
            )
    return entries


def _runtime_module_pointer(
    module: Any, *, version: str, distribution_name: str
) -> dict[str, Any]:
    path_value = getattr(module, "__file__", None)
    if not path_value:
        raise StageEditMlxSmokeRunnerError(
            f"runtime module {module.__name__} has no source path"
        )
    resolved_path = Path(path_value).resolve()
    distribution = importlib.metadata.distribution(distribution_name)
    record_paths = sorted(
        str(package_path)
        for package_path in distribution.files or ()
        if Path(package_path.locate()).resolve() == resolved_path
    )
    if len(record_paths) != 1:
        raise StageEditMlxSmokeRunnerError(
            f"runtime module {module.__name__} is not uniquely present in "
            f"the {distribution_name} RECORD"
        )
    return {
        "module_name": str(module.__name__),
        "version": version,
        "distribution_name": distribution_name,
        "distribution_record_path": record_paths[0],
        **_pointer(resolved_path, f"{module.__name__} runtime module"),
    }


def _runtime_provenance(
    budget: DeviceMemoryBudget, repo: Path
) -> dict[str, Any]:
    if mx is None or nn is None or optim is None:
        raise StageEditMlxSmokeRunnerError("MLX is unavailable")
    numpy_version = importlib.metadata.version("numpy")
    numpy_native = __import__(
        "numpy._core._multiarray_umath", fromlist=("_multiarray_umath",)
    )
    mlx_linear = __import__("mlx.nn.layers.linear", fromlist=("linear",))
    mlx_transformer = __import__(
        "mlx.nn.layers.transformer", fromlist=("transformer",)
    )
    mlx_optimizer_impl = __import__(
        "mlx.optimizers.optimizers", fromlist=("optimizers",)
    )
    return {
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "mlx_version": budget.mlx_version,
        "default_device": str(mx.default_device()),
        "process_isolation_contract": SMOKE_PROCESS_ISOLATION_CONTRACT,
        "worker_environment_contract": WORKER_ENVIRONMENT_CONTRACT,
        "worker_pythonpath_entries": _worker_pythonpath_entries(repo),
        "ambient_pythonpath_inherited": False,
        "python_no_user_site": True,
        "python_safe_path": True,
        "site_import_disabled": True,
        "installed_bytecode_cache_read_disabled": True,
        "imported_runtime_modules": {
            "numpy": _runtime_module_pointer(
                np, version=numpy_version, distribution_name="numpy"
            ),
            "mlx_core": _runtime_module_pointer(
                mx, version=budget.mlx_version, distribution_name="mlx"
            ),
            "mlx_nn": _runtime_module_pointer(
                nn, version=budget.mlx_version, distribution_name="mlx"
            ),
            "mlx_optimizers": _runtime_module_pointer(
                optim, version=budget.mlx_version, distribution_name="mlx"
            ),
            "numpy_multiarray_native": _runtime_module_pointer(
                numpy_native, version=numpy_version, distribution_name="numpy"
            ),
            "mlx_nn_linear_impl": _runtime_module_pointer(
                mlx_linear,
                version=budget.mlx_version,
                distribution_name="mlx",
            ),
            "mlx_nn_transformer_impl": _runtime_module_pointer(
                mlx_transformer,
                version=budget.mlx_version,
                distribution_name="mlx",
            ),
            "mlx_optimizer_impl": _runtime_module_pointer(
                mlx_optimizer_impl,
                version=budget.mlx_version,
                distribution_name="mlx",
            ),
        },
        "runtime_distributions": {
            "numpy": _distribution_record_provenance("numpy"),
            "mlx": _distribution_record_provenance("mlx"),
        },
    }


def _verify_isolated_worker_environment(repo: Path) -> None:
    expected_pythonpath = os.pathsep.join(_worker_pythonpath_entries(repo))
    nonce_sha256 = _WORKER_EVENT_CONTEXT.get("supervisor_run_nonce_sha256")
    expected_pycache_prefix = _worker_pycache_prefix(str(nonce_sha256))
    if (
        os.environ.get("PYTHONPATH") != expected_pythonpath
        or os.environ.get("PYTHONNOUSERSITE") != "1"
        or os.environ.get("PYTHONSAFEPATH") != "1"
        or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
        or os.environ.get("PYTHONPYCACHEPREFIX") != expected_pycache_prefix
        or os.environ.get("PYTHONHASHSEED") != "0"
        or WORKER_RUN_NONCE_ENV in os.environ
    ):
        raise StageEditMlxSmokeRunnerError(
            "worker Python environment is not isolated to the verified import roots"
        )


def _worker_pycache_prefix(run_nonce_sha256: str) -> str:
    if (
        len(run_nonce_sha256) != 64
        or any(character not in "0123456789abcdef" for character in run_nonce_sha256)
    ):
        raise StageEditMlxSmokeRunnerError("worker nonce digest is invalid")
    return str(
        (
            Path(tempfile.gettempdir()).resolve()
            / f"stageedit-mlx-smoke-pycache-{run_nonce_sha256}"
        ).resolve()
    )


def _build_isolated_worker_environment(
    *,
    repo: Path,
    runtime_provenance: Mapping[str, Any],
    run_nonce: str,
) -> dict[str, str]:
    """Construct the exact child environment used by the real supervisor.

    No ambient PYTHONPATH or automatic user site is inherited. MLX is a
    user-site-only install on the release host, so its already verified
    distribution root (and NumPy's) is explicitly allowlisted instead.
    """

    if not isinstance(run_nonce, str) or len(run_nonce) != 64:
        raise StageEditMlxSmokeRunnerError("worker run nonce is invalid")
    try:
        bytes.fromhex(run_nonce)
    except ValueError as exc:
        raise StageEditMlxSmokeRunnerError("worker run nonce is invalid") from exc
    expected_entries = _worker_pythonpath_entries(repo)
    recorded_entries = runtime_provenance.get("worker_pythonpath_entries")
    if recorded_entries != expected_entries:
        raise StageEditMlxSmokeRunnerError(
            "runtime provenance worker import roots differ live"
        )
    inherited_environment_keys = (
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "PATH",
        "SHELL",
        "TMPDIR",
        "USER",
        "__CF_USER_TEXT_ENCODING",
    )
    environment = {
        key: os.environ[key]
        for key in inherited_environment_keys
        if key in os.environ
    }
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(expected_entries),
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": _worker_pycache_prefix(
                hashlib.sha256(run_nonce.encode("ascii")).hexdigest()
            ),
            "PYTHONHASHSEED": "0",
            WORKER_RUN_NONCE_ENV: run_nonce,
        }
    )
    return environment


def _git_output(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ("git", *args),
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise StageEditMlxSmokeRunnerError(
            f"git {' '.join(args)} failed with {result.returncode}"
        )
    return result.stdout


def _git_blob_sha256(repo: Path, commit: str, relative_path: str) -> str | None:
    result = subprocess.run(
        ("git", "show", f"{commit}:{relative_path}"),
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _code_provenance(
    *,
    expected_git_commit: str,
    tracked_diff_policy: str,
) -> tuple[dict[str, Any], Path]:
    runner_path = Path(__file__).resolve()
    repo = Path(
        _git_output(runner_path.parent, "rev-parse", "--show-toplevel")
        .decode("utf-8")
        .strip()
    ).resolve()
    observed = (
        _git_output(repo, "rev-parse", "HEAD").decode("ascii").strip().lower()
    )
    expected = expected_git_commit.strip().lower()
    if (
        len(expected) != 40
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise StageEditMlxSmokeRunnerError("--expected-git-commit must be full SHA-1")
    if tracked_diff_policy not in {"require_clean", "record_development"}:
        raise StageEditMlxSmokeRunnerError("tracked diff policy is invalid")
    diff = _git_output(repo, "diff", "--binary", "HEAD", "--")
    status_porcelain = _git_output(
        repo, "status", "--porcelain=v1", "--untracked-files=all"
    )
    status_lines = status_porcelain.decode(
        "utf-8", errors="replace"
    ).splitlines()
    untracked = sorted(
        line[3:].strip()
        for line in status_lines
        if line.startswith("?? ") and line[3:].strip()
    )
    source_paths = {
        "runner": runner_path,
        "capacity_probe": Path(capacity_probe_module.__file__).resolve(),
        "dynamic_crop_head": Path(dynamic_crop_head_module.__file__).resolve(),
        "transformer": Path(transformer_module.__file__).resolve(),
        "feature_snapshot_loader": Path(
            feature_materializer_module.__file__
        ).resolve(),
        "feature_snapshot_manifest_contract": Path(
            feature_snapshot_job_module.__file__
        ).resolve(),
        "reference_adapter": Path(reference_adapter_module.__file__).resolve(),
        "dynamic_crop_training": Path(
            dynamic_crop_training_module.__file__
        ).resolve(),
        "production_training": Path(
            production_training_module.__file__
        ).resolve(),
        "dynamic_crop_manifest_validator": Path(
            dynamic_crop_job_module.__file__
        ).resolve(),
        "dynamic_crop_supervision_loader": Path(
            dynamic_crop_supervision_module.__file__
        ).resolve(),
    }
    sources: dict[str, dict[str, Any]] = {}
    for role, source_path in source_paths.items():
        pointer = _pointer(source_path, f"{role} source")
        try:
            relative_path = str(source_path.relative_to(repo))
        except ValueError as exc:
            raise StageEditMlxSmokeRunnerError(
                f"{role} source is outside the git repository"
            ) from exc
        blob_sha256 = _git_blob_sha256(repo, observed, relative_path)
        sources[role] = {
            **pointer,
            "repo_relative_path": relative_path,
            "observed_commit_blob_sha256": blob_sha256,
            "matches_observed_commit": blob_sha256 == pointer["file_sha256"],
        }
    dependency_paths = {
        str(Path(pointer["path"]).resolve().relative_to(repo))
        for pointer in sources.values()
    }
    untracked_dependencies = sorted(dependency_paths & set(untracked))
    tracked_clean = not diff
    repository_status_clean = not status_porcelain
    source_files_match_observed_commit = all(
        source["matches_observed_commit"] is True for source in sources.values()
    )
    release_trust = (
        tracked_diff_policy == "require_clean"
        and observed == expected
        and tracked_clean
        and repository_status_clean
        and not untracked_dependencies
        and source_files_match_observed_commit
    )
    return (
        {
            "runner_contract": SMOKE_RUNNER_CONTRACT,
            "expected_git_commit": expected,
            "observed_git_commit": observed,
            "tracked_diff_policy": tracked_diff_policy,
            "tracked_diff_clean": tracked_clean,
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "repository_status_clean": repository_status_clean,
            "repository_status_entry_count": len(status_lines),
            "repository_status_porcelain_sha256": hashlib.sha256(
                status_porcelain
            ).hexdigest(),
            "untracked_dependency_paths": untracked_dependencies,
            "source_files_match_observed_commit": source_files_match_observed_commit,
            "release_trust_authorized": release_trust,
            "source_files": sources,
        },
        repo,
    )


def _emit_worker_event(event: Mapping[str, Any]) -> None:
    overlap = set(event) & set(_WORKER_EVENT_CONTEXT)
    if overlap:
        raise RuntimeError(f"worker event overrides process context: {sorted(overlap)}")
    print(
        WORKER_EVENT_PREFIX
        + _canonical_bytes({**dict(event), **_WORKER_EVENT_CONTEXT}).decode(
            "utf-8"
        ),
        flush=True,
    )


def _require_worker_mlx() -> None:
    if mx is None or nn is None or optim is None or tree_flatten is None:
        detail = f": {_MLX_IMPORT_ERROR}" if _MLX_IMPORT_ERROR else ""
        raise RuntimeError(f"MLX worker dependencies are unavailable{detail}")


def _to_mlx_batch(arrays: Mapping[str, np.ndarray]) -> StageEditTensorBatch:
    assert mx is not None
    return StageEditTensorBatch(
        context_features=mx.array(arrays["context_features"]),
        source_features=mx.array(arrays["source_features"]),
        sequence_mask=mx.array(arrays["sequence_mask"]),
        source_candidate_mask=mx.array(arrays["source_candidate_mask"]),
        hard_source_gate_mask=mx.array(arrays["hard_source_gate_mask"]),
        # No semantic evaluator consumes slot identities in this smoke-only loss.
        source_ids=((), ()),
        sequence_ids=("mlx-smoke-lane-A", "mlx-smoke-lane-B"),
    )


def _to_mlx_base_supervision(
    arrays: Mapping[str, np.ndarray],
) -> StageEditSupervisionBatch:
    assert mx is not None
    return StageEditSupervisionBatch(
        **{name: mx.array(value) for name, value in arrays.items()}
    )


def _to_mlx_dynamic_supervision(
    arrays: Mapping[str, np.ndarray],
) -> StageEditDynamicCropSupervisionBatchV2:
    assert mx is not None
    return StageEditDynamicCropSupervisionBatchV2(
        **{name: mx.array(value) for name, value in arrays.items()},
        sequence_ids=("mlx-smoke-lane-A", "mlx-smoke-lane-B"),
    )


def _all_gradients_finite(gradients: Mapping[str, Any]) -> bool:
    assert mx is not None and tree_flatten is not None
    leaves = [value for _, value in tree_flatten(gradients)]
    return bool(leaves) and all(
        bool(mx.all(mx.isfinite(value)).item()) for value in leaves
    )


def _safe_mlx_memory(name: str) -> int | None:
    if mx is None:
        return None
    try:
        return int(getattr(mx, name)())
    except BaseException:
        return None


def _worker_failure_kind(exc: BaseException) -> str:
    lowered = str(exc).lower()
    if isinstance(exc, MemoryError) or any(
        token in lowered for token in ("out of memory", "memoryerror", "oom")
    ):
        return "out_of_memory"
    return "runtime_error"


def _tree_evidence(tree: Any) -> dict[str, Any]:
    assert mx is not None and tree_flatten is not None
    leaves = list(tree_flatten(tree))
    key_shape_dtype = [
        {
            "key": key,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for key, value in leaves
    ]
    nonfinite_count = sum(
        int(mx.sum(mx.logical_not(mx.isfinite(value))).item())
        for _, value in leaves
    )
    return {
        "leaf_count": len(leaves),
        "element_count": sum(int(value.size) for _, value in leaves),
        "nbytes": sum(int(value.nbytes) for _, value in leaves),
        "key_digest_sha256": _sha256(key_shape_dtype),
        "nonfinite_count": nonfinite_count,
    }


def _sync_memory_stage(stage: str, started_at: float) -> dict[str, Any]:
    assert mx is not None
    mx.synchronize(mx.gpu)
    return {
        "stage": stage,
        "synchronized": True,
        "active_memory_bytes": int(mx.get_active_memory()),
        "cache_memory_bytes": int(mx.get_cache_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "elapsed_seconds": max(0.0, time.perf_counter() - started_at),
    }


def _worker_main(args: argparse.Namespace) -> int:
    global _WORKER_EVENT_CONTEXT
    run_nonce = os.environ.pop(WORKER_RUN_NONCE_ENV, "")
    nonce_is_valid = (
        len(run_nonce) == 64
        and all(character in "0123456789abcdef" for character in run_nonce)
    )
    suite_id = str(args.suite_id or "")
    suite_id_is_valid = (
        len(suite_id) == 64
        and all(character in "0123456789abcdef" for character in suite_id)
    )
    _WORKER_EVENT_CONTEXT = {
        "worker_pid": os.getpid(),
        "supervisor_run_nonce_sha256": (
            hashlib.sha256(run_nonce.encode("ascii")).hexdigest()
            if nonce_is_valid
            else None
        ),
        "supervisor_suite_id": suite_id if suite_id_is_valid else None,
    }
    try:
        if not nonce_is_valid:
            raise StageEditMlxSmokeRunnerError(
                "worker requires a valid one-time supervisor nonce"
            )
        if not suite_id_is_valid:
            raise StageEditMlxSmokeRunnerError(
                "worker requires a valid supervisor suite id"
            )
        repo_hint = Path(__file__).resolve().parents[2]
        _verify_isolated_worker_environment(repo_hint)
        _require_worker_mlx()
        assert mx is not None and nn is not None and optim is not None
        candidates = {
            candidate.candidate_id: candidate
            for candidate in default_dynamic_crop_v2_probe_candidates()
        }
        candidate = candidates.get(args.candidate_id)
        if candidate is None:
            raise StageEditMlxSmokeRunnerError("unknown capacity candidate id")
        manifest_map = _parse_manifest_map(
            args.feature_manifest, argument_name="feature manifest"
        )
        materialization_map = _parse_manifest_map(
            args.materialization_manifest,
            argument_name="materialization manifest",
        )
        dynamic_manifest_map = _parse_manifest_map(
            args.dynamic_crop_manifest, argument_name="dynamic crop manifest"
        )
        feature_provenance, snapshots, references = _load_bbr_feature_dataset(
            manifest_map
        )
        target_provenance, examples, dynamic_targets = _load_bbr_target_bindings(
            materialization_manifest_paths_by_song=materialization_map,
            dynamic_crop_manifest_paths_by_song=dynamic_manifest_map,
            snapshots_by_reference=snapshots,
            references_by_song=references,
            build_training_corpora=True,
        )
        dataset_provenance = {**feature_provenance, **target_provenance}
        lanes = _lane_assignments(references)
        arrays, base_arrays, dynamic_arrays, tensor_provenance = (
            _pack_fixed_lane_training_arrays(
                examples_by_reference=examples,
                dynamic_by_reference=dynamic_targets,
                lane_assignments=lanes,
                candidate=candidate,
            )
        )
        budget = _live_budget()
        code_provenance, repo = _code_provenance(
            expected_git_commit=args.expected_git_commit,
            tracked_diff_policy=args.tracked_diff_policy,
        )
        if repo != repo_hint:
            raise StageEditMlxSmokeRunnerError(
                "worker source repository differs from isolated import root"
            )
        runtime_provenance = _runtime_provenance(budget, repo)
        hardware_provenance = _hardware_provenance(budget)
        observed_hashes = {
            "dataset": _sha256(dataset_provenance),
            "code": _sha256(code_provenance),
            "runtime": _sha256(runtime_provenance),
            "hardware": _sha256(hardware_provenance),
        }
        expected_hashes = {
            "dataset": args.expected_dataset_provenance_sha256,
            "code": args.expected_code_provenance_sha256,
            "runtime": args.expected_runtime_provenance_sha256,
            "hardware": args.expected_hardware_provenance_sha256,
        }
        candidate_contract_sha256 = _sha256(candidate.to_dict())
        provenance_matched = (
            observed_hashes == expected_hashes
            and candidate_contract_sha256
            == args.expected_candidate_contract_sha256
            and code_provenance["observed_git_commit"]
            == args.expected_git_commit.strip().lower()
        )
        _emit_worker_event(
            {
                "event": "provenance_verified",
                "candidate_id": candidate.candidate_id,
                "candidate_contract_sha256": candidate_contract_sha256,
                "expected_dataset_provenance_sha256": expected_hashes[
                    "dataset"
                ],
                "observed_dataset_provenance_sha256": observed_hashes[
                    "dataset"
                ],
                "expected_code_provenance_sha256": expected_hashes["code"],
                "observed_code_provenance_sha256": observed_hashes["code"],
                "expected_runtime_provenance_sha256": expected_hashes[
                    "runtime"
                ],
                "observed_runtime_provenance_sha256": observed_hashes[
                    "runtime"
                ],
                "observed_runtime_modules_sha256": _sha256(
                    runtime_provenance["imported_runtime_modules"]
                ),
                "observed_runtime_distributions_sha256": _sha256(
                    runtime_provenance["runtime_distributions"]
                ),
                "worker_environment_contract": runtime_provenance[
                    "worker_environment_contract"
                ],
                "worker_pythonpath_entries": runtime_provenance[
                    "worker_pythonpath_entries"
                ],
                "expected_hardware_provenance_sha256": expected_hashes[
                    "hardware"
                ],
                "observed_hardware_provenance_sha256": observed_hashes[
                    "hardware"
                ],
                "expected_git_commit": args.expected_git_commit.strip().lower(),
                "observed_git_commit": code_provenance["observed_git_commit"],
                "matched": provenance_matched,
            }
        )
        if not provenance_matched:
            raise RuntimeError(
                "worker dataset/code/runtime/hardware provenance handshake differs"
            )
        previous_memory_limit = int(mx.set_memory_limit(budget.allowed_training_bytes))
        previous_cache_limit = int(mx.set_cache_limit(SMOKE_CACHE_LIMIT_BYTES))
        started_at = time.perf_counter()
        mx.clear_cache()
        mx.synchronize(mx.gpu)
        pre_reset_active = int(mx.get_active_memory())
        pre_reset_cache = int(mx.get_cache_memory())
        _emit_worker_event(
            {
                "event": "input_and_limits_ready",
                "tensor_provenance": tensor_provenance,
                "previous_memory_limit_bytes": previous_memory_limit,
                "applied_memory_limit_bytes": budget.allowed_training_bytes,
                "previous_cache_limit_bytes": previous_cache_limit,
                "applied_cache_limit_bytes": SMOKE_CACHE_LIMIT_BYTES,
                "memory_limit_contract": SMOKE_MEMORY_LIMIT_CONTRACT,
                "pre_reset_active_memory_bytes": pre_reset_active,
                "pre_reset_cache_memory_bytes": pre_reset_cache,
                "elapsed_seconds": max(0.0, time.perf_counter() - started_at),
            }
        )
        if pre_reset_active != 0:
            raise RuntimeError(
                "fresh MLX worker has nonzero active memory before peak reset"
            )
        mx.reset_peak_memory()
        mx.random.seed(int(PROBE_TRAINING_SCHEDULE["seed"]))
        batch = _to_mlx_batch(arrays)
        base_supervision = _to_mlx_base_supervision(base_arrays)
        dynamic_supervision = _to_mlx_dynamic_supervision(dynamic_arrays)
        mx.eval(
            batch.context_features,
            batch.source_features,
            batch.sequence_mask,
            batch.source_candidate_mask,
            batch.hard_source_gate_mask,
            base_supervision.cut_targets,
            base_supervision.cut_mask,
            base_supervision.source_targets,
            base_supervision.source_mask,
            base_supervision.rejected_source_targets,
            base_supervision.pairwise_source_mask,
            base_supervision.duration_targets,
            base_supervision.duration_mask,
            base_supervision.transition_targets,
            base_supervision.transition_mask,
            base_supervision.crop_targets,
            base_supervision.crop_mask,
            dynamic_supervision.values,
            dynamic_supervision.normalized_times,
            dynamic_supervision.keyframe_mask,
            dynamic_supervision.presence_targets,
            dynamic_supervision.presence_mask,
            dynamic_supervision.motion_targets,
            dynamic_supervision.motion_mask,
            dynamic_supervision.shot_duration_seconds,
            dynamic_supervision.track_mask,
            dynamic_supervision.keyframe_count_targets,
        )
        input_stage = _sync_memory_stage("input_materialized", started_at)
        _emit_worker_event(
            {"event": "input_materialized", "memory_stage": input_stage}
        )
        model = StageEditTransformerDynamicCropV2(
            candidate.model_config, candidate.dynamic_crop_config
        )
        model.train()
        mx.eval(model.parameters())
        model_stage = _sync_memory_stage("model_allocated", started_at)
        parameter_tree = _tree_evidence(model.parameters())
        _emit_worker_event(
            {
                "event": "model_allocated",
                "memory_stage": model_stage,
                "parameter_tree": parameter_tree,
            }
        )
        optimizer = optim.AdamW(
            learning_rate=float(PROBE_TRAINING_SCHEDULE["learning_rate"]),
            weight_decay=float(PROBE_TRAINING_SCHEDULE["weight_decay"]),
        )

        def loss_function(
            trained_model: StageEditTransformerDynamicCropV2,
            smoke_batch: StageEditTensorBatch,
            smoke_base: StageEditSupervisionBatch,
            smoke_dynamic: StageEditDynamicCropSupervisionBatchV2,
        ) -> Any:
            return stage_edit_dynamic_crop_v2_loss(
                trained_model,
                smoke_batch,
                smoke_base,
                smoke_dynamic,
                candidate.dynamic_crop_loss_contract,
            )[0]

        loss_and_grad = nn.value_and_grad(model, loss_function)
        loss, gradients = loss_and_grad(
            model, batch, base_supervision, dynamic_supervision
        )
        mx.eval(loss, gradients)
        backward_stage = _sync_memory_stage(
            "forward_backward_completed", started_at
        )
        loss_value = float(loss.item())
        gradient_tree = _tree_evidence(gradients)
        _emit_worker_event(
            {
                "event": "forward_backward_completed",
                "loss_value_for_execution_only": loss_value,
                "gradient_tree": gradient_tree,
                "memory_stage": backward_stage,
            }
        )
        clipped_gradients, gradient_norm = optim.clip_grad_norm(
            gradients, float(PROBE_TRAINING_SCHEDULE["gradient_clip_norm"])
        )
        mx.eval(clipped_gradients, gradient_norm)
        clip_stage = _sync_memory_stage("gradient_clip_completed", started_at)
        gradient_norm_value = float(gradient_norm.item())
        _emit_worker_event(
            {
                "event": "gradient_clip_completed",
                "gradient_global_norm_before_clip": gradient_norm_value,
                "memory_stage": clip_stage,
            }
        )
        optimizer.update(model, clipped_gradients)
        mx.eval(model.parameters(), optimizer.state)
        optimizer_stage = _sync_memory_stage(
            "optimizer_step_completed", started_at
        )
        gradients_finite = (
            math.isfinite(loss_value)
            and math.isfinite(gradient_norm_value)
            and _all_gradients_finite(gradients)
        )
        optimizer_state_materialized = bool(tree_flatten(optimizer.state))
        post_update_parameter_tree = _tree_evidence(model.parameters())
        optimizer_tree = _tree_evidence(optimizer.state)
        updated_parameters_finite = (
            post_update_parameter_tree["nonfinite_count"] == 0
        )
        optimizer_state_finite = optimizer_tree["nonfinite_count"] == 0
        peak = optimizer_stage["peak_memory_bytes"]
        _emit_worker_event(
            {
                "event": "optimizer_step_completed",
                "loss_value_for_execution_only": loss_value,
                "gradient_global_norm_before_clip": gradient_norm_value,
                "gradients_finite": gradients_finite,
                "optimizer_state_materialized": optimizer_state_materialized,
                "post_update_parameter_tree": post_update_parameter_tree,
                "updated_parameters_finite": updated_parameters_finite,
                "optimizer_state_tree": optimizer_tree,
                "optimizer_state_finite": optimizer_state_finite,
                "memory_stage": optimizer_stage,
                "measured_peak_bytes": peak,
                "elapsed_seconds": optimizer_stage["elapsed_seconds"],
                "smoke_target_contract": SMOKE_TARGET_CONTRACT,
                "semantic_metrics_authorized": False,
            }
        )
        return (
            0
            if gradients_finite
            and optimizer_state_materialized
            and updated_parameters_finite
            and optimizer_state_finite
            else 71
        )
    except BaseException as exc:  # Preserve even OOM/runtime failures as evidence.
        message = str(exc)
        failure_kind = _worker_failure_kind(exc)
        _emit_worker_event(
            {
                "event": "worker_failure",
                "failure_kind": failure_kind,
                "exception_type": type(exc).__name__,
                "exception_message": message[:4000],
                "measured_peak_bytes": (
                    _safe_mlx_memory("get_peak_memory")
                ),
            }
        )
        return 70


def _parse_worker_events(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.startswith(WORKER_EVENT_PREFIX):
            continue
        try:
            value = json.loads(line[len(WORKER_EVENT_PREFIX) :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _write_runner_log(
    destination: Path,
    *,
    candidate_id: str,
    command: Sequence[str],
    returncode: int,
    stdout: str,
    stderr: str,
    supervisor_pid: int,
    worker_pid: int,
    run_nonce_sha256: str,
    started_at_unix_ns: int,
    ended_at_unix_ns: int,
    suite_id: str,
    suite_started_at_unix_ns: int,
    suite_ended_at_unix_ns: int,
) -> Path:
    worker_events = _parse_worker_events(stdout)
    receipt_core = {
        "candidate_id": candidate_id,
        "command": list(command),
        "supervisor_pid": supervisor_pid,
        "worker_pid": worker_pid,
        "returncode": returncode,
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "run_nonce_sha256": run_nonce_sha256,
        "started_at_unix_ns": started_at_unix_ns,
        "ended_at_unix_ns": ended_at_unix_ns,
        "suite_id": suite_id,
        "suite_started_at_unix_ns": suite_started_at_unix_ns,
        "suite_ended_at_unix_ns": suite_ended_at_unix_ns,
    }
    unsigned = {
        "schema_version": RUNNER_LOG_SCHEMA_VERSION,
        "analysis_basis": RUNNER_LOG_ANALYSIS_BASIS,
        "candidate_id": candidate_id,
        "command": list(command),
        "supervisor_pid": supervisor_pid,
        "worker_pid": worker_pid,
        "returncode": returncode,
        "stdout": stdout,
        "worker_event_count": len(worker_events),
        "stdout_event_sha256": _sha256(worker_events),
        "stderr": stderr,
        "run_nonce_sha256": run_nonce_sha256,
        "subprocess_receipt_sha256": _sha256(receipt_core),
        "started_at_unix_ns": started_at_unix_ns,
        "ended_at_unix_ns": ended_at_unix_ns,
        "suite_id": suite_id,
        "suite_started_at_unix_ns": suite_started_at_unix_ns,
        "suite_ended_at_unix_ns": suite_ended_at_unix_ns,
        "artifact_content_sha256": None,
    }
    digest = _sha256(unsigned)
    payload = {**unsigned, "artifact_content_sha256": digest}
    return _atomic_exclusive_write(
        destination / "logs" / f"{RUNNER_LOG_PREFIX}{digest}.json", payload
    )


def _execution_from_process(
    *,
    candidate: DynamicCropCapacityCandidate,
    budget: DeviceMemoryBudget,
    code_provenance: Mapping[str, Any],
    provenance_expectations: Mapping[str, str],
    tensor_template: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    returncode: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tensor = dict(tensor_template)
    limits: dict[str, Any] = {
        "previous_memory_limit_bytes": None,
        "applied_memory_limit_bytes": None,
        "previous_cache_limit_bytes": None,
        "applied_cache_limit_bytes": None,
        "memory_limit_contract": SMOKE_MEMORY_LIMIT_CONTRACT,
        "pre_reset_active_memory_bytes": None,
        "pre_reset_cache_memory_bytes": None,
    }
    model_allocated = False
    forward_backward_event: Mapping[str, Any] | None = None
    clip_event: Mapping[str, Any] | None = None
    final_event: Mapping[str, Any] | None = None
    worker_failure: Mapping[str, Any] | None = None
    peak_values: list[int] = []
    sync_stages: list[dict[str, Any]] = []
    parameter_tree = None
    gradient_tree = None
    optimizer_state_tree = None
    post_update_parameter_tree = None
    provenance_event: Mapping[str, Any] | None = None
    for event in events:
        event_name = event.get("event")
        peak = event.get("measured_peak_bytes")
        if isinstance(peak, int) and not isinstance(peak, bool) and peak > 0:
            peak_values.append(peak)
        memory_stage = event.get("memory_stage")
        if isinstance(memory_stage, Mapping):
            normalized_stage = dict(memory_stage)
            sync_stages.append(normalized_stage)
            stage_peak = normalized_stage.get("peak_memory_bytes")
            if (
                isinstance(stage_peak, int)
                and not isinstance(stage_peak, bool)
                and stage_peak > 0
            ):
                peak_values.append(stage_peak)
        if event_name == "provenance_verified":
            provenance_event = event
        elif event_name == "input_and_limits_ready":
            candidate_tensor = event.get("tensor_provenance")
            if isinstance(candidate_tensor, Mapping):
                tensor = dict(candidate_tensor)
            for key in tuple(limits):
                if key in event:
                    limits[key] = event[key]
        elif event_name == "model_allocated":
            model_allocated = True
            parameter_tree = event.get("parameter_tree")
        elif event_name == "forward_backward_completed":
            forward_backward_event = event
            gradient_tree = event.get("gradient_tree")
        elif event_name == "gradient_clip_completed":
            clip_event = event
        elif event_name == "optimizer_step_completed":
            final_event = event
            optimizer_state_tree = event.get("optimizer_state_tree")
            post_update_parameter_tree = event.get(
                "post_update_parameter_tree"
            )
        elif event_name == "worker_failure":
            worker_failure = event
    termination_signal = -returncode if returncode < 0 else None
    process_exit_code = returncode if returncode >= 0 else None
    expected_candidate_hash = provenance_expectations.get(
        "candidate_contract_sha256"
    )
    expected_dataset_hash = provenance_expectations.get(
        "dataset_provenance_sha256"
    )
    expected_code_hash = provenance_expectations.get("code_provenance_sha256")
    expected_runtime_hash = provenance_expectations.get(
        "runtime_provenance_sha256"
    )
    expected_hardware_hash = provenance_expectations.get(
        "hardware_provenance_sha256"
    )
    expected_git_commit = provenance_expectations.get("git_commit")
    provenance_matched = (
        provenance_event is not None
        and provenance_event.get("matched") is True
        and provenance_event.get("candidate_id") == candidate.candidate_id
        and provenance_event.get("candidate_contract_sha256")
        == expected_candidate_hash
        and provenance_event.get("expected_dataset_provenance_sha256")
        == expected_dataset_hash
        and provenance_event.get("observed_dataset_provenance_sha256")
        == expected_dataset_hash
        and provenance_event.get("expected_code_provenance_sha256")
        == expected_code_hash
        and provenance_event.get("observed_code_provenance_sha256")
        == expected_code_hash
        and provenance_event.get("expected_runtime_provenance_sha256")
        == expected_runtime_hash
        and provenance_event.get("observed_runtime_provenance_sha256")
        == expected_runtime_hash
        and provenance_event.get("expected_hardware_provenance_sha256")
        == expected_hardware_hash
        and provenance_event.get("observed_hardware_provenance_sha256")
        == expected_hardware_hash
        and provenance_event.get("expected_git_commit") == expected_git_commit
        and provenance_event.get("observed_git_commit") == expected_git_commit
    )
    provenance_handshake = {
        "candidate_id": candidate.candidate_id,
        "candidate_contract_sha256": expected_candidate_hash,
        "supervisor_dataset_provenance_sha256": expected_dataset_hash,
        "worker_dataset_provenance_sha256": (
            provenance_event.get("observed_dataset_provenance_sha256")
            if provenance_event is not None
            else None
        ),
        "supervisor_code_provenance_sha256": expected_code_hash,
        "worker_code_provenance_sha256": (
            provenance_event.get("observed_code_provenance_sha256")
            if provenance_event is not None
            else None
        ),
        "supervisor_runtime_provenance_sha256": expected_runtime_hash,
        "worker_runtime_provenance_sha256": (
            provenance_event.get("observed_runtime_provenance_sha256")
            if provenance_event is not None
            else None
        ),
        "supervisor_hardware_provenance_sha256": expected_hardware_hash,
        "worker_hardware_provenance_sha256": (
            provenance_event.get("observed_hardware_provenance_sha256")
            if provenance_event is not None
            else None
        ),
        "expected_git_commit": expected_git_commit,
        "worker_observed_git_commit": (
            provenance_event.get("observed_git_commit")
            if provenance_event is not None
            else None
        ),
        "matched": provenance_matched,
    }
    if final_event is not None:
        gradients_finite = final_event.get("gradients_finite")
        optimizer_state_materialized = final_event.get(
            "optimizer_state_materialized"
        )
        updated_parameters_finite = final_event.get(
            "updated_parameters_finite"
        )
        optimizer_state_finite = final_event.get("optimizer_state_finite")
        loss_value = (
            forward_backward_event.get("loss_value_for_execution_only")
            if forward_backward_event is not None
            else None
        )
        gradient_norm = (
            clip_event.get("gradient_global_norm_before_clip")
            if clip_event is not None
            else None
        )
    else:
        gradients_finite = None
        optimizer_state_materialized = False
        updated_parameters_finite = None
        optimizer_state_finite = None
        loss_value = None
        gradient_norm = None
    forward_backward_completed = forward_backward_event is not None
    gradient_clip_completed = clip_event is not None
    optimizer_step_completed = final_event is not None
    peak = max(peak_values) if peak_values else None
    runtime_pass = (
        model_allocated
        and forward_backward_completed
        and gradient_clip_completed
        and optimizer_step_completed
        and provenance_matched
        and limits["pre_reset_active_memory_bytes"] == 0
        and [stage.get("stage") for stage in sync_stages]
        == [
            "input_materialized",
            "model_allocated",
            "forward_backward_completed",
            "gradient_clip_completed",
            "optimizer_step_completed",
        ]
        and gradients_finite is True
        and optimizer_state_materialized is True
        and updated_parameters_finite is True
        and optimizer_state_finite is True
        and process_exit_code == 0
        and termination_signal is None
        and isinstance(loss_value, (int, float))
        and math.isfinite(float(loss_value))
        and float(loss_value) >= 0.0
        and isinstance(gradient_norm, (int, float))
        and math.isfinite(float(gradient_norm))
        and float(gradient_norm) >= 0.0
        and peak is not None
    )
    trusted = code_provenance["release_trust_authorized"] is True
    passed = runtime_pass and trusted and peak <= budget.allowed_training_bytes
    if passed:
        failure_kind = None
    elif runtime_pass and peak is not None and peak > budget.allowed_training_bytes:
        failure_kind = "budget_exceeded"
    elif runtime_pass and not trusted:
        failure_kind = "provenance_not_release_trusted"
    elif termination_signal is not None:
        failure_kind = "process_signal"
    elif worker_failure is not None and worker_failure.get("failure_kind") == "out_of_memory":
        failure_kind = "out_of_memory"
    elif final_event is not None:
        failure_kind = "non_finite_loss_or_gradients"
    else:
        failure_kind = "runtime_error"
    execution = {
        "model_allocated": model_allocated,
        "forward_completed": forward_backward_completed,
        "loss_completed": forward_backward_completed,
        "backward_completed": forward_backward_completed,
        "gradients_finite": gradients_finite,
        "gradient_global_norm_before_clip": (
            float(gradient_norm) if gradient_norm is not None else None
        ),
        "gradient_clip_norm": PROBE_TRAINING_SCHEDULE["gradient_clip_norm"],
        "gradient_clip_completed": gradient_clip_completed,
        "optimizer_step_completed": optimizer_step_completed,
        "optimizer_state_materialized": bool(optimizer_state_materialized),
        "parameter_tree": parameter_tree,
        "gradient_tree": gradient_tree,
        "post_update_parameter_tree": post_update_parameter_tree,
        "updated_parameters_finite": updated_parameters_finite,
        "optimizer_state_tree": optimizer_state_tree,
        "optimizer_state_finite": optimizer_state_finite,
        "provenance_handshake": provenance_handshake,
        "optimizer_name": "mlx.optimizers.AdamW",
        "optimizer_config": {
            "learning_rate": PROBE_TRAINING_SCHEDULE["learning_rate"],
            "weight_decay": PROBE_TRAINING_SCHEDULE["weight_decay"],
        },
        "loss_contract_sha256": candidate.contract_dict()[
            "dynamic_crop_loss_contract_sha256"
        ],
        "loss_value_for_execution_only": (
            float(loss_value) if loss_value is not None else None
        ),
        **limits,
        "sync_stages": sync_stages,
        "final_active_memory_bytes": (
            sync_stages[-1].get("active_memory_bytes") if sync_stages else None
        ),
        "final_cache_memory_bytes": (
            sync_stages[-1].get("cache_memory_bytes") if sync_stages else None
        ),
        "elapsed_seconds": (
            float(sync_stages[-1].get("elapsed_seconds"))
            if sync_stages
            and isinstance(sync_stages[-1].get("elapsed_seconds"), (int, float))
            else None
        ),
        "measured_peak_bytes": peak,
        "allowed_training_bytes": budget.allowed_training_bytes,
        "reserve_after_peak_bytes": (
            budget.allowed_training_bytes - peak if peak is not None else None
        ),
        "runtime_execution_passed": runtime_pass,
        "release_trust_authorized": trusted,
        "passed": passed,
        "failure_kind": failure_kind,
        "process_exit_code": process_exit_code,
        "termination_signal": termination_signal,
    }
    return tensor, execution


def _external_destination(path: Path) -> Path:
    root = EXTERNAL_VOLUME_ROOT.resolve()
    destination = Path(path).expanduser().resolve()
    if not root.is_dir() or not os.path.ismount(root):
        raise StageEditMlxSmokeRunnerError("/Volumes/CrossCutMedia is not a mounted volume")
    if destination != root and root not in destination.parents:
        raise StageEditMlxSmokeRunnerError(
            "MLX smoke artifacts must be written under /Volumes/CrossCutMedia"
        )
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise StageEditMlxSmokeRunnerError("smoke destination must not be a symlink")
    return destination


def _supervisor_main(args: argparse.Namespace) -> int:
    destination = _external_destination(args.destination)
    manifest_map = _parse_manifest_map(
        args.feature_manifest, argument_name="feature manifest"
    )
    materialization_map = _parse_manifest_map(
        args.materialization_manifest, argument_name="materialization manifest"
    )
    dynamic_manifest_map = _parse_manifest_map(
        args.dynamic_crop_manifest, argument_name="dynamic crop manifest"
    )
    dataset, snapshots, references = _load_bbr_feature_dataset(manifest_map)
    target_provenance, _, _ = _load_bbr_target_bindings(
        materialization_manifest_paths_by_song=materialization_map,
        dynamic_crop_manifest_paths_by_song=dynamic_manifest_map,
        snapshots_by_reference=snapshots,
        references_by_song=references,
        build_training_corpora=False,
    )
    dataset.update(target_provenance)
    lanes = _lane_assignments(references)
    budget = _live_budget()
    code, repo = _code_provenance(
        expected_git_commit=args.expected_git_commit,
        tracked_diff_policy=args.tracked_diff_policy,
    )
    runtime = _runtime_provenance(budget, repo)
    hardware = _hardware_provenance(budget)
    all_candidates = default_dynamic_crop_v2_probe_candidates()
    by_id = {candidate.candidate_id: candidate for candidate in all_candidates}
    if len(all_candidates) != 4 or len(by_id) != 4:
        raise StageEditMlxSmokeRunnerError(
            "release MLX smoke requires the four pinned capacity candidates"
        )
    selected_ids = tuple(args.candidate_id or tuple(by_id))
    if (
        len(selected_ids) != 4
        or len(set(selected_ids)) != 4
        or set(selected_ids) != set(by_id)
    ):
        raise StageEditMlxSmokeRunnerError(
            "supervisor must run all four pinned candidates exactly once"
        )
    suite_id = secrets.token_hex(32)
    suite_started_at_unix_ns = time.time_ns()
    supervisor_pid = os.getpid()
    process_results: list[dict[str, Any]] = []
    for candidate_id in selected_ids:
        candidate = by_id[candidate_id]
        tensor_template = _tensor_provenance_template(
            snapshots, lanes, candidate
        )
        provenance_expectations = {
            "candidate_contract_sha256": _sha256(candidate.to_dict()),
            "dataset_provenance_sha256": _sha256(dataset),
            "code_provenance_sha256": _sha256(code),
            "runtime_provenance_sha256": _sha256(runtime),
            "hardware_provenance_sha256": _sha256(hardware),
            "git_commit": code["expected_git_commit"],
        }
        command = [
            sys.executable,
            "-P",
            "-S",
            "-m",
            "crosscut.stage_edit_dynamic_crop_mlx_smoke_cli",
            "--worker",
            "--candidate-id",
            candidate_id,
            "--suite-id",
            suite_id,
            "--expected-candidate-contract-sha256",
            provenance_expectations["candidate_contract_sha256"],
            "--expected-dataset-provenance-sha256",
            provenance_expectations["dataset_provenance_sha256"],
            "--expected-code-provenance-sha256",
            provenance_expectations["code_provenance_sha256"],
            "--expected-runtime-provenance-sha256",
            provenance_expectations["runtime_provenance_sha256"],
            "--expected-hardware-provenance-sha256",
            provenance_expectations["hardware_provenance_sha256"],
            "--expected-git-commit",
            args.expected_git_commit,
            "--tracked-diff-policy",
            args.tracked_diff_policy,
        ]
        for song in SELECTION_SONG_IDS:
            command.extend(
                ["--feature-manifest", f"{song}={manifest_map[song].resolve()}"]
            )
            command.extend(
                [
                    "--materialization-manifest",
                    f"{song}={materialization_map[song].resolve()}",
                ]
            )
            command.extend(
                [
                    "--dynamic-crop-manifest",
                    f"{song}={dynamic_manifest_map[song].resolve()}",
                ]
            )
        run_nonce = secrets.token_hex(32)
        run_nonce_sha256 = hashlib.sha256(
            run_nonce.encode("ascii")
        ).hexdigest()
        environment = _build_isolated_worker_environment(
            repo=repo,
            runtime_provenance=runtime,
            run_nonce=run_nonce,
        )
        started_at_unix_ns = time.time_ns()
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate()
        ended_at_unix_ns = time.time_ns()
        returncode = int(process.returncode)
        events = _parse_worker_events(stdout)
        tensor, execution = _execution_from_process(
            candidate=candidate,
            budget=budget,
            code_provenance=code,
            provenance_expectations=provenance_expectations,
            tensor_template=tensor_template,
            events=events,
            returncode=returncode,
        )
        process_results.append(
            {
                "candidate": candidate,
                "candidate_id": candidate_id,
                "command": command,
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
                "worker_pid": process.pid,
                "run_nonce": run_nonce,
                "run_nonce_sha256": run_nonce_sha256,
                "started_at_unix_ns": started_at_unix_ns,
                "ended_at_unix_ns": ended_at_unix_ns,
                "tensor": tensor,
                "execution": execution,
            }
        )

    suite_ended_at_unix_ns = time.time_ns()
    outputs: list[Path] = []
    log_paths: dict[str, Path] = {}
    all_passed = True
    for result in process_results:
        candidate = result["candidate"]
        candidate_id = result["candidate_id"]
        execution = result["execution"]
        log_path = _write_runner_log(
            destination,
            candidate_id=candidate_id,
            command=result["command"],
            returncode=result["returncode"],
            stdout=result["stdout"],
            stderr=result["stderr"],
            supervisor_pid=supervisor_pid,
            worker_pid=result["worker_pid"],
            run_nonce_sha256=result["run_nonce_sha256"],
            started_at_unix_ns=result["started_at_unix_ns"],
            ended_at_unix_ns=result["ended_at_unix_ns"],
            suite_id=suite_id,
            suite_started_at_unix_ns=suite_started_at_unix_ns,
            suite_ended_at_unix_ns=suite_ended_at_unix_ns,
        )
        evidence_path = _publish_supervised_mlx_smoke_evidence(
            destination,
            candidate=candidate,
            budget=budget,
            evidence={
                "semantic_metrics_authorized": False,
                "candidate": candidate.to_dict(),
                "dataset_provenance": dataset,
                "code_provenance": code,
                "runtime_provenance": runtime,
                "hardware_provenance": hardware,
                "tensor_provenance": result["tensor"],
                "execution": execution,
                "runner_log": _pointer(log_path, "MLX smoke runner log"),
            },
            runner_log_path=log_path,
            run_nonce=result["run_nonce"],
        )
        outputs.append(evidence_path)
        log_paths[candidate_id] = log_path
        all_passed = all_passed and execution["passed"] is True
        print(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "candidate_name": candidate.candidate_name,
                    "evidence_path": str(evidence_path),
                    "runtime_execution_passed": execution[
                        "runtime_execution_passed"
                    ],
                    "release_trust_authorized": execution[
                        "release_trust_authorized"
                    ],
                    "passed": execution["passed"],
                    "failure_kind": execution["failure_kind"],
                    "measured_peak_bytes": execution["measured_peak_bytes"],
                    "allowed_training_bytes": budget.allowed_training_bytes,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    suite_unsigned = {
        "schema_version": SMOKE_SUITE_SCHEMA_VERSION,
        "analysis_basis": SMOKE_SUITE_ANALYSIS_BASIS,
        "suite_id": suite_id,
        "supervisor_pid": supervisor_pid,
        "suite_started_at_unix_ns": suite_started_at_unix_ns,
        "suite_ended_at_unix_ns": suite_ended_at_unix_ns,
        "candidate_ids": sorted(by_id),
        "smoke_evidence": {
            result["candidate_id"]: _pointer(path, "MLX smoke evidence")
            for result, path in zip(process_results, outputs, strict=True)
        },
        "runner_logs": {
            candidate_id: _pointer(log_paths[candidate_id], "MLX smoke runner log")
            for candidate_id in sorted(log_paths)
        },
        "all_release_smokes_passed": all_passed,
        "artifact_content_sha256": None,
    }
    suite_digest = _sha256(suite_unsigned)
    suite_path = _atomic_exclusive_write(
        destination / f"{SMOKE_SUITE_PREFIX}{suite_digest}.json",
        {**suite_unsigned, "artifact_content_sha256": suite_digest},
    )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_count": len(outputs),
                "all_release_smokes_passed": all_passed,
                "suite_id": suite_id,
                "suite_path": str(suite_path),
                "evidence_paths": [str(path) for path in outputs],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if all_passed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run each Dynamic Crop v2 capacity candidate in a fresh MLX process "
            "against the exact BBR-15 fixed release tensor."
        )
    )
    parser.add_argument(
        "--feature-manifest",
        action="append",
        required=True,
        metavar="SONG_ID=/ABSOLUTE/PATH.json",
    )
    parser.add_argument(
        "--materialization-manifest",
        action="append",
        required=True,
        metavar="SONG_ID=/ABSOLUTE/PATH.json",
    )
    parser.add_argument(
        "--dynamic-crop-manifest",
        action="append",
        required=True,
        metavar="SONG_ID=/ABSOLUTE/PATH.json",
    )
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        help="repeat for all four pinned candidates, or omit to run all four",
    )
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--expected-git-commit")
    parser.add_argument(
        "--expected-candidate-contract-sha256", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--expected-dataset-provenance-sha256", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--expected-code-provenance-sha256", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--expected-runtime-provenance-sha256", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--expected-hardware-provenance-sha256", help=argparse.SUPPRESS
    )
    parser.add_argument("--suite-id", help=argparse.SUPPRESS)
    parser.add_argument(
        "--tracked-diff-policy",
        choices=("require_clean", "record_development"),
        default="require_clean",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.worker:
            if len(args.candidate_id) != 1:
                raise StageEditMlxSmokeRunnerError(
                    "worker requires exactly one --candidate-id"
                )
            required_worker_values = {
                "expected_git_commit": args.expected_git_commit,
                "expected_candidate_contract_sha256": args.expected_candidate_contract_sha256,
                "expected_dataset_provenance_sha256": args.expected_dataset_provenance_sha256,
                "expected_code_provenance_sha256": args.expected_code_provenance_sha256,
                "expected_runtime_provenance_sha256": args.expected_runtime_provenance_sha256,
                "expected_hardware_provenance_sha256": args.expected_hardware_provenance_sha256,
                "suite_id": args.suite_id,
            }
            if any(
                not isinstance(value, str) or not value.strip()
                for value in required_worker_values.values()
            ):
                raise StageEditMlxSmokeRunnerError(
                    "worker requires the complete supervisor provenance handshake"
                )
            args.candidate_id = args.candidate_id[0]
            return _worker_main(args)
        if args.destination is None or args.expected_git_commit is None:
            raise StageEditMlxSmokeRunnerError(
                "supervisor requires --destination and --expected-git-commit"
            )
        return _supervisor_main(args)
    except (DynamicCropCapacityProbeError, StageEditMlxSmokeRunnerError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FIXED_BATCH_SIZE",
    "FIXED_MAX_SOURCES",
    "FIXED_SEQUENCE_LENGTH",
    "FIXED_SOURCE_DIMENSION",
    "SMOKE_TARGET_CONTRACT",
    "StageEditMlxSmokeRunnerError",
    "build_parser",
    "main",
]
