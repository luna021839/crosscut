"""Immutable materialization of current StageEdit reference supervision.

The expensive reverse-attribution and crop analyses are intentionally kept
immutable.  When the dataset/supervision adapters evolve, this job rebuilds
only their small JSON products from the exact final attribution and verified
crop artifacts.  It never rewrites the original evidence directory and it
records a content-addressed input fingerprint so an interrupted run can be
resumed without silently mixing adapter schemas.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .reference_crop_supervision import load_reference_crop_supervision
from .reference_dataset import SCHEMA_VERSION as REFERENCE_DATASET_SCHEMA_VERSION
from .reference_dataset import build_reference_dataset
from .reference_supervision import (
    SCHEMA_VERSION as REFERENCE_SUPERVISION_SCHEMA_VERSION,
)
from .reference_supervision import build_reference_supervision


SCHEMA_VERSION = 1
ANALYSIS_BASIS = "immutable_reference_training_materialization_v1"
_REFERENCE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True, slots=True)
class ReferenceTrainingMaterializationResult:
    manifest_path: Path
    dataset_paths: Mapping[str, Path]
    supervision_paths: Mapping[str, Path]
    input_fingerprint_sha256: str


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} root must be an object: {path}")
    return dict(payload)


def _safe_reference_id(value: object) -> str:
    reference_id = str(value or "").strip()
    if not _REFERENCE_ID.fullmatch(reference_id):
        raise ValueError(f"unsafe reference id: {reference_id!r}")
    return reference_id


def _source_count(attribution: Mapping[str, Any]) -> int:
    sources = attribution.get("sources")
    if not isinstance(sources, list):
        raise ValueError("final attribution must contain a source catalog")
    source_ids: set[str] = set()
    for index, item in enumerate(sources):
        if not isinstance(item, Mapping):
            raise ValueError(f"source catalog entry {index} must be an object")
        source_id = str(item.get("video_id") or item.get("source_id") or "").strip()
        if not source_id or source_id in source_ids:
            raise ValueError("source catalog contains a blank or duplicate id")
        source_ids.add(source_id)
    return len(source_ids)


def _known_shot_indices(attribution: Mapping[str, Any]) -> set[int]:
    shots = attribution.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("final attribution must contain non-empty shots")
    known: set[int] = set()
    for fallback_index, item in enumerate(shots):
        if not isinstance(item, Mapping):
            raise ValueError(f"shot {fallback_index} must be an object")
        source_id = str(item.get("attributed_video_id") or "").strip()
        if not source_id:
            continue
        raw_index = item.get("shot_index", fallback_index)
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError("attributed shot index must be an integer")
        if raw_index != fallback_index:
            raise ValueError("attributed shot indices must be contiguous")
        known.add(raw_index)
    if not known:
        raise ValueError("final attribution contains no retained known shots")
    return known


def _discover_crop_artifacts(
    reference_id: str,
    crop_directories: Sequence[Path],
    *,
    expected_shot_indices: set[int],
) -> tuple[tuple[Path, ...], tuple[Path, ...], list[dict[str, Any]]]:
    pattern = (
        f"{reference_id}-reference-crop-supervision-crop-v1-shot-*.json"
    )
    candidates: dict[Path, None] = {}
    for directory in crop_directories:
        for path in directory.glob(pattern):
            if path.name.startswith("._") or not path.is_file():
                continue
            candidates[path.resolve()] = None
    if not candidates:
        raise ValueError(f"no crop artifacts found for {reference_id}")

    by_shot: dict[int, tuple[Path, bool]] = {}
    manifest: list[dict[str, Any]] = []
    for path in sorted(candidates, key=lambda item: str(item)):
        artifact = load_reference_crop_supervision(
            path,
            verify_input_files=False,
        )
        shot = artifact.payload.get("shot")
        if not isinstance(shot, Mapping):
            raise ValueError(f"crop artifact has no shot object: {path}")
        shot_index = shot.get("shot_index")
        if isinstance(shot_index, bool) or not isinstance(shot_index, int):
            raise ValueError(f"crop artifact shot index is invalid: {path}")
        if shot_index in by_shot:
            raise ValueError(
                f"duplicate crop artifact for {reference_id} shot {shot_index}"
            )
        if shot_index not in expected_shot_indices:
            raise ValueError(
                f"crop artifact targets a non-retained shot: {reference_id} {shot_index}"
            )
        by_shot[shot_index] = (path, artifact.eligible)
        manifest.append(
            {
                "path": str(path),
                "file_sha256": _file_sha256(path),
                "shot_index": shot_index,
                "eligible": artifact.eligible,
                "artifact_content_sha256": artifact.payload["provenance"][
                    "artifact_content_sha256"
                ],
            }
        )
    observed = set(by_shot)
    if observed != expected_shot_indices:
        missing = sorted(expected_shot_indices - observed)
        extra = sorted(observed - expected_shot_indices)
        raise ValueError(
            f"crop coverage mismatch for {reference_id}: missing={missing}, extra={extra}"
        )
    all_paths = tuple(by_shot[index][0] for index in sorted(by_shot))
    eligible_paths = tuple(
        by_shot[index][0] for index in sorted(by_shot) if by_shot[index][1]
    )
    return all_paths, eligible_paths, manifest


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if _load_mapping(path, "immutable output") != dict(payload):
            raise FileExistsError(f"immutable output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale materialization temporary file: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"immutable output appeared during write: {path}")
        os.rename(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resume_result(
    manifest_path: Path,
    *,
    expected_fingerprint: str,
) -> ReferenceTrainingMaterializationResult:
    manifest = _load_mapping(manifest_path, "materialization manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("materialization manifest schema mismatch")
    if manifest.get("analysis_basis") != ANALYSIS_BASIS:
        raise ValueError("materialization manifest analysis basis mismatch")
    if manifest.get("input_fingerprint_sha256") != expected_fingerprint:
        raise ValueError("materialization manifest fingerprint mismatch")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("materialization manifest outputs are missing")
    datasets: dict[str, Path] = {}
    supervision: dict[str, Path] = {}
    for item in outputs:
        if not isinstance(item, Mapping):
            raise ValueError("materialization output entry must be an object")
        reference_id = _safe_reference_id(item.get("reference_id"))
        dataset_path = Path(str(item.get("dataset_path") or ""))
        supervision_path = Path(str(item.get("supervision_path") or ""))
        for path, digest_key in (
            (dataset_path, "dataset_file_sha256"),
            (supervision_path, "supervision_file_sha256"),
        ):
            if not path.is_absolute() or not path.is_file():
                raise ValueError(f"materialized output is missing: {path}")
            if _file_sha256(path) != item.get(digest_key):
                raise ValueError(f"materialized output hash mismatch: {path}")
        datasets[reference_id] = dataset_path
        supervision[reference_id] = supervision_path
    return ReferenceTrainingMaterializationResult(
        manifest_path=manifest_path,
        dataset_paths=datasets,
        supervision_paths=supervision,
        input_fingerprint_sha256=expected_fingerprint,
    )


def materialize_reference_training_artifacts(
    attribution_paths_by_reference: Mapping[str, Path],
    *,
    crop_directories: Sequence[Path],
    style_id: str,
    output_directory: Path,
    expected_reference_count: int = 5,
    expected_source_count: int = 10,
) -> ReferenceTrainingMaterializationResult:
    """Rebuild current schema JSON without touching immutable analysis inputs."""

    if expected_reference_count <= 0 or expected_source_count <= 1:
        raise ValueError("expected reference/source counts are invalid")
    normalized_style = style_id.strip()
    if not normalized_style:
        raise ValueError("style_id cannot be blank")
    if len(attribution_paths_by_reference) != expected_reference_count:
        raise ValueError(
            "exact reference count mismatch: "
            f"{len(attribution_paths_by_reference)} != {expected_reference_count}"
        )
    crop_roots = tuple(Path(item).expanduser().resolve() for item in crop_directories)
    if not crop_roots or any(not item.is_dir() for item in crop_roots):
        raise ValueError("crop directories must be existing directories")
    output_root = Path(output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    inputs: list[dict[str, Any]] = []
    loaded: dict[str, tuple[Path, dict[str, Any], tuple[Path, ...]]] = {}
    for raw_reference_id, raw_path in sorted(attribution_paths_by_reference.items()):
        reference_id = _safe_reference_id(raw_reference_id)
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file() or path.name.startswith("._"):
            raise ValueError(f"final attribution is unavailable: {path}")
        attribution = _load_mapping(path, "final attribution")
        observed_sources = _source_count(attribution)
        if observed_sources != expected_source_count:
            raise ValueError(
                f"source count mismatch for {reference_id}: "
                f"{observed_sources} != {expected_source_count}"
            )
        known = _known_shot_indices(attribution)
        _all_crop, eligible_crop, crop_manifest = _discover_crop_artifacts(
            reference_id,
            crop_roots,
            expected_shot_indices=known,
        )
        attribution_sha256 = _file_sha256(path)
        inputs.append(
            {
                "reference_id": reference_id,
                "attribution_path": str(path),
                "attribution_file_sha256": attribution_sha256,
                "known_shot_count": len(known),
                "source_count": observed_sources,
                "crop_artifact_count": len(crop_manifest),
                "eligible_crop_artifact_count": len(eligible_crop),
                "crop_artifacts": crop_manifest,
            }
        )
        loaded[reference_id] = (path, attribution, eligible_crop)

    fingerprint_payload = {
        "analysis_basis": ANALYSIS_BASIS,
        "style_id": normalized_style,
        "reference_dataset_schema_version": REFERENCE_DATASET_SCHEMA_VERSION,
        "reference_supervision_schema_version": (
            REFERENCE_SUPERVISION_SCHEMA_VERSION
        ),
        "expected_reference_count": expected_reference_count,
        "expected_source_count": expected_source_count,
        "inputs": inputs,
    }
    fingerprint = _sha256_bytes(_canonical_bytes(fingerprint_payload))
    manifest_path = output_root / (
        f"reference-training-materialization-{fingerprint[:16]}.json"
    )
    lock_path = output_root / ".reference-training-materialization.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if manifest_path.exists():
            return _resume_result(
                manifest_path,
                expected_fingerprint=fingerprint,
            )

        dataset_paths: dict[str, Path] = {}
        supervision_paths: dict[str, Path] = {}
        outputs: list[dict[str, Any]] = []
        for reference_id, (attribution_path, attribution, eligible_crop) in sorted(
            loaded.items()
        ):
            dataset = build_reference_dataset(
                attribution,
                style_id=normalized_style,
                reference_id=reference_id,
                crop_supervision_artifacts=eligible_crop,
            ).payload
            supervision = build_reference_supervision(
                attribution,
                style_id=normalized_style,
                reference_id=reference_id,
                crop_supervision_artifacts=eligible_crop,
            ).payload
            if dataset.get("schema_version") != REFERENCE_DATASET_SCHEMA_VERSION:
                raise ValueError("rebuilt reference dataset schema mismatch")
            if (
                supervision.get("schema_version")
                != REFERENCE_SUPERVISION_SCHEMA_VERSION
            ):
                raise ValueError("rebuilt reference supervision schema mismatch")
            dataset_path = output_root / (
                f"{reference_id}-reference-dataset-stageedit-"
                f"schema{REFERENCE_DATASET_SCHEMA_VERSION}-{fingerprint[:16]}.json"
            )
            supervision_path = output_root / (
                f"{reference_id}-reference-supervision-stageedit-"
                f"schema{REFERENCE_SUPERVISION_SCHEMA_VERSION}-{fingerprint[:16]}.json"
            )
            _write_immutable_json(dataset_path, dataset)
            _write_immutable_json(supervision_path, supervision)
            dataset_paths[reference_id] = dataset_path
            supervision_paths[reference_id] = supervision_path
            outputs.append(
                {
                    "reference_id": reference_id,
                    "attribution_path": str(attribution_path),
                    "dataset_path": str(dataset_path),
                    "dataset_file_sha256": _file_sha256(dataset_path),
                    "supervision_path": str(supervision_path),
                    "supervision_file_sha256": _file_sha256(supervision_path),
                    "known_shot_label_count": supervision["statistics"][
                        "known_shot_label_count"
                    ],
                    "dense_editorial_source_switch_supervision_count": (
                        supervision["statistics"][
                            "dense_editorial_source_switch_supervision_count"
                        ]
                    ),
                    "crop_supervision_count": supervision["statistics"][
                        "crop_supervision_count"
                    ],
                }
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "analysis_basis": ANALYSIS_BASIS,
            "created_at": datetime.now(UTC).isoformat(),
            "input_fingerprint_sha256": fingerprint,
            "input_contract": fingerprint_payload,
            "outputs": outputs,
            "statistics": {
                "reference_count": len(outputs),
                "known_shot_label_count": sum(
                    int(item["known_shot_label_count"]) for item in outputs
                ),
                "dense_editorial_source_switch_supervision_count": sum(
                    int(item["dense_editorial_source_switch_supervision_count"])
                    for item in outputs
                ),
                "crop_supervision_count": sum(
                    int(item["crop_supervision_count"]) for item in outputs
                ),
            },
            "immutable_inputs_modified": False,
        }
        _write_immutable_json(manifest_path, manifest)
        return _resume_result(
            manifest_path,
            expected_fingerprint=fingerprint,
        )


def _keyed_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        reference_id, separator, raw_path = value.partition("=")
        reference_id = _safe_reference_id(reference_id)
        if not separator or not raw_path.strip():
            raise ValueError("--attribution must use REFERENCE_ID=/absolute/path.json")
        if reference_id in result:
            raise ValueError(f"duplicate attribution id: {reference_id}")
        result[reference_id] = Path(raw_path.strip())
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild immutable current-schema StageEdit reference JSON"
    )
    parser.add_argument(
        "--attribution",
        action="append",
        required=True,
        metavar="REFERENCE_ID=PATH",
    )
    parser.add_argument(
        "--crop-directory", action="append", required=True, type=Path
    )
    parser.add_argument("--style-id", required=True)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--expected-reference-count", type=int, default=5)
    parser.add_argument("--expected-source-count", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize_reference_training_artifacts(
        _keyed_paths(args.attribution),
        crop_directories=args.crop_directory,
        style_id=args.style_id,
        output_directory=args.output_directory,
        expected_reference_count=args.expected_reference_count,
        expected_source_count=args.expected_source_count,
    )
    print(
        json.dumps(
            {
                "manifest_path": str(result.manifest_path),
                "input_fingerprint_sha256": result.input_fingerprint_sha256,
                "datasets": {
                    key: str(value) for key, value in result.dataset_paths.items()
                },
                "supervision": {
                    key: str(value)
                    for key, value in result.supervision_paths.items()
                },
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_BASIS",
    "SCHEMA_VERSION",
    "ReferenceTrainingMaterializationResult",
    "materialize_reference_training_artifacts",
]
