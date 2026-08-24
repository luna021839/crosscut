"""Immutable exact-source bootstrap for professional-reference attribution.

This module turns one strict acquisition manifest plus one professional cadence
profile into the reusable inputs consumed by ``reference_expansion_job``:

* a provenance-bound training source manifest,
* a fully synchronized source catalog,
* canonical seed evidence and its deterministic attribution/dataset stages.

Large media are read only in a real run.  ``--dry-run`` validates identities,
counts, paths, and immutable output collisions without hashing or decoding the
media files.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .professional_reference_cadence import ProfessionalReferenceCadenceProfile
from .reference_attribution import resolve_reference_attribution_payload
from .reference_boundary_supervision import augment_reference_boundary_requests
from .reference_dataset import build_reference_dataset
from .reference_evidence_builder import (
    ANALYSIS_BASIS as REFERENCE_EVIDENCE_ANALYSIS_BASIS,
    SCHEMA_VERSION as REFERENCE_EVIDENCE_SCHEMA_VERSION,
    ReferenceSourceClock,
    build_reference_evidence,
)
from .reference_expansion_job import (
    _publish_json_no_replace as _expansion_publish_json_no_replace,
    _recover_reserved_artifact as _expansion_recover_reserved_artifact,
    _validate_boundary_evidence as _expansion_validate_boundary_evidence,
    _validate_dataset_payload as _expansion_validate_dataset_payload,
)
from .reference_supervision import build_reference_supervision
from .reference_training_source_pool import (
    MINIMUM_SOURCE_COUNT,
    build_reference_training_source_manifest,
    build_synchronized_training_source_catalog_from_manifest,
    validate_reference_training_source_manifest,
)


SCHEMA_VERSION = 1
ANALYSIS_BASIS = "strict_professional_reference_training_bootstrap_v1"
CATALOG_ANALYSIS_BASIS = "strict_synchronized_reference_training_source_catalog_v1"
SEED_PURPOSE = "same_song_professional_reference_learning_10source_training_pool"
DEFAULT_EXPECTED_REFERENCE_COUNT = 5
DEFAULT_EXPECTED_SOURCE_COUNT = 10


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, name: str) -> tuple[dict[str, Any], bytes]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"{name} is not a regular file: {source}")
    raw = source.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} root must be an object")
    return payload, raw


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _artifact_slug(artist: str, song: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", f"{artist}-{song}").strip("-")
    if not slug:
        raise ValueError("artist/song cannot produce an empty artifact slug")
    return slug.upper()


def _publish_json_no_replace(
    destination: Path,
    payload: Mapping[str, Any],
) -> tuple[str, str]:
    """Publish canonical JSON atomically, or validate an identical artifact."""

    path = Path(destination).expanduser().resolve()
    expected = _pretty_json_bytes(payload)
    if path.exists():
        recovered = _expansion_recover_reserved_artifact(
            path,
            validator=lambda observed: (
                None
                if _canonical_json_bytes(observed) == _canonical_json_bytes(payload)
                else (_ for _ in ()).throw(
                    ValueError(f"reserved immutable artifact differs: {path}")
                )
            ),
        )
        if recovered is not None:
            _observed, result = recovered
            return str(result["action"]), str(result["sha256"])
        if not path.is_file() or path.is_symlink():
            raise FileExistsError(f"immutable destination is not a regular file: {path}")
        existing, raw = _load_json(path, "existing immutable artifact")
        if _canonical_json_bytes(existing) != _canonical_json_bytes(payload):
            raise FileExistsError(f"immutable artifact differs from expected: {path}")
        return "validated_existing", _sha256_bytes(raw)

    try:
        _expansion_publish_json_no_replace(path, payload)
        return "created", _sha256_bytes(expected)
    except FileExistsError:
        existing, raw = _load_json(path, "raced immutable artifact")
        if _canonical_json_bytes(existing) != _canonical_json_bytes(payload):
            raise FileExistsError(
                f"raced immutable artifact differs from expected: {path}"
            )
        return "validated_existing", _sha256_bytes(raw)


def _catalog_content_sha256(payload: Mapping[str, Any]) -> str:
    normalized = deepcopy(dict(payload))
    provenance = _mapping(normalized.get("provenance"), "catalog provenance")
    mutable = deepcopy(dict(provenance))
    mutable["content_sha256"] = None
    normalized["provenance"] = mutable
    return _sha256_bytes(_canonical_json_bytes(normalized))


def _catalog_payload(
    sources: Sequence[Mapping[str, Any]],
    *,
    artist: str,
    song: str,
    manifest_path: Path,
    manifest_file_sha256: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis": CATALOG_ANALYSIS_BASIS,
        "artist": artist,
        "song": song,
        "source_count": len(sources),
        "source_ids": [str(item["video_id"]) for item in sources],
        "sources": [deepcopy(dict(item)) for item in sources],
        "provenance": {
            "training_manifest_path": str(manifest_path),
            "training_manifest_file_sha256": manifest_file_sha256,
            "content_sha256": None,
        },
    }
    payload["provenance"]["content_sha256"] = _catalog_content_sha256(payload)
    return payload


def _validate_catalog_payload(
    payload: Mapping[str, Any],
    *,
    training_manifest: Mapping[str, Any],
    training_manifest_path: Path,
    training_manifest_file_sha256: str,
    artist: str,
    song: str,
) -> tuple[Mapping[str, Any], ...]:
    source_ids = validate_reference_training_source_manifest(
        training_manifest,
        expected_artist=artist,
        expected_song=song,
        verify_acquisition_binding=True,
    )
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("analysis") != CATALOG_ANALYSIS_BASIS
        or payload.get("artist") != artist
        or payload.get("song") != song
        or payload.get("source_count") != len(source_ids)
        or tuple(str(value) for value in _array(payload.get("source_ids"), "source_ids"))
        != source_ids
    ):
        raise ValueError("synchronized source catalog identity/count mismatch")
    provenance = _mapping(payload.get("provenance"), "catalog provenance")
    if (
        Path(str(provenance.get("training_manifest_path") or ""))
        .expanduser()
        .resolve()
        != training_manifest_path
        or provenance.get("training_manifest_file_sha256")
        != training_manifest_file_sha256
        or provenance.get("content_sha256") != _catalog_content_sha256(payload)
    ):
        raise ValueError("synchronized source catalog provenance/hash mismatch")
    selected = _array(training_manifest.get("selected"), "manifest selected")
    sources = _array(payload.get("sources"), "catalog sources")
    if len(sources) != len(selected):
        raise ValueError("synchronized source catalog is incomplete")
    normalized: list[Mapping[str, Any]] = []
    for index, (raw_source, raw_selected) in enumerate(
        zip(sources, selected, strict=True)
    ):
        source = _mapping(raw_source, f"catalog source {index}")
        selected_source = _mapping(raw_selected, f"manifest source {index}")
        video_id = str(selected_source.get("video_id") or "")
        if source.get("video_id") != video_id:
            raise ValueError("catalog and manifest source order differ")
        expected_path = Path(str(selected_source.get("local_path") or "")).resolve()
        if Path(str(source.get("path") or "")).resolve() != expected_path:
            raise ValueError(f"catalog source path mismatch: {video_id}")
        raw_clock = _mapping(source.get("clock"), f"catalog clock {video_id}")
        if _canonical_json_bytes(ReferenceSourceClock.from_mapping(raw_clock).to_dict()) != (
            _canonical_json_bytes(raw_clock)
        ):
            raise ValueError(f"catalog source clock is not canonical: {video_id}")
        _mapping(source.get("provenance"), f"catalog source provenance {video_id}")
        probe = _mapping(source.get("media_probe"), f"catalog media probe {video_id}")
        expected_size = _mapping(
            selected_source.get("content_integrity"), "content integrity"
        ).get("size_bytes")
        if probe.get("size_bytes") != expected_size:
            raise ValueError(f"catalog media size mismatch: {video_id}")
        normalized.append(source)
    return tuple(normalized)


def _reference_specs(
    profile_payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    provenance = _mapping(profile_payload.get("provenance"), "profile provenance")
    return tuple(
        _mapping(value, f"profile reference {index}")
        for index, value in enumerate(
            _array(provenance.get("references"), "profile references")
        )
    )


def _validate_seed_evidence(
    payload: Mapping[str, Any],
    *,
    reference: Mapping[str, Any],
    reference_clock: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    caller: Mapping[str, Any],
) -> None:
    if (
        payload.get("schema_version") != REFERENCE_EVIDENCE_SCHEMA_VERSION
        or payload.get("analysis") != REFERENCE_EVIDENCE_ANALYSIS_BASIS
    ):
        raise ValueError("seed evidence schema/analysis mismatch")
    reference_path = Path(str(reference.get("media_path") or "")).resolve()
    if Path(str(payload.get("reference") or "")).resolve() != reference_path:
        raise ValueError("seed evidence reference path mismatch")
    provenance = _mapping(payload.get("provenance"), "seed provenance")
    if _canonical_json_bytes(provenance.get("caller")) != _canonical_json_bytes(caller):
        raise ValueError("seed evidence caller provenance mismatch")
    observed_sources = _array(payload.get("sources"), "seed sources")
    if _canonical_json_bytes(observed_sources) != _canonical_json_bytes(list(sources)):
        raise ValueError("seed evidence synchronized source catalog mismatch")
    clock_policy = _mapping(payload.get("clock_policy"), "seed clock policy")
    if _canonical_json_bytes(clock_policy.get("reference_mapping")) != (
        _canonical_json_bytes(reference_clock)
    ):
        raise ValueError("seed evidence reference clock mismatch")
    cuts = _array(payload.get("detected_reference_cuts"), "seed detected cuts")
    excluded_cuts = _array(
        payload.get("excluded_detected_reference_cuts", []),
        "seed excluded detected cuts",
    )
    if len(cuts) + len(excluded_cuts) != int(
        reference.get("audit_cut_count") or 0
    ):
        raise ValueError("seed evidence detected-cut count differs from cadence audit")
    statistics = _mapping(payload.get("statistics"), "seed statistics")
    if statistics.get("hard_cut_count") != len(cuts):
        raise ValueError("seed evidence hard-cut statistic mismatch")
    if excluded_cuts and (
        statistics.get("detector_hard_cut_count") != len(cuts) + len(excluded_cuts)
        or statistics.get("excluded_detected_hard_cut_count")
        != len(excluded_cuts)
    ):
        raise ValueError("seed evidence excluded hard-cut accounting mismatch")
    if _array(payload.get("boundary_requests", []), "seed boundary requests"):
        raise ValueError("seed evidence must remain unaugmented")


def _planned_paths(
    output_directory: Path,
    *,
    artifact_slug: str,
    source_count: int,
    seed_reference_id: str,
) -> dict[str, Path]:
    root = Path(output_directory).expanduser().resolve()
    return {
        "manifest": root
        / f"{artifact_slug}-reference-training-source-manifest-{source_count}.json",
        "catalog": root
        / f"{artifact_slug}-reference-training-source-catalog-{source_count}.json",
        "evidence": root
        / f"{seed_reference_id}-reference-evidence-canonical-v2.json",
        "initial_attribution": root
        / f"{seed_reference_id}-reference-attribution-canonical-v2-initial.json",
        "boundary_evidence": root
        / f"{seed_reference_id}-reference-evidence-canonical-v2-boundaries.json",
        "final_attribution": root
        / f"{seed_reference_id}-reference-attribution-canonical-v2-final.json",
        "dataset": root
        / f"{seed_reference_id}-reference-dataset-canonical-v2-final.json",
        "supervision": root
        / f"{seed_reference_id}-reference-supervision-canonical-v2-final.json",
    }


def run_reference_training_bootstrap(
    *,
    cadence_profile_path: Path,
    acquisition_manifest_path: Path,
    output_directory: Path,
    style_id: str,
    expected_artist: str,
    expected_song: str,
    expected_reference_count: int = DEFAULT_EXPECTED_REFERENCE_COUNT,
    expected_source_count: int = DEFAULT_EXPECTED_SOURCE_COUNT,
    additional_source_ids: Sequence[str] = (),
    seed_reference_id: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate or materialize one strict seed corpus without overwrites."""

    if expected_reference_count <= 0:
        raise ValueError("expected_reference_count must be positive")
    if expected_source_count < MINIMUM_SOURCE_COUNT:
        raise ValueError(f"expected_source_count must be at least {MINIMUM_SOURCE_COUNT}")
    normalized_style_id = str(style_id).strip()
    if not normalized_style_id:
        raise ValueError("style_id cannot be blank")
    profile_path = Path(cadence_profile_path).expanduser().resolve()
    acquisition_path = Path(acquisition_manifest_path).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    profile = ProfessionalReferenceCadenceProfile.from_file(
        profile_path,
        verify_input_hashes=False,
    )
    if (
        profile.artist != expected_artist
        or profile.song != expected_song
        or profile.reference_count != expected_reference_count
    ):
        raise ValueError("cadence profile identity/reference count mismatch")
    profile_payload = profile.to_dict()
    references = _reference_specs(profile_payload)
    reference_by_id = {str(item.get("video_id") or ""): item for item in references}
    chosen_seed_id = str(seed_reference_id or references[0].get("video_id") or "")
    if chosen_seed_id not in reference_by_id:
        raise ValueError("seed_reference_id is not one cadence reference")
    chosen_reference = reference_by_id[chosen_seed_id]
    paths = _planned_paths(
        output,
        artifact_slug=_artifact_slug(expected_artist, expected_song),
        source_count=expected_source_count,
        seed_reference_id=chosen_seed_id,
    )
    training_manifest = build_reference_training_source_manifest(
        acquisition_path,
        additional_source_ids=additional_source_ids,
        expected_artist=expected_artist,
        expected_song=expected_song,
        expected_source_count=expected_source_count,
    )
    source_ids = validate_reference_training_source_manifest(
        training_manifest,
        expected_artist=expected_artist,
        expected_song=expected_song,
        verify_acquisition_binding=True,
    )
    if Path(str(training_manifest["selected"][0]["local_path"])).resolve() != (
        profile.master_media_path
    ):
        raise ValueError("cadence master is not the first training source")
    if training_manifest["selected"][0]["content_integrity"]["sha256"] != (
        profile.master_media_sha256
    ):
        raise ValueError("cadence master hash differs from training source manifest")

    plan = {
        name: {
            "path": str(path),
            "action": "validate_existing" if path.exists() else "create",
        }
        for name, path in paths.items()
    }
    base_result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_BASIS,
        "dry_run": dry_run,
        "artist": expected_artist,
        "song": expected_song,
        "style_id": normalized_style_id,
        "reference_count": expected_reference_count,
        "source_count": expected_source_count,
        "source_ids": list(source_ids),
        "seed_reference_id": chosen_seed_id,
        "cadence_profile": {
            "path": str(profile_path),
            "file_sha256": profile.file_sha256,
            "content_sha256": profile.profile_content_sha256,
        },
        "acquisition_manifest": {"path": str(acquisition_path)},
        "artifacts": plan,
        "writes_performed": False,
    }
    if dry_run:
        base_result["status"] = "ready"
        return base_result

    manifest_action, manifest_file_sha256 = _publish_json_no_replace(
        paths["manifest"], training_manifest
    )
    training_manifest, manifest_raw = _load_json(
        paths["manifest"], "published training manifest"
    )
    manifest_file_sha256 = _sha256_bytes(manifest_raw)
    base_result["artifacts"]["manifest"]["action"] = manifest_action
    base_result["artifacts"]["manifest"]["file_sha256"] = manifest_file_sha256

    if paths["catalog"].exists():
        catalog, catalog_raw = _load_json(paths["catalog"], "source catalog")
        sources = _validate_catalog_payload(
            catalog,
            training_manifest=training_manifest,
            training_manifest_path=paths["manifest"],
            training_manifest_file_sha256=manifest_file_sha256,
            artist=expected_artist,
            song=expected_song,
        )
        catalog_action = "validated_existing"
        catalog_file_sha256 = _sha256_bytes(catalog_raw)
    else:
        sources = build_synchronized_training_source_catalog_from_manifest(
            training_manifest,
            expected_artist=expected_artist,
            expected_song=expected_song,
        )
        catalog = _catalog_payload(
            sources,
            artist=expected_artist,
            song=expected_song,
            manifest_path=paths["manifest"],
            manifest_file_sha256=manifest_file_sha256,
        )
        catalog_action, catalog_file_sha256 = _publish_json_no_replace(
            paths["catalog"], catalog
        )
    base_result["artifacts"]["catalog"]["action"] = catalog_action
    base_result["artifacts"]["catalog"]["file_sha256"] = catalog_file_sha256

    reference_path = Path(str(chosen_reference.get("media_path") or "")).resolve()
    audit_path = Path(str(chosen_reference.get("audit_path") or "")).resolve()
    if _sha256_file(reference_path) != chosen_reference.get("media_sha256"):
        raise ValueError("seed reference media SHA-256 mismatch")
    if _sha256_file(audit_path) != chosen_reference.get("audit_sha256"):
        raise ValueError("seed reference audit SHA-256 mismatch")
    raw_reference_clock = _mapping(
        chosen_reference.get("local_timewarp"), "seed reference local_timewarp"
    )
    if raw_reference_clock.get("scalar_fallback") is not False:
        raise ValueError("seed reference used a forbidden scalar-fallback clock")
    reference_clock = ReferenceSourceClock.from_mapping(raw_reference_clock).to_dict()
    master = _mapping(
        _mapping(profile_payload.get("provenance"), "profile provenance").get("master"),
        "profile master",
    )
    master_end_seconds = float(master.get("decoded_audio_duration_seconds"))
    caller = {
        "artist": expected_artist,
        "song": expected_song,
        "reference_video_id": chosen_seed_id,
        "purpose": SEED_PURPOSE,
        "training_source_manifest": {
            "path": str(paths["manifest"]),
            "file_sha256": manifest_file_sha256,
        },
        "synchronized_source_catalog": {
            "path": str(paths["catalog"]),
            "file_sha256": catalog_file_sha256,
        },
        "production_render_admission_modified": False,
    }

    if paths["evidence"].exists():
        evidence, evidence_raw = _load_json(paths["evidence"], "seed evidence")
        _validate_seed_evidence(
            evidence,
            reference=chosen_reference,
            reference_clock=reference_clock,
            sources=sources,
            caller=caller,
        )
        evidence_action = "validated_existing"
        evidence_file_sha256 = _sha256_bytes(evidence_raw)
    else:
        evidence = build_reference_evidence(
            reference_path,
            sources,
            master_start_seconds=0.0,
            master_end_seconds=master_end_seconds,
            reference_clock=reference_clock,
            provenance=caller,
        ).payload
        _validate_seed_evidence(
            evidence,
            reference=chosen_reference,
            reference_clock=reference_clock,
            sources=sources,
            caller=caller,
        )
        evidence_action, evidence_file_sha256 = _publish_json_no_replace(
            paths["evidence"], evidence
        )
    base_result["artifacts"]["evidence"]["action"] = evidence_action
    base_result["artifacts"]["evidence"]["file_sha256"] = evidence_file_sha256

    initial = resolve_reference_attribution_payload(evidence)
    initial_action, initial_file_sha256 = _publish_json_no_replace(
        paths["initial_attribution"], initial
    )
    base_result["artifacts"]["initial_attribution"]["action"] = initial_action
    base_result["artifacts"]["initial_attribution"][
        "file_sha256"
    ] = initial_file_sha256

    expected_boundary_evidence = augment_reference_boundary_requests(evidence).payload
    if paths["boundary_evidence"].exists():
        boundary_evidence, boundary_raw = _load_json(
            paths["boundary_evidence"], "seed boundary evidence"
        )
        _expansion_validate_boundary_evidence(
            boundary_evidence,
            base_evidence=evidence,
            expected=expected_boundary_evidence,
        )
        boundary_action = "validated_existing"
        boundary_file_sha256 = _sha256_bytes(boundary_raw)
    else:
        boundary_evidence = expected_boundary_evidence
        boundary_action, boundary_file_sha256 = _publish_json_no_replace(
            paths["boundary_evidence"], boundary_evidence
        )
    base_result["artifacts"]["boundary_evidence"]["action"] = boundary_action
    base_result["artifacts"]["boundary_evidence"][
        "file_sha256"
    ] = boundary_file_sha256

    final = resolve_reference_attribution_payload(boundary_evidence)
    dataset = build_reference_dataset(
        final,
        style_id=normalized_style_id,
        reference_id=chosen_seed_id,
    ).payload
    supervision = build_reference_supervision(
        final,
        style_id=normalized_style_id,
        reference_id=chosen_seed_id,
    ).payload
    derived = {
        "final_attribution": final,
        "supervision": supervision,
    }
    for name, payload in derived.items():
        action, file_sha256 = _publish_json_no_replace(paths[name], payload)
        base_result["artifacts"][name]["action"] = action
        base_result["artifacts"][name]["file_sha256"] = file_sha256

    if paths["dataset"].exists():
        observed_dataset, dataset_raw = _load_json(
            paths["dataset"], "seed reference dataset"
        )
        _expansion_validate_dataset_payload(observed_dataset, dataset)
        dataset_action = "validated_existing"
        dataset_file_sha256 = _sha256_bytes(dataset_raw)
    else:
        dataset_action, dataset_file_sha256 = _publish_json_no_replace(
            paths["dataset"], dataset
        )
    base_result["artifacts"]["dataset"]["action"] = dataset_action
    base_result["artifacts"]["dataset"]["file_sha256"] = dataset_file_sha256

    base_result["writes_performed"] = any(
        item["action"] == "created" for item in base_result["artifacts"].values()
    )
    base_result["status"] = "complete"
    base_result["completed_at"] = datetime.now(UTC).isoformat()
    base_result["next_commands"] = {
        "expansion": [
            "python3",
            "-m",
            "crosscut.reference_expansion_job",
            "--cadence-profile",
            str(profile_path),
            "--seed-evidence",
            str(paths["evidence"]),
            "--source-manifest",
            str(paths["manifest"]),
            "--output-directory",
            str(output),
            "--style-id",
            normalized_style_id,
            "--expected-artist",
            expected_artist,
            "--expected-song",
            expected_song,
            "--expected-reference-count",
            str(expected_reference_count),
            "--expected-source-count",
            str(expected_source_count),
        ],
        "crop": [
            "python3",
            "-m",
            "crosscut.reference_crop_job",
            "--cadence-profile",
            str(profile_path),
            "--seed-evidence",
            str(paths["evidence"]),
            "--source-manifest",
            str(paths["manifest"]),
            "--output-directory",
            str(output),
            "--style-id",
            normalized_style_id,
            "--expected-artist",
            expected_artist,
            "--expected-song",
            expected_song,
            "--expected-reference-count",
            str(expected_reference_count),
            "--expected-source-count",
            str(expected_source_count),
        ],
    }
    return base_result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build immutable exact-source professional-reference seed inputs."
    )
    parser.add_argument("--cadence-profile", required=True, type=Path)
    parser.add_argument("--acquisition-manifest", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--style-id", required=True)
    parser.add_argument("--expected-artist", required=True)
    parser.add_argument("--expected-song", required=True)
    parser.add_argument(
        "--expected-reference-count",
        type=int,
        default=DEFAULT_EXPECTED_REFERENCE_COUNT,
    )
    parser.add_argument(
        "--expected-source-count",
        type=int,
        default=DEFAULT_EXPECTED_SOURCE_COUNT,
    )
    parser.add_argument("--additional-source-id", action="append", default=[])
    parser.add_argument("--seed-reference-id")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_reference_training_bootstrap(
        cadence_profile_path=args.cadence_profile,
        acquisition_manifest_path=args.acquisition_manifest,
        output_directory=args.output_directory,
        style_id=args.style_id,
        expected_artist=args.expected_artist,
        expected_song=args.expected_song,
        expected_reference_count=args.expected_reference_count,
        expected_source_count=args.expected_source_count,
        additional_source_ids=tuple(args.additional_source_id),
        seed_reference_id=args.seed_reference_id,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_BASIS",
    "CATALOG_ANALYSIS_BASIS",
    "DEFAULT_EXPECTED_REFERENCE_COUNT",
    "DEFAULT_EXPECTED_SOURCE_COUNT",
    "SEED_PURPOSE",
    "run_reference_training_bootstrap",
]
