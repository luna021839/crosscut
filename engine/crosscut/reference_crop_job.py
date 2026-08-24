"""Resumable crop-supervision expansion for canonical human references.

This job is deliberately downstream of :mod:`reference_expansion_job`.  It
accepts only the immutable ``canonical-v2-final`` attributions produced for the
five cadence references, materializes one self-hashed multi-frame crop audit
for every conservatively accepted shot, and then rebuilds dataset and
supervision outputs using only crop audits whose geometry gates accepted a
renderer-native target.

``--dry-run`` never hashes or decodes large media and never writes.  A real run
is blocked before media hashing whenever ``crosscut_cli.py analyze`` is active.
Published crop, dataset, and supervision JSON files are immutable; existing
files are skipped only after schema, self-hash, exact shot, clock, media,
attribution, configuration, and caller-provenance validation.  Mutable status
manifests are atomic but are never treated as completion evidence.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Mapping, Sequence

from .reference_crop_supervision import (
    ANALYSIS_BASIS as CROP_SUPERVISION_ANALYSIS_BASIS,
    SCHEMA_VERSION as CROP_SUPERVISION_SCHEMA_VERSION,
    ReferenceCropSupervisionConfig,
    ReferenceCropSupervisionRequest,
    build_reference_crop_supervision,
    validate_reference_crop_supervision_payload,
)
from .reference_attribution import resolve_reference_attribution_payload
from .reference_boundary_supervision import augment_reference_boundary_requests
from .reference_dataset import build_reference_dataset
from .reference_evidence_builder import ReferenceSourceClock
from .reference_expansion_job import (
    DEFAULT_EXPECTED_REFERENCE_COUNT,
    DEFAULT_EXPECTED_SOURCE_COUNT,
    MINIMUM_EXPECTED_SOURCE_COUNT,
    ActiveAnalysisError,
    ExistingArtifactValidationError,
    ExpansionInputs,
    MediaIdentity,
    ReferenceExpansionError,
    ReferenceSpec,
    SourceSpec,
    _active_analysis_processes,
    _array,
    _artifact_paths,
    _assert_canonical_json_equal,
    _assert_stat_unchanged,
    _atomic_update_json,
    _caller_provenance as _expansion_caller_provenance,
    _canonical_json_bytes,
    _canonical_json_sha256,
    _cleanup_stale_temps,
    _finite,
    _job_lock,
    _load_existing_artifact,
    _load_inputs,
    _mapping,
    _materialize_or_skip,
    _module_sha256,
    _sha256_bytes,
    _sha256_string,
    _stat_identity,
    _stable_read_json,
    _utc_now,
    _validate_dataset_payload,
    _validate_boundary_evidence,
    _validate_generated_evidence,
)
from .reference_supervision import build_reference_supervision


SCHEMA_VERSION = 1
ANALYSIS_BASIS = "resumable_canonical_reference_multiframe_crop_v1"
PIPELINE_REVISION = "canonical-v2-final-to-crop-v1"
DEFAULT_MINIMUM_FREE_BYTES = 512 * 1024 * 1024

_STAGE_NAMES = ("crop_artifacts", "dataset", "supervision")


class ReferenceCropJobError(ReferenceExpansionError):
    """Raised when crop-v1 orchestration cannot proceed safely."""


class MissingFinalAttributionError(ReferenceCropJobError):
    """Raised when expansion has not published every requested attribution."""


@dataclass(frozen=True, slots=True)
class CropOutputPaths:
    output_directory: Path
    reference_id: str
    dataset: Path
    supervision: Path
    status_manifest: Path

    @property
    def crop_glob(self) -> str:
        return (
            f"{self.reference_id}-reference-crop-supervision-crop-v1-shot-*.json"
        )

    def crop_artifact(self, shot_index: int, source_id: str) -> Path:
        if shot_index < 0:
            raise ValueError("shot_index must be non-negative")
        if not source_id or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
            for character in source_id
        ):
            raise ValueError(f"unsafe crop source id: {source_id!r}")
        return self.output_directory / (
            f"{self.reference_id}-reference-crop-supervision-crop-v1-"
            f"shot-{shot_index:04d}-{source_id}.json"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "crop_artifact_glob": str(self.output_directory / self.crop_glob),
            "dataset": str(self.dataset),
            "supervision": str(self.supervision),
            "status_manifest": str(self.status_manifest),
        }


@dataclass(frozen=True, slots=True)
class ValidatedAttribution:
    path: Path
    payload: Mapping[str, Any]
    file_sha256: str
    canonical_json_sha256: str
    accepted_shots: tuple[Mapping[str, Any], ...]
    stat_identity: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class CropJobInputs:
    expansion: ExpansionInputs
    target_reference_ids: tuple[str, ...]
    config: ReferenceCropSupervisionConfig
    pipeline_code_sha256: Mapping[str, str]
    job_fingerprint: str

    @property
    def reference_by_id(self) -> dict[str, ReferenceSpec]:
        return self.expansion.reference_by_id

    @property
    def source_by_id(self) -> dict[str, SourceSpec]:
        return {item.video_id: item for item in self.expansion.sources}


def _crop_output_paths(output_directory: Path, reference_id: str) -> CropOutputPaths:
    # _artifact_paths is the single authority for reference-id safety and the
    # exact upstream final-attribution name.
    _artifact_paths(output_directory, reference_id)
    prefix = output_directory / reference_id
    return CropOutputPaths(
        output_directory=output_directory,
        reference_id=reference_id,
        dataset=Path(
            f"{prefix}-reference-dataset-canonical-v2-final-crop-v1.json"
        ),
        supervision=Path(
            f"{prefix}-reference-supervision-canonical-v2-final-crop-v1.json"
        ),
        status_manifest=Path(f"{prefix}-reference-crop-v1-status.json"),
    )


def _pipeline_code_hashes() -> dict[str, str]:
    pairs = (
        _module_sha256(_pipeline_code_hashes),
        _module_sha256(build_reference_crop_supervision),
        _module_sha256(build_reference_dataset),
        _module_sha256(build_reference_supervision),
        _module_sha256(_load_inputs),
    )
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ReferenceCropJobError("crop pipeline module filenames are not unique")
    return dict(sorted(result.items()))


def _selected_reference_ids(
    inputs: ExpansionInputs,
    requested: Sequence[str],
) -> tuple[str, ...]:
    ordered = tuple(item.video_id for item in inputs.references)
    if not requested:
        return ordered
    normalized = tuple(str(item).strip() for item in requested)
    if any(not item for item in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError("crop reference ids must be non-empty and unique")
    unknown = set(normalized) - set(ordered)
    if unknown:
        raise ValueError(f"unknown crop reference ids: {sorted(unknown)}")
    # Preserve the cadence profile order even when CLI arguments arrive in a
    # different order.  This makes resume status and output deterministic.
    selected = set(normalized)
    return tuple(item for item in ordered if item in selected)


def _load_crop_inputs(
    *,
    cadence_profile_path: Path,
    seed_evidence_path: Path,
    source_manifest_path: Path,
    output_directory: Path,
    style_id: str,
    expected_artist: str,
    expected_song: str,
    expected_reference_count: int,
    expected_source_count: int,
    target_reference_ids: Sequence[str],
    verify_media_hashes: bool,
    config: ReferenceCropSupervisionConfig,
) -> CropJobInputs:
    # The expansion loader is reused so cadence self-hash, seed evidence,
    # acquisition manifest, five reference/audit identities, every caller-
    # declared source clock, source provenance, and (for real runs) every live
    # media hash stay under one authority.  Expansion's target set intentionally
    # excludes its seed; crop-v1 selects its own set afterwards and includes all
    # five.
    expansion = _load_inputs(
        cadence_profile_path=cadence_profile_path,
        seed_evidence_path=seed_evidence_path,
        source_manifest_path=source_manifest_path,
        output_directory=output_directory,
        style_id=style_id,
        expected_artist=expected_artist,
        expected_song=expected_song,
        expected_reference_count=expected_reference_count,
        expected_source_count=expected_source_count,
        target_reference_ids=(),
        verify_media_hashes=verify_media_hashes,
    )
    targets = _selected_reference_ids(expansion, target_reference_ids)
    code_hashes = _pipeline_code_hashes()
    fingerprint = _canonical_json_sha256(
        {
            "analysis": ANALYSIS_BASIS,
            "pipeline_revision": PIPELINE_REVISION,
            "expansion_job_fingerprint": expansion.job_fingerprint,
            "cadence_profile_file_sha256": expansion.profile_file_sha256,
            "cadence_profile_content_sha256": (
                expansion.profile.profile_content_sha256
            ),
            "seed_evidence_file_sha256": expansion.seed_evidence_file_sha256,
            "source_manifest_file_sha256": expansion.source_manifest_file_sha256,
            "style_id": expansion.style_id,
            "configuration": config.to_dict(),
            "pipeline_code_sha256": code_hashes,
        }
    )
    return CropJobInputs(
        expansion=expansion,
        target_reference_ids=targets,
        config=config,
        pipeline_code_sha256=code_hashes,
        job_fingerprint=fingerprint,
    )


def _all_media(inputs: CropJobInputs) -> tuple[MediaIdentity, ...]:
    return (
        *(item.media for item in inputs.expansion.references),
        *(item.audit for item in inputs.expansion.references),
        *(item.media for item in inputs.expansion.sources),
    )


def _source_catalog_identity(source: SourceSpec) -> dict[str, Any]:
    return {
        "video_id": source.video_id,
        "path": str(source.media.path),
        "clock": deepcopy(dict(source.clock)),
        "provenance": deepcopy(dict(source.provenance)),
    }


def _validate_source_catalog(
    value: object,
    *,
    sources: Sequence[SourceSpec],
) -> None:
    observed = _array(value, "canonical attribution sources")
    if len(observed) != len(sources):
        raise ValueError("canonical attribution source count mismatch")
    expected_by_id = {item.video_id: item for item in sources}
    seen: set[str] = set()
    for index, raw in enumerate(observed):
        item = _mapping(raw, f"canonical attribution source {index}")
        source_id = str(item.get("video_id") or item.get("source_id") or "")
        if source_id in seen or source_id not in expected_by_id:
            raise ValueError(f"unknown or duplicate attribution source: {source_id!r}")
        seen.add(source_id)
        expected = expected_by_id[source_id]
        path = Path(str(item.get("path") or item.get("local_path") or "")).expanduser()
        if not path.is_absolute() or path.resolve() != expected.media.path:
            raise ValueError(f"attribution source path mismatch: {source_id}")
        raw_clock = _mapping(item.get("clock"), f"source clock {source_id}")
        normalized_clock = ReferenceSourceClock.from_mapping(raw_clock).to_dict()
        if (
            _canonical_json_bytes(raw_clock)
            != _canonical_json_bytes(expected.clock)
            or _canonical_json_bytes(normalized_clock)
            != _canonical_json_bytes(expected.clock)
        ):
            raise ValueError(f"attribution source clock mismatch: {source_id}")
        provenance = _mapping(
            item.get("provenance"), f"source provenance {source_id}"
        )
        if _canonical_json_bytes(provenance) != _canonical_json_bytes(
            expected.provenance
        ):
            raise ValueError(f"attribution source provenance mismatch: {source_id}")
        media_probe = _mapping(item.get("media_probe"), f"media probe {source_id}")
        if media_probe.get("size_bytes") != expected.media.size_bytes:
            raise ValueError(f"attribution source byte size mismatch: {source_id}")
    if seen != set(expected_by_id):
        raise ValueError("canonical attribution source catalog is incomplete")


def _validate_attribution_caller(
    value: object,
    *,
    inputs: CropJobInputs,
    reference: ReferenceSpec,
) -> None:
    caller = _mapping(value, "reference evidence caller")
    expected_common = {
        "artist": inputs.expansion.profile.artist,
        "song": inputs.expansion.profile.song,
        "reference_video_id": reference.video_id,
    }
    for key, expected in expected_common.items():
        if caller.get(key) != expected:
            raise ValueError(f"reference attribution caller {key} mismatch")
    purpose = caller.get("purpose")
    if purpose == "same_song_professional_reference_learning":
        pass
    elif (
        reference.video_id == inputs.expansion.seed_reference_id
        and purpose
        == "same_song_professional_reference_learning_10source_training_pool"
    ):
        if len(inputs.expansion.sources) <= DEFAULT_EXPECTED_SOURCE_COUNT:
            raise ValueError(
                "training-pool seed caller requires an expanded source pool"
            )
        manifest_binding = _mapping(
            caller.get("training_source_manifest"),
            "training-pool seed source manifest binding",
        )
        if (
            Path(str(manifest_binding.get("path") or "")).expanduser().resolve()
            != inputs.expansion.source_manifest_path
            or manifest_binding.get("file_sha256")
            != inputs.expansion.source_manifest_file_sha256
        ):
            raise ValueError("training-pool seed manifest binding mismatch")
        if caller.get("production_render_admission_modified") is not False:
            raise ValueError("training-pool seed modified production admission")
        catalog_binding = _mapping(
            caller.get("synchronized_source_catalog"),
            "training-pool seed synchronized catalog binding",
        )
        catalog_path = Path(
            str(catalog_binding.get("path") or "")
        ).expanduser().resolve()
        if (
            not catalog_path.is_file()
            or catalog_path.is_symlink()
            or catalog_binding.get("file_sha256") != _sha256_bytes(
                catalog_path.read_bytes()
            )
        ):
            raise ValueError("training-pool seed synchronized catalog binding mismatch")
    else:
        raise ValueError("reference attribution caller purpose mismatch")
    if reference.video_id != inputs.expansion.seed_reference_id:
        _assert_canonical_json_equal(
            caller,
            _expansion_caller_provenance(inputs.expansion, reference.video_id),
            "expansion final-attribution caller",
        )


def _load_final_attribution(
    inputs: CropJobInputs,
    reference: ReferenceSpec,
) -> ValidatedAttribution:
    path = _artifact_paths(
        inputs.expansion.output_directory, reference.video_id
    ).final_attribution
    if not path.is_file() or path.is_symlink():
        raise MissingFinalAttributionError(
            f"missing immutable expansion final attribution: {path}"
        )
    attribution_identity = _stat_identity(path)
    payload, raw = _stable_read_json(path)
    if _stat_identity(path) != attribution_identity:
        raise ReferenceCropJobError(
            f"final attribution changed while entering crop-v1 validation: {path}"
        )
    if reference.video_id == inputs.expansion.seed_reference_id:
        expected_boundary = augment_reference_boundary_requests(
            inputs.expansion.seed_evidence
        ).payload
    else:
        expansion_paths = _artifact_paths(
            inputs.expansion.output_directory, reference.video_id
        )
        if not expansion_paths.evidence.is_file() or not (
            expansion_paths.boundary_evidence.is_file()
        ):
            raise MissingFinalAttributionError(
                "final attribution is missing its immutable expansion evidence "
                f"chain: {expansion_paths.evidence}, "
                f"{expansion_paths.boundary_evidence}"
            )
        base_evidence, _base_raw = _stable_read_json(expansion_paths.evidence)
        _validate_generated_evidence(
            base_evidence,
            inputs=inputs.expansion,
            reference=reference,
        )
        expected_boundary = augment_reference_boundary_requests(
            base_evidence
        ).payload
        boundary_evidence, _boundary_raw = _stable_read_json(
            expansion_paths.boundary_evidence
        )
        _validate_boundary_evidence(
            boundary_evidence,
            base_evidence=base_evidence,
            expected=expected_boundary,
        )
        expected_boundary = boundary_evidence
    expected_final = resolve_reference_attribution_payload(expected_boundary)
    _assert_canonical_json_equal(
        payload,
        expected_final,
        "crop-v1 upstream final canonical attribution",
    )
    if Path(str(payload.get("reference") or "")).expanduser().resolve() != (
        reference.media.path
    ):
        raise ValueError("canonical attribution reference path mismatch")
    _validate_source_catalog(payload.get("sources"), sources=inputs.expansion.sources)

    provenance = _mapping(
        payload.get("reference_evidence_provenance"),
        "reference_evidence_provenance",
    )
    reference_video = _mapping(
        provenance.get("reference_video"), "reference evidence video"
    )
    if Path(str(reference_video.get("path") or "")).expanduser().resolve() != (
        reference.media.path
    ):
        raise ValueError("reference evidence video path mismatch")
    if reference_video.get("size_bytes") != reference.media.size_bytes:
        raise ValueError("reference evidence video byte size mismatch")
    nested_provenance = _mapping(provenance.get("provenance"), "evidence provenance")
    _validate_attribution_caller(
        nested_provenance.get("caller"),
        inputs=inputs,
        reference=reference,
    )

    # The canonical dataset builder is the authority for detector partition,
    # attribution, dense-path, source-run, and boundary self-hashes.  No crop
    # input is supplied here, so this validation performs no large-media I/O.
    base_dataset = build_reference_dataset(
        payload,
        style_id=inputs.expansion.style_id,
        reference_id=reference.video_id,
    ).payload
    shots = _array(payload.get("shots"), "canonical attribution shots")
    accepted: list[Mapping[str, Any]] = []
    accepted_indices: list[int] = []
    source_ids = {item.video_id for item in inputs.expansion.sources}
    for index, raw_shot in enumerate(shots):
        shot = _mapping(raw_shot, f"canonical attribution shot {index}")
        if shot.get("shot_index") != index:
            raise ValueError("canonical attribution shot index mismatch")
        if shot.get("accepted") is True:
            source_id = str(shot.get("attributed_video_id") or "")
            if source_id not in source_ids:
                raise ValueError("accepted attribution shot has unknown source")
            if not isinstance(shot.get("canonical_detected_shot"), Mapping):
                raise ValueError("accepted shot has no canonical detector identity")
            accepted.append(deepcopy(dict(shot)))
            accepted_indices.append(index)
    record_indices = [int(item["shot_index"]) for item in base_dataset["records"]]
    if record_indices != accepted_indices:
        raise ValueError("canonical dataset known shots disagree with accepted shots")
    schema = _mapping(payload.get("canonical_shot_schema"), "canonical shot schema")
    if schema.get("canonical_shot_count") != len(shots):
        raise ValueError("canonical shot count mismatch")
    for key in ("partition_content_sha256", "attribution_content_sha256"):
        _sha256_string(schema.get(key), f"canonical shot schema {key}")
        if provenance.get(key) != schema.get(key):
            raise ValueError(f"canonical provenance {key} mismatch")
    return ValidatedAttribution(
        path=path.resolve(),
        payload=payload,
        file_sha256=_sha256_bytes(raw),
        canonical_json_sha256=_canonical_json_sha256(payload),
        accepted_shots=tuple(accepted),
        stat_identity=attribution_identity,
    )


def _assert_attribution_unchanged(attribution: ValidatedAttribution) -> None:
    if _stat_identity(attribution.path) != attribution.stat_identity:
        raise ReferenceCropJobError(
            "canonical final attribution changed during crop-v1 job: "
            f"{attribution.path}"
        )


def _crop_caller_provenance(
    inputs: CropJobInputs,
    *,
    reference: ReferenceSpec,
    attribution: ValidatedAttribution,
    shot: Mapping[str, Any],
    source_id: str,
) -> dict[str, Any]:
    return {
        "artist": inputs.expansion.profile.artist,
        "song": inputs.expansion.profile.song,
        "reference_video_id": reference.video_id,
        "shot_index": int(shot["shot_index"]),
        "source_video_id": source_id,
        "analysis": ANALYSIS_BASIS,
        "pipeline_revision": PIPELINE_REVISION,
        "style_id": inputs.expansion.style_id,
        "cadence_profile": {
            "path": str(inputs.expansion.profile_path),
            "file_sha256": inputs.expansion.profile_file_sha256,
            "content_sha256": inputs.expansion.profile.profile_content_sha256,
        },
        "seed_evidence": {
            "path": str(inputs.expansion.seed_evidence_path),
            "file_sha256": inputs.expansion.seed_evidence_file_sha256,
        },
        "source_manifest": {
            "path": str(inputs.expansion.source_manifest_path),
            "file_sha256": inputs.expansion.source_manifest_file_sha256,
        },
        "canonical_attribution": {
            "path": str(attribution.path),
            "file_sha256": attribution.file_sha256,
            "canonical_json_sha256": attribution.canonical_json_sha256,
        },
        "configuration": inputs.config.to_dict(),
        "pipeline_code_sha256": deepcopy(dict(inputs.pipeline_code_sha256)),
        "job_fingerprint": inputs.job_fingerprint,
    }


def _crop_request(
    inputs: CropJobInputs,
    *,
    reference: ReferenceSpec,
    attribution: ValidatedAttribution,
    shot: Mapping[str, Any],
) -> ReferenceCropSupervisionRequest:
    source_id = str(shot.get("attributed_video_id") or "")
    source = inputs.source_by_id[source_id]
    return ReferenceCropSupervisionRequest(
        attributed_shot=shot,
        attribution_path=attribution.path,
        reference_media_path=reference.media.path,
        source_media_path=source.media.path,
        reference_clock=ReferenceSourceClock.from_mapping(reference.reference_clock),
        source_clock=ReferenceSourceClock.from_mapping(source.clock),
        source_id=source_id,
        expected_attribution_file_sha256=attribution.file_sha256,
        expected_reference_media_sha256=reference.media.sha256,
        expected_source_media_sha256=source.media.sha256,
        caller_provenance=_crop_caller_provenance(
            inputs,
            reference=reference,
            attribution=attribution,
            shot=shot,
            source_id=source_id,
        ),
    )


def _close_enough(left: object, right: object, name: str) -> None:
    if not math.isclose(_finite(left, name), _finite(right, name), abs_tol=1e-9):
        raise ValueError(f"{name} mismatch")


def _validate_crop_artifact(
    payload: Mapping[str, Any],
    *,
    inputs: CropJobInputs,
    reference: ReferenceSpec,
    attribution: ValidatedAttribution,
    shot: Mapping[str, Any],
) -> None:
    validate_reference_crop_supervision_payload(payload, verify_input_files=False)
    if payload.get("schema_version") != CROP_SUPERVISION_SCHEMA_VERSION:
        raise ValueError("crop supervision schema mismatch")
    if payload.get("analysis_basis") != CROP_SUPERVISION_ANALYSIS_BASIS:
        raise ValueError("crop supervision analysis mismatch")
    if _canonical_json_bytes(payload.get("configuration")) != _canonical_json_bytes(
        inputs.config.to_dict()
    ):
        raise ValueError("crop supervision configuration mismatch")
    source_id = str(shot.get("attributed_video_id") or "")
    source = inputs.source_by_id[source_id]
    observed_shot = _mapping(payload.get("shot"), "crop shot")
    if observed_shot.get("shot_index") != shot.get("shot_index"):
        raise ValueError("crop shot index mismatch")
    if observed_shot.get("source_id") != source_id:
        raise ValueError("crop shot source mismatch")
    _close_enough(
        observed_shot.get("master_start_seconds"),
        shot.get("reference_start_seconds"),
        "crop shot start",
    )
    _close_enough(
        observed_shot.get("master_end_seconds"),
        shot.get("reference_end_seconds"),
        "crop shot end",
    )
    if observed_shot.get("attributed_shot_content_sha256") != (
        _canonical_json_sha256(shot)
    ):
        raise ValueError("crop attributed-shot hash mismatch")
    if _canonical_json_bytes(observed_shot.get("canonical_detected_shot")) != (
        _canonical_json_bytes(shot.get("canonical_detected_shot"))
    ):
        raise ValueError("crop canonical detected-shot identity mismatch")

    mapping = _mapping(payload.get("time_mapping"), "crop time mapping")
    expected_reference_clock = ReferenceSourceClock.from_mapping(
        reference.reference_clock
    ).to_dict()
    expected_source_clock = ReferenceSourceClock.from_mapping(source.clock).to_dict()
    if _canonical_json_bytes(mapping.get("reference_master_to_media")) != (
        _canonical_json_bytes(expected_reference_clock)
    ) or _canonical_json_bytes(mapping.get("source_master_to_media")) != (
        _canonical_json_bytes(expected_source_clock)
    ):
        raise ValueError("crop supervision clock mismatch")

    provenance = _mapping(payload.get("provenance"), "crop provenance")
    expected_inputs = {
        "attribution": (
            attribution.path,
            "file_sha256",
            attribution.file_sha256,
        ),
        "reference_media": (
            reference.media.path,
            "media_sha256",
            reference.media.sha256,
        ),
        "source_media": (
            source.media.path,
            "media_sha256",
            source.media.sha256,
        ),
    }
    for name, (expected_path, digest_key, expected_digest) in expected_inputs.items():
        item = _mapping(provenance.get(name), f"crop provenance {name}")
        if Path(str(item.get("path") or "")).expanduser().resolve() != expected_path:
            raise ValueError(f"crop {name} path mismatch")
        if item.get(digest_key) != expected_digest:
            raise ValueError(f"crop {name} SHA-256 mismatch")
    attribution_provenance = _mapping(
        provenance.get("attribution"), "crop attribution provenance"
    )
    if attribution_provenance.get("canonical_json_sha256") != (
        attribution.canonical_json_sha256
    ):
        raise ValueError("crop canonical-attribution hash mismatch")
    if _mapping(provenance.get("reference_media"), "reference media").get(
        "byte_size"
    ) != reference.media.size_bytes:
        raise ValueError("crop reference byte size mismatch")
    if _mapping(provenance.get("source_media"), "source media").get(
        "byte_size"
    ) != source.media.size_bytes:
        raise ValueError("crop source byte size mismatch")
    expected_caller = _crop_caller_provenance(
        inputs,
        reference=reference,
        attribution=attribution,
        shot=shot,
        source_id=source_id,
    )
    if _canonical_json_bytes(provenance.get("caller")) != _canonical_json_bytes(
        expected_caller
    ):
        raise ValueError("crop caller provenance mismatch")
    if provenance.get("input_hashes_rechecked_after_decode") is not True:
        raise ValueError("crop input hashes were not rechecked after decode")

    samples = _array(payload.get("samples"), "crop samples")
    fractions = inputs.config.sample_fractions
    if len(samples) != len(fractions):
        raise ValueError("crop sample count mismatch")
    start = _finite(shot.get("reference_start_seconds"), "shot start")
    end = _finite(shot.get("reference_end_seconds"), "shot end")
    for index, (raw_sample, fraction) in enumerate(zip(samples, fractions, strict=True)):
        sample = _mapping(raw_sample, f"crop sample {index}")
        if sample.get("sample_index") != index:
            raise ValueError("crop sample index mismatch")
        _close_enough(sample.get("shot_fraction"), fraction, "sample fraction")
        _close_enough(
            sample.get("master_seconds"),
            start + (end - start) * fraction,
            "sample master time",
        )
        if not isinstance(sample.get("accepted"), bool):
            raise ValueError("crop sample decision must be boolean")


def _expected_crop_paths(
    paths: CropOutputPaths,
    attribution: ValidatedAttribution,
) -> dict[Path, Mapping[str, Any]]:
    result: dict[Path, Mapping[str, Any]] = {}
    for shot in attribution.accepted_shots:
        source_id = str(shot["attributed_video_id"])
        path = paths.crop_artifact(int(shot["shot_index"]), source_id)
        if path in result:
            raise ValueError("two accepted shots map to one crop artifact path")
        result[path] = shot
    return result


def _reject_unexpected_crop_artifacts(
    paths: CropOutputPaths,
    expected: Mapping[Path, Mapping[str, Any]],
) -> None:
    expected_paths = set(expected)
    unexpected = sorted(
        (
            item
            for item in paths.output_directory.glob(paths.crop_glob)
            if not item.name.startswith("._") and item not in expected_paths
        ),
        key=str,
    )
    if unexpected:
        raise ExistingArtifactValidationError(
            "crop-v1 artifacts do not belong to the current canonical attribution: "
            + ", ".join(str(item) for item in unexpected)
        )


def _load_or_plan_crop_artifacts(
    inputs: CropJobInputs,
    *,
    reference: ReferenceSpec,
    attribution: ValidatedAttribution,
    dry_run: bool,
    status_callback: Callable[[list[dict[str, Any]]], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = _crop_output_paths(inputs.expansion.output_directory, reference.video_id)
    expected = _expected_crop_paths(paths, attribution)
    _reject_unexpected_crop_artifacts(paths, expected)
    payloads: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    media = _all_media(inputs)
    for artifact_path, shot in expected.items():
        validator = lambda value, current_shot=shot: _validate_crop_artifact(
            value,
            inputs=inputs,
            reference=reference,
            attribution=attribution,
            shot=current_shot,
        )
        if dry_run:
            if artifact_path.exists() or artifact_path.is_symlink():
                payload, digest = _load_existing_artifact(
                    artifact_path,
                    validator=validator,
                )
                action = "validate_and_skip_if_canonical"
                result = {
                    "action": action,
                    "validation": "passed_without_live_media_rehash",
                    "path": str(artifact_path),
                    "sha256": digest,
                    "shot_index": int(shot["shot_index"]),
                    "source_id": str(shot["attributed_video_id"]),
                    "eligible": payload["supervision"]["eligible"] is True,
                    "rejection_reason": payload["supervision"].get(
                        "rejection_reason"
                    ),
                }
                payloads.append(payload)
            else:
                result = {
                    "action": "create_after_baseline_and_expansion",
                    "path": str(artifact_path),
                    "shot_index": int(shot["shot_index"]),
                    "source_id": str(shot["attributed_video_id"]),
                    "decode_sample_count": len(inputs.config.sample_fractions),
                }
            results.append(result)
            continue

        def producer(current_shot: Mapping[str, Any] = shot) -> Mapping[str, Any]:
            active = _active_analysis_processes()
            if active:
                raise ActiveAnalysisError(
                    f"active crosscut analysis blocks crop decode: {active}"
                )
            _assert_stat_unchanged(media)
            built = build_reference_crop_supervision(
                _crop_request(
                    inputs,
                    reference=reference,
                    attribution=attribution,
                    shot=current_shot,
                ),
                config=inputs.config,
            ).payload
            active = _active_analysis_processes()
            if active:
                raise ActiveAnalysisError(
                    "crosscut analysis started during crop decode; refusing "
                    f"artifact publication: {active}"
                )
            _assert_stat_unchanged(media)
            return built

        payload, publication = _materialize_or_skip(
            artifact_path,
            producer=producer,
            validator=validator,
        )
        payloads.append(payload)
        result = {
            **publication,
            "shot_index": int(shot["shot_index"]),
            "source_id": str(shot["attributed_video_id"]),
            "eligible": payload["supervision"]["eligible"] is True,
            "rejection_reason": payload["supervision"].get("rejection_reason"),
            "artifact_content_sha256": payload["provenance"][
                "artifact_content_sha256"
            ],
        }
        results.append(result)
        if status_callback is not None:
            status_callback(results)
    return payloads, results


def _validate_crop_output_plan(
    payload: Mapping[str, Any],
    *,
    inputs: CropJobInputs,
    reference: ReferenceSpec,
    attribution: ValidatedAttribution,
    eligible_artifacts: Sequence[Mapping[str, Any]],
    supervision: bool,
) -> None:
    expected_schema = attribution.payload.get("canonical_shot_schema")
    if _canonical_json_bytes(payload.get("canonical_shot_schema")) != (
        _canonical_json_bytes(expected_schema)
    ):
        raise ValueError("crop output canonical shot schema mismatch")
    if payload.get("style_id") != inputs.expansion.style_id:
        raise ValueError("crop output style id mismatch")
    if payload.get("reference_id") != reference.video_id:
        raise ValueError("crop output reference id mismatch")
    crop_schema = _mapping(payload.get("crop_supervision_schema"), "crop schema")
    expected_indices = sorted(
        int(item["shot"]["shot_index"]) for item in eligible_artifacts
    )
    known_indices = sorted(int(item["shot_index"]) for item in attribution.accepted_shots)
    if crop_schema.get("accepted_shot_indices") != expected_indices:
        raise ValueError("crop output accepted indices mismatch")
    if crop_schema.get("rejected_shot_indices") != []:
        raise ValueError("crop-v1 output must integrate only eligible artifacts")
    if crop_schema.get("missing_shot_indices") != sorted(
        set(known_indices) - set(expected_indices)
    ):
        raise ValueError("crop output loss-mask indices mismatch")
    manifest = _array(crop_schema.get("artifact_manifest"), "crop manifest")
    expected_hashes = [
        item["provenance"]["artifact_content_sha256"]
        for item in eligible_artifacts
    ]
    if [item.get("artifact_content_sha256") for item in manifest] != expected_hashes:
        raise ValueError("crop output artifact manifest mismatch")
    records_key = "shot_labels" if supervision else "records"
    records = _array(payload.get(records_key), f"crop output {records_key}")
    if [int(item["shot_index"]) for item in records] != known_indices:
        raise ValueError("crop output known-shot records mismatch")


def _status_base(
    inputs: CropJobInputs,
    *,
    reference: ReferenceSpec,
    attribution: ValidatedAttribution,
    paths: CropOutputPaths,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_BASIS,
        "pipeline_revision": PIPELINE_REVISION,
        "job_fingerprint": inputs.job_fingerprint,
        "reference_id": reference.video_id,
        "state": "running",
        "style_id": inputs.expansion.style_id,
        "reference_media": {
            "path": str(reference.media.path),
            "sha256": reference.media.sha256,
            "clock": deepcopy(dict(reference.reference_clock)),
        },
        "canonical_attribution": {
            "path": str(attribution.path),
            "file_sha256": attribution.file_sha256,
            "canonical_json_sha256": attribution.canonical_json_sha256,
            "accepted_shot_count": len(attribution.accepted_shots),
        },
        "inputs": {
            "cadence_profile_file_sha256": inputs.expansion.profile_file_sha256,
            "cadence_profile_content_sha256": (
                inputs.expansion.profile.profile_content_sha256
            ),
            "seed_evidence_file_sha256": inputs.expansion.seed_evidence_file_sha256,
            "source_manifest_file_sha256": (
                inputs.expansion.source_manifest_file_sha256
            ),
            "reference_count": len(inputs.expansion.references),
            "source_count": len(inputs.expansion.sources),
            "media_sha256_verified": inputs.expansion.media_sha256_verified,
            "pipeline_code_sha256": deepcopy(dict(inputs.pipeline_code_sha256)),
        },
        "configuration": inputs.config.to_dict(),
        "artifacts": paths.to_dict(),
        "stages": {
            "crop_artifacts": {
                "status": "pending",
                "expected_count": len(attribution.accepted_shots),
                "shots": [],
            },
            "dataset": {"status": "pending", "path": str(paths.dataset)},
            "supervision": {
                "status": "pending",
                "path": str(paths.supervision),
            },
        },
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "completed_at": None,
        "failure": None,
    }


def _load_previous_status(
    path: Path,
    *,
    inputs: CropJobInputs,
    reference_id: str,
) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ExistingArtifactValidationError(f"invalid crop status manifest: {path}")
    payload, _raw = _stable_read_json(path)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("analysis") != ANALYSIS_BASIS
        or payload.get("pipeline_revision") != PIPELINE_REVISION
        or payload.get("job_fingerprint") != inputs.job_fingerprint
        or payload.get("reference_id") != reference_id
    ):
        raise ExistingArtifactValidationError(
            f"crop status manifest belongs to another or invalid job: {path}"
        )
    return payload


def _write_status(path: Path, payload: Mapping[str, Any]) -> None:
    _cleanup_stale_temps(path)
    _atomic_update_json(path, payload)


def _run_one_reference(
    inputs: CropJobInputs,
    *,
    reference: ReferenceSpec,
    attribution: ValidatedAttribution,
    emit: Callable[[Mapping[str, Any]], None],
) -> dict[str, Any]:
    paths = _crop_output_paths(inputs.expansion.output_directory, reference.video_id)
    previous = _load_previous_status(
        paths.status_manifest,
        inputs=inputs,
        reference_id=reference.video_id,
    )
    status = _status_base(
        inputs,
        reference=reference,
        attribution=attribution,
        paths=paths,
    )
    if previous is not None:
        status["started_at"] = previous.get("started_at", status["started_at"])
    _write_status(paths.status_manifest, status)

    def update_crop_status(results: list[dict[str, Any]]) -> None:
        eligible_count = sum(item.get("eligible") is True for item in results)
        crop_stage = status["stages"]["crop_artifacts"]
        crop_stage.update(
            {
                "status": "running",
                "completed_count": len(results),
                "eligible_count": eligible_count,
                "rejected_count": len(results) - eligible_count,
                "shots": deepcopy(results),
            }
        )
        status["updated_at"] = _utc_now()
        _write_status(paths.status_manifest, status)

    try:
        active = _active_analysis_processes()
        if active:
            raise ActiveAnalysisError(
                f"active crosscut analysis blocks crop-v1 media work: {active}"
            )
        _assert_stat_unchanged(_all_media(inputs))
        _assert_attribution_unchanged(attribution)
        artifacts, crop_results = _load_or_plan_crop_artifacts(
            inputs,
            reference=reference,
            attribution=attribution,
            dry_run=False,
            status_callback=update_crop_status,
        )
        eligible = [
            item for item in artifacts if item["supervision"]["eligible"] is True
        ]
        status["stages"]["crop_artifacts"].update(
            {
                "status": "complete",
                "completed_count": len(crop_results),
                "eligible_count": len(eligible),
                "rejected_count": len(crop_results) - len(eligible),
                "shots": deepcopy(crop_results),
                "completed_at": _utc_now(),
            }
        )
        status["updated_at"] = _utc_now()
        _write_status(paths.status_manifest, status)
        _assert_attribution_unchanged(attribution)
        emit(
            {
                "reference_id": reference.video_id,
                "stage": "crop_artifacts",
                "status": "complete",
                "artifact_count": len(artifacts),
                "eligible_count": len(eligible),
            }
        )

        def build_dataset() -> Mapping[str, Any]:
            active_now = _active_analysis_processes()
            if active_now:
                raise ActiveAnalysisError(
                    f"active analysis blocks crop dataset rebuild: {active_now}"
                )
            _assert_stat_unchanged(_all_media(inputs))
            _assert_attribution_unchanged(attribution)
            built = build_reference_dataset(
                attribution.payload,
                style_id=inputs.expansion.style_id,
                reference_id=reference.video_id,
                crop_supervision_artifacts=eligible,
            ).payload
            _assert_stat_unchanged(_all_media(inputs))
            _assert_attribution_unchanged(attribution)
            active_now = _active_analysis_processes()
            if active_now:
                raise ActiveAnalysisError(
                    "analysis started during crop dataset rebuild; refusing "
                    f"publication: {active_now}"
                )
            return built

        expected_dataset = build_dataset()
        _dataset, dataset_result = _materialize_or_skip(
            paths.dataset,
            producer=lambda: expected_dataset,
            validator=lambda payload: _validate_dataset_payload(
                payload, expected_dataset
            ),
        )
        status["stages"]["dataset"] = {
            "status": "complete",
            **dataset_result,
            "completed_at": _utc_now(),
        }
        status["updated_at"] = _utc_now()
        _write_status(paths.status_manifest, status)
        emit(
            {
                "reference_id": reference.video_id,
                "stage": "dataset",
                "status": "complete",
                **dataset_result,
            }
        )

        def build_supervision() -> Mapping[str, Any]:
            active_now = _active_analysis_processes()
            if active_now:
                raise ActiveAnalysisError(
                    f"active analysis blocks crop supervision rebuild: {active_now}"
                )
            _assert_stat_unchanged(_all_media(inputs))
            _assert_attribution_unchanged(attribution)
            built = build_reference_supervision(
                attribution.payload,
                style_id=inputs.expansion.style_id,
                reference_id=reference.video_id,
                crop_supervision_artifacts=eligible,
            ).payload
            _assert_stat_unchanged(_all_media(inputs))
            _assert_attribution_unchanged(attribution)
            active_now = _active_analysis_processes()
            if active_now:
                raise ActiveAnalysisError(
                    "analysis started during crop supervision rebuild; refusing "
                    f"publication: {active_now}"
                )
            return built

        expected_supervision = build_supervision()
        _supervision, supervision_result = _materialize_or_skip(
            paths.supervision,
            producer=lambda: expected_supervision,
            validator=lambda payload: _assert_canonical_json_equal(
                payload,
                expected_supervision,
                "crop-v1 reference supervision",
            ),
        )
        status["stages"]["supervision"] = {
            "status": "complete",
            **supervision_result,
            "completed_at": _utc_now(),
        }
        status["state"] = "complete"
        status["completed_at"] = _utc_now()
        status["updated_at"] = status["completed_at"]
        _write_status(paths.status_manifest, status)
        emit(
            {
                "reference_id": reference.video_id,
                "stage": "supervision",
                "status": "complete",
                **supervision_result,
            }
        )
        return {
            "reference_id": reference.video_id,
            "status": "complete",
            "status_manifest": str(paths.status_manifest),
            "accepted_attribution_shot_count": len(attribution.accepted_shots),
            "crop_artifact_count": len(artifacts),
            "crop_eligible_count": len(eligible),
            "crop_rejected_count": len(artifacts) - len(eligible),
            "dataset_record_count": len(expected_dataset["records"]),
            "supervision_shot_label_count": len(
                expected_supervision["shot_labels"]
            ),
            "stages": deepcopy(status["stages"]),
        }
    except BaseException as exc:
        status["state"] = "failed"
        status["updated_at"] = _utc_now()
        status["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "at": status["updated_at"],
        }
        _write_status(paths.status_manifest, status)
        raise


def _plan_one_reference(
    inputs: CropJobInputs,
    reference: ReferenceSpec,
) -> dict[str, Any]:
    upstream = _artifact_paths(
        inputs.expansion.output_directory, reference.video_id
    ).final_attribution
    paths = _crop_output_paths(inputs.expansion.output_directory, reference.video_id)
    existing_crop_files = [
        item
        for item in paths.output_directory.glob(paths.crop_glob)
        if not item.name.startswith("._")
    ]
    if not upstream.exists():
        unexpected = existing_crop_files + [
            item
            for item in (paths.dataset, paths.supervision, paths.status_manifest)
            if item.exists() or item.is_symlink()
        ]
        if unexpected:
            raise ExistingArtifactValidationError(
                "crop-v1 derived artifacts exist without final canonical attribution: "
                + ", ".join(str(item) for item in unexpected)
            )
        return {
            "reference_id": reference.video_id,
            "status": "waiting_for_expansion_final_attribution",
            "final_attribution": str(upstream),
            "outputs": paths.to_dict(),
            "crop_artifact_count": None,
            "decode_sample_count": None,
        }

    attribution = _load_final_attribution(inputs, reference)
    artifacts, crop_actions = _load_or_plan_crop_artifacts(
        inputs,
        reference=reference,
        attribution=attribution,
        dry_run=True,
    )
    missing_count = sum(item["action"].startswith("create") for item in crop_actions)
    eligible_existing = [
        item for item in artifacts if item["supervision"]["eligible"] is True
    ]
    outputs: dict[str, Any] = {}
    for name, path, supervision in (
        ("dataset", paths.dataset, False),
        ("supervision", paths.supervision, True),
    ):
        if path.exists() or path.is_symlink():
            if missing_count:
                raise ExistingArtifactValidationError(
                    f"{name} exists before every crop artifact is complete: {path}"
                )
            _payload, digest = _load_existing_artifact(
                path,
                validator=lambda payload, is_supervision=supervision: (
                    _validate_crop_output_plan(
                        payload,
                        inputs=inputs,
                        reference=reference,
                        attribution=attribution,
                        eligible_artifacts=eligible_existing,
                        supervision=is_supervision,
                    )
                ),
            )
            outputs[name] = {
                "action": "existing_requires_full_real_run_validation",
                "validation": "deferred_not_claimed_passed",
                "structural_envelope_check": "passed",
                "full_semantic_rebuild_comparison": "deferred_to_real_run",
                "path": str(path),
                "sha256": digest,
            }
        else:
            outputs[name] = {
                "action": "create_after_crop_artifacts",
                "path": str(path),
            }
    return {
        "reference_id": reference.video_id,
        "status": "planned",
        "final_attribution": {
            "path": str(attribution.path),
            "file_sha256": attribution.file_sha256,
            "canonical_json_sha256": attribution.canonical_json_sha256,
            "validation": "passed_without_live_media_rehash",
        },
        "accepted_attribution_shot_count": len(attribution.accepted_shots),
        "crop_artifacts_missing_count": missing_count,
        "crop_artifacts": crop_actions,
        "outputs": outputs,
    }


def run_reference_crop_job(
    *,
    cadence_profile_path: Path,
    seed_evidence_path: Path,
    source_manifest_path: Path,
    output_directory: Path,
    style_id: str,
    expected_artist: str,
    expected_song: str,
    expected_reference_count: int = DEFAULT_EXPECTED_REFERENCE_COUNT,
    expected_source_count: int = DEFAULT_EXPECTED_SOURCE_COUNT,
    target_reference_ids: Sequence[str] = (),
    minimum_free_bytes: int = DEFAULT_MINIMUM_FREE_BYTES,
    dry_run: bool = False,
    config: ReferenceCropSupervisionConfig | None = None,
    emit: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Plan or resume crop-v1 artifacts for canonical final attributions."""

    if expected_reference_count != DEFAULT_EXPECTED_REFERENCE_COUNT:
        raise ValueError(
            "crop-v1 requires exactly "
            f"{DEFAULT_EXPECTED_REFERENCE_COUNT} cadence references"
        )
    if expected_source_count < MINIMUM_EXPECTED_SOURCE_COUNT:
        raise ValueError(
            "crop-v1 requires at least "
            f"{MINIMUM_EXPECTED_SOURCE_COUNT} synchronized sources"
        )
    if minimum_free_bytes < 0:
        raise ValueError("minimum_free_bytes cannot be negative")
    active = _active_analysis_processes()
    if active and not dry_run:
        raise ActiveAnalysisError(
            "another crosscut analysis is active; refusing media hashes/decode: "
            f"{active}"
        )
    inputs = _load_crop_inputs(
        cadence_profile_path=cadence_profile_path,
        seed_evidence_path=seed_evidence_path,
        source_manifest_path=source_manifest_path,
        output_directory=output_directory,
        style_id=style_id,
        expected_artist=expected_artist,
        expected_song=expected_song,
        expected_reference_count=expected_reference_count,
        expected_source_count=expected_source_count,
        target_reference_ids=target_reference_ids,
        verify_media_hashes=not dry_run,
        config=config or ReferenceCropSupervisionConfig(),
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_BASIS,
        "pipeline_revision": PIPELINE_REVISION,
        "dry_run": dry_run,
        "job_fingerprint": inputs.job_fingerprint,
        "active_analysis_processes": [dict(item) for item in active],
        "input_validation": {
            "cadence_profile": "passed",
            "cadence_profile_file_sha256": inputs.expansion.profile_file_sha256,
            "cadence_profile_content_sha256": (
                inputs.expansion.profile.profile_content_sha256
            ),
            "seed_evidence": "passed",
            "seed_evidence_file_sha256": (
                inputs.expansion.seed_evidence_file_sha256
            ),
            "source_manifest": "passed",
            "source_manifest_file_sha256": (
                inputs.expansion.source_manifest_file_sha256
            ),
            "reference_count": len(inputs.expansion.references),
            "source_count": len(inputs.expansion.sources),
            "all_reference_and_source_clocks": "passed",
            "media_sha256": (
                "passed"
                if inputs.expansion.media_sha256_verified
                else "deferred_until_real_run"
            ),
            "pipeline_code_sha256": deepcopy(dict(inputs.pipeline_code_sha256)),
        },
        "configuration": inputs.config.to_dict(),
        "target_reference_ids": list(inputs.target_reference_ids),
        "reference_count": len(inputs.target_reference_ids),
        "references": [],
    }
    if dry_run:
        result["execution_blocked_until_active_analysis_exits"] = bool(active)
        result["writes_performed"] = False
        result["references"] = [
            _plan_one_reference(inputs, inputs.reference_by_id[reference_id])
            for reference_id in inputs.target_reference_ids
        ]
        result["ready_reference_count"] = sum(
            item["status"] == "planned" for item in result["references"]
        )
        result["waiting_reference_count"] = sum(
            item["status"] == "waiting_for_expansion_final_attribution"
            for item in result["references"]
        )
        return result

    free_bytes = shutil.disk_usage(inputs.expansion.output_directory).free
    if free_bytes < minimum_free_bytes:
        raise ReferenceCropJobError(
            f"insufficient free space: {free_bytes} < {minimum_free_bytes}"
        )
    result["free_bytes_before"] = free_bytes
    # Validate every requested final attribution before publishing any crop-v1
    # state.  This prevents a partly expanded corpus from being mistaken for a
    # complete five-reference run.
    attributions = {
        reference_id: _load_final_attribution(
            inputs, inputs.reference_by_id[reference_id]
        )
        for reference_id in inputs.target_reference_ids
    }
    event_sink = emit or (lambda _event: None)
    with _job_lock(inputs.expansion.output_directory):
        for reference_id in inputs.target_reference_ids:
            result["references"].append(
                _run_one_reference(
                    inputs,
                    reference=inputs.reference_by_id[reference_id],
                    attribution=attributions[reference_id],
                    emit=event_sink,
                )
            )
    result["writes_performed"] = True
    result["status"] = "complete"
    result["completed_reference_count"] = len(result["references"])
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m crosscut.reference_crop_job",
        description=(
            "safely resume multiframe crop-v1 audits and immutable canonical "
            "dataset/supervision rebuilds for five professional references"
        ),
    )
    parser.add_argument("--cadence-profile", required=True, type=Path)
    parser.add_argument("--seed-evidence", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
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
    parser.add_argument(
        "--reference-id",
        action="append",
        default=[],
        help="optional cadence reference id; repeat to run a subset",
    )
    parser.add_argument(
        "--minimum-free-bytes",
        type=int,
        default=DEFAULT_MINIMUM_FREE_BYTES,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _stderr_event(event: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            {"analysis": ANALYSIS_BASIS, **dict(event)},
            ensure_ascii=False,
            allow_nan=False,
        ),
        file=sys.stderr,
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        result = run_reference_crop_job(
            cadence_profile_path=arguments.cadence_profile,
            seed_evidence_path=arguments.seed_evidence,
            source_manifest_path=arguments.source_manifest,
            output_directory=arguments.output_directory,
            style_id=arguments.style_id,
            expected_artist=arguments.expected_artist,
            expected_song=arguments.expected_song,
            expected_reference_count=arguments.expected_reference_count,
            expected_source_count=arguments.expected_source_count,
            target_reference_ids=tuple(arguments.reference_id),
            minimum_free_bytes=arguments.minimum_free_bytes,
            dry_run=arguments.dry_run,
            emit=_stderr_event,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "analysis": ANALYSIS_BASIS,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_BASIS",
    "PIPELINE_REVISION",
    "SCHEMA_VERSION",
    "MissingFinalAttributionError",
    "ReferenceCropJobError",
    "main",
    "run_reference_crop_job",
]
