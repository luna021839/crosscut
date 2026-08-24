"""Resumable crop-v2 materialization over immutable final attributions.

The job intentionally accepts explicit ``(reference id, attribution, evidence)``
triples.  It does not rediscover an evidence file by filename and it does not
fall back to crop-v1.  Every conservatively accepted detector-owned shot is
either materialized, or an existing byte- and provenance-bound crop-v2
artifact is validated and reported as an identical resume.

Dry runs read and hash the comparatively small JSON inputs, but never hash or
decode media and never write.  Real runs delegate pixel analysis to the public
``reference_dynamic_crop_supervision`` API.  Ineligible analyses remain
immutable loss-masked artifacts; no shot is silently skipped, copied, or given
a synthetic time-zero keyframe.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import threading
from typing import Any, Mapping, Sequence

from . import reference_dynamic_crop_supervision as _dynamic_crop_module
from .dynamic_crop_input_verification import verify_dynamic_crop_input_files
from .reference_attribution import (
    REFERENCE_ATTRIBUTION_BASIS,
    REFERENCE_EVIDENCE_ANALYSIS_BASIS,
)
from .reference_dynamic_crop_supervision import (
    ANALYSIS_BASIS as DYNAMIC_CROP_ANALYSIS_BASIS,
    DYNAMIC_CROP_NAMESPACE,
    DYNAMIC_CROP_TARGET_CONTRACT,
    ReferenceDynamicCropSupervisionArtifact,
    ReferenceDynamicCropSupervisionConfig,
    ReferenceDynamicCropSupervisionRequest,
    build_reference_dynamic_crop_supervision,
    load_reference_dynamic_crop_supervision,
    reference_dynamic_crop_v2_filename,
    validate_reference_dynamic_crop_supervision_payload,
)
from .reference_evidence_builder import ReferenceSourceClock


SCHEMA_VERSION = 1
ANALYSIS_BASIS = "canonical_final_reference_dynamic_crop_batch_v3"
PIPELINE_REVISION = "explicit_final_attribution_evidence_to_crop_v2_v3"
MANIFEST_HASH_BASIS = (
    "sha256_canonical_json_with_provenance.manifest_content_sha256_null"
)
MANIFEST_PREFIX = "reference-dynamic-crop-v3-manifest"
REJECTED_PERSPECTIVE_DIAGNOSTIC_MISSING_REASON = (
    "undefined_invalid_projective_denominator"
)


_ESTIMATOR_AUDIT_ADAPTER_LOCK = threading.RLock()


class ReferenceDynamicCropJobError(RuntimeError):
    """Raised when crop-v2 orchestration cannot prove an exact result."""


class ExistingDynamicCropArtifactError(ReferenceDynamicCropJobError):
    """Raised when an existing immutable artifact is not the expected one."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _nonfinite_float_locations(
    value: object,
    path: tuple[str | int, ...] = (),
) -> list[tuple[tuple[str | int, ...], float]]:
    """Return every non-finite JSON number without coercing any payload value."""

    if isinstance(value, float):
        return [] if math.isfinite(value) else [(path, value)]
    if isinstance(value, Mapping):
        result: list[tuple[tuple[str | int, ...], float]] = []
        for key, item in value.items():
            result.extend(_nonfinite_float_locations(item, (*path, str(key))))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, item in enumerate(value):
            result.extend(_nonfinite_float_locations(item, (*path, index)))
        return result
    return []


def _normalize_rejected_estimator_audit(
    transform: object,
    rejection_reason: object,
    diagnostics: object,
) -> tuple[object, object, object]:
    """Make one known rejected-candidate diagnostic JSON-safe, or fail closed.

    ``reference_crop_supervision._perspective_distortion`` deliberately returns
    positive infinity when a projective denominator is invalid.  The estimator
    correctly rejects that candidate as ``invalid_perspective_geometry``, but
    its audit dictionary historically retained the infinity and therefore
    could not be content-addressed as strict JSON.  For that already rejected,
    target-free candidate only, retain the semantic absence as ``None`` and an
    explicit reason.  No transform, score, accepted candidate, or target value
    is clamped or synthesized.
    """

    locations = _nonfinite_float_locations(diagnostics)
    if not locations:
        return transform, rejection_reason, diagnostics
    allowed = (
        transform is None
        and rejection_reason == "invalid_perspective_geometry"
        and len(locations) == 1
        and locations[0][0] == ("perspective_distortion",)
        and math.isinf(locations[0][1])
        and locations[0][1] > 0.0
        and isinstance(diagnostics, Mapping)
    )
    if not allowed:
        rendered = ", ".join(
            f"{'.'.join(str(part) for part in location) or '<root>'}={numeric!r}"
            for location, numeric in locations
        )
        raise ReferenceDynamicCropJobError(
            "unsafe non-finite estimator diagnostic outside the exact rejected "
            f"invalid-perspective contract: {rendered}"
        )
    normalized = deepcopy(dict(diagnostics))
    normalized["perspective_distortion"] = None
    normalized["perspective_distortion_missing"] = {
        "reason": REJECTED_PERSPECTIVE_DIAGNOSTIC_MISSING_REASON,
        "candidate_rejected": True,
        "target_value_used": False,
        "numeric_value_clamped_or_substituted": False,
    }
    if _nonfinite_float_locations(normalized):
        raise ReferenceDynamicCropJobError(
            "rejected perspective diagnostic normalization left non-finite data"
        )
    _canonical_json_bytes(normalized)
    return transform, rejection_reason, normalized


def _build_reference_dynamic_crop_supervision_job_safe(
    request: ReferenceDynamicCropSupervisionRequest,
    *,
    config: ReferenceDynamicCropSupervisionConfig,
) -> ReferenceDynamicCropSupervisionArtifact:
    """Call the public generator with a narrow rejected-audit JSON adapter.

    The public generator imports the static estimator as a module global.  The
    adapter is installed only around one synchronous build, serialized within
    this process, and restored in ``finally``.  This keeps the generator source
    and its provenance hash unchanged while refusing every non-finite case
    except the known target-free invalid-perspective audit value above.
    """

    with _ESTIMATOR_AUDIT_ADAPTER_LOCK:
        estimator = _dynamic_crop_module._estimate_sample

        def estimator_with_json_safe_rejected_audit(
            *args: object, **kwargs: object
        ) -> tuple[object, object, object]:
            return _normalize_rejected_estimator_audit(
                *estimator(*args, **kwargs)
            )

        _dynamic_crop_module._estimate_sample = (
            estimator_with_json_safe_rejected_audit
        )
        try:
            return build_reference_dynamic_crop_supervision(request, config=config)
        finally:
            _dynamic_crop_module._estimate_sample = estimator


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _stable_json(path: Path, name: str) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file: {path}")
    before = _stat_identity(path)
    raw = path.read_bytes()
    after = _stat_identity(path)
    if before != after:
        raise ReferenceDynamicCropJobError(f"{name} changed while being read: {path}")
    value = json.loads(raw)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} JSON root must be an object")
    return deepcopy(dict(value)), hashlib.sha256(raw).hexdigest()


def _clean_reference_id(value: str) -> str:
    normalized = str(value).strip()
    if not normalized or re.fullmatch(r"[A-Za-z0-9_-]+", normalized) is None:
        raise ValueError(f"unsafe reference id: {value!r}")
    return normalized


@dataclass(frozen=True, slots=True)
class DynamicCropReferencePair:
    reference_id: str
    attribution_path: Path
    evidence_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_id", _clean_reference_id(self.reference_id))
        for name in ("attribution_path", "evidence_path"):
            path = Path(getattr(self, name)).expanduser()
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(f"{name} is not a regular file: {path}")
            object.__setattr__(self, name, path.resolve())
        if self.attribution_path == self.evidence_path:
            raise ValueError("attribution and evidence paths must be distinct")


@dataclass(frozen=True, slots=True)
class DynamicCropJobResult:
    dry_run: bool
    plan: tuple[Mapping[str, Any], ...]
    built_count: int
    resumed_count: int
    manifest_path: Path | None
    manifest_content_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "built_count": self.built_count,
            "resumed_identical_count": self.resumed_count,
            "plan": [deepcopy(dict(item)) for item in self.plan],
            "manifest_path": (
                str(self.manifest_path) if self.manifest_path is not None else None
            ),
            "manifest_content_sha256": self.manifest_content_sha256,
        }


def discover_reference_dynamic_crop_pairs(
    directory: Path,
    *,
    expected_reference_count: int | None = 5,
) -> tuple[DynamicCropReferencePair, ...]:
    """Discover only exact final-attribution/boundary-evidence filename pairs.

    Discovery is deliberately strict: AppleDouble files are ignored, an
    attribution without its exact same-prefix boundary evidence is fatal, and
    the default five-reference count prevents an active expansion directory
    from being mistaken for a complete batch input.
    """

    root = Path(directory).expanduser()
    if not root.is_dir() or root.is_symlink():
        raise NotADirectoryError(root)
    if expected_reference_count is not None and (
        isinstance(expected_reference_count, bool)
        or not isinstance(expected_reference_count, int)
        or expected_reference_count <= 0
    ):
        raise ValueError("expected_reference_count must be positive or None")
    attribution_suffix = "-reference-attribution-canonical-v2-final.json"
    evidence_suffix = "-reference-evidence-canonical-v2-boundaries.json"
    attributions = sorted(
        path
        for path in root.glob(f"*{attribution_suffix}")
        if path.is_file() and not path.is_symlink() and not path.name.startswith("._")
    )
    pairs: list[DynamicCropReferencePair] = []
    for attribution_path in attributions:
        reference_id = attribution_path.name[: -len(attribution_suffix)]
        reference_id = _clean_reference_id(reference_id)
        evidence_path = root / f"{reference_id}{evidence_suffix}"
        if not evidence_path.is_file() or evidence_path.is_symlink():
            raise FileNotFoundError(
                "final attribution is missing its exact boundary evidence pair: "
                f"{evidence_path}"
            )
        pairs.append(
            DynamicCropReferencePair(reference_id, attribution_path, evidence_path)
        )
    if expected_reference_count is not None and len(pairs) != expected_reference_count:
        raise ReferenceDynamicCropJobError(
            "discovered final attribution/evidence pair count mismatch: "
            f"expected {expected_reference_count}, found {len(pairs)} in {root.resolve()}"
        )
    if not pairs:
        raise ReferenceDynamicCropJobError(
            f"no final attribution/evidence pairs discovered in {root.resolve()}"
        )
    return tuple(pairs)


@dataclass(frozen=True, slots=True)
class _ValidatedPair:
    spec: DynamicCropReferencePair
    attribution: Mapping[str, Any]
    evidence: Mapping[str, Any]
    attribution_file_sha256: str
    evidence_file_sha256: str
    reference_media_path: Path
    source_by_id: Mapping[str, Mapping[str, Any]]
    accepted_shots: tuple[Mapping[str, Any], ...]
    attribution_stat_identity: tuple[int, int, int, int]
    evidence_stat_identity: tuple[int, int, int, int]


def _evidence_reference_id(value: Mapping[str, Any]) -> str:
    provenance = value.get("provenance")
    caller = provenance.get("caller") if isinstance(provenance, Mapping) else None
    if not isinstance(caller, Mapping):
        raise ValueError("evidence caller provenance is missing")
    reference_id = str(caller.get("reference_video_id") or "")
    return _clean_reference_id(reference_id)


def _attribution_reference_id(value: Mapping[str, Any]) -> str:
    provenance = _mapping(
        value.get("reference_evidence_provenance"),
        "attribution reference-evidence provenance",
    )
    nested = provenance.get("provenance")
    caller = nested.get("caller") if isinstance(nested, Mapping) else None
    if not isinstance(caller, Mapping):
        raise ValueError("attribution evidence caller provenance is missing")
    return _clean_reference_id(str(caller.get("reference_video_id") or ""))


def _validate_media_stat(path: Path, probe: Mapping[str, Any], name: str) -> None:
    if not path.is_absolute() or not path.is_file():
        raise FileNotFoundError(f"{name} is not an existing absolute file: {path}")
    size = probe.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{name} probe size_bytes is invalid")
    if path.stat().st_size != size:
        raise ValueError(f"{name} byte size no longer matches evidence")
    frame_rate = _finite(probe.get("frame_rate"), f"{name} frame rate")
    frame_count = probe.get("declared_frame_count")
    if frame_rate <= 0.0 or isinstance(frame_count, bool) or not isinstance(
        frame_count, int
    ) or frame_count <= 0:
        raise ValueError(f"{name} media probe is incomplete")


def _partition_content_sha256(evidence: Mapping[str, Any]) -> str:
    start = _finite(evidence.get("master_start_seconds"), "evidence master start")
    end = _finite(evidence.get("master_end_seconds"), "evidence master end")
    if start < 0.0 or end <= start:
        raise ValueError("evidence master bounds are invalid")
    cuts = _array(evidence.get("detected_reference_cuts"), "detected cuts")
    cut_times: list[float] = []
    previous = start
    for index, raw in enumerate(cuts):
        cut = _mapping(raw, f"detected cut {index}")
        timestamp = _finite(cut.get("master_seconds"), "detected cut time")
        if not previous < timestamp < end:
            raise ValueError("detected cut times must be strictly chronological")
        cut_times.append(timestamp)
        previous = timestamp
    samples = _array(evidence.get("samples"), "evidence samples")
    if not samples:
        raise ValueError("evidence samples cannot be empty")
    groups: list[list[int]] = [[] for _ in range(len(cut_times) + 1)]
    for index, raw in enumerate(samples):
        sample = _mapping(raw, f"evidence sample {index}")
        timestamp = _finite(sample.get("master_seconds"), "sample master time")
        if timestamp < start - 1e-9 or timestamp >= end + 1e-9:
            raise ValueError("evidence sample is outside the master domain")
        group = 0
        while group < len(cut_times) and timestamp >= cut_times[group]:
            group += 1
        groups[group].append(index)
    return _canonical_json_sha256(
        {
            "master_start_seconds": start,
            "master_end_seconds": end,
            "detected_reference_cuts": deepcopy(cuts),
            "sample_assignment": groups,
        }
    )


def _validate_pair(spec: DynamicCropReferencePair) -> _ValidatedPair:
    attribution, attribution_hash = _stable_json(
        spec.attribution_path, "final attribution"
    )
    evidence, evidence_hash = _stable_json(spec.evidence_path, "reference evidence")
    if attribution.get("analysis") != REFERENCE_ATTRIBUTION_BASIS:
        raise ValueError("pair attribution is not a canonical final attribution")
    if evidence.get("analysis") != REFERENCE_EVIDENCE_ANALYSIS_BASIS:
        raise ValueError("pair evidence has an unsupported analysis basis")
    if _evidence_reference_id(evidence) != spec.reference_id:
        raise ValueError("pair reference id differs from evidence caller provenance")
    if _attribution_reference_id(attribution) != spec.reference_id:
        raise ValueError("pair reference id differs from attribution caller provenance")

    reference_paths = (
        Path(str(evidence.get("reference") or "")).expanduser(),
        Path(
            str(_mapping(evidence.get("reference_video"), "reference video").get("path") or "")
        ).expanduser(),
        Path(str(attribution.get("reference") or "")).expanduser(),
    )
    if any(not path.is_absolute() for path in reference_paths):
        raise ValueError("pair reference media paths must be absolute")
    resolved_reference_paths = tuple(path.resolve() for path in reference_paths)
    if len(set(resolved_reference_paths)) != 1:
        raise ValueError("attribution and evidence reference media paths differ")
    reference_path = resolved_reference_paths[0]
    _validate_media_stat(
        reference_path,
        _mapping(evidence.get("reference_video"), "reference media probe"),
        "reference media",
    )

    evidence_sources = _array(evidence.get("sources"), "evidence sources")
    attribution_sources = _array(attribution.get("sources"), "attribution sources")
    if _canonical_json_bytes(evidence_sources) != _canonical_json_bytes(
        attribution_sources
    ):
        raise ValueError("attribution and evidence source catalogs differ")
    source_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(evidence_sources):
        source = _mapping(raw, f"source catalog entry {index}")
        source_id = _clean_reference_id(str(source.get("video_id") or ""))
        if source_id in source_by_id:
            raise ValueError(f"duplicate source catalog id: {source_id}")
        source_path = Path(str(source.get("path") or "")).expanduser()
        if not source_path.is_absolute():
            raise ValueError(f"source media path must be absolute: {source_id}")
        resolved_source_path = source_path.resolve()
        if resolved_source_path == reference_path:
            raise ValueError("reference media cannot also be an attributed source")
        _validate_media_stat(
            resolved_source_path,
            _mapping(source.get("media_probe"), f"source media probe {source_id}"),
            f"source media {source_id}",
        )
        clock = _mapping(source.get("clock"), f"source clock {source_id}")
        ReferenceSourceClock.from_mapping(clock)
        normalized = deepcopy(dict(source))
        normalized["resolved_path"] = str(resolved_source_path)
        source_by_id[source_id] = normalized
    if not source_by_id:
        raise ValueError("source catalog cannot be empty")

    clock_policy = _mapping(evidence.get("clock_policy"), "evidence clock policy")
    ReferenceSourceClock.from_mapping(
        _mapping(clock_policy.get("reference_mapping"), "reference clock")
    )
    schema = _mapping(attribution.get("canonical_shot_schema"), "shot schema")
    if (
        schema.get("schema_version") != 1
        or schema.get("partition_basis") != "reference_hard_cut_detector"
        or schema.get("source_runs_are_attribution_only_not_shot_partitions")
        is not True
    ):
        raise ValueError("attribution is not detector-partition-owned")
    partition_hash = _partition_content_sha256(evidence)
    if schema.get("partition_content_sha256") != partition_hash:
        raise ValueError("attribution does not match the exact evidence partition")
    attribution_content_hash = schema.get("attribution_content_sha256")
    if not _is_sha256(attribution_content_hash):
        raise ValueError("attribution canonical content hash is missing")
    pair_provenance = _mapping(
        attribution.get("reference_evidence_provenance"),
        "attribution evidence provenance",
    )
    if (
        pair_provenance.get("partition_content_sha256") != partition_hash
        or pair_provenance.get("attribution_content_sha256")
        != attribution_content_hash
    ):
        raise ValueError("attribution evidence provenance is inconsistent")

    samples = _array(evidence.get("samples"), "evidence samples")
    shots = _array(attribution.get("shots"), "attribution shots")
    if schema.get("canonical_shot_count") not in (None, len(shots)):
        raise ValueError("canonical shot count is inconsistent")
    accepted: list[Mapping[str, Any]] = []
    for index, raw in enumerate(shots):
        shot = _mapping(raw, f"attribution shot {index}")
        if shot.get("shot_index") != index:
            raise ValueError("attribution shot indices must be dense and ordered")
        start = _finite(shot.get("reference_start_seconds"), "shot start")
        end = _finite(shot.get("reference_end_seconds"), "shot end")
        if start < 0.0 or end <= start:
            raise ValueError("attribution shot span is invalid")
        if shot.get("accepted") is not True:
            continue
        source_id = str(shot.get("attributed_video_id") or "")
        if source_id not in source_by_id:
            raise ValueError("retained shot refers to an unknown source")
        canonical = _mapping(
            shot.get("canonical_detected_shot"), "canonical detector shot"
        )
        if (
            canonical.get("schema_version") != 1
            or canonical.get("detector_shot_index") != index
            or canonical.get("source_run_merge_does_not_define_shot_partition")
            is not True
        ):
            raise ValueError("retained shot lacks canonical detector ownership")
        indices = _array(canonical.get("sample_indices"), "canonical sample indices")
        if (
            not indices
            or any(isinstance(item, bool) or not isinstance(item, int) for item in indices)
            or indices != sorted(set(indices))
            or indices[-1] >= len(samples)
        ):
            raise ValueError("retained shot canonical sample indices are invalid")
        for sample_index in indices:
            sample = _mapping(samples[sample_index], "canonical evidence sample")
            timestamp = _finite(sample.get("master_seconds"), "sample master time")
            if timestamp < start - 1e-9 or timestamp >= end + 1e-9:
                raise ValueError("retained shot sample lies outside its shot")
        accepted.append(deepcopy(dict(shot)))
    return _ValidatedPair(
        spec=spec,
        attribution=attribution,
        evidence=evidence,
        attribution_file_sha256=attribution_hash,
        evidence_file_sha256=evidence_hash,
        reference_media_path=reference_path,
        source_by_id=source_by_id,
        accepted_shots=tuple(accepted),
        attribution_stat_identity=_stat_identity(spec.attribution_path),
        evidence_stat_identity=_stat_identity(spec.evidence_path),
    )


def _generator_module_sha256() -> str:
    source = getattr(_dynamic_crop_module, "__file__", None)
    if not source:
        raise ReferenceDynamicCropJobError("cannot locate dynamic crop generator source")
    return _file_sha256(Path(source).resolve())


def _caller_provenance(
    pair: _ValidatedPair,
    *,
    config: ReferenceDynamicCropSupervisionConfig,
    generator_module_sha256: str,
) -> dict[str, Any]:
    return {
        "analysis": ANALYSIS_BASIS,
        "pipeline_revision": PIPELINE_REVISION,
        "reference_id": pair.spec.reference_id,
        "canonical_final_attribution": {
            "path": str(pair.spec.attribution_path),
            "file_sha256": pair.attribution_file_sha256,
        },
        "canonical_reference_evidence": {
            "path": str(pair.spec.evidence_path),
            "file_sha256": pair.evidence_file_sha256,
        },
        "configuration": config.to_dict(),
        "generator": {
            "analysis_basis": DYNAMIC_CROP_ANALYSIS_BASIS,
            "target_contract": DYNAMIC_CROP_TARGET_CONTRACT,
            "module_sha256": generator_module_sha256,
        },
        "policy": {
            "accepted_detector_shots_only": True,
            "silent_skip_or_copy_allowed": False,
            "synthetic_time_zero_keyframe_allowed": False,
            "ineligible_tracks_are_persisted_loss_masked": True,
        },
    }


def _artifact_path(
    output_directory: Path,
    *,
    reference_id: str,
    shot_index: int,
    source_id: str,
) -> Path:
    filename = reference_dynamic_crop_v2_filename(
        reference_id, shot_index, source_id
    )
    return output_directory / "artifacts" / reference_id / filename


def _expected_request(
    pair: _ValidatedPair,
    *,
    shot: Mapping[str, Any],
    caller: Mapping[str, Any],
) -> ReferenceDynamicCropSupervisionRequest:
    source_id = str(shot["attributed_video_id"])
    source = pair.source_by_id[source_id]
    return ReferenceDynamicCropSupervisionRequest(
        attributed_shot=shot,
        attribution_path=pair.spec.attribution_path,
        evidence_path=pair.spec.evidence_path,
        reference_media_path=pair.reference_media_path,
        source_media_path=Path(str(source["resolved_path"])),
        source_id=source_id,
        expected_attribution_file_sha256=pair.attribution_file_sha256,
        expected_evidence_file_sha256=pair.evidence_file_sha256,
        caller_provenance=caller,
    )


def _close(left: object, right: object, name: str) -> None:
    if not math.isclose(
        _finite(left, name), _finite(right, name), abs_tol=1e-9, rel_tol=0.0
    ):
        raise ValueError(f"{name} mismatch")


def _validate_expected_artifact(
    payload: Mapping[str, Any],
    *,
    pair: _ValidatedPair,
    shot: Mapping[str, Any],
    config: ReferenceDynamicCropSupervisionConfig,
    caller: Mapping[str, Any],
) -> None:
    validate_reference_dynamic_crop_supervision_payload(
        payload, verify_input_files=False
    )
    if _canonical_json_bytes(payload.get("configuration")) != _canonical_json_bytes(
        config.to_dict()
    ):
        raise ValueError("dynamic crop artifact configuration mismatch")
    source_id = str(shot["attributed_video_id"])
    source_path = Path(str(pair.source_by_id[source_id]["resolved_path"]))
    observed_shot = _mapping(payload.get("shot"), "dynamic crop shot")
    if observed_shot.get("shot_index") != shot.get("shot_index"):
        raise ValueError("dynamic crop shot index mismatch")
    if observed_shot.get("source_id") != source_id:
        raise ValueError("dynamic crop source id mismatch")
    _close(
        observed_shot.get("master_start_seconds"),
        shot.get("reference_start_seconds"),
        "dynamic crop shot start",
    )
    _close(
        observed_shot.get("master_end_seconds"),
        shot.get("reference_end_seconds"),
        "dynamic crop shot end",
    )
    if observed_shot.get("attributed_shot_content_sha256") != _canonical_json_sha256(
        shot
    ):
        raise ValueError("dynamic crop attributed-shot hash mismatch")
    if _canonical_json_bytes(observed_shot.get("canonical_detected_shot")) != (
        _canonical_json_bytes(shot.get("canonical_detected_shot"))
    ):
        raise ValueError("dynamic crop detector-shot identity mismatch")
    provenance = _mapping(payload.get("provenance"), "dynamic crop provenance")
    expected = {
        "attribution": (
            pair.spec.attribution_path,
            "file_sha256",
            pair.attribution_file_sha256,
        ),
        "evidence": (
            pair.spec.evidence_path,
            "file_sha256",
            pair.evidence_file_sha256,
        ),
        "reference_media": (pair.reference_media_path, None, None),
        "source_media": (source_path, None, None),
    }
    for name, (expected_path, digest_key, digest) in expected.items():
        item = _mapping(provenance.get(name), f"dynamic crop {name} provenance")
        observed_path = Path(str(item.get("path") or "")).expanduser()
        if not observed_path.is_absolute() or observed_path.resolve() != expected_path:
            raise ValueError(f"dynamic crop {name} path mismatch")
        if digest_key is not None and item.get(digest_key) != digest:
            raise ValueError(f"dynamic crop {name} JSON hash mismatch")
    if _canonical_json_bytes(provenance.get("caller")) != _canonical_json_bytes(caller):
        raise ValueError("dynamic crop caller provenance mismatch")
    supervision = _mapping(payload.get("supervision"), "dynamic crop supervision")
    if supervision.get("eligible") is True:
        keyframes = _array(
            supervision.get("renderer_keyframes"), "dynamic crop renderer keyframes"
        )
        if not keyframes or _finite(
            _mapping(keyframes[0], "first keyframe").get("time_within_shot_seconds"),
            "first keyframe time",
        ) != 0.0:
            raise ValueError("eligible crop-v2 artifact lacks a real time-zero target")
    target = _mapping(payload.get("target_contract"), "dynamic crop target contract")
    if (
        target.get("real_observed_shot_start_keyframe_required") is not True
        or target.get("interior_keyframes_may_not_be_shifted_to_zero") is not True
    ):
        raise ValueError("dynamic crop artifact permits synthesized time zero")


def _load_existing(
    path: Path,
    *,
    pair: _ValidatedPair,
    shot: Mapping[str, Any],
    config: ReferenceDynamicCropSupervisionConfig,
    caller: Mapping[str, Any],
    verify_media: bool,
) -> ReferenceDynamicCropSupervisionArtifact:
    if path.is_symlink():
        raise ExistingDynamicCropArtifactError(
            f"existing crop-v2 artifact cannot be a symlink: {path}"
        )
    try:
        artifact = load_reference_dynamic_crop_supervision(
            path, verify_input_files=verify_media
        )
        _validate_expected_artifact(
            artifact.payload,
            pair=pair,
            shot=shot,
            config=config,
            caller=caller,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExistingDynamicCropArtifactError(
            f"existing immutable crop-v2 artifact is not identical: {path}: {exc}"
        ) from exc
    return artifact


def _selected_shots(
    pairs: Sequence[_ValidatedPair],
    *,
    reference_ids: Sequence[str],
    shot_indices: Mapping[str, Sequence[int]] | None,
) -> tuple[tuple[_ValidatedPair, Mapping[str, Any]], ...]:
    pair_by_id = {item.spec.reference_id: item for item in pairs}
    if len(pair_by_id) != len(pairs):
        raise ValueError("reference pair ids must be unique")
    if reference_ids:
        normalized = tuple(_clean_reference_id(item) for item in reference_ids)
        if len(set(normalized)) != len(normalized):
            raise ValueError("reference filters must be unique")
        unknown = set(normalized) - set(pair_by_id)
        if unknown:
            raise ValueError(f"unknown reference filters: {sorted(unknown)}")
        selected_reference_ids = set(normalized)
    else:
        selected_reference_ids = set(pair_by_id)
    normalized_indices: dict[str, set[int]] = {}
    for reference_id, values in (shot_indices or {}).items():
        normalized_id = _clean_reference_id(reference_id)
        if normalized_id not in selected_reference_ids:
            raise ValueError(
                f"shot filter reference is not selected: {normalized_id}"
            )
        indices: set[int] = set()
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("shot filters must be non-negative integers")
            if value in indices:
                raise ValueError(f"duplicate shot filter: {normalized_id}:{value}")
            indices.add(value)
        normalized_indices[normalized_id] = indices
    result: list[tuple[_ValidatedPair, Mapping[str, Any]]] = []
    for reference_id in sorted(selected_reference_ids):
        pair = pair_by_id[reference_id]
        retained_by_index = {
            int(shot["shot_index"]): shot for shot in pair.accepted_shots
        }
        requested = normalized_indices.get(reference_id)
        if requested is not None:
            unknown = requested - set(retained_by_index)
            if unknown:
                raise ValueError(
                    "shot filters are not retained canonical shots: "
                    f"{reference_id}:{sorted(unknown)}"
                )
            indices = sorted(requested)
        else:
            indices = sorted(retained_by_index)
        result.extend((pair, retained_by_index[index]) for index in indices)
    return tuple(result)


def _artifact_record(
    *,
    path: Path,
    pair: _ValidatedPair,
    shot: Mapping[str, Any],
    artifact: ReferenceDynamicCropSupervisionArtifact,
) -> dict[str, Any]:
    payload = artifact.payload
    supervision = _mapping(payload.get("supervision"), "dynamic crop supervision")
    sample_rejections: dict[str, int] = {}
    for raw in _array(payload.get("samples"), "dynamic crop samples"):
        sample = _mapping(raw, "dynamic crop sample")
        reason = sample.get("rejection_reason")
        if sample.get("accepted") is False and isinstance(reason, str):
            sample_rejections[reason] = sample_rejections.get(reason, 0) + 1
    provenance = _mapping(payload.get("provenance"), "dynamic crop provenance")
    return {
        "reference_id": pair.spec.reference_id,
        "shot_index": int(shot["shot_index"]),
        "source_id": str(shot["attributed_video_id"]),
        "artifact": {
            "path": str(path.resolve()),
            "file_sha256": _file_sha256(path),
            "content_sha256": provenance["artifact_content_sha256"],
            "build_fingerprint_sha256": provenance["build_fingerprint_sha256"],
        },
        "eligible": bool(supervision["eligible"]),
        "motion_label": supervision.get("motion_label"),
        "motion_label_masked": bool(
            _mapping(
                supervision.get("motion_label_loss_mask"), "motion label mask"
            )["masked"]
        ),
        "keyframe_count": int(supervision["keyframe_count"]),
        "track_rejection_reason": supervision.get("rejection_reason"),
        "sample_rejections_by_reason": dict(sorted(sample_rejections.items())),
        "real_observed_time_zero_required": True,
        "synthetic_time_zero_keyframe_created": False,
    }


def _aggregate_statistics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    track_rejections: dict[str, int] = {}
    sample_rejections: dict[str, int] = {}
    eligible = static = dynamic = keyframes = masked_motion = 0
    references: set[str] = set()
    sources: set[str] = set()
    for record in records:
        references.add(str(record["reference_id"]))
        sources.add(str(record["source_id"]))
        is_eligible = record.get("eligible") is True
        eligible += int(is_eligible)
        label = record.get("motion_label")
        static += int(label == "static")
        dynamic += int(label == "dynamic")
        keyframes += int(record["keyframe_count"])
        masked_motion += int(record.get("motion_label_masked") is True)
        reason = record.get("track_rejection_reason")
        if isinstance(reason, str):
            track_rejections[reason] = track_rejections.get(reason, 0) + 1
        for raw_reason, raw_count in _mapping(
            record.get("sample_rejections_by_reason"), "sample rejections"
        ).items():
            reason_key = str(raw_reason)
            count = int(raw_count)
            sample_rejections[reason_key] = sample_rejections.get(reason_key, 0) + count
    return {
        "reference_count": len(references),
        "source_count": len(sources),
        "artifact_count": len(records),
        "eligible_count": eligible,
        "ineligible_count": len(records) - eligible,
        "static_label_count": static,
        "dynamic_label_count": dynamic,
        "motion_label_masked_count": masked_motion,
        "renderer_keyframe_count": keyframes,
        "track_rejections_by_reason": dict(sorted(track_rejections.items())),
        "sample_rejections_by_reason": dict(sorted(sample_rejections.items())),
        "silent_skip_count": 0,
        "copied_target_count": 0,
        "synthetic_time_zero_keyframe_count": 0,
    }


def _manifest_content_sha256(value: Mapping[str, Any]) -> str:
    normalized = deepcopy(dict(value))
    provenance = _mapping(normalized.get("provenance"), "manifest provenance")
    normalized_provenance = deepcopy(dict(provenance))
    normalized_provenance["manifest_content_sha256"] = None
    normalized["provenance"] = normalized_provenance
    return _canonical_json_sha256(normalized)


def _build_manifest(
    *,
    validated_pairs: Sequence[_ValidatedPair],
    selected: Sequence[tuple[_ValidatedPair, Mapping[str, Any]]],
    records: Sequence[Mapping[str, Any]],
    config: ReferenceDynamicCropSupervisionConfig,
    generator_module_sha256: str,
) -> dict[str, Any]:
    selected_reference_ids = sorted({pair.spec.reference_id for pair, _ in selected})
    pair_by_id = {pair.spec.reference_id: pair for pair in validated_pairs}
    inputs = []
    for reference_id in selected_reference_ids:
        pair = pair_by_id[reference_id]
        inputs.append(
            {
                "reference_id": reference_id,
                "attribution": {
                    "path": str(pair.spec.attribution_path),
                    "file_sha256": pair.attribution_file_sha256,
                },
                "evidence": {
                    "path": str(pair.spec.evidence_path),
                    "file_sha256": pair.evidence_file_sha256,
                },
                "reference_media_path": str(pair.reference_media_path),
                "source_catalog_content_sha256": _canonical_json_sha256(
                    pair.attribution["sources"]
                ),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_basis": ANALYSIS_BASIS,
        "pipeline_revision": PIPELINE_REVISION,
        "namespace": DYNAMIC_CROP_NAMESPACE,
        "target_contract": DYNAMIC_CROP_TARGET_CONTRACT,
        "configuration": config.to_dict(),
        "inputs": inputs,
        "selection": {
            "reference_ids": selected_reference_ids,
            "shot_indices_by_reference": {
                reference_id: sorted(
                    int(shot["shot_index"])
                    for pair, shot in selected
                    if pair.spec.reference_id == reference_id
                )
                for reference_id in selected_reference_ids
            },
            "only_retained_canonical_shots": True,
        },
        "artifacts": [deepcopy(dict(item)) for item in records],
        "statistics": _aggregate_statistics(records),
        "provenance": {
            "content_hash_basis": MANIFEST_HASH_BASIS,
            "manifest_content_sha256": None,
            "generator_module_sha256": generator_module_sha256,
            "immutable_per_shot_artifacts": True,
            "media_hashes_verified_by_each_real_artifact": True,
            "no_silent_skip_copy_or_synthesized_time_zero": True,
        },
    }
    payload["provenance"]["manifest_content_sha256"] = _manifest_content_sha256(
        payload
    )
    validate_reference_dynamic_crop_manifest(payload, verify_artifact_files=False)
    return payload


def validate_reference_dynamic_crop_manifest(
    value: Mapping[str, Any],
    *,
    verify_artifact_files: bool = True,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("dynamic crop manifest must be an object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported dynamic crop manifest schema")
    if value.get("analysis_basis") != ANALYSIS_BASIS:
        raise ValueError("unsupported dynamic crop manifest analysis")
    if value.get("pipeline_revision") != PIPELINE_REVISION:
        raise ValueError("unsupported dynamic crop manifest revision")
    if value.get("namespace") != DYNAMIC_CROP_NAMESPACE:
        raise ValueError("dynamic crop manifest namespace mismatch")
    if value.get("target_contract") != DYNAMIC_CROP_TARGET_CONTRACT:
        raise ValueError("dynamic crop manifest target contract mismatch")
    provenance = _mapping(value.get("provenance"), "manifest provenance")
    content_hash = provenance.get("manifest_content_sha256")
    if (
        provenance.get("content_hash_basis") != MANIFEST_HASH_BASIS
        or not _is_sha256(content_hash)
        or content_hash != _manifest_content_sha256(value)
    ):
        raise ValueError("dynamic crop manifest content hash mismatch")
    if (
        not _is_sha256(provenance.get("generator_module_sha256"))
        or provenance.get("immutable_per_shot_artifacts") is not True
        or provenance.get("media_hashes_verified_by_each_real_artifact") is not True
        or provenance.get("no_silent_skip_copy_or_synthesized_time_zero") is not True
    ):
        raise ValueError("dynamic crop manifest provenance policy is invalid")
    records = _array(value.get("artifacts"), "manifest artifacts")
    identities: list[tuple[str, int]] = []
    verified_input_sha256_cache: dict[str, str] = {}
    for index, raw in enumerate(records):
        record = _mapping(raw, f"manifest artifact {index}")
        identity = (
            _clean_reference_id(str(record.get("reference_id") or "")),
            int(record.get("shot_index")),
        )
        identities.append(identity)
        if record.get("synthetic_time_zero_keyframe_created") is not False:
            raise ValueError("manifest records a synthesized time-zero target")
        artifact = _mapping(record.get("artifact"), "manifest artifact provenance")
        path = Path(str(artifact.get("path") or ""))
        if not path.is_absolute():
            raise ValueError("manifest artifact path must be absolute")
        if not _is_sha256(artifact.get("file_sha256")) or not _is_sha256(
            artifact.get("content_sha256")
        ):
            raise ValueError("manifest artifact hashes are invalid")
        if verify_artifact_files:
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"manifest artifact is unavailable: {path}")
            if _file_sha256(path) != artifact["file_sha256"]:
                raise ValueError(f"manifest artifact file hash mismatch: {path}")
            loaded = load_reference_dynamic_crop_supervision(
                path, verify_input_files=False
            )
            verify_dynamic_crop_input_files(
                loaded.payload,
                verified_sha256_cache=verified_input_sha256_cache,
            )
            if (
                loaded.payload["provenance"]["artifact_content_sha256"]
                != artifact["content_sha256"]
            ):
                raise ValueError("manifest artifact content hash mismatch")
    if identities != sorted(identities) or len(set(identities)) != len(identities):
        raise ValueError("manifest artifacts must be unique and ordered")
    expected_statistics = _aggregate_statistics(records)
    if _canonical_json_bytes(value.get("statistics")) != _canonical_json_bytes(
        expected_statistics
    ):
        raise ValueError("dynamic crop manifest statistics mismatch")


def _publish_manifest(output_directory: Path, payload: Mapping[str, Any]) -> Path:
    content_hash = str(payload["provenance"]["manifest_content_sha256"])
    path = output_directory / f"{MANIFEST_PREFIX}-{content_hash}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(serialized)
            handle.flush()
    except FileExistsError:
        observed = path.read_bytes()
        if observed != serialized:
            raise ExistingDynamicCropArtifactError(
                f"content-addressed manifest path contains different bytes: {path}"
            )
    return path.resolve()


def run_reference_dynamic_crop_job(
    pairs: Sequence[DynamicCropReferencePair],
    *,
    output_directory: Path,
    reference_ids: Sequence[str] = (),
    shot_indices: Mapping[str, Sequence[int]] | None = None,
    config: ReferenceDynamicCropSupervisionConfig | None = None,
    dry_run: bool = False,
) -> DynamicCropJobResult:
    """Plan or materialize exact crop-v2 artifacts for retained shots."""

    if not pairs:
        raise ValueError("at least one explicit attribution/evidence pair is required")
    if any(not isinstance(item, DynamicCropReferencePair) for item in pairs):
        raise TypeError("pairs must contain DynamicCropReferencePair values")
    settings = config or ReferenceDynamicCropSupervisionConfig()
    destination = Path(output_directory).expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(destination)
    validated_pairs = tuple(_validate_pair(item) for item in pairs)
    selected = _selected_shots(
        validated_pairs,
        reference_ids=reference_ids,
        shot_indices=shot_indices,
    )
    generator_hash = _generator_module_sha256()
    plan: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    built = resumed = 0
    for pair, shot in selected:
        source_id = str(shot["attributed_video_id"])
        path = _artifact_path(
            destination,
            reference_id=pair.spec.reference_id,
            shot_index=int(shot["shot_index"]),
            source_id=source_id,
        )
        caller = _caller_provenance(
            pair, config=settings, generator_module_sha256=generator_hash
        )
        if path.exists():
            artifact = _load_existing(
                path,
                pair=pair,
                shot=shot,
                config=settings,
                caller=caller,
                verify_media=not dry_run,
            )
            status = (
                "would_resume_payload_exact_media_unverified"
                if dry_run
                else "resumed_identical"
            )
            resumed += int(not dry_run)
        elif dry_run:
            artifact = None
            status = "would_build"
        else:
            request = _expected_request(pair, shot=shot, caller=caller)
            artifact = _build_reference_dynamic_crop_supervision_job_safe(
                request, config=settings
            )
            _validate_expected_artifact(
                artifact.payload,
                pair=pair,
                shot=shot,
                config=settings,
                caller=caller,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                artifact.write(path)
                built += 1
                status = "built"
            except FileExistsError:
                artifact = _load_existing(
                    path,
                    pair=pair,
                    shot=shot,
                    config=settings,
                    caller=caller,
                    verify_media=True,
                )
                resumed += 1
                status = "resumed_identical_after_publish_race"
        plan.append(
            {
                "reference_id": pair.spec.reference_id,
                "shot_index": int(shot["shot_index"]),
                "source_id": source_id,
                "output_path": str(path),
                "status": status,
            }
        )
        if artifact is not None and not dry_run:
            records.append(
                _artifact_record(
                    path=path,
                    pair=pair,
                    shot=shot,
                    artifact=artifact,
                )
            )
    if dry_run:
        return DynamicCropJobResult(
            dry_run=True,
            plan=tuple(plan),
            built_count=0,
            resumed_count=0,
            manifest_path=None,
            manifest_content_sha256=None,
        )
    for pair in validated_pairs:
        if (
            _stat_identity(pair.spec.attribution_path)
            != pair.attribution_stat_identity
            or _stat_identity(pair.spec.evidence_path) != pair.evidence_stat_identity
        ):
            raise ReferenceDynamicCropJobError(
                "an attribution/evidence input changed during the crop-v2 job"
            )
    if _generator_module_sha256() != generator_hash:
        raise ReferenceDynamicCropJobError(
            "dynamic crop generator changed during the crop-v2 job"
        )
    records.sort(key=lambda item: (str(item["reference_id"]), int(item["shot_index"])))
    manifest = _build_manifest(
        validated_pairs=validated_pairs,
        selected=selected,
        records=records,
        config=settings,
        generator_module_sha256=generator_hash,
    )
    manifest_path = _publish_manifest(destination, manifest)
    validate_reference_dynamic_crop_manifest(manifest, verify_artifact_files=True)
    return DynamicCropJobResult(
        dry_run=False,
        plan=tuple(plan),
        built_count=built,
        resumed_count=resumed,
        manifest_path=manifest_path,
        manifest_content_sha256=str(
            manifest["provenance"]["manifest_content_sha256"]
        ),
    )


def _parse_shot_filters(
    values: Sequence[str], selected_reference_ids: Sequence[str]
) -> dict[str, tuple[int, ...]]:
    result: dict[str, list[int]] = {}
    for raw in values:
        token = str(raw).strip()
        if ":" in token:
            reference_id, index_text = token.rsplit(":", 1)
            reference_id = _clean_reference_id(reference_id)
        else:
            if len(selected_reference_ids) != 1:
                raise ValueError(
                    "a plain --shot-index requires exactly one --reference-id"
                )
            reference_id = _clean_reference_id(selected_reference_ids[0])
            index_text = token
        try:
            index = int(index_text)
        except ValueError as exc:
            raise ValueError(f"invalid shot filter: {raw!r}") from exc
        if index < 0:
            raise ValueError("shot filters must be non-negative")
        result.setdefault(reference_id, []).append(index)
    return {key: tuple(value) for key, value in result.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build immutable real-media crop-v2 artifacts from explicit final "
            "attribution/evidence pairs."
        )
    )
    pair_source = parser.add_mutually_exclusive_group(required=True)
    pair_source.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("REFERENCE_ID", "ATTRIBUTION_JSON", "EVIDENCE_JSON"),
    )
    pair_source.add_argument(
        "--discover-directory",
        type=Path,
        help=(
            "strictly discover exact *-canonical-v2-final.json and matching "
            "*-canonical-v2-boundaries.json pairs"
        ),
    )
    parser.add_argument(
        "--expected-reference-count",
        type=int,
        default=5,
        help="completion gate for --discover-directory (default: 5)",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--reference-id", action="append", default=[])
    parser.add_argument(
        "--shot-index",
        action="append",
        default=[],
        help="REFERENCE_ID:INDEX, or INDEX when exactly one reference is selected",
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        if arguments.discover_directory is not None:
            pairs = discover_reference_dynamic_crop_pairs(
                arguments.discover_directory,
                expected_reference_count=arguments.expected_reference_count,
            )
        else:
            pairs = tuple(
                DynamicCropReferencePair(
                    reference_id, Path(attribution), Path(evidence)
                )
                for reference_id, attribution, evidence in arguments.pair
            )
        selected_ids = tuple(arguments.reference_id)
        if arguments.shot_index and not selected_ids:
            pair_ids = tuple(item.reference_id for item in pairs)
            selected_ids = pair_ids if len(pair_ids) == 1 else ()
        shot_filters = _parse_shot_filters(arguments.shot_index, selected_ids)
        result = run_reference_dynamic_crop_job(
            pairs,
            output_directory=arguments.output_directory,
            reference_ids=tuple(arguments.reference_id),
            shot_indices=shot_filters,
            dry_run=arguments.dry_run,
        )
    except (OSError, TypeError, ValueError, ReferenceDynamicCropJobError) as exc:
        print(f"reference dynamic crop job failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_BASIS",
    "MANIFEST_HASH_BASIS",
    "PIPELINE_REVISION",
    "SCHEMA_VERSION",
    "DynamicCropJobResult",
    "DynamicCropReferencePair",
    "ExistingDynamicCropArtifactError",
    "ReferenceDynamicCropJobError",
    "discover_reference_dynamic_crop_pairs",
    "run_reference_dynamic_crop_job",
    "validate_reference_dynamic_crop_manifest",
]
