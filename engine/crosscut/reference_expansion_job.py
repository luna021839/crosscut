"""Resumable, fail-closed expansion of professional reference datasets.

The job intentionally separates cheap planning from media-heavy execution.
``--dry-run`` validates the self-hashed cadence profile, seed evidence, source
manifest, clocks, provenance, and any already materialized JSON artifacts, but
never hashes or decodes large media.  A real run refuses to start while another
``crosscut_cli.py analyze`` process is active, then verifies every declared
media SHA-256 before processing references sequentially.

Artifacts are immutable once published.  A resumable run skips an existing
artifact only after rebuilding or otherwise fully validating its expected
content.  Invalid existing files are never replaced.  Mutable per-reference
status manifests are written atomically and are not trusted as completion
evidence; the artifacts themselves are always revalidated.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import ctypes
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import errno
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence

from .professional_reference_cadence import (
    PROFILE_ANALYSIS_BASIS,
    PROFILE_SCHEMA_VERSION,
    ProfessionalReferenceCadenceProfile,
)
from .reference_attribution import (
    REFERENCE_ATTRIBUTION_BASIS,
    REFERENCE_CANONICAL_SHOT_SCHEMA_VERSION,
    resolve_reference_attribution_payload,
)
from .reference_boundary_supervision import (
    ANALYSIS_BASIS as REFERENCE_BOUNDARY_ANALYSIS_BASIS,
    augment_reference_boundary_requests,
)
from .reference_dataset import (
    SCHEMA_VERSION as REFERENCE_DATASET_SCHEMA_VERSION,
    build_reference_dataset,
)
from .reference_evidence_builder import (
    ANALYSIS_BASIS as REFERENCE_EVIDENCE_ANALYSIS_BASIS,
    SCHEMA_VERSION as REFERENCE_EVIDENCE_SCHEMA_VERSION,
    ReferenceSourceClock,
    build_reference_evidence,
)
from .reference_supervision import (
    SCHEMA_VERSION as REFERENCE_SUPERVISION_SCHEMA_VERSION,
    build_reference_supervision,
)
from .reference_training_source_pool import (
    validate_reference_training_source_manifest,
)


SCHEMA_VERSION = 1
ANALYSIS_BASIS = "resumable_professional_reference_expansion_canonical_v2"
PIPELINE_REVISION = "canonical_detected_shots_v3_bounded_attribution_samples"
RESERVATION_ANALYSIS_BASIS = "crosscut_immutable_artifact_reservation_v1"
DEFAULT_EXPECTED_REFERENCE_COUNT = 5
DEFAULT_EXPECTED_SOURCE_COUNT = 8
MINIMUM_EXPECTED_SOURCE_COUNT = 8
DEFAULT_MINIMUM_FREE_BYTES = 1024 * 1024 * 1024

_VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_ACTIVE_ANALYSIS_PATTERN = re.compile(
    r"(?:^|[/\s])crosscut_cli\.py\s+analyze(?:\s|$)"
)
_STAGE_NAMES = (
    "evidence",
    "initial_attribution",
    "boundary_evidence",
    "final_attribution",
    "dataset",
    "supervision",
)


class ReferenceExpansionError(RuntimeError):
    """Base exception for fail-closed job failures."""


class ActiveAnalysisError(ReferenceExpansionError):
    """Raised before heavy I/O when another analysis process is active."""


class ExistingArtifactValidationError(ReferenceExpansionError):
    """Raised when an immutable destination exists but is not canonical."""


@dataclass(frozen=True, slots=True)
class MediaIdentity:
    path: Path
    sha256: str
    size_bytes: int
    stat_identity: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class ReferenceSpec:
    video_id: str
    media: MediaIdentity
    audit: MediaIdentity
    audit_cut_count: int
    reference_clock: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SourceSpec:
    video_id: str
    media: MediaIdentity
    clock: Mapping[str, Any]
    provenance: Mapping[str, Any]
    media_probe: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    evidence: Path
    initial_attribution: Path
    boundary_evidence: Path
    final_attribution: Path
    dataset: Path
    supervision: Path
    status_manifest: Path

    def stage_path(self, stage: str) -> Path:
        if stage not in _STAGE_NAMES:
            raise KeyError(stage)
        return Path(getattr(self, stage))

    def to_dict(self) -> dict[str, str]:
        return {
            stage: str(self.stage_path(stage)) for stage in _STAGE_NAMES
        } | {"status_manifest": str(self.status_manifest)}


@dataclass(frozen=True, slots=True)
class ExpansionInputs:
    profile: ProfessionalReferenceCadenceProfile
    profile_payload: Mapping[str, Any]
    profile_path: Path
    profile_file_sha256: str
    seed_evidence: Mapping[str, Any]
    seed_evidence_path: Path
    seed_evidence_file_sha256: str
    source_manifest_path: Path
    source_manifest_file_sha256: str
    references: tuple[ReferenceSpec, ...]
    sources: tuple[SourceSpec, ...]
    seed_reference_id: str
    target_reference_ids: tuple[str, ...]
    style_id: str
    output_directory: Path
    pipeline_code_sha256: Mapping[str, str]
    job_fingerprint: str
    media_sha256_verified: bool
    master_end_seconds: float | None = None

    @property
    def reference_by_id(self) -> dict[str, ReferenceSpec]:
        return {item.video_id: item for item in self.references}

    @property
    def source_catalog(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "video_id": source.video_id,
                "path": str(source.media.path),
                "clock": deepcopy(dict(source.clock)),
                "provenance": deepcopy(dict(source.provenance)),
            }
            for source in self.sources
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_detector_cut_accounting(
    payload: Mapping[str, Any],
    *,
    expected_detector_count: int,
    name: str,
) -> tuple[list[Any], list[Any]]:
    """Bind trainable cuts plus audited out-of-domain cuts to the raw audit."""

    cuts = _array(payload.get("detected_reference_cuts"), f"{name} detected cuts")
    excluded = _array(
        payload.get("excluded_detected_reference_cuts", []),
        f"{name} excluded detected cuts",
    )
    if len(cuts) + len(excluded) != expected_detector_count:
        raise ValueError(f"{name} detector/audit cut count mismatch")
    master_start = _finite(payload.get("master_start_seconds"), f"{name} master start")
    master_end = _finite(payload.get("master_end_seconds"), f"{name} master end")
    if master_end <= master_start:
        raise ValueError(f"{name} master bounds are invalid")
    previous = master_start
    for index, raw in enumerate(cuts):
        cut = _mapping(raw, f"{name} detected cut {index}")
        timestamp = _finite(cut.get("master_seconds"), f"{name} detected cut time")
        if not previous < timestamp < master_end:
            raise ValueError(f"{name} trainable detected cuts are not chronological")
        previous = timestamp
    for index, raw in enumerate(excluded):
        cut = _mapping(raw, f"{name} excluded detected cut {index}")
        timestamp = _finite(
            cut.get("master_seconds"), f"{name} excluded detected cut time"
        )
        reason = cut.get("exclusion_reason")
        if reason == "at_or_before_performance_master_start":
            if timestamp > master_start:
                raise ValueError(f"{name} pre-roll cut lies inside the master domain")
        elif reason == "at_or_after_reference_master_end":
            if timestamp < master_end:
                raise ValueError(f"{name} post-roll cut lies inside the master domain")
        else:
            raise ValueError(f"{name} excluded detected cut reason is unsupported")
    statistics = _mapping(payload.get("statistics"), f"{name} statistics")
    if statistics.get("hard_cut_count") != len(cuts):
        raise ValueError(f"{name} hard_cut_count statistic mismatch")
    if excluded:
        if statistics.get("detector_hard_cut_count") != expected_detector_count:
            raise ValueError(f"{name} detector_hard_cut_count statistic mismatch")
        if statistics.get("excluded_detected_hard_cut_count") != len(excluded):
            raise ValueError(f"{name} excluded hard-cut statistic mismatch")
    return (cuts, excluded)


def _sha256_string(value: object, name: str) -> str:
    result = str(value)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return result


def _resolved_file(value: object, name: str) -> Path:
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path}")
    return path


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _stable_read_bytes(path: Path) -> bytes:
    before = _stat_identity(path)
    raw = path.read_bytes()
    after = _stat_identity(path)
    if before != after or len(raw) != after[2]:
        raise ReferenceExpansionError(f"file changed while being read: {path}")
    return raw


def _stable_read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _stable_read_bytes(path)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload, raw


def _stable_sha256(path: Path) -> str:
    before = _stat_identity(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    after = _stat_identity(path)
    if before != after:
        raise ReferenceExpansionError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _assert_stat_unchanged(media: Sequence[MediaIdentity]) -> None:
    for item in media:
        if _stat_identity(item.path) != item.stat_identity:
            raise ReferenceExpansionError(
                f"validated media changed during the job: {item.path}"
            )


def _active_analysis_processes() -> tuple[dict[str, Any], ...]:
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReferenceExpansionError(
            f"could not verify active analysis processes: {exc}"
        ) from exc
    active: list[dict[str, Any]] = []
    current_pid = os.getpid()
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        pid_text, separator, command = line.partition(" ")
        if not separator or not pid_text.isdigit():
            continue
        pid = int(pid_text)
        command = command.strip()
        if pid != current_pid and _ACTIVE_ANALYSIS_PATTERN.search(command):
            active.append({"pid": pid, "command": command})
    return tuple(active)


def _module_sha256(module_function: Callable[..., Any]) -> tuple[str, str]:
    source = inspect.getsourcefile(module_function)
    if source is None:
        raise ReferenceExpansionError("pipeline module source path is unavailable")
    path = Path(source).resolve()
    return (path.name, _stable_sha256(path))


def _pipeline_code_hashes() -> dict[str, str]:
    pairs = (
        _module_sha256(_pipeline_code_hashes),
        _module_sha256(build_reference_evidence),
        _module_sha256(resolve_reference_attribution_payload),
        _module_sha256(augment_reference_boundary_requests),
        _module_sha256(build_reference_dataset),
        _module_sha256(build_reference_supervision),
    )
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ReferenceExpansionError("pipeline module filenames are not unique")
    return dict(sorted(result.items()))


def _artifact_paths(output_directory: Path, reference_id: str) -> ArtifactPaths:
    if not _VIDEO_ID_PATTERN.fullmatch(reference_id):
        raise ValueError(f"unsafe reference id: {reference_id!r}")
    prefix = output_directory / reference_id
    return ArtifactPaths(
        evidence=Path(f"{prefix}-reference-evidence-canonical-v2.json"),
        initial_attribution=Path(
            f"{prefix}-reference-attribution-canonical-v2-initial.json"
        ),
        boundary_evidence=Path(
            f"{prefix}-reference-evidence-canonical-v2-boundaries.json"
        ),
        final_attribution=Path(
            f"{prefix}-reference-attribution-canonical-v2-final.json"
        ),
        dataset=Path(f"{prefix}-reference-dataset-canonical-v2-final.json"),
        supervision=Path(
            f"{prefix}-reference-supervision-canonical-v2-final.json"
        ),
        status_manifest=Path(
            f"{prefix}-reference-expansion-canonical-v2-status.json"
        ),
    )


def _normalized_clock(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    try:
        return ReferenceSourceClock.from_mapping(value).to_dict()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name}: {exc}") from exc


def _load_profile(
    path: Path,
    *,
    expected_artist: str,
    expected_song: str,
    expected_reference_count: int,
) -> tuple[ProfessionalReferenceCadenceProfile, dict[str, Any], str]:
    # from_file(..., False) still validates the profile's self hash and all
    # derived clocks/clusters; large media hashes are handled once below.
    profile = ProfessionalReferenceCadenceProfile.from_file(
        path,
        verify_input_hashes=False,
    )
    if profile.artist != expected_artist or profile.song != expected_song:
        raise ValueError(
            "cadence song identity mismatch: "
            f"expected {expected_artist!r}/{expected_song!r}, "
            f"got {profile.artist!r}/{profile.song!r}"
        )
    if profile.reference_count != expected_reference_count:
        raise ValueError(
            f"expected {expected_reference_count} cadence references, "
            f"found {profile.reference_count}"
        )
    payload = profile.to_dict()
    if payload.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported cadence profile schema")
    if payload.get("analysis_basis") != PROFILE_ANALYSIS_BASIS:
        raise ValueError("unsupported cadence profile analysis basis")
    raw = _stable_read_bytes(path)
    file_sha256 = _sha256_bytes(raw)
    if profile.file_sha256 != file_sha256:
        raise ReferenceExpansionError("cadence profile changed while loading")
    return (profile, payload, file_sha256)


def _reference_specs(
    profile_payload: Mapping[str, Any],
) -> tuple[ReferenceSpec, ...]:
    provenance = _mapping(profile_payload.get("provenance"), "profile provenance")
    raw_references = _array(provenance.get("references"), "profile references")
    result: list[ReferenceSpec] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_references):
        reference = _mapping(raw, f"profile reference {index}")
        video_id = str(reference.get("video_id") or "")
        if not _VIDEO_ID_PATTERN.fullmatch(video_id) or video_id in seen:
            raise ValueError(f"invalid or duplicate reference id: {video_id!r}")
        seen.add(video_id)
        media_path = _resolved_file(reference.get("media_path"), "reference media")
        audit_path = _resolved_file(reference.get("audit_path"), "reference audit")
        media_hash = _sha256_string(
            reference.get("media_sha256"), "reference media_sha256"
        )
        audit_hash = _sha256_string(
            reference.get("audit_sha256"), "reference audit_sha256"
        )
        raw_clock = _mapping(
            reference.get("local_timewarp"), "reference local_timewarp"
        )
        normalized_clock = _normalized_clock(raw_clock, "reference clock")
        if raw_clock.get("scalar_fallback") is not False:
            raise ValueError(f"reference {video_id} used a scalar-fallback clock")
        result.append(
            ReferenceSpec(
                video_id=video_id,
                media=MediaIdentity(
                    media_path,
                    media_hash,
                    media_path.stat().st_size,
                    _stat_identity(media_path),
                ),
                audit=MediaIdentity(
                    audit_path,
                    audit_hash,
                    audit_path.stat().st_size,
                    _stat_identity(audit_path),
                ),
                audit_cut_count=_positive_int(
                    reference.get("audit_cut_count"), "reference audit_cut_count"
                ),
                reference_clock=normalized_clock,
            )
        )
    return tuple(result)


def _validate_seed_evidence(
    payload: Mapping[str, Any],
    *,
    references: Sequence[ReferenceSpec],
    expected_artist: str,
    expected_song: str,
    expected_source_count: int,
) -> tuple[str, tuple[Mapping[str, Any], ...]]:
    if payload.get("schema_version") != REFERENCE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("seed evidence schema is unsupported")
    if payload.get("analysis") != REFERENCE_EVIDENCE_ANALYSIS_BASIS:
        raise ValueError("seed evidence analysis basis is unsupported")
    reference_path = _resolved_file(payload.get("reference"), "seed reference")
    matching = [
        item for item in references if item.media.path == reference_path
    ]
    if len(matching) != 1:
        raise ValueError("seed evidence reference does not identify one cadence reference")
    seed_reference = matching[0]
    caller = _mapping(
        _mapping(payload.get("provenance"), "seed provenance").get("caller"),
        "seed caller provenance",
    )
    if caller.get("artist") != expected_artist or caller.get("song") != expected_song:
        raise ValueError("seed evidence caller song identity mismatch")
    if caller.get("reference_video_id") != seed_reference.video_id:
        raise ValueError("seed evidence caller reference id mismatch")

    reference_video = _mapping(
        payload.get("reference_video"), "seed reference_video"
    )
    if Path(str(reference_video.get("path"))).expanduser().resolve() != reference_path:
        raise ValueError("seed evidence reference_video path mismatch")
    if _positive_int(reference_video.get("size_bytes"), "seed reference size") != (
        seed_reference.media.size_bytes
    ):
        raise ValueError("seed reference size differs from cadence media")

    clock_policy = _mapping(payload.get("clock_policy"), "seed clock_policy")
    observed_clock = _mapping(
        clock_policy.get("reference_mapping"), "seed reference mapping"
    )
    if _canonical_json_bytes(observed_clock) != _canonical_json_bytes(
        seed_reference.reference_clock
    ):
        raise ValueError("seed reference clock differs from cadence profile")

    raw_sources = _array(payload.get("sources"), "seed sources")
    if len(raw_sources) != expected_source_count:
        raise ValueError(
            f"expected {expected_source_count} seed sources, found {len(raw_sources)}"
        )
    sources: list[Mapping[str, Any]] = []
    seen_sources: set[str] = set()
    for index, raw in enumerate(raw_sources):
        source = _mapping(raw, f"seed source {index}")
        video_id = str(source.get("video_id") or "")
        if not _VIDEO_ID_PATTERN.fullmatch(video_id) or video_id in seen_sources:
            raise ValueError(f"invalid or duplicate seed source id: {video_id!r}")
        seen_sources.add(video_id)
        _resolved_file(source.get("path"), f"seed source {video_id}")
        _normalized_clock(
            _mapping(source.get("clock"), f"seed source {video_id} clock"),
            f"seed source {video_id} clock",
        )
        _mapping(source.get("provenance"), f"seed source {video_id} provenance")
        media_probe = _mapping(
            source.get("media_probe"), f"seed source {video_id} media_probe"
        )
        _positive_int(media_probe.get("size_bytes"), "seed source size")
        _finite(media_probe.get("frame_rate"), "seed source frame_rate")
        _positive_int(media_probe.get("width"), "seed source width")
        _positive_int(media_probe.get("height"), "seed source height")
        sources.append(source)

    cuts, _excluded_cuts = _validate_detector_cut_accounting(
        payload,
        expected_detector_count=seed_reference.audit_cut_count,
        name="seed evidence",
    )
    samples = _array(payload.get("samples"), "seed samples")
    if not samples:
        raise ValueError("seed evidence contains no samples")
    statistics = _mapping(payload.get("statistics"), "seed statistics")
    if statistics.get("source_count") != len(sources):
        raise ValueError("seed source_count statistic mismatch")
    if statistics.get("sample_count") != len(samples):
        raise ValueError("seed sample_count statistic mismatch")
    boundary_requests = _array(
        payload.get("boundary_requests", []), "seed boundary_requests"
    )
    if boundary_requests:
        raise ValueError("seed evidence must be the unaugmented artifact")
    return (seed_reference.video_id, tuple(sources))


def _load_source_specs(
    manifest: Mapping[str, Any],
    *,
    seed_sources: Sequence[Mapping[str, Any]],
    expected_artist: str,
    expected_song: str,
    expected_source_count: int,
) -> tuple[SourceSpec, ...]:
    if manifest.get("schema_version") != 1:
        raise ValueError("source acquisition manifest schema is unsupported")
    if manifest.get("group") != expected_artist or manifest.get("song") != expected_song:
        raise ValueError("source acquisition manifest song identity mismatch")
    training_source_ids = validate_reference_training_source_manifest(
        manifest,
        expected_artist=expected_artist,
        expected_song=expected_song,
        verify_acquisition_binding=True,
    )
    selected = _array(manifest.get("selected"), "source manifest selected")
    if len(selected) != expected_source_count:
        raise ValueError(
            f"expected {expected_source_count} selected sources, found {len(selected)}"
        )
    if len(seed_sources) != expected_source_count:
        raise ValueError("seed/source manifest counts disagree")

    result: list[SourceSpec] = []
    seen: set[str] = set()
    for index, (raw_selected, raw_seed) in enumerate(
        zip(selected, seed_sources, strict=True)
    ):
        selected_source = _mapping(raw_selected, f"selected source {index}")
        seed_source = _mapping(raw_seed, f"seed source {index}")
        video_id = str(selected_source.get("video_id") or "")
        if video_id != seed_source.get("video_id"):
            raise ValueError("seed and manifest source order/identity differ")
        if not _VIDEO_ID_PATTERN.fullmatch(video_id) or video_id in seen:
            raise ValueError(f"invalid or duplicate source id: {video_id!r}")
        seen.add(video_id)
        manifest_path = _resolved_file(
            selected_source.get("local_path"), f"source {video_id} media"
        )
        seed_path = _resolved_file(seed_source.get("path"), f"seed source {video_id}")
        if manifest_path != seed_path:
            raise ValueError(f"source {video_id} path differs between inputs")
        content = _mapping(
            selected_source.get("content_integrity"),
            f"source {video_id} content_integrity",
        )
        if content.get("contract_version") != 1 or content.get("algorithm") != "sha256":
            raise ValueError(f"source {video_id} content integrity is unsupported")
        size_bytes = _positive_int(content.get("size_bytes"), "source size_bytes")
        digest = _sha256_string(content.get("sha256"), "source sha256")
        if manifest_path.stat().st_size != size_bytes:
            raise ValueError(f"source {video_id} current size mismatch")
        seed_probe = _mapping(
            seed_source.get("media_probe"), f"source {video_id} seed media_probe"
        )
        if seed_probe.get("size_bytes") != size_bytes:
            raise ValueError(f"source {video_id} seed size mismatch")
        validation = _mapping(
            selected_source.get("media_validation"),
            f"source {video_id} media_validation",
        )
        if validation.get("passed") is not True:
            raise ValueError(f"source {video_id} failed acquisition media QA")
        production_eligible = selected_source.get("group_quality_eligible") is True
        training_eligible = (
            bool(training_source_ids)
            and video_id in training_source_ids
            and selected_source.get("training_attribution_eligible") is True
        )
        if not production_eligible and not training_eligible:
            raise ValueError(f"source {video_id} is not group-quality eligible")
        clock = _normalized_clock(
            _mapping(seed_source.get("clock"), f"source {video_id} clock"),
            f"source {video_id} clock",
        )
        provenance = deepcopy(
            dict(
                _mapping(
                    seed_source.get("provenance"),
                    f"source {video_id} provenance",
                )
            )
        )
        result.append(
            SourceSpec(
                video_id=video_id,
                media=MediaIdentity(
                    manifest_path,
                    digest,
                    size_bytes,
                    _stat_identity(manifest_path),
                ),
                clock=clock,
                provenance=provenance,
                media_probe=deepcopy(dict(seed_probe)),
            )
        )
    return tuple(result)


def _verify_master_binding(
    profile_payload: Mapping[str, Any], sources: Sequence[SourceSpec]
) -> MediaIdentity:
    provenance = _mapping(profile_payload.get("provenance"), "profile provenance")
    master = _mapping(provenance.get("master"), "profile master")
    master_path = _resolved_file(master.get("media_path"), "profile master media")
    master_hash = _sha256_string(master.get("media_sha256"), "master media_sha256")
    matches = [source for source in sources if source.media.path == master_path]
    if len(matches) != 1:
        raise ValueError("cadence master is not exactly one selected source")
    if matches[0].media.sha256 != master_hash:
        raise ValueError("cadence master hash differs from source manifest")
    return matches[0].media


def _profile_master_end_seconds(profile_payload: Mapping[str, Any]) -> float:
    provenance = _mapping(profile_payload.get("provenance"), "profile provenance")
    master = _mapping(provenance.get("master"), "profile master")
    duration = _finite(
        master.get("decoded_audio_duration_seconds"),
        "profile master decoded audio duration",
    )
    if duration <= 0.0:
        raise ValueError("profile master decoded audio duration must be positive")
    return duration


def _verify_live_media_hashes(
    *,
    references: Sequence[ReferenceSpec],
    sources: Sequence[SourceSpec],
    master: MediaIdentity,
) -> None:
    expected: dict[Path, str] = {master.path: master.sha256}
    for reference in references:
        for media in (reference.media, reference.audit):
            previous = expected.setdefault(media.path, media.sha256)
            if previous != media.sha256:
                raise ValueError(f"conflicting expected hash for {media.path}")
    for source in sources:
        previous = expected.setdefault(source.media.path, source.media.sha256)
        if previous != source.media.sha256:
            raise ValueError(f"conflicting expected hash for {source.media.path}")
    for path, expected_hash in sorted(expected.items(), key=lambda item: str(item[0])):
        active = _active_analysis_processes()
        if active:
            raise ActiveAnalysisError(
                f"crosscut analysis started before media hashing: {active}"
            )
        observed = _stable_sha256(path)
        active = _active_analysis_processes()
        if active:
            raise ActiveAnalysisError(
                f"crosscut analysis started during media hashing: {active}"
            )
        if observed != expected_hash:
            raise ValueError(f"media SHA-256 mismatch: {path}")


def _load_inputs(
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
) -> ExpansionInputs:
    profile_path = cadence_profile_path.expanduser().resolve()
    seed_path = seed_evidence_path.expanduser().resolve()
    manifest_path = source_manifest_path.expanduser().resolve()
    output = output_directory.expanduser().resolve()
    if not profile_path.is_file() or not seed_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("cadence profile, seed evidence, and source manifest are required")
    if not output.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output}")
    if not style_id.strip():
        raise ValueError("style_id cannot be blank")

    profile, profile_payload, profile_file_hash = _load_profile(
        profile_path,
        expected_artist=expected_artist,
        expected_song=expected_song,
        expected_reference_count=expected_reference_count,
    )
    references = _reference_specs(profile_payload)
    seed_payload, seed_raw = _stable_read_json(seed_path)
    seed_reference_id, seed_sources = _validate_seed_evidence(
        seed_payload,
        references=references,
        expected_artist=expected_artist,
        expected_song=expected_song,
        expected_source_count=expected_source_count,
    )
    manifest_payload, manifest_raw = _stable_read_json(manifest_path)
    sources = _load_source_specs(
        manifest_payload,
        seed_sources=seed_sources,
        expected_artist=expected_artist,
        expected_song=expected_song,
        expected_source_count=expected_source_count,
    )
    master = _verify_master_binding(profile_payload, sources)
    master_end_seconds = _profile_master_end_seconds(profile_payload)

    known_reference_ids = {item.video_id for item in references}
    if target_reference_ids:
        targets = tuple(target_reference_ids)
        if len(set(targets)) != len(targets):
            raise ValueError("target reference ids must be unique")
        unknown = set(targets) - known_reference_ids
        if unknown:
            raise ValueError(f"unknown target reference ids: {sorted(unknown)}")
        if seed_reference_id in targets:
            raise ValueError("the seed reference cannot be an expansion target")
    else:
        targets = tuple(
            item.video_id for item in references if item.video_id != seed_reference_id
        )
    if len(targets) != expected_reference_count - 1 and not target_reference_ids:
        raise ValueError("default target set is not every non-seed reference")

    if verify_media_hashes:
        _verify_live_media_hashes(
            references=references,
            sources=sources,
            master=master,
        )

    code_hashes = _pipeline_code_hashes()
    seed_hash = _sha256_bytes(seed_raw)
    manifest_hash = _sha256_bytes(manifest_raw)
    fingerprint_payload = {
        "analysis": ANALYSIS_BASIS,
        "pipeline_revision": PIPELINE_REVISION,
        "profile_file_sha256": profile_file_hash,
        "profile_content_sha256": profile.profile_content_sha256,
        "seed_evidence_file_sha256": seed_hash,
        "source_manifest_file_sha256": manifest_hash,
        "pipeline_code_sha256": code_hashes,
        "style_id": style_id,
    }
    return ExpansionInputs(
        profile=profile,
        profile_payload=profile_payload,
        profile_path=profile_path,
        profile_file_sha256=profile_file_hash,
        seed_evidence=seed_payload,
        seed_evidence_path=seed_path,
        seed_evidence_file_sha256=seed_hash,
        source_manifest_path=manifest_path,
        source_manifest_file_sha256=manifest_hash,
        references=references,
        sources=sources,
        seed_reference_id=seed_reference_id,
        target_reference_ids=targets,
        style_id=style_id,
        output_directory=output,
        pipeline_code_sha256=code_hashes,
        job_fingerprint=_canonical_json_sha256(fingerprint_payload),
        media_sha256_verified=verify_media_hashes,
        master_end_seconds=master_end_seconds,
    )


def _caller_provenance(inputs: ExpansionInputs, reference_id: str) -> dict[str, Any]:
    return {
        "artist": inputs.profile.artist,
        "song": inputs.profile.song,
        "reference_video_id": reference_id,
        "purpose": "same_song_professional_reference_learning",
        "pipeline_revision": PIPELINE_REVISION,
        "cadence_profile": {
            "path": str(inputs.profile_path),
            "file_sha256": inputs.profile_file_sha256,
            "content_sha256": inputs.profile.profile_content_sha256,
        },
        "seed_evidence": {
            "path": str(inputs.seed_evidence_path),
            "file_sha256": inputs.seed_evidence_file_sha256,
        },
        "source_manifest": {
            "path": str(inputs.source_manifest_path),
            "file_sha256": inputs.source_manifest_file_sha256,
        },
        "pipeline_code_sha256": deepcopy(dict(inputs.pipeline_code_sha256)),
        "job_fingerprint": inputs.job_fingerprint,
    }


def _assert_canonical_json_equal(
    observed: Mapping[str, Any], expected: Mapping[str, Any], name: str
) -> None:
    observed_hash = _canonical_json_sha256(observed)
    expected_hash = _canonical_json_sha256(expected)
    if observed_hash != expected_hash:
        raise ExistingArtifactValidationError(
            f"{name} canonical content mismatch; observed={observed_hash}, "
            f"expected={expected_hash}"
        )


def _validate_generated_evidence(
    payload: Mapping[str, Any],
    *,
    inputs: ExpansionInputs,
    reference: ReferenceSpec,
) -> None:
    if payload.get("schema_version") != REFERENCE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("reference evidence schema mismatch")
    if payload.get("analysis") != REFERENCE_EVIDENCE_ANALYSIS_BASIS:
        raise ValueError("reference evidence analysis mismatch")
    if Path(str(payload.get("reference"))).expanduser().resolve() != reference.media.path:
        raise ValueError("reference evidence media path mismatch")
    if _finite(payload.get("master_start_seconds"), "master_start_seconds") != 0.0:
        raise ValueError("reference evidence must start at master zero")
    if _finite(payload.get("master_end_seconds"), "master_end_seconds") <= 0.0:
        raise ValueError("reference evidence master range is empty")

    reference_video = _mapping(payload.get("reference_video"), "reference_video")
    if Path(str(reference_video.get("path"))).expanduser().resolve() != reference.media.path:
        raise ValueError("reference_video path mismatch")
    if reference_video.get("size_bytes") != reference.media.size_bytes:
        raise ValueError("reference_video size mismatch")
    for key in ("width", "height", "declared_frame_count"):
        _positive_int(reference_video.get(key), f"reference_video {key}")
    if _finite(reference_video.get("frame_rate"), "reference frame_rate") <= 0.0:
        raise ValueError("reference frame rate must be positive")

    clock_policy = _mapping(payload.get("clock_policy"), "clock_policy")
    observed_clock = _mapping(
        clock_policy.get("reference_mapping"), "reference mapping"
    )
    if _canonical_json_bytes(observed_clock) != _canonical_json_bytes(
        reference.reference_clock
    ):
        raise ValueError("reference clock does not match cadence profile")
    if clock_policy.get("decode_policy") != "one_forward_only_union_decode_per_source":
        raise ValueError("reference evidence decode policy is unsupported")

    evidence_provenance = _mapping(
        payload.get("provenance"), "evidence provenance"
    )
    _validated_timestamp(
        evidence_provenance.get("created_at"), "evidence created_at"
    )
    caller = _mapping(
        evidence_provenance.get("caller"),
        "evidence caller",
    )
    _assert_canonical_json_equal(
        caller,
        _caller_provenance(inputs, reference.video_id),
        "evidence caller provenance",
    )

    raw_sources = _array(payload.get("sources"), "evidence sources")
    if len(raw_sources) != len(inputs.sources):
        raise ValueError("reference evidence source count mismatch")
    expected_ids = [source.video_id for source in inputs.sources]
    for index, (raw, expected) in enumerate(zip(raw_sources, inputs.sources, strict=True)):
        source = _mapping(raw, f"evidence source {index}")
        if source.get("video_id") != expected.video_id:
            raise ValueError("reference evidence source order mismatch")
        if Path(str(source.get("path"))).expanduser().resolve() != expected.media.path:
            raise ValueError(f"source {expected.video_id} path mismatch")
        observed_clock = _mapping(source.get("clock"), "evidence source clock")
        if _canonical_json_bytes(observed_clock) != _canonical_json_bytes(expected.clock):
            raise ValueError(f"source {expected.video_id} clock mismatch")
        observed_provenance = _mapping(
            source.get("provenance"), "evidence source provenance"
        )
        if _canonical_json_bytes(observed_provenance) != _canonical_json_bytes(
            expected.provenance
        ):
            raise ValueError(f"source {expected.video_id} provenance mismatch")
        probe = _mapping(source.get("media_probe"), "evidence source media_probe")
        if probe.get("size_bytes") != expected.media.size_bytes:
            raise ValueError(f"source {expected.video_id} media size mismatch")
        for key in ("width", "height", "declared_frame_count"):
            _positive_int(probe.get(key), f"source {expected.video_id} {key}")
        if _finite(probe.get("frame_rate"), "source frame_rate") <= 0.0:
            raise ValueError("source frame rate must be positive")

    cuts, _excluded_cuts = _validate_detector_cut_accounting(
        payload,
        expected_detector_count=reference.audit_cut_count,
        name=f"reference {reference.video_id}",
    )
    samples = _array(payload.get("samples"), "evidence samples")
    if not samples:
        raise ValueError("reference evidence contains no samples")
    previous_master = -math.inf
    for sample_index, raw_sample in enumerate(samples):
        sample = _mapping(raw_sample, f"evidence sample {sample_index}")
        master_seconds = _finite(sample.get("master_seconds"), "sample master_seconds")
        if (
            inputs.master_end_seconds is not None
            and master_seconds > inputs.master_end_seconds + 1e-9
        ):
            raise ValueError("reference attribution sample exceeds cadence master domain")
        if master_seconds < previous_master:
            raise ValueError("evidence samples are not chronological")
        previous_master = master_seconds
        ranked = _array(sample.get("ranked_candidates"), "ranked_candidates")
        candidate_ids = [
            str(_mapping(item, "ranked candidate").get("video_id"))
            for item in ranked
        ]
        if len(candidate_ids) != len(expected_ids) or set(candidate_ids) != set(expected_ids):
            raise ValueError("evidence sample does not contain every source candidate")
    statistics = _mapping(payload.get("statistics"), "evidence statistics")
    if statistics.get("source_count") != len(inputs.sources):
        raise ValueError("evidence source_count statistic mismatch")
    if statistics.get("sample_count") != len(samples):
        raise ValueError("evidence sample_count statistic mismatch")
    if _array(payload.get("boundary_requests", []), "boundary_requests"):
        raise ValueError("base reference evidence is already augmented")


def _validate_canonical_attribution(
    payload: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("reference attribution schema mismatch")
    if payload.get("analysis") != REFERENCE_ATTRIBUTION_BASIS:
        raise ValueError("reference attribution analysis mismatch")
    schema = _mapping(payload.get("canonical_shot_schema"), "canonical_shot_schema")
    if schema.get("schema_version") != REFERENCE_CANONICAL_SHOT_SCHEMA_VERSION:
        raise ValueError("canonical shot schema mismatch")
    if schema.get("partition_basis") != "reference_hard_cut_detector":
        raise ValueError("canonical shot partition basis mismatch")
    cuts = _array(evidence.get("detected_reference_cuts"), "evidence cuts")
    shots = _array(payload.get("shots"), "attribution shots")
    boundaries = _array(payload.get("detected_boundaries"), "detected_boundaries")
    if schema.get("detected_cut_count") != len(cuts):
        raise ValueError("canonical detected-cut count mismatch")
    if schema.get("canonical_shot_count") != len(cuts) + 1:
        raise ValueError("canonical shot count is not cut_count + 1")
    if len(shots) != len(cuts) + 1 or len(boundaries) != len(cuts):
        raise ValueError("canonical shot/boundary arrays do not preserve all cuts")
    if schema.get("source_runs_are_attribution_only_not_shot_partitions") is not True:
        raise ValueError("source runs were allowed to define shot partitions")
    if schema.get("same_source_adjacent_detector_shots_are_never_merged") is not True:
        raise ValueError("same-source detector shots may have been merged")
    _sha256_string(schema.get("partition_content_sha256"), "partition hash")
    _sha256_string(schema.get("attribution_content_sha256"), "attribution hash")
    provenance = _mapping(
        payload.get("reference_evidence_provenance"),
        "reference_evidence_provenance",
    )
    if provenance.get("analysis") != evidence.get("analysis"):
        raise ValueError("attribution/evidence analysis provenance mismatch")
    if _canonical_json_bytes(provenance.get("provenance")) != _canonical_json_bytes(
        evidence.get("provenance")
    ):
        raise ValueError("attribution/evidence caller provenance mismatch")


def _normalize_boundary_timestamp(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(dict(payload))
    augmentation = _mapping(
        normalized.get("boundary_request_augmentation"),
        "boundary_request_augmentation",
    )
    created_at = augmentation.get("created_at")
    _validated_timestamp(created_at, "boundary augmentation created_at")
    mutable = dict(augmentation)
    mutable["created_at"] = "<validated-utc-timestamp>"
    normalized["boundary_request_augmentation"] = mutable
    return normalized


def _validated_timestamp(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} lacks a timezone")
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{name} is not UTC")


def _normalize_dataset_timestamp(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(dict(payload))
    if normalized.get("schema_version") != REFERENCE_DATASET_SCHEMA_VERSION:
        raise ValueError("reference dataset schema mismatch")
    if normalized.get("dataset_kind") != "reference_stage_mix_source_ranking":
        raise ValueError("reference dataset kind mismatch")
    _validated_timestamp(normalized.get("created_at"), "dataset created_at")
    normalized["created_at"] = "<validated-utc-timestamp>"
    return normalized


def _validate_dataset_payload(
    payload: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    _assert_canonical_json_equal(
        _normalize_dataset_timestamp(payload),
        _normalize_dataset_timestamp(expected),
        "reference dataset",
    )


def _validate_boundary_evidence(
    payload: Mapping[str, Any],
    *,
    base_evidence: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    augmentation = _mapping(
        payload.get("boundary_request_augmentation"),
        "boundary_request_augmentation",
    )
    if augmentation.get("analysis") != REFERENCE_BOUNDARY_ANALYSIS_BASIS:
        raise ValueError("boundary augmentation analysis mismatch")
    if payload.get("analysis") != base_evidence.get("analysis"):
        raise ValueError("boundary evidence changed its base analysis")
    _assert_canonical_json_equal(
        _normalize_boundary_timestamp(payload),
        _normalize_boundary_timestamp(expected),
        "boundary evidence",
    )


def _write_json_temporary(destination: Path, payload: Mapping[str, Any]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.crosscut-reference-expansion-",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _publish_json_no_replace(destination: Path, payload: Mapping[str, Any]) -> Path:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {destination}")
    temporary = _write_json_temporary(destination, payload)
    try:
        # The job holds an exclusive directory-level advisory lock.  Recheck
        # immediately before the atomic rename so a pre-existing artifact is
        # never replaced during normal or resumed execution.
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite immutable artifact: {destination}"
            )
        _rename_no_replace(temporary, destination)
        return destination.resolve()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish one same-filesystem file without replacement.

    Darwin exposes ``renamex_np(RENAME_EXCL)`` for exactly this operation.  On
    other POSIX systems a hard-link publication is also atomic and no-clobber.
    Unsupported filesystems fail closed instead of falling back to a racy
    check-then-overwrite rename.
    """

    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        renamex = libc.renamex_np
        renamex.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex.restype = ctypes.c_int
        rename_excl = 0x00000004
        result = renamex(
            os.fsencode(source),
            os.fsencode(destination),
            rename_excl,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(
                error_number,
                os.strerror(error_number),
                destination,
            )
        if error_number not in (errno.ENOTSUP, errno.EOPNOTSUPP):
            raise OSError(error_number, os.strerror(error_number), destination)
        _reserved_replace_no_clobber(source, destination)
        return

    try:
        os.link(source, destination)
    except FileExistsError:
        raise
    except OSError as exc:
        raise ReferenceExpansionError(
            "filesystem does not support atomic no-replace publication: "
            f"{destination}: {exc}"
        ) from exc
    source.unlink()


def _reservation_document(source: Path, destination: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "analysis": RESERVATION_ANALYSIS_BASIS,
        "destination_name": destination.name,
        "temporary_name": source.name,
        "temporary_file_sha256": _stable_sha256(source),
        "created_at": _utc_now(),
        "semantics": "exclusive_destination_reservation_not_a_complete_artifact",
    }


def _reserved_replace_no_clobber(source: Path, destination: Path) -> None:
    """Use an exclusive, recoverable reservation on filesystems such as ExFAT."""

    reservation = _reservation_document(source, destination)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(destination, flags, 0o600)
    reservation_identity: tuple[int, int, int, int] | None = None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                reservation,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        reservation_identity = _stat_identity(destination)
        observed, _raw = _stable_read_json(destination)
        if _canonical_json_bytes(observed) != _canonical_json_bytes(reservation):
            raise ReferenceExpansionError(
                f"artifact reservation changed before publication: {destination}"
            )
        if _stat_identity(destination) != reservation_identity:
            raise ReferenceExpansionError(
                f"artifact reservation identity changed: {destination}"
            )
        # This replaces only the exact reservation that this process created
        # with O_EXCL.  If the process dies first, the validated temp and
        # reservation are recovered by _materialize_or_skip on the next run.
        os.replace(source, destination)
    except BaseException:
        try:
            if (
                reservation_identity is not None
                and destination.exists()
                and not destination.is_symlink()
                and _stat_identity(destination) == reservation_identity
            ):
                destination.unlink()
        finally:
            raise


def _reservation_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.is_symlink() or not path.is_file():
        return None
    try:
        payload, _raw = _stable_read_json(path)
    except (OSError, ValueError, ReferenceExpansionError):
        return None
    if payload.get("analysis") != RESERVATION_ANALYSIS_BASIS:
        return None
    expected_keys = {
        "schema_version",
        "analysis",
        "destination_name",
        "temporary_name",
        "temporary_file_sha256",
        "created_at",
        "semantics",
    }
    if set(payload) != expected_keys or payload.get("schema_version") != 1:
        raise ExistingArtifactValidationError(
            f"malformed artifact reservation: {path}"
        )
    if payload.get("destination_name") != path.name:
        raise ExistingArtifactValidationError(
            f"artifact reservation destination mismatch: {path}"
        )
    if payload.get("semantics") != (
        "exclusive_destination_reservation_not_a_complete_artifact"
    ):
        raise ExistingArtifactValidationError(
            f"artifact reservation semantics mismatch: {path}"
        )
    _sha256_string(
        payload.get("temporary_file_sha256"),
        "reservation temporary_file_sha256",
    )
    _validated_timestamp(payload.get("created_at"), "reservation created_at")
    return payload


def _recover_reserved_artifact(
    destination: Path,
    *,
    validator: Callable[[Mapping[str, Any]], None],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    reservation = _reservation_payload(destination)
    if reservation is None:
        return None
    temporary_name = str(reservation["temporary_name"])
    expected_prefix = f".{destination.name}.crosscut-reference-expansion-"
    if (
        Path(temporary_name).name != temporary_name
        or not temporary_name.startswith(expected_prefix)
        or not temporary_name.endswith(".tmp")
    ):
        raise ExistingArtifactValidationError(
            f"artifact reservation temporary name is unsafe: {destination}"
        )
    temporary = destination.parent / temporary_name
    if not temporary.exists():
        # Only a self-identifying job reservation is removed.  An ordinary or
        # malformed existing artifact is never treated as cleanup material.
        destination.unlink()
        return None
    if temporary.is_symlink() or not stat.S_ISREG(os.lstat(temporary).st_mode):
        raise ExistingArtifactValidationError(
            f"reserved artifact temporary is not a regular file: {temporary}"
        )
    expected_hash = str(reservation["temporary_file_sha256"])
    if _stable_sha256(temporary) != expected_hash:
        raise ExistingArtifactValidationError(
            f"reserved artifact temporary hash mismatch: {temporary}"
        )
    payload, _raw = _stable_read_json(temporary)
    validator(payload)
    os.replace(temporary, destination)
    observed, raw = _stable_read_json(destination)
    _assert_canonical_json_equal(observed, payload, destination.name)
    return (
        observed,
        {
            "action": "recovered_valid_job_temporary",
            "path": str(destination.resolve()),
            "sha256": _sha256_bytes(raw),
        },
    )


def _atomic_update_json(destination: Path, payload: Mapping[str, Any]) -> Path:
    if destination.is_symlink():
        raise ReferenceExpansionError(
            f"refusing to replace symlink status manifest: {destination}"
        )
    if destination.exists() and not stat.S_ISREG(os.lstat(destination).st_mode):
        raise ReferenceExpansionError(
            f"status manifest is not a regular file: {destination}"
        )
    temporary = _write_json_temporary(destination, payload)
    try:
        os.replace(temporary, destination)
        return destination.resolve()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _cleanup_stale_temps(destination: Path) -> int:
    prefix = f".{destination.name}.crosscut-reference-expansion-"
    count = 0
    for candidate in destination.parent.glob(f"{prefix}*.tmp"):
        mode = os.lstat(candidate).st_mode
        if not stat.S_ISREG(mode):
            raise ReferenceExpansionError(
                f"refusing to clean non-regular job temporary: {candidate}"
            )
        candidate.unlink()
        count += 1
    return count


@contextmanager
def _job_lock(output_directory: Path) -> Iterator[None]:
    lock_path = output_directory / ".reference-expansion-canonical-v2.lock"
    if lock_path.is_symlink():
        raise ReferenceExpansionError(f"job lock is a symlink: {lock_path}")
    if lock_path.exists() and not stat.S_ISREG(os.lstat(lock_path).st_mode):
        raise ReferenceExpansionError(f"job lock is not a regular file: {lock_path}")
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ReferenceExpansionError(
                f"another reference-expansion job holds {lock_path}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        json.dump(
            {"pid": os.getpid(), "acquired_at": _utc_now()},
            handle,
            ensure_ascii=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_existing_artifact(
    path: Path,
    *,
    validator: Callable[[Mapping[str, Any]], None],
) -> tuple[dict[str, Any], str]:
    if path.is_symlink():
        raise ExistingArtifactValidationError(
            f"immutable artifact cannot be a symlink: {path}"
        )
    if not path.is_file():
        raise ExistingArtifactValidationError(
            f"immutable artifact is not a regular file: {path}"
        )
    try:
        payload, raw = _stable_read_json(path)
        validator(payload)
    except Exception as exc:
        raise ExistingArtifactValidationError(
            f"existing artifact failed validation and will not be overwritten: "
            f"{path}: {exc}"
        ) from exc
    return (payload, _sha256_bytes(raw))


def _materialize_or_skip(
    path: Path,
    *,
    producer: Callable[[], Mapping[str, Any]],
    validator: Callable[[Mapping[str, Any]], None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    recovered = _recover_reserved_artifact(path, validator=validator)
    if recovered is not None:
        return recovered
    if path.exists() or path.is_symlink():
        payload, digest = _load_existing_artifact(path, validator=validator)
        return (
            payload,
            {
                "action": "skipped_valid_existing",
                "path": str(path.resolve()),
                "sha256": digest,
            },
        )
    _cleanup_stale_temps(path)
    payload = deepcopy(dict(producer()))
    validator(payload)
    _publish_json_no_replace(path, payload)
    observed, raw = _stable_read_json(path)
    _assert_canonical_json_equal(observed, payload, path.name)
    return (
        observed,
        {
            "action": "created",
            "path": str(path.resolve()),
            "sha256": _sha256_bytes(raw),
        },
    )


def _status_base(
    inputs: ExpansionInputs,
    reference: ReferenceSpec,
    paths: ArtifactPaths,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_BASIS,
        "pipeline_revision": PIPELINE_REVISION,
        "job_fingerprint": inputs.job_fingerprint,
        "reference_id": reference.video_id,
        "reference_media": {
            "path": str(reference.media.path),
            "sha256": reference.media.sha256,
            "audit_path": str(reference.audit.path),
            "audit_sha256": reference.audit.sha256,
            "audit_cut_count": reference.audit_cut_count,
        },
        "inputs": {
            "cadence_profile": str(inputs.profile_path),
            "cadence_profile_file_sha256": inputs.profile_file_sha256,
            "cadence_profile_content_sha256": inputs.profile.profile_content_sha256,
            "seed_evidence": str(inputs.seed_evidence_path),
            "seed_evidence_file_sha256": inputs.seed_evidence_file_sha256,
            "source_manifest": str(inputs.source_manifest_path),
            "source_manifest_file_sha256": inputs.source_manifest_file_sha256,
            "source_count": len(inputs.sources),
            "media_sha256_verified": inputs.media_sha256_verified,
            "pipeline_code_sha256": deepcopy(dict(inputs.pipeline_code_sha256)),
        },
        "style_id": inputs.style_id,
        "artifacts": paths.to_dict(),
        "state": "running",
        "stages": {
            stage: {"status": "pending", "path": str(paths.stage_path(stage))}
            for stage in _STAGE_NAMES
        },
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "completed_at": None,
        "failure": None,
    }


def _load_previous_status(path: Path, inputs: ExpansionInputs, reference_id: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload, _raw = _stable_read_json(path)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("analysis") != ANALYSIS_BASIS
        or payload.get("job_fingerprint") != inputs.job_fingerprint
        or payload.get("reference_id") != reference_id
    ):
        raise ExistingArtifactValidationError(
            f"status manifest belongs to another or invalid job: {path}"
        )
    return payload


def _write_status(path: Path, payload: Mapping[str, Any]) -> None:
    _cleanup_stale_temps(path)
    _atomic_update_json(path, payload)


def _update_stage_status(
    status: dict[str, Any], stage: str, result: Mapping[str, Any]
) -> None:
    stages = _mapping(status.get("stages"), "status stages")
    mutable = dict(stages)
    mutable[stage] = {
        "status": "complete",
        **deepcopy(dict(result)),
        "completed_at": _utc_now(),
    }
    status["stages"] = mutable
    status["updated_at"] = _utc_now()


def _reference_media(inputs: ExpansionInputs, reference: ReferenceSpec) -> tuple[MediaIdentity, ...]:
    return (
        reference.media,
        reference.audit,
        *(source.media for source in inputs.sources),
    )


def _run_one_reference(
    inputs: ExpansionInputs,
    reference: ReferenceSpec,
    *,
    emit: Callable[[Mapping[str, Any]], None],
) -> dict[str, Any]:
    paths = _artifact_paths(inputs.output_directory, reference.video_id)
    previous = _load_previous_status(paths.status_manifest, inputs, reference.video_id)
    status = _status_base(inputs, reference, paths)
    if previous is not None:
        status["started_at"] = previous.get("started_at", status["started_at"])
    _write_status(paths.status_manifest, status)

    def finish_stage(stage: str, result: Mapping[str, Any]) -> None:
        _update_stage_status(status, stage, result)
        _write_status(paths.status_manifest, status)
        emit(
            {
                "reference_id": reference.video_id,
                "stage": stage,
                "status": "complete",
                **dict(result),
            }
        )

    try:
        active = _active_analysis_processes()
        if active:
            raise ActiveAnalysisError(
                f"active crosscut analysis blocks media work: {active}"
            )
        media = _reference_media(inputs, reference)
        _assert_stat_unchanged(media)

        def produce_evidence() -> Mapping[str, Any]:
            before = _active_analysis_processes()
            if before:
                raise ActiveAnalysisError(
                    f"active crosscut analysis blocks evidence build: {before}"
                )
            _assert_stat_unchanged(media)
            built = build_reference_evidence(
                reference.media.path,
                inputs.source_catalog,
                master_start_seconds=0.0,
                master_end_seconds=inputs.master_end_seconds,
                reference_clock=reference.reference_clock,
                provenance=_caller_provenance(inputs, reference.video_id),
            ).payload
            after = _active_analysis_processes()
            if after:
                raise ActiveAnalysisError(
                    "crosscut analysis started during evidence build; refusing "
                    f"artifact publication: {after}"
                )
            _assert_stat_unchanged(media)
            return built

        evidence, evidence_result = _materialize_or_skip(
            paths.evidence,
            producer=produce_evidence,
            validator=lambda payload: _validate_generated_evidence(
                payload,
                inputs=inputs,
                reference=reference,
            ),
        )
        _assert_stat_unchanged(media)
        finish_stage("evidence", evidence_result)

        expected_initial = resolve_reference_attribution_payload(evidence)
        _validate_canonical_attribution(expected_initial, evidence=evidence)
        initial, initial_result = _materialize_or_skip(
            paths.initial_attribution,
            producer=lambda: expected_initial,
            validator=lambda payload: (
                _validate_canonical_attribution(payload, evidence=evidence),
                _assert_canonical_json_equal(
                    payload, expected_initial, "initial attribution"
                ),
            ),
        )
        del initial
        finish_stage("initial_attribution", initial_result)

        augmentation = augment_reference_boundary_requests(evidence)
        expected_boundary = augmentation.payload
        boundary_evidence, boundary_result = _materialize_or_skip(
            paths.boundary_evidence,
            producer=lambda: expected_boundary,
            validator=lambda payload: _validate_boundary_evidence(
                payload,
                base_evidence=evidence,
                expected=expected_boundary,
            ),
        )
        finish_stage("boundary_evidence", boundary_result)

        expected_final = resolve_reference_attribution_payload(boundary_evidence)
        _validate_canonical_attribution(expected_final, evidence=boundary_evidence)
        final_attribution, final_result = _materialize_or_skip(
            paths.final_attribution,
            producer=lambda: expected_final,
            validator=lambda payload: (
                _validate_canonical_attribution(payload, evidence=boundary_evidence),
                _assert_canonical_json_equal(
                    payload, expected_final, "final attribution"
                ),
            ),
        )
        finish_stage("final_attribution", final_result)

        expected_dataset = build_reference_dataset(
            final_attribution,
            style_id=inputs.style_id,
            reference_id=reference.video_id,
        ).payload
        dataset, dataset_result = _materialize_or_skip(
            paths.dataset,
            producer=lambda: expected_dataset,
            validator=lambda payload: _validate_dataset_payload(
                payload, expected_dataset
            ),
        )
        del dataset
        finish_stage("dataset", dataset_result)

        expected_supervision = build_reference_supervision(
            final_attribution,
            style_id=inputs.style_id,
            reference_id=reference.video_id,
        ).payload
        supervision, supervision_result = _materialize_or_skip(
            paths.supervision,
            producer=lambda: expected_supervision,
            validator=lambda payload: _assert_canonical_json_equal(
                payload, expected_supervision, "reference supervision"
            ),
        )
        del supervision
        finish_stage("supervision", supervision_result)

        status["state"] = "complete"
        status["completed_at"] = _utc_now()
        status["updated_at"] = status["completed_at"]
        _write_status(paths.status_manifest, status)
        return {
            "reference_id": reference.video_id,
            "status": "complete",
            "status_manifest": str(paths.status_manifest),
            "stages": deepcopy(status["stages"]),
            "canonical_shot_count": expected_final["canonical_shot_schema"][
                "canonical_shot_count"
            ],
            "detected_cut_count": expected_final["canonical_shot_schema"][
                "detected_cut_count"
            ],
            "dataset_record_count": len(expected_dataset["records"]),
            "supervision_listwise_count": len(
                expected_supervision["listwise_records"]
            ),
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
    inputs: ExpansionInputs,
    reference: ReferenceSpec,
) -> dict[str, Any]:
    paths = _artifact_paths(inputs.output_directory, reference.video_id)
    actions = {
        stage: {
            "action": "validate_and_skip_if_canonical"
            if paths.stage_path(stage).exists()
            else "create",
            "path": str(paths.stage_path(stage)),
        }
        for stage in _STAGE_NAMES
    }
    # Full semantic validation is possible in dry-run only when the immutable
    # base evidence already exists.  If it does, walk the complete dependency
    # chain and fail on any mismatch.  Missing base evidence means no derived
    # artifact is allowed to pre-exist because it cannot be validated.
    if paths.evidence.exists():
        evidence, evidence_hash = _load_existing_artifact(
            paths.evidence,
            validator=lambda payload: _validate_generated_evidence(
                payload,
                inputs=inputs,
                reference=reference,
            ),
        )
        actions["evidence"].update(
            {"validation": "passed", "sha256": evidence_hash}
        )
        expected_initial = resolve_reference_attribution_payload(evidence)
        augmentation = augment_reference_boundary_requests(evidence)
        expected_boundary = augmentation.payload
        dependency_payloads: dict[str, tuple[Mapping[str, Any], Callable[[Mapping[str, Any]], None]]] = {
            "initial_attribution": (
                expected_initial,
                lambda payload: (
                    _validate_canonical_attribution(payload, evidence=evidence),
                    _assert_canonical_json_equal(
                        payload, expected_initial, "initial attribution"
                    ),
                ),
            ),
            "boundary_evidence": (
                expected_boundary,
                lambda payload: _validate_boundary_evidence(
                    payload,
                    base_evidence=evidence,
                    expected=expected_boundary,
                ),
            ),
        }
        boundary_payload: Mapping[str, Any] = expected_boundary
        if paths.boundary_evidence.exists():
            boundary_payload, _ = _load_existing_artifact(
                paths.boundary_evidence,
                validator=dependency_payloads["boundary_evidence"][1],
            )
        expected_final = resolve_reference_attribution_payload(boundary_payload)
        expected_dataset = build_reference_dataset(
            expected_final,
            style_id=inputs.style_id,
            reference_id=reference.video_id,
        ).payload
        expected_supervision = build_reference_supervision(
            expected_final,
            style_id=inputs.style_id,
            reference_id=reference.video_id,
        ).payload
        dependency_payloads.update(
            {
                "final_attribution": (
                    expected_final,
                    lambda payload: (
                        _validate_canonical_attribution(
                            payload, evidence=boundary_payload
                        ),
                        _assert_canonical_json_equal(
                            payload, expected_final, "final attribution"
                        ),
                    ),
                ),
                "dataset": (
                    expected_dataset,
                    lambda payload: _validate_dataset_payload(
                        payload, expected_dataset
                    ),
                ),
                "supervision": (
                    expected_supervision,
                    lambda payload: _assert_canonical_json_equal(
                        payload, expected_supervision, "reference supervision"
                    ),
                ),
            }
        )
        for stage, (_expected, validator) in dependency_payloads.items():
            path = paths.stage_path(stage)
            if path.exists():
                _payload, digest = _load_existing_artifact(path, validator=validator)
                actions[stage].update(
                    {"validation": "passed", "sha256": digest}
                )
    else:
        unexpected = [
            str(paths.stage_path(stage))
            for stage in _STAGE_NAMES[1:]
            if paths.stage_path(stage).exists()
        ]
        if unexpected:
            raise ExistingArtifactValidationError(
                "derived artifacts exist without canonical base evidence: "
                + ", ".join(unexpected)
            )
    return {
        "reference_id": reference.video_id,
        "status": "planned",
        "audit_cut_count": reference.audit_cut_count,
        "status_manifest": str(paths.status_manifest),
        "stages": actions,
    }


def run_reference_expansion_job(
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
    emit: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Validate inputs, plan safely, or materialize every target sequentially."""

    if expected_reference_count < 2:
        raise ValueError("expected reference count must be at least 2")
    if expected_source_count < MINIMUM_EXPECTED_SOURCE_COUNT:
        raise ValueError(
            "expected source count must be at least "
            f"{MINIMUM_EXPECTED_SOURCE_COUNT}"
        )
    if minimum_free_bytes < 0:
        raise ValueError("minimum_free_bytes cannot be negative")
    active = _active_analysis_processes()
    if active and not dry_run:
        raise ActiveAnalysisError(
            "another crosscut analysis is active; refusing media hashes/decode: "
            f"{active}"
        )

    inputs = _load_inputs(
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
    )
    reference_by_id = inputs.reference_by_id
    base_result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_BASIS,
        "pipeline_revision": PIPELINE_REVISION,
        "dry_run": dry_run,
        "job_fingerprint": inputs.job_fingerprint,
        "active_analysis_processes": [dict(item) for item in active],
        "input_validation": {
            "cadence_profile": "passed",
            "cadence_profile_file_sha256": inputs.profile_file_sha256,
            "cadence_profile_content_sha256": inputs.profile.profile_content_sha256,
            "seed_evidence": "passed",
            "seed_evidence_file_sha256": inputs.seed_evidence_file_sha256,
            "source_manifest": "passed",
            "source_manifest_file_sha256": inputs.source_manifest_file_sha256,
            "reference_count": len(inputs.references),
            "source_count": len(inputs.sources),
            "seed_reference_id": inputs.seed_reference_id,
            "media_sha256": (
                "passed" if inputs.media_sha256_verified else "deferred_until_real_run"
            ),
            "pipeline_code_sha256": deepcopy(dict(inputs.pipeline_code_sha256)),
        },
        "target_reference_ids": list(inputs.target_reference_ids),
        "reference_count": len(inputs.target_reference_ids),
        "references": [],
    }

    if dry_run:
        base_result["execution_blocked_until_active_analysis_exits"] = bool(active)
        base_result["writes_performed"] = False
        base_result["references"] = [
            _plan_one_reference(inputs, reference_by_id[reference_id])
            for reference_id in inputs.target_reference_ids
        ]
        return base_result

    free_bytes = shutil.disk_usage(inputs.output_directory).free
    if free_bytes < minimum_free_bytes:
        raise ReferenceExpansionError(
            f"insufficient free space: {free_bytes} < {minimum_free_bytes}"
        )
    base_result["free_bytes_before"] = free_bytes
    event_sink = emit or (lambda _event: None)
    with _job_lock(inputs.output_directory):
        for reference_id in inputs.target_reference_ids:
            result = _run_one_reference(
                inputs,
                reference_by_id[reference_id],
                emit=event_sink,
            )
            base_result["references"].append(result)
    base_result["writes_performed"] = True
    base_result["status"] = "complete"
    base_result["completed_reference_count"] = len(base_result["references"])
    return base_result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m crosscut.reference_expansion_job",
        description=(
            "safely resume expansion of non-seed professional references into "
            "canonical-v2 "
            "evidence, attribution, dataset, and supervision artifacts"
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
        help="optional non-seed reference id; repeat to run a subset",
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
        result = run_reference_expansion_job(
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


if __name__ == "__main__":  # pragma: no cover - exercised as a module CLI.
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_BASIS",
    "PIPELINE_REVISION",
    "SCHEMA_VERSION",
    "ActiveAnalysisError",
    "ExistingArtifactValidationError",
    "ReferenceExpansionError",
    "main",
    "run_reference_expansion_job",
]
