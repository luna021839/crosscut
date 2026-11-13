"""Bootstrap human same-song stage mixes into immutable StageEdit evidence.

The cadence audit schema predates strict artifact provenance and is intentionally
closed: adding fields would make existing BANG readers reject it.  This module
therefore publishes two bound artifacts per reference:

* ``*-human-cut-quality-audit.json`` keeps the exact BANG-compatible schema.
* ``*.provenance.json`` binds that audit to the media/config/code/source hashes
  and stores compact ``-2,-1,0,+1`` scalar/geometry/motion evidence.

Nothing writes over an existing artifact.  A resumed job may only reuse an
existing pair after revalidating every immutable binding.  The production
contract is exactly five human references and ten selected 1080p sources.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Callable, Iterator, Mapping, Sequence

import cv2

from . import (
    professional_reference_cadence as cadence_module,
    professional_reference_gate as gate_module,
    reference_match as reference_match_module,
    rendered_proxy_review as proxy_review_module,
    vision as vision_module,
)
from .professional_reference_cadence import (
    AUDIT_ANALYSIS_BASIS,
    CadenceConfig,
    ProfessionalReferenceCadenceProfile,
    ReferenceCadenceInput,
    build_professional_reference_cadence_profile,
)
from .professional_reference_gate import (
    ProfessionalReferenceCut,
    summarize_professional_reference_cuts,
)
from .reference_match import ReferenceMatchConfig, decode_target_frames_sequential
from .reference_training_source_pool import (
    validate_reference_training_source_manifest,
)
from .rendered_proxy_review import (
    BOUNDARY_FRAME_OFFSETS,
    HUMAN_REFERENCE_BOUNDARY_ANALYSIS_BASIS,
    GeometryProvider,
    ProxyReviewConfig,
    review_hard_cut_frame_window,
)
from .vision import SceneChange, SceneDetectionConfig, detect_scene_changes_in_video


SCHEMA_VERSION = 1
ANALYSIS_BASIS = "resumable_human_reference_stageedit_bootstrap_v2"
AUDIT_PROVENANCE_ANALYSIS_BASIS = (
    "human_stage_mix_cut_quality_audit_provenance_v2"
)
SOURCE_CONTRACT_ANALYSIS_BASIS = "exact_selected_source_training_contract_v1"
PIPELINE_REVISION = "human-reference-stageedit-bootstrap-v2"
EXPECTED_REFERENCE_COUNT = 5
EXPECTED_SOURCE_COUNT = 10
AUDIT_WORKING_WIDTH = 854
MINIMUM_SOURCE_SHORT_EDGE = 1080
MINIMUM_SOURCE_FRAME_RATE = 29.0
_VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class HumanReferenceBootstrapError(RuntimeError):
    """Raised when immutable or technical evidence cannot be trusted."""


@dataclass(frozen=True, slots=True)
class MediaIdentity:
    path: Path
    sha256: str
    size_bytes: int
    mtime_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
        }


@dataclass(frozen=True, slots=True)
class ReferenceIdentity:
    video_id: str
    media: MediaIdentity


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    source_index: int
    video_id: str
    media: MediaIdentity
    width: int
    height: int
    frame_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_index": self.source_index,
            "video_id": self.video_id,
            "media": self.media.to_dict(),
            "width": self.width,
            "height": self.height,
            "frame_rate": self.frame_rate,
        }


@dataclass(frozen=True, slots=True)
class HumanReferenceBootstrapConfig:
    """Fixed production contract plus deterministic analysis settings."""

    expected_reference_count: int = EXPECTED_REFERENCE_COUNT
    expected_source_count: int = EXPECTED_SOURCE_COUNT
    master_source_index: int = 0
    audit_working_width: int = AUDIT_WORKING_WIDTH
    scene_detection: SceneDetectionConfig = field(
        default_factory=SceneDetectionConfig
    )
    boundary_review: ProxyReviewConfig = field(default_factory=ProxyReviewConfig)
    cadence: CadenceConfig = field(default_factory=CadenceConfig)
    cadence_sample_rate: int = 8_000
    require_zero_technical_boundary_issues: bool = True

    def __post_init__(self) -> None:
        if self.expected_reference_count != EXPECTED_REFERENCE_COUNT:
            raise ValueError(
                f"human-reference bootstrap requires exactly "
                f"{EXPECTED_REFERENCE_COUNT} references"
            )
        if self.expected_source_count != EXPECTED_SOURCE_COUNT:
            raise ValueError(
                f"human-reference bootstrap requires exactly "
                f"{EXPECTED_SOURCE_COUNT} selected sources"
            )
        if (
            isinstance(self.master_source_index, bool)
            or not isinstance(self.master_source_index, int)
            or not 0 <= self.master_source_index < self.expected_source_count
        ):
            raise ValueError("master_source_index is outside the selected pool")
        if (
            isinstance(self.audit_working_width, bool)
            or not isinstance(self.audit_working_width, int)
            or self.audit_working_width != AUDIT_WORKING_WIDTH
        ):
            raise ValueError(
                f"audit_working_width must remain the calibrated {AUDIT_WORKING_WIDTH}"
            )
        if tuple(BOUNDARY_FRAME_OFFSETS) != (-2, -1, 0, 1):
            raise ValueError("production boundary frame offsets changed unexpectedly")
        if (
            isinstance(self.cadence_sample_rate, bool)
            or not isinstance(self.cadence_sample_rate, int)
            or self.cadence_sample_rate <= 0
        ):
            raise ValueError("cadence_sample_rate must be a positive integer")
        if not isinstance(self.require_zero_technical_boundary_issues, bool):
            raise TypeError("require_zero_technical_boundary_issues must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_basis": ANALYSIS_BASIS,
            "pipeline_revision": PIPELINE_REVISION,
            "expected_reference_count": self.expected_reference_count,
            "expected_source_count": self.expected_source_count,
            "master_source_index": self.master_source_index,
            "minimum_source_short_edge": MINIMUM_SOURCE_SHORT_EDGE,
            "minimum_source_frame_rate": MINIMUM_SOURCE_FRAME_RATE,
            "audit_working_width": self.audit_working_width,
            "boundary_frame_offsets": list(BOUNDARY_FRAME_OFFSETS),
            "hard_cut_detection": {
                "threshold": self.scene_detection.threshold,
                "minimum_frame_difference": (
                    self.scene_detection.minimum_frame_difference
                ),
                "minimum_histogram_distance": (
                    self.scene_detection.minimum_histogram_distance
                ),
                "minimum_gap_frames": self.scene_detection.min_gap_frames,
                "frame_difference_weight": (
                    self.scene_detection.frame_difference_weight
                ),
                "histogram_weight": self.scene_detection.histogram_weight,
                "difference_gain": self.scene_detection.difference_gain,
                "analysis_size": list(self.scene_detection.analysis_size),
                "histogram_bins": self.scene_detection.histogram_bins,
            },
            "boundary_review": self.boundary_review.to_dict(),
            "cadence": self.cadence.to_dict(),
            "cadence_sample_rate": self.cadence_sample_rate,
            "require_zero_technical_boundary_issues": (
                self.require_zero_technical_boundary_issues
            ),
        }


@dataclass(frozen=True, slots=True)
class HumanCutAuditResult:
    audit: dict[str, Any]
    boundary_evidence: tuple[dict[str, Any], ...]
    frame_rate: float
    declared_frame_count: int


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    info = path.stat()
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _sha256_file_stable(path: Path) -> str:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    before = _stat_identity(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    after = _stat_identity(source)
    if before != after:
        raise HumanReferenceBootstrapError(f"input changed while hashing: {source}")
    return digest.hexdigest()


def _stable_read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    source = Path(path).expanduser().resolve()
    before = _stat_identity(source)
    raw = source.read_bytes()
    after = _stat_identity(source)
    if before != after:
        raise HumanReferenceBootstrapError(f"JSON changed while loading: {source}")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root must be an object: {source}")
    return (deepcopy(dict(payload)), raw)


def _media_identity(path: Path, *, expected_sha256: str | None = None) -> MediaIdentity:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"media must be a regular non-symlink file: {source}")
    digest = _sha256_file_stable(source)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"media SHA-256 mismatch: {source}")
    info = source.stat()
    return MediaIdentity(
        path=source,
        sha256=digest,
        size_bytes=int(info.st_size),
        mtime_ns=int(info.st_mtime_ns),
    )


def _module_code_hashes() -> dict[str, str]:
    modules = {
        "crosscut.human_reference_bootstrap": Path(__file__),
        "crosscut.professional_reference_cadence": Path(cadence_module.__file__),
        "crosscut.professional_reference_gate": Path(gate_module.__file__),
        "crosscut.reference_match": Path(reference_match_module.__file__),
        "crosscut.rendered_proxy_review": Path(proxy_review_module.__file__),
        "crosscut.vision": Path(vision_module.__file__),
    }
    return {
        name: _sha256_file_stable(path.resolve())
        for name, path in sorted(modules.items())
    }


def _content_hash(payload: Mapping[str, Any], field_name: str) -> str:
    copied = deepcopy(dict(payload))
    copied[field_name] = None
    return _canonical_json_sha256(copied)


def _write_bytes_no_replace(destination: Path, payload: bytes) -> Path:
    """Durably create one immutable file and never replace any existing inode."""

    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor: int | None = None
    created_identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(target, flags, 0o600)
        opened = os.fstat(descriptor)
        created_identity = (int(opened.st_dev), int(opened.st_ino))
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return target
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if created_identity is not None:
            try:
                observed = os.lstat(target)
                if (
                    stat.S_ISREG(observed.st_mode)
                    and (int(observed.st_dev), int(observed.st_ino))
                    == created_identity
                ):
                    target.unlink()
            except FileNotFoundError:
                pass
        raise


def _write_json_no_replace(destination: Path, payload: Mapping[str, Any]) -> Path:
    return _write_bytes_no_replace(destination, _pretty_json_bytes(payload))


def _video_id_from_path(path: Path) -> str:
    match = re.match(r"^([A-Za-z0-9_-]{11})(?:\.|-|$)", path.name)
    if match is None or not _VIDEO_ID_PATTERN.fullmatch(match.group(1)):
        raise ValueError(f"reference filename does not begin with a YouTube id: {path}")
    return match.group(1)


def discover_reference_media(
    reference_directory: Path,
    *,
    expected_count: int = EXPECTED_REFERENCE_COUNT,
) -> tuple[Path, ...]:
    root = Path(reference_directory).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"reference directory is unavailable: {root}")
    media = tuple(
        sorted(
            path.resolve()
            for path in root.glob("*.mp4")
            if not path.name.startswith("._")
        )
    )
    if len(media) != expected_count:
        raise ValueError(
            f"expected exactly {expected_count} human MP4 references, found {len(media)}"
        )
    video_ids = tuple(_video_id_from_path(path) for path in media)
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("human reference video ids are not unique")
    if any(not path.is_file() or path.is_symlink() for path in media):
        raise ValueError("human references must be regular non-symlink files")
    return media


def _load_source_contract(
    source_manifest_path: Path,
    *,
    expected_artist: str,
    expected_song: str,
    expected_count: int,
) -> tuple[tuple[SourceIdentity, ...], str]:
    manifest_path = Path(source_manifest_path).expanduser().resolve()
    manifest, raw = _stable_read_json(manifest_path)
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
    if len(selected) != expected_count:
        raise ValueError(
            f"expected exactly {expected_count} selected sources, found {len(selected)}"
        )
    if manifest.get("desired_source_count") != expected_count:
        raise ValueError("manifest desired_source_count does not match exact contract")

    sources: list[SourceIdentity] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for expected_index, raw_source in enumerate(selected):
        source = _mapping(raw_source, f"selected source {expected_index}")
        source_index = _integer(source.get("source_index"), "source_index")
        if source_index != expected_index:
            raise ValueError("selected source indices must be contiguous and ordered")
        video_id = str(source.get("video_id") or "")
        if not _VIDEO_ID_PATTERN.fullmatch(video_id) or video_id in seen_ids:
            raise ValueError(f"invalid or duplicate selected source id: {video_id!r}")
        seen_ids.add(video_id)
        media_path = Path(str(source.get("local_path") or "")).expanduser().resolve()
        if media_path in seen_paths:
            raise ValueError("selected source paths are not unique")
        seen_paths.add(media_path)
        validation = _mapping(
            source.get("media_validation"),
            f"selected source {video_id} media_validation",
        )
        if validation.get("passed") is not True:
            raise ValueError(f"selected source {video_id} failed acquisition media QA")
        width = _integer(validation.get("width"), f"source {video_id} width")
        height = _integer(validation.get("height"), f"source {video_id} height")
        frame_rate = _finite(
            validation.get("frame_rate"),
            f"source {video_id} frame_rate",
        )
        if min(width, height) < MINIMUM_SOURCE_SHORT_EDGE:
            raise ValueError(f"selected source {video_id} is below 1080p")
        if frame_rate < MINIMUM_SOURCE_FRAME_RATE:
            raise ValueError(f"selected source {video_id} frame rate is too low")
        if (
            source.get("group_quality_eligible") is False
            and not training_source_ids
        ):
            raise ValueError(f"selected source {video_id} is explicitly quality-ineligible")
        content = _mapping(
            source.get("content_integrity"),
            f"selected source {video_id} content_integrity",
        )
        if content.get("contract_version") != 1 or content.get("algorithm") != "sha256":
            raise ValueError(f"selected source {video_id} content integrity is unsupported")
        declared_sha = str(content.get("sha256") or "")
        if not _SHA256_PATTERN.fullmatch(declared_sha):
            raise ValueError(f"selected source {video_id} SHA-256 is invalid")
        declared_size = _integer(
            content.get("size_bytes"),
            f"selected source {video_id} size_bytes",
        )
        identity = _media_identity(media_path, expected_sha256=declared_sha)
        if identity.size_bytes != declared_size:
            raise ValueError(f"selected source {video_id} size mismatch")
        sources.append(
            SourceIdentity(
                source_index=source_index,
                video_id=video_id,
                media=identity,
                width=width,
                height=height,
                frame_rate=frame_rate,
            )
        )
    return (tuple(sources), _sha256_bytes(raw))


def _audit_cut_payload(boundary: Mapping[str, Any], frame_rate: float) -> dict[str, Any]:
    frame_index = _integer(boundary.get("cut_frame_index"), "cut_frame_index")
    discontinuities = deepcopy(
        dict(_mapping(boundary.get("discontinuities"), "discontinuities"))
    )
    reasons = [
        str(reason)
        for reason in _array(boundary.get("issue_reasons"), "issue_reasons")
    ]
    score = _finite(boundary.get("weighted_score"), "weighted_score")
    # Constructor validation guarantees the exact component/value contract used
    # by both the professional-reference profile and learned reviewer.
    ProfessionalReferenceCut(
        score=score,
        discontinuities=discontinuities,
        absolute_issue_reasons=tuple(reasons),
    )
    return {
        "frame_index": frame_index,
        "seconds": frame_index / frame_rate,
        "score": score,
        "discontinuities": discontinuities,
        "issue_reasons": reasons,
    }


def build_human_cut_quality_audit(
    reference_path: Path,
    *,
    scene_detection: SceneDetectionConfig | None = None,
    boundary_review: ProxyReviewConfig | None = None,
    geometry_provider: GeometryProvider | None = None,
    audit_working_width: int = AUDIT_WORKING_WIDTH,
    require_zero_technical_boundary_issues: bool = True,
) -> HumanCutAuditResult:
    """Detect every hard cut and build exact BANG-compatible audit evidence."""

    reference = Path(reference_path).expanduser().resolve()
    if not reference.is_file() or reference.is_symlink():
        raise FileNotFoundError(reference)
    if audit_working_width != AUDIT_WORKING_WIDTH:
        raise ValueError("human audit must use the calibrated 854px proxy width")
    detector_config = scene_detection or SceneDetectionConfig()
    review_config = boundary_review or ProxyReviewConfig()
    changes = detect_scene_changes_in_video(
        reference,
        config=detector_config,
        sample_every=1,
    )
    if not changes:
        raise HumanReferenceBootstrapError(
            f"hard-cut detector found no cuts in human reference: {reference}"
        )
    target_indices = tuple(
        sorted(
            {
                change.frame_index + offset
                for change in changes
                for offset in BOUNDARY_FRAME_OFFSETS
            }
        )
    )
    if target_indices[0] < 0:
        raise HumanReferenceBootstrapError(
            "human reference has a cut without two outgoing review frames"
        )
    decoder_config = ReferenceMatchConfig(
        working_width=audit_working_width,
        temporal_jitter_frames=(0,),
    )
    decoded = decode_target_frames_sequential(
        reference,
        target_indices,
        config=decoder_config,
    )
    if decoded.missing_indices:
        raise HumanReferenceBootstrapError(
            "human boundary frames could not be decoded: "
            + ", ".join(str(index) for index in decoded.missing_indices)
        )
    if decoded.declared_frame_count <= target_indices[-1]:
        raise HumanReferenceBootstrapError(
            "declared frame count does not cover the final boundary review frame"
        )

    evidence: list[dict[str, Any]] = []
    audit_cuts: list[dict[str, Any]] = []
    technical_failures: list[dict[str, Any]] = []
    changes_by_frame = {change.frame_index: change for change in changes}
    if len(changes_by_frame) != len(changes):
        raise HumanReferenceBootstrapError("hard-cut detector returned duplicate indices")
    for change in changes:
        frame_window = {
            change.frame_index + offset: decoded.frames[change.frame_index + offset]
            for offset in BOUNDARY_FRAME_OFFSETS
        }
        review = review_hard_cut_frame_window(
            frame_window,
            cut_frame_index=change.frame_index,
            config=review_config,
            geometry_provider=geometry_provider,
        )
        if review.get("analysis_basis") != HUMAN_REFERENCE_BOUNDARY_ANALYSIS_BASIS:
            raise HumanReferenceBootstrapError("boundary review analysis basis drifted")
        technical = tuple(
            str(reason)
            for reason in _array(
                review.get("technical_integrity_issue_reasons"),
                "technical_integrity_issue_reasons",
            )
        )
        if technical:
            technical_failures.append(
                {
                    "cut_frame_index": change.frame_index,
                    "issue_reasons": list(technical),
                }
            )
        audit_cuts.append(_audit_cut_payload(review, decoded.frame_rate))
        evidence.append(
            {
                "detector": {
                    "frame_index": change.frame_index,
                    "score": change.score,
                    "frame_difference": change.frame_difference,
                    "histogram_distance": change.histogram_distance,
                },
                "boundary_review": review,
            }
        )
    if require_zero_technical_boundary_issues and technical_failures:
        raise HumanReferenceBootstrapError(
            "human reference contains technical boundary failures: "
            + json.dumps(technical_failures, ensure_ascii=False, allow_nan=False)
        )

    professional_cuts = tuple(
        ProfessionalReferenceCut.from_mapping(cut) for cut in audit_cuts
    )
    audit = {
        "analysis_basis": AUDIT_ANALYSIS_BASIS,
        "reference": str(reference),
        "summary": summarize_professional_reference_cuts(professional_cuts),
        "cuts": audit_cuts,
    }
    if set(audit) != {"analysis_basis", "reference", "summary", "cuts"}:
        raise AssertionError("human cut audit root schema changed")
    for cut in audit_cuts:
        if set(cut) != {
            "frame_index",
            "seconds",
            "score",
            "discontinuities",
            "issue_reasons",
        }:
            raise AssertionError("human cut audit cut schema changed")
    json.dumps(audit, ensure_ascii=False, allow_nan=False)
    return HumanCutAuditResult(
        audit=audit,
        boundary_evidence=tuple(evidence),
        frame_rate=decoded.frame_rate,
        declared_frame_count=decoded.declared_frame_count,
    )


def _validate_audit_payload(
    payload: Mapping[str, Any],
    *,
    reference: ReferenceIdentity,
) -> tuple[ProfessionalReferenceCut, ...]:
    if set(payload) != {"analysis_basis", "reference", "summary", "cuts"}:
        raise ValueError("human cut audit top-level keys do not match BANG schema")
    if payload.get("analysis_basis") != AUDIT_ANALYSIS_BASIS:
        raise ValueError("human cut audit analysis basis is unsupported")
    if Path(str(payload.get("reference"))).expanduser().resolve() != reference.media.path:
        raise ValueError("human cut audit media binding is incorrect")
    raw_cuts = _array(payload.get("cuts"), "human cut audit cuts")
    if not raw_cuts:
        raise ValueError("human cut audit contains no cuts")
    cuts: list[ProfessionalReferenceCut] = []
    previous_frame = -1
    previous_seconds = -math.inf
    for index, raw_cut in enumerate(raw_cuts):
        cut = _mapping(raw_cut, f"human cut audit cut {index}")
        if set(cut) != {
            "frame_index",
            "seconds",
            "score",
            "discontinuities",
            "issue_reasons",
        }:
            raise ValueError("human cut audit cut keys do not match BANG schema")
        frame_index = _integer(cut.get("frame_index"), "audit frame_index")
        seconds = _finite(cut.get("seconds"), "audit seconds")
        if frame_index <= previous_frame or seconds <= previous_seconds:
            raise ValueError("human cut audit order is not strictly increasing")
        previous_frame = frame_index
        previous_seconds = seconds
        cuts.append(ProfessionalReferenceCut.from_mapping(cut))
    expected_summary = summarize_professional_reference_cuts(tuple(cuts))
    if _canonical_json_bytes(payload.get("summary")) != _canonical_json_bytes(
        expected_summary
    ):
        raise ValueError("human cut audit summary does not match recomputed cuts")
    return tuple(cuts)


def _audit_paths(output_directory: Path, reference_id: str) -> tuple[Path, Path]:
    audit = output_directory / f"{reference_id}-human-cut-quality-audit.json"
    provenance = output_directory / (
        f"{reference_id}-human-cut-quality-audit.provenance.json"
    )
    return (audit, provenance)


def _song_slug(song: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", song.upper()).strip("-")
    if not slug:
        raise ValueError("song cannot produce an artifact slug")
    return slug


def _profile_path(output_directory: Path, artist: str, song: str) -> Path:
    return output_directory / (
        f"{_song_slug(artist)}-{_song_slug(song)}-"
        "professional-reference-cadence-v1.json"
    )


def _job_manifest_path(output_directory: Path, artist: str, song: str) -> Path:
    return output_directory / (
        f"{_song_slug(artist)}-{_song_slug(song)}-"
        "human-reference-stageedit-bootstrap-v2.json"
    )


@contextmanager
def _job_lock(output_directory: Path) -> Iterator[None]:
    lock_path = output_directory / ".human-reference-stageedit-bootstrap-v2.lock"
    if lock_path.is_symlink():
        raise HumanReferenceBootstrapError(f"job lock is a symlink: {lock_path}")
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise HumanReferenceBootstrapError(
                f"another bootstrap job holds {lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_provenance(
    payload: Mapping[str, Any],
    *,
    job_fingerprint: str,
    config_sha256: str,
    code_sha256: Mapping[str, str],
    source_manifest_sha256: str,
    reference: ReferenceIdentity,
    audit_path: Path,
    audit_sha256: str,
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("audit provenance schema is unsupported")
    if payload.get("analysis") != AUDIT_PROVENANCE_ANALYSIS_BASIS:
        raise ValueError("audit provenance analysis basis is unsupported")
    if payload.get("provenance_content_sha256") != _content_hash(
        payload,
        "provenance_content_sha256",
    ):
        raise ValueError("audit provenance content SHA-256 mismatch")
    if payload.get("job_fingerprint") != job_fingerprint:
        raise ValueError("audit provenance job fingerprint mismatch")
    if payload.get("configuration_sha256") != config_sha256:
        raise ValueError("audit provenance configuration SHA-256 mismatch")
    if _canonical_json_bytes(payload.get("pipeline_code_sha256")) != (
        _canonical_json_bytes(code_sha256)
    ):
        raise ValueError("audit provenance code SHA-256 mismatch")
    source = _mapping(payload.get("source_contract"), "source_contract")
    if source.get("manifest_file_sha256") != source_manifest_sha256:
        raise ValueError("audit provenance source manifest SHA-256 mismatch")
    media = _mapping(payload.get("reference_media"), "reference_media")
    if Path(str(media.get("path"))).expanduser().resolve() != reference.media.path:
        raise ValueError("audit provenance reference path mismatch")
    if media.get("sha256") != reference.media.sha256:
        raise ValueError("audit provenance reference SHA-256 mismatch")
    artifact = _mapping(payload.get("audit_artifact"), "audit_artifact")
    if Path(str(artifact.get("path"))).expanduser().resolve() != audit_path:
        raise ValueError("audit provenance artifact path mismatch")
    if artifact.get("sha256") != audit_sha256:
        raise ValueError("audit provenance artifact SHA-256 mismatch")
    if payload.get("boundary_frame_offsets") != list(BOUNDARY_FRAME_OFFSETS):
        raise ValueError("audit provenance boundary offsets mismatch")


def _build_audit_provenance(
    *,
    reference: ReferenceIdentity,
    audit_path: Path,
    audit_sha256: str,
    result: HumanCutAuditResult,
    source_manifest_path: Path,
    source_manifest_sha256: str,
    sources: Sequence[SourceIdentity],
    job_fingerprint: str,
    configuration: Mapping[str, Any],
    configuration_sha256: str,
    pipeline_code_sha256: Mapping[str, str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis": AUDIT_PROVENANCE_ANALYSIS_BASIS,
        "provenance_content_sha256": None,
        "created_at": _utc_now(),
        "job_fingerprint": job_fingerprint,
        "configuration": deepcopy(dict(configuration)),
        "configuration_sha256": configuration_sha256,
        "pipeline_code_sha256": deepcopy(dict(pipeline_code_sha256)),
        "reference_video_id": reference.video_id,
        "reference_media": reference.media.to_dict(),
        "audit_artifact": {
            "path": str(audit_path),
            "sha256": audit_sha256,
            "analysis_basis": AUDIT_ANALYSIS_BASIS,
            "cut_count": len(result.audit["cuts"]),
        },
        "source_contract": {
            "analysis_basis": SOURCE_CONTRACT_ANALYSIS_BASIS,
            "manifest_path": str(source_manifest_path),
            "manifest_file_sha256": source_manifest_sha256,
            "selected_source_count": len(sources),
            "selected_source_ids": [source.video_id for source in sources],
            "all_selected_media_sha256_verified": True,
        },
        "decoder": {
            "policy": "forward_only_exact_target_union_decode_v1",
            "working_width": AUDIT_WORKING_WIDTH,
            "frame_rate": result.frame_rate,
            "declared_frame_count": result.declared_frame_count,
        },
        "hard_cut_detection_analysis_basis": (
            "ive_component_gated_hard_cut_detector_v2"
        ),
        "boundary_review_analysis_basis": (
            HUMAN_REFERENCE_BOUNDARY_ANALYSIS_BASIS
        ),
        "boundary_frame_offsets": list(BOUNDARY_FRAME_OFFSETS),
        "boundaries": [deepcopy(item) for item in result.boundary_evidence],
        "technical_integrity": {
            "zero_tolerance": True,
            "issue_count": 0,
            "passed": True,
        },
        "serialization_policy": {
            "raw_frames_serialized": False,
            "descriptors_or_keypoints_serialized": False,
            "appearance_embeddings_serialized": False,
            "compact_scalar_geometry_motion_evidence_only": True,
        },
    }
    payload["provenance_content_sha256"] = _content_hash(
        payload,
        "provenance_content_sha256",
    )
    return payload


def _validate_existing_profile(
    path: Path,
    *,
    artist: str,
    song: str,
    master: SourceIdentity,
    references: Sequence[ReferenceIdentity],
    audit_paths: Mapping[str, Path],
) -> ProfessionalReferenceCadenceProfile:
    profile = ProfessionalReferenceCadenceProfile.from_file(
        path,
        verify_input_hashes=True,
    )
    if profile.artist != artist or profile.song != song:
        raise ValueError("existing cadence profile song identity mismatch")
    if profile.reference_count != len(references):
        raise ValueError("existing cadence profile reference count mismatch")
    if profile.master_media_path != master.media.path:
        raise ValueError("existing cadence profile master path mismatch")
    if profile.master_media_sha256 != master.media.sha256:
        raise ValueError("existing cadence profile master SHA-256 mismatch")
    expected = {
        reference.video_id: {
            "video_id": reference.video_id,
            "media_path": str(reference.media.path),
            "media_sha256": reference.media.sha256,
            "audit_path": str(audit_paths[reference.video_id]),
            "audit_sha256": _sha256_file_stable(audit_paths[reference.video_id]),
        }
        for reference in references
    }
    observed = {item["video_id"]: item for item in profile.reference_provenance()}
    if _canonical_json_bytes(observed) != _canonical_json_bytes(expected):
        raise ValueError("existing cadence profile reference bindings mismatch")
    return profile


def run_human_reference_bootstrap(
    *,
    reference_directory: Path,
    source_manifest_path: Path,
    output_directory: Path,
    artist: str,
    song: str,
    config: HumanReferenceBootstrapConfig | None = None,
    ffmpeg_bin: str = "ffmpeg",
    dry_run: bool = False,
    geometry_provider: GeometryProvider | None = None,
    emit: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build/resume audits, immutable sidecars, and one cadence profile."""

    settings = config or HumanReferenceBootstrapConfig()
    normalized_artist = str(artist).strip()
    normalized_song = str(song).strip()
    if not normalized_artist or not normalized_song:
        raise ValueError("artist and song cannot be blank")
    output = Path(output_directory).expanduser().resolve()
    if output.exists() and (not output.is_dir() or output.is_symlink()):
        raise ValueError("output_directory must be a real directory")
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = Path(source_manifest_path).expanduser().resolve()
    reference_paths = discover_reference_media(
        Path(reference_directory),
        expected_count=settings.expected_reference_count,
    )
    event_sink = emit or (lambda _event: None)
    event_sink({"stage": "hash_reference_media", "count": len(reference_paths)})
    references = tuple(
        ReferenceIdentity(_video_id_from_path(path), _media_identity(path))
        for path in reference_paths
    )
    event_sink({"stage": "validate_exact_source_contract"})
    sources, source_manifest_sha256 = _load_source_contract(
        source_manifest,
        expected_artist=normalized_artist,
        expected_song=normalized_song,
        expected_count=settings.expected_source_count,
    )
    master = sources[settings.master_source_index]
    configuration = settings.to_dict()
    configuration_sha256 = _canonical_json_sha256(configuration)
    pipeline_code_sha256 = _module_code_hashes()
    fingerprint_payload = {
        "analysis": ANALYSIS_BASIS,
        "pipeline_revision": PIPELINE_REVISION,
        "artist": normalized_artist,
        "song": normalized_song,
        "configuration_sha256": configuration_sha256,
        "pipeline_code_sha256": pipeline_code_sha256,
        "source_manifest_file_sha256": source_manifest_sha256,
        "references": [
            {
                "video_id": reference.video_id,
                "path": str(reference.media.path),
                "sha256": reference.media.sha256,
            }
            for reference in references
        ],
    }
    job_fingerprint = _canonical_json_sha256(fingerprint_payload)
    profile_path = _profile_path(output, normalized_artist, normalized_song)
    manifest_path = _job_manifest_path(output, normalized_artist, normalized_song)
    planned_audits = {
        reference.video_id: _audit_paths(output, reference.video_id)
        for reference in references
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_BASIS,
        "status": "planned" if dry_run else "running",
        "artist": normalized_artist,
        "song": normalized_song,
        "job_fingerprint": job_fingerprint,
        "configuration_sha256": configuration_sha256,
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": source_manifest_sha256,
            "selected_source_count": len(sources),
            "master_source_index": master.source_index,
            "master_source_video_id": master.video_id,
            "all_selected_media_sha256_verified": True,
        },
        "reference_count": len(references),
        "output_directory": str(output),
        "cadence_profile_path": str(profile_path),
        "job_manifest_path": str(manifest_path),
        "references": [
            {
                "video_id": reference.video_id,
                "media_path": str(reference.media.path),
                "media_sha256": reference.media.sha256,
                "audit_path": str(planned_audits[reference.video_id][0]),
                "provenance_path": str(planned_audits[reference.video_id][1]),
                "status": "planned",
            }
            for reference in references
        ],
    }
    if dry_run:
        return result

    reference_results: list[dict[str, Any]] = []
    audit_paths: dict[str, Path] = {}
    with _job_lock(output):
        for reference in references:
            audit_path, provenance_path = planned_audits[reference.video_id]
            audit_paths[reference.video_id] = audit_path
            if provenance_path.exists() and not audit_path.exists():
                raise HumanReferenceBootstrapError(
                    f"provenance exists without its audit: {provenance_path}"
                )
            if audit_path.exists() and provenance_path.exists():
                audit_payload, audit_raw = _stable_read_json(audit_path)
                _validate_audit_payload(audit_payload, reference=reference)
                provenance_payload, _raw = _stable_read_json(provenance_path)
                _validate_provenance(
                    provenance_payload,
                    job_fingerprint=job_fingerprint,
                    config_sha256=configuration_sha256,
                    code_sha256=pipeline_code_sha256,
                    source_manifest_sha256=source_manifest_sha256,
                    reference=reference,
                    audit_path=audit_path,
                    audit_sha256=_sha256_bytes(audit_raw),
                )
                reference_results.append(
                    {
                        "video_id": reference.video_id,
                        "status": "validated_existing",
                        "audit_path": str(audit_path),
                        "audit_sha256": _sha256_bytes(audit_raw),
                        "provenance_path": str(provenance_path),
                        "provenance_sha256": _sha256_file_stable(provenance_path),
                        "cut_count": len(audit_payload["cuts"]),
                    }
                )
                event_sink(
                    {
                        "stage": "reference_complete",
                        "video_id": reference.video_id,
                        "status": "validated_existing",
                    }
                )
                continue

            event_sink(
                {"stage": "analyze_reference", "video_id": reference.video_id}
            )
            audit_result = build_human_cut_quality_audit(
                reference.media.path,
                scene_detection=settings.scene_detection,
                boundary_review=settings.boundary_review,
                geometry_provider=geometry_provider,
                audit_working_width=settings.audit_working_width,
                require_zero_technical_boundary_issues=(
                    settings.require_zero_technical_boundary_issues
                ),
            )
            computed_audit_bytes = _pretty_json_bytes(audit_result.audit)
            if audit_path.exists():
                existing_payload, audit_raw = _stable_read_json(audit_path)
                _validate_audit_payload(existing_payload, reference=reference)
                if _canonical_json_bytes(existing_payload) != _canonical_json_bytes(
                    audit_result.audit
                ):
                    raise HumanReferenceBootstrapError(
                        f"existing audit differs from recomputed evidence: {audit_path}"
                    )
            else:
                _write_bytes_no_replace(audit_path, computed_audit_bytes)
                audit_raw = audit_path.read_bytes()
            audit_sha256 = _sha256_bytes(audit_raw)
            provenance_payload = _build_audit_provenance(
                reference=reference,
                audit_path=audit_path,
                audit_sha256=audit_sha256,
                result=audit_result,
                source_manifest_path=source_manifest,
                source_manifest_sha256=source_manifest_sha256,
                sources=sources,
                job_fingerprint=job_fingerprint,
                configuration=configuration,
                configuration_sha256=configuration_sha256,
                pipeline_code_sha256=pipeline_code_sha256,
            )
            _write_json_no_replace(provenance_path, provenance_payload)
            reference_results.append(
                {
                    "video_id": reference.video_id,
                    "status": "created",
                    "audit_path": str(audit_path),
                    "audit_sha256": audit_sha256,
                    "provenance_path": str(provenance_path),
                    "provenance_sha256": _sha256_file_stable(provenance_path),
                    "cut_count": len(audit_result.audit["cuts"]),
                }
            )
            event_sink(
                {
                    "stage": "reference_complete",
                    "video_id": reference.video_id,
                    "status": "created",
                    "cut_count": len(audit_result.audit["cuts"]),
                }
            )

        if profile_path.exists():
            profile = _validate_existing_profile(
                profile_path,
                artist=normalized_artist,
                song=normalized_song,
                master=master,
                references=references,
                audit_paths=audit_paths,
            )
            profile_status = "validated_existing"
        else:
            event_sink({"stage": "build_cadence_profile"})
            cadence_inputs = tuple(
                ReferenceCadenceInput(
                    reference.video_id,
                    reference.media.path,
                    audit_paths[reference.video_id],
                )
                for reference in references
            )
            profile = build_professional_reference_cadence_profile(
                master.media.path,
                cadence_inputs,
                artist=normalized_artist,
                song=normalized_song,
                config=settings.cadence,
                sample_rate=settings.cadence_sample_rate,
                ffmpeg_bin=ffmpeg_bin,
            )
            _write_json_no_replace(profile_path, profile.to_dict())
            profile = _validate_existing_profile(
                profile_path,
                artist=normalized_artist,
                song=normalized_song,
                master=master,
                references=references,
                audit_paths=audit_paths,
            )
            profile_status = "created"

        manifest_payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "analysis": ANALYSIS_BASIS,
            "manifest_content_sha256": None,
            "created_at": _utc_now(),
            "status": "complete",
            "artist": normalized_artist,
            "song": normalized_song,
            "job_fingerprint": job_fingerprint,
            "configuration": configuration,
            "configuration_sha256": configuration_sha256,
            "pipeline_code_sha256": pipeline_code_sha256,
            "source_contract": {
                "analysis_basis": SOURCE_CONTRACT_ANALYSIS_BASIS,
                "manifest_path": str(source_manifest),
                "manifest_file_sha256": source_manifest_sha256,
                "selected_source_count": len(sources),
                "sources": [source.to_dict() for source in sources],
                "master_source_index": master.source_index,
                "master_source_video_id": master.video_id,
                "all_selected_media_sha256_verified": True,
            },
            "references": reference_results,
            "cadence_profile": {
                "path": str(profile_path),
                "file_sha256": _sha256_file_stable(profile_path),
                "profile_content_sha256": profile.profile_content_sha256,
                "reference_count": profile.reference_count,
            },
            "safety_policy": {
                "existing_artifacts_overwritten": False,
                "technical_boundary_issues_allowed": False,
                "raw_frames_serialized": False,
                "input_media_sha256_verified": True,
            },
        }
        manifest_payload["manifest_content_sha256"] = _content_hash(
            manifest_payload,
            "manifest_content_sha256",
        )
        if manifest_path.exists():
            existing_manifest, _raw = _stable_read_json(manifest_path)
            if existing_manifest.get("manifest_content_sha256") != _content_hash(
                existing_manifest,
                "manifest_content_sha256",
            ):
                raise ValueError("existing job manifest content SHA-256 mismatch")
            if existing_manifest.get("job_fingerprint") != job_fingerprint:
                raise ValueError("existing job manifest fingerprint mismatch")
            # Creation timestamps and per-reference created/validated status are
            # operational history; every immutable dependency must still match.
            for key in (
                "configuration_sha256",
                "pipeline_code_sha256",
                "source_contract",
                "cadence_profile",
                "safety_policy",
            ):
                if _canonical_json_bytes(existing_manifest.get(key)) != (
                    _canonical_json_bytes(manifest_payload.get(key))
                ):
                    raise ValueError(f"existing job manifest {key} mismatch")
            manifest_status = "validated_existing"
        else:
            _write_json_no_replace(manifest_path, manifest_payload)
            manifest_status = "created"

    result["status"] = "complete"
    result["references"] = reference_results
    result["cadence_profile"] = {
        "status": profile_status,
        "path": str(profile_path),
        "file_sha256": _sha256_file_stable(profile_path),
        "profile_content_sha256": profile.profile_content_sha256,
        "reference_count": profile.reference_count,
    }
    result["job_manifest"] = {
        "status": manifest_status,
        "path": str(manifest_path),
        "file_sha256": _sha256_file_stable(manifest_path),
    }
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m crosscut.human_reference_bootstrap",
        description=(
            "build exact-five human hard-cut audits, immutable boundary "
            "provenance, and a cadence profile against an exact-ten source pool"
        ),
    )
    parser.add_argument("--reference-directory", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--artist", required=True)
    parser.add_argument("--song", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
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
        result = run_human_reference_bootstrap(
            reference_directory=arguments.reference_directory,
            source_manifest_path=arguments.source_manifest,
            output_directory=arguments.output_directory,
            artist=arguments.artist,
            song=arguments.song,
            ffmpeg_bin=arguments.ffmpeg,
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
    "AUDIT_PROVENANCE_ANALYSIS_BASIS",
    "EXPECTED_REFERENCE_COUNT",
    "EXPECTED_SOURCE_COUNT",
    "HumanCutAuditResult",
    "HumanReferenceBootstrapConfig",
    "HumanReferenceBootstrapError",
    "build_human_cut_quality_audit",
    "discover_reference_media",
    "run_human_reference_bootstrap",
]
