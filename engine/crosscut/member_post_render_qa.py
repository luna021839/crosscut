"""Mandatory final-media gate for automatic IVE member crosscuts.

The group delivery gate intentionally consumes group-only candidate-selection
evidence.  Automatic member fancams have a different product contract: an
uncropped portrait render, exact all-eligible-source coverage, and pixel
provenance back to the synchronized source clock.  Their retained source
switches also need the same renderer-bound actual-PTS motion-phase proof as a
GROUP delivery.  This module evaluates those contracts without pretending that
a member render is a GROUP artifact set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence

from .cut_local_sync import (
    MEMBER_ACTIVATION as MEMBER_CUT_LOCAL_SYNC_ACTIVATION,
    validate_cut_local_sync_evidence,
)
from .post_render_perceptual_quality import (
    ANALYSIS_BASIS as PERCEPTUAL_ANALYSIS_BASIS,
    PerceptualQualityConfig,
    evaluate_post_render_perceptual_quality,
)
from .render import RENDER_PLAN_SOURCE_INDEX_DOMAIN


SCHEMA_VERSION = 1
ANALYSIS_BASIS = "automatic_ive_member_final_media_and_pixel_provenance_qa_v1"
ARTIFACT_FINGERPRINT_CONTRACT = (
    "automatic_ive_member_analysis_plan_video_content_sha256_v1"
)
RENDER_RECEIPT_CONTRACT = (
    "automatic_ive_member_exact_source_time_pixel_provenance_receipt_v1"
)
AUTOMATIC_IVE_MEMBER_POST_RENDER_QA_CONTRACT = (
    "automatic_ive_member_mandatory_post_render_qa_v1"
)
AUTOMATIC_IVE_MEMBER_APP_RESULT_EXPOSURE_CONTRACT = (
    "automatic_ive_member_app_result_exposure_v1"
)

_REQUEST_SOURCE_INDEX_DOMAIN = "request_source_index"
_ELIGIBLE_SOURCE_INDEX_DOMAIN = "eligible_source_index"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _mapping(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _sequence(value: Any) -> list[Any] | None:
    return list(value) if isinstance(value, list) else None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _numbers_close(left: Any, right: Any, *, tolerance: float = 1e-6) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    return bool(
        left_number is not None
        and right_number is not None
        and math.isclose(
            left_number,
            right_number,
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    )


def _parse_rate(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        return result if math.isfinite(result) and result > 0.0 else None
    if not isinstance(value, str) or not value or value in {"N/A", "0/0"}:
        return None
    try:
        numerator, denominator = value.split("/", 1)
        result = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def _resolved_declared_path(value: Any, *, parent: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("member render artifact path is missing")
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else parent / candidate).resolve()


@dataclass(frozen=True, slots=True)
class AutomaticMemberPostRenderArtifactSet:
    analysis_path: Path
    render_plan_path: Path
    video_path: Path

    def __post_init__(self) -> None:
        analysis = Path(self.analysis_path).resolve()
        plan = Path(self.render_plan_path).resolve()
        video = Path(self.video_path).resolve()
        object.__setattr__(self, "analysis_path", analysis)
        object.__setattr__(self, "render_plan_path", plan)
        object.__setattr__(self, "video_path", video)
        if not analysis.name.endswith("-analysis.json"):
            raise ValueError("member QA analysis filename is invalid")
        if not plan.name.endswith("-render-plan.json"):
            raise ValueError("member QA render-plan filename is invalid")
        if not video.name.endswith("-member-crosscut.mp4"):
            raise ValueError("member QA video filename is invalid")
        if len({analysis.parent, plan.parent, video.parent}) != 1:
            raise ValueError("member QA artifacts must share one output directory")
        analysis_base = analysis.name[: -len("-analysis.json")]
        video_base = video.name[: -len("-member-crosscut.mp4")]
        plan_base = plan.name[: -len("-render-plan.json")]
        if not (
            video_base == plan_base
            and video_base.startswith(f"{analysis_base}-")
        ):
            raise ValueError("member QA filenames do not share one timestamp-bound base")

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        manifest: Mapping[str, Any],
    ) -> "AutomaticMemberPostRenderArtifactSet":
        analysis_path = Path(manifest_path).resolve()
        render = _mapping(manifest.get("render"))
        if render is None:
            raise ValueError("automatic member manifest lacks render evidence")
        return cls(
            analysis_path=analysis_path,
            render_plan_path=_resolved_declared_path(
                render.get("plan_path"), parent=analysis_path.parent
            ),
            video_path=_resolved_declared_path(
                render.get("video_path"), parent=analysis_path.parent
            ),
        )

    @property
    def output_directory(self) -> Path:
        return self.analysis_path.parent

    @property
    def default_report_path(self) -> Path:
        base = self.video_path.name[: -len("-member-crosscut.mp4")]
        return self.output_directory / f"{base}-member-post-render-qa.json"

    def to_dict(self) -> dict[str, str]:
        return {
            "analysis_path": str(self.analysis_path),
            "render_plan_path": str(self.render_plan_path),
            "video_path": str(self.video_path),
        }


@dataclass(frozen=True, slots=True)
class AutomaticMemberPostRenderQAConfig:
    ffprobe: str = "ffprobe"
    ffmpeg: str = "ffmpeg"
    tool_timeout_seconds: float = 900.0
    fps_tolerance: float = 0.05
    duration_tolerance_frames: float = 1.0
    maximum_sync_residual_p95_frames: float = 0.5
    minimum_structure_samples_per_eligible_source: int = 1
    perceptual_quality: PerceptualQualityConfig = field(
        default_factory=PerceptualQualityConfig
    )

    def __post_init__(self) -> None:
        if not isinstance(self.perceptual_quality, PerceptualQualityConfig):
            raise TypeError("perceptual_quality must be PerceptualQualityConfig")
        if (
            not math.isfinite(self.tool_timeout_seconds)
            or self.tool_timeout_seconds <= 0.0
            or not math.isfinite(self.fps_tolerance)
            or self.fps_tolerance < 0.0
            or not math.isfinite(self.duration_tolerance_frames)
            or self.duration_tolerance_frames < 0.0
            or not math.isfinite(self.maximum_sync_residual_p95_frames)
            or self.maximum_sync_residual_p95_frames <= 0.0
        ):
            raise ValueError("member QA numeric configuration is invalid")
        if self.minimum_structure_samples_per_eligible_source < 1:
            raise ValueError("member QA requires at least one source proof sample")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ffprobe": self.ffprobe,
            "ffmpeg": self.ffmpeg,
            "tool_timeout_seconds": self.tool_timeout_seconds,
            "fps_tolerance": self.fps_tolerance,
            "duration_tolerance_frames": self.duration_tolerance_frames,
            "maximum_sync_residual_p95_frames": (
                self.maximum_sync_residual_p95_frames
            ),
            "minimum_structure_samples_per_eligible_source": (
                self.minimum_structure_samples_per_eligible_source
            ),
            "perceptual_quality": self.perceptual_quality.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class AutomaticMemberPostRenderQAReport:
    artifact_set: AutomaticMemberPostRenderArtifactSet
    passed: bool
    issues: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, Any]
    input_artifact_fingerprints: Mapping[str, Any]
    final_artifact_fingerprints: Mapping[str, Any]
    render_receipt: Mapping[str, Any]
    policy: AutomaticMemberPostRenderQAConfig
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "analysis": ANALYSIS_BASIS,
            "created_at": self.created_at.astimezone(UTC).isoformat(),
            "artifact_set": self.artifact_set.to_dict(),
            "policy": self.policy.to_dict(),
            "verdict": self.verdict,
            "passed": self.passed,
            "issue_count": len(self.issues),
            "issues": [dict(item) for item in self.issues],
            "input_artifact_fingerprints": dict(
                self.input_artifact_fingerprints
            ),
            "final_artifact_fingerprints": dict(
                self.final_artifact_fingerprints
            ),
            "metrics": dict(self.metrics),
            "render_receipt": dict(self.render_receipt),
        }

    def write_atomic(self, output_path: Path | None = None) -> Path:
        destination = Path(output_path or self.artifact_set.default_report_path)
        protected = {
            self.artifact_set.analysis_path,
            self.artifact_set.render_plan_path,
            self.artifact_set.video_path,
        }
        if destination.resolve() in protected:
            raise ValueError("member QA report cannot overwrite an input artifact")
        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            temporary.unlink(missing_ok=True)
            raise
        return destination.resolve()


class _Issues:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self._codes: set[str] = set()

    def add(
        self,
        code: str,
        message: str,
        *,
        path: str,
        value: Any = None,
        threshold: Any = None,
    ) -> None:
        if code in self._codes:
            return
        self._codes.add(code)
        self.items.append(
            {
                "code": code,
                "severity": "fail",
                "message": message,
                "path": path,
                "value": value,
                "threshold": threshold,
            }
        )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def _fingerprint_file(path: Path) -> dict[str, Any]:
    source = Path(path)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        path_before = source.lstat()
        if not stat.S_ISREG(path_before.st_mode):
            raise ValueError(f"artifact is not a regular file: {source.name}")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(source, flags)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if (path_before.st_dev, path_before.st_ino) != (
                before.st_dev,
                before.st_ino,
            ):
                raise ValueError(f"artifact path changed before open: {source.name}")
            if before.st_size <= 0:
                raise ValueError(f"artifact is empty: {source.name}")
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
            after = os.fstat(stream.fileno())
        path_after = source.lstat()
    except OSError as exc:
        raise ValueError(f"could not fingerprint {source.name}: {exc}") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    path_identity_after = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    if (
        not stat.S_ISREG(path_after.st_mode)
        or identity_before != identity_after
        or path_identity_after != identity_after
        or byte_count != after.st_size
    ):
        raise ValueError(f"artifact changed while hashing: {source.name}")
    return {
        "path": str(source.resolve()),
        "size_bytes": byte_count,
        "sha256": digest.hexdigest(),
    }


def fingerprint_automatic_member_artifacts(
    artifacts: AutomaticMemberPostRenderArtifactSet,
) -> dict[str, Any]:
    return {
        "contract": ARTIFACT_FINGERPRINT_CONTRACT,
        "analysis": _fingerprint_file(artifacts.analysis_path),
        "render_plan": _fingerprint_file(artifacts.render_plan_path),
        "video": _fingerprint_file(artifacts.video_path),
    }


def _source_identity(path: Path) -> dict[str, Any]:
    source = Path(path)
    state = source.lstat()
    if not stat.S_ISREG(state.st_mode):
        raise ValueError(f"render source is not a regular file: {source}")
    return {
        "path": str(source.resolve()),
        "device": state.st_dev,
        "inode": state.st_ino,
        "size_bytes": state.st_size,
        "mtime_ns": state.st_mtime_ns,
        "ctime_ns": state.st_ctime_ns,
    }


def _path_matches(value: Any, path: Path, *, parent: Path) -> bool:
    try:
        return _resolved_declared_path(value, parent=parent) == path.resolve()
    except ValueError:
        return False


def automatic_member_post_render_qa_required(
    analysis: Mapping[str, Any],
) -> bool:
    """Resolve activation only from persisted request and render evidence."""

    request = _mapping(analysis.get("request")) or {}
    render = _mapping(analysis.get("render")) or {}
    return bool(
        str(request.get("group", "")).strip().upper() == "IVE"
        and str(request.get("source_mode", "")).strip().lower()
        == "automatic"
        and str(request.get("edit_mode", "")).strip().lower() == "member"
        and render.get("direct_full_frame") is True
    )


def _validate_member_contract(
    artifacts: AutomaticMemberPostRenderArtifactSet,
    analysis: Mapping[str, Any],
    plan: Mapping[str, Any],
    issues: _Issues,
) -> tuple[list[int], list[str]]:
    request = _mapping(analysis.get("request")) or {}
    render = _mapping(analysis.get("render")) or {}
    member_direct = _mapping(analysis.get("member_direct_full_frame"))
    sources = _sequence(plan.get("sources")) or []
    request_sources = _sequence(request.get("sources")) or []

    checks: dict[str, bool] = {
        "activation_scope": automatic_member_post_render_qa_required(analysis),
        "render_direct_full_frame": render.get("direct_full_frame") is True,
        "member_direct_present_enabled": bool(
            member_direct is not None and member_direct.get("enabled") is True
        ),
        "render_video_path": _path_matches(
            render.get("video_path"),
            artifacts.video_path,
            parent=artifacts.output_directory,
        ),
        "render_plan_path": _path_matches(
            render.get("plan_path"),
            artifacts.render_plan_path,
            parent=artifacts.output_directory,
        ),
        "plan_output_path": _path_matches(
            plan.get("output"),
            artifacts.video_path,
            parent=artifacts.output_directory,
        ),
        "request_plan_sources_exact": bool(
            request_sources
            and sources == request_sources
            and all(isinstance(value, str) and value for value in sources)
        ),
        "portrait_output": (
            request.get("aspect_ratio") == "9:16"
            and plan.get("aspect_ratio") == "9:16"
        ),
    }
    for name, passed in checks.items():
        if not passed:
            issues.add(
                f"member_contract_{name}",
                "Automatic member render contract is missing or inconsistent.",
                path=f"member_contract.{name}",
                value=passed,
                threshold=True,
            )

    eligibility = _sequence(analysis.get("alignment_eligibility"))
    eligible: list[int] = []
    eligibility_valid = bool(eligibility is not None and len(eligibility) == len(sources))
    if eligibility is not None:
        for index, raw in enumerate(eligibility):
            item = _mapping(raw)
            valid = bool(
                item is not None
                and item.get("index_domain") == _REQUEST_SOURCE_INDEX_DOMAIN
                and item.get("source_index") == index
                and item.get("source") == sources[index]
                and isinstance(item.get("render_eligible"), bool)
            )
            eligibility_valid = eligibility_valid and valid
            if valid and item["render_eligible"] is True:
                eligible.append(index)
    if not eligibility_valid or not eligible:
        issues.add(
            "member_alignment_eligibility_not_canonical",
            "Alignment eligibility is not the exact request-source-domain list.",
            path="alignment_eligibility",
            value=eligibility,
        )

    mapping = _mapping(analysis.get("source_index_mapping")) or {}
    request_to_eligible: list[int | None] = [None] * len(sources)
    entries: list[dict[str, Any]] = []
    for eligible_index, request_index in enumerate(eligible):
        request_to_eligible[request_index] = eligible_index
        entries.append(
            {
                "eligible_source_index": eligible_index,
                "request_source_index": request_index,
                "source": sources[request_index],
            }
        )
    mapping_valid = bool(
        mapping.get("request_index_domain") == _REQUEST_SOURCE_INDEX_DOMAIN
        and mapping.get("eligible_index_domain") == _ELIGIBLE_SOURCE_INDEX_DOMAIN
        and mapping.get("render_plan_index_domain")
        == RENDER_PLAN_SOURCE_INDEX_DOMAIN
        and mapping.get("domain_equivalences")
        == {RENDER_PLAN_SOURCE_INDEX_DOMAIN: _REQUEST_SOURCE_INDEX_DOMAIN}
        and mapping.get("eligible_to_request") == eligible
        and mapping.get("request_to_eligible") == request_to_eligible
        and mapping.get("eligible_sources") == entries
    )
    if not mapping_valid:
        issues.add(
            "member_source_index_mapping_not_canonical",
            "Source-index mapping does not exactly describe the eligible request sources.",
            path="source_index_mapping",
            value=mapping,
        )

    segments = _sequence(plan.get("segments")) or []
    segment_indices = [
        _integer((_mapping(raw) or {}).get("source_index")) for raw in segments
    ]
    actual_used = sorted(
        {value for value in segment_indices if value is not None}
    )
    usage = _mapping(plan.get("source_usage")) or {}
    derived_counts = [0] * len(sources)
    derived_durations = [0.0] * len(sources)
    segments_valid = bool(segments and len(segment_indices) == len(segments))
    for raw, source_index in zip(segments, segment_indices, strict=True):
        segment = _mapping(raw) or {}
        duration = _number(segment.get("duration"))
        if (
            source_index is None
            or not 0 <= source_index < len(sources)
            or duration is None
            or duration <= 0.0
        ):
            segments_valid = False
            continue
        derived_counts[source_index] += 1
        derived_durations[source_index] += duration
    derived_usage = {
        "index_domain": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
        "distinct_source_count": len(actual_used),
        "used_source_indices": actual_used,
        "total_segment_count": len(segments),
        "total_duration_seconds": sum(derived_durations),
        "sources": [
            {
                "source_index": index,
                "segment_count": derived_counts[index],
                "duration_seconds": derived_durations[index],
            }
            for index in range(len(sources))
        ],
    }
    usage_valid = bool(
        segments_valid
        and actual_used == eligible
        and usage == derived_usage
        and member_direct is not None
        and member_direct.get("eligible_request_source_indices") == eligible
        and member_direct.get("required_used_request_source_indices") == eligible
        and member_direct.get("minimum_required_distinct_source_count")
        == len(eligible)
        and member_direct.get("final_source_usage") == derived_usage
        and render.get("final_source_usage") == derived_usage
    )
    if not usage_valid:
        issues.add(
            "member_all_eligible_sources_not_exactly_used",
            "The final member plan does not use every and only eligible source.",
            path="render_plan.source_usage",
            value={"eligible": eligible, "used": actual_used, "usage": usage},
            threshold={"used": eligible},
        )

    direct_segments = (
        _sequence(member_direct.get("segments")) if member_direct is not None else None
    )
    direct_segment_binding = bool(
        direct_segments is not None
        and len(direct_segments) == len(segments)
        and all(
            (_mapping(direct) or {}).get("segment_index") == index
            and (_mapping(direct) or {}).get("request_source_index")
            == segment_indices[index]
            and (_mapping(direct) or {}).get("crop_applied") is False
            and _numbers_close(
                (_mapping(direct) or {}).get("source_local_start"),
                (_mapping(segments[index]) or {}).get("source_local_start"),
            )
            and _numbers_close(
                (_mapping(direct) or {}).get("source_local_end"),
                (_mapping(segments[index]) or {}).get("source_local_end"),
            )
            for index, direct in enumerate(direct_segments)
        )
    )
    if not direct_segment_binding:
        issues.add(
            "member_direct_segments_not_bound_to_render_plan",
            "Direct-member source clocks do not exactly match the render plan.",
            path="member_direct_full_frame.segments",
        )
    return eligible, [str(value) for value in sources if isinstance(value, str)]


def _member_cut_local_sync_metrics(
    analysis: Mapping[str, Any],
    plan: Mapping[str, Any],
    config: AutomaticMemberPostRenderQAConfig,
    issues: _Issues,
) -> dict[str, Any]:
    """Revalidate the renderer-bound PTS/time-warp proof for every source cut."""

    render = _mapping(analysis.get("render")) or {}
    evidence = _mapping(render.get("cut_local_sync"))
    raw_timewarps = _sequence(analysis.get("local_timewarps"))
    if raw_timewarps is None:
        issues.add(
            "member_cut_local_sync_timewarp_evidence_missing",
            "Automatic member cut-local QA lacks serialized source clocks.",
            path="local_timewarps",
        )
    try:
        validated = validate_cut_local_sync_evidence(
            evidence,
            plan=plan,
            timewarp_payloads=raw_timewarps or [],
            expected_activation=MEMBER_CUT_LOCAL_SYNC_ACTIVATION,
            maximum_residual_p95_frames=(
                config.maximum_sync_residual_p95_frames
            ),
        )
    except Exception:
        validated = {
            "available": evidence is not None,
            "hash_bindings_valid": False,
            "errors": ["cut_local_sync_validation_failed"],
            "unknown_reason_counts": {},
            "passed": False,
        }
    metrics = dict(validated)
    metrics.update(
        {
            "required": True,
            "activation": MEMBER_CUT_LOCAL_SYNC_ACTIVATION,
            "maximum_allowed_residual_p95_frames": (
                config.maximum_sync_residual_p95_frames
            ),
        }
    )
    errors = _sequence(metrics.get("errors")) or []
    unknown_count = _integer(metrics.get("unknown_boundary_count")) or 0
    residual_p95 = _number(metrics.get("residual_p95_frames"))
    if not bool(metrics.get("available")):
        issues.add(
            "member_cut_local_sync_evidence_missing",
            "Automatic member render lacks retained-cut local sync evidence.",
            path="render.cut_local_sync",
        )
    if errors:
        issues.add(
            "member_cut_local_sync_binding_invalid",
            "Member cut-local sync evidence is not bound to the final plan and clocks.",
            path="render.cut_local_sync",
            value=errors,
            threshold=[],
        )
    if unknown_count > 0:
        issues.add(
            "member_cut_local_sync_unknown_boundaries",
            "At least one retained member source switch lacks unique motion timing evidence.",
            path="render.cut_local_sync.per_boundary",
            value={
                "unknown_boundary_count": unknown_count,
                "unknown_reason_counts": metrics.get("unknown_reason_counts"),
            },
            threshold={"unknown_boundary_count": 0},
        )
    if (
        residual_p95 is not None
        and residual_p95
        > config.maximum_sync_residual_p95_frames + 1e-12
    ):
        issues.add(
            "member_cut_local_sync_residual_exceeds_half_frame",
            "Member retained-cut motion phase exceeds the half-frame target.",
            path="render.cut_local_sync.residual_p95_frames",
            value=residual_p95,
            threshold=config.maximum_sync_residual_p95_frames,
        )
    if metrics.get("passed") is not True:
        issues.add(
            "member_cut_local_sync_required_gate_failed",
            "Automatic direct-member render did not satisfy cut-local synchronization QA.",
            path="render.cut_local_sync",
            value=metrics.get("passed"),
            threshold=True,
        )
    return metrics


RunCommand = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]
PerceptualEvaluator = Callable[..., Mapping[str, Any]]


def _run_command(
    command: Sequence[str], timeout: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), check=False, capture_output=True, text=True, timeout=timeout
    )


def _probe_and_decode(
    artifacts: AutomaticMemberPostRenderArtifactSet,
    plan: Mapping[str, Any],
    config: AutomaticMemberPostRenderQAConfig,
    runner: RunCommand,
    issues: _Issues,
) -> dict[str, Any]:
    probe_command = (
        config.ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(artifacts.video_path),
    )
    decode_command = (
        config.ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(artifacts.video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-f",
        "null",
        "-",
    )
    probe_payload: dict[str, Any] | None = None
    probe_result: subprocess.CompletedProcess[str] | None = None
    decode_result: subprocess.CompletedProcess[str] | None = None
    try:
        probe_result = runner(probe_command, config.tool_timeout_seconds)
        if probe_result.returncode == 0:
            value = json.loads(probe_result.stdout or "")
            probe_payload = value if isinstance(value, dict) else None
        if probe_result.returncode != 0 or probe_payload is None:
            issues.add(
                "member_ffprobe_failed",
                "ffprobe could not inspect the final member MP4.",
                path="tool_checks.ffprobe",
                value=(probe_result.stderr or "")[-1000:],
                threshold=0,
            )
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        issues.add(
            "member_ffprobe_failed",
            "ffprobe could not inspect the final member MP4.",
            path="tool_checks.ffprobe",
            value=f"{type(exc).__name__}: {exc}",
        )
    try:
        decode_result = runner(decode_command, config.tool_timeout_seconds)
        if decode_result.returncode != 0:
            issues.add(
                "member_full_decode_failed",
                "The final member MP4 did not fully decode without errors.",
                path="tool_checks.full_decode",
                value=(decode_result.stderr or "")[-1000:],
                threshold=0,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        issues.add(
            "member_full_decode_failed",
            "The final member MP4 full-decode command failed.",
            path="tool_checks.full_decode",
            value=f"{type(exc).__name__}: {exc}",
        )

    streams = _sequence((probe_payload or {}).get("streams")) or []
    video_streams = [
        _mapping(item) or {}
        for item in streams
        if (_mapping(item) or {}).get("codec_type") == "video"
    ]
    audio_streams = [
        _mapping(item) or {}
        for item in streams
        if (_mapping(item) or {}).get("codec_type") == "audio"
    ]
    video = video_streams[0] if len(video_streams) == 1 else {}
    expected_fps = _number(plan.get("fps"))
    actual_fps = _parse_rate(video.get("avg_frame_rate"))
    if actual_fps is None:
        actual_fps = _parse_rate(video.get("r_frame_rate"))
    expected_frames = _integer(plan.get("output_frame_count"))
    raw_frames = video.get("nb_read_frames")
    actual_frames = (
        int(raw_frames)
        if isinstance(raw_frames, str) and raw_frames.isdigit()
        else _integer(raw_frames)
    )
    expected_duration = _number(plan.get("render_duration"))
    actual_duration = _number(video.get("duration"))
    if actual_duration is None:
        raw_duration = (_mapping((probe_payload or {}).get("format")) or {}).get(
            "duration"
        )
        try:
            actual_duration = float(raw_duration)
        except (TypeError, ValueError):
            actual_duration = None
    media_contract = bool(
        len(video_streams) == 1
        and len(audio_streams) >= 1
        and video.get("width") == plan.get("width")
        and video.get("height") == plan.get("height")
        and isinstance(video.get("codec_name"), str)
        and bool(video.get("codec_name"))
        and expected_fps is not None
        and actual_fps is not None
        and abs(actual_fps - expected_fps) <= config.fps_tolerance
        and expected_frames is not None
        and actual_frames == expected_frames
        and expected_duration is not None
        and actual_duration is not None
        and abs(actual_duration - expected_duration)
        <= config.duration_tolerance_frames / expected_fps
    )
    if not media_contract:
        issues.add(
            "member_ffprobe_plan_contract_mismatch",
            "Final member media metadata does not match the bound render plan.",
            path="tool_checks.ffprobe.plan_binding",
            value={
                "video_stream_count": len(video_streams),
                "audio_stream_count": len(audio_streams),
                "width": video.get("width"),
                "height": video.get("height"),
                "fps": actual_fps,
                "frames": actual_frames,
                "duration": actual_duration,
            },
            threshold={
                "width": plan.get("width"),
                "height": plan.get("height"),
                "fps": expected_fps,
                "frames": expected_frames,
                "duration": expected_duration,
            },
        )
    return {
        "ffprobe_command": list(probe_command),
        "ffprobe_returncode": (
            probe_result.returncode if probe_result is not None else None
        ),
        "full_decode_command": list(decode_command),
        "full_decode_returncode": (
            decode_result.returncode if decode_result is not None else None
        ),
        "full_decode_passed": bool(
            decode_result is not None and decode_result.returncode == 0
        ),
        "plan_binding_passed": media_contract,
        "probe": probe_payload,
    }


def _perceptual_and_receipt(
    artifacts: AutomaticMemberPostRenderArtifactSet,
    plan: Mapping[str, Any],
    eligible: Sequence[int],
    source_paths: Sequence[str],
    fingerprints: Mapping[str, Any],
    source_identities_before: Mapping[int, Mapping[str, Any]],
    config: AutomaticMemberPostRenderQAConfig,
    evaluator: PerceptualEvaluator,
    issues: _Issues,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        raw = evaluator(
            artifacts.video_path,
            plan,
            config=config.perceptual_quality,
        )
        payload = _mapping(raw)
    except Exception as exc:
        payload = None
        issues.add(
            "member_perceptual_evaluation_failed",
            "Final member perceptual and exact-source audit failed.",
            path="perceptual_quality",
            value=f"{type(exc).__name__}: {exc}",
        )
    if payload is None:
        payload = {}
    contract_valid = bool(
        payload.get("schema_version") == 2
        and payload.get("analysis") == PERCEPTUAL_ANALYSIS_BASIS
        and payload.get("verdict") in {"PASS", "CONDITIONAL", "FAIL"}
        and payload.get("passed") is (payload.get("verdict") == "PASS")
        and isinstance(payload.get("issues"), list)
        and (payload.get("verdict") == "PASS")
        is (len(payload.get("issues")) == 0)
        and isinstance(payload.get("metrics"), Mapping)
    )
    if not contract_valid or payload.get("passed") is not True:
        issues.add(
            "member_perceptual_quality_not_pass",
            "Final member perceptual QA and source retention must return PASS.",
            path="perceptual_quality",
            value={
                "contract_valid": contract_valid,
                "verdict": payload.get("verdict"),
                "passed": payload.get("passed"),
            },
            threshold={"verdict": "PASS", "passed": True},
        )

    metrics = _mapping(payload.get("metrics")) or {}
    retention = _mapping(metrics.get("exact_source_retention")) or {}
    raw_per_source = _sequence(retention.get("per_source")) or []
    per_source_by_index: dict[int, dict[str, Any]] = {}
    duplicate = False
    for raw_item in raw_per_source:
        item = _mapping(raw_item) or {}
        source_index = _integer(item.get("source_index"))
        if source_index is None or source_index in per_source_by_index:
            duplicate = True
            continue
        per_source_by_index[source_index] = item
    raw_per_segment = _sequence(retention.get("per_segment")) or []
    per_source_segments: dict[int, list[dict[str, Any]]] = {
        index: [] for index in eligible
    }
    for raw_item in raw_per_segment:
        item = _mapping(raw_item) or {}
        source_index = _integer(item.get("source_index"))
        if source_index in per_source_segments:
            per_source_segments[source_index].append(item)

    receipt_sources: list[dict[str, Any]] = []
    source_proof_passed = not duplicate and set(per_source_by_index) == set(eligible)
    for source_index in eligible:
        item = per_source_by_index.get(source_index, {})
        structure_count = _integer(item.get("structure_comparable_sample_count"))
        decoded_count = _integer(item.get("decoded_source_sample_count"))
        bound_segments = [
            segment
            for segment in per_source_segments.get(source_index, [])
            if segment.get("structure_binding_passed") is True
            and (_integer(segment.get("structure_comparable_sample_count")) or 0)
            >= 1
        ]
        proof = bool(
            structure_count is not None
            and structure_count
            >= config.minimum_structure_samples_per_eligible_source
            and decoded_count is not None
            and decoded_count >= structure_count
            and bound_segments
        )
        source_proof_passed = source_proof_passed and proof
        receipt_sources.append(
            {
                "request_source_index": source_index,
                "source_path": (
                    source_paths[source_index]
                    if 0 <= source_index < len(source_paths)
                    else None
                ),
                "source_file_identity": dict(
                    source_identities_before.get(source_index, {})
                ),
                "decoded_source_sample_count": decoded_count,
                "safe_interior_structure_sample_count": structure_count,
                "bound_segment_indices": [
                    segment.get("segment_index") for segment in bound_segments
                ],
                "exact_structure_similarity": item.get(
                    "exact_structure_similarity"
                ),
                "passed": proof,
            }
        )
    if not source_proof_passed:
        issues.add(
            "member_pixel_provenance_incomplete",
            "Every eligible source needs an exact mapped-time safe-interior pixel proof.",
            path="render_receipt.sources",
            value=receipt_sources,
            threshold={
                "eligible_source_indices": list(eligible),
                "minimum_samples_per_source": (
                    config.minimum_structure_samples_per_eligible_source
                ),
            },
        )
    receipt = {
        "contract": RENDER_RECEIPT_CONTRACT,
        "artifact_fingerprint_contract": ARTIFACT_FINGERPRINT_CONTRACT,
        "artifact_fingerprints": dict(fingerprints),
        "source_index_domain": _REQUEST_SOURCE_INDEX_DOMAIN,
        "eligible_source_indices": list(eligible),
        "mapping_basis": retention.get("mapping_basis"),
        "comparison_domain": retention.get("comparison_domain"),
        "segment_sampling_policy": retention.get("segment_sampling_policy"),
        "minimum_safe_interior_structure_samples_per_source": (
            config.minimum_structure_samples_per_eligible_source
        ),
        "sources": receipt_sources,
        "all_eligible_sources_pixel_proven": source_proof_passed,
        "raw_frames_persisted": False,
        "passed": source_proof_passed,
    }
    return payload, receipt


def evaluate_automatic_member_post_render_qa(
    artifacts: AutomaticMemberPostRenderArtifactSet,
    *,
    config: AutomaticMemberPostRenderQAConfig | None = None,
    runner: RunCommand = _run_command,
    perceptual_evaluator: PerceptualEvaluator = (
        evaluate_post_render_perceptual_quality
    ),
    created_at: datetime | None = None,
) -> AutomaticMemberPostRenderQAReport:
    """Evaluate one pending automatic member render without mutating inputs."""

    settings = config or AutomaticMemberPostRenderQAConfig()
    issues = _Issues()
    metrics: dict[str, Any] = {}
    initial_fingerprints: dict[str, Any] = {}
    final_fingerprints: dict[str, Any] = {}
    receipt: dict[str, Any] = {
        "contract": RENDER_RECEIPT_CONTRACT,
        "passed": False,
        "sources": [],
    }
    try:
        initial_fingerprints = fingerprint_automatic_member_artifacts(artifacts)
    except ValueError as exc:
        issues.add(
            "member_artifact_fingerprint_unavailable",
            "Pending member artifacts could not be content-fingerprinted.",
            path="artifact_set",
            value=str(exc),
        )
    try:
        analysis = _load_json_object(artifacts.analysis_path)
        plan = _load_json_object(artifacts.render_plan_path)
    except ValueError as exc:
        issues.add(
            "member_artifact_json_invalid",
            "Pending member analysis or render plan is invalid.",
            path="artifact_set",
            value=str(exc),
        )
        analysis, plan = {}, {}

    eligible, source_paths = _validate_member_contract(
        artifacts, analysis, plan, issues
    )
    cut_local_sync = _member_cut_local_sync_metrics(
        analysis,
        plan,
        settings,
        issues,
    )
    metrics["cut_local_sync"] = cut_local_sync
    source_identities_before: dict[int, Mapping[str, Any]] = {}
    for source_index in eligible:
        try:
            source_identities_before[source_index] = _source_identity(
                Path(source_paths[source_index])
            )
        except (IndexError, OSError, ValueError) as exc:
            issues.add(
                f"member_source_identity_unavailable_{source_index}",
                "An eligible source file is unavailable for provenance review.",
                path=f"render_plan.sources[{source_index}]",
                value=f"{type(exc).__name__}: {exc}",
            )

    metrics["tool_checks"] = _probe_and_decode(
        artifacts, plan, settings, runner, issues
    )
    perceptual, receipt = _perceptual_and_receipt(
        artifacts,
        plan,
        eligible,
        source_paths,
        initial_fingerprints,
        source_identities_before,
        settings,
        perceptual_evaluator,
        issues,
    )
    metrics["perceptual_quality"] = perceptual
    receipt["cut_local_sync"] = dict(cut_local_sync)

    source_identity_stable = True
    source_identities_after: dict[int, Mapping[str, Any]] = {}
    for source_index in eligible:
        try:
            source_identities_after[source_index] = _source_identity(
                Path(source_paths[source_index])
            )
        except (IndexError, OSError, ValueError):
            source_identity_stable = False
    source_identity_stable = bool(
        source_identity_stable
        and source_identities_before == source_identities_after
        and len(source_identities_after) == len(eligible)
    )
    if not source_identity_stable:
        issues.add(
            "member_source_files_changed_during_pixel_review",
            "At least one source file changed during mapped-time pixel review.",
            path="render_receipt.source_file_identity",
            value={
                "before": source_identities_before,
                "after": source_identities_after,
            },
        )
    try:
        final_fingerprints = fingerprint_automatic_member_artifacts(artifacts)
    except ValueError as exc:
        issues.add(
            "member_artifact_final_fingerprint_unavailable",
            "Member artifacts could not be re-fingerprinted after QA.",
            path="artifact_set",
            value=str(exc),
        )
    artifact_stable = bool(
        initial_fingerprints
        and final_fingerprints
        and initial_fingerprints == final_fingerprints
    )
    if not artifact_stable:
        issues.add(
            "member_artifacts_changed_during_qa",
            "Analysis, render plan, or final MP4 changed during member QA.",
            path="artifact_set",
            value={"before": initial_fingerprints, "after": final_fingerprints},
        )
    receipt["source_file_identities_stable"] = source_identity_stable
    receipt["artifact_content_stable"] = artifact_stable
    receipt["passed"] = bool(
        receipt.get("all_eligible_sources_pixel_proven") is True
        and cut_local_sync.get("passed") is True
        and source_identity_stable
        and artifact_stable
    )
    metrics["content_stability"] = {
        "artifact_content_stable": artifact_stable,
        "source_file_identities_stable": source_identity_stable,
    }
    passed = not issues.items and receipt["passed"] is True
    return AutomaticMemberPostRenderQAReport(
        artifact_set=artifacts,
        passed=passed,
        issues=tuple(issues.items),
        metrics=metrics,
        input_artifact_fingerprints=initial_fingerprints,
        final_artifact_fingerprints=final_fingerprints,
        render_receipt=receipt,
        policy=settings,
        created_at=created_at or datetime.now(UTC),
    )


def evaluate_and_write_automatic_member_post_render_qa(
    artifacts: AutomaticMemberPostRenderArtifactSet,
    *,
    output_path: Path | None = None,
    config: AutomaticMemberPostRenderQAConfig | None = None,
    runner: RunCommand = _run_command,
    perceptual_evaluator: PerceptualEvaluator = (
        evaluate_post_render_perceptual_quality
    ),
    created_at: datetime | None = None,
) -> tuple[AutomaticMemberPostRenderQAReport, Path]:
    report = evaluate_automatic_member_post_render_qa(
        artifacts,
        config=config,
        runner=runner,
        perceptual_evaluator=perceptual_evaluator,
        created_at=created_at,
    )
    return report, report.write_atomic(output_path)


__all__ = [
    "ANALYSIS_BASIS",
    "ARTIFACT_FINGERPRINT_CONTRACT",
    "AUTOMATIC_IVE_MEMBER_APP_RESULT_EXPOSURE_CONTRACT",
    "AUTOMATIC_IVE_MEMBER_POST_RENDER_QA_CONTRACT",
    "AutomaticMemberPostRenderArtifactSet",
    "AutomaticMemberPostRenderQAConfig",
    "AutomaticMemberPostRenderQAReport",
    "RENDER_RECEIPT_CONTRACT",
    "SCHEMA_VERSION",
    "automatic_member_post_render_qa_required",
    "evaluate_and_write_automatic_member_post_render_qa",
    "evaluate_automatic_member_post_render_qa",
    "fingerprint_automatic_member_artifacts",
]
