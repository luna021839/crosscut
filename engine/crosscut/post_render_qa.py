"""Independent post-render QA for one completed automatic GROUP artifact set.

The evaluator consumes only the finished timestamp-bound analysis,
render-plan, candidate-selection, and MP4 artifacts.  It performs read-only
probing/decoding and JSON-only professional edit diagnostics; the pipeline can
invoke it as the final delivery gate and atomically persist its report.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
import errno
import hashlib
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
from typing import Any, Callable, Mapping, Sequence

from .cut_local_sync import (
    resolve_cut_local_sync_activation,
    validate_cut_local_sync_evidence,
)
from .post_render_perceptual_quality import (
    PerceptualQualityConfig,
    evaluate_post_render_perceptual_quality,
)
from .ive_product_quality_floor import (
    automatic_ive_group_product_floor_required,
    evaluate_ive_product_quality_floor,
)
from .professional_montage_grammar import (
    PROFILE_ANALYSIS_BASIS as MONTAGE_PROFILE_ANALYSIS_BASIS,
    PROFILE_SCHEMA_VERSION as MONTAGE_PROFILE_SCHEMA_VERSION,
    ProfessionalBoundaryGrammarEvidence,
    assess_professional_boundary_grammar,
)
from .group_boundary_grammar_adapter import (
    adapt_group_boundary_grammar_evidence,
)
from .ive import AspectRatio
from .joint_release_policy import (
    JOINT_RELEASE_CANDIDATE_COUNT,
    joint_release_policy_payload,
)
from .stage_edit_release_policy import (
    stage_edit_release_binding_valid, stage_edit_release_policy_payload,
)
from .review_first_selection_contract import (
    REVIEW_FIRST_RELEASE_CANDIDATE_COUNT,
    review_first_release_binding_valid,
    review_first_release_policy_payload,
)
from .render import (
    BoundaryLookKeyframe,
    BoundaryLookTrack,
    EditorialTransformKeyframe,
    EditorialTransformTrack,
    NormalizedCrop,
    PanCropKeyframe,
    PanCropTrack,
    RenderPlan,
    RenderSegment,
    RenderTransition,
    SourceLookAdjustment,
)
from .stage_edit_dynamic_crop_motion_authority_v3 import (
    DETERMINISTIC_GEOMETRY_OVERRIDE_DISPOSITION,
    DynamicCropMotionAuthorityV3Error,
    DynamicCropMotionTimelineEvidenceV3,
    apply_dynamic_crop_motion_authority_to_render_plan_v3,
    corroborate_dynamic_crop_motion_timeline_v3,
    load_dynamic_crop_motion_authority_from_registered_identity_v3,
)
from .stage_edit_dynamic_crop_semimarkov_adapter_v3 import (
    DynamicCropTrackDecodeV3,
)
from .stage_edit_production_registry import (
    RegisteredStageEditMotionAuthorityIdentityV3,
)
from .rendered_montage_grammar_review import (
    RENDERED_MONTAGE_GRAMMAR_ANALYSIS_BASIS,
    RenderedMontageGrammarConfig,
    review_rendered_montage_grammar,
)
from .stage_edit_transition_authority import (
    RENDERER_OWNED_TRANSITION_AUTHORITY,
    StageEditTransitionAuthority,
)
from .stage_edit_windowed_beam_redecode import (
    STAGE_EDIT_WINDOWED_BEAM_CONTRACT,
)


SCHEMA_VERSION = 2
ANALYSIS_BASIS = (
    "content_bound_full_decode_professional_post_render_qa_v2"
)
INPUT_ARTIFACT_FINGERPRINT_CONTRACT = (
    "post_render_qa_input_artifact_content_sha256_v1"
)
DYNAMIC_CROP_V3_POST_RENDER_EVIDENCE_REPLAY_CONTRACT = (
    "final_render_plan_registered_release_freeze_decode_and_motion_evidence_"
    "independent_replay_v1"
)

_ANALYSIS_SUFFIX = "-analysis.json"
_PLAN_SUFFIX = "-render-plan.json"
_SELECTION_SUFFIX = "-rendered-candidate-selection.json"
_VIDEO_SUFFIX = "-group-crosscut.mp4"
_REPORT_SUFFIX = "-post-render-qa.json"
_BASE_PATTERN = re.compile(r"^(?P<base>IVE-.+-(?P<timestamp>\d{8}-\d{6}))$")

# Post-render QA intentionally does not import the proxy-review implementation:
# it audits serialized artifacts as an independent consumer.  Keep the exact
# review contracts here instead of accepting an open-ended string prefix.  The
# latter allowed a newly introduced, domain-aware v3 review to be rejected while
# an arbitrary unsupported future suffix could have been accepted.
_LEGACY_PROXY_REVIEW_BASIS = (
    "decoded_output_proxy_all_transition_quality_review_v2"
)
_DOMAIN_AWARE_PROXY_REVIEW_BASIS = (
    "decoded_output_proxy_domain_aware_quality_review_v3"
)
_INTRODUCED_CROSS_SOURCE_DOMAIN = "introduced_cross_source"
_TIER_D_CONTINUITY_DOMAIN = "tier_d_continuity"
_NATIVE_SOURCE_CUT_DOMAIN = "native_source_cut"
_BOUNDARY_REVIEW_DOMAINS = (
    _INTRODUCED_CROSS_SOURCE_DOMAIN,
    _TIER_D_CONTINUITY_DOMAIN,
    _NATIVE_SOURCE_CUT_DOMAIN,
)
_MONTAGE_PROFILE_IDENTITY_KEYS = frozenset(
    {
        "analysis_basis",
        "schema_version",
        "file_sha256",
        "content_sha256",
    }
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and _SHA256_PATTERN.fullmatch(value) is not None
    )


class PostRenderQAVerdict(str, Enum):
    PASS = "PASS"
    CONDITIONAL = "CONDITIONAL"
    FAIL = "FAIL"


class DynamicCropV3PostRenderEvidenceReplayError(ValueError):
    """Final v3 renderer motion evidence could not be independently replayed."""


class QAIssueSeverity(str, Enum):
    CONDITIONAL = "conditional"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class PostRenderQAConfig:
    required_complete_candidate_count: int = 8
    minimum_source_switches_per_minute_hard: float = 12.5
    preferred_minimum_source_switches_per_minute: float = 13.5
    preferred_maximum_source_switches_per_minute: float = 17.5
    automatic_ive_preferred_maximum_source_switches_per_minute: float = 30.0
    preferred_minimum_source_switch_ratio: float = 0.34
    preferred_maximum_source_switch_ratio: float = 0.64
    maximum_visual_shot_median_seconds: float = 2.2
    preferred_minimum_source_run_seconds: float = 2.5
    preferred_maximum_source_run_seconds: float = 5.5
    maximum_source_run_seconds: float = 7.2
    preferred_maximum_dominant_source_share: float = 0.28
    hard_maximum_dominant_source_share: float = 0.45
    minimum_distinct_sources_when_required: int = 8
    minimum_overall_quality_p10: float = 0.36
    maximum_source_quality_median_span: float = 0.22
    minimum_boundary_score_p10: float = 58.0
    minimum_proxy_weighted_score: float = 60.0
    professional_maximum_boundary_issue_count: int = 0
    professional_maximum_geometry_provider_failure_count: int = 0
    standard_boundary_repair_maximum_frames: int = 3
    partial_reedit_maximum_frames: int = 8
    exact_beat_window_frames: float = 1.0
    standard_beat_window_frames: float = 3.0
    partial_beat_window_frames: float = 8.0
    maximum_source_local_reuse_seconds: float = 0.0001
    ffprobe_fps_tolerance: float = 0.05
    final_duration_tolerance_frames: float = 1.0
    maximum_sync_residual_p95_frames: float = 0.5
    minimum_local_timewarp_anchor_count: int = 3
    maximum_editorial_zoom: float = 1.18
    maximum_editorial_center_shift: float = 0.025
    maximum_editorial_roll_degrees: float = 1.6
    minimum_editorial_event_spacing_seconds: float = 3.5
    tool_timeout_seconds: float = 900.0
    ffprobe: str = "ffprobe"
    ffmpeg: str = "ffmpeg"
    perceptual_quality: PerceptualQualityConfig = field(
        default_factory=PerceptualQualityConfig
    )
    rendered_montage_grammar: RenderedMontageGrammarConfig = field(
        default_factory=RenderedMontageGrammarConfig
    )

    def __post_init__(self) -> None:
        if not isinstance(self.perceptual_quality, PerceptualQualityConfig):
            raise TypeError("perceptual_quality must be PerceptualQualityConfig")
        if not isinstance(
            self.rendered_montage_grammar,
            RenderedMontageGrammarConfig,
        ):
            raise TypeError(
                "rendered_montage_grammar must be RenderedMontageGrammarConfig"
            )
        if self.required_complete_candidate_count < 1:
            raise ValueError("required candidate count must be positive")
        if self.minimum_local_timewarp_anchor_count < 2:
            raise ValueError("local time-warp anchor count must be at least two")
        if not (
            0.0
            < self.minimum_source_switches_per_minute_hard
            <= self.preferred_minimum_source_switches_per_minute
            <= self.preferred_maximum_source_switches_per_minute
        ):
            raise ValueError("source-switch density thresholds are invalid")
        if (
            self.automatic_ive_preferred_maximum_source_switches_per_minute
            < self.preferred_minimum_source_switches_per_minute
        ):
            raise ValueError(
                "automatic IVE source-switch maximum cannot be below preferred minimum"
            )
        if not (
            0.0
            <= self.preferred_minimum_source_switch_ratio
            <= self.preferred_maximum_source_switch_ratio
            <= 1.0
        ):
            raise ValueError("source-switch ratio thresholds are invalid")
        if not (
            0
            <= self.standard_boundary_repair_maximum_frames
            <= self.partial_reedit_maximum_frames
        ):
            raise ValueError("boundary-repair frame limits are invalid")
        if not (
            0.0
            <= self.exact_beat_window_frames
            <= self.standard_beat_window_frames
            <= self.partial_beat_window_frames
        ):
            raise ValueError("beat windows are invalid")
        finite_values = (
            value
            for name in self.__dataclass_fields__
            for value in (getattr(self, name),)
            if isinstance(value, float)
        )
        if any(not math.isfinite(value) or value < 0.0 for value in finite_values):
            raise ValueError("numeric QA configuration must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            name: (
                value.to_dict()
                if isinstance(value := getattr(self, name), PerceptualQualityConfig)
                else {
                    field_name: getattr(value, field_name)
                    for field_name in value.__dataclass_fields__
                }
                if isinstance(value, RenderedMontageGrammarConfig)
                else value
            )
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class PostRenderArtifactSet:
    base_name: str
    timestamp: str
    analysis_path: Path
    render_plan_path: Path
    selection_path: Path
    video_path: Path

    def __post_init__(self) -> None:
        match = _BASE_PATTERN.fullmatch(self.base_name)
        if match is None or match.group("timestamp") != self.timestamp:
            raise ValueError("artifact base name does not end in its timestamp")
        expected = {
            "analysis_path": f"{self.base_name}{_ANALYSIS_SUFFIX}",
            "render_plan_path": f"{self.base_name}{_PLAN_SUFFIX}",
            "selection_path": f"{self.base_name}{_SELECTION_SUFFIX}",
            "video_path": f"{self.base_name}{_VIDEO_SUFFIX}",
        }
        parents: set[Path] = set()
        for name, filename in expected.items():
            path = Path(getattr(self, name))
            object.__setattr__(self, name, path)
            if path.name != filename:
                raise ValueError(f"{name} is not bound to the artifact base name")
            parents.add(path.parent.resolve())
        if len(parents) != 1:
            raise ValueError("artifact files must share one output directory")

    @classmethod
    def from_analysis_path(cls, analysis_path: Path) -> "PostRenderArtifactSet":
        analysis = Path(analysis_path)
        if not analysis.name.endswith(_ANALYSIS_SUFFIX):
            raise ValueError("analysis filename does not use the expected suffix")
        base = analysis.name[: -len(_ANALYSIS_SUFFIX)]
        match = _BASE_PATTERN.fullmatch(base)
        if match is None:
            raise ValueError("analysis filename does not contain a terminal timestamp")
        root = analysis.parent
        return cls(
            base_name=base,
            timestamp=match.group("timestamp"),
            analysis_path=analysis,
            render_plan_path=root / f"{base}{_PLAN_SUFFIX}",
            selection_path=root / f"{base}{_SELECTION_SUFFIX}",
            video_path=root / f"{base}{_VIDEO_SUFFIX}",
        )

    @property
    def output_directory(self) -> Path:
        return self.analysis_path.parent

    @property
    def default_report_path(self) -> Path:
        return self.output_directory / f"{self.base_name}{_REPORT_SUFFIX}"

    def to_dict(self) -> dict[str, str]:
        return {
            "base_name": self.base_name,
            "timestamp": self.timestamp,
            "analysis_path": str(self.analysis_path.resolve()),
            "render_plan_path": str(self.render_plan_path.resolve()),
            "selection_path": str(self.selection_path.resolve()),
            "video_path": str(self.video_path.resolve()),
        }


@dataclass(frozen=True, slots=True)
class PostRenderQAEvaluationSnapshot:
    """Private, same-filesystem read authority for one QA evaluation.

    The snapshot uses independent inodes. On APFS the video is normally a
    clonefile copy-on-write clone; filesystems without clone support fall back
    to a byte copy. Canonical output paths may therefore undergo A->B->A
    replacement without changing what the evaluator reads or what a passing
    delivery later materializes.
    """

    declared_artifact_set: PostRenderArtifactSet
    artifact_set: PostRenderArtifactSet
    directory: Path
    creation_methods: Mapping[str, str]
    declared_input_fingerprints: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": "post_render_qa_private_read_authority_snapshot_v1",
            "directory": str(self.directory.resolve()),
            "retention": "ephemeral_removed_before_delivery_manifest_commit",
            "creation_methods": dict(self.creation_methods),
            "declared_input_fingerprints": dict(
                self.declared_input_fingerprints
            ),
            "threat_model": {
                "blocks": [
                    "declared_path_aba_between_initial_hash_and_evaluator_open",
                    "declared_path_replacement_between_qa_and_delivery_manifest",
                    "preexisting_writer_fd_to_declared_input_inode",
                ],
                "snapshot_inode_independent_from_declared_input": True,
                "large_video_copy_policy": (
                    "macos_clonefile_first_then_byte_copy_fallback"
                ),
                "same_uid_malicious_snapshot_discovery_fully_prevented": False,
                "same_uid_limitation": (
                    "a malicious same-UID process that discovers the private "
                    "snapshot can chmod or replace it; path-bound hashes and "
                    "atomic inode promotion detect ordinary mutation but do "
                    "not claim a kernel security boundary"
                ),
            },
        }


def discover_completed_artifact_sets(
    output_directory: Path,
) -> tuple[PostRenderArtifactSet, ...]:
    """Return only exact, non-empty four-file GROUP sets in timestamp order."""

    root = Path(output_directory)
    if not root.is_dir():
        return ()
    discovered: list[PostRenderArtifactSet] = []
    for analysis_path in sorted(root.glob(f"IVE-*{_ANALYSIS_SUFFIX}")):
        try:
            artifact_set = PostRenderArtifactSet.from_analysis_path(analysis_path)
        except ValueError:
            continue
        paths = (
            artifact_set.analysis_path,
            artifact_set.render_plan_path,
            artifact_set.selection_path,
            artifact_set.video_path,
        )
        try:
            complete = all(
                path.is_file() and path.stat().st_size > 0 for path in paths
            )
        except OSError:
            complete = False
        if complete:
            discovered.append(artifact_set)
    return tuple(sorted(discovered, key=lambda item: (item.timestamp, item.base_name)))


@dataclass(frozen=True, slots=True)
class QAIssue:
    code: str
    severity: QAIssueSeverity
    message: str
    path: str | None = None
    value: Any = None
    threshold: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "path": self.path,
            "value": self.value,
            "threshold": self.threshold,
        }


@dataclass(frozen=True, slots=True)
class PostRenderQAReport:
    artifact_set: PostRenderArtifactSet
    verdict: PostRenderQAVerdict
    metrics: Mapping[str, Any]
    issues: tuple[QAIssue, ...]
    policy: PostRenderQAConfig
    input_artifact_fingerprints: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    analysis_basis: str = ANALYSIS_BASIS

    @property
    def passed(self) -> bool:
        return self.verdict is PostRenderQAVerdict.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "analysis": self.analysis_basis,
            "created_at": self.created_at.astimezone(UTC).isoformat(),
            "artifact_set": self.artifact_set.to_dict(),
            "input_artifact_fingerprints": dict(
                self.input_artifact_fingerprints
            ),
            "policy": self.policy.to_dict(),
            "verdict": self.verdict.value,
            "passed": self.passed,
            "issue_count": len(self.issues),
            "fail_issue_count": sum(
                issue.severity is QAIssueSeverity.FAIL for issue in self.issues
            ),
            "conditional_issue_count": sum(
                issue.severity is QAIssueSeverity.CONDITIONAL
                for issue in self.issues
            ),
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": dict(self.metrics),
        }

    def write_atomic(self, output_path: Path | None = None) -> Path:
        destination = Path(output_path or self.artifact_set.default_report_path)
        protected = {
            self.artifact_set.analysis_path.resolve(),
            self.artifact_set.render_plan_path.resolve(),
            self.artifact_set.selection_path.resolve(),
            self.artifact_set.video_path.resolve(),
        }
        if destination.resolve() in protected:
            raise ValueError("post-render QA report cannot overwrite an input artifact")
        payload = self.to_dict()
        # Serialize before creating a temporary file so NaN/type failures leave
        # an existing report untouched.
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
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
        except OSError:
            try:
                os.close(descriptor)
            except OSError:
                pass
            temporary.unlink(missing_ok=True)
            raise
        return destination.resolve()


RunCommand = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]


def _run_command(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class _IssueCollector:
    def __init__(self) -> None:
        self.issues: list[QAIssue] = []
        self._keys: set[tuple[str, str | None]] = set()

    def add(
        self,
        code: str,
        severity: QAIssueSeverity,
        message: str,
        *,
        path: str | None = None,
        value: Any = None,
        threshold: Any = None,
    ) -> None:
        key = (code, path)
        if key in self._keys:
            return
        self._keys.add(key)
        self.issues.append(
            QAIssue(code, severity, message, path, value, threshold)
        )

    def verdict(self) -> PostRenderQAVerdict:
        if any(issue.severity is QAIssueSeverity.FAIL for issue in self.issues):
            return PostRenderQAVerdict.FAIL
        if self.issues:
            return PostRenderQAVerdict.CONDITIONAL
        return PostRenderQAVerdict.PASS


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"required artifact is missing or empty: {path.name}")
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: _reject_nonfinite_json(token),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON artifact {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path.name}")
    return value


def _file_content_fingerprint(path: Path) -> dict[str, Any]:
    """Hash one stable regular file and reject links or path replacement.

    Product promotion must bind the bytes opened by the evaluator to the
    artifact path itself.  Following a final-component symlink, or merely
    comparing the descriptor before/after reading, leaves the recorded path
    replaceable independently of the bytes that were hashed.
    """

    source = Path(path)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        path_before = source.lstat()
        if not stat.S_ISREG(path_before.st_mode):
            raise ValueError(
                f"required artifact is not a regular non-symlink file: "
                f"{source.name}"
            )
        resolved_before = source.resolve(strict=True)
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(source, flags)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or (path_before.st_dev, path_before.st_ino)
                != (before.st_dev, before.st_ino)
            ):
                raise ValueError(
                    f"artifact path changed before fingerprinting: "
                    f"{source.name}"
                )
            if before.st_size <= 0:
                raise ValueError(
                    f"required artifact is missing or empty: {source.name}"
                )
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
            after = os.fstat(stream.fileno())
            path_after = source.lstat()
            resolved_after = source.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"could not fingerprint artifact {source.name}: {exc}"
        ) from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    path_before_identity = (
        path_before.st_dev,
        path_before.st_ino,
        path_before.st_size,
        path_before.st_mtime_ns,
        path_before.st_ctime_ns,
    )
    path_after_identity = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    if (
        not stat.S_ISREG(path_after.st_mode)
        or before_identity != after_identity
        or path_before_identity != before_identity
        or path_after_identity != after_identity
        or resolved_before != resolved_after
        or byte_count != after.st_size
    ):
        raise ValueError(
            f"artifact changed while being fingerprinted: {source.name}"
        )
    return {
        "path": str(resolved_after),
        "size_bytes": byte_count,
        "sha256": digest.hexdigest(),
    }


def fingerprint_post_render_artifacts(
    artifact_set: PostRenderArtifactSet,
) -> dict[str, Any]:
    """Return exact content identities for the four mandatory QA inputs."""

    return {
        "contract": INPUT_ARTIFACT_FINGERPRINT_CONTRACT,
        "analysis": _file_content_fingerprint(artifact_set.analysis_path),
        "render_plan": _file_content_fingerprint(
            artifact_set.render_plan_path
        ),
        "selection": _file_content_fingerprint(artifact_set.selection_path),
        "video": _file_content_fingerprint(artifact_set.video_path),
    }


def normalize_post_render_artifact_fingerprints(
    fingerprints: Mapping[str, Any],
    declared_artifact_set: PostRenderArtifactSet,
) -> dict[str, Any]:
    """Bind snapshot content hashes to the externally declared artifact paths."""

    declared_paths = {
        "analysis": declared_artifact_set.analysis_path,
        "render_plan": declared_artifact_set.render_plan_path,
        "selection": declared_artifact_set.selection_path,
        "video": declared_artifact_set.video_path,
    }
    if fingerprints.get("contract") != INPUT_ARTIFACT_FINGERPRINT_CONTRACT:
        raise ValueError("post-render artifact fingerprints use the wrong contract")
    normalized: dict[str, Any] = {
        "contract": INPUT_ARTIFACT_FINGERPRINT_CONTRACT,
    }
    for key, declared_path in declared_paths.items():
        payload = fingerprints.get(key)
        if not isinstance(payload, Mapping):
            raise ValueError(f"post-render artifact fingerprint missing {key}")
        size_bytes = payload.get("size_bytes")
        sha256 = payload.get("sha256")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
            or not _valid_sha256(sha256)
        ):
            raise ValueError(
                f"post-render artifact fingerprint malformed for {key}"
            )
        normalized[key] = {
            "path": str(declared_path.resolve()),
            "size_bytes": size_bytes,
            "sha256": sha256,
        }
    return normalized


def _clone_or_copy_snapshot_file(source: Path, destination: Path) -> str:
    """Create an independent snapshot inode, preferring APFS clonefile."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"QA snapshot destination already exists: {destination}")
    clone_succeeded = False
    if sys.platform == "darwin" and Path("/bin/cp").is_file():
        completed = subprocess.run(
            ["/bin/cp", "-c", str(source.resolve()), str(destination)],
            check=False,
            capture_output=True,
            text=True,
        )
        clone_succeeded = completed.returncode == 0
        if not clone_succeeded:
            destination.unlink(missing_ok=True)
    if clone_succeeded:
        method = "macos_clonefile_copy_on_write"
    else:
        try:
            with source.open("rb") as input_stream, destination.open("xb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
                output_stream.flush()
                os.fsync(output_stream.fileno())
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        method = "byte_copy_fallback"
    destination.chmod(stat.S_IRUSR)
    return method


def materialize_post_render_qa_snapshot(
    artifact_set: PostRenderArtifactSet,
) -> PostRenderQAEvaluationSnapshot:
    """Freeze four QA inputs in a private same-filesystem artifact set.

    Content equality is checked after all independent snapshot inodes exist.
    The evaluator must receive only the returned ``artifact_set``; the original
    paths are declaration and eventual delivery destinations, not read
    authority during QA.
    """

    declared_fingerprints = fingerprint_post_render_artifacts(artifact_set)
    directory = Path(
        tempfile.mkdtemp(
            prefix=f".{artifact_set.base_name}-qa-snapshot-",
            dir=artifact_set.output_directory,
        )
    )
    frozen = PostRenderArtifactSet(
        base_name=artifact_set.base_name,
        timestamp=artifact_set.timestamp,
        analysis_path=directory / artifact_set.analysis_path.name,
        render_plan_path=directory / artifact_set.render_plan_path.name,
        selection_path=directory / artifact_set.selection_path.name,
        video_path=directory / artifact_set.video_path.name,
    )
    source_and_destination = (
        ("analysis", artifact_set.analysis_path, frozen.analysis_path),
        ("render_plan", artifact_set.render_plan_path, frozen.render_plan_path),
        ("selection", artifact_set.selection_path, frozen.selection_path),
        ("video", artifact_set.video_path, frozen.video_path),
    )
    methods: dict[str, str] = {}
    try:
        for key, source, destination in source_and_destination:
            methods[key] = _clone_or_copy_snapshot_file(source, destination)
        frozen_fingerprints = fingerprint_post_render_artifacts(frozen)
        normalized_frozen = normalize_post_render_artifact_fingerprints(
            frozen_fingerprints,
            artifact_set,
        )
        if normalized_frozen != declared_fingerprints:
            raise ValueError(
                "QA read-authority snapshot does not match declared inputs"
            )
    except Exception:
        for _key, _source, destination in source_and_destination:
            destination.unlink(missing_ok=True)
        directory.rmdir()
        raise
    return PostRenderQAEvaluationSnapshot(
        declared_artifact_set=artifact_set,
        artifact_set=frozen,
        directory=directory,
        creation_methods=methods,
        declared_input_fingerprints=declared_fingerprints,
    )


def cleanup_post_render_qa_snapshot(
    snapshot: PostRenderQAEvaluationSnapshot,
) -> None:
    """Remove only files created inside one private QA snapshot directory."""

    directory = Path(snapshot.directory)
    try:
        directory_state = directory.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(directory_state.st_mode):
        raise RuntimeError("QA snapshot cleanup target lost directory identity")
    # ExFAT AppleDouble sidecars can disappear as their primary file is
    # removed.  Treat a child that vanishes between iterdir(), lstat(), and
    # unlink() as an already-completed cleanup operation.  Repeat the bounded
    # scan because macOS can also materialize a sidecar immediately before the
    # first rmdir().  Nested real directories remain a hard failure.
    for _attempt in range(8):
        for child in tuple(directory.iterdir()):
            try:
                state = child.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(state.st_mode) and not stat.S_ISLNK(state.st_mode):
                raise RuntimeError(
                    f"unexpected directory inside QA snapshot: {child.name}"
                )
            child.unlink(missing_ok=True)
        try:
            directory.rmdir()
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            if exc.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
                raise
    raise RuntimeError("QA snapshot cleanup could not remove transient sidecars")


def _reject_nonfinite_json(token: str) -> Any:
    raise ValueError(f"non-finite JSON number is not allowed: {token}")


def _mapping(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _sequence(value: Any) -> list[Any] | None:
    return list(value) if isinstance(value, list) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _boundary_editorial_track_from_serialized(
    value: object, *, label: str,
) -> EditorialTransformTrack | None:
    if value is None:
        return None
    payload = _mapping(value)
    if payload is None:
        raise ValueError(f"{label} is malformed")
    raw_keyframes = _sequence(payload.get("keyframes"))
    reason = payload.get("reason")
    if not raw_keyframes or not isinstance(reason, str) or not reason:
        raise ValueError(f"{label} is incomplete")
    keyframes = []
    for raw in raw_keyframes:
        item = _mapping(raw)
        if item is None:
            raise ValueError(f"{label} keyframe is malformed")
        values = {
            key: _number(item.get(key))
            for key in ("time_seconds", "zoom", "center_x", "center_y", "roll_degrees")
        }
        if any(value is None for value in values.values()):
            raise ValueError(f"{label} keyframe is incomplete")
        keyframes.append(EditorialTransformKeyframe(**values))
    return EditorialTransformTrack(tuple(keyframes), reason=reason)


def _boundary_render_plan_from_serialized(value: object) -> RenderPlan:
    """Rebuild the typed boundary authority from the persisted final plan.

    The professional boundary gate only consumes source/timeline clocks,
    transitions, and editorial camera tracks.  Reconstruct those fields here
    instead of trusting the already-derived evidence embedded in the analysis
    JSON.  Redundant serialized clock fields are checked before construction so
    a self-consistent re-hash cannot hide drift in the final RenderPlan.
    """

    payload = _mapping(value)
    if payload is None:
        raise ValueError("render plan must be an object")

    raw_sources = _sequence(payload.get("sources"))
    if not raw_sources or any(
        not isinstance(item, str) or not item for item in raw_sources
    ):
        raise ValueError("render plan sources are missing or malformed")
    sources = tuple(Path(item) for item in raw_sources)

    fps = _number(payload.get("fps"))
    if fps is None or fps <= 0.0:
        raise ValueError("render plan fps is invalid")
    raw_segments = _sequence(payload.get("segments"))
    if not raw_segments:
        raise ValueError("render plan segments are missing")
    segments: list[RenderSegment] = []
    for index, raw_segment in enumerate(raw_segments):
        segment = _mapping(raw_segment)
        if segment is None:
            raise ValueError(f"render plan segment {index} is malformed")
        source_index = _integer(segment.get("source_index"))
        source_start = _number(segment.get("source_start"))
        duration = _number(segment.get("duration"))
        master_start = _number(segment.get("master_start"))
        if (
            source_index is None
            or source_start is None
            or duration is None
            or master_start is None
        ):
            raise ValueError(f"render plan segment {index} clock is malformed")
        raw_source_duration = segment.get("source_duration")
        source_duration = (
            None if raw_source_duration is None else _number(raw_source_duration)
        )
        if raw_source_duration is not None and source_duration is None:
            raise ValueError(
                f"render plan segment {index} source_duration is malformed"
            )

        transform = _boundary_editorial_track_from_serialized(
            segment.get("editorial_transform"),
            label=f"render plan segment {index} editorial transform",
        )
        overlap_transform = _boundary_editorial_track_from_serialized(
            segment.get("outgoing_overlap_transform"),
            label=f"render plan segment {index} outgoing overlap transform",
        )

        rebuilt = RenderSegment(
            source_index=source_index,
            source_start=source_start,
            duration=duration,
            master_start=master_start,
            source_duration=source_duration,
            editorial_transform=transform,
            outgoing_overlap_transform=overlap_transform,
        )
        redundant_numbers = {
            "source_local_start": rebuilt.source_start,
            "source_local_end": rebuilt.source_end,
            "effective_source_duration": rebuilt.effective_source_duration,
            "source_playback_rate": rebuilt.source_playback_rate,
        }
        for key, expected in redundant_numbers.items():
            observed = _number(segment.get(key))
            if observed is None or not math.isclose(
                observed,
                expected,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"render plan segment {index} redundant {key} disagrees"
                )
        segments.append(rebuilt)

    raw_transitions = _sequence(payload.get("transitions"))
    if raw_transitions is None:
        raise ValueError("render plan transitions are missing")
    transitions: list[RenderTransition] = []
    for index, raw_transition in enumerate(raw_transitions):
        transition = _mapping(raw_transition)
        if transition is None:
            raise ValueError(f"render plan transition {index} is malformed")
        frame_count = _integer(transition.get("frame_count"))
        reason = transition.get("reason")
        kind = transition.get("kind")
        if (
            frame_count is None
            or not isinstance(reason, str)
            or not reason
            or not isinstance(kind, str)
        ):
            raise ValueError(f"render plan transition {index} is incomplete")

        optional_numbers: dict[str, float | None] = {}
        for key in (
            "face_mismatch",
            "appearance_min_cosine_similarity",
            "future_pose_max_mismatch",
            "future_silhouette_max_mismatch",
        ):
            raw_number = transition.get(key)
            optional_numbers[key] = (
                None if raw_number is None else _number(raw_number)
            )
            if raw_number is not None and optional_numbers[key] is None:
                raise ValueError(
                    f"render plan transition {index} {key} is malformed"
                )
        transform_overlap_authority = transition.get(
            "transform_overlap_authority"
        )
        if transform_overlap_authority is not None and (
            not isinstance(transform_overlap_authority, str)
            or not transform_overlap_authority
        ):
            raise ValueError(
                f"render plan transition {index} transform overlap authority "
                "is malformed"
            )
        rebuilt_transition = RenderTransition(
            kind=kind,
            frame_count=frame_count,
            reason=reason,
            face_mismatch=optional_numbers["face_mismatch"],
            tier=transition.get("tier"),
            appearance_min_cosine_similarity=optional_numbers[
                "appearance_min_cosine_similarity"
            ],
            future_pose_max_mismatch=optional_numbers[
                "future_pose_max_mismatch"
            ],
            future_silhouette_max_mismatch=optional_numbers[
                "future_silhouette_max_mismatch"
            ],
            transform_overlap_authority=transform_overlap_authority,
        )
        serialized_duration = _number(transition.get("duration_seconds"))
        if serialized_duration is None or not math.isclose(
            serialized_duration,
            rebuilt_transition.duration_seconds(fps),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"render plan transition {index} duration disagrees"
            )
        transitions.append(rebuilt_transition)

    reference_source = _integer(payload.get("reference_source"))
    width = _integer(payload.get("width"))
    height = _integer(payload.get("height"))
    master_audio_start = _number(payload.get("master_audio_start"))
    aspect_ratio = payload.get("aspect_ratio")
    output = payload.get("output")
    if (
        reference_source is None
        or width is None
        or height is None
        or master_audio_start is None
        or not isinstance(aspect_ratio, str)
        or not isinstance(output, str)
        or not output
    ):
        raise ValueError("render plan header is malformed")
    rebuilt_plan = RenderPlan(
        sources=sources,
        segments=tuple(segments),
        reference_source=reference_source,
        aspect_ratio=AspectRatio(aspect_ratio),
        output=Path(output),
        master_audio_start=master_audio_start,
        width=width,
        height=height,
        fps=fps,
        transitions=tuple(transitions),
        member_pose_dissolve_requested=payload.get(
            "member_pose_dissolve_requested"
        ),
        member_pose_dissolve_effective=payload.get(
            "member_pose_dissolve_effective"
        ),
        member_pose_dissolve_count=payload.get("member_pose_dissolve_count"),
    )
    for key, expected in (
        ("duration", rebuilt_plan.duration),
        ("render_duration", rebuilt_plan.render_duration),
    ):
        observed = _number(payload.get(key))
        if observed is None or not math.isclose(
            observed,
            expected,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"render plan {key} disagrees with typed reconstruction")
    if _integer(payload.get("output_frame_count")) != rebuilt_plan.output_frame_count:
        raise ValueError("render plan output_frame_count disagrees")
    return rebuilt_plan


def _dynamic_crop_canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DynamicCropV3PostRenderEvidenceReplayError(
            "Dynamic Crop v3 post-render input is not canonical JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _exact_numeric_mapping(
    value: object,
    *,
    keys: frozenset[str],
    label: str,
) -> dict[str, float]:
    payload = _mapping(value)
    if payload is None or set(payload) != keys:
        raise DynamicCropV3PostRenderEvidenceReplayError(
            f"{label} keys are not exact"
        )
    result: dict[str, float] = {}
    for key in keys:
        number = _number(payload[key])
        if number is None:
            raise DynamicCropV3PostRenderEvidenceReplayError(
                f"{label}.{key} is not finite"
            )
        result[key] = number
    return result


def _normalized_crop_from_serialized(
    value: object,
    *,
    label: str,
) -> NormalizedCrop | None:
    if value is None:
        return None
    values = _exact_numeric_mapping(
        value,
        keys=frozenset({"x", "y", "width", "height"}),
        label=label,
    )
    return NormalizedCrop(**values)


def _pan_crop_from_serialized(
    value: object,
    *,
    label: str,
) -> PanCropTrack | None:
    if value is None:
        return None
    payload = _mapping(value)
    if payload is None or set(payload) != {"keyframes"}:
        raise DynamicCropV3PostRenderEvidenceReplayError(
            f"{label} keys are not exact"
        )
    raw_keyframes = _sequence(payload["keyframes"])
    if not raw_keyframes:
        raise DynamicCropV3PostRenderEvidenceReplayError(
            f"{label} keyframes are missing"
        )
    keyframes: list[PanCropKeyframe] = []
    for index, raw_keyframe in enumerate(raw_keyframes):
        keyframe = _mapping(raw_keyframe)
        if keyframe is None or set(keyframe) != {"time_seconds", "crop"}:
            raise DynamicCropV3PostRenderEvidenceReplayError(
                f"{label} keyframe {index} keys are not exact"
            )
        time_seconds = _number(keyframe["time_seconds"])
        crop = _normalized_crop_from_serialized(
            keyframe["crop"],
            label=f"{label}.keyframes[{index}].crop",
        )
        if time_seconds is None or crop is None:
            raise DynamicCropV3PostRenderEvidenceReplayError(
                f"{label} keyframe {index} is malformed"
            )
        keyframes.append(PanCropKeyframe(time_seconds=time_seconds, crop=crop))
    return PanCropTrack(tuple(keyframes))


def _source_look_from_serialized(
    value: object,
    *,
    label: str,
) -> SourceLookAdjustment | None:
    if value is None:
        return None
    payload = _mapping(value)
    if payload is None:
        raise DynamicCropV3PostRenderEvidenceReplayError(
            f"{label} must be an object"
        )
    legacy_keys = frozenset(
        {"brightness", "contrast", "saturation", "temperature_balance"}
    )
    keys = frozenset(payload)
    optional_keys = frozenset({"tint_balance", "detail_enhancement"})
    if not legacy_keys.issubset(keys) or not keys.issubset(
        legacy_keys | optional_keys
    ):
        raise DynamicCropV3PostRenderEvidenceReplayError(
            f"{label} keys are not exact"
        )
    payload.setdefault("tint_balance", 0.0)
    payload.setdefault("detail_enhancement", 0.0)
    values = _exact_numeric_mapping(
        payload,
        keys=legacy_keys | {"tint_balance", "detail_enhancement"},
        label=label,
    )
    return SourceLookAdjustment(**values)


def _boundary_look_from_serialized(
    value: object,
    *,
    label: str,
) -> BoundaryLookTrack | None:
    if value is None:
        return None
    payload = _mapping(value)
    if payload is None or set(payload) != {"keyframes", "reason"}:
        raise DynamicCropV3PostRenderEvidenceReplayError(
            f"{label} keys are not exact"
        )
    reason = payload["reason"]
    raw_keyframes = _sequence(payload["keyframes"])
    if not isinstance(reason, str) or not reason or not raw_keyframes:
        raise DynamicCropV3PostRenderEvidenceReplayError(
            f"{label} is malformed"
        )
    keyframes: list[BoundaryLookKeyframe] = []
    keys = frozenset(
        {
            "time_seconds",
            "brightness",
            "contrast",
            "saturation",
            "temperature_balance",
        }
    )
    for index, raw_keyframe in enumerate(raw_keyframes):
        values = _exact_numeric_mapping(
            raw_keyframe,
            keys=keys,
            label=f"{label}.keyframes[{index}]",
        )
        keyframes.append(BoundaryLookKeyframe(**values))
    return BoundaryLookTrack(tuple(keyframes), reason=reason)


def _dynamic_crop_render_plan_from_serialized(value: object) -> RenderPlan:
    """Strictly restore every final RenderPlan field used by its content hash."""

    payload = _mapping(value)
    if payload is None:
        raise DynamicCropV3PostRenderEvidenceReplayError(
            "final serialized RenderPlan must be an object"
        )
    try:
        partial = _boundary_render_plan_from_serialized(payload)
    except (TypeError, ValueError) as exc:
        raise DynamicCropV3PostRenderEvidenceReplayError(
            f"final serialized RenderPlan is invalid: {exc}"
        ) from exc
    raw_segments = _sequence(payload.get("segments"))
    if raw_segments is None or len(raw_segments) != len(partial.segments):
        raise DynamicCropV3PostRenderEvidenceReplayError(
            "final serialized RenderPlan segment count changed"
        )
    segments: list[RenderSegment] = []
    for index, (base_segment, raw_segment) in enumerate(
        zip(partial.segments, raw_segments, strict=True)
    ):
        segment = _mapping(raw_segment)
        if segment is None:
            raise DynamicCropV3PostRenderEvidenceReplayError(
                f"final RenderPlan segment {index} is malformed"
            )
        try:
            segments.append(
                replace(
                    base_segment,
                    crop=_normalized_crop_from_serialized(
                        segment.get("crop"),
                        label=f"segments[{index}].crop",
                    ),
                    pan_crop=_pan_crop_from_serialized(
                        segment.get("pan_crop"),
                        label=f"segments[{index}].pan_crop",
                    ),
                    look_adjustment=_source_look_from_serialized(
                        segment.get("look_adjustment"),
                        label=f"segments[{index}].look_adjustment",
                    ),
                    boundary_look_track=_boundary_look_from_serialized(
                        segment.get("boundary_look_track"),
                        label=f"segments[{index}].boundary_look_track",
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, DynamicCropV3PostRenderEvidenceReplayError):
                raise
            raise DynamicCropV3PostRenderEvidenceReplayError(
                f"final RenderPlan segment {index} cannot be reconstructed: {exc}"
            ) from exc
    try:
        rebuilt = replace(partial, segments=tuple(segments))
    except (TypeError, ValueError) as exc:
        raise DynamicCropV3PostRenderEvidenceReplayError(
            f"final RenderPlan typed reconstruction failed: {exc}"
        ) from exc
    if _dynamic_crop_canonical_sha256(
        payload
    ) != _dynamic_crop_canonical_sha256(rebuilt.to_json()):
        raise DynamicCropV3PostRenderEvidenceReplayError(
            "final serialized RenderPlan differs from its exact typed reconstruction"
        )
    return rebuilt


def replay_dynamic_crop_v3_post_render_evidence(
    *,
    serialized_render_plan: object,
    registered_motion_authority_identity: object,
    serialized_decodes: object,
    persisted_timeline_evidence: object,
) -> DynamicCropMotionTimelineEvidenceV3:
    """Independently replay final Dynamic Crop v3 renderer-motion evidence.

    The final plan must already contain the exact authority-sanitized tracks.
    Runtime then replays the live registered release freeze, restores every
    selected decode, corroborates renderer-frame physical motion, and requires
    byte-semantic equality with the persisted content-addressed evidence.
    """

    plan = _dynamic_crop_render_plan_from_serialized(serialized_render_plan)
    identity_payload = _mapping(registered_motion_authority_identity)
    if identity_payload is None:
        raise DynamicCropV3PostRenderEvidenceReplayError(
            "registered Dynamic Crop v3 motion authority identity is missing"
        )
    try:
        identity = RegisteredStageEditMotionAuthorityIdentityV3.from_dict(
            identity_payload
        )
        authority = load_dynamic_crop_motion_authority_from_registered_identity_v3(
            identity
        )
    except (OSError, TypeError, ValueError) as exc:
        raise DynamicCropV3PostRenderEvidenceReplayError(
            f"registered Dynamic Crop v3 motion authority replay failed: {exc}"
        ) from exc

    raw_decodes = _sequence(serialized_decodes)
    if raw_decodes is None:
        raise DynamicCropV3PostRenderEvidenceReplayError(
            "serialized Dynamic Crop v3 decode list is missing"
        )
    try:
        decodes = tuple(
            DynamicCropTrackDecodeV3.from_dict(raw_decode)
            if isinstance(raw_decode, Mapping)
            else (_ for _ in ()).throw(
                ValueError("Dynamic Crop v3 decode must be an object")
            )
            for raw_decode in raw_decodes
        )
    except (TypeError, ValueError) as exc:
        raise DynamicCropV3PostRenderEvidenceReplayError(
            f"serialized Dynamic Crop v3 decode is invalid: {exc}"
        ) from exc

    persisted_payload = _mapping(persisted_timeline_evidence)
    if persisted_payload is None:
        raise DynamicCropV3PostRenderEvidenceReplayError(
            "persisted Dynamic Crop v3 motion evidence is missing"
        )
    try:
        persisted = DynamicCropMotionTimelineEvidenceV3.from_dict(
            persisted_payload
        )
    except (TypeError, ValueError) as exc:
        raise DynamicCropV3PostRenderEvidenceReplayError(
            f"persisted Dynamic Crop v3 motion evidence is invalid: {exc}"
        ) from exc
    if persisted.authority != authority:
        raise DynamicCropV3PostRenderEvidenceReplayError(
            "persisted motion evidence authority differs from the live registry"
        )
    deterministic_geometry_override_indices = tuple(
        item.segment_index
        for item in persisted.segment_evidence
        if item.overlay_disposition
        == DETERMINISTIC_GEOMETRY_OVERRIDE_DISPOSITION
    )

    try:
        sanitized = apply_dynamic_crop_motion_authority_to_render_plan_v3(
            render_plan=plan,
            decodes=decodes,
            authority=authority,
            deterministic_geometry_override_segment_indices=(
                deterministic_geometry_override_indices
            ),
        )
    except (TypeError, ValueError) as exc:
        raise DynamicCropV3PostRenderEvidenceReplayError(
            f"final RenderPlan motion-authority application failed: {exc}"
        ) from exc
    if _dynamic_crop_canonical_sha256(
        sanitized.to_json()
    ) != _dynamic_crop_canonical_sha256(plan.to_json()):
        raise DynamicCropV3PostRenderEvidenceReplayError(
            "final RenderPlan was not motion-authority sanitized before rendering"
        )
    try:
        reconstructed = corroborate_dynamic_crop_motion_timeline_v3(
            render_plan=sanitized,
            decodes=decodes,
            authority=authority,
            deterministic_geometry_override_segment_indices=(
                deterministic_geometry_override_indices
            ),
        )
    except (DynamicCropMotionAuthorityV3Error, TypeError, ValueError) as exc:
        raise DynamicCropV3PostRenderEvidenceReplayError(
            f"final Dynamic Crop v3 motion evidence reconstruction failed: {exc}"
        ) from exc
    if persisted.to_dict() != reconstructed.to_dict():
        raise DynamicCropV3PostRenderEvidenceReplayError(
            "persisted Dynamic Crop v3 motion evidence differs from reconstruction"
        )
    return reconstructed


def _boundary_scene_cut_times_from_analysis(
    analysis: Mapping[str, Any],
    *,
    plan: RenderPlan,
) -> tuple[tuple[float, ...], ...]:
    """Recover every source-local hard-cut clock from its detector authority."""

    raw_records = _sequence(analysis.get("scene_analysis"))
    if raw_records is None or len(raw_records) != len(plan.sources):
        raise ValueError(
            "scene_analysis must contain one detector record per RenderPlan source"
        )
    clocks: list[tuple[float, ...]] = []
    for source_index, raw_record in enumerate(raw_records):
        record = _mapping(raw_record)
        if (
            record is None
            or record.get("index_domain") != "request_source_index"
            or _integer(record.get("source_index")) != source_index
            or record.get("source") != str(plan.sources[source_index])
        ):
            raise ValueError(
                f"scene_analysis source record {source_index} is not domain-bound"
            )
        frame_rate = _number(record.get("frame_rate"))
        raw_changes = _sequence(record.get("changes"))
        if frame_rate is None or frame_rate <= 0.0 or raw_changes is None:
            raise ValueError(
                f"scene_analysis source record {source_index} is incomplete"
            )
        source_clocks: list[float] = []
        previous_frame = -1
        for change_index, raw_change in enumerate(raw_changes):
            change = _mapping(raw_change)
            if change is None:
                raise ValueError("scene_analysis change is malformed")
            frame_index = _integer(change.get("frame_index"))
            timestamp = _number(change.get("timestamp_seconds"))
            if (
                frame_index is None
                or frame_index < 0
                or frame_index <= previous_frame
                or timestamp is None
                or not math.isclose(
                    timestamp,
                    frame_index / frame_rate,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(
                    "scene_analysis source "
                    f"{source_index} change {change_index} clock is malformed"
                )
            previous_frame = frame_index
            source_clocks.append(timestamp)
        clocks.append(tuple(source_clocks))
    return tuple(clocks)


def _probe_number(value: Any) -> float | None:
    result = _number(value)
    if result is not None:
        return result
    if not isinstance(value, str) or value.strip() in {"", "N/A"}:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _probe_integer(value: Any) -> int | None:
    result = _integer(value)
    if result is not None:
        return result
    if not isinstance(value, str) or not value.strip().isdigit():
        return None
    return int(value)


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = fraction * (len(clean) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    alpha = position - low
    return clean[low] * (1.0 - alpha) + clean[high] * alpha


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    clean = tuple(float(value) for value in values if math.isfinite(float(value)))
    return {
        "count": len(clean),
        "minimum": min(clean) if clean else None,
        "p50": _quantile(clean, 0.50),
        "p90": _quantile(clean, 0.90),
        "maximum": max(clean) if clean else None,
    }


def _declared_path_matches(value: Any, actual: Path) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    declared = Path(value).expanduser()
    if declared.name != actual.name:
        return False
    if declared.is_absolute():
        return declared.resolve() == actual.resolve()
    return (actual.parent / declared).resolve() == actual.resolve()


def _parse_rate(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _number(value)
    if not isinstance(value, str) or not value or value in {"N/A", "0/0"}:
        return None
    try:
        numerator, denominator = value.split("/", 1)
        denominator_value = float(denominator)
        if denominator_value == 0.0:
            return None
        result = float(numerator) / denominator_value
    except (ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def _codec_family(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"h264", "avc1", "libx264", "h264_videotoolbox"}:
        return "h264"
    if normalized in {"h265", "hevc", "hvc1", "libx265", "hevc_videotoolbox"}:
        return "hevc"
    return normalized or None


def _probe_and_decode(
    video_path: Path,
    *,
    config: PostRenderQAConfig,
    runner: RunCommand,
    collector: _IssueCollector,
) -> dict[str, Any]:
    ffprobe_command = (
        config.ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(video_path),
    )
    ffmpeg_command = (
        config.ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-f",
        "null",
        "-",
    )
    payload: dict[str, Any] | None = None
    probe_result: subprocess.CompletedProcess[str] | None = None
    try:
        probe_result = runner(ffprobe_command, config.tool_timeout_seconds)
        if probe_result.returncode != 0:
            collector.add(
                "ffprobe_failed",
                QAIssueSeverity.FAIL,
                "ffprobe could not inspect the final MP4.",
                path="tool_checks.ffprobe",
                value=(probe_result.stderr or "").strip()[-1000:],
                threshold=0,
            )
        else:
            decoded = json.loads(
                probe_result.stdout or "",
                parse_constant=lambda token: _reject_nonfinite_json(token),
            )
            if not isinstance(decoded, dict):
                raise ValueError("ffprobe result is not an object")
            payload = decoded
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError) as exc:
        collector.add(
            "ffprobe_unavailable_or_invalid",
            QAIssueSeverity.FAIL,
            "ffprobe did not return valid JSON metadata.",
            path="tool_checks.ffprobe",
            value=f"{type(exc).__name__}: {exc}",
        )

    decode_result: subprocess.CompletedProcess[str] | None = None
    try:
        decode_result = runner(ffmpeg_command, config.tool_timeout_seconds)
        if decode_result.returncode != 0:
            collector.add(
                "full_decode_failed",
                QAIssueSeverity.FAIL,
                "The final MP4 did not fully decode without errors.",
                path="tool_checks.full_decode",
                value=(decode_result.stderr or "").strip()[-1000:],
                threshold=0,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        collector.add(
            "full_decode_unavailable",
            QAIssueSeverity.FAIL,
            "The full-decode command could not be completed.",
            path="tool_checks.full_decode",
            value=f"{type(exc).__name__}: {exc}",
        )

    return {
        "ffprobe": {
            "command": list(ffprobe_command),
            "returncode": probe_result.returncode if probe_result is not None else None,
            "payload": payload,
        },
        "full_decode": {
            "command": list(ffmpeg_command),
            "returncode": decode_result.returncode if decode_result is not None else None,
            "stderr": (
                (decode_result.stderr or "").strip()[-1000:]
                if decode_result is not None
                else None
            ),
            "passed": decode_result is not None and decode_result.returncode == 0,
        },
    }


def _validate_artifact_binding(
    artifacts: PostRenderArtifactSet,
    analysis: Mapping[str, Any],
    plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    collector: _IssueCollector,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}

    created_at = analysis.get("created_at")
    timestamp_matches = False
    if isinstance(created_at, str):
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            timestamp_matches = (
                created.astimezone(UTC).strftime("%Y%m%d-%H%M%S")
                == artifacts.timestamp
            )
        except ValueError:
            timestamp_matches = False
    checks["analysis_created_at_matches_filename_timestamp"] = timestamp_matches

    render_payload = _mapping(analysis.get("render")) or {}
    checks["analysis_video_path_matches"] = _declared_path_matches(
        render_payload.get("video_path"), artifacts.video_path
    )
    checks["analysis_plan_path_matches"] = _declared_path_matches(
        render_payload.get("plan_path"), artifacts.render_plan_path
    )
    checks["render_plan_output_matches"] = _declared_path_matches(
        plan.get("output"), artifacts.video_path
    )
    checks["standalone_selection_matches_analysis"] = (
        analysis.get("rendered_candidate_selection") == selection
    )
    embedded_render_selection = render_payload.get("rendered_candidate_selection")
    checks["render_selection_matches_standalone"] = (
        embedded_render_selection is None or embedded_render_selection == selection
    )
    try:
        checks["video_nonempty"] = (
            artifacts.video_path.is_file()
            and artifacts.video_path.stat().st_size > 0
        )
    except OSError:
        checks["video_nonempty"] = False

    for name, passed in checks.items():
        if not passed:
            collector.add(
                f"artifact_binding_{name}",
                QAIssueSeverity.FAIL,
                "The timestamp-bound artifact set is incomplete or internally inconsistent.",
                path=f"artifact_binding.{name}",
                value=passed,
                threshold=True,
            )

    residue_patterns = (
        ".ive-crosscut-candidate-proxy-*",
        ".ive-crosscut-proxy-*",
        ".partial-candidate-*.mp4",
    )
    try:
        residues = sorted(
            {
                str(path.resolve())
                for pattern in residue_patterns
                for path in artifacts.output_directory.glob(pattern)
            }
        )
    except OSError as exc:
        residues = []
        collector.add(
            "temporary_proxy_residue_scan_failed",
            QAIssueSeverity.FAIL,
            "The output directory could not be checked for temporary proxy residue.",
            path="artifact_binding.temporary_proxy_residue",
            value=f"{type(exc).__name__}: {exc}",
        )
    if residues:
        collector.add(
            "temporary_proxy_residue",
            QAIssueSeverity.FAIL,
            "Temporary candidate proxy residue remains beside the completed artifacts.",
            path="artifact_binding.temporary_proxy_residue",
            value=residues,
            threshold=[],
        )
    return {"checks": checks, "temporary_proxy_residue": residues}


def _expected_review_boundaries(
    segment_payloads: Sequence[Any],
    transition_payloads: Sequence[Any],
) -> list[tuple[int, int, str, str | None, str | None, bool | None]]:
    segments = [_mapping(raw) or {} for raw in segment_payloads]
    transitions = [_mapping(raw) or {} for raw in transition_payloads]
    if len(segments) != len(transitions) + 1:
        return []
    expected: list[
        tuple[int, int, str, str | None, str | None, bool | None]
    ] = []
    for index, transition in enumerate(transitions):
        outgoing = _integer(segments[index].get("source_index"))
        incoming = _integer(segments[index + 1].get("source_index"))
        tier = transition.get("tier")
        reason = transition.get("reason")
        if outgoing is None or incoming is None:
            return []
        left_source_end = _number(segments[index].get("source_local_end"))
        if left_source_end is None:
            left_source_end = _number(segments[index].get("source_end"))
        right_source_start = _number(
            segments[index + 1].get("source_local_start")
        )
        if right_source_start is None:
            right_source_start = _number(segments[index + 1].get("source_start"))
        source_clock_contiguous = (
            False
            if outgoing != incoming
            else (
                abs(left_source_end - right_source_start) <= 1e-4
                if left_source_end is not None and right_source_start is not None
                else None
            )
        )
        if outgoing != incoming:
            domain = _INTRODUCED_CROSS_SOURCE_DOMAIN
        elif tier == "D":
            domain = _TIER_D_CONTINUITY_DOMAIN
        else:
            domain = _NATIVE_SOURCE_CUT_DOMAIN
        expected.append(
            (
                outgoing,
                incoming,
                domain,
                tier if isinstance(tier, str) else None,
                reason if isinstance(reason, str) else None,
                source_clock_contiguous,
            )
        )
    return expected


def _domain_aware_boundary_contract_valid(
    review: Mapping[str, Any],
    *,
    expected_boundaries: Sequence[
        tuple[int, int, str, str | None, str | None, bool | None]
    ],
) -> bool:
    """Validate v3 boundary domains against the authored render plan.

    Same-source coalescing is a renderer implementation detail.  The review
    still owns one record per authored boundary and must distinguish an
    invisible Tier-D continuation from a broadcaster/native camera cut.  This
    check therefore binds every serialized domain, source pair, tier and reason
    back to the uncoalesced plan rather than merely trusting aggregate counts.
    """

    boundaries = [
        _mapping(raw) or {}
        for raw in (_sequence(review.get("boundaries")) or [])
    ]
    if len(boundaries) != len(expected_boundaries):
        return False
    if any(
        boundary.get("boundary_index") != index
        or boundary.get("outgoing_source_index") != expected[0]
        or boundary.get("incoming_source_index") != expected[1]
        or boundary.get("review_domain") != expected[2]
        or boundary.get("transition_tier") != expected[3]
        or boundary.get("transition_reason") != expected[4]
        or expected[5] is None
        or boundary.get("source_clock_contiguous") is not expected[5]
        for index, (boundary, expected) in enumerate(
            zip(boundaries, expected_boundaries, strict=True)
        )
    ):
        return False

    expected_counts = Counter(item[2] for item in expected_boundaries)
    serialized_count_fields = {
        _INTRODUCED_CROSS_SOURCE_DOMAIN: "introduced_cross_source_count",
        _TIER_D_CONTINUITY_DOMAIN: "tier_d_continuity_count",
        _NATIVE_SOURCE_CUT_DOMAIN: "native_source_cut_count",
    }
    if any(
        _integer(review.get(field_name)) != expected_counts[domain]
        for domain, field_name in serialized_count_fields.items()
    ):
        return False

    domains = _mapping(review.get("boundary_domains")) or {}
    if set(domains) != set(_BOUNDARY_REVIEW_DOMAINS):
        return False
    for domain in _BOUNDARY_REVIEW_DOMAINS:
        payload = _mapping(domains.get(domain)) or {}
        count = expected_counts[domain]
        issue_indices = _sequence(payload.get("issue_indices"))
        issue_count = _integer(payload.get("issue_count"))
        issue_fraction = _number(payload.get("issue_fraction"))
        score = _mapping(payload.get("score")) or {}
        distribution = _mapping(score.get("distribution")) or {}
        score_p10 = _number(score.get("p10"))
        expected_pass_policy = (
            "editorial_boundary_p10_and_issue_fraction_gate"
            if domain == _INTRODUCED_CROSS_SOURCE_DOMAIN
            else "zero_technical_integrity_issues"
        )
        if not (
            payload.get("domain") == domain
            and payload.get("count_basis") == domain
            and _integer(payload.get("boundary_count")) == count
            and issue_indices is not None
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value < len(boundaries)
                and expected_boundaries[value][2] == domain
                for value in issue_indices
            )
            and len(set(issue_indices)) == len(issue_indices)
            and issue_count == len(issue_indices)
            and issue_fraction is not None
            and math.isclose(
                issue_fraction,
                len(issue_indices) / count if count else 0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and _integer(distribution.get("count")) == count
            and score_p10 is not None
            and (
                count > 0
                or math.isclose(
                    score_p10,
                    100.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            and payload.get("pass_policy") == expected_pass_policy
            and isinstance(payload.get("passed"), bool)
        ):
            return False
    return True


def _decoded_candidate_review_contract_valid(
    review: Mapping[str, Any],
    *,
    expected_decoded_frames: int | None,
    expected_proxy_duration: float | None,
    expected_proxy_fps: float | None,
    expected_transition_count: int | None,
    expected_source_indices: Sequence[int],
    expected_boundaries: Sequence[
        tuple[int, int, str, str | None, str | None, bool | None]
    ],
) -> bool:
    proxy = _mapping(review.get("proxy")) or {}
    decoded_frames = _integer(proxy.get("decoded_frame_count"))
    proxy_duration = _number(proxy.get("duration_seconds"))
    proxy_fps = _number(proxy.get("fps"))
    transition_count = _integer(review.get("transition_count"))
    reviewed_count = _integer(review.get("reviewed_boundary_count"))
    hard_count = _integer(review.get("hard_cut_count"))
    dissolve_count = _integer(review.get("dissolve_count"))
    boundaries = _sequence(review.get("boundaries")) or []
    overall = _mapping(review.get("overall_quality")) or {}
    source_quality = _mapping(review.get("source_quality")) or {}
    source_quality_items = [
        item
        for raw in (_sequence(source_quality.get("per_source")) or [])
        if (item := _mapping(raw)) is not None
    ]
    source_quality_indices = [
        _integer(item.get("source_index")) for item in source_quality_items
    ]
    source_quality_medians = [
        _number((_mapping(item.get("quality")) or {}).get("median"))
        for item in source_quality_items
    ]
    source_quality_p10s = [
        _number((_mapping(item.get("quality")) or {}).get("p10"))
        for item in source_quality_items
    ]
    serialized_median_span = _number(source_quality.get("median_span"))
    serialized_p10_span = _number(source_quality.get("p10_span"))
    derived_median_span = (
        max(value for value in source_quality_medians if value is not None)
        - min(value for value in source_quality_medians if value is not None)
        if source_quality_medians
        and all(value is not None for value in source_quality_medians)
        else None
    )
    derived_p10_span = (
        max(value for value in source_quality_p10s if value is not None)
        - min(value for value in source_quality_p10s if value is not None)
        if source_quality_p10s
        and all(value is not None for value in source_quality_p10s)
        else None
    )
    boundary_score = _mapping(review.get("boundary_score")) or {}
    analysis_basis = review.get("analysis")
    schema_version = _integer(review.get("schema_version"))
    review_contract_valid = (
        schema_version == 2
        and analysis_basis == _LEGACY_PROXY_REVIEW_BASIS
    ) or (
        schema_version == 3
        and analysis_basis == _DOMAIN_AWARE_PROXY_REVIEW_BASIS
        and _domain_aware_boundary_contract_valid(
            review,
            expected_boundaries=expected_boundaries,
        )
    )
    return bool(
        review_contract_valid
        and isinstance(review.get("passed"), bool)
        and proxy.get("path") is None
        and proxy.get("temporary_proxy_removed_after_review") is True
        and expected_decoded_frames is not None
        and decoded_frames == expected_decoded_frames
        and expected_proxy_duration is not None
        and proxy_duration is not None
        and math.isclose(
            proxy_duration,
            expected_proxy_duration,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and expected_proxy_fps is not None
        and proxy_fps is not None
        and math.isclose(
            proxy_fps,
            expected_proxy_fps,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and expected_transition_count is not None
        and transition_count == expected_transition_count
        and reviewed_count == expected_transition_count
        and hard_count is not None
        and dissolve_count is not None
        and hard_count + dissolve_count == expected_transition_count
        and len(boundaries) == expected_transition_count
        and all(
            (_mapping(item) or {}).get("boundary_index") == index
            and isinstance((_mapping(item) or {}).get("passed"), bool)
            and _number((_mapping(item) or {}).get("weighted_score")) is not None
            for index, item in enumerate(boundaries)
        )
        and _integer(overall.get("count")) == expected_decoded_frames
        and _number(overall.get("p10")) is not None
        and all(index is not None for index in source_quality_indices)
        and len(source_quality_indices) == len(expected_source_indices)
        and len(set(source_quality_indices)) == len(source_quality_indices)
        and set(source_quality_indices) == set(expected_source_indices)
        and all(value is not None for value in source_quality_medians)
        and all(value is not None for value in source_quality_p10s)
        and serialized_median_span is not None
        and derived_median_span is not None
        and math.isclose(
            serialized_median_span,
            derived_median_span,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and serialized_p10_span is not None
        and derived_p10_span is not None
        and math.isclose(
            serialized_p10_span,
            derived_p10_span,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and _number(boundary_score.get("p10")) is not None
        and _integer(boundary_score.get("issue_count")) is not None
        and _number(review.get("weighted_score")) is not None
        and _integer(review.get("geometry_provider_failure_count")) is not None
    )


def _validate_selection(
    analysis: Mapping[str, Any],
    plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    config: PostRenderQAConfig,
    collector: _IssueCollector,
    *,
    allow_failed_review: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    required = config.required_complete_candidate_count
    final_review = _mapping(selection.get("final_selected_review")) or {}
    expected_decoded_frames = _integer(plan.get("output_frame_count"))
    expected_proxy_duration = _number(plan.get("render_duration"))
    expected_proxy_fps = _number(plan.get("fps"))
    expected_source_indices = sorted(
        {
            index
            for raw in (_sequence(plan.get("segments")) or [])
            if (item := _mapping(raw)) is not None
            and (index := _integer(item.get("source_index"))) is not None
        }
    )
    plan_segments = [
        _mapping(raw) or {}
        for raw in (_sequence(plan.get("segments")) or [])
    ]
    plan_transitions = [
        _mapping(raw) or {} for raw in (_sequence(plan.get("transitions")) or [])
    ]
    expected_transition_count = len(plan_transitions)
    expected_boundaries = _expected_review_boundaries(
        plan_segments,
        plan_transitions,
    )
    checks = {
        "selection_passed": (
            isinstance(selection.get("passed"), bool)
            if allow_failed_review
            else selection.get("passed") is True
        ),
        "requested_review_count": selection.get("requested_review_count") == required,
        "reviewed_candidate_count_field": (
            selection.get("reviewed_candidate_count") == required
        ),
        "minimum_required_candidate_count": (
            selection.get("minimum_required_candidate_count") == required
        ),
        "minimum_candidate_count_satisfied": (
            selection.get("minimum_candidate_count_satisfied") is True
        ),
        "complete_candidate_count": selection.get("complete_candidate_count") == required,
        "successful_review_count": selection.get("successful_review_count") == required,
        "exact_complete_candidate_count_satisfied": (
            selection.get("exact_complete_candidate_count_satisfied") is True
        ),
        "exact_successful_decoded_review_count_satisfied": (
            selection.get("exact_successful_decoded_review_count_satisfied") is True
        ),
    }
    candidates = _sequence(selection.get("candidates")) or []
    complete_candidates = _sequence(selection.get("complete_candidates")) or []
    complete_candidates_by_index = {
        index: item
        for raw in complete_candidates
        if (item := _mapping(raw)) is not None
        and (index := _integer(item.get("candidate_index"))) is not None
    }
    candidate_objects = [_mapping(item) or {} for item in candidates]
    candidate_indices = [
        _integer(item.get("candidate_index")) for item in candidate_objects
    ]
    successful_candidate_flags: list[bool] = []
    review_pass_flags_match = True
    sanitized_proxy_flags = True
    for item in candidate_objects:
        review = _mapping(item.get("review"))
        candidate_source_indices = expected_source_indices
        candidate_transition_count = expected_transition_count
        candidate_boundaries = expected_boundaries
        if (
            review is not None
            and review.get("analysis") == _DOMAIN_AWARE_PROXY_REVIEW_BASIS
        ):
            complete_candidate = complete_candidates_by_index.get(
                _integer(item.get("candidate_index"))
            )
            candidate_choices = _sequence(
                (
                    _mapping(
                        (complete_candidate or {}).get("visual_timeline")
                    )
                    or {}
                ).get("choices")
            ) or []
            candidate_transitions = _sequence(
                (
                    _mapping(
                        (complete_candidate or {}).get("group_transitions")
                    )
                    or {}
                ).get("transitions")
            ) or []
            candidate_boundaries = _expected_review_boundaries(
                candidate_choices,
                candidate_transitions,
            )
            candidate_transition_count = len(candidate_transitions)
            candidate_source_indices = sorted(
                {
                    index
                    for raw in candidate_choices
                    if (choice := _mapping(raw)) is not None
                    and (index := _integer(choice.get("source_index"))) is not None
                }
            )
        successful = (
            review is not None
            and item.get("error") is None
            and _decoded_candidate_review_contract_valid(
                review,
                expected_decoded_frames=expected_decoded_frames,
                expected_proxy_duration=expected_proxy_duration,
                expected_proxy_fps=expected_proxy_fps,
                expected_transition_count=candidate_transition_count,
                expected_source_indices=candidate_source_indices,
                expected_boundaries=candidate_boundaries,
            )
        )
        successful_candidate_flags.append(successful)
        if successful:
            review_pass_flags_match = (
                review_pass_flags_match
                and item.get("passed") is review.get("passed")
            )
            proxy = _mapping(review.get("proxy")) or {}
            sanitized_proxy_flags = (
                sanitized_proxy_flags
                and proxy.get("path") is None
                and proxy.get("temporary_proxy_removed_after_review") is True
            )
    actual_successful_count = sum(successful_candidate_flags)
    actual_passing_count = sum(
        item.get("passed") is True for item in candidate_objects
    )
    selected_index = _integer(selection.get("selected_candidate_index"))
    selected_entries = [
        item
        for item in candidate_objects
        if item.get("selected") is True
    ]
    checks["reviewed_candidate_count"] = len(candidates) == required
    checks["complete_candidates_payload_count"] = (
        len(complete_candidates) == required
    )
    complete_candidate_indices = [
        _integer((_mapping(item) or {}).get("candidate_index"))
        for item in complete_candidates
    ]
    checks["complete_candidate_indices_match_reviewed"] = (
        all(index is not None for index in complete_candidate_indices)
        and len(set(complete_candidate_indices)) == required
        and set(complete_candidate_indices) == set(candidate_indices)
    )
    checks["candidate_indices_valid_and_unique"] = (
        all(index is not None for index in candidate_indices)
        and len(set(candidate_indices)) == required
    )
    checks["actual_successful_review_count_matches"] = (
        actual_successful_count == required
        and selection.get("successful_review_count") == actual_successful_count
    )
    checks["actual_passing_candidate_count_matches"] = (
        selection.get("passing_candidate_count") == actual_passing_count
    )
    checks["candidate_pass_flags_match_reviews"] = review_pass_flags_match
    checks["candidate_proxy_payloads_sanitized"] = sanitized_proxy_flags
    checks["selected_candidate_marker_matches"] = (
        selected_index is not None
        and len(selected_entries) == 1
        and selected_entries[0].get("candidate_index") == selected_index
    )
    storage_policy = _mapping(selection.get("storage_policy")) or {}
    checks["one_proxy_at_a_time_storage_policy"] = (
        storage_policy.get("one_proxy_at_a_time") is True
        and storage_policy.get("temporary_proxy_removed_after_each_review") is True
        and storage_policy.get("decoded_frames_persisted") is False
    )
    checks["final_selected_review_passed"] = (
        isinstance(final_review.get("passed"), bool)
        if allow_failed_review
        else final_review.get("passed") is True
    )
    checks["final_selected_decoded_review_contract_valid"] = (
        _decoded_candidate_review_contract_valid(
            final_review,
            expected_decoded_frames=expected_decoded_frames,
            expected_proxy_duration=expected_proxy_duration,
            expected_proxy_fps=expected_proxy_fps,
            expected_transition_count=expected_transition_count,
            expected_source_indices=expected_source_indices,
            expected_boundaries=expected_boundaries,
        )
    )
    checks["analysis_final_review_matches"] = (
        analysis.get("rendered_proxy_review") == final_review
    )
    for name, passed in checks.items():
        if not passed:
            collector.add(
                f"selection_{name}",
                QAIssueSeverity.FAIL,
                "The required decoded candidate selection contract did not pass.",
                path=f"selection.{name}",
                value=passed,
                threshold=True,
            )

    version = selection.get("final_selected_version")
    partial = _mapping(selection.get("proxy_triggered_partial_reedit")) or {}
    checks["nested_selected_version_matches"] = (
        partial.get("selected_version") == version
    )
    if version == "original" and selected_entries:
        checks["original_final_review_matches_selected_candidate"] = (
            isinstance(selected_entries[0].get("passed"), bool)
            and selected_entries[0].get("passed") is final_review.get("passed")
            and selected_entries[0].get("review") == final_review
        )
    else:
        checks["original_final_review_matches_selected_candidate"] = True
    for name in (
        "nested_selected_version_matches",
        "original_final_review_matches_selected_candidate",
    ):
        if not checks[name]:
            collector.add(
                f"selection_{name}",
                QAIssueSeverity.FAIL,
                "The final selection version or review is internally inconsistent.",
                path=f"selection.{name}",
                value=checks[name],
                threshold=True,
            )
    partial_checks: dict[str, Any] = {
        "selected_version": version,
        "selected_partial_reedit": version == "partial_reedit",
    }
    if version not in {
        "original",
        "partial_reedit",
        "windowed_semimarkov_redecode",
    }:
        collector.add(
            "selection_unknown_final_version",
            QAIssueSeverity.FAIL,
            "The final selected candidate version is unknown.",
            path="selection.final_selected_version",
            value=version,
        )
    if version == "partial_reedit":
        invariant_checks = _mapping(partial.get("invariant_checks")) or {}
        required_invariant_names = {
            "segment_count_preserved",
            "source_sequence_preserved",
            "render_source_sequence_preserved",
            "canonical_first_last_range_preserved",
            "outside_issue_boundary_timestamps_preserved",
            "actual_changed_boundaries_subset_of_requested",
            "output_frame_count_preserved",
            "render_duration_preserved",
        }
        valid_invariants = (
            required_invariant_names.issubset(invariant_checks)
            and all(value is True for value in invariant_checks.values())
        )
        changed_indices = _sequence(partial.get("changed_boundary_indices")) or []
        transition_count = len(
            _sequence(
                (_mapping(analysis.get("group_transitions")) or {}).get(
                    "transitions"
                )
            )
            or []
        )
        repair_items = [
            item
            for raw in (
                _sequence((_mapping(partial.get("repair")) or {}).get("repairs"))
                or []
            )
            if (item := _mapping(raw)) is not None
        ]
        changed_repair_indices = [
            index
            for item in repair_items
            if (index := _integer(item.get("boundary_index"))) is not None
            and (
                (_mapping(item.get("result")) or {}).get("changed") is True
                or (
                    _integer(
                        (_mapping(item.get("result")) or {}).get(
                            "selected_frame_offset"
                        )
                    )
                    or 0
                )
                != 0
            )
        ]
        valid_changed_indices = (
            bool(changed_indices)
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                and value < transition_count
                for value in changed_indices
            )
            and len(set(changed_indices)) == len(changed_indices)
            and set(changed_indices) == set(changed_repair_indices)
            and len(changed_repair_indices) == len(set(changed_repair_indices))
        )
        selected_original_review = (
            _mapping(selected_entries[0].get("review"))
            if len(selected_entries) == 1
            else None
        )
        improvement_proof_valid = (
            selected_original_review is not None
            and selected_original_review.get("passed") is not True
            and partial.get("before_review") == selected_original_review
            and partial.get("after_review") == final_review
            and final_review.get("passed") is True
        )
        partial_checks.update(
            {
                "attempted": partial.get("attempted") is True,
                "improved": partial.get("improved") is True,
                "outside_issue_boundaries_preserved": (
                    partial.get("outside_issue_boundaries_preserved") is True
                ),
                "all_invariants_passed": valid_invariants,
                "changed_boundary_indices_valid": valid_changed_indices,
                "decoded_improvement_proof_valid": improvement_proof_valid,
            }
        )
        for name in (
            "attempted",
            "improved",
            "outside_issue_boundaries_preserved",
            "all_invariants_passed",
            "changed_boundary_indices_valid",
            "decoded_improvement_proof_valid",
        ):
            if not partial_checks[name]:
                collector.add(
                    f"partial_reedit_{name}",
                    QAIssueSeverity.FAIL,
                    "The selected partial re-edit lacks a required invariant or improvement proof.",
                    path=f"selection.proxy_triggered_partial_reedit.{name}",
                    value=partial_checks[name],
                    threshold=True,
                )
    windowed_checks: dict[str, Any] = {
        "selected_windowed_semimarkov_redecode": (
            version == "windowed_semimarkov_redecode"
        ),
    }
    if version == "windowed_semimarkov_redecode":
        repair_loop = _mapping(partial.get("repair_loop")) or {}
        beam_audit = _mapping(partial.get("windowed_beam_redecode")) or {}
        beam_calls = [
            item
            for raw in (_sequence(beam_audit.get("calls")) or [])
            if (item := _mapping(raw)) is not None
        ]
        evidence = _mapping(partial.get("selected_redecode_evidence")) or {}
        lineage = [
            item
            for raw in (_sequence(partial.get("candidate_lineage_after")) or [])
            if (item := _mapping(raw)) is not None
        ]
        final_lineage_events = [
            item
            for item in lineage
            if item.get("stage")
            == "reviewer_issue_window_actual_semimarkov_redecode"
        ]
        final_lineage = (
            final_lineage_events[0]
            if len(final_lineage_events) == 1
            else {}
        )
        proposal_markers = [
            (index, item)
            for index, item in enumerate(lineage)
            if item.get("stage") == "windowed_semimarkov_proposal_lineage"
        ]
        iterations = [
            item
            for raw in (_sequence(repair_loop.get("iterations")) or [])
            if (item := _mapping(raw)) is not None
        ]
        committed_iterations = [
            item
            for item in iterations
            if isinstance(item.get("committed_proposal_id"), str)
            and bool(item.get("committed_proposal_id"))
            and _valid_sha256(item.get("committed_edit_sha256"))
        ]
        committed_count = _integer(repair_loop.get("committed_iteration_count"))
        initial_sha = repair_loop.get("initial_edit_sha256")
        final_sha = repair_loop.get("final_edit_sha256")
        selected_original_review = (
            _mapping(selected_entries[0].get("review"))
            if len(selected_entries) == 1
            else None
        )
        perceptual_required = (
            evidence.get("perceptual_targeted_source_substitution_required")
            is True
        )
        required_evidence_keys = {
            "actual_semimarkov_redecode",
            "new_proxy_review_available",
            "new_proxy_passed",
            "new_proxy_perceptual_source_admission_passed",
            "source_sequence_changed_by_redecode",
            "perceptual_targeted_source_substitution_required",
            "perceptual_targeted_source_substitution_satisfied",
            "proposal_materialization_lineage_preserved",
            "final_authoritative_exact_shot_clock_count",
        }
        evidence_contract_valid = (
            required_evidence_keys.issubset(evidence)
            and evidence.get("actual_semimarkov_redecode") is True
            and evidence.get("new_proxy_review_available") is True
            and evidence.get("new_proxy_passed") is True
            and type(evidence.get("source_sequence_changed_by_redecode")) is bool
            and type(
                evidence.get(
                    "perceptual_targeted_source_substitution_required"
                )
            )
            is bool
            and type(
                evidence.get(
                    "perceptual_targeted_source_substitution_satisfied"
                )
            )
            is bool
            and evidence.get("proposal_materialization_lineage_preserved") is True
            and evidence.get("final_authoritative_exact_shot_clock_count") == 1
            and (
                not perceptual_required
                or (
                    evidence.get("new_proxy_perceptual_source_admission_passed")
                    is True
                    and evidence.get(
                        "perceptual_targeted_source_substitution_satisfied"
                    )
                    is True
                )
            )
        )
        proposal_lineage_contract_valid = False
        if len(proposal_markers) == 1 and len(final_lineage_events) == 1:
            marker_index, marker = proposal_markers[0]
            proposal_count = _integer(marker.get("event_count"))
            final_index = lineage.index(final_lineage_events[0])
            proposal_lineage_contract_valid = (
                proposal_count is not None
                and proposal_count >= 0
                and marker_index + proposal_count + 1 == final_index
                and final_lineage.get(
                    "proposal_materialization_lineage_event_count"
                )
                == proposal_count
            )
        lineage_contract_valid = (
            len(final_lineage_events) == 1
            and proposal_lineage_contract_valid
            and final_lineage.get("contract") == STAGE_EDIT_WINDOWED_BEAM_CONTRACT
            and final_lineage.get("repair_loop_input_sha256")
            == repair_loop.get("input_sha256")
            and final_lineage.get("initial_edit_sha256") == initial_sha
            and final_lineage.get("final_edit_sha256") == final_sha
            and final_lineage.get("committed_iteration_count") == committed_count
            and final_lineage.get("actual_public_decoder_invoked") is True
            and final_lineage.get("decoded_proxy_revalidated_after_splice") is True
            and final_lineage.get(
                "heuristic_timing_repair_used_as_reviewer_repair"
            )
            is False
            and all(final_lineage.get(key) == value for key, value in evidence.items())
        )
        promotion = _mapping(selection.get("stage_edit_render_promotion"))
        promotion_lineage_valid = (
            promotion is not None
            and promotion.get("selected_version") == version
            and promotion.get("selected_candidate_index") == selected_index
            and promotion.get("selected_candidate_lineage") == lineage
        )
        windowed_checks.update(
            {
                "analysis_basis": (
                    partial.get("analysis_basis")
                    == STAGE_EDIT_WINDOWED_BEAM_CONTRACT
                ),
                "attempted": partial.get("attempted") is True,
                "actual_local_semimarkov_redecode": (
                    partial.get("actual_local_semimarkov_redecode") is True
                ),
                "heuristic_repair_not_used": (
                    partial.get("heuristic_timing_repair_used") is False
                ),
                "repair_changed": (
                    partial.get("repair_loop_changed") is True
                    and partial.get("changed") is True
                    and _valid_sha256(initial_sha)
                    and _valid_sha256(final_sha)
                    and initial_sha != final_sha
                ),
                "repair_loop_contract": (
                    repair_loop.get("orchestrator") == "stage_edit_repair_loop_v1"
                    and _valid_sha256(repair_loop.get("input_sha256"))
                    # The repair loop receives exactly `(base_edit,)`; this is
                    # its local tuple domain, not the rendered-candidate index.
                    and repair_loop.get("selected_candidate_index") == 0
                    and committed_count is not None
                    and committed_count > 0
                    and committed_count == len(committed_iterations)
                    and committed_iterations[-1].get("committed_edit_sha256")
                    == final_sha
                    and bool(_mapping(repair_loop.get("initial_review")))
                    and bool(_mapping(repair_loop.get("final_review")))
                ),
                "beam_decoder_contract": (
                    beam_audit.get("contract")
                    == STAGE_EDIT_WINDOWED_BEAM_CONTRACT
                    and beam_audit.get("actual_public_decoder")
                    == (
                        "crosscut.stage_edit_semimarkov_beam."
                        "decode_stage_edit_semimarkov_beam"
                    )
                    and (_integer(beam_audit.get("call_count")) or 0) > 0
                    and len(beam_calls)
                    == _integer(beam_audit.get("call_count"))
                    and all(
                        call.get("contract")
                        == STAGE_EDIT_WINDOWED_BEAM_CONTRACT
                        and (_integer(call.get("iteration")) or 0) > 0
                        for call in beam_calls
                    )
                ),
                "decoded_final_review_contract": (
                    selected_original_review is not None
                    and partial.get("before_review") == selected_original_review
                    and partial.get("after_review") == final_review
                    and final_review.get("passed") is True
                    and partial.get("reason")
                    == "strict_objectives_improved_and_proxy_revalidated"
                ),
                "selected_redecode_evidence_contract": evidence_contract_valid,
                "candidate_lineage_contract": lineage_contract_valid,
                "promotion_lineage_contract": promotion_lineage_valid,
            }
        )
        for name, passed in windowed_checks.items():
            if name == "selected_windowed_semimarkov_redecode":
                continue
            if not passed:
                collector.add(
                    f"windowed_semimarkov_redecode_{name}",
                    QAIssueSeverity.FAIL,
                    "The selected windowed Semi-Markov re-decode lacks required "
                    "changed-edit, revalidation, or lineage evidence.",
                    path=f"selection.proxy_triggered_partial_reedit.{name}",
                    value=passed,
                    threshold=True,
                )
    return (
        {
            "required_candidate_count": required,
            "checks": checks,
            "final_selected_candidate_index": selection.get(
                "selected_candidate_index"
            ),
            "actual_successful_review_count": actual_successful_count,
            "actual_passing_candidate_count": actual_passing_count,
            "final_selected_version": version,
            "passing_candidate_count": selection.get("passing_candidate_count"),
            "partial_reedit": partial_checks,
            "windowed_semimarkov_redecode": windowed_checks,
        },
        final_review,
    )


def _validate_plan_invariants(
    analysis: Mapping[str, Any],
    plan: Mapping[str, Any],
    final_review: Mapping[str, Any],
    collector: _IssueCollector,
) -> dict[str, Any]:
    segments = _sequence(plan.get("segments")) or []
    transitions = _sequence(plan.get("transitions")) or []
    timeline = _mapping(analysis.get("timeline_review")) or {}
    group = _mapping(analysis.get("group_transitions")) or {}
    fps = _number(plan.get("fps"))
    duration = _number(plan.get("duration"))
    render_duration = _number(plan.get("render_duration"))
    output_frames = _integer(plan.get("output_frame_count"))
    common_range = _mapping(analysis.get("common_master_range")) or {}
    common_start = _number(common_range.get("start_seconds"))
    common_end = _number(common_range.get("end_seconds"))
    common_duration = _number(common_range.get("duration_seconds"))

    expected_frames = (
        int(round(duration * fps))
        if duration is not None and fps is not None and fps > 0.0
        else None
    )
    expected_render_duration = (
        output_frames / fps
        if output_frames is not None and fps is not None and fps > 0.0
        else None
    )
    n = len(transitions)
    hard_count = _integer(final_review.get("hard_cut_count"))
    dissolve_count = _integer(final_review.get("dissolve_count"))
    group_hard = _integer(group.get("hard_cut_boundary_count"))
    group_dissolve = _integer(group.get("dissolve_count"))
    introduced_hard = _integer(group.get("introduced_hard_cut_count"))
    same_source_hold = _integer(group.get("same_source_hold_count"))
    timeline_switches = _integer(timeline.get("source_switch_count"))
    plan_hard_count = sum(
        (_mapping(item) or {}).get("kind") == "hard_cut" for item in transitions
    )
    plan_dissolve_count = sum(
        (_mapping(item) or {}).get("kind") == "face_dissolve"
        for item in transitions
    )
    plan_cross_source_switch_count = 0
    plan_introduced_hard_count = 0
    plan_tier_d_hold_count = 0
    plan_native_source_cut_count = 0
    for index, raw_transition in enumerate(transitions):
        transition = _mapping(raw_transition) or {}
        left = _mapping(segments[index]) or {} if index < len(segments) else {}
        right = (
            _mapping(segments[index + 1]) or {}
            if index + 1 < len(segments)
            else {}
        )
        same_source = left.get("source_index") == right.get("source_index")
        if not same_source:
            plan_cross_source_switch_count += 1
        if transition.get("kind") == "hard_cut" and transition.get("tier") != "D":
            plan_introduced_hard_count += 1
        if transition.get("kind") == "hard_cut" and transition.get("tier") == "D":
            plan_tier_d_hold_count += 1
        if (
            transition.get("kind") == "hard_cut"
            and transition.get("tier") != "D"
            and same_source
        ):
            plan_native_source_cut_count += 1
    timeline_boundaries = _sequence(timeline.get("boundaries")) or []
    checks: dict[str, bool] = {
        "renderer_owned_transition_authority_exact": (
            group.get("analysis_basis")
            != "tiered_future_face_head_sface_appearance_body_continuity_v4"
            or group.get("transition_authority")
            == StageEditTransitionAuthority(
                mode=RENDERER_OWNED_TRANSITION_AUTHORITY
            ).to_dict()
        ),
        "one_transition_per_boundary": len(segments) > 0 and n == len(segments) - 1,
        "timeline_boundary_count_matches": timeline.get("boundary_count") == n,
        "timeline_boundary_payload_count_matches": len(timeline_boundaries) == n,
        "proxy_transition_count_matches": final_review.get("transition_count") == n,
        "proxy_reviewed_boundary_count_matches": (
            final_review.get("reviewed_boundary_count") == n
        ),
        "proxy_transition_kinds_sum": (
            hard_count is not None
            and dissolve_count is not None
            and hard_count + dissolve_count == n
        ),
        "proxy_transition_kind_counts_match_plan": (
            hard_count == plan_hard_count
            and dissolve_count == plan_dissolve_count
        ),
        "group_transition_kinds_sum": (
            group_hard is not None
            and group_dissolve is not None
            and group_hard + group_dissolve == n
        ),
        "group_transition_kind_counts_match_plan": (
            group_hard == plan_hard_count
            and group_dissolve == plan_dissolve_count
        ),
        "visible_source_switch_count_matches": (
            timeline_switches == plan_cross_source_switch_count
        ),
        "group_introduced_hard_count_matches_plan": (
            introduced_hard == plan_introduced_hard_count
        ),
        "group_same_source_hold_count_matches_plan": (
            same_source_hold == plan_tier_d_hold_count
        ),
        "hard_cut_count_decomposition": (
            introduced_hard is not None
            and same_source_hold is not None
            and hard_count is not None
            and introduced_hard + same_source_hold == hard_count
        ),
        "output_frame_count_formula": (
            expected_frames is not None and output_frames == expected_frames
        ),
        "render_duration_formula": (
            expected_render_duration is not None
            and render_duration is not None
            and math.isclose(
                render_duration,
                expected_render_duration,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ),
        "proxy_decoded_frame_count_matches": (
            final_review.get("proxy", {}).get("decoded_frame_count") == output_frames
            if isinstance(final_review.get("proxy"), Mapping)
            else False
        ),
    }

    previous_end = 0.0
    segment_duration_sum = 0.0
    contiguous = True
    positive = True
    segment_source_indices: list[int] = []
    for index, raw in enumerate(segments):
        item = _mapping(raw)
        if item is None:
            contiguous = positive = False
            break
        master_start = _number(item.get("master_start"))
        item_duration = _number(item.get("duration"))
        source_index = _integer(item.get("source_index"))
        if source_index is not None:
            segment_source_indices.append(source_index)
        if master_start is None or item_duration is None or item_duration <= 0.0:
            positive = False
            continue
        if not math.isclose(master_start, previous_end, rel_tol=0.0, abs_tol=1e-6):
            contiguous = False
        previous_end = master_start + item_duration
        segment_duration_sum += item_duration
    checks["segments_positive"] = positive
    checks["segments_contiguous"] = contiguous
    checks["segment_duration_sum_matches_plan"] = (
        duration is not None
        and math.isclose(
            segment_duration_sum,
            duration,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    )

    sources = _sequence(plan.get("sources")) or []
    checks["segment_source_indices_in_range"] = (
        len(segment_source_indices) == len(segments)
        and all(0 <= index < len(sources) for index in segment_source_indices)
    )
    actual_used_source_indices = sorted(set(segment_source_indices))
    source_usage = _mapping(plan.get("source_usage")) or {}
    serialized_used_source_indices = _sequence(
        source_usage.get("used_source_indices")
    )
    checks["source_usage_matches_segments"] = (
        source_usage.get("distinct_source_count") == len(actual_used_source_indices)
        and serialized_used_source_indices == actual_used_source_indices
        and source_usage.get("total_segment_count") == len(segments)
        and (
            (usage_duration := _number(source_usage.get("total_duration_seconds")))
            is not None
        )
        and duration is not None
        and math.isclose(usage_duration, duration, rel_tol=0.0, abs_tol=1e-6)
    )
    request_payload = _mapping(analysis.get("request")) or {}
    render_payload = _mapping(analysis.get("render")) or {}
    automatic_member_direct_required = bool(
        str(request_payload.get("group", "")).strip().upper() == "IVE"
        and str(request_payload.get("source_mode", "")).strip().lower()
        == "automatic"
        and str(request_payload.get("edit_mode", "")).strip().lower()
        == "member"
        and render_payload.get("direct_full_frame") is True
    )
    member_direct = _mapping(analysis.get("member_direct_full_frame"))
    if member_direct is not None and member_direct.get("enabled") is True:
        raw_alignment_eligibility = _sequence(
            analysis.get("alignment_eligibility")
        )
        source_index_mapping = _mapping(analysis.get("source_index_mapping"))
        canonical_eligible_member_source_indices: list[int] = []
        canonical_alignment_eligibility_well_formed = bool(
            raw_alignment_eligibility is not None
            and len(raw_alignment_eligibility) == len(sources)
        )
        if raw_alignment_eligibility is not None:
            for position, raw_eligibility in enumerate(
                raw_alignment_eligibility
            ):
                eligibility = _mapping(raw_eligibility)
                if not (
                    eligibility is not None
                    and eligibility.get("index_domain")
                    == "request_source_index"
                    and eligibility.get("source_index") == position
                    and isinstance(eligibility.get("render_eligible"), bool)
                ):
                    canonical_alignment_eligibility_well_formed = False
                    continue
                if eligibility["render_eligible"] is True:
                    canonical_eligible_member_source_indices.append(position)
        expected_request_to_eligible: list[int | None] = [None] * len(sources)
        for eligible_position, request_source_index in enumerate(
            canonical_eligible_member_source_indices
        ):
            expected_request_to_eligible[request_source_index] = (
                eligible_position
            )
        canonical_source_mapping_matches = bool(
            source_index_mapping is not None
            and source_index_mapping.get("request_index_domain")
            == "request_source_index"
            and source_index_mapping.get("eligible_index_domain")
            == "eligible_source_index"
            and source_index_mapping.get("render_plan_index_domain")
            == "render_plan_sources_array_index"
            and source_index_mapping.get("eligible_to_request")
            == canonical_eligible_member_source_indices
            and source_index_mapping.get("request_to_eligible")
            == expected_request_to_eligible
        )
        eligible_member_sources = _sequence(
            member_direct.get("eligible_request_source_indices")
        )
        required_member_sources = _sequence(
            member_direct.get("required_used_request_source_indices")
        )
        direct_final_usage = _mapping(member_direct.get("final_source_usage")) or {}
        direct_used_sources = _sequence(
            direct_final_usage.get("used_source_indices")
        )
        eligible_member_source_indices = (
            [
                value
                for value in eligible_member_sources
                if isinstance(value, int) and not isinstance(value, bool)
            ]
            if eligible_member_sources is not None
            else []
        )
        member_source_contract_well_formed = bool(
            canonical_alignment_eligibility_well_formed
            and canonical_source_mapping_matches
            and eligible_member_sources is not None
            and eligible_member_source_indices == eligible_member_sources
            and len(set(eligible_member_source_indices))
            == len(eligible_member_source_indices)
            and eligible_member_source_indices
            == sorted(eligible_member_source_indices)
            and eligible_member_source_indices
            == canonical_eligible_member_source_indices
            and required_member_sources == eligible_member_source_indices
            and member_direct.get("minimum_required_distinct_source_count")
            == len(eligible_member_source_indices)
        )
        checks["member_direct_all_eligible_sources_used"] = bool(
            member_source_contract_well_formed
            and actual_used_source_indices == eligible_member_source_indices
            and direct_final_usage.get("distinct_source_count")
            == len(eligible_member_source_indices)
            and direct_used_sources == eligible_member_source_indices
            and dict(direct_final_usage) == dict(source_usage)
        )
    elif automatic_member_direct_required:
        checks["member_direct_all_eligible_sources_used"] = False
    else:
        checks["member_direct_all_eligible_sources_used"] = True
    checks["timeline_unique_sources_matches_plan"] = (
        timeline.get("unique_source_count") == len(actual_used_source_indices)
    )
    visual_choices = _sequence(
        (_mapping(analysis.get("visual_timeline")) or {}).get("choices")
    ) or []
    checks["visual_choices_match_plan_segments"] = (
        len(visual_choices) == len(segments)
        and all(
            (_mapping(choice) or {}).get("segment_index") == index
            and (_mapping(choice) or {}).get("source_index")
            == (_mapping(segments[index]) or {}).get("source_index")
            for index, choice in enumerate(visual_choices)
        )
    )
    timeline_duration = _number(timeline.get("total_duration_seconds"))
    checks["analysis_duration_contract_matches_plan"] = (
        duration is not None
        and timeline_duration is not None
        and common_start is not None
        and common_end is not None
        and common_duration is not None
        and math.isclose(timeline_duration, duration, rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(common_duration, duration, rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(
            common_end,
            common_start + common_duration,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    )

    transition_payloads_valid = True
    for index, transition_raw in enumerate(transitions):
        transition = _mapping(transition_raw) or {}
        left = _mapping(segments[index]) or {} if index < len(segments) else {}
        right = (
            _mapping(segments[index + 1]) or {}
            if index + 1 < len(segments)
            else {}
        )
        if transition.get("tier") == "D" and left.get("source_index") != right.get(
            "source_index"
        ):
            checks["tier_d_same_source"] = False
        if transition.get("kind") == "face_dissolve":
            if transition.get("tier") != "A" or left.get("source_index") == right.get(
                "source_index"
            ):
                checks["dissolve_tier_a_real_switch"] = False
        kind = transition.get("kind")
        frame_count = _integer(transition.get("frame_count"))
        transition_duration = _number(transition.get("duration_seconds"))
        if not (
            kind in {"hard_cut", "face_dissolve"}
            and frame_count is not None
            and frame_count >= 0
            and transition_duration is not None
            and fps is not None
            and fps > 0.0
            and math.isclose(
                transition_duration,
                frame_count / fps,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and (kind != "hard_cut" or frame_count == 0)
            and (kind != "face_dissolve" or frame_count > 0)
        ):
            transition_payloads_valid = False
    checks.setdefault("tier_d_same_source", True)
    checks.setdefault("dissolve_tier_a_real_switch", True)
    checks["transition_payloads_valid"] = transition_payloads_valid

    group_transitions = _sequence(group.get("transitions"))
    checks["group_transitions_match_render_plan"] = group_transitions == transitions

    render_payload = _mapping(analysis.get("render")) or {}
    final_segments = _sequence(render_payload.get("final_render_segments")) or []
    checks["final_render_segment_count_matches"] = len(final_segments) == len(segments)
    final_segments_match = len(final_segments) == len(segments)
    for index, (raw_plan_segment, raw_final_segment) in enumerate(
        zip(segments, final_segments, strict=False)
    ):
        plan_segment = _mapping(raw_plan_segment) or {}
        final_segment = _mapping(raw_final_segment) or {}
        plan_start = _number(plan_segment.get("master_start"))
        plan_duration = _number(plan_segment.get("duration"))
        final_start = _number(final_segment.get("output_start_seconds"))
        final_end = _number(final_segment.get("output_end_seconds"))
        final_canonical_start = _number(
            final_segment.get("canonical_master_start")
        )
        final_canonical_end = _number(final_segment.get("canonical_master_end"))
        plan_source_start = _number(plan_segment.get("source_local_start"))
        plan_source_end = _number(plan_segment.get("source_local_end"))
        if plan_source_start is None:
            plan_source_start = _number(plan_segment.get("source_start"))
        if plan_source_end is None:
            plan_source_end = _number(plan_segment.get("source_end"))
        final_source_start = _number(final_segment.get("source_local_start"))
        final_source_end = _number(final_segment.get("source_local_end"))
        if not (
            final_segment.get("segment_index") == index
            and final_segment.get("request_source_index")
            == plan_segment.get("source_index")
            and plan_start is not None
            and plan_duration is not None
            and final_start is not None
            and final_end is not None
            and final_canonical_start is not None
            and final_canonical_end is not None
            and common_start is not None
            and math.isclose(
                final_canonical_start,
                common_start + plan_start,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and math.isclose(
                final_canonical_end,
                common_start + plan_start + plan_duration,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and math.isclose(final_start, plan_start, rel_tol=0.0, abs_tol=1e-6)
            and math.isclose(
                final_end,
                plan_start + plan_duration,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and plan_source_start is not None
            and plan_source_end is not None
            and final_source_start is not None
            and final_source_end is not None
            and math.isclose(
                final_source_start,
                plan_source_start,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and math.isclose(
                final_source_end,
                plan_source_end,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            final_segments_match = False
    checks["final_render_segments_match_plan"] = final_segments_match

    proxy_boundaries = _sequence(final_review.get("boundaries")) or []
    proxy_boundaries_match = len(proxy_boundaries) == n
    timeline_boundaries_match = len(timeline_boundaries) == n
    derived_timeline_switch_count = 0
    cumulative_seconds = 0.0
    for index, transition_raw in enumerate(transitions):
        segment = _mapping(segments[index]) or {} if index < len(segments) else {}
        item_duration = _number(segment.get("duration"))
        if item_duration is None or fps is None:
            proxy_boundaries_match = False
            continue
        cumulative_seconds += item_duration
        expected_cut_frame = int(round(cumulative_seconds * fps))
        review_boundary = (
            _mapping(proxy_boundaries[index]) or {}
            if index < len(proxy_boundaries)
            else {}
        )
        timeline_boundary = (
            _mapping(timeline_boundaries[index]) or {}
            if index < len(timeline_boundaries)
            else {}
        )
        transition = _mapping(transition_raw) or {}
        left_source_index = (_mapping(segments[index]) or {}).get("source_index")
        right_source_index = (
            (_mapping(segments[index + 1]) or {}).get("source_index")
            if index + 1 < len(segments)
            else None
        )
        if not (
            review_boundary.get("boundary_index") == index
            and review_boundary.get("cut_frame_index") == expected_cut_frame
            and review_boundary.get("outgoing_source_index") == left_source_index
            and review_boundary.get("incoming_source_index") == right_source_index
            and review_boundary.get("transition_kind") == transition.get("kind")
            and review_boundary.get("transition_frame_count")
            == transition.get("frame_count")
        ):
            proxy_boundaries_match = False
        timeline_at = _number(timeline_boundary.get("at_seconds"))
        expected_source_switch = left_source_index != right_source_index
        derived_timeline_switch_count += expected_source_switch
        if not (
            timeline_boundary.get("boundary_index") == index
            and timeline_boundary.get("outgoing_segment_index") == index
            and timeline_boundary.get("incoming_segment_index") == index + 1
            and timeline_boundary.get("outgoing_source_id")
            == str(left_source_index)
            and timeline_boundary.get("incoming_source_id")
            == str(right_source_index)
            and timeline_boundary.get("source_switch") is expected_source_switch
            and timeline_at is not None
            and common_start is not None
            and math.isclose(
                timeline_at,
                common_start + cumulative_seconds,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            timeline_boundaries_match = False
    checks["proxy_boundary_schedule_matches_plan"] = proxy_boundaries_match
    checks["timeline_boundary_schedule_matches_plan"] = timeline_boundaries_match
    checks["timeline_source_switch_flags_match_plan"] = (
        timeline_switches == derived_timeline_switch_count
    )

    for name, passed in checks.items():
        if not passed:
            collector.add(
                f"plan_invariant_{name}",
                QAIssueSeverity.FAIL,
                "A render-plan, transition, timeline, or decoded-review invariant failed.",
                path=f"plan_invariants.{name}",
                value=passed,
                threshold=True,
            )
    return {
        "transition_count": n,
        "segment_count": len(segments),
        "checks": checks,
        "visible_source_switch_count": timeline_switches,
        "introduced_hard_cut_count": introduced_hard,
        "dissolve_count": group_dissolve,
        "derived_cross_source_switch_count": plan_cross_source_switch_count,
        "derived_native_source_cut_count": plan_native_source_cut_count,
        "plan_hard_cut_count": plan_hard_count,
        "plan_dissolve_count": plan_dissolve_count,
        "same_source_hold_count": same_source_hold,
        "used_source_indices": actual_used_source_indices,
    }


def _validate_ffprobe_against_plan(
    tool_checks: Mapping[str, Any],
    analysis: Mapping[str, Any],
    plan: Mapping[str, Any],
    config: PostRenderQAConfig,
    collector: _IssueCollector,
) -> dict[str, Any]:
    ffprobe = _mapping(tool_checks.get("ffprobe")) or {}
    payload = _mapping(ffprobe.get("payload"))
    if payload is None:
        return {"available": False}
    streams = _sequence(payload.get("streams")) or []
    video_streams = [
        item for raw in streams if (item := _mapping(raw)) is not None and item.get("codec_type") == "video"
    ]
    audio_streams = [
        item for raw in streams if (item := _mapping(raw)) is not None and item.get("codec_type") == "audio"
    ]
    checks: dict[str, bool] = {
        "exactly_one_video_stream": len(video_streams) == 1,
        "exactly_one_audio_stream": len(audio_streams) == 1,
    }
    if not checks["exactly_one_video_stream"]:
        video: dict[str, Any] = {}
    else:
        video = video_streams[0]
    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(
        video.get("r_frame_rate")
    )
    plan_fps = _number(plan.get("fps"))
    width = _probe_integer(video.get("width"))
    height = _probe_integer(video.get("height"))
    read_frames = _probe_integer(video.get("nb_read_frames")) or _probe_integer(
        video.get("nb_frames")
    )
    expected_frames = _integer(plan.get("output_frame_count"))
    stream_duration = _probe_number(video.get("duration"))
    format_payload = _mapping(payload.get("format")) or {}
    duration = stream_duration or _probe_number(format_payload.get("duration"))
    expected_duration = _number(plan.get("render_duration"))
    tolerance_seconds = (
        config.final_duration_tolerance_frames / plan_fps
        if plan_fps is not None and plan_fps > 0.0
        else 0.0
    )
    checks.update(
        {
            "width_matches": width == _integer(plan.get("width")),
            "height_matches": height == _integer(plan.get("height")),
            "fps_matches": (
                fps is not None
                and plan_fps is not None
                and abs(fps - plan_fps) <= config.ffprobe_fps_tolerance
            ),
            "decoded_frame_count_matches": read_frames == expected_frames,
            "duration_matches": (
                duration is not None
                and expected_duration is not None
                and abs(duration - expected_duration) <= tolerance_seconds + 1e-9
            ),
            "codec_matches": _codec_family(video.get("codec_name"))
            == _codec_family((_mapping(analysis.get("render")) or {}).get("video_codec")),
        }
    )
    if audio_streams and duration is not None:
        audio_duration = _probe_number(audio_streams[0].get("duration"))
        checks["audio_duration_matches_video"] = (
            audio_duration is not None
            and abs(audio_duration - duration) <= tolerance_seconds + 1e-9
        )
    else:
        checks["audio_duration_matches_video"] = False
        audio_duration = None

    for name, passed in checks.items():
        if not passed:
            collector.add(
                f"final_media_{name}",
                QAIssueSeverity.FAIL,
                "The decoded final MP4 does not match the render contract.",
                path=f"final_media.{name}",
                value=passed,
                threshold=True,
            )
    return {
        "available": True,
        "checks": checks,
        "video": {
            "codec_name": video.get("codec_name"),
            "width": width,
            "height": height,
            "fps": fps,
            "decoded_frame_count": read_frames,
            "duration_seconds": duration,
        },
        "audio": {
            "stream_count": len(audio_streams),
            "duration_seconds": audio_duration,
        },
        "duration_tolerance_seconds": tolerance_seconds,
    }


def _cadence_and_repetition_metrics(
    analysis: Mapping[str, Any],
    plan: Mapping[str, Any],
    config: PostRenderQAConfig,
    collector: _IssueCollector,
) -> dict[str, Any]:
    review = _mapping(analysis.get("timeline_review")) or {}
    if review.get("passed") is not True:
        collector.add(
            "timeline_review_failed",
            QAIssueSeverity.FAIL,
            "The final structural timeline review did not pass.",
            path="timeline_review.passed",
            value=review.get("passed"),
            threshold=True,
        )
    hard_failures = _sequence(review.get("hard_failures")) or []
    if hard_failures:
        collector.add(
            "timeline_hard_failures_present",
            QAIssueSeverity.FAIL,
            "The final timeline contains structural hard failures.",
            path="timeline_review.hard_failures",
            value=hard_failures,
            threshold=[],
        )
    warnings = _sequence(review.get("warnings")) or []
    if warnings:
        collector.add(
            "timeline_warnings_present",
            QAIssueSeverity.CONDITIONAL,
            "The final timeline retains structural warnings.",
            path="timeline_review.warnings",
            value=warnings,
            threshold=[],
        )

    plan_segments = [
        item
        for raw in (_sequence(plan.get("segments")) or [])
        if (item := _mapping(raw)) is not None
    ]
    total_duration = _number(plan.get("duration"))
    plan_source_sequence = [
        _integer(item.get("source_index")) for item in plan_segments
    ]
    segment_output_durations = [
        _number(item.get("duration")) for item in plan_segments
    ]
    plan_structure_available = (
        bool(plan_segments)
        and all(index is not None for index in plan_source_sequence)
        and all(
            duration is not None and duration > 0.0
            for duration in segment_output_durations
        )
    )
    switch_count = (
        sum(
            left != right
            for left, right in zip(
                plan_source_sequence, plan_source_sequence[1:]
            )
        )
        if plan_structure_available
        else None
    )
    serialized_switch_count = _integer(review.get("source_switch_count"))
    if switch_count is None or serialized_switch_count != switch_count:
        collector.add(
            "source_switch_count_plan_mismatch",
            QAIssueSeverity.FAIL,
            "Timeline source-switch count does not match the render plan.",
            path="timeline_review.source_switch_count",
            value={"serialized": serialized_switch_count, "derived": switch_count},
        )
    serialized_switches_per_minute = _number(
        review.get("source_switches_per_minute")
    )
    switches_per_minute = (
        switch_count * 60.0 / total_duration
        if switch_count is not None
        and total_duration is not None
        and total_duration > 0.0
        else None
    )
    automatic_ive_product_scope = _automatic_ive_group_product_scope(analysis)
    applied_preferred_maximum_switches_per_minute = (
        config.automatic_ive_preferred_maximum_source_switches_per_minute
        if automatic_ive_product_scope
        else config.preferred_maximum_source_switches_per_minute
    )
    if not (
        switches_per_minute is not None
        and serialized_switches_per_minute is not None
        and math.isclose(
            switches_per_minute,
            serialized_switches_per_minute,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        collector.add(
            "source_switch_density_payload_mismatch",
            QAIssueSeverity.FAIL,
            "Serialized source-switch density does not match count and duration.",
            path="timeline_review.source_switches_per_minute",
            value={
                "serialized": serialized_switches_per_minute,
                "derived": switches_per_minute,
            },
        )
    if switches_per_minute is None or (
        switches_per_minute < config.minimum_source_switches_per_minute_hard
    ):
        collector.add(
            "source_switch_density_below_hard_minimum",
            QAIssueSeverity.FAIL,
            "Source-switch density is below the production hard minimum.",
            path="timeline_review.source_switches_per_minute",
            value=switches_per_minute,
            threshold=config.minimum_source_switches_per_minute_hard,
        )
    elif not (
        config.preferred_minimum_source_switches_per_minute
        <= switches_per_minute
        <= applied_preferred_maximum_switches_per_minute
    ):
        collector.add(
            "source_switch_density_outside_professional_band",
            QAIssueSeverity.CONDITIONAL,
            "Source-switch density is outside the reference-calibrated preferred band.",
            path="timeline_review.source_switches_per_minute",
            value=switches_per_minute,
            threshold=[
                config.preferred_minimum_source_switches_per_minute,
                applied_preferred_maximum_switches_per_minute,
            ],
        )

    comparable_boundary_count = _integer(
        review.get("comparable_visual_boundary_count")
    )
    serialized_switch_ratio = _number(review.get("source_switch_rate"))
    switch_ratio = (
        switch_count / comparable_boundary_count
        if switch_count is not None
        and comparable_boundary_count is not None
        and comparable_boundary_count > 0
        else None
    )
    if not (
        switch_ratio is not None
        and serialized_switch_ratio is not None
        and math.isclose(
            switch_ratio,
            serialized_switch_ratio,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        collector.add(
            "source_switch_ratio_payload_mismatch",
            QAIssueSeverity.FAIL,
            "Serialized source-switch ratio does not match structural counts.",
            path="timeline_review.source_switch_rate",
            value={"serialized": serialized_switch_ratio, "derived": switch_ratio},
        )
    if switch_ratio is None or not (
        config.preferred_minimum_source_switch_ratio
        <= switch_ratio
        <= config.preferred_maximum_source_switch_ratio
    ):
        collector.add(
            "source_switch_ratio_outside_professional_band",
            QAIssueSeverity.CONDITIONAL,
            "Source-switch ratio is outside the reference-calibrated preferred band.",
            path="timeline_review.source_switch_rate",
            value=switch_ratio,
            threshold=[
                config.preferred_minimum_source_switch_ratio,
                config.preferred_maximum_source_switch_ratio,
            ],
        )

    shot_durations = [
        value
        for raw in (_sequence(review.get("visual_shot_durations_seconds")) or [])
        if (value := _number(raw)) is not None and value > 0.0
    ]
    serialized_shot_median = _number(review.get("visual_shot_median_seconds"))
    shot_median = _quantile(shot_durations, 0.5)
    if not (
        shot_median is not None
        and serialized_shot_median is not None
        and len(shot_durations) == review.get("visual_shot_count")
        and math.isclose(
            shot_median,
            serialized_shot_median,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        collector.add(
            "visual_shot_median_payload_mismatch",
            QAIssueSeverity.FAIL,
            "Serialized visual-shot cadence does not match its duration payload.",
            path="timeline_review.visual_shot_median_seconds",
            value={"serialized": serialized_shot_median, "derived": shot_median},
        )
    if shot_median is None or shot_median > config.maximum_visual_shot_median_seconds:
        collector.add(
            "visual_shot_median_too_slow",
            QAIssueSeverity.CONDITIONAL,
            "Visual-shot cadence is slower than the professional warning ceiling.",
            path="timeline_review.visual_shot_median_seconds",
            value=shot_median,
            threshold=config.maximum_visual_shot_median_seconds,
        )

    source_runs = [
        item
        for raw in (_sequence(review.get("source_runs")) or [])
        if (item := _mapping(raw)) is not None
    ]
    serialized_run_durations = [
        value
        for item in source_runs
        if (value := _number(item.get("duration_seconds"))) is not None
        and value > 0.0
    ]
    run_durations: list[float] = []
    if plan_structure_available:
        for source_index, duration in zip(
            plan_source_sequence,
            segment_output_durations,
            strict=True,
        ):
            assert source_index is not None and duration is not None
            if run_durations and source_index == previous_run_source:
                run_durations[-1] += duration
            else:
                run_durations.append(duration)
                previous_run_source = source_index
    serialized_run_median = _number(review.get("source_run_median_seconds"))
    run_median = _quantile(run_durations, 0.5)
    if not (
        run_median is not None
        and serialized_run_median is not None
        and len(serialized_run_durations) == len(source_runs)
        and len(run_durations) == len(serialized_run_durations)
        and all(
            math.isclose(left, right, rel_tol=0.0, abs_tol=1e-6)
            for left, right in zip(
                run_durations,
                serialized_run_durations,
                strict=True,
            )
        )
        and math.isclose(
            run_median,
            serialized_run_median,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        collector.add(
            "source_run_median_payload_mismatch",
            QAIssueSeverity.FAIL,
            "Serialized source-run median does not match its run payload.",
            path="timeline_review.source_run_median_seconds",
            value={"serialized": serialized_run_median, "derived": run_median},
        )
    if run_median is None or not (
        config.preferred_minimum_source_run_seconds
        <= run_median
        <= config.preferred_maximum_source_run_seconds
    ):
        collector.add(
            "source_run_median_outside_professional_band",
            QAIssueSeverity.CONDITIONAL,
            "Median continuous source run is outside the preferred band.",
            path="timeline_review.source_run_median_seconds",
            value=run_median,
            threshold=[
                config.preferred_minimum_source_run_seconds,
                config.preferred_maximum_source_run_seconds,
            ],
        )
    serialized_longest = _number(review.get("longest_source_run_seconds"))
    longest = max(run_durations, default=None)
    if not (
        longest is not None
        and serialized_longest is not None
        and math.isclose(
            longest,
            serialized_longest,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        collector.add(
            "source_run_longest_payload_mismatch",
            QAIssueSeverity.FAIL,
            "Serialized longest source run does not match its run payload.",
            path="timeline_review.longest_source_run_seconds",
            value={"serialized": serialized_longest, "derived": longest},
        )
    if longest is None or longest > config.maximum_source_run_seconds:
        collector.add(
            "source_run_exceeds_hard_maximum",
            QAIssueSeverity.FAIL,
            "A continuous source run exceeds the reference-p90 ceiling.",
            path="timeline_review.longest_source_run_seconds",
            value=longest,
            threshold=config.maximum_source_run_seconds,
        )

    derived_unique_sources = (
        len(set(plan_source_sequence)) if plan_structure_available else None
    )
    unique_sources = _integer(review.get("unique_source_count"))
    if derived_unique_sources is None or unique_sources != derived_unique_sources:
        collector.add(
            "unique_source_count_plan_mismatch",
            QAIssueSeverity.FAIL,
            "Timeline unique-source count does not match the render plan.",
            path="timeline_review.unique_source_count",
            value={"serialized": unique_sources, "derived": derived_unique_sources},
        )
    required_sources = _integer(review.get("required_unique_source_count"))
    available_sources = _integer(review.get("available_source_count"))
    minimum_sources = required_sources
    if (
        available_sources is not None
        and available_sources >= config.minimum_distinct_sources_when_required
    ):
        minimum_sources = max(
            minimum_sources or 0,
            config.minimum_distinct_sources_when_required,
        )
    if minimum_sources is None or (
        unique_sources is None or unique_sources < minimum_sources
    ):
        collector.add(
            "insufficient_source_diversity",
            QAIssueSeverity.FAIL,
            "The final edit uses fewer distinct sources than required.",
            path="timeline_review.unique_source_count",
            value=unique_sources,
            threshold=minimum_sources,
        )
    shares = _mapping(review.get("per_source_share")) or {}
    serialized_share_values: dict[int, float] = {}
    share_keys_valid = True
    for raw_key, raw_value in shares.items():
        value = _number(raw_value)
        if not (
            isinstance(raw_key, str)
            and raw_key.isdigit()
            and str(source_index := int(raw_key)) == raw_key
            and value is not None
            and source_index not in serialized_share_values
        ):
            share_keys_valid = False
            continue
        serialized_share_values[source_index] = value
    source_durations = _mapping(review.get("source_duration_seconds")) or {}
    numeric_source_durations: dict[int, float] = {}
    duration_keys_valid = True
    for raw_key, raw_value in source_durations.items():
        value = _number(raw_value)
        if not (
            isinstance(raw_key, str)
            and raw_key.isdigit()
            and str(source_index := int(raw_key)) == raw_key
            and value is not None
            and value > 0.0
            and source_index not in numeric_source_durations
        ):
            duration_keys_valid = False
            continue
        numeric_source_durations[source_index] = value
    plan_source_durations: dict[int, float] = {}
    if plan_structure_available:
        for source_index, duration in zip(
            plan_source_sequence,
            segment_output_durations,
            strict=True,
        ):
            assert source_index is not None and duration is not None
            plan_source_durations[source_index] = (
                plan_source_durations.get(source_index, 0.0) + duration
            )
    derived_shares = (
        {
            key: value / total_duration
            for key, value in numeric_source_durations.items()
        }
        if total_duration is not None and total_duration > 0.0
        else {}
    )
    source_share_payload_matches = (
        unique_sources is not None
        and share_keys_valid
        and duration_keys_valid
        and len(numeric_source_durations) == unique_sources
        and len(serialized_share_values) == unique_sources
        and len(plan_source_durations) == unique_sources
        and set(serialized_share_values) == set(derived_shares)
        and total_duration is not None
        and math.isclose(
            sum(numeric_source_durations.values()),
            total_duration,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and set(numeric_source_durations) == set(plan_source_durations)
        and all(
            math.isclose(
                numeric_source_durations[index],
                plan_source_durations[index],
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            for index in plan_source_durations
        )
        and all(
            math.isclose(
                serialized_share_values[key],
                derived_shares[key],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for key in derived_shares
        )
    )
    if not source_share_payload_matches:
        collector.add(
            "source_share_payload_mismatch",
            QAIssueSeverity.FAIL,
            "Source durations and shares do not match the final duration.",
            path="timeline_review.per_source_share",
            value={
                "serialized": serialized_share_values,
                "derived": derived_shares,
            },
        )
    share_values = list(derived_shares.values())
    dominant = max(share_values, default=None)
    if not (
        unique_sources is not None
        and len(share_values) == unique_sources
        and all(0.0 < value <= 1.0 for value in share_values)
        and math.isclose(sum(share_values), 1.0, rel_tol=0.0, abs_tol=1e-6)
    ):
        collector.add(
            "source_share_distribution_invalid",
            QAIssueSeverity.FAIL,
            "Per-source shares must cover used sources and sum to one.",
            path="timeline_review.per_source_share",
            value={"count": len(share_values), "sum": sum(share_values)},
            threshold={"count": unique_sources, "sum": 1.0},
        )
    if dominant is None or dominant > config.hard_maximum_dominant_source_share:
        collector.add(
            "dominant_source_share_hard",
            QAIssueSeverity.FAIL,
            "One source dominates too much of the final edit.",
            path="timeline_review.per_source_share",
            value=dominant,
            threshold=config.hard_maximum_dominant_source_share,
        )
    elif dominant > config.preferred_maximum_dominant_source_share:
        collector.add(
            "dominant_source_share_above_preferred",
            QAIssueSeverity.CONDITIONAL,
            "One source exceeds the preferred share.",
            path="timeline_review.per_source_share",
            value=dominant,
            threshold=config.preferred_maximum_dominant_source_share,
        )
    short_flashes = _integer(review.get("short_flash_count"))
    if short_flashes is None or short_flashes > 0:
        collector.add(
            "short_source_flash_present",
            QAIssueSeverity.FAIL,
            "The final edit contains an unreadably short internal source flash.",
            path="timeline_review.short_flash_count",
            value=short_flashes,
            threshold=0,
        )
    ping_pongs = _sequence(review.get("ping_pongs")) or []
    risky_ping_pongs = sum(
        (_mapping(item) or {}).get("within_review_window") is True
        for item in ping_pongs
    )
    if risky_ping_pongs:
        collector.add(
            "risky_source_ping_pong_present",
            QAIssueSeverity.CONDITIONAL,
            "The final edit contains a fast A-B-A source reversal.",
            path="timeline_review.ping_pongs",
            value=risky_ping_pongs,
            threshold=0,
        )
    for field_name in (
        "scene_cut_risk_count",
        "boundary_side_instability_count",
    ):
        value = _integer(review.get(field_name))
        if value is None or value > 0:
            collector.add(
                f"timeline_{field_name}",
                QAIssueSeverity.FAIL,
                "A source switch retains unsafe native-cut or boundary-side instability.",
                path=f"timeline_review.{field_name}",
                value=value,
                threshold=0,
            )
    return {
        "source_switches_per_minute": switches_per_minute,
        "preferred_source_switches_per_minute_band": [
            config.preferred_minimum_source_switches_per_minute,
            applied_preferred_maximum_switches_per_minute,
        ],
        "automatic_ive_product_cadence_band_applied": (
            automatic_ive_product_scope
        ),
        "source_switch_rate": switch_ratio,
        "visual_shot_median_seconds": shot_median,
        "source_run_median_seconds": run_median,
        "longest_source_run_seconds": longest,
        "unique_source_count": unique_sources,
        "required_unique_source_count": required_sources,
        "available_source_count": available_sources,
        "effective_minimum_unique_source_count": minimum_sources,
        "dominant_source_share": dominant,
        "source_share_sum": sum(share_values),
        "short_flash_count": short_flashes,
        "risky_ping_pong_count": risky_ping_pongs,
        "hard_failures": hard_failures,
        "warnings": warnings,
    }


def _proxy_quality_and_geometry_metrics(
    analysis: Mapping[str, Any],
    final_review: Mapping[str, Any],
    config: PostRenderQAConfig,
    collector: _IssueCollector,
) -> dict[str, Any]:
    if final_review.get("passed") is not True:
        collector.add(
            "decoded_proxy_review_failed",
            QAIssueSeverity.FAIL,
            "The final selected decoded proxy did not pass.",
            path="rendered_proxy_review.passed",
            value=final_review.get("passed"),
            threshold=True,
        )
    overall = _mapping(final_review.get("overall_quality")) or {}
    source_quality = _mapping(final_review.get("source_quality")) or {}
    boundary_score = _mapping(final_review.get("boundary_score")) or {}
    quality_p10 = _number(overall.get("p10"))
    source_median_span = _number(source_quality.get("median_span"))
    boundary_p10 = _number(boundary_score.get("p10"))
    weighted = _number(final_review.get("weighted_score"))
    proxy_payload = _mapping(final_review.get("proxy")) or {}
    decoded_frame_count = _integer(proxy_payload.get("decoded_frame_count"))
    overall_quality_count = _integer(overall.get("count"))
    if (
        decoded_frame_count is None
        or overall_quality_count != decoded_frame_count
    ):
        collector.add(
            "proxy_overall_quality_count_mismatch",
            QAIssueSeverity.FAIL,
            "Overall proxy quality must cover every decoded frame.",
            path="rendered_proxy_review.overall_quality.count",
            value={
                "quality_count": overall_quality_count,
                "decoded_frame_count": decoded_frame_count,
            },
        )
    gates = (
        (
            "overall_quality_p10",
            quality_p10,
            config.minimum_overall_quality_p10,
            lambda value, threshold: value >= threshold,
        ),
        (
            "source_quality_median_span",
            source_median_span,
            config.maximum_source_quality_median_span,
            lambda value, threshold: value <= threshold,
        ),
        (
            "boundary_score_p10",
            boundary_p10,
            config.minimum_boundary_score_p10,
            lambda value, threshold: value >= threshold,
        ),
        (
            "proxy_weighted_score",
            weighted,
            config.minimum_proxy_weighted_score,
            lambda value, threshold: value >= threshold,
        ),
    )
    for name, value, threshold, predicate in gates:
        if value is None or not predicate(value, threshold):
            collector.add(
                f"proxy_quality_{name}",
                QAIssueSeverity.FAIL,
                "Decoded proxy quality is outside the production gate.",
                path=f"rendered_proxy_review.{name}",
                value=value,
                threshold=threshold,
            )
    issue_count = _integer(boundary_score.get("issue_count"))
    if issue_count is None or issue_count > config.professional_maximum_boundary_issue_count:
        collector.add(
            "proxy_boundary_issues_require_manual_review",
            QAIssueSeverity.CONDITIONAL,
            "The proxy passed globally but retains at least one known boundary issue.",
            path="rendered_proxy_review.boundary_score.issue_count",
            value=issue_count,
            threshold=config.professional_maximum_boundary_issue_count,
        )
    geometry_failures = _integer(final_review.get("geometry_provider_failure_count"))
    if geometry_failures is None or (
        geometry_failures
        > config.professional_maximum_geometry_provider_failure_count
    ):
        collector.add(
            "proxy_geometry_provider_failures",
            QAIssueSeverity.CONDITIONAL,
            "Some decoded proxy boundary geometry could not be measured.",
            path="rendered_proxy_review.geometry_provider_failure_count",
            value=geometry_failures,
            threshold=config.professional_maximum_geometry_provider_failure_count,
        )

    boundaries = _sequence(final_review.get("boundaries")) or []
    boundary_indices = [
        _integer((_mapping(item) or {}).get("boundary_index"))
        for item in boundaries
    ]
    if boundary_indices != list(range(len(boundaries))):
        collector.add(
            "proxy_boundary_indices_invalid",
            QAIssueSeverity.FAIL,
            "Decoded proxy boundary indices are not unique and sequential.",
            path="rendered_proxy_review.boundaries[].boundary_index",
            value=boundary_indices,
            threshold=list(range(len(boundaries))),
        )
    actual_issue_count = sum(
        (_mapping(item) or {}).get("passed") is not True for item in boundaries
    )
    actual_issue_indices = [
        index
        for index, item in enumerate(boundaries)
        if (_mapping(item) or {}).get("passed") is not True
    ]
    serialized_issue_indices = _sequence(
        final_review.get("boundary_issue_indices")
    )
    if serialized_issue_indices != actual_issue_indices:
        collector.add(
            "proxy_boundary_issue_indices_mismatch",
            QAIssueSeverity.FAIL,
            "Serialized boundary issue indices do not match boundary pass flags.",
            path="rendered_proxy_review.boundary_issue_indices",
            value={
                "serialized": serialized_issue_indices,
                "actual": actual_issue_indices,
            },
        )
    if issue_count != actual_issue_count:
        collector.add(
            "proxy_boundary_issue_count_mismatch",
            QAIssueSeverity.FAIL,
            "Serialized boundary issue count does not match boundary pass flags.",
            path="rendered_proxy_review.boundary_score.issue_count",
            value={"serialized": issue_count, "actual": actual_issue_count},
        )
    boundary_scores = [
        value
        for raw in boundaries
        if (value := _number((_mapping(raw) or {}).get("weighted_score")))
        is not None
    ]
    boundary_distribution = _mapping(boundary_score.get("distribution")) or {}
    derived_boundary_p10 = _quantile(boundary_scores, 0.10)
    if not (
        len(boundary_scores) == len(boundaries)
        and boundary_distribution.get("count") == len(boundaries)
        and derived_boundary_p10 is not None
        and boundary_p10 is not None
        and math.isclose(
            derived_boundary_p10,
            boundary_p10,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        collector.add(
            "proxy_boundary_score_distribution_mismatch",
            QAIssueSeverity.FAIL,
            "Boundary-score summary does not match serialized boundary reviews.",
            path="rendered_proxy_review.boundary_score",
            value={
                "serialized_p10": boundary_p10,
                "derived_p10": derived_boundary_p10,
                "score_count": len(boundary_scores),
            },
        )
    insufficient = 0
    covered = 0
    closeup_relevant = 0
    closeup_covered = 0
    discontinuity_fields = (
        "face_scale_discontinuity",
        "face_position_discontinuity",
        "silhouette_scale_discontinuity",
        "silhouette_position_discontinuity",
    )
    discontinuities: dict[str, list[float]] = {
        field_name: [] for field_name in discontinuity_fields
    }
    discontinuities_by_kind: dict[str, dict[str, list[float]]] = {}
    boundary_sample_contract_valid = True
    choices = _sequence((_mapping(analysis.get("visual_timeline")) or {}).get("choices")) or []
    for index, raw in enumerate(boundaries):
        boundary = _mapping(raw) or {}
        transition_kind = str(boundary.get("transition_kind") or "unknown")
        cut_frame = _integer(boundary.get("cut_frame_index"))
        transition_frames = _integer(boundary.get("transition_frame_count"))
        expected_offsets = (
            [-2, -1, 0, 1]
            if transition_kind == "hard_cut"
            else [-2, -1, transition_frames, transition_frames + 1]
            if transition_kind == "face_dissolve"
            and transition_frames is not None
            and transition_frames > 0
            else None
        )
        sample_offsets = _sequence(boundary.get("sample_offsets"))
        frame_indices = _sequence(boundary.get("frame_indices"))
        by_kind = discontinuities_by_kind.setdefault(
            transition_kind,
            {field_name: [] for field_name in discontinuity_fields},
        )
        serialized_discontinuities = _mapping(boundary.get("discontinuities")) or {}
        for field_name in discontinuity_fields:
            value = _number(serialized_discontinuities.get(field_name))
            if value is not None:
                discontinuities[field_name].append(value)
                by_kind[field_name].append(value)
        reasons = _sequence(boundary.get("issue_reasons")) or []
        insufficient += "insufficient_post_transition_review_handle" in reasons
        metrics = _sequence(boundary.get("frame_metrics")) or []
        if not (
            expected_offsets is not None
            and cut_frame is not None
            and sample_offsets == expected_offsets
            and frame_indices
            == [cut_frame + offset for offset in expected_offsets]
            and len(metrics) == len(expected_offsets)
            and all(
                (_mapping(item) or {}).get("offset") == offset
                and (_mapping(item) or {}).get("frame_index")
                == cut_frame + offset
                for item, offset in zip(metrics, expected_offsets, strict=True)
            )
        ):
            boundary_sample_contract_valid = False
        geometries = [
            _mapping((_mapping(item) or {}).get("subject_geometry")) or {}
            for item in metrics
        ]
        pair = (geometries[1], geometries[2]) if len(geometries) >= 3 else ({}, {})
        face_pair = all(item.get("face_bbox_xywh") is not None for item in pair)
        silhouette_pair = all(
            item.get("silhouette_bbox_xywh") is not None for item in pair
        )
        covered += face_pair or silhouette_pair
        left_choice = _mapping(choices[index]) or {} if index < len(choices) else {}
        right_choice = (
            _mapping(choices[index + 1]) or {}
            if index + 1 < len(choices)
            else {}
        )
        left_type = left_choice.get("exit_shot_type") or left_choice.get("shot_type")
        right_type = right_choice.get("entry_shot_type") or right_choice.get("shot_type")
        relevant = "close_up" in {left_type, right_type}
        closeup_relevant += relevant
        closeup_covered += relevant and face_pair
    if not boundary_sample_contract_valid:
        collector.add(
            "proxy_boundary_sample_contract_invalid",
            QAIssueSeverity.FAIL,
            "Boundary samples do not match duration-aware hard-cut/dissolve endpoints.",
            path="rendered_proxy_review.boundaries",
            value=False,
            threshold=True,
        )
    if insufficient:
        collector.add(
            "insufficient_post_transition_review_handle",
            QAIssueSeverity.FAIL,
            "A dissolve lacks enough completed-transition review handle.",
            path="rendered_proxy_review.boundaries",
            value=insufficient,
            threshold=0,
        )
    geometry_coverage = _fraction(covered, len(boundaries))
    closeup_coverage = _fraction(closeup_covered, closeup_relevant)
    if geometry_coverage is None or geometry_coverage < 1.0:
        collector.add(
            "proxy_subject_geometry_coverage_incomplete",
            QAIssueSeverity.CONDITIONAL,
            "Not every transition has paired face or silhouette endpoint geometry.",
            path="rendered_proxy_review.boundaries[].frame_metrics",
            value=geometry_coverage,
            threshold=1.0,
        )
    if closeup_relevant and (closeup_coverage is None or closeup_coverage < 1.0):
        collector.add(
            "proxy_closeup_face_coverage_incomplete",
            QAIssueSeverity.CONDITIONAL,
            "A close-up transition lacks paired decoded face geometry.",
            path="rendered_proxy_review.boundaries[].frame_metrics",
            value=closeup_coverage,
            threshold=1.0,
        )

    choices_used = {
        _integer((_mapping(choice) or {}).get("source_index"))
        for choice in choices
    }
    choices_used.discard(None)
    quality_items = [
        item
        for raw in (_sequence(source_quality.get("per_source")) or [])
        if (item := _mapping(raw)) is not None
    ]
    quality_indices = [
        _integer(item.get("source_index")) for item in quality_items
    ]
    missing_quality_indices = sorted(
        index for index in choices_used if index not in quality_indices
    )
    duplicate_quality_indices = sorted(
        index
        for index, count in Counter(quality_indices).items()
        if index is not None and count > 1
    )
    if missing_quality_indices or duplicate_quality_indices:
        collector.add(
            "used_source_quality_coverage_invalid",
            QAIssueSeverity.FAIL,
            "Decoded quality evidence must cover every used source exactly once.",
            path="rendered_proxy_review.source_quality.per_source",
            value={
                "missing": missing_quality_indices,
                "duplicates": duplicate_quality_indices,
            },
        )

    look = _mapping(analysis.get("source_look_match"))
    look_metrics: dict[str, Any] = {"available": look is not None}
    if look is None:
        collector.add(
            "source_look_match_unavailable",
            QAIssueSeverity.CONDITIONAL,
            "Source look-match evidence is unavailable.",
            path="source_look_match",
        )
    else:
        evidence = _sequence(look.get("evidence")) or []
        evidence_items = [
            item for raw in evidence if (item := _mapping(raw)) is not None
        ]
        evidence_indices = [
            _integer(item.get("source_index")) for item in evidence_items
        ]
        missing_look_indices = sorted(
            index for index in choices_used if index not in evidence_indices
        )
        duplicate_look_indices = sorted(
            index
            for index, count in Counter(evidence_indices).items()
            if index is not None and count > 1
        )
        if missing_look_indices or duplicate_look_indices:
            collector.add(
                "used_source_look_evidence_coverage_invalid",
                QAIssueSeverity.CONDITIONAL,
                "Look-match evidence does not cover every used source exactly once.",
                path="source_look_match.evidence",
                value={
                    "missing": missing_look_indices,
                    "duplicates": duplicate_look_indices,
                },
            )
        unstable_used: list[int] = []
        insufficient_used: list[int] = []
        for item in evidence_items:
            source_index = _integer(item.get("source_index"))
            if source_index not in choices_used:
                continue
            if item.get("stable") is not True:
                unstable_used.append(source_index)  # type: ignore[arg-type]
            decoded_samples = _integer(item.get("decoded_sample_count"))
            if decoded_samples is None or decoded_samples < 6:
                insufficient_used.append(source_index)  # type: ignore[arg-type]
        if look.get("target_mode") == "unavailable_identity" or unstable_used:
            collector.add(
                "used_source_look_evidence_unstable",
                QAIssueSeverity.CONDITIONAL,
                "At least one used source lacks stable synchronized look evidence.",
                path="source_look_match.evidence",
                value=sorted(unstable_used),
                threshold=[],
            )
        if insufficient_used:
            collector.add(
                "used_source_look_samples_insufficient",
                QAIssueSeverity.CONDITIONAL,
                "At least one used source has fewer than six look samples.",
                path="source_look_match.evidence[].decoded_sample_count",
                value=sorted(insufficient_used),
                threshold=6,
            )
        look_metrics.update(
            {
                "target_mode": look.get("target_mode"),
                "used_source_indices": sorted(choices_used),
                "unstable_used_source_indices": sorted(unstable_used),
                "insufficient_used_source_indices": sorted(insufficient_used),
                "missing_used_source_indices": missing_look_indices,
                "duplicate_source_indices": duplicate_look_indices,
            }
        )
    return {
        "overall_quality_p10": quality_p10,
        "source_quality_median_span": source_median_span,
        "used_source_indices": sorted(choices_used),
        "missing_used_source_quality_indices": missing_quality_indices,
        "duplicate_source_quality_indices": duplicate_quality_indices,
        "boundary_score_p10": boundary_p10,
        "weighted_score": weighted,
        "boundary_issue_count": issue_count,
        "actual_boundary_issue_count": actual_issue_count,
        "actual_boundary_issue_indices": actual_issue_indices,
        "boundary_sample_contract_valid": boundary_sample_contract_valid,
        "derived_boundary_score_p10": derived_boundary_p10,
        "geometry_provider_failure_count": geometry_failures,
        "insufficient_transition_handle_count": insufficient,
        "subject_geometry_coverage": geometry_coverage,
        "closeup_relevant_boundary_count": closeup_relevant,
        "closeup_face_coverage": closeup_coverage,
        "subject_discontinuities": {
            field_name: _distribution(values)
            for field_name, values in discontinuities.items()
        },
        "subject_discontinuities_by_transition_kind": {
            kind: {
                field_name: _distribution(values)
                for field_name, values in fields.items()
            }
            for kind, fields in sorted(discontinuities_by_kind.items())
        },
        "source_look_match": look_metrics,
    }


def _source_local_reuse_metrics(
    plan: Mapping[str, Any],
    config: PostRenderQAConfig,
    collector: _IssueCollector,
) -> dict[str, Any]:
    segments = _sequence(plan.get("segments")) or []
    by_source: dict[int, list[tuple[float, float, int]]] = {}
    total = 0.0
    for index, raw in enumerate(segments):
        item = _mapping(raw) or {}
        source_index = _integer(item.get("source_index"))
        start = _number(item.get("source_local_start"))
        end = _number(item.get("source_local_end"))
        if start is None:
            start = _number(item.get("source_start"))
        if end is None:
            end = _number(item.get("source_end"))
        if source_index is None or start is None or end is None or end <= start:
            collector.add(
                "source_local_interval_invalid",
                QAIssueSeverity.FAIL,
                "A render segment lacks a valid source-local interval.",
                path=f"render_plan.segments[{index}]",
                value={"source_index": source_index, "start": start, "end": end},
            )
            continue
        by_source.setdefault(source_index, []).append((start, end, index))
        total += end - start

    union_total = 0.0
    overlaps: list[dict[str, Any]] = []
    per_source: list[dict[str, Any]] = []
    for source_index, intervals in sorted(by_source.items()):
        ordered = sorted(intervals)
        cursor_start, cursor_end, cursor_index = ordered[0]
        source_union = 0.0
        source_overlap = 0.0
        for start, end, segment_index in ordered[1:]:
            if start < cursor_end - config.maximum_source_local_reuse_seconds:
                overlap = min(cursor_end, end) - start
                if overlap > config.maximum_source_local_reuse_seconds:
                    source_overlap += overlap
                    overlaps.append(
                        {
                            "source_index": source_index,
                            "left_segment_index": cursor_index,
                            "right_segment_index": segment_index,
                            "overlap_seconds": overlap,
                        }
                    )
            if start > cursor_end:
                source_union += cursor_end - cursor_start
                cursor_start, cursor_end, cursor_index = start, end, segment_index
            elif end > cursor_end:
                cursor_end = end
                cursor_index = segment_index
        source_union += cursor_end - cursor_start
        union_total += source_union
        per_source.append(
            {
                "source_index": source_index,
                "segment_interval_count": len(intervals),
                "union_duration_seconds": source_union,
                "overlap_seconds": source_overlap,
            }
        )
    reused = max(0.0, total - union_total)
    ratio = reused / total if total > 0.0 else None
    if overlaps or reused > config.maximum_source_local_reuse_seconds:
        collector.add(
            "source_local_content_reused",
            QAIssueSeverity.FAIL,
            "The final plan reuses an already-consumed source-local interval.",
            path="render_plan.segments",
            value={"reused_seconds": reused, "overlaps": overlaps},
            threshold=config.maximum_source_local_reuse_seconds,
        )
    return {
        "total_source_local_duration_seconds": total,
        "union_source_local_duration_seconds": union_total,
        "reused_source_local_duration_seconds": reused,
        "reused_content_ratio": ratio,
        "overlaps": overlaps,
        "per_source": per_source,
    }


def _normalized_entropy(values: Sequence[str]) -> float | None:
    if not values:
        return None
    counts = Counter(values)
    if len(counts) == 1:
        return 0.0
    total = len(values)
    entropy = -sum(
        (count / total) * math.log(count / total) for count in counts.values()
    )
    return entropy / math.log(len(counts))


def _transform_metrics(
    analysis: Mapping[str, Any],
    plan: Mapping[str, Any],
    config: PostRenderQAConfig,
    collector: _IssueCollector,
) -> dict[str, Any]:
    motion = _mapping(analysis.get("editorial_motion"))
    if motion is None:
        collector.add(
            "editorial_motion_unavailable",
            QAIssueSeverity.CONDITIONAL,
            "Editorial transform planning is unavailable.",
            path="editorial_motion",
        )
        return {"available": False}
    segments = _sequence(motion.get("segments")) or []
    motion_segments = [_mapping(raw) or {} for raw in segments]
    plan_segments = _sequence(plan.get("segments")) or []
    motion_indices = [
        _integer(item.get("segment_index")) for item in motion_segments
    ]
    plan_transitions = _sequence(plan.get("transitions")) or []
    dissolve_endpoint_indices = {
        segment_index
        for transition_index, raw in enumerate(plan_transitions)
        if (_mapping(raw) or {}).get("kind") == "face_dissolve"
        for segment_index in (transition_index, transition_index + 1)
    }
    motion_contract_checks = {
        "one_motion_segment_per_plan_segment": (
            motion_indices == list(range(len(plan_segments)))
        ),
        "motion_source_indices_match_plan": (
            len(motion_segments) == len(plan_segments)
            and all(
                item.get("source_index")
                == (_mapping(plan_segments[index]) or {}).get("source_index")
                for index, item in enumerate(motion_segments)
            )
        ),
        "serialized_applied_segment_count_matches": (
            motion.get("applied_segment_count")
            == sum(item.get("applied") is True for item in motion_segments)
        ),
        "applied_flags_boolean": all(
            isinstance(item.get("applied"), bool) for item in motion_segments
        ),
        "event_counts_nonnegative_integers": all(
            (value := _integer(motion.get(field_name))) is not None and value >= 0
            for field_name in (
                "boundary_match_count",
                "affine_boundary_match_count",
                "zoom_event_count",
                "pan_event_count",
                "roll_event_count",
                "interior_zoom_pulse_count",
            )
        ),
        "dissolve_endpoint_transforms_disabled": all(
            item.get("applied") is not True
            for item in motion_segments
            if _integer(item.get("segment_index")) in dissolve_endpoint_indices
        ),
    }
    for name, passed in motion_contract_checks.items():
        if not passed:
            collector.add(
                f"editorial_motion_{name}",
                QAIssueSeverity.FAIL,
                "Editorial motion segments do not match the final render plan.",
                path=f"editorial_motion.{name}",
                value=passed,
                threshold=True,
            )
    applied = [item for item in motion_segments if item.get("applied") is True]
    signatures: list[str] = []
    reason_categories: list[str] = []
    unsafe_keyframes: list[dict[str, Any]] = []
    invalid_tracks: list[dict[str, Any]] = []
    final_segments = {
        _integer(item.get("segment_index")): item
        for raw in (
            _sequence(
                (_mapping(analysis.get("render")) or {}).get(
                    "final_render_segments"
                )
            )
            or []
        )
        if (item := _mapping(raw)) is not None
        and _integer(item.get("segment_index")) is not None
    }
    global_state_points: list[
        tuple[float, float, float, float, float]
    ] = []
    for item in applied:
        serialized_reasons = tuple(
            sorted(
                str(value)
                for value in (_sequence(item.get("reasons")) or [])
                if isinstance(value, str) and value.strip()
            )
        )
        keyframes = [
            child
            for raw in (_sequence(item.get("keyframes")) or [])
            if (child := _mapping(raw)) is not None
        ]
        zooms = [value for frame in keyframes if (value := _number(frame.get("zoom"))) is not None]
        centers = [
            (x, y)
            for frame in keyframes
            if (x := _number(frame.get("center_x"))) is not None
            and (y := _number(frame.get("center_y"))) is not None
        ]
        rolls = [value for frame in keyframes if (value := _number(frame.get("roll_degrees"))) is not None]
        times = [value for frame in keyframes if (value := _number(frame.get("time_seconds"))) is not None]
        segment_index = _integer(item.get("segment_index"))
        plan_segment = (
            _mapping(plan_segments[segment_index]) or {}
            if segment_index is not None and 0 <= segment_index < len(plan_segments)
            else {}
        )
        plan_transform = _mapping(plan_segment.get("editorial_transform")) or {}
        plan_transform_reason = plan_transform.get("reason")
        plan_transform_keyframes = _sequence(plan_transform.get("keyframes"))
        fallback_reason_bound_to_render_plan = bool(
            not serialized_reasons
            and isinstance(plan_transform_reason, str)
            and plan_transform_reason.strip()
            and plan_transform_keyframes == keyframes
        )
        reasons = (
            serialized_reasons
            if serialized_reasons
            else (
                (plan_transform_reason,)
                if fallback_reason_bound_to_render_plan
                else ()
            )
        )
        segment_duration = _number(plan_segment.get("duration"))
        full_numeric_keyframe_count = sum(
            all(
                _number(frame.get(field_name)) is not None
                for field_name in (
                    "time_seconds",
                    "zoom",
                    "center_x",
                    "center_y",
                    "roll_degrees",
                )
            )
            for frame in keyframes
        )
        valid_track = (
            bool(reasons)
            and (bool(serialized_reasons) or fallback_reason_bound_to_render_plan)
            and bool(keyframes)
            and full_numeric_keyframe_count == len(keyframes)
            and len(times) == len(keyframes)
            and math.isclose(times[0], 0.0, rel_tol=0.0, abs_tol=1e-9)
            and all(left < right for left, right in zip(times, times[1:]))
            and segment_duration is not None
            and times[-1] <= segment_duration + 1e-6
        )
        if not valid_track:
            invalid_tracks.append(
                {
                    "segment_index": segment_index,
                    "keyframe_count": len(keyframes),
                    "numeric_keyframe_count": full_numeric_keyframe_count,
                    "times_seconds": times,
                    "segment_duration_seconds": segment_duration,
                    "serialized_reasons": list(serialized_reasons),
                    "effective_reasons": list(reasons),
                    "plan_transform_reason": plan_transform_reason,
                    "fallback_reason_bound_to_render_plan": (
                        fallback_reason_bound_to_render_plan
                    ),
                }
            )
        max_zoom = max(zooms, default=1.0)
        min_zoom = min(zooms, default=1.0)
        max_shift = max(
            (max(abs(x - 0.5), abs(y - 0.5)) for x, y in centers),
            default=0.0,
        )
        max_roll = max((abs(value) for value in rolls), default=0.0)
        if (
            min_zoom < 1.0 - 1e-9
            or
            max_zoom > config.maximum_editorial_zoom + 1e-9
            or max_shift > config.maximum_editorial_center_shift + 1e-9
            or max_roll > config.maximum_editorial_roll_degrees + 1e-9
        ):
            unsafe_keyframes.append(
                {
                    "segment_index": item.get("segment_index"),
                    "minimum_zoom": min_zoom,
                    "maximum_zoom": max_zoom,
                    "maximum_center_shift": max_shift,
                    "maximum_roll_degrees": max_roll,
                }
            )
        for reason in reasons:
            reason_categories.append(reason.split(":", 1)[0])
        signature_reasons = tuple(
            sorted({reason.split(":", 1)[0] for reason in reasons})
        )
        zoom_bin = round((max_zoom - 1.0) / 0.01) if max_zoom > 1.0 else 0
        pan_angle_bin = 0
        if centers and max_shift > 1e-9:
            furthest = max(
                centers,
                key=lambda point: math.hypot(
                    point[0] - 0.5, point[1] - 0.5
                ),
            )
            pan_angle_bin = int(
                round(math.atan2(furthest[1] - 0.5, furthest[0] - 0.5) / (math.pi / 4.0))
            )
        pan_magnitude_bin = round(max_shift / 0.005)
        signed_peak_roll = max(rolls, key=abs, default=0.0)
        roll_bin = round(signed_peak_roll / 0.25)
        duration_bin = (
            round((max(times) - min(times)) / 0.1) if len(times) >= 2 else 0
        )
        signatures.append(
            json.dumps(
                {
                    "reason_categories": signature_reasons,
                    "zoom_bin_1pct": zoom_bin,
                    "pan_octant": pan_angle_bin,
                    "pan_magnitude_bin_half_percent": pan_magnitude_bin,
                    "roll_bin_quarter_degree": roll_bin,
                    "track_duration_bin_100ms": duration_bin,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        final_segment = final_segments.get(segment_index, {})
        output_start = _number(final_segment.get("output_start_seconds"))
        if keyframes and output_start is not None:
            def state_components(
                frame: Mapping[str, Any],
            ) -> tuple[float, float, float, float]:
                zoom = _number(frame.get("zoom"))
                center_x = _number(frame.get("center_x"))
                center_y = _number(frame.get("center_y"))
                roll = _number(frame.get("roll_degrees"))
                zoom_delta = abs((zoom if zoom is not None else 1.0) - 1.0)
                pan_delta = math.hypot(
                    (center_x if center_x is not None else 0.5) - 0.5,
                    (center_y if center_y is not None else 0.5) - 0.5,
                )
                roll_delta = abs(roll if roll is not None else 0.0)
                magnitude = zoom_delta + pan_delta + roll_delta / 10.0
                return (magnitude, zoom_delta, pan_delta, roll_delta)

            for keyframe in keyframes:
                local_time = _number(keyframe.get("time_seconds"))
                if local_time is not None:
                    magnitude, zoom_delta, pan_delta, roll_delta = (
                        state_components(keyframe)
                    )
                    global_state_points.append(
                        (
                            output_start + local_time,
                            magnitude,
                            zoom_delta,
                            pan_delta,
                            roll_delta,
                        )
                    )
    non_applied_with_keyframes = [
        item.get("segment_index")
        for item in motion_segments
        if item.get("applied") is not True
        and bool(_sequence(item.get("keyframes")) or [])
    ]
    if invalid_tracks or non_applied_with_keyframes:
        collector.add(
            "editorial_motion_track_payload_invalid",
            QAIssueSeverity.FAIL,
            "Applied transform tracks must be numeric, ordered, bounded by their segment, and consistently marked.",
            path="editorial_motion.segments",
            value={
                "invalid_applied_tracks": invalid_tracks,
                "non_applied_segments_with_keyframes": non_applied_with_keyframes,
            },
        )
    for item in motion_segments:
        if item.get("applied") is True:
            continue
        segment_index = _integer(item.get("segment_index"))
        final_segment = final_segments.get(segment_index, {})
        output_start = _number(final_segment.get("output_start_seconds"))
        output_end = _number(final_segment.get("output_end_seconds"))
        if output_start is not None:
            global_state_points.append((output_start, 0.0, 0.0, 0.0, 0.0))
        if output_end is not None:
            global_state_points.append((output_end, 0.0, 0.0, 0.0, 0.0))
    if unsafe_keyframes:
        collector.add(
            "editorial_transform_safety_bound_exceeded",
            QAIssueSeverity.FAIL,
            "Serialized editorial keyframes exceed conservative transform bounds.",
            path="editorial_motion.segments[].keyframes",
            value=unsafe_keyframes,
        )

    section_plan = _mapping(analysis.get("section_plan")) or {}
    sections = _sequence(section_plan.get("sections")) or []
    target_zoom = 0
    if sections:
        target_zoom = int(
            round(
                sum(
                    (_number((_mapping(raw) or {}).get("duration_seconds")) or 0.0)
                    * (
                        4.0
                        + 8.0
                        * (
                            _number((_mapping(raw) or {}).get("camera_motion_budget"))
                            or 0.0
                        )
                    )
                    / 60.0
                    for raw in sections
                )
            )
        )
    actual_zoom = _integer(motion.get("zoom_event_count"))

    consecutive_duplicate_count = sum(
        left == right for left, right in zip(signatures, signatures[1:])
    )
    fps = _number(plan.get("fps")) or 0.0
    states_by_time: dict[float, tuple[float, float, float, float]] = {}
    for event_time, magnitude, zoom_delta, pan_delta, roll_delta in global_state_points:
        prior = states_by_time.get(event_time, (0.0, 0.0, 0.0, 0.0))
        states_by_time[event_time] = (
            max(magnitude, prior[0]),
            max(zoom_delta, prior[1]),
            max(pan_delta, prior[2]),
            max(roll_delta, prior[3]),
        )
    ordered_state_points = sorted(states_by_time.items())
    deduplicated_event_times: list[float] = []
    derived_zoom_event_count = 0
    derived_pan_event_count = 0
    derived_roll_event_count = 0
    active_excursion: list[tuple[float, float, float, float, float]] = []

    def finish_excursion() -> None:
        nonlocal derived_zoom_event_count
        nonlocal derived_pan_event_count
        nonlocal derived_roll_event_count
        if not active_excursion:
            return
        deduplicated_event_times.append(
            max(active_excursion, key=lambda item: item[1])[0]
        )
        derived_zoom_event_count += any(item[2] > 1e-9 for item in active_excursion)
        derived_pan_event_count += any(item[3] > 1e-9 for item in active_excursion)
        derived_roll_event_count += any(item[4] > 1e-9 for item in active_excursion)

    for event_time, state in ordered_state_points:
        magnitude, zoom_delta, pan_delta, roll_delta = state
        if magnitude > 1e-9:
            active_excursion.append(
                (event_time, magnitude, zoom_delta, pan_delta, roll_delta)
            )
            continue
        if active_excursion:
            finish_excursion()
            active_excursion = []
    if active_excursion:
        finish_excursion()
    reason_sets = [
        {
            str(reason).split(":", 1)[0]
            for reason in (_sequence(item.get("reasons")) or [])
        }
        for item in motion_segments
    ]
    derived_boundary_match_count = 0
    derived_affine_boundary_match_count = 0
    for boundary_index, raw_transition in enumerate(plan_transitions):
        transition = _mapping(raw_transition) or {}
        if (
            transition.get("kind") != "hard_cut"
            or boundary_index + 1 >= len(reason_sets)
        ):
            continue
        common_reasons = reason_sets[boundary_index] & reason_sets[boundary_index + 1]
        matched_reasons = {
            reason
            for reason in common_reasons
            if reason.endswith("_matched_hard_cut")
        }
        if matched_reasons:
            derived_boundary_match_count += 1
        if any(reason.startswith("orb_affine_") for reason in matched_reasons):
            derived_affine_boundary_match_count += 1

    def contiguous_reason_event_count(reasons: set[str]) -> int:
        count = 0
        active = False
        for segment_reasons in reason_sets:
            present = bool(segment_reasons & reasons)
            if present and not active:
                count += 1
            active = present
        return count

    derived_interior_zoom_count = contiguous_reason_event_count(
        {"reference_calibrated_same_source_interior_zoom_pulse"}
    )
    derived_decorative_zoom_count = sum(
        contiguous_reason_event_count({reason})
        for reason in (
            "reference_calibrated_push_in",
            "reference_calibrated_pull_out",
        )
    )
    derived_reason_zoom_event_count = (
        derived_boundary_match_count
        + derived_interior_zoom_count
        + derived_decorative_zoom_count
    )
    derived_transform_event_count = len(deduplicated_event_times)
    attainment = (
        derived_reason_zoom_event_count / target_zoom if target_zoom > 0 else None
    )
    if target_zoom > 0 and derived_reason_zoom_event_count > target_zoom:
        collector.add(
            "editorial_zoom_target_exceeded",
            QAIssueSeverity.FAIL,
            "Derived non-neutral transform events exceed the section-directed cap.",
            path="editorial_motion.segments",
            value=derived_reason_zoom_event_count,
            threshold=target_zoom,
        )
    elif target_zoom > 0 and derived_reason_zoom_event_count < target_zoom:
        collector.add(
            "editorial_zoom_target_underfilled",
            QAIssueSeverity.CONDITIONAL,
            "Safe serialized tracks did not fill the section-directed transform target.",
            path="editorial_motion.segments",
            value=derived_reason_zoom_event_count,
            threshold=target_zoom,
        )

    serialized_pan_count = _integer(motion.get("pan_event_count"))
    serialized_roll_count = _integer(motion.get("roll_event_count"))
    serialized_boundary_count = _integer(motion.get("boundary_match_count"))
    serialized_affine_count = _integer(motion.get("affine_boundary_match_count"))
    serialized_interior_count = _integer(motion.get("interior_zoom_pulse_count"))
    count_reconciliation_checks = {
        "zoom_track_evidence": (
            actual_zoom is not None
            and serialized_roll_count is not None
            and (
                (actual_zoom + serialized_roll_count > 0)
                == (derived_zoom_event_count > 0)
            )
        ),
        "pan_track_evidence": (
            serialized_pan_count is not None
            and serialized_pan_count == derived_pan_event_count
        ),
        "roll_track_evidence": (
            serialized_roll_count is not None
            and serialized_roll_count == derived_roll_event_count
        ),
        "aggregate_structural_bounds": (
            actual_zoom is not None
            and serialized_pan_count is not None
            and serialized_roll_count is not None
            and serialized_boundary_count is not None
            and serialized_affine_count is not None
            and serialized_interior_count is not None
            and serialized_affine_count <= serialized_boundary_count <= actual_zoom
            and serialized_boundary_count + serialized_interior_count <= actual_zoom
            and serialized_pan_count <= serialized_boundary_count
        ),
        "boundary_event_count_matches_reasons": (
            serialized_boundary_count == derived_boundary_match_count
        ),
        "affine_event_count_matches_reasons": (
            serialized_affine_count == derived_affine_boundary_match_count
        ),
        "interior_zoom_count_matches_reasons": (
            serialized_interior_count == derived_interior_zoom_count
        ),
        "zoom_event_count_matches_reason_events": (
            actual_zoom == derived_reason_zoom_event_count
        ),
    }
    for name, passed in count_reconciliation_checks.items():
        if not passed:
            collector.add(
                f"editorial_motion_{name}",
                QAIssueSeverity.FAIL,
                "Serialized transform event counts are inconsistent with applied tracks.",
                path=f"editorial_motion.{name}",
                value=passed,
                threshold=True,
            )
    event_spacings = [
        right - left
        for left, right in zip(
            deduplicated_event_times, deduplicated_event_times[1:]
        )
    ]
    minimum_spacing = min(event_spacings, default=None)
    if (
        minimum_spacing is not None
        and minimum_spacing
        < config.minimum_editorial_event_spacing_seconds - 1e-9
    ):
        collector.add(
            "editorial_event_spacing_below_minimum",
            QAIssueSeverity.FAIL,
            "Serialized editorial transform events violate the minimum spacing.",
            path="editorial_motion.segments[].keyframes",
            value=minimum_spacing,
            threshold=config.minimum_editorial_event_spacing_seconds,
        )
    return {
        "available": True,
        "applied_segment_count": len(applied),
        "segment_count": len(motion_segments),
        "plan_segment_count": len(plan_segments),
        "motion_contract_checks": motion_contract_checks,
        "dissolve_endpoint_segment_indices": sorted(dissolve_endpoint_indices),
        "applied_segment_coverage": _fraction(len(applied), len(plan_segments)),
        "zoom_event_count": actual_zoom,
        "pan_event_count": serialized_pan_count,
        "roll_event_count": serialized_roll_count,
        "boundary_match_count": serialized_boundary_count,
        "affine_boundary_match_count": serialized_affine_count,
        "interior_zoom_pulse_count": serialized_interior_count,
        "count_reconciliation_checks": count_reconciliation_checks,
        "serialized_event_subtype_count_status": (
            "exact_from_segment_reason_identity_and_track_excursions"
        ),
        "target_zoom_event_count": target_zoom,
        "target_attainment": attainment,
        "target_attainment_basis": "derived_reason_identified_zoom_events",
        "unique_transform_signature_count": len(set(signatures)),
        "transform_signature_count": len(signatures),
        "signature_reuse_ratio": (
            1.0 - len(set(signatures)) / len(signatures) if signatures else None
        ),
        "consecutive_duplicate_signature_count": consecutive_duplicate_count,
        "consecutive_duplicate_signature_fraction": _fraction(
            consecutive_duplicate_count, max(0, len(signatures) - 1)
        ),
        "signature_basis": (
            "reason_category_plus_zoom_pan_roll_amplitude_direction_and_duration_bins"
        ),
        "derived_event_times_seconds": deduplicated_event_times,
        "derived_transform_event_count": derived_transform_event_count,
        "derived_zoom_event_count": derived_zoom_event_count,
        "derived_pan_event_count": derived_pan_event_count,
        "derived_roll_event_count": derived_roll_event_count,
        "derived_reason_zoom_event_count": derived_reason_zoom_event_count,
        "derived_boundary_match_count": derived_boundary_match_count,
        "derived_affine_boundary_match_count": (
            derived_affine_boundary_match_count
        ),
        "derived_interior_zoom_pulse_count": derived_interior_zoom_count,
        "derived_decorative_zoom_event_count": derived_decorative_zoom_count,
        "derived_event_spacing_seconds": _distribution(event_spacings),
        "minimum_derived_event_spacing_seconds": minimum_spacing,
        "reason_category_entropy_normalized": _normalized_entropy(reason_categories),
        "diversity_threshold_status": "reference_calibration_required",
        "unsafe_keyframes": unsafe_keyframes,
        "invalid_tracks": invalid_tracks,
        "skipped_boundary_reasons": motion.get("skipped_boundary_reasons"),
    }


def _selected_boundary_repair_contracts(
    analysis: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> tuple[
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
]:
    """Resolve the same final standard/partial repair records used on screen."""

    standard = _mapping(analysis.get("visual_boundary_repairs")) or {}
    selected_version = selection.get("final_selected_version")
    # The complete-candidate payload is serialized before an accepted windowed
    # re-decode.  Its repair offsets, pose evidence, and beat/phrase clock are
    # therefore superseded.  The analysis payload is the final-bundle
    # authority after that promotion.
    if selected_version != "windowed_semimarkov_redecode":
        selected_index = selection.get("selected_candidate_index")
        for raw in _sequence(selection.get("complete_candidates")) or []:
            candidate = _mapping(raw) or {}
            candidate_repairs = _mapping(candidate.get("boundary_repairs"))
            if (
                candidate.get("candidate_index") == selected_index
                and candidate_repairs is not None
            ):
                standard = candidate_repairs
                break

    plans: list[Mapping[str, Any]] = [standard]
    by_index: dict[int, Mapping[str, Any]] = {}
    for raw in _sequence(standard.get("repairs")) or []:
        repair = _mapping(raw)
        index = _integer((repair or {}).get("boundary_index"))
        if repair is not None and index is not None:
            by_index[index] = repair

    partial = _mapping(selection.get("proxy_triggered_partial_reedit")) or {}
    if selected_version == "partial_reedit":
        partial_plan = _mapping(partial.get("repair")) or {}
        plans.append(partial_plan)
        changed = {
            value
            for value in (
                _sequence(partial.get("changed_boundary_indices")) or []
            )
            if isinstance(value, int) and not isinstance(value, bool)
        }
        for raw in _sequence(partial_plan.get("repairs")) or []:
            repair = _mapping(raw)
            index = _integer((repair or {}).get("boundary_index"))
            if repair is not None and index is not None and index in changed:
                by_index[index] = repair
    return (tuple(plans), tuple(by_index[index] for index in sorted(by_index)))


_TEMPORAL_POSE_MOTION_MARKER = "temporal_pose_motion_audit_v1"
_TEMPORAL_POSE_MOTION_THRESHOLD_KEYS = frozenset(
    {
        "minimum_boundary_motion_magnitude",
        "minimum_endpoint_pose_cosine",
        "minimum_velocity_direction_cosine",
        "minimum_phase_compatibility",
        "pose_norm_epsilon",
    }
)
_TEMPORAL_POSE_MOTION_EVIDENCE_KEYS = frozenset(
    {
        "contract",
        "calibration_status",
        "analysis_executed",
        "input_valid",
        "qualifies",
        "classification",
        "failure_reasons",
        "feature_dimension",
        "outgoing_sample_count",
        "incoming_sample_count",
        "outgoing_boundary_motion_magnitude",
        "incoming_boundary_motion_magnitude",
        "endpoint_pose_cosine",
        "velocity_direction_cosine",
        "phase_compatibility",
        "confidence",
        "static_pose_match",
        "dynamic_pose_phase_match",
        "direction_evidence_available",
        "dynamic_direction_match_proven",
        "soft_preference_applied",
        "threshold_values",
        "threshold_identity_sha256",
        "preference_weight",
        "configuration_fingerprint_sha256",
    }
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _validate_temporal_pose_motion_configuration(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    raw = _mapping(plan.get("temporal_pose_motion_configuration"))
    if raw is None:
        return (None, ("temporal_pose_motion_configuration_missing",))
    expected_keys = {
        "contract",
        "calibration_status",
        "threshold_values",
        "threshold_identity_sha256",
        "preference_weight",
        "configuration_fingerprint_sha256",
    }
    reasons: list[str] = []
    if set(raw) != expected_keys:
        reasons.append("configuration_keys_invalid")
    if raw.get("contract") != "visual_boundary_temporal_pose_motion_preference_v1":
        reasons.append("configuration_contract_invalid")
    status = raw.get("calibration_status")
    if status not in {"calibration_unbound", "calibration_bound"}:
        reasons.append("calibration_status_invalid")
    preference_weight = _number(raw.get("preference_weight"))
    if preference_weight is None or preference_weight < 0.0:
        reasons.append("preference_weight_invalid")
    threshold_values = _mapping(raw.get("threshold_values"))
    threshold_identity = raw.get("threshold_identity_sha256")
    if status == "calibration_unbound":
        if raw.get("threshold_values") is not None:
            reasons.append("unbound_threshold_values_present")
        if threshold_identity is not None:
            reasons.append("unbound_threshold_identity_present")
    elif status == "calibration_bound":
        if (
            threshold_values is None
            or set(threshold_values) != _TEMPORAL_POSE_MOTION_THRESHOLD_KEYS
        ):
            reasons.append("bound_threshold_values_invalid")
        else:
            values = {
                key: _number(threshold_values.get(key))
                for key in _TEMPORAL_POSE_MOTION_THRESHOLD_KEYS
            }
            if any(value is None for value in values.values()):
                reasons.append("bound_threshold_value_nonfinite")
            else:
                motion = values["minimum_boundary_motion_magnitude"]
                endpoint = values["minimum_endpoint_pose_cosine"]
                direction = values["minimum_velocity_direction_cosine"]
                phase = values["minimum_phase_compatibility"]
                epsilon = values["pose_norm_epsilon"]
                if not (
                    motion is not None
                    and motion > 0.0
                    and endpoint is not None
                    and -1.0 <= endpoint <= 1.0
                    and direction is not None
                    and -1.0 <= direction <= 1.0
                    and phase is not None
                    and 0.0 <= phase <= 1.0
                    and epsilon is not None
                    and epsilon > 0.0
                ):
                    reasons.append("bound_threshold_value_range_invalid")
        if (
            not isinstance(threshold_identity, str)
            or _SHA256_PATTERN.fullmatch(threshold_identity) is None
        ):
            reasons.append("bound_threshold_identity_invalid")
        elif threshold_values is not None:
            expected_identity = _canonical_sha256(
                {
                    "contract": "temporal_pose_motion_thresholds_v1",
                    "values": threshold_values,
                }
            )
            if threshold_identity != expected_identity:
                reasons.append("bound_threshold_identity_mismatch")

    fingerprint = raw.get("configuration_fingerprint_sha256")
    if (
        not isinstance(fingerprint, str)
        or _SHA256_PATTERN.fullmatch(fingerprint) is None
    ):
        reasons.append("configuration_fingerprint_invalid")
    elif preference_weight is not None:
        expected_fingerprint = _canonical_sha256(
            {
                "contract": "visual_boundary_temporal_pose_motion_preference_v1",
                "threshold_identity_sha256": threshold_identity,
                "preference_weight": preference_weight,
            }
        )
        if fingerprint != expected_fingerprint:
            reasons.append("configuration_fingerprint_mismatch")
    return (raw, tuple(sorted(set(reasons))))


def _validate_temporal_pose_motion_evidence(
    evidence: Mapping[str, Any],
    *,
    configurations_by_fingerprint: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    raw = _mapping(evidence.get("temporal_pose_motion"))
    if raw is None:
        return (None, ("temporal_pose_motion_evidence_missing",))
    reasons: list[str] = []
    if set(raw) != _TEMPORAL_POSE_MOTION_EVIDENCE_KEYS:
        reasons.append("evidence_keys_invalid_or_nonscalar_payload_present")
    if raw.get("contract") != "scalar_temporal_pose_motion_boundary_evidence_v1":
        reasons.append("evidence_contract_invalid")

    bool_fields = (
        "analysis_executed",
        "input_valid",
        "qualifies",
        "static_pose_match",
        "dynamic_pose_phase_match",
        "direction_evidence_available",
        "dynamic_direction_match_proven",
    )
    if any(type(raw.get(name)) is not bool for name in bool_fields):
        reasons.append("boolean_scalar_fields_invalid")
    failure_reasons = _sequence(raw.get("failure_reasons"))
    if failure_reasons is None or any(
        not isinstance(value, str) or not value for value in failure_reasons
    ):
        reasons.append("failure_reasons_invalid")
    for name in (
        "feature_dimension",
        "outgoing_sample_count",
        "incoming_sample_count",
    ):
        value = raw.get(name)
        if value is not None and (_integer(value) is None or int(value) < 0):
            reasons.append(f"{name}_invalid")
    confidence = _number(raw.get("confidence"))
    if confidence is None or not 0.0 <= confidence <= 1.0:
        reasons.append("confidence_invalid")
    for name in (
        "outgoing_boundary_motion_magnitude",
        "incoming_boundary_motion_magnitude",
        "endpoint_pose_cosine",
        "velocity_direction_cosine",
        "phase_compatibility",
    ):
        value = raw.get(name)
        number = _number(value)
        if value is not None and number is None:
            reasons.append(f"{name}_nonscalar_or_nonfinite")
        elif number is not None:
            if "motion_magnitude" in name and number < 0.0:
                reasons.append(f"{name}_negative")
            if "cosine" in name and not -1.0 <= number <= 1.0:
                reasons.append(f"{name}_unnormalized")
            if name == "phase_compatibility" and not 0.0 <= number <= 1.0:
                reasons.append("phase_compatibility_unnormalized")

    fingerprint = raw.get("configuration_fingerprint_sha256")
    configuration = (
        configurations_by_fingerprint.get(fingerprint)
        if isinstance(fingerprint, str)
        else None
    )
    if configuration is None:
        reasons.append("evidence_configuration_not_bound_to_plan")
    else:
        for evidence_key, configuration_key in (
            ("calibration_status", "calibration_status"),
            ("threshold_values", "threshold_values"),
            ("threshold_identity_sha256", "threshold_identity_sha256"),
            ("preference_weight", "preference_weight"),
        ):
            if raw.get(evidence_key) != configuration.get(configuration_key):
                reasons.append(f"{evidence_key}_differs_from_plan")

    status = raw.get("calibration_status")
    classification = raw.get("classification")
    soft_preference = _number(raw.get("soft_preference_applied"))
    preference_weight = _number(raw.get("preference_weight"))
    if soft_preference is None or soft_preference < 0.0:
        reasons.append("soft_preference_invalid")
    if status == "calibration_unbound":
        if not (
            raw.get("analysis_executed") is False
            and raw.get("input_valid") is False
            and raw.get("qualifies") is False
            and classification == "calibration_unbound"
            and raw.get("direction_evidence_available") is False
            and raw.get("dynamic_direction_match_proven") is False
            and raw.get("dynamic_pose_phase_match") is False
            and raw.get("static_pose_match") is False
            and raw.get("feature_dimension") is None
            and raw.get("outgoing_sample_count") == 0
            and raw.get("incoming_sample_count") == 0
            and confidence == 0.0
            and soft_preference == 0.0
            and failure_reasons
            == ["temporal_pose_motion_thresholds_unbound"]
        ):
            reasons.append("unbound_evidence_masquerades_as_direction_proof")
        for name in (
            "outgoing_boundary_motion_magnitude",
            "incoming_boundary_motion_magnitude",
            "endpoint_pose_cosine",
            "velocity_direction_cosine",
            "phase_compatibility",
        ):
            if raw.get(name) is not None:
                reasons.append("unbound_evidence_contains_motion_scalar")
                break
    elif status == "calibration_bound":
        if raw.get("analysis_executed") is not True:
            reasons.append("bound_analysis_not_executed")
        if classification not in {
            "invalid",
            "static_pose_match",
            "dynamic_pose_phase_match",
            "dynamic_pose_phase_mismatch",
        }:
            reasons.append("bound_classification_invalid")
        qualifies = raw.get("qualifies") is True
        if not (
            (raw.get("dynamic_pose_phase_match") is True) == qualifies
            and (raw.get("dynamic_direction_match_proven") is True) == qualifies
            and (classification == "dynamic_pose_phase_match") == qualifies
        ):
            reasons.append("dynamic_match_flags_inconsistent")
        if (classification == "static_pose_match") != (
            raw.get("static_pose_match") is True
        ):
            reasons.append("static_match_flag_inconsistent")
        if raw.get("input_valid") is True:
            if not (
                raw.get("feature_dimension") == 60
                and raw.get("outgoing_sample_count") == 2
                and raw.get("incoming_sample_count") == 2
                and _number(
                    raw.get("outgoing_boundary_motion_magnitude")
                )
                is not None
                and _number(
                    raw.get("incoming_boundary_motion_magnitude")
                )
                is not None
                and _number(raw.get("endpoint_pose_cosine")) is not None
            ):
                reasons.append("valid_analysis_missing_four_pose_samples")
        elif not (
            classification == "invalid"
            and raw.get("qualifies") is False
            and raw.get("dynamic_pose_phase_match") is False
            and raw.get("dynamic_direction_match_proven") is False
        ):
            reasons.append("invalid_analysis_masquerades_as_match")
        if qualifies != (failure_reasons == []):
            reasons.append("qualification_failure_reasons_inconsistent")
        direction_available = (
            raw.get("input_valid") is True
            and _number(raw.get("velocity_direction_cosine")) is not None
        )
        if (raw.get("direction_evidence_available") is True) != direction_available:
            reasons.append("direction_evidence_availability_inconsistent")
        if (
            (not qualifies or preference_weight == 0.0)
            and soft_preference != 0.0
        ):
            reasons.append("nonqualifying_motion_received_preference")
    else:
        reasons.append("evidence_calibration_status_invalid")

    base_total_cost = _number(evidence.get("base_total_cost"))
    total_cost = _number(evidence.get("total_cost"))
    if base_total_cost is None or total_cost is None:
        reasons.append("boundary_cost_scalars_missing")
    elif (
        total_cost < -1e-12
        or total_cost > base_total_cost + 1e-12
        or soft_preference is None
        or not math.isclose(
            base_total_cost - total_cost,
            soft_preference,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        reasons.append("soft_preference_cost_reconciliation_failed")
    return (raw, tuple(sorted(set(reasons))))


def _boundary_pose_evidence_metrics(
    analysis: Mapping[str, Any],
    selection: Mapping[str, Any],
    collector: _IssueCollector,
) -> dict[str, Any]:
    """Audit exact versus temporal real-pose coverage without a pose quota."""

    plans, repairs = _selected_boundary_repair_contracts(analysis, selection)
    contract_plans = tuple(
        plan
        for plan in plans
        if "actual_pose_boundary_repair_v3"
        in str(plan.get("analysis_basis") or "")
    )
    if not contract_plans:
        return {
            "available": False,
            "contract": "formation_layout_and_actual_pose_boundary_repair_v3",
            "legacy_or_unavailable": True,
        }
    temporal_contract_plans = tuple(
        plan
        for plan in contract_plans
        if _TEMPORAL_POSE_MOTION_MARKER
        in str(plan.get("analysis_basis") or "")
    )
    temporal_plan_contract_failures: list[dict[str, Any]] = []
    temporal_configurations_by_fingerprint: dict[
        str, Mapping[str, Any]
    ] = {}
    for position, plan in enumerate(temporal_contract_plans):
        configuration, reasons = _validate_temporal_pose_motion_configuration(
            plan
        )
        if reasons:
            temporal_plan_contract_failures.append(
                {"plan_index": position, "reasons": list(reasons)}
            )
            continue
        assert configuration is not None
        fingerprint = configuration.get(
            "configuration_fingerprint_sha256"
        )
        assert isinstance(fingerprint, str)
        temporal_configurations_by_fingerprint[fingerprint] = configuration

    geometry_attempts = sum(
        _integer(plan.get("geometry_provider_attempt_count")) or 0
        for plan in contract_plans
    )
    geometry_failures_values = [
        _integer(plan.get("geometry_provider_failure_count"))
        for plan in contract_plans
    ]
    missing_plan_contract = any(
        _integer(plan.get("geometry_provider_attempt_count")) is None
        or failure is None
        for plan, failure in zip(
            contract_plans, geometry_failures_values, strict=True
        )
    )
    geometry_failures = sum(value or 0 for value in geometry_failures_values)
    retained_failure_indices = sorted(
        {
            index
            for repair_plan in contract_plans
            for raw in (_sequence(repair_plan.get("failures")) or [])
            if (failure := _mapping(raw)) is not None
            and (index := _integer(failure.get("boundary_index"))) is not None
        }
    )
    if geometry_failures:
        collector.add(
            "boundary_repair_geometry_provider_failures",
            QAIssueSeverity.FAIL,
            "Cut-local geometry decoding failed for at least one requested sample.",
            path="visual_boundary_repairs.geometry_provider_failure_count",
            value=geometry_failures,
            threshold=0,
        )

    accepted_pose_bases = {
        "exact_frame_opencv_zoo_mediapipe_pose",
        "nearest_interior_temporal_opencv_zoo_mediapipe_pose",
        "actual_pose_pair_unavailable",
    }
    professional_bases = {
        "exact_actual_pose_plus_screen_geometry",
        "nearest_interior_temporal_actual_pose_plus_screen_geometry",
        "formation_assignment_plus_screen_geometry",
        "subject_layout_plus_silhouette",
    }
    invalid_records: list[dict[str, Any]] = []
    professional_gate_failures: list[int] = []
    hog_pose_indices: list[int] = []
    basis_counts: Counter[str] = Counter()
    missing_mask_counts: Counter[str] = Counter()
    exact_pose_side_count = 0
    exact_pose_pair_count = 0
    temporal_pose_side_count = 0
    temporal_pose_pair_count = 0
    nearest_pose_pair_count = 0
    formation_side_count = 0
    formation_pair_count = 0
    formation_upper_body_side_count = 0
    formation_upper_body_pair_count = 0
    layout_side_count = 0
    layout_pair_count = 0
    silhouette_side_count = 0
    silhouette_pair_count = 0
    temporal_invalid_records: list[dict[str, Any]] = []
    temporal_classification_counts: Counter[str] = Counter()
    temporal_calibration_status_counts: Counter[str] = Counter()
    temporal_direction_evidence_count = 0
    temporal_dynamic_direction_match_count = 0
    temporal_soft_preference_count = 0
    temporal_soft_preference_total = 0.0

    def bool_pair(
        coverage: Mapping[str, Any], name: str
    ) -> tuple[bool, bool] | None:
        values = _sequence(coverage.get(name))
        if (
            values is None
            or len(values) != 2
            or any(type(value) is not bool for value in values)
        ):
            return None
        return (bool(values[0]), bool(values[1]))

    def count_pair(
        coverage: Mapping[str, Any], name: str
    ) -> tuple[int, int] | None:
        values = _sequence(coverage.get(name))
        if values is None or len(values) != 2:
            return None
        converted = tuple(_integer(value) for value in values)
        if any(value is None or value < 0 for value in converted):
            return None
        return (int(converted[0]), int(converted[1]))  # type: ignore[arg-type]

    for position, repair in enumerate(repairs):
        index = _integer(repair.get("boundary_index"))
        evidence = _mapping(repair.get("selected_visual_evidence")) or {}
        if temporal_contract_plans:
            temporal_evidence, temporal_reasons = (
                _validate_temporal_pose_motion_evidence(
                    evidence,
                    configurations_by_fingerprint=(
                        temporal_configurations_by_fingerprint
                    ),
                )
            )
            if temporal_reasons:
                temporal_invalid_records.append(
                    {
                        "boundary_index": (
                            index if index is not None else position
                        ),
                        "reasons": list(temporal_reasons),
                    }
                )
            if temporal_evidence is not None:
                temporal_classification_counts.update(
                    [str(temporal_evidence.get("classification"))]
                )
                temporal_calibration_status_counts.update(
                    [str(temporal_evidence.get("calibration_status"))]
                )
                temporal_direction_evidence_count += int(
                    temporal_evidence.get("direction_evidence_available")
                    is True
                )
                temporal_dynamic_direction_match_count += int(
                    temporal_evidence.get("dynamic_direction_match_proven")
                    is True
                )
                preference = _number(
                    temporal_evidence.get("soft_preference_applied")
                )
                if preference is not None and preference > 0.0:
                    temporal_soft_preference_count += 1
                    temporal_soft_preference_total += preference
        coverage = _mapping(evidence.get("evidence_coverage"))
        reasons: list[str] = []
        if coverage is None:
            invalid_records.append(
                {
                    "boundary_index": index,
                    "reasons": ["evidence_coverage_missing"],
                }
            )
            continue
        exact = bool_pair(coverage, "exact_pose_available")
        temporal = bool_pair(coverage, "temporal_pose_available")
        formation = count_pair(coverage, "formation_person_counts")
        formation_upper = count_pair(
            coverage, "formation_upper_body_counts"
        )
        layout = bool_pair(coverage, "person_layout_available")
        silhouette = bool_pair(coverage, "silhouette_available")
        offsets_raw = _sequence(
            coverage.get("temporal_pose_offsets_frames")
        )
        offsets = (
            tuple(
                None if value is None else _integer(value)
                for value in offsets_raw
            )
            if offsets_raw is not None and len(offsets_raw) == 2
            else None
        )
        for name, value in (
            ("exact_pose_available", exact),
            ("temporal_pose_available", temporal),
            ("formation_person_counts", formation),
            ("formation_upper_body_counts", formation_upper),
            ("person_layout_available", layout),
            ("silhouette_available", silhouette),
            ("temporal_pose_offsets_frames", offsets),
        ):
            if value is None:
                reasons.append(f"{name}_invalid")

        pose_basis = evidence.get("pose_basis")
        if pose_basis == "structure_hog_fallback":
            if index is not None:
                hog_pose_indices.append(index)
        if pose_basis not in accepted_pose_bases:
            reasons.append("pose_basis_invalid")
        exact_mismatch = _number(evidence.get("exact_pose_mismatch"))
        temporal_mismatch = _number(
            evidence.get("nearest_temporal_pose_mismatch")
        )
        selected_pose_mismatch = _number(evidence.get("pose_mismatch"))
        if pose_basis == "exact_frame_opencv_zoo_mediapipe_pose" and not (
            exact == (True, True)
            and exact_mismatch is not None
            and selected_pose_mismatch is not None
        ):
            reasons.append("exact_pose_basis_not_supported_by_exact_pair")
        if pose_basis == "nearest_interior_temporal_opencv_zoo_mediapipe_pose":
            directional_offsets_valid = bool(
                offsets is not None
                and offsets[0] is not None
                and offsets[1] is not None
                and -3 <= offsets[0] <= 0
                and 0 <= offsets[1] <= 3
                and any(value != 0 for value in offsets)
            )
            if not (
                temporal == (True, True)
                and temporal_mismatch is not None
                and selected_pose_mismatch is not None
                and directional_offsets_valid
            ):
                reasons.append(
                    "temporal_pose_basis_not_supported_by_interior_pair"
                )
        if pose_basis == "actual_pose_pair_unavailable" and (
            selected_pose_mismatch is not None
            or exact_mismatch is not None
            or temporal_mismatch is not None
        ):
            reasons.append("unavailable_pose_basis_contains_pose_mismatch")

        professional_basis = coverage.get("professional_evidence_basis")
        professional_passed = coverage.get("professional_gate_passed")
        if professional_basis not in professional_bases:
            reasons.append("professional_evidence_basis_invalid")
        if professional_passed is not True:
            if index is not None:
                professional_gate_failures.append(index)
            reasons.append("professional_combined_evidence_gate_failed")

        missing_mask_raw = _sequence(coverage.get("missing_mask"))
        if missing_mask_raw is None or any(
            not isinstance(value, str) or not value
            for value in missing_mask_raw
        ):
            reasons.append("missing_mask_invalid")
        else:
            missing_mask_counts.update(str(value) for value in missing_mask_raw)

        if reasons:
            invalid_records.append(
                {
                    "boundary_index": index if index is not None else position,
                    "reasons": sorted(set(reasons)),
                }
            )
        if exact is not None:
            exact_pose_side_count += sum(exact)
            exact_pose_pair_count += int(all(exact))
        if temporal is not None:
            temporal_pose_side_count += sum(temporal)
            temporal_pose_pair_count += int(all(temporal))
            nearest_pose_pair_count += int(
                all(temporal)
                and offsets is not None
                and any(value not in (None, 0) for value in offsets)
            )
        if formation is not None:
            formation_side_count += sum(value > 0 for value in formation)
            formation_pair_count += int(all(value > 0 for value in formation))
        if formation_upper is not None:
            formation_upper_body_side_count += sum(
                value > 0 for value in formation_upper
            )
            formation_upper_body_pair_count += int(
                all(value > 0 for value in formation_upper)
            )
        if layout is not None:
            layout_side_count += sum(layout)
            layout_pair_count += int(all(layout))
        if silhouette is not None:
            silhouette_side_count += sum(silhouette)
            silhouette_pair_count += int(all(silhouette))
        basis_counts[str(professional_basis)] += 1

    if hog_pose_indices:
        collector.add(
            "boundary_hog_mislabeled_as_pose",
            QAIssueSeverity.FAIL,
            "HOG structure evidence was serialized as if it were actual pose.",
            path="visual_boundary_repairs.repairs[].selected_visual_evidence.pose_basis",
            value=sorted(hog_pose_indices),
            threshold=[],
        )
    if professional_gate_failures:
        collector.add(
            "boundary_professional_combined_evidence_gate_failed",
            QAIssueSeverity.FAIL,
            "A selected cut lacks the required combined formation/layout/pose evidence.",
            path="visual_boundary_repairs.repairs[].selected_visual_evidence.evidence_coverage",
            value=sorted(professional_gate_failures),
            threshold=[],
        )
    if missing_plan_contract or invalid_records:
        collector.add(
            "boundary_pose_evidence_contract_invalid",
            QAIssueSeverity.FAIL,
            "Cut-local pose evidence is missing or internally inconsistent.",
            path="visual_boundary_repairs",
            value={
                "plan_contract_missing": missing_plan_contract,
                "invalid_records": invalid_records,
            },
            threshold={
                "plan_contract_missing": False,
                "invalid_records": [],
            },
        )
    if temporal_plan_contract_failures or temporal_invalid_records:
        collector.add(
            "boundary_temporal_pose_motion_contract_invalid",
            QAIssueSeverity.FAIL,
            "Temporal pose-motion evidence is non-scalar, misbound, or internally inconsistent.",
            path=(
                "visual_boundary_repairs.repairs[].selected_visual_evidence."
                "temporal_pose_motion"
            ),
            value={
                "plan_contract_failures": temporal_plan_contract_failures,
                "invalid_records": temporal_invalid_records,
            },
            threshold={
                "plan_contract_failures": [],
                "invalid_records": [],
            },
        )

    boundary_count = len(repairs)
    side_count = 2 * boundary_count
    return {
        "available": True,
        "contract": "formation_layout_and_actual_pose_boundary_repair_v3",
        "boundary_count": boundary_count,
        "retained_without_selected_repair_boundary_indices": (
            retained_failure_indices
        ),
        "side_count": side_count,
        "geometry_provider_attempt_count": geometry_attempts,
        "geometry_provider_failure_count": geometry_failures,
        "exact_actual_pose_side_count": exact_pose_side_count,
        "exact_actual_pose_side_coverage": _fraction(
            exact_pose_side_count, side_count
        ),
        "exact_actual_pose_pair_count": exact_pose_pair_count,
        "exact_actual_pose_pair_coverage": _fraction(
            exact_pose_pair_count, boundary_count
        ),
        "temporal_actual_pose_side_count": temporal_pose_side_count,
        "temporal_actual_pose_pair_count": temporal_pose_pair_count,
        "nearest_temporal_actual_pose_pair_count": nearest_pose_pair_count,
        "formation_side_count": formation_side_count,
        "formation_pair_count": formation_pair_count,
        "formation_upper_body_side_count": formation_upper_body_side_count,
        "formation_upper_body_pair_count": formation_upper_body_pair_count,
        "person_layout_side_count": layout_side_count,
        "person_layout_pair_count": layout_pair_count,
        "silhouette_side_count": silhouette_side_count,
        "silhouette_pair_count": silhouette_pair_count,
        "professional_evidence_basis_counts": dict(sorted(basis_counts.items())),
        "professional_gate_failure_boundary_indices": sorted(
            professional_gate_failures
        ),
        "hog_used_as_pose_boundary_indices": sorted(hog_pose_indices),
        "missing_mask_counts": dict(sorted(missing_mask_counts.items())),
        "invalid_records": invalid_records,
        "temporal_pose_motion": {
            "available": bool(temporal_contract_plans),
            "contract": (
                "scalar_temporal_pose_motion_boundary_evidence_v1"
                if temporal_contract_plans
                else None
            ),
            "plan_contract_failures": temporal_plan_contract_failures,
            "invalid_records": temporal_invalid_records,
            "calibration_status_counts": dict(
                sorted(temporal_calibration_status_counts.items())
            ),
            "classification_counts": dict(
                sorted(temporal_classification_counts.items())
            ),
            "direction_evidence_available_count": (
                temporal_direction_evidence_count
            ),
            "dynamic_direction_match_proven_count": (
                temporal_dynamic_direction_match_count
            ),
            "soft_preference_applied_count": temporal_soft_preference_count,
            "soft_preference_applied_total": temporal_soft_preference_total,
            "passed": (
                not temporal_plan_contract_failures
                and not temporal_invalid_records
            ),
        },
        "passed": (
            geometry_failures == 0
            and not missing_plan_contract
            and not professional_gate_failures
            and not hog_pose_indices
            and not invalid_records
            and not temporal_plan_contract_failures
            and not temporal_invalid_records
        ),
    }


def _beat_phrase_metrics(
    analysis: Mapping[str, Any],
    plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    config: PostRenderQAConfig,
    collector: _IssueCollector,
) -> dict[str, Any]:
    rhythm = _mapping(analysis.get("rhythm_analysis")) or {}
    timeline = _mapping(analysis.get("timeline_review")) or {}
    boundaries = [
        item
        for raw in (_sequence(timeline.get("boundaries")) or [])
        if (item := _mapping(raw)) is not None and item.get("source_switch") is True
    ]
    beats = [
        item
        for raw in (_sequence(rhythm.get("beat_grid")) or [])
        if (item := _mapping(raw)) is not None
        and _number(item.get("time_seconds")) is not None
    ]
    beat_times = [float(item["time_seconds"]) for item in beats]
    phrase_items = [
        item
        for raw in (_sequence(rhythm.get("phrase_candidates")) or [])
        if (item := _mapping(raw)) is not None
    ]
    phrase_times = sorted(
        {
            value
            for item in phrase_items
            for key in ("start_seconds", "end_seconds")
            if (value := _number(item.get(key))) is not None
        }
    )
    fps = _number(plan.get("fps")) or 0.0
    if not beat_times or fps <= 0.0:
        collector.add(
            "beat_grid_unavailable",
            QAIssueSeverity.FAIL,
            "The final edit cannot be checked against a valid beat grid.",
            path="rhythm_analysis.beat_grid",
        )
    if not phrase_times:
        collector.add(
            "phrase_candidates_unavailable",
            QAIssueSeverity.CONDITIONAL,
            "Phrase-boundary alignment cannot be measured.",
            path="rhythm_analysis.phrase_candidates",
        )

    beat_distances: list[float] = []
    phrase_distances: list[float] = []
    nearest_strengths: list[float] = []
    per_boundary: list[dict[str, Any]] = []
    for boundary in boundaries:
        time_value = _number(boundary.get("at_seconds"))
        if time_value is None:
            continue
        if beat_times and fps > 0.0:
            nearest_index = min(
                range(len(beat_times)),
                key=lambda index: abs(beat_times[index] - time_value),
            )
            beat_distance = abs(beat_times[nearest_index] - time_value) * fps
            strength = _number(beats[nearest_index].get("strength"))
            beat_distances.append(beat_distance)
            if strength is not None:
                nearest_strengths.append(strength)
        else:
            beat_distance = None
            strength = None
        phrase_distance = (
            min(abs(value - time_value) for value in phrase_times) * fps
            if phrase_times and fps > 0.0
            else None
        )
        if phrase_distance is not None:
            phrase_distances.append(phrase_distance)
        per_boundary.append(
            {
                "boundary_index": boundary.get("boundary_index"),
                "canonical_master_seconds": time_value,
                "nearest_beat_distance_frames": beat_distance,
                "nearest_beat_strength": strength,
                "nearest_phrase_boundary_distance_frames": phrase_distance,
            }
        )

    _repair_plans, final_repairs = _selected_boundary_repair_contracts(
        analysis,
        selection,
    )
    offsets: dict[int, int] = {}
    selected_boundary_seconds: dict[int, float] = {}
    nominal_boundary_seconds: dict[int, float] = {}
    repair_payload_consistency: dict[int, bool] = {}
    for raw in final_repairs:
        repair = _mapping(raw) or {}
        index = _integer(repair.get("boundary_index"))
        result = _mapping(repair.get("result")) or {}
        offset = _integer(result.get("selected_frame_offset"))
        if index is not None and offset is not None:
            offsets[index] = offset
            selected_seconds = _number(result.get("selected_boundary_seconds"))
            if selected_seconds is not None:
                selected_boundary_seconds[index] = selected_seconds
            nominal_seconds = _number(result.get("nominal_boundary_seconds"))
            if nominal_seconds is not None:
                nominal_boundary_seconds[index] = nominal_seconds
            nested_selected = _mapping(result.get("selected")) or {}
            nested_selected_seconds = _number(
                nested_selected.get("boundary_seconds")
            )
            repair_payload_consistency[index] = (
                nested_selected.get("frame_offset") == offset
                and nested_selected_seconds is not None
                and selected_seconds is not None
                and math.isclose(
                    nested_selected_seconds,
                    selected_seconds,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )

    partial = _mapping(selection.get("proxy_triggered_partial_reedit")) or {}
    selected_partial_reedit = (
        selection.get("final_selected_version") == "partial_reedit"
    )
    partial_changed = (
        {
            value
            for value in (
                _sequence(partial.get("changed_boundary_indices")) or []
            )
            if isinstance(value, int) and not isinstance(value, bool)
        }
        if selected_partial_reedit
        else set()
    )
    repair_records: list[dict[str, Any]] = []
    visible_indices = {
        _integer(item.get("boundary_index")) for item in boundaries
    }
    visible_indices.discard(None)
    missing_repair_records = sorted(index for index in visible_indices if index not in offsets)
    if missing_repair_records:
        collector.add(
            "visible_boundary_repair_evidence_missing",
            QAIssueSeverity.CONDITIONAL,
            "At least one visible source switch lacks serialized local-repair evidence.",
            path="visual_boundary_repairs.repairs",
            value=missing_repair_records,
            threshold=[],
        )
    boundary_times = {
        index: time_value
        for item in boundaries
        if (index := _integer(item.get("boundary_index"))) is not None
        and (time_value := _number(item.get("at_seconds"))) is not None
    }
    for index in sorted(visible_indices):
        if index not in offsets:
            continue
        offset = offsets[index]
        maximum = (
            config.partial_reedit_maximum_frames
            if index in partial_changed
            else config.standard_boundary_repair_maximum_frames
        )
        passed = abs(offset) <= maximum
        selected_seconds = selected_boundary_seconds.get(index)
        nominal_seconds = nominal_boundary_seconds.get(index)
        timeline_seconds = boundary_times.get(index)
        selected_time_matches_timeline = (
            selected_seconds is not None
            and timeline_seconds is not None
            and math.isclose(
                selected_seconds,
                timeline_seconds,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        )
        offset_time_matches = (
            nominal_seconds is not None
            and selected_seconds is not None
            and fps > 0.0
            and math.isclose(
                selected_seconds,
                nominal_seconds + offset / fps,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        )
        payload_consistent = repair_payload_consistency.get(index) is True
        repair_records.append(
            {
                "boundary_index": index,
                "selected_frame_offset": offset,
                "selected_boundary_seconds": selected_seconds,
                "nominal_boundary_seconds": nominal_seconds,
                "timeline_boundary_seconds": timeline_seconds,
                "selected_time_matches_timeline": selected_time_matches_timeline,
                "offset_time_matches": offset_time_matches,
                "payload_consistent": payload_consistent,
                "maximum_allowed_frames": maximum,
                "partial_reedit_boundary": index in partial_changed,
                "passed": passed,
            }
        )
        if not passed:
            collector.add(
                "boundary_repair_exceeds_allowed_radius",
                QAIssueSeverity.FAIL,
                "A visible boundary moved beyond its permitted local repair radius.",
                path=f"boundary_repairs[{index}]",
                value=offset,
                threshold=maximum,
            )
        if not selected_time_matches_timeline:
            collector.add(
                "boundary_repair_time_mismatch",
                QAIssueSeverity.FAIL,
                "The selected repair timestamp does not match the final visible boundary.",
                path=f"boundary_repairs[{index}].selected_boundary_seconds",
                value={
                    "selected_boundary_seconds": selected_seconds,
                    "timeline_boundary_seconds": timeline_seconds,
                },
                threshold="absolute difference <= 1e-6 seconds",
            )
        if not offset_time_matches or not payload_consistent:
            collector.add(
                "boundary_repair_payload_inconsistent",
                QAIssueSeverity.FAIL,
                "Boundary repair offset and nested timestamps are internally inconsistent.",
                path=f"boundary_repairs[{index}]",
                value={
                    "nominal_boundary_seconds": nominal_seconds,
                    "selected_boundary_seconds": selected_seconds,
                    "selected_frame_offset": offset,
                    "nested_payload_consistent": payload_consistent,
                },
                threshold="selected = nominal + frame_offset / fps",
            )

    beat_distribution = _distribution(beat_distances)
    phrase_distribution = _distribution(phrase_distances)
    if beat_distances and max(beat_distances) > config.partial_beat_window_frames:
        collector.add(
            "visible_cut_far_from_beat_grid",
            QAIssueSeverity.CONDITIONAL,
            "At least one visible source switch is more than eight frames from the nearest beat.",
            path="beat_phrase.per_boundary",
            value=max(beat_distances),
            threshold=config.partial_beat_window_frames,
        )

    section_metrics: list[dict[str, Any]] = []
    beat_period = _number(rhythm.get("beat_period_seconds")) or 0.0
    for raw in _sequence((_mapping(analysis.get("section_plan")) or {}).get("sections")) or []:
        section = _mapping(raw) or {}
        start = _number(section.get("start_seconds"))
        end = _number(section.get("end_seconds"))
        target_spm = _number(section.get("target_source_switches_per_minute"))
        if start is None or end is None or end <= start or target_spm is None:
            continue
        in_section = [
            item
            for item in per_boundary
            if start <= float(item["canonical_master_seconds"]) < end
        ]
        duration = end - start
        achieved_spm = len(in_section) * 60.0 / duration
        expected_count = int(round(target_spm * duration / 60.0))
        last_beat_cut_count = (
            sum(
                end - beat_period
                <= float(item["canonical_master_seconds"])
                < end - 1e-9
                for item in in_section
            )
            if beat_period > 0.0
            and section.get("protect_last_beat_from_extra_cut") is True
            else 0
        )
        if last_beat_cut_count:
            collector.add(
                "protected_section_last_beat_cut",
                QAIssueSeverity.CONDITIONAL,
                "A protected final section beat contains an extra source switch.",
                path=f"section_plan.sections[{section.get('section_index')}]",
                value=last_beat_cut_count,
                threshold=0,
            )
        section_metrics.append(
            {
                "section_index": section.get("section_index"),
                "role": section.get("role"),
                "duration_seconds": duration,
                "source_switch_count": len(in_section),
                "achieved_source_switches_per_minute": achieved_spm,
                "target_source_switches_per_minute": target_spm,
                "expected_source_switch_count": expected_count,
                "count_delta_from_target": len(in_section) - expected_count,
                "protected_last_beat_cut_count": last_beat_cut_count,
            }
        )
    return {
        "visible_source_switch_count": len(per_boundary),
        "beat_distance_frames": {
            **beat_distribution,
            "within_1_frame_count": sum(
                value <= config.exact_beat_window_frames for value in beat_distances
            ),
            "within_1_frame_fraction": _fraction(
                sum(value <= config.exact_beat_window_frames for value in beat_distances),
                len(beat_distances),
            ),
            "within_3_frames_count": sum(
                value <= config.standard_beat_window_frames for value in beat_distances
            ),
            "within_3_frames_fraction": _fraction(
                sum(value <= config.standard_beat_window_frames for value in beat_distances),
                len(beat_distances),
            ),
            "within_8_frames_count": sum(
                value <= config.partial_beat_window_frames for value in beat_distances
            ),
            "within_8_frames_fraction": _fraction(
                sum(value <= config.partial_beat_window_frames for value in beat_distances),
                len(beat_distances),
            ),
        },
        "phrase_distance_frames": phrase_distribution,
        "phrase_hit_within_3_frames_fraction": _fraction(
            sum(value <= config.standard_beat_window_frames for value in phrase_distances),
            len(phrase_distances),
        ),
        "nearest_beat_strength": _distribution(nearest_strengths),
        "per_boundary": per_boundary,
        "repair_radius_checks": repair_records,
        "missing_repair_boundary_indices": missing_repair_records,
        "sections": section_metrics,
        "phrase_professional_threshold_status": "reference_calibration_required",
    }


def _sync_quality_metrics(
    analysis: Mapping[str, Any],
    plan: Mapping[str, Any],
    config: PostRenderQAConfig,
    collector: _IssueCollector,
    *,
    selection: Mapping[str, Any] | None = None,
    trusted_registered_stage_edit: bool | None = None,
) -> dict[str, Any]:
    """Fail closed on the production PTS clock used by every rendered source.

    ``local_timewarps`` is serialized in request-source index space.  Current
    pipeline plans retain the full request sources array, while older imported
    plans can use a compact eligible-source array.  The explicit domain
    equivalence/mapping is therefore part of the evidence; guessing the domain
    could silently audit a different camera after an ineligible source is
    removed.  The residual is converted to output frames so the professional
    half-frame release target is independent of 29.97/30/59.94 fps delivery.
    """

    activation = resolve_cut_local_sync_activation(
        analysis,
        selection,
        trusted_registered_stage_edit=trusted_registered_stage_edit,
    )
    cut_local_required = bool(activation["required"])
    activation_errors = list(activation["errors"])
    if activation_errors:
        collector.add(
            "production_cut_local_sync_activation_contract_invalid",
            QAIssueSeverity.FAIL,
            "StageEdit activation markers are missing, malformed, or disagree.",
            path="stage_edit_generation,selection.stage_edit_render_promotion,render",
            value=activation_errors,
            threshold=[],
        )
    fps = _number(plan.get("fps"))
    usage = _mapping(plan.get("source_usage")) or {}
    raw_used_indices = _sequence(usage.get("used_source_indices"))
    used_render_indices = (
        tuple(_integer(value) for value in raw_used_indices)
        if raw_used_indices is not None
        else ()
    )
    mapping = _mapping(analysis.get("source_index_mapping")) or {}
    eligible_to_request = _sequence(mapping.get("eligible_to_request"))
    domain_equivalences = _mapping(mapping.get("domain_equivalences")) or {}
    render_indices_are_request_indices = (
        domain_equivalences.get("render_plan_sources_array_index")
        == "request_source_index"
    )
    plan_sources = _sequence(plan.get("sources"))
    local_timewarps = _sequence(analysis.get("local_timewarps"))

    domain_valid = (
        fps is not None
        and fps > 0.0
        and bool(used_render_indices)
        and all(index is not None and index >= 0 for index in used_render_indices)
        and (
            render_indices_are_request_indices
            and plan_sources is not None
            and all(
                index is not None and index < len(plan_sources)
                for index in used_render_indices
            )
            or not render_indices_are_request_indices
            and eligible_to_request is not None
            and all(
                index is not None and index < len(eligible_to_request)
                for index in used_render_indices
            )
        )
        and local_timewarps is not None
    )
    if not domain_valid:
        collector.add(
            "production_sync_evidence_missing_or_unmapped",
            QAIssueSeverity.FAIL,
            "Used render sources cannot be bound to their PTS time-warp evidence.",
            path="local_timewarps",
        )
        if cut_local_required:
            collector.add(
                "production_cut_local_sync_required_gate_failed",
                QAIssueSeverity.FAIL,
                "Registered automatic StageEdit did not satisfy the retained-cut local synchronization gate.",
                path="render.cut_local_sync",
                value=False,
                threshold=True,
            )
        return {
            "basis": "used_source_piecewise_pts_residual_p95_frames_v1",
            "domain_mapping_valid": False,
            "used_render_source_indices": [
                value for value in used_render_indices if value is not None
            ],
            "per_source": [],
            "maximum_residual_p95_frames": None,
            "cut_local_sync": {
                "required": cut_local_required,
                "activation_contract": activation,
                "available": False,
                "passed": False if cut_local_required else None,
            },
            "passed": False,
        }

    assert fps is not None
    assert local_timewarps is not None
    timewarp_by_request: dict[int, dict[str, Any]] = {}
    duplicate_request_indices: set[int] = set()
    for raw_item in local_timewarps:
        item = _mapping(raw_item)
        if item is None:
            continue
        request_index = _integer(item.get("source_index"))
        if request_index is None or request_index < 0:
            continue
        if request_index in timewarp_by_request:
            duplicate_request_indices.add(request_index)
        else:
            timewarp_by_request[request_index] = item

    if duplicate_request_indices:
        collector.add(
            "production_sync_duplicate_source_evidence",
            QAIssueSeverity.FAIL,
            "A request source has more than one serialized time-warp record.",
            path="local_timewarps",
            value=sorted(duplicate_request_indices),
        )

    per_source: list[dict[str, Any]] = []
    residual_frames_values: list[float] = []
    for render_index_value in used_render_indices:
        assert render_index_value is not None
        request_index = (
            render_index_value
            if render_indices_are_request_indices
            else _integer(eligible_to_request[render_index_value])
            if eligible_to_request is not None
            else None
        )
        item = (
            timewarp_by_request.get(request_index)
            if request_index is not None
            else None
        )
        path = f"local_timewarps[request_source_index={request_index}]"
        if request_index is None or item is None:
            collector.add(
                "production_sync_used_source_evidence_missing",
                QAIssueSeverity.FAIL,
                "A rendered source has no bound local time-warp record.",
                path=path,
                value={
                    "render_plan_source_index": render_index_value,
                    "request_source_index": request_index,
                },
            )
            per_source.append(
                {
                    "render_plan_source_index": render_index_value,
                    "request_source_index": request_index,
                    "passed": False,
                    "reason": "timewarp_missing",
                }
            )
            continue

        residuals = _mapping(item.get("residuals")) or {}
        residual_seconds = _number(residuals.get("p95_seconds"))
        sample_count = _integer(residuals.get("sample_count"))
        anchor_count = _integer(item.get("accepted_anchor_count"))
        residual_definition = residuals.get("definition")
        scalar_fallback = item.get("scalar_fallback")
        status = item.get("status")
        residual_frames = (
            residual_seconds * fps
            if residual_seconds is not None and residual_seconds >= 0.0
            else None
        )
        piecewise_valid = (
            scalar_fallback is False
            and status == "local_monotonic_timewarp"
            and residual_definition
            == "absolute_anchor_residual_to_robust_affine_trend"
            and anchor_count is not None
            and anchor_count >= config.minimum_local_timewarp_anchor_count
            and sample_count is not None
            and sample_count >= config.minimum_local_timewarp_anchor_count
            and residual_frames is not None
        )
        within_threshold = (
            residual_frames is not None
            and residual_frames <= config.maximum_sync_residual_p95_frames + 1e-12
        )
        passed = piecewise_valid and within_threshold
        if not piecewise_valid:
            collector.add(
                "production_sync_used_source_not_piecewise_verified",
                QAIssueSeverity.FAIL,
                "A rendered source lacks a verified monotonic local PTS clock.",
                path=path,
                value={
                    "status": status,
                    "scalar_fallback": scalar_fallback,
                    "accepted_anchor_count": anchor_count,
                    "residual_sample_count": sample_count,
                    "residual_definition": residual_definition,
                },
                threshold={
                    "minimum_anchor_count": config.minimum_local_timewarp_anchor_count,
                    "scalar_fallback": False,
                },
            )
        elif not within_threshold:
            collector.add(
                "production_sync_residual_exceeds_half_frame",
                QAIssueSeverity.FAIL,
                "A rendered source exceeds the professional PTS residual target.",
                path=path,
                value=residual_frames,
                threshold=config.maximum_sync_residual_p95_frames,
            )
        if residual_frames is not None:
            residual_frames_values.append(residual_frames)
        per_source.append(
            {
                "render_plan_source_index": render_index_value,
                "request_source_index": request_index,
                "status": status,
                "scalar_fallback": scalar_fallback,
                "accepted_anchor_count": anchor_count,
                "residual_sample_count": sample_count,
                "residual_definition": residual_definition,
                "residual_p95_seconds": residual_seconds,
                "residual_p95_frames": residual_frames,
                "passed": passed,
            }
        )

    source_clock_passed = (
        len(per_source) == len(used_render_indices)
        and all(item.get("passed") is True for item in per_source)
        and not duplicate_request_indices
    )
    cut_local_metrics: dict[str, Any]
    if cut_local_required:
        render_payload = _mapping(analysis.get("render")) or {}
        evidence = _mapping(render_payload.get("cut_local_sync"))
        cut_local_metrics = validate_cut_local_sync_evidence(
            evidence,
            plan=plan,
            timewarp_payloads=local_timewarps,
            maximum_residual_p95_frames=(
                config.maximum_sync_residual_p95_frames
            ),
        )
        cut_local_metrics["required"] = True
        errors = _sequence(cut_local_metrics.get("errors")) or []
        unknown_count = _integer(
            cut_local_metrics.get("unknown_boundary_count")
        ) or 0
        residual_p95 = _number(cut_local_metrics.get("residual_p95_frames"))
        if not bool(cut_local_metrics.get("available")):
            collector.add(
                "production_cut_local_sync_evidence_missing",
                QAIssueSeverity.FAIL,
                "Registered automatic StageEdit lacks retained-cut local sync evidence.",
                path="render.cut_local_sync",
            )
        if errors:
            collector.add(
                "production_cut_local_sync_binding_invalid",
                QAIssueSeverity.FAIL,
                "Retained-cut sync evidence is not bound to the final plan and clocks.",
                path="render.cut_local_sync",
                value=errors,
                threshold=[],
            )
        if unknown_count > 0:
            collector.add(
                "production_cut_local_sync_unknown_boundaries",
                QAIssueSeverity.FAIL,
                "At least one retained source switch lacks unique pose/body-motion timing evidence.",
                path="render.cut_local_sync.per_boundary",
                value={
                    "unknown_boundary_count": unknown_count,
                    "unknown_reason_counts": cut_local_metrics.get(
                        "unknown_reason_counts"
                    ),
                },
                threshold={"unknown_boundary_count": 0},
            )
        if (
            residual_p95 is not None
            and residual_p95
            > config.maximum_sync_residual_p95_frames + 1e-12
        ):
            collector.add(
                "production_cut_local_sync_residual_exceeds_half_frame",
                QAIssueSeverity.FAIL,
                "Retained-cut pose/body motion phase exceeds the half-frame target.",
                path="render.cut_local_sync.residual_p95_frames",
                value=residual_p95,
                threshold=config.maximum_sync_residual_p95_frames,
            )
        if cut_local_metrics.get("passed") is not True:
            collector.add(
                "production_cut_local_sync_required_gate_failed",
                QAIssueSeverity.FAIL,
                "Registered automatic StageEdit did not satisfy the retained-cut local synchronization gate.",
                path="render.cut_local_sync",
                value=cut_local_metrics.get("passed"),
                threshold=True,
            )
        cut_local_metrics["activation_contract"] = activation
    else:
        cut_local_metrics = {
            "required": False,
            "activation": activation["mode"],
            "activation_contract": activation,
            "available": False,
            "passed": None,
        }

    return {
        "basis": (
            "used_source_piecewise_pts_plus_retained_cut_pose_body_motion_v2"
            if cut_local_required
            else "used_source_piecewise_pts_residual_p95_frames_v1"
        ),
        "domain_mapping_valid": True,
        "fps": fps,
        "used_render_source_indices": [
            int(value) for value in used_render_indices if value is not None
        ],
        "render_indices_are_request_indices": (
            render_indices_are_request_indices
        ),
        "maximum_allowed_residual_p95_frames": (
            config.maximum_sync_residual_p95_frames
        ),
        "minimum_local_timewarp_anchor_count": (
            config.minimum_local_timewarp_anchor_count
        ),
        "maximum_residual_p95_frames": (
            max(residual_frames_values) if residual_frames_values else None
        ),
        "per_source": per_source,
        "cut_local_sync": cut_local_metrics,
        "passed": source_clock_passed
        and not activation_errors
        and (
            bool(cut_local_metrics.get("passed"))
            if cut_local_required
            else True
        ),
    }


PerceptualEvaluator = Callable[..., Mapping[str, Any]]


def _perceptual_quality_metrics(
    artifacts: PostRenderArtifactSet,
    plan: Mapping[str, Any],
    config: PostRenderQAConfig,
    collector: _IssueCollector,
    evaluator: PerceptualEvaluator,
) -> dict[str, Any]:
    """Run the final-resolution scalar audit and import its fail-closed issues."""

    try:
        raw_payload = evaluator(
            artifacts.video_path,
            plan,
            config=config.perceptual_quality,
        )
        payload = _mapping(raw_payload)
    except Exception as exc:
        payload = None
        collector.add(
            "perceptual_quality_evaluation_failed",
            QAIssueSeverity.FAIL,
            "Final-video perceptual quality could not be evaluated.",
            path="perceptual_quality",
            value=f"{type(exc).__name__}: {exc}",
        )
    if payload is None:
        return {
            "analysis": None,
            "verdict": "FAIL",
            "passed": False,
            "issues": [],
            "metrics": {},
        }

    serialized_issues = _sequence(payload.get("issues"))
    verdict = payload.get("verdict")
    passed = payload.get("passed")
    contract_valid = (
        payload.get("schema_version") == 2
        and payload.get("analysis")
        == (
            "full_final_decode_temporal_pumping_exact_source_retention_and_"
            "structure_binding_v2"
        )
        and verdict in {"PASS", "CONDITIONAL", "FAIL"}
        and passed is (verdict == "PASS")
        and serialized_issues is not None
        and (verdict == "PASS") is (len(serialized_issues) == 0)
        and isinstance(payload.get("metrics"), Mapping)
        and (
            (_mapping(payload.get("serialization_policy")) or {}).get(
                "raw_frames_serialized"
            )
            is False
        )
        and (
            (_mapping(payload.get("serialization_policy")) or {}).get(
                "per_frame_descriptors_serialized"
            )
            is False
        )
    )
    if not contract_valid:
        collector.add(
            "perceptual_quality_contract_invalid",
            QAIssueSeverity.FAIL,
            "Final-video perceptual quality returned incomplete or inconsistent evidence.",
            path="perceptual_quality",
            value={"verdict": verdict, "passed": passed},
        )
        return payload

    for index, raw_issue in enumerate(serialized_issues):
        issue = _mapping(raw_issue)
        severity = (issue or {}).get("severity")
        code = (issue or {}).get("code")
        message = (issue or {}).get("message")
        if not (
            isinstance(code, str)
            and code
            and severity in {"conditional", "fail"}
            and isinstance(message, str)
            and message
        ):
            collector.add(
                "perceptual_quality_issue_contract_invalid",
                QAIssueSeverity.FAIL,
                "A perceptual-quality issue record is malformed.",
                path=f"perceptual_quality.issues[{index}]",
                value=raw_issue,
            )
            continue
        collector.add(
            code,
            (
                QAIssueSeverity.FAIL
                if severity == "fail"
                else QAIssueSeverity.CONDITIONAL
            ),
            message,
            path=(issue or {}).get("path"),
            value=(issue or {}).get("value"),
            threshold=(issue or {}).get("threshold"),
        )
    return payload


def _montage_profile_identity(
    value: Any,
    *,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = _mapping(value)
    if profile is None:
        raise ValueError(f"{name} must be an object")
    keys = frozenset(profile)
    if keys not in {
        _MONTAGE_PROFILE_IDENTITY_KEYS,
        _MONTAGE_PROFILE_IDENTITY_KEYS | {"path"},
    }:
        raise ValueError(f"{name} keys are invalid")
    if (
        profile.get("analysis_basis") != MONTAGE_PROFILE_ANALYSIS_BASIS
        or profile.get("schema_version") != MONTAGE_PROFILE_SCHEMA_VERSION
    ):
        raise ValueError(f"{name} schema identity is invalid")
    for key in ("file_sha256", "content_sha256"):
        digest = profile.get(key)
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"{name}.{key} is invalid")
    if "path" in profile:
        path = profile.get("path")
        if not isinstance(path, str) or not path or not Path(path).is_absolute():
            raise ValueError(f"{name}.path must be an absolute non-empty path")
    identity = {
        key: profile[key]
        for key in sorted(_MONTAGE_PROFILE_IDENTITY_KEYS)
    }
    return profile, identity


def _automatic_ive_group_product_scope(analysis: Mapping[str, Any]) -> bool:
    return automatic_ive_group_product_floor_required(analysis)


def _dynamic_crop_v3_motion_evidence_metrics(
    analysis: Mapping[str, Any],
    plan: Mapping[str, Any],
    collector: _IssueCollector,
) -> tuple[dict[str, Any], DynamicCropMotionTimelineEvidenceV3 | None]:
    """Replay the release-bound v3 evidence before any montage recomputation."""

    version = analysis.get("dynamic_crop_version")
    serialized_decodes = analysis.get("dynamic_crop_v3_decodes")
    persisted_evidence = analysis.get("dynamic_crop_motion_v3")
    generation = _mapping(analysis.get("stage_edit_generation"))
    activation = _mapping(generation.get("activation")) if generation else None
    authority_record = (
        _mapping(activation.get("dynamic_crop_v3_motion_authority"))
        if activation is not None
        else None
    )
    selection_policy = (
        _mapping(generation.get("selection_policy"))
        if generation is not None
        else None
    )
    generation_version = (
        selection_policy.get("dynamic_crop_version")
        if selection_policy is not None
        else None
    )
    activation_version = (
        activation.get("dynamic_crop_version")
        if activation is not None
        else None
    )
    registered_v3_claimed = bool(
        generation_version == 3
        or activation_version == 3
        or (
            authority_record is not None
            and (
                authority_record.get("qualified") is True
                or authority_record.get("identity") is not None
            )
        )
    )
    for marker, marker_path in (
        (version, "dynamic_crop_version"),
        (
            generation_version,
            "stage_edit_generation.selection_policy.dynamic_crop_version",
        ),
        (
            activation_version,
            "stage_edit_generation.activation.dynamic_crop_version",
        ),
    ):
        if marker is None:
            continue
        if (
            isinstance(marker, bool)
            or not isinstance(marker, int)
            or marker not in (2, 3)
        ):
            collector.add(
                "dynamic_crop_version_invalid",
                QAIssueSeverity.FAIL,
                "The final analysis has an invalid Dynamic Crop version marker.",
                path=marker_path,
                value=marker,
                threshold={"allowed": [2, 3]},
            )
            return {
                "required": registered_v3_claimed,
                "available": False,
                "passed": False,
                "failure": "dynamic_crop_version_invalid",
            }, None
    if registered_v3_claimed and version != 3:
        collector.add(
            "dynamic_crop_v3_activation_version_mismatch",
            QAIssueSeverity.FAIL,
            "Registered Dynamic Crop v3 authority was downgraded or removed from the final analysis marker.",
            path=(
                "stage_edit_generation.activation.dynamic_crop_version,"
                "stage_edit_generation.activation.dynamic_crop_v3_motion_authority,"
                "dynamic_crop_version"
            ),
            value={
                "activation_dynamic_crop_version": activation_version,
                "generation_dynamic_crop_version": generation_version,
                "analysis_dynamic_crop_version": version,
                "registered_v3_authority_claimed": registered_v3_claimed,
            },
            threshold={
                "activation_dynamic_crop_version": 3,
                "generation_dynamic_crop_version": 3,
                "analysis_dynamic_crop_version": 3,
            },
        )
        return {
            "required": True,
            "available": False,
            "passed": False,
            "dynamic_crop_version": version,
            "failure": "dynamic_crop_v3_activation_version_mismatch",
        }, None
    if version != 3:
        unexpected = bool(
            serialized_decodes is not None or persisted_evidence is not None
        )
        if unexpected:
            collector.add(
                "dynamic_crop_v3_evidence_without_version",
                QAIssueSeverity.FAIL,
                "Dynamic Crop v3 replay evidence exists without a v3 version marker.",
                path="dynamic_crop_v3_decodes,dynamic_crop_motion_v3",
            )
        return {
            "required": False,
            "available": unexpected,
            "passed": not unexpected,
            "dynamic_crop_version": version,
        }, None

    if activation_version != 3 or generation_version != 3:
        collector.add(
            "dynamic_crop_v3_activation_version_mismatch",
            QAIssueSeverity.FAIL,
            "Final Dynamic Crop v3 evidence lacks a matching registered activation version.",
            path=(
                "stage_edit_generation.activation.dynamic_crop_version,"
                "dynamic_crop_version"
            ),
            value={
                "activation_dynamic_crop_version": activation_version,
                "generation_dynamic_crop_version": generation_version,
                "analysis_dynamic_crop_version": version,
            },
            threshold={
                "activation_dynamic_crop_version": 3,
                "generation_dynamic_crop_version": 3,
                "analysis_dynamic_crop_version": 3,
            },
        )
        return {
            "required": True,
            "available": False,
            "passed": False,
            "dynamic_crop_version": 3,
            "failure": "dynamic_crop_v3_activation_version_mismatch",
        }, None
    identity = (
        authority_record.get("identity")
        if authority_record is not None
        and authority_record.get("qualified") is True
        else None
    )
    try:
        replayed = replay_dynamic_crop_v3_post_render_evidence(
            serialized_render_plan=plan,
            registered_motion_authority_identity=identity,
            serialized_decodes=serialized_decodes,
            persisted_timeline_evidence=persisted_evidence,
        )
    except (OSError, TypeError, ValueError) as exc:
        collector.add(
            "dynamic_crop_v3_post_render_evidence_replay_failed",
            QAIssueSeverity.FAIL,
            "The final Dynamic Crop v3 motion evidence could not be independently replayed.",
            path=(
                "stage_edit_generation.activation.dynamic_crop_v3_motion_authority,"
                "dynamic_crop_v3_decodes,dynamic_crop_motion_v3"
            ),
            value=f"{type(exc).__name__}: {exc}",
            threshold={
                "contract": DYNAMIC_CROP_V3_POST_RENDER_EVIDENCE_REPLAY_CONTRACT,
                "passed": True,
            },
        )
        return {
            "required": True,
            "available": False,
            "passed": False,
            "dynamic_crop_version": 3,
            "contract": DYNAMIC_CROP_V3_POST_RENDER_EVIDENCE_REPLAY_CONTRACT,
            "error": f"{type(exc).__name__}: {exc}",
        }, None
    return {
        "required": True,
        "available": True,
        "passed": True,
        "dynamic_crop_version": 3,
        "contract": DYNAMIC_CROP_V3_POST_RENDER_EVIDENCE_REPLAY_CONTRACT,
        "render_plan_content_sha256": replayed.render_plan_content_sha256,
        "authority_content_sha256": (
            replayed.authority.authority_content_sha256
        ),
        "timeline_evidence_content_sha256": (
            replayed.timeline_evidence_content_sha256
        ),
        "segment_count": replayed.segment_count,
        "framing_track_applied_count": (
            replayed.framing_track_applied_count
        ),
        "motion_event_applied_count": replayed.motion_event_applied_count,
        "motion_event_applied_fraction": replayed.motion_event_applied_fraction,
        "static_or_cut_time_reframe_excluded_count": (
            replayed.static_or_cut_time_reframe_excluded_count
        ),
        "renderer_motion_without_loso_authority_count": (
            replayed.renderer_motion_without_loso_authority_count
        ),
        "motion_budget_within_range": replayed.motion_budget_within_range,
    }, replayed


def _rendered_montage_grammar_metrics(
    analysis: Mapping[str, Any],
    plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    config: PostRenderQAConfig,
    collector: _IssueCollector,
    *,
    trusted_registered_stage_edit: bool | None = None,
    replayed_dynamic_crop_motion_v3: (
        DynamicCropMotionTimelineEvidenceV3 | None
    ) = None,
) -> dict[str, Any]:
    """Recompute the final montage grammar for every IVE product delivery."""

    activation = resolve_cut_local_sync_activation(
        analysis,
        selection,
        trusted_registered_stage_edit=trusted_registered_stage_edit,
    )
    ive_product_required = _automatic_ive_group_product_scope(analysis)
    montage_required = bool(activation["required"] or ive_product_required)
    if not montage_required:
        return {
            "required": False,
            "activation_contract": activation,
            "ive_product_scope_required": False,
            "available": False,
            "passed": None,
        }

    persisted = _mapping(analysis.get("rendered_montage_grammar_review"))
    generation = _mapping(analysis.get("stage_edit_generation"))
    generation_profile_raw = (
        generation.get("professional_montage_grammar_profile")
        if generation is not None
        else None
    )
    evidence_valid = True
    configuration_matches = False
    serialized_review_matches = False
    profile_identity_matches = False
    professional_boundary_evidence_valid = False
    professional_boundary_assessment_matches = False
    professional_boundary_render_plan_binding_matches = False
    professional_boundary_authority_reconstruction_matches = False
    professional_boundary_legacy_pass_binding_matches = False
    recomputed: dict[str, Any] | None = None
    recomputed_boundary_assessment: dict[str, Any] | None = None
    generation_profile: dict[str, Any] | None = None
    persisted_profile: dict[str, Any] | None = None
    failure_details: list[str] = []

    if persisted is None:
        evidence_valid = False
        failure_details.append("serialized_review_missing")
        collector.add(
            "production_rendered_montage_grammar_evidence_missing_or_invalid",
            QAIssueSeverity.FAIL,
            "Registered automatic StageEdit lacks a serialized final montage-grammar review.",
            path="rendered_montage_grammar_review",
        )

    generation_identity: dict[str, Any] | None = None
    persisted_identity: dict[str, Any] | None = None
    try:
        generation_profile, generation_identity = _montage_profile_identity(
            generation_profile_raw,
            name="stage_edit_generation.professional_montage_grammar_profile",
        )
        if persisted is None:
            raise ValueError("rendered montage review is missing")
        if (
            persisted.get("schema_version") != 1
            or persisted.get("analysis_basis")
            != RENDERED_MONTAGE_GRAMMAR_ANALYSIS_BASIS
            or not isinstance(persisted.get("passed"), bool)
            or not isinstance(persisted.get("failure_codes"), list)
        ):
            raise ValueError("serialized rendered montage review contract is invalid")
        persisted_profile, persisted_identity = _montage_profile_identity(
            persisted.get("reference_profile"),
            name="rendered_montage_grammar_review.reference_profile",
        )
    except (TypeError, ValueError) as exc:
        evidence_valid = False
        failure_details.append("profile_or_review_contract_invalid")
        collector.add(
            "production_rendered_montage_grammar_evidence_missing_or_invalid",
            QAIssueSeverity.FAIL,
            "Registered automatic StageEdit montage evidence is missing or malformed.",
            path=(
                "rendered_montage_grammar_review,"
                "stage_edit_generation.professional_montage_grammar_profile"
            ),
            value=f"{type(exc).__name__}: {exc}",
        )

    if generation_identity is not None and persisted_identity is not None:
        profile_identity_matches = generation_identity == persisted_identity
        if not profile_identity_matches:
            failure_details.append("reference_profile_identity_mismatch")
            collector.add(
                "production_rendered_montage_grammar_profile_identity_mismatch",
                QAIssueSeverity.FAIL,
                "The final montage review is bound to a different human-reference profile.",
                path="rendered_montage_grammar_review.reference_profile",
                value=persisted_identity,
                threshold=generation_identity,
            )

    recompute_analysis = deepcopy(dict(analysis))
    if analysis.get("dynamic_crop_version") == 3:
        recompute_analysis["dynamic_crop_motion_v3"] = (
            replayed_dynamic_crop_motion_v3.to_dict()
            if replayed_dynamic_crop_motion_v3 is not None
            else None
        )
    try:
        recomputed = review_rendered_montage_grammar(
            recompute_analysis,
            render_plan=plan,
            config=config.rendered_montage_grammar,
        )
    except Exception as exc:
        evidence_valid = False
        failure_details.append("independent_recompute_failed")
        collector.add(
            "production_rendered_montage_grammar_evidence_missing_or_invalid",
            QAIssueSeverity.FAIL,
            "The final montage grammar could not be recomputed from bound artifacts.",
            path="rendered_montage_grammar_review",
            value=f"{type(exc).__name__}: {exc}",
        )

    boundary_evidence: ProfessionalBoundaryGrammarEvidence | None = None
    if persisted is not None and recomputed is not None:
        try:
            serialized_boundary_evidence = _mapping(
                persisted.get("professional_boundary_grammar_evidence")
            )
            serialized_boundary_assessment = _mapping(
                persisted.get("professional_boundary_grammar_assessment")
            )
            legacy_pass = persisted.get(
                "passed_before_professional_boundary_grammar"
            )
            if serialized_boundary_evidence is None:
                raise ValueError(
                    "professional boundary grammar evidence must be an object"
                )
            if serialized_boundary_assessment is None:
                raise ValueError(
                    "professional boundary grammar assessment must be an object"
                )
            if not isinstance(legacy_pass, bool):
                raise ValueError(
                    "passed_before_professional_boundary_grammar must be boolean"
                )

            boundary_evidence = ProfessionalBoundaryGrammarEvidence.from_mapping(
                serialized_boundary_evidence
            )
            professional_boundary_evidence_valid = True
            canonical_plan_sha256 = hashlib.sha256(
                json.dumps(
                    plan,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            expected_performance_id = (
                f"ive-group-{canonical_plan_sha256[:20]}"
            )
            typed_boundary_plan = (
                _dynamic_crop_render_plan_from_serialized(plan)
                if analysis.get("dynamic_crop_version") == 3
                else _boundary_render_plan_from_serialized(plan)
            )
            source_scene_cut_times = _boundary_scene_cut_times_from_analysis(
                analysis,
                plan=typed_boundary_plan,
            )
            reconstructed_boundary_evidence = (
                adapt_group_boundary_grammar_evidence(
                    performance_id=expected_performance_id,
                    render_plan=typed_boundary_plan,
                    source_index_mapping=analysis.get("source_index_mapping"),
                    selected_timeline=analysis.get("visual_timeline"),
                    source_scene_cut_times=source_scene_cut_times,
                    timeline_review=analysis.get("timeline_review"),
                    section_plan=analysis.get("section_plan"),
                    rendered_proxy_review=analysis.get(
                        "rendered_proxy_review"
                    ),
                    editorial_motion=analysis.get("editorial_motion"),
                    dynamic_crop_version=analysis.get(
                        "dynamic_crop_version"
                    ),
                    dynamic_crop_motion_v3=(
                        replayed_dynamic_crop_motion_v3
                    ),
                )
            )
            professional_boundary_authority_reconstruction_matches = bool(
                boundary_evidence.to_dict()
                == reconstructed_boundary_evidence.to_dict()
            )
            professional_boundary_render_plan_binding_matches = bool(
                professional_boundary_authority_reconstruction_matches
                and reconstructed_boundary_evidence.performance_id
                == expected_performance_id
            )
            if not professional_boundary_authority_reconstruction_matches:
                failure_details.append(
                    "professional_boundary_authority_reconstruction_mismatch"
                )
                collector.add(
                    "production_professional_boundary_grammar_authority_reconstruction_mismatch",
                    QAIssueSeverity.FAIL,
                    "Professional boundary evidence disagrees with an independent reconstruction from final plan, scene, section, proxy, timeline, and transform authorities.",
                    path=(
                        "rendered_montage_grammar_review."
                        "professional_boundary_grammar_evidence"
                    ),
                    value=boundary_evidence.to_dict(),
                    threshold=reconstructed_boundary_evidence.to_dict(),
                )

            recomputed_boundary_assessment = (
                assess_professional_boundary_grammar(
                    reconstructed_boundary_evidence
                ).to_dict()
            )
            professional_boundary_assessment_matches = bool(
                dict(serialized_boundary_assessment)
                == recomputed_boundary_assessment
            )
            if not professional_boundary_assessment_matches:
                failure_details.append(
                    "professional_boundary_assessment_recompute_mismatch"
                )
                collector.add(
                    "production_professional_boundary_grammar_assessment_mismatch",
                    QAIssueSeverity.FAIL,
                    "The serialized professional boundary assessment does not match an independent recomputation.",
                    path=(
                        "rendered_montage_grammar_review."
                        "professional_boundary_grammar_assessment"
                    ),
                )

            professional_boundary_legacy_pass_binding_matches = bool(
                legacy_pass is (recomputed.get("passed") is True)
            )
            if not professional_boundary_legacy_pass_binding_matches:
                failure_details.append(
                    "professional_boundary_legacy_pass_binding_mismatch"
                )
                collector.add(
                    "production_professional_boundary_grammar_legacy_pass_binding_mismatch",
                    QAIssueSeverity.FAIL,
                    "The professional boundary gate is not bound to the independently recomputed legacy montage verdict.",
                    path=(
                        "rendered_montage_grammar_review."
                        "passed_before_professional_boundary_grammar"
                    ),
                    value=legacy_pass,
                    threshold=recomputed.get("passed") is True,
                )

            if not professional_boundary_render_plan_binding_matches:
                failure_details.append(
                    "professional_boundary_render_plan_binding_mismatch"
                )
                collector.add(
                    "production_professional_boundary_grammar_render_plan_binding_mismatch",
                    QAIssueSeverity.FAIL,
                    "Professional source-switch clocks or identity disagree with the exact final RenderPlan.",
                    path=(
                        "rendered_montage_grammar_review."
                        "professional_boundary_grammar_evidence"
                    ),
                    value=boundary_evidence.to_dict(),
                    threshold=reconstructed_boundary_evidence.to_dict(),
                )

            recomputed[
                "professional_boundary_grammar_evidence"
            ] = reconstructed_boundary_evidence.to_dict()
            recomputed[
                "professional_boundary_grammar_assessment"
            ] = recomputed_boundary_assessment
            recomputed[
                "passed_before_professional_boundary_grammar"
            ] = recomputed.get("passed") is True
            recomputed["passed"] = bool(
                recomputed[
                    "passed_before_professional_boundary_grammar"
                ]
                and recomputed_boundary_assessment.get("passed") is True
            )
        except Exception as exc:
            professional_boundary_evidence_valid = False
            evidence_valid = False
            failure_details.append(
                "professional_boundary_evidence_missing_or_invalid"
            )
            collector.add(
                "production_professional_boundary_grammar_evidence_missing_or_invalid",
                QAIssueSeverity.FAIL,
                "Professional boundary evidence is missing, malformed, or cannot be independently assessed.",
                path=(
                    "rendered_montage_grammar_review."
                    "professional_boundary_grammar_evidence"
                ),
                value=f"{type(exc).__name__}: {exc}",
            )

    if persisted is not None and recomputed is not None:
        configuration_matches = (
            persisted.get("configuration") == recomputed.get("configuration")
        )
        if not configuration_matches:
            failure_details.append("configuration_mismatch")
            collector.add(
                "production_rendered_montage_grammar_configuration_mismatch",
                QAIssueSeverity.FAIL,
                "The serialized montage review used a different release policy.",
                path="rendered_montage_grammar_review.configuration",
                value=persisted.get("configuration"),
                threshold=recomputed.get("configuration"),
            )
        persisted_core = dict(persisted)
        persisted_core.pop("reference_profile", None)
        serialized_review_matches = persisted_core == recomputed
        if not serialized_review_matches:
            failure_details.append("serialized_review_recompute_mismatch")
            collector.add(
                "production_rendered_montage_grammar_serialized_review_mismatch",
                QAIssueSeverity.FAIL,
                "The serialized final montage review does not match an independent recomputation.",
                path="rendered_montage_grammar_review",
            )

    recomputed_passed = bool(
        recomputed is not None and recomputed.get("passed") is True
    )
    persisted_passed = bool(
        persisted is not None and persisted.get("passed") is True
    )
    passed = bool(
        evidence_valid
        and profile_identity_matches
        and configuration_matches
        and serialized_review_matches
        and professional_boundary_evidence_valid
        and professional_boundary_assessment_matches
        and professional_boundary_render_plan_binding_matches
        and professional_boundary_authority_reconstruction_matches
        and professional_boundary_legacy_pass_binding_matches
        and persisted_passed
        and recomputed_passed
    )
    if not passed:
        collector.add(
            "production_rendered_montage_grammar_required_gate_failed",
            QAIssueSeverity.FAIL,
            "Registered automatic StageEdit did not satisfy the final rendered montage-grammar gate.",
            path="rendered_montage_grammar_review",
            value={
                "failure_details": failure_details,
                "persisted_passed": persisted_passed,
                "recomputed_passed": recomputed_passed,
                "recomputed_failure_codes": (
                    recomputed.get("failure_codes")
                    if recomputed is not None
                    else None
                ),
            },
            threshold={"passed": True},
        )

    recomputed_with_profile = dict(recomputed) if recomputed is not None else None
    if recomputed_with_profile is not None and generation_profile is not None:
        recomputed_with_profile["reference_profile"] = generation_profile
    return {
        "required": True,
        "activation_contract": activation,
        "ive_product_scope_required": ive_product_required,
        "available": persisted is not None,
        "evidence_contract_valid": evidence_valid,
        "configuration_matches": configuration_matches,
        "serialized_review_matches_recomputation": serialized_review_matches,
        "reference_profile_identity_matches": profile_identity_matches,
        "professional_boundary_evidence_valid": (
            professional_boundary_evidence_valid
        ),
        "professional_boundary_assessment_matches_recomputation": (
            professional_boundary_assessment_matches
        ),
        "professional_boundary_render_plan_binding_matches": (
            professional_boundary_render_plan_binding_matches
        ),
        "professional_boundary_authority_reconstruction_matches": (
            professional_boundary_authority_reconstruction_matches
        ),
        "professional_boundary_legacy_pass_binding_matches": (
            professional_boundary_legacy_pass_binding_matches
        ),
        "generation_reference_profile": generation_profile,
        "persisted_reference_profile": persisted_profile,
        "persisted_review": persisted,
        "recomputed_review": recomputed_with_profile,
        "passed": passed,
    }


def _ive_product_quality_floor_metrics(
    analysis: Mapping[str, Any],
    selection: Mapping[str, Any],
    collector: _IssueCollector,
) -> dict[str, Any]:
    """Apply the IVE floor regardless of registered versus explicit activation."""

    try:
        result = evaluate_ive_product_quality_floor(
            analysis,
            selection=selection,
        )
        payload = result.to_dict()
    except Exception as exc:
        collector.add(
            "ive_product_quality_floor_evidence_invalid",
            QAIssueSeverity.FAIL,
            "The IVE delivery floor could not validate its immutable evidence.",
            path="ive_product_quality_floor",
            value=f"{type(exc).__name__}: {exc}",
        )
        return {
            "required": _automatic_ive_group_product_scope(analysis),
            "passed": False,
            "evaluation_error": f"{type(exc).__name__}: {exc}",
        }
    if result.required and result.passed is not True:
        collector.add(
            "ive_product_quality_floor_failed",
            QAIssueSeverity.FAIL,
            "The automatic IVE GROUP render regressed below its delivery floor.",
            path="ive_product_quality_floor",
            value={
                "evaluation_mode": result.evaluation_mode,
                "failure_codes": list(result.failure_codes),
            },
            threshold={"passed": True},
        )
    return payload


def evaluate_post_render_qa(
    artifact_set: PostRenderArtifactSet,
    *,
    declared_artifact_set: PostRenderArtifactSet | None = None,
    config: PostRenderQAConfig | None = None,
    runner: RunCommand = _run_command,
    perceptual_evaluator: PerceptualEvaluator | None = None,
    created_at: datetime | None = None,
    trusted_registered_stage_edit: bool | None = None,
) -> PostRenderQAReport:
    """Evaluate one explicit completed set without modifying any input artifact."""

    settings = config or PostRenderQAConfig()
    collector = _IssueCollector()
    metrics: dict[str, Any] = {}
    input_artifact_fingerprints: dict[str, Any] = {}
    try:
        input_artifact_fingerprints = fingerprint_post_render_artifacts(
            artifact_set
        )
    except ValueError as exc:
        collector.add(
            "artifact_content_fingerprint_unavailable",
            QAIssueSeverity.FAIL,
            "The completed artifact set could not be content-fingerprinted.",
            path="artifact_set",
            value=str(exc),
        )
    try:
        analysis = _load_json_object(artifact_set.analysis_path)
        plan = _load_json_object(artifact_set.render_plan_path)
        selection = _load_json_object(artifact_set.selection_path)
    except ValueError as exc:
        collector.add(
            "artifact_json_unavailable_or_invalid",
            QAIssueSeverity.FAIL,
            "The timestamp-bound JSON artifact set could not be loaded.",
            path="artifact_set",
            value=str(exc),
        )
        return PostRenderQAReport(
            artifact_set=artifact_set,
            verdict=collector.verdict(),
            metrics=metrics,
            issues=tuple(collector.issues),
            policy=settings,
            input_artifact_fingerprints=input_artifact_fingerprints,
            created_at=created_at or datetime.now(UTC),
        )

    # The policy must agree in BOTH independently loaded artifacts. Never use
    # an arbitrary serialized candidate count to weaken the decoded QA gates.
    render_payload = _mapping(analysis.get("render")) or {}
    release_policy = selection.get("release_policy")
    review_first_release_active = False
    if release_policy is not None or render_payload.get("release_policy") is not None:
        if (
            release_policy == joint_release_policy_payload()
            and render_payload.get("release_policy") == release_policy
            and selection.get("candidate_pool") == "joint_decoded_visual_selector"
            and render_payload.get("candidate_origin") == "joint_decoded_visual_selector"
        ):
            settings = replace(
                settings,
                required_complete_candidate_count=JOINT_RELEASE_CANDIDATE_COUNT,
            )
        elif stage_edit_release_binding_valid(selection, render_payload):
            settings = replace(settings, required_complete_candidate_count=1)
        elif review_first_release_binding_valid(selection, render_payload):
            review_first_release_active = True
            settings = replace(
                settings,
                required_complete_candidate_count=(
                    REVIEW_FIRST_RELEASE_CANDIDATE_COUNT
                ),
            )
        else:
            collector.add(
                "candidate_release_policy_mismatch", QAIssueSeverity.FAIL,
                "Candidate release policy is invalid or differs across artifacts.",
                path="selection.release_policy", value=release_policy,
                threshold={"joint": joint_release_policy_payload(),
                           "stage_edit": stage_edit_release_policy_payload(),
                           "review_first": review_first_release_policy_payload()},
            )

    metrics["artifact_binding"] = _validate_artifact_binding(
        declared_artifact_set or artifact_set,
        analysis,
        plan,
        selection,
        collector,
    )
    selection_metrics, final_review = _validate_selection(
        analysis,
        plan,
        selection,
        settings,
        collector,
        allow_failed_review=review_first_release_active,
    )
    metrics["selection"] = selection_metrics
    metrics["plan_invariants"] = _validate_plan_invariants(
        analysis, plan, final_review, collector
    )
    tool_checks = _probe_and_decode(
        artifact_set.video_path,
        config=settings,
        runner=runner,
        collector=collector,
    )
    metrics["tool_checks"] = tool_checks
    metrics["final_media"] = _validate_ffprobe_against_plan(
        tool_checks, analysis, plan, settings, collector
    )
    metrics["perceptual_quality"] = _perceptual_quality_metrics(
        artifact_set,
        plan,
        settings,
        collector,
        perceptual_evaluator or evaluate_post_render_perceptual_quality,
    )
    metrics["production_sync"] = _sync_quality_metrics(
        analysis,
        plan,
        settings,
        collector,
        selection=selection,
        trusted_registered_stage_edit=trusted_registered_stage_edit,
    )
    (
        dynamic_crop_v3_motion_metrics,
        replayed_dynamic_crop_motion_v3,
    ) = _dynamic_crop_v3_motion_evidence_metrics(
        analysis,
        plan,
        collector,
    )
    metrics["dynamic_crop_v3_motion_evidence"] = (
        dynamic_crop_v3_motion_metrics
    )
    metrics["rendered_montage_grammar"] = _rendered_montage_grammar_metrics(
        analysis,
        plan,
        selection,
        settings,
        collector,
        trusted_registered_stage_edit=trusted_registered_stage_edit,
        replayed_dynamic_crop_motion_v3=(
            replayed_dynamic_crop_motion_v3
        ),
    )
    metrics["ive_product_quality_floor"] = _ive_product_quality_floor_metrics(
        analysis,
        selection,
        collector,
    )
    metrics["cadence_and_repetition"] = _cadence_and_repetition_metrics(
        analysis, plan, settings, collector
    )
    metrics["proxy_quality_and_geometry"] = _proxy_quality_and_geometry_metrics(
        analysis, final_review, settings, collector
    )
    metrics["source_local_reuse"] = _source_local_reuse_metrics(
        plan, settings, collector
    )
    metrics["transforms"] = _transform_metrics(
        analysis, plan, settings, collector
    )
    metrics["boundary_pose_evidence"] = _boundary_pose_evidence_metrics(
        analysis, selection, collector
    )
    metrics["beat_and_phrase"] = _beat_phrase_metrics(
        analysis, plan, selection, settings, collector
    )
    final_artifact_fingerprints: dict[str, Any] = {}
    try:
        final_artifact_fingerprints = fingerprint_post_render_artifacts(
            artifact_set
        )
    except ValueError as exc:
        collector.add(
            "artifact_content_stability_unavailable",
            QAIssueSeverity.FAIL,
            "The QA inputs could not be re-fingerprinted after evaluation.",
            path="artifact_set",
            value=str(exc),
        )
    stable = bool(
        input_artifact_fingerprints
        and final_artifact_fingerprints
        and input_artifact_fingerprints == final_artifact_fingerprints
    )
    metrics["artifact_content_stability"] = {
        "contract": INPUT_ARTIFACT_FINGERPRINT_CONTRACT,
        "stable_across_evaluation": stable,
        "final_fingerprints": final_artifact_fingerprints,
    }
    if not stable:
        collector.add(
            "artifact_content_changed_during_qa",
            QAIssueSeverity.FAIL,
            "At least one QA input changed while the report was being evaluated.",
            path="artifact_set",
            value={
                "initial": input_artifact_fingerprints,
                "final": final_artifact_fingerprints,
            },
        )
    return PostRenderQAReport(
        artifact_set=artifact_set,
        verdict=collector.verdict(),
        metrics=metrics,
        issues=tuple(collector.issues),
        policy=settings,
        input_artifact_fingerprints=input_artifact_fingerprints,
        created_at=created_at or datetime.now(UTC),
    )


def evaluate_and_write_post_render_qa(
    artifact_set: PostRenderArtifactSet,
    *,
    output_path: Path | None = None,
    declared_artifact_set: PostRenderArtifactSet | None = None,
    config: PostRenderQAConfig | None = None,
    runner: RunCommand = _run_command,
    created_at: datetime | None = None,
    trusted_registered_stage_edit: bool | None = None,
) -> tuple[PostRenderQAReport, Path]:
    report = evaluate_post_render_qa(
        artifact_set,
        declared_artifact_set=declared_artifact_set,
        config=config,
        runner=runner,
        created_at=created_at,
        trusted_registered_stage_edit=trusted_registered_stage_edit,
    )
    if declared_artifact_set is not None:
        normalized_fingerprints = normalize_post_render_artifact_fingerprints(
            report.input_artifact_fingerprints,
            declared_artifact_set,
        )
        normalized_metrics = deepcopy(dict(report.metrics))
        stability = normalized_metrics.get("artifact_content_stability")
        if isinstance(stability, dict):
            final_fingerprints = stability.get("final_fingerprints")
            if isinstance(final_fingerprints, Mapping):
                stability["final_fingerprints"] = (
                    normalize_post_render_artifact_fingerprints(
                        final_fingerprints,
                        declared_artifact_set,
                    )
                )
        report = replace(
            report,
            artifact_set=declared_artifact_set,
            metrics=normalized_metrics,
            input_artifact_fingerprints=normalized_fingerprints,
        )
    return (report, report.write_atomic(output_path))


__all__ = [
    "ANALYSIS_BASIS",
    "DYNAMIC_CROP_V3_POST_RENDER_EVIDENCE_REPLAY_CONTRACT",
    "INPUT_ARTIFACT_FINGERPRINT_CONTRACT",
    "DynamicCropV3PostRenderEvidenceReplayError",
    "PostRenderArtifactSet",
    "PostRenderQAEvaluationSnapshot",
    "PostRenderQAConfig",
    "PostRenderQAReport",
    "PostRenderQAVerdict",
    "QAIssue",
    "QAIssueSeverity",
    "SCHEMA_VERSION",
    "discover_completed_artifact_sets",
    "evaluate_and_write_post_render_qa",
    "evaluate_post_render_qa",
    "cleanup_post_render_qa_snapshot",
    "fingerprint_post_render_artifacts",
    "materialize_post_render_qa_snapshot",
    "normalize_post_render_artifact_fingerprints",
    "replay_dynamic_crop_v3_post_render_evidence",
]
