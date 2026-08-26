"""Top-level offline analysis pipeline used by the native app."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import stat
from statistics import median
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence

from numpy.typing import NDArray
import numpy as np
import cv2

from .acquisition import (
    VerifiedMemberAcquisitionIdentityAuthority,
    canonical_music_show_program,
    is_verified_member_acquisition_identity_authority,
)
from .boundary_affine_plan import BoundaryAffinePlan, build_boundary_affine_plan
from .boundary_subject_alignment import (
    BoundarySubjectAlignmentPlan,
    build_boundary_subject_alignment_plan,
)
from .boundary_look_match import (
    BoundaryLookMatchResult,
    BoundaryLookMeasurementCache,
    analyze_boundary_look_match,
    verify_boundary_look_match_render_domain,
)
from .boundary_repair import BoundaryRepairConfig
from .cut_local_sync import (
    MEMBER_ACTIVATION as MEMBER_CUT_LOCAL_SYNC_ACTIVATION,
    analyze_cut_local_sync,
)
from .audio import (
    DEFAULT_ANALYSIS_SAMPLE_RATE,
    AlignmentResult,
    AudioDecodeError,
    SourceAlignment,
    extract_mono_pcm,
    normalized_cross_correlation_offset,
    spectral_flux_alignment_offset,
)
from .direct_member_visual import (
    DescriptorProvider,
    DirectMemberVisualConfig,
    DirectMemberVisualSelection,
    DirectMemberVisualSelectionError,
    _VideoDescriptorProvider,
    select_balanced_direct_member_sources,
)
from .direct_member_editorial_motion import (
    DirectMemberDenseFaceProvider,
    DirectMemberEditorialMotionConfig,
    build_direct_member_editorial_motion_plan,
)
from .direct_member_professional_gate import (
    evaluate_direct_member_professional_pre_render_gate,
)
from .direct_member_look import (
    DirectMemberLookPlan,
    build_direct_member_look_plan,
)
from .direct_member_transition import (
    OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT,
    build_direct_member_pose_dissolve_plan,
)
from .editorial_motion import EditorialMotionPlan, build_editorial_motion_plan
from .events import EventReporter
from .face_match import FaceMatchPlan, build_face_match_plan
from .group_transition import GroupTransitionPlan, build_group_transition_plan
from .group_boundary_grammar_adapter import (
    adapt_group_boundary_grammar_evidence,
)
from .draft import DraftSegment, GroupDraftTimeline, build_group_draft
from .ive import AspectRatio, EditMode, SourceMode
from .ive_causal_sensing_pipeline_inputs import (
    IVECausalSensingPipelineAuditArtifact,
    materialize_ive_causal_sensing_for_stage_edit_pipeline,
    write_ive_causal_sensing_pipeline_audit,
)
from .media import MediaInfo, probe_media
from .member_draft import (
    MemberDraftSegment,
    MemberDraftTimeline,
    MemberSegmentKind,
    build_member_draft,
)
from .member_editorial_motion import build_member_editorial_motion_plan
from .member_post_render_qa import (
    AUTOMATIC_IVE_MEMBER_APP_RESULT_EXPOSURE_CONTRACT,
    AUTOMATIC_IVE_MEMBER_POST_RENDER_QA_CONTRACT,
    AutomaticMemberPostRenderArtifactSet,
    automatic_member_post_render_qa_required,
    evaluate_and_write_automatic_member_post_render_qa,
    fingerprint_automatic_member_artifacts,
)
from .performance_range import estimate_shared_performance_range
from .post_render_qa import (
    INPUT_ARTIFACT_FINGERPRINT_CONTRACT,
    PostRenderArtifactSet,
    PostRenderQAEvaluationSnapshot,
    PostRenderQAConfig,
    cleanup_post_render_qa_snapshot,
    evaluate_and_write_post_render_qa,
    fingerprint_post_render_artifacts,
    materialize_post_render_qa_snapshot,
    normalize_post_render_artifact_fingerprints,
)
from .ive_product_quality_floor import (
    APP_RESULT_EXPOSURE_CONTRACT,
    PRE_REVIEW_REPAIR_LINEAGE_CONTRACT,
)
from .professional_reference_gate import ProfessionalReferenceProfile
from .professional_reference_cadence import (
    ProfessionalReferenceCadenceProfile,
)
from .professional_montage_policy import (
    CLOSE_BURST_DURATION_STATISTIC_PATH,
    CLOSE_BURST_SHOT_COUNT_STATISTIC_PATH,
    CLOSE_BURST_SOURCE_SWITCH_FRACTION_STATISTIC_PATH,
    REFERENCE_FLOOR_QUANTILE,
    ProfessionalMontagePolicy,
    close_burst_policy_binding_sha256,
    load_professional_montage_policy,
)
from .professional_montage_grammar import (
    assess_professional_boundary_grammar,
)
from .professional_motion_cut_profile import (
    ProfessionalMotionCutProfile,
    load_professional_motion_cut_profile,
)
from .render import (
    NormalizedCrop,
    RENDER_PLAN_SOURCE_INDEX_DOMAIN,
    RenderPlan,
    RenderSegment,
    RenderTransition,
    TransitionKind,
    aligned_render_segment,
    output_size,
    render,
    write_plan,
)
from .request import (
    AnalysisRequest,
    MAXIMUM_AUTOMATIC_MEMBER_SOURCE_COUNT,
    MINIMUM_AUTOMATIC_MEMBER_SOURCE_COUNT,
    MINIMUM_AUTOMATIC_SOURCE_COUNT,
)
from .rendered_candidate_selection import (
    RenderedCandidate,
    RenderedCandidateRankMetadata,
    RenderedCandidateSelection,
    review_rendered_candidates,
)
from .rendered_proxy_review import (
    PERCEPTUAL_SOURCE_ADMISSION_ANALYSIS_BASIS,
    RenderedProxyReview,
    render_and_review_proxy,
)
from .rendered_montage_grammar_review import (
    EXACT_STAGE_EDIT_SHOT_CLOCK_CONTRACT,
    RenderedMontageGrammarConfig,
    review_rendered_montage_grammar,
)
from .rhythm import RhythmAnalysis, analyze_rhythm
from .section_director import SectionPlan, SectionRole, build_section_plan
from .source_look_match import SourceLookMatchResult, analyze_source_look_match
from .source_scalar_cache import (
    PersistentSourceScalarCache,
    source_content_sha256,
)
from .source_derivative_diversity import (
    SourceDerivativeDiversityResult,
    analyze_source_derivative_diversity,
)
from .stage_edit_candidate_rerank import (
    StageEditCandidateRerankResult,
    rerank_visual_timeline_candidates_with_stage_edit,
)
from .stage_edit import StageEdit
from .stage_edit_dynamic_crop_motion_authority_v3 import (
    DynamicCropMotionAuthorityV3,
    DynamicCropMotionTimelineEvidenceV3,
    apply_dynamic_crop_motion_authority_to_render_plan_v3,
    corroborate_dynamic_crop_motion_timeline_v3,
    load_dynamic_crop_motion_authority_from_registered_identity_v3,
)
from .stage_edit_dynamic_crop_semimarkov_adapter_v3 import (
    DynamicCropTrackDecodeV3,
)
from .stage_edit_learned_reviewer_adapter import (
    LearnedReviewerProductionReviewConfig,
    LearnedReviewerProductionSession,
    augment_stage_edit_repair_review_with_learned_issues,
)
from .stage_edit_inference_snapshot import (
    StageEditInferenceSnapshotResult,
    materialize_stage_edit_inference_snapshot,
)
from .stage_edit_pipeline_bridge import (
    RetainedStageEditCandidate,
    StageEditPipelineBridgePreflight,
    StageEditPipelineBridgeResult,
    preflight_stage_edit_pipeline_bridge,
    run_stage_edit_pipeline_bridge,
)
from .stage_edit_continuity_authority import (
    HANDCRAFTED_CONTINUITY_AUTHORITY,
    PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
    PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
)
from .stage_edit_semimarkov_beam import (
    HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
    HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT,
    StageEditSemiMarkovConfig,
)
from .stage_edit_source_pair_scorer import (
    STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT,
)
from .stage_edit_source_pair_contextual_v2 import (
    STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SCORER_CONTRACT,
)
from .stage_edit_transformer import stage_edit_readiness_report
from .stage_edit_transition_authority import (
    RENDERER_OWNED_TRANSITION_AUTHORITY,
)
from .stage_edit_production_registry import (
    RegisteredStageEditProductionBundle,
    discover_registered_stage_edit_production_bundle,
)
from .stage_edit_visual_candidate_adapter import (
    STAGE_EDIT_VISUAL_CANDIDATE_CONTRACT,
    StageEditMaterializedVisualCandidate,
    StageEditVisualCandidateBatch,
    StageEditVisualCandidateError,
    apply_stage_edit_dynamic_crop_to_render_plan,
    materialize_stage_edit_visual_candidate,
    materialize_stage_edit_visual_candidates,
)
from .stage_edit_repair_loop import (
    StageEditRepairConfig,
    StageEditRepairIssue,
    StageEditRepairReview,
    repair_stage_edit_candidates,
)
from .stage_edit_windowed_beam_redecode import (
    STAGE_EDIT_WINDOWED_BEAM_CONTRACT,
    StageEditWindowedBeamRedecoder,
)
from .tracking import (
    AnchoredTrackerConfig,
    MemberAnchor,
    MemberTrackingResult,
    TrackingState,
    track_anchored_member,
)
from .timeline_review import TimelineReview, review_timeline
from .timewarp import (
    TimeWarpAnchor,
    TimeWarpConfig,
    TimeWarpControlPoint,
    TimeWarpMap,
    TimeWarpResidualDiagnostics,
    estimate_local_timewarp,
)
from .timeline_interchange import write_timeline_interchange_bundle
from .vision import (
    SceneDetectionConfig,
    detect_scene_changes_in_video_cached,
    iter_video_frames,
)
from .visual_candidate_materialization import (
    MaterializedVisualTimelineCandidate,
    materialize_visual_timeline_candidate_record,
    materialize_visual_timeline_candidates,
)
from .visual_score import (
    VisualTimelineCandidateSet,
    VisualTimelinePlan,
    build_visual_timeline,
    build_visual_timeline_candidate_set,
)
from .visual_boundary_repair import (
    VisualBoundaryRepairPlan,
    repair_visual_timeline_boundaries,
)


ANALYSIS_SAMPLE_RATE = min(DEFAULT_ANALYSIS_SAMPLE_RATE, 4_000)
REQUEST_SOURCE_INDEX_DOMAIN = "request_source_index"
ELIGIBLE_SOURCE_INDEX_DOMAIN = "eligible_source_index"
OFFICIAL_TARGET_FANCAM_DIRECT_FULL_FRAME_POLICY = (
    "official_target_fancam_direct_full_frame"
)
AUTOMATIC_MEMBER_PORTRAIT_FILL_CONTRACT = (
    "automatic_official_member_lossless_portrait_fill_v1"
)
_MEMBER_PRODUCTION_WIDTH = 1080
_MEMBER_PRODUCTION_HEIGHT = 1920
AUTOMATIC_MEMBER_ALIGNMENT_PEER_CONSENSUS_CONTRACT = (
    "automatic_member_borderline_multi_peer_alignment_consensus_v1"
)
_AUTOMATIC_MEMBER_ALIGNMENT_PEER_CONSENSUS_SEAL = object()
_MEMBER_PEER_CONSENSUS_DIRECT_SCORE_FLOOR = 0.18
_MEMBER_PEER_CONSENSUS_DIRECT_MARGIN_FLOOR = 0.025
_MEMBER_PEER_CONSENSUS_DIRECT_OVERLAP_FLOOR = 0.80
_MEMBER_PEER_CONSENSUS_PEER_SCORE_FLOOR = 0.20
_MEMBER_PEER_CONSENSUS_PEER_MARGIN_FLOOR = 0.05
_MEMBER_PEER_CONSENSUS_PEER_OVERLAP_FLOOR = 0.80
_MEMBER_PEER_CONSENSUS_OFFSET_TOLERANCE_SECONDS = 1.5 / 29.97
AUTOMATIC_STAGE_EDIT_POST_RENDER_QA_CONTRACT = (
    "automatic_ive_group_stage_edit_mandatory_post_render_qa_v2"
)
DEFAULT_IVE_PROFESSIONAL_MONTAGE_PROFILE_PATH = Path(
    "/Volumes/CrossCutMedia/CrossCut-Training/release/profiles/"
    "ive-professional-montage-grammar-v2.json"
)
DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_PATH = Path(
    "/Volumes/CrossCutMedia/CrossCut-Training/release/profiles/"
    "ive-professional-motion-cut-profile-v1.json"
)
DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_FILE_SHA256 = (
    "005a21f79ab1e0b5de76d3299ca2f61b54eb460b88cfb2f88dd392d93e9419b6"
)
_DIRECT_MEMBER_TARGET_RUN_SECONDS = 3.9
_DIRECT_MEMBER_MIN_SWITCHES_PER_MINUTE = 15.0
_DIRECT_MEMBER_MAX_SWITCHES_PER_MINUTE = 21.0
_DIRECT_MEMBER_INITIAL_MAX_SWITCHES_PER_MINUTE = 19.5
_DIRECT_MEMBER_MIN_MEDIAN_RUN_SECONDS = 3.5
_DIRECT_MEMBER_MAX_MEDIAN_RUN_SECONDS = 4.2
_DIRECT_MEMBER_MAX_RUN_SECONDS = 7.2
_DIRECT_MEMBER_ROLE_AWARE_SOURCE_USAGE_POLICIES = frozenset(
    {
        "heterogeneous_role_aware_safe_hold_v1",
        "heterogeneous_role_aware_cinematic_v1",
    }
)
_DIRECT_MEMBER_HIGHLIGHT_TARGET_RUN_SECONDS = 2.25
_DIRECT_MEMBER_HIGHLIGHT_MINIMUM_RUN_SECONDS = 1.8
_DIRECT_MEMBER_HIGHLIGHT_DENSE_SWITCHES_PER_MINUTE = 19.3
_DIRECT_MEMBER_HIGHLIGHT_MINIMUM_SWITCHES_PER_MINUTE = 18.0
_DIRECT_MEMBER_HIGHLIGHT_MINIMUM_DENSITY_LIFT_PER_MINUTE = 2.0
_DIRECT_MEMBER_HIGHLIGHT_MINIMUM_MOVING_CUT_FRACTION = 0.30
_DIRECT_MEMBER_HIGHLIGHT_MINIMUM_MOVING_CUTS_PER_MINUTE = 5.0
_DIRECT_MEMBER_HIGHLIGHT_MINIMUM_AUDIT_DURATION_SECONDS = 8.0
_DIRECT_MEMBER_MAXIMUM_COMPENSATED_HIGHLIGHT_HOLD_COUNT = 4
_DIRECT_MEMBER_MAX_PARTITION_PROFILE_COUNT = 64
_DIRECT_MEMBER_MAX_PARTITION_PROFILE_UPPER_BOUND = 4_163
_DIRECT_MEMBER_MAX_BOTTLENECK_BOUNDARY_COUNT = 2
_DIRECT_MEMBER_MAX_SHIFTED_BOUNDARY_COUNT = 7
_DIRECT_MEMBER_MAX_VISUAL_FRONTIER_SPLITS_PER_PROFILE = 4
_DIRECT_MEMBER_COUNTERFACTUAL_SECONDARY_SPLIT_TRIGGER_COUNT = 1
_DIRECT_MEMBER_MAX_COUNTERFACTUAL_SECONDARY_SPLIT_PROFILE_COUNT = 10
# Audition only the strongest proof-progress representatives.  The terminal
# selector independently caps holds by fraction, count, consecutive use, run
# length, highlight priority, and verified effect spacing.
_DIRECT_MEMBER_MAX_SAFE_HOLD_PROFILE_COUNT = 12
# The strict all-cut search is deliberately wider than the initial queue, but
# it must not chase the combinatorial static upper bound before trying the
# audited no-cut/effect bridge.  Two complete terminal cohorts expose
# materially different frontiers while retaining a deterministic wall against
# multi-hour expansion.  Every candidate still runs the unrelaxed selector;
# this cap only decides when the independently constrained audition begins.
_DIRECT_MEMBER_MAX_STRICT_TERMINAL_AUDITION_COUNT = (
    2 * _DIRECT_MEMBER_MAX_SAFE_HOLD_PROFILE_COUNT
)
_DIRECT_MEMBER_PARTITION_WARM_START_FILENAME = (
    "direct-member-partition-candidate-cache-v1.json"
)
_DIRECT_MEMBER_PARTITION_WARM_START_CONTRACT = (
    "exact_acquisition_manifest_and_base_partition_bound_full_recheck_"
    "candidate_cache_v1"
)
_DIRECT_MEMBER_PARTITION_WARM_START_MAX_BYTES = 256 * 1024
_DIRECT_MEMBER_PARTITION_WARM_START_MAX_ENTRY_COUNT = 32
_DIRECT_MEMBER_PARTITION_WARM_START_MAX_MATCH_COUNT = 4
_DIRECT_MEMBER_VISUAL_SPLIT_EXACT_LIFELINE_RETRY_STATE_LIMIT = 1_000_000
_DIRECT_MEMBER_MAX_EXTRA_QUOTA_PROFILE_COUNT = 10
_DIRECT_MEMBER_MAX_EXTRA_QUOTA_ASSIGNMENT_COUNT = 20
_DIRECT_MEMBER_FAILURE_REPORT_MAX_BYTES = 64 * 1024 * 1024
_DIRECT_MEMBER_FAILURE_DIAGNOSTIC_FINGERPRINT_CONTRACT = (
    "canonical_json_sha256_v1"
)
_DIRECT_MEMBER_FAILURE_REPORT_STORAGE_CONTRACT = (
    "bounded_direct_member_failure_report_v1"
)
@dataclass(frozen=True, slots=True)
class AnalysisArtifact:
    manifest_path: Path
    manifest: dict[str, Any]
    output_paths: tuple[Path, ...]


def _alignment_result_sha256(alignment: AlignmentResult) -> str:
    encoded = json.dumps(
        alignment.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class VerifiedAutomaticMemberAlignmentPeerConsensus(Mapping[str, Any]):
    """Immutable, process-local authority for a member alignment rescue.

    Serialized dictionaries are diagnostics only.  Eligibility accepts this
    sealed result, bound to the exact ordered ``AlignmentResult``, so a caller
    cannot promote a weak source by replaying or forging manifest JSON.
    """

    _payload_json: str
    _alignment_sha256: str
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _AUTOMATIC_MEMBER_ALIGNMENT_PEER_CONSENSUS_SEAL:
            raise TypeError(
                "member alignment peer consensus must come from the verifier"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", self._alignment_sha256):
            raise ValueError("member alignment consensus binding is invalid")
        payload = json.loads(self._payload_json)
        if not isinstance(payload, dict) or payload.get("contract") != (
            AUTOMATIC_MEMBER_ALIGNMENT_PEER_CONSENSUS_CONTRACT
        ):
            raise ValueError("member alignment consensus payload is invalid")

    def _payload(self) -> dict[str, Any]:
        payload = json.loads(self._payload_json)
        if not isinstance(payload, dict):  # pragma: no cover - constructor guard
            raise RuntimeError("sealed member alignment consensus was corrupted")
        return payload

    def __getitem__(self, key: str) -> Any:
        return self._payload()[key]

    def __iter__(self):
        return iter(self._payload())

    def __len__(self) -> int:
        return len(self._payload())

    def verified_for(self, alignment: AlignmentResult) -> bool:
        return self._seal is _AUTOMATIC_MEMBER_ALIGNMENT_PEER_CONSENSUS_SEAL and (
            self._alignment_sha256 == _alignment_result_sha256(alignment)
        )

    def to_dict(self) -> dict[str, Any]:
        return self._payload()


def _write_analysis_manifest_atomically(
    path: Path,
    manifest: Mapping[str, Any],
) -> Path:
    """Replace one analysis manifest without exposing a partial JSON state."""

    destination = Path(path)
    encoded = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
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
    persisted, _fingerprint = _read_stable_regular_file(destination)
    if persisted != encoded.encode("utf-8"):
        raise OSError("analysis manifest changed during atomic replacement")
    return destination


def _read_stable_regular_file(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Read bytes from a path-bound regular descriptor without following links."""

    source = Path(path)
    try:
        path_before = source.lstat()
        if not stat.S_ISREG(path_before.st_mode):
            raise ValueError(f"not a regular non-symlink file: {source.name}")
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
                raise ValueError(f"path changed before reading: {source.name}")
            content = stream.read()
            after = os.fstat(stream.fileno())
            path_after = source.lstat()
            resolved_after = source.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"could not read stable file {source.name}: {exc}") from exc
    descriptor_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    final_descriptor_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    final_path_identity = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    if (
        not stat.S_ISREG(path_after.st_mode)
        or descriptor_identity != final_descriptor_identity
        or final_path_identity != final_descriptor_identity
        or resolved_before != resolved_after
        or len(content) != after.st_size
    ):
        raise ValueError(f"file changed while being read: {source.name}")
    return (
        content,
        {
            "path": str(resolved_after),
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        },
    )


def _fingerprint_stable_regular_file_streaming(path: Path) -> dict[str, Any]:
    """Hash a potentially large regular file through one stable descriptor."""

    source = Path(path)
    try:
        path_before = source.lstat()
        if not stat.S_ISREG(path_before.st_mode):
            raise ValueError(f"not a regular non-symlink file: {source.name}")
        resolved_before = source.resolve(strict=True)
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(source, flags)
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or (path_before.st_dev, path_before.st_ino)
                != (before.st_dev, before.st_ino)
            ):
                raise ValueError(f"path changed before hashing: {source.name}")
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
            after = os.fstat(stream.fileno())
        path_after = source.lstat()
        resolved_after = source.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"could not hash stable file {source.name}: {exc}") from exc
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
    path_identity = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    if (
        not stat.S_ISREG(path_after.st_mode)
        or before_identity != after_identity
        or after_identity != path_identity
        or resolved_before != resolved_after
        or size != after.st_size
    ):
        raise ValueError(f"file changed while hashing: {source.name}")
    return {
        "path": str(resolved_after),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }


def _persist_qa_input_analysis_snapshot(
    analysis_path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Preserve the exact pre-promotion analysis bytes for later audit."""

    source = Path(analysis_path)
    content, source_fingerprint = _read_stable_regular_file(source)
    digest = str(source_fingerprint["sha256"])
    if digest != expected_sha256:
        raise ValueError("analysis content changed before QA snapshot persistence")
    suffix = "-analysis.json"
    if not source.name.endswith(suffix):
        raise ValueError("QA input analysis path has an unsupported name")
    base_name = source.name[: -len(suffix)]
    destination = source.with_name(
        f"{base_name}-qa-input-analysis-{digest[:16]}.json"
    )
    if destination.exists():
        existing, _existing_fingerprint = _read_stable_regular_file(
            destination
        )
        if existing != content:
            raise FileExistsError("QA input analysis snapshot collision")
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
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
    persisted, persisted_fingerprint = _read_stable_regular_file(destination)
    if persisted != content or persisted_fingerprint["sha256"] != digest:
        raise ValueError("QA input analysis snapshot changed before promotion")
    return persisted_fingerprint


def _atomically_materialize_verified_snapshot_file(
    source: Path,
    destination: Path,
    *,
    expected_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    """Move one verified snapshot inode onto its canonical delivery path."""

    source_path = Path(source)
    destination_path = Path(destination)
    before = source_path.lstat()
    expected_size = expected_fingerprint.get("size_bytes")
    expected_sha256 = expected_fingerprint.get("sha256")
    if (
        not stat.S_ISREG(before.st_mode)
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or before.st_size != expected_size
        or not isinstance(expected_sha256, str)
    ):
        raise RuntimeError(
            f"verified QA snapshot is no longer promotable: {source_path.name}"
        )
    source_identity = (before.st_dev, before.st_ino, before.st_size)
    os.replace(source_path, destination_path)
    after = destination_path.lstat()
    destination_identity = (after.st_dev, after.st_ino, after.st_size)
    if not stat.S_ISREG(after.st_mode) or destination_identity != source_identity:
        raise RuntimeError(
            f"QA snapshot inode promotion lost authority: {destination_path.name}"
        )
    return {
        "path": str(destination_path.resolve()),
        "size_bytes": expected_size,
        "sha256": expected_sha256,
        "materialization": "same_filesystem_atomic_inode_replace",
        "snapshot_device": before.st_dev,
        "snapshot_inode": before.st_ino,
        "canonical_device": after.st_dev,
        "canonical_inode": after.st_ino,
        "inode_identity_preserved": True,
    }


def _enforce_automatic_stage_edit_post_render_qa(
    *,
    registered_stage_edit_bundle: RegisteredStageEditProductionBundle | None,
    manifest: dict[str, Any],
    manifest_path: Path,
    reporter: EventReporter,
    automatic_ive_group_delivery: bool | None = None,
    rendered_montage_config: RenderedMontageGrammarConfig | None = None,
) -> Path | None:
    """Require PASS from final-media QA for every automatic IVE GROUP delivery.

    The evaluator consumes the already persisted timestamp-bound analysis,
    render-plan, candidate-selection, and final MP4.  Automatic product media
    remains an unexposed ``pending_qa`` artifact until this function atomically
    promotes a passing report to ``delivery``.  A failed report is persisted
    as ``rejected`` and can never be restored by the app as a completed result.
    ``None`` retains the old direct-call inference for compatibility; the
    production caller always supplies the explicit request-scoped decision.
    """

    required = (
        registered_stage_edit_bundle is not None
        if automatic_ive_group_delivery is None
        else automatic_ive_group_delivery
    )
    if not isinstance(required, bool):
        raise TypeError("automatic_ive_group_delivery must resolve to boolean")
    if not required:
        return None
    if required:
        request_payload = manifest.get("request")
        if not (
            isinstance(request_payload, Mapping)
            and str(request_payload.get("group", "")).strip().upper()
            == "IVE"
            and str(request_payload.get("edit_mode", "")).strip().lower()
            == "group"
            and str(request_payload.get("source_mode", "")).strip().lower()
            == "automatic"
        ):
            raise RuntimeError(
                "mandatory automatic IVE GROUP post-render QA lost its exact "
                "request scope"
            )
        exposure = manifest.get("app_result_exposure")
        render_payload = manifest.get("render")
        if not (
            manifest.get("artifact_role") == "pending_qa"
            and isinstance(exposure, Mapping)
            and exposure.get("contract") == APP_RESULT_EXPOSURE_CONTRACT
            and exposure.get("video_output_exposed") is False
            and exposure.get("delivery_completion_message_allowed") is False
            and isinstance(render_payload, Mapping)
            and render_payload.get("artifact_role") == "pending_qa"
        ):
            raise RuntimeError(
                "mandatory automatic IVE GROUP post-render QA requires an "
                "unexposed pending_qa artifact"
            )
    if rendered_montage_config is not None and not isinstance(
        rendered_montage_config, RenderedMontageGrammarConfig
    ):
        raise TypeError(
            "rendered_montage_config must be RenderedMontageGrammarConfig"
        )

    declared_artifact_set = PostRenderArtifactSet.from_analysis_path(
        manifest_path
    )
    evaluation_snapshot = materialize_post_render_qa_snapshot(
        declared_artifact_set
    )
    try:
        return _complete_automatic_stage_edit_post_render_qa(
            registered_stage_edit_bundle=registered_stage_edit_bundle,
            manifest=manifest,
            manifest_path=manifest_path,
            reporter=reporter,
            rendered_montage_config=rendered_montage_config,
            evaluation_snapshot=evaluation_snapshot,
        )
    finally:
        cleanup_post_render_qa_snapshot(evaluation_snapshot)


def _complete_automatic_stage_edit_post_render_qa(
    *,
    registered_stage_edit_bundle: RegisteredStageEditProductionBundle | None,
    manifest: dict[str, Any],
    manifest_path: Path,
    reporter: EventReporter,
    rendered_montage_config: RenderedMontageGrammarConfig | None,
    evaluation_snapshot: PostRenderQAEvaluationSnapshot,
) -> Path:
    """Evaluate, bind, and materialize one already-frozen QA artifact set."""

    declared_artifact_set = evaluation_snapshot.declared_artifact_set
    artifact_set = evaluation_snapshot.artifact_set
    reporter.status(
        "IVE GROUP 최종 MP4를 전체 디코딩하고 "
        "전문 후처리 QA를 실행하는 중…"
    )
    qa_arguments: dict[str, Any] = {
        "trusted_registered_stage_edit": (
            True if registered_stage_edit_bundle is not None else None
        )
    }
    if rendered_montage_config is not None:
        qa_arguments["config"] = PostRenderQAConfig(
            rendered_montage_grammar=rendered_montage_config
        )
    report, report_path = evaluate_and_write_post_render_qa(
        artifact_set,
        output_path=artifact_set.default_report_path,
        declared_artifact_set=declared_artifact_set,
        **qa_arguments,
    )
    report_payload = report.to_dict()
    reported_input_fingerprints = report_payload.get(
        "input_artifact_fingerprints"
    )
    try:
        expected_input_fingerprints = (
            normalize_post_render_artifact_fingerprints(
                reported_input_fingerprints,
                declared_artifact_set,
            )
            if isinstance(reported_input_fingerprints, Mapping)
            else {}
        )
    except ValueError:
        expected_input_fingerprints = {}
    try:
        current_input_fingerprints = (
            normalize_post_render_artifact_fingerprints(
                fingerprint_post_render_artifacts(artifact_set),
                declared_artifact_set,
            )
        )
    except ValueError:
        current_input_fingerprints = {}
    input_content_matches = bool(
        expected_input_fingerprints.get("contract")
        == INPUT_ARTIFACT_FINGERPRINT_CONTRACT
        and expected_input_fingerprints == current_input_fingerprints
    )
    persisted_analysis_payload: dict[str, Any] | None = None
    try:
        raw_persisted_analysis = json.loads(
            artifact_set.analysis_path.read_text(encoding="utf-8")
        )
        if isinstance(raw_persisted_analysis, dict):
            persisted_analysis_payload = raw_persisted_analysis
    except (OSError, UnicodeError, json.JSONDecodeError):
        persisted_analysis_payload = None
    in_memory_manifest_matches_evaluated_analysis = bool(
        persisted_analysis_payload is not None
        and persisted_analysis_payload == manifest
    )
    qa_input_analysis_snapshot: dict[str, Any] = {}
    expected_analysis_fingerprint = (
        expected_input_fingerprints.get("analysis")
    )
    expected_analysis_sha256 = (
        expected_analysis_fingerprint.get("sha256")
        if isinstance(expected_analysis_fingerprint, Mapping)
        else None
    )
    if (
        input_content_matches
        and in_memory_manifest_matches_evaluated_analysis
        and isinstance(expected_analysis_sha256, str)
    ):
        try:
            qa_input_analysis_snapshot = _persist_qa_input_analysis_snapshot(
                artifact_set.analysis_path,
                expected_sha256=expected_analysis_sha256,
            )
        except (OSError, ValueError, FileExistsError):
            qa_input_analysis_snapshot = {}
    input_binding_matches = bool(
        input_content_matches
        and in_memory_manifest_matches_evaluated_analysis
        and qa_input_analysis_snapshot.get("sha256")
        == expected_analysis_sha256
    )
    expected_report_path = artifact_set.default_report_path
    candidate_report_path = Path(report_path)
    report_path_contract_matches = False
    try:
        report_path_contract_matches = bool(
            candidate_report_path.name == expected_report_path.name
            and candidate_report_path.parent.resolve(strict=True)
            == expected_report_path.parent.resolve(strict=True)
            and not candidate_report_path.is_symlink()
        )
    except OSError:
        report_path_contract_matches = False
    report_content_fingerprint: dict[str, Any] = {}
    persisted_report_matches = False
    try:
        report_bytes, report_content_fingerprint = _read_stable_regular_file(
            candidate_report_path
        )
        persisted_report = json.loads(report_bytes.decode("utf-8"))
        persisted_report_matches = bool(
            report_path_contract_matches
            and isinstance(persisted_report, Mapping)
            and dict(persisted_report) == report_payload
        )
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError):
        persisted_report_matches = False
    snapshot_stable_through_promotion = False
    if qa_input_analysis_snapshot:
        try:
            _snapshot_bytes, final_snapshot_fingerprint = (
                _read_stable_regular_file(
                    Path(str(qa_input_analysis_snapshot["path"]))
                )
            )
            snapshot_stable_through_promotion = bool(
                final_snapshot_fingerprint == qa_input_analysis_snapshot
            )
        except (KeyError, TypeError, ValueError):
            snapshot_stable_through_promotion = False
    report_stable_through_promotion = False
    if persisted_report_matches:
        try:
            final_report_bytes, final_report_fingerprint = (
                _read_stable_regular_file(candidate_report_path)
            )
            final_report_payload = json.loads(
                final_report_bytes.decode("utf-8")
            )
            report_stable_through_promotion = bool(
                final_report_fingerprint == report_content_fingerprint
                and isinstance(final_report_payload, Mapping)
                and dict(final_report_payload) == report_payload
            )
        except (OSError, ValueError, UnicodeError, json.JSONDecodeError):
            report_stable_through_promotion = False
    pre_promotion_input_fingerprints: dict[str, Any] = {}
    try:
        pre_promotion_input_fingerprints = (
            normalize_post_render_artifact_fingerprints(
                fingerprint_post_render_artifacts(artifact_set),
                declared_artifact_set,
            )
        )
    except ValueError:
        pre_promotion_input_fingerprints = {}
    inputs_stable_through_promotion = bool(
        input_content_matches
        and pre_promotion_input_fingerprints == current_input_fingerprints
    )
    promotion_binding_passed = bool(
        input_binding_matches
        and persisted_report_matches
        and inputs_stable_through_promotion
        and snapshot_stable_through_promotion
        and report_stable_through_promotion
    )
    effective_passed = bool(report.passed and promotion_binding_passed)
    delivery_gate_verdict = "PASS" if effective_passed else "FAIL"
    delivery_materialization: dict[str, Any] = {
        "contract": "post_render_qa_verified_snapshot_materialization_v1",
        "performed": False,
        "input_artifacts": {},
        "report": {},
        "analysis_snapshot": {},
    }
    if promotion_binding_passed:
        promoted_inputs: dict[str, Any] = {}
        for key, source, destination in (
            (
                "render_plan",
                artifact_set.render_plan_path,
                declared_artifact_set.render_plan_path,
            ),
            (
                "selection",
                artifact_set.selection_path,
                declared_artifact_set.selection_path,
            ),
            (
                "video",
                artifact_set.video_path,
                declared_artifact_set.video_path,
            ),
        ):
            promoted_inputs[key] = (
                _atomically_materialize_verified_snapshot_file(
                    source,
                    destination,
                    expected_fingerprint=pre_promotion_input_fingerprints[key],
                )
            )
        delivery_materialization["performed"] = True
        delivery_materialization["input_artifacts"] = promoted_inputs
    if report_stable_through_promotion:
        delivery_materialization["report"] = (
            _atomically_materialize_verified_snapshot_file(
                candidate_report_path,
                declared_artifact_set.default_report_path,
                expected_fingerprint=report_content_fingerprint,
            )
        )
    if snapshot_stable_through_promotion:
        snapshot_source = Path(str(qa_input_analysis_snapshot["path"]))
        snapshot_destination = declared_artifact_set.output_directory / (
            snapshot_source.name
        )
        delivery_materialization["analysis_snapshot"] = (
            _atomically_materialize_verified_snapshot_file(
                snapshot_source,
                snapshot_destination,
                expected_fingerprint=qa_input_analysis_snapshot,
            )
        )
        qa_input_analysis_snapshot = {
            "path": delivery_materialization["analysis_snapshot"]["path"],
            "size_bytes": delivery_materialization["analysis_snapshot"][
                "size_bytes"
            ],
            "sha256": delivery_materialization["analysis_snapshot"]["sha256"],
        }
    promotion_binding = {
        "contract": "post_render_qa_delivery_promotion_binding_v1",
        "input_artifact_contract": INPUT_ARTIFACT_FINGERPRINT_CONTRACT,
        "evaluated_input_artifacts": (
            dict(expected_input_fingerprints)
            if isinstance(expected_input_fingerprints, Mapping)
            else {}
        ),
        "current_inputs_match_evaluated_content": input_content_matches,
        "inputs_stable_through_promotion": inputs_stable_through_promotion,
        "pre_promotion_input_artifacts": pre_promotion_input_fingerprints,
        "in_memory_manifest_matches_evaluated_analysis": (
            in_memory_manifest_matches_evaluated_analysis
        ),
        "pre_promotion_analysis_snapshot": qa_input_analysis_snapshot,
        "persisted_report_matches_evaluator_payload": (
            persisted_report_matches
        ),
        "report_path_matches_bound_default": report_path_contract_matches,
        "report_stable_through_promotion": report_stable_through_promotion,
        "report_content_fingerprint": report_content_fingerprint,
        "evaluation_snapshot": evaluation_snapshot.to_dict(),
        "delivery_materialization": delivery_materialization,
        "passed": promotion_binding_passed,
    }
    promoted_report = delivery_materialization.get("report")
    report_audit_path = str(
        promoted_report.get("path")
        if isinstance(promoted_report, Mapping) and promoted_report.get("path")
        else report_content_fingerprint.get(
            "path",
            candidate_report_path.absolute(),
        )
    )
    provenance = {
        "contract": AUTOMATIC_STAGE_EDIT_POST_RENDER_QA_CONTRACT,
        "required": True,
        "activation": (
            "automatic_registered_production_bundle"
            if registered_stage_edit_bundle is not None
            else "automatic_request_scoped_ive_group_delivery"
        ),
        "ive_product_floor_required_independent_of_activation": True,
        "success_requires_pass": True,
        "report_path": report_audit_path,
        "verdict": report.verdict.value,
        "passed": effective_passed,
        "delivery_gate_verdict": delivery_gate_verdict,
        "report_verdict": report.verdict.value,
        "report_passed": report.passed,
        "analysis_basis": report.analysis_basis,
        "issue_count": report_payload.get("issue_count"),
        "fail_issue_count": report_payload.get("fail_issue_count"),
        "conditional_issue_count": report_payload.get(
            "conditional_issue_count"
        ),
        "failure_artifact_policy": {
            "final_mp4_preserved": True,
            "qa_report_preserved": True,
            "analysis_manifest_preserved": True,
            "app_result_exposure_blocked": not effective_passed,
        },
        "promotion_binding": promotion_binding,
    }
    # No delivery manifest is published until every unpromoted private clone,
    # including a fallback full video copy, has been removed. The outer
    # ``finally`` repeats this idempotently for evaluator and promotion errors.
    cleanup_post_render_qa_snapshot(evaluation_snapshot)
    final_role = "delivery" if effective_passed else "rejected"
    exposure = {
        "contract": APP_RESULT_EXPOSURE_CONTRACT,
        "video_output_exposed": effective_passed,
        "delivery_completion_message_allowed": effective_passed,
    }
    if (
        not in_memory_manifest_matches_evaluated_analysis
        and persisted_analysis_payload is not None
    ):
        manifest.clear()
        manifest.update(deepcopy(persisted_analysis_payload))
    manifest["artifact_role"] = final_role
    manifest["app_result_exposure"] = dict(exposure)
    manifest["post_render_qa"] = dict(provenance)
    render_payload = manifest.get("render")
    if isinstance(render_payload, dict):
        render_payload["artifact_role"] = final_role
        render_payload["app_result_exposure"] = dict(exposure)
        render_payload["post_render_qa"] = dict(provenance)
    generation_payload = manifest.get("stage_edit_generation")
    if isinstance(generation_payload, dict):
        generation_payload["post_render_qa"] = dict(provenance)
    _write_analysis_manifest_atomically(manifest_path, manifest)
    reporter.log(
        "IVE GROUP 최종 후처리 QA "
        f"{report.verdict.value} · 보고서 {report_audit_path}"
    )
    if not effective_passed:
        failure_reason = (
            f"post-render QA verdict {report.verdict.value}"
            if not report.passed
            else "post-render QA artifact content binding failure"
        )
        raise RuntimeError(
            "automatic IVE GROUP final render blocked by mandatory "
            f"{failure_reason}; rejected media "
            "is retained only for diagnostics and is not exposed as a "
            f"completed result; QA report preserved at {report_audit_path}"
        )
    return Path(report_audit_path)


def _enforce_automatic_member_post_render_qa(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    reporter: EventReporter,
) -> Path | None:
    """Promote an automatic direct-member MP4 only after final pixel QA.

    Activation is resolved from the persisted request plus
    ``render.direct_full_frame``.  The caller cannot opt a matching product out
    of this gate, and no result path is exposed while its role is
    ``pending_qa``.
    """

    if not automatic_member_post_render_qa_required(manifest):
        return None
    request_payload = manifest.get("request")
    render_payload = manifest.get("render")
    exposure = manifest.get("app_result_exposure")
    if not (
        isinstance(request_payload, Mapping)
        and isinstance(render_payload, Mapping)
        and render_payload.get("direct_full_frame") is True
        and manifest.get("artifact_role") == "pending_qa"
        and render_payload.get("artifact_role") == "pending_qa"
        and isinstance(exposure, Mapping)
        and exposure.get("contract")
        == AUTOMATIC_IVE_MEMBER_APP_RESULT_EXPOSURE_CONTRACT
        and exposure.get("video_output_exposed") is False
        and exposure.get("delivery_completion_message_allowed") is False
    ):
        raise RuntimeError(
            "mandatory automatic IVE MEMBER post-render QA requires an "
            "unexposed pending_qa direct-full-frame artifact"
        )
    try:
        persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "automatic IVE MEMBER pending analysis manifest is unreadable"
        ) from exc
    if not isinstance(persisted, dict) or persisted != manifest:
        raise RuntimeError(
            "automatic IVE MEMBER pending analysis differs from persisted QA input"
        )

    artifacts = AutomaticMemberPostRenderArtifactSet.from_manifest(
        manifest_path, manifest
    )
    reporter.status(
        "IVE 멤버 최종 MP4 전체 디코드·화질·소스별 픽셀 출처·"
        "컷 위치·포즈·동작 연결성을 검증하는 중…"
    )
    report, report_path = (
        evaluate_and_write_automatic_member_post_render_qa(artifacts)
    )
    report_payload = report.to_dict()
    try:
        pre_exposure_fingerprints = fingerprint_automatic_member_artifacts(
            artifacts
        )
    except ValueError:
        pre_exposure_fingerprints = {}
    content_binding_passed = bool(
        report.input_artifact_fingerprints
        and report.final_artifact_fingerprints
        and report.input_artifact_fingerprints
        == report.final_artifact_fingerprints
        == pre_exposure_fingerprints
    )
    effective_passed = bool(report.passed and content_binding_passed)
    final_role = "delivery" if effective_passed else "rejected"
    final_exposure = {
        "contract": AUTOMATIC_IVE_MEMBER_APP_RESULT_EXPOSURE_CONTRACT,
        "video_output_exposed": effective_passed,
        "delivery_completion_message_allowed": effective_passed,
    }
    provenance = {
        "contract": AUTOMATIC_IVE_MEMBER_POST_RENDER_QA_CONTRACT,
        "required": True,
        "activation": (
            "persisted_request_automatic_ive_member_and_render_direct_full_frame"
        ),
        "success_requires_pass": True,
        "report_path": str(Path(report_path).resolve()),
        "verdict": report.verdict,
        "report_passed": report.passed,
        "content_binding_passed": content_binding_passed,
        "passed": effective_passed,
        "issue_count": report_payload.get("issue_count"),
        "input_artifact_fingerprints": dict(
            report.input_artifact_fingerprints
        ),
        "pre_exposure_artifact_fingerprints": pre_exposure_fingerprints,
        "render_receipt": dict(report.render_receipt),
        "failure_artifact_policy": {
            "final_mp4_preserved": True,
            "qa_report_preserved": True,
            "analysis_manifest_preserved": True,
            "app_result_exposure_blocked": not effective_passed,
        },
    }
    manifest["artifact_role"] = final_role
    manifest["app_result_exposure"] = dict(final_exposure)
    manifest["post_render_qa"] = dict(provenance)
    mutable_render = manifest.get("render")
    if isinstance(mutable_render, dict):
        mutable_render["artifact_role"] = final_role
        mutable_render["app_result_exposure"] = dict(final_exposure)
        mutable_render["post_render_qa"] = dict(provenance)
    _write_analysis_manifest_atomically(manifest_path, manifest)
    reporter.log(
        "IVE MEMBER 최종 후처리 QA "
        f"{report.verdict} · 보고서 {Path(report_path).resolve()}"
    )
    if not effective_passed:
        reason = (
            f"verdict {report.verdict}"
            if not report.passed
            else "artifact content binding failure"
        )
        raise RuntimeError(
            "automatic IVE MEMBER final render blocked by mandatory "
            f"post-render QA {reason}; rejected media is retained only for "
            "diagnostics and is not exposed as a completed result; QA report "
            f"preserved at {Path(report_path).resolve()}"
        )
    return Path(report_path).resolve()


def _materialize_ranked_visual_proxy_candidates(
    candidate_set: VisualTimelineCandidateSet,
    *,
    request: AnalysisRequest,
    rhythm_analysis: Any,
    section_plan: SectionPlan | None,
    reporter: EventReporter,
) -> tuple[
    tuple[MaterializedVisualTimelineCandidate, ...],
    StageEditCandidateRerankResult | None,
]:
    """Keep the legacy order unless explicit production inference is requested."""

    if request.stage_edit_checkpoint_directory is None:
        return (
            materialize_visual_timeline_candidates(
                candidate_set,
                maximum_candidates=8,
                passed_only=True,
            ),
            None,
        )
    report = request.stage_edit_production_training_report
    if report is None:  # AnalysisRequest enforces the pair; retain fail-close here.
        raise RuntimeError(
            "StageEdit checkpoint activation requires its exact production report"
        )
    reporter.status(
        "검증된 StageEdit Transformer로 시각 후보 순서를 평가하는 중…"
    )
    result = rerank_visual_timeline_candidates_with_stage_edit(
        candidate_set,
        checkpoint_directory=request.stage_edit_checkpoint_directory,
        production_training_report=report,
        rhythm=rhythm_analysis,
        section_plan=section_plan,
    )
    ordered_records = result.ordered_records(candidate_set)
    materialized_values: list[MaterializedVisualTimelineCandidate] = []
    for record in ordered_records:
        if not record.review.passed:
            continue
        materialized_values.append(
            materialize_visual_timeline_candidate_record(candidate_set, record)
        )
        if len(materialized_values) >= 8:
            break
    materialized = tuple(materialized_values)
    reporter.log(
        "StageEdit Transformer가 보존된 시각 후보 "
        f"{len(ordered_records)}개의 1차 렌더 순서를 재평가"
    )
    return materialized, result


def _rendered_candidate_selection_index_domains() -> dict[str, str]:
    """Describe nested source indices in the standalone candidate audit."""

    return {
        "visual_candidate_set.candidates[].source_sequence[]": (
            ELIGIBLE_SOURCE_INDEX_DOMAIN
        ),
        "complete_candidates[].visual_timeline.choices[].source_index": (
            ELIGIBLE_SOURCE_INDEX_DOMAIN
        ),
        "complete_candidates[].boundary_repairs.repairs[].outgoing_source_index": (
            ELIGIBLE_SOURCE_INDEX_DOMAIN
        ),
        "complete_candidates[].boundary_repairs.repairs[].incoming_source_index": (
            ELIGIBLE_SOURCE_INDEX_DOMAIN
        ),
        "complete_candidates[].face_match.segments[].source_index": (
            ELIGIBLE_SOURCE_INDEX_DOMAIN
        ),
        "complete_candidates[].boundary_affine.decisions[].outgoing_source_index": (
            ELIGIBLE_SOURCE_INDEX_DOMAIN
        ),
        "complete_candidates[].boundary_affine.decisions[].incoming_source_index": (
            ELIGIBLE_SOURCE_INDEX_DOMAIN
        ),
        "complete_candidates[].editorial_motion.segments[].source_index": (
            ELIGIBLE_SOURCE_INDEX_DOMAIN
        ),
        (
            "complete_candidates[].editorial_motion.boundary_subject_alignment"
            ".decisions[].outgoing_source_index"
        ): ELIGIBLE_SOURCE_INDEX_DOMAIN,
        (
            "complete_candidates[].editorial_motion.boundary_subject_alignment"
            ".decisions[].incoming_source_index"
        ): ELIGIBLE_SOURCE_INDEX_DOMAIN,
        (
            "complete_candidates[].editorial_motion.boundary_subject_alignment"
            ".segments[].source_index"
        ): ELIGIBLE_SOURCE_INDEX_DOMAIN,
        (
            "complete_candidates[].editorial_motion.boundary_subject_alignment"
            ".source_resolution_headroom[].source_index"
        ): ELIGIBLE_SOURCE_INDEX_DOMAIN,
        (
            "complete_candidates[].boundary_look_match.decisions[]"
            ".outgoing_source_index"
        ): REQUEST_SOURCE_INDEX_DOMAIN,
        (
            "complete_candidates[].boundary_look_match.decisions[]"
            ".incoming_source_index"
        ): REQUEST_SOURCE_INDEX_DOMAIN,
        "complete_candidates[].final_render_segments[].request_source_index": (
            REQUEST_SOURCE_INDEX_DOMAIN
        ),
        "candidates[].review.source_quality.per_source[].source_index": (
            REQUEST_SOURCE_INDEX_DOMAIN
        ),
        "candidates[].review.boundaries[].outgoing_source_index": (
            REQUEST_SOURCE_INDEX_DOMAIN
        ),
        "candidates[].review.boundaries[].incoming_source_index": (
            REQUEST_SOURCE_INDEX_DOMAIN
        ),
        "final_selected_review.source_quality.per_source[].source_index": (
            REQUEST_SOURCE_INDEX_DOMAIN
        ),
        "final_selected_review.boundaries[].outgoing_source_index": (
            REQUEST_SOURCE_INDEX_DOMAIN
        ),
        "final_selected_review.boundaries[].incoming_source_index": (
            REQUEST_SOURCE_INDEX_DOMAIN
        ),
        (
            "proxy_triggered_partial_reedit.before_review.source_quality"
            ".per_source[].source_index"
        ): REQUEST_SOURCE_INDEX_DOMAIN,
        (
            "proxy_triggered_partial_reedit.before_review.boundaries[]"
            ".outgoing_source_index"
        ): REQUEST_SOURCE_INDEX_DOMAIN,
        (
            "proxy_triggered_partial_reedit.before_review.boundaries[]"
            ".incoming_source_index"
        ): REQUEST_SOURCE_INDEX_DOMAIN,
        (
            "proxy_triggered_partial_reedit.after_review.source_quality"
            ".per_source[].source_index"
        ): REQUEST_SOURCE_INDEX_DOMAIN,
        (
            "proxy_triggered_partial_reedit.after_review.boundaries[]"
            ".outgoing_source_index"
        ): REQUEST_SOURCE_INDEX_DOMAIN,
        (
            "proxy_triggered_partial_reedit.after_review.boundaries[]"
            ".incoming_source_index"
        ): REQUEST_SOURCE_INDEX_DOMAIN,
        (
            "proxy_triggered_partial_reedit.repair.repairs[]"
            ".outgoing_source_index"
        ): ELIGIBLE_SOURCE_INDEX_DOMAIN,
        (
            "proxy_triggered_partial_reedit.repair.repairs[]"
            ".incoming_source_index"
        ): ELIGIBLE_SOURCE_INDEX_DOMAIN,
    }


def _source_index_mapping(
    sources: Sequence[Path],
    eligible_indices: Sequence[int],
) -> dict[str, Any]:
    """Describe both source-index domains used by the analysis artifacts.

    This pipeline copies ``request.sources`` into each render plan without
    reordering it. The standalone plan names those same numbers by their local
    ``sources[]`` array domain, and ``domain_equivalences`` records that the two
    domains are identical here. Analysis stages that operate only on
    synchronization-safe sources use a compact local index instead. The explicit
    bidirectional mapping prevents that local index from being mistaken for a
    request index after an excluded source creates a gap.
    """

    request_count = len(sources)
    eligible = tuple(eligible_indices)
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= request_count
        for index in eligible
    ):
        raise ValueError("eligible source mapping contains an invalid request index")
    if len(set(eligible)) != len(eligible):
        raise ValueError("eligible source mapping contains duplicate request indices")

    request_to_eligible: list[int | None] = [None] * request_count
    entries: list[dict[str, Any]] = []
    for eligible_index, request_index in enumerate(eligible):
        request_to_eligible[request_index] = eligible_index
        entries.append(
            {
                "eligible_source_index": eligible_index,
                "request_source_index": request_index,
                "source": str(sources[request_index]),
            }
        )

    return {
        "request_index_domain": REQUEST_SOURCE_INDEX_DOMAIN,
        "eligible_index_domain": ELIGIBLE_SOURCE_INDEX_DOMAIN,
        "render_plan_index_domain": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
        "domain_equivalences": {
            RENDER_PLAN_SOURCE_INDEX_DOMAIN: REQUEST_SOURCE_INDEX_DOMAIN,
        },
        "eligible_to_request": list(eligible),
        "request_to_eligible": request_to_eligible,
        "eligible_sources": entries,
        "payload_index_domains": {
            # Paths ending in ``[]`` without a child field describe the
            # zero-based array position.  The serialized media/alignment and
            # audio-offset items do not contain a synthetic ``position`` key.
            "request.sources[]": REQUEST_SOURCE_INDEX_DOMAIN,
            "request.reference_source": REQUEST_SOURCE_INDEX_DOMAIN,
            "request.member_anchors[].source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "media[]": REQUEST_SOURCE_INDEX_DOMAIN,
            "alignment.alignments[]": REQUEST_SOURCE_INDEX_DOMAIN,
            "alignment_eligibility[].source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "alignment_peer_consensus.sources[].source_index": (
                REQUEST_SOURCE_INDEX_DOMAIN
            ),
            (
                "alignment_peer_consensus.sources[].peer_alignments[]"
                ".peer_source_index"
            ): REQUEST_SOURCE_INDEX_DOMAIN,
            "local_timewarps[].source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            (
                "performance_range_trim.windows[].source_correlations[]"
                ".source_index"
            ): REQUEST_SOURCE_INDEX_DOMAIN,
            "scene_analysis[].source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "member_auto_anchors[].source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "member_tracking[].source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "member_anchor_trim.anchors[].source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "member_anchor_trim.limiting_source_indices[]": REQUEST_SOURCE_INDEX_DOMAIN,
            "member_anchor_trim.full_frame_only_source_indices[]": REQUEST_SOURCE_INDEX_DOMAIN,
            "member_direct_full_frame.eligible_request_source_indices[]": REQUEST_SOURCE_INDEX_DOMAIN,
            "member_direct_full_frame.required_used_request_source_indices[]": REQUEST_SOURCE_INDEX_DOMAIN,
            "member_direct_full_frame.segments[].eligible_source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_direct_full_frame.segments[].request_source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "member_direct_full_frame.visual_selection.selected_source_indices[]": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_direct_full_frame.visual_selection.source_usage[].eligible_source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_direct_full_frame.visual_selection.source_usage[].request_source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "member_direct_full_frame.visual_selection.span_content_validity[].sources[].eligible_source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_direct_full_frame.visual_selection.span_content_validity[].sources[].request_source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "member_direct_full_frame.visual_selection.boundaries[].outgoing_eligible_source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_direct_full_frame.visual_selection.boundaries[].incoming_eligible_source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_direct_full_frame.visual_selection.boundaries[].outgoing_request_source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "member_direct_full_frame.visual_selection.boundaries[].incoming_request_source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "member_direct_full_frame.editorial_motion.segments[].source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_direct_full_frame.final_source_usage.used_source_indices[]": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
            "member_direct_full_frame.final_source_usage.sources[].source_index": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
            "group_draft.segments[].source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "group_draft.audio_offsets[]": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "group_draft.sync_scores[]": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "visual_timeline.choices[].source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "source_look_match.evidence[].source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "source_look_match.target_source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "source_derivative_diversity.source_decode_audit[].source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "source_derivative_diversity.pairs[].left_source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "source_derivative_diversity.pairs[].right_source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "source_derivative_diversity.family_ids[]": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            (
                "boundary_look_match.decisions[].outgoing_source_index"
            ): REQUEST_SOURCE_INDEX_DOMAIN,
            (
                "boundary_look_match.decisions[].incoming_source_index"
            ): REQUEST_SOURCE_INDEX_DOMAIN,
            "face_match.segments[].source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "editorial_motion.segments[].source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            (
                "editorial_motion.boundary_subject_alignment.decisions[]"
                ".outgoing_source_index"
            ): ELIGIBLE_SOURCE_INDEX_DOMAIN,
            (
                "editorial_motion.boundary_subject_alignment.decisions[]"
                ".incoming_source_index"
            ): ELIGIBLE_SOURCE_INDEX_DOMAIN,
            (
                "editorial_motion.boundary_subject_alignment.segments[]"
                ".source_index"
            ): ELIGIBLE_SOURCE_INDEX_DOMAIN,
            (
                "editorial_motion.boundary_subject_alignment"
                ".source_resolution_headroom[].source_index"
            ): ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_draft.segments[].source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_draft.audio_offsets[]": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_draft.group_wide_fallback_source": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_draft.selection_diagnostics.safe_source_indices[]": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_draft.selection_diagnostics.quality_eligible_source_indices[]": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_draft.selection_diagnostics.selected_source_indices[]": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_draft.selection_diagnostics.no_safe_candidate_source_indices[]": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_draft.selection_diagnostics.quality_guard_excluded_source_indices[]": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_draft.selection_diagnostics.unused_quality_source_indices[]": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "member_draft.selection_diagnostics.sources[].source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "render.automatic_full_frame_fallbacks[].eligible_source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "render.automatic_full_frame_fallbacks[].request_source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "render.final_segments[].request_source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "render.final_source_usage.used_source_indices[]": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
            "render.final_source_usage.sources[].source_index": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
            "render.cut_local_sync.used_source_indices[]": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
            "render.cut_local_sync.source_decode_audit[].source_index": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
            "render.cut_local_sync.per_boundary[].outgoing_source_index": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
            "render.cut_local_sync.per_boundary[].incoming_source_index": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
            "render.cut_local_sync.per_boundary[].renderer_frame_selection.outgoing.source_index": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
            "render.cut_local_sync.per_boundary[].renderer_frame_selection.incoming.source_index": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
            "render.editorial_motion.segments[].source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "render.source_look_match.evidence[].source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            "render.source_look_match.target_source_index": ELIGIBLE_SOURCE_INDEX_DOMAIN,
            (
                "render.boundary_look_match.decisions[]"
                ".outgoing_source_index"
            ): REQUEST_SOURCE_INDEX_DOMAIN,
            (
                "render.boundary_look_match.decisions[]"
                ".incoming_source_index"
            ): REQUEST_SOURCE_INDEX_DOMAIN,
            "rendered_proxy_review.source_quality.per_source[].source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "rendered_proxy_review.boundaries[].outgoing_source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "rendered_proxy_review.boundaries[].incoming_source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "render.rendered_proxy_review.source_quality.per_source[].source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "render.rendered_proxy_review.boundaries[].outgoing_source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            "render.rendered_proxy_review.boundaries[].incoming_source_index": REQUEST_SOURCE_INDEX_DOMAIN,
            **{
                f"rendered_candidate_selection.{path}": domain
                for path, domain in (
                    _rendered_candidate_selection_index_domains().items()
                )
            },
            "render_plan.sources[]": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
            "render_plan.reference_source": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
            "render_plan.segments[].source_index": RENDER_PLAN_SOURCE_INDEX_DOMAIN,
        },
    }


def _with_index_domain(payload: dict[str, Any], domain: str) -> dict[str, Any]:
    """Attach a machine-readable source-index domain without mutating a payload."""

    if domain not in (REQUEST_SOURCE_INDEX_DOMAIN, ELIGIBLE_SOURCE_INDEX_DOMAIN):
        raise ValueError("unknown source index domain")
    result = dict(payload)
    result["index_domain"] = domain
    return result


def _review_visual_timeline_for_render(
    visual_timeline: VisualTimelinePlan,
    *,
    available_source_count: int,
    reporter: EventReporter,
    warnings: list[str],
    comparison_render_allowed: bool = True,
) -> TimelineReview | None:
    """Run non-blocking structural QA on the final pre-render group timeline."""

    if not isinstance(comparison_render_allowed, bool):
        raise TypeError("comparison_render_allowed must be a boolean")
    try:
        review = review_timeline(
            visual_timeline,
            available_source_count=available_source_count,
        )
    except (TypeError, ValueError) as exc:
        disposition = (
            "명시적으로 허용된 비교용 렌더는 계속합니다"
            if comparison_render_allowed
            else "자동 GROUP 렌더를 차단합니다"
        )
        message = f"타임라인 사전 QA를 계산하지 못했습니다. {disposition}: {exc}"
        warnings.append(message)
        reporter.emit("log", level="warning", message=message)
        return None

    outcome = "통과" if review.passed else "실패"
    reporter.log(
        f"타임라인 사전 QA {outcome} · {review.weighted_score:.1f}/100 · "
        f"소스 {review.unique_source_count}/{available_source_count} · "
        f"교차 전환 {review.source_switch_count}개 "
        f"({review.source_switch_rate * 100.0:.1f}%, "
        f"{review.source_switches_per_minute:.2f}회/분) · "
        f"시각 숏 중앙값 {review.visual_shot_median_seconds:.2f}초 · "
        f"소스 런 중앙값/최장 "
        f"{review.source_run_median_seconds:.2f}/"
        f"{review.longest_source_run_seconds:.2f}초 · "
        f"하드 실패 {len(review.hard_failures)}개"
    )
    if not review.passed:
        failure_codes = ", ".join(issue.code for issue in review.hard_failures)
        if comparison_render_allowed:
            warnings.append(
                "타임라인 사전 QA 실패 "
                f"({failure_codes}). 비교용 렌더를 계속했습니다."
            )
        else:
            warnings.append(
                "타임라인 사전 QA 실패 "
                f"({failure_codes}). 자동 GROUP 렌더를 차단합니다."
            )
    return review


def _require_reviewed_automatic_group_timeline(
    request: AnalysisRequest,
    *,
    visual_timeline: VisualTimelinePlan | None,
    timeline_review: TimelineReview | None,
    allow_failed_comparison_render: bool,
) -> None:
    """Keep the automatic product path from rendering an unreviewed draft."""

    if (
        request.edit_mode is not EditMode.GROUP
        or request.source_mode is not SourceMode.AUTOMATIC
        or allow_failed_comparison_render
    ):
        return
    if visual_timeline is None:
        raise RuntimeError(
            "automatic GROUP render requires a visual timeline; "
            "the timing-only group draft is analysis-only"
        )
    if timeline_review is None:
        raise RuntimeError(
            "automatic GROUP render requires a completed timeline review"
        )
    if not timeline_review.passed:
        failure_codes = ", ".join(
            issue.code for issue in timeline_review.hard_failures
        ) or "unknown_review_failure"
        raise RuntimeError(
            "automatic GROUP render blocked by timeline review failures: "
            f"{failure_codes}"
        )


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z가-힣._-]+", "-", value.strip()).strip("-._")
    return slug or "untitled"


def _member_render_selection_basis(selection_policy: str) -> str:
    """Name the complete anchored-tracking and global-selection contract."""

    policy = selection_policy.strip()
    if not policy:
        raise ValueError("member selection policy cannot be blank")
    return f"anchored_target_tracking_virtual_camera_{policy}"


def _capabilities_payload(
    *,
    group_ready: bool,
    edit_mode: EditMode,
    member_draft: MemberDraftTimeline | None,
    rendered_video_path: Path | None,
    used_automatic_member_anchors: bool,
    used_direct_full_frame: bool = False,
) -> dict[str, Any]:
    """Describe implemented analysis/render capabilities without claiming re-ID."""

    return {
        "group_timeline_inputs_ready": group_ready,
        "group_video_rendered": (
            edit_mode is EditMode.GROUP and rendered_video_path is not None
        ),
        "member_tracking_ready": (
            member_draft is not None
            and all(
                item.kind is MemberSegmentKind.TARGET_CROP
                for item in member_draft.segments
            )
        ),
        "member_video_rendered": (
            edit_mode is EditMode.MEMBER and rendered_video_path is not None
        ),
        "member_direct_full_frame_ready": (
            edit_mode is EditMode.MEMBER
            and used_direct_full_frame
            and rendered_video_path is not None
        ),
        "member_tracking_note": (
            (
                "자동 수집된 공식 대상 멤버 세로 직캠의 전체 프레임을 직접 "
                "사용하며, 자동 얼굴 신원 판정·크롭·추적은 수행하지 않습니다."
                if used_direct_full_frame
                else (
                    (
                        "공식 대상 멤버 직캠 메타데이터와 기하학적으로 안정적인 "
                        "중심 인물에서 시작한 자동 추적이며 얼굴 신원을 증명하지 "
                        "않습니다. "
                        if used_automatic_member_anchors
                        else "사용자 확인 앵커에서 "
                    )
                    + "시작한 전방향 연속 추적만 신뢰하며, "
                    "카메라 컷 뒤 자동 재식별은 사용하지 않습니다."
                )
            )
        ),
    }


def _validate_media(info: MediaInfo) -> None:
    if not info.video_streams:
        raise ValueError(f"video stream not found: {info.path}")
    if not info.audio_streams:
        raise ValueError(f"audio stream not found: {info.path}")
    if not info.duration_seconds or info.duration_seconds <= 0:
        raise ValueError(f"media duration is unavailable: {info.path}")


_AUDIO_ALIGNMENT_SCALAR_FIELDS = frozenset(
    {
        "offset_samples",
        "offset_seconds",
        "score",
        "overlap_samples",
        "overlap_seconds",
        "second_best_score",
        "score_margin",
        "overlap_ratio",
        "backend",
    }
)


def _audio_alignment_cache_analyzer(reference_sha256: str) -> dict[str, Any]:
    return {
        "analysis_contract": "ive_pairwise_audio_alignment_v1",
        "reference_source_sha256": reference_sha256,
        "sample_rate": ANALYSIS_SAMPLE_RATE,
        "max_offset_seconds": 120.0,
        "minimum_overlap_policy": "min_10_max_2_shorter_duration_div_3",
        "primary_backend": "spectral_flux_ncc",
        "fallback_backend": "waveform_ncc",
        "audio_implementation_sha256": hashlib.sha256(
            Path(__file__).with_name("audio.py").read_bytes()
        ).hexdigest(),
    }


def _audio_alignment_scalar_payload(
    alignment: SourceAlignment,
) -> dict[str, Any]:
    payload = alignment.to_dict()
    payload.pop("source", None)
    payload.pop("is_reference", None)
    return payload


def _audio_alignment_from_scalar_payload(
    payload: Mapping[str, Any],
    *,
    source: Path,
) -> SourceAlignment:
    if set(payload) != _AUDIO_ALIGNMENT_SCALAR_FIELDS:
        raise ValueError("audio alignment cache schema does not match")
    offset_samples = payload["offset_samples"]
    overlap_samples = payload["overlap_samples"]
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (offset_samples, overlap_samples)
    ) or int(overlap_samples) <= 0:
        raise ValueError("audio alignment cached sample counts are invalid")
    numeric_keys = (
        "offset_seconds",
        "score",
        "overlap_seconds",
        "score_margin",
        "overlap_ratio",
    )
    if any(
        isinstance(payload[key], bool)
        or not isinstance(payload[key], (int, float))
        for key in numeric_keys
    ):
        raise ValueError("audio alignment cache contains non-numeric values")
    numeric = {key: float(payload[key]) for key in numeric_keys}
    if any(not math.isfinite(value) for value in numeric.values()):
        raise ValueError("audio alignment cache contains non-finite values")
    if not -1.0 <= numeric["score"] <= 1.0:
        raise ValueError("audio alignment cached score is invalid")
    if numeric["overlap_seconds"] <= 0.0:
        raise ValueError("audio alignment cached overlap is invalid")
    if numeric["score_margin"] < 0.0:
        raise ValueError("audio alignment cached score margin is invalid")
    if not 0.0 <= numeric["overlap_ratio"] <= 1.0:
        raise ValueError("audio alignment cached overlap ratio is invalid")
    second_best = payload["second_best_score"]
    if second_best is not None and (
        not isinstance(second_best, (int, float))
        or isinstance(second_best, bool)
        or not math.isfinite(float(second_best))
        or not -1.0 <= float(second_best) <= 1.0
    ):
        raise ValueError("audio alignment cached second-best score is invalid")
    backend = payload["backend"]
    if backend not in {"spectral_flux_ncc", "waveform_ncc"}:
        raise ValueError("audio alignment cached backend is invalid")
    if not math.isclose(
        numeric["offset_seconds"],
        int(offset_samples) / ANALYSIS_SAMPLE_RATE,
        rel_tol=0.0,
        abs_tol=1.0 / ANALYSIS_SAMPLE_RATE + 1e-9,
    ):
        raise ValueError("audio alignment cached offset/sample count disagree")
    if not math.isclose(
        numeric["overlap_seconds"],
        int(overlap_samples) / ANALYSIS_SAMPLE_RATE,
        rel_tol=0.0,
        abs_tol=1.0 / ANALYSIS_SAMPLE_RATE + 1e-9,
    ):
        raise ValueError("audio alignment cached overlap/sample count disagree")
    return SourceAlignment(
        source=str(source),
        is_reference=False,
        offset_samples=int(offset_samples),
        offset_seconds=numeric["offset_seconds"],
        score=numeric["score"],
        overlap_samples=int(overlap_samples),
        overlap_seconds=numeric["overlap_seconds"],
        second_best_score=(
            float(second_best) if second_best is not None else None
        ),
        score_margin=numeric["score_margin"],
        overlap_ratio=numeric["overlap_ratio"],
        backend=str(backend),
    )


def _align_sequentially(
    request: AnalysisRequest,
    reporter: EventReporter,
) -> tuple[AlignmentResult, NDArray[np.float32]]:
    reference_path = request.sources[request.reference_source]
    reporter.status("기준 음원을 추출하는 중…")
    reference_pcm = extract_mono_pcm(reference_path, sample_rate=ANALYSIS_SAMPLE_RATE)
    alignments: list[SourceAlignment] = []
    try:
        reference_sha256 = source_content_sha256(reference_path)
    except (OSError, TypeError, ValueError):
        reference_sha256 = None

    for index, source in enumerate(request.sources):
        if index == request.reference_source:
            alignments.append(
                SourceAlignment(
                    source=str(source),
                    is_reference=True,
                    offset_samples=0,
                    offset_seconds=0.0,
                    score=1.0,
                    overlap_samples=int(reference_pcm.size),
                    overlap_seconds=reference_pcm.size / ANALYSIS_SAMPLE_RATE,
                    second_best_score=None,
                    score_margin=1.0,
                    overlap_ratio=1.0,
                    backend="reference",
                )
            )
            continue

        reporter.status(f"음원 정렬 중: {source.name}")
        alignment_cache: PersistentSourceScalarCache | None = None
        cached_alignment: SourceAlignment | None = None
        if (
            request.source_cache_directory is not None
            and reference_sha256 is not None
        ):
            try:
                alignment_cache = PersistentSourceScalarCache(
                    source_cache_directory=request.source_cache_directory,
                    source_path=source,
                    namespace="audio-alignment-v1",
                    analyzer=_audio_alignment_cache_analyzer(
                        reference_sha256
                    ),
                )
                cached_payload = alignment_cache.get()
                if cached_payload is not None:
                    cached_alignment = _audio_alignment_from_scalar_payload(
                        cached_payload,
                        source=source,
                    )
            except (OSError, TypeError, ValueError):
                alignment_cache = None
                cached_alignment = None
        if cached_alignment is not None:
            alignments.append(cached_alignment)
            reporter.log(
                f"{source.name}: 기준 대비 "
                f"{cached_alignment.offset_seconds:+.3f}초, 정렬 신뢰도 "
                f"{cached_alignment.score:.3f}, 후보 간격 "
                f"{cached_alignment.score_margin:.3f} "
                f"({cached_alignment.backend}) · 영속 캐시 재사용"
            )
            completed_sources = index + 1
            reporter.progress(
                0.22 + 0.38 * completed_sources / len(request.sources),
                "음방 오디오를 정렬하는 중…",
            )
            continue
        source_pcm = extract_mono_pcm(source, sample_rate=ANALYSIS_SAMPLE_RATE)
        minimum_overlap = min(
            10.0,
            max(
                2.0,
                min(reference_pcm.size, source_pcm.size)
                / ANALYSIS_SAMPLE_RATE
                / 3.0,
            ),
        )
        try:
            estimate = spectral_flux_alignment_offset(
                reference_pcm,
                source_pcm,
                sample_rate=ANALYSIS_SAMPLE_RATE,
                max_offset_seconds=120.0,
                min_overlap_seconds=minimum_overlap,
            )
        except ValueError:
            estimate = normalized_cross_correlation_offset(
                reference_pcm,
                source_pcm,
                sample_rate=ANALYSIS_SAMPLE_RATE,
                max_offset_seconds=120.0,
                min_overlap_seconds=minimum_overlap,
            )
        source_alignment = SourceAlignment(
            source=str(source),
            is_reference=False,
            offset_samples=estimate.offset_samples,
            offset_seconds=estimate.offset_seconds,
            score=estimate.score,
            overlap_samples=estimate.overlap_samples,
            overlap_seconds=estimate.overlap_seconds,
            second_best_score=estimate.second_best_score,
            score_margin=estimate.score_margin,
            overlap_ratio=estimate.overlap_ratio,
            backend=estimate.backend,
        )
        alignments.append(source_alignment)
        if alignment_cache is not None:
            try:
                alignment_cache.put(
                    _audio_alignment_scalar_payload(source_alignment)
                )
            except (OSError, TypeError, ValueError):
                pass
        reporter.log(
            f"{source.name}: 기준 대비 {estimate.offset_seconds:+.3f}초, "
            f"정렬 신뢰도 {estimate.score:.3f}, 후보 간격 {estimate.score_margin:.3f} "
            f"({estimate.backend})"
        )
        completed_sources = index + 1
        reporter.progress(
            0.22 + 0.38 * completed_sources / len(request.sources),
            "음방 오디오를 정렬하는 중…",
        )

    # Keep request/source order, including when the reference is not the first file.
    by_source = {alignment.source: alignment for alignment in alignments}
    ordered = tuple(by_source[str(source)] for source in request.sources)
    return (
        AlignmentResult(
            reference=str(reference_path),
            sample_rate=ANALYSIS_SAMPLE_RATE,
            alignments=ordered,
            group="IVE",
        ),
        reference_pcm,
    )


_TIMEWARP_SCALAR_FIELDS = frozenset(
    {
        "global_offset_seconds",
        "master_duration_seconds",
        "source_duration_seconds",
        "control_points",
        "anchors",
        "residuals",
        "config",
        "scalar_fallback",
        "fallback_reason",
        "backend",
    }
)


def _timewarp_cache_analyzer(
    *,
    reference_sha256: str,
    global_offset_seconds: float,
) -> dict[str, Any]:
    return {
        "analysis_contract": "ive_local_audio_timewarp_v1",
        "reference_source_sha256": reference_sha256,
        "sample_rate": ANALYSIS_SAMPLE_RATE,
        "global_offset_seconds": global_offset_seconds,
        "configuration": TimeWarpConfig().to_dict(),
        "timewarp_implementation_sha256": hashlib.sha256(
            Path(__file__).with_name("timewarp.py").read_bytes()
        ).hexdigest(),
        "rhythm_implementation_sha256": hashlib.sha256(
            Path(__file__).with_name("rhythm.py").read_bytes()
        ).hexdigest(),
    }


def _timewarp_scalar_payload(timewarp: TimeWarpMap) -> dict[str, Any]:
    return {
        "global_offset_seconds": timewarp.global_offset_seconds,
        "master_duration_seconds": timewarp.master_duration_seconds,
        "source_duration_seconds": timewarp.source_duration_seconds,
        "control_points": [
            point.to_dict() for point in timewarp.control_points
        ],
        "anchors": [anchor.to_dict() for anchor in timewarp.anchors],
        "residuals": timewarp.residuals.to_dict(),
        "config": timewarp.config.to_dict(),
        "scalar_fallback": timewarp.scalar_fallback,
        "fallback_reason": timewarp.fallback_reason,
        "backend": timewarp.backend,
    }


def _timewarp_from_scalar_payload(
    payload: Mapping[str, Any],
) -> TimeWarpMap:
    if set(payload) != _TIMEWARP_SCALAR_FIELDS:
        raise ValueError("timewarp cache schema does not match")

    def finite(value: Any, label: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"cached {label} is invalid")
        return float(value)

    raw_config = payload["config"]
    expected_config_keys = set(TimeWarpConfig().to_dict())
    if not isinstance(raw_config, dict) or set(raw_config) != expected_config_keys:
        raise ValueError("timewarp cached configuration schema does not match")
    minimum_anchor_count = raw_config["minimum_anchor_count"]
    if (
        isinstance(minimum_anchor_count, bool)
        or not isinstance(minimum_anchor_count, int)
    ):
        raise ValueError("timewarp cached minimum anchor count is invalid")
    config = TimeWarpConfig(
        window_seconds=finite(raw_config["window_seconds"], "window seconds"),
        hop_seconds=finite(raw_config["hop_seconds"], "hop seconds"),
        search_radius_seconds=finite(
            raw_config["search_radius_seconds"],
            "search radius",
        ),
        peak_exclusion_seconds=finite(
            raw_config["peak_exclusion_seconds"],
            "peak exclusion",
        ),
        minimum_score=finite(raw_config["minimum_score"], "minimum score"),
        minimum_score_margin=finite(
            raw_config["minimum_score_margin"],
            "minimum score margin",
        ),
        minimum_anchor_count=minimum_anchor_count,
        minimum_anchor_fraction=finite(
            raw_config["minimum_anchor_fraction"],
            "minimum anchor fraction",
        ),
        maximum_drift_rate=finite(
            raw_config["maximum_drift_rate"],
            "maximum drift rate",
        ),
        maximum_trend_residual_seconds=finite(
            raw_config["maximum_trend_residual_seconds"],
            "maximum trend residual",
        ),
    )
    if not math.isclose(
        finite(raw_config["minimum_slope"], "minimum slope"),
        config.minimum_slope,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        finite(raw_config["maximum_slope"], "maximum slope"),
        config.maximum_slope,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("timewarp cached slope bounds disagree with config")

    raw_points = payload["control_points"]
    if not isinstance(raw_points, list):
        raise TypeError("timewarp cached control points must be a list")
    points: list[TimeWarpControlPoint] = []
    for raw in raw_points:
        if not isinstance(raw, dict) or set(raw) != {
            "master_seconds",
            "source_seconds",
            "offset_seconds",
        }:
            raise ValueError("timewarp cached control point schema does not match")
        master = finite(raw["master_seconds"], "control master time")
        source = finite(raw["source_seconds"], "control source time")
        if not math.isclose(
            finite(raw["offset_seconds"], "control offset"),
            source - master,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("timewarp cached control offset disagrees")
        points.append(
            TimeWarpControlPoint(
                master_seconds=master,
                source_seconds=source,
            )
        )

    raw_anchors = payload["anchors"]
    if not isinstance(raw_anchors, list):
        raise TypeError("timewarp cached anchors must be a list")
    anchors: list[TimeWarpAnchor] = []
    anchor_keys = {
        "window_index",
        "window_start_seconds",
        "window_end_seconds",
        "master_seconds",
        "source_seconds",
        "offset_seconds",
        "score",
        "second_best_score",
        "score_margin",
        "accepted",
        "rejection_reason",
        "trend_residual_seconds",
    }
    for raw in raw_anchors:
        if not isinstance(raw, dict) or set(raw) != anchor_keys:
            raise ValueError("timewarp cached anchor schema does not match")
        window_index = raw["window_index"]
        accepted = raw["accepted"]
        rejection_reason = raw["rejection_reason"]
        if (
            isinstance(window_index, bool)
            or not isinstance(window_index, int)
            or window_index < 0
            or not isinstance(accepted, bool)
            or (
                rejection_reason is not None
                and not isinstance(rejection_reason, str)
            )
        ):
            raise ValueError("timewarp cached anchor metadata is invalid")

        def optional_finite(value: Any, label: str) -> float | None:
            return None if value is None else finite(value, label)

        anchor = TimeWarpAnchor(
                window_index=window_index,
                window_start_seconds=finite(
                    raw["window_start_seconds"],
                    "anchor window start",
                ),
                window_end_seconds=finite(
                    raw["window_end_seconds"],
                    "anchor window end",
                ),
                master_seconds=finite(raw["master_seconds"], "anchor master"),
                source_seconds=finite(raw["source_seconds"], "anchor source"),
                offset_seconds=finite(raw["offset_seconds"], "anchor offset"),
                score=optional_finite(raw["score"], "anchor score"),
                second_best_score=optional_finite(
                    raw["second_best_score"],
                    "anchor second-best score",
                ),
                score_margin=optional_finite(
                    raw["score_margin"],
                    "anchor score margin",
                ),
                accepted=accepted,
                rejection_reason=rejection_reason,
                trend_residual_seconds=optional_finite(
                    raw["trend_residual_seconds"],
                    "anchor trend residual",
                ),
        )
        if (
            anchor.window_end_seconds <= anchor.window_start_seconds
            or not math.isclose(
                anchor.offset_seconds,
                anchor.source_seconds - anchor.master_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or (anchor.accepted and anchor.rejection_reason is not None)
            or (
                not anchor.accepted
                and (
                    anchor.rejection_reason is None
                    or not anchor.rejection_reason
                )
            )
            or (
                anchor.score is not None
                and not -1.0 <= anchor.score <= 1.0
            )
            or (
                anchor.second_best_score is not None
                and not -1.0 <= anchor.second_best_score <= 1.0
            )
            or (
                anchor.score_margin is not None
                and anchor.score_margin < 0.0
            )
            or (
                anchor.trend_residual_seconds is not None
                and anchor.trend_residual_seconds < 0.0
            )
        ):
            raise ValueError("timewarp cached anchor invariants are invalid")
        anchors.append(anchor)

    raw_residuals = payload["residuals"]
    if not isinstance(raw_residuals, dict) or set(raw_residuals) != {
        "p50_seconds",
        "p95_seconds",
        "maximum_seconds",
        "sample_count",
        "definition",
    }:
        raise ValueError("timewarp cached residual schema does not match")
    sample_count = raw_residuals["sample_count"]
    definition = raw_residuals["definition"]
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 0
        or not isinstance(definition, str)
    ):
        raise ValueError("timewarp cached residual metadata is invalid")

    def residual_value(key: str) -> float | None:
        value = raw_residuals[key]
        return None if value is None else finite(value, f"residual {key}")

    residual_p50 = residual_value("p50_seconds")
    residual_p95 = residual_value("p95_seconds")
    residual_maximum = residual_value("maximum_seconds")
    if (
        definition != "absolute_anchor_residual_to_robust_affine_trend"
        or any(
            value is not None and value < 0.0
            for value in (residual_p50, residual_p95, residual_maximum)
        )
    ):
        raise ValueError("timewarp cached residual values are invalid")

    scalar_fallback = payload["scalar_fallback"]
    fallback_reason = payload["fallback_reason"]
    backend = payload["backend"]
    if (
        not isinstance(scalar_fallback, bool)
        or (fallback_reason is not None and not isinstance(fallback_reason, str))
        or backend != "spectral_flux_local_ncc"
    ):
        raise ValueError("timewarp cached policy metadata is invalid")
    accepted_anchor_count = sum(anchor.accepted for anchor in anchors)
    if scalar_fallback:
        if (
            points
            or accepted_anchor_count
            or sample_count != 0
            or any(
                value is not None
                for value in (residual_p50, residual_p95, residual_maximum)
            )
        ):
            raise ValueError("cached scalar fallback contains drift evidence")
    elif (
        accepted_anchor_count < config.minimum_anchor_count
        or sample_count != accepted_anchor_count
    ):
        raise ValueError("cached timewarp accepted-anchor evidence is incomplete")
    return TimeWarpMap(
        global_offset_seconds=finite(
            payload["global_offset_seconds"],
            "global offset",
        ),
        master_duration_seconds=finite(
            payload["master_duration_seconds"],
            "master duration",
        ),
        source_duration_seconds=finite(
            payload["source_duration_seconds"],
            "source duration",
        ),
        control_points=tuple(points),
        anchors=tuple(anchors),
        residuals=TimeWarpResidualDiagnostics(
            p50_seconds=residual_p50,
            p95_seconds=residual_p95,
            maximum_seconds=residual_maximum,
            sample_count=sample_count,
            definition=definition,
        ),
        config=config,
        scalar_fallback=scalar_fallback,
        fallback_reason=fallback_reason,
        backend=backend,
    )


def _estimate_source_timewarps(
    request: AnalysisRequest,
    alignment: AlignmentResult,
    reference_pcm: NDArray[np.float32],
    *,
    eligible_indices: Sequence[int],
    reporter: EventReporter,
    decoded_pcm_by_source: dict[int, NDArray[np.float32]] | None = None,
) -> tuple[tuple[TimeWarpMap | None, ...], tuple[dict[str, Any], ...]]:
    """Estimate checked local source clocks without weakening scalar fallback.

    Every eligible source is measured independently against the already-decoded
    reference signal.  A rejected estimator result is still a ``TimeWarpMap``
    whose ``master_to_source`` method uses the exact global offset.  Decode or
    validation failures are represented as ``None`` and therefore use the same
    scalar contract in visual analysis and rendering.
    """

    if reference_pcm is None or np.asarray(reference_pcm).size < 2:
        return (
            (None,) * len(request.sources),
            tuple(
                {
                    "source_index": index,
                    "status": "global_scalar_offset",
                    "reason": "reference_pcm_unavailable",
                }
                for index in range(len(request.sources))
            ),
        )

    if decoded_pcm_by_source is not None:
        decoded_pcm_by_source[request.reference_source] = np.asarray(
            reference_pcm,
            dtype=np.float32,
        )

    eligible = set(int(index) for index in eligible_indices)
    maps: list[TimeWarpMap | None] = [None] * len(request.sources)
    payloads: list[dict[str, Any]] = []
    reference_index = request.reference_source
    try:
        reference_sha256 = source_content_sha256(
            request.sources[reference_index]
        )
    except (OSError, TypeError, ValueError):
        reference_sha256 = None
    for index, source in enumerate(request.sources):
        if index not in eligible:
            payloads.append(
                {
                    "source_index": index,
                    "status": "not_estimated",
                    "reason": "source_not_render_eligible",
                }
            )
            continue
        reporter.status(f"로컬 음원 드리프트 분석 중: {source.name}")
        timewarp_cache: PersistentSourceScalarCache | None = None
        cached_timewarp: TimeWarpMap | None = None
        if (
            request.source_cache_directory is not None
            and reference_sha256 is not None
        ):
            try:
                timewarp_cache = PersistentSourceScalarCache(
                    source_cache_directory=request.source_cache_directory,
                    source_path=source,
                    namespace="local-timewarp-v1",
                    analyzer=_timewarp_cache_analyzer(
                        reference_sha256=reference_sha256,
                        global_offset_seconds=(
                            alignment.alignments[index].offset_seconds
                        ),
                    ),
                )
                cached_payload = timewarp_cache.get()
                if cached_payload is not None:
                    cached_timewarp = _timewarp_from_scalar_payload(
                        cached_payload
                    )
                    if not math.isclose(
                        cached_timewarp.global_offset_seconds,
                        alignment.alignments[index].offset_seconds,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    ):
                        raise ValueError(
                            "cached timewarp global offset does not match alignment"
                        )
            except (OSError, TypeError, ValueError):
                cached_timewarp = None
        if cached_timewarp is not None:
            try:
                if decoded_pcm_by_source is not None and index != reference_index:
                    decoded_pcm_by_source[index] = np.asarray(
                        extract_mono_pcm(
                            source,
                            sample_rate=ANALYSIS_SAMPLE_RATE,
                        ),
                        dtype=np.float32,
                    )
            except (AudioDecodeError, OSError, TypeError, ValueError):
                cached_timewarp = None
            else:
                maps[index] = cached_timewarp
                payload = cached_timewarp.to_dict()
                payload.update(
                    {
                        "source_index": index,
                        "status": (
                            "global_scalar_offset"
                            if cached_timewarp.scalar_fallback
                            else "local_monotonic_timewarp"
                        ),
                        "persistent_scalar_cache": {
                            **timewarp_cache.diagnostics(),
                            "analysis_reused": True,
                        },
                    }
                )
                payloads.append(payload)
                if cached_timewarp.scalar_fallback:
                    reporter.log(
                        f"{source.name}: 로컬 드리프트 근거 부족 · "
                        "단일 오프셋 유지 "
                        f"({cached_timewarp.fallback_reason}) · 영속 캐시 재사용"
                    )
                else:
                    reporter.log(
                        f"{source.name}: 로컬 드리프트 앵커 "
                        f"{len(cached_timewarp.accepted_anchors)}개 · "
                        "최대 재생률 편차 "
                        f"{max(abs(slope - 1.0) for slope in cached_timewarp.segment_slopes) * 100.0:.3f}% "
                        "· 영속 캐시 재사용"
                    )
                continue
        try:
            source_pcm = (
                reference_pcm
                if index == reference_index
                else extract_mono_pcm(source, sample_rate=ANALYSIS_SAMPLE_RATE)
            )
            if decoded_pcm_by_source is not None:
                decoded_pcm_by_source[index] = np.asarray(
                    source_pcm,
                    dtype=np.float32,
                )
            timewarp = estimate_local_timewarp(
                reference_pcm,
                source_pcm,
                sample_rate=ANALYSIS_SAMPLE_RATE,
                global_offset_seconds=alignment.alignments[index].offset_seconds,
            )
        except (AudioDecodeError, OSError, TypeError, ValueError) as exc:
            payloads.append(
                {
                    "source_index": index,
                    "status": "global_scalar_offset",
                    "reason": "local_timewarp_estimation_failed",
                    "error": str(exc),
                }
            )
            reporter.emit(
                "log",
                level="warning",
                message=(
                    f"{source.name}: 로컬 드리프트를 확정하지 못해 "
                    f"단일 오프셋 유지 ({exc})"
                ),
            )
            continue

        maps[index] = timewarp
        if timewarp_cache is not None:
            try:
                timewarp_cache.put(_timewarp_scalar_payload(timewarp))
            except (OSError, TypeError, ValueError):
                pass
        payload = timewarp.to_dict()
        payload.update(
            {
                "source_index": index,
                "status": (
                    "global_scalar_offset"
                    if timewarp.scalar_fallback
                    else "local_monotonic_timewarp"
                ),
            }
        )
        if timewarp_cache is not None:
            payload["persistent_scalar_cache"] = {
                **timewarp_cache.diagnostics(),
                "analysis_reused": False,
            }
        payloads.append(payload)
        if timewarp.scalar_fallback:
            reporter.log(
                f"{source.name}: 로컬 드리프트 근거 부족 · 단일 오프셋 유지 "
                f"({timewarp.fallback_reason})"
            )
        else:
            reporter.log(
                f"{source.name}: 로컬 드리프트 앵커 "
                f"{len(timewarp.accepted_anchors)}개 · "
                f"최대 재생률 편차 "
                f"{max(abs(slope - 1.0) for slope in timewarp.segment_slopes) * 100.0:.3f}%"
            )
    return (tuple(maps), tuple(payloads))


def _timewarp_safe_common_range(
    common_start: float,
    common_end: float,
    *,
    media: Sequence[MediaInfo],
    source_timewarps: Sequence[TimeWarpMap | None],
    eligible_indices: Sequence[int],
) -> tuple[float, float]:
    """Trim the common master range so every checked local clock stays decodable."""

    start = float(common_start)
    end = float(common_end)
    for index in eligible_indices:
        timewarp = source_timewarps[index]
        if timewarp is None or timewarp.scalar_fallback:
            continue
        duration = float(media[index].duration_seconds or 0.0)
        if duration <= 0.0:
            continue
        if timewarp.master_to_source(start) < 0.0:
            low, high = start, end
            for _ in range(48):
                middle = (low + high) / 2.0
                if timewarp.master_to_source(middle) < 0.0:
                    low = middle
                else:
                    high = middle
            start = high
        if timewarp.master_to_source(end) > duration:
            low, high = start, end
            for _ in range(48):
                middle = (low + high) / 2.0
                if timewarp.master_to_source(middle) <= duration:
                    low = middle
                else:
                    high = middle
            end = low
    if end - start <= 2.0 / 29.97:
        raise ValueError("locally aligned source videos have no common usable range")
    return (start, end)


def _common_master_range(
    media: list[MediaInfo],
    alignment: AlignmentResult,
    source_indices: list[int] | tuple[int, ...] | None = None,
) -> tuple[float, float]:
    starts: list[float] = [0.0]
    ends: list[float] = []
    indices = range(len(media)) if source_indices is None else source_indices
    for index in indices:
        info = media[index]
        source_alignment = alignment.alignments[index]
        assert info.duration_seconds is not None
        # master time t maps to source time t + offset.
        starts.append(-source_alignment.offset_seconds)
        ends.append(info.duration_seconds - source_alignment.offset_seconds)
    start = max(starts)
    end = min(ends)
    if end <= start:
        raise ValueError("aligned source videos have no common usable range")
    return (max(0.0, start), end)


def _member_anchor_trim(
    *,
    common_master_start: float,
    common_master_end: float,
    alignment: AlignmentResult,
    eligible_indices: list[int] | tuple[int, ...],
    tracking_results: dict[int, MemberTrackingResult],
    source_timewarps: Sequence[TimeWarpMap | None] | None = None,
    preserve_automatic_full_frame_prefix: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Return the first canonical time covered by every forward-only track.

    Manual member tracking deliberately starts at the user-confirmed frame and
    never guesses backwards.  A source-local observation at ``s`` maps through
    the inverse checked source clock (or ``s - audio_offset`` under scalar
    fallback). The safe shared start is therefore the latest canonical anchor
    among the eligible sources.
    """

    if source_timewarps is not None and eligible_indices:
        if max(eligible_indices) >= len(source_timewarps):
            raise ValueError("member source_timewarps do not cover eligible sources")
    anchor_details: list[dict[str, Any]] = []
    full_frame_only_source_indices: list[int] = []
    for source_index in eligible_indices:
        result = tracking_results.get(source_index)
        if result is None or not result.observations:
            if preserve_automatic_full_frame_prefix:
                full_frame_only_source_indices.append(source_index)
                continue
            raise ValueError(
                "eligible member sources need a tracked anchor observation"
            )
        source_local_time = result.observations[0].timestamp_seconds
        offset = alignment.alignments[source_index].offset_seconds
        timewarp = (
            source_timewarps[source_index]
            if source_timewarps is not None
            else None
        )
        if timewarp is not None and not math.isclose(
            timewarp.global_offset_seconds,
            offset,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("member timewarp global offset does not match alignment")
        canonical_time = (
            timewarp.source_to_master(source_local_time)
            if timewarp is not None
            else source_local_time - offset
        )
        anchor_details.append(
            {
                "source_index": source_index,
                "source": alignment.alignments[source_index].source,
                "source_local_time_seconds": source_local_time,
                "audio_offset_seconds": offset,
                "canonical_anchor_time_seconds": canonical_time,
                "source_clock_basis": (
                    "local_monotonic_timewarp"
                    if timewarp is not None and not timewarp.scalar_fallback
                    else "global_scalar_offset"
                ),
            }
        )

    if preserve_automatic_full_frame_prefix:
        # These are official, target-member-labelled fixed fancams. Before an
        # accepted geometry-only anchor (or for a source with no safe anchor),
        # retaining the broadcaster's complete portrait frame is safe; only a
        # target crop is prohibited. This preserves the aligned song opening
        # and lets the render-level hard-cut fallback use all verified sources.
        effective_start = common_master_start
    elif not anchor_details:
        effective_start = common_master_start
    else:
        effective_start = max(
            common_master_start,
            max(
                float(item["canonical_anchor_time_seconds"])
                for item in anchor_details
            ),
        )
    trimmed_seconds = max(0.0, effective_start - common_master_start)
    minimum_render_duration = 2.0 / 29.97
    usable = common_master_end - effective_start >= minimum_render_duration
    limiting_indices = (
        []
        if preserve_automatic_full_frame_prefix
        else [
            int(item["source_index"])
            for item in anchor_details
            if abs(
                float(item["canonical_anchor_time_seconds"]) - effective_start
            )
            <= 1e-7
        ]
    )
    return effective_start, {
        "index_domain": REQUEST_SOURCE_INDEX_DOMAIN,
        "policy": (
            "preserve_aligned_range_with_official_member_fancam_full_frame_fallback"
            if preserve_automatic_full_frame_prefix
            else "trim_to_latest_canonical_anchor"
        ),
        "reverse_tracking_used": False,
        "applied": trimmed_seconds > 1e-7,
        "aligned_start_seconds": common_master_start,
        "effective_start_seconds": effective_start,
        "end_seconds": common_master_end,
        "trimmed_seconds": trimmed_seconds,
        "usable_duration_seconds": max(0.0, common_master_end - effective_start),
        "usable_for_render": usable,
        "limiting_source_indices": limiting_indices,
        "full_frame_only_source_indices": full_frame_only_source_indices,
        "anchors": anchor_details,
    }


def _alignment_eligibility(
    alignment: AlignmentResult,
    *,
    edit_mode: EditMode = EditMode.GROUP,
    member_peer_consensus: (
        VerifiedAutomaticMemberAlignmentPeerConsensus | None
    ) = None,
) -> list[dict[str, Any]]:
    consensus_by_source: dict[int, Mapping[str, Any]] = {}
    if (
        isinstance(
            member_peer_consensus,
            VerifiedAutomaticMemberAlignmentPeerConsensus,
        )
        and member_peer_consensus.verified_for(alignment)
    ):
        raw_sources = member_peer_consensus.get("sources")
        if (
            member_peer_consensus.get("contract")
            == AUTOMATIC_MEMBER_ALIGNMENT_PEER_CONSENSUS_CONTRACT
            and isinstance(raw_sources, list)
        ):
            for raw in raw_sources:
                if not isinstance(raw, Mapping):
                    continue
                source_index = raw.get("source_index")
                if (
                    isinstance(source_index, int)
                    and not isinstance(source_index, bool)
                    and source_index not in consensus_by_source
                ):
                    consensus_by_source[source_index] = raw
    results: list[dict[str, Any]] = []
    for source_index, item in enumerate(alignment.alignments):
        reasons: list[str] = []
        minimum_score: float | None = None
        minimum_margin: float | None = None
        peer_consensus_candidate_score_floor: float | None = None
        peer_consensus_candidate_minimum_margin: float | None = None
        peer_consensus_candidate_minimum_overlap_ratio: float | None = None
        peer_consensus_candidate = False
        peer_consensus_acceptance_applied = False
        peer_consensus_record = consensus_by_source.get(source_index)
        if edit_mode is EditMode.MEMBER and peer_consensus_record is not None:
            direct = peer_consensus_record.get("direct_alignment")
            peer_consensus_acceptance_applied = bool(
                peer_consensus_record.get("authorized") is True
                and isinstance(direct, Mapping)
                and direct.get("backend") == item.backend
                and peer_consensus_record.get("reason")
                == "independent_peer_offset_consensus"
                and isinstance(
                    peer_consensus_record.get("reliable_peer_count"), int
                )
                and not isinstance(
                    peer_consensus_record.get("reliable_peer_count"), bool
                )
                and peer_consensus_record["reliable_peer_count"] >= 2
                and all(
                    isinstance(direct.get(name), (int, float))
                    and not isinstance(direct.get(name), bool)
                    and math.isclose(
                        float(direct[name]),
                        float(expected),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    for name, expected in (
                        ("offset_seconds", item.offset_seconds),
                        ("score", item.score),
                        ("score_margin", item.score_margin),
                        ("overlap_ratio", item.overlap_ratio),
                    )
                )
            )
        policy = (
            "member_fancam_multi_peer_consensus_v3"
            if edit_mode is EditMode.MEMBER
            else "group_broadcast_feed_strict_v1"
        )
        if not item.is_reference:
            minimum_margin = (
                0.06 if item.backend == "spectral_flux_ncc" else 0.04
            )
            # Official fixed-member fancams retain the same programme audio but
            # often mix crowd/stage microphones much more heavily than the
            # canonical group feed. Their sharp peak margin is a better guard
            # against a false lag than requiring the group-feed correlation
            # amplitude. Keep group mode strict while accepting the verified
            # IVE member-fancam range observed in real broadcasts.
            if edit_mode is EditMode.MEMBER:
                minimum_score = (
                    0.30 if item.backend == "spectral_flux_ncc" else 0.35
                )
                # A fixed-member fancam can sit below the programme-feed
                # correlation floor because its stage/crowd microphone mix is
                # different.  The broader floor below does not itself admit
                # the source: it merely schedules the independent multi-peer
                # verifier.  Eligibility changes only when its sealed result
                # binds at least two reliable peer offsets to this exact
                # ordered AlignmentResult.
                if item.backend == "spectral_flux_ncc":
                    peer_consensus_candidate_score_floor = (
                        _MEMBER_PEER_CONSENSUS_DIRECT_SCORE_FLOOR
                    )
                    peer_consensus_candidate_minimum_margin = (
                        _MEMBER_PEER_CONSENSUS_DIRECT_MARGIN_FLOOR
                    )
                    peer_consensus_candidate_minimum_overlap_ratio = (
                        _MEMBER_PEER_CONSENSUS_DIRECT_OVERLAP_FLOOR
                    )
                    peer_consensus_candidate = bool(
                        (
                            item.score < minimum_score
                            or item.score_margin < minimum_margin
                        )
                        and item.score >= peer_consensus_candidate_score_floor
                        and item.score_margin
                        >= peer_consensus_candidate_minimum_margin
                        and item.overlap_ratio
                        >= peer_consensus_candidate_minimum_overlap_ratio
                    )
            else:
                minimum_score = (
                    0.55 if item.backend == "spectral_flux_ncc" else 0.45
                )
            if (
                item.score < minimum_score
                and not peer_consensus_acceptance_applied
            ):
                reasons.append(
                    f"상관 점수 {item.score:.3f} < {minimum_score:.3f}"
                )
            if (
                item.score_margin < minimum_margin
                and not peer_consensus_acceptance_applied
            ):
                reasons.append(
                    f"후보 간격 {item.score_margin:.3f} < {minimum_margin:.3f}"
                )
            if item.overlap_ratio < 0.35:
                reasons.append(f"겹침 비율 {item.overlap_ratio:.3f} < 0.350")
        results.append(
            {
                "index_domain": REQUEST_SOURCE_INDEX_DOMAIN,
                "source_index": source_index,
                "source": item.source,
                "render_eligible": not reasons,
                "reasons": reasons,
                "policy": policy,
                "thresholds": {
                    "minimum_score": minimum_score,
                    "minimum_margin": minimum_margin,
                    "minimum_overlap_ratio": (
                        None if item.is_reference else 0.35
                    ),
                    "peer_consensus_candidate_score_floor": (
                        peer_consensus_candidate_score_floor
                    ),
                    "peer_consensus_candidate_minimum_margin": (
                        peer_consensus_candidate_minimum_margin
                    ),
                    "peer_consensus_candidate_minimum_overlap_ratio": (
                        peer_consensus_candidate_minimum_overlap_ratio
                    ),
                },
                "measured": {
                    "offset_seconds": item.offset_seconds,
                    "score": item.score,
                    "score_margin": item.score_margin,
                    "overlap_ratio": item.overlap_ratio,
                    "backend": item.backend,
                    "peer_consensus_candidate": peer_consensus_candidate,
                    "peer_consensus_acceptance_applied": (
                        peer_consensus_acceptance_applied
                    ),
                },
            }
        )
    return results


def _automatic_member_acquisition_source_identities(
    request: AnalysisRequest,
    authority: VerifiedMemberAcquisitionIdentityAuthority | None,
) -> dict[str, Any]:
    """Bind each automatic member source to one acquired broadcast slot."""

    if not is_verified_member_acquisition_identity_authority(authority):
        raise ValueError("verified member acquisition authority is unavailable")
    assert authority is not None
    cache_directory = getattr(request, "source_cache_directory", None)
    if cache_directory is None:
        raise ValueError("automatic member acquisition cache is unavailable")
    manifest_path = Path(cache_directory) / "acquisition-manifest.json"
    encoded, manifest_fingerprint = _read_stable_regular_file(manifest_path)
    if not (
        str(manifest_path.resolve(strict=True)) == authority.manifest_path
        and manifest_fingerprint["size_bytes"]
        == authority.manifest_size_bytes
        and manifest_fingerprint["sha256"] == authority.manifest_sha256
    ):
        raise ValueError("member acquisition manifest authority mismatch")
    try:
        manifest = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("automatic member acquisition manifest is invalid") from exc
    if not isinstance(manifest, dict):
        raise ValueError("automatic member acquisition manifest must be an object")
    expected_member = (
        request.target_member.value
        if getattr(request, "target_member", None) is not None
        else None
    )
    if not (
        str(manifest.get("group", "")).strip().upper() == "IVE"
        and manifest.get("song") == request.song
        and str(manifest.get("edit_mode", "")).strip().lower() == "member"
        and manifest.get("target_member") == expected_member
        and authority.group == "IVE"
        and authority.song == request.song
        and authority.target_member == expected_member
    ):
        raise ValueError("automatic member acquisition manifest scope mismatch")
    selected = manifest.get("selected")
    if not isinstance(selected, list) or len(selected) != len(request.sources):
        raise ValueError("automatic member acquisition source count mismatch")
    if len(authority.sources) != len(request.sources):
        raise ValueError("member acquisition authority source count mismatch")

    identities: list[dict[str, Any]] = []
    seen_video_ids: set[str] = set()
    for source_index, (raw_entry, source, trusted_identity) in enumerate(
        zip(selected, request.sources, authority.sources, strict=True)
    ):
        if not isinstance(raw_entry, Mapping):
            raise ValueError("automatic member acquisition entry is invalid")
        video_id = raw_entry.get("video_id")
        local_path = raw_entry.get("local_path")
        broadcast_date = raw_entry.get("broadcast_date")
        title = raw_entry.get("title")
        channel_id = raw_entry.get("channel_id")
        declared_integrity = raw_entry.get("content_integrity")
        if not (
            raw_entry.get("source_index") == source_index
            and trusted_identity.source_index == source_index
            and isinstance(video_id, str)
            and re.fullmatch(r"[0-9A-Za-z_-]{6,32}", video_id)
            and video_id not in seen_video_ids
            and isinstance(local_path, str)
            and Path(local_path).expanduser().resolve(strict=True)
            == Path(source).expanduser().resolve(strict=True)
            and trusted_identity.source_path
            == str(Path(source).expanduser().resolve(strict=True))
            and isinstance(broadcast_date, str)
            and re.fullmatch(r"\d{4}-\d{2}-\d{2}", broadcast_date)
            and isinstance(title, str)
            and isinstance(declared_integrity, Mapping)
        ):
            raise ValueError("automatic member acquisition identity mismatch")
        programme = raw_entry.get("broadcast_program")
        if not isinstance(programme, str) or not programme.strip():
            programme = canonical_music_show_program(title)
        if not isinstance(programme, str) or not programme.strip():
            raise ValueError("automatic member broadcast programme is unknown")
        current_fingerprint = _fingerprint_stable_regular_file_streaming(
            Path(source)
        )
        if not (
            declared_integrity.get("algorithm") == "sha256"
            and declared_integrity.get("size_bytes")
            == current_fingerprint["size_bytes"]
            and declared_integrity.get("sha256")
            == current_fingerprint["sha256"]
            and trusted_identity.video_id == video_id
            and trusted_identity.broadcast_date == broadcast_date
            and trusted_identity.broadcast_program == programme.strip()
            and trusted_identity.source_size_bytes
            == current_fingerprint["size_bytes"]
            and trusted_identity.source_sha256
            == current_fingerprint["sha256"]
        ):
            raise ValueError("automatic member source content binding mismatch")
        seen_video_ids.add(video_id)
        identities.append(
            {
                "index_domain": REQUEST_SOURCE_INDEX_DOMAIN,
                "source_index": source_index,
                "video_id": video_id,
                "broadcast_date": broadcast_date,
                "broadcast_program": programme.strip(),
                "broadcast_slot": trusted_identity.broadcast_slot,
                "channel_id": channel_id if isinstance(channel_id, str) else None,
                "source_content_sha256": current_fingerprint["sha256"],
            }
        )
    return {
        "analysis_basis": (
            "stable_acquisition_manifest_exact_source_content_and_"
            "broadcast_date_program_v1"
        ),
        "manifest": manifest_fingerprint,
        "sources": identities,
    }


def _verify_automatic_member_borderline_alignments(
    request: AnalysisRequest,
    alignment: AlignmentResult,
    *,
    reporter: EventReporter,
    reference_pcm: NDArray[np.float32] | None = None,
    acquisition_identity_authority: (
        VerifiedMemberAcquisitionIdentityAuthority | None
    ) = None,
) -> VerifiedAutomaticMemberAlignmentPeerConsensus:
    """Second-pass ambiguous member audio against independent accepted peers.

    A borderline official member fancam is never rescued merely because it is
    close to one numeric threshold.  At least two independently accepted
    non-reference broadcasts must predict the same source-to-master offset,
    and that consensus must also agree with the original reference estimate.
    """

    result: dict[str, Any] = {
        "contract": AUTOMATIC_MEMBER_ALIGNMENT_PEER_CONSENSUS_CONTRACT,
        "applicable": bool(
            request.source_mode is SourceMode.AUTOMATIC
            and request.edit_mode is EditMode.MEMBER
        ),
        "attempted_source_count": 0,
        "authorized_source_count": 0,
        "sources": [],
        "policy": {
            "minimum_direct_score": (
                _MEMBER_PEER_CONSENSUS_DIRECT_SCORE_FLOOR
            ),
            "minimum_direct_score_margin": (
                _MEMBER_PEER_CONSENSUS_DIRECT_MARGIN_FLOOR
            ),
            "minimum_direct_overlap_ratio": (
                _MEMBER_PEER_CONSENSUS_DIRECT_OVERLAP_FLOOR
            ),
            "minimum_peer_count": 2,
            "maximum_peer_count": 3,
            "minimum_peer_score": (
                _MEMBER_PEER_CONSENSUS_PEER_SCORE_FLOOR
            ),
            "minimum_peer_score_margin": (
                _MEMBER_PEER_CONSENSUS_PEER_MARGIN_FLOOR
            ),
            "minimum_peer_overlap_ratio": (
                _MEMBER_PEER_CONSENSUS_PEER_OVERLAP_FLOOR
            ),
            "maximum_offset_deviation_seconds": (
                _MEMBER_PEER_CONSENSUS_OFFSET_TOLERANCE_SECONDS
            ),
            "reference_peer_reuse_allowed": False,
            "duplicate_peer_audio_reuse_allowed": False,
        },
    }
    def sealed() -> VerifiedAutomaticMemberAlignmentPeerConsensus:
        return VerifiedAutomaticMemberAlignmentPeerConsensus(
            _payload_json=json.dumps(
                result,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            _alignment_sha256=_alignment_result_sha256(alignment),
            _seal=_AUTOMATIC_MEMBER_ALIGNMENT_PEER_CONSENSUS_SEAL,
        )

    if not result["applicable"]:
        return sealed()
    if len(alignment.alignments) != len(request.sources):
        raise ValueError("member peer consensus alignment/source count mismatch")

    try:
        broadcast_identity = _automatic_member_acquisition_source_identities(
            request,
            acquisition_identity_authority,
        )
        raw_broadcast_sources = broadcast_identity["sources"]
        if not isinstance(raw_broadcast_sources, list) or len(
            raw_broadcast_sources
        ) != len(request.sources):
            raise ValueError("member broadcast identity count mismatch")
        broadcast_sources = tuple(raw_broadcast_sources)
        result["broadcast_identity_verification"] = {
            **broadcast_identity,
            "verified": True,
        }
    except (OSError, TypeError, ValueError) as exc:
        broadcast_sources = ()
        result["broadcast_identity_verification"] = {
            "verified": False,
            "error": str(exc),
        }

    baseline = _alignment_eligibility(alignment, edit_mode=EditMode.MEMBER)
    accepted_peer_indices = tuple(
        sorted(
            (
                index
                for index, item in enumerate(baseline)
                if item["render_eligible"]
                and not alignment.alignments[index].is_reference
            ),
            key=lambda index: (
                -alignment.alignments[index].score_margin,
                -alignment.alignments[index].score,
                -alignment.alignments[index].overlap_ratio,
                index,
            ),
        )
    )
    cache: dict[int, NDArray[np.float32]] = {}
    reference_indices = tuple(
        index
        for index, item in enumerate(alignment.alignments)
        if item.is_reference
    )
    if len(reference_indices) != 1:
        raise ValueError("member peer consensus requires exactly one reference")
    reference_source_index = reference_indices[0]

    def decoded(source_index: int) -> NDArray[np.float32]:
        cached = cache.get(source_index)
        if cached is None:
            cached = np.asarray(
                extract_mono_pcm(
                    request.sources[source_index],
                    sample_rate=ANALYSIS_SAMPLE_RATE,
                ),
                dtype=np.float32,
            )
            cache[source_index] = cached
        return cached

    try:
        resolved_reference_pcm = (
            np.asarray(reference_pcm, dtype=np.float32)
            if reference_pcm is not None
            else decoded(reference_source_index)
        )
        if resolved_reference_pcm.ndim != 1 or resolved_reference_pcm.size == 0:
            raise ValueError("reference PCM is empty or non-mono")
        reference_pcm_sha256 = hashlib.sha256(
            np.ascontiguousarray(
                resolved_reference_pcm,
                dtype=np.float32,
            ).tobytes()
        ).hexdigest()
    except (AudioDecodeError, OSError, TypeError, ValueError) as exc:
        reference_pcm_sha256 = None
        result["reference_audio_identity"] = {
            "verified": False,
            "error": str(exc),
        }
    else:
        result["reference_audio_identity"] = {
            "verified": True,
            "source_index": reference_source_index,
            "pcm_sha256": reference_pcm_sha256,
        }

    records: list[dict[str, Any]] = []
    for source_index, eligibility in enumerate(baseline):
        direct = alignment.alignments[source_index]
        if direct.is_reference or eligibility["render_eligible"]:
            continue
        direct_payload = {
            "offset_seconds": direct.offset_seconds,
            "score": direct.score,
            "score_margin": direct.score_margin,
            "overlap_ratio": direct.overlap_ratio,
            "backend": direct.backend,
        }
        record: dict[str, Any] = {
            "index_domain": REQUEST_SOURCE_INDEX_DOMAIN,
            "source_index": source_index,
            "source": direct.source,
            "direct_alignment": direct_payload,
            "baseline_reasons": list(eligibility["reasons"]),
            "attempted": False,
            "authorized": False,
            "reason": "outside_borderline_peer_consensus_envelope",
            "peer_alignments": [],
        }
        direct_borderline = bool(
            direct.backend == "spectral_flux_ncc"
            and direct.score
            >= _MEMBER_PEER_CONSENSUS_DIRECT_SCORE_FLOOR
            and direct.score_margin
            >= _MEMBER_PEER_CONSENSUS_DIRECT_MARGIN_FLOOR
            and direct.overlap_ratio
            >= _MEMBER_PEER_CONSENSUS_DIRECT_OVERLAP_FLOOR
        )
        peer_indices = tuple(
            index
            for index in accepted_peer_indices
            if index != source_index
        )
        if not direct_borderline:
            records.append(record)
            continue
        if not broadcast_sources:
            record["reason"] = "verified_broadcast_identity_unavailable"
            records.append(record)
            continue
        if reference_pcm_sha256 is None:
            record["reason"] = "reference_audio_identity_unavailable"
            records.append(record)
            continue
        target_identity = broadcast_sources[source_index]
        reference_identity = broadcast_sources[reference_source_index]
        record["target_broadcast_identity"] = dict(target_identity)
        record["reference_broadcast_identity"] = dict(reference_identity)
        if len(peer_indices) < 2:
            record["reason"] = "fewer_than_two_independent_accepted_peers"
            records.append(record)
            continue

        record["attempted"] = True
        result["attempted_source_count"] += 1
        try:
            target_pcm = decoded(source_index)
        except (AudioDecodeError, OSError, TypeError, ValueError) as exc:
            record["reason"] = "target_audio_decode_failed"
            record["error"] = str(exc)
            records.append(record)
            continue

        target_pcm_sha256 = hashlib.sha256(
            np.ascontiguousarray(target_pcm, dtype=np.float32).tobytes()
        ).hexdigest()
        peer_payloads: list[dict[str, Any]] = []
        predictions: list[float] = []
        accepted_peer_pcm_sha256: set[str] = set()
        accepted_peer_broadcast_slots: set[str] = set()
        for peer_source_index in peer_indices:
            if len(predictions) >= 3:
                break
            peer = alignment.alignments[peer_source_index]
            peer_identity = broadcast_sources[peer_source_index]
            peer_broadcast_slot = peer_identity["broadcast_slot"]
            excluded_broadcast_slots = {
                target_identity["broadcast_slot"],
                reference_identity["broadcast_slot"],
            }
            if (
                peer_broadcast_slot in excluded_broadcast_slots
                or peer_broadcast_slot in accepted_peer_broadcast_slots
            ):
                peer_payloads.append(
                    {
                        "index_domain": REQUEST_SOURCE_INDEX_DOMAIN,
                        "peer_source_index": peer_source_index,
                        "broadcast_identity": dict(peer_identity),
                        "reliable": False,
                        "reason": "broadcast_slot_not_independent",
                    }
                )
                continue
            try:
                peer_pcm = decoded(peer_source_index)
                peer_pcm_sha256 = hashlib.sha256(
                    np.ascontiguousarray(peer_pcm, dtype=np.float32).tobytes()
                ).hexdigest()
                minimum_overlap = min(
                    10.0,
                    max(
                        2.0,
                        min(peer_pcm.size, target_pcm.size)
                        / ANALYSIS_SAMPLE_RATE
                        / 3.0,
                    ),
                )
                estimate = spectral_flux_alignment_offset(
                    peer_pcm,
                    target_pcm,
                    sample_rate=ANALYSIS_SAMPLE_RATE,
                    max_offset_seconds=120.0,
                    min_overlap_seconds=minimum_overlap,
                )
                predicted_offset = peer.offset_seconds + estimate.offset_seconds
                reliable = bool(
                    estimate.score
                    >= _MEMBER_PEER_CONSENSUS_PEER_SCORE_FLOOR
                    and estimate.score_margin
                    >= _MEMBER_PEER_CONSENSUS_PEER_MARGIN_FLOOR
                    and estimate.overlap_ratio
                    >= _MEMBER_PEER_CONSENSUS_PEER_OVERLAP_FLOOR
                    and peer_pcm_sha256 != target_pcm_sha256
                    and peer_pcm_sha256 != reference_pcm_sha256
                    and peer_pcm_sha256 not in accepted_peer_pcm_sha256
                )
                peer_payload = {
                    "index_domain": REQUEST_SOURCE_INDEX_DOMAIN,
                    "peer_source_index": peer_source_index,
                    "peer_master_offset_seconds": peer.offset_seconds,
                    "peer_to_target_offset_seconds": estimate.offset_seconds,
                    "predicted_target_master_offset_seconds": predicted_offset,
                    "score": estimate.score,
                    "score_margin": estimate.score_margin,
                    "overlap_ratio": estimate.overlap_ratio,
                    "backend": estimate.backend,
                    "broadcast_identity": dict(peer_identity),
                    "peer_pcm_sha256": peer_pcm_sha256,
                    "target_pcm_sha256_equal": (
                        peer_pcm_sha256 == target_pcm_sha256
                    ),
                    "reference_pcm_sha256_equal": (
                        peer_pcm_sha256 == reference_pcm_sha256
                    ),
                    "duplicate_reliable_peer_audio": (
                        peer_pcm_sha256 in accepted_peer_pcm_sha256
                    ),
                    "reliable": reliable,
                }
                if reliable:
                    predictions.append(predicted_offset)
                    accepted_peer_pcm_sha256.add(peer_pcm_sha256)
                    accepted_peer_broadcast_slots.add(peer_broadcast_slot)
            except (AudioDecodeError, OSError, TypeError, ValueError) as exc:
                peer_payload = {
                    "index_domain": REQUEST_SOURCE_INDEX_DOMAIN,
                    "peer_source_index": peer_source_index,
                    "reliable": False,
                    "error": str(exc),
                }
            peer_payloads.append(peer_payload)
        record["peer_alignments"] = peer_payloads
        if len(predictions) < 2:
            record["reason"] = "fewer_than_two_reliable_peer_predictions"
            records.append(record)
            continue
        consensus_offset = float(median(predictions))
        maximum_deviation = max(
            abs(value - consensus_offset) for value in predictions
        )
        direct_deviation = abs(consensus_offset - direct.offset_seconds)
        tolerance = _MEMBER_PEER_CONSENSUS_OFFSET_TOLERANCE_SECONDS
        authorized = bool(
            maximum_deviation <= tolerance + 1e-12
            and direct_deviation <= tolerance + 1e-12
        )
        record.update(
            {
                "reliable_peer_count": len(predictions),
                "reliable_peer_broadcast_slot_count": len(
                    accepted_peer_broadcast_slots
                ),
                "consensus_offset_seconds": consensus_offset,
                "maximum_peer_deviation_seconds": maximum_deviation,
                "direct_consensus_deviation_seconds": direct_deviation,
                "authorized": authorized,
                "reason": (
                    "independent_peer_offset_consensus"
                    if authorized
                    else "peer_offsets_or_direct_estimate_disagree"
                ),
            }
        )
        if authorized:
            result["authorized_source_count"] += 1
            reporter.log(
                f"{Path(direct.source).name}: 경계 정렬값을 "
                f"독립 직캠 {len(predictions)}개 오프셋 합의로 복구 "
                f"({consensus_offset:+.3f}초)"
            )
        records.append(record)
    result["sources"] = records
    return sealed()


def _tracking_frames(
    source: Path,
    *,
    sample_every: int = 2,
    maximum_width: int = 640,
) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    for frame_index, frame in iter_video_frames(source, sample_every=sample_every):
        height, width = frame.shape[:2]
        if width > maximum_width:
            scale = maximum_width / width
            frame = cv2.resize(
                frame,
                (maximum_width, max(2, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        yield frame_index, frame


def _member_tracking_payload(
    source_index: int,
    result: MemberTrackingResult,
    anchor_time_seconds: float,
    anchor_bbox: tuple[float, float, float, float],
) -> dict[str, Any]:
    counts = {
        state.value: sum(item.state is state for item in result.observations)
        for state in TrackingState
    }
    # Keep a diagnostic path at roughly 6 Hz rather than writing every decoded
    # frame into the analysis manifest.
    keyframe_stride = max(1, int(round(result.fps / 6.0)))
    keyframes = [
        {
            "frame_index": item.frame_index,
            "time_seconds": item.timestamp_seconds,
            "bbox_xyxy": list(item.bbox_xyxy),
            "confidence": item.confidence,
            "state": item.state.value,
        }
        for item in result.observations
        if item.bbox_xyxy is not None
        and (
            item.frame_index == result.observations[0].frame_index
            or item.frame_index % keyframe_stride == 0
        )
    ]
    return {
        "index_domain": REQUEST_SOURCE_INDEX_DOMAIN,
        "source_index": source_index,
        "source": result.source_id,
        "member": result.member.value,
        "anchor": {
            "time_seconds": anchor_time_seconds,
            "bbox_xyxy": list(anchor_bbox),
        },
        "backend": result.backend_name,
        "terminal_failure": (
            result.terminal_failure.value if result.terminal_failure else None
        ),
        "observation_counts": counts,
        "keyframes": keyframes,
    }


def _automatic_member_portrait_source_ratios(
    request: AnalysisRequest,
    media: Sequence[MediaInfo],
) -> tuple[float, ...] | None:
    """Fail closed on the materialized automatic portrait-output contract.

    Automatic acquisition is the provenance boundary: it admits only official
    target-member fancams. Native portrait sources can be retained unchanged;
    wider or taller official FaceCams are also valid when their decoded pixels
    can cover a 1080x1920 portrait output without upscaling. Missing dimensions
    or insufficient crop headroom remains a contract failure rather than
    permission to enter anchor/tracking fallback.
    """

    if (
        request.edit_mode is not EditMode.MEMBER
        or request.source_mode is not SourceMode.AUTOMATIC
        or request.aspect_ratio is not AspectRatio.PORTRAIT
        or bool(request.member_anchors)
    ):
        return None
    maximum_count = min(
        request.desired_source_count,
        MAXIMUM_AUTOMATIC_MEMBER_SOURCE_COUNT,
    )
    if not (
        MINIMUM_AUTOMATIC_MEMBER_SOURCE_COUNT
        <= len(request.sources)
        <= maximum_count
    ):
        raise ValueError(
            "automatic MEMBER+PORTRAIT direct-full-frame requires between "
            f"{MINIMUM_AUTOMATIC_MEMBER_SOURCE_COUNT} and {maximum_count} "
            f"materialized portrait sources; got {len(request.sources)}"
        )
    if len(media) != len(request.sources):
        raise ValueError(
            "automatic MEMBER+PORTRAIT direct-full-frame aspect contract "
            "cannot validate every materialized source"
        )
    ratios: list[float] = []
    for source_index, info in enumerate(media):
        if not info.video_streams:
            raise ValueError(
                "automatic MEMBER+PORTRAIT direct-full-frame aspect contract "
                f"failed for source {source_index}: first video stream missing"
            )
        stream = info.video_streams[0]
        if not stream.width or not stream.height:
            raise ValueError(
                "automatic MEMBER+PORTRAIT direct-full-frame aspect contract "
                f"failed for source {source_index}: dimensions unavailable"
            )
        ratio = stream.width / stream.height
        lossless_zoom = min(
            stream.width / _MEMBER_PRODUCTION_WIDTH,
            stream.height / _MEMBER_PRODUCTION_HEIGHT,
        )
        if lossless_zoom < 1.0 - 1e-6:
            raise ValueError(
                "automatic MEMBER+PORTRAIT lossless-fill contract failed for "
                f"source {source_index}: {stream.width}x{stream.height} cannot "
                f"cover {_MEMBER_PRODUCTION_WIDTH}x{_MEMBER_PRODUCTION_HEIGHT} "
                "without upscaling"
            )
        ratios.append(ratio)
    return tuple(ratios)


def _automatic_member_direct_full_frame_ratios(
    request: AnalysisRequest,
    media: Sequence[MediaInfo],
    eligible_indices: Sequence[int],
) -> tuple[float, ...] | None:
    """Gate the provenance-scoped portrait member path.

    The historical function name remains stable for analysis compatibility;
    returned ratios may now include lossless-croppable 4K landscape FaceCams.
    """

    ratios = _automatic_member_portrait_source_ratios(request, media)
    if ratios is None:
        return None
    eligible = tuple(int(index) for index in eligible_indices)
    maximum_count = min(
        request.desired_source_count,
        MAXIMUM_AUTOMATIC_MEMBER_SOURCE_COUNT,
        len(ratios),
    )
    if not (
        MINIMUM_AUTOMATIC_MEMBER_SOURCE_COUNT
        <= len(eligible)
        <= maximum_count
    ):
        raise ValueError(
            "automatic MEMBER+PORTRAIT direct-full-frame requires between "
            f"{MINIMUM_AUTOMATIC_MEMBER_SOURCE_COUNT} and {maximum_count} "
            f"audio-aligned portrait sources; got {len(eligible)}"
        )
    if any(index < 0 or index >= len(ratios) for index in eligible):
        raise ValueError(
            "automatic MEMBER+PORTRAIT direct-full-frame eligible source "
            "index is outside the materialized portrait source list"
        )
    return tuple(ratios[index] for index in eligible)


def _lossless_member_portrait_fill_crop(
    source_aspect_ratio: float,
) -> NormalizedCrop | None:
    """Return the centered source crop needed to fill a 9:16 output.

    ``None`` preserves an already-exact portrait source. The acquisition and
    materialization gates independently guarantee enough decoded pixels; this
    helper only expresses the geometry and never authorizes upscaling.
    """

    if (
        not math.isfinite(source_aspect_ratio)
        or source_aspect_ratio <= 0.0
    ):
        raise ValueError("member portrait-fill source aspect ratio is invalid")
    target_ratio = 9.0 / 16.0
    if math.isclose(
        source_aspect_ratio,
        target_ratio,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        return None
    if source_aspect_ratio > target_ratio:
        width = target_ratio / source_aspect_ratio
        return NormalizedCrop(
            x=(1.0 - width) / 2.0,
            y=0.0,
            width=width,
            height=1.0,
        )
    height = source_aspect_ratio / target_ratio
    return NormalizedCrop(
        x=0.0,
        y=(1.0 - height) / 2.0,
        width=1.0,
        height=height,
    )


@dataclass(frozen=True, slots=True)
class _DirectFullFrameSpan:
    canonical_master_start: float
    canonical_master_end: float
    timing_segment_count: int
    estimated_beat_count: float
    start_boundary_reasons: tuple[str, ...]
    end_boundary_reasons: tuple[str, ...]

    @property
    def duration_seconds(self) -> float:
        return self.canonical_master_end - self.canonical_master_start


@dataclass(frozen=True, slots=True)
class _DirectFullFrameBoundaryShift:
    boundary_index: int
    shift_units: int
    left_unit_count: int
    right_unit_count: int
    canonical_master_seconds: float
    boundary_reasons: tuple[str, ...]
    trigger: str
    parent_profile_id: str
    parent_partition_signature_sha256: str
    proof_failure: str | None
    proof_failed_segment_index: int | None
    proof_forward_reachable_left_source_indices: tuple[int, ...]
    proof_left_valid_source_indices: tuple[int, ...]
    proof_right_valid_source_indices: tuple[int, ...]
    proof_local_nonsevere_edge_count: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_index": self.boundary_index,
            "shift_units": self.shift_units,
            "replacement_unit_counts": [
                self.left_unit_count,
                self.right_unit_count,
            ],
            "canonical_master_seconds": self.canonical_master_seconds,
            "boundary_reasons": list(self.boundary_reasons),
            "trigger": self.trigger,
            "parent_profile_id": self.parent_profile_id,
            "parent_partition_signature_sha256": (
                self.parent_partition_signature_sha256
            ),
            "proof": {
                "failure": self.proof_failure,
                "failed_segment_index": self.proof_failed_segment_index,
                "forward_reachable_left_source_indices": list(
                    self.proof_forward_reachable_left_source_indices
                ),
                "left_valid_source_indices": list(
                    self.proof_left_valid_source_indices
                ),
                "right_valid_source_indices": list(
                    self.proof_right_valid_source_indices
                ),
                "local_nonsevere_cross_source_edge_count": (
                    self.proof_local_nonsevere_edge_count
                ),
                "constraint_infeasibility_proven": True,
            },
        }


@dataclass(frozen=True, slots=True)
class _DirectFullFrameProofDirectedWindowRedistribution:
    """Move two cuts in one three-span window around a proven frontier."""

    window_start_span_index: int
    proof_frontier_boundary_index: int
    proof_frontier_offset_in_window: int
    original_unit_counts: tuple[int, int, int]
    replacement_unit_counts: tuple[int, int, int]
    original_boundary_master_seconds: tuple[float, float]
    replacement_boundary_master_seconds: tuple[float, float]
    replacement_boundary_reasons: tuple[tuple[str, ...], tuple[str, ...]]
    trigger: str
    parent_profile_id: str
    parent_partition_signature_sha256: str
    proof_failure: str | None
    proof_failed_segment_index: int | None
    proof_forward_reachable_left_source_indices: tuple[int, ...]
    proof_left_valid_source_indices: tuple[int, ...]
    proof_right_valid_source_indices: tuple[int, ...]
    proof_local_nonsevere_edge_count: int

    @property
    def boundary_displacement_units(self) -> int:
        old_first, old_second, _old_third = self.original_unit_counts
        new_first, new_second, _new_third = self.replacement_unit_counts
        return abs(new_first - old_first) + abs(
            (new_first + new_second) - (old_first + old_second)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_start_span_index": self.window_start_span_index,
            "moved_boundary_indices": [
                self.window_start_span_index,
                self.window_start_span_index + 1,
            ],
            "proof_frontier_boundary_index": (
                self.proof_frontier_boundary_index
            ),
            "proof_frontier_offset_in_window": (
                self.proof_frontier_offset_in_window
            ),
            "original_unit_counts": list(self.original_unit_counts),
            "replacement_unit_counts": list(self.replacement_unit_counts),
            "original_boundary_master_seconds": list(
                self.original_boundary_master_seconds
            ),
            "replacement_boundary_master_seconds": list(
                self.replacement_boundary_master_seconds
            ),
            "replacement_boundary_reasons": [
                list(reasons) for reasons in self.replacement_boundary_reasons
            ],
            "boundary_displacement_units": self.boundary_displacement_units,
            "trigger": self.trigger,
            "parent_profile_id": self.parent_profile_id,
            "parent_partition_signature_sha256": (
                self.parent_partition_signature_sha256
            ),
            "proof": {
                "failure": self.proof_failure,
                "failed_segment_index": self.proof_failed_segment_index,
                "forward_reachable_left_source_indices": list(
                    self.proof_forward_reachable_left_source_indices
                ),
                "left_valid_source_indices": list(
                    self.proof_left_valid_source_indices
                ),
                "right_valid_source_indices": list(
                    self.proof_right_valid_source_indices
                ),
                "local_nonsevere_cross_source_edge_count": (
                    self.proof_local_nonsevere_edge_count
                ),
                "constraint_infeasibility_proven": True,
            },
            "safety_contract": {
                "window_total_duration_preserved": True,
                "cut_count_preserved": True,
                "existing_timing_boundaries_only": True,
                "visual_thresholds_relaxed": False,
                "all_three_spans_content_validity_recomputed": True,
                "all_three_spans_native_scene_cut_gate_recomputed": True,
                "balanced_source_quotas_recomputed": True,
                "full_selector_recheck_required": True,
            },
        }


@dataclass(frozen=True, slots=True)
class _DirectFullFrameVisualFrontierSplit:
    """One proof-bound added cut on an existing musical/action boundary."""

    parent_span_index: int
    split_unit_offset: int
    left_unit_count: int
    right_unit_count: int
    canonical_master_seconds: float
    boundary_reasons: tuple[str, ...]
    trigger: str
    parent_profile_id: str
    parent_partition_signature_sha256: str
    proof_failure: str | None
    proof_failed_segment_index: int | None
    proof_frontier_boundary_index: int
    proof_forward_reachable_left_source_indices: tuple[int, ...]
    proof_right_valid_source_indices: tuple[int, ...]
    evidence_contract: str
    evidence_kind: str
    evidence_source_indices: tuple[int, ...]
    evidence_master_seconds: tuple[float, ...]
    evidence_bracket_master_seconds: tuple[float, float] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_span_index": self.parent_span_index,
            "split_unit_offset": self.split_unit_offset,
            "replacement_unit_counts": [
                self.left_unit_count,
                self.right_unit_count,
            ],
            "canonical_master_seconds": self.canonical_master_seconds,
            "boundary_reasons": list(self.boundary_reasons),
            "trigger": self.trigger,
            "parent_profile_id": self.parent_profile_id,
            "parent_partition_signature_sha256": (
                self.parent_partition_signature_sha256
            ),
            "proof": {
                "failure": self.proof_failure,
                "failed_segment_index": self.proof_failed_segment_index,
                "frontier_boundary_index": self.proof_frontier_boundary_index,
                "forward_reachable_left_source_indices": list(
                    self.proof_forward_reachable_left_source_indices
                ),
                "right_valid_source_indices": list(
                    self.proof_right_valid_source_indices
                ),
                "constraint_infeasibility_proven": True,
            },
            "proposal_evidence": {
                "contract": self.evidence_contract,
                "kind": self.evidence_kind,
                "eligible_source_indices": list(self.evidence_source_indices),
                "master_seconds": list(self.evidence_master_seconds),
                "bracket_master_seconds": (
                    list(self.evidence_bracket_master_seconds)
                    if self.evidence_bracket_master_seconds is not None
                    else None
                ),
                "proposal_only_full_selector_recheck_required": True,
            },
        }


@dataclass(frozen=True, slots=True)
class _DirectFullFrameProofDirectedBoundaryMerge:
    """One proven-unmovable cut removed without relaxing a visual gate."""

    boundary_index: int
    left_unit_count: int
    right_unit_count: int
    merged_unit_count: int
    canonical_master_seconds: float
    boundary_reasons: tuple[str, ...]
    trigger: str
    parent_profile_id: str
    parent_partition_signature_sha256: str
    proof_failure: str | None
    proof_failed_segment_index: int | None
    proof_forward_reachable_left_source_indices: tuple[int, ...]
    proof_left_valid_source_indices: tuple[int, ...]
    proof_right_valid_source_indices: tuple[int, ...]
    proof_local_nonsevere_edge_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_index": self.boundary_index,
            "replaced_unit_counts": [
                self.left_unit_count,
                self.right_unit_count,
            ],
            "merged_unit_count": self.merged_unit_count,
            "canonical_master_seconds": self.canonical_master_seconds,
            "boundary_reasons": list(self.boundary_reasons),
            "trigger": self.trigger,
            "parent_profile_id": self.parent_profile_id,
            "parent_partition_signature_sha256": (
                self.parent_partition_signature_sha256
            ),
            "proof": {
                "failure": self.proof_failure,
                "failed_segment_index": self.proof_failed_segment_index,
                "forward_reachable_left_source_indices": list(
                    self.proof_forward_reachable_left_source_indices
                ),
                "left_valid_source_indices": list(
                    self.proof_left_valid_source_indices
                ),
                "right_valid_source_indices": list(
                    self.proof_right_valid_source_indices
                ),
                "local_nonsevere_cross_source_edge_count": (
                    self.proof_local_nonsevere_edge_count
                ),
                "constraint_infeasibility_proven": True,
            },
            "safety_contract": {
                "visual_thresholds_relaxed": False,
                "merged_span_content_validity_recomputed": True,
                "merged_span_native_scene_cut_gate_recomputed": True,
                "balanced_source_quotas_recomputed": True,
                "full_selector_recheck_required": True,
            },
        }


@dataclass(frozen=True, slots=True)
class _DirectFullFramePartitionProfile:
    profile_id: str
    unit_counts: tuple[int, ...]
    spans: tuple[_DirectFullFrameSpan, ...]
    boundary_shifts: tuple[_DirectFullFrameBoundaryShift, ...] = ()
    window_redistributions: tuple[
        _DirectFullFrameProofDirectedWindowRedistribution,
        ...,
    ] = ()
    visual_frontier_splits: tuple[
        _DirectFullFrameVisualFrontierSplit,
        ...,
    ] = ()
    boundary_merges: tuple[
        _DirectFullFrameProofDirectedBoundaryMerge,
        ...,
    ] = ()
    latest_operation_kind: str | None = None
    warm_start_binding_sha256: str | None = None
    warm_start_cache_file_sha256: str | None = None

    @property
    def shifted_boundary_index(self) -> int | None:
        return self.boundary_shifts[0].boundary_index if self.boundary_shifts else None

    @property
    def boundary_shift_units(self) -> int:
        return self.boundary_shifts[0].shift_units if self.boundary_shifts else 0

    @property
    def shifted_boundary_reasons(self) -> tuple[str, ...]:
        return (
            self.boundary_shifts[0].boundary_reasons
            if self.boundary_shifts
            else ()
        )

    @property
    def secondary_shifted_boundary_index(self) -> int | None:
        return (
            self.boundary_shifts[1].boundary_index
            if len(self.boundary_shifts) >= 2
            else None
        )

    @property
    def secondary_boundary_shift_units(self) -> int:
        return self.boundary_shifts[1].shift_units if len(self.boundary_shifts) >= 2 else 0

    @property
    def secondary_shifted_boundary_reasons(self) -> tuple[str, ...]:
        return (
            self.boundary_shifts[1].boundary_reasons
            if len(self.boundary_shifts) >= 2
            else ()
        )

    @property
    def secondary_shift_trigger(self) -> str | None:
        return self.boundary_shifts[1].trigger if len(self.boundary_shifts) >= 2 else None

    @property
    def latest_proof_operation(self) -> Any | None:
        """Return the explicitly recorded final lineage operation.

        Profile identifiers are diagnostic labels, not ancestry depths.  A
        previous implementation compared their string lengths, which could
        misidentify a short but newer boundary shift after a long merge label.
        Constructors now seal the operation kind directly.  The single-kind
        fallback keeps simple test/legacy profiles readable while refusing an
        ambiguous multi-operation lineage.
        """

        operations = {
            "boundary_shift": (
                self.boundary_shifts[-1] if self.boundary_shifts else None
            ),
            "window_redistribution": (
                self.window_redistributions[-1]
                if self.window_redistributions
                else None
            ),
            "visual_frontier_split": (
                self.visual_frontier_splits[-1]
                if self.visual_frontier_splits
                else None
            ),
            "boundary_merge": (
                self.boundary_merges[-1] if self.boundary_merges else None
            ),
        }
        if self.latest_operation_kind is not None:
            if self.latest_operation_kind not in operations:
                raise RuntimeError(
                    "direct partition profile has an unknown lineage operation"
                )
            operation = operations[self.latest_operation_kind]
            if operation is None:
                raise RuntimeError(
                    "direct partition profile lineage operation is missing"
                )
            return operation
        populated = tuple(
            operation for operation in operations.values() if operation is not None
        )
        if len(populated) <= 1:
            return populated[0] if populated else None
        raise RuntimeError(
            "direct partition profile has ambiguous unsealed lineage"
        )

    @property
    def parent_profile_id(self) -> str | None:
        operation = self.latest_proof_operation
        return operation.parent_profile_id if operation is not None else None

    @property
    def changed_span_count(self) -> int:
        changed_indices: set[int] = set()
        for shift in self.boundary_shifts:
            changed_indices.update((shift.boundary_index, shift.boundary_index + 1))
        for redistribution in self.window_redistributions:
            changed_indices.update(
                range(
                    redistribution.window_start_span_index,
                    redistribution.window_start_span_index + 3,
                )
            )
        for split in self.visual_frontier_splits:
            changed_indices.update(
                (split.parent_span_index, split.parent_span_index + 1)
            )
        for merge in self.boundary_merges:
            changed_indices.update(
                (merge.boundary_index, merge.boundary_index + 1)
            )
        return len(changed_indices)

    @property
    def structural_change_count(self) -> int:
        return (
            len(self.boundary_shifts)
            + 2 * len(self.window_redistributions)
            + len(self.visual_frontier_splits)
            + len(self.boundary_merges)
        )

    @property
    def proof_directed_boundary_motion_count(self) -> int:
        return len(self.boundary_shifts) + 2 * len(
            self.window_redistributions
        )

    @property
    def total_unit_displacement(self) -> int:
        return sum(
            abs(shift.shift_units) for shift in self.boundary_shifts
        ) + sum(
            redistribution.boundary_displacement_units
            for redistribution in self.window_redistributions
        )

    @property
    def musical_boundary_penalty(self) -> int:
        def _penalty(reasons: Sequence[str]) -> int:
            values = set(reasons)
            if "phrase" in values:
                return 0
            if "strong_beat" in values:
                return 1
            if "beat" in values:
                return 2
            return 3

        penalties = [
            _penalty(shift.boundary_reasons) for shift in self.boundary_shifts
        ]
        penalties.extend(
            _penalty(reasons)
            for redistribution in self.window_redistributions
            for reasons in redistribution.replacement_boundary_reasons
        )
        penalties.extend(
            _penalty(split.boundary_reasons)
            for split in self.visual_frontier_splits
        )
        penalties.extend(
            _penalty(merge.boundary_reasons)
            for merge in self.boundary_merges
        )
        return sum(penalties) if penalties else 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "partition_unit_counts": list(self.unit_counts),
            "partition_signature_sha256": hashlib.sha256(
                json.dumps(
                    list(self.unit_counts),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "coarse_span_count": len(self.spans),
            "shifted_boundary_index": self.shifted_boundary_index,
            "boundary_shift_units": self.boundary_shift_units,
            "changed_span_count": self.changed_span_count,
            "shifted_boundary_count": len(self.boundary_shifts),
            "window_redistribution_count": len(
                self.window_redistributions
            ),
            "proof_directed_boundary_motion_count": (
                self.proof_directed_boundary_motion_count
            ),
            "total_unit_displacement": self.total_unit_displacement,
            "visual_frontier_split_count": len(self.visual_frontier_splits),
            "proof_directed_boundary_merge_count": len(
                self.boundary_merges
            ),
            "structural_change_count": self.structural_change_count,
            "shifted_boundary_reasons": list(self.shifted_boundary_reasons),
            "secondary_shifted_boundary_index": (
                self.secondary_shifted_boundary_index
            ),
            "secondary_boundary_shift_units": (
                self.secondary_boundary_shift_units
            ),
            "secondary_shifted_boundary_reasons": list(
                self.secondary_shifted_boundary_reasons
            ),
            "secondary_shift_trigger": self.secondary_shift_trigger,
            "parent_profile_id": self.parent_profile_id,
            "latest_operation_kind": self.latest_operation_kind,
            "input_bound_partition_warm_start": (
                {
                    "contract": _DIRECT_MEMBER_PARTITION_WARM_START_CONTRACT,
                    "binding_sha256": self.warm_start_binding_sha256,
                    "cache_file_sha256": self.warm_start_cache_file_sha256,
                    "full_selector_recheck_required": True,
                    "visual_gates_relaxed": False,
                }
                if self.warm_start_binding_sha256 is not None
                else None
            ),
            "proof_directed_boundary_shifts": [
                shift.to_dict() for shift in self.boundary_shifts
            ],
            "proof_directed_window_redistributions": [
                redistribution.to_dict()
                for redistribution in self.window_redistributions
            ],
            "proof_directed_visual_frontier_splits": [
                split.to_dict() for split in self.visual_frontier_splits
            ],
            "proof_directed_boundary_merges": [
                merge.to_dict() for merge in self.boundary_merges
            ],
            "musical_boundary_penalty": self.musical_boundary_penalty,
            "minimum_span_seconds": min(
                span.duration_seconds for span in self.spans
            ),
            "median_span_seconds": median(
                span.duration_seconds for span in self.spans
            ),
            "maximum_span_seconds": max(
                span.duration_seconds for span in self.spans
            ),
        }


def _direct_full_frame_spans_from_partition(
    timing_segments: Sequence[DraftSegment],
    unit_counts: Sequence[int],
) -> tuple[_DirectFullFrameSpan, ...]:
    """Materialize one contiguous musical partition without moving range edges."""

    items = tuple(timing_segments)
    counts = tuple(int(value) for value in unit_counts)
    if not items or not counts or any(value <= 0 for value in counts):
        raise ValueError("direct full-frame partition must contain positive units")
    if sum(counts) != len(items):
        raise ValueError("direct full-frame partition must consume every timing unit")
    spans: list[_DirectFullFrameSpan] = []
    cursor = 0
    for count in counts:
        group = items[cursor : cursor + count]
        duration = (
            group[-1].canonical_master_end
            - group[0].canonical_master_start
        )
        if duration <= 0.0 or duration > _DIRECT_MEMBER_MAX_RUN_SECONDS + 1e-9:
            raise ValueError(
                "direct full-frame partition violates the 7.2-second hold bound"
            )
        spans.append(
            _DirectFullFrameSpan(
                canonical_master_start=group[0].canonical_master_start,
                canonical_master_end=group[-1].canonical_master_end,
                timing_segment_count=count,
                estimated_beat_count=sum(
                    item.estimated_beat_count for item in group
                ),
                start_boundary_reasons=group[0].start_boundary_reasons,
                end_boundary_reasons=group[-1].end_boundary_reasons,
            )
        )
        cursor += count
    return tuple(spans)


def _coarsen_direct_full_frame_spans(
    timing_segments: Sequence[DraftSegment],
) -> tuple[_DirectFullFrameSpan, ...]:
    """Merge the one-strong-beat timing draft into reference-like shot holds.

    The metadata draft often exposes a boundary roughly every two detected
    tempo beats.  Four such units are therefore the stable default (about eight
    tempo beats / 3.9 seconds for IVE ``I AM``).  Two- and six-unit alternatives
    remain available when a phrase or strong-beat boundary is materially closer
    to the calibrated target.  A final short remainder is merged whenever the
    7.2-second hard maximum permits it.
    """

    items = tuple(timing_segments)
    if not items:
        raise ValueError("direct full-frame timing draft cannot be empty")
    spans: list[_DirectFullFrameSpan] = []
    cursor = 0
    while cursor < len(items):
        remaining = len(items) - cursor
        candidate_counts = [
            count for count in (2, 4, 6) if count <= remaining
        ]
        if remaining not in candidate_counts:
            candidate_counts.append(remaining)
        candidates: list[tuple[tuple[float, float, int], int]] = []
        for count in candidate_counts:
            end_index = cursor + count - 1
            duration = (
                items[end_index].canonical_master_end
                - items[cursor].canonical_master_start
            )
            if duration <= 0.0 or duration > _DIRECT_MEMBER_MAX_RUN_SECONDS + 1e-9:
                continue
            reasons = set(items[end_index].end_boundary_reasons)
            boundary_bonus = (
                0.18
                if "phrase" in reasons
                else 0.08
                if "strong_beat" in reasons
                else 0.0
            )
            candidates.append(
                (
                    (
                        abs(duration - _DIRECT_MEMBER_TARGET_RUN_SECONDS)
                        - boundary_bonus,
                        abs(float(count) - 4.0),
                        count,
                    ),
                    count,
                )
            )
        if not candidates:
            raise RuntimeError(
                "direct full-frame cadence cannot satisfy the 7.2-second hold bound"
            )
        count = min(candidates)[1]
        group = items[cursor : cursor + count]
        spans.append(
            _DirectFullFrameSpan(
                canonical_master_start=group[0].canonical_master_start,
                canonical_master_end=group[-1].canonical_master_end,
                timing_segment_count=len(group),
                estimated_beat_count=sum(
                    item.estimated_beat_count for item in group
                ),
                start_boundary_reasons=group[0].start_boundary_reasons,
                end_boundary_reasons=group[-1].end_boundary_reasons,
            )
        )
        cursor += count

    if len(spans) > 1 and spans[-1].duration_seconds < 2.0:
        left = spans[-2]
        right = spans[-1]
        merged_duration = right.canonical_master_end - left.canonical_master_start
        if merged_duration <= _DIRECT_MEMBER_MAX_RUN_SECONDS + 1e-9:
            spans[-2:] = [
                _DirectFullFrameSpan(
                    canonical_master_start=left.canonical_master_start,
                    canonical_master_end=right.canonical_master_end,
                    timing_segment_count=(
                        left.timing_segment_count + right.timing_segment_count
                    ),
                    estimated_beat_count=(
                        left.estimated_beat_count + right.estimated_beat_count
                    ),
                    start_boundary_reasons=left.start_boundary_reasons,
                    end_boundary_reasons=right.end_boundary_reasons,
                )
            ]
    return tuple(spans)


def _normalized_direct_member_highlight_intervals(
    intervals: Sequence[tuple[float, float]],
    *,
    range_start: float,
    range_end: float,
) -> tuple[tuple[float, float], ...]:
    """Clip and merge the section plan into one deterministic time union."""

    if not math.isfinite(range_start) or not math.isfinite(range_end):
        raise ValueError("direct-member highlight range must be finite")
    if range_end <= range_start:
        raise ValueError("direct-member highlight range must be positive")
    clipped: list[tuple[float, float]] = []
    for raw_start, raw_end in intervals:
        start = float(raw_start)
        end = float(raw_end)
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError(
                "direct-member highlight intervals must be finite and positive"
            )
        start = max(range_start, start)
        end = min(range_end, end)
        if end > start + 1e-9:
            clipped.append((start, end))
    if not clipped:
        return ()
    clipped.sort()
    merged: list[tuple[float, float]] = [clipped[0]]
    for start, end in clipped[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + 1e-7:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _direct_member_interval_overlap_seconds(
    start: float,
    end: float,
    intervals: Sequence[tuple[float, float]],
) -> float:
    return sum(
        max(0.0, min(end, interval_end) - max(start, interval_start))
        for interval_start, interval_end in intervals
    )


def _direct_member_boundary_is_highlight(
    boundary_seconds: float,
    intervals: Sequence[tuple[float, float]],
) -> bool:
    return any(
        start - 1e-7 <= boundary_seconds <= end + 1e-7
        for start, end in intervals
    )


def _direct_member_highlight_priority_boundary_indices(
    spans: Sequence[_DirectFullFrameSpan],
    intervals: Sequence[tuple[float, float]],
) -> tuple[int, ...]:
    """Return section-critical cuts governed by the compensated-hold budget."""

    items = tuple(spans)
    if len(items) < 2 or not intervals:
        return ()
    return tuple(
        boundary_index
        for boundary_index, span in enumerate(items[:-1])
        if _direct_member_boundary_is_highlight(
            span.canonical_master_end,
            intervals,
        )
    )


def _densify_direct_full_frame_highlights(
    timing_segments: Sequence[DraftSegment],
    base_spans: Sequence[_DirectFullFrameSpan],
    highlight_intervals: Sequence[tuple[float, float]],
) -> tuple[_DirectFullFrameSpan, ...]:
    """Add musically bound highlight cuts without weakening visual gates.

    The reference cadence remains roughly four timing units outside active
    sections.  Inside BUILD/CHORUS-like intervals, long holds are split only
    at boundaries already emitted by ``build_group_draft``. A one-unit child
    is eligible only when its real duration still reaches the 1.8-second floor.
    The global target is deliberately bounded below the 19.5/min hard ceiling;
    the visual selector must still independently approve every added cut.
    """

    items = tuple(timing_segments)
    baseline = tuple(base_spans)
    if not items or not baseline or not highlight_intervals:
        return baseline
    normalized = _normalized_direct_member_highlight_intervals(
        highlight_intervals,
        range_start=items[0].canonical_master_start,
        range_end=items[-1].canonical_master_end,
    )
    if not normalized:
        return baseline
    total_duration = (
        items[-1].canonical_master_end - items[0].canonical_master_start
    )
    target_span_count = min(
        math.floor(
            total_duration
            * _DIRECT_MEMBER_INITIAL_MAX_SWITCHES_PER_MINUTE
            / 60.0
        )
        + 1,
        max(
            len(baseline),
            math.ceil(
                total_duration
                * _DIRECT_MEMBER_HIGHLIGHT_DENSE_SWITCHES_PER_MINUTE
                / 60.0
            )
            + 1,
        ),
    )
    counts = [span.timing_segment_count for span in baseline]
    while len(counts) < target_span_count:
        candidates: list[
            tuple[
                tuple[float, float, int, float, int, int],
                int,
                int,
            ]
        ] = []
        unit_start = 0
        for span_index, count in enumerate(counts):
            if count < 2:
                unit_start += count
                continue
            group_start = items[unit_start].canonical_master_start
            group_end = items[unit_start + count - 1].canonical_master_end
            group_duration = group_end - group_start
            overlap = _direct_member_interval_overlap_seconds(
                group_start,
                group_end,
                normalized,
            )
            overlap_fraction = overlap / group_duration
            if overlap_fraction < 0.50:
                unit_start += count
                continue
            for split_offset in range(1, count):
                boundary_unit = items[unit_start + split_offset - 1]
                boundary_seconds = boundary_unit.canonical_master_end
                if not _direct_member_boundary_is_highlight(
                    boundary_seconds,
                    normalized,
                ):
                    continue
                left_duration = boundary_seconds - group_start
                right_duration = group_end - boundary_seconds
                if (
                    min(left_duration, right_duration)
                    + 1e-9
                    < _DIRECT_MEMBER_HIGHLIGHT_MINIMUM_RUN_SECONDS
                ):
                    continue
                reasons = set(boundary_unit.end_boundary_reasons)
                musical_penalty = (
                    0
                    if "phrase" in reasons
                    else 1
                    if "strong_beat" in reasons
                    else 2
                    if "beat" in reasons
                    else 3
                )
                target_errors = sorted(
                    (
                        abs(
                            left_duration
                            - _DIRECT_MEMBER_HIGHLIGHT_TARGET_RUN_SECONDS
                        ),
                        abs(
                            right_duration
                            - _DIRECT_MEMBER_HIGHLIGHT_TARGET_RUN_SECONDS
                        ),
                    )
                )
                candidates.append(
                    (
                        (
                            -overlap_fraction,
                            target_errors[0],
                            musical_penalty,
                            target_errors[1],
                            span_index,
                            split_offset,
                        ),
                        span_index,
                        split_offset,
                    )
                )
            unit_start += count
        if not candidates:
            break
        _score, span_index, split_offset = min(candidates)
        original_count = counts[span_index]
        counts[span_index : span_index + 1] = [
            split_offset,
            original_count - split_offset,
        ]
    return _direct_full_frame_spans_from_partition(items, counts)


def _direct_full_frame_base_partition_profile(
    timing_segments: Sequence[DraftSegment],
    *,
    highlight_intervals: Sequence[tuple[float, float]] = (),
) -> _DirectFullFramePartitionProfile:
    baseline = _coarsen_direct_full_frame_spans(timing_segments)
    spans = _densify_direct_full_frame_highlights(
        timing_segments,
        baseline,
        highlight_intervals,
    )
    return _DirectFullFramePartitionProfile(
        profile_id=(
            "highlight-dense-reference-coarsening-v1"
            if len(spans) > len(baseline)
            else "base-reference-coarsening"
        ),
        unit_counts=tuple(span.timing_segment_count for span in spans),
        spans=spans,
    )


def _direct_partition_failure_bottlenecks(
    diagnostics: dict[str, Any],
    *,
    span_count: int,
) -> tuple[int, ...]:
    """Return proven fixed-boundary bottlenecks, final boundary first."""

    if span_count < 2:
        return ()
    candidates: list[int] = []
    connectivity = diagnostics.get("hard_boundary_connectivity", {})
    if isinstance(connectivity, dict):
        raw_zero = connectivity.get(
            "zero_local_nonsevere_transition_boundary_indices"
        )
        if raw_zero is None:
            # Compatibility for diagnostics written before localized hard-edge
            # counts were added. New runs never use global reachability zeros
            # to localize a partition shift.
            raw_zero = connectivity.get(
                "zero_reachable_transition_boundary_indices",
                (),
            )
        if isinstance(raw_zero, list):
            candidates.extend(
                int(value)
                for value in raw_zero
                if isinstance(value, int) and 0 <= value < span_count - 1
            )

    search = diagnostics.get("search", {})
    if isinstance(search, dict):
        exact = search.get("exact_lifeline", {})
        if isinstance(exact, dict):
            roots = exact.get("root_candidate_diagnostics", ())
            if (
                isinstance(roots, list)
                and roots
                and all(
                    isinstance(root, dict)
                    and root.get("search_outcome")
                    == "root_predecessor_exhausted"
                    for root in roots
                )
            ):
                candidates.append(span_count - 2)

    unique = sorted(
        set(candidates),
        key=lambda boundary: (
            boundary != span_count - 2,
            -boundary,
        ),
    )
    return tuple(unique[:_DIRECT_MEMBER_MAX_BOTTLENECK_BOUNDARY_COUNT])


def _direct_partition_failure_is_proven(
    diagnostics: dict[str, Any],
) -> bool:
    if diagnostics.get("failure") == "search_capacity_exhausted":
        return False
    if diagnostics.get("constraint_infeasibility_proven") is True:
        return True
    return diagnostics.get("failure") in {
        "content_validity_quota_precheck_failed",
        "boundary_diagnostic_eligible_pair_exhausted",
        "validity_quota_path_exhausted",
        "duration_balance_threshold_unsatisfied",
    }


def _direct_partition_failure_result(
    diagnostics: Mapping[str, Any],
) -> str:
    """Name a selector failure without calling an unproved result hard."""

    if _direct_partition_failure_is_proven(dict(diagnostics)):
        return "hard_infeasible"
    if diagnostics.get("failure") == "search_capacity_exhausted":
        return "capacity_exhausted"
    return "unproven_failure"


def _direct_partition_aggregate_outcome(
    statuses: Sequence[str],
    *,
    profile_cap_reached: bool,
) -> tuple[str, str, dict[str, int]]:
    """Summarize a bounded search without upgrading uncertainty to proof."""

    allowed = (
        "hard_infeasible",
        "capacity_exhausted",
        "unproven_failure",
    )
    items = tuple(str(status) for status in statuses)
    if any(status not in allowed for status in items):
        raise ValueError("direct partition aggregate received an unknown status")
    counts = {
        outcome: sum(status == outcome for status in items)
        for outcome in allowed
    }
    if profile_cap_reached:
        return (
            "direct_partition_search_capacity_exhausted",
            "profile_cap_reached",
            counts,
        )
    if counts["capacity_exhausted"]:
        return (
            "direct_partition_selector_capacity_exhausted",
            "selector_capacity_exhausted",
            counts,
        )
    if counts["unproven_failure"]:
        return (
            "direct_partition_unproven_failure",
            "unproven_failure",
            counts,
        )
    return (
        "direct_partition_profiles_exhausted",
        "bounded_hard_infeasible",
        counts,
    )


def _direct_counterfactual_secondary_signature_action(
    signature: tuple[int, ...],
    *,
    evaluated_signatures: set[tuple[int, ...]],
    queued_signatures: set[tuple[int, ...]],
    reserved_signatures: set[tuple[int, ...]],
    seen_signatures: set[tuple[int, ...]],
) -> str:
    """Decide whether a bounded secondary candidate is new or promoted.

    A canonical partition already waiting in the primary queue is promoted to
    the reserved +10 pass rather than discarded.  That distinction matters
    when the primary queue later reaches its static cap.
    """

    if signature in evaluated_signatures or signature in reserved_signatures:
        return "duplicate"
    if signature in queued_signatures:
        return "promote_queued"
    if signature in seen_signatures:
        return "duplicate"
    return "new"


@dataclass(frozen=True, slots=True)
class _DirectPartitionBoundaryProof:
    boundary_index: int
    trigger: str
    failure: str | None
    failed_segment_index: int | None
    forward_reachable_left_source_indices: tuple[int, ...]
    left_valid_source_indices: tuple[int, ...]
    right_valid_source_indices: tuple[int, ...]
    local_nonsevere_edge_count: int


def _direct_partition_exact_boundary_proof(
    diagnostics: dict[str, Any],
    *,
    boundary_index: int,
    trigger: str,
    span_count: int,
) -> _DirectPartitionBoundaryProof | None:
    """Bind a shift target to exact, scalar connectivity evidence."""

    if not _direct_partition_failure_is_proven(diagnostics):
        return None
    connectivity = diagnostics.get("hard_boundary_connectivity")
    if not isinstance(connectivity, dict) or connectivity.get(
        "analysis_basis"
    ) != "content_valid_nonsevere_no_repeat_reachability_v1":
        return None
    raw_boundaries = connectivity.get("boundaries")
    if not isinstance(raw_boundaries, list) or len(raw_boundaries) != span_count - 1:
        return None
    if not 0 <= boundary_index < len(raw_boundaries):
        return None
    item = raw_boundaries[boundary_index]
    if not isinstance(item, dict) or item.get("boundary_index") != boundary_index:
        return None
    left_valid = item.get("left_valid_source_indices")
    right_valid = item.get("right_valid_source_indices")
    forward = item.get("forward_reachable_left_source_indices")
    local_edges = item.get("nonsevere_cross_source_edge_count")
    if (
        not isinstance(left_valid, list)
        or not left_valid
        or not isinstance(right_valid, list)
        or not right_valid
        or not isinstance(forward, list)
        or not isinstance(local_edges, int)
        or local_edges < 0
    ):
        # Moving a cut cannot be used to conceal an empty/invalid content side.
        return None
    if trigger == "zero_local_nonsevere_transition_edges":
        raw_zero = connectivity.get(
            "zero_local_nonsevere_transition_boundary_indices"
        )
        if (
            not isinstance(raw_zero, list)
            or boundary_index not in raw_zero
            or local_edges != 0
        ):
            return None
    elif trigger == "proven_forward_connectivity_frontier_exhaustion":
        if local_edges <= 0 or _direct_partition_forward_exhaustion_bottleneck(
            diagnostics,
            span_count=span_count,
        ) != boundary_index:
            return None
    else:
        return None
    return _DirectPartitionBoundaryProof(
        boundary_index=boundary_index,
        trigger=trigger,
        failure=(
            str(diagnostics["failure"])
            if diagnostics.get("failure") is not None
            else None
        ),
        failed_segment_index=(
            int(diagnostics["failed_at_segment_index"])
            if isinstance(diagnostics.get("failed_at_segment_index"), int)
            else None
        ),
        forward_reachable_left_source_indices=tuple(int(value) for value in forward),
        left_valid_source_indices=tuple(int(value) for value in left_valid),
        right_valid_source_indices=tuple(int(value) for value in right_valid),
        local_nonsevere_edge_count=local_edges,
    )


def _direct_partition_exact_shift_proofs(
    diagnostics: dict[str, Any],
    *,
    span_count: int,
) -> tuple[_DirectPartitionBoundaryProof, ...]:
    """Return deterministic, independently proven targets for one parent."""

    connectivity = diagnostics.get("hard_boundary_connectivity")
    if not isinstance(connectivity, dict):
        return ()
    raw_zero = connectivity.get(
        "zero_local_nonsevere_transition_boundary_indices"
    )
    local_indices = (
        sorted(
            {
                int(value)
                for value in raw_zero
                if isinstance(value, int) and 0 <= value < span_count - 1
            },
            reverse=True,
        )
        if isinstance(raw_zero, list)
        else []
    )
    proofs: list[_DirectPartitionBoundaryProof] = []
    for boundary_index in local_indices:
        proof = _direct_partition_exact_boundary_proof(
            diagnostics,
            boundary_index=boundary_index,
            trigger="zero_local_nonsevere_transition_edges",
            span_count=span_count,
        )
        if proof is not None:
            proofs.append(proof)
        if len(proofs) >= _DIRECT_MEMBER_MAX_BOTTLENECK_BOUNDARY_COUNT:
            return tuple(proofs)
    if proofs:
        return tuple(proofs)
    frontier = _direct_partition_forward_exhaustion_bottleneck(
        diagnostics,
        span_count=span_count,
    )
    if frontier is None:
        return ()
    proof = _direct_partition_exact_boundary_proof(
        diagnostics,
        boundary_index=frontier,
        trigger="proven_forward_connectivity_frontier_exhaustion",
        span_count=span_count,
    )
    return (proof,) if proof is not None else ()


def _direct_partition_signature_sha256(unit_counts: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps(
            [int(value) for value in unit_counts],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _direct_member_partition_warm_start_binding(
    feature_cache_directory: Path | None,
    base: _DirectFullFramePartitionProfile,
    *,
    source_count: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Bind reusable cadence candidates to one exact acquired source set.

    A cached cadence is never a visual-selection result.  It only decides
    which existing musical partition receives an early, complete selector
    audition.  Binding it to the acquisition manifest and base partition
    avoids wasting that audition after sources or timing have changed.
    """

    diagnostics: dict[str, Any] = {
        "contract": _DIRECT_MEMBER_PARTITION_WARM_START_CONTRACT,
        "requested": feature_cache_directory is not None,
        "cache_filename": _DIRECT_MEMBER_PARTITION_WARM_START_FILENAME,
        "full_selector_recheck_required": True,
        "visual_gates_relaxed": False,
    }
    if feature_cache_directory is None:
        diagnostics["status"] = "feature_cache_unavailable"
        return None, diagnostics
    if isinstance(source_count, bool) or source_count <= 0:
        raise ValueError("direct member warm-start source count must be positive")
    manifest_path = Path(feature_cache_directory) / "acquisition-manifest.json"
    try:
        _manifest_bytes, manifest_fingerprint = _read_stable_regular_file(
            manifest_path
        )
    except (OSError, ValueError) as exc:
        diagnostics.update(
            {
                "status": "acquisition_manifest_unavailable",
                "rejection": type(exc).__name__,
            }
        )
        return None, diagnostics
    binding_payload = {
        "contract": _DIRECT_MEMBER_PARTITION_WARM_START_CONTRACT,
        "acquisition_manifest_sha256": manifest_fingerprint["sha256"],
        "base_partition_signature_sha256": (
            _direct_partition_signature_sha256(base.unit_counts)
        ),
        "timing_segment_count": sum(base.unit_counts),
        "source_count": int(source_count),
    }
    binding_sha256 = hashlib.sha256(
        json.dumps(
            binding_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    binding = {**binding_payload, "binding_sha256": binding_sha256}
    diagnostics.update(
        {
            "status": "binding_ready",
            "acquisition_manifest_sha256": manifest_fingerprint["sha256"],
            "base_partition_signature_sha256": (
                binding_payload["base_partition_signature_sha256"]
            ),
            "binding_sha256": binding_sha256,
        }
    )
    return binding, diagnostics


def _load_direct_member_partition_warm_start_profiles(
    feature_cache_directory: Path | None,
    timing_segments: Sequence[DraftSegment],
    base: _DirectFullFramePartitionProfile,
    *,
    source_count: int,
) -> tuple[tuple[_DirectFullFramePartitionProfile, ...], dict[str, Any]]:
    """Load exact-input-bound cadence candidates for a full visual recheck."""

    binding, diagnostics = _direct_member_partition_warm_start_binding(
        feature_cache_directory,
        base,
        source_count=source_count,
    )
    diagnostics.update(
        {
            "entry_count": 0,
            "matching_entry_count": 0,
            "loaded_candidate_count": 0,
            "rejected_entry_count": 0,
            "rejection_reason_counts": {},
        }
    )
    if binding is None or feature_cache_directory is None:
        return (), diagnostics
    cache_path = (
        Path(feature_cache_directory)
        / _DIRECT_MEMBER_PARTITION_WARM_START_FILENAME
    )
    if not cache_path.exists():
        diagnostics["status"] = "cache_missing"
        return (), diagnostics
    try:
        encoded, cache_fingerprint = _read_stable_regular_file(cache_path)
    except (OSError, ValueError) as exc:
        diagnostics.update(
            {
                "status": "cache_rejected",
                "rejection": type(exc).__name__,
            }
        )
        return (), diagnostics
    if len(encoded) > _DIRECT_MEMBER_PARTITION_WARM_START_MAX_BYTES:
        diagnostics.update(
            {
                "status": "cache_rejected",
                "rejection": "cache_size_limit_exceeded",
            }
        )
        return (), diagnostics
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        diagnostics.update(
            {"status": "cache_rejected", "rejection": "invalid_json"}
        )
        return (), diagnostics
    if not (
        isinstance(payload, dict)
        and payload.get("schema_version") == 1
        and payload.get("contract")
        == _DIRECT_MEMBER_PARTITION_WARM_START_CONTRACT
        and isinstance(payload.get("entries"), list)
        and len(payload["entries"])
        <= _DIRECT_MEMBER_PARTITION_WARM_START_MAX_ENTRY_COUNT
    ):
        diagnostics.update(
            {"status": "cache_rejected", "rejection": "schema_mismatch"}
        )
        return (), diagnostics

    entries = payload["entries"]
    diagnostics["entry_count"] = len(entries)
    rejection_reasons: dict[str, int] = {}

    def _reject(reason: str) -> None:
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    items = tuple(timing_segments)
    candidates: list[_DirectFullFramePartitionProfile] = []
    seen_signatures = {base.unit_counts}
    for entry in entries:
        if not isinstance(entry, dict):
            _reject("entry_not_object")
            continue
        if any(entry.get(key) != value for key, value in binding.items()):
            _reject("input_binding_mismatch")
            continue
        diagnostics["matching_entry_count"] += 1
        raw_counts = entry.get("partition_unit_counts")
        if not (
            isinstance(raw_counts, list)
            and len(raw_counts) >= source_count
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                for value in raw_counts
            )
        ):
            _reject("invalid_partition_unit_counts")
            continue
        counts = tuple(raw_counts)
        signature_sha256 = _direct_partition_signature_sha256(counts)
        if entry.get("candidate_partition_signature_sha256") != signature_sha256:
            _reject("candidate_signature_mismatch")
            continue
        if counts in seen_signatures:
            _reject("duplicate_partition")
            continue
        try:
            spans = _direct_full_frame_spans_from_partition(items, counts)
        except ValueError:
            _reject("timing_partition_mismatch")
            continue
        duration = sum(span.duration_seconds for span in spans)
        switches_per_minute = (len(spans) - 1) * 60.0 / duration
        span_median = median(span.duration_seconds for span in spans)
        if not (
            _DIRECT_MEMBER_MIN_SWITCHES_PER_MINUTE
            <= switches_per_minute
            <= _DIRECT_MEMBER_MAX_SWITCHES_PER_MINUTE
            and _DIRECT_MEMBER_MIN_MEDIAN_RUN_SECONDS
            <= span_median
            <= _DIRECT_MEMBER_MAX_MEDIAN_RUN_SECONDS
            and max(span.duration_seconds for span in spans)
            <= _DIRECT_MEMBER_MAX_RUN_SECONDS + 1e-9
        ):
            _reject("cadence_gate_failed")
            continue
        candidates.append(
            _DirectFullFramePartitionProfile(
                profile_id=f"input-bound-warm-start-{signature_sha256[:16]}",
                unit_counts=counts,
                spans=spans,
                warm_start_binding_sha256=binding["binding_sha256"],
                warm_start_cache_file_sha256=cache_fingerprint["sha256"],
            )
        )
        seen_signatures.add(counts)
        if len(candidates) >= _DIRECT_MEMBER_PARTITION_WARM_START_MAX_MATCH_COUNT:
            break
    diagnostics.update(
        {
            "status": "candidates_loaded" if candidates else "no_match",
            "cache_file_sha256": cache_fingerprint["sha256"],
            "loaded_candidate_count": len(candidates),
            "loaded_candidate_partition_signatures_sha256": [
                _direct_partition_signature_sha256(profile.unit_counts)
                for profile in candidates
            ],
            "rejected_entry_count": sum(rejection_reasons.values()),
            "rejection_reason_counts": rejection_reasons,
        }
    )
    return tuple(candidates), diagnostics


def _persist_direct_member_partition_warm_start_profile(
    feature_cache_directory: Path | None,
    base: _DirectFullFramePartitionProfile,
    selected: _DirectFullFramePartitionProfile,
    *,
    source_count: int,
) -> dict[str, Any]:
    """Persist a fully selected cadence as a future exact-bound candidate."""

    binding, diagnostics = _direct_member_partition_warm_start_binding(
        feature_cache_directory,
        base,
        source_count=source_count,
    )
    diagnostics["persisted"] = False
    if binding is None or feature_cache_directory is None:
        return diagnostics
    cache_path = (
        Path(feature_cache_directory)
        / _DIRECT_MEMBER_PARTITION_WARM_START_FILENAME
    )
    entries: list[dict[str, Any]] = []
    if cache_path.exists():
        try:
            encoded, _fingerprint = _read_stable_regular_file(cache_path)
            existing = json.loads(encoded.decode("utf-8"))
        except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            diagnostics.update(
                {
                    "status": "existing_cache_rejected",
                    "rejection": type(exc).__name__,
                }
            )
            return diagnostics
        if not (
            isinstance(existing, dict)
            and existing.get("schema_version") == 1
            and existing.get("contract")
            == _DIRECT_MEMBER_PARTITION_WARM_START_CONTRACT
            and isinstance(existing.get("entries"), list)
            and len(existing["entries"])
            <= _DIRECT_MEMBER_PARTITION_WARM_START_MAX_ENTRY_COUNT
        ):
            diagnostics.update(
                {
                    "status": "existing_cache_rejected",
                    "rejection": "schema_mismatch",
                }
            )
            return diagnostics
        entries = [item for item in existing["entries"] if isinstance(item, dict)]
    candidate_signature = _direct_partition_signature_sha256(
        selected.unit_counts
    )
    entries = [
        item
        for item in entries
        if not (
            item.get("binding_sha256") == binding["binding_sha256"]
            and item.get("candidate_partition_signature_sha256")
            == candidate_signature
        )
    ]
    entries.insert(
        0,
        {
            **binding,
            "candidate_partition_signature_sha256": candidate_signature,
            "partition_unit_counts": list(selected.unit_counts),
            "origin": "full_visual_selector_feasible",
            "full_selector_recheck_required_on_reuse": True,
            "visual_gates_relaxed": False,
        },
    )
    payload = {
        "schema_version": 1,
        "contract": _DIRECT_MEMBER_PARTITION_WARM_START_CONTRACT,
        "entries": entries[:_DIRECT_MEMBER_PARTITION_WARM_START_MAX_ENTRY_COUNT],
    }
    try:
        _write_analysis_manifest_atomically(cache_path, payload)
        _persisted, fingerprint = _read_stable_regular_file(cache_path)
    except (OSError, ValueError) as exc:
        diagnostics.update(
            {"status": "persistence_failed", "rejection": type(exc).__name__}
        )
        return diagnostics
    diagnostics.update(
        {
            "status": "persisted",
            "persisted": True,
            "candidate_partition_signature_sha256": candidate_signature,
            "cache_file_sha256": fingerprint["sha256"],
        }
    )
    return diagnostics


def _direct_partition_queue_signature_digest(
    profiles: Sequence[_DirectFullFramePartitionProfile],
) -> str:
    """Hash one queue frontier independently of insertion order."""

    signatures = sorted(
        _direct_partition_signature_sha256(profile.unit_counts)
        for profile in profiles
    )
    if len(signatures) != len(set(signatures)):
        raise ValueError("direct partition queue contains duplicate signatures")
    return hashlib.sha256(
        json.dumps(
            signatures,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _direct_partition_current_boundary_index_at_time(
    profile: _DirectFullFramePartitionProfile,
    canonical_master_seconds: float,
) -> int | None:
    """Resolve a lineage cut in the current partition's coordinate system."""

    if not math.isfinite(canonical_master_seconds):
        raise ValueError("direct partition lineage cut time must be finite")
    for boundary_index, span in enumerate(profile.spans[:-1]):
        if math.isclose(
            span.canonical_master_end,
            canonical_master_seconds,
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            return boundary_index
    return None


def _direct_partition_current_shifted_boundary_indices(
    profile: _DirectFullFramePartitionProfile,
) -> tuple[int, ...]:
    """Map every surviving moved cut after any index-changing merge."""

    return tuple(
        sorted(
            {
                boundary_index
                for canonical_master_seconds in (
                    tuple(
                        shift.canonical_master_seconds
                        for shift in profile.boundary_shifts
                    )
                    + tuple(
                        boundary_seconds
                        for redistribution in profile.window_redistributions
                        for boundary_seconds in (
                            redistribution.replacement_boundary_master_seconds
                        )
                    )
                )
                if (
                    boundary_index
                    := _direct_partition_current_boundary_index_at_time(
                        profile,
                        canonical_master_seconds,
                    )
                )
                is not None
            }
        )
    )


def _direct_partition_proof_directed_boundary_merge_limit(
    profile: _DirectFullFramePartitionProfile,
) -> int:
    """Derive the total merge budget from the professional density floor.

    Every admitted merge removes exactly one potential source switch.  The
    budget therefore stops at the first integer switch count that would fall
    below the 15-per-minute professional floor for this exact song range.  It
    is a search-capacity calculation only: materialized children must still
    pass the 7.2-second hold bound, median cadence, content, transition,
    source-balance, highlight, and rendered-continuity gates independently.
    """

    if not profile.spans:
        return len(profile.boundary_merges)
    duration = sum(span.duration_seconds for span in profile.spans)
    minimum_switch_count = math.ceil(
        _DIRECT_MEMBER_MIN_SWITCHES_PER_MINUTE * duration / 60.0 - 1e-9
    )
    current_switch_count = len(profile.spans) - 1
    remaining_density_capacity = max(
        0,
        current_switch_count - minimum_switch_count,
    )
    return len(profile.boundary_merges) + remaining_density_capacity


def _direct_partition_proof_directed_boundary_shift_limit(
    profile: _DirectFullFramePartitionProfile,
) -> int:
    """Grant one compensating proof-directed cut motion per admitted merge.

    The base seven-shift bound limits ordinary search branching.  A merge is
    different: it removes the proved cut and can expose either neighbouring
    cut as the next exact connectivity frontier.  Keeping the old fixed bound
    after such a merge made a saturated lineage jump directly to another
    merge or a visual split even when a cadence-preserving boundary shift was
    still available.  One additional moved-cut slot per independently admitted
    merge covers that newly exposed local frontier without relaxing any visual
    or cadence gate.  A three-span redistribution consumes two slots.  The
    number of surviving cuts remains an absolute bound.
    """

    return min(
        max(0, len(profile.spans) - 1),
        _DIRECT_MEMBER_MAX_SHIFTED_BOUNDARY_COUNT
        + len(profile.boundary_merges),
    )


def _direct_partition_future_operation_state_key(
    parent: _DirectFullFramePartitionProfile,
    diagnostics: dict[str, Any],
    children: Sequence[_DirectFullFramePartitionProfile],
) -> tuple[Any, ...]:
    """Return a lossless key for one proof-authorized expansion domain.

    This is intentionally stricter than a failure/frontier label.  The full
    parent partition, every surviving already-used physical cut, the most
    recent cut, remaining depth, exact scalar proof, and every cadence-valid
    child partition all participate.  Consequently two states are dominated
    only when every future operator available *now* is identical; visual
    feasibility is never inferred or merged.
    """

    proofs = _direct_partition_exact_shift_proofs(
        diagnostics,
        span_count=len(parent.spans),
    )
    used_boundaries = _direct_partition_current_shifted_boundary_indices(
        parent
    )
    last_boundary = (
        _direct_partition_current_boundary_index_at_time(
            parent,
            parent.boundary_shifts[-1].canonical_master_seconds,
        )
        if parent.boundary_shifts
        else None
    )
    proof_key = tuple(
        (
            proof.boundary_index,
            proof.trigger,
            proof.failure,
            proof.failed_segment_index,
            proof.forward_reachable_left_source_indices,
            proof.left_valid_source_indices,
            proof.right_valid_source_indices,
            proof.local_nonsevere_edge_count,
        )
        for proof in proofs
    )
    child_signatures = tuple(
        sorted(tuple(child.unit_counts) for child in children)
    )
    return (
        tuple(parent.unit_counts),
        used_boundaries,
        last_boundary,
        _direct_partition_proof_directed_boundary_shift_limit(parent)
        - parent.proof_directed_boundary_motion_count,
        tuple(
            (
                redistribution.window_start_span_index,
                redistribution.proof_frontier_boundary_index,
                redistribution.original_unit_counts,
                redistribution.replacement_unit_counts,
            )
            for redistribution in parent.window_redistributions
        ),
        tuple(
            (
                split.parent_span_index,
                split.split_unit_offset,
                split.proof_frontier_boundary_index,
                split.evidence_contract,
                split.evidence_kind,
            )
            for split in parent.visual_frontier_splits
        ),
        (
            _DIRECT_MEMBER_MAX_VISUAL_FRONTIER_SPLITS_PER_PROFILE
            - len(parent.visual_frontier_splits)
        ),
        tuple(
            (
                merge.boundary_index,
                merge.merged_unit_count,
                merge.trigger,
            )
            for merge in parent.boundary_merges
        ),
        (
            _direct_partition_proof_directed_boundary_merge_limit(parent)
            - len(parent.boundary_merges)
        ),
        proof_key,
        child_signatures,
    )


def _direct_full_frame_shifted_partition_profiles(
    timing_segments: Sequence[DraftSegment],
    base: _DirectFullFramePartitionProfile,
    bottleneck_boundaries: Sequence[int],
    *,
    parent_diagnostics: dict[str, Any] | None = None,
) -> tuple[_DirectFullFramePartitionProfile, ...]:
    """Move only diagnosed 4+4 boundaries by one timing unit."""

    items = tuple(timing_segments)
    profiles: list[_DirectFullFramePartitionProfile] = []
    seen_counts = {base.unit_counts}
    for boundary_index in bottleneck_boundaries:
        if not 0 <= boundary_index < len(base.unit_counts) - 1:
            continue
        if base.unit_counts[boundary_index : boundary_index + 2] != (4, 4):
            continue
        # At the final boundary prefer a shorter tail first.  Else keep a
        # stable later-then-earlier ordering for deterministic diagnostics.
        for left_count, right_count, shift_units in (
            (5, 3, 1),
            (3, 5, -1),
        ):
            counts = list(base.unit_counts)
            counts[boundary_index : boundary_index + 2] = [
                left_count,
                right_count,
            ]
            signature = tuple(counts)
            if signature in seen_counts:
                continue
            try:
                spans = _direct_full_frame_spans_from_partition(items, signature)
            except ValueError:
                continue
            duration = sum(span.duration_seconds for span in spans)
            switches_per_minute = (len(spans) - 1) * 60.0 / duration
            span_median = median(span.duration_seconds for span in spans)
            if not (
                _DIRECT_MEMBER_MIN_SWITCHES_PER_MINUTE
                <= switches_per_minute
                <= _DIRECT_MEMBER_MAX_SWITCHES_PER_MINUTE
                and _DIRECT_MEMBER_MIN_MEDIAN_RUN_SECONDS
                <= span_median
                <= _DIRECT_MEMBER_MAX_MEDIAN_RUN_SECONDS
            ):
                continue
            new_boundary_unit_end = sum(signature[: boundary_index + 1])
            boundary_reasons = items[new_boundary_unit_end - 1].end_boundary_reasons
            exact_proof = next(
                (
                    proof
                    for proof in _direct_partition_exact_shift_proofs(
                        parent_diagnostics,
                        span_count=len(base.spans),
                    )
                    if proof.boundary_index == boundary_index
                ),
                None,
            ) if parent_diagnostics is not None else None
            shift = _DirectFullFrameBoundaryShift(
                boundary_index=boundary_index,
                shift_units=shift_units,
                left_unit_count=left_count,
                right_unit_count=right_count,
                canonical_master_seconds=(
                    spans[boundary_index].canonical_master_end
                ),
                boundary_reasons=boundary_reasons,
                trigger=(
                    exact_proof.trigger
                    if exact_proof is not None
                    else "caller_supplied_proven_bottleneck"
                ),
                parent_profile_id=base.profile_id,
                parent_partition_signature_sha256=(
                    _direct_partition_signature_sha256(base.unit_counts)
                ),
                proof_failure=(
                    exact_proof.failure if exact_proof is not None else None
                ),
                proof_failed_segment_index=(
                    exact_proof.failed_segment_index
                    if exact_proof is not None
                    else None
                ),
                proof_forward_reachable_left_source_indices=(
                    exact_proof.forward_reachable_left_source_indices
                    if exact_proof is not None
                    else ()
                ),
                proof_left_valid_source_indices=(
                    exact_proof.left_valid_source_indices
                    if exact_proof is not None
                    else ()
                ),
                proof_right_valid_source_indices=(
                    exact_proof.right_valid_source_indices
                    if exact_proof is not None
                    else ()
                ),
                proof_local_nonsevere_edge_count=(
                    exact_proof.local_nonsevere_edge_count
                    if exact_proof is not None
                    else None
                ),
            )
            profiles.append(
                _DirectFullFramePartitionProfile(
                    profile_id=(
                        f"boundary-{boundary_index}-"
                        f"{left_count}-plus-{right_count}"
                    ),
                    unit_counts=signature,
                    spans=spans,
                    boundary_shifts=base.boundary_shifts + (shift,),
                    window_redistributions=base.window_redistributions,
                    visual_frontier_splits=base.visual_frontier_splits,
                    boundary_merges=base.boundary_merges,
                    latest_operation_kind="boundary_shift",
                )
            )
            seen_counts.add(signature)
            if len(profiles) >= _DIRECT_MEMBER_MAX_PARTITION_PROFILE_COUNT - 1:
                return tuple(profiles)
    return tuple(profiles)


def _direct_partition_forward_exhaustion_bottleneck(
    diagnostics: dict[str, Any],
    *,
    span_count: int,
) -> int | None:
    """Locate one globally unreachable cut without weakening a local gate.

    ``forward_reachable_left_source_indices`` is the exact content-valid,
    nonsevere, no-repeat frontier before each fixed cut.  If it becomes empty
    after a boundary whose local cross-source graph still has edges, the
    current partition has proved a path-connectivity conflict rather than a
    bad-frame or severe-edge shortage.  The returned boundary may be moved and
    every visual gate recomputed; it never authorizes using the rejected edge.
    """

    if span_count < 3 or not _direct_partition_failure_is_proven(diagnostics):
        return None
    connectivity = diagnostics.get("hard_boundary_connectivity")
    if not isinstance(connectivity, dict) or connectivity.get(
        "analysis_basis"
    ) != "content_valid_nonsevere_no_repeat_reachability_v1":
        return None
    raw_boundaries = connectivity.get("boundaries")
    if not isinstance(raw_boundaries, list) or len(raw_boundaries) != span_count - 1:
        return None
    for boundary_index, boundary in enumerate(raw_boundaries):
        if not isinstance(boundary, dict):
            return None
        previous_frontier = boundary.get(
            "forward_reachable_left_source_indices"
        )
        current_frontier = boundary.get(
            "forward_reachable_right_source_indices"
        )
        if current_frontier is None and boundary_index + 1 < len(raw_boundaries):
            following = raw_boundaries[boundary_index + 1]
            current_frontier = (
                following.get("forward_reachable_left_source_indices")
                if isinstance(following, dict)
                else None
            )
        if not isinstance(previous_frontier, list) or not isinstance(current_frontier, list):
            return None
        if not previous_frontier or current_frontier:
            continue
        if boundary.get("boundary_index") != boundary_index:
            return None
        if int(boundary.get("nonsevere_cross_source_edge_count", 0)) <= 0:
            # A locally edge-free cut is handled by the primary proof-directed
            # shift.  This helper is only for the remaining global frontier
            # conflict after that primary repair.
            return None
        if not boundary.get("left_valid_source_indices") or not boundary.get(
            "right_valid_source_indices"
        ):
            return None
        return boundary_index
    return None


def _direct_full_frame_proof_directed_child_profiles(
    timing_segments: Sequence[DraftSegment],
    parent: _DirectFullFramePartitionProfile,
    diagnostics: dict[str, Any],
) -> tuple[_DirectFullFramePartitionProfile, ...]:
    """Create deterministic children from this parent's exact failure proof.

    The ordinary operator moves one independently proven boundary by one unit.
    When one moved cut exposes an adjacent exact failure, that adjacent cut may
    also move.  Adjacent moves are ordinary compositions of the final canonical
    partition: every materialized span and every visual, quota, cadence, and
    transition constraint is recomputed before selection.  A physical cut time
    is never moved twice, even after an earlier merge changes boundary indices.
    """

    if (
        parent.proof_directed_boundary_motion_count
        >= _direct_partition_proof_directed_boundary_shift_limit(parent)
    ):
        return ()
    proofs = _direct_partition_exact_shift_proofs(
        diagnostics,
        span_count=len(parent.spans),
    )
    if not proofs:
        return ()

    items = tuple(timing_segments)
    profiles: list[_DirectFullFramePartitionProfile] = []
    used_boundaries = set(
        _direct_partition_current_shifted_boundary_indices(parent)
    )
    seen_signatures: set[tuple[int, ...]] = set()
    for proof in proofs:
        boundary_index = proof.boundary_index
        if boundary_index in used_boundaries:
            continue
        left_units, right_units = parent.unit_counts[
            boundary_index : boundary_index + 2
        ]
        if left_units < 1 or right_units < 1:
            continue
        # Unit counts are not durations: one draft unit can be a complete
        # 1.9-second musical phraselet.  Admit a one-unit side only when the
        # materialized span independently passes the real-time 1.8-second
        # professional floor below.  A globally exhausted frontier authorizes
        # every integer placement inside the same two-span window, nearest
        # first.  The former +/-2 cap skipped valid 3+5 and 2+6 placements
        # after a 6+2 merge even though total duration, cut count, and every
        # downstream visual/cadence gate were unchanged and fully rechecked.
        if (
            proof.trigger
            == "proven_forward_connectivity_frontier_exhaustion"
        ):
            maximum_distance = max(left_units - 1, right_units - 1)
            shift_options = tuple(
                shift_units
                for distance in range(1, maximum_distance + 1)
                for shift_units in (distance, -distance)
            )
        else:
            shift_options = (1, -1)
        for shift_units in shift_options:
            left_count = left_units + shift_units
            right_count = right_units - shift_units
            if left_count < 1 or right_count < 1:
                continue
            counts = list(parent.unit_counts)
            counts[boundary_index : boundary_index + 2] = [
                left_count,
                right_count,
            ]
            signature = tuple(counts)
            if signature in seen_signatures:
                continue
            try:
                spans = _direct_full_frame_spans_from_partition(items, signature)
            except ValueError:
                continue
            if (
                min(
                    spans[boundary_index].duration_seconds,
                    spans[boundary_index + 1].duration_seconds,
                )
                + 1e-9
                < _DIRECT_MEMBER_HIGHLIGHT_MINIMUM_RUN_SECONDS
            ):
                continue
            duration = sum(span.duration_seconds for span in spans)
            switches_per_minute = (len(spans) - 1) * 60.0 / duration
            span_median = median(span.duration_seconds for span in spans)
            if not (
                _DIRECT_MEMBER_MIN_SWITCHES_PER_MINUTE
                <= switches_per_minute
                <= _DIRECT_MEMBER_MAX_SWITCHES_PER_MINUTE
                and _DIRECT_MEMBER_MIN_MEDIAN_RUN_SECONDS
                <= span_median
                <= _DIRECT_MEMBER_MAX_MEDIAN_RUN_SECONDS
            ):
                continue
            new_boundary_unit_end = sum(signature[: boundary_index + 1])
            boundary_reasons = items[
                new_boundary_unit_end - 1
            ].end_boundary_reasons
            shift = _DirectFullFrameBoundaryShift(
                boundary_index=boundary_index,
                shift_units=shift_units,
                left_unit_count=left_count,
                right_unit_count=right_count,
                canonical_master_seconds=(
                    spans[boundary_index].canonical_master_end
                ),
                boundary_reasons=boundary_reasons,
                trigger=proof.trigger,
                parent_profile_id=parent.profile_id,
                parent_partition_signature_sha256=(
                    _direct_partition_signature_sha256(parent.unit_counts)
                ),
                proof_failure=proof.failure,
                proof_failed_segment_index=proof.failed_segment_index,
                proof_forward_reachable_left_source_indices=(
                    proof.forward_reachable_left_source_indices
                ),
                proof_left_valid_source_indices=proof.left_valid_source_indices,
                proof_right_valid_source_indices=proof.right_valid_source_indices,
                proof_local_nonsevere_edge_count=(
                    proof.local_nonsevere_edge_count
                ),
            )
            label = (
                "frontier"
                if proof.trigger
                == "proven_forward_connectivity_frontier_exhaustion"
                else "local"
            )
            profile_id = (
                f"boundary-{boundary_index}-{left_count}-plus-{right_count}"
                if not parent.boundary_shifts
                and not parent.window_redistributions
                and not parent.boundary_merges
                else (
                    f"{parent.profile_id}__{label}-{boundary_index}-"
                    f"{left_count}-plus-{right_count}"
                )
            )
            profiles.append(
                _DirectFullFramePartitionProfile(
                    profile_id=profile_id,
                    unit_counts=signature,
                    spans=spans,
                    boundary_shifts=parent.boundary_shifts + (shift,),
                    window_redistributions=parent.window_redistributions,
                    visual_frontier_splits=parent.visual_frontier_splits,
                    boundary_merges=parent.boundary_merges,
                    latest_operation_kind="boundary_shift",
                )
            )
            seen_signatures.add(signature)
    return tuple(profiles)


def _direct_full_frame_proof_directed_merge_profiles(
    timing_segments: Sequence[DraftSegment],
    parent: _DirectFullFramePartitionProfile,
    diagnostics: dict[str, Any],
) -> tuple[_DirectFullFramePartitionProfile, ...]:
    """Remove only a proven bottleneck after no new safe shift remains.

    A two-sided minimum-duration cadence such as ``2+2`` has no legal shifted
    representation: ``1+3`` would create a sub-1.8-second flash cut.  The
    repair dispatcher invokes this operator only after the shift operator has
    produced no new child for the exact current proof.  Merging the two
    adjacent spans is then the remaining non-relaxed local representation. The
    merged child is still sent through complete content, native-cut,
    transition, quota, duration, and highlight-density audits before render.
    """

    if (
        parent.visual_frontier_splits
        or len(parent.boundary_merges)
        >= _direct_partition_proof_directed_boundary_merge_limit(parent)
    ):
        return ()
    proofs = _direct_partition_exact_shift_proofs(
        diagnostics,
        span_count=len(parent.spans),
    )
    if not proofs:
        return ()
    items = tuple(timing_segments)
    profiles: list[_DirectFullFramePartitionProfile] = []
    for proof in proofs:
        boundary_index = proof.boundary_index
        if not 0 <= boundary_index < len(parent.unit_counts) - 1:
            continue
        left_count, right_count = parent.unit_counts[
            boundary_index : boundary_index + 2
        ]
        counts = list(parent.unit_counts)
        merged_count = left_count + right_count
        counts[boundary_index : boundary_index + 2] = [merged_count]
        signature = tuple(counts)
        try:
            spans = _direct_full_frame_spans_from_partition(items, signature)
        except ValueError:
            continue
        duration = sum(span.duration_seconds for span in spans)
        switches_per_minute = (len(spans) - 1) * 60.0 / duration
        span_median = median(span.duration_seconds for span in spans)
        if not (
            _DIRECT_MEMBER_MIN_SWITCHES_PER_MINUTE
            <= switches_per_minute
            <= _DIRECT_MEMBER_MAX_SWITCHES_PER_MINUTE
            and _DIRECT_MEMBER_MIN_MEDIAN_RUN_SECONDS
            <= span_median
            <= _DIRECT_MEMBER_MAX_MEDIAN_RUN_SECONDS
        ):
            continue
        boundary_unit_end = sum(parent.unit_counts[: boundary_index + 1])
        boundary_reasons = items[boundary_unit_end - 1].end_boundary_reasons
        merge = _DirectFullFrameProofDirectedBoundaryMerge(
            boundary_index=boundary_index,
            left_unit_count=left_count,
            right_unit_count=right_count,
            merged_unit_count=merged_count,
            canonical_master_seconds=(
                parent.spans[boundary_index].canonical_master_end
            ),
            boundary_reasons=boundary_reasons,
            trigger=proof.trigger,
            parent_profile_id=parent.profile_id,
            parent_partition_signature_sha256=(
                _direct_partition_signature_sha256(parent.unit_counts)
            ),
            proof_failure=proof.failure,
            proof_failed_segment_index=proof.failed_segment_index,
            proof_forward_reachable_left_source_indices=(
                proof.forward_reachable_left_source_indices
            ),
            proof_left_valid_source_indices=proof.left_valid_source_indices,
            proof_right_valid_source_indices=proof.right_valid_source_indices,
            proof_local_nonsevere_edge_count=(
                proof.local_nonsevere_edge_count
            ),
        )
        profiles.append(
            _DirectFullFramePartitionProfile(
                profile_id=(
                    f"boundary-{boundary_index}-merge-{merged_count}"
                    if not parent.boundary_shifts
                    and not parent.window_redistributions
                    and not parent.boundary_merges
                    else (
                        f"{parent.profile_id}__merge-{boundary_index}-"
                        f"{merged_count}"
                    )
                ),
                unit_counts=signature,
                spans=spans,
                boundary_shifts=parent.boundary_shifts,
                window_redistributions=parent.window_redistributions,
                visual_frontier_splits=parent.visual_frontier_splits,
                boundary_merges=parent.boundary_merges + (merge,),
                latest_operation_kind="boundary_merge",
            )
        )
    return tuple(profiles)


def _direct_full_frame_proof_directed_window_profiles(
    timing_segments: Sequence[DraftSegment],
    parent: _DirectFullFramePartitionProfile,
    diagnostics: dict[str, Any],
) -> tuple[_DirectFullFramePartitionProfile, ...]:
    """Redistribute two cuts across three spans around an exact frontier.

    This operator is reached only after every cadence-valid placement inside
    the proved two-span window has been fully selected and rejected.  It keeps
    the surrounding three-span range and both cut counts fixed, moves the
    proved cut plus one previously untouched adjacent cut, and enumerates only
    existing timing-unit boundaries.  Every child is materialized through the
    ordinary duration/cadence gates here and the complete visual selector in
    the caller; no severe edge or visual threshold is relaxed.
    """

    if (
        parent.visual_frontier_splits
        or parent.proof_directed_boundary_motion_count + 2
        > _direct_partition_proof_directed_boundary_shift_limit(parent)
    ):
        return ()
    proofs = tuple(
        proof
        for proof in _direct_partition_exact_shift_proofs(
            diagnostics,
            span_count=len(parent.spans),
        )
        if proof.trigger
        == "proven_forward_connectivity_frontier_exhaustion"
    )
    if not proofs:
        return ()

    items = tuple(timing_segments)
    used_moved_boundaries = set(
        _direct_partition_current_shifted_boundary_indices(parent)
    )
    seen_signatures: set[tuple[int, ...]] = set()
    profiles: list[_DirectFullFramePartitionProfile] = []
    for proof in proofs:
        frontier = proof.boundary_index
        window_starts = tuple(
            start
            for start in (frontier, frontier - 1)
            if 0 <= start and start + 2 < len(parent.unit_counts)
        )
        for window_start in window_starts:
            adjacent_boundary_index = (
                window_start + 1
                if frontier == window_start
                else window_start
            )
            if adjacent_boundary_index in used_moved_boundaries:
                continue
            original = tuple(
                int(value)
                for value in parent.unit_counts[
                    window_start : window_start + 3
                ]
            )
            if len(original) != 3:
                continue
            original_first_boundary_units = original[0]
            original_second_boundary_units = original[0] + original[1]
            total_units = sum(original)
            for first_count in range(1, total_units - 1):
                for second_count in range(1, total_units - first_count):
                    third_count = total_units - first_count - second_count
                    replacement = (
                        first_count,
                        second_count,
                        third_count,
                    )
                    if replacement == original:
                        continue
                    if (
                        first_count == original_first_boundary_units
                        or first_count + second_count
                        == original_second_boundary_units
                    ):
                        # A three-span repair must move both cuts.  Single-cut
                        # placements were exhausted by the preceding operator.
                        continue
                    counts = list(parent.unit_counts)
                    counts[window_start : window_start + 3] = replacement
                    signature = tuple(counts)
                    if signature in seen_signatures:
                        continue
                    try:
                        spans = _direct_full_frame_spans_from_partition(
                            items,
                            signature,
                        )
                    except ValueError:
                        continue
                    if (
                        min(
                            span.duration_seconds
                            for span in spans[
                                window_start : window_start + 3
                            ]
                        )
                        + 1e-9
                        < _DIRECT_MEMBER_HIGHLIGHT_MINIMUM_RUN_SECONDS
                    ):
                        continue
                    duration = sum(span.duration_seconds for span in spans)
                    switches_per_minute = (len(spans) - 1) * 60.0 / duration
                    span_median = median(
                        span.duration_seconds for span in spans
                    )
                    if not (
                        _DIRECT_MEMBER_MIN_SWITCHES_PER_MINUTE
                        <= switches_per_minute
                        <= _DIRECT_MEMBER_MAX_SWITCHES_PER_MINUTE
                        and _DIRECT_MEMBER_MIN_MEDIAN_RUN_SECONDS
                        <= span_median
                        <= _DIRECT_MEMBER_MAX_MEDIAN_RUN_SECONDS
                    ):
                        continue
                    first_unit_end = sum(signature[: window_start + 1])
                    second_unit_end = sum(signature[: window_start + 2])
                    replacement_reasons = (
                        items[first_unit_end - 1].end_boundary_reasons,
                        items[second_unit_end - 1].end_boundary_reasons,
                    )
                    redistribution = (
                        _DirectFullFrameProofDirectedWindowRedistribution(
                            window_start_span_index=window_start,
                            proof_frontier_boundary_index=frontier,
                            proof_frontier_offset_in_window=(
                                frontier - window_start
                            ),
                            original_unit_counts=original,
                            replacement_unit_counts=replacement,
                            original_boundary_master_seconds=(
                                parent.spans[
                                    window_start
                                ].canonical_master_end,
                                parent.spans[
                                    window_start + 1
                                ].canonical_master_end,
                            ),
                            replacement_boundary_master_seconds=(
                                spans[window_start].canonical_master_end,
                                spans[
                                    window_start + 1
                                ].canonical_master_end,
                            ),
                            replacement_boundary_reasons=(
                                tuple(replacement_reasons[0]),
                                tuple(replacement_reasons[1]),
                            ),
                            trigger=(
                                "proven_forward_connectivity_frontier_"
                                "three_span_redistribution"
                            ),
                            parent_profile_id=parent.profile_id,
                            parent_partition_signature_sha256=(
                                _direct_partition_signature_sha256(
                                    parent.unit_counts
                                )
                            ),
                            proof_failure=proof.failure,
                            proof_failed_segment_index=(
                                proof.failed_segment_index
                            ),
                            proof_forward_reachable_left_source_indices=(
                                proof.forward_reachable_left_source_indices
                            ),
                            proof_left_valid_source_indices=(
                                proof.left_valid_source_indices
                            ),
                            proof_right_valid_source_indices=(
                                proof.right_valid_source_indices
                            ),
                            proof_local_nonsevere_edge_count=(
                                proof.local_nonsevere_edge_count
                            ),
                        )
                    )
                    profiles.append(
                        _DirectFullFramePartitionProfile(
                            profile_id=(
                                f"{parent.profile_id}__window-{window_start}-"
                                f"{first_count}-plus-{second_count}-plus-"
                                f"{third_count}"
                            ),
                            unit_counts=signature,
                            spans=spans,
                            boundary_shifts=parent.boundary_shifts,
                            window_redistributions=(
                                parent.window_redistributions
                                + (redistribution,)
                            ),
                            visual_frontier_splits=(
                                parent.visual_frontier_splits
                            ),
                            boundary_merges=parent.boundary_merges,
                            latest_operation_kind="window_redistribution",
                        )
                    )
                    seen_signatures.add(signature)
    profiles.sort(
        key=lambda profile: (
            profile.window_redistributions[-1].boundary_displacement_units,
            profile.musical_boundary_penalty,
            profile.profile_id,
        )
    )
    return tuple(profiles)


def _direct_visual_split_sample_is_hard_safe_proposal_point(
    sample: dict[str, Any],
    *,
    quality_score: Any,
    lighting_rescue_applied: bool,
    config: DirectMemberVisualConfig,
) -> bool:
    """Recognize a strict sampled safe point without inferring identity.

    This is proposal evidence only.  It cannot make a source/span valid; the
    child partition is always sent through the complete selector again.
    """

    if sample.get("descriptor_available") is not True:
        return False
    human = bool(
        sample.get("attention_basis") is not None
        or int(sample.get("face_count", 0)) > 0
        or int(sample.get("detected_person_count", 0)) > 0
        or int(sample.get("formation_person_count", 0)) > 0
        or float(sample.get("silhouette_confidence", 0.0))
        >= config.minimum_silhouette_confidence
    )
    if not human:
        return False
    obstruction = sample.get("foreground_obstruction_score")
    if not isinstance(obstruction, (int, float)) or not math.isfinite(
        float(obstruction)
    ):
        return False
    if float(obstruction) >= config.foreground_obstruction_rejection_score:
        return False
    if lighting_rescue_applied:
        return True
    return bool(
        isinstance(quality_score, (int, float))
        and math.isfinite(float(quality_score))
        and float(quality_score) >= config.minimum_acceptable_quality_score
    )


def _direct_severe_counterfactual_retry_is_authorized(
    diagnostics: Mapping[str, Any],
    *,
    span_count: int,
) -> bool:
    """Accept only an integrity-bound, non-authoritative exact witness.

    The witness may propose one more timing partition, but it can never select
    a source path or relax a visual/transition gate.  Keeping this validation
    in the pipeline prevents a partial or stale diagnostic payload from
    silently expanding the production search.
    """

    if span_count < 3 or not _direct_partition_failure_is_proven(dict(diagnostics)):
        return False
    if diagnostics.get("constraint_infeasibility_proven") is not True:
        return False
    search = diagnostics.get("search")
    if not isinstance(search, dict):
        return False
    exact = search.get("exact_lifeline")
    if not isinstance(exact, dict) or not (
        exact.get("invoked") is True
        and exact.get("state_capacity_exhausted") is False
        and exact.get("duration_state_quantization_applied") is False
    ):
        return False
    counterfactual = search.get("severe_transition_counterfactual")
    if not isinstance(counterfactual, dict) or not (
        counterfactual.get("contract")
        == "diagnostic_only_severe_transition_counterfactual_v1"
        and counterfactual.get("optimization_contract")
        == (
            "exact_iterative_deepening_minimum_severe_transition_count_"
            "then_deterministic_backward_source_order_v1"
        )
        and counterfactual.get("severe_transition_cost_policy")
        == "one_per_adjacent_transition_with_any_original_severe_flag"
        and counterfactual.get("tie_break_contract")
        == (
            "final_source_order_then_reverse_predecessor_"
            "severe_cost_mismatch_cost_source_order_v1"
        )
        and counterfactual.get("invoked") is True
        and counterfactual.get("witness_found") is True
        and counterfactual.get("integrity_check_passed") is True
        and counterfactual.get("optimality_proven") is True
        and counterfactual.get("minimum_severe_transition_count") == 1
        and counterfactual.get(
            "production_single_split_proposal_authorized"
        ) is True
        and counterfactual.get("maximum_diagnostic_severe_transition_budget")
        == 3
        and counterfactual.get("state_capacity_exhausted") is False
        and counterfactual.get("duration_state_quantization_applied") is False
        and counterfactual.get("constraint_infeasibility_proven") is False
        and counterfactual.get("severe_transition_flags_disabled") is True
        and counterfactual.get("original_matrices_mutated") is False
        and counterfactual.get("selection_eligible") is False
        and counterfactual.get("render_authorized") is False
        and counterfactual.get("visual_gates_relaxed") is False
    ):
        return False
    unchanged_markers = (
        "candidate_validity_semantics_changed",
        "source_quota_semantics_changed",
        "exact_duration_semantics_changed",
        "source_order_semantics_changed",
        "no_adjacent_repeat_semantics_changed",
    )
    if any(counterfactual.get(name) is not False for name in unchanged_markers):
        return False

    digest_names = (
        "original_hard_constraint_sha256",
        "original_failure_certificate_sha256",
        "severe_disabled_hard_constraint_sha256",
        "severe_transition_budget_mask_sha256",
        "severe_disabled_witness_sha256",
        "minimum_witness_path_sha256",
    )
    digests = tuple(counterfactual.get(name) for name in digest_names)
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in digests
    ):
        return False
    if digests[0] == digests[2] or digests[4] != digests[5]:
        return False

    zero_budget = counterfactual.get("zero_budget_hard_proof")
    lower_budget_proofs = counterfactual.get("lower_budget_hard_proofs")
    budget_attempts = counterfactual.get("budget_attempts")
    if not (
        isinstance(zero_budget, dict)
        and zero_budget.get("severe_transition_budget") == 0
        and zero_budget.get("path_found") is False
        and zero_budget.get("state_capacity_exhausted") is False
        and zero_budget.get("constraint_infeasibility_proven") is True
        and zero_budget.get("duration_state_quantization_applied") is False
        and zero_budget.get("certificate_sha256") == digests[1]
        and isinstance(lower_budget_proofs, list)
        and len(lower_budget_proofs) == 1
        and lower_budget_proofs[0] == zero_budget
        and isinstance(budget_attempts, list)
        and len(budget_attempts) == 1
    ):
        return False
    minimum_attempt = budget_attempts[0]
    if not (
        isinstance(minimum_attempt, dict)
        and minimum_attempt.get("severe_transition_budget") == 1
        and minimum_attempt.get("path_found") is True
        and minimum_attempt.get("state_capacity_exhausted") is False
        and minimum_attempt.get("constraint_infeasibility_proven") is False
        and minimum_attempt.get("duration_state_quantization_applied") is False
        and isinstance(minimum_attempt.get("explored_state_count"), int)
        and minimum_attempt["explored_state_count"] > 0
        and isinstance(minimum_attempt.get("memoized_failed_state_count"), int)
        and minimum_attempt["memoized_failed_state_count"] >= 0
        and isinstance(minimum_attempt.get("certificate_sha256"), str)
        and len(minimum_attempt["certificate_sha256"]) == 64
        and all(
            character in "0123456789abcdef"
            for character in minimum_attempt["certificate_sha256"]
        )
        and isinstance(counterfactual.get("state_limit_per_budget"), int)
        and counterfactual["state_limit_per_budget"] > 0
    ):
        return False

    raw_indices = counterfactual.get("blocking_boundary_indices")
    raw_transitions = counterfactual.get("blocking_transitions")
    if not isinstance(raw_indices, list) or not isinstance(raw_transitions, list):
        return False
    indices = tuple(raw_indices)
    if (
        not indices
        or indices != tuple(sorted(set(indices)))
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < span_count - 1
            for value in indices
        )
    ):
        return False
    transition_indices: list[int] = []
    for transition in raw_transitions:
        if not isinstance(transition, dict):
            return False
        boundary_index = transition.get("boundary_index")
        outgoing = transition.get("outgoing_eligible_source_index")
        incoming = transition.get("incoming_eligible_source_index")
        flags = transition.get("original_severe_flags")
        if not (
            isinstance(boundary_index, int)
            and not isinstance(boundary_index, bool)
            and boundary_index in indices
            and isinstance(outgoing, int)
            and not isinstance(outgoing, bool)
            and outgoing >= 0
            and isinstance(incoming, int)
            and not isinstance(incoming, bool)
            and incoming >= 0
            and outgoing != incoming
            and isinstance(flags, list)
            and flags
            and all(isinstance(flag, str) and flag for flag in flags)
        ):
            return False
        transition_indices.append(boundary_index)
    if (
        tuple(sorted(set(transition_indices))) != indices
        or len(raw_transitions) != 1
        or len(indices) != 1
    ):
        return False
    maximum_deviation = counterfactual.get("maximum_duration_relative_deviation")
    return bool(
        isinstance(maximum_deviation, (int, float))
        and not isinstance(maximum_deviation, bool)
        and math.isfinite(float(maximum_deviation))
        and float(maximum_deviation) >= 0.0
    )


def _direct_counterfactual_boundary_proof(
    diagnostics: Mapping[str, Any],
    *,
    boundary_index: int,
    span_count: int,
) -> _DirectPartitionBoundaryProof | None:
    """Bind one severe-only witness edge to live hard-connectivity evidence."""

    if not _direct_severe_counterfactual_retry_is_authorized(
        diagnostics,
        span_count=span_count,
    ):
        return None
    search = diagnostics["search"]
    counterfactual = search["severe_transition_counterfactual"]
    if boundary_index not in counterfactual["blocking_boundary_indices"]:
        return None
    transitions = tuple(
        item
        for item in counterfactual["blocking_transitions"]
        if item["boundary_index"] == boundary_index
    )
    outgoing_sources = tuple(
        sorted({int(item["outgoing_eligible_source_index"]) for item in transitions})
    )
    incoming_sources = tuple(
        sorted({int(item["incoming_eligible_source_index"]) for item in transitions})
    )
    connectivity = diagnostics.get("hard_boundary_connectivity")
    if not isinstance(connectivity, dict) or connectivity.get(
        "analysis_basis"
    ) != "content_valid_nonsevere_no_repeat_reachability_v1":
        return None
    boundaries = connectivity.get("boundaries")
    if not isinstance(boundaries, list) or len(boundaries) != span_count - 1:
        return None
    item = boundaries[boundary_index]
    if not isinstance(item, dict) or item.get("boundary_index") != boundary_index:
        return None
    left_valid = item.get("left_valid_source_indices")
    right_valid = item.get("right_valid_source_indices")
    forward_reachable = item.get("forward_reachable_left_source_indices")
    local_edges = item.get("nonsevere_cross_source_edge_count")
    if not (
        isinstance(left_valid, list)
        and left_valid
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in left_valid
        )
        and isinstance(right_valid, list)
        and right_valid
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in right_valid
        )
        and isinstance(forward_reachable, list)
        and forward_reachable
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in forward_reachable
        )
        and isinstance(local_edges, int)
        and not isinstance(local_edges, bool)
        and local_edges >= 0
        and outgoing_sources
        and incoming_sources
        and set(outgoing_sources).issubset(left_valid)
        and set(outgoing_sources).issubset(forward_reachable)
        and set(incoming_sources).issubset(right_valid)
    ):
        return None
    return _DirectPartitionBoundaryProof(
        boundary_index=boundary_index,
        trigger="diagnostic_severe_transition_counterfactual_safe_bracket",
        failure=(
            str(diagnostics["failure"])
            if diagnostics.get("failure") is not None
            else None
        ),
        failed_segment_index=boundary_index + 1,
        forward_reachable_left_source_indices=tuple(
            int(value) for value in forward_reachable
        ),
        left_valid_source_indices=tuple(int(value) for value in left_valid),
        right_valid_source_indices=tuple(int(value) for value in right_valid),
        local_nonsevere_edge_count=local_edges,
    )


def _direct_full_frame_visual_frontier_split_child_profiles(
    timing_segments: Sequence[DraftSegment],
    parent: _DirectFullFramePartitionProfile,
    diagnostics: dict[str, Any],
    *,
    config: DirectMemberVisualConfig,
    allow_counterfactual_secondary_split: bool = False,
    _counterfactual_boundary_index: int | None = None,
) -> tuple[_DirectFullFramePartitionProfile, ...]:
    """Add one evidence-bound micro-boundary beside a proven path frontier.

    Native-cut timestamps are exact master-time events.  Sample-derived visual
    interval edges are deliberately represented only as brackets: no midpoint
    is promoted to an exact edge.  The selected cut is an existing timing-unit
    boundary inside that bracket (or within the local half-unit snap domain of
    an exact native cut).  This operator never admits content itself; the full
    content, native-cut, severe-edge, quota, duration, and transition selector
    is rerun for the child.
    """

    secondary_counterfactual = bool(
        allow_counterfactual_secondary_split
        and len(parent.visual_frontier_splits)
        == _DIRECT_MEMBER_COUNTERFACTUAL_SECONDARY_SPLIT_TRIGGER_COUNT
        and _direct_severe_counterfactual_retry_is_authorized(
            diagnostics,
            span_count=len(parent.spans),
        )
    )
    if len(parent.visual_frontier_splits) >= (
        _DIRECT_MEMBER_MAX_VISUAL_FRONTIER_SPLITS_PER_PROFILE
        + int(secondary_counterfactual)
    ):
        return ()
    if secondary_counterfactual and _counterfactual_boundary_index is None:
        counterfactual = diagnostics["search"][
            "severe_transition_counterfactual"
        ]
        for boundary_index in counterfactual["blocking_boundary_indices"]:
            children = _direct_full_frame_visual_frontier_split_child_profiles(
                timing_segments,
                parent,
                diagnostics,
                config=config,
                allow_counterfactual_secondary_split=True,
                _counterfactual_boundary_index=boundary_index,
            )
            if children:
                return children
        return ()
    if secondary_counterfactual:
        if _counterfactual_boundary_index is None:
            return ()
        proof = _direct_counterfactual_boundary_proof(
            diagnostics,
            boundary_index=_counterfactual_boundary_index,
            span_count=len(parent.spans),
        )
        if proof is None:
            return ()
    else:
        proofs = tuple(
            proof
            for proof in _direct_partition_exact_shift_proofs(
                diagnostics,
                span_count=len(parent.spans),
            )
            if proof.trigger == "proven_forward_connectivity_frontier_exhaustion"
        )
        if len(proofs) != 1:
            return ()
        proof = proofs[0]
    if proof.failed_segment_index != proof.boundary_index + 1:
        return ()

    raw_span_diagnostics = diagnostics.get("span_content_validity")
    if not isinstance(raw_span_diagnostics, list):
        return ()
    span_diagnostics = {
        int(raw["segment_index"]): raw
        for raw in raw_span_diagnostics
        if isinstance(raw, dict)
        and isinstance(raw.get("segment_index"), int)
    }
    items = tuple(timing_segments)
    if not items:
        return ()

    # One exact frontier may split only its incoming span or its immediately
    # outgoing neighbor.  The incoming side is preferred because it creates a
    # bridge after the proved forward frontier without perturbing normal spans.
    adjacent_span_indices = (
        proof.boundary_index + 1,
        proof.boundary_index,
    )
    proposals: list[
        tuple[
            tuple[float | int, ...],
            int,
            int,
            float,
            tuple[str, ...],
            str,
            tuple[int, ...],
            tuple[float, ...],
            tuple[float, float] | None,
        ]
    ] = []
    for side_rank, span_index in enumerate(adjacent_span_indices):
        if not 0 <= span_index < len(parent.unit_counts):
            continue
        unit_count = parent.unit_counts[span_index]
        if unit_count < 2:
            continue
        raw_span = span_diagnostics.get(span_index)
        if not isinstance(raw_span, dict):
            continue
        span = parent.spans[span_index]
        if not (
            math.isclose(
                float(raw_span.get("canonical_master_start", math.nan)),
                span.canonical_master_start,
                rel_tol=0.0,
                abs_tol=1e-7,
            )
            and math.isclose(
                float(raw_span.get("canonical_master_end", math.nan)),
                span.canonical_master_end,
                rel_tol=0.0,
                abs_tol=1e-7,
            )
        ):
            continue
        raw_sources = raw_span.get("sources")
        if not isinstance(raw_sources, list):
            continue

        exact_native_events: list[tuple[float, int]] = []
        sample_brackets: list[tuple[float, float, int]] = []
        hard_valid_sources: list[int] = []
        hard_valid_sample_ranges: list[tuple[float, float, int]] = []
        for raw_source in raw_sources:
            if not isinstance(raw_source, dict) or not isinstance(
                raw_source.get("eligible_source_index"),
                int,
            ):
                continue
            source_index = int(raw_source["eligible_source_index"])
            raw_cuts = raw_source.get("native_scene_cut_master_seconds", ())
            if isinstance(raw_cuts, list):
                exact_native_events.extend(
                    (float(value), source_index)
                    for value in raw_cuts
                    if isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    and span.canonical_master_start
                    < float(value)
                    < span.canonical_master_end
                )
            samples = raw_source.get("samples")
            quality = raw_source.get("quality_sample_scores")
            if not isinstance(samples, list) or not isinstance(quality, list):
                continue
            if len(samples) != len(quality) or len(samples) < 2:
                continue
            lighting_rescue = bool(
                raw_source.get(
                    "portrait_fancam_lighting_quality_rescue_applied",
                    False,
                )
            )
            safe_flags = [
                _direct_visual_split_sample_is_hard_safe_proposal_point(
                    sample,
                    quality_score=quality[index],
                    lighting_rescue_applied=lighting_rescue,
                    config=config,
                )
                if isinstance(sample, dict)
                else False
                for index, sample in enumerate(samples)
            ]
            sample_times = [
                float(sample.get("canonical_master_seconds", math.nan))
                if isinstance(sample, dict)
                else math.nan
                for sample in samples
            ]
            if (
                all(math.isfinite(value) for value in sample_times)
                and all(
                    span.canonical_master_start - 1e-9
                    <= value
                    <= span.canonical_master_end + 1e-9
                    for value in sample_times
                )
                and all(
                    right > left
                    for left, right in zip(sample_times, sample_times[1:])
                )
            ):
                for sample_index in range(len(samples) - 1):
                    if safe_flags[sample_index] == safe_flags[sample_index + 1]:
                        continue
                    left_time, right_time = sorted(
                        (sample_times[sample_index], sample_times[sample_index + 1])
                    )
                    sample_brackets.append(
                        (left_time, right_time, source_index)
                    )
                if raw_source.get("valid") is True and all(safe_flags):
                    hard_valid_sources.append(source_index)
                    hard_valid_sample_ranges.append(
                        (sample_times[0], sample_times[-1], source_index)
                    )

        unit_cursor = sum(parent.unit_counts[:span_index])
        internal_boundaries: list[
            tuple[int, float, tuple[str, ...], float]
        ] = []
        for split_offset in range(1, unit_count):
            left_item = items[unit_cursor + split_offset - 1]
            right_item = items[unit_cursor + split_offset]
            boundary_seconds = left_item.canonical_master_end
            left_duration = (
                left_item.canonical_master_end
                - left_item.canonical_master_start
            )
            right_duration = (
                right_item.canonical_master_end
                - right_item.canonical_master_start
            )
            snap_tolerance = 0.5 * min(left_duration, right_duration)
            internal_boundaries.append(
                (
                    split_offset,
                    boundary_seconds,
                    left_item.end_boundary_reasons,
                    snap_tolerance,
                )
            )

        def _boundary_reason_penalty(reasons: Sequence[str]) -> int:
            values = set(reasons)
            if "phrase" in values:
                return 0
            if "strong_beat" in values:
                return 1
            if "beat" in values:
                return 2
            return 3

        for split_offset, boundary_seconds, reasons, snap_tolerance in (
            internal_boundaries
        ):
            left_count = split_offset
            right_count = unit_count - split_offset
            one_unit_penalty = int(min(left_count, right_count) < 2)
            reason_penalty = _boundary_reason_penalty(reasons)
            native_matches = [
                (abs(boundary_seconds - event_seconds), event_seconds, source_index)
                for event_seconds, source_index in exact_native_events
                if abs(boundary_seconds - event_seconds)
                <= snap_tolerance + 1e-9
            ]
            if native_matches:
                distance, event_seconds, source_index = min(native_matches)
                proposals.append(
                    (
                        (
                            0,
                            distance,
                            reason_penalty,
                            one_unit_penalty,
                            side_rank,
                            boundary_seconds,
                        ),
                        span_index,
                        split_offset,
                        boundary_seconds,
                        reasons,
                        "confirmed_source_native_cut_master_time_snap",
                        (source_index,),
                        (event_seconds,),
                        None,
                    )
                )
                continue
            bracket_matches = [
                (left, right, source_index)
                for left, right, source_index in sample_brackets
                if left - 1e-9 <= boundary_seconds <= right + 1e-9
            ]
            if bracket_matches:
                left, right, source_index = min(
                    bracket_matches,
                    key=lambda value: (
                        abs(boundary_seconds - (value[0] + value[1]) / 2.0),
                        value[2],
                    ),
                )
                proposals.append(
                    (
                        (
                            1,
                            abs(boundary_seconds - (left + right) / 2.0),
                            reason_penalty,
                            one_unit_penalty,
                            side_rank,
                            boundary_seconds,
                        ),
                        span_index,
                        split_offset,
                        boundary_seconds,
                        reasons,
                        "sample_hard_safety_change_bracket",
                        (source_index,),
                        (),
                        (left, right),
                    )
                )
                continue

            # A globally exhausted frontier can need an intermediate source
            # even when every candidate is content-valid.  Use only the
            # complete hard-valid sample range and require a candidate outside
            # the current forward source set; this is still proposal evidence,
            # never permission to cross a severe edge.
            bridge_sources = sorted(
                set(hard_valid_sources)
                .intersection(proof.right_valid_source_indices)
                .difference(proof.forward_reachable_left_source_indices)
            )
            safe_ranges = [
                (left, right, source_index)
                for left, right, source_index in hard_valid_sample_ranges
                if source_index in bridge_sources
                and left - 1e-9 <= boundary_seconds <= right + 1e-9
            ]
            if proof.forward_reachable_left_source_indices and safe_ranges:
                left = max(value[0] for value in safe_ranges)
                right = min(value[1] for value in safe_ranges)
                if left <= boundary_seconds <= right:
                    proposals.append(
                        (
                            (
                                2,
                                reason_penalty,
                                one_unit_penalty,
                                abs(boundary_seconds - (left + right) / 2.0),
                                side_rank,
                                boundary_seconds,
                            ),
                            span_index,
                            split_offset,
                            boundary_seconds,
                            reasons,
                            "hard_valid_bridge_source_entry_exit_bracket",
                            tuple(
                                sorted(
                                    set(
                                        proof.forward_reachable_left_source_indices
                                    ).union(bridge_sources)
                                )
                            ),
                            (),
                            (left, right),
                        )
                    )

    if not proposals:
        return ()
    (
        _score,
        span_index,
        split_offset,
        boundary_seconds,
        boundary_reasons,
        evidence_kind,
        evidence_source_indices,
        evidence_master_seconds,
        evidence_bracket,
    ) = min(proposals, key=lambda item: item[0])
    original_count = parent.unit_counts[span_index]
    left_count = split_offset
    right_count = original_count - split_offset
    counts = list(parent.unit_counts)
    counts[span_index : span_index + 1] = [left_count, right_count]
    signature = tuple(counts)
    try:
        spans = _direct_full_frame_spans_from_partition(items, signature)
    except ValueError:
        return ()
    duration = sum(span.duration_seconds for span in spans)
    switches_per_minute = (len(spans) - 1) * 60.0 / duration
    span_median = median(span.duration_seconds for span in spans)
    if not (
        _DIRECT_MEMBER_MIN_SWITCHES_PER_MINUTE
        <= switches_per_minute
        <= _DIRECT_MEMBER_MAX_SWITCHES_PER_MINUTE
        and _DIRECT_MEMBER_MIN_MEDIAN_RUN_SECONDS
        <= span_median
        <= _DIRECT_MEMBER_MAX_MEDIAN_RUN_SECONDS
    ):
        return ()
    split = _DirectFullFrameVisualFrontierSplit(
        parent_span_index=span_index,
        split_unit_offset=split_offset,
        left_unit_count=left_count,
        right_unit_count=right_count,
        canonical_master_seconds=boundary_seconds,
        boundary_reasons=boundary_reasons,
        trigger=(
            "diagnostic_severe_counterfactual_visual_micro_split"
            if secondary_counterfactual
            else "proven_forward_connectivity_frontier_visual_micro_split"
        ),
        parent_profile_id=parent.profile_id,
        parent_partition_signature_sha256=(
            _direct_partition_signature_sha256(parent.unit_counts)
        ),
        proof_failure=proof.failure,
        proof_failed_segment_index=proof.failed_segment_index,
        proof_frontier_boundary_index=proof.boundary_index,
        proof_forward_reachable_left_source_indices=(
            proof.forward_reachable_left_source_indices
        ),
        proof_right_valid_source_indices=proof.right_valid_source_indices,
        evidence_contract=(
            "diagnostic_severe_counterfactual_bound_existing_timing_boundary_v1"
            if secondary_counterfactual
            else "visual_bracket_bound_existing_timing_boundary_v1"
        ),
        evidence_kind=evidence_kind,
        evidence_source_indices=evidence_source_indices,
        evidence_master_seconds=evidence_master_seconds,
        evidence_bracket_master_seconds=evidence_bracket,
    )
    return (
        _DirectFullFramePartitionProfile(
            profile_id=(
                f"{parent.profile_id}__visual-split-{span_index}-"
                f"{left_count}-plus-{right_count}"
            ),
            unit_counts=signature,
            spans=spans,
            boundary_shifts=parent.boundary_shifts,
            window_redistributions=parent.window_redistributions,
            visual_frontier_splits=parent.visual_frontier_splits + (split,),
            boundary_merges=parent.boundary_merges,
            latest_operation_kind="visual_frontier_split",
        ),
    )


def _direct_full_frame_proof_directed_repair_profiles(
    timing_segments: Sequence[DraftSegment],
    parent: _DirectFullFramePartitionProfile,
    diagnostics: dict[str, Any],
    *,
    config: DirectMemberVisualConfig,
) -> tuple[_DirectFullFramePartitionProfile, ...]:
    """Continue an exact frontier repair without leaving a dead-end gap.

    Boundary shifts remain the least structural repair and a merge remains the
    fallback for an immovable minimum-duration cut.  If every two-span
    placement is exhausted, a global frontier may redistribute both cuts in
    the surrounding three-span window while preserving duration and cut count.
    Only after those fixed-count repairs fail may the same exact proof add one
    evidence-bound musical micro-boundary.  Every returned child is fully
    reselected by the caller.
    """

    children = _direct_full_frame_proof_directed_child_profiles(
        timing_segments,
        parent,
        diagnostics,
    )
    if not children:
        children = _direct_full_frame_proof_directed_merge_profiles(
            timing_segments,
            parent,
            diagnostics,
        )
    if not children:
        children = _direct_full_frame_proof_directed_window_profiles(
            timing_segments,
            parent,
            diagnostics,
        )
    if not children:
        children = _direct_full_frame_visual_frontier_split_child_profiles(
            timing_segments,
            parent,
            diagnostics,
            config=config,
        )
    return children


def _render_cadence_payload(
    segments: Sequence[RenderSegment],
) -> dict[str, Any]:
    items = tuple(segments)
    if not items:
        raise ValueError("render cadence needs at least one segment")
    run_durations: list[float] = []
    current_source = items[0].source_index
    current_duration = 0.0
    switch_count = 0
    for item in items:
        if item.source_index != current_source:
            run_durations.append(current_duration)
            current_source = item.source_index
            current_duration = 0.0
            switch_count += 1
        current_duration += item.duration
    run_durations.append(current_duration)
    duration = sum(item.duration for item in items)
    return {
        "source_switch_count": switch_count,
        "source_switches_per_minute": switch_count * 60.0 / duration,
        "source_run_count": len(run_durations),
        "source_run_median_seconds": median(run_durations),
        "longest_source_run_seconds": max(run_durations),
        "target_source_switches_per_minute": [
            _DIRECT_MEMBER_MIN_SWITCHES_PER_MINUTE,
            _DIRECT_MEMBER_MAX_SWITCHES_PER_MINUTE,
        ],
        "target_source_run_median_seconds": [
            _DIRECT_MEMBER_MIN_MEDIAN_RUN_SECONDS,
            _DIRECT_MEMBER_MAX_MEDIAN_RUN_SECONDS,
        ],
        "maximum_source_run_seconds": _DIRECT_MEMBER_MAX_RUN_SECONDS,
    }


def _direct_member_highlight_cadence_payload(
    *,
    spans: Sequence[_DirectFullFrameSpan],
    visual_selection: DirectMemberVisualSelection,
    highlight_intervals: Sequence[tuple[float, float]],
) -> dict[str, Any]:
    """Audit the exact selected cuts separately inside active sections."""

    items = tuple(spans)
    if not items:
        raise ValueError("highlight cadence audit needs direct-member spans")
    if len(visual_selection.source_indices) != len(items):
        raise ValueError("highlight cadence selection does not match spans")
    if len(visual_selection.boundaries) != len(items) - 1:
        raise ValueError("highlight cadence boundary count does not match spans")
    normalized = (
        _normalized_direct_member_highlight_intervals(
            highlight_intervals,
            range_start=items[0].canonical_master_start,
            range_end=items[-1].canonical_master_end,
        )
        if highlight_intervals
        else ()
    )
    total_duration = (
        items[-1].canonical_master_end - items[0].canonical_master_start
    )
    highlight_duration = sum(end - start for start, end in normalized)
    non_highlight_duration = max(0.0, total_duration - highlight_duration)
    highlight_boundary_indices = tuple(
        index
        for index, span in enumerate(items[:-1])
        if _direct_member_boundary_is_highlight(
            span.canonical_master_end,
            normalized,
        )
    )
    highlight_boundary_set = frozenset(highlight_boundary_indices)
    applied_cut_indices = tuple(
        index
        for index, boundary in enumerate(visual_selection.boundaries)
        if boundary.cut_applied
    )
    highlight_cut_indices = tuple(
        index for index in applied_cut_indices if index in highlight_boundary_set
    )
    highlight_moving_cut_indices = tuple(
        index
        for index in highlight_cut_indices
        if (
            visual_selection.boundaries[index].mismatch.motion_connection
            is not None
            and visual_selection.boundaries[
                index
            ].mismatch.motion_connection.both_moving
        )
    )
    non_highlight_cut_indices = tuple(
        index for index in applied_cut_indices if index not in highlight_boundary_set
    )
    highlight_hold_indices = tuple(
        index
        for index in highlight_boundary_indices
        if not visual_selection.boundaries[index].cut_applied
    )
    highlight_switches_per_minute = (
        len(highlight_cut_indices) * 60.0 / highlight_duration
        if highlight_duration > 1e-9
        else 0.0
    )
    non_highlight_switches_per_minute = (
        len(non_highlight_cut_indices) * 60.0 / non_highlight_duration
        if non_highlight_duration > 1e-9
        else 0.0
    )
    highlight_moving_cuts_per_minute = (
        len(highlight_moving_cut_indices) * 60.0 / highlight_duration
        if highlight_duration > 1e-9
        else 0.0
    )
    highlight_moving_cut_fraction = (
        len(highlight_moving_cut_indices) / len(highlight_cut_indices)
        if highlight_cut_indices
        else 0.0
    )

    highlight_gap_seconds: list[float] = []
    highlight_cut_times = tuple(
        items[index].canonical_master_end for index in highlight_cut_indices
    )
    for interval_start, interval_end in normalized:
        points = [interval_start]
        points.extend(
            value
            for value in highlight_cut_times
            if interval_start + 1e-7 < value < interval_end - 1e-7
        )
        points.append(interval_end)
        highlight_gap_seconds.extend(
            right - left
            for left, right in zip(points, points[1:])
            if right > left + 1e-9
        )

    source_runs: list[tuple[float, float]] = []
    run_start = items[0].canonical_master_start
    for boundary_index, boundary in enumerate(visual_selection.boundaries):
        if boundary.cut_applied:
            run_end = items[boundary_index].canonical_master_end
            source_runs.append((run_start, run_end))
            run_start = run_end
    source_runs.append((run_start, items[-1].canonical_master_end))
    highlight_run_overlaps = [
        overlap
        for run_start, run_end in source_runs
        for interval_start, interval_end in normalized
        if (
            overlap := max(
                0.0,
                min(run_end, interval_end) - max(run_start, interval_start),
            )
        )
        > 1e-9
    ]

    audit_required = (
        highlight_duration
        >= _DIRECT_MEMBER_HIGHLIGHT_MINIMUM_AUDIT_DURATION_SECONDS
    )
    failure_reasons: list[str] = []
    if audit_required:
        if (
            highlight_switches_per_minute
            + 1e-9
            < _DIRECT_MEMBER_HIGHLIGHT_MINIMUM_SWITCHES_PER_MINUTE
        ):
            failure_reasons.append("highlight_switch_density_below_minimum")
        if (
            non_highlight_duration
            >= _DIRECT_MEMBER_HIGHLIGHT_MINIMUM_AUDIT_DURATION_SECONDS
            and highlight_switches_per_minute + 1e-9
            < non_highlight_switches_per_minute
            + _DIRECT_MEMBER_HIGHLIGHT_MINIMUM_DENSITY_LIFT_PER_MINUTE
        ):
            failure_reasons.append("highlight_density_lift_below_minimum")
        if (
            len(highlight_hold_indices)
            > _DIRECT_MEMBER_MAXIMUM_COMPENSATED_HIGHLIGHT_HOLD_COUNT
        ):
            failure_reasons.append(
                "compensated_highlight_hold_budget_exceeded"
            )
        if (
            highlight_moving_cut_fraction + 1e-9
            < _DIRECT_MEMBER_HIGHLIGHT_MINIMUM_MOVING_CUT_FRACTION
        ):
            failure_reasons.append(
                "highlight_moving_cut_fraction_below_minimum"
            )
        if (
            highlight_moving_cuts_per_minute + 1e-9
            < _DIRECT_MEMBER_HIGHLIGHT_MINIMUM_MOVING_CUTS_PER_MINUTE
        ):
            failure_reasons.append(
                "highlight_moving_cut_density_below_minimum"
            )
    return {
        "contract": "human_like_section_aware_member_cut_density_v1",
        "audit_required": audit_required,
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "highlight_intervals": [list(value) for value in normalized],
        "highlight_duration_seconds": highlight_duration,
        "non_highlight_duration_seconds": non_highlight_duration,
        "planned_highlight_boundary_count": len(highlight_boundary_indices),
        "actual_highlight_cut_count": len(highlight_cut_indices),
        "actual_highlight_moving_cut_count": len(
            highlight_moving_cut_indices
        ),
        "highlight_moving_cut_boundary_indices": list(
            highlight_moving_cut_indices
        ),
        "actual_non_highlight_cut_count": len(non_highlight_cut_indices),
        "highlight_no_cut_hold_count": len(highlight_hold_indices),
        "highlight_no_cut_hold_boundary_indices": list(highlight_hold_indices),
        "maximum_compensated_highlight_hold_count": (
            _DIRECT_MEMBER_MAXIMUM_COMPENSATED_HIGHLIGHT_HOLD_COUNT
        ),
        "highlight_holds_require_redundant_cut_density": True,
        "highlight_source_switches_per_minute": (
            highlight_switches_per_minute
        ),
        "non_highlight_source_switches_per_minute": (
            non_highlight_switches_per_minute
        ),
        "density_lift_per_minute": (
            highlight_switches_per_minute
            - non_highlight_switches_per_minute
        ),
        "highlight_moving_cuts_per_minute": (
            highlight_moving_cuts_per_minute
        ),
        "highlight_moving_cut_fraction": highlight_moving_cut_fraction,
        "minimum_highlight_source_switches_per_minute": (
            _DIRECT_MEMBER_HIGHLIGHT_MINIMUM_SWITCHES_PER_MINUTE
        ),
        "minimum_density_lift_per_minute": (
            _DIRECT_MEMBER_HIGHLIGHT_MINIMUM_DENSITY_LIFT_PER_MINUTE
        ),
        "minimum_highlight_moving_cut_fraction": (
            _DIRECT_MEMBER_HIGHLIGHT_MINIMUM_MOVING_CUT_FRACTION
        ),
        "minimum_highlight_moving_cuts_per_minute": (
            _DIRECT_MEMBER_HIGHLIGHT_MINIMUM_MOVING_CUTS_PER_MINUTE
        ),
        "target_highlight_run_seconds": [1.8, 2.6],
        "highlight_cut_gap_median_seconds": (
            median(highlight_gap_seconds) if highlight_gap_seconds else None
        ),
        "highlight_cut_gap_maximum_seconds": (
            max(highlight_gap_seconds) if highlight_gap_seconds else None
        ),
        "highlight_longest_source_run_seconds": (
            max(highlight_run_overlaps) if highlight_run_overlaps else 0.0
        ),
        "cut_times_use_existing_musical_boundaries": True,
        "visual_gates_relaxed": False,
    }


def _direct_member_visual_payload(
    selection: DirectMemberVisualSelection,
    *,
    eligible_indices: Sequence[int],
) -> dict[str, Any]:
    """Add request-domain aliases without obscuring compact selector indices."""

    eligible = tuple(int(index) for index in eligible_indices)
    payload = selection.to_dict()
    for item in payload["source_usage"]:
        eligible_source_index = int(item["eligible_source_index"])
        item["request_source_index"] = eligible[eligible_source_index]
    for span in payload["span_content_validity"]:
        for item in span["sources"]:
            eligible_source_index = int(item["eligible_source_index"])
            item["request_source_index"] = eligible[eligible_source_index]
    for item in payload["boundaries"]:
        outgoing = int(item["outgoing_eligible_source_index"])
        incoming = int(item["incoming_eligible_source_index"])
        item["outgoing_request_source_index"] = eligible[outgoing]
        item["incoming_request_source_index"] = eligible[incoming]
    return payload


def _direct_member_selection_failure_summary(
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Compress selector diagnostics for one machine-readable error event."""

    reason_counts: dict[str, int] = {}
    affected_segments: set[int] = set()
    affected_sources: set[int] = set()
    invalid_candidate_count = 0
    spans = diagnostics.get("span_content_validity", ())
    if isinstance(spans, list):
        for span in spans:
            if not isinstance(span, dict):
                continue
            segment_index = span.get("segment_index")
            sources = span.get("sources", ())
            if not isinstance(sources, list):
                continue
            for source in sources:
                if not isinstance(source, dict) or source.get("valid") is not False:
                    continue
                invalid_candidate_count += 1
                if isinstance(segment_index, int):
                    affected_segments.add(segment_index)
                source_index = source.get("eligible_source_index")
                if isinstance(source_index, int):
                    affected_sources.add(source_index)
                reasons = source.get("rejection_reasons", ())
                if isinstance(reasons, list):
                    for reason in reasons:
                        if isinstance(reason, str):
                            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    for shortfall in diagnostics.get("quota_shortfalls", ()):
        if isinstance(shortfall, dict):
            source_index = shortfall.get("eligible_source_index")
            if isinstance(source_index, int):
                affected_sources.add(source_index)
    failed_segment = diagnostics.get("failed_at_segment_index")
    if isinstance(failed_segment, int):
        affected_segments.add(failed_segment)
    return {
        "failure": diagnostics.get("failure", "unknown_direct_selection_failure"),
        "failed_segment_index": failed_segment,
        "invalid_candidate_count": invalid_candidate_count,
        "affected_segment_indices": sorted(affected_segments),
        "affected_eligible_source_indices": sorted(affected_sources),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "quota_shortfall_count": len(diagnostics.get("quota_shortfalls", ())),
        "empty_segment_count": len(diagnostics.get("empty_segment_indices", ())),
    }


def _canonical_json_content_fingerprint(value: Any) -> dict[str, Any]:
    """Hash canonical JSON without materializing another large string."""

    digest = hashlib.sha256()
    size_bytes = 0
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    for text_chunk in encoder.iterencode(value):
        chunk = text_chunk.encode("utf-8")
        digest.update(chunk)
        size_bytes += len(chunk)
    return {
        "contract": _DIRECT_MEMBER_FAILURE_DIAGNOSTIC_FINGERPRINT_CONTRACT,
        "sha256": digest.hexdigest(),
        "canonical_size_bytes": size_bytes,
    }


def _bounded_pretty_json_bytes(
    value: Any,
    *,
    maximum_bytes: int,
) -> tuple[bytes | None, dict[str, Any]]:
    """Serialize at most ``maximum_bytes`` while hashing the full payload."""

    if maximum_bytes <= 0:
        raise ValueError("JSON byte bound must be positive")
    digest = hashlib.sha256()
    size_bytes = 0
    chunks: list[bytes] = []
    overflowed = False
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    for text_chunk in encoder.iterencode(value):
        chunk = text_chunk.encode("utf-8")
        digest.update(chunk)
        size_bytes += len(chunk)
        if not overflowed and size_bytes <= maximum_bytes:
            chunks.append(chunk)
        else:
            overflowed = True
            chunks.clear()
    fingerprint = {
        "sha256": digest.hexdigest(),
        "serialized_size_bytes": size_bytes,
    }
    return (None if overflowed else b"".join(chunks), fingerprint)


def _direct_member_failure_report_fallback_payload(
    payload: Mapping[str, Any],
    *,
    oversized_report_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    """Retain hash-bound scalar proof when a report exceeds its hard cap."""

    diagnostics_raw = payload.get("diagnostics", {})
    diagnostics = (
        diagnostics_raw if isinstance(diagnostics_raw, dict) else {}
    )
    partition_raw = diagnostics.get("partition_search", {})
    partition = partition_raw if isinstance(partition_raw, dict) else {}
    profiles_raw = partition.get("profiles_attempted", ())
    profiles = profiles_raw if isinstance(profiles_raw, list) else []
    compact_partition = {
        key: partition.get(key)
        for key in (
            "policy",
            "initial_profile_count",
            "maximum_profile_count",
            "static_profile_upper_bound",
            "absolute_profile_upper_bound",
            "maximum_profile_limit_expansion_count",
            "profile_limit_expansions",
            "maximum_shifted_boundary_count",
            "base_maximum_shifted_boundary_count",
            "shift_capacity_contract",
            "proof_directed_window_redistribution_enabled",
            "maximum_proof_directed_boundary_merge_count",
            "maximum_visual_frontier_split_count_per_profile",
            "maximum_counterfactual_visual_frontier_split_count_per_profile",
            "profile_cap_reached",
            "generated_profile_count",
            "canonical_signature_deduplication_count",
            "proof_equivalent_future_state_dominance_count",
            "future_operation_state_count",
            "shift_depth_pruned_parent_count",
            "input_bound_partition_warm_start",
            "exact_lifeline_capacity_retry",
            "strict_extra_quota_assignment_search",
            "last_resort_safe_hold_search",
            "severe_transition_counterfactual_secondary_partition",
            "queue_order",
            "render_started",
        )
        if key in partition
    }
    compact_partition["profiles_attempted_storage"] = {
        "records_persisted": False,
        "record_count": len(profiles),
        "omission_reason": "failure_report_hard_cap_exceeded",
        "content_fingerprint": _canonical_json_content_fingerprint(profiles),
    }
    compact_diagnostics = {
        "failure": diagnostics.get("failure"),
        "failed_at_segment_index": diagnostics.get("failed_at_segment_index"),
        "constraint_infeasibility_proven": diagnostics.get(
            "constraint_infeasibility_proven"
        ),
        "profile_outcome_counts": diagnostics.get("profile_outcome_counts"),
        "visual_gates_relaxed": diagnostics.get("visual_gates_relaxed", False),
        "latest_failure_summary": _direct_member_selection_failure_summary(
            diagnostics
        ),
        "original_diagnostics_content_fingerprint": (
            _canonical_json_content_fingerprint(diagnostics)
        ),
        "latest_severe_transition_counterfactual": (
            _compact_direct_severe_counterfactual(diagnostics)
        ),
        "partition_search": compact_partition,
    }
    return {
        key: value
        for key, value in payload.items()
        if key not in {"diagnostics", "report_storage"}
    } | {
        "diagnostics": compact_diagnostics,
        "report_storage": {
            "contract": _DIRECT_MEMBER_FAILURE_REPORT_STORAGE_CONTRACT,
            "hard_cap_bytes": _DIRECT_MEMBER_FAILURE_REPORT_MAX_BYTES,
            "bounded_fallback_used": True,
            "oversized_report_fingerprint": dict(
                oversized_report_fingerprint
            ),
        },
    }


def _persist_direct_member_selection_failure(
    error: DirectMemberVisualSelectionError,
    *,
    output_directory: Path,
    base_filename: str,
    target_member: str,
    eligible_indices: Sequence[int],
) -> tuple[Path, dict[str, Any]]:
    """Atomically preserve fail-closed selector evidence outside transient stderr."""

    summary = _direct_member_selection_failure_summary(error.diagnostics)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "stage": "automatic_member_direct_visual_selection",
        "message": str(error),
        "diagnostics_source_index_domain": ELIGIBLE_SOURCE_INDEX_DOMAIN,
        "eligible_request_source_indices": [int(index) for index in eligible_indices],
        "summary": summary,
        "diagnostics": error.diagnostics,
        "privacy": {
            "face_identity_inference_used": False,
            "face_appearance_embeddings_persisted": False,
            "raw_pose_silhouette_layout_embeddings_persisted": False,
        },
        "report_storage": {
            "contract": _DIRECT_MEMBER_FAILURE_REPORT_STORAGE_CONTRACT,
            "hard_cap_bytes": _DIRECT_MEMBER_FAILURE_REPORT_MAX_BYTES,
            "bounded_fallback_used": False,
        },
    }
    path = output_directory / (
        f"{base_filename}-{_safe_slug(target_member)}-"
        "direct-selection-failure.json"
    )
    temporary = path.with_name(f".{path.name}.tmp")
    encoded, fingerprint = _bounded_pretty_json_bytes(
        payload,
        maximum_bytes=_DIRECT_MEMBER_FAILURE_REPORT_MAX_BYTES,
    )
    if encoded is None:
        fallback = _direct_member_failure_report_fallback_payload(
            payload,
            oversized_report_fingerprint=fingerprint,
        )
        encoded, _fallback_fingerprint = _bounded_pretty_json_bytes(
            fallback,
            maximum_bytes=_DIRECT_MEMBER_FAILURE_REPORT_MAX_BYTES,
        )
        if encoded is None:
            raise RuntimeError(
                "bounded direct-member failure manifest exceeds its hard cap"
            )
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return (path, summary)


def _report_direct_member_selection_failure(
    error: DirectMemberVisualSelectionError,
    *,
    output_directory: Path,
    base_filename: str,
    target_member: str,
    eligible_indices: Sequence[int],
    reporter: EventReporter,
) -> DirectMemberVisualSelectionError:
    """Persist full evidence, emit its path, and retain it in CLI text."""

    diagnostics_path, failure_summary = _persist_direct_member_selection_failure(
        error,
        output_directory=output_directory,
        base_filename=base_filename,
        target_member=target_member,
        eligible_indices=eligible_indices,
    )
    reporter.emit(
        "error",
        message=(
            "직캠 시각 안전 조건과 소스 균형을 동시에 만족하지 "
            "못해 렌더를 시작하지 않았습니다. 진단: "
            f"{diagnostics_path}"
        ),
        diagnostics_path=str(diagnostics_path),
        failure_summary=failure_summary,
    )
    return DirectMemberVisualSelectionError(
        f"{error}; diagnostics: {diagnostics_path}",
        error.diagnostics,
    )


def _compact_direct_partition_failure(
    error: DirectMemberVisualSelectionError,
    *,
    span_count: int,
) -> dict[str, Any]:
    """Persist one bounded, hash-bound scalar failure record.

    Full selector diagnostics remain available to the in-process proof search,
    but are never copied once per attempted partition into the failure artifact.
    """

    diagnostics = error.diagnostics
    search = diagnostics.get("search", {})
    exact: dict[str, Any] = {}
    if isinstance(search, dict):
        raw_exact = search.get("exact_lifeline", {})
        if isinstance(raw_exact, dict):
            exact = {
                key: raw_exact.get(key)
                for key in (
                    "path_found",
                    "explored_state_count",
                    "memoized_failed_state_count",
                    "state_capacity_exhausted",
                    "duration_state_quantization_applied",
                )
                if key in raw_exact
            }
            roots = raw_exact.get("root_candidate_diagnostics", ())
            if isinstance(roots, list):
                outcome_counts: dict[str, int] = {}
                for root in roots:
                    if not isinstance(root, dict):
                        continue
                    outcome = root.get("search_outcome")
                    if isinstance(outcome, str):
                        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
                exact["root_candidate_count"] = len(roots)
                exact["root_search_outcome_counts"] = dict(
                    sorted(outcome_counts.items())
                )
    connectivity = diagnostics.get("hard_boundary_connectivity", {})
    zero_boundaries: list[int] = []
    if isinstance(connectivity, dict):
        raw_zero = connectivity.get(
            "zero_local_nonsevere_transition_boundary_indices"
        )
        if raw_zero is None:
            raw_zero = connectivity.get(
                "zero_reachable_transition_boundary_indices",
                (),
            )
        if isinstance(raw_zero, list):
            zero_boundaries = [
                int(value) for value in raw_zero if isinstance(value, int)
            ]
    next_proofs = _direct_partition_exact_shift_proofs(
        diagnostics,
        span_count=span_count,
    )
    raw_valid_counts = diagnostics.get("valid_count_by_source", ())
    compact_valid_counts: list[dict[str, int]] = []
    if isinstance(raw_valid_counts, list):
        for raw in raw_valid_counts:
            if not isinstance(raw, dict):
                continue
            scalar: dict[str, int] = {}
            for key in (
                "eligible_source_index",
                "quota",
                "valid_span_count",
                "invalid_span_count",
            ):
                value = raw.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    scalar[key] = value
            if scalar:
                compact_valid_counts.append(scalar)
    return {
        "storage_contract": "hash_bound_scalar_partition_failure_v1",
        "full_diagnostics_persisted": False,
        "diagnostics_content_fingerprint": (
            _canonical_json_content_fingerprint(diagnostics)
        ),
        "message": str(error),
        "failure": diagnostics.get("failure"),
        "failed_at_segment_index": diagnostics.get(
            "failed_at_segment_index"
        ),
        "constraint_infeasibility_proven": diagnostics.get(
            "constraint_infeasibility_proven",
            _direct_partition_failure_is_proven(diagnostics),
        ),
        "visual_gates_relaxed": False,
        "summary": _direct_member_selection_failure_summary(diagnostics),
        "zero_local_nonsevere_transition_boundary_indices": zero_boundaries,
        "next_exact_shift_proofs": [
            {
                "boundary_index": proof.boundary_index,
                "trigger": proof.trigger,
                "failed_segment_index": proof.failed_segment_index,
                "forward_reachable_left_source_indices": list(
                    proof.forward_reachable_left_source_indices
                ),
                "left_valid_source_indices": list(
                    proof.left_valid_source_indices
                ),
                "right_valid_source_indices": list(
                    proof.right_valid_source_indices
                ),
                "local_nonsevere_cross_source_edge_count": (
                    proof.local_nonsevere_edge_count
                ),
            }
            for proof in next_proofs
        ],
        "exact_lifeline": exact,
        "severe_transition_counterfactual": (
            _compact_direct_severe_counterfactual(diagnostics)
        ),
        # Per-segment index arrays remain available in the one latest runtime
        # diagnostic.  Repeating them for every attempted profile made an
        # otherwise scalar failure ledger grow quadratically with search depth.
        "valid_count_by_source": compact_valid_counts,
    }


def _compact_direct_severe_counterfactual(
    diagnostics: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Retain bounded minimum-severe proof scalars in compact reports."""

    search = diagnostics.get("search")
    if not isinstance(search, dict):
        return None
    raw = search.get("severe_transition_counterfactual")
    if not isinstance(raw, dict):
        return None
    compact = {
        key: raw.get(key)
        for key in (
            "contract",
            "optimization_contract",
            "severe_transition_cost_policy",
            "tie_break_contract",
            "invoked",
            "original_hard_constraint_sha256",
            "original_failure_certificate_sha256",
            "severe_disabled_hard_constraint_sha256",
            "severe_transition_budget_mask_sha256",
            "minimum_witness_path_sha256",
            "minimum_severe_transition_count",
            "minimum_severe_transition_count_lower_bound",
            "maximum_diagnostic_severe_transition_budget",
            "optimality_proven",
            "production_single_split_proposal_authorized",
            "witness_found",
            "integrity_check_passed",
            "blocking_boundary_indices",
            "blocking_transitions",
            "maximum_duration_relative_deviation",
            "state_limit_per_budget",
            "explored_state_count",
            "memoized_failed_state_count",
            "state_capacity_exhausted",
            "duration_state_quantization_applied",
            "selection_eligible",
            "render_authorized",
            "visual_gates_relaxed",
        )
        if key in raw
    }
    for key in (
        "zero_budget_hard_proof",
        "lower_budget_hard_proofs",
        "budget_attempts",
    ):
        value = raw.get(key)
        if isinstance(value, (dict, list)):
            compact[key] = value
    compact["full_counterfactual_content_fingerprint"] = (
        _canonical_json_content_fingerprint(raw)
    )
    return compact


def _direct_partition_selection_rank(
    profile: _DirectFullFramePartitionProfile,
    selection: DirectMemberVisualSelection,
) -> tuple[int, float, int, int, str]:
    payload = selection.to_dict()
    aggregate = payload.get("aggregate", {})
    warning_count = (
        int(aggregate.get("warning_boundary_count", 0))
        if isinstance(aggregate, dict)
        else 0
    )
    optimizer_cost = float(
        getattr(selection, "optimizer_cost", payload.get("optimizer_cost", 0.0))
    )
    return (
        warning_count,
        optimizer_cost,
        profile.changed_span_count,
        profile.musical_boundary_penalty,
        profile.profile_id,
    )


def _direct_partition_structural_search_cost(
    profile: _DirectFullFramePartitionProfile,
) -> tuple[int, int, int]:
    """Return the proof-preserving edit cost independent of queue order."""

    return (
        profile.structural_change_count,
        profile.total_unit_displacement,
        profile.musical_boundary_penalty,
    )


def _direct_partition_proof_progress(
    profile: _DirectFullFramePartitionProfile,
) -> int:
    """Return the furthest exact parent failure represented by a profile."""

    operation = profile.latest_proof_operation
    if operation is None:
        return -1
    candidates = [
        getattr(operation, "proof_failed_segment_index", None),
    ]
    frontier = getattr(operation, "proof_frontier_boundary_index", None)
    if isinstance(frontier, int):
        candidates.append(frontier + 1)
    boundary = getattr(operation, "boundary_index", None)
    if isinstance(boundary, int):
        candidates.append(boundary + 1)
    return max(
        (value for value in candidates if isinstance(value, int)),
        default=-1,
    )


def _direct_partition_terminal_candidate_progress(
    profile: _DirectFullFramePartitionProfile,
    error: DirectMemberVisualSelectionError,
) -> int:
    """Rank a terminal candidate by its current proved reachable prefix.

    A profile can contain an old repair at a far-away boundary while its latest
    full re-selection still fails near the start.  That historical operation
    location is useful for queue exploration, but must never outrank a profile
    whose current hard-feasible path actually reaches farther through the song.
    """

    failed = error.diagnostics.get("failed_at_segment_index")
    if (
        isinstance(failed, int)
        and not isinstance(failed, bool)
        and failed >= 0
    ):
        return failed
    return _direct_partition_proof_progress(profile)


def _direct_partition_queue_priority(
    profile: _DirectFullFramePartitionProfile,
) -> tuple[int, int, int, int]:
    """Prefer the furthest proved frontier before cheaper sibling variants.

    Structural cost is still retained in the key and separately enforced once
    a feasible profile exists.  This changes only discovery order: it prevents
    hundreds of shallow siblings from delaying an already advancing exact
    proof path.
    """

    return (
        -_direct_partition_proof_progress(profile),
        *_direct_partition_structural_search_cost(profile),
    )


def _direct_partition_terminal_audition_due(
    *,
    evaluated_profile_count: int,
    feasible_profile_count: int,
    queued_profile_count: int,
) -> bool:
    """Bound strict expansion before the audited hold/effect audition.

    Crossing this budget is not a visual-gate relaxation and is not reported
    as hard infeasibility: it transfers the best proven failures to the
    independently constrained terminal selector.
    """

    for name, value in (
        ("evaluated_profile_count", evaluated_profile_count),
        ("feasible_profile_count", feasible_profile_count),
        ("queued_profile_count", queued_profile_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    return bool(
        feasible_profile_count == 0
        and queued_profile_count > 0
        and evaluated_profile_count
        >= _DIRECT_MEMBER_MAX_STRICT_TERMINAL_AUDITION_COUNT
    )


def _direct_visual_split_exact_lifeline_retry_is_authorized(
    profile: _DirectFullFramePartitionProfile,
    error: DirectMemberVisualSelectionError,
    *,
    initial_state_limit: int,
) -> bool:
    """Authorize one larger exact search only on a proved capacity boundary."""

    if not profile.visual_frontier_splits:
        return False
    diagnostics = error.diagnostics
    if (
        diagnostics.get("failure") != "search_capacity_exhausted"
        or diagnostics.get("constraint_infeasibility_proven") is not False
    ):
        return False
    search = diagnostics.get("search")
    if not isinstance(search, dict):
        return False
    exact = search.get("exact_lifeline")
    if not isinstance(exact, dict):
        return False
    return bool(
        exact.get("invoked") is True
        and exact.get("state_capacity_exhausted") is True
        and exact.get("duration_state_quantization_applied") is False
        and exact.get("explored_state_count") == initial_state_limit
        and initial_state_limit
        < _DIRECT_MEMBER_VISUAL_SPLIT_EXACT_LIFELINE_RETRY_STATE_LIMIT
    )


def _direct_strict_extra_quota_assignment_candidates(
    *,
    span_count: int,
    source_count: int,
    source_order: Sequence[int],
    diagnostics: Mapping[str, Any],
) -> tuple[tuple[int, ...], ...]:
    """Enumerate omitted exactly-balanced quota owners by validity slack.

    Source order remains immutable.  Only the set receiving the remainder of
    ``span_count / source_count`` changes, so every candidate still uses every
    source either ``base`` or ``base + 1`` times.  Enumeration is admitted only
    when the complete combination domain fits the fixed finite bound and the
    selector supplied a valid-count certificate for every source.
    """

    order = tuple(int(index) for index in source_order)
    if (
        source_count < 2
        or len(order) != source_count
        or set(order) != set(range(source_count))
    ):
        return ()
    base, extra = divmod(int(span_count), source_count)
    if extra <= 0:
        return ()
    combination_count = math.comb(source_count, extra)
    if combination_count > _DIRECT_MEMBER_MAX_EXTRA_QUOTA_ASSIGNMENT_COUNT:
        return ()
    raw_valid_counts = diagnostics.get("valid_count_by_source")
    if not isinstance(raw_valid_counts, list):
        return ()
    valid_counts: dict[int, int] = {}
    for item in raw_valid_counts:
        if not isinstance(item, dict):
            continue
        source_index = item.get("eligible_source_index")
        valid_span_count = item.get("valid_span_count")
        if isinstance(source_index, int) and isinstance(valid_span_count, int):
            valid_counts[source_index] = valid_span_count
    if set(valid_counts) != set(range(source_count)):
        return ()

    order_rank = {source_index: rank for rank, source_index in enumerate(order)}
    default_extra_set = frozenset(order[:extra])
    candidates: list[tuple[tuple[Any, ...], tuple[int, ...]]] = []
    for raw_assignment in combinations(range(source_count), extra):
        assignment_set = frozenset(raw_assignment)
        if assignment_set == default_extra_set:
            continue
        assignment = tuple(
            source_index for source_index in order if source_index in assignment_set
        )
        slacks = sorted(
            valid_counts[source_index]
            - (base + int(source_index in assignment_set))
            for source_index in range(source_count)
        )
        # Maximize the weakest validity slack first, then the next weakest.
        # Source-order rank is only the deterministic final tie breaker.
        priority = (
            tuple(-value for value in slacks),
            tuple(order_rank[index] for index in assignment),
        )
        candidates.append((priority, assignment))
    candidates.sort(key=lambda item: item[0])
    return tuple(assignment for _priority, assignment in candidates)


def _direct_strict_extra_quota_assignment_failure_is_proven(
    diagnostics: Mapping[str, Any],
) -> bool:
    """Accept only hard, non-quantized evidence for an alternate quota set."""

    failure = diagnostics.get("failure")
    if failure == "content_validity_quota_precheck_failed":
        return True
    if failure == "boundary_diagnostic_eligible_pair_exhausted":
        return True
    if (
        failure
        not in {
            "validity_quota_path_exhausted",
            "duration_balance_threshold_unsatisfied",
        }
        or diagnostics.get("constraint_infeasibility_proven") is not True
    ):
        return False
    search = diagnostics.get("search")
    if not isinstance(search, dict):
        return False
    exact = search.get("exact_lifeline")
    return bool(
        isinstance(exact, dict)
        and exact.get("invoked") is True
        and exact.get("state_capacity_exhausted") is False
        and exact.get("duration_state_quantization_applied") is False
    )


def _select_direct_member_partition_profile(
    *,
    timing_draft: GroupDraftTimeline,
    eligible_indices: Sequence[int],
    eligible_sources: Sequence[Path],
    source_timewarps: Sequence[TimeWarpMap | None] | None,
    scene_cut_times: Sequence[Sequence[float]],
    source_cycle: Sequence[int],
    highlight_intervals: Sequence[tuple[float, float]] = (),
    continuous_official_portrait_fancam_contract: bool = False,
    cinematic_bridge_enabled: bool = False,
    eligible_source_frame_sizes: Sequence[tuple[int, int]] | None = None,
    output_frame_size: tuple[int, int] = (1080, 1920),
    feature_cache_directory: Path | None = None,
    descriptor_provider: DescriptorProvider | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[
    _DirectFullFramePartitionProfile,
    DirectMemberVisualSelection,
    dict[str, Any],
]:
    """Try one fixed cadence, then bounded proof-directed beat partitions."""

    eligible = tuple(int(index) for index in eligible_indices)
    compact_timewarps = (
        tuple(source_timewarps[index] for index in eligible)
        if source_timewarps is not None
        else None
    )
    normalized_highlight_intervals = (
        _normalized_direct_member_highlight_intervals(
            highlight_intervals,
            range_start=timing_draft.common_master_start,
            range_end=timing_draft.common_master_end,
        )
        if highlight_intervals
        else ()
    )
    undensified_base_span_count = len(
        _coarsen_direct_full_frame_spans(timing_draft.segments)
    )
    base = _direct_full_frame_base_partition_profile(
        timing_draft.segments,
        highlight_intervals=normalized_highlight_intervals,
    )
    if len(base.spans) < len(source_cycle):
        raise RuntimeError("coarse direct cadence cannot show every eligible source")
    if continuous_official_portrait_fancam_contract:
        warm_start_profiles, warm_start_diagnostics = (
            _load_direct_member_partition_warm_start_profiles(
                feature_cache_directory,
                timing_draft.segments,
                base,
                source_count=len(eligible_sources),
            )
        )
    else:
        warm_start_profiles = ()
        warm_start_diagnostics = {
            "contract": _DIRECT_MEMBER_PARTITION_WARM_START_CONTRACT,
            "requested": False,
            "status": "continuous_official_member_contract_required",
            "loaded_candidate_count": 0,
            "full_selector_recheck_required": True,
            "visual_gates_relaxed": False,
        }
    warm_start_diagnostics["candidate_evaluations"] = []
    partition_search_max_boundary_merge_count = (
        _direct_partition_proof_directed_boundary_merge_limit(base)
    )
    partition_search_max_shifted_boundary_count = min(
        max(0, len(base.spans) - 1),
        _DIRECT_MEMBER_MAX_SHIFTED_BOUNDARY_COUNT
        + partition_search_max_boundary_merge_count,
    )
    partition_search_max_limit_expansion_count = (
        partition_search_max_shifted_boundary_count
        + partition_search_max_boundary_merge_count
    )

    owned_provider: _VideoDescriptorProvider | None = None
    provider = descriptor_provider
    if provider is None:
        owned_provider = _VideoDescriptorProvider(
            eligible_sources,
            feature_cache_directory=feature_cache_directory,
        )
        provider = owned_provider

    attempted: list[dict[str, Any]] = []
    feasible: list[
        tuple[_DirectFullFramePartitionProfile, DirectMemberVisualSelection]
    ] = []
    failure_count = 0
    all_failures_proven = True
    latest_failure_diagnostics: dict[str, Any] | None = None
    selection_config = DirectMemberVisualConfig(
        boundary_search_radius_frames=6,
        boundary_search_step_frames=2,
    )
    partition_search_cap_reached = False
    partition_search_evaluation_limit = min(
        _DIRECT_MEMBER_MAX_PARTITION_PROFILE_COUNT,
        _DIRECT_MEMBER_MAX_PARTITION_PROFILE_UPPER_BOUND,
    )
    partition_search_limit_expansions: list[dict[str, Any]] = []
    partition_search_generated_profile_count = 1 + len(warm_start_profiles)
    partition_search_signature_deduplication_count = 0
    partition_search_future_state_dominance_count = 0
    partition_search_future_operation_state_count = 0
    partition_search_shift_depth_pruned_count = 0
    partition_search_exact_lifeline_retry_count = 0
    partition_search_exact_lifeline_retry_success_count = 0
    partition_search_exact_lifeline_retry_capacity_exhausted_count = 0
    partition_search_extra_quota_profile_count = 0
    partition_search_extra_quota_assignment_evaluation_count = 0
    partition_search_extra_quota_feasible_profile_count = 0
    partition_search_extra_quota_exhausted_profile_count = 0
    partition_search_extra_quota_capacity_exhausted_profile_count = 0
    partition_search_counterfactual_qualified_parent_count = 0
    partition_search_counterfactual_generated_profile_count = 0
    partition_search_counterfactual_evaluation_count = 0
    partition_search_counterfactual_feasible_profile_count = 0
    partition_search_counterfactual_hard_infeasible_profile_count = 0
    partition_search_counterfactual_capacity_exhausted_profile_count = 0
    partition_search_counterfactual_unproven_failure_profile_count = 0
    partition_search_counterfactual_promoted_queued_profile_count = 0
    partition_search_terminal_safe_hold_candidate_count = 0
    partition_search_terminal_safe_hold_evaluation_count = 0
    partition_search_terminal_safe_hold_success_count = 0
    partition_search_strict_terminal_audition_limit_reached = False
    partition_search_strict_terminal_audition_pruned_profile_count = 0
    partition_search_post_feasible_cost_pruned_profile_count = 0
    partition_search_counterfactual_authorization_proofs: list[
        dict[str, Any]
    ] = []
    partition_search_primary_evaluation_count = 0

    def _emit_progress(**payload: Any) -> None:
        if progress_callback is not None:
            progress_callback(
                {
                    "schema_version": 1,
                    "event": "direct_member_partition_profile_evaluation",
                    **payload,
                }
            )

    def _profile_progress_payload(
        profile: _DirectFullFramePartitionProfile,
        *,
        evaluation_index: int,
    ) -> dict[str, Any]:
        latest_operation = profile.latest_proof_operation
        return {
            "evaluation_index": evaluation_index,
            "evaluation_limit": partition_search_evaluation_limit,
            "profile_id": profile.profile_id,
            "shift_depth": profile.proof_directed_boundary_motion_count,
            "ordinary_shift_count": len(profile.boundary_shifts),
            "window_redistribution_count": len(
                profile.window_redistributions
            ),
            "visual_frontier_split_count": len(
                profile.visual_frontier_splits
            ),
            "proof_directed_boundary_merge_count": len(
                profile.boundary_merges
            ),
            "changed_boundary_count": profile.structural_change_count,
            "total_unit_displacement": profile.total_unit_displacement,
            "partition_signature_sha256": (
                _direct_partition_signature_sha256(profile.unit_counts)
            ),
            "parent_profile_id": (
                latest_operation.parent_profile_id
                if latest_operation is not None
                else None
            ),
            "parent_failed_segment_index": (
                latest_operation.proof_failed_segment_index
                if latest_operation is not None
                else None
            ),
            "proof_frontier_boundary_index": (
                getattr(
                    latest_operation,
                    "proof_frontier_boundary_index",
                    getattr(latest_operation, "boundary_index", None),
                )
                if latest_operation is not None
                else None
            ),
            "shifted_boundaries": [
                {
                    "boundary_index": shift.boundary_index,
                    "shift_units": shift.shift_units,
                }
                for shift in profile.boundary_shifts
            ],
            "window_redistributions": [
                {
                    "window_start_span_index": (
                        redistribution.window_start_span_index
                    ),
                    "proof_frontier_boundary_index": (
                        redistribution.proof_frontier_boundary_index
                    ),
                    "original_unit_counts": list(
                        redistribution.original_unit_counts
                    ),
                    "replacement_unit_counts": list(
                        redistribution.replacement_unit_counts
                    ),
                    "boundary_displacement_units": (
                        redistribution.boundary_displacement_units
                    ),
                }
                for redistribution in profile.window_redistributions
            ],
            "visual_frontier_splits": [
                {
                    "parent_span_index": split.parent_span_index,
                    "split_unit_offset": split.split_unit_offset,
                    "canonical_master_seconds": split.canonical_master_seconds,
                    "evidence_contract": split.evidence_contract,
                    "evidence_kind": split.evidence_kind,
                }
                for split in profile.visual_frontier_splits
            ],
            "proof_directed_boundary_merges": [
                {
                    "boundary_index": merge.boundary_index,
                    "merged_unit_count": merge.merged_unit_count,
                    "canonical_master_seconds": (
                        merge.canonical_master_seconds
                    ),
                }
                for merge in profile.boundary_merges
            ],
        }

    def _select_profile_with_config(
        profile: _DirectFullFramePartitionProfile,
        config: DirectMemberVisualConfig,
        *,
        extra_quota_source_indices: Sequence[int] | None = None,
        allow_safe_hold_search: bool = False,
    ) -> DirectMemberVisualSelection:
        """Run one partition, deferring no-cut holds unless explicitly terminal."""

        defer_safe_hold_search = bool(
            continuous_official_portrait_fancam_contract
            and not allow_safe_hold_search
        )
        effective_config = (
            replace(config, maximum_same_source_hold_count=0)
            if defer_safe_hold_search
            else config
        )
        priority_cut_boundary_indices = (
            _direct_member_highlight_priority_boundary_indices(
                profile.spans,
                normalized_highlight_intervals,
            )
        )
        highlight_duration_seconds = sum(
            end - start for start, end in normalized_highlight_intervals
        )
        minimum_required_highlight_cut_count = math.ceil(
            highlight_duration_seconds
            * _DIRECT_MEMBER_HIGHLIGHT_MINIMUM_SWITCHES_PER_MINUTE
            / 60.0
        )
        maximum_priority_cut_hold_count = min(
            _DIRECT_MEMBER_MAXIMUM_COMPENSATED_HIGHLIGHT_HOLD_COUNT,
            max(
                0,
                len(priority_cut_boundary_indices)
                - minimum_required_highlight_cut_count,
            ),
        )
        return select_balanced_direct_member_sources(
            sources=eligible_sources,
            canonical_spans=tuple(
                (span.canonical_master_start, span.canonical_master_end)
                for span in profile.spans
            ),
            audio_offsets=timing_draft.audio_offsets,
            source_timewarps=compact_timewarps,
            source_scene_cut_times=scene_cut_times,
            source_frame_sizes=eligible_source_frame_sizes,
            output_frame_size=output_frame_size,
            continuous_official_portrait_fancam_contract=(
                continuous_official_portrait_fancam_contract
            ),
            cinematic_bridge_enabled=bool(
                allow_safe_hold_search and cinematic_bridge_enabled
            ),
            preferred_source_order=source_cycle,
            extra_quota_source_indices=extra_quota_source_indices,
            priority_cut_boundary_indices=priority_cut_boundary_indices,
            maximum_priority_cut_hold_count=maximum_priority_cut_hold_count,
            config=effective_config,
            descriptor_provider=provider,
        )

    def _evaluate(
        profile: _DirectFullFramePartitionProfile,
    ) -> DirectMemberVisualSelectionError | None:
        nonlocal failure_count
        nonlocal all_failures_proven
        nonlocal latest_failure_diagnostics
        nonlocal partition_search_exact_lifeline_retry_count
        nonlocal partition_search_exact_lifeline_retry_success_count
        nonlocal partition_search_exact_lifeline_retry_capacity_exhausted_count
        nonlocal partition_search_extra_quota_profile_count
        nonlocal partition_search_extra_quota_assignment_evaluation_count
        nonlocal partition_search_extra_quota_feasible_profile_count
        nonlocal partition_search_extra_quota_exhausted_profile_count
        nonlocal partition_search_extra_quota_capacity_exhausted_profile_count
        evaluation_index = len(attempted) + 1
        progress_payload = _profile_progress_payload(
            profile,
            evaluation_index=evaluation_index,
        )
        _emit_progress(
            phase="started",
            result=None,
            **progress_payload,
        )

        selection: DirectMemberVisualSelection | None = None
        error: DirectMemberVisualSelectionError | None = None
        exact_lifeline_retry: dict[str, Any] | None = None
        strict_extra_quota_assignment_search: dict[str, Any] | None = None
        try:
            selection = _select_profile_with_config(profile, selection_config)
        except DirectMemberVisualSelectionError as caught:
            error = caught
        if error is not None and _direct_visual_split_exact_lifeline_retry_is_authorized(
            profile,
            error,
            initial_state_limit=selection_config.feasibility_lifeline_state_limit,
        ):
            retry_config = replace(
                selection_config,
                feasibility_lifeline_state_limit=(
                    _DIRECT_MEMBER_VISUAL_SPLIT_EXACT_LIFELINE_RETRY_STATE_LIMIT
                ),
            )
            original_exact = error.diagnostics.get("search", {}).get(
                "exact_lifeline",
                {},
            )
            exact_lifeline_retry = {
                "contract": (
                    "visual_frontier_exact_lifeline_capacity_retry_v1"
                ),
                "attempted": True,
                "initial_state_limit": (
                    selection_config.feasibility_lifeline_state_limit
                ),
                "retry_state_limit": (
                    retry_config.feasibility_lifeline_state_limit
                ),
                "trigger": "exact_lifeline_state_capacity_exhausted",
                "initial_explored_state_count": original_exact.get(
                    "explored_state_count"
                ) if isinstance(original_exact, dict) else None,
                "visual_gates_relaxed": False,
                "duration_state_quantization_applied": False,
            }
            partition_search_exact_lifeline_retry_count += 1
            _emit_progress(
                event="direct_member_visual_split_exact_lifeline_retry",
                phase="exact_lifeline_retry_started",
                result=None,
                **exact_lifeline_retry,
                **progress_payload,
            )
            try:
                selection = _select_profile_with_config(
                    profile,
                    retry_config,
                )
                error = None
            except DirectMemberVisualSelectionError as retry_error:
                error = retry_error
            final_exact = (
                error.diagnostics.get("search", {}).get(
                    "exact_lifeline",
                    {},
                )
                if error is not None
                else {}
            )
            retry_result = (
                "feasible"
                if error is None
                else "capacity_exhausted"
                if error.diagnostics.get("failure")
                == "search_capacity_exhausted"
                else "hard_infeasible"
            )
            exact_lifeline_retry.update(
                {
                    "result": retry_result,
                    "final_failure": (
                        error.diagnostics.get("failure")
                        if error is not None
                        else None
                    ),
                    "final_explored_state_count": (
                        final_exact.get("explored_state_count")
                        if isinstance(final_exact, dict)
                        else None
                    ),
                    "final_state_capacity_exhausted": (
                        final_exact.get("state_capacity_exhausted")
                        if isinstance(final_exact, dict)
                        else False
                    ),
                    "final_duration_state_quantization_applied": (
                        final_exact.get("duration_state_quantization_applied")
                        if isinstance(final_exact, dict)
                        else False
                    ),
                }
            )
            if error is None:
                partition_search_exact_lifeline_retry_success_count += 1
            elif error.diagnostics.get("failure") == "search_capacity_exhausted":
                partition_search_exact_lifeline_retry_capacity_exhausted_count += 1
            _emit_progress(
                event="direct_member_visual_split_exact_lifeline_retry",
                phase="exact_lifeline_retry_completed",
                **exact_lifeline_retry,
                **progress_payload,
            )
        if (
            error is not None
            and exact_lifeline_retry is not None
            and exact_lifeline_retry.get("result") == "hard_infeasible"
            and error.diagnostics.get("failure")
            == "validity_quota_path_exhausted"
            and error.diagnostics.get("constraint_infeasibility_proven") is True
            and exact_lifeline_retry.get("final_state_capacity_exhausted") is False
            and exact_lifeline_retry.get(
                "final_duration_state_quantization_applied"
            ) is False
            and exact_lifeline_retry.get("visual_gates_relaxed") is False
            and partition_search_extra_quota_profile_count
            < _DIRECT_MEMBER_MAX_EXTRA_QUOTA_PROFILE_COUNT
        ):
            extra_quota_assignments = (
                _direct_strict_extra_quota_assignment_candidates(
                    span_count=len(profile.spans),
                    source_count=len(eligible_sources),
                    source_order=source_cycle,
                    diagnostics=error.diagnostics,
                )
            )
            if extra_quota_assignments:
                _base_quota, extra_quota_count = divmod(
                    len(profile.spans),
                    len(eligible_sources),
                )
                default_extra_assignment = tuple(
                    int(index) for index in source_cycle[:extra_quota_count]
                )
                assignment_results: list[dict[str, Any]] = []
                feasible_assignments: list[
                    tuple[tuple[int, ...], DirectMemberVisualSelection]
                ] = []
                capacity_errors: list[DirectMemberVisualSelectionError] = []
                proven_errors: list[DirectMemberVisualSelectionError] = []
                unproven_errors: list[DirectMemberVisualSelectionError] = []
                partition_search_extra_quota_profile_count += 1
                _emit_progress(
                    event="direct_member_strict_extra_quota_assignment_search",
                    phase="extra_quota_assignment_search_started",
                    result=None,
                    contract="strict_balanced_extra_quota_assignment_search_v1",
                    default_extra_quota_source_indices=list(
                        default_extra_assignment
                    ),
                    alternative_assignment_count=len(extra_quota_assignments),
                    maximum_qualified_profile_count=(
                        _DIRECT_MEMBER_MAX_EXTRA_QUOTA_PROFILE_COUNT
                    ),
                    maximum_assignment_count=(
                        _DIRECT_MEMBER_MAX_EXTRA_QUOTA_ASSIGNMENT_COUNT
                    ),
                    source_search_order=list(source_cycle),
                    source_search_order_changed=False,
                    counts_differ_by_at_most_one=True,
                    visual_gates_relaxed=False,
                    **progress_payload,
                )
                for assignment_index, assignment in enumerate(
                    extra_quota_assignments,
                    start=1,
                ):
                    partition_search_extra_quota_assignment_evaluation_count += 1
                    alternate_selection: DirectMemberVisualSelection | None = None
                    alternate_error: DirectMemberVisualSelectionError | None = None
                    try:
                        alternate_selection = _select_profile_with_config(
                            profile,
                            retry_config,
                            extra_quota_source_indices=assignment,
                        )
                    except DirectMemberVisualSelectionError as caught:
                        alternate_error = caught
                    alternate_exact = (
                        alternate_error.diagnostics.get("search", {}).get(
                            "exact_lifeline",
                            {},
                        )
                        if alternate_error is not None
                        else {}
                    )
                    if alternate_selection is not None:
                        assignment_result = "feasible"
                        feasible_assignments.append(
                            (assignment, alternate_selection)
                        )
                    elif (
                        alternate_error is not None
                        and alternate_error.diagnostics.get("failure")
                        == "search_capacity_exhausted"
                    ):
                        assignment_result = "capacity_exhausted"
                        capacity_errors.append(alternate_error)
                    elif (
                        alternate_error is not None
                        and _direct_strict_extra_quota_assignment_failure_is_proven(
                            alternate_error.diagnostics
                        )
                    ):
                        assignment_result = "hard_infeasible"
                        proven_errors.append(alternate_error)
                    else:
                        assignment_result = "unproven_failure"
                        if alternate_error is not None:
                            unproven_errors.append(alternate_error)
                    item = {
                        "assignment_index": assignment_index,
                        "extra_quota_source_indices": list(assignment),
                        "result": assignment_result,
                        "failure": (
                            alternate_error.diagnostics.get("failure")
                            if alternate_error is not None
                            else None
                        ),
                        "constraint_infeasibility_proven": (
                            alternate_error.diagnostics.get(
                                "constraint_infeasibility_proven"
                            )
                            if alternate_error is not None
                            else False
                        ),
                        "exact_lifeline_explored_state_count": (
                            alternate_exact.get("explored_state_count")
                            if isinstance(alternate_exact, dict)
                            else None
                        ),
                        "exact_lifeline_state_capacity_exhausted": (
                            alternate_exact.get("state_capacity_exhausted")
                            if isinstance(alternate_exact, dict)
                            else False
                        ),
                        "duration_state_quantization_applied": (
                            alternate_exact.get(
                                "duration_state_quantization_applied"
                            )
                            if isinstance(alternate_exact, dict)
                            else False
                        ),
                    }
                    assignment_results.append(item)
                    _emit_progress(
                        event=(
                            "direct_member_strict_extra_quota_assignment_search"
                        ),
                        phase="extra_quota_assignment_evaluated",
                        **item,
                        **progress_payload,
                    )

                total_assignment_domain_count = math.comb(
                    len(eligible_sources),
                    extra_quota_count,
                )
                assignment_domain_complete = (
                    not capacity_errors
                    and not unproven_errors
                    and len(assignment_results) == len(extra_quota_assignments)
                    and len(extra_quota_assignments) + 1
                    == total_assignment_domain_count
                )
                selected_assignment: tuple[int, ...] | None = None
                if capacity_errors:
                    selection = None
                    error = capacity_errors[0]
                    partition_search_extra_quota_capacity_exhausted_profile_count += 1
                elif unproven_errors:
                    selection = None
                    error = unproven_errors[0]
                elif feasible_assignments and assignment_domain_complete:
                    selected_assignment, selection = min(
                        feasible_assignments,
                        key=lambda item: (
                            _direct_partition_selection_rank(profile, item[1]),
                            item[0],
                        ),
                    )
                    error = None
                    partition_search_extra_quota_feasible_profile_count += 1
                else:
                    selection = None
                    if proven_errors:
                        error = proven_errors[-1]
                    partition_search_extra_quota_exhausted_profile_count += 1
                certificate_payload = {
                    "default_extra_quota_source_indices": list(
                        default_extra_assignment
                    ),
                    "total_assignment_domain_count": (
                        total_assignment_domain_count
                    ),
                    "alternative_assignments": assignment_results,
                }
                strict_extra_quota_assignment_search = {
                    "contract": (
                        "strict_balanced_extra_quota_assignment_search_v1"
                    ),
                    "trigger": (
                        "retry_qualified_visual_split_default_quota_"
                        "hard_infeasible"
                    ),
                    "state_limit": (
                        _DIRECT_MEMBER_VISUAL_SPLIT_EXACT_LIFELINE_RETRY_STATE_LIMIT
                    ),
                    "state_limit_expanded": False,
                    "source_search_order": list(source_cycle),
                    "source_search_order_changed": False,
                    "base_quota": _base_quota,
                    "extra_quota_count": extra_quota_count,
                    "default_extra_quota_source_indices": list(
                        default_extra_assignment
                    ),
                    "alternative_assignment_count": len(
                        extra_quota_assignments
                    ),
                    "total_assignment_domain_count": (
                        total_assignment_domain_count
                    ),
                    "maximum_assignment_count": (
                        _DIRECT_MEMBER_MAX_EXTRA_QUOTA_ASSIGNMENT_COUNT
                    ),
                    "maximum_qualified_profile_count": (
                        _DIRECT_MEMBER_MAX_EXTRA_QUOTA_PROFILE_COUNT
                    ),
                    "assignments_evaluated": assignment_results,
                    "complete_assignment_domain_evaluated": (
                        assignment_domain_complete
                    ),
                    "default_assignment_proven_hard_infeasible": True,
                    "all_alternatives_proven_hard_infeasible": bool(
                        assignment_domain_complete
                        and not feasible_assignments
                        and len(proven_errors) == len(extra_quota_assignments)
                    ),
                    "capacity_exhausted": bool(capacity_errors),
                    "feasible_assignment_count": len(feasible_assignments),
                    "selected_extra_quota_source_indices": (
                        list(selected_assignment)
                        if selected_assignment is not None
                        else None
                    ),
                    "assignment_certificate_sha256": hashlib.sha256(
                        json.dumps(
                            certificate_payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                    "counts_differ_by_at_most_one": True,
                    "visual_gates_relaxed": False,
                }
                if error is not None:
                    error = DirectMemberVisualSelectionError(
                        str(error),
                        {
                            **error.diagnostics,
                            "strict_extra_quota_assignment_search": (
                                strict_extra_quota_assignment_search
                            ),
                        },
                    )
                _emit_progress(
                    event="direct_member_strict_extra_quota_assignment_search",
                    phase="extra_quota_assignment_search_completed",
                    result=(
                        "capacity_exhausted"
                        if capacity_errors
                        else "feasible"
                        if selection is not None
                        else "all_assignments_hard_infeasible"
                    ),
                    complete_assignment_domain_evaluated=(
                        assignment_domain_complete
                    ),
                    alternative_assignment_count=len(
                        extra_quota_assignments
                    ),
                    total_assignment_domain_count=(
                        total_assignment_domain_count
                    ),
                    feasible_assignment_count=len(feasible_assignments),
                    selected_extra_quota_source_indices=(
                        list(selected_assignment)
                        if selected_assignment is not None
                        else None
                    ),
                    assignment_certificate_sha256=(
                        strict_extra_quota_assignment_search[
                            "assignment_certificate_sha256"
                        ]
                    ),
                    visual_gates_relaxed=False,
                    **progress_payload,
                )
        if error is not None:
            failure_count += 1
            failure_is_proven = _direct_partition_failure_is_proven(
                error.diagnostics
            )
            all_failures_proven = all_failures_proven and failure_is_proven
            latest_failure_diagnostics = error.diagnostics
            failure_result = _direct_partition_failure_result(error.diagnostics)
            attempted.append(
                {
                    **profile.to_dict(),
                    "status": failure_result,
                    "hard_violation_count": None,
                    "visual_gates_relaxed": False,
                    **(
                        {"exact_lifeline_retry": exact_lifeline_retry}
                        if exact_lifeline_retry is not None
                        else {}
                    ),
                    **(
                        {
                            "strict_extra_quota_assignment_search": (
                                strict_extra_quota_assignment_search
                            )
                        }
                        if strict_extra_quota_assignment_search is not None
                        else {}
                    ),
                    "diagnostics": _compact_direct_partition_failure(
                        error,
                        span_count=len(profile.spans),
                    ),
                }
            )
            next_proofs = _direct_partition_exact_shift_proofs(
                error.diagnostics,
                span_count=len(profile.spans),
            )
            _emit_progress(
                phase="completed",
                result=failure_result,
                failure=error.diagnostics.get("failure"),
                failed_segment_index=error.diagnostics.get(
                    "failed_at_segment_index"
                ),
                next_proven_frontier_boundary_indices=[
                    proof.boundary_index for proof in next_proofs
                ],
                profile_cap_reached=False,
                **progress_payload,
            )
            return error
        if selection is None:
            raise RuntimeError("direct member selector returned no result")
        feasible.append((profile, selection))
        rank = _direct_partition_selection_rank(profile, selection)
        attempted.append(
            {
                **profile.to_dict(),
                "status": "feasible",
                "hard_violation_count": 0,
                "visual_gates_relaxed": False,
                **(
                    {"exact_lifeline_retry": exact_lifeline_retry}
                    if exact_lifeline_retry is not None
                    else {}
                ),
                **(
                    {
                        "strict_extra_quota_assignment_search": (
                            strict_extra_quota_assignment_search
                        )
                    }
                    if strict_extra_quota_assignment_search is not None
                    else {}
                ),
                "selection_rank": {
                    "warning_boundary_count": rank[0],
                    "optimizer_cost": rank[1],
                    "changed_span_count": rank[2],
                    "musical_boundary_penalty": rank[3],
                },
            }
        )
        _emit_progress(
            phase="completed",
            result="feasible",
            failed_segment_index=None,
            next_proven_frontier_boundary_indices=[],
            profile_cap_reached=False,
            **progress_payload,
        )
        return None

    try:
        base_error = _evaluate(base)
        partition_search_primary_evaluation_count = len(attempted)
        if (
            continuous_official_portrait_fancam_contract
            and not feasible
            and base_error is not None
            and _direct_partition_failure_is_proven(base_error.diagnostics)
            and warm_start_profiles
        ):
            for warm_start_index, warm_profile in enumerate(
                warm_start_profiles,
                start=1,
            ):
                warm_signature = _direct_partition_signature_sha256(
                    warm_profile.unit_counts
                )
                warm_error = _evaluate(warm_profile)
                partition_search_primary_evaluation_count = len(attempted)
                warm_evaluation: dict[str, Any] = {
                    "candidate_index": warm_start_index,
                    "candidate_count": len(warm_start_profiles),
                    "partition_signature_sha256": warm_signature,
                    "strict_result": (
                        "feasible" if warm_error is None else "infeasible"
                    ),
                    "strict_failure": (
                        warm_error.diagnostics.get("failure")
                        if warm_error is not None
                        else None
                    ),
                    "strict_constraint_infeasibility_proven": (
                        _direct_partition_failure_is_proven(
                            warm_error.diagnostics
                        )
                        if warm_error is not None
                        else False
                    ),
                    "terminal_audition_attempted": False,
                    "full_selector_recheck_completed": True,
                    "visual_gates_relaxed": False,
                }
                warm_start_diagnostics["candidate_evaluations"].append(
                    warm_evaluation
                )
                if warm_error is None:
                    warm_evaluation["result"] = "strict_feasible"
                    break
                if not _direct_partition_failure_is_proven(
                    warm_error.diagnostics
                ):
                    warm_evaluation["result"] = (
                        "strict_failure_not_terminal_authorized"
                    )
                    continue

                partition_search_terminal_safe_hold_candidate_count += 1
                partition_search_terminal_safe_hold_evaluation_count += 1
                warm_evaluation["terminal_audition_attempted"] = True
                terminal_payload = {
                    "contract": (
                        "exact_input_bound_partition_warm_start_"
                        "independent_safe_hold_effect_audition_v1"
                    ),
                    "attempted": True,
                    "trigger": "exact_input_binding_and_strict_hard_failure",
                    "candidate_count": len(warm_start_profiles),
                    "evaluation_index": warm_start_index,
                    "evaluation_limit": len(warm_start_profiles),
                    "selected_profile_id": warm_profile.profile_id,
                    "selected_partition_signature_sha256": warm_signature,
                    "selected_strict_failed_segment_index": (
                        warm_error.diagnostics.get("failed_at_segment_index")
                    ),
                    "full_selector_recheck_required": True,
                    "visual_gates_relaxed": False,
                }
                _emit_progress(
                    event="direct_member_input_bound_partition_warm_start",
                    phase="terminal_audition_started",
                    result=None,
                    **terminal_payload,
                )
                terminal_selection: DirectMemberVisualSelection | None = None
                terminal_hold_error: DirectMemberVisualSelectionError | None = None
                try:
                    terminal_selection = _select_profile_with_config(
                        warm_profile,
                        selection_config,
                        allow_safe_hold_search=True,
                    )
                except DirectMemberVisualSelectionError as caught:
                    terminal_hold_error = caught
                terminal_payload["result"] = (
                    "feasible"
                    if terminal_selection is not None
                    else "infeasible"
                )
                terminal_payload["final_failure"] = (
                    terminal_hold_error.diagnostics.get("failure")
                    if terminal_hold_error is not None
                    else None
                )
                attempted_item = attempted[-1]
                if (
                    attempted_item.get("partition_signature_sha256")
                    != warm_signature
                ):
                    raise RuntimeError(
                        "warm-start strict attempt lost its terminal binding"
                    )
                attempted_item["input_bound_warm_start_audition"] = dict(
                    terminal_payload
                )
                warm_evaluation.update(
                    {
                        "terminal_result": terminal_payload["result"],
                        "terminal_failure": terminal_payload["final_failure"],
                        "result": terminal_payload["result"],
                    }
                )
                if terminal_selection is not None:
                    partition_search_terminal_safe_hold_success_count += 1
                    feasible.append((warm_profile, terminal_selection))
                    terminal_rank = _direct_partition_selection_rank(
                        warm_profile,
                        terminal_selection,
                    )
                    attempted_item[
                        "strict_status_before_input_bound_warm_start_audition"
                    ] = attempted_item.get("status")
                    strict_diagnostics = attempted_item.pop("diagnostics", None)
                    if strict_diagnostics is not None:
                        attempted_item[
                            "strict_failure_before_input_bound_warm_start_audition"
                        ] = strict_diagnostics
                    attempted_item["status"] = "feasible"
                    attempted_item["hard_violation_count"] = 0
                    attempted_item["selection_rank"] = {
                        "warning_boundary_count": terminal_rank[0],
                        "optimizer_cost": terminal_rank[1],
                        "changed_span_count": terminal_rank[2],
                        "musical_boundary_penalty": terminal_rank[3],
                    }
                else:
                    assert terminal_hold_error is not None
                    latest_failure_diagnostics = terminal_hold_error.diagnostics
                    all_failures_proven = bool(
                        all_failures_proven
                        and _direct_partition_failure_is_proven(
                            terminal_hold_error.diagnostics
                        )
                    )
                    attempted_item["input_bound_warm_start_failure"] = (
                        _compact_direct_partition_failure(
                            terminal_hold_error,
                            span_count=len(warm_profile.spans),
                        )
                    )
                _emit_progress(
                    event="direct_member_input_bound_partition_warm_start",
                    phase="terminal_audition_completed",
                    **terminal_payload,
                )
                if terminal_selection is not None:
                    break
        if (
            continuous_official_portrait_fancam_contract
            and not feasible
            and base_error is not None
            and _direct_partition_failure_is_proven(base_error.diagnostics)
        ):
            # Bounded best-first search over causally generated children.  A
            # signature enters the queue once, every child is bound to its
            # parent's exact connectivity proof, and a failed child may move
            # only one new proof-authorized boundary before being fully rechecked.
            queued: list[
                tuple[
                    tuple[int, int, int, int],
                    int,
                    _DirectFullFramePartitionProfile,
                ]
            ] = []
            queued_sequence = 0
            seen_signatures = {
                base.unit_counts,
                *(profile.unit_counts for profile in warm_start_profiles),
            }
            evaluated_signatures = set(seen_signatures)
            seen_future_operation_states: set[tuple[Any, ...]] = set()
            counterfactual_secondary_profiles: list[
                _DirectFullFramePartitionProfile
            ] = []
            terminal_safe_hold_candidates: dict[
                tuple[int, ...],
                tuple[
                    _DirectFullFramePartitionProfile,
                    DirectMemberVisualSelectionError,
                ],
            ] = {}

            def _record_terminal_safe_hold_candidate(
                profile: _DirectFullFramePartitionProfile,
                error: DirectMemberVisualSelectionError,
            ) -> None:
                if _direct_partition_failure_is_proven(error.diagnostics):
                    terminal_safe_hold_candidates.setdefault(
                        profile.unit_counts,
                        (profile, error),
                    )

            def _collect_counterfactual_secondary_profile(
                parent: _DirectFullFramePartitionProfile,
                error: DirectMemberVisualSelectionError,
            ) -> None:
                nonlocal partition_search_generated_profile_count
                nonlocal partition_search_signature_deduplication_count
                nonlocal partition_search_counterfactual_qualified_parent_count
                nonlocal partition_search_counterfactual_generated_profile_count
                nonlocal partition_search_counterfactual_promoted_queued_profile_count
                if (
                    len(counterfactual_secondary_profiles)
                    >= _DIRECT_MEMBER_MAX_COUNTERFACTUAL_SECONDARY_SPLIT_PROFILE_COUNT
                    or partition_search_counterfactual_qualified_parent_count
                    >= _DIRECT_MEMBER_MAX_COUNTERFACTUAL_SECONDARY_SPLIT_PROFILE_COUNT
                    or len(parent.visual_frontier_splits)
                    != _DIRECT_MEMBER_COUNTERFACTUAL_SECONDARY_SPLIT_TRIGGER_COUNT
                    or not _direct_severe_counterfactual_retry_is_authorized(
                        error.diagnostics,
                        span_count=len(parent.spans),
                    )
                ):
                    return
                partition_search_counterfactual_qualified_parent_count += 1
                children = _direct_full_frame_visual_frontier_split_child_profiles(
                    timing_draft.segments,
                    parent,
                    error.diagnostics,
                    config=selection_config,
                    allow_counterfactual_secondary_split=True,
                )
                if len(children) > 1:
                    raise RuntimeError(
                        "counterfactual secondary split generated multiple children"
                    )
                for child in children:
                    action = _direct_counterfactual_secondary_signature_action(
                        child.unit_counts,
                        evaluated_signatures=evaluated_signatures,
                        queued_signatures={
                            item[2].unit_counts for item in queued
                        },
                        reserved_signatures={
                            item.unit_counts
                            for item in counterfactual_secondary_profiles
                        },
                        seen_signatures=seen_signatures,
                    )
                    if action == "duplicate":
                        partition_search_signature_deduplication_count += 1
                        continue
                    if action == "promote_queued":
                        queued[:] = [
                            item
                            for item in queued
                            if item[2].unit_counts != child.unit_counts
                        ]
                        partition_search_counterfactual_promoted_queued_profile_count += 1
                    else:
                        seen_signatures.add(child.unit_counts)
                        partition_search_generated_profile_count += 1
                    counterfactual_secondary_profiles.append(child)
                    partition_search_counterfactual_generated_profile_count += 1
                    compact_authorization = (
                        _compact_direct_severe_counterfactual(
                            error.diagnostics
                        )
                    )
                    if compact_authorization is None:
                        raise RuntimeError(
                            "authorized counterfactual secondary split lost "
                            "its compact minimum-severe proof"
                        )
                    partition_search_counterfactual_authorization_proofs.append(
                        {
                            "parent_profile_id": parent.profile_id,
                            "parent_partition_signature_sha256": (
                                _direct_partition_signature_sha256(
                                    parent.unit_counts
                                )
                            ),
                            "child_profile_id": child.profile_id,
                            "child_partition_signature_sha256": (
                                _direct_partition_signature_sha256(
                                    child.unit_counts
                                )
                            ),
                            "authorization": compact_authorization,
                        }
                    )
                    _emit_progress(
                        event=(
                            "direct_member_severe_counterfactual_secondary_split"
                        ),
                        phase="secondary_profile_queued",
                        result=None,
                        contract=(
                            "bounded_diagnostic_severe_counterfactual_"
                            "secondary_partition_v1"
                        ),
                        parent_profile_id=parent.profile_id,
                        profile_id=child.profile_id,
                        maximum_profile_count=(
                            _DIRECT_MEMBER_MAX_COUNTERFACTUAL_SECONDARY_SPLIT_PROFILE_COUNT
                        ),
                        queued_profile_count=len(counterfactual_secondary_profiles),
                        witness_selection_eligible=False,
                        witness_render_authorized=False,
                        full_selector_recheck_required=True,
                        visual_gates_relaxed=False,
                    )

            def _enqueue_children(
                parent: _DirectFullFramePartitionProfile,
                error: DirectMemberVisualSelectionError,
            ) -> int:
                nonlocal queued_sequence
                nonlocal partition_search_generated_profile_count
                nonlocal partition_search_signature_deduplication_count
                nonlocal partition_search_future_state_dominance_count
                nonlocal partition_search_future_operation_state_count
                nonlocal partition_search_shift_depth_pruned_count
                children = _direct_full_frame_proof_directed_repair_profiles(
                    timing_draft.segments,
                    parent,
                    error.diagnostics,
                    config=selection_config,
                )
                if (
                    not children
                    and parent.proof_directed_boundary_motion_count
                    >= _direct_partition_proof_directed_boundary_shift_limit(
                        parent
                    )
                ):
                    partition_search_shift_depth_pruned_count += 1
                    return 0
                future_state = _direct_partition_future_operation_state_key(
                    parent,
                    error.diagnostics,
                    children,
                )
                if future_state in seen_future_operation_states:
                    partition_search_future_state_dominance_count += 1
                    return 0
                seen_future_operation_states.add(future_state)
                partition_search_future_operation_state_count += 1
                enqueued_count = 0
                for child in children:
                    if child.unit_counts in seen_signatures:
                        partition_search_signature_deduplication_count += 1
                        continue
                    seen_signatures.add(child.unit_counts)
                    priority = _direct_partition_queue_priority(child)
                    queued.append((priority, queued_sequence, child))
                    queued_sequence += 1
                    partition_search_generated_profile_count += 1
                    enqueued_count += 1
                return enqueued_count

            _record_terminal_safe_hold_candidate(base, base_error)
            _enqueue_children(base, base_error)
            minimum_feasible_structural_cost: tuple[int, int, int] | None = None
            while queued:
                if _direct_partition_terminal_audition_due(
                    evaluated_profile_count=len(attempted),
                    feasible_profile_count=len(feasible),
                    queued_profile_count=len(queued),
                ):
                    partition_search_strict_terminal_audition_limit_reached = True
                    partition_search_strict_terminal_audition_pruned_profile_count = (
                        len(queued)
                    )
                    _emit_progress(
                        event=(
                            "direct_member_strict_terminal_audition_completed"
                        ),
                        phase="strict_terminal_audition_completed",
                        result="bounded_terminal_cohort_ready",
                        evaluated_terminal_candidate_count=len(
                            terminal_safe_hold_candidates
                        ),
                        maximum_terminal_candidate_count=(
                            _DIRECT_MEMBER_MAX_STRICT_TERMINAL_AUDITION_COUNT
                        ),
                        maximum_proof_progress=max(
                            (
                                _direct_partition_terminal_candidate_progress(
                                    item[0], item[1]
                                )
                                for item in terminal_safe_hold_candidates.values()
                            ),
                            default=-1,
                        ),
                        unevaluated_structural_profile_count=len(queued),
                        visual_gates_relaxed=False,
                    )
                    break
                if minimum_feasible_structural_cost is None:
                    queued.sort(key=lambda item: (item[0], item[1]))
                else:
                    eligible_queue = [
                        item
                        for item in queued
                        if item[0][1:] <= minimum_feasible_structural_cost
                    ]
                    partition_search_post_feasible_cost_pruned_profile_count += (
                        len(queued) - len(eligible_queue)
                    )
                    queued = eligible_queue
                    if not queued:
                        break
                    queued.sort(
                        key=lambda item: (
                            item[0][1:],
                            item[0][0],
                            item[1],
                        )
                    )
                if len(attempted) >= partition_search_evaluation_limit:
                    if (
                        len(partition_search_limit_expansions)
                        >= partition_search_max_limit_expansion_count
                    ):
                        partition_search_cap_reached = True
                        break
                    queued_profiles = tuple(item[2] for item in queued)
                    queued_unique_count = len(
                        {profile.unit_counts for profile in queued_profiles}
                    )
                    if queued_unique_count != len(queued_profiles):
                        raise RuntimeError(
                            "direct partition queue canonical deduplication drifted"
                        )
                    next_limit = len(attempted) + queued_unique_count
                    if next_limit <= partition_search_evaluation_limit:
                        partition_search_cap_reached = True
                        break
                    previous_limit = partition_search_evaluation_limit
                    admitted_limit = min(
                        next_limit,
                        _DIRECT_MEMBER_MAX_PARTITION_PROFILE_UPPER_BOUND,
                    )
                    if admitted_limit <= previous_limit:
                        partition_search_cap_reached = True
                        break
                    epoch = len(partition_search_limit_expansions) + 1
                    expansion = {
                        "epoch": epoch,
                        "max_epochs": (
                            partition_search_max_limit_expansion_count
                        ),
                        "initial_limit": (
                            _DIRECT_MEMBER_MAX_PARTITION_PROFILE_COUNT
                        ),
                        "previous_limit": previous_limit,
                        "evaluated_at_expansion": len(attempted),
                        "queued_unique_profile_count": queued_unique_count,
                        "queue_signature_digest_sha256": (
                            _direct_partition_queue_signature_digest(
                                queued_profiles
                            )
                        ),
                        "minimum_queued_shift_depth": min(
                            profile.proof_directed_boundary_motion_count
                            for profile in queued_profiles
                        ),
                        "maximum_queued_shift_depth": max(
                            profile.proof_directed_boundary_motion_count
                            for profile in queued_profiles
                        ),
                        "exact_frontier_continuation_upper_bound": next_limit,
                        "static_profile_upper_bound": (
                            _DIRECT_MEMBER_MAX_PARTITION_PROFILE_UPPER_BOUND
                        ),
                        "truncated_by_static_upper_bound": (
                            admitted_limit < next_limit
                        ),
                        "new_limit": admitted_limit,
                        "basis": (
                            "already_generated_parent_proof_bound_"
                            "canonical_queue_snapshot"
                        ),
                    }
                    partition_search_limit_expansions.append(expansion)
                    partition_search_evaluation_limit = admitted_limit
                    _emit_progress(
                        event=(
                            "direct_member_partition_search_limit_extended"
                        ),
                        phase="search_limit_extended",
                        result="exact_frontier_continuation_admitted",
                        evaluation_index=len(attempted),
                        evaluation_limit=partition_search_evaluation_limit,
                        **expansion,
                    )
                priority, _sequence, profile = queued.pop(0)
                error = _evaluate(profile)
                evaluated_signatures.add(profile.unit_counts)
                if error is None:
                    # Proof progress orders equal-cost exploration only.  It
                    # must not prune another candidate with the same changed
                    # boundary count, displacement, and musical penalty: that
                    # candidate may win the actual warning/optimizer rank.
                    structural_cost = priority[1:]
                    if (
                        minimum_feasible_structural_cost is None
                        or structural_cost < minimum_feasible_structural_cost
                    ):
                        minimum_feasible_structural_cost = structural_cost
                    continue
                _collect_counterfactual_secondary_profile(profile, error)
                if minimum_feasible_structural_cost is None and (
                    _direct_partition_failure_is_proven(error.diagnostics)
                ):
                    _record_terminal_safe_hold_candidate(profile, error)
                    _enqueue_children(profile, error)
            if (
                not feasible
                and queued
                and len(attempted) >= partition_search_evaluation_limit
                and not partition_search_strict_terminal_audition_limit_reached
            ):
                partition_search_cap_reached = True
            partition_search_primary_evaluation_count = len(attempted)
            if not feasible and counterfactual_secondary_profiles:
                partition_search_evaluation_limit = max(
                    partition_search_evaluation_limit,
                    partition_search_primary_evaluation_count
                    + len(counterfactual_secondary_profiles),
                )
                for profile in counterfactual_secondary_profiles:
                    partition_search_counterfactual_evaluation_count += 1
                    error = _evaluate(profile)
                    evaluated_signatures.add(profile.unit_counts)
                    if error is None:
                        partition_search_counterfactual_feasible_profile_count += 1
                    elif _direct_partition_failure_result(error.diagnostics) == (
                        "hard_infeasible"
                    ):
                        partition_search_counterfactual_hard_infeasible_profile_count += 1
                    elif _direct_partition_failure_result(error.diagnostics) == (
                        "capacity_exhausted"
                    ):
                        partition_search_counterfactual_capacity_exhausted_profile_count += 1
                    else:
                        partition_search_counterfactual_unproven_failure_profile_count += 1
                    if error is not None:
                        _record_terminal_safe_hold_candidate(profile, error)
                _emit_progress(
                    event="direct_member_severe_counterfactual_secondary_split",
                    phase="secondary_search_completed",
                    result=(
                        "feasible"
                        if feasible
                        else "capacity_exhausted"
                        if partition_search_counterfactual_capacity_exhausted_profile_count
                        else "unproven_failure"
                        if partition_search_counterfactual_unproven_failure_profile_count
                        else "hard_infeasible"
                    ),
                    contract=(
                        "bounded_diagnostic_severe_counterfactual_"
                        "secondary_partition_v1"
                    ),
                    primary_evaluated_profile_count=(
                        partition_search_primary_evaluation_count
                    ),
                    qualified_parent_count=(
                        partition_search_counterfactual_qualified_parent_count
                    ),
                    generated_profile_count=(
                        partition_search_counterfactual_generated_profile_count
                    ),
                    evaluated_profile_count=(
                        partition_search_counterfactual_evaluation_count
                    ),
                    feasible_profile_count=(
                        partition_search_counterfactual_feasible_profile_count
                    ),
                    hard_infeasible_profile_count=(
                        partition_search_counterfactual_hard_infeasible_profile_count
                    ),
                    capacity_exhausted_profile_count=(
                        partition_search_counterfactual_capacity_exhausted_profile_count
                    ),
                    unproven_failure_profile_count=(
                        partition_search_counterfactual_unproven_failure_profile_count
                    ),
                    promoted_queued_profile_count=(
                        partition_search_counterfactual_promoted_queued_profile_count
                    ),
                    maximum_profile_count=(
                        _DIRECT_MEMBER_MAX_COUNTERFACTUAL_SECONDARY_SPLIT_PROFILE_COUNT
                    ),
                    witness_selection_eligible=False,
                    witness_render_authorized=False,
                    every_profile_fully_reselected=True,
                    visual_gates_relaxed=False,
                )
            if (
                not feasible
                and (
                    not queued
                    or partition_search_strict_terminal_audition_limit_reached
                )
                and not partition_search_cap_reached
                and terminal_safe_hold_candidates
            ):
                partition_search_terminal_safe_hold_candidate_count = len(
                    terminal_safe_hold_candidates
                )
                terminal_candidates = tuple(
                    sorted(
                        terminal_safe_hold_candidates.values(),
                        key=lambda item: (
                            -_direct_partition_terminal_candidate_progress(
                                item[0], item[1]
                            ),
                            item[0].structural_change_count,
                            item[0].total_unit_displacement,
                            item[0].musical_boundary_penalty,
                            item[0].profile_id,
                        ),
                    )[:_DIRECT_MEMBER_MAX_SAFE_HOLD_PROFILE_COUNT]
                )
                terminal_contract = (
                    "proof_progress_ordered_strict_terminal_audition_"
                    "then_bounded_safe_hold_cohort_v4"
                )
                terminal_trigger = (
                    "bounded_strict_terminal_audition_completed"
                    if partition_search_strict_terminal_audition_limit_reached
                    else "global_structural_candidate_queue_exhausted"
                )
                for terminal_evaluation_index, (
                    terminal_profile,
                    terminal_error,
                ) in enumerate(terminal_candidates, start=1):
                    terminal_signature = _direct_partition_signature_sha256(
                        terminal_profile.unit_counts
                    )
                    attempted_index = next(
                        (
                            index
                            for index, item in enumerate(attempted)
                            if item.get("partition_signature_sha256")
                            == terminal_signature
                        ),
                        None,
                    )
                    if attempted_index is None:
                        raise RuntimeError(
                            "terminal safe-hold candidate has no strict attempt"
                        )
                    terminal_payload = {
                        "contract": terminal_contract,
                        "attempted": True,
                        "trigger": terminal_trigger,
                        "candidate_count": (
                            partition_search_terminal_safe_hold_candidate_count
                        ),
                        "evaluation_index": terminal_evaluation_index,
                        "evaluation_limit": len(terminal_candidates),
                        "maximum_safe_hold_profile_count": (
                            _DIRECT_MEMBER_MAX_SAFE_HOLD_PROFILE_COUNT
                        ),
                        "maximum_strict_terminal_audition_count": (
                            _DIRECT_MEMBER_MAX_STRICT_TERMINAL_AUDITION_COUNT
                        ),
                        "strict_terminal_audition_limit_reached": (
                            partition_search_strict_terminal_audition_limit_reached
                        ),
                        "unevaluated_structural_profile_count": (
                            partition_search_strict_terminal_audition_pruned_profile_count
                        ),
                        "selected_profile_id": terminal_profile.profile_id,
                        "selected_partition_signature_sha256": terminal_signature,
                        "selected_strict_failed_segment_index": (
                            terminal_error.diagnostics.get(
                                "failed_at_segment_index"
                            )
                        ),
                        "selected_proof_progress": (
                            _direct_partition_terminal_candidate_progress(
                                terminal_profile,
                                terminal_error,
                            )
                        ),
                        "selection_policy": (
                            "all_proof_progress_ranked_terminal_profiles_"
                            "then_best_feasible_visual_rank"
                        ),
                        "visual_gates_relaxed": False,
                    }
                    partition_search_terminal_safe_hold_evaluation_count += 1
                    _emit_progress(
                        event="direct_member_last_resort_safe_hold_search",
                        phase="last_resort_safe_hold_started",
                        result=None,
                        **terminal_payload,
                    )
                    terminal_selection: DirectMemberVisualSelection | None = None
                    terminal_hold_error: (
                        DirectMemberVisualSelectionError | None
                    ) = None
                    try:
                        terminal_selection = _select_profile_with_config(
                            terminal_profile,
                            selection_config,
                            allow_safe_hold_search=True,
                        )
                    except DirectMemberVisualSelectionError as caught:
                        terminal_hold_error = caught
                    terminal_payload["result"] = (
                        "feasible"
                        if terminal_selection is not None
                        else "infeasible"
                    )
                    terminal_payload["final_failure"] = (
                        terminal_hold_error.diagnostics.get("failure")
                        if terminal_hold_error is not None
                        else None
                    )
                    attempted_item = attempted[attempted_index]
                    attempted_item["last_resort_safe_hold_search"] = dict(
                        terminal_payload
                    )
                    if terminal_selection is not None:
                        partition_search_terminal_safe_hold_success_count += 1
                        feasible.append((terminal_profile, terminal_selection))
                        terminal_rank = _direct_partition_selection_rank(
                            terminal_profile,
                            terminal_selection,
                        )
                        attempted_item[
                            "strict_status_before_last_resort_safe_hold"
                        ] = attempted_item.get("status")
                        strict_diagnostics = attempted_item.pop(
                            "diagnostics",
                            None,
                        )
                        if strict_diagnostics is not None:
                            attempted_item[
                                "strict_failure_before_last_resort_safe_hold"
                            ] = strict_diagnostics
                        attempted_item["status"] = "feasible"
                        attempted_item["hard_violation_count"] = 0
                        attempted_item["selection_rank"] = {
                            "warning_boundary_count": terminal_rank[0],
                            "optimizer_cost": terminal_rank[1],
                            "changed_span_count": terminal_rank[2],
                            "musical_boundary_penalty": terminal_rank[3],
                        }
                    else:
                        assert terminal_hold_error is not None
                        latest_failure_diagnostics = (
                            terminal_hold_error.diagnostics
                        )
                        all_failures_proven = bool(
                            all_failures_proven
                            and _direct_partition_failure_is_proven(
                                terminal_hold_error.diagnostics
                            )
                        )
                        attempted_item["last_resort_safe_hold_failure"] = (
                            _compact_direct_partition_failure(
                                terminal_hold_error,
                                span_count=len(terminal_profile.spans),
                            )
                        )
                    _emit_progress(
                        event="direct_member_last_resort_safe_hold_search",
                        phase="last_resort_safe_hold_completed",
                        **terminal_payload,
                    )
                if (
                    partition_search_terminal_safe_hold_success_count == 0
                    and partition_search_strict_terminal_audition_limit_reached
                    and queued
                ):
                    # Every authorized hold cohort member failed while bounded
                    # structural siblings remain. Report search capacity, never
                    # claim those unevaluated siblings were hard-infeasible.
                    partition_search_cap_reached = True
        elif not feasible and base_error is not None:
            if _direct_partition_failure_is_proven(base_error.diagnostics):
                bottlenecks = _direct_partition_failure_bottlenecks(
                    base_error.diagnostics,
                    span_count=len(base.spans),
                )
                primary_profiles = _direct_full_frame_shifted_partition_profiles(
                    timing_draft.segments,
                    base,
                    bottlenecks,
                    parent_diagnostics=base_error.diagnostics,
                )
                for profile in primary_profiles:
                    _evaluate(profile)
                partition_search_primary_evaluation_count = len(attempted)
    finally:
        if owned_provider is not None:
            provider_cache_diagnostics = owned_provider.cache_diagnostics()
            owned_provider.close()
        else:
            cache_method = getattr(provider, "cache_diagnostics", None)
            provider_cache_diagnostics = (
                cache_method() if callable(cache_method) else None
            )

    if not feasible:
        latest_diagnostics = (
            dict(latest_failure_diagnostics)
            if latest_failure_diagnostics is not None
            else {}
        )
        all_proven = (
            failure_count > 0
            and all_failures_proven
            and not partition_search_cap_reached
        )
        (
            aggregate_failure,
            aggregate_result,
            attempted_outcome_counts,
        ) = _direct_partition_aggregate_outcome(
            tuple(str(item.get("status")) for item in attempted),
            profile_cap_reached=partition_search_cap_reached,
        )
        aggregate_diagnostics: dict[str, Any] = {
            **latest_diagnostics,
            "failure": aggregate_failure,
            "constraint_infeasibility_proven": all_proven,
            "profile_outcome_counts": attempted_outcome_counts,
            "visual_gates_relaxed": False,
            "partition_search": {
                "policy": (
                    "canonical_signature_deduplicated_proof_directed_"
                    "adaptive_frontier_boundary_merge_dead_end_"
                    "three_span_fixed_cut_window_redistribution_"
                    "visual_micro_split_"
                    "connectivity_precheck_best_first_global_terminal_hold_"
                    "exact_lifeline_capacity_retry_strict_extra_quota_"
                    "assignment_proof_progress_terminal_audition_"
                    "input_bound_warm_start_"
                    "severe_counterfactual_secondary_partition_v15"
                ),
                "initial_profile_count": (
                    _DIRECT_MEMBER_MAX_PARTITION_PROFILE_COUNT
                ),
                "maximum_profile_count": (
                    partition_search_evaluation_limit
                ),
                "static_profile_upper_bound": (
                    _DIRECT_MEMBER_MAX_PARTITION_PROFILE_UPPER_BOUND
                ),
                "absolute_profile_upper_bound": (
                    _DIRECT_MEMBER_MAX_PARTITION_PROFILE_UPPER_BOUND
                    + _DIRECT_MEMBER_MAX_COUNTERFACTUAL_SECONDARY_SPLIT_PROFILE_COUNT
                ),
                "maximum_profile_limit_expansion_count": (
                    partition_search_max_limit_expansion_count
                ),
                "profile_limit_expansions": (
                    partition_search_limit_expansions
                ),
                "maximum_shifted_boundary_count": (
                    partition_search_max_shifted_boundary_count
                ),
                "base_maximum_shifted_boundary_count": (
                    _DIRECT_MEMBER_MAX_SHIFTED_BOUNDARY_COUNT
                ),
                "shift_capacity_contract": (
                    "base_bound_plus_one_compensating_shift_per_"
                    "independently_admitted_merge_v1"
                ),
                "proof_directed_window_redistribution_enabled": True,
                "maximum_proof_directed_boundary_merge_count": (
                    partition_search_max_boundary_merge_count
                ),
                "maximum_visual_frontier_split_count_per_profile": (
                    _DIRECT_MEMBER_MAX_VISUAL_FRONTIER_SPLITS_PER_PROFILE
                ),
                "maximum_counterfactual_visual_frontier_split_count_per_profile": (
                    _DIRECT_MEMBER_MAX_VISUAL_FRONTIER_SPLITS_PER_PROFILE + 1
                ),
                "profile_cap_reached": partition_search_cap_reached,
                "generated_profile_count": (
                    partition_search_generated_profile_count
                ),
                "canonical_signature_deduplication_count": (
                    partition_search_signature_deduplication_count
                ),
                "proof_equivalent_future_state_dominance_count": (
                    partition_search_future_state_dominance_count
                ),
                "future_operation_state_count": (
                    partition_search_future_operation_state_count
                ),
                "shift_depth_pruned_parent_count": (
                    partition_search_shift_depth_pruned_count
                ),
                "queue_priority_contract": (
                    "furthest_proven_frontier_then_structural_cost_v1"
                ),
                "input_bound_partition_warm_start": warm_start_diagnostics,
                "maximum_strict_terminal_audition_count": (
                    _DIRECT_MEMBER_MAX_STRICT_TERMINAL_AUDITION_COUNT
                ),
                "strict_terminal_audition_limit_reached": (
                    partition_search_strict_terminal_audition_limit_reached
                ),
                "strict_terminal_audition_pruned_profile_count": (
                    partition_search_strict_terminal_audition_pruned_profile_count
                ),
                "post_feasible_structural_cost_pruned_profile_count": (
                    partition_search_post_feasible_cost_pruned_profile_count
                ),
                "exact_lifeline_capacity_retry": {
                    "contract": (
                        "visual_frontier_exact_lifeline_capacity_retry_v1"
                    ),
                    "initial_state_limit": (
                        selection_config.feasibility_lifeline_state_limit
                    ),
                    "retry_state_limit": (
                        _DIRECT_MEMBER_VISUAL_SPLIT_EXACT_LIFELINE_RETRY_STATE_LIMIT
                    ),
                    "retry_count": (
                        partition_search_exact_lifeline_retry_count
                    ),
                    "success_count": (
                        partition_search_exact_lifeline_retry_success_count
                    ),
                    "capacity_exhausted_count": (
                        partition_search_exact_lifeline_retry_capacity_exhausted_count
                    ),
                    "visual_frontier_split_required": True,
                    "duration_state_quantization_allowed": False,
                    "visual_gates_relaxed": False,
                },
                "strict_extra_quota_assignment_search": {
                    "contract": (
                        "strict_balanced_extra_quota_assignment_search_v1"
                    ),
                    "maximum_assignment_count": (
                        _DIRECT_MEMBER_MAX_EXTRA_QUOTA_ASSIGNMENT_COUNT
                    ),
                    "maximum_qualified_profile_count": (
                        _DIRECT_MEMBER_MAX_EXTRA_QUOTA_PROFILE_COUNT
                    ),
                    "qualified_profile_count": (
                        partition_search_extra_quota_profile_count
                    ),
                    "assignment_evaluation_count": (
                        partition_search_extra_quota_assignment_evaluation_count
                    ),
                    "feasible_profile_count": (
                        partition_search_extra_quota_feasible_profile_count
                    ),
                    "exhausted_profile_count": (
                        partition_search_extra_quota_exhausted_profile_count
                    ),
                    "capacity_exhausted_profile_count": (
                        partition_search_extra_quota_capacity_exhausted_profile_count
                    ),
                    "source_search_order_changed": False,
                    "counts_differ_by_at_most_one": True,
                    "state_limit_expanded": False,
                    "visual_gates_relaxed": False,
                },
                "last_resort_safe_hold_search": {
                    "contract": (
                        "input_bound_candidate_audition_then_proof_progress_"
                        "ordered_strict_terminal_audition_then_bounded_"
                        "safe_hold_cohort_v5"
                    ),
                    "candidate_count": (
                        partition_search_terminal_safe_hold_candidate_count
                    ),
                    "evaluation_count": (
                        partition_search_terminal_safe_hold_evaluation_count
                    ),
                    "success_count": (
                        partition_search_terminal_safe_hold_success_count
                    ),
                    "maximum_evaluation_count": (
                        _DIRECT_MEMBER_MAX_SAFE_HOLD_PROFILE_COUNT
                    ),
                    "maximum_strict_terminal_audition_count": (
                        _DIRECT_MEMBER_MAX_STRICT_TERMINAL_AUDITION_COUNT
                    ),
                    "strict_terminal_audition_limit_reached": (
                        partition_search_strict_terminal_audition_limit_reached
                    ),
                    "unevaluated_structural_profile_count": (
                        partition_search_strict_terminal_audition_pruned_profile_count
                    ),
                    "global_structural_queue_must_be_exhausted": False,
                    "global_queue_exhaustion_or_bounded_terminal_audition_required": True,
                    "visual_gates_relaxed": False,
                },
                "severe_transition_counterfactual_secondary_partition": {
                    "contract": (
                        "bounded_diagnostic_severe_counterfactual_"
                        "secondary_partition_v1"
                    ),
                    "maximum_qualified_profile_count": (
                        _DIRECT_MEMBER_MAX_COUNTERFACTUAL_SECONDARY_SPLIT_PROFILE_COUNT
                    ),
                    "primary_evaluated_profile_count": (
                        partition_search_primary_evaluation_count
                    ),
                    "actual_evaluated_profile_count": len(attempted),
                    "actual_evaluation_upper_bound": (
                        partition_search_primary_evaluation_count
                        + _DIRECT_MEMBER_MAX_COUNTERFACTUAL_SECONDARY_SPLIT_PROFILE_COUNT
                    ),
                    "qualified_parent_count": (
                        partition_search_counterfactual_qualified_parent_count
                    ),
                    "generated_profile_count": (
                        partition_search_counterfactual_generated_profile_count
                    ),
                    "evaluation_count": (
                        partition_search_counterfactual_evaluation_count
                    ),
                    "feasible_profile_count": (
                        partition_search_counterfactual_feasible_profile_count
                    ),
                    "hard_infeasible_profile_count": (
                        partition_search_counterfactual_hard_infeasible_profile_count
                    ),
                    "capacity_exhausted_profile_count": (
                        partition_search_counterfactual_capacity_exhausted_profile_count
                    ),
                    "unproven_failure_profile_count": (
                        partition_search_counterfactual_unproven_failure_profile_count
                    ),
                    "promoted_queued_profile_count": (
                        partition_search_counterfactual_promoted_queued_profile_count
                    ),
                    "authorization_proofs": (
                        partition_search_counterfactual_authorization_proofs
                    ),
                    "maximum_added_split_count_per_profile": 1,
                    "existing_timing_boundary_required": True,
                    "proposal_evidence_required": True,
                    "sample_bracket_required_for_sample_evidence": True,
                    "confirmed_native_cut_snap_allowed": True,
                    "full_selector_recheck_required": True,
                    "balanced_source_quota_hard_gate_rechecked": True,
                    "strict_extra_quota_domain_search_reused": True,
                    "witness_selection_eligible": False,
                    "witness_render_authorized": False,
                    "duration_state_quantization_allowed": False,
                    "visual_gates_relaxed": False,
                },
                "queue_order": (
                    "furthest_parent_proof_progress_then_structural_change_"
                    "count_then_total_unit_displacement_then_musical_"
                    "penalty_then_generation_order"
                ),
                "profiles_attempted": attempted,
                "descriptor_cache": provider_cache_diagnostics,
                "render_started": False,
            },
        }
        _emit_progress(
            event="direct_member_partition_search_completed",
            phase="search_completed",
            result=(
                aggregate_result
            ),
            evaluated_profile_count=len(attempted),
            evaluation_limit=partition_search_evaluation_limit,
            initial_evaluation_limit=(
                _DIRECT_MEMBER_MAX_PARTITION_PROFILE_COUNT
            ),
            profile_limit_expansion_count=len(
                partition_search_limit_expansions
            ),
            generated_profile_count=partition_search_generated_profile_count,
            exact_lifeline_capacity_retry_count=(
                partition_search_exact_lifeline_retry_count
            ),
            exact_lifeline_capacity_retry_success_count=(
                partition_search_exact_lifeline_retry_success_count
            ),
            exact_lifeline_capacity_retry_capacity_exhausted_count=(
                partition_search_exact_lifeline_retry_capacity_exhausted_count
            ),
            strict_extra_quota_assignment_qualified_profile_count=(
                partition_search_extra_quota_profile_count
            ),
            strict_extra_quota_assignment_evaluation_count=(
                partition_search_extra_quota_assignment_evaluation_count
            ),
            strict_extra_quota_assignment_feasible_profile_count=(
                partition_search_extra_quota_feasible_profile_count
            ),
            strict_extra_quota_assignment_exhausted_profile_count=(
                partition_search_extra_quota_exhausted_profile_count
            ),
            strict_extra_quota_assignment_capacity_exhausted_profile_count=(
                partition_search_extra_quota_capacity_exhausted_profile_count
            ),
            counterfactual_secondary_generated_profile_count=(
                partition_search_counterfactual_generated_profile_count
            ),
            counterfactual_secondary_evaluation_count=(
                partition_search_counterfactual_evaluation_count
            ),
            counterfactual_secondary_feasible_profile_count=(
                partition_search_counterfactual_feasible_profile_count
            ),
            counterfactual_secondary_hard_infeasible_profile_count=(
                partition_search_counterfactual_hard_infeasible_profile_count
            ),
            counterfactual_secondary_capacity_exhausted_profile_count=(
                partition_search_counterfactual_capacity_exhausted_profile_count
            ),
            counterfactual_secondary_unproven_failure_profile_count=(
                partition_search_counterfactual_unproven_failure_profile_count
            ),
            profile_cap_reached=partition_search_cap_reached,
            selected_profile_id=None,
        )
        raise DirectMemberVisualSelectionError(
            (
                "all bounded direct-member musical partitions were "
                "proven hard-infeasible"
                if all_proven
                else "direct-member partition search ended without a "
                "proved feasible selection"
            ),
            aggregate_diagnostics,
        )

    selected_profile, selected_selection = min(
        feasible,
        key=lambda item: _direct_partition_selection_rank(item[0], item[1]),
    )
    warm_start_diagnostics["persistence"] = (
        _persist_direct_member_partition_warm_start_profile(
            feature_cache_directory,
            base,
            selected_profile,
            source_count=len(eligible_sources),
        )
    )
    partition_search = {
        "policy": (
            "canonical_signature_deduplicated_proof_directed_"
            "adaptive_frontier_boundary_merge_dead_end_"
            "three_span_fixed_cut_window_redistribution_"
            "visual_micro_split_best_first_"
            "connectivity_precheck_global_terminal_hold_"
            "exact_lifeline_capacity_retry_strict_extra_quota_"
            "assignment_proof_progress_terminal_audition_"
            "input_bound_warm_start_"
            "severe_counterfactual_secondary_partition_v15"
        ),
        "initial_profile_count": _DIRECT_MEMBER_MAX_PARTITION_PROFILE_COUNT,
        "maximum_profile_count": partition_search_evaluation_limit,
        "static_profile_upper_bound": (
            _DIRECT_MEMBER_MAX_PARTITION_PROFILE_UPPER_BOUND
        ),
        "absolute_profile_upper_bound": (
            _DIRECT_MEMBER_MAX_PARTITION_PROFILE_UPPER_BOUND
            + _DIRECT_MEMBER_MAX_COUNTERFACTUAL_SECONDARY_SPLIT_PROFILE_COUNT
        ),
        "maximum_profile_limit_expansion_count": (
            partition_search_max_limit_expansion_count
        ),
        "profile_limit_expansions": partition_search_limit_expansions,
        "maximum_shifted_boundary_count": (
            partition_search_max_shifted_boundary_count
        ),
        "base_maximum_shifted_boundary_count": (
            _DIRECT_MEMBER_MAX_SHIFTED_BOUNDARY_COUNT
        ),
        "shift_capacity_contract": (
            "base_bound_plus_one_compensating_shift_per_"
            "independently_admitted_merge_v1"
        ),
        "proof_directed_window_redistribution_enabled": True,
        "maximum_proof_directed_boundary_merge_count": (
            partition_search_max_boundary_merge_count
        ),
        "maximum_visual_frontier_split_count_per_profile": (
            _DIRECT_MEMBER_MAX_VISUAL_FRONTIER_SPLITS_PER_PROFILE
        ),
        "maximum_counterfactual_visual_frontier_split_count_per_profile": (
            _DIRECT_MEMBER_MAX_VISUAL_FRONTIER_SPLITS_PER_PROFILE + 1
        ),
        "profile_cap_reached": partition_search_cap_reached,
        "generated_profile_count": partition_search_generated_profile_count,
        "canonical_signature_deduplication_count": (
            partition_search_signature_deduplication_count
        ),
        "proof_equivalent_future_state_dominance_count": (
            partition_search_future_state_dominance_count
        ),
        "future_operation_state_count": (
            partition_search_future_operation_state_count
        ),
        "shift_depth_pruned_parent_count": (
            partition_search_shift_depth_pruned_count
        ),
        "queue_priority_contract": (
            "furthest_proven_frontier_then_structural_cost_v1"
        ),
        "input_bound_partition_warm_start": warm_start_diagnostics,
        "maximum_strict_terminal_audition_count": (
            _DIRECT_MEMBER_MAX_STRICT_TERMINAL_AUDITION_COUNT
        ),
        "strict_terminal_audition_limit_reached": (
            partition_search_strict_terminal_audition_limit_reached
        ),
        "strict_terminal_audition_pruned_profile_count": (
            partition_search_strict_terminal_audition_pruned_profile_count
        ),
        "post_feasible_structural_cost_pruned_profile_count": (
            partition_search_post_feasible_cost_pruned_profile_count
        ),
        "exact_lifeline_capacity_retry": {
            "contract": "visual_frontier_exact_lifeline_capacity_retry_v1",
            "initial_state_limit": (
                selection_config.feasibility_lifeline_state_limit
            ),
            "retry_state_limit": (
                _DIRECT_MEMBER_VISUAL_SPLIT_EXACT_LIFELINE_RETRY_STATE_LIMIT
            ),
            "retry_count": partition_search_exact_lifeline_retry_count,
            "success_count": (
                partition_search_exact_lifeline_retry_success_count
            ),
            "capacity_exhausted_count": (
                partition_search_exact_lifeline_retry_capacity_exhausted_count
            ),
            "visual_frontier_split_required": True,
            "duration_state_quantization_allowed": False,
            "visual_gates_relaxed": False,
        },
        "strict_extra_quota_assignment_search": {
            "contract": "strict_balanced_extra_quota_assignment_search_v1",
            "maximum_assignment_count": (
                _DIRECT_MEMBER_MAX_EXTRA_QUOTA_ASSIGNMENT_COUNT
            ),
            "maximum_qualified_profile_count": (
                _DIRECT_MEMBER_MAX_EXTRA_QUOTA_PROFILE_COUNT
            ),
            "qualified_profile_count": (
                partition_search_extra_quota_profile_count
            ),
            "assignment_evaluation_count": (
                partition_search_extra_quota_assignment_evaluation_count
            ),
            "feasible_profile_count": (
                partition_search_extra_quota_feasible_profile_count
            ),
            "exhausted_profile_count": (
                partition_search_extra_quota_exhausted_profile_count
            ),
            "capacity_exhausted_profile_count": (
                partition_search_extra_quota_capacity_exhausted_profile_count
            ),
            "source_search_order_changed": False,
            "counts_differ_by_at_most_one": True,
            "state_limit_expanded": False,
            "visual_gates_relaxed": False,
        },
        "last_resort_safe_hold_search": {
            "contract": (
                "input_bound_candidate_audition_then_proof_progress_"
                "ordered_strict_terminal_audition_then_bounded_safe_hold_"
                "cohort_v5"
            ),
            "candidate_count": (
                partition_search_terminal_safe_hold_candidate_count
            ),
            "evaluation_count": (
                partition_search_terminal_safe_hold_evaluation_count
            ),
            "success_count": partition_search_terminal_safe_hold_success_count,
            "maximum_evaluation_count": (
                _DIRECT_MEMBER_MAX_SAFE_HOLD_PROFILE_COUNT
            ),
            "maximum_strict_terminal_audition_count": (
                _DIRECT_MEMBER_MAX_STRICT_TERMINAL_AUDITION_COUNT
            ),
            "strict_terminal_audition_limit_reached": (
                partition_search_strict_terminal_audition_limit_reached
            ),
            "unevaluated_structural_profile_count": (
                partition_search_strict_terminal_audition_pruned_profile_count
            ),
            "global_structural_queue_must_be_exhausted": False,
            "global_queue_exhaustion_or_bounded_terminal_audition_required": True,
            "visual_gates_relaxed": False,
        },
        "severe_transition_counterfactual_secondary_partition": {
            "contract": (
                "bounded_diagnostic_severe_counterfactual_"
                "secondary_partition_v1"
            ),
            "maximum_qualified_profile_count": (
                _DIRECT_MEMBER_MAX_COUNTERFACTUAL_SECONDARY_SPLIT_PROFILE_COUNT
            ),
            "primary_evaluated_profile_count": (
                partition_search_primary_evaluation_count
            ),
            "actual_evaluated_profile_count": len(attempted),
            "actual_evaluation_upper_bound": (
                partition_search_primary_evaluation_count
                + _DIRECT_MEMBER_MAX_COUNTERFACTUAL_SECONDARY_SPLIT_PROFILE_COUNT
            ),
            "qualified_parent_count": (
                partition_search_counterfactual_qualified_parent_count
            ),
            "generated_profile_count": (
                partition_search_counterfactual_generated_profile_count
            ),
            "evaluation_count": (
                partition_search_counterfactual_evaluation_count
            ),
            "feasible_profile_count": (
                partition_search_counterfactual_feasible_profile_count
            ),
            "hard_infeasible_profile_count": (
                partition_search_counterfactual_hard_infeasible_profile_count
            ),
            "capacity_exhausted_profile_count": (
                partition_search_counterfactual_capacity_exhausted_profile_count
            ),
            "unproven_failure_profile_count": (
                partition_search_counterfactual_unproven_failure_profile_count
            ),
            "promoted_queued_profile_count": (
                partition_search_counterfactual_promoted_queued_profile_count
            ),
            "authorization_proofs": (
                partition_search_counterfactual_authorization_proofs
            ),
            "maximum_added_split_count_per_profile": 1,
            "existing_timing_boundary_required": True,
            "proposal_evidence_required": True,
            "sample_bracket_required_for_sample_evidence": True,
            "confirmed_native_cut_snap_allowed": True,
            "full_selector_recheck_required": True,
            "balanced_source_quota_hard_gate_rechecked": True,
            "strict_extra_quota_domain_search_reused": True,
            "witness_selection_eligible": False,
            "witness_render_authorized": False,
            "duration_state_quantization_allowed": False,
            "visual_gates_relaxed": False,
        },
        "queue_order": (
            "furthest_parent_proof_progress_then_structural_change_count_"
            "then_total_unit_displacement_then_musical_penalty_then_"
            "generation_order"
        ),
        "profiles_attempted": attempted,
        "selected_profile_id": selected_profile.profile_id,
        "selected_partition_unit_counts": list(selected_profile.unit_counts),
        "selected_window_redistributions": [
            redistribution.to_dict()
            for redistribution in selected_profile.window_redistributions
        ],
        "highlight_dense_cadence": {
            "contract": "section_role_bound_existing_musical_cut_density_v1",
            "highlight_intervals": [
                [start, end]
                for start, end in normalized_highlight_intervals
            ],
            "undensified_base_span_count": undensified_base_span_count,
            "initial_highlight_dense_span_count": len(base.spans),
            "selected_span_count": len(selected_profile.spans),
            "added_initial_span_count": (
                len(base.spans) - undensified_base_span_count
            ),
            "target_global_switches_per_minute": (
                _DIRECT_MEMBER_HIGHLIGHT_DENSE_SWITCHES_PER_MINUTE
            ),
            "highlight_target_run_seconds": (
                _DIRECT_MEMBER_HIGHLIGHT_TARGET_RUN_SECONDS
            ),
            "priority_selected_boundary_indices": list(
                _direct_member_highlight_priority_boundary_indices(
                    selected_profile.spans,
                    normalized_highlight_intervals,
                )
            ),
            "maximum_compensated_highlight_hold_count": (
                _DIRECT_MEMBER_MAXIMUM_COMPENSATED_HIGHLIGHT_HOLD_COUNT
            ),
            "minimum_highlight_switches_per_minute_after_holds": (
                _DIRECT_MEMBER_HIGHLIGHT_MINIMUM_SWITCHES_PER_MINUTE
            ),
            "priority_hold_requires_remaining_density_gate": True,
            "visual_selector_recheck_required_for_every_added_cut": True,
        },
        "fallback_used": bool(
            selected_profile.boundary_shifts
            or selected_profile.window_redistributions
            or selected_profile.visual_frontier_splits
            or selected_profile.boundary_merges
            or selected_profile.warm_start_binding_sha256 is not None
        ),
        "visual_gates_relaxed": False,
        "continuous_official_portrait_fancam_contract": (
            continuous_official_portrait_fancam_contract
        ),
        "descriptor_cache": provider_cache_diagnostics,
        "render_started_after_profile_selection": True,
        "partition_queue_priority_contract": (
            "furthest_proven_frontier_then_structural_cost_v1"
        ),
        "post_feasible_structural_cost_pruned_profile_count": (
            partition_search_post_feasible_cost_pruned_profile_count
        ),
        "safe_hold_search_deferred_until_bounded_terminal_audition": True,
        "safe_hold_search_deferred_while_proof_directed_repair_remains": (
            not partition_search_strict_terminal_audition_limit_reached
        ),
    }
    _emit_progress(
        event="direct_member_partition_search_completed",
        phase="search_completed",
        result="feasible",
        evaluated_profile_count=len(attempted),
        evaluation_limit=partition_search_evaluation_limit,
        initial_evaluation_limit=_DIRECT_MEMBER_MAX_PARTITION_PROFILE_COUNT,
        profile_limit_expansion_count=len(
            partition_search_limit_expansions
        ),
        generated_profile_count=partition_search_generated_profile_count,
        exact_lifeline_capacity_retry_count=(
            partition_search_exact_lifeline_retry_count
        ),
        exact_lifeline_capacity_retry_success_count=(
            partition_search_exact_lifeline_retry_success_count
        ),
        exact_lifeline_capacity_retry_capacity_exhausted_count=(
            partition_search_exact_lifeline_retry_capacity_exhausted_count
        ),
        strict_extra_quota_assignment_qualified_profile_count=(
            partition_search_extra_quota_profile_count
        ),
        strict_extra_quota_assignment_evaluation_count=(
            partition_search_extra_quota_assignment_evaluation_count
        ),
        strict_extra_quota_assignment_feasible_profile_count=(
            partition_search_extra_quota_feasible_profile_count
        ),
        strict_extra_quota_assignment_exhausted_profile_count=(
            partition_search_extra_quota_exhausted_profile_count
        ),
        strict_extra_quota_assignment_capacity_exhausted_profile_count=(
            partition_search_extra_quota_capacity_exhausted_profile_count
        ),
        counterfactual_secondary_generated_profile_count=(
            partition_search_counterfactual_generated_profile_count
        ),
        counterfactual_secondary_evaluation_count=(
            partition_search_counterfactual_evaluation_count
        ),
        counterfactual_secondary_feasible_profile_count=(
            partition_search_counterfactual_feasible_profile_count
        ),
        counterfactual_secondary_hard_infeasible_profile_count=(
            partition_search_counterfactual_hard_infeasible_profile_count
        ),
        counterfactual_secondary_capacity_exhausted_profile_count=(
            partition_search_counterfactual_capacity_exhausted_profile_count
        ),
        counterfactual_secondary_unproven_failure_profile_count=(
            partition_search_counterfactual_unproven_failure_profile_count
        ),
        profile_cap_reached=partition_search_cap_reached,
        selected_profile_id=selected_profile.profile_id,
        selected_partition_signature_sha256=(
            _direct_partition_signature_sha256(selected_profile.unit_counts)
        ),
        selected_shifted_boundaries=[
            {
                "boundary_index": shift.boundary_index,
                "shift_units": shift.shift_units,
            }
            for shift in selected_profile.boundary_shifts
        ],
        selected_window_redistributions=[
            {
                "window_start_span_index": (
                    redistribution.window_start_span_index
                ),
                "proof_frontier_boundary_index": (
                    redistribution.proof_frontier_boundary_index
                ),
                "original_unit_counts": list(
                    redistribution.original_unit_counts
                ),
                "replacement_unit_counts": list(
                    redistribution.replacement_unit_counts
                ),
                "boundary_displacement_units": (
                    redistribution.boundary_displacement_units
                ),
            }
            for redistribution in selected_profile.window_redistributions
        ],
        selected_visual_frontier_splits=[
            {
                "parent_span_index": split.parent_span_index,
                "split_unit_offset": split.split_unit_offset,
                "canonical_master_seconds": split.canonical_master_seconds,
                "evidence_contract": split.evidence_contract,
                "evidence_kind": split.evidence_kind,
            }
            for split in selected_profile.visual_frontier_splits
        ],
        selected_proof_directed_boundary_merges=[
            {
                "boundary_index": merge.boundary_index,
                "merged_unit_count": merge.merged_unit_count,
                "canonical_master_seconds": merge.canonical_master_seconds,
            }
            for merge in selected_profile.boundary_merges
        ],
    )
    return (selected_profile, selected_selection, partition_search)


def _materialize_direct_member_local_cut_times(
    spans: Sequence[_DirectFullFrameSpan],
    selection: DirectMemberVisualSelection,
) -> tuple[_DirectFullFrameSpan, ...]:
    """Apply the selector's jointly audited near-beat cut states to render time."""

    items = tuple(spans)
    boundaries = tuple(selection.selected_boundary_master_seconds)
    if len(boundaries) != len(items) - 1:
        raise RuntimeError(
            "direct member local cut times do not match the selected partition"
        )
    edges = (
        items[0].canonical_master_start,
        *boundaries,
        items[-1].canonical_master_end,
    )
    if any(
        not math.isfinite(value) for value in edges
    ) or any(left >= right for left, right in zip(edges, edges[1:])):
        raise RuntimeError("direct member local cut timeline is not increasing")
    maximum_offset = (
        selection.config.boundary_search_radius_frames
        * selection.config.sample_frame_step_seconds
    )
    for index, selected in enumerate(boundaries):
        canonical = items[index].canonical_master_end
        if abs(selected - canonical) > maximum_offset + 1e-7:
            raise RuntimeError(
                "direct member local cut exceeded its configured beat window"
            )
    return tuple(
        replace(
            span,
            canonical_master_start=edges[index],
            canonical_master_end=edges[index + 1],
        )
        for index, span in enumerate(items)
    )


def _official_target_fancam_direct_full_frame_segments(
    *,
    timing_draft: GroupDraftTimeline,
    eligible_indices: Sequence[int],
    eligible_sources: Sequence[Path],
    common_master_start: float,
    source_timewarps: Sequence[TimeWarpMap | None] | None,
    scene_cut_times: Sequence[Sequence[float]],
    eligible_source_aspect_ratios: Sequence[float] | None = None,
    eligible_source_frame_sizes: Sequence[tuple[int, int]] | None = None,
    output_frame_size: tuple[int, int] = (1080, 1920),
    feature_cache_directory: Path | None = None,
    descriptor_provider: DescriptorProvider | None = None,
    eligible_source_durations_seconds: Sequence[float] | None = None,
    source_metadata_contract: str | None = None,
    highlight_intervals: Sequence[tuple[float, float]] = (),
    member_pose_dissolve_enabled: bool = True,
    output_fps: float = 29.97,
    partition_progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[
    tuple[RenderSegment, ...],
    tuple[RenderTransition, ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
    DirectMemberVisualSelection,
]:
    """Build a visually selected, all-source portrait render."""

    eligible = tuple(int(index) for index in eligible_indices)
    if len(eligible) < MINIMUM_AUTOMATIC_MEMBER_SOURCE_COUNT:
        raise RuntimeError(
            "direct full-frame member render needs at least "
            f"{MINIMUM_AUTOMATIC_MEMBER_SOURCE_COUNT} sources"
        )
    if len(eligible_sources) != len(eligible):
        raise ValueError("direct eligible_sources must match eligible indices")
    source_aspect_ratios = (
        tuple(9.0 / 16.0 for _ in eligible)
        if eligible_source_aspect_ratios is None
        else tuple(float(value) for value in eligible_source_aspect_ratios)
    )
    if len(source_aspect_ratios) != len(eligible):
        raise ValueError("direct source aspect ratios must match eligible indices")
    if eligible_source_frame_sizes is not None and (
        len(eligible_source_frame_sizes) != len(eligible)
    ):
        raise ValueError("direct source frame sizes must match eligible indices")
    portrait_fill_crops = tuple(
        _lossless_member_portrait_fill_crop(value)
        for value in source_aspect_ratios
    )
    if len(timing_draft.audio_offsets) != len(eligible):
        raise ValueError("direct timing offsets must match eligible sources")
    if source_timewarps is not None and max(eligible) >= len(source_timewarps):
        raise ValueError("direct source_timewarps do not cover eligible sources")
    if len(scene_cut_times) != len(eligible):
        raise ValueError("direct scene_cut_times must match eligible sources")

    preferred_cycle = tuple(
        dict.fromkeys(item.source_index for item in timing_draft.segments)
    )
    if any(index >= len(eligible) for index in preferred_cycle):
        raise RuntimeError("timing draft references an unknown eligible source")
    # Timing supplies the synchronized musical boundaries and its first-use
    # order is a useful preference, but it is not allowed to silently starve an
    # otherwise eligible official fancam. Append any missing sources so the
    # direct path's explicit all-source coverage guarantee remains true.
    source_cycle = preferred_cycle + tuple(
        index for index in range(len(eligible)) if index not in preferred_cycle
    )
    continuous_official_portrait_fancam_contract = bool(
        source_metadata_contract
        == OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT
    )
    partition_profile, visual_selection, partition_search = (
        _select_direct_member_partition_profile(
            timing_draft=timing_draft,
            eligible_indices=eligible,
            eligible_sources=eligible_sources,
            source_timewarps=source_timewarps,
            scene_cut_times=scene_cut_times,
            source_cycle=source_cycle,
            highlight_intervals=highlight_intervals,
            continuous_official_portrait_fancam_contract=(
                continuous_official_portrait_fancam_contract
            ),
            cinematic_bridge_enabled=member_pose_dissolve_enabled,
            eligible_source_frame_sizes=eligible_source_frame_sizes,
            output_frame_size=output_frame_size,
            feature_cache_directory=feature_cache_directory,
            descriptor_provider=descriptor_provider,
            progress_callback=partition_progress_callback,
        )
    )
    spans = _materialize_direct_member_local_cut_times(
        partition_profile.spans,
        visual_selection,
    )

    resolved: list[RenderSegment] = []
    details: list[dict[str, Any]] = []
    for segment_index, span in enumerate(spans):
        eligible_source_index = visual_selection.source_indices[segment_index]
        request_source_index = eligible[eligible_source_index]
        timewarp = (
            source_timewarps[request_source_index]
            if source_timewarps is not None
            else None
        )
        segment = aligned_render_segment(
            source_index=request_source_index,
            reference_start=span.canonical_master_start,
            common_master_start=common_master_start,
            source_offset=timing_draft.audio_offsets[eligible_source_index],
            duration=span.duration_seconds,
            timewarp=timewarp,
        )
        resolved.append(segment)
        details.append(
            {
                "segment_index": segment_index,
                "policy": OFFICIAL_TARGET_FANCAM_DIRECT_FULL_FRAME_POLICY,
                "eligible_source_index": eligible_source_index,
                "request_source_index": request_source_index,
                "canonical_master_start": span.canonical_master_start,
                "canonical_master_end": span.canonical_master_end,
                "output_start_seconds": segment.master_start,
                "output_end_seconds": segment.master_start + segment.duration,
                "source_local_start": segment.source_start,
                "source_local_end": segment.source_end,
                "source_duration_seconds": segment.effective_source_duration,
                "source_playback_rate": segment.source_playback_rate,
                "source_clock_basis": (
                    "local_monotonic_timewarp"
                    if segment.source_duration is not None
                    else "global_scalar_offset"
                ),
                "crop_applied": portrait_fill_crops[eligible_source_index]
                is not None,
                "portrait_fill_contract": (
                    AUTOMATIC_MEMBER_PORTRAIT_FILL_CONTRACT
                ),
                "source_aspect_ratio": source_aspect_ratios[
                    eligible_source_index
                ],
                "portrait_fill_crop": (
                    {
                        "x": portrait_fill_crops[eligible_source_index].x,
                        "y": portrait_fill_crops[eligible_source_index].y,
                        "width": portrait_fill_crops[eligible_source_index].width,
                        "height": portrait_fill_crops[eligible_source_index].height,
                    }
                    if portrait_fill_crops[eligible_source_index] is not None
                    else None
                ),
                "timing_segment_count": span.timing_segment_count,
                "estimated_beat_count": span.estimated_beat_count,
                "start_boundary_reasons": list(span.start_boundary_reasons),
                "end_boundary_reasons": list(span.end_boundary_reasons),
            }
        )

    boundary_diagnostics = tuple(visual_selection.boundaries)
    if len(boundary_diagnostics) != len(resolved) - 1:
        raise RuntimeError(
            "direct member visual selection boundary count disagrees with render"
        )
    hold_boundary_indices: set[int] = set()
    transitions_list: list[RenderTransition] = []
    for boundary_index, (left, right, diagnostic) in enumerate(
        zip(resolved[:-1], resolved[1:], boundary_diagnostics, strict=True)
    ):
        cut_applied = getattr(diagnostic, "cut_applied", True)
        if not isinstance(cut_applied, bool):
            raise RuntimeError("direct member cut-applied diagnostic is invalid")
        same_source = left.source_index == right.source_index
        if same_source != (not cut_applied):
            raise RuntimeError(
                "direct member same-source continuation disagrees with "
                "the audited cut decision"
            )
        if same_source:
            hold_boundary_indices.add(boundary_index)
            transitions_list.append(
                RenderTransition(
                    tier="D",
                    reason="same_source_hold_no_visible_switch",
                )
            )
        else:
            transitions_list.append(
                RenderTransition(
                    reason=OFFICIAL_TARGET_FANCAM_DIRECT_FULL_FRAME_POLICY
                )
            )
    transitions = tuple(transitions_list)
    counts = {source: 0 for source in eligible}
    for segment in resolved:
        counts[segment.source_index] += 1
    used = {source for source, count in counts.items() if count > 0}
    if (
        len(used) < MINIMUM_AUTOMATIC_MEMBER_SOURCE_COUNT
        or used != set(eligible)
    ):
        raise RuntimeError("direct full-frame render failed all-source coverage")
    source_usage_policy = getattr(
        visual_selection,
        "usage_policy",
        "exact_balanced_quota_v1",
    )
    if source_usage_policy == "exact_balanced_quota_v1":
        if max(counts.values()) - min(counts.values()) > 1:
            raise RuntimeError(
                "direct full-frame render failed balanced visual selection"
            )
    elif source_usage_policy in _DIRECT_MEMBER_ROLE_AWARE_SOURCE_USAGE_POLICIES:
        rendered_counts = tuple(counts[source] for source in eligible)
        if any(
            actual < minimum or actual > maximum
            for actual, minimum, maximum in zip(
                rendered_counts,
                visual_selection.source_count_minimums,
                visual_selection.source_count_maximums,
            )
        ):
            raise RuntimeError(
                "direct full-frame render missed heterogeneous source-count bounds"
            )
    else:
        raise RuntimeError("direct full-frame render has unknown usage policy")

    # Recompute duration exposure from the exact output segments.  Do not rely
    # only on the selector's serialized diagnostics: this is the final render
    # boundary and must independently enforce its quota-adjusted duration gate.
    duration_totals = [0.0] * len(eligible)
    for segment_index, segment in enumerate(resolved):
        eligible_source_index = visual_selection.source_indices[segment_index]
        if not 0 <= eligible_source_index < len(eligible):
            raise RuntimeError(
                "direct full-frame render duration balance references an unknown source"
            )
        duration_totals[eligible_source_index] += segment.duration
    total_duration = sum(segment.duration for segment in resolved)
    if source_usage_policy in _DIRECT_MEMBER_ROLE_AWARE_SOURCE_USAGE_POLICIES:
        duration_targets = tuple(
            visual_selection.source_target_durations_seconds
        )
        duration_minimums = tuple(
            visual_selection.source_duration_minimums_seconds
        )
        duration_maximums = tuple(
            visual_selection.source_duration_maximums_seconds
        )
        if any(
            actual < minimum - 1e-9 or actual > maximum + 1e-9
            for actual, minimum, maximum in zip(
                duration_totals,
                duration_minimums,
                duration_maximums,
            )
        ):
            raise RuntimeError(
                "direct full-frame render missed heterogeneous source-duration bounds"
            )
    else:
        duration_targets = tuple(
            total_duration
            * visual_selection.source_counts[source_index]
            / len(resolved)
            for source_index in range(len(eligible))
        )
        duration_minimums = ()
        duration_maximums = ()
    duration_deviations = tuple(
        abs(actual - target) / target
        for actual, target in zip(duration_totals, duration_targets)
    )
    default_duration_threshold = (
        DirectMemberVisualConfig().maximum_source_duration_relative_deviation
    )
    selection_config = getattr(visual_selection, "config", None)
    maximum_duration_deviation = getattr(
        selection_config,
        "maximum_source_duration_relative_deviation",
        default_duration_threshold,
    )
    if (
        source_usage_policy == "exact_balanced_quota_v1"
        and max(duration_deviations) > maximum_duration_deviation + 1e-9
    ):
        raise RuntimeError(
            "direct full-frame render failed source-duration balance: "
            f"max relative deviation {max(duration_deviations):.6f} exceeds "
            f"{maximum_duration_deviation:.6f}"
        )
    reported_durations = getattr(
        visual_selection,
        "source_durations_seconds",
        None,
    )
    if reported_durations is not None and (
        len(reported_durations) != len(duration_totals)
        or any(
            not math.isclose(actual, reported, rel_tol=1e-9, abs_tol=1e-7)
            for actual, reported in zip(duration_totals, reported_durations)
        )
    ):
        raise RuntimeError(
            "direct full-frame selector duration diagnostics disagree with render segments"
        )
    cadence = _render_cadence_payload(resolved)
    switches_per_minute = float(cadence["source_switches_per_minute"])
    median_run = float(cadence["source_run_median_seconds"])
    longest_run = float(cadence["longest_source_run_seconds"])
    maximum_allowed_run_seconds = float(
        visual_selection.optimizer_search_diagnostics.get(
            "maximum_same_source_run_seconds",
            _DIRECT_MEMBER_MAX_RUN_SECONDS,
        )
    )
    cadence["maximum_source_run_seconds"] = maximum_allowed_run_seconds
    if not (
        _DIRECT_MEMBER_MIN_SWITCHES_PER_MINUTE
        <= switches_per_minute
        <= _DIRECT_MEMBER_MAX_SWITCHES_PER_MINUTE
        and _DIRECT_MEMBER_MIN_MEDIAN_RUN_SECONDS
        <= median_run
        <= _DIRECT_MEMBER_MAX_MEDIAN_RUN_SECONDS
        and longest_run <= maximum_allowed_run_seconds
    ):
        raise RuntimeError(
            "direct full-frame cadence missed calibrated safety gates: "
            f"{switches_per_minute:.2f}/min, median {median_run:.2f}s, "
            f"max {longest_run:.2f}s"
        )
    cadence["passed"] = True
    highlight_cadence = _direct_member_highlight_cadence_payload(
        spans=spans,
        visual_selection=visual_selection,
        highlight_intervals=highlight_intervals,
    )
    cadence["highlight_cut_density"] = highlight_cadence
    if not highlight_cadence["passed"]:
        raise RuntimeError(
            "direct full-frame highlight cadence missed human-like density "
            "gates: "
            + ", ".join(highlight_cadence["failure_reasons"])
        )
    cadence["source_duration_balance"] = {
        "passed": True,
        "target_policy": (
            "heterogeneous_role_soft_target_with_hard_exposure_bounds"
            if source_usage_policy
            in _DIRECT_MEMBER_ROLE_AWARE_SOURCE_USAGE_POLICIES
            else "total_duration_times_source_quota_divided_by_span_count"
        ),
        "source_usage_policy": source_usage_policy,
        "source_durations_seconds": duration_totals,
        "source_target_durations_seconds": list(duration_targets),
        "source_minimum_durations_seconds": list(duration_minimums),
        "source_maximum_durations_seconds": list(duration_maximums),
        "source_relative_deviations": list(duration_deviations),
        "maximum_relative_deviation": max(duration_deviations),
        "maximum_allowed_relative_deviation": (
            None
            if source_usage_policy
            in _DIRECT_MEMBER_ROLE_AWARE_SOURCE_USAGE_POLICIES
            else maximum_duration_deviation
        ),
        "enforced_from_render_segments": True,
    }
    cadence["policy"] = (
        "section_aware_reference_calibrated_proof_directed_partition_fallback_v3"
    )
    cadence["partition_search"] = partition_search
    required_cinematic_bridge_indices = tuple(
        boundary.boundary_index
        for boundary in boundary_diagnostics
        if boundary.mismatch.cinematic_bridge is not None
    )
    dissolve_planner_error: str | None = None
    try:
        dissolve_plan = build_direct_member_pose_dissolve_plan(
            visual_selection=visual_selection,
            render_segments=resolved,
            eligible_request_source_indices=eligible,
            eligible_source_durations_seconds=eligible_source_durations_seconds,
            common_master_start=common_master_start,
            source_metadata_contract=source_metadata_contract,
            enabled=member_pose_dissolve_enabled,
            highlight_intervals=highlight_intervals,
            hard_cut_reason=OFFICIAL_TARGET_FANCAM_DIRECT_FULL_FRAME_POLICY,
            fps=output_fps,
        )
    except (
        AttributeError,
        IndexError,
        TypeError,
        ValueError,
        FloatingPointError,
    ) as exc:
        if required_cinematic_bridge_indices:
            raise RuntimeError(
                "required direct-member cinematic bridge planner failed"
            ) from exc
        # Ordinary pose dissolves are optional. A stale or malformed optional
        # diagnostic must never weaken the validated source path.
        dissolve_planner_error = str(exc)
        dissolve_plan = build_direct_member_pose_dissolve_plan(
            visual_selection=visual_selection,
            render_segments=resolved,
            eligible_request_source_indices=eligible,
            eligible_source_durations_seconds=None,
            common_master_start=common_master_start,
            source_metadata_contract=None,
            enabled=member_pose_dissolve_enabled,
            highlight_intervals=(),
            hard_cut_reason=OFFICIAL_TARGET_FANCAM_DIRECT_FULL_FRAME_POLICY,
            fps=output_fps,
        )
    transitions = dissolve_plan.transitions
    transitions = tuple(
        RenderTransition(
            tier="D",
            reason="same_source_hold_no_visible_switch",
        )
        if boundary_index in hold_boundary_indices
        else transition
        for boundary_index, transition in enumerate(transitions)
    )
    for boundary_index in required_cinematic_bridge_indices:
        bridge = boundary_diagnostics[
            boundary_index
        ].mismatch.cinematic_bridge
        expected_kind = (
            TransitionKind.MEMBER_ZOOM_BRIDGE
            if bridge is not None and bridge.style == "zoom_in"
            else TransitionKind.MEMBER_BLUR_BRIDGE
        )
        if transitions[boundary_index].kind is not expected_kind:
            raise RuntimeError(
                "required direct-member cinematic bridge was not materialized"
            )
    dissolve_payload = dissolve_plan.to_dict()
    dissolve_payload["planner_completed"] = dissolve_planner_error is None
    dissolve_payload["planner_error"] = dissolve_planner_error
    cadence["pose_matched_dissolves"] = dissolve_payload
    cadence["member_pose_dissolve_requested"] = bool(
        dissolve_payload["member_pose_dissolve_requested"]
    )
    cadence["member_pose_dissolve_effective"] = bool(
        dissolve_payload["member_pose_dissolve_effective"]
    )
    cadence["member_pose_dissolve_count"] = int(
        dissolve_payload["member_pose_dissolve_count"]
    )
    cadence["member_cinematic_bridge_count"] = int(
        dissolve_payload["cinematic_bridge_count"]
    )
    cadence["member_cinematic_bridge_effective"] = bool(
        dissolve_payload["cinematic_bridge_effective"]
    )
    for segment_index, detail in enumerate(details):
        detail["outgoing_transition"] = (
            transitions[segment_index].to_dict(output_fps)
            if segment_index < len(transitions)
            else None
        )
    return (
        tuple(resolved),
        transitions,
        tuple(details),
        cadence,
        visual_selection,
    )


def _final_render_segments_payload(
    segments: Sequence[RenderSegment],
    *,
    common_master_start: float,
) -> list[dict[str, Any]]:
    """Serialize the exact post-TimeWarp endpoints actually sent to FFmpeg."""

    return [
        {
            "segment_index": index,
            "request_source_index": segment.source_index,
            "canonical_master_start": common_master_start + segment.master_start,
            "canonical_master_end": (
                common_master_start + segment.master_start + segment.duration
            ),
            "output_start_seconds": segment.master_start,
            "output_end_seconds": segment.master_start + segment.duration,
            "source_local_start": segment.source_start,
            "source_local_end": segment.source_end,
            "source_duration_seconds": segment.effective_source_duration,
            "source_playback_rate": segment.source_playback_rate,
            "source_clock_basis": (
                "local_monotonic_timewarp"
                if segment.source_duration is not None
                else "global_scalar_offset"
            ),
            "crop_mode": (
                "dynamic_pan_crop_plus_editorial_camera"
                if segment.pan_crop is not None
                and segment.editorial_transform is not None
                else "static_crop_plus_editorial_camera"
                if segment.crop is not None
                and segment.editorial_transform is not None
                else "editorial_camera"
                if segment.editorial_transform is not None
                else "dynamic_pan_crop"
                if segment.pan_crop is not None
                else "static_crop"
                if segment.crop is not None
                else "full_frame"
            ),
            "editorial_transform_keyframe_count": (
                len(segment.editorial_transform.keyframes)
                if segment.editorial_transform is not None
                else 0
            ),
            "source_look_adjustment": (
                {
                    "brightness": segment.look_adjustment.brightness,
                    "contrast": segment.look_adjustment.contrast,
                    "saturation": segment.look_adjustment.saturation,
                    "temperature_balance": (
                        segment.look_adjustment.temperature_balance
                    ),
                    "tint_balance": segment.look_adjustment.tint_balance,
                    "identity": segment.look_adjustment.is_identity,
                }
                if segment.look_adjustment is not None
                else None
            ),
            "boundary_look_track": (
                {
                    "reason": segment.boundary_look_track.reason,
                    "post_global_residual": True,
                    "keyframes": [
                        {
                            "time_seconds": keyframe.time_seconds,
                            "brightness": keyframe.brightness,
                            "contrast": keyframe.contrast,
                            "saturation": keyframe.saturation,
                            "temperature_balance": (
                                keyframe.temperature_balance
                            ),
                        }
                        for keyframe in segment.boundary_look_track.keyframes
                    ],
                }
                if segment.boundary_look_track is not None
                else None
            ),
        }
        for index, segment in enumerate(segments)
    ]


def _member_render_segments(
    *,
    member_segments: Sequence[MemberDraftSegment],
    timing_segments: Sequence[DraftSegment],
    eligible_indices: Sequence[int],
    common_master_start: float,
    source_timewarps: Sequence[TimeWarpMap | None] | None = None,
) -> tuple[
    tuple[RenderSegment, ...],
    tuple[RenderTransition, ...],
    tuple[dict[str, Any], ...],
]:
    """Resolve confirmed-anchor local member crops, failing on any gap."""

    member_items = tuple(member_segments)
    timing_items = tuple(timing_segments)
    eligible = tuple(eligible_indices)
    if source_timewarps is not None and eligible:
        if max(eligible) >= len(source_timewarps):
            raise ValueError("member source_timewarps do not cover eligible sources")
    if len(member_items) != len(timing_items):
        raise ValueError("member and timing drafts must contain the same segments")

    resolved: list[RenderSegment] = []
    for index, (member_item, timing_item) in enumerate(
        zip(member_items, timing_items, strict=True)
    ):
        if member_item.segment_index != index or timing_item.segment_index != index:
            raise ValueError("member and timing segment indices must be contiguous")
        if (
            abs(
                member_item.canonical_master_start
                - timing_item.canonical_master_start
            )
            > 1e-7
            or abs(
                member_item.canonical_master_end
                - timing_item.canonical_master_end
            )
            > 1e-7
        ):
            raise ValueError("member and timing segments must share canonical spans")

        if member_item.kind is MemberSegmentKind.TARGET_CROP:
            if (
                member_item.source_index is None
                or member_item.audio_offset_seconds is None
                or member_item.source_index >= len(eligible)
            ):
                raise ValueError("target crop references an unknown eligible source")
            eligible_source_index = member_item.source_index
            source_offset = member_item.audio_offset_seconds
            pan_crop = member_item.to_pan_crop_track()
        elif member_item.kind is MemberSegmentKind.UNAVAILABLE:
            raise ValueError(
                "member crop is unavailable; local member mode cannot "
                "fall back to an unverified full frame"
            )
        else:
            raise ValueError("group-wide fallback is not valid for target-member rendering")

        request_source_index = eligible[eligible_source_index]
        rendered_segment = aligned_render_segment(
            source_index=request_source_index,
            reference_start=member_item.canonical_master_start,
            common_master_start=common_master_start,
            source_offset=source_offset,
            duration=member_item.duration_seconds,
            pan_crop=pan_crop,
            timewarp=(
                source_timewarps[request_source_index]
                if source_timewarps is not None
                else None
            ),
        )
        resolved.append(rendered_segment)

    transitions = tuple(
        RenderTransition(reason="default_hard_cut")
        for _ in range(max(0, len(resolved) - 1))
    )
    return (tuple(resolved), transitions, ())


def _attach_member_editorial_motion_tracks(
    *,
    member_segments: Sequence[MemberDraftSegment],
    render_segments: Sequence[RenderSegment],
    motion_plan: EditorialMotionPlan,
) -> tuple[RenderSegment, ...]:
    """Attach only camera tracks backed by confirmed member geometry.

    The planner is expected to enforce this policy too, but the pipeline keeps
    an independent fail-closed boundary so a future planner regression cannot
    add an identity-sensitive camera move to an unavailable or unconfirmed
    member span.
    """

    member_items = tuple(member_segments)
    render_items = tuple(render_segments)
    if len(member_items) != len(render_items):
        raise ValueError(
            "member editorial motion requires one render segment per member segment"
        )

    attached: list[RenderSegment] = []
    for index, (member_item, render_item) in enumerate(
        zip(member_items, render_items, strict=True)
    ):
        if member_item.segment_index != index:
            raise ValueError(
                "member editorial motion segment indices must be contiguous"
            )
        track = motion_plan.track_for(index)
        if track is None:
            attached.append(render_item)
            continue
        if (
            member_item.kind is not MemberSegmentKind.TARGET_CROP
            or not member_item.subject_samples
            or render_item.pan_crop is None
        ):
            raise ValueError(
                "member editorial motion requires confirmed target geometry and crop"
            )
        attached.append(replace(render_item, editorial_transform=track))
    return tuple(attached)


def _attach_direct_member_editorial_motion_tracks(
    *,
    visual_selection: DirectMemberVisualSelection,
    eligible_indices: Sequence[int],
    render_segments: Sequence[RenderSegment],
    motion_plan: EditorialMotionPlan,
) -> tuple[RenderSegment, ...]:
    """Atomically attach dense-reviewed tracks to direct full-frame spans.

    The direct selector and editorial planner use eligible-source indices,
    while ``RenderSegment`` stores request-source indices.  Rechecking that
    mapping here keeps a malformed or stale motion plan from being applied to
    the wrong official fancam.  No partially attached sequence is returned.
    """

    source_path = tuple(int(value) for value in visual_selection.source_indices)
    eligible = tuple(int(value) for value in eligible_indices)
    renders = tuple(render_segments)
    if len(source_path) != len(renders):
        raise ValueError(
            "direct member editorial motion requires one source per render segment"
        )
    if len(motion_plan.segments) != len(renders):
        raise ValueError(
            "direct member editorial motion requires one plan item per render segment"
        )

    attached: list[RenderSegment] = []
    for segment_index, (eligible_source_index, render_item) in enumerate(
        zip(source_path, renders, strict=True)
    ):
        if not 0 <= eligible_source_index < len(eligible):
            raise ValueError(
                "direct member editorial motion references an unknown eligible source"
            )
        request_source_index = eligible[eligible_source_index]
        if render_item.source_index != request_source_index:
            raise ValueError(
                "direct member editorial motion source path disagrees with render"
            )
        if (
            render_item.crop is not None
            or render_item.pan_crop is not None
            or render_item.editorial_transform is not None
        ):
            raise ValueError(
                "direct member editorial motion requires untouched full-frame segments"
            )
        plan_items = tuple(
            item
            for item in motion_plan.segments
            if item.segment_index == segment_index
        )
        if len(plan_items) != 1:
            raise ValueError(
                "direct member editorial motion plan indices must be contiguous"
            )
        plan_item = plan_items[0]
        if plan_item.source_index != eligible_source_index:
            raise ValueError(
                "direct member editorial motion plan source disagrees with selection"
            )
        attached.append(
            render_item
            if plan_item.track is None
            else replace(render_item, editorial_transform=plan_item.track)
        )
    return tuple(attached)


def _attach_direct_member_portrait_fill_geometry(
    *,
    visual_selection: DirectMemberVisualSelection,
    eligible_indices: Sequence[int],
    eligible_source_aspect_ratios: Sequence[float],
    render_segments: Sequence[RenderSegment],
) -> tuple[RenderSegment, ...]:
    """Apply a lossless 9:16 fill to spans without an authored camera track.

    Authored editorial transforms already begin with aspect-fill scaling in the
    renderer. Untouched spans instead receive one deterministic centered crop.
    This is attached only after dense motion planning so face coordinates stay
    in the original decoded-frame domain during review.
    """

    source_path = tuple(int(value) for value in visual_selection.source_indices)
    eligible = tuple(int(value) for value in eligible_indices)
    source_aspect_ratios = tuple(
        float(value) for value in eligible_source_aspect_ratios
    )
    renders = tuple(render_segments)
    if len(source_path) != len(renders):
        raise ValueError(
            "direct member portrait fill requires one source per render segment"
        )
    if len(source_aspect_ratios) != len(eligible):
        raise ValueError(
            "direct member portrait fill aspect ratios must match eligible sources"
        )

    attached: list[RenderSegment] = []
    for eligible_source_index, render_item in zip(
        source_path,
        renders,
        strict=True,
    ):
        if not 0 <= eligible_source_index < len(eligible):
            raise ValueError(
                "direct member portrait fill references an unknown eligible source"
            )
        if render_item.source_index != eligible[eligible_source_index]:
            raise ValueError(
                "direct member portrait fill source path disagrees with render"
            )
        if render_item.crop is not None or render_item.pan_crop is not None:
            raise ValueError(
                "direct member portrait fill requires an unclaimed base crop"
            )
        if render_item.editorial_transform is not None:
            attached.append(render_item)
            continue
        crop = _lossless_member_portrait_fill_crop(
            source_aspect_ratios[eligible_source_index]
        )
        attached.append(
            render_item if crop is None else replace(render_item, crop=crop)
        )
    return tuple(attached)


@dataclass(frozen=True, slots=True)
class _GroupRenderCandidateBundle:
    """One visual alternative after every production post-processing pass.

    The bundle is retained in memory while decoded proxies are ranked.  This
    prevents a winning proxy from being paired accidentally with crops,
    transitions, or camera motion derived from a different timeline.
    """

    candidate_index: int
    review_rank: int
    cadence_profile: str
    pre_render_score: float
    visual_timeline: VisualTimelinePlan
    boundary_repairs: VisualBoundaryRepairPlan | None
    timeline_review: TimelineReview | None
    face_match_plan: FaceMatchPlan | None
    group_transition_plan: GroupTransitionPlan
    boundary_affine_plan: BoundaryAffinePlan | None
    editorial_motion_plan: EditorialMotionPlan | None
    render_plan: RenderPlan
    boundary_look_match: BoundaryLookMatchResult | None = None
    diagnostics: tuple[str, ...] = ()
    candidate_origin: str = "ordinary_visual_baseline"
    candidate_lineage: tuple[dict[str, Any], ...] = ()
    stage_edit_exact_shot_clock: Mapping[str, Any] | None = None
    stage_edit_dynamic_crop_version: int | None = None
    stage_edit_dynamic_crop_v3_decodes: (
        tuple[DynamicCropTrackDecodeV3, ...] | None
    ) = None
    dynamic_crop_motion_timeline_v3: (
        DynamicCropMotionTimelineEvidenceV3 | None
    ) = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "review_rank": self.review_rank,
            "cadence_profile": self.cadence_profile,
            "pre_render_score": self.pre_render_score,
            "visual_timeline": self.visual_timeline.to_dict(),
            "boundary_repairs": (
                self.boundary_repairs.to_dict()
                if self.boundary_repairs is not None
                else None
            ),
            "timeline_review": (
                self.timeline_review.to_dict()
                if self.timeline_review is not None
                else None
            ),
            "face_match": (
                self.face_match_plan.to_dict()
                if self.face_match_plan is not None
                else None
            ),
            "group_transitions": self.group_transition_plan.to_dict(
                self.render_plan.fps
            ),
            "boundary_affine": (
                self.boundary_affine_plan.to_dict()
                if self.boundary_affine_plan is not None
                else None
            ),
            "editorial_motion": (
                self.editorial_motion_plan.to_dict()
                if self.editorial_motion_plan is not None
                else None
            ),
            "boundary_look_match": (
                self.boundary_look_match.to_dict()
                if self.boundary_look_match is not None
                else None
            ),
            "final_render_segments": _final_render_segments_payload(
                self.render_plan.segments,
                common_master_start=self.render_plan.master_audio_start,
            ),
            "diagnostics": list(self.diagnostics),
            "candidate_origin": self.candidate_origin,
            "candidate_lineage": [
                dict(item) for item in self.candidate_lineage
            ],
            "stage_edit_exact_shot_clock": (
                deepcopy(dict(self.stage_edit_exact_shot_clock))
                if self.stage_edit_exact_shot_clock is not None
                else None
            ),
            "stage_edit_dynamic_crop_version": (
                self.stage_edit_dynamic_crop_version
            ),
            "dynamic_crop_v3_decodes": (
                [item.to_dict() for item in self.stage_edit_dynamic_crop_v3_decodes]
                if self.stage_edit_dynamic_crop_v3_decodes is not None
                else None
            ),
            "dynamic_crop_motion_v3": (
                self.dynamic_crop_motion_timeline_v3.to_dict()
                if self.dynamic_crop_motion_timeline_v3 is not None
                else None
            ),
        }


def _apply_stage_edit_candidate_render_contract(
    bundle: _GroupRenderCandidateBundle,
    materialized: StageEditMaterializedVisualCandidate,
    *,
    motion_authority_v3: DynamicCropMotionAuthorityV3 | None = None,
) -> _GroupRenderCandidateBundle:
    """Preserve the learned camera track after ordinary visual derivation."""

    dynamic_crop_version = materialized.dynamic_crop_version
    plan = apply_stage_edit_dynamic_crop_to_render_plan(
        materialized,
        bundle.render_plan,
    )
    dynamic_crop_v3_decodes: tuple[DynamicCropTrackDecodeV3, ...] | None = None
    motion_timeline_v3: DynamicCropMotionTimelineEvidenceV3 | None = None
    if dynamic_crop_version == 3:
        if not isinstance(motion_authority_v3, DynamicCropMotionAuthorityV3):
            raise StageEditVisualCandidateError(
                "Dynamic Crop v3 render promotion requires a strictly replayed "
                "production motion authority"
            )
        raw_decodes = tuple(
            shot.dynamic_crop_decode_v3
            for shot in materialized.retained_candidate.edit.shots
        )
        if any(
            not isinstance(item, DynamicCropTrackDecodeV3)
            for item in raw_decodes
        ):
            raise StageEditVisualCandidateError(
                "Dynamic Crop v3 final edit requires one exact typed decode per shot"
            )
        dynamic_crop_v3_decodes = tuple(
            item for item in raw_decodes if isinstance(item, DynamicCropTrackDecodeV3)
        )
        try:
            plan = apply_dynamic_crop_motion_authority_to_render_plan_v3(
                render_plan=plan,
                decodes=dynamic_crop_v3_decodes,
                authority=motion_authority_v3,
            )
            motion_timeline_v3 = corroborate_dynamic_crop_motion_timeline_v3(
                render_plan=plan,
                decodes=dynamic_crop_v3_decodes,
                authority=motion_authority_v3,
            )
        except (TypeError, ValueError) as exc:
            raise StageEditVisualCandidateError(
                "Dynamic Crop v3 renderer motion authority rejected the candidate"
            ) from exc
    elif motion_authority_v3 is not None:
        raise StageEditVisualCandidateError(
            "Dynamic Crop v2 cannot consume a v3 motion authority"
        )
    original_segments = tuple(
        shot.segment
        for shot in materialized.retained_candidate.edit.shots
    )
    reparameterized = tuple(
        index
        for index, (original, current) in enumerate(
            zip(original_segments, plan.segments, strict=True)
        )
        if not math.isclose(
            original.duration,
            current.duration,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )
    face_dissolve_alignment_segments = tuple(
        sorted(
            {
                segment_index
                for boundary_index, transition in enumerate(
                    bundle.render_plan.transitions
                )
                if transition.kind is TransitionKind.FACE_DISSOLVE
                for segment_index in (boundary_index, boundary_index + 1)
            }
        )
    )
    stripped_materialized_lineage, superseded_clock_hashes = (
        _strip_superseded_exact_stage_edit_shot_clocks(materialized.lineage)
    )
    overlay_event = {
        "adapter": STAGE_EDIT_VISUAL_CANDIDATE_CONTRACT,
        "stage": f"dynamic_crop_v{dynamic_crop_version}_render_plan_overlay",
        "dynamic_crop_version": dynamic_crop_version,
        "dynamic_crop_render_contract": (
            f"stage_edit_dynamic_crop_render_plan_overlay_v{dynamic_crop_version}"
        ),
        "ordinary_editorial_motion_discarded": True,
        "source_path_preserved": True,
        "timing_reparameterized_segment_indices": list(reparameterized),
        "verified_face_dissolve_alignment_preserved": True,
        "dynamic_crop_suppressed_segment_indices": list(
            face_dissolve_alignment_segments
        ),
        "deterministic_render_plan_validation_passed": True,
        "superseded_candidate_history_exact_shot_clock_sha256": list(
            superseded_clock_hashes
        ),
    }
    if dynamic_crop_version == 2:
        overlay_event["dynamic_crop_v2_suppressed_segment_indices"] = list(
            face_dissolve_alignment_segments
        )
    else:
        assert motion_timeline_v3 is not None
        overlay_event.update(
            {
                "dynamic_crop_v3_motion_authority_content_sha256": (
                    motion_timeline_v3.authority.authority_content_sha256
                ),
                "dynamic_crop_v3_motion_timeline_evidence_content_sha256": (
                    motion_timeline_v3.timeline_evidence_content_sha256
                ),
                "dynamic_crop_v3_framing_track_applied_count": (
                    motion_timeline_v3.framing_track_applied_count
                ),
                "dynamic_crop_v3_motion_event_applied_count": (
                    motion_timeline_v3.motion_event_applied_count
                ),
                "dynamic_crop_v3_motion_event_applied_fraction": (
                    motion_timeline_v3.motion_event_applied_fraction
                ),
                "dynamic_crop_v3_motion_budget_within_range": (
                    motion_timeline_v3.motion_budget_within_range
                ),
                "dynamic_crop_v3_static_or_cut_time_reframe_excluded_count": (
                    motion_timeline_v3.static_or_cut_time_reframe_excluded_count
                ),
            }
        )
    lineage = stripped_materialized_lineage + (overlay_event,)
    return replace(
        bundle,
        editorial_motion_plan=None,
        render_plan=plan,
        diagnostics=bundle.diagnostics
        + (
            f"stage_edit_dynamic_crop_v{dynamic_crop_version}_render_contract_applied",
        ),
        candidate_origin="explicit_stage_edit_generation",
        candidate_lineage=lineage,
        stage_edit_exact_shot_clock=_stage_edit_exact_shot_clock_from_edit(
            materialized.retained_candidate.edit
        ),
        stage_edit_dynamic_crop_version=dynamic_crop_version,
        stage_edit_dynamic_crop_v3_decodes=dynamic_crop_v3_decodes,
        dynamic_crop_motion_timeline_v3=motion_timeline_v3,
    )


def _refresh_stage_edit_dynamic_crop_motion_evidence_v3(
    bundle: _GroupRenderCandidateBundle,
) -> _GroupRenderCandidateBundle:
    """Rebind v3 evidence after the final delivery output path is selected."""

    version = bundle.stage_edit_dynamic_crop_version
    if version != 3:
        if (
            bundle.stage_edit_dynamic_crop_v3_decodes is not None
            or bundle.dynamic_crop_motion_timeline_v3 is not None
        ):
            raise StageEditVisualCandidateError(
                "non-v3 bundle cannot retain Dynamic Crop v3 motion evidence"
            )
        return bundle
    decodes = bundle.stage_edit_dynamic_crop_v3_decodes
    previous = bundle.dynamic_crop_motion_timeline_v3
    if decodes is None or not isinstance(
        previous, DynamicCropMotionTimelineEvidenceV3
    ):
        raise StageEditVisualCandidateError(
            "Dynamic Crop v3 final plan lost its typed replay evidence"
        )
    try:
        plan = apply_dynamic_crop_motion_authority_to_render_plan_v3(
            render_plan=bundle.render_plan,
            decodes=decodes,
            authority=previous.authority,
        )
        refreshed = corroborate_dynamic_crop_motion_timeline_v3(
            render_plan=plan,
            decodes=decodes,
            authority=previous.authority,
        )
    except (TypeError, ValueError) as exc:
        raise StageEditVisualCandidateError(
            "Dynamic Crop v3 final delivery plan failed strict motion replay"
        ) from exc
    return replace(
        bundle,
        render_plan=plan,
        dynamic_crop_motion_timeline_v3=refreshed,
        candidate_lineage=bundle.candidate_lineage
        + (
            {
                "stage": "dynamic_crop_v3_final_delivery_plan_evidence_refresh",
                "render_plan_content_sha256": refreshed.render_plan_content_sha256,
                "motion_timeline_evidence_content_sha256": (
                    refreshed.timeline_evidence_content_sha256
                ),
                "motion_authority_content_sha256": (
                    refreshed.authority.authority_content_sha256
                ),
            },
        ),
    )


_BOUNDARY_REPAIR_ELIGIBLE_PRE_REVIEW_FAILURES = frozenset(
    {"focus_mismatch_exceeded"}
)


def _pre_review_hard_failure_codes(
    candidate: Any,
) -> tuple[str, ...]:
    record = getattr(candidate, "candidate_record", candidate)
    review = getattr(record, "review", None)
    return tuple(
        str(code)
        for issue in getattr(review, "hard_failures", ())
        for code in (getattr(issue, "code", None),)
        if code is not None
    )


def _is_focus_only_boundary_repair_candidate(
    candidate: Any,
) -> bool:
    """Admit only candidates whose complete pre-review failure is repairable.

    The decoded boundary repair is specifically allowed to move a source-switch
    timestamp by a few frames to restore focal-subject continuity.  Cadence,
    source-density, duration, and other structural failures are not repairable
    by that operation and therefore remain excluded from proxy candidacy.
    """

    record = getattr(candidate, "candidate_record", candidate)
    review = getattr(record, "review", None)
    if review is None or bool(getattr(review, "passed", False)):
        return False
    codes = frozenset(_pre_review_hard_failure_codes(candidate))
    return bool(codes) and codes.issubset(
        _BOUNDARY_REPAIR_ELIGIBLE_PRE_REVIEW_FAILURES
    )


def _post_repair_review_rejection_payload(
    materialized: MaterializedVisualTimelineCandidate,
    bundle: _GroupRenderCandidateBundle,
) -> dict[str, Any]:
    review = bundle.timeline_review
    return {
        "candidate_index": materialized.candidate_index,
        "review_rank": materialized.review_rank,
        "pre_repair_hard_failure_codes": list(
            _pre_review_hard_failure_codes(materialized)
        ),
        "post_repair_review_available": review is not None,
        "post_repair_hard_failure_codes": [
            str(item.code) for item in review.hard_failures
        ]
        if review is not None
        else [],
        "diagnostics": list(bundle.diagnostics),
    }


def _derive_complete_group_candidate(
    materialized: MaterializedVisualTimelineCandidate,
    *,
    request: AnalysisRequest,
    eligible_indices: Sequence[int],
    audio_offsets: Sequence[float],
    scene_cut_times: Sequence[Sequence[float]],
    source_timewarps: Sequence[TimeWarpMap | None],
    source_aspect_ratios: Sequence[float],
    source_durations_seconds: Sequence[float],
    beat_period_seconds: float | None,
    phrase_times: Sequence[float],
    section_plan: SectionPlan | None,
    common_master_start: float,
    width: int,
    height: int,
    output: Path,
    visual_timeline_override: VisualTimelinePlan | None = None,
    precomputed_boundary_repairs: VisualBoundaryRepairPlan | None = None,
    source_look_match: SourceLookMatchResult | None = None,
    boundary_look_measurement_cache: BoundaryLookMeasurementCache | None = None,
) -> _GroupRenderCandidateBundle:
    """Derive a renderer-ready plan without sharing mutable selected state."""

    local_sources = tuple(
        request.sources[index] for index in eligible_indices
    )
    if not (
        len(local_sources)
        == len(audio_offsets)
        == len(scene_cut_times)
        == len(source_timewarps)
        == len(source_aspect_ratios)
        == len(source_durations_seconds)
    ):
        raise ValueError("complete group candidate source arrays must match")
    diagnostics: list[str] = []
    visual_timeline = (
        visual_timeline_override
        if visual_timeline_override is not None
        else materialized.visual_plan
    )
    boundary_repairs: VisualBoundaryRepairPlan | None = None
    if precomputed_boundary_repairs is not None:
        boundary_repairs = precomputed_boundary_repairs
        visual_timeline = precomputed_boundary_repairs.timeline
    else:
        try:
            boundary_repairs = repair_visual_timeline_boundaries(
                visual_timeline=visual_timeline,
                sources=local_sources,
                audio_offsets=audio_offsets,
                scene_cut_times=scene_cut_times,
                beat_period_seconds=beat_period_seconds,
                source_timewarps=source_timewarps,
            )
            visual_timeline = boundary_repairs.timeline
        except (TypeError, ValueError, cv2.error) as exc:
            diagnostics.append(f"boundary_repair_failed: {exc}")

    timeline_review: TimelineReview | None = None
    try:
        timeline_review = review_timeline(
            visual_timeline,
            available_source_count=len(local_sources),
        )
    except (TypeError, ValueError) as exc:
        diagnostics.append(f"timeline_review_failed: {exc}")

    face_match_plan: FaceMatchPlan | None = None
    try:
        face_match_plan = build_face_match_plan(
            visual_timeline=visual_timeline,
            source_aspect_ratios=source_aspect_ratios,
        )
    except (TypeError, ValueError, cv2.error) as exc:
        diagnostics.append(f"face_match_failed: {exc}")

    group_transition_plan: GroupTransitionPlan | None = None
    try:
        group_transition_plan = build_group_transition_plan(
            visual_timeline=visual_timeline,
            face_match_plan=face_match_plan,
            source_durations_seconds=source_durations_seconds,
        )
    except (TypeError, ValueError) as exc:
        diagnostics.append(f"group_transition_failed: {exc}")
    if group_transition_plan is None:
        group_transition_plan = GroupTransitionPlan(
            tuple(
                RenderTransition(
                    tier="C",
                    reason=(
                        "rhythm_aligned_hard_cut_visual_transition_"
                        "analysis_unavailable"
                    ),
                )
                for _ in range(max(0, len(visual_timeline.choices) - 1))
            ),
            analysis_basis="rhythm_only_hard_cut_fallback_v1",
        )

    boundary_affine_plan: BoundaryAffinePlan | None = None
    boundary_subject_alignment_plan: BoundarySubjectAlignmentPlan | None = None
    editorial_motion_plan: EditorialMotionPlan | None = None
    source_aspects_are_editorial_safe = (
        len(source_aspect_ratios) == len(local_sources)
        and all(
            abs(float(value) - 16.0 / 9.0) <= 0.03
            for value in source_aspect_ratios
        )
    )
    if (
        request.aspect_ratio is AspectRatio.LANDSCAPE
        and source_aspects_are_editorial_safe
    ):
        try:
            boundary_affine_plan = build_boundary_affine_plan(
                visual_timeline=visual_timeline,
                sources=local_sources,
                transitions=group_transition_plan.transitions,
                face_match_plan=face_match_plan,
            )
        except (TypeError, ValueError, cv2.error) as exc:
            diagnostics.append(f"boundary_affine_failed: {exc}")
        try:
            boundary_subject_alignment_plan = build_boundary_subject_alignment_plan(
                visual_timeline=visual_timeline,
                sources=local_sources,
                transitions=group_transition_plan.transitions,
                face_match_plan=face_match_plan,
                output_width=width,
                output_height=height,
            )
        except (OSError, TypeError, ValueError, cv2.error) as exc:
            diagnostics.append(f"boundary_subject_alignment_failed: {exc}")
        try:
            editorial_motion_plan = build_editorial_motion_plan(
                visual_timeline=visual_timeline,
                transitions=group_transition_plan.transitions,
                face_match_plan=face_match_plan,
                boundary_affine_plan=boundary_affine_plan,
                boundary_subject_alignment_plan=boundary_subject_alignment_plan,
                phrase_times=phrase_times,
                section_plan=section_plan,
            )
        except (TypeError, ValueError) as exc:
            diagnostics.append(f"editorial_motion_failed: {exc}")
    else:
        diagnostics.append("editorial_motion_skipped: output_or_source_aspect_unsafe")

    segments = tuple(
        replace(
            aligned_render_segment(
                source_index=eligible_indices[item.source_index],
                reference_start=item.canonical_master_start,
                common_master_start=common_master_start,
                source_offset=audio_offsets[item.source_index],
                duration=item.duration_seconds,
                timewarp=source_timewarps[item.source_index],
                pan_crop=(
                    face_match_plan.pan_crop_for(item.segment_index)
                    if face_match_plan is not None
                    else None
                ),
                editorial_transform=(
                    editorial_motion_plan.track_for(item.segment_index)
                    if editorial_motion_plan is not None
                    else None
                ),
            ),
            look_adjustment=(
                source_look_match.adjustment_for_source(item.source_index)
                if source_look_match is not None
                else None
            ),
        )
        for item in visual_timeline.choices
    )
    plan = RenderPlan(
        sources=request.sources,
        segments=segments,
        reference_source=request.reference_source,
        aspect_ratio=request.aspect_ratio,
        output=Path(output),
        master_audio_start=common_master_start,
        width=width,
        height=height,
        transitions=group_transition_plan.transitions,
    )
    boundary_look_match: BoundaryLookMatchResult | None = None
    if (
        getattr(request, "source_mode", None) is SourceMode.AUTOMATIC
        and request.aspect_ratio is AspectRatio.LANDSCAPE
        and source_aspects_are_editorial_safe
    ):
        try:
            boundary_look_match = analyze_boundary_look_match(
                plan,
                group_16x9_safe=True,
                measurement_cache=boundary_look_measurement_cache,
            )
            boundary_look_match = verify_boundary_look_match_render_domain(
                plan,
                boundary_look_match,
            )
            plan = boundary_look_match.apply(plan)
        except (OSError, TypeError, ValueError, cv2.error) as exc:
            # Fail closed: a missing or volatile measurement leaves the
            # original pixels untouched and the unchanged decoded-proxy gate
            # remains responsible for rejecting the candidate.
            diagnostics.append(f"boundary_look_match_failed: {exc}")
    record = materialized.candidate_record
    pre_render_score = float(
        timeline_review.weighted_score
        if timeline_review is not None
        else record.optimizer_score
    )
    return _GroupRenderCandidateBundle(
        candidate_index=materialized.candidate_index,
        review_rank=materialized.review_rank,
        cadence_profile=materialized.cadence_profile,
        pre_render_score=pre_render_score,
        visual_timeline=visual_timeline,
        boundary_repairs=boundary_repairs,
        timeline_review=timeline_review,
        face_match_plan=face_match_plan,
        group_transition_plan=group_transition_plan,
        boundary_affine_plan=boundary_affine_plan,
        editorial_motion_plan=editorial_motion_plan,
        render_plan=plan,
        boundary_look_match=boundary_look_match,
        diagnostics=tuple(diagnostics),
    )


def _stage_edit_montage_analysis_payload(
    *,
    bundle: _GroupRenderCandidateBundle,
    proxy_review_payload: Mapping[str, Any],
    section_plan: SectionPlan,
    source_derivative_diversity: SourceDerivativeDiversityResult,
    source_index_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one decoded proxy to its complete learned edit and source domains."""

    plan_payload = bundle.render_plan.to_json()
    exact_clock_value = getattr(bundle, "stage_edit_exact_shot_clock", None)
    if not isinstance(exact_clock_value, Mapping):
        raise RuntimeError(
            "StageEdit rendered montage review requires the final edit's exact "
            "serialized-token shot clock"
        )
    exact_clock = deepcopy(dict(exact_clock_value))
    if exact_clock.get("contract") != EXACT_STAGE_EDIT_SHOT_CLOCK_CONTRACT:
        raise RuntimeError("StageEdit exact serialized-token shot clock changed")
    dynamic_crop_version = bundle.stage_edit_dynamic_crop_version
    if dynamic_crop_version not in (2, 3):
        raise RuntimeError(
            "StageEdit montage analysis requires an exact Dynamic Crop version"
        )
    decodes_v3 = bundle.stage_edit_dynamic_crop_v3_decodes
    motion_v3 = bundle.dynamic_crop_motion_timeline_v3
    if dynamic_crop_version == 3:
        if (
            decodes_v3 is None
            or len(decodes_v3) != len(bundle.render_plan.segments)
            or not isinstance(motion_v3, DynamicCropMotionTimelineEvidenceV3)
        ):
            raise RuntimeError(
                "Dynamic Crop v3 montage analysis lost strict decode/motion evidence"
            )
    elif decodes_v3 is not None or motion_v3 is not None:
        raise RuntimeError(
            "Dynamic Crop v2 montage analysis cannot retain v3 evidence"
        )
    return {
        "section_plan": section_plan.to_dict(),
        "visual_timeline": _with_index_domain(
            bundle.visual_timeline.to_dict(),
            ELIGIBLE_SOURCE_INDEX_DOMAIN,
        ),
        "source_derivative_diversity": _with_index_domain(
            source_derivative_diversity.to_dict(),
            ELIGIBLE_SOURCE_INDEX_DOMAIN,
        ),
        "source_index_mapping": deepcopy(dict(source_index_mapping)),
        "stage_edit_exact_shot_clock": exact_clock,
        "dynamic_crop_version": dynamic_crop_version,
        "dynamic_crop_v3_decodes": (
            [item.to_dict() for item in decodes_v3]
            if decodes_v3 is not None
            else None
        ),
        "dynamic_crop_motion_v3": (
            motion_v3.to_dict() if motion_v3 is not None else None
        ),
        "editorial_motion": (
            bundle.editorial_motion_plan.to_dict()
            if bundle.editorial_motion_plan is not None
            else None
        ),
        # Dynamic Crop v2/v3 is attached directly to RenderSegment after the
        # ordinary editorial-motion plan is intentionally discarded.
        "render": {
            "segments": deepcopy(plan_payload["segments"]),
            "transitions": deepcopy(plan_payload["transitions"]),
            "master_audio_start": plan_payload["master_audio_start"],
            "fps": plan_payload["fps"],
        },
        "rendered_proxy_review": deepcopy(dict(proxy_review_payload)),
    }


def _stage_edit_exact_shot_clock_from_edit(edit: StageEdit) -> dict[str, Any]:
    """Serialize the final edit's exact shot clock, including repaired shots."""

    shots: list[dict[str, Any]] = []
    for position, shot in enumerate(edit.shots):
        if shot.shot_index != position:
            raise RuntimeError("StageEdit exact shot clock indices changed")
        if (
            shot.section_role is None
            or shot.section_progress is None
            or shot.phrase_progress is None
        ):
            raise RuntimeError(
                "StageEdit final shot lacks exact serialized-token music structure"
            )
        shots.append(
            {
                "shot_index": shot.shot_index,
                "source_index": shot.segment.source_index,
                "section_role": shot.section_role,
                "section_progress": shot.section_progress,
                "phrase_progress": shot.phrase_progress,
            }
        )
    return {
        "contract": EXACT_STAGE_EDIT_SHOT_CLOCK_CONTRACT,
        "source_index_domain": ELIGIBLE_SOURCE_INDEX_DOMAIN,
        "shots": shots,
    }


def _strip_superseded_exact_stage_edit_shot_clocks(
    lineage: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    """Keep repair provenance without retaining two authoritative clocks."""

    stripped: list[dict[str, Any]] = []
    superseded_hashes: list[str] = []
    for raw in lineage:
        item = deepcopy(dict(raw))
        exact = item.pop("exact_music_structure_shot_clock", None)
        if exact is not None:
            if not isinstance(exact, Mapping):
                raise RuntimeError("superseded StageEdit shot clock is malformed")
            encoded = json.dumps(
                exact,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            digest = hashlib.sha256(encoded).hexdigest()
            superseded_hashes.append(digest)
            item["superseded_exact_music_structure_shot_clock"] = {
                "sha256": digest,
                "authoritative": False,
                "reason": "replaced_by_accepted_windowed_semimarkov_redecode",
            }
        stripped.append(item)
    return tuple(stripped), tuple(superseded_hashes)


def _review_stage_edit_bundle_montage(
    *,
    bundle: _GroupRenderCandidateBundle,
    proxy_review_payload: Mapping[str, Any],
    section_plan: SectionPlan,
    source_derivative_diversity: SourceDerivativeDiversityResult,
    source_index_mapping: Mapping[str, Any],
    source_scene_cut_times: Sequence[Sequence[float]],
    config: RenderedMontageGrammarConfig,
    reference_profile: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    analysis = _stage_edit_montage_analysis_payload(
        bundle=bundle,
        proxy_review_payload=proxy_review_payload,
        section_plan=section_plan,
        source_derivative_diversity=source_derivative_diversity,
        source_index_mapping=source_index_mapping,
    )
    review = review_rendered_montage_grammar(
        analysis,
        render_plan=bundle.render_plan.to_json(),
        config=config,
    )
    if bundle.timeline_review is None:
        raise RuntimeError(
            "StageEdit professional boundary grammar requires the exact "
            "selected timeline review"
        )
    render_plan_identity = hashlib.sha256(
        json.dumps(
            bundle.render_plan.to_json(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    boundary_evidence = adapt_group_boundary_grammar_evidence(
        performance_id=f"ive-group-{render_plan_identity[:20]}",
        render_plan=bundle.render_plan,
        source_index_mapping=source_index_mapping,
        selected_timeline=bundle.visual_timeline,
        source_scene_cut_times=source_scene_cut_times,
        timeline_review=bundle.timeline_review,
        section_plan=section_plan,
        rendered_proxy_review=proxy_review_payload,
        editorial_motion=bundle.editorial_motion_plan,
        dynamic_crop_version=bundle.stage_edit_dynamic_crop_version,
        dynamic_crop_motion_v3=bundle.dynamic_crop_motion_timeline_v3,
    )
    boundary_assessment = assess_professional_boundary_grammar(
        boundary_evidence
    )
    review["professional_boundary_grammar_evidence"] = (
        boundary_evidence.to_dict()
    )
    review["professional_boundary_grammar_assessment"] = (
        boundary_assessment.to_dict()
    )
    review["passed_before_professional_boundary_grammar"] = bool(
        review.get("passed") is True
    )
    review["passed"] = bool(
        review.get("passed") is True and boundary_assessment.passed
    )
    review["reference_profile"] = deepcopy(dict(reference_profile))
    return review, analysis


def _apply_bounded_human_target_proxy_rank(
    payload: dict[str, Any],
    montage_review: Mapping[str, Any],
    *,
    technical_passed: bool,
) -> None:
    """Apply bounded human-target rank signals without changing technical score.

    Hard gates are evaluated first and cannot be rescued.  The close/non-wide
    P50 terms are symmetric, while the P75 wide-run term saturates at or below
    its target, so ever-more close-ups or ever-shorter wide shots cannot accrue
    unbounded rank.  The raw decoded score remains authoritative and unchanged
    for perceptual admission, repair, and final-media QA; only a separate
    candidate-ranking score receives the editorial adjustment.  A qualifying
    close-only burst is a separate zero-or-additive soft signal weighted by its
    observed event support (six events / twenty references), so absence does
    not lower the established three-component alignment.
    """

    aggregate = montage_review.get("highlight_aggregate")
    soft_targets = (
        aggregate.get("soft_targets")
        if isinstance(aggregate, Mapping)
        else None
    )
    raw_alignment = (
        soft_targets.get("aggregate_alignment_score")
        if isinstance(soft_targets, Mapping)
        else None
    )
    hard_passed = bool(
        technical_passed and montage_review.get("passed") is True
    )
    if raw_alignment is None:
        if hard_passed:
            raise RuntimeError(
                "hard-passing montage review lacks bounded human-target "
                "alignment evidence"
            )
        return
    if (
        isinstance(raw_alignment, bool)
        or not isinstance(raw_alignment, (int, float))
        or not math.isfinite(float(raw_alignment))
        or not 0.0 <= float(raw_alignment) <= 1.0
    ):
        raise RuntimeError(
            "rendered montage human-target alignment score is malformed"
        )
    raw_score = payload.get("weighted_score")
    if (
        isinstance(raw_score, bool)
        or not isinstance(raw_score, (int, float))
        or not math.isfinite(float(raw_score))
    ):
        raise RuntimeError("decoded proxy weighted score is malformed")
    alignment = float(raw_alignment)
    close_burst_evidence = (
        soft_targets.get("close_only_multi_source_burst_event_p10")
        if isinstance(soft_targets, Mapping)
        else None
    )
    close_burst_bonus = 0.0
    if close_burst_evidence is not None:
        if not isinstance(close_burst_evidence, Mapping):
            raise RuntimeError("rendered close-burst soft evidence is malformed")
        close_alignment = close_burst_evidence.get("alignment_score")
        support_weight = close_burst_evidence.get(
            "event_occurrence_support_weight"
        )
        observed_event_count = close_burst_evidence.get("observed_event_count")
        human_reference_count = close_burst_evidence.get(
            "human_reference_count"
        )
        binding_active = close_burst_evidence.get("policy_binding_active")
        if binding_active is False:
            if any(
                value is not None
                for value in (
                    close_alignment,
                    support_weight,
                    close_burst_evidence.get(
                        "evidence_weighted_alignment_score"
                    ),
                    close_burst_evidence.get("policy_binding"),
                )
            ):
                raise RuntimeError(
                    "inactive rendered close-burst evidence claims an active "
                    "score or policy binding"
                )
        elif not (
            binding_active is True
            and isinstance(close_burst_evidence.get("policy_binding"), Mapping)
            and isinstance(observed_event_count, int)
            and not isinstance(observed_event_count, bool)
            and observed_event_count > 0
            and isinstance(human_reference_count, int)
            and not isinstance(human_reference_count, bool)
            and human_reference_count > 0
            and isinstance(close_alignment, (int, float))
            and not isinstance(close_alignment, bool)
            and math.isfinite(float(close_alignment))
            and 0.0 <= float(close_alignment) <= 1.0
            and isinstance(support_weight, (int, float))
            and not isinstance(support_weight, bool)
            and math.isfinite(float(support_weight))
            and math.isclose(
                float(support_weight),
                min(1.0, observed_event_count / human_reference_count),
                abs_tol=1.0e-12,
            )
        ):
            raise RuntimeError(
                "rendered close-burst policy binding or support weight is "
                "malformed"
            )
        evidence_weighted = close_burst_evidence.get(
            "evidence_weighted_alignment_score"
        )
        if evidence_weighted is not None:
            if (
                isinstance(evidence_weighted, bool)
                or not isinstance(evidence_weighted, (int, float))
                or not math.isfinite(float(evidence_weighted))
                or not 0.0 <= float(evidence_weighted) <= 1.0
            ):
                raise RuntimeError(
                    "rendered close-burst evidence-weighted score is malformed"
                )
            if not math.isclose(
                float(evidence_weighted),
                float(close_alignment) * float(support_weight),
                abs_tol=1.0e-12,
            ):
                raise RuntimeError(
                    "rendered close-burst evidence-weighted score disagrees "
                    "with its hash-bound support"
                )
            close_burst_bonus = (
                float(evidence_weighted) if hard_passed else 0.0
            )
    maximum_absolute_adjustment = 1.0
    adjustment = (
        maximum_absolute_adjustment * (2.0 * alignment - 1.0)
        if hard_passed
        else 0.0
    )
    adjusted = min(
        100.0,
        max(0.0, float(raw_score) + adjustment + close_burst_bonus),
    )
    payload["technical_proxy_weighted_score_before_human_target"] = float(
        raw_score
    )
    payload["human_target_bounded_ranking"] = {
        "contract": (
            "hard_pass_bounded_symmetric_human_target_plus_evidence_"
            "weighted_close_burst_rerank_v2"
        ),
        "eligible_only_after_technical_and_montage_hard_pass": True,
        "eligible": hard_passed,
        "aggregate_alignment_score": alignment,
        "maximum_absolute_adjustment_points": maximum_absolute_adjustment,
        "applied_adjustment_points": adjustment,
        "close_burst_zero_or_additive_bonus_points": close_burst_bonus,
        "close_burst_bonus_weight_is_observed_event_support_fraction": True,
        "close_burst_absence_lowers_existing_alignment": False,
        "total_applied_adjustment_points": adjustment + close_burst_bonus,
        "technical_weighted_score": float(raw_score),
        "adjusted_weighted_score": adjusted,
        "closeup_target_is_symmetric": True,
        "nonwide_target_is_symmetric": True,
        "wide_run_reward_saturates_at_or_below_p75": True,
        "over_close_past_target_is_rewarded": False,
        "soft_target_evidence": deepcopy(dict(soft_targets)),
    }
    payload["ranking_weighted_score"] = adjusted


def _review_complete_group_candidates(
    candidates: Sequence[_GroupRenderCandidateBundle],
    *,
    scratch_directory: Path,
    progress_callback: Any | None = None,
    professional_reference_profile: ProfessionalReferenceProfile | None = None,
    professional_reference_cadence_profile: (
        ProfessionalReferenceCadenceProfile | None
    ) = None,
    learned_reviewer_session: LearnedReviewerProductionSession | None = None,
    stage_edit_montage_section_plan: SectionPlan | None = None,
    stage_edit_source_derivative_diversity: (
        SourceDerivativeDiversityResult | None
    ) = None,
    stage_edit_source_index_mapping: Mapping[str, Any] | None = None,
    stage_edit_source_scene_cut_times: (
        Sequence[Sequence[float]] | None
    ) = None,
    stage_edit_montage_config: RenderedMontageGrammarConfig | None = None,
    stage_edit_montage_reference_profile: Mapping[str, Any] | None = None,
) -> RenderedCandidateSelection:
    """Rank complete bundles with actual decoded proxies, never metadata only."""

    stage_edit_candidates = tuple(
        item
        for item in candidates
        if getattr(item, "candidate_origin", "ordinary_visual_baseline")
        == "explicit_stage_edit_generation"
    )
    stage_edit_montage_required = bool(stage_edit_candidates)
    if stage_edit_montage_required:
        if len(stage_edit_candidates) != len(candidates):
            raise ValueError(
                "StageEdit proxy review cannot mix learned and baseline candidates"
            )
        if (
            stage_edit_montage_section_plan is None
            or stage_edit_source_derivative_diversity is None
            or stage_edit_source_index_mapping is None
            or stage_edit_source_scene_cut_times is None
            or stage_edit_montage_config is None
            or stage_edit_montage_reference_profile is None
        ):
            raise ValueError(
                "StageEdit decoded candidates require montage grammar, source-family, "
                "index-domain, and immutable reference-profile evidence"
            )

    rendered = tuple(
        RenderedCandidate(
            candidate_index=item.candidate_index,
            render_plan=item.render_plan,
            pre_render_score=item.pre_render_score,
            payload=item,
            rank_metadata=(
                RenderedCandidateRankMetadata(
                    source_switches_per_minute=(
                        getattr(
                            item,
                            "timeline_review",
                        ).source_switches_per_minute
                    )
                )
                if getattr(item, "timeline_review", None) is not None
                else None
            ),
        )
        for item in candidates
    )
    bundles_by_plan_identity = {
        id(item.render_plan): item for item in candidates
    }

    def render_review(
        plan: RenderPlan,
        proxy_output: Path,
    ) -> Any:
        proxy_plan, review = render_and_review_proxy(
            plan,
            proxy_output,
            professional_reference_profile=professional_reference_profile,
        )
        if not stage_edit_montage_required:
            return proxy_plan, review
        bundle = bundles_by_plan_identity.get(id(plan))
        if bundle is None:
            raise ValueError(
                "decoded StageEdit proxy plan is not bound to one candidate bundle"
            )
        assert stage_edit_montage_section_plan is not None
        assert stage_edit_source_derivative_diversity is not None
        assert stage_edit_source_index_mapping is not None
        assert stage_edit_source_scene_cut_times is not None
        assert stage_edit_montage_config is not None
        assert stage_edit_montage_reference_profile is not None
        payload = deepcopy(review.payload)
        technical_passed = bool(payload.get("passed"))
        montage_review, _montage_analysis = _review_stage_edit_bundle_montage(
            bundle=bundle,
            proxy_review_payload=payload,
            section_plan=stage_edit_montage_section_plan,
            source_derivative_diversity=(
                stage_edit_source_derivative_diversity
            ),
            source_index_mapping=stage_edit_source_index_mapping,
            source_scene_cut_times=stage_edit_source_scene_cut_times,
            config=stage_edit_montage_config,
            reference_profile=stage_edit_montage_reference_profile,
        )
        payload["technical_proxy_passed_before_montage_grammar"] = (
            technical_passed
        )
        payload["rendered_montage_grammar_review"] = montage_review
        payload["passed"] = bool(
            technical_passed and montage_review.get("passed") is True
        )
        _apply_bounded_human_target_proxy_rank(
            payload,
            montage_review,
            technical_passed=technical_passed,
        )
        return proxy_plan, RenderedProxyReview(payload)

    return review_rendered_candidates(
        rendered,
        scratch_directory=Path(scratch_directory),
        maximum_review_count=8,
        minimum_required_candidate_count=8,
        render_review=render_review,
        progress_callback=progress_callback,
        professional_reference_profile=professional_reference_profile,
        professional_reference_cadence_profile=(
            professional_reference_cadence_profile
        ),
        learned_reviewer_session=learned_reviewer_session,
    )


@dataclass(frozen=True, slots=True)
class _ProxyTriggeredPartialReeditResult:
    selected_bundle: _GroupRenderCandidateBundle
    selected_review_payload: dict[str, Any] | None
    audit_payload: dict[str, Any]


def _decoded_proxy_rank(
    payload: dict[str, Any] | None,
    *,
    pre_render_score: float,
) -> tuple[bool, bool, float, float, float, float]:
    if payload is None:
        return (False, False, -math.inf, -1.0, -math.inf, pre_render_score)
    boundary = payload.get("visible_boundary_score")
    if not isinstance(boundary, dict):
        boundary = payload.get("boundary_score")
    boundary_payload = boundary if isinstance(boundary, dict) else {}
    weighted_score = float(payload.get("weighted_score", -math.inf))
    issue_fraction = float(boundary_payload.get("issue_fraction", 1.0))
    boundary_p10 = float(boundary_payload.get("p10", -math.inf))
    return (
        bool(payload.get("passed")),
        True,
        weighted_score,
        -issue_fraction,
        boundary_p10,
        pre_render_score,
    )


def _partial_reedit_invariant_check(
    original: _GroupRenderCandidateBundle,
    partial: _GroupRenderCandidateBundle,
    *,
    requested_issue_indices: Sequence[int],
) -> tuple[tuple[int, ...], dict[str, bool], tuple[str, ...]]:
    """Prove a local timing repair did not mutate unrelated edit decisions."""

    original_choices = original.visual_timeline.choices
    partial_choices = partial.visual_timeline.choices
    requested = set(int(value) for value in requested_issue_indices)
    errors: list[str] = []
    same_segment_count = len(original_choices) == len(partial_choices)
    if not same_segment_count:
        errors.append("segment_count_changed")
    comparable_count = min(len(original_choices), len(partial_choices))
    original_sources = tuple(
        item.source_index for item in original_choices
    )
    partial_sources = tuple(item.source_index for item in partial_choices)
    source_sequence_preserved = original_sources == partial_sources
    if not source_sequence_preserved:
        errors.append("source_sequence_changed")
    render_source_sequence_preserved = tuple(
        item.source_index for item in original.render_plan.segments
    ) == tuple(item.source_index for item in partial.render_plan.segments)
    if not render_source_sequence_preserved:
        errors.append("render_source_sequence_changed")

    range_preserved = bool(original_choices and partial_choices) and math.isclose(
        original_choices[0].canonical_master_start,
        partial_choices[0].canonical_master_start,
        rel_tol=0.0,
        abs_tol=1e-7,
    ) and math.isclose(
        original_choices[-1].canonical_master_end,
        partial_choices[-1].canonical_master_end,
        rel_tol=0.0,
        abs_tol=1e-7,
    )
    if not range_preserved:
        errors.append("canonical_first_or_last_range_changed")

    actual_changed = tuple(
        boundary_index
        for boundary_index in range(max(0, comparable_count - 1))
        if not math.isclose(
            original_choices[boundary_index].canonical_master_end,
            partial_choices[boundary_index].canonical_master_end,
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        or not math.isclose(
            original_choices[boundary_index + 1].canonical_master_start,
            partial_choices[boundary_index + 1].canonical_master_start,
            rel_tol=0.0,
            abs_tol=1e-7,
        )
    )
    changed_boundaries_requested = set(actual_changed).issubset(requested)
    if not changed_boundaries_requested:
        errors.append("changed_boundary_outside_requested_issue_indices")
    outside_timestamps_preserved = same_segment_count and all(
        math.isclose(
            original_choices[index].canonical_master_end,
            partial_choices[index].canonical_master_end,
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        and math.isclose(
            original_choices[index + 1].canonical_master_start,
            partial_choices[index + 1].canonical_master_start,
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        for index in range(max(0, comparable_count - 1))
        if index not in requested
    )
    if not outside_timestamps_preserved:
        errors.append("non_issue_boundary_timestamp_changed")
    output_frame_count_preserved = (
        original.render_plan.output_frame_count
        == partial.render_plan.output_frame_count
    )
    if not output_frame_count_preserved:
        errors.append("output_frame_count_changed")
    render_duration_preserved = math.isclose(
        original.render_plan.render_duration,
        partial.render_plan.render_duration,
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    if not render_duration_preserved:
        errors.append("render_duration_changed")
    checks = {
        "segment_count_preserved": same_segment_count,
        "source_sequence_preserved": source_sequence_preserved,
        "render_source_sequence_preserved": render_source_sequence_preserved,
        "canonical_first_last_range_preserved": range_preserved,
        "outside_issue_boundary_timestamps_preserved": (
            outside_timestamps_preserved
        ),
        "actual_changed_boundaries_subset_of_requested": (
            changed_boundaries_requested
        ),
        "output_frame_count_preserved": output_frame_count_preserved,
        "render_duration_preserved": render_duration_preserved,
    }
    return (actual_changed, checks, tuple(dict.fromkeys(errors)))


def _attempt_proxy_triggered_partial_reedit(
    selected_bundle: _GroupRenderCandidateBundle,
    selected_materialized: Any,
    original_review_payload: dict[str, Any] | None,
    *,
    request: AnalysisRequest,
    eligible_indices: Sequence[int],
    audio_offsets: Sequence[float],
    scene_cut_times: Sequence[Sequence[float]],
    source_timewarps: Sequence[TimeWarpMap | None],
    source_aspect_ratios: Sequence[float],
    source_durations_seconds: Sequence[float],
    beat_period_seconds: float | None,
    phrase_times: Sequence[float],
    section_plan: SectionPlan | None,
    common_master_start: float,
    width: int,
    height: int,
    scratch_directory: Path,
    source_look_match: SourceLookMatchResult | None = None,
    boundary_look_measurement_cache: BoundaryLookMeasurementCache | None = None,
    professional_reference_profile: ProfessionalReferenceProfile | None = None,
    learned_issue_boundary_indices: Sequence[int] | None = None,
    candidate_bundle_postprocessor: Any | None = None,
) -> _ProxyTriggeredPartialReeditResult:
    """Try one decoded-QA-targeted timing repair, retaining the original on ties."""

    audit: dict[str, Any] = {
        "analysis_basis": (
            "single_pass_decoded_issue_targeted_plus_minus_eight_frame_reedit_v1"
        ),
        "attempted": False,
        "search_radius_frames": 8,
        "interior_scene_cut_guard_frames": 9,
        "outside_issue_boundaries_preserved": False,
        "issue_boundary_indices": [],
        "ignored_issue_boundary_indices": [],
        "non_introduced_issue_boundary_indices": [],
        "issue_index_source_field": None,
        "changed_boundary_indices": [],
        "changed_count": 0,
        "before_review": deepcopy(original_review_payload),
        "after_review": None,
        "partial_review_error": None,
        "improved": False,
        "selected_version": "original",
        "reason": None,
        "professional_reference_profile": (
            professional_reference_profile.provenance_dict()
            if professional_reference_profile is not None
            else None
        ),
        "candidate_origin": selected_bundle.candidate_origin,
        "candidate_lineage_before": [
            dict(item) for item in selected_bundle.candidate_lineage
        ],
    }
    learned_suggestions: tuple[int, ...] | None = None
    if learned_issue_boundary_indices is not None:
        if isinstance(learned_issue_boundary_indices, (str, bytes)):
            raise TypeError("learned issue boundary indices must be a sequence")
        learned_suggestions = tuple(learned_issue_boundary_indices)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in learned_suggestions
        ) or len(set(learned_suggestions)) != len(learned_suggestions):
            raise ValueError(
                "learned issue boundary indices must be unique non-negative integers"
            )
        audit["learned_reviewer_issue_suggestions"] = {
            "usage": "repair_window_proposal_only",
            "deterministic_commit_authority_unchanged": True,
            "boundary_indices": list(learned_suggestions),
        }
    if original_review_payload is None:
        audit["reason"] = "selected_proxy_review_unavailable"
        return _ProxyTriggeredPartialReeditResult(
            selected_bundle,
            original_review_payload,
            audit,
        )
    if bool(original_review_payload.get("passed")):
        audit["reason"] = "selected_proxy_already_passed"
        return _ProxyTriggeredPartialReeditResult(
            selected_bundle,
            original_review_payload,
            audit,
        )
    issue_index_source_field = "partial_reedit_boundary_issue_indices"
    raw_issue_indices = original_review_payload.get(issue_index_source_field)
    if not isinstance(raw_issue_indices, list):
        issue_index_source_field = (
            "introduced_cross_source_boundary_issue_indices"
        )
        raw_issue_indices = original_review_payload.get(issue_index_source_field)
    if not isinstance(raw_issue_indices, list):
        # Schema-v2 compatibility: filter its all-boundary issue list against
        # the actual selected source sequence before attempting a repair.
        issue_index_source_field = "boundary_issue_indices"
        raw_issue_indices = original_review_payload.get(issue_index_source_field)
    if not isinstance(raw_issue_indices, list):
        audit["reason"] = "boundary_issue_indices_unavailable"
        return _ProxyTriggeredPartialReeditResult(
            selected_bundle,
            original_review_payload,
            audit,
        )
    boundary_count = max(0, len(selected_bundle.visual_timeline.choices) - 1)
    valid_indices = tuple(
        sorted(
            {
                int(value)
                for value in raw_issue_indices
                if isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value < boundary_count
            }
        )
    )
    learned_valid_indices = tuple(
        sorted(
            value
            for value in (learned_suggestions or ())
            if value < boundary_count
        )
    )
    combined_valid_indices = tuple(
        sorted(set(valid_indices) | set(learned_valid_indices))
    )
    choices = selected_bundle.visual_timeline.choices
    non_introduced = tuple(
        index
        for index in combined_valid_indices
        if choices[index].source_index == choices[index + 1].source_index
    )
    issue_indices = tuple(
        index
        for index in combined_valid_indices
        if index not in set(non_introduced)
    )
    ignored = [
        value
        for value in raw_issue_indices
        if not (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value < boundary_count
        )
    ]
    ignored.extend(non_introduced)
    ignored.extend(
        value
        for value in (learned_suggestions or ())
        if value >= boundary_count
    )
    audit["issue_index_source_field"] = (
        f"{issue_index_source_field}_plus_learned_reviewer_suggestions"
        if learned_suggestions is not None
        else issue_index_source_field
    )
    audit["issue_boundary_indices"] = list(issue_indices)
    audit["ignored_issue_boundary_indices"] = ignored
    audit["non_introduced_issue_boundary_indices"] = list(non_introduced)
    if learned_suggestions is not None:
        audit["learned_reviewer_issue_suggestions"].update(
            {
                "valid_boundary_indices": list(learned_valid_indices),
                "used_boundary_indices": [
                    value
                    for value in learned_valid_indices
                    if value in set(issue_indices)
                ],
            }
        )
    if not issue_indices:
        audit["reason"] = (
            "no_introduced_cross_source_issue_boundaries"
            if non_introduced
            else "no_valid_issue_boundaries"
        )
        return _ProxyTriggeredPartialReeditResult(
            selected_bundle,
            original_review_payload,
            audit,
        )

    audit["attempted"] = True
    local_sources = tuple(request.sources[index] for index in eligible_indices)
    try:
        targeted_repairs = repair_visual_timeline_boundaries(
            visual_timeline=selected_bundle.visual_timeline,
            sources=local_sources,
            audio_offsets=audio_offsets,
            scene_cut_times=scene_cut_times,
            beat_period_seconds=beat_period_seconds,
            source_timewarps=source_timewarps,
            boundary_indices=issue_indices,
            config=BoundaryRepairConfig(
                search_radius_frames=8,
                interior_scene_cut_guard_frames=9,
            ),
        )
        changed_boundary_indices = [
            item.boundary_index
            for item in targeted_repairs.repairs
            if item.result.changed
        ]
        audit["repair"] = targeted_repairs.to_dict()
        audit["reported_changed_boundary_indices"] = changed_boundary_indices
        audit["changed_count"] = targeted_repairs.changed_count
        if not changed_boundary_indices:
            audit["reason"] = "targeted_repair_made_no_timing_change"
            return _ProxyTriggeredPartialReeditResult(
                selected_bundle,
                original_review_payload,
                audit,
            )
        partial_bundle = _derive_complete_group_candidate(
            selected_materialized,
            request=request,
            eligible_indices=eligible_indices,
            audio_offsets=audio_offsets,
            scene_cut_times=scene_cut_times,
            source_timewarps=source_timewarps,
            source_aspect_ratios=source_aspect_ratios,
            source_durations_seconds=source_durations_seconds,
            beat_period_seconds=beat_period_seconds,
            phrase_times=phrase_times,
            section_plan=section_plan,
            common_master_start=common_master_start,
            width=width,
            height=height,
            output=(
                Path(scratch_directory)
                / (
                    f".partial-candidate-"
                    f"{selected_bundle.candidate_index:02d}.mp4"
                )
            ),
            visual_timeline_override=targeted_repairs.timeline,
            precomputed_boundary_repairs=targeted_repairs,
            source_look_match=source_look_match,
            boundary_look_measurement_cache=boundary_look_measurement_cache,
        )
        if candidate_bundle_postprocessor is not None:
            partial_bundle = candidate_bundle_postprocessor(
                partial_bundle,
                selected_materialized,
            )
        (
            actual_changed_boundary_indices,
            invariant_checks,
            invariant_errors,
        ) = _partial_reedit_invariant_check(
            selected_bundle,
            partial_bundle,
            requested_issue_indices=issue_indices,
        )
        audit["changed_boundary_indices"] = list(
            actual_changed_boundary_indices
        )
        audit["changed_count"] = len(actual_changed_boundary_indices)
        audit["invariant_checks"] = invariant_checks
        audit["invariant_errors"] = list(invariant_errors)
        if invariant_errors:
            audit["reason"] = (
                "partial_reedit_invariant_violation_original_retained"
            )
            audit["error"] = "; ".join(invariant_errors)
            return _ProxyTriggeredPartialReeditResult(
                selected_bundle,
                original_review_payload,
                audit,
            )
        audit["outside_issue_boundaries_preserved"] = True
        if not actual_changed_boundary_indices:
            audit["reason"] = "targeted_repair_made_no_timing_change"
            return _ProxyTriggeredPartialReeditResult(
                selected_bundle,
                original_review_payload,
                audit,
            )
        if candidate_bundle_postprocessor is not None:
            postprocessor_tail = partial_bundle.candidate_lineage[-1:]
            partial_bundle = replace(
                partial_bundle,
                candidate_origin=selected_bundle.candidate_origin,
                candidate_lineage=selected_bundle.candidate_lineage
                + (
                    {
                        "stage": "decoded_proxy_issue_targeted_boundary_repair",
                        "requested_issue_boundary_indices": list(issue_indices),
                        "changed_boundary_indices": list(
                            actual_changed_boundary_indices
                        ),
                        "outside_issue_boundaries_preserved": True,
                    },
                )
                + postprocessor_tail,
            )
        render_review = render_and_review_proxy
        if professional_reference_profile is not None:
            def render_review(
                plan: RenderPlan,
                proxy_output: Path,
            ) -> Any:
                return render_and_review_proxy(
                    plan,
                    proxy_output,
                    professional_reference_profile=(
                        professional_reference_profile
                    ),
                )

        partial_selection = review_rendered_candidates(
            (
                RenderedCandidate(
                    candidate_index=partial_bundle.candidate_index,
                    render_plan=partial_bundle.render_plan,
                    pre_render_score=partial_bundle.pre_render_score,
                    payload=partial_bundle,
                ),
            ),
            scratch_directory=Path(scratch_directory),
            maximum_review_count=1,
            minimum_required_candidate_count=1,
            render_review=render_review,
        )
        partial_review = partial_selection.selected
        after_review = deepcopy(partial_review.review_payload)
        audit["after_review"] = after_review
        if after_review is None:
            audit["partial_review_error"] = deepcopy(
                getattr(partial_review, "error", None)
            )
            audit["reason"] = (
                "partial_reedit_proxy_review_failed_original_retained"
            )
            return _ProxyTriggeredPartialReeditResult(
                selected_bundle,
                original_review_payload,
                audit,
            )
        improved = _decoded_proxy_rank(
            after_review,
            pre_render_score=partial_bundle.pre_render_score,
        ) > _decoded_proxy_rank(
            original_review_payload,
            pre_render_score=selected_bundle.pre_render_score,
        )
        audit["improved"] = improved
        if improved:
            audit["selected_version"] = "partial_reedit"
            audit["reason"] = "decoded_proxy_rank_improved"
            audit["candidate_lineage_after"] = [
                dict(item) for item in partial_bundle.candidate_lineage
            ]
            return _ProxyTriggeredPartialReeditResult(
                partial_bundle,
                after_review,
                audit,
            )
        audit["reason"] = "decoded_proxy_rank_did_not_improve"
    except (OSError, RuntimeError, TypeError, ValueError, cv2.error) as exc:
        audit["reason"] = "partial_reedit_failed_original_retained"
        audit["error"] = f"{type(exc).__name__}: {exc}"
    return _ProxyTriggeredPartialReeditResult(
        selected_bundle,
        original_review_payload,
        audit,
    )


def _stage_edit_render_signature(edit: StageEdit) -> str:
    payload = edit.to_render_plan().to_json()
    payload.pop("output", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stage_edit_boundary_issues(
    edit: StageEdit,
    boundary_indices: Sequence[int],
    *,
    prefix: str,
    causes: Sequence[str],
    radius_seconds: float = 0.20,
) -> tuple[StageEditRepairIssue, ...]:
    result: list[StageEditRepairIssue] = []
    end = edit.master_audio_start + edit.duration_seconds
    for boundary_index in sorted(set(int(value) for value in boundary_indices)):
        if not 0 <= boundary_index < len(edit.boundaries):
            continue
        outgoing = edit.shots[boundary_index].segment
        at_seconds = (
            edit.master_audio_start
            + outgoing.master_start
            + outgoing.duration
        )
        result.append(
            StageEditRepairIssue(
                issue_id=f"{prefix}-boundary-{boundary_index}",
                start_seconds=max(
                    edit.master_audio_start,
                    at_seconds - radius_seconds,
                ),
                end_seconds=min(end, at_seconds + radius_seconds),
                causes=tuple(causes),
                severity="error",
            )
        )
    return tuple(result)


_STAGE_EDIT_PERCEPTUAL_REPAIR_INDEX_SOURCE_FIELD = (
    "perceptual_source_admission.admission."
    "repair_target_boundary_indices"
)
_STAGE_EDIT_PERCEPTUAL_SOURCE_INDEX_DOMAIN = RENDER_PLAN_SOURCE_INDEX_DOMAIN
_STAGE_EDIT_PERCEPTUAL_BOUNDARY_INDEX_DOMAIN = (
    "render_plan_transition_index"
)
_STAGE_EDIT_PERCEPTUAL_PASS_BONUS = 101.0
_STAGE_EDIT_PERCEPTUAL_SOURCE_SUBSTITUTION_ANALYSIS_BASIS = (
    "targeted_perceptual_source_sequence_substitution_v1"
)


def _stage_edit_source_path_in_window(
    edit: StageEdit,
    *,
    start_seconds: float,
    end_seconds: float,
) -> tuple[int, ...]:
    path: list[int] = []
    for shot in edit.shots:
        shot_start = edit.master_audio_start + shot.segment.master_start
        shot_end = shot_start + shot.segment.duration
        if (
            shot_end <= start_seconds + 1e-6
            or shot_start >= end_seconds - 1e-6
        ):
            continue
        source_index = shot.segment.source_index
        if not path or path[-1] != source_index:
            path.append(source_index)
    if not path:
        raise ValueError(
            "StageEdit perceptual target window has no candidate source path"
        )
    return tuple(path)


def _stage_edit_targeted_source_substitution_evidence(
    original: StageEdit,
    candidate: StageEdit,
    boundary_indices: Sequence[int],
) -> dict[str, Any]:
    """Prove a repair changed source decisions, not only cut/crop timing."""

    raw_indices = tuple(boundary_indices)
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < len(original.boundaries)
        for value in raw_indices
    ):
        raise ValueError(
            "StageEdit perceptual substitution target index is invalid"
        )
    normalized_indices = tuple(sorted(set(raw_indices)))
    if (
        tuple(Path(value).expanduser().resolve() for value in original.sources)
        != tuple(Path(value).expanduser().resolve() for value in candidate.sources)
    ):
        raise ValueError(
            "StageEdit perceptual substitution candidate changed its source list"
        )
    if (
        not math.isclose(
            original.master_audio_start,
            candidate.master_audio_start,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            original.duration_seconds,
            candidate.duration_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError(
            "StageEdit perceptual substitution candidate changed the master range"
        )
    original_full_source_path = tuple(
        shot.segment.source_index for shot in original.shots
    )
    candidate_full_source_path = tuple(
        shot.segment.source_index for shot in candidate.shots
    )
    global_source_sequence_changed = (
        candidate_full_source_path != original_full_source_path
    )

    windows: list[dict[str, Any]] = []
    substituted_indices: list[int] = []
    for boundary_index in normalized_indices:
        outgoing = original.shots[boundary_index].segment
        incoming = original.shots[boundary_index + 1].segment
        start_seconds = original.master_audio_start + outgoing.master_start
        end_seconds = (
            original.master_audio_start
            + incoming.master_start
            + incoming.duration
        )
        original_path = _stage_edit_source_path_in_window(
            original,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        candidate_path = _stage_edit_source_path_in_window(
            candidate,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        window_path_changed = candidate_path != original_path
        substituted = bool(
            global_source_sequence_changed and window_path_changed
        )
        if substituted:
            substituted_indices.append(boundary_index)
        windows.append(
            {
                "boundary_index": boundary_index,
                "boundary_index_domain": (
                    _STAGE_EDIT_PERCEPTUAL_BOUNDARY_INDEX_DOMAIN
                ),
                "window_start_seconds": start_seconds,
                "window_end_seconds": end_seconds,
                "original_stage_edit_source_path": list(original_path),
                "candidate_stage_edit_source_path": list(candidate_path),
                "source_index_domain": ELIGIBLE_SOURCE_INDEX_DOMAIN,
                "window_collapsed_source_path_changed": window_path_changed,
                "source_sequence_substituted": substituted,
            }
        )
    required = bool(normalized_indices)
    satisfied = bool(substituted_indices) if required else True
    return {
        "analysis_basis": (
            _STAGE_EDIT_PERCEPTUAL_SOURCE_SUBSTITUTION_ANALYSIS_BASIS
        ),
        "required": required,
        "satisfied": satisfied,
        "global_source_sequence_changed": global_source_sequence_changed,
        "original_full_stage_edit_source_path": list(
            original_full_source_path
        ),
        "candidate_full_stage_edit_source_path": list(
            candidate_full_source_path
        ),
        "target_boundary_indices": list(normalized_indices),
        "substituted_target_boundary_indices": substituted_indices,
        "target_window_policy": (
            "collapsed_source_path_over_original_two_shots_touching_boundary"
        ),
        "timing_or_crop_only_change_counts_as_substitution": False,
        "serialized_media_paths": False,
        "windows": windows,
    }


def _stage_edit_perceptual_repair_targets(
    edit: StageEdit,
    review_payload: Mapping[str, Any] | None,
    *,
    required: bool = False,
    review_render_plan: RenderPlan | None = None,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Strict-load scalar-only perceptual repair targets for one StageEdit.

    Older/synthetic review payloads may omit this newly-added evidence.  Once
    the initial selected proxy supplies it, however, every proposal review must
    supply the same contract; silently falling back to timing-only evidence
    would let a source-pumping candidate bypass the decoded gate.
    """

    audit: dict[str, Any] = {
        "available": False,
        "required_for_proposal_reviews": bool(required),
        "index_source_field": (
            _STAGE_EDIT_PERCEPTUAL_REPAIR_INDEX_SOURCE_FIELD
        ),
        "source_index_domain": None,
        "stage_edit_source_index_domain": ELIGIBLE_SOURCE_INDEX_DOMAIN,
        "boundary_index_domain": None,
        "review_render_plan_supplied_explicitly": (
            review_render_plan is not None
        ),
        "admission_passed": None,
        "reason_codes": [],
        "reported_boundary_indices": [],
        "used_boundary_indices": [],
        "boundary_source_pairs": [],
        "selection_penalty_points": None,
        "weighted_score_before_perceptual_admission": None,
        "penalized_weighted_score": None,
        "decoded_pass_bonus_points": _STAGE_EDIT_PERCEPTUAL_PASS_BONUS,
        "decoded_selection_gate": (
            "admission_pass_first_bonus_101_then_penalized_weighted_score"
        ),
    }
    if review_payload is None or "perceptual_source_admission" not in review_payload:
        if required:
            raise ValueError(
                "StageEdit proposal review is missing required scalar payload "
                "perceptual_source_admission"
            )
        return (), audit

    raw_source_admission = review_payload["perceptual_source_admission"]
    if not isinstance(raw_source_admission, Mapping):
        raise ValueError(
            "StageEdit perceptual_source_admission must be an object"
        )
    if (
        raw_source_admission.get("analysis_basis")
        != PERCEPTUAL_SOURCE_ADMISSION_ANALYSIS_BASIS
    ):
        raise ValueError(
            "StageEdit perceptual_source_admission analysis_basis is invalid"
        )
    if raw_source_admission.get("raw_frames_persisted") is not False:
        raise ValueError(
            "StageEdit perceptual_source_admission must declare "
            "raw_frames_persisted=false"
        )
    if raw_source_admission.get("original_pixels_serialized") is not False:
        raise ValueError(
            "StageEdit perceptual_source_admission must declare "
            "original_pixels_serialized=false"
        )

    raw_domains = raw_source_admission.get("index_domains")
    if not isinstance(raw_domains, Mapping):
        raise ValueError(
            "StageEdit perceptual_source_admission index_domains must be an object"
        )
    source_index_domain = raw_domains.get("source_index")
    boundary_index_domain = raw_domains.get("boundary_index")
    if source_index_domain != _STAGE_EDIT_PERCEPTUAL_SOURCE_INDEX_DOMAIN:
        raise ValueError(
            "StageEdit perceptual source_index domain does not match the "
            "render-plan sources-array contract"
        )
    if boundary_index_domain != _STAGE_EDIT_PERCEPTUAL_BOUNDARY_INDEX_DOMAIN:
        raise ValueError(
            "StageEdit perceptual boundary_index domain does not match the "
            "render-plan transition contract"
        )

    raw_admission = raw_source_admission.get("admission")
    if not isinstance(raw_admission, Mapping):
        raise ValueError(
            "StageEdit perceptual_source_admission.admission must be an object"
        )
    admission_passed = raw_admission.get("passed")
    if not isinstance(admission_passed, bool):
        raise ValueError(
            "StageEdit perceptual admission passed must be a boolean"
        )
    raw_reason_codes = raw_admission.get("reason_codes")
    if (
        not isinstance(raw_reason_codes, list)
        or any(
            not isinstance(value, str) or not value.strip()
            for value in raw_reason_codes
        )
        or len(set(raw_reason_codes)) != len(raw_reason_codes)
    ):
        raise ValueError(
            "StageEdit perceptual admission reason_codes must be unique "
            "non-blank strings"
        )
    if admission_passed != (not raw_reason_codes):
        raise ValueError(
            "StageEdit perceptual admission passed/reason_codes disagree"
        )
    overall_passed = review_payload.get("passed")
    if not isinstance(overall_passed, bool):
        raise ValueError("StageEdit decoded proxy passed must be a boolean")
    if not admission_passed and overall_passed:
        raise ValueError(
            "StageEdit decoded proxy cannot pass a failed perceptual admission"
        )

    raw_indices = raw_admission.get("repair_target_boundary_indices")
    if not isinstance(raw_indices, list):
        raise ValueError(
            "StageEdit perceptual repair target boundary indices must be an array"
        )
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < len(edit.boundaries)
            for value in raw_indices
        )
        or raw_indices != sorted(set(raw_indices))
    ):
        raise ValueError(
            "StageEdit perceptual repair target boundary indices must be "
            "sorted unique in-range integers"
        )
    if not admission_passed and not raw_indices:
        raise ValueError(
            "StageEdit failing perceptual admission must identify at least "
            "one repair target boundary"
        )
    checked_review_plan = (
        review_render_plan
        if review_render_plan is not None
        else edit.to_render_plan()
    )
    if not isinstance(checked_review_plan, RenderPlan):
        raise TypeError("StageEdit perceptual review_render_plan must be a RenderPlan")
    if (
        len(checked_review_plan.segments) != len(edit.shots)
        or len(checked_review_plan.transitions) != len(edit.boundaries)
    ):
        raise ValueError(
            "StageEdit perceptual review RenderPlan does not preserve the "
            "one-shot/one-transition index contract"
        )
    segment_source_translation: list[tuple[int, int]] = []
    for shot, render_segment in zip(
        edit.shots,
        checked_review_plan.segments,
        strict=True,
    ):
        stage_edit_source_index = shot.segment.source_index
        render_plan_source_index = render_segment.source_index
        if (
            Path(edit.sources[stage_edit_source_index]).expanduser().resolve()
            != Path(
                checked_review_plan.sources[render_plan_source_index]
            ).expanduser().resolve()
        ):
            raise ValueError(
                "StageEdit perceptual review RenderPlan source mapping does "
                "not match the StageEdit media identity"
            )
        segment_source_translation.append(
            (stage_edit_source_index, render_plan_source_index)
        )

    source_pairs: list[dict[str, int | str]] = []
    for boundary_index in raw_indices:
        outgoing_stage_edit_index, outgoing_render_plan_index = (
            segment_source_translation[boundary_index]
        )
        incoming_stage_edit_index, incoming_render_plan_index = (
            segment_source_translation[boundary_index + 1]
        )
        if (
            outgoing_stage_edit_index == incoming_stage_edit_index
            or outgoing_render_plan_index == incoming_render_plan_index
        ):
            raise ValueError(
                "StageEdit perceptual repair target does not identify a "
                "cross-source boundary"
            )
        source_pairs.append(
            {
                "boundary_index": boundary_index,
                "boundary_index_domain": (
                    _STAGE_EDIT_PERCEPTUAL_BOUNDARY_INDEX_DOMAIN
                ),
                "outgoing_render_plan_source_index": (
                    outgoing_render_plan_index
                ),
                "incoming_render_plan_source_index": (
                    incoming_render_plan_index
                ),
                "render_plan_source_index_domain": (
                    RENDER_PLAN_SOURCE_INDEX_DOMAIN
                ),
                "outgoing_stage_edit_source_index": (
                    outgoing_stage_edit_index
                ),
                "incoming_stage_edit_source_index": (
                    incoming_stage_edit_index
                ),
                "stage_edit_source_index_domain": (
                    ELIGIBLE_SOURCE_INDEX_DOMAIN
                ),
            }
        )

    def finite_number(value: Any, field: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"StageEdit {field} must be a finite number")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"StageEdit {field} must be a finite number"
            ) from exc
        if not math.isfinite(number):
            raise ValueError(f"StageEdit {field} must be a finite number")
        return number

    selection_penalty = finite_number(
        raw_admission.get("selection_penalty_points"),
        "perceptual admission selection_penalty_points",
    )
    mirrored_penalty = finite_number(
        review_payload.get("perceptual_source_admission_penalty_points"),
        "perceptual_source_admission_penalty_points",
    )
    weighted_score_before = finite_number(
        review_payload.get(
            "weighted_score_before_perceptual_source_admission"
        ),
        "weighted_score_before_perceptual_source_admission",
    )
    weighted_score = finite_number(
        review_payload.get("weighted_score"),
        "perceptual admission penalized weighted_score",
    )
    if not 0.0 <= selection_penalty <= 20.0:
        raise ValueError(
            "StageEdit perceptual admission selection penalty is out of range"
        )
    if not math.isclose(selection_penalty, mirrored_penalty, abs_tol=1e-9):
        raise ValueError(
            "StageEdit perceptual admission selection penalty mirror disagrees"
        )
    if not 0.0 <= weighted_score_before <= 100.0:
        raise ValueError(
            "StageEdit weighted score before perceptual admission is out of range"
        )
    if not 0.0 <= weighted_score <= 100.0:
        raise ValueError(
            "StageEdit perceptual admission penalized weighted_score is out of range"
        )
    if not math.isclose(
        weighted_score,
        max(0.0, weighted_score_before - selection_penalty),
        abs_tol=1e-9,
    ):
        raise ValueError(
            "StageEdit perceptual admission weighted score/penalty disagree"
        )
    if admission_passed and not math.isclose(
        selection_penalty,
        0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "StageEdit passing perceptual admission cannot retain a selection penalty"
        )
    if not admission_passed and selection_penalty <= 0.0:
        raise ValueError(
            "StageEdit failing perceptual admission must carry a positive "
            "selection penalty"
        )

    used_indices = tuple(raw_indices) if not admission_passed else ()
    audit.update(
        {
            "available": True,
            "source_index_domain": source_index_domain,
            "boundary_index_domain": boundary_index_domain,
            "admission_passed": admission_passed,
            "reason_codes": list(raw_reason_codes),
            "reported_boundary_indices": list(raw_indices),
            "used_boundary_indices": list(used_indices),
            "boundary_source_pairs": source_pairs,
            "selection_penalty_points": selection_penalty,
            "weighted_score_before_perceptual_admission": (
                weighted_score_before
            ),
            "penalized_weighted_score": weighted_score,
        }
    )
    return used_indices, audit


def _stage_edit_proxy_repair_review(
    edit: StageEdit,
    review_payload: Mapping[str, Any] | None,
    *,
    require_perceptual_source_admission: bool = False,
    review_render_plan: RenderPlan | None = None,
) -> StageEditRepairReview:
    (
        perceptual_issue_indices,
        perceptual_admission_audit,
    ) = _stage_edit_perceptual_repair_targets(
        edit,
        review_payload,
        required=require_perceptual_source_admission,
        review_render_plan=review_render_plan,
    )
    if review_payload is None:
        return StageEditRepairReview(
            decoded_objective=-1_000_000.0,
            professional_objective=-1_000_000.0,
            cadence_objective=-1_000_000.0,
            technical_errors=("decoded_proxy_review_unavailable",),
            audit={
                "decoded_proxy_review_available": False,
                "perceptual_source_admission": perceptual_admission_audit,
            },
        )

    def finite_metric(value: Any, fallback: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return fallback
        return number if math.isfinite(number) else fallback

    visible = review_payload.get("visible_boundary_score")
    visible_payload = visible if isinstance(visible, Mapping) else {}
    professional = review_payload.get("professional_reference_gate")
    professional_payload = professional if isinstance(professional, Mapping) else {}
    scalar = professional_payload.get("scalar_score")
    scalar_payload = scalar if isinstance(scalar, Mapping) else {}
    weighted_score = finite_metric(
        review_payload.get("weighted_score"),
        -1_000_000.0,
    )
    decoded_objective = weighted_score + (
        _STAGE_EDIT_PERCEPTUAL_PASS_BONUS
        if (
            perceptual_admission_audit["available"]
            and perceptual_admission_audit["admission_passed"] is True
        )
        else 0.0
    )
    visible_p10 = finite_metric(visible_payload.get("p10"), weighted_score)
    professional_objective = finite_metric(
        scalar_payload.get("candidate_p10"),
        visible_p10,
    )

    switches = sum(
        left.segment.source_index != right.segment.source_index
        for left, right in zip(edit.shots, edit.shots[1:])
    )
    switches_per_minute = switches * 60.0 / max(edit.duration_seconds, 1e-9)
    target_switches_per_minute = sum(
        section.target_source_switches_per_minute * section.duration_seconds
        for section in edit.section_plan.sections
    ) / max(edit.duration_seconds, 1e-9)
    cadence_objective = -abs(
        switches_per_minute - target_switches_per_minute
    )

    raw_technical = review_payload.get(
        "introduced_cross_source_technical_issue_reasons",
        (),
    )
    technical = tuple(
        sorted(
            {
                str(value).strip()
                for value in raw_technical
                if str(value).strip()
            }
        )
    ) if isinstance(raw_technical, (list, tuple)) else (
        "malformed_decoded_proxy_technical_evidence",
    )
    raw_issue_indices = review_payload.get(
        "partial_reedit_boundary_issue_indices",
        (),
    )
    issue_indices = (
        tuple(
            value
            for value in raw_issue_indices
            if isinstance(value, int) and not isinstance(value, bool)
        )
        if isinstance(raw_issue_indices, (list, tuple))
        else ()
    )
    visible_issues = _stage_edit_boundary_issues(
        edit,
        issue_indices,
        prefix="decoded-proxy",
        causes=("decoded_proxy_visible_boundary_failure",),
    )
    perceptual_issues = _stage_edit_boundary_issues(
        edit,
        perceptual_issue_indices,
        prefix="decoded-proxy-perceptual-source-admission",
        causes=("decoded_proxy_perceptual_source_admission_failure",),
    )
    return StageEditRepairReview(
        decoded_objective=decoded_objective,
        professional_objective=professional_objective,
        cadence_objective=cadence_objective,
        technical_errors=technical,
        issue_windows=visible_issues + perceptual_issues,
        audit={
            "decoded_proxy_review_available": True,
            "decoded_proxy_passed": bool(review_payload.get("passed")),
            "weighted_score": weighted_score,
            "decoded_objective": decoded_objective,
            "decoded_objective_gate": (
                "perceptual_admission_pass_first_bonus_101_then_"
                "penalized_weighted_score"
            ),
            "perceptual_source_admission": perceptual_admission_audit,
            "professional_or_visible_p10": professional_objective,
            "source_switches_per_minute": switches_per_minute,
            "target_source_switches_per_minute": (
                target_switches_per_minute
            ),
            "cadence_distance_objective": cadence_objective,
        },
    )


def _committed_window_continuity_lineage(
    repair_result: Any,
    redecoder: StageEditWindowedBeamRedecoder,
) -> dict[str, Any]:
    """Fail closed unless every committed splice retains exact scorer evidence."""

    def finite_number(value: Any, name: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RuntimeError(
                f"committed windowed continuity {name} is not finite"
            )
        return float(value)

    call_by_iteration: dict[int, Mapping[str, Any]] = {}
    for raw_call in redecoder.audit_payload().get("calls", ()):
        if not isinstance(raw_call, Mapping):
            raise RuntimeError("windowed re-decode call audit is malformed")
        iteration = raw_call.get("iteration")
        if (
            isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or iteration < 1
            or iteration in call_by_iteration
        ):
            raise RuntimeError("windowed re-decode call iteration is invalid")
        call_by_iteration[iteration] = raw_call

    identity_fields = (
        "continuity_authority",
        "continuity_scorer_contract",
        "continuity_scorer_fingerprint",
    )
    committed_evaluations: list[dict[str, Any]] = []
    committed_identities: set[tuple[str, str, str]] = set()
    for iteration_audit in repair_result.audit.iterations:
        committed_id = iteration_audit.committed_proposal_id
        if committed_id is None:
            continue
        committed_records = tuple(
            record
            for record in iteration_audit.proposal_audits
            if record.get("committed") is True
        )
        if (
            len(committed_records) != 1
            or committed_records[0].get("proposal_id") != committed_id
        ):
            raise RuntimeError(
                "committed windowed repair proposal audit is ambiguous"
            )
        call_audit = call_by_iteration.get(iteration_audit.iteration)
        if call_audit is None:
            raise RuntimeError(
                "committed windowed repair lacks its decoder call audit"
            )
        record = committed_records[0]
        proposal_audit = record.get("decoder_audit")
        if not isinstance(proposal_audit, Mapping):
            raise RuntimeError(
                "committed windowed repair lacks proposal decoder evidence"
            )
        identity_values: list[str] = []
        for field_name in identity_fields:
            value = proposal_audit.get(field_name)
            if not isinstance(value, str) or not value:
                raise RuntimeError(
                    "committed windowed repair continuity identity is missing"
                )
            if value != call_audit.get(field_name):
                raise RuntimeError(
                    "committed windowed repair continuity identity changed"
                )
            identity_values.append(value)
        authority, scorer_contract, scorer_fingerprint = identity_values
        if (
            len(scorer_fingerprint) != 64
            or any(value not in "0123456789abcdef" for value in scorer_fingerprint)
        ):
            raise RuntimeError(
                "committed windowed repair scorer fingerprint is invalid"
            )
        learned_bound = proposal_audit.get(
            "learned_source_pair_scorer_bound"
        )
        expected_learned_bound = (
            authority == PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
        )
        if authority not in {
            HANDCRAFTED_CONTINUITY_AUTHORITY,
            PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
        }:
            raise RuntimeError(
                "committed windowed repair continuity authority is unsupported"
            )
        if (
            learned_bound is not expected_learned_bound
            or call_audit.get("learned_source_pair_scorer_bound")
            is not expected_learned_bound
        ):
            raise RuntimeError(
                "committed windowed repair scorer binding is untruthful"
            )
        expected_contract = (
            STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT
            if expected_learned_bound
            else HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
        )
        if scorer_contract != expected_contract:
            raise RuntimeError(
                "committed windowed repair scorer contract conflicts with authority"
            )
        if (
            not expected_learned_bound
            and scorer_fingerprint
            != HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
        ):
            raise RuntimeError(
                "committed handcrafted continuity fingerprint changed"
            )

        exterior = proposal_audit.get("exterior_splice_continuity")
        if not isinstance(exterior, Mapping):
            raise RuntimeError(
                "committed windowed repair lacks exterior continuity evidence"
            )
        if exterior.get("learned_source_pair_scorer_bound") is not (
            expected_learned_bound
        ):
            raise RuntimeError(
                "committed exterior continuity scorer binding changed"
            )
        if any(
            exterior.get(name) != value
            for name, value in zip(
                identity_fields,
                identity_values,
                strict=True,
            )
        ):
            raise RuntimeError(
                "committed exterior continuity identity changed"
            )
        request_payload = iteration_audit.request
        expected_sides = tuple(
            side
            for side, context_field in (
                ("left", "left_context_decision_id"),
                ("right", "right_context_decision_id"),
            )
            if request_payload.get(context_field) is not None
        )
        edges = exterior.get("edges")
        if not isinstance(edges, (list, tuple)) or any(
            not isinstance(edge, Mapping) for edge in edges
        ):
            raise RuntimeError(
                "committed exterior continuity edges are malformed"
            )
        expected_edge_count = exterior.get("expected_edge_count")
        evaluated_edge_count = exterior.get("evaluated_edge_count")
        if (
            exterior.get("full_edit_validation_passed") is not True
            or expected_edge_count != len(expected_sides)
            or evaluated_edge_count != expected_edge_count
            or len(edges) != expected_edge_count
            or tuple(edge.get("side") for edge in edges) != expected_sides
        ):
            raise RuntimeError(
                "committed exterior continuity did not validate every splice edge"
            )
        raw_total = 0.0
        weighted_total = 0.0
        for edge in edges:
            if edge.get("authority_rule_evaluated") is not True:
                raise RuntimeError(
                    "committed exterior continuity edge was not evaluated"
                )
            if any(
                edge.get(name) != value
                for name, value in zip(
                    identity_fields,
                    identity_values,
                    strict=True,
                )
            ):
                raise RuntimeError(
                    "committed exterior edge scorer identity changed"
                )
            scorer_evaluated = edge.get("scorer_evaluated")
            if (
                edge.get("cross_derivative_family") is not True
                or edge.get("learned_source_pair_scorer_bound") is not (
                    expected_learned_bound
                )
                or not isinstance(scorer_evaluated, bool)
                or edge.get("scorer_invoked") is not scorer_evaluated
                or (expected_learned_bound and not scorer_evaluated)
                or (
                    not scorer_evaluated
                    and edge.get("neutral_reason")
                    != "handcrafted_missing_observation_neutral"
                )
            ):
                raise RuntimeError(
                    "committed exterior edge scorer evaluation is untruthful"
                )
            raw_score = finite_number(edge.get("raw_score"), "raw score")
            weight = finite_number(edge.get("weight"), "score weight")
            weighted_score = finite_number(
                edge.get("weighted_score"),
                "weighted score",
            )
            if (
                not -1.0 <= raw_score <= 0.0
                or weight < 0.0
                or not math.isclose(
                    weighted_score,
                    raw_score * weight,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise RuntimeError(
                    "committed exterior continuity score is outside its contract"
                )
            raw_total += raw_score
            weighted_total += weighted_score
        reported_raw_total = finite_number(
            exterior.get("raw_score_total"),
            "raw score total",
        )
        reported_weighted_total = finite_number(
            exterior.get("weighted_score_total"),
            "weighted score total",
        )
        if (
            not math.isclose(
                raw_total,
                reported_raw_total,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                weighted_total,
                reported_weighted_total,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise RuntimeError(
                "committed exterior continuity totals are inconsistent"
            )
        decoder_score = finite_number(
            proposal_audit.get("decoder_timeline_score"),
            "decoder timeline score",
        )
        selection_score = finite_number(
            proposal_audit.get("candidate_selection_score"),
            "candidate selection score",
        )
        if (
            proposal_audit.get(
                "exterior_score_applied_to_candidate_selection"
            )
            is not True
            or proposal_audit.get(
                "full_edit_exterior_continuity_rechecked"
            )
            is not True
            or not math.isclose(
                selection_score,
                decoder_score + weighted_total,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise RuntimeError(
                "committed exterior continuity was not applied to selection"
            )
        committed_identities.add(
            (authority, scorer_contract, scorer_fingerprint)
        )
        committed_evaluations.append(
            {
                "iteration": iteration_audit.iteration,
                "proposal_id": committed_id,
                "base_edit_sha256": request_payload.get("base_edit_sha256"),
                "committed_edit_sha256": iteration_audit.committed_edit_sha256,
                "decoder_timeline_rank": proposal_audit.get(
                    "decoder_timeline_rank"
                ),
                "candidate_selection_rank": proposal_audit.get(
                    "candidate_selection_rank"
                ),
                "candidate_selection_score": selection_score,
                "exterior_splice_continuity": deepcopy(dict(exterior)),
            }
        )
    if (
        len(committed_evaluations)
        != repair_result.audit.committed_iteration_count
        or not committed_evaluations
        or len(committed_identities) != 1
    ):
        raise RuntimeError(
            "committed windowed repair continuity lineage is incomplete or mixed"
        )
    authority, scorer_contract, scorer_fingerprint = next(
        iter(committed_identities)
    )
    return {
        "continuity_authority": authority,
        "continuity_scorer_contract": scorer_contract,
        "continuity_scorer_fingerprint": scorer_fingerprint,
        "continuity_provenance_owner": "committed_window_semimarkov_redecode",
        "initial_decode_continuity_provenance_authoritative_for_repaired_edges": False,
        "committed_window_exterior_continuity_evaluations": committed_evaluations,
    }


def _committed_repair_dynamic_crop_version(
    proposal_lineage: Sequence[Mapping[str, Any]],
    bridge_result: StageEditPipelineBridgeResult,
) -> int:
    """Bind repaired lineage to the exact bridge model generation."""

    expected_version = bridge_result.preflight.dynamic_crop_version
    versions: set[int] = set()
    for item in proposal_lineage:
        if not isinstance(item, Mapping):
            raise RuntimeError(
                "accepted StageEdit repair lineage contains a malformed event"
            )
        if "dynamic_crop_version" not in item:
            continue
        value = item["dynamic_crop_version"]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in (2, 3)
        ):
            raise RuntimeError(
                "accepted StageEdit repair lineage has an invalid Dynamic Crop version"
            )
        versions.add(value)
    if not versions:
        if expected_version == 2:
            # Pre-v3 v2 lineage did not always serialize its model generation.
            return 2
        raise RuntimeError(
            "accepted Dynamic Crop v3 repair lineage omitted its model version"
        )
    if versions != {expected_version}:
        raise RuntimeError(
            "accepted StageEdit repair lineage differs from the bridge Dynamic Crop version"
        )
    return expected_version


def _attempt_stage_edit_windowed_beam_reedit(
    selected_bundle: _GroupRenderCandidateBundle,
    selected_materialized: StageEditMaterializedVisualCandidate,
    original_review_payload: dict[str, Any] | None,
    *,
    bridge_result: StageEditPipelineBridgeResult,
    learned_repair_issues: Sequence[Mapping[str, Any]],
    request: AnalysisRequest,
    eligible_indices: Sequence[int],
    audio_offsets: Sequence[float],
    sync_scores: Sequence[float],
    scene_cut_times: Sequence[Sequence[float]],
    source_timewarps: Sequence[TimeWarpMap | None],
    source_aspect_ratios: Sequence[float],
    source_durations_seconds: Sequence[float],
    beat_period_seconds: float | None,
    phrase_times: Sequence[float],
    section_plan: SectionPlan,
    common_master_start: float,
    width: int,
    height: int,
    scratch_directory: Path,
    source_look_match: SourceLookMatchResult | None = None,
    boundary_look_measurement_cache: BoundaryLookMeasurementCache | None = None,
    professional_reference_profile: ProfessionalReferenceProfile | None = None,
    professional_reference_cadence_profile: (
        ProfessionalReferenceCadenceProfile | None
    ) = None,
    learned_reviewer_session: LearnedReviewerProductionSession | None = None,
    motion_authority_v3: DynamicCropMotionAuthorityV3 | None = None,
) -> _ProxyTriggeredPartialReeditResult:
    """Run actual local semi-Markov re-decode and proxy-score every proposal."""

    if not isinstance(selected_materialized, StageEditMaterializedVisualCandidate):
        raise TypeError("StageEdit beam repair needs a StageEdit visual candidate")
    base_retained = selected_materialized.retained_candidate
    base_edit = base_retained.edit
    deterministic_indices: tuple[int, ...] = ()
    ignored_deterministic_indices: list[Any] = []
    if original_review_payload is not None:
        raw = original_review_payload.get(
            "partial_reedit_boundary_issue_indices",
            (),
        )
        if isinstance(raw, list):
            deterministic_indices = tuple(
                sorted(
                    {
                        value
                        for value in raw
                        if isinstance(value, int)
                        and not isinstance(value, bool)
                        and 0 <= value < len(base_edit.boundaries)
                    }
                )
            )
            ignored_deterministic_indices = [
                value
                for value in raw
                if not (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and 0 <= value < len(base_edit.boundaries)
                )
            ]
    (
        perceptual_indices,
        perceptual_admission_audit,
    ) = _stage_edit_perceptual_repair_targets(
        base_edit,
        original_review_payload,
        review_render_plan=selected_bundle.render_plan,
    )
    issues_by_id: dict[str, StageEditRepairIssue] = {}
    for raw_issue in learned_repair_issues:
        issue = StageEditRepairIssue.from_mapping(raw_issue)
        issues_by_id[issue.issue_id] = issue
    for issue in _stage_edit_boundary_issues(
        base_edit,
        deterministic_indices,
        prefix="decoded-proxy-initial",
        causes=("decoded_proxy_visible_boundary_failure",),
    ):
        issues_by_id.setdefault(issue.issue_id, issue)
    for issue in _stage_edit_boundary_issues(
        base_edit,
        perceptual_indices,
        prefix="decoded-proxy-perceptual-source-admission-initial",
        causes=("decoded_proxy_perceptual_source_admission_failure",),
    ):
        issues_by_id.setdefault(issue.issue_id, issue)
    reviewer_issues = tuple(issues_by_id.values())
    audit: dict[str, Any] = {
        "analysis_basis": STAGE_EDIT_WINDOWED_BEAM_CONTRACT,
        "attempted": bool(reviewer_issues),
        "actual_local_semimarkov_redecode": bool(reviewer_issues),
        "heuristic_timing_repair_used": False,
        "selected_version": "original",
        "reason": None,
        "reviewer_issue_windows": [
            issue.to_dict() for issue in reviewer_issues
        ],
        "issue_index_source_fields": [
            "partial_reedit_boundary_issue_indices",
            _STAGE_EDIT_PERCEPTUAL_REPAIR_INDEX_SOURCE_FIELD,
            "learned_repair_issues",
        ],
        "deterministic_visible_boundary_issue_indices": list(
            deterministic_indices
        ),
        "ignored_deterministic_visible_boundary_issue_indices": (
            ignored_deterministic_indices
        ),
        "perceptual_source_admission": perceptual_admission_audit,
        "perceptual_repair_boundary_indices": list(perceptual_indices),
        "learned_repair_issue_ids": [
            StageEditRepairIssue.from_mapping(value).issue_id
            for value in learned_repair_issues
        ],
        "before_review": deepcopy(original_review_payload),
        "after_review": None,
        "candidate_origin": selected_bundle.candidate_origin,
        "candidate_lineage_before": [
            dict(item) for item in selected_bundle.candidate_lineage
        ],
    }
    if not reviewer_issues:
        audit["reason"] = "no_reviewer_issue_windows"
        return _ProxyTriggeredPartialReeditResult(
            selected_bundle,
            original_review_payload,
            audit,
        )

    local_sources = tuple(request.sources[index] for index in eligible_indices)
    redecoder = StageEditWindowedBeamRedecoder(
        bridge_result=bridge_result,
        sources=local_sources,
        source_durations_seconds=tuple(source_durations_seconds),
        source_start_offsets_seconds=tuple(audio_offsets),
        internal_cut_times_seconds=tuple(
            tuple(value) for value in scene_cut_times
        ),
        source_timewarps=tuple(source_timewarps),
        source_quality_log_priors=(0.0,) * len(local_sources),
        output=Path(scratch_directory) / ".stage-edit-window-redecode.mp4",
    )

    cache: dict[
        str,
        tuple[
            _GroupRenderCandidateBundle | None,
            dict[str, Any] | None,
            StageEditRepairReview,
        ],
    ] = {}
    original_signature = _stage_edit_render_signature(base_edit)
    original_repair_review = _stage_edit_proxy_repair_review(
        base_edit,
        original_review_payload,
        require_perceptual_source_admission=bool(
            perceptual_admission_audit["available"]
        ),
        review_render_plan=selected_bundle.render_plan,
    )
    cache[original_signature] = (
        selected_bundle,
        deepcopy(original_review_payload),
        original_repair_review,
    )

    def review_scorer(edit: StageEdit) -> StageEditRepairReview:
        signature = _stage_edit_render_signature(edit)
        cached = cache.get(signature)
        if cached is not None:
            return cached[2]
        retained = RetainedStageEditCandidate(
            rank=base_retained.rank,
            score=base_retained.score,
            edit=edit,
            exact_clock_reprojected=True,
            timewarped_segment_count=sum(
                value is not None and not value.scalar_fallback
                for value in source_timewarps
            ),
        )
        try:
            materialized = materialize_stage_edit_visual_candidate(
                retained,
                candidate_index=selected_bundle.candidate_index,
                review_rank=selected_bundle.review_rank,
                sources=local_sources,
                audio_offsets=audio_offsets,
                sync_scores=sync_scores,
                scene_cut_times=scene_cut_times,
                source_timewarps=source_timewarps,
                feature_cache_directory=request.source_cache_directory,
            )
            bundle = _derive_complete_group_candidate(
                materialized,
                request=request,
                eligible_indices=eligible_indices,
                audio_offsets=audio_offsets,
                scene_cut_times=scene_cut_times,
                source_timewarps=source_timewarps,
                source_aspect_ratios=source_aspect_ratios,
                source_durations_seconds=source_durations_seconds,
                beat_period_seconds=beat_period_seconds,
                phrase_times=phrase_times,
                section_plan=section_plan,
                common_master_start=common_master_start,
                width=width,
                height=height,
                output=(
                    Path(scratch_directory)
                    / (
                        ".stage-edit-window-proposal-"
                        f"{len(cache):02d}.mp4"
                    )
                ),
                source_look_match=source_look_match,
                boundary_look_measurement_cache=(
                    boundary_look_measurement_cache
                ),
            )
            bundle = _apply_stage_edit_candidate_render_contract(
                bundle,
                materialized,
                motion_authority_v3=motion_authority_v3,
            )
            if bundle.timeline_review is None or not bundle.timeline_review.passed:
                raise StageEditVisualCandidateError(
                    "windowed beam proposal failed deterministic timeline review"
                )
        except (StageEditVisualCandidateError, TypeError, ValueError, cv2.error) as exc:
            failed = StageEditRepairReview(
                decoded_objective=-1_000_000.0,
                professional_objective=-1_000_000.0,
                cadence_objective=-1_000_000.0,
                technical_errors=(
                    f"visual_or_render_derivation_failed:{type(exc).__name__}",
                ),
                audit={"error": f"{type(exc).__name__}: {exc}"},
            )
            cache[signature] = (None, None, failed)
            return failed

        render_review = render_and_review_proxy
        if professional_reference_profile is not None:
            def render_review(
                plan: RenderPlan,
                proxy_output: Path,
            ) -> Any:
                return render_and_review_proxy(
                    plan,
                    proxy_output,
                    professional_reference_profile=(
                        professional_reference_profile
                    ),
                )
        source_switches_per_minute = float(
            getattr(
                bundle.timeline_review,
                "source_switches_per_minute",
                sum(
                    left.source_index != right.source_index
                    for left, right in zip(
                        bundle.render_plan.segments,
                        bundle.render_plan.segments[1:],
                    )
                )
                * 60.0
                / max(bundle.render_plan.duration, 1e-9),
            )
        )
        selection = review_rendered_candidates(
            (
                RenderedCandidate(
                    candidate_index=bundle.candidate_index,
                    render_plan=bundle.render_plan,
                    pre_render_score=bundle.pre_render_score,
                    payload=bundle,
                    rank_metadata=RenderedCandidateRankMetadata(
                        source_switches_per_minute=source_switches_per_minute
                    ),
                ),
            ),
            scratch_directory=Path(scratch_directory),
            maximum_review_count=1,
            minimum_required_candidate_count=1,
            render_review=render_review,
            professional_reference_profile=professional_reference_profile,
            professional_reference_cadence_profile=(
                professional_reference_cadence_profile
            ),
        )
        payload = deepcopy(selection.selected.review_payload)
        repair_review = _stage_edit_proxy_repair_review(
            edit,
            payload,
            require_perceptual_source_admission=bool(
                perceptual_admission_audit["available"]
            ),
            review_render_plan=bundle.render_plan,
        )
        if learned_reviewer_session is not None and payload is not None:
            learned_review = learned_reviewer_session.review_candidate(
                candidate_index=bundle.candidate_index,
                render_plan=bundle.render_plan,
                decoded_review=payload,
                source_switches_per_minute=source_switches_per_minute,
            )
            repair_review = (
                augment_stage_edit_repair_review_with_learned_issues(
                    repair_review,
                    learned_review,
                    session_provenance=(
                        learned_reviewer_session.provenance_dict()
                    ),
                )
            )
        cache[signature] = (bundle, payload, repair_review)
        return repair_review

    repair_result = repair_stage_edit_candidates(
        candidates=(base_edit,),
        reviewer_issues=reviewer_issues,
        canonical_shot_boundaries_seconds=(
            redecoder.canonical_boundaries_seconds
        ),
        beam_redecoder=redecoder,
        review_scorer=review_scorer,
        config=StageEditRepairConfig(
            context_seconds=3.0,
            maximum_iterations=4,
            maximum_consecutive_no_improvement=1,
            maximum_proposals_per_iteration=8,
        ),
    )
    audit["repair_loop"] = repair_result.audit.to_dict()
    audit["windowed_beam_redecode"] = redecoder.audit_payload()
    source_substitution_evidence = (
        _stage_edit_targeted_source_substitution_evidence(
            base_edit,
            repair_result.edit,
            perceptual_indices,
        )
    )
    audit["perceptual_targeted_source_substitution"] = (
        source_substitution_evidence
    )
    audit["repair_loop_changed"] = repair_result.changed
    audit["changed"] = bool(
        repair_result.changed
        and source_substitution_evidence["satisfied"]
    )
    if not repair_result.changed:
        audit["reason"] = repair_result.audit.termination_reason
        return _ProxyTriggeredPartialReeditResult(
            selected_bundle,
            original_review_payload,
            audit,
        )
    if not source_substitution_evidence["satisfied"]:
        audit["reason"] = (
            "perceptual_source_repair_requires_targeted_source_substitution"
        )
        audit["rejected_repair_edit_sha256"] = (
            repair_result.audit.final_edit_sha256
        )
        return _ProxyTriggeredPartialReeditResult(
            selected_bundle,
            original_review_payload,
            audit,
        )

    final_signature = _stage_edit_render_signature(repair_result.edit)
    final_cached = cache.get(final_signature)
    if final_cached is None or final_cached[0] is None:
        raise RuntimeError(
            "committed StageEdit beam repair lacks its revalidated render bundle"
        )
    final_bundle, final_payload, final_review = final_cached
    assert final_bundle is not None
    proposal_lineage = tuple(
        dict(item) for item in final_bundle.candidate_lineage
    )
    (
        superseded_base_lineage,
        superseded_exact_clock_hashes,
    ) = _strip_superseded_exact_stage_edit_shot_clocks(
        selected_bundle.candidate_lineage
    )
    expected_final_exact_clock = _stage_edit_exact_shot_clock_from_edit(
        repair_result.edit
    )
    actual_final_exact_clock = getattr(
        final_bundle, "stage_edit_exact_shot_clock", None
    )
    if actual_final_exact_clock != expected_final_exact_clock:
        raise RuntimeError(
            "accepted windowed StageEdit repair exact shot clock is stale or "
            "does not match the final repaired source path"
        )
    committed_continuity_lineage = _committed_window_continuity_lineage(
        repair_result,
        redecoder,
    )
    proposal_dynamic_crop_version = _committed_repair_dynamic_crop_version(
        proposal_lineage,
        bridge_result,
    )
    suppressed_dynamic_crop_indices = tuple(
        sorted(
            {
                value
                for item in proposal_lineage
                for raw_values in (
                    item.get(
                        "dynamic_crop_suppressed_segment_indices",
                        item.get(
                            "dynamic_crop_v2_suppressed_segment_indices",
                            (),
                        ),
                    ),
                )
                for value in (
                    raw_values
                    if isinstance(raw_values, (list, tuple))
                    else ()
                )
                if isinstance(value, int) and not isinstance(value, bool)
            }
        )
    )
    learned_reviewer_reinvoked = bool(
        learned_reviewer_session is not None
        and "learned_reviewer_issue_suggestions" in final_review.audit
    )
    raw_final_perceptual_admission = final_review.audit.get(
        "perceptual_source_admission"
    )
    final_perceptual_admission = (
        raw_final_perceptual_admission
        if isinstance(raw_final_perceptual_admission, Mapping)
        else {}
    )
    source_sequence_changed = bool(
        source_substitution_evidence["global_source_sequence_changed"]
    )
    selected_redecode_evidence = {
        "actual_semimarkov_redecode": True,
        "new_proxy_review_available": final_payload is not None,
        "new_proxy_passed": bool(
            final_payload is not None and final_payload.get("passed")
        ),
        "learned_reviewer_configured": learned_reviewer_session is not None,
        "learned_reviewer_reinvoked_after_splice": (
            learned_reviewer_reinvoked
        ),
        "perceptual_source_admission_required": bool(
            perceptual_admission_audit["available"]
        ),
        "new_proxy_perceptual_source_admission_passed": (
            final_perceptual_admission.get("admission_passed") is True
            if perceptual_admission_audit["available"]
            else None
        ),
        "new_proxy_penalized_weighted_score": (
            final_perceptual_admission.get("penalized_weighted_score")
            if perceptual_admission_audit["available"]
            else None
        ),
        "source_sequence_changed_by_redecode": source_sequence_changed,
        "perceptual_targeted_source_substitution_required": (
            source_substitution_evidence["required"]
        ),
        "perceptual_targeted_source_substitution_satisfied": (
            source_substitution_evidence["satisfied"]
        ),
        "perceptual_substituted_target_boundary_indices": (
            source_substitution_evidence[
                "substituted_target_boundary_indices"
            ]
        ),
        "dynamic_crop_version": proposal_dynamic_crop_version,
        "dynamic_crop_suppressed_segment_indices": list(
            suppressed_dynamic_crop_indices
        ),
        "proposal_materialization_lineage_preserved": True,
        "final_authoritative_exact_shot_clock_count": 1,
        "superseded_exact_shot_clock_sha256": list(
            superseded_exact_clock_hashes
        ),
        **committed_continuity_lineage,
    }
    if proposal_dynamic_crop_version == 2:
        selected_redecode_evidence[
            "dynamic_crop_v2_suppressed_segment_indices"
        ] = list(suppressed_dynamic_crop_indices)
    lineage_event = {
        "stage": "reviewer_issue_window_actual_semimarkov_redecode",
        "contract": STAGE_EDIT_WINDOWED_BEAM_CONTRACT,
        "repair_loop_input_sha256": repair_result.audit.input_sha256,
        "initial_edit_sha256": repair_result.audit.initial_edit_sha256,
        "final_edit_sha256": repair_result.audit.final_edit_sha256,
        "committed_iteration_count": (
            repair_result.audit.committed_iteration_count
        ),
        "actual_public_decoder_invoked": True,
        "decoded_proxy_revalidated_after_splice": final_payload is not None,
        "heuristic_timing_repair_used_as_reviewer_repair": False,
        "proposal_materialization_lineage_event_count": len(proposal_lineage),
        **selected_redecode_evidence,
    }
    final_bundle = replace(
        final_bundle,
        candidate_origin=selected_bundle.candidate_origin,
        candidate_lineage=(
            superseded_base_lineage
            + (
                {
                    "stage": "windowed_semimarkov_proposal_lineage",
                    "event_count": len(proposal_lineage),
                },
            )
            + proposal_lineage
            + (lineage_event,)
        ),
    )
    audit["selected_version"] = "windowed_semimarkov_redecode"
    audit["reason"] = "strict_objectives_improved_and_proxy_revalidated"
    audit["after_review"] = deepcopy(final_payload)
    audit["selected_redecode_evidence"] = selected_redecode_evidence
    audit["candidate_lineage_after"] = [
        dict(item) for item in final_bundle.candidate_lineage
    ]
    return _ProxyTriggeredPartialReeditResult(
        final_bundle,
        final_payload,
        audit,
    )


def _normalized_song_identity(value: str) -> str:
    return "".join(
        character
        for character in str(value).casefold()
        if character.isalnum()
    )


def _load_professional_reference_profile(
    request: AnalysisRequest,
) -> ProfessionalReferenceProfile | None:
    """Strict-load the request profile once for every decoded proxy review."""

    profile_path = request.professional_reference_profile
    if profile_path is None:
        return None
    profile = ProfessionalReferenceProfile.from_file(profile_path)
    if _normalized_song_identity(profile.song) != _normalized_song_identity(
        request.song
    ):
        raise ValueError(
            "professional-reference profile song does not match request song: "
            f"{profile.song!r} != {request.song!r}"
        )
    return profile


def _load_professional_reference_cadence_profile(
    request: AnalysisRequest,
    quality_profile: ProfessionalReferenceProfile | None,
) -> ProfessionalReferenceCadenceProfile | None:
    """Strict-load and bind cadence evidence to this exact production request."""

    profile_path = request.professional_reference_cadence_profile
    if profile_path is None:
        return None
    if quality_profile is None:
        raise ValueError(
            "professional-reference cadence profile requires the matching "
            "professional-reference quality profile"
        )
    profile = ProfessionalReferenceCadenceProfile.from_file(profile_path)
    if _normalized_song_identity(profile.artist) != _normalized_song_identity(
        quality_profile.artist
    ):
        raise ValueError(
            "professional-reference cadence artist does not match quality "
            f"profile: {profile.artist!r} != {quality_profile.artist!r}"
        )
    if _normalized_song_identity(profile.song) != _normalized_song_identity(
        request.song
    ) or _normalized_song_identity(profile.song) != _normalized_song_identity(
        quality_profile.song
    ):
        raise ValueError(
            "professional-reference cadence song does not match request and "
            f"quality profile: {profile.song!r}"
        )
    quality_references = {
        (
            str(reference.path.expanduser().resolve()),
            reference.sha256,
            str(reference.audit_path.expanduser().resolve()),
            reference.audit_sha256,
        )
        for reference in quality_profile.references
    }
    cadence_references = {
        (
            str(item["media_path"]),
            str(item["media_sha256"]),
            str(item["audit_path"]),
            str(item["audit_sha256"]),
        )
        for item in profile.reference_provenance()
    }
    if cadence_references != quality_references:
        raise ValueError(
            "professional-reference cadence provenance/hashes do not match "
            "the quality profile"
        )
    if not request.sources:
        raise ValueError(
            "professional-reference cadence profile needs materialized sources"
        )
    request_master = request.sources[request.reference_source].resolve()
    if profile.master_media_path != request_master:
        raise ValueError(
            "professional-reference cadence master media does not match "
            "request reference source"
        )
    return profile


def _load_learned_reviewer_session(
    request: AnalysisRequest,
) -> LearnedReviewerProductionSession | None:
    """Strict-load the explicit reviewer bundle before expensive media work."""

    checkpoint = request.stage_edit_learned_reviewer_checkpoint_directory
    report = request.stage_edit_learned_reviewer_production_training_report
    if checkpoint is None:
        if report is not None:
            raise RuntimeError(
                "learned reviewer report cannot be supplied without checkpoint"
            )
        return None
    if report is None:
        raise RuntimeError(
            "learned reviewer checkpoint requires its exact production report"
        )
    config = LearnedReviewerProductionReviewConfig(
        checkpoint_directory=checkpoint,
        production_training_report=report,
        song_id=request.song,
    )
    return LearnedReviewerProductionSession.load(config)


def _load_stage_edit_generation_preflight(
    request: AnalysisRequest,
) -> StageEditPipelineBridgePreflight | None:
    """Strict-load explicit learned generation before expensive media work."""

    checkpoint = request.stage_edit_generation_checkpoint_directory
    snapshot = request.stage_edit_generation_feature_snapshot
    source_ids = request.stage_edit_generation_source_ids
    if checkpoint is None:
        if snapshot is not None or source_ids:
            raise RuntimeError(
                "StageEdit generation activation must provide checkpoint, "
                "feature snapshot, and source ids together"
            )
        return None
    if snapshot is None or not source_ids:
        raise RuntimeError(
            "StageEdit generation checkpoint requires feature snapshot and source ids"
        )
    readiness = stage_edit_readiness_report(checkpoint)
    dynamic_crop_version = getattr(readiness, "dynamic_crop_version", None)
    if (
        isinstance(dynamic_crop_version, bool)
        or not isinstance(dynamic_crop_version, int)
        or dynamic_crop_version not in (2, 3)
    ):
        raise RuntimeError(
            "explicit StageEdit generation checkpoint has an unsupported "
            "Dynamic Crop version"
        )
    # The three-field advanced request predates learned source-pair authority
    # and carries none of its immutable scorer identities.  It can honestly
    # route either checkpoint generation only under handcrafted continuity;
    # production learned continuity remains registry-only until the request
    # schema can bind the full scorer identity.
    return preflight_stage_edit_pipeline_bridge(
        feature_snapshot_path=snapshot,
        checkpoint_directory=checkpoint,
        source_ids=source_ids,
        edit_mode=request.edit_mode,
        target_member=request.target_member,
        dynamic_crop_version=dynamic_crop_version,
        continuity_authority=HANDCRAFTED_CONTINUITY_AUTHORITY,
    )


def _discover_automatic_stage_edit_production_bundle(
    request: AnalysisRequest,
) -> RegisteredStageEditProductionBundle | None:
    """Resolve machine-installed learned generation for the simple app path.

    Explicit advanced requests retain their exact supplied snapshot.  Otherwise
    only automatic GROUP mode is eligible for registry discovery.  Once an
    active record exists, registry/checkpoint failures propagate and cannot
    silently demote the request to the ordinary visual editor.
    """

    explicit_learned_fields = (
        "stage_edit_generation_checkpoint_directory",
        "stage_edit_checkpoint_directory",
        "stage_edit_learned_reviewer_checkpoint_directory",
    )
    if any(getattr(request, name, None) is not None for name in explicit_learned_fields):
        return None
    if (
        getattr(request, "source_mode", None) is not SourceMode.AUTOMATIC
        or getattr(request, "edit_mode", None) is not EditMode.GROUP
        or bool(getattr(request, "acquire_only", False))
    ):
        return None
    return discover_registered_stage_edit_production_bundle()


def _registered_dynamic_crop_motion_authority_v3(
    bundle: RegisteredStageEditProductionBundle | None,
) -> DynamicCropMotionAuthorityV3 | None:
    """Strictly replay the registry-issued authority before any v3 candidate."""

    if bundle is None:
        return None
    if bundle.dynamic_crop_version == 2:
        if bundle.motion_authority_identity_v3 is not None:
            raise RuntimeError(
                "registered Dynamic Crop v2 bundle unexpectedly carries v3 motion authority"
            )
        return None
    if (
        bundle.dynamic_crop_version != 3
        or not bundle.dynamic_crop_v3_motion_authority_qualified
        or bundle.motion_authority_identity_v3 is None
    ):
        raise RuntimeError(
            "registered Dynamic Crop v3 bundle lacks qualified release-freeze "
            "motion authority"
        )
    try:
        return load_dynamic_crop_motion_authority_from_registered_identity_v3(
            bundle.motion_authority_identity_v3
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "registered Dynamic Crop v3 motion authority failed strict replay"
        ) from exc


def _load_stage_edit_montage_policy(
    *,
    registered_bundle: RegisteredStageEditProductionBundle | None,
    explicit_preflight: StageEditPipelineBridgePreflight | None,
) -> ProfessionalMontagePolicy | None:
    """Load the immutable IVE visual grammar whenever StageEdit is active."""

    if registered_bundle is None and explicit_preflight is None:
        return None
    if registered_bundle is not None:
        profile_path = getattr(
            registered_bundle,
            "professional_montage_grammar_profile_path",
            None,
        )
        expected_sha256 = getattr(
            registered_bundle,
            "professional_montage_grammar_profile_file_sha256",
            None,
        )
        if not isinstance(profile_path, Path) or not isinstance(
            expected_sha256, str
        ):
            raise RuntimeError(
                "registered StageEdit bundle lacks immutable professional "
                "montage-grammar profile evidence"
            )
    else:
        profile_path = DEFAULT_IVE_PROFESSIONAL_MONTAGE_PROFILE_PATH
        expected_sha256 = None
    policy = load_professional_montage_policy(
        profile_path,
        expected_file_sha256=expected_sha256,
    )
    if registered_bundle is not None:
        expected_identity = getattr(
            registered_bundle,
            "professional_montage_grammar_profile_identity",
            None,
        )
        if callable(expected_identity):
            expected_identity = expected_identity()
        if not isinstance(expected_identity, Mapping):
            raise RuntimeError(
                "registered StageEdit montage profile identity is unavailable"
            )
        actual_without_path = {
            key: value
            for key, value in policy.profile_identity.items()
            if key != "path"
        }
        expected_without_path = {
            key: value
            for key, value in dict(expected_identity).items()
            if key != "path"
        }
        if actual_without_path != expected_without_path:
            raise RuntimeError(
                "registered StageEdit montage profile identity changed"
            )
    return policy


def _stage_edit_motion_cut_profile_manifest_payload(
    profile: ProfessionalMotionCutProfile,
    montage_policy: ProfessionalMontagePolicy,
) -> dict[str, Any]:
    """Bind motion calibration to its exact file and current montage source."""

    if not isinstance(profile, ProfessionalMotionCutProfile):
        raise TypeError("profile must be ProfessionalMotionCutProfile")
    if not isinstance(montage_policy, ProfessionalMontagePolicy):
        raise TypeError("montage_policy must be ProfessionalMontagePolicy")
    source_path = profile.source_path
    file_sha256 = profile.file_sha256
    if (
        not isinstance(source_path, Path)
        or source_path.resolve()
        != DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_PATH.resolve()
        or file_sha256
        != DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_FILE_SHA256
    ):
        raise RuntimeError(
            "StageEdit motion-cut calibration file identity changed"
        )
    profile_payload = profile.to_dict()
    provenance = profile_payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("StageEdit motion-cut calibration lacks provenance")
    source_montage_identity = provenance.get("source_montage_profile")
    current_montage_identity = montage_policy.profile_identity
    if (
        not isinstance(source_montage_identity, Mapping)
        or not isinstance(current_montage_identity, Mapping)
        or dict(source_montage_identity) != dict(current_montage_identity)
    ):
        raise RuntimeError(
            "StageEdit motion-cut calibration source montage identity differs "
            "from the active montage policy"
        )
    recommendation = profile.recommendation
    recommendation_payload = recommendation.to_dict()
    if (
        recommendation.hard_gate is not False
        or recommendation.minimum_required_switches_per_section is not None
        or recommendation.pose_phase_hard_threshold is not None
    ):
        raise RuntimeError(
            "StageEdit motion-cut calibration attempted to create a hard gate"
        )
    return {
        "path": str(source_path.resolve()),
        "file_sha256": file_sha256,
        "content_sha256": profile.profile_content_sha256,
        "source_montage_profile_identity": deepcopy(
            dict(source_montage_identity)
        ),
        "active_montage_policy_identity": deepcopy(
            dict(current_montage_identity)
        ),
        "source_identity_binding_passed": True,
        "recommendation": recommendation_payload,
        "runtime_binding": {
            # Retained for consumers of the v1 manifest; this is the motion
            # qualifier authority, not the decoder's primary boundary scorer.
            "continuity_authority": HANDCRAFTED_CONTINUITY_AUTHORITY,
            "continuity_role": (
                "independent_handcrafted_motion_eligibility_qualifier"
            ),
            "continuity_scorer_contract": (
                HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
            ),
            "continuity_scorer_fingerprint": (
                HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
            ),
            "independent_from_primary_boundary_continuity": True,
            "primary_boundary_score_reused": False,
            "both_motion_minimum_pose_motion_p10": (
                recommendation.both_moving_minimum_pose_motion
            ),
            "continuity_minimum_score_p10": (
                recommendation.continuity_compatible_minimum_score
            ),
            "same_time_pose_cosine_soft_bonus_qualifier_p50": (
                recommendation.pose_phase_compatible_preferred_cosine
            ),
            "soft_bonus_only": True,
            "hard_gate": False,
            "minimum_required_switches_per_section": None,
            "highlight_minimum_motion_closeup_switch_boundaries": 0,
            "direction_or_velocity_proven": False,
        },
        "direction_or_velocity_proven": False,
    }


def _load_stage_edit_motion_cut_profile(
    *,
    registered_bundle: RegisteredStageEditProductionBundle | None,
    explicit_preflight: StageEditPipelineBridgePreflight | None,
    montage_policy: ProfessionalMontagePolicy | None,
) -> ProfessionalMotionCutProfile | None:
    """Strict-load the fixed IVE motion calibration for every StageEdit path."""

    if registered_bundle is None and explicit_preflight is None:
        return None
    if montage_policy is None:
        raise RuntimeError(
            "StageEdit motion-cut calibration requires the active montage policy"
        )
    profile = load_professional_motion_cut_profile(
        DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_PATH,
        expected_file_sha256=(
            DEFAULT_IVE_PROFESSIONAL_MOTION_CUT_PROFILE_FILE_SHA256
        ),
        # The raw profile is fixed by SHA-256 and its source montage identity
        # is compared to the independently strict-loaded current policy below.
        verify_source_inputs=False,
    )
    _stage_edit_motion_cut_profile_manifest_payload(profile, montage_policy)
    return profile


def _report_stage_edit_motion_cut_profile(
    reporter: EventReporter,
    payload: Mapping[str, Any],
) -> None:
    runtime = payload.get("runtime_binding")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("StageEdit motion-cut reporter payload is incomplete")
    reporter.emit(
        "log",
        message=(
            "IVE 하이라이트 동작중 포즈 연결 soft qualifier 엄격 검증 완료 · "
            f"path {payload['path']} · file SHA-256 "
            f"{payload['file_sha256']} · content SHA-256 "
            f"{payload['content_sha256']} · P10 motion "
            f"{runtime['both_motion_minimum_pose_motion_p10']} · "
            f"P10 continuity {runtime['continuity_minimum_score_p10']} · "
            "P50 same-time pose cosine "
            f"{runtime['same_time_pose_cosine_soft_bonus_qualifier_p50']} · "
            "hard gate/quota 없음 · direction/velocity 증명 아님"
        ),
        professional_motion_cut_profile=deepcopy(dict(payload)),
    )


def _registered_stage_edit_bridge_kwargs(
    bundle: RegisteredStageEditProductionBundle,
) -> dict[str, Any]:
    """Resolve one registry version/continuity identity without fallback."""

    version = getattr(bundle, "dynamic_crop_version", 2)
    authority = getattr(bundle, "continuity_authority", None)
    if version == 2:
        if authority is not None:
            raise RuntimeError(
                "registered Dynamic Crop v2 bundle cannot declare a continuity authority"
            )
        return {
            "dynamic_crop_version": 2,
            "continuity_authority": HANDCRAFTED_CONTINUITY_AUTHORITY,
        }
    if version != 3 or authority not in {
        HANDCRAFTED_CONTINUITY_AUTHORITY,
        PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
        PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
    }:
        raise RuntimeError(
            "registered StageEdit bundle has an unsupported Dynamic Crop/continuity identity"
        )
    resolved: dict[str, Any] = {
        "dynamic_crop_version": 3,
        "continuity_authority": authority,
    }
    pair_fields = {
        "source_pair_scorer_path": getattr(
            bundle, "source_pair_scorer_path", None
        ),
        "source_pair_scorer_file_sha256": getattr(
            bundle, "source_pair_scorer_file_sha256", None
        ),
        "source_pair_scorer_fingerprint": getattr(
            bundle, "source_pair_scorer_fingerprint", None
        ),
        "source_pair_checkpoint_sha256": getattr(
            bundle, "source_pair_checkpoint_sha256", None
        ),
        "source_pair_corpus_sha256": getattr(
            bundle, "source_pair_corpus_sha256", None
        ),
    }
    contextual_fields = {
        "contextual_source_pair_bundle_directory": getattr(
            bundle, "contextual_source_pair_bundle_directory", None
        ),
        "contextual_source_pair_checkpoint_sha256": getattr(
            bundle, "contextual_source_pair_checkpoint_sha256", None
        ),
        "contextual_source_pair_checkpoint_file_sha256": getattr(
            bundle, "contextual_source_pair_checkpoint_file_sha256", None
        ),
        "contextual_source_pair_report_sha256": getattr(
            bundle, "contextual_source_pair_report_sha256", None
        ),
        "contextual_source_pair_report_file_sha256": getattr(
            bundle, "contextual_source_pair_report_file_sha256", None
        ),
        "contextual_source_pair_beam_qa_report_sha256": getattr(
            bundle, "contextual_source_pair_beam_qa_report_sha256", None
        ),
        "contextual_source_pair_corpus_sha256": getattr(
            bundle, "contextual_source_pair_corpus_sha256", None
        ),
        "contextual_source_pair_corpus_file_sha256": getattr(
            bundle, "contextual_source_pair_corpus_file_sha256", None
        ),
        "contextual_source_pair_token_schema_fingerprint": getattr(
            bundle, "contextual_source_pair_token_schema_fingerprint", None
        ),
        "contextual_source_pair_verified_beam_qa_cases": getattr(
            bundle, "contextual_source_pair_verified_beam_qa_cases", None
        )
        or None,
    }
    if authority == PRODUCTION_LEARNED_CONTINUITY_AUTHORITY:
        if any(value is None for value in pair_fields.values()):
            raise RuntimeError(
                "registered learned continuity authority lacks an exact "
                "source-pair scorer identity"
            )
        if any(value is not None for value in contextual_fields.values()):
            raise RuntimeError(
                "registered learned-v1 continuity authority unexpectedly binds "
                "contextual-v2 evidence"
            )
        resolved.update(pair_fields)
    elif authority == PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2:
        if any(value is not None for value in pair_fields.values()):
            raise RuntimeError(
                "registered contextual-v2 continuity authority unexpectedly "
                "binds learned-v1 evidence"
            )
        if any(value is None for value in contextual_fields.values()):
            raise RuntimeError(
                "registered contextual-v2 continuity authority lacks its exact "
                "bundle and Beam-QA identity"
            )
        cases = tuple(
            contextual_fields[
                "contextual_source_pair_verified_beam_qa_cases"
            ]
        )
        from .stage_edit_contextual_beam_qa_adapter_v2 import (
            is_verified_stage_edit_contextual_beam_qa_case_v2,
        )

        if not cases or any(
            not is_verified_stage_edit_contextual_beam_qa_case_v2(value)
            for value in cases
        ):
            raise RuntimeError(
                "registered contextual-v2 continuity authority requires "
                "adapter-issued verified Beam-QA envelopes"
            )
        contextual_fields[
            "contextual_source_pair_verified_beam_qa_cases"
        ] = cases
        resolved.update(contextual_fields)
    elif any(value is not None for value in pair_fields.values()) or any(
        value is not None for value in contextual_fields.values()
    ):
        raise RuntimeError(
            "registered handcrafted continuity authority unexpectedly binds "
            "source-pair scorer evidence"
        )
    return resolved


def _stage_edit_continuity_config_identity(
    *,
    registered_bundle: RegisteredStageEditProductionBundle | None,
    explicit_preflight: StageEditPipelineBridgePreflight | None,
) -> tuple[str, str, str]:
    if registered_bundle is not None:
        activation = _registered_stage_edit_bridge_kwargs(registered_bundle)
        authority = activation["continuity_authority"]
        fingerprint = (
            activation["source_pair_scorer_fingerprint"]
            if authority == PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
            else getattr(
                registered_bundle,
                "contextual_source_pair_scorer_fingerprint",
                None,
            )
            if authority == PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2
            else HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
        )
    elif explicit_preflight is not None:
        authority = getattr(
            explicit_preflight,
            "continuity_authority",
            HANDCRAFTED_CONTINUITY_AUTHORITY,
        )
        observed_fingerprint = getattr(
            explicit_preflight,
            "source_pair_scorer_fingerprint",
            None,
        )
        fingerprint = (
            observed_fingerprint
            if observed_fingerprint is not None
            else HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
        )
    else:
        authority = HANDCRAFTED_CONTINUITY_AUTHORITY
        fingerprint = HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
    contract = (
        STAGE_EDIT_SOURCE_PAIR_SCORER_CONTRACT
        if authority == PRODUCTION_LEARNED_CONTINUITY_AUTHORITY
        else STAGE_EDIT_CONTEXTUAL_SOURCE_PAIR_V2_SCORER_CONTRACT
        if authority == PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2
        else HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
    )
    return authority, contract, fingerprint


def _professional_montage_policy_fingerprint(
    policy: ProfessionalMontagePolicy,
) -> str:
    """Hash the exact immutable policy payload used by both production gates."""

    if not isinstance(policy, ProfessionalMontagePolicy):
        raise TypeError("policy must be ProfessionalMontagePolicy")
    encoded = json.dumps(
        policy.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _close_burst_binding_from_policy(
    policy: ProfessionalMontagePolicy,
    *,
    policy_fingerprint_sha256: str,
) -> tuple[int, str]:
    """Resolve the sparse event P10s without inventing an occurrence quota."""

    expected = (
        (
            policy.minimum_close_burst_shot_count,
            CLOSE_BURST_SHOT_COUNT_STATISTIC_PATH,
        ),
        (
            policy.minimum_close_burst_duration_seconds,
            CLOSE_BURST_DURATION_STATISTIC_PATH,
        ),
        (
            policy.minimum_close_burst_source_switch_fraction,
            CLOSE_BURST_SOURCE_SWITCH_FRACTION_STATISTIC_PATH,
        ),
    )
    counts = {item.reference_count for item, _path in expected}
    if len(counts) != 1 or None in counts:
        raise RuntimeError(
            "professional montage close-burst event counts are incomplete"
        )
    for threshold, statistic_path in expected:
        provenance = threshold.provenance
        if not (
            provenance.profile_identity == policy.identity
            and provenance.statistic_path == statistic_path
            and provenance.statistic == "observed_burst_event_distribution_p10"
            and math.isclose(
                provenance.quantile,
                REFERENCE_FLOOR_QUANTILE,
                abs_tol=1.0e-12,
            )
            and provenance.reference_floor_basis is None
        ):
            raise RuntimeError(
                "professional montage close-burst provenance is not exact"
            )
    reference_count = int(next(iter(counts)))
    binding = close_burst_policy_binding_sha256(
        profile_identity=policy.identity,
        policy_fingerprint_sha256=policy_fingerprint_sha256,
        minimum_shot_count=policy.minimum_close_burst_shot_count.value,
        minimum_duration_seconds=(
            policy.minimum_close_burst_duration_seconds.value
        ),
        minimum_source_switch_fraction=(
            policy.minimum_close_burst_source_switch_fraction.value
        ),
        event_reference_count=reference_count,
    )
    return reference_count, binding


def _stage_edit_semimarkov_config_from_montage_policy(
    policy: ProfessionalMontagePolicy,
    *,
    motion_cut_profile: ProfessionalMotionCutProfile | None = None,
    require_motion_cut_calibration: bool = False,
    minimum_unique_sources: int = 3,
    maximum_source_occupancy_fraction: float = 0.60,
    continuity_authority: str = HANDCRAFTED_CONTINUITY_AUTHORITY,
    continuity_scorer_contract: str = HANDCRAFTED_CONTINUITY_SCORER_CONTRACT,
    continuity_scorer_fingerprint: str = (
        HANDCRAFTED_CONTINUITY_SCORER_FINGERPRINT
    ),
) -> StageEditSemiMarkovConfig:
    if continuity_authority not in {
        HANDCRAFTED_CONTINUITY_AUTHORITY,
        PRODUCTION_LEARNED_CONTINUITY_AUTHORITY,
        PRODUCTION_CONTEXTUAL_CONTINUITY_AUTHORITY_V2,
    }:
        raise RuntimeError(
            "StageEdit primary boundary-continuity authority is unsupported"
        )
    if require_motion_cut_calibration and motion_cut_profile is None:
        raise RuntimeError(
            "active StageEdit path is missing its professional motion-cut "
            "calibration"
        )
    motion_calibration = None
    motion_recommendation = None
    if motion_cut_profile is not None:
        _stage_edit_motion_cut_profile_manifest_payload(
            motion_cut_profile,
            policy,
        )
        motion_calibration = motion_cut_profile.to_stage_edit_calibration()
        motion_recommendation = motion_cut_profile.recommendation
        if not (
            motion_calibration.continuity_authority
            == HANDCRAFTED_CONTINUITY_AUTHORITY
            and motion_calibration.continuity_scorer_contract
            == HANDCRAFTED_CONTINUITY_SCORER_CONTRACT
        ):
            raise RuntimeError(
                "StageEdit motion-cut profile must bind the independent "
                "handcrafted continuity qualifier"
            )
    product = policy.product_editorial_requirements
    identity = policy.identity
    policy_fingerprint = _professional_montage_policy_fingerprint(policy)
    close_burst_reference_count, close_burst_binding = (
        _close_burst_binding_from_policy(
            policy,
            policy_fingerprint_sha256=policy_fingerprint,
        )
    )
    return StageEditSemiMarkovConfig(
        minimum_unique_sources=minimum_unique_sources,
        maximum_source_occupancy_fraction=maximum_source_occupancy_fraction,
        transition_authority=RENDERER_OWNED_TRANSITION_AUTHORITY,
        transition_score_weight=0.0,
        highlight_minimum_closeup_fraction=(
            policy.minimum_close_duration_fraction.value
        ),
        highlight_minimum_nonwide_fraction=(
            policy.minimum_nonwide_duration_fraction.value
        ),
        highlight_pooled_product_minimum_closeup_fraction=(
            policy.highlight_pooled_product_minimum_close_duration_fraction.value
        ),
        highlight_pooled_product_minimum_nonwide_fraction=(
            policy.highlight_pooled_product_minimum_nonwide_duration_fraction.value
        ),
        highlight_preferred_closeup_fraction=(
            policy.preferred_close_duration_fraction.value
        ),
        highlight_preferred_nonwide_fraction=(
            policy.preferred_nonwide_duration_fraction.value
        ),
        highlight_maximum_consecutive_wide_run_seconds=(
            policy.maximum_consecutive_wide_run_seconds.value
        ),
        highlight_preferred_maximum_consecutive_wide_run_seconds=(
            policy.preferred_maximum_consecutive_wide_run_seconds.value
        ),
        highlight_policy_profile_analysis_basis=identity.analysis_basis,
        highlight_policy_profile_schema_version=identity.schema_version,
        highlight_policy_profile_path=identity.path,
        highlight_policy_profile_file_sha256=identity.file_sha256,
        highlight_policy_profile_content_sha256=identity.content_sha256,
        highlight_policy_fingerprint_sha256=policy_fingerprint,
        highlight_minimum_close_burst_shot_count=(
            policy.minimum_close_burst_shot_count.value
        ),
        highlight_minimum_close_burst_duration_seconds=(
            policy.minimum_close_burst_duration_seconds.value
        ),
        highlight_minimum_close_burst_source_switch_fraction=(
            policy.minimum_close_burst_source_switch_fraction.value
        ),
        highlight_close_burst_event_reference_count=(
            close_burst_reference_count
        ),
        highlight_close_burst_policy_binding_sha256=close_burst_binding,
        highlight_motion_calibration=motion_calibration,
        highlight_motion_boundary_minimum_pose_motion=(
            motion_recommendation.both_moving_minimum_pose_motion
            if motion_recommendation is not None
            else None
        ),
        highlight_motion_boundary_minimum_continuity_score=(
            motion_recommendation.continuity_compatible_minimum_score
            if motion_recommendation is not None
            else None
        ),
        highlight_motion_boundary_minimum_pose_phase_cosine=(
            motion_recommendation.pose_phase_compatible_preferred_cosine
            if motion_recommendation is not None
            else None
        ),
        highlight_motion_boundary_minimum_sharpness=None,
        highlight_minimum_motion_closeup_switch_boundaries=0,
        highlight_minimum_wide_involved_source_switch_fraction=(
            policy.minimum_wide_involved_source_switch_fraction.value
        ),
        highlight_preferred_wide_involved_source_switch_fraction=(
            policy.preferred_wide_involved_source_switch_fraction.value
        ),
        highlight_wide_involved_source_switch_reference_count=(
            policy.preferred_wide_involved_source_switch_fraction.reference_count
        ),
        highlight_product_minimum_source_switches_per_minute=(
            policy.highlight_product_minimum_source_switches_per_minute.value
        ),
        highlight_preferred_source_switches_per_minute=(
            policy.highlight_preferred_source_switches_per_minute.value
        ),
        highlight_cadence_reference_count=(
            policy.highlight_cadence_reference_count
        ),
        highlight_minimum_early_wide_anchor_fraction=(
            policy.minimum_early_wide_anchor_fraction.value
        ),
        highlight_minimum_phrase_entry_wide_anchor_fraction=(
            policy.minimum_phrase_entry_wide_anchor_fraction.value
        ),
        highlight_minimum_nonwide_burst_boundaries=(
            product.minimum_nonwide_multi_family_burst_count
        ),
        minimum_formation_anchor_person_count_normalized=(
            product.formation_evidence_threshold_normalized
        ),
        highlight_minimum_post_burst_formation_resets=(
            product.minimum_post_burst_formation_evidenced_wide_reset_count
        ),
        continuity_authority=continuity_authority,
        continuity_scorer_contract=continuity_scorer_contract,
        continuity_scorer_fingerprint=continuity_scorer_fingerprint,
    )


def _rendered_montage_config_from_policy(
    policy: ProfessionalMontagePolicy,
    *,
    require_complete_role_duration_policy: bool = False,
) -> RenderedMontageGrammarConfig:
    product = policy.product_editorial_requirements
    identity = policy.identity
    policy_fingerprint = _professional_montage_policy_fingerprint(policy)
    close_burst_reference_count, close_burst_binding = (
        _close_burst_binding_from_policy(
            policy,
            policy_fingerprint_sha256=policy_fingerprint,
        )
    )
    role_policies = {
        item.role: item for item in policy.role_conditioned_duration_policies
    }
    if len(role_policies) != len(policy.role_conditioned_duration_policies):
        raise RuntimeError(
            "professional montage policy has duplicate role-duration evidence"
        )
    expected_roles = {"build", "chorus_like", "bridge_like"}
    unexpected_roles = set(role_policies) - expected_roles
    if unexpected_roles:
        raise RuntimeError(
            "professional montage policy has unsupported role-duration evidence: "
            + ", ".join(sorted(unexpected_roles))
        )
    missing_roles = expected_roles - set(role_policies)
    if require_complete_role_duration_policy and missing_roles:
        raise RuntimeError(
            "production rendered montage policy requires complete build, "
            "chorus_like, and bridge_like duration evidence; missing: "
            + ", ".join(sorted(missing_roles))
        )

    default_config: RenderedMontageGrammarConfig | None = None

    def fallback_config_value(field: str) -> Any:
        nonlocal default_config
        if default_config is None:
            default_config = RenderedMontageGrammarConfig()
        return getattr(default_config, field)

    def required_role_value(
        role: str,
        attribute: str,
        fallback_config_field: str,
    ) -> float:
        role_policy = role_policies.get(role)
        if role_policy is None:
            # Partial profiles remain usable by policy-loader unit tests.  The
            # sole production call below opts into complete-role fail-closed
            # validation, so these constructor defaults cannot mask missing
            # build or bridge evidence in an automatic StageEdit render.
            return float(fallback_config_value(fallback_config_field))
        threshold = getattr(role_policy, attribute)
        if threshold is None:
            raise RuntimeError(
                f"professional montage role {role!r} lacks required "
                f"{attribute} evidence"
            )
        return float(threshold.value)

    def optional_role_value(
        role: str,
        attribute: str,
        fallback_config_field: str,
    ) -> float | None:
        role_policy = role_policies.get(role)
        if role_policy is None:
            fallback = fallback_config_value(fallback_config_field)
            return None if fallback is None else float(fallback)
        threshold = getattr(role_policy, attribute)
        # A hash-bound zero P10 is intentionally represented by None.  Do not
        # promote it to an unsupported positive local minimum.
        return None if threshold is None else float(threshold.value)

    def role_reference_count(
        role: str,
        fallback_config_field: str,
    ) -> int:
        role_policy = role_policies.get(role)
        if role_policy is None:
            return int(fallback_config_value(fallback_config_field))
        return int(role_policy.reference_count)

    return RenderedMontageGrammarConfig(
        minimum_closeup_fraction=required_role_value(
            "chorus_like",
            "local_minimum_close_duration_fraction",
            "minimum_closeup_fraction",
        ),
        minimum_nonwide_fraction=(
            required_role_value(
                "chorus_like",
                "local_minimum_nonwide_duration_fraction",
                "minimum_nonwide_fraction",
            )
        ),
        pooled_product_minimum_closeup_fraction=(
            policy.highlight_pooled_product_minimum_close_duration_fraction.value
        ),
        pooled_product_minimum_nonwide_fraction=(
            policy.highlight_pooled_product_minimum_nonwide_duration_fraction.value
        ),
        preferred_closeup_fraction=(
            policy.preferred_close_duration_fraction.value
        ),
        preferred_nonwide_fraction=(
            policy.preferred_nonwide_duration_fraction.value
        ),
        maximum_consecutive_wide_run_seconds=(
            required_role_value(
                "chorus_like",
                "maximum_consecutive_wide_run_seconds",
                "maximum_consecutive_wide_run_seconds",
            )
        ),
        preferred_maximum_consecutive_wide_run_seconds=(
            policy.preferred_maximum_consecutive_wide_run_seconds.value
        ),
        chorus_role_pooled_product_minimum_closeup_fraction=(
            required_role_value(
                "chorus_like",
                "pooled_product_minimum_close_duration_fraction",
                "chorus_role_pooled_product_minimum_closeup_fraction",
            )
        ),
        chorus_role_pooled_product_minimum_nonwide_fraction=(
            required_role_value(
                "chorus_like",
                "pooled_product_minimum_nonwide_duration_fraction",
                "chorus_role_pooled_product_minimum_nonwide_fraction",
            )
        ),
        chorus_role_preferred_closeup_fraction=required_role_value(
            "chorus_like",
            "preferred_close_duration_fraction",
            "chorus_role_preferred_closeup_fraction",
        ),
        chorus_role_preferred_nonwide_fraction=required_role_value(
            "chorus_like",
            "preferred_nonwide_duration_fraction",
            "chorus_role_preferred_nonwide_fraction",
        ),
        chorus_role_preferred_maximum_consecutive_wide_run_seconds=(
            required_role_value(
                "chorus_like",
                "preferred_maximum_consecutive_wide_run_seconds",
                "chorus_role_preferred_maximum_consecutive_wide_run_seconds",
            )
        ),
        build_minimum_closeup_fraction=optional_role_value(
            "build",
            "local_minimum_close_duration_fraction",
            "build_minimum_closeup_fraction",
        ),
        build_minimum_nonwide_fraction=optional_role_value(
            "build",
            "local_minimum_nonwide_duration_fraction",
            "build_minimum_nonwide_fraction",
        ),
        build_pooled_product_minimum_closeup_fraction=required_role_value(
            "build",
            "pooled_product_minimum_close_duration_fraction",
            "build_pooled_product_minimum_closeup_fraction",
        ),
        build_pooled_product_minimum_nonwide_fraction=required_role_value(
            "build",
            "pooled_product_minimum_nonwide_duration_fraction",
            "build_pooled_product_minimum_nonwide_fraction",
        ),
        build_preferred_closeup_fraction=required_role_value(
            "build",
            "preferred_close_duration_fraction",
            "build_preferred_closeup_fraction",
        ),
        build_preferred_nonwide_fraction=required_role_value(
            "build",
            "preferred_nonwide_duration_fraction",
            "build_preferred_nonwide_fraction",
        ),
        build_maximum_consecutive_wide_run_seconds=required_role_value(
            "build",
            "maximum_consecutive_wide_run_seconds",
            "build_maximum_consecutive_wide_run_seconds",
        ),
        build_preferred_maximum_consecutive_wide_run_seconds=(
            required_role_value(
                "build",
                "preferred_maximum_consecutive_wide_run_seconds",
                "build_preferred_maximum_consecutive_wide_run_seconds",
            )
        ),
        build_duration_reference_count=role_reference_count(
            "build",
            "build_duration_reference_count",
        ),
        bridge_minimum_closeup_fraction=optional_role_value(
            "bridge_like",
            "local_minimum_close_duration_fraction",
            "bridge_minimum_closeup_fraction",
        ),
        bridge_minimum_nonwide_fraction=optional_role_value(
            "bridge_like",
            "local_minimum_nonwide_duration_fraction",
            "bridge_minimum_nonwide_fraction",
        ),
        bridge_pooled_product_minimum_closeup_fraction=required_role_value(
            "bridge_like",
            "pooled_product_minimum_close_duration_fraction",
            "bridge_pooled_product_minimum_closeup_fraction",
        ),
        bridge_pooled_product_minimum_nonwide_fraction=required_role_value(
            "bridge_like",
            "pooled_product_minimum_nonwide_duration_fraction",
            "bridge_pooled_product_minimum_nonwide_fraction",
        ),
        bridge_preferred_closeup_fraction=required_role_value(
            "bridge_like",
            "preferred_close_duration_fraction",
            "bridge_preferred_closeup_fraction",
        ),
        bridge_preferred_nonwide_fraction=required_role_value(
            "bridge_like",
            "preferred_nonwide_duration_fraction",
            "bridge_preferred_nonwide_fraction",
        ),
        bridge_maximum_consecutive_wide_run_seconds=required_role_value(
            "bridge_like",
            "maximum_consecutive_wide_run_seconds",
            "bridge_maximum_consecutive_wide_run_seconds",
        ),
        bridge_preferred_maximum_consecutive_wide_run_seconds=(
            required_role_value(
                "bridge_like",
                "preferred_maximum_consecutive_wide_run_seconds",
                "bridge_preferred_maximum_consecutive_wide_run_seconds",
            )
        ),
        bridge_duration_reference_count=role_reference_count(
            "bridge_like",
            "bridge_duration_reference_count",
        ),
        highlight_product_minimum_source_switches_per_minute=(
            policy.highlight_product_minimum_source_switches_per_minute.value
        ),
        highlight_preferred_source_switches_per_minute=(
            policy.highlight_preferred_source_switches_per_minute.value
        ),
        highlight_cadence_reference_count=(
            policy.highlight_cadence_reference_count
        ),
        policy_profile_analysis_basis=identity.analysis_basis,
        policy_profile_schema_version=identity.schema_version,
        policy_profile_path=identity.path,
        policy_profile_file_sha256=identity.file_sha256,
        policy_profile_content_sha256=identity.content_sha256,
        policy_fingerprint_sha256=policy_fingerprint,
        minimum_close_burst_shot_count=(
            policy.minimum_close_burst_shot_count.value
        ),
        minimum_close_burst_duration_seconds=(
            policy.minimum_close_burst_duration_seconds.value
        ),
        minimum_close_burst_source_switch_fraction=(
            policy.minimum_close_burst_source_switch_fraction.value
        ),
        close_burst_event_reference_count=close_burst_reference_count,
        close_burst_policy_binding_sha256=close_burst_binding,
        minimum_wide_involved_source_switch_fraction=(
            policy.minimum_wide_involved_source_switch_fraction.value
        ),
        preferred_wide_involved_source_switch_fraction=(
            policy.preferred_wide_involved_source_switch_fraction.value
        ),
        wide_involved_source_switch_reference_count=(
            policy.preferred_wide_involved_source_switch_fraction.reference_count
        ),
        minimum_early_wide_anchor_fraction=(
            policy.minimum_early_wide_anchor_fraction.value
        ),
        minimum_phrase_entry_wide_anchor_fraction=(
            policy.minimum_phrase_entry_wide_anchor_fraction.value
        ),
        minimum_formation_anchor_person_count=1,
        minimum_nonwide_burst_boundaries=(
            product.minimum_nonwide_multi_family_burst_count
        ),
        minimum_post_burst_formation_reset_count=(
            product.minimum_post_burst_formation_evidenced_wide_reset_count
        ),
    )


def _materialize_registered_stage_edit_preflight(
    *,
    bundle: RegisteredStageEditProductionBundle,
    request: AnalysisRequest,
    alignment: AlignmentResult,
    alignment_eligibility: Sequence[Mapping[str, Any]],
    source_timewarps: Sequence[TimeWarpMap | None],
    source_scene_cut_times: Sequence[Sequence[float]],
    timing_draft: GroupDraftTimeline,
    rhythm: RhythmAnalysis,
    section_plan: SectionPlan,
) -> tuple[
    StageEditInferenceSnapshotResult,
    StageEditPipelineBridgePreflight,
    IVECausalSensingPipelineAuditArtifact,
]:
    """Bind one verified installed model to this exact synchronized edit."""

    inference_artifact_root = (
        request.source_cache_directory or request.output_directory
    ) / "derived" / "stage-edit-inference-v1"
    alignment_hard_gates = tuple(
        bool(item["render_eligible"]) for item in alignment_eligibility
    )
    causal_sensing = materialize_ive_causal_sensing_for_stage_edit_pipeline(
        song=request.song,
        edit_mode=request.edit_mode,
        target_member=request.target_member,
        sources=request.sources,
        alignment=alignment,
        alignment_hard_gates=alignment_hard_gates,
        source_timewarps=source_timewarps,
        source_scene_cut_times=source_scene_cut_times,
        timing_draft=timing_draft,
        rhythm=rhythm,
        section_plan=section_plan,
        maximum_sequence_length=bundle.maximum_sequence_length,
        maximum_sources=bundle.maximum_sources,
        artifact_directory=inference_artifact_root / "causal-sensing",
        created_at_utc=(
            datetime.now(UTC).isoformat().replace("+00:00", "Z")
        ),
    )
    materialized = materialize_stage_edit_inference_snapshot(
        song_id=request.song,
        sources=request.sources,
        alignment=alignment,
        alignment_hard_gates=alignment_hard_gates,
        source_timewarps=source_timewarps,
        timing_draft=timing_draft,
        rhythm=rhythm,
        section_plan=section_plan,
        output_directory=inference_artifact_root,
        feature_cache_directory=request.source_cache_directory,
        maximum_sequence_length=bundle.maximum_sequence_length,
        maximum_sources=bundle.maximum_sources,
        causal_sensing=causal_sensing.stage_edit_artifacts,
        edit_mode=request.edit_mode,
        target_member=request.target_member,
    )
    preflight = preflight_stage_edit_pipeline_bridge(
        feature_snapshot_path=materialized.snapshot_path,
        checkpoint_directory=bundle.checkpoint_directory,
        source_ids=materialized.source_ids,
        edit_mode=request.edit_mode,
        target_member=request.target_member,
        **_registered_stage_edit_bridge_kwargs(bundle),
    )
    if preflight.readiness.checkpoint_path != str(bundle.checkpoint_directory):
        raise RuntimeError(
            "registered StageEdit checkpoint identity changed during materialization"
        )
    persisted_causal_sensing_audit = write_ive_causal_sensing_pipeline_audit(
        pipeline_result=causal_sensing,
        inference_snapshot=materialized,
        artifact_directory=inference_artifact_root / "causal-sensing-audit",
    )
    return materialized, preflight, persisted_causal_sensing_audit


def _enforce_automatic_group_stage_edit_requirement(
    request: AnalysisRequest,
    registered_bundle: RegisteredStageEditProductionBundle | None,
) -> None:
    """Keep the simple automatic product path from silently using baseline."""

    requires_professional_generation = bool(
        getattr(request, "source_mode", None) is SourceMode.AUTOMATIC
        and getattr(request, "edit_mode", None) is EditMode.GROUP
        and not bool(getattr(request, "acquire_only", False))
    )
    if (
        requires_professional_generation
        and registered_bundle is None
        and getattr(
            request,
            "stage_edit_generation_checkpoint_directory",
            None,
        )
        is None
    ):
        raise RuntimeError(
            "automatic IVE group editing requires a verified registered "
            "StageEdit production bundle; the ordinary visual baseline is "
            "not an allowed fallback"
        )


def analyze(
    request: AnalysisRequest,
    reporter: EventReporter,
    *,
    allow_failed_comparison_render: bool = False,
    review_rendered_proxies: bool = False,
    acquisition_identity_authority: (
        VerifiedMemberAcquisitionIdentityAuthority | None
    ) = None,
) -> AnalysisArtifact:
    if not isinstance(allow_failed_comparison_render, bool):
        raise TypeError("allow_failed_comparison_render must be a boolean")
    if not isinstance(review_rendered_proxies, bool):
        raise TypeError("review_rendered_proxies must be a boolean")
    if acquisition_identity_authority is not None and not (
        request.source_mode is SourceMode.AUTOMATIC
        and request.edit_mode is EditMode.MEMBER
        and is_verified_member_acquisition_identity_authority(
            acquisition_identity_authority
        )
    ):
        raise TypeError(
            "acquisition identity authority is valid only for automatic member requests"
        )
    registered_stage_edit_bundle = (
        _discover_automatic_stage_edit_production_bundle(request)
    )
    registered_stage_edit_motion_authority_v3 = (
        _registered_dynamic_crop_motion_authority_v3(
            registered_stage_edit_bundle
        )
    )
    _enforce_automatic_group_stage_edit_requirement(
        request,
        registered_stage_edit_bundle,
    )
    if (
        (
            request.stage_edit_checkpoint_directory is not None
            or getattr(
                request,
                "stage_edit_generation_checkpoint_directory",
                None,
            )
            is not None
            or getattr(
                request,
                "stage_edit_learned_reviewer_checkpoint_directory",
                None,
            )
            is not None
            or registered_stage_edit_bundle is not None
        )
        and not review_rendered_proxies
    ):
        raise RuntimeError(
            "StageEdit learned inference requires decoded proxy review"
        )
    reporter.progress(0.01, "요청을 확인하는 중…")
    request.output_directory.mkdir(parents=True, exist_ok=True)
    professional_reference_profile = _load_professional_reference_profile(
        request
    )
    professional_reference_cadence_profile = (
        _load_professional_reference_cadence_profile(
            request,
            professional_reference_profile,
        )
    )
    learned_reviewer_session = _load_learned_reviewer_session(request)
    stage_edit_generation_preflight = _load_stage_edit_generation_preflight(
        request
    )
    stage_edit_montage_policy = _load_stage_edit_montage_policy(
        registered_bundle=registered_stage_edit_bundle,
        explicit_preflight=stage_edit_generation_preflight,
    )
    stage_edit_motion_cut_profile = _load_stage_edit_motion_cut_profile(
        registered_bundle=registered_stage_edit_bundle,
        explicit_preflight=stage_edit_generation_preflight,
        montage_policy=stage_edit_montage_policy,
    )
    stage_edit_motion_cut_profile_payload = (
        _stage_edit_motion_cut_profile_manifest_payload(
            stage_edit_motion_cut_profile,
            stage_edit_montage_policy,
        )
        if stage_edit_motion_cut_profile is not None
        and stage_edit_montage_policy is not None
        else None
    )
    (
        stage_edit_continuity_authority,
        stage_edit_continuity_scorer_contract,
        stage_edit_continuity_scorer_fingerprint,
    ) = _stage_edit_continuity_config_identity(
        registered_bundle=registered_stage_edit_bundle,
        explicit_preflight=stage_edit_generation_preflight,
    )
    stage_edit_semimarkov_config = (
        _stage_edit_semimarkov_config_from_montage_policy(
            stage_edit_montage_policy,
            motion_cut_profile=stage_edit_motion_cut_profile,
            require_motion_cut_calibration=True,
            minimum_unique_sources=(
                MINIMUM_AUTOMATIC_SOURCE_COUNT
                if request.source_mode is SourceMode.AUTOMATIC
                and request.edit_mode is EditMode.GROUP
                else 3
            ),
            maximum_source_occupancy_fraction=(
                0.45
                if request.source_mode is SourceMode.AUTOMATIC
                and request.edit_mode is EditMode.GROUP
                else 0.60
            ),
            continuity_authority=stage_edit_continuity_authority,
            continuity_scorer_contract=(
                stage_edit_continuity_scorer_contract
            ),
            continuity_scorer_fingerprint=(
                stage_edit_continuity_scorer_fingerprint
            ),
        )
        if stage_edit_montage_policy is not None
        else None
    )
    rendered_montage_config = (
        _rendered_montage_config_from_policy(
            stage_edit_montage_policy,
            require_complete_role_duration_policy=True,
        )
        if stage_edit_montage_policy is not None
        else None
    )
    if professional_reference_profile is not None:
        reporter.log(
            "인간 교차편집 상대 품질 프로필 "
            f"{professional_reference_profile.reference_count}편 / "
            f"{professional_reference_profile.reference_cut_count}컷 엄격 검증 완료 · "
            f"SHA-256 {professional_reference_profile.profile_sha256[:12]}"
        )
    if professional_reference_cadence_profile is not None:
        reporter.log(
            "인간 교차편집 컷 타이밍 프로필 "
            f"{professional_reference_cadence_profile.reference_count}편 엄격 검증 완료 · "
            "파일 SHA-256 "
            f"{str(professional_reference_cadence_profile.file_sha256)[:12]}"
        )
    if learned_reviewer_session is not None:
        reporter.log(
            "검증된 StageEdit learned reviewer 활성화 · checkpoint "
            f"{learned_reviewer_session.metadata.checkpoint_id} · bundle "
            f"{learned_reviewer_session.evidence['bundle_sha256'][:12]}"
        )
    if stage_edit_generation_preflight is not None:
        preflight_dynamic_crop_version = getattr(
            stage_edit_generation_preflight, "dynamic_crop_version", 2
        )
        preflight_continuity_authority = getattr(
            stage_edit_generation_preflight,
            "continuity_authority",
            HANDCRAFTED_CONTINUITY_AUTHORITY,
        )
        reporter.log(
            "StageEdit Dynamic Crop "
            f"v{preflight_dynamic_crop_version} "
            "후보 생성을 명시적으로 활성화 · "
            f"snapshot {stage_edit_generation_preflight.snapshot.snapshot_sha256[:12]} "
            f"· source {len(stage_edit_generation_preflight.source_ids)}개 · "
            "continuity "
            f"{preflight_continuity_authority} "
            "· "
            "후보 8개 전수 시각·프록시 QA 통과 전에는 렌더 승격 없음"
        )
    if stage_edit_montage_policy is not None:
        reporter.log(
            "IVE 사람 편집본 몽타주 문법 프로필 엄격 검증 완료 · "
            f"4곡/20편 · file SHA-256 "
            f"{stage_edit_montage_policy.identity.file_sha256[:12]} · "
            "대형 앵커→근접 연쇄→대형 복귀 게이트 활성화"
        )
    if stage_edit_motion_cut_profile_payload is not None:
        _report_stage_edit_motion_cut_profile(
            reporter,
            stage_edit_motion_cut_profile_payload,
        )
    if registered_stage_edit_bundle is not None:
        reporter.log(
            "등록된 production StageEdit Dynamic Crop "
            f"v{getattr(registered_stage_edit_bundle, 'dynamic_crop_version', 2)} "
            "자동 활성화 · "
            f"checkpoint {registered_stage_edit_bundle.checkpoint_id} · "
            f"registry {registered_stage_edit_bundle.registry_sha256[:12]} · "
            "현재 동기화 원본에서 feature snapshot을 새로 생성한 뒤 "
            "후보 8개 전수 시각·프록시 QA를 적용"
        )

    reporter.status("영상 메타데이터를 읽는 중…")
    media: list[MediaInfo] = []
    for index, source in enumerate(request.sources):
        info = probe_media(source)
        _validate_media(info)
        media.append(info)
        reporter.log(
            f"{source.name}: {info.duration_seconds:.2f}초, "
            f"{len(info.video_streams)} video / {len(info.audio_streams)} audio"
        )
        reporter.progress(0.05 + 0.15 * (index + 1) / len(request.sources), "영상 확인 중…")

    # An automatic portrait-member request is already materialized at this
    # point. Enforce its native-or-lossless-croppable 9:16 output promise before
    # alignment or scene work so a bad cache can never fall through into
    # anchor/tracking.
    _automatic_member_portrait_source_ratios(request, media)

    alignment, reference_pcm = _align_sequentially(request, reporter)
    alignment_peer_consensus = (
        _verify_automatic_member_borderline_alignments(
            request,
            alignment,
            reporter=reporter,
            reference_pcm=reference_pcm,
            acquisition_identity_authority=acquisition_identity_authority,
        )
    )
    alignment_eligibility = _alignment_eligibility(
        alignment,
        edit_mode=request.edit_mode,
        member_peer_consensus=alignment_peer_consensus,
    )
    eligible_indices = [
        index
        for index, item in enumerate(alignment_eligibility)
        if item["render_eligible"]
    ]
    if (
        request.source_mode is SourceMode.AUTOMATIC
        and request.edit_mode is EditMode.GROUP
        and len(eligible_indices) < MINIMUM_AUTOMATIC_SOURCE_COUNT
    ):
        raise ValueError(
            "음원 동기화 후 사용 가능한 자동 소스가 "
            f"{len(eligible_indices)}개입니다. 최소 "
            f"{MINIMUM_AUTOMATIC_SOURCE_COUNT}개가 필요합니다."
        )
    common_start, common_end = _common_master_range(
        media,
        alignment,
        eligible_indices,
    )
    collect_group_performance_pcm = (
        request.source_mode is SourceMode.AUTOMATIC
        and request.edit_mode is EditMode.GROUP
    )
    decoded_pcm_by_source: dict[int, NDArray[np.float32]] = {}
    source_timewarps, local_timewarp_payloads = _estimate_source_timewarps(
        request,
        alignment,
        reference_pcm,
        eligible_indices=eligible_indices,
        reporter=reporter,
        decoded_pcm_by_source=(
            decoded_pcm_by_source if collect_group_performance_pcm else None
        ),
    )
    scalar_common_start, scalar_common_end = common_start, common_end
    common_start, common_end = _timewarp_safe_common_range(
        common_start,
        common_end,
        media=media,
        source_timewarps=source_timewarps,
        eligible_indices=eligible_indices,
    )
    if (
        abs(common_start - scalar_common_start) > 1e-6
        or abs(common_end - scalar_common_end) > 1e-6
    ):
        reporter.log(
            "로컬 드리프트 범위를 반영해 공통 구간을 "
            f"{common_start:.3f}–{common_end:.3f}초로 조정"
        )

    rhythm_payload: dict[str, Any] | None = None
    rhythm_analysis = None
    rhythm_warning: str | None = None
    reporter.status("기준 곡의 박자와 프레이즈를 분석하는 중…")
    try:
        rhythm_analysis = analyze_rhythm(reference_pcm, ANALYSIS_SAMPLE_RATE)
        rhythm_payload = rhythm_analysis.to_dict()
        # The full onset envelope is useful for diagnostics but unnecessarily
        # large for the app contract. Keep its timing metadata and derived grid.
        onset = rhythm_payload.get("onset_envelope")
        if isinstance(onset, dict):
            onset.pop("times_seconds", None)
            onset.pop("values", None)
        reporter.log(
            f"추정 템포 {rhythm_analysis.bpm:.2f} BPM · "
            f"박자 {len(rhythm_analysis.beats)}개 · "
            f"프레이즈 후보 {len(rhythm_analysis.phrase_candidates)}개"
        )
    except ValueError as exc:
        rhythm_warning = f"박자 분석을 확정하지 못함: {exc}"
        reporter.emit("log", level="warning", message=rhythm_warning)
    reporter.progress(0.64, "박자 분석 완료")

    scene_payloads: list[dict[str, Any]] = []
    performance_range_trim_payload: dict[str, Any] | None = None
    scene_config = SceneDetectionConfig(min_gap_frames=8)
    for index, (source, info) in enumerate(zip(request.sources, media, strict=True)):
        reporter.status(f"카메라 컷 분석 중: {source.name}")
        changes, scene_cache_diagnostics = detect_scene_changes_in_video_cached(
            source,
            config=scene_config,
            sample_every=2,
            source_cache_directory=request.source_cache_directory,
        )
        frame_rate = info.video_streams[0].frame_rate or 30.0
        scene_payloads.append(
            {
                "index_domain": REQUEST_SOURCE_INDEX_DOMAIN,
                "source_index": index,
                "source": str(source),
                "frame_rate": frame_rate,
                "detection_policy": {
                    "analysis_basis": (
                        "ive_attributed_component_gated_hard_cut_detector_v2"
                    ),
                    "combined_threshold": scene_config.threshold,
                    "minimum_frame_difference": (
                        scene_config.minimum_frame_difference
                    ),
                    "minimum_histogram_distance": (
                        scene_config.minimum_histogram_distance
                    ),
                    "minimum_gap_frames": scene_config.min_gap_frames,
                    "sample_every_source_frames": 2,
                },
                "persistent_scalar_cache": scene_cache_diagnostics,
                "changes": [
                    {
                        "frame_index": change.frame_index,
                        "timestamp_seconds": change.frame_index / frame_rate,
                        "score": change.score,
                        "frame_difference": change.frame_difference,
                        "histogram_distance": change.histogram_distance,
                    }
                    for change in changes
                ],
            }
        )
        cache_label = (
            " · 영속 캐시 재사용"
            if scene_cache_diagnostics.get("analysis_reused") is True
            else ""
        )
        reporter.log(
            f"{source.name}: 방송 카메라 컷 {len(changes)}개 감지{cache_label}"
        )
        reporter.progress(
            0.64 + 0.30 * (index + 1) / len(request.sources),
            "방송 카메라 컷을 분석하는 중…",
        )

    if collect_group_performance_pcm:
        reporter.status("공통 공연 시작·마지막 포즈 구간을 확인하는 중…")
        try:
            reference_index = request.reference_source
            reference_clock = source_timewarps[reference_index]
            reference_offset = alignment.alignments[
                reference_index
            ].offset_seconds
            reference_scene_cut_master_times = tuple(
                (
                    reference_clock.source_to_master(
                        float(change["timestamp_seconds"])
                    )
                    if reference_clock is not None
                    else float(change["timestamp_seconds"]) - reference_offset
                )
                for change in scene_payloads[reference_index]["changes"]
            )
            performance_range_estimate = estimate_shared_performance_range(
                decoded_pcm_by_source,
                sample_rate=ANALYSIS_SAMPLE_RATE,
                common_start_seconds=common_start,
                common_end_seconds=common_end,
                source_indices=eligible_indices,
                reference_source_index=reference_index,
                audio_offsets={
                    index: alignment.alignments[index].offset_seconds
                    for index in eligible_indices
                },
                source_timewarps={
                    index: source_timewarps[index]
                    for index in eligible_indices
                },
                reference_scene_cut_master_seconds=(
                    reference_scene_cut_master_times
                ),
            )
            performance_range_trim_payload = performance_range_estimate.to_dict()
            common_start = performance_range_estimate.final_start_seconds
            common_end = performance_range_estimate.final_end_seconds
            if performance_range_estimate.applied:
                reporter.log(
                    "공통 공연 음원 합의와 기준 방송 하드컷으로 렌더 구간을 "
                    f"{common_start:.3f}–{common_end:.3f}초로 조정"
                )
            else:
                reporter.log(
                    "공통 공연 구간 트림을 보수적으로 적용하지 않고 기존 범위를 "
                    f"유지 ({performance_range_estimate.reason})"
                )
        except (IndexError, KeyError, TypeError, ValueError, FloatingPointError) as exc:
            performance_range_trim_payload = {
                "analysis_basis": (
                    "local_pcm_spectral_flux_reference_peer_consensus_"
                    "with_reference_hard_cut_pose_tail_v1"
                ),
                "scope": "automatic_group_only",
                "applied": False,
                "evidence_strong": False,
                "reason": "estimation_failed_original_range_retained",
                "error": str(exc),
                "original_range": {
                    "start_seconds": common_start,
                    "end_seconds": common_end,
                    "duration_seconds": common_end - common_start,
                },
                "final_range": {
                    "start_seconds": common_start,
                    "end_seconds": common_end,
                    "duration_seconds": common_end - common_start,
                    "trimmed_start_seconds": 0.0,
                    "trimmed_end_seconds": 0.0,
                },
                "privacy": {
                    "raw_pcm_persisted": False,
                    "spectral_flux_vectors_persisted": False,
                },
            }
            reporter.emit(
                "log",
                level="warning",
                message=(
                    "공통 공연 구간 근거를 확정하지 못해 기존 범위를 유지: "
                    f"{exc}"
                ),
            )
        finally:
            decoded_pcm_by_source.clear()

    member_tracking_results: dict[int, MemberTrackingResult] = {}
    member_tracking_payloads: list[dict[str, Any]] = []
    # Retained as an empty schema field for older analysis readers. Automatic
    # member requests are direct full-frame only and never infer anchors.
    member_auto_anchor_payloads: list[dict[str, Any]] = []
    direct_full_frame_ratios = _automatic_member_direct_full_frame_ratios(
        request,
        media,
        eligible_indices,
    )
    use_direct_member_full_frame = direct_full_frame_ratios is not None
    member_direct_full_frame_payload: dict[str, Any] | None = None
    member_direct_dense_face_review_payload: dict[str, Any] | None = None
    member_direct_professional_pre_render_payload: dict[str, Any] | None = None
    member_look_plan: DirectMemberLookPlan | None = None
    used_automatic_member_anchors = False
    member_common_start = common_start
    member_anchor_trim_payload: dict[str, Any] | None = None
    if request.edit_mode is EditMode.MEMBER and not use_direct_member_full_frame:
        assert request.target_member is not None
        if request.source_mode is not SourceMode.LOCAL:
            raise RuntimeError(
                "automatic member mode must use the verified direct full-frame path"
            )
        anchors_by_source = {
            anchor.source_index: anchor for anchor in request.member_anchors
        }
        for source_index, (source, info) in enumerate(
            zip(request.sources, media, strict=True)
        ):
            if source_index not in eligible_indices:
                continue
            anchor_request = anchors_by_source.get(source_index)
            if anchor_request is None:
                raise ValueError(f"멤버 앵커가 없습니다: {source.name}")
            assert info.duration_seconds is not None
            if anchor_request.time_seconds >= info.duration_seconds:
                raise ValueError(
                    f"멤버 앵커 시각이 영상 길이를 벗어남: {source.name}"
                )
            fps = info.video_streams[0].frame_rate or 30.0
            sample_every = 1
            anchor_frame = int(round(anchor_request.time_seconds * fps / sample_every)) * sample_every
            reporter.status(f"{request.target_member.value} 추적 중: {source.name}")
            scene_frames = {
                int(change["frame_index"])
                for change in scene_payloads[source_index]["changes"]
                if int(change["frame_index"]) > anchor_frame
            }
            result = track_anchored_member(
                _tracking_frames(source, sample_every=sample_every),
                anchor=MemberAnchor(
                    frame_index=anchor_frame,
                    bbox_xyxy=anchor_request.bbox_xyxy,
                    member=request.target_member,
                    source_id=str(source),
                ),
                fps=fps,
                config=AnchoredTrackerConfig(
                    minimum_confidence=0.42,
                    minimum_appearance_similarity=0.08,
                    maximum_center_jump=0.26,
                    hold_frames=4,
                ),
                scene_change_frames=scene_frames,
            )
            member_tracking_results[source_index] = result
            member_tracking_payloads.append(
                _member_tracking_payload(
                    source_index,
                    result,
                    anchor_request.time_seconds,
                    anchor_request.bbox_xyxy,
                )
            )
            tracked_count = sum(
                item.state is TrackingState.TRACKED for item in result.observations
            )
            reporter.log(
                f"{source.name}: 안전 추적 {tracked_count}프레임 · "
                f"종료 사유 {result.terminal_failure.value if result.terminal_failure else '없음'}"
            )
            reporter.progress(
                0.94 + 0.035 * (source_index + 1) / len(request.sources),
                "대상 멤버를 추적하는 중…",
            )
        member_common_start, member_anchor_trim_payload = _member_anchor_trim(
            common_master_start=common_start,
            common_master_end=common_end,
            alignment=alignment,
            eligible_indices=eligible_indices,
            tracking_results=member_tracking_results,
            source_timewarps=source_timewarps,
        )

    warnings: list[str] = []
    if rhythm_warning:
        warnings.append(rhythm_warning)
    for eligibility in alignment_eligibility:
        reasons = eligibility["reasons"]
        if reasons:
            warnings.append(
                f"자동 편집 후보에서 제외: {Path(str(eligibility['source'])).name} "
                f"({'; '.join(str(reason) for reason in reasons)})"
            )
    member_identity_verification = alignment_peer_consensus.get(
        "broadcast_identity_verification"
    )
    if (
        request.source_mode is SourceMode.AUTOMATIC
        and request.edit_mode is EditMode.MEMBER
        and isinstance(member_identity_verification, Mapping)
        and member_identity_verification.get("verified") is not True
    ):
        warnings.append(
            "수집 단계의 방송일·음방 프로그램 권한을 검증하지 못해 "
            "기본 오디오 문턱 아래 소스는 다중 방송 합의로 복구하지 "
            "않았습니다. 공식 메타데이터 새로고침 후 다시 시도합니다."
        )
    if (
        member_anchor_trim_payload is not None
        and member_anchor_trim_payload["applied"]
    ):
        warnings.append(
            "수동 멤버 앵커보다 앞선 구간은 뒤로 추적하지 않습니다. "
            f"가장 늦은 공통 앵커까지 "
            f"{float(member_anchor_trim_payload['trimmed_seconds']):.3f}초를 "
            "안전하게 제외했습니다."
        )
    if member_anchor_trim_payload is not None:
        full_frame_only = member_anchor_trim_payload.get(
            "full_frame_only_source_indices", []
        )
        if full_frame_only:
            warnings.append(
                f"공식 대상 멤버 직캠 {len(full_frame_only)}개는 안전한 "
                "자동 앵커가 없어 대상 크롭 없이 원본 전체 프레임 "
                "하드컷 후보로만 유지합니다."
            )

    group_draft = None
    section_plan: SectionPlan | None = None
    source_look_match_result: SourceLookMatchResult | None = None
    source_derivative_diversity_result: (
        SourceDerivativeDiversityResult | None
    ) = None
    source_derivative_audit_path: Path | None = None
    boundary_look_match_result: BoundaryLookMatchResult | None = None
    visual_candidate_set: VisualTimelineCandidateSet | None = None
    stage_edit_candidate_rerank_result: StageEditCandidateRerankResult | None = None
    stage_edit_inference_snapshot_result: (
        StageEditInferenceSnapshotResult | None
    ) = None
    stage_edit_causal_sensing_pipeline_audit: (
        IVECausalSensingPipelineAuditArtifact | None
    ) = None
    stage_edit_generation_result: StageEditPipelineBridgeResult | None = None
    stage_edit_generation_audit_path: Path | None = None
    stage_edit_visual_candidate_batch: StageEditVisualCandidateBatch | None = None
    materialized_group_candidates: tuple[Any, ...] = ()
    complete_group_candidates: tuple[_GroupRenderCandidateBundle, ...] = ()
    candidate_derivation_failures: list[dict[str, Any]] = []
    candidate_post_repair_review_rejections: list[dict[str, Any]] = []
    pre_repair_structure_passing_materialized_count = 0
    total_materialized_candidate_count = 0
    repair_eligible_candidate_count = 0
    repair_eligible_materialized_count = 0
    repair_eligible_attempted_count = 0
    visual_timeline: VisualTimelinePlan | None = None
    visual_boundary_repairs: VisualBoundaryRepairPlan | None = None
    timeline_review: TimelineReview | None = None
    face_match_plan: FaceMatchPlan | None = None
    group_transition_plan: GroupTransitionPlan | None = None
    boundary_affine_plan: BoundaryAffinePlan | None = None
    boundary_subject_alignment_plan: BoundarySubjectAlignmentPlan | None = None
    editorial_motion_plan: EditorialMotionPlan | None = None
    member_editorial_motion_plan: EditorialMotionPlan | None = None
    member_draft: MemberDraftTimeline | None = None
    cut_local_sync_payload: dict[str, Any] | None = None
    render_payload: dict[str, Any] | None = None
    rendered_proxy_review_payload: dict[str, Any] | None = None
    rendered_montage_grammar_review_payload: dict[str, Any] | None = None
    stage_edit_exact_shot_clock_payload: dict[str, Any] | None = None
    stage_edit_dynamic_crop_version_payload: int | None = None
    dynamic_crop_v3_decodes_payload: list[dict[str, Any]] | None = None
    dynamic_crop_motion_v3_payload: dict[str, Any] | None = None
    rendered_proxy_review_path: Path | None = None
    rendered_candidate_selection_payload: dict[str, Any] | None = None
    rendered_candidate_selection_path: Path | None = None
    rendered_video_path: Path | None = None
    render_plan_path: Path | None = None
    timeline_interchange_paths: tuple[Path, ...] = ()
    timestamp = datetime.now(UTC)
    base_filename = f"IVE-{_safe_slug(request.song)}-{timestamp.strftime('%Y%m%d-%H%M%S')}"
    group_ready = len(eligible_indices) >= 2 and rhythm_analysis is not None
    if (
        stage_edit_generation_preflight is not None
        or registered_stage_edit_bundle is not None
    ) and not group_ready:
        raise RuntimeError(
            "StageEdit generation requires synchronized sources and "
            "a completed rhythm analysis; no silent baseline fallback is allowed"
        )
    if request.edit_mode is EditMode.GROUP and group_ready:
        try:
            section_plan = build_section_plan(
                rhythm_analysis,
                start_seconds=common_start,
                end_seconds=common_end,
            )
            reporter.log(
                "음향 구간 지시 "
                f"{len(section_plan.sections)}개 생성 · "
                "구간별 원본 유지·편집 카메라 예산 적용"
            )
        except (AttributeError, TypeError, ValueError, FloatingPointError) as exc:
            warnings.append(
                "음향 구간 지시를 확정하지 못해 전곡 공통 편집 예산을 사용함: "
                f"{exc}"
            )
        reporter.progress(0.95, "자동 교차편집 타임라인을 만드는 중…")
        offsets = tuple(
            alignment.alignments[index].offset_seconds for index in eligible_indices
        )
        sync_scores = tuple(
            alignment.alignments[index].score for index in eligible_indices
        )
        scene_cut_times = tuple(
            tuple(
                float(change["timestamp_seconds"])
                for change in scene_payloads[index]["changes"]
            )
            for index in eligible_indices
        )
        eligible_timewarps = tuple(
            source_timewarps[index] for index in eligible_indices
        )
        phrase_times = tuple(
            sorted(
                {
                    float(time_value)
                    for phrase in rhythm_analysis.phrase_candidates
                    for time_value in (phrase.start_seconds, phrase.end_seconds)
                }
            )
        )
        group_draft = build_group_draft(
            source_count=len(eligible_indices),
            common_master_start=common_start,
            common_master_end=common_end,
            audio_offsets=offsets,
            sync_scores=sync_scores,
            beat_times=rhythm_analysis.beat_times_seconds,
            phrase_times=phrase_times,
            scene_cut_times=scene_cut_times,
        )
        source_aspect_ratios = tuple(
            (
                float(media[index].video_streams[0].width)
                / float(media[index].video_streams[0].height)
            )
            if media[index].video_streams[0].width
            and media[index].video_streams[0].height
            else 0.0
            for index in eligible_indices
        )
        local_durations = tuple(
            float(
                media[index].video_streams[0].duration_seconds
                or media[index].duration_seconds
                or 0.0
            )
            for index in eligible_indices
        )
        should_analyze_source_derivatives = (
            request.source_mode is SourceMode.AUTOMATIC
            or registered_stage_edit_bundle is not None
            or stage_edit_generation_preflight is not None
        )
        if should_analyze_source_derivatives:
            reporter.status(
                "동일 카메라 재인코딩·크롭 파생본을 실제 동기 프레임으로 "
                "검사하는 중…"
            )
            source_derivative_audit_path = (
                request.output_directory
                / f"{base_filename}-source-derivative-diversity.json"
            )
            try:
                source_derivative_diversity_result = (
                    analyze_source_derivative_diversity(
                        tuple(
                            request.sources[index] for index in eligible_indices
                        ),
                        source_offsets_seconds=offsets,
                        source_timewarps=eligible_timewarps,
                        common_master_start=common_start,
                        common_master_end=common_end,
                    )
                )
                source_derivative_audit_path.write_text(
                    json.dumps(
                        _with_index_domain(
                            source_derivative_diversity_result.to_dict(),
                            ELIGIBLE_SOURCE_INDEX_DOMAIN,
                        ),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                reporter.log(
                    "실제 PTS 다중 시점 소스 파생본 검사 · "
                    f"파일 {len(eligible_indices)}개 → 독립 카메라 "
                    f"계열 {source_derivative_diversity_result.family_count}개 · "
                    "확정 파생 쌍 "
                    f"{source_derivative_diversity_result.confirmed_derivative_pair_count}개 · "
                    "불확실 쌍 "
                    f"{source_derivative_diversity_result.uncertain_pair_count}개"
                )
            except (cv2.error, OSError, RuntimeError, TypeError, ValueError) as exc:
                source_derivative_audit_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "analysis_basis": (
                                "canonical_timewarp_actual_pts_multiview_"
                                "structure_geometry_v1"
                            ),
                            "index_domain": ELIGIBLE_SOURCE_INDEX_DOMAIN,
                            "production_ready": False,
                            "status": "analysis_failed_closed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "raw_frames_persisted": False,
                            "descriptors_persisted": False,
                            "embeddings_persisted": False,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                warnings.append(
                    "동일 카메라 파생본 계열을 확정하지 못함: "
                    f"{exc}"
                )
            if (
                registered_stage_edit_bundle is not None
                or stage_edit_generation_preflight is not None
            ) and (
                source_derivative_diversity_result is None
                or not source_derivative_diversity_result.production_ready
            ):
                raise RuntimeError(
                    "production StageEdit generation blocked because actual-PTS "
                    "source derivative families are uncertain or clock-incomplete; "
                    f"audit: {source_derivative_audit_path}"
                )
        if registered_stage_edit_bundle is not None:
            if section_plan is None:
                raise RuntimeError(
                    "registered StageEdit generation requires a completed "
                    "section plan; no silent visual fallback is allowed"
                )
            if tuple(eligible_indices) != tuple(range(len(request.sources))):
                raise RuntimeError(
                    "registered StageEdit generation requires every current "
                    "source to pass the synchronization hard gate; no silent "
                    "visual fallback is allowed"
                )
            reporter.status(
                "현재 동기화 원본에서 Transformer feature snapshot을 "
                "생성하는 중…"
            )
            (
                stage_edit_inference_snapshot_result,
                stage_edit_generation_preflight,
                stage_edit_causal_sensing_pipeline_audit,
            ) = _materialize_registered_stage_edit_preflight(
                bundle=registered_stage_edit_bundle,
                request=request,
                alignment=alignment,
                alignment_eligibility=alignment_eligibility,
                source_timewarps=source_timewarps,
                source_scene_cut_times=scene_cut_times,
                timing_draft=group_draft,
                rhythm=rhythm_analysis,
                section_plan=section_plan,
            )
            reporter.log(
                "실제 원본 feature snapshot 엄격 검증 완료 · "
                f"{len(stage_edit_inference_snapshot_result.decision_times)} "
                "decision frames · snapshot "
                f"{stage_edit_inference_snapshot_result.snapshot.snapshot_sha256[:12]}"
            )
        if stage_edit_generation_preflight is not None:
            preflight_dynamic_crop_version = getattr(
                stage_edit_generation_preflight, "dynamic_crop_version", 2
            )
            if section_plan is None:
                raise RuntimeError(
                    "StageEdit generation requires a completed section plan"
                )
            if (
                stage_edit_montage_policy is None
                or stage_edit_semimarkov_config is None
                or rendered_montage_config is None
            ):
                raise RuntimeError(
                    "StageEdit generation requires a verified IVE professional "
                    "montage policy before decoding"
                )
            if tuple(eligible_indices) != tuple(range(len(request.sources))):
                raise RuntimeError(
                    "StageEdit generation requires every configured "
                    "source to pass the current synchronization hard gate"
                )
            reporter.status(
                "Dynamic Crop "
                f"v{preflight_dynamic_crop_version} "
                "Transformer 헤드를 semi-Markov 후보로 "
                "디코딩하는 중…"
            )
            generation_width, generation_height = output_size(
                request.aspect_ratio
            )
            stage_edit_generation_result = run_stage_edit_pipeline_bridge(
                preflight=stage_edit_generation_preflight,
                sources=request.sources,
                source_durations_seconds=local_durations,
                source_start_offsets_seconds=offsets,
                internal_cut_times_seconds=scene_cut_times,
                source_timewarps=eligible_timewarps,
                # The Transformer already owns learned source preference.
                # Do not inject a second, uncalibrated sync-derived score.
                source_quality_log_priors=(0.0,) * len(request.sources),
                source_derivative_metadata=(
                    source_derivative_diversity_result.decoder_metadata()
                    if source_derivative_diversity_result is not None
                    else None
                ),
                require_verified_source_derivative_metadata=True,
                section_plan=section_plan,
                output=(
                    request.output_directory
                    / f".{base_filename}-stageedit-retained-candidate.mp4"
                ),
                reference_source=request.reference_source,
                aspect_ratio=request.aspect_ratio,
                width=generation_width,
                height=generation_height,
                fps=29.97,
                config=stage_edit_semimarkov_config,
            )
            stage_edit_generation_audit_path = (
                request.output_directory
                / f"{base_filename}-stageedit-generation-audit.json"
            )
            stage_edit_generation_result.write_json(
                stage_edit_generation_audit_path
            )
            learned_director = (
                stage_edit_generation_result.learned_section_director
            )
            if learned_director is None:
                raise RuntimeError(
                    "production StageEdit generation did not apply its learned "
                    "Section Director before semi-Markov decoding"
                )
            reporter.log(
                "Transformer Section Director "
                f"{len(learned_director.evidence)}개 구간의 컷 밀도·홀드·"
                "카메라 예산을 validation 동결값으로 결정"
            )
            reporter.log(
                "StageEdit 후보 "
                f"{len(stage_edit_generation_result.retained_candidates)}개를 "
                "별도 감사 자료로 보존 · 시각 하드 게이트와 "
                "실렌더 프록시 QA 전에는 선택하지 않음"
            )
        if request.source_mode is SourceMode.AUTOMATIC:
            reporter.status("방송 원본 사이의 안정적인 색감 차이를 맞추는 중…")
            try:
                reference_local_index = (
                    eligible_indices.index(request.reference_source)
                    if request.reference_source in eligible_indices
                    else None
                )
                source_look_match_result = analyze_source_look_match(
                    tuple(
                        request.sources[index] for index in eligible_indices
                    ),
                    source_offsets_seconds=offsets,
                    common_master_start=common_start,
                    common_master_end=common_end,
                    reference_source=reference_local_index,
                    source_timewarps=eligible_timewarps,
                )
                applied_look_count = sum(
                    not item.is_identity
                    for item in source_look_match_result.adjustments
                )
                reporter.log(
                    "동기화 원본 룩 매칭 "
                    f"{applied_look_count}/{len(eligible_indices)}개 적용 · "
                    f"기준 {source_look_match_result.target_mode}"
                )
            except (TypeError, ValueError, cv2.error) as exc:
                warnings.append(
                    "원본별 색감 차이를 안정적으로 확정하지 못해 "
                    f"색 보정 없이 유지함: {exc}"
                )
        reporter.status("컷 전후 구도와 움직임을 비교하는 중…")
        try:
            visual_builder_arguments = {
                "sources": tuple(
                    request.sources[index] for index in eligible_indices
                ),
                "canonical_spans": tuple(
                    (
                        item.canonical_master_start,
                        item.canonical_master_end,
                    )
                    for item in group_draft.segments
                ),
                "audio_offsets": offsets,
                "sync_scores": sync_scores,
                "scene_cut_times": scene_cut_times,
                "source_timewarps": eligible_timewarps,
                # Only source-local numeric descriptors are persisted here.
                # Timeline optimization/review/exact-eight/proxy gates always
                # run again for every analysis.
                "feature_cache_directory": request.source_cache_directory,
                "preferred_source_indices": tuple(
                    item.source_index for item in group_draft.segments
                ),
                "progress_callback": lambda completed, total: reporter.progress(
                    0.96 + 0.018 * completed / max(1, total),
                    f"컷 전후 대형을 비교하는 중… {completed}/{total}",
                ),
            }
            if (
                request.source_mode is SourceMode.AUTOMATIC
                and review_rendered_proxies
            ):
                if stage_edit_generation_result is not None:
                    reporter.status(
                        "StageEdit 후보 8개의 실제 프레임·경계 안전성을 "
                        "전수 검증하는 중…"
                    )
                    try:
                        stage_edit_visual_candidate_batch = (
                            materialize_stage_edit_visual_candidates(
                                stage_edit_generation_result,
                                sources=tuple(
                                    request.sources[index]
                                    for index in eligible_indices
                                ),
                                audio_offsets=offsets,
                                sync_scores=sync_scores,
                                scene_cut_times=scene_cut_times,
                                source_timewarps=eligible_timewarps,
                                feature_cache_directory=(
                                    request.source_cache_directory
                                ),
                                progress_callback=(
                                    lambda candidate_position,
                                    candidate_total,
                                    completed,
                                    total: reporter.progress(
                                        0.96
                                        + 0.018
                                        * (
                                            (candidate_position - 1)
                                            + completed / max(1, total)
                                        )
                                        / max(1, candidate_total),
                                        "StageEdit 시각 하드 게이트 검증 중… "
                                        f"{candidate_position}/{candidate_total} · "
                                        f"{completed}/{total}",
                                    )
                                ),
                            )
                        )
                    except StageEditVisualCandidateError as exc:
                        rendered_candidate_selection_payload = {
                            "analysis_basis": (
                                "explicit_stage_edit_visual_hard_gate_failure_v1"
                            ),
                            "candidate_pool": "stage_edit_generation_only",
                            "ordinary_visual_baseline_candidate_pool_eligible": False,
                            "silent_baseline_fallback_allowed": False,
                            "requested_review_count": 8,
                            "reviewed_candidate_count": 0,
                            "passed": False,
                            "blocking_reason": (
                                "stage_edit_prescribed_visual_hard_gate_failed"
                            ),
                            "error": f"{type(exc).__name__}: {exc}",
                            "stage_edit_generation": (
                                stage_edit_generation_result.to_dict()
                            ),
                        }
                        rendered_candidate_selection_path = (
                            request.output_directory
                            / (
                                f"{base_filename}-"
                                "rendered-candidate-selection.json"
                            )
                        )
                        rendered_candidate_selection_path.write_text(
                            json.dumps(
                                rendered_candidate_selection_payload,
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        raise RuntimeError(
                            "explicit StageEdit generation failed visual hard "
                            "gates; no ordinary visual baseline fallback was "
                            f"used; report: {rendered_candidate_selection_path}"
                        ) from exc
                    materialized_group_candidates = (
                        stage_edit_visual_candidate_batch.candidates
                    )
                    reporter.log(
                        "StageEdit 후보 8개가 각자 제안한 소스·컷 경로 "
                        "그대로 기존 시각 하드 게이트 통과 · "
                        "실패 시 일반 시각 베이스라인 후보 보충 없음"
                    )
                else:
                    visual_candidate_set = build_visual_timeline_candidate_set(
                        **visual_builder_arguments,
                        maximum_results=32,
                    )
                    (
                        materialized_group_candidates,
                        stage_edit_candidate_rerank_result,
                    ) = _materialize_ranked_visual_proxy_candidates(
                        visual_candidate_set,
                        request=request,
                        rhythm_analysis=rhythm_analysis,
                        section_plan=section_plan,
                        reporter=reporter,
                    )
                pre_repair_structure_passing_materialized_count = len(
                    materialized_group_candidates
                )
                total_materialized_candidate_count = len(
                    materialized_group_candidates
                )
                derived: list[_GroupRenderCandidateBundle] = []
                attempted_candidate_indices: set[int] = set()
                attempted_materialized_candidates: list[Any] = []
                width, height = output_size(request.aspect_ratio)
                # Scalar-only and process-local: candidates commonly reuse the
                # same source-local cut frames, so decode once without ever
                # persisting pixels or bypassing per-candidate proxy review.
                boundary_look_measurement_cache = (
                    BoundaryLookMeasurementCache()
                )

                def derive_and_gate_candidate(
                    materialized: Any,
                ) -> None:
                    if materialized.candidate_index in attempted_candidate_indices:
                        return
                    attempted_candidate_indices.add(materialized.candidate_index)
                    attempted_materialized_candidates.append(materialized)
                    reporter.status(
                        "후보 경계 보정·완전 파생 중… "
                        f"{len(attempted_candidate_indices)}/최대 32"
                    )
                    try:
                        bundle = _derive_complete_group_candidate(
                            materialized,
                            request=request,
                            eligible_indices=eligible_indices,
                            audio_offsets=offsets,
                            scene_cut_times=scene_cut_times,
                            source_timewarps=eligible_timewarps,
                            source_aspect_ratios=source_aspect_ratios,
                            source_durations_seconds=local_durations,
                            beat_period_seconds=(
                                group_draft.beat_period_seconds
                            ),
                            phrase_times=phrase_times,
                            section_plan=section_plan,
                            common_master_start=common_start,
                            width=width,
                            height=height,
                            output=(
                                request.output_directory
                                / (
                                    f".{base_filename}-candidate-"
                                    f"{materialized.candidate_index:02d}.mp4"
                                )
                            ),
                            source_look_match=source_look_match_result,
                            boundary_look_measurement_cache=(
                                boundary_look_measurement_cache
                            ),
                        )
                        if isinstance(
                            materialized,
                            StageEditMaterializedVisualCandidate,
                        ):
                            bundle = _apply_stage_edit_candidate_render_contract(
                                bundle,
                                materialized,
                                motion_authority_v3=(
                                    registered_stage_edit_motion_authority_v3
                                ),
                            )
                    except (
                        StageEditVisualCandidateError,
                        TypeError,
                        ValueError,
                        cv2.error,
                    ) as exc:
                        candidate_derivation_failures.append(
                            {
                                "candidate_index": materialized.candidate_index,
                                "review_rank": materialized.review_rank,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        return
                    if (
                        bundle.timeline_review is None
                        or not bundle.timeline_review.passed
                    ):
                        candidate_post_repair_review_rejections.append(
                            _post_repair_review_rejection_payload(
                                materialized,
                                bundle,
                            )
                        )
                        return
                    pre_review_failures = _pre_review_hard_failure_codes(
                        materialized
                    )
                    if pre_review_failures:
                        repair = bundle.boundary_repairs
                        if repair is None or repair.changed_count < 1:
                            candidate_post_repair_review_rejections.append(
                                {
                                    **_post_repair_review_rejection_payload(
                                        materialized,
                                        bundle,
                                    ),
                                    "delivery_lineage_failure": (
                                        "failed_pre_review_without_changed_"
                                        "boundary_repair"
                                    ),
                                }
                            )
                            return
                        bundle = replace(
                            bundle,
                            candidate_lineage=bundle.candidate_lineage
                            + (
                                {
                                    "stage": (
                                        "pre_review_failure_repair_validation"
                                    ),
                                    "contract": (
                                        PRE_REVIEW_REPAIR_LINEAGE_CONTRACT
                                    ),
                                    "candidate_index": (
                                        materialized.candidate_index
                                    ),
                                    "pre_review_passed": False,
                                    "pre_review_hard_failure_codes": list(
                                        pre_review_failures
                                    ),
                                    "boundary_repair_analysis_basis": (
                                        repair.analysis_basis
                                    ),
                                    "boundary_repair_changed_count": (
                                        repair.changed_count
                                    ),
                                    "post_repair_timeline_review_passed": True,
                                    "post_repair_hard_failure_codes": [],
                                    "delivery_eligible": True,
                                },
                            ),
                        )
                    derived.append(bundle)

                for materialized in materialized_group_candidates:
                    derive_and_gate_candidate(materialized)

                if len(derived) < 8 and stage_edit_visual_candidate_batch is None:
                    recovery_record_order = (
                        stage_edit_candidate_rerank_result.ordered_records(
                            visual_candidate_set
                        )
                        if stage_edit_candidate_rerank_result is not None
                        else getattr(
                            visual_candidate_set,
                            "ranked_candidates",
                            (),
                        )
                    )
                    repair_eligible_records = tuple(
                        record
                        for record in recovery_record_order
                        if record.candidate_index not in attempted_candidate_indices
                        and _is_focus_only_boundary_repair_candidate(record)
                    )
                    repair_eligible_candidate_count = len(
                        repair_eligible_records
                    )
                    if repair_eligible_records:
                        reporter.log(
                            "사전 초점 경계만 탈락한 후보를 디코딩 경계 수선 후 "
                            "최종 구조 QA로 재검증"
                        )
                    for record in repair_eligible_records:
                        if len(derived) >= 8:
                            break
                        repair_eligible_attempted_count += 1
                        try:
                            materialized = (
                                materialize_visual_timeline_candidate_record(
                                    visual_candidate_set,
                                    record,
                                )
                            )
                        except (TypeError, ValueError, cv2.error) as exc:
                            candidate_derivation_failures.append(
                                {
                                    "candidate_index": record.candidate_index,
                                    "review_rank": record.review_rank,
                                    "stage": "repair_eligible_materialization",
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                            continue
                        repair_eligible_materialized_count += 1
                        total_materialized_candidate_count += 1
                        derive_and_gate_candidate(materialized)

                materialized_group_candidates = tuple(
                    attempted_materialized_candidates
                )
                complete_group_candidates = tuple(derived)
                if len(complete_group_candidates) != 8:
                    complete_candidate_count = len(
                        complete_group_candidates
                    )
                    rendered_candidate_selection_payload = {
                        "analysis_basis": (
                            "sequential_decoded_proxy_candidate_rerank_v1"
                        ),
                        "requested_review_count": 8,
                        "reviewed_candidate_count": 0,
                        "minimum_required_candidate_count": 8,
                        "minimum_candidate_count_satisfied": False,
                        "structure_passing_materialized_count": (
                            pre_repair_structure_passing_materialized_count
                        ),
                        "total_materialized_candidate_count": (
                            total_materialized_candidate_count
                        ),
                        "repair_eligible_candidate_count": (
                            repair_eligible_candidate_count
                        ),
                        "repair_eligible_materialized_count": (
                            repair_eligible_materialized_count
                        ),
                        "repair_eligible_attempted_count": (
                            repair_eligible_attempted_count
                        ),
                        "post_repair_structure_passing_complete_count": (
                            complete_candidate_count
                        ),
                        "successful_review_count": 0,
                        "complete_candidate_count": complete_candidate_count,
                        "exact_complete_candidate_count_satisfied": False,
                        "exact_successful_decoded_review_count_satisfied": False,
                        "passed": False,
                        "blocking_reason": (
                            "automatic_group_requires_exactly_eight_"
                            "complete_candidates"
                        ),
                        "source_index_mapping": _source_index_mapping(
                            request.sources,
                            eligible_indices,
                        ),
                        "nested_index_domains": (
                            _rendered_candidate_selection_index_domains()
                        ),
                        "candidate_pool": (
                            "stage_edit_generation_only"
                            if stage_edit_visual_candidate_batch is not None
                            else "ordinary_visual_baseline"
                        ),
                        "visual_candidate_set": (
                            visual_candidate_set.to_dict()
                            if visual_candidate_set is not None
                            else None
                        ),
                        "stage_edit_visual_candidate_batch": (
                            stage_edit_visual_candidate_batch.to_dict()
                            if stage_edit_visual_candidate_batch is not None
                            else None
                        ),
                        "silent_baseline_fallback_allowed": False,
                        "candidate_derivation_failures": (
                            candidate_derivation_failures
                        ),
                        "candidate_post_repair_review_rejections": (
                            candidate_post_repair_review_rejections
                        ),
                        "complete_candidates": [
                            item.to_dict()
                            for item in complete_group_candidates
                        ],
                        "candidates": [],
                    }
                    if stage_edit_candidate_rerank_result is not None:
                        rendered_candidate_selection_payload[
                            "stage_edit_candidate_rerank"
                        ] = stage_edit_candidate_rerank_result.to_dict()
                    rendered_candidate_selection_path = (
                        request.output_directory
                        / (
                            f"{base_filename}-"
                            "rendered-candidate-selection.json"
                        )
                    )
                    rendered_candidate_selection_path.write_text(
                        json.dumps(
                            rendered_candidate_selection_payload,
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    raise RuntimeError(
                        "automatic GROUP full render requires exactly 8 "
                        "complete candidates; got "
                        f"{complete_candidate_count}; report: "
                        f"{rendered_candidate_selection_path}"
                    )
                reporter.log(
                    "시각 후보 "
                    f"{len(materialized_group_candidates)}개 중 완전 렌더 후보 "
                    f"{len(complete_group_candidates)}개 파생"
                )
            else:
                visual_timeline = build_visual_timeline(
                    **visual_builder_arguments,
                )
                reporter.log(
                    f"프레임 품질·구도·움직임 전역 선택 점수 "
                    f"{visual_timeline.optimizer_score:.3f}"
                )
        except (ValueError, cv2.error) as exc:
            warnings.append(
                f"프레임 시각 점수를 확정하지 못해 리듬 타임라인을 사용함: {exc}"
            )
        if visual_timeline is not None:
            reporter.status("선택된 컷을 앞뒤 3프레임에서 정밀 보정하는 중…")
            try:
                visual_boundary_repairs = repair_visual_timeline_boundaries(
                    visual_timeline=visual_timeline,
                    sources=tuple(
                        request.sources[index] for index in eligible_indices
                    ),
                    audio_offsets=offsets,
                    scene_cut_times=scene_cut_times,
                    beat_period_seconds=group_draft.beat_period_seconds,
                    source_timewarps=eligible_timewarps,
                )
                visual_timeline = visual_boundary_repairs.timeline
                reporter.log(
                    "교차소스 경계 "
                    f"{visual_boundary_repairs.reviewed_cross_source_count}개 검토 · "
                    f"{visual_boundary_repairs.changed_count}개 컷시점 보정 · "
                    f"{len(visual_boundary_repairs.failures)}개 원위치 유지"
                )
            except (ValueError, cv2.error) as exc:
                warnings.append(
                    "국소 컷시점 보정에 실패해 전역 선택 경계를 유지함: "
                    f"{exc}"
                )
        if visual_timeline is not None:
            timeline_review = _review_visual_timeline_for_render(
                visual_timeline,
                available_source_count=len(eligible_indices),
                reporter=reporter,
                warnings=warnings,
                comparison_render_allowed=(
                    request.source_mode is not SourceMode.AUTOMATIC
                    or allow_failed_comparison_render
                ),
            )
        if (
            visual_candidate_set is None
            and stage_edit_visual_candidate_batch is None
        ):
            _require_reviewed_automatic_group_timeline(
                request,
                visual_timeline=visual_timeline,
                timeline_review=timeline_review,
                allow_failed_comparison_render=allow_failed_comparison_render,
            )
        if visual_timeline is not None:
            try:
                if any(value <= 0.0 for value in source_aspect_ratios):
                    raise ValueError(
                        "얼굴 맞춤 크롭을 위한 원본 해상도를 확인할 수 없음"
                    )
                face_match_plan = build_face_match_plan(
                    visual_timeline=visual_timeline,
                    source_aspect_ratios=source_aspect_ratios,
                )
                reporter.log(
                    "얼굴 크기·위치 맞춤 "
                    f"{face_match_plan.applied_count}개 적용 · "
                    f"{face_match_plan.fallback_count}개 원본 화면 유지"
                )
            except (ValueError, cv2.error) as exc:
                warnings.append(
                    "얼굴 맞춤을 안전하게 확정하지 못해 "
                    f"원본 화면 컷을 사용함: {exc}"
                )
            try:
                group_transition_plan = build_group_transition_plan(
                    visual_timeline=visual_timeline,
                    face_match_plan=face_match_plan,
                    source_durations_seconds=local_durations,
                )
                tier_counts = {
                    tier: sum(
                        transition.tier == tier
                        for transition in group_transition_plan.transitions
                    )
                    for tier in ("A", "B", "C", "D")
                }
                reporter.log(
                    "전환 안전 등급 "
                    + " · ".join(
                        f"{tier} {tier_counts[tier]}개"
                        for tier in ("A", "B", "C", "D")
                    )
                )
            except ValueError as exc:
                warnings.append(
                    "전환 안전 등급을 확정하지 못해 모든 경계를 "
                    f"하드 컷으로 유지함: {exc}"
                )
            if group_transition_plan is None:
                group_transition_plan = GroupTransitionPlan(
                    tuple(
                        RenderTransition(
                            tier="C",
                            reason=(
                                "rhythm_aligned_hard_cut_visual_transition_"
                                "analysis_unavailable"
                            ),
                        )
                        for _ in range(max(0, len(visual_timeline.choices) - 1))
                    ),
                    analysis_basis="rhythm_only_hard_cut_fallback_v1",
                )
            source_aspects_are_editorial_safe = all(
                abs(value - 16.0 / 9.0) <= 0.03
                for value in source_aspect_ratios
            ) and len(source_aspect_ratios) == len(eligible_indices)
            timeline_is_editorial_safe = bool(
                timeline_review is not None and timeline_review.passed
            )
            if (
                request.aspect_ratio is AspectRatio.LANDSCAPE
                and source_aspects_are_editorial_safe
                and timeline_is_editorial_safe
            ):
                try:
                    reporter.status(
                        "실제 컷 프레임의 특징점과 피사체 위치를 맞추는 중…"
                    )
                    boundary_affine_plan = build_boundary_affine_plan(
                        visual_timeline=visual_timeline,
                        sources=tuple(
                            request.sources[index] for index in eligible_indices
                        ),
                        transitions=group_transition_plan.transitions,
                        face_match_plan=face_match_plan,
                    )
                    reporter.log(
                        "실프레임 affine 경계 "
                        f"{boundary_affine_plan.analyzed_count}개 분석 · "
                        f"{boundary_affine_plan.eligible_count}개 안전 후보"
                    )
                except (ValueError, cv2.error) as exc:
                    warnings.append(
                        "실제 컷 프레임의 affine 보정을 확정하지 못해 "
                        f"피사체 크기·위치 기반 보정만 사용함: {exc}"
                    )
                try:
                    reporter.status(
                        "컷 전후 14프레임의 피사체 크기·위치·회전을 검증하는 중…"
                    )
                    editorial_width, editorial_height = output_size(
                        request.aspect_ratio
                    )
                    boundary_subject_alignment_plan = (
                        build_boundary_subject_alignment_plan(
                            visual_timeline=visual_timeline,
                            sources=tuple(
                                request.sources[index]
                                for index in eligible_indices
                            ),
                            transitions=group_transition_plan.transitions,
                            face_match_plan=face_match_plan,
                            output_width=editorial_width,
                            output_height=editorial_height,
                        )
                    )
                    reporter.log(
                        "경계 국소 피사체 정렬 "
                        f"{len(boundary_subject_alignment_plan.decisions)}개 검토 · "
                        f"{boundary_subject_alignment_plan.eligible_count}개 전환 · "
                        f"{boundary_subject_alignment_plan.validated_output_frame_count}"
                        "프레임 전수 검증"
                    )
                except (OSError, TypeError, ValueError, cv2.error) as exc:
                    warnings.append(
                        "경계 국소 피사체 정렬을 전수 검증하지 못해 "
                        f"기존 보수적 편집 카메라만 사용함: {exc}"
                    )
                try:
                    editorial_motion_plan = build_editorial_motion_plan(
                        visual_timeline=visual_timeline,
                        transitions=group_transition_plan.transitions,
                        face_match_plan=face_match_plan,
                        boundary_affine_plan=boundary_affine_plan,
                        boundary_subject_alignment_plan=(
                            boundary_subject_alignment_plan
                        ),
                        phrase_times=phrase_times,
                        section_plan=section_plan,
                    )
                    reporter.log(
                        "편집 카메라 "
                        f"{editorial_motion_plan.applied_count}개 세그먼트 · "
                        f"크기맞춤 {editorial_motion_plan.boundary_match_count}개 · "
                        f"실프레임 {editorial_motion_plan.affine_boundary_match_count}개 · "
                        f"줌 {editorial_motion_plan.zoom_event_count}회 · "
                        f"팬 {editorial_motion_plan.pan_event_count}회 · "
                        f"미세회전 {editorial_motion_plan.roll_event_count}회"
                    )
                except ValueError as exc:
                    warnings.append(
                        "편집 카메라를 안전하게 확정하지 못해 기존 크롭만 "
                        f"사용함: {exc}"
                    )
            else:
                warnings.append(
                    "편집 카메라는 사전 QA를 통과한 16:9 가로 원본·출력에서만 "
                    "활성화하며 현재 요청은 기존 프레이밍을 유지했습니다."
                )
        width, height = output_size(request.aspect_ratio)
        rendered_video_path = request.output_directory / (
            f"{base_filename}-group-crosscut.mp4"
        )
        render_plan_path = request.output_directory / (
            f"{base_filename}-render-plan.json"
        )
        if (
            visual_candidate_set is not None
            or stage_edit_visual_candidate_batch is not None
        ):
            reporter.status(
                "완전 편집 후보 8개를 480p로 순차 렌더·디코딩하는 중…"
            )
            rendered_selection = _review_complete_group_candidates(
                complete_group_candidates,
                scratch_directory=request.output_directory,
                professional_reference_profile=(
                    professional_reference_profile
                ),
                professional_reference_cadence_profile=(
                    professional_reference_cadence_profile
                ),
                learned_reviewer_session=learned_reviewer_session,
                stage_edit_montage_section_plan=(
                    stage_edit_generation_result.learned_section_director.section_plan
                    if stage_edit_generation_result is not None
                    and stage_edit_generation_result.learned_section_director
                    is not None
                    else None
                ),
                stage_edit_source_derivative_diversity=(
                    source_derivative_diversity_result
                    if stage_edit_visual_candidate_batch is not None
                    else None
                ),
                stage_edit_source_index_mapping=(
                    _source_index_mapping(request.sources, eligible_indices)
                    if stage_edit_visual_candidate_batch is not None
                    else None
                ),
                stage_edit_source_scene_cut_times=(
                    tuple(
                        tuple(
                            float(change["timestamp_seconds"])
                            for change in payload["changes"]
                        )
                        for payload in scene_payloads
                    )
                    if stage_edit_visual_candidate_batch is not None
                    else None
                ),
                stage_edit_montage_config=(
                    rendered_montage_config
                    if stage_edit_visual_candidate_batch is not None
                    else None
                ),
                stage_edit_montage_reference_profile=(
                    stage_edit_montage_policy.profile_identity
                    if stage_edit_visual_candidate_batch is not None
                    and stage_edit_montage_policy is not None
                    else None
                ),
                progress_callback=lambda position, total, candidate_index: (
                    reporter.progress(
                        0.979 + 0.006 * position / max(1, total),
                        (
                            "실렌더 후보 검수 중… "
                            f"{position}/{total} (후보 {candidate_index})"
                        ),
                    )
                ),
            )
            rendered_candidate_selection_payload = rendered_selection.to_dict()
            rendered_candidate_selection_payload.update(
                {
                    "candidate_pool": (
                        "stage_edit_generation_only"
                        if stage_edit_visual_candidate_batch is not None
                        else "ordinary_visual_baseline"
                    ),
                    "visual_candidate_set": (
                        visual_candidate_set.to_dict()
                        if visual_candidate_set is not None
                        else None
                    ),
                    "stage_edit_visual_candidate_batch": (
                        stage_edit_visual_candidate_batch.to_dict()
                        if stage_edit_visual_candidate_batch is not None
                        else None
                    ),
                    "silent_baseline_fallback_allowed": False,
                    "source_index_mapping": _source_index_mapping(
                        request.sources,
                        eligible_indices,
                    ),
                    "nested_index_domains": (
                        _rendered_candidate_selection_index_domains()
                    ),
                    "candidate_derivation_failures": (
                        candidate_derivation_failures
                    ),
                    "candidate_post_repair_review_rejections": (
                        candidate_post_repair_review_rejections
                    ),
                    "complete_candidates": [
                        item.to_dict() for item in complete_group_candidates
                    ],
                    "complete_candidate_count": len(
                        complete_group_candidates
                    ),
                    "structure_passing_materialized_count": (
                        pre_repair_structure_passing_materialized_count
                    ),
                    "total_materialized_candidate_count": (
                        total_materialized_candidate_count
                    ),
                    "repair_eligible_candidate_count": (
                        repair_eligible_candidate_count
                    ),
                    "repair_eligible_materialized_count": (
                        repair_eligible_materialized_count
                    ),
                    "repair_eligible_attempted_count": (
                        repair_eligible_attempted_count
                    ),
                    "post_repair_structure_passing_complete_count": len(
                        complete_group_candidates
                    ),
                    "successful_review_count": (
                        rendered_selection.successful_review_count
                    ),
                    "exact_successful_decoded_review_count_satisfied": (
                        rendered_selection.successful_review_count == 8
                    ),
                    "exact_complete_candidate_count_satisfied": (
                        len(complete_group_candidates) == 8
                        and len(rendered_selection.reviews) == 8
                        and rendered_selection.successful_review_count == 8
                    ),
                    "professional_reference_profile": (
                        professional_reference_profile.provenance_dict()
                        if professional_reference_profile is not None
                        else None
                    ),
                }
            )
            if stage_edit_candidate_rerank_result is not None:
                rendered_candidate_selection_payload[
                    "stage_edit_candidate_rerank"
                ] = stage_edit_candidate_rerank_result.to_dict()
            selected_bundle = next(
                item
                for item in complete_group_candidates
                if item.candidate_index
                == rendered_selection.selected_candidate_index
            )
            selected_materialized = next(
                item
                for item in materialized_group_candidates
                if item.candidate_index
                == rendered_selection.selected_candidate_index
            )
            rendered_proxy_review_payload = deepcopy(
                rendered_selection.selected.review_payload
            )
            learned_issue_boundary_indices: tuple[int, ...] | None = None
            learned_repair_issues: tuple[Mapping[str, Any], ...] = ()
            if learned_reviewer_session is not None:
                learned_rerank = rendered_selection.learned_reviewer_rerank
                if not isinstance(learned_rerank, dict):
                    raise RuntimeError(
                        "configured learned reviewer did not produce rerank audit"
                    )
                raw_learned_issue_indices = learned_rerank.get(
                    "selected_repair_issue_boundary_indices"
                )
                if (
                    not isinstance(raw_learned_issue_indices, list)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                        for value in raw_learned_issue_indices
                    )
                    or len(set(raw_learned_issue_indices))
                    != len(raw_learned_issue_indices)
                ):
                    raise RuntimeError(
                        "configured learned reviewer produced malformed repair "
                        "issue boundary indices"
                    )
                learned_issue_boundary_indices = tuple(
                    raw_learned_issue_indices
                )
                raw_learned_repair_issues = learned_rerank.get(
                    "selected_repair_issues"
                )
                if (
                    not isinstance(raw_learned_repair_issues, list)
                    or any(
                        not isinstance(value, Mapping)
                        for value in raw_learned_repair_issues
                    )
                    or len(raw_learned_repair_issues)
                    != len(learned_issue_boundary_indices)
                ):
                    raise RuntimeError(
                        "configured learned reviewer produced malformed repair "
                        "issue windows"
                    )
                try:
                    learned_repair_issues = tuple(
                        StageEditRepairIssue.from_mapping(value).to_dict()
                        for value in raw_learned_repair_issues
                    )
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "configured learned reviewer produced invalid repair "
                        "issue windows"
                    ) from exc
            if stage_edit_visual_candidate_batch is not None:
                selected_bundle = replace(
                    selected_bundle,
                    candidate_lineage=selected_bundle.candidate_lineage
                    + (
                        {
                            "stage": "decoded_proxy_candidate_selection",
                            "selected_candidate_index": (
                                rendered_selection.selected_candidate_index
                            ),
                            "decoded_proxy_review_available": (
                                rendered_proxy_review_payload is not None
                            ),
                            "decoded_proxy_passed_before_repair": bool(
                                rendered_proxy_review_payload is not None
                                and rendered_proxy_review_payload.get("passed")
                            ),
                            "learned_reviewer_configured": (
                                learned_reviewer_session is not None
                            ),
                            "learned_reviewer_rerank_recorded_in_selection_audit": (
                                learned_reviewer_session is not None
                            ),
                            "learned_reviewer_repair_issue_boundary_indices": list(
                                learned_issue_boundary_indices or ()
                            ),
                            "deterministic_hard_gates_remain_authority": True,
                        },
                    ),
                )
            if stage_edit_visual_candidate_batch is not None:
                if (
                    stage_edit_generation_result is None
                    or section_plan is None
                ):
                    raise RuntimeError(
                        "StageEdit windowed beam repair requires retained "
                        "generation evidence and a section plan"
                    )
                partial_reedit = _attempt_stage_edit_windowed_beam_reedit(
                    selected_bundle,
                    selected_materialized,
                    rendered_proxy_review_payload,
                    bridge_result=stage_edit_generation_result,
                    learned_repair_issues=learned_repair_issues,
                    request=request,
                    eligible_indices=eligible_indices,
                    audio_offsets=offsets,
                    sync_scores=sync_scores,
                    scene_cut_times=scene_cut_times,
                    source_timewarps=eligible_timewarps,
                    source_aspect_ratios=source_aspect_ratios,
                    source_durations_seconds=local_durations,
                    beat_period_seconds=group_draft.beat_period_seconds,
                    phrase_times=phrase_times,
                    section_plan=(
                        stage_edit_generation_result.learned_section_director.section_plan
                        if stage_edit_generation_result.learned_section_director
                        is not None
                        else section_plan
                    ),
                    common_master_start=common_start,
                    width=width,
                    height=height,
                    scratch_directory=request.output_directory,
                    source_look_match=source_look_match_result,
                    boundary_look_measurement_cache=(
                        boundary_look_measurement_cache
                    ),
                    professional_reference_profile=(
                        professional_reference_profile
                    ),
                    professional_reference_cadence_profile=(
                        professional_reference_cadence_profile
                    ),
                    learned_reviewer_session=learned_reviewer_session,
                    motion_authority_v3=(
                        registered_stage_edit_motion_authority_v3
                    ),
                )
            else:
                partial_reedit = _attempt_proxy_triggered_partial_reedit(
                    selected_bundle,
                    selected_materialized,
                    rendered_proxy_review_payload,
                    request=request,
                    eligible_indices=eligible_indices,
                    audio_offsets=offsets,
                    scene_cut_times=scene_cut_times,
                    source_timewarps=eligible_timewarps,
                    source_aspect_ratios=source_aspect_ratios,
                    source_durations_seconds=local_durations,
                    beat_period_seconds=group_draft.beat_period_seconds,
                    phrase_times=phrase_times,
                    section_plan=section_plan,
                    common_master_start=common_start,
                    width=width,
                    height=height,
                    scratch_directory=request.output_directory,
                    source_look_match=source_look_match_result,
                    boundary_look_measurement_cache=(
                        boundary_look_measurement_cache
                    ),
                    professional_reference_profile=(
                        professional_reference_profile
                    ),
                    learned_issue_boundary_indices=(
                        learned_issue_boundary_indices
                    ),
                )
            selected_bundle = partial_reedit.selected_bundle
            if stage_edit_visual_candidate_batch is not None:
                # The final professional-boundary evidence hashes the complete
                # RenderPlan.  Candidate proxies still point at hidden scratch
                # outputs, so bind the promoted bundle to the real delivery
                # path before the final independent montage review.  Replacing
                # the path only after that review would serialize a valid hash
                # for a plan that is not the one ultimately written/rendered.
                selected_bundle = replace(
                    selected_bundle,
                    render_plan=replace(
                        selected_bundle.render_plan,
                        output=rendered_video_path,
                    ),
                )
                selected_bundle = (
                    _refresh_stage_edit_dynamic_crop_motion_evidence_v3(
                        selected_bundle
                    )
                )
            rendered_proxy_review_payload = deepcopy(
                partial_reedit.selected_review_payload
            )
            rendered_candidate_selection_payload[
                "proxy_triggered_partial_reedit"
            ] = partial_reedit.audit_payload
            if stage_edit_visual_candidate_batch is not None:
                if (
                    rendered_proxy_review_payload is None
                    or stage_edit_generation_result is None
                    or stage_edit_generation_result.learned_section_director
                    is None
                    or source_derivative_diversity_result is None
                    or rendered_montage_config is None
                    or stage_edit_montage_policy is None
                ):
                    raise RuntimeError(
                        "final StageEdit selection lacks mandatory montage "
                        "grammar evidence"
                    )
                technical_proxy_passed = bool(
                    rendered_proxy_review_payload.get(
                        "technical_proxy_passed_before_montage_grammar",
                        rendered_proxy_review_payload.get("passed"),
                    )
                )
                (
                    rendered_montage_grammar_review_payload,
                    _final_montage_analysis,
                ) = _review_stage_edit_bundle_montage(
                    bundle=selected_bundle,
                    proxy_review_payload=rendered_proxy_review_payload,
                    section_plan=(
                        stage_edit_generation_result.learned_section_director.section_plan
                    ),
                    source_derivative_diversity=(
                        source_derivative_diversity_result
                    ),
                    source_index_mapping=_source_index_mapping(
                        request.sources,
                        eligible_indices,
                    ),
                    source_scene_cut_times=tuple(
                        tuple(
                            float(change["timestamp_seconds"])
                            for change in payload["changes"]
                        )
                        for payload in scene_payloads
                    ),
                    config=rendered_montage_config,
                    reference_profile=stage_edit_montage_policy.profile_identity,
                )
                stage_edit_exact_shot_clock_payload = deepcopy(
                    _final_montage_analysis["stage_edit_exact_shot_clock"]
                )
                stage_edit_dynamic_crop_version_payload = int(
                    _final_montage_analysis["dynamic_crop_version"]
                )
                raw_decodes_v3 = _final_montage_analysis[
                    "dynamic_crop_v3_decodes"
                ]
                raw_motion_v3 = _final_montage_analysis[
                    "dynamic_crop_motion_v3"
                ]
                dynamic_crop_v3_decodes_payload = (
                    deepcopy(raw_decodes_v3)
                    if isinstance(raw_decodes_v3, list)
                    else None
                )
                dynamic_crop_motion_v3_payload = (
                    deepcopy(raw_motion_v3)
                    if isinstance(raw_motion_v3, Mapping)
                    else None
                )
                if stage_edit_dynamic_crop_version_payload == 3 and (
                    dynamic_crop_v3_decodes_payload is None
                    or dynamic_crop_motion_v3_payload is None
                ):
                    raise RuntimeError(
                        "final Dynamic Crop v3 montage analysis lost replay evidence"
                    )
                rendered_proxy_review_payload[
                    "technical_proxy_passed_before_montage_grammar"
                ] = technical_proxy_passed
                rendered_proxy_review_payload[
                    "rendered_montage_grammar_review"
                ] = deepcopy(rendered_montage_grammar_review_payload)
                rendered_proxy_review_payload["passed"] = bool(
                    technical_proxy_passed
                    and rendered_montage_grammar_review_payload.get("passed")
                    is True
                )
                _apply_bounded_human_target_proxy_rank(
                    rendered_proxy_review_payload,
                    rendered_montage_grammar_review_payload,
                    technical_passed=technical_proxy_passed,
                )
            final_proxy_passed = bool(
                rendered_proxy_review_payload is not None
                and rendered_proxy_review_payload.get("passed")
            )
            rendered_candidate_selection_payload["final_selected_review"] = (
                deepcopy(rendered_proxy_review_payload)
            )
            rendered_candidate_selection_payload[
                "final_selected_version"
            ] = partial_reedit.audit_payload["selected_version"]
            rendered_candidate_selection_payload["passed"] = final_proxy_passed
            if stage_edit_visual_candidate_batch is not None:
                exact_complete_candidates_satisfied = bool(
                    rendered_candidate_selection_payload[
                        "exact_complete_candidate_count_satisfied"
                    ]
                )
                exact_decoded_reviews_satisfied = bool(
                    rendered_candidate_selection_payload[
                        "exact_successful_decoded_review_count_satisfied"
                    ]
                )
                stage_edit_promotion_eligible = bool(
                    exact_complete_candidates_satisfied
                    and exact_decoded_reviews_satisfied
                    and final_proxy_passed
                    and rendered_montage_grammar_review_payload is not None
                    and rendered_montage_grammar_review_payload.get("passed")
                    is True
                )
                selected_bundle = replace(
                    selected_bundle,
                    candidate_lineage=selected_bundle.candidate_lineage
                    + (
                        {
                            "stage": "selected_candidate_render_promotion",
                            "selected_version": partial_reedit.audit_payload[
                                "selected_version"
                            ],
                            "exact_eight_complete_candidates_satisfied": bool(
                                exact_complete_candidates_satisfied
                            ),
                            "exact_eight_decoded_proxy_reviews_satisfied": bool(
                                exact_decoded_reviews_satisfied
                            ),
                            "decoded_selected_proxy_passed": final_proxy_passed,
                            "rendered_montage_grammar_passed": bool(
                                rendered_montage_grammar_review_payload
                                is not None
                                and rendered_montage_grammar_review_payload.get(
                                    "passed"
                                )
                                is True
                            ),
                            "promotion_state": (
                                "eligible_for_final_render"
                                if stage_edit_promotion_eligible
                                else "blocked"
                            ),
                        },
                    ),
                )
                rendered_candidate_selection_payload[
                    "stage_edit_render_promotion"
                ] = {
                    "contract": STAGE_EDIT_VISUAL_CANDIDATE_CONTRACT,
                    "activation": (
                        "automatic_registered_production_bundle"
                        if registered_stage_edit_bundle is not None
                        else "explicit_stage_edit_generation_request_only"
                    ),
                    "candidate_pool": "stage_edit_generation_only",
                    "ordinary_visual_baseline_candidate_pool_eligible": False,
                    "silent_baseline_fallback_allowed": False,
                    "exact_eight_complete_candidates_required": True,
                    "exact_eight_decoded_proxy_reviews_required": True,
                    "deterministic_hard_gates_remain_authority": True,
                    "learned_reviewer_configured": (
                        learned_reviewer_session is not None
                    ),
                    "selected_candidate_index": selected_bundle.candidate_index,
                    "selected_candidate_origin": selected_bundle.candidate_origin,
                    "selected_candidate_lineage": [
                        dict(item) for item in selected_bundle.candidate_lineage
                    ],
                    "selected_version": partial_reedit.audit_payload[
                        "selected_version"
                    ],
                    "all_promotion_hard_gates_satisfied": (
                        stage_edit_promotion_eligible
                    ),
                    "decoded_selected_proxy_passed": final_proxy_passed,
                    "rendered_montage_grammar_required": True,
                    "rendered_montage_grammar_passed": bool(
                        rendered_montage_grammar_review_payload is not None
                        and rendered_montage_grammar_review_payload.get("passed")
                        is True
                    ),
                    "promotion_state": (
                        "eligible_for_final_render"
                        if stage_edit_promotion_eligible
                        else "blocked"
                    ),
                }
            if not bool(
                rendered_candidate_selection_payload[
                    "exact_complete_candidate_count_satisfied"
                ]
            ):
                rendered_candidate_selection_payload[
                    "decoded_selected_proxy_passed"
                ] = final_proxy_passed
                rendered_candidate_selection_payload["passed"] = False
                rendered_candidate_selection_payload["blocking_reason"] = (
                    (
                        "automatic_group_requires_exactly_eight_"
                        "complete_candidates"
                    )
                    if len(complete_group_candidates) != 8
                    else (
                        "automatic_group_requires_exactly_eight_"
                        "successful_decoded_candidate_reviews"
                    )
                )
            rendered_candidate_selection_path = request.output_directory / (
                f"{base_filename}-rendered-candidate-selection.json"
            )
            rendered_candidate_selection_path.write_text(
                json.dumps(
                    rendered_candidate_selection_payload,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            if not bool(
                rendered_candidate_selection_payload[
                    "exact_complete_candidate_count_satisfied"
                ]
            ):
                raise RuntimeError(
                    "automatic GROUP full render requires exactly 8 complete "
                    "candidates with 8 successful decoded reviews; got "
                    f"{len(complete_group_candidates)} complete / "
                    f"{rendered_selection.successful_review_count} decoded; "
                    f"report: {rendered_candidate_selection_path}"
                )
            visual_timeline = selected_bundle.visual_timeline
            visual_boundary_repairs = selected_bundle.boundary_repairs
            timeline_review = selected_bundle.timeline_review
            face_match_plan = selected_bundle.face_match_plan
            group_transition_plan = selected_bundle.group_transition_plan
            boundary_affine_plan = selected_bundle.boundary_affine_plan
            editorial_motion_plan = selected_bundle.editorial_motion_plan
            boundary_look_match_result = selected_bundle.boundary_look_match
            plan = replace(
                selected_bundle.render_plan,
                output=rendered_video_path,
            )
            segments = plan.segments
            selected_proxy_score = float(
                rendered_proxy_review_payload.get("weighted_score", -math.inf)
                if rendered_proxy_review_payload is not None
                else -math.inf
            )
            reporter.log(
                "실렌더 후보 선택 "
                f"{rendered_selection.selected_candidate_index}번 · "
                f"{len(rendered_selection.reviews)}개 실제 디코딩 · "
                f"점수 {selected_proxy_score:.2f}"
            )
            if not rendered_selection.minimum_candidate_count_satisfied:
                warnings.append(
                    "서로 다른 완전 렌더 후보가 8개보다 적어 가능한 후보 "
                    f"{len(rendered_selection.reviews)}개를 모두 실제 검수했습니다."
                )
            _require_reviewed_automatic_group_timeline(
                request,
                visual_timeline=visual_timeline,
                timeline_review=timeline_review,
                allow_failed_comparison_render=allow_failed_comparison_render,
            )
            if not final_proxy_passed:
                if (
                    stage_edit_visual_candidate_batch is not None
                    or not allow_failed_comparison_render
                ):
                    raise RuntimeError(
                        "automatic GROUP full render blocked by decoded proxy review; "
                        f"report: {rendered_candidate_selection_path}"
                    )
                warnings.append(
                    "선택된 실렌더 프록시 검수는 통과하지 못했지만 명시적인 "
                    "비교용 렌더 옵션으로 계속했습니다."
                )
        else:
            if visual_timeline is not None:
                segments = tuple(
                    replace(
                        aligned_render_segment(
                            source_index=eligible_indices[item.source_index],
                            reference_start=item.canonical_master_start,
                            common_master_start=common_start,
                            source_offset=offsets[item.source_index],
                            duration=item.duration_seconds,
                            timewarp=eligible_timewarps[item.source_index],
                            pan_crop=(
                                face_match_plan.pan_crop_for(item.segment_index)
                                if face_match_plan is not None
                                else None
                            ),
                            editorial_transform=(
                                editorial_motion_plan.track_for(item.segment_index)
                                if editorial_motion_plan is not None
                                else None
                            ),
                        ),
                        look_adjustment=(
                            source_look_match_result.adjustment_for_source(
                                item.source_index
                            )
                            if source_look_match_result is not None
                            else None
                        ),
                    )
                    for item in visual_timeline.choices
                )
            else:
                segments = tuple(
                    replace(
                        aligned_render_segment(
                            source_index=eligible_indices[item.source_index],
                            reference_start=item.canonical_master_start,
                            common_master_start=common_start,
                            source_offset=item.audio_offset_seconds,
                            duration=item.duration_seconds,
                            timewarp=eligible_timewarps[item.source_index],
                        ),
                        look_adjustment=(
                            source_look_match_result.adjustment_for_source(
                                item.source_index
                            )
                            if source_look_match_result is not None
                            else None
                        ),
                    )
                    for item in group_draft.segments
                )
            if group_transition_plan is None:
                group_transition_plan = GroupTransitionPlan(
                    tuple(
                        RenderTransition(
                            tier="C",
                            reason=(
                                "rhythm_aligned_hard_cut_visual_analysis_unavailable"
                            ),
                        )
                        for _ in range(max(0, len(segments) - 1))
                    ),
                    analysis_basis="rhythm_only_hard_cut_fallback_v1",
                )
            plan = RenderPlan(
                sources=request.sources,
                segments=segments,
                reference_source=request.reference_source,
                aspect_ratio=request.aspect_ratio,
                output=rendered_video_path,
                master_audio_start=common_start,
                width=width,
                height=height,
                transitions=group_transition_plan.transitions,
            )
            if review_rendered_proxies:
                reporter.status(
                    "480p 후보 영상을 실제 렌더해 컷 전후를 검수하는 중…"
                )
                with tempfile.TemporaryDirectory(
                    prefix=".ive-crosscut-proxy-",
                    dir=request.output_directory,
                ) as temporary_directory:
                    _proxy_plan, proxy_review = render_and_review_proxy(
                        plan,
                        Path(temporary_directory) / "candidate-00.mp4",
                        professional_reference_profile=(
                            professional_reference_profile
                        ),
                    )
                    rendered_proxy_review_payload = deepcopy(proxy_review.payload)
                proxy_metadata = rendered_proxy_review_payload.get("proxy")
                if isinstance(proxy_metadata, dict):
                    proxy_metadata["path"] = None
                    proxy_metadata["temporary_proxy_removed_after_review"] = True
                rendered_proxy_review_path = request.output_directory / (
                    f"{base_filename}-rendered-proxy-review.json"
                )
                rendered_proxy_review_path.write_text(
                    json.dumps(
                        rendered_proxy_review_payload,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                reporter.log(
                    "실렌더 프록시 검수 점수 "
                    f"{float(rendered_proxy_review_payload['weighted_score']):.2f} · "
                    f"문제 경계 "
                    f"{len(rendered_proxy_review_payload['boundary_issue_indices'])}개"
                )
                if not bool(rendered_proxy_review_payload["passed"]):
                    if not allow_failed_comparison_render:
                        raise RuntimeError(
                            "automatic GROUP full render blocked by decoded proxy review; "
                            f"report: {rendered_proxy_review_path}"
                        )
                    warnings.append(
                        "실렌더 프록시 검수는 통과하지 못했지만 명시적인 비교용 "
                        "렌더 옵션으로 계속했습니다."
                    )
        if registered_stage_edit_bundle is not None:
            reporter.status(
                "등록된 StageEdit 최종 컷을 실제 PTS·포즈·"
                "바디 모션으로 로컬 동기 검증 중…"
            )
            try:
                cut_local_sync_payload = analyze_cut_local_sync(
                    plan,
                    source_timewarps=source_timewarps,
                    timewarp_payloads=local_timewarp_payloads,
                    pts_cache_directory=request.source_cache_directory,
                )
                reporter.log(
                    "보존 교차소스 경계 "
                    f"{cut_local_sync_payload['retained_cross_source_boundary_count']}개 · "
                    f"측정 {cut_local_sync_payload['measured_boundary_count']}개 · "
                    f"불확실 {cut_local_sync_payload['unknown_boundary_count']}개 · "
                    "로컬 잔차 p95 "
                    f"{cut_local_sync_payload['residual_p95_frames']} frame"
                )
            except (cv2.error, OSError, RuntimeError, TypeError, ValueError) as exc:
                cut_local_sync_payload = {
                    "schema_version": 2,
                    "analysis": "actual_pts_renderer_bound_retained_cut_pose_body_motion_phase_v2",
                    "activation": "automatic_registered_stage_edit_group",
                    "required": True,
                    "passed": False,
                    "error_code": (
                        "cut_local_sync_analysis_failed_"
                        f"{type(exc).__name__}"
                    ),
                    "unknown_reason_counts": {
                        "cut_local_sync_analysis_failed": 1,
                    },
                    "serialization_policy": {
                        "raw_frames_persisted": False,
                        "pose_embeddings_persisted": False,
                        "formation_descriptors_persisted": False,
                        "body_motion_series_persisted": False,
                        "sample_pixels_persisted": False,
                        "motion_timing_provenance_persisted": True,
                        "scalar_aggregate_and_hash_provenance_only": True,
                    },
                }
                reporter.emit(
                    "log",
                    level="warning",
                    message=(
                        "등록된 StageEdit 컷-로컬 동기 증거를 "
                        "완성하지 못해 최종 QA에서 차단됩니다 "
                        f"({type(exc).__name__})."
                    ),
                )
            if not bool(cut_local_sync_payload.get("passed")):
                raise RuntimeError(
                    "automatic GROUP cut-local sync proof rejected before "
                    "render: "
                    f"unknown={cut_local_sync_payload.get('unknown_boundary_count')}; "
                    f"over_threshold={cut_local_sync_payload.get('threshold_exceeding_boundary_count')}; "
                    f"reasons={cut_local_sync_payload.get('unknown_reason_counts')}"
                )
        write_plan(plan, render_plan_path)
        reporter.status("연속 기준 음원으로 교차편집 영상을 합성하는 중…")
        codec = render(plan)
        timeline_interchange_paths = write_timeline_interchange_bundle(
            plan,
            output_directory=request.output_directory,
            base_filename=f"{base_filename}-group-timeline",
            project_name=f"IVE {request.song} group crosscut",
        )
        render_payload = {
            "video_path": str(rendered_video_path),
            "plan_path": str(render_plan_path),
            "video_codec": codec,
            "interchange": {
                "fcpxml_path": str(timeline_interchange_paths[0]),
                "otio_path": str(timeline_interchange_paths[1]),
                "edl_path": str(timeline_interchange_paths[2]),
                "fidelity_audit_path": str(timeline_interchange_paths[3]),
                "authoritative_pixels": str(rendered_video_path),
            },
            "allow_failed_comparison_render": allow_failed_comparison_render,
            "selection_basis": (
                visual_timeline.analysis_basis
                if visual_timeline is not None
                else group_draft.selection_basis
            ),
            "candidate_origin": (
                "explicit_stage_edit_generation"
                if stage_edit_visual_candidate_batch is not None
                else "ordinary_visual_baseline"
            ),
            "stage_edit_promotion_contract": (
                STAGE_EDIT_VISUAL_CANDIDATE_CONTRACT
                if stage_edit_visual_candidate_batch is not None
                else None
            ),
            "source_clock_basis": (
                visual_timeline.source_clock_basis
                if visual_timeline is not None
                else (
                    "local_monotonic_timewarp_with_per_source_scalar_fallback"
                    if any(
                        timewarp is not None and not timewarp.scalar_fallback
                        for timewarp in eligible_timewarps
                    )
                    else "global_scalar_offset"
                )
            ),
            "cut_local_sync": cut_local_sync_payload,
            "source_look_match": (
                _with_index_domain(
                    source_look_match_result.to_dict(),
                    ELIGIBLE_SOURCE_INDEX_DOMAIN,
                )
                if source_look_match_result is not None
                else None
            ),
            "boundary_look_match": (
                boundary_look_match_result.to_dict()
                if boundary_look_match_result is not None
                else None
            ),
            "face_match": (
                {
                    "analysis_basis": face_match_plan.analysis_basis,
                    "applied_count": face_match_plan.applied_count,
                    "fallback_count": face_match_plan.fallback_count,
                    "count_basis": "selected_segments",
                    "aligned_segment_count": (
                        face_match_plan.aligned_segment_count
                    ),
                    "raw_segment_count": face_match_plan.raw_segment_count,
                }
                if face_match_plan is not None
                else None
            ),
            "group_transitions": (
                group_transition_plan.to_dict(plan.fps)
                if group_transition_plan is not None
                else None
            ),
            "editorial_motion": (
                editorial_motion_plan.to_dict()
                if editorial_motion_plan is not None
                else None
            ),
            "boundary_affine": (
                boundary_affine_plan.to_dict()
                if boundary_affine_plan is not None
                else None
            ),
            "final_render_segments": _final_render_segments_payload(
                segments,
                common_master_start=common_start,
            ),
            "boundary_repairs": (
                visual_boundary_repairs.to_dict()
                if visual_boundary_repairs is not None
                else None
            ),
            "timeline_review": (
                _with_index_domain(
                    timeline_review.to_dict(),
                    ELIGIBLE_SOURCE_INDEX_DOMAIN,
                )
                if timeline_review is not None
                else None
            ),
            "rendered_proxy_review": rendered_proxy_review_payload,
            "rendered_montage_grammar_review": (
                deepcopy(rendered_montage_grammar_review_payload)
                if rendered_montage_grammar_review_payload is not None
                else None
            ),
            "stage_edit_exact_shot_clock": (
                deepcopy(stage_edit_exact_shot_clock_payload)
                if stage_edit_exact_shot_clock_payload is not None
                else None
            ),
            "dynamic_crop_version": stage_edit_dynamic_crop_version_payload,
            "dynamic_crop_v3_decodes": (
                deepcopy(dynamic_crop_v3_decodes_payload)
                if dynamic_crop_v3_decodes_payload is not None
                else None
            ),
            "dynamic_crop_motion_v3": (
                deepcopy(dynamic_crop_motion_v3_payload)
                if dynamic_crop_motion_v3_payload is not None
                else None
            ),
            "rendered_candidate_selection": (
                rendered_candidate_selection_payload
            ),
        }
        reporter.progress(0.99, "교차편집 영상 합성 완료")
    elif request.edit_mode is EditMode.GROUP:
        warnings.append("안전하게 정렬된 소스 2개와 박자 분석이 필요해 영상을 만들지 않았습니다.")

    member_range_ready = (
        use_direct_member_full_frame
        or (
            member_anchor_trim_payload is not None
            and bool(member_anchor_trim_payload["usable_for_render"])
        )
    )
    if request.edit_mode is EditMode.MEMBER and group_ready and member_range_ready:
        assert request.target_member is not None
        reporter.progress(0.98, "멤버 교차직캠 타임라인을 만드는 중…")
        offsets = tuple(
            alignment.alignments[index].offset_seconds for index in eligible_indices
        )
        sync_scores = tuple(
            alignment.alignments[index].score for index in eligible_indices
        )
        scene_cut_times = tuple(
            tuple(
                float(change["timestamp_seconds"])
                for change in scene_payloads[index]["changes"]
            )
            for index in eligible_indices
        )
        phrase_times = tuple(
            sorted(
                {
                    float(time_value)
                    for phrase in rhythm_analysis.phrase_candidates
                    for time_value in (phrase.start_seconds, phrase.end_seconds)
                }
            )
        )
        timing_draft = build_group_draft(
            source_count=len(eligible_indices),
            common_master_start=member_common_start,
            common_master_end=common_end,
            audio_offsets=offsets,
            sync_scores=sync_scores,
            beat_times=rhythm_analysis.beat_times_seconds,
            phrase_times=phrase_times,
            scene_cut_times=scene_cut_times,
        )
        aspect_ratios: list[float] = []
        for index in eligible_indices:
            video_stream = media[index].video_streams[0]
            if not video_stream.width or not video_stream.height:
                raise ValueError("멤버 크롭을 위한 원본 영상 해상도를 확인할 수 없습니다.")
            aspect_ratios.append(video_stream.width / video_stream.height)
        source_aspect_ratio = aspect_ratios[0]
        if (
            not use_direct_member_full_frame
            and any(
                abs(value - source_aspect_ratio) > 0.03
                for value in aspect_ratios[1:]
            )
        ):
            raise ValueError("멤버 모드 원본 영상들의 화면 비율이 서로 다릅니다.")
        direct_segment_details: tuple[dict[str, Any], ...] = ()
        direct_cadence: dict[str, Any] | None = None
        direct_visual_selection: DirectMemberVisualSelection | None = None
        full_frame_fallbacks: tuple[dict[str, Any], ...] = ()
        can_render_member = True
        if use_direct_member_full_frame:
            reporter.status(
                "직캠별 동작·위치 연속성과 아웃트로 유효성을 비교하는 중…"
            )
            direct_highlight_intervals: tuple[tuple[float, float], ...] = ()
            try:
                direct_section_plan = build_section_plan(
                    rhythm_analysis,
                    start_seconds=member_common_start,
                    end_seconds=common_end,
                )
                direct_highlight_intervals = tuple(
                    (section.start_seconds, section.end_seconds)
                    for section in direct_section_plan.sections
                    if section.role
                    in {SectionRole.BUILD, SectionRole.CHORUS_LIKE}
                    and section.camera_motion_budget >= 0.50
                )
            except (AttributeError, TypeError, ValueError, FloatingPointError) as exc:
                reporter.emit(
                    "log",
                    level="warning",
                    message=(
                        "직캠 디졸브 하이라이트 우선순위를 확정하지 못해 "
                        f"동작·포즈 근거만 사용: {exc}"
                    ),
                )

            def _report_direct_partition_progress(
                payload: dict[str, Any],
            ) -> None:
                reporter.emit("direct_member_partition_profile", **payload)
                phase = payload.get("phase")
                if phase == "started":
                    reporter.status(
                        "직캠 분할 프로필 "
                        f"{payload['evaluation_index']}/"
                        f"{payload['evaluation_limit']} 평가 중 · "
                        f"이동 경계 {payload['shift_depth']}개 · "
                        f"서명 {str(payload['partition_signature_sha256'])[:10]}"
                    )
                elif phase == "completed":
                    reporter.log(
                        "직캠 분할 프로필 "
                        f"{payload['evaluation_index']} "
                        f"{payload['result']} · "
                        "부모 병목 세그먼트 "
                        f"{payload.get('parent_failed_segment_index')} · "
                        "다음 경계 "
                        f"{payload.get('next_proven_frontier_boundary_indices', [])}"
                    )
                elif phase == "search_limit_extended":
                    reporter.log(
                        "직캠 분할 검색 범위 확장 · "
                        f"epoch {payload['epoch']}/"
                        f"{payload['max_epochs']} · "
                        f"exact queue {payload['queued_unique_profile_count']}개 · "
                        f"상한 {payload['new_limit']}"
                    )
                elif phase == "search_completed":
                    reporter.log(
                        "직캠 분할 검색 완료 · "
                        f"{payload['evaluated_profile_count']}/"
                        f"{payload['evaluation_limit']} 평가 · "
                        f"결과 {payload['result']}"
                    )
            try:
                (
                    segments,
                    member_transitions,
                    direct_segment_details,
                    direct_cadence,
                    direct_visual_selection,
                ) = _official_target_fancam_direct_full_frame_segments(
                    timing_draft=timing_draft,
                    eligible_indices=eligible_indices,
                    eligible_sources=tuple(
                        request.sources[index] for index in eligible_indices
                    ),
                    eligible_source_aspect_ratios=(
                        direct_full_frame_ratios
                    ),
                    eligible_source_frame_sizes=tuple(
                        (
                            int(media[index].video_streams[0].width),
                            int(media[index].video_streams[0].height),
                        )
                        for index in eligible_indices
                    ),
                    output_frame_size=output_size(request.aspect_ratio),
                    common_master_start=member_common_start,
                    source_timewarps=source_timewarps,
                    scene_cut_times=scene_cut_times,
                    feature_cache_directory=request.source_cache_directory,
                    eligible_source_durations_seconds=tuple(
                        float(
                            media[index].video_streams[0].duration_seconds
                            or media[index].duration_seconds
                            or 0.0
                        )
                        for index in eligible_indices
                    ),
                    source_metadata_contract=(
                        OFFICIAL_TARGET_MEMBER_SOURCE_METADATA_CONTRACT
                    ),
                    highlight_intervals=direct_highlight_intervals,
                    member_pose_dissolve_enabled=bool(
                        getattr(
                            request,
                            "member_pose_dissolve_enabled",
                            True,
                        )
                    ),
                    output_fps=29.97,
                    partition_progress_callback=(
                        _report_direct_partition_progress
                    ),
                )
            except DirectMemberVisualSelectionError as exc:
                raise _report_direct_member_selection_failure(
                    exc,
                    output_directory=request.output_directory,
                    base_filename=base_filename,
                    target_member=request.target_member.value,
                    eligible_indices=eligible_indices,
                    reporter=reporter,
                ) from exc
            visual_aggregate = direct_visual_selection.to_dict()["aggregate"]
            reporter.log(
                "비신원 시각 경계 선택 "
                f"{visual_aggregate['selected_boundary_count']}개 · "
                "시각 경고 "
                f"{visual_aggregate['warning_boundary_count']}개 · "
                "콘텐츠 제외 후보 "
                f"{visual_aggregate['content_validity']['invalid_candidate_count']}개"
            )
            dissolve_payload = direct_cadence["pose_matched_dissolves"]
            reporter.log(
                "직캠 포즈 연결 디졸브 "
                f"{dissolve_payload['selected_dissolve_count']}개 / "
                f"안전 후보 {dissolve_payload['eligible_candidate_count']}개 · "
                "하이라이트 "
                f"{dissolve_payload['selected_highlight_count']}개 · "
                "동작 연결 "
                f"{dissolve_payload['selected_both_moving_count']}개"
            )
            dense_face_provider: DirectMemberDenseFaceProvider | None = None
            direct_motion_error: str | None = None
            untouched_direct_segments = segments
            required_transform_boundary_indices = tuple(
                boundary.boundary_index
                for boundary in direct_visual_selection.boundaries
                if boundary.mismatch.transform_repair is not None
            )
            try:
                dense_face_provider = DirectMemberDenseFaceProvider(
                    tuple(request.sources[index] for index in eligible_indices)
                )

                def direct_member_geometry_provider(
                    eligible_source_index: int,
                    canonical_master_seconds: float,
                ):
                    if (
                        isinstance(eligible_source_index, bool)
                        or not isinstance(eligible_source_index, int)
                        or not 0 <= eligible_source_index < len(eligible_indices)
                    ):
                        raise ValueError(
                            "direct member dense review source index is invalid"
                        )
                    request_source_index = eligible_indices[eligible_source_index]
                    timewarp = source_timewarps[request_source_index]
                    source_local_seconds = (
                        timewarp.master_to_source(canonical_master_seconds)
                        if timewarp is not None
                        else canonical_master_seconds + offsets[eligible_source_index]
                    )
                    assert dense_face_provider is not None
                    return dense_face_provider.sample(
                        eligible_source_index,
                        source_local_seconds,
                    )

                direct_visual_config = direct_visual_selection.config
                candidate_member_motion_plan = (
                    build_direct_member_editorial_motion_plan(
                        visual_selection=direct_visual_selection,
                        render_segments=segments,
                        transitions=member_transitions,
                        common_master_start=member_common_start,
                        geometry_provider=direct_member_geometry_provider,
                        fps=29.97,
                        config=DirectMemberEditorialMotionConfig(
                            maximum_boundary_zoom=(
                                direct_visual_config.transform_repair_maximum_zoom
                            ),
                            maximum_center_shift=(
                                direct_visual_config.
                                transform_repair_maximum_center_shift
                            ),
                            maximum_boundary_center_distance=(
                                direct_visual_config.
                                transform_repair_maximum_boundary_center_distance
                            ),
                            maximum_residual_center_distance=(
                                direct_visual_config.
                                transform_repair_maximum_residual_center_distance
                            ),
                            minimum_mismatch_improvement_fraction=(
                                direct_visual_config.
                                transform_repair_minimum_improvement_fraction
                            ),
                            minimum_face_visible_fraction=(
                                direct_visual_config.
                                transform_repair_minimum_face_visible_fraction
                            ),
                            legacy_face_only_maximum_zoom=(
                                direct_visual_config.
                                transform_repair_legacy_face_only_maximum_zoom
                            ),
                            legacy_face_only_maximum_center_shift=(
                                direct_visual_config.
                                transform_repair_legacy_face_only_maximum_center_shift
                            ),
                            minimum_widened_subject_confidence=(
                                direct_visual_config.
                                transform_repair_minimum_subject_confidence
                            ),
                            minimum_widened_subject_visible_fraction=(
                                direct_visual_config.
                                transform_repair_minimum_subject_visible_fraction
                            ),
                            enable_interior_zoom_pulses=True,
                        ),
                    )
                )
                if not set(required_transform_boundary_indices).issubset(
                    candidate_member_motion_plan.accepted_boundary_indices
                ):
                    raise ValueError(
                        "required direct-member transform was not densely accepted"
                    )
                attached_segments = _attach_direct_member_editorial_motion_tracks(
                    visual_selection=direct_visual_selection,
                    eligible_indices=eligible_indices,
                    render_segments=segments,
                    motion_plan=candidate_member_motion_plan,
                )
                segments = attached_segments
                member_editorial_motion_plan = candidate_member_motion_plan
                reporter.log(
                    "직캠 편집 카메라 "
                    f"{member_editorial_motion_plan.applied_count}개 세그먼트 · "
                    f"크기맞춤 {member_editorial_motion_plan.boundary_match_count}개 · "
                    f"줌 {member_editorial_motion_plan.zoom_event_count}회 · "
                    f"팬 {member_editorial_motion_plan.pan_event_count}회 · "
                    "무손실 내부 펄스 "
                    f"{member_editorial_motion_plan.interior_zoom_pulse_count}회"
                )
                if member_editorial_motion_plan.skipped_boundary_reasons:
                    reporter.log(
                        "직캠 편집 카메라 보류 사유 "
                        + " · ".join(
                            f"{reason} {count}개"
                            for reason, count in (
                                member_editorial_motion_plan.skipped_boundary_reasons
                            )
                        )
                    )
            except (
                IndexError,
                OSError,
                TypeError,
                ValueError,
                FloatingPointError,
                cv2.error,
            ) as exc:
                if required_transform_boundary_indices:
                    raise RuntimeError(
                        "공식 대상 멤버 직캠의 선택 필수 확대·이동 경계를 "
                        "밀집 프레임 검증으로 재현하지 못해 렌더를 중단했습니다: "
                        f"{exc}"
                    ) from exc
                segments = untouched_direct_segments
                member_editorial_motion_plan = None
                direct_motion_error = str(exc)
                warnings.append(
                    "공식 대상 멤버 직캠의 밀집 얼굴 안전 검사를 "
                    "통과한 편집 카메라를 확정하지 못해 "
                    f"기존 풀프레임을 사용함: {exc}"
                )
                reporter.emit(
                    "log",
                    level="warning",
                    message=(
                        "직캠 편집 카메라를 안전하게 적용하지 못해 "
                        f"원본 풀프레임 유지: {exc}"
                    ),
                )
            finally:
                provider_diagnostics = (
                    dense_face_provider.diagnostics()
                    if dense_face_provider is not None
                    else {
                        "backend": (
                            "yunet_face_or_verified_full_subject_geometry_"
                            "dense_selected_ramps"
                        ),
                        "request_count": 0,
                        "cache_hit_count": 0,
                        "decode_count": 0,
                        "cache_entry_count": 0,
                        "pixels_persisted": False,
                        "face_embeddings_computed": False,
                    }
                )
                member_direct_dense_face_review_payload = {
                    **provider_diagnostics,
                    "attempted": True,
                    "completed": direct_motion_error is None,
                    "fallback_to_untouched_full_frame": (
                        direct_motion_error is not None
                    ),
                    "error": direct_motion_error,
                    "applied_segment_count": (
                        member_editorial_motion_plan.applied_count
                        if member_editorial_motion_plan is not None
                        else 0
                    ),
                }
                if dense_face_provider is not None:
                    dense_face_provider.close()
            assert direct_full_frame_ratios is not None
            segments = _attach_direct_member_portrait_fill_geometry(
                visual_selection=direct_visual_selection,
                eligible_indices=eligible_indices,
                eligible_source_aspect_ratios=direct_full_frame_ratios,
                render_segments=segments,
            )
            member_look_plan = build_direct_member_look_plan(
                visual_selection=direct_visual_selection,
                render_segments=segments,
            )
            segments = member_look_plan.apply(segments)
            reporter.log(
                "저장된 직캠 특징으로 최종 컷 룩 보정 · "
                f"빠른 경계 {member_look_plan.rapid_boundary_count}개 중 "
                f"예측 충격 {member_look_plan.predicted_high_shock_count}개 · "
                "추가 원본 디코드 0회"
            )
            for detail, segment in zip(
                direct_segment_details,
                segments,
                strict=True,
            ):
                detail["crop_applied"] = segment.crop is not None
                detail["editorial_transform_applied"] = (
                    segment.editorial_transform is not None
                )
            professional_gate_path = request.output_directory / (
                f"{base_filename}-{_safe_slug(request.target_member.value)}-"
                "member-pre-render-quality.json"
            )
            professional_gate_result = (
                evaluate_direct_member_professional_pre_render_gate(
                    visual_selection=direct_visual_selection,
                    cadence=direct_cadence,
                    editorial_motion_plan=member_editorial_motion_plan,
                    transitions=member_transitions,
                    duration_seconds=sum(segment.duration for segment in segments),
                    dense_geometry_review_completed=bool(
                        member_direct_dense_face_review_payload is not None
                        and member_direct_dense_face_review_payload.get("completed")
                        is True
                    ),
                )
            )
            member_direct_professional_pre_render_payload = {
                **professional_gate_result.to_dict(),
                "artifact_path": str(professional_gate_path),
            }
            _write_analysis_manifest_atomically(
                professional_gate_path,
                member_direct_professional_pre_render_payload,
            )
            reporter.log(
                "직캠 전문가 렌더 사전 판정 · "
                f"컷 {professional_gate_result.metrics['visible_switches_per_minute']:.2f}/분 · "
                "포즈 p95 "
                f"{professional_gate_result.metrics['pose_mismatch_p95']} · "
                "실루엣 p95 "
                f"{professional_gate_result.metrics['silhouette_mismatch_p95']} · "
                "편집 카메라 "
                f"{professional_gate_result.metrics['editorial_transform_segment_count']}개"
            )
            if not professional_gate_result.passed:
                raise RuntimeError(
                    "automatic member professional pre-render gate rejected "
                    "the edit before FFmpeg: "
                    + ", ".join(professional_gate_result.failure_reasons)
                    + f"; report={professional_gate_path}"
                )
            selection_basis = OFFICIAL_TARGET_FANCAM_DIRECT_FULL_FRAME_POLICY
        else:
            member_draft = build_member_draft(
                canonical_spans=timing_draft.segments,
                audio_offsets=offsets,
                tracking_by_source=tuple(
                    member_tracking_results.get(index) for index in eligible_indices
                ),
                target_member=request.target_member,
                output_aspect=request.aspect_ratio,
                source_aspect_ratio=source_aspect_ratio,
                source_timewarps=tuple(
                    source_timewarps[index] for index in eligible_indices
                ),
            )
            unavailable = tuple(
                item
                for item in member_draft.segments
                if item.kind is not MemberSegmentKind.TARGET_CROP
            )
            if unavailable:
                warnings.append(
                    f"대상 멤버를 안전하게 확인하지 못한 구간 {len(unavailable)}개가 있어 "
                    "다른 멤버를 잘못 크롭하지 않도록 영상 합성을 중단했습니다. "
                    "컷이 없는 멤버 직캠을 사용하거나 앵커 장면을 바꿔 주세요."
                )
                can_render_member = False
                segments = ()
                member_transitions = ()
                selection_basis = _member_render_selection_basis(
                    member_draft.selection_diagnostics.policy
                )
            else:
                segments, member_transitions, full_frame_fallbacks = (
                    _member_render_segments(
                        member_segments=member_draft.segments,
                        timing_segments=timing_draft.segments,
                        eligible_indices=eligible_indices,
                        common_master_start=member_common_start,
                        source_timewarps=source_timewarps,
                    )
                )
                try:
                    candidate_member_motion_plan = (
                        build_member_editorial_motion_plan(
                            member_segments=member_draft.segments,
                            render_segments=segments,
                            transitions=member_transitions,
                            source_scene_cut_times=scene_cut_times,
                            fps=29.97,
                        )
                    )
                    segments = _attach_member_editorial_motion_tracks(
                        member_segments=member_draft.segments,
                        render_segments=segments,
                        motion_plan=candidate_member_motion_plan,
                    )
                    member_editorial_motion_plan = candidate_member_motion_plan
                    reporter.log(
                        "멤버 편집 카메라 "
                        f"{member_editorial_motion_plan.applied_count}개 세그먼트 · "
                        f"크기맞춤 {member_editorial_motion_plan.boundary_match_count}개 · "
                        f"줌 {member_editorial_motion_plan.zoom_event_count}회 · "
                        f"팬 {member_editorial_motion_plan.pan_event_count}회 · "
                        f"미세회전 {member_editorial_motion_plan.roll_event_count}회"
                    )
                    if member_editorial_motion_plan.skipped_boundary_reasons:
                        reporter.log(
                            "멤버 편집 카메라 보류 사유 "
                            + " · ".join(
                                f"{reason} {count}개"
                                for reason, count in (
                                    member_editorial_motion_plan.skipped_boundary_reasons
                                )
                            )
                        )
                except ValueError as exc:
                    warnings.append(
                        "멤버 편집 카메라를 확인된 대상 추적 안에서 "
                        f"안전하게 확정하지 못해 기존 멤버 크롭을 사용함: {exc}"
                    )
                selection_basis = _member_render_selection_basis(
                    member_draft.selection_diagnostics.policy
                )

        if can_render_member:
            width, height = output_size(request.aspect_ratio)
            rendered_video_path = request.output_directory / (
                f"{base_filename}-{_safe_slug(request.target_member.value)}-member-crosscut.mp4"
            )
            render_plan_path = request.output_directory / (
                f"{base_filename}-{_safe_slug(request.target_member.value)}-render-plan.json"
            )
            plan = RenderPlan(
                sources=request.sources,
                segments=segments,
                reference_source=request.reference_source,
                aspect_ratio=request.aspect_ratio,
                output=rendered_video_path,
                master_audio_start=member_common_start,
                width=width,
                height=height,
                transitions=member_transitions,
                member_pose_dissolve_requested=(
                    bool(direct_cadence["member_pose_dissolve_requested"])
                    if use_direct_member_full_frame and direct_cadence is not None
                    else None
                ),
                member_pose_dissolve_effective=(
                    bool(direct_cadence["member_pose_dissolve_effective"])
                    if use_direct_member_full_frame and direct_cadence is not None
                    else None
                ),
                member_pose_dissolve_count=(
                    int(direct_cadence["member_pose_dissolve_count"])
                    if use_direct_member_full_frame and direct_cadence is not None
                    else None
                ),
            )
            if (
                request.source_mode is SourceMode.AUTOMATIC
                and use_direct_member_full_frame
            ):
                reporter.status(
                    "멤버 최종 컷을 실제 PTS·포즈·바디 모션으로 "
                    "로컬 동기 검증 중…"
                )
                try:
                    cut_local_sync_payload = analyze_cut_local_sync(
                        plan,
                        source_timewarps=source_timewarps,
                        timewarp_payloads=local_timewarp_payloads,
                        activation=MEMBER_CUT_LOCAL_SYNC_ACTIVATION,
                        pts_cache_directory=request.source_cache_directory,
                    )
                    reporter.log(
                        "멤버 보존 교차소스 경계 "
                        f"{cut_local_sync_payload['retained_cross_source_boundary_count']}개 · "
                        f"측정 {cut_local_sync_payload['measured_boundary_count']}개 · "
                        f"불확실 {cut_local_sync_payload['unknown_boundary_count']}개 · "
                        "로컬 잔차 p95 "
                        f"{cut_local_sync_payload['residual_p95_frames']} frame"
                    )
                    if not bool(cut_local_sync_payload.get("passed")):
                        raise RuntimeError(
                            "member cut-local sync proof rejected: "
                            f"unknown={cut_local_sync_payload.get('unknown_boundary_count')}; "
                            f"over_threshold={cut_local_sync_payload.get('threshold_exceeding_boundary_count')}; "
                            f"reasons={cut_local_sync_payload.get('unknown_reason_counts')}"
                        )
                except (
                    cv2.error,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    reporter.emit(
                        "log",
                        level="error",
                        message=(
                            "멤버 컷-로컬 동기 증거를 완성하지 "
                            "못해 렌더를 중단합니다 "
                            f"({type(exc).__name__}: {exc})."
                        ),
                    )
                    raise RuntimeError(
                        "automatic member cut-local sync proof failed before "
                        f"render: {type(exc).__name__}: {exc}"
                    ) from exc
            write_plan(plan, render_plan_path)
            reporter.status(
                f"{request.target_member.value} 멤버 교차직캠을 합성하는 중…"
            )
            codec = render(plan)
            timeline_interchange_paths = write_timeline_interchange_bundle(
                plan,
                output_directory=request.output_directory,
                base_filename=(
                    f"{base_filename}-{_safe_slug(request.target_member.value)}-member-timeline"
                ),
                project_name=(
                    f"IVE {request.song} {request.target_member.value} member crosscut"
                ),
            )
            final_source_usage = plan.source_usage_payload()
            final_segments = _final_render_segments_payload(
                plan.segments,
                common_master_start=member_common_start,
            )
            if use_direct_member_full_frame:
                assert direct_full_frame_ratios is not None
                assert direct_cadence is not None
                assert direct_visual_selection is not None
                assert member_look_plan is not None
                member_direct_full_frame_payload = {
                    "enabled": True,
                    "policy": OFFICIAL_TARGET_FANCAM_DIRECT_FULL_FRAME_POLICY,
                    "provenance": (
                        "automatic_acquisition_official_target_member_fancam"
                    ),
                    "identity_policy": (
                        "official_target_member_provenance_no_identity_inference_"
                        "with_lossless_portrait_fill"
                    ),
                    "portrait_fill_contract": (
                        AUTOMATIC_MEMBER_PORTRAIT_FILL_CONTRACT
                    ),
                    "output_aspect": request.aspect_ratio.value,
                    "output_pixel_aspect_ratio": 9.0 / 16.0,
                    "source_aspect_ratios": list(direct_full_frame_ratios),
                    "eligible_request_source_indices": list(eligible_indices),
                    "minimum_required_distinct_source_count": (
                        len(eligible_indices)
                    ),
                    "required_used_request_source_indices": list(
                        eligible_indices
                    ),
                    "cadence": direct_cadence,
                    "member_pose_dissolve_requested": direct_cadence[
                        "member_pose_dissolve_requested"
                    ],
                    "member_pose_dissolve_effective": direct_cadence[
                        "member_pose_dissolve_effective"
                    ],
                    "member_pose_dissolve_count": direct_cadence[
                        "member_pose_dissolve_count"
                    ],
                    "pose_matched_dissolves": direct_cadence[
                        "pose_matched_dissolves"
                    ],
                    "visual_selection": _direct_member_visual_payload(
                        direct_visual_selection,
                        eligible_indices=eligible_indices,
                    ),
                    "editorial_motion": (
                        member_editorial_motion_plan.to_dict()
                        if member_editorial_motion_plan is not None
                        else None
                    ),
                    "dense_face_review": (
                        member_direct_dense_face_review_payload
                    ),
                    "professional_pre_render_gate": (
                        member_direct_professional_pre_render_payload
                    ),
                    "look_calibration": member_look_plan.to_dict(),
                    "final_source_usage": final_source_usage,
                    "segments": list(direct_segment_details),
                }
            render_payload = {
                "video_path": str(rendered_video_path),
                "plan_path": str(render_plan_path),
                "video_codec": codec,
                "interchange": {
                    "fcpxml_path": str(timeline_interchange_paths[0]),
                    "otio_path": str(timeline_interchange_paths[1]),
                    "edl_path": str(timeline_interchange_paths[2]),
                    "fidelity_audit_path": str(timeline_interchange_paths[3]),
                    "authoritative_pixels": str(rendered_video_path),
                },
                "selection_basis": selection_basis,
                "source_clock_basis": (
                    "local_monotonic_timewarp_with_per_source_scalar_fallback"
                    if any(
                        timewarp is not None and not timewarp.scalar_fallback
                        for index, timewarp in enumerate(source_timewarps)
                        if index in eligible_indices
                    )
                    else "global_scalar_offset"
                ),
                "cut_local_sync": cut_local_sync_payload,
                "effective_master_start_seconds": member_common_start,
                "automatic_full_frame_fallback_count": len(
                    full_frame_fallbacks
                ),
                "automatic_full_frame_fallbacks": list(full_frame_fallbacks),
                "direct_full_frame": use_direct_member_full_frame,
                "direct_full_frame_policy": (
                    OFFICIAL_TARGET_FANCAM_DIRECT_FULL_FRAME_POLICY
                    if use_direct_member_full_frame
                    else None
                ),
                "cadence": direct_cadence,
                "editorial_motion": (
                    member_editorial_motion_plan.to_dict()
                    if member_editorial_motion_plan is not None
                    else None
                ),
                "member_look_calibration": (
                    member_look_plan.to_dict()
                    if member_look_plan is not None
                    else None
                ),
                "final_source_usage": final_source_usage,
                "final_segments": final_segments,
            }
            reporter.progress(0.99, "멤버 교차직캠 합성 완료")
    elif request.edit_mode is EditMode.MEMBER:
        if group_ready and not member_range_ready:
            warnings.append(
                "가장 늦은 멤버 앵커 뒤에 합성할 공통 구간이 "
                "2프레임보다 짧아 멤버 영상을 만들지 않았습니다. "
                "모든 소스에서 더 앞쪽 장면을 앵커로 지정해 주세요."
            )
        else:
            warnings.append(
                "안전하게 정렬된 소스 2개와 박자 분석이 필요해 멤버 영상을 만들지 않았습니다."
            )

    automatic_ive_group_artifact = bool(
        request.source_mode is SourceMode.AUTOMATIC
        and request.edit_mode is EditMode.GROUP
    )
    automatic_ive_member_artifact = bool(
        request.source_mode is SourceMode.AUTOMATIC
        and request.edit_mode is EditMode.MEMBER
    )
    automatic_ive_member_direct_artifact = bool(
        automatic_ive_member_artifact
        and rendered_video_path is not None
        and isinstance(render_payload, Mapping)
        and render_payload.get("direct_full_frame") is True
    )
    automatic_ive_product_artifact = bool(
        automatic_ive_group_artifact or automatic_ive_member_artifact
    )
    artifact_role: str | None = None
    if automatic_ive_group_artifact:
        if rendered_video_path is None:
            artifact_role = "diagnostic"
        elif allow_failed_comparison_render:
            artifact_role = "comparison"
        else:
            # A rendered file is not a product result yet.  Mandatory final
            # media QA below is the sole authority that may promote this
            # unexposed state to ``delivery``.
            artifact_role = "pending_qa"
    elif automatic_ive_member_artifact:
        if rendered_video_path is None:
            artifact_role = "diagnostic"
        elif automatic_ive_member_direct_artifact:
            # Direct member pixels are not an app result until their exact
            # all-source and mapped-time provenance receipt passes below.
            artifact_role = "pending_qa"
        else:
            # An automatic member render outside the direct-full-frame
            # contract has no production QA authority and must never leak into
            # the result list.
            artifact_role = "rejected"
    if render_payload is not None and artifact_role is not None:
        render_payload["artifact_role"] = artifact_role

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": timestamp.isoformat(),
        "request": request.as_dict(),
        "source_index_mapping": _source_index_mapping(
            request.sources,
            eligible_indices,
        ),
        "media": [info.to_dict() for info in media],
        "alignment": alignment.to_dict(),
        "alignment_peer_consensus": alignment_peer_consensus.to_dict(),
        "alignment_eligibility": alignment_eligibility,
        "local_timewarps": list(local_timewarp_payloads),
        "performance_range_trim": performance_range_trim_payload,
        "common_master_range": {
            "start_seconds": common_start,
            "end_seconds": common_end,
            "duration_seconds": common_end - common_start,
        },
        "scene_analysis": scene_payloads,
        "rhythm_analysis": rhythm_payload,
        "section_plan": (
            stage_edit_generation_result.learned_section_director.section_plan.to_dict()
            if stage_edit_generation_result is not None
            and stage_edit_generation_result.learned_section_director is not None
            else section_plan.to_dict()
            if section_plan is not None
            else None
        ),
        "source_look_match": (
            _with_index_domain(
                source_look_match_result.to_dict(),
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            if source_look_match_result is not None
            else None
        ),
        "source_derivative_diversity": (
            _with_index_domain(
                source_derivative_diversity_result.to_dict(),
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            if source_derivative_diversity_result is not None
            else None
        ),
        "boundary_look_match": (
            boundary_look_match_result.to_dict()
            if boundary_look_match_result is not None
            else None
        ),
        "boundary_affine": (
            _with_index_domain(
                boundary_affine_plan.to_dict(),
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            if boundary_affine_plan is not None
            else None
        ),
        "rendered_proxy_review": rendered_proxy_review_payload,
        "rendered_montage_grammar_review": (
            deepcopy(rendered_montage_grammar_review_payload)
            if rendered_montage_grammar_review_payload is not None
            else None
        ),
        "stage_edit_exact_shot_clock": (
            deepcopy(stage_edit_exact_shot_clock_payload)
            if stage_edit_exact_shot_clock_payload is not None
            else None
        ),
        "dynamic_crop_version": stage_edit_dynamic_crop_version_payload,
        "dynamic_crop_v3_decodes": (
            deepcopy(dynamic_crop_v3_decodes_payload)
            if dynamic_crop_v3_decodes_payload is not None
            else None
        ),
        "dynamic_crop_motion_v3": (
            deepcopy(dynamic_crop_motion_v3_payload)
            if dynamic_crop_motion_v3_payload is not None
            else None
        ),
        "rendered_candidate_selection": rendered_candidate_selection_payload,
        "stage_edit_generation": (
            {
                **stage_edit_generation_result.to_dict(),
                "audit_path": str(stage_edit_generation_audit_path),
                "activation": (
                    registered_stage_edit_bundle.to_dict()
                    if registered_stage_edit_bundle is not None
                    else {
                        "activation": "explicit_request_bundle",
                        "silent_fallback_allowed": False,
                    }
                ),
                "inference_snapshot": (
                    stage_edit_inference_snapshot_result.to_dict()
                    if stage_edit_inference_snapshot_result is not None
                    else None
                ),
                "real_causal_sensing_audit": (
                    stage_edit_causal_sensing_pipeline_audit.to_dict()
                    if stage_edit_causal_sensing_pipeline_audit is not None
                    else None
                ),
                "professional_montage_grammar_profile": (
                    deepcopy(stage_edit_montage_policy.profile_identity)
                    if stage_edit_montage_policy is not None
                    else None
                ),
                "professional_montage_policy": (
                    stage_edit_montage_policy.to_dict()
                    if stage_edit_montage_policy is not None
                    else None
                ),
                "professional_motion_cut_profile": (
                    deepcopy(stage_edit_motion_cut_profile_payload)
                    if stage_edit_motion_cut_profile_payload is not None
                    else None
                ),
            }
            if stage_edit_generation_result is not None
            and stage_edit_generation_audit_path is not None
            else None
        ),
        "member_tracking": member_tracking_payloads,
        "member_auto_anchors": member_auto_anchor_payloads,
        "member_anchor_trim": member_anchor_trim_payload,
        "member_direct_full_frame": member_direct_full_frame_payload,
        "member_direct_professional_pre_render_gate": (
            member_direct_professional_pre_render_payload
        ),
        "member_look_calibration": (
            member_look_plan.to_dict()
            if member_look_plan is not None
            else None
        ),
        "group_draft": (
            _with_index_domain(
                group_draft.to_dict(),
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            if group_draft is not None
            else None
        ),
        "visual_timeline": (
            _with_index_domain(
                visual_timeline.to_dict(),
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            if visual_timeline is not None
            else None
        ),
        "visual_candidate_set": (
            _with_index_domain(
                visual_candidate_set.to_dict(),
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            if visual_candidate_set is not None
            else None
        ),
        "stage_edit_visual_candidate_batch": (
            _with_index_domain(
                stage_edit_visual_candidate_batch.to_dict(),
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            if stage_edit_visual_candidate_batch is not None
            else None
        ),
        "visual_boundary_repairs": (
            _with_index_domain(
                visual_boundary_repairs.to_dict(),
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            if visual_boundary_repairs is not None
            else None
        ),
        "timeline_review": (
            _with_index_domain(
                timeline_review.to_dict(),
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            if timeline_review is not None
            else None
        ),
        "face_match": (
            _with_index_domain(
                face_match_plan.to_dict(),
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            if face_match_plan is not None
            else None
        ),
        "group_transitions": (
            group_transition_plan.to_dict(29.97)
            if group_transition_plan is not None
            else None
        ),
        "editorial_motion": (
            _with_index_domain(
                (
                    editorial_motion_plan
                    if editorial_motion_plan is not None
                    else member_editorial_motion_plan
                ).to_dict(),
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            if (
                editorial_motion_plan is not None
                or member_editorial_motion_plan is not None
            )
            else None
        ),
        "member_draft": (
            _with_index_domain(
                member_draft.to_dict(),
                ELIGIBLE_SOURCE_INDEX_DOMAIN,
            )
            if member_draft is not None
            else None
        ),
        "render": render_payload,
        "warnings": warnings,
        "capabilities": _capabilities_payload(
            group_ready=group_ready,
            edit_mode=request.edit_mode,
            member_draft=member_draft,
            rendered_video_path=rendered_video_path,
            used_automatic_member_anchors=used_automatic_member_anchors,
            used_direct_full_frame=(member_direct_full_frame_payload is not None),
        ),
    }
    if automatic_ive_product_artifact:
        manifest["artifact_role"] = artifact_role
        manifest["app_result_exposure"] = {
            "contract": (
                APP_RESULT_EXPOSURE_CONTRACT
                if automatic_ive_group_artifact
                else AUTOMATIC_IVE_MEMBER_APP_RESULT_EXPOSURE_CONTRACT
            ),
            "video_output_exposed": artifact_role == "delivery",
            "delivery_completion_message_allowed": artifact_role == "delivery",
        }
    if stage_edit_candidate_rerank_result is not None:
        manifest["stage_edit_candidate_rerank"] = (
            stage_edit_candidate_rerank_result.to_dict()
        )
    filename = f"{base_filename}-analysis.json"
    manifest_path = request.output_directory / filename
    _write_analysis_manifest_atomically(manifest_path, manifest)
    mandatory_group_post_render_qa_path = (
        _enforce_automatic_stage_edit_post_render_qa(
            registered_stage_edit_bundle=registered_stage_edit_bundle,
            manifest=manifest,
            manifest_path=manifest_path,
            reporter=reporter,
            automatic_ive_group_delivery=bool(
                request.source_mode is SourceMode.AUTOMATIC
                and request.edit_mode is EditMode.GROUP
                and not request.acquire_only
                and rendered_video_path is not None
                and artifact_role == "pending_qa"
            ),
            rendered_montage_config=rendered_montage_config,
        )
    )
    mandatory_member_post_render_qa_path = (
        _enforce_automatic_member_post_render_qa(
            manifest=manifest,
            manifest_path=manifest_path,
            reporter=reporter,
        )
    )
    mandatory_post_render_qa_path = (
        mandatory_group_post_render_qa_path
        if mandatory_group_post_render_qa_path is not None
        else mandatory_member_post_render_qa_path
    )
    if automatic_ive_group_artifact:
        promoted_role = manifest.get("artifact_role")
        if promoted_role not in {
            "diagnostic",
            "comparison",
            "delivery",
        }:
            raise RuntimeError(
                "automatic IVE GROUP artifact did not reach a terminal "
                "exposure state"
            )
        artifact_role = str(promoted_role)
    elif automatic_ive_member_artifact:
        promoted_role = manifest.get("artifact_role")
        if promoted_role not in {"diagnostic", "rejected", "delivery"}:
            raise RuntimeError(
                "automatic IVE MEMBER artifact did not reach a terminal "
                "exposure state"
            )
        artifact_role = str(promoted_role)
    reporter.progress(1.0, "분석 결과를 저장했습니다.")
    output_paths = [manifest_path]
    if render_plan_path is not None:
        output_paths.append(render_plan_path)
    if rendered_proxy_review_path is not None:
        output_paths.append(rendered_proxy_review_path)
    if rendered_candidate_selection_path is not None:
        output_paths.append(rendered_candidate_selection_path)
    if stage_edit_inference_snapshot_result is not None:
        output_paths.extend(
            (
                stage_edit_inference_snapshot_result.snapshot_path,
                stage_edit_inference_snapshot_result.manifest_path,
            )
        )
    if stage_edit_causal_sensing_pipeline_audit is not None:
        output_paths.append(stage_edit_causal_sensing_pipeline_audit.path)
    if stage_edit_generation_audit_path is not None:
        output_paths.append(stage_edit_generation_audit_path)
    if source_derivative_audit_path is not None:
        output_paths.append(source_derivative_audit_path)
    if mandatory_post_render_qa_path is not None:
        output_paths.append(mandatory_post_render_qa_path)
    if rendered_video_path is not None and (
        not automatic_ive_product_artifact or artifact_role == "delivery"
    ):
        output_paths.append(rendered_video_path)
    output_paths.extend(timeline_interchange_paths)
    return AnalysisArtifact(
        manifest_path=manifest_path,
        manifest=manifest,
        output_paths=tuple(output_paths),
    )
