"""Top-level offline analysis pipeline used by the native app."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
import tempfile
from typing import Any, Iterator, Mapping, Sequence

from numpy.typing import NDArray
import numpy as np
import cv2

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
from .cut_local_sync import analyze_cut_local_sync
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
    build_direct_member_editorial_motion_plan,
)
from .editorial_motion import EditorialMotionPlan, build_editorial_motion_plan
from .events import EventReporter
from .face_match import FaceMatchPlan, build_face_match_plan
from .group_transition import GroupTransitionPlan, build_group_transition_plan
from .draft import DraftSegment, GroupDraftTimeline, build_group_draft
from .ive import AspectRatio, EditMode, SourceMode
from .media import MediaInfo, probe_media
from .member_draft import (
    MemberDraftSegment,
    MemberDraftTimeline,
    MemberSegmentKind,
    build_member_draft,
)
from .member_editorial_motion import build_member_editorial_motion_plan
from .performance_range import estimate_shared_performance_range
from .post_render_qa import (
    PostRenderArtifactSet,
    evaluate_and_write_post_render_qa,
)
from .professional_reference_gate import ProfessionalReferenceProfile
from .professional_reference_cadence import (
    ProfessionalReferenceCadenceProfile,
)
from .professional_montage_policy import (
    ProfessionalMontagePolicy,
    load_professional_montage_policy,
)
from .render import (
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
from .section_director import SectionPlan, build_section_plan
from .source_look_match import SourceLookMatchResult, analyze_source_look_match
from .source_derivative_diversity import (
    SourceDerivativeDiversityResult,
    analyze_source_derivative_diversity,
)
from .stage_edit_candidate_rerank import (
    StageEditCandidateRerankResult,
    rerank_visual_timeline_candidates_with_stage_edit,
)
from .stage_edit import StageEdit
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
from .stage_edit_semimarkov_beam import StageEditSemiMarkovConfig
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
from .timewarp import TimeWarpMap, estimate_local_timewarp
from .timeline_interchange import write_timeline_interchange_bundle
from .vision import (
    SceneDetectionConfig,
    detect_scene_changes_in_video,
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
AUTOMATIC_STAGE_EDIT_POST_RENDER_QA_CONTRACT = (
    "automatic_registered_stage_edit_mandatory_post_render_qa_v1"
)
DEFAULT_IVE_PROFESSIONAL_MONTAGE_PROFILE_PATH = Path(
    "/Volumes/CrossCutMedia/CrossCut-Training/release/profiles/"
    "ive-professional-montage-grammar-v1.json"
)
_DIRECT_MEMBER_TARGET_RUN_SECONDS = 3.9
_DIRECT_MEMBER_MIN_SWITCHES_PER_MINUTE = 13.5
_DIRECT_MEMBER_MAX_SWITCHES_PER_MINUTE = 17.5
_DIRECT_MEMBER_MIN_MEDIAN_RUN_SECONDS = 3.5
_DIRECT_MEMBER_MAX_MEDIAN_RUN_SECONDS = 4.2
_DIRECT_MEMBER_MAX_RUN_SECONDS = 7.2
_DIRECT_MEMBER_MAX_PARTITION_PROFILE_COUNT = 5
_DIRECT_MEMBER_MAX_BOTTLENECK_BOUNDARY_COUNT = 2


@dataclass(frozen=True, slots=True)
class AnalysisArtifact:
    manifest_path: Path
    manifest: dict[str, Any]
    output_paths: tuple[Path, ...]


def _enforce_automatic_stage_edit_post_render_qa(
    *,
    registered_stage_edit_bundle: RegisteredStageEditProductionBundle | None,
    manifest: dict[str, Any],
    manifest_path: Path,
    reporter: EventReporter,
) -> Path | None:
    """Require PASS from the final-media QA only for registered automatic StageEdit.

    The evaluator consumes the already persisted timestamp-bound analysis,
    render-plan, candidate-selection, and final MP4.  Its report and the final
    MP4 remain in place even when the verdict blocks the CLI success event.
    """

    if registered_stage_edit_bundle is None:
        return None

    artifact_set = PostRenderArtifactSet.from_analysis_path(manifest_path)
    reporter.status(
        "등록된 StageEdit 최종 MP4를 전체 디코딩하고 "
        "전문 후처리 QA를 실행하는 중…"
    )
    report, report_path = evaluate_and_write_post_render_qa(
        artifact_set,
        trusted_registered_stage_edit=True,
    )
    report_payload = report.to_dict()
    provenance = {
        "contract": AUTOMATIC_STAGE_EDIT_POST_RENDER_QA_CONTRACT,
        "required": True,
        "activation": "automatic_registered_production_bundle",
        "success_requires_pass": True,
        "report_path": str(report_path.resolve()),
        "verdict": report.verdict.value,
        "passed": report.passed,
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
        },
    }
    manifest["post_render_qa"] = dict(provenance)
    render_payload = manifest.get("render")
    if isinstance(render_payload, dict):
        render_payload["post_render_qa"] = dict(provenance)
    generation_payload = manifest.get("stage_edit_generation")
    if isinstance(generation_payload, dict):
        generation_payload["post_render_qa"] = dict(provenance)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    reporter.log(
        "StageEdit 최종 후처리 QA "
        f"{report.verdict.value} · 보고서 {report_path}"
    )
    if not report.passed:
        raise RuntimeError(
            "automatic registered StageEdit final render blocked by mandatory "
            f"post-render QA verdict {report.verdict.value}; final MP4 preserved "
            f"at {artifact_set.video_path.resolve()}; QA report preserved at "
            f"{report_path.resolve()}"
        )
    return report_path.resolve()


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


def _align_sequentially(
    request: AnalysisRequest,
    reporter: EventReporter,
) -> tuple[AlignmentResult, NDArray[np.float32]]:
    reference_path = request.sources[request.reference_source]
    reporter.status("기준 음원을 추출하는 중…")
    reference_pcm = extract_mono_pcm(reference_path, sample_rate=ANALYSIS_SAMPLE_RATE)
    alignments: list[SourceAlignment] = []

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
        alignments.append(
            SourceAlignment(
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
        )
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
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for source_index, item in enumerate(alignment.alignments):
        reasons: list[str] = []
        minimum_score: float | None = None
        minimum_margin: float | None = None
        policy = (
            "member_fancam_crowd_mix_peak_margin_v1"
            if edit_mode is EditMode.MEMBER
            else "group_broadcast_feed_strict_v1"
        )
        if not item.is_reference:
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
            else:
                minimum_score = (
                    0.55 if item.backend == "spectral_flux_ncc" else 0.45
                )
            minimum_margin = 0.06 if item.backend == "spectral_flux_ncc" else 0.04
            if item.score < minimum_score:
                reasons.append(
                    f"상관 점수 {item.score:.3f} < {minimum_score:.3f}"
                )
            if item.score_margin < minimum_margin:
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
                },
                "measured": {
                    "offset_seconds": item.offset_seconds,
                    "score": item.score,
                    "score_margin": item.score_margin,
                    "overlap_ratio": item.overlap_ratio,
                    "backend": item.backend,
                },
            }
        )
    return results


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
    """Fail closed on the materialized automatic portrait-source contract.

    Automatic acquisition is the provenance boundary: it admits only official
    target-member fancams.  The product path promises to retain those portrait
    frames uncropped, so missing dimensions or a non-9:16 source is a contract
    failure rather than permission to enter anchor/tracking fallback.
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
    expected_ratio = 9.0 / 16.0
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
        if abs(ratio - expected_ratio) > 0.01:
            raise ValueError(
                "automatic MEMBER+PORTRAIT direct-full-frame aspect contract "
                f"failed for source {source_index}: width/height {ratio:.6f} "
                "is outside 9:16±0.01"
            )
        ratios.append(ratio)
    return tuple(ratios)


def _automatic_member_direct_full_frame_ratios(
    request: AnalysisRequest,
    media: Sequence[MediaInfo],
    eligible_indices: Sequence[int],
) -> tuple[float, ...] | None:
    """Gate the provenance-scoped portrait direct-full-frame member path."""

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
class _DirectFullFramePartitionProfile:
    profile_id: str
    unit_counts: tuple[int, ...]
    spans: tuple[_DirectFullFrameSpan, ...]
    shifted_boundary_index: int | None = None
    boundary_shift_units: int = 0
    shifted_boundary_reasons: tuple[str, ...] = ()

    @property
    def changed_span_count(self) -> int:
        return 0 if self.shifted_boundary_index is None else 2

    @property
    def musical_boundary_penalty(self) -> int:
        reasons = set(self.shifted_boundary_reasons)
        if "phrase" in reasons:
            return 0
        if "strong_beat" in reasons:
            return 1
        if "beat" in reasons:
            return 2
        return 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "partition_unit_counts": list(self.unit_counts),
            "coarse_span_count": len(self.spans),
            "shifted_boundary_index": self.shifted_boundary_index,
            "boundary_shift_units": self.boundary_shift_units,
            "changed_span_count": self.changed_span_count,
            "shifted_boundary_reasons": list(self.shifted_boundary_reasons),
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


def _direct_full_frame_base_partition_profile(
    timing_segments: Sequence[DraftSegment],
) -> _DirectFullFramePartitionProfile:
    spans = _coarsen_direct_full_frame_spans(timing_segments)
    return _DirectFullFramePartitionProfile(
        profile_id="base-reference-coarsening",
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


def _direct_full_frame_shifted_partition_profiles(
    timing_segments: Sequence[DraftSegment],
    base: _DirectFullFramePartitionProfile,
    bottleneck_boundaries: Sequence[int],
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
            profiles.append(
                _DirectFullFramePartitionProfile(
                    profile_id=(
                        f"boundary-{boundary_index}-"
                        f"{left_count}-plus-{right_count}"
                    ),
                    unit_counts=signature,
                    spans=spans,
                    shifted_boundary_index=boundary_index,
                    boundary_shift_units=shift_units,
                    shifted_boundary_reasons=boundary_reasons,
                )
            )
            seen_counts.add(signature)
            if len(profiles) >= _DIRECT_MEMBER_MAX_PARTITION_PROFILE_COUNT - 1:
                return tuple(profiles)
    return tuple(profiles)


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
    }
    path = output_directory / (
        f"{base_filename}-{_safe_slug(target_member)}-"
        "direct-selection-failure.json"
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
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
) -> dict[str, Any]:
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
                    "state_capacity_exhausted",
                    "root_candidate_diagnostics",
                )
                if key in raw_exact
            }
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
    return {
        "message": str(error),
        "failure": diagnostics.get("failure"),
        "constraint_infeasibility_proven": diagnostics.get(
            "constraint_infeasibility_proven",
            _direct_partition_failure_is_proven(diagnostics),
        ),
        "visual_gates_relaxed": False,
        "summary": _direct_member_selection_failure_summary(diagnostics),
        "zero_local_nonsevere_transition_boundary_indices": zero_boundaries,
        "exact_lifeline": exact,
        "valid_count_by_source": diagnostics.get("valid_count_by_source", []),
    }


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


def _select_direct_member_partition_profile(
    *,
    timing_draft: GroupDraftTimeline,
    eligible_indices: Sequence[int],
    eligible_sources: Sequence[Path],
    source_timewarps: Sequence[TimeWarpMap | None] | None,
    scene_cut_times: Sequence[Sequence[float]],
    source_cycle: Sequence[int],
    descriptor_provider: DescriptorProvider | None = None,
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
    base = _direct_full_frame_base_partition_profile(timing_draft.segments)
    if len(base.spans) < len(source_cycle):
        raise RuntimeError("coarse direct cadence cannot show every eligible source")

    owned_provider: _VideoDescriptorProvider | None = None
    provider = descriptor_provider
    if provider is None:
        owned_provider = _VideoDescriptorProvider(eligible_sources)
        provider = owned_provider

    attempted: list[dict[str, Any]] = []
    failures: list[
        tuple[_DirectFullFramePartitionProfile, DirectMemberVisualSelectionError]
    ] = []
    feasible: list[
        tuple[_DirectFullFramePartitionProfile, DirectMemberVisualSelection]
    ] = []
    selection_config = DirectMemberVisualConfig()

    def _evaluate(profile: _DirectFullFramePartitionProfile) -> None:
        try:
            selection = select_balanced_direct_member_sources(
                sources=eligible_sources,
                canonical_spans=tuple(
                    (span.canonical_master_start, span.canonical_master_end)
                    for span in profile.spans
                ),
                audio_offsets=timing_draft.audio_offsets,
                source_timewarps=compact_timewarps,
                source_scene_cut_times=scene_cut_times,
                preferred_source_order=source_cycle,
                config=selection_config,
                descriptor_provider=provider,
            )
        except DirectMemberVisualSelectionError as error:
            failures.append((profile, error))
            attempted.append(
                {
                    **profile.to_dict(),
                    "status": "hard_infeasible",
                    "hard_violation_count": None,
                    "visual_gates_relaxed": False,
                    "diagnostics": _compact_direct_partition_failure(error),
                }
            )
            return
        feasible.append((profile, selection))
        rank = _direct_partition_selection_rank(profile, selection)
        attempted.append(
            {
                **profile.to_dict(),
                "status": "feasible",
                "hard_violation_count": 0,
                "visual_gates_relaxed": False,
                "selection_rank": {
                    "warning_boundary_count": rank[0],
                    "optimizer_cost": rank[1],
                    "changed_span_count": rank[2],
                    "musical_boundary_penalty": rank[3],
                },
            }
        )

    try:
        _evaluate(base)
        if not feasible and failures:
            base_error = failures[0][1]
            if _direct_partition_failure_is_proven(base_error.diagnostics):
                bottlenecks = _direct_partition_failure_bottlenecks(
                    base_error.diagnostics,
                    span_count=len(base.spans),
                )
                for profile in _direct_full_frame_shifted_partition_profiles(
                    timing_draft.segments,
                    base,
                    bottlenecks,
                ):
                    _evaluate(profile)
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
            dict(failures[-1][1].diagnostics) if failures else {}
        )
        all_proven = bool(failures) and all(
            _direct_partition_failure_is_proven(error.diagnostics)
            for _profile, error in failures
        )
        aggregate_diagnostics: dict[str, Any] = {
            **latest_diagnostics,
            "failure": "direct_partition_profiles_exhausted",
            "constraint_infeasibility_proven": all_proven,
            "visual_gates_relaxed": False,
            "partition_search": {
                "policy": (
                    "base_then_proven_bottleneck_adjacent_4_4_to_"
                    "5_3_or_3_5_v1"
                ),
                "maximum_profile_count": (
                    _DIRECT_MEMBER_MAX_PARTITION_PROFILE_COUNT
                ),
                "profiles_attempted": [
                    {
                        **profile.to_dict(),
                        "status": "hard_infeasible",
                        "visual_gates_relaxed": False,
                        "diagnostics": error.diagnostics,
                    }
                    for profile, error in failures
                ],
                "descriptor_cache": provider_cache_diagnostics,
                "render_started": False,
            },
        }
        raise DirectMemberVisualSelectionError(
            "all bounded direct-member musical partitions were hard-infeasible",
            aggregate_diagnostics,
        )

    selected_profile, selected_selection = min(
        feasible,
        key=lambda item: _direct_partition_selection_rank(item[0], item[1]),
    )
    partition_search = {
        "policy": (
            "base_then_proven_bottleneck_adjacent_4_4_to_5_3_or_3_5_v1"
        ),
        "maximum_profile_count": _DIRECT_MEMBER_MAX_PARTITION_PROFILE_COUNT,
        "profiles_attempted": attempted,
        "selected_profile_id": selected_profile.profile_id,
        "selected_partition_unit_counts": list(selected_profile.unit_counts),
        "fallback_used": selected_profile.shifted_boundary_index is not None,
        "visual_gates_relaxed": False,
        "descriptor_cache": provider_cache_diagnostics,
        "render_started_after_profile_selection": True,
    }
    return (selected_profile, selected_selection, partition_search)


def _official_target_fancam_direct_full_frame_segments(
    *,
    timing_draft: GroupDraftTimeline,
    eligible_indices: Sequence[int],
    eligible_sources: Sequence[Path],
    common_master_start: float,
    source_timewarps: Sequence[TimeWarpMap | None] | None,
    scene_cut_times: Sequence[Sequence[float]],
    descriptor_provider: DescriptorProvider | None = None,
) -> tuple[
    tuple[RenderSegment, ...],
    tuple[RenderTransition, ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
    DirectMemberVisualSelection,
]:
    """Build a visually selected, all-source, uncropped portrait render."""

    eligible = tuple(int(index) for index in eligible_indices)
    if len(eligible) < MINIMUM_AUTOMATIC_MEMBER_SOURCE_COUNT:
        raise RuntimeError(
            "direct full-frame member render needs at least "
            f"{MINIMUM_AUTOMATIC_MEMBER_SOURCE_COUNT} sources"
        )
    if len(eligible_sources) != len(eligible):
        raise ValueError("direct eligible_sources must match eligible indices")
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
    partition_profile, visual_selection, partition_search = (
        _select_direct_member_partition_profile(
            timing_draft=timing_draft,
            eligible_indices=eligible,
            eligible_sources=eligible_sources,
            source_timewarps=source_timewarps,
            scene_cut_times=scene_cut_times,
            source_cycle=source_cycle,
            descriptor_provider=descriptor_provider,
        )
    )
    spans = partition_profile.spans

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
                "crop_applied": False,
                "timing_segment_count": span.timing_segment_count,
                "estimated_beat_count": span.estimated_beat_count,
                "start_boundary_reasons": list(span.start_boundary_reasons),
                "end_boundary_reasons": list(span.end_boundary_reasons),
            }
        )

    transitions = tuple(
        RenderTransition(reason=OFFICIAL_TARGET_FANCAM_DIRECT_FULL_FRAME_POLICY)
        for _ in range(len(resolved) - 1)
    )
    counts = {source: 0 for source in eligible}
    for segment in resolved:
        counts[segment.source_index] += 1
    used = {source for source, count in counts.items() if count > 0}
    if (
        len(used) < MINIMUM_AUTOMATIC_MEMBER_SOURCE_COUNT
        or used != set(eligible)
    ):
        raise RuntimeError("direct full-frame render failed all-source coverage")
    if max(counts.values()) - min(counts.values()) > 1:
        raise RuntimeError("direct full-frame render failed balanced visual selection")

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
    duration_targets = tuple(
        total_duration * visual_selection.source_counts[source_index] / len(resolved)
        for source_index in range(len(eligible))
    )
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
    if max(duration_deviations) > maximum_duration_deviation + 1e-9:
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
    if any(
        left.source_index == right.source_index
        for left, right in zip(resolved, resolved[1:])
    ):
        raise RuntimeError("direct full-frame render repeated an adjacent source")

    cadence = _render_cadence_payload(resolved)
    switches_per_minute = float(cadence["source_switches_per_minute"])
    median_run = float(cadence["source_run_median_seconds"])
    longest_run = float(cadence["longest_source_run_seconds"])
    if not (
        _DIRECT_MEMBER_MIN_SWITCHES_PER_MINUTE
        <= switches_per_minute
        <= _DIRECT_MEMBER_MAX_SWITCHES_PER_MINUTE
        and _DIRECT_MEMBER_MIN_MEDIAN_RUN_SECONDS
        <= median_run
        <= _DIRECT_MEMBER_MAX_MEDIAN_RUN_SECONDS
        and longest_run <= _DIRECT_MEMBER_MAX_RUN_SECONDS
    ):
        raise RuntimeError(
            "direct full-frame cadence missed calibrated safety gates: "
            f"{switches_per_minute:.2f}/min, median {median_run:.2f}s, "
            f"max {longest_run:.2f}s"
        )
    cadence["passed"] = True
    cadence["source_duration_balance"] = {
        "passed": True,
        "target_policy": (
            "total_duration_times_source_quota_divided_by_span_count"
        ),
        "source_durations_seconds": duration_totals,
        "source_target_durations_seconds": list(duration_targets),
        "source_relative_deviations": list(duration_deviations),
        "maximum_relative_deviation": max(duration_deviations),
        "maximum_allowed_relative_deviation": maximum_duration_deviation,
        "enforced_from_render_segments": True,
    }
    cadence["policy"] = (
        "reference_calibrated_proof_directed_partition_fallback_v2"
    )
    cadence["partition_search"] = partition_search
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
        }


def _apply_stage_edit_candidate_render_contract(
    bundle: _GroupRenderCandidateBundle,
    materialized: StageEditMaterializedVisualCandidate,
) -> _GroupRenderCandidateBundle:
    """Preserve the learned camera track after ordinary visual derivation."""

    plan = apply_stage_edit_dynamic_crop_to_render_plan(
        materialized,
        bundle.render_plan,
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
    lineage = stripped_materialized_lineage + (
        {
            "adapter": STAGE_EDIT_VISUAL_CANDIDATE_CONTRACT,
            "stage": "dynamic_crop_v2_render_plan_overlay",
            "ordinary_editorial_motion_discarded": True,
            "source_path_preserved": True,
            "timing_reparameterized_segment_indices": list(reparameterized),
            "verified_face_dissolve_alignment_preserved": True,
            "dynamic_crop_v2_suppressed_segment_indices": list(
                face_dissolve_alignment_segments
            ),
            "deterministic_render_plan_validation_passed": True,
            "superseded_candidate_history_exact_shot_clock_sha256": list(
                superseded_clock_hashes
            ),
        },
    )
    return replace(
        bundle,
        editorial_motion_plan=None,
        render_plan=plan,
        diagnostics=bundle.diagnostics
        + ("stage_edit_dynamic_crop_v2_render_contract_applied",),
        candidate_origin="explicit_stage_edit_generation",
        candidate_lineage=lineage,
        stage_edit_exact_shot_clock=_stage_edit_exact_shot_clock_from_edit(
            materialized.retained_candidate.edit
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
        "editorial_motion": (
            bundle.editorial_motion_plan.to_dict()
            if bundle.editorial_motion_plan is not None
            else None
        ),
        # Dynamic Crop v2 is attached directly to RenderSegment after the
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
    review["reference_profile"] = deepcopy(dict(reference_profile))
    return review, analysis


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
    suppressed_dynamic_crop_indices = tuple(
        sorted(
            {
                value
                for item in proposal_lineage
                for raw_values in (
                    item.get(
                        "dynamic_crop_v2_suppressed_segment_indices",
                        (),
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
        "dynamic_crop_v2_suppressed_segment_indices": list(
            suppressed_dynamic_crop_indices
        ),
        "proposal_materialization_lineage_preserved": True,
        "final_authoritative_exact_shot_clock_count": 1,
        "superseded_exact_shot_clock_sha256": list(
            superseded_exact_clock_hashes
        ),
    }
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
    return preflight_stage_edit_pipeline_bridge(
        feature_snapshot_path=snapshot,
        checkpoint_directory=checkpoint,
        source_ids=source_ids,
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


def _stage_edit_semimarkov_config_from_montage_policy(
    policy: ProfessionalMontagePolicy,
) -> StageEditSemiMarkovConfig:
    product = policy.product_editorial_requirements
    return StageEditSemiMarkovConfig(
        highlight_minimum_closeup_fraction=(
            policy.minimum_close_duration_fraction.value
        ),
        # No non-wide duration reference floor exists.  This is an explicit
        # IVE product editorial target, while close/burst/wide-transition
        # floors below retain their profile provenance.
        highlight_minimum_nonwide_fraction=0.35,
        highlight_minimum_wide_involved_source_switch_fraction=(
            policy.minimum_wide_involved_source_switch_fraction.value
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
    )


def _rendered_montage_config_from_policy(
    policy: ProfessionalMontagePolicy,
) -> RenderedMontageGrammarConfig:
    product = policy.product_editorial_requirements
    return RenderedMontageGrammarConfig(
        minimum_closeup_fraction=policy.minimum_close_duration_fraction.value,
        minimum_nonwide_fraction=0.35,
        minimum_wide_involved_source_switch_fraction=(
            policy.minimum_wide_involved_source_switch_fraction.value
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
    timing_draft: GroupDraftTimeline,
    rhythm: RhythmAnalysis,
    section_plan: SectionPlan,
) -> tuple[StageEditInferenceSnapshotResult, StageEditPipelineBridgePreflight]:
    """Bind one verified installed model to this exact synchronized edit."""

    inference_artifact_root = (
        request.source_cache_directory or request.output_directory
    ) / "derived" / "stage-edit-inference-v1"
    materialized = materialize_stage_edit_inference_snapshot(
        song_id=request.song,
        sources=request.sources,
        alignment=alignment,
        alignment_hard_gates=tuple(
            bool(item["render_eligible"]) for item in alignment_eligibility
        ),
        source_timewarps=source_timewarps,
        timing_draft=timing_draft,
        rhythm=rhythm,
        section_plan=section_plan,
        output_directory=inference_artifact_root,
        feature_cache_directory=request.source_cache_directory,
        maximum_sequence_length=bundle.maximum_sequence_length,
        maximum_sources=bundle.maximum_sources,
    )
    preflight = preflight_stage_edit_pipeline_bridge(
        feature_snapshot_path=materialized.snapshot_path,
        checkpoint_directory=bundle.checkpoint_directory,
        source_ids=materialized.source_ids,
    )
    if preflight.readiness.checkpoint_path != str(bundle.checkpoint_directory):
        raise RuntimeError(
            "registered StageEdit checkpoint identity changed during materialization"
        )
    return materialized, preflight


def analyze(
    request: AnalysisRequest,
    reporter: EventReporter,
    *,
    allow_failed_comparison_render: bool = False,
    review_rendered_proxies: bool = False,
) -> AnalysisArtifact:
    if not isinstance(allow_failed_comparison_render, bool):
        raise TypeError("allow_failed_comparison_render must be a boolean")
    if not isinstance(review_rendered_proxies, bool):
        raise TypeError("review_rendered_proxies must be a boolean")
    registered_stage_edit_bundle = (
        _discover_automatic_stage_edit_production_bundle(request)
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
    stage_edit_semimarkov_config = (
        _stage_edit_semimarkov_config_from_montage_policy(
            stage_edit_montage_policy
        )
        if stage_edit_montage_policy is not None
        else None
    )
    rendered_montage_config = (
        _rendered_montage_config_from_policy(stage_edit_montage_policy)
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
        reporter.log(
            "StageEdit Dynamic Crop v2 후보 생성을 명시적으로 활성화 · "
            f"snapshot {stage_edit_generation_preflight.snapshot.snapshot_sha256[:12]} "
            f"· source {len(stage_edit_generation_preflight.source_ids)}개 · "
            "후보 8개 전수 시각·프록시 QA 통과 전에는 렌더 승격 없음"
        )
    if stage_edit_montage_policy is not None:
        reporter.log(
            "IVE 사람 편집본 몽타주 문법 프로필 엄격 검증 완료 · "
            f"4곡/20편 · file SHA-256 "
            f"{stage_edit_montage_policy.identity.file_sha256[:12]} · "
            "대형 앵커→근접 연쇄→대형 복귀 게이트 활성화"
        )
    if registered_stage_edit_bundle is not None:
        reporter.log(
            "등록된 production StageEdit Dynamic Crop v2 자동 활성화 · "
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
    # point.  Enforce its uncropped 9:16 source promise before alignment or
    # scene work so a bad cache can never fall through into anchor/tracking.
    _automatic_member_portrait_source_ratios(request, media)

    alignment, reference_pcm = _align_sequentially(request, reporter)
    alignment_eligibility = _alignment_eligibility(
        alignment,
        edit_mode=request.edit_mode,
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
        changes = detect_scene_changes_in_video(
            source,
            config=scene_config,
            sample_every=2,
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
        reporter.log(f"{source.name}: 방송 카메라 컷 {len(changes)}개 감지")
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
            ) = _materialize_registered_stage_edit_preflight(
                bundle=registered_stage_edit_bundle,
                request=request,
                alignment=alignment,
                alignment_eligibility=alignment_eligibility,
                source_timewarps=source_timewarps,
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
                "Dynamic Crop v2 Transformer 헤드를 semi-Markov 후보로 "
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
                    config=rendered_montage_config,
                    reference_profile=stage_edit_montage_policy.profile_identity,
                )
                stage_edit_exact_shot_clock_payload = deepcopy(
                    _final_montage_analysis["stage_edit_exact_shot_clock"]
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
        if any(abs(value - source_aspect_ratio) > 0.03 for value in aspect_ratios[1:]):
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
                    common_master_start=member_common_start,
                    source_timewarps=source_timewarps,
                    scene_cut_times=scene_cut_times,
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
            dense_face_provider: DirectMemberDenseFaceProvider | None = None
            direct_motion_error: str | None = None
            untouched_direct_segments = segments
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

                candidate_member_motion_plan = (
                    build_direct_member_editorial_motion_plan(
                        visual_selection=direct_visual_selection,
                        render_segments=segments,
                        transitions=member_transitions,
                        common_master_start=member_common_start,
                        geometry_provider=direct_member_geometry_provider,
                        fps=29.97,
                    )
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
                    f"팬 {member_editorial_motion_plan.pan_event_count}회"
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
                        "backend": "yunet_dominant_face_dense_selected_ramps",
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
            )
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
                member_direct_full_frame_payload = {
                    "enabled": True,
                    "policy": OFFICIAL_TARGET_FANCAM_DIRECT_FULL_FRAME_POLICY,
                    "provenance": (
                        "automatic_acquisition_official_target_member_fancam"
                    ),
                    "identity_policy": (
                        "no_geometry_identity_claim_no_tracking_no_crop"
                    ),
                    "output_aspect": request.aspect_ratio.value,
                    "output_pixel_aspect_ratio": 9.0 / 16.0,
                    "source_aspect_ratios": list(direct_full_frame_ratios),
                    "eligible_request_source_indices": list(eligible_indices),
                    "minimum_required_distinct_source_count": (
                        MINIMUM_AUTOMATIC_MEMBER_SOURCE_COUNT
                    ),
                    "cadence": direct_cadence,
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
            }
            if stage_edit_generation_result is not None
            and stage_edit_generation_audit_path is not None
            else None
        ),
        "member_tracking": member_tracking_payloads,
        "member_auto_anchors": member_auto_anchor_payloads,
        "member_anchor_trim": member_anchor_trim_payload,
        "member_direct_full_frame": member_direct_full_frame_payload,
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
    if stage_edit_candidate_rerank_result is not None:
        manifest["stage_edit_candidate_rerank"] = (
            stage_edit_candidate_rerank_result.to_dict()
        )
    filename = f"{base_filename}-analysis.json"
    manifest_path = request.output_directory / filename
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    mandatory_post_render_qa_path = (
        _enforce_automatic_stage_edit_post_render_qa(
            registered_stage_edit_bundle=registered_stage_edit_bundle,
            manifest=manifest,
            manifest_path=manifest_path,
            reporter=reporter,
        )
    )
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
    if stage_edit_generation_audit_path is not None:
        output_paths.append(stage_edit_generation_audit_path)
    if source_derivative_audit_path is not None:
        output_paths.append(source_derivative_audit_path)
    if mandatory_post_render_qa_path is not None:
        output_paths.append(mandatory_post_render_qa_path)
    if rendered_video_path is not None:
        output_paths.append(rendered_video_path)
    output_paths.extend(timeline_interchange_paths)
    return AnalysisArtifact(
        manifest_path=manifest_path,
        manifest=manifest,
        output_paths=tuple(output_paths),
    )
